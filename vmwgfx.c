/*
 * Copyright 2018 The FydeOS Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifdef DRV_VMWGFX

#include "drv_helpers.h"
#include "drv_priv.h"
#include "external/svga3d_types.h"
#include "util.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <vmwgfx_drm.h>
#include <xf86drm.h>

static const uint32_t render_target_formats[] = { DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888,
						  DRM_FORMAT_XRGB1555, DRM_FORMAT_RGB565,
						  DRM_FORMAT_ABGR8888, DRM_FORMAT_XBGR8888 };

static const uint32_t texture_source_formats[] = { DRM_FORMAT_R8, DRM_FORMAT_NV12, DRM_FORMAT_YUYV,
						   DRM_FORMAT_YVU420, DRM_FORMAT_YVU420_ANDROID };

static int vmwgfx_init(struct driver *drv)
{
	drv_add_combinations(drv, render_target_formats, ARRAY_SIZE(render_target_formats),
			     &LINEAR_METADATA, BO_USE_RENDER_MASK | BO_USE_SCANOUT);

	drv_add_combinations(drv, texture_source_formats, ARRAY_SIZE(texture_source_formats),
			     &LINEAR_METADATA, BO_USE_TEXTURE_MASK);
	drv_modify_combination(drv, DRM_FORMAT_NV12, &LINEAR_METADATA,
			       BO_USE_SCANOUT | BO_USE_SW_MASK | BO_USE_LINEAR |
				   BO_USE_CAMERA_READ | BO_USE_CAMERA_WRITE |
				   BO_USE_HW_VIDEO_DECODER | BO_USE_HW_VIDEO_ENCODER);
	drv_modify_combination(drv, DRM_FORMAT_YUYV, &LINEAR_METADATA,
			       BO_USE_HW_VIDEO_DECODER | BO_USE_SCANOUT | BO_USE_LINEAR |
				   BO_USE_TEXTURE | BO_USE_CAMERA_READ | BO_USE_SW_MASK);
	drv_modify_combination(drv, DRM_FORMAT_R8, &LINEAR_METADATA,
			       BO_USE_SW_MASK | BO_USE_LINEAR | BO_USE_HW_VIDEO_DECODER |
				   BO_USE_HW_VIDEO_ENCODER | BO_USE_CAMERA_READ |
				   BO_USE_CAMERA_WRITE);
	drv_logi("vmwgfx minigbm inited.");
	return drv_modify_linear_combinations(drv);
}

static bool is_texture(uint32_t format)
{
	for (size_t i = 0; i < sizeof(texture_source_formats) / sizeof(texture_source_formats[0]);
	     i++) {
		if (format == texture_source_formats[i])
			return true;
	}
	return false;
}

static void store_map_handle(struct bo_metadata *meta, uint64_t map_handle)
{
	meta->blob_id = (uint32_t)map_handle;
	meta->map_info = (uint32_t)(map_handle >> 32);
	meta->memory_idx = 0xFF;
}

static bool exist_map_handle(struct bo_metadata *meta)
{
	return meta->memory_idx == 0xFF;
}

static uint64_t get_map_handle(struct bo_metadata *meta)
{
	uint64_t map_handle = meta->map_info;
	return (map_handle << 32) & meta->blob_id;
}

static const uint32_t drm_format_to_svga_format(uint32_t format)
{
	switch (format) {
	case DRM_FORMAT_R8:
		return SVGA3D_P8;
	case DRM_FORMAT_ARGB8888:
		return SVGA3D_B8G8R8A8_UNORM;
	case DRM_FORMAT_XRGB8888:
	case DRM_FORMAT_XBGR8888:
		return SVGA3D_B8G8R8X8_UNORM;
	case DRM_FORMAT_XRGB1555:
		return SVGA3D_X1R5G5B5;
	case DRM_FORMAT_RGB565:
		return SVGA3D_R5G6B5;
	case DRM_FORMAT_YVU420_ANDROID:
	case DRM_FORMAT_YVU420:
		return SVGA3D_UYVY;
	case DRM_FORMAT_YUYV:
		return SVGA3D_YUY2;
	case DRM_FORMAT_NV12:
		return SVGA3D_NV12;
	case DRM_FORMAT_ABGR8888:
		return SVGA3D_R8G8B8A8_UNORM;
	default:
		drv_logi("UNKNOWN FORMAT %d\n", format);
		return format;
	}
}

static int vmwgfx_bo_create_with_modifiers(struct bo *bo, uint32_t width, uint32_t height,
					   uint32_t format, const uint64_t *modifiers,
					   uint32_t count)
{
	int ret;
	uint32_t stride;
	uint32_t aligned_width, aligned_height;
	union drm_vmw_gb_surface_create_arg create_arg;
	struct drm_vmw_gb_surface_create_rep *rep = &create_arg.rep;
	struct drm_vmw_gb_surface_create_req *req = &create_arg.req;
	uint32_t extra_flag = 0;

	if (!drv_has_modifier(modifiers, count, DRM_FORMAT_MOD_LINEAR)) {
		drv_loge("no usable modifier found\n");
		return -EINVAL;
	}
	aligned_width = width;
	aligned_height = height;
	switch (format) {
	case DRM_FORMAT_YVU420_ANDROID:
		/* Align width to 32 pixels, so chroma strides are 16 bytes as
		 * Android requires. */
		aligned_width = ALIGN(width, 32);
		/* Adjust the height to include room for chroma planes.
		 *
		 * HAL_PIXEL_FORMAT_YV12 requires that the buffer's height not
		 * be aligned. */
		aligned_height = 3 * DIV_ROUND_UP(height, 2);
		extra_flag =
		    SVGA3D_SURFACE_BIND_STREAM_OUTPUT | SVGA3D_SURFACE_TRANSFER_FROM_BUFFER;
		break;
	case DRM_FORMAT_YVU420:
	case DRM_FORMAT_YUYV:
	case DRM_FORMAT_NV12:
		/* Adjust the height to include room for chroma planes */
		aligned_height = 3 * DIV_ROUND_UP(height, 2);
		extra_flag =
		    SVGA3D_SURFACE_BIND_STREAM_OUTPUT | SVGA3D_SURFACE_TRANSFER_FROM_BUFFER;
		break;
	default:
		break;
	}

	bo->meta.total_size = 0;
	memset(&create_arg, 0, sizeof(create_arg));
	req->svga3d_flags = SVGA3D_SURFACE_SCREENTARGET | extra_flag |
			    SVGA3D_SURFACE_HINT_RENDERTARGET | SVGA3D_SURFACE_HINT_DYNAMIC;
	req->format = drm_format_to_svga_format(format);
	req->mip_levels = 1;
	req->drm_surface_flags =
	    drm_vmw_surface_flag_shareable | drm_vmw_surface_flag_create_buffer;
	if (!is_texture(format))
		req->drm_surface_flags |= drm_vmw_surface_flag_scanout;
	req->array_size = 0;
	req->buffer_handle = SVGA3D_INVALID_ID;
	req->base_size.width = aligned_width;
	req->base_size.height = aligned_height;
	req->base_size.depth = 1;
	ret = drmCommandWriteRead(bo->drv->fd, DRM_VMW_GB_SURFACE_CREATE, &create_arg,
				  sizeof(create_arg));

	if (ret) {
		drv_loge("DRM_VMW_GB_SURFACE_CREATE failed (%d, %d), req format: 0x%x, drm format: "
			 "0x%x\n",
			 bo->drv->fd, ret, req->format, format);
		return ret;
	}
	stride = drv_stride_from_format(format, aligned_width, 0);
	drv_bo_from_format(bo, stride, 1, aligned_height, format);
	bo->handle.u32 = rep->handle;
	store_map_handle(&bo->meta, rep->buffer_map_handle);
	return 0;
}

static int vmwgfx_bo_create(struct bo *bo, uint32_t width, uint32_t height, uint32_t format,
			    uint64_t use_flags)
{
	static const uint64_t modifiers[] = {
		DRM_FORMAT_MOD_LINEAR,
	};
	return vmwgfx_bo_create_with_modifiers(bo, width, height, format, modifiers,
					       ARRAY_SIZE(modifiers));
}

static void *vmwgfx_bo_map(struct bo *bo, struct vma *vma, uint32_t map_flags)
{
	union drm_vmw_gb_surface_reference_arg s_arg;
	struct drm_vmw_surface_arg *req = &s_arg.req;
	struct drm_vmw_gb_surface_ref_rep *rep = &s_arg.rep;
	int ret;
	void *addr;
	vma->length = bo->meta.total_size;
	if (exist_map_handle(&bo->meta)) {
		addr = mmap(0, vma->length, drv_get_prot(map_flags), MAP_SHARED, bo->drv->fd,
			    get_map_handle(&bo->meta));
	} else {
		req->sid = bo->handle.u32;
		req->handle_type = DRM_VMW_HANDLE_LEGACY;
		ret =
		    drmCommandWriteRead(bo->drv->fd, DRM_VMW_GB_SURFACE_REF, &s_arg, sizeof(s_arg));
		if (ret) {
			drv_loge("Error get gb surface reference, ret:%d\n", ret);
			return MAP_FAILED;
		}
		addr = mmap(0, vma->length, drv_get_prot(map_flags), MAP_SHARED, bo->drv->fd,
			    rep->crep.buffer_map_handle);
	}
	drv_logd("map success: handle:%u, map_flags:0x%x\n", bo->handle.u32, map_flags);
	return addr;
}

int vmwgfx_bo_destroy(struct bo *bo)
{
	struct drm_vmw_unref_dmabuf_arg unref_arg = {};
	int ret;
	unref_arg.handle = bo->handle.u32;
	ret = drmCommandWrite(bo->drv->fd, DRM_VMW_UNREF_DMABUF, &unref_arg, sizeof(unref_arg));
	if (ret) {
		drv_loge("DRM_IOCTL_VMW_UNREF_DMABUF failed(handle=%x),error %d\n", bo->handle.u32,
			 ret);
	}
	return ret;
}

const struct backend backend_vmwgfx = {
	.name = "vmwgfx",
	.init = vmwgfx_init,
	.bo_create = vmwgfx_bo_create,
	.bo_create_with_modifiers = vmwgfx_bo_create_with_modifiers,
	.bo_import = drv_prime_bo_import,
	.bo_destroy = vmwgfx_bo_destroy,
	.bo_map = vmwgfx_bo_map,
	.bo_unmap = drv_bo_munmap,
	.resolve_format_and_use_flags = drv_resolve_format_and_use_flags_helper,
};

#endif
