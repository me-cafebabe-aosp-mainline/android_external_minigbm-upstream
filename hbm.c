/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifdef DRV_HBM_HELPER

/* There are a few noticeable differences between minigbm and hbm
 *
 *  - minigbm uses DRM_FORMAT_R8 for buffers while hbm uses DRM_FORMAT_INVALID
 *  - minigbm supports dma-buf import with an implicit modifier while hbm
 *    requires an explicit modifier
 *    - unless the underlying vulkan driver lacks explicit modifier support
 *  - minigbm memory mapping respects implicit fencing while hbm memory
 *    mapping does not
 *  - minigbm memory mapping always returns a linear view while hbm memory
 *    mapping maps the bo directly
 *
 * This glue layer tries to hide some of the differences
 *
 *  - DRM_FORMAT_R8 is translated DRM_FORMAT_INVALID, when the use flags
 *    contain BO_USE_GPU_DATA_BUFFER or BO_USE_SENSOR_DIRECT_DATA
 *  - implicit modifier is passed through to hbm, which can be rejected
 *    however
 *  - implicit fencing is simulated via dma-buf polling
 *    - we could potentially use DMA_BUF_IOCTL_{EXPORT,IMPORT}_SYNC_FILE to
 *      convert between implicit and explicit fencing
 *  - a staging bo is used when the bo is tiled
 */

#include "hbm.h"
#include "hbm_minigbm.h"

#include <assert.h>
#include <drm_fourcc.h>
#include <errno.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xf86drm.h>

#include "drv_helpers.h"

struct hbm {
	struct hbm_device *device;
	/* not owned */
	int driver_fd;

	uint32_t staging_mt;
};

struct hbm_resource {
	struct hbm_bo *bo;
	uint32_t format;

	bool use_sw;
	/* owned */
	int implicit_fence_dmabuf;

	uint32_t staging_size;
	uint32_t staging_offsets[DRV_MAX_PLANES];
	uint32_t staging_strides[DRV_MAX_PLANES];
};

static uint32_t pick_staging_memory_type(struct hbm *hbm);

static void hbm_log(int32_t lv, const char *msg, void *data)
{
	enum drv_log_level drv_lv;
	switch (lv) {
	case HBM_LOG_LEVEL_ERROR:
	case HBM_LOG_LEVEL_WARN:
		drv_lv = DRV_LOGE;
		break;
	case HBM_LOG_LEVEL_INFO:
		drv_lv = DRV_LOGI;
		break;
	case HBM_LOG_LEVEL_DEBUG:
	default:
		drv_lv = DRV_LOGD;
		break;
	}

	_drv_log(drv_lv, "hbm: %s\n", msg);
}

static bool get_device_id(int drv_fd, dev_t *out_dev_id)
{
	const int node_type = DRM_NODE_RENDER;

	struct stat st;
	if (fstat(drv_fd, &st))
		return false;

	dev_t dev_id = st.st_rdev;
	if (drmGetNodeTypeFromDevId(dev_id) != node_type) {
		drmDevicePtr dev;
		if (drmGetDeviceFromDevId(dev_id, 0, &dev))
			return false;

		const bool ok =
		    (dev->available_nodes & (1 << node_type)) && !stat(dev->nodes[node_type], &st);
		drmFreeDevice(&dev);
		if (!ok)
			return false;

		dev_id = st.st_rdev;
	}

	*out_dev_id = dev_id;

	return true;
}

struct hbm *hbm_create(int drv_fd)
{
	const bool debug = false;

	dev_t dev_id;
	if (!get_device_id(drv_fd, &dev_id))
		return NULL;

	hbm_log_init(debug ? HBM_LOG_LEVEL_DEBUG : HBM_LOG_LEVEL_WARN, hbm_log, NULL);

	struct hbm_device *dev = hbm_device_create(dev_id, debug);
	if (!dev)
		return NULL;

	struct hbm *hbm = calloc(1, sizeof(*hbm));
	if (!hbm) {
		hbm_device_destroy(dev);
		return NULL;
	}

	hbm->device = dev;
	/* no ownership transfer */
	hbm->driver_fd = drv_fd;

	hbm->staging_mt = pick_staging_memory_type(hbm);
	if (!hbm->staging_mt) {
		hbm_destroy(hbm);
		return NULL;
	}

	return hbm;
}

void hbm_destroy(struct hbm *hbm)
{
	hbm_device_destroy(hbm->device);
	free(hbm);
}

static bool use_overlay(uint64_t use_flags)
{
	/* Other than HBM_USAGE_GPU_SCANOUT_HACK, which is ignored unless hbm
	 * lacks explicit modifier support, hbm assumes no knowledge about the
	 * display.  Instead, this glue layer assumes
	 *
	 *  - SCANOUT and CURSOR must use LOCAL and non-CACHED mt
	 *    - as a result, COPY is preferred for readback
	 *  - CURSOR must be LINEAR
	 */
	return use_flags & (BO_USE_SCANOUT | BO_USE_CURSOR);
}

static bool use_gpu(uint64_t use_flags)
{
	return use_flags & (BO_USE_RENDERING | BO_USE_TEXTURE | BO_USE_GPU_DATA_BUFFER);
}

static bool use_blob(uint64_t use_flags)
{
	return use_flags & (BO_USE_GPU_DATA_BUFFER | BO_USE_SENSOR_DIRECT_DATA);
}

static bool use_sw_read(uint64_t use_flags)
{
	return use_flags & (BO_USE_SW_READ_OFTEN | BO_USE_SW_READ_RARELY);
}

static bool use_sw_write(uint64_t use_flags)
{
	return use_flags & (BO_USE_SW_WRITE_OFTEN | BO_USE_SW_WRITE_RARELY);
}

static bool use_sw(uint64_t use_flags)
{
	return use_sw_read(use_flags) || use_sw_write(use_flags);
}

static bool use_sw_often(uint64_t use_flags)
{
	return use_flags & (BO_USE_SW_READ_OFTEN | BO_USE_SW_WRITE_OFTEN);
}

static bool prefer_map(uint64_t use_flags)
{
	assert(use_sw(use_flags));

	/* choose between MAP or COPY */
	bool prefer_map;
	if (use_overlay(use_flags))
		prefer_map = use_sw_often(use_flags) && !use_sw_read(use_flags);
	else if (use_gpu(use_flags))
		prefer_map = use_sw_often(use_flags);
	else
		prefer_map = true;

	return prefer_map;
}

static void init_description(struct hbm *hbm, uint32_t fmt, uint64_t mod, uint64_t use_flags,
			     struct hbm_description *out_desc)
{
	uint32_t flags = HBM_FLAG_EXTERNAL;
	if (use_flags & BO_USE_PROTECTED)
		flags |= HBM_FLAG_PROTECTED;
	if (use_flags & BO_USE_FRONT_RENDERING)
		flags |= HBM_FLAG_NO_COMPRESSION;

	uint32_t usage = 0;
	if (use_flags & BO_USE_RENDERING)
		usage |= HBM_USAGE_GPU_COLOR;
	if (use_flags & BO_USE_TEXTURE)
		usage |= HBM_USAGE_GPU_SAMPLED;
	if (use_flags & BO_USE_GPU_DATA_BUFFER)
		usage |= HBM_USAGE_GPU_UNIFORM | HBM_USAGE_GPU_STORAGE;
	assert(use_gpu(use_flags) == !!usage);

	/* translate R8 to INVALID */
	if (use_blob(use_flags)) {
		assert(fmt == DRM_FORMAT_R8);
		fmt = DRM_FORMAT_INVALID;
	}

	if (mod == DRM_FORMAT_MOD_INVALID) {
		if (use_flags & (BO_USE_LINEAR | BO_USE_CURSOR))
			mod = DRM_FORMAT_MOD_LINEAR;
		if (use_overlay(use_flags))
			usage |= HBM_USAGE_GPU_SCANOUT_HACK;
	}

	if (use_sw(use_flags)) {
		/* for simplicity */
		flags |= HBM_FLAG_MAP | HBM_FLAG_COPY;

		if (mod == DRM_FORMAT_MOD_INVALID && prefer_map(use_flags)) {
			const struct hbm_description test_desc = {
				.flags = flags,
				.format = fmt,
				.modifier = mod,
				.usage = usage,
			};
			if (hbm_device_has_modifier(hbm->device, &test_desc, DRM_FORMAT_MOD_LINEAR))
				mod = DRM_FORMAT_MOD_LINEAR;
		}
	}

	*out_desc = (struct hbm_description){
		.flags = flags,
		.format = fmt,
		.modifier = mod,
		.usage = usage,
	};
}

uint64_t *hbm_get_format_modifiers(struct hbm *hbm, uint32_t fmt, uint64_t use_flags,
				   uint32_t *out_count)
{
	struct hbm_description desc;
	init_description(hbm, fmt, DRM_FORMAT_MOD_INVALID, use_flags, &desc);

	uint32_t count = hbm_device_get_modifiers(hbm->device, &desc, 0, NULL);
	if (!count)
		return NULL;

	uint64_t *mods = malloc(sizeof(*mods) * count);
	if (!mods)
		return NULL;

	*out_count = hbm_device_get_modifiers(hbm->device, &desc, count, mods);

	return mods;
}

static void init_extent(uint32_t fmt, uint32_t width, uint32_t height, union hbm_extent *out_extent)
{
	if (fmt == DRM_FORMAT_INVALID) {
		assert(height == 1);
		out_extent->buffer = (struct hbm_extent_buffer){
			.size = width,
		};
	} else {
		out_extent->image = (struct hbm_extent_image){
			.width = width,
			.height = height,
		};
	}
}

static uint32_t *get_memory_types(struct hbm *hbm, struct hbm_bo *bo, uint32_t *out_count)
{
	uint32_t count = hbm_bo_memory_types(bo, 0, NULL);
	uint32_t *mts = malloc(sizeof(*mts) * count);
	if (!mts)
		return NULL;
	count = hbm_bo_memory_types(bo, count, mts);

	*out_count = count;
	return mts;
}

static bool pick_memory_type(struct hbm *hbm, struct hbm_bo *bo, uint64_t modifier,
			     uint64_t use_flags, uint32_t *out_mt, bool *out_use_staging)
{
	uint32_t required_flags = 0;
	uint32_t disallowed_flags = 0;
	uint32_t preferred_flag = 0;
	bool use_staging = false;

	if (use_overlay(use_flags)) {
		required_flags |= HBM_MEMORY_TYPE_LOCAL;
		disallowed_flags |= HBM_MEMORY_TYPE_CACHED;
	}

	if (use_sw(use_flags)) {
		/* this is an oversimplification */
		if (modifier == DRM_FORMAT_MOD_LINEAR && prefer_map(use_flags)) {
			required_flags |= HBM_MEMORY_TYPE_MAPPABLE;
			preferred_flag = HBM_MEMORY_TYPE_CACHED;
		} else {
			preferred_flag = HBM_MEMORY_TYPE_LOCAL;
			use_staging = true;
		}
	} else {
		preferred_flag = HBM_MEMORY_TYPE_LOCAL;
	}

	if (disallowed_flags & preferred_flag)
		preferred_flag = 0;

	uint32_t count;
	uint32_t *mts = get_memory_types(hbm, bo, &count);
	if (!mts)
		return false;

	int32_t best_mt = -1;
	for (uint32_t i = 0; i < count; i++) {
		if ((mts[i] & required_flags) != required_flags || (mts[i] & disallowed_flags))
			continue;

		if (mts[i] & preferred_flag) {
			best_mt = mts[i];
			break;
		} else if (!best_mt) {
			best_mt = mts[i];
			if (!preferred_flag)
				break;
		}
	}

	free(mts);

	if (best_mt == -1)
		return false;

	*out_mt = best_mt;
	*out_use_staging = use_staging;

	return true;
}

static struct hbm_resource *create_resource(struct hbm *hbm, struct hbm_bo *bo,
					    const struct hbm_description *desc,
					    const union hbm_extent *extent,
					    const struct hbm_layout *layout, uint64_t use_flags,
					    int dmabuf)
{
	uint32_t mt;
	bool use_staging;
	if (!pick_memory_type(hbm, bo, layout->modifier, use_flags, &mt, &use_staging))
		return NULL;

	if (dmabuf >= 0) {
		dmabuf = dup(dmabuf);
		if (dmabuf < 0)
			return NULL;
	}

	/* dmabuf ownership is always transferred */
	if (!hbm_bo_bind_memory(bo, mt, dmabuf))
		return NULL;

	struct hbm_resource *res = calloc(1, sizeof(*res));
	if (!res)
		return NULL;

	res->bo = bo;
	res->format = desc->format;

	res->use_sw = use_sw(use_flags);
	res->implicit_fence_dmabuf = -1;

	if (use_staging) {
		if (desc->format == DRM_FORMAT_INVALID) {
			res->staging_size = extent->buffer.size;
		} else {
			const uint32_t width = extent->image.width;
			const uint32_t height = extent->image.height;
			const uint32_t plane_count = drv_num_planes_from_format(desc->format);

			uint32_t offset = 0;
			for (uint32_t plane = 0; plane < plane_count; plane++) {
				const uint32_t stride =
				    drv_stride_from_format(desc->format, width, plane);
				const uint32_t size =
				    drv_size_from_format(desc->format, stride, height, plane);

				res->staging_offsets[plane] = offset;
				res->staging_strides[plane] = stride;
				offset += size;
			}

			res->staging_size = offset;
		}
	}

	return res;
}

static void init_bo_metadata(const struct hbm_layout *layout, struct bo_metadata *out_meta)
{
	out_meta->total_size = layout->size;
	out_meta->format_modifier = layout->modifier;
	out_meta->num_planes = layout->plane_count;

	for (uint32_t i = 0; i < layout->plane_count; i++) {
		out_meta->offsets[i] = layout->offsets[i];
		out_meta->strides[i] = layout->strides[i];

		/* assume planes are ordered */
		const uint64_t next_offset =
		    (i + 1 < layout->plane_count) ? layout->offsets[i + 1] : layout->size;
		out_meta->sizes[i] = next_offset - layout->offsets[i];
	}
}

struct hbm_resource *hbm_allocate(struct hbm *hbm, uint32_t width, uint32_t height, uint32_t fmt,
				  uint64_t use_flags, const uint64_t *mods, uint32_t mod_count,
				  struct bo_metadata *out_meta)
{
	const uint64_t desc_mod = mod_count == 1 ? mods[0] : DRM_FORMAT_MOD_INVALID;
	struct hbm_description desc;
	init_description(hbm, fmt, desc_mod, use_flags, &desc);

	union hbm_extent extent;
	init_extent(desc.format, width, height, &extent);

	const struct hbm_constraint con = {
		.modifiers = mods,
		.modifier_count = mod_count,
	};

	struct hbm_bo *bo =
	    hbm_bo_create_with_constraint(hbm->device, &desc, &extent, mod_count > 1 ? &con : NULL);
	if (!bo)
		return NULL;

	struct hbm_layout layout;
	hbm_bo_layout(bo, &layout);

	struct hbm_resource *res = create_resource(hbm, bo, &desc, &extent, &layout, use_flags, -1);
	if (!res) {
		hbm_bo_destroy(bo);
		return NULL;
	}

	init_bo_metadata(&layout, out_meta);

	return res;
}

static bool init_layout(uint32_t fmt, const struct drv_import_fd_data *import_data,
			struct hbm_layout *out_layout)
{
	const off_t size = lseek(import_data->fds[0], 0, SEEK_END);
	if (size == (off_t)-1)
		return false;

	*out_layout = (struct hbm_layout){
		.size = size,
	};
	if (fmt == DRM_FORMAT_INVALID)
		return true;

	out_layout->modifier = import_data->format_modifier;
	for (int i = 0; i < DRV_MAX_PLANES; i++) {
		if (import_data->fds[i] < 0)
			break;

		out_layout->plane_count++;
		out_layout->offsets[i] = import_data->offsets[i];
		out_layout->strides[i] = import_data->strides[i];
	}

	return true;
}

struct hbm_resource *hbm_import(struct hbm *hbm, const struct drv_import_fd_data *import_data,
				struct bo_metadata *out_meta)
{
	struct hbm_description desc;
	init_description(hbm, import_data->format, import_data->format_modifier,
			 import_data->use_flags, &desc);

	union hbm_extent extent;
	init_extent(desc.format, import_data->width, import_data->height, &extent);

	struct hbm_layout layout;
	if (!init_layout(desc.format, import_data, &layout))
		return NULL;

	struct hbm_bo *bo =
	    hbm_bo_create_with_layout(hbm->device, &desc, &extent, &layout, import_data->fds[0]);
	if (!bo)
		return NULL;

	struct hbm_resource *res = create_resource(hbm, bo, &desc, &extent, &layout,
						   import_data->use_flags, import_data->fds[0]);
	if (!res) {
		hbm_bo_destroy(bo);
		return NULL;
	}

	init_bo_metadata(&layout, out_meta);

	return res;
}

void hbm_free(struct hbm *hbm, struct hbm_resource *res)
{
	hbm_bo_destroy(res->bo);

	if (res->implicit_fence_dmabuf >= 0)
		close(res->implicit_fence_dmabuf);

	free(res);
}

uint32_t hbm_reimport_to_driver(struct hbm *hbm, struct hbm_resource *res,
				const struct drv_import_fd_data *import_data)
{
	/* get dmabuf first */
	int dmabuf;
	bool owned;
	if (import_data) {
		dmabuf = import_data->fds[0];
		owned = false;

		if (res->use_sw) {
			dmabuf = dup(dmabuf);
			owned = true;
		}
	} else {
		dmabuf = hbm_bo_export_dma_buf(res->bo, "minigbm");
		owned = true;
	}
	if (dmabuf < 0)
		return 0;

	/* re-import to driver_fd; no ownership transfer */
	uint32_t gem_handle;
	if (drmPrimeFDToHandle(hbm->driver_fd, dmabuf, &gem_handle)) {
		if (owned)
			close(dmabuf);
		return 0;
	}

	if (res->use_sw) {
		/* dmabuf ownership is transferred */
		res->implicit_fence_dmabuf = dmabuf;
		assert(owned);
	} else if (owned) {
		close(dmabuf);
	}

	return gem_handle;
}

static struct hbm_bo *create_staging(struct hbm *hbm, uint64_t size)
{
	const struct hbm_description desc = {
		.flags = HBM_FLAG_MAP | HBM_FLAG_COPY,
		.format = DRM_FORMAT_INVALID,
		.modifier = DRM_FORMAT_MOD_INVALID,
	};
	const union hbm_extent extent = {
		.buffer = {
			.size = size,
		},
	};

	struct hbm_bo *bo = hbm_bo_create_with_constraint(hbm->device, &desc, &extent, NULL);
	if (!bo)
		return NULL;

	return bo;
}

static uint32_t pick_staging_memory_type(struct hbm *hbm)
{
	struct hbm_bo *bo = create_staging(hbm, 1);
	if (!bo)
		return 0;

	uint32_t count;
	uint32_t *mts = get_memory_types(hbm, bo, &count);
	hbm_bo_destroy(bo);
	if (!mts)
		return 0;

	/* should we prefer CACHED over COHERENT? */
	const uint32_t required_flags = HBM_MEMORY_TYPE_MAPPABLE | HBM_MEMORY_TYPE_COHERENT;
	const uint32_t preferred_flag = HBM_MEMORY_TYPE_CACHED;

	uint32_t best_mt = 0;
	for (uint32_t i = 0; i < count; i++) {
		if ((mts[i] & required_flags) != required_flags)
			continue;

		if (mts[i] & preferred_flag) {
			best_mt = mts[i];
			break;
		} else if (!best_mt) {
			best_mt = mts[i];
		}
	}

	free(mts);

	return best_mt;
}

void *hbm_map(struct hbm *hbm, struct hbm_resource *res, struct vma *vma, uint32_t map_flags)
{
	if (!res->staging_size)
		return hbm_bo_map(res->bo);

	struct hbm_bo *staging = create_staging(hbm, res->staging_size);
	if (!staging)
		return NULL;
	if (!hbm_bo_bind_memory(staging, hbm->staging_mt, -1)) {
		hbm_bo_destroy(staging);
		return NULL;
	}

	void *ptr = hbm_bo_map(staging);
	if (!ptr) {
		hbm_bo_destroy(staging);
		return NULL;
	}

	vma->priv = staging;

	return ptr;
}

void hbm_unmap(struct hbm *hbm, struct hbm_resource *res, struct vma *vma)
{
	if (!res->staging_size) {
		hbm_bo_unmap(res->bo);
		return;
	}

	struct hbm_bo *staging = vma->priv;
	hbm_bo_unmap(staging);
	hbm_bo_destroy(staging);
}

static bool wait_resource(struct hbm_resource *res, uint32_t map_flags)
{
	const int timeout = -1;

	if (res->implicit_fence_dmabuf < 0)
		return true;

	struct pollfd pollfd = {
		.fd = res->implicit_fence_dmabuf,
		.events = (map_flags & BO_MAP_WRITE) ? POLLOUT : POLLIN,
	};

	while (true) {
		const int ret = poll(&pollfd, 1, timeout);
		if (ret > 0)
			return pollfd.revents & pollfd.events;
		else if (ret == 0 || !(errno == EINTR || errno == EAGAIN))
			return false;
	}
}

bool hbm_sync(struct hbm *hbm, struct hbm_resource *res, const struct mapping *mapping,
	      uint32_t plane, bool flush)
{
	const struct rectangle *rect = &mapping->rect;

	if (!wait_resource(res, mapping->vma->map_flags))
		return false;

	if (!res->staging_size) {
		/* TODO respect rect */
		if (flush)
			hbm_bo_flush(res->bo);
		else
			hbm_bo_invalidate(res->bo);
		return true;
	}

	/* create_staging requires HBM_MEMORY_TYPE_COHERENT and there is no
	 * need to flush/invalidate
	 */
	struct hbm_bo *src;
	struct hbm_bo *dst;
	if (flush) {
		src = mapping->vma->priv;
		dst = res->bo;
	} else {
		src = res->bo;
		dst = mapping->vma->priv;
	}

	bool ret;
	if (res->format == DRM_FORMAT_INVALID) {
		const struct hbm_copy_buffer copy = {
			.src_offset = rect->x,
			.dst_offset = rect->x,
			.size = rect->width,
		};
		ret = hbm_bo_copy_buffer(dst, src, &copy, -1, NULL);
	} else {
		const uint32_t bpp = drv_bytes_per_pixel_from_format(res->format, plane);
		const uint64_t stride = res->staging_strides[plane];
		const uint64_t offset =
		    res->staging_offsets[plane] + stride * rect->y + bpp * rect->x;

		const struct hbm_copy_buffer_image copy = {
			.offset = offset,
			.stride = stride,
			.plane = plane,
			.x = rect->x,
			.y = rect->y,
			.width = rect->width,
			.height = rect->height,
		};
		ret = hbm_bo_copy_buffer_image(dst, src, &copy, -1, NULL);
	}

	return ret;
}

#ifdef DRV_AMDGPU

#include "dri.h"
#include <string.h>
#include <sys/mman.h>

void *dri_dlopen(const char *dri_so_path)
{
	return NULL;
}

void dri_dlclose(void *dri_so_handle)
{
}

struct dri_driver *dri_init(struct driver *drv, const char *dri_so_path, const char *driver_suffix)
{
	return (struct dri_driver *)hbm_create(drv_get_fd(drv));
}

void dri_close(struct dri_driver *dri)
{
	struct hbm *hbm = (struct hbm *)dri;
	hbm_destroy(hbm);
}

size_t dri_num_planes_from_modifier(struct dri_driver *dri, uint32_t format, uint64_t modifier)
{
	struct hbm *hbm = (struct hbm *)dri;

	/* amdgpu_import_bo can call this with DRM_FORMAT_MOD_INVALID */
	return modifier == DRM_FORMAT_MOD_INVALID
		   ? drv_num_planes_from_format(format)
		   : hbm_device_get_plane_count(hbm->device, format, modifier);
}

bool dri_query_modifiers(struct dri_driver *dri, uint32_t format, int max, uint64_t *modifiers,
			 int *count)
{
	struct hbm *hbm = (struct hbm *)dri;

	/* we have to guess the use flags */
	const uint64_t use_flags = BO_USE_RENDERING;

	struct hbm_description desc;
	init_description(hbm, format, DRM_FORMAT_MOD_INVALID, use_flags, &desc);

	/* if the device supports DRM_FORMAT_MOD_INVALID, it lacks explicit modifier support */
	if (hbm_device_has_modifier(hbm->device, &desc, DRM_FORMAT_MOD_INVALID))
		return false;

	*count = hbm_device_get_modifiers(hbm->device, &desc, max, modifiers);
	return *count >= 0 ? true : false;
}

int dri_bo_create(struct dri_driver *dri, struct bo *bo, uint32_t width, uint32_t height,
		  uint32_t format, uint64_t use_flags)
{
	return dri_bo_create_with_modifiers(dri, bo, width, height, format, use_flags, NULL, 0);
}

int dri_bo_create_with_modifiers(struct dri_driver *dri, struct bo *bo, uint32_t width,
				 uint32_t height, uint32_t format, uint64_t use_flags,
				 const uint64_t *modifiers, uint32_t modifier_count)
{
	struct hbm *hbm = (struct hbm *)dri;

	/* if there is no use flags, we have to guess (should we include USE_SW?) */
	if (!use_flags)
		use_flags = BO_USE_TEXTURE;

	bo->priv = hbm_allocate(hbm, width, height, format, use_flags, modifiers, modifier_count,
				&bo->meta);
	if (!bo->priv)
		return -1;

	/* TODO if there is no USE_SW, we can in theory destroy bo->priv after
	 * re-import
	 */
	bo->handle.u32 = hbm_reimport_to_driver(hbm, bo->priv, NULL);
	if (!bo->handle.u32) {
		hbm_free(hbm, bo->priv);
		return -1;
	}

	return 0;
}

int dri_bo_import(struct dri_driver *dri, struct bo *bo, struct drv_import_fd_data *data)
{
	struct hbm *hbm = (struct hbm *)dri;

	/* chrome's ProtectedBufferManager imports dma-bufs with invalid
	 * parameters, only to get their unique gem handles.  hbm rightfully
	 * rejects them so we have to work around.
	 */
	if (data->format_modifier == DRM_FORMAT_MOD_INVALID && !data->strides[0])
		return drv_prime_bo_import(bo, data);

	/* TODO if there is no USE_SW, we can in theory skip bo->priv */
	bo->priv = hbm_import(hbm, data, &bo->meta);
	if (!bo->priv)
		return -1;

	bo->handle.u32 = hbm_reimport_to_driver(hbm, bo->priv, data);
	if (!bo->handle.u32) {
		hbm_free(hbm, bo->priv);
		return -1;
	}

	return 0;
}

int dri_bo_release(struct dri_driver *dri, struct bo *bo)
{
	struct hbm *hbm = (struct hbm *)dri;
	hbm_free(hbm, bo->priv);
	return 0;
}

int dri_bo_destroy(struct dri_driver *dri, struct bo *bo)
{
	drv_gem_close(bo->drv, bo->handle.u32);
	return 0;
}

void *dri_bo_map(struct dri_driver *dri, struct bo *bo, struct vma *vma, size_t plane,
		 uint32_t map_flags)
{
	struct hbm *hbm = (struct hbm *)dri;

	assert(!plane);

	void *ptr = hbm_map(hbm, bo->priv, vma, map_flags);
	/* gbm returns NULL but minigbm returns MAP_FAILED on errors */
	if (!ptr)
		return MAP_FAILED;

	if (map_flags & BO_MAP_READ) {
		const struct mapping mapping = {
			.vma = vma,
			.rect = {
				.width = bo->meta.width,
				.height = bo->meta.height,
			},
		};
		const bool flush = false;
		hbm_sync(hbm, bo->priv, &mapping, 0, flush);
	}

	return ptr;
}

int dri_bo_unmap(struct dri_driver *dri, struct bo *bo, struct vma *vma)
{
	struct hbm *hbm = (struct hbm *)dri;

	if (vma->map_flags & BO_MAP_WRITE) {
		const struct mapping mapping = {
			.vma = vma,
			.rect = {
				.width = bo->meta.width,
				.height = bo->meta.height,
			},
		};
		const bool flush = true;
		hbm_sync(hbm, bo->priv, &mapping, 0, flush);
	}

	hbm_unmap(hbm, bo->priv, vma);
	return 0;
}

#endif /* DRV_AMDGPU */

#endif /* DRV_HBM_HELPER */
