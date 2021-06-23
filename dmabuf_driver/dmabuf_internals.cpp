#define LOG_TAG "DMABUF-GRALLOC"

extern "C" {
#include "drv_helpers.h"
}

#include "UniqueFd.h"
#include "dma-heap.h"
#include "drv_priv.h"
#include "util.h"
#include <algorithm>
#include <cutils/properties.h>
#include <errno.h>
#include <fcntl.h>
#include <gbm.h>
#include <glob.h>
#include <iterator>
#include <linux/dma-buf.h>
#include <log/log.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <xf86drm.h>
#include <xf86drmMode.h>

void dmabuf_resolve_format_and_use_flags(struct driver *drv, uint32_t format, uint64_t use_flags,
					 uint32_t *out_format, uint64_t *out_use_flags)
{
	*out_format = format;
	*out_use_flags = use_flags;
	switch (format) {
	case DRM_FORMAT_FLEX_IMPLEMENTATION_DEFINED:
		/* Camera subsystem requires NV12. */
		if (use_flags & (BO_USE_CAMERA_READ | BO_USE_CAMERA_WRITE)) {
			*out_format = DRM_FORMAT_NV12;
		} else {
			/*HACK: See b/28671744 */
			*out_format = DRM_FORMAT_XBGR8888;
		}
		break;
	case DRM_FORMAT_FLEX_YCbCr_420_888:
		*out_format = DRM_FORMAT_NV12;
		break;
	case DRM_FORMAT_BGR565:
		/* mesa3d doesn't support BGR565 */
		*out_format = DRM_FORMAT_RGB565;
		break;
	}
}

int dmabuf_driver_init(struct driver *drv)
{
	return 0;
}

struct DmabufDriver {
	UniqueFd system_heap_fd;
	UniqueFd system_heap_uncached_fd;
	UniqueFd cma_heap_fd;
};

struct DmabufDriverPriv {
	std::shared_ptr<DmabufDriver> dmabuf_drv;
};

static std::shared_ptr<DmabufDriver> dmabuf_get_or_init_driver(struct driver *drv)
{
	std::shared_ptr<DmabufDriver> dmabuf_drv;

	if (!drv->priv) {
		dmabuf_drv = std::make_unique<DmabufDriver>();
		dmabuf_drv->system_heap_fd =
		    UniqueFd(open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC));

		if (!dmabuf_drv->system_heap_fd) {
			drv_loge("Can't open system heap, errno: %i", -errno);
			return nullptr;
		}

		dmabuf_drv->system_heap_uncached_fd =
		    UniqueFd(open("/dev/dma_heap/system-uncached", O_RDONLY | O_CLOEXEC));

		if (!dmabuf_drv->system_heap_uncached_fd) {
			drv_logi("No system-uncached dmabuf-heap found. Falling back to system.");
			dmabuf_drv->system_heap_uncached_fd =
			    UniqueFd(dup(dmabuf_drv->system_heap_fd.Get()));
		}

		dmabuf_drv->cma_heap_fd =
		    UniqueFd(open("/dev/dma_heap/linux,cma", O_RDONLY | O_CLOEXEC));
		if (!dmabuf_drv->cma_heap_fd) {
			drv_logi("No system-uncached dmabuf-heap found. Falling back to system.");
			dmabuf_drv->cma_heap_fd = UniqueFd(dup(dmabuf_drv->system_heap_fd.Get()));
		}

		auto priv = new DmabufDriverPriv();
		priv->dmabuf_drv = dmabuf_drv;
		drv->priv = priv;
	} else {
		dmabuf_drv = ((DmabufDriverPriv *)drv->priv)->dmabuf_drv;
	}

	return dmabuf_drv;
}

void dmabuf_driver_close(struct driver *drv)
{
	if (drv->priv) {
		delete (DmabufDriverPriv *)(drv->priv);
		drv->priv = nullptr;
	}
}

struct DmabufBoPriv {
	UniqueFd fds[DRV_MAX_PLANES];
};

static bool unmask64(uint64_t *value, uint64_t mask)
{
	if ((*value & mask) != 0) {
		*value &= ~mask;
		return true;
	}
	return false;
}

static const uint32_t supported_formats[] = {
	DRM_FORMAT_ARGB8888,	   DRM_FORMAT_XRGB8888, DRM_FORMAT_ABGR8888, DRM_FORMAT_XBGR8888,
	DRM_FORMAT_RGB565,	   DRM_FORMAT_BGR888,	DRM_FORMAT_NV12,     DRM_FORMAT_YVU420,
	DRM_FORMAT_YVU420_ANDROID, DRM_FORMAT_R8,
};

static bool is_format_supported(uint32_t format)
{
	return std::find(std::begin(supported_formats), std::end(supported_formats), format) !=
	       std::end(supported_formats);
}

int dmabuf_bo_create2(struct bo *bo, uint32_t width, uint32_t height, uint32_t format,
		      uint64_t use_flags, bool test_only)
{
	auto drv = dmabuf_get_or_init_driver(bo->drv);
	uint64_t l_use_flags = use_flags;

	/* Check if format is supported */
	if (!is_format_supported(format)) {
		char format_str[5] = { 0 };
		memcpy(format_str, &format, 4);
		drv_loge("Format %s is not supported", format_str);
		return -EINVAL;
	}

	if (drv == nullptr)
		return -EINVAL;

	int stride = drv_stride_from_format(format, width, 0);

	bool force_cma = false;
	int heap_fd = drv->system_heap_fd.Get();
	uint32_t size_align = 4096;

	bool sw_mask = unmask64(&l_use_flags, BO_USE_SW_MASK);

	//	if (!(use_flags & BO_USE_SW_MASK))
//	heap_fd = drv->system_heap_uncached_fd.Get();

	if (unmask64(&l_use_flags, BO_USE_SCANOUT))
		force_cma = true;

	/* RPI4 camera over libcamera */
	if (unmask64(&l_use_flags, BO_USE_CAMERA_READ | BO_USE_CAMERA_WRITE)) {
		force_cma = true;
		stride = ALIGN(stride, 32);
		if (height > 1)
			size_align = (ALIGN(width, 32) * ALIGN(height, 16) * 3) >> 1;
	}

	/* RPI4 hwcodecs */
	if (unmask64(&l_use_flags, BO_USE_HW_VIDEO_DECODER | BO_USE_HW_VIDEO_ENCODER)) {
		force_cma = true;
		stride = ALIGN(stride, 32);
		//				if (height > 1)
		//					height = ALIGN(height, 16);
		size_align = 4096;
	}

	/* On RPI4 the following flags fits any HEAP type */
	unmask64(&l_use_flags, BO_USE_CURSOR | BO_USE_TEXTURE | BO_USE_RENDERING);

	if (l_use_flags != 0) {
		char use_str[128] = { 0 };
		drv_use_flags_to_string(l_use_flags, use_str, sizeof(use_str));
		drv_loge("Unsupported use flags: %s", use_str);
		return -EINVAL;
	}

	if (test_only)
		return 0;

	if (force_cma)
		heap_fd = drv->cma_heap_fd.Get();

	if (!force_cma && !sw_mask) {
		// TODO: GPU-only buffers can be made tiled, but calculations are too complex
	}

	drv_bo_from_format(bo, stride, 1, height, format);

	struct dma_heap_allocation_data heap_data {
		.len = ALIGN(bo->meta.total_size, size_align), .fd_flags = O_RDWR | O_CLOEXEC,
	};

	char use_str[128];
	use_str[0] = 0;

	drv_use_flags_to_string(use_flags, use_str, sizeof(use_str));

	char format_str[5] = { 0 };
	memcpy(format_str, &format, 4);

	drv_logi("Allocate buffer, %s %dx%d, stride %d, total_size: %llu, use: %s", format_str,
		 width, height, stride, heap_data.len, use_str);

	int ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &heap_data);
	if (ret) {
		drv_loge("Failed to allocate dmabuf: %s", strerror(errno));
		return -errno;
	}

	use_str[0] = 0;
	drv_use_flags_to_string_short(use_flags, use_str, sizeof(use_str));
	auto buf_fd = UniqueFd(heap_data.fd);

	char dmabuf_name[128];
	snprintf(dmabuf_name, sizeof(dmabuf_name), "%dx%d %c%c%c%c %s", width, height,
		 format & 0x7f, (format >> 8) & 0x7f, (format >> 16) & 0x7f, (format >> 24) & 0x7f,
		 use_str);
	ret = ioctl(buf_fd.Get(), DMA_BUF_SET_NAME, dmabuf_name);
	if (ret)
		drv_loge("DMA_BUF_SET_NAME failed");

	auto priv = new DmabufBoPriv();
	bo->inode = drv_get_inode(buf_fd.Get());
	for (size_t plane = 0; plane < bo->meta.num_planes; plane++) {
		priv->fds[plane] = UniqueFd(dup(buf_fd.Get()));

	}

	bo->priv = priv;

	return 0;
}

int dmabuf_bo_import(struct bo *bo, struct drv_import_fd_data *data)
{
	if (bo->priv) {
		drv_loge("%s bo isn't empty", __func__);
		return -EINVAL;
	}
	auto priv = new DmabufBoPriv();
	for (size_t plane = 0; plane < bo->meta.num_planes; plane++) {
		priv->fds[plane] = UniqueFd(dup(data->fds[plane]));
	}

	bo->priv = priv;

	return 0;
}

int dmabuf_bo_destroy(struct bo *bo)
{
	if (bo->priv) {
		delete (DmabufBoPriv *)(bo->priv);
		bo->priv = nullptr;
	}
	return 0;
}

int dmabuf_bo_get_plane_fd(struct bo *bo, size_t plane)
{
	return dup(((DmabufBoPriv *)bo->priv)->fds[plane].Get());
}

void *dmabuf_bo_map(struct bo *bo, struct vma *vma, uint32_t map_flags)
{
	vma->length = bo->meta.total_size;

	auto priv = (DmabufBoPriv *)bo->priv;

	void *buf =
	    mmap(0, vma->length, drv_get_prot(map_flags), MAP_SHARED, priv->fds[0].Get(), 0);
	if (buf == MAP_FAILED) {
		drv_loge("%s mmap err, errno: %i", __func__, -errno);
		return buf;
	}

	struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW };
	int ret = ioctl(priv->fds[0].Get(), DMA_BUF_IOCTL_SYNC, &sync);
	if (ret)
		drv_loge("DMA_BUF_IOCTL_SYNC DMA_BUF_SYNC_START failed");

	return buf;
}

int dmabuf_bo_unmap(struct bo *bo, struct vma *vma)
{
	auto priv = (DmabufBoPriv *)bo->priv;

	struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW };
	int ret = ioctl(priv->fds[0].Get(), DMA_BUF_IOCTL_SYNC, &sync);
	if (ret)
		drv_loge("DMA_BUF_IOCTL_SYNC DMA_BUF_SYNC_END failed");

	return munmap(vma->addr, vma->length);
}

int dmabuf_bo_flush(struct bo *bo, struct mapping *mapping)
{
	auto priv = (DmabufBoPriv *)bo->priv;
	struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW };

	int ret = ioctl(priv->fds[0].Get(), DMA_BUF_IOCTL_SYNC, &sync);
	if (ret)
		drv_loge("DMA_BUF_IOCTL_SYNC DMA_BUF_SYNC_END failed");

	sync = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW };
	ret = ioctl(priv->fds[0].Get(), DMA_BUF_IOCTL_SYNC, &sync);
	if (ret)
		drv_loge("DMA_BUF_IOCTL_SYNC DMA_BUF_SYNC_START failed");

	return 0;
}
