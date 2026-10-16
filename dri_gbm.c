/*
 * Copyright 2026 The AnyBook Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

/*
 * The dri.h interface of the amdgpu backend, implemented on Mesa's public GBM API.
 *
 * Mesa 24.3 and later no longer export __driDriverGetExtensions_<driver> from the
 * gallium megadriver, so dri.c cannot reach radeonsi any more. libgbm is Mesa's
 * stable allocation ABI, so this file allocates tiled buffers through it instead:
 *
 *  - libgbm is dlopen()ed RTLD_LOCAL: libgbm_mesa.so on Android (minigbm owns the
 *    libgbm.so name there), libgbm.so.1 elsewhere. Mesa's gbm.h is not included
 *    because minigbm's own gbm.h uses different flag values; the few types and
 *    constants used here are restated with Mesa's ABI values.
 *  - The GBM device sits on its own render-node fd, so its GEM handles never alias
 *    minigbm's (as in dri.c). Buffers reach minigbm's fd through PRIME, and the GBM
 *    bo is dropped right after allocation: minigbm's handle keeps the buffer alive,
 *    and radeonsi has stored the tiling metadata in the kernel BO when exporting it.
 *  - The GBM device, which loads the gallium driver, is created on first use. Imports
 *    only take a PRIME handle, so mapper and gralloc0 instances in app processes load
 *    the driver only if they map a tiled buffer. Each such mapping imports its own GBM
 *    bo, so a mapping never depends on the lifetime of the bo that created it.
 *  - GBM cannot enumerate modifiers. dri_query_modifiers() reports none, so the
 *    amdgpu backend uses TILE_TYPE_DRI and the driver chooses the layout.
 *
 * debug.minigbm.amdgpu.linear_only=true (or an environment variable of that name)
 * skips GBM; the amdgpu backend then allocates linear buffers only.
 */

#ifdef DRV_AMDGPU

#include <dlfcn.h>
#include <drm_fourcc.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>

#include "dri.h"
#include "drv_helpers.h"
#include "drv_priv.h"
#include "util.h"

#ifdef __LP64__
#define GBM_LIBRARY_BASEPATH "/vendor/lib64/"
#else
#define GBM_LIBRARY_BASEPATH "/vendor/lib/"
#endif

#ifdef __ANDROID__
#define GBM_LIBRARY GBM_LIBRARY_BASEPATH "libgbm_mesa.so"
#else
#define GBM_LIBRARY "libgbm.so.1"
#endif

#define LINEAR_ONLY_OPTION "vendor.minigbm.amdgpu.linear_only"

/* Mesa gbm.h ABI. */
struct gbm_device;
struct gbm_bo;

#define MESA_GBM_BO_USE_SCANOUT (1 << 0)
#define MESA_GBM_BO_USE_CURSOR (1 << 1)
#define MESA_GBM_BO_USE_RENDERING (1 << 2)
#define MESA_GBM_BO_USE_LINEAR (1 << 4)
#define MESA_GBM_BO_USE_PROTECTED (1 << 5)
#define MESA_GBM_BO_IMPORT_FD_MODIFIER 0x5504
#define MESA_GBM_MAX_PLANES 4

struct mesa_gbm_import_fd_modifier_data {
	uint32_t width;
	uint32_t height;
	uint32_t format;
	uint32_t num_fds;
	int fds[MESA_GBM_MAX_PLANES];
	int strides[MESA_GBM_MAX_PLANES];
	int offsets[MESA_GBM_MAX_PLANES];
	uint64_t modifier;
};

/* gbm_bo_map() takes GBM_BO_TRANSFER_READ (1) / _WRITE (2), the values of BO_MAP_*. */
_Static_assert(BO_MAP_READ == 1 && BO_MAP_WRITE == 2, "BO_MAP_* must match GBM_BO_TRANSFER_*");
_Static_assert(DRV_MAX_PLANES <= MESA_GBM_MAX_PLANES, "plane count");

struct gbm_api {
	struct gbm_device *(*create_device)(int fd);
	void (*device_destroy)(struct gbm_device *gbm);
	int (*device_get_format_modifier_plane_count)(struct gbm_device *gbm, uint32_t format,
						      uint64_t modifier);
	struct gbm_bo *(*bo_create)(struct gbm_device *gbm, uint32_t width, uint32_t height,
				    uint32_t format, uint32_t flags);
	struct gbm_bo *(*bo_create_with_modifiers2)(struct gbm_device *gbm, uint32_t width,
						    uint32_t height, uint32_t format,
						    const uint64_t *modifiers,
						    const unsigned int count, uint32_t flags);
	struct gbm_bo *(*bo_import)(struct gbm_device *gbm, uint32_t type, void *buffer,
				    uint32_t flags);
	void (*bo_destroy)(struct gbm_bo *bo);
	uint64_t (*bo_get_modifier)(struct gbm_bo *bo);
	int (*bo_get_plane_count)(struct gbm_bo *bo);
	int (*bo_get_fd)(struct gbm_bo *bo);
	uint32_t (*bo_get_stride_for_plane)(struct gbm_bo *bo, int plane);
	uint32_t (*bo_get_offset)(struct gbm_bo *bo, int plane);
	void *(*bo_map)(struct gbm_bo *bo, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
			uint32_t flags, uint32_t *stride, void **map_data);
	void (*bo_unmap)(struct gbm_bo *bo, void *map_data);
};

struct dri_driver {
	int fd; /* GBM's own render node */
	void *lib;
	struct gbm_api api;
	pthread_mutex_t lock;
	struct gbm_device *gbm; /* created on first use */
	bool gbm_failed;
};

/* vma->priv of a mapping made here. */
struct dri_mapping {
	struct gbm_bo *gbm_bo;
	void *map_data;
};

/*
 * bo->priv of every buffer created or imported through this file. amdgpu.c only tests it
 * for NULL to route release/destroy/map/unmap here; no per-buffer state is needed.
 */
static char dri_bo_marker;

static bool load_gbm(struct dri_driver *dri)
{
	struct {
		void **fn;
		const char *name;
		bool required;
	} syms[] = {
		{ (void **)&dri->api.create_device, "gbm_create_device", true },
		{ (void **)&dri->api.device_destroy, "gbm_device_destroy", true },
		{ (void **)&dri->api.device_get_format_modifier_plane_count,
		  "gbm_device_get_format_modifier_plane_count", true },
		{ (void **)&dri->api.bo_create, "gbm_bo_create", true },
		{ (void **)&dri->api.bo_create_with_modifiers2, "gbm_bo_create_with_modifiers2", false },
		{ (void **)&dri->api.bo_import, "gbm_bo_import", true },
		{ (void **)&dri->api.bo_destroy, "gbm_bo_destroy", true },
		{ (void **)&dri->api.bo_get_modifier, "gbm_bo_get_modifier", true },
		{ (void **)&dri->api.bo_get_plane_count, "gbm_bo_get_plane_count", true },
		{ (void **)&dri->api.bo_get_fd, "gbm_bo_get_fd", true },
		{ (void **)&dri->api.bo_get_stride_for_plane, "gbm_bo_get_stride_for_plane", true },
		{ (void **)&dri->api.bo_get_offset, "gbm_bo_get_offset", true },
		{ (void **)&dri->api.bo_map, "gbm_bo_map", true },
		{ (void **)&dri->api.bo_unmap, "gbm_bo_unmap", true },
	};

	dri->lib = dlopen(GBM_LIBRARY, RTLD_NOW | RTLD_LOCAL);
	if (!dri->lib) {
		drv_logi("amdgpu: %s not loaded: %s\n", GBM_LIBRARY, dlerror());
		return false;
	}

	for (size_t i = 0; i < ARRAY_SIZE(syms); i++) {
		*syms[i].fn = dlsym(dri->lib, syms[i].name);
		if (!*syms[i].fn && syms[i].required) {
			drv_loge("amdgpu: %s lacks %s\n", GBM_LIBRARY, syms[i].name);
			dlclose(dri->lib);
			dri->lib = NULL;
			return false;
		}
	}

	return true;
}

static struct gbm_device *dri_gbm(struct dri_driver *dri)
{
	struct gbm_device *gbm;

	if (!dri)
		return NULL;

	pthread_mutex_lock(&dri->lock);
	if (!dri->gbm && !dri->gbm_failed) {
		dri->gbm = dri->api.create_device(dri->fd);
		if (!dri->gbm) {
			dri->gbm_failed = true;
			drv_loge("amdgpu: gbm_create_device failed\n");
		}
	}
	gbm = dri->gbm;
	pthread_mutex_unlock(&dri->lock);

	return gbm;
}

void *dri_dlopen(const char *dri_so_path)
{
	/* amdgpu_preload() names radeonsi_dri.so; what this file loads is libgbm. */
	(void)dri_so_path;
	return dlopen(GBM_LIBRARY, RTLD_NOW | RTLD_LOCAL);
}

void dri_dlclose(void *dri_so_handle)
{
	if (dri_so_handle)
		dlclose(dri_so_handle);
}

struct dri_driver *dri_init(struct driver *drv, const char *dri_so_path, const char *driver_suffix)
{
	const char *linear_only = drv_get_os_option(LINEAR_ONLY_OPTION);
	struct dri_driver *dri;
	char *node_name;

	(void)dri_so_path;
	(void)driver_suffix;

	if (linear_only && (!strcmp(linear_only, "1") || !strcmp(linear_only, "true"))) {
		drv_logi("amdgpu: %s is set\n", LINEAR_ONLY_OPTION);
		return NULL;
	}

	dri = calloc(1, sizeof(*dri));
	if (!dri)
		return NULL;

	node_name = drmGetRenderDeviceNameFromFd(drv_get_fd(drv));
	if (!node_name)
		goto free_dri;

	dri->fd = open(node_name, O_RDWR | O_CLOEXEC);
	free(node_name);
	if (dri->fd < 0)
		goto free_dri;

	if (!load_gbm(dri))
		goto close_dri_fd;

	pthread_mutex_init(&dri->lock, NULL);
	return dri;

close_dri_fd:
	close(dri->fd);
free_dri:
	free(dri);
	return NULL;
}

void dri_close(struct dri_driver *dri)
{
	if (!dri)
		return;

	if (dri->gbm)
		dri->api.device_destroy(dri->gbm);
	dlclose(dri->lib);
	close(dri->fd);
	pthread_mutex_destroy(&dri->lock);
	free(dri);
}

static uint32_t gbm_flags_from_use_flags(uint64_t use_flags)
{
	uint32_t flags = MESA_GBM_BO_USE_RENDERING;

	if (use_flags & BO_USE_SCANOUT)
		flags |= MESA_GBM_BO_USE_SCANOUT;
	if (use_flags & BO_USE_CURSOR)
		flags |= MESA_GBM_BO_USE_CURSOR;
	if (use_flags & BO_USE_LINEAR)
		flags |= MESA_GBM_BO_USE_LINEAR;
	if (use_flags & BO_USE_PROTECTED)
		flags |= MESA_GBM_BO_USE_PROTECTED;

	return flags;
}

/* Take the layout and a PRIME handle on minigbm's fd from a freshly allocated GBM bo. */
static int import_into_minigbm(struct dri_driver *dri, struct bo *bo, struct gbm_bo *gbm_bo)
{
	uint32_t handle = 0;
	int ret, num_planes, prime_fd;
	off_t dmabuf_size;

	num_planes = dri->api.bo_get_plane_count(gbm_bo);
	if (num_planes <= 0 || num_planes > DRV_MAX_PLANES)
		return -EINVAL;

	prime_fd = dri->api.bo_get_fd(gbm_bo);
	if (prime_fd < 0)
		return -EINVAL;

	dmabuf_size = lseek(prime_fd, 0, SEEK_END);
	if (dmabuf_size == (off_t)-1) {
		ret = -errno;
		close(prime_fd);
		return ret;
	}

	ret = drmPrimeFDToHandle(bo->drv->fd, prime_fd, &handle);
	close(prime_fd);
	if (ret) {
		drv_loge("drmPrimeFDToHandle failed with %s\n", strerror(errno));
		return -errno;
	}

	bo->handle.u32 = handle;
	bo->meta.format_modifier = dri->api.bo_get_modifier(gbm_bo);
	bo->meta.num_planes = num_planes;
	for (int i = 0; i < num_planes; ++i) {
		bo->meta.strides[i] = dri->api.bo_get_stride_for_plane(gbm_bo, i);
		bo->meta.offsets[i] = dri->api.bo_get_offset(gbm_bo, i);
		if (i > 0)
			bo->meta.sizes[i - 1] = bo->meta.offsets[i] - bo->meta.offsets[i - 1];
	}
	bo->meta.sizes[num_planes - 1] = dmabuf_size - bo->meta.offsets[num_planes - 1];
	bo->meta.total_size = dmabuf_size;

	return 0;
}

static int finish_create(struct dri_driver *dri, struct bo *bo, struct gbm_bo *gbm_bo)
{
	int ret;

	if (!gbm_bo)
		return errno ? -errno : -EINVAL;

	ret = import_into_minigbm(dri, bo, gbm_bo);
	dri->api.bo_destroy(gbm_bo);
	if (ret)
		return ret;

	bo->priv = &dri_bo_marker;
	return 0;
}

int dri_bo_create(struct dri_driver *dri, struct bo *bo, uint32_t width, uint32_t height,
		  uint32_t format, uint64_t use_flags)
{
	struct gbm_device *gbm = dri_gbm(dri);

	if (!gbm)
		return -ENODEV;

	errno = 0;
	return finish_create(dri, bo,
			     dri->api.bo_create(gbm, width, height, format,
						gbm_flags_from_use_flags(use_flags)));
}

int dri_bo_create_with_modifiers(struct dri_driver *dri, struct bo *bo, uint32_t width,
				 uint32_t height, uint32_t format, uint64_t use_flags,
				 const uint64_t *modifiers, uint32_t modifier_count)
{
	struct gbm_device *gbm = dri_gbm(dri);

	if (!gbm)
		return -ENODEV;
	if (!dri->api.bo_create_with_modifiers2)
		return -ENOENT;

	errno = 0;
	return finish_create(dri, bo,
			     dri->api.bo_create_with_modifiers2(
				 gbm, width, height, format, modifiers, modifier_count,
				 gbm_flags_from_use_flags(use_flags) & ~MESA_GBM_BO_USE_LINEAR));
}

/* Works without a dri_driver (linear-only processes): an import needs only the handle. */
int dri_bo_import(struct dri_driver *dri, struct bo *bo, struct drv_import_fd_data *data)
{
	int ret;

	(void)dri;

	ret = drv_prime_bo_import(bo, data);
	if (ret)
		return ret;

	bo->priv = &dri_bo_marker;
	return 0;
}

int dri_bo_release(struct dri_driver *dri, struct bo *bo)
{
	/* Nothing per buffer; bo->priv stays set so that destroy is routed here. */
	(void)dri;
	(void)bo;
	return 0;
}

int dri_bo_destroy(struct dri_driver *dri, struct bo *bo)
{
	(void)dri;
	drv_gem_close(bo->drv, bo->handle.u32);
	bo->priv = NULL;
	return 0;
}

/*
 * Map a buffer through a GBM bo imported for this mapping only. The driver does the
 * detiling (and retiling on unmap) blit when the layout is not linear.
 *
 * This function itself is not thread-safe; we rely on the fact that the caller
 * locks a per-driver mutex.
 */
void *dri_bo_map(struct dri_driver *dri, struct bo *bo, struct vma *vma, size_t plane,
		 uint32_t map_flags)
{
	struct mesa_gbm_import_fd_modifier_data data = { 0 };
	struct gbm_device *gbm = dri_gbm(dri);
	struct dri_mapping *mapping;
	void *addr;
	int prime_fd;

	if (!gbm)
		return MAP_FAILED;

	if (drmPrimeHandleToFD(bo->drv->fd, bo->handle.u32, DRM_CLOEXEC | DRM_RDWR, &prime_fd)) {
		drv_loge("drmPrimeHandleToFD failed with %s\n", strerror(errno));
		return MAP_FAILED;
	}

	data.width = bo->meta.width;
	data.height = bo->meta.height;
	data.format = drv_get_standard_fourcc(bo->meta.format);
	data.num_fds = bo->meta.num_planes;
	data.modifier = bo->meta.format_modifier;
	for (size_t p = 0; p < bo->meta.num_planes; p++) {
		data.fds[p] = prime_fd;
		data.strides[p] = bo->meta.strides[p];
		data.offsets[p] = bo->meta.offsets[p];
	}

	mapping = calloc(1, sizeof(*mapping));
	if (!mapping) {
		close(prime_fd);
		return MAP_FAILED;
	}

	mapping->gbm_bo = dri->api.bo_import(gbm, MESA_GBM_BO_IMPORT_FD_MODIFIER, &data, 0);
	close(prime_fd);
	if (!mapping->gbm_bo) {
		drv_loge("amdgpu: gbm_bo_import failed (%ux%u, format %.4s, modifier 0x%" PRIx64
			 ")\n",
			 data.width, data.height, (const char *)&data.format, data.modifier);
		free(mapping);
		return MAP_FAILED;
	}

	addr = dri->api.bo_map(mapping->gbm_bo, 0, 0, bo->meta.width, bo->meta.height, map_flags,
			       &vma->map_strides[plane], &mapping->map_data);
	if (!addr) {
		dri->api.bo_destroy(mapping->gbm_bo);
		free(mapping);
		return MAP_FAILED;
	}

	vma->priv = mapping;
	return addr;
}

int dri_bo_unmap(struct dri_driver *dri, struct bo *bo, struct vma *vma)
{
	struct dri_mapping *mapping = vma->priv;

	(void)bo;
	if (!mapping)
		return -EINVAL;

	/* gbm_bo_unmap() flushes, so a write-back blit has been submitted on return. */
	dri->api.bo_unmap(mapping->gbm_bo, mapping->map_data);
	dri->api.bo_destroy(mapping->gbm_bo);
	free(mapping);
	vma->priv = NULL;
	return 0;
}

size_t dri_num_planes_from_modifier(struct dri_driver *dri, uint32_t format, uint64_t modifier)
{
	struct gbm_device *gbm;
	int planes;

	if (modifier == DRM_FORMAT_MOD_INVALID || modifier == DRM_FORMAT_MOD_LINEAR)
		return drv_num_planes_from_format(format);

	/* AMD modifiers: DCC adds a metadata plane, and a displayable DCC copy when retiled. */
	if (fourcc_mod_is_vendor(modifier, AMD)) {
		if (!AMD_FMT_MOD_GET(DCC, modifier))
			return drv_num_planes_from_format(format);
		return AMD_FMT_MOD_GET(DCC_RETILE, modifier) ? 3 : 2;
	}

	gbm = dri_gbm(dri);
	if (gbm) {
		planes = dri->api.device_get_format_modifier_plane_count(gbm, format, modifier);
		if (planes > 0)
			return planes;
	}

	return drv_num_planes_from_format(format);
}

bool dri_query_modifiers(struct dri_driver *dri, uint32_t format, int max, uint64_t *modifiers,
			 int *count)
{
	/* GBM has no modifier query; see the comment at the top of this file. */
	(void)dri;
	(void)format;
	(void)max;
	(void)modifiers;
	(void)count;
	return false;
}

#endif
