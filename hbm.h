/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifdef DRV_HBM_HELPER

#include "drv.h"
#include "drv_priv.h"

struct hbm;
struct hbm_resource;

struct hbm *hbm_create(int drv_fd);
void hbm_destroy(struct hbm *hbm);

uint64_t *hbm_get_format_modifiers(struct hbm *hbm, uint32_t fmt, uint64_t use_flags,
				   uint32_t *out_count);

struct hbm_resource *hbm_allocate(struct hbm *hbm, uint32_t width, uint32_t height, uint32_t fmt,
				  uint64_t use_flags, const uint64_t *mods, uint32_t mod_count,
				  struct bo_metadata *out_meta);
struct hbm_resource *hbm_import(struct hbm *hbm, const struct drv_import_fd_data *import_data,
				struct bo_metadata *out_meta);
void hbm_free(struct hbm *hbm, struct hbm_resource *res);

uint32_t hbm_reimport_to_driver(struct hbm *hbm, struct hbm_resource *res,
				const struct drv_import_fd_data *import_data);

void *hbm_map(struct hbm *hbm, struct hbm_resource *res, struct vma *vma, uint32_t map_flags);
void hbm_unmap(struct hbm *hbm, struct hbm_resource *res, struct vma *vma);
bool hbm_sync(struct hbm *hbm, struct hbm_resource *res, const struct mapping *mapping,
	      uint32_t plane, bool flush);

#endif /* DRV_HBM_HELPER */
