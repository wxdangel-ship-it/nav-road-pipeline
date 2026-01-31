from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from rasterio.transform import from_origin


@dataclass
class RasterBundle:
    height_p10: np.ndarray
    intensity_max: np.ndarray
    roughness: np.ndarray
    density: np.ndarray
    density_all: np.ndarray
    transform: object
    res_m: float
    minx: float
    miny: float
    width: int
    height: int
    point_ix: np.ndarray
    point_iy: np.ndarray
    point_valid: np.ndarray
    point_height_p10: np.ndarray


def _grid_spec(bounds: Tuple[float, float, float, float], res_m: float) -> Tuple[float, float, int, int]:
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / res_m)) + 1
    height = int(np.ceil((maxy - miny) / res_m)) + 1
    return float(minx), float(miny), width, height


def _group_slices(lin_idx: np.ndarray):
    order = np.argsort(lin_idx, kind="mergesort")
    lin_sorted = lin_idx[order]
    uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    return order, uniq, start, counts


def build_rasters(
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    roi_geom: object,
    res_m: float,
    ground_band_dz_m: float,
) -> RasterBundle:
    if points_xyz.size == 0:
        minx, miny, maxx, maxy = roi_geom.bounds
        minx, miny, width, height = _grid_spec((minx, miny, maxx, maxy), res_m)
        transform = from_origin(minx, miny + height * res_m, res_m, res_m)
        nan = np.full((height, width), np.nan, dtype=np.float32)
        zeros = np.zeros((height, width), dtype=np.float32)
        return RasterBundle(
            height_p10=nan.copy(),
            intensity_max=nan.copy(),
            roughness=nan.copy(),
            density=zeros.copy(),
            density_all=zeros.copy(),
            transform=transform,
            res_m=res_m,
            minx=minx,
            miny=miny,
            width=width,
            height=height,
            point_ix=np.empty((0,), dtype=np.int32),
            point_iy=np.empty((0,), dtype=np.int32),
            point_valid=np.empty((0,), dtype=bool),
            point_height_p10=np.empty((0,), dtype=np.float32),
        )

    minx, miny, maxx, maxy = roi_geom.bounds
    minx, miny, width, height = _grid_spec((minx, miny, maxx, maxy), res_m)
    transform = from_origin(minx, miny + height * res_m, res_m, res_m)

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    ix = np.floor((x - minx) / res_m).astype(np.int32)
    iy = np.floor((y - miny) / res_m).astype(np.int32)
    valid = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
    ixv = ix[valid]
    iyv = iy[valid]
    zv = z[valid]
    iv = intensity[valid]

    lin = iyv.astype(np.int64) * int(width) + ixv.astype(np.int64)
    order, uniq, start, counts = _group_slices(lin)
    ix_sorted = ixv[order]
    iy_sorted = iyv[order]
    z_sorted = zv[order]
    i_sorted = iv[order]

    height_p10 = np.full((height, width), np.nan, dtype=np.float32)
    density_all = np.zeros((height, width), dtype=np.float32)
    for u, s, c in zip(uniq, start, counts):
        sl = slice(s, s + c)
        cx = int(ix_sorted[s])
        cy = int(iy_sorted[s])
        height_p10[cy, cx] = np.percentile(z_sorted[sl], 10).astype(np.float32)
        density_all[cy, cx] = float(c)

    # Ground band per point: z within p10 + dz
    p10_per_point = height_p10[iyv, ixv]
    ground_band = np.isfinite(p10_per_point) & (zv <= (p10_per_point + float(ground_band_dz_m)))
    ixg = ixv[ground_band]
    iyg = iyv[ground_band]
    zg = zv[ground_band]
    ig = iv[ground_band]

    intensity_max = np.full((height, width), np.nan, dtype=np.float32)
    roughness = np.full((height, width), np.nan, dtype=np.float32)
    density = np.zeros((height, width), dtype=np.float32)

    if ixg.size > 0:
        lin_g = iyg.astype(np.int64) * int(width) + ixg.astype(np.int64)
        order_g, uniq_g, start_g, counts_g = _group_slices(lin_g)
        ixg_s = ixg[order_g]
        iyg_s = iyg[order_g]
        zg_s = zg[order_g]
        ig_s = ig[order_g]
        for u, s, c in zip(uniq_g, start_g, counts_g):
            sl = slice(s, s + c)
            cx = int(ixg_s[s])
            cy = int(iyg_s[s])
            density[cy, cx] = float(c)
            intensity_max[cy, cx] = float(np.max(ig_s[sl]))
            roughness[cy, cx] = float(np.std(zg_s[sl]))

    point_height = np.full_like(x, np.nan, dtype=np.float32)
    point_height[valid] = height_p10[iyv, ixv]

    return RasterBundle(
        height_p10=height_p10,
        intensity_max=intensity_max,
        roughness=roughness,
        density=density,
        density_all=density_all,
        transform=transform,
        res_m=res_m,
        minx=minx,
        miny=miny,
        width=width,
        height=height,
        point_ix=ix,
        point_iy=iy,
        point_valid=valid,
        point_height_p10=point_height,
    )


__all__ = ["RasterBundle", "build_rasters"]
