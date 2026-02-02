from __future__ import annotations

"""
BEV 标线特征导出：从地面点云生成 tiles 与 ROI patches。
确定性参数，不做 auto-tune。
"""

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from pipeline.bev_markings import box_sum, load_polygon_bounds, read_laz_points

LOG = logging.getLogger("bev_dataset_export")


def _res_tag(res_m: float) -> str:
    return f"r{int(round(res_m * 100)):03d}m"


def bev_tiles_dir(out_dir: Path, res_m: float) -> Path:
    return out_dir / f"bev_markings_utm32_tiles_{_res_tag(res_m)}"


def bev_tiles_index_path(out_dir: Path, res_m: float) -> Path:
    return out_dir / f"bev_markings_tiles_index_{_res_tag(res_m)}.geojson"


def _grid_bounds_from_points(
    laz_paths: Iterable[Path],
    pad_m: float,
) -> Tuple[float, float, float, float]:
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    for p in laz_paths:
        import laspy

        with laspy.open(p) as reader:
            hdr = reader.header
            minx = min(minx, float(hdr.mins[0]))
            miny = min(miny, float(hdr.mins[1]))
            maxx = max(maxx, float(hdr.maxs[0]))
            maxy = max(maxy, float(hdr.maxs[1]))
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        raise RuntimeError("points_bounds_invalid")
    return (minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


def _tile_index(
    bounds: Tuple[float, float, float, float],
    res_m: float,
    tile_size_px: int,
) -> Tuple[int, int, float, float, float]:
    minx, miny, maxx, maxy = bounds
    tile_size_m = tile_size_px * res_m
    tiles_x = int(math.ceil((maxx - minx) / tile_size_m))
    tiles_y = int(math.ceil((maxy - miny) / tile_size_m))
    return tiles_x, tiles_y, tile_size_m, minx, maxy


def _ensure_tile_arrays(
    tiles: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    tile_key: Tuple[int, int],
    tile_size_px: int,
) -> Dict[str, np.ndarray]:
    if tile_key in tiles:
        return tiles[tile_key]
    arrays = {
        "intensity_max": np.zeros((tile_size_px, tile_size_px), dtype=np.uint16),
        "density": np.zeros((tile_size_px, tile_size_px), dtype=np.uint32),
    }
    tiles[tile_key] = arrays
    return arrays


def _compute_top_hat(
    intensity_max: np.ndarray,
    mask: np.ndarray,
    win_large_m: float,
    win_small_m: float,
    res_m: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_l = int(round((win_large_m / 2.0) / res_m))
    r_s = int(round((win_small_m / 2.0) / res_m))
    a = intensity_max.astype(np.float32) * mask.astype(np.float32)
    w = mask.astype(np.float32)
    sum_a = box_sum(a, r_l)
    sum_w = box_sum(w, r_l)
    mean_l = sum_a / np.maximum(sum_w, 1.0)
    std_l = box_sum(a * a, r_l) / np.maximum(sum_w, 1.0) - mean_l * mean_l
    std_l[std_l < 0] = 0
    std_l = np.sqrt(std_l)
    top_hat = intensity_max.astype(np.float32) - mean_l
    top_hat[top_hat < 0] = 0
    z_l = top_hat / (std_l + eps)
    if r_s > 0:
        sum_s = box_sum(a, r_s)
        sum_sw = box_sum(w, r_s)
        mean_s = sum_s / np.maximum(sum_sw, 1.0)
        dog = mean_s - mean_l
    else:
        dog = np.zeros_like(mean_l, dtype=np.float32)
    return top_hat.astype(np.float32), z_l.astype(np.float32), dog.astype(np.float32)


def build_bev_tiles_from_points(
    laz_paths: Iterable[Path],
    out_dir: Path,
    res_m: float,
    tile_size_px: int,
    tile_overlap_px: int,
    pad_m: float,
    win_large_m: float,
    win_small_m: float,
    tophat_clip_max: int,
    polygon_path: Optional[Path] = None,
    layer: str = "road_surface",
    chunk_points: int = 1_000_000,
    tile_keep_mask_area_m2: float = 50.0,
    eps: float = 1e-3,
) -> Dict[str, object]:
    import rasterio
    import rasterio.features
    import rasterio.transform
    import geopandas as gpd
    from shapely.geometry import box

    if polygon_path and polygon_path.exists():
        geom, bounds = load_polygon_bounds(polygon_path, layer=layer)
    else:
        geom = None
        bounds = _grid_bounds_from_points(laz_paths, pad_m)

    tiles_x, tiles_y, tile_size_m, minx, maxy = _tile_index(bounds, res_m, tile_size_px)
    if tile_overlap_px != 0:
        LOG.warning("tile_overlap_px not supported, ignored")

    tiles: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    total_points = 0
    used_points = 0
    for laz_path in laz_paths:
        for xs, ys, _zs, intens in read_laz_points(laz_path, chunk_points):
            total_points += int(xs.shape[0])
            tile_x = np.floor((xs - minx) / tile_size_m).astype(np.int64)
            tile_y = np.floor((maxy - ys) / tile_size_m).astype(np.int64)
            valid = (tile_x >= 0) & (tile_x < tiles_x) & (tile_y >= 0) & (tile_y < tiles_y)
            if not valid.any():
                continue
            xs = xs[valid]
            ys = ys[valid]
            intens = intens[valid].astype(np.uint16)
            tile_x = tile_x[valid]
            tile_y = tile_y[valid]
            used_points += int(xs.size)

            keys = np.unique(np.stack([tile_x, tile_y], axis=1), axis=0)
            for kx, ky in keys:
                mask = (tile_x == kx) & (tile_y == ky)
                if not mask.any():
                    continue
                xk = xs[mask]
                yk = ys[mask]
                ik = intens[mask]
                tile_minx = minx + kx * tile_size_m
                tile_maxy = maxy - ky * tile_size_m
                cols = np.floor((xk - tile_minx) / res_m).astype(np.int64)
                rows = np.floor((tile_maxy - yk) / res_m).astype(np.int64)
                valid_rc = (cols >= 0) & (cols < tile_size_px) & (rows >= 0) & (rows < tile_size_px)
                if not valid_rc.any():
                    continue
                cols = cols[valid_rc]
                rows = rows[valid_rc]
                ik = ik[valid_rc]
                arrs = _ensure_tile_arrays(tiles, (int(kx), int(ky)), tile_size_px)
                np.maximum.at(arrs["intensity_max"], (rows, cols), ik)
                np.add.at(arrs["density"], (rows, cols), 1)

    tiles_dir = bev_tiles_dir(out_dir, res_m)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    index_items = []
    kept = 0
    empty_tiles = 0
    mask_area_m2_total = 0.0
    points_in_mask_total = 0
    top_hat_vals: List[np.ndarray] = []
    for ky in range(tiles_y):
        for kx in range(tiles_x):
            arrs = tiles.get((kx, ky))
            if arrs is None:
                empty_tiles += 1
                continue
            intensity_max = arrs["intensity_max"]
            density = arrs["density"]
            mask = (density > 0).astype(np.uint8)
            if geom is not None:
                tile_minx = minx + kx * tile_size_m
                tile_maxx = tile_minx + tile_size_m
                tile_maxy = maxy - ky * tile_size_m
                tile_miny = tile_maxy - tile_size_m
                transform = rasterio.transform.from_origin(tile_minx, tile_maxy, res_m, res_m)
                poly_mask = rasterio.features.rasterize(
                    [(geom, 1)],
                    out_shape=(tile_size_px, tile_size_px),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False,
                )
                mask = (mask > 0) & (poly_mask > 0)
                mask = mask.astype(np.uint8)

            mask_area_m2 = float(mask.sum()) * res_m * res_m
            if mask_area_m2 < tile_keep_mask_area_m2:
                empty_tiles += 1
                continue
            top_hat, _z_l, _dog = _compute_top_hat(
                intensity_max, mask, win_large_m, win_small_m, res_m, eps
            )
            top_hat_u16 = np.clip(top_hat, 0, tophat_clip_max).astype(np.uint16)
            top_hat_u16[mask == 0] = 0
            density_u16 = np.clip(density, 0, 65535).astype(np.uint16)

            tile_minx = minx + kx * tile_size_m
            tile_maxx = tile_minx + tile_size_m
            tile_maxy = maxy - ky * tile_size_m
            tile_miny = tile_maxy - tile_size_m
            transform = rasterio.transform.from_origin(tile_minx, tile_maxy, res_m, res_m)
            out_name = f"tile_x{kx:04d}_y{ky:04d}.tif"
            out_path = tiles_dir / out_name
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=tile_size_px,
                width=tile_size_px,
                count=4,
                dtype="uint16",
                crs="EPSG:32632",
                transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(intensity_max.astype(np.uint16), 1)
                dst.set_band_description(1, "intensity_max")
                dst.write(density_u16, 2)
                dst.set_band_description(2, "density")
                dst.write(top_hat_u16, 3)
                dst.set_band_description(3, "top_hat_u16")
                dst.write(mask.astype(np.uint8), 4)
                dst.set_band_description(4, "road_mask")

            # preview
            try:
                import matplotlib.pyplot as plt

                preview = np.clip(top_hat_u16.astype(np.float32) / float(tophat_clip_max) * 255.0, 0, 255).astype(
                    np.uint8
                )
                plt.figure(figsize=(4, 4))
                plt.imshow(preview, cmap="gray")
                plt.axis("off")
                plt.savefig(tiles_dir / f"tile_x{kx:04d}_y{ky:04d}_preview.png", dpi=150, bbox_inches="tight", pad_inches=0)
                plt.close()
            except Exception:
                pass

            index_items.append(
                {
                    "tile_x": int(kx),
                    "tile_y": int(ky),
                    "path": str(out_path),
                    "bbox": [tile_minx, tile_miny, tile_maxx, tile_maxy],
                    "res_m": res_m,
                    "mask_area_m2": mask_area_m2,
                }
            )
            mask_area_m2_total += mask_area_m2
            points_in_mask_total += int(density[mask > 0].sum())
            vals = top_hat_u16[mask > 0]
            if vals.size > 0:
                top_hat_vals.append(vals.astype(np.float32))
            kept += 1

    index_path = bev_tiles_index_path(out_dir, res_m)
    if index_items:
        gdf = gpd.GeoDataFrame(
            index_items,
            geometry=[box(*item["bbox"]) for item in index_items],
            crs="EPSG:32632",
        )
        gdf.to_file(index_path, driver="GeoJSON")

    empty_ratio = float(empty_tiles) / float(max(tiles_x * tiles_y, 1))
    top_hat_p95 = None
    top_hat_p98 = None
    if top_hat_vals:
        merged = np.concatenate(top_hat_vals, axis=0)
        if merged.size > 0:
            top_hat_p95 = float(np.percentile(merged, 95))
            top_hat_p98 = float(np.percentile(merged, 98))
    return {
        "tiles_count": kept,
        "empty_tile_ratio": empty_ratio,
        "tiles_index": str(index_path) if index_items else "",
        "points_total_read": total_points,
        "points_used_in_extent": used_points,
        "points_in_mask": points_in_mask_total,
        "mask_area_m2": mask_area_m2_total,
        "top_hat_p95": top_hat_p95,
        "top_hat_p98": top_hat_p98,
        "tiles_res_m": res_m,
        "tile_size_px": tile_size_px,
        "tile_overlap_px": tile_overlap_px,
    }


__all__ = ["build_bev_tiles_from_points", "bev_tiles_dir", "bev_tiles_index_path"]
