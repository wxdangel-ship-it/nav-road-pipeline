from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("road_surface_evidence")


def bbox_check_utm32(bounds: Tuple[float, float, float, float]) -> Dict[str, object]:
    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy:
        return {"ok": False, "reason": "bbox_invalid_order"}
    if minx < 100000.0 or maxx > 900000.0:
        return {"ok": False, "reason": "bbox_easting_out_of_range"}
    if miny < 0.0 or maxy > 10000000.0:
        return {"ok": False, "reason": "bbox_northing_out_of_range"}
    return {"ok": True, "reason": "ok"}


def load_fusion_bounds(fusion_dir: Path) -> Tuple[float, float, float, float]:
    geojson = fusion_dir / "outputs" / "bbox_utm32.geojson"
    if geojson.exists():
        data = json.loads(geojson.read_text(encoding="utf-8"))
        coords = data["features"][0]["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (min(xs), min(ys), max(xs), max(ys))
    meta = fusion_dir / "outputs" / "fused_points_utm32.meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        bbox = m.get("bbox", {})
        return (
            float(bbox.get("xmin", 0.0)),
            float(bbox.get("ymin", 0.0)),
            float(bbox.get("xmax", 0.0)),
            float(bbox.get("ymax", 0.0)),
        )
    raise FileNotFoundError("fusion_bbox_not_found")


def compute_grid_shape(bounds: Tuple[float, float, float, float], grid_res: float, pad_m: float) -> Tuple[int, int, Tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = bounds
    minx -= pad_m
    miny -= pad_m
    maxx += pad_m
    maxy += pad_m
    width = int(math.ceil((maxx - minx) / grid_res))
    height = int(math.ceil((maxy - miny) / grid_res))
    if width <= 0 or height <= 0:
        raise RuntimeError("grid_invalid")
    return width, height, (minx, miny, maxx, maxy)


def read_laz_points(laz_paths: Iterable[Path], chunk_points: int):
    import laspy

    for laz_path in laz_paths:
        with laspy.open(laz_path) as reader:
            for chunk in reader.chunk_iterator(chunk_points):
                xs = np.asarray(chunk.x)
                ys = np.asarray(chunk.y)
                zs = np.asarray(chunk.z)
                intens = np.asarray(chunk.intensity)
                yield xs, ys, zs, intens


def build_reference_surface_from_points_near_traj(
    laz_paths: Iterable[Path],
    traj_line,
    traj_xy: np.ndarray,
    traj_z: np.ndarray,
    grid_bounds: Tuple[float, float, float, float],
    ref_grid_res: float,
    ref_local_radius_m: float,
    ref_q_low: float,
    ref_min_pts: int,
    ref_smooth_win_m: float,
    layer_reject_below_m: Optional[float],
    layer_reject_above_m: Optional[float],
    chunk_points: int,
) -> Dict[str, object]:
    from shapely.prepared import prep

    minx, miny, maxx, maxy = grid_bounds
    width = int(math.ceil((maxx - minx) / ref_grid_res))
    height = int(math.ceil((maxy - miny) / ref_grid_res))
    if width <= 0 or height <= 0:
        raise RuntimeError("ref_grid_invalid")

    ref_roi = traj_line.buffer(ref_local_radius_m)
    try:
        from shapely import contains as shp_contains
        from shapely import points as shp_points
        use_vector = True
    except Exception:
        from shapely.geometry import Point
        use_vector = False
    ref_prep = prep(ref_roi)

    k_keep = 64
    z_keep = np.full((height, width, k_keep), np.inf, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.uint32)
    max_val = np.full((height, width), -np.inf, dtype=np.float32)
    max_idx = np.zeros((height, width), dtype=np.int16)

    def _traj_index(x: float, y: float) -> Optional[int]:
        dx = traj_xy[:, 0] - x
        dy = traj_xy[:, 1] - y
        if dx.size == 0:
            return None
        idx = int(np.argmin(dx * dx + dy * dy))
        return idx

    used_points = 0
    for xs, ys, zs, _intens in read_laz_points(laz_paths, chunk_points):
        xs_f = xs
        ys_f = ys
        bbox_mask = (xs_f >= minx) & (xs_f <= maxx) & (ys_f >= miny) & (ys_f <= maxy)
        if not bbox_mask.any():
            continue
        xs_f = xs_f[bbox_mask]
        ys_f = ys_f[bbox_mask]
        zs_f = zs[bbox_mask]
        if use_vector:
            pts = shp_points(xs_f, ys_f)
            in_mask = shp_contains(ref_roi, pts)
            if not np.any(in_mask):
                continue
            xs_f = xs_f[in_mask]
            ys_f = ys_f[in_mask]
            zs_f = zs_f[in_mask]
            idx_iter = range(xs_f.shape[0])
            get_point = None
        else:
            idx_iter = range(xs_f.shape[0])
            get_point = Point
        for i in idx_iter:
            x = float(xs_f[i])
            y = float(ys_f[i])
            if not use_vector:
                if not ref_prep.contains(get_point(x, y)):
                    continue
            z = float(zs_f[i])
            if layer_reject_below_m is not None or layer_reject_above_m is not None:
                idx = _traj_index(x, y)
                if idx is not None:
                    tz = float(traj_z[idx])
                    if layer_reject_below_m is not None and z < (tz - layer_reject_below_m):
                        continue
                    if layer_reject_above_m is not None and z > (tz + layer_reject_above_m):
                        continue
            col = int(math.floor((x - minx) / ref_grid_res))
            row = int(math.floor((maxy - y) / ref_grid_res))
            if row < 0 or row >= height or col < 0 or col >= width:
                continue
            used_points += 1
            c = count[row, col]
            if c < k_keep:
                z_keep[row, col, c] = z
                count[row, col] = c + 1
                if z > max_val[row, col]:
                    max_val[row, col] = z
                    max_idx[row, col] = int(c)
            else:
                if z < max_val[row, col]:
                    idx_max = int(max_idx[row, col])
                    z_keep[row, col, idx_max] = z
                    # recompute max
                    vals = z_keep[row, col, :]
                    mi = int(np.argmax(vals))
                    max_idx[row, col] = mi
                    max_val[row, col] = float(vals[mi])

    z_ref = np.full((height, width), np.nan, dtype=np.float32)
    valid = count >= ref_min_pts
    if np.any(valid):
        for r in range(height):
            for c in range(width):
                if not valid[r, c]:
                    continue
                vals = z_keep[r, c, : int(count[r, c])]
                if vals.size == 0:
                    continue
                z_ref[r, c] = float(np.quantile(vals, ref_q_low))

    # smooth with mean filter
    win = max(1, int(round(ref_smooth_win_m / ref_grid_res)))
    pad = win
    z_pad = np.pad(z_ref, pad, mode="constant", constant_values=np.nan)
    valid_pad = np.isfinite(z_pad).astype(np.float32)
    z_pad[np.isnan(z_pad)] = 0.0
    csum = z_pad.cumsum(axis=0).cumsum(axis=1)
    vsum = valid_pad.cumsum(axis=0).cumsum(axis=1)
    z_smooth = np.full_like(z_ref, np.nan, dtype=np.float32)
    for r in range(height):
        r0 = r
        r1 = r + 2 * pad
        for c in range(width):
            c0 = c
            c1 = c + 2 * pad
            total = csum[r1, c1] - csum[r0, c1] - csum[r1, c0] + csum[r0, c0]
            cnt = vsum[r1, c1] - vsum[r0, c1] - vsum[r1, c0] + vsum[r0, c0]
            if cnt > 0:
                z_smooth[r, c] = float(total / cnt)

    # fill invalid by local neighbor
    filled = 0
    for r in range(height):
        for c in range(width):
            if np.isfinite(z_smooth[r, c]):
                continue
            r0 = max(0, r - pad)
            r1 = min(height, r + pad + 1)
            c0 = max(0, c - pad)
            c1 = min(width, c + pad + 1)
            window = z_smooth[r0:r1, c0:c1]
            if np.isfinite(window).any():
                z_smooth[r, c] = float(np.nanmedian(window))
                filled += 1

    valid_mask = np.isfinite(z_smooth)
    ref_valid_ratio = float(np.sum(valid_mask)) / float(z_smooth.size)
    ref_filled_ratio = float(filled) / float(z_smooth.size)
    return {
        "z_ref": z_smooth,
        "valid_mask": valid_mask.astype(np.uint8),
        "ref_points_count": count,
        "grid_bounds": grid_bounds,
        "grid_res": ref_grid_res,
        "used_points": used_points,
        "ref_valid_ratio": ref_valid_ratio,
        "ref_filled_ratio": ref_filled_ratio,
    }


def compute_candidate_stats_with_ref(
    laz_paths: Iterable[Path],
    z_ref: np.ndarray,
    valid_mask: np.ndarray,
    grid_bounds: Tuple[float, float, float, float],
    grid_res: float,
    dz_max: float,
    chunk_points: int,
) -> Dict[str, np.ndarray]:
    width = z_ref.shape[1]
    height = z_ref.shape[0]
    cand_count = np.zeros((height, width), dtype=np.uint32)
    cand_sum_z = np.zeros((height, width), dtype=np.float64)
    cand_sum_z2 = np.zeros((height, width), dtype=np.float64)
    cand_intensity_max = np.zeros((height, width), dtype=np.uint16)
    minx, miny, maxx, maxy = grid_bounds

    for xs, ys, zs, intens in read_laz_points(laz_paths, chunk_points):
        cols = np.floor((xs - minx) / grid_res).astype(np.int64)
        rows = np.floor((maxy - ys) / grid_res).astype(np.int64)
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if not valid.any():
            continue
        cols = cols[valid]
        rows = rows[valid]
        zvals = zs[valid].astype(np.float32)
        ivals = intens[valid].astype(np.uint16)
        ref = z_ref[rows, cols]
        vmask = valid_mask[rows, cols] > 0
        if not vmask.any():
            continue
        cols = cols[vmask]
        rows = rows[vmask]
        zvals = zvals[vmask]
        ivals = ivals[vmask]
        ref = ref[vmask]
        dz = zvals - ref
        cand = dz <= dz_max
        if not cand.any():
            continue
        cols = cols[cand]
        rows = rows[cand]
        zvals = zvals[cand]
        ivals = ivals[cand]
        np.add.at(cand_count, (rows, cols), 1)
        np.add.at(cand_sum_z, (rows, cols), zvals)
        np.add.at(cand_sum_z2, (rows, cols), zvals * zvals)
        np.maximum.at(cand_intensity_max, (rows, cols), ivals)
    return {
        "cand_count": cand_count,
        "cand_sum_z": cand_sum_z,
        "cand_sum_z2": cand_sum_z2,
        "cand_intensity_max": cand_intensity_max,
    }


def seed_connected_mask(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    from collections import deque

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    q = deque()
    seed_idx = np.argwhere((seed_mask > 0) & (mask > 0))
    for r, c in seed_idx:
        visited[r, c] = 1
        q.append((int(r), int(c)))
    if not q:
        return mask
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if visited[nr, nc] > 0:
                    continue
                if mask[nr, nc] == 0:
                    continue
                visited[nr, nc] = 1
                q.append((nr, nc))
    return visited


def binary_close(mask: np.ndarray, radius_cells: int, iterations: int = 1) -> np.ndarray:
    if radius_cells <= 0 or iterations <= 0:
        return mask
    try:
        from scipy.ndimage import binary_closing

        struct = np.ones((radius_cells * 2 + 1, radius_cells * 2 + 1), dtype=bool)
        out = mask.astype(bool)
        for _ in range(iterations):
            out = binary_closing(out, structure=struct)
        return out.astype(np.uint8)
    except Exception:
        out = mask.astype(bool)
        for _ in range(iterations):
            out = _binary_dilate(out, radius_cells)
            out = _binary_erode(out, radius_cells)
        return out.astype(np.uint8)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= padded[radius + dy : radius + dy + mask.shape[0], radius + dx : radius + dx + mask.shape[1]]
    return out


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=True)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out &= padded[radius + dy : radius + dy + mask.shape[0], radius + dx : radius + dx + mask.shape[1]]
    return out


def keep_components_intersecting_seed(
    mask: np.ndarray, seed_mask: np.ndarray, max_components: int = 5
) -> Tuple[np.ndarray, Dict[str, object]]:
    from collections import deque

    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    comp_id = 0
    comp_sizes: Dict[int, int] = {}
    comp_seed: Dict[int, bool] = {}
    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0 or labels[r, c] != 0:
                continue
            comp_id += 1
            q = deque()
            q.append((r, c))
            labels[r, c] = comp_id
            size = 0
            has_seed = False
            while q:
                rr, cc = q.popleft()
                size += 1
                if seed_mask[rr, cc] > 0:
                    has_seed = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            continue
                        if mask[nr, nc] == 0 or labels[nr, nc] != 0:
                            continue
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
            comp_sizes[comp_id] = size
            comp_seed[comp_id] = has_seed

    keep = [cid for cid, ok in comp_seed.items() if ok]
    keep_sorted = sorted(keep, key=lambda cid: comp_sizes.get(cid, 0), reverse=True)
    keep_sorted = keep_sorted[:max_components]
    out = np.isin(labels, keep_sorted).astype(np.uint8)
    kept_area = int(out.sum())
    return out, {"kept_component_ids": keep_sorted, "kept_area_cells": kept_area}


def _cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "cache_meta.json"


def _load_cache_meta(cache_dir: Path) -> Dict[str, object]:
    path = _cache_meta_path(cache_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache_meta(cache_dir: Path, meta: Dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_meta_path(cache_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _inputs_signature(laz_paths: Iterable[Path]) -> List[Dict[str, object]]:
    sig = []
    for p in laz_paths:
        stat = p.stat()
        sig.append({"path": str(p), "size": stat.st_size, "mtime": stat.st_mtime})
    return sig


def compute_z_min_grid(
    laz_paths: Iterable[Path],
    bounds: Tuple[float, float, float, float],
    grid_res: float,
    pad_m: float,
    cache_dir: Path,
    chunk_points: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    width, height, grid_bounds = compute_grid_shape(bounds, grid_res, pad_m)
    meta = {
        "grid_res": grid_res,
        "pad_m": pad_m,
        "bounds": list(bounds),
        "grid_bounds": list(grid_bounds),
        "inputs": _inputs_signature(laz_paths),
    }
    cached = _load_cache_meta(cache_dir)
    z_path = cache_dir / "z_min.npy"
    if cached.get("z_min") == meta and z_path.exists():
        LOG.info("reuse z_min cache")
        z_min = np.load(z_path)
        return z_min, meta

    z_min = np.full((height, width), np.inf, dtype=np.float32)
    minx, miny, maxx, maxy = grid_bounds
    for xs, ys, zs, _intens in read_laz_points(laz_paths, chunk_points):
        cols = np.floor((xs - minx) / grid_res).astype(np.int64)
        rows = np.floor((maxy - ys) / grid_res).astype(np.int64)
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if not valid.any():
            continue
        cols = cols[valid]
        rows = rows[valid]
        zvals = zs[valid].astype(np.float32)
        np.minimum.at(z_min, (rows, cols), zvals)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(z_path, z_min)
    cached["z_min"] = meta
    _save_cache_meta(cache_dir, cached)
    return z_min, meta


def compute_candidate_stats(
    laz_paths: Iterable[Path],
    z_min: np.ndarray,
    grid_bounds: Tuple[float, float, float, float],
    grid_res: float,
    dz_max: float,
    cache_dir: Path,
    chunk_points: int,
) -> Dict[str, np.ndarray]:
    width = z_min.shape[1]
    height = z_min.shape[0]
    meta = {
        "grid_res": grid_res,
        "dz_max": dz_max,
        "grid_bounds": list(grid_bounds),
        "inputs": _inputs_signature(laz_paths),
    }
    cached = _load_cache_meta(cache_dir)
    count_path = cache_dir / f"cand_count_dz{dz_max:.3f}.npy"
    sum_path = cache_dir / f"cand_sum_z_dz{dz_max:.3f}.npy"
    sum2_path = cache_dir / f"cand_sum_z2_dz{dz_max:.3f}.npy"
    imax_path = cache_dir / f"cand_intensity_max_dz{dz_max:.3f}.npy"
    if cached.get("cand_stats") == meta and all(p.exists() for p in [count_path, sum_path, sum2_path, imax_path]):
        LOG.info("reuse candidate stats cache")
        return {
            "cand_count": np.load(count_path),
            "cand_sum_z": np.load(sum_path),
            "cand_sum_z2": np.load(sum2_path),
            "cand_intensity_max": np.load(imax_path),
        }

    cand_count = np.zeros((height, width), dtype=np.uint32)
    cand_sum_z = np.zeros((height, width), dtype=np.float64)
    cand_sum_z2 = np.zeros((height, width), dtype=np.float64)
    cand_intensity_max = np.zeros((height, width), dtype=np.uint16)
    minx, miny, maxx, maxy = grid_bounds

    for xs, ys, zs, intens in read_laz_points(laz_paths, chunk_points):
        cols = np.floor((xs - minx) / grid_res).astype(np.int64)
        rows = np.floor((maxy - ys) / grid_res).astype(np.int64)
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if not valid.any():
            continue
        cols = cols[valid]
        rows = rows[valid]
        zvals = zs[valid].astype(np.float32)
        ivals = intens[valid].astype(np.uint16)
        z_min_vals = z_min[rows, cols]
        good = np.isfinite(z_min_vals)
        if not good.any():
            continue
        cols = cols[good]
        rows = rows[good]
        zvals = zvals[good]
        ivals = ivals[good]
        zmin = z_min_vals[good]
        dz = zvals - zmin
        cand = dz <= dz_max
        if not cand.any():
            continue
        cols = cols[cand]
        rows = rows[cand]
        zvals = zvals[cand]
        ivals = ivals[cand]
        np.add.at(cand_count, (rows, cols), 1)
        np.add.at(cand_sum_z, (rows, cols), zvals)
        np.add.at(cand_sum_z2, (rows, cols), zvals * zvals)
        np.maximum.at(cand_intensity_max, (rows, cols), ivals)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(count_path, cand_count)
    np.save(sum_path, cand_sum_z)
    np.save(sum2_path, cand_sum_z2)
    np.save(imax_path, cand_intensity_max)
    cached["cand_stats"] = meta
    _save_cache_meta(cache_dir, cached)
    return {
        "cand_count": cand_count,
        "cand_sum_z": cand_sum_z,
        "cand_sum_z2": cand_sum_z2,
        "cand_intensity_max": cand_intensity_max,
    }


def compute_slope(z_min: np.ndarray, grid_res: float) -> np.ndarray:
    z = z_min.copy()
    z[~np.isfinite(z)] = np.nan
    gx, gy = np.gradient(z, grid_res)
    slope = np.sqrt(gx * gx + gy * gy)
    slope[np.isnan(slope)] = np.inf
    return slope.astype(np.float32)


def mask_from_stats(
    z_min: np.ndarray,
    cand_count: np.ndarray,
    cand_sum_z: np.ndarray,
    cand_sum_z2: np.ndarray,
    slope: np.ndarray,
    grid_bounds: Tuple[float, float, float, float],
    grid_res: float,
    min_pts: int,
    rough_max: float,
    slope_max: float,
    island_min_m2: float,
    hole_max_m2: float,
    keep_mode: str = "largest",
) -> Tuple[np.ndarray, object, Dict[str, object]]:
    import rasterio.features
    from shapely.geometry import shape, Polygon
    from shapely.ops import unary_union

    count = cand_count.astype(np.float32)
    mean = np.zeros_like(count, dtype=np.float32)
    std = np.zeros_like(count, dtype=np.float32)
    valid = count > 0
    mean[valid] = (cand_sum_z[valid] / count[valid]).astype(np.float32)
    var = np.zeros_like(count, dtype=np.float32)
    var[valid] = (cand_sum_z2[valid] / count[valid] - mean[valid] ** 2).astype(np.float32)
    var[var < 0] = 0
    std[valid] = np.sqrt(var[valid]).astype(np.float32)

    mask = (
        (count >= float(min_pts))
        & (std <= rough_max)
        & (slope <= slope_max)
        & np.isfinite(z_min)
    )

    minx, miny, maxx, maxy = grid_bounds
    transform = rasterio.transform.from_origin(minx, maxy, grid_res, grid_res)
    shapes = list(rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform))
    polys = [shape(geom) for geom, val in shapes if val == 1]
    if not polys:
        raise RuntimeError("mask_empty")

    if keep_mode == "largest":
        polys = [max(polys, key=lambda g: g.area)]

    # filter islands
    polys = [p for p in polys if p.area >= island_min_m2]
    if not polys:
        raise RuntimeError("mask_empty_after_island_filter")

    def _drop_small_holes(poly: Polygon) -> Polygon:
        if not poly.interiors:
            return poly
        new_interiors = []
        for ring in poly.interiors:
            hole = Polygon(ring)
            if hole.area >= hole_max_m2:
                new_interiors.append(ring)
        return Polygon(poly.exterior, new_interiors)

    polys = [_drop_small_holes(p) for p in polys]
    merged = unary_union(polys)

    # rasterize back
    out_mask = rasterio.features.rasterize(
        [(merged, 1)],
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    # stats
    components = len(polys)
    hole_cnt = 0
    try:
        if merged.geom_type == "Polygon":
            hole_cnt = len(merged.interiors)
        elif merged.geom_type == "MultiPolygon":
            hole_cnt = sum(len(p.interiors) for p in merged.geoms)
    except Exception:
        hole_cnt = 0
    return out_mask, merged, {"components": components, "holes_cnt": hole_cnt}
