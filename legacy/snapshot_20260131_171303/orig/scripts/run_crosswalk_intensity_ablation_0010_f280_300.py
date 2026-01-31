
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from pipeline._io import load_yaml
from pipeline.calib.kitti360_world import transform_points_V_to_W
from pipeline.datasets.kitti360_io import load_kitti360_lidar_points
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import ensure_overwrite, now_ts, relpath, setup_logging, write_csv, write_gpkg_layer, write_json, write_text


LOG = logging.getLogger("crosswalk_intensity_ablation_0010_f280_300")

TRUTH_GPKG = Path(r"E:\Work\nav-road-pipeline\crosswalk_truth_utm32.gpkg")
RANSAC_SAMPLE_MAX = 20000

REQUIRED_KEYS = [
    "FRAME_START",
    "FRAME_END",
    "BUFFER_M",
    "FIT_RANGE_MIN_M",
    "FIT_RANGE_MAX_M",
    "FIT_Y_ABS_MAX_M",
    "FIT_Z_MAX_M",
    "CLASSIFY_DZ_M",
    "POST_GRID_RES_M",
    "POST_MIN_CELL_PTS",
    "POST_RESID_MAX_M",
    "GRID_RES_M",
    "INT_STAT",
    "BG_WIN_RADIUS_M",
    "SCORE_NORM_PCTL",
    "TOP_SCORE_PCTL",
    "TOP_INT_PCTL",
    "MIN_CELL_PTS",
    "OVERWRITE",
]


@dataclass
class BevBundle:
    p95: np.ndarray
    score: np.ndarray
    score_mask: np.ndarray
    transform: rasterio.Affine
    valid_mask: np.ndarray


def _load_cfg(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return dict(load_yaml(path) or {})


def _normalize_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    def _norm(v):
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v.keys())}
        if isinstance(v, list):
            return [_norm(x) for x in v]
        return v

    return _norm(cfg)


def _hash_cfg(cfg: Dict[str, object]) -> str:
    raw = json.dumps(_normalize_cfg(cfg), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _write_resolved(run_dir: Path, cfg: Dict[str, object]) -> str:
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return params_hash


def _auto_find_kitti_root(scans: List[str]) -> Optional[Path]:
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        scans.append(env_root)
        p = Path(env_root)
        if p.exists():
            return p
    for cand in [r"E:\KITTI360\KITTI-360", r"D:\KITTI360\KITTI-360", r"C:\KITTI360\KITTI-360"]:
        scans.append(cand)
        p = Path(cand)
        if p.exists():
            return p
    repo = Path(".").resolve()
    for base in [repo / "data", repo / "datasets"]:
        if not base.exists():
            continue
        for child in base.iterdir():
            scans.append(str(child))
            if child.is_dir() and ("KITTI-360" in child.name or "KITTI360" in child.name):
                return child
    return None


def _select_drive_0010(data_root: Path) -> str:
    drives_file = Path("configs/golden_drives.txt")
    if drives_file.exists():
        drives = [ln.strip() for ln in drives_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        raw_root = data_root / "data_3d_raw"
        drives = sorted([p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("2013_05_28_drive_")])
    for d in drives:
        if "_0010_" in d:
            return d
    raise RuntimeError("no_0010_drive_found")


def _frame_ids(start: int, end: int) -> List[str]:
    return [f"{i:010d}" for i in range(int(start), int(end) + 1)]


def _list_layers(path: Path) -> List[Dict[str, str]]:
    import pyogrio

    layers_raw = pyogrio.list_layers(str(path))
    rows = []
    for item in layers_raw:
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            gtype = str(item.get("geometry_type", ""))
        else:
            name = str(item[0]) if len(item) > 0 else ""
            gtype = str(item[1]) if len(item) > 1 else ""
        rows.append({"name": name, "geometry_type": gtype})
    return rows


def _pick_truth_layer(layers: List[Dict[str, str]]) -> Optional[str]:
    def is_poly(row: Dict[str, str]) -> bool:
        gt = str(row.get("geometry_type", "")).lower()
        return "polygon" in gt

    for row in layers:
        name = row["name"].lower()
        if ("crosswalk" in name or "truth" in name) and is_poly(row):
            return row["name"]
    for row in layers:
        if is_poly(row):
            return row["name"]
    return None


def _truth_geom(gdf: gpd.GeoDataFrame) -> Polygon:
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
    if geom.is_empty:
        raise RuntimeError("truth_geometry_empty")
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if not polys:
            raise RuntimeError("truth_geometry_no_polygon")
        return unary_union(polys)
    raise RuntimeError(f"truth_geometry_invalid:{geom.geom_type}")


def _mask_points_in_polygon(points_xy: np.ndarray, poly: object) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0,), dtype=bool)
    try:
        from shapely import vectorized as shp_vec

        return np.asarray(shp_vec.contains(poly, points_xy[:, 0], points_xy[:, 1]), dtype=bool)
    except Exception:
        pass
    prep_poly = prep(poly)
    if hasattr(prep_poly, "contains_xy"):
        try:
            mask = prep_poly.contains_xy(points_xy[:, 0], points_xy[:, 1])
            return np.asarray(mask, dtype=bool)
        except Exception:
            return np.array([bool(prep_poly.contains_xy(x, y)) for x, y in points_xy], dtype=bool)
    return np.array([bool(prep_poly.contains(Polygon([(x, y), (x + 0.01, y), (x, y + 0.01)]))) for x, y in points_xy], dtype=bool)


def _intensity_field(raw: np.ndarray) -> Optional[np.ndarray]:
    if raw.ndim == 2 and raw.shape[1] >= 4:
        return raw[:, 3]
    if raw.dtype.fields:
        for key in ("intensity", "reflectance"):
            if key in raw.dtype.fields:
                return raw[key]
    return None


def _map_intensity(raw: Optional[np.ndarray]) -> Tuple[np.ndarray, str, bool]:
    if raw is None:
        return np.zeros((0,), dtype=np.uint16), "missing", True
    if raw.size == 0:
        return raw.astype(np.uint16), "empty", False
    if raw.dtype.kind == "f":
        min_val = float(np.nanmin(raw))
        max_val = float(np.nanmax(raw))
        if min_val >= 0.0 and max_val <= 1.0:
            scaled = np.round(np.clip(raw, 0.0, 1.0) * 65535.0).astype(np.uint16)
            return scaled, "float01_to_uint16", False
        if min_val >= 0.0 and max_val <= 255.0:
            scaled = np.round(np.clip(raw, 0.0, 255.0) * 256.0).astype(np.uint16)
            return scaled, "float255_to_uint16", False
        return np.clip(raw, 0, 65535).astype(np.uint16), "float_unexpected_clipped", False
    if raw.dtype.kind in {"u", "i"}:
        max_val = float(np.max(raw))
        if max_val <= 255.0:
            return (raw.astype(np.uint16) * 256).astype(np.uint16), "uint8_to_uint16", False
        return np.clip(raw, 0, 65535).astype(np.uint16), "uint16_direct", False
    return np.zeros((raw.shape[0],), dtype=np.uint16), "unsupported_dtype_zeroed", True


def _intensity_stats(inten: np.ndarray) -> Dict[str, float]:
    if inten.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "nonzero_ratio": 0.0,
            "dynamic_range": 0.0,
        }
    vals = inten.astype(np.float64)
    p50 = float(np.percentile(vals, 50))
    p90 = float(np.percentile(vals, 90))
    p95 = float(np.percentile(vals, 95))
    p99 = float(np.percentile(vals, 99))
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "nonzero_ratio": float(np.mean(vals > 0)),
        "dynamic_range": float(p99 - p50),
    }


def _score_stats(score: np.ndarray, score_mask: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    valid = score[np.isfinite(score)]
    if valid.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "valid_ratio": 0.0, "mask_ratio": 0.0}
    return {
        "p50": float(np.percentile(valid, 50)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "valid_ratio": float(np.mean(valid_mask)),
        "mask_ratio": float(np.mean(score_mask[valid_mask])) if np.any(valid_mask) else 0.0,
    }


def _ransac_plane(
    points: np.ndarray, iters: int, dist_thresh: float, normal_min_z: float
) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
    if points.shape[0] < 3:
        return None, None, np.zeros((0,), dtype=bool)
    rng = np.random.default_rng(0)
    best_inliers = np.zeros((points.shape[0],), dtype=bool)
    best_count = 0
    for _ in range(int(iters)):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = float(np.linalg.norm(n))
        if norm < 1e-6:
            continue
        n = n / norm
        if abs(float(n[2])) < float(normal_min_z):
            continue
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        inliers = dist < float(dist_thresh)
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_count == 0:
        return None, None, np.zeros((points.shape[0],), dtype=bool)
    inlier_pts = points[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    _, _, vv = np.linalg.svd(inlier_pts - centroid)
    n = vv[-1, :]
    n = n / max(1e-6, float(np.linalg.norm(n)))
    if n[2] < 0:
        n = -n
    d = -float(np.dot(n, centroid))
    dist = np.abs(points @ n + d)
    inliers = dist < float(dist_thresh)
    return n, d, inliers


def _fit_plane_with_sampling(
    points: np.ndarray, iters: int, dist_thresh: float, normal_min_z: float, sample_max: int
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if points.size == 0:
        return None, None
    if points.shape[0] > int(sample_max):
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=int(sample_max), replace=False)
        sample = points[idx]
    else:
        sample = points
    n, d, _ = _ransac_plane(sample, iters, dist_thresh, normal_min_z)
    return n, d


def _post_clean(points_xyz: np.ndarray, res_m: float, min_cell_pts: int, resid_max: float) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    minx = float(np.min(points_xyz[:, 0]))
    miny = float(np.min(points_xyz[:, 1]))
    maxx = float(np.max(points_xyz[:, 0]))
    maxy = float(np.max(points_xyz[:, 1]))
    width = int(np.ceil((maxx - minx) / res_m)) + 1
    height = int(np.ceil((maxy - miny) / res_m)) + 1
    col = np.floor((points_xyz[:, 0] - minx) / res_m).astype(np.int64)
    row = np.floor((maxy - points_xyz[:, 1]) / res_m).astype(np.int64)
    valid = (col >= 0) & (row >= 0) & (col < width) & (row < height)
    col_v = col[valid]
    row_v = row[valid]
    z_v = points_xyz[valid, 2]
    lin = row_v * int(width) + col_v
    order = np.argsort(lin, kind="mergesort")
    lin_sorted = lin[order]
    z_sorted = z_v[order]
    uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    median = np.full((height, width), np.nan, dtype=np.float32)
    for u, s, c in zip(uniq, start, counts):
        if c < int(min_cell_pts):
            continue
        y = int(lin_sorted[s] // width)
        x = int(lin_sorted[s] % width)
        median[y, x] = float(np.median(z_sorted[s : s + c]))
    cell_m = median[row, col]
    good = np.isfinite(cell_m) & (np.abs(points_xyz[:, 2] - cell_m) <= float(resid_max))
    return points_xyz[good]


def _grid_spec(points_xy: np.ndarray, res_m: float) -> Tuple[float, float, float, int, int, rasterio.Affine]:
    minx = float(np.min(points_xy[:, 0]))
    miny = float(np.min(points_xy[:, 1]))
    maxx = float(np.max(points_xy[:, 0]))
    maxy = float(np.max(points_xy[:, 1]))
    width = int(np.ceil((maxx - minx) / res_m)) + 1
    height = int(np.ceil((maxy - miny) / res_m)) + 1
    maxy_aligned = miny + height * res_m
    transform = from_origin(minx, maxy_aligned, res_m, res_m)
    return minx, miny, maxy_aligned, width, height, transform


def _group_slices(lin_idx: np.ndarray):
    order = np.argsort(lin_idx, kind="mergesort")
    lin_sorted = lin_idx[order]
    uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    return order, uniq, start, counts


def _box_mean(values: np.ndarray, valid_mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.where(valid_mask, values, np.nan)
    vals = np.where(valid_mask, values, 0.0).astype(np.float64)
    counts = valid_mask.astype(np.int64)
    pad = radius_px
    vals_p = np.pad(vals, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    cnts_p = np.pad(counts, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    sat = np.cumsum(np.cumsum(vals_p, axis=0), axis=1)
    sat_c = np.cumsum(np.cumsum(cnts_p, axis=0), axis=1)
    sat = np.pad(sat, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    sat_c = np.pad(sat_c, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    h, w = values.shape
    y0 = np.arange(0, h)
    x0 = np.arange(0, w)
    y1 = y0 + 2 * pad
    x1 = x0 + 2 * pad
    sum_block = (
        sat[y1[:, None] + 1, x1[None, :] + 1]
        - sat[y0[:, None], x1[None, :] + 1]
        - sat[y1[:, None] + 1, x0[None, :]]
        + sat[y0[:, None], x0[None, :]]
    )
    cnt_block = (
        sat_c[y1[:, None] + 1, x1[None, :] + 1]
        - sat_c[y0[:, None], x1[None, :] + 1]
        - sat_c[y1[:, None] + 1, x0[None, :]]
        + sat_c[y0[:, None], x0[None, :]]
    )
    out = np.full((h, w), np.nan, dtype=np.float32)
    valid = cnt_block > 0
    out[valid] = (sum_block[valid] / cnt_block[valid]).astype(np.float32)
    return out

def _build_bev(points_xyz: np.ndarray, intensity: np.ndarray, cfg: Dict[str, object]) -> Optional[BevBundle]:
    if points_xyz.size == 0:
        return None
    res_m = float(cfg["GRID_RES_M"])
    minx, miny, maxy_aligned, width, height, transform = _grid_spec(points_xyz[:, :2], res_m)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    col = np.floor((x - minx) / res_m).astype(np.int64)
    row = np.floor((maxy_aligned - y) / res_m).astype(np.int64)
    valid = (col >= 0) & (row >= 0) & (col < width) & (row < height)
    col_v = col[valid]
    row_v = row[valid]
    int_v = intensity[valid].astype(np.float64)
    lin = row_v * int(width) + col_v
    order, uniq, start, counts = _group_slices(lin)
    col_s = col_v[order]
    row_s = row_v[order]
    int_s = int_v[order]
    min_cell = int(cfg["MIN_CELL_PTS"])
    pctl = 95 if str(cfg["INT_STAT"]).lower() == "p95" else 95
    p95 = np.full((height, width), np.nan, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)
    for u, s, c in zip(uniq, start, counts):
        if c < min_cell:
            continue
        cx = int(col_s[s])
        cy = int(row_s[s])
        sl = slice(int(s), int(s + c))
        p95[cy, cx] = float(np.percentile(int_s[sl], pctl))
        count[cy, cx] = int(c)
    valid_mask = (count >= min_cell) & np.isfinite(p95)
    radius_px = int(np.ceil(float(cfg["BG_WIN_RADIUS_M"]) / res_m))
    bg = _box_mean(p95, valid_mask, radius_px)
    score = np.where(valid_mask & np.isfinite(bg), np.maximum(0.0, p95 - bg), np.nan).astype(np.float32)
    score_valid = score[np.isfinite(score)]
    if score_valid.size == 0:
        score_norm = np.zeros_like(score, dtype=np.float32)
    else:
        denom = float(np.percentile(score_valid, float(cfg["SCORE_NORM_PCTL"])))
        denom = max(denom, 1e-6)
        score_norm = np.clip(score / denom, 0.0, 1.0).astype(np.float32)
    score_valid_norm = score_norm[np.isfinite(score_norm)]
    if score_valid_norm.size == 0:
        top_thr = 1.1
    else:
        top_thr = float(np.percentile(score_valid_norm, float(cfg["TOP_SCORE_PCTL"])))
    score_mask = (score_norm >= top_thr) & np.isfinite(score_norm)
    return BevBundle(p95=p95, score=score_norm, score_mask=score_mask, transform=transform, valid_mask=valid_mask)


def _write_raster(path: Path, arr: np.ndarray, transform: rasterio.Affine, epsg: int, nodata: Optional[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=str(arr.dtype),
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as ds:
        ds.write(arr, 1)


def _plot_bev_density(points_xy: np.ndarray, out_path: Path, title: str, res_m: float) -> None:
    if points_xy.size == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title)
        ax.set_axis_off()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return
    minx, miny = np.min(points_xy[:, 0]), np.min(points_xy[:, 1])
    maxx, maxy = np.max(points_xy[:, 0]), np.max(points_xy[:, 1])
    width = int(np.ceil((maxx - minx) / res_m)) + 1
    height = int(np.ceil((maxy - miny) / res_m)) + 1
    col = np.floor((points_xy[:, 0] - minx) / res_m).astype(np.int32)
    row = np.floor((maxy - points_xy[:, 1]) / res_m).astype(np.int32)
    valid = (col >= 0) & (row >= 0) & (col < width) & (row < height)
    col = col[valid]
    row = row[valid]
    img = np.zeros((height, width), dtype=np.float32)
    for r, c in zip(row, col):
        img[r, c] += 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="upper", cmap="inferno")
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_score(arr: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, origin="upper", cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_triptych(raw: np.ndarray, pre: np.ndarray, clean: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, arr, title in zip(axes, [raw, pre, clean], ["raw", "preclean", "clean"]):
        ax.imshow(arr, origin="upper", cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _high_intensity(points_xyz: np.ndarray, intensity: np.ndarray, pctl: float) -> Tuple[np.ndarray, np.ndarray]:
    if intensity.size == 0:
        return points_xyz[:0], intensity[:0]
    thr = float(np.percentile(intensity.astype(np.float64), pctl))
    mask = intensity >= thr
    return points_xyz[mask], intensity[mask]


def _points_by_score_mask(points_xyz: np.ndarray, intensity: np.ndarray, bev: BevBundle, res_m: float) -> Tuple[np.ndarray, np.ndarray]:
    minx, miny, maxy_aligned, width, height, _ = _grid_spec(points_xyz[:, :2], res_m)
    col = np.floor((points_xyz[:, 0] - minx) / res_m).astype(np.int64)
    row = np.floor((maxy_aligned - points_xyz[:, 1]) / res_m).astype(np.int64)
    valid = (col >= 0) & (row >= 0) & (col < width) & (row < height)
    mask = np.zeros((points_xyz.shape[0],), dtype=bool)
    idx = np.where(valid)[0]
    mask[idx] = bev.score_mask[row[valid], col[valid]]
    return points_xyz[mask], intensity[mask]


def main() -> None:
    cfg_path = Path("configs/lidar_crosswalk_intensity_ablation_0010_f280_300.yaml")
    cfg = _load_cfg(cfg_path)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required key {key} in {cfg_path}")

    run_id = now_ts()
    run_dir = Path("runs") / f"crosswalk_intensity_ablation_0010_f280_300_{run_id}"
    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    setup_logging(run_dir / "run.log")
    _write_resolved(run_dir, cfg)

    gis_dir = run_dir / "gis"
    pc_dir = run_dir / "pointcloud"
    ras_dir = run_dir / "rasters"
    tbl_dir = run_dir / "tables"
    img_dir = run_dir / "images"

    layers = _list_layers(TRUTH_GPKG)
    write_csv(tbl_dir / "layers.csv", layers, ["name", "geometry_type"])
    layer_name = _pick_truth_layer(layers)
    if layer_name is None:
        report = [
            "# Crosswalk intensity ablation 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: no_polygon_layer_in_truth_gpkg",
            f"- truth: {TRUTH_GPKG}",
            f"- layers_csv: {relpath(run_dir, tbl_dir / 'layers.csv')}",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    truth_gdf = gpd.read_file(TRUTH_GPKG, layer=layer_name)
    if truth_gdf.crs is None:
        truth_gdf = truth_gdf.set_crs(32632)
    if int(truth_gdf.crs.to_epsg() or 0) != 32632:
        truth_gdf = truth_gdf.to_crs(32632)
    truth_geom = _truth_geom(truth_gdf)
    truth_gdf = gpd.GeoDataFrame(geometry=[truth_geom], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "truth_selected_utm32.gpkg", "truth_selected", truth_gdf, 32632, [])
    truth_buf = truth_geom.buffer(float(cfg["BUFFER_M"]))
    buf_gdf = gpd.GeoDataFrame(geometry=[truth_buf], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "truth_buffer10_utm32.gpkg", "truth_buffer10", buf_gdf, 32632, [])

    scans: List[str] = []
    data_root = _auto_find_kitti_root(scans)
    if data_root is None:
        raise RuntimeError(f"data_root_not_found:scanned={scans}")
    drive_id = _select_drive_0010(data_root)
    frame_ids = _frame_ids(int(cfg["FRAME_START"]), int(cfg["FRAME_END"]))

    raw_points: List[np.ndarray] = []
    raw_intensity: List[np.ndarray] = []
    pre_points: List[np.ndarray] = []
    pre_intensity: List[np.ndarray] = []
    intensity_rule = "unknown"
    intensity_missing = False

    for fid in frame_ids:
        raw = load_kitti360_lidar_points(data_root, drive_id, fid)
        if raw.size == 0:
            continue
        intensity_raw = _intensity_field(raw)
        if intensity_raw is None:
            intensity_raw = np.zeros((raw.shape[0],), dtype=np.float32)
        else:
            intensity_raw = intensity_raw.astype(np.float32)
        mapped, rule, missing_flag = _map_intensity(intensity_raw)
        if intensity_rule == "unknown":
            intensity_rule = rule
        elif intensity_rule != rule:
            intensity_rule = "mixed"
        if missing_flag:
            intensity_missing = True

        pts_v = raw[:, :3]
        n_raw, d_raw = _fit_plane_with_sampling(pts_v, 200, 0.12, 0.90, RANSAC_SAMPLE_MAX)
        if n_raw is None:
            continue
        dist_raw = np.abs(pts_v @ n_raw + float(d_raw))
        inliers_raw = dist_raw <= 0.12
        ground_raw = pts_v[inliers_raw]
        inten_raw = mapped[inliers_raw]

        fit_mask = (
            (np.linalg.norm(pts_v[:, :2], axis=1) >= float(cfg["FIT_RANGE_MIN_M"]))
            & (np.linalg.norm(pts_v[:, :2], axis=1) <= float(cfg["FIT_RANGE_MAX_M"]))
            & (np.abs(pts_v[:, 1]) <= float(cfg["FIT_Y_ABS_MAX_M"]))
            & (pts_v[:, 2] <= float(cfg["FIT_Z_MAX_M"]))
        )
        pts_fit = pts_v[fit_mask]
        n_fit, d_fit = _fit_plane_with_sampling(pts_fit, 200, 0.12, 0.90, RANSAC_SAMPLE_MAX)
        if n_fit is None:
            continue
        dist = np.abs(pts_v @ n_fit + float(d_fit))
        inliers_pre = dist <= float(cfg["CLASSIFY_DZ_M"])
        ground_pre = pts_v[inliers_pre]
        inten_pre = mapped[inliers_pre]

        world_raw = transform_points_V_to_W(ground_raw, data_root, drive_id, fid)
        world_pre = transform_points_V_to_W(ground_pre, data_root, drive_id, fid)
        mask_raw = _mask_points_in_polygon(world_raw[:, :2], truth_buf)
        mask_pre = _mask_points_in_polygon(world_pre[:, :2], truth_buf)
        if mask_raw.any():
            raw_points.append(world_raw[mask_raw].astype(np.float32))
            raw_intensity.append(inten_raw[mask_raw].astype(np.uint16))
        if mask_pre.any():
            pre_points.append(world_pre[mask_pre].astype(np.float32))
            pre_intensity.append(inten_pre[mask_pre].astype(np.uint16))

    if not raw_points or not pre_points:
        decision = {"status": "FAIL", "reason": "no_points_in_roi"}
        write_json(run_dir / "decision.json", decision)
        report = [
            "# Crosswalk intensity ablation 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: no_points_in_roi",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    raw_pts = np.concatenate(raw_points, axis=0)
    raw_int = np.concatenate(raw_intensity, axis=0)
    pre_pts = np.concatenate(pre_points, axis=0)
    pre_int = np.concatenate(pre_intensity, axis=0)
    clean_pts = _post_clean(pre_pts, float(cfg["POST_GRID_RES_M"]), int(cfg["POST_MIN_CELL_PTS"]), float(cfg["POST_RESID_MAX_M"]))

    if clean_pts.size == 0:
        clean_pts = pre_pts[:0]
        clean_int = pre_int[:0]
    else:
        minx = float(np.min(pre_pts[:, 0]))
        miny = float(np.min(pre_pts[:, 1]))
        maxx = float(np.max(pre_pts[:, 0]))
        maxy = float(np.max(pre_pts[:, 1]))
        width = int(np.ceil((maxx - minx) / float(cfg["POST_GRID_RES_M"]))) + 1
        height = int(np.ceil((maxy - miny) / float(cfg["POST_GRID_RES_M"]))) + 1
        col = np.floor((pre_pts[:, 0] - minx) / float(cfg["POST_GRID_RES_M"])).astype(np.int64)
        row = np.floor((maxy - pre_pts[:, 1]) / float(cfg["POST_GRID_RES_M"])).astype(np.int64)
        valid = (col >= 0) & (row >= 0) & (col < width) & (row < height)
        col_v = col[valid]
        row_v = row[valid]
        z_v = pre_pts[valid, 2]
        lin = row_v * int(width) + col_v
        order = np.argsort(lin, kind="mergesort")
        lin_sorted = lin[order]
        z_sorted = z_v[order]
        uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
        median = np.full((height, width), np.nan, dtype=np.float32)
        for u, s, c in zip(uniq, start, counts):
            if c < int(cfg["POST_MIN_CELL_PTS"]):
                continue
            y = int(lin_sorted[s] // width)
            x = int(lin_sorted[s] % width)
            median[y, x] = float(np.median(z_sorted[s : s + c]))
        cell_m = median[row, col]
        good = np.isfinite(cell_m) & (np.abs(pre_pts[:, 2] - cell_m) <= float(cfg["POST_RESID_MAX_M"]))
        clean_pts = pre_pts[good]
        clean_int = pre_int[good]

    write_las(pc_dir / "roi_ground_raw_utm32.laz", raw_pts, raw_int, np.ones((raw_pts.shape[0],), dtype=np.uint8), 32632)
    write_las(pc_dir / "roi_ground_preclean_utm32.laz", pre_pts, pre_int, np.ones((pre_pts.shape[0],), dtype=np.uint8), 32632)
    write_las(pc_dir / "roi_ground_clean_utm32.laz", clean_pts, clean_int, np.ones((clean_pts.shape[0],), dtype=np.uint8), 32632)

    highI_raw_pts, highI_raw_int = _high_intensity(raw_pts, raw_int, float(cfg["TOP_INT_PCTL"]))
    highI_pre_pts, highI_pre_int = _high_intensity(pre_pts, pre_int, float(cfg["TOP_INT_PCTL"]))
    highI_clean_pts, highI_clean_int = _high_intensity(clean_pts, clean_int, float(cfg["TOP_INT_PCTL"]))
    write_las(pc_dir / "roi_highI_raw_utm32.laz", highI_raw_pts, highI_raw_int, np.ones((highI_raw_pts.shape[0],), dtype=np.uint8), 32632)
    write_las(pc_dir / "roi_highI_preclean_utm32.laz", highI_pre_pts, highI_pre_int, np.ones((highI_pre_pts.shape[0],), dtype=np.uint8), 32632)
    write_las(pc_dir / "roi_highI_clean_utm32.laz", highI_clean_pts, highI_clean_int, np.ones((highI_clean_pts.shape[0],), dtype=np.uint8), 32632)

    raw_bev = _build_bev(raw_pts, raw_int, cfg)
    pre_bev = _build_bev(pre_pts, pre_int, cfg)
    clean_bev = _build_bev(clean_pts, clean_int, cfg)

    if raw_bev is None or pre_bev is None or clean_bev is None:
        decision = {"status": "FAIL", "reason": "bev_empty"}
        write_json(run_dir / "decision.json", decision)
        report = [
            "# Crosswalk intensity ablation 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: bev_empty",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    _write_raster(ras_dir / "raw_intensity_p95_utm32.tif", raw_bev.p95.astype(np.float32), raw_bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "raw_score_utm32.tif", raw_bev.score.astype(np.float32), raw_bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "raw_score_mask_utm32.tif", raw_bev.score_mask.astype(np.uint8), raw_bev.transform, 32632, 0)
    _write_raster(ras_dir / "preclean_intensity_p95_utm32.tif", pre_bev.p95.astype(np.float32), pre_bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "preclean_score_utm32.tif", pre_bev.score.astype(np.float32), pre_bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "preclean_score_mask_utm32.tif", pre_bev.score_mask.astype(np.uint8), pre_bev.transform, 32632, 0)
    _write_raster(ras_dir / "clean_intensity_p95_utm32.tif", clean_bev.p95.astype(np.float32), clean_bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "clean_score_utm32.tif", clean_bev.score.astype(np.float32), clean_bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "clean_score_mask_utm32.tif", clean_bev.score_mask.astype(np.uint8), clean_bev.transform, 32632, 0)

    img_dir.mkdir(parents=True, exist_ok=True)
    _plot_bev_density(raw_pts[:, :2], img_dir / "bev_density_raw.png", "density raw", 0.2)
    _plot_bev_density(pre_pts[:, :2], img_dir / "bev_density_preclean.png", "density preclean", 0.2)
    _plot_bev_density(clean_pts[:, :2], img_dir / "bev_density_clean.png", "density clean", 0.2)
    _plot_bev_density(highI_raw_pts[:, :2], img_dir / "bev_highI_raw.png", "highI raw", 0.2)
    _plot_bev_density(highI_pre_pts[:, :2], img_dir / "bev_highI_preclean.png", "highI preclean", 0.2)
    _plot_bev_density(highI_clean_pts[:, :2], img_dir / "bev_highI_clean.png", "highI clean", 0.2)
    _plot_score(raw_bev.score, img_dir / "bev_score_raw.png", "score raw")
    _plot_score(pre_bev.score, img_dir / "bev_score_preclean.png", "score preclean")
    _plot_score(clean_bev.score, img_dir / "bev_score_clean.png", "score clean")
    _plot_triptych(raw_bev.score, pre_bev.score, clean_bev.score, img_dir / "compare_triptych.png")

    raw_score_pts, raw_score_int = _points_by_score_mask(raw_pts, raw_int, raw_bev, float(cfg["GRID_RES_M"]))
    pre_score_pts, pre_score_int = _points_by_score_mask(pre_pts, pre_int, pre_bev, float(cfg["GRID_RES_M"]))
    clean_score_pts, clean_score_int = _points_by_score_mask(clean_pts, clean_int, clean_bev, float(cfg["GRID_RES_M"]))
    write_las(pc_dir / "roi_highScore_raw_utm32.laz", raw_score_pts, raw_score_int, np.ones((raw_score_pts.shape[0],), dtype=np.uint8), 32632)
    write_las(pc_dir / "roi_highScore_preclean_utm32.laz", pre_score_pts, pre_score_int, np.ones((pre_score_pts.shape[0],), dtype=np.uint8), 32632)
    write_las(pc_dir / "roi_highScore_clean_utm32.laz", clean_score_pts, clean_score_int, np.ones((clean_score_pts.shape[0],), dtype=np.uint8), 32632)

    raw_int_stats = _intensity_stats(raw_int)
    pre_int_stats = _intensity_stats(pre_int)
    clean_int_stats = _intensity_stats(clean_int)
    raw_score_stats = _score_stats(raw_bev.score, raw_bev.score_mask, raw_bev.valid_mask)
    pre_score_stats = _score_stats(pre_bev.score, pre_bev.score_mask, pre_bev.valid_mask)
    clean_score_stats = _score_stats(clean_bev.score, clean_bev.score_mask, clean_bev.valid_mask)
    write_json(tbl_dir / "intensity_stats_raw.json", raw_int_stats)
    write_json(tbl_dir / "intensity_stats_preclean.json", pre_int_stats)
    write_json(tbl_dir / "intensity_stats_clean.json", clean_int_stats)
    write_json(tbl_dir / "score_stats_raw.json", raw_score_stats)
    write_json(tbl_dir / "score_stats_preclean.json", pre_score_stats)
    write_json(tbl_dir / "score_stats_clean.json", clean_score_stats)

    decision = {
        "status": "PASS",
        "raw_points": int(raw_pts.shape[0]),
        "preclean_points": int(pre_pts.shape[0]),
        "clean_points": int(clean_pts.shape[0]),
        "intensity_rule": intensity_rule,
        "intensity_missing": bool(intensity_missing),
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Crosswalk intensity ablation 0010 f280-300",
        "",
        f"- input_drive: {drive_id}",
        f"- frames: {cfg['FRAME_START']}..{cfg['FRAME_END']}",
        f"- intensity_rule: {intensity_rule}",
        f"- intensity_missing: {bool(intensity_missing)}",
        f"- raw_points: {raw_pts.shape[0]}",
        f"- preclean_points: {pre_pts.shape[0]}",
        f"- clean_points: {clean_pts.shape[0]}",
        f"- highI_counts: raw={highI_raw_pts.shape[0]}, preclean={highI_pre_pts.shape[0]}, clean={highI_clean_pts.shape[0]}",
        f"- highScore_counts: raw={raw_score_pts.shape[0]}, preclean={pre_score_pts.shape[0]}, clean={clean_score_pts.shape[0]}",
        f"- score_valid_ratio: raw={raw_score_stats['valid_ratio']:.4f}, preclean={pre_score_stats['valid_ratio']:.4f}, clean={clean_score_stats['valid_ratio']:.4f}",
        f"- score_mask_ratio: raw={raw_score_stats['mask_ratio']:.4f}, preclean={pre_score_stats['mask_ratio']:.4f}, clean={clean_score_stats['mask_ratio']:.4f}",
        f"- intensity_dyn_range(p99-p50): raw={raw_int_stats['dynamic_range']:.1f}, preclean={pre_int_stats['dynamic_range']:.1f}, clean={clean_int_stats['dynamic_range']:.1f}",
        "",
        "## outputs",
        f"- {relpath(run_dir, pc_dir / 'roi_ground_raw_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_ground_preclean_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_ground_clean_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_highI_raw_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_highI_preclean_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_highI_clean_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_highScore_raw_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_highScore_preclean_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_highScore_clean_utm32.laz')}",
        f"- {relpath(run_dir, img_dir / 'compare_triptych.png')}",
    ]
    write_text(run_dir / "report.md", "\n".join(report) + "\n")


if __name__ == "__main__":
    main()
