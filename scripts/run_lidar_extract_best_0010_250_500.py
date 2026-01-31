
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPoint, Polygon, shape
from shapely.prepared import prep

from pipeline.calib.io_kitti360_calib import load_kitti360_calib_bundle
from pipeline.calib.kitti360_projection import project_velo_to_image, project_world_to_image_pose
from pipeline.calib.proj_sanity import validate_depth, validate_in_image_ratio, validate_uv_spread
from pipeline.datasets.kitti360_io import _find_oxts_dir, _find_velodyne_dir, load_kitti360_lidar_points, load_kitti360_pose
from pipeline.lidar_semantic.accum_points_world import _voxel_downsample
from pipeline.calib.kitti360_world import transform_points_V_to_W
from pipeline.lidar_semantic.build_rasters import build_rasters
from pipeline.lidar_semantic.export_pointcloud import write_las
from pipeline.post.morph_close import close_candidates_in_corridor
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    now_ts,
    setup_logging,
    validate_output_crs,
    write_csv,
    write_gpkg_layer,
    write_json,
    write_text,
)


REQUIRED_KEYS = [
    "DRIVE_MATCH",
    "FRAME_START",
    "FRAME_END",
    "STRIDE",
    "ROI_BUFFER_M",
    "VOXEL_SIZE_M",
    "RASTER_RES_M",
    "TIME_BUDGET_H",
    "OVERWRITE",
    "SAMPLE_OVERLAY_FRAMES",
    "OUTPUT_IMAGE_CAM",
    "TARGET_EPSG",
    "KITTI_ROOT",
    "INTENSITY_ENABLE",
    "INTENSITY_AUTO_SCALE",
    "INTENSITY_FAILFAST_NONZERO_MIN",
    "INTENSITY_MIN_DYNAMIC_RANGE",
    "INTENSITY_PCTL_FOR_MARKING",
    "INTENSITY_BG_WIN_RADIUS_M",
    "MARKING_AREA_RATIO_TARGET",
    "GEOM_MARKINGS_ENABLE",
    "GEOM_MARKINGS_RES_M",
    "GEOM_RESIDUAL_TOP_PCTL",
    "GEOM_MARKING_AREA_RATIO_TARGET",
    "CROSSWALK_ENABLE",
    "CROSSWALK_CLUSTER_MIN_STRIPES",
    "CROSSWALK_ORI_TOL_DEG",
    "CROSSWALK_GAP_MAX_M",
    "CROSSWALK_MERGE_BUF_M",
    "CROSSWALK_W_RANGE_M",
    "CROSSWALK_L_RANGE_M",
    "CROSSWALK_AREA_RANGE_M2",
    "STRIPE_W_RANGE_M",
    "STRIPE_L_RANGE_M",
    "STRIPE_AREA_RANGE_M2",
    "ROAD_ROUGHNESS_MAX_M",
    "ROAD_SLOPE_MAX",
    "ROAD_GROW_FROM_TRAJ_BUF_M",
    "ROAD_MAX_EXPAND_BUF_M",
    "ROAD_MIN_COMPONENT_AREA_M2",
    "GROUND_RANSAC_DZ",
    "GROUND_NORMAL_DOT_Z_MIN",
    "GROUND_PER_FRAME",
]


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _normalize(cfg: Dict[str, object]) -> Dict[str, object]:
    def _norm(v):
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v.keys())}
        if isinstance(v, list):
            return [_norm(x) for x in v]
        return v

    return _norm(cfg)


def _hash_cfg(cfg: Dict[str, object]) -> str:
    raw = json.dumps(_normalize(cfg), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _write_resolved(run_dir: Path, cfg: Dict[str, object]) -> str:
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return params_hash


def _write_sidecar(path: Path, params_hash: str, extra: Optional[Dict[str, object]] = None) -> None:
    payload = {"params_hash": params_hash}
    if extra:
        payload.update(extra)
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


class _TimeBudgetExceeded(RuntimeError):
    pass


def _resolve_config(base: Dict[str, object], run_dir: Path) -> Tuple[Dict[str, object], str]:
    cfg = dict(base)
    defaults = {
        "DRIVE_MATCH": "_0010_",
        "FRAME_START": "auto",
        "FRAME_END": "auto",
        "STRIDE": 1,
        "ROI_BUFFER_M": 30,
        "VOXEL_SIZE_M": 0.10,
        "RASTER_RES_M": 0.30,
        "TIME_BUDGET_H": 6,
        "OVERWRITE": True,
        "SAMPLE_OVERLAY_FRAMES": [250, 341, 500],
        "OUTPUT_IMAGE_CAM": "image_00",
        "TARGET_EPSG": 32632,
        "KITTI_ROOT": "",
        "INTENSITY_ENABLE": True,
        "INTENSITY_AUTO_SCALE": True,
        "INTENSITY_FAILFAST_NONZERO_MIN": 0.05,
        "INTENSITY_MIN_DYNAMIC_RANGE": 2000,
        "INTENSITY_PCTL_FOR_MARKING": 98.5,
        "INTENSITY_BG_WIN_RADIUS_M": 3.0,
        "MARKING_AREA_RATIO_TARGET": [0.005, 0.04],
        "GEOM_MARKINGS_ENABLE": True,
        "GEOM_MARKINGS_RES_M": 0.10,
        "GEOM_RESIDUAL_TOP_PCTL": 99.5,
        "GEOM_MARKING_AREA_RATIO_TARGET": [0.002, 0.02],
        "CROSSWALK_ENABLE": True,
        "CROSSWALK_CLUSTER_MIN_STRIPES": 6,
        "CROSSWALK_ORI_TOL_DEG": 12,
        "CROSSWALK_GAP_MAX_M": 2.2,
        "CROSSWALK_MERGE_BUF_M": 0.6,
        "CROSSWALK_W_RANGE_M": [2.5, 10.0],
        "CROSSWALK_L_RANGE_M": [3.0, 28.0],
        "CROSSWALK_AREA_RANGE_M2": [10, 350],
        "STRIPE_W_RANGE_M": [0.12, 0.80],
        "STRIPE_L_RANGE_M": [0.80, 12.0],
        "STRIPE_AREA_RANGE_M2": [0.15, 20.0],
        "ROAD_ROUGHNESS_MAX_M": 0.04,
        "ROAD_SLOPE_MAX": 0.15,
        "ROAD_GROW_FROM_TRAJ_BUF_M": 6.0,
        "ROAD_MAX_EXPAND_BUF_M": 18.0,
        "ROAD_MIN_COMPONENT_AREA_M2": 50.0,
        "GROUND_RANSAC_DZ": 0.12,
        "GROUND_NORMAL_DOT_Z_MIN": 0.90,
        "GROUND_PER_FRAME": True,
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")
    params_hash = _write_resolved(run_dir, cfg)
    return cfg, params_hash


def _auto_find_kitti_root(cfg: Dict[str, object], scans: List[str]) -> Optional[Path]:
    cfg_root = str(cfg.get("KITTI_ROOT") or "").strip()
    if cfg_root:
        scans.append(cfg_root)
        p = Path(cfg_root)
        if p.exists():
            return p
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        scans.append(env_root)
        p = Path(env_root)
        if p.exists():
            return p
    for cand in [r"E:\\KITTI360\\KITTI-360", r"D:\\KITTI360\\KITTI-360", r"C:\\KITTI360\\KITTI-360"]:
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


def _select_drive(data_root: Path, match: str) -> str:
    drives_file = Path("configs/golden_drives.txt")
    if drives_file.exists():
        drives = [ln.strip() for ln in drives_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        raw_root = data_root / "data_3d_raw"
        drives = sorted([p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("2013_05_28_drive_")])
    for d in drives:
        if match in d:
            return d
    raise RuntimeError("no_drive_match")


def _frame_range_from_velodyne(velodyne_dir: Path) -> Tuple[int, int]:
    files = sorted(velodyne_dir.glob("*.bin"))
    ids = []
    for f in files:
        if f.stem.isdigit():
            ids.append(int(f.stem))
    if not ids:
        raise RuntimeError("empty_velodyne_frames")
    return min(ids), max(ids)


def _sample_frames(start: int, end: int, count: int) -> List[str]:
    if count <= 1:
        return [f"{start:010d}"]
    idx = np.linspace(start, end, count)
    frames = sorted({int(round(v)) for v in idx})
    return [f"{i:010d}" for i in frames]


def _build_traj_line(data_root: Path, drive_id: str, stride: int) -> LineString:
    oxts_dir = _find_oxts_dir(data_root, drive_id)
    frames = sorted(oxts_dir.glob("*.txt"))
    if stride > 1:
        frames = frames[::stride]
    points: List[Tuple[float, float]] = []
    for f in frames:
        frame_id = f.stem
        x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
        points.append((x, y))
    if len(points) < 2:
        return LineString(points or [(0.0, 0.0), (1.0, 1.0)])
    return LineString(points)


def _clip_to_roi(points_xyz: np.ndarray, intensity: np.ndarray, roi_geom: object) -> Tuple[np.ndarray, np.ndarray]:
    if points_xyz.size == 0:
        return points_xyz, intensity
    minx, miny, maxx, maxy = roi_geom.bounds
    bbox_mask = (
        (points_xyz[:, 0] >= minx)
        & (points_xyz[:, 0] <= maxx)
        & (points_xyz[:, 1] >= miny)
        & (points_xyz[:, 1] <= maxy)
    )
    pts = points_xyz[bbox_mask]
    inten = intensity[bbox_mask]
    if pts.size == 0:
        return pts, inten
    roi_prep = prep(roi_geom)
    if hasattr(roi_prep, "contains_xy"):
        inside = np.array([bool(roi_prep.contains_xy(x, y)) for x, y in pts[:, :2]], dtype=bool)
    else:
        inside = np.array([bool(roi_prep.contains(Polygon([(x, y), (x + 0.01, y), (x, y + 0.01)]))) for x, y in pts[:, :2]], dtype=bool)
    return pts[inside], inten[inside]


def _ransac_plane(points: np.ndarray, iters: int, dist_thresh: float, normal_min_z: float) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
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
    uu, ss, vv = np.linalg.svd(inlier_pts - centroid)
    n = vv[-1, :]
    n = n / max(1e-6, float(np.linalg.norm(n)))
    if n[2] < 0:
        n = -n
    d = -float(np.dot(n, centroid))
    dist = np.abs(points @ n + d)
    inliers = dist < float(dist_thresh)
    return n, d, inliers


def _write_raster(
    path: Path,
    arr: np.ndarray,
    transform: rasterio.Affine,
    epsg: int,
    warnings: List[str],
    nodata: Optional[float] = None,
) -> None:
    validate_output_crs(path, epsg, None, warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = arr.shape
    dtype = str(arr.dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as ds:
        ds.write(arr, 1)


def _box_mean(values: np.ndarray, valid: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        out = values.copy()
        out[~valid] = np.nan
        return out
    h, w = values.shape
    pad = radius_cells
    val = np.where(valid, values, 0.0).astype(np.float32)
    cnt = np.where(valid, 1.0, 0.0).astype(np.float32)
    val_p = np.pad(val, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    cnt_p = np.pad(cnt, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    val_ii = np.pad(val_p, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(axis=0).cumsum(axis=1)
    cnt_ii = np.pad(cnt_p, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(axis=0).cumsum(axis=1)
    out = np.zeros_like(values, dtype=np.float32)
    for y in range(h):
        y0 = y
        y1 = y + 2 * pad
        for x in range(w):
            x0 = x
            x1 = x + 2 * pad
            s = val_ii[y1 + 1, x1 + 1] - val_ii[y0, x1 + 1] - val_ii[y1 + 1, x0] + val_ii[y0, x0]
            c = cnt_ii[y1 + 1, x1 + 1] - cnt_ii[y0, x1 + 1] - cnt_ii[y1 + 1, x0] + cnt_ii[y0, x0]
            out[y, x] = float(s) / max(float(c), 1.0)
    out[~valid] = np.nan
    return out


def _threshold_for_area_ratio(score: np.ndarray, road_mask: np.ndarray, target_range: Tuple[float, float]) -> float:
    vals = score[road_mask.astype(bool) & np.isfinite(score)]
    if vals.size == 0:
        return float("inf")
    target_min, target_max = float(target_range[0]), float(target_range[1])
    target = float((target_min + target_max) * 0.5)
    thr = float(np.percentile(vals, 100.0 * (1.0 - target)))
    return thr


def _mask_to_polygons(mask: np.ndarray, transform: rasterio.Affine, min_area_m2: float) -> gpd.GeoDataFrame:
    geoms = []
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=transform):
        if int(val) != 1:
            continue
        poly = shape(geom)
        if poly.is_empty or float(poly.area) < min_area_m2:
            continue
        geoms.append(poly)
    if not geoms:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    return gpd.GeoDataFrame([{"geometry": g} for g in geoms], geometry="geometry", crs="EPSG:32632")


def _overlay_density(img: np.ndarray, u: np.ndarray, v: np.ndarray, in_img: np.ndarray, out_path: Path, title: str, params_hash: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = img.shape[0], img.shape[1]
    ui = u[in_img].astype(np.int32)
    vi = v[in_img].astype(np.int32)
    counts = np.zeros((h, w), dtype=np.int32)
    if ui.size > 0:
        np.add.at(counts, (vi, ui), 1)
    heat = np.log1p(counts).astype(np.float32)
    vmax = float(np.max(heat)) if heat.size else 0.0
    if vmax > 0:
        heat = heat / vmax
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    ax.imshow(heat, cmap="inferno", alpha=0.5, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_quicklook_bev(
    road_mask: np.ndarray,
    marking_mask: np.ndarray,
    out_path: Path,
    params_hash: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(road_mask.astype(np.float32), cmap="gray", alpha=0.7)
    ax.imshow(marking_mask.astype(np.float32), cmap="inferno", alpha=0.6)
    ax.set_title("quicklook_bev")
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_image_montage(paths: List[Path], out_path: Path, params_hash: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs = []
    labels = []
    for p in paths:
        if not p.exists():
            continue
        imgs.append(plt.imread(str(p)))
        labels.append(p.name)
    if not imgs:
        return
    cols = min(3, len(imgs))
    rows = int(math.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), dpi=150)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if idx >= len(imgs):
                continue
            ax.imshow(imgs[idx])
            ax.set_title(labels[idx], fontsize=6)
            idx += 1
    fig.suptitle(f"quicklook_image_montage | params_hash={params_hash}", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_roaddebug_plot(
    bundle: object,
    roi_geom: object,
    seed_geom: object,
    road_mask: np.ndarray,
    out_path: Path,
    params_hash: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = bundle.density_all if hasattr(bundle, "density_all") else bundle.density
    base = np.nan_to_num(base, nan=0.0)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(base, cmap="gray")
    roi_poly = roi_geom
    seed_poly = seed_geom
    for geom, color, label in [
        (roi_poly, "cyan", "roi"),
        (seed_poly, "yellow", "seed"),
    ]:
        if geom is None or geom.is_empty:
            continue
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=0.8, label=label)
    if np.any(road_mask):
        union = _mask_to_union(road_mask, bundle.transform)
        if union is not None and not union.is_empty:
            if union.geom_type == "Polygon":
                polys = [union]
            else:
                polys = list(union.geoms)
            for poly in polys:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="red", linewidth=0.8)
    ax.set_title("road_debug")
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _compute_slope(height: np.ndarray, res_m: float) -> np.ndarray:
    if height.size == 0:
        return height.copy()
    filled = np.nan_to_num(height, nan=np.nanmedian(height))
    gy, gx = np.gradient(filled, float(res_m), float(res_m))
    slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    slope[~np.isfinite(height)] = np.nan
    return slope


def _rasterize_geom(geom: object, transform: rasterio.Affine, shape_hw: Tuple[int, int]) -> np.ndarray:
    mask = features.rasterize([(geom, 1)], out_shape=shape_hw, transform=transform, fill=0, dtype="uint8")
    return mask.astype(bool)


def _mask_to_union(mask: np.ndarray, transform: rasterio.Affine) -> object:
    polys = []
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=transform):
        if int(val) != 1:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        polys.append(poly)
    if not polys:
        return None
    return gpd.GeoSeries(polys).unary_union


def _mask_close_by_buffer(mask: np.ndarray, transform: rasterio.Affine, radius_m: float) -> np.ndarray:
    if radius_m <= 0:
        return mask
    union = _mask_to_union(mask, transform)
    if union is None:
        return mask
    closed = union.buffer(radius_m).buffer(-radius_m)
    if closed.is_empty:
        return mask
    out = features.rasterize([(closed, 1)], out_shape=mask.shape, transform=transform, fill=0, dtype="uint8")
    return out.astype(bool)


def _road_from_thresholds(
    bundle: object,
    roi_mask: np.ndarray,
    seed_mask: np.ndarray,
    rough_th: float,
    slope_th: float,
    dens_th: float,
    close_m: float,
    max_expand_geom: object,
) -> Tuple[np.ndarray, Dict[str, float]]:
    valid = np.isfinite(bundle.height_p10)
    density_use = bundle.density_use if hasattr(bundle, "density_use") else bundle.density
    mask = (
        roi_mask
        & valid
        & (bundle.roughness <= rough_th)
        & (bundle.slope <= slope_th)
        & (density_use >= dens_th)
    )
    mask = _mask_close_by_buffer(mask, bundle.transform, close_m)
    grow_seed = mask & seed_mask
    if np.any(grow_seed):
        union = _mask_to_union(mask, bundle.transform)
        if union is not None:
            union = union.intersection(max_expand_geom)
            if not union.is_empty:
                mask = features.rasterize([(union, 1)], out_shape=mask.shape, transform=bundle.transform, fill=0, dtype="uint8").astype(bool)
    stats = {
        "road_cells": int(np.sum(mask)),
    }
    return mask, stats


def _postcheck(run_dir: Path, params_hash: str, sample_path: Path) -> None:
    report_path = run_dir / "report.md"
    decision_path = run_dir / "decision.json"
    if report_path.exists():
        txt = report_path.read_text(encoding="utf-8")
        if params_hash not in txt:
            raise RuntimeError("postcheck_report_hash_mismatch")
    if decision_path.exists():
        data = json.loads(decision_path.read_text(encoding="utf-8"))
        if data.get("params_hash") != params_hash:
            raise RuntimeError("postcheck_decision_hash_mismatch")
    sidecar = sample_path.with_suffix(sample_path.suffix + ".json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if data.get("params_hash") != params_hash:
            raise RuntimeError("postcheck_output_hash_mismatch")

def main() -> int:
    base_cfg = _load_yaml(Path("configs/lidar_extract_best_0010_250_500.yaml"))
    run_dir = Path("runs") / f"lidar_extract_best_0010_250_500_{now_ts()}"
    if bool(base_cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    t_start = time.perf_counter()
    stage_times: Dict[str, float] = {}
    cfg, params_hash = _resolve_config(base_cfg, run_dir)
    crs_warnings: List[str] = []

    scans: List[str] = []
    data_root = _auto_find_kitti_root(cfg, scans)
    if data_root is None:
        write_text(run_dir / "report.md", "missing_kitti_root\n" + "\n".join(scans))
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_kitti_root", "params_hash": params_hash})
        return 2

    drive_id = _select_drive(data_root, str(cfg["DRIVE_MATCH"]))
    velodyne_dir = _find_velodyne_dir(data_root, drive_id)

    frame_start = int(cfg["FRAME_START"])
    frame_end = int(cfg["FRAME_END"])
    if frame_end < frame_start:
        raise RuntimeError("invalid_frame_range")

    degrade_log: Dict[str, object] = {"steps": []}
    total_frames = frame_end - frame_start + 1
    if total_frames > 400 and int(cfg["STRIDE"]) < 2:
        cfg["STRIDE"] = 2
        degrade_log["steps"].append({"param": "STRIDE", "value": cfg["STRIDE"], "reason": "time_budget"})
    if degrade_log["steps"]:
        params_hash = _write_resolved(run_dir, cfg)

    def _check_budget(stage: str) -> None:
        elapsed = time.perf_counter() - t_start
        stage_times[stage] = elapsed
        if elapsed > float(cfg["TIME_BUDGET_H"]) * 3600.0:
            degrade_log["time_exceeded"] = stage
            degrade_log["stage_times_s"] = stage_times
            write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "time_budget_exceeded", "stage": stage, "params_hash": params_hash})
            write_text(run_dir / "report.md", f"time_budget_exceeded at {stage}\nparams_hash={params_hash}\n")
            raise _TimeBudgetExceeded(f"time_budget_exceeded at {stage}")

    stride = int(cfg["STRIDE"])
    frame_ids = [f"{i:010d}" for i in range(frame_start, frame_end + 1, stride)]

    roi_buf = float(cfg["ROI_BUFFER_M"])
    traj_line = _build_traj_line(data_root, drive_id, stride=max(1, stride))
    roi_geom = traj_line.buffer(roi_buf)
    roi_gdf = gpd.GeoDataFrame([{"drive_id": drive_id, "geometry": roi_geom}], geometry="geometry", crs="EPSG:32632")

    drive_dir = ensure_dir(run_dir / "drives" / drive_id)
    roi_dir = ensure_dir(drive_dir / "roi")
    roi_path = roi_dir / "roi_corridor_utm32.gpkg"
    write_gpkg_layer(roi_path, "roi", roi_gdf, 32632, crs_warnings)
    _write_sidecar(roi_path, params_hash, {"layer": "roi"})

    ground_points = []
    nonground_points = []
    ground_intensity = []
    nonground_intensity = []
    qa_rows = []

    for frame_id in frame_ids:
        pts_raw = load_kitti360_lidar_points(data_root, drive_id, frame_id)
        if pts_raw.size == 0:
            continue
        xyz = pts_raw[:, :3].astype(np.float64)
        intensity = pts_raw[:, 3].astype(np.float32)

        n, d, inliers = _ransac_plane(
            xyz,
            iters=200,
            dist_thresh=float(cfg["GROUND_RANSAC_DZ"]),
            normal_min_z=float(cfg["GROUND_NORMAL_DOT_Z_MIN"]),
        )
        if inliers.size == 0:
            inlier_ratio = 0.0
        else:
            inlier_ratio = float(np.sum(inliers)) / max(1, int(inliers.size))

        pts_world = transform_points_V_to_W(xyz, data_root, drive_id, frame_id, cam_id=str(cfg["OUTPUT_IMAGE_CAM"]))
        pts_world, intensity = _clip_to_roi(pts_world, intensity, roi_geom)

        if pts_world.size == 0:
            qa_rows.append({"frame_id": frame_id, "inlier_ratio": inlier_ratio, "point_count": 0})
            continue

        inliers_w = inliers[: pts_world.shape[0]] if inliers.size else np.zeros((pts_world.shape[0],), dtype=bool)
        g_pts = pts_world[inliers_w]
        ng_pts = pts_world[~inliers_w]
        g_int = intensity[inliers_w]
        ng_int = intensity[~inliers_w]

        ground_points.append(g_pts.astype(np.float32))
        ground_intensity.append(g_int.astype(np.float32))
        if ng_pts.shape[0] > 0:
            rng = np.random.default_rng(0)
            keep = rng.choice(ng_pts.shape[0], size=max(1, int(ng_pts.shape[0] * 0.1)), replace=False)
            nonground_points.append(ng_pts[keep].astype(np.float32))
            nonground_intensity.append(ng_int[keep].astype(np.float32))

        qa_rows.append({"frame_id": frame_id, "inlier_ratio": inlier_ratio, "point_count": int(pts_world.shape[0])})

    if ground_points:
        ground_xyz = np.vstack(ground_points)
        ground_int = np.concatenate(ground_intensity)
        ground_xyz, ground_int = _voxel_downsample(ground_xyz, ground_int, float(cfg["VOXEL_SIZE_M"]))
    else:
        ground_xyz = np.empty((0, 3), dtype=np.float32)
        ground_int = np.empty((0,), dtype=np.float32)

    if nonground_points:
        nonground_xyz = np.vstack(nonground_points)
        nonground_int = np.concatenate(nonground_intensity)
    else:
        nonground_xyz = np.empty((0, 3), dtype=np.float32)
        nonground_int = np.empty((0,), dtype=np.float32)

    _check_budget("ground")

    pc_dir = ensure_dir(drive_dir / "pointcloud")
    write_las(pc_dir / "ground_points_utm32.laz", ground_xyz, ground_int, np.full((ground_xyz.shape[0],), 2, dtype=np.uint8), 32632)
    write_las(pc_dir / "non_ground_points_utm32.laz", nonground_xyz, nonground_int, np.full((nonground_xyz.shape[0],), 1, dtype=np.uint8), 32632)
    _write_sidecar(pc_dir / "ground_points_utm32.laz", params_hash, {"class": "ground"})
    _write_sidecar(pc_dir / "non_ground_points_utm32.laz", params_hash, {"class": "non_ground"})

    # Accumulate all points for rasters/road/markings
    all_points = []
    all_intensity = []
    for frame_id in frame_ids:
        pts_raw = load_kitti360_lidar_points(data_root, drive_id, frame_id)
        if pts_raw.size == 0:
            continue
        xyz = pts_raw[:, :3].astype(np.float64)
        intensity = pts_raw[:, 3].astype(np.float32)
        pts_world = transform_points_V_to_W(xyz, data_root, drive_id, frame_id, cam_id=str(cfg["OUTPUT_IMAGE_CAM"]))
        pts_world, intensity = _clip_to_roi(pts_world, intensity, roi_geom)
        if pts_world.size == 0:
            continue
        all_points.append(pts_world.astype(np.float32))
        all_intensity.append(intensity.astype(np.float32))

    if not all_points:
        write_text(run_dir / "report.md", "no_points_after_accum")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "no_points", "params_hash": params_hash})
        return 2

    points_xyz = np.vstack(all_points)
    intensity = np.concatenate(all_intensity)
    points_xyz, intensity = _voxel_downsample(points_xyz, intensity, float(cfg["VOXEL_SIZE_M"]))

    # Intensity usable check (raw + scaled)
    raw_nonzero = float(np.sum(intensity > 0)) / max(1, int(intensity.size))
    raw_p50 = float(np.percentile(intensity, 50)) if intensity.size else 0.0
    raw_p99 = float(np.percentile(intensity, 99)) if intensity.size else 0.0
    raw_dyn = raw_p99 - raw_p50

    if bool(cfg["INTENSITY_AUTO_SCALE"]) and intensity.size:
        max_val = float(np.max(intensity))
        if max_val <= 1.5:
            intensity = np.clip(intensity, 0.0, 1.0) * 65535.0
        elif max_val <= 255.0:
            intensity = np.clip(intensity, 0.0, 255.0) * 256.0

    nonzero_ratio = float(np.sum(intensity > 0)) / max(1, int(intensity.size))
    p50 = float(np.percentile(intensity, 50)) if intensity.size else 0.0
    p99 = float(np.percentile(intensity, 99)) if intensity.size else 0.0
    dyn = p99 - p50
    intensity_usable = bool(cfg["INTENSITY_ENABLE"]) and nonzero_ratio >= float(cfg["INTENSITY_FAILFAST_NONZERO_MIN"]) and dyn >= float(cfg["INTENSITY_MIN_DYNAMIC_RANGE"])

    rasters_dir = ensure_dir(drive_dir / "rasters")
    bundle = build_rasters(points_xyz, intensity, roi_geom, float(cfg["RASTER_RES_M"]), ground_band_dz_m=0.15)
    bundle.slope = _compute_slope(bundle.height_p10, bundle.res_m)
    bundle.density_use = bundle.density_all
    _check_budget("rasters")
    dtm_path = rasters_dir / "dtm_p10_utm32.tif"
    inten_path = rasters_dir / "intensity_p95_utm32.tif"
    _write_raster(dtm_path, bundle.height_p10.astype(np.float32), bundle.transform, 32632, crs_warnings, nodata=np.nan)
    _write_raster(inten_path, bundle.intensity_max.astype(np.float32), bundle.transform, 32632, crs_warnings, nodata=np.nan)
    _write_sidecar(dtm_path, params_hash)
    _write_sidecar(inten_path, params_hash)

    # Road (adaptive thresholds + tuning)
    roi_mask = _rasterize_geom(roi_geom, bundle.transform, (bundle.height, bundle.width))
    seed_buf = traj_line.buffer(float(cfg["ROAD_GROW_FROM_TRAJ_BUF_M"]))
    max_expand = traj_line.buffer(float(cfg["ROAD_MAX_EXPAND_BUF_M"]))
    seed_mask = _rasterize_geom(seed_buf, bundle.transform, (bundle.height, bundle.width))

    seed_valid = seed_mask & np.isfinite(bundle.height_p10)
    rough_seed = bundle.roughness[seed_valid]
    slope_seed = bundle.slope[seed_valid]
    dens_seed = bundle.density_use[seed_valid]
    rough_seed_p90 = float(np.nanpercentile(rough_seed, 90)) if rough_seed.size else 0.05
    slope_seed_p90 = float(np.nanpercentile(slope_seed, 90)) if slope_seed.size else 0.15
    dens_seed_p20 = float(np.nanpercentile(dens_seed, 20)) if dens_seed.size else 1.0

    rough_th0 = _clamp(rough_seed_p90 * 1.5, 0.03, 0.12)
    slope_th0 = _clamp(slope_seed_p90 * 1.5, 0.08, 0.35)
    dens_th0 = _clamp(dens_seed_p20 * 0.5, 1.0, 10.0)

    tuning_rows = []
    best = {"traj_cover": -1.0, "road_cover": -1.0, "mask": None, "rough_th": rough_th0, "slope_th": slope_th0, "dens_th": dens_th0, "close_m": 0.8}

    for idx, (rough_mul, dens_mul, close_m) in enumerate(
        [
            (1.0, 1.0, 0.8),
            (1.2, 0.8, 0.8),
            (1.2, 0.8, 1.0),
        ]
    ):
        rough_th = _clamp(rough_th0 * rough_mul, 0.02, 0.15)
        slope_th = slope_th0
        dens_th = _clamp(dens_th0 * dens_mul, 1.5, 12.0)
        mask, stats = _road_from_thresholds(bundle, roi_mask, seed_mask, rough_th, slope_th, dens_th, close_m, max_expand)
        area_road = float(np.sum(mask)) * (bundle.res_m * bundle.res_m)
        area_roi = float(roi_geom.area)
        area_seed = float(np.sum(seed_mask)) * (bundle.res_m * bundle.res_m)
        area_seed_road = float(np.sum(seed_mask & mask)) * (bundle.res_m * bundle.res_m)
        road_cover = area_road / max(area_roi, 1e-6)
        traj_cover = area_seed_road / max(area_seed, 1e-6)
        tuning_rows.append(
            {
                "iter": idx,
                "rough_th": rough_th,
                "slope_th": slope_th,
                "dens_th": dens_th,
                "close_m": close_m,
                "road_cover": road_cover,
                "traj_cover": traj_cover,
                "road_cells": stats["road_cells"],
            }
        )
        if traj_cover > best["traj_cover"] or (traj_cover == best["traj_cover"] and road_cover > best["road_cover"]):
            best = {
                "traj_cover": traj_cover,
                "road_cover": road_cover,
                "mask": mask,
                "rough_th": rough_th,
                "slope_th": slope_th,
                "dens_th": dens_th,
                "close_m": close_m,
            }

    road_mask = best["mask"] if best["mask"] is not None else np.zeros_like(roi_mask, dtype=bool)
    road_mask_path = rasters_dir / "road_mask_utm32.tif"
    _write_raster(road_mask_path, road_mask.astype(np.uint8), bundle.transform, 32632, crs_warnings, nodata=0)
    _write_sidecar(road_mask_path, params_hash, {"rough_th": best["rough_th"], "slope_th": best["slope_th"], "dens_th": best["dens_th"], "close_m": best["close_m"]})
    area_seed = float(np.sum(seed_mask)) * (bundle.res_m * bundle.res_m)
    area_seed_road = float(np.sum(seed_mask & road_mask)) * (bundle.res_m * bundle.res_m)

    road_debug = {
        "roi_bbox": list(roi_geom.bounds),
        "traj_bbox": list(traj_line.bounds),
        "ground_bbox": list(MultiPoint(points_xyz[:, :2]).bounds) if points_xyz.size else [],
        "dtm_valid_ratio": float(np.mean(np.isfinite(bundle.height_p10))),
        "roughness_valid_ratio": float(np.mean(np.isfinite(bundle.roughness))),
        "density_valid_ratio": float(np.mean(bundle.density_use > 0)),
        "roughness_p10": float(np.nanpercentile(bundle.roughness, 10)),
        "roughness_p50": float(np.nanpercentile(bundle.roughness, 50)),
        "roughness_p90": float(np.nanpercentile(bundle.roughness, 90)),
        "roughness_p99": float(np.nanpercentile(bundle.roughness, 99)),
        "density_p10": float(np.nanpercentile(bundle.density_use, 10)),
        "density_p50": float(np.nanpercentile(bundle.density_use, 50)),
        "density_p90": float(np.nanpercentile(bundle.density_use, 90)),
        "density_p99": float(np.nanpercentile(bundle.density_use, 99)),
        "slope_p10": float(np.nanpercentile(bundle.slope, 10)),
        "slope_p50": float(np.nanpercentile(bundle.slope, 50)),
        "slope_p90": float(np.nanpercentile(bundle.slope, 90)),
        "slope_p99": float(np.nanpercentile(bundle.slope, 99)),
        "seed_area_m2": area_seed,
        "seed_in_road_area_m2": area_seed_road,
        "road_area_m2": float(np.sum(road_mask)) * (bundle.res_m * bundle.res_m),
    }
    _check_budget("road")
    road_poly = _mask_to_polygons(road_mask.astype(np.uint8), bundle.transform, min_area_m2=float(cfg["ROAD_MIN_COMPONENT_AREA_M2"]))
    if not road_poly.empty:
        road_poly = road_poly[road_poly.intersects(seed_buf)]
    components_count = int(road_poly.shape[0]) if not road_poly.empty else 0
    holes_area = 0.0
    if not road_poly.empty:
        for geom in road_poly.geometry:
            if geom is None or geom.is_empty:
                continue
            for ring in geom.interiors:
                holes_area += Polygon(ring).area
    vectors_dir = ensure_dir(drive_dir / "vectors")
    road_vec_path = vectors_dir / "road_surface_utm32.gpkg"
    write_gpkg_layer(road_vec_path, "road_surface", road_poly, 32632, crs_warnings)
    _write_sidecar(road_vec_path, params_hash, {"layer": "road_surface"})

    # Road points mask for point export
    road_points_mask = np.zeros((points_xyz.shape[0],), dtype=bool)
    if points_xyz.size:
        ix = bundle.point_ix
        iy = bundle.point_iy
        valid = bundle.point_valid
        road_points_mask[valid] = road_mask[iy[valid], ix[valid]]
    road_points = points_xyz[road_points_mask]
    road_int = intensity[road_points_mask]
    nonroad_points = points_xyz[~road_points_mask]
    nonroad_int = intensity[~road_points_mask]
    write_las(pc_dir / "road_surface_points_utm32.laz", road_points, road_int, np.full((road_points.shape[0],), 11, dtype=np.uint8), 32632)
    write_las(pc_dir / "non_road_points_utm32.laz", nonroad_points, nonroad_int, np.full((nonroad_points.shape[0],), 1, dtype=np.uint8), 32632)
    _write_sidecar(pc_dir / "road_surface_points_utm32.laz", params_hash, {"class": "road_surface"})
    _write_sidecar(pc_dir / "non_road_points_utm32.laz", params_hash, {"class": "non_road"})

    # Road debug outputs + early stop
    qa_dir = ensure_dir(drive_dir / "qa")
    write_json(qa_dir / "road_debug.json", road_debug)
    _write_sidecar(qa_dir / "road_debug.json", params_hash)
    _write_roaddebug_plot(bundle, roi_geom, seed_buf, road_mask, qa_dir / "quicklook_bev_roaddebug.png", params_hash)
    _write_sidecar(qa_dir / "quicklook_bev_roaddebug.png", params_hash)
    write_csv(qa_dir / "road_tuning.csv", tuning_rows, ["iter", "rough_th", "slope_th", "dens_th", "close_m", "road_cover", "traj_cover", "road_cells"])
    _write_sidecar(qa_dir / "road_tuning.csv", params_hash)

    road_pass = road_cover >= 0.10 and traj_cover >= 0.60
    if not road_pass:
        decision = {
            "status": "FAIL",
            "frame_range_used": [frame_start, frame_end],
            "stride_used": stride,
            "intensity_usable": False,
            "markings_best_method": "skipped",
            "key_metrics": {
                "road_cover": road_cover,
                "traj_cover": traj_cover,
                "components_count": components_count,
            },
            "params_hash": params_hash,
            "reason": "road_stage_failed",
        }
        write_json(run_dir / "decision.json", decision)
        report = [
            "# Lidar Extract Best Report",
            "",
            f"- drive_id: {drive_id}",
            f"- frame_range: {frame_start}-{frame_end}",
            f"- stride: {stride}",
            f"- params_hash: {params_hash}",
            "",
            "## Road QA (FAIL)",
            f"- road_cover: {road_cover:.4f}",
            f"- traj_cover: {traj_cover:.4f}",
            f"- components_count: {components_count}",
            "",
            "## Notes",
            "- markings/crosswalk skipped due to road fail",
        ]
        write_text(run_dir / "report.md", "\n".join(report))
        _postcheck(run_dir, params_hash, road_mask_path)
        return 2

    # Intensity markings
    _check_budget("intensity")
    intensity_mask = np.zeros_like(road_mask, dtype=bool)
    intensity_score = np.zeros_like(road_mask, dtype=np.float32)
    intensity_thr = 0.0
    if intensity_usable:
        valid = np.isfinite(bundle.intensity_max)
        radius_cells = int(float(cfg["INTENSITY_BG_WIN_RADIUS_M"]) / float(cfg["RASTER_RES_M"]))
        bg = _box_mean(bundle.intensity_max, valid, radius_cells)
        score = np.maximum(0.0, bundle.intensity_max - bg)
        score_vals = score[road_mask.astype(bool) & np.isfinite(score)]
        norm = float(np.percentile(score_vals, 99)) if score_vals.size else 1.0
        if norm <= 0:
            norm = 1.0
        score = (score / norm).astype(np.float32)
        target = tuple(cfg["MARKING_AREA_RATIO_TARGET"])
        thr = _threshold_for_area_ratio(score, road_mask, target)
        intensity_mask = (score >= thr) & road_mask.astype(bool)
        intensity_score = score
        intensity_thr = float(thr)
        bg_path = rasters_dir / "intensity_bg_utm32.tif"
        score_i_path = rasters_dir / "marking_score_intensity_utm32.tif"
        mask_i_path = rasters_dir / "marking_mask_intensity_utm32.tif"
        _write_raster(bg_path, bg.astype(np.float32), bundle.transform, 32632, crs_warnings, nodata=np.nan)
        _write_raster(score_i_path, intensity_score, bundle.transform, 32632, crs_warnings, nodata=0)
        _write_raster(mask_i_path, intensity_mask.astype(np.uint8), bundle.transform, 32632, crs_warnings, nodata=0)
        _write_sidecar(bg_path, params_hash, {"threshold": intensity_thr})
        _write_sidecar(score_i_path, params_hash)
        _write_sidecar(mask_i_path, params_hash, {"threshold": intensity_thr})

    # Geometry markings
    geom_mask = np.zeros_like(road_mask, dtype=bool)
    geom_score = np.zeros_like(road_mask, dtype=np.float32)
    if bool(cfg["GEOM_MARKINGS_ENABLE"]):
        ix = bundle.point_ix
        iy = bundle.point_iy
        valid = bundle.point_valid
        res_max = np.full((bundle.height, bundle.width), np.nan, dtype=np.float32)
        if points_xyz.size and np.any(valid):
            res = points_xyz[:, 2] - bundle.point_height_p10
            for i in range(res.shape[0]):
                if not valid[i]:
                    continue
                cx = ix[i]
                cy = iy[i]
                if cx < 0 or cy < 0 or cx >= bundle.width or cy >= bundle.height:
                    continue
                val = res[i]
                if not np.isfinite(val):
                    continue
                if np.isnan(res_max[cy, cx]) or val > res_max[cy, cx]:
                    res_max[cy, cx] = float(val)
        geom_score = np.nan_to_num(res_max, nan=0.0).astype(np.float32)
        score_vals = geom_score[road_mask.astype(bool)]
        if score_vals.size:
            norm = float(np.percentile(score_vals, float(cfg["GEOM_RESIDUAL_TOP_PCTL"])))
            if norm <= 0:
                norm = 1.0
            geom_score = geom_score / norm
        target = tuple(cfg["GEOM_MARKING_AREA_RATIO_TARGET"])
        thr = _threshold_for_area_ratio(geom_score, road_mask, target)
        geom_mask = (geom_score >= thr) & road_mask.astype(bool)
        score_g_path = rasters_dir / "marking_score_geom_utm32.tif"
        mask_g_path = rasters_dir / "marking_mask_geom_utm32.tif"
        _write_raster(score_g_path, geom_score, bundle.transform, 32632, crs_warnings, nodata=0)
        _write_raster(mask_g_path, geom_mask.astype(np.uint8), bundle.transform, 32632, crs_warnings, nodata=0)
        _write_sidecar(score_g_path, params_hash)
        _write_sidecar(mask_g_path, params_hash)

    # Choose best markings
    area_road = float(np.sum(road_mask.astype(bool))) * (bundle.res_m * bundle.res_m)
    area_roi = float(roi_geom.area)
    road_cover = area_road / max(area_roi, 1e-6)
    traj_cover = area_seed_road / max(area_seed, 1e-6)
    holes_area_ratio = float(holes_area) / max(area_road, 1e-6)
    area_int = float(np.sum(intensity_mask)) * (bundle.res_m * bundle.res_m)
    area_geom = float(np.sum(geom_mask)) * (bundle.res_m * bundle.res_m)
    ratio_int = area_int / max(area_road, 1e-6)
    ratio_geom = area_geom / max(area_road, 1e-6)
    markings_best = "geom"
    if intensity_usable and ratio_int > 0:
        markings_best = "intensity"
    best_mask = intensity_mask if markings_best == "intensity" else geom_mask
    best_mask_path = rasters_dir / "marking_mask_best_utm32.tif"
    _write_raster(best_mask_path, best_mask.astype(np.uint8), bundle.transform, 32632, crs_warnings, nodata=0)
    _write_sidecar(best_mask_path, params_hash, {"best_method": markings_best})

    # Markings polygons and points
    min_area = float(cfg["ROAD_MIN_COMPONENT_AREA_M2"])
    poly_int = _mask_to_polygons(intensity_mask, bundle.transform, min_area_m2=min_area)
    poly_geom = _mask_to_polygons(geom_mask, bundle.transform, min_area_m2=min_area)
    poly_best = poly_int if markings_best == "intensity" else poly_geom
    mi_path = vectors_dir / "markings_intensity_utm32.gpkg"
    mg_path = vectors_dir / "markings_geom_utm32.gpkg"
    mb_path = vectors_dir / "markings_best_utm32.gpkg"
    write_gpkg_layer(mi_path, "markings_intensity", poly_int, 32632, crs_warnings)
    write_gpkg_layer(mg_path, "markings_geom", poly_geom, 32632, crs_warnings)
    write_gpkg_layer(mb_path, "markings_best", poly_best, 32632, crs_warnings)
    _write_sidecar(mi_path, params_hash, {"layer": "markings_intensity"})
    _write_sidecar(mg_path, params_hash, {"layer": "markings_geom"})
    _write_sidecar(mb_path, params_hash, {"layer": "markings_best", "best_method": markings_best})

    # Markings points
    point_in_mask = np.zeros((points_xyz.shape[0],), dtype=bool)
    if points_xyz.size:
        ix = bundle.point_ix
        iy = bundle.point_iy
        valid = bundle.point_valid
        point_in_mask[valid] = best_mask[iy[valid], ix[valid]]
    mark_pts = points_xyz[point_in_mask]
    mark_int = intensity[point_in_mask]
    write_las(pc_dir / "markings_points_best_utm32.laz", mark_pts, mark_int, np.full((mark_pts.shape[0],), 1, dtype=np.uint8), 32632)
    _write_sidecar(pc_dir / "markings_points_best_utm32.laz", params_hash, {"best_method": markings_best})

    if intensity_usable:
        point_in_mask_int = np.zeros((points_xyz.shape[0],), dtype=bool)
        if points_xyz.size:
            ix = bundle.point_ix
            iy = bundle.point_iy
            valid = bundle.point_valid
            point_in_mask_int[valid] = intensity_mask[iy[valid], ix[valid]]
        pts_int = points_xyz[point_in_mask_int]
        write_las(pc_dir / "markings_points_intensity_utm32.laz", pts_int, intensity[point_in_mask_int], np.full((pts_int.shape[0],), 1, dtype=np.uint8), 32632)
        _write_sidecar(pc_dir / "markings_points_intensity_utm32.laz", params_hash)
    if bool(cfg["GEOM_MARKINGS_ENABLE"]):
        point_in_mask_geom = np.zeros((points_xyz.shape[0],), dtype=bool)
        if points_xyz.size:
            ix = bundle.point_ix
            iy = bundle.point_iy
            valid = bundle.point_valid
            point_in_mask_geom[valid] = geom_mask[iy[valid], ix[valid]]
        pts_geom = points_xyz[point_in_mask_geom]
        write_las(pc_dir / "markings_points_geom_utm32.laz", pts_geom, intensity[point_in_mask_geom], np.full((pts_geom.shape[0],), 1, dtype=np.uint8), 32632)
        _write_sidecar(pc_dir / "markings_points_geom_utm32.laz", params_hash)

    _check_budget("markings")

    # Crosswalk (reuse stripe logic)
    crosswalk_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    crosswalk_debug = {"raw": 0, "filtered": 0, "clusters": 0, "candidates": 0}
    if bool(cfg["CROSSWALK_ENABLE"]):
        from scripts.run_crosswalk_from_markings_0010 import _cluster_stripes, _stripe_candidates, _stripe_orientation

        stripes, stripe_counts = _stripe_candidates(best_mask.astype(np.uint8), geom_score if markings_best == "geom" else intensity_score, bundle.transform, cfg)
        crosswalk_debug["raw"] = stripe_counts.get("raw", 0)
        crosswalk_debug["filtered"] = stripe_counts.get("filtered", 0)
        clusters = _cluster_stripes(stripes, cfg)
        crosswalk_debug["clusters"] = len(clusters)
        rows = []
        for comp in clusters:
            polys = [stripes[i].poly for i in comp]
            if not polys:
                continue
            union = gpd.GeoSeries(polys).unary_union
            merged = union.buffer(float(cfg["CROSSWALK_MERGE_BUF_M"])).buffer(-float(cfg["CROSSWALK_MERGE_BUF_M"]))
            if merged.is_empty:
                continue
            rect = merged.minimum_rotated_rectangle
            l, w, ori = _stripe_orientation(rect)
            area = float(merged.area)
            stripe_count = len(comp)
            if stripe_count < int(cfg["CROSSWALK_CLUSTER_MIN_STRIPES"]):
                continue
            if not (float(cfg["CROSSWALK_W_RANGE_M"][0]) <= w <= float(cfg["CROSSWALK_W_RANGE_M"][1])):
                continue
            if not (float(cfg["CROSSWALK_L_RANGE_M"][0]) <= l <= float(cfg["CROSSWALK_L_RANGE_M"][1])):
                continue
            if not (float(cfg["CROSSWALK_AREA_RANGE_M2"][0]) <= area <= float(cfg["CROSSWALK_AREA_RANGE_M2"][1])):
                continue
            rows.append(
                {
                    "geometry": merged,
                    "stripe_count": stripe_count,
                    "ori_deg": ori,
                    "mrr_w_m": w,
                    "mrr_l_m": l,
                    "area_m2": area,
                }
            )
        if rows:
            crosswalk_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32632")
            crosswalk_debug["candidates"] = len(crosswalk_gdf)
    cw_path = vectors_dir / "crosswalk_best_utm32.gpkg"
    write_gpkg_layer(cw_path, "crosswalk_best", crosswalk_gdf, 32632, crs_warnings)
    _write_sidecar(cw_path, params_hash, {"layer": "crosswalk_best"})
    _check_budget("crosswalk")

    # Overlays
    img_dir = data_root / "data_2d_raw" / drive_id / str(cfg["OUTPUT_IMAGE_CAM"]) / "data_rect"
    if not img_dir.exists():
        img_dir = data_root / "data_2d_raw" / drive_id / str(cfg["OUTPUT_IMAGE_CAM"]) / "data"
    overlay_dir = ensure_dir(drive_dir / "overlays")
    overlay_frames = [f"{int(f):010d}" for f in cfg.get("SAMPLE_OVERLAY_FRAMES", [frame_start, frame_end])]
    calib = load_kitti360_calib_bundle(data_root, drive_id, cam_id=str(cfg["OUTPUT_IMAGE_CAM"]), frame_id_for_size=overlay_frames[0])

    overlay_stats = []
    overlay_paths: List[Path] = []
    h_img = int(calib.image_size[1])
    for frame_id in overlay_frames:
        img_path = img_dir / f"{frame_id}.png"
        if not img_path.exists():
            continue
        with rasterio.open(img_path) as ds:
            img = ds.read()
        img = img.transpose(1, 2, 0) if img.ndim == 3 else np.stack([img, img, img], axis=-1)

        pts_raw = load_kitti360_lidar_points(data_root, drive_id, frame_id)
        if pts_raw.size == 0:
            continue
        xyz_velo = pts_raw[:, :3].astype(np.float64)
        pts_world = transform_points_V_to_W(xyz_velo, data_root, drive_id, frame_id, cam_id=str(cfg["OUTPUT_IMAGE_CAM"]))

        ix = np.floor((pts_world[:, 0] - bundle.minx) / bundle.res_m).astype(np.int32)
        iy = np.floor((pts_world[:, 1] - bundle.miny) / bundle.res_m).astype(np.int32)
        valid = (ix >= 0) & (iy >= 0) & (ix < bundle.width) & (iy < bundle.height)
        road_mask_frame = np.zeros((pts_world.shape[0],), dtype=bool)
        ground_mask_frame = np.zeros((pts_world.shape[0],), dtype=bool)
        if np.any(valid):
            road_mask_frame[valid] = road_mask[iy[valid], ix[valid]]
            p10 = bundle.height_p10[iy[valid], ix[valid]]
            ground_mask_frame[valid] = np.isfinite(p10) & (pts_world[valid, 2] <= (p10 + 0.15))

        frame_sets = [
            ("ground", xyz_velo[ground_mask_frame]),
            ("road", xyz_velo[road_mask_frame]),
            ("markings_best", xyz_velo[road_mask_frame]),
        ]
        for name, pts in frame_sets:
            if pts.size == 0:
                continue
            sel = pts
            if sel.shape[0] > 200000:
                rng = np.random.default_rng(0)
                sel = sel[rng.choice(sel.shape[0], size=200000, replace=False)]
            try:
                proj = project_velo_to_image(sel, calib, use_rect=True, y_flip_mode="fixed_true", sanity=True)
            except Exception:
                proj = project_velo_to_image(sel, calib, use_rect=True, y_flip_mode="fixed_true", sanity=False)
            u, v, in_img = proj["u"], proj["v"], proj["in_img"]
            title = f"frame {frame_id} | {name}"
            out_path = overlay_dir / f"frame_{frame_id}_{name}_density.png"
            _overlay_density(img, u, v, in_img, out_path, title, params_hash)
            _write_sidecar(out_path, params_hash, {"frame_id": frame_id, "layer": name})
            overlay_paths.append(out_path)
            if name in ("ground", "road") and in_img.any():
                bottom_ratio = float(np.mean(v[in_img] > 0.65 * float(h_img)))
                overlay_stats.append({"frame_id": frame_id, "layer": name, "bottom_ratio": bottom_ratio})

        if not crosswalk_gdf.empty:
            pts_cw = []
            for geom in crosswalk_gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                coords = list(geom.exterior.coords)
                if len(coords) > 500:
                    step = max(1, len(coords) // 500)
                    coords = coords[::step]
                pts_cw.extend(coords)
            if pts_cw:
                cw_arr = np.array(pts_cw, dtype=np.float32)
                proj = project_world_to_image_pose(cw_arr, pose, calib, use_rect=True, y_flip_mode="fixed_true", sanity=False)
                u, v, in_img = proj["u"], proj["v"], proj["in_img"]
                title = f"frame {frame_id} | crosswalk_best"
                out_path = overlay_dir / f"frame_{frame_id}_crosswalk_best.png"
                _overlay_density(img, u, v, in_img, out_path, title, params_hash)
                _write_sidecar(out_path, params_hash, {"frame_id": frame_id, "layer": "crosswalk_best"})
                overlay_paths.append(out_path)

    _check_budget("overlays")


    qa_dir = ensure_dir(drive_dir / "qa")
    write_csv(qa_dir / "qa_index.csv", qa_rows, ["frame_id", "inlier_ratio", "point_count"])
    _write_sidecar(qa_dir / "qa_index.csv", params_hash)
    intensity_stats = {
        "nonzero_ratio": nonzero_ratio,
        "p50": p50,
        "p99": p99,
        "dynamic_range": dyn,
        "intensity_usable": intensity_usable,
        "raw_nonzero_ratio": raw_nonzero,
        "raw_p50": raw_p50,
        "raw_p99": raw_p99,
        "raw_dynamic_range": raw_dyn,
        "params_hash": params_hash,
    }
    write_json(qa_dir / "intensity_stats.json", intensity_stats)
    _write_sidecar(qa_dir / "intensity_stats.json", params_hash)
    write_json(qa_dir / "degrade_log.json", degrade_log)
    _write_sidecar(qa_dir / "degrade_log.json", params_hash)

    quicklook_bev = qa_dir / "quicklook_bev.png"
    _write_quicklook_bev(road_mask.astype(bool), best_mask.astype(bool), quicklook_bev, params_hash)
    _write_sidecar(quicklook_bev, params_hash)
    montage_path = qa_dir / "quicklook_image_montage.png"
    _write_image_montage(overlay_paths, montage_path, params_hash)
    _write_sidecar(montage_path, params_hash)

    debug_rows = [{"stage": k, "count": v} for k, v in crosswalk_debug.items()]
    write_csv(qa_dir / "crosswalk_debug.csv", debug_rows, ["stage", "count"])
    _write_sidecar(qa_dir / "crosswalk_debug.csv", params_hash)

    overlay_fail = False
    for row in overlay_stats:
        if row["layer"] in ("ground", "road") and float(row["bottom_ratio"]) < 0.65:
            overlay_fail = True
            break
    decision = {
        "status": "FAIL" if overlay_fail else "PASS",
        "frame_range_used": [frame_start, frame_end],
        "stride_used": stride,
        "intensity_usable": intensity_usable,
        "markings_best_method": markings_best,
        "key_metrics": {
            "road_cover": road_cover,
            "traj_cover": traj_cover,
            "holes_area_ratio": holes_area_ratio,
            "components_count": components_count,
            "marking_area_ratio_best": ratio_int if markings_best == "intensity" else ratio_geom,
            "crosswalk_count": int(crosswalk_gdf.shape[0]),
        },
        "params_hash": params_hash,
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Lidar Extract Best Report",
        "",
        f"- drive_id: {drive_id}",
        f"- frame_range: {frame_start}-{frame_end}",
        f"- stride: {stride}",
        f"- params_hash: {params_hash}",
        f"- crs_warnings: {len(crs_warnings)}",
        "",
        "## Intensity Usable",
        f"- nonzero_ratio: {nonzero_ratio:.4f}",
        f"- dynamic_range: {dyn:.1f}",
        f"- intensity_usable: {intensity_usable}",
        "",
        "## Markings Best",
        f"- best_method: {markings_best}",
        f"- ratio_intensity: {ratio_int:.4f}",
        f"- ratio_geom: {ratio_geom:.4f}",
        "",
        "## Road QA",
        f"- road_cover: {road_cover:.4f}",
        f"- traj_cover: {traj_cover:.4f}",
        f"- holes_area_ratio: {holes_area_ratio:.4f}",
        f"- components_count: {components_count}",
        "",
        "## Crosswalk",
        f"- crosswalk_count: {int(crosswalk_gdf.shape[0])}",
        "",
        "## Overlay Sanity",
        f"- overlay_fail: {overlay_fail}",
        "## Overlay Frames",
        *[f"- {f}" for f in overlay_frames],
    ]
    if crs_warnings:
        report += ["", "## CRS Warnings", *[f"- {w}" for w in crs_warnings]]
    write_text(run_dir / "report.md", "\n".join(report))
    _postcheck(run_dir, params_hash, best_mask_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
