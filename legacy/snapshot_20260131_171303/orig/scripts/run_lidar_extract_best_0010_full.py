
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
from pipeline.calib.kitti360_projection import project_world_to_image_pose
from pipeline.calib.proj_sanity import validate_depth, validate_in_image_ratio, validate_uv_spread
from pipeline.datasets.kitti360_io import _find_oxts_dir, _find_velodyne_dir, load_kitti360_lidar_points, load_kitti360_pose, load_kitti360_pose_full
from pipeline.lidar_semantic.accum_points_world import _voxel_downsample
from pipeline.calib.kitti360_world import transform_points_V_to_W
from pipeline.lidar_semantic.build_rasters import build_rasters
from pipeline.lidar_semantic.classify_road import classify_road
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
    "SAMPLE_OVERLAY_FRAMES_MODE",
    "SAMPLE_OVERLAY_COUNT",
    "SAMPLE_OVERLAY_STRATEGY",
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
        "SAMPLE_OVERLAY_FRAMES_MODE": "auto",
        "SAMPLE_OVERLAY_COUNT": 5,
        "SAMPLE_OVERLAY_STRATEGY": "uniform",
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


def _write_raster(path: Path, arr: np.ndarray, transform: rasterio.Affine, epsg: int, nodata: Optional[float] = None) -> None:
    validate_output_crs(str(path), epsg, arr)
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
    val = np.where(valid, values, 0.0)
    cnt = np.where(valid, 1.0, 0.0)
    val_p = np.pad(val, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    cnt_p = np.pad(cnt, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    val_ii = val_p.cumsum(axis=0).cumsum(axis=1)
    cnt_ii = cnt_p.cumsum(axis=0).cumsum(axis=1)
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

def main() -> int:
    base_cfg = _load_yaml(Path("configs/lidar_extract_best_0010_full.yaml"))
    run_dir = Path("runs") / f"lidar_extract_best_0010_full_{now_ts()}"
    if bool(base_cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    cfg, params_hash = _resolve_config(base_cfg, run_dir)

    scans: List[str] = []
    data_root = _auto_find_kitti_root(cfg, scans)
    if data_root is None:
        write_text(run_dir / "report.md", "missing_kitti_root\n" + "\n".join(scans))
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_kitti_root", "params_hash": params_hash})
        return 2

    drive_id = _select_drive(data_root, str(cfg["DRIVE_MATCH"]))
    velodyne_dir = _find_velodyne_dir(data_root, drive_id)

    frame_start, frame_end = _frame_range_from_velodyne(velodyne_dir)
    if str(cfg["FRAME_START"]).lower() != "auto":
        frame_start = int(cfg["FRAME_START"])
    if str(cfg["FRAME_END"]).lower() != "auto":
        frame_end = int(cfg["FRAME_END"])

    stride = int(cfg["STRIDE"])
    frame_ids = [f"{i:010d}" for i in range(frame_start, frame_end + 1, stride)]

    roi_buf = float(cfg["ROI_BUFFER_M"])
    traj_line = _build_traj_line(data_root, drive_id, stride=max(1, stride))
    roi_geom = traj_line.buffer(roi_buf)
    roi_gdf = gpd.GeoDataFrame([{"drive_id": drive_id, "geometry": roi_geom}], geometry="geometry", crs="EPSG:32632")

    drive_dir = ensure_dir(run_dir / "drives" / drive_id)
    roi_dir = ensure_dir(drive_dir / "roi")
    write_gpkg_layer(roi_dir / "roi_corridor_utm32.gpkg", "roi", roi_gdf)

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

    pc_dir = ensure_dir(drive_dir / "pointcloud")
    write_las(pc_dir / "ground_points_utm32.laz", ground_xyz, ground_int, np.full((ground_xyz.shape[0],), 2, dtype=np.uint8), 32632)
    write_las(pc_dir / "non_ground_points_utm32.laz", nonground_xyz, nonground_int, np.full((nonground_xyz.shape[0],), 1, dtype=np.uint8), 32632)

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

    # Intensity usable check
    nonzero_ratio = float(np.sum(intensity > 0)) / max(1, int(intensity.size))
    p50 = float(np.percentile(intensity, 50)) if intensity.size else 0.0
    p99 = float(np.percentile(intensity, 99)) if intensity.size else 0.0
    dyn = p99 - p50
    intensity_usable = bool(cfg["INTENSITY_ENABLE"]) and nonzero_ratio >= float(cfg["INTENSITY_FAILFAST_NONZERO_MIN"]) and dyn >= float(cfg["INTENSITY_MIN_DYNAMIC_RANGE"])

    if bool(cfg["INTENSITY_AUTO_SCALE"]) and intensity.size:
        max_val = float(np.max(intensity))
        if max_val <= 1.5:
            intensity = np.clip(intensity, 0.0, 1.0) * 65535.0
        elif max_val <= 255.0:
            intensity = np.clip(intensity, 0.0, 255.0) * 256.0

    rasters_dir = ensure_dir(drive_dir / "rasters")
    bundle = build_rasters(points_xyz, intensity, roi_geom, float(cfg["RASTER_RES_M"]), ground_band_dz_m=0.15)
    _write_raster(rasters_dir / "dtm_p10_utm32.tif", bundle.height_p10.astype(np.float32), bundle.transform, 32632, nodata=np.nan)
    _write_raster(rasters_dir / "intensity_p95_utm32.tif", bundle.intensity_max.astype(np.float32), bundle.transform, 32632, nodata=np.nan)

    # Road
    road_res = classify_road(
        bundle=bundle,
        points_xyz=points_xyz,
        corridor_geom=roi_geom,
        ground_band_dz_m=0.15,
        min_density=6.0,
        roughness_max_m=float(cfg["ROAD_ROUGHNESS_MAX_M"]),
        close_radius_m=0.6,
    )
    _write_raster(rasters_dir / "road_mask_utm32.tif", road_res.road_mask.astype(np.uint8), bundle.transform, 32632, nodata=0)
    road_poly = road_res.road_polygons
    if not road_poly.empty:
        seed_buf = traj_line.buffer(float(cfg["ROAD_GROW_FROM_TRAJ_BUF_M"]))
        road_poly = road_poly[road_poly.intersects(seed_buf)]
    vectors_dir = ensure_dir(drive_dir / "vectors")
    write_gpkg_layer(vectors_dir / "road_surface_utm32.gpkg", "road_surface", road_poly)

    # Road points mask for point export
    road_points_mask = road_res.road_points_mask
    road_points = points_xyz[road_points_mask]
    road_int = intensity[road_points_mask]
    nonroad_points = points_xyz[~road_points_mask]
    nonroad_int = intensity[~road_points_mask]
    write_las(pc_dir / "road_surface_points_utm32.laz", road_points, road_int, np.full((road_points.shape[0],), 11, dtype=np.uint8), 32632)
    write_las(pc_dir / "non_road_points_utm32.laz", nonroad_points, nonroad_int, np.full((nonroad_points.shape[0],), 1, dtype=np.uint8), 32632)

    # Intensity markings
    intensity_mask = np.zeros_like(road_res.road_mask, dtype=bool)
    intensity_score = np.zeros_like(road_res.road_mask, dtype=np.float32)
    intensity_thr = 0.0
    if intensity_usable:
        valid = np.isfinite(bundle.intensity_max)
        radius_cells = int(float(cfg["INTENSITY_BG_WIN_RADIUS_M"]) / float(cfg["RASTER_RES_M"]))
        bg = _box_mean(bundle.intensity_max, valid, radius_cells)
        score = np.maximum(0.0, bundle.intensity_max - bg)
        score_vals = score[road_res.road_mask.astype(bool) & np.isfinite(score)]
        norm = float(np.percentile(score_vals, 99)) if score_vals.size else 1.0
        if norm <= 0:
            norm = 1.0
        score = (score / norm).astype(np.float32)
        target = tuple(cfg["MARKING_AREA_RATIO_TARGET"])
        thr = _threshold_for_area_ratio(score, road_res.road_mask, target)
        intensity_mask = (score >= thr) & road_res.road_mask.astype(bool)
        intensity_score = score
        intensity_thr = float(thr)
        _write_raster(rasters_dir / "intensity_bg_utm32.tif", bg.astype(np.float32), bundle.transform, 32632, nodata=np.nan)
        _write_raster(rasters_dir / "marking_score_intensity_utm32.tif", intensity_score, bundle.transform, 32632, nodata=0)
        _write_raster(rasters_dir / "marking_mask_intensity_utm32.tif", intensity_mask.astype(np.uint8), bundle.transform, 32632, nodata=0)

    # Geometry markings
    geom_mask = np.zeros_like(road_res.road_mask, dtype=bool)
    geom_score = np.zeros_like(road_res.road_mask, dtype=np.float32)
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
        score_vals = geom_score[road_res.road_mask.astype(bool)]
        if score_vals.size:
            norm = float(np.percentile(score_vals, float(cfg["GEOM_RESIDUAL_TOP_PCTL"])))
            if norm <= 0:
                norm = 1.0
            geom_score = geom_score / norm
        target = tuple(cfg["GEOM_MARKING_AREA_RATIO_TARGET"])
        thr = _threshold_for_area_ratio(geom_score, road_res.road_mask, target)
        geom_mask = (geom_score >= thr) & road_res.road_mask.astype(bool)
        _write_raster(rasters_dir / "marking_score_geom_utm32.tif", geom_score, bundle.transform, 32632, nodata=0)
        _write_raster(rasters_dir / "marking_mask_geom_utm32.tif", geom_mask.astype(np.uint8), bundle.transform, 32632, nodata=0)

    # Choose best markings
    area_road = float(np.sum(road_res.road_mask.astype(bool))) * (bundle.res_m * bundle.res_m)
    area_int = float(np.sum(intensity_mask)) * (bundle.res_m * bundle.res_m)
    area_geom = float(np.sum(geom_mask)) * (bundle.res_m * bundle.res_m)
    ratio_int = area_int / max(area_road, 1e-6)
    ratio_geom = area_geom / max(area_road, 1e-6)
    markings_best = "geom"
    if intensity_usable and ratio_int > 0:
        markings_best = "intensity"
    best_mask = intensity_mask if markings_best == "intensity" else geom_mask
    _write_raster(rasters_dir / "marking_mask_best_utm32.tif", best_mask.astype(np.uint8), bundle.transform, 32632, nodata=0)

    # Markings polygons and points
    min_area = float(cfg["ROAD_MIN_COMPONENT_AREA_M2"])
    poly_int = _mask_to_polygons(intensity_mask, bundle.transform, min_area_m2=min_area)
    poly_geom = _mask_to_polygons(geom_mask, bundle.transform, min_area_m2=min_area)
    poly_best = poly_int if markings_best == "intensity" else poly_geom
    write_gpkg_layer(vectors_dir / "markings_intensity_utm32.gpkg", "markings_intensity", poly_int)
    write_gpkg_layer(vectors_dir / "markings_geom_utm32.gpkg", "markings_geom", poly_geom)
    write_gpkg_layer(vectors_dir / "markings_best_utm32.gpkg", "markings_best", poly_best)

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

    if intensity_usable:
        point_in_mask_int = np.zeros((points_xyz.shape[0],), dtype=bool)
        if points_xyz.size:
            ix = bundle.point_ix
            iy = bundle.point_iy
            valid = bundle.point_valid
            point_in_mask_int[valid] = intensity_mask[iy[valid], ix[valid]]
        pts_int = points_xyz[point_in_mask_int]
        write_las(pc_dir / "markings_points_intensity_utm32.laz", pts_int, intensity[point_in_mask_int], np.full((pts_int.shape[0],), 1, dtype=np.uint8), 32632)
    if bool(cfg["GEOM_MARKINGS_ENABLE"]):
        point_in_mask_geom = np.zeros((points_xyz.shape[0],), dtype=bool)
        if points_xyz.size:
            ix = bundle.point_ix
            iy = bundle.point_iy
            valid = bundle.point_valid
            point_in_mask_geom[valid] = geom_mask[iy[valid], ix[valid]]
        pts_geom = points_xyz[point_in_mask_geom]
        write_las(pc_dir / "markings_points_geom_utm32.laz", pts_geom, intensity[point_in_mask_geom], np.full((pts_geom.shape[0],), 1, dtype=np.uint8), 32632)

    # Crosswalk (reuse stripe logic)
    crosswalk_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    if bool(cfg["CROSSWALK_ENABLE"]):
        from scripts.run_crosswalk_from_markings_0010 import _cluster_stripes, _stripe_candidates, _stripe_orientation

        stripes, stripe_counts = _stripe_candidates(best_mask.astype(np.uint8), geom_score if markings_best == "geom" else intensity_score, bundle.transform, cfg)
        clusters = _cluster_stripes(stripes, cfg)
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
    write_gpkg_layer(vectors_dir / "crosswalk_best_utm32.gpkg", "crosswalk_best", crosswalk_gdf)

    # Overlays
    img_dir = data_root / "data_2d_raw" / drive_id / str(cfg["OUTPUT_IMAGE_CAM"]) / "data_rect"
    if not img_dir.exists():
        img_dir = data_root / "data_2d_raw" / drive_id / str(cfg["OUTPUT_IMAGE_CAM"]) / "data"
    overlay_dir = ensure_dir(drive_dir / "overlays")
    overlay_frames = _sample_frames(frame_start, frame_end, int(cfg["SAMPLE_OVERLAY_COUNT"]))
    calib = load_kitti360_calib_bundle(data_root, drive_id, cam_id=str(cfg["OUTPUT_IMAGE_CAM"]), frame_id_for_size=overlay_frames[0])

    for frame_id in overlay_frames:
        img_path = img_dir / f"{frame_id}.png"
        if not img_path.exists():
            continue
        with rasterio.open(img_path) as ds:
            img = ds.read()
        img = img.transpose(1, 2, 0) if img.ndim == 3 else np.stack([img, img, img], axis=-1)
        x, y, z, roll, pitch, yaw = load_kitti360_pose_full(data_root, drive_id, frame_id)
        pose = (x, y, z, roll, pitch, yaw)

        for name, pts in [
            ("ground", ground_xyz),
            ("road", road_points),
            ("markings_best", mark_pts),
        ]:
            if pts.size == 0:
                continue
            sel = pts
            if sel.shape[0] > 200000:
                rng = np.random.default_rng(0)
                sel = sel[rng.choice(sel.shape[0], size=200000, replace=False)]
            proj = project_world_to_image_pose(sel, pose, calib, use_rect=True, y_flip_mode="fixed_true", sanity=False)
            u, v, in_img = proj["u"], proj["v"], proj["in_img"]
            title = f"frame {frame_id} | {name}"
            _overlay_density(img, u, v, in_img, overlay_dir / f"frame_{frame_id}_{name}_density.png", title, params_hash)

    qa_dir = ensure_dir(drive_dir / "qa")
    write_csv(qa_dir / "qa_index.csv", qa_rows, ["frame_id", "inlier_ratio", "point_count"])
    intensity_stats = {
        "nonzero_ratio": nonzero_ratio,
        "p50": p50,
        "p99": p99,
        "dynamic_range": dyn,
        "intensity_usable": intensity_usable,
        "params_hash": params_hash,
    }
    write_json(qa_dir / "intensity_stats.json", intensity_stats)

    decision = {
        "status": "PASS",
        "frame_range_used": [frame_start, frame_end],
        "stride_used": stride,
        "intensity_usable": intensity_usable,
        "markings_best_method": markings_best,
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
        "## Overlay Frames",
        *[f"- {f}" for f in overlay_frames],
    ]
    write_text(run_dir / "report.md", "\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
