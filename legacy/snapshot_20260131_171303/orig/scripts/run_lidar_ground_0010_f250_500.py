from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString
from shapely.prepared import prep

from pipeline.calib.io_kitti360_calib import load_kitti360_calib_bundle
from pipeline.calib.kitti360_projection import project_velo_to_image
from pipeline.datasets.kitti360_io import (
    _find_oxts_dir,
    _find_velodyne_dir,
    _resolve_velodyne_path,
    load_kitti360_lidar_points,
    load_kitti360_pose,
)
from pipeline.calib.kitti360_world import transform_points_V_to_W
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    now_ts,
    relpath,
    setup_logging,
    write_csv,
    write_json,
    write_text,
)


LOG = logging.getLogger("lidar_ground_0010_f250_500")

REQUIRED_KEYS = [
    "FRAME_START",
    "FRAME_END",
    "STRIDE",
    "TARGET_EPSG",
    "ROI_BUFFER_M",
    "RANGE_MIN_M",
    "RANGE_MAX_M",
    "Y_ABS_MAX_M",
    "GROUND_RANSAC_ITERS",
    "GROUND_RANSAC_DZ",
    "GROUND_NORMAL_DOT_Z_MIN",
    "GRID_RES_M",
    "MIN_CELL_PTS",
    "OVERWRITE",
    "FIT_RANGE_MIN_M",
    "FIT_RANGE_MAX_M",
    "FIT_Y_ABS_MAX_M",
    "FIT_Z_MAX_M",
    "CLASSIFY_DZ_M",
    "POST_GRID_RES_M",
    "POST_MIN_CELL_PTS",
    "POST_RESID_MAX_M",
    "EXPORT_NON_GROUND_SAMPLE_RATE",
]

OVERLAY_FRAMES = [250, 341, 500]


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
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return params_hash


def _resolve_config(base: Dict[str, object], run_dir: Path) -> Tuple[Dict[str, object], str]:
    cfg = dict(base)
    defaults = {
        "FRAME_START": 250,
        "FRAME_END": 500,
        "STRIDE": 1,
        "TARGET_EPSG": 32632,
        "ROI_BUFFER_M": 30,
        "RANGE_MIN_M": 3.0,
        "RANGE_MAX_M": 45.0,
        "Y_ABS_MAX_M": 20.0,
        "GROUND_RANSAC_ITERS": 200,
        "GROUND_RANSAC_DZ": 0.12,
        "GROUND_NORMAL_DOT_Z_MIN": 0.90,
        "GRID_RES_M": 0.20,
        "MIN_CELL_PTS": 5,
        "OVERWRITE": True,
        "FIT_RANGE_MIN_M": 5.0,
        "FIT_RANGE_MAX_M": 35.0,
        "FIT_Y_ABS_MAX_M": 8.0,
        "FIT_Z_MAX_M": 1.0,
        "CLASSIFY_DZ_M": 0.07,
        "POST_GRID_RES_M": 0.20,
        "POST_MIN_CELL_PTS": 5,
        "POST_RESID_MAX_M": 0.08,
        "EXPORT_NON_GROUND_SAMPLE_RATE": 0.05,
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


def _frame_ids(start: int, end: int, stride: int) -> List[str]:
    return [f"{i:010d}" for i in range(int(start), int(end) + 1, max(1, int(stride)))]


def _build_traj_line(data_root: Path, drive_id: str, frame_ids: List[str]) -> LineString:
    points: List[Tuple[float, float]] = []
    for fid in frame_ids:
        x, y, _ = load_kitti360_pose(data_root, drive_id, fid)
        points.append((x, y))
    if len(points) < 2:
        return LineString(points or [(0.0, 0.0), (1.0, 1.0)])
    return LineString(points)


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
    from shapely.geometry import Point

    return np.array([bool(prep_poly.contains(Point(x, y))) for x, y in points_xy], dtype=bool)


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


def _bbox_ok(bounds: Tuple[float, float, float, float, float, float]) -> Tuple[bool, str]:
    minx, miny, minz, maxx, maxy, maxz = bounds
    if all(abs(v) < 1e-9 for v in bounds):
        return False, "bbox_all_zero"
    if any(np.isnan(v) for v in bounds):
        return False, "bbox_has_nan"
    if minx >= maxx or miny >= maxy:
        return False, "bbox_invalid_order"
    if minx < 100000.0 or maxx > 900000.0:
        return False, "bbox_easting_out_of_range"
    if miny < 0.0 or maxy > 10000000.0:
        return False, "bbox_northing_out_of_range"
    if minz == maxz == 0.0:
        return False, "bbox_z_all_zero"
    return True, "ok"


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


def _dtm_grid(points_xyz: np.ndarray, res_m: float, min_cell_pts: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine]:
    if points_xyz.size == 0:
        arr = np.zeros((1, 1), dtype=np.float32)
        transform = from_origin(0.0, 1.0, res_m, res_m)
        return arr, arr.copy(), arr.copy(), transform
    minx, miny = points_xyz[:, 0].min(), points_xyz[:, 1].min()
    maxx, maxy = points_xyz[:, 0].max(), points_xyz[:, 1].max()
    width = int(np.ceil((maxx - minx) / res_m)) + 1
    height = int(np.ceil((maxy - miny) / res_m)) + 1
    idx_x = np.floor((points_xyz[:, 0] - minx) / res_m).astype(np.int64)
    idx_y = np.floor((maxy - points_xyz[:, 1]) / res_m).astype(np.int64)
    lin = idx_y * width + idx_x
    order = np.argsort(lin, kind="mergesort")
    lin_sorted = lin[order]
    z_sorted = points_xyz[order, 2]
    unique_mask = np.empty_like(lin_sorted, dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = lin_sorted[1:] != lin_sorted[:-1]
    group_starts = np.where(unique_mask)[0]
    group_ends = np.append(group_starts[1:], lin_sorted.size)
    median = np.full((height, width), np.nan, dtype=np.float32)
    std = np.full((height, width), np.nan, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)
    for start, end in zip(group_starts, group_ends):
        pts = z_sorted[start:end]
        if pts.size < int(min_cell_pts):
            continue
        lin_id = int(lin_sorted[start])
        y = lin_id // width
        x = lin_id % width
        median[y, x] = float(np.median(pts))
        std[y, x] = float(np.std(pts))
        count[y, x] = int(pts.size)
    transform = from_origin(minx, maxy, res_m, res_m)
    return median, std, count, transform


def _dtm_std_stats(std: np.ndarray) -> Dict[str, float]:
    vals = std[np.isfinite(std)]
    if vals.size == 0:
        return {"median": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "median": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p99": float(np.percentile(vals, 99)),
    }


def _plot_bev_density(points_xy: np.ndarray, out_path: Path, max_points: int = 300000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if points_xy.size == 0:
        return
    rng = np.random.default_rng(0)
    idx = np.arange(points_xy.shape[0])
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    pts = points_xy[idx]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="black", alpha=0.15)
    ax.set_title("BEV Ground Density")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_std_heatmap(std: np.ndarray, transform: rasterio.Affine, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if std.size == 0:
        return
    h, w = std.shape
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(h, w, transform)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    im = ax.imshow(std, cmap="magma", origin="upper", extent=(minx, maxx, miny, maxy))
    ax.set_title("DTM STD (m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _find_image_dir(data_root: Path, drive_id: str, cam: str) -> Optional[Path]:
    candidates = [
        data_root / "data_2d_raw" / drive_id / cam / "data",
        data_root / "data_2d_raw" / drive_id / cam / "data_rect",
        data_root / drive_id / cam / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_image_path(img_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_dir / f"{frame_id}{ext}"
        if p.exists():
            return p
    return None


def _overlay_ground_density(img_path: Path, u: np.ndarray, v: np.ndarray, in_img: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    if u.size:
        ax.scatter(u[in_img], v[in_img], s=0.6, c="red", alpha=0.45)
    ax.set_title("Ground inliers projection")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    base_cfg = _load_yaml(Path("configs/lidar_ground_0010_f250_500.yaml"))
    run_id = now_ts()
    run_dir = Path("runs") / f"lidar_ground_0010_f250_500_{run_id}"
    if bool(base_cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")
    LOG.info("run_start")

    cfg, params_hash = _resolve_config(base_cfg, run_dir)
    scans: List[str] = []
    data_root = _auto_find_kitti_root(cfg, scans)
    if data_root is None:
        write_text(run_dir / "report.md", "data_root_missing")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "data_root_missing", "scan_paths": scans})
        return 2

    drive_id = _select_drive_0010(data_root)
    if drive_id != "2013_05_28_drive_0010_sync":
        write_text(run_dir / "report.md", f"drive_id_mismatch:{drive_id}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "drive_id_mismatch"})
        return 2

    frame_start = int(cfg["FRAME_START"])
    frame_end = int(cfg["FRAME_END"])
    stride = int(cfg["STRIDE"])
    frame_ids = _frame_ids(frame_start, frame_end, stride)

    velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    missing = []
    for fid in frame_ids:
        if _resolve_velodyne_path(velodyne_dir, fid) is None:
            missing.append(fid)
    if missing:
        write_text(run_dir / "report.md", f"missing_frames:{missing}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_frames", "missing": missing})
        return 2

    corridor = _build_traj_line(data_root, drive_id, frame_ids).buffer(float(cfg["ROI_BUFFER_M"]))


    range_min = float(cfg["RANGE_MIN_M"])
    range_max = float(cfg["RANGE_MAX_M"])
    y_abs_max = float(cfg["Y_ABS_MAX_M"])
    fit_range_min = float(cfg["FIT_RANGE_MIN_M"])
    fit_range_max = float(cfg["FIT_RANGE_MAX_M"])
    fit_y_abs_max = float(cfg["FIT_Y_ABS_MAX_M"])
    fit_z_max = float(cfg["FIT_Z_MAX_M"])
    classify_dz = float(cfg["CLASSIFY_DZ_M"])
    ransac_iters = int(cfg["GROUND_RANSAC_ITERS"])
    ransac_dz = float(cfg["GROUND_RANSAC_DZ"])
    normal_min_z = float(cfg["GROUND_NORMAL_DOT_Z_MIN"])
    post_grid_res = float(cfg["POST_GRID_RES_M"])
    post_min_cell_pts = int(cfg["POST_MIN_CELL_PTS"])
    post_resid_max = float(cfg["POST_RESID_MAX_M"])
    non_ground_sample_rate = float(cfg["EXPORT_NON_GROUND_SAMPLE_RATE"])

    ground_points: List[np.ndarray] = []
    ground_intensity: List[np.ndarray] = []
    nonground_points: List[np.ndarray] = []
    nonground_intensity: List[np.ndarray] = []
    per_frame_rows: List[Dict[str, object]] = []
    intensity_rule = "unknown"
    intensity_missing = False
    low_inlier_frames = 0
    max_consec_low = 0

    for idx_f, fid in enumerate(frame_ids):
        raw = load_kitti360_lidar_points(data_root, drive_id, fid)
        if raw.size == 0:
            per_frame_rows.append({"frame_id": fid, "inlier_ratio": 0.0, "a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "inliers": 0, "total": 0})
            continue
        pts = raw[:, :3].astype(np.float32)
        forward = pts[:, 0]
        mask = (forward >= range_min) & (forward <= range_max) & (np.abs(pts[:, 1]) <= y_abs_max)
        pts = pts[mask]
        if pts.size == 0:
            per_frame_rows.append({"frame_id": fid, "inlier_ratio": 0.0, "a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "inliers": 0, "total": 0})
            continue
        intensity_raw = _intensity_field(raw)
        missing_this_frame = False
        if intensity_raw is None:
            missing_this_frame = True
            intensity_raw = np.zeros((raw.shape[0],), dtype=np.float32)
        else:
            intensity_raw = intensity_raw.astype(np.float32)
        intensity_raw = intensity_raw[mask]
        if missing_this_frame:
            intensity = np.zeros((intensity_raw.shape[0],), dtype=np.uint16)
            rule = "missing_zero"
            missing_flag = True
        else:
            intensity, rule, missing_flag = _map_intensity(intensity_raw)
        if intensity_rule == "unknown":
            intensity_rule = rule
        elif intensity_rule != rule:
            intensity_rule = "mixed"
        if missing_flag:
            intensity_missing = True

        fit_mask = (
            (pts[:, 0] >= fit_range_min)
            & (pts[:, 0] <= fit_range_max)
            & (np.abs(pts[:, 1]) <= fit_y_abs_max)
            & (pts[:, 2] <= fit_z_max)
        )
        pts_fit = pts[fit_mask]
        n_fit = int(pts_fit.shape[0])
        n, d, inliers_fit = _ransac_plane(pts_fit, ransac_iters, ransac_dz, normal_min_z)
        inlier_ratio_fit = float(np.mean(inliers_fit)) if inliers_fit.size else 0.0

        if n is None or d is None or inliers_fit.size == 0:
            per_frame_rows.append(
                {
                    "frame_id": fid,
                    "inlier_ratio": 0.0,
                    "a": 0.0,
                    "b": 0.0,
                    "c": 0.0,
                    "d": 0.0,
                    "inliers": 0,
                    "total": int(pts.shape[0]),
                    "n_fit": n_fit,
                    "inlier_ratio_fit": inlier_ratio_fit,
                    "n_ground_frame": 0,
                    "ground_ratio_frame": 0.0,
                }
            )
            low_inlier_frames += 1
            max_consec_low = max(max_consec_low, low_inlier_frames)
            continue

        dist = np.abs(pts @ n + d)
        inliers = dist <= classify_dz
        inlier_ratio = float(np.mean(inliers)) if inliers.size else 0.0
        if inlier_ratio < 0.05:
            low_inlier_frames += 1
            max_consec_low = max(max_consec_low, low_inlier_frames)
        else:
            low_inlier_frames = 0
        per_frame_rows.append(
            {
                "frame_id": fid,
                "inlier_ratio": inlier_ratio,
                "a": float(n[0]),
                "b": float(n[1]),
                "c": float(n[2]),
                "d": float(d),
                "inliers": int(np.sum(inliers)),
                "total": int(pts.shape[0]),
                "n_fit": n_fit,
                "inlier_ratio_fit": inlier_ratio_fit,
                "n_ground_frame": int(np.sum(inliers)),
                "ground_ratio_frame": float(np.sum(inliers)) / max(1, int(pts.shape[0])),
            }
        )

        pts_in = pts[inliers]
        inten_in = intensity[inliers]
        pts_out = pts[~inliers]
        inten_out = intensity[~inliers]

        pts_world = transform_points_V_to_W(pts_in, data_root, drive_id, fid, cam_id="image_00")
        if pts_world.size:
            inside = _mask_points_in_polygon(pts_world[:, :2], corridor)
            pts_world = pts_world[inside]
            inten_in = inten_in[inside]
        if pts_world.size:
            ground_points.append(pts_world.astype(np.float32))
            ground_intensity.append(inten_in.astype(np.uint16))

        if pts_out.size:
            rng = np.random.default_rng(0)
            take = int(max(1, round(pts_out.shape[0] * non_ground_sample_rate)))
            take = min(take, pts_out.shape[0])
            idx = rng.choice(pts_out.shape[0], size=take, replace=False)
            pts_out = pts_out[idx]
            inten_out = inten_out[idx]
            pts_world = transform_points_V_to_W(pts_out, data_root, drive_id, fid, cam_id="image_00")
            if pts_world.size:
                inside = _mask_points_in_polygon(pts_world[:, :2], corridor)
                pts_world = pts_world[inside]
                inten_out = inten_out[inside]
            if pts_world.size:
                nonground_points.append(pts_world.astype(np.float32))
                nonground_intensity.append(inten_out.astype(np.uint16))

        if (idx_f + 1) % 25 == 0:
            LOG.info("frame_progress: %s/%s", idx_f + 1, len(frame_ids))

    if not ground_points:
        write_text(run_dir / "report.md", "ground_points_empty")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "ground_points_empty"})
        return 2

    ground_xyz = np.vstack(ground_points)
    ground_int = np.concatenate(ground_intensity)
    non_xyz = np.vstack(nonground_points) if nonground_points else np.empty((0, 3), dtype=np.float32)
    non_int = np.concatenate(nonground_intensity) if nonground_intensity else np.empty((0,), dtype=np.uint16)

    bounds = (
        float(ground_xyz[:, 0].min()),
        float(ground_xyz[:, 1].min()),
        float(ground_xyz[:, 2].min()),
        float(ground_xyz[:, 0].max()),
        float(ground_xyz[:, 1].max()),
        float(ground_xyz[:, 2].max()),
    )
    bbox_ok, bbox_reason = _bbox_ok(bounds)

    rasters_dir = ensure_dir(run_dir / "rasters")
    dtm_median, dtm_std, dtm_count, transform = _dtm_grid(
        ground_xyz, float(cfg["GRID_RES_M"]), int(cfg["MIN_CELL_PTS"])
    )
    _write_raster(rasters_dir / "dtm_median_utm32.tif", dtm_median.astype(np.float32), transform, int(cfg["TARGET_EPSG"]), np.nan)
    _write_raster(rasters_dir / "dtm_std_raw_utm32.tif", dtm_std.astype(np.float32), transform, int(cfg["TARGET_EPSG"]), np.nan)
    _write_raster(rasters_dir / "dtm_count_utm32.tif", dtm_count.astype(np.int32), transform, int(cfg["TARGET_EPSG"]), 0)

    valid_ratio_raw = float(np.mean(np.isfinite(dtm_median)))
    std_stats_raw = _dtm_std_stats(dtm_std)

    # Post DTM residual filter (clean)
    if ground_xyz.size:
        post_median, _, post_count, post_transform = _dtm_grid(ground_xyz, post_grid_res, post_min_cell_pts)
        rows, cols = rasterio.transform.rowcol(post_transform, ground_xyz[:, 0], ground_xyz[:, 1])
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        valid = (rows >= 0) & (rows < post_median.shape[0]) & (cols >= 0) & (cols < post_median.shape[1])
        rows = rows[valid]
        cols = cols[valid]
        z_vals = ground_xyz[valid, 2]
        med_vals = post_median[rows, cols]
        cnt_vals = post_count[rows, cols]
        keep = np.isfinite(med_vals) & (cnt_vals >= post_min_cell_pts) & (np.abs(z_vals - med_vals) <= post_resid_max)
        keep_idx = np.where(valid)[0][keep]
        clean_xyz = ground_xyz[keep_idx]
        clean_int = ground_int[keep_idx]
    else:
        clean_xyz = np.empty((0, 3), dtype=np.float32)
        clean_int = np.empty((0,), dtype=np.uint16)

    dtm_median_clean, dtm_std_clean, dtm_count_clean, transform_clean = _dtm_grid(
        clean_xyz, float(cfg["GRID_RES_M"]), int(cfg["MIN_CELL_PTS"])
    )
    _write_raster(rasters_dir / "dtm_std_clean_utm32.tif", dtm_std_clean.astype(np.float32), transform_clean, int(cfg["TARGET_EPSG"]), np.nan)

    valid_ratio_clean = float(np.mean(np.isfinite(dtm_median_clean))) if clean_xyz.size else 0.0
    std_stats_clean = _dtm_std_stats(dtm_std_clean)

    tables_dir = ensure_dir(run_dir / "tables")
    write_json(
        tables_dir / "dtm_std_stats.json",
        {
            "raw": std_stats_raw,
            "clean": std_stats_clean,
            "valid_ratio_raw": valid_ratio_raw,
            "valid_ratio_clean": valid_ratio_clean,
            "count_raw": int(ground_xyz.shape[0]),
            "count_clean": int(clean_xyz.shape[0]),
        },
    )

    pc_dir = ensure_dir(run_dir / "pointcloud")
    write_las(
        pc_dir / "ground_points_raw_utm32.laz",
        ground_xyz.astype(np.float32),
        ground_int.astype(np.uint16),
        np.full((ground_xyz.shape[0],), 2, dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )
    write_las(
        pc_dir / "ground_points_clean_utm32.laz",
        clean_xyz.astype(np.float32),
        clean_int.astype(np.uint16),
        np.full((clean_xyz.shape[0],), 2, dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )
    # Backward-compatible alias
    write_las(
        pc_dir / "ground_points_utm32.laz",
        clean_xyz.astype(np.float32),
        clean_int.astype(np.uint16),
        np.full((clean_xyz.shape[0],), 2, dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )
    non_ground_path = pc_dir / "non_ground_sample_utm32.laz"
    if non_xyz.size:
        write_las(
            non_ground_path,
            non_xyz.astype(np.float32),
            non_int.astype(np.uint16),
            np.ones((non_xyz.shape[0],), dtype=np.uint8),
            int(cfg["TARGET_EPSG"]),
        )

    write_csv(
        tables_dir / "per_frame_ground_plane.csv",
        per_frame_rows,
        ["frame_id", "inlier_ratio", "a", "b", "c", "d", "inliers", "total", "n_fit", "inlier_ratio_fit", "n_ground_frame", "ground_ratio_frame"],
    )

    images_dir = ensure_dir(run_dir / "images")
    _plot_bev_density(ground_xyz[:, :2], images_dir / "bev_ground_density.png")
    _plot_std_heatmap(dtm_std.astype(np.float32), transform, images_dir / "bev_dtm_std_raw.png")
    _plot_std_heatmap(dtm_std_clean.astype(np.float32), transform_clean, images_dir / "bev_dtm_std_clean.png")

    overlays_dir = ensure_dir(run_dir / "overlays")
    overlay_status = "ok"
    try:
        calib = load_kitti360_calib_bundle(data_root, drive_id, cam_id="image_00", frame_id_for_size=f"{OVERLAY_FRAMES[0]:010d}")
        img_dir = _find_image_dir(data_root, drive_id, "image_00")
        if img_dir is None:
            raise FileNotFoundError("missing_image_dir")
        for frame_id in OVERLAY_FRAMES:
            fid = f"{frame_id:010d}"
            img_path = _find_image_path(img_dir, fid)
            if img_path is None:
                overlay_status = "missing_images"
                continue
            raw = load_kitti360_lidar_points(data_root, drive_id, fid)
            pts = raw[:, :3].astype(np.float32)
            forward = pts[:, 0]
            mask = (forward >= range_min) & (forward <= range_max) & (np.abs(pts[:, 1]) <= y_abs_max)
            pts = pts[mask]
            n, d, inliers = _ransac_plane(pts, ransac_iters, ransac_dz, normal_min_z)
            if n is None or d is None or inliers.size == 0:
                continue
            pts_in = pts[inliers]
            proj = project_velo_to_image(pts_in, calib, use_rect=True, y_flip_mode="fixed_true", sanity=False)
            out_path = overlays_dir / f"frame_{fid}_ground_density.png"
            _overlay_ground_density(img_path, proj["u"], proj["v"], proj["in_img"], out_path)
    except Exception as exc:
        overlay_status = f"skipped:{exc}"

    warn_reasons: List[str] = []
    if std_stats_clean["p90"] > 0.12:
        warn_reasons.append("dtm_std_p90_high")
    if clean_xyz.shape[0] < int(ground_xyz.shape[0] * 0.2):
        warn_reasons.append("clean_points_low")
    if max_consec_low >= 3:
        warn_reasons.append("low_inlier_ratio_consecutive")

    status = "PASS"
    if ground_xyz.shape[0] == 0 or valid_ratio_raw < 0.05 or not bbox_ok:
        status = "FAIL"
    elif warn_reasons:
        status = "WARN"

    decision = {
        "status": status,
        "dtm_valid_ratio_raw": valid_ratio_raw,
        "dtm_valid_ratio_clean": valid_ratio_clean,
        "dtm_std_p90_raw": std_stats_raw["p90"],
        "dtm_std_p99_raw": std_stats_raw["p99"],
        "dtm_std_p90_clean": std_stats_clean["p90"],
        "dtm_std_p99_clean": std_stats_clean["p99"],
        "max_consecutive_low_inlier": max_consec_low,
        "bbox_ok": bool(bbox_ok),
        "bbox_reason": bbox_reason,
        "warnings": warn_reasons,
        "counts": {
            "raw": int(ground_xyz.shape[0]),
            "clean": int(clean_xyz.shape[0]),
        },
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# LiDAR ground 0010 f250-500",
        "",
        f"- drive_id: {drive_id}",
        f"- frames: {frame_start}-{frame_end}",
        f"- stride: {stride}",
        f"- total_ground_points_raw: {ground_xyz.shape[0]}",
        f"- total_ground_points_clean: {clean_xyz.shape[0]}",
        f"- bbox_ok: {bbox_ok} ({bbox_reason})",
        f"- dtm_valid_ratio_raw: {valid_ratio_raw:.3f}",
        f"- dtm_std_median_raw: {std_stats_raw['median']:.3f}",
        f"- dtm_std_p90_raw: {std_stats_raw['p90']:.3f}",
        f"- dtm_std_p99_raw: {std_stats_raw['p99']:.3f}",
        f"- dtm_valid_ratio_clean: {valid_ratio_clean:.3f}",
        f"- dtm_std_median_clean: {std_stats_clean['median']:.3f}",
        f"- dtm_std_p90_clean: {std_stats_clean['p90']:.3f}",
        f"- dtm_std_p99_clean: {std_stats_clean['p99']:.3f}",
        f"- intensity_rule: {intensity_rule}",
        f"- intensity_missing: {bool(intensity_missing)}",
        f"- overlays: {overlay_status}",
        "",
        "## Interpretation",
        "- p90 <= 0.05~0.08m: vertical consistency usually good",
        "- p90 >= 0.12m or stripe-like hot zones: possible vertical jitter / lever-arm / time sync issues",
        "- if clean p90 drops significantly: non-ground contamination was the main cause",
        "- if clean remains high: consider multi-plane / slope-aware ground modeling (next stage)",
        "",
        "## Outputs",
        f"- {relpath(run_dir, pc_dir / 'ground_points_raw_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'ground_points_clean_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'ground_points_utm32.laz')}",
        f"- {relpath(run_dir, rasters_dir / 'dtm_median_utm32.tif')}",
        f"- {relpath(run_dir, rasters_dir / 'dtm_std_raw_utm32.tif')}",
        f"- {relpath(run_dir, rasters_dir / 'dtm_std_clean_utm32.tif')}",
        f"- {relpath(run_dir, rasters_dir / 'dtm_count_utm32.tif')}",
        f"- {relpath(run_dir, tables_dir / 'per_frame_ground_plane.csv')}",
        f"- {relpath(run_dir, tables_dir / 'dtm_std_stats.json')}",
        f"- {relpath(run_dir, images_dir / 'bev_ground_density.png')}",
        f"- {relpath(run_dir, images_dir / 'bev_dtm_std_raw.png')}",
        f"- {relpath(run_dir, images_dir / 'bev_dtm_std_clean.png')}",
    ]
    if non_ground_path.exists():
        report.insert(report.index("## Outputs") + 1, f"- {relpath(run_dir, non_ground_path)}")
    if overlay_status == "ok":
        report.extend(
            [
                f"- {relpath(run_dir, overlays_dir / 'frame_000250_ground_density.png')}",
                f"- {relpath(run_dir, overlays_dir / 'frame_000341_ground_density.png')}",
                f"- {relpath(run_dir, overlays_dir / 'frame_000500_ground_density.png')}",
            ]
        )
    if intensity_missing:
        report.append("")
        report.append("## Notes")
        report.append("- intensity_missing: true (filled with zeros)")
    write_text(run_dir / "report.md", "\n".join(report))
    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
