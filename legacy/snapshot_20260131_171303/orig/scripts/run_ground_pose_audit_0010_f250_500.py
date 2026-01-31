from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point
from shapely.prepared import prep

from pipeline.calib.io_kitti360_calib import load_kitti360_calib_bundle
from pipeline.calib.kitti360_projection import project_velo_to_image, project_cam0_to_image
from pipeline.calib.kitti360_world import get_T_W_C0, get_T_W_V, transform_points_V_to_W
from pipeline.datasets.kitti360_io import (
    _find_velodyne_dir,
    _resolve_velodyne_path,
    load_kitti360_lidar_points,
    load_kitti360_pose,
)
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


LOG = logging.getLogger("ground_pose_audit_0010_f250_500")

RUN_BASE = Path(r"runs\lidar_ground_0010_f250_500_20260129_232551")
INPUT_GROUND_LAZ = RUN_BASE / "pointcloud" / "ground_points_utm32.laz"
INPUT_PER_FRAME = RUN_BASE / "tables" / "per_frame_ground_plane.csv"
INPUT_DTM_MEDIAN = RUN_BASE / "rasters" / "dtm_median_utm32.tif"
INPUT_DTM_STD = RUN_BASE / "rasters" / "dtm_std_utm32.tif"
INPUT_DTM_COUNT = RUN_BASE / "rasters" / "dtm_count_utm32.tif"

# Fixed params
FRAME_START = 250
FRAME_END = 500
STRIDE = 1
TARGET_EPSG = 32632
ROI_BUFFER_M = 30.0
RANGE_MIN_M = 3.0
RANGE_MAX_M = 45.0
Y_ABS_MAX_M = 20.0
GROUND_RANSAC_ITERS = 200
GROUND_RANSAC_DZ = 0.12
GROUND_NORMAL_DOT_Z_MIN = 0.90
GRID_RES_M = 0.20
MIN_CELL_PTS = 5
ROUNDTRIP_SAMPLE_N = 500
HOTSPOT_STD_TH = 0.12

SEGMENTS = [
    (250, 300),
    (301, 350),
    (351, 400),
    (401, 450),
    (451, 500),
]


def _read_csv(path: Path) -> List[Dict[str, object]]:
    import csv

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


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
            return np.asarray(prep_poly.contains_xy(points_xy[:, 0], points_xy[:, 1]), dtype=bool)
        except Exception:
            return np.array([bool(prep_poly.contains_xy(x, y)) for x, y in points_xy], dtype=bool)
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
    std = np.full((height, width), np.nan, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)
    for start, end in zip(group_starts, group_ends):
        pts = z_sorted[start:end]
        if pts.size < int(min_cell_pts):
            continue
        lin_id = int(lin_sorted[start])
        y = lin_id // width
        x = lin_id % width
        std[y, x] = float(np.std(pts))
        count[y, x] = int(pts.size)
    transform = from_origin(minx, maxy, res_m, res_m)
    return std, count, transform


def _p_stats(values: np.ndarray, pcts: Tuple[int, int]) -> Tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 0.0, 0.0
    return float(np.percentile(vals, pcts[0])), float(np.percentile(vals, pcts[1]))


def _plot_series(x: List[int], y: List[float], out_path: Path, title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=160)
    ax.plot(x, y, linewidth=1.2, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_std_hotspots(std: np.ndarray, transform: rasterio.Affine, out_path: Path, th: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = std.shape
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(h, w, transform)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    im = ax.imshow(std, cmap="magma", origin="upper", extent=(minx, maxx, miny, maxy))
    mask = np.isfinite(std) & (std >= th)
    if np.any(mask):
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors="cyan", linewidths=0.6, origin="upper", extent=(minx, maxx, miny, maxy))
    ax.set_title(f"DTM STD hotspots >= {th:.2f} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    run_id = now_ts()
    run_dir = Path("runs") / f"ground_pose_audit_0010_f250_500_{run_id}"
    ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")
    LOG.info("run_start")

    scans: List[str] = []
    data_root = _auto_find_kitti_root(scans)
    if data_root is None:
        write_text(run_dir / "report.md", "data_root_missing")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "data_root_missing", "scan_paths": scans})
        return 2

    drive_id = _select_drive_0010(data_root)
    frame_ids = _frame_ids(FRAME_START, FRAME_END, STRIDE)
    velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    missing = [fid for fid in frame_ids if _resolve_velodyne_path(velodyne_dir, fid) is None]
    if missing:
        write_text(run_dir / "report.md", f"missing_frames:{missing}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_frames", "missing": missing})
        return 2

    if not INPUT_PER_FRAME.exists():
        write_text(run_dir / "report.md", f"missing_input:{INPUT_PER_FRAME}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_per_frame_csv"})
        return 2
    if not INPUT_DTM_MEDIAN.exists() or not INPUT_DTM_STD.exists() or not INPUT_DTM_COUNT.exists():
        write_text(run_dir / "report.md", "missing_dtm_rasters")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_dtm_rasters"})
        return 2

    with rasterio.open(INPUT_DTM_MEDIAN) as ds:
        dtm_median = ds.read(1)
        dtm_transform = ds.transform
    with rasterio.open(INPUT_DTM_STD) as ds:
        dtm_std = ds.read(1)
    with rasterio.open(INPUT_DTM_COUNT) as ds:
        dtm_count = ds.read(1)

    corridor = _build_traj_line(data_root, drive_id, frame_ids).buffer(ROI_BUFFER_M)

    pose_rows: List[Dict[str, object]] = []
    offset_rows: List[Dict[str, object]] = []
    roundtrip_rows: List[Dict[str, object]] = []
    per_frame_offsets: List[float] = []
    pose_z_list: List[float] = []
    pose_dz_list: List[float] = []
    roundtrip_errs: List[float] = []
    frame_ground_world: Dict[str, np.ndarray] = {}

    calib = None
    img_dir = None
    try:
        calib = load_kitti360_calib_bundle(data_root, drive_id, cam_id="image_00", frame_id_for_size=frame_ids[0])
        img_dir = data_root / "data_2d_raw" / drive_id / "image_00" / "data"
        if not img_dir.exists():
            img_dir = data_root / "data_2d_raw" / drive_id / "image_00" / "data_rect"
    except Exception:
        calib = None

    for idx_f, fid in enumerate(frame_ids):
        raw = load_kitti360_lidar_points(data_root, drive_id, fid)
        pts = raw[:, :3].astype(np.float32)
        forward = pts[:, 0]
        mask = (forward >= RANGE_MIN_M) & (forward <= RANGE_MAX_M) & (np.abs(pts[:, 1]) <= Y_ABS_MAX_M)
        pts = pts[mask]
        if pts.size == 0:
            continue

        n, d, inliers = _ransac_plane(pts, GROUND_RANSAC_ITERS, GROUND_RANSAC_DZ, GROUND_NORMAL_DOT_Z_MIN)
        if n is None or d is None or inliers.size == 0:
            continue

        t_w_v = get_T_W_V(data_root, drive_id, fid, cam_id="image_00")
        origin_world = (t_w_v @ np.array([0.0, 0.0, 0.0, 1.0], dtype=float))[:3]
        pose_z = float(origin_world[2])
        if pose_z_list:
            dz = float(pose_z - pose_z_list[-1])
        else:
            dz = 0.0
        pose_z_list.append(pose_z)
        pose_dz_list.append(dz)
        pose_rows.append({"frame_id": fid, "z_pose": pose_z, "dz_pose": dz})

        pts_in = pts[inliers]
        pts_world = transform_points_V_to_W(pts_in, data_root, drive_id, fid, cam_id="image_00")
        if pts_world.size:
            inside = _mask_points_in_polygon(pts_world[:, :2], corridor)
            pts_world = pts_world[inside]
        if pts_world.size:
            frame_ground_world[fid] = pts_world.astype(np.float32)
        if pts_world.size:
            rows, cols = rasterio.transform.rowcol(dtm_transform, pts_world[:, 0], pts_world[:, 1])
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            valid = (rows >= 0) & (rows < dtm_median.shape[0]) & (cols >= 0) & (cols < dtm_median.shape[1])
            rows = rows[valid]
            cols = cols[valid]
            z_vals = pts_world[valid, 2]
            dtm_vals = dtm_median[rows, cols]
            valid2 = np.isfinite(dtm_vals)
            if np.any(valid2):
                residuals = z_vals[valid2] - dtm_vals[valid2]
                med = float(np.median(residuals))
            else:
                med = 0.0
        else:
            med = 0.0
        per_frame_offsets.append(med)
        offset_rows.append({"frame_id": fid, "median_residual": med})

        if calib is not None and pts.shape[0] >= 10:
            rng = np.random.default_rng(0)
            sample_n = min(ROUNDTRIP_SAMPLE_N, pts.shape[0])
            idx = rng.choice(pts.shape[0], size=sample_n, replace=False)
            pts_s = pts[idx]
            proj_b = project_velo_to_image(pts_s, calib, use_rect=True, y_flip_mode="fixed_true", sanity=False)

            pts_world = transform_points_V_to_W(pts_s, data_root, drive_id, fid, cam_id="image_00")
            t_c0_w = np.linalg.inv(get_T_W_C0(data_root, drive_id, fid))
            pts_w_h = np.hstack([pts_world[:, :3], np.ones((pts_world.shape[0], 1), dtype=pts_world.dtype)])
            pts_c0 = (t_c0_w @ pts_w_h.T).T[:, :3]
            proj_a = project_cam0_to_image(pts_c0, calib, use_rect=True, y_flip=True, sanity=False)

            valid = proj_a["in_img"] & proj_b["in_img"]
            if np.any(valid):
                du = np.abs(proj_a["u"][valid] - proj_b["u"][valid])
                dv = np.abs(proj_a["v"][valid] - proj_b["v"][valid])
                err = np.sqrt(du * du + dv * dv)
                roundtrip_errs.extend(err.tolist())
                roundtrip_rows.append(
                    {
                        "frame_id": fid,
                        "du_p50": float(np.percentile(du, 50)),
                        "du_p90": float(np.percentile(du, 90)),
                        "dv_p50": float(np.percentile(dv, 50)),
                        "dv_p90": float(np.percentile(dv, 90)),
                        "err_p50": float(np.percentile(err, 50)),
                        "err_p90": float(np.percentile(err, 90)),
                    }
                )

        if (idx_f + 1) % 25 == 0:
            LOG.info("frame_progress: %s/%s", idx_f + 1, len(frame_ids))

    tables_dir = ensure_dir(run_dir / "tables")
    checks_dir = ensure_dir(run_dir / "checks")
    write_csv(tables_dir / "frame_pose_z.csv", pose_rows, ["frame_id", "z_pose", "dz_pose"])
    write_csv(tables_dir / "frame_ground_offset.csv", offset_rows, ["frame_id", "median_residual"])
    plane_rows = _read_csv(INPUT_PER_FRAME)
    if plane_rows:
        write_csv(tables_dir / "frame_plane_params.csv", plane_rows, list(plane_rows[0].keys()))
    else:
        write_csv(tables_dir / "frame_plane_params.csv", [], ["frame_id", "inlier_ratio", "a", "b", "c", "d", "inliers", "total"])
    if roundtrip_rows:
        write_csv(
            checks_dir / "world_vs_projection_roundtrip.csv",
            roundtrip_rows,
            ["frame_id", "du_p50", "du_p90", "dv_p50", "dv_p90", "err_p50", "err_p90"],
        )

    seg_rows = []
    for seg_start, seg_end in SEGMENTS:
        seg_ids = _frame_ids(seg_start, seg_end, STRIDE)
        seg_ground = [frame_ground_world[fid] for fid in seg_ids if fid in frame_ground_world]
        if seg_ground:
            seg_xyz = np.vstack(seg_ground)
            seg_std, seg_cnt, seg_tf = _dtm_grid(seg_xyz, GRID_RES_M, MIN_CELL_PTS)
            p90, p99 = _p_stats(seg_std, (90, 99))
            valid_ratio = float(np.mean(np.isfinite(seg_std)))
        else:
            p90, p99, valid_ratio = 0.0, 0.0, 0.0
        seg_rows.append(
            {
                "segment": f"{seg_start}-{seg_end}",
                "dtm_std_p90": p90,
                "dtm_std_p99": p99,
                "dtm_valid_ratio": valid_ratio,
            }
        )
    write_csv(tables_dir / "dtm_std_by_segment.csv", seg_rows, ["segment", "dtm_std_p90", "dtm_std_p99", "dtm_valid_ratio"])

    pose_dz = np.array([abs(v) for v in pose_dz_list[1:]], dtype=np.float64) if len(pose_dz_list) > 1 else np.zeros((0,))
    pose_dz_p90 = float(np.percentile(pose_dz, 90)) if pose_dz.size else 0.0
    frame_offset = np.array([abs(v) for v in per_frame_offsets], dtype=np.float64)
    frame_offset_p90 = float(np.percentile(frame_offset, 90)) if frame_offset.size else 0.0
    roundtrip_p90 = float(np.percentile(np.array(roundtrip_errs, dtype=np.float64), 90)) if roundtrip_errs else 0.0

    if roundtrip_p90 > 3.0:
        root_cause = "chain_inconsistent"
    elif pose_dz_p90 > 0.1 and frame_offset_p90 > 0.2:
        root_cause = "pose_z_jitter"
    else:
        outlier_ratio = float(np.mean(frame_offset > 0.3)) if frame_offset.size else 0.0
        if outlier_ratio > 0.05 and pose_dz_p90 <= 0.1:
            root_cause = "ground_fit_outliers"
        else:
            root_cause = "mixed"

    status = "PASS"
    if roundtrip_p90 > 3.0:
        status = "FAIL"
    elif pose_dz_p90 > 0.15 or frame_offset_p90 > 0.4:
        status = "WARN"

    images_dir = ensure_dir(run_dir / "images")
    _plot_series([int(fid) for fid in frame_ids], pose_z_list, images_dir / "z_series.png", "Pose Z (world)", "z (m)")
    _plot_series([int(fid) for fid in frame_ids], per_frame_offsets, images_dir / "ground_offset_series.png", "Frame ground median offset", "delta z (m)")
    _plot_std_hotspots(dtm_std.astype(np.float32), dtm_transform, images_dir / "dtm_std_hotspots.png", HOTSPOT_STD_TH)

    decision = {
        "status": status,
        "root_cause": root_cause,
        "key_numbers": {
            "dtm_std_p90": float(np.percentile(dtm_std[np.isfinite(dtm_std)], 90)) if np.isfinite(dtm_std).any() else 0.0,
            "pose_dz_p90": pose_dz_p90,
            "frame_offset_p90": frame_offset_p90,
            "roundtrip_px_p90": roundtrip_p90,
        },
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Ground pose audit 0010 f250-500",
        "",
        f"- run_base: {RUN_BASE}",
        f"- dtm_std_p90: {decision['key_numbers']['dtm_std_p90']:.3f}",
        f"- pose_dz_p90: {pose_dz_p90:.3f}",
        f"- frame_offset_p90: {frame_offset_p90:.3f}",
        f"- roundtrip_px_p90: {roundtrip_p90:.3f}",
        f"- root_cause: {root_cause}",
        f"- status: {status}",
        "",
        "## Outputs",
        f"- {relpath(run_dir, tables_dir / 'frame_pose_z.csv')}",
        f"- {relpath(run_dir, tables_dir / 'frame_ground_offset.csv')}",
        f"- {relpath(run_dir, tables_dir / 'frame_plane_params.csv')}",
        f"- {relpath(run_dir, tables_dir / 'dtm_std_by_segment.csv')}",
        f"- {relpath(run_dir, images_dir / 'z_series.png')}",
        f"- {relpath(run_dir, images_dir / 'ground_offset_series.png')}",
        f"- {relpath(run_dir, images_dir / 'dtm_std_hotspots.png')}",
    ]
    if (checks_dir / "world_vs_projection_roundtrip.csv").exists():
        report.append(f"- {relpath(run_dir, checks_dir / 'world_vs_projection_roundtrip.csv')}")
    write_text(run_dir / "report.md", "\n".join(report))
    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
