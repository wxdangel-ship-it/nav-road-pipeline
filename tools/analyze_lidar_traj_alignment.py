import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.datasets.kitti360_io import (  # noqa: E402
    load_kitti360_cam_to_pose,
    load_kitti360_lidar_points,
    load_kitti360_pose_full,
)


def _parse_frame_id(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        return int(float(text))
    except ValueError:
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_pose_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    c1 = float(np.cos(yaw))
    s1 = float(np.sin(yaw))
    c2 = float(np.cos(pitch))
    s2 = float(np.sin(pitch))
    c3 = float(np.cos(roll))
    s3 = float(np.sin(roll))
    r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
    return r_z @ r_y @ r_x


def _rasterize_points(
    points: np.ndarray,
    values: np.ndarray,
    grid_m: float,
    min_points: int,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], float]:
    if points.shape[0] == 0:
        return np.zeros((1, 1), dtype=float), (0.0, 0.0, 0.0, 0.0), grid_m
    minx = float(points[:, 0].min())
    miny = float(points[:, 1].min())
    maxx = float(points[:, 0].max())
    maxy = float(points[:, 1].max())
    width = int(np.ceil((maxx - minx) / grid_m)) + 1
    height = int(np.ceil((maxy - miny) / grid_m)) + 1
    raster = np.zeros((height, width), dtype=float)
    counts = np.zeros((height, width), dtype=int)
    xs = ((points[:, 0] - minx) / grid_m).astype(int)
    ys = ((maxy - points[:, 1]) / grid_m).astype(int)
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    for x, y, val in zip(xs, ys, values):
        raster[y, x] += val
        counts[y, x] += 1
    raster[counts < min_points] = 0.0
    return raster, (minx, miny, maxx, maxy), grid_m


def _build_velo_to_pose(data_root: Path, cam_id: str) -> np.ndarray:
    t_cam_to_pose = load_kitti360_cam_to_pose(data_root, cam_id)
    calib_path = data_root / "calibration" / "calib_cam_to_velo.txt"
    nums = [float(v) for v in calib_path.read_text(encoding="utf-8").split()]
    if len(nums) != 12:
        raise ValueError("parse_error:calib_cam_to_velo")
    mat = np.array(nums, dtype=float).reshape(3, 4)
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
    t_cam_to_velo = np.vstack([mat, bottom])
    t_velo_to_cam = np.linalg.inv(t_cam_to_velo)
    return t_cam_to_pose @ t_velo_to_cam


def _transform_points_full_pose(
    points: np.ndarray,
    pose: Tuple[float, float, float, float, float, float],
    t_pose_velo: np.ndarray,
) -> np.ndarray:
    x, y, z, roll, pitch, yaw = pose
    r_world_pose = _build_pose_rotation(roll, pitch, yaw)
    pts = points[:, :3]
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.hstack([pts, ones])
    pts_pose = (t_pose_velo @ pts_h.T)[:3].T
    pts_world = (r_world_pose @ pts_pose.T).T + np.array([x, y, z], dtype=float)
    return pts_world


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--traj-points", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--camera", default="image_00")
    ap.add_argument("--lidar-grid-m", type=float, default=0.5)
    ap.add_argument("--lidar-min-points", type=int, default=5)
    ap.add_argument("--lidar-max-points-per-frame", type=int, default=20000)
    ap.add_argument("--lidar-z-min", type=float, default=-2.0)
    ap.add_argument("--lidar-z-max", type=float, default=0.5)
    args = ap.parse_args()

    data_root = Path(args.kitti_root)
    traj_path = Path(args.traj_points)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    traj = gpd.read_file(traj_path)
    if "drive_id" in traj.columns:
        traj = traj[traj["drive_id"].astype(str) == args.drive]
    if traj.empty:
        raise RuntimeError("traj_points_empty")

    traj["frame_id_parsed"] = traj["frame_id"].apply(_parse_frame_id)
    traj = traj[traj["frame_id_parsed"].notna()]
    if traj.empty:
        raise RuntimeError("traj_points_no_frame_id")

    frame_start = args.frame_start if args.frame_start is not None else int(traj["frame_id_parsed"].min())
    frame_end = args.frame_end if args.frame_end is not None else int(traj["frame_id_parsed"].max())
    traj = traj[(traj["frame_id_parsed"] >= frame_start) & (traj["frame_id_parsed"] <= frame_end)]
    traj = traj.sort_values("frame_id_parsed")
    if traj.empty:
        raise RuntimeError("traj_points_empty_after_range")

    traj_lookup: Dict[int, Point] = {}
    for _, row in traj.iterrows():
        fid = int(row["frame_id_parsed"])
        if fid not in traj_lookup:
            traj_lookup[fid] = row.geometry

    t_pose_velo = _build_velo_to_pose(data_root, args.camera)
    center_rows: List[dict] = []
    delta_rows: List[List[float]] = []
    raster_points: List[np.ndarray] = []
    raster_vals: List[np.ndarray] = []

    rng = np.random.default_rng(42)
    for fid in range(frame_start, frame_end + 1):
        frame_id = f"{fid:010d}"
        try:
            pose = load_kitti360_pose_full(data_root, args.drive, frame_id)
        except Exception:
            continue
        r_world_pose = _build_pose_rotation(pose[3], pose[4], pose[5])
        velo_center_pose = t_pose_velo[:3, 3]
        velo_center_world = r_world_pose @ velo_center_pose + np.array([pose[0], pose[1], pose[2]], dtype=float)
        center_rows.append(
            {
                "drive_id": args.drive,
                "frame_id": frame_id,
                "geometry": Point(float(velo_center_world[0]), float(velo_center_world[1])),
                "source": "lidar_center",
            }
        )
        traj_pt = traj_lookup.get(fid)
        if traj_pt is not None:
            dx = float(velo_center_world[0] - traj_pt.x)
            dy = float(velo_center_world[1] - traj_pt.y)
            dist = float(np.hypot(dx, dy))
            delta_rows.append([fid, dx, dy, dist])

        try:
            pts = load_kitti360_lidar_points(data_root, args.drive, frame_id)
        except Exception:
            continue
        if pts.size == 0:
            continue
        mask = (pts[:, 2] >= args.lidar_z_min) & (pts[:, 2] <= args.lidar_z_max)
        pts = pts[mask]
        if pts.shape[0] == 0:
            continue
        if pts.shape[0] > args.lidar_max_points_per_frame:
            idx = rng.choice(pts.shape[0], size=args.lidar_max_points_per_frame, replace=False)
            pts = pts[idx]
        pts_world = _transform_points_full_pose(pts, pose, t_pose_velo)
        raster_points.append(pts_world[:, :2])
        raster_vals.append(np.ones(pts_world.shape[0], dtype=float))

    center_gdf = gpd.GeoDataFrame(center_rows, geometry="geometry", crs="EPSG:32632")
    center_path = out_dir / "lidar_center_utm32.gpkg"
    if center_path.exists():
        center_path.unlink()
    center_gdf.to_file(center_path, layer="lidar_center", driver="GPKG")

    delta_path = out_dir / "lidar_center_delta.csv"
    delta_header = "frame_id,dx,dy,dist"
    delta_lines = [delta_header] + [",".join(map(str, row)) for row in delta_rows]
    delta_path.write_text("\n".join(delta_lines) + "\n", encoding="utf-8")

    raster_summary = {"points": 0}
    if raster_points:
        pts = np.vstack(raster_points)
        vals = np.concatenate(raster_vals)
        raster, bounds, res = _rasterize_points(pts, vals, args.lidar_grid_m, args.lidar_min_points)
        raster_summary["points"] = int(pts.shape[0])
        minx, miny, maxx, maxy = bounds
        transform = from_origin(minx, maxy + res, res, res)
        intensity_path = out_dir / f"lidar_intensity_utm32_fullpose_{args.drive}.tif"
        height_path = out_dir / f"lidar_height_utm32_fullpose_{args.drive}.tif"
        for path in [intensity_path, height_path]:
            if path.exists():
                path.unlink()
        with rasterio.open(
            intensity_path,
            "w",
            driver="GTiff",
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            crs="EPSG:32632",
            transform=transform,
            nodata=0.0,
        ) as dst:
            dst.write(raster, 1)
        height_vals = raster.copy()
        with rasterio.open(
            height_path,
            "w",
            driver="GTiff",
            height=height_vals.shape[0],
            width=height_vals.shape[1],
            count=1,
            dtype=height_vals.dtype,
            crs="EPSG:32632",
            transform=transform,
            nodata=0.0,
        ) as dst:
            dst.write(height_vals, 1)

    dx_vals = np.array([row[1] for row in delta_rows], dtype=float) if delta_rows else np.array([])
    dy_vals = np.array([row[2] for row in delta_rows], dtype=float) if delta_rows else np.array([])
    dist_vals = np.array([row[3] for row in delta_rows], dtype=float) if delta_rows else np.array([])
    stats = {
        "dx_p50": float(np.percentile(dx_vals, 50)) if dx_vals.size else 0.0,
        "dx_p90": float(np.percentile(dx_vals, 90)) if dx_vals.size else 0.0,
        "dy_p50": float(np.percentile(dy_vals, 50)) if dy_vals.size else 0.0,
        "dy_p90": float(np.percentile(dy_vals, 90)) if dy_vals.size else 0.0,
        "dist_p50": float(np.percentile(dist_vals, 50)) if dist_vals.size else 0.0,
        "dist_p90": float(np.percentile(dist_vals, 90)) if dist_vals.size else 0.0,
        "dx_std": float(np.std(dx_vals)) if dx_vals.size else 0.0,
        "dy_std": float(np.std(dy_vals)) if dy_vals.size else 0.0,
    }
    if dx_vals.size:
        summary = {
            "drive_id": args.drive,
            "frame_range": [frame_start, frame_end],
            "center_count": int(len(center_rows)),
            "delta_count": int(len(delta_rows)),
            "stats": stats,
            "raster_points": raster_summary.get("points", 0),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        (out_dir / "alignment_fullpose_summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    report = out_dir / "alignment_fullpose_report.md"
    lines = [
        "# Lidar vs Trajectory Alignment (Full Pose)",
        f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- drive_id: {args.drive}",
        f"- frame_range: {frame_start}-{frame_end}",
        f"- center_count: {len(center_rows)}",
        f"- delta_count: {len(delta_rows)}",
        f"- dx_p50: {stats['dx_p50']:.3f}",
        f"- dx_p90: {stats['dx_p90']:.3f}",
        f"- dy_p50: {stats['dy_p50']:.3f}",
        f"- dy_p90: {stats['dy_p90']:.3f}",
        f"- dist_p50: {stats['dist_p50']:.3f}",
        f"- dist_p90: {stats['dist_p90']:.3f}",
        f"- dx_std: {stats['dx_std']:.3f}",
        f"- dy_std: {stats['dy_std']:.3f}",
        f"- raster_points: {raster_summary.get('points', 0)}",
        "",
        "## Outputs",
        f"- lidar_center_utm32: {center_path}",
        f"- lidar_center_delta: {delta_path}",
        f"- lidar_height_fullpose: {out_dir / f'lidar_height_utm32_fullpose_{args.drive}.tif'}",
        f"- lidar_intensity_fullpose: {out_dir / f'lidar_intensity_utm32_fullpose_{args.drive}.tif'}",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
