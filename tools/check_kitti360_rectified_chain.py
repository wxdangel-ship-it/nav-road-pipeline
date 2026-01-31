from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.datasets.kitti360_io import (
    load_kitti360_calib,
    load_kitti360_cam_to_pose,
    load_kitti360_lidar_points,
)
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text

LOG = logging.getLogger("rect_chain_check")

# =========================
# 参数区（按需修改）
# =========================
KITTI_ROOT = r"E:\KITTI360\KITTI-360"
DRIVE_ID = "2013_05_28_drive_0010_sync"
FRAME_START = 0
FRAME_END = 300
SAMPLE_FRAMES_STRIDE = 10
POINTS_PER_FRAME = 20000


def _find_pose_file(data_root: Path, drive_id: str, names: List[str]) -> Path:
    base_dirs = [
        data_root / "data_poses" / drive_id,
        data_root / "data_poses" / drive_id / "poses",
        data_root / "data_poses" / drive_id / "pose",
    ]
    for base in base_dirs:
        if not base.exists():
            continue
        for name in names:
            cand = base / name
            if cand.exists():
                return cand
    raise FileNotFoundError(f"missing_pose_file:{names}")


def _parse_pose_map(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        parts = [p for p in line.strip().split() if p]
        if not parts:
            continue
        frame_id = None
        nums = parts
        if len(parts) in {13, 17}:
            frame_id = parts[0]
            nums = parts[1:]
        elif len(parts) in {12, 16}:
            frame_id = f"{idx:010d}"
        else:
            continue
        try:
            vals = np.array([float(v) for v in nums], dtype=float)
        except ValueError:
            continue
        if vals.size == 12:
            mat = vals.reshape(3, 4)
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
            mat = np.vstack([mat, bottom])
        elif vals.size == 16:
            mat = vals.reshape(4, 4)
        else:
            continue
        key = str(frame_id)
        out[key] = mat
        if key.isdigit():
            out[f"{int(key):010d}"] = mat
    return out


def _sample_points(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    if points.shape[0] <= n:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=n, replace=False)
    return points[idx]


def _apply_transform(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)])
    out = (t @ pts_h.T).T
    return out[:, :3]


def _rot_err_deg(r_ref: np.ndarray, r_est: np.ndarray) -> float:
    r_rel = r_ref.T @ r_est
    trace = float(np.trace(r_rel))
    val = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(math.degrees(math.acos(val)))


def main() -> int:
    data_root = Path(KITTI_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root_missing:{data_root}")

    run_dir = Path("runs") / f"rect_chain_check_0010_0_300_{now_ts()}"
    ensure_overwrite(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    calib = load_kitti360_calib(data_root, "image_00")
    t_cam0_to_velo = calib["t_cam_to_velo"]
    t_velo_to_cam0 = np.linalg.inv(t_cam0_to_velo)
    r_rect = calib.get("r_rect")
    if r_rect is None:
        raise FileNotFoundError("missing_r_rect_00")
    t_rect = np.eye(4, dtype=float)
    t_rect[:3, :3] = r_rect
    t_rect_inv = np.linalg.inv(t_rect)

    t_cam0_to_pose = load_kitti360_cam_to_pose(data_root, "image_00")

    cam0_to_world_path = _find_pose_file(data_root, DRIVE_ID, ["cam0_to_world.txt"])
    poses_path = _find_pose_file(data_root, DRIVE_ID, ["poses.txt"])
    cam0_to_world = _parse_pose_map(cam0_to_world_path)
    poses_map = _parse_pose_map(poses_path)

    frame_ids = [
        f"{i:010d}"
        for i in range(int(FRAME_START), int(FRAME_END) + 1, max(1, int(SAMPLE_FRAMES_STRIDE)))
    ]

    rect_rows: List[Dict[str, object]] = []
    pose_rows: List[Dict[str, object]] = []
    rms_list: List[float] = []
    max_list: List[float] = []
    trans_list: List[float] = []
    rot_list: List[float] = []

    for fid in frame_ids:
        if fid not in cam0_to_world:
            LOG.warning("missing_cam0_to_world_frame:%s", fid)
            continue
        try:
            raw = load_kitti360_lidar_points(data_root, DRIVE_ID, fid)
        except Exception as exc:
            LOG.warning("missing_velodyne:%s:%s", fid, exc)
            continue
        if raw.size == 0:
            continue
        pts = _sample_points(raw[:, :3].astype(np.float64), POINTS_PER_FRAME, seed=int(fid))
        t_rectcam0_to_world = cam0_to_world[fid]

        t_chain_a = t_rectcam0_to_world @ t_velo_to_cam0
        t_chain_b = t_rectcam0_to_world @ t_rect @ t_velo_to_cam0
        pa = _apply_transform(pts, t_chain_a)
        pb = _apply_transform(pts, t_chain_b)
        diff = pa - pb
        norms = np.linalg.norm(diff, axis=1)
        rms = float(np.sqrt(np.mean(norms**2)))
        mx = float(np.max(norms))
        rect_rows.append({"frame": fid, "rms_AB_m": rms, "max_AB_m": mx, "n_points": int(pts.shape[0])})
        rms_list.append(rms)
        max_list.append(mx)

        if fid in poses_map:
            t_imu_to_world = poses_map[fid]
            t_rectcam0_to_world_est = t_imu_to_world @ (t_cam0_to_pose @ t_rect_inv)
            t_ref = t_rectcam0_to_world
            trans_err = float(np.linalg.norm(t_ref[:3, 3] - t_rectcam0_to_world_est[:3, 3]))
            rot_err = _rot_err_deg(t_ref[:3, :3], t_rectcam0_to_world_est[:3, :3])
            pose_rows.append({"frame": fid, "trans_err_m": trans_err, "rot_err_deg": rot_err})
            trans_list.append(trans_err)
            rot_list.append(rot_err)

    if not rect_rows:
        raise RuntimeError("no_valid_frames_for_rect_check")

    median_rms = float(np.median(rms_list)) if rms_list else 0.0
    max_rms = float(np.max(rms_list)) if rms_list else 0.0
    median_trans = float(np.median(trans_list)) if trans_list else 0.0
    median_rot = float(np.median(rot_list)) if rot_list else 0.0

    rect_consistent = median_rms <= 0.02 and max_rms <= 0.05
    poses_ok = median_trans < 0.2 and median_rot < 0.5
    use_r_rect = not rect_consistent

    report = {
        "drive_id": DRIVE_ID,
        "frame_range": [FRAME_START, FRAME_END],
        "sample_stride": SAMPLE_FRAMES_STRIDE,
        "points_per_frame": POINTS_PER_FRAME,
        "frames_checked": len(rect_rows),
        "rect_diff": {
            "median_rms_AB_m": median_rms,
            "max_rms_AB_m": max_rms,
            "max_max_AB_m": float(np.max(max_list)) if max_list else 0.0,
        },
        "pose_compare": {
            "frames_compared": len(pose_rows),
            "median_trans_err_m": median_trans,
            "median_rot_err_deg": median_rot,
        },
        "recommendation": {
            "use_r_rect_with_cam0_to_world": bool(use_r_rect),
            "poses_fallback_allowed": bool(poses_ok),
        },
        "paths": {
            "cam0_to_world": str(cam0_to_world_path),
            "poses": str(poses_path),
        },
    }

    summary_lines = [
        "# KITTI-360 rectified chain check",
        "",
        f"- drive_id: {DRIVE_ID}",
        f"- frame_range: {FRAME_START}-{FRAME_END}",
        f"- sample_stride: {SAMPLE_FRAMES_STRIDE}",
        f"- points_per_frame: {POINTS_PER_FRAME}",
        f"- frames_checked: {len(rect_rows)}",
        f"- rect_diff median_rms_AB_m: {median_rms:.4f}",
        f"- rect_diff max_rms_AB_m: {max_rms:.4f}",
        f"- pose_compare median_trans_err_m: {median_trans:.4f}",
        f"- pose_compare median_rot_err_deg: {median_rot:.4f}",
        f"- recommend use_r_rect_with_cam0_to_world: {use_r_rect}",
        f"- recommend poses_fallback_allowed: {poses_ok}",
    ]

    write_csv(run_dir / "report" / "per_frame_rect_diff.csv", rect_rows, ["frame", "rms_AB_m", "max_AB_m", "n_points"])
    write_csv(run_dir / "report" / "pose_compare.csv", pose_rows, ["frame", "trans_err_m", "rot_err_deg"])
    write_json(run_dir / "report" / "rect_chain_report.json", report)
    write_text(run_dir / "report" / "rect_chain_summary.md", "\n".join(summary_lines))

    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
