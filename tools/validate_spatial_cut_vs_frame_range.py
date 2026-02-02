from __future__ import annotations

"""
验证 spatial_cut vs frame_range 分块的边界覆盖度。
输出 coverage_ratio / uplift_vs_old / no_dup_check。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

# =========================
# 参数区（按需修改）
# =========================
KITTI_ROOT = r"E:\KITTI360\KITTI-360"
DRIVE_ID = "2013_05_28_drive_0010_sync"
FRAME_START = 0
FRAME_END = 300
RUN_DIR_A = r""  # 旧：frame_range
RUN_DIR_B = r""  # 新：spatial_cut
RADIUS_M = 80.0


def _now_ts() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_index(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "outputs" / "fused_points_utm32_index.json"
    if not path.exists():
        raise FileNotFoundError(f"index_missing:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_meta(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "outputs" / "fused_points_utm32.meta.json"
    if not path.exists():
        raise FileNotFoundError(f"meta_missing:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _count_points_in_circle(laz_path: Path, center: Tuple[float, float], radius: float) -> int:
    import laspy

    cx, cy = center
    r2 = radius * radius
    total = 0
    with laspy.open(laz_path) as reader:
        for chunk in reader.chunk_iterator(1_000_000):
            xs = np.asarray(chunk.x, dtype=np.float64)
            ys = np.asarray(chunk.y, dtype=np.float64)
            mask = (xs >= cx - radius) & (xs <= cx + radius) & (ys >= cy - radius) & (ys <= cy + radius)
            if not np.any(mask):
                continue
            xs = xs[mask]
            ys = ys[mask]
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            total += int(np.sum(d2 <= r2))
    return total


def _count_points_in_forward_window(
    laz_path: Path,
    center: Tuple[float, float],
    radius: float,
    forward_vec: Tuple[float, float],
) -> int:
    import laspy

    cx, cy = center
    vx, vy = forward_vec
    r2 = radius * radius
    total = 0
    with laspy.open(laz_path) as reader:
        for chunk in reader.chunk_iterator(1_000_000):
            xs = np.asarray(chunk.x, dtype=np.float64)
            ys = np.asarray(chunk.y, dtype=np.float64)
            dx = xs - cx
            dy = ys - cy
            mask = (dx * dx + dy * dy) <= r2
            if not np.any(mask):
                continue
            dx = dx[mask]
            dy = dy[mask]
            dot = dx * vx + dy * vy
            total += int(np.sum(dot >= 0.0))
    return total


def _parse_pose_map(path: Path) -> Dict[str, np.ndarray]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 13:
            continue
        frame = parts[0]
        vals = np.asarray([float(v) for v in parts[1:13]], dtype=np.float64).reshape(3, 4)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = vals
        out[frame] = mat
        if frame.isdigit():
            out[f"{int(frame):010d}"] = mat
    return out


def _apply_world_to_utm32(points_w: np.ndarray, tf: Dict[str, float]) -> np.ndarray:
    yaw = math.radians(float(tf.get("yaw_deg", 0.0)))
    scale = float(tf.get("scale", 1.0))
    dx = float(tf.get("dx", 0.0))
    dy = float(tf.get("dy", 0.0))
    dz = float(tf.get("dz", 0.0))
    c = math.cos(yaw)
    s = math.sin(yaw)
    x = points_w[:, 0]
    y = points_w[:, 1]
    z = points_w[:, 2]
    x2 = scale * (c * x - s * y) + dx
    y2 = scale * (s * x + c * y) + dy
    z2 = z + dz
    return np.stack([x2, y2, z2], axis=1)


def _load_transform(meta: Dict[str, object]) -> Dict[str, float]:
    tf = meta.get("transform") or {}
    return {
        "dx": float(tf.get("dx", 0.0)),
        "dy": float(tf.get("dy", 0.0)),
        "dz": float(tf.get("dz", 0.0)),
        "yaw_deg": float(tf.get("yaw_deg", 0.0)),
        "scale": float(tf.get("scale", 1.0)),
    }


def _find_part_for_frame(parts: List[Dict[str, object]], frame_id: int) -> int:
    for idx, p in enumerate(parts):
        fs = p.get("frame_start")
        fe = p.get("frame_end")
        if fs is None or fe is None:
            continue
        if int(fs) <= frame_id <= int(fe):
            return idx
    return -1


def main() -> int:
    if not RUN_DIR_A or not RUN_DIR_B:
        raise SystemExit("RUN_DIR_A/RUN_DIR_B required")
    run_a = Path(RUN_DIR_A)
    run_b = Path(RUN_DIR_B)
    index_a = _load_index(run_a)
    index_b = _load_index(run_b)
    meta_b = _load_meta(run_b)

    boundary_frames = index_b.get("boundary_frames") or []
    if not boundary_frames:
        raise RuntimeError("boundary_frames_missing_in_runB")
    parts_a = index_a.get("parts") or []
    parts_b = index_b.get("parts") or []

    pose_path = Path(KITTI_ROOT) / "data_poses" / DRIVE_ID / "cam0_to_world.txt"
    pose_map = _parse_pose_map(pose_path)
    tf = _load_transform(meta_b)

    rows = []
    for j, bf in enumerate(boundary_frames):
        if bf not in pose_map:
            continue
        mat = pose_map[bf]
        pos_w = mat[:3, 3].reshape(1, 3)
        pos_u = _apply_world_to_utm32(pos_w, tf)[0]
        center = (float(pos_u[0]), float(pos_u[1]))

        # forward vector (traj tangent)
        fid_int = int(bf)
        prev_id = str(fid_int - 1)
        next_id = str(fid_int + 1)
        if prev_id in pose_map and next_id in pose_map:
            p0 = _apply_world_to_utm32(pose_map[prev_id][:3, 3].reshape(1, 3), tf)[0]
            p1 = _apply_world_to_utm32(pose_map[next_id][:3, 3].reshape(1, 3), tf)[0]
        elif next_id in pose_map:
            p0 = _apply_world_to_utm32(pose_map[bf][:3, 3].reshape(1, 3), tf)[0]
            p1 = _apply_world_to_utm32(pose_map[next_id][:3, 3].reshape(1, 3), tf)[0]
        elif prev_id in pose_map:
            p0 = _apply_world_to_utm32(pose_map[prev_id][:3, 3].reshape(1, 3), tf)[0]
            p1 = _apply_world_to_utm32(pose_map[bf][:3, 3].reshape(1, 3), tf)[0]
        else:
            p0 = pos_u
            p1 = pos_u + np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v = p1[:2] - p0[:2]
        n = float(np.hypot(v[0], v[1]))
        if n <= 0:
            v = np.array([1.0, 0.0], dtype=np.float64)
        else:
            v = v / n

        frame_int = int(bf)
        part_a = _find_part_for_frame(parts_a, frame_int)
        part_b = j + 1
        if part_a < 0 or part_b >= len(parts_b):
            continue
        path_a = Path(parts_a[part_a]["path"])
        path_a_prev = Path(parts_a[part_a - 1]["path"]) if part_a - 1 >= 0 else None
        path_b = Path(parts_b[part_b]["path"])

        count_a = _count_points_in_circle(path_a, center, RADIUS_M)
        count_a_union = count_a
        if path_a_prev is not None:
            count_a_union += _count_points_in_circle(path_a_prev, center, RADIUS_M)
        count_b = _count_points_in_circle(path_b, center, RADIUS_M)

        forward_a = _count_points_in_forward_window(path_a, center, RADIUS_M, (float(v[0]), float(v[1])))
        forward_union = forward_a
        if path_a_prev is not None:
            forward_union += _count_points_in_forward_window(
                path_a_prev, center, RADIUS_M, (float(v[0]), float(v[1]))
            )
        forward_b = _count_points_in_forward_window(path_b, center, RADIUS_M, (float(v[0]), float(v[1])))

        coverage_ratio = count_b / max(count_a_union, 1)
        uplift = count_b / max(count_a, 1)
        coverage_forward = forward_b / max(forward_union, 1)
        uplift_forward = forward_b / max(forward_a, 1)
        rows.append(
            {
                "boundary_frame": bf,
                "part_a": part_a,
                "part_b": part_b,
                "runA_start_only": int(count_a),
                "runA_union": int(count_a_union),
                "runB_start_only": int(count_b),
                "coverage_ratio": float(coverage_ratio),
                "uplift_vs_old": float(uplift),
                "forward_expected": int(forward_union),
                "forward_actual": int(forward_b),
                "coverage_ratio_forward_halfplane": float(coverage_forward),
                "uplift_vs_old_forward": float(uplift_forward),
                "forward_pass": bool(coverage_forward >= 0.90),
            }
        )

    total_written = int(meta_b.get("written_points", 0))
    sum_parts = int(sum(int(p.get("point_count", 0)) for p in parts_b))
    no_dup_check = (sum_parts == total_written)

    out_dir = RUNS_DIR / f"spatial_cut_validation_{_now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "report" / "spatial_cut_validation.csv"
    json_path = out_dir / "report" / "spatial_cut_validation.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        import csv

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "run_dir_a": str(run_a),
        "run_dir_b": str(run_b),
        "boundaries": len(rows),
        "no_dup_check": no_dup_check,
        "sum_parts": sum_parts,
        "written_points": total_written,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"validation_done: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
