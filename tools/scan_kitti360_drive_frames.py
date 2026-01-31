from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text

LOG = logging.getLogger("drive_scan")

# =========================
# 参数区（按需修改）
# =========================
KITTI_ROOT = r"E:\KITTI360\KITTI-360"
DRIVE_ID = "2013_05_28_drive_0010_sync"


def _find_velodyne_dir(data_root: Path, drive_id: str) -> Path:
    candidates = [
        data_root / "data_3d_raw" / drive_id / "velodyne_points" / "data",
        data_root / "data_3d_raw" / drive_id / "velodyne_points" / "data" / "1",
        data_root / drive_id / "velodyne_points" / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"missing_velodyne_dir:{drive_id}")


def _frame_id_from_name(name: str) -> int:
    stem = Path(name).stem
    if stem.isdigit():
        return int(stem)
    return -1


def _sample_points_count(paths: List[Path], max_samples: int = 5) -> List[int]:
    if not paths:
        return []
    picks = [paths[0]]
    if len(paths) > 1:
        picks.append(paths[len(paths) // 2])
    if len(paths) > 2:
        picks.append(paths[-1])
    if len(paths) > 3:
        picks.append(paths[len(paths) // 4])
    if len(paths) > 4:
        picks.append(paths[(len(paths) * 3) // 4])
    picks = picks[:max_samples]
    counts: List[int] = []
    for p in picks:
        raw = np.fromfile(str(p), dtype=np.float32)
        counts.append(int(raw.size // 4))
    return counts


def main() -> int:
    data_root = Path(KITTI_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root_missing:{data_root}")

    run_dir = Path("runs") / f"drive_scan_0010_{now_ts()}"
    ensure_overwrite(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    velodyne_dir = _find_velodyne_dir(data_root, DRIVE_ID)
    files = sorted(velodyne_dir.glob("*.bin"))
    frame_ids = [(_frame_id_from_name(p.name), p) for p in files]
    frame_ids = [(fid, p) for fid, p in frame_ids if fid >= 0]
    frame_ids.sort(key=lambda x: x[0])
    if not frame_ids:
        raise RuntimeError("no_velodyne_frames_found")

    ids = [fid for fid, _ in frame_ids]
    min_frame = min(ids)
    max_frame = max(ids)
    frame_count = len(ids)
    expected = list(range(min_frame, max_frame + 1))
    missing = sorted(set(expected) - set(ids))
    gap_count = len(missing)

    # sample list (first/last 50)
    sample_rows: List[Dict[str, object]] = []
    for fid in ids[:50]:
        sample_rows.append({"frame_id": f"{fid:010d}"})
    for fid in ids[-50:]:
        sample_rows.append({"frame_id": f"{fid:010d}"})

    counts = _sample_points_count([p for _, p in frame_ids])
    avg_points = float(np.mean(counts)) if counts else 0.0
    est_total_points = int(avg_points * frame_count)
    est_laz_mb = float(est_total_points * 16 / (1024 * 1024)) if est_total_points > 0 else 0.0

    report = {
        "drive_id": DRIVE_ID,
        "min_frame": int(min_frame),
        "max_frame": int(max_frame),
        "frame_count": int(frame_count),
        "missing_frames_count": int(gap_count),
        "missing_frames_sample": [f"{fid:010d}" for fid in missing[:50]],
        "sample_points_counts": counts,
        "avg_points_per_frame_est": avg_points,
        "estimated_total_points": est_total_points,
        "estimated_laz_mb": est_laz_mb,
        "velodyne_dir": str(velodyne_dir),
    }

    summary = (
        f"drive={DRIVE_ID} frames={frame_count} range={min_frame}-{max_frame} "
        f"missing={gap_count} est_total_points={est_total_points} est_laz_mb={est_laz_mb:.1f}"
    )

    write_json(run_dir / "report" / "drive_frames_report.json", report)
    write_csv(run_dir / "report" / "drive_frames_list_sample.csv", sample_rows, ["frame_id"])
    write_text(run_dir / "report" / "drive_frames_summary.md", summary)

    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
