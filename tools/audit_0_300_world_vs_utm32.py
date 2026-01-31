from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text

LOG = logging.getLogger("audit_world_vs_utm32")

# =========================
# 参数区（按需修改）
# =========================
REPO_ROOT = r"E:\Work\nav-road-pipeline"
KITTI_ROOT = r"E:\KITTI360\KITTI-360"
DRIVE_ID = "2013_05_28_drive_0010_sync"
FRAME_START = 0
FRAME_END = 300
WORLD_LAZ_PATH = r""  # auto if empty
UTM32_LAZ_PATH = r""  # auto if empty


def _latest_run_with(name: str) -> Optional[Path]:
    runs = Path(REPO_ROOT) / "runs"
    candidates = []
    for p in runs.glob("lidar_fusion_0010_f000_300_*"):
        target = p / "outputs" / name
        if target.exists():
            candidates.append(target)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_missing_frames(run_dir: Path) -> Tuple[int, Dict[str, int], set]:
    missing_csv = run_dir / "outputs" / "missing_frames.csv"
    if not missing_csv.exists():
        return 0, {}, set()
    missing = {}
    frames = set()
    for line in missing_csv.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        frame_id, reason = parts
        frames.add(frame_id.strip())
        missing[reason] = missing.get(reason, 0) + 1
    return len(frames), missing, frames


def _read_header(path: Path) -> Dict[str, object]:
    with laspy.open(path) as reader:
        h = reader.header
        epsg = None
        try:
            crs = h.parse_crs()
            if crs is not None:
                epsg = crs.to_epsg()
        except Exception:
            epsg = None
        return {
            "path": str(path),
            "point_count": int(h.point_count),
            "scales": [float(x) for x in h.scales],
            "offsets": [float(x) for x in h.offsets],
            "mins": [float(x) for x in h.mins],
            "maxs": [float(x) for x in h.maxs],
            "epsg": epsg,
        }


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


def _bin_points_count(bin_path: Path) -> int:
    if not bin_path.exists():
        return 0
    size = bin_path.stat().st_size
    return int(size // 16)


def main() -> int:
    run_dir = Path(REPO_ROOT) / "runs" / f"audit_world_vs_utm32_0010_0_300_{now_ts()}"
    ensure_overwrite(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    world_path = Path(WORLD_LAZ_PATH) if WORLD_LAZ_PATH else _latest_run_with("fused_points_world.laz")
    utm_path = Path(UTM32_LAZ_PATH) if UTM32_LAZ_PATH else _latest_run_with("fused_points_utm32.laz")
    if world_path is None or utm_path is None:
        raise SystemExit("missing world/utm32 laz; set WORLD_LAZ_PATH/UTM32_LAZ_PATH")

    world_header = _read_header(world_path)
    utm_header = _read_header(utm_path)
    write_json(run_dir / "report" / "laz_headers.json", {"world": world_header, "utm32": utm_header})

    # raw bin points
    velodyne_dir = _find_velodyne_dir(Path(KITTI_ROOT), DRIVE_ID)
    total_points = 0
    per_frame_rows: List[Dict[str, object]] = []
    missing_bin = 0
    missing_bins = []
    for i in range(int(FRAME_START), int(FRAME_END) + 1):
        fid = f"{i:010d}"
        bin_path = velodyne_dir / f"{fid}.bin"
        if not bin_path.exists():
            missing_bin += 1
            if len(missing_bins) < 50:
                missing_bins.append(fid)
            pts = 0
        else:
            pts = _bin_points_count(bin_path)
        total_points += pts
        if i % 10 == 0:
            per_frame_rows.append({"frame": fid, "bin_points": pts, "expected_written": ""})

    raw_summary = {
        "frame_range": [FRAME_START, FRAME_END],
        "total_points": total_points,
        "avg_points": total_points / max(1, (FRAME_END - FRAME_START + 1 - missing_bin)),
        "missing_bin_frames": missing_bin,
        "missing_bin_sample": missing_bins,
    }
    write_json(run_dir / "report" / "raw_bin_points_sum.json", raw_summary)
    write_csv(run_dir / "report" / "per_frame_points_sample.csv", per_frame_rows, ["frame", "bin_points", "expected_written"])

    # missing frames from runs
    world_missing_count, world_missing_summary, _ = _load_missing_frames(world_path.parents[1])
    utm_missing_count, utm_missing_summary, _ = _load_missing_frames(utm_path.parents[1])
    frame_cov = {
        "world_missing_frames_unique": world_missing_count,
        "utm32_missing_frames_unique": utm_missing_count,
        "world_missing_summary": world_missing_summary,
        "utm32_missing_summary": utm_missing_summary,
    }
    write_json(run_dir / "report" / "frame_coverage.json", frame_cov)

    # quantization estimate
    world_scales = world_header["scales"]
    utm_scales = utm_header["scales"]
    quant = {
        "world_quant_err": [s / 2 for s in world_scales],
        "utm32_quant_err": [s / 2 for s in utm_scales],
        "world_scales": world_scales,
        "utm32_scales": utm_scales,
    }
    write_json(run_dir / "report" / "quantization_estimate.json", quant)

    world_count = world_header["point_count"]
    utm_count = utm_header["point_count"]
    case = "case_3"
    if world_count != utm_count:
        case = "case_1"
    else:
        scale_ratio = max(utm_scales) / max(world_scales) if max(world_scales) > 0 else 1.0
        if scale_ratio > 5.0:
            case = "case_2"

    summary = {
        "world_point_count": world_count,
        "utm32_point_count": utm_count,
        "raw_bin_total_points": total_points,
        "missing_bin_frames": missing_bin,
        "missing_pose_frames_unique": utm_missing_count,
        "world_scales": world_scales,
        "utm32_scales": utm_scales,
        "case": case,
    }
    write_json(run_dir / "report" / "compare_summary.json", summary)

    verdict = {
        "case_1": "真丢点/抽帧",
        "case_2": "量化精度变粗导致糊",
        "case_3": "更可能是显示抽稀",
    }[case]
    report_lines = [
        "# world vs utm32 compare (0-300)",
        f"- world_point_count: {world_count}",
        f"- utm32_point_count: {utm_count}",
        f"- raw_bin_total_points: {total_points} (missing_bin_frames={missing_bin})",
        f"- missing_pose_frames_unique: {utm_missing_count}",
        f"- world_scales: {world_scales}",
        f"- utm32_scales: {utm_scales}",
        f"- conclusion: {verdict} ({case})",
    ]
    write_text(run_dir / "report" / "compare_report.md", "\n".join(report_lines))

    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
