from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_json, write_text

LOG = logging.getLogger("density_audit")


def _read_point_count(path: Path) -> int:
    with laspy.open(path) as reader:
        return int(reader.header.point_count)


def _load_index(path: Path) -> List[Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = data.get("parts", [])
    out = []
    for part in parts:
        p = part.get("path")
        if p:
            out.append(Path(p))
    return out


def _find_metrics(base: Path) -> Optional[Path]:
    cand = base / "report" / "metrics.json"
    if cand.exists():
        return cand
    for meta in base.glob("outputs/*meta.json"):
        return meta
    return None


def _summary_decision(missing_ratio: Optional[float], points_ratio: Optional[float]) -> str:
    if missing_ratio is not None and missing_ratio > 0.1:
        return "真抽帧"
    if points_ratio is not None and points_ratio < 0.7:
        return "真抽点"
    return "显示抽稀"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laz", type=str, default="", help="single laz file")
    parser.add_argument("--index", type=str, default="", help="index json with parts")
    parser.add_argument("--scan-report", type=str, default="", help="drive_frames_report.json")
    args = parser.parse_args()

    laz_path = Path(args.laz) if args.laz else None
    index_path = Path(args.index) if args.index else None
    if not laz_path and not index_path:
        raise SystemExit("need --laz or --index")

    run_dir = Path("runs") / f"density_audit_{now_ts()}"
    ensure_overwrite(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    parts: List[Path] = []
    if index_path:
        parts = _load_index(index_path)
    elif laz_path:
        parts = [laz_path]

    if not parts:
        raise RuntimeError("no_parts_found")

    total_points = 0
    total_size = 0
    part_info: List[Dict[str, object]] = []
    for p in parts:
        if not p.exists():
            continue
        pts = _read_point_count(p)
        size = p.stat().st_size
        part_info.append({"path": str(p), "points": pts, "size": size})
        total_points += pts
        total_size += size

    expected_total_points = None
    if args.scan_report:
        scan = json.loads(Path(args.scan_report).read_text(encoding="utf-8"))
        expected_total_points = scan.get("estimated_total_points")

    base = parts[0].parents[1] if len(parts[0].parents) > 1 else parts[0].parent
    metrics_path = _find_metrics(base)
    metrics = None
    if metrics_path and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    missing_ratio = None
    points_ratio = None
    frame_cov = {}
    if metrics:
        total_frames = metrics.get("total_frames")
        missing_frames = metrics.get("missing_frames")
        stride = metrics.get("stride")
        frame_cov = {
            "total_frames": total_frames,
            "missing_frames": missing_frames,
            "stride": stride,
        }
        if total_frames:
            missing_ratio = float(missing_frames or 0) / float(total_frames)
    if expected_total_points:
        points_ratio = float(total_points) / float(expected_total_points)

    conclusion = _summary_decision(missing_ratio, points_ratio)
    report = {
        "total_points": total_points,
        "total_size": total_size,
        "parts": part_info,
        "expected_total_points": expected_total_points,
        "points_ratio": points_ratio,
        "missing_ratio": missing_ratio,
        "conclusion": conclusion,
    }

    write_json(run_dir / "report" / "density_audit.json", report)
    write_text(run_dir / "report" / "density_summary.md", f"结论：{conclusion}")
    if frame_cov:
        write_json(run_dir / "report" / "frame_coverage.json", frame_cov)

    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
