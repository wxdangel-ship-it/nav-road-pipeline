from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_lidar_param_sweep")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _parse_list(values: str, cast) -> List:
    items = [v.strip() for v in values.split(",") if v.strip()]
    return [cast(v) for v in items]


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--provider", default="pc_open3dis_v1")
    ap.add_argument("--base-zoo", default="configs/lidar_model_zoo.yaml")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--road-root", default="")
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--grid-sizes", default="2,4,6")
    ap.add_argument("--min-areas", default="0.5,1.0")
    ap.add_argument("--max-runs", type=int, default=4)
    args = ap.parse_args()

    log = _setup_logger()
    out_root = Path(args.out_root) if args.out_root else Path("runs") / f"lidar_param_sweep_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_root.mkdir(parents=True, exist_ok=True)

    base = _load_yaml(Path(args.base_zoo))
    models = base.get("models") or []
    target = None
    for model in models:
        if model.get("model_id") == args.provider:
            target = model
            break
    if target is None:
        log.error("provider not found in zoo: %s", args.provider)
        return 2

    grid_sizes = _parse_list(args.grid_sizes, float)
    min_areas = _parse_list(args.min_areas, float)
    combos: List[Tuple[float, float]] = []
    for g in grid_sizes:
        for a in min_areas:
            combos.append((g, a))
    combos = combos[: max(1, args.max_runs)]

    summary_rows = []
    for grid_size, min_area in combos:
        tag = f"grid{grid_size}_area{min_area}".replace(".", "p")
        run_dir = out_root / f"run_{tag}"
        zoo_path = out_root / f"zoo_{tag}.yaml"

        tuned = json.loads(json.dumps(base))
        for model in tuned.get("models", []):
            if model.get("model_id") == args.provider:
                model["grid_cell_size_m"] = grid_size
                model["min_area_m2"] = min_area
        _write_yaml(zoo_path, tuned)

        log.info("running %s (grid=%s area=%s)", tag, grid_size, min_area)
        subprocess.check_call(
            [
                sys.executable,
                "tools/run_lidar_providers.py",
                "--index",
                args.index,
                "--providers",
                args.provider,
                "--zoo",
                str(zoo_path),
                "--out-run",
                str(run_dir),
            ]
        )
        report_path = run_dir / "report.md"
        report_json = run_dir / "report.json"
        subprocess.check_call(
            [
                sys.executable,
                "tools/ab_eval_lidar_evidence.py",
                "--run-dir",
                str(run_dir),
                "--provider",
                args.provider,
                "--road-root",
                args.road_root,
                "--out",
                str(report_path),
                "--out-json",
                str(report_json),
                "--frame-stride",
                str(args.frame_stride),
            ]
        )

        payload = json.loads(report_json.read_text(encoding="utf-8"))
        stability = payload.get("stability_summary", {})
        frames_p50_vals = []
        collision_total = 0
        avg_inst_pf = []
        for cls, stat in stability.items():
            vals = stat.get("frames_hit") or []
            frames_p50_vals.extend(vals)
            collision_total += int(stat.get("collision_count") or 0)
            avg_inst_pf.extend(stat.get("avg_inst_pf") or [])
        summary_rows.append(
            {
                "tag": tag,
                "grid_cell_size_m": grid_size,
                "min_area_m2": min_area,
                "frames_hit_p50": _median(frames_p50_vals),
                "collision_count": collision_total,
                "avg_instances_per_frame": _median(avg_inst_pf),
                "run_dir": str(run_dir),
            }
        )

    def _score(row: dict) -> float:
        return (
            float(row.get("frames_hit_p50", 0.0))
            - 0.05 * float(row.get("avg_instances_per_frame", 0.0))
            - 0.01 * float(row.get("collision_count", 0.0))
        )

    summary_rows = sorted(summary_rows, key=_score, reverse=True)
    best = summary_rows[0] if summary_rows else None

    report_lines = [
        "# LiDAR Param Sweep Report",
        "",
        f"- index: {args.index}",
        f"- provider: {args.provider}",
        "",
        "## Candidates",
    ]
    for row in summary_rows:
        report_lines.append(
            f"- {row['tag']}: grid={row['grid_cell_size_m']} area={row['min_area_m2']} "
            f"frames_hit_p50={row['frames_hit_p50']} avg_inst_pf={row['avg_instances_per_frame']} "
            f"collision_count={row['collision_count']} run_dir={row['run_dir']}"
        )

    report_lines.append("")
    report_lines.append("## Recommendation")
    if best:
        report_lines.append(
            f"- recommended: grid_cell_size_m={best['grid_cell_size_m']} min_area_m2={best['min_area_m2']} "
            f"(frames_hit_p50={best['frames_hit_p50']}, avg_inst_pf={best['avg_instances_per_frame']}, "
            f"collision_count={best['collision_count']})"
        )
        risk = []
        if best["frames_hit_p50"] < 2:
            risk.append("frames_hit_p50<2 (possible fragmentation)")
        if best["collision_count"] > 0:
            risk.append("collision_count>0 (possible over-merge)")
        if risk:
            report_lines.append(f"- risk: {', '.join(risk)}")
    else:
        report_lines.append("- no candidates")

    out_report = out_root / "sweep_report.md"
    out_report.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("wrote sweep report: %s", out_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
