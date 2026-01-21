from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _count_src(geojson_path: Path) -> Dict[str, int]:
    if not geojson_path.exists():
        return {}
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    counts: Dict[str, int] = {}
    for feat in data.get("features", []) or []:
        src = (feat.get("properties") or {}).get("src", "")
        counts[src] = counts.get(src, 0) + 1
    return counts


def _score(row: dict) -> float:
    kept = row.get("sat_kept_as_final_cnt", 0) or 0
    mean_overlap = row.get("sat_add_mean_overlap")
    p95_dist = row.get("sat_add_p95_dist")
    mean_overlap = float(mean_overlap) if mean_overlap is not None else 0.0
    p95_dist = float(p95_dist) if p95_dist is not None else 0.0
    return float(kept) * mean_overlap / (1.0 + p95_dist)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-root", default="")
    args = ap.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else Path("runs") / f"hybrid_augment_sweep_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    configs_dir = out_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(Path(args.base_config))
    base_hybrid = base_cfg.get("hybrid", {}) or {}

    grid_overlap = [0.10, 0.20, 0.30]
    grid_dist = [20.0, 35.0, 50.0]

    rows: List[dict] = []
    run_id = 0

    for min_overlap in grid_overlap:
        for max_dist in grid_dist:
            run_id += 1
            cfg = {"hybrid": dict(base_hybrid)}
            cfg["hybrid"]["sat_augment_min_overlap"] = float(min_overlap)
            cfg["hybrid"]["sat_augment_max_dist"] = float(max_dist)
            cfg_name = f"augment_o{int(min_overlap*100):02d}_d{int(max_dist):02d}.yaml"
            cfg_path = configs_dir / cfg_name
            _write_yaml(cfg_path, cfg)

            run_dir = out_root / f"run_{run_id:02d}_o{int(min_overlap*100):02d}_d{int(max_dist):02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            merged_dir = run_dir / "merged"
            diag_dir = run_dir / "diag"

            cmd_hybrid = [
                "cmd",
                "/c",
                "scripts\\intersections_hybrid.cmd",
                "--index",
                args.index,
                "--stage",
                "full",
                "--config",
                str(cfg_path),
                "--out-dir",
                str(run_dir),
            ]
            subprocess.run(cmd_hybrid, check=True)

            cmd_merge = [
                sys.executable,
                "tools\\merge_geom_outputs.py",
                "--index",
                args.index,
                "--best-postopt",
                "runs\\sweep_geom_postopt_20260119_061421\\best_postopt.yaml",
                "--out-dir",
                str(merged_dir),
            ]
            subprocess.run(cmd_merge, check=True)

            cmd_diag = [
                sys.executable,
                "tools\\diagnose_hybrid_sat_usage.py",
                "--index",
                args.index,
                "--stage",
                "full",
                "--sat-out-dir",
                "runs\\sat_intersections_full_golden8\\outputs",
                "--config",
                str(cfg_path),
                "--out-dir",
                str(diag_dir),
            ]
            subprocess.run(cmd_diag, check=True)

            diag_summary_path = diag_dir / "hybrid_sat_diag_summary.json"
            diag_summary = json.loads(diag_summary_path.read_text(encoding="utf-8"))
            totals = diag_summary.get("totals", {})
            drive_count = int(diag_summary.get("drives", 0) or 0)
            sat_unmatched = int(totals.get("sat_unmatched_cnt") or 0)
            merged_final_path = merged_dir / "merged_intersections_final.geojson"
            src_counts = _count_src(merged_final_path)

            row = {
                "run_id": run_dir.name,
                "sat_augment_min_overlap": min_overlap,
                "sat_augment_max_dist": max_dist,
                "sat_augment_min_conf": base_hybrid.get("sat_augment_min_conf"),
                "dup_iou": base_hybrid.get("dup_iou"),
                "sat_kept_as_final_cnt": int(totals.get("sat_kept_as_final_cnt") or 0),
                "sat_kept_per_drive": round(
                    float(totals.get("sat_kept_as_final_cnt") or 0) / max(1, drive_count), 3
                ),
                "sat_pass_overlap_ratio": round(
                    float(totals.get("sat_pass_overlap_cnt") or 0) / max(1, sat_unmatched), 4
                ),
                "sat_pass_dist_ratio": round(
                    float(totals.get("sat_pass_dist_cnt") or 0) / max(1, sat_unmatched), 4
                ),
                "sat_pass_conf_ratio": round(
                    float(totals.get("sat_pass_conf_cnt") or 0) / max(1, sat_unmatched), 4
                ),
                "sat_add_mean_overlap": totals.get("sat_add_mean_overlap"),
                "sat_add_p95_dist": totals.get("sat_add_p95_dist"),
                "merged_src_algo": src_counts.get("algo", 0),
                "merged_src_sat": src_counts.get("sat", 0),
                "merged_src_union": src_counts.get("union", 0),
                "merged_final_cnt": sum(src_counts.values()),
            }
            row["score"] = round(_score(row), 6)
            rows.append(row)

    rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)
    summary_path = out_root / "summary.csv"
    if rows:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print(f"[SWEEP] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
