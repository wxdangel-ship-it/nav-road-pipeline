from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml


def _read_index(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def _load_best_candidate(path: Path) -> str:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(data.get("candidate_id") or "")


def _read_summary(outputs_dir: Path) -> Dict[str, object]:
    summary_path = outputs_dir / "GeomSummary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


    center_frames = []
    road_frames = []
    inter_frames = []
    inter_algo_frames = []
    inter_sat_frames = []

    for entry in entries:
        outputs_dir = Path(entry["outputs_dir"])
        drive = entry.get("drive") or ""
        run_id = entry.get("geom_run_id") or ""
        candidate_id = entry.get("candidate_id") or ""
        summary = _read_summary(outputs_dir)
        backend_used = summary.get("intersection_backend_used") or summary.get("backend_used") or ""
        sat_present = bool(summary.get("sat_present"))
        sat_conf = summary.get("sat_confidence_avg")


    if center_frames:
    if road_frames:
    if inter_frames:
    if inter_algo_frames:
    if inter_sat_frames:


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="postopt_index.jsonl")
    ap.add_argument("--best-postopt", required=True, help="best_postopt.yaml")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"ERROR: index not found: {index_path}")

    if not best_candidate:
        raise SystemExit("ERROR: best_postopt candidate_id not found")

    entries = _read_index(index_path)
    entries = [
        e for e in entries
        if e.get("stage") == "full"
        and e.get("candidate_id") == best_candidate
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]
    if not entries:
        raise SystemExit("ERROR: no full entries found for best candidate")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
