from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_baseline(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in data.get("drives", []) or []:
        drive = item.get("drive")
        if drive:
            out[str(drive)] = item
    return out


def _read_index(path: Path, stage: str | None) -> List[Dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if stage and entry.get("stage") != stage:
            continue
        items.append(entry)
    return items


def _score_entry(
    entry: Dict[str, Any],
    baseline: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
) -> Tuple[str, Optional[float], str]:
    if entry.get("status") != "PASS":
        return "SKIPPED", None, entry.get("reason") or "status_not_pass"

    outputs_dir = entry.get("outputs_dir")
    if not outputs_dir:
        return "SKIPPED", None, "missing_outputs_dir"

    summary = _read_json(Path(outputs_dir) / "GeomSummary.json")
    if not summary:
        return "SKIPPED", None, "missing_summary"

    drive = entry.get("drive") or ""
    base = baseline.get(drive) or {}
    if base:
        base_ratio = base.get("centerlines_in_polygon_ratio")
        ratio = summary.get("centerlines_in_polygon_ratio")
        if isinstance(base_ratio, (int, float)) and isinstance(ratio, (int, float)):
            if ratio < float(base_ratio) - 0.02:
                return "FAIL", None, "centerlines_ratio_drop"

    if summary.get("road_component_count_after") != 1:
        return "FAIL", None, "road_components_not_1"

    osm = _read_json(Path(outputs_dir) / "osm_ref_metrics.json")
    if not osm:
        return "SKIPPED", None, "missing_osm_metrics"
    if not osm.get("metrics_valid"):
        return "SKIPPED", None, "osm_metrics_invalid"

    match_ratio = osm.get("match_ratio")
    dist_p95 = osm.get("dist_p95_m")
    if match_ratio is None or dist_p95 is None:
        return "SKIPPED", None, "missing_osm_fields"

    score = (
        float(match_ratio) * weights.get("osm_match_ratio", 1.0)
        + float(dist_p95) * weights.get("dist_p95_m", -0.05)
    )
    return "PASS", float(score), ""


def _aggregate(
    entries: List[Dict[str, Any]],
    baseline: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        cid = entry.get("candidate_id") or "unknown"
        by_candidate.setdefault(cid, []).append(entry)

    candidate_rows: List[Dict[str, Any]] = []
    drive_rows: List[Dict[str, Any]] = []

    for cid, items in sorted(by_candidate.items()):
        pass_count = fail_count = skipped_count = 0
        scores: List[float] = []
        reasons: List[str] = []
        for entry in items:
            status, score, reason = _score_entry(entry, baseline, weights)
            if status == "PASS":
                pass_count += 1
                if score is not None:
                    scores.append(score)
            elif status == "FAIL":
                fail_count += 1
                if reason:
                    reasons.append(reason)
            else:
                skipped_count += 1
                if reason:
                    reasons.append(reason)
            drive_rows.append(
                {
                    "candidate_id": cid,
                    "drive": entry.get("drive"),
                    "status": status,
                    "score": score,
                    "reason": reason,
                    "outputs_dir": entry.get("outputs_dir"),
                }
            )

        if fail_count > 0:
            status = "FAIL"
        elif pass_count == 0:
            status = "SKIPPED"
        else:
            status = "PASS"

        avg_score = sum(scores) / len(scores) if scores else None
        candidate_rows.append(
            {
                "candidate_id": cid,
                "status": status,
                "avg_score": avg_score,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "skipped_count": skipped_count,
                "reasons": ";".join(sorted(set(reasons))),
            }
        )

    return candidate_rows, drive_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_md(path: Path, candidates: List[Dict[str, Any]], drives: List[Dict[str, Any]]) -> None:
    total = len(candidates)
    counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for row in candidates:
        status = row.get("status")
        if status in counts:
            counts[status] += 1

    lines = [
        "# Geom RoadSeg Score",
        "",
        f"- total_candidates: {total}",
        f"- pass: {counts['PASS']}",
        f"- fail: {counts['FAIL']}",
        f"- skipped: {counts['SKIPPED']}",
        "",
        "## Candidates",
        "",
        "| candidate_id | status | avg_score | pass | fail | skipped | reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in candidates:
        lines.append(
            "| {cid} | {status} | {score} | {p} | {f} | {s} | {r} |".format(
                cid=row.get("candidate_id") or "",
                status=row.get("status") or "",
                score="" if row.get("avg_score") is None else f"{row.get('avg_score'):.4f}",
                p=row.get("pass_count") or 0,
                f=row.get("fail_count") or 0,
                s=row.get("skipped_count") or 0,
                r=row.get("reasons") or "",
            )
        )
    lines.append("")
    lines.append("## Per-Drive")
    lines.append("")
    lines.append("| candidate_id | drive | status | score | reason |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for row in drives:
        score = row.get("score")
        lines.append(
            "| {cid} | {drive} | {status} | {score} | {reason} |".format(
                cid=row.get("candidate_id") or "",
                drive=row.get("drive") or "",
                status=row.get("status") or "",
                score="" if score is None else f"{score:.4f}",
                reason=row.get("reason") or "",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="sweep_index.jsonl")
    ap.add_argument("--baseline", default="configs/geom_regress_baseline.yaml")
    ap.add_argument("--stage", default="", help="optional stage filter (quick/full)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", default="", help="optional json ranking output")
    ap.add_argument("--weight-match", type=float, default=1.0)
    ap.add_argument("--weight-dist-p95", type=float, default=-0.05)
    args = ap.parse_args()

    weights = {"osm_match_ratio": args.weight_match, "dist_p95_m": args.weight_dist_p95}
    index_path = Path(args.index)
    entries = _read_index(index_path, args.stage or None)
    baseline = _load_baseline(Path(args.baseline))
    candidates, drives = _aggregate(entries, baseline, weights)

    _write_csv(Path(args.out_csv), candidates)
    _write_md(Path(args.out_md), candidates, drives)

    if args.out_json:
        ranked = sorted(
            candidates,
            key=lambda r: (r.get("status") != "PASS", -(r.get("avg_score") or -1e9)),
        )
        Path(args.out_json).write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
