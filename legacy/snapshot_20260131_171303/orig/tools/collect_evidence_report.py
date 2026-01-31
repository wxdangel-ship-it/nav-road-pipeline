from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _latest_entries(lines: List[str]) -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        key = (entry.get("arm_id"), entry.get("drive"))
        ts = entry.get("timestamp") or ""
        if key not in latest or ts >= (latest[key].get("timestamp") or ""):
            latest[key] = entry
    return list(latest.values())


def _flatten(entry: Dict[str, Any]) -> Dict[str, Any]:
    base_out = entry.get("base_outputs_dir") or ""
    summary = _read_json(Path(base_out) / "GeomSummary.json") if base_out else {}
    osm_metrics = _read_json(Path(entry.get("osm_metrics_path"))) if entry.get("osm_metrics_path") else {}
    dop20_layers = _read_json(Path(entry.get("dop20_layers_path"))) if entry.get("dop20_layers_path") else {}

    return {
        "arm": entry.get("arm_id"),
        "drive": entry.get("drive"),
        "status": entry.get("status"),
        "reason": entry.get("reason"),
        "base_geom_run_id": entry.get("base_geom_run_id"),
        "base_outputs_dir": base_out,
        "backend_used": summary.get("backend_used"),
        "model_id": summary.get("model_id"),
        "camera": summary.get("camera"),
        "stride": summary.get("stride"),
        "bbox_dx_m": summary.get("bbox_dx_m"),
        "bbox_dy_m": summary.get("bbox_dy_m"),
        "bbox_diag_m": summary.get("bbox_diag_m"),
        "centerline_total_len_m": summary.get("centerline_total_len_m"),
        "road_component_count_after": summary.get("road_component_count_after"),
        "intersections_count": summary.get("intersections_count"),
        "intersections_area_total_m2": summary.get("intersections_area_total_m2"),
        "centerlines_in_polygon_ratio": summary.get("centerlines_in_polygon_ratio"),
        "osm_present": osm_metrics.get("osm_present"),
        "coverage_ok": osm_metrics.get("coverage_ok"),
        "metrics_valid": osm_metrics.get("metrics_valid"),
        "osm_source": osm_metrics.get("osm_source"),
        "match_ratio": osm_metrics.get("match_ratio"),
        "dist_p50_m": osm_metrics.get("dist_p50_m"),
        "dist_p95_m": osm_metrics.get("dist_p95_m"),
        "dop20_present": dop20_layers.get("dop20_present"),
        "dop20_tiles_dir": dop20_layers.get("tiles_dir"),
        "timestamp": entry.get("timestamp"),
    }


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


def _write_md(path: Path, rows: List[Dict[str, Any]], run_id: str) -> None:
    arms = sorted({r.get("arm") for r in rows if r.get("arm")})
    status_counts: Dict[str, Dict[str, int]] = {a: {"PASS": 0, "FAIL": 0, "SKIPPED": 0} for a in arms}
    for r in rows:
        status = r.get("status") or "SKIPPED"
        arm = r.get("arm")
        if arm in status_counts:
            status_counts[arm][status] += 1

    def _table(status: str) -> str:
        items = [r for r in rows if r.get("status") == status]
        if not items:
            return f"## {status}\n\n- None\n"
        lines = [
            f"## {status}",
            "",
            "| arm | drive | centerline_total_m | intersections | inter_area_m2 | road_comp_after | coverage_ok | match_ratio | dist_p95_m | dop20_present |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in items:
            lines.append(
                "| {arm} | {drive} | {center} | {inter} | {area} | {road} | {cov} | {osm} | {p95} | {dop} |".format(
                    arm=r.get("arm") or "",
                    drive=r.get("drive") or "",
                    center=r.get("centerline_total_len_m") or "",
                    inter=r.get("intersections_count") or "",
                    area=r.get("intersections_area_total_m2") or "",
                    road=r.get("road_component_count_after") or "",
                    cov=r.get("coverage_ok") if r.get("coverage_ok") is not None else "",
                    osm=r.get("match_ratio") if r.get("match_ratio") is not None else "",
                    p95=r.get("dist_p95_m") if r.get("dist_p95_m") is not None else "",
                    dop=r.get("dop20_present") if r.get("dop20_present") is not None else "",
                )
            )
        lines.append("")
        return "\n".join(lines)

    lines = [
        "# EvidenceRegress",
        "",
        f"- run_id: {run_id}",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        "| arm | pass | fail | skipped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in arms:
        counts = status_counts[arm]
        lines.append(f"| {arm} | {counts['PASS']} | {counts['FAIL']} | {counts['SKIPPED']} |")
    lines.append("")
    lines.append(_table("PASS"))
    lines.append(_table("FAIL"))
    lines.append(_table("SKIPPED"))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regress-dir", required=True)
    args = ap.parse_args()

    regress_dir = Path(args.regress_dir)
    index_path = regress_dir / "evidence_index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"ERROR: evidence_index.jsonl not found in {regress_dir}")

    entries = _latest_entries(index_path.read_text(encoding="utf-8").splitlines())
    rows = [_flatten(e) for e in entries]

    _write_csv(regress_dir / "EvidenceRegress.csv", rows)
    _write_md(regress_dir / "EvidenceRegress.md", rows, regress_dir.name)
    print(f"[EVIDENCE] wrote {regress_dir / 'EvidenceRegress.md'}")
    print(f"[EVIDENCE] wrote {regress_dir / 'EvidenceRegress.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
