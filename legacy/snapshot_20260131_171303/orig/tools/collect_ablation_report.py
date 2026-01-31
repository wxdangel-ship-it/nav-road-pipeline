from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_summary(outputs_dir: Optional[str]) -> Dict[str, Any]:
    if not outputs_dir:
        return {}
    summary = _read_json(Path(outputs_dir) / "GeomSummary.json")
    return summary or {}


def _flatten(entry: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    ref_osm = summary.get("ref_osm") or {}
    ref_dop20 = summary.get("ref_dop20") or {}
    return {
        "arm": entry.get("arm"),
        "drive": entry.get("drive"),
        "status": entry.get("status"),
        "reason": entry.get("reason"),
        "geom_run_id": entry.get("geom_run_id"),
        "outputs_dir": entry.get("outputs_dir"),
        "backend_used": summary.get("backend_used") or summary.get("backend"),
        "model_id": summary.get("model_id"),
        "camera": summary.get("camera"),
        "stride": summary.get("stride"),
        "internal_epsg": summary.get("internal_epsg"),
        "wgs84_epsg": summary.get("wgs84_epsg"),
        "bbox_dx_m": summary.get("bbox_dx_m") or summary.get("road_bbox_dx_m"),
        "bbox_dy_m": summary.get("bbox_dy_m") or summary.get("road_bbox_dy_m"),
        "bbox_diag_m": summary.get("bbox_diag_m") or summary.get("road_bbox_diag_m"),
        "centerline_total_len_m": summary.get("centerline_total_len_m") or summary.get("centerline_total_length_m"),
        "road_component_count_after": summary.get("road_component_count_after"),
        "intersections_count": summary.get("intersections_count"),
        "intersections_area_total_m2": summary.get("intersections_area_total_m2"),
        "centerlines_in_polygon_ratio": summary.get("centerlines_in_polygon_ratio"),
        "osm_present": ref_osm.get("osm_present"),
        "osm_match_ratio": ref_osm.get("osm_match_ratio"),
        "dist_to_osm_p50_m": ref_osm.get("dist_to_osm_p50_m"),
        "dist_to_osm_p95_m": ref_osm.get("dist_to_osm_p95_m"),
        "dop20_present": ref_dop20.get("dop20_present"),
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
    total = len(rows)
    counts = {}
    for r in rows:
        key = (r.get("arm") or "unknown", r.get("status") or "SKIPPED")
        counts[key] = counts.get(key, 0) + 1

    arms = sorted({r.get("arm") for r in rows if r.get("arm")})
    drives = sorted({r.get("drive") for r in rows if r.get("drive")})

    def _arm_table(arm: str) -> str:
        items = [r for r in rows if r.get("arm") == arm]
        lines = [
            f"## {arm}",
            "",
            "| drive | status | backend | centerline_total_m | intersections | inter_area_m2 | road_comp_after | osm_present | dop20_present |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for r in items:
            lines.append(
                "| {drive} | {status} | {backend_used} | {centerline_total_len_m} | {intersections_count} | {intersections_area_total_m2} | {road_component_count_after} | {osm_present} | {dop20_present} |".format(
                    drive=r.get("drive") or "",
                    status=r.get("status") or "",
                    backend_used=r.get("backend_used") or "",
                    centerline_total_len_m=r.get("centerline_total_len_m") or "",
                    intersections_count=r.get("intersections_count") or "",
                    intersections_area_total_m2=r.get("intersections_area_total_m2") or "",
                    road_component_count_after=r.get("road_component_count_after") or "",
                    osm_present=r.get("osm_present"),
                    dop20_present=r.get("dop20_present"),
                )
            )
        lines.append("")
        return "\n".join(lines)

    def _drive_table() -> str:
        lines = [
            "## Drive Comparison",
            "",
            "| drive | ArmA_center_m | ArmB_center_m | ArmC_center_m | ArmD_center_m | ArmA_intersections | ArmB_intersections | ArmC_intersections | ArmD_intersections |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for d in drives:
            row = {r["arm"]: r for r in rows if r.get("drive") == d}
            lines.append(
                "| {drive} | {a_center} | {b_center} | {c_center} | {d_center} | {a_inter} | {b_inter} | {c_inter} | {d_inter} |".format(
                    drive=d,
                    a_center=row.get("ArmA", {}).get("centerline_total_len_m", ""),
                    b_center=row.get("ArmB", {}).get("centerline_total_len_m", ""),
                    c_center=row.get("ArmC", {}).get("centerline_total_len_m", ""),
                    d_center=row.get("ArmD", {}).get("centerline_total_len_m", ""),
                    a_inter=row.get("ArmA", {}).get("intersections_count", ""),
                    b_inter=row.get("ArmB", {}).get("intersections_count", ""),
                    c_inter=row.get("ArmC", {}).get("intersections_count", ""),
                    d_inter=row.get("ArmD", {}).get("intersections_count", ""),
                )
            )
        lines.append("")
        return "\n".join(lines)

    summary_lines = [
        "# AblationRegress",
        "",
        f"- run_id: {run_id}",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- total: {total}",
        "",
        "## Summary",
        "",
        "| arm | pass | fail | skipped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in arms:
        pass_cnt = counts.get((arm, "PASS"), 0)
        fail_cnt = counts.get((arm, "FAIL"), 0)
        skip_cnt = counts.get((arm, "SKIPPED"), 0)
        summary_lines.append(f"| {arm} | {pass_cnt} | {fail_cnt} | {skip_cnt} |")
    summary_lines.append("")

    content = "\n".join(summary_lines + [_arm_table(a) for a in arms] + [_drive_table()])
    path.write_text(content, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regress-dir", required=True, help="runs/regress_ablation_YYYYMMDD_HHMMSS")
    args = ap.parse_args()

    regress_dir = Path(args.regress_dir)
    index_path = regress_dir / "ablation_index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"ERROR: ablation_index.jsonl not found in {regress_dir}")

    rows: List[Dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        summary = _load_summary(entry.get("outputs_dir"))
        rows.append(_flatten(entry, summary))

    _write_csv(regress_dir / "AblationRegress.csv", rows)
    _write_md(regress_dir / "AblationRegress.md", rows, regress_dir.name)
    print(f"[ABLATION] wrote {regress_dir / 'AblationRegress.md'}")
    print(f"[ABLATION] wrote {regress_dir / 'AblationRegress.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
