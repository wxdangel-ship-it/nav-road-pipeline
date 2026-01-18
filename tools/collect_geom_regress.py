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


def _summary_from_qc(qc: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": entry.get("geom_run_id"),
        "drive": entry.get("drive"),
        "backend": entry.get("backend"),
        "internal_epsg": 32632,
        "road_bbox_dx_m": qc.get("road_bbox_dx_m"),
        "road_bbox_dy_m": qc.get("road_bbox_dy_m"),
        "road_bbox_diag_m": qc.get("road_bbox_diag_m"),
        "centerline_total_length_m": qc.get("centerline_total_length_m"),
        "road_component_count_before": qc.get("road_component_count_before"),
        "road_component_count_after": qc.get("road_component_count_after"),
        "intersections_count": qc.get("intersections_count"),
        "intersections_area_total_m2": qc.get("intersections_area_total_m2"),
        "width_median_m": qc.get("width_median_m"),
        "width_p95_m": qc.get("width_p95_m"),
    }


def _load_summary(entry: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    summary_path = entry.get("summary_path")
    if summary_path:
        data = _read_json(Path(summary_path))
        if data is not None:
            return data, None
    outputs_dir = entry.get("outputs_dir")
    if outputs_dir:
        qc = _read_json(Path(outputs_dir) / "qc.json")
        if qc is not None:
            return _summary_from_qc(qc, entry), "summary_from_qc"
    return None, "summary_missing"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "drive",
        "status",
        "reason",
        "geom_run_id",
        "backend",
        "internal_epsg",
        "road_bbox_dx_m",
        "road_bbox_dy_m",
        "road_bbox_diag_m",
        "centerline_total_length_m",
        "road_component_count_before",
        "road_component_count_after",
        "intersections_count",
        "intersections_area_total_m2",
        "width_median_m",
        "width_p95_m",
        "outputs_dir",
        "wgs84_road",
        "wgs84_centerlines",
        "wgs84_intersections",
        "crs_path",
        "summary_path",
        "note",
        "timestamp",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_md(path: Path, rows: List[Dict[str, Any]], run_id: str) -> None:
    total = len(rows)
    counts = {"PASS": 0, "SKIPPED": 0, "FAIL": 0}
    for r in rows:
        status = r.get("status") or "SKIPPED"
        counts[status] = counts.get(status, 0) + 1

    def _section(status: str) -> str:
        items = [r for r in rows if r.get("status") == status]
        if not items:
            return f"## {status}\n\n- None\n"
        lines = [
            f"## {status}",
            "",
            "| drive | backend | run_id | centerline_total_m | intersections | inter_area_m2 | road_diag_m | note |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in items:
            lines.append(
                "| {drive} | {backend} | {geom_run_id} | {centerline_total_length_m} | {intersections_count} | {intersections_area_total_m2} | {road_bbox_diag_m} | {note} |".format(
                    drive=r.get("drive") or "",
                    backend=r.get("backend") or "",
                    geom_run_id=r.get("geom_run_id") or "",
                    centerline_total_length_m=r.get("centerline_total_length_m") or "",
                    intersections_count=r.get("intersections_count") or "",
                    intersections_area_total_m2=r.get("intersections_area_total_m2") or "",
                    road_bbox_diag_m=r.get("road_bbox_diag_m") or "",
                    note=r.get("note") or "",
                )
            )
        lines.append("")
        return "\n".join(lines)

    content = [
        "# GeomRegress",
        "",
        f"- run_id: {run_id}",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- total: {total}",
        f"- pass: {counts.get('PASS', 0)}",
        f"- skipped: {counts.get('SKIPPED', 0)}",
        f"- fail: {counts.get('FAIL', 0)}",
        "",
        _section("PASS"),
        _section("SKIPPED"),
        _section("FAIL"),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regress-dir", required=True, help="runs/regress_geom_YYYYMMDD_HHMMSS")
    args = ap.parse_args()

    regress_dir = Path(args.regress_dir)
    index_path = regress_dir / "geom_index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"ERROR: geom_index.jsonl not found in {regress_dir}")

    rows: List[Dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        summary, note = _load_summary(entry)
        row = {
            "drive": entry.get("drive"),
            "status": entry.get("status"),
            "reason": entry.get("reason"),
            "geom_run_id": entry.get("geom_run_id"),
            "backend": entry.get("backend"),
            "outputs_dir": entry.get("outputs_dir"),
            "wgs84_road": entry.get("wgs84_road"),
            "wgs84_centerlines": entry.get("wgs84_centerlines"),
            "wgs84_intersections": entry.get("wgs84_intersections"),
            "crs_path": entry.get("crs_path"),
            "summary_path": entry.get("summary_path"),
            "note": note or "",
            "timestamp": entry.get("timestamp"),
        }
        if summary:
            row.update(summary)
        rows.append(row)

    run_id = regress_dir.name
    _write_csv(regress_dir / "GeomRegress.csv", rows)
    _write_md(regress_dir / "GeomRegress.md", rows, run_id)
    print(f"[GEOM-REGRESS] wrote {regress_dir / 'GeomRegress.md'}")
    print(f"[GEOM-REGRESS] wrote {regress_dir / 'GeomRegress.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
