from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _rows_from_index(index_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        summary_path = entry.get("summary_path")
        summary = _read_json(Path(summary_path)) if summary_path else None
        row = dict(entry)
        if summary:
            row.update(summary)
        rows.append(row)
    return rows


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "drive": row.get("drive"),
        "backend_used": row.get("backend_used") or row.get("backend"),
        "internal_epsg": _to_int(row.get("internal_epsg") or row.get("crs_epsg")),
        "centerline_total_len_m": _to_float(
            row.get("centerline_total_len_m") or row.get("centerline_total_length_m")
        ),
        "intersections_count": _to_int(row.get("intersections_count")),
        "intersections_area_total_m2": _to_float(row.get("intersections_area_total_m2")),
        "width_median_m": _to_float(row.get("width_median_m")),
        "width_p95_m": _to_float(row.get("width_p95_m")),
        "road_component_count_after": _to_int(row.get("road_component_count_after")),
        "centerlines_in_polygon_ratio": _to_float(row.get("centerlines_in_polygon_ratio")),
    }


def _write_yaml(path: Path, items: List[Dict[str, Any]]) -> None:
    payload = {"drives": items}
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Update geom regression baseline.")
    ap.add_argument("--regress-dir", default="", help="runs/regress_geom_YYYYMMDD_HHMMSS")
    ap.add_argument("--csv", default="", help="GeomRegress.csv path")
    ap.add_argument(
        "--index-file",
        default="geom_index.jsonl",
        help="index filename under regress-dir",
    )
    ap.add_argument(
        "--out",
        default="configs/geom_regress_baseline.yaml",
        help="output baseline yaml",
    )
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    csv_path = Path(args.csv) if args.csv else None
    if csv_path and csv_path.exists():
        rows = _rows_from_csv(csv_path)
    else:
        if not args.regress_dir:
            raise SystemExit("ERROR: provide --csv or --regress-dir.")
        regress_dir = Path(args.regress_dir)
        index_path = regress_dir / args.index_file
        if not index_path.exists():
            raise SystemExit(f"ERROR: index not found: {index_path}")
        rows = _rows_from_index(index_path)

    items: List[Dict[str, Any]] = []
    for row in rows:
        status = row.get("status")
        if status and status != "PASS":
            continue
        item = _normalize_row(row)
        drive = item.get("drive")
        if not drive:
            continue
        items.append(item)

    items.sort(key=lambda r: str(r.get("drive")))
    out_path = Path(args.out)
    _write_yaml(out_path, items)
    print(f"[GEOM-BASELINE] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
