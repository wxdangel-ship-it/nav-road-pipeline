from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform as geom_transform
from pyproj import Transformer
import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_index(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def _read_drives(path: Path) -> List[str]:
    if not path.exists():
        return []
    drives = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        drives.append(line)
    return drives


def _read_geojson(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", []) or []


def _write_geojson(path: Path, features: List[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_wgs84(features: List[dict], crs_epsg: int) -> List[dict]:
    wgs84 = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)
    out = []
    for feat in features:
        geom = geom_transform(wgs84.transform, shape(feat["geometry"]))
        out.append({"type": "Feature", "geometry": mapping(geom), "properties": feat.get("properties") or {}})
    return out


def _compactness(area: float, perim: float) -> float:
    if area <= 0 or perim <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))


def _aspect_ratio(poly) -> float:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0
    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        edges.append(math.hypot(x2 - x1, y2 - y1))
    if not edges:
        return 0.0
    edges = sorted(edges)
    if edges[0] <= 0:
        return 0.0
    return float(edges[-1] / edges[0])


def _validate_poly(poly, road_poly, centerline_union, cfg: dict) -> Tuple[bool, dict]:
    if poly.is_empty:
        return False, {"reason": "empty"}
    area = float(poly.area)
    perim = float(poly.length)
    compact = _compactness(area, perim)
    aspect = _aspect_ratio(poly)
    overlap = 0.0
    if road_poly is not None and not road_poly.is_empty:
        overlap = float(poly.intersection(road_poly).area) / max(1e-6, area)
    center_dist = float(poly.centroid.distance(centerline_union)) if centerline_union is not None else 0.0
    ok = True
    if area < cfg["min_area_m2"] or area > cfg["max_area_m2"]:
        ok = False
    if compact < cfg["min_compactness"]:
        ok = False
    if aspect > cfg["max_aspect_ratio"]:
        ok = False
    if overlap < cfg["min_overlap_ratio"]:
        ok = False
    if center_dist > cfg["max_centerline_dist_m"]:
        ok = False
    return ok, {
        "area_m2": round(area, 3),
        "compactness": round(compact, 4),
        "aspect_ratio": round(aspect, 4),
        "overlap_ratio": round(overlap, 4),
        "centerline_dist_m": round(center_dist, 3),
    }


def _iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def _write_csv(path: Path, rows: List[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("N/A" if row.get(k) is None else row.get(k, "")) for k in fieldnames})


def _write_missing_reason_summary(out_csv: Path, expected_drives: List[str], report_type: str, run_id: str) -> dict:
    import csv
    from collections import Counter

    rows = []
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    counts = Counter()
    non_ok = []
    for row in rows:
        reason = (row.get("missing_reason") or "").strip()
        norm = "" if reason in {"", "N/A"} else reason
        counts[norm or "OK"] += 1
        if norm:
            non_ok.append({"drive_id": row.get("drive_id"), "reason": norm})
    payload = {
        "expected_drives": expected_drives,
        "missing_reason_counts": dict(counts),
        "non_ok_drives": non_ok,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "report_type": report_type,
    }
    json_path = out_csv.with_name(out_csv.stem + "_missing_reason_summary.json")
    md_path = out_csv.with_name(out_csv.stem + "_missing_reason_summary.md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Missing Reason Summary",
        "",
        f"- report_type: {report_type}",
        f"- run_id: {run_id}",
        f"- generated_at: {payload['generated_at']}",
        "",
        "## expected_drives",
        "```json",
        json.dumps(expected_drives, ensure_ascii=False, indent=2),
        "```",
        "",
        "## missing_reason_counts",
        "```json",
        json.dumps(payload["missing_reason_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## non_ok_drives",
        "```json",
        json.dumps(non_ok, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="postopt_index.jsonl or similar")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--config", default="configs/intersections_hybrid.yaml")
    ap.add_argument("--out-dir", default="", help="report output dir")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config)).get("hybrid", {}) or {}
    expected_drives = _read_drives(Path(cfg.get("expected_drives_file", "configs/golden_drives.txt")))

    entries = _read_index(Path(args.index))
    entries = [
        e for e in entries
        if e.get("stage") == args.stage
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]
    if args.candidate:
        entries = [e for e in entries if e.get("candidate_id") == args.candidate]

    per_drive_rows = []
    seen_drives = set()

    for entry in entries:
        drive = str(entry.get("drive") or entry.get("tile_id") or "")
        out_dir = Path(entry.get("outputs_dir"))
        if not drive or not out_dir.exists():
            continue
        seen_drives.add(drive)

        final_path = out_dir / "intersections_final.geojson"
        if args.resume and final_path.exists():
            continue

        algo_path = out_dir / "intersections_algo.geojson"
        if not algo_path.exists():
            algo_path = out_dir / "intersections.geojson"
        sat_path = out_dir / "intersections_sat.geojson"
        road_path = out_dir / "road_polygon.geojson"
        center_path = out_dir / "centerlines.geojson"

        if not algo_path.exists() or not road_path.exists() or not center_path.exists():
            per_drive_rows.append(
                {
                    "drive_id": drive,
                    "tile_id": drive,
                    "status": "FAIL",
                    "missing_reason": "missing_inputs",
                    "final_count": 0,
                }
            )
            continue

        algo_feats = _read_geojson(algo_path)
        sat_feats = _read_geojson(sat_path) if sat_path.exists() else []
        road_feats = _read_geojson(road_path)
        center_feats = _read_geojson(center_path)

        road_poly = None
        if road_feats:
            road_poly = unary_union([shape(f.get("geometry")) for f in road_feats])
        centerlines = [shape(f.get("geometry")) for f in center_feats if f.get("geometry")]
        center_union = unary_union(centerlines) if centerlines else None

        algo_polys = [(shape(f.get("geometry")), f.get("properties") or {}) for f in algo_feats]
        sat_polys = [(shape(f.get("geometry")), f.get("properties") or {}) for f in sat_feats]

        sat_used = set()
        final_features = []

        for algo_geom, algo_props in algo_polys:
            if algo_geom.is_empty:
                continue
            best_iou = 0.0
            best_idx = -1
            for i, (sat_geom, _) in enumerate(sat_polys):
                score = _iou(algo_geom, sat_geom)
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            algo_ok, algo_qc = _validate_poly(algo_geom, road_poly, center_union, cfg)
            sat_geom = None
            sat_props = None
            sat_ok = False
            if best_idx >= 0 and best_iou >= cfg["match_iou_min"]:
                sat_geom, sat_props = sat_polys[best_idx]
                sat_ok, _ = _validate_poly(sat_geom, road_poly, center_union, cfg)
                sat_conf = sat_props.get("sat_confidence")
                if isinstance(sat_conf, (int, float)) and sat_conf < cfg.get("sat_conf_min", 0.0):
                    sat_ok = False

            chosen = None
            reason = "invalid_filtered"
            src = "none"
            conf = None

            if algo_ok and cfg.get("prefer_algo", True):
                chosen = algo_geom
                reason = "algo_keep"
                src = "algo"
            elif sat_ok:
                chosen = sat_geom
                reason = "sat_fallback"
                src = "sat"
            elif algo_ok:
                chosen = algo_geom
                reason = "algo_keep"
                src = "algo"
            elif sat_ok:
                chosen = sat_geom
                reason = "sat_fallback"
                src = "sat"
            elif cfg.get("union_enabled", False) and best_idx >= 0 and best_iou >= cfg["union_iou_min"]:
                union_geom = algo_geom.union(sat_geom)
                union_ok, _ = _validate_poly(union_geom, road_poly, center_union, cfg)
                if union_ok:
                    chosen = union_geom
                    reason = "union_merge"
                    src = "union"

            if chosen is not None:
                if src == "sat" and sat_props:
                    conf = sat_props.get("sat_confidence")
                if src == "union" and sat_props:
                    conf = sat_props.get("sat_confidence")
                props = {
                    "drive_id": drive,
                    "tile_id": drive,
                    "src": src,
                    "reason": reason,
                    "conf": conf if conf is not None else "N/A",
                }
                props.update(algo_qc)
                final_features.append({"type": "Feature", "geometry": mapping(chosen), "properties": props})
                if best_idx >= 0:
                    sat_used.add(best_idx)

        for i, (sat_geom, sat_props) in enumerate(sat_polys):
            if i in sat_used:
                continue
            sat_ok, sat_qc = _validate_poly(sat_geom, road_poly, center_union, cfg)
            sat_conf = sat_props.get("sat_confidence")
            if isinstance(sat_conf, (int, float)) and sat_conf < cfg.get("sat_conf_min", 0.0):
                sat_ok = False
            if not sat_ok:
                continue
            props = {
                "drive_id": drive,
                "tile_id": drive,
                "src": "sat",
                "reason": "sat_only",
                "conf": sat_props.get("sat_confidence", "N/A"),
            }
            props.update(sat_qc)
            final_features.append({"type": "Feature", "geometry": mapping(sat_geom), "properties": props})

        _write_geojson(final_path, final_features)
        _write_geojson(out_dir / "intersections_final_wgs84.geojson", _to_wgs84(final_features, int(cfg["crs_epsg"])))

        missing_reason = "OK" if final_features else "hybrid_no_valid"
        per_drive_rows.append(
            {
                "drive_id": drive,
                "tile_id": drive,
                "status": "OK" if final_features else "EMPTY",
                "missing_reason": missing_reason,
                "final_count": len(final_features),
            }
        )

    for d in expected_drives:
        if d in seen_drives:
            continue
        per_drive_rows.append(
            {
                "drive_id": d,
                "tile_id": d,
                "status": "FAIL",
                "missing_reason": "missing_entry",
                "final_count": 0,
            }
        )

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(entries[0]["outputs_dir"]).parents[1] if entries else Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = out_dir / f"{args.stage}_hybrid_report_per_drive.csv"
    report_json = out_dir / f"{args.stage}_hybrid_report_per_drive.json"
    _write_csv(report_csv, per_drive_rows)
    report_json.write_text(json.dumps(per_drive_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_missing_reason_summary(report_csv, expected_drives, args.stage, out_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
