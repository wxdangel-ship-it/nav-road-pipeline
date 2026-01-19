from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import shape
from shapely.ops import unary_union

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


def _read_geojson(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", []) or []


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("N/A" if row.get(k) is None else row.get(k, "")) for k in fieldnames})


def _select_outputs(entries: List[dict], stage: str) -> Dict[str, Path]:
    picks: Dict[str, Path] = {}
    for entry in entries:
        if entry.get("stage") != stage:
            continue
        if entry.get("status") != "PASS":
            continue
        drive = entry.get("drive") or entry.get("tile_id")
        out_dir = entry.get("outputs_dir")
        if not drive or not out_dir:
            continue
        picks[drive] = Path(out_dir)
    return picks


def _compactness(area: float, perim: float) -> float:
    if area <= 0.0 or perim <= 0.0:
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


def _poly_metrics(poly, road_poly, center_union) -> Dict[str, float]:
    area = float(poly.area)
    perim = float(poly.length)
    compact = _compactness(area, perim)
    aspect = _aspect_ratio(poly)
    overlap = 0.0
    if road_poly is not None and not road_poly.is_empty:
        overlap = float(poly.intersection(road_poly).area) / max(1e-6, area)
    center_dist = float(poly.centroid.distance(center_union)) if center_union is not None else 0.0
    return {
        "area_m2": area,
        "compactness": compact,
        "aspect_ratio": aspect,
        "overlap_ratio": overlap,
        "centerline_dist_m": center_dist,
    }


def _iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def _max_iou(poly, geoms: List) -> float:
    best = 0.0
    for g in geoms:
        best = max(best, _iou(poly, g))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--stage", default="full")
    ap.add_argument("--sat-out-dir", required=True)
    ap.add_argument("--config", default="configs/intersections_hybrid.yaml")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--dup-iou", type=float, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config)).get("hybrid", {}) or {}
    dup_iou = args.dup_iou if args.dup_iou is not None else float(cfg.get("dup_iou", cfg.get("match_iou_min", 0.2)))
    min_overlap = float(cfg.get("min_overlap_ratio", 0.0))
    max_dist = float(cfg.get("max_centerline_dist_m", 0.0))
    min_area = float(cfg.get("min_area_m2", 0.0))
    max_area = float(cfg.get("max_area_m2", 0.0))
    min_compact = float(cfg.get("min_compactness", 0.0))
    max_aspect = float(cfg.get("max_aspect_ratio", 0.0))
    min_conf = float(cfg.get("sat_conf_min", 0.0))

    entries = _read_index(Path(args.index))
    drive_outputs = _select_outputs(entries, args.stage)

    sat_out_dir = Path(args.sat_out_dir)
    sat_global = _read_geojson(sat_out_dir / "intersections_sat.geojson")
    sat_global_by_drive: Dict[str, List[dict]] = defaultdict(list)
    for feat in sat_global:
        props = feat.get("properties") or {}
        drive = props.get("drive") or props.get("drive_id") or props.get("tile_id")
        if drive:
            sat_global_by_drive[str(drive)].append(feat)

    out_dir = Path(args.out_dir) if args.out_dir else sat_out_dir / "diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    summary_reasons = Counter()
    summary_counts = Counter()

    for drive, out_path in sorted(drive_outputs.items()):
        algo_path = out_path / "intersections_algo.geojson"
        if not algo_path.exists():
            algo_path = out_path / "intersections.geojson"
        road_path = out_path / "road_polygon.geojson"
        center_path = out_path / "centerlines.geojson"
        sat_path = out_path / "intersections_sat.geojson"
        final_path = out_path / "intersections_final.geojson"

        algo_feats = _read_geojson(algo_path)
        sat_feats = _read_geojson(sat_path) if sat_path.exists() else sat_global_by_drive.get(drive, [])
        road_feats = _read_geojson(road_path)
        center_feats = _read_geojson(center_path)
        final_feats = _read_geojson(final_path)

        algo_geoms = [shape(f.get("geometry")) for f in algo_feats if f.get("geometry")]
        sat_geoms = [shape(f.get("geometry")) for f in sat_feats if f.get("geometry")]
        sat_props = [f.get("properties") or {} for f in sat_feats if f.get("geometry")]

        road_poly = unary_union([shape(f.get("geometry")) for f in road_feats]) if road_feats else None
        center_geoms = [shape(f.get("geometry")) for f in center_feats if f.get("geometry")]
        center_union = unary_union(center_geoms) if center_geoms else None

        sat_dup_cnt = 0
        sat_unmatched_cnt = 0
        sat_pass_overlap_cnt = 0
        sat_pass_dist_cnt = 0
        sat_pass_conf_cnt = 0
        sat_kept_as_final_cnt = 0
        reject_counts = Counter()

        for feat in final_feats:
            props = feat.get("properties") or {}
            if props.get("src") == "sat":
                sat_kept_as_final_cnt += 1

        for geom, props in zip(sat_geoms, sat_props):
            if geom.is_empty:
                continue
            max_iou = _max_iou(geom, algo_geoms)
            if max_iou >= dup_iou:
                sat_dup_cnt += 1
                reject_counts["dedup"] += 1
                continue
            sat_unmatched_cnt += 1
            metrics = _poly_metrics(geom, road_poly, center_union)
            conf = props.get("sat_confidence")
            conf_val = float(conf) if isinstance(conf, (int, float)) else 0.0

            pass_conf = conf_val >= min_conf
            pass_overlap = metrics["overlap_ratio"] >= min_overlap
            pass_dist = metrics["centerline_dist_m"] <= max_dist if center_union is not None else True
            pass_shape = True
            if min_area and metrics["area_m2"] < min_area:
                pass_shape = False
            if max_area and metrics["area_m2"] > max_area:
                pass_shape = False
            if min_compact and metrics["compactness"] < min_compact:
                pass_shape = False
            if max_aspect and metrics["aspect_ratio"] > max_aspect:
                pass_shape = False

            if pass_conf:
                sat_pass_conf_cnt += 1
            if pass_overlap:
                sat_pass_overlap_cnt += 1
            if pass_dist:
                sat_pass_dist_cnt += 1

            if not pass_conf:
                reject_counts["conf"] += 1
            elif not pass_overlap:
                reject_counts["overlap"] += 1
            elif not pass_dist:
                reject_counts["dist"] += 1
            elif not pass_shape:
                reject_counts["shape"] += 1
            else:
                reject_counts["kept_candidate"] += 1

        top_reject_reason = ""
        if reject_counts:
            top_reject_reason = reject_counts.most_common(1)[0][0]

        row = {
            "drive_id": drive,
            "algo_cnt": len(algo_geoms),
            "sat_cnt": len(sat_geoms),
            "sat_dup_cnt": sat_dup_cnt,
            "sat_unmatched_cnt": sat_unmatched_cnt,
            "sat_pass_overlap_cnt": sat_pass_overlap_cnt,
            "sat_pass_dist_cnt": sat_pass_dist_cnt,
            "sat_pass_conf_cnt": sat_pass_conf_cnt,
            "sat_kept_as_final_cnt": sat_kept_as_final_cnt,
            "top_reject_reason": top_reject_reason,
        }
        rows.append(row)
        summary_counts.update(row)
        for k, v in reject_counts.items():
            summary_reasons[k] += v

    out_csv = out_dir / "hybrid_sat_diag_per_drive.csv"
    out_json = out_dir / "hybrid_sat_diag_per_drive.json"
    _write_csv(out_csv, rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "drives": len(rows),
        "reason_counts": dict(summary_reasons),
        "totals": {
            "algo_cnt": sum(r["algo_cnt"] for r in rows),
            "sat_cnt": sum(r["sat_cnt"] for r in rows),
            "sat_dup_cnt": sum(r["sat_dup_cnt"] for r in rows),
            "sat_unmatched_cnt": sum(r["sat_unmatched_cnt"] for r in rows),
            "sat_pass_overlap_cnt": sum(r["sat_pass_overlap_cnt"] for r in rows),
            "sat_pass_dist_cnt": sum(r["sat_pass_dist_cnt"] for r in rows),
            "sat_pass_conf_cnt": sum(r["sat_pass_conf_cnt"] for r in rows),
            "sat_kept_as_final_cnt": sum(r["sat_kept_as_final_cnt"] for r in rows),
        },
    }
    (out_dir / "hybrid_sat_diag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DIAG] wrote {out_csv} {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
