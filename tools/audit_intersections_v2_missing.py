from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon, shape, mapping
from shapely.ops import unary_union, transform as geom_transform
from pyproj import Transformer


def _read_geojson(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", []) or []


def _write_geojson(path: Path, features: List[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_index(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def _is_wgs84_coords(x: float, y: float) -> bool:
    return abs(x) <= 180 and abs(y) <= 90


def _to_utm32(geom):
    wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    return geom_transform(wgs84.transform, geom)


def _to_wgs84(geom):
    utm32 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    return geom_transform(utm32.transform, geom)


def _collect_lines(features: List[dict]) -> List[LineString]:
    lines = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shape(geom)
        if shp.is_empty:
            continue
        if shp.geom_type == "LineString":
            lines.append(shp)
        elif shp.geom_type == "MultiLineString":
            lines.extend(list(shp.geoms))
    return lines


def _collect_polys(features: List[dict]) -> List[Polygon]:
    polys = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shape(geom)
        if shp.is_empty:
            continue
        if shp.geom_type == "Polygon":
            polys.append(shp)
        elif shp.geom_type == "MultiPolygon":
            polys.extend(list(shp.geoms))
    return polys


def _snap_point(point: Point, nodes: List[Tuple[float, float]], grid: Dict[Tuple[int, int], List[int]], snap_m: float) -> int:
    if snap_m <= 0:
        nodes.append((point.x, point.y))
        return len(nodes) - 1
    cell = int(point.x // snap_m), int(point.y // snap_m)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            idxs = grid.get((cell[0] + dx, cell[1] + dy), [])
            for idx in idxs:
                x, y = nodes[idx]
                if point.distance(Point(x, y)) <= snap_m:
                    return idx
    nodes.append((point.x, point.y))
    idx = len(nodes) - 1
    grid.setdefault(cell, []).append(idx)
    return idx


def _osm_degree_junctions(lines: List[LineString], snap_m: float, min_degree: int) -> List[Point]:
    nodes: List[Tuple[float, float]] = []
    grid: Dict[Tuple[int, int], List[int]] = {}
    edges: Dict[int, set] = {}
    for line in lines:
        if line.is_empty or len(line.coords) < 2:
            continue
        p1 = Point(line.coords[0])
        p2 = Point(line.coords[-1])
        n1 = _snap_point(p1, nodes, grid, snap_m)
        n2 = _snap_point(p2, nodes, grid, snap_m)
        if n1 == n2:
            continue
        edges.setdefault(n1, set()).add(n2)
        edges.setdefault(n2, set()).add(n1)
    points = []
    for idx, neighbors in edges.items():
        if len(neighbors) >= min_degree:
            x, y = nodes[idx]
            points.append(Point(x, y))
    return points


def _entry_bbox_wgs84(entry: dict) -> Optional[Tuple[float, float, float, float]]:
    bbox = entry.get("bbox_wgs84") or entry.get("bbox4326")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return None


def _assign_sat_features_to_drives(
    sat_feats: List[dict],
    entries: List[dict],
    road_bboxes: Dict[str, Tuple[float, float, float, float]],
) -> Tuple[Dict[str, List[dict]], dict]:
    by_drive: Dict[str, List[dict]] = {}
    unmatched = 0
    for feat in sat_feats:
        geom = feat.get("geometry")
        if not geom:
            unmatched += 1
            continue
        shp = shape(geom)
        if shp.is_empty:
            unmatched += 1
            continue
        c = shp.centroid
        matched = None
        for entry in entries:
            drive = str(entry.get("drive") or "")
            bbox = _entry_bbox_wgs84(entry) or road_bboxes.get(drive)
            if bbox is None:
                continue
            minx, miny, maxx, maxy = bbox
            if minx <= c.x <= maxx and miny <= c.y <= maxy:
                matched = drive
                break
        if matched is None:
            unmatched += 1
            continue
        by_drive.setdefault(matched, []).append(feat)
    stats = {
        "sat_total": len(sat_feats),
        "sat_matched": sum(len(v) for v in by_drive.values()),
        "sat_unmatched": unmatched,
    }
    return by_drive, stats


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", required=True, help="runs/intersections_v2_*")
    ap.add_argument("--index", required=True, help="postopt_index.jsonl")
    ap.add_argument("--osm-path", required=True, help="OSM roads geojson")
    ap.add_argument("--radius-m", type=float, default=20.0)
    ap.add_argument("--sat-path", default="runs/sat_intersections_full_golden8/outputs/intersections_sat_wgs84.geojson")
    ap.add_argument("--osm-snap-m", type=float, default=2.0)
    ap.add_argument("--osm-min-degree", type=int, default=3)
    args = ap.parse_args()

    v2_dir = Path(args.v2_dir)
    outputs_dir = v2_dir / "outputs"
    audit_dir = outputs_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    entries = _read_index(Path(args.index))
    entries = [
        e for e in entries
        if e.get("stage") == "full"
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]

    road_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for entry in entries:
        drive = str(entry.get("drive") or "")
        outputs_dir = Path(entry.get("outputs_dir") or "")
        if not drive or not outputs_dir.exists():
            continue
        road_wgs_path = outputs_dir / "road_polygon_wgs84.geojson"
        road_path = outputs_dir / "road_polygon.geojson"
        road_feats = _read_geojson(road_wgs_path) if road_wgs_path.exists() else _read_geojson(road_path)
        polys = _collect_polys(road_feats)
        if not polys:
            continue
        poly = unary_union(polys)
        if not road_wgs_path.exists():
            poly = _to_wgs84(poly)
        minx, miny, maxx, maxy = poly.bounds
        road_bboxes[drive] = (minx, miny, maxx, maxy)

    rows = []
    missing_counts: Dict[str, int] = {}
    total_missing = 0

    for entry in entries:
        drive = str(entry.get("drive") or "")
        drive_out = outputs_dir / drive
        seed_path = drive_out / "intersections_seeds.geojson"
        final_path = drive_out / "intersections_final_wgs84.geojson"
        seeds = _read_geojson(seed_path) if seed_path.exists() else []
        final = _read_geojson(final_path) if final_path.exists() else []
        counts = {
            "traj": 0,
            "osm": 0,
            "sat": 0,
            "geom": 0,
        }
        for feat in seeds:
            src = (feat.get("properties") or {}).get("src_seed")
            if src in counts:
                counts[src] += 1
        missing_reason = "OK"
        if counts["traj"] == 0 and counts["osm"] == 0 and counts["sat"] == 0 and counts["geom"] > 0:
            missing_reason = "missing_seed_inputs"
        if seeds and not final:
            missing_reason = "shape_empty"
        rows.append(
            {
                "drive_id": drive,
                "seed_traj_cnt": counts["traj"],
                "seed_osm_cnt": counts["osm"],
                "seed_sat_cnt": counts["sat"],
                "seed_geom_cnt": counts["geom"],
                "final_cnt": len(final),
                "missing_reason": missing_reason,
            }
        )
        if missing_reason != "OK":
            missing_counts[missing_reason] = missing_counts.get(missing_reason, 0) + 1
            total_missing += 1

    _write_csv(audit_dir / "seeds_count_by_src_per_drive.csv", rows)

    osm_path = Path(args.osm_path)
    osm_feats = _read_geojson(osm_path)
    osm_lines = _collect_lines(osm_feats)
    if not osm_lines:
        raise SystemExit(f"ERROR: no OSM lines found in {osm_path}")
    osm_coords = osm_lines[0].coords[0]
    is_wgs84 = _is_wgs84_coords(osm_coords[0], osm_coords[1])
    if not is_wgs84:
        osm_lines = [_to_wgs84(line) for line in osm_lines]
    junctions = _osm_degree_junctions(osm_lines, args.osm_snap_m, args.osm_min_degree)
    osm_junctions = [{"type": "Feature", "geometry": mapping(pt), "properties": {}} for pt in junctions]
    _write_geojson(audit_dir / "osm_junctions_wgs84.geojson", osm_junctions)

    final_polys = []
    for entry in entries:
        drive = str(entry.get("drive") or "")
        final_path = outputs_dir / drive / "intersections_final_wgs84.geojson"
        if not final_path.exists():
            continue
        final_polys.extend(_collect_polys(_read_geojson(final_path)))
    final_union = unary_union(final_polys) if final_polys else Polygon()
    final_union_utm = _to_utm32(final_union) if not final_union.is_empty else Polygon()

    missing_points = []
    for pt in junctions:
        pt_utm = _to_utm32(pt)
        dist = pt_utm.distance(final_union_utm) if not final_union_utm.is_empty else float("inf")
        if dist > args.radius_m:
            missing_points.append({"type": "Feature", "geometry": mapping(pt), "properties": {"dist_m": round(dist, 3)}})
    _write_geojson(audit_dir / "missing_osm_junctions_wgs84.geojson", missing_points)

    sat_path = Path(args.sat_path)
    sat_stats = {}
    if sat_path.exists():
        sat_feats = _read_geojson(sat_path)
        sat_by_drive, stats = _assign_sat_features_to_drives(sat_feats, entries, road_bboxes)
        sat_stats = stats
        if stats.get("sat_unmatched"):
            missing_counts["sat_unassigned"] = int(stats["sat_unmatched"])
            total_missing += int(stats["sat_unmatched"])

    summary = {
        "missing_total": total_missing,
        "missing_reason_counts": missing_counts,
        "osm_junctions_total": len(junctions),
        "missing_osm_junctions": len(missing_points),
        "sat_assignment": sat_stats,
    }
    (audit_dir / "missing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
