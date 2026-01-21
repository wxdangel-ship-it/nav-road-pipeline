from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon, shape, mapping
from shapely.ops import unary_union, transform as geom_transform
from pyproj import Transformer


def _read_geojson(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("features", []) or []


def _write_geojson(path: Path, features: List[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
    utm = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    return geom_transform(utm.transform, geom)


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


def _collect_lines_with_types(features: List[dict]) -> Tuple[List[LineString], List[str]]:
    lines = []
    types = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        props = feat.get("properties") or {}
        hw = props.get("highway")
        if isinstance(hw, (list, tuple)):
            hw_val = str(hw[0]) if hw else ""
        else:
            hw_val = str(hw) if hw is not None else ""
        shp = shape(geom)
        if shp.is_empty:
            continue
        if shp.geom_type == "LineString":
            lines.append(shp)
            types.append(hw_val)
        elif shp.geom_type == "MultiLineString":
            for seg in shp.geoms:
                lines.append(seg)
                types.append(hw_val)
    return lines, types


def _collect_polys(features: List[dict]) -> List[Polygon]:
    polys = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shape(geom)
        if shp.is_empty:
            continue
        try:
            if not shp.is_valid:
                shp = shp.buffer(0)
        except Exception:
            pass
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


def _osm_degree_junctions_info(
    lines: List[LineString],
    types_by_line: List[str],
    snap_m: float,
    min_degree: int,
) -> List[Tuple[Point, int, List[str]]]:
    nodes: List[Tuple[float, float]] = []
    grid: Dict[Tuple[int, int], List[int]] = {}
    edges: Dict[int, set] = {}
    node_highways: Dict[int, set] = {}
    for line_idx, line in enumerate(lines):
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
        hw = types_by_line[line_idx] if line_idx < len(types_by_line) else ""
        if hw:
            node_highways.setdefault(n1, set()).add(hw)
            node_highways.setdefault(n2, set()).add(hw)
    out = []
    for idx, neighbors in edges.items():
        degree = len(neighbors)
        if degree >= min_degree:
            x, y = nodes[idx]
            types = sorted(node_highways.get(idx, set()))
            out.append((Point(x, y), degree, types))
    return out


def _bbox_to_utm(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    xs, ys = wgs84.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return minx, miny, maxx, maxy


def _expand_bbox_wgs84(bbox: Tuple[float, float, float, float], margin_m: float) -> Tuple[float, float, float, float]:
    if margin_m <= 0:
        return bbox
    wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    utm = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    xs, ys = wgs84.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minx -= margin_m
    miny -= margin_m
    maxx += margin_m
    maxy += margin_m
    lon, lat = utm.transform([minx, maxx], [miny, maxy])
    return min(lon), min(lat), max(lon), max(lat)


def _drive_bboxes(entries: List[dict]) -> Dict[str, Tuple[float, float, float, float]]:
    bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for entry in entries:
        drive = str(entry.get("drive") or "")
        if not drive:
            continue
        bbox = entry.get("bbox_wgs84") or entry.get("bbox4326")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bboxes[drive] = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            continue
        outputs_dir = Path(entry.get("outputs_dir") or "")
        if not outputs_dir.exists():
            continue
        road_wgs = outputs_dir / "road_polygon_wgs84.geojson"
        road = road_wgs if road_wgs.exists() else outputs_dir / "road_polygon.geojson"
        road_feats = _read_geojson(road)
        polys = _collect_polys(road_feats)
        if not polys:
            continue
        poly = unary_union(polys)
        if not road_wgs.exists():
            poly = _to_wgs84(poly)
        minx, miny, maxx, maxy = poly.bounds
        bboxes[drive] = (minx, miny, maxx, maxy)
    return bboxes


def _group_final_features(features: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for feat in features:
        props = feat.get("properties") or {}
        drive = props.get("drive_id") or props.get("drive")
        if not drive:
            continue
        grouped[str(drive)].append(feat)
    return grouped


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * q))
    idx = max(0, min(len(vals) - 1, idx))
    return float(vals[idx])


def _arm_distribution(arm_counts: List[int]) -> Dict[str, int]:
    dist = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for v in arm_counts:
        if v >= 3:
            dist["3+"] += 1
        elif v == 2:
            dist["2"] += 1
        elif v == 1:
            dist["1"] += 1
        else:
            dist["0"] += 1
    return dist


def _compute_metrics(
    name: str,
    final_path: Path,
    bboxes: Dict[str, Tuple[float, float, float, float]],
    junctions: List[Point],
    buffer_m: float,
    missing_dist_m: float,
) -> Dict[str, object]:
    features = _read_geojson(final_path)
    grouped = _group_final_features(features)
    per_drive = {}
    totals = {
        "missing_osm_junctions_count": 0,
        "final_cnt": 0,
        "arm_count_dist": {"0": 0, "1": 0, "2": 0, "3+": 0},
        "arm_count_ge3_ratio": None,
        "weak_arms_cnt": 0,
        "arm_count_heading_p50": None,
        "arm_count_heading_p75": None,
        "radius_m_used_p50": None,
        "radius_m_used_p75": None,
        "spur_removed_cnt": 0,
        "spur_removed_len_sum": 0.0,
        "area_m2_p50": None,
        "coverage_local_p25": None,
        "coverage_local_p50": None,
        "arm_cap_m_p50": None,
        "arm_span_p50": None,
        "arm_span_p75": None,
        "circularity_p50": None,
        "circularity_p75": None,
        "overlap_road_p25": None,
        "overlap_road_p50": None,
        "reason_counts": {},
    }
    circ_all = []
    overlap_all = []
    arm_all = []
    radius_all = []
    area_all = []
    coverage_all = []
    arm_cap_all = []
    arm_span_all = []
    reason_all = Counter()
    for drive, feats in grouped.items():
        polys = _collect_polys(feats)
        union = unary_union(polys) if polys else Polygon()
        union_utm = _to_utm32(union) if not union.is_empty else Polygon()
        bbox = bboxes.get(drive)
        if bbox is None:
            missing = 0
            junction_count = 0
        else:
            bbox = _expand_bbox_wgs84(bbox, buffer_m)
            junction_count = 0
            missing = 0
            for pt in junctions:
                if bbox[0] <= pt.x <= bbox[2] and bbox[1] <= pt.y <= bbox[3]:
                    junction_count += 1
                    dist = _to_utm32(pt).distance(union_utm) if not union_utm.is_empty else float("inf")
                    if dist > missing_dist_m:
                        missing += 1
        arm_counts = []
        circularity = []
        overlap = []
        radius_used = []
        areas = []
        coverage_vals = []
        arm_cap_vals = []
        arm_span_vals = []
        spur_removed_cnt = 0
        spur_removed_len_sum = 0.0
        reason_counts = Counter()
        for feat in feats:
            props = feat.get("properties") or {}
            arm_counts.append(int(props.get("arm_count_heading", props.get("arm_count", 0)) or 0))
            if props.get("circularity") is not None:
                circularity.append(float(props.get("circularity")))
            if props.get("overlap_road") is not None:
                overlap.append(float(props.get("overlap_road")))
            if props.get("radius_m_used") is not None:
                radius_used.append(float(props.get("radius_m_used")))
            if props.get("coverage_local") is not None:
                coverage_vals.append(float(props.get("coverage_local")))
            if props.get("arm_cap_m") is not None:
                arm_cap_vals.append(float(props.get("arm_cap_m")))
            if props.get("spur_removed_cnt") is not None:
                spur_removed_cnt += int(props.get("spur_removed_cnt") or 0)
            if props.get("spur_removed_len_sum") is not None:
                spur_removed_len_sum += float(props.get("spur_removed_len_sum") or 0.0)
            geom = feat.get("geometry")
            if geom:
                poly = shape(geom)
                if not poly.is_empty:
                    poly_utm = _to_utm32(poly)
                    areas.append(float(poly_utm.area))
                    minx, miny, maxx, maxy = poly_utm.bounds
                    arm_span_vals.append(max(maxx - minx, maxy - miny))
            if props.get("reason"):
                reason_counts[str(props.get("reason"))] += 1
        arm_dist = _arm_distribution(arm_counts)
        arm_ge3 = arm_dist["3+"]
        arm_ratio = (arm_ge3 / max(1, len(arm_counts))) if arm_counts else 0.0
        weak_arms = sum(1 for v in arm_counts if v <= 1)
        arm_sorted = sorted(arm_counts)
        arm_p50 = arm_sorted[len(arm_sorted) // 2] if arm_sorted else None
        arm_p75 = arm_sorted[int(len(arm_sorted) * 0.75)] if arm_sorted else None
        radius_used_sorted = sorted(radius_used)
        radius_p50 = radius_used_sorted[len(radius_used_sorted) // 2] if radius_used_sorted else None
        radius_p75 = radius_used_sorted[int(len(radius_used_sorted) * 0.75)] if radius_used_sorted else None
        areas_sorted = sorted(areas)
        area_p50 = areas_sorted[len(areas_sorted) // 2] if areas_sorted else None
        coverage_sorted = sorted(coverage_vals)
        coverage_p25 = coverage_sorted[int(len(coverage_sorted) * 0.25)] if coverage_sorted else None
        coverage_p50 = coverage_sorted[len(coverage_sorted) // 2] if coverage_sorted else None
        arm_cap_sorted = sorted(arm_cap_vals)
        arm_cap_p50 = arm_cap_sorted[len(arm_cap_sorted) // 2] if arm_cap_sorted else None
        arm_span_sorted = sorted(arm_span_vals)
        arm_span_p50 = arm_span_sorted[len(arm_span_sorted) // 2] if arm_span_sorted else None
        arm_span_p75 = arm_span_sorted[int(len(arm_span_sorted) * 0.75)] if arm_span_sorted else None
        per_drive[drive] = {
            "drive_id": drive,
            "junction_total": junction_count,
            "missing_osm_junctions_count": missing,
            "final_cnt": len(feats),
            "arm_count_dist": arm_dist,
            "arm_count_ge3_ratio": round(arm_ratio, 4),
            "weak_arms_cnt": weak_arms,
            "arm_count_heading_p50": arm_p50,
            "arm_count_heading_p75": arm_p75,
            "radius_m_used_p50": radius_p50,
            "radius_m_used_p75": radius_p75,
            "spur_removed_cnt": spur_removed_cnt,
            "spur_removed_len_sum": round(spur_removed_len_sum, 3),
            "area_m2_p50": area_p50,
            "coverage_local_p25": coverage_p25,
            "coverage_local_p50": coverage_p50,
            "arm_cap_m_p50": arm_cap_p50,
            "arm_span_p50": arm_span_p50,
            "arm_span_p75": arm_span_p75,
            "circularity_p50": _percentile(circularity, 0.5),
            "circularity_p75": _percentile(circularity, 0.75),
            "overlap_road_p25": _percentile(overlap, 0.25),
            "overlap_road_p50": _percentile(overlap, 0.5),
            "reason_counts": dict(reason_counts),
        }
        totals["missing_osm_junctions_count"] += missing
        totals["final_cnt"] += len(feats)
        for k in totals["arm_count_dist"]:
            totals["arm_count_dist"][k] += arm_dist.get(k, 0)
        totals["weak_arms_cnt"] += weak_arms
        totals["spur_removed_cnt"] += spur_removed_cnt
        totals["spur_removed_len_sum"] += float(spur_removed_len_sum)
        circ_all.extend(circularity)
        overlap_all.extend(overlap)
        arm_all.extend(arm_counts)
        radius_all.extend(radius_used)
        area_all.extend(areas)
        coverage_all.extend(coverage_vals)
        arm_cap_all.extend(arm_cap_vals)
        arm_span_all.extend(arm_span_vals)
        reason_all.update(reason_counts)
    totals["arm_count_ge3_ratio"] = round(totals["arm_count_dist"]["3+"] / max(1, sum(totals["arm_count_dist"].values())), 4)
    arm_all_sorted = sorted(arm_all)
    totals["arm_count_heading_p50"] = arm_all_sorted[len(arm_all_sorted) // 2] if arm_all_sorted else None
    totals["arm_count_heading_p75"] = arm_all_sorted[int(len(arm_all_sorted) * 0.75)] if arm_all_sorted else None
    radius_all_sorted = sorted(radius_all)
    totals["radius_m_used_p50"] = radius_all_sorted[len(radius_all_sorted) // 2] if radius_all_sorted else None
    totals["radius_m_used_p75"] = radius_all_sorted[int(len(radius_all_sorted) * 0.75)] if radius_all_sorted else None
    area_all_sorted = sorted(area_all)
    totals["area_m2_p50"] = area_all_sorted[len(area_all_sorted) // 2] if area_all_sorted else None
    coverage_all_sorted = sorted(coverage_all)
    totals["coverage_local_p25"] = coverage_all_sorted[int(len(coverage_all_sorted) * 0.25)] if coverage_all_sorted else None
    totals["coverage_local_p50"] = coverage_all_sorted[len(coverage_all_sorted) // 2] if coverage_all_sorted else None
    arm_cap_all_sorted = sorted(arm_cap_all)
    totals["arm_cap_m_p50"] = arm_cap_all_sorted[len(arm_cap_all_sorted) // 2] if arm_cap_all_sorted else None
    arm_span_all_sorted = sorted(arm_span_all)
    totals["arm_span_p50"] = arm_span_all_sorted[len(arm_span_all_sorted) // 2] if arm_span_all_sorted else None
    totals["arm_span_p75"] = arm_span_all_sorted[int(len(arm_span_all_sorted) * 0.75)] if arm_span_all_sorted else None
    totals["circularity_p50"] = _percentile(circ_all, 0.5)
    totals["circularity_p75"] = _percentile(circ_all, 0.75)
    totals["overlap_road_p25"] = _percentile(overlap_all, 0.25)
    totals["overlap_road_p50"] = _percentile(overlap_all, 0.5)
    totals["reason_counts"] = dict(reason_all)
    return {"name": name, "totals": totals, "per_drive": per_drive}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-final", required=True)
    ap.add_argument("--cand-final-s1", required=True)
    ap.add_argument("--cand-final-s2", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--osm-path", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drive-bbox-buffer-m", type=float, default=150.0)
    ap.add_argument("--missing-dist-m", type=float, default=20.0)
    ap.add_argument("--osm-snap-m", type=float, default=2.0)
    ap.add_argument("--osm-min-degree", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_entries = _read_index(Path(args.index))
    index_entries = [
        e for e in index_entries
        if e.get("stage") == "full"
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]
    bboxes = _drive_bboxes(index_entries)

    osm_path = Path(args.osm_path) if args.osm_path else None
    if osm_path is None or not osm_path.exists():
        raise SystemExit("ERROR: osm_path not found")
    osm_feats = _read_geojson(osm_path)
    osm_lines, osm_types = _collect_lines_with_types(osm_feats)
    if not osm_lines:
        raise SystemExit("ERROR: no OSM lines")
    c0 = list(osm_lines[0].coords)[0]
    is_wgs = _is_wgs84_coords(c0[0], c0[1])
    osm_lines_utm = [_to_utm32(line) for line in osm_lines] if is_wgs else osm_lines
    junctions_utm = _osm_degree_junctions_info(osm_lines_utm, osm_types, args.osm_snap_m, args.osm_min_degree)
    junctions = [_to_wgs84(pt) for pt, _, _ in junctions_utm]

    baseline = _compute_metrics(
        "baseline",
        Path(args.baseline_final),
        bboxes,
        junctions,
        args.drive_bbox_buffer_m,
        args.missing_dist_m,
    )
    cand_s1 = _compute_metrics(
        "s1",
        Path(args.cand_final_s1),
        bboxes,
        junctions,
        args.drive_bbox_buffer_m,
        args.missing_dist_m,
    )
    cand_s2 = _compute_metrics(
        "s2",
        Path(args.cand_final_s2),
        bboxes,
        junctions,
        args.drive_bbox_buffer_m,
        args.missing_dist_m,
    )

    def _rank_key(item: Dict[str, object]) -> Tuple[float, float, float, float]:
        totals = item["totals"]
        missing = -float(totals.get("missing_osm_junctions_count") or 0.0)
        arm_ratio = float(totals.get("arm_count_ge3_ratio") or 0.0)
        no_arms = -float((totals.get("arm_count_dist") or {}).get("0", 0))
        circ = -float(totals.get("circularity_p50") or 0.0)
        overlap = float(totals.get("overlap_road_p50") or 0.0)
        return (missing, arm_ratio + 0.01 * no_arms, circ, overlap)

    cand_list = [cand_s1, cand_s2]
    winner = max(cand_list, key=_rank_key)
    winner_name = winner["name"]

    report = {
        "baseline": baseline,
        "candidates": {"s1": cand_s1, "s2": cand_s2},
        "winner": winner_name,
        "selection_rule": [
            "priority_1: missing_osm_junctions_count lower is better",
            "priority_2: arm_count_ge3_ratio higher and no_arms fewer",
            "priority_3: circularity lower and overlap_road higher",
        ],
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.4f}" if isinstance(v, (float, int)) else "n/a"

    def _summary_block(item: Dict[str, object]) -> List[str]:
        t = item["totals"]
        return [
            f"- missing_osm_junctions_count: {t.get('missing_osm_junctions_count')}",
            f"- final_cnt: {t.get('final_cnt')}",
            f"- arm_count_dist: {t.get('arm_count_dist')}",
            f"- arm_count_ge3_ratio: {_fmt(t.get('arm_count_ge3_ratio'))}",
            f"- weak_arms_cnt: {t.get('weak_arms_cnt')}",
            f"- arm_count_heading_p50/p75: {t.get('arm_count_heading_p50')} / {t.get('arm_count_heading_p75')}",
            f"- radius_m_used_p50/p75: {t.get('radius_m_used_p50')} / {t.get('radius_m_used_p75')}",
            f"- spur_removed_cnt/len_sum: {t.get('spur_removed_cnt')} / {t.get('spur_removed_len_sum')}",
            f"- area_m2_p50: {t.get('area_m2_p50')}",
            f"- coverage_local_p25/p50: {t.get('coverage_local_p25')} / {t.get('coverage_local_p50')}",
            f"- arm_cap_m_p50: {t.get('arm_cap_m_p50')}",
            f"- arm_span_p50/p75: {t.get('arm_span_p50')} / {t.get('arm_span_p75')}",
            f"- circularity_p50/p75: {_fmt(t.get('circularity_p50'))} / {_fmt(t.get('circularity_p75'))}",
            f"- overlap_road_p25/p50: {_fmt(t.get('overlap_road_p25'))} / {_fmt(t.get('overlap_road_p50'))}",
            f"- reason_counts: {t.get('reason_counts')}",
        ]

    lines = ["# Intersections Quality Eval", ""]
    lines.append("## Baseline")
    lines.extend(_summary_block(baseline))
    lines.append("")
    lines.append("## Candidate S1")
    lines.extend(_summary_block(cand_s1))
    lines.append("")
    lines.append("## Candidate S2")
    lines.extend(_summary_block(cand_s2))
    lines.append("")
    lines.append(f"## Winner: {winner_name}")
    lines.append("")
    lines.append("## Compare Table")
    lines.append("| name | arm_span_p50 | arm_span_p75 | coverage_p50 | coverage_p25 | coverage_fallback_road_local | arm_ge3_ratio | missing_osm_junctions |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    def _fallback_cnt(item: Dict[str, object]) -> int:
        counts = item["totals"].get("reason_counts") or {}
        return int(counts.get("coverage_fallback_road_local", 0))
    for item in [baseline, cand_s1, cand_s2]:
        t = item["totals"]
        lines.append(
            f"| {item['name']} | {t.get('arm_span_p50')} | {t.get('arm_span_p75')} | "
            f"{t.get('coverage_local_p50')} | {t.get('coverage_local_p25')} | "
            f"{_fallback_cnt(item)} | {t.get('arm_count_ge3_ratio')} | {t.get('missing_osm_junctions_count')} |"
        )

    drive_007 = "2013_05_28_drive_0007_sync"
    lines.append("")
    lines.append("## Drive 0007 Details")
    for label, item in [("baseline", baseline), ("s1", cand_s1), ("s2", cand_s2)]:
        d = item["per_drive"].get(drive_007, {})
        lines.append(
            f"- {label}: missing_osm_junctions={d.get('missing_osm_junctions_count')}, "
            f"arm_count_dist={d.get('arm_count_dist')}, "
            f"weak_arms_cnt={d.get('weak_arms_cnt')}, "
            f"coverage_p50={d.get('coverage_local_p50')}, "
            f"arm_span_p50={d.get('arm_span_p50')}, "
            f"spur_removed_cnt={d.get('spur_removed_cnt')}"
        )

    (out_dir / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
