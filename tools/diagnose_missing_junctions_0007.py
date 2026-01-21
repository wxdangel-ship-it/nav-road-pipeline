import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyproj import Transformer
from shapely.geometry import Point, shape, mapping
from shapely.ops import unary_union, transform as geom_transform


def _read_geojson(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", []) or []


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _is_wgs84_coords(x: float, y: float) -> bool:
    return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0


def _to_utm32(geom):
    wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    return geom_transform(wgs84.transform, geom)


def _to_wgs84(geom):
    wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    return geom_transform(wgs84.transform, geom)


def _load_union_polys(path: Path) -> Optional[object]:
    feats = _read_geojson(path)
    polys = []
    for feat in feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shape(geom)
        if shp.is_empty:
            continue
        polys.append(shp)
    if not polys:
        return None
    return unary_union(polys)


def _nearest_seed(
    junction_pt: Point,
    seeds: List[Tuple[Point, dict]],
) -> Tuple[Optional[Point], Optional[dict], float]:
    best = None
    best_props = None
    best_dist = float("inf")
    for pt, props in seeds:
        dist = junction_pt.distance(pt)
        if dist < best_dist:
            best_dist = dist
            best = pt
            best_props = props
    if best is None:
        return None, None, float("inf")
    return best, best_props, float(best_dist)


def _trace_map(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in rows:
        seed_id = str(row.get("seed_id") or "")
        if not seed_id:
            continue
        if seed_id not in out:
            out[seed_id] = row
            continue
        if row.get("kept") is True:
            out[seed_id] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--road-poly", required=True)
    parser.add_argument("--seed-match-thr", type=float, default=15.0)
    parser.add_argument("--cover-dist-m", type=float, default=20.0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = in_dir / "diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)

    junctions_path = in_dir / "osm_junctions_wgs84.geojson"
    seeds_path = in_dir / "intersections_seeds_wgs84.geojson"
    if not seeds_path.exists():
        seeds_path = in_dir / "intersections_seeds.geojson"
    final_path = in_dir / "intersections_final_wgs84.geojson"
    refined_path = in_dir / "refined_wgs84.geojson"
    road_local_path = in_dir / "road_local_wgs84.geojson"
    drop_trace_path = in_dir / "debug_drop_trace.jsonl"

    road_poly_path = Path(args.road_poly)

    junction_feats = _read_geojson(junctions_path)
    seed_feats = _read_geojson(seeds_path)
    final_feats = _read_geojson(final_path)
    refined_feats = _read_geojson(refined_path) if refined_path.exists() else []
    road_local_feats = _read_geojson(road_local_path) if road_local_path.exists() else []
    drop_trace = _read_jsonl(drop_trace_path) if drop_trace_path.exists() else []

    road_poly = _load_union_polys(road_poly_path)
    if road_poly is None:
        raise SystemExit("ERROR: road_polygon is empty")

    # Detect road polygon CRS, assume WGS84 if so, otherwise keep as-is.
    road_poly_wgs = road_poly
    if road_poly is not None:
        minx, miny, maxx, maxy = road_poly.bounds
        if not _is_wgs84_coords(minx, miny):
            road_poly_utm = road_poly
            road_poly_wgs = _to_wgs84(road_poly_utm)
        else:
            road_poly_utm = _to_utm32(road_poly)
    else:
        road_poly_utm = None

    # Build seed list in UTM
    seeds_utm = []
    for feat in seed_feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        pt = shape(geom)
        if pt.is_empty:
            continue
        props = feat.get("properties") or {}
        pt_utm = _to_utm32(pt) if _is_wgs84_coords(pt.x, pt.y) else pt
        seeds_utm.append((pt_utm, props))

    # Final polygons union (UTM)
    final_polys = []
    final_by_seed: Dict[str, dict] = {}
    for feat in final_feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        poly_utm = _to_utm32(poly) if _is_wgs84_coords(*poly.bounds[:2]) else poly
        final_polys.append(poly_utm)
        props = feat.get("properties") or {}
        seed_id = str(props.get("seed_id") or "")
        if seed_id:
            area = float(poly_utm.area)
            if seed_id not in final_by_seed or area > final_by_seed[seed_id]["area"]:
                final_by_seed[seed_id] = {"poly": poly_utm, "props": props, "area": area}

    final_union = unary_union(final_polys) if final_polys else None

    trace_by_seed = _trace_map(drop_trace)

    rows = []
    status_features = []
    no_seed_dists = []
    drop_reasons = {}
    road_local_zero = []
    drop_stage_counts = {}

    for feat in junction_feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        pt = shape(geom)
        if pt.is_empty:
            continue
        props = feat.get("properties") or {}
        junction_id = str(props.get("junction_id") or "")
        degree = int(props.get("degree") or 0)
        pt_utm = _to_utm32(pt) if _is_wgs84_coords(pt.x, pt.y) else pt

        seed_pt, seed_props, seed_dist = _nearest_seed(pt_utm, seeds_utm)
        matched_seed_id = ""
        seed_src = ""
        seed_reason = ""
        if seed_props:
            matched_seed_id = str(seed_props.get("junction_id") or seed_props.get("seed_id") or "")
            seed_src = str(seed_props.get("src_seed") or "")
            seed_reason = str(seed_props.get("reason") or "")

        final_info = final_by_seed.get(matched_seed_id) if matched_seed_id else None
        trace = trace_by_seed.get(matched_seed_id) if matched_seed_id else None
        refined_area = float(final_info["area"]) if final_info else float((trace or {}).get("refined_area_m2") or 0.0)
        arms_src = str((trace or {}).get("arms_src") or (final_info or {}).get("props", {}).get("arms_src") or "")
        arm_count_incident = int((trace or {}).get("arm_count_incident") or (final_info or {}).get("props", {}).get("arm_count_incident") or 0)
        arm_count_axis = int((trace or {}).get("arm_count_axis") or (final_info or {}).get("props", {}).get("arm_count_axis") or 0)
        arm_count_approach = int((trace or {}).get("arm_count_approach") or (final_info or {}).get("props", {}).get("arm_count_approach") or 0)
        has_arms = int((trace or {}).get("has_arms") or (final_info or {}).get("props", {}).get("has_arms") or 0)

        arm_cap_m = float((trace or {}).get("arm_cap_m") or (final_info or {}).get("props", {}).get("arm_cap_m") or 25.0)
        road_local_src = str((final_info or {}).get("props", {}).get("road_local_src") or (trace or {}).get("road_local_src") or "")
        retry_radius = float(
            (final_info or {}).get("props", {}).get("road_local_retry_radius_m") or (trace or {}).get("road_local_retry_radius_m") or arm_cap_m
        )
        road_local_area = 0.0
        has_road_local = 0
        coverage_local = 0.0
        if road_local_src == "seed_circle":
            road_local_area = float(seed_pt.buffer(retry_radius).area) if seed_pt is not None else 0.0
            has_road_local = 1
            coverage_local = 1.0 if refined_area > 0 else 0.0
        elif road_poly_utm is not None and seed_pt is not None:
            road_local = road_poly_utm.intersection(seed_pt.buffer(retry_radius))
            if road_local is not None and not road_local.is_empty:
                road_local_area = float(road_local.area)
                has_road_local = 1 if road_local_area > 0 else 0
                if refined_area > 0:
                    refined_poly = final_info["poly"] if final_info else None
                    if refined_poly is not None and not refined_poly.is_empty:
                        coverage_local = float(refined_poly.intersection(road_local).area) / max(1e-6, road_local_area)
        if road_local_src in ("sat_fill", "seed_circle") and has_road_local == 0:
            has_road_local = 1
        if final_info is None and coverage_local == 0.0 and trace is not None:
            coverage_local = float((trace or {}).get("coverage_local") or 0.0)

        final_covered = 0
        if final_union is not None and not final_union.is_empty:
            dist_final = float(pt_utm.distance(final_union))
            final_covered = 1 if dist_final <= float(args.cover_dist_m) else 0
        else:
            dist_final = float("inf")

        drop_stage = "covered"
        notes = ""
        if seed_dist > float(args.seed_match_thr):
            drop_stage = "no_seed"
            no_seed_dists.append(seed_dist)
        elif road_local_area <= 0:
            drop_stage = "road_local_empty"
            road_local_zero.append(junction_id)
        elif has_arms <= 0 or arms_src == "none":
            drop_stage = "arms_empty"
        elif refined_area <= 0:
            drop_stage = "shape_empty"
        elif final_covered == 0:
            reason = str((trace or {}).get("dropped_reason") or "")
            if reason.startswith("gate_drop"):
                drop_stage = "gate_drop"
            elif reason.startswith("dedup_iou"):
                drop_stage = "dedup_drop"
            else:
                drop_stage = "shape_empty"
            if reason:
                notes = reason

        drop_stage_counts[drop_stage] = drop_stage_counts.get(drop_stage, 0) + 1
        drop_reason = str((trace or {}).get("dropped_reason") or "")
        if drop_reason:
            drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1

        rows.append(
            {
                "junction_id": junction_id,
                "degree": degree,
                "lon": round(float(pt.x), 7),
                "lat": round(float(pt.y), 7),
                "nearest_seed_dist_m": round(float(seed_dist), 3),
                "matched_seed_id": matched_seed_id,
                "seed_src_seed": seed_src,
                "seed_reason": seed_reason,
                "has_road_local": has_road_local,
                "road_local_area_m2": round(float(road_local_area), 3),
                "has_arms": has_arms,
                "arms_src": arms_src,
                "arm_count_incident": arm_count_incident,
                "arm_count_axis": arm_count_axis,
                "arm_count_approach": arm_count_approach,
                "refined_area_m2": round(float(refined_area), 3),
                "coverage_local": round(float(coverage_local), 4),
                "final_covered": final_covered,
                "drop_stage": drop_stage,
                "notes": notes,
            }
        )

        status_features.append(
            {
                "type": "Feature",
                "geometry": mapping(pt),
                "properties": {
                    "junction_id": junction_id,
                    "degree": degree,
                    "drop_stage": drop_stage,
                    "final_covered": final_covered,
                    "nearest_seed_dist_m": round(float(seed_dist), 3),
                },
            }
        )

    csv_path = out_dir / "junction_diagnose_0007.csv"
    md_path = out_dir / "junction_diagnose_0007.md"
    geo_path = out_dir / "junction_status_wgs84.geojson"

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        csv_path.write_text("", encoding="utf-8")

    md_lines = []
    md_lines.append("junction_diagnose_0007")
    md_lines.append("")
    md_lines.append(f"total_junctions: {len(rows)}")
    md_lines.append(f"drop_stage_counts: {drop_stage_counts}")
    if road_local_zero:
        md_lines.append(f"road_local_empty_junctions: {road_local_zero}")
    if no_seed_dists:
        md_lines.append(f"no_seed_dist_m_min: {min(no_seed_dists):.3f}")
        md_lines.append(f"no_seed_dist_m_p50: {sorted(no_seed_dists)[len(no_seed_dists)//2]:.3f}")
        md_lines.append(f"no_seed_dist_m_max: {max(no_seed_dists):.3f}")
    if drop_reasons:
        md_lines.append(f"dropped_reason_counts: {drop_reasons}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    geo_path.write_text(json.dumps({"type": "FeatureCollection", "features": status_features}, ensure_ascii=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
