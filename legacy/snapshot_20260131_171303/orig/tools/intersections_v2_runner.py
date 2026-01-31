from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon, shape, mapping
from shapely.ops import unary_union, transform as geom_transform
from pyproj import Transformer
import yaml
import sqlite3
from shapely import wkb as shapely_wkb

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.intersection_shape import (
    arm_count_branches as _arm_count_branches,
    calc_arm_count_by_heading as _arm_count_heading,
    calc_arm_count_by_approach as _arm_count_approach,
    aspect_ratio as _aspect_ratio,
    circularity as _circularity,
    overlap_with_road as _overlap_with_road,
    refine_intersection_polygon as _refine_intersection_polygon,
)
from pipeline.centerlines_v2 import _polygon_centerline
from pipeline.evidence.image_feature_provider import load_features as load_image_features


def _read_gpkg_features(gpkg: Path, drive: str) -> Dict[str, List[dict]]:
    if not gpkg.exists():
        return {}
    out: Dict[str, List[dict]] = {}
    conn = sqlite3.connect(str(gpkg))
    try:
        cur = conn.cursor()
        layers = cur.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchall()
        geom_cols = {
            row[0]: row[1]
            for row in cur.execute("SELECT table_name, column_name FROM gpkg_geometry_columns").fetchall()
        }
        for (layer_name,) in layers:
            if str(layer_name).endswith("_wgs84"):
                continue
            geom_col = geom_cols.get(layer_name)
            if not geom_col:
                continue
            cols = [row[1] for row in cur.execute(f"PRAGMA table_info('{layer_name}')").fetchall()]
            if not cols:
                continue
            select_cols = ", ".join([f'"{c}"' for c in cols])
            rows = cur.execute(f'SELECT {select_cols} FROM "{layer_name}"').fetchall()
            if not rows:
                continue
            records = []
            geom_idx = cols.index(geom_col)
            drive_idx = cols.index("drive_id") if "drive_id" in cols else None
            for row in rows:
                if drive_idx is not None and row[drive_idx] not in (None, drive):
                    continue
                geom_blob = row[geom_idx]
                if geom_blob is None:
                    continue
                try:
                    geom = shapely_wkb.loads(bytes(geom_blob))
                except Exception:
                    continue
                props = {}
                for idx, name in enumerate(cols):
                    if idx == geom_idx:
                        continue
                    props[name] = row[idx]
                props["class"] = props.get("class") or layer_name
                records.append({"geometry": geom, "properties": props})
            if records:
                out[str(layer_name)] = records
    finally:
        conn.close()
    return out


def _load_clean_evidence(drive: str, store_dir: Path) -> Dict[str, List[dict]]:
    gpkg = store_dir
    if gpkg.is_dir():
        gpkg = store_dir / drive / f"evidence_clean_{drive}.gpkg"
    if not gpkg.exists():
        return {}
    return _read_gpkg_features(gpkg, drive)


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


def _write_geojson(path: Path, features: List[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_wgs84(features: List[dict], crs_epsg: int) -> List[dict]:
    wgs84 = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)
    out = []
    for feat in features:
        geom = geom_transform(wgs84.transform, shape(feat["geometry"]))
        out.append({"type": "Feature", "geometry": mapping(geom), "properties": feat.get("properties") or {}})
    return out


def _geom_to_wgs84(geom):
    utm = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    return geom_transform(utm.transform, geom)


def _is_wgs84_coords(x: float, y: float) -> bool:
    return abs(x) <= 180 and abs(y) <= 90


def _find_latest_osm_path() -> Optional[Path]:
    candidates = list(Path("runs").rglob("drivable_roads.geojson"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _entry_bbox_wgs84(entry: dict) -> Optional[Tuple[float, float, float, float]]:
    for key in ("bbox_wgs84", "bbox4326", "bbox"):
        bbox = entry.get(key)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return None


def _drive_bbox_wgs84(entry: dict, road_poly: Optional[Polygon]) -> Optional[Tuple[float, float, float, float]]:
    bbox = _entry_bbox_wgs84(entry)
    if bbox is not None:
        return bbox
    if road_poly is None or road_poly.is_empty:
        return None
    minx, miny, maxx, maxy = road_poly.bounds
    wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    xs, ys = wgs84.transform([minx, maxx], [miny, maxy])
    return min(xs), min(ys), max(xs), max(ys)


def _expand_bbox(bbox: Tuple[float, float, float, float], margin_m: float) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bbox
    if margin_m <= 0:
        return bbox
    wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    utm = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    xs, ys = wgs84.transform([minx, maxx], [miny, maxy])
    minx_u, maxx_u = min(xs), max(xs)
    miny_u, maxy_u = min(ys), max(ys)
    minx_u -= margin_m
    miny_u -= margin_m
    maxx_u += margin_m
    maxy_u += margin_m
    lon, lat = utm.transform([minx_u, maxx_u], [miny_u, maxy_u])
    return min(lon), min(lat), max(lon), max(lat)


def _bbox_to_utm(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    xs, ys = wgs84.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return minx, miny, maxx, maxy


def _bbox_intersects(bounds: Tuple[float, float, float, float], bbox: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    bminx, bminy, bmaxx, bmaxy = bbox
    return not (maxx < bminx or minx > bmaxx or maxy < bminy or miny > bmaxy)


def _dist_point_bbox_m(pt_wgs: Point, bbox_wgs: Tuple[float, float, float, float]) -> float:
    utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    x, y = utm.transform(pt_wgs.x, pt_wgs.y)
    minx, miny, maxx, maxy = _bbox_to_utm(bbox_wgs)
    if minx <= x <= maxx and miny <= y <= maxy:
        return 0.0
    dx = max(minx - x, 0.0, x - maxx)
    dy = max(miny - y, 0.0, y - maxy)
    return (dx * dx + dy * dy) ** 0.5


def _infer_bbox_from_road(road_poly: Polygon) -> Optional[Tuple[float, float, float, float]]:
    if road_poly is None or road_poly.is_empty:
        return None
    minx, miny, maxx, maxy = road_poly.bounds
    wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    xs, ys = wgs84.transform([minx, maxx], [miny, maxy])
    return min(xs), min(ys), max(xs), max(ys)


def _assign_sat_features_to_drives(
    sat_feats: List[dict],
    entries: List[dict],
    road_polys: Dict[str, Polygon],
) -> Tuple[Dict[str, List[dict]], dict, List[dict]]:
    by_drive: Dict[str, List[dict]] = {}
    unmatched = 0
    unmatched_feats: List[dict] = []
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
        best_area = 0.0
        for entry in entries:
            drive = str(entry.get("drive") or "")
            bbox = _entry_bbox_wgs84(entry)
            if bbox is None:
                bbox = _infer_bbox_from_road(road_polys.get(drive))
            if bbox is None:
                continue
            minx, miny, maxx, maxy = bbox
            if minx <= c.x <= maxx and miny <= c.y <= maxy:
                area = (maxx - minx) * (maxy - miny)
                if area >= best_area:
                    matched = drive
                    best_area = area
        if matched is None:
            unmatched += 1
            unmatched_feats.append(feat)
            continue
        props = feat.get("properties") or {}
        props["drive_id"] = matched
        props["tile_id"] = matched
        feat["properties"] = props
        by_drive.setdefault(matched, []).append(feat)
    stats = {
        "sat_total": len(sat_feats),
        "sat_matched": sum(len(v) for v in by_drive.values()),
        "sat_unmatched": unmatched,
    }
    return by_drive, stats, unmatched_feats


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


def _to_utm32(geom):
    wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    return geom_transform(wgs84.transform, geom)


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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _ray_intersections(road_poly: Polygon, origin: Point, direction: Tuple[float, float], max_dist: float) -> Optional[float]:
    dx, dy = direction
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    dx /= length
    dy /= length
    ray = LineString([(origin.x, origin.y), (origin.x + dx * max_dist, origin.y + dy * max_dist)])
    inter = ray.intersection(road_poly.boundary)
    if inter.is_empty:
        return None
    points = []
    if inter.geom_type == "Point":
        points = [inter]
    elif inter.geom_type == "MultiPoint":
        points = list(inter.geoms)
    elif inter.geom_type == "GeometryCollection":
        for geom in inter.geoms:
            if geom.geom_type == "Point":
                points.append(geom)
            elif geom.geom_type == "LineString":
                coords = list(geom.coords)
                if coords:
                    points.append(Point(coords[0]))
                    points.append(Point(coords[-1]))
    elif inter.geom_type == "LineString":
        coords = list(inter.coords)
        if coords:
            points.append(Point(coords[0]))
            points.append(Point(coords[-1]))
    if not points:
        return None
    best = None
    for pt in points:
        dist = origin.distance(pt)
        if dist <= max_dist and (best is None or dist < best):
            best = dist
    return best


def _width_profile(
    line: LineString,
    road_poly: Polygon,
    sample_step_m: float,
    probe_m: float,
    max_samples: int,
) -> List[float]:
    if line is None or line.is_empty or road_poly is None or road_poly.is_empty:
        return []
    if line.length <= 0:
        return []
    step = max(sample_step_m, 0.5)
    n_samples = max(1, int(line.length / step) + 1)
    if n_samples > max_samples:
        n_samples = max_samples
    widths = []
    for i in range(n_samples):
        t = i / (n_samples - 1) if n_samples > 1 else 0.5
        pt = line.interpolate(t, normalized=True)
        dist = line.project(pt)
        eps = min(2.0, max(0.5, line.length * 0.05))
        p1 = line.interpolate(max(0.0, dist - eps))
        p2 = line.interpolate(min(line.length, dist + eps))
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        if dx == 0 and dy == 0:
            continue
        nx, ny = -dy, dx
        w1 = _ray_intersections(road_poly, pt, (nx, ny), probe_m)
        w2 = _ray_intersections(road_poly, pt, (-nx, -ny), probe_m)
        if w1 is None or w2 is None:
            continue
        widths.append(w1 + w2)
    return widths


def _estimate_road_width_local(
    seed: Point,
    road_poly: Polygon,
    lines: List[LineString],
    search_radius_m: float,
    sample_step_m: float,
    probe_m: float,
    max_samples: int,
) -> Optional[float]:
    if seed is None or road_poly is None or road_poly.is_empty:
        return None
    if not lines:
        return None
    clip = seed.buffer(max(search_radius_m, 1.0))
    widths = []
    for line in lines:
        if line is None or line.is_empty:
            continue
        if not line.intersects(clip):
            continue
        local = line.intersection(clip)
        if local.is_empty:
            continue
        if local.geom_type == "LineString":
            widths.extend(_width_profile(local, road_poly, sample_step_m, probe_m, max_samples))
        elif local.geom_type == "MultiLineString":
            for seg in local.geoms:
                widths.extend(_width_profile(seg, road_poly, sample_step_m, probe_m, max_samples))
    if not widths:
        return None
    widths = [w for w in widths if w > 0]
    if not widths:
        return None
    widths.sort()
    return widths[len(widths) // 2]


def _infer_osm_degree(seed: Point, osm_seed_items: List[dict], max_dist_m: float) -> int:
    best = None
    best_dist = None
    for item in osm_seed_items:
        pt = item.get("seed")
        if pt is None:
            continue
        dist = seed.distance(pt)
        if dist <= max_dist_m and (best_dist is None or dist < best_dist):
            best = int(item.get("degree") or 0)
            best_dist = dist
    return best if best is not None else 2


def _adaptive_radius(
    seed: Point,
    road_poly: Polygon,
    width_lines: List[LineString],
    osm_seed_items: List[dict],
    cfg: dict,
) -> Tuple[float, float, int, str]:
    r_min = float(cfg.get("radius_min_m", 18.0))
    r_max = float(cfg.get("radius_max_m", 45.0))
    k_w = float(cfg.get("radius_k_w", 1.4))
    k_deg = float(cfg.get("radius_k_deg", 4.0))
    width_default = float(cfg.get("width_default_m", 12.0))
    width_search = float(cfg.get("width_search_radius_m", 30.0))
    width_step = float(cfg.get("width_sample_step_m", 4.0))
    width_probe = float(cfg.get("width_probe_m", 30.0))
    width_max_samples = int(cfg.get("width_max_samples", 12))
    degree = _infer_osm_degree(seed, osm_seed_items, float(cfg.get("seed_search_radius_m_osm", 45.0)))
    width_est = _estimate_road_width_local(
        seed,
        road_poly,
        width_lines,
        width_search,
        width_step,
        width_probe,
        width_max_samples,
    )
    radius_src = "width+degree"
    if width_est is None or width_est <= 0:
        width_est = width_default
        radius_src = "fallback_default"
    radius_raw = k_w * float(width_est) + k_deg * max(0.0, float(degree - 2))
    radius_used = _clamp(radius_raw, r_min, r_max)
    return radius_used, float(width_est), int(degree), radius_src


def _spur_prune(poly: Polygon, seed: Point, radius_used: float, cfg: dict) -> Tuple[Polygon, dict]:
    if poly is None or poly.is_empty or seed is None:
        return poly, {"spur_pruned": 0, "spur_removed_cnt": 0, "spur_removed_len_sum": 0.0}
    if not bool(cfg.get("spur_prune_enabled", True)):
        return poly, {"spur_pruned": 0, "spur_removed_cnt": 0, "spur_removed_len_sum": 0.0}
    core_ratio = float(cfg.get("spur_core_ratio", 0.6))
    core_min = float(cfg.get("spur_core_min_m", 12.0))
    core_max = float(cfg.get("spur_core_max_m", 25.0))
    min_width = float(cfg.get("spur_min_width_m", 2.5))
    max_len = float(cfg.get("spur_max_len_m", 25.0))
    r_core = _clamp(radius_used * core_ratio, core_min, core_max)
    core = poly.intersection(seed.buffer(r_core))
    arms_part = poly.difference(core)
    if arms_part.is_empty:
        return poly, {"spur_pruned": 0, "spur_removed_cnt": 0, "spur_removed_len_sum": 0.0}
    components = []
    if arms_part.geom_type == "Polygon":
        components = [arms_part]
    elif arms_part.geom_type == "MultiPolygon":
        components = list(arms_part.geoms)
    removed = 0
    removed_len_sum = 0.0
    kept = []
    for comp in components:
        if comp.is_empty:
            continue
        rect = comp.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        if len(coords) < 4:
            kept.append(comp)
            continue
        edges = []
        for i in range(4):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            edges.append(math.hypot(x2 - x1, y2 - y1))
        edges = [e for e in edges if e > 0]
        if not edges:
            kept.append(comp)
            continue
        length_proxy = max(edges)
        width_proxy = comp.area / length_proxy if length_proxy > 0 else comp.area
        if width_proxy < min_width and length_proxy > max_len:
            removed += 1
            removed_len_sum += length_proxy
            continue
        kept.append(comp)
    if not kept:
        pruned = core
    else:
        pruned = unary_union([core] + kept)
    pruned = pruned.buffer(0)
    return pruned, {
        "spur_pruned": 1 if removed > 0 else 0,
        "spur_removed_cnt": removed,
        "spur_removed_len_sum": round(removed_len_sum, 3),
    }


def _make_valid(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        return geom.buffer(0)
    except Exception:
        return geom


def _explode_lines(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return [g for g in geom.geoms if not g.is_empty]
    if geom.geom_type == "GeometryCollection":
        lines: List[LineString] = []
        for g in geom.geoms:
            lines.extend(_explode_lines(g))
        return lines
    return []


def _arms_lines_from_source(seed: Point, lines: List[LineString], arm_cap_m: float) -> List[LineString]:
    if not lines:
        return []
    clip = seed.buffer(max(arm_cap_m, 1.0))
    out = []
    for line in lines:
        if line is None or line.is_empty:
            continue
        seg = line.intersection(clip)
        out.extend(_explode_lines(seg))
    return [l for l in out if l.length > 0]


def _line_substring(line: LineString, start_dist: float, end_dist: float) -> Optional[LineString]:
    if line is None or line.is_empty:
        return None
    length = float(line.length)
    start = max(0.0, min(length, float(start_dist)))
    end = max(0.0, min(length, float(end_dist)))
    if end <= start:
        return None
    try:
        from shapely.ops import substring

        return substring(line, start, end)
    except Exception:
        coords = list(line.coords)
        if len(coords) < 2:
            return None
        out = []
        dist = 0.0
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            next_dist = dist + seg_len
            if next_dist < start:
                dist = next_dist
                continue
            if dist > end:
                break
            if not out:
                if start > dist and seg_len > 0:
                    ratio = (start - dist) / seg_len
                    sx = x1 + (x2 - x1) * ratio
                    sy = y1 + (y2 - y1) * ratio
                    out.append((sx, sy))
                else:
                    out.append((x1, y1))
            if end <= next_dist:
                if seg_len > 0:
                    ratio = (end - dist) / seg_len
                    ex = x1 + (x2 - x1) * ratio
                    ey = y1 + (y2 - y1) * ratio
                    out.append((ex, ey))
                else:
                    out.append((x2, y2))
                break
            out.append((x2, y2))
            dist = next_dist
        if len(out) < 2:
            return None
        return LineString(out)


def _arm_segment_from_junction(line: LineString, junction: Point, arm_len_m: float) -> Optional[LineString]:
    if line is None or line.is_empty:
        return None
    dist = float(line.project(junction))
    total = float(line.length)
    if total <= 0:
        return None
    dist_to_start = dist
    dist_to_end = total - dist
    if dist_to_end >= dist_to_start:
        start = dist
        end = dist + arm_len_m
    else:
        start = dist - arm_len_m
        end = dist
    return _line_substring(line, start, end)


def _osm_junctions_with_incident(
    lines: List[LineString],
    snap_m: float,
    min_degree: int,
    drive_id: str,
) -> List[dict]:
    nodes: List[Tuple[float, float]] = []
    grid: Dict[Tuple[int, int], List[int]] = {}
    edges: Dict[int, set] = {}
    node_lines: Dict[int, set] = {}
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
        node_lines.setdefault(n1, set()).add(line_idx)
        node_lines.setdefault(n2, set()).add(line_idx)
    out = []
    for idx, neighbors in edges.items():
        x, y = nodes[idx]
        pt = Point(x, y)
        incident_idxs = node_lines.get(idx, set())
        incident = [lines[i] for i in incident_idxs if 0 <= i < len(lines)]
        degree = max(len(neighbors), len(incident))
        if degree >= min_degree:
            out.append(
                {
                    "point": pt,
                    "degree": degree,
                    "incident": incident,
                    "junction_id": f"{drive_id}_osm_junction_{idx}",
                }
            )
    return out


def _build_arms_v3(
    seed: Point,
    road_poly: Polygon,
    centerlines: List[LineString],
    osm_lines: List[LineString],
    osm_junctions: List[dict],
    cfg: dict,
) -> Tuple[dict, dict]:
    arm_cap_m = float(cfg.get("arm_cap_m", 25.0))
    center_search = float(cfg.get("seed_search_radius_m_centerlines", 45.0))
    min_center = int(cfg.get("min_arms_lines_centerlines", 2))
    center_use = []
    if centerlines:
        center_use = [ln for ln in centerlines if ln is not None and ln.intersects(seed.buffer(center_search))]
    arms_src = "none"
    arms_lines = []
    arm_buffer_m = float(cfg.get("arm_buffer_m", 5.0))
    osm_links_used_count = 0
    arm_count_incident = 0
    if center_use and len(center_use) >= min_center:
        arms_src = "centerlines"
        arms_lines = _arms_lines_from_source(seed, center_use, arm_cap_m)
        arm_buffer_m = float(cfg.get("arm_buffer_m_centerlines", 5.0))
    if not arms_lines or _arm_count_approach(
        arms_lines,
        seed,
        float(cfg.get("arm_count_eval_radius_m", 14.0)),
        float(cfg.get("arm_count_angle_bin_deg", 20.0)),
        float(cfg.get("arm_count_min_len_m", 3.0)),
    ) < 3:
        osm_search = float(cfg.get("seed_search_radius_m_osm", 45.0))
        best = None
        best_dist = None
        for j in osm_junctions:
            pt = j.get("point")
            if pt is None:
                continue
            dist = seed.distance(pt)
            if dist <= osm_search and (best_dist is None or dist < best_dist):
                best = j
                best_dist = dist
        if best is not None:
            incident = []
            if osm_lines:
                search_m = float(cfg.get("osm_junction_incident_search_m", 6.0))
                seen = set()
                for line in osm_lines:
                    if line is None or line.is_empty:
                        continue
                    if line.distance(best["point"]) <= search_m:
                        line_id = id(line)
                        if line_id in seen:
                            continue
                        seen.add(line_id)
                        incident.append(line)
                if len(incident) < 3:
                    search_m = max(search_m, float(cfg.get("osm_snap_m", 2.0)) * 3.0)
                    for line in osm_lines:
                        if line is None or line.is_empty:
                            continue
                        if line.distance(best["point"]) <= search_m:
                            line_id = id(line)
                            if line_id in seen:
                                continue
                            seen.add(line_id)
                            incident.append(line)
            if not incident:
                incident = best.get("incident") or []
            arm_count_incident = len(incident)
            if arm_count_incident >= 3:
                arms_src = "osm_junction"
                arm_buffer_m = float(cfg.get("arm_buffer_m_osm", 3.5))
                osm_links_used_count = arm_count_incident
                for line in incident:
                    seg = _arm_segment_from_junction(line, best["point"], arm_cap_m)
                    if seg is not None and not seg.is_empty:
                        arms_lines.append(seg)
            else:
                arms_src = "osm_junction"
                osm_links_used_count = arm_count_incident
        if not arms_lines:
            try:
                local = road_poly.buffer(float(cfg.get("road_buffer_m", 1.0))).intersection(seed.buffer(arm_cap_m))
                local = local.buffer(0)
                skeleton = _polygon_centerline(local) if local is not None and not local.is_empty else None
            except Exception:
                skeleton = None
            if skeleton is not None and not skeleton.is_empty:
                arms_src = "skeleton"
                arms_lines = _arms_lines_from_source(seed, [skeleton], arm_cap_m)
                arm_buffer_m = float(cfg.get("arm_buffer_m_skeleton", 3.5))
            elif center_use and len(center_use) >= min_center:
                arms_src = "centerlines"
                arms_lines = _arms_lines_from_source(seed, center_use, arm_cap_m)
                arm_buffer_m = float(cfg.get("arm_buffer_m_centerlines", 5.0))
    arms_mask = None
    if arms_lines:
        corridors = [ln.buffer(arm_buffer_m, cap_style=2, join_style=2) for ln in arms_lines]
        arms_mask = unary_union(corridors) if corridors else None
    arm_count_axis = _arm_count_heading(
        arms_lines,
        seed,
        float(cfg.get("arm_count_eval_radius_m", 14.0)),
        float(cfg.get("arm_count_angle_bin_deg", 20.0)),
        float(cfg.get("arm_count_min_len_m", 3.0)),
    )
    arm_count_approach = _arm_count_approach(
        arms_lines,
        seed,
        float(cfg.get("arm_count_eval_radius_m", 14.0)),
        float(cfg.get("arm_count_angle_bin_deg", 20.0)),
        float(cfg.get("arm_count_min_len_m", 3.0)),
    )
    meta = {
        "arms_src": arms_src,
        "arms_lines": arms_lines,
        "arms_mask": arms_mask,
        "arm_cap_m": arm_cap_m,
        "arm_buffer_m_used": arm_buffer_m,
        "osm_links_used_count": osm_links_used_count,
        "arm_count_incident": arm_count_incident,
        "arm_count_axis": arm_count_axis,
        "arm_count_approach": arm_count_approach,
        "osm_junction": best.get("point") if best is not None else None,
    }
    return meta, {
        "arms_lines": arms_lines,
        "arms_mask": arms_mask,
        "arms_src": arms_src,
        "arm_cap_m": arm_cap_m,
        "arm_buffer_m_used": arm_buffer_m,
        "osm_links_used_count": osm_links_used_count,
        "arm_count_incident": arm_count_incident,
        "arm_count_axis": arm_count_axis,
        "arm_count_approach": arm_count_approach,
        "osm_junction": best.get("point") if best is not None else None,
    }


def _shape_metrics(
    seed: Point,
    refined: Polygon,
    road_poly: Polygon,
    meta: dict,
    cfg: dict,
    arms_src: str,
) -> dict:
    local = meta.get("local")
    arms_geom = meta.get("arms")
    arms_lines = meta.get("arms_lines") or arms_geom
    circ = _circularity(refined)
    aspect = _aspect_ratio(refined)
    overlap = _overlap_with_road(refined, road_poly)
    arm_count_legacy = _arm_count_branches(seed, arms_geom, float(cfg.get("arm_count_seed_radius_m", 6.0)))
    arm_count_axis = _arm_count_heading(
        arms_lines,
        seed,
        float(cfg.get("arm_count_eval_radius_m", 14.0)),
        float(cfg.get("arm_count_angle_bin_deg", 20.0)),
        float(cfg.get("arm_count_min_len_m", 3.0)),
    )
    arm_count_approach = _arm_count_approach(
        arms_lines,
        seed,
        float(cfg.get("arm_count_eval_radius_m", 14.0)),
        float(cfg.get("arm_count_angle_bin_deg", 20.0)),
        float(cfg.get("arm_count_min_len_m", 3.0)),
    )
    return {
        "circularity": circ,
        "aspect_ratio": aspect,
        "overlap_road": overlap,
        "arm_count": arm_count_approach,
        "arm_count_heading": arm_count_approach,
        "arm_count_axis": arm_count_axis,
        "arm_count_approach": arm_count_approach,
        "arm_count_legacy": arm_count_legacy,
        "has_arms": 1 if arms_geom is not None and not arms_geom.is_empty else 0,
        "arms_area": float(arms_geom.area) if arms_geom is not None and not arms_geom.is_empty else 0.0,
        "local_area": float(local.area) if local is not None and not local.is_empty else 0.0,
        "refined_area": float(refined.area),
        "local": local,
        "arms": arms_geom,
        "arms_lines": arms_lines,
        "arms_src": arms_src,
        "arms_mask": meta.get("arms_mask"),
        "arm_cap_m": meta.get("arm_cap_m"),
        "arm_buffer_m_used": meta.get("arm_buffer_m_used"),
        "arm_count_incident": meta.get("arm_count_incident"),
        "osm_junction": meta.get("osm_junction"),
        "core_radius_m": meta.get("core_radius_m"),
        "coverage_local": meta.get("coverage_local"),
        "quality_low": meta.get("quality_low", 0),
        "reason": meta.get("reason") or "refined",
    }


def _filter_osm_features(features: List[dict], allowlist: List[str]) -> List[dict]:
    if not allowlist:
        return features
    allow = {str(a).strip() for a in allowlist if str(a).strip()}
    if not allow:
        return features
    allow.update({a for a in allow if a.endswith("_link")})
    allow.update({f"{a}_link" for a in list(allow) if not a.endswith("_link")})
    kept = []
    for feat in features:
        props = feat.get("properties") or {}
        hw = props.get("highway")
        if hw is None:
            continue
        if isinstance(hw, (list, tuple)):
            values = {str(v).strip() for v in hw}
        else:
            values = {str(hw).strip()}
        if values & allow:
            kept.append(feat)
    return kept


def _osm_highway_counts(features: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for feat in features:
        props = feat.get("properties") or {}
        hw = props.get("highway")
        if hw is None:
            continue
        if isinstance(hw, (list, tuple)):
            values = [str(v).strip() for v in hw]
        else:
            values = [str(hw).strip()]
        for v in values:
            if not v:
                continue
            counts[v] = counts.get(v, 0) + 1
    return counts


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


def _osm_degree_seeds(
    lines: List[LineString],
    snap_m: float,
    min_degree: int,
    highway_by_line: List[str],
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
        hw = highway_by_line[line_idx] if line_idx < len(highway_by_line) else ""
        if hw:
            node_highways.setdefault(n1, set()).add(hw)
            node_highways.setdefault(n2, set()).add(hw)
    seeds = []
    for idx, neighbors in edges.items():
        degree = len(neighbors)
        if degree >= min_degree:
            x, y = nodes[idx]
            types = sorted(node_highways.get(idx, set()))
            seeds.append((Point(x, y), degree, types))
    return seeds


def _traj_seeds_from_line(line: LineString, angle_deg: float, min_sep_m: float) -> List[Point]:
    coords = list(line.coords)
    if len(coords) < 3:
        return []
    out = []
    last = None
    thr = math.radians(angle_deg)
    for i in range(1, len(coords) - 1):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 == 0 or n2 == 0:
            continue
        cos_ang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos_ang = max(-1.0, min(1.0, cos_ang))
        ang = math.acos(cos_ang)
        if ang >= thr:
            pt = Point(x1, y1)
            if last is None or pt.distance(last) >= min_sep_m:
                out.append(pt)
                last = pt
    return out


def _centerline_junctions(lines: List[LineString]) -> List[Point]:
    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            inter = lines[i].intersection(lines[j])
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                points.append(inter)
            elif inter.geom_type == "MultiPoint":
                points.extend(list(inter.geoms))
    return points


def _nearest_point_on_lines(pt: Point, lines: List[LineString]) -> Optional[Point]:
    best = None
    best_dist = float("inf")
    for line in lines:
        proj = line.project(pt)
        candidate = line.interpolate(proj)
        dist = pt.distance(candidate)
        if dist < best_dist:
            best = candidate
            best_dist = dist
    return best


def _refine_seed(seed: Point, lines: List[LineString], search_radius: float) -> Tuple[Point, str, float]:
    if not lines:
        return seed, "none", 0.0
    junctions = _centerline_junctions(lines)
    best = None
    best_dist = float("inf")
    for j in junctions:
        dist = seed.distance(j)
        if dist < best_dist:
            best_dist = dist
            best = j
    if best is not None and best_dist <= search_radius:
        conf = max(0.0, 1.0 - best_dist / max(1.0, search_radius))
        return best, "geometry", conf
    candidate = _nearest_point_on_lines(seed, lines)
    if candidate is not None and seed.distance(candidate) <= search_radius:
        dist = seed.distance(candidate)
        conf = max(0.0, 1.0 - dist / max(1.0, search_radius))
        return candidate, "geometry", conf
    return seed, "none", 0.0


def _refine_seed_markings(
    seed: Point,
    features: List[dict],
    max_dist_m: float,
    min_conf: float,
) -> Optional[Point]:
    if not features:
        return None
    best_geom = None
    best_dist = None
    for feat in features:
        geom = feat.get("geometry")
        if geom is None or geom.is_empty:
            continue
        props = feat.get("properties") or {}
        conf = props.get("conf")
        if conf is None:
            conf = props.get("score")
        if conf is not None:
            try:
                if float(conf) < min_conf:
                    continue
            except (TypeError, ValueError):
                pass
        dist = seed.distance(geom)
        if dist <= max_dist_m and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_geom = geom
    if best_geom is None:
        return None
    try:
        from shapely.ops import nearest_points

        _, p2 = nearest_points(seed, best_geom)
        return p2
    except Exception:
        return seed


def _shape_from_seed(
    seed: Point,
    road_poly: Polygon,
    centerlines: List[LineString],
    osm_lines: List[LineString],
    osm_junctions: List[dict],
    cfg: dict,
) -> Tuple[Optional[Polygon], dict]:
    def _lines_near_seed(lines: List[LineString], seed_pt: Point, radius_m: float) -> List[LineString]:
        if not lines:
            return []
        radius = max(float(radius_m), 0.1)
        area = seed_pt.buffer(radius)
        out = []
        for line in lines:
            if line is None or line.is_empty:
                continue
            if line.intersects(area):
                out.append(line)
        return out

    if bool(cfg.get("arm_shape_v3_enabled", False)):
        arms_meta, _ = _build_arms_v3(seed, road_poly, centerlines, osm_lines, osm_junctions, cfg)
        if (arms_meta.get("arms_src") or "") == "osm_junction" and int(arms_meta.get("arm_count_incident") or 0) < 3:
            best = None
            best_dist = None
            for j in osm_junctions:
                pt = j.get("point")
                if pt is None:
                    continue
                dist = seed.distance(pt)
                if best_dist is None or dist < best_dist:
                    best = j
                    best_dist = dist
            degree = int(best.get("degree") or 0) if best is not None else 0
            if degree >= 3:
                arms_meta["arm_count_incident"] = degree
                if int(arms_meta.get("osm_links_used_count") or 0) < degree:
                    arms_meta["osm_links_used_count"] = degree
        arm_cap_m = float(arms_meta.get("arm_cap_m", cfg.get("arm_cap_m", 25.0)))
        core_radius = float(cfg.get("core_radius_m", 0.0))
        if core_radius <= 0:
            core_ratio = float(cfg.get("core_radius_ratio", 0.6))
            core_min = float(cfg.get("core_radius_min_m", 12.0))
            core_max = float(cfg.get("core_radius_max_m", 22.0))
            core_radius = _clamp(core_ratio * arm_cap_m, core_min, core_max)
        road_local = road_poly.intersection(seed.buffer(arm_cap_m))
        road_local = _make_valid(road_local)
        road_local_src = "road_polygon"
        road_local_retry_steps = 0
        road_local_retry_radius = arm_cap_m
        if road_local is None or road_local.is_empty:
            for step, retry_r in enumerate((50.0, 80.0), start=1):
                retry_local = road_poly.intersection(seed.buffer(retry_r))
                retry_local = _make_valid(retry_local)
                if retry_local is not None and not retry_local.is_empty:
                    road_local = retry_local
                    road_local_retry_steps = step
                    road_local_retry_radius = retry_r
                    break
        if road_local is not None and not road_local.is_empty:
            max_offset = float(cfg.get("road_local_max_offset_m", 30.0))
            if max_offset > 0 and seed.distance(road_local) > max_offset:
                road_local = None
        if road_local is None or road_local.is_empty:
            sat_poly = cfg.get("_sat_road_poly")
            if sat_poly is not None and not sat_poly.is_empty:
                sat_local = sat_poly.intersection(seed.buffer(80.0))
                sat_local = _make_valid(sat_local)
                if sat_local is not None and not sat_local.is_empty:
                    road_local = sat_local
                    road_local_src = "sat_fill"
                    road_local_retry_steps = 2
                    road_local_retry_radius = 80.0
        if road_local is None or road_local.is_empty:
            road_local = _make_valid(seed.buffer(12.0))
            road_local_src = "seed_circle"
            road_local_retry_steps = 2
            road_local_retry_radius = 12.0
        core = road_poly.intersection(seed.buffer(core_radius))
        core = _make_valid(core)
        if road_local_src != "road_polygon":
            core = road_local
        arms_mask = arms_meta.get("arms_mask")
        shape_arms = road_poly.intersection(arms_mask) if arms_mask is not None and not arms_mask.is_empty else None
        if shape_arms is not None and not shape_arms.is_empty:
            refined = _make_valid(shape_arms.union(core))
            reason = "refined_arms"
        else:
            refined = _make_valid(core)
            reason = "core_only"
        if road_local_src == "sat_fill":
            reason = "sat_fill_fallback"
        if road_local_src == "seed_circle":
            reason = "seed_circle_fallback"

        coverage_min = float(cfg.get("coverage_local_min", 0.9))
        coverage_local = 0.0
        if road_local is not None and not road_local.is_empty and refined is not None and not refined.is_empty:
            coverage_local = float(refined.intersection(road_local).area) / max(1e-6, float(road_local.area))

        retry_used = False
        arm_count_heading = int(arms_meta.get("arm_count_heading", 0) or 0)
        if coverage_local < coverage_min and arm_count_heading >= 3:
            arm_buffer_retry = float(arms_meta.get("arm_buffer_m_used", cfg.get("arm_buffer_m", 5.0))) + float(
                cfg.get("coverage_retry_buffer_add_m", 0.5)
            )
            arms_lines = arms_meta.get("arms_lines") or []
            if arms_lines:
                corridors = [ln.buffer(arm_buffer_retry, cap_style=2, join_style=2) for ln in arms_lines]
                arms_mask_retry = unary_union(corridors) if corridors else None
                shape_arms_retry = (
                    road_poly.intersection(arms_mask_retry)
                    if arms_mask_retry is not None and not arms_mask_retry.is_empty
                    else None
                )
                if shape_arms_retry is not None and not shape_arms_retry.is_empty:
                    refined_retry = _make_valid(shape_arms_retry.union(core))
                    cov_retry = (
                        float(refined_retry.intersection(road_local).area) / max(1e-6, float(road_local.area))
                        if road_local is not None and not road_local.is_empty
                        else 0.0
                    )
                    if cov_retry >= coverage_min:
                        refined = refined_retry
                        arms_meta["arms_mask"] = arms_mask_retry
                        arms_meta["arm_buffer_m_used"] = arm_buffer_retry
                        coverage_local = cov_retry
                        retry_used = True

        quality_low = False
        if retry_used:
            reason = "coverage_retry"
        if coverage_local < coverage_min and road_local is not None and not road_local.is_empty:
            refined = road_local
            quality_low = True
            reason = "coverage_fallback_road_local"
            coverage_local = 1.0

        meta = {
            "local": road_local,
            "arms": arms_meta.get("arms_mask"),
            "arms_lines": arms_meta.get("arms_lines"),
            "arms_mask": arms_meta.get("arms_mask"),
            "arms_src": arms_meta.get("arms_src"),
            "arm_cap_m": arm_cap_m,
            "arm_buffer_m_used": float(arms_meta.get("arm_buffer_m_used", cfg.get("arm_buffer_m", 5.0))),
            "core_radius_m": core_radius,
            "coverage_local": coverage_local,
            "quality_low": 1 if quality_low else 0,
            "reason": reason,
            "road_local_retry_steps": road_local_retry_steps,
            "road_local_retry_radius_m": road_local_retry_radius,
            "road_local_src": road_local_src,
            "osm_links_used_count": arms_meta.get("osm_links_used_count", 0),
            "arm_count_incident": arms_meta.get("arm_count_incident"),
            "arm_count_axis": arms_meta.get("arm_count_axis"),
            "arm_count_approach": arms_meta.get("arm_count_approach"),
            "osm_junction": arms_meta.get("osm_junction"),
            "osm_incident_insufficient": 1 if (arms_meta.get("arm_count_incident") or 0) < 3 else 0,
        }
        metrics = _shape_metrics(seed, refined, road_poly, meta, cfg, meta.get("arms_src") or "none")
        metrics["road_local_retry_steps"] = road_local_retry_steps
        metrics["road_local_retry_radius_m"] = road_local_retry_radius
        metrics["road_local_src"] = road_local_src
        if road_local_src == "seed_circle":
            metrics["quality_flag_override"] = "very_low"
        if (metrics.get("arms_src") or "") == "osm_junction" and int(metrics.get("arm_count_incident") or 0) < 3:
            best = None
            best_dist = None
            for j in osm_junctions:
                pt = j.get("point")
                if pt is None:
                    continue
                dist = seed.distance(pt)
                if best_dist is None or dist < best_dist:
                    best = j
                    best_dist = dist
            degree = int(best.get("degree") or 0) if best is not None else 0
            if degree >= 3:
                metrics["arm_count_incident"] = degree
                metrics["osm_incident_insufficient"] = 0
        return refined, metrics

    min_src_arm_count = int(cfg.get("min_arm_count_for_src", 2))

    center_search = float(cfg.get("seed_search_radius_m_centerlines", cfg.get("seed_search_radius_m", 40.0)))
    center_use = _lines_near_seed(centerlines, seed, center_search) if centerlines else []
    refined, meta = _refine_intersection_polygon(
        seed_pt=seed,
        poly_candidate=seed.buffer(float(cfg["radius_m"])),
        road_polygon=road_poly,
        centerlines=center_use,
        cfg=cfg,
    )
    if refined is None or refined.is_empty:
        return None, {"reason": "empty", "arms_src": "none"}
    metrics = _shape_metrics(seed, refined, road_poly, meta, cfg, "centerlines")

    if metrics["has_arms"] == 0 or metrics["arm_count_approach"] < min_src_arm_count:
        if osm_lines:
            osm_search = float(cfg.get("seed_search_radius_m_osm", cfg.get("seed_search_radius_m", 40.0)))
            osm_use = _lines_near_seed(osm_lines, seed, osm_search)
            if len(osm_use) <= 1:
                osm_search = osm_search + float(cfg.get("seed_search_radius_m_osm_expand", 15.0))
                osm_use = _lines_near_seed(osm_lines, seed, osm_search)
            if not osm_use:
                osm_use = osm_lines
            cfg_osm = dict(cfg)
            cfg_osm["arm_length_m"] = float(cfg.get("arm_length_m_osm", float(cfg.get("arm_length_m", 25.0)) + 10.0))
            cfg_osm["arm_buffer_m"] = float(cfg.get("arm_buffer_m_osm", 5.0))
            refined_osm, meta_osm = _refine_intersection_polygon(
                seed_pt=seed,
                poly_candidate=seed.buffer(float(cfg["radius_m"])),
                road_polygon=road_poly,
                centerlines=osm_use,
                cfg=cfg_osm,
            )
            if refined_osm is not None and not refined_osm.is_empty:
                metrics_osm = _shape_metrics(seed, refined_osm, road_poly, meta_osm, cfg_osm, "osm")
                metrics_osm["osm_links_used_count"] = len(osm_use)
                if metrics_osm["arm_count_approach"] > metrics["arm_count_approach"] or metrics["has_arms"] == 0:
                    refined, metrics = refined_osm, metrics_osm

    if metrics["has_arms"] == 0 or metrics["arm_count_approach"] < min_src_arm_count:
        cfg_sk = dict(cfg)
        sk_search = float(cfg.get("seed_search_radius_m_skeleton", cfg.get("seed_search_radius_m", 40.0)))
        cfg_sk["arm_length_m"] = float(cfg.get("arm_length_m_skeleton", float(cfg.get("arm_length_m", 25.0)) + 10.0))
        cfg_sk["arm_buffer_m"] = float(cfg.get("arm_buffer_m_skeleton", 5.0))
        cfg_sk["radius_m"] = max(float(cfg.get("radius_m", 20.0)), sk_search)
        try:
            local = road_poly.buffer(float(cfg_sk.get("road_buffer_m", 1.0))).intersection(
                seed.buffer(float(cfg_sk["radius_m"]))
            )
            local = local.buffer(0)
            skeleton = _polygon_centerline(local) if local is not None and not local.is_empty else None
        except Exception:
            skeleton = None
        if skeleton is not None and not skeleton.is_empty:
            refined_sk, meta_sk = _refine_intersection_polygon(
                seed_pt=seed,
                poly_candidate=seed.buffer(float(cfg_sk["radius_m"])),
                road_polygon=road_poly,
                centerlines=[skeleton],
                cfg=cfg_sk,
            )
            if refined_sk is not None and not refined_sk.is_empty:
                metrics_sk = _shape_metrics(seed, refined_sk, road_poly, meta_sk, cfg_sk, "skeleton")
                if metrics_sk["arm_count_approach"] > metrics["arm_count_approach"] or metrics["has_arms"] == 0:
                    refined, metrics = refined_sk, metrics_sk

    if metrics.get("has_arms") == 0:
        metrics["arms_src"] = "none"
        metrics["reason"] = "no_arms"
    elif metrics.get("arm_count_approach", 0) <= 1:
        metrics["reason"] = "weak_arms"
    return refined, metrics


def _shape_gate_ok(metrics: dict, gate: dict, area_m2: float) -> bool:
    if area_m2 < gate["min_area_m2"] or area_m2 > gate["max_area_m2"]:
        return False
    if metrics.get("arms_src") in (None, "none") or metrics.get("has_arms") == 0:
        return False
    if metrics["overlap_road"] < gate["min_overlap_road"]:
        return False
    if metrics["circularity"] > gate["max_circularity"]:
        return False
    return True


def _iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


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


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_missing_reason_summary(out_csv: Path, expected_drives: List[str], report_type: str, run_id: str) -> dict:
    from collections import Counter

    rows = []
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    counts = Counter()
    non_ok = []
    for row in rows:
        reason = (row.get("missing_reason") or "").strip()
        norm = "" if reason in {"", "N/A", "OK"} else reason
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
    ap.add_argument("--index", required=True, help="postopt_index.jsonl")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--config", default="configs/intersections_v2.yaml")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    run_cfg = cfg.get("run", {}) or {}
    seeds_cfg = cfg.get("seeds", {}) or {}
    refine_cfg = cfg.get("refine", {}) or {}
    shape_cfg = cfg.get("shape", {}) or {}
    gate_cfg = cfg.get("gate", {}) or {}
    debug_cfg = cfg.get("debug", {}) or {}
    debug_drop_trace = bool(debug_cfg.get("drop_trace", False) or debug_cfg.get("debug_drop_trace", False))

    expected_drives = []
    expected_path = Path(run_cfg.get("expected_drives_file", "configs/golden_drives.txt"))
    if expected_path.exists():
        expected_drives = [ln.strip() for ln in expected_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    entries = _read_index(Path(args.index))
    entries = [
        e for e in entries
        if e.get("stage") == args.stage
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]
    if args.candidate:
        entries = [e for e in entries if e.get("candidate_id") == args.candidate]
    if not entries:
        raise SystemExit("ERROR: no entries found")

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"intersections_v2_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_outputs = out_dir / "outputs"
    out_outputs.mkdir(parents=True, exist_ok=True)

    road_polys_by_drive: Dict[str, Polygon] = {}
    drive_bboxes_wgs84: Dict[str, Tuple[float, float, float, float]] = {}
    for entry in entries:
        drive = str(entry.get("drive") or "")
        outputs_dir = Path(entry.get("outputs_dir") or "")
        if not drive or not outputs_dir.exists():
            continue
        road_path = outputs_dir / "road_polygon.geojson"
        if not road_path.exists():
            continue
        road_feats = _read_geojson(road_path)
        road_polys = _collect_polys(road_feats)
        if road_polys:
            road_poly = unary_union(road_polys)
            road_polys_by_drive[drive] = road_poly
            bbox = _drive_bbox_wgs84(entry, road_poly)
            if bbox is not None:
                drive_bboxes_wgs84[drive] = bbox

    sat_road_poly = None
    sat_road_path = shape_cfg.get("sat_road_polygon_path")
    if sat_road_path:
        sat_path = Path(sat_road_path)
        if sat_path.exists():
            sat_feats = _read_geojson(sat_path)
            sat_polys = _collect_polys(sat_feats)
            if sat_polys:
                sat_poly = unary_union(sat_polys)
                minx, miny, _, _ = sat_poly.bounds
                if _is_wgs84_coords(minx, miny):
                    sat_road_poly = _to_utm32(sat_poly)
                else:
                    sat_road_poly = sat_poly
                shape_cfg["_sat_road_poly"] = sat_road_poly

    sat_features_by_drive: Dict[str, List[dict]] = {}
    sat_outputs_dir = Path(seeds_cfg.get("sat_outputs_dir", ""))
    if seeds_cfg.get("sat_enabled", True) and seeds_cfg.get("sat_seed_input_mode") == "polygons" and sat_outputs_dir.exists():
        sat_path = sat_outputs_dir / "intersections_sat.geojson"
        if not sat_path.exists():
            sat_path = sat_outputs_dir / "intersections_sat_wgs84.geojson"
        if sat_path.exists():
            sat_feats = _read_geojson(sat_path)
            missing_drive = sum(1 for f in sat_feats if not (f.get("properties") or {}).get("drive_id"))
            if missing_drive:
                sat_features_by_drive, stats, unmatched = _assign_sat_features_to_drives(sat_feats, entries, road_polys_by_drive)
                print(
                    f"[V2] SAT assign: total={stats['sat_total']} matched={stats['sat_matched']} "
                    f"unmatched={stats['sat_unmatched']}"
                )
                if unmatched:
                    sat_unassigned = out_outputs / "sat_unassigned.csv"
                    rows = []
                    for feat in unmatched:
                        props = feat.get("properties") or {}
                        rows.append(
                            {
                                "sat_confidence": props.get("sat_confidence"),
                                "drive_id": props.get("drive_id"),
                                "tile_id": props.get("tile_id"),
                            }
                        )
                    _write_csv(sat_unassigned, rows)
            else:
                for feat in sat_feats:
                    drive_id = str((feat.get("properties") or {}).get("drive_id") or "")
                    if not drive_id:
                        continue
                    sat_features_by_drive.setdefault(drive_id, []).append(feat)

    osm_path_cfg = str(seeds_cfg.get("osm_roads_path") or "")
    osm_global_path = Path(osm_path_cfg) if osm_path_cfg else None
    if osm_global_path and not osm_global_path.exists():
        print(f"[V2][WARN] configured osm_roads_path not found: {osm_global_path}")
        osm_global_path = None
    if osm_global_path is None:
        osm_global_path = _find_latest_osm_path()
        if osm_global_path:
            print(f"[V2] using OSM roads from: {osm_global_path}")
    if osm_global_path is None:
        raise SystemExit("ERROR: drivable_roads.geojson not found. Set seeds.osm_roads_path to a drivable_roads.geojson path.")

    osm_seeds_by_drive: Dict[str, List[dict]] = {}
    if seeds_cfg.get("osm_enabled", True) and osm_global_path.exists():
        osm_feats = _read_geojson(osm_global_path)
        all_counts = _osm_highway_counts(osm_feats)
        allowlist = seeds_cfg.get("osm_highway_allowlist") or []
        osm_feats = _filter_osm_features(osm_feats, allowlist)
        filt_counts = _osm_highway_counts(osm_feats)
        if filt_counts:
            print(f"[V2] OSM highway counts (filtered): {filt_counts}")
        else:
            print(f"[V2][WARN] OSM highway counts empty after filter; raw counts: {all_counts}")
        osm_lines, osm_types = _collect_lines_with_types(osm_feats)
        if osm_lines:
            c0 = list(osm_lines[0].coords)[0]
            is_wgs84 = _is_wgs84_coords(c0[0], c0[1])
        else:
            is_wgs84 = False
        if is_wgs84:
            osm_lines_utm = [_to_utm32(line) for line in osm_lines]
            seeds_in_utm = True
        else:
            osm_lines_utm = osm_lines
            seeds_in_utm = False
        snap_m = float(seeds_cfg.get("osm_snap_m", 2.0))
        min_degree = int(seeds_cfg.get("osm_min_degree", 3))
        osm_seeds = _osm_degree_seeds(osm_lines_utm, snap_m, min_degree, osm_types)
        if not osm_seeds and min_degree > 2:
            osm_seeds = _osm_degree_seeds(osm_lines_utm, snap_m, 2, osm_types)
            if osm_seeds:
                print(f"[V2][WARN] no osm degree>={min_degree} seeds, fallback to degree>=2")
                min_degree = 2
        if not osm_seeds:
            junctions = _centerline_junctions(osm_lines_utm)
            if junctions:
                print("[V2][WARN] no osm degree seeds, fallback to osm intersections")
                osm_seeds = [(pt, 2, []) for pt in junctions]

        buffer_m = float(seeds_cfg.get("drive_bbox_buffer_m", 150.0))
        assign_max_dist_m = float(seeds_cfg.get("assign_max_dist_m", 200.0))
        multi_assign = bool(seeds_cfg.get("multi_assign", True))
        drive_bboxes_utm = {d: _bbox_to_utm(b) for d, b in drive_bboxes_wgs84.items()}
        osm_lines_by_drive: Dict[str, List[LineString]] = {d: [] for d in drive_bboxes_utm}
        for line in osm_lines_utm:
            if line is None or line.is_empty:
                continue
            lminx, lminy, lmaxx, lmaxy = line.bounds
            for d, bbox in drive_bboxes_utm.items():
                bminx, bminy, bmaxx, bmaxy = bbox
                bbox_buf = (bminx - buffer_m, bminy - buffer_m, bmaxx + buffer_m, bmaxy + buffer_m)
                if _bbox_intersects((lminx, lminy, lmaxx, lmaxy), bbox_buf):
                    osm_lines_by_drive.setdefault(d, []).append(line)
        osm_junctions_by_drive: Dict[str, List[dict]] = {}
        for d, lines in osm_lines_by_drive.items():
            if not lines:
                continue
            osm_junctions_by_drive[d] = _osm_junctions_with_incident(lines, snap_m, min_degree, d)
        for pt, degree, hw_types in osm_seeds:
            pt_wgs = _geom_to_wgs84(pt) if seeds_in_utm else pt
            hits = []
            for drive, bbox in drive_bboxes_wgs84.items():
                bbox_buf = _expand_bbox(bbox, buffer_m)
                if bbox_buf[0] <= pt_wgs.x <= bbox_buf[2] and bbox_buf[1] <= pt_wgs.y <= bbox_buf[3]:
                    hits.append(drive)
            if hits:
                if not multi_assign:
                    hits = hits[:1]
                for drive in hits:
                    osm_seeds_by_drive.setdefault(drive, []).append(
                        {
                            "seed": pt,
                            "src_seed": "osm",
                            "reason": f"osm_degree{min_degree}",
                            "conf_prior": min(1.0, 0.4 + 0.1 * float(degree)),
                            "radius_m": float(seeds_cfg.get("seed_radius_default_m", 18.0)),
                            "degree": degree,
                            "highway_types": hw_types,
                            "assign_reason": "bbox_buffer",
                            "multi_assign": 1 if len(hits) > 1 else 0,
                        }
                    )
                continue
            if assign_max_dist_m > 0:
                best_drive = None
                best_dist = float("inf")
                for drive, bbox in drive_bboxes_wgs84.items():
                    dist = _dist_point_bbox_m(pt_wgs, bbox)
                    if dist < best_dist:
                        best_dist = dist
                        best_drive = drive
                if best_drive is not None and best_dist <= assign_max_dist_m:
                    osm_seeds_by_drive.setdefault(best_drive, []).append(
                        {
                            "seed": pt,
                            "src_seed": "osm",
                            "reason": f"osm_degree{min_degree}",
                            "conf_prior": min(1.0, 0.4 + 0.1 * float(degree)),
                            "radius_m": float(seeds_cfg.get("seed_radius_default_m", 18.0)),
                            "degree": degree,
                            "highway_types": hw_types,
                            "assign_reason": "nearest_drive_fallback",
                            "multi_assign": 0,
                        }
                    )

    rows = []
    seen_drives = set()
    for entry in entries:
        drive = str(entry.get("drive") or "")
        outputs_dir = Path(entry.get("outputs_dir"))
        if not drive or not outputs_dir.exists():
            continue
        seen_drives.add(drive)
        drive_dir = out_outputs / drive
        drive_dir.mkdir(parents=True, exist_ok=True)
        final_path = drive_dir / "intersections_final.geojson"
        if args.resume and final_path.exists():
            continue

        road_path = outputs_dir / "road_polygon.geojson"
        center_path = outputs_dir / "centerlines_both.geojson"
        if not center_path.exists():
            center_path = outputs_dir / "centerlines.geojson"
        if not road_path.exists():
            rows.append(
                {
                    "drive_id": drive,
                    "status": "FAIL",
                    "missing_reason": "missing_inputs",
                    "final_cnt": 0,
                }
            )
            continue

        road_feats = _read_geojson(road_path)
        road_polys = _collect_polys(road_feats)
        road_poly = unary_union(road_polys) if road_polys else Polygon()

        center_lines: List[LineString] = []
        if center_path.exists():
            center_feats = _read_geojson(center_path)
            center_lines = _collect_lines(center_feats)
        if not center_lines:
            print(f"[V2][WARN] centerlines missing/empty for {drive}, will fallback to OSM/skeleton arms")

        markings_feats: List[dict] = []
        markings_by_class: Dict[str, List[dict]] = {}
        if refine_cfg.get("markings_enabled"):
            store_dir = refine_cfg.get("markings_feature_store_dir")
            if store_dir:
                store_path = Path(store_dir)
                feat_map = _load_clean_evidence(drive, store_path)
                if feat_map:
                    print(f"[V2][MARKINGS] using evidence_clean for {drive}")
                else:
                    feat_map = load_image_features(drive, None, store_path)
                for cls in ("stop_line", "crosswalk", "gore_marking", "arrow"):
                    feats = feat_map.get(cls) or []
                    markings_feats.extend(feats)
                    markings_by_class[cls] = feats

        missing_reasons = []
        if not center_lines:
            missing_reasons.append("centerlines_missing_or_empty")
        seed_features = []
        refined_seed_features = []
        debug_local = []
        debug_arms = []
        debug_arms_mask = []
        debug_arms_lines = []
        debug_refined = []
        debug_pruned = []
        debug_osm_junctions = []
        debug_osm_arms_lines = []
        drop_trace = []
        armcount_rows = []

        seeds: List[dict] = []
        seed_counts = {"traj": 0, "osm": 0, "sat": 0, "geom": 0}
        if seeds_cfg.get("traj_enabled", True):
            traj_path_tmpl = str(seeds_cfg.get("traj_points_path_template") or "")
            traj_path = Path(traj_path_tmpl.format(drive=drive)) if traj_path_tmpl else Path()
            traj_seeds = []
            if traj_path_tmpl and traj_path.exists():
                traj_feats = _read_geojson(traj_path)
                traj_lines = _collect_lines(traj_feats)
                angle_deg = float(seeds_cfg.get("traj_turn_angle_deg", 35.0))
                min_sep = float(seeds_cfg.get("traj_min_sep_m", 25.0))
                for line in traj_lines:
                    traj_seeds.extend(_traj_seeds_from_line(line, angle_deg, min_sep))
            else:
                missing_reasons.append("missing_traj_inputs")
            for pt in traj_seeds:
                seeds.append(
                    {
                        "seed": pt,
                        "src_seed": "traj",
                        "reason": "traj_split",
                        "conf_prior": 0.8,
                        "radius_m": float(seeds_cfg.get("seed_radius_default_m", 18.0)),
                    }
                )
                seed_counts["traj"] += 1

        osm_primary = bool(seeds_cfg.get("osm_junction_as_primary", False))
        junction_seed_radius = float(seeds_cfg.get("seed_dedup_radius_m", 5.0))
        junction_seed_ids = set()
        if seeds_cfg.get("osm_enabled", True):
            osm_junctions = osm_junctions_by_drive.get(drive, [])
            if osm_primary and osm_junctions:
                for j in osm_junctions:
                    pt = j.get("point")
                    if pt is None:
                        continue
                    junction_id = j.get("junction_id") or f"{drive}_osm_junction"
                    junction_seed_ids.add(junction_id)
                    seeds.append(
                        {
                            "seed": pt,
                            "src_seed": "osm",
                            "reason": "osm_junction_primary",
                            "conf_prior": min(1.0, 0.4 + 0.1 * float(j.get("degree") or 0)),
                            "radius_m": float(seeds_cfg.get("seed_radius_default_m", 18.0)),
                            "degree": int(j.get("degree") or 0),
                            "junction_id": junction_id,
                        }
                    )
                    seed_counts["osm"] += 1
            if not osm_primary:
                for item in osm_seeds_by_drive.get(drive, []):
                    seeds.append(item)
                    seed_counts["osm"] += 1
            if drive not in osm_seeds_by_drive:
                missing_reasons.append("missing_osm_inputs")

        if seeds_cfg.get("geom_enabled", True):
            geom_inter_path = outputs_dir / "intersections_algo.geojson"
            geom_feats = _read_geojson(geom_inter_path)
            for feat in geom_feats:
                poly = shape(feat.get("geometry")) if feat.get("geometry") else None
                if poly is None or poly.is_empty:
                    continue
                radius = math.sqrt(poly.area / math.pi) if poly.area > 0 else float(seeds_cfg.get("seed_radius_default_m", 18.0))
                seeds.append(
                    {
                        "seed": poly.centroid,
                        "src_seed": "geom",
                        "reason": "geom_junction",
                        "conf_prior": 0.6,
                        "radius_m": float(radius),
                    }
                )
                seed_counts["geom"] += 1

        if seeds_cfg.get("sat_enabled", True) and seeds_cfg.get("sat_seed_input_mode") == "polygons":
            sat_feats = sat_features_by_drive.get(drive, [])
            for feat in sat_feats:
                poly = shape(feat.get("geometry")) if feat.get("geometry") else None
                if poly is None or poly.is_empty:
                    continue
                props = feat.get("properties") or {}
                conf = props.get("sat_confidence") or props.get("conf")
                radius = math.sqrt(poly.area / math.pi) if poly.area > 0 else float(seeds_cfg.get("seed_radius_default_m", 18.0))
                seeds.append(
                    {
                        "seed": poly.centroid,
                        "src_seed": "sat",
                        "reason": "sat_polygon",
                        "conf_prior": float(conf) if isinstance(conf, (int, float)) else 0.5,
                        "radius_m": float(radius),
                    }
                )
                seed_counts["sat"] += 1

        refined_seeds = []
        markings_hits = 0
        markings_query_counts = {k: 0 for k in ("stop_line", "crosswalk", "gore_marking", "arrow")}
        seen_junctions = []
        for item in seeds:
            seed = item["seed"]
            ref_seed = seed
            refine_src = "none"
            conf_refine = None
            seed_raw = seed
            if osm_primary and item.get("junction_id"):
                if junction_seed_radius > 0 and any(seed.distance(p) <= junction_seed_radius for p in seen_junctions):
                    continue
                seen_junctions.append(seed)
                refined_seeds.append(
                    {
                        **item,
                        "seed": seed,
                        "seed_raw": seed_raw,
                        "refine_src": "junction",
                        "conf_refine": None,
                    }
                )
                seed_features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(seed),
                        "properties": {
                            "src_seed": item["src_seed"],
                            "reason": item["reason"],
                            "degree": item.get("degree"),
                            "junction_id": item.get("junction_id"),
                        },
                    }
                )
                refined_seed_features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(seed),
                        "properties": {
                            "src_seed": item["src_seed"],
                            "reason": item["reason"],
                            "refine_src": "junction",
                            "conf_refine": None,
                            "degree": item.get("degree"),
                            "junction_id": item.get("junction_id"),
                        },
                    }
                )
                continue
            if refine_cfg.get("markings_enabled") and markings_feats:
                max_dist = float(refine_cfg.get("markings_max_dist_m", 20.0))
                min_conf = float(refine_cfg.get("markings_min_conf", 0.0))
                for cls, feats in markings_by_class.items():
                    if not feats:
                        continue
                    hit = _refine_seed_markings(seed, feats, max_dist, min_conf)
                    if hit is not None:
                        markings_query_counts[cls] += 1
                marking_seed = _refine_seed_markings(seed, markings_feats, max_dist, min_conf)
                if marking_seed is not None:
                    ref_seed = marking_seed
                    refine_src = "markings"
                    conf_refine = 0.8
                    markings_hits += 1
            if refine_cfg.get("enabled", True) and refine_src != "markings":
                ref_seed, refine_src, conf_refine = _refine_seed(
                    seed,
                    center_lines,
                    float(refine_cfg.get("search_radius_m", 25.0)),
                )
            refined_seeds.append(
                {
                    **item,
                    "seed": ref_seed,
                    "seed_raw": seed_raw,
                    "refine_src": refine_src,
                    "conf_refine": conf_refine,
                }
            )

            seed_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(seed),
                    "properties": {
                        "src_seed": item["src_seed"],
                        "reason": item["reason"],
                        "degree": item.get("degree"),
                        "highway_types": item.get("highway_types"),
                    },
                }
            )
            refined_seed_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(ref_seed),
                    "properties": {
                        "src_seed": item["src_seed"],
                        "reason": item["reason"],
                        "refine_src": refine_src,
                        "conf_refine": conf_refine,
                        "degree": item.get("degree"),
                        "highway_types": item.get("highway_types"),
                    },
                }
            )

        algo_features = []
        sat_features = []
        final_candidates = []
        gate_fail_counts = {"sat": 0, "other": 0}
        no_arms_count = 0
        weak_arms_count = 0

        osm_lines = osm_lines_by_drive.get(drive, [])
        osm_seed_items = osm_seeds_by_drive.get(drive, [])
        osm_junctions = osm_junctions_by_drive.get(drive, [])
        if osm_junctions:
            for j in osm_junctions:
                pt = j.get("point")
                if pt is None:
                    continue
                debug_osm_junctions.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(pt),
                        "properties": {
                            "degree": int(j.get("degree") or 0),
                            "junction_id": j.get("junction_id"),
                            "arms_src": "osm_junction",
                        },
                    }
                )
        for seed_idx, item in enumerate(refined_seeds):
            src_seed = item["src_seed"]
            seed = item["seed"]
            seed_raw = item.get("seed_raw") or seed
            refine_src = item.get("refine_src") or "none"
            conf_refine = item.get("conf_refine")
            seed_id = item.get("junction_id") or f"{drive}_{src_seed}_{seed_idx}"
            trace_entry = None
            if debug_drop_trace:
                trace_entry = {
                    "drive_id": drive,
                    "seed_id": seed_id,
                    "src_seed": src_seed,
                    "refine_src": refine_src,
                    "seed_reason": item.get("reason") or "",
                    "kept": None,
                    "dropped_reason": "",
                    "dedup_best_match_iou": None,
                    "best_kept_seed_id": "",
                }
            shape_cfg_use = dict(shape_cfg)
            radius_m = float(item["radius_m"])
            width_lines = center_lines if center_lines else osm_lines
            if not bool(shape_cfg_use.get("arm_shape_v3_enabled", False)):
                if bool(shape_cfg_use.get("adaptive_radius_enabled", True)):
                    radius_m, width_est, osm_degree, radius_src = _adaptive_radius(
                        seed,
                        road_poly,
                        width_lines,
                        osm_seed_items,
                        shape_cfg_use,
                    )
                else:
                    width_est = float(shape_cfg_use.get("width_default_m", 12.0))
                    osm_degree = int(item.get("degree") or 2)
                    radius_src = "fixed"
                shape_cfg_use["radius_m"] = radius_m
            else:
                width_est = float(shape_cfg_use.get("width_default_m", 12.0))
                osm_degree = int(item.get("degree") or 2)
                radius_src = "arms_v3"
            poly, metrics = _shape_from_seed(seed, road_poly, center_lines, osm_lines, osm_junctions, shape_cfg_use)
            if poly is None or poly.is_empty:
                if trace_entry is not None:
                    trace_entry["dropped_reason"] = "empty_geom"
                    trace_entry["kept"] = False
                    drop_trace.append(trace_entry)
                continue
            prune_info = {"spur_pruned": 0, "spur_removed_cnt": 0, "spur_removed_len_sum": 0.0}
            road_local_info = {
                "road_local_retry_steps": metrics.get("road_local_retry_steps", 0),
                "road_local_retry_radius_m": metrics.get("road_local_retry_radius_m", 0.0),
                "road_local_src": metrics.get("road_local_src", ""),
                "quality_flag_override": metrics.get("quality_flag_override"),
            }
            if bool(shape_cfg_use.get("spur_prune_enabled", True)):
                pruned, prune_info = _spur_prune(poly, seed, radius_m, shape_cfg_use)
                if pruned is not None and not pruned.is_empty:
                    meta = {
                        "local": metrics.get("local"),
                        "arms": metrics.get("arms"),
                        "arms_lines": metrics.get("arms_lines"),
                        "arms_mask": metrics.get("arms_mask"),
                        "arm_cap_m": metrics.get("arm_cap_m"),
                        "arm_buffer_m_used": metrics.get("arm_buffer_m_used"),
                        "core_radius_m": metrics.get("core_radius_m"),
                        "coverage_local": metrics.get("coverage_local"),
                        "quality_low": metrics.get("quality_low", 0),
                        "reason": metrics.get("reason"),
                        "arm_count_incident": metrics.get("arm_count_incident"),
                        "arm_count_axis": metrics.get("arm_count_axis"),
                        "arm_count_approach": metrics.get("arm_count_approach"),
                        "osm_junction": metrics.get("osm_junction"),
                    }
                    metrics = _shape_metrics(seed, pruned, road_poly, meta, shape_cfg_use, metrics.get("arms_src") or "none")
                    metrics.update(road_local_info)
                    poly = pruned
                else:
                    metrics.update(road_local_info)
                metrics["spur_pruned"] = prune_info.get("spur_pruned", 0)
                metrics["spur_removed_cnt"] = prune_info.get("spur_removed_cnt", 0)
                metrics["spur_removed_len_sum"] = prune_info.get("spur_removed_len_sum", 0.0)
                metrics["refined_pruned"] = poly
            if trace_entry is not None:
                trace_entry.update(
                    {
                        "arms_src": metrics.get("arms_src") or "none",
                        "arm_count_incident": int(metrics.get("arm_count_incident", 0) or 0),
                        "arm_count_axis": int(metrics.get("arm_count_axis", 0) or 0),
                        "arm_count_approach": int(metrics.get("arm_count_approach", 0) or 0),
                        "has_arms": int(metrics.get("has_arms", 0) or 0),
                        "coverage_local": float(metrics.get("coverage_local", 0.0) or 0.0),
                        "refined_area_m2": float(metrics.get("refined_area", 0.0) or 0.0),
                        "reason": metrics.get("reason") or item.get("reason") or "",
                        "road_local_src": str(metrics.get("road_local_src") or ""),
                        "road_local_retry_radius_m": float(metrics.get("road_local_retry_radius_m", 0.0) or 0.0),
                    }
                )
            markings_gate_pass = None
            if refine_src == "markings":
                min_overlap = float(refine_cfg.get("markings_min_overlap_road", 0.0))
                if min_overlap > 0 and metrics.get("overlap_road", 0.0) < min_overlap:
                    markings_gate_pass = 0
                    if bool(refine_cfg.get("markings_reject_on_fail", True)):
                        fallback_seed, fallback_src, fallback_conf = _refine_seed(
                            seed_raw,
                            center_lines,
                            float(refine_cfg.get("search_radius_m", 25.0)),
                        )
                        poly_fb, metrics_fb = _shape_from_seed(
                            fallback_seed,
                            road_poly,
                            center_lines,
                            osm_lines,
                            osm_junctions,
                            shape_cfg_use,
                        )
                        if poly_fb is None or poly_fb.is_empty:
                            continue
                        seed = fallback_seed
                        refine_src = fallback_src
                        conf_refine = fallback_conf
                        poly, metrics = poly_fb, metrics_fb
                        metrics["reason"] = "markings_reject"
                else:
                    markings_gate_pass = 1
            area = float(poly.area)
            gate_ok_geom = _shape_gate_ok(metrics, gate_cfg, area)
            if not gate_ok_geom and gate_cfg.get("retry_shrink_ratio", 0.0) and not bool(shape_cfg_use.get("arm_shape_v3_enabled", False)):
                shrink_cfg = dict(shape_cfg_use)
                shrink_cfg["radius_m"] = float(shape_cfg_use["radius_m"]) * float(gate_cfg["retry_shrink_ratio"])
                poly, metrics = _shape_from_seed(seed, road_poly, center_lines, osm_lines, osm_junctions, shrink_cfg)
                area = float(poly.area) if poly is not None else 0.0
                gate_ok_geom = _shape_gate_ok(metrics, gate_cfg, area) if poly is not None else False
            if not gate_ok_geom and src_seed == "sat" and gate_cfg.get("sat_drop_on_fail", True):
                gate_fail_counts["sat"] += 1
                if trace_entry is not None:
                    trace_entry["dropped_reason"] = "gate_drop_sat"
                    trace_entry["kept"] = False
                    drop_trace.append(trace_entry)
                continue
            if not gate_ok_geom:
                gate_fail_counts["other"] += 1
                if metrics.get("local") is not None:
                    poly = metrics["local"]
                else:
                    if trace_entry is not None:
                        trace_entry["dropped_reason"] = "gate_drop_geom"
                        trace_entry["kept"] = False
                        drop_trace.append(trace_entry)
                    continue
            if poly is None or poly.is_empty:
                continue
            if metrics.get("has_arms") == 0:
                no_arms_count += 1
            arm_count_approach = int(metrics.get("arm_count_approach", metrics.get("arm_count", 0)) or 0)
            if arm_count_approach <= 1:
                weak_arms_count += 1
            quality_flag = "weak"
            if arm_count_approach >= 3:
                quality_flag = "good"
            elif arm_count_approach == 2:
                quality_flag = "ok"
            if metrics.get("quality_flag_override"):
                quality_flag = str(metrics.get("quality_flag_override"))
            shape_gate_pass = 1 if (gate_ok_geom and arm_count_approach >= 2) else 0

            props = {
                "drive_id": drive,
                "tile_id": drive,
                "src_seed": src_seed,
                "refine_src": refine_src,
                "reason": metrics.get("reason") or item.get("reason"),
                "seed_reason": item.get("reason"),
                "seed_id": seed_id,
                "conf_prior": item.get("conf_prior"),
                "conf_refine": conf_refine,
                "radius_m_used": round(float(metrics.get("arm_cap_m", radius_m) or radius_m), 3),
                "road_width_local_est": round(float(width_est), 3),
                "osm_degree": int(metrics.get("osm_degree", osm_degree) or osm_degree),
                "radius_src": radius_src,
                "arm_count": arm_count_approach,
                "arm_count_heading": arm_count_approach,
                "arm_count_axis": int(metrics.get("arm_count_axis", 0) or 0),
                "arm_count_approach": arm_count_approach,
                "arm_count_legacy": int(metrics.get("arm_count_legacy", 0) or 0),
                "arms_src": metrics.get("arms_src") or "none",
                "quality_flag": quality_flag,
                "arm_cap_m": float(metrics.get("arm_cap_m", 0.0) or 0.0),
                "arm_buffer_m_used": float(metrics.get("arm_buffer_m_used", 0.0) or 0.0),
                "core_radius_m": float(metrics.get("core_radius_m", 0.0) or 0.0),
                "coverage_local": round(float(metrics.get("coverage_local", 0.0) or 0.0), 4),
                "quality_low": int(metrics.get("quality_low", 0)),
                "road_local_retry_steps": int(metrics.get("road_local_retry_steps", 0) or 0),
                "road_local_retry_radius_m": float(metrics.get("road_local_retry_radius_m", 0.0) or 0.0),
                "road_local_src": str(metrics.get("road_local_src") or ""),
                "arm_count_incident": int(metrics.get("arm_count_incident", 0) or 0),
                "overlap_road": round(float(metrics.get("overlap_road", 0.0)), 4),
                "circularity": round(float(metrics.get("circularity", 0.0)), 4),
                "aspect_ratio": round(float(metrics.get("aspect_ratio", 0.0)), 4),
                "shape_gate_pass": shape_gate_pass,
                "has_arms": int(metrics.get("has_arms", 0)),
                "arms_area": round(float(metrics.get("arms_area", 0.0)), 3),
                "local_area": round(float(metrics.get("local_area", 0.0)), 3),
                "refined_area": round(float(metrics.get("refined_area", 0.0)), 3),
                "spur_pruned": int(metrics.get("spur_pruned", 0)),
                "spur_removed_cnt": int(metrics.get("spur_removed_cnt", 0)),
                "spur_removed_len_sum": round(float(metrics.get("spur_removed_len_sum", 0.0)), 3),
            }
            minx, miny, maxx, maxy = poly.bounds
            props["arm_span_proxy"] = round(float(max(maxx - minx, maxy - miny)), 3)
            if metrics.get("osm_links_used_count") is not None:
                props["osm_links_used_count"] = int(metrics.get("osm_links_used_count", 0))
            if markings_gate_pass is not None:
                props["markings_gate_pass"] = int(markings_gate_pass)
            feat = {"type": "Feature", "geometry": mapping(poly), "properties": props}
            if src_seed == "sat":
                sat_features.append(feat)
            else:
                algo_features.append(feat)
            trace_idx = None
            if trace_entry is not None:
                trace_entry["shape_gate_pass"] = int(shape_gate_pass)
                trace_entry["gate_ok_geom"] = int(gate_ok_geom)
                drop_trace.append(trace_entry)
                trace_idx = len(drop_trace) - 1
            final_candidates.append((poly, props, trace_idx))
            if debug_cfg.get("enable_debug_layers", True):
                if metrics.get("local") is not None:
                    debug_local.append({"type": "Feature", "geometry": mapping(metrics["local"]), "properties": {"src_seed": src_seed}})
                if metrics.get("arms") is not None and not metrics["arms"].is_empty:
                    debug_arms.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(metrics["arms"]),
                            "properties": {"src_seed": src_seed, "arms_src": metrics.get("arms_src") or "none"},
                        }
                    )
                if metrics.get("arms_mask") is not None and not metrics["arms_mask"].is_empty:
                    debug_arms_mask.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(metrics["arms_mask"]),
                            "properties": {"src_seed": src_seed, "arms_src": metrics.get("arms_src") or "none"},
                        }
                    )
                if metrics.get("arms_lines"):
                    arms_lines_geom = metrics.get("arms_lines")
                    if isinstance(arms_lines_geom, list):
                        line_items = arms_lines_geom
                    else:
                        line_items = _explode_lines(arms_lines_geom)
                    for line in line_items:
                        debug_arms_lines.append(
                            {
                                "type": "Feature",
                                "geometry": mapping(line),
                                "properties": {"src_seed": src_seed, "arms_src": metrics.get("arms_src") or "none"},
                            }
                        )
                        if (metrics.get("arms_src") or "none") == "osm_junction":
                            debug_osm_arms_lines.append(
                                {
                                    "type": "Feature",
                                    "geometry": mapping(line),
                                    "properties": {"src_seed": src_seed, "arms_src": "osm_junction"},
                                }
                            )
                debug_refined.append({"type": "Feature", "geometry": mapping(poly), "properties": {"src_seed": src_seed}})
                if metrics.get("refined_pruned") is not None and not metrics["refined_pruned"].is_empty:
                    debug_pruned.append({"type": "Feature", "geometry": mapping(metrics["refined_pruned"]), "properties": {"src_seed": src_seed}})
            if metrics.get("arms_src") == "osm_junction":
                armcount_rows.append(
                    {
                        "drive_id": drive,
                        "src_seed": src_seed,
                        "osm_degree": int(item.get("degree") or 0),
                        "arm_count_incident": int(metrics.get("arm_count_incident", 0) or 0),
                        "arm_count_axis": int(metrics.get("arm_count_axis", 0) or 0),
                        "arm_count_approach": int(metrics.get("arm_count_approach", 0) or 0),
                        "arm_cap_m": float(metrics.get("arm_cap_m", 0.0) or 0.0),
                        "coverage_local": float(metrics.get("coverage_local", 0.0) or 0.0),
                        "reason": str(metrics.get("reason") or ""),
                    }
                )

        def _priority(src: str) -> int:
            return {"traj": 4, "osm": 3, "sat": 2, "geom": 1}.get(src, 0)

        final_features = []
        for poly, props, trace_idx in final_candidates:
            kept_idx = None
            for idx, kept in enumerate(final_features):
                if _iou(poly, shape(kept["geometry"])) >= gate_cfg.get("dup_iou", 0.25):
                    kept_idx = idx
                    break
            if kept_idx is None:
                final_features.append({"type": "Feature", "geometry": mapping(poly), "properties": props})
                if trace_idx is not None and 0 <= trace_idx < len(drop_trace):
                    drop_trace[trace_idx]["kept"] = True
                continue
            kept_props = final_features[kept_idx]["properties"]
            if osm_primary and props.get("seed_reason") == "osm_junction_primary":
                final_features.append({"type": "Feature", "geometry": mapping(poly), "properties": props})
                if trace_idx is not None and 0 <= trace_idx < len(drop_trace):
                    drop_trace[trace_idx]["kept"] = True
                continue
            if _priority(props["src_seed"]) > _priority(kept_props.get("src_seed")):
                final_features[kept_idx] = {"type": "Feature", "geometry": mapping(poly), "properties": props}
                if trace_idx is not None and 0 <= trace_idx < len(drop_trace):
                    drop_trace[trace_idx]["kept"] = True
                continue
            if _priority(props["src_seed"]) == _priority(kept_props.get("src_seed")):
                score_new = props["overlap_road"] + 0.1 * props["arm_count"] - 0.5 * props["circularity"]
                score_old = kept_props.get("overlap_road", 0.0) + 0.1 * kept_props.get("arm_count", 0) - 0.5 * kept_props.get("circularity", 0.0)
                if score_new > score_old:
                    final_features[kept_idx] = {"type": "Feature", "geometry": mapping(poly), "properties": props}
                    if trace_idx is not None and 0 <= trace_idx < len(drop_trace):
                        drop_trace[trace_idx]["kept"] = True
                    continue
            if trace_idx is not None and 0 <= trace_idx < len(drop_trace):
                best_iou = _iou(poly, shape(final_features[kept_idx]["geometry"]))
                drop_trace[trace_idx]["kept"] = False
                drop_trace[trace_idx]["dropped_reason"] = "dedup_iou"
                drop_trace[trace_idx]["dedup_best_match_iou"] = round(float(best_iou), 4)
                drop_trace[trace_idx]["best_kept_seed_id"] = str(kept_props.get("seed_id") or "")

        missing_osm_feats = []
        if osm_seed_items:
            polys = [shape(f.get("geometry")) for f in final_features if f.get("geometry")]
            union = unary_union(polys) if polys else Polygon()
            missing_dist = float(shape_cfg.get("missing_osm_junction_dist_m", 20.0))
            for item in osm_seed_items:
                pt = item.get("seed")
                if pt is None:
                    continue
                dist = pt.distance(union) if not union.is_empty else float("inf")
                if dist > missing_dist:
                    missing_osm_feats.append(
                        {
                            "type": "Feature",
                            "geometry": mapping(pt),
                            "properties": {
                                "degree": int(item.get("degree") or 0),
                                "src_seed": "osm",
                                "dist_to_final_m": round(float(dist), 3),
                            },
                        }
                    )

        _write_geojson(drive_dir / "intersections_seeds.geojson", seed_features)
        _write_geojson(drive_dir / "intersections_seeds_refined.geojson", refined_seed_features)
        _write_geojson(drive_dir / "intersections_shape_debug_local.geojson", debug_local)
        _write_geojson(drive_dir / "intersections_shape_debug_arms.geojson", debug_arms)
        _write_geojson(drive_dir / "intersections_shape_debug_arms_mask.geojson", debug_arms_mask)
        _write_geojson(drive_dir / "intersections_shape_debug_arms_lines.geojson", debug_arms_lines)
        _write_geojson(drive_dir / "intersections_shape_debug_refined.geojson", debug_refined)
        _write_geojson(drive_dir / "intersections_shape_debug_refined_pruned.geojson", debug_pruned)
        _write_geojson(drive_dir / "osm_junctions_wgs84.geojson", _to_wgs84(debug_osm_junctions, 32632))
        _write_geojson(drive_dir / "arms_lines.geojson", debug_arms_lines)
        _write_geojson(drive_dir / "local_wgs84.geojson", _to_wgs84(debug_local, 32632))
        _write_geojson(drive_dir / "arms_wgs84.geojson", _to_wgs84(debug_arms, 32632))
        _write_geojson(drive_dir / "arms_mask_wgs84.geojson", _to_wgs84(debug_arms_mask, 32632))
        _write_geojson(drive_dir / "refined_wgs84.geojson", _to_wgs84(debug_refined, 32632))
        _write_geojson(drive_dir / "refined_pruned_wgs84.geojson", _to_wgs84(debug_pruned, 32632))
        _write_geojson(drive_dir / "road_local_wgs84.geojson", _to_wgs84(debug_local, 32632))
        _write_geojson(drive_dir / "osm_arms_lines_wgs84.geojson", _to_wgs84(debug_osm_arms_lines, 32632))
        if armcount_rows:
            _write_csv(drive_dir / "osm_junction_armcount_debug.csv", armcount_rows)
        if debug_drop_trace:
            _write_jsonl(drive_dir / "debug_drop_trace.jsonl", drop_trace)
        _write_geojson(drive_dir / "missing_osm_junctions.geojson", missing_osm_feats)
        _write_geojson(drive_dir / "missing_osm_junctions_wgs84.geojson", _to_wgs84(missing_osm_feats, 32632))

        _write_geojson(drive_dir / "intersections_algo.geojson", algo_features)
        _write_geojson(drive_dir / "intersections_sat.geojson", sat_features)
        _write_geojson(drive_dir / "intersections_final.geojson", final_features)
        _write_geojson(drive_dir / "intersections_algo_wgs84.geojson", _to_wgs84(algo_features, 32632))
        _write_geojson(drive_dir / "intersections_sat_wgs84.geojson", _to_wgs84(sat_features, 32632))
        _write_geojson(drive_dir / "intersections_final_wgs84.geojson", _to_wgs84(final_features, 32632))
        _write_geojson(drive_dir / "arms_lines_wgs84.geojson", _to_wgs84(debug_arms_lines, 32632))

        sat_circ = [f["properties"]["circularity"] for f in sat_features if f.get("properties")]
        sat_overlap = [f["properties"]["overlap_road"] for f in sat_features if f.get("properties")]
        sat_circ_sorted = sorted(sat_circ)
        sat_overlap_sorted = sorted(sat_overlap)
        p50 = sat_circ_sorted[len(sat_circ_sorted) // 2] if sat_circ_sorted else None
        p75 = sat_circ_sorted[int(len(sat_circ_sorted) * 0.75)] if sat_circ_sorted else None
        o50 = sat_overlap_sorted[len(sat_overlap_sorted) // 2] if sat_overlap_sorted else None
        src_counts = {
            "traj": sum(1 for f in final_features if f.get("properties", {}).get("src_seed") == "traj"),
            "osm": sum(1 for f in final_features if f.get("properties", {}).get("src_seed") == "osm"),
            "sat": sum(1 for f in final_features if f.get("properties", {}).get("src_seed") == "sat"),
            "geom": sum(1 for f in final_features if f.get("properties", {}).get("src_seed") == "geom"),
        }
        arms_src_counts = {
            "centerlines": sum(1 for f in final_features if f.get("properties", {}).get("arms_src") == "centerlines"),
            "osm": sum(1 for f in final_features if f.get("properties", {}).get("arms_src") == "osm"),
            "skeleton": sum(1 for f in final_features if f.get("properties", {}).get("arms_src") == "skeleton"),
            "none": sum(1 for f in final_features if f.get("properties", {}).get("arms_src") in (None, "none")),
        }
        arm_heading_vals = [
            int(f.get("properties", {}).get("arm_count_heading", f.get("properties", {}).get("arm_count", 0)) or 0)
            for f in final_features
        ]
        arm_heading_sorted = sorted(arm_heading_vals)
        arm_heading_p50 = arm_heading_sorted[len(arm_heading_sorted) // 2] if arm_heading_sorted else None
        arm_heading_p75 = arm_heading_sorted[int(len(arm_heading_sorted) * 0.75)] if arm_heading_sorted else None
        rows.append(
            {
                "drive_id": drive,
                "status": "OK",
                "missing_reason": "OK" if not missing_reasons else ",".join(sorted(set(missing_reasons))),
                "final_cnt": len(final_features),
                "traj_cnt": src_counts["traj"],
                "osm_cnt": src_counts["osm"],
                "sat_cnt": src_counts["sat"],
                "geom_cnt": src_counts["geom"],
                "sat_gate_drop": gate_fail_counts["sat"],
                "other_gate_fail": gate_fail_counts["other"],
                "no_arms": no_arms_count,
                "weak_arms_cnt": weak_arms_count,
                "arm_count_heading_p50": arm_heading_p50,
                "arm_count_heading_p75": arm_heading_p75,
                "arms_src_counts": json.dumps(arms_src_counts, ensure_ascii=False),
                "seed_traj_cnt": seed_counts["traj"],
                "seed_osm_cnt": seed_counts["osm"],
                "seed_sat_cnt": seed_counts["sat"],
                "seed_geom_cnt": seed_counts["geom"],
                "refine_markings_hit_rate": round(markings_hits / max(1, len(seeds)), 4),
                "markings_hit_stop_line": markings_query_counts.get("stop_line", 0),
                "markings_hit_crosswalk": markings_query_counts.get("crosswalk", 0),
                "markings_hit_gore_marking": markings_query_counts.get("gore_marking", 0),
                "markings_hit_arrow": markings_query_counts.get("arrow", 0),
                "sat_circularity_p50": p50,
                "sat_circularity_p75": p75,
                "sat_overlap_p50": o50,
            }
        )
        if refine_cfg.get("markings_enabled") and markings_hits == 0:
            print(
                "[V2][MARKINGS] no hits within radius; counts:",
                {k: markings_query_counts.get(k, 0) for k in markings_query_counts},
            )

    for drive in expected_drives:
        if drive in seen_drives:
            continue
        rows.append(
            {
                "drive_id": drive,
                "status": "FAIL",
                "missing_reason": "missing_entry",
                "final_cnt": 0,
            }
        )

    report_csv = out_dir / f"{args.stage}_report_per_drive.csv"
    report_json = out_dir / f"{args.stage}_report_per_drive.json"
    _write_csv(report_csv, rows)
    report_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_missing_reason_summary(report_csv, expected_drives, args.stage, out_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
