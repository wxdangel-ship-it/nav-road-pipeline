from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

try:
    from shapely import make_valid as _shapely_make_valid
except Exception:
    try:
        from shapely.validation import make_valid as _shapely_make_valid
    except Exception:
        _shapely_make_valid = None


def _make_valid(geom):
    if geom is None or geom.is_empty:
        return geom
    if _shapely_make_valid is None:
        return geom.buffer(0)
    try:
        return _shapely_make_valid(geom)
    except Exception:
        return geom.buffer(0)


def _largest_polygon(geom) -> Optional[Polygon]:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None


def circularity(poly: Polygon) -> float:
    if poly is None or poly.is_empty:
        return 0.0
    area = float(poly.area)
    perim = float(poly.length)
    if area <= 0 or perim <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))


def aspect_ratio(poly: Polygon) -> float:
    if poly is None or poly.is_empty:
        return 0.0
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0
    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        edges.append(math.hypot(x2 - x1, y2 - y1))
    edges = [e for e in edges if e > 0]
    if not edges:
        return 0.0
    edges = sorted(edges)
    return float(edges[-1] / edges[0])


def overlap_with_road(poly: Polygon, road_poly: Polygon) -> float:
    if poly is None or poly.is_empty or road_poly is None or road_poly.is_empty:
        return 0.0
    area = float(poly.area)
    if area <= 0:
        return 0.0
    return float(poly.intersection(road_poly).area) / area


def arm_count(poly: Polygon, centerlines: Iterable[LineString], buffer_m: float) -> int:
    if poly is None or poly.is_empty:
        return 0
    if not centerlines:
        return 0
    touch = poly.buffer(buffer_m)
    count = 0
    for line in centerlines:
        if line is None or line.is_empty:
            continue
        if line.intersects(touch):
            count += 1
    return count


def arm_count_branches(seed_pt: Point, arms_geom, radius_m: float) -> int:
    if seed_pt is None or arms_geom is None or arms_geom.is_empty:
        return 0
    radius = max(float(radius_m), 0.1)
    local = arms_geom.intersection(seed_pt.buffer(radius))
    if local.is_empty:
        return 0
    if local.geom_type == "Polygon":
        return 1
    if local.geom_type == "MultiPolygon":
        return len(list(local.geoms))
    if local.geom_type == "GeometryCollection":
        return sum(1 for g in local.geoms if g.geom_type in ("Polygon", "MultiPolygon"))
    return 1


def calc_arm_count_by_heading(
    arms_lines_or_corridor,
    seed_pt: Point,
    r_eval_m: float,
    angle_bin_deg: float,
    min_len_m: float,
) -> int:
    if seed_pt is None or arms_lines_or_corridor is None:
        return 0
    if hasattr(arms_lines_or_corridor, "is_empty") and arms_lines_or_corridor.is_empty:
        return 0
    radius = max(float(r_eval_m), 0.1)
    clip = seed_pt.buffer(radius)

    def _collect_lines(geom) -> List[LineString]:
        if geom is None:
            return []
        if isinstance(geom, list):
            return [g for g in geom if isinstance(g, LineString) and not g.is_empty]
        if geom.geom_type == "LineString":
            return [geom]
        if geom.geom_type == "MultiLineString":
            return [g for g in geom.geoms if not g.is_empty]
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            boundary = geom.boundary
            return _collect_lines(boundary)
        if geom.geom_type == "GeometryCollection":
            lines: List[LineString] = []
            for g in geom.geoms:
                lines.extend(_collect_lines(g))
            return lines
        return []

    def _segment_angles(lines: List[LineString]) -> List[float]:
        angles: List[float] = []
        for line in lines:
            if line is None or line.is_empty:
                continue
            local = line.intersection(clip)
            if local.is_empty:
                continue
            segments = _collect_lines(local)
            for seg in segments:
                if seg.length < min_len_m:
                    continue
                coords = list(seg.coords)
                if len(coords) < 2:
                    continue
                x1, y1 = coords[0]
                x2, y2 = coords[-1]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                angle = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
                angles.append(angle)
        return angles

    lines = _collect_lines(arms_lines_or_corridor)
    angles = _segment_angles(lines)
    if not angles:
        return 0
    bin_deg = max(float(angle_bin_deg), 1.0)
    angles.sort()
    clusters = [[angles[0]]]
    for ang in angles[1:]:
        if abs(ang - clusters[-1][-1]) <= bin_deg:
            clusters[-1].append(ang)
        else:
            clusters.append([ang])
    if len(clusters) > 1:
        wrap_gap = (clusters[0][0] + 180.0) - clusters[-1][-1]
        if wrap_gap <= bin_deg:
            clusters[0] = clusters[-1] + clusters[0]
            clusters = clusters[:-1]
    return len(clusters)


def calc_arm_count_by_approach(
    arms_lines_or_corridor,
    seed_pt: Point,
    r_eval_m: float,
    angle_bin_deg: float,
    min_len_m: float,
) -> int:
    if seed_pt is None or arms_lines_or_corridor is None:
        return 0
    if hasattr(arms_lines_or_corridor, "is_empty") and arms_lines_or_corridor.is_empty:
        return 0
    radius = max(float(r_eval_m), 0.1)
    clip = seed_pt.buffer(radius)

    def _collect_lines(geom) -> List[LineString]:
        if geom is None:
            return []
        if isinstance(geom, list):
            return [g for g in geom if isinstance(g, LineString) and not g.is_empty]
        if geom.geom_type == "LineString":
            return [geom]
        if geom.geom_type == "MultiLineString":
            return [g for g in geom.geoms if not g.is_empty]
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            boundary = geom.boundary
            return _collect_lines(boundary)
        if geom.geom_type == "GeometryCollection":
            lines: List[LineString] = []
            for g in geom.geoms:
                lines.extend(_collect_lines(g))
            return lines
        return []

    def _segment_angles(lines: List[LineString]) -> List[float]:
        angles: List[float] = []
        for line in lines:
            if line is None or line.is_empty:
                continue
            local = line.intersection(clip)
            if local.is_empty:
                continue
            segments = _collect_lines(local)
            for seg in segments:
                if seg.length < min_len_m:
                    continue
                coords = list(seg.coords)
                if len(coords) < 2:
                    continue
                p0 = Point(coords[0])
                p1 = Point(coords[-1])
                if seed_pt.distance(p0) <= seed_pt.distance(p1):
                    dx = p1.x - p0.x
                    dy = p1.y - p0.y
                else:
                    dx = p0.x - p1.x
                    dy = p0.y - p1.y
                if dx == 0 and dy == 0:
                    continue
                angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
                angles.append(angle)
        return angles

    lines = _collect_lines(arms_lines_or_corridor)
    angles = _segment_angles(lines)
    if not angles:
        return 0
    bin_deg = max(float(angle_bin_deg), 1.0)
    angles.sort()
    clusters = [[angles[0]]]
    for ang in angles[1:]:
        if abs(ang - clusters[-1][-1]) <= bin_deg:
            clusters[-1].append(ang)
        else:
            clusters.append([ang])
    if len(clusters) > 1:
        wrap_gap = (clusters[0][0] + 360.0) - clusters[-1][-1]
        if wrap_gap <= bin_deg:
            clusters[0] = clusters[-1] + clusters[0]
            clusters = clusters[:-1]
    return len(clusters)


def refine_intersection_polygon(
    seed_pt: Point,
    poly_candidate: Polygon,
    road_polygon: Polygon,
    centerlines: Iterable[LineString],
    cfg: dict,
) -> Tuple[Optional[Polygon], dict]:
    radius_m = float(cfg.get("radius_m", 15.0))
    road_buffer_m = float(cfg.get("road_buffer_m", 1.0))
    arm_length_m = float(cfg.get("arm_length_m", 25.0))
    arm_buffer_m = float(cfg.get("arm_buffer_m", 6.0))
    simplify_m = float(cfg.get("simplify_m", 0.5))
    min_area_m2 = float(cfg.get("min_area_m2", 30.0))
    min_part_area_m2 = float(cfg.get("min_part_area_m2", min_area_m2))
    min_hole_area_m2 = float(cfg.get("min_hole_area_m2", min_area_m2))

    seed_buffer = seed_pt.buffer(radius_m)
    local = road_polygon.buffer(road_buffer_m).intersection(seed_buffer)
    local = _make_valid(local)

    arm_buffer = seed_pt.buffer(arm_length_m)
    arm_segments = []
    for line in centerlines:
        if line is None or line.is_empty:
            continue
        seg = line.intersection(arm_buffer)
        if seg.is_empty:
            continue
        arm_segments.append(seg)
    arms_lines = unary_union([s for s in arm_segments]) if arm_segments else None
    arms = arms_lines
    if arms is not None and not arms.is_empty:
        arms = arms.buffer(arm_buffer_m, cap_style=2, join_style=2)
        refined = local.intersection(arms)
    else:
        refined = local

    refined = _make_valid(refined)
    if refined is not None and not refined.is_empty:
        parts = []
        if refined.geom_type == "Polygon":
            parts = [refined]
        elif refined.geom_type == "MultiPolygon":
            parts = list(refined.geoms)
        if parts:
            parts = [p for p in parts if p.area >= min_part_area_m2]
            refined = unary_union(parts) if parts else refined
    if refined is not None and not refined.is_empty and min_hole_area_m2 > 0:
        cleaned = []
        polys = []
        if refined.geom_type == "Polygon":
            polys = [refined]
        elif refined.geom_type == "MultiPolygon":
            polys = list(refined.geoms)
        for poly in polys:
            keep = []
            for ring in poly.interiors:
                hole = Polygon(ring)
                if hole.area >= min_hole_area_m2:
                    keep.append(ring)
            cleaned.append(Polygon(poly.exterior, keep))
        refined = unary_union(cleaned) if cleaned else refined
    refined = refined.simplify(simplify_m, preserve_topology=True) if simplify_m > 0 else refined
    refined_poly = _largest_polygon(refined)

    reason = "refined"
    if refined_poly is None or refined_poly.is_empty or refined_poly.area < min_area_m2:
        refined_poly = _largest_polygon(local)
        reason = "fallback_local"

    refined_poly = _make_valid(refined_poly) if refined_poly is not None else None

    meta = {
        "local": local,
        "arms": arms,
        "arms_lines": arms_lines,
        "reason": reason,
    }
    return refined_poly, meta
