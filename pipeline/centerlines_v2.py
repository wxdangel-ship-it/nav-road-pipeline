from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon, MultiPolygon, mapping
from shapely.ops import split

try:
    from shapely.ops import medial_axis as _medial_axis
except Exception:
    _medial_axis = None


def _longest_line(geom) -> Optional[LineString]:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        lines = list(geom.geoms)
        if not lines:
            return None
        return max(lines, key=lambda g: g.length)
    if geom.geom_type == "GeometryCollection":
        lines = []
        for g in geom.geoms:
            line = _longest_line(g)
            if line is not None:
                lines.append(line)
        if not lines:
            return None
        return max(lines, key=lambda g: g.length)
    return None


def _line_direction(line: LineString) -> Tuple[float, float]:
    coords = list(line.coords)
    if len(coords) < 2:
        return 1.0, 0.0
    x0, y0 = coords[0]
    x1, y1 = coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return 1.0, 0.0
    return dx / length, dy / length


def _bbox_axis(poly: Polygon | MultiPolygon) -> Tuple[float, float]:
    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny
    if dx >= dy:
        return 1.0, 0.0
    return 0.0, 1.0


def _split_line_through_point(point: Point, direction: Tuple[float, float], bounds: tuple) -> LineString:
    minx, miny, maxx, maxy = bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    if diag == 0:
        diag = 1.0
    dx, dy = direction
    if dx == 0 and dy == 0:
        dx, dy = 1.0, 0.0
    scale = diag * 2.0
    x0 = point.x - dx * scale
    y0 = point.y - dy * scale
    x1 = point.x + dx * scale
    y1 = point.y + dy * scale
    return LineString([(x0, y0), (x1, y1)])


def _split_by_line(poly: Polygon, line: LineString) -> list[Polygon]:
    try:
        pieces = split(poly, line)
    except Exception:
        return []
    polys = [p for p in pieces.geoms if p.geom_type == "Polygon"]
    return polys


def _polygon_axis_line(poly: Polygon) -> Optional[LineString]:
    if poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny
    if dx <= 0 or dy <= 0:
        return None
    if dx >= dy:
        return LineString([(minx, (miny + maxy) * 0.5), (maxx, (miny + maxy) * 0.5)])
    return LineString([((minx + maxx) * 0.5, miny), ((minx + maxx) * 0.5, maxy)])


def _polygon_centerline(poly: Polygon, simplify_m: float = 0.5) -> Optional[LineString]:
    if poly.is_empty:
        return None
    if _medial_axis is None:
        return _polygon_axis_line(poly)
    try:
        skel = _medial_axis(poly)
    except Exception:
        return _polygon_axis_line(poly)
    line = _longest_line(skel)
    if line is None or line.is_empty:
        return _polygon_axis_line(poly)
    if simplify_m and simplify_m > 0:
        line = line.simplify(simplify_m, preserve_topology=True)
    return line


def _assign_lr(lines: list[LineString], base_line: LineString) -> list[tuple[str, LineString]]:
    if len(lines) != 2:
        return [("L", lines[0])] if lines else []
    dir_x, dir_y = _line_direction(base_line)
    base_pt = base_line.centroid
    scored = []
    for ln in lines:
        vec_x = ln.centroid.x - base_pt.x
        vec_y = ln.centroid.y - base_pt.y
        cross = dir_x * vec_y - dir_y * vec_x
        side = "L" if cross > 0 else "R"
        scored.append((side, ln, abs(cross)))
    scored.sort(key=lambda t: t[2], reverse=True)
    left = next((ln for side, ln, _ in scored if side == "L"), scored[0][1])
    right = next((ln for side, ln, _ in scored if side == "R" and ln != left), scored[-1][1])
    return [("L", left), ("R", right)]


def _dual_sep_m(line_a: LineString, line_b: LineString, step_m: float = 5.0) -> Optional[float]:
    if line_a is None or line_b is None or line_a.is_empty or line_b.is_empty:
        return None
    length = max(0.0, float(line_a.length))
    if length <= 0:
        return None
    steps = max(2, int(length / max(1.0, step_m)))
    dists = []
    for i in range(steps + 1):
        pt = line_a.interpolate((i / steps) * length)
        dists.append(pt.distance(line_b))
    if not dists:
        return None
    return float(sum(dists) / len(dists))


def _divider_from_geometry(
    road_poly: Polygon | MultiPolygon,
    base_line: LineString,
) -> tuple[bool, str, list[Polygon], Optional[LineString]]:
    if road_poly.geom_type == "MultiPolygon":
        polys = sorted(list(road_poly.geoms), key=lambda g: g.area, reverse=True)
        if len(polys) >= 2:
            return True, "geom_multipoly", polys[:2], None
    if road_poly.geom_type == "Polygon" and road_poly.interiors:
        holes = [Polygon(ring) for ring in road_poly.interiors if ring and len(ring.coords) >= 3]
        if holes:
            hole = max(holes, key=lambda g: g.area)
            direction = _line_direction(base_line)
            if direction == (0.0, 0.0):
                direction = _bbox_axis(road_poly)
            split_line = _split_line_through_point(hole.centroid, direction, road_poly.bounds)
            polys = _split_by_line(road_poly, split_line)
            polys = sorted(polys, key=lambda g: g.area, reverse=True)
            if len(polys) >= 2:
                return True, "geom_hole", polys[:2], split_line
    return False, "none", [], None


def _split_by_divider(
    road_poly: Polygon | MultiPolygon,
    divider_lines: list[LineString],
    buffer_m: float,
) -> list[Polygon]:
    if not divider_lines:
        return []
    divider_union = divider_lines[0]
    if len(divider_lines) > 1:
        from shapely.ops import unary_union

        divider_union = unary_union(divider_lines)
    try:
        carved = road_poly.difference(divider_union.buffer(buffer_m))
    except Exception:
        return []
    if carved.is_empty:
        return []
    if carved.geom_type == "Polygon":
        return [carved]
    if carved.geom_type == "MultiPolygon":
        return list(carved.geoms)
    return []


def build_centerlines_v2(
    traj_line: LineString,
    road_poly: Polygon | MultiPolygon,
    center_cfg: dict,
    center_offset_default: float,
    divider_lines: Optional[list[LineString]] = None,
    divider_src_hint: Optional[str] = None,
) -> dict:
    base_line = traj_line.intersection(road_poly) if road_poly is not None else traj_line
    if base_line.is_empty:
        base_line = traj_line
    if base_line.geom_type != "LineString":
        base_line = _longest_line(base_line) or traj_line

    width_median = float(center_cfg.get("dual_width_threshold_m", 0.0))
    width_score = 0.0
    if width_median > 0:
        width_score = min(1.0, max(0.0, float(center_cfg.get("width_median_m", 0.0)) / width_median))

    divider_sources = center_cfg.get("divider_sources", ["geom"])
    if isinstance(divider_sources, str):
        divider_sources = [divider_sources]
    use_geom = "geom" in {str(s).lower() for s in divider_sources}
    divider_found, divider_src, carriageways, divider_line = False, "none", [], None
    if divider_lines:
        buffer_m = float(center_cfg.get("divider_buffer_m", 1.5))
        split_polys = _split_by_divider(road_poly, divider_lines, buffer_m)
        split_polys = sorted(split_polys, key=lambda g: g.area, reverse=True)
        if len(split_polys) >= 2:
            divider_found = True
            divider_src = divider_src_hint or "seg_divider"
            carriageways = split_polys[:2]
            if len(divider_lines) == 1:
                divider_line = divider_lines[0]
            else:
                try:
                    from shapely.ops import unary_union

                    divider_line = unary_union(divider_lines)
                except Exception:
                    divider_line = divider_lines[0]
    if not divider_found and use_geom:
        divider_found, divider_src, carriageways, divider_line = _divider_from_geometry(road_poly, base_line)
    dual_conf = width_score if divider_found else 0.0
    dual_allowed = divider_found and dual_conf >= float(center_cfg.get("dual_conf_threshold", 0.0))

    dual_lines: list[LineString] = []
    if dual_allowed and len(carriageways) >= 2:
        for poly in carriageways[:2]:
            line = _polygon_centerline(poly, simplify_m=float(center_cfg.get("simplify_m", 0.5)))
            if line is not None:
                dual_lines.append(line)
        if len(dual_lines) != 2:
            dual_lines = []

    outputs: dict[str, list[dict]] = {"single": [], "dual": [], "both": [], "auto": []}
    outputs_lines: dict[str, list[LineString]] = {"single": [], "dual": [], "both": [], "auto": []}

    base_feature = {
        "type": "Feature",
        "geometry": mapping(base_line),
        "properties": {
            "line_type": "single",
            "side": None,
            "mode": "single",
            "offset_m": 0.0,
            "dual_conf": round(float(dual_conf), 4),
            "divider_src": divider_src,
            "divider_found": 1 if divider_found else 0,
            "dual_method": "split_skeleton",
            "qc_sep_m": None,
        },
    }

    outputs["single"].append(base_feature)
    outputs_lines["single"].append(base_line)

    dual_features: list[dict] = []
    dual_pairs: list[tuple[str, LineString]] = []
    if dual_lines:
        dual_pairs = _assign_lr(dual_lines, base_line)
        sep_m = _dual_sep_m(dual_lines[0], dual_lines[1], step_m=float(center_cfg.get("step_m", 5.0)))
        for side, line in dual_pairs:
            dual_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(line),
                    "properties": {
                        "line_type": "dual",
                        "side": side,
                        "mode": "dual",
                        "offset_m": 0.0,
                        "dual_conf": round(float(dual_conf), 4),
                        "divider_src": divider_src,
                        "divider_found": 1 if divider_found else 0,
                        "dual_method": "split_skeleton",
                        "qc_sep_m": None if sep_m is None else round(float(sep_m), 3),
                    },
                }
            )

    if dual_features:
        outputs["dual"].extend(dual_features)
        outputs_lines["dual"].extend([line for _, line in dual_pairs])

    mode = str(center_cfg.get("mode", "auto")).lower()
    dual_fallback = False
    if mode == "dual":
        if dual_features:
            outputs["auto"] = dual_features
            outputs_lines["auto"] = outputs_lines["dual"]
        elif center_cfg.get("dual_fallback_single", True):
            outputs["auto"] = outputs["single"]
            outputs_lines["auto"] = outputs_lines["single"]
            dual_fallback = True
    elif mode == "both":
        outputs["both"] = outputs["single"] + (dual_features if dual_features else [])
        outputs_lines["both"] = outputs_lines["single"] + (outputs_lines["dual"] if dual_features else [])
        outputs["auto"] = outputs["both"]
        outputs_lines["auto"] = outputs_lines["both"]
    elif mode == "auto":
        if dual_features:
            outputs["auto"] = dual_features
            outputs_lines["auto"] = outputs_lines["dual"]
        else:
            outputs["auto"] = outputs["single"]
            outputs_lines["auto"] = outputs_lines["single"]
    else:
        outputs["auto"] = outputs["single"]
        outputs_lines["auto"] = outputs_lines["single"]

    if dual_fallback and outputs.get("auto"):
        for feat in outputs["auto"]:
            props = feat.get("properties") or {}
            props["dual_fallback"] = 1
            feat["properties"] = props

    return {
        "outputs": outputs,
        "outputs_lines": outputs_lines,
        "active_features": outputs["auto"],
        "active_lines": outputs_lines["auto"],
        "dual_triggered": bool(dual_features),
        "dual_conf": float(dual_conf),
        "dual_sep_m": _dual_sep_m(dual_lines[0], dual_lines[1]) if len(dual_lines) == 2 else None,
        "divider_found": divider_found,
        "divider_src": divider_src,
        "divider_line": divider_line,
        "carriageways": carriageways,
    }
