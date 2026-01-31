from __future__ import annotations

from typing import Iterable, Optional

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


def _iter_polygons(geom) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        polys = []
        for g in geom.geoms:
            polys.extend(_iter_polygons(g))
        return polys
    return []


def _rebuild_polygon(poly: Polygon, corridor_pad: Optional[Polygon], max_hole_area_m2: float) -> Polygon:
    if poly.is_empty:
        return poly
    keep_holes = []
    for ring in poly.interiors:
        hole_poly = Polygon(ring)
        if hole_poly.is_empty:
            continue
        small_enough = hole_poly.area <= max_hole_area_m2
        corridor_ok = True
        if corridor_pad is not None and not corridor_pad.is_empty:
            corridor_ok = hole_poly.intersects(corridor_pad)
        if small_enough and corridor_ok:
            # Drop this interior ring to fill a small corridor-aligned hole.
            continue
        keep_holes.append(ring.coords)
    rebuilt = Polygon(poly.exterior.coords, holes=keep_holes)
    try:
        rebuilt = rebuilt.make_valid()
    except Exception:
        rebuilt = rebuilt.buffer(0)
    return rebuilt


def fill_small_holes_by_corridor(
    geom,
    corridor_geom,
    max_hole_area_m2: float,
    corridor_pad_m: float,
):
    """
    在OSM走廊附近填补小洞，避免对真实隔离带/绿化带的误填。

    参数均假设处于EPSG:32632（米制）下：
    - max_hole_area_m2: 允许填补的最大洞面积（平方米）
    - corridor_pad_m: 对走廊做buffer的容错距离（米）
    """
    if geom is None or geom.is_empty:
        return geom
    if max_hole_area_m2 <= 0:
        return geom

    corridor_pad = None
    if corridor_geom is not None and not corridor_geom.is_empty:
        try:
            corridor_pad = corridor_geom.buffer(max(0.0, float(corridor_pad_m)))
        except Exception:
            corridor_pad = corridor_geom

    polys = _iter_polygons(geom)
    if not polys:
        return geom

    rebuilt = [_rebuild_polygon(p, corridor_pad, float(max_hole_area_m2)) for p in polys]
    merged = unary_union([p for p in rebuilt if not p.is_empty])
    try:
        merged = merged.make_valid()
    except Exception:
        merged = merged.buffer(0)
    return merged


__all__ = ["fill_small_holes_by_corridor"]

