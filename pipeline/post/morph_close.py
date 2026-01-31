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


def _make_valid(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        return geom.make_valid()
    except Exception:
        return geom.buffer(0)


def close_geom_by_buffer(geom, radius_m: float, join_style: str = "round"):
    """
    使用buffer等价实现closing：buffer(+r)再buffer(-r)。

    所有距离单位假设为米制投影（例如EPSG:32632）。
    """
    if geom is None or geom.is_empty:
        return geom
    r = float(radius_m)
    if r <= 0.0:
        return _make_valid(geom)

    # shapely join_style: 1=round, 2=mitre, 3=bevel
    join_style_map = {"round": 1, "mitre": 2, "bevel": 3}
    js = join_style_map.get(join_style, 1)
    closed = geom.buffer(r, join_style=js).buffer(-r, join_style=js)
    return _make_valid(closed)


def close_candidates_in_corridor(
    cand_geom,
    corridor_geom,
    radius_m: float,
    corridor_pad_m: float,
):
    """
    只在corridor的容错范围内做closing，降低外溢风险。

    处理流程：
    1) corridor_pad = corridor.buffer(pad)
    2) cand_clip = cand ∩ corridor_pad
    3) closing(cand_clip, r)
    4) 再次裁回 corridor_pad
    """
    if cand_geom is None or cand_geom.is_empty:
        return cand_geom
    if corridor_geom is None or corridor_geom.is_empty:
        return close_geom_by_buffer(cand_geom, radius_m)

    corridor_pad = _make_valid(corridor_geom.buffer(max(0.0, float(corridor_pad_m))))
    cand_clip = _make_valid(cand_geom.intersection(corridor_pad))
    cand_closed = close_geom_by_buffer(cand_clip, radius_m)
    cand_closed = _make_valid(cand_closed.intersection(corridor_pad))
    # 归并为稳定的多面结构，避免GeometryCollection散落。
    polys = [p for p in _iter_polygons(cand_closed) if not p.is_empty]
    if not polys:
        return cand_closed
    return _make_valid(unary_union(polys))


__all__ = ["close_geom_by_buffer", "close_candidates_in_corridor"]

