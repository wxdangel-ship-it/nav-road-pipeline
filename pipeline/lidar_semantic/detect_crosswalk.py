from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


@dataclass
class CrosswalkResult:
    crosswalks: gpd.GeoDataFrame
    stats: Dict[str, float]


def _iter_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def _rect_dims(rect: Polygon) -> Dict[str, float]:
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return {"w": 0.0, "l": 0.0}
    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        edges.append((dx * dx + dy * dy) ** 0.5)
    edges = sorted(edges)
    return {"w": float(edges[0]), "l": float(edges[-1])}


def detect_crosswalks(
    markings: gpd.GeoDataFrame,
    corridor_geom: object,
    merge_radius_m: float,
    w_min_m: float,
    w_max_m: float,
    l_min_m: float,
    l_max_m: float,
    area_min_m2: float,
    area_max_m2: float,
    min_components: int,
) -> CrosswalkResult:
    if markings is None or markings.empty:
        empty = gpd.GeoDataFrame(columns=["class", "components", "geometry"], geometry=[], crs="EPSG:32632")
        return CrosswalkResult(crosswalks=empty, stats={"crosswalk_count": 0.0})

    merged = unary_union([g for g in markings.geometry if g is not None and not g.is_empty])
    corridor_clip = merged.intersection(corridor_geom) if corridor_geom is not None else merged
    clusters = corridor_clip.buffer(float(merge_radius_m)).buffer(-float(merge_radius_m))
    polys = _iter_polygons(clusters)

    rows = []
    for poly in polys:
        if poly.is_empty:
            continue
        rect = poly.minimum_rotated_rectangle
        dims = _rect_dims(rect)
        area = float(poly.area)
        comp_cnt = int(markings.intersects(poly).sum())
        ok = (
            (w_min_m <= dims["w"] <= w_max_m)
            and (l_min_m <= dims["l"] <= l_max_m)
            and (area_min_m2 <= area <= area_max_m2)
            and (comp_cnt >= int(min_components))
        )
        if not ok:
            continue
        rows.append(
            {
                "class": "crosswalk",
                "components": comp_cnt,
                "w_m": round(dims["w"], 3),
                "l_m": round(dims["l"], 3),
                "area_m2": round(area, 2),
                "geometry": poly,
            }
        )

    if not rows:
        gdf = gpd.GeoDataFrame(columns=["class", "components", "w_m", "l_m", "area_m2", "geometry"], geometry=[], crs="EPSG:32632")
    else:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32632")
    stats = {"crosswalk_count": float(len(gdf))}
    return CrosswalkResult(crosswalks=gdf, stats=stats)


__all__ = ["CrosswalkResult", "detect_crosswalks"]
