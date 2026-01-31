from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.geometry import shape

from pipeline.post.morph_close import close_candidates_in_corridor

from .build_rasters import RasterBundle


@dataclass
class RoadResult:
    road_mask: np.ndarray
    road_points_mask: np.ndarray
    road_polygons: gpd.GeoDataFrame
    stats: Dict[str, float]


def _mask_to_polygons(mask: np.ndarray, bundle: RasterBundle, min_area_m2: float) -> gpd.GeoDataFrame:
    geoms = []
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=bundle.transform):
        if int(val) != 1:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        if float(poly.area) < min_area_m2:
            continue
        geoms.append(poly)
    if not geoms:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    return gpd.GeoDataFrame([{"geometry": g} for g in geoms], geometry="geometry", crs="EPSG:32632")


def classify_road(
    bundle: RasterBundle,
    points_xyz: np.ndarray,
    corridor_geom: object,
    ground_band_dz_m: float,
    min_density: float,
    roughness_max_m: float,
    close_radius_m: float,
) -> RoadResult:
    height_ok = np.isfinite(bundle.height_p10)
    density_ok = bundle.density_all >= float(min_density)
    rough_ok = np.isnan(bundle.roughness) | (bundle.roughness <= float(roughness_max_m))
    road_mask = height_ok & density_ok & rough_ok

    # Vector closing within corridor to reduce small holes/gaps.
    min_area_m2 = max(1.0, bundle.res_m * bundle.res_m * 4.0)
    road_polys = _mask_to_polygons(road_mask, bundle, min_area_m2=min_area_m2)
    if not road_polys.empty and close_radius_m > 0:
        union = road_polys.unary_union
        closed = close_candidates_in_corridor(union, corridor_geom, float(close_radius_m), corridor_pad_m=0.0)
        if closed is not None and not closed.is_empty:
            closed_mask = features.rasterize(
                [(closed, 1)],
                out_shape=(bundle.height, bundle.width),
                transform=bundle.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            road_polys = _mask_to_polygons(closed_mask, bundle, min_area_m2=min_area_m2)
        else:
            road_polys = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")

    ix = bundle.point_ix
    iy = bundle.point_iy
    valid = bundle.point_valid
    road_cell = np.zeros_like(valid, dtype=bool)
    if points_xyz.size and np.any(valid):
        road_cell[valid] = road_mask[iy[valid], ix[valid]]
    ground_band = valid & np.isfinite(bundle.point_height_p10) & (
        points_xyz[:, 2] <= (bundle.point_height_p10 + float(ground_band_dz_m))
    )
    road_points_mask = road_cell & ground_band

    road_area = float(road_polys.geometry.area.sum()) if not road_polys.empty else 0.0
    corridor_area = float(corridor_geom.area) if corridor_geom is not None else 1.0
    stats = {
        "road_area_m2": road_area,
        "corridor_area_m2": corridor_area,
        "road_cover": road_area / max(corridor_area, 1e-6),
    }
    return RoadResult(road_mask=road_mask.astype("uint8"), road_points_mask=road_points_mask, road_polygons=road_polys, stats=stats)


__all__ = ["RoadResult", "classify_road"]
