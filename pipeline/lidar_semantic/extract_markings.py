from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.geometry import shape

from pipeline.post.morph_close import close_candidates_in_corridor

from .build_rasters import RasterBundle


@dataclass
class MarkingResult:
    marking_score: np.ndarray
    marking_mask: np.ndarray
    markings_points_mask: np.ndarray
    markings_polygons: gpd.GeoDataFrame
    stats: Dict[str, float]
    intensity_threshold: float


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


def extract_markings(
    bundle: RasterBundle,
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    road_points_mask: np.ndarray,
    road_mask: np.ndarray,
    corridor_geom: object,
    intensity_pctl: float,
    height_max_m: float,
    close_radius_m: float,
) -> MarkingResult:
    road_intensity = intensity[road_points_mask]
    if road_intensity.size == 0:
        thr = float(np.percentile(intensity, 99.0)) if intensity.size else 0.0
    else:
        thr = float(np.percentile(road_intensity, float(intensity_pctl)))

    valid = bundle.point_valid
    ix = bundle.point_ix
    iy = bundle.point_iy
    height_ref = bundle.point_height_p10
    near_ground = valid & np.isfinite(height_ref) & (points_xyz[:, 2] <= (height_ref + float(height_max_m)))
    markings_points_mask = road_points_mask & near_ground & (intensity >= thr)

    # Per-cell marking score on road cells.
    score = np.zeros_like(bundle.density, dtype=np.float32)
    if np.any(road_points_mask):
        road_valid = road_points_mask & valid
        lin = iy[road_valid].astype(np.int64) * int(bundle.width) + ix[road_valid].astype(np.int64)
        order = np.argsort(lin, kind="mergesort")
        lin_sorted = lin[order]
        uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
        mark_valid = markings_points_mask & valid
        mark_lin = iy[mark_valid].astype(np.int64) * int(bundle.width) + ix[mark_valid].astype(np.int64)
        mark_counts = {}
        if mark_lin.size:
            mu, mc = np.unique(mark_lin, return_counts=True)
            mark_counts = {int(k): int(v) for k, v in zip(mu, mc)}
        ix_sorted = ix[road_valid][order]
        iy_sorted = iy[road_valid][order]
        for u, s, c in zip(uniq, start, counts):
            cx = int(ix_sorted[s])
            cy = int(iy_sorted[s])
            m = float(mark_counts.get(int(u), 0))
            score[cy, cx] = float(m) / max(float(c), 1.0)

    # Threshold on score within road cells.
    score_thresh = 0.05
    marking_mask = (score >= score_thresh) & (road_mask.astype(bool))
    min_area_m2 = max(0.5, bundle.res_m * bundle.res_m * 2.0)
    polys = _mask_to_polygons(marking_mask, bundle, min_area_m2=min_area_m2)
    if not polys.empty and close_radius_m > 0:
        union = polys.unary_union
        closed = close_candidates_in_corridor(union, corridor_geom, float(close_radius_m), corridor_pad_m=0.0)
        if closed is not None and not closed.is_empty:
            marking_mask = features.rasterize(
                [(closed, 1)],
                out_shape=(bundle.height, bundle.width),
                transform=bundle.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            polys = _mask_to_polygons(marking_mask, bundle, min_area_m2=min_area_m2)
        else:
            marking_mask = np.zeros_like(marking_mask, dtype=bool)
            polys = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")

    marking_area = float(polys.geometry.area.sum()) if not polys.empty else 0.0
    road_area = float(
        np.sum(road_mask.astype(bool)) * (bundle.res_m * bundle.res_m)
    )
    stats = {
        "marking_area_m2": marking_area,
        "marking_cover_on_road": marking_area / max(road_area, 1e-6),
        "marking_points_ratio": float(markings_points_mask.sum()) / max(float(road_points_mask.sum()), 1.0),
        "score_thresh": score_thresh,
    }
    return MarkingResult(
        marking_score=score,
        marking_mask=marking_mask.astype("uint8"),
        markings_points_mask=markings_points_mask,
        markings_polygons=polys,
        stats=stats,
        intensity_threshold=thr,
    )


__all__ = ["MarkingResult", "extract_markings"]
