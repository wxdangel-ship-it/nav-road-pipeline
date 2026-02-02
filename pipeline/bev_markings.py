from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("bev_markings")


def bbox_check_utm32(bounds: Tuple[float, float, float, float]) -> Dict[str, object]:
    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy:
        return {"ok": False, "reason": "bbox_invalid_order"}
    if minx < 100000.0 or maxx > 900000.0:
        return {"ok": False, "reason": "bbox_easting_out_of_range"}
    if miny < 0.0 or maxy > 10000000.0:
        return {"ok": False, "reason": "bbox_northing_out_of_range"}
    return {"ok": True, "reason": "ok"}


def load_polygon_bounds(gpkg: Path, layer: Optional[str] = None) -> Tuple[object, Tuple[float, float, float, float]]:
    import geopandas as gpd
    from shapely.ops import unary_union

    gdf = gpd.read_file(str(gpkg), layer=layer)
    if gdf.empty:
        raise RuntimeError("gpkg_empty")
    if gdf.crs is None:
        raise RuntimeError("gpkg_crs_missing")
    if gdf.crs.to_epsg() != 32632:
        gdf = gdf.to_crs(epsg=32632)
    geom = unary_union(gdf.geometry.tolist())
    return geom, geom.bounds


def compute_grid_shape(bounds: Tuple[float, float, float, float], res_m: float, pad_m: float) -> Tuple[int, int, Tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = bounds
    minx -= pad_m
    miny -= pad_m
    maxx += pad_m
    maxy += pad_m
    width = int(math.ceil((maxx - minx) / res_m))
    height = int(math.ceil((maxy - miny) / res_m))
    if width <= 0 or height <= 0:
        raise RuntimeError("grid_invalid")
    return width, height, (minx, miny, maxx, maxy)


def read_laz_points(laz_path: Path, chunk_points: int):
    import laspy

    with laspy.open(laz_path) as reader:
        for chunk in reader.chunk_iterator(chunk_points):
            xs = np.asarray(chunk.x)
            ys = np.asarray(chunk.y)
            zs = np.asarray(chunk.z)
            intens = np.asarray(chunk.intensity)
            yield xs, ys, zs, intens


def rasterize_polygon_mask(geom, grid_bounds: Tuple[float, float, float, float], res_m: float, shape: Tuple[int, int]) -> np.ndarray:
    import rasterio.features
    import rasterio.transform

    minx, miny, maxx, maxy = grid_bounds
    transform = rasterio.transform.from_origin(minx, maxy, res_m, res_m)
    mask = rasterio.features.rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask


def write_multiband_geotiff(
    path: Path,
    bands: List[np.ndarray],
    dtypes: List[str],
    descriptions: List[str],
    crs: str,
    transform,
    nodata: List[float],
) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=bands[0].shape[0],
        width=bands[0].shape[1],
        count=len(bands),
        dtype=dtypes[0],
        crs=crs,
        transform=transform,
        nodata=nodata[0],
        compress="deflate",
    ) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band.astype(dtypes[i - 1]), i)
            dst.set_band_description(i, descriptions[i - 1])


def box_sum(image: np.ndarray, radius: int) -> np.ndarray:
    """计算以 radius 为半径的方形窗口求和（积分图实现）."""
    if radius <= 0:
        return image.astype(np.float64)
    img = image.astype(np.float64)
    h, w = img.shape
    s = img.cumsum(axis=0).cumsum(axis=1)

    def _at(y: int, x: int) -> float:
        if y < 0 or x < 0:
            return 0.0
        y = min(y, h - 1)
        x = min(x, w - 1)
        return float(s[y, x])

    out = np.zeros_like(img, dtype=np.float64)
    for y in range(h):
        y0 = y - radius - 1
        y1 = y + radius
        for x in range(w):
            x0 = x - radius - 1
            x1 = x + radius
            out[y, x] = _at(y1, x1) - _at(y0, x1) - _at(y1, x0) + _at(y0, x0)
    return out
