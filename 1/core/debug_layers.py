from __future__ import annotations

from pathlib import Path
from typing import Tuple


def write_demo_debug_gpkg(out_gpkg: Path, epsg: int = 3857, bbox: Tuple[float, float, float, float] = (0, 0, 100, 100)) -> Path:
    """写一个最小可视化 gpkg，用于验证：GeoPandas + 写入驱动是否正常。

    图层：
    - window_bbox：bbox 面
    - sample_points：5 个点
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box, Point
    except Exception as e:
        raise RuntimeError("缺少 geopandas/shapely，无法生成 debug_layers.gpkg。请先安装 requirements.txt") from e

    xmin, ymin, xmax, ymax = bbox
    bbox_poly = box(xmin, ymin, xmax, ymax)

    gdf_bbox = gpd.GeoDataFrame(
        [{"name": "demo_window"}],
        geometry=[bbox_poly],
        crs=f"EPSG:{epsg}",
    )
    pts = [Point(xmin + (xmax - xmin) * i / 4.0, ymin + (ymax - ymin) * i / 4.0) for i in range(5)]
    gdf_pts = gpd.GeoDataFrame(
        [{"idx": i} for i in range(len(pts))],
        geometry=pts,
        crs=f"EPSG:{epsg}",
    )

    # 强制用 pyogrio，引导 Windows 下稳定写 gpkg
    gdf_bbox.to_file(out_gpkg, layer="window_bbox", driver="GPKG", engine="pyogrio")
    gdf_pts.to_file(out_gpkg, layer="sample_points", driver="GPKG", engine="pyogrio")
    return out_gpkg
