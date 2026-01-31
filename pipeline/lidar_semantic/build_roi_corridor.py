from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
from shapely.geometry import LineString, MultiPoint

from pipeline.datasets.kitti360_io import _find_oxts_dir, load_kitti360_pose


@dataclass
class RoiResult:
    roi_geom: object
    roi_source: str
    errors: List[str]


def build_roi_corridor(
    data_root: Path,
    drive_id: str,
    buffer_m: float,
    stride: int,
    target_epsg: int,
) -> RoiResult:
    errors: List[str] = []
    try:
        oxts_dir = _find_oxts_dir(data_root, drive_id)
        frames = sorted(oxts_dir.glob("*.txt"))
    except Exception as exc:  # pragma: no cover - data dependent
        errors.append(f"oxts_missing:{exc}")
        frames = []

    if stride > 1 and frames:
        frames = frames[::stride]

    points: List[Tuple[float, float]] = []
    for frame in frames:
        frame_id = frame.stem
        try:
            x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
        except Exception as exc:  # pragma: no cover - data dependent
            errors.append(f"pose_failed:{frame_id}:{exc}")
            continue
        points.append((x, y))

    if not points:
        # 退化：给一个很小的占位ROI，避免流程中断。
        roi_geom = MultiPoint([(0.0, 0.0)]).buffer(max(1.0, buffer_m))
        return RoiResult(roi_geom=roi_geom, roi_source="degraded_empty", errors=errors)

    if len(points) == 1:
        traj = MultiPoint(points)
    else:
        traj = LineString(points)
    roi_geom = traj.buffer(buffer_m)
    return RoiResult(roi_geom=roi_geom, roi_source="trajectory", errors=errors)


def roi_to_gdf(roi_geom: object, drive_id: str, target_epsg: int) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame([{"drive_id": drive_id, "geometry": roi_geom}], geometry="geometry", crs=f"EPSG:{target_epsg}")


__all__ = ["RoiResult", "build_roi_corridor", "roi_to_gdf"]

