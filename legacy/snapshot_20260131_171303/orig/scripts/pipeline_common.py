from __future__ import annotations

import csv
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
from shapely.geometry import box

from pipeline._io import ensure_dir, load_yaml, new_run_id, write_text

LOG = logging.getLogger("pipeline_common")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def ensure_overwrite(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def relpath(run_dir: Path, target: Path) -> str:
    try:
        return str(target.relative_to(run_dir))
    except ValueError:
        return str(target)


def _ensure_wgs84_range(gdf: gpd.GeoDataFrame) -> bool:
    if gdf.empty:
        return True
    minx, miny, maxx, maxy = gdf.total_bounds
    return -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0


def validate_output_crs(path: Path, epsg: int, gdf: Optional[gpd.GeoDataFrame], warnings: List[str]) -> None:
    name = path.name.lower()
    if "_wgs84" in name:
        if epsg != 4326:
            raise ValueError(f"CRS mismatch for {path}: expected EPSG:4326")
        if gdf is not None and not _ensure_wgs84_range(gdf):
            raise ValueError(f"WGS84 range check failed for {path}")
    elif "_utm32" in name:
        if epsg != 32632:
            raise ValueError(f"CRS mismatch for {path}: expected EPSG:32632")
    else:
        if epsg != 32632:
            raise ValueError(f"CRS mismatch for {path}: default expects EPSG:32632")
        warnings.append(f"Output missing CRS suffix (assumed utm32): {path}")


def ensure_required_columns(
    gdf: gpd.GeoDataFrame, required: Iterable[str], defaults: Optional[Dict[str, object]] = None
) -> gpd.GeoDataFrame:
    defaults = defaults or {}
    for col in required:
        if col not in gdf.columns:
            gdf[col] = defaults.get(col, None)
    return gdf


def empty_gdf(required: Iterable[str], crs: str) -> gpd.GeoDataFrame:
    cols = list(required) + ["geometry"]
    gdf = gpd.GeoDataFrame(columns=cols, geometry=[], crs=crs)
    return gdf


def write_gpkg_layer(
    path: Path,
    layer: str,
    gdf: gpd.GeoDataFrame,
    epsg: int,
    warnings: List[str],
    overwrite: bool = True,
) -> None:
    if overwrite and path.exists():
        path.unlink()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg)
    validate_output_crs(path, epsg, gdf, warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, layer=layer, driver="GPKG")


def bbox_polygon(bounds: Tuple[float, float, float, float]):
    return box(bounds[0], bounds[1], bounds[2], bounds[3])


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


__all__ = [
    "LOG",
    "bbox_polygon",
    "empty_gdf",
    "ensure_dir",
    "ensure_overwrite",
    "ensure_required_columns",
    "load_yaml",
    "new_run_id",
    "now_ts",
    "relpath",
    "setup_logging",
    "validate_output_crs",
    "write_csv",
    "write_json",
    "write_text",
    "write_gpkg_layer",
]
