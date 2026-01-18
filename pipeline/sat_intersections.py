from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import Point, mapping, shape
from shapely.ops import transform as geom_transform

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is required for SAT
    np = None

try:
    import rasterio
    from rasterio.windows import Window
except Exception:  # pragma: no cover - optional dependency
    rasterio = None
    Window = None

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - optional dependency
    Transformer = None

RasterIndex = List[Dict[str, float]]


def _safe_write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _scan_tiles(tiles_dir: Path) -> RasterIndex:
    if rasterio is None:
        raise RuntimeError("missing_rasterio")
    items: RasterIndex = []
    for ext in ("*.tif", "*.tiff", "*.jp2", "*.jpg", "*.jpeg"):
        for path in tiles_dir.rglob(ext):
            try:
                with rasterio.open(path) as ds:
                    bounds = ds.bounds
                items.append(
                    {
                        "path": str(path),
                        "minx": float(bounds.left),
                        "miny": float(bounds.bottom),
                        "maxx": float(bounds.right),
                        "maxy": float(bounds.top),
                    }
                )
            except Exception:
                continue
    return items


def _load_index_cache(cache_path: Path) -> Optional[RasterIndex]:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _save_index_cache(cache_path: Path, items: RasterIndex) -> None:
    cache_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_path(dop20_root: Path) -> Path:
    override = os.environ.get("DOP20_INDEX_CACHE", "").strip()
    if override:
        return Path(override)
    return dop20_root / "dop20_tiles_index.json"


def load_tile_index(dop20_root: Path, tiles_dir: Path) -> RasterIndex:
    cache_path = _cache_path(dop20_root)
    cached = _load_index_cache(cache_path)
    if cached is not None:
        return cached
    items = _scan_tiles(tiles_dir)
    try:
        _save_index_cache(cache_path, items)
    except PermissionError:
        fallback = Path("cache") / "dop20_tiles_index.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        _save_index_cache(fallback, items)
    return items


def _find_tile_for_point(items: RasterIndex, x: float, y: float) -> Optional[str]:
    for item in items:
        if item["minx"] <= x <= item["maxx"] and item["miny"] <= y <= item["maxy"]:
            return str(item["path"])
    return None


def _read_patch(tile_path: str, x: float, y: float, patch_m: float) -> Optional[np.ndarray]:
    if rasterio is None or Window is None:
        return None
    with rasterio.open(tile_path) as ds:
        col, row = ds.index(x, y)
        res_x = abs(ds.transform.a)
        res_y = abs(ds.transform.e)
        half_w = int((patch_m / res_x) / 2.0)
        half_h = int((patch_m / res_y) / 2.0)
        window = Window(col - half_w, row - half_h, half_w * 2, half_h * 2)
        patch = ds.read(window=window)
        if patch is None or patch.size == 0:
            return None
        patch = np.moveaxis(patch, 0, -1)
        return patch


def _road_mask_simple(img: np.ndarray) -> Tuple[np.ndarray, float]:
    if img.ndim != 3:
        return np.zeros(img.shape[:2], dtype=bool), 0.0
    img = img.astype("float32")
    if img.max() > 1.0:
        img = img / 255.0
    if cv2 is not None:
        hsv = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        road = (s.astype("float32") / 255.0 < 0.25) & (v.astype("float32") / 255.0 > 0.25)
        kernel = np.ones((5, 5), dtype=np.uint8)
        road = cv2.morphologyEx(road.astype("uint8"), cv2.MORPH_CLOSE, kernel) > 0
    else:
        sat = img.max(axis=2) - img.min(axis=2)
        val = img.mean(axis=2)
        road = (sat < 0.2) & (val > 0.25)
    ratio = float(road.mean()) if road.size > 0 else 0.0
    return road, ratio


def _compactness(area: float, perimeter: float) -> float:
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def run_sat_intersections(
    drive: str,
    candidates: List[Point],
    traj_points: Iterable[Tuple[float, float]],
    outputs_dir: Path,
    crs_epsg: int = 32632,
    patch_m: float = 256.0,
    conf_thr: float = 0.3,
    dop20_root: Optional[Path] = None,
) -> Dict[str, object]:
    if np is None or rasterio is None:
        return {"present": False, "reason": "missing_dependencies"}
    if dop20_root is None or not dop20_root.exists():
        return {"present": False, "reason": "dop20_root_missing"}
    tiles_dir = dop20_root / "tiles_utm32"
    if not tiles_dir.exists():
        return {"present": False, "reason": "tiles_dir_missing"}
    if not candidates:
        return {"present": False, "reason": "no_candidates"}

    try:
        index_items = load_tile_index(dop20_root, tiles_dir)
    except Exception as exc:
        return {"present": False, "reason": f"index_failed:{exc}"}
    if not index_items:
        return {"present": False, "reason": "no_tiles"}

    features = []
    metrics = []
    polys = []
    traj_pts = [Point(xy) for xy in traj_points]

    for idx, pt in enumerate(candidates):
        tile_path = _find_tile_for_point(index_items, pt.x, pt.y)
        if not tile_path:
            continue
        patch = _read_patch(tile_path, pt.x, pt.y, patch_m)
        if patch is None:
            continue
        _, ratio = _road_mask_simple(patch)
        conf = float(ratio)
        if conf < conf_thr:
            continue
        radius = 12.0 + 20.0 * conf
        poly = pt.buffer(radius)
        polys.append(poly)
        traj_support = sum(1 for p in traj_pts if poly.contains(p))
        area = float(poly.area)
        perim = float(poly.length)
        feature = {
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "drive": drive,
                "candidate_id": f"sat_{idx}",
                "sat_confidence": round(conf, 4),
                "traj_support": int(traj_support),
                "lidar_support": 0,
                "area_m2": round(area, 2),
                "compactness": round(_compactness(area, perim), 4),
            },
        }
        features.append(feature)
        metrics.append(feature["properties"])

    sat_path = outputs_dir / "intersections_sat.geojson"
    sat_wgs84_path = outputs_dir / "intersections_sat_wgs84.geojson"
    _safe_write_json(sat_path, {"type": "FeatureCollection", "features": features})

    if Transformer is not None:
        wgs84 = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)
        wgs84_features = []
        for feat in features:
            geom = geom_transform(wgs84.transform, shape(feat["geometry"]))
            wgs84_features.append(
                {"type": "Feature", "geometry": mapping(geom), "properties": feat["properties"]}
            )
        _safe_write_json(sat_wgs84_path, {"type": "FeatureCollection", "features": wgs84_features})

    metrics_path = outputs_dir / "intersections_sat_metrics.json"
    _safe_write_json(
        metrics_path,
        {
            "drive": drive,
            "sat_present": bool(features),
            "patch_m": patch_m,
            "conf_thr": conf_thr,
            "count": len(features),
            "items": metrics,
        },
    )

    avg_conf = 0.0
    if metrics:
        avg_conf = sum(m.get("sat_confidence", 0.0) for m in metrics) / len(metrics)
    return {
        "present": bool(features),
        "count": len(features),
        "metrics_path": str(metrics_path),
        "polys": polys,
        "avg_confidence": round(float(avg_conf), 4),
    }
