from __future__ import annotations

import datetime
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
IndexMeta = Dict[str, object]


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


def _load_index_cache(cache_path: Path) -> Tuple[Optional[RasterIndex], Optional[IndexMeta]]:
    if not cache_path.exists():
        return None, None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        items = payload.get("items")
        meta = payload.get("meta")
        if isinstance(items, list):
            return items, meta if isinstance(meta, dict) else None
    return None, None


def _index_bbox(items: RasterIndex) -> Optional[Dict[str, float]]:
    if not items:
        return None
    return {
        "minx": float(min(item["minx"] for item in items)),
        "miny": float(min(item["miny"] for item in items)),
        "maxx": float(max(item["maxx"] for item in items)),
        "maxy": float(max(item["maxy"] for item in items)),
    }


def _save_index_cache(cache_path: Path, items: RasterIndex, meta: Optional[IndexMeta]) -> None:
    payload = {"meta": meta or {}, "items": items}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_path(dop20_root: Path) -> Path:
    override = os.environ.get("DOP20_INDEX_CACHE", "").strip()
    if override:
        return Path(override)
    return dop20_root / "dop20_tiles_index.json"


def load_tile_index(
    dop20_root: Path,
    tiles_dir: Path,
    *,
    crs_epsg: Optional[int] = None,
    force_rebuild: bool = False,
) -> Tuple[RasterIndex, Optional[IndexMeta]]:
    cache_path = _cache_path(dop20_root)
    cached, meta = _load_index_cache(cache_path)
    dop20_root_abs = str(dop20_root.resolve())
    if cached is not None and not force_rebuild:
        cached_root = (meta or {}).get("root_abs")
        if cached_root:
            try:
                if str(Path(str(cached_root)).resolve()) != dop20_root_abs:
                    cached = None
            except Exception:
                cached = None
        else:
            cached = None
    if cached is not None:
        return cached, meta
    items = _scan_tiles(tiles_dir)
    meta = {
        "root_abs": dop20_root_abs,
        "tiles_dir_abs": str(tiles_dir.resolve()),
        "tiles_count": len(items),
        "bbox": _index_bbox(items),
        "crs_epsg": crs_epsg,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        _save_index_cache(cache_path, items, meta)
    except PermissionError:
        fallback = Path("cache") / "dop20_tiles_index.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        _save_index_cache(fallback, items, meta)
    return items, meta


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
    base = {
        "present": False,
        "reason": "",
        "count": 0,
        "avg_confidence": 0.0,
        "conf_mean": 0.0,
        "conf_p50": 0.0,
        "conf_p95": 0.0,
        "candidates_total": int(len(candidates)),
        "candidates_used": 0,
        "tiles_total": 0,
        "tiles_hit": 0,
        "patches_read": 0,
        "read_errors": 0,
    }
    if np is None or rasterio is None:
        return {**base, "reason": "missing_dependencies"}
    if dop20_root is None or not dop20_root.exists():
        return {**base, "reason": "dop20_root_missing"}
    tiles_dir = dop20_root / "tiles_utm32"
    if not tiles_dir.exists():
        return {**base, "reason": "tiles_dir_missing"}
    if not candidates:
        return {**base, "reason": "no_candidates"}

    try:
        index_items, _ = load_tile_index(dop20_root, tiles_dir, crs_epsg=crs_epsg)
    except Exception as exc:
        return {**base, "reason": f"index_failed:{exc}"}
    if not index_items:
        return {**base, "reason": "no_tiles"}
    base["tiles_total"] = len(index_items)

    features = []
    metrics = []
    polys = []
    traj_pts = [Point(xy) for xy in traj_points]
    conf_values: List[float] = []

    for idx, pt in enumerate(candidates):
        tile_path = _find_tile_for_point(index_items, pt.x, pt.y)
        if not tile_path:
            continue
        base["tiles_hit"] += 1
        patch = _read_patch(tile_path, pt.x, pt.y, patch_m)
        if patch is None:
            base["read_errors"] += 1
            continue
        base["patches_read"] += 1
        _, ratio = _road_mask_simple(patch)
        conf = float(ratio)
        conf_values.append(conf)
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

    avg_conf = 0.0
    if metrics:
        avg_conf = sum(m.get("sat_confidence", 0.0) for m in metrics) / len(metrics)
    if conf_values:
        conf_sorted = sorted(conf_values)
        base["conf_mean"] = round(float(sum(conf_values) / len(conf_values)), 4)
        mid = conf_sorted[len(conf_sorted) // 2]
        base["conf_p50"] = round(float(mid), 4)
        base["conf_p95"] = round(float(conf_sorted[int(max(0, len(conf_sorted) * 0.95) - 1)]), 4)

    reason = "OK"
    if not features:
        if base["candidates_total"] <= 0:
            reason = "no_candidates"
        elif base["tiles_total"] <= 0:
            reason = "no_tiles"
        elif base["tiles_hit"] <= 0:
            reason = "no_tiles"
        elif base["patches_read"] <= 0:
            reason = "read_error"
        else:
            reason = "low_confidence"
    metrics_path = outputs_dir / "intersections_sat_metrics.json"
    base.update(
        {
            "present": bool(features),
            "reason": reason,
            "count": len(features),
            "metrics_path": str(metrics_path),
            "polys": polys,
            "avg_confidence": round(float(avg_conf), 4),
            "candidates_used": len(features),
        }
    )
    _safe_write_json(
        metrics_path,
        {
            "drive": drive,
            "sat_present": bool(features),
            "patch_m": patch_m,
            "conf_thr": conf_thr,
            "count": len(features),
            "missing_reason": reason,
            "candidates_total": base["candidates_total"],
            "candidates_used": len(features),
            "tiles_hit": base["tiles_hit"],
            "tiles_total": base["tiles_total"],
            "patches_read": base["patches_read"],
            "read_errors": base["read_errors"],
            "items": metrics,
        },
    )
    base["metrics_path"] = str(metrics_path)
    return base
