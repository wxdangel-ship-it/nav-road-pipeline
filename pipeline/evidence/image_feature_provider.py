from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import geopandas as gpd

try:
    import pyogrio
except Exception:
    pyogrio = None
from shapely.geometry import box, shape


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_records(gdf: gpd.GeoDataFrame) -> List[dict]:
    out = []
    for _, row in gdf.iterrows():
        props = dict(row.drop(labels=["geometry"], errors="ignore"))
        props["conf"] = _to_float(props.get("conf"))
        out.append({"geometry": row.geometry, "properties": props})
    return out


def _read_gpkg(path: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    layers = []
    if pyogrio is not None:
        try:
            layers = [name for name, _ in pyogrio.list_layers(path)]
        except Exception:
            layers = []
    if not layers:
        try:
            layers = gpd.io.file.fiona.listlayers(str(path))
        except Exception:
            layers = []
    for layer in layers:
        gdf = gpd.read_file(path, layer=layer)
        out[layer] = _as_records(gdf)
    return out


def _read_geojson(path: Path) -> Dict[str, List[dict]]:
    gdf = gpd.read_file(path)
    grouped: Dict[str, List[dict]] = {}
    if "class" in gdf.columns:
        for cls, sub in gdf.groupby("class"):
            grouped[str(cls)] = _as_records(sub)
    else:
        grouped["unknown"] = _as_records(gdf)
    return grouped


def _read_detection_json(path: Path) -> List[dict]:
    data = _load_json(path)
    if isinstance(data, dict) and "detections" in data:
        return data["detections"]
    if isinstance(data, list):
        return data
    return []


def _bbox_to_geom(bbox: Iterable[float]):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return box(x1, y1, x2, y2)


def load_features(
    drive_id: str,
    frame_ids: Optional[Iterable[str]],
    feature_store_dir: Path,
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    drive_dir = feature_store_dir / drive_id
    if not drive_dir.exists():
        return out

    if frame_ids is None:
        frame_dirs = sorted([p for p in drive_dir.iterdir() if p.is_dir()])
    else:
        frame_dirs = [drive_dir / str(fid) for fid in frame_ids]

    for frame_dir in frame_dirs:
        gpkg = frame_dir / "image_features.gpkg"
        geojson = frame_dir / "image_features.geojson"
        det_json = frame_dir / "traffic_light_dets.json"

        if gpkg.exists():
            layers = _read_gpkg(gpkg)
            for cls, recs in layers.items():
                out.setdefault(cls, []).extend(recs)
        elif geojson.exists():
            layers = _read_geojson(geojson)
            for cls, recs in layers.items():
                out.setdefault(cls, []).extend(recs)

        if det_json.exists():
            dets = _read_detection_json(det_json)
            records = []
            for det in dets:
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                props = dict(det)
                props["conf"] = _to_float(props.get("conf"))
                props["class"] = props.get("class") or "traffic_light"
                props["frame_id"] = props.get("frame_id") or frame_dir.name
                props["drive_id"] = props.get("drive_id") or drive_id
                records.append({"geometry": _bbox_to_geom(bbox), "properties": props})
            if records:
                out.setdefault("traffic_light", []).extend(records)

    return out
