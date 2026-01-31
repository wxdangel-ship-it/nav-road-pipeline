from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon, box

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import rasterio
    from rasterio import features as rio_features
except Exception:
    rasterio = None
    rio_features = None


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _extract_frame_id(path: Path, regex: Optional[str]) -> str:
    text = path.stem
    if regex:
        m = re.search(regex, text)
        if m:
            return m.group(1)
    m = re.search(r"(\d{6,})", text)
    if m:
        return m.group(1)
    return text


def _find_candidates(root: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    seg = []
    det = []
    other = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        name = str(p).lower()
        if p.suffix.lower() in {".png", ".npy"} and any(k in name for k in ("seg", "mask", "semantic")):
            seg.append(p)
        elif p.suffix.lower() in {".json", ".jsonl"} and any(k in name for k in ("det", "bbox", "objects")):
            det.append(p)
        else:
            other.append(p)
    return seg, det, other


def _load_meta(path: Path) -> tuple[Optional[dict], Optional[str]]:
    if not path.exists():
        return None, "meta.json not found; assume outputs already in original image px"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "meta.json parse failed; assume outputs already in original image px"
    return data, None


def _build_transform(meta: dict | None):
    if not meta:
        return None
    resize_mode = str(meta.get("resize_mode") or "none").lower()
    if resize_mode == "none":
        return None
    if resize_mode == "letterbox":
        scale = float(meta.get("scale") or meta.get("letterbox", {}).get("scale") or 1.0)
        pad = meta.get("pad") or meta.get("letterbox", {}).get("pad") or [0.0, 0.0]
        pad_x, pad_y = float(pad[0]), float(pad[1])

        def _transform_xy(x: float, y: float) -> tuple[float, float]:
            return (x - pad_x) / max(1e-6, scale), (y - pad_y) / max(1e-6, scale)

        return _transform_xy
    if resize_mode == "resize":
        scale = meta.get("scale") or meta.get("resize", {}).get("scale")
        if scale and isinstance(scale, (int, float)):
            sx = sy = float(scale)
        else:
            sx = float(meta.get("scale_x") or meta.get("resize", {}).get("scale_x") or 1.0)
            sy = float(meta.get("scale_y") or meta.get("resize", {}).get("scale_y") or 1.0)

        def _transform_xy(x: float, y: float) -> tuple[float, float]:
            return x / max(1e-6, sx), y / max(1e-6, sy)

        return _transform_xy
    return None


def _apply_transform_geom(geom, transform_xy):
    if transform_xy is None or geom is None or geom.is_empty:
        return geom
    from shapely.ops import transform as shapely_transform

    return shapely_transform(lambda x, y, z=None: transform_xy(x, y), geom)


def _apply_transform_bbox(bbox: Iterable[float], transform_xy):
    if transform_xy is None:
        return bbox
    x1, y1, x2, y2 = [float(v) for v in bbox]
    nx1, ny1 = transform_xy(x1, y1)
    nx2, ny2 = transform_xy(x2, y2)
    return [nx1, ny1, nx2, ny2]


def _read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".png":
        if Image is None:
            raise RuntimeError("PIL is required to read PNG masks.")
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 3:
            raise RuntimeError("RGB mask PNG not supported yet; provide single-channel class_id PNG.")
        return arr
    raise RuntimeError(f"Unsupported mask extension: {path.suffix}")


def _mask_polygons(mask: np.ndarray, class_id: int) -> List[Polygon]:
    if rio_features is None:
        return []
    geom_list = []
    for geom, value in rio_features.shapes(mask.astype(np.int32), mask=mask == class_id):
        if int(value) != class_id:
            continue
        poly = Polygon(geom["coordinates"][0])
        if not poly.is_empty and poly.area > 0:
            geom_list.append(poly)
    return geom_list


def _poly_to_line(poly: Polygon) -> LineString:
    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny
    if dx >= dy:
        return LineString([(minx, (miny + maxy) * 0.5), (maxx, (miny + maxy) * 0.5)])
    return LineString([((minx + maxx) * 0.5, miny), ((minx + maxx) * 0.5, maxy)])


def _vectorize_mask(mask: np.ndarray, class_id: int, geom_type: str) -> List[Any]:
    if geom_type == "Polygon":
        return _mask_polygons(mask, class_id)
    if geom_type == "LineString":
        polys = _mask_polygons(mask, class_id)
        return [_poly_to_line(p) for p in polys]
    return []


def _class_from_id(class_id: int, seg_schema: dict) -> Optional[str]:
    mapping = seg_schema.get("id_to_class") or {}
    return mapping.get(class_id)


def _class_from_name(name: str, seg_schema: dict) -> Optional[str]:
    mapping = seg_schema.get("name_to_class") or {}
    return mapping.get(name)


def _load_feature_schema(path: Path) -> dict:
    data = _load_yaml(path)
    return data.get("feature_schema") or data


def _write_gpkg(path: Path, by_class: Dict[str, List[dict]]) -> None:
    if path.exists():
        path.unlink()
    for cls, feats in by_class.items():
        if not feats:
            continue
        gdf = gpd.GeoDataFrame(
            [f["properties"] for f in feats],
            geometry=[f["geometry"] for f in feats],
            crs=None,
        )
        gdf.to_file(path, layer=cls, driver="GPKG")


def _write_geojson(path: Path, feats: List[dict]) -> None:
    if not feats:
        return
    gdf = gpd.GeoDataFrame(
        [f["properties"] for f in feats],
        geometry=[f["geometry"] for f in feats],
        crs=None,
    )
    gdf.to_file(path, driver="GeoJSON")


def _conf_stats(values: List[float]) -> dict:
    if not values:
        return {"p50": None, "p90": None}
    arr = np.array(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--model-out-dir", required=True)
    ap.add_argument("--out-run-dir", required=True)
    ap.add_argument("--seg-schema", default="configs/seg_schema.yaml")
    ap.add_argument("--feature-schema", default="configs/feature_schema.yaml")
    ap.add_argument("--frame-regex", default=None)
    ap.add_argument("--adapter", default="auto", choices=["auto", "seg", "det"])
    ap.add_argument("--model-id", default="unknown")
    ap.add_argument("--model-version", default="v1")
    ap.add_argument("--geometry-frame", default="image_px", choices=["image_px", "cam", "ego", "map"])
    ap.add_argument("--write-geojson", type=int, default=0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    drive_id = args.drive
    model_out_dir = Path(args.model_out_dir)
    out_run_dir = Path(args.out_run_dir)
    seg_schema = _load_yaml(Path(args.seg_schema))
    feature_schema = _load_feature_schema(Path(args.feature_schema))

    if not seg_schema.get("id_to_class") and not seg_schema.get("name_to_class"):
        raise SystemExit("ERROR: seg_schema missing id_to_class/name_to_class mapping.")

    seg_files, det_files, other_files = _find_candidates(model_out_dir)
    if seg_files and rio_features is None:
        print("[FEATURES][WARN] rasterio not available; mask vectorization will be empty.")
    if args.adapter == "seg":
        det_files = []
    elif args.adapter == "det":
        seg_files = []

    meta, meta_note = _load_meta(model_out_dir / "meta.json")
    transform_xy = _build_transform(meta)

    if not seg_files and not det_files:
        out_dir = out_run_dir / "feature_store"
        out_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "drive_id": drive_id,
            "model_out_dir": str(model_out_dir),
            "adapter": args.adapter,
            "counts": {},
            "frames_with_class": {},
            "conf_stats": {},
            "note": "no recognizable model outputs found",
            "meta": meta,
            "meta_note": meta_note,
            "probe_paths": {
                "seg_candidates": [str(p) for p in seg_files],
                "det_candidates": [str(p) for p in det_files],
                "other_files": [str(p) for p in other_files[:50]],
            },
        }
        (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        print("[FEATURES] no recognizable model outputs found.")
        return 0

    det_cache: Dict[Path, List[dict]] = {}
    frame_map: Dict[str, Dict[str, List[Path]]] = {}
    for p in seg_files:
        fid = _extract_frame_id(p, args.frame_regex)
        frame_map.setdefault(fid, {}).setdefault("seg", []).append(p)
    for p in det_files:
        fid = _extract_frame_id(p, args.frame_regex)
        frame_map.setdefault(fid, {}).setdefault("det", []).append(p)

    if args.max_frames and args.max_frames > 0:
        selected = sorted(frame_map.keys())[: args.max_frames]
        frame_map = {k: frame_map[k] for k in selected}

    out_root = out_run_dir / "feature_store" / drive_id
    out_root.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    frames_with: Dict[str, int] = {}
    confs: Dict[str, List[float]] = {}

    map_hint = model_out_dir / "divider_map_hint.geojson"
    map_hint_feats = None
    if map_hint.exists():
        try:
            map_hint_feats = gpd.read_file(map_hint)
        except Exception:
            map_hint_feats = None

    for frame_id, sources in sorted(frame_map.items()):
        frame_dir = out_root / frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        gpkg_path = frame_dir / "image_features.gpkg"
        geojson_path = frame_dir / "image_features.geojson"
        det_json_path = frame_dir / "traffic_light_dets.json"

        if args.resume and gpkg_path.exists():
            continue

        by_class: Dict[str, List[dict]] = {}
        all_feats: List[dict] = []

        if map_hint_feats is not None and not map_hint_feats.empty:
            for _, row in map_hint_feats.iterrows():
                geom = row.geometry
                cls = str(row.get("class") or "divider_median")
                props = {
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "timestamp": None,
                    "src_modality": "image",
                    "model_id": args.model_id,
                    "model_version": args.model_version,
                    "class": cls,
                    "subtype": row.get("subtype"),
                    "conf": _to_float(row.get("conf")),
                    "geometry_frame": "map",
                    "notes": "map_hint",
                }
                feat = {"geometry": geom, "properties": props}
                by_class.setdefault(cls, []).append(feat)
                all_feats.append(feat)
                counts[cls] = counts.get(cls, 0) + 1
                frames_with[cls] = frames_with.get(cls, 0) + 1

        for seg_path in sources.get("seg", []):
            mask = _read_mask(seg_path)
            class_ids = np.unique(mask.astype(np.int32)).tolist()
            for cid in class_ids:
                if cid == 0:
                    continue
                cls = _class_from_id(int(cid), seg_schema)
                if not cls or cls == "background":
                    continue
                geom_type = feature_schema.get(cls, {}).get("geometry_type", "LineString")
                geoms = _vectorize_mask(mask, int(cid), geom_type)
                for geom in geoms:
                    geom = _apply_transform_geom(geom, transform_xy)
                    props = {
                        "drive_id": drive_id,
                        "frame_id": frame_id,
                        "timestamp": None,
                        "src_modality": "image",
                        "model_id": args.model_id,
                        "model_version": args.model_version,
                        "class": cls,
                        "subtype": None,
                        "conf": None,
                        "geometry_frame": args.geometry_frame,
                        "notes": None,
                    }
                    feat = {"geometry": geom, "properties": props}
                    by_class.setdefault(cls, []).append(feat)
                    all_feats.append(feat)
                    counts[cls] = counts.get(cls, 0) + 1
            if class_ids:
                for cid in class_ids:
                    cls = _class_from_id(int(cid), seg_schema)
                    if cls and cls != "background":
                        frames_with[cls] = frames_with.get(cls, 0) + 1

        det_records: List[dict] = []
        for det_path in sources.get("det", []):
            if det_path in det_cache:
                det_records = det_cache[det_path]
            else:
                if det_path.suffix.lower() == ".jsonl":
                    lines = det_path.read_text(encoding="utf-8").splitlines()
                    for line in lines:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if isinstance(rec, list):
                            det_records.extend(rec)
                        else:
                            det_records.append(rec)
                else:
                    data = json.loads(det_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "detections" in data:
                        det_records.extend(data["detections"])
                    elif isinstance(data, list):
                        det_records.extend(data)
                det_cache[det_path] = det_records

        det_out: List[dict] = []
        for det in det_records:
            det_frame = det.get("frame_id")
            if det_frame is not None and str(det_frame) != str(frame_id):
                continue
            cls_raw = det.get("class") or det.get("class_name")
            cls = _class_from_name(str(cls_raw).lower(), seg_schema) if cls_raw else None
            if cls is None:
                cls = "traffic_light"
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bbox = _apply_transform_bbox(bbox, transform_xy)
            geom = box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            conf = _to_float(det.get("conf") or det.get("score"))
            props = {
                "drive_id": drive_id,
                "frame_id": frame_id,
                "timestamp": det.get("timestamp"),
                "src_modality": "image",
                "model_id": args.model_id,
                "model_version": args.model_version,
                "class": cls,
                "subtype": det.get("subtype"),
                "conf": conf,
                "geometry_frame": args.geometry_frame,
                "notes": None,
            }
            det_out.append({"bbox": bbox, "class": cls, "conf": conf, "track_id": det.get("track_id")})
            feat = {"geometry": geom, "properties": props}
            by_class.setdefault(cls, []).append(feat)
            all_feats.append(feat)
            counts[cls] = counts.get(cls, 0) + 1
            if conf is not None:
                confs.setdefault(cls, []).append(conf)

        _write_gpkg(gpkg_path, by_class)
        if args.write_geojson:
            _write_geojson(geojson_path, all_feats)
        if det_out:
            det_json_path.write_text(json.dumps(det_out, indent=2), encoding="utf-8")

    conf_stats = {cls: _conf_stats(vals) for cls, vals in confs.items()}
    index = {
        "drive_id": drive_id,
        "model_out_dir": str(model_out_dir),
        "adapter": args.adapter,
        "counts": counts,
        "frames_with_class": frames_with,
        "conf_stats": conf_stats,
        "model_id": args.model_id,
        "model_version": args.model_version,
        "meta": meta,
        "meta_note": meta_note,
    }
    (out_run_dir / "feature_store" / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    print("[FEATURES] built feature_store:", out_run_dir / "feature_store")
    print("[FEATURES] counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
