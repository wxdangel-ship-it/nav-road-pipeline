from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, unary_union

try:
    import pyogrio
except Exception:
    pyogrio = None

try:
    from shapely import set_precision as _set_precision
except Exception:
    _set_precision = None


LINE_CLASSES = {"divider_median", "lane_marking", "road_edge", "stop_line"}
POLY_CLASSES = {"crosswalk", "arrow", "gore_marking"}


def _list_layers(path: Path) -> List[str]:
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
    return layers


def _read_all_layers(path: Path) -> gpd.GeoDataFrame:
    frames = []
    for layer in _list_layers(path):
        try:
            gdf = gpd.read_file(path, layer=layer)
        except Exception:
            continue
        if "class" not in gdf.columns:
            gdf["class"] = layer
        frames.append(gdf)
    if not frames:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry")


def _snap_geom(geom, grid: float):
    if _set_precision is None:
        return geom
    try:
        return _set_precision(geom, grid)
    except Exception:
        return geom


def _clip_geom(geom, clip_poly, min_overlap: float):
    if geom is None or geom.is_empty:
        return None
    try:
        inter = geom.intersection(clip_poly)
    except Exception:
        try:
            inter = geom.buffer(0).intersection(clip_poly)
        except Exception:
            return None
    if inter.is_empty:
        return None
    if geom.geom_type in {"Polygon", "MultiPolygon"}:
        try:
            ratio = inter.area / max(1e-6, geom.area)
        except Exception:
            ratio = 0.0
        if ratio < min_overlap:
            return None
    return inter


def _lengths(gdf: gpd.GeoDataFrame) -> List[float]:
    if gdf.empty:
        return []
    return [float(v) for v in gdf.length.tolist()]


def _areas(gdf: gpd.GeoDataFrame) -> List[float]:
    if gdf.empty:
        return []
    return [float(v) for v in gdf.area.tolist()]


def _stats(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None}
    arr = np.array(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}


def _clean_lines(gdf: gpd.GeoDataFrame, grid: float, simplify_m: float, min_len_m: float) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty and g.geom_type in {"LineString", "MultiLineString"}])
    if geom.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    geom = _snap_geom(geom, grid)
    geom = linemerge(geom)
    geoms = []
    if isinstance(geom, LineString):
        geoms = [geom]
    else:
        try:
            geoms = list(geom.geoms)
        except Exception:
            geoms = []
    cleaned = []
    for g in geoms:
        if simplify_m > 0:
            g = g.simplify(simplify_m)
        if g.length >= min_len_m:
            cleaned.append(g)
    return gpd.GeoDataFrame(geometry=cleaned, crs=gdf.crs)


def _clean_polys(gdf: gpd.GeoDataFrame, grid: float, simplify_m: float, min_area_m2: float) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
    if geom.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    geom = _snap_geom(geom, grid)
    geoms = []
    if isinstance(geom, Polygon):
        geoms = [geom]
    else:
        try:
            geoms = list(geom.geoms)
        except Exception:
            geoms = []
    cleaned = []
    for g in geoms:
        if simplify_m > 0:
            g = g.simplify(simplify_m)
        if g.area >= min_area_m2:
            cleaned.append(g)
    return gpd.GeoDataFrame(geometry=cleaned, crs=gdf.crs)


def _write_layers(gdf: gpd.GeoDataFrame, out_path: Path, write_wgs84: bool):
    if gdf.empty:
        return
    for cls, sub in gdf.groupby("class"):
        sub.to_file(out_path, layer=str(cls), driver="GPKG")
        if write_wgs84:
            sub_wgs84 = sub.to_crs(4326)
            sub_wgs84.to_file(out_path, layer=f"{cls}_wgs84", driver="GPKG")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-store-map-root", required=True)
    ap.add_argument("--road-polygons-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grid-m", type=float, default=0.1)
    ap.add_argument("--simplify-m", type=float, default=0.2)
    ap.add_argument("--min-line-len-m", type=float, default=0.5)
    ap.add_argument("--min-poly-area-m2", type=float, default=0.3)
    ap.add_argument("--clip-buffer-m", type=float, default=2.0)
    ap.add_argument("--min-overlap-road", type=float, default=0.5)
    ap.add_argument("--write-wgs84", type=int, default=1)
    ap.add_argument("--write-merged", type=int, default=1)
    args = ap.parse_args()

    fs_root = Path(args.feature_store_map_root)
    road_root = Path(args.road_polygons_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"drives": {}, "params": vars(args)}
    merged_frames: List[gpd.GeoDataFrame] = []

    for drive_dir in sorted([p for p in fs_root.iterdir() if p.is_dir()]):
        drive_id = drive_dir.name
        road_path = road_root / drive_id / "geom_outputs" / "road_polygon.geojson"
        if not road_path.exists():
            road_path = road_root / drive_id / "road_polygon.geojson"
        if not road_path.exists():
            report["drives"][drive_id] = {"status": "skip", "reason": "missing_road_polygon"}
            continue
        road_gdf = gpd.read_file(road_path)
        if road_gdf.empty:
            report["drives"][drive_id] = {"status": "skip", "reason": "empty_road_polygon"}
            continue
        road_poly = road_gdf.geometry.unary_union
        clip_poly = road_poly.buffer(float(args.clip_buffer_m))

        drive_frames: List[gpd.GeoDataFrame] = []
        for frame_dir in sorted([p for p in drive_dir.iterdir() if p.is_dir()]):
            gpkg = frame_dir / "image_features.gpkg"
            if not gpkg.exists():
                continue
            gdf = _read_all_layers(gpkg)
            if gdf.empty:
                continue
            gdf["drive_id"] = gdf.get("drive_id", drive_id)
            if gdf.crs is None:
                gdf = gdf.set_crs(32632, allow_override=True)
            gdf["geometry"] = gdf["geometry"].apply(lambda g: _clip_geom(g, clip_poly, float(args.min_overlap_road)))
            gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()]
            if not gdf.empty:
                drive_frames.append(gdf)

        if not drive_frames:
            report["drives"][drive_id] = {"status": "skip", "reason": "no_features_after_clip"}
            continue

        raw = gpd.GeoDataFrame(pd.concat(drive_frames, ignore_index=True), geometry="geometry")
        raw_stats = {
            "count_raw": int(len(raw)),
            "len_raw": _stats(_lengths(raw[raw["class"].isin(LINE_CLASSES)])),
            "area_raw": _stats(_areas(raw[raw["class"].isin(POLY_CLASSES)])),
        }

        cleaned_layers = []
        for cls, sub in raw.groupby("class"):
            if cls in LINE_CLASSES:
                clean = _clean_lines(sub, args.grid_m, args.simplify_m, args.min_line_len_m)
            elif cls in POLY_CLASSES:
                clean = _clean_polys(sub, args.grid_m, args.simplify_m, args.min_poly_area_m2)
            else:
                clean = sub.copy()
            if clean.empty:
                continue
            clean["class"] = cls
            for col in ["drive_id", "frame_id", "conf", "subtype", "model_id"]:
                if col not in clean.columns:
                    clean[col] = None
            cleaned_layers.append(clean)

        if not cleaned_layers:
            report["drives"][drive_id] = {"status": "skip", "reason": "no_features_after_clean"}
            continue

        cleaned = gpd.GeoDataFrame(pd.concat(cleaned_layers, ignore_index=True), geometry="geometry")
        clean_stats = {
            "count_clean": int(len(cleaned)),
            "len_clean": _stats(_lengths(cleaned[cleaned["class"].isin(LINE_CLASSES)])),
            "area_clean": _stats(_areas(cleaned[cleaned["class"].isin(POLY_CLASSES)])),
        }

        out_drive_dir = out_dir / drive_id
        out_drive_dir.mkdir(parents=True, exist_ok=True)
        out_gpkg = out_drive_dir / f"evidence_clean_{drive_id}.gpkg"
        _write_layers(cleaned, out_gpkg, bool(args.write_wgs84))
        merged_frames.append(cleaned)
        report["drives"][drive_id] = {"status": "ok", **raw_stats, **clean_stats}

    if args.write_merged and merged_frames:
        merged = gpd.GeoDataFrame(pd.concat(merged_frames, ignore_index=True), geometry="geometry")
        out_gpkg = out_dir / "evidence_clean_golden8.gpkg"
        _write_layers(merged, out_gpkg, bool(args.write_wgs84))

    (out_dir / "evidence_clean_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[EVIDENCE] cleaned outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
