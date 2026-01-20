from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd

try:
    import pyogrio
except Exception:
    pyogrio = None


KEEP_FIELDS = ["drive_id", "frame_id", "conf", "class", "subtype", "model_id"]


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


def _normalize_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for col in KEEP_FIELDS:
        if col not in gdf.columns:
            gdf[col] = None
    gdf = gdf[KEEP_FIELDS + ["geometry"]].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(32632, allow_override=True)
    return gdf


def _write_layers(gdf: gpd.GeoDataFrame, out_path: Path, write_wgs84: bool) -> None:
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--write-wgs84", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.feature_store_map_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wgs84 = bool(args.write_wgs84)

    merged_frames: List[gpd.GeoDataFrame] = []
    for drive_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        drive_id = drive_dir.name
        drive_out_dir = out_dir / drive_id
        drive_out_dir.mkdir(parents=True, exist_ok=True)
        out_gpkg = drive_out_dir / f"evidence_{drive_id}.gpkg"
        drive_frames: List[gpd.GeoDataFrame] = []
        for frame_dir in sorted([p for p in drive_dir.iterdir() if p.is_dir()]):
            gpkg = frame_dir / "image_features.gpkg"
            if not gpkg.exists():
                continue
            gdf = _read_all_layers(gpkg)
            if gdf.empty:
                continue
            gdf = _normalize_gdf(gdf)
            drive_frames.append(gdf)
        if drive_frames:
            merged_drive = gpd.GeoDataFrame(pd.concat(drive_frames, ignore_index=True), geometry="geometry")
            _write_layers(merged_drive, out_gpkg, write_wgs84)
            merged_frames.append(merged_drive)

    merged_gpkg = out_dir / "evidence_golden8.gpkg"
    if merged_frames:
        merged_all = gpd.GeoDataFrame(pd.concat(merged_frames, ignore_index=True), geometry="geometry")
        _write_layers(merged_all, merged_gpkg, write_wgs84)

    report = {
        "feature_store_map_root": str(root),
        "out_dir": str(out_dir),
        "drives": len([p for p in root.iterdir() if p.is_dir()]),
        "merged_written": merged_gpkg.exists(),
    }
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[EVIDENCE] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
