from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import yaml
from shapely.ops import unary_union


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_validation_250_500")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _drive_tag(drive_id: str) -> str:
    if not drive_id:
        return "unknown"
    return drive_id.split("_")[-2] if "_" in drive_id else drive_id


def _find_latest_sat_output(runs_dir: Path) -> Optional[Path]:
    candidates: list[Path] = []
    patterns = ["sat_intersections_*", "sat_intersections_full_*", "sat_intersections_full_golden8"]
    for pat in patterns:
        for run in runs_dir.glob(pat):
            outputs = run / "outputs"
            if not outputs.exists():
                continue
            for name in ("intersections_sat.geojson", "intersections_sat_wgs84.geojson"):
                path = outputs / name
                if path.exists():
                    candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_sat_roi(path: Path, drive_id: str, out_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        if "wgs84" in path.name.lower():
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            gdf = gdf.set_crs(out_crs, allow_override=True)
    if drive_id and "drive_id" in gdf.columns:
        gdf = gdf[gdf["drive_id"].astype(str) == drive_id]
    if gdf.crs and str(gdf.crs).lower() != out_crs.lower():
        gdf = gdf.to_crs(out_crs)
    return gdf


def _write_empty_geojson(path: Path) -> None:
    payload = {"type": "FeatureCollection", "features": [], "crs": {"type": "name", "properties": {"name": "EPSG:32632"}}}
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _safe_read_layer(path: Path, layer: str, crs: str) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    try:
        return gpd.read_file(path, layer=layer)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _filter_by_roi(gdf: gpd.GeoDataFrame, roi: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if roi is None or roi.empty or gdf.empty:
        return gdf
    geom_union = unary_union([geom for geom in roi.geometry if geom is not None and not geom.is_empty])
    if geom_union is None or geom_union.is_empty:
        return gdf
    mask = gdf.geometry.apply(lambda g: g is not None and not g.is_empty and g.intersects(geom_union))
    return gdf[mask].copy()


def _run_crosswalk_monitor_range(python_bin: Path, config: str, out_run: Path, args: argparse.Namespace) -> int:
    cmd = [
        str(python_bin),
        str(Path("tools") / "run_crosswalk_monitor_range.py"),
        "--config",
        config,
        "--out-run",
        str(out_run),
    ]
    if args.drive:
        cmd += ["--drive", args.drive]
    if args.frame_start is not None:
        cmd += ["--frame-start", str(args.frame_start)]
    if args.frame_end is not None:
        cmd += ["--frame-end", str(args.frame_end)]
    if args.kitti_root:
        cmd += ["--kitti-root", args.kitti_root]
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_range_250_500_strict.yaml")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--out-run", default="")
    ap.add_argument("--sat-path", default="")
    args = ap.parse_args()

    log = _setup_logger()
    cfg = _load_yaml(Path(args.config))
    drive_id = args.drive or str(cfg.get("drive_id") or "")
    frame_start = args.frame_start if args.frame_start is not None else int(cfg.get("frame_start", 0))
    frame_end = args.frame_end if args.frame_end is not None else int(cfg.get("frame_end", 0))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_run) if args.out_run else Path("runs") / f"validation_{_drive_tag(drive_id)}_{frame_start}_{frame_end}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    python_bin = Path(sys.executable)
    ret = _run_crosswalk_monitor_range(python_bin, args.config, run_dir, args)
    if ret != 0:
        log.error("crosswalk monitor range failed (code=%d)", ret)
        return ret

    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sat_path = Path(args.sat_path) if args.sat_path else None
    if sat_path and not sat_path.exists():
        sat_path = None
    if sat_path is None:
        sat_path = _find_latest_sat_output(Path("runs"))

    sat_roi_path = outputs_dir / "sat_roi.geojson"
    sat_roi_gdf: Optional[gpd.GeoDataFrame] = None
    if sat_path and sat_path.exists():
        try:
            sat_roi_gdf = _load_sat_roi(sat_path, drive_id, "EPSG:32632")
            if sat_roi_gdf.empty:
                _write_empty_geojson(sat_roi_path)
            else:
                sat_roi_gdf.to_file(sat_roi_path, driver="GeoJSON")
            log.info("sat_roi: %s", sat_roi_path)
        except Exception as exc:
            log.warning("sat_roi load failed: %s", exc)
            _write_empty_geojson(sat_roi_path)
    else:
        _write_empty_geojson(sat_roi_path)
        log.info("sat_roi: empty (no sat outputs found)")

    entities_path = outputs_dir / "crosswalk_entities_utm32.gpkg"
    candidates_layer = _safe_read_layer(entities_path, "crosswalk_candidate_poly", "EPSG:32632")
    candidates_out = outputs_dir / "crosswalk_candidates.gpkg"
    if candidates_out.exists():
        candidates_out.unlink()
    candidates_layer.to_file(candidates_out, layer="crosswalk_candidate_poly", driver="GPKG")

    review_layer = _safe_read_layer(entities_path, "crosswalk_review_poly", "EPSG:32632")
    final_layer = _safe_read_layer(entities_path, "crosswalk_poly", "EPSG:32632")
    review_layer = _filter_by_roi(review_layer, sat_roi_gdf)
    final_layer = _filter_by_roi(final_layer, sat_roi_gdf)
    fused_out = outputs_dir / "fused_candidates.gpkg"
    if fused_out.exists():
        fused_out.unlink()
    review_layer.to_file(fused_out, layer="crosswalk_review_poly", driver="GPKG")
    final_layer.to_file(fused_out, layer="crosswalk_poly", driver="GPKG", mode="a")

    log.info("outputs_dir=%s", outputs_dir)
    log.info("crosswalk_candidates=%s", candidates_out)
    log.info("fused_candidates=%s", fused_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
