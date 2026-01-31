from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from pipeline.calib.kitti360_backproject import configure_default_context, world_to_pixel_cam0
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_json, write_text
from pipeline._io import load_yaml


CFG_DEFAULT = Path("configs/crosswalk_range_250_500_strict.yaml")


def _find_latest_crosswalk_run() -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        outputs = run_dir / "outputs"
        if not outputs.exists():
            continue
        if (outputs / "crosswalk_entities_utm32.gpkg").exists() or (outputs / "merged" / "crosswalk_candidates_utm32.gpkg").exists():
            candidates.append(run_dir)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_image_path(data_root: Path, drive_id: str, cam_id: str, frame_id: str) -> Optional[Path]:
    candidates = [
        data_root / "data_2d_raw" / drive_id / cam_id / "data_rect",
        data_root / "data_2d_raw" / drive_id / cam_id / "data",
        data_root / drive_id / cam_id / "data_rect",
        data_root / drive_id / cam_id / "data",
    ]
    for base in candidates:
        if not base.exists():
            continue
        for ext in [".png", ".jpg", ".jpeg"]:
            p = base / f"{frame_id}{ext}"
            if p.exists():
                return p
    return None


def _find_crosswalk_layer(gpkg: Path) -> Optional[str]:
    try:
        import pyogrio

        for name, _ in pyogrio.list_layers(gpkg):
            if "crosswalk" in name.lower():
                return name
    except Exception:
        pass
    try:
        for name in gpd.io.file.fiona.listlayers(str(gpkg)):
            if "crosswalk" in name.lower():
                return name
    except Exception:
        pass
    return None


def _load_raw_pixel_poly(feature_store_root: Path, drive_id: str, frame_id: str) -> Optional[Polygon]:
    gpkg_path = feature_store_root / "feature_store" / drive_id / frame_id / "image_features.gpkg"
    if not gpkg_path.exists():
        return None
    layer = _find_crosswalk_layer(gpkg_path)
    if not layer:
        return None
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except Exception:
        return None
    if gdf.empty:
        return None
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return unary_union(list(geom.geoms))
    return None


def _load_world_poly_from_run(run_dir: Path, frame_id: str) -> Optional[Polygon]:
    frame_path = run_dir / "outputs" / "frames" / frame_id / "crosswalk_frame_utm32.gpkg"
    if frame_path.exists():
        try:
            gdf = gpd.read_file(frame_path, layer="crosswalk_frame")
        except Exception:
            gdf = gpd.GeoDataFrame()
        if not gdf.empty:
            geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
            if geom.is_empty:
                return None
            if geom.geom_type == "Polygon":
                return geom
            if geom.geom_type == "MultiPolygon":
                return unary_union(list(geom.geoms))
    fallback = run_dir / "outputs" / "frame_candidates_utm32.gpkg"
    if fallback.exists():
        try:
            gdf = gpd.read_file(fallback, layer="frame_candidates")
        except Exception:
            gdf = gpd.GeoDataFrame()
        if not gdf.empty and "frame_id" in gdf.columns:
            gdf = gdf[gdf["frame_id"].astype(str) == frame_id]
            if not gdf.empty:
                geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
                if geom.is_empty:
                    return None
                if geom.geom_type == "Polygon":
                    return geom
                if geom.geom_type == "MultiPolygon":
                    return unary_union(list(geom.geoms))
    return None


def _sample_boundary(poly: Polygon, n: int) -> List[Tuple[float, float]]:
    if poly is None or poly.is_empty:
        return []
    boundary = poly.boundary
    if boundary.is_empty or boundary.length <= 0:
        return []
    length = boundary.length
    pts = []
    for i in range(n):
        p = boundary.interpolate((i / max(1, n)) * length)
        pts.append((float(p.x), float(p.y)))
    return pts


def _sample_dtm_z(dtm, nodata: Optional[float], x: float, y: float) -> Optional[float]:
    if dtm is None:
        return None
    try:
        val = next(dtm.sample([(float(x), float(y))]))
    except Exception:
        return None
    if val is None or len(val) == 0:
        return None
    z = float(val[0])
    if nodata is not None and np.isfinite(nodata):
        if abs(z - float(nodata)) < 1e-6:
            return None
    if not np.isfinite(z):
        return None
    return z


def _overlay_frame(
    image_path: Path,
    raw_poly: Polygon,
    proj_pts: List[Tuple[float, float]],
    out_path: Path,
) -> None:
    base = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")
    if raw_poly is not None and not raw_poly.is_empty:
        if raw_poly.geom_type == "Polygon":
            rings = [raw_poly.exterior.coords]
        else:
            rings = [g.exterior.coords for g in raw_poly.geoms]
        for ring in rings:
            pts = [(float(x), float(y)) for x, y in ring]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=(255, 0, 0, 220), width=2)
    if proj_pts:
        draw.line(proj_pts + [proj_pts[0]], fill=(0, 255, 0, 220), width=2)
    base.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_DEFAULT))
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--overlays", nargs="*", default=["250", "341", "500"])
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if not cfg:
        raise SystemExit(f"missing config: {cfg_path}")
    drive_id = str(cfg.get("drive_id") or "")
    frame_start = int(cfg.get("frame_start", 250))
    frame_end = int(cfg.get("frame_end", 500))
    kitti_root = Path(str(cfg.get("kitti_root") or ""))
    camera = str(cfg.get("camera") or "image_00")
    image_run = Path(str(cfg.get("image_run") or ""))
    image_provider = str(cfg.get("image_provider") or "grounded_sam2_v1")
    if not drive_id or not kitti_root.exists():
        raise SystemExit("missing drive_id or kitti_root")

    run_id = now_ts()
    run_dir = Path("runs") / f"image_crosswalk_proj_regress_0010_250_500_{run_id}"
    ensure_overwrite(run_dir)
    setup_logging(run_dir / "run.log")

    source_run = Path(args.run_dir) if args.run_dir else _find_latest_crosswalk_run()
    if source_run is None or not source_run.exists():
        raise SystemExit("crosswalk_run_not_found")

    feature_store_root = image_run / f"feature_store_{image_provider}"
    if not feature_store_root.exists():
        raise SystemExit("feature_store_missing")

    back_cfg = cfg.get("backproject", {}) if isinstance(cfg.get("backproject"), dict) else {}
    dtm_path_cfg = str(back_cfg.get("dtm_path") or "").strip()
    dtm_path = Path(dtm_path_cfg) if dtm_path_cfg else None
    if dtm_path is None:
        for cand in sorted(Path("runs").glob("lidar_ground_0010_f250_500_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            cand_path = cand / "rasters" / "dtm_median_utm32.tif"
            if cand_path.exists():
                dtm_path = cand_path
                break
    dtm = None
    nodata = None
    if dtm_path is not None:
        try:
            import rasterio

            dtm = rasterio.open(dtm_path)
            nodata = dtm.nodata
        except Exception:
            dtm = None
            nodata = None
    fixed_z0 = float(back_cfg.get("fixed_plane_z0", 0.0))
    proj_ctx = configure_default_context(kitti_root, drive_id, cam_id=camera, dtm_path=dtm_path, frame_id_for_size=f"{frame_start:010d}")

    tables_dir = run_dir / "tables"
    overlays_dir = run_dir / "overlays"
    tables_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_dists: List[float] = []
    used_frames = 0
    missing = 0
    for frame in range(frame_start, frame_end + 1):
        frame_id = f"{frame:010d}"
        world_poly = _load_world_poly_from_run(source_run, frame_id)
        raw_poly = _load_raw_pixel_poly(feature_store_root, drive_id, frame_id)
        if world_poly is None or raw_poly is None:
            missing += 1
            continue
        pts_world = _sample_boundary(world_poly, 200)
        if not pts_world:
            missing += 1
            continue
        zs = []
        for x, y in pts_world:
            z = _sample_dtm_z(dtm, nodata, x, y)
            zs.append(z if z is not None else fixed_z0)
        xyz = np.column_stack([np.array([p[0] for p in pts_world]), np.array([p[1] for p in pts_world]), np.array(zs)])
        u, v, valid = world_to_pixel_cam0(frame_id, xyz, ctx=proj_ctx)
        raw_boundary = raw_poly.boundary
        dists = []
        for uu, vv, ok in zip(u, v, valid):
            if not ok:
                continue
            d = raw_boundary.distance(Point(float(uu), float(vv)))
            dists.append(float(d))
        if not dists:
            missing += 1
            continue
        used_frames += 1
        all_dists.extend(dists)
        rows.append(
            {
                "frame_id": frame_id,
                "n_points": len(dists),
                "p50_px": float(np.percentile(dists, 50)),
                "p90_px": float(np.percentile(dists, 90)),
                "mean_px": float(np.mean(dists)),
            }
        )

        if frame_id in {f"{int(x):010d}" for x in args.overlays}:
            img_path = _find_image_path(kitti_root, drive_id, camera, frame_id)
            if img_path and img_path.exists():
                proj_pts = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
                _overlay_frame(img_path, raw_poly, proj_pts, overlays_dir / f"frame_{frame_id}.png")

    tables_path = tables_dir / "roundtrip_px_errors.csv"
    if rows:
        import pandas as pd

        pd.DataFrame(rows).to_csv(tables_path, index=False)
    else:
        tables_path.write_text("frame_id,n_points,p50_px,p90_px,mean_px\n", encoding="utf-8")

    if not all_dists:
        status = "FAIL"
        p90 = None
    else:
        p90 = float(np.percentile(all_dists, 90))
        if p90 <= 5.0:
            status = "PASS"
        elif p90 <= 15.0:
            status = "WARN"
        else:
            status = "FAIL"

    decision = {
        "status": status,
        "roundtrip_p90_px": p90,
        "frames_used": used_frames,
        "frames_missing": missing,
        "source_run": str(source_run),
    }
    write_json(run_dir / "decision.json", decision)

    report_lines = [
        "# Image crosswalk projection regression (0010 f250-500)",
        "",
        f"- status: {status}",
        f"- roundtrip_p90_px: {p90 if p90 is not None else 'NA'}",
        f"- frames_used: {used_frames}",
        f"- frames_missing: {missing}",
        f"- source_run: {source_run}",
        "",
        "## outputs",
        f"- tables/roundtrip_px_errors.csv",
        f"- overlays/frame_000250.png",
        f"- overlays/frame_000341.png",
        f"- overlays/frame_000500.png",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
