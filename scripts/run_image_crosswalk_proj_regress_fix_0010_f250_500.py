from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from pipeline.calib.kitti360_backproject import (
    configure_default_context,
    pixel_to_world_on_ground,
    world_to_pixel_cam0,
)
from pipeline.calib.kitti360_world import kitti_world_to_utm32, utm32_to_kitti_world, utm32_to_wk
from pipeline._io import load_yaml
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_json, write_text

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


def _load_feature_store_poly(feature_store_root: Path, drive_id: str, frame_id: str) -> Tuple[Optional[Polygon], Optional[str], Optional[str]]:
    gpkg_path = feature_store_root / "feature_store" / drive_id / frame_id / "image_features.gpkg"
    if not gpkg_path.exists():
        return None, None, "not_generated"
    layer = _find_crosswalk_layer(gpkg_path)
    if not layer:
        return None, str(gpkg_path), "wrong_layer"
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except Exception:
        return None, str(gpkg_path), "wrong_layer"
    if gdf.empty:
        return None, str(gpkg_path), "empty_geom"
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
    if geom.is_empty:
        return None, str(gpkg_path), "empty_geom"
    if geom.geom_type == "Polygon":
        return geom, str(gpkg_path), None
    if geom.geom_type == "MultiPolygon":
        return unary_union(list(geom.geoms)), str(gpkg_path), None
    return None, str(gpkg_path), "empty_geom"


def _mask_paths_for_frame(outputs_dir: Path, frame_id: str) -> List[Path]:
    mask_root = outputs_dir / "stage2_masks"
    if not mask_root.exists():
        return []
    return sorted(mask_root.rglob(f"{frame_id}.*"))


def _poly_from_mask(path: Path) -> Optional[Polygon]:
    try:
        import rasterio.features
        from affine import Affine
    except Exception:
        return None
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    if arr.size == 0:
        return None
    binary = (arr > 0).astype(np.uint8)
    if binary.sum() == 0:
        return None
    shapes = []
    for geom, val in rasterio.features.shapes(binary, transform=Affine.identity()):
        if int(val) != 1:
            continue
        try:
            poly = Polygon(geom["coordinates"][0])
        except Exception:
            continue
        if not poly.is_empty:
            shapes.append(poly)
    if not shapes:
        return None
    return unary_union(shapes)


def _load_mask_poly(outputs_dir: Path, frame_id: str) -> Tuple[Optional[Polygon], Optional[str], Optional[str]]:
    paths = _mask_paths_for_frame(outputs_dir, frame_id)
    if not paths:
        return None, None, "not_generated"
    polys = []
    for p in paths:
        poly = _poly_from_mask(p)
        if poly is not None and not poly.is_empty:
            polys.append(poly)
    if not polys:
        return None, str(paths[0]), "empty_geom"
    return unary_union(polys), str(paths[0]), None


def _load_pixel_poly(feature_store_root: Path, outputs_dir: Path, drive_id: str, frame_id: str) -> Tuple[Optional[Polygon], Optional[str], Optional[str], str]:
    poly, path, reason = _load_mask_poly(outputs_dir, frame_id)
    if poly is not None:
        return poly, path, reason, "stage2_mask"
    poly, path, reason = _load_feature_store_poly(feature_store_root, drive_id, frame_id)
    if poly is not None:
        return poly, path, reason, "feature_store"
    return None, path, reason, "missing"


def _load_world_poly_from_run(
    run_dir: Path, frame_id: str
) -> Tuple[Optional[Polygon], Optional[str], Optional[str], Optional[str]]:
    candidates = [
        run_dir / "outputs" / "frames" / frame_id / "world_crosswalk_utm32.gpkg",
        run_dir / "outputs" / "frames" / frame_id / "world_crosswalk_wk.gpkg",
        run_dir / "outputs" / "frames" / frame_id / "world_crosswalk.gpkg",
    ]
    frame_path = next((p for p in candidates if p.exists()), None)
    if frame_path is not None:
        for layer in [
            "crosswalk_frame_raw",
            "crosswalk_frame_canonical_v3",
            "crosswalk_frame_raw_wk",
            "crosswalk_frame_raw_utm32",
            "crosswalk_frame_canonical_wk",
            "crosswalk_frame_canonical_utm32",
        ]:
            try:
                gdf = gpd.read_file(frame_path, layer=layer)
            except Exception:
                continue
            if gdf.empty:
                continue
            crs = str(gdf.crs) if gdf.crs else None
            geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                return geom, str(frame_path), None, crs
            if geom.geom_type == "MultiPolygon":
                return unary_union(list(geom.geoms)), str(frame_path), None, crs
        return None, str(frame_path), "empty_geom", None
    fallback = run_dir / "outputs" / "frame_candidates_utm32.gpkg"
    if fallback.exists():
        try:
            gdf = gpd.read_file(fallback, layer="frame_candidates")
        except Exception:
            return None, str(fallback), "wrong_layer", None
        if not gdf.empty and "frame_id" in gdf.columns:
            gdf = gdf[gdf["frame_id"].astype(str) == frame_id]
            if not gdf.empty:
                crs = str(gdf.crs) if gdf.crs else None
                geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
                if geom is None or geom.is_empty:
                    return None, str(fallback), "empty_geom", crs
                if geom.geom_type == "Polygon":
                    return geom, str(fallback), "wrong_path", crs
                if geom.geom_type == "MultiPolygon":
                    return unary_union(list(geom.geoms)), str(fallback), "wrong_path", crs
    return None, None, "not_generated", None


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


def _overlay_frame(
    image_path: Optional[Path],
    raw_poly: Optional[Polygon],
    proj_pts: List[Tuple[float, float]],
    out_path: Path,
    note: str,
    stats: Optional[Dict[str, float]] = None,
) -> None:
    if image_path and image_path.exists():
        base = Image.open(image_path).convert("RGBA")
    else:
        base = Image.new("RGBA", (1242, 375), (255, 255, 255, 255))
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
    text = note
    if stats:
        text = (
            f"{note} | mean={stats.get('mean_px', 0):.2f} p90={stats.get('p90_px', 0):.2f}"
            f" n={int(stats.get('n_samples', 0))} vratio={stats.get('valid_ratio', 0):.2f}"
        )
    draw.rectangle([6, 6, 600, 28], fill=(255, 255, 255, 200))
    draw.text((10, 8), text, fill=(0, 0, 0, 255))
    base.save(out_path)


def _distance_to_boundary(poly: Polygon, pts: Iterable[Tuple[float, float]]) -> List[float]:
    boundary = poly.boundary
    dists = []
    for x, y in pts:
        dists.append(float(boundary.distance(Point(float(x), float(y)))))
    return dists


def _sample_points_in_poly(poly: Polygon, n: int, rng: np.random.Generator) -> List[Tuple[float, float]]:
    if poly is None or poly.is_empty:
        return []
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 50:
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        tries += 1
        if poly.contains(Point(x, y)):
            pts.append((x, y))
    return pts


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
    run_dir = Path("runs") / f"image_crosswalk_proj_regress_fix_0010_250_500_{run_id}"
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
    if dtm is not None:
        try:
            import numpy as np

            band = dtm.read(1, masked=True)
            vals = np.asarray(band.compressed(), dtype=float)
            if vals.size > 0:
                fixed_z0 = float(np.median(vals))
        except Exception:
            pass
    ground_model = {
        "mode": "lidar_clean_dtm" if dtm is not None else "fixed_plane",
        "dtm_path": str(dtm_path) if dtm_path is not None else None,
        "z0": fixed_z0,
    }
    proj_ctx = configure_default_context(kitti_root, drive_id, cam_id=camera, dtm_path=dtm_path, frame_id_for_size=f"{frame_start:010d}")

    tables_dir = run_dir / "tables"
    overlays_dir = run_dir / "overlays"
    images_dir = run_dir / "images"
    tables_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = []
    missing_rows = []

    all_dists: List[float] = []
    used_frames = 0
    invalid_frames = 0
    overlay_set = {f"{int(x):010d}" for x in args.overlays}

    roundtrip_rows = []
    per_frame_stats: Dict[str, Dict[str, float]] = {}

    for frame in range(frame_start, frame_end + 1):
        frame_id = f"{frame:010d}"
        world_poly, world_path, world_reason, world_crs = _load_world_poly_from_run(source_run, frame_id)
        pixel_poly, pixel_path, pixel_reason, pixel_source = _load_pixel_poly(feature_store_root, source_run / "outputs", drive_id, frame_id)
        mask_paths = _mask_paths_for_frame(source_run / "outputs", frame_id)

        if world_poly is not None and world_crs:
            if "32632" in world_crs or "EPSG:32632" in world_crs.upper():
                geom_list = list(world_poly.geoms) if hasattr(world_poly, "geoms") else [world_poly]
                polys = []
                for g in geom_list:
                    coords = list(g.exterior.coords) if hasattr(g, "exterior") else []
                    if not coords:
                        continue
                    pts_wu = np.array([[float(x), float(y), 0.0] for x, y in coords], dtype=np.float64)
                    pts_wk = utm32_to_wk(pts_wu, kitti_root, drive_id, frame_id)
                    try:
                        polys.append(Polygon([(float(p[0]), float(p[1])) for p in pts_wk]))
                    except Exception:
                        continue
                if polys:
                    world_poly = unary_union(polys)

        has_world = world_poly is not None and not world_poly.is_empty
        has_pixel = pixel_poly is not None and not pixel_poly.is_empty
        inventory_rows.append(
            {
                "frame_id": frame_id,
                "has_mask": bool(mask_paths),
                "has_pixel_poly": has_pixel,
                "pixel_poly_source": pixel_source,
                "pixel_poly_path": pixel_path or "",
                "has_world_poly": has_world,
                "world_poly_path": world_path or "",
            }
        )

        reason_world = world_reason if not has_world else "ok"
        reason_pixel = pixel_reason if not has_pixel else "ok"
        reason_overall = "ok" if has_world and has_pixel else "missing"
        missing_rows.append(
            {
                "frame_id": frame_id,
                "reason_world": reason_world,
                "reason_pixel": reason_pixel,
                "reason_overall": reason_overall,
            }
        )

        if not has_world or not has_pixel:
            if frame_id in overlay_set:
                img_path = _find_image_path(kitti_root, drive_id, camera, frame_id)
                note = f"{frame_id} missing"
                _overlay_frame(img_path, pixel_poly, [], overlays_dir / f"frame_{frame_id}.png", note, None)
            continue

        pts_world = _sample_boundary(world_poly, 200)
        if not pts_world:
            invalid_frames += 1
            roundtrip_rows.append(
                {
                    "frame_id": frame_id,
                    "n_samples": 0,
                    "valid_ratio": 0.0,
                    "p50_px": "",
                    "p90_px": "",
                    "max_px": "",
                    "reason_if_invalid": "empty_world_boundary",
                }
            )
            continue

        world_is_utm = False
        if world_crs:
            world_is_utm = "32632" in world_crs or "EPSG:32632" in world_crs.upper()
        zs = []
        if world_is_utm:
            pts_wu = []
            for x, y in pts_world:
                z = None
                if dtm is not None:
                    try:
                        z = float(next(dtm.sample([(float(x), float(y))]))[0])
                        if nodata is not None and np.isnan(nodata):
                            if not np.isfinite(z):
                                z = None
                        elif nodata is not None and abs(float(z) - float(nodata)) < 1e-6:
                            z = None
                        if z is not None and not np.isfinite(z):
                            z = None
                    except Exception:
                        z = None
                z = z if z is not None else fixed_z0
                pts_wu.append([float(x), float(y), float(z)])
                zs.append(z)
            pts_wu_np = np.array(pts_wu, dtype=np.float64)
            xyz = utm32_to_kitti_world(pts_wu_np, kitti_root, drive_id, frame_id)
        else:
            pts_wk = []
            for x, y in pts_world:
                z_seed = fixed_z0
                if dtm is not None:
                    try:
                        pts_wu = kitti_world_to_utm32(
                            np.array([[float(x), float(y), float(z_seed)]], dtype=np.float64),
                            kitti_root,
                            drive_id,
                            frame_id,
                        )
                        xu, yu = float(pts_wu[0, 0]), float(pts_wu[0, 1])
                        z = float(next(dtm.sample([(xu, yu)]))[0])
                        if nodata is not None and np.isnan(nodata):
                            if not np.isfinite(z):
                                z = None
                        elif nodata is not None and abs(float(z) - float(nodata)) < 1e-6:
                            z = None
                        if z is not None and not np.isfinite(z):
                            z = None
                    except Exception:
                        z = None
                    if z is None:
                        z = fixed_z0
                else:
                    z = fixed_z0
                pts_wk.append([float(x), float(y), float(z)])
                zs.append(z)
            xyz = np.array(pts_wk, dtype=np.float64)
        u, v, valid = world_to_pixel_cam0(frame_id, xyz, ctx=proj_ctx)
        n_valid = int(np.count_nonzero(valid))
        valid_ratio = n_valid / max(1, len(valid))
        if valid_ratio < 0.3:
            invalid_frames += 1
            roundtrip_rows.append(
                {
                    "frame_id": frame_id,
                    "n_samples": len(valid),
                    "valid_ratio": valid_ratio,
                    "p50_px": "",
                    "p90_px": "",
                    "max_px": "",
                    "reason_if_invalid": "low_valid_ratio",
                }
            )
            if frame_id in overlay_set:
                img_path = _find_image_path(kitti_root, drive_id, camera, frame_id)
                proj_pts = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
                _overlay_frame(img_path, pixel_poly, proj_pts, overlays_dir / f"frame_{frame_id}.png", "low_valid_ratio", None)
            continue

        proj_pts = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
        dists = _distance_to_boundary(pixel_poly, proj_pts)
        if not dists:
            invalid_frames += 1
            roundtrip_rows.append(
                {
                    "frame_id": frame_id,
                    "n_samples": len(valid),
                    "valid_ratio": valid_ratio,
                    "p50_px": "",
                    "p90_px": "",
                    "max_px": "",
                    "reason_if_invalid": "no_distance",
                }
            )
            continue

        used_frames += 1
        all_dists.extend(dists)
        stats = {
            "mean_px": float(np.mean(dists)),
            "p50_px": float(np.percentile(dists, 50)),
            "p90_px": float(np.percentile(dists, 90)),
            "max_px": float(np.max(dists)),
            "n_samples": float(len(dists)),
            "valid_ratio": float(valid_ratio),
        }
        per_frame_stats[frame_id] = stats
        roundtrip_rows.append(
            {
                "frame_id": frame_id,
                "n_samples": len(dists),
                "valid_ratio": valid_ratio,
                "p50_px": stats["p50_px"],
                "p90_px": stats["p90_px"],
                "max_px": stats["max_px"],
                "reason_if_invalid": "",
            }
        )

        if frame_id in overlay_set:
            img_path = _find_image_path(kitti_root, drive_id, camera, frame_id)
            _overlay_frame(img_path, pixel_poly, proj_pts, overlays_dir / f"frame_{frame_id}.png", "ok", stats)

    import pandas as pd

    pd.DataFrame(inventory_rows).to_csv(tables_dir / "frames_inventory.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(tables_dir / "missing_reason.csv", index=False)
    pd.DataFrame(roundtrip_rows).to_csv(tables_dir / "roundtrip_px_errors.csv", index=False)

    if not all_dists or used_frames < 1:
        status = "FAIL"
        p90 = None
    else:
        p90 = float(np.percentile(np.array(all_dists, dtype=np.float64), 90))
        overlay_ok = 0
        for key in [f"{int(x):010d}" for x in ["250", "341", "500"]]:
            if key in per_frame_stats and per_frame_stats[key]["p90_px"] <= 8.0:
                overlay_ok += 1
        if p90 <= 5.0 and used_frames >= 20 and overlay_ok >= 2:
            status = "PASS"
        elif p90 <= 15.0 and used_frames >= 20:
            status = "WARN"
        else:
            status = "FAIL"

    invalid_reason_counts = {}
    for row in roundtrip_rows:
        reason = row.get("reason_if_invalid") or ""
        if reason:
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1
    decision = {
        "status": status,
        "roundtrip_p90_px": p90,
        "frames_used": used_frames,
        "frames_invalid": invalid_frames,
        "invalid_reasons_top": dict(sorted(invalid_reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]),
        "source_run": str(source_run),
    }
    write_json(run_dir / "decision.json", decision)

    reason_counts = pd.DataFrame(missing_rows).groupby(["reason_world", "reason_pixel"]).size().reset_index(name="count")
    reason_counts.to_csv(tables_dir / "missing_reason_summary.csv", index=False)

    report_lines = [
        "# Image crosswalk projection regression fix (0010 f250-500)",
        "",
        f"- status: {status}",
        f"- roundtrip_p90_px: {p90 if p90 is not None else 'NA'}",
        f"- frames_used: {used_frames}",
        f"- frames_invalid: {invalid_frames}",
        f"- source_run: {source_run}",
        f"- ground_model: {ground_model.get('mode')}",
        f"- r_rect_00: {'yes' if proj_ctx.calib.r_rect_00 is not None else 'no'}",
        "",
        "## outputs",
        "- tables/frames_inventory.csv",
        "- tables/missing_reason.csv",
        "- tables/missing_reason_summary.csv",
        "- tables/roundtrip_px_errors.csv",
        "- overlays/frame_0000000250.png",
        "- overlays/frame_0000000341.png",
        "- overlays/frame_0000000500.png",
    ]

    if p90 is not None and p90 > 15.0 and used_frames >= 20:
        rng = np.random.default_rng(42)
        cycle_rows = []
        valid_frames = [row["frame_id"] for row in roundtrip_rows if row.get("reason_if_invalid") == ""]
        if not valid_frames:
            valid_frames = []
        frames_for_cycle = [f"{int(x):010d}" for x in ["250", "341", "500"] if f"{int(x):010d}" in valid_frames]
        if not frames_for_cycle:
            frames_for_cycle = valid_frames[:3]
        for frame_id in frames_for_cycle:
            pixel_poly, _, _, _ = _load_pixel_poly(feature_store_root, source_run / "outputs", drive_id, frame_id)
            if pixel_poly is None:
                continue
            samples = _sample_points_in_poly(pixel_poly, 500, rng)
            if not samples:
                continue
            errs = []
            for u0, v0 in samples:
                xyz = pixel_to_world_on_ground(frame_id, u0, v0, ground_model, ctx=proj_ctx)
                if xyz is None:
                    continue
                u1, v1, ok = world_to_pixel_cam0(frame_id, np.array([xyz]), ctx=proj_ctx)
                if len(u1) == 0 or not bool(ok[0]):
                    continue
                err = float(np.hypot(float(u1[0]) - float(u0), float(v1[0]) - float(v0)))
                errs.append(err)
            if not errs:
                continue
            cycle_rows.append(
                {
                    "frame_id": frame_id,
                    "n_samples": len(errs),
                    "p50_px": float(np.percentile(errs, 50)),
                    "p90_px": float(np.percentile(errs, 90)),
                    "max_px": float(np.max(errs)),
                }
            )

        if cycle_rows:
            pd.DataFrame(cycle_rows).to_csv(tables_dir / "pixel_world_pixel_cycle.csv", index=False)
            try:
                import matplotlib.pyplot as plt

                all_cycle = []
                for row in cycle_rows:
                    all_cycle.append(row["p90_px"])
                plt.figure(figsize=(6, 3))
                plt.hist(all_cycle, bins=20, color="#4c78a8", alpha=0.85)
                plt.title("pixel->world->pixel p90 error")
                plt.xlabel("px")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(images_dir / "pixel_cycle_error_hist.png", dpi=150)
                plt.close()
                report_lines.extend(
                    [
                        "",
                        "## pixel cycle diagnostics",
                        "- tables/pixel_world_pixel_cycle.csv",
                        "- images/pixel_cycle_error_hist.png",
                    ]
                )
            except Exception:
                report_lines.extend(
                    [
                        "",
                        "## pixel cycle diagnostics",
                        "- tables/pixel_world_pixel_cycle.csv",
                        "- images/pixel_cycle_error_hist.png (skipped)",
                    ]
                )

    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
