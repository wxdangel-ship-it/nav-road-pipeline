from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import load_kitti360_calib, load_kitti360_pose
from tools.build_image_sample_index import _extract_frame_id, _find_image_dir, _list_images
from tools.build_road_entities import _geom_to_image_points


LOG = logging.getLogger("run_crosswalk_monitor_drive")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_monitor_drive")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _merge_config(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for key, val in overrides.items():
        if val is None:
            continue
        out[key] = val
    return out


def _normalize_frame_id(frame_id: str) -> str:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    if not digits:
        return str(frame_id)
    return digits.zfill(10)


def _build_index_records(
    kitti_root: Path,
    drive_id: str,
    camera: str,
) -> List[dict]:
    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    if not image_dir:
        return []
    records: List[dict] = []
    for path in _list_images(image_dir):
        frame_id = _normalize_frame_id(_extract_frame_id(path))
        records.append(
            {
                "drive_id": drive_id,
                "camera": camera,
                "frame_id": frame_id,
                "image_path": str(path),
                "scene_profile": "car",
            }
        )
    return records


def _write_index(records: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _list_layers(path: Path) -> List[str]:
    try:
        import pyogrio

        return [name for name, _ in pyogrio.list_layers(path)]
    except Exception:
        pass
    try:
        return list(gpd.io.file.fiona.listlayers(str(path)))
    except Exception:
        return []


def _find_crosswalk_layer(path: Path) -> str:
    if not path.exists():
        return ""
    for name in _list_layers(path):
        if "crosswalk" in name.lower():
            return name
    return ""


def _load_crosswalk_raw(
    feature_store_root: Path,
    drive_id: str,
    frame_id: str,
    cache: Dict[Tuple[str, str], Tuple[gpd.GeoDataFrame, float, str]],
) -> Tuple[gpd.GeoDataFrame, float, str]:
    key = (drive_id, frame_id)
    if key in cache:
        return cache[key]
    gpkg_path = feature_store_root / "feature_store" / drive_id / frame_id / "image_features.gpkg"
    if not gpkg_path.exists():
        cache[key] = (gpd.GeoDataFrame(), 0.0, "missing_feature_store")
        return cache[key]
    layer = _find_crosswalk_layer(gpkg_path)
    if not layer:
        cache[key] = (gpd.GeoDataFrame(), 0.0, "missing_layer")
        return cache[key]
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except Exception:
        gdf = gpd.GeoDataFrame()
    score = 0.0
    if not gdf.empty:
        for col in ("conf", "score", "confidence"):
            if col in gdf.columns:
                vals = []
                for v in gdf[col].tolist():
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if np.isnan(fv):
                        continue
                    vals.append(fv)
                if vals:
                    score = max(vals)
                    break
    status = "ok" if not gdf.empty else "no_crosswalk"
    cache[key] = (gdf, score, status)
    return cache[key]


def _render_text_overlay(
    image_path: str,
    out_path: Path,
    lines: List[str],
) -> None:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(255, 128, 0, 220))
        y += 26
    base.save(out_path)


def _render_raw_overlay(
    image_path: str,
    raw_gdf: gpd.GeoDataFrame,
    out_path: Path,
    missing_status: str,
    raw_fallback_text: bool,
) -> None:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    if raw_fallback_text and missing_status and missing_status != "ok":
        draw.text((10, 10), f"MISSING_FEATURE_STORE:{missing_status}", fill=(255, 128, 0, 220))
        draw.text((10, 36), "NO_RAW_PRED", fill=(255, 0, 0, 220))
        base.save(out_path)
        return
    if raw_gdf is None or raw_gdf.empty:
        if missing_status and missing_status != "ok":
            draw.text((10, 10), f"MISSING_FEATURE_STORE:{missing_status}", fill=(255, 128, 0, 220))
        draw.text((10, 36), "NO_CROSSWALK_DETECTED", fill=(255, 0, 0, 220))
        base.save(out_path)
        return
    for _, row in raw_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            coords = [(float(x), float(y)) for x, y in geom.exterior.coords]
            if len(coords) >= 2:
                draw.line(coords, fill=(255, 0, 0, 220), width=3)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
                if len(coords) >= 2:
                    draw.line(coords, fill=(255, 0, 0, 220), width=3)
        elif geom.geom_type == "LineString":
            coords = [(float(x), float(y)) for x, y in geom.coords]
            if len(coords) >= 2:
                draw.line(coords, fill=(255, 0, 0, 220), width=3)
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords = [(float(x), float(y)) for x, y in line.coords]
                if len(coords) >= 2:
                    draw.line(coords, fill=(255, 0, 0, 220), width=3)
    base.save(out_path)


def _ensure_raw_overlays(
    qa_index_path: Path,
    outputs_dir: Path,
    image_run: Path,
    provider_id: str,
    index_lookup: Dict[Tuple[str, str], str],
    raw_fallback_text: bool,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    if not qa_index_path.exists():
        return {}
    qa_gdf = gpd.read_file(qa_index_path)
    if qa_gdf.empty:
        return {}
    feature_store_root = image_run / f"feature_store_{provider_id}"
    qa_dir = outputs_dir / "qa_images"
    qa_dir.mkdir(parents=True, exist_ok=True)
    raw_cache: Dict[Tuple[str, str], Tuple[gpd.GeoDataFrame, float, str]] = {}
    raw_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    missing_rows = []
    for idx, row in qa_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not drive_id or not frame_id:
            continue
        image_path = index_lookup.get((drive_id, frame_id), "")
        out_path = qa_dir / drive_id / f"{frame_id}_overlay_raw.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_gdf, raw_score, raw_status = _load_crosswalk_raw(feature_store_root, drive_id, frame_id, raw_cache)
        raw_stats[(drive_id, frame_id)] = {
            "raw_has_crosswalk": 0.0 if raw_gdf.empty else 1.0,
            "raw_top_score": float(raw_score),
            "raw_status": raw_status,
        }
        qa_gdf.at[idx, "raw_has_crosswalk"] = int(0 if raw_gdf.empty else 1)
        qa_gdf.at[idx, "raw_top_score"] = float(raw_score)
        qa_gdf.at[idx, "raw_status"] = raw_status
        if raw_status == "missing_feature_store":
            missing_rows.append(
                {
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "raw_status": raw_status,
                    "image_path": image_path,
                }
            )
        _render_raw_overlay(image_path, raw_gdf, out_path, raw_status, raw_fallback_text)
        qa_gdf.at[idx, "overlay_raw_path"] = str(out_path)
    qa_index_path.write_text(qa_gdf.to_json(), encoding="utf-8")
    missing_path = outputs_dir / "missing_feature_store_list.csv"
    import pandas as pd

    pd.DataFrame(
        missing_rows,
        columns=["drive_id", "frame_id", "raw_status", "image_path"],
    ).to_csv(missing_path, index=False)
    return raw_stats


def _read_candidates(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(path, layer="crosswalk_candidate_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _read_final(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(path, layer="crosswalk_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _ensure_wgs84_range(gdf: gpd.GeoDataFrame) -> bool:
    if gdf.empty:
        return True
    try:
        bounds = gdf.total_bounds
    except Exception:
        return False
    minx, miny, maxx, maxy = bounds
    return -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0


def _write_crosswalk_gpkg(src_gpkg: Path, out_gpkg: Path) -> None:
    if out_gpkg.exists():
        out_gpkg.unlink()
    for layer in ("crosswalk_candidate_poly", "crosswalk_poly"):
        try:
            gdf = gpd.read_file(src_gpkg, layer=layer)
        except Exception:
            gdf = gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")


def _build_final_support(final_gdf: gpd.GeoDataFrame) -> Dict[Tuple[str, str], List[str]]:
    support_map: Dict[Tuple[str, str], List[str]] = {}
    if final_gdf.empty:
        return support_map
    for _, row in final_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frames_raw = row.get("support_frames", "")
        frames = []
        if isinstance(frames_raw, str) and frames_raw:
            try:
                frames = json.loads(frames_raw)
            except Exception:
                frames = []
        for frame_id in frames:
            support_map.setdefault((drive_id, str(frame_id)), []).append(row.get("entity_id"))
    return support_map


def _project_point_to_image(
    x: float,
    y: float,
    pose_xy_yaw: Tuple[float, float, float],
    calib: Dict[str, np.ndarray],
) -> Tuple[float, float, bool]:
    x0, y0, yaw = pose_xy_yaw
    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    dx = x - x0
    dy = y - y0
    x_ego = c * dx - s * dy
    y_ego = s * dx + c * dy
    z_ego = 0.0
    pts_h = np.array([[x_ego], [y_ego], [z_ego], [1.0]])
    cam = calib["t_velo_to_cam"] @ pts_h
    proj = calib["p_rect"] @ np.vstack([cam[:3, :], np.ones((1, cam.shape[1]))])
    z = float(proj[2, 0])
    if z <= 1e-3:
        return 0.0, 0.0, False
    u = float(proj[0, 0] / z)
    v = float(proj[1, 0] / z)
    return u, v, True


def _render_gated_overlay(
    out_path: Path,
    image_path: str,
    frame_id: str,
    candidates: gpd.GeoDataFrame,
    pose: Tuple[float, float, float] | None,
    calib: Dict[str, np.ndarray] | None,
    draw_rejected: bool,
) -> Tuple[int, List[str]]:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    if pose is None or calib is None:
        _render_text_overlay(image_path, out_path, ["NO_CALIB_OR_POSE"])
        return 0, []

    kept = 0
    reject_reasons: List[str] = []
    for _, row in candidates.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        reasons = str(row.get("reject_reasons") or "")
        is_rejected = bool(reasons)
        if is_rejected and not draw_rejected:
            continue
        pts = _geom_to_image_points(geom, pose, calib)
        if len(pts) < 2:
            continue
        color = (0, 255, 255, 200) if not is_rejected else (160, 160, 160, 200)
        draw.polygon(pts, outline=color)
        if is_rejected:
            reject_reasons.extend([r for r in reasons.split(",") if r])
            cx, cy = geom.centroid.x, geom.centroid.y
            u, v, ok = _project_point_to_image(cx, cy, pose, calib)
            if ok:
                draw.text((u, v), reasons.replace(",", "|"), fill=(180, 180, 180, 220))
        else:
            kept += 1

    if len(candidates) == 0:
        draw.text((10, 10), "NO_GATED_CANDIDATE", fill=(255, 128, 0, 220))
    elif kept == 0:
        top_reject = sorted(set(reject_reasons))
        draw.text((10, 10), "NO_GATED_CANDIDATE", fill=(255, 128, 0, 220))
        if top_reject:
            draw.text((10, 36), "REJECT_REASONS:" + "|".join(top_reject[:5]), fill=(200, 200, 200, 220))
    base.save(out_path)
    return kept, sorted(set(reject_reasons))


def _render_entities_overlay(
    out_path: Path,
    image_path: str,
    drive_id: str,
    frame_id: str,
    final_gdf: gpd.GeoDataFrame,
    final_support: Dict[Tuple[str, str], List[str]],
    raw_has_crosswalk: int,
    pose: Tuple[float, float, float] | None,
    calib: Dict[str, np.ndarray] | None,
) -> None:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    if pose is None or calib is None:
        _render_text_overlay(image_path, out_path, ["NO_CALIB_OR_POSE"])
        return

    support_ids = final_support.get((drive_id, frame_id), [])
    if not support_ids:
        tag = "RAW_HAS_CROSSWALK=1" if raw_has_crosswalk == 1 else "RAW_HAS_CROSSWALK=0"
        draw.text((10, 10), "NO_FINAL_ENTITY", fill=(255, 128, 0, 220))
        draw.text((10, 36), tag, fill=(255, 200, 0, 220))
        base.save(out_path)
        return

    for _, row in final_gdf.iterrows():
        if str(row.get("drive_id") or "") != drive_id:
            continue
        frames_raw = row.get("support_frames", "")
        frames = []
        if isinstance(frames_raw, str) and frames_raw:
            try:
                frames = json.loads(frames_raw)
            except Exception:
                frames = []
        if frame_id not in {str(f) for f in frames}:
            continue
        geom = row.geometry
        pts = _geom_to_image_points(geom, pose, calib)
        if len(pts) < 2:
            continue
        draw.polygon(pts, outline=(0, 128, 255, 220), fill=(0, 128, 255, 80))
    base.save(out_path)


def _update_qa_index(
    qa_index_path: Path,
    index_lookup: Dict[Tuple[str, str], str],
    final_support: Dict[Tuple[str, str], List[str]],
    candidate_ids: Dict[Tuple[str, str], List[str]],
) -> None:
    if not qa_index_path.exists():
        return
    qa = gpd.read_file(qa_index_path)
    overlay_raw = []
    overlay_gated = []
    overlay_entities = []
    cand_ids = []
    final_ids = []
    for idx, row in qa.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        overlay_raw.append(str(row.get("overlay_raw_path") or ""))
        overlay_gated.append(str(row.get("overlay_gated_path") or ""))
        overlay_entities.append(str(row.get("overlay_entities_path") or ""))
        cand_ids.append(json.dumps(sorted(set(candidate_ids.get((drive_id, frame_id), []))), ensure_ascii=True))
        final_ids.append(json.dumps(sorted(set(final_support.get((drive_id, frame_id), []))), ensure_ascii=True))
    qa["overlay_raw_path"] = overlay_raw
    qa["overlay_gated_path"] = overlay_gated
    qa["overlay_entities_path"] = overlay_entities
    qa["candidate_ids_nearby"] = cand_ids
    qa["crosswalk_final_ids_nearby"] = final_ids
    qa["final_entity_ids_nearby"] = final_ids
    qa.to_file(qa_index_path, driver="GeoJSON")


def _compress_ranges(frames: List[str]) -> str:
    if not frames:
        return ""
    ints = sorted({int(f) for f in frames if str(f).isdigit()})
    ranges = []
    start = prev = ints[0]
    for val in ints[1:]:
        if val == prev + 1:
            prev = val
            continue
        ranges.append((start, prev))
        start = prev = val
    ranges.append((start, prev))
    out = []
    for s, e in ranges:
        if s == e:
            out.append(f"{s:010d}")
        else:
            out.append(f"{s:010d}-{e:010d}")
    return ", ".join(out)


def _build_trace(
    records: List[dict],
    out_path: Path,
) -> None:
    if out_path.exists():
        out_path.unlink()
    import pandas as pd

    pd.DataFrame(records).to_csv(out_path, index=False)


def _build_report(
    out_path: Path,
    drive_id: str,
    total_frames: int,
    raw_frames: List[str],
    candidate_zero: int,
    reject_counts: Dict[str, int],
    final_gdf: gpd.GeoDataFrame,
    final_support: Dict[Tuple[str, str], List[str]],
    outputs_dir: Path,
) -> None:
    lines = []
    lines.append("# Crosswalk Monitor Report\n")
    lines.append(f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- drive_id: {drive_id}")
    lines.append(f"- total_frames: {total_frames}")
    lines.append("")
    lines.append(f"- raw_has_crosswalk_count: {len(raw_frames)}")
    lines.append(f"- raw_has_crosswalk_ranges: {_compress_ranges(raw_frames)}")
    lines.append("")
    lines.append(f"- candidate_written_zero: {candidate_zero}")
    lines.append("")
    lines.append("## Reject Reason Top10")
    if reject_counts:
        for reason, count in sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Final Summary")
    lines.append(f"- final_count: {len(final_gdf)}")
    if not final_gdf.empty and "frames_hit" in final_gdf.columns:
        frames_hit = [int(v) for v in final_gdf["frames_hit"].tolist() if v]
        if frames_hit:
            lines.append(f"- frames_hit_p50: {np.percentile(frames_hit, 50):.1f}")
            lines.append(f"- frames_hit_p90: {np.percentile(frames_hit, 90):.1f}")
    if final_gdf.empty:
        lines.append("- final_entity_ids: none")
    else:
        lines.append(f"- final_entity_ids: {', '.join(final_gdf['entity_id'].astype(str).tolist())}")
    if final_support:
        ranges = {}
        for (drv, frame_id), entity_ids in final_support.items():
            if drv != drive_id:
                continue
            for ent in entity_ids:
                ranges.setdefault(ent, []).append(frame_id)
        for ent, frames in ranges.items():
            lines.append(f"- {ent}: support_frames={_compress_ranges(frames)}")
    lines.append("")
    lines.append(f"- outputs_dir: {outputs_dir}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_monitor_drive.yaml")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--out-run", default="")
    args = ap.parse_args()

    log = _setup_logger()
    cfg = _load_yaml(Path(args.config))
    merged = _merge_config(
        cfg,
        {
            "drive_id": args.drive,
            "kitti_root": args.kitti_root,
        },
    )

    drive_id = str(merged.get("drive_id") or "")
    kitti_root = Path(str(merged.get("kitti_root") or ""))
    camera = str(merged.get("camera") or "image_00")
    stage1_stride = int(merged.get("stage1_stride", 1))
    export_all_frames = bool(merged.get("export_all_frames", True))
    min_frames_hit_final = int(merged.get("min_frames_hit_final", 3))
    raw_fallback_text = bool(merged.get("raw_fallback_text", True))
    draw_rejected = bool(merged.get("draw_rejected_candidates", True))
    write_wgs84 = bool(merged.get("write_wgs84", True))
    image_run = Path(str(merged.get("image_run") or ""))
    image_provider = str(merged.get("image_provider") or "grounded_sam2_v1")
    image_evidence_gpkg = str(merged.get("image_evidence_gpkg") or "")
    road_root = Path(str(merged.get("road_root") or ""))
    base_config = Path(str(merged.get("config") or "configs/road_entities.yaml"))

    if not drive_id:
        log.error("drive_id required")
        return 2
    if not kitti_root.exists():
        log.error("kitti_root missing: %s", kitti_root)
        return 2
    if not image_run.exists():
        log.error("image_run missing: %s", image_run)
        return 2
    if not road_root.exists():
        log.error("road_root missing: %s", road_root)
        return 2

    os.environ["POC_DATA_ROOT"] = str(kitti_root)
    if stage1_stride != 1:
        log.warning("stage1_stride=%s ignored; monitor runs full coverage.", stage1_stride)
    if not export_all_frames:
        log.warning("export_all_frames=false ignored; monitor exports all frames.")

    run_dir = Path(args.out_run) if args.out_run else Path("runs") / f"crosswalk_monitor_{drive_id}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    outputs_dir = run_dir / "outputs"
    debug_dir = run_dir / "debug"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    records = _build_index_records(kitti_root, drive_id, camera)
    if not records:
        log.error("no frames found for drive=%s", drive_id)
        return 3
    index_path = debug_dir / "monitor_index.jsonl"
    _write_index(records, index_path)
    log.info("index=%s total=%d", index_path, len(records))

    stage_cfg = debug_dir / "crosswalk_monitor.yaml"
    stage_cfg.write_text(
        yaml.safe_dump(
            _merge_config(_load_yaml(base_config), {"crosswalk_final": {"min_frames_hit": min_frames_hit_final}}),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    stage_dir = run_dir / "stage1"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    cmd = [
        sys.executable,
        "tools/build_road_entities.py",
        "--index",
        str(index_path),
        "--image-run",
        str(image_run),
        "--image-provider",
        str(image_provider),
        "--image-evidence-gpkg",
        str(image_evidence_gpkg),
        "--config",
        str(stage_cfg),
        "--road-root",
        str(road_root),
        "--out-dir",
        str(stage_dir),
        "--emit-qa-images",
        "1",
    ]
    log.info("stage1: %s", " ".join(cmd))
    if subprocess.run(cmd, check=False).returncode != 0:
        log.error("stage1 failed")
        return 4

    stage_outputs = stage_dir / "outputs"
    qa_images_src = stage_outputs / "qa_images"
    qa_images_dst = outputs_dir / "qa_images"
    if qa_images_dst.exists():
        shutil.rmtree(qa_images_dst)
    if qa_images_src.exists():
        shutil.copytree(qa_images_src, qa_images_dst)

    qa_index_src = stage_outputs / "qa_index_wgs84.geojson"
    qa_index_path = outputs_dir / "qa_index_wgs84.geojson"
    if qa_index_path.exists():
        qa_index_path.unlink()
    if qa_index_src.exists():
        shutil.copy2(qa_index_src, qa_index_path)

    index_lookup = {(r["drive_id"], _normalize_frame_id(r["frame_id"])): r.get("image_path", "") for r in records}
    raw_stats = _ensure_raw_overlays(
        qa_index_path,
        outputs_dir,
        image_run,
        image_provider,
        index_lookup,
        raw_fallback_text,
    )

    stage_gpkg = stage_outputs / "road_entities_utm32.gpkg"
    candidate_gdf = _read_candidates(stage_gpkg)
    final_gdf = _read_final(stage_gpkg)
    out_gpkg = outputs_dir / "crosswalk_entities_utm32.gpkg"
    _write_crosswalk_gpkg(stage_gpkg, out_gpkg)
    if write_wgs84:
        out_wgs84 = outputs_dir / "crosswalk_entities_wgs84.gpkg"
        if out_wgs84.exists():
            out_wgs84.unlink()
        cand_wgs84 = candidate_gdf.to_crs("EPSG:4326") if not candidate_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:4326")
        final_wgs84 = final_gdf.to_crs("EPSG:4326") if not final_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:4326")
        if not _ensure_wgs84_range(cand_wgs84) or not _ensure_wgs84_range(final_wgs84):
            log.error("wgs84 range check failed")
            return 5
        cand_wgs84.to_file(out_wgs84, layer="crosswalk_candidate_poly", driver="GPKG")
        final_wgs84.to_file(out_wgs84, layer="crosswalk_poly", driver="GPKG")

    final_support = _build_final_support(final_gdf)

    candidate_ids: Dict[Tuple[str, str], List[str]] = {}
    candidate_rejects: Dict[Tuple[str, str], List[str]] = {}
    candidate_by_frame: Dict[str, gpd.GeoDataFrame] = {}
    if not candidate_gdf.empty:
        candidate_gdf = candidate_gdf.copy()
        candidate_gdf["frame_id_norm"] = candidate_gdf["frame_id"].apply(_normalize_frame_id)
        for frame_id, group in candidate_gdf.groupby("frame_id_norm"):
            candidate_by_frame[str(frame_id)] = group
            for _, row in group.iterrows():
                d = str(row.get("drive_id") or "")
                if not d:
                    continue
                key = (d, str(frame_id))
                candidate_ids.setdefault(key, []).append(str(row.get("candidate_id") or ""))
                reasons = str(row.get("reject_reasons") or "")
                if reasons:
                    for token in [r for r in reasons.split(",") if r]:
                        candidate_rejects.setdefault(key, []).append(token)

    try:
        calib = load_kitti360_calib(kitti_root, camera)
    except Exception:
        calib = None
    pose_map: Dict[str, Tuple[float, float, float]] = {}
    for record in records:
        frame_id = _normalize_frame_id(record["frame_id"])
        try:
            x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
            pose_map[frame_id] = (x, y, yaw)
        except Exception:
            continue

    qa = gpd.read_file(qa_index_path)
    for idx, row in qa.iterrows():
        drive = str(row.get("drive_id") or "")
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        image_path = index_lookup.get((drive, frame_id), "")
        qa_dir = outputs_dir / "qa_images" / drive
        qa_dir.mkdir(parents=True, exist_ok=True)
        gated_path = qa_dir / f"{frame_id}_overlay_gated.png"
        entities_path = qa_dir / f"{frame_id}_overlay_entities.png"
        candidates = candidate_by_frame.get(frame_id, gpd.GeoDataFrame())
        kept, _ = _render_gated_overlay(
            gated_path,
            image_path,
            frame_id,
            candidates,
            pose_map.get(frame_id),
            calib,
            draw_rejected,
        )
        raw_has = int(row.get("raw_has_crosswalk", 0) or 0)
        _render_entities_overlay(
            entities_path,
            image_path,
            drive,
            frame_id,
            final_gdf,
            final_support,
            raw_has,
            pose_map.get(frame_id),
            calib,
        )
        qa.at[idx, "overlay_gated_path"] = str(gated_path)
        qa.at[idx, "overlay_entities_path"] = str(entities_path)
        qa.at[idx, "gated_kept"] = kept
    qa.to_file(qa_index_path, driver="GeoJSON")

    _update_qa_index(qa_index_path, index_lookup, final_support, candidate_ids)

    trace_records = []
    reject_counts: Dict[str, int] = {}
    raw_frames = []
    candidate_zero = 0
    for record in records:
        frame_id = _normalize_frame_id(record["frame_id"])
        key = (drive_id, frame_id)
        raw_info = raw_stats.get(
            key,
            {"raw_has_crosswalk": 0.0, "raw_top_score": 0.0, "raw_status": "unknown"},
        )
        cand_ids = sorted(set(candidate_ids.get(key, [])))
        rejects = sorted(set(candidate_rejects.get(key, [])))
        for r in rejects:
            reject_counts[r] = reject_counts.get(r, 0) + 1
        gated_kept = 1 if cand_ids and not rejects else 0
        final_ids = sorted(set(final_support.get(key, [])))
        if int(raw_info.get("raw_has_crosswalk", 0)) == 1:
            raw_frames.append(frame_id)
        if not cand_ids:
            candidate_zero += 1
        note = ""
        if not cand_ids:
            if int(raw_info.get("raw_has_crosswalk", 0)) == 0:
                note = "raw_empty"
            else:
                note = "candidate_missing"
        elif rejects and not gated_kept:
            note = "rejected"
        trace_records.append(
            {
                "drive_id": drive_id,
                "frame_id": frame_id,
                "image_path": record.get("image_path", ""),
                "raw_status": raw_info.get("raw_status", "unknown"),
                "raw_has_crosswalk": int(raw_info.get("raw_has_crosswalk", 0)),
                "raw_top_score": raw_info.get("raw_top_score", 0.0),
                "candidate_written": 1 if cand_ids else 0,
                "reject_reasons": "|".join(rejects),
                "gated_kept": gated_kept,
                "final_support": 1 if final_ids else 0,
                "final_entity_ids": "|".join(final_ids),
                "note": note,
            }
        )

    trace_path = outputs_dir / "crosswalk_trace.csv"
    _build_trace(trace_records, trace_path)

    report_path = outputs_dir / "crosswalk_monitor_report.md"
    _build_report(
        report_path,
        drive_id,
        len(records),
        raw_frames,
        candidate_zero,
        reject_counts,
        final_gdf,
        final_support,
        outputs_dir,
    )

    log.info("done: %s", outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
