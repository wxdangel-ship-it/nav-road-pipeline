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


LOG = logging.getLogger("run_crosswalk_monitor_range")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_monitor_range")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _normalize_frame_id(frame_id: str) -> str:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    if not digits:
        return str(frame_id)
    return digits.zfill(10)


def _merge_config(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for key, val in overrides.items():
        if val is None:
            continue
        out[key] = val
    return out


def _build_index_records(
    kitti_root: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    camera: str,
) -> List[dict]:
    records: List[dict] = []
    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    image_by_frame: Dict[str, str] = {}
    if image_dir:
        for path in _list_images(image_dir):
            frame_id = _extract_frame_id(path)
            image_by_frame[_normalize_frame_id(frame_id)] = str(path)
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        records.append(
            {
                "drive_id": drive_id,
                "camera": camera,
                "frame_id": frame_id,
                "image_path": image_by_frame.get(frame_id, ""),
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
        y += 24
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


def _update_qa_index_with_final(
    qa_index_path: Path,
    final_gdf: gpd.GeoDataFrame,
) -> Dict[Tuple[str, str], List[str]]:
    if not qa_index_path.exists():
        return {}
    qa = gpd.read_file(qa_index_path)
    support_map: Dict[Tuple[str, str], List[str]] = {}
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
    supports = []
    final_ids = []
    for _, row in qa.iterrows():
        key = (str(row.get("drive_id") or ""), str(row.get("frame_id") or ""))
        ids = support_map.get(key, [])
        supports.append("yes" if ids else "no")
        final_ids.append(json.dumps(sorted(set(ids)), ensure_ascii=True))
    qa["entity_support_frames"] = supports
    qa["crosswalk_final_ids_nearby"] = final_ids
    qa["final_entity_ids_nearby"] = final_ids
    qa.to_file(qa_index_path, driver="GeoJSON")
    return support_map


def _ensure_gated_entities_images(
    qa_index_path: Path,
    outputs_dir: Path,
    candidate_gdf: gpd.GeoDataFrame,
    final_support: Dict[Tuple[str, str], List[str]],
    index_lookup: Dict[Tuple[str, str], str],
    final_gdf: gpd.GeoDataFrame,
    kitti_root: Path,
    camera: str,
) -> None:
    if not qa_index_path.exists():
        return
    qa = gpd.read_file(qa_index_path)
    candidate_rejects: Dict[Tuple[str, str], List[str]] = {}
    candidate_ids: Dict[Tuple[str, str], List[str]] = {}
    if not candidate_gdf.empty:
        for _, row in candidate_gdf.iterrows():
            drive_id = str(row.get("drive_id") or "")
            frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
            if not drive_id or not frame_id:
                continue
            key = (drive_id, frame_id)
            candidate_ids.setdefault(key, []).append(str(row.get("candidate_id") or ""))
            reasons = str(row.get("reject_reasons") or "")
            if reasons:
                for token in [r for r in reasons.split(",") if r]:
                    candidate_rejects.setdefault(key, []).append(token)
    for idx, row in qa.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not drive_id or not frame_id:
            continue
        image_path = index_lookup.get((drive_id, frame_id), "")
        qa_dir = outputs_dir / "qa_images" / drive_id
        qa_dir.mkdir(parents=True, exist_ok=True)
        gated_path = qa_dir / f"{frame_id}_overlay_gated.png"
        entities_path = qa_dir / f"{frame_id}_overlay_entities.png"
        if not gated_path.exists():
            rejects = sorted(set(candidate_rejects.get((drive_id, frame_id), [])))
            lines = [
                "NO_GATED_RENDER",
                f"REJECT_REASONS:{'|'.join(rejects) if rejects else 'none'}",
            ]
            _render_text_overlay(image_path, gated_path, lines)
            qa.at[idx, "overlay_gated_path"] = str(gated_path)
        support_ids = final_support.get((drive_id, frame_id), [])
        if support_ids:
            _render_final_entities_image(
                entities_path,
                image_path,
                final_gdf,
                drive_id,
                frame_id,
                kitti_root,
                camera,
            )
            qa.at[idx, "overlay_entities_path"] = str(entities_path)
        elif not entities_path.exists():
            _render_text_overlay(image_path, entities_path, ["NO_FINAL_ENTITY"])
            qa.at[idx, "overlay_entities_path"] = str(entities_path)
        qa.at[idx, "candidate_ids_nearby"] = json.dumps(sorted(set(candidate_ids.get((drive_id, frame_id), []))), ensure_ascii=True)
    qa.to_file(qa_index_path, driver="GeoJSON")


def _render_final_entities_image(
    out_path: Path,
    image_path: str,
    final_gdf: gpd.GeoDataFrame,
    drive_id: str,
    frame_id: str,
    kitti_root: Path,
    camera: str,
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
    try:
        calib = load_kitti360_calib(kitti_root, camera)
    except Exception:
        _render_text_overlay(image_path, out_path, ["NO_CALIB"])
        return
    try:
        x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
    except Exception:
        _render_text_overlay(image_path, out_path, ["NO_POSE"])
        return
    pose = (x, y, yaw)
    if final_gdf.empty:
        _render_text_overlay(image_path, out_path, ["NO_FINAL_ENTITY"])
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


def _build_trace(
    out_path: Path,
    records: List[dict],
) -> None:
    if out_path.exists():
        out_path.unlink()
    import pandas as pd

    pd.DataFrame(records).to_csv(out_path, index=False)


def _build_trace_records(
    drive_id: str,
    frame_start: int,
    frame_end: int,
    index_lookup: Dict[Tuple[str, str], str],
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    candidate_gdf: gpd.GeoDataFrame,
    final_support: Dict[Tuple[str, str], List[str]],
    feature_store_map_root: Path,
) -> List[dict]:
    candidate_ids: Dict[Tuple[str, str], List[str]] = {}
    candidate_rejects: Dict[Tuple[str, str], List[str]] = {}
    if not candidate_gdf.empty:
        for _, row in candidate_gdf.iterrows():
            d = str(row.get("drive_id") or "")
            f = _normalize_frame_id(str(row.get("frame_id") or ""))
            if not d or not f:
                continue
            key = (d, f)
            candidate_ids.setdefault(key, []).append(str(row.get("candidate_id") or ""))
            reasons = str(row.get("reject_reasons") or "")
            if reasons:
                for token in [r for r in reasons.split(",") if r]:
                    candidate_rejects.setdefault(key, []).append(token)
    records: List[dict] = []
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        key = (drive_id, frame_id)
        raw_info = raw_stats.get(
            key,
            {"raw_has_crosswalk": 0.0, "raw_top_score": 0.0, "raw_status": "unknown"},
        )
        map_path = feature_store_map_root / drive_id / frame_id / "map_evidence_utm32.gpkg"
        project_ok = 1 if map_path.exists() else 0
        cand_ids = sorted(set(candidate_ids.get(key, [])))
        rejects = sorted(set(candidate_rejects.get(key, [])))
        final_ids = sorted(set(final_support.get(key, [])))
        drop_reason = ""
        if not cand_ids:
            if int(raw_info.get("raw_has_crosswalk", 0)) == 0:
                drop_reason = "raw_empty"
            elif project_ok == 0:
                drop_reason = "project_failed"
            else:
                drop_reason = "candidate_missing"
        records.append(
            {
                "drive_id": drive_id,
                "frame_id": frame_id,
                "image_path": index_lookup.get(key, ""),
                "raw_status": raw_info.get("raw_status", "unknown"),
                "raw_has_crosswalk": int(raw_info.get("raw_has_crosswalk", 0)),
                "raw_top_score": raw_info.get("raw_top_score", 0.0),
                "project_ok": project_ok,
                "candidate_written": 1 if cand_ids else 0,
                "candidate_id": "|".join(cand_ids),
                "reject_reasons": "|".join(rejects),
                "final_support": 1 if final_ids else 0,
                "final_entity_ids": "|".join(final_ids),
                "drop_reason": drop_reason,
            }
        )
    return records


def _build_report(
    out_path: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    trace_records: List[dict],
    final_gdf: gpd.GeoDataFrame,
    top_reject: List[dict],
    outputs_dir: Path,
) -> None:
    lines = []
    lines.append("# Crosswalk Monitor Report\n")
    lines.append(f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- drive_id: {drive_id}")
    lines.append(f"- frame_range: {frame_start}-{frame_end}")
    lines.append(f"- total_frames: {frame_end - frame_start + 1}")
    lines.append("")
    raw_hits = [r["frame_id"] for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1]
    lines.append(f"- raw_has_crosswalk_count: {len(raw_hits)}")
    lines.append(f"- raw_has_crosswalk_frames: {', '.join(raw_hits)}")
    lines.append("")
    cand_missing = [r for r in trace_records if int(r.get("candidate_written", 0)) == 0]
    lines.append(f"- candidate_written_zero: {len(cand_missing)}")
    drop_counts: Dict[str, int] = {}
    for row in cand_missing:
        reason = str(row.get("drop_reason") or "")
        if reason:
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
    if drop_counts:
        lines.append("## Candidate Drop Reasons")
        for reason, count in sorted(drop_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {reason}: {count}")
        lines.append("")
    reject_counts: Dict[str, int] = {}
    for row in trace_records:
        for reason in [r for r in str(row.get("reject_reasons") or "").split("|") if r]:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    if reject_counts:
        lines.append("## Reject Reason Summary")
        for reason, count in sorted(reject_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {reason}: {count}")
        lines.append("")
    lines.append("## Final Summary")
    lines.append(f"- final_count: {len(final_gdf)}")
    if not final_gdf.empty:
        frames = [int(v) for v in final_gdf["frames_hit"].tolist() if v]
        if frames:
            lines.append(f"- frames_hit_p50: {np.percentile(frames, 50):.1f}")
            lines.append(f"- frames_hit_p90: {np.percentile(frames, 90):.1f}")
        lines.append(f"- final_entity_ids: {', '.join(final_gdf['entity_id'].astype(str).tolist())}")
    lines.append("")
    lines.append("## Top Reject Samples")
    if not top_reject:
        lines.append("- none")
    else:
        for row in top_reject[:10]:
            lines.append(
                f"- {row.get('drive_id')}:{row.get('frame_id')} score={row.get('score_total'):.3f} reject={row.get('reject_reasons')}"
            )
    lines.append("")
    lines.append(f"- outputs_dir: {outputs_dir}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_monitor.yaml")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--out-run", default="")
    args = ap.parse_args()

    log = _setup_logger()
    cfg = _load_yaml(Path(args.config))
    merged = _merge_config(
        cfg,
        {
            "drive_id": args.drive,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "kitti_root": args.kitti_root,
        },
    )

    drive_id = str(merged.get("drive_id") or "")
    frame_start = int(merged.get("frame_start", 0))
    frame_end = int(merged.get("frame_end", 0))
    kitti_root = Path(str(merged.get("kitti_root") or ""))
    camera = str(merged.get("camera") or "image_00")
    stage1_stride = int(merged.get("stage1_stride", 1))
    export_all_frames = bool(merged.get("export_all_frames", True))
    min_frames_hit_final = int(merged.get("min_frames_hit_final", 3))
    write_wgs84 = bool(merged.get("write_wgs84", True))
    raw_fallback_text = bool(merged.get("raw_fallback_text", True))
    image_run = Path(str(merged.get("image_run") or ""))
    image_provider = str(merged.get("image_provider") or "grounded_sam2_v1")
    image_evidence_gpkg = str(merged.get("image_evidence_gpkg") or "")
    road_root = Path(str(merged.get("road_root") or ""))
    base_config = Path(str(merged.get("config") or "configs/road_entities.yaml"))

    if not drive_id or frame_end < frame_start:
        log.error("invalid drive/frame range")
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

    run_dir = Path(args.out_run) if args.out_run else Path("runs") / f"crosswalk_monitor_{drive_id}_{frame_start}_{frame_end}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    outputs_dir = run_dir / "outputs"
    debug_dir = run_dir / "debug"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    index_records = _build_index_records(kitti_root, drive_id, frame_start, frame_end, camera)
    index_path = debug_dir / "monitor_index.jsonl"
    _write_index(index_records, index_path)
    log.info("index=%s total=%d", index_path, len(index_records))

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
        return 3

    stage_outputs = stage_dir / "outputs"
    qa_images_src = stage_outputs / "qa_images"
    qa_images_dst = outputs_dir / "qa_images"
    if qa_images_dst.exists():
        shutil.rmtree(qa_images_dst)
    if qa_images_src.exists():
        shutil.copytree(qa_images_src, qa_images_dst)

    qa_index_path = stage_outputs / "qa_index_wgs84.geojson"
    qa_out_path = outputs_dir / "qa_index_wgs84.geojson"
    if qa_out_path.exists():
        qa_out_path.unlink()
    if qa_index_path.exists():
        shutil.copy2(qa_index_path, qa_out_path)

    index_lookup = {(r["drive_id"], _normalize_frame_id(r["frame_id"])): r.get("image_path", "") for r in index_records}
    raw_stats = _ensure_raw_overlays(
        qa_out_path,
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
            return 4
        cand_wgs84.to_file(out_wgs84, layer="crosswalk_candidate_poly", driver="GPKG")
        final_wgs84.to_file(out_wgs84, layer="crosswalk_poly", driver="GPKG")

    final_support = _update_qa_index_with_final(qa_out_path, final_gdf)
    _ensure_gated_entities_images(
        qa_out_path,
        outputs_dir,
        candidate_gdf,
        final_support,
        index_lookup,
        final_gdf,
        kitti_root,
        camera,
    )

    feature_store_map_root = image_run / f"feature_store_map_{image_provider}"
    trace_records = _build_trace_records(
        drive_id,
        frame_start,
        frame_end,
        index_lookup,
        raw_stats,
        candidate_gdf,
        final_support,
        feature_store_map_root,
    )
    trace_path = outputs_dir / "crosswalk_trace.csv"
    _build_trace(trace_path, trace_records)

    top_reject = []
    if not candidate_gdf.empty and "score_total" in candidate_gdf.columns:
        top_reject = (
            candidate_gdf.sort_values("score_total", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )
    report_path = outputs_dir / "crosswalk_monitor_report.md"
    _build_report(
        report_path,
        drive_id,
        frame_start,
        frame_end,
        trace_records,
        final_gdf,
        top_reject,
        outputs_dir,
    )

    if export_all_frames and qa_out_path.exists():
        qa = gpd.read_file(qa_out_path)
        if len(qa) != frame_end - frame_start + 1:
            log.warning("qa_index_count=%d expected=%d", len(qa), frame_end - frame_start + 1)

    log.info("done: %s", outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
