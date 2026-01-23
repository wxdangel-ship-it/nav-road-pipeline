from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import yaml
import sys
from shapely.geometry import Point
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_image_sample_index import _find_image_dir, _list_images, _extract_frame_id


LOG = logging.getLogger("run_crosswalk_full")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_full")


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _load_index_drives(path: Path) -> List[str]:
    drives = []
    if not path.exists():
        return drives
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        drive = row.get("drive_id") or row.get("drive") or row.get("tile_id")
        if drive:
            drives.append(str(drive))
    return sorted(set(drives))


def _load_golden_drives(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _expand_drive_filter(drives: List[str]) -> List[str]:
    expanded: List[str] = []
    for token in drives:
        key = token.strip().lower()
        if key in {"golden8", "golden"}:
            expanded.extend(_load_golden_drives(Path("configs") / "golden_drives.txt"))
        elif token:
            expanded.append(token)
    return sorted(set(expanded))


def _write_index(records: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_index_map(path: Path) -> Dict[Tuple[str, str], str]:
    if not path.exists():
        return {}
    out: Dict[Tuple[str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        drive_id = str(row.get("drive_id") or "")
        frame_id = str(row.get("frame_id") or "")
        image_path = str(row.get("image_path") or "")
        if drive_id and frame_id:
            out[(drive_id, frame_id)] = image_path
    return out


def _load_index_frames(path: Path) -> Dict[str, List[str]]:
    frames_by_drive: Dict[str, List[str]] = {}
    if not path.exists():
        return frames_by_drive
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        drive_id = str(row.get("drive_id") or "")
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not drive_id or not frame_id:
            continue
        frames_by_drive.setdefault(drive_id, []).append(frame_id)
    for drive_id, frames in frames_by_drive.items():
        frames_by_drive[drive_id] = sorted(set(frames))
    return frames_by_drive


def _build_stride_index(
    data_root: Path,
    drives: List[str],
    stride: int,
    out_path: Path,
    camera: str,
) -> None:
    records: List[dict] = []
    for drive in drives:
        img_dir = _find_image_dir(data_root, drive, camera)
        if not img_dir:
            LOG.warning("drive=%s image dir not found", drive)
            continue
        images = _list_images(img_dir)
        if not images:
            continue
        sampled = images[::max(1, stride)]
        for path in sampled:
            records.append(
                {
                    "drive_id": drive,
                    "camera": camera,
                    "frame_id": _extract_frame_id(path),
                    "image_path": str(path),
                    "scene_profile": "car",
                }
            )
    _write_index(records, out_path)
    LOG.info("stage1 index=%s total=%d", out_path, len(records))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _frame_id_int(frame_id: str) -> int:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    return int(digits) if digits else -1


def _normalize_frame_id(frame_id: str) -> str:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    if not digits:
        return str(frame_id)
    return digits.zfill(10)


def _score_candidate(row: gpd.GeoSeries) -> float:
    inside = float(row.get("inside_road_ratio", 0.0) or 0.0)
    rect = float(row.get("rectangularity", 0.0) or 0.0)
    heading = row.get("heading_diff_to_perp_deg", "")
    heading_score = 0.5
    if heading != "" and heading is not None:
        heading_score = max(0.0, 1.0 - min(float(heading) / 45.0, 1.0))
    return float(0.6 * inside + 0.25 * rect + 0.15 * heading_score)


def _cluster_by_centroid(gdf: gpd.GeoDataFrame, eps_m: float) -> List[List[int]]:
    if gdf.empty:
        return []
    centroids = [geom.centroid for geom in gdf.geometry]
    try:
        from shapely.strtree import STRtree

        tree = STRtree(centroids)
        geom_id = {id(g): idx for idx, g in enumerate(centroids)}
    except Exception:
        tree = None
        geom_id = {}
    parent = list(range(len(centroids)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri = _find(i)
        rj = _find(j)
        if ri != rj:
            parent[rj] = ri

    for i, center in enumerate(centroids):
        candidates = []
        if tree is not None:
            for cand in tree.query(center.buffer(eps_m)):
                if isinstance(cand, (int, np.integer)):
                    idx = int(cand)
                else:
                    idx = geom_id.get(id(cand))
                if idx is not None:
                    candidates.append(idx)
        else:
            candidates = list(range(len(centroids)))
        for j in candidates:
            if j <= i:
                continue
            if center.distance(centroids[j]) <= eps_m:
                _union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(centroids)):
        root = _find(idx)
        clusters.setdefault(root, []).append(idx)
    return list(clusters.values())


def _write_stage1_candidates(cand_gdf: gpd.GeoDataFrame, out_csv: Path) -> gpd.GeoDataFrame:
    if cand_gdf.empty:
        _safe_unlink(out_csv)
        return cand_gdf
    cand = cand_gdf.copy()
    cand["score_total"] = cand.apply(_score_candidate, axis=1)
    cand = cand.sort_values(["drive_id", "score_total"], ascending=[True, False])
    cols = [
        "drive_id",
        "frame_id",
        "candidate_id",
        "score_total",
        "inside_road_ratio",
        "rectangularity",
        "rect_w_m",
        "rect_l_m",
        "heading_diff_to_perp_deg",
        "reject_reasons",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(out_csv)
    cand[cols].to_csv(out_csv, index=False)
    return cand


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
    layers = _list_layers(path)
    for name in layers:
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


def _geom_to_xy(geom) -> List[List[Tuple[float, float]]]:
    if geom is None or geom.is_empty:
        return []
    geoms = []
    if geom.geom_type == "Polygon":
        geoms.append(geom)
    elif geom.geom_type == "MultiPolygon":
        geoms.extend(list(geom.geoms))
    elif geom.geom_type == "LineString":
        return [[(float(x), float(y)) for x, y in geom.coords]]
    elif geom.geom_type == "MultiLineString":
        return [[(float(x), float(y)) for x, y in line.coords] for line in geom.geoms]
    else:
        return []
    out = []
    for poly in geoms:
        out.append([(float(x), float(y)) for x, y in poly.exterior.coords])
    return out


def _render_raw_overlay(
    image_path: str,
    raw_gdf: gpd.GeoDataFrame,
    out_path: Path,
    missing_status: str,
    raw_fallback_mode: str = "text_on_image",
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
    if raw_fallback_mode == "text_on_image" and missing_status and missing_status != "ok":
        draw.text((10, 10), f"MISSING_FEATURE_STORE:{missing_status}", fill=(255, 128, 0, 220))
        draw.text((10, 30), "RAW_FALLBACK=text_on_image", fill=(255, 255, 255, 220))
        base.save(out_path)
        return
    if raw_gdf is None or raw_gdf.empty:
        if missing_status and missing_status != "ok":
            draw.text((10, 10), f"MISSING_FEATURE_STORE:{missing_status}", fill=(255, 128, 0, 220))
        draw.text((10, 30), "NO_CROSSWALK_DETECTED", fill=(255, 0, 0, 220))
        base.save(out_path)
        return
    for _, row in raw_gdf.iterrows():
        geom = row.geometry
        for coords in _geom_to_xy(geom):
            if len(coords) < 2:
                continue
            draw.line(coords, fill=(255, 0, 0, 220), width=3)
    base.save(out_path)


def _ensure_raw_overlays(
    qa_index_path: Path,
    outputs_dir: Path,
    image_run: Path,
    provider_id: str,
    index_lookup: Dict[Tuple[str, str], str],
    emit_missing_list: bool,
    raw_fallback_mode: str,
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
        if not image_path or not Path(image_path).exists():
            if raw_status == "missing_feature_store" and raw_fallback_mode == "text_on_image":
                _render_raw_overlay("", raw_gdf, out_path, raw_status, raw_fallback_mode)
                qa_gdf.at[idx, "overlay_raw_path"] = str(out_path)
            continue
        _render_raw_overlay(image_path, raw_gdf, out_path, raw_status, raw_fallback_mode)
        qa_gdf.at[idx, "overlay_raw_path"] = str(out_path)
    if qa_index_path.exists():
        _safe_unlink(qa_index_path)
    qa_gdf.to_file(qa_index_path, driver="GeoJSON")
    if emit_missing_list:
        missing_path = outputs_dir / "missing_feature_store_list.csv"
        if missing_path.exists():
            missing_path.unlink()
        import pandas as pd

        pd.DataFrame(
            missing_rows,
            columns=["drive_id", "frame_id", "raw_status", "image_path"],
        ).to_csv(missing_path, index=False)
    return raw_stats


def _write_trace(
    trace_path: Path,
    records: List[dict],
) -> None:
    if trace_path.exists():
        trace_path.unlink()
    if not records:
        return
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_csv(trace_path, index=False)
    jsonl_path = trace_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        jsonl_path.unlink()
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_trace_records(
    stage: str,
    qa_frames: Dict[str, List[str]],
    candidate_gdf: gpd.GeoDataFrame,
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    feature_store_map_root: Path,
) -> List[dict]:
    records: List[dict] = []
    candidate_lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not candidate_gdf.empty:
        for _, row in candidate_gdf.iterrows():
            drive_id = str(row.get("drive_id") or "")
            frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
            if not drive_id or not frame_id:
                continue
            key = (drive_id, frame_id)
            candidate_lookup.setdefault(key, {"candidate_id": "", "reject_reasons": ""})
            candidate_lookup[key]["candidate_id"] = str(row.get("candidate_id") or "")
            candidate_lookup[key]["reject_reasons"] = str(row.get("reject_reasons") or "")
    for drive_id, frames in qa_frames.items():
        for frame_id in frames:
            norm_frame = _normalize_frame_id(frame_id)
            key = (drive_id, norm_frame)
            raw_info = raw_stats.get(
                key,
                {"raw_has_crosswalk": 0.0, "raw_top_score": 0.0, "raw_status": "unknown"},
            )
            map_path = feature_store_map_root / drive_id / norm_frame / "map_evidence_utm32.gpkg"
            project_ok = 1 if map_path.exists() else 0
            cand = candidate_lookup.get(key, {})
            candidate_written = 1 if cand.get("candidate_id") else 0
            drop_reason = ""
            if candidate_written == 0:
                if raw_info["raw_has_crosswalk"] == 0:
                    drop_reason = "raw_empty"
                elif project_ok == 0:
                    drop_reason = "project_failed"
                else:
                    drop_reason = "candidate_missing"
            records.append(
                {
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "stage": stage,
                    "raw_has_crosswalk": int(raw_info["raw_has_crosswalk"]),
                    "raw_top_score": raw_info["raw_top_score"],
                    "raw_status": raw_info.get("raw_status", "unknown"),
                    "project_ok": project_ok,
                    "candidate_written": candidate_written,
                    "candidate_id": cand.get("candidate_id", ""),
                    "reject_reasons": cand.get("reject_reasons", ""),
                    "dropped_hard": 1 if (candidate_written == 0 and raw_info["raw_has_crosswalk"] == 1 and project_ok == 1) else 0,
                    "drop_reason": drop_reason,
                }
            )
    return records


def _build_stage2_windows(
    cand_gdf: gpd.GeoDataFrame,
    out_jsonl: Path,
    eps_m: float,
    window_half: int,
    topk_per_drive: int,
) -> List[dict]:
    windows: List[dict] = []
    if cand_gdf.empty:
        _safe_unlink(out_jsonl)
        return windows
    for drive_id, group in cand_gdf.groupby("drive_id"):
        clusters = _cluster_by_centroid(group, eps_m)
        scored_clusters = []
        for cluster_id, cluster in enumerate(clusters):
            hits = group.iloc[cluster].copy()
            hits["score_total"] = hits.apply(_score_candidate, axis=1)
            best = hits.sort_values("score_total", ascending=False).iloc[0]
            scored_clusters.append((cluster_id, cluster, float(best.get("score_total", 0.0)), best))
        scored_clusters = sorted(scored_clusters, key=lambda x: x[2], reverse=True)
        for rank, (cluster_id, cluster, score_total, best) in enumerate(scored_clusters[: max(1, topk_per_drive)]):
            center_frame = _frame_id_int(best.get("frame_id"))
            if center_frame < 0:
                continue
            window = {
                "drive_id": drive_id,
                "frame_start": max(0, center_frame - window_half),
                "frame_end": center_frame + window_half,
                "reason": "stage1_candidate",
                "center_frame": center_frame,
                "score_total": score_total,
                "cluster_id": cluster_id,
                "rank": rank + 1,
            }
            windows.append(window)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(out_jsonl)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in windows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return windows


def _build_stage2_index(
    data_root: Path,
    windows: List[dict],
    out_path: Path,
    camera: str,
) -> None:
    records: List[dict] = []
    for window in windows:
        drive_id = str(window["drive_id"])
        frame_start = int(window["frame_start"])
        frame_end = int(window["frame_end"])
        img_dir = _find_image_dir(data_root, drive_id, camera)
        if not img_dir:
            LOG.warning("stage2 drive=%s image dir missing", drive_id)
            continue
        images = _list_images(img_dir)
        for path in images:
            frame_id = _extract_frame_id(path)
            frame_int = _frame_id_int(frame_id)
            if frame_int < frame_start or frame_int > frame_end:
                continue
            records.append(
                {
                    "drive_id": drive_id,
                    "camera": camera,
                    "frame_id": frame_id,
                    "image_path": str(path),
                    "scene_profile": "car",
                }
            )
    # dedup
    seen = set()
    unique = []
    for row in records:
        key = (row["drive_id"], row["frame_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    _write_index(unique, out_path)
    LOG.info("stage2 index=%s total=%d", out_path, len(unique))


def _update_config(
    base_cfg: Path,
    overrides: Dict[str, dict],
    out_path: Path,
) -> None:
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
    for key, value in overrides.items():
        if key not in cfg or not isinstance(cfg[key], dict):
            cfg[key] = {}
        cfg[key].update(value)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _read_candidates(gpkg_path: Path) -> gpd.GeoDataFrame:
    if not gpkg_path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(gpkg_path, layer="crosswalk_candidate_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _read_final(gpkg_path: Path) -> gpd.GeoDataFrame:
    if not gpkg_path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(gpkg_path, layer="crosswalk_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _write_crosswalk_gpkg(src_gpkg: Path, out_gpkg: Path) -> None:
    layers = {}
    for layer in ("crosswalk_candidate_poly", "crosswalk_poly"):
        try:
            layers[layer] = gpd.read_file(src_gpkg, layer=layer)
        except Exception:
            layers[layer] = gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    if out_gpkg.exists():
        out_gpkg.unlink()
    for layer, gdf in layers.items():
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:32632")
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")


def _select_qa_frames(
    final_gdf: gpd.GeoDataFrame,
    cand_gdf: gpd.GeoDataFrame,
    qa_frames_per_entity: int,
    topn_reject: int,
) -> Tuple[Dict[str, List[str]], List[dict]]:
    per_drive_frames: Dict[str, List[str]] = {}
    if not final_gdf.empty:
        for _, row in final_gdf.iterrows():
            drive_id = str(row.get("drive_id") or "")
            if not drive_id:
                continue
            frame_ids = str(row.get("frame_ids") or "")
            if ".." in frame_ids:
                parts = frame_ids.split("..")
                start = _frame_id_int(parts[0])
                end = _frame_id_int(parts[1].split()[0]) if parts[1] else start
                if start >= 0 and end >= start:
                    frames = [f"{i:010d}" for i in range(start, end + 1)]
                else:
                    frames = []
            else:
                frames = [frame_ids] if frame_ids else []
            if not frames:
                continue
            if qa_frames_per_entity > 0:
                step = max(1, len(frames) // qa_frames_per_entity)
                frames = frames[::step][:qa_frames_per_entity]
            per_drive_frames.setdefault(drive_id, []).extend(frames)

    reject_samples = []
    if not cand_gdf.empty:
        cand = cand_gdf.copy()
        cand["score_total"] = cand.apply(_score_candidate, axis=1)
        cand = cand[cand["reject_reasons"].fillna("") != ""]
        cand = cand.sort_values("score_total", ascending=False).head(topn_reject)
        for _, row in cand.iterrows():
            drive_id = str(row.get("drive_id") or "")
            frame_id = str(row.get("frame_id") or "")
            if drive_id and frame_id:
                per_drive_frames.setdefault(drive_id, []).append(frame_id)
            reject_samples.append(
                {
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "reject_reasons": row.get("reject_reasons", ""),
                    "score_total": float(row.get("score_total", 0.0)),
                }
            )

    # de-dup frames
    for drive_id in list(per_drive_frames.keys()):
        frames = sorted(set(per_drive_frames[drive_id]))
        per_drive_frames[drive_id] = frames
    return per_drive_frames, reject_samples


def _filter_qa_assets(
    qa_index_path: Path,
    outputs_dir: Path,
    per_drive_frames: Dict[str, List[str]],
) -> Path:
    qa_gdf = gpd.read_file(qa_index_path)
    if qa_gdf.empty:
        return qa_index_path
    keep = []
    for _, row in qa_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frame_id = str(row.get("frame_id") or "")
        if frame_id in set(per_drive_frames.get(drive_id, [])):
            keep.append(True)
        else:
            keep.append(False)
    qa_gdf = qa_gdf[keep]
    new_path = outputs_dir / "qa_index_wgs84.geojson"
    if new_path.exists():
        new_path.unlink()
    for idx, row in qa_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        for key in ("overlay_raw_path", "overlay_gated_path", "overlay_entities_path"):
            path = row.get(key, "")
            if not path:
                continue
            name = Path(path).name
            if drive_id:
                qa_gdf.at[idx, key] = str(outputs_dir / "qa_images" / drive_id / name)
    qa_gdf.to_file(new_path, driver="GeoJSON")

    # prune images
    qa_dir = outputs_dir / "qa_images"
    if qa_dir.exists():
        keep_files = set()
        for _, row in qa_gdf.iterrows():
            for key in ("overlay_raw_path", "overlay_gated_path", "overlay_entities_path"):
                path = row.get(key, "")
                if path:
                    keep_files.add(Path(path).resolve())
        for path in qa_dir.rglob("*.png"):
            if path.resolve() not in keep_files:
                path.unlink()
    return new_path


def _load_road_poly(road_root: Path, drive_id: str) -> gpd.GeoSeries:
    candidates = [
        road_root / drive_id / "geom_outputs" / "road_polygon_wgs84.geojson",
        road_root / drive_id / "geom_outputs" / "road_polygon_utm32.gpkg",
        road_root / drive_id / "geom_outputs" / "road_polygon_utm32.geojson",
        road_root / drive_id / "geom_outputs" / "road_polygon.geojson",
    ]
    for path in candidates:
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        if gdf.empty:
            continue
        if "wgs84" in path.name.lower():
            gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:32632")
        elif gdf.crs is None:
            gdf = gdf.set_crs("EPSG:32632")
        return gdf.geometry.union_all()
    return None


def _rebuild_final_from_candidates(
    cand_gdf: gpd.GeoDataFrame,
    road_root: Path,
    cfg: dict,
) -> gpd.GeoDataFrame:
    if cand_gdf.empty:
        return gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    cross_cfg = cfg.get("crosswalk", {})
    final_cfg = cfg.get("crosswalk_final", {})
    cross_eps = float(cfg.get("clustering", {}).get("crosswalk_eps_m", 3.0))
    cross_buf = float(cross_cfg.get("cluster_buffer_m", 3.0))
    eps_m = max(cross_eps, cross_buf * 2.0)

    entities = []
    for drive_id, group in cand_gdf.groupby("drive_id"):
        road_poly = _load_road_poly(road_root, drive_id)
        clusters = _cluster_by_centroid(group, eps_m)
        for idx, indices in enumerate(clusters):
            hits = group.iloc[indices].copy()
            hits_support = hits[hits.get("gore_overlap", 0) != 1]
            frames = set(hits_support["frame_id"].tolist()) if not hits_support.empty else set(hits["frame_id"].tolist())
            frames_hit = len(frames)
            min_frames_hit = int(final_cfg.get("min_frames_hit", 3))
            if frames_hit < min_frames_hit:
                continue

            strict = hits[hits["reject_reasons"].fillna("") == ""]
            geom_hits = strict if not strict.empty else hits
            if geom_hits.empty:
                continue
            med_w = float(geom_hits["rect_w_m"].median())
            med_l = float(geom_hits["rect_l_m"].median())
            geom_hits = geom_hits.copy()
            geom_hits["_size_dist"] = (geom_hits["rect_w_m"] - med_w).abs() + (geom_hits["rect_l_m"] - med_l).abs()
            if "heading_diff_to_perp_deg" in geom_hits.columns:
                geom_hits["_heading_diff"] = geom_hits["heading_diff_to_perp_deg"].fillna(9999.0)
                rep_row = geom_hits.sort_values(["_heading_diff", "_size_dist"]).iloc[0]
            else:
                rep_row = geom_hits.sort_values("_size_dist").iloc[0]

            rect_w = float(rep_row.get("rect_w_m", 0.0))
            rect_l = float(rep_row.get("rect_l_m", 0.0))
            aspect = float(rep_row.get("aspect", 0.0))
            rectangularity = float(rep_row.get("rectangularity", 0.0))
            heading_diff = float(rep_row.get("heading_diff_to_perp_deg", 0.0))
            inside_ratio = float(rep_row.get("inside_road_ratio", 0.0))

            jitter_p90 = 0.0
            angle_jitter_p90 = 0.0
            if not hits_support.empty:
                centroids = [geom.centroid for geom in hits_support.geometry]
                xs = [pt.x for pt in centroids]
                ys = [pt.y for pt in centroids]
                med_x = float(np.median(xs))
                med_y = float(np.median(ys))
                dists = [Point(med_x, med_y).distance(pt) for pt in centroids]
                jitter_p90 = float(np.percentile(dists, 90)) if dists else 0.0
                diffs = [float(v) for v in hits_support["heading_diff_to_perp_deg"].tolist() if v != ""]
                angle_jitter_p90 = float(np.percentile(diffs, 90)) if diffs else 0.0

            frame_span = 0
            frame_ids_num = [int(str(f)) for f in frames if str(f).isdigit()]
            if frame_ids_num:
                frame_span = max(frame_ids_num) - min(frame_ids_num)

            if inside_ratio < float(final_cfg.get("min_inside_ratio", 0.5)):
                continue
            if heading_diff > float(final_cfg.get("max_heading_diff_deg", 25.0)):
                continue
            if rect_w < float(final_cfg.get("min_rect_w_m", 1.5)) or rect_w > float(final_cfg.get("max_rect_w_m", 30.0)):
                continue
            if rect_l < float(final_cfg.get("min_rect_l_m", 3.0)) or rect_l > float(final_cfg.get("max_rect_l_m", 40.0)):
                continue
            if aspect < float(final_cfg.get("min_aspect", 1.3)) or aspect > float(final_cfg.get("max_aspect", 15.0)):
                continue
            if rectangularity < float(final_cfg.get("min_rectangularity", 0.45)):
                continue
            thin_long_l = float(final_cfg.get("thin_long_rect_l_min", 0.0))
            thin_long_w = float(final_cfg.get("thin_long_rect_w_max", 0.0))
            if thin_long_l > 0 and thin_long_w > 0:
                if rect_l >= thin_long_l and rect_w <= thin_long_w:
                    continue
            if jitter_p90 > float(final_cfg.get("jitter_p90_max", 8.0)) or angle_jitter_p90 > float(
                final_cfg.get("angle_jitter_p90_max", 35.0)
            ):
                continue
            span_max = int(final_cfg.get("frame_span_max", 0))
            if span_max > 0 and frame_span > span_max:
                continue

            entities.append(
                {
                    "geometry": rep_row.geometry,
                    "drive_id": drive_id,
                    "entity_id": f"{drive_id}_crosswalk_{idx:04d}",
                    "frames_hit": frames_hit,
                    "support_frames": json.dumps(sorted(frames), ensure_ascii=True),
                    "rect_w_m": rect_w,
                    "rect_l_m": rect_l,
                    "rectangularity": rectangularity,
                    "heading_diff_to_perp_deg": heading_diff,
                    "inside_road_ratio": inside_ratio,
                    "jitter_p90": jitter_p90,
                    "angle_jitter_p90": angle_jitter_p90,
                    "frame_span": frame_span,
                }
            )

    if not entities:
        return gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    out = gpd.GeoDataFrame(entities, geometry="geometry", crs="EPSG:32632")
    return out


def _update_qa_index_with_final(
    qa_index_path: Path,
    final_gdf: gpd.GeoDataFrame,
) -> None:
    if not qa_index_path.exists():
        return
    qa = gpd.read_file(qa_index_path)
    support_map = {}
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
    qa.to_file(qa_index_path, driver="GeoJSON")

def _write_report(
    out_path: Path,
    stage1_csv: Path,
    final_gdf: gpd.GeoDataFrame,
    reject_samples: List[dict],
    qa_index_path: Path,
    trace_path: Path,
    missing_list_path: Path,
) -> None:
    lines = []
    lines.append("# Crosswalk Full Report\n")
    lines.append(f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    if stage1_csv.exists():
        import pandas as pd

        df = pd.read_csv(stage1_csv)
        lines.append(f"- stage1_candidate_count: {len(df)}\n")
        for drive_id, group in df.groupby("drive_id"):
            lines.append(f"- stage1_{drive_id}_count: {len(group)}")
        lines.append("")
        if "reject_reasons" in df.columns:
            counts = df["reject_reasons"].fillna("").str.split(",", expand=False)
            flat = {}
            for reasons in counts:
                for reason in [r for r in reasons if r]:
                    flat[reason] = flat.get(reason, 0) + 1
            if flat:
                lines.append("## Stage1 Reject Reasons")
                for reason, count in sorted(flat.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {reason}: {count}")
                lines.append("")

    lines.append("## Stage2 Final Summary")
    if final_gdf.empty:
        lines.append("- final_count: 0\n")
    else:
        lines.append(f"- final_count: {len(final_gdf)}")
        for drive_id, group in final_gdf.groupby("drive_id"):
            lines.append(f"- final_{drive_id}_count: {len(group)}")
        lines.append("")
        frames = [int(v) for v in final_gdf["frames_hit"].tolist() if v]
        if frames:
            lines.append(f"- frames_hit_p50: {np.percentile(frames, 50):.1f}")
            lines.append(f"- frames_hit_p90: {np.percentile(frames, 90):.1f}")
        if "rect_w_m" in final_gdf.columns:
            widths = [float(v) for v in final_gdf["rect_w_m"].tolist() if v]
            lengths = [float(v) for v in final_gdf["rect_l_m"].tolist() if v]
            if widths and lengths:
                lines.append(f"- rect_w_p50: {np.percentile(widths, 50):.2f}")
                lines.append(f"- rect_l_p50: {np.percentile(lengths, 50):.2f}")
        lines.append("")
        if qa_index_path.exists():
            qa = gpd.read_file(qa_index_path)
            qa_lookup = {(str(r["drive_id"]), str(r["frame_id"])): r for _, r in qa.iterrows()}
        else:
            qa_lookup = {}
        lines.append("### Final Diagnostics")
        for _, row in final_gdf.iterrows():
            drive_id = str(row.get("drive_id"))
            entity_id = row.get("entity_id")
            jitter = row.get("jitter_p90")
            angle_jitter = row.get("angle_jitter_p90")
            lines.append(f"- {drive_id}:{entity_id} frames_hit={row.get('frames_hit')} jitter_p90={jitter} angle_jitter_p90={angle_jitter}")
            support_frames = []
            raw = row.get("support_frames", "")
            if isinstance(raw, str) and raw:
                try:
                    support_frames = json.loads(raw)
                except Exception:
                    support_frames = []
            for frame_id in support_frames[:3]:
                key = (drive_id, str(frame_id))
                qa_row = qa_lookup.get(key)
                if qa_row is None:
                    continue
                lines.append(
                    f"  - frame {frame_id}: raw={qa_row.get('overlay_raw_path')} gated={qa_row.get('overlay_gated_path')} entities={qa_row.get('overlay_entities_path')}"
                )
        lines.append("")

    lines.append("## Top Reject Samples")
    if not reject_samples:
        lines.append("- none\n")
    else:
        for row in reject_samples[:10]:
            lines.append(
                f"- {row.get('drive_id')}:{row.get('frame_id')} score={row.get('score_total'):.3f} reject={row.get('reject_reasons')}"
            )
    lines.append("")
    if trace_path.exists():
        import pandas as pd

        df = pd.read_csv(trace_path)
        if not df.empty:
            lines.append("## Trace Summary")
            for drive_id, group in df.groupby("drive_id"):
                raw_ratio = float(group["raw_has_crosswalk"].mean())
                project_fail = int((group["project_ok"] == 0).sum())
                candidate_missing = int((group["candidate_written"] == 0).sum())
                lines.append(
                    f"- {drive_id}: raw_has_crosswalk_ratio={raw_ratio:.3f} project_fail={project_fail} candidate_missing={candidate_missing}"
                )
            reasons = df["reject_reasons"].fillna("").str.split(",", expand=False)
            flat = {}
            for items in reasons:
                for reason in [r for r in items if r]:
                    flat[reason] = flat.get(reason, 0) + 1
            if flat:
                lines.append("### Trace Reject Reasons")
                for reason, count in sorted(flat.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {reason}: {count}")
            lines.append("")
    if missing_list_path.exists():
        import pandas as pd

        df = pd.read_csv(missing_list_path)
        lines.append("## Missing Feature Store")
        lines.append(f"- missing_count: {len(df)}")
        lines.append(f"- list_path: {missing_list_path}")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--image-run", required=True)
    ap.add_argument("--image-provider", default="grounded_sam2_v1")
    ap.add_argument("--image-evidence-gpkg", default="")
    ap.add_argument("--road-root", required=True)
    ap.add_argument("--config", default="configs/road_entities.yaml")
    ap.add_argument("--index", default="")
    ap.add_argument("--drives", default="")
    ap.add_argument("--full-stride", type=int, default=5)
    ap.add_argument("--stage2", type=int, default=1)
    ap.add_argument("--min-frames-hit-final", type=int, default=3)
    ap.add_argument("--min-frames-hit-coarse", type=int, default=2)
    ap.add_argument("--stage2-topk-per-drive", type=int, default=5)
    ap.add_argument("--stage2-window-half", type=int, default=40)
    ap.add_argument("--qa-frames-per-entity", type=int, default=20)
    ap.add_argument("--qa-topn-reject", type=int, default=30)
    ap.add_argument("--raw-fallback-mode", default="text_on_image")
    ap.add_argument("--emit-missing-feature-store-list", type=int, default=0)
    args = ap.parse_args()

    log = _setup_logger()
    data_root = Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        log.error("POC_DATA_ROOT not set or invalid.")
        return 2

    run_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"crosswalk_full_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    outputs_dir = run_dir / "outputs"
    debug_dir = run_dir / "debug"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    drive_filter = [d.strip() for d in str(args.drives).split(",") if d.strip()]
    drive_filter = _expand_drive_filter(drive_filter)
    if args.index:
        index_path = Path(args.index)
    else:
        golden8_index = Path("runs") / "image_samples_golden8_full" / "sample_index.jsonl"
        index_path = golden8_index if golden8_index.exists() else (run_dir / "stage1_index.jsonl")
    if not index_path.exists():
        drives = _load_index_drives(Path(args.index)) if args.index else []
        if not drives:
            candidate = data_root / "data_2d_raw"
            if candidate.exists():
                drives = sorted([p.name for p in candidate.iterdir() if p.is_dir()])
        if drive_filter:
            drives = [d for d in drives if d in drive_filter]
        _build_stride_index(data_root, drives, args.full_stride, index_path, "image_00")
    if drive_filter and index_path.exists():
        rows = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("drive_id")) in drive_filter:
                rows.append(row)
        filtered = run_dir / "stage1_index.jsonl"
        _write_index(rows, filtered)
        index_path = filtered

    stage1_cfg = debug_dir / "crosswalk_stage1.yaml"
    _update_config(
        Path(args.config),
        {
            "gates": {"crosswalk_min_inside_ratio": 0.0},
            "crosswalk_final": {
                "min_frames_hit": int(args.min_frames_hit_coarse),
                "min_inside_ratio": 0.3,
                "max_heading_diff_deg": 40.0,
                "min_rectangularity": 0.30,
                "min_rect_w_m": 1.0,
                "max_rect_w_m": 15.0,
                "min_rect_l_m": 2.0,
                "max_rect_l_m": 35.0,
            }
        },
        stage1_cfg,
    )

    stage1_dir = run_dir / "stage1"
    if stage1_dir.exists():
        shutil.rmtree(stage1_dir)
    cmd = [
        sys.executable,
        "tools/build_road_entities.py",
        "--index",
        str(index_path),
        "--image-run",
        str(args.image_run),
        "--image-provider",
        str(args.image_provider),
        "--image-evidence-gpkg",
        str(args.image_evidence_gpkg),
        "--config",
        str(stage1_cfg),
        "--road-root",
        str(args.road_root),
        "--out-dir",
        str(stage1_dir),
        "--emit-qa-images",
        "0",
    ]
    log.info("stage1: %s", " ".join(cmd))
    if subprocess.run(cmd, check=False).returncode != 0:
        log.error("stage1 failed")
        return 3

    stage1_gpkg = stage1_dir / "outputs" / "road_entities_utm32.gpkg"
    cand_gdf = _read_candidates(stage1_gpkg)
    stage1_csv = debug_dir / "stage1_candidates.csv"
    cand_scored = _write_stage1_candidates(cand_gdf, stage1_csv)

    windows = _build_stage2_windows(
        cand_scored,
        debug_dir / "stage2_windows.jsonl",
        eps_m=6.0,
        window_half=int(args.stage2_window_half),
        topk_per_drive=int(args.stage2_topk_per_drive),
    )
    if not windows:
        log.warning("no stage2 windows, skipping stage2")
        return 0

    stage2_index = run_dir / "stage2_index.jsonl"
    _build_stage2_index(data_root, windows, stage2_index, "image_00")

    stage2_cfg = debug_dir / "crosswalk_stage2.yaml"
    _update_config(Path(args.config), {"crosswalk_final": {"min_frames_hit": int(args.min_frames_hit_final)}}, stage2_cfg)

    stage2_dir = run_dir / "stage2"
    if stage2_dir.exists():
        shutil.rmtree(stage2_dir)
    cmd = [
        sys.executable,
        "tools/build_road_entities.py",
        "--index",
        str(stage2_index),
        "--image-run",
        str(args.image_run),
        "--image-provider",
        str(args.image_provider),
        "--image-evidence-gpkg",
        str(args.image_evidence_gpkg),
        "--config",
        str(stage2_cfg),
        "--road-root",
        str(args.road_root),
        "--out-dir",
        str(stage2_dir),
        "--emit-qa-images",
        "1",
    ]
    log.info("stage2: %s", " ".join(cmd))
    if subprocess.run(cmd, check=False).returncode != 0:
        log.error("stage2 failed")
        return 4

    stage2_gpkg = stage2_dir / "outputs" / "road_entities_utm32.gpkg"
    cand_gdf = _read_candidates(stage2_gpkg)
    final_gdf = _rebuild_final_from_candidates(cand_gdf, Path(args.road_root), _load_yaml(Path(args.config)))

    out_gpkg = outputs_dir / "crosswalk_entities_utm32.gpkg"
    _write_crosswalk_gpkg(stage2_gpkg, out_gpkg)
    if out_gpkg.exists() and not final_gdf.empty:
        final_gdf.to_file(out_gpkg, layer="crosswalk_poly", driver="GPKG")

    out_wgs84 = outputs_dir / "crosswalk_entities_wgs84.gpkg"
    if out_wgs84.exists():
        out_wgs84.unlink()
    gdf_wgs84 = gpd.read_file(stage2_dir / "outputs" / "road_entities_wgs84.gpkg", layer="crosswalk_candidate_poly")
    gdf_final = final_gdf.to_crs("EPSG:4326") if not final_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:4326")
    gdf_wgs84.to_file(out_wgs84, layer="crosswalk_candidate_poly", driver="GPKG")
    gdf_final.to_file(out_wgs84, layer="crosswalk_poly", driver="GPKG")

    qa_index_path = stage2_dir / "outputs" / "qa_index_wgs84.geojson"
    per_drive_frames, reject_samples = _select_qa_frames(
        final_gdf,
        cand_gdf,
        args.qa_frames_per_entity,
        args.qa_topn_reject,
    )
    stage2_frames_by_drive = _load_index_frames(stage2_index)
    for drive_id, frames in per_drive_frames.items():
        if args.qa_frames_per_entity <= 0:
            continue
        if len(frames) >= args.qa_frames_per_entity:
            continue
        pool = stage2_frames_by_drive.get(drive_id, [])
        for frame_id in pool:
            if frame_id in frames:
                continue
            frames.append(frame_id)
            if len(frames) >= args.qa_frames_per_entity:
                break
        per_drive_frames[drive_id] = sorted(set(frames))
    if (stage2_dir / "outputs" / "qa_images").exists():
        if (outputs_dir / "qa_images").exists():
            shutil.rmtree(outputs_dir / "qa_images")
        shutil.copytree(stage2_dir / "outputs" / "qa_images", outputs_dir / "qa_images")
    if qa_index_path.exists():
        _filter_qa_assets(qa_index_path, outputs_dir, per_drive_frames)
        qa_index_path = outputs_dir / "qa_index_wgs84.geojson"
        stage2_index_map = _load_index_map(stage2_index)
        stage1_index_map = _load_index_map(index_path)
        merged_index_map = dict(stage1_index_map)
        merged_index_map.update(stage2_index_map)
        raw_stats = _ensure_raw_overlays(
            qa_index_path,
            outputs_dir,
            Path(args.image_run),
            args.image_provider,
            merged_index_map,
            bool(args.emit_missing_feature_store_list),
            args.raw_fallback_mode,
        )
        _update_qa_index_with_final(qa_index_path, final_gdf)

        feature_store_map_root = Path(args.image_run) / f"feature_store_map_{args.image_provider}"
        trace_records = []
        if stage1_index_map:
            stage1_frames: Dict[str, List[str]] = {}
            for drive_id, frames in per_drive_frames.items():
                stage1_frames[drive_id] = [
                    frame_id for frame_id in frames if (drive_id, frame_id) in stage1_index_map
                ]
            stage1_candidates = _read_candidates(stage1_gpkg)
            trace_records.extend(
                _build_trace_records(
                    "stage1",
                    stage1_frames,
                    stage1_candidates,
                    raw_stats,
                    feature_store_map_root,
                )
            )
        trace_records.extend(
            _build_trace_records(
                "stage2",
                per_drive_frames,
                cand_gdf,
                raw_stats,
                feature_store_map_root,
            )
        )
        trace_path = outputs_dir / "crosswalk_trace.csv"
        _write_trace(trace_path, trace_records)

        qa_gdf = gpd.read_file(qa_index_path)
        missing = []
        for _, row in qa_gdf.iterrows():
            raw_status = str(row.get("raw_status") or "")
            for key in ("overlay_raw_path", "overlay_gated_path", "overlay_entities_path"):
                if raw_status == "missing_feature_store" and key in ("overlay_gated_path", "overlay_entities_path"):
                    continue
                path = row.get(key, "")
                if not path or not Path(path).exists():
                    missing.append((row.get("drive_id"), row.get("frame_id"), key))
        if missing:
            for drive_id, frame_id, key in missing[:10]:
                log.error("qa_missing: %s %s %s", drive_id, frame_id, key)
            log.error("qa_missing_total=%d", len(missing))
            return 5

    report_path = outputs_dir / "crosswalk_full_report.md"
    trace_path = outputs_dir / "crosswalk_trace.csv"
    missing_list_path = outputs_dir / "missing_feature_store_list.csv"
    _write_report(
        report_path,
        stage1_csv,
        final_gdf,
        reject_samples,
        outputs_dir / "qa_index_wgs84.geojson",
        trace_path,
        missing_list_path,
    )

    log.info("done: %s", outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
