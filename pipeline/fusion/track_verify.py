from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Point, Polygon

from tools.sam2_video_propagate import (
    build_video_predictor,
    mask_area_px,
    prepare_video_frames,
    propagate_seed,
    save_masks,
)


def _load_image_size(image_path: str) -> Tuple[int, int]:
    if image_path and Path(image_path).exists():
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            pass
    return 0, 0


def _normalize_frame_id(frame_id: str) -> str:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    if not digits:
        return str(frame_id)
    return digits.zfill(10)


def _mask_to_polygon(mask: np.ndarray, max_points: int = 4000) -> Polygon | None:
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask)
    if xs.size < 10:
        return None
    if xs.size > max_points:
        step = int(max(1, xs.size // max_points))
        xs = xs[::step]
        ys = ys[::step]
    pts = [Point(float(x), float(y)) for x, y in zip(xs, ys)]
    hull = MultiPoint(pts).convex_hull
    if hull is None or hull.is_empty or hull.geom_type != "Polygon":
        return None
    return hull


def _mask_to_bbox(mask: np.ndarray) -> List[float] | None:
    if mask is None or mask.size == 0:
        return None
    if mask.ndim >= 3:
        mask = np.squeeze(mask)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    minx = float(np.min(xs))
    maxx = float(np.max(xs))
    miny = float(np.min(ys))
    maxy = float(np.max(ys))
    if maxx <= minx or maxy <= miny:
        return None
    return [minx, miny, maxx, maxy]


def _expand_bbox(bbox: List[float], margin: float, width: int, height: int) -> List[float]:
    minx, miny, maxx, maxy = bbox
    minx = max(0.0, minx - margin)
    miny = max(0.0, miny - margin)
    maxx = min(float(width - 1), maxx + margin)
    maxy = min(float(height - 1), maxy + margin)
    if maxx <= minx:
        maxx = minx + 1.0
    if maxy <= miny:
        maxy = miny + 1.0
    return [minx, miny, maxx, maxy]


def _shrink_bbox_to_max(bbox: List[float], max_side: float, width: int, height: int) -> List[float]:
    minx, miny, maxx, maxy = bbox
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    w = maxx - minx
    h = maxy - miny
    side = max(w, h)
    if side <= max_side:
        return [minx, miny, maxx, maxy]
    scale = max_side / side
    half_w = 0.5 * w * scale
    half_h = 0.5 * h * scale
    minx = max(0.0, cx - half_w)
    maxx = min(float(width - 1), cx + half_w)
    miny = max(0.0, cy - half_h)
    maxy = min(float(height - 1), cy + half_h)
    if maxx <= minx:
        maxx = minx + 1.0
    if maxy <= miny:
        maxy = miny + 1.0
    return [minx, miny, maxx, maxy]


def _crop_mask_to_bbox(mask: np.ndarray, bbox: List[float]) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    if bbox is None or len(bbox) != 4:
        return mask
    minx, miny, maxx, maxy = [int(round(v)) for v in bbox]
    minx = max(0, minx)
    miny = max(0, miny)
    maxx = min(mask.shape[1] - 1, maxx)
    maxy = min(mask.shape[0] - 1, maxy)
    cropped = np.zeros_like(mask, dtype=bool)
    cropped[miny : maxy + 1, minx : maxx + 1] = mask[miny : maxy + 1, minx : maxx + 1]
    return cropped


def _bbox_contains_point(bbox: List[float], pt: Tuple[float, float]) -> bool:
    return bbox[0] <= pt[0] <= bbox[2] and bbox[1] <= pt[1] <= bbox[3]


def _bbox_iou(b1: List[float], b2: List[float]) -> float:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    denom = area1 + area2 - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _rect_metrics(geom: Polygon) -> dict:
    if geom is None or geom.is_empty:
        return {
            "rect_w_m": 0.0,
            "rect_l_m": 0.0,
            "aspect": 0.0,
            "rectangularity": 0.0,
        }
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom is None or geom.is_empty:
        return {
            "rect_w_m": 0.0,
            "rect_l_m": 0.0,
            "aspect": 0.0,
            "rectangularity": 0.0,
        }
    rect = geom.minimum_rotated_rectangle
    if rect is None or rect.is_empty:
        return {
            "rect_w_m": 0.0,
            "rect_l_m": 0.0,
            "aspect": 0.0,
            "rectangularity": 0.0,
        }
    if not rect.is_valid:
        rect = rect.buffer(0)
    rect_coords = list(rect.exterior.coords) if rect is not None and rect.geom_type == "Polygon" else []
    edge_lengths = []
    for i in range(len(rect_coords) - 1):
        p0 = rect_coords[i]
        p1 = rect_coords[i + 1]
        length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if length > 1e-6:
            edge_lengths.append(length)
    rect_l = max(edge_lengths) if edge_lengths else 0.0
    rect_w = min(edge_lengths) if edge_lengths else 0.0
    aspect = rect_l / max(1e-6, rect_w) if rect_w > 0 else 0.0
    rect_area = rect.area if rect is not None else 0.0
    area = geom.area if geom is not None else 0.0
    rectangularity = area / rect_area if rect_area > 0 else 0.0
    return {
        "rect_w_m": rect_w,
        "rect_l_m": rect_l,
        "aspect": aspect,
        "rectangularity": rectangularity,
    }


def _choose_stage2_seeds(
    drive_id: str,
    cluster_id: str,
    candidate_gdf: gpd.GeoDataFrame,
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    raw_frames: Dict[Tuple[str, str], dict],
    index_lookup: Dict[Tuple[str, str], str],
    seed_dir: Path,
    seeds_per_cluster: int,
    min_seed_dist_m: float,
) -> List[dict]:
    if seeds_per_cluster <= 0:
        return []
    if candidate_gdf.empty:
        return []
    subset = candidate_gdf[candidate_gdf.get("cluster_id", "") == cluster_id]
    if subset.empty:
        return []
    scored = []
    for _, row in subset.iterrows():
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not frame_id:
            continue
        raw_info = raw_stats.get((drive_id, frame_id), {})
        score = float(raw_info.get("raw_top_score", 0.0))
        rectangularity = float(row.get("rectangularity") or 0.0)
        gore_like = bool(row.get("gore_like", False))
        base = score + rectangularity
        if gore_like:
            base *= 0.2
        geom = row.get("geometry")
        centroid = geom.centroid if geom is not None else None
        scored.append((base, frame_id, row, centroid))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen: List[dict] = []
    chosen_centroids: List[Point] = []
    for _, frame_id, row, centroid in scored:
        if len(chosen) >= seeds_per_cluster:
            break
        if centroid is not None and chosen_centroids:
            if any(centroid.distance(c) < min_seed_dist_m for c in chosen_centroids):
                continue
        image_path = index_lookup.get((drive_id, frame_id), "")
        bbox_px = row.get("bbox_px")
        if isinstance(bbox_px, str) and bbox_px.startswith("["):
            try:
                bbox_px = json.loads(bbox_px)
            except Exception:
                bbox_px = None
        raw_info = raw_frames.get((drive_id, frame_id), {})
        seed_mask_path = None
        raw_gdf = raw_info.get("gdf")
        if raw_gdf is not None and not raw_gdf.empty and image_path:
            seed_mask_path = _raw_gdf_to_mask(
                raw_gdf,
                image_path,
                seed_dir / f"{cluster_id}_seed_{frame_id}.png",
            )
        chosen.append(
            {
                "cluster_id": cluster_id,
                "seed_frame_id": frame_id,
                "seed_bbox_px": bbox_px,
                "seed_mask_path": str(seed_mask_path) if seed_mask_path else "",
                "seed_score": float(raw_stats.get((drive_id, frame_id), {}).get("raw_top_score", 0.0)),
                "seed_is_gore_like": int(bool(row.get("gore_like", False))),
            }
        )
        if centroid is not None:
            chosen_centroids.append(centroid)
    return chosen


def _raw_gdf_to_mask(
    raw_gdf: gpd.GeoDataFrame,
    image_path: str,
    out_path: Path,
) -> Path | None:
    if raw_gdf is None or raw_gdf.empty:
        return None
    width, height = _load_image_size(image_path)
    if width <= 0 or height <= 0:
        return None
    base = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(base)
    for _, row in raw_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            pts = [(float(x), float(y)) for x, y in geom.exterior.coords]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                pts = [(float(x), float(y)) for x, y in poly.exterior.coords]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_path)
    return out_path if out_path.exists() else None


def _write_seeds_jsonl(seeds: List[dict], out_path: Path) -> None:
    out_path.write_text(
        "\n".join([json.dumps(s, ensure_ascii=True) for s in seeds if s]) + "\n",
        encoding="utf-8",
    )


def run_stage2_track_verify(
    drive_id: str,
    frame_ids: List[str],
    candidate_gdf: gpd.GeoDataFrame,
    cluster_info: Dict[str, dict],
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    raw_frames: Dict[Tuple[str, str], dict],
    index_lookup: Dict[Tuple[str, str], str],
    outputs_dir: Path,
    debug_dir: Path,
    stage2_cfg: dict,
    image_provider: str,
    image_dir: Optional[Path],
) -> Tuple[gpd.GeoDataFrame, Dict[str, dict], Dict[Tuple[str, str], dict], Dict[str, List[dict]]]:
    stage2_stats: Dict[Tuple[str, str], dict] = {}
    stage2_overlay: Dict[str, List[dict]] = {}
    if not stage2_cfg:
        return candidate_gdf, cluster_info, stage2_stats, stage2_overlay

    stage2_window = int(stage2_cfg.get("stage2_window", 50))
    prop_min_area = float(stage2_cfg.get("prop_min_area_px", 400.0))
    prop_max_ratio = float(stage2_cfg.get("prop_max_area_ratio", 0.6))
    prop_min_ratio = float(stage2_cfg.get("prop_min_area_ratio", stage2_cfg.get("AREA_RATIO_MIN", 0.0)))
    rel_area_min = float(stage2_cfg.get("area_ratio_min", stage2_cfg.get("AREA_RATIO_MIN", 0.0)))
    rel_area_max = float(stage2_cfg.get("area_ratio_max", stage2_cfg.get("AREA_RATIO_MAX", 3.0)))
    rect_min_support = float(stage2_cfg.get("rectangularity_min_support", 0.35))
    allow_gore_support = bool(stage2_cfg.get("allow_gore_like_support", False))
    roi_margin_px = int(stage2_cfg.get("roi_margin_px", stage2_cfg.get("ROI_MARGIN_PX", 80)))
    roi_long_ratio = float(stage2_cfg.get("roi_long_ratio_max", 2.5))
    iou_prev_min = float(stage2_cfg.get("iou_prev_min", stage2_cfg.get("IOU_PREV_MIN", 0.05)))
    center_drift_ratio = float(stage2_cfg.get("center_drift_ratio", stage2_cfg.get("CENTER_DRIFT_RATIO", 0.35)))
    drift_consec_stop = int(stage2_cfg.get("drift_consec_stop", stage2_cfg.get("DRIFT_CONSEC_STOP", 3)))

    if image_dir is None or not image_dir.exists():
        return candidate_gdf, cluster_info, stage2_stats, stage2_overlay
    frame_index = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
    stage2_video_dir = debug_dir / "stage2_video_frames"
    prepare_video_frames(image_dir, frame_ids, stage2_video_dir)
    predictor = build_video_predictor(image_provider)

    topk_clusters = int(stage2_cfg.get("topk_clusters", stage2_cfg.get("TOPK_CLUSTERS", 0)))
    cluster_ids = list(cluster_info.keys())
    if topk_clusters > 0 and cluster_ids:
        cluster_ids = sorted(
            cluster_ids,
            key=lambda cid: float(cluster_info.get(cid, {}).get("frames_hit_support", 0)),
            reverse=True,
        )[:topk_clusters]
    seeds = []
    seed_dir = outputs_dir / "stage2_seeds"
    for cid in cluster_ids:
        seeds.extend(
            _choose_stage2_seeds(
                drive_id,
                cid,
                candidate_gdf,
                raw_stats,
                raw_frames,
                index_lookup,
                seed_dir,
                int(stage2_cfg.get("seeds_per_cluster", 1)),
                float(stage2_cfg.get("min_seed_dist_m", 5.0)),
            )
        )
    _write_seeds_jsonl(seeds, outputs_dir / "seeds.jsonl")

    stage2_candidate_rows: List[dict] = []
    if predictor is None:
        return candidate_gdf, cluster_info, stage2_stats, stage2_overlay

    for seed in seeds:
        cid = seed["cluster_id"]
        seed_frame = seed["seed_frame_id"]
        seed_idx = frame_ids.index(seed_frame) if seed_frame in frame_ids else None
        if seed_idx is None:
            continue
        window_start_idx = max(0, seed_idx - stage2_window)
        window_end_idx = min(len(frame_ids) - 1, seed_idx + stage2_window)
        window_start_id = frame_ids[window_start_idx]
        window_end_id = frame_ids[window_end_idx]
        seed_mask_path = Path(seed["seed_mask_path"]) if seed.get("seed_mask_path") else None
        seed_bbox = seed.get("seed_bbox_px")
        if seed_bbox is None and seed_mask_path is not None and seed_mask_path.exists():
            seed_mask = np.array(Image.open(seed_mask_path).convert("L")) > 0
            seed_bbox = _mask_to_bbox(seed_mask)
        seed_bbox = seed_bbox if isinstance(seed_bbox, list) else None
        masks = propagate_seed(
            predictor,
            stage2_video_dir,
            seed_idx,
            seed_mask_path,
            seed_bbox,
            stage2_window,
        )
        out_mask_dir = outputs_dir / "stage2_masks" / cid
        saved = save_masks(masks, frame_ids, out_mask_dir)
        if not saved:
            continue
        image_path = index_lookup.get((drive_id, seed_frame), "")
        width, height = _load_image_size(image_path)
        roi_bbox = None
        roi_too_long = False
        if seed_bbox is not None and width > 0 and height > 0 and roi_margin_px > 0:
            roi_bbox = _expand_bbox(seed_bbox, float(roi_margin_px), width, height)
            seed_w = float(seed_bbox[2] - seed_bbox[0])
            seed_h = float(seed_bbox[3] - seed_bbox[1])
            seed_long = max(seed_w, seed_h)
            roi_w = float(roi_bbox[2] - roi_bbox[0])
            roi_h = float(roi_bbox[3] - roi_bbox[1])
            roi_long = max(roi_w, roi_h)
            if seed_long > 1.0 and roi_long > seed_long * roi_long_ratio:
                roi_bbox = _shrink_bbox_to_max(roi_bbox, seed_long * roi_long_ratio, width, height)
                roi_too_long = True
        prev_bbox = None
        prev_area = None
        consec_drift = 0
        for frame_id, mask_path in sorted(saved.items(), key=lambda item: frame_index.get(item[0], 1_000_000)):
            frame_idx = frame_index.get(frame_id)
            if frame_idx is None or frame_idx < window_start_idx or frame_idx > window_end_idx:
                continue
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            if roi_bbox is not None and roi_too_long:
                mask = _crop_mask_to_bbox(mask, roi_bbox)
            area = mask_area_px(mask)
            image_path = index_lookup.get((drive_id, frame_id), "")
            width, height = _load_image_size(image_path)
            image_area = float(width * height) if width > 0 and height > 0 else 0.0
            area_ratio = (area / image_area) if image_area > 0 else 0.0
            area_max = image_area * prop_max_ratio if image_area > 0 else 0.0
            mask_bbox = _mask_to_bbox(mask)
            drift_reasons = []
            if area < prop_min_area:
                drift_reasons.append("area_min")
            if image_area > 0 and area_ratio < prop_min_ratio:
                drift_reasons.append("area_ratio_min")
            if area_max > 0.0 and area > area_max:
                drift_reasons.append("area_ratio_max")
            if mask_bbox is None:
                drift_reasons.append("bbox_missing")
            if roi_bbox is not None and mask_bbox is not None:
                center = _bbox_center(mask_bbox)
                if not _bbox_contains_point(roi_bbox, center):
                    drift_reasons.append("roi_center")
            if roi_too_long:
                drift_reasons.append("roi_too_long")
            if prev_bbox is not None and mask_bbox is not None:
                if _bbox_iou(prev_bbox, mask_bbox) < iou_prev_min:
                    drift_reasons.append("iou_prev")
                center = _bbox_center(mask_bbox)
                prev_center = _bbox_center(prev_bbox)
                max_dim = float(max(width, height)) if width > 0 and height > 0 else 0.0
                if roi_bbox is not None:
                    max_dim = float(max(roi_bbox[2] - roi_bbox[0], roi_bbox[3] - roi_bbox[1]))
                if max_dim > 0:
                    dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
                    if dist / max_dim > center_drift_ratio:
                        drift_reasons.append("center_drift")
            if prev_area is not None and prev_area > 0:
                ratio = area / prev_area if prev_area > 0 else 0.0
                if ratio < rel_area_min or ratio > rel_area_max:
                    drift_reasons.append("area_ratio_prev")
            prop_ok = int(len(drift_reasons) == 0)
            prop_drift = int(len(drift_reasons) > 0)
            if prop_drift:
                consec_drift += 1
            else:
                consec_drift = 0
            if drift_consec_stop > 0 and consec_drift >= drift_consec_stop:
                break
            key = (drive_id, frame_id)
            stat = stage2_stats.setdefault(
                key,
                {
                    "cluster_id": cid,
                    "stage2_added": 0,
                    "prop_ok": 0,
                    "prop_area_px": 0.0,
                    "prop_drift_flag": 0,
                    "prop_reason": "",
                    "prop_window_start": window_start_id,
                    "prop_window_end": window_end_id,
                },
            )
            stat["prop_window_start"] = window_start_id
            stat["prop_window_end"] = window_end_id
            stat["cluster_id"] = cid
            stat["prop_area_px"] = area
            stat["prop_drift_flag"] = prop_drift
            stat["prop_ok"] = prop_ok
            stat["prop_reason"] = ",".join(drift_reasons)
            if prop_ok:
                stat["stage2_added"] = 1
                prev_bbox = mask_bbox or prev_bbox
                prev_area = area
                info = cluster_info.get(cid, {})
                info["stage2_added_frames_count"] = int(info.get("stage2_added_frames_count", 0)) + 1
                poly = _mask_to_polygon(mask)
                rect_val = 0.0
                if poly is not None:
                    metrics = _rect_metrics(poly)
                    rect_val = float(metrics.get("rectangularity", 0.0))
                is_gore = False
                if not allow_gore_support:
                    existing = candidate_gdf[candidate_gdf.get("frame_id", "") == frame_id]
                    if not existing.empty:
                        is_gore = bool(existing.get("gore_like", False).fillna(False).any())
                if rect_val >= rect_min_support and not is_gore:
                    info.setdefault("stage2_support_frames", []).append(frame_id)
                bbox = None
                if poly is not None:
                    minx, miny, maxx, maxy = poly.bounds
                    bbox = [float(minx), float(miny), float(maxx), float(maxy)]
                stage2_overlay.setdefault(frame_id, []).append(
                    {"mask_path": str(mask_path), "bbox_px": bbox, "cluster_id": cid}
                )
                info["stage2_support_frames"] = list(dict.fromkeys(info.get("stage2_support_frames", [])))
                support_frames = info.get("support_frames", [])
                merged_support = list(dict.fromkeys(support_frames + info.get("stage2_support_frames", [])))
                info["support_frames"] = merged_support
                info["frames_hit_support"] = len(merged_support)
                info["frames_hit_all"] = int(info.get("frames_hit_all_before", 0)) + int(
                    info.get("stage2_added_frames_count", 0)
                )
                if bbox is None and seed_bbox:
                    bbox = seed_bbox
                if bbox:
                    stage2_candidate_rows.append(
                        {
                            "geometry": info.get("refined_geom"),
                            "properties": {
                                "candidate_id": f"{drive_id}_crosswalk_stage2_{cid}_{frame_id}",
                                "drive_id": drive_id,
                                "frame_id": frame_id,
                                "source_frame_id": seed_frame,
                                "entity_type": "crosswalk",
                                "reject_reasons": "stage2_video",
                                "proj_method": "stage2_video",
                                "geom_ok": 0,
                                "stage2_added": 1,
                                "bbox_px": bbox,
                                "cluster_id": cid,
                                "qa_flag": "stage2_added",
                            },
                        }
                    )
    if stage2_candidate_rows:
        stage2_gdf = gpd.GeoDataFrame.from_features(stage2_candidate_rows, crs="EPSG:32632")
        candidate_gdf = gpd.GeoDataFrame(
            pd.concat([candidate_gdf, stage2_gdf], ignore_index=True),
            geometry="geometry",
            crs="EPSG:32632",
        )
    return candidate_gdf, cluster_info, stage2_stats, stage2_overlay
