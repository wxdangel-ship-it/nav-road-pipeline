from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline._io import load_yaml
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


CFG_DEFAULT = Path("configs/image_stage12_ensemble_gdino_yoloworld_0010_f000_300.yaml")


def _cfg_get(cfg: dict, key: str, default=None):
    for k in (key, key.lower(), key.upper()):
        if k in cfg:
            return cfg[k]
    return default


def _cfg_list(cfg: dict, key: str, default=None) -> List:
    val = _cfg_get(cfg, key, default or [])
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def _clamp01(val: float) -> float:
    try:
        v = float(val)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _find_data_root(cfg_root: str) -> Path:
    if cfg_root:
        path = Path(cfg_root)
        if path.exists():
            return path
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        path = Path(env_root)
        if path.exists():
            return path
    default_root = Path(r"E:\\KITTI360\\KITTI-360")
    if default_root.exists():
        return default_root
    raise SystemExit("missing data root: set POC_DATA_ROOT or config.KITTI_ROOT")


def _find_image_dir(data_root: Path, drive: str, camera: str) -> Path:
    candidates = [
        data_root / "data_2d_raw" / drive / camera / "data_rect",
        data_root / "data_2d_raw" / drive / camera / "data",
        data_root / drive / camera / "data_rect",
        data_root / drive / camera / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"image data not found for drive={drive} camera={camera}")


def _find_frame_path(image_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        path = image_dir / f"{frame_id}{ext}"
        if path.exists():
            return path
    return None


def _collect_frames(image_dir: Path, frame_start: int, frame_end: int) -> Tuple[List[str], Dict[str, Path], List[str]]:
    existing: List[str] = []
    mapping: Dict[str, Path] = {}
    missing: List[str] = []
    for frame in range(frame_start, frame_end + 1):
        frame_id = f"{frame:010d}"
        path = _find_frame_path(image_dir, frame_id)
        if path is None:
            missing.append(frame_id)
            continue
        existing.append(frame_id)
        mapping[frame_id] = path
    return existing, mapping, missing


def _select_qa_frames(
    frame_start: int,
    frame_end: int,
    seed: int,
    sample_n: int,
    force_include: List[int],
) -> List[int]:
    rng = random.Random(seed)
    pool = list(range(frame_start, frame_end + 1))
    sampled = rng.sample(pool, min(sample_n, len(pool))) if pool else []
    merged = set(sampled)
    merged.update(int(v) for v in force_include)
    merged = {v for v in merged if frame_start <= v <= frame_end}
    return sorted(merged)


def _load_model_cfg(zoo_path: Path, model_id: str) -> dict:
    zoo = load_yaml(zoo_path) if zoo_path.exists() else {}
    models = zoo.get("models") or []
    for m in models:
        if str(m.get("model_id") or "") == model_id:
            return m
    return {}


def _bounded_find(root: Path, filename: str, max_depth: int = 6) -> Optional[Path]:
    if not root.exists():
        return None
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).resolve().relative_to(root)
        if len(rel.parts) > max_depth:
            dirnames[:] = []
            continue
        if filename in filenames:
            return Path(dirpath) / filename
    return None


def _resolve_sam2_ckpt(model_cfg: dict) -> Optional[Path]:
    download_cfg = model_cfg.get("download") or {}
    ckpt_name = str(download_cfg.get("sam2_checkpoint") or "")
    if not ckpt_name:
        return None
    ckpt_path = Path(ckpt_name)
    if ckpt_path.exists():
        return ckpt_path
    candidates = [
        Path("cache") / "hf" / "sam2" / ckpt_name,
        Path("cache") / "model_weights" / "sam2" / ckpt_name,
        Path("runs") / "cache_ps" / "sam2" / ckpt_name,
        Path("runs") / "hf_cache_user" / "sam2" / ckpt_name,
    ]
    for env_key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        base = os.environ.get(env_key)
        if base:
            candidates.append(Path(base) / ckpt_name)
            found = _bounded_find(Path(base), ckpt_name)
            if found:
                return found
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _resolve_yoloworld_weights(model_cfg: dict) -> Optional[Path]:
    weights = (model_cfg.get("download") or {}).get("weights") or "yolov8s-worldv2.pt"
    if not weights:
        return None
    weights_path = Path(str(weights))
    if weights_path.exists():
        return weights_path
    candidates = [
        Path("cache") / "model_weights" / "yolo" / weights,
        Path("cache") / "model_weights" / weights,
        Path("cache") / "ultralytics" / weights,
        Path("runs") / "cache_ps" / weights,
        Path("runs") / "cache_ps" / "yolo" / weights,
        Path(weights),
    ]
    for env_key in ("ULTRALYTICS_CACHE_DIR", "YOLO_CACHE_DIR"):
        base = os.environ.get(env_key)
        if base:
            candidates.append(Path(base) / weights)
            found = _bounded_find(Path(base), weights)
            if found:
                return found
    for cand in candidates:
        if cand.exists():
            return cand
    found = _bounded_find(Path("."), weights)
    return found


def _check_gdino_weights(model_cfg: dict) -> List[str]:
    missing: List[str] = []
    dino_dir = model_cfg.get("dino_weights_path")
    dino_file = model_cfg.get("dino_filename")
    if dino_dir:
        dino_dir = Path(str(dino_dir))
        if not dino_dir.exists():
            missing.append(f"gdino_weights_dir:{dino_dir}")
        elif dino_file and not (dino_dir / dino_file).exists():
            missing.append(f"gdino_weights_file:{dino_dir / dino_file}")
    else:
        missing.append("gdino_weights_dir:unset")
    ckpt = _resolve_sam2_ckpt(model_cfg)
    if ckpt is None or not ckpt.exists():
        missing.append("sam2_checkpoint:not_found")
    return missing


def _check_yoloworld_weights(model_cfg: dict) -> List[str]:
    missing: List[str] = []
    weights = _resolve_yoloworld_weights(model_cfg)
    if weights is None or not weights.exists():
        missing.append("yoloworld_weights:not_found")
    return missing


def _device_auto() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: List[List[float]], scores: List[float], iou_thr: float) -> List[int]:
    if not boxes:
        return []
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        idxs = [i for i in idxs if _bbox_iou(boxes[cur], boxes[i]) < iou_thr]
    return keep


def _nms_merge(
    boxes: List[List[float]],
    scores: List[float],
    model_tags: List[str],
    iou_thr: float,
) -> List[Dict[str, object]]:
    if not boxes:
        return []
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    merged: List[Dict[str, object]] = []
    while idxs:
        cur = idxs.pop(0)
        cluster = [cur]
        rest = []
        for i in idxs:
            if _bbox_iou(boxes[cur], boxes[i]) >= iou_thr:
                cluster.append(i)
            else:
                rest.append(i)
        idxs = rest
        support = {model_tags[i] for i in cluster if i < len(model_tags)}
        merged.append(
            {
                "bbox": boxes[cur],
                "score": float(scores[cur]) if cur < len(scores) else 0.0,
                "cluster": cluster,
                "support_models": sorted(support),
                "support_count": len(support),
                "support_scores": [float(scores[i]) for i in cluster if i < len(scores)],
                "support_tags": [model_tags[i] for i in cluster if i < len(model_tags)],
            }
        )
    return merged


def _roi_crop(img: Image.Image, bottom_ratio: float, side_crop: Tuple[float, float]) -> Tuple[Image.Image, Tuple[int, int]]:
    w, h = img.size
    x0 = int(w * float(side_crop[0]))
    x1 = int(w * float(side_crop[1]))
    y0 = int(h * float(bottom_ratio))
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    return img.crop((x0, y0, x1, h)), (x0, y0)


def _draw_overlay_stage1_boxes(
    img: Image.Image,
    gdino_boxes: List[List[float]],
    gdino_scores: List[float],
    yolo_boxes: List[List[float]],
    yolo_scores: List[float],
    merged_boxes: List[List[float]],
    merged_scores: List[float],
    out_path: Path,
) -> None:
    overlay = img.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    for idx, box in enumerate(gdino_boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        score = gdino_scores[idx] if idx < len(gdino_scores) else None
        label = f"gdino:{score:.2f}" if score is not None else "gdino"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64, 200), width=2)
        draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 0, 255))
    for idx, box in enumerate(yolo_boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        score = yolo_scores[idx] if idx < len(yolo_scores) else None
        label = f"yolo:{score:.2f}" if score is not None else "yolo"
        draw.rectangle([x1, y1, x2, y2], outline=(64, 128, 255, 200), width=2)
        draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 255, 255))
    for idx, box in enumerate(merged_boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        score = merged_scores[idx] if idx < len(merged_scores) else None
        label = f"merge:{score:.2f}" if score is not None else "merge"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0, 220), width=2)
        draw.text((x1 + 2, y1 + 14), label, fill=(255, 255, 0, 255))
    overlay.convert("RGB").save(out_path)


def _draw_overlay_stage1_seeds(
    img: Image.Image,
    masks: List[np.ndarray],
    out_path: Path,
) -> None:
    overlay = img.convert("RGBA")
    if masks:
        color = (0, 255, 0, 90)
        mask_canvas = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        arr = np.array(mask_canvas)
        merged = np.zeros((img.size[1], img.size[0]), dtype=bool)
        for m in masks:
            if m.shape[:2] != merged.shape:
                continue
            merged |= m.astype(bool)
        arr[merged] = color
        mask_canvas = Image.fromarray(arr, mode="RGBA")
        overlay = Image.alpha_composite(overlay, mask_canvas)
    overlay.convert("RGB").save(out_path)


def _draw_overlay_stage2(img: Image.Image, mask: np.ndarray, out_path: Path, label: str) -> None:
    overlay = img.convert("RGBA")
    if mask is not None and mask.size:
        if mask.ndim >= 3:
            mask = np.squeeze(mask)
        color = (0, 128, 255, 90)
        mask_canvas = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        arr = np.array(mask_canvas)
        sel = mask.astype(bool)
        if sel.shape == (overlay.size[1], overlay.size[0]):
            arr[sel] = color
        mask_canvas = Image.fromarray(arr, mode="RGBA")
        overlay = Image.alpha_composite(overlay, mask_canvas)
    draw = ImageDraw.Draw(overlay)
    draw.text((8, 8), label, fill=(255, 255, 255, 255))
    overlay.convert("RGB").save(out_path)


def _write_mask(path: Path, mask: np.ndarray, shape: Tuple[int, int]) -> None:
    if mask is None or mask.size == 0:
        img = Image.fromarray(np.zeros((shape[1], shape[0]), dtype=np.uint8))
    else:
        if mask.ndim >= 3:
            mask = np.squeeze(mask)
        m = mask.astype(np.uint8) * 255
        img = Image.fromarray(m)
    img.save(path)


def _montage(images: List[Tuple[str, Path]], out_path: Path, cols: int = 4) -> None:
    if not images:
        return
    loaded = []
    for label, path in images:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), label, fill=(255, 255, 255))
        loaded.append(img)
    if not loaded:
        return
    cols = max(1, cols)
    rows = (len(loaded) + cols - 1) // cols
    w, h = loaded[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for idx, img in enumerate(loaded):
        r = idx // cols
        c = idx % cols
        canvas.paste(img, (c * w, r * h))
    canvas.save(out_path)

def _connected_components(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if mask.size == 0:
        return []
    try:
        import cv2

        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        boxes = []
        for idx in range(1, num):
            x, y, w, h, _area = stats[idx]
            boxes.append((int(x), int(y), int(w), int(h)))
        return boxes
    except Exception:
        pass
    try:
        from scipy.ndimage import label as ndi_label

        labels, num = ndi_label(mask.astype(bool))
        boxes = []
        for idx in range(1, num + 1):
            ys, xs = np.where(labels == idx)
            if ys.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
        return boxes
    except Exception:
        pass
    visited = np.zeros(mask.shape, dtype=bool)
    boxes = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            minx = maxx = x
            miny = maxy = y
            while stack:
                cy, cx = stack.pop()
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= h:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= w:
                            continue
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            boxes.append((minx, miny, maxx - minx + 1, maxy - miny + 1))
    return boxes


def _stripe_count(mask: np.ndarray, aspect_min: float) -> int:
    boxes = _connected_components(mask)
    count = 0
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect >= aspect_min:
            count += 1
    return count


def _mask_mrr_aspect(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    try:
        import cv2

        ys, xs = np.where(mask)
        if xs.size < 3:
            return 0.0
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        w, h = rect[1]
        if w <= 0 or h <= 0:
            return 0.0
        return max(w, h) / max(1.0, min(w, h))
    except Exception:
        ys, xs = np.where(mask)
        if xs.size < 3:
            return 0.0
        w = float(xs.max() - xs.min() + 1)
        h = float(ys.max() - ys.min() + 1)
        if w <= 0 or h <= 0:
            return 0.0
        return max(w, h) / max(1.0, min(w, h))


def _filter_stage1_boxes(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[str],
    image_h: int,
    aspect_min: float,
    h_ratio_max: float,
    center_v_min: float,
) -> Tuple[List[List[float]], List[float], List[str], List[Dict[str, float]]]:
    kept_boxes: List[List[float]] = []
    kept_scores: List[float] = []
    kept_labels: List[str] = []
    metrics: List[Dict[str, float]] = []
    for bbox, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        aspect = w / h
        h_ratio = h / max(1.0, float(image_h))
        center_v_ratio = (0.5 * (y1 + y2)) / max(1.0, float(image_h))
        if aspect < aspect_min:
            continue
        if h_ratio > h_ratio_max:
            continue
        if center_v_ratio < center_v_min:
            continue
        kept_boxes.append(bbox)
        kept_scores.append(score)
        kept_labels.append(label)
        metrics.append(
            {
                "box_aspect": aspect,
                "box_h_ratio": h_ratio,
                "box_center_v_ratio": center_v_ratio,
            }
        )
    return kept_boxes, kept_scores, kept_labels, metrics


def _run_stage1_on_image(
    img: Image.Image,
    gdino_processor,
    gdino_model,
    yolo_model,
    sam2_predictor,
    stage1_cfg: Dict[str, object],
    score_cfg: Dict[str, float],
) -> Tuple[
    List[List[float]],
    List[float],
    List[List[float]],
    List[float],
    List[List[float]],
    List[float],
    List[List[float]],
    List[str],
    List[Dict[str, object]],
    Dict[str, int],
]:
    import torch

    roi_img, roi_offset = _roi_crop(
        img,
        float(stage1_cfg.get("roi_bottom_crop", 0.50)),
        tuple(stage1_cfg.get("roi_side_crop", [0.05, 0.95])),
    )
    offset_x, offset_y = roi_offset

    gdino_prompts = [str(p).strip().lower() for p in (stage1_cfg.get("gdino_text_prompt") or []) if str(p).strip()]
    inputs = gdino_processor(images=roi_img, text=[gdino_prompts], return_tensors="pt").to(gdino_model.device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    target_sizes = [(roi_img.size[1], roi_img.size[0])]
    box_th = float(stage1_cfg.get("gdino_box_th", 0.23))
    text_th = float(stage1_cfg.get("gdino_text_th", 0.23))
    try:
        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_th,
            text_threshold=text_th,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_th,
            text_threshold=text_th,
            target_sizes=target_sizes,
        )[0]

    gdino_boxes_raw = results.get("boxes", [])
    gdino_scores_raw = results.get("scores", [])

    gdino_boxes: List[List[float]] = []
    gdino_scores: List[float] = []
    for idx, box in enumerate(gdino_boxes_raw):
        bbox = [float(v) for v in (box.tolist() if hasattr(box, "tolist") else box)]
        bbox[0] += offset_x
        bbox[2] += offset_x
        bbox[1] += offset_y
        bbox[3] += offset_y
        gdino_boxes.append(bbox)
        gdino_scores.append(float(gdino_scores_raw[idx]) if idx < len(gdino_scores_raw) else 0.0)

    gdino_topk = int(stage1_cfg.get("gdino_topk", 12))
    if gdino_topk > 0 and len(gdino_scores) > gdino_topk:
        order = sorted(range(len(gdino_scores)), key=lambda i: gdino_scores[i], reverse=True)[:gdino_topk]
        gdino_boxes = [gdino_boxes[i] for i in order]
        gdino_scores = [gdino_scores[i] for i in order]

    yolo_conf = float(stage1_cfg.get("yoloworld_conf_th", 0.25))
    yolo_topk = int(stage1_cfg.get("yoloworld_topk", 30))
    yolo_iou = float(stage1_cfg.get("yoloworld_iou_th", 0.5))
    yolo_imgsz = int(stage1_cfg.get("yoloworld_imgsz", 1024))
    yolo_device = stage1_cfg.get("yoloworld_device") or None

    yolo_neg_boxes: List[List[float]] = []
    yolo_neg_labels: List[str] = []
    yolo_neg_prompts = stage1_cfg.get("yolo_neg_prompts") or []
    if yolo_neg_prompts and hasattr(yolo_model, "set_classes"):
        yolo_model.set_classes([str(p) for p in yolo_neg_prompts])
    if yolo_neg_prompts:
        yolo_results = yolo_model.predict(
            source=np.array(roi_img),
            conf=yolo_conf,
            iou=yolo_iou,
            imgsz=yolo_imgsz,
            device=yolo_device,
            verbose=False,
        )
        if yolo_results:
            res = yolo_results[0]
            boxes = res.boxes
            names = res.names or {}
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
                if isinstance(names, list) and 0 <= cls_id < len(names):
                    label = str(names[cls_id])
                else:
                    label = str(getattr(names, "get", lambda _k, _d="": _d)(cls_id, ""))
                label = label.strip().lower()
                bbox = [float(x) for x in xyxy]
                bbox[0] += offset_x
                bbox[2] += offset_x
                bbox[1] += offset_y
                bbox[3] += offset_y
                yolo_neg_boxes.append(bbox)
                yolo_neg_labels.append(label)

    yolo_boxes: List[List[float]] = []
    yolo_scores: List[float] = []
    use_crosswalk = bool(stage1_cfg.get("yolo_use_crosswalk", False))
    yolo_cross_prompts = stage1_cfg.get("yoloworld_text_prompt") or []
    if use_crosswalk and yolo_cross_prompts and hasattr(yolo_model, "set_classes"):
        yolo_model.set_classes([str(p) for p in yolo_cross_prompts])
        yolo_results = yolo_model.predict(
            source=np.array(roi_img),
            conf=yolo_conf,
            iou=yolo_iou,
            imgsz=yolo_imgsz,
            device=yolo_device,
            verbose=False,
        )
        if yolo_results:
            res = yolo_results[0]
            boxes = res.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
                bbox = [float(x) for x in xyxy]
                bbox[0] += offset_x
                bbox[2] += offset_x
                bbox[1] += offset_y
                bbox[3] += offset_y
                yolo_boxes.append(bbox)
                yolo_scores.append(conf)

        if yolo_topk > 0 and len(yolo_scores) > yolo_topk:
            order = sorted(range(len(yolo_scores)), key=lambda i: yolo_scores[i], reverse=True)[:yolo_topk]
            yolo_boxes = [yolo_boxes[i] for i in order]
            yolo_scores = [yolo_scores[i] for i in order]

    all_boxes = gdino_boxes + yolo_boxes
    all_scores = gdino_scores + yolo_scores
    all_tags = ["gdino"] * len(gdino_boxes) + (["yolo"] * len(yolo_boxes) if yolo_boxes else [])
    nms_iou = float(stage1_cfg.get("nms_iou_th", 0.5))
    merged = _nms_merge(all_boxes, all_scores, all_tags, nms_iou)
    merged = sorted(merged, key=lambda m: float(m.get("score", 0.0)), reverse=True)
    max_merge = int(stage1_cfg.get("max_boxes_after_merge", 12))
    if max_merge > 0 and len(merged) > max_merge:
        merged = merged[:max_merge]

    merged_boxes = [m["bbox"] for m in merged]
    merged_scores = [float(m.get("score", 0.0)) for m in merged]

    box_aspect_min = float(stage1_cfg.get("box_aspect_min", 2.0))
    box_h_ratio_max = float(stage1_cfg.get("box_h_ratio_max", 0.35))
    box_center_v_min = float(stage1_cfg.get("box_center_v_min", 0.50))
    filtered_boxes: List[List[float]] = []
    filtered_scores: List[float] = []
    filtered_support: List[Dict[str, object]] = []
    box_metrics: List[Dict[str, float]] = []
    for idx, bbox in enumerate(merged_boxes):
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        aspect = w / h
        h_ratio = h / max(1.0, float(img.height))
        center_v_ratio = (0.5 * (y1 + y2)) / max(1.0, float(img.height))
        if aspect < box_aspect_min or h_ratio > box_h_ratio_max or center_v_ratio < box_center_v_min:
            continue
        filtered_boxes.append(bbox)
        filtered_scores.append(merged_scores[idx] if idx < len(merged_scores) else 0.0)
        filtered_support.append(merged[idx])
        box_metrics.append({"box_aspect": aspect, "box_h_ratio": h_ratio, "box_center_v_ratio": center_v_ratio})

    sam2_predictor.set_image(np.array(img))
    max_masks = int(stage1_cfg.get("sam2_image_max_masks", 2))
    min_area = float(stage1_cfg.get("sam2_mask_min_area_px", 600))
    stripe_aspect_min = float(stage1_cfg.get("stripe_aspect_ratio_min", 2.0))
    stripe_min = int(stage1_cfg.get("stripe_min_count", 6))
    mrr_aspect_min = float(stage1_cfg.get("mask_mrr_aspect_min", 2.0))
    candidates: List[Dict[str, object]] = []
    masks_total = 0
    roi_ratio = float(stage1_cfg.get("roi_bottom_crop", 0.50))
    roi_side = tuple(stage1_cfg.get("roi_side_crop", [0.05, 0.95]))
    roi_x0 = int(img.width * roi_side[0])
    roi_x1 = int(img.width * roi_side[1])
    roi_y0 = int(img.height * roi_ratio)

    neg_iou_thr = float(stage1_cfg.get("yolo_neg_iou_th", 0.3))
    neg_score_mult = float(stage1_cfg.get("yolo_neg_score_mult", 0.3))
    for bbox, score, support, metrics in zip(filtered_boxes, filtered_scores, filtered_support, box_metrics):
        yolo_neg_iou = 0.0
        yolo_neg_hit = 0
        center_in_person = False
        if yolo_neg_boxes:
            cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
            cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
            for yb, label in zip(yolo_neg_boxes, yolo_neg_labels):
                iou = _bbox_iou(bbox, yb)
                if iou > yolo_neg_iou:
                    yolo_neg_iou = iou
                if "person" in str(label).lower():
                    if yb[0] <= cx <= yb[2] and yb[1] <= cy <= yb[3]:
                        center_in_person = True
        if center_in_person:
            continue
        if yolo_neg_iou >= neg_iou_thr:
            yolo_neg_hit = 1
        try:
            masks, _, _ = sam2_predictor.predict(box=np.array(bbox), multimask_output=True)
        except Exception:
            continue
        if masks is None:
            continue
        for m in masks[:max_masks]:
            masks_total += 1
            mask_bin = m.astype(bool)
            area = float(np.sum(mask_bin))
            if area < min_area:
                continue
            mrr_aspect = _mask_mrr_aspect(mask_bin)
            if mrr_aspect < mrr_aspect_min:
                continue
            stripe_count = _stripe_count(mask_bin, stripe_aspect_min)
            if stripe_count < stripe_min:
                continue
            roi_mask = mask_bin.copy()
            roi_mask[:roi_y0, :] = False
            if roi_x0 > 0:
                roi_mask[:, :roi_x0] = False
            if roi_x1 < img.width:
                roi_mask[:, roi_x1:] = False
            in_roi_ratio = float(np.sum(roi_mask)) / max(1.0, area)
            support_count = int(support.get("support_count", 0))
            model_score = 0.0
            support_scores = support.get("support_scores") or []
            if support_scores:
                model_score = max(float(v) for v in support_scores)
            support_norm = min(1.0, support_count / 2.0)
            stripe_ok = 1.0 if stripe_count >= stripe_min else 0.2
            area_ok = 1.0 if area >= min_area else 0.2
            aspect_ok = 1.0 if mrr_aspect >= mrr_aspect_min else 0.2
            seed_score_before = (
                score_cfg["w_support"] * support_norm
                + score_cfg["w_model_score"] * _clamp01(model_score)
                + score_cfg["w_in_roi"] * _clamp01(in_roi_ratio)
                + score_cfg["w_stripe"] * stripe_ok
                + score_cfg["w_area"] * area_ok
                + score_cfg["w_aspect"] * aspect_ok
            )
            seed_score_after = seed_score_before * (neg_score_mult if yolo_neg_hit else 1.0)
            candidates.append(
                {
                    "mask": mask_bin,
                    "area": area,
                    "stripe_count": stripe_count,
                    "score": float(score),
                    "model_score": float(model_score),
                    "support_models": support.get("support_models", []),
                    "support_count": support_count,
                    "bbox": bbox,
                    "mask_mrr_aspect": mrr_aspect,
                    "in_roi_ratio": in_roi_ratio,
                    "seed_score_before": seed_score_before,
                    "seed_score_after": seed_score_after,
                    "stripe_ok": stripe_ok,
                    "area_ok": area_ok,
                    "aspect_ok": aspect_ok,
                    "yolo_neg_hit": yolo_neg_hit,
                    "yolo_neg_iou_max": yolo_neg_iou,
                    **metrics,
                }
            )

    stats = {
        "gdino_boxes": len(gdino_boxes),
        "yolo_boxes": len(yolo_boxes) if use_crosswalk else len(yolo_neg_boxes),
        "merged_boxes": len(merged_boxes),
        "boxes_pass": len(filtered_boxes),
        "masks_total": masks_total,
        "masks_pass": len(candidates),
    }
    return (
        gdino_boxes,
        gdino_scores,
        yolo_boxes,
        yolo_scores,
        merged_boxes,
        merged_scores,
        yolo_neg_boxes,
        yolo_neg_labels,
        candidates,
        stats,
    )


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _segment_count(mask_flags: List[int]) -> int:
    count = 0
    in_seg = False
    for v in mask_flags:
        if v and not in_seg:
            count += 1
            in_seg = True
        elif not v:
            in_seg = False
    return count


def _closest_existing_frame(existing_frames: List[str], target: int) -> Optional[int]:
    if not existing_frames:
        return None
    values = [int(f) for f in existing_frames]
    return min(values, key=lambda v: abs(v - target))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_DEFAULT))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if not cfg:
        raise SystemExit(f"missing config: {cfg_path}")

    drive_id = str(_cfg_get(cfg, "DRIVE_ID", "2013_05_28_drive_0010_sync"))
    frame_start = int(_cfg_get(cfg, "FRAME_START", 0))
    frame_end = int(_cfg_get(cfg, "FRAME_END", 300))
    image_cam = str(_cfg_get(cfg, "IMAGE_CAM", "image_00"))
    model_zoo = Path(str(_cfg_get(cfg, "IMAGE_MODEL_ZOO", "configs/image_model_zoo.yaml")))
    gdino_model_id = str(_cfg_get(cfg, "GDINO_MODEL_ID", "gdino_sam2_v1"))
    yolo_model_id = str(_cfg_get(cfg, "YOLO_MODEL_ID", "yolo_world_v1"))
    overwrite = bool(_cfg_get(cfg, "OVERWRITE", True))

    seed_frame = int(_cfg_get(cfg, "SEED_FRAME", 290))

    stage1_stride = int(_cfg_get(cfg, "STAGE1_STRIDE", 5))
    stage1_force_frames = [int(v) for v in _cfg_list(cfg, "STAGE1_FORCE_FRAMES", [seed_frame])]
    if seed_frame not in stage1_force_frames:
        stage1_force_frames.append(seed_frame)

    qa_seed = int(_cfg_get(cfg, "QA_RANDOM_SEED", 20260130))
    qa_sample_n = int(_cfg_get(cfg, "QA_SAMPLE_N", 12))
    qa_force = [int(v) for v in _cfg_list(cfg, "QA_FORCE_INCLUDE", [0, 50, 100, 150, 200, 250, 290, 300])]

    stage2_window_pre = int(_cfg_get(cfg, "STAGE2_WINDOW_PRE", 30))
    stage2_window_post = int(_cfg_get(cfg, "STAGE2_WINDOW_POST", 30))
    stage2_max_seeds_total = int(_cfg_get(cfg, "STAGE2_MAX_SEEDS_TOTAL", 24))
    stage2_propagate = str(_cfg_get(cfg, "SAM2_VIDEO_PROPAGATE", "both")).lower()

    output_mask_format = "png"
    output_overlay_only_qa = True

    stage1_cfg = {
        "gdino_text_prompt": _cfg_list(cfg, "GDINO_TEXT_PROMPT", []),
        "gdino_box_th": float(_cfg_get(cfg, "GDINO_BOX_TH", 0.23)),
        "gdino_text_th": float(_cfg_get(cfg, "GDINO_TEXT_TH", 0.23)),
        "gdino_topk": int(_cfg_get(cfg, "GDINO_TOPK", 12)),
        "yoloworld_text_prompt": _cfg_list(cfg, "YOLOWORLD_TEXT_PROMPT", []),
        "yoloworld_conf_th": float(_cfg_get(cfg, "YOLOWORLD_CONF_TH", 0.25)),
        "yoloworld_topk": int(_cfg_get(cfg, "YOLOWORLD_TOPK", 30)),
        "nms_iou_th": float(_cfg_get(cfg, "NMS_IOU_TH", 0.5)),
        "max_boxes_after_merge": int(_cfg_get(cfg, "MAX_BOXES_AFTER_MERGE", 12)),
        "roi_bottom_crop": float(_cfg_get(cfg, "ROI_BOTTOM_CROP", 0.50)),
        "roi_side_crop": tuple(_cfg_get(cfg, "ROI_SIDE_CROP", [0.05, 0.95])),
        "box_aspect_min": float(_cfg_get(cfg, "BOX_ASPECT_MIN", 2.0)),
        "box_h_ratio_max": float(_cfg_get(cfg, "BOX_H_MAX_RATIO", 0.35)),
        "box_center_v_min": float(_cfg_get(cfg, "BOX_CENTER_V_MIN_RATIO", 0.50)),
        "max_seeds_per_frame": int(_cfg_get(cfg, "MAX_SEEDS_PER_FRAME", 4)),
        "sam2_image_max_masks": int(_cfg_get(cfg, "SAM2_IMAGE_MAX_MASKS", 2)),
        "sam2_mask_min_area_px": float(_cfg_get(cfg, "SAM2_MASK_MIN_AREA_PX", 600)),
        "mask_mrr_aspect_min": float(_cfg_get(cfg, "MASK_MRR_ASPECT_MIN", 2.0)),
        "stripe_aspect_ratio_min": float(_cfg_get(cfg, "STRIPE_ASPECT_RATIO_MIN", 2.0)),
        "stripe_min_count": int(_cfg_get(cfg, "STRIPE_MIN_COUNT", 6)),
        "yolo_neg_prompts": _cfg_list(cfg, "YOLO_NEG_PROMPTS", []),
        "yolo_neg_iou_th": float(_cfg_get(cfg, "YOLO_NEG_IOU_TH", 0.3)),
        "yolo_neg_score_mult": float(_cfg_get(cfg, "YOLO_NEG_SCORE_MULT", 0.3)),
        "yolo_use_crosswalk": bool(_cfg_get(cfg, "YOLO_USE_CROSSWALK", False)),
    }
    score_cfg = {
        "w_support": float(_cfg_get(cfg, "W_SUPPORT", 0.30)),
        "w_model_score": float(_cfg_get(cfg, "W_MODEL_SCORE", 0.20)),
        "w_in_roi": float(_cfg_get(cfg, "W_IN_ROI", 0.15)),
        "w_stripe": float(_cfg_get(cfg, "W_STRIPE", 0.15)),
        "w_area": float(_cfg_get(cfg, "W_AREA", 0.10)),
        "w_aspect": float(_cfg_get(cfg, "W_ASPECT", 0.10)),
    }

    run_id = now_ts()
    run_dir = Path("runs") / f"image_stage12_ensemble_0010_000_300_{run_id}"
    if overwrite:
        ensure_overwrite(run_dir)
    elif run_dir.exists():
        raise SystemExit(f"run_dir exists and overwrite is false: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    qa_dir = run_dir / "qa"
    stage1_dir = run_dir / "stage1"
    stage1_frames_dir = stage1_dir / "frames"
    stage2_dir = run_dir / "stage2"
    stage2_windows_dir = stage2_dir / "windows"
    stage2_merged_dir = stage2_dir / "merged_masks"
    stage2_overlay_dir = stage2_dir / "overlays"
    tables_dir = run_dir / "tables"
    for d in (
        qa_dir,
        stage1_dir,
        stage1_frames_dir,
        stage2_windows_dir,
        stage2_merged_dir,
        stage2_overlay_dir,
        tables_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    data_root = _find_data_root(str(_cfg_get(cfg, "KITTI_ROOT", "")))
    image_dir = _find_image_dir(data_root, drive_id, image_cam)

    existing_frames, frame_paths, missing_frames = _collect_frames(image_dir, frame_start, frame_end)
    missing_rows = [{"frame_id": fid} for fid in missing_frames]
    write_csv(tables_dir / "missing_frames.csv", missing_rows, ["frame_id"])

    if not existing_frames:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "no_existing_frames"})
        raise SystemExit("no existing frames in range")

    required_seed_frame = seed_frame
    if f"{seed_frame:010d}" not in existing_frames:
        nearest = _closest_existing_frame(existing_frames, seed_frame)
        if nearest is None:
            write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "seed_frame_missing"})
            raise SystemExit("seed frame missing")
        required_seed_frame = int(nearest)
    if required_seed_frame not in stage1_force_frames:
        stage1_force_frames.append(required_seed_frame)

    required_window_start = max(frame_start, required_seed_frame - stage2_window_pre)
    required_window_end = min(frame_end, required_seed_frame + stage2_window_post)

    first_img = Image.open(frame_paths[existing_frames[0]]).convert("RGB")
    base_size = first_img.size

    qa_frames = _select_qa_frames(frame_start, frame_end, qa_seed, qa_sample_n, qa_force)
    qa_frames_json = {
        "seed": qa_seed,
        "sample_n": qa_sample_n,
        "force_include": qa_force,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frames": qa_frames,
    }
    write_json(qa_dir / "qa_frames.json", qa_frames_json)

    gdino_cfg = _load_model_cfg(model_zoo, gdino_model_id)
    if not gdino_cfg:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "gdino_model_id_not_found"})
        raise SystemExit(f"gdino model id not found: {gdino_model_id}")

    yolo_cfg = _load_model_cfg(model_zoo, yolo_model_id)
    if not yolo_cfg:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "yoloworld_model_id_not_found"})
        raise SystemExit(f"yolo model id not found: {yolo_model_id}")

    missing_weights = _check_gdino_weights(gdino_cfg) + _check_yoloworld_weights(yolo_cfg)
    if missing_weights:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_weights", "items": missing_weights})
        raise SystemExit(f"missing weights: {missing_weights}")

    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from ultralytics import YOLO
    except Exception as exc:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": f"missing_dependencies:{exc}"})
        raise SystemExit(f"missing dependencies: {exc}")

    device = _device_auto()

    dino_repo = gdino_cfg.get("dino_repo_id") or (gdino_cfg.get("download") or {}).get("repo")
    dino_local_dir = Path(str(gdino_cfg.get("dino_weights_path") or ""))
    if not dino_repo or not dino_local_dir.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "gdino_repo_or_weights_missing"})
        raise SystemExit("gdino repo or weights missing")

    processor = AutoProcessor.from_pretrained(dino_local_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_local_dir, local_files_only=True).to(device)

    yolo_weights = _resolve_yoloworld_weights(yolo_cfg)
    if yolo_weights is None or not yolo_weights.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "yoloworld_weights_not_found"})
        raise SystemExit("yolo-world weights not found")
    yolo_model = YOLO(str(yolo_weights))
    yolo_device = str(_cfg_get(cfg, "YOLOWORLD_DEVICE", device))
    try:
        yolo_model.to(yolo_device)
    except Exception:
        pass
    yolo_prompts = stage1_cfg.get("yoloworld_text_prompt") or yolo_cfg.get("prompts") or []
    if yolo_prompts and hasattr(yolo_model, "set_classes"):
        yolo_model.set_classes([str(p) for p in yolo_prompts])
    yolo_runtime = yolo_cfg.get("runtime") or {}
    if "yoloworld_iou_th" not in stage1_cfg:
        stage1_cfg["yoloworld_iou_th"] = float(yolo_runtime.get("iou_threshold", 0.5))
    if "yoloworld_imgsz" not in stage1_cfg:
        stage1_cfg["yoloworld_imgsz"] = int(yolo_cfg.get("input_size", 1024))
    stage1_cfg["yoloworld_device"] = yolo_device

    sam2_cfg = str((gdino_cfg.get("download") or {}).get("sam2_model_cfg") or "")
    if "sam2/configs/" in sam2_cfg:
        sam2_cfg = "configs/" + sam2_cfg.split("sam2/configs/", 1)[1]
    elif sam2_cfg and not sam2_cfg.startswith("configs/"):
        sam2_cfg = f"configs/{sam2_cfg}"
    ckpt_path = _resolve_sam2_ckpt(gdino_cfg)
    if ckpt_path is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "sam2_checkpoint_not_found"})
        raise SystemExit("sam2 checkpoint not found")

    sam2 = build_sam2(sam2_cfg, str(ckpt_path), device=device)
    sam2_predictor = SAM2ImagePredictor(sam2)
    sam2_video = build_sam2_video_predictor(sam2_cfg, ckpt_path=str(ckpt_path), device=device)

    stage1_frames = set(range(frame_start, frame_end + 1, stage1_stride))
    for f in stage1_force_frames:
        if frame_start <= f <= frame_end:
            stage1_frames.add(f)
    stage1_frames = sorted(stage1_frames)

    stage1_stats = {
        "gdino_boxes": 0,
        "yolo_boxes": 0,
        "merged_boxes": 0,
        "boxes_pass": 0,
        "masks_total": 0,
        "masks_pass": 0,
    }
    seeds: List[Dict[str, object]] = []
    stage1_montage_boxes_inputs: List[Tuple[str, Path]] = []
    stage1_montage_seeds_inputs: List[Tuple[str, Path]] = []
    stage1_rows: List[Dict[str, object]] = []
    seed_score_rows: List[Dict[str, object]] = []
    seed_id = 0
    cand_id = 0
    stage1_qa_dir = stage1_dir / "qa_frames"
    stage1_qa_dir.mkdir(parents=True, exist_ok=True)

    for frame in stage1_frames:
        frame_id = f"{frame:010d}"
        img_path = frame_paths.get(frame_id)
        if img_path is None or not img_path.exists():
            stage1_rows.append(
                {"frame_id": frame_id, "gdino_count": 0, "yolo_count": 0, "merged_count": 0, "seed_count": 0}
            )
            continue
        img = Image.open(img_path).convert("RGB")
        (
            gdino_boxes,
            gdino_scores,
            yolo_boxes,
            yolo_scores,
            merged_boxes,
            merged_scores,
            yolo_neg_boxes,
            yolo_neg_labels,
            candidates,
            stats,
        ) = _run_stage1_on_image(img, processor, model, yolo_model, sam2_predictor, stage1_cfg, score_cfg)
        for k in stage1_stats:
            stage1_stats[k] += int(stats.get(k, 0))

        if candidates:
            candidates_sorted = sorted(candidates, key=lambda c: (c["seed_score_after"], c["area"]), reverse=True)
            max_per_frame = int(stage1_cfg.get("max_seeds_per_frame", 4))
            if max_per_frame > 0:
                candidates_sorted = candidates_sorted[:max_per_frame]
        else:
            candidates_sorted = []

        frame_masks = [c["mask"] for c in candidates_sorted]
        stage1_rows.append(
            {
                "frame_id": frame_id,
                "gdino_count": len(gdino_boxes),
                "yolo_count": len(yolo_neg_boxes) if not stage1_cfg.get("yolo_use_crosswalk") else len(yolo_boxes),
                "merged_count": len(merged_boxes),
                "seed_count": len(frame_masks),
            }
        )
        frame_dir = stage1_frames_dir / f"frame_{frame_id}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / "seed_masks").mkdir(parents=True, exist_ok=True)
        (frame_dir / "gdino_boxes.json").write_text(
            json.dumps([{"bbox": b, "score": s} for b, s in zip(gdino_boxes, gdino_scores)], indent=2),
            encoding="utf-8",
        )
        (frame_dir / "yoloworld_boxes.json").write_text(
            json.dumps([{"bbox": b, "score": s} for b, s in zip(yolo_boxes, yolo_scores)], indent=2),
            encoding="utf-8",
        )
        (frame_dir / "boxes_merged.json").write_text(
            json.dumps([{"bbox": b, "score": s} for b, s in zip(merged_boxes, merged_scores)], indent=2),
            encoding="utf-8",
        )
        for cand in candidates:
            seed_score_rows.append(
                {
                    "candidate_id": cand_id,
                    "frame_id": frame_id,
                    "seed_score_before": float(cand["seed_score_before"]),
                    "seed_score_after": float(cand["seed_score_after"]),
                    "support_models": ",".join(cand.get("support_models") or []),
                    "support_count": int(cand.get("support_count", 0)),
                    "model_score": float(cand.get("model_score", 0.0)),
                    "in_roi_ratio": float(cand.get("in_roi_ratio", 0.0)),
                    "stripe_ok": float(cand.get("stripe_ok", 0.0)),
                    "area_ok": float(cand.get("area_ok", 0.0)),
                    "aspect_ok": float(cand.get("aspect_ok", 0.0)),
                    "yolo_neg_hit": int(cand.get("yolo_neg_hit", 0)),
                    "yolo_neg_iou_max": float(cand.get("yolo_neg_iou_max", 0.0)),
                    "area": float(cand["area"]),
                    "stripe_count": int(cand["stripe_count"]),
                    "mask_mrr_aspect": float(cand["mask_mrr_aspect"]),
                    "box_aspect": float(cand["box_aspect"]),
                    "box_h_ratio": float(cand["box_h_ratio"]),
                    "box_center_v_ratio": float(cand["box_center_v_ratio"]),
                    "bbox_x1": float(cand["bbox"][0]),
                    "bbox_y1": float(cand["bbox"][1]),
                    "bbox_x2": float(cand["bbox"][2]),
                    "bbox_y2": float(cand["bbox"][3]),
                }
            )
            cand_id += 1
        for cand in candidates_sorted:
            seeds.append(
                {
                    "seed_id": seed_id,
                    "seed_frame": frame,
                    "seed_frame_id": frame_id,
                    "seed_score_before": float(cand["seed_score_before"]),
                    "seed_score_after": float(cand["seed_score_after"]),
                    "model_score": float(cand.get("model_score", 0.0)),
                    "support_models": ",".join(cand.get("support_models") or []),
                    "support_count": int(cand.get("support_count", 0)),
                    "in_roi_ratio": float(cand.get("in_roi_ratio", 0.0)),
                    "stripe_ok": float(cand.get("stripe_ok", 0.0)),
                    "area_ok": float(cand.get("area_ok", 0.0)),
                    "aspect_ok": float(cand.get("aspect_ok", 0.0)),
                    "yolo_neg_hit": int(cand.get("yolo_neg_hit", 0)),
                    "yolo_neg_iou_max": float(cand.get("yolo_neg_iou_max", 0.0)),
                    "area": float(cand["area"]),
                    "stripe_count": int(cand["stripe_count"]),
                    "bbox_x1": float(cand["bbox"][0]),
                    "bbox_y1": float(cand["bbox"][1]),
                    "bbox_x2": float(cand["bbox"][2]),
                    "bbox_y2": float(cand["bbox"][3]),
                    "box_aspect": float(cand["box_aspect"]),
                    "box_h_ratio": float(cand["box_h_ratio"]),
                    "box_center_v_ratio": float(cand["box_center_v_ratio"]),
                    "mask_mrr_aspect": float(cand["mask_mrr_aspect"]),
                    "selected_for_stage2": False,
                    "mask": cand["mask"],
                }
            )
            mask_path = frame_dir / "seed_masks" / f"seed_{seed_id}.png"
            _write_mask(mask_path, cand["mask"], base_size)
            seed_id += 1

        if frame in qa_frames:
            overlay_boxes = stage1_qa_dir / f"frame_{frame_id}_overlay_stage1_boxes.png"
            _draw_overlay_stage1_boxes(
                img, gdino_boxes, gdino_scores, yolo_boxes, yolo_scores, merged_boxes, merged_scores, overlay_boxes
            )
            stage1_montage_boxes_inputs.append((frame_id, overlay_boxes))
            overlay_seeds = stage1_qa_dir / f"frame_{frame_id}_overlay_stage1_seeds.png"
            _draw_overlay_stage1_seeds(img, frame_masks, overlay_seeds)
            stage1_montage_seeds_inputs.append((frame_id, overlay_seeds))
        elif not output_overlay_only_qa:
            overlay_boxes = frame_dir / "overlay_stage1_boxes.png"
            _draw_overlay_stage1_boxes(
                img, gdino_boxes, gdino_scores, yolo_boxes, yolo_scores, merged_boxes, merged_scores, overlay_boxes
            )
            overlay_seeds = frame_dir / "overlay_stage1_seeds.png"
            _draw_overlay_stage1_seeds(img, frame_masks, overlay_seeds)

    seeds_index_rows = []
    for s in seeds:
        row = {k: v for k, v in s.items() if k != "mask"}
        seeds_index_rows.append(row)
    write_csv(
        stage1_dir / "seeds_index.csv",
        seeds_index_rows,
        [
            "seed_id",
            "seed_frame",
            "seed_frame_id",
            "seed_score_before",
            "seed_score_after",
            "model_score",
            "support_models",
            "support_count",
            "in_roi_ratio",
            "stripe_ok",
            "area_ok",
            "aspect_ok",
            "yolo_neg_hit",
            "yolo_neg_iou_max",
            "area",
            "stripe_count",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "box_aspect",
            "box_h_ratio",
            "box_center_v_ratio",
            "mask_mrr_aspect",
            "selected_for_stage2",
        ],
    )

    if seed_score_rows:
        write_csv(
            tables_dir / "seed_score_table.csv",
            seed_score_rows,
            [
                "candidate_id",
                "frame_id",
                "seed_score_before",
                "seed_score_after",
                "support_models",
                "support_count",
                "model_score",
                "in_roi_ratio",
                "stripe_ok",
                "area_ok",
                "aspect_ok",
                "yolo_neg_hit",
                "yolo_neg_iou_max",
                "area",
                "stripe_count",
                "mask_mrr_aspect",
                "box_aspect",
                "box_h_ratio",
                "box_center_v_ratio",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
            ],
        )

    if stage1_montage_boxes_inputs:
        _montage(stage1_montage_boxes_inputs, qa_dir / "montage_stage1_boxes.png")
    if stage1_montage_seeds_inputs:
        _montage(stage1_montage_seeds_inputs, qa_dir / "montage_stage1_seeds.png")
    write_csv(
        tables_dir / "per_frame_stage1_counts.csv",
        stage1_rows,
        ["frame_id", "gdino_count", "yolo_count", "merged_count", "seed_count"],
    )

    if not seeds:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "no_seeds_after_stage1"})
        raise SystemExit("no seeds after stage1")

    seed_required = [s for s in seeds if int(s["seed_frame"]) == int(required_seed_frame)]
    if not seed_required:
        write_json(
            run_dir / "decision.json",
            {"status": "FAIL", "reason": "seed_required_missing", "seed_frame": required_seed_frame},
        )
        raise SystemExit(f"seed at frame {required_seed_frame} missing")

    seeds_sorted = sorted(seeds, key=lambda s: (s["seed_score_after"], s["area"]), reverse=True)
    seeds_used = seeds_sorted[: stage2_max_seeds_total] if stage2_max_seeds_total > 0 else seeds_sorted[:]
    if seed_required:
        seed_ids = {s["seed_id"] for s in seed_required}
        if not any(s["seed_id"] in seed_ids for s in seeds_used):
            seeds_used.append(seed_required[0])
            if stage2_max_seeds_total > 0 and len(seeds_used) > stage2_max_seeds_total:
                seeds_used = sorted(
                    seeds_used, key=lambda s: (s["seed_score_after"], s["area"]), reverse=True
                )[:stage2_max_seeds_total]

    window_seeds = [
        s
        for s in seeds_sorted
        if required_window_start <= int(s["seed_frame"]) <= required_window_end
    ]
    if window_seeds:
        window_top2 = []
        seen_frames = set()
        for s in window_seeds:
            frame = int(s["seed_frame"])
            if frame in seen_frames:
                continue
            window_top2.append(s)
            seen_frames.add(frame)
            if len(window_top2) >= 2:
                break
        existing_ids = {s["seed_id"] for s in seeds_used}
        for s in window_top2:
            if s["seed_id"] not in existing_ids:
                seeds_used.append(s)
                existing_ids.add(s["seed_id"])

    seeds_used_ids = {s["seed_id"] for s in seeds_used}
    for s in seeds:
        if s["seed_id"] in seeds_used_ids:
            s["selected_for_stage2"] = True

    write_csv(
        stage1_dir / "seeds_index.csv",
        [{k: v for k, v in s.items() if k != "mask"} for s in seeds],
        [
            "seed_id",
            "seed_frame",
            "seed_frame_id",
            "seed_score_before",
            "seed_score_after",
            "model_score",
            "support_models",
            "support_count",
            "in_roi_ratio",
            "stripe_ok",
            "area_ok",
            "aspect_ok",
            "yolo_neg_hit",
            "yolo_neg_iou_max",
            "area",
            "stripe_count",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "box_aspect",
            "box_h_ratio",
            "box_center_v_ratio",
            "mask_mrr_aspect",
            "selected_for_stage2",
        ],
    )

    frame_ids_sorted = [f"{f:010d}" for f in range(frame_start, frame_end + 1)]
    video_frames = [f for f in frame_ids_sorted if f in frame_paths]
    video_dir = stage2_dir / "video_frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame_id in enumerate(video_frames):
        img_path = frame_paths[frame_id]
        dst = video_dir / f"{idx:05d}.jpg"
        if dst.exists():
            continue
        Image.open(img_path).convert("RGB").save(dst, format="JPEG", quality=95)

    video_index = {fid: idx for idx, fid in enumerate(video_frames)}
    merged_masks: Dict[str, np.ndarray] = {}
    window_ranges: List[Tuple[int, int]] = []

    for seed in seeds_used:
        seed_frame = int(seed["seed_frame"])
        seed_frame_id = str(seed["seed_frame_id"])
        seed_mask = seed["mask"]
        if seed_frame_id not in video_index:
            continue
        win_start = max(frame_start, seed_frame - stage2_window_pre)
        win_end = min(frame_end, seed_frame + stage2_window_post)
        window_ranges.append((win_start, win_end))
        window_dir = stage2_windows_dir / f"seed_{seed_frame_id}_{win_start:04d}_{win_end:04d}"
        window_masks_dir = window_dir / "masks"
        window_overlays_dir = window_dir / "overlays"
        window_masks_dir.mkdir(parents=True, exist_ok=True)
        window_overlays_dir.mkdir(parents=True, exist_ok=True)

        seed_index = video_index[seed_frame_id]
        window_indices = [
            idx
            for idx, fid in enumerate(video_frames)
            if win_start <= int(fid) <= win_end
        ]
        if not window_indices:
            continue
        start_idx = min(window_indices)
        end_idx = max(window_indices)

        inference_state = sam2_video.init_state(video_path=str(video_dir), offload_video_to_cpu=True)
        sam2_video.add_new_mask(inference_state, seed_index, 1, seed_mask)

        window_masks: Dict[int, np.ndarray] = {}
        if stage2_propagate in ("both", "forward"):
            max_forward = end_idx - seed_index + 1
            for frame_idx, _obj_ids, video_masks in sam2_video.propagate_in_video(
                inference_state,
                start_frame_idx=seed_index,
                max_frame_num_to_track=max_forward,
                reverse=False,
            ):
                if video_masks is None or len(video_masks) == 0:
                    continue
                mask = np.squeeze(video_masks[0].cpu().numpy()) > 0
                window_masks[frame_idx] = mask
        if stage2_propagate in ("both", "backward"):
            max_backward = seed_index - start_idx + 1
            for frame_idx, _obj_ids, video_masks in sam2_video.propagate_in_video(
                inference_state,
                start_frame_idx=seed_index,
                max_frame_num_to_track=max_backward,
                reverse=True,
            ):
                if video_masks is None or len(video_masks) == 0:
                    continue
                mask = np.squeeze(video_masks[0].cpu().numpy()) > 0
                window_masks[frame_idx] = mask if frame_idx not in window_masks else (window_masks[frame_idx] | mask)

        for frame in range(win_start, win_end + 1):
            frame_id = f"{frame:010d}"
            if frame_id in video_index:
                idx = video_index[frame_id]
                mask = window_masks.get(idx)
                if mask is None:
                    mask = np.zeros((base_size[1], base_size[0]), dtype=bool)
                out_mask = window_masks_dir / f"frame_{frame_id}.{output_mask_format}"
                _write_mask(out_mask, mask, base_size)
                merged_masks[frame_id] = mask if frame_id not in merged_masks else (merged_masks[frame_id] | mask)
                if output_overlay_only_qa and frame in qa_frames:
                    img = Image.open(frame_paths[frame_id]).convert("RGB")
                    out_overlay = window_overlays_dir / f"frame_{frame_id}_overlay.png"
                    _draw_overlay_stage2(img, mask, out_overlay, f"frame {frame_id}")
            else:
                out_mask = window_masks_dir / f"frame_{frame_id}.{output_mask_format}"
                _write_mask(out_mask, None, base_size)

    per_frame_rows = []
    hit_frames = 0
    qa_stage2_inputs: List[Tuple[str, Path]] = []
    mask_flags: List[int] = []
    for frame in range(frame_start, frame_end + 1):
        frame_id = f"{frame:010d}"
        if frame_id in merged_masks:
            mask = merged_masks[frame_id]
        else:
            mask = np.zeros((base_size[1], base_size[0]), dtype=bool)
        area = float(np.sum(mask))
        mask_flags.append(1 if area > 0 else 0)
        if area > 0:
            hit_frames += 1
        out_mask_path = stage2_merged_dir / f"frame_{frame_id}.{output_mask_format}"
        _write_mask(out_mask_path, mask, base_size)
        if output_overlay_only_qa and frame in qa_frames and frame_id in frame_paths:
            img = Image.open(frame_paths[frame_id]).convert("RGB")
            out_overlay = stage2_overlay_dir / f"frame_{frame_id}_overlay.png"
            _draw_overlay_stage2(img, mask, out_overlay, f"frame {frame_id}")
            qa_stage2_inputs.append((frame_id, out_overlay))
        per_frame_rows.append({"frame_id": frame_id, "mask_area_px": area})

    write_csv(tables_dir / "per_frame_mask_area.csv", per_frame_rows, ["frame_id", "mask_area_px"])

    if qa_stage2_inputs:
        _montage(qa_stage2_inputs, qa_dir / "montage_stage2.png")

    existing_count = len(existing_frames)
    total_frames = frame_end - frame_start + 1
    coverage = (hit_frames / existing_count) if existing_count > 0 else 0.0
    window_cover = 0
    if window_ranges:
        covered = set()
        for start, end in window_ranges:
            for f in range(start, end + 1):
                covered.add(f)
        window_cover = len(covered)
    window_coverage_ratio = window_cover / total_frames if total_frames > 0 else 0.0

    warnings: List[str] = []
    if len(seeds_used) < 2:
        warnings.append("seed_count_low")
    if hit_frames == 0:
        warnings.append("stage2_all_empty")

    window_flags = []
    for f in range(required_window_start, required_window_end + 1):
        fid = f"{f:010d}"
        if fid not in frame_paths:
            continue
        idx = f - frame_start
        if 0 <= idx < len(mask_flags):
            window_flags.append(mask_flags[idx])
    window_all_empty = bool(window_flags) and all(v == 0 for v in window_flags)
    window_seed_selected = sum(
        1 for s in seeds_used if required_window_start <= int(s["seed_frame"]) <= required_window_end
    )
    if window_seed_selected < 2:
        warnings.append("required_window_gap")

    status = "PASS"
    reason = []
    if not seed_required:
        status = "FAIL"
        reason.append("seed_required_missing")
    if hit_frames == 0 or window_all_empty:
        status = "FAIL"
        reason.append("stage2_all_empty")
    if status != "FAIL" and warnings:
        status = "WARN"
        reason.extend(warnings)

    decision = {
        "status": status,
        "reasons": reason,
        "seed_total": len(seeds),
        "seeds_used": len(seeds_used),
        "window_count": len(seeds_used),
        "window_coverage_ratio": round(window_coverage_ratio, 4),
        "frames_existing": existing_count,
        "frames_hit": hit_frames,
        "coverage_ratio": round(coverage, 4),
        "missing_frames": len(missing_frames),
        "required_seed_frame": required_seed_frame,
        "required_window_start": required_window_start,
        "required_window_end": required_window_end,
        "stage1_stats": stage1_stats,
    }
    write_json(run_dir / "decision.json", decision)

    resolved_cfg = dict(cfg)
    resolved_cfg.update(
        {
            "resolved": {
                "run_id": run_id,
                "data_root": str(data_root),
                "image_dir": str(image_dir),
                "device": device,
                "gdino_model_id": gdino_model_id,
                "yolo_model_id": yolo_model_id,
                "sam2_checkpoint": str(ckpt_path),
                "yoloworld_weights": str(yolo_weights),
                "yolo_use_crosswalk": bool(stage1_cfg.get("yolo_use_crosswalk", False)),
                "yolo_neg_prompts": stage1_cfg.get("yolo_neg_prompts", []),
                "yolo_neg_iou_th": float(stage1_cfg.get("yolo_neg_iou_th", 0.3)),
                "yolo_neg_score_mult": float(stage1_cfg.get("yolo_neg_score_mult", 0.3)),
            }
        }
    )
    resolved_path = run_dir / "resolved_config.yaml"
    import yaml

    resolved_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "params_hash.txt").write_text(_hash_file(resolved_path), encoding="utf-8")

    gate_ratio = 0.0
    if stage1_stats["masks_total"] > 0:
        gate_ratio = stage1_stats["masks_pass"] / stage1_stats["masks_total"]

    report_lines = [
        "# Image Stage1+Stage2 Ensemble (GDINO+YOLO-World) 0010 f0-300",
        "",
        f"- run_id: {run_id}",
        f"- drive_id: {drive_id}",
        f"- frame_range: {frame_start}-{frame_end}",
        f"- stage1_stride: {stage1_stride}",
        f"- stage1_frames: {stage1_frames}",
        f"- seed_frame_required: {required_seed_frame}",
        f"- stage1_seed_total: {len(seeds)}",
        f"- stage1_gate_pass_ratio: {gate_ratio:.3f}",
        f"- stage2_seeds_used: {len(seeds_used)}",
        f"- stage2_window_count: {len(seeds_used)}",
        f"- window_coverage_ratio: {window_coverage_ratio:.3f}",
        f"- merged_mask_frames_hit: {hit_frames}",
        f"- missing_frames: {len(missing_frames)}",
        "",
        "## outputs",
        f"- seeds_index: stage1/seeds_index.csv",
        f"- qa montage stage1 boxes: qa/montage_stage1_boxes.png",
        f"- qa montage stage1 seeds: qa/montage_stage1_seeds.png",
        f"- qa montage stage2: qa/montage_stage2.png",
        f"- stage2 windows: stage2/windows/seed_*",
        f"- merged masks: stage2/merged_masks/frame_*.{output_mask_format}",
        f"- per-frame area: tables/per_frame_mask_area.csv",
        f"- per-frame counts: tables/per_frame_stage1_counts.csv",
        f"- seed score table: tables/seed_score_table.csv",
        f"- missing frames: tables/missing_frames.csv",
        f"- decision: decision.json",
        "",
        "## next",
        "- review QA montages and seed_score_table before any world landing",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
