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


CFG_DEFAULT = Path("configs/image_stage12_sparse_seed_0010_f000_500.yaml")


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


def _check_weights(model_cfg: dict) -> List[str]:
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


def _roi_crop(img: Image.Image, bottom_ratio: float, side_crop: Tuple[float, float]) -> Tuple[Image.Image, Tuple[int, int]]:
    w, h = img.size
    x0 = int(w * float(side_crop[0]))
    x1 = int(w * float(side_crop[1]))
    y0 = int(h * float(bottom_ratio))
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    return img.crop((x0, y0, x1, h)), (x0, y0)


def _draw_overlay_stage1(
    img: Image.Image,
    boxes: List[List[float]],
    scores: List[float],
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
    draw = ImageDraw.Draw(overlay)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        score = scores[idx] if idx < len(scores) else None
        label = f"crosswalk:{score:.2f}" if score is not None else "crosswalk"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 200), width=2)
        draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 0, 255))
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
    processor,
    model,
    sam2_predictor,
    stage1_cfg: Dict[str, object],
) -> Tuple[List[List[float]], List[float], List[str], List[Dict[str, object]], Dict[str, int]]:
    import torch

    roi_img, roi_offset = _roi_crop(
        img,
        float(stage1_cfg.get("roi_bottom_crop", 0.55)),
        tuple(stage1_cfg.get("roi_side_crop", [0.05, 0.95])),
    )
    prompts = [str(p).strip().lower() for p in (stage1_cfg.get("gdino_text_prompt") or []) if str(p).strip()]
    inputs = processor(images=roi_img, text=[prompts], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [(roi_img.size[1], roi_img.size[0])]
    box_th = float(stage1_cfg.get("gdino_box_th", 0.25))
    text_th = float(stage1_cfg.get("gdino_text_th", 0.25))
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_th,
            text_threshold=text_th,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_th,
            text_threshold=text_th,
            target_sizes=target_sizes,
        )[0]

    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    text_labels = results.get("text_labels", [])
    offset_x, offset_y = roi_offset
    full_boxes: List[List[float]] = []
    full_scores: List[float] = []
    full_labels: List[str] = []
    for idx, box in enumerate(boxes):
        bbox = [float(v) for v in (box.tolist() if hasattr(box, "tolist") else box)]
        bbox[0] += offset_x
        bbox[2] += offset_x
        bbox[1] += offset_y
        bbox[3] += offset_y
        full_boxes.append(bbox)
        full_scores.append(float(scores[idx]) if idx < len(scores) else 0.0)
        full_labels.append(str(text_labels[idx]) if idx < len(text_labels) else "")

    nms_iou = float(stage1_cfg.get("nms_iou_th", 0.5))
    keep = _nms(full_boxes, full_scores, nms_iou)
    full_boxes = [full_boxes[i] for i in keep]
    full_scores = [full_scores[i] for i in keep]
    full_labels = [full_labels[i] for i in keep]

    topk = int(stage1_cfg.get("gdino_topk", 5))
    if topk > 0 and len(full_scores) > topk:
        order = sorted(range(len(full_scores)), key=lambda i: full_scores[i], reverse=True)[:topk]
        full_boxes = [full_boxes[i] for i in order]
        full_scores = [full_scores[i] for i in order]
        full_labels = [full_labels[i] for i in order]

    total_boxes = len(full_boxes)
    box_aspect_min = float(stage1_cfg.get("box_aspect_min", 2.0))
    box_h_ratio_max = float(stage1_cfg.get("box_h_ratio_max", 0.35))
    box_center_v_min = float(stage1_cfg.get("box_center_v_min", 0.50))
    full_boxes, full_scores, full_labels, box_metrics = _filter_stage1_boxes(
        full_boxes,
        full_scores,
        full_labels,
        img.height,
        box_aspect_min,
        box_h_ratio_max,
        box_center_v_min,
    )

    sam2_predictor.set_image(np.array(img))
    max_masks = int(stage1_cfg.get("sam2_image_max_masks", 3))
    min_area = float(stage1_cfg.get("sam2_mask_min_area_px", 800))
    aspect_min = float(stage1_cfg.get("stripe_aspect_ratio_min", 2.0))
    stripe_min = int(stage1_cfg.get("stripe_min_count", 6))
    mrr_aspect_min = float(stage1_cfg.get("mask_mrr_aspect_min", 2.0))
    candidates: List[Dict[str, object]] = []
    masks_total = 0
    for bbox, score, label, metrics in zip(full_boxes, full_scores, full_labels, box_metrics):
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
            stripe_count = _stripe_count(mask_bin, aspect_min)
            if stripe_count < stripe_min:
                continue
            candidates.append(
                {
                    "mask": mask_bin,
                    "area": area,
                    "stripe_count": stripe_count,
                    "score": float(score),
                    "label": str(label),
                    "bbox": bbox,
                    "mask_mrr_aspect": mrr_aspect,
                    **metrics,
                }
            )

    stats = {
        "boxes_total": total_boxes,
        "boxes_pass": len(full_boxes),
        "masks_total": masks_total,
        "masks_pass": len(candidates),
    }
    return full_boxes, full_scores, full_labels, candidates, stats


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

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_DEFAULT))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if not cfg:
        raise SystemExit(f"missing config: {cfg_path}")

    drive_id = str(_cfg_get(cfg, "DRIVE_ID", ""))
    frame_start = int(_cfg_get(cfg, "FRAME_START", 0))
    frame_end = int(_cfg_get(cfg, "FRAME_END", 500))
    image_cam = str(_cfg_get(cfg, "IMAGE_CAM", "image_00"))
    model_id = str(_cfg_get(cfg, "IMAGE_MODEL_ID", "gdino_sam2_v1"))
    model_zoo = Path(str(_cfg_get(cfg, "IMAGE_MODEL_ZOO", "configs/image_model_zoo.yaml")))
    overwrite = bool(_cfg_get(cfg, "OVERWRITE", True))

    seed_frame = int(_cfg_get(cfg, "SEED_FRAME", 290))

    stage1_stride = int(_cfg_get(cfg, "STAGE1_STRIDE", 10))
    stage1_force_frames = [int(v) for v in _cfg_list(cfg, "STAGE1_FORCE_FRAMES", [seed_frame])]
    if seed_frame not in stage1_force_frames:
        stage1_force_frames.append(seed_frame)

    qa_seed = int(_cfg_get(cfg, "STAGE1_QA_RANDOM_SEED", 20260130))
    qa_sample_n = int(_cfg_get(cfg, "STAGE1_QA_SAMPLE_N", 12))
    qa_force = [int(v) for v in _cfg_list(cfg, "STAGE1_QA_FORCE_INCLUDE", [0, 100, 250, 290, 400, 500])]

    stage2_window_pre = int(_cfg_get(cfg, "STAGE2_WINDOW_PRE", 30))
    stage2_window_post = int(_cfg_get(cfg, "STAGE2_WINDOW_POST", 30))
    stage2_max_seeds_total = int(_cfg_get(cfg, "STAGE2_MAX_SEEDS_TOTAL", 20))
    stage2_propagate = str(_cfg_get(cfg, "STAGE2_PROPAGATE", "both")).lower()

    output_mask_format = str(_cfg_get(cfg, "OUTPUT_MASK_FORMAT", "png")).lower()
    output_overlay_only_qa = bool(_cfg_get(cfg, "OUTPUT_OVERLAY_QA_ONLY", True))

    stage1_cfg = {
        "gdino_text_prompt": _cfg_list(cfg, "GDINO_TEXT_PROMPT", []),
        "gdino_box_th": float(_cfg_get(cfg, "GDINO_BOX_TH", 0.23)),
        "gdino_text_th": float(_cfg_get(cfg, "GDINO_TEXT_TH", 0.23)),
        "gdino_topk": int(_cfg_get(cfg, "GDINO_TOPK", 12)),
        "nms_iou_th": float(_cfg_get(cfg, "NMS_IOU_TH", 0.5)),
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
    }

    run_id = now_ts()
    run_dir = Path("runs") / f"image_stage12_sparse_seed_0010_000_500_{run_id}"
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

    model_cfg = _load_model_cfg(model_zoo, model_id)
    if not model_cfg:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "model_id_not_found"})
        raise SystemExit(f"model_id not found: {model_id}")

    missing_weights = _check_weights(model_cfg)
    if missing_weights:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_weights", "items": missing_weights})
        raise SystemExit(f"missing weights: {missing_weights}")

    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": f"missing_dependencies:{exc}"})
        raise SystemExit(f"missing dependencies: {exc}")

    device = _device_auto()

    dino_repo = model_cfg.get("dino_repo_id") or (model_cfg.get("download") or {}).get("repo")
    dino_local_dir = Path(str(model_cfg.get("dino_weights_path") or ""))
    if not dino_repo or not dino_local_dir.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "gdino_repo_or_weights_missing"})
        raise SystemExit("gdino repo or weights missing")

    processor = AutoProcessor.from_pretrained(dino_local_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_local_dir, local_files_only=True).to(device)

    sam2_cfg = str((model_cfg.get("download") or {}).get("sam2_model_cfg") or "")
    if "sam2/configs/" in sam2_cfg:
        sam2_cfg = "configs/" + sam2_cfg.split("sam2/configs/", 1)[1]
    elif sam2_cfg and not sam2_cfg.startswith("configs/"):
        sam2_cfg = f"configs/{sam2_cfg}"
    ckpt_path = _resolve_sam2_ckpt(model_cfg)
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

    stage1_stats = {"boxes_total": 0, "boxes_pass": 0, "masks_total": 0, "masks_pass": 0}
    seeds: List[Dict[str, object]] = []
    stage1_montage_inputs: List[Tuple[str, Path]] = []
    stage1_rows: List[Dict[str, object]] = []
    seed_id = 0
    stage1_qa_dir = stage1_dir / "qa_frames"
    stage1_qa_dir.mkdir(parents=True, exist_ok=True)

    for frame in stage1_frames:
        frame_id = f"{frame:010d}"
        img_path = frame_paths.get(frame_id)
        if img_path is None or not img_path.exists():
            stage1_rows.append({"frame_id": frame_id, "gdino_boxes": 0, "seed_masks": 0})
            continue
        img = Image.open(img_path).convert("RGB")
        boxes, scores, labels, candidates, stats = _run_stage1_on_image(
            img, processor, model, sam2_predictor, stage1_cfg
        )
        for k in stage1_stats:
            stage1_stats[k] += int(stats.get(k, 0))

        if candidates:
            candidates_sorted = sorted(candidates, key=lambda c: (c["score"], c["area"]), reverse=True)
            max_per_frame = int(stage1_cfg.get("max_seeds_per_frame", 4))
            if max_per_frame > 0:
                candidates_sorted = candidates_sorted[:max_per_frame]
        else:
            candidates_sorted = []

        frame_masks = [c["mask"] for c in candidates_sorted]
        stage1_rows.append({"frame_id": frame_id, "gdino_boxes": len(boxes), "seed_masks": len(frame_masks)})
        for cand in candidates_sorted:
            seeds.append(
                {
                    "seed_id": seed_id,
                    "seed_frame": frame,
                    "seed_frame_id": frame_id,
                    "score": float(cand["score"]),
                    "prompt": str(cand["label"]),
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
            seed_id += 1

        if frame in qa_frames:
            overlay_path = stage1_qa_dir / f"frame_{frame_id}_overlay_stage1.png"
            _draw_overlay_stage1(img, boxes, scores, frame_masks, overlay_path)
            stage1_montage_inputs.append((frame_id, overlay_path))
        elif not output_overlay_only_qa:
            frame_dir = stage1_frames_dir / f"frame_{frame_id}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = frame_dir / "overlay_stage1.png"
            _draw_overlay_stage1(img, boxes, scores, frame_masks, overlay_path)

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
            "score",
            "prompt",
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

    if stage1_montage_inputs:
        _montage(stage1_montage_inputs, qa_dir / "montage_stage1_qa.png")
    write_csv(
        tables_dir / "per_frame_stage1_counts.csv",
        stage1_rows,
        ["frame_id", "gdino_boxes", "seed_masks"],
    )

    if not seeds:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "no_seeds_after_stage1"})
        raise SystemExit("no seeds after stage1")

    seed_required = [s for s in seeds if int(s["seed_frame"]) == int(seed_frame)]
    if not seed_required:
        write_json(
            run_dir / "decision.json",
            {"status": "FAIL", "reason": "seed_required_missing", "seed_frame": seed_frame},
        )
        raise SystemExit(f"seed at frame {seed_frame} missing")

    seeds_sorted = sorted(seeds, key=lambda s: (s["score"], s["area"]), reverse=True)
    seeds_used = seeds_sorted[: stage2_max_seeds_total] if stage2_max_seeds_total > 0 else seeds_sorted[:]
    if seed_required:
        seed_ids = {s["seed_id"] for s in seed_required}
        if not any(s["seed_id"] in seed_ids for s in seeds_used):
            seeds_used.append(seed_required[0])
            if stage2_max_seeds_total > 0 and len(seeds_used) > stage2_max_seeds_total:
                seeds_used = sorted(
                    seeds_used, key=lambda s: (s["score"], s["area"]), reverse=True
                )[:stage2_max_seeds_total]

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
            "score",
            "prompt",
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
        _montage(qa_stage2_inputs, qa_dir / "montage_stage2_qa.png")

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
    if len(seeds_used) <= 1:
        warnings.append("seed_count_low")
    if hit_frames < 5:
        warnings.append("stage2_masks_sparse")
    if _segment_count(mask_flags) > max(1, len(seeds_used)):
        warnings.append("mask_area_fragmented")

    status = "PASS"
    reason = []
    if not seed_required:
        status = "FAIL"
        reason.append("seed_required_missing")
    if hit_frames == 0:
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
                "model_id": model_id,
                "sam2_checkpoint": str(ckpt_path),
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
        "# Image Stage1+Stage2 Sparse Seed Crosswalk (0010 f0-500)",
        "",
        f"- run_id: {run_id}",
        f"- drive_id: {drive_id}",
        f"- frame_range: {frame_start}-{frame_end}",
        f"- stage1_stride: {stage1_stride}",
        f"- stage1_frames: {stage1_frames}",
        f"- seed_frame: {seed_frame}",
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
        f"- qa montage stage1: qa/montage_stage1_qa.png",
        f"- qa montage stage2: qa/montage_stage2_qa.png",
        f"- stage2 windows: stage2/windows/seed_*",
        f"- merged masks: stage2/merged_masks/frame_*.{output_mask_format}",
        f"- per-frame area: tables/per_frame_mask_area.csv",
        f"- missing frames: tables/missing_frames.csv",
        f"- decision: decision.json",
        "",
        "## next",
        "- proceed to world candidates landing",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
