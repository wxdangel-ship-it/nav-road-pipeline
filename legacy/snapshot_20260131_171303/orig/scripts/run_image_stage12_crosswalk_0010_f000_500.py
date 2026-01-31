from __future__ import annotations

import argparse
import datetime as dt
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
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_json, write_text, write_csv


CFG_DEFAULT = Path("configs/image_stage12_crosswalk_0010_f000_500.yaml")


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
    default_root = Path(r"E:\KITTI360\KITTI-360")
    if default_root.exists():
        return default_root
    raise SystemExit("missing data root: set POC_DATA_ROOT or config.kitti_root")


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


def _find_latest_stage12_run(exclude: Optional[Path] = None) -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = []
    for pattern in ("image_stage12_crosswalk_0010_000_500_*", "image_stage12_crosswalk_0010_250_500_*"):
        for p in runs_dir.glob(pattern):
            if not p.is_dir():
                continue
            if exclude is not None and p.resolve() == exclude.resolve():
                continue
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


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


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shutil

        shutil.copy2(src, dst)
    except Exception:
        pass


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
) -> Tuple[List[List[float]], List[float], List[str]]:
    kept_boxes: List[List[float]] = []
    kept_scores: List[float] = []
    kept_labels: List[str] = []
    for bbox, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        aspect = w / h
        if aspect < aspect_min:
            continue
        if h > h_ratio_max * float(image_h):
            continue
        center_v = 0.5 * (y1 + y2)
        if center_v < center_v_min * float(image_h):
            continue
        kept_boxes.append(bbox)
        kept_scores.append(score)
        kept_labels.append(label)
    return kept_boxes, kept_scores, kept_labels


def _run_stage1_on_image(
    img: Image.Image,
    processor,
    model,
    sam2_predictor,
    stage1_cfg: Dict[str, object],
) -> Tuple[List[List[float]], List[float], List[str], List[np.ndarray], List[int]]:
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

    box_aspect_min = float(stage1_cfg.get("box_aspect_min", 2.0))
    box_h_ratio_max = float(stage1_cfg.get("box_h_ratio_max", 0.35))
    box_center_v_min = float(stage1_cfg.get("box_center_v_min", 0.50))
    full_boxes, full_scores, full_labels = _filter_stage1_boxes(
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
    candidates: List[Tuple[float, np.ndarray, int]] = []
    for bbox in full_boxes:
        try:
            masks, _, _ = sam2_predictor.predict(box=np.array(bbox), multimask_output=True)
        except Exception:
            continue
        if masks is None:
            continue
        for m in masks[:max_masks]:
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
            candidates.append((area, mask_bin, stripe_count))

    max_seeds = int(stage1_cfg.get("max_seeds_per_frame", 6))
    if max_seeds > 0 and len(candidates) > max_seeds:
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:max_seeds]

    seed_masks = [c[1] for c in candidates]
    stripe_counts = [int(c[2]) for c in candidates]
    return full_boxes, full_scores, full_labels, seed_masks, stripe_counts


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_DEFAULT))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if not cfg:
        raise SystemExit(f"missing config: {cfg_path}")

    drive_id = str(cfg.get("drive_id") or "")
    frame_start = int(cfg.get("frame_start", 250))
    frame_end = int(cfg.get("frame_end", 500))
    image_cam = str(cfg.get("image_cam") or "image_00")
    seed_frame = int(cfg.get("seed_frame", 290))
    qa_seed = int(cfg.get("qa_random_seed", 20260130))
    qa_sample_n = int(cfg.get("qa_sample_n", 12))
    qa_force = [int(v) for v in (cfg.get("qa_force_include") or [])]
    model_id = str(cfg.get("image_model_id") or "gdino_sam2_v1")
    model_zoo = Path(str(cfg.get("image_model_zoo") or "configs/image_model_zoo.yaml"))
    stage1_cfg = cfg.get("stage1") or {}
    stage2_cfg = cfg.get("stage2") or {}

    run_id = now_ts()
    run_dir = Path("runs") / f"image_stage12_crosswalk_0010_000_500_{run_id}"
    ensure_overwrite(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    tables_dir = run_dir / "tables"
    qa_dir = run_dir / "qa"
    stage1_dir = run_dir / "stage1" / f"frame_{seed_frame:010d}"
    stage1_qa_dir = run_dir / "stage1" / "qa_frames"
    stage2_masks_dir = run_dir / "stage2" / "masks"
    stage2_overlay_dir = run_dir / "stage2" / "overlays"
    for d in (tables_dir, qa_dir, stage1_dir, stage1_qa_dir, stage2_masks_dir, stage2_overlay_dir):
        d.mkdir(parents=True, exist_ok=True)

    data_root = _find_data_root(str(cfg.get("kitti_root") or ""))
    image_dir = _find_image_dir(data_root, drive_id, image_cam)

    existing_frames, frame_paths, missing_frames = _collect_frames(image_dir, frame_start, frame_end)
    missing_rows = [{"frame_id": fid} for fid in missing_frames]
    write_csv(tables_dir / "missing_frames.csv", missing_rows, ["frame_id"])

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

    stage1_rows: List[Dict[str, object]] = []
    stage1_montage_inputs: List[Tuple[str, Path]] = []

    model_cfg = _load_model_cfg(model_zoo, model_id)
    if not model_cfg:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "model_id_not_found"})
        raise SystemExit(f"model_id not found: {model_id}")

    missing_weights = _check_weights(model_cfg)
    if missing_weights:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_weights", "items": missing_weights})
        raise SystemExit(f"missing weights: {missing_weights}")

    try:
        import torch
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

    seed_frame_id = f"{seed_frame:010d}"
    seed_img_path = frame_paths.get(seed_frame_id)
    if seed_img_path is None or not seed_img_path.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "seed_frame_missing"})
        raise SystemExit("seed frame missing")

    seed_masks: List[np.ndarray] = []
    seed_img: Optional[Image.Image] = None
    frame_ids_sorted = [f"{f:010d}" for f in range(frame_start, frame_end + 1)]
    for frame_id in frame_ids_sorted:
        img_path = frame_paths.get(frame_id)
        if img_path is None or not img_path.exists():
            stage1_rows.append({"frame_id": frame_id, "gdino_boxes": 0, "seed_masks": 0})
            continue
        img = Image.open(img_path).convert("RGB")
        boxes, scores, labels, masks, _stripes = _run_stage1_on_image(
            img, processor, model, sam2_predictor, stage1_cfg
        )
        stage1_rows.append(
            {
                "frame_id": frame_id,
                "gdino_boxes": len(boxes),
                "seed_masks": len(masks),
            }
        )
        if frame_id == seed_frame_id:
            seed_img = img
            seed_masks = masks
            dets = []
            for bbox, score, label in zip(boxes, scores, labels):
                dets.append({"bbox": bbox, "score": score, "label": label})
            write_json(stage1_dir / "gdino_boxes.json", {"frame_id": seed_frame_id, "detections": dets})
            _draw_overlay_stage1(seed_img, boxes, scores, seed_masks, stage1_dir / "overlay_stage1.png")
            stage1_montage_inputs.append((seed_frame_id, stage1_dir / "overlay_stage1.png"))
            seed_mask_dir = stage1_dir / "sam2_masks"
            seed_mask_dir.mkdir(parents=True, exist_ok=True)
            for idx, mask in enumerate(seed_masks):
                out_path = seed_mask_dir / f"seed_{idx:02d}.png"
                _write_mask(out_path, mask, img.size)
        if int(frame_id) in qa_frames:
            qa_overlay_path = stage1_qa_dir / f"frame_{frame_id}_overlay_stage1.png"
            _draw_overlay_stage1(img, boxes, scores, masks, qa_overlay_path)
            stage1_montage_inputs.append((frame_id, qa_overlay_path))

    if not seed_masks:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "no_seed_masks"})
        raise SystemExit("no seed masks after stage1")
    if seed_img is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "seed_frame_missing"})
        raise SystemExit("seed frame missing")

    if stage1_montage_inputs:
        _montage(stage1_montage_inputs, qa_dir / "qa_montage_stage1.png")
    write_csv(
        tables_dir / "per_frame_stage1_counts.csv",
        stage1_rows,
        ["frame_id", "gdino_boxes", "seed_masks"],
    )

    video_frames = [f for f in frame_ids_sorted if f in frame_paths]
    video_dir = run_dir / "stage2" / "video_frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame_id in enumerate(video_frames):
        img_path = frame_paths[frame_id]
        dst = video_dir / f"{idx:05d}.jpg"
        if dst.exists():
            continue
        Image.open(img_path).convert("RGB").save(dst, format="JPEG", quality=95)

    seed_index = video_frames.index(seed_frame_id)
    window = len(video_frames) if stage2_cfg.get("sam2_video_max_frames") == "all" else int(stage2_cfg.get("sam2_video_max_frames", len(video_frames)))

    combined_masks: Dict[int, np.ndarray] = {}
    for idx, _mask in enumerate(seed_masks):
        seed_path = seed_mask_dir / f"seed_{idx:02d}.png"
        inference_state = sam2_video.init_state(video_path=str(video_dir), offload_video_to_cpu=True)
        sam2_video.add_new_mask(inference_state, seed_index, idx + 1, np.array(Image.open(seed_path).convert("L")) > 0)
        for frame_idx, _obj_ids, video_masks in sam2_video.propagate_in_video(
            inference_state,
            start_frame_idx=seed_index,
            max_frame_num_to_track=window,
            reverse=False,
        ):
            if video_masks is None or len(video_masks) == 0:
                continue
            mask = np.squeeze(video_masks[0].cpu().numpy()) > 0
            combined_masks[frame_idx] = mask if frame_idx not in combined_masks else (combined_masks[frame_idx] | mask)
        for frame_idx, _obj_ids, video_masks in sam2_video.propagate_in_video(
            inference_state,
            start_frame_idx=seed_index,
            max_frame_num_to_track=window,
            reverse=True,
        ):
            if video_masks is None or len(video_masks) == 0:
                continue
            mask = np.squeeze(video_masks[0].cpu().numpy()) > 0
            combined_masks[frame_idx] = mask if frame_idx not in combined_masks else (combined_masks[frame_idx] | mask)

    per_frame_rows = []
    hit_frames = 0
    for frame_id in frame_ids_sorted:
        if frame_id in frame_paths:
            idx = video_frames.index(frame_id)
            mask = combined_masks.get(idx)
            if mask is None:
                mask = np.zeros((seed_img.size[1], seed_img.size[0]), dtype=bool)
            area = float(np.sum(mask))
            if area > 0:
                hit_frames += 1
            out_mask_path = stage2_masks_dir / f"frame_{frame_id}.png"
            _write_mask(out_mask_path, mask, seed_img.size)
            if int(stage2_cfg.get("output_overlay_only_qa", 1)):
                if int(frame_id) in qa_frames:
                    img = Image.open(frame_paths[frame_id]).convert("RGB")
                    out_overlay = stage2_overlay_dir / f"frame_{frame_id}_overlay_stage2.png"
                    _draw_overlay_stage2(img, mask, out_overlay, f"frame {frame_id}")
        else:
            out_mask_path = stage2_masks_dir / f"frame_{frame_id}.png"
            _write_mask(out_mask_path, None, seed_img.size)
            area = 0.0
        per_frame_rows.append({"frame_id": frame_id, "mask_area_px": area})

    write_csv(tables_dir / "per_frame_mask_area.csv", per_frame_rows, ["frame_id", "mask_area_px"])

    montage_inputs = []
    for frame in qa_frames:
        frame_id = f"{frame:010d}"
        overlay_path = stage2_overlay_dir / f"frame_{frame_id}_overlay_stage2.png"
        montage_inputs.append((frame_id, overlay_path))
    _montage(montage_inputs, qa_dir / "qa_montage_stage2.png")
    if (qa_dir / "qa_montage_stage1.png").exists():
        _copy_if_exists(qa_dir / "qa_montage_stage1.png", qa_dir / "qa_montage_stage1_after.png")
    if (qa_dir / "qa_montage_stage2.png").exists():
        _copy_if_exists(qa_dir / "qa_montage_stage2.png", qa_dir / "qa_montage_stage2_after.png")

    prev_run = _find_latest_stage12_run(run_dir)
    if prev_run is not None:
        prev_stage1 = prev_run / "qa" / "qa_montage_stage1.png"
        if not prev_stage1.exists():
            prev_stage1 = prev_run / "qa" / "qa_montage_stage1_after.png"
        prev_stage2 = prev_run / "qa" / "qa_montage_stage2.png"
        if not prev_stage2.exists():
            prev_stage2 = prev_run / "qa" / "qa_montage_stage2_after.png"
        if prev_stage1.exists():
            _copy_if_exists(prev_stage1, qa_dir / "qa_montage_stage1_before.png")
        if prev_stage2.exists():
            _copy_if_exists(prev_stage2, qa_dir / "qa_montage_stage2_before.png")

    existing_count = len(existing_frames)
    coverage = (hit_frames / existing_count) if existing_count > 0 else 0.0
    status = "PASS" if coverage >= 0.7 else "WARN"
    decision = {
        "status": status,
        "seed_frame": seed_frame_id,
        "seed_masks": len(seed_masks),
        "coverage_ratio": round(coverage, 4),
        "frames_existing": existing_count,
        "frames_hit": hit_frames,
        "missing_frames": len(missing_frames),
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

    report_lines = [
        "# Image Stage1+Stage2 Crosswalk Primitive Evidence (0010 f0-500)",
        "",
        f"- run_id: {run_id}",
        f"- drive_id: {drive_id}",
        f"- frame_range: {frame_start}-{frame_end}",
        f"- seed_frame: {seed_frame_id}",
        f"- seed_masks: {len(seed_masks)}",
        f"- stage1_mode: per-frame",
        f"- coverage_ratio: {coverage:.3f}",
        f"- missing_frames: {len(missing_frames)}",
        "",
        "## outputs",
        f"- stage1 overlay: stage1/frame_{seed_frame_id}/overlay_stage1.png",
        f"- stage2 overlay (seed): stage2/overlays/frame_{seed_frame_id}_overlay_stage2.png",
        f"- stage2 masks: stage2/masks/frame_*.png",
        f"- qa frames: qa/qa_frames.json",
        f"- qa montage stage1 before: qa/qa_montage_stage1_before.png",
        f"- qa montage stage1 after: qa/qa_montage_stage1_after.png",
        f"- qa montage stage2 before: qa/qa_montage_stage2_before.png",
        f"- qa montage stage2 after: qa/qa_montage_stage2_after.png",
        f"- per-frame stage1 counts: tables/per_frame_stage1_counts.csv",
        f"- per-frame area: tables/per_frame_mask_area.csv",
        f"- missing frames: tables/missing_frames.csv",
        f"- decision: decision.json",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
