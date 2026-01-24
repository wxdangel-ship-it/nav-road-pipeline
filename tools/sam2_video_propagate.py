from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from tools.run_image_basemodel import _ensure_cache_env, _resolve_sam2_checkpoint


def _load_yaml(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_image_size(image_path: str) -> Tuple[int, int]:
    if not image_path:
        return 0, 0
    path = Path(image_path)
    if not path.exists():
        return 0, 0
    try:
        img = Image.open(path)
        return img.size
    except Exception:
        return 0, 0


def _load_image_model_cfg(model_id: str, zoo_path: Path) -> dict:
    zoo = _load_yaml(zoo_path)
    models = zoo.get("models") or []
    for model in models:
        if str(model.get("model_id") or "") == model_id:
            return model
    return {}


def _resolve_sam2_ckpt(model_cfg: dict) -> Path | None:
    download_cfg = model_cfg.get("download") or {}
    ckpt_name = str(download_cfg.get("sam2_checkpoint") or "")
    try:
        ckpt = _resolve_sam2_checkpoint(download_cfg, _ensure_cache_env())
        return Path(ckpt)
    except Exception:
        candidates = [
            Path("cache") / "hf" / "sam2" / ckpt_name,
            Path("runs") / "cache_ps" / "sam2" / ckpt_name,
            Path("runs") / "hf_cache_user" / "sam2" / ckpt_name,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
    return None


def build_video_predictor(model_id: str) -> object | None:
    model_cfg = _load_image_model_cfg(model_id, Path("configs/image_model_zoo.yaml"))
    if not model_cfg:
        return None
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except Exception:
        return None
    ckpt = _resolve_sam2_ckpt(model_cfg)
    if ckpt is None:
        return None
    sam2_cfg = (model_cfg.get("download") or {}).get("sam2_model_cfg")
    if not sam2_cfg:
        return None
    sam2_cfg = str(sam2_cfg)
    if "sam2/configs/" in sam2_cfg:
        sam2_cfg = "configs/" + sam2_cfg.split("sam2/configs/", 1)[1]
    elif not sam2_cfg.startswith("configs/"):
        sam2_cfg = f"configs/{sam2_cfg}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_sam2_video_predictor(sam2_cfg, ckpt_path=str(ckpt), device=device)


def prepare_video_frames(
    image_dir: Path,
    frame_ids: List[str],
    out_dir: Path,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame_id in enumerate(frame_ids):
        src = image_dir / f"{frame_id}.png"
        dst = out_dir / f"{idx:05d}.jpg"
        if dst.exists():
            continue
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        img.save(dst, format="JPEG", quality=95)
    return frame_ids


def _load_mask(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L")
    return np.array(mask) > 0


def propagate_seed(
    predictor: object,
    video_dir: Path,
    seed_frame_idx: int,
    seed_mask_path: Path | None,
    seed_bbox_px: List[float] | None,
    window: int,
) -> Dict[int, np.ndarray]:
    if predictor is None:
        return {}
    inference_state = predictor.init_state(video_path=str(video_dir), offload_video_to_cpu=True)
    obj_id = 1
    if seed_mask_path is not None and seed_mask_path.exists():
        mask = _load_mask(seed_mask_path)
        if mask is None:
            return {}
        predictor.add_new_mask(inference_state, seed_frame_idx, obj_id, mask)
    elif seed_bbox_px and len(seed_bbox_px) == 4:
        predictor.add_new_points_or_box(
            inference_state,
            seed_frame_idx,
            obj_id,
            box=seed_bbox_px,
        )
    else:
        return {}

    masks_by_frame: Dict[int, np.ndarray] = {}
    for frame_idx, _obj_ids, video_masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=seed_frame_idx,
        max_frame_num_to_track=window,
        reverse=False,
    ):
        if video_masks is None or len(video_masks) == 0:
            continue
        masks_by_frame[int(frame_idx)] = video_masks[0].cpu().numpy() > 0

    for frame_idx, _obj_ids, video_masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=seed_frame_idx,
        max_frame_num_to_track=window,
        reverse=True,
    ):
        if video_masks is None or len(video_masks) == 0:
            continue
        masks_by_frame[int(frame_idx)] = video_masks[0].cpu().numpy() > 0
    return masks_by_frame


def save_masks(
    masks: Dict[int, np.ndarray],
    frame_ids: List[str],
    out_dir: Path,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    for frame_idx, mask in masks.items():
        if frame_idx < 0 or frame_idx >= len(frame_ids):
            continue
        if mask.ndim >= 3:
            mask = np.squeeze(mask)
        frame_id = frame_ids[frame_idx]
        out_path = out_dir / f"{frame_id}.png"
        img = Image.fromarray((mask.astype(np.uint8) * 255))
        img.save(out_path)
        saved[frame_id] = out_path
    return saved


def mask_area_px(mask: np.ndarray) -> float:
    if mask is None:
        return 0.0
    if mask.ndim >= 3:
        mask = np.squeeze(mask)
    return float(np.sum(mask > 0))
