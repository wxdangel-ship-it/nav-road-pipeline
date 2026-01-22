from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None

from pipeline.evidence.image_provider_registry import ImageProvider, ProviderContext, register_provider


def _resolve_cache_env() -> dict:
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    torch_home = os.environ.get("TORCH_HOME")
    if not hf_home and not hf_hub_cache and not transformers_cache and not torch_home:
        hf_home = r"E:\hf"
        hf_hub_cache = r"E:\hf\hub"
        transformers_cache = r"E:\hf\transformers"
        torch_home = r"E:\hf\torch"
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = hf_hub_cache
        os.environ["TRANSFORMERS_CACHE"] = transformers_cache
        os.environ["TORCH_HOME"] = torch_home
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return {
        "hf_home": hf_home,
        "hf_hub_cache": hf_hub_cache,
        "transformers_cache": transformers_cache,
        "torch_home": torch_home,
    }


def _ensure_cache_env() -> str:
    cache_dirs = _resolve_cache_env()
    for key in ("hf_home", "hf_hub_cache", "transformers_cache", "torch_home"):
        val = cache_dirs.get(key)
        if val:
            Path(val).mkdir(parents=True, exist_ok=True)
    clip_cache = os.environ.get("CLIP_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    if not clip_cache:
        clip_cache = r"E:\clip"
        os.environ["CLIP_CACHE_DIR"] = clip_cache
        os.environ["XDG_CACHE_HOME"] = clip_cache
    Path(clip_cache).mkdir(parents=True, exist_ok=True)
    return cache_dirs.get("hf_home") or cache_dirs.get("hf_hub_cache") or ""


def _download_file(url: str, dst: Path) -> Path:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)
    return dst


def _resolve_sam2_checkpoint(download_cfg: dict, cache_root: str) -> Path:
    ckpt = download_cfg.get("sam2_checkpoint")
    if not ckpt:
        raise RuntimeError("sam2_checkpoint not configured")
    ckpt_path = Path(ckpt)
    if ckpt_path.exists():
        return ckpt_path
    url = download_cfg.get("sam2_checkpoint_url")
    if url:
        return _download_file(url, Path(cache_root) / "sam2" / Path(url).name)
    repo = download_cfg.get("sam2_repo")
    if repo:
        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id=repo, filename=ckpt, cache_dir=cache_root, token=False))
        except Exception as exc:  # pragma: no cover - network optional
            raise RuntimeError(f"sam2 checkpoint download failed: {exc}") from exc
    raise RuntimeError("sam2 checkpoint not found")


def _class_to_id_map(seg_schema: dict) -> dict:
    mapping = {}
    for k, v in (seg_schema.get("id_to_class") or {}).items():
        try:
            mapping[str(v)] = int(k)
        except Exception:
            continue
    return mapping


def _norm_name(name: str) -> str:
    return str(name or "").strip().lower()


def _frame_id_from_path(path: Path) -> str:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits if digits else stem


def _save_mask(path: Path, mask: np.ndarray) -> None:
    if Image is None:
        raise RuntimeError("PIL is required to write PNG masks.")
    img = Image.fromarray(mask.astype(np.uint8))
    img.save(path)


def _draw_overlay(
    img: "Image.Image",
    mask: np.ndarray,
    dets: List[dict],
    class_colors: Dict[int, Tuple[int, int, int]],
    out_path: Path,
) -> None:
    if Image is None or ImageDraw is None:
        return
    overlay = img.copy().convert("RGBA")
    if mask is not None and mask.size:
        color_img = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        arr = np.array(color_img)
        for cls_id, color in class_colors.items():
            if cls_id <= 0:
                continue
            sel = mask == cls_id
            if sel.any():
                arr[sel] = (color[0], color[1], color[2], 80)
        color_img = Image.fromarray(arr, mode="RGBA")
        overlay = Image.alpha_composite(overlay, color_img)

    draw = ImageDraw.Draw(overlay)
    for det in dets:
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        label = det.get("class") or "obj"
        score = det.get("conf")
        text = f"{label}:{score:.2f}" if score is not None else label
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 200), width=2)
        draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 0, 255))
    overlay.convert("RGB").save(out_path)


def _iter_tiles(img_size: Tuple[int, int], tile_size: int, overlap: int) -> Iterable[Tuple[int, int, int, int]]:
    width, height = img_size
    step = max(1, tile_size - overlap)
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            x1 = min(width, x0 + tile_size)
            y1 = min(height, y0 + tile_size)
            yield x0, y0, x1, y1


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


@register_provider("yolo_world")
class YoloWorldProvider(ImageProvider):
    def __init__(self, ctx: ProviderContext) -> None:
        super().__init__(ctx)
        self.model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"missing ultralytics: {exc}") from exc
        weights = (self.ctx.model_cfg.get("download") or {}).get("weights") or "yolov8s-worldv2.pt"
        clip_cache = os.environ.get("CLIP_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
        if not clip_cache:
            os.environ["CLIP_CACHE_DIR"] = r"E:\clip"
            os.environ["XDG_CACHE_HOME"] = r"E:\clip"
        self.model = YOLO(weights)
        prompts = self.ctx.model_cfg.get("prompts")
        if prompts and hasattr(self.model, "set_classes"):
            self.model.set_classes([str(p) for p in prompts])

    def infer(self, images: List[Any], out_dir: Path, debug_dir: Optional[Path] = None) -> dict:
        if self.model is None:
            self.load()
        det_dir = out_dir / "det_outputs"
        det_dir.mkdir(parents=True, exist_ok=True)
        counts: Dict[str, int] = {}
        for item in images:
            img_path = Path(item.path)
            frame_id = item.frame_id
            results = self.model.predict(
                source=str(img_path),
                device=self.ctx.device,
                conf=float((self.ctx.model_cfg.get("runtime") or {}).get("conf_threshold", 0.25)),
                iou=float((self.ctx.model_cfg.get("runtime") or {}).get("iou_threshold", 0.5)),
                imgsz=int(self.ctx.model_cfg.get("input_size", 1024)),
                verbose=False,
            )
            dets = []
            if results:
                res = results[0]
                boxes = res.boxes
                names = res.names or {}
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].tolist()
                    conf = float(boxes.conf[i].item()) if boxes.conf is not None else None
                    cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
                    cls_name = _norm_name(names.get(cls_id, ""))
                    dets.append({"class": cls_name, "bbox": [float(x) for x in xyxy], "conf": conf, "frame_id": frame_id})
                    counts[cls_name] = counts.get(cls_name, 0) + 1
            (det_dir / f"{img_path.stem}_det.json").write_text(json.dumps(dets, indent=2), encoding="utf-8")
        return {"status": "ok", "counts": counts}


def _load_dino_components(
    ctx: ProviderContext,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[Path] = None,
) -> tuple[Any, Any, List[str], Dict[str, str], Dict[str, int]]:
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except Exception as exc:
        raise RuntimeError(f"missing transformers/torch: {exc}") from exc

    download_cfg = ctx.model_cfg.get("download") or {}
    dino_repo = repo_id or download_cfg.get("repo")
    if not dino_repo:
        raise RuntimeError("grounding_dino repo not configured")

    from huggingface_hub import snapshot_download

    cache_dirs = _resolve_cache_env()
    if local_dir and local_dir.exists():
        snapshot_dir = str(local_dir)
    else:
        local_snap = (
            Path(cache_dirs["hf_hub_cache"] or cache_dirs["hf_home"] or ".")
            / "local_snapshots"
            / dino_repo.replace("/", "--")
        )
        local_snap.mkdir(parents=True, exist_ok=True)
        snapshot_dir = snapshot_download(
            repo_id=dino_repo,
            cache_dir=cache_dirs["hf_hub_cache"] or cache_dirs["hf_home"],
            token=False,
            revision=revision or "main",
            local_dir=str(local_snap),
            local_dir_use_symlinks=False,
        )

    try:
        processor = AutoProcessor.from_pretrained(
            snapshot_dir,
            local_files_only=True,
            token=False,
            cache_dir=cache_dirs["transformers_cache"] or cache_dirs["hf_home"],
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            snapshot_dir,
            local_files_only=True,
            token=False,
            cache_dir=cache_dirs["transformers_cache"] or cache_dirs["hf_home"],
        ).to(ctx.device)
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            snapshot_dir,
            local_files_only=True,
            cache_dir=cache_dirs["transformers_cache"] or cache_dirs["hf_home"],
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            snapshot_dir,
            local_files_only=True,
            cache_dir=cache_dirs["transformers_cache"] or cache_dirs["hf_home"],
        ).to(ctx.device)

    class_map = {str(k).lower(): str(v) for k, v in (ctx.model_cfg.get("class_map") or {}).items()}
    class_to_id = _class_to_id_map(ctx.seg_schema)
    prompts = ctx.model_cfg.get("prompts") or list(class_map.keys())
    prompts = [str(p).strip().lower() for p in prompts if str(p).strip()]
    return processor, model, prompts, class_map, class_to_id


def _resolve_weight_path(path_str: Optional[str], url: Optional[str], cache_root: str, tag: str) -> Path:
    if path_str:
        path = Path(path_str)
        if path.exists():
            return path
    if url:
        return _download_file(url, Path(cache_root) / tag / Path(url).name)
    raise RuntimeError(f"{tag} weights not found")


def _import_from_string(path: str):
    parts = path.split(":")
    module_name = parts[0]
    attr = parts[1] if len(parts) > 1 else None
    module = __import__(module_name, fromlist=[attr] if attr else [])
    if not attr:
        return module
    return getattr(module, attr)


class _DinoSam2Base(ImageProvider):
    def __init__(self, ctx: ProviderContext) -> None:
        super().__init__(ctx)
        self.model = None
        self.processor = None
        self.predictor = None
        self.prompts: List[str] = []
        self.class_map: Dict[str, str] = {}
        self.class_to_id: Dict[str, int] = {}
        self.prompt_packs: Dict[str, Any] = {}

    def _load_prompt_packs(self) -> Dict[str, Any]:
        pack_path = self.ctx.model_cfg.get("prompt_packs_path")
        if not pack_path:
            return {}
        try:
            import yaml

            data = yaml.safe_load(Path(pack_path).read_text(encoding="utf-8")) or {}
            return data.get("prompt_packs") or {}
        except Exception:
            return {}

    def _build_prompt_table(self, scene_profile: str) -> tuple[List[str], Dict[str, dict], Dict[str, str]]:
        packs = self.prompt_packs or {}
        table: Dict[str, dict] = {}
        prompt_to_class: Dict[str, str] = {}
        profile = scene_profile
        if profile == "sat" and "aerial" in packs:
            profile = "aerial"
        pack = packs.get(profile) or packs.get("default") or {}
        items = pack.get("items") or []
        for item in items:
            cls = str(item.get("class") or "").strip().lower()
            if not cls:
                continue
            texts = [str(t).strip().lower() for t in (item.get("texts") or []) if str(t).strip()]
            if not texts:
                continue
            table[cls] = {
                "score_thr": float(item.get("score_thr") or 0.0),
                "max_instances": int(item.get("max_instances") or 0),
                "min_box_size": float(item.get("min_box_size") or 0.0),
                "texts": texts,
            }
            for t in texts:
                prompt_to_class[t] = cls
        prompt_list: List[str] = []
        for cls, cfg in table.items():
            prompt_list.extend(cfg.get("texts") or [])
        return prompt_list, table, prompt_to_class

    def load(self) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(f"missing sam2: {exc}") from exc

        download_cfg = self.ctx.model_cfg.get("download") or {}
        sam2_cfg = download_cfg.get("sam2_model_cfg")
        if not sam2_cfg:
            raise RuntimeError("sam2 checkpoint/model_cfg not configured")
        sam2_cfg = str(sam2_cfg)
        if "sam2/configs/" in sam2_cfg:
            sam2_cfg = "configs/" + sam2_cfg.split("sam2/configs/", 1)[1]
        elif not sam2_cfg.startswith("configs/"):
            sam2_cfg = f"configs/{sam2_cfg}"
        sam2_ckpt = _resolve_sam2_checkpoint(download_cfg, _ensure_cache_env())
        self.prompt_packs = self._load_prompt_packs()
        repo_id = self.ctx.model_cfg.get("dino_repo_id") or download_cfg.get("repo")
        revision = self.ctx.model_cfg.get("dino_revision") or "main"
        local_dir = self.ctx.model_cfg.get("dino_weights_path")
        local_dir = Path(local_dir) if local_dir else None
        self.processor, self.model, self.prompts, self.class_map, self.class_to_id = _load_dino_components(
            self.ctx,
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir if local_dir and local_dir.exists() else None,
        )

        sam2 = build_sam2(sam2_cfg, str(sam2_ckpt), device=self.ctx.device)
        self.predictor = SAM2ImagePredictor(sam2)

    def _run_dino(self, img: "Image.Image", prompts: List[str]) -> dict:
        import torch

        w, h = img.size
        inputs = self.processor(images=img, text=[prompts], return_tensors="pt").to(self.ctx.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = [(h, w)]
        box_thresh = float((self.ctx.model_cfg.get("runtime") or {}).get("conf_threshold", 0.25))
        text_thresh = float((self.ctx.model_cfg.get("runtime") or {}).get("text_threshold", 0.25))
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_thresh,
                text_threshold=text_thresh,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_thresh,
                text_threshold=text_thresh,
                target_sizes=target_sizes,
            )[0]
        return results

    def _run_dino_tiled(self, img: "Image.Image", prompts: List[str]) -> dict:
        tile_size = int(self.ctx.model_cfg.get("tile_size") or 0)
        overlap = int(self.ctx.model_cfg.get("tile_overlap") or 0)
        if tile_size <= 0:
            return self._run_dino(img, prompts)
        w, h = img.size
        if w <= tile_size and h <= tile_size:
            return self._run_dino(img, prompts)
        boxes_all: List[List[float]] = []
        scores_all: List[float] = []
        labels_all: List[Any] = []
        text_labels_all: List[str] = []
        for x0, y0, x1, y1 in _iter_tiles((w, h), tile_size, overlap):
            tile = img.crop((x0, y0, x1, y1))
            results = self._run_dino(tile, prompts)
            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            labels = results.get("labels", [])
            text_labels = results.get("text_labels", [])
            for idx, box in enumerate(boxes):
                bbox = [float(v) for v in box.tolist()]
                bbox[0] += x0
                bbox[2] += x0
                bbox[1] += y0
                bbox[3] += y0
                boxes_all.append(bbox)
                scores_all.append(float(scores[idx]) if idx < len(scores) else 0.0)
                labels_all.append(labels[idx] if idx < len(labels) else -1)
                if idx < len(text_labels):
                    text_labels_all.append(text_labels[idx])
                else:
                    text_labels_all.append("")
        per_prompt = int(self.ctx.model_cfg.get("tile_max_per_prompt") or 0)
        if per_prompt > 0 and text_labels_all:
            buckets: Dict[str, List[int]] = {}
            for idx, label in enumerate(text_labels_all):
                key = str(label or "").lower()
                buckets.setdefault(key, []).append(idx)
            keep_idx = []
            for key, inds in buckets.items():
                inds_sorted = sorted(inds, key=lambda i: scores_all[i], reverse=True)[:per_prompt]
                keep_idx.extend(inds_sorted)
            keep_idx = sorted(set(keep_idx))
            boxes_all = [boxes_all[i] for i in keep_idx]
            scores_all = [scores_all[i] for i in keep_idx]
            labels_all = [labels_all[i] for i in keep_idx]
            text_labels_all = [text_labels_all[i] for i in keep_idx]

        iou_thr = float(self.ctx.model_cfg.get("global_nms_iou") or self.ctx.model_cfg.get("tile_nms_iou") or 0.5)
        keep = _nms(boxes_all, scores_all, iou_thr)
        boxes_all = [boxes_all[i] for i in keep]
        scores_all = [scores_all[i] for i in keep]
        labels_all = [labels_all[i] for i in keep]
        text_labels_all = [text_labels_all[i] for i in keep]
        return {"boxes": boxes_all, "scores": scores_all, "labels": labels_all, "text_labels": text_labels_all}

    def infer(self, images: List[Any], out_dir: Path, debug_dir: Optional[Path] = None) -> dict:
        if self.model is None or self.predictor is None:
            self.load()
        seg_dir = out_dir / "seg_masks"
        det_dir = out_dir / "det_outputs"
        raw_dir = out_dir / "raw_dino"
        seg_dir.mkdir(parents=True, exist_ok=True)
        det_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        allowed = set(self.class_to_id.keys()) | set(self.class_map.values())
        counts: Dict[str, int] = {}
        boxes_total = 0
        masks_total = 0
        for item in images:
            if Image is None:
                raise RuntimeError("PIL not available")
            img_path = Path(item.path)
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            prompt_list, prompt_table, prompt_to_class = self._build_prompt_table(item.scene_profile)
            if not prompt_list:
                prompt_list = list(self.prompts)
            results = self._run_dino_tiled(img, prompt_list)
            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            labels = results.get("labels", [])
            text_labels = results.get("text_labels", [])
            boxes_total += len(boxes)
            frame_id = item.frame_id
            dets = []
            mask = np.zeros((h, w), dtype=np.uint8)
            conf_map = np.zeros((h, w), dtype=np.float32)
            self.predictor.set_image(np.array(img))
            per_class_dets: Dict[str, List[tuple]] = {}
            for idx, box in enumerate(boxes):
                label_val = labels[idx] if idx < len(labels) else -1
                if idx < len(text_labels):
                    prompt = text_labels[idx]
                elif isinstance(label_val, int) and 0 <= label_val < len(self.prompts):
                    prompt = self.prompts[int(label_val)]
                elif isinstance(label_val, str):
                    prompt = label_val
                else:
                    prompt = ""
                prompt = _norm_name(prompt)
                mapped = prompt_to_class.get(prompt)
                if not mapped:
                    mapped = self.class_map.get(str(prompt).lower(), prompt)
                mapped = str(mapped).lower()
                if mapped not in allowed:
                    continue
                conf = float(scores[idx]) if idx < len(scores) else None
                bbox = [float(x) for x in box.tolist()] if hasattr(box, "tolist") else [float(x) for x in box]
                per_class_dets.setdefault(mapped, []).append((conf, bbox, prompt))

            max_boxes_default = int(self.ctx.model_cfg.get("max_boxes_per_prompt") or 0)
            min_box_size_default = float(self.ctx.model_cfg.get("min_box_size") or 0.0)
            for cls, items in per_class_dets.items():
                cfg = prompt_table.get(cls) or {}
                score_thr = float(cfg.get("score_thr") or 0.0)
                max_instances = int(cfg.get("max_instances") or 0) or max_boxes_default
                min_box_size = float(cfg.get("min_box_size") or 0.0) or min_box_size_default
                filtered = []
                for conf, bbox, prompt in items:
                    if conf is not None and conf < score_thr:
                        continue
                    if min_box_size > 0:
                        if (bbox[2] - bbox[0]) < min_box_size or (bbox[3] - bbox[1]) < min_box_size:
                            continue
                    filtered.append((conf, bbox, prompt))
                if max_instances > 0 and len(filtered) > max_instances:
                    logging.warning("truncate %s boxes: %d -> %d", cls, len(filtered), max_instances)
                    filtered = sorted(filtered, key=lambda x: x[0] or 0.0, reverse=True)[:max_instances]

                for conf, bbox, prompt in filtered:
                    dets.append({"class": cls, "bbox": bbox, "conf": conf, "frame_id": frame_id})
                    counts[cls] = counts.get(cls, 0) + 1
                    class_id = self.class_to_id.get(cls)
                    if class_id is None:
                        continue
                    try:
                        masks, _, _ = self.predictor.predict(box=np.array(bbox), multimask_output=False)
                    except Exception:
                        continue
                    if masks is None:
                        continue
                    mask_bin = masks[0].astype(bool)
                    conf_val = conf if conf is not None else 0.5
                    update = mask_bin & (conf_val > conf_map)
                    mask[update] = class_id
                    conf_map[update] = conf_val
                    masks_total += 1

            _save_mask(seg_dir / f"{img_path.stem}_seg.png", mask)
            (det_dir / f"{img_path.stem}_det.json").write_text(json.dumps(dets, indent=2), encoding="utf-8")
            def _to_list(val):
                if hasattr(val, "tolist"):
                    return val.tolist()
                return val

            raw_dump = {
                "frame_id": frame_id,
                "boxes": [_to_list(b) for b in boxes],
                "scores": [_to_list(s) for s in scores],
                "labels": [_to_list(l) for l in labels],
                "text_labels": text_labels,
            }
            (raw_dir / f"{img_path.stem}_dino.json").write_text(json.dumps(raw_dump, indent=2), encoding="utf-8")
            if debug_dir:
                class_colors = {cid: (255, 0, 0) for cid in self.class_to_id.values()}
                _draw_overlay(img, mask, dets, class_colors, debug_dir / f"{img_path.stem}_overlay.png")

        return {"status": "ok", "counts": counts, "boxes_total": boxes_total, "masks_total": masks_total}


@register_provider("grounded_sam2")
class GroundedSam2Provider(_DinoSam2Base):
    pass


@register_provider("gdino_sam2")
class GdinoSam2Provider(_DinoSam2Base):
    def __init__(self, ctx: ProviderContext) -> None:
        super().__init__(ctx)
        self.backend_status = "unknown"
        self.fallback_used = False
        self.fallback_from = "gdino_sam2_v1"
        self.fallback_to = ""
        self.backend_reason = ""

    def load(self) -> None:
        backend_mode = str(self.ctx.model_cfg.get("backend") or "auto").lower()
        download_cfg = self.ctx.model_cfg.get("download") or {}
        repo_id = self.ctx.model_cfg.get("dino_repo_id") or download_cfg.get("repo")
        revision = self.ctx.model_cfg.get("dino_revision") or "main"
        local_dir = self.ctx.model_cfg.get("dino_weights_path")
        local_dir = Path(local_dir) if local_dir else None
        filename = self.ctx.model_cfg.get("dino_filename")

        if backend_mode == "real":
            if not local_dir or not local_dir.exists():
                self.backend_status = "unavailable"
                self.backend_reason = "weights_not_found"
                raise RuntimeError("gdino weights not found for real backend")
            if filename and not (local_dir / filename).exists():
                self.backend_status = "unavailable"
                self.backend_reason = "weights_not_found"
                raise RuntimeError("gdino weights file missing for real backend")
            self.backend_status = "real"
            self.fallback_used = False
            self.backend_reason = ""
            self.processor, self.model, self.prompts, self.class_map, self.class_to_id = _load_dino_components(
                self.ctx, repo_id=repo_id, revision=revision, local_dir=local_dir
            )
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except Exception as exc:
                self.backend_status = "unavailable"
                self.backend_reason = "missing_dependency"
                raise RuntimeError(f"missing sam2: {exc}") from exc
            sam2_cfg = (download_cfg.get("sam2_model_cfg") or "")
            sam2_cfg = str(sam2_cfg)
            if "sam2/configs/" in sam2_cfg:
                sam2_cfg = "configs/" + sam2_cfg.split("sam2/configs/", 1)[1]
            elif sam2_cfg and not sam2_cfg.startswith("configs/"):
                sam2_cfg = f"configs/{sam2_cfg}"
            sam2_ckpt = _resolve_sam2_checkpoint(download_cfg, _ensure_cache_env())
            sam2 = build_sam2(sam2_cfg, str(sam2_ckpt), device=self.ctx.device)
            self.predictor = SAM2ImagePredictor(sam2)
            return

        try:
            local_exists = local_dir and local_dir.exists()
            self.processor, self.model, self.prompts, self.class_map, self.class_to_id = _load_dino_components(
                self.ctx, repo_id=repo_id, revision=revision, local_dir=local_dir if local_exists else None
            )
            if backend_mode == "fallback":
                self.backend_status = "fallback"
                self.fallback_used = True
                self.fallback_to = "gdino_sam2_v1@hub"
                self.backend_reason = "forced_fallback"
            elif local_exists:
                self.backend_status = "real"
                self.fallback_used = False
            else:
                self.backend_status = "fallback"
                self.fallback_used = True
                self.fallback_to = "gdino_sam2_v1@hub"
                self.backend_reason = "weights_not_found"
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except Exception as exc:
                self.backend_status = "unavailable"
                self.backend_reason = "missing_dependency"
                raise RuntimeError(f"missing sam2: {exc}") from exc
            sam2_cfg = (download_cfg.get("sam2_model_cfg") or "")
            sam2_cfg = str(sam2_cfg)
            if "sam2/configs/" in sam2_cfg:
                sam2_cfg = "configs/" + sam2_cfg.split("sam2/configs/", 1)[1]
            elif sam2_cfg and not sam2_cfg.startswith("configs/"):
                sam2_cfg = f"configs/{sam2_cfg}"
            sam2_ckpt = _resolve_sam2_checkpoint(download_cfg, _ensure_cache_env())
            sam2 = build_sam2(sam2_cfg, str(sam2_ckpt), device=self.ctx.device)
            self.predictor = SAM2ImagePredictor(sam2)
        except Exception as exc:
            if backend_mode == "fallback":
                self.backend_status = "fallback"
                self.fallback_used = True
                self.fallback_to = "gdino_sam2_v1@hub"
                self.backend_reason = "runtime_error"
                raise RuntimeError(f"gdino fallback failed: {exc}") from exc
            self.backend_status = "unavailable"
            self.backend_reason = "runtime_error"
            raise

    def infer(self, images: List[Any], out_dir: Path, debug_dir: Optional[Path] = None) -> dict:
        report = super().infer(images, out_dir, debug_dir=debug_dir)
        report.update(
            {
                "backend_status": self.backend_status,
                "fallback_used": self.fallback_used,
                "fallback_from": self.fallback_from,
                "fallback_to": self.fallback_to,
                "backend_reason": self.backend_reason,
            }
        )
        return report


@register_provider("sam3")
class Sam3Provider(_DinoSam2Base):
    def __init__(self, ctx: ProviderContext) -> None:
        super().__init__(ctx)
        self.backend_status = "fallback"
        self.fallback_used = True
        self.fallback_from = "sam3_v1"
        self.fallback_to = "gdino_sam2_v1"
        self.backend_reason = "missing_backend"
        self.use_sam3 = False

    def load(self) -> None:
        backend_mode = str(self.ctx.model_cfg.get("backend") or "auto").lower()
        if backend_mode in {"real", "auto"}:
            try:
                builder_path = str(self.ctx.model_cfg.get("sam3_builder") or "sam3.build_sam:build_sam3")
                predictor_path = str(
                    self.ctx.model_cfg.get("sam3_predictor") or "sam3.sam3_image_predictor:SAM3ImagePredictor"
                )
                build_sam3 = _import_from_string(builder_path)
                predictor_cls = _import_from_string(predictor_path)

                download_cfg = self.ctx.model_cfg.get("download") or {}
                sam3_cfg = self.ctx.model_cfg.get("sam3_model_cfg") or download_cfg.get("sam3_model_cfg")
                if not sam3_cfg:
                    raise RuntimeError("sam3_model_cfg not configured")
                sam3_cfg = str(sam3_cfg)
                if "sam3/configs/" in sam3_cfg:
                    sam3_cfg = "configs/" + sam3_cfg.split("sam3/configs/", 1)[1]
                elif not sam3_cfg.startswith("configs/"):
                    sam3_cfg = f"configs/{sam3_cfg}"

                cache_root = _ensure_cache_env()
                weights_path = _resolve_weight_path(
                    self.ctx.model_cfg.get("weights_path") or download_cfg.get("sam3_checkpoint"),
                    self.ctx.model_cfg.get("weights_url") or download_cfg.get("sam3_checkpoint_url"),
                    cache_root,
                    "sam3",
                )

                self.processor, self.model, self.prompts, self.class_map, self.class_to_id = _load_dino_components(
                    self.ctx
                )
                sam3_model = build_sam3(sam3_cfg, str(weights_path), device=self.ctx.device)
                self.predictor = predictor_cls(sam3_model)
                self.use_sam3 = True
                self.backend_status = "real"
                self.fallback_used = False
                self.fallback_to = ""
                self.backend_reason = ""
                return
            except Exception as exc:
                reason = str(exc)
                if isinstance(exc, ModuleNotFoundError) or isinstance(exc, ImportError):
                    self.backend_reason = "import_error"
                elif "weights" in reason or "checkpoint" in reason:
                    self.backend_reason = "weights_not_found"
                elif "missing" in reason or "No module" in reason:
                    self.backend_reason = "missing_dependency"
                else:
                    self.backend_reason = "runtime_error"
                logging.error("sam3 real backend error: %s", exc)
                if backend_mode == "real":
                    self.backend_status = "unavailable"
                    self.fallback_used = False
                    self.fallback_to = ""
                    raise RuntimeError(f"sam3 real backend failed: {exc}") from exc

        self.backend_status = "fallback"
        self.fallback_used = True
        self.fallback_to = "gdino_sam2_v1"
        if backend_mode == "fallback":
            self.backend_reason = "forced_fallback"
        elif not self.backend_reason:
            self.backend_reason = "missing_backend"
        logging.warning("sam3 backend not available; falling back to gdino+sam2 path")
        super().load()

    def infer(self, images: List[Any], out_dir: Path, debug_dir: Optional[Path] = None) -> dict:
        report = super().infer(images, out_dir, debug_dir=debug_dir)
        report.update(
            {
                "backend_status": self.backend_status,
                "fallback_used": self.fallback_used,
                "fallback_from": self.fallback_from,
                "fallback_to": self.fallback_to,
                "backend_reason": self.backend_reason,
            }
        )
        return report
