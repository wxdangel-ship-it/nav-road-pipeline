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


class _DinoSam2Base(ImageProvider):
    def __init__(self, ctx: ProviderContext) -> None:
        super().__init__(ctx)
        self.model = None
        self.processor = None
        self.predictor = None
        self.prompts: List[str] = []
        self.class_map: Dict[str, str] = {}
        self.class_to_id: Dict[str, int] = {}

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except Exception as exc:
            raise RuntimeError(f"missing transformers/torch: {exc}") from exc
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(f"missing sam2: {exc}") from exc

        download_cfg = self.ctx.model_cfg.get("download") or {}
        dino_repo = download_cfg.get("repo")
        if not dino_repo:
            raise RuntimeError("grounding_dino repo not configured")
        sam2_cfg = download_cfg.get("sam2_model_cfg")
        if not sam2_cfg:
            raise RuntimeError("sam2 checkpoint/model_cfg not configured")
        sam2_cfg = str(sam2_cfg)
        if "sam2/configs/" in sam2_cfg:
            sam2_cfg = "configs/" + sam2_cfg.split("sam2/configs/", 1)[1]
        elif not sam2_cfg.startswith("configs/"):
            sam2_cfg = f"configs/{sam2_cfg}"
        sam2_ckpt = _resolve_sam2_checkpoint(download_cfg, _ensure_cache_env())

        from huggingface_hub import snapshot_download

        cache_dirs = _resolve_cache_env()
        local_dir = (
            Path(cache_dirs["hf_hub_cache"] or cache_dirs["hf_home"] or ".")
            / "local_snapshots"
            / dino_repo.replace("/", "--")
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = snapshot_download(
            repo_id=dino_repo,
            cache_dir=cache_dirs["hf_hub_cache"] or cache_dirs["hf_home"],
            token=False,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

        try:
            self.processor = AutoProcessor.from_pretrained(
                snapshot_dir,
                local_files_only=True,
                token=False,
                cache_dir=cache_dirs["transformers_cache"] or cache_dirs["hf_home"],
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                snapshot_dir,
                local_files_only=True,
                token=False,
                cache_dir=cache_dirs["transformers_cache"] or cache_dirs["hf_home"],
            ).to(self.ctx.device)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(
                snapshot_dir,
                local_files_only=True,
                cache_dir=cache_dirs["transformers_cache"] or cache_dirs["hf_home"],
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                snapshot_dir,
                local_files_only=True,
                cache_dir=cache_dirs["transformers_cache"] or cache_dirs["hf_home"],
            ).to(self.ctx.device)

        sam2 = build_sam2(sam2_cfg, str(sam2_ckpt), device=self.ctx.device)
        self.predictor = SAM2ImagePredictor(sam2)

        self.class_map = {str(k).lower(): str(v) for k, v in (self.ctx.model_cfg.get("class_map") or {}).items()}
        self.class_to_id = _class_to_id_map(self.ctx.seg_schema)
        prompts = self.ctx.model_cfg.get("prompts") or list(self.class_map.keys())
        self.prompts = [str(p).strip().lower() for p in prompts if str(p).strip()]

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

    def _run_dino_tiled(self, img: "Image.Image") -> dict:
        tile_size = int(self.ctx.model_cfg.get("tile_size") or 0)
        overlap = int(self.ctx.model_cfg.get("tile_overlap") or 0)
        if tile_size <= 0:
            return self._run_dino(img, self.prompts)
        w, h = img.size
        if w <= tile_size and h <= tile_size:
            return self._run_dino(img, self.prompts)
        boxes_all: List[List[float]] = []
        scores_all: List[float] = []
        labels_all: List[Any] = []
        text_labels_all: List[str] = []
        for x0, y0, x1, y1 in _iter_tiles((w, h), tile_size, overlap):
            tile = img.crop((x0, y0, x1, y1))
            results = self._run_dino(tile, self.prompts)
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
        iou_thr = float(self.ctx.model_cfg.get("tile_nms_iou") or 0.5)
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
            results = self._run_dino_tiled(img)
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
                mapped = self.class_map.get(str(prompt).lower(), prompt)
                mapped = str(mapped).lower()
                if mapped not in allowed:
                    continue
                conf = float(scores[idx]) if idx < len(scores) else None
                bbox = [float(x) for x in box.tolist()] if hasattr(box, "tolist") else [float(x) for x in box]
                dets.append({"class": mapped, "bbox": bbox, "conf": conf, "frame_id": frame_id})
                counts[mapped] = counts.get(mapped, 0) + 1
                class_id = self.class_to_id.get(mapped)
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
    pass


@register_provider("sam3")
class Sam3Provider(_DinoSam2Base):
    def load(self) -> None:
        logging.warning("sam3 backend not available; falling back to gdino+sam2 path")
        super().load()

    def infer(self, images: List[Any], out_dir: Path, debug_dir: Optional[Path] = None) -> dict:
        report = super().infer(images, out_dir, debug_dir=debug_dir)
        report["fallback_used"] = "gdino_sam2"
        return report
