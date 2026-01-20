from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_seg_schema(path: Path) -> dict:
    data = _load_yaml(path)
    return data


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
    raise SystemExit(f"ERROR: image data not found for drive: {drive}, camera: {camera}")


def _ensure_cache_env() -> str:
    cache_root = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("TORCH_HOME")
    if cache_root:
        return cache_root
    cache_root = r"E:\hf_cache_shared"
    os.environ["HF_HOME"] = cache_root
    os.environ["TRANSFORMERS_CACHE"] = cache_root
    os.environ["TORCH_HOME"] = cache_root
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cache_root


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

            return Path(hf_hub_download(repo_id=repo, filename=ckpt, cache_dir=cache_root))
        except Exception as exc:
            raise RuntimeError(f"sam2 checkpoint download failed: {exc}") from exc
    raise RuntimeError("sam2 checkpoint not found")


def _device_auto() -> tuple[str, bool]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", True
    except Exception:
        pass
    return "cpu", False


def _iter_frames(img_dir: Path, max_frames: int) -> List[Path]:
    files = sorted([p for p in img_dir.glob("*.png")])
    if max_frames and max_frames > 0:
        files = files[:max_frames]
    return files


def _frame_id_from_path(path: Path) -> str:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits if digits else stem


def _write_meta(path: Path, img_size: tuple[int, int], input_size: tuple[int, int], resize_mode: str) -> None:
    meta = {
        "image_w": int(img_size[0]),
        "image_h": int(img_size[1]),
        "model_input_w": int(input_size[0]),
        "model_input_h": int(input_size[1]),
        "resize_mode": resize_mode,
        "scale": 1.0,
        "pad": [0.0, 0.0],
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _save_mask(path: Path, mask: np.ndarray) -> None:
    if Image is None:
        raise RuntimeError("PIL is required to write PNG masks.")
    img = Image.fromarray(mask.astype(np.uint8))
    img.save(path)


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


def _family_yolo_world(
    model_cfg: dict,
    images: List[Path],
    out_dir: Path,
    device: str,
    class_map: dict,
) -> dict:
    report = {"status": "fail", "reason": ""}
    try:
        from ultralytics import YOLO
    except Exception as exc:
        report["reason"] = f"missing ultralytics: {exc}"
        return report

    weights = (model_cfg.get("download") or {}).get("weights") or "yolov8s-worldv2.pt"
    model = YOLO(weights)
    prompt_classes = list(class_map.keys()) if class_map else []
    if hasattr(model, "set_classes") and prompt_classes:
        model.set_classes(prompt_classes)

    det_dir = out_dir / "det_outputs"
    det_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for img_path in images:
        results = model.predict(
            source=str(img_path),
            device=device,
            conf=float((model_cfg.get("runtime") or {}).get("conf_threshold", 0.25)),
            imgsz=int(model_cfg.get("input_size", 1024)),
            verbose=False,
        )
        frame_id = _frame_id_from_path(img_path)
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
                mapped = class_map.get(cls_name, cls_name) if class_map else cls_name
                dets.append(
                    {
                        "class": mapped,
                        "bbox": [float(x) for x in xyxy],
                        "conf": conf,
                        "track_id": None,
                        "frame_id": frame_id,
                    }
                )
                counts[mapped] = counts.get(mapped, 0) + 1
        out_path = det_dir / f"{img_path.stem}_det.json"
        out_path.write_text(json.dumps(dets, indent=2), encoding="utf-8")

    report["status"] = "ok"
    report["counts"] = counts
    return report


def _family_grounded_sam2(
    model_cfg: dict,
    images: List[Path],
    out_dir: Path,
    device: str,
    class_map: dict,
    seg_schema: dict,
) -> dict:
    report = {"status": "fail", "reason": ""}
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForObjectDetection
    except Exception as exc:
        report["reason"] = f"missing transformers/torch: {exc}"
        return report
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        report["reason"] = f"missing sam2: {exc}"
        return report

    download_cfg = model_cfg.get("download") or {}
    dino_repo = download_cfg.get("repo")
    if not dino_repo:
        report["reason"] = "grounding_dino repo not configured"
        return report
    sam2_cfg = download_cfg.get("sam2_model_cfg")
    if not sam2_cfg:
        report["reason"] = "sam2 checkpoint/model_cfg not configured"
        return report
    try:
        sam2_ckpt = _resolve_sam2_checkpoint(download_cfg, _ensure_cache_env())
    except Exception as exc:
        report["reason"] = str(exc)
        return report

    processor = AutoProcessor.from_pretrained(dino_repo)
    model = AutoModelForObjectDetection.from_pretrained(dino_repo).to(device)

    sam2 = build_sam2(sam2_cfg, str(sam2_ckpt), device=device)
    predictor = SAM2ImagePredictor(sam2)

    class_to_id = _class_to_id_map(seg_schema)
    prompts = model_cfg.get("prompts") or list(class_map.keys())

    seg_dir = out_dir / "seg_masks"
    det_dir = out_dir / "det_outputs"
    seg_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for img_path in images:
        if Image is None:
            report["reason"] = "PIL not available"
            return report
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        inputs = processor(images=img, text=prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[h, w]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            box_threshold=float((model_cfg.get("runtime") or {}).get("conf_threshold", 0.25)),
            text_threshold=0.25,
        )[0]
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("labels", [])
        phrases = results.get("phrases", [])

        frame_id = _frame_id_from_path(img_path)
        dets = []
        mask = np.zeros((h, w), dtype=np.uint8)
        conf_map = np.zeros((h, w), dtype=np.float32)

        predictor.set_image(np.array(img))
        for idx, box in enumerate(boxes):
            label_idx = int(labels[idx]) if idx < len(labels) else -1
            phrase = phrases[idx] if idx < len(phrases) else ""
            prompt = prompts[label_idx] if 0 <= label_idx < len(prompts) else phrase
            mapped = class_map.get(str(prompt).lower(), prompt)
            mapped = str(mapped).lower()
            conf = float(scores[idx]) if idx < len(scores) else None

            bbox = [float(x) for x in box.tolist()]
            dets.append(
                {"class": mapped, "bbox": bbox, "conf": conf, "track_id": None, "frame_id": frame_id}
            )
            counts[mapped] = counts.get(mapped, 0) + 1

            class_id = class_to_id.get(mapped)
            if class_id is None:
                continue
            try:
                masks, _, _ = predictor.predict(box=np.array(bbox), multimask_output=False)
            except Exception:
                continue
            if masks is None:
                continue
            mask_bin = masks[0].astype(bool)
            if conf is None:
                conf = 0.5
            update = mask_bin & (conf > conf_map)
            mask[update] = class_id
            conf_map[update] = conf

        _save_mask(seg_dir / f"{img_path.stem}_seg.png", mask)
        (det_dir / f"{img_path.stem}_det.json").write_text(json.dumps(dets, indent=2), encoding="utf-8")

    report["status"] = "ok"
    report["counts"] = counts
    return report


def run_model(
    model_cfg: dict,
    images: List[Path],
    out_dir: Path,
    device: str,
    seg_schema: dict,
) -> dict:
    family = str(model_cfg.get("family", "")).lower()
    class_map = {str(k).lower(): str(v) for k, v in (model_cfg.get("class_map") or {}).items()}
    if family == "grounded_sam2":
        return _family_grounded_sam2(model_cfg, images, out_dir, device, class_map, seg_schema)
    if family == "yolo_world":
        return _family_yolo_world(model_cfg, images, out_dir, device, class_map)
    return {"status": "fail", "reason": f"unknown family: {family}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out-run", default="")
    ap.add_argument("--camera", default="image_00")
    ap.add_argument("--zoo", default="configs/image_model_zoo.yaml")
    ap.add_argument("--seg-schema", default="configs/seg_schema.yaml")
    args = ap.parse_args()

    cache_root = _ensure_cache_env()
    device, has_cuda = _device_auto()

    max_frames = int(args.max_frames)
    if device == "cpu" and max_frames > 50:
        max_frames = 50

    data_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if not data_root:
        raise SystemExit("ERROR: POC_DATA_ROOT is not set.")
    data_root = Path(data_root)

    zoo = _load_yaml(Path(args.zoo))
    models = zoo.get("models") or []
    model_cfg = next((m for m in models if m.get("model_id") == args.model_id), None)
    if not model_cfg:
        raise SystemExit(f"ERROR: model_id not found in zoo: {args.model_id}")

    img_dir = _find_image_dir(data_root, args.drive, args.camera)
    images = _iter_frames(img_dir, max_frames)
    if not images:
        raise SystemExit("ERROR: no images found.")

    out_run = Path(args.out_run) if args.out_run else Path("runs") / f"basemodel_auto_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    model_out_dir = out_run / "model_outputs" / args.model_id / args.drive
    seg_dir = model_out_dir / "seg_masks"
    det_dir = model_out_dir / "det_outputs"
    seg_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    seg_schema = _load_seg_schema(Path(args.seg_schema))
    report = run_model(model_cfg, images, model_out_dir, device, seg_schema)
    t1 = time.time()

    if images:
        if Image is None:
            raise SystemExit("ERROR: PIL not available.")
        img = Image.open(images[0])
        _write_meta(model_out_dir / "meta.json", img.size, img.size, "none")

    infer_report = {
        "drive": args.drive,
        "model_id": args.model_id,
        "family": model_cfg.get("family"),
        "device": device,
        "cuda_available": has_cuda,
        "max_frames": max_frames,
        "frames_processed": len(images),
        "cache_root": cache_root,
        "model_out_dir": str(model_out_dir),
        "status": report.get("status"),
        "reason": report.get("reason"),
        "counts": report.get("counts"),
        "elapsed_sec": round(t1 - t0, 3),
    }
    (out_run / "infer_report.json").write_text(json.dumps(infer_report, indent=2), encoding="utf-8")
    print("[BASEMODEL] infer_report:", infer_report)
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
