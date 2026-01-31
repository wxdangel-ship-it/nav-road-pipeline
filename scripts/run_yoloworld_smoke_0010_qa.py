from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from pipeline._io import load_yaml
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


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
    default_root = Path(r"E:\KITTI360\KITTI-360")
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


def _roi_crop(img: Image.Image, bottom_ratio: float, side_crop: Tuple[float, float]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    w, h = img.size
    x0 = int(w * float(side_crop[0]))
    x1 = int(w * float(side_crop[1]))
    y0 = int(h * float(bottom_ratio))
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    return img.crop((x0, y0, x1, h)), (x0, y0, x1, h)


def _draw_overlay(
    img: Image.Image,
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    crop_rect: Tuple[int, int, int, int],
    out_path: Path,
) -> None:
    overlay = img.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = crop_rect
    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0, 200), width=2)
    for idx, box in enumerate(boxes):
        x1b, y1b, x2b, y2b = [float(v) for v in box]
        score = scores[idx] if idx < len(scores) else None
        label = labels[idx] if idx < len(labels) else ""
        text = f"{label}:{score:.2f}" if score is not None else label
        draw.rectangle([x1b, y1b, x2b, y2b], outline=(255, 0, 0, 200), width=2)
        draw.text((x1b + 2, y1b + 2), text, fill=(255, 255, 0, 255))
    overlay.convert("RGB").save(out_path)


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
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_model_cfg(zoo_path: Path, model_id: str) -> dict:
    zoo = load_yaml(zoo_path) if zoo_path.exists() else {}
    models = zoo.get("models") or []
    for m in models:
        if str(m.get("model_id") or "") == model_id:
            return m
    return {}


def _run_yolo(
    model,
    img: Image.Image,
    prompts: List[str],
    conf_th: float,
    topk: int,
    roi_crop: Tuple[float, float],
    roi_side: Tuple[float, float],
    device: str,
) -> Tuple[List[List[float]], List[str], List[float], Tuple[int, int, int, int]]:
    roi_img, crop_rect = _roi_crop(img, roi_crop, roi_side)
    x0, y0, _x1, _y1 = crop_rect
    if prompts and hasattr(model, "set_classes"):
        model.set_classes([str(p) for p in prompts])
    results = model.predict(
        source=np.array(roi_img),
        conf=conf_th,
        imgsz=1024,
        device=device,
        verbose=False,
    )
    boxes_out: List[List[float]] = []
    labels_out: List[str] = []
    scores_out: List[float] = []
    if results:
        res = results[0]
        names = res.names or {}
        boxes = res.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
            if isinstance(names, list) and 0 <= cls_id < len(names):
                label = str(names[cls_id])
            else:
                label = str(getattr(names, "get", lambda _k, _d="": _d)(cls_id, ""))
            label = label.strip().lower()
            bbox = [float(x) for x in xyxy]
            bbox[0] += x0
            bbox[2] += x0
            bbox[1] += y0
            bbox[3] += y0
            boxes_out.append(bbox)
            labels_out.append(label)
            scores_out.append(conf)
    if topk > 0 and len(scores_out) > topk:
        order = sorted(range(len(scores_out)), key=lambda i: scores_out[i], reverse=True)[:topk]
        boxes_out = [boxes_out[i] for i in order]
        labels_out = [labels_out[i] for i in order]
        scores_out = [scores_out[i] for i in order]
    return boxes_out, labels_out, scores_out, crop_rect


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yoloworld_smoke_0010.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if not cfg:
        raise SystemExit(f"missing config: {cfg_path}")

    drive_id = str(_cfg_get(cfg, "DRIVE_ID", "2013_05_28_drive_0010_sync"))
    image_cam = str(_cfg_get(cfg, "IMAGE_CAM", "image_00"))
    frames = [int(v) for v in _cfg_list(cfg, "FRAMES", [0, 100, 250, 290, 300])]
    roi_bottom = float(_cfg_get(cfg, "ROI_BOTTOM_CROP", 0.50))
    roi_side = tuple(_cfg_get(cfg, "ROI_SIDE_CROP", [0.05, 0.95]))
    conf_th = float(_cfg_get(cfg, "YOLOWORLD_CONF_TH", 0.25))
    yolo_device = str(_cfg_get(cfg, "YOLOWORLD_DEVICE", "cpu"))
    topk = int(_cfg_get(cfg, "YOLOWORLD_TOPK", 50))
    prompts_sanity = [str(p) for p in _cfg_list(cfg, "PROMPTS_SANITY", [])]
    prompts_cross = [str(p) for p in _cfg_list(cfg, "PROMPTS_CROSSWALK", [])]
    output_overlays = bool(_cfg_get(cfg, "OUTPUT_OVERLAYS", True))
    model_zoo = Path(str(_cfg_get(cfg, "IMAGE_MODEL_ZOO", "configs/image_model_zoo.yaml")))
    yolo_model_id = str(_cfg_get(cfg, "YOLO_MODEL_ID", "yolo_world_v1"))
    overwrite = bool(_cfg_get(cfg, "OVERWRITE", True))

    run_id = now_ts()
    run_dir = Path("runs") / f"yoloworld_smoke_0010_{run_id}"
    if overwrite:
        ensure_overwrite(run_dir)
    elif run_dir.exists():
        raise SystemExit(f"run_dir exists and overwrite is false: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")
    tables_dir = run_dir / "tables"
    overlays_dir = run_dir / "overlays"
    tables_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    data_root = _find_data_root(str(_cfg_get(cfg, "KITTI_ROOT", "")))
    image_dir = _find_image_dir(data_root, drive_id, image_cam)

    model_cfg = _load_model_cfg(model_zoo, yolo_model_id)
    if not model_cfg:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "yoloworld_model_id_not_found"})
        raise SystemExit("yolo model id not found")
    weights = _resolve_yoloworld_weights(model_cfg)
    if weights is None or not weights.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "yoloworld_weights_not_found"})
        raise SystemExit("yolo weights not found")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": f"missing_ultralytics:{exc}"})
        raise SystemExit(f"missing ultralytics: {exc}")

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
    model = YOLO(str(weights))
    try:
        model.to(yolo_device)
    except Exception:
        pass
    has_set_classes = hasattr(model, "set_classes")

    report_lines = [
        "# YOLO-World Smoke (0010 QA)",
        "",
        f"- run_id: {run_id}",
        f"- drive_id: {drive_id}",
        f"- frames: {frames}",
        f"- weight_path: {weights}",
        f"- has_set_classes: {has_set_classes}",
        f"- set_classes_called: {bool(has_set_classes and (prompts_sanity or prompts_cross))}",
        f"- conf_th: {conf_th}",
        f"- yolo_device: {yolo_device}",
        f"- roi_bottom_crop: {roi_bottom}",
        f"- roi_side_crop: {roi_side}",
        f"- prompts_sanity: {prompts_sanity}",
        f"- prompts_crosswalk: {prompts_cross}",
    ]

    sanity_rows = []
    cross_rows = []
    sanity_any = False
    cross_any = False

    crop_logged = False
    for frame in frames:
        frame_id = f"{frame:010d}"
        img_path = _find_frame_path(image_dir, frame_id)
        if img_path is None:
            sanity_rows.append({"frame_id": frame_id, "count": 0, "top_scores": ""})
            cross_rows.append({"frame_id": frame_id, "count": 0, "top_scores": ""})
            continue
        img = Image.open(img_path).convert("RGB")

        boxes_s, labels_s, scores_s, crop_rect = _run_yolo(
            model,
            img,
            prompts_sanity,
            conf_th,
            topk,
            roi_bottom,
            roi_side,
            yolo_device,
        )
        if not crop_logged:
            x0, y0, x1, y1 = crop_rect
            report_lines.append(f"- crop_rect: x0={x0}, y0={y0}, w={x1 - x0}, h={y1 - y0}")
            crop_logged = True
        top_scores_s = ",".join([f"{s:.3f}" for s in scores_s[:10]])
        sanity_rows.append({"frame_id": frame_id, "count": len(boxes_s), "top_scores": top_scores_s})
        if len(boxes_s) > 0:
            sanity_any = True
        if output_overlays:
            _draw_overlay(
                img,
                boxes_s,
                labels_s,
                scores_s,
                crop_rect,
                overlays_dir / f"frame_{frame_id}_sanity.png",
            )

        boxes_c, labels_c, scores_c, _crop_rect = _run_yolo(
            model,
            img,
            prompts_cross,
            conf_th,
            topk,
            roi_bottom,
            roi_side,
            yolo_device,
        )
        top_scores_c = ",".join([f"{s:.3f}" for s in scores_c[:10]])
        cross_rows.append({"frame_id": frame_id, "count": len(boxes_c), "top_scores": top_scores_c})
        if len(boxes_c) > 0:
            cross_any = True
        if output_overlays:
            _draw_overlay(
                img,
                boxes_c,
                labels_c,
                scores_c,
                crop_rect,
                overlays_dir / f"frame_{frame_id}_crosswalk.png",
            )

    write_csv(tables_dir / "boxes_sanity.csv", sanity_rows, ["frame_id", "count", "top_scores"])
    write_csv(tables_dir / "boxes_crosswalk.csv", cross_rows, ["frame_id", "count", "top_scores"])

    if not sanity_any and conf_th >= 0.25:
        report_lines.append("- sanity_result: zero_boxes_at_conf_0_25")
        report_lines.append("- note: consider lowering conf_th to 0.15 once if allowed")
    elif sanity_any:
        report_lines.append("- sanity_result: boxes_present")
    if not cross_any:
        report_lines.append("- crosswalk_result: zero_boxes")
    else:
        report_lines.append("- crosswalk_result: boxes_present")

    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    write_json(
        run_dir / "decision.json",
        {
            "status": "OK",
            "sanity_any": sanity_any,
            "crosswalk_any": cross_any,
            "conf_th": conf_th,
            "weights": str(weights),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
