# -*- coding: utf-8 -*-
"""
可视化基模型输出：seg mask（class_id 单通道）+ det json bbox，并可叠加到原始车载图像上。

使用方法：
1) 修改下面的参数：MODEL_OUT_DIR / DRIVE_ID / FRAME_ID / POC_DATA_ROOT（可选）
2) 运行：python tools/vis_basemodel_frame.py
3) 输出到 OUT_DIR：
   - mask_color.png（彩色 mask）
   - mask_scaled.png（把 class_id 拉伸到 0-255 便于肉眼看）
   - overlay.png（原图+mask 叠加）
   - overlay_det.png（叠加 bbox）
"""

import os
import json
import glob
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ========== 参数区（请按你的实际路径修改） ==========
MODEL_OUT_DIR = r"E:\Work\nav-road-pipeline\runs\basemodel_select_20260120_213415\model_outputs\grounded_sam2_v1\2013_05_28_drive_0007_sync"
DRIVE_ID = "2013_05_28_drive_0007_sync"
FRAME_ID = "0000000000"  # 10位零填充
POC_DATA_ROOT = r"E:\KITTI360\KITTI-360"  # 找原图用；不想叠加原图可留空 ""
OUT_DIR = r"E:\Work\nav-road-pipeline\runs\vis_basemodel_0007"
ALPHA = 0.45  # mask 叠加透明度

# class_id -> RGB（按你们 seg_schema.yaml 的 id_to_class 自行调整）
# 这里先给一套默认配色：0=背景黑，其它为亮色
ID2COLOR = {
    0: (0, 0, 0),
    1: (255, 80, 80),    # divider_median
    2: (80, 255, 80),    # lane_marking
    3: (80, 80, 255),    # road_edge
    4: (255, 255, 80),   # stop_line
    5: (80, 255, 255),   # crosswalk
    6: (255, 80, 255),   # gore_marking
    7: (255, 160, 0),    # arrow
}
# ===============================================


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger("vis_basemodel")


def find_mask_path(model_out_dir: str, drive_id: str, frame_id: str) -> str:
    seg_dir = Path(model_out_dir) / "seg_masks"
    if not seg_dir.exists():
        raise FileNotFoundError(f"seg_masks 不存在：{seg_dir}")
    # 兼容多种命名：包含 drive_id 与 frame_id 且含 seg/mask
    pats = [
        str(seg_dir / f"*{drive_id}*{frame_id}*seg*.png"),
        str(seg_dir / f"*{frame_id}*seg*.png"),
        str(seg_dir / f"*{drive_id}*{frame_id}*mask*.png"),
        str(seg_dir / f"*{frame_id}*mask*.png"),
        str(seg_dir / f"*{drive_id}*{frame_id}*seg*.npy"),
        str(seg_dir / f"*{frame_id}*seg*.npy"),
    ]
    for pat in pats:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"未找到 mask：drive={drive_id}, frame={frame_id}，在 {seg_dir}")


def find_det_path(model_out_dir: str, drive_id: str, frame_id: str) -> str:
    det_dir = Path(model_out_dir) / "det_outputs"
    if not det_dir.exists():
        return ""
    pats = [
        str(det_dir / f"*{drive_id}*{frame_id}*det*.json"),
        str(det_dir / f"*{frame_id}*det*.json"),
    ]
    for pat in pats:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    # jsonl 也可能是聚合文件
    jsonl_hits = glob.glob(str(det_dir / "*.jsonl"))
    return jsonl_hits[0] if jsonl_hits else ""


def load_mask(mask_path: str) -> np.ndarray:
    if mask_path.lower().endswith(".npy"):
        m = np.load(mask_path)
        return m.astype(np.int32)
    im = Image.open(mask_path)
    m = np.array(im)
    # 若误读成 RGB，取第一通道（并提示）
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.int32)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in ID2COLOR.items():
        rgb[mask == cid] = col
    # 未配置的 id 给随机色（便于发现）
    unknown = np.setdiff1d(np.unique(mask), np.array(list(ID2COLOR.keys()), dtype=np.int32))
    for i, cid in enumerate(unknown.tolist()):
        col = (int((37 * i) % 255), int((91 * i) % 255), int((173 * i) % 255))
        rgb[mask == cid] = col
    return Image.fromarray(rgb, mode="RGB")


def scale_mask(mask: np.ndarray) -> Image.Image:
    """把小整数 class_id 拉伸到 0-255，便于肉眼看（不是语义色）"""
    mx = int(mask.max())
    if mx <= 0:
        scaled = np.zeros_like(mask, dtype=np.uint8)
    else:
        scaled = np.clip((mask.astype(np.float32) / mx) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def find_kitti360_image(poc_root: str, drive_id: str, frame_id: str) -> str:
    if not poc_root:
        return ""
    # KITTI-360 常见路径（尽量少扫盘）
    candidates = [
        Path(poc_root) / "data_2d_raw" / drive_id / "image_00" / "data_rect" / f"{frame_id}.png",
        Path(poc_root) / "data_2d_raw" / drive_id / "image_01" / "data_rect" / f"{frame_id}.png",
        Path(poc_root) / drive_id / "image_00" / "data_rect" / f"{frame_id}.png",
        Path(poc_root) / drive_id / "image_01" / "data_rect" / f"{frame_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def draw_dets(img: Image.Image, det_path: str, frame_id: str) -> Image.Image:
    if not det_path:
        return img
    draw = ImageDraw.Draw(img)
    dets = []
    if det_path.lower().endswith(".jsonl"):
        with open(det_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, list):
                    dets.extend([d for d in obj if str(d.get("frame_id", "")).zfill(10) == frame_id])
                else:
                    if str(obj.get("frame_id", "")).zfill(10) == frame_id:
                        dets.append(obj)
    else:
        with open(det_path, "r", encoding="utf-8") as f:
            dets = json.load(f)

    for d in dets:
        bbox = d.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cls = d.get("class", "obj")
        conf = d.get("conf", None)
        txt = f"{cls}" + (f" {conf:.2f}" if isinstance(conf, (int, float)) else "")
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1 + 2, y1 + 2), txt, fill=(255, 0, 0))
    return img


def main():
    logger = setup_logger()
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    mask_path = find_mask_path(MODEL_OUT_DIR, DRIVE_ID, FRAME_ID)
    det_path = find_det_path(MODEL_OUT_DIR, DRIVE_ID, FRAME_ID)

    mask = load_mask(mask_path)
    uniq, counts = np.unique(mask, return_counts=True)
    logger.info(f"mask_path = {mask_path}")
    logger.info(f"mask shape={mask.shape}, dtype={mask.dtype}, min={mask.min()}, max={mask.max()}")
    logger.info(f"unique ids (head) = {uniq[:20].tolist()}  (total={len(uniq)})")
    nz = int((mask > 0).sum())
    logger.info(f"nonzero_pixels = {nz}")

    # 生成彩色与拉伸灰度
    color_img = colorize_mask(mask)
    color_out = str(Path(OUT_DIR) / f"{DRIVE_ID}_{FRAME_ID}_mask_color.png")
    color_img.save(color_out)
    logger.info(f"saved: {color_out}")

    scaled_img = scale_mask(mask)
    scaled_out = str(Path(OUT_DIR) / f"{DRIVE_ID}_{FRAME_ID}_mask_scaled.png")
    scaled_img.save(scaled_out)
    logger.info(f"saved: {scaled_out}")

    # 叠加原图（如找到）
    img_path = find_kitti360_image(POC_DATA_ROOT, DRIVE_ID, FRAME_ID)
    if img_path:
        base = Image.open(img_path).convert("RGB")
        # mask 可能和原图尺寸不同（resize/letterbox），先做简单 resize 叠加（粗看效果）
        m_rgb = color_img.resize(base.size, resample=Image.NEAREST)
        overlay = Image.blend(base, m_rgb, alpha=ALPHA)
        overlay_out = str(Path(OUT_DIR) / f"{DRIVE_ID}_{FRAME_ID}_overlay.png")
        overlay.save(overlay_out)
        logger.info(f"saved: {overlay_out}")

        # 叠加 bbox
        overlay_det = overlay.copy()
        overlay_det = draw_dets(overlay_det, det_path, FRAME_ID)
        overlay_det_out = str(Path(OUT_DIR) / f"{DRIVE_ID}_{FRAME_ID}_overlay_det.png")
        overlay_det.save(overlay_det_out)
        logger.info(f"saved: {overlay_det_out}")
    else:
        logger.warning("未找到原始图像（POC_DATA_ROOT/路径模式不匹配），仅输出 mask 可视化。")
        logger.info(f"det_path = {det_path if det_path else '(none)'}")


if __name__ == "__main__":
    main()
