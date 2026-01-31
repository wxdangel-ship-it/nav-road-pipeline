from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("build_image_sample_index")


def _load_index_drives(path: Path) -> List[str]:
    drives = []
    if not path.exists():
        return drives
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        drive = row.get("drive_id") or row.get("drive") or row.get("tile_id")
        if drive:
            drives.append(str(drive))
    return sorted(set(drives))


def _find_image_dir(data_root: Path, drive: str, camera: str) -> Optional[Path]:
    candidates = [
        data_root / "data_2d_raw" / drive / camera / "data_rect",
        data_root / "data_2d_raw" / drive / camera / "data",
        data_root / drive / camera / "data_rect",
        data_root / drive / camera / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _list_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.glob("*.png")])


def _extract_frame_id(path: Path) -> str:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits if digits else stem


def _sample_frames(paths: List[Path], per_drive: int, stride: int) -> List[Path]:
    if not paths:
        return []
    if stride > 0:
        sampled = paths[::stride]
        return sampled[:per_drive] if per_drive > 0 else sampled
    if per_drive <= 0 or per_drive >= len(paths):
        return paths
    step = max(1, len(paths) // per_drive)
    return paths[::step][:per_drive]


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--camera", default="image_00")
    ap.add_argument("--frames-per-drive", type=int, default=30)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--scene-profile", default="car")
    args = ap.parse_args()

    log = _setup_logger()
    data_root = Path(args.data_root) if args.data_root else Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        log.error("POC_DATA_ROOT not set or invalid.")
        return 2

    drives = _load_index_drives(Path(args.index)) if args.index else []
    if not drives:
        candidate = data_root / "data_2d_raw"
        if candidate.exists():
            drives = sorted([p.name for p in candidate.iterdir() if p.is_dir()])
    if not drives:
        log.error("No drives found.")
        return 3

    out_path = Path(args.out) if args.out else Path("runs") / f"image_samples_{datetime.now():%Y%m%d_%H%M%S}" / "sample_index.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(out_path)

    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for drive in drives:
            img_dir = _find_image_dir(data_root, drive, args.camera)
            if not img_dir:
                log.warning("drive=%s camera=%s image dir not found", drive, args.camera)
                continue
            images = _list_images(img_dir)
            sampled = _sample_frames(images, args.frames_per_drive, args.stride)
            if not sampled:
                log.warning("drive=%s no images found", drive)
                continue
            for path in sampled:
                record = {
                    "drive_id": drive,
                    "camera": args.camera,
                    "frame_id": _extract_frame_id(path),
                    "image_path": str(path),
                    "scene_profile": args.scene_profile,
                }
                f.write(json.dumps(record) + "\n")
                total += 1
            log.info("drive=%s sampled=%d", drive, len(sampled))

    log.info("wrote sample index: %s (total=%d)", out_path, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
