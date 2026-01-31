from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


def _read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".png":
        if Image is None:
            raise RuntimeError("PIL is required to read PNG masks.")
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 3:
            raise RuntimeError("RGB mask PNG not supported; provide single-channel class_id PNG.")
        return arr
    raise RuntimeError(f"Unsupported mask extension: {path.suffix}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    seg_dir = Path(args.seg_dir)
    if not seg_dir.exists():
        raise SystemExit(f"ERROR: seg-dir not found: {seg_dir}")

    files = sorted([p for p in seg_dir.rglob("*") if p.suffix.lower() in {".png", ".npy"}])
    files = files[: args.max_frames] if args.max_frames > 0 else files
    if not files:
        raise SystemExit("ERROR: no mask files found.")

    unique_ids = set()
    for p in files:
        mask = _read_mask(p)
        ids = np.unique(mask.astype(np.int32)).tolist()
        unique_ids.update(ids)

    unique_list = sorted(int(i) for i in unique_ids)
    payload = {"seg_dir": str(seg_dir), "frames_scanned": len(files), "unique_ids": unique_list}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
