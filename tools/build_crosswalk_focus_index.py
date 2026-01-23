from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-id", required=True)
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-end", type=int, default=100)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drive_id = args.drive_id
    image_root = Path(args.image_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame in range(args.frame_start, args.frame_end + 1):
        frame_id = f"{frame:010d}"
        image_path = image_root / f"{frame_id}.png"
        rows.append(
            {
                "drive_id": drive_id,
                "frame_id": frame_id,
                "image_path": str(image_path),
            }
        )
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
