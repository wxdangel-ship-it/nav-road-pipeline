from __future__ import annotations
from pathlib import Path
import argparse
import json
import datetime

from pipeline.adapters.kitti360_adapter import discover_drives, index_drive


def build_index(data_root: Path, drives: list[str] | None, max_frames: int | None) -> dict:
    drive_list = drives or discover_drives(data_root)
    tiles = [index_drive(data_root, d, max_frames=max_frames) for d in drive_list]

    total_lidar = sum(t["lidar_count"] for t in tiles)
    total_img_any = sum(t["img_any_match"] for t in tiles)
    total_pose = sum(t["pose_match"] for t in tiles)

    image_cov = (total_img_any / total_lidar) if total_lidar > 0 else 0.0
    pose_cov = (total_pose / total_lidar) if total_lidar > 0 else 0.0

    missing_pose_drives = [t["tile_id"] for t in tiles if not t["has_pose"]]

    return {
        "meta": {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "data_root": str(data_root),
            "max_frames": max_frames,
            "drive_count": len(drive_list),
            "drives": drive_list,
        },
        "summary": {
            "total_lidar": total_lidar,
            "total_img_any": total_img_any,
            "total_pose": total_pose,
            "image_coverage": round(image_cov, 4),
            "pose_coverage": round(pose_cov, 4),
            "missing_pose_drives": missing_pose_drives,
        },
        "tiles": tiles,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="KITTI-360 root, e.g. E:\\KITTI360\\KITTI-360")
    ap.add_argument("--out", default="cache\\kitti360_index.json", help="output index json path")
    ap.add_argument("--drives", default="", help="comma separated drive list (optional)")
    ap.add_argument("--max-frames", type=int, default=0, help="limit frames per drive (0=all)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.out)

    drives = [x.strip() for x in args.drives.split(",") if x.strip()] if args.drives else None
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None

    idx = build_index(data_root, drives, max_frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INDEX] OK -> {out} (drives={idx['meta']['drive_count']}, max_frames={max_frames})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
