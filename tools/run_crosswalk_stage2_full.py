from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _write_config(cfg: Dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_stage2_sam2video.yaml")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--lidar-world-mode", default=None)
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--out-run", default="")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    merged = _merge(
        cfg,
        {
            "drive_id": args.drive,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "camera": args.camera,
            "lidar_world_mode": args.lidar_world_mode,
            "kitti_root": args.kitti_root,
        },
    )
    drive_id = str(merged.get("drive_id") or "unknown")
    frame_start = merged.get("frame_start")
    frame_end = merged.get("frame_end")
    if args.out_run:
        run_dir = Path(args.out_run)
    else:
        tag = drive_id.split("_")[-2] if "_" in drive_id else drive_id
        run_dir = Path("runs") / f"crosswalk_stage2_full_{tag}_{frame_start}_{frame_end}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "trial_config.yaml"
    _write_config(merged, cfg_path)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_crosswalk_monitor_range.py"),
        "--config",
        str(cfg_path),
        "--out-run",
        str(run_dir),
    ]
    if args.drive:
        cmd.extend(["--drive", args.drive])
    if args.frame_start is not None:
        cmd.extend(["--frame-start", str(args.frame_start)])
    if args.frame_end is not None:
        cmd.extend(["--frame-end", str(args.frame_end)])
    if args.kitti_root:
        cmd.extend(["--kitti-root", args.kitti_root])

    print(json.dumps({"cmd": cmd}, ensure_ascii=True))
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
