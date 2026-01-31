from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calib.kitti360_world import kitti_world_to_utm32, utm32_to_kitti_world, wk_to_utm32, utm32_to_wk


def main() -> int:
    data_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if not data_root:
        data_root = r"E:\KITTI360\KITTI-360"
    data_root = Path(data_root)
    drive_id = "2013_05_28_drive_0010_sync"
    frame_id = "0000000290"

    rng = random.Random(20260130)
    pts = []
    for _ in range(10):
        pts.append([rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-5, 5)])
    pts = np.array(pts, dtype=np.float64)

    wu = kitti_world_to_utm32(pts, data_root, drive_id, frame_id)
    wk = utm32_to_kitti_world(wu, data_root, drive_id, frame_id)
    err = np.linalg.norm(wk - pts, axis=1)
    wu2 = wk_to_utm32(pts, data_root, drive_id, frame_id)
    wk2 = utm32_to_wk(wu2, data_root, drive_id, frame_id)
    err2 = np.linalg.norm(wk2 - pts, axis=1)
    max_err = float(np.max(err)) if err.size else 0.0
    max_err2 = float(np.max(err2)) if err2.size else 0.0
    if max_err > 1e-6 or max_err2 > 1e-6:
        raise SystemExit(f"roundtrip_error_too_large: {max_err} {max_err2}")
    print({"status": "ok", "max_err": max_err, "max_err2": max_err2})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
