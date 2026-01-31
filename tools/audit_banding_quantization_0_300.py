from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import laspy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_json, write_text, write_csv

LOG = logging.getLogger("banding_audit")

# =========================
# 参数区（按需修改）
# =========================
REPO_ROOT = r"E:\Work\nav-road-pipeline"
WORLD_LAZ = r""  # auto if empty
UTM32_LAZ = r""  # auto if empty
MAX_SAMPLE_POINTS = 2_000_000


def _latest_run_with(name: str) -> Optional[Path]:
    runs = Path(REPO_ROOT) / "runs"
    candidates = []
    for p in runs.glob("lidar_fusion_0010_f000_300_*"):
        target = p / "outputs" / name
        if target.exists():
            candidates.append(target)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _sample_y(path: Path, max_points: int) -> np.ndarray:
    with laspy.open(path) as reader:
        total = reader.header.point_count
        if total <= 0:
            return np.array([], dtype=np.float64)
        if total <= max_points:
            points = reader.read()
            return points.y.astype(np.float64)
        step = max(1, total // max_points)
        ys: List[float] = []
        idx = 0
        for chunk in reader.chunk_iterator(1_000_000):
            y = np.asarray(chunk.y, dtype=np.float64)
            for j in range(0, y.size, step):
                ys.append(float(y[j]))
                idx += 1
                if idx >= max_points:
                    return np.array(ys, dtype=np.float64)
        return np.array(ys, dtype=np.float64)


def _stats_for(path: Path) -> Dict[str, object]:
    y = _sample_y(path, MAX_SAMPLE_POINTS)
    if y.size == 0:
        return {
            "path": str(path),
            "point_count_sample": 0,
            "unique_y_1mm": 0,
            "min_nonzero_dy": None,
            "dy_p50": None,
            "dy_p90": None,
            "grid_step_hint": False,
        }
    y_round = np.round(y, 3)
    y_unique = np.unique(y_round)
    dy = np.diff(np.sort(y_unique))
    nonzero = dy[dy > 0]
    min_nonzero = float(np.min(nonzero)) if nonzero.size else None
    dy_p50 = float(np.quantile(nonzero, 0.5)) if nonzero.size else None
    dy_p90 = float(np.quantile(nonzero, 0.9)) if nonzero.size else None
    grid_hint = bool(min_nonzero is not None and min_nonzero > 0.05)
    return {
        "path": str(path),
        "point_count_sample": int(y.size),
        "unique_y_1mm": int(y_unique.size),
        "min_nonzero_dy": min_nonzero,
        "dy_p50": dy_p50,
        "dy_p90": dy_p90,
        "grid_step_hint": grid_hint,
    }


def main() -> int:
    run_dir = Path(REPO_ROOT) / "runs" / f"audit_banding_0010_0_300_{now_ts()}"
    ensure_overwrite(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    world = Path(WORLD_LAZ) if WORLD_LAZ else _latest_run_with("fused_points_world.laz")
    utm32 = Path(UTM32_LAZ) if UTM32_LAZ else _latest_run_with("fused_points_utm32.laz")
    if world is None or utm32 is None:
        raise SystemExit("missing world/utm32 laz; set WORLD_LAZ/UTM32_LAZ")

    world_stats = _stats_for(world)
    utm_stats = _stats_for(utm32)

    conclusion = "B"
    w_min = world_stats["min_nonzero_dy"]
    u_min = utm_stats["min_nonzero_dy"]
    if u_min is not None and w_min is not None:
        if u_min >= 0.5 and w_min <= 0.005:
            conclusion = "A"
        elif u_min <= 0.005 and w_min <= 0.005:
            conclusion = "B"

    audit = {"world": world_stats, "utm32": utm_stats, "conclusion": conclusion}
    write_json(run_dir / "report" / "banding_audit.json", audit)
    summary = f"结论：{'A(数据量化塌陷)' if conclusion=='A' else 'B(显示端量化)'}"
    write_text(run_dir / "report" / "banding_summary.md", summary)

    # optional y stats sample
    rows = []
    rows.append(
        {
            "label": "world",
            "min_nonzero_dy": world_stats["min_nonzero_dy"],
            "dy_p50": world_stats["dy_p50"],
            "dy_p90": world_stats["dy_p90"],
            "unique_y_1mm": world_stats["unique_y_1mm"],
        }
    )
    rows.append(
        {
            "label": "utm32",
            "min_nonzero_dy": utm_stats["min_nonzero_dy"],
            "dy_p50": utm_stats["dy_p50"],
            "dy_p90": utm_stats["dy_p90"],
            "unique_y_1mm": utm_stats["unique_y_1mm"],
        }
    )
    write_csv(run_dir / "report" / "y_value_stats.csv", rows, ["label", "min_nonzero_dy", "dy_p50", "dy_p90", "unique_y_1mm"])

    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
