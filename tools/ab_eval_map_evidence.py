from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import load_kitti360_pose


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("ab_eval_map_evidence")


def _read_all_layers(path: Path) -> Dict[str, gpd.GeoDataFrame]:
    layers = []
    try:
        import pyogrio

        layers = [name for name, _ in pyogrio.list_layers(path)]
    except Exception:
        layers = []
    if not layers:
        try:
            layers = gpd.io.file.fiona.listlayers(str(path))
        except Exception:
            layers = []
    out = {}
    for layer in layers:
        try:
            out[layer] = gpd.read_file(path, layer=layer)
        except Exception:
            continue
    return out


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def _build_drive_traj(data_root: Path, drive_id: str, frame_ids: List[str]) -> Optional[LineString]:
    coords = []
    for fid in frame_ids:
        try:
            x, y, _ = load_kitti360_pose(data_root, drive_id, fid)
        except Exception:
            continue
        coords.append((x, y))
    if len(coords) < 2:
        return None
    return LineString(coords)


def _collect_stats(map_path: Path, traj: Optional[LineString]) -> dict:
    layers = _read_all_layers(map_path)
    stats = {}
    for cls, gdf in layers.items():
        lengths = []
        points_counts = []
        dists = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                lengths.append(float(geom.length))
            if "points_count" in row:
                try:
                    points_counts.append(int(row.get("points_count") or 0))
                except Exception:
                    pass
            if traj is not None:
                try:
                    dists.append(float(geom.distance(traj)))
                except Exception:
                    pass
        stats[cls] = {
            "count": int(len(gdf)),
            "points_p50": _percentile(points_counts, 50),
            "points_p90": _percentile(points_counts, 90),
            "length_p50": _percentile(lengths, 50),
            "length_p90": _percentile(lengths, 90),
            "dist_p50": _percentile(dists, 50),
            "dist_p90": _percentile(dists, 90),
        }
    return stats


def _load_index(map_dir: Path) -> dict:
    path = map_dir / "index.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summary_stats(stats: dict) -> Tuple[int, Optional[float]]:
    total = 0
    dist_vals = []
    for cls, row in stats.items():
        total += int(row.get("count") or 0)
        dist = row.get("dist_p50")
        if dist is not None:
            dist_vals.append(dist)
    if dist_vals:
        return total, float(np.mean(dist_vals))
    return total, None


def _load_debug_samples(map_dir: Path, drive_id: str) -> dict:
    path = map_dir / drive_id / "map_debug_samples.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-a", required=True, help="map evidence dir A (feature_store_map_xxx)")
    ap.add_argument("--map-b", required=True, help="map evidence dir B (feature_store_map_xxx)")
    ap.add_argument("--drive", required=True)
    ap.add_argument("--frame-index", required=True, help="sample index jsonl")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", default="report_map_eval.md")
    ap.add_argument("--topn", type=int, default=5)
    args = ap.parse_args()

    log = _setup_logger()
    data_root = Path(args.data_root) if args.data_root else Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        log.error("POC_DATA_ROOT not set or invalid.")
        return 2

    frames = []
    for line in Path(args.frame_index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("drive_id") == args.drive:
            frames.append(str(row.get("frame_id")))
    traj = _build_drive_traj(data_root, args.drive, frames)

    path_a = Path(args.map_a) / args.drive / "map_evidence_utm32.gpkg"
    path_b = Path(args.map_b) / args.drive / "map_evidence_utm32.gpkg"
    if not path_a.exists() or not path_b.exists():
        log.error("map evidence not found: %s or %s", path_a, path_b)
        return 3

    stats_a = _collect_stats(path_a, traj)
    stats_b = _collect_stats(path_b, traj)
    out_path = Path(args.out)
    idx_a = _load_index(Path(args.map_a))
    idx_b = _load_index(Path(args.map_b))
    total_a, dist_a = _summary_stats(stats_a)
    total_b, dist_b = _summary_stats(stats_b)

    recommendation = "manual_review"
    if dist_a is not None and dist_b is not None:
        if dist_a < dist_b:
            recommendation = "map_a"
        elif dist_b < dist_a:
            recommendation = "map_b"
        elif total_a >= total_b:
            recommendation = "map_a"
        else:
            recommendation = "map_b"
    elif dist_a is not None:
        recommendation = "map_a"
    elif dist_b is not None:
        recommendation = "map_b"

    debug_a = _load_debug_samples(Path(args.map_a), args.drive)
    debug_b = _load_debug_samples(Path(args.map_b), args.drive)

    lines = [
        "# Map Evidence AB Report",
        "",
        f"- drive: {args.drive}",
        f"- map_a: {path_a}",
        f"- map_b: {path_b}",
        "",
        "## Summary",
        f"- map_a_mode: {idx_a.get('map_mode', 'unknown')}",
        f"- map_b_mode: {idx_b.get('map_mode', 'unknown')}",
        f"- map_a_total: {total_a} | map_b_total: {total_b}",
        f"- map_a_dist_p50_mean: {dist_a} | map_b_dist_p50_mean: {dist_b}",
        f"- recommendation: {recommendation}",
        "",
        "## Per-Class Stats",
        "",
    ]
    classes = sorted(set(stats_a.keys()) | set(stats_b.keys()))
    for cls in classes:
        a = stats_a.get(cls, {})
        b = stats_b.get(cls, {})
        lines.extend(
            [
                f"### {cls}",
                f"- A count: {a.get('count')} | B count: {b.get('count')}",
                f"- A points_p50: {a.get('points_p50')} | B points_p50: {b.get('points_p50')}",
                f"- A length_p50: {a.get('length_p50')} | B length_p50: {b.get('length_p50')}",
                f"- A dist_p50: {a.get('dist_p50')} | B dist_p50: {b.get('dist_p50')}",
                "",
            ]
        )

    if debug_a or debug_b:
        lines.extend(["## Sample Checks", ""])
        classes = sorted(set(debug_a.keys()) | set(debug_b.keys()))
        for cls in classes:
            a_samples = debug_a.get(cls, [])[: args.topn]
            b_samples = debug_b.get(cls, [])[: args.topn]
            lines.append(f"### {cls}")
            if a_samples:
                lines.append(f"- map_a_samples: {len(a_samples)}")
                for s in a_samples:
                    lines.append(f"  - {s.get('overlay')}")
            else:
                lines.append("- map_a_samples: 0")
            if b_samples:
                lines.append(f"- map_b_samples: {len(b_samples)}")
                for s in b_samples:
                    lines.append(f"  - {s.get('overlay')}")
            else:
                lines.append("- map_b_samples: 0")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote report: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
