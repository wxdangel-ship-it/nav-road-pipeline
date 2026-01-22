from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
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


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


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


def _load_road_polygon(path: Path) -> Optional[gpd.GeoDataFrame]:
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    if gdf.empty:
        return None
    if gdf.crs is None:
        if "wgs84" in path.name.lower():
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.set_crs("EPSG:32632")
    if "wgs84" in path.name.lower():
        gdf = gdf.to_crs("EPSG:32632")
    return gdf


def _find_road_polygon(road_root: Path, drive_id: str) -> Optional[Path]:
    candidates = [
        road_root / drive_id / "geom_outputs" / "road_polygon_utm32.gpkg",
        road_root / drive_id / "geom_outputs" / "road_polygon_utm32.geojson",
        road_root / drive_id / "geom_outputs" / "road_polygon_wgs84.geojson",
        road_root / drive_id / "geom_outputs" / "road_polygon_wgs84.gpkg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _in_road_ratio(geom, road_geom) -> Optional[float]:
    if road_geom is None or geom is None or geom.is_empty:
        return None
    try:
        if geom.geom_type in {"LineString", "MultiLineString"}:
            total = float(geom.length)
            if total <= 0:
                return None
            return float(geom.intersection(road_geom).length) / total
        if geom.geom_type in {"Polygon", "MultiPolygon"}:
            total = float(geom.area)
            if total <= 0:
                return None
            return float(geom.intersection(road_geom).area) / total
        if geom.geom_type in {"Point", "MultiPoint"}:
            return 1.0 if geom.within(road_geom) else 0.0
    except Exception:
        return None
    return None


def _collect_class_data(
    gdf: gpd.GeoDataFrame,
    traj: Optional[LineString],
    road_geom,
    strong_only: bool,
) -> dict:
    if gdf.empty:
        return {}
    if strong_only and "evidence_strength" in gdf.columns:
        gdf = gdf[gdf["evidence_strength"] == "strong"]
    if gdf.empty:
        return {}
    lengths = []
    points_counts = []
    dists = []
    in_road = []
    frame_counts = []
    weak_count = 0
    weak_reasons = Counter()

    if "frame_id" in gdf.columns:
        frame_counts = list(gdf.groupby("frame_id").size().astype(int))

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type in {"LineString", "MultiLineString"}:
            lengths.append(float(geom.length))
        if "points_count" in row:
            try:
                points_counts.append(int(row.get("points_count") or 0))
            except Exception:
                pass
        if "evidence_strength" in row and str(row.get("evidence_strength")) == "weak":
            weak_count += 1
            reason = str(row.get("weak_reason") or "unknown")
            weak_reasons[reason] += 1
        if traj is not None:
            try:
                dists.append(float(geom.distance(traj)))
            except Exception:
                pass
        ratio = _in_road_ratio(geom, road_geom)
        if ratio is not None:
            in_road.append(float(ratio))

    return {
        "count": int(len(gdf)),
        "lengths": lengths,
        "points_counts": points_counts,
        "dists": dists,
        "in_road": in_road,
        "frame_counts": frame_counts,
        "weak_count": weak_count,
        "weak_reasons": weak_reasons,
    }


def _summarize_class(data: dict) -> dict:
    if not data:
        return {}
    return {
        "count": data.get("count", 0),
        "points_p50": _percentile(data.get("points_counts", []), 50),
        "points_p90": _percentile(data.get("points_counts", []), 90),
        "length_p10": _percentile(data.get("lengths", []), 10),
        "length_p50": _percentile(data.get("lengths", []), 50),
        "dist_p50": _percentile(data.get("dists", []), 50),
        "dist_p90": _percentile(data.get("dists", []), 90),
        "in_road_p50": _percentile(data.get("in_road", []), 50),
        "instances_per_frame_p50": _percentile(data.get("frame_counts", []), 50),
        "instances_per_frame_p90": _percentile(data.get("frame_counts", []), 90),
        "weak_ratio": float(data.get("weak_count", 0)) / float(data.get("count", 1)),
        "weak_reasons": data.get("weak_reasons", Counter()),
    }


def _collect_stats(
    map_path: Path,
    traj: Optional[LineString],
    road_geom,
    strong_only: bool,
) -> dict:
    layers = _read_all_layers(map_path)
    stats = {}
    for cls, gdf in layers.items():
        stats[cls] = _collect_class_data(gdf, traj, road_geom, strong_only)
    return stats


def _load_index(map_dir: Path) -> dict:
    path = map_dir / "index.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summary_stats(stats: dict) -> Tuple[int, Optional[float], float, Counter]:
    total = 0
    dist_vals = []
    weak_total = 0
    reason_counter: Counter = Counter()
    for cls, row in stats.items():
        total += int(row.get("count") or 0)
        dist = row.get("dist_p50")
        if dist is not None:
            dist_vals.append(dist)
        weak_total += int(round(float(row.get("weak_ratio") or 0.0) * float(row.get("count") or 0)))
        reasons = row.get("weak_reasons")
        if isinstance(reasons, Counter):
            reason_counter.update(reasons)
    weak_ratio = float(weak_total) / float(total) if total > 0 else 0.0
    if dist_vals:
        return total, float(np.mean(dist_vals)), weak_ratio, reason_counter
    return total, None, weak_ratio, reason_counter


def _mean_metric(stats: dict, key: str) -> Optional[float]:
    values = []
    for row in stats.values():
        val = row.get(key)
        if val is not None:
            values.append(float(val))
    if not values:
        return None
    return float(np.mean(values))


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
    ap.add_argument("--drive", default="")
    ap.add_argument("--drives", default="", help="comma-separated drives; overrides --drive")
    ap.add_argument("--frame-index", required=True, help="sample index jsonl")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--road-polygon", default="", help="road polygon path (optional)")
    ap.add_argument("--road-root", default="", help="root to locate per-drive road_polygon_* files")
    ap.add_argument("--road-buffer-m", type=float, default=5.0)
    ap.add_argument("--out", default="report_map_eval.md")
    ap.add_argument("--topn", type=int, default=5)
    args = ap.parse_args()

    log = _setup_logger()
    data_root = Path(args.data_root) if args.data_root else Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        log.error("POC_DATA_ROOT not set or invalid.")
        return 2

    drives = [d.strip() for d in args.drives.split(",") if d.strip()] if args.drives else [d for d in [args.drive] if d]
    if not drives:
        log.error("no drives specified")
        return 4
    strong_only = str(os.environ.get("STRONG_ONLY", "0")).strip().lower() in {"1", "true", "yes", "y"}
    out_path = Path(args.out)
    _safe_unlink(out_path)

    idx_a = _load_index(Path(args.map_a))
    idx_b = _load_index(Path(args.map_b))

    all_stats_a = {}
    all_stats_b = {}
    per_drive_lines = []

    for drive in drives:
        frames = []
        for line in Path(args.frame_index).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("drive_id") == drive:
                frames.append(str(row.get("frame_id")))
        traj = _build_drive_traj(data_root, drive, frames)

        road_geom = None
        if args.road_polygon:
            road_path = Path(args.road_polygon)
        elif args.road_root:
            road_path = _find_road_polygon(Path(args.road_root), drive)
        else:
            road_path = None
        if road_path and road_path.exists():
            road_gdf = _load_road_polygon(road_path)
            if road_gdf is not None and not road_gdf.empty:
                road_geom = road_gdf.unary_union
                if args.road_buffer_m > 0:
                    road_geom = road_geom.buffer(args.road_buffer_m)

        path_a = Path(args.map_a) / drive / "map_evidence_utm32.gpkg"
        path_b = Path(args.map_b) / drive / "map_evidence_utm32.gpkg"
        if not path_a.exists() or not path_b.exists():
            log.error("map evidence not found: %s or %s", path_a, path_b)
            return 3

        stats_a = _collect_stats(path_a, traj, road_geom, strong_only)
        stats_b = _collect_stats(path_b, traj, road_geom, strong_only)
        all_stats_a.setdefault("drives", {})[drive] = stats_a
        all_stats_b.setdefault("drives", {})[drive] = stats_b

        summary_a = {cls: _summarize_class(data) for cls, data in stats_a.items()}
        summary_b = {cls: _summarize_class(data) for cls, data in stats_b.items()}

        total_a, dist_a, weak_a, _ = _summary_stats(summary_a)
        total_b, dist_b, weak_b, _ = _summary_stats(summary_b)
        per_drive_lines.extend(
            [
                f"### {drive}",
                f"- A total: {total_a} | B total: {total_b}",
                f"- A dist_p50_mean: {dist_a} | B dist_p50_mean: {dist_b}",
                f"- A weak_ratio: {weak_a:.3f} | B weak_ratio: {weak_b:.3f}",
                "",
            ]
        )

    def _merge_stats(all_stats: dict) -> dict:
        merged: Dict[str, dict] = {}
        for drive_stats in (all_stats.get("drives") or {}).values():
            for cls, row in drive_stats.items():
                agg = merged.setdefault(
                    cls,
                    {
                        "count": 0,
                        "points_counts": [],
                        "lengths": [],
                        "dists": [],
                        "in_road": [],
                        "frame_counts": [],
                        "weak_count": 0,
                        "weak_reasons": Counter(),
                    },
                )
                agg["count"] += int(row.get("count") or 0)
                agg["points_counts"].extend(row.get("points_counts", []))
                agg["lengths"].extend(row.get("lengths", []))
                agg["dists"].extend(row.get("dists", []))
                agg["in_road"].extend(row.get("in_road", []))
                agg["frame_counts"].extend(row.get("frame_counts", []))
                agg["weak_count"] += int(row.get("weak_count") or 0)
                reasons = row.get("weak_reasons")
                if isinstance(reasons, Counter):
                    agg["weak_reasons"].update(reasons)
        return {cls: _summarize_class(data) for cls, data in merged.items()}

    stats_a = _merge_stats(all_stats_a)
    stats_b = _merge_stats(all_stats_b)
    total_a, dist_a, weak_a, reasons_a = _summary_stats(stats_a)
    total_b, dist_b, weak_b, reasons_b = _summary_stats(stats_b)

    in_road_a = _mean_metric(stats_a, "in_road_p50")
    in_road_b = _mean_metric(stats_b, "in_road_p50")
    inst_p90_a = _mean_metric(stats_a, "instances_per_frame_p90")
    inst_p90_b = _mean_metric(stats_b, "instances_per_frame_p90")

    recommendation = "manual_review"
    risks = []
    if in_road_a is None or in_road_b is None:
        risks.append("in_road_ratio_missing")
    elif in_road_b + 0.05 < in_road_a:
        risks.append("in_road_ratio_drop")
    if weak_b >= 0.60:
        risks.append("weak_ratio_high")
    if inst_p90_a is not None and inst_p90_b is not None and inst_p90_b > inst_p90_a * 3.0:
        risks.append("fragmentation_high")

    if not risks:
        recommendation = "map_b"
    else:
        recommendation = "map_a"

    debug_a = _load_debug_samples(Path(args.map_a), drives[0])
    debug_b = _load_debug_samples(Path(args.map_b), drives[0])

    lines = [
        "# Map Evidence AB Report",
        "",
        f"- drives: {', '.join(drives)}",
        f"- map_a: {args.map_a}",
        f"- map_b: {args.map_b}",
        f"- strong_only: {strong_only}",
        "",
        "## Summary",
        f"- map_a_mode: {idx_a.get('map_mode', 'unknown')}",
        f"- map_b_mode: {idx_b.get('map_mode', 'unknown')}",
        f"- map_a_total: {total_a} | map_b_total: {total_b}",
        f"- map_a_dist_p50_mean: {dist_a} | map_b_dist_p50_mean: {dist_b}",
        f"- map_a_in_road_p50_mean: {in_road_a} | map_b_in_road_p50_mean: {in_road_b}",
        f"- map_a_inst_p90_mean: {inst_p90_a} | map_b_inst_p90_mean: {inst_p90_b}",
        f"- map_a_weak_ratio: {weak_a:.3f} | map_b_weak_ratio: {weak_b:.3f}",
        f"- map_a_weak_top: {reasons_a.most_common(3)}",
        f"- map_b_weak_top: {reasons_b.most_common(3)}",
        f"- recommendation: {recommendation}",
        f"- risk_flags: {', '.join(risks) if risks else 'none'}",
        "- note: dist_to_trajectory is auxiliary and may be biased",
        "",
        "## Per-Drive Summary",
        *per_drive_lines,
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
                f"- A inst_per_frame_p50: {a.get('instances_per_frame_p50')} | B inst_per_frame_p50: {b.get('instances_per_frame_p50')}",
                f"- A inst_per_frame_p90: {a.get('instances_per_frame_p90')} | B inst_per_frame_p90: {b.get('instances_per_frame_p90')}",
                f"- A length_p10: {a.get('length_p10')} | B length_p10: {b.get('length_p10')}",
                f"- A length_p50: {a.get('length_p50')} | B length_p50: {b.get('length_p50')}",
                f"- A in_road_p50: {a.get('in_road_p50')} | B in_road_p50: {b.get('in_road_p50')}",
                f"- A weak_ratio: {a.get('weak_ratio')} | B weak_ratio: {b.get('weak_ratio')}",
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
