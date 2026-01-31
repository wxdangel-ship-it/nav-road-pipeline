from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("ab_eval_lidar_evidence")


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


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


def _collect_stats(map_path: Path, road_geom) -> dict:
    layers = _read_all_layers(map_path)
    stats = {}
    for cls, gdf in layers.items():
        points_counts = []
        lengths = []
        areas = []
        in_road = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if "points_count" in row:
                try:
                    points_counts.append(int(row.get("points_count") or 0))
                except Exception:
                    pass
            if geom.geom_type in {"LineString", "MultiLineString"}:
                lengths.append(float(geom.length))
            if geom.geom_type in {"Polygon", "MultiPolygon"}:
                areas.append(float(geom.area))
            ratio = _in_road_ratio(geom, road_geom)
            if ratio is not None:
                in_road.append(float(ratio))
        stats[cls] = {
            "count": int(len(gdf)),
            "points_p50": _percentile(points_counts, 50),
            "length_p50": _percentile(lengths, 50),
            "area_p50": _percentile(areas, 50),
            "in_road_p50": _percentile(in_road, 50),
        }
    return stats


def _parse_frame_id(frame_id: str) -> Optional[int]:
    if frame_id is None:
        return None
    if isinstance(frame_id, int):
        return frame_id
    s = str(frame_id)
    if not s:
        return None
    if s.isdigit():
        return int(s)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None


def _load_debug_instances(run_dir: Path, provider: str, drive: str) -> List[dict]:
    path = run_dir / "debug" / provider / drive / "open3dis_results.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
    return rows


def _frames_hit_hist(values: List[int]) -> Dict[str, int]:
    buckets = {"1": 0, "2-3": 0, "4-7": 0, "8-15": 0, "16+": 0}
    for v in values:
        if v <= 1:
            buckets["1"] += 1
        elif v <= 3:
            buckets["2-3"] += 1
        elif v <= 7:
            buckets["4-7"] += 1
        elif v <= 15:
            buckets["8-15"] += 1
        else:
            buckets["16+"] += 1
    return buckets


def _collect_stability(
    run_dir: Path, provider: str, drive: str, road_geom, frame_stride: int, collision_thresh_m: float
) -> dict:
    import shapely.wkt

    rows = _load_debug_instances(run_dir, provider, drive)
    per_class = {}
    per_drive_instances = []
    unique_frames = set()

    for row in rows:
        props = row.get("properties") or {}
        label = props.get("label", "unknown")
        frames = props.get("frames") or []
        centroids = props.get("centroids") or []
        frames_hit = int(props.get("frames_hit") or len(frames) or 0)
        geom = None
        if row.get("geometry_wkt"):
            try:
                geom = shapely.wkt.loads(row["geometry_wkt"])
            except Exception:
                geom = None
        area = float(geom.area) if geom is not None and not geom.is_empty else None
        in_road = _in_road_ratio(geom, road_geom)
        instance_id = props.get("instance_id", "")

        frame_nums = []
        for fid in frames:
            num = _parse_frame_id(fid)
            if num is not None:
                frame_nums.append(num)
        frame_nums = sorted(set(frame_nums))
        for fid in frame_nums:
            unique_frames.add(fid)

        gaps = []
        if len(frame_nums) > 1:
            gaps = [frame_nums[i + 1] - frame_nums[i] for i in range(len(frame_nums) - 1)]

        jitter = []
        if len(centroids) > 1:
            for i in range(len(centroids) - 1):
                dx = float(centroids[i + 1][0]) - float(centroids[i][0])
                dy = float(centroids[i + 1][1]) - float(centroids[i][1])
                jitter.append(float(np.hypot(dx, dy)))

        collision = 0
        if centroids:
            xs = [float(c[0]) for c in centroids]
            ys = [float(c[1]) for c in centroids]
            max_dist = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
            if max_dist > collision_thresh_m:
                collision = 1
        else:
            max_dist = None

        per_drive_instances.append(
            {
                "instance_id": instance_id,
                "label": label,
                "frames_hit": frames_hit,
                "area": area,
                "in_road": in_road,
            }
        )

        stat = per_class.setdefault(
            label,
            {
                "frames_hit": [],
                "gaps": [],
                "jitter": [],
                "collision_count": 0,
                "instances": [],
                "hits_total": 0,
            },
        )
        stat["frames_hit"].append(frames_hit)
        stat["gaps"].extend(gaps)
        stat["jitter"].extend(jitter)
        stat["collision_count"] += collision
        stat["instances"].append(instance_id)
        stat["hits_total"] += frames_hit

    unique_frames_count = len(unique_frames)
    avg_instances_per_frame = None
    if unique_frames_count > 0:
        total_hits = sum(s.get("hits_total", 0) for s in per_class.values())
        avg_instances_per_frame = float(total_hits) / float(unique_frames_count)

    for cls, stat in per_class.items():
        stat["frames_hit_p50"] = _percentile(stat["frames_hit"], 50)
        stat["frames_hit_p90"] = _percentile(stat["frames_hit"], 90)
        stat["frames_hit_max"] = max(stat["frames_hit"]) if stat["frames_hit"] else None
        stat["frames_hit_hist"] = _frames_hit_hist([int(v) for v in stat["frames_hit"]])
        stat["gap_p50"] = _percentile(stat["gaps"], 50)
        stat["gap_p90"] = _percentile(stat["gaps"], 90)
        stat["gap_gt_2x_stride_ratio"] = (
            float(sum(1 for g in stat["gaps"] if g > frame_stride * 2)) / float(len(stat["gaps"]))
            if stat["gaps"]
            else None
        )
        stat["jitter_p50"] = _percentile(stat["jitter"], 50)
        stat["jitter_p90"] = _percentile(stat["jitter"], 90)
        stat["avg_instances_per_frame"] = (
            float(stat["hits_total"]) / float(unique_frames_count) if unique_frames_count > 0 else None
        )

    return {
        "per_class": per_class,
        "unique_instances_total": len(per_drive_instances),
        "avg_instances_per_frame": avg_instances_per_frame,
        "top_instances": sorted(per_drive_instances, key=lambda r: r["frames_hit"], reverse=True)[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--road-root", default="")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--collision-thresh-m", type=float, default=30.0)
    args = ap.parse_args()

    log = _setup_logger()
    run_dir = Path(args.run_dir)
    map_root = run_dir / f"lidar_map_{args.provider}"
    if not map_root.exists():
        log.error("lidar map dir missing: %s", map_root)
        return 2

    report_lines = [
        "# LiDAR Evidence Report",
        "",
        f"- run_dir: {run_dir}",
        f"- provider: {args.provider}",
        "",
    ]

    drives = sorted([p.name for p in map_root.iterdir() if p.is_dir()])
    road_root = Path(args.road_root) if args.road_root else None

    report_lines.append("## Per-Drive Summary")
    stability_by_drive = {}
    for drive in drives:
        map_path = map_root / drive / "lidar_evidence_utm32.gpkg"
        if not map_path.exists():
            continue
        road_geom = None
        if road_root is not None:
            road_path = _find_road_polygon(road_root, drive)
            if road_path:
                road_gdf = _load_road_polygon(road_path)
                if road_gdf is not None and not road_gdf.empty:
                    road_geom = road_gdf.geometry.union_all()
        stats = _collect_stats(map_path, road_geom)
        total = sum(int(v.get("count") or 0) for v in stats.values())
        stability = _collect_stability(
            run_dir, args.provider, drive, road_geom, args.frame_stride, args.collision_thresh_m
        )
        stability_by_drive[drive] = stability
        report_lines.append(f"- {drive}: total={total}")
    report_lines.append("")

    report_lines.append("## Stability Summary")
    report_lines.append("")
    report_lines.append("| class | frames_hit_p50 | gap_p90 | jitter_p90 | collision_count |")
    report_lines.append("| --- | --- | --- | --- | --- |")
    stability_agg = {}
    for drive in drives:
        stability = stability_by_drive.get(drive)
        if not stability:
            continue
        for cls, stat in stability.get("per_class", {}).items():
            agg = stability_agg.setdefault(
                cls,
                {"frames_hit": [], "gap_p90": [], "jitter_p90": [], "collision_count": 0, "avg_inst_pf": []},
            )
            if stat.get("frames_hit_p50") is not None:
                agg["frames_hit"].append(stat.get("frames_hit_p50"))
            if stat.get("gap_p90") is not None:
                agg["gap_p90"].append(stat.get("gap_p90"))
            if stat.get("jitter_p90") is not None:
                agg["jitter_p90"].append(stat.get("jitter_p90"))
            agg["collision_count"] += int(stat.get("collision_count") or 0)
            if stat.get("avg_instances_per_frame") is not None:
                agg["avg_inst_pf"].append(stat.get("avg_instances_per_frame"))

    for cls, agg in stability_agg.items():
        report_lines.append(
            f"| {cls} | { _percentile(agg['frames_hit'], 50) } | { _percentile(agg['gap_p90'], 50) } | "
            f"{ _percentile(agg['jitter_p90'], 50) } | { agg['collision_count'] } |"
        )
    report_lines.append("")

    report_lines.append("## Risk Flags")
    risk_lines = []
    for cls, agg in stability_agg.items():
        frames_hit_p50 = _percentile(agg["frames_hit"], 50)
        avg_inst_pf = _percentile(agg["avg_inst_pf"], 50)
        if frames_hit_p50 is not None and avg_inst_pf is not None:
            if frames_hit_p50 < 2 and avg_inst_pf > 10:
                risk_lines.append(f"- {cls}: possible over-fragmentation (frames_hit_p50<{2}, avg_inst_pf>{10})")
        if agg["collision_count"] > 0:
            risk_lines.append(f"- {cls}: possible over-merge (collision_count={agg['collision_count']})")
    if not risk_lines:
        report_lines.append("- none")
    else:
        report_lines.extend(risk_lines)
    report_lines.append("")

    report_lines.append("## Per-Class Stats")
    for drive in drives[:1]:
        map_path = map_root / drive / "lidar_evidence_utm32.gpkg"
        if not map_path.exists():
            continue
        road_geom = None
        if road_root is not None:
            road_path = _find_road_polygon(road_root, drive)
            if road_path:
                road_gdf = _load_road_polygon(road_path)
                if road_gdf is not None and not road_gdf.empty:
                    road_geom = road_gdf.geometry.union_all()
        stats = _collect_stats(map_path, road_geom)
        for cls, row in stats.items():
            report_lines.extend(
                [
                    f"### {cls}",
                    f"- count: {row.get('count')}",
                    f"- points_p50: {row.get('points_p50')}",
                    f"- length_p50: {row.get('length_p50')}",
                    f"- area_p50: {row.get('area_p50')}",
                    f"- in_road_p50: {row.get('in_road_p50')}",
                    "",
                ]
            )
        break

    report_lines.append("## Per-Drive Stability Details")
    for drive in drives:
        stability = stability_by_drive.get(drive)
        if not stability:
            continue
        report_lines.append(f"### {drive}")
        report_lines.append(f"- unique_instances_total: {stability.get('unique_instances_total')}")
        report_lines.append(f"- avg_instances_per_frame: {stability.get('avg_instances_per_frame')}")
        report_lines.append("- top10_frames_hit:")
        for item in stability.get("top_instances", []):
            report_lines.append(
                f"  - {item.get('instance_id')} | {item.get('label')} | "
                f"frames_hit={item.get('frames_hit')} | area={item.get('area')} | "
                f"in_road={item.get('in_road')}"
            )
        report_lines.append("")

        report_lines.append("#### frames_hit_hist_by_class")
        for cls, stat in stability.get("per_class", {}).items():
            report_lines.append(
                f"- {cls}: p50={stat.get('frames_hit_p50')} p90={stat.get('frames_hit_p90')} "
                f"max={stat.get('frames_hit_max')} hist={stat.get('frames_hit_hist')}"
            )
        report_lines.append("")

        report_lines.append("#### stability_by_class")
        for cls, stat in stability.get("per_class", {}).items():
            report_lines.append(
                f"- {cls}: gap_p90={stat.get('gap_p90')} jitter_p90={stat.get('jitter_p90')} "
                f"collision_count={stat.get('collision_count')} gap_gt_2x_stride_ratio={stat.get('gap_gt_2x_stride_ratio')}"
            )
        report_lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("wrote report: %s", out_path)

    if args.out_json:
        json_path = Path(args.out_json)
        payload = {
            "run_dir": str(run_dir),
            "provider": args.provider,
            "stability_summary": stability_agg,
            "per_drive": stability_by_drive,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("wrote report json: %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
