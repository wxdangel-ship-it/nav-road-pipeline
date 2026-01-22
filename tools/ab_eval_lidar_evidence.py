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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--road-root", default="")
    ap.add_argument("--out", default="report.md")
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
        report_lines.append(f"- {drive}: total={total}")
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

    out_path = Path(args.out)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("wrote report: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
