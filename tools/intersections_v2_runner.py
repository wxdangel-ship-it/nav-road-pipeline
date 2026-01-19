from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon, shape, mapping
from shapely.ops import unary_union, transform as geom_transform
from pyproj import Transformer
import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.intersection_shape import (
    arm_count as _arm_count,
    aspect_ratio as _aspect_ratio,
    circularity as _circularity,
    overlap_with_road as _overlap_with_road,
    refine_intersection_polygon as _refine_intersection_polygon,
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_index(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def _read_geojson(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", []) or []


def _write_geojson(path: Path, features: List[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_wgs84(features: List[dict], crs_epsg: int) -> List[dict]:
    wgs84 = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)
    out = []
    for feat in features:
        geom = geom_transform(wgs84.transform, shape(feat["geometry"]))
        out.append({"type": "Feature", "geometry": mapping(geom), "properties": feat.get("properties") or {}})
    return out


def _find_latest_osm_path() -> Optional[Path]:
    candidates = list(Path("runs").rglob("drivable_roads.geojson"))
    candidates.extend(Path("runs").rglob("osm_ref_roads.geojson"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _entry_bbox_wgs84(entry: dict) -> Optional[Tuple[float, float, float, float]]:
    for key in ("bbox_wgs84", "bbox4326", "bbox"):
        bbox = entry.get(key)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return None


def _infer_bbox_from_road(road_poly: Polygon) -> Optional[Tuple[float, float, float, float]]:
    if road_poly is None or road_poly.is_empty:
        return None
    minx, miny, maxx, maxy = road_poly.bounds
    wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    xs, ys = wgs84.transform([minx, maxx], [miny, maxy])
    return min(xs), min(ys), max(xs), max(ys)


def _assign_sat_features_to_drives(
    sat_feats: List[dict],
    entries: List[dict],
    road_polys: Dict[str, Polygon],
) -> Tuple[Dict[str, List[dict]], dict, List[dict]]:
    by_drive: Dict[str, List[dict]] = {}
    unmatched = 0
    unmatched_feats: List[dict] = []
    for feat in sat_feats:
        geom = feat.get("geometry")
        if not geom:
            unmatched += 1
            continue
        shp = shape(geom)
        if shp.is_empty:
            unmatched += 1
            continue
        c = shp.centroid
        matched = None
        best_area = 0.0
        for entry in entries:
            drive = str(entry.get("drive") or "")
            bbox = _entry_bbox_wgs84(entry)
            if bbox is None:
                bbox = _infer_bbox_from_road(road_polys.get(drive))
            if bbox is None:
                continue
            minx, miny, maxx, maxy = bbox
            if minx <= c.x <= maxx and miny <= c.y <= maxy:
                area = (maxx - minx) * (maxy - miny)
                if area >= best_area:
                    matched = drive
                    best_area = area
        if matched is None:
            unmatched += 1
            unmatched_feats.append(feat)
            continue
        props = feat.get("properties") or {}
        props["drive_id"] = matched
        props["tile_id"] = matched
        feat["properties"] = props
        by_drive.setdefault(matched, []).append(feat)
    stats = {
        "sat_total": len(sat_feats),
        "sat_matched": sum(len(v) for v in by_drive.values()),
        "sat_unmatched": unmatched,
    }
    return by_drive, stats, unmatched_feats


def _collect_lines(features: List[dict]) -> List[LineString]:
    lines = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shape(geom)
        if shp.is_empty:
            continue
        if shp.geom_type == "LineString":
            lines.append(shp)
        elif shp.geom_type == "MultiLineString":
            lines.extend(list(shp.geoms))
    return lines


def _collect_polys(features: List[dict]) -> List[Polygon]:
    polys = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shape(geom)
        if shp.is_empty:
            continue
        if shp.geom_type == "Polygon":
            polys.append(shp)
        elif shp.geom_type == "MultiPolygon":
            polys.extend(list(shp.geoms))
    return polys


def _filter_osm_features(features: List[dict], allowlist: List[str]) -> List[dict]:
    if not allowlist:
        return features
    allow = {str(a).strip() for a in allowlist if str(a).strip()}
    if not allow:
        return features
    kept = []
    for feat in features:
        props = feat.get("properties") or {}
        hw = props.get("highway")
        if hw is None:
            continue
        if isinstance(hw, (list, tuple)):
            values = {str(v).strip() for v in hw}
        else:
            values = {str(hw).strip()}
        if values & allow:
            kept.append(feat)
    return kept


def _snap_point(point: Point, nodes: List[Tuple[float, float]], grid: Dict[Tuple[int, int], List[int]], snap_m: float) -> int:
    if snap_m <= 0:
        nodes.append((point.x, point.y))
        return len(nodes) - 1
    cell = int(point.x // snap_m), int(point.y // snap_m)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            idxs = grid.get((cell[0] + dx, cell[1] + dy), [])
            for idx in idxs:
                x, y = nodes[idx]
                if point.distance(Point(x, y)) <= snap_m:
                    return idx
    nodes.append((point.x, point.y))
    idx = len(nodes) - 1
    grid.setdefault(cell, []).append(idx)
    return idx


def _osm_degree_seeds(lines: List[LineString], snap_m: float, min_degree: int) -> List[Tuple[Point, int]]:
    nodes: List[Tuple[float, float]] = []
    grid: Dict[Tuple[int, int], List[int]] = {}
    edges: Dict[int, set] = {}
    for line in lines:
        if line.is_empty or len(line.coords) < 2:
            continue
        p1 = Point(line.coords[0])
        p2 = Point(line.coords[-1])
        n1 = _snap_point(p1, nodes, grid, snap_m)
        n2 = _snap_point(p2, nodes, grid, snap_m)
        if n1 == n2:
            continue
        edges.setdefault(n1, set()).add(n2)
        edges.setdefault(n2, set()).add(n1)
    seeds = []
    for idx, neighbors in edges.items():
        degree = len(neighbors)
        if degree >= min_degree:
            x, y = nodes[idx]
            seeds.append((Point(x, y), degree))
    return seeds


def _traj_seeds_from_line(line: LineString, angle_deg: float, min_sep_m: float) -> List[Point]:
    coords = list(line.coords)
    if len(coords) < 3:
        return []
    out = []
    last = None
    thr = math.radians(angle_deg)
    for i in range(1, len(coords) - 1):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 == 0 or n2 == 0:
            continue
        cos_ang = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos_ang = max(-1.0, min(1.0, cos_ang))
        ang = math.acos(cos_ang)
        if ang >= thr:
            pt = Point(x1, y1)
            if last is None or pt.distance(last) >= min_sep_m:
                out.append(pt)
                last = pt
    return out


def _centerline_junctions(lines: List[LineString]) -> List[Point]:
    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            inter = lines[i].intersection(lines[j])
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                points.append(inter)
            elif inter.geom_type == "MultiPoint":
                points.extend(list(inter.geoms))
    return points


def _nearest_point_on_lines(pt: Point, lines: List[LineString]) -> Optional[Point]:
    best = None
    best_dist = float("inf")
    for line in lines:
        proj = line.project(pt)
        candidate = line.interpolate(proj)
        dist = pt.distance(candidate)
        if dist < best_dist:
            best = candidate
            best_dist = dist
    return best


def _refine_seed(seed: Point, lines: List[LineString], search_radius: float) -> Tuple[Point, str, float]:
    if not lines:
        return seed, "none", 0.0
    junctions = _centerline_junctions(lines)
    best = None
    best_dist = float("inf")
    for j in junctions:
        dist = seed.distance(j)
        if dist < best_dist:
            best_dist = dist
            best = j
    if best is not None and best_dist <= search_radius:
        conf = max(0.0, 1.0 - best_dist / max(1.0, search_radius))
        return best, "geometry", conf
    candidate = _nearest_point_on_lines(seed, lines)
    if candidate is not None and seed.distance(candidate) <= search_radius:
        dist = seed.distance(candidate)
        conf = max(0.0, 1.0 - dist / max(1.0, search_radius))
        return candidate, "geometry", conf
    return seed, "none", 0.0


def _shape_from_seed(
    seed: Point,
    road_poly: Polygon,
    centerlines: List[LineString],
    cfg: dict,
) -> Tuple[Optional[Polygon], dict]:
    refined, meta = _refine_intersection_polygon(
        seed_pt=seed,
        poly_candidate=seed.buffer(float(cfg["radius_m"])),
        road_polygon=road_poly,
        centerlines=centerlines,
        cfg=cfg,
    )
    if refined is None or refined.is_empty:
        return None, {"reason": "empty"}
    local = meta.get("local")
    arms = meta.get("arms")
    circ = _circularity(refined)
    aspect = _aspect_ratio(refined)
    overlap = _overlap_with_road(refined, road_poly)
    arms = _arm_count(refined, centerlines, float(cfg["arm_buffer_m"]))
    metrics = {
        "circularity": circ,
        "aspect_ratio": aspect,
        "overlap_road": overlap,
        "arm_count": arms,
        "has_arms": 1 if meta.get("arms") is not None and not meta.get("arms").is_empty else 0,
        "arms_area": float(meta.get("arms").area) if meta.get("arms") is not None and not meta.get("arms").is_empty else 0.0,
        "local_area": float(local.area) if local is not None and not local.is_empty else 0.0,
        "refined_area": float(refined.area),
        "local": local,
        "arms": meta.get("arms"),
        "reason": meta.get("reason") or "refined",
    }
    return refined, metrics


def _shape_gate_ok(metrics: dict, gate: dict, area_m2: float) -> bool:
    if area_m2 < gate["min_area_m2"] or area_m2 > gate["max_area_m2"]:
        return False
    if metrics["overlap_road"] < gate["min_overlap_road"]:
        return False
    if metrics["circularity"] > gate["max_circularity"]:
        return False
    if metrics["arm_count"] < gate["min_arm_count"]:
        return False
    return True


def _iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_missing_reason_summary(out_csv: Path, expected_drives: List[str], report_type: str, run_id: str) -> dict:
    from collections import Counter

    rows = []
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    counts = Counter()
    non_ok = []
    for row in rows:
        reason = (row.get("missing_reason") or "").strip()
        norm = "" if reason in {"", "N/A", "OK"} else reason
        counts[norm or "OK"] += 1
        if norm:
            non_ok.append({"drive_id": row.get("drive_id"), "reason": norm})
    payload = {
        "expected_drives": expected_drives,
        "missing_reason_counts": dict(counts),
        "non_ok_drives": non_ok,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "report_type": report_type,
    }
    json_path = out_csv.with_name(out_csv.stem + "_missing_reason_summary.json")
    md_path = out_csv.with_name(out_csv.stem + "_missing_reason_summary.md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Missing Reason Summary",
        "",
        f"- report_type: {report_type}",
        f"- run_id: {run_id}",
        f"- generated_at: {payload['generated_at']}",
        "",
        "## expected_drives",
        "```json",
        json.dumps(expected_drives, ensure_ascii=False, indent=2),
        "```",
        "",
        "## missing_reason_counts",
        "```json",
        json.dumps(payload["missing_reason_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## non_ok_drives",
        "```json",
        json.dumps(non_ok, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="postopt_index.jsonl")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--config", default="configs/intersections_v2.yaml")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    run_cfg = cfg.get("run", {}) or {}
    seeds_cfg = cfg.get("seeds", {}) or {}
    refine_cfg = cfg.get("refine", {}) or {}
    shape_cfg = cfg.get("shape", {}) or {}
    gate_cfg = cfg.get("gate", {}) or {}
    debug_cfg = cfg.get("debug", {}) or {}

    expected_drives = []
    expected_path = Path(run_cfg.get("expected_drives_file", "configs/golden_drives.txt"))
    if expected_path.exists():
        expected_drives = [ln.strip() for ln in expected_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    entries = _read_index(Path(args.index))
    entries = [
        e for e in entries
        if e.get("stage") == args.stage
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]
    if args.candidate:
        entries = [e for e in entries if e.get("candidate_id") == args.candidate]
    if not entries:
        raise SystemExit("ERROR: no entries found")

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"intersections_v2_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_outputs = out_dir / "outputs"
    out_outputs.mkdir(parents=True, exist_ok=True)

    road_polys_by_drive: Dict[str, Polygon] = {}
    for entry in entries:
        drive = str(entry.get("drive") or "")
        outputs_dir = Path(entry.get("outputs_dir") or "")
        if not drive or not outputs_dir.exists():
            continue
        road_path = outputs_dir / "road_polygon.geojson"
        if not road_path.exists():
            continue
        road_feats = _read_geojson(road_path)
        road_polys = _collect_polys(road_feats)
        if road_polys:
            road_polys_by_drive[drive] = unary_union(road_polys)

    sat_features_by_drive: Dict[str, List[dict]] = {}
    sat_outputs_dir = Path(seeds_cfg.get("sat_outputs_dir", ""))
    if seeds_cfg.get("sat_enabled", True) and seeds_cfg.get("sat_seed_input_mode") == "polygons" and sat_outputs_dir.exists():
        sat_path = sat_outputs_dir / "intersections_sat.geojson"
        if not sat_path.exists():
            sat_path = sat_outputs_dir / "intersections_sat_wgs84.geojson"
        if sat_path.exists():
            sat_feats = _read_geojson(sat_path)
            missing_drive = sum(1 for f in sat_feats if not (f.get("properties") or {}).get("drive_id"))
            if missing_drive:
                sat_features_by_drive, stats, unmatched = _assign_sat_features_to_drives(sat_feats, entries, road_polys_by_drive)
                print(
                    f"[V2] SAT assign: total={stats['sat_total']} matched={stats['sat_matched']} "
                    f"unmatched={stats['sat_unmatched']}"
                )
                if unmatched:
                    sat_unassigned = out_outputs / "sat_unassigned.csv"
                    rows = []
                    for feat in unmatched:
                        props = feat.get("properties") or {}
                        rows.append(
                            {
                                "sat_confidence": props.get("sat_confidence"),
                                "drive_id": props.get("drive_id"),
                                "tile_id": props.get("tile_id"),
                            }
                        )
                    _write_csv(sat_unassigned, rows)
            else:
                for feat in sat_feats:
                    drive_id = str((feat.get("properties") or {}).get("drive_id") or "")
                    if not drive_id:
                        continue
                    sat_features_by_drive.setdefault(drive_id, []).append(feat)

    osm_path_cfg = str(seeds_cfg.get("osm_roads_path") or "")
    osm_global_path = Path(osm_path_cfg) if osm_path_cfg else None
    if osm_global_path and not osm_global_path.exists():
        print(f"[V2][WARN] configured osm_roads_path not found: {osm_global_path}")
        osm_global_path = None
    if osm_global_path is None:
        osm_global_path = _find_latest_osm_path()
        if osm_global_path:
            print(f"[V2] using OSM roads from: {osm_global_path}")

    rows = []
    seen_drives = set()
    for entry in entries:
        drive = str(entry.get("drive") or "")
        outputs_dir = Path(entry.get("outputs_dir"))
        if not drive or not outputs_dir.exists():
            continue
        seen_drives.add(drive)
        drive_dir = out_outputs / drive
        drive_dir.mkdir(parents=True, exist_ok=True)
        final_path = drive_dir / "intersections_final.geojson"
        if args.resume and final_path.exists():
            continue

        road_path = outputs_dir / "road_polygon.geojson"
        center_path = outputs_dir / "centerlines_both.geojson"
        if not center_path.exists():
            center_path = outputs_dir / "centerlines.geojson"
        if not road_path.exists() or not center_path.exists():
            rows.append(
                {
                    "drive_id": drive,
                    "status": "FAIL",
                    "missing_reason": "missing_inputs",
                    "final_cnt": 0,
                }
            )
            continue

        road_feats = _read_geojson(road_path)
        road_polys = _collect_polys(road_feats)
        road_poly = unary_union(road_polys) if road_polys else Polygon()

        center_feats = _read_geojson(center_path)
        center_lines = _collect_lines(center_feats)

        missing_reasons = []
        seed_features = []
        refined_seed_features = []
        debug_local = []
        debug_arms = []
        debug_refined = []

        seeds: List[dict] = []
        seed_counts = {"traj": 0, "osm": 0, "sat": 0, "geom": 0}
        if seeds_cfg.get("traj_enabled", True):
            traj_path_tmpl = str(seeds_cfg.get("traj_points_path_template") or "")
            traj_path = Path(traj_path_tmpl.format(drive=drive)) if traj_path_tmpl else Path()
            traj_seeds = []
            if traj_path_tmpl and traj_path.exists():
                traj_feats = _read_geojson(traj_path)
                traj_lines = _collect_lines(traj_feats)
                angle_deg = float(seeds_cfg.get("traj_turn_angle_deg", 35.0))
                min_sep = float(seeds_cfg.get("traj_min_sep_m", 25.0))
                for line in traj_lines:
                    traj_seeds.extend(_traj_seeds_from_line(line, angle_deg, min_sep))
            else:
                missing_reasons.append("missing_traj_inputs")
            for pt in traj_seeds:
                seeds.append(
                    {
                        "seed": pt,
                        "src_seed": "traj",
                        "reason": "traj_split",
                        "conf_prior": 0.8,
                        "radius_m": float(seeds_cfg.get("seed_radius_default_m", 18.0)),
                    }
                )
                seed_counts["traj"] += 1

        if seeds_cfg.get("osm_enabled", True):
            osm_path = osm_global_path or (outputs_dir / "drivable_roads.geojson")
            if osm_path is not None and osm_path.exists():
                osm_feats = _read_geojson(osm_path)
                allowlist = seeds_cfg.get("osm_highway_allowlist") or []
                osm_feats = _filter_osm_features(osm_feats, allowlist)
                osm_lines = _collect_lines(osm_feats)
                snap_m = float(seeds_cfg.get("osm_snap_m", 2.0))
                min_degree = int(seeds_cfg.get("osm_min_degree", 3))
                osm_seeds = _osm_degree_seeds(osm_lines, snap_m, min_degree)
                if not osm_seeds and min_degree > 2:
                    osm_seeds = _osm_degree_seeds(osm_lines, snap_m, 2)
                    if osm_seeds:
                        print(f"[V2][WARN] no osm degree>={min_degree} seeds for {drive}, fallback to degree>=2")
                        min_degree = 2
                if not osm_seeds:
                    junctions = _centerline_junctions(osm_lines)
                    if junctions:
                        print(f"[V2][WARN] no osm degree seeds for {drive}, fallback to osm intersections")
                        for pt in junctions:
                            seeds.append(
                                {
                                    "seed": pt,
                                    "src_seed": "osm",
                                    "reason": "osm_intersection_fallback",
                                    "conf_prior": 0.5,
                                    "radius_m": float(seeds_cfg.get("seed_radius_default_m", 18.0)),
                                }
                            )
                            seed_counts["osm"] += 1
                for pt, degree in osm_seeds:
                    conf = min(1.0, 0.4 + 0.1 * float(degree))
                    seeds.append(
                        {
                            "seed": pt,
                            "src_seed": "osm",
                            "reason": f"osm_degree{min_degree}",
                            "conf_prior": conf,
                            "radius_m": float(seeds_cfg.get("seed_radius_default_m", 18.0)),
                        }
                    )
                    seed_counts["osm"] += 1
            else:
                missing_reasons.append("missing_osm_inputs")

        if seeds_cfg.get("geom_enabled", True):
            geom_inter_path = outputs_dir / "intersections_algo.geojson"
            geom_feats = _read_geojson(geom_inter_path)
            for feat in geom_feats:
                poly = shape(feat.get("geometry")) if feat.get("geometry") else None
                if poly is None or poly.is_empty:
                    continue
                radius = math.sqrt(poly.area / math.pi) if poly.area > 0 else float(seeds_cfg.get("seed_radius_default_m", 18.0))
                seeds.append(
                    {
                        "seed": poly.centroid,
                        "src_seed": "geom",
                        "reason": "geom_junction",
                        "conf_prior": 0.6,
                        "radius_m": float(radius),
                    }
                )
                seed_counts["geom"] += 1

        if seeds_cfg.get("sat_enabled", True) and seeds_cfg.get("sat_seed_input_mode") == "polygons":
            sat_feats = sat_features_by_drive.get(drive, [])
            for feat in sat_feats:
                poly = shape(feat.get("geometry")) if feat.get("geometry") else None
                if poly is None or poly.is_empty:
                    continue
                props = feat.get("properties") or {}
                conf = props.get("sat_confidence") or props.get("conf")
                radius = math.sqrt(poly.area / math.pi) if poly.area > 0 else float(seeds_cfg.get("seed_radius_default_m", 18.0))
                seeds.append(
                    {
                        "seed": poly.centroid,
                        "src_seed": "sat",
                        "reason": "sat_polygon",
                        "conf_prior": float(conf) if isinstance(conf, (int, float)) else 0.5,
                        "radius_m": float(radius),
                    }
                )
                seed_counts["sat"] += 1

        refined_seeds = []
        for item in seeds:
            seed = item["seed"]
            ref_seed = seed
            refine_src = "none"
            conf_refine = None
            if refine_cfg.get("enabled", True):
                ref_seed, refine_src, conf_refine = _refine_seed(
                    seed,
                    center_lines,
                    float(refine_cfg.get("search_radius_m", 25.0)),
                )
            refined_seeds.append({**item, "seed": ref_seed, "refine_src": refine_src, "conf_refine": conf_refine})

            seed_features.append(
                {"type": "Feature", "geometry": mapping(seed), "properties": {"src_seed": item["src_seed"], "reason": item["reason"]}}
            )
            refined_seed_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(ref_seed),
                    "properties": {
                        "src_seed": item["src_seed"],
                        "reason": item["reason"],
                        "refine_src": refine_src,
                        "conf_refine": conf_refine,
                    },
                }
            )

        algo_features = []
        sat_features = []
        final_candidates = []
        gate_fail_counts = {"sat": 0, "other": 0}
        no_arms_count = 0

        for item in refined_seeds:
            src_seed = item["src_seed"]
            seed = item["seed"]
            radius_m = float(item["radius_m"])
            shape_cfg_use = dict(shape_cfg)
            shape_cfg_use["radius_m"] = radius_m
            poly, metrics = _shape_from_seed(seed, road_poly, center_lines, shape_cfg_use)
            if poly is None or poly.is_empty:
                continue
            area = float(poly.area)
            gate_ok = _shape_gate_ok(metrics, gate_cfg, area)
            if not gate_ok and gate_cfg.get("retry_shrink_ratio", 0.0):
                shrink_cfg = dict(shape_cfg_use)
                shrink_cfg["radius_m"] = float(shape_cfg_use["radius_m"]) * float(gate_cfg["retry_shrink_ratio"])
                poly, metrics = _shape_from_seed(seed, road_poly, center_lines, shrink_cfg)
                area = float(poly.area) if poly is not None else 0.0
                gate_ok = _shape_gate_ok(metrics, gate_cfg, area) if poly is not None else False
            if not gate_ok and src_seed == "sat" and gate_cfg.get("sat_drop_on_fail", True):
                gate_fail_counts["sat"] += 1
                continue
            if not gate_ok:
                gate_fail_counts["other"] += 1
                if metrics.get("local") is not None:
                    poly = metrics["local"]
            if metrics.get("has_arms") == 0:
                no_arms_count += 1

            props = {
                "drive_id": drive,
                "tile_id": drive,
                "src_seed": src_seed,
                "refine_src": item.get("refine_src") or "none",
                "reason": "no_arms" if metrics.get("has_arms") == 0 else item.get("reason"),
                "conf_prior": item.get("conf_prior"),
                "conf_refine": item.get("conf_refine"),
                "arm_count": int(metrics.get("arm_count", 0)),
                "overlap_road": round(float(metrics.get("overlap_road", 0.0)), 4),
                "circularity": round(float(metrics.get("circularity", 0.0)), 4),
                "aspect_ratio": round(float(metrics.get("aspect_ratio", 0.0)), 4),
                "shape_gate_pass": 1 if gate_ok else 0,
                "has_arms": int(metrics.get("has_arms", 0)),
                "arms_area": round(float(metrics.get("arms_area", 0.0)), 3),
                "local_area": round(float(metrics.get("local_area", 0.0)), 3),
                "refined_area": round(float(metrics.get("refined_area", 0.0)), 3),
            }
            feat = {"type": "Feature", "geometry": mapping(poly), "properties": props}
            if src_seed == "sat":
                sat_features.append(feat)
            else:
                algo_features.append(feat)
            final_candidates.append((poly, props))

            if debug_cfg.get("enable_debug_layers", True):
                if metrics.get("local") is not None:
                    debug_local.append({"type": "Feature", "geometry": mapping(metrics["local"]), "properties": {"src_seed": src_seed}})
                if metrics.get("arms") is not None and not metrics["arms"].is_empty:
                    debug_arms.append({"type": "Feature", "geometry": mapping(metrics["arms"]), "properties": {"src_seed": src_seed}})
                debug_refined.append({"type": "Feature", "geometry": mapping(poly), "properties": {"src_seed": src_seed}})

        def _priority(src: str) -> int:
            return {"traj": 4, "osm": 3, "sat": 2, "geom": 1}.get(src, 0)

        final_features = []
        for poly, props in final_candidates:
            kept_idx = None
            for idx, kept in enumerate(final_features):
                if _iou(poly, shape(kept["geometry"])) >= gate_cfg.get("dup_iou", 0.25):
                    kept_idx = idx
                    break
            if kept_idx is None:
                final_features.append({"type": "Feature", "geometry": mapping(poly), "properties": props})
                continue
            kept_props = final_features[kept_idx]["properties"]
            if _priority(props["src_seed"]) > _priority(kept_props.get("src_seed")):
                final_features[kept_idx] = {"type": "Feature", "geometry": mapping(poly), "properties": props}
                continue
            if _priority(props["src_seed"]) == _priority(kept_props.get("src_seed")):
                score_new = props["overlap_road"] + 0.1 * props["arm_count"] - 0.5 * props["circularity"]
                score_old = kept_props.get("overlap_road", 0.0) + 0.1 * kept_props.get("arm_count", 0) - 0.5 * kept_props.get("circularity", 0.0)
                if score_new > score_old:
                    final_features[kept_idx] = {"type": "Feature", "geometry": mapping(poly), "properties": props}

        _write_geojson(drive_dir / "intersections_seeds.geojson", seed_features)
        _write_geojson(drive_dir / "intersections_seeds_refined.geojson", refined_seed_features)
        _write_geojson(drive_dir / "intersections_shape_debug_local.geojson", debug_local)
        _write_geojson(drive_dir / "intersections_shape_debug_arms.geojson", debug_arms)
        _write_geojson(drive_dir / "intersections_shape_debug_refined.geojson", debug_refined)

        _write_geojson(drive_dir / "intersections_algo.geojson", algo_features)
        _write_geojson(drive_dir / "intersections_sat.geojson", sat_features)
        _write_geojson(drive_dir / "intersections_final.geojson", final_features)
        _write_geojson(drive_dir / "intersections_algo_wgs84.geojson", _to_wgs84(algo_features, 32632))
        _write_geojson(drive_dir / "intersections_sat_wgs84.geojson", _to_wgs84(sat_features, 32632))
        _write_geojson(drive_dir / "intersections_final_wgs84.geojson", _to_wgs84(final_features, 32632))

        sat_circ = [f["properties"]["circularity"] for f in sat_features if f.get("properties")]
        sat_overlap = [f["properties"]["overlap_road"] for f in sat_features if f.get("properties")]
        sat_circ_sorted = sorted(sat_circ)
        sat_overlap_sorted = sorted(sat_overlap)
        p50 = sat_circ_sorted[len(sat_circ_sorted) // 2] if sat_circ_sorted else None
        p75 = sat_circ_sorted[int(len(sat_circ_sorted) * 0.75)] if sat_circ_sorted else None
        o50 = sat_overlap_sorted[len(sat_overlap_sorted) // 2] if sat_overlap_sorted else None
        src_counts = {
            "traj": sum(1 for f in final_features if f.get("properties", {}).get("src_seed") == "traj"),
            "osm": sum(1 for f in final_features if f.get("properties", {}).get("src_seed") == "osm"),
            "sat": sum(1 for f in final_features if f.get("properties", {}).get("src_seed") == "sat"),
            "geom": sum(1 for f in final_features if f.get("properties", {}).get("src_seed") == "geom"),
        }
        rows.append(
            {
                "drive_id": drive,
                "status": "OK",
                "missing_reason": "OK" if not missing_reasons else ",".join(sorted(set(missing_reasons))),
                "final_cnt": len(final_features),
                "traj_cnt": src_counts["traj"],
                "osm_cnt": src_counts["osm"],
                "sat_cnt": src_counts["sat"],
                "geom_cnt": src_counts["geom"],
                "sat_gate_drop": gate_fail_counts["sat"],
                "other_gate_fail": gate_fail_counts["other"],
                "no_arms": no_arms_count,
                "seed_traj_cnt": seed_counts["traj"],
                "seed_osm_cnt": seed_counts["osm"],
                "seed_sat_cnt": seed_counts["sat"],
                "seed_geom_cnt": seed_counts["geom"],
                "sat_circularity_p50": p50,
                "sat_circularity_p75": p75,
                "sat_overlap_p50": o50,
            }
        )

    for drive in expected_drives:
        if drive in seen_drives:
            continue
        rows.append(
            {
                "drive_id": drive,
                "status": "FAIL",
                "missing_reason": "missing_entry",
                "final_cnt": 0,
            }
        )

    report_csv = out_dir / f"{args.stage}_report_per_drive.csv"
    report_json = out_dir / f"{args.stage}_report_per_drive.json"
    _write_csv(report_csv, rows)
    report_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_missing_reason_summary(report_csv, expected_drives, args.stage, out_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
