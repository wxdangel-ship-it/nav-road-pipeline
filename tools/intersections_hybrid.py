from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform as geom_transform
from pyproj import Transformer
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from pipeline.intersection_shape import (
        aspect_ratio as _intersection_aspect_ratio,
        arm_count as _intersection_arm_count,
        circularity as _intersection_circularity,
        overlap_with_road as _intersection_overlap_with_road,
        refine_intersection_polygon as _refine_intersection_polygon,
    )
except Exception as exc:
    raise SystemExit(f"ERROR: failed to import intersection shape helpers: {exc}") from exc


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_index(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def _read_drives(path: Path) -> List[str]:
    if not path.exists():
        return []
    drives = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        drives.append(line)
    return drives


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


def _compactness(area: float, perim: float) -> float:
    if area <= 0 or perim <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))


def _load_shape_refine_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("shape_refine") or data


def _shape_refine_defaults() -> dict:
    return {
        "radius_m": 20.0,
        "road_buffer_m": 1.0,
        "arm_length_m": 25.0,
        "arm_buffer_m": 7.0,
        "simplify_m": 0.5,
        "min_area_m2": 30.0,
        "min_part_area_m2": 30.0,
        "min_hole_area_m2": 30.0,
        "max_circularity": 0.9,
        "min_overlap_road": 0.7,
    }


def _aspect_ratio(poly) -> float:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0
    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        edges.append(math.hypot(x2 - x1, y2 - y1))
    if not edges:
        return 0.0
    edges = sorted(edges)
    if edges[0] <= 0:
        return 0.0
    return float(edges[-1] / edges[0])


def _poly_metrics(poly, road_poly, centerline_union) -> dict:
    if poly.is_empty:
        return {"empty": True}
    area = float(poly.area)
    perim = float(poly.length)
    compact = _compactness(area, perim)
    aspect = _aspect_ratio(poly)
    overlap = 0.0
    if road_poly is not None and not road_poly.is_empty:
        overlap = float(poly.intersection(road_poly).area) / max(1e-6, area)
    center_dist = float(poly.centroid.distance(centerline_union)) if centerline_union is not None else 0.0
    return {
        "empty": False,
        "area_m2": round(area, 3),
        "compactness": round(compact, 4),
        "aspect_ratio": round(aspect, 4),
        "overlap_ratio": round(overlap, 4),
        "centerline_dist_m": round(center_dist, 3),
    }


def _validate_poly(poly, road_poly, centerline_union, cfg: dict) -> Tuple[bool, dict, dict]:
    metrics = _poly_metrics(poly, road_poly, centerline_union)
    if metrics.get("empty"):
        return False, {"reason": "empty"}, {"empty": True}
    flags = {
        "area": False,
        "compactness": False,
        "aspect": False,
        "overlap": False,
        "dist": False,
    }
    if metrics["area_m2"] < cfg["min_area_m2"] or metrics["area_m2"] > cfg["max_area_m2"]:
        flags["area"] = True
    if metrics["compactness"] < cfg["min_compactness"]:
        flags["compactness"] = True
    if metrics["aspect_ratio"] > cfg["max_aspect_ratio"]:
        flags["aspect"] = True
    if metrics["overlap_ratio"] < cfg["min_overlap_ratio"]:
        flags["overlap"] = True
    if metrics["centerline_dist_m"] > cfg["max_centerline_dist_m"]:
        flags["dist"] = True
    ok = not any(flags.values())
    return ok, {k: v for k, v in metrics.items() if k != "empty"}, flags


def _validate_sat_custom(
    poly,
    road_poly,
    centerline_union,
    cfg: dict,
    *,
    min_overlap_ratio: float,
    max_centerline_dist_m: float,
) -> Tuple[bool, dict, dict]:
    metrics = _poly_metrics(poly, road_poly, centerline_union)
    if metrics.get("empty"):
        return False, {"reason": "empty"}, {"empty": True}
    flags = {
        "area": False,
        "compactness": False,
        "aspect": False,
        "overlap": False,
        "dist": False,
    }
    if metrics["area_m2"] < cfg["min_area_m2"] or metrics["area_m2"] > cfg["max_area_m2"]:
        flags["area"] = True
    if metrics["compactness"] < cfg["min_compactness"]:
        flags["compactness"] = True
    if metrics["aspect_ratio"] > cfg["max_aspect_ratio"]:
        flags["aspect"] = True
    if metrics["overlap_ratio"] < min_overlap_ratio:
        flags["overlap"] = True
    if metrics["centerline_dist_m"] > max_centerline_dist_m:
        flags["dist"] = True
    ok = not any(flags.values())
    return ok, {k: v for k, v in metrics.items() if k != "empty"}, flags


def _iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def _max_iou(poly, polys: List) -> float:
    best = 0.0
    for other in polys:
        best = max(best, _iou(poly, other))
    return best


def _coerce_conf(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shape_metrics(poly, road_poly, centerlines, cfg: dict, refined: bool) -> dict:
    if poly is None or poly.is_empty:
        return {"shape_refined": 0 if not refined else 1}
    circ = _intersection_circularity(poly)
    aspect = _intersection_aspect_ratio(poly)
    overlap = _intersection_overlap_with_road(poly, road_poly) if road_poly is not None else 0.0
    arms = _intersection_arm_count(poly, centerlines, float(cfg["arm_buffer_m"]))
    props = {
        "shape_refined": 1 if refined else 0,
        "circularity": round(circ, 4),
        "aspect_ratio": round(aspect, 4),
        "overlap_road": round(overlap, 4),
        "arm_count": int(arms),
    }
    return props


def _refine_choice(poly, road_poly, centerlines, cfg: dict) -> Tuple[object, dict, dict]:
    if poly is None or poly.is_empty:
        return poly, {"shape_refined": 0}, {}
    seed = poly.centroid
    pre_circ = _intersection_circularity(poly)
    refined, meta = _refine_intersection_polygon(
        seed_pt=seed,
        poly_candidate=poly,
        road_polygon=road_poly,
        centerlines=centerlines,
        cfg=cfg,
    )
    debug = {"seed": seed, "local": meta.get("local"), "arms": meta.get("arms"), "refined": refined}
    if refined is None or refined.is_empty:
        refined = poly
    post_circ = _intersection_circularity(refined)
    arms = _intersection_arm_count(refined, centerlines, float(cfg["arm_buffer_m"]))
    max_circ = float(cfg.get("max_circularity", 0.9))
    if post_circ > max_circ and arms >= 3:
        shrink_cfg = dict(cfg)
        shrink_cfg["radius_m"] = float(cfg["radius_m"]) * 0.7
        refined2, meta2 = _refine_intersection_polygon(
            seed_pt=seed,
            poly_candidate=poly,
            road_polygon=road_poly,
            centerlines=centerlines,
            cfg=shrink_cfg,
        )
        if refined2 is not None and not refined2.is_empty:
            refined = refined2
            post_circ = _intersection_circularity(refined)
            arms = _intersection_arm_count(refined, centerlines, float(cfg["arm_buffer_m"]))
            debug = {"seed": seed, "local": meta2.get("local"), "arms": meta2.get("arms"), "refined": refined}
    props = _shape_metrics(refined, road_poly, centerlines, cfg, refined=True)
    props["pre_circularity"] = round(pre_circ, 4)
    props["post_circularity"] = round(post_circ, 4)
    return refined, props, debug


def _shape_gate_ok(props: dict, cfg: dict) -> bool:
    max_circ = float(cfg.get("max_circularity", 0.9))
    min_overlap = float(cfg.get("min_overlap_road", 0.7))
    return props.get("circularity", 0.0) <= max_circ and props.get("overlap_road", 0.0) >= min_overlap


def _shape_gate_reason(props: dict, cfg: dict) -> Optional[str]:
    max_circ = float(cfg.get("max_circularity", 0.9))
    min_overlap = float(cfg.get("min_overlap_road", 0.7))
    if props.get("circularity", 0.0) > max_circ:
        return "circularity"
    if props.get("overlap_road", 0.0) < min_overlap:
        return "overlap"
    return None


def _write_csv(path: Path, rows: List[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("N/A" if row.get(k) is None else row.get(k, "")) for k in fieldnames})


def _write_missing_reason_summary(out_csv: Path, expected_drives: List[str], report_type: str, run_id: str) -> dict:
    import csv
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
    ap.add_argument("--index", required=True, help="postopt_index.jsonl or similar")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--config", default="configs/intersections_hybrid.yaml")
    ap.add_argument("--out-dir", default="", help="report output dir")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config)).get("hybrid", {}) or {}
    expected_drives = _read_drives(Path(cfg.get("expected_drives_file", "configs/golden_drives.txt")))
    sat_mode = cfg.get("sat_mode", "fallback_only")
    dup_iou = float(cfg.get("dup_iou", cfg.get("match_iou_min", 0.2)))
    sat_augment_min_conf = float(cfg.get("sat_augment_min_conf", cfg.get("sat_conf_min", 0.0)))
    sat_augment_min_overlap = float(cfg.get("sat_augment_min_overlap", cfg.get("min_overlap_ratio", 0.0)))
    sat_augment_max_dist = float(cfg.get("sat_augment_max_dist", cfg.get("max_centerline_dist_m", 0.0)))
    debug_reject = bool(cfg.get("debug_reject_breakdown", False))
    shape_refine_cfg = _shape_refine_defaults()
    refine_path = Path(cfg.get("shape_refine_config", "configs/intersections_shape_refine.yaml"))
    shape_refine_cfg.update(_load_shape_refine_cfg(refine_path))
    shape_refine_enabled = bool(cfg.get("shape_refine_enabled", True))

    entries = _read_index(Path(args.index))
    entries = [
        e for e in entries
        if e.get("stage") == args.stage
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]
    if args.candidate:
        entries = [e for e in entries if e.get("candidate_id") == args.candidate]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(entries[0]["outputs_dir"]).parents[1] if entries else Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_drive_rows = []
    seen_drives = set()

    for entry in entries:
        drive = str(entry.get("drive") or entry.get("tile_id") or "")
        out_dir = Path(entry.get("outputs_dir"))
        if not drive or not out_dir.exists():
            continue
        seen_drives.add(drive)

        final_path = out_dir / "intersections_final.geojson"
        if args.resume and final_path.exists():
            continue

        algo_path = out_dir / "intersections_algo.geojson"
        if not algo_path.exists():
            algo_path = out_dir / "intersections.geojson"
        sat_path = out_dir / "intersections_sat.geojson"
        road_path = out_dir / "road_polygon.geojson"
        center_path = out_dir / "centerlines.geojson"

        if not algo_path.exists() or not road_path.exists() or not center_path.exists():
            per_drive_rows.append(
                {
                    "drive_id": drive,
                    "tile_id": drive,
                    "status": "FAIL",
                    "missing_reason": "missing_inputs",
                    "final_count": 0,
                    "sat_reject_conf": 0,
                    "sat_reject_overlap": 0,
                    "sat_reject_dist": 0,
                    "sat_reject_shape": 0,
                    "sat_dedup": 0,
                }
            )
            continue

        algo_feats = _read_geojson(algo_path)
        sat_feats = _read_geojson(sat_path) if sat_path.exists() else []
        road_feats = _read_geojson(road_path)
        center_feats = _read_geojson(center_path)

        road_poly = None
        if road_feats:
            road_poly = unary_union([shape(f.get("geometry")) for f in road_feats])
        centerlines = [shape(f.get("geometry")) for f in center_feats if f.get("geometry")]
        center_union = unary_union(centerlines) if centerlines else None

        algo_polys = [(shape(f.get("geometry")), f.get("properties") or {}) for f in algo_feats]
        algo_geoms = [geom for geom, _ in algo_polys]
        sat_polys = [(shape(f.get("geometry")), f.get("properties") or {}) for f in sat_feats]

        final_features = []
        algo_any_valid = False
        reject_counts = {
            "sat_reject_conf": 0,
            "sat_reject_overlap": 0,
            "sat_reject_dist": 0,
            "sat_reject_shape": 0,
            "sat_dedup": 0,
        }
        debug_local = []
        debug_arms = []
        debug_refined = []

        for algo_geom, algo_props in algo_polys:
            if algo_geom.is_empty:
                continue
            best_iou = 0.0
            best_idx = -1
            for i, (sat_geom, _) in enumerate(sat_polys):
                score = _iou(algo_geom, sat_geom)
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            algo_ok, algo_qc, algo_flags = _validate_poly(algo_geom, road_poly, center_union, cfg)
            if algo_ok:
                algo_any_valid = True
            sat_geom = None
            sat_props = None
            sat_ok = False
            if best_idx >= 0 and best_iou >= cfg["match_iou_min"]:
                sat_geom, sat_props = sat_polys[best_idx]
                sat_ok, _, sat_flags = _validate_poly(sat_geom, road_poly, center_union, cfg)
                sat_conf = sat_props.get("sat_confidence")
                if isinstance(sat_conf, (int, float)) and sat_conf < cfg.get("sat_conf_min", 0.0):
                    sat_ok = False

            chosen = None
            reason = "invalid_filtered"
            src = "none"
            conf = None

            if algo_ok and cfg.get("prefer_algo", True):
                chosen = algo_geom
                reason = "algo_keep"
                src = "algo"
            elif sat_ok and sat_mode in {"fallback_only", "augment_unmatched"}:
                chosen = sat_geom
                reason = "sat_fallback"
                src = "sat"
            elif algo_ok:
                chosen = algo_geom
                reason = "algo_keep"
                src = "algo"
            elif sat_ok:
                chosen = sat_geom
                reason = "sat_fallback"
                src = "sat"
            elif cfg.get("union_enabled", False) and best_idx >= 0 and best_iou >= cfg["union_iou_min"]:
                union_geom = algo_geom.union(sat_geom)
                union_ok, _ = _validate_poly(union_geom, road_poly, center_union, cfg)
                if union_ok:
                    chosen = union_geom
                    reason = "union_merge"
                    src = "union"

            if chosen is not None:
                if src in {"sat", "union"} and sat_props:
                    conf = _coerce_conf(sat_props.get("sat_confidence"))
                debug = {}
                if shape_refine_enabled:
                    chosen, refine_props, debug = _refine_choice(chosen, road_poly, centerlines, shape_refine_cfg)
                    if src == "sat":
                        gate_reason = _shape_gate_reason(refine_props, shape_refine_cfg)
                        if gate_reason is not None:
                            if gate_reason == "overlap":
                                reject_counts["sat_reject_overlap"] += 1
                            else:
                                reject_counts["sat_reject_shape"] += 1
                            if debug.get("refined") is not None:
                                debug_refined.append(
                                    {"type": "Feature", "geometry": mapping(debug["refined"]), "properties": {"src": src, "reason": "shape_gate"}}
                                )
                            continue
                else:
                    refine_props = _shape_metrics(chosen, road_poly, centerlines, shape_refine_cfg, refined=False)
                if debug.get("local") is not None:
                    debug_local.append({"type": "Feature", "geometry": mapping(debug["local"]), "properties": {"src": src}})
                if debug.get("arms") is not None and not debug["arms"].is_empty:
                    debug_arms.append({"type": "Feature", "geometry": mapping(debug["arms"]), "properties": {"src": src}})
                if debug.get("refined") is not None:
                    debug_refined.append({"type": "Feature", "geometry": mapping(debug["refined"]), "properties": {"src": src}})
                props = {
                    "drive_id": drive,
                    "tile_id": drive,
                    "src": src,
                    "reason": reason,
                    "conf": conf,
                }
                props.update(algo_qc)
                props.update(refine_props)
                final_features.append({"type": "Feature", "geometry": mapping(chosen), "properties": props})

        for i, (sat_geom, sat_props) in enumerate(sat_polys):
            max_iou = _max_iou(sat_geom, algo_geoms) if algo_geoms else 0.0
            if max_iou >= dup_iou:
                reject_counts["sat_dedup"] += 1
                continue
            if sat_mode == "never":
                continue
            sat_conf = sat_props.get("sat_confidence")
            sat_conf_val = float(sat_conf) if isinstance(sat_conf, (int, float)) else 0.0

            if sat_mode == "augment_unmatched" and algo_any_valid:
                sat_ok, sat_qc, sat_flags = _validate_sat_custom(
                    sat_geom,
                    road_poly,
                    center_union,
                    cfg,
                    min_overlap_ratio=sat_augment_min_overlap,
                    max_centerline_dist_m=sat_augment_max_dist,
                )
                if sat_conf_val < sat_augment_min_conf:
                    sat_ok = False
                    reject_counts["sat_reject_conf"] += 1
                if sat_flags.get("overlap"):
                    reject_counts["sat_reject_overlap"] += 1
                if sat_flags.get("dist"):
                    reject_counts["sat_reject_dist"] += 1
                if sat_flags.get("area") or sat_flags.get("compactness") or sat_flags.get("aspect"):
                    reject_counts["sat_reject_shape"] += 1
                if not sat_ok:
                    continue
                chosen = sat_geom
                debug = {}
                if shape_refine_enabled:
                    chosen, refine_props, debug = _refine_choice(sat_geom, road_poly, centerlines, shape_refine_cfg)
                    gate_reason = _shape_gate_reason(refine_props, shape_refine_cfg)
                    if gate_reason is not None:
                        if gate_reason == "overlap":
                            reject_counts["sat_reject_overlap"] += 1
                        else:
                            reject_counts["sat_reject_shape"] += 1
                        if debug.get("refined") is not None:
                            debug_refined.append(
                                {"type": "Feature", "geometry": mapping(debug["refined"]), "properties": {"src": "sat", "reason": "shape_gate"}}
                            )
                        continue
                else:
                    refine_props = _shape_metrics(chosen, road_poly, centerlines, shape_refine_cfg, refined=False)
                if debug.get("local") is not None:
                    debug_local.append({"type": "Feature", "geometry": mapping(debug["local"]), "properties": {"src": "sat"}})
                if debug.get("arms") is not None and not debug["arms"].is_empty:
                    debug_arms.append({"type": "Feature", "geometry": mapping(debug["arms"]), "properties": {"src": "sat"}})
                if debug.get("refined") is not None:
                    debug_refined.append({"type": "Feature", "geometry": mapping(debug["refined"]), "properties": {"src": "sat"}})
                props = {
                    "drive_id": drive,
                    "tile_id": drive,
                    "src": "sat",
                    "reason": "sat_augment",
                    "conf": _coerce_conf(sat_conf),
                }
                props.update(sat_qc)
                props.update(refine_props)
                final_features.append({"type": "Feature", "geometry": mapping(chosen), "properties": props})
                continue

            if sat_mode == "fallback_only" and algo_any_valid:
                continue

            sat_ok, sat_qc, sat_flags = _validate_poly(sat_geom, road_poly, center_union, cfg)
            if sat_conf_val < cfg.get("sat_conf_min", 0.0):
                sat_ok = False
                reject_counts["sat_reject_conf"] += 1
            if sat_flags.get("overlap"):
                reject_counts["sat_reject_overlap"] += 1
            if sat_flags.get("dist"):
                reject_counts["sat_reject_dist"] += 1
            if sat_flags.get("area") or sat_flags.get("compactness") or sat_flags.get("aspect"):
                reject_counts["sat_reject_shape"] += 1
            if not sat_ok:
                continue
            chosen = sat_geom
            debug = {}
            if shape_refine_enabled:
                chosen, refine_props, debug = _refine_choice(sat_geom, road_poly, centerlines, shape_refine_cfg)
                gate_reason = _shape_gate_reason(refine_props, shape_refine_cfg)
                if gate_reason is not None:
                    if gate_reason == "overlap":
                        reject_counts["sat_reject_overlap"] += 1
                    else:
                        reject_counts["sat_reject_shape"] += 1
                    if debug.get("refined") is not None:
                        debug_refined.append(
                            {"type": "Feature", "geometry": mapping(debug["refined"]), "properties": {"src": "sat", "reason": "shape_gate"}}
                        )
                    continue
            else:
                refine_props = _shape_metrics(chosen, road_poly, centerlines, shape_refine_cfg, refined=False)
            if debug.get("local") is not None:
                debug_local.append({"type": "Feature", "geometry": mapping(debug["local"]), "properties": {"src": "sat"}})
            if debug.get("arms") is not None and not debug["arms"].is_empty:
                debug_arms.append({"type": "Feature", "geometry": mapping(debug["arms"]), "properties": {"src": "sat"}})
            if debug.get("refined") is not None:
                debug_refined.append({"type": "Feature", "geometry": mapping(debug["refined"]), "properties": {"src": "sat"}})
            props = {
                "drive_id": drive,
                "tile_id": drive,
                "src": "sat",
                "reason": "sat_fallback" if not algo_any_valid else "sat_only",
                "conf": _coerce_conf(sat_props.get("sat_confidence")),
            }
            props.update(sat_qc)
            props.update(refine_props)
            final_features.append({"type": "Feature", "geometry": mapping(chosen), "properties": props})

        _write_geojson(final_path, final_features)
        _write_geojson(out_dir / "intersections_final_wgs84.geojson", _to_wgs84(final_features, int(cfg["crs_epsg"])))
        if debug_local or debug_arms or debug_refined:
            debug_dir = out_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            if debug_local:
                _write_geojson(debug_dir / "intersections_local_clip.geojson", debug_local)
            if debug_arms:
                _write_geojson(debug_dir / "intersections_arms.geojson", debug_arms)
            if debug_refined:
                _write_geojson(debug_dir / "intersections_refined.geojson", debug_refined)

        missing_reason = "OK" if final_features else "hybrid_no_valid"
        per_drive_row = {
            "drive_id": drive,
            "tile_id": drive,
            "status": "OK" if final_features else "EMPTY",
            "missing_reason": missing_reason,
            "final_count": len(final_features),
            "sat_reject_conf": reject_counts["sat_reject_conf"],
            "sat_reject_overlap": reject_counts["sat_reject_overlap"],
            "sat_reject_dist": reject_counts["sat_reject_dist"],
            "sat_reject_shape": reject_counts["sat_reject_shape"],
            "sat_dedup": reject_counts["sat_dedup"],
        }
        per_drive_rows.append(per_drive_row)
        if debug_reject:
            breakdown_path = out_dir / f"hybrid_reject_breakdown_{drive}.json"
            breakdown_path.write_text(json.dumps(reject_counts, ensure_ascii=False, indent=2), encoding="utf-8")

    for d in expected_drives:
        if d in seen_drives:
            continue
        per_drive_rows.append(
            {
                "drive_id": d,
                "tile_id": d,
                "status": "FAIL",
                "missing_reason": "missing_entry",
                "final_count": 0,
                "sat_reject_conf": 0,
                "sat_reject_overlap": 0,
                "sat_reject_dist": 0,
                "sat_reject_shape": 0,
                "sat_dedup": 0,
            }
        )

    report_csv = out_dir / f"{args.stage}_hybrid_report_per_drive.csv"
    report_json = out_dir / f"{args.stage}_hybrid_report_per_drive.json"
    _write_csv(report_csv, per_drive_rows)
    report_json.write_text(json.dumps(per_drive_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_missing_reason_summary(report_csv, expected_drives, args.stage, out_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
