from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional at runtime
    gpd = None

try:
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - optional at runtime
    unary_union = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.calib.kitti360_backproject import configure_default_context, world_to_pixel_cam0
from pipeline.calib.kitti360_world import utm32_to_kitti_world
from scripts.pipeline_common import now_ts, setup_logging, write_csv, write_json, write_text


DRIVE_ID = "2013_05_28_drive_0010_sync"
FRAME_START = 0
FRAME_END = 300
IMAGE_CAM = "image_00"
OUTPUT_EPSG = 32632

MIN_SUPPORT_FRAMES = 5
MERGE_DIST_M = 1.5
MERGE_IOU_MIN = 0.30
AREA_RANGE_M2 = (12.0, 300.0)
WIDTH_RANGE_M = (2.5, 8.5)
LENGTH_RANGE_M = (3.0, 25.0)
ASPECT_MIN = 1.8
CONTIG_MIN = 3


def _find_data_root(cfg_root: str) -> Path:
    if cfg_root:
        path = Path(cfg_root)
        if path.exists():
            return path
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        path = Path(env_root)
        if path.exists():
            return path
    default_root = Path(r"E:\KITTI360\KITTI-360")
    if default_root.exists():
        return default_root
    raise SystemExit("missing data root: set POC_DATA_ROOT or config.kitti_root")


def _find_image_dir(data_root: Path, drive: str, camera: str) -> Path:
    candidates = [
        data_root / "data_2d_raw" / drive / camera / "data_rect",
        data_root / "data_2d_raw" / drive / camera / "data",
        data_root / drive / camera / "data_rect",
        data_root / drive / camera / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"image data not found for drive={drive} camera={camera}")


def _find_frame_path(image_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        path = image_dir / f"{frame_id}{ext}"
        if path.exists():
            return path
    return None


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _find_latest_run() -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.glob("world_crosswalk_from_stage2_0010_000_300_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_latest_pass_run() -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.glob("world_crosswalk_from_stage2_0010_000_300_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        dec = cand / "decision.json"
        if not dec.exists():
            continue
        try:
            payload = json.loads(dec.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("status", "")).strip().upper() == "PASS":
            return cand
    return None


def _find_latest_dtm_0010() -> Optional[Path]:
    runs_dir = Path("runs")
    candidates = []
    for p in runs_dir.glob("lidar_ground_0010_*"):
        if not p.is_dir():
            continue
        cand = p / "rasters" / "dtm_median_clean_utm32.tif"
        if cand.exists():
            candidates.append(cand)
        else:
            cand2 = p / "rasters" / "dtm_median_utm32.tif"
            if cand2.exists():
                candidates.append(cand2)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _check_utm_bbox(geom) -> bool:
    if geom is None or geom.is_empty:
        return False
    minx, miny, maxx, maxy = geom.bounds
    if not (100000 <= minx <= 900000 and 100000 <= maxx <= 900000):
        return False
    if not (1000000 <= miny <= 9000000 and 1000000 <= maxy <= 9000000):
        return False
    return True


def _rect_metrics(geom) -> Tuple[float, float]:
    if geom is None or geom.is_empty:
        return 0.0, 0.0
    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords) if hasattr(rect, "exterior") else []
    if len(coords) < 4:
        return 0.0, 0.0
    pts = np.array(coords[:-1], dtype=np.float64)
    edges = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.size == 0:
        return 0.0, 0.0
    length = float(np.max(lengths))
    width = float(np.min(lengths))
    return length, width


def _aspect(length: float, width: float) -> float:
    if width <= 0:
        return 0.0
    return float(length / width)


def _geom_iou(a, b) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
        uni = a.union(b).area
    except Exception:
        return 0.0
    if uni <= 0:
        return 0.0
    return float(inter / uni)


def _longest_contig(frames: List[int]) -> int:
    if not frames:
        return 0
    frames = sorted(set(frames))
    best = 1
    cur = 1
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
    best = max(best, cur)
    return int(best)


def _load_per_frame_canon(frames_dir: Path) -> Dict[int, object]:
    if gpd is None:
        return {}
    out = {}
    for p in sorted(frames_dir.iterdir()):
        if not p.is_dir():
            continue
        frame_id = p.name
        gpkg = p / "world_crosswalk_utm32.gpkg"
        if not gpkg.exists():
            continue
        try:
            gdf = gpd.read_file(gpkg, layer="canonical_utm32")
        except Exception:
            continue
        if gdf.empty:
            continue
        geom = gdf.geometry.iloc[0]
        try:
            fid = int(frame_id)
        except Exception:
            continue
        out[fid] = geom
    return out


def _assign_support_frames(candidates: List[Dict[str, object]], frame_geoms: Dict[int, object]) -> None:
    for cand in candidates:
        geom = cand["geometry"]
        frames = []
        for fid, fg in frame_geoms.items():
            if fg is None or fg.is_empty:
                continue
            dist = float(geom.centroid.distance(fg.centroid))
            iou = _geom_iou(geom, fg)
            if dist <= MERGE_DIST_M and iou >= MERGE_IOU_MIN:
                frames.append(fid)
        cand["support_frame_ids"] = frames
        if "support_frames" not in cand or not isinstance(cand["support_frames"], int):
            cand["support_frames"] = len(frames)


def _merge_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if unary_union is None:
        return candidates
    clusters: List[Dict[str, object]] = []
    for item in candidates:
        geom = item["geometry"]
        best_idx = -1
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            cgeom = cluster["geometry"]
            dist = float(geom.centroid.distance(cgeom.centroid))
            if dist > MERGE_DIST_M:
                continue
            iou = _geom_iou(geom, cgeom)
            if iou < MERGE_IOU_MIN:
                continue
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            cluster = clusters[best_idx]
            cluster["geoms"].append(geom)
            cluster["support_frames"].extend(item.get("support_frame_ids", []))
            if unary_union:
                cluster["geometry"] = unary_union(cluster["geoms"])
        else:
            clusters.append(
                {
                    "geoms": [geom],
                    "geometry": geom,
                    "support_frames": list(item.get("support_frame_ids", [])),
                }
            )
    merged = []
    for idx, cluster in enumerate(clusters, start=1):
        geoms = [g for g in cluster["geoms"] if g is not None and not g.is_empty]
        if not geoms:
            continue
        geom = unary_union(geoms)
        if geom is None or geom.is_empty:
            continue
        if not _check_utm_bbox(geom):
            continue
        frames = sorted(set(int(f) for f in cluster["support_frames"]))
        length, width = _rect_metrics(geom)
        merged.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "support_frames": len(frames),
                "support_frame_ids": frames,
                "geometry": geom,
                "area_m2": float(geom.area),
                "mrr_len_m": float(length),
                "mrr_w_m": float(width),
                "aspect": _aspect(length, width),
                "centroid_x": float(geom.centroid.x),
                "centroid_y": float(geom.centroid.y),
                "contig_len": _longest_contig(frames),
            }
        )
    return merged


def _sample_boundary_points(geom, n: int) -> List[Tuple[float, float]]:
    if geom is None or geom.is_empty:
        return []
    boundary = geom.boundary
    if boundary.is_empty or boundary.length <= 0:
        return []
    length = boundary.length
    pts = []
    for i in range(n):
        p = boundary.interpolate((i / max(1, n)) * length)
        pts.append((float(p.x), float(p.y)))
    return pts


def _draw_candidates(
    image: Image.Image,
    candidates: List[Dict[str, object]],
    ctx,
    data_root: Path,
    frame_id: str,
    dtm_path: Optional[Path],
    color: Tuple[int, int, int],
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for cand in candidates:
        geom = cand.get("geometry")
        if geom is None or geom.is_empty:
            continue
        pts = _sample_boundary_points(geom, 80)
        if not pts:
            continue
        z_list = []
        if dtm_path is not None and ctx.dtm is not None:
            for x, y in pts:
                try:
                    val = next(ctx.dtm.sample([(float(x), float(y))]))
                except Exception:
                    val = None
                z = None
                if val is not None and len(val) > 0:
                    z = float(val[0])
                    if ctx.dtm_nodata is not None and np.isfinite(ctx.dtm_nodata):
                        if abs(z - float(ctx.dtm_nodata)) < 1e-6:
                            z = None
                if z is None:
                    z = 0.0
                z_list.append(z)
        else:
            z_list = [0.0 for _ in pts]
        pts_wu = np.array([[x, y, z] for (x, y), z in zip(pts, z_list)], dtype=np.float64)
        pts_wk = utm32_to_kitti_world(pts_wu, data_root, DRIVE_ID, frame_id)
        u, v, valid = world_to_pixel_cam0(frame_id, pts_wk, ctx=ctx)
        proj = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
        if len(proj) >= 2:
            draw.line(proj + [proj[0]], fill=color, width=2)
    return image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    args = ap.parse_args()

    run_id = now_ts()
    run_dir = Path("runs") / f"crosswalk_precision_filter_0010_000_300_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    latest = _find_latest_run()
    if latest is None:
        latest = _find_latest_pass_run()
    if args.input:
        latest = Path(args.input)
    if latest is None or not latest.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "input_run_not_found"})
        return 0

    merged_dir_in = latest / "merged"
    tables_dir_in = latest / "tables"
    frames_dir_in = latest / "frames"

    cand_path = merged_dir_in / "crosswalk_candidates_canonical_utm32.gpkg"
    raw_path = merged_dir_in / "crosswalk_candidates_raw_utm32.gpkg"
    if not cand_path.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_candidates_gpkg"})
        return 0
    if gpd is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "geopandas_missing"})
        return 0

    gdf = gpd.read_file(cand_path)
    if gdf.empty:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "empty_candidates"})
        return 0
    if gdf.crs is None:
        gdf.set_crs("EPSG:32632", inplace=True)

    candidates = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not _check_utm_bbox(geom):
            continue
        support = row.get("support_frames")
        support = int(support) if support is not None and str(support).strip() != "" else None
        length, width = _rect_metrics(geom)
        cand = {
            "candidate_id": str(row.get("candidate_id") or f"cand_{idx:03d}"),
            "support_frames": support if support is not None else 0,
            "geometry": geom,
            "area_m2": float(geom.area),
            "mrr_len_m": float(length),
            "mrr_w_m": float(width),
            "aspect": _aspect(length, width),
            "centroid_x": float(geom.centroid.x),
            "centroid_y": float(geom.centroid.y),
            "contig_len": 0,
            "support_frame_ids": [],
        }
        candidates.append(cand)

    frame_geoms = _load_per_frame_canon(frames_dir_in)
    _assign_support_frames(candidates, frame_geoms)
    for cand in candidates:
        cand["contig_len"] = _longest_contig([int(f) for f in cand.get("support_frame_ids", [])])

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    before_rows = [
        {
            "candidate_id": c["candidate_id"],
            "support_frames": c["support_frames"],
            "contig_len": c["contig_len"],
            "area_m2": round(c["area_m2"], 3),
            "mrr_len_m": round(c["mrr_len_m"], 3),
            "mrr_w_m": round(c["mrr_w_m"], 3),
            "aspect": round(c["aspect"], 3),
            "centroid_x": round(c["centroid_x"], 3),
            "centroid_y": round(c["centroid_y"], 3),
        }
        for c in candidates
    ]
    write_csv(
        tables_dir / "candidate_stats_before.csv",
        before_rows,
        [
            "candidate_id",
            "support_frames",
            "contig_len",
            "area_m2",
            "mrr_len_m",
            "mrr_w_m",
            "aspect",
            "centroid_x",
            "centroid_y",
        ],
    )

    reasons = {}
    filtered = []
    for cand in candidates:
        reason = ""
        if cand["support_frames"] < MIN_SUPPORT_FRAMES:
            reason = "support_frames"
        elif cand["contig_len"] < CONTIG_MIN:
            reason = "contig_len"
        else:
            area_ok = AREA_RANGE_M2[0] <= cand["area_m2"] <= AREA_RANGE_M2[1]
            len_ok = LENGTH_RANGE_M[0] <= cand["mrr_len_m"] <= LENGTH_RANGE_M[1]
            wid_ok = WIDTH_RANGE_M[0] <= cand["mrr_w_m"] <= WIDTH_RANGE_M[1]
            asp_ok = cand["aspect"] >= ASPECT_MIN
            if not (area_ok and len_ok and wid_ok and asp_ok):
                reason = "geom_range"
        if reason:
            reasons[cand["candidate_id"]] = reason
        else:
            filtered.append(cand)

    reason_counts = Counter(reasons.values())
    reason_rows = [{"candidate_id": k, "reason": v} for k, v in reasons.items()]
    write_csv(tables_dir / "filter_reasons_summary.csv", reason_rows, ["candidate_id", "reason"])

    merged = _merge_candidates(filtered)

    after_rows = [
        {
            "candidate_id": c["candidate_id"],
            "support_frames": c["support_frames"],
            "contig_len": c["contig_len"],
            "area_m2": round(c["area_m2"], 3),
            "mrr_len_m": round(c["mrr_len_m"], 3),
            "mrr_w_m": round(c["mrr_w_m"], 3),
            "aspect": round(c["aspect"], 3),
            "centroid_x": round(c["centroid_x"], 3),
            "centroid_y": round(c["centroid_y"], 3),
        }
        for c in merged
    ]
    write_csv(
        tables_dir / "candidate_stats_after.csv",
        after_rows,
        [
            "candidate_id",
            "support_frames",
            "contig_len",
            "area_m2",
            "mrr_len_m",
            "mrr_w_m",
            "aspect",
            "centroid_x",
            "centroid_y",
        ],
    )

    merged_out_dir = run_dir / "merged"
    merged_out_dir.mkdir(parents=True, exist_ok=True)
    gdf_before = gpd.read_file(cand_path)
    gdf_before.to_file(
        merged_out_dir / "crosswalk_candidates_canonical_before_utm32.gpkg",
        layer="crosswalk_candidates",
        driver="GPKG",
    )
    gdf_after = (
        gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:32632")
        if merged
        else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32632")
    )
    gdf_after.to_file(
        merged_out_dir / "crosswalk_candidates_canonical_filtered_utm32.gpkg",
        layer="crosswalk_candidates",
        driver="GPKG",
    )

    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_root = _find_data_root("")
    image_dir = _find_image_dir(data_root, DRIVE_ID, IMAGE_CAM)
    dtm_path = _find_latest_dtm_0010()
    ctx = configure_default_context(data_root, DRIVE_ID, cam_id=IMAGE_CAM, dtm_path=dtm_path, frame_id_for_size=f"{FRAME_START:010d}")

    qa_frame = 290
    if qa_frame not in frame_geoms and frame_geoms:
        counts = Counter()
        for cand in candidates:
            for fid in cand.get("support_frame_ids", []):
                counts[int(fid)] += 1
        if counts:
            qa_frame = max(counts.items(), key=lambda x: x[1])[0]
    qa_frame_id = f"{qa_frame:010d}"
    img_path = _find_frame_path(image_dir, qa_frame_id)
    if img_path and img_path.exists():
        base = Image.open(img_path).convert("RGB")
        before_img = base.copy()
        after_img = base.copy()
        before_img = _draw_candidates(before_img, candidates, ctx, data_root, qa_frame_id, dtm_path, (0, 120, 255))
        after_img = _draw_candidates(after_img, merged, ctx, data_root, qa_frame_id, dtm_path, (0, 120, 255))
        before_path = images_dir / "qa_overlay_before.png"
        after_path = images_dir / "qa_overlay_after.png"
        before_img.save(before_path)
        after_img.save(after_path)
        w, h = before_img.size
        side = Image.new("RGB", (w * 2, h), (0, 0, 0))
        side.paste(before_img, (0, 0))
        side.paste(after_img, (w, 0))
        side.save(images_dir / "qa_overlay_side_by_side.png")

    before_count = len(candidates)
    after_count = len(merged)
    status = "PASS"
    if after_count == 0:
        status = "FAIL"
    elif after_count == before_count:
        status = "WARN"

    decision = {
        "status": status,
        "before_count": before_count,
        "after_count": after_count,
        "qa_frame": qa_frame,
        "filter_reason_top3": reason_counts.most_common(3),
        "input_run": str(latest),
    }
    write_json(run_dir / "decision.json", decision)

    resolved_cfg = {
        "INPUT_RUN": str(latest),
        "FRAME_START": FRAME_START,
        "FRAME_END": FRAME_END,
        "MIN_SUPPORT_FRAMES": MIN_SUPPORT_FRAMES,
        "MERGE_DIST_M": MERGE_DIST_M,
        "MERGE_IOU_MIN": MERGE_IOU_MIN,
        "AREA_RANGE_M2": AREA_RANGE_M2,
        "WIDTH_RANGE_M": WIDTH_RANGE_M,
        "LENGTH_RANGE_M": LENGTH_RANGE_M,
        "ASPECT_MIN": ASPECT_MIN,
        "CONTIG_MIN": CONTIG_MIN,
    }
    import yaml

    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "params_hash.txt").write_text(_hash_file(resolved_path), encoding="utf-8")

    report_lines = [
        "# Crosswalk precision filter (0010 f0-300)",
        "",
        f"- status: {status}",
        f"- before_count: {before_count}",
        f"- after_count: {after_count}",
        f"- qa_frame: {qa_frame}",
        "",
        "## filter_reason_top3",
        *[f"- {k}: {v}" for k, v in reason_counts.most_common(3)],
        "",
        "## outputs",
        "- merged/crosswalk_candidates_canonical_before_utm32.gpkg",
        "- merged/crosswalk_candidates_canonical_filtered_utm32.gpkg",
        "- tables/candidate_stats_before.csv",
        "- tables/candidate_stats_after.csv",
        "- tables/filter_reasons_summary.csv",
        "- images/qa_overlay_before.png",
        "- images/qa_overlay_after.png",
        "- images/qa_overlay_side_by_side.png",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
