from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from pipeline.calib.kitti360_backproject import configure_default_context, world_to_pixel_cam0
from pipeline.calib.kitti360_world import utm32_to_kitti_world
from scripts.pipeline_common import (
    ensure_overwrite,
    load_yaml,
    now_ts,
    relpath,
    setup_logging,
    write_csv,
    write_gpkg_layer,
    write_json,
    write_text,
)


def _find_data_root(cfg_root: str) -> Path:
    if cfg_root:
        p = Path(cfg_root)
        if p.exists():
            return p
    import os

    env_val = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
    raise SystemExit("missing data root: set POC_DATA_ROOT or config.KITTI_ROOT")


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
    raise FileNotFoundError(f"missing image dir: {drive}/{camera}")


def _find_image_path(img_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_dir / f"{frame_id}{ext}"
        if p.exists():
            return p
    return None


def _load_image(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"missing image: {path}")
    return img


def _list_candidate_paths(root: Path, name: str) -> List[Path]:
    return list(root.rglob(name))


def _pick_latest(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _infer_drive_id(path: Path) -> str:
    text = str(path)
    import os
    import re

    for part in text.split(os.sep):
        if part.startswith("2013_05_28_drive_") and part.endswith("_sync"):
            return part
    for token in text.replace("\\", "/").split("/"):
        if token.startswith("2013_05_28_drive_") and token.endswith("_sync"):
            return token
    m = re.search(r"_([0-9]{4})_", text)
    if m:
        return f"2013_05_28_drive_{m.group(1)}_sync"
    return ""


def _resolve_inputs(cfg: Dict[str, object], runs_root: Path) -> Tuple[Path, Optional[Path]]:
    manual = str(cfg.get("INPUT_GPKG_PATH", "") or "").strip()
    manual_evidence = str(cfg.get("INPUT_GPKG_EVIDENCE_PATH", "") or "").strip()
    if manual:
        main = Path(manual)
        evidence = Path(manual_evidence) if manual_evidence else None
        return main, evidence

    support_paths = _list_candidate_paths(runs_root, "crosswalk_candidates_canonical_support3_utm32.gpkg")
    all_paths = _list_candidate_paths(runs_root, "crosswalk_candidates_canonical_all_utm32.gpkg")
    canon_paths = _list_candidate_paths(runs_root, "crosswalk_candidates_canonical_utm32.gpkg")

    main = _pick_latest(support_paths)
    evidence = None
    if main is not None:
        run_dir = main.parents[1]
        candidate_all = run_dir / "merged" / "crosswalk_candidates_canonical_all_utm32.gpkg"
        if candidate_all.exists():
            evidence = candidate_all
        return main, evidence

    main = _pick_latest(canon_paths)
    if main is not None:
        return main, None

    main = _pick_latest(all_paths)
    if main is not None:
        return main, None

    raise FileNotFoundError("missing candidates gpkg in runs/")


def _utm_sanity_bounds(geom) -> bool:
    if geom is None or geom.is_empty:
        return False
    minx, miny, maxx, maxy = geom.bounds
    return 100000 <= minx <= 900000 and 100000 <= maxx <= 900000 and 1000000 <= miny <= 9000000 and 1000000 <= maxy <= 9000000


def _mrr_metrics(poly: Polygon) -> Tuple[float, float, float, float, float]:
    if poly is None or poly.is_empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    p0 = np.array(coords[0])
    p1 = np.array(coords[1])
    p2 = np.array(coords[2])
    e1 = np.linalg.norm(p1 - p0)
    e2 = np.linalg.norm(p2 - p1)
    if e1 >= e2:
        length = float(e1)
        width = float(e2)
        vec = p1 - p0
    else:
        length = float(e2)
        width = float(e1)
        vec = p2 - p1
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
    angle = angle % 180.0
    center = mrr.centroid
    return length, width, angle, float(center.x), float(center.y)


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


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def _ensure_candidate_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "candidate_id" not in gdf.columns:
        gdf["candidate_id"] = [f"cand_{i:06d}" for i in range(len(gdf))]
    return gdf


def _candidate_support(val: object) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _build_clusters(
    items: List[Dict[str, object]],
    dist_m: float,
    iou_min: float,
    edge_dist_m: float,
    ori_deg: float,
) -> List[List[int]]:
    n = len(items)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a = items[i]
            b = items[j]
            d = a["centroid"].distance(b["centroid"])
            if d > dist_m:
                continue
            iou = _geom_iou(a["geom"], b["geom"])
            if iou < iou_min:
                bd = a["geom"].boundary.distance(b["geom"].boundary)
                if bd > edge_dist_m:
                    continue
            if _angle_diff(a["angle"], b["angle"]) > ori_deg:
                continue
            adj[i].append(j)
            adj[j].append(i)

    seen = [False] * n
    clusters: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        clusters.append(comp)
    return clusters


def _feature_id(center_x: float, center_y: float, angle: float, area: float, drive_id: str) -> str:
    key = f"{center_x:.2f}|{center_y:.2f}|{angle:.1f}|{area:.1f}|{drive_id}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _write_sources_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        write_csv(path, [], ["feature_id", "candidate_id", "drive_id", "candidate_area", "candidate_bbox"])
        return
    write_csv(
        path,
        rows,
        ["feature_id", "candidate_id", "drive_id", "candidate_area", "candidate_bbox"],
    )


def _write_stats_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        write_csv(
            path,
            [],
            [
                "feature_id",
                "drive_id",
                "support",
                "confidence",
                "mrr_len_m",
                "mrr_w_m",
                "angle_deg",
                "area_m2",
                "center_x",
                "center_y",
                "source_count",
            ],
        )
        return
    write_csv(
        path,
        rows,
        [
            "feature_id",
            "drive_id",
            "support",
            "confidence",
            "mrr_len_m",
            "mrr_w_m",
            "angle_deg",
            "area_m2",
            "center_x",
            "center_y",
            "source_count",
        ],
    )


def _draw_features_overlay(
    img_path: Path,
    out_path: Path,
    features: Sequence[Dict[str, object]],
    data_root: Path,
    drive_id: str,
    frame_id: str,
    image_cam: str,
) -> None:
    import cv2

    img = _load_image(img_path)
    if not features:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        return

    ctx = configure_default_context(data_root, drive_id, cam_id=image_cam, frame_id_for_size=frame_id)
    for feat in features:
        geom = feat.get("geom") if isinstance(feat, dict) else None
        if geom is None and isinstance(feat, dict):
            geom = feat.get("geometry")
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords)
        else:
            try:
                coords = list(geom.geoms[0].exterior.coords)
            except Exception:
                continue
        pts_wu = np.array(coords, dtype=np.float64)
        if pts_wu.size == 0:
            continue
        if pts_wu.ndim == 2 and pts_wu.shape[1] == 2:
            z = np.zeros((pts_wu.shape[0], 1), dtype=np.float64)
            pts_wu = np.hstack([pts_wu, z])
        pts_wk = utm32_to_kitti_world(pts_wu, data_root, drive_id, frame_id)
        u, v, valid = world_to_pixel_cam0(frame_id, pts_wk, ctx=ctx)
        poly = []
        h, w = img.shape[:2]
        for uu, vv, ok in zip(u, v, valid):
            if not ok:
                continue
            if 0 <= uu < w and 0 <= vv < h:
                poly.append([int(round(uu)), int(round(vv))])
        if len(poly) >= 2:
            cv2.polylines(img, [np.array(poly, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def _montage(images: List[Tuple[str, Path]], out_path: Path, cols: int = 3) -> None:
    if not images:
        return
    from PIL import Image, ImageDraw

    loaded = []
    labels = []
    for label, path in images:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        loaded.append(img)
        labels.append(label)
    if not loaded:
        return
    cols = max(1, cols)
    rows = (len(loaded) + cols - 1) // cols
    w, h = loaded[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, img in enumerate(loaded):
        r = idx // cols
        c = idx % cols
        canvas.paste(img, (c * w, r * h))
        draw.text((c * w + 8, r * h + 8), labels[idx], fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/map_features_crosswalk_merge.yaml")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config)) if args.config else {}
    runs_root = Path("runs")
    run_id = now_ts()
    run_dir = runs_root / f"map_features_crosswalk_merge_{run_id}"
    outputs_dir = run_dir / "outputs"
    qa_dir = run_dir / "qa"

    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)

    setup_logging(run_dir / "run.log")

    main_path, evidence_path = _resolve_inputs(cfg, runs_root)
    gdf = gpd.read_file(main_path)
    if gdf.crs is None or (gdf.crs.to_epsg() != 32632):
        report = ["- status: FAIL", f"- reason: input_crs_invalid ({gdf.crs})", f"- input: {main_path}"]
        write_text(run_dir / "report.md", "\n".join(report))
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "input_crs_invalid"})
        raise SystemExit(1)

    gdf = _ensure_candidate_id(gdf)

    drive_id_default = str(cfg.get("DRIVE_ID", "") or "")
    merge_across = bool(cfg.get("MERGE_ACROSS_DRIVES", False))

    inferred_drive = _infer_drive_id(main_path)
    if "drive_id" not in gdf.columns:
        gdf["drive_id"] = drive_id_default or inferred_drive or "unknown"
    else:
        if gdf["drive_id"].isna().all():
            gdf["drive_id"] = drive_id_default or inferred_drive or "unknown"

    items: List[Dict[str, object]] = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        length, width, angle, cx, cy = _mrr_metrics(geom)
        items.append(
            {
                "idx": idx,
                "candidate_id": row.get("candidate_id"),
                "drive_id": row.get("drive_id"),
                "geom": geom,
                "centroid": geom.centroid,
                "length": length,
                "width": width,
                "angle": angle,
                "center": (cx, cy),
                "support": row.get("support_frames"),
            }
        )

    cluster_dist = float(cfg.get("CLUSTER_DIST_M", 2.5))
    cluster_iou = float(cfg.get("CLUSTER_IOU_MIN", 0.20))
    cluster_ori = float(cfg.get("CLUSTER_ORI_DEG", 15.0))
    edge_dist = float(cfg.get("EDGE_DIST_M", 1.0))

    clusters: List[List[int]] = []
    if not items:
        clusters = []
    else:
        if merge_across:
            clusters = _build_clusters(items, cluster_dist, cluster_iou, edge_dist, cluster_ori)
        else:
            groups: Dict[str, List[int]] = {}
            for i, item in enumerate(items):
                groups.setdefault(str(item["drive_id"]), []).append(i)
            for _, idxs in groups.items():
                sub_items = [items[i] for i in idxs]
                sub_clusters = _build_clusters(sub_items, cluster_dist, cluster_iou, edge_dist, cluster_ori)
                for comp in sub_clusters:
                    clusters.append([idxs[i] for i in comp])

    features_rows: List[Dict[str, object]] = []
    sources_rows: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []
    warnings: List[str] = []

    conf_high = int(cfg.get("CONF_HIGH_SUPPORT", 5))
    conf_med = int(cfg.get("CONF_MED_SUPPORT", 3))
    margin = float(cfg.get("FEATURE_MARGIN_M", 0.30))
    min_area = float(cfg.get("MIN_FEATURE_AREA_M2", 10.0))
    max_area = float(cfg.get("MAX_FEATURE_AREA_M2", 350.0))

    for comp in clusters:
        geoms = [items[i]["geom"] for i in comp]
        union = unary_union(geoms) if geoms else None
        if union is None or union.is_empty:
            continue
        mrr = union.minimum_rotated_rectangle
        try:
            feature_geom = mrr.buffer(margin).buffer(-margin)
        except Exception:
            feature_geom = mrr
        length, width, angle, cx, cy = _mrr_metrics(feature_geom)
        area = float(feature_geom.area) if feature_geom is not None else 0.0
        drive_id = str(items[comp[0]]["drive_id"]) if comp else "unknown"

        supports = []
        for i in comp:
            s = _candidate_support(items[i].get("support"))
            if s is not None:
                supports.append(s)
        if supports:
            support = max(supports)
        else:
            support = len(comp)

        confidence = "low"
        if support >= conf_high:
            confidence = "high"
        elif support >= conf_med:
            confidence = "med"
        if area < min_area or area > max_area:
            confidence = "low"

        fid = _feature_id(cx, cy, angle, area, drive_id)
        source_ids = [str(items[i]["candidate_id"]) for i in comp]

        features_rows.append(
            {
                "feature_id": fid,
                "drive_id": drive_id,
                "support": int(support),
                "confidence": confidence,
                "mrr_len_m": length,
                "mrr_w_m": width,
                "angle_deg": angle,
                "area_m2": area,
                "center_x": cx,
                "center_y": cy,
                "source_count": len(comp),
                "source_ids": json.dumps(source_ids, ensure_ascii=False),
                "geometry": feature_geom,
            }
        )
        stats_rows.append(
            {
                "feature_id": fid,
                "drive_id": drive_id,
                "support": int(support),
                "confidence": confidence,
                "mrr_len_m": length,
                "mrr_w_m": width,
                "angle_deg": angle,
                "area_m2": area,
                "center_x": cx,
                "center_y": cy,
                "source_count": len(comp),
            }
        )
        for i in comp:
            geom = items[i]["geom"]
            sources_rows.append(
                {
                    "feature_id": fid,
                    "candidate_id": items[i]["candidate_id"],
                    "drive_id": items[i]["drive_id"],
                    "candidate_area": float(geom.area) if geom is not None else 0.0,
                    "candidate_bbox": json.dumps(list(geom.bounds) if geom is not None else []),
                }
            )

    if features_rows:
        features_gdf = gpd.GeoDataFrame(features_rows, geometry="geometry", crs="EPSG:32632")
    else:
        features_gdf = gpd.GeoDataFrame(
            columns=[
                "feature_id",
                "drive_id",
                "support",
                "confidence",
                "mrr_len_m",
                "mrr_w_m",
                "angle_deg",
                "area_m2",
                "center_x",
                "center_y",
                "source_count",
                "source_ids",
            ],
            geometry=[],
            crs="EPSG:32632",
        )
    high_gdf = features_gdf[features_gdf["confidence"] == "high"].copy()
    med_gdf = features_gdf[features_gdf["confidence"] == "med"].copy()

    if features_gdf.empty:
        feature_count = 0
    else:
        feature_count = int(len(features_gdf))

    bbox_ok = True
    for geom in features_gdf.geometry:
        if not _utm_sanity_bounds(geom):
            bbox_ok = False
            break

    write_gpkg_layer(outputs_dir / "crosswalk_features_utm32.gpkg", "crosswalk_features", features_gdf, 32632, warnings)
    write_gpkg_layer(
        outputs_dir / "crosswalk_features_highconf_utm32.gpkg", "crosswalk_features", high_gdf, 32632, warnings
    )
    write_gpkg_layer(
        outputs_dir / "crosswalk_features_medconf_utm32.gpkg", "crosswalk_features", med_gdf, 32632, warnings
    )

    _write_sources_csv(outputs_dir / "crosswalk_feature_sources.csv", sources_rows)
    _write_stats_csv(outputs_dir / "crosswalk_feature_stats.csv", stats_rows)

    # QA overlays
    qa_frames = []
    input_run_dir = main_path.parents[1]
    qa_frames_path = input_run_dir / "qa" / "qa_frames.json"
    if qa_frames_path.exists():
        try:
            qa_payload = json.loads(qa_frames_path.read_text(encoding="utf-8"))
            qa_frames = [int(f) for f in qa_payload.get("frames", [])]
        except Exception:
            qa_frames = []
    if not qa_frames:
        qa_frames = list(cfg.get("QA_FORCE_INCLUDE", [290]))
        rng = random.Random(int(cfg.get("QA_RANDOM_SEED", 20260130)))
        for _ in range(int(cfg.get("QA_RANDOM_N", 2))):
            qa_frames.append(rng.randint(0, 500))
    qa_frames = sorted(set(qa_frames))

    data_root = _find_data_root(str(cfg.get("KITTI_ROOT", "")))
    drive_id = str(gdf["drive_id"].iloc[0]) if not gdf.empty else (drive_id_default or inferred_drive or "2013_05_28_drive_0010_sync")
    image_cam = str(cfg.get("IMAGE_CAM", "image_00"))
    img_dir = _find_image_dir(data_root, drive_id, image_cam)

    overlay_items: List[Tuple[str, Path]] = []
    for frame in qa_frames:
        frame_id = f"{int(frame):010d}"
        img_path = _find_image_path(img_dir, frame_id)
        if img_path is None:
            continue
        out_path = qa_dir / "overlays" / f"frame_{frame_id}_features.png"
        try:
            _draw_features_overlay(
                img_path,
                out_path,
                [row for row in features_rows],
                data_root,
                drive_id,
                frame_id,
                image_cam,
            )
            overlay_items.append((frame_id, out_path))
        except Exception:
            continue

    qa_dir.mkdir(parents=True, exist_ok=True)
    write_json(qa_dir / "qa_frames.json", {"frames": qa_frames})
    _montage(overlay_items, qa_dir / "montage_features.png", cols=3)

    resolved = dict(cfg)
    resolved["RESOLVED"] = {
        "run_id": run_id,
        "input_main": str(main_path),
        "input_evidence": str(evidence_path) if evidence_path else "",
    }
    write_text(run_dir / "resolved_config.yaml", json.dumps(resolved, ensure_ascii=False, indent=2))
    params_hash = hashlib.sha1(json.dumps(resolved, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    write_text(run_dir / "params_hash.txt", params_hash)

    status = "PASS"
    if feature_count == 0:
        status = "WARN"
    if not bbox_ok:
        status = "FAIL"

    decision = {
        "status": status,
        "feature_count": feature_count,
        "bbox_sanity": bbox_ok,
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        f"- status: {status}",
        f"- feature_count: {feature_count}",
        f"- bbox_sanity: {bbox_ok}",
        f"- input_main: {relpath(run_dir, main_path)}",
    ]
    if evidence_path is not None:
        report.append(f"- input_evidence: {relpath(run_dir, evidence_path)}")
    report.extend(
        [
            "",
            "## outputs",
            f"- {relpath(run_dir, outputs_dir / 'crosswalk_features_utm32.gpkg')}",
            f"- {relpath(run_dir, outputs_dir / 'crosswalk_feature_sources.csv')}",
            f"- {relpath(run_dir, qa_dir / 'montage_features.png')}",
        ]
    )
    write_text(run_dir / "report.md", "\n".join(report))


if __name__ == "__main__":
    main()
