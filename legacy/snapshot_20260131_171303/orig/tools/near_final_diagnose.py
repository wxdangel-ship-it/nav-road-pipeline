from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import load_kitti360_pose, load_kitti360_pose_full


def _normalize_frame_id(frame_id: str) -> str:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    if not digits:
        return str(frame_id)
    return digits.zfill(10)


def _infer_kitti_root(image_path: str) -> Path | None:
    if not image_path:
        return None
    path = Path(image_path)
    parts = [p.lower() for p in path.parts]
    if "data_2d_raw" in parts:
        idx = parts.index("data_2d_raw")
        return Path(*path.parts[:idx])
    return None


def _rect_heading_deg(geom) -> float | None:
    if geom is None or geom.is_empty:
        return None
    try:
        rect = geom.minimum_rotated_rectangle
    except Exception:
        return None
    if rect is None or rect.is_empty:
        return None
    coords = list(rect.exterior.coords)
    if len(coords) < 2:
        return None
    best_len = -1.0
    best_ang = None
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_ang = math.degrees(math.atan2(dy, dx))
    if best_ang is None:
        return None
    ang = best_ang % 180.0
    return ang


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def _sample_even(items: List[str], count: int) -> List[str]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return items
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in idx]


def _fill_unique(primary: List[str], fallback: List[str], count: int) -> List[str]:
    out = []
    for item in primary + fallback:
        if item and item not in out:
            out.append(item)
        if len(out) >= count:
            break
    return out


def _draw_arrow(draw: ImageDraw.ImageDraw, origin: Tuple[int, int], angle_deg: float, length: int, color):
    if angle_deg is None:
        return
    ang = math.radians(angle_deg)
    x0, y0 = origin
    x1 = x0 + length * math.cos(ang)
    y1 = y0 - length * math.sin(ang)
    draw.line((x0, y0, x1, y1), fill=color, width=3)
    head_len = max(6, length * 0.2)
    left = ang + math.radians(150)
    right = ang - math.radians(150)
    lx = x1 + head_len * math.cos(left)
    ly = y1 - head_len * math.sin(left)
    rx = x1 + head_len * math.cos(right)
    ry = y1 - head_len * math.sin(right)
    draw.line((x1, y1, lx, ly), fill=color, width=3)
    draw.line((x1, y1, rx, ry), fill=color, width=3)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    arr = np.array(values, dtype=float)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _split_rejects(value: str) -> List[str]:
    if not value:
        return []
    parts = []
    for token in value.replace(",", "|").split("|"):
        token = token.strip()
        if token:
            parts.append(token)
    return parts


def _select_clusters(cluster_df: pd.DataFrame, limit: int = 3) -> List[str]:
    if cluster_df.empty:
        return []
    df = cluster_df.copy()
    if "final_pass" in df.columns:
        df = df[df["final_pass"].fillna(0).astype(int) == 0]
    for col in ("frames_hit_support", "frames_hit_all", "stage2_added_frames_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df.sort_values(
        ["frames_hit_support", "stage2_added_frames_count", "frames_hit_all"],
        ascending=False,
    )
    return [str(v) for v in df["cluster_id"].tolist()[:limit]]


def _load_candidates(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(path, layer="crosswalk_candidate_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _build_frame_records(
    cluster_id: str,
    drive_id: str,
    trace_df: pd.DataFrame,
    cand_df: gpd.GeoDataFrame,
    overlay_dir: Path,
    kitti_root: Path | None,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    trace_subset = trace_df[trace_df["cluster_id"] == cluster_id].copy()
    if trace_subset.empty:
        trace_subset = trace_df.iloc[0:0].copy()
    trace_subset["frame_id_norm"] = trace_subset["frame_id"].apply(_normalize_frame_id)
    cand_df = cand_df[cand_df["cluster_id"] == cluster_id].copy()
    if not cand_df.empty:
        cand_df["frame_id_norm"] = cand_df["frame_id"].apply(_normalize_frame_id)
    by_frame = {}
    if not cand_df.empty:
        for frame_id, group in cand_df.groupby("frame_id_norm"):
            if "geom_area_m2" in group.columns:
                idx = group["geom_area_m2"].astype(float).fillna(0).idxmax()
            elif "area_m2" in group.columns:
                idx = group["area_m2"].astype(float).fillna(0).idxmax()
            else:
                idx = group.index[0]
            row = group.loc[idx]
            by_frame[frame_id] = row

    frame_rows = []
    geom_centroids: Dict[str, Tuple[float, float]] = {}
    rect_headings: Dict[str, float] = {}
    for _, row in trace_subset.iterrows():
        frame_id = _normalize_frame_id(row["frame_id"])
        cand = by_frame.get(frame_id)
        geom = cand.geometry if cand is not None else None
        centroid = geom.centroid if geom is not None and not geom.is_empty else None
        if centroid is not None:
            geom_centroids[frame_id] = (float(centroid.x), float(centroid.y))
        rect_heading = _rect_heading_deg(geom) if geom is not None else None
        if rect_heading is not None:
            rect_headings[frame_id] = rect_heading

    if geom_centroids:
        xs = [v[0] for v in geom_centroids.values()]
        ys = [v[1] for v in geom_centroids.values()]
        med_x = float(np.median(xs))
        med_y = float(np.median(ys))
    else:
        med_x = med_y = float("nan")

    for _, row in trace_subset.iterrows():
        frame_id = _normalize_frame_id(row["frame_id"])
        cand = by_frame.get(frame_id)
        geom = cand.geometry if cand is not None else None
        rect_w = float(cand.get("rect_w_m", np.nan)) if cand is not None else float(row.get("rect_w_m", np.nan))
        rect_l = float(cand.get("rect_l_m", np.nan)) if cand is not None else float(row.get("rect_l_m", np.nan))
        rectangularity = float(cand.get("rectangularity", np.nan)) if cand is not None else float(row.get("rectangularity", np.nan))
        heading_diff = float(cand.get("heading_diff_to_perp_deg", np.nan)) if cand is not None else float("nan")
        rect_heading = rect_headings.get(frame_id)
        road_heading = None
        if kitti_root is not None:
            try:
                x, y, z, roll, pitch, yaw = load_kitti360_pose_full(kitti_root, drive_id, frame_id)
                road_heading = math.degrees(yaw)
            except Exception:
                try:
                    x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
                    road_heading = math.degrees(yaw)
                except Exception:
                    road_heading = None
        if road_heading is None and rect_heading is not None:
            road_heading = (rect_heading - 90.0) % 360.0
        if not np.isfinite(heading_diff) and rect_heading is not None and road_heading is not None:
            diff = _angle_diff_deg(rect_heading, road_heading)
            heading_diff = min(diff, abs(diff - 90.0))
        centroid = geom.centroid if geom is not None and not geom.is_empty else None
        if centroid is not None and math.isfinite(med_x) and math.isfinite(med_y):
            jitter = math.hypot(centroid.x - med_x, centroid.y - med_y)
            centroid_x = float(centroid.x)
            centroid_y = float(centroid.y)
        else:
            jitter = float("nan")
            centroid_x = float("nan")
            centroid_y = float("nan")
        overlay_raw = overlay_dir / f"{frame_id}_overlay_raw.png"
        overlay_gated = overlay_dir / f"{frame_id}_overlay_gated.png"
        overlay_entities = overlay_dir / f"{frame_id}_overlay_entities.png"
        frame_rows.append(
            {
                "frame_id": frame_id,
                "raw_has_crosswalk": int(row.get("raw_has_crosswalk", 0)),
                "raw_top_score": float(row.get("raw_top_score", 0.0)),
                "stage2_added": int(row.get("stage2_added", 0)),
                "proj_method": str(row.get("proj_method", "")),
                "geom_ok": int(row.get("geom_ok", 0)),
                "geom_area_m2": float(row.get("geom_area_m2", 0.0)),
                "rect_w": rect_w,
                "rect_l": rect_l,
                "rectangularity": rectangularity,
                "road_heading_deg": road_heading if road_heading is not None else "",
                "rect_heading_deg": rect_heading if rect_heading is not None else "",
                "heading_diff_deg": heading_diff if not np.isnan(heading_diff) else "",
                "jitter_to_cluster_m": jitter if not np.isnan(jitter) else "",
                "centroid_x": centroid_x if not np.isnan(centroid_x) else "",
                "centroid_y": centroid_y if not np.isnan(centroid_y) else "",
                "support_flag": int(row.get("support_flag", 0)) if "support_flag" in row else 0,
                "reject_reasons": str(row.get("reject_reasons", "")),
                "overlay_raw_path": str(overlay_raw),
                "overlay_gated_path": str(overlay_gated),
                "overlay_entities_path": str(overlay_entities),
            }
        )
    return pd.DataFrame(frame_rows), {"med_x": med_x, "med_y": med_y}


def _annotate_frames(
    frames_df: pd.DataFrame,
    out_dir: Path,
    prefer_gated: bool = True,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_paths = []
    for _, row in frames_df.iterrows():
        frame_id = row["frame_id"]
        path = row["overlay_gated_path"] if prefer_gated else row["overlay_raw_path"]
        if not path or not Path(path).exists():
            path = row["overlay_raw_path"]
        if not path or not Path(path).exists():
            continue
        img = Image.open(path).convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")
        w, h = img.size
        origin = (40, h - 40)
        road_heading = row.get("road_heading_deg")
        rect_heading = row.get("rect_heading_deg")
        try:
            road_heading = float(road_heading)
        except Exception:
            road_heading = None
        try:
            rect_heading = float(rect_heading)
        except Exception:
            rect_heading = None
        _draw_arrow(draw, origin, road_heading, 60, (0, 128, 255, 220))
        _draw_arrow(draw, origin, rect_heading, 60, (255, 64, 64, 220))
        label = (
            f"heading_diff={row.get('heading_diff_deg')} "
            f"rect={row.get('rect_w')}x{row.get('rect_l')} "
            f"rectangularity={row.get('rectangularity')} "
            f"stage2_added={row.get('stage2_added')}"
        )
        draw.text((origin[0] + 80, origin[1] - 30), label, fill=(255, 255, 255, 220))
        out_path = out_dir / f"{frame_id}_gated_annot.png"
        img.save(out_path)
        annotated_paths.append(str(out_path))
    return annotated_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--clusters", default="")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.name != "outputs":
        run_dir = run_dir / "outputs"
    if not run_dir.exists():
        raise SystemExit(f"missing_run_dir:{run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "near_final_diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / "crosswalk_trace.csv"
    cluster_path = run_dir / "cluster_summary.csv"
    gpkg_path = run_dir / "crosswalk_entities_utm32.gpkg"
    if not trace_path.exists() or not cluster_path.exists():
        raise SystemExit("missing_trace_or_cluster_summary")

    trace_df = pd.read_csv(trace_path)
    trace_df["frame_id"] = trace_df["frame_id"].apply(_normalize_frame_id)
    cluster_df = pd.read_csv(cluster_path)

    if args.clusters:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]
    else:
        clusters = _select_clusters(cluster_df, limit=3)
    if not clusters:
        raise SystemExit("no_clusters_selected")

    cand_df = _load_candidates(gpkg_path)
    if not cand_df.empty:
        cand_df["frame_id"] = cand_df["frame_id"].apply(_normalize_frame_id)

    drive_id = str(trace_df["drive_id"].iloc[0]) if not trace_df.empty else ""
    image_path = str(trace_df["image_path"].iloc[0]) if not trace_df.empty else ""
    kitti_root = _infer_kitti_root(image_path)

    overview_rows = []
    report_lines = ["# Near-Final Diagnose Report", ""]
    report_lines.append("## Overview")

    for cluster_id in clusters:
        cluster_dir = out_dir / "clusters" / cluster_id
        cluster_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir = run_dir / "qa_images" / drive_id

        frames_df, centroid_info = _build_frame_records(
            cluster_id,
            drive_id,
            trace_df,
            cand_df,
            overlay_dir,
            kitti_root,
        )
        if frames_df.empty:
            continue
        frames_all = frames_df.copy()

        rect_stats = _percentiles([v for v in frames_df["rect_w"].tolist() if pd.notna(v)])
        rect_l_stats = _percentiles([v for v in frames_df["rect_l"].tolist() if pd.notna(v)])
        rectr_stats = _percentiles([v for v in frames_df["rectangularity"].tolist() if pd.notna(v)])
        hd_stats = _percentiles([float(v) for v in frames_df["heading_diff_deg"].tolist() if str(v) != "" and pd.notna(v)])
        jitter_stats = _percentiles([float(v) for v in frames_df["jitter_to_cluster_m"].tolist() if str(v) != "" and pd.notna(v)])

        _write_json(
            cluster_dir / "cluster_stats.json",
            {
                "rect_w": rect_stats,
                "rect_l": rect_l_stats,
                "rectangularity": rectr_stats,
                "heading_diff": hd_stats,
                "jitter": jitter_stats,
                "centroid_median": centroid_info,
            },
        )

        timeseries_rect = frames_all[["frame_id", "rect_w", "rect_l", "geom_area_m2", "rectangularity"]]
        timeseries_rect.to_csv(cluster_dir / "timeseries_rect.csv", index=False)
        timeseries_heading = frames_all[["frame_id", "road_heading_deg", "rect_heading_deg", "heading_diff_deg"]]
        timeseries_heading.to_csv(cluster_dir / "timeseries_heading.csv", index=False)
        timeseries_jitter = frames_all[["frame_id", "centroid_x", "centroid_y", "jitter_to_cluster_m"]].copy()
        timeseries_jitter.to_csv(cluster_dir / "timeseries_jitter.csv", index=False)

        if "support_flag" in frames_df.columns:
            support_frames = frames_df[frames_df["support_flag"].astype(int) == 1].copy()
        else:
            support_frames = frames_df[frames_df.get("stage2_added", 0).astype(int) == 1].copy()
        if support_frames.empty:
            support_frames = frames_df.sort_values("raw_top_score", ascending=False)
        support_frames = support_frames.head(10)

        random_frames = _sample_even(frames_df["frame_id"].tolist(), 10)
        failure_frames = []
        if "heading_diff_deg" in frames_df.columns:
            failure_frames.extend(
                frames_df.sort_values("heading_diff_deg", ascending=False).head(2)["frame_id"].tolist()
            )
        failure_frames.extend(frames_df.sort_values("rect_w", ascending=False).head(1)["frame_id"].tolist())
        failure_frames.extend(frames_df.sort_values("rect_w", ascending=True).head(1)["frame_id"].tolist())
        if "jitter_to_cluster_m" in frames_df.columns:
            failure_frames.extend(
                frames_df.sort_values("jitter_to_cluster_m", ascending=False).head(1)["frame_id"].tolist()
            )
        if "rectangularity" in frames_df.columns:
            failure_frames.extend(
                frames_df.sort_values("rectangularity", ascending=True).head(1)["frame_id"].tolist()
            )
        failure_frames = [f for f in failure_frames if f]
        all_frames = frames_df["frame_id"].tolist()
        best_candidates = support_frames.sort_values("raw_top_score", ascending=False)["frame_id"].tolist()
        best_frames = _fill_unique(best_candidates, all_frames, 3)
        worst_frames = _fill_unique(failure_frames, all_frames, 3)
        selected = list(
            dict.fromkeys(
                support_frames["frame_id"].tolist()
                + random_frames
                + failure_frames
            )
        )
        selected = list(dict.fromkeys(selected + best_frames + worst_frames))
        selected_df = frames_df[frames_df["frame_id"].isin(selected)]
        selected_df.to_csv(cluster_dir / "frames.csv", index=False)

        annotated_paths = _annotate_frames(selected_df, cluster_dir / "annotated")

        reject_counts: Dict[str, int] = {}
        for val in frames_df["reject_reasons"].tolist():
            for token in _split_rejects(str(val)):
                reject_counts[token] = reject_counts.get(token, 0) + 1
        fail_top3 = ",".join([k for k, _ in sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)[:3]])

        summary_row = cluster_df[cluster_df["cluster_id"] == cluster_id]
        if not summary_row.empty:
            row = summary_row.iloc[0]
            frames_hit_all = int(row.get("frames_hit_all", 0))
            frames_hit_support = int(row.get("frames_hit_support", 0))
            stage2_added_frames = int(row.get("stage2_added_frames_count", 0))
        else:
            frames_hit_all = len(frames_df)
            frames_hit_support = int(support_frames.shape[0])
            stage2_added_frames = int(frames_df["stage2_added"].sum())

        overview_rows.append(
            {
                "cluster_id": cluster_id,
                "frames_hit_all": frames_hit_all,
                "frames_hit_support": frames_hit_support,
                "stage2_added_frames": stage2_added_frames,
                "fail_reasons_top3": fail_top3,
                "rect_w_p50": rect_stats["p50"],
                "rect_w_p90": rect_stats["p90"],
                "rect_l_p50": rect_l_stats["p50"],
                "rect_l_p90": rect_l_stats["p90"],
                "heading_diff_p50": hd_stats["p50"],
                "heading_diff_p90": hd_stats["p90"],
                "jitter_p50": jitter_stats["p50"],
                "jitter_p90": jitter_stats["p90"],
                "rectangularity_p10": rectr_stats["p10"],
                "rectangularity_p50": rectr_stats["p50"],
            }
        )

        report_lines.append(
            f"- {cluster_id} frames_hit_all={frames_hit_all} frames_hit_support={frames_hit_support} "
            f"stage2_added={stage2_added_frames} fail_top3={fail_top3}"
        )

        diagnosis = []
        if not math.isnan(hd_stats["p50"]) and hd_stats["p50"] > 25:
            diagnosis.append("heading_diff_high")
        if not math.isnan(rect_stats["p90"]) and rect_stats["p90"] > 30:
            diagnosis.append("rect_w_too_wide")
        if not math.isnan(rect_stats["p10"]) and rect_stats["p10"] < 1.5:
            diagnosis.append("rect_w_too_narrow")
        if not math.isnan(rectr_stats["p10"]) and rectr_stats["p10"] < 0.35:
            diagnosis.append("rectangularity_low")
        if not math.isnan(jitter_stats["p90"]) and jitter_stats["p90"] > 8:
            diagnosis.append("jitter_high")
        if not diagnosis:
            diagnosis.append("gate_or_metric_mismatch")
        report_lines.append(f"  - diagnosis_hint: {', '.join(diagnosis)}")
        if "heading_diff_high" in diagnosis or "rect_w_too_wide" in diagnosis or "rectangularity_low" in diagnosis:
            conclusion = "geometry_or_heading_mismatch_likely"
        elif "jitter_high" in diagnosis:
            conclusion = "propagation_drift_likely"
        else:
            conclusion = "gate_or_metric_mismatch_possible"
        report_lines.append(f"  - diagnosis_conclusion: {conclusion}")

        best_paths = [
            str(cluster_dir / "annotated" / f"{fid}_gated_annot.png") for fid in best_frames if fid
        ]
        worst_paths = [
            str(cluster_dir / "annotated" / f"{fid}_gated_annot.png") for fid in worst_frames if fid
        ]
        report_lines.append(f"  - best_frames: {', '.join(best_paths)}")
        report_lines.append(f"  - worst_frames: {', '.join(worst_paths)}")
        report_lines.append(f"  - annotated_examples: {', '.join(annotated_paths[:6])}")

    overview_path = out_dir / "overview.csv"
    pd.DataFrame(overview_rows).to_csv(overview_path, index=False)

    report_path = out_dir / "near_final_report.md"
    report_lines.append("")
    report_lines.append(f"- overview_csv: {overview_path}")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
