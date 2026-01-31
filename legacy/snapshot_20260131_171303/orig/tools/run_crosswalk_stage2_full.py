from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import geopandas as gpd
import pandas as pd

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def _write_config(cfg: Dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _load_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    return gpd.read_file(path, layer=layer)


def _parse_support_frames(frames_str: str) -> List[str]:
    if not frames_str:
        return []
    return [f for f in str(frames_str).split("|") if f]


def _expected_clusters(
    cluster_df: pd.DataFrame,
    review_gdf: gpd.GeoDataFrame,
    trace_df: pd.DataFrame,
    min_frames_hit: int,
    min_reproj_iou_bbox_p50: float,
    max_gore_like_ratio: float,
) -> Dict[str, Any]:
    expected = {}
    if cluster_df.empty or review_gdf.empty:
        return expected
    reproj_p50 = {}
    if not trace_df.empty and "cluster_id" in trace_df.columns and "reproj_iou_bbox" in trace_df.columns:
        grouped = trace_df.dropna(subset=["cluster_id"]).groupby("cluster_id")["reproj_iou_bbox"]
        reproj_p50 = grouped.median().to_dict()
    review = review_gdf.copy()
    review["cluster_id"] = review.get("cluster_id", "").astype(str)
    for _, row in cluster_df.iterrows():
        cid = str(row.get("cluster_id") or "")
        if not cid:
            continue
        if int(row.get("frames_hit_support", 0)) < min_frames_hit:
            continue
        if float(row.get("gore_like_ratio", 0.0)) > max_gore_like_ratio:
            continue
        if float(reproj_p50.get(cid, 0.0) or 0.0) < min_reproj_iou_bbox_p50:
            continue
        subset = review[review["cluster_id"] == cid]
        if subset.empty:
            continue
        geom = subset.unary_union
        if geom is None or geom.is_empty:
            continue
        expected[cid] = geom.centroid
    return expected


def _select_qa_frames(
    final_gdf: gpd.GeoDataFrame,
    cluster_df: pd.DataFrame,
    expected_centroids: List[Any],
    fp_dist_m: float,
    topk_support_frames: int,
    topn_fp: int,
) -> List[str]:
    frames: List[str] = []
    if final_gdf.empty:
        return frames
    cluster_df = cluster_df.copy() if not cluster_df.empty else pd.DataFrame()
    cluster_df["cluster_id"] = cluster_df.get("cluster_id", "").astype(str)
    support_map = {
        str(row.get("cluster_id") or ""): _parse_support_frames(str(row.get("support_frames") or ""))
        for _, row in cluster_df.iterrows()
    }
    final_gdf = final_gdf.copy()
    final_gdf["cluster_id"] = final_gdf.get("cluster_id", "").astype(str)
    fp_clusters: List[Tuple[str, float]] = []
    for _, row in final_gdf.iterrows():
        cid = str(row.get("cluster_id") or "")
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not expected_centroids:
            fp_clusters.append((cid, fp_dist_m + 1.0))
            continue
        dist = min(geom.centroid.distance(exp) for exp in expected_centroids)
        if dist > fp_dist_m:
            fp_clusters.append((cid, dist))
    fp_clusters = sorted(fp_clusters, key=lambda x: x[1], reverse=True)[:topn_fp]
    fp_set = {cid for cid, _ in fp_clusters}
    for _, row in final_gdf.iterrows():
        cid = str(row.get("cluster_id") or "")
        support = support_map.get(cid, [])
        if support:
            frames.extend(support[:topk_support_frames])
    for cid in fp_set:
        support = support_map.get(cid, [])
        if support:
            frames.extend(support[:topk_support_frames])
    return sorted({f for f in frames if f})


def _filter_qa_outputs(outputs_dir: Path, drive_id: str, keep_frames: Iterable[str]) -> None:
    qa_index = outputs_dir / "qa_index_wgs84.geojson"
    if qa_index.exists():
        qa_gdf = gpd.read_file(qa_index)
        if not qa_gdf.empty and "frame_id" in qa_gdf.columns:
            keep = {str(f) for f in keep_frames}
            qa_gdf = qa_gdf[qa_gdf["frame_id"].astype(str).isin(keep)]
            qa_gdf.to_file(qa_index, driver="GeoJSON")
    qa_dir = outputs_dir / "qa_images" / drive_id
    if qa_dir.exists():
        keep = {str(f) for f in keep_frames}
        for path in qa_dir.glob("*.png"):
            name = path.stem
            parts = name.split("_")
            if not parts:
                continue
            frame_id = parts[0]
            if frame_id not in keep:
                path.unlink()


def _write_trial_summary(outputs_dir: Path) -> None:
    trace_path = outputs_dir / "crosswalk_trace.csv"
    cluster_path = outputs_dir / "cluster_summary.csv"
    gpkg_path = outputs_dir / "crosswalk_entities_utm32.gpkg"
    summary = {}
    if trace_path.exists():
        trace_df = pd.read_csv(trace_path)
        summary["trace_rows"] = int(len(trace_df))
        summary["raw_has_crosswalk"] = int((trace_df.get("raw_has_crosswalk", 0) == 1).sum()) if "raw_has_crosswalk" in trace_df.columns else 0
    if cluster_path.exists():
        cluster_df = pd.read_csv(cluster_path)
        summary["clusters"] = int(len(cluster_df))
        summary["final_pass"] = int((cluster_df.get("final_pass", 0).astype(int) == 1).sum()) if "final_pass" in cluster_df.columns else 0
    if gpkg_path.exists():
        final_gdf = _load_layer(gpkg_path, "crosswalk_poly")
        summary["final_count"] = int(len(final_gdf))
    (outputs_dir / "trial_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_stage2_sam2video.yaml")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--lidar-world-mode", default=None)
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--out-run", default="")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    merged = _merge(
        cfg,
        {
            "drive_id": args.drive,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "camera": args.camera,
            "lidar_world_mode": args.lidar_world_mode,
            "kitti_root": args.kitti_root,
        },
    )
    drive_id = str(merged.get("drive_id") or "unknown")
    frame_start = merged.get("frame_start")
    frame_end = merged.get("frame_end")
    if args.out_run:
        run_dir = Path(args.out_run)
    else:
        tag = drive_id.split("_")[-2] if "_" in drive_id else drive_id
        run_dir = Path("runs") / f"crosswalk_stage2_full_{tag}_{frame_start}_{frame_end}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "trial_config.yaml"
    _write_config(merged, cfg_path)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_crosswalk_monitor_range.py"),
        "--config",
        str(cfg_path),
        "--out-run",
        str(run_dir),
    ]
    if args.drive:
        cmd.extend(["--drive", args.drive])
    if args.frame_start is not None:
        cmd.extend(["--frame-start", str(args.frame_start)])
    if args.frame_end is not None:
        cmd.extend(["--frame-end", str(args.frame_end)])
    if args.kitti_root:
        cmd.extend(["--kitti-root", args.kitti_root])

    print(json.dumps({"cmd": cmd}, ensure_ascii=True))
    result = subprocess.run(cmd, check=False)
    outputs_dir = run_dir / "outputs"
    _write_trial_summary(outputs_dir)
    if outputs_dir.exists():
        qa_policy = merged.get("qa_policy", {}) if isinstance(merged.get("qa_policy"), dict) else {}
        if qa_policy.get("enable", False):
            gpkg_path = outputs_dir / "crosswalk_entities_utm32.gpkg"
            trace_path = outputs_dir / "crosswalk_trace.csv"
            cluster_path = outputs_dir / "cluster_summary.csv"
            review_gdf = _load_layer(gpkg_path, "crosswalk_review_poly")
            final_gdf = _load_layer(gpkg_path, "crosswalk_poly")
            cluster_df = pd.read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()
            trace_df = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
            expected_cfg = qa_policy.get("expected", {})
            expected = _expected_clusters(
                cluster_df,
                review_gdf,
                trace_df,
                int(expected_cfg.get("min_frames_hit_support", 10)),
                float(expected_cfg.get("min_reproj_iou_bbox_p50", 0.1)),
                float(expected_cfg.get("max_gore_like_ratio", 0.5)),
            )
            keep_frames = _select_qa_frames(
                final_gdf,
                cluster_df,
                list(expected.values()),
                float(qa_policy.get("fp_dist_m", 15.0)),
                int(qa_policy.get("topk_support_frames", 10)),
                int(qa_policy.get("topn_fp", 20)),
            )
            _filter_qa_outputs(outputs_dir, drive_id, keep_frames)
            near_cfg = qa_policy.get("near_final", {}) if isinstance(qa_policy.get("near_final"), dict) else {}
            if near_cfg.get("enable", False) and cluster_df is not None and not cluster_df.empty:
                topk = int(near_cfg.get("topk", 3))
                cand = cluster_df[cluster_df.get("final_pass", 0).astype(int) == 0]
                cand = cand.sort_values("frames_hit_support", ascending=False)
                cluster_ids = [str(cid) for cid in cand.get("cluster_id", []).tolist()[:topk] if cid]
                tool_path = REPO_ROOT / "tools" / "near_final_diagnose.py"
                if tool_path.exists() and cluster_ids:
                    out_dir = outputs_dir / "near_final_diagnose"
                    cmd = [
                        sys.executable,
                        str(tool_path),
                        "--run-dir",
                        str(run_dir),
                        "--clusters",
                        ",".join(cluster_ids),
                        "--out-dir",
                        str(out_dir),
                    ]
                    subprocess.run(cmd, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
