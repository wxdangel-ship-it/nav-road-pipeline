from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _flatten_params(keys: List[str], values: List[Any]) -> Dict[str, Any]:
    return {k: v for k, v in zip(keys, values)}


def _load_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    return gpd.read_file(path, layer=layer)


def _expected_clusters(
    cluster_df: pd.DataFrame,
    candidate_gdf: gpd.GeoDataFrame,
    min_frames_hit: int,
    max_gore_like_ratio: float,
) -> Dict[str, Any]:
    expected = {}
    if cluster_df.empty or candidate_gdf.empty:
        return expected
    cand = candidate_gdf.copy()
    cand["cluster_id"] = cand.get("cluster_id", "").astype(str)
    cand["stage2_added"] = cand.get("stage2_added", 0).fillna(0).astype(int)
    for _, row in cluster_df.iterrows():
        cid = str(row.get("cluster_id") or "")
        if not cid:
            continue
        if int(row.get("frames_hit_stage1", 0)) < min_frames_hit:
            continue
        if float(row.get("gore_like_ratio", 0.0)) > max_gore_like_ratio:
            continue
        subset = cand[(cand["cluster_id"] == cid) & (cand["stage2_added"] != 1)]
        if subset.empty:
            subset = cand[cand["cluster_id"] == cid]
        if subset.empty:
            continue
        geom = subset.unary_union
        if geom is None or geom.is_empty:
            continue
        expected[cid] = geom.centroid
    return expected


def _score_trial(
    outputs_dir: Path,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    gpkg_path = outputs_dir / "crosswalk_entities_utm32.gpkg"
    trace_path = outputs_dir / "crosswalk_trace.csv"
    cluster_path = outputs_dir / "cluster_summary.csv"
    final_gdf = _load_layer(gpkg_path, "crosswalk_poly")
    cand_gdf = _load_layer(gpkg_path, "crosswalk_candidate_poly")
    cluster_df = pd.read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()
    trace_df = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()

    expected_cfg = cfg.get("autotune", {}).get("expected", {})
    expected = _expected_clusters(
        cluster_df,
        cand_gdf,
        int(expected_cfg.get("min_frames_hit_stage1", 3)),
        float(expected_cfg.get("max_gore_like_ratio", 0.7)),
    )
    expected_centroids = list(expected.values())
    expected_count = len(expected_centroids)
    final_centroids = [geom.centroid for geom in final_gdf.geometry] if not final_gdf.empty else []
    match_dist = float(expected_cfg.get("match_dist_m", 8.0))
    fp_dist = float(expected_cfg.get("fp_dist_m", 15.0))

    matched = 0
    for exp in expected_centroids:
        if final_centroids and min(exp.distance(fc) for fc in final_centroids) <= match_dist:
            matched += 1
    recall = (matched / expected_count) if expected_count > 0 else 0.0

    fp = 0
    for fc in final_centroids:
        if not expected_centroids:
            fp += 1
            continue
        if min(fc.distance(exp) for exp in expected_centroids) > fp_dist:
            fp += 1

    drift_ratio = 0.0
    if not trace_df.empty and "prop_drift_flag" in trace_df.columns:
        drift_count = int((trace_df["prop_drift_flag"].fillna(0).astype(int) == 1).sum())
        attempted = int((trace_df.get("stage2_cluster_id", "") != "").sum()) if "stage2_cluster_id" in trace_df.columns else len(trace_df)
        drift_ratio = float(drift_count / attempted) if attempted > 0 else 0.0

    stability = 0.0
    if not cluster_df.empty:
        finals = cluster_df[cluster_df.get("final_pass", 0).astype(int) == 1]
        if not finals.empty:
            stability = float(finals.get("frames_hit_support", 0).mean() or 0.0)
            stability = min(stability / 20.0, 1.0)

    scoring = cfg.get("autotune", {}).get("scoring", {})
    score = (
        float(scoring.get("w_recall", 1.0)) * recall
        - float(scoring.get("w_fp", 0.8)) * fp
        - float(scoring.get("w_drift", 0.5)) * drift_ratio
        + float(scoring.get("w_stability", 0.3)) * stability
    )

    return {
        "expected_count": expected_count,
        "final_count": int(len(final_centroids)),
        "recall_proxy": recall,
        "fp_count": fp,
        "drift_ratio": drift_ratio,
        "stability": stability,
        "score": score,
    }


def _stop_condition(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    stop_cfg = cfg.get("autotune", {}).get("stop", {})
    return (
        metrics.get("recall_proxy", 0.0) >= float(stop_cfg.get("recall_min", 0.8))
        and metrics.get("fp_count", 999) <= int(stop_cfg.get("max_fp", 1))
        and metrics.get("final_count", 0) >= int(stop_cfg.get("min_final_count", 2))
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_stage2_autotune.yaml")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    base_cfg_path = Path(str(cfg.get("base_config") or ""))
    base_cfg = _load_yaml(base_cfg_path) if base_cfg_path.exists() else {}
    run_overrides = cfg.get("run_overrides", {}) if isinstance(cfg.get("run_overrides"), dict) else {}
    autotune_cfg = cfg.get("autotune", {}) if isinstance(cfg.get("autotune"), dict) else {}

    drive_id = str(run_overrides.get("drive_id") or base_cfg.get("drive_id") or "unknown")
    tag = drive_id.split("_")[-2] if "_" in drive_id else drive_id
    out_root = Path(args.out_dir) if args.out_dir else Path("runs") / f"crosswalk_autotune_{tag}_250_500_{dt.datetime.now():%Y%m%d_%H%M%S}"
    out_root.mkdir(parents=True, exist_ok=True)

    param_space = autotune_cfg.get("param_space", {})
    keys = sorted(param_space.keys())
    values = [param_space[k] for k in keys]
    combos: List[Tuple[Any, ...]] = []
    if keys:
        from itertools import product

        combos = list(product(*values))
    max_trials = int(autotune_cfg.get("max_trials", 12))
    rng = random.Random(int(autotune_cfg.get("seed", 42)))
    rng.shuffle(combos)
    combos = combos[:max_trials] if combos else [()]

    leaderboard = []
    best_trial = None
    best_score = None

    trials_dir = out_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    for idx, combo in enumerate(combos):
        trial_id = f"trial_{idx:03d}"
        trial_dir = trials_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_cfg = deepcopy(base_cfg)
        trial_cfg.update(run_overrides)
        params = _flatten_params(keys, list(combo))
        for path, value in params.items():
            _set_nested(trial_cfg, path, value)
        trial_cfg_path = trial_dir / "trial_config.yaml"
        trial_cfg_path.write_text(yaml.safe_dump(trial_cfg, sort_keys=False), encoding="utf-8")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_crosswalk_stage2_full.py"),
            "--config",
            str(trial_cfg_path),
            "--out-run",
            str(trial_dir),
        ]
        subprocess.run(cmd, check=False)

        outputs_dir = trial_dir / "outputs"
        metrics = _score_trial(outputs_dir, cfg)
        summary = {
            "trial_id": trial_id,
            "params": params,
            "metrics": metrics,
        }
        (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        leaderboard.append(
            {
                "trial_id": trial_id,
                "score": metrics.get("score", 0.0),
                "final_count": metrics.get("final_count", 0),
                "expected_count": metrics.get("expected_count", 0),
                "recall_proxy": metrics.get("recall_proxy", 0.0),
                "fp_count": metrics.get("fp_count", 0),
                "drift_ratio": metrics.get("drift_ratio", 0.0),
                "stability": metrics.get("stability", 0.0),
            }
        )
        if best_score is None or metrics.get("score", 0.0) > best_score:
            best_score = metrics.get("score", 0.0)
            best_trial = trial_id
        if _stop_condition(metrics, cfg):
            break

    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv(out_root / "leaderboard.csv", index=False)

    report_lines = [
        "# Autotune Report",
        "",
        f"- total_trials: {len(leaderboard)}",
        f"- best_trial: {best_trial}",
        f"- best_score: {best_score}",
    ]
    if best_trial:
        best_dir = trials_dir / best_trial / "outputs"
        best_out = out_root / "best" / "outputs"
        if best_out.exists():
            shutil.rmtree(best_out)
        if best_dir.exists():
            best_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(best_dir, best_out)
        report_lines.append(f"- best_outputs: {best_out}")
    (out_root / "autotune_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
