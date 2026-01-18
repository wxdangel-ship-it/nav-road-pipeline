from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _read_drives(path: Path) -> List[str]:
    drives = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        drives.append(line)
    return drives


def _latest_run(prefix: str) -> Path | None:
    runs = Path("runs")
    if not runs.exists():
        return None
    dirs = [p for p in runs.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime)
    return dirs[-1]


def _write_index(path: Path, entry: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _run_cmd(cmd: List[str], env: Dict[str, str]) -> int:
    return subprocess.run(cmd, env=env).returncode


def _ensure_hf_cache_env(out_dir: Path, candidate_id: str, use_shared: bool) -> Dict[str, str]:
    if use_shared:
        base = (
            os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("TRANSFORMERS_CACHE")
            or os.environ.get("HF_DATASETS_CACHE")
        )
        if base:
            base_path = Path(base)
        else:
            base_path = Path("cache") / "hf_shared"
    else:
        base_path = out_dir / "hf_cache" / candidate_id
    base_path = base_path.resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    defaults = {
        "HF_HOME": str(base_path),
        "HUGGINGFACE_HUB_CACHE": str(base_path / "hub"),
        "TRANSFORMERS_CACHE": str(base_path / "transformers"),
        "HF_DATASETS_CACHE": str(base_path / "datasets"),
        "HF_HUB_DISABLE_SYMLINKS": "1",
    }
    for path in defaults.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return defaults


def _run_build_geom(
    candidate: Dict[str, Any],
    drive: str,
    max_frames: int,
    allow_empty_intersections: bool,
    hf_env: Dict[str, str],
) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["GEOM_BACKEND"] = "nn"
    env.update(hf_env)
    drivable = candidate.get("postprocess_params", {}).get("drivable_classes", ["road", "sidewalk"])
    args = [
        "cmd",
        "/c",
        "scripts\\build_geom.cmd",
        "--drive",
        drive,
        "--max-frames",
        str(max_frames),
        "--allow-empty-intersections",
        "1" if allow_empty_intersections else "0",
        "--nn-model-family",
        str(candidate.get("model_family") or ""),
        "--nn-model",
        str(candidate.get("model_id") or ""),
        "--nn-camera",
        str(candidate.get("camera") or "image_00"),
        "--nn-stride",
        str(int(candidate.get("stride") or 5)),
        "--nn-mask-threshold",
        str(float(candidate.get("mask_threshold") or 0.5)),
        "--nn-drivable-classes",
        ",".join(drivable),
    ]
    rc = _run_cmd(args, env=env)
    if rc != 0:
        return rc, "", ""
    run_dir = _latest_run("geom_")
    if not run_dir:
        return 1, "", ""
    return 0, run_dir.name, str(run_dir / "outputs")


def _run_osm_ref(outputs_dir: str) -> int:
    cmd = [
        "cmd",
        "/c",
        ".venv\\Scripts\\python.exe",
        "tools\\osm_ref_extract.py",
        "--outputs-dir",
        outputs_dir,
    ]
    return _run_cmd(cmd, env=os.environ.copy())


def _score_stage(
    out_dir: Path,
    stage: str,
    baseline: str,
    weight_match: float,
    weight_p95: float,
) -> Path:
    index_path = out_dir / "sweep_index.jsonl"
    out_csv = out_dir / f"{stage}_report.csv"
    out_md = out_dir / f"{stage}_report.md"
    out_json = out_dir / f"{stage}_ranking.json"
    cmd = [
        ".venv\\Scripts\\python.exe",
        "tools\\score_geom_roadseg.py",
        "--index",
        str(index_path),
        "--baseline",
        baseline,
        "--stage",
        stage,
        "--out-csv",
        str(out_csv),
        "--out-md",
        str(out_md),
        "--out-json",
        str(out_json),
        "--weight-match",
        str(weight_match),
        "--weight-dist-p95",
        str(weight_p95),
    ]
    subprocess.run(cmd, check=False)
    return out_json


def _write_runcard(
    out_dir: Path,
    commit: str,
    drives: List[str],
    candidates: List[Dict[str, Any]],
    quick_frames: int,
    full_frames: int,
    topk: int,
    weight_match: float,
    weight_p95: float,
    best_config: str,
    hf_env: Dict[str, str],
    use_shared_cache: bool,
    hf_cache_root: str,
) -> None:
    lines = [
        "# RunCard",
        "",
        f"- commit: {commit}",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- drives: {', '.join(drives)}",
        f"- quick_max_frames: {quick_frames}",
        f"- full_max_frames: {full_frames}",
        f"- topk: {topk}",
        f"- weights: osm_match_ratio={weight_match}, dist_p95_m={weight_p95}",
        f"- best_config: {best_config}",
        "",
        "## HF Cache",
        "",
        f"- use_shared_cache: {use_shared_cache}",
        f"- cache_root: {hf_cache_root}",
        f"- HF_HOME: {hf_env.get('HF_HOME')}",
        f"- HUGGINGFACE_HUB_CACHE: {hf_env.get('HUGGINGFACE_HUB_CACHE')}",
        f"- TRANSFORMERS_CACHE: {hf_env.get('TRANSFORMERS_CACHE')}",
        f"- HF_DATASETS_CACHE: {hf_env.get('HF_DATASETS_CACHE')}",
        f"- HF_HUB_DISABLE_SYMLINKS: {hf_env.get('HF_HUB_DISABLE_SYMLINKS')}",
        "",
        "## Candidates",
        "",
    ]
    for cand in candidates:
        lines.append(f"- {cand.get('candidate_id')} ({cand.get('model_family')}:{cand.get('model_id')})")
    (out_dir / "RunCard.md").write_text("\n".join(lines), encoding="utf-8")


def _select_topk(candidates: List[Dict[str, Any]], ranking_path: Path, topk: int) -> List[str]:
    if ranking_path.exists():
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        ordered = [r.get("candidate_id") for r in ranking if r.get("candidate_id")]
        ordered = [c for c in ordered if c]
        if ordered:
            return ordered[:topk]
    fallback = [c.get("candidate_id") for c in candidates if c.get("candidate_id")]
    return fallback[:topk]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_zoo.yaml")
    ap.add_argument("--drives-file", default="configs/golden_drives.txt")
    ap.add_argument("--quick-max-frames", type=int, default=400)
    ap.add_argument("--full-max-frames", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--weight-match", type=float, default=1.0)
    ap.add_argument("--weight-dist-p95", type=float, default=-0.05)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--baseline", default="configs/geom_regress_baseline.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    candidates = cfg.get("candidates", []) or []
    drives = _read_drives(Path(args.drives_file))
    if not drives:
        raise SystemExit("ERROR: no drives found in drives-file")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"sweep_geom_roadseg_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    use_shared_cache = os.environ.get("USE_SHARED_HF_CACHE", "").strip() == "1"
    hf_env = _ensure_hf_cache_env(out_dir, "run", use_shared_cache)
    index_path = out_dir / "sweep_index.jsonl"
    if index_path.exists():
        index_path.unlink()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    for stage, max_frames, subset in [
        ("quick", args.quick_max_frames, candidates),
    ]:
        for cand in subset:
            cand_id = cand.get("candidate_id") or "unknown"
            if cand.get("status") == "not_implemented" or cand.get("implemented") is False:
                for drive in drives:
                    _write_index(
                        index_path,
                        {
                            "stage": stage,
                            "candidate_id": cand_id,
                            "drive": drive,
                            "status": "SKIPPED",
                            "reason": "not_implemented",
                            "geom_run_id": None,
                            "outputs_dir": None,
                            "model_family": cand.get("model_family"),
                            "model_id": cand.get("model_id"),
                            "timing_sec": 0.0,
                        },
                    )
                continue

            for drive in drives:
                start = time.time()
                cand_id = cand.get("candidate_id") or "unknown"
                hf_env = _ensure_hf_cache_env(out_dir, cand_id, use_shared_cache)
                rc, run_id, out_dir_geom = _run_build_geom(
                    cand, drive, max_frames, allow_empty_intersections=True, hf_env=hf_env
                )
                status = "PASS" if rc == 0 else "FAIL"
                reason = "" if rc == 0 else "build_geom_failed"
                if rc == 0:
                    osm_rc = _run_osm_ref(out_dir_geom)
                    if osm_rc != 0:
                        status = "FAIL"
                        reason = "osm_ref_failed"
                _write_index(
                    index_path,
                    {
                        "stage": stage,
                        "candidate_id": cand_id,
                        "drive": drive,
                        "status": status,
                        "reason": reason or None,
                        "geom_run_id": run_id or None,
                        "outputs_dir": out_dir_geom or None,
                        "model_family": cand.get("model_family"),
                        "model_id": cand.get("model_id"),
                        "timing_sec": round(time.time() - start, 2),
                    },
                )

        stage1_rank = _score_stage(
            out_dir, "quick", args.baseline, args.weight_match, args.weight_dist_p95
        )
        topk_ids = _select_topk(candidates, stage1_rank, args.topk)

    best_config = ""
    if args.full_max_frames > 0:
        topk_candidates = [c for c in candidates if c.get("candidate_id") in set(topk_ids)]
        for cand in topk_candidates:
            cand_id = cand.get("candidate_id") or "unknown"
            if cand.get("status") == "not_implemented" or cand.get("implemented") is False:
                for drive in drives:
                    _write_index(
                        index_path,
                        {
                            "stage": "full",
                            "candidate_id": cand_id,
                            "drive": drive,
                            "status": "SKIPPED",
                            "reason": "not_implemented",
                            "geom_run_id": None,
                            "outputs_dir": None,
                            "model_family": cand.get("model_family"),
                            "model_id": cand.get("model_id"),
                            "timing_sec": 0.0,
                        },
                    )
                continue

            for drive in drives:
                start = time.time()
                hf_env = _ensure_hf_cache_env(out_dir, cand_id, use_shared_cache)
                rc, run_id, out_dir_geom = _run_build_geom(
                    cand, drive, args.full_max_frames, allow_empty_intersections=False, hf_env=hf_env
                )
                status = "PASS" if rc == 0 else "FAIL"
                reason = "" if rc == 0 else "build_geom_failed"
                if rc == 0:
                    osm_rc = _run_osm_ref(out_dir_geom)
                    if osm_rc != 0:
                        status = "FAIL"
                        reason = "osm_ref_failed"
                _write_index(
                    index_path,
                    {
                        "stage": "full",
                        "candidate_id": cand_id,
                        "drive": drive,
                        "status": status,
                        "reason": reason or None,
                        "geom_run_id": run_id or None,
                        "outputs_dir": out_dir_geom or None,
                        "model_family": cand.get("model_family"),
                        "model_id": cand.get("model_id"),
                        "timing_sec": round(time.time() - start, 2),
                    },
                )

        stage2_rank = _score_stage(
            out_dir, "full", args.baseline, args.weight_match, args.weight_dist_p95
        )
        if stage2_rank.exists():
            ranking = json.loads(stage2_rank.read_text(encoding="utf-8"))
            if ranking:
                best_id = ranking[0].get("candidate_id")
                for cand in candidates:
                    if cand.get("candidate_id") == best_id:
                        best_path = out_dir / "best_config.yaml"
                        best_path.write_text(yaml.safe_dump(cand, sort_keys=False), encoding="utf-8")
                        best_config = str(best_path)
                        break
    if not best_config and candidates:
        best_path = out_dir / "best_config.yaml"
        best_path.write_text(yaml.safe_dump(candidates[0], sort_keys=False), encoding="utf-8")
        best_config = str(best_path)

    _write_runcard(
        out_dir,
        commit,
        drives,
        candidates,
        args.quick_max_frames,
        args.full_max_frames,
        args.topk,
        args.weight_match,
        args.weight_dist_p95,
        best_config,
        hf_env,
        use_shared_cache,
        str((out_dir / "hf_cache").resolve()) if not use_shared_cache else hf_env.get("HF_HOME", ""),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
