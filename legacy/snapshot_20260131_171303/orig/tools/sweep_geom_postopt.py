from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _ensure_hf_cache_env() -> Dict[str, str]:
    base = (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
        or os.environ.get("HF_DATASETS_CACHE")
    )
    if base:
        base_path = Path(base)
    else:
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            base_path = Path(r"E:\hf_cache_shared")
        else:
            base_path = Path(r"E:\hf_cache_shared")
    base_path = base_path.resolve()
    defaults = {
        "HF_HOME": str(base_path),
        "HUGGINGFACE_HUB_CACHE": str(base_path / "hub"),
        "TRANSFORMERS_CACHE": str(base_path / "transformers"),
        "HF_DATASETS_CACHE": str(base_path / "datasets"),
        "HF_HUB_DISABLE_SYMLINKS": "1",
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
    for path in (defaults["HUGGINGFACE_HUB_CACHE"], defaults["TRANSFORMERS_CACHE"], defaults["HF_DATASETS_CACHE"]):
        Path(path).mkdir(parents=True, exist_ok=True)
    return defaults


def _check_cache_writable(path: Path) -> Tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / "write_test.tmp"
        test_file.write_text("ok", encoding="ascii")
        test_file.unlink()
        return True, ""
    except Exception as exc:
        return False, str(exc)


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


def _latest_sweep_dir() -> Path | None:
    runs = Path("runs")
    if not runs.exists():
        return None
    dirs = [p for p in runs.iterdir() if p.is_dir() and p.name.startswith("sweep_geom_postopt_")]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime)
    return dirs[-1]


def _write_index(path: Path, entry: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _read_index(path: Path, stage: str | None, candidate: str | None) -> List[Dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if stage and entry.get("stage") != stage:
            continue
        if candidate and entry.get("candidate_id") != candidate:
            continue
        items.append(entry)
    return items


def _run_cmd(cmd: List[str], env: Dict[str, str]) -> int:
    return subprocess.run(cmd, env=env).returncode


def _run_build_geom(
    candidate: Dict[str, Any],
    drive: str,
    max_frames: int,
    nn_best_cfg: str,
) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["GEOM_BACKEND"] = "nn"
    env["GEOM_NN_FIXED"] = "1"
    env["GEOM_NN_BEST_CFG"] = nn_best_cfg
    for key, value in (candidate.get("post") or {}).items():
        env[str(key)] = str(value)

    args = [
        "cmd",
        "/c",
        "scripts\\build_geom.cmd",
        "--drive",
        drive,
        "--max-frames",
        str(max_frames),
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
    candidate: str | None = None,
) -> Path:
    index_path = out_dir / "postopt_index.jsonl"
    out_csv = out_dir / f"{stage}_report.csv"
    out_md = out_dir / f"{stage}_report.md"
    out_json = out_dir / f"{stage}_ranking.json"
    cmd = [
        ".venv\\Scripts\\python.exe",
        "tools\\score_geom_postopt.py",
        "--index",
        str(index_path),
        "--baseline",
        baseline,
        "--stage",
        stage,
        "--candidate",
        candidate or "",
        "--out-csv",
        str(out_csv),
        "--out-md",
        str(out_md),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=False)
    return out_json


def _select_topk(ranking_path: Path, topk: int) -> List[str]:
    if not ranking_path.exists():
        return []
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ordered = [r.get("candidate_id") for r in ranking if r.get("candidate_id")]
    return ordered[:topk]


def _write_topk(out_dir: Path, topk_ids: List[str]) -> None:
    (out_dir / "topk_candidates.json").write_text(
        json.dumps(topk_ids, indent=2),
        encoding="utf-8",
    )


def _load_topk(out_dir: Path, topk: int) -> List[str]:
    topk_path = out_dir / "topk_candidates.json"
    if topk_path.exists():
        data = json.loads(topk_path.read_text(encoding="utf-8"))
        return [c for c in data if c]
    ranking_path = out_dir / "quick_ranking.json"
    if ranking_path.exists():
        return _select_topk(ranking_path, topk)
    return []


def _write_runcard(
    out_dir: Path,
    commit: str,
    nn_best_cfg: str,
    candidates: List[Dict[str, Any]],
    quick_drives: List[str],
    full_drives: List[str],
    quick_frames: int,
    full_frames: int,
    topk: int,
    baseline: str,
) -> None:
    lines = [
        "# RunCard",
        "",
        f"- commit: {commit}",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- nn_best_cfg: {nn_best_cfg}",
        f"- quick_drives: {', '.join(quick_drives)}",
        f"- full_drives: {', '.join(full_drives)}",
        f"- quick_max_frames: {quick_frames}",
        f"- full_max_frames: {full_frames}",
        f"- topk: {topk}",
        f"- baseline: {baseline}",
        f"- OSM_ROOT: {os.environ.get('OSM_ROOT', '')}",
        f"- HF_HOME: {os.environ.get('HF_HOME', '')}",
        f"- HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', '')}",
        f"- TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '')}",
        f"- HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', '')}",
        "",
        "## Candidates",
        "",
    ]
    for cand in candidates:
        lines.append(f"- {cand.get('candidate_id')}")
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "- quick: scripts\\sweep_geom_postopt.cmd",
            "- full by rank: scripts\\sweep_geom_postopt.cmd --full-only --full-rank <1|2|3> --resume --out-dir <runs\\sweep_geom_postopt_...>",
            "- finalize: scripts\\sweep_geom_postopt.cmd --finalize --resume --out-dir <runs\\sweep_geom_postopt_...>",
        ]
    )
    (out_dir / "RunCard.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/geom_postopt_candidates.yaml")
    ap.add_argument("--drives-file", default="configs/golden_drives.txt")
    ap.add_argument("--quick-max-frames", type=int, default=400)
    ap.add_argument("--full-max-frames", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--baseline", default="configs/geom_regress_baseline.yaml")
    ap.add_argument("--full-only", action="store_true", help="skip quick and only run full")
    ap.add_argument("--quick-only", action="store_true", help="run quick only and stop before full")
    ap.add_argument("--full-candidate", default="", help="run a specific candidate_id in full")
    ap.add_argument("--full-rank", type=int, default=0, help="run candidate by topk rank (1-based)")
    ap.add_argument("--resume", action="store_true", help="reuse existing index and skip completed candidates")
    ap.add_argument("--finalize", action="store_true", help="finalize full report + best_postopt + merge")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    candidates = cfg.get("candidates", []) or []
    nn_best_cfg = cfg.get("nn_best_cfg", "configs/geom_nn_best.yaml")

    drives = _read_drives(Path(args.drives_file))
    if not drives:
        raise SystemExit("ERROR: no drives found in drives-file")

    quick_drives = ["2013_05_28_drive_0007_sync", "2013_05_28_drive_0004_sync"]
    if os.environ.get("QUICK_DRIVES"):
        quick_drives = [d.strip() for d in os.environ["QUICK_DRIVES"].split(",") if d.strip()]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.resume or args.full_only or args.finalize:
        out_dir = _latest_sweep_dir() or Path("runs") / f"sweep_geom_postopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"sweep_geom_postopt_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "postopt_index.jsonl"
    if index_path.exists() and not args.resume and not args.full_only and not args.finalize:
        index_path.unlink()

    hf_env = _ensure_hf_cache_env()
    ok, err = _check_cache_writable(Path(hf_env["HF_HOME"]))
    if not ok:
        raise SystemExit(
            "ERROR: HF cache not writable: {path} ({err}). "
            "Please add the cache path and .venv\\Scripts\\python.exe to Windows Defender exclusions."
            .format(path=hf_env["HF_HOME"], err=err)
        )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    done_quick: set[tuple[str, str]] = set()
    if args.resume and index_path.exists():
        for entry in _read_index(index_path, "quick", None):
            if entry.get("status") == "PASS":
                done_quick.add((str(entry.get("candidate_id") or ""), str(entry.get("drive") or "")))

    if not args.full_only and not args.finalize:
        for cand in candidates:
            cand_id = cand.get("candidate_id") or "unknown"
            for drive in quick_drives:
                if (cand_id, drive) in done_quick:
                    continue
                start = time.time()
                rc, run_id, out_dir_geom = _run_build_geom(cand, drive, args.quick_max_frames, nn_best_cfg)
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
                        "stage": "quick",
                        "candidate_id": cand_id,
                        "drive": drive,
                        "status": status,
                        "reason": reason or None,
                        "geom_run_id": run_id or None,
                        "outputs_dir": out_dir_geom or None,
                        "timing_sec": round(time.time() - start, 2),
                    },
                )

        quick_rank = _score_stage(out_dir, "quick", args.baseline)
        topk_ids = _select_topk(quick_rank, args.topk)
        if not topk_ids:
            topk_ids = [c.get("candidate_id") for c in candidates if c.get("candidate_id")][: args.topk]
        _write_topk(out_dir, topk_ids)
        if args.quick_only:
            _write_runcard(
                out_dir,
                commit,
                nn_best_cfg,
                candidates,
                quick_drives,
                drives,
                args.quick_max_frames,
                args.full_max_frames,
                args.topk,
                args.baseline,
            )
            return 0
    else:
        topk_ids = _load_topk(out_dir, args.topk)

    full_target_ids = topk_ids
    if args.full_candidate:
        full_target_ids = [args.full_candidate]
    elif args.full_rank:
        if not topk_ids:
            raise SystemExit("ERROR: topk list not found; run quick first or use --full-candidate")
        rank_idx = max(1, int(args.full_rank)) - 1
        if rank_idx >= len(topk_ids):
            raise SystemExit(f"ERROR: full-rank {args.full_rank} out of range")
        full_target_ids = [topk_ids[rank_idx]]

    if args.finalize:
        full_target_ids = []

    if full_target_ids:
        topk_candidates = [c for c in candidates if c.get("candidate_id") in set(full_target_ids)]
        for cand in topk_candidates:
            cand_id = cand.get("candidate_id") or "unknown"
            if args.resume:
                existing = [
                    e for e in _read_index(index_path, "full", cand_id)
                    if e.get("status") == "PASS" and e.get("outputs_dir")
                ]
                if len(existing) >= len(drives):
                    continue
            for drive in drives:
                start = time.time()
                rc, run_id, out_dir_geom = _run_build_geom(cand, drive, args.full_max_frames, nn_best_cfg)
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
                        "timing_sec": round(time.time() - start, 2),
                    },
                )

            _score_stage(out_dir, "full", args.baseline, candidate=cand_id)
            (out_dir / f"full_candidate_{cand_id}.csv").write_text(
                (out_dir / "full_report.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (out_dir / f"full_candidate_{cand_id}.md").write_text(
                (out_dir / "full_report.md").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    if args.finalize:
        if not topk_ids:
            topk_ids = _load_topk(out_dir, args.topk)
        if not topk_ids:
            raise SystemExit("ERROR: topk list not found; run quick first")
        for cid in topk_ids:
            full_items = _read_index(index_path, "full", cid)
            pass_items = [e for e in full_items if e.get("status") == "PASS"]
            if len(pass_items) < len(drives):
                raise SystemExit(f"ERROR: full results incomplete for {cid} ({len(pass_items)}/{len(drives)})")
        full_rank = _score_stage(out_dir, "full", args.baseline)
        best_candidate = ""
        if full_rank.exists():
            ranking = json.loads(full_rank.read_text(encoding="utf-8"))
            for item in ranking:
                if item.get("status") == "PASS":
                    best_candidate = item.get("candidate_id") or ""
                    break

        if best_candidate:
            for cand in candidates:
                if cand.get("candidate_id") == best_candidate:
                    best_postopt = {"candidate_id": best_candidate, **(cand.get("post") or {})}
                    (out_dir / "best_postopt.yaml").write_text(
                        yaml.safe_dump(best_postopt, sort_keys=False), encoding="utf-8"
                    )
                    break

            merge_out = out_dir / "merged"
            cmd = [
                ".venv\\Scripts\\python.exe",
                "tools\\merge_geom_outputs.py",
                "--index",
                str(index_path),
                "--best-postopt",
                str(out_dir / "best_postopt.yaml"),
                "--out-dir",
                str(merge_out),
            ]
            subprocess.run(cmd, check=False)

    _write_runcard(
        out_dir,
        commit,
        nn_best_cfg,
        candidates,
        quick_drives,
        drives,
        args.quick_max_frames,
        args.full_max_frames,
        args.topk,
        args.baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
