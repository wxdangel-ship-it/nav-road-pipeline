from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _read_drives(path: Path) -> List[str]:
    drives = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        drives.append(line)
    return drives


def _latest_run(prefix: str) -> Optional[Path]:
    runs = Path("runs")
    if not runs.exists():
        return None
    dirs = [p for p in runs.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime)
    return dirs[-1]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


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
            local = os.environ.get("LOCALAPPDATA", "")
            if local:
                base_path = Path(local) / "hf_cache_shared"
            else:
                base_path = out_dir / "hf_cache_shared"
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


def _run_cmd(cmd: List[str], env: Dict[str, str]) -> int:
    return subprocess.run(cmd, env=env).returncode


def _run_build_geom(
    candidate: Dict[str, Any],
    drive: str,
    max_frames: int,
    hf_env: Dict[str, str],
) -> Tuple[int, str, str]:
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


def _run_osm_ref(outputs_dir: str, env: Dict[str, str]) -> int:
    cmd = [
        "cmd",
        "/c",
        ".venv\\Scripts\\python.exe",
        "tools\\osm_ref_extract.py",
        "--outputs-dir",
        outputs_dir,
    ]
    return _run_cmd(cmd, env=env)


def _write_index(path: Path, entry: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _load_candidates(cfg_path: Path) -> Dict[str, Dict[str, Any]]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    candidates = cfg.get("candidates", []) or []
    by_id = {}
    for cand in candidates:
        cid = cand.get("candidate_id")
        if cid:
            by_id[str(cid)] = cand
    return by_id


def _select_candidates(
    primary_ids: List[str],
    compare_ids: List[str],
    candidates: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    primary_ids = [cid for cid in primary_ids if cid]
    compare_ids = [cid for cid in compare_ids if cid]
    selected = []
    for cid in primary_ids + compare_ids:
        cand = candidates.get(cid)
        if cand:
            cand = dict(cand)
            cand["compare_only"] = cid in compare_ids
            selected.append(cand)
    return selected, primary_ids, compare_ids


def _load_quick_ranking(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in data.get("drives", []) or []:
        drive = item.get("drive")
        if drive:
            out[str(drive)] = item
    return out


def _score_drive(
    outputs_dir: Path,
    drive: str,
    baseline: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[float], Dict[str, Any]]:
    summary = _read_json(outputs_dir / "GeomSummary.json")
    if not summary:
        return "FAIL", None, {"reason": "missing_summary"}
    if summary.get("status") != "PASS":
        return "FAIL", None, {"reason": summary.get("reason") or "summary_fail"}

    base = baseline.get(drive) or {}
    base_ratio = base.get("centerlines_in_polygon_ratio")
    ratio = summary.get("centerlines_in_polygon_ratio")
    if isinstance(base_ratio, (int, float)) and isinstance(ratio, (int, float)):
        if ratio < float(base_ratio) - 0.02:
            return "FAIL", None, {"reason": "centerlines_ratio_drop"}

    if summary.get("road_component_count_after") != 1:
        return "FAIL", None, {"reason": "road_components_not_1"}

    osm = _read_json(outputs_dir / "osm_ref_metrics.json")
    if not osm:
        return "FAIL", None, {"reason": "missing_osm_metrics"}
    if not osm.get("osm_present"):
        return "FAIL", None, {"reason": "osm_not_present"}

    metrics_valid = bool(osm.get("metrics_valid"))
    match_ratio = osm.get("match_ratio") if metrics_valid else None
    dist_p95 = osm.get("dist_p95_m") if metrics_valid else None
    score = None
    if match_ratio is not None and dist_p95 is not None:
        score = float(match_ratio) * 1.0 + float(dist_p95) * -0.05
    return "PASS", score, {
        "metrics_valid": metrics_valid,
        "match_ratio": match_ratio,
        "dist_p95_m": dist_p95,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_md(path: Path, candidates: List[Dict[str, Any]], drives: List[Dict[str, Any]]) -> None:
    counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for row in candidates:
        status = row.get("status")
        if status in counts:
            counts[status] += 1

    lines = [
        "# Geom RoadSeg Full Report",
        "",
        f"- total_candidates: {len(candidates)}",
        f"- pass: {counts['PASS']}",
        f"- fail: {counts['FAIL']}",
        f"- skipped: {counts['SKIPPED']}",
        "",
        "## Candidates",
        "",
        "| candidate_id | status | avg_score | avg_timing_sec | pass | fail | skipped | compare_only | reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in candidates:
        lines.append(
            "| {cid} | {status} | {score} | {time} | {p} | {f} | {s} | {co} | {r} |".format(
                cid=row.get("candidate_id") or "",
                status=row.get("status") or "",
                score="" if row.get("avg_score") is None else f"{row.get('avg_score'):.4f}",
                time="" if row.get("avg_timing_sec") is None else f"{row.get('avg_timing_sec'):.2f}",
                p=row.get("pass_count") or 0,
                f=row.get("fail_count") or 0,
                s=row.get("skipped_count") or 0,
                co=row.get("compare_only") or False,
                r=row.get("reasons") or "",
            )
        )
    lines.append("")
    lines.append("## Per-Drive")
    lines.append("")
    lines.append("| candidate_id | drive | status | score | metrics_valid | reason |")
    lines.append("| --- | --- | --- | ---: | --- | --- |")
    for row in drives:
        score = row.get("score")
        lines.append(
            "| {cid} | {drive} | {status} | {score} | {mv} | {reason} |".format(
                cid=row.get("candidate_id") or "",
                drive=row.get("drive") or "",
                status=row.get("status") or "",
                score="" if score is None else f"{score:.4f}",
                mv=row.get("metrics_valid") if row.get("metrics_valid") is not None else "",
                reason=row.get("reason") or "",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _score_candidates(
    entries: List[Dict[str, Any]],
    drives: List[str],
    baseline: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        cid = entry.get("candidate_id") or "unknown"
        by_candidate.setdefault(cid, []).append(entry)

    candidate_rows: List[Dict[str, Any]] = []
    drive_rows: List[Dict[str, Any]] = []
    for cid, items in sorted(by_candidate.items()):
        pass_count = fail_count = skipped_count = 0
        scores: List[float] = []
        timing: List[float] = []
        reasons: List[str] = []
        compare_only = False
        for entry in items:
            compare_only = bool(entry.get("compare_only"))
            status = entry.get("status")
            if status != "PASS":
                fail_count += 1
                reasons.append(entry.get("reason") or "build_geom_failed")
                drive_rows.append(
                    {
                        "candidate_id": cid,
                        "drive": entry.get("drive"),
                        "status": "FAIL",
                        "score": None,
                        "metrics_valid": None,
                        "reason": entry.get("reason") or "build_geom_failed",
                    }
                )
                continue

            outputs_dir = entry.get("outputs_dir")
            if not outputs_dir:
                fail_count += 1
                reasons.append("missing_outputs_dir")
                drive_rows.append(
                    {
                        "candidate_id": cid,
                        "drive": entry.get("drive"),
                        "status": "FAIL",
                        "score": None,
                        "metrics_valid": None,
                        "reason": "missing_outputs_dir",
                    }
                )
                continue

            drive = entry.get("drive") or ""
            status2, score, meta = _score_drive(Path(outputs_dir), drive, baseline)
            if status2 == "PASS":
                pass_count += 1
                if score is not None:
                    scores.append(score)
                timing.append(float(entry.get("timing_sec") or 0.0))
            else:
                fail_count += 1
                reasons.append(meta.get("reason") or "qc_failed")

            drive_rows.append(
                {
                    "candidate_id": cid,
                    "drive": drive,
                    "status": status2,
                    "score": score,
                    "metrics_valid": meta.get("metrics_valid"),
                    "match_ratio": meta.get("match_ratio"),
                    "dist_p95_m": meta.get("dist_p95_m"),
                    "reason": meta.get("reason") or "",
                    "outputs_dir": entry.get("outputs_dir"),
                }
            )

        if fail_count > 0:
            status = "FAIL"
        elif pass_count == 0:
            status = "SKIPPED"
        else:
            status = "PASS"

        avg_score = sum(scores) / len(scores) if scores else None
        avg_time = sum(timing) / len(timing) if timing else None
        candidate_rows.append(
            {
                "candidate_id": cid,
                "status": status,
                "avg_score": avg_score,
                "avg_timing_sec": avg_time,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "skipped_count": skipped_count,
                "compare_only": compare_only,
                "reasons": ";".join(sorted(set(reasons))),
            }
        )
    return candidate_rows, drive_rows


def _write_runcard(
    out_dir: Path,
    commit: str,
    quick_dir: Path,
    drives: List[str],
    candidates: List[Dict[str, Any]],
    primary_ids: List[str],
    compare_ids: List[str],
    max_frames: int,
    baseline: str,
    osm_root: str,
    use_shared_cache: bool,
    hf_env: Dict[str, str],
    cache_root: str,
) -> None:
    quick_note = str(quick_dir) if str(quick_dir) else "manual_override"
    lines = [
        "# RunCard",
        "",
        f"- commit: {commit}",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- quick_dir: {quick_note}",
        f"- drives: {', '.join(drives)}",
        f"- full_max_frames: {max_frames}",
        f"- baseline: {baseline}",
        f"- osm_root: {osm_root}",
        "",
        "## Candidate Selection",
        "",
        "- rule: manual override (primary/compare specified)",
        f"- primary: {', '.join(primary_ids) if primary_ids else 'none'}",
        f"- compare_only: {', '.join(compare_ids) if compare_ids else 'none'}",
        "",
        "## HF Cache",
        "",
        f"- use_shared_cache: {use_shared_cache}",
        f"- cache_root: {cache_root}",
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
        lines.append(
            "- {cid} ({family}:{mid}) compare_only={co}".format(
                cid=cand.get("candidate_id"),
                family=cand.get("model_family"),
                mid=cand.get("model_id"),
                co=cand.get("compare_only", False),
            )
        )
    (out_dir / "RunCard.md").write_text("\n".join(lines), encoding="utf-8")


def _write_best_summary(
    out_dir: Path,
    winner: Optional[Dict[str, Any]],
    drives: List[Dict[str, Any]],
    baseline_path: str,
    baseline: Dict[str, Dict[str, Any]],
) -> None:
    lines = [
        "# Best Summary",
        "",
        f"- baseline: {baseline_path}",
    ]
    if winner:
        lines.extend(
            [
                f"- winner: {winner.get('candidate_id')}",
                f"- avg_score: {winner.get('avg_score')}",
                f"- avg_timing_sec: {winner.get('avg_timing_sec')}",
                "- reason: hard gates pass on 8/8 drives, highest avg_score, tie-breaker by timing",
                "",
            ]
        )
    else:
        lines.append("- winner: none (no eligible candidates)")
        lines.append("")

    lines.extend(
        [
            "## Per-Drive",
            "",
            "| drive | status | centerline_len_m | inter_count | ratio | base_ratio | road_components | match_ratio | dist_p95_m | reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in drives:
        if not winner or row.get("candidate_id") != winner.get("candidate_id"):
            continue
        drive = row.get("drive") or ""
        summary = {}
        outputs_dir = row.get("outputs_dir")
        if outputs_dir:
            summary = _read_json(Path(outputs_dir) / "GeomSummary.json")
        base = baseline.get(drive) or {}
        center_len = summary.get("centerline_total_len_m")
        inter_cnt = summary.get("intersections_count")
        ratio = summary.get("centerlines_in_polygon_ratio")
        road_comp = summary.get("road_component_count_after")
        base_ratio = base.get("centerlines_in_polygon_ratio")
        lines.append(
            "| {drive} | {status} | {cl} | {ic} | {ratio} | {br} | {rc} | {mr} | {dp} | {reason} |".format(
                drive=drive,
                status=row.get("status"),
                cl="" if center_len is None else f"{center_len:.2f}",
                ic="" if inter_cnt is None else f"{inter_cnt}",
                ratio="" if ratio is None else f"{ratio:.3f}",
                br="" if base_ratio is None else f"{base_ratio:.3f}",
                rc="" if road_comp is None else f"{road_comp}",
                mr="" if row.get("match_ratio") is None else f"{row.get('match_ratio'):.4f}",
                dp="" if row.get("dist_p95_m") is None else f"{row.get('dist_p95_m'):.2f}",
                reason=row.get("reason") or "",
            )
        )
    (out_dir / "best_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick-dir", default="", help="quick sweep directory (optional)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--config", default="configs/model_zoo.yaml")
    ap.add_argument("--drives-file", default="configs/golden_drives.txt")
    ap.add_argument("--full-max-frames", type=int, default=2000)
    ap.add_argument("--baseline", default="configs/geom_regress_baseline.yaml")
    ap.add_argument("--primary", default="", help="comma-separated primary candidate_ids")
    ap.add_argument("--compare", default="", help="comma-separated compare candidate_ids")
    ap.add_argument("--hf-cache-root", default="", help="override HF cache root")
    args = ap.parse_args()

    drives = _read_drives(Path(args.drives_file))
    if not drives:
        raise SystemExit("ERROR: no drives found in drives-file")

    candidates_map = _load_candidates(Path(args.config))
    if args.primary:
        primary_ids = [c.strip() for c in args.primary.split(",") if c.strip()]
    else:
        primary_ids = []
    if args.compare:
        compare_ids = [c.strip() for c in args.compare.split(",") if c.strip()]
    else:
        compare_ids = []
    if not primary_ids:
        raise SystemExit("ERROR: primary candidates required (use --primary)")
    selected, primary_ids, compare_ids = _select_candidates(primary_ids, compare_ids, candidates_map)
    if not selected:
        raise SystemExit("ERROR: no candidates selected for full run")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"sweep_geom_roadseg_full_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_shared_cache = True
    if os.environ.get("USE_SHARED_HF_CACHE", "").strip() == "0":
        use_shared_cache = False
    if args.hf_cache_root:
        os.environ["HF_HOME"] = args.hf_cache_root
    hf_env = _ensure_hf_cache_env(out_dir, "run", use_shared_cache)
    cache_root = str((out_dir / "hf_cache").resolve()) if not use_shared_cache else hf_env.get("HF_HOME", "")
    ok, err = _check_cache_writable(Path(hf_env.get("HF_HOME", "")))
    if not ok:
        raise SystemExit(
            "ERROR: HF cache not writable: {path} ({err}). "
            "Please add the cache path and .venv\\Scripts\\python.exe to Windows Defender exclusions."
            .format(path=hf_env.get("HF_HOME", ""), err=err)
        )

    index_path = out_dir / "full_index.jsonl"
    if index_path.exists():
        index_path.unlink()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    for cand in selected:
        cand_id = cand.get("candidate_id") or "unknown"
        compare_only = bool(cand.get("compare_only"))
        for drive in drives:
            start = time.time()
            hf_env = _ensure_hf_cache_env(out_dir, cand_id, use_shared_cache)
            rc, run_id, outputs_dir = _run_build_geom(cand, drive, args.full_max_frames, hf_env=hf_env)
            status = "PASS" if rc == 0 else "FAIL"
            reason = "" if rc == 0 else "build_geom_failed"
            if rc == 0:
                osm_rc = _run_osm_ref(outputs_dir, env={**os.environ.copy(), **hf_env})
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
                    "outputs_dir": outputs_dir or None,
                    "model_family": cand.get("model_family"),
                    "model_id": cand.get("model_id"),
                    "timing_sec": round(time.time() - start, 2),
                    "compare_only": compare_only,
                },
            )

    entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = _load_baseline(Path(args.baseline))
    candidate_rows, drive_rows = _score_candidates(entries, drives, baseline)

    full_csv = out_dir / "full_report.csv"
    full_md = out_dir / "full_report.md"
    _write_csv(full_csv, candidate_rows)
    _write_md(full_md, candidate_rows, drive_rows)

    primary_set = set(primary_ids)
    eligible = [
        row for row in candidate_rows
        if row.get("candidate_id") in primary_set
        and row.get("status") == "PASS"
        and int(row.get("pass_count") or 0) == len(drives)
    ]
    eligible.sort(
        key=lambda r: (
            -(r.get("avg_score") or -1e9),
            r.get("avg_timing_sec") if r.get("avg_timing_sec") is not None else 1e9,
        )
    )
    winner = eligible[0] if eligible else None

    best_config = None
    if winner:
        best_config = candidates_map.get(winner.get("candidate_id"))
    if best_config:
        best_path = out_dir / "best_config.yaml"
        best_path.write_text(yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8")

    _write_best_summary(out_dir, winner, drive_rows, args.baseline, baseline)

    _write_runcard(
        out_dir,
        commit,
        Path(args.quick_dir) if args.quick_dir else Path(""),
        drives,
        selected,
        primary_ids,
        compare_ids,
        args.full_max_frames,
        args.baseline,
        os.environ.get("OSM_ROOT", ""),
        use_shared_cache,
        hf_env,
        cache_root,
    )

    if winner and best_config:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rerun_dir = out_dir / f"best_rerun_{ts}"
        rerun_dir.mkdir(parents=True, exist_ok=True)
        rerun_index = rerun_dir / "best_rerun_index.jsonl"
        for drive in drives:
            start = time.time()
            hf_env = _ensure_hf_cache_env(out_dir, winner.get("candidate_id") or "best", use_shared_cache)
            rc, run_id, outputs_dir = _run_build_geom(best_config, drive, args.full_max_frames, hf_env=hf_env)
            status = "PASS" if rc == 0 else "FAIL"
            reason = "" if rc == 0 else "build_geom_failed"
            if rc == 0:
                osm_rc = _run_osm_ref(outputs_dir, env={**os.environ.copy(), **hf_env})
                if osm_rc != 0:
                    status = "FAIL"
                    reason = "osm_ref_failed"
            _write_index(
                rerun_index,
                {
                    "candidate_id": winner.get("candidate_id"),
                    "drive": drive,
                    "status": status,
                    "reason": reason or None,
                    "geom_run_id": run_id or None,
                    "outputs_dir": outputs_dir or None,
                    "timing_sec": round(time.time() - start, 2),
                },
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
