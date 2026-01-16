from __future__ import annotations
from pathlib import Path
import argparse
import json
import os
import random
import re
import subprocess
import yaml
from copy import deepcopy
from pipeline._io import load_yaml, ensure_dir, new_run_id
from pipeline.registry import load_registry, group_by_module, sample_params
from pipeline.sim_metrics import gate as _gate, signature as _signature, simulate_metrics as _simulate_metrics

def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, allow_unicode=False, sort_keys=False)
    path.write_text(text, encoding="utf-8")

def _parse_drives(drives: str) -> list[str]:
    return [x.strip() for x in drives.split(",") if x.strip()]

def _load_index_any(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

def _select_representative_drives(idx: dict, k: int) -> list[str]:
    tiles = idx.get("tiles", []) or []
    tiles = [t for t in tiles if t.get("lidar_count", 0) > 0]
    if not tiles:
        return []
    tiles_sorted = sorted(
        tiles,
        key=lambda t: (float(t.get("image_coverage", 0.0)), int(t.get("lidar_count", 0))),
    )
    if k <= 1:
        return [tiles_sorted[len(tiles_sorted) // 2].get("tile_id")]
    picks = []
    for i in range(k):
        pos = round(i * (len(tiles_sorted) - 1) / (k - 1))
        picks.append(tiles_sorted[pos].get("tile_id"))
    # unique preserve order
    seen = set()
    out = []
    for d in picks:
        if d and d not in seen:
            out.append(d)
            seen.add(d)
    return out

def _extract_eval_run_dir(text: str, repo: Path) -> Path | None:
    m = re.search(r"\[EVAL\]\s+DONE\s+->\s+([^\r\n(]+)", text)
    if not m:
        return None
    raw = m.group(1).strip().strip("\"'")
    p = Path(raw)
    if not p.is_absolute():
        p = repo / p
    return p

def _latest_eval_run(repo: Path) -> Path | None:
    runs_dir = repo / "runs"
    if not runs_dir.exists():
        return None
    eval_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("eval_")]
    if not eval_dirs:
        return None
    eval_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return eval_dirs[0]

def _extract_json_section(text: str, section: str) -> dict | None:
    pattern = rf"## {re.escape(section)}\s+```json\s+(.*?)\s+```"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None

def _extract_metrics_block(text: str) -> dict:
    blocks = re.findall(r"```json\s+(.*?)\s+```", text, re.DOTALL)
    for blk in blocks:
        try:
            data = json.loads(blk)
        except json.JSONDecodeError:
            continue
        if all(k in data for k in ["C", "B_roughness", "A_dangling_per_km"]):
            return data
    return {}

def _read_run_card(path: Path) -> dict:
    json_path = path.with_suffix(".json")
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    text = path.read_text(encoding="utf-8")
    metrics = _extract_json_section(text, "metrics") or _extract_metrics_block(text)
    gate = None
    gate_reason = None
    for line in text.splitlines():
        if line.startswith("- gate:"):
            gate = line.split(":", 1)[1].strip()
        elif line.startswith("- gate_reason:"):
            gate_reason = line.split(":", 1)[1].strip()
    return {"metrics": metrics, "gate": gate, "gate_reason": gate_reason}

def _run_eval_cmd(
    repo: Path,
    config_path: Path,
    max_frames: int,
    data_root: str | None,
    prior_root: str | None,
    index_path: Path | None,
    drives: list[str] | None,
    eval_mode: str,
    drive: str | None,
) -> Path:
    cmd = [
        "cmd.exe",
        "/c",
        str(repo / "scripts" / "eval.cmd"),
        "--config",
        str(config_path),
    ]
    if data_root:
        cmd += ["--data-root", data_root]
    if prior_root:
        cmd += ["--prior-root", prior_root]
    if max_frames and max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]
    if index_path:
        cmd += ["--index", str(index_path)]
    if drives:
        cmd += ["--drives", ",".join(drives)]
    if eval_mode:
        cmd += ["--eval-mode", eval_mode]
    if drive:
        cmd += ["--drive", drive]

    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    run_dir = _extract_eval_run_dir(output, repo)
    if run_dir is None:
        run_dir = _latest_eval_run(repo)
    if proc.returncode != 0 or run_dir is None:
        raise SystemExit(f"ERROR: eval.cmd failed (code={proc.returncode}). Output:\n{output}")
    return run_dir

def _run_index_cmd(repo: Path, data_root: str, max_frames: int, out_path: Path) -> None:
    cmd = [
        "cmd.exe",
        "/c",
        str(repo / "scripts" / "index.cmd"),
        "--data-root",
        data_root,
        "--max-frames",
        str(max_frames),
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    if proc.returncode != 0:
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise SystemExit(f"ERROR: index.cmd failed (code={proc.returncode}). Output:\n{output}")

def _score_one_config_real(
    repo: Path,
    cfg: dict,
    arms: dict,
    gates: dict,
    run_dir: Path,
    max_frames: int,
    data_root: str | None,
    prior_root: str | None,
    index_path: Path | None,
    drives: list[str] | None,
    eval_mode: str,
    drive: str | None,
    only_arms: list[str] | None = None,
) -> tuple[bool, float, dict, Path]:
    cfg_id = cfg.get("config_id", "CFG")
    cfg_path = run_dir / "candidates" / f"{cfg_id}.yaml"
    _write_yaml(cfg_path, cfg)
    eval_run_dir = _run_eval_cmd(repo, cfg_path, max_frames, data_root, prior_root, index_path, drives, eval_mode, drive)

    total = 0.0
    any_fail = False
    per_arm = {}
    arm_list = only_arms or list(arms.keys())
    for arm_name in arm_list:
        rc_path = eval_run_dir / f"RunCard_{arm_name}.md"
        if not rc_path.exists():
            any_fail = True
            break
        rc = _read_run_card(rc_path)
        m = rc.get("metrics", {})
        ok, _ = _gate(m, gates)
        per_arm[arm_name] = {"ok": ok, "metrics": m, "run_dir": str(eval_run_dir)}
        if not ok:
            any_fail = True
            break
        total += m["C"] - 0.25 * m["B_roughness"] - 0.01 * m["A_dangling_per_km"] - 0.5 * m["conflict_rate"]
    return (not any_fail), total, per_arm, eval_run_dir

def _score_one_config(cfg: dict, arms: dict, gates: dict) -> tuple[bool, float, dict]:
    total = 0.0
    any_fail = False
    per_arm = {}
    for arm_name, a in arms.items():
        use_osm = bool(a.get("use_osm", False))
        use_sat = bool(a.get("use_sat", False))
        sig = _signature(cfg, arm_name, use_osm, use_sat)
        m = _simulate_metrics(sig, use_osm, use_sat)
        ok, _ = _gate(m, gates)
        per_arm[arm_name] = {"ok": ok, "metrics": m}
        if not ok:
            any_fail = True
            break
        total += m["C"] - 0.25 * m["B_roughness"] - 0.01 * m["A_dangling_per_km"] - 0.5 * m["conflict_rate"]
    return (not any_fail), total, per_arm

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sim", choices=["sim", "real"], help="autotune mode")
    ap.add_argument("--data-root", default="", help="KITTI-360 root (optional, default=POC_DATA_ROOT)")
    ap.add_argument("--prior-root", default="", help="prior root (optional, default=POC_PRIOR_ROOT or data-root)")
    ap.add_argument("--index", default="cache/kitti360_index.json", help="index cache path")
    ap.add_argument("--drives", default="", help="comma separated drives subset (optional)")
    ap.add_argument("--max-frames", type=int, default=2000, help="max frames for real-mode evals")
    ap.add_argument("--eval-mode", default="summary", choices=["summary", "geom"], help="eval mode for real")
    ap.add_argument("--drive", default="2013_05_28_drive_0000_sync", help="single drive for geom eval")
    ap.add_argument("--stageA-max-frames", type=int, default=0, help="override stage A max_frames")
    ap.add_argument("--stageB-max-frames", type=int, default=0, help="override stage B max_frames")
    ap.add_argument("--stageC-max-frames", type=int, default=0, help="override stage C max_frames")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg_active = load_yaml(repo / "configs" / "active.yaml")
    arms = load_yaml(repo / "configs" / "arms.yaml").get("arms", {})
    gates = load_yaml(repo / "configs" / "gates.yaml")
    search = load_yaml(repo / "configs" / "search.yaml").get("autotune", {})

    reg = load_registry(repo)
    by_mod = group_by_module(reg)

    topk = int(search.get("stageA_topk", 2))
    budget = int(search.get("stageB_budget_trials", 20))
    topn = int(search.get("stageC_topn", 3))
    seed = int(search.get("seed", 42))
    rng = random.Random(seed)

    run_id = new_run_id("autotune")
    run_dir = ensure_dir(repo / "runs" / run_id)

    mode = str(args.mode).lower()
    data_root = args.data_root or os.environ.get("POC_DATA_ROOT", "")
    prior_root = args.prior_root or os.environ.get("POC_PRIOR_ROOT", "")
    index_path = repo / args.index

    default_max_frames = int(args.max_frames) if args.max_frames and args.max_frames > 0 else 2000
    stageA_max_frames = int(args.stageA_max_frames) if args.stageA_max_frames > 0 else default_max_frames
    stageB_max_frames = int(args.stageB_max_frames) if args.stageB_max_frames > 0 else default_max_frames
    stageC_max_frames = int(args.stageC_max_frames) if args.stageC_max_frames > 0 else default_max_frames
    stageA_drive_count = int(search.get("real_stageA_drive_count", 3))
    drives_arg = _parse_drives(args.drives) if args.drives else []
    eval_mode = str(args.eval_mode).lower()
    geom_drive = str(args.drive) if args.drive else "2013_05_28_drive_0000_sync"

    if mode == "real":
        if not data_root:
            raise SystemExit("ERROR: --data-root or POC_DATA_ROOT is required for real mode.")
        if str(args.index) == "cache/kitti360_index.json":
            index_path = repo / f"cache/kitti360_index_{default_max_frames}.json"
        if not index_path.exists():
            _run_index_cmd(repo, data_root, default_max_frames, index_path)
        idx = _load_index_any(index_path)
        if not drives_arg and eval_mode != "geom":
            if idx is None:
                raise SystemExit("ERROR: index cache missing and --drives not provided for real mode.")
            stageA_drives = _select_representative_drives(idx, stageA_drive_count)
        else:
            stageA_drives = drives_arg
        if not stageA_drives:
            if eval_mode != "geom":
                raise SystemExit("ERROR: no drives selected for Stage A.")

    # -------- Stage A: per-module screening --------
    stageA = {}
    baseline_modules = cfg_active.get("modules", {})

    # 只筛选“在注册表里有 >=2 个实现”的模块
    candidate_modules = [m for m, lst in by_mod.items() if len(lst) >= 2 and m in ["M0", "M2", "M4", "M6a"]]
    if not candidate_modules:
        candidate_modules = [m for m, lst in by_mod.items() if len(lst) >= 2]

    for mod in candidate_modules:
        impls = by_mod.get(mod, [])
        scored = []
        for impl in impls:
            cand = deepcopy(cfg_active)
            cand["config_id"] = f"{cfg_active.get('config_id')}_A_{mod}_{impl.get('impl_id')}"
            cand_modules = deepcopy(baseline_modules)
            # active.yaml 的 modules key 可能是 M6a，而 registry module 也是 M6a
            key = mod
            if key not in cand_modules:
                # 容错：跳过不在 active 的模块
                continue
            cand_modules[key] = {"impl_id": impl.get("impl_id"), "params": sample_params(impl, rng)}
            cand["modules"] = cand_modules

            if mode == "sim":
                ok, score, _ = _score_one_config(cand, {"Arm0": arms["Arm0"]}, gates)  # Stage A 默认只跑 Arm0
            else:
                ok, score, _per_arm, _ = _score_one_config_real(
                    repo=repo,
                    cfg=cand,
                    arms=arms,
                    gates=gates,
                    run_dir=run_dir,
                    max_frames=stageA_max_frames,
                    data_root=data_root,
                    prior_root=prior_root or data_root,
                    index_path=index_path,
                    drives=stageA_drives if eval_mode != "geom" else None,
                    eval_mode=eval_mode,
                    drive=geom_drive if eval_mode == "geom" else None,
                    only_arms=["Arm0"],
                )
            if ok:
                scored.append((score, impl.get("impl_id"), cand_modules[key]["params"]))
        scored.sort(reverse=True, key=lambda x: x[0])
        stageA[mod] = scored[:topk]

    (run_dir / "stageA_topk.json").write_text(json.dumps(stageA, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------- Stage B: joint search --------
    # 为每个模块建立可选实现列表：若 StageA 有结果用它，否则用 active 的 impl
    space = {}
    for mod_key, base in baseline_modules.items():
        if mod_key in stageA and stageA[mod_key]:
            space[mod_key] = stageA[mod_key]
        else:
            # fallback: 只保留 baseline 自己
            space[mod_key] = [(0.0, base.get("impl_id"), base.get("params", {}))]

    trials = []
    stageB_drives = drives_arg if drives_arg else (stageA_drives if mode == "real" else None)
    stageC_drives = drives_arg if drives_arg else None

    for i in range(budget):
        cand = deepcopy(cfg_active)
        cand["config_id"] = f"{cfg_active.get('config_id')}_B_T{i:03d}"
        cand_modules = {}
        for mod_key, choices in space.items():
            _, impl_id, _params_hint = rng.choice(choices)
            # 取 registry 里对应 impl 的 param_schema 来采样
            # 如果找不到，就沿用 hint
            params = deepcopy(_params_hint)
            # 尝试从 registry 重采样（如果存在）
            impl_obj = None
            for obj in by_mod.get(mod_key, []):
                if str(obj.get("impl_id")) == str(impl_id):
                    impl_obj = obj
                    break
            if impl_obj is not None:
                params = sample_params(impl_obj, rng)
            cand_modules[mod_key] = {"impl_id": impl_id, "params": params}
        cand["modules"] = cand_modules

        if mode == "sim":
            ok, score, per_arm = _score_one_config(cand, arms, gates)
            if ok:
                trials.append({"score": score, "config": cand, "per_arm": per_arm})
        else:
            ok, score, per_arm, eval_run_dir = _score_one_config_real(
                repo=repo,
                cfg=cand,
                arms=arms,
                gates=gates,
                run_dir=run_dir,
                max_frames=stageB_max_frames,
                data_root=data_root,
                prior_root=prior_root or data_root,
                index_path=index_path,
                drives=stageB_drives if eval_mode != "geom" else None,
                eval_mode=eval_mode,
                drive=geom_drive if eval_mode == "geom" else None,
                only_arms=None,
            )
            if ok:
                trials.append({"score": score, "config": cand, "per_arm": per_arm, "run_dir": str(eval_run_dir)})

    trials.sort(reverse=True, key=lambda x: x["score"])
    (run_dir / "trials_top.json").write_text(json.dumps(trials[:topn], ensure_ascii=False, indent=2), encoding="utf-8")

    # Leaderboard
    lines = ["# Leaderboard", "", f"- run_id: {run_id}", f"- budget_trials: {budget}", f"- mode: {mode}", ""]
    for rank, t in enumerate(trials[:10], 1):
        lines.append(f"{rank}. {t['config']['config_id']}  score={t['score']:.6f}")
    (run_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -------- Stage C: freeze candidate --------
    if trials:
        winner = trials[0]["config"]
        if mode == "real":
            stageC_results = []
            for t in trials[:topn]:
                ok, score, per_arm, eval_run_dir = _score_one_config_real(
                    repo=repo,
                    cfg=t["config"],
                    arms=arms,
                    gates=gates,
                    run_dir=run_dir,
                    max_frames=stageC_max_frames,
                    data_root=data_root,
                    prior_root=prior_root or data_root,
                    index_path=index_path,
                    drives=stageC_drives if eval_mode != "geom" else None,
                    eval_mode=eval_mode,
                    drive=geom_drive if eval_mode == "geom" else None,
                    only_arms=None,
                )
                if ok:
                    stageC_results.append({"score": score, "config": t["config"], "per_arm": per_arm, "run_dir": str(eval_run_dir)})
            if stageC_results:
                stageC_results.sort(reverse=True, key=lambda x: x["score"])
                winner = stageC_results[0]["config"]

        (run_dir / "winner_active.yaml").write_text(json.dumps(winner, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "winner_hint.md").write_text(
            f"# Winner\n\n- config_id: {winner.get('config_id')}\n- note: review winner_active.yaml then decide whether to apply to configs/active.yaml\n",
            encoding="utf-8"
        )

    print(f"[AUTOTUNE] DONE -> {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

