from __future__ import annotations
from pathlib import Path
import json
import random
from copy import deepcopy
from pipeline._io import load_yaml, ensure_dir, new_run_id
from pipeline.registry import load_registry, group_by_module, sample_params
from pipeline.eval_all import _gate, _signature, _simulate_metrics

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

            ok, score, _ = _score_one_config(cand, {"Arm0": arms["Arm0"]}, gates)  # Stage A 默认只跑 Arm0
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

        ok, score, per_arm = _score_one_config(cand, arms, gates)
        if ok:
            trials.append({"score": score, "config": cand, "per_arm": per_arm})

    trials.sort(reverse=True, key=lambda x: x["score"])
    (run_dir / "trials_top.json").write_text(json.dumps(trials[:topn], ensure_ascii=False, indent=2), encoding="utf-8")

    # Leaderboard
    lines = ["# Leaderboard", "", f"- run_id: {run_id}", f"- budget_trials: {budget}", ""]
    for rank, t in enumerate(trials[:10], 1):
        lines.append(f"{rank}. {t['config']['config_id']}  score={t['score']:.6f}")
    (run_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -------- Stage C: freeze candidate --------
    if trials:
        winner = trials[0]["config"]
        (run_dir / "winner_active.yaml").write_text(json.dumps(winner, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "winner_hint.md").write_text(
            f"# Winner\n\n- config_id: {winner.get('config_id')}\n- note: review winner_active.yaml then decide whether to apply to configs/active.yaml\n",
            encoding="utf-8"
        )

    print(f"[AUTOTUNE] DONE -> {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
