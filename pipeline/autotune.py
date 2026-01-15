from __future__ import annotations
from pathlib import Path
import random
from pipeline._io import load_yaml, ensure_dir, new_run_id
from pipeline.eval_all import _simulate_metrics, _gate

def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    search = load_yaml(repo / "configs" / "search.yaml").get("autotune", {})
    gates = load_yaml(repo / "configs" / "gates.yaml")
    cfg = load_yaml(repo / "configs" / "active.yaml")
    arms = load_yaml(repo / "configs" / "arms.yaml").get("arms", {})

    budget = int(search.get("stageB_budget_trials", 20))
    seed = int(search.get("seed", 42))
    random.seed(seed)

    run_id = new_run_id("autotune")
    run_dir = ensure_dir(repo / "runs" / run_id)

    baseline_id = str(cfg.get("config_id", "CFG_BASELINE"))
    results = []

    for i in range(budget):
        cand_id = f"{baseline_id}_T{i:03d}"
        score = 0.0
        pass_all = True
        for arm_name, a in arms.items():
            m = _simulate_metrics(cand_id, arm_name, bool(a.get("use_osm")), bool(a.get("use_sat")))
            ok, _ = _gate(m, gates)
            if not ok:
                pass_all = False
                break
            score += m["C"] - 0.2 * m["B_roughness"] - 0.01 * m["A_dangling_per_km"]
        if pass_all:
            results.append((score, cand_id))

    results.sort(reverse=True, key=lambda x: x[0])
    top = results[:10]

    lines = ["# Leaderboard", "", f"- run_id: {run_id}", f"- budget_trials: {budget}", ""]
    for rank, (s, cid) in enumerate(top, 1):
        lines.append(f"{rank}. {cid}  score={s:.6f}")
    (run_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[AUTOTUNE] DONE -> {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
