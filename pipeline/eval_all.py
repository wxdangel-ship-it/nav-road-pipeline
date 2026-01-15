from __future__ import annotations
from pathlib import Path
import hashlib
from pipeline._io import load_yaml, ensure_dir, new_run_id, RUNTIME_TARGET
from pipeline._report import write_run_card, write_sync_pack

def _stable_rand01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return (x % 100000) / 100000.0

def _simulate_metrics(config_id: str, arm: str, use_osm: bool, use_sat: bool) -> dict:
    base = 0.85 + 0.05 * (_stable_rand01(config_id + arm) - 0.5)
    bonus = 0.0
    if use_osm:
        bonus += 0.01 + 0.01 * (_stable_rand01("osm" + config_id) - 0.5)
    if use_sat:
        bonus += 0.01 + 0.01 * (_stable_rand01("sat" + config_id) - 0.5)

    C = min(0.99, base + bonus)
    B = 0.25 + 0.10 * (_stable_rand01("B" + config_id + arm))
    A = 2.0 + 4.0 * (_stable_rand01("A" + config_id + arm))

    prior_used = "NONE"
    if use_osm and use_sat:
        prior_used = "BOTH"
    elif use_osm:
        prior_used = "OSM"
    elif use_sat:
        prior_used = "SAT"

    return {
        "C": round(C, 4),
        "B_roughness": round(B, 4),
        "A_dangling_per_km": round(A, 3),
        "prior_used": prior_used,
        "prior_confidence_p50": round(0.7 + 0.2 * _stable_rand01("pc50" + config_id + arm), 3),
        "alignment_residual_p50": round(0.8 + 0.8 * _stable_rand01("ar50" + config_id + arm), 3),
        "conflict_rate": round(0.02 + 0.06 * _stable_rand01("cr" + config_id + arm), 3),
    }

def _gate(metrics: dict, gates: dict) -> tuple[bool, str]:
    g = gates.get("gate", {})
    if metrics["C"] < g.get("C_min", 0.8):
        return False, "C below threshold"
    if metrics["B_roughness"] > g.get("B_max_roughness", 0.35):
        return False, "B roughness above threshold"
    if metrics["A_dangling_per_km"] > g.get("A_max_dangling_per_km", 5.0):
        return False, "A dangling above threshold"
    return True, "PASS"

def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_yaml(repo / "configs" / "active.yaml")
    arms = load_yaml(repo / "configs" / "arms.yaml").get("arms", {})
    gates = load_yaml(repo / "configs" / "gates.yaml")

    run_id = new_run_id("eval")
    run_dir = ensure_dir(repo / "runs" / run_id)

    (run_dir / "StateSnapshot.md").write_text(
        f"run_id={run_id}\nconfig_id={cfg.get('config_id')}\nruntime_target={RUNTIME_TARGET}\n",
        encoding="utf-8"
    )

    config_id = str(cfg.get("config_id", "CFG_UNKNOWN"))

    for arm_name, a in arms.items():
        use_osm = bool(a.get("use_osm", False))
        use_sat = bool(a.get("use_sat", False))
        m = _simulate_metrics(config_id, arm_name, use_osm, use_sat)
        ok, reason = _gate(m, gates)

        run_card = {
            "run_id": run_id,
            "arm": arm_name,
            "config_id": config_id,
            "runtime_target": RUNTIME_TARGET,
            "gate": "PASS" if ok else "FAIL",
            "gate_reason": reason,
            "metrics": m,
        }
        write_run_card(run_dir / f"RunCard_{arm_name}.md", run_card)
        write_sync_pack(
            run_dir / f"SyncPack_{arm_name}.md",
            diff={"config_id": config_id, "arm": arm_name},
            evidence=run_card,
            ask="Review results and propose next actions (<=3)."
        )

    print(f"[EVAL] DONE -> {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
