from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
from pipeline._io import load_yaml, ensure_dir, new_run_id, RUNTIME_TARGET
from pipeline._report import write_run_card, write_sync_pack

def _stable_rand01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return (x % 100000) / 100000.0

def _signature(config: dict, arm_name: str, use_osm: bool, use_sat: bool) -> str:
    payload = {
        "config_id": config.get("config_id"),
        "modules": config.get("modules", {}),
        "arm": arm_name,
        "use_osm": use_osm,
        "use_sat": use_sat,
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return s

def _simulate_metrics(sig: str, use_osm: bool, use_sat: bool) -> dict:
    # C 优先：受组合与先验影响
    base = 0.84 + 0.06 * (_stable_rand01("C" + sig) - 0.5)

    bonus = 0.0
    if use_osm:
        bonus += 0.008 + 0.010 * (_stable_rand01("OSM" + sig) - 0.5)
    if use_sat:
        bonus += 0.008 + 0.010 * (_stable_rand01("SAT" + sig) - 0.5)

    C = min(0.99, base + bonus)

    # B/A：受组合与先验冲突影响（冲突率越高，B/A 趋于变差）
    conflict = 0.02 + 0.08 * _stable_rand01("CR" + sig)
    B = 0.22 + 0.18 * _stable_rand01("B" + sig) + 0.10 * conflict
    A = 1.8 + 5.0 * _stable_rand01("A" + sig) + 8.0 * conflict

    prior_used = "NONE"
    if use_osm and use_sat:
        prior_used = "BOTH"
    elif use_osm:
        prior_used = "OSM"
    elif use_sat:
        prior_used = "SAT"

    prior_conf = 0.55 + 0.35 * _stable_rand01("PC" + sig)
    align_res = 0.5 + 1.5 * _stable_rand01("AR" + sig)

    return {
        "C": round(C, 4),
        "B_roughness": round(B, 4),
        "A_dangling_per_km": round(A, 3),
        "prior_used": prior_used,
        "prior_confidence_p50": round(prior_conf, 3),
        "alignment_residual_p50": round(align_res, 3),
        "conflict_rate": round(conflict, 3),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/active.yaml", help="config yaml path")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_yaml(repo / args.config)
    arms = load_yaml(repo / "configs" / "arms.yaml").get("arms", {})
    gates = load_yaml(repo / "configs" / "gates.yaml")

    run_id = new_run_id("eval")
    run_dir = ensure_dir(repo / "runs" / run_id)

    # StateSnapshot
    snap = {
        "run_id": run_id,
        "runtime_target": RUNTIME_TARGET,
        "config_id": cfg.get("config_id"),
        "modules": cfg.get("modules", {}),
    }
    (run_dir / "StateSnapshot.md").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    for arm_name, a in arms.items():
        use_osm = bool(a.get("use_osm", False))
        use_sat = bool(a.get("use_sat", False))
        sig = _signature(cfg, arm_name, use_osm, use_sat)
        m = _simulate_metrics(sig, use_osm, use_sat)
        ok, reason = _gate(m, gates)

        run_card = {
            "run_id": run_id,
            "arm": arm_name,
            "config_id": cfg.get("config_id"),
            "runtime_target": RUNTIME_TARGET,
            "gate": "PASS" if ok else "FAIL",
            "gate_reason": reason,
            "metrics": m,
        }
        write_run_card(run_dir / f"RunCard_{arm_name}.md", run_card)
        write_sync_pack(
            run_dir / f"SyncPack_{arm_name}.md",
            diff={"config_id": cfg.get("config_id"), "arm": arm_name, "config_path": args.config},
            evidence=run_card,
            ask="If FAIL, propose <=3 fixes. If PASS, suggest next autotune actions."
        )

    print(f"[EVAL] DONE -> {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
