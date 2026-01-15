from __future__ import annotations
import hashlib
import json
from typing import Dict, Any, Tuple

def _stable_rand01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return (x % 100000) / 100000.0

def signature(cfg: Dict[str, Any], arm_name: str, use_osm: bool, use_sat: bool) -> str:
    payload = {
        "config_id": cfg.get("config_id"),
        "modules": cfg.get("modules", {}),
        "arm": arm_name,
        "use_osm": use_osm,
        "use_sat": use_sat,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)

def simulate_metrics(sig: str, use_osm: bool, use_sat: bool) -> Dict[str, Any]:
    base = 0.84 + 0.06 * (_stable_rand01("C" + sig) - 0.5)

    bonus = 0.0
    if use_osm:
        bonus += 0.008 + 0.010 * (_stable_rand01("OSM" + sig) - 0.5)
    if use_sat:
        bonus += 0.008 + 0.010 * (_stable_rand01("SAT" + sig) - 0.5)

    C = min(0.99, base + bonus)

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

def gate(metrics: Dict[str, Any], gates: Dict[str, Any]) -> Tuple[bool, str]:
    g = gates.get("gate", {})
    if metrics["C"] < g.get("C_min", 0.8):
        return False, "C below threshold"
    if metrics["B_roughness"] > g.get("B_max_roughness", 0.35):
        return False, "B roughness above threshold"
    if metrics["A_dangling_per_km"] > g.get("A_max_dangling_per_km", 5.0):
        return False, "A dangling above threshold"
    return True, "PASS"
