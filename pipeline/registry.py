from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
import yaml

def load_registry(repo: Path) -> Dict[str, Any]:
    p = repo / "modules_registry.yaml"
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def group_by_module(reg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for item in reg.get("implementations", []) or []:
        mod = str(item.get("module"))
        out.setdefault(mod, []).append(item)
    return out

def find_impl(reg: Dict[str, Any], module: str, impl_id: str) -> Optional[Dict[str, Any]]:
    for item in reg.get("implementations", []) or []:
        if str(item.get("module")) == module and str(item.get("impl_id")) == impl_id:
            return item
    return None

def _sample_one(ps: Dict[str, Any], rng: random.Random):
    t = ps.get("type")
    if t == "float":
        lo, hi = ps.get("range", [0.0, 1.0])
        return float(lo) + (float(hi) - float(lo)) * rng.random()
    if t == "int":
        lo, hi = ps.get("range", [0, 10])
        return int(rng.randint(int(lo), int(hi)))
    if t == "bool":
        return bool(rng.random() < 0.5)
    if t == "enum":
        choices = ps.get("choices", [])
        if not choices:
            return ps.get("default")
        return rng.choice(choices)
    return ps.get("default")

def sample_params(impl: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ps in impl.get("param_schema", []) or []:
        if ps.get("tunable", True) is False:
            out[str(ps.get("name"))] = ps.get("default")
        else:
            out[str(ps.get("name"))] = _sample_one(ps, rng)
    return out
