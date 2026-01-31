from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import yaml

from .config_resolve import get_params_hash


def _read_report_hash(report_path: Path) -> str:
    if not report_path.exists():
        return ""
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.lower().strip().startswith("- params_hash:"):
            return line.split(":", 1)[-1].strip()
    return ""


def _read_meta_hash(meta_path: Path) -> str:
    if not meta_path.exists():
        return ""
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return str(payload.get("params_hash") or "")
    except Exception:
        return ""


def postcheck(run_dir: Path, report_path: Path, meta_path: Path) -> Tuple[bool, str]:
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.exists():
        return False, "missing_resolved_config"
    cfg = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    params_hash_run = get_params_hash(cfg)

    summary_path = run_dir / "run_summary.json"
    params_hash_summary = ""
    if summary_path.exists():
        try:
            params_hash_summary = str(json.loads(summary_path.read_text(encoding="utf-8")).get("params_hash") or "")
        except Exception:
            params_hash_summary = ""

    params_hash_report = _read_report_hash(report_path)
    params_hash_out = _read_meta_hash(meta_path)

    if params_hash_summary and params_hash_summary != params_hash_run:
        return False, "summary_hash_mismatch"
    if params_hash_report and params_hash_report != params_hash_run:
        return False, "report_hash_mismatch"
    if params_hash_out and params_hash_out != params_hash_run:
        return False, "output_hash_mismatch"
    if not params_hash_out:
        return False, "output_hash_missing"
    return True, "ok"


__all__ = ["postcheck"]
