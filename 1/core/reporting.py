from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def write_metrics(run_dir: Path, metrics: Dict[str, Any]) -> Path:
    out = run_dir / "metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_skill_report(run_dir: Path, lines: List[str]) -> Path:
    out = run_dir / "SkillReport.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
