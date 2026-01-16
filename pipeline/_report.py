from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

def md_kv(title: str, kv: Dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for k, v in kv.items():
        if isinstance(v, (dict, list)):
            vv = json.dumps(v, ensure_ascii=False, indent=2)
            lines.append(f"## {k}\n```json\n{vv}\n```")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)

def write_run_card(path: Path, kv: Dict[str, Any]) -> None:
    path.write_text(md_kv("RunCard", kv), encoding="utf-8")

def write_sync_pack(path: Path, diff: Dict[str, Any], evidence: Dict[str, Any], ask: str) -> None:
    kv = {"Diff": diff, "Evidence": evidence, "Ask": ask}
    path.write_text(md_kv("SyncPack", kv), encoding="utf-8")
