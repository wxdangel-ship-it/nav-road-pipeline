from __future__ import annotations

import argparse
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def _git_lines(args: List[str]) -> List[str]:
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output.decode("utf-8", errors="ignore").splitlines()
    return out.decode("utf-8", errors="ignore").splitlines()


def _collect_git_status() -> List[str]:
    return _git_lines(["git", "status", "--porcelain"])


def _collect_git_index() -> List[str]:
    return _git_lines(["git", "diff", "--name-only", "--cached"])


def _scan_paths_in_text(path: Path) -> List[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"(scripts[\\/][^\\s\"'`]+\\.cmd)", re.IGNORECASE)
    return sorted(set(m.group(1) for m in pattern.finditer(text)))


def _check_paths_exist(paths: List[str]) -> List[str]:
    missing = []
    for raw in paths:
        norm = raw.replace("/", "\\")
        if not Path(norm).exists():
            missing.append(raw)
    return missing


def _filter_forbidden(paths: List[str]) -> List[str]:
    forbidden_prefix = ("runs\\", "cache\\", "data\\")
    forbidden_ext = (".pt", ".pth", ".ckpt", ".onnx")
    hits = []
    for p in paths:
        norm = p.replace("/", "\\")
        if norm.startswith(forbidden_prefix) or norm.endswith(forbidden_ext):
            hits.append(p)
    return hits


def _write_report(out_path: Path, status_lines: List[str], index_lines: List[str], missing_paths: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Precommit Checklist")
    lines.append("")
    lines.append(f"- generated_at: {timestamp}")
    lines.append("")
    lines.append("## Git status (porcelain)")
    if status_lines:
        lines.extend([f"- {line}" for line in status_lines])
    else:
        lines.append("- clean")
    lines.append("")
    lines.append("## Staged files (index)")
    if index_lines:
        lines.extend([f"- {line}" for line in index_lines])
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Forbidden in index (runs/cache/data/weights)")
    forbidden = _filter_forbidden(index_lines)
    if forbidden:
        lines.extend([f"- {line}" for line in forbidden])
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## README/SPEC script path check")
    if missing_paths:
        lines.extend([f"- missing: {line}" for line in missing_paths])
    else:
        lines.append("- ok")
    lines.append("")
    lines.append("## Manual reminders")
    lines.append("- do not add runs/, cache/, data/, or weights to git")
    lines.append("- confirm all .cmd paths referenced in README.md/SPEC.md exist")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="scripts/precommit_checklist.md")
    args = ap.parse_args()

    status_lines = _collect_git_status()
    index_lines = _collect_git_index()
    readme_paths = _scan_paths_in_text(Path("README.md"))
    spec_paths = _scan_paths_in_text(Path("SPEC.md"))
    missing = _check_paths_exist(sorted(set(readme_paths + spec_paths)))

    _write_report(Path(args.write), status_lines, index_lines, missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
