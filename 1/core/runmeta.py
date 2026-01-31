import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _try_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None


def _try_git_status(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="ignore").strip()
        return s
    except Exception:
        return None


@dataclass
class RunMeta:
    run_id: str
    created_at: str
    repo_commit: Optional[str]
    repo_dirty: Optional[bool]
    python: str
    platform: str
    params: Dict[str, Any]
    inputs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_runmeta(run_dir: Path, repo_root: Path, run_id: str, params: Dict[str, Any], inputs: Dict[str, Any]) -> Path:
    commit = _try_git_commit(repo_root)
    status = _try_git_status(repo_root)
    meta = RunMeta(
        run_id=run_id,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        repo_commit=commit,
        repo_dirty=None if status is None else (len(status) > 0),
        python=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.version()})",
        params=params,
        inputs=inputs,
    )
    out = run_dir / "RunMeta.json"
    out.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out
