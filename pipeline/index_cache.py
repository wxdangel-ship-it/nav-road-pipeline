from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json


def _norm(p: Path) -> str:
    try:
        return str(p.resolve()).lower()
    except Exception:
        return str(p).lower()


def load_index(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def try_use_index(index_path: Path, data_root: Path, max_frames: Optional[int]) -> Optional[Dict[str, Any]]:
    if not index_path.exists():
        return None
    idx = load_index(index_path)
    meta = idx.get("meta", {}) or {}

    meta_root = meta.get("data_root", "")
    if not meta_root:
        return None

    if _norm(Path(meta_root)) != _norm(data_root):
        return None

    meta_max = meta.get("max_frames", None)
    want_max = max_frames if (max_frames is not None and max_frames > 0) else None

    # json 中 null -> None
    if meta_max != want_max:
        return None

    return idx
