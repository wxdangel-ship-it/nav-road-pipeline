from __future__ import annotations

"""
生成大文件清单：记录路径、大小、mtime、sha256_head。
仅读取前 N 字节用于 hash_head，避免全量读取。
"""

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List


def hash_head(path: Path, head_bytes: int = 2 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        data = f.read(head_bytes)
        h.update(data)
    return h.hexdigest()


def build_manifest(paths: Iterable[Path], head_bytes: int = 2 * 1024 * 1024) -> List[Dict[str, object]]:
    items = []
    for p in paths:
        stat = p.stat()
        items.append(
            {
                "path": str(p),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256_head": hash_head(p, head_bytes=head_bytes),
            }
        )
    return items


__all__ = ["hash_head", "build_manifest"]
