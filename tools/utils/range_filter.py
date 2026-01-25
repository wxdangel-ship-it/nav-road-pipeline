from __future__ import annotations

from typing import Any


def _frame_id_to_int(frame_id: Any) -> int | None:
    if frame_id is None:
        return None
    try:
        return int(str(frame_id))
    except Exception:
        return None


def in_range(frame_id: Any, start: int, end: int) -> bool:
    fid = _frame_id_to_int(frame_id)
    if fid is None:
        return False
    return start <= fid <= end
