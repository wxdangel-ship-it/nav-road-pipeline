from __future__ import annotations

from pathlib import Path


PATTERNS = [
    "_fullpose_transform(",
    "load_kitti360_lidar_points_world_full(",
    "r_world_pose @",
]


def main() -> int:
    repo = Path(".").resolve()
    files = [p for p in repo.rglob("*.py") if "runs" not in str(p)]
    hits = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in PATTERNS:
            if pat in text:
                hits.append((str(path), pat))
    if hits:
        print("[WARN] legacy world transform patterns found:")
        for path, pat in hits:
            print(f"- {path}: {pat}")
        return 1
    print("[OK] no legacy world transform patterns found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
