from __future__ import annotations
from pathlib import Path
import subprocess

BAD = {
    "\ufeff",
    "\u202a","\u202b","\u202c","\u202d","\u202e",
    "\u2066","\u2067","\u2068","\u2069",
    "\u200e","\u200f","\u061c",
}

def is_text_like(p: Path) -> bool:
    # 只扫常见文本类型，避免误扫二进制
    name = p.name.lower()
    if name in [".gitignore", "license", "readme.md", "spec.md", "agents.md"]:
        return True
    ext = p.suffix.lower()
    return ext in [".py",".md",".txt",".yaml",".yml",".cmd",".ps1",".json",".toml",".ini",".cfg"]

def main() -> int:
    root = Path(".").resolve()
    files = subprocess.check_output(["git", "ls-files"], text=True, encoding="utf-8").splitlines()

    bad_files = []
    for rel in files:
        p = root / rel
        if not p.exists() or not is_text_like(p):
            continue
        raw = p.read_bytes()
        had_bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig", errors="replace")
        bad_count = sum(1 for ch in text if ch in BAD)
        if had_bom or bad_count > 0:
            bad_files.append((rel, had_bom, bad_count))

    if not bad_files:
        print("[SCAN] OK: no BOM / hidden unicode found in tracked text files.")
        return 0

    print("[SCAN] Found suspicious files:")
    for rel, had_bom, bad_count in bad_files:
        print(f"  - {rel}  (bom={had_bom}, hidden_chars={bad_count})")

    print(f"[SCAN] Total: {len(bad_files)} files")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
