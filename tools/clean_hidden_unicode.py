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
    name = p.name.lower()
    if name in [".gitignore", "license", "readme.md", "spec.md", "agents.md"]:
        return True
    ext = p.suffix.lower()
    return ext in [".py",".md",".txt",".yaml",".yml",".cmd",".ps1",".json",".toml",".ini",".cfg"]

def main() -> int:
    root = Path(".").resolve()
    files = subprocess.check_output(["git", "ls-files"], text=True, encoding="utf-8").splitlines()

    cleaned_any = False
    for rel in files:
        p = root / rel
        if not p.exists() or not is_text_like(p):
            continue
        raw = p.read_bytes()
        had_bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig", errors="replace")
        removed = sum(1 for ch in text if ch in BAD)
        if had_bom or removed > 0:
            cleaned = "".join(ch for ch in text if ch not in BAD)
            p.write_text(cleaned, encoding="utf-8", newline="\n")
            print(f"[CLEAN] {rel} (had_bom={had_bom}, removed={removed})")
            cleaned_any = True

    if not cleaned_any:
        print("[CLEAN] nothing to do.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
