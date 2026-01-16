from pathlib import Path

# 典型隐藏/双向控制字符（含 BOM、方向控制、LRM/RLM 等）
BAD = {
    "\ufeff",  # BOM
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi override
    "\u2066", "\u2067", "\u2068", "\u2069",            # isolates
    "\u200e", "\u200f",                                # LRM/RLM
    "\u061c",                                          # Arabic letter mark
}

targets = [Path(".gitignore"), Path("pipeline/autotune.py")]

for p in targets:
    if not p.exists():
        print("skip (missing):", p)
        continue

    raw = p.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")

    # utf-8-sig 会自动去 BOM
    text = raw.decode("utf-8-sig", errors="replace")

    removed = sum(1 for ch in text if ch in BAD)
    cleaned = "".join(ch for ch in text if ch not in BAD)

    # 关键：只要有 BOM 或 removed>0，就必须写回
    if had_bom or removed > 0:
        p.write_text(cleaned, encoding="utf-8", newline="\n")
        print(f"cleaned: {p} (had_bom={had_bom}, removed={removed})")
    else:
        print(f"no change: {p}")
