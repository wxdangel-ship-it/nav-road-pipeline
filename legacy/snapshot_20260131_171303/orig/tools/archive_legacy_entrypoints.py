# -*- coding: utf-8 -*-
"""
功能：
    Phase1 - Legacy Vault：归档现有 cmd/py 入口脚本，并生成索引与依赖关系（不跑链路，不改算法）。
    目标是从“混乱不可见”变为“可检索、可分组、可追溯”，为后续 Skill 化迁移做准备。

输入：
    - REPO_ROOT：项目根目录（示例路径应使用原始字符串 r"D:\\Work\\xxx" 形式以避免反斜杠转义问题）
输出：
    legacy/
      snapshot_<timestamp>/
        orig/                      # 归档副本（按原相对路径保留）
        catalog.csv                # 全量清单（含未拷贝者）
        catalog.json
        deps_cmd_edges.csv         # cmd 调用依赖边（from -> to）
        LEGACY_INDEX.md            # 人读分组摘要
        manifest.json              # 快照元信息（时间、commit、统计）
        logs/run.log
      ACTIVE_SNAPSHOT.txt          # 指向最新 snapshot 目录

参数：
    - ARCHIVE_PY_MODE：
        * "entrypoints"：仅归档入口类 py（tools/scripts 下 或包含 main guard）
        * "tools_scripts"：归档 tools/ 与 scripts/ 下所有 py
        * "all"：归档所有 py（不建议，通常太大）
    - CMD_EXTS：默认只处理 .cmd（可扩展 .bat/.ps1）
    - EXCLUDE_DIR_NAMES：排除 runs/outputs/.venv/.git 等

说明：
    - 归档采用“复制副本”，不会移动原文件，避免破坏现有调用。
    - 依赖解析仅做轻量：从 cmd 中提取 call/python 相关行，构建边列表。
"""

import os
import re
import json
import csv
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


# =========================
# 参数区（集中可编辑）
# =========================
REPO_ROOT = r"E:\Work\nav-road-pipeline"  # TODO：改成你的项目根目录（Windows路径建议用原始字符串）
OVERWRITE = False  # snapshot 一般不覆盖；若同名存在可改 True

# 归档策略
ARCHIVE_PY_MODE = "entrypoints"  # "entrypoints" | "tools_scripts" | "all"
CMD_EXTS = {".cmd"}  # 如需扩展可加 ".bat", ".ps1"

# 扫描范围
INCLUDE_PY = True
INCLUDE_CMD = True

EXCLUDE_DIR_NAMES = {
    ".git", ".svn", ".hg",
    "__pycache__", ".idea",
    ".venv", "venv",
    "runs", "outputs", "baselines",
    "dist", "build", ".mypy_cache", ".pytest_cache"
}

MAX_BYTES_FOR_HEAD = 64 * 1024  # 读取文件头部用于摘要/入口判断
LOG_LEVEL = "INFO"


# =========================
# 函数区
# =========================
def setup_logger(log_path: Path) -> logging.Logger:
    """创建日志对象（控制台+文件）。"""
    logger = logging.getLogger("legacy_archive")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    ch.setFormatter(fmt)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    """运行命令并返回(返回码, 输出)。"""
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return p.returncode, (p.stdout or "").strip()
    except Exception as e:
        return 999, f"ERROR: {e}"


def ensure_dir(path: Path) -> None:
    """创建目录（若不存在）。"""
    path.mkdir(parents=True, exist_ok=True)


def read_head_text(path: Path, max_bytes: int) -> str:
    """读取文件头部文本（用于摘要/入口判断）。"""
    try:
        data = path.read_bytes()[:max_bytes]
        for enc in ("utf-8", "gbk", "utf-16"):
            try:
                return data.decode(enc, errors="ignore")
            except Exception:
                continue
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def sha1_partial(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    """计算 sha1（最多读取前2MB，足够用于变更指纹）。"""
    h = hashlib.sha1()
    read_bytes = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(256 * 1024)
            if not chunk:
                break
            h.update(chunk)
            read_bytes += len(chunk)
            if read_bytes >= max_bytes:
                break
    return h.hexdigest()


def is_python_entrypoint(rel_path: str, head_text: str) -> bool:
    """
    判断 py 是否入口：
    - tools/ 或 scripts/ 下的 py：按模式决定
    - 或包含 if __name__ == "__main__"
    - 或包含 argparse/click/typer 等 CLI 迹象
    """
    p = rel_path.replace("\\", "/").lower()

    has_main_guard = ('__name__' in head_text and '__main__' in head_text)
    has_cli = ("argparse" in head_text) or ("click" in head_text) or ("typer" in head_text)

    if ARCHIVE_PY_MODE == "all":
        return True
    if ARCHIVE_PY_MODE == "tools_scripts":
        return p.startswith("tools/") or p.startswith("scripts/")
    # entrypoints
    if p.startswith("tools/") or p.startswith("scripts/"):
        return True
    return has_main_guard or has_cli


def extract_summary(ext: str, head_text: str) -> str:
    """提取短摘要：优先取注释/文档字符串首句。"""
    lines = [ln.strip() for ln in head_text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""

    if ext == ".py":
        joined = "\n".join(lines[:30])
        m = re.search(r'"""(.*?)"""', joined, flags=re.DOTALL)
        if m:
            s = m.group(1).strip().splitlines()[0].strip()
            return s[:200]
        m2 = re.search(r"'''(.*?)'''", joined, flags=re.DOTALL)
        if m2:
            s = m2.group(1).strip().splitlines()[0].strip()
            return s[:200]

    for ln in lines[:15]:
        if ln.startswith("#") or ln.startswith("::") or ln.startswith("REM") or ln.startswith("//"):
            s = ln.lstrip("#/:REM ").strip()
            if s:
                return s[:200]
    return lines[0][:200]


def tag_by_path(rel_path: str) -> List[str]:
    """基于路径/文件名做粗标签（便于分组）。"""
    p = rel_path.replace("\\", "/").lower()
    name = Path(p).name
    tags = []

    keys = [
        ("projection", ["proj", "projection", "crs", "wgs84", "utm", "rect", "roundtrip"]),
        ("crosswalk", ["crosswalk", "zebra", "ped", "walk"]),
        ("lidar", ["lidar", "velo", "velodyne", "las", "laz", "pointcloud"]),
        ("image", ["image", "camera", "rgb", "overlay"]),
        ("sat", ["sat", "dop", "aerial"]),
        ("osm", ["osm"]),
        ("topo", ["topo"]),
        ("geom", ["geom"]),
        ("qa", ["qa", "validate", "check", "smoke", "report", "summary"]),
        ("download", ["download", "fetch"]),
        ("run", ["run_", "runner", "pipeline"]),
        ("debug", ["debug", "trace"]),
        ("tool", ["tools/"]),
        ("script", ["scripts/"]),
    ]
    for t, kws in keys:
        for kw in kws:
            if kw in p or kw in name:
                tags.append(t)
                break
    # 去重保持顺序
    seen = set()
    out = []
    for x in tags:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def parse_cmd_edges(rel_path: str, text: str) -> List[Tuple[str, str, str]]:
    """
    从 cmd 文本提取依赖边：
    - call xxx.cmd
    - python xxx.py
    - .venv\\Scripts\\python.exe xxx.py
    返回：(from, to, kind)
    """
    edges = []
    lines = text.splitlines()
    for ln in lines:
        s = ln.strip()
        if not s or s.lower().startswith("rem"):
            continue

        # call
        m = re.search(r"(?i)\bcall\s+([^\s]+)", s)
        if m:
            tgt = m.group(1).strip().strip('"')
            edges.append((rel_path, tgt, "call"))
            continue

        # python
        if re.search(r"(?i)\bpython(\.exe)?\b", s) or ("Scripts\\python.exe" in s) or ("Scripts/python.exe" in s):
            # 取第一个 .py 作为目标
            m2 = re.search(r"([A-Za-z0-9_\-./\\]+\.py)", s)
            if m2:
                tgt = m2.group(1).strip()
                edges.append((rel_path, tgt, "python"))
    return edges


def copy_to_snapshot(src: Path, dst: Path) -> None:
    """拷贝文件到 snapshot（保留目录结构）。"""
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))


def build_legacy_index_md(records: List[Dict[str, Any]]) -> str:
    """生成 LEGACY_INDEX.md（按标签分组的人读摘要）。"""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        for t in r.get("tags", []):
            groups.setdefault(t, []).append(r)

    # 排序：组按数量降序
    group_items = sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))

    lines = []
    lines.append("# Legacy Index\n")
    lines.append("本文件由 tools/archive_legacy_entrypoints.py 自动生成。\n")
    lines.append("目标：让后续 Skill 化迁移有清晰的“入口地图”和粗分类。\n")

    total = len(records)
    cmd_n = sum(1 for r in records if r["ext"] in CMD_EXTS)
    py_n = sum(1 for r in records if r["ext"] == ".py")
    py_entry_n = sum(1 for r in records if r["ext"] == ".py" and r.get("py_entrypoint"))

    lines.append("## Summary\n")
    lines.append(f"- total: {total}")
    lines.append(f"- cmd: {cmd_n}")
    lines.append(f"- py: {py_n} (entrypoints={py_entry_n})\n")

    lines.append("## Groups\n")
    for tag, items in group_items:
        lines.append(f"### {tag} ({len(items)})\n")
        # 每组只列前若干，避免太长
        items_sorted = sorted(items, key=lambda x: (x.get("rel_path", "")))
        for r in items_sorted[:80]:
            mark = ""
            if r["ext"] == ".py" and r.get("py_entrypoint"):
                mark = " [py-entry]"
            if r["ext"] in CMD_EXTS:
                mark = " [cmd]"
            lines.append(f"- `{r['rel_path']}`{mark} — {r.get('summary','')}")
        if len(items_sorted) > 80:
            lines.append(f"- ... ({len(items_sorted)-80} more)")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(REPO_ROOT)
    if not repo_root.exists():
        raise FileNotFoundError(f"REPO_ROOT 不存在：{repo_root}")

    legacy_root = repo_root / "legacy"
    ensure_dir(legacy_root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = legacy_root / f"snapshot_{ts}"
    if snapshot_dir.exists() and OVERWRITE:
        shutil.rmtree(str(snapshot_dir), ignore_errors=True)
    ensure_dir(snapshot_dir)

    orig_dir = snapshot_dir / "orig"
    logs_dir = snapshot_dir / "logs"
    ensure_dir(orig_dir)
    ensure_dir(logs_dir)

    logger = setup_logger(logs_dir / "run.log")
    logger.info("=== Legacy Vault 归档开始 ===")
    logger.info(f"repo_root = {repo_root}")
    logger.info(f"snapshot_dir = {snapshot_dir}")
    logger.info(f"ARCHIVE_PY_MODE = {ARCHIVE_PY_MODE}")

    # git 快照（可选）
    git_branch = ""
    git_commit = ""
    code, out = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if code == 0:
        git_branch = out
    code, out = run_cmd(["git", "rev-parse", "HEAD"], repo_root)
    if code == 0:
        git_commit = out

    records: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str, str]] = []

    # 扫描
    for root, dirnames, filenames in os.walk(repo_root):
        # 过滤目录
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]

        for fn in filenames:
            p = Path(root) / fn
            ext = p.suffix.lower()

            if INCLUDE_CMD and ext in CMD_EXTS:
                pass
            elif INCLUDE_PY and ext == ".py":
                pass
            else:
                continue

            rel_path = str(p.relative_to(repo_root)).replace("\\", "/")
            head = read_head_text(p, MAX_BYTES_FOR_HEAD)
            summary = extract_summary(ext, head)
            tags = tag_by_path(rel_path)

            stat = p.stat()
            size_bytes = int(stat.st_size)
            mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")

            py_entry = False
            if ext == ".py":
                py_entry = is_python_entrypoint(rel_path, head)

            # 决定是否拷贝归档
            copied = False
            if ext in CMD_EXTS:
                copied = True
            elif ext == ".py":
                copied = py_entry  # entrypoints 模式默认只拷入口 py
                if ARCHIVE_PY_MODE == "tools_scripts":
                    copied = rel_path.lower().startswith("tools/") or rel_path.lower().startswith("scripts/")
                if ARCHIVE_PY_MODE == "all":
                    copied = True

            # hash
            try:
                h = sha1_partial(p)
            except Exception:
                h = ""

            rec = {
                "rel_path": rel_path,
                "ext": ext,
                "size_bytes": size_bytes,
                "mtime": mtime,
                "sha1_head2mb": h,
                "copied": copied,
                "py_entrypoint": py_entry if ext == ".py" else "",
                "summary": summary,
                "tags": tags,
            }
            records.append(rec)

            # cmd edges
            if ext in CMD_EXTS:
                try:
                    full_text = p.read_text(encoding="utf-8", errors="ignore")
                    edges.extend(parse_cmd_edges(rel_path, full_text))
                except Exception:
                    pass

            # copy
            if copied:
                dst = orig_dir / rel_path
                try:
                    copy_to_snapshot(p, dst)
                except Exception as e:
                    logger.warning(f"拷贝失败：{rel_path} -> {e}")

    # 写出 catalog
    catalog_csv = snapshot_dir / "catalog.csv"
    catalog_json = snapshot_dir / "catalog.json"
    deps_csv = snapshot_dir / "deps_cmd_edges.csv"
    index_md = snapshot_dir / "LEGACY_INDEX.md"
    manifest_json = snapshot_dir / "manifest.json"

    # CSV
    fields = ["rel_path", "ext", "copied", "py_entrypoint", "size_bytes", "mtime", "sha1_head2mb", "summary", "tags"]
    with catalog_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = r.copy()
            row["tags"] = "|".join(r.get("tags", []))
            w.writerow({k: row.get(k, "") for k in fields})

    catalog_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # deps edges
    with deps_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from", "to", "kind"])
        for a, b, k in edges:
            w.writerow([a, b, k])

    # index md
    index_md.write_text(build_legacy_index_md(records), encoding="utf-8")

    # manifest
    stats = {
        "total": len(records),
        "cmd": sum(1 for r in records if r["ext"] in CMD_EXTS),
        "py": sum(1 for r in records if r["ext"] == ".py"),
        "py_entrypoints": sum(1 for r in records if r["ext"] == ".py" and r.get("py_entrypoint")),
        "copied": sum(1 for r in records if r.get("copied")),
        "edges": len(edges),
    }
    manifest = {
        "snapshot": snapshot_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "git": {"branch": git_branch, "commit": git_commit},
        "params": {"ARCHIVE_PY_MODE": ARCHIVE_PY_MODE},
        "stats": stats,
        "outputs": {
            "catalog_csv": str(catalog_csv.relative_to(snapshot_dir)),
            "catalog_json": str(catalog_json.relative_to(snapshot_dir)),
            "deps_cmd_edges_csv": str(deps_csv.relative_to(snapshot_dir)),
            "legacy_index_md": str(index_md.relative_to(snapshot_dir)),
        },
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # active pointer
    (legacy_root / "ACTIVE_SNAPSHOT.txt").write_text(str(snapshot_dir), encoding="utf-8")

    logger.info("=== 完成 ===")
    logger.info(f"stats: {stats}")
    logger.info(f"ACTIVE_SNAPSHOT: {legacy_root / 'ACTIVE_SNAPSHOT.txt'}")


if __name__ == "__main__":
    main()
