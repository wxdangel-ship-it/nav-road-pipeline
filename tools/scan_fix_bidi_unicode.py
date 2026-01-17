# -*- coding: utf-8 -*-
"""
功能：
    扫描并（可选）自动清理 Git 仓库中可能触发 GitHub “hidden or bidirectional Unicode text” 提示的字符：
    - 双向文本控制字符（Bidi controls）: U+202A..U+202E, U+2066..U+2069, U+200E, U+200F, U+061C 等
    -（可选）异常 BOM：U+FEFF 若出现在文件中间也会被视为可疑并清理（文件开头 BOM 默认保留）

输入：
    - Git 仓库（默认从本脚本所在目录向上定位 git root）
    - 仅扫描 git 已跟踪文件（git ls-files）

输出：
    - 终端日志（INFO/WARNING/ERROR）
    - 报告目录：runs/bidi_scan_<timestamp>/
        - report.json  机器可读
        - report.md    人类可读

使用：
    1) 放到 tools/scan_fix_bidi_unicode.py
    2) 在 PyCharm 直接运行
    3) 如果想先只扫描不修复，把 AUTO_FIX 改成 False

注意：
    - 默认会为被修复的文件生成备份：<filename>.bidi.bak（可在参数区关闭）
    - 修复仅删除“Bidi 控制字符”与“文件中间的 BOM”，不会动普通中文/emoji 等内容
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# =========================
# 参数区（可编辑）
# =========================
AUTO_FIX = True                 # True=自动修复；False=仅扫描
CREATE_BACKUP = True            # 自动修复时是否生成 .bak 备份
BACKUP_SUFFIX = ".bidi.bak"

# 仅扫描 git 已跟踪文件；如果 git 不可用，会回退到 os.walk 扫描
ONLY_GIT_TRACKED = True

# 修复范围：只修复 Bidi 控制字符 + “中间 BOM”
FIX_BIDI_CONTROLS = True
FIX_MID_FILE_BOM = True

# 过滤：最大文件大小（MB），超过跳过
MAX_FILE_MB = 10

# 回退 os.walk 时的排除目录
EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "dist", "build", ".venv", "venv",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "runs",  # 一般不建议扫生成物（如果你想扫也可以删掉这条）
}

# 输出报告目录（默认写到仓库 runs/ 下）
REPORT_PARENT_DIRNAME = "runs"
REPORT_DIR_PREFIX = "bidi_scan_"
# =========================


# GitHub 常见触发 “bidirectional Unicode text” 的控制字符集合
BIDI_CONTROL_CODEPOINTS: Dict[int, str] = {
    0x061C: "ARABIC LETTER MARK",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202B: "RIGHT-TO-LEFT EMBEDDING",
    0x202C: "POP DIRECTIONAL FORMATTING",
    0x202D: "LEFT-TO-RIGHT OVERRIDE",
    0x202E: "RIGHT-TO-LEFT OVERRIDE",
    0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2067: "RIGHT-TO-LEFT ISOLATE",
    0x2068: "FIRST STRONG ISOLATE",
    0x2069: "POP DIRECTIONAL ISOLATE",
}

BOM_CODEPOINT = 0xFEFF  # ZERO WIDTH NO-BREAK SPACE / BOM


@dataclass
class Finding:
    file: str                 # 相对仓库根目录
    line: int
    col: int
    codepoint: str            # e.g., "U+202E"
    name: str                 # e.g., "RIGHT-TO-LEFT OVERRIDE"
    category: str             # "bidi" | "bom"
    line_preview: str         # 将不可见字符替换为 [U+XXXX] 后的可读行片段


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("bidi_scan")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def detect_git_root(start_dir: str, logger: logging.Logger) -> Optional[str]:
    code, out, err = run_cmd(["git", "rev-parse", "--show-toplevel"], cwd=start_dir)
    if code != 0:
        logger.warning("未能定位 git root（将回退到 os.walk 扫描）。原因：%s", err or "unknown")
        return None
    return out


def list_git_tracked_files(repo_root: str, logger: logging.Logger) -> List[str]:
    # 用 -z 以避免空格/特殊字符路径解析问题
    p = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        logger.warning("git ls-files 失败（将回退到 os.walk 扫描）。原因：%s", p.stderr.decode("utf-8", "replace"))
        return []
    raw = p.stdout.split(b"\x00")
    files = [x.decode("utf-8", "replace") for x in raw if x]
    return files


def is_binary_like(data: bytes) -> bool:
    # 简单判定：包含 NUL 基本可视为二进制
    return b"\x00" in data[:4096]


def read_text_utf8(path: str) -> Optional[str]:
    # 只做 utf-8 尝试，失败则跳过（避免误修复非 utf-8 文件导致乱码 diff）
    with open(path, "rb") as f:
        data = f.read()
    if is_binary_like(data):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def make_visible_line(line: str) -> str:
    # 将可疑字符替换为 [U+XXXX]，方便人眼审阅
    out_chars: List[str] = []
    for ch in line:
        cp = ord(ch)
        if cp in BIDI_CONTROL_CODEPOINTS or cp == BOM_CODEPOINT:
            out_chars.append(f"[U+{cp:04X}]")
        else:
            out_chars.append(ch)
    s = "".join(out_chars)
    # 控制输出长度，避免超长行刷屏
    if len(s) > 240:
        return s[:240] + " …"
    return s


def scan_text(rel_path: str, text: str) -> List[Finding]:
    findings: List[Finding] = []
    lines = text.splitlines(keepends=False)
    for li, line in enumerate(lines, start=1):
        for ci, ch in enumerate(line, start=1):
            cp = ord(ch)
            if cp in BIDI_CONTROL_CODEPOINTS:
                findings.append(
                    Finding(
                        file=rel_path,
                        line=li,
                        col=ci,
                        codepoint=f"U+{cp:04X}",
                        name=BIDI_CONTROL_CODEPOINTS.get(cp) or unicodedata.name(ch, "UNKNOWN"),
                        category="bidi",
                        line_preview=make_visible_line(line),
                    )
                )
            elif cp == BOM_CODEPOINT:
                # 文件开头的 BOM 通常无害；这里先标注为可疑，后续修复策略会只修复“中间 BOM”
                findings.append(
                    Finding(
                        file=rel_path,
                        line=li,
                        col=ci,
                        codepoint="U+FEFF",
                        name="BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE",
                        category="bom",
                        line_preview=make_visible_line(line),
                    )
                )
    return findings


def should_fix_char(cp: int, absolute_index: int) -> bool:
    if FIX_BIDI_CONTROLS and cp in BIDI_CONTROL_CODEPOINTS:
        return True
    if FIX_MID_FILE_BOM and cp == BOM_CODEPOINT:
        # 仅修复“文件中间”的 BOM；absolute_index==0 认为是文件开头 BOM，保留
        return absolute_index != 0
    return False


def apply_fix(text: str) -> Tuple[str, int]:
    fixed_chars: List[str] = []
    removed = 0
    for idx, ch in enumerate(text):
        cp = ord(ch)
        if should_fix_char(cp, idx):
            removed += 1
            continue
        fixed_chars.append(ch)
    return "".join(fixed_chars), removed


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fallback_walk_files(repo_root: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # 原地剪枝
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            out.append(os.path.join(dirpath, fn))
    return out


def main() -> int:
    logger = setup_logger()
    script_dir = os.path.abspath(os.path.dirname(__file__))

    repo_root = detect_git_root(script_dir, logger)
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(script_dir, ".."))
        logger.info("使用回退 repo_root：%s", repo_root)
        use_git = False
    else:
        logger.info("定位到 git root：%s", repo_root)
        use_git = True

    # 报告目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root = os.path.join(repo_root, REPORT_PARENT_DIRNAME, f"{REPORT_DIR_PREFIX}{ts}")
    ensure_dir(report_root)
    report_json = os.path.join(report_root, "report.json")
    report_md = os.path.join(report_root, "report.md")

    # 文件列表
    file_paths: List[str] = []
    rel_paths: List[str] = []

    if ONLY_GIT_TRACKED and use_git:
        rel_paths = list_git_tracked_files(repo_root, logger)
        for rp in rel_paths:
            ap = os.path.join(repo_root, rp)
            if os.path.isfile(ap):
                file_paths.append(ap)
    else:
        # 回退：walk 全目录（会排除 runs/node_modules 等）
        file_paths = fallback_walk_files(repo_root)
        rel_paths = [os.path.relpath(p, repo_root) for p in file_paths]

    logger.info("待扫描文件数：%d", len(file_paths))

    all_findings: List[Finding] = []
    skipped_non_utf8 = 0
    skipped_large = 0
    scanned = 0
    fixed_files = 0
    fixed_chars_total = 0

    # 扫描 + 可选修复
    for ap, rp in zip(file_paths, rel_paths):
        try:
            size_mb = os.path.getsize(ap) / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                skipped_large += 1
                continue

            text = read_text_utf8(ap)
            if text is None:
                skipped_non_utf8 += 1
                continue

            scanned += 1
            findings = scan_text(rp, text)

            # 过滤：BOM 只在“文件开头”出现则不算问题（不影响 GitHub 警告，也不修）
            # 这里仍然记录 BOM，但在汇总时单独区分；修复时只会修复中间 BOM
            has_bidi = any(f.category == "bidi" for f in findings)
            has_mid_bom = False
            # 判断是否存在中间 BOM：在原始 text 中查找 FEFF 且不在 index 0
            mid_bom_pos = text.find("\ufeff")
            if mid_bom_pos > 0:
                has_mid_bom = True

            if findings:
                all_findings.extend(findings)

            if AUTO_FIX and (has_bidi or has_mid_bom):
                fixed_text, removed = apply_fix(text)
                if removed > 0 and fixed_text != text:
                    if CREATE_BACKUP:
                        bak = ap + BACKUP_SUFFIX
                        if not os.path.exists(bak):
                            with open(bak, "wb") as fb:
                                fb.write(text.encode("utf-8"))
                    with open(ap, "wb") as fw:
                        fw.write(fixed_text.encode("utf-8"))
                    fixed_files += 1
                    fixed_chars_total += removed

        except Exception as e:
            logger.error("处理失败：%s | %s: %s", rp, type(e).__name__, str(e))

    # 汇总统计
    by_cp = {}
    for f in all_findings:
        by_cp[f.codepoint] = by_cp.get(f.codepoint, 0) + 1

    files_with_bidi = sorted({f.file for f in all_findings if f.category == "bidi"})
    files_with_mid_bom = []
    # 粗略：如果 report 里出现 BOM，且不是第一行第一列也可能是中间 BOM，但更可靠的判断已在修复时做
    # 这里为了报告可读性，仍列一下出现过 BOM 的文件
    files_with_bom = sorted({f.file for f in all_findings if f.category == "bom"})
    files_with_mid_bom = files_with_bom  # 报告层面不再深区分

    summary = {
        "repo_root": repo_root,
        "scanned_files": scanned,
        "skipped_non_utf8_or_binary": skipped_non_utf8,
        "skipped_large_files": skipped_large,
        "total_findings": len(all_findings),
        "findings_by_codepoint": by_cp,
        "files_with_bidi_controls": files_with_bidi,
        "files_with_bom": files_with_bom,
        "auto_fix": AUTO_FIX,
        "fixed_files": fixed_files,
        "fixed_chars_total": fixed_chars_total,
        "report_dir": os.path.relpath(report_root, repo_root),
    }

    # 写 report.json
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "findings": [asdict(x) for x in all_findings],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 写 report.md
    md_lines: List[str] = []
    md_lines.append(f"# Bidi/Hidden Unicode 扫描报告\n")
    md_lines.append(f"- repo_root: `{repo_root}`")
    md_lines.append(f"- scanned_files: **{scanned}**")
    md_lines.append(f"- skipped_non_utf8_or_binary: **{skipped_non_utf8}**")
    md_lines.append(f"- skipped_large_files(>{MAX_FILE_MB}MB): **{skipped_large}**")
    md_lines.append(f"- total_findings: **{len(all_findings)}**")
    md_lines.append(f"- auto_fix: **{AUTO_FIX}**")
    md_lines.append(f"- fixed_files: **{fixed_files}**")
    md_lines.append(f"- fixed_chars_total: **{fixed_chars_total}**\n")

    md_lines.append("## 按 codepoint 统计\n")
    if by_cp:
        for cp, cnt in sorted(by_cp.items(), key=lambda x: (-x[1], x[0])):
            md_lines.append(f"- {cp}: **{cnt}**")
    else:
        md_lines.append("- （无）")

    md_lines.append("\n## 涉及文件（bidi controls）\n")
    if files_with_bidi:
        for fp in files_with_bidi:
            md_lines.append(f"- `{fp}`")
    else:
        md_lines.append("- （无）")

    md_lines.append("\n## 发现明细（前 200 条）\n")
    if all_findings:
        for i, fnd in enumerate(all_findings[:200], start=1):
            md_lines.append(
                f"{i}. `{fnd.file}` L{fnd.line}:C{fnd.col} {fnd.codepoint} {fnd.name} "
                f"({fnd.category})\n    - {fnd.line_preview}"
            )
        if len(all_findings) > 200:
            md_lines.append(f"\n（其余 {len(all_findings) - 200} 条略）")
    else:
        md_lines.append("- （无）")

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # 终端输出
    logger.info("扫描完成：scanned=%d, findings=%d", scanned, len(all_findings))
    logger.info("报告已写入：%s", report_root)
    if AUTO_FIX:
        logger.info("自动修复：fixed_files=%d, removed_chars=%d", fixed_files, fixed_chars_total)
        if fixed_files > 0:
            logger.info("建议执行：git diff 复核改动；确认无误后再 commit/push 更新 PR。")

    # 若存在 bidi controls（真正触发 GitHub 警告的）则返回 1 方便 CI/脚本判定
    has_bidi_any = any(f.category == "bidi" for f in all_findings)
    return 1 if has_bidi_any and not AUTO_FIX else 0


if __name__ == "__main__":
    raise SystemExit(main())
