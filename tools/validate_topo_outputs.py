# -*- coding: utf-8 -*-
"""
功能：
    校验 build_topo 输出的 TopoSummary.md 与 TopoIssues（JSONL/NDJSON）结构一致性，
    并输出统计报告（issues 行数、字段缺失、重复 issue_id、Summary 口径自洽性等）。

输入：
    - TopoSummary.md（内容应为 JSON 或 markdown 包裹的 JSON）
    - TopoIssues（每行一个 JSON 对象，JSONL/NDJSON）

输出：
    - 终端报告
    - 进程退出码：0=通过（无致命错误）；1=存在致命错误（JSON 解析失败/缺字段/重复ID等）

使用：
    直接在 PyCharm 运行本脚本即可（无需命令行参数）。
"""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# =========================
# 参数区（按需修改）
# =========================
SUMMARY_PATH = r"runs\topo_20260117_073912\outputs\TopoSummary.md"
ISSUES_PATH = r"runs\topo_20260117_073912\outputs\TopoIssues.jsonl"  # 按实际文件名改

# TOPOISSUE_SPEC（你们真实 spec 如有更严格要求，可在这里补充）
REQUIRED_ISSUE_KEYS = [
    "issue_id",
    "tile_id",
    "involved_edges",
    "involved_nodes",
    "rule_failed",
    "severity",
    "description",
    "recommend_actions",
    "evidence_summary",
    "error_code",
]

ALLOWED_SEVERITY = {"S0", "S1", "S2", "S3"}
# =========================


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("validate_topo_outputs")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def read_json_from_markdown(path: str) -> Dict[str, Any]:
    """从 TopoSummary.md 中提取 JSON 对象并解析。"""
    text = open(path, "r", encoding="utf-8").read().strip()

    # 允许文件就是纯 JSON
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # 允许 markdown 包裹：抓取第一个 { 到最后一个 }
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("TopoSummary.md 中未找到 JSON 对象（未匹配到 {...}）")
    return json.loads(m.group(0))


def iter_jsonl(path: str) -> Tuple[int, List[Dict[str, Any]], List[str]]:
    """读取 JSONL/NDJSON：每行一个 JSON。返回（行数，有效对象列表，解析失败的行摘要）。"""
    objs: List[Dict[str, Any]] = []
    bad_lines: List[str] = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    bad_lines.append(f"line{total}: not a json object")
                    continue
                objs.append(obj)
            except Exception as e:
                bad_lines.append(f"line{total}: {type(e).__name__}: {str(e)[:120]}")
    return total, objs, bad_lines


def validate_summary(summary: Dict[str, Any], logger: logging.Logger) -> List[str]:
    """校验 Summary 自洽性，返回 warnings。"""
    warns: List[str] = []

    node_count = summary.get("node_count")
    isolated_removed = summary.get("isolated_nodes_removed")
    deg_hist = summary.get("degree_histogram", {})
    comp = summary.get("components", {})

    # degree_histogram 合计
    if isinstance(deg_hist, dict) and deg_hist:
        try:
            deg_total = sum(int(v) for v in deg_hist.values())
            if isinstance(node_count, int) and isinstance(isolated_removed, int):
                if deg_total not in (node_count, node_count + isolated_removed):
                    warns.append(f"degree_histogram 合计={deg_total} 与 node_count={node_count} / +isolated_removed={node_count+isolated_removed} 均不匹配")
            elif isinstance(node_count, int) and deg_total != node_count:
                warns.append(f"degree_histogram 合计={deg_total} 与 node_count={node_count} 不匹配（且 isolated_nodes_removed 缺失/非整数）")
        except Exception:
            warns.append("degree_histogram 解析失败（值非整数？）")

    # components sizes 合计
    if isinstance(comp, dict) and "sizes" in comp:
        sizes = comp.get("sizes", [])
        if isinstance(sizes, list) and sizes:
            try:
                size_total = sum(int(x) for x in sizes)
                if isinstance(node_count, int) and isinstance(isolated_removed, int):
                    if size_total not in (node_count, node_count + isolated_removed):
                        warns.append(f"components.sizes 合计={size_total} 与 node_count={node_count} / +isolated_removed={node_count+isolated_removed} 均不匹配")
                elif isinstance(node_count, int) and size_total != node_count:
                    warns.append(f"components.sizes 合计={size_total} 与 node_count={node_count} 不匹配（且 isolated_nodes_removed 缺失/非整数）")
            except Exception:
                warns.append("components.sizes 解析失败（值非整数？）")

    # dangling_nodes 与 dangling_total
    dang_nodes = summary.get("dangling_nodes", [])
    dang_total = summary.get("dangling_total")
    if isinstance(dang_nodes, list) and isinstance(dang_total, int):
        if len(dang_nodes) != dang_total:
            warns.append(f"dangling_nodes 数量={len(dang_nodes)} 与 dangling_total={dang_total} 不一致（确认口径：final vs detected）")

    # dangling_removed > dangling_total 提示
    dang_detected = summary.get("dangling_detected")
    dang_removed = summary.get("dangling_removed")
    dang_merged = summary.get("dangling_merged")
    dang_unfixed = summary.get("dangling_unfixed")

    if isinstance(dang_detected, int) and all(isinstance(x, int) for x in [dang_removed, dang_merged, dang_unfixed]):
        s = dang_removed + dang_merged + dang_unfixed
        if dang_detected != s:
            warns.append(f"dangling invariant 不成立：detected={dang_detected} != removed+merged+unfixed={s}")
    else:
        # 仅在没有 dangling_detected 的旧版本里，才保留 removed > total 的提醒
        if isinstance(summary.get("dangling_total"), int) and isinstance(dang_removed, int) and dang_removed > summary[
            "dangling_total"]:
            warns.append(...)

    if warns:
        logger.warning("TopoSummary 自洽性存在 %d 条提醒：", len(warns))
        for w in warns:
            logger.warning("  - %s", w)
    else:
        logger.info("TopoSummary 自洽性检查：未发现明显问题")

    return warns


def validate_issues(objs: List[Dict[str, Any]], logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """校验 issues 必填字段/枚举/重复ID；返回（errors, warnings）。"""
    errors: List[str] = []
    warns: List[str] = []

    ids = [o.get("issue_id") for o in objs]
    dup = [k for k, c in Counter(ids).items() if k and c > 1]
    if dup:
        errors.append(f"存在重复 issue_id：{dup[:10]}{'...' if len(dup)>10 else ''}")

    missing_counter = Counter()
    sev_bad = 0
    for i, o in enumerate(objs, start=1):
        for k in REQUIRED_ISSUE_KEYS:
            if k not in o:
                missing_counter[k] += 1
        sev = o.get("severity")
        if sev is not None and sev not in ALLOWED_SEVERITY:
            sev_bad += 1

        # 基本类型检查（轻量）
        if "involved_edges" in o and not isinstance(o["involved_edges"], list):
            warns.append(f"issue #{i} involved_edges 非 list")
        if "involved_nodes" in o and not isinstance(o["involved_nodes"], list):
            warns.append(f"issue #{i} involved_nodes 非 list")

    for k, c in missing_counter.items():
        errors.append(f"字段缺失：{k} 缺失 {c} 次")

    if sev_bad > 0:
        warns.append(f"severity 非法枚举 {sev_bad} 条（允许：{sorted(ALLOWED_SEVERITY)}）")

    # 统计
    by_rule = Counter(o.get("rule_failed", "UNKNOWN") for o in objs)
    by_sev = Counter(o.get("severity", "UNKNOWN") for o in objs)
    logger.info("TopoIssues 统计：总数=%d", len(objs))
    logger.info("  - 按 rule_failed：%s", dict(by_rule))
    logger.info("  - 按 severity：%s", dict(by_sev))

    if errors:
        logger.error("TopoIssues 结构存在 %d 条致命问题：", len(errors))
        for e in errors:
            logger.error("  - %s", e)
    if warns:
        logger.warning("TopoIssues 结构存在 %d 条提醒：", len(warns))
        for w in warns[:20]:
            logger.warning("  - %s", w)
        if len(warns) > 20:
            logger.warning("  - ...（其余 %d 条略）", len(warns) - 20)

    return errors, warns


def main() -> int:
    logger = setup_logger()

    if not os.path.exists(SUMMARY_PATH):
        logger.error("TopoSummary 不存在：%s", SUMMARY_PATH)
        return 1
    if not os.path.exists(ISSUES_PATH):
        logger.error("TopoIssues 不存在：%s", ISSUES_PATH)
        return 1

    # 1) Summary
    try:
        summary = read_json_from_markdown(SUMMARY_PATH)
    except Exception as e:
        logger.error("TopoSummary 解析失败：%s: %s", type(e).__name__, str(e))
        return 1

    logger.info("TopoSummary 读取成功：run_id=%s", summary.get("run_id"))
    validate_summary(summary, logger)

    # 2) Issues
    total_lines, objs, bad_lines = iter_jsonl(ISSUES_PATH)
    if bad_lines:
        logger.error("TopoIssues JSONL 解析失败行数=%d（示例前10条）：%s", len(bad_lines), bad_lines[:10])
        return 1

    # 3) issues_count 对齐
    issues_count = summary.get("issues_count")
    if isinstance(issues_count, int) and issues_count != len(objs):
        logger.warning("Summary issues_count=%d 与 TopoIssues 实际条数=%d 不一致", issues_count, len(objs))

    errors, _warns = validate_issues(objs, logger)

    if errors:
        return 1

    logger.info("校验通过（无致命错误）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
