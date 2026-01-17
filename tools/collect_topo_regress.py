import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


def read_json_from_markdown(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"TopoSummary JSON not found in {path}")
    return json.loads(m.group(0))


def read_index(index_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect topo regression summaries.")
    ap.add_argument("--regress-dir", required=True, help="regression run directory")
    ap.add_argument(
        "--index-file",
        default="regress_index.jsonl",
        help="index filename under regress-dir (default: regress_index.jsonl)",
    )
    ap.add_argument(
        "--gate-config",
        default="configs/topo_regress_gate.yaml",
        help="gate config path (optional for report summary)",
    )
    ap.add_argument(
        "--baseline",
        default="",
        help="baseline yaml path (optional for delta columns)",
    )
    return ap.parse_args()


def load_gate_config(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def load_baseline(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    drives = data.get("drives", [])
    if not isinstance(drives, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in drives:
        if not isinstance(item, dict):
            continue
        drive = item.get("drive")
        if drive:
            out[str(drive)] = item
    return out


def summarize_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for item in items:
        status = item.get("status", "")
        if status in counts:
            counts[status] += 1
    return counts


def main() -> int:
    args = parse_args()
    regress_dir = Path(args.regress_dir)
    if not regress_dir.exists():
        raise SystemExit(f"ERROR: regress dir not found: {regress_dir}")

    index_path = regress_dir / args.index_file
    if not index_path.exists():
        raise SystemExit(f"ERROR: index not found: {index_path}")

    items = read_index(index_path)
    counts = summarize_counts(items)
    gate_cfg = load_gate_config(Path(args.gate_config))
    baseline_map = load_baseline(Path(args.baseline)) if args.baseline else {}

    pass_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []
    action_types: set[str] = set()

    for item in items:
        status = item.get("status")
        if status == "PASS":
            summary_path = item.get("summary_path")
            summary = {}
            if summary_path:
                summary_file = Path(summary_path)
                if summary_file.exists():
                    summary = read_json_from_markdown(summary_file)
            actions_by_type = (
                summary.get("actions_by_type")
                if isinstance(summary.get("actions_by_type"), dict)
                else {}
            )
            action_types.update(actions_by_type.keys())
            drive = item.get("drive")
            baseline_item = baseline_map.get(drive, {})
            delta_max_degree = None
            delta_issues_count = None
            delta_actions_count = None
            if baseline_item:
                if summary.get("max_degree") is not None and baseline_item.get("max_degree") is not None:
                    delta_max_degree = summary.get("max_degree") - baseline_item.get("max_degree")
                if summary.get("issues_count") is not None and baseline_item.get("issues_count") is not None:
                    delta_issues_count = summary.get("issues_count") - baseline_item.get("issues_count")
                if summary.get("actions_count") is not None and baseline_item.get("actions_count") is not None:
                    delta_actions_count = summary.get("actions_count") - baseline_item.get("actions_count")

            pass_rows.append(
                {
                    "drive": drive,
                    "run_id": summary.get("run_id") or item.get("run_id"),
                    "node_count_post_prune": summary.get("node_count_post_prune", summary.get("node_count")),
                    "max_degree": summary.get("max_degree"),
                    "dangling_detected": summary.get("dangling_detected"),
                    "dangling_removed": summary.get("dangling_removed"),
                    "dangling_merged": summary.get("dangling_merged"),
                    "dangling_unfixed": summary.get("dangling_unfixed"),
                    "issues_count": summary.get("issues_count"),
                    "actions_count": summary.get("actions_count"),
                    "delta_max_degree": delta_max_degree,
                    "delta_issues_count": delta_issues_count,
                    "delta_actions_count": delta_actions_count,
                    "actions_by_type": actions_by_type,
                }
            )
        elif status == "SKIPPED":
            skipped_rows.append(
                {"drive": item.get("drive"), "reason": item.get("reason")}
            )
        elif status == "FAIL":
            fail_rows.append(
                {"drive": item.get("drive"), "reason": item.get("reason")}
            )

    action_cols = [f"actions_{k}" for k in sorted(action_types)]
    delta_cols = []
    if baseline_map:
        delta_cols = ["delta_max_degree", "delta_issues_count", "delta_actions_count"]
    headers = [
        "drive",
        "run_id",
        "node_count_post_prune",
        "max_degree",
        "dangling_detected",
        "dangling_removed",
        "dangling_merged",
        "dangling_unfixed",
        "issues_count",
        "actions_count",
    ] + delta_cols + action_cols

    out_csv = regress_dir / "TopoRegress.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in pass_rows:
            out_row = dict(row)
            if not delta_cols:
                out_row.pop("delta_max_degree", None)
                out_row.pop("delta_issues_count", None)
                out_row.pop("delta_actions_count", None)
            actions_by_type = out_row.pop("actions_by_type", {}) or {}
            for col in action_cols:
                key = col.replace("actions_", "")
                out_row[col] = actions_by_type.get(key)
            writer.writerow(out_row)

    out_md = regress_dir / "TopoRegress.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Topo Regress Report\n\n")
        f.write("## Summary\n\n")
        f.write(
            f"- PASS: {counts['PASS']}  FAIL: {counts['FAIL']}  SKIPPED: {counts['SKIPPED']}\n"
        )
        if gate_cfg:
            f.write("\n- Gate config:\n")
            for k, v in gate_cfg.items():
                f.write(f"  - {k}: {v}\n")
        if baseline_map:
            f.write("\n- Baseline: provided\n")

        f.write("\n## PASS Drives\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in pass_rows:
            actions_by_type = row.get("actions_by_type", {}) or {}
            vals = []
            for h in headers:
                if h.startswith("actions_"):
                    key = h.replace("actions_", "")
                    vals.append(str(actions_by_type.get(key, "")))
                else:
                    val = row.get(h, "")
                    vals.append("" if val is None else str(val))
            f.write("| " + " | ".join(vals) + " |\n")

        if skipped_rows:
            f.write("\n## SKIPPED Drives\n\n")
            f.write("| drive | reason |\n")
            f.write("|---|---|\n")
            for row in skipped_rows:
                f.write(f"| {row.get('drive','')} | {row.get('reason','')} |\n")

        if fail_rows:
            f.write("\n## FAIL Drives\n\n")
            f.write("| drive | reason |\n")
            f.write("|---|---|\n")
            for row in fail_rows:
                f.write(f"| {row.get('drive','')} | {row.get('reason','')} |\n")

        if action_cols:
            f.write("\n## Actions By Type (per drive)\n\n")
            for row in pass_rows:
                f.write(f"- {row.get('drive')}: {row.get('actions_by_type', {})}\n")

    print(f"[REGRESS] report -> {out_md}")
    print(f"[REGRESS] report -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
