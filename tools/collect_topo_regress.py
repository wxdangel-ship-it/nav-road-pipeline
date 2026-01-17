import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def discover_from_dir(regress_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in regress_dir.rglob("TopoSummary.md"):
        items.append({"summary_path": str(path)})
    return items


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect topo regression summaries.")
    ap.add_argument("--regress-dir", required=True, help="regression run directory")
    ap.add_argument(
        "--index-file",
        default="regress_index.jsonl",
        help="index filename under regress-dir (default: regress_index.jsonl)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    regress_dir = Path(args.regress_dir)
    if not regress_dir.exists():
        raise SystemExit(f"ERROR: regress dir not found: {regress_dir}")

    index_path = regress_dir / args.index_file
    if index_path.exists():
        items = read_index(index_path)
    else:
        items = discover_from_dir(regress_dir)

    rows: List[Dict[str, Any]] = []
    action_types: set[str] = set()
    for item in items:
        summary_path = item.get("summary_path")
        topo_outputs = item.get("topo_outputs")
        if not summary_path and topo_outputs:
            summary_path = str(Path(topo_outputs) / "TopoSummary.md")
        if not summary_path:
            continue
        summary_file = Path(summary_path)
        if not summary_file.exists():
            continue
        summary = read_json_from_markdown(summary_file)
        actions_by_type = summary.get("actions_by_type") if isinstance(summary.get("actions_by_type"), dict) else {}
        action_types.update(actions_by_type.keys())

        rows.append(
            {
                "drive": item.get("drive"),
                "run_id": summary.get("run_id") or item.get("topo_run_id"),
                "node_count_post_prune": summary.get("node_count_post_prune", summary.get("node_count")),
                "max_degree": summary.get("max_degree"),
                "dangling_detected": summary.get("dangling_detected"),
                "dangling_removed": summary.get("dangling_removed"),
                "dangling_merged": summary.get("dangling_merged"),
                "dangling_unfixed": summary.get("dangling_unfixed"),
                "issues_count": summary.get("issues_count"),
                "actions_count": summary.get("actions_count"),
                "actions_by_type": actions_by_type,
            }
        )

    action_cols = [f"actions_{k}" for k in sorted(action_types)]
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
    ] + action_cols

    out_csv = regress_dir / "TopoRegress.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            actions_by_type = out_row.pop("actions_by_type", {}) or {}
            for col in action_cols:
                key = col.replace("actions_", "")
                out_row[col] = actions_by_type.get(key)
            writer.writerow(out_row)

    out_md = regress_dir / "TopoRegress.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
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

        if action_cols:
            f.write("\n## Actions By Type (per drive)\n\n")
            for row in rows:
                f.write(f"- {row.get('drive')}: {row.get('actions_by_type', {})}\n")

    print(f"[REGRESS] report -> {out_md}")
    print(f"[REGRESS] report -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
