import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    ap = argparse.ArgumentParser(description="Check topo regression gate.")
    ap.add_argument("--index", required=True, help="regress_index.jsonl path")
    ap.add_argument(
        "--config",
        default="configs/topo_regress_gate.yaml",
        help="gate config yaml path",
    )
    ap.add_argument(
        "--baseline",
        default="",
        help="baseline yaml path (or env TOPO_BASELINE_PATH)",
    )
    ap.add_argument(
        "--baseline-mode",
        default="",
        choices=["compare", "off", "update"],
        help="baseline mode: compare|off|update (or env TOPO_BASELINE_MODE)",
    )
    return ap.parse_args()


def load_gate_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Gate config must be a mapping")
    return data


def load_baseline(path: Path) -> Dict[str, Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Baseline must be a mapping")
    drives = data.get("drives", [])
    if not isinstance(drives, list):
        raise ValueError("Baseline drives must be a list")
    out: Dict[str, Dict[str, Any]] = {}
    for item in drives:
        if not isinstance(item, dict):
            continue
        drive = item.get("drive")
        if drive:
            out[str(drive)] = item
    return out


def write_baseline(path: Path, entries: Dict[str, Dict[str, Any]]) -> None:
    ordered: List[Dict[str, Any]] = []
    for drive in sorted(entries.keys()):
        item = entries[drive]
        ordered_item: Dict[str, Any] = {
            "drive": drive,
            "max_degree": item.get("max_degree"),
            "dangling_detected": item.get("dangling_detected"),
            "dangling_removed": item.get("dangling_removed"),
            "dangling_merged": item.get("dangling_merged"),
            "dangling_unfixed": item.get("dangling_unfixed"),
            "issues_count": item.get("issues_count"),
            "actions_count": item.get("actions_count"),
        }
        actions_by_type = item.get("actions_by_type")
        if isinstance(actions_by_type, dict) and actions_by_type:
            ordered_item["actions_by_type"] = {
                k: actions_by_type[k] for k in sorted(actions_by_type.keys())
            }
        ordered.append(ordered_item)
    payload = {"drives": ordered}
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def summarize_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for item in items:
        status = item.get("status", "")
        if status in counts:
            counts[status] += 1
    return counts


def check_absolute(summary: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    max_degree_max = cfg.get("max_degree_max")
    if max_degree_max is not None:
        max_degree = summary.get("max_degree")
        if isinstance(max_degree, (int, float)) and max_degree > max_degree_max:
            reasons.append(f"max_degree={max_degree} > {max_degree_max}")

    require_dangling_merged_min = cfg.get("require_dangling_merged_min")
    if require_dangling_merged_min is not None:
        dangling_merged = summary.get("dangling_merged")
        if isinstance(dangling_merged, (int, float)) and dangling_merged < require_dangling_merged_min:
            reasons.append(
                f"dangling_merged={dangling_merged} < {require_dangling_merged_min}"
            )

    dangling_detected = summary.get("dangling_detected")
    if isinstance(dangling_detected, int):
        dangling_removed = int(summary.get("dangling_removed", 0))
        dangling_merged = int(summary.get("dangling_merged", 0))
        dangling_unfixed = int(summary.get("dangling_unfixed", 0))
        if dangling_detected != dangling_removed + dangling_merged + dangling_unfixed:
            reasons.append(
                "dangling invariant failed: "
                f"{dangling_detected} != {dangling_removed}+{dangling_merged}+{dangling_unfixed}"
            )
    return reasons


def check_baseline(
    summary: Dict[str, Any],
    baseline: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []
    delta_max_degree = int(cfg.get("delta_max_degree", 0))
    delta_issues_count = int(cfg.get("delta_issues_count", 0))
    delta_actions_count = int(cfg.get("delta_actions_count", 0))

    max_degree = summary.get("max_degree")
    base_max_degree = baseline.get("max_degree")
    if isinstance(max_degree, (int, float)) and isinstance(base_max_degree, (int, float)):
        if max_degree > base_max_degree + delta_max_degree:
            reasons.append(
                f"max_degree={max_degree} > baseline {base_max_degree} + {delta_max_degree}"
            )
    else:
        reasons.append("baseline missing max_degree")

    dangling_merged = summary.get("dangling_merged")
    base_dangling_merged = baseline.get("dangling_merged")
    if isinstance(dangling_merged, (int, float)) and isinstance(base_dangling_merged, (int, float)):
        if dangling_merged < base_dangling_merged:
            reasons.append(
                f"dangling_merged={dangling_merged} < baseline {base_dangling_merged}"
            )
    else:
        reasons.append("baseline missing dangling_merged")

    issues_count = summary.get("issues_count")
    base_issues_count = baseline.get("issues_count")
    if isinstance(issues_count, (int, float)) and isinstance(base_issues_count, (int, float)):
        if issues_count > base_issues_count + delta_issues_count:
            reasons.append(
                f"issues_count={issues_count} > baseline {base_issues_count} + {delta_issues_count}"
            )
    else:
        reasons.append("baseline missing issues_count")

    actions_count = summary.get("actions_count")
    base_actions_count = baseline.get("actions_count")
    if isinstance(actions_count, (int, float)) and isinstance(base_actions_count, (int, float)):
        if actions_count > base_actions_count + delta_actions_count:
            reasons.append(
                f"actions_count={actions_count} > baseline {base_actions_count} + {delta_actions_count}"
            )
    else:
        reasons.append("baseline missing actions_count")

    return reasons


def main() -> int:
    args = parse_args()
    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"ERROR: index not found: {index_path}")
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"ERROR: gate config not found: {cfg_path}")

    baseline_path = args.baseline or ""
    if not baseline_path:
        baseline_path = str(Path.cwd() / "configs" / "topo_regress_baseline.yaml")
    if "TOPO_BASELINE_PATH" in os.environ and not args.baseline:
        baseline_path = os.environ["TOPO_BASELINE_PATH"]
    baseline_path = str(baseline_path)

    baseline_mode = args.baseline_mode or os.environ.get("TOPO_BASELINE_MODE", "")
    baseline_file = Path(baseline_path)
    if not baseline_mode:
        baseline_mode = "compare" if baseline_file.exists() else "off"

    cfg = load_gate_config(cfg_path)
    items = read_index(index_path)
    counts = summarize_counts(items)

    baseline_map: Dict[str, Dict[str, Any]] = {}
    if baseline_file.exists():
        baseline_map = load_baseline(baseline_file)

    pass_items = [item for item in items if item.get("status") == "PASS"]
    updated_entries: Dict[str, Dict[str, Any]] = dict(baseline_map)
    if baseline_mode == "update":
        for item in pass_items:
            summary_path = item.get("summary_path")
            if not summary_path:
                continue
            summary_file = Path(summary_path)
            if not summary_file.exists():
                continue
            summary = read_json_from_markdown(summary_file)
            drive = item.get("drive")
            if not drive:
                continue
            updated_entries[drive] = {
                "drive": drive,
                "max_degree": summary.get("max_degree"),
                "dangling_detected": summary.get("dangling_detected"),
                "dangling_removed": summary.get("dangling_removed"),
                "dangling_merged": summary.get("dangling_merged"),
                "dangling_unfixed": summary.get("dangling_unfixed"),
                "issues_count": summary.get("issues_count"),
                "actions_count": summary.get("actions_count"),
                "actions_by_type": summary.get("actions_by_type", {}),
            }
        write_baseline(baseline_file, updated_entries)
        baseline_map = load_baseline(baseline_file)
        baseline_mode = "compare"
        print(f"[GATE] baseline updated -> {baseline_file}")

    errors: List[str] = []
    infos: List[str] = []

    min_pass_drives = int(cfg.get("min_pass_drives", 1))
    if counts["PASS"] < min_pass_drives:
        errors.append(f"PASS drives {counts['PASS']} < min_pass_drives {min_pass_drives}")

    allow_skipped = bool(cfg.get("allow_skipped", True))
    if not allow_skipped and counts["SKIPPED"] > 0:
        errors.append(f"SKIPPED drives {counts['SKIPPED']} not allowed")

    for item in pass_items:
        drive = item.get("drive", "UNKNOWN")
        summary_path = item.get("summary_path")
        if not summary_path:
            errors.append(f"{drive}: missing_summary_path")
            continue
        summary_file = Path(summary_path)
        if not summary_file.exists():
            errors.append(f"{drive}: summary_not_found")
            continue
        summary = read_json_from_markdown(summary_file)
        abs_reasons = check_absolute(summary, cfg)
        if abs_reasons:
            errors.append(f"{drive}: " + "; ".join(abs_reasons))
        if baseline_mode == "compare":
            baseline_item = baseline_map.get(drive)
            if baseline_item is None:
                infos.append(f"{drive}: no baseline")
            else:
                base_reasons = check_baseline(summary, baseline_item, cfg)
                if base_reasons:
                    errors.append(f"{drive}: " + "; ".join(base_reasons))

    print("[GATE] summary:", counts)
    print("[GATE] config:", cfg)
    print("[GATE] baseline_mode:", baseline_mode)
    print("[GATE] baseline_path:", baseline_file)
    for info in infos:
        print("[GATE] info:", info)
    if errors:
        print("[GATE] FAIL")
        for err in errors:
            print(" -", err)
        return 1
    print("[GATE] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
