"""
覆盖率验证工具。

检查:
  1. 所有 33 个事件类型是否每个至少有一个 event_script.json
  2. CAAC 14 个紧急事件是否每个有覆盖
  3. 5 个 event mechanism 类别是否每个至少 3 个脚本
  4. 6 个 ODD Layer 是否每个至少 2 个脚本

使用方式:
    python validate_coverage.py [--dataset-root Dataset/]
"""

from __future__ import annotations
import json, sys, argparse
from pathlib import Path
from collections import defaultdict


def load_taxonomy(taxonomy_path: Path) -> dict:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_scenario_scripts(scenarios_dir: Path) -> dict[str, list[Path]]:
    """
    遍历 scenarios/ 目录，找到所有 event_script.json。
    返回 {event_type_id: [Path, ...]}
    """
    scripts: dict[str, list[Path]] = defaultdict(list)

    for json_file in scenarios_dir.rglob("event_script.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            scenario_id = data.get("scenario_id", "")
            # scenario_id 格式: "L4-5_v1" — 提取 event_type_id
            parts = scenario_id.split("_")
            if len(parts) >= 1:
                etype_raw = parts[0]
                # L4-5_v1 → L4-5
                if "-" in etype_raw:
                    scripts[etype_raw].append(json_file)
        except Exception:
            pass

    return dict(scripts)


def check_event_type_coverage(taxonomy: dict, found: dict[str, list[Path]]) -> list[str]:
    """检查每个事件类型是否有至少一个脚本。"""
    issues = []
    types = taxonomy.get("event_types", [])
    for et in types:
        eid = et["event_type_id"]
        if eid not in found or len(found[eid]) == 0:
            issues.append(f"MISSING: event type {eid} ({et['name']}) has no scripts")
    return issues


def check_caac_coverage(taxonomy: dict, found: dict[str, list[Path]]) -> list[str]:
    """检查 CAAC 覆盖率。"""
    issues = []
    caac_matrix = taxonomy.get("caac_coverage_matrix", {})
    for caac_id, info in caac_matrix.items():
        coverage = info.get("coverage", "")
        # 检查 coverage 中引用的 event types 是否都有脚本
        covered_types = []
        for et in taxonomy.get("event_types", []):
            if et["event_type_id"] in coverage:
                covered_types.append(et["event_type_id"])

        missing = [t for t in covered_types if t not in found]
        if missing:
            issues.append(f"CAAC {caac_id} ({info['description'][:40]}...): "
                         f"referenced types not found: {missing}")
    return issues


def check_mechanism_coverage(taxonomy: dict, found: dict[str, list[Path]]) -> list[str]:
    """检查每个 mechanism 类别至少有 3 个脚本。"""
    issues = []
    mech_count: dict[str, int] = defaultdict(int)
    for et in taxonomy.get("event_types", []):
        eid = et["event_type_id"]
        mech = et.get("mechanism", "unknown")
        if eid in found:
            mech_count[mech] += len(found[eid])

    for mech in ["collision", "failure", "environmental", "violation", "operational"]:
        n = mech_count.get(mech, 0)
        if n < 3:
            issues.append(f"Mechanism '{mech}': only {n} scripts (need >= 3)")
    return issues


def check_layer_coverage(taxonomy: dict, found: dict[str, list[Path]]) -> list[str]:
    """检查每个 ODD Layer 至少有 2 个脚本。"""
    issues = []
    layer_count: dict[str, int] = defaultdict(int)
    for et in taxonomy.get("event_types", []):
        eid = et["event_type_id"]
        layer = eid.split("-")[0] if "-" in eid else "unknown"
        if eid in found:
            layer_count[layer] += len(found[eid])

    for layer in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        n = layer_count.get(layer, 0)
        if n < 2:
            issues.append(f"ODD Layer '{layer}': only {n} scripts (need >= 2)")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate UAM dataset coverage")
    parser.add_argument("--dataset-root", default="Dataset", help="Path to Dataset/ directory")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.exists():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    taxonomy_path = root / "taxonomy.json"
    scenarios_dir = root / "scenarios"

    if not taxonomy_path.exists():
        print(f"ERROR: taxonomy.json not found at {taxonomy_path}", file=sys.stderr)
        sys.exit(1)

    taxonomy = load_taxonomy(taxonomy_path)
    found = find_scenario_scripts(scenarios_dir)

    total_scripts = sum(len(v) for v in found.values())
    covered_types = len(found)

    print("=" * 60)
    print("UAM Dataset Coverage Report")
    print("=" * 60)
    print(f"Total event_script.json found: {total_scripts}")
    print(f"Event types covered: {covered_types}/{len(taxonomy['event_types'])}")
    print()

    all_issues = []
    all_issues.extend(check_event_type_coverage(taxonomy, found))
    all_issues.extend(check_caac_coverage(taxonomy, found))
    all_issues.extend(check_mechanism_coverage(taxonomy, found))
    all_issues.extend(check_layer_coverage(taxonomy, found))

    if all_issues:
        print(f"ISSUES ({len(all_issues)}):")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("All coverage checks PASSED.")

    print()
    print("Per-layer breakdown:")
    layer_count = defaultdict(int)
    for et in taxonomy["event_types"]:
        eid = et["event_type_id"]
        layer = eid.split("-")[0]
        n = len(found.get(eid, []))
        layer_count[layer] += n
    for layer in sorted(layer_count):
        print(f"  {layer}: {layer_count[layer]} scripts")

    sys.exit(0 if not all_issues else 1)


if __name__ == "__main__":
    main()
