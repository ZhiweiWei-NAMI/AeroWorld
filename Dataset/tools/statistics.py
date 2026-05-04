"""
数据集统计报告。

分析场景分布、事件分布、图规模等统计信息。

使用方式:
    python statistics.py --dataset-root Dataset/
"""

from __future__ import annotations
import json, argparse
from pathlib import Path
from collections import defaultdict, Counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset statistics")
    parser.add_argument("--dataset-root", default="Dataset", help="Path to Dataset/")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()

    # Load taxonomy
    taxonomy_path = root / "taxonomy.json"
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    # Find all event_script.json
    scenarios_dir = root / "scenarios"
    scripts = list(scenarios_dir.rglob("event_script.json"))

    # Load all event scripts
    all_scripts = []
    for sp in scripts:
        with open(sp, "r", encoding="utf-8") as f:
            all_scripts.append(json.load(f))

    print("=" * 60)
    print("UAM Dataset Statistics")
    print("=" * 60)

    # Scenario counts by layer
    layer_counts: dict[str, int] = defaultdict(int)
    for s in all_scripts:
        sid = s.get("scenario_id", "")
        layer = sid.split("-")[0] if "-" in sid else "unknown"
        layer_counts[layer] += 1

    print(f"\nTotal scenarios: {len(all_scripts)}")
    print("\nPer ODD Layer:")
    for layer in sorted(layer_counts):
        print(f"  {layer}: {layer_counts[layer]}")

    # Event chain statistics
    chain_lengths = []
    event_counts = []
    trigger_types: dict[str, int] = Counter()
    action_types: dict[str, int] = Counter()

    for s in all_scripts:
        events = s.get("events", [])
        event_counts.append(len(events))

        # Chain length = number of events with on_fire.emit_events
        chain_events = [e for e in events if e.get("on_fire", {}).get("emit_events")]
        chain_lengths.append(len(chain_events))

        # Trigger types
        for t in s.get("triggers", []):
            trigger_types[t.get("type", "unknown")] += 1

        # Action types
        for e in events:
            for a in e.get("actions", []):
                action_types[a.get("type", "unknown")] += 1

    print(f"\nEvents per scenario: min={min(event_counts)}, max={max(event_counts)}, "
          f"mean={sum(event_counts)/len(event_counts):.1f}")
    print(f"Chain events per scenario: min={min(chain_lengths)}, max={max(chain_lengths)}, "
          f"mean={sum(chain_lengths)/len(chain_lengths):.1f}")

    print("\nTrigger types:")
    for ttype, count in trigger_types.most_common():
        print(f"  {ttype}: {count}")

    print("\nAction types:")
    for atype, count in action_types.most_common():
        print(f"  {atype}: {count}")

    # Mechanism distribution
    mech_counts: dict[str, int] = Counter()
    for et in taxonomy["event_types"]:
        mech = et.get("mechanism", "unknown")
        mech_counts[mech] += len(et.get("variants", []))

    print("\nPer Mechanism Category:")
    for mech in sorted(mech_counts):
        print(f"  {mech}: {mech_counts[mech]}")

    # CAAC coverage
    print("\nCAAC Emergency Event Coverage:")
    caac = taxonomy.get("caac_coverage_matrix", {})
    covered = 0
    for caac_id in sorted(caac.keys()):
        info = caac[caac_id]
        coverage_text = info.get("coverage", "Not covered")
        is_covered = "Not " not in coverage_text.split(".")[0]
        if is_covered:
            covered += 1
        status = "OK" if is_covered else "PARTIAL"
        print(f"  {caac_id}: {status} — {coverage_text[:70]}")
    print(f"\nCAAC coverage: {covered}/{len(caac)}")


if __name__ == "__main__":
    main()
