"""Report structural statistics and stable ontology-module selections.

Usage:
    python statistics.py --dataset-root Dataset
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset statistics")
    parser.add_argument("--dataset-root", default="Dataset", help="Path to Dataset/")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()

    selection_manifest_path = (
        root / "knowledge_graph" / "profiles" / "ontology_selection_manifest.json"
    )
    with selection_manifest_path.open("r", encoding="utf-8") as f:
        selection_manifest = json.load(f)

    scenarios_dir = root / "scenarios"
    scripts = sorted(scenarios_dir.rglob("event_script.json"))

    all_scripts = []
    for script_path in scripts:
        with script_path.open("r", encoding="utf-8") as f:
            all_scripts.append(json.load(f))

    print("=" * 60)
    print("AeroWorld Dataset Statistics")
    print("=" * 60)
    print(f"\nTotal scenarios: {len(all_scripts)}")

    selected_modules = selection_manifest["selected_modules"]
    episodes = selection_manifest["episodes"]
    module_episode_counts = {module_id: len(episodes) for module_id in selected_modules}

    print(
        f"\nStable ontology modules: {len(selected_modules)} "
        f"(mapped episodes: {len(episodes)})"
    )
    for module_id in selected_modules:
        print(f"  {module_id}: {module_episode_counts[module_id]}")

    chain_lengths = []
    event_counts = []
    trigger_types: Counter[str] = Counter()
    action_types: Counter[str] = Counter()

    for script in all_scripts:
        events = script.get("events", [])
        event_counts.append(len(events))

        chain_events = [
            event for event in events if event.get("on_fire", {}).get("emit_events")
        ]
        chain_lengths.append(len(chain_events))

        for trigger in script.get("triggers", []):
            trigger_types[trigger.get("type", "unknown")] += 1

        for event in events:
            for action in event.get("actions", []):
                action_types[action.get("type", "unknown")] += 1

    if event_counts:
        print(
            f"\nEvents per scenario: min={min(event_counts)}, "
            f"max={max(event_counts)}, "
            f"mean={sum(event_counts) / len(event_counts):.1f}"
        )
        print(
            f"Chain events per scenario: min={min(chain_lengths)}, "
            f"max={max(chain_lengths)}, "
            f"mean={sum(chain_lengths) / len(chain_lengths):.1f}"
        )
    else:
        print("\nNo event scripts found.")

    print("\nTrigger types:")
    for trigger_type, count in trigger_types.most_common():
        print(f"  {trigger_type}: {count}")

    print("\nAction types:")
    for action_type, count in action_types.most_common():
        print(f"  {action_type}: {count}")


if __name__ == "__main__":
    main()
