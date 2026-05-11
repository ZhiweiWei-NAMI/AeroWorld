"""Contract-first coverage validation for the canonical 70 scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semantic_event_contract import EPISODE_CONTRACTS, all_contracts, normalize_scenario_id


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "Dataset"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{path}: failed to read deterministic JSON: {exc}") from exc


def scenario_paths(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "scenarios").rglob("event_script.json"))


def validate_contract_coverage(dataset_root: Path) -> list[str]:
    issues: list[str] = []
    scenario_files = scenario_paths(dataset_root)
    found: dict[str, Path] = {}

    for path in scenario_files:
        data = read_json(path)
        scenario_id = normalize_scenario_id(str(data.get("scenario_id") or path.parent.name))
        if scenario_id in found:
            issues.append(f"duplicate scenario script: {scenario_id} -> {found[scenario_id]} and {path}")
            continue
        found[scenario_id] = path
        if scenario_id not in EPISODE_CONTRACTS:
            issues.append(f"unexpected non-contract scenario: {scenario_id} ({path})")

    expected = set(EPISODE_CONTRACTS)
    actual = set(found)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        issues.append(f"missing contract scenarios: {', '.join(missing)}")
    if extra:
        issues.append(f"extra non-contract scenarios: {', '.join(extra)}")

    for contract in all_contracts():
        path = found.get(contract.scenario_id)
        if not path:
            continue
        data = read_json(path)
        payload = dict(data.get("parameters", {}).get("semantic_event_contract") or {})
        if str(payload.get("schema") or "") != "low_altitude_event_chain_contract_v1":
            issues.append(f"{contract.scenario_id}: missing semantic_event_contract payload")
        if dict(payload.get("exact_counts") or {}) != contract.counts:
            issues.append(f"{contract.scenario_id}: exact_counts mismatch")
        if str(payload.get("required_event") or "") != contract.required_event:
            issues.append(f"{contract.scenario_id}: required_event mismatch")
        background = dict(payload.get("background_semantics") or {})
        if str(background.get("vehicle_role") or "") != contract.vehicle_role:
            issues.append(f"{contract.scenario_id}: background vehicle_role mismatch")
        if str(background.get("pedestrian_role") or "") != contract.pedestrian_role:
            issues.append(f"{contract.scenario_id}: background pedestrian_role mismatch")
        if str(payload.get("weather") or "") != contract.weather:
            issues.append(f"{contract.scenario_id}: weather mismatch")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exact low-altitude contract coverage")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        raise SystemExit(1)

    issues = validate_contract_coverage(dataset_root)
    print("=" * 72)
    print("Coverage Validation")
    print("=" * 72)
    print(f"Contract scenarios: {len(EPISODE_CONTRACTS)}")
    print(f"Issues: {len(issues)}")
    if issues:
        for issue in issues[:200]:
            print(f"  - {issue}")
        if len(issues) > 200:
            print(f"  ... {len(issues) - 200} additional issues")
        raise SystemExit(1)
    print("All contract coverage checks PASSED.")


if __name__ == "__main__":
    main()
