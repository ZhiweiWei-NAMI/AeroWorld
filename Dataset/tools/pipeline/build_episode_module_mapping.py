#!/usr/bin/env python3
"""Build the unified ontology-module selection from authored scenario identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = PROJECT_ROOT / "Dataset" / "scenarios"
PROFILE_ROOT = PROJECT_ROOT / "Dataset" / "knowledge_graph" / "profiles"
MANIFEST_PATH = PROFILE_ROOT / "ontology_selection_manifest.json"
SCHEMA_PATH = PROFILE_ROOT / "ontology_selection_manifest.schema.json"
ONTOLOGY_VERSION = "0.3.0"
FORMAL_PROFILE_COUNT = 70
FORMAL_EPISODE_COUNT = 210
SEEDS = (0, 1, 2)
MODULES = (
    "airspace_operations",
    "ground_mobility",
    "mobility_regulation",
    "agent_interaction_safety",
    "facility_energy_service",
    "communication_process",
    "computation_process",
    "positioning_navigation",
    "environment_hazard",
    "digital_security",
    "operational_constraint_emergency",
    "utm",
)


def _scenario_ids() -> tuple[str, ...]:
    scenario_ids: list[str] = []
    for spec_path in sorted(SCENARIO_ROOT.rglob("spec.py")):
        scenario_root = spec_path.parent
        event_script_path = scenario_root / "event_script.json"
        scene_setup_path = scenario_root / "scene_setup.json"
        if not event_script_path.is_file() or not scene_setup_path.is_file():
            raise ValueError(f"scenario source set is incomplete: {scenario_root}")
        event_script = json.loads(event_script_path.read_text(encoding="utf-8"))
        scenario_id = scenario_root.name
        if event_script.get("scenario_id") != scenario_id:
            raise ValueError(f"scenario identity diverges at {event_script_path}")
        scenario_ids.append(scenario_id)
    if len(scenario_ids) != FORMAL_PROFILE_COUNT:
        raise ValueError(
            f"scenario source count must be {FORMAL_PROFILE_COUNT}: {len(scenario_ids)}"
        )
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario identities must be unique")
    return tuple(scenario_ids)


def build_manifest() -> dict[str, object]:
    scenario_ids = _scenario_ids()
    profiles = [
        {
            "scenario_id": scenario_id,
            "seeds": list(SEEDS),
            "episode_ids": [f"{scenario_id}__seed{seed:02d}" for seed in SEEDS],
        }
        for scenario_id in scenario_ids
    ]
    episodes = [
        {
            "episode_id": f"{scenario_id}__seed{seed:02d}",
            "scenario_id": scenario_id,
            "seed": seed,
        }
        for scenario_id in scenario_ids
        for seed in SEEDS
    ]
    if len(episodes) != FORMAL_EPISODE_COUNT:
        raise ValueError(
            f"episode identity count must be {FORMAL_EPISODE_COUNT}: {len(episodes)}"
        )
    return {
        "schema_name": "semantic_ontology_selection_manifest",
        "schema_version": "1.0.0",
        "manifest_id": "aeroworld_210_episode_stable_module_selection",
        "dataset_id": "aeroworld_70_scenarios_3_seeds",
        "ontology_version": ONTOLOGY_VERSION,
        "module_policy": "unified_stable_domain_modules",
        "selected_modules": list(MODULES),
        "profile_count": len(profiles),
        "episode_count": len(episodes),
        "scenario_profiles": profiles,
        "episodes": episodes,
    }


def build_schema() -> dict[str, object]:
    module_schema = {
        "type": "array",
        "prefixItems": [{"const": module} for module in MODULES],
        "minItems": len(MODULES),
        "maxItems": len(MODULES),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "AeroWorld unified ontology selection manifest",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_name",
            "schema_version",
            "manifest_id",
            "dataset_id",
            "ontology_version",
            "module_policy",
            "selected_modules",
            "profile_count",
            "episode_count",
            "scenario_profiles",
            "episodes",
        ],
        "properties": {
            "schema_name": {"const": "semantic_ontology_selection_manifest"},
            "schema_version": {"const": "1.0.0"},
            "manifest_id": {"const": "aeroworld_210_episode_stable_module_selection"},
            "dataset_id": {"const": "aeroworld_70_scenarios_3_seeds"},
            "ontology_version": {"const": ONTOLOGY_VERSION},
            "module_policy": {"const": "unified_stable_domain_modules"},
            "selected_modules": module_schema,
            "profile_count": {"const": FORMAL_PROFILE_COUNT},
            "episode_count": {"const": FORMAL_EPISODE_COUNT},
            "scenario_profiles": {
                "type": "array",
                "minItems": FORMAL_PROFILE_COUNT,
                "maxItems": FORMAL_PROFILE_COUNT,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["scenario_id", "seeds", "episode_ids"],
                    "properties": {
                        "scenario_id": {"type": "string", "minLength": 1},
                        "seeds": {
                            "type": "array",
                            "prefixItems": [{"const": seed} for seed in SEEDS],
                            "minItems": len(SEEDS),
                            "maxItems": len(SEEDS),
                        },
                        "episode_ids": {
                            "type": "array",
                            "minItems": len(SEEDS),
                            "maxItems": len(SEEDS),
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
            "episodes": {
                "type": "array",
                "minItems": FORMAL_EPISODE_COUNT,
                "maxItems": FORMAL_EPISODE_COUNT,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["episode_id", "scenario_id", "seed"],
                    "properties": {
                        "episode_id": {"type": "string", "minLength": 1},
                        "scenario_id": {"type": "string", "minLength": 1},
                        "seed": {"enum": list(SEEDS)},
                    },
                },
            },
        },
    }


def _render(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_or_check(path: Path, content: str, *, check: bool) -> None:
    if check:
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"generated ontology selection artifact differs: {path}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    _write_or_check(MANIFEST_PATH, _render(build_manifest()), check=args.check)
    _write_or_check(SCHEMA_PATH, _render(build_schema()), check=args.check)
    print(
        json.dumps(
            {
                "mode": "check" if args.check else "write",
                "profile_count": FORMAL_PROFILE_COUNT,
                "episode_count": FORMAL_EPISODE_COUNT,
                "selected_module_count": len(MODULES),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
