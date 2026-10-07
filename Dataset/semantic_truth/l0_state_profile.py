"""Validated profile access for deterministic L0 state completion."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from Dataset.tools.runtime_state_contract import (
    invalid_runtime_state_value_paths,
    unconsumed_runtime_state_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_L0_STATE_PROFILE_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "l0_state_supplement_profile.json"
)


class L0StateProfileError(ValueError):
    """Raised when the governed L0 completion profile is malformed."""


def load_l0_state_profile(
    path: Path = DEFAULT_L0_STATE_PROFILE_PATH,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise L0StateProfileError(f"{path}: root must be an object")
    if value.get("schema_name") != "l0_state_supplement_profile":
        raise L0StateProfileError(f"{path}: unexpected schema_name")
    temperatures = value.get("weather_temperature_c_by_condition")
    if not isinstance(temperatures, dict) or not temperatures:
        raise L0StateProfileError(f"{path}: weather temperature mapping is missing")
    for condition, raw in temperatures.items():
        if (
            not isinstance(condition, str)
            or isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise L0StateProfileError(
                f"{path}: invalid temperature for condition {condition!r}"
            )
    baselines = value.get("runtime_nominal_state_by_category")
    if not isinstance(baselines, dict) or not baselines:
        raise L0StateProfileError(f"{path}: runtime nominal states are missing")
    for category, state in baselines.items():
        if not isinstance(category, str) or not isinstance(state, dict):
            raise L0StateProfileError(f"{path}: invalid runtime baseline category")
        invalid = invalid_runtime_state_value_paths(state)
        unconsumed = unconsumed_runtime_state_paths(state)
        if invalid or unconsumed:
            raise L0StateProfileError(
                f"{path}: invalid runtime baseline {category}: "
                f"invalid={list(invalid)}, unconsumed={list(unconsumed)}"
            )
    contracts = value.get("source_unavailability_contracts")
    if not isinstance(contracts, dict) or not contracts:
        raise L0StateProfileError(f"{path}: source availability contracts are missing")
    for contract_id, contract in contracts.items():
        if not isinstance(contract_id, str) or not isinstance(contract, dict):
            raise L0StateProfileError(f"{path}: invalid source contract")
        predicate_ids = contract.get("predicate_ids")
        if (
            contract.get("executable") is not False
            or not isinstance(predicate_ids, list)
            or not predicate_ids
            or not all(isinstance(item, str) and item for item in predicate_ids)
            or not isinstance(contract.get("scope_type"), str)
            or not isinstance(contract.get("non_executable_reason"), str)
            or not isinstance(contract.get("closure_statement"), str)
        ):
            raise L0StateProfileError(
                f"{path}: incomplete source contract {contract_id!r}"
            )
    return value


def runtime_baseline_for_category(
    profile: Mapping[str, Any],
    category: str,
) -> dict[str, Any]:
    baselines = profile["runtime_nominal_state_by_category"]
    key = "facility" if category in {
        "charging_pad",
        "charging_station",
        "facility",
        "ground_station",
        "landing_facility",
        "landing_pad",
        "pad",
    } else category
    value = baselines.get(key, {})
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


__all__ = [
    "DEFAULT_L0_STATE_PROFILE_PATH",
    "L0StateProfileError",
    "load_l0_state_profile",
    "runtime_baseline_for_category",
]
