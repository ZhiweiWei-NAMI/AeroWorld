"""Resolve ARM state observation from its producer's row markers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[3]


def state_profile(source_file: str | Path, scope: Mapping[str, Any] | None = None) -> str:
    """Read the producer profile from an exact ARM state source path."""
    path = Path(source_file)
    relative = path.relative_to(REPO) if path.is_absolute() else path
    parts = relative.parts
    if len(parts) != 7 or parts[0] != "arms" or parts[5:] != ("raw", "states.jsonl"):
        raise ValueError(f"ARM state source path differs: {source_file}")
    profile = parts[1]
    if scope is not None and scope.get("profile") != profile:
        raise ValueError(f"ARM state source profile differs from sample: {source_file}")
    return profile


def record_observed(profile: str, layer: str, row: Mapping[str, Any],
                    path: str | Path, line: int | None) -> bool:
    """Use one producer-specific observation rule for graph and catalog readers."""
    if layer == "trajectory":
        if any(key in row for key in ("observed", "present", "observation_status", "execution_active")):
            raise ValueError(f"trajectory row has unexpected observation markers: {path}:{line}")
        return True
    if layer != "state":
        raise ValueError(f"ARM observation layer is not declared: {layer}")
    if profile.startswith("L2-"):
        observed = row.get("observed")
        if type(observed) is not bool:
            raise ValueError(f"L2 state lacks a boolean observed marker: {path}:{line}")
        return observed
    if profile.startswith("L5-"):
        status, active = row.get("observation_status"), row.get("execution_active")
        if status == "observed" and active is True:
            observed = True
        elif status == "engine_inactive" and active is False:
            observed = False
        else:
            raise ValueError(f"L5 state has inconsistent observation markers: {path}:{line}")
        if "observed" in row and row["observed"] is not observed:
            raise ValueError(f"L5 state observed marker conflicts with producer status: {path}:{line}")
        return observed
    if profile.startswith("L6-"):
        present = row.get("present")
        if type(present) is not bool:
            raise ValueError(f"L6 state lacks a boolean present marker: {path}:{line}")
        if any(key in row for key in ("observed", "observation_status", "execution_active")):
            raise ValueError(f"L6 state has unexpected observation markers: {path}:{line}")
        values = row.get("values")
        if type(values) is not dict:
            raise ValueError(f"L6 state values is not an object: {path}:{line}")
        if present is False and values:
            raise ValueError(f"L6 absent state carries values: {path}:{line}")
        return present
    if profile.startswith(("L1-", "L3-", "L4-", "X")):
        if any(key in row for key in ("observed", "observation_status", "present")):
            raise ValueError(f"direct trajectory state has unexpected observation markers: {path}:{line}")
        return True
    raise ValueError(f"ARM state producer is not declared: {profile}: {path}:{line}")
