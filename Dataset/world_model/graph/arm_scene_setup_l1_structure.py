"""Closed scene grammar for the exact scene referenced by an ARM manifest.

The formal scenario scene is an authoring input.  Its existing authoring
grammar closes the complete JSON root, including fixed array positions.
The source path remains an explicit argument because profiles have different
authored entities and plans.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from Dataset.world_model.graph.authoring_structure import authoring_source_grammar
from Dataset.world_model.graph.contract import REPO


_AUTHORING_DIRS = {
    "L1": ("L1_airspace", frozenset({"interaction"})),
    "L2": ("L2_infrastructure", frozenset({"failure", "interaction"})),
    "L3": ("L3_dynamic_constraints", frozenset({"interaction"})),
    "L4": ("L4_agents", frozenset({"collision", "failure", "interaction"})),
    "L5": ("L5_environment", frozenset({"interaction"})),
    "L6": ("L6_digital_layer", frozenset({"failure"})),
}


def _canonical_relative(source_file: str | Path) -> Path:
    supplied = Path(source_file)
    source = (supplied if supplied.is_absolute() else REPO / supplied).resolve(strict=True)
    try:
        relative = source.relative_to(REPO)
    except ValueError as exc:
        raise ValueError(f"scene source outside repository: {source_file}") from exc
    if (supplied != source if supplied.is_absolute()
            else supplied.as_posix() != relative.as_posix()):
        raise ValueError(f"scene source path is not canonical: {source_file}")
    return relative


def _authored_scene_schema(level: str, source_file: str | Path) -> dict[str, Any]:
    if level not in _AUTHORING_DIRS:
        raise ValueError(f"unsupported authored ARM scene level: {level!r}")
    relative = _canonical_relative(source_file)
    parts = relative.parts
    section, branches = _AUTHORING_DIRS[level]
    if (len(parts) != 6 or parts[:3] != ("Dataset", "scenarios", section)
            or parts[3] not in branches or parts[5] != "scene_setup.json"
            or not re.fullmatch(rf"{level}-[0-9]+_v[0-9]+", parts[4])):
        raise ValueError(f"{relative}: not an authored {level} scene")
    return authoring_source_grammar("formal_source_scene_setup", relative).schema


def arm_scene_setup_schema(source_file: str | Path) -> dict[str, Any]:
    """Return the full L1 scene schema for a manifest's exact scene reference."""
    return _authored_scene_schema("L1", source_file)
