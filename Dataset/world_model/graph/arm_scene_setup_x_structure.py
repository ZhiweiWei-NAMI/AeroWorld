"""X ARM scene grammar for the producer's scoped scene copy."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from Dataset.world_model.graph.arm_scene_setup_l1_structure import _canonical_relative
from Dataset.world_model.graph.authoring_structure import (
    _same_json, authoring_source_grammar,
)
from Dataset.world_model.graph.contract import REPO


def arm_scene_setup_schema(source_file: str | Path) -> dict[str, Any]:
    relative = _canonical_relative(source_file)
    parts = relative.parts
    if (len(parts) != 5 or parts[0] != "interventions"
            or not re.fullmatch(r"X[1-6]_[a-z0-9_]+", parts[1])
            or not re.fullmatch(r"seed[0-9]{2}", parts[2])
            or parts[3] != "grp_cross_layer" or parts[4] != "scene_setup.json"):
        raise ValueError(f"{relative}: not a declared X scoped scene")

    authored_ref = Path("Dataset/scenarios/X_cross_layer") / parts[1] / "scene_setup.json"
    authored = json.loads((REPO / authored_ref).read_text(encoding="utf-8"))
    scoped = json.loads((REPO / relative).read_text(encoding="utf-8"))
    entities = authored["entities"]
    if type(entities) is not list or any(type(entity) is not dict for entity in entities):
        raise ValueError(f"{authored_ref}: producer expects an entity array")
    keep = [index for index, entity in enumerate(entities)
            if entity.get("initial_state", {}).get("role") != "semantic_background_vehicle"]
    expected = dict(authored)
    expected["entities"] = [entities[index] for index in keep]
    if not _same_json(scoped, expected):
        raise ValueError(f"{relative}: scoped scene differs from x_arm_pipeline.prepare")

    schema = authoring_source_grammar("formal_source_scene_setup", authored_ref).schema
    entity_schema = schema["properties"]["entities"]
    positions = entity_schema["prefixItems"]
    if len(positions) != len(entities):
        raise ValueError(f"{authored_ref}: authoring schema entity positions differ")
    entity_schema["prefixItems"] = [positions[index] for index in keep]
    entity_schema["minItems"] = entity_schema["maxItems"] = len(keep)
    return schema
