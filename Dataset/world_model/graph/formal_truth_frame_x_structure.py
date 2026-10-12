"""Closed whole-frame grammar for X ``formal_truth_frame`` records.

Each nonblank ``truth_frames.jsonl`` line is one complete frame root.
``convert_to_render_ready.py`` writes the fixed 26-key root and four entity
producer families; ``filter_render_ready_truth_for_capture.py`` then filters
the entity list and rebuilds the roster and motion summaries.  The dynamic
``entity_motion_state`` and ``sumo_traffic_light_states`` keys remain instance
identities through the shared root schema's anchored ``propertyNames`` rules.

The root, SUMO lifecycle forms, global UAV form, global pad form, and common
source-row fields are identical to the reviewed L1 grammar.  This module takes
a detached copy of that schema and applies only X producer differences:

* all ten executable runtime-state families can be preserved on authored
  source rows or merged onto SUMO rows;
* ordinary X task rows add pedestrian, vehicle, and crowd-anchor categories,
  while the police-tape prop and no-fly-zone facility retain the producer's
  ``event_script_only`` marker;
* X SUMO traffic adds the ambulance and delivery vehicle types and their two
  registered logical assets.

The X inventory declares six profiles with seeds 00-02 (18 present files).
Bounded validation uses one seed00 file per profile; schema construction does
not read those files.  The module exports only the pure constructor and leaves
registration and dispatch to the integration stage.  These records contain no
publication-excluded integrity subtree.
"""

from __future__ import annotations

from copy import deepcopy

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
)
from Dataset.world_model.graph import formal_truth_frame_l1_structure as _shared
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
    runtime_state_source_guard_rule,
)

__all__ = ["formal_truth_frame_schema"]


_SHARED_VEHICLE_TYPE_PATTERN = (
    r"^(aero_passenger|aero_emergency)(@[^@\s]+)?$"
)
_X_VEHICLE_TYPE_PATTERN = (
    r"^(aero_passenger|aero_emergency|aero_ambulance|aero_delivery)"
    r"(@[^@\s]+)?$"
)
_SHARED_LOGICAL_ASSETS = {
    "vehicle.emergency.suv.v1",
    "vehicle.ground.boxcar.v1",
}
_X_LOGICAL_ASSETS = [
    "vehicle.emergency.ambulance.v1",
    "vehicle.emergency.suv.v1",
    "vehicle.ground.boxcar.v1",
    "vehicle.service.box.v1",
]


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _family_enum_members(family: str) -> tuple[str, ...]:
    for (enum_family, _), members in RUNTIME_STATE_ENUM_FIELDS.items():
        if enum_family == family:
            return tuple(sorted(members))
    raise ValueError(f"runtime state family has no governed enum set: {family}")


def _runtime_state_field_rule(family: str, field: str) -> dict:
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return nonblank_python_string_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        members = _family_enum_members(family)
        return {"oneOf": [
            runtime_enum_string_rule(members),
            {
                "type": "array",
                "minItems": 1,
                "items": runtime_enum_string_rule(members),
            },
        ]}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        members = tuple(sorted(RUNTIME_STATE_ENUM_FIELDS[(family, field)]))
        return runtime_enum_string_rule(members)
    if family == "facility_state" and field in {"capacity", "request_count"}:
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field == "requester_ids":
        return {"type": "array", "items": _identifier()}
    if family == "facility_state" and field in {
        "priority_granted_to", "arbitration_winner_id",
    }:
        return nonblank_python_string_rule()
    if family == "communication_state" and field in {
        "availability", "link_availability",
    }:
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in {
        "hazard_concentration_ppm", "hazard_radius_m",
    }:
        return {"type": "number", "minimum": 0}
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}"
    )


def _runtime_state_family(family: str) -> dict:
    fields = sorted(
        RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        **runtime_state_source_guard_rule("engine_whole"),
        "properties": {
            name: _runtime_state_field_rule(family, name) for name in fields
        },
    }


def _runtime_state_properties() -> dict:
    return {
        family: _runtime_state_family(family) for family in RUNTIME_STATE_FIELDS
    }


def _x_task_row(*, categories: tuple[str, ...], scripted_spawn: bool) -> dict:
    properties = _shared._source_task_properties(category="pedestrian")
    properties["category"] = (
        {"enum": list(categories)}
        if len(categories) > 1
        else {"const": categories[0]}
    )
    properties.update(_runtime_state_properties())
    required = _shared._ENTITY_CORE_REQUIRED + _shared._SOURCE_TASK_REQUIRED
    if scripted_spawn:
        properties["spawn_policy"] = {"const": "event_script_only"}
        required += ("spawn_policy",)
    return _shared._frame_entity_branch(properties, required)


def _only_semantic_facility(source_branches: list[dict]) -> dict:
    found = []
    for branch in source_branches:
        properties = branch["properties"]
        if (
            properties.get("category", {}).get("const") == "facility"
            and "semantic_scope" in branch["required"]
            and "contract_facility" not in properties
            and "scene_occupancy_repair" not in properties
        ):
            found.append(branch)
    if len(found) != 1:
        raise ValueError(
            "shared truth-frame schema must have one plain semantic facility branch"
        )
    return found[0]


def _extend_sumo_branch(branch: dict) -> None:
    properties = branch["properties"]
    vehicle_type_rules = (
        properties["background_vehicle"]["properties"]["vehicle_type"],
        properties["sumo_vehicle"]["properties"]["vehicle_type"],
    )
    for rule in vehicle_type_rules:
        if rule.get("pattern") != _SHARED_VEHICLE_TYPE_PATTERN:
            raise ValueError("shared SUMO vehicle-type rule changed")
        rule["pattern"] = _X_VEHICLE_TYPE_PATTERN

    metadata = properties["sumo_vehicle"]["properties"]["semantic_metadata"]
    if len(metadata.get("oneOf", [])) != 2:
        raise ValueError("shared SUMO semantic-metadata branches changed")
    for metadata_branch in metadata["oneOf"]:
        logical_asset = metadata_branch["properties"]["logical_asset_id"]
        if set(logical_asset.get("enum", [])) != _SHARED_LOGICAL_ASSETS:
            raise ValueError("shared SUMO logical-asset rule changed")
        logical_asset["enum"] = list(_X_LOGICAL_ASSETS)


def formal_truth_frame_schema() -> dict:
    """Return the closed schema for one X ``truth_frames.jsonl`` row."""
    schema = deepcopy(_shared.formal_truth_frame_schema())
    schema["$id"] = (
        "https://aeroworld.local/schema/formal_truth_frame_x.record.schema.json"
    )
    schema["title"] = "AERO_WORLD formal_truth_frame X frame record schema"

    entity_branches = schema["properties"]["entities"]["items"]["oneOf"]
    sumo_branches = [
        branch for branch in entity_branches
        if branch["properties"].get("source", {}).get("const") == "sumo_traci"
    ]
    source_branches = [
        branch for branch in entity_branches
        if "activation_tick" in branch["properties"]
    ]
    if len(sumo_branches) != 3 or len(source_branches) != 10:
        raise ValueError(
            "shared truth-frame schema must have three SUMO and ten source branches"
        )

    for branch in (*sumo_branches, *source_branches):
        branch["properties"].update(_runtime_state_properties())
    for branch in sumo_branches:
        _extend_sumo_branch(branch)

    scripted_facility = deepcopy(_only_semantic_facility(source_branches))
    scripted_facility["properties"]["spawn_policy"] = {
        "const": "event_script_only",
    }
    scripted_facility["required"].append("spawn_policy")

    entity_branches.extend([
        _x_task_row(
            categories=("crowd_anchor", "pedestrian", "vehicle"),
            scripted_spawn=False,
        ),
        _x_task_row(categories=("prop",), scripted_spawn=True),
        scripted_facility,
    ])
    return schema
