"""Whole-root UE ARM branch roster grammar, with an explicit layer selector.

``arm_ue_export.export_arm`` writes a JSON root containing ``entities``.
Untouched background entries are copied from the capture-filtered formal
roster. Window-engine entries come from ``raw/ue_roster.json`` and receive
profile identity, site/ROI, mode and initial pose from measured frames.
``write_engine_inputs`` adds activation/deactivation and spawned to every
engine entry, and removes current runtime-state families. The layer's formal
roster branches provide the already-closed nested producer contracts; the
engine branches retain only fields that its preserve list can write.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module

from Dataset.tools.runtime_state_contract import RUNTIME_STATE_FIELDS

__all__ = ["arm_branch_roster_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_LEVELS = {"L1", "L2", "L3", "L4", "L5", "L6", "X"}
_ENGINE_FIXED = frozenset({
    "entity_id", "asset_id", "label_class", "initial_yaw_deg",
    "activation_tick", "deactivation_tick", "spawned", "site_id",
    "roi_id", "entity_category", "entity_kind", "entity_type",
    "proxy_template_id", "logical_asset_id", "mode",
    "initial_position_enu_m",
})
# Dataset/tools/batch_generate.py::PRESERVED_ENTITY_FIELDS is the fixed
# producer list consumed by arm_ue_export.write_engine_inputs. It is declared
# here because that tool is an executable script with tools-local imports.
_ENGINE_PRESERVED = frozenset({
    "semantic_scope", "entity_kind", "task_id", "role", "state_sequence",
    "semantic_role", "background_role", "contract_scenario_id",
    "contract_inspect_uav", "background_vehicle", "background_pedestrian",
    "ground_flow_contract", "event_actor_motion_contract", "contract_facility",
    "contract_logical_sidecar", "uav_corridor_role", "uav_corridor",
    "assigned_altitude_m", "inspect_altitude_code", "inspect_altitude_m",
    "min_path_length_m", "full_episode_presence", "observer_lifecycle",
    "ground_reference_z_m", "ground_reference_source", "lifecycle",
    "activation_tick", "deactivation_tick", "spawn_policy", "category",
    "route_waypoints_enu_m", "planned_route_waypoints_enu_m",
    "path_deviation_contract", "task_kind", "task_state",
})
_ENGINE_ALLOWED = (_ENGINE_FIXED | _ENGINE_PRESERVED) - set(RUNTIME_STATE_FIELDS)


def _branches(schema: dict) -> list[dict]:
    """Flatten only the formal producer's explicit entity alternatives."""
    result = []
    for branch in schema["properties"]["entities"]["items"]["oneOf"]:
        if "oneOf" in branch:
            result.extend(deepcopy(branch["oneOf"]))
        else:
            result.append(deepcopy(branch))
    return result


def _engine_branch(formal_branch: dict) -> dict:
    properties = {key: rule for key, rule in formal_branch["properties"].items()
                  if key in _ENGINE_ALLOWED}
    properties.update({
        "activation_tick": {"type": "integer", "minimum": 0},
        "deactivation_tick": {"oneOf": [
            {"type": "integer", "minimum": 0}, {"type": "null"},
        ]},
        "spawned": {"type": "boolean"},
    })
    if "spawn_policy" not in properties:
        # Scene-authored event actors and hazard anchors retain this value
        # through the engine preserve list even when their formal converter
        # branch does not expose it.
        properties["spawn_policy"] = {"const": "event_script_only"}
    # The writer copies label_class from the measured engine row. The formal
    # converter may normalize it for its own branch (notably L3 road props).
    properties["label_class"] = {"type": "string", "minLength": 1}
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(_ENGINE_FIXED),
        "properties": properties,
    }


def _logical_anchor_branch(category: str, asset: str, label: str, *,
                           entity_kind: str | None = None,
                           entity_type: str | None = None) -> dict:
    """Close the logical-only profile retained from ARM scene entities."""
    from Dataset.world_model.graph.formal_roster_l4_structure import (
        _semantic_scope,
    )
    kind = entity_kind or f"{category}.{label}"
    scope = _semantic_scope()
    if kind == "ground_station.ground_control_station":
        scope["properties"]["scope_subtype"]["enum"].append(
            "ground_control_station")
        scope["properties"]["ontology_class_id"]["enum"].append(
            "world:GroundControlStation")
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(_ENGINE_FIXED | {
            "category", "role", "route_waypoints_enu_m", "semantic_role",
            "state_sequence", "task_id",
        }),
        "properties": {
            "entity_id": {"type": "string", "minLength": 1},
            "asset_id": {"const": asset},
            "label_class": {"const": label},
            "initial_yaw_deg": {"type": "number"},
            "activation_tick": {"type": "integer", "minimum": 0},
            "deactivation_tick": {"oneOf": [
                {"type": "integer", "minimum": 0}, {"type": "null"},
            ]},
            "spawned": {"type": "boolean"},
            "site_id": {"type": "string", "minLength": 1},
            "roi_id": {"type": "string", "minLength": 1},
            "entity_category": {"const": category},
            "entity_kind": {"const": kind},
            "entity_type": {"const": entity_type or kind},
            "proxy_template_id": {"const": ""},
            "logical_asset_id": {"type": "string", "minLength": 1},
            "mode": {"const": "logical_contract"},
            "initial_position_enu_m": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {"type": "number"},
            },
            "category": {"const": category},
            "role": {"const": "semantic_context"},
            "route_waypoints_enu_m": {
                "type": "array", "items": {
                    "type": "array", "minItems": 3, "maxItems": 3,
                    "items": {"type": "number"},
                },
            },
            "semantic_role": {"type": "string", "minLength": 1},
            "semantic_scope": scope,
            "state_sequence": {"type": "array", "items": {
                "type": "string", "minLength": 1,
            }},
            "task_id": {"type": "string", "minLength": 1},
        },
    }


def arm_branch_roster_schema(level: str) -> dict:
    """Return a fresh, closed JSON-root schema for the requested ARM layer."""
    if level not in _LEVELS:
        raise ValueError(f"unsupported ARM roster level: {level!r}")
    suffix = level.lower()
    module = import_module(f"Dataset.world_model.graph.formal_roster_{suffix}_structure")
    formal = module.formal_roster_schema()
    branches = _branches(formal)
    # L1/L2/L3/L5/L6/X formal conversion can replace authored semantic
    # background vehicles with SUMO entries. ARM keeps their engine identity.
    if level != "L4":
        from Dataset.world_model.graph.formal_roster_l4_structure import (
            _semantic_background_vehicle_row,
        )
        branches.append(_semantic_background_vehicle_row())
    if level in {"L2", "L4"}:
        branches.append(_logical_anchor_branch(
            "facade_anchor", "semantic.asset_anchor", "asset_anchor"))
    if level == "L3":
        branches.append(_logical_anchor_branch(
            "vehicle_anchor", "semantic.spawn_zone", "spawn_zone"))
    if level in {"L3", "L4", "X"}:
        branches.append(_logical_anchor_branch(
            "crowd_anchor", "semantic.spawn_zone", "spawn_zone"))
    if level == "L6":
        branches.append(_logical_anchor_branch(
            "facility", "semantic.asset_anchor", "asset_anchor",
            entity_kind="ground_station.ground_control_station",
            entity_type="facility.asset_anchor"))
    return {
        "$schema": _DRAFT,
        "type": "object", "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {
                "type": "array",
                "items": {"oneOf": [
                    formal["properties"]["entities"]["items"],
                    {"anyOf": [_engine_branch(branch) for branch in branches]},
                ]},
            },
        },
    }
