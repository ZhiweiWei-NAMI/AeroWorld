"""Closed whole-frame grammar for L2 formal truth JSONL rows.

``convert_to_render_ready.py`` writes one complete 26-key frame per JSONL
line.  ``filter_render_ready_truth_for_capture.py`` crops the entity list and
rebuilds its dependent summaries without changing that record carrier.  The
root and the three SUMO lifecycle forms are shared with L1; this module starts
from detached L1 schema fragments and projects the 16 closed L2 formal-roster
leaves into frame entities.

The projection yields 18 mutually exclusive frame branches: three SUMO
lifecycle/retention forms, one global-flow UAV, an ordinary and a repaired
global pad, and twelve local L2 roster forms.  Only fields copied by
``preserved_fields_from`` (plus the post-conversion scene occupancy repair)
survive from a local roster row.  Dynamic frame maps remain inherited from L1,
so entity and traffic-light-controller keys stay instance-bound instead of
becoming schema properties.

The frozen L2-5 frames and the current incident writer differ at one governed
path.  Frozen all-red incidents omit ``traffic_light_id`` because their plan
stores it only below ``traffic_light``; current ``_compact_incident`` requires
and writes the top-level id for the same accident class.  Two closed branches,
selected by that key's presence, retain both authorities without admitting an
unconstrained union.  Frames carry no digest, hash, or integrity field, so no
publication exclusion applies.

Schema factories used here return fresh structures.  This module reads no
data and compiles nothing at import time.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

from Dataset.tools.runtime_state_contract import RUNTIME_STATE_FIELDS
from Dataset.world_model.graph import formal_roster_l2_structure as _roster_l2
from Dataset.world_model.graph import formal_truth_frame_l1_structure as _shared_l1

__all__ = ["formal_truth_frame_schema"]


_EPISODE_PATTERN = (
    r"^(?:L2-[1-4]_v[12]|L2-5_v1)__seed(?:00|01|02)$"
)
_FRAME_PATTERN = (
    r"^(?:L2-[1-4]_v[12]|L2-5_v1)__seed(?:00|01|02)_tick_[0-9]+$"
)

_IDENTITY_FIELDS = frozenset({
    "entity_id", "entity_category", "entity_kind", "entity_type",
    "label_class", "logical_asset_id", "proxy_template_id", "roi_id",
    "site_id", "tags",
})

# Dataset/tools/convert_to_render_ready.py:PRESERVED_ENTITY_FIELDS.  The list
# stays local so importing this pure schema module does not import the writer.
_PRESERVED_ENTITY_FIELDS = frozenset({
    "semantic_scope", "entity_kind", "task_id", "role", "state_sequence",
    "semantic_role", "background_role", "contract_scenario_id",
    "contract_inspect_uav", "capture_boundary", "capture_boundary_id",
    "capture_contract", "background_vehicle", "background_pedestrian",
    "ground_flow_contract", "event_actor_motion_contract",
    "contract_facility", "contract_logical_sidecar", "motion_contract",
    "uav_corridor_role", "uav_corridor", "inspect_capture_corridor",
    "uav_boundary_crossing_required", "uav_crosses_boundary",
    "inspect_observes_boundary", "mission_boundary_crossing",
    "pad_boundary_policy", "inspect_fov_coverage_required",
    "assigned_altitude_m", "inspect_altitude_code", "inspect_altitude_m",
    "min_path_length_m", "full_episode_presence", "observer_lifecycle",
    "ground_reference_z_m", "ground_reference_source", "lifecycle",
    "activation_tick", "deactivation_tick", "spawn_policy", "category",
    "route_waypoints_enu_m", "planned_route_waypoints_enu_m",
    "path_deviation_contract", "task_kind", "task_state",
    *RUNTIME_STATE_FIELDS,
})

_FRAME_CORE_REQUIRED = frozenset({
    "annotations", "entity_category", "entity_id", "entity_kind",
    "entity_type", "label_class", "logical_asset_id", "proxy_template_id",
    "render_presence", "roi_id", "runtime_visibility", "site_id",
    "state_revision", "tags", "truth_pose", "visual_revision",
})

_SUMO_ROSTER_OVERRIDES = (
    "active_semantic_event",
    "canonical_logical_asset_id",
    "control_role",
    "logical_asset_authority",
    "semantic_metadata",
    "semantic_vehicle_state",
    "vehicle_type",
)


def _closed_branch(properties: Mapping[str, Any],
                   required: Iterable[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(set(required)),
        "properties": dict(properties),
    }


def _roster_branch_sets(
    roster_schema: Mapping[str, Any],
) -> tuple[dict, dict, list[dict], list[dict]]:
    """Split L2 roster leaves by producer marker, never by list position."""
    try:
        branches = roster_schema["properties"]["entities"]["items"]["oneOf"]
    except (KeyError, TypeError) as exc:
        raise ValueError("L2 formal-roster schema has no entity branches") from exc

    sumo: list[dict] = []
    global_uav: list[dict] = []
    global_pad: list[dict] = []
    local: list[dict] = []
    for branch in branches:
        if (branch.get("type") != "object" or
                branch.get("additionalProperties") is not False):
            raise ValueError("L2 formal-roster entity branch is not closed")
        properties = branch.get("properties", {})
        source = properties.get("source", {}).get("const")
        if source == "sumo_traci":
            sumo.append(branch)
        elif source == "uav_global_flow" and "uav_global_flow" in properties:
            global_uav.append(branch)
        elif source == "uav_global_flow" and "uav_global_pad" in properties:
            global_pad.append(branch)
        else:
            local.append(branch)

    counts = (len(branches), len(sumo), len(global_uav),
              len(global_pad), len(local))
    if counts != (16, 1, 1, 2, 12):
        raise ValueError(
            "L2 formal-roster branch contract changed: expected "
            "16 leaves = 1 SUMO + 1 global UAV + 2 global pads + 12 local"
        )
    repair_markers = sorted(
        "scene_occupancy_repair" in branch["properties"]
        for branch in global_pad
    )
    if repair_markers != [False, True]:
        raise ValueError(
            "L2 global-pad branches must contain one repaired and one ordinary row"
        )
    return sumo[0], global_uav[0], global_pad, local


def _frame_core(roster_branch: Mapping[str, Any]) -> dict:
    properties = _shared_l1._entity_core_properties()
    roster_properties = roster_branch["properties"]
    for name in _IDENTITY_FIELDS & roster_properties.keys():
        properties[name] = copy.deepcopy(roster_properties[name])
    return properties


def _local_truth_branch(roster_branch: Mapping[str, Any]) -> dict:
    roster_properties = roster_branch["properties"]
    roster_required = set(roster_branch["required"])
    properties = _frame_core(roster_branch)
    copied = _PRESERVED_ENTITY_FIELDS | {"scene_occupancy_repair"}
    for name in copied & roster_properties.keys():
        properties[name] = copy.deepcopy(roster_properties[name])

    required = _FRAME_CORE_REQUIRED | (copied & roster_required)
    if roster_properties.get("category", {}).get("const") == "uav":
        required |= {"uav_visibility"}
    return _closed_branch(properties, required)


def _sumo_truth_branch(base: Mapping[str, Any],
                       roster_branch: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(base))
    properties = result["properties"]
    roster_properties = roster_branch["properties"]

    for name in _IDENTITY_FIELDS & roster_properties.keys():
        properties[name] = copy.deepcopy(roster_properties[name])
    for name in (
        "background_role", "background_vehicle", "ground_flow_contract",
        "source", "sumo_segment",
    ):
        properties[name] = copy.deepcopy(roster_properties[name])

    # The converter may merge every validated runtime family into a SUMO row.
    # L1 intentionally declares only its two used families; L2's formal roster
    # provides the complete governed family rules for this layer.
    for name in RUNTIME_STATE_FIELDS:
        properties.pop(name, None)
    for name in RUNTIME_STATE_FIELDS:
        if name in roster_properties:
            properties[name] = copy.deepcopy(roster_properties[name])

    sumo_vehicle = properties["sumo_vehicle"]
    roster_vehicle = roster_properties["sumo_vehicle"]
    for name in _SUMO_ROSTER_OVERRIDES:
        sumo_vehicle["properties"][name] = copy.deepcopy(
            roster_vehicle["properties"][name]
        )
    return result


def _global_uav_truth_branch(roster_branch: Mapping[str, Any]) -> dict:
    result = _shared_l1._uav_flow_row()
    properties = result["properties"]
    roster_properties = roster_branch["properties"]
    for name in _IDENTITY_FIELDS & roster_properties.keys():
        properties[name] = copy.deepcopy(roster_properties[name])
    for name in (
        "background_role", "motion_contract", "planned_route_waypoints_enu_m",
        "semantic_role", "source", "uav_segment",
    ):
        properties[name] = copy.deepcopy(roster_properties[name])

    payload = properties["uav_global_flow"]
    roster_payload = roster_properties["uav_global_flow"]
    payload["properties"].update(copy.deepcopy(roster_payload["properties"]))
    payload["required"] = sorted(
        set(payload["required"]) | set(roster_payload["required"])
    )
    result["required"] = sorted(
        set(result["required"]) | {"planned_route_waypoints_enu_m"}
    )
    return result


def _global_pad_truth_branch(roster_branch: Mapping[str, Any]) -> dict:
    result = _shared_l1._pad_row()
    properties = result["properties"]
    roster_properties = roster_branch["properties"]
    for name in _IDENTITY_FIELDS & roster_properties.keys():
        properties[name] = copy.deepcopy(roster_properties[name])
    for name in (
        "background_role", "motion_contract", "semantic_scope", "source",
        "uav_global_pad",
    ):
        properties[name] = copy.deepcopy(roster_properties[name])

    required = set(result["required"]) | {"semantic_scope"}
    if "scene_occupancy_repair" in roster_properties:
        properties["scene_occupancy_repair"] = copy.deepcopy(
            roster_properties["scene_occupancy_repair"]
        )
        required.add("scene_occupancy_repair")
    result["required"] = sorted(required)
    return result


def _entities(roster_schema: Mapping[str, Any]) -> dict:
    sumo, global_uav, global_pads, local = _roster_branch_sets(roster_schema)
    sumo_forms = [
        _sumo_truth_branch(
            _shared_l1._sumo_row(
                required_lifecycle=required_lifecycle,
                retained=retained,
            ),
            sumo,
        )
        for required_lifecycle, retained in (
            (False, False),
            (True, False),
            (True, True),
        )
    ]
    branches = [
        *sumo_forms,
        _global_uav_truth_branch(global_uav),
        *(_global_pad_truth_branch(branch) for branch in global_pads),
        *(_local_truth_branch(branch) for branch in local),
    ]
    if len(branches) != 18:
        raise ValueError("L2 truth-frame projection must produce 18 branches")
    return {"type": "array", "items": {"oneOf": branches}}


def _incident_row(shared_schema: Mapping[str, Any]) -> dict:
    base = copy.deepcopy(
        shared_schema["properties"]["sumo_active_incidents"]["items"]
    )
    if "traffic_light_id" not in base["properties"]:
        raise ValueError("shared incident rule lacks traffic_light_id")
    base["properties"]["accident_class"] = {
        "const": "traffic_light_all_red_fault",
    }

    # Frozen L2-5 carrier: its producer plan nests the controller id below
    # traffic_light, while the compact frame row omits that plan-only object.
    frozen = copy.deepcopy(base)
    frozen["properties"].pop("traffic_light_id")
    frozen["required"] = [
        name for name in frozen["required"] if name != "traffic_light_id"
    ]

    # Current truth_integration._compact_incident branch: the same accident
    # class fails closed without a nonempty top-level controller id.
    current = copy.deepcopy(base)
    current["required"] = sorted(
        set(current["required"]) | {"traffic_light_id"}
    )
    return {"oneOf": [frozen, current]}


def _sumo_semantics(shared_schema: Mapping[str, Any],
                    incident: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(shared_schema["properties"]["sumo_semantics"])
    branches = result.get("oneOf", [result])
    objects = [branch for branch in branches if branch.get("type") == "object"]
    if len(objects) != 1:
        raise ValueError("shared sumo_semantics must have one object branch")
    objects[0]["properties"]["active_incidents"] = {
        "type": "array",
        "items": copy.deepcopy(dict(incident)),
    }
    return result


def formal_truth_frame_schema() -> dict:
    """Return the closed schema for one L2 ``truth_frames.jsonl`` row."""
    shared = _shared_l1.formal_truth_frame_schema()
    roster = _roster_l2.formal_roster_schema()
    schema = copy.deepcopy(shared)
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "formal_truth_frame_l2.record.schema.json"
    )
    schema["title"] = "AERO_WORLD formal_truth_frame L2 frame record schema"
    schema["properties"]["episode_id"] = {
        "type": "string",
        "pattern": _EPISODE_PATTERN,
    }
    schema["properties"]["frame_id"] = {
        "type": "string",
        "pattern": _FRAME_PATTERN,
    }
    schema["properties"]["entities"] = _entities(roster)
    incident = _incident_row(shared)
    schema["properties"]["sumo_active_incidents"] = {
        "type": "array",
        "items": copy.deepcopy(incident),
    }
    schema["properties"]["sumo_semantics"] = _sumo_semantics(
        shared, incident
    )
    return schema
