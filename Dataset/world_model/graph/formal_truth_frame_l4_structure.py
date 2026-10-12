"""Closed whole-row grammar for L4 formal truth frames.

``convert_to_render_ready.convert_episode`` writes one complete frame per
JSONL row.  ``filter_render_ready_truth_for_capture`` crops entity lists and
rebuilds their summaries without changing the record carrier.  The root and
the SUMO/global-flow writers are shared with L1, so this module starts from a
detached L1 schema and replaces the parts whose producer space differs at L4.

Local L4 entity branches are derived from the public L4 formal-roster schema
using the converter's explicit ``PRESERVED_ENTITY_FIELDS`` projection.  The
derivation retains each roster branch separately, copies only producer-owned
preserved fields, adds the fixed frame fields, and closes every result.  It
therefore keeps event actors, logical facade anchors, pedestrians, facilities,
corridors, and scoped UAVs distinct instead of accepting an optional-key
union.  SUMO, global UAV, and global-pad rows use their dedicated frame
writers and are declared separately.

The schema constructors called here are pure and return detached objects.
This module reads no data and compiles nothing at import time.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

from Dataset.world_model.graph import formal_roster_l4_structure as _roster_l4
from Dataset.world_model.graph import formal_truth_frame_l1_structure as _shared_l1
from Dataset.world_model.graph.structure_fragments import (
    runtime_state_source_guard_rule,
)

__all__ = ["formal_truth_frame_schema"]


_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

_L4_PROFILE_PATTERN = (
    r"(?:L4-1_v[12]|L4-2_v[12]|L4-3_v[123]|L4-4_v[12]|L4-5_v[123]|"
    r"L4-6_v[12]|L4-7_v[12]|L4-8_v[12]|L4-9_v[12]|L4-10_v[12]|"
    r"L4-11_v[12])"
)
_EPISODE_PATTERN = rf"^{_L4_PROFILE_PATTERN}__seed(?:00|01|02)$"
_FRAME_PATTERN = (
    rf"^{_L4_PROFILE_PATTERN}__seed(?:00|01|02)_tick_[0-9]+$"
)

_IDENTITY_FIELDS = frozenset({
    "entity_id", "entity_category", "entity_kind", "entity_type",
    "label_class", "logical_asset_id", "proxy_template_id", "roi_id",
    "site_id", "tags",
})

# Dataset/tools/convert_to_render_ready.py:PRESERVED_ENTITY_FIELDS.  Keeping
# this list local avoids importing the executable converter into a pure schema
# module.  A roster field outside this set is not copied into a local frame
# entity by ``preserved_fields_from``.
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
    "control_state", "sensor_state", "vehicle_state", "mission_state",
    "incident_state", "pedestrian_state", "facility_state",
    "navigation_state", "security_state", "communication_state",
})

_RUNTIME_STATE_FIELDS = frozenset({
    "control_state", "sensor_state", "vehicle_state", "mission_state",
    "incident_state", "pedestrian_state", "facility_state",
    "navigation_state", "security_state", "communication_state",
})

_FRAME_CORE_REQUIRED = (
    "annotations", "entity_category", "entity_id", "entity_kind",
    "entity_type", "label_class", "logical_asset_id", "proxy_template_id",
    "render_presence", "roi_id", "runtime_visibility", "site_id",
    "state_revision", "tags", "truth_pose", "visual_revision",
)


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _float2() -> dict:
    return {
        "type": "array", "minItems": 2, "maxItems": 2,
        "items": _float_leaf(),
    }


def _float3() -> dict:
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": _float_leaf(),
    }


def _float3_array() -> dict:
    return {"type": "array", "items": _float3()}


def _nullable_float() -> dict:
    return {"oneOf": [_float_leaf(), {"type": "null"}]}


def _truth_pose() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "authority_mode", "authority_owner", "coordinate_contract_id",
            "position_enu_m", "rotation_deg", "velocity_enu_mps",
        ],
        "properties": {
            "authority_mode": {"const": "authoritative_input"},
            "authority_owner": {"const": "dataset_converter"},
            "coordinate_contract_id": {"const": "coord.external_enu_m.v1"},
            "position_enu_m": _float3(),
            "rotation_deg": {
                "type": "object", "additionalProperties": False,
                "required": ["pitch_deg", "roll_deg", "yaw_deg"],
                "properties": {
                    "pitch_deg": {"const": 0.0},
                    "roll_deg": {"const": 0.0},
                    "yaw_deg": _float_leaf(),
                },
            },
            "velocity_enu_mps": _float3(),
        },
    }


def _render_presence() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "global_roster", "offstage", "offstage_reason",
            "roi_membership", "submission_state", "visibility_state",
        ],
        "properties": {
            "global_roster": {"const": True},
            "offstage": {"const": False},
            "offstage_reason": {"const": "none"},
            "roi_membership": {
                "type": "array", "minItems": 1, "maxItems": 1,
                "items": _identifier(),
            },
            "submission_state": {"const": "submit_to_ue"},
            "visibility_state": {"const": "visible"},
        },
    }


def _runtime_visibility() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "distance_to_capture_boundary_m", "expanded_boundary_padding_m",
            "policy", "runtime_visible",
        ],
        "properties": {
            "distance_to_capture_boundary_m": _nullable_float(),
            "expanded_boundary_padding_m": {"const": 60.0},
            "policy": {
                "const": "roi_polygon_expanded_60m_runtime_truth_crop_v1",
            },
            "runtime_visible": {"type": "boolean"},
        },
    }


def _sumo_runtime_visibility(*, retained: bool) -> dict:
    schema = _runtime_visibility()
    schema["properties"]["runtime_visible"] = {"const": not retained}
    if retained:
        schema["required"].extend([
            "retained_outside_runtime_boundary", "retention_policy",
        ])
        schema["properties"].update({
            "retained_outside_runtime_boundary": {"const": True},
            "retention_policy": {
                "const": "required_sumo_vehicle_preserve_source_lifecycle_v1",
            },
        })
    return schema


def _uav_visibility() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "camera_call_policy", "capture_roi_policy",
            "inspect_observation_distance_m", "roi_capture_distance_m",
            "roi_capture_eligible", "selected_for_capture_truth",
        ],
        "properties": {
            "camera_call_policy": {
                "const": "do_not_call_camera_when_roi_capture_eligible_is_false",
            },
            "capture_roi_policy": {
                "const": "camera_only_while_uav_position_inside_capture_roi_polygon_v1",
            },
            "inspect_observation_distance_m": _float_leaf(),
            "roi_capture_distance_m": _nullable_float(),
            "roi_capture_eligible": {"type": "boolean"},
            "selected_for_capture_truth": {"type": "boolean"},
        },
    }


def _annotations() -> dict:
    activity = {
        "type": "object", "additionalProperties": False,
        "required": [
            "activity_type", "animation_hint", "posture", "social_state",
        ],
        "properties": {
            "activity_type": _identifier(),
            "animation_hint": _identifier(),
            "posture": _identifier(),
            "social_state": _identifier(),
        },
    }
    network = {
        "type": "object", "additionalProperties": False,
        "required": ["latency_ms", "packet_loss", "status"],
        "properties": {
            "latency_ms": {"const": 0.0},
            "packet_loss": {"const": 0.0},
            "status": {"const": "nominal"},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["activity_type", "speed_mps", "state_facets"],
        "properties": {
            "activity_type": _identifier(),
            "speed_mps": _float_leaf(),
            "state_facets": {
                "type": "object", "additionalProperties": False,
                "required": ["activity", "network"],
                "properties": {"activity": activity, "network": network},
            },
        },
    }


def _frame_core_properties(roster_properties: Mapping[str, Any]) -> dict:
    properties = {
        "annotations": _annotations(),
        "entity_category": _identifier(),
        "entity_id": _identifier(),
        "entity_kind": _identifier(),
        "entity_type": _identifier(),
        "label_class": _identifier(),
        "logical_asset_id": _identifier(),
        # ``logical_only_profile`` deliberately writes an empty proxy id for
        # semantic facade anchors.
        "proxy_template_id": {"type": "string"},
        "render_presence": _render_presence(),
        "roi_id": _identifier(),
        "runtime_visibility": _runtime_visibility(),
        "site_id": _identifier(),
        "state_revision": {"type": "integer", "minimum": 1},
        "tags": {"type": "array", "items": _identifier()},
        "truth_pose": _truth_pose(),
        "visual_revision": {"const": 1},
    }
    # The roster branches pin logical-only and category-specific identities.
    # Those rules remain valid after the converter creates the frame entity.
    for name in _IDENTITY_FIELDS:
        if name in roster_properties:
            properties[name] = copy.deepcopy(roster_properties[name])
    return properties


def _closed_branch(properties: Mapping[str, Any], required: Iterable[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(set(required)),
        "properties": dict(properties),
    }


def _leaf_branches(schema: Mapping[str, Any]) -> list[dict]:
    """Flatten only the roster's branch-only ``oneOf`` wrappers."""
    if set(schema) == {"oneOf"}:
        result: list[dict] = []
        for child in schema["oneOf"]:
            result.extend(_leaf_branches(child))
        return result
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("L4 roster entity branch is not a closed object")
    return [copy.deepcopy(dict(schema))]


def _runtime_guarded(name: str, schema: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(schema))
    if name in _RUNTIME_STATE_FIELDS:
        result.update(runtime_state_source_guard_rule("engine_whole"))
    return result


def _roster_branch_sets(roster_schema: Mapping[str, Any]) -> tuple[dict, dict, dict, list[dict]]:
    try:
        item_schema = roster_schema["properties"]["entities"]["items"]
    except (KeyError, TypeError) as exc:
        raise ValueError("L4 formal-roster schema has no entities item grammar") from exc
    leaves = _leaf_branches(item_schema)
    sumo: list[dict] = []
    global_uav: list[dict] = []
    global_pad: list[dict] = []
    local: list[dict] = []
    for branch in leaves:
        properties = branch["properties"]
        source = properties.get("source", {}).get("const")
        if source == "sumo_traci":
            sumo.append(branch)
        elif source == "uav_global_flow" and "uav_global_flow" in properties:
            global_uav.append(branch)
        elif source == "uav_global_flow" and "uav_global_pad" in properties:
            global_pad.append(branch)
        elif source == "uav_global_flow":
            raise ValueError(
                "L4 formal-roster uav_global_flow branch carries no global "
                "flow or global pad payload")
        elif "source" not in properties:
            # Scoped, contract and other local producers write no source
            # discriminator, so every such leaf maps 1:1 onto a local frame
            # entity.  Branch counts grow with the roster and are not pinned.
            local.append(branch)
        else:
            raise ValueError(
                "L4 formal-roster branch has an unknown source discriminator: "
                f"{source!r}")
    if not (len(sumo) == len(global_uav) == len(global_pad) == 1):
        raise ValueError(
            "L4 formal-roster branch contract changed: expected exactly one "
            "SUMO, one global UAV and one global pad producer branch"
        )
    return sumo[0], global_uav[0], global_pad[0], local


def _local_truth_branch(roster_branch: Mapping[str, Any]) -> dict:
    roster_properties = roster_branch["properties"]
    roster_required = set(roster_branch["required"])
    properties = _frame_core_properties(roster_properties)
    copied = _PRESERVED_ENTITY_FIELDS | {"scene_occupancy_repair"}
    for name in copied:
        if name in roster_properties:
            properties[name] = _runtime_guarded(name, roster_properties[name])
    # Source rows carry state only when the sampled trajectory wrote a
    # nonempty state.  Its absence is a producer branch, not missing data.
    properties["state"] = _identifier()
    required = set(_FRAME_CORE_REQUIRED) | (copied & roster_required)
    if roster_properties.get("category", {}).get("const") == "uav":
        properties["uav_visibility"] = _uav_visibility()
        required.add("uav_visibility")
    return _closed_branch(properties, required)


def _global_uav_truth_branch(roster_branch: Mapping[str, Any]) -> dict:
    roster_properties = roster_branch["properties"]
    properties = _frame_core_properties(roster_properties)
    for name in (
        "background_role", "motion_contract", "planned_route_waypoints_enu_m",
        "semantic_role", "source", "task_id", "uav_segment",
    ):
        properties[name] = copy.deepcopy(roster_properties[name])

    payload = copy.deepcopy(roster_properties["uav_global_flow"])
    payload["properties"].update({
        "active_event_ids": {"type": "array", "items": _identifier()},
        "source_alpha": _float_leaf(),
        "source_next_time_s": _float_leaf(),
        "source_prev_time_s": _float_leaf(),
    })
    payload["required"] = sorted(set(payload["required"]) | {
        "active_event_ids", "source_alpha", "source_next_time_s",
        "source_prev_time_s",
    })
    properties["uav_global_flow"] = payload
    properties["uav_visibility"] = _uav_visibility()
    # The writer adds these top-level values only when the sampled flow row
    # supplies a nonempty value.
    properties["mission_type"] = copy.deepcopy(
        payload["properties"]["mission_type"])
    properties["task_id"] = copy.deepcopy(payload["properties"]["task_id"])
    required = set(_FRAME_CORE_REQUIRED) | {
        "background_role", "motion_contract", "planned_route_waypoints_enu_m",
        "semantic_role", "source", "uav_global_flow", "uav_segment",
        "uav_visibility",
    }
    return _closed_branch(properties, required)


def _global_pad_truth_branch(roster_branch: Mapping[str, Any]) -> dict:
    roster_properties = roster_branch["properties"]
    properties = _frame_core_properties(roster_properties)
    names = {
        "background_role", "motion_contract", "semantic_scope", "source",
        "uav_global_pad",
    }
    for name in names:
        properties[name] = copy.deepcopy(roster_properties[name])
    # convert_to_render_ready.py writes the closed scene-occupancy repair block
    # onto repaired facility pads.  The roster declares it without requiring it,
    # so it is copied verbatim as an optional field of the pad truth branch.
    properties["scene_occupancy_repair"] = _runtime_guarded(
        "scene_occupancy_repair", roster_properties["scene_occupancy_repair"])
    return _closed_branch(properties, set(_FRAME_CORE_REQUIRED) | names)


def _shared_sumo_branch(shared_schema: Mapping[str, Any]) -> dict:
    try:
        branches = shared_schema["properties"]["entities"]["items"]["oneOf"]
    except (KeyError, TypeError) as exc:
        raise ValueError("shared truth-frame schema has no entity branches") from exc
    found = []
    for branch in branches:
        properties = branch.get("properties", {})
        if properties.get("source", {}).get("const") != "sumo_traci":
            continue
        vehicle_required = set(
            properties["sumo_vehicle"].get("required", []))
        visibility_required = set(
            properties["runtime_visibility"].get("required", []))
        if (
            "required_lifecycle_preservation_policy" not in vehicle_required
            and "retained_outside_runtime_boundary" not in visibility_required
        ):
            found.append(branch)
    if len(found) != 1:
        raise ValueError(
            "shared truth-frame schema must have one ordinary in-boundary "
            "SUMO entity branch"
        )
    return copy.deepcopy(found[0])


def _visual_state() -> dict:
    # build_semantic_vehicle_plans forwards a governed set_visual_state
    # payload.  Named common keys stay exact; other authored keys retain the
    # action grammar's constrained key and scalar/list value space.
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        "properties": {
            "lights_on": {"type": "boolean"},
            "mode": _identifier(),
        },
        "patternProperties": {
            "^(?!lights_on$|mode$)[A-Za-z][A-Za-z0-9_]*$": {
                "oneOf": [
                    _identifier(), {"type": "boolean"}, {"type": "number"},
                    {"type": "array", "minItems": 1,
                     "items": _identifier()},
                ],
            },
        },
    }


def _active_semantic_event() -> dict:
    return {
        "oneOf": [
            {"type": "null"},
            {
                "type": "object", "additionalProperties": False,
                "required": ["event_id", "intent", "scenario_id", "title"],
                "properties": {
                    # The plan writer applies ``str(value or "")`` before
                    # serializing these four fields.
                    "event_id": {"type": "string"},
                    "intent": {"type": "string"},
                    "observation_only": {"const": True},
                    "scenario_id": {"type": "string"},
                    "title": {"type": "string"},
                },
            },
        ],
    }


def _semantic_vehicle_state() -> dict:
    return {
        "oneOf": [
            {"type": "null"},
            {
                "type": "object", "additionalProperties": False,
                "required": ["event_id", "intent", "mode", "visual_state"],
                "properties": {
                    "event_id": {"type": "string"},
                    "intent": {"type": "string"},
                    "mode": _identifier(),
                    "visual_state": _visual_state(),
                },
            },
        ],
    }


def _semantic_metadata(roster_rule: Mapping[str, Any],
                       canonical_asset_rule: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(roster_rule))
    authorities = {
        "enum": [
            "explicit_vehicle_plan_type", "source_roster",
            "sumo_vehicle_type",
        ],
    }
    for branch in result["oneOf"]:
        branch["properties"]["logical_asset_authority"] = copy.deepcopy(
            authorities)
    # explicit_vehicle_plan.py's ROI endpoint picker writes this scope when the assigned
    # endpoint is a local component source (roi_local_component_source), the
    # paired alternative to roi_background_route_crossing_flow.  Additive to the
    # copied roster enum: every other member and rule stays exactly as declared.
    scope_rules = [
        branch["properties"]["traffic_coverage_scope"]
        for branch in result["oneOf"]
        if type(branch.get("properties", {}).get("traffic_coverage_scope")) is dict
        and "enum" in branch["properties"]["traffic_coverage_scope"]
    ]
    if len(scope_rules) != 1:
        raise ValueError(
            "L4 formal-roster semantic metadata must declare exactly one "
            "traffic_coverage_scope enum")
    members = list(scope_rules[0]["enum"])
    if "roi_background_local_component_flow" not in members:
        members.append("roi_background_local_component_flow")
    scope_rules[0]["enum"] = members
    # Scenario vehicles inserted by build_semantic_vehicle_plans carry this
    # fixed metadata block.  The converter adds logical_asset_authority.
    result["oneOf"].append({
        "type": "object", "additionalProperties": False,
        "required": [
            "logical_asset_authority", "logical_asset_id",
            "scenario_vehicle_actor", "semantic_event_actor",
            "source_activation_tick", "source_entity_id", "vehicle_role",
        ],
        "properties": {
            "logical_asset_authority": authorities,
            "logical_asset_id": copy.deepcopy(dict(canonical_asset_rule)),
            "scenario_vehicle_actor": {"const": True},
            "semantic_event_actor": {"type": "boolean"},
            "source_activation_tick": {"type": "integer", "minimum": 0},
            "source_entity_id": _identifier(),
            "vehicle_role": _identifier(),
        },
    })
    return result


def _sumo_vehicle(shared_sumo_branch: Mapping[str, Any],
                  roster_sumo_branch: Mapping[str, Any], *,
                  required_lifecycle: bool) -> dict:
    shared = copy.deepcopy(
        shared_sumo_branch["properties"]["sumo_vehicle"])
    roster = roster_sumo_branch["properties"]["sumo_vehicle"]
    roster_properties = roster["properties"]
    properties = shared["properties"]
    properties.pop("required_lifecycle_preservation_policy", None)
    shared_required = set(shared["required"])
    shared_required.discard("required_lifecycle_preservation_policy")
    canonical_asset_rule = roster_properties["canonical_logical_asset_id"]
    properties.update({
        "active_semantic_event": _active_semantic_event(),
        "canonical_logical_asset_id": copy.deepcopy(canonical_asset_rule),
        "control_role": _identifier(),
        "logical_asset_authority": {
            "enum": [
                "explicit_vehicle_plan_type", "source_roster",
                "sumo_vehicle_type",
            ],
        },
        "semantic_metadata": _semantic_metadata(
            roster_properties["semantic_metadata"], canonical_asset_rule),
        "semantic_vehicle_state": _semantic_vehicle_state(),
        "vehicle_type": copy.deepcopy(roster_properties["vehicle_type"]),
    })
    # These fields are emitted by the per-frame SUMO truth writer even
    # though the roster-level schema stops at static road context.
    properties.update({
        "accel_mps2": _nullable_float(),
        "dimensions_m": {
            "type": "object", "additionalProperties": False,
            "required": ["height", "length", "width"],
            "properties": {
                "height": _float_leaf(), "length": _float_leaf(),
                "width": _float_leaf(),
            },
        },
        "signals": {"type": "integer"},
        "source_alpha": _float_leaf(),
        "source_next_time_s": _float_leaf(),
        "source_prev_time_s": _float_leaf(),
        "speed_mps": _nullable_float(),
        "sumo_angle_deg": _nullable_float(),
        "sumo_front_bumper_xy_m": {
            "oneOf": [_float2(), {"type": "null"}],
        },
        "sumo_xy_m": {"oneOf": [_float2(), {"type": "null"}]},
        "truth_front_bumper_enu_m": _float3(),
    })
    required = shared_required | {
        "accel_mps2", "dimensions_m", "signals", "source_alpha",
        "source_next_time_s", "source_prev_time_s", "speed_mps",
        "sumo_angle_deg", "sumo_front_bumper_xy_m", "sumo_xy_m",
        "truth_front_bumper_enu_m",
    }
    if required_lifecycle:
        properties["required_lifecycle_preservation_policy"] = {
            "const": "required_sumo_vehicle_preserve_source_lifecycle_v1",
        }
        required.add("required_lifecycle_preservation_policy")
    return _closed_branch(properties, required)


def _sumo_truth_branch(roster_branch: Mapping[str, Any],
                       shared_sumo_branch: Mapping[str, Any], *,
                       required_lifecycle: bool, retained: bool) -> dict:
    if retained and not required_lifecycle:
        raise ValueError("retained SUMO truth requires lifecycle preservation")
    roster_properties = roster_branch["properties"]
    properties = _frame_core_properties(roster_properties)
    properties["runtime_visibility"] = _sumo_runtime_visibility(
        retained=retained)
    for name in (
        "background_role", "background_vehicle", "ground_flow_contract",
        "source", "sumo_segment",
    ):
        properties[name] = copy.deepcopy(roster_properties[name])
    # The live SUMO control role is not restricted to background vehicles.
    properties["background_vehicle"]["properties"]["control_role"] = _identifier()
    properties["background_vehicle"]["properties"]["vehicle_type"] = copy.deepcopy(
        roster_properties["sumo_vehicle"]["properties"]["vehicle_type"])
    for name in _RUNTIME_STATE_FIELDS:
        if name in roster_properties:
            properties[name] = _runtime_guarded(name, roster_properties[name])
    properties.update({
        "capture_boundary_id": _identifier(),
        "state": _identifier(),
        "sumo_vehicle": _sumo_vehicle(
            shared_sumo_branch, roster_branch,
            required_lifecycle=required_lifecycle),
        "sumo_visibility": {
            "type": "object", "additionalProperties": False,
            "required": [
                "inspect_observation_distance_m",
                "selected_for_capture_truth",
            ],
            "properties": {
                "inspect_observation_distance_m": _float_leaf(),
                "selected_for_capture_truth": {"const": True},
            },
        },
    })
    required = set(_FRAME_CORE_REQUIRED) | {
        "background_role", "background_vehicle", "capture_boundary_id",
        "ground_flow_contract", "source", "state", "sumo_segment",
        "sumo_vehicle", "sumo_visibility",
    }
    return _closed_branch(properties, required)


def _entities(roster_schema: Mapping[str, Any],
              shared_schema: Mapping[str, Any]) -> dict:
    sumo, global_uav, global_pad, local = _roster_branch_sets(roster_schema)
    shared_sumo = _shared_sumo_branch(shared_schema)
    branches = [
        _sumo_truth_branch(
            sumo, shared_sumo, required_lifecycle=False, retained=False),
        _sumo_truth_branch(
            sumo, shared_sumo, required_lifecycle=True, retained=False),
        _sumo_truth_branch(
            sumo, shared_sumo, required_lifecycle=True, retained=True),
        _global_uav_truth_branch(global_uav),
        _global_pad_truth_branch(global_pad),
        *(_local_truth_branch(branch) for branch in local),
    ]
    return {"type": "array", "items": {"oneOf": branches}}


def _incident_row(shared_schema: Mapping[str, Any]) -> dict:
    shared = copy.deepcopy(
        shared_schema["properties"]["sumo_active_incidents"]["items"])
    shared["properties"]["intent_stage"] = {
        "oneOf": [_identifier(), {"type": "null"}],
    }
    return shared


def _sumo_semantics(shared_schema: Mapping[str, Any], incident: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(shared_schema["properties"]["sumo_semantics"])
    branches = result.get("oneOf", [result])
    objects = [branch for branch in branches if branch.get("type") == "object"]
    if len(objects) != 1:
        raise ValueError("shared sumo_semantics must have one object branch")
    payload = objects[0]
    payload["properties"]["traffic_light_count"] = {
        "type": "integer", "minimum": 0,
    }
    payload["properties"]["active_incidents"] = {
        "type": "array", "items": copy.deepcopy(dict(incident)),
    }
    payload["required"] = sorted(set(payload["required"]) | {
        "traffic_light_count",
    })
    return result if "oneOf" in result else {"oneOf": [result, {"type": "null"}]}


def _uav_frame_payload(shared_schema: Mapping[str, Any]) -> dict:
    shared = copy.deepcopy(shared_schema["properties"]["uav_global_flow"])
    branches = shared.get("oneOf", [shared])
    objects = [branch for branch in branches if branch.get("type") == "object"]
    if len(objects) != 1:
        raise ValueError("shared uav_global_flow must have one object branch")
    payload = objects[0]
    payload["properties"].pop("active_selected_vehicle_count", None)
    payload["required"] = [
        name for name in payload["required"]
        if name != "active_selected_vehicle_count"
    ]
    if "oneOf" in shared:
        return shared
    return {"oneOf": [shared, {"type": "null"}]}


def _object_branch(schema: Mapping[str, Any], *, name: str) -> dict:
    branches = schema.get("oneOf", [schema])
    objects = [branch for branch in branches if branch.get("type") == "object"]
    if len(objects) != 1:
        raise ValueError(f"{name} must have one object branch")
    return copy.deepcopy(objects[0])


def _sumo_root_presence(shared_schema: Mapping[str, Any],
                        incident: Mapping[str, Any],
                        semantics: Mapping[str, Any]) -> dict:
    return {
        "if": {
            "properties": {"sumo_segment": {"type": "null"}},
            "required": ["sumo_segment"],
        },
        "then": {
            "properties": {
                "sumo_active_incidents": {
                    "type": "array", "maxItems": 0,
                    "items": copy.deepcopy(dict(incident)),
                },
                "sumo_segment": {"type": "null"},
                "sumo_semantics": {"type": "null"},
                "sumo_traffic_light_states": {
                    "type": "object", "additionalProperties": False,
                    "maxProperties": 0,
                },
            },
        },
        "else": {
            "properties": {
                "sumo_active_incidents": {
                    "type": "array",
                    "items": copy.deepcopy(dict(incident)),
                },
                "sumo_segment": _object_branch(
                    shared_schema["properties"]["sumo_segment"],
                    name="sumo_segment"),
                "sumo_semantics": _object_branch(
                    semantics, name="sumo_semantics"),
                "sumo_traffic_light_states": copy.deepcopy(
                    shared_schema["properties"]["sumo_traffic_light_states"]),
            },
        },
    }


def _uav_root_presence(shared_schema: Mapping[str, Any],
                       payload: Mapping[str, Any]) -> dict:
    return {
        "if": {
            "properties": {"uav_segment": {"type": "null"}},
            "required": ["uav_segment"],
        },
        "then": {
            "properties": {
                "uav_global_flow": {"type": "null"},
                "uav_segment": {"type": "null"},
            },
        },
        "else": {
            "properties": {
                "uav_global_flow": _object_branch(
                    payload, name="uav_global_flow"),
                "uav_segment": _object_branch(
                    shared_schema["properties"]["uav_segment"],
                    name="uav_segment"),
            },
        },
    }


def formal_truth_frame_schema() -> dict:
    """Return the closed schema for one L4 ``truth_frames.jsonl`` row."""
    shared = _shared_l1.formal_truth_frame_schema()
    roster = _roster_l4.formal_roster_schema()
    schema = copy.deepcopy(shared)
    schema["$schema"] = _SCHEMA
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "formal_truth_frame_l4.record.schema.json"
    )
    schema["title"] = "AERO_WORLD formal_truth_frame L4 frame record schema"
    schema.pop("additionalProperties", None)
    schema["unevaluatedProperties"] = False
    incident = _incident_row(shared)
    semantics = _sumo_semantics(shared, incident)
    uav_payload = _uav_frame_payload(shared)
    schema["properties"]["episode_id"] = {
        "type": "string", "pattern": _EPISODE_PATTERN,
    }
    schema["properties"]["frame_id"] = {
        "type": "string", "pattern": _FRAME_PATTERN,
    }
    schema["properties"]["entities"] = _entities(roster, shared)
    grouped = (
        "sumo_active_incidents", "sumo_segment", "sumo_semantics",
        "sumo_traffic_light_states", "uav_global_flow", "uav_segment",
    )
    for name in grouped:
        schema["properties"].pop(name, None)
    schema.setdefault("allOf", []).extend([
        _sumo_root_presence(shared, incident, semantics),
        _uav_root_presence(shared, uav_payload),
    ])
    return schema
