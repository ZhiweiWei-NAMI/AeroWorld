"""Closed record grammar for one L6 formal truth-frame JSONL row.

``convert_to_render_ready.py`` writes the 26-key frame root in its tick
loop.  ``filter_render_ready_truth_for_capture.py`` keeps that root and
rebuilds the entity list, motion-state map, roster summary, and active flow
counts at the capture boundary.  A JSONL row is therefore the record root.

The scene-authored entity branches reuse fresh L6 roster schema fragments
and retain only fields copied by ``preserved_fields_from``.  Frame-only pose,
render-presence, visibility, SUMO sample, and flow fields are declared here.
The two instance-keyed maps use anchored key rules; entity and controller IDs
remain keys rather than becoming field declarations.

No frame field is an integrity/publication-exclusion field.  This module is a
pure schema constructor and performs no source reads or registration.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_roster_l4_structure as _sumo_roster
from Dataset.world_model.graph import formal_roster_l6_structure as _l6_roster

__all__ = ["formal_truth_frame_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


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


def _nullable_float() -> dict:
    return {"oneOf": [_float_leaf(), {"type": "null"}]}


def _segment(phase_rule: dict) -> dict:
    return _sumo_roster._segment(phase_rule)


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
            "retained_outside_runtime_boundary": {"const": True},
            "retention_policy": {
                "const": "required_sumo_vehicle_preserve_source_lifecycle_v1",
            },
            "runtime_visible": {"type": "boolean"},
        },
    }


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


def _entity_core_properties() -> dict:
    return {
        "annotations": _annotations(),
        "entity_category": _identifier(),
        "entity_id": _identifier(),
        "entity_kind": _identifier(),
        "entity_type": _identifier(),
        "label_class": _identifier(),
        "logical_asset_id": _identifier(),
        # Logical-only source rows legitimately carry an empty proxy template.
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


_ENTITY_CORE_REQUIRED = [
    "annotations", "entity_category", "entity_id", "entity_kind",
    "entity_type", "label_class", "logical_asset_id", "proxy_template_id",
    "render_presence", "roi_id", "runtime_visibility", "site_id",
    "state_revision", "tags", "truth_pose", "visual_revision",
]


def _entity_branch(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": list(dict.fromkeys(_ENTITY_CORE_REQUIRED + required)),
        "properties": {**_entity_core_properties(), **properties},
    }


_PRESERVED_TRUTH_FIELDS = frozenset({
    "semantic_scope", "task_id", "role", "state_sequence", "semantic_role",
    "background_role", "contract_scenario_id", "contract_inspect_uav",
    "capture_boundary", "capture_boundary_id", "capture_contract",
    "background_vehicle", "background_pedestrian", "ground_flow_contract",
    "event_actor_motion_contract", "contract_facility",
    "contract_logical_sidecar", "motion_contract", "uav_corridor_role",
    "uav_corridor", "inspect_capture_corridor",
    "uav_boundary_crossing_required", "uav_crosses_boundary",
    "inspect_observes_boundary", "mission_boundary_crossing",
    "pad_boundary_policy", "inspect_fov_coverage_required",
    "assigned_altitude_m", "inspect_altitude_code", "inspect_altitude_m",
    "min_path_length_m", "full_episode_presence", "observer_lifecycle",
    "ground_reference_z_m", "ground_reference_source", "lifecycle",
    "activation_tick", "deactivation_tick", "spawn_policy", "category",
    "route_waypoints_enu_m", "planned_route_waypoints_enu_m",
    "path_deviation_contract", "task_kind", "task_state",
    # repair_service_facility_placements adds this after the tick loop to
    # truth-frame entities with a successful service-facility reanchor.
    "scene_occupancy_repair",
})


def _source_truth_branch(roster_branch: dict) -> dict:
    allowed = _PRESERVED_TRUTH_FIELDS | frozenset(
        _l6_roster._runtime_state_properties()
    )
    properties = {
        name: rule for name, rule in roster_branch["properties"].items()
        if name in allowed
    }
    properties["runtime_visibility"] = _runtime_visibility()
    properties["state"] = _identifier()
    if roster_branch["properties"].get("category", {}).get("const") == "uav":
        properties["uav_visibility"] = _uav_visibility()
    if "scene_occupancy_repair" in properties:
        # repair_service_facility_placements copies only successful repairs
        # to frame entities; failed repairs remain roster-only.
        properties["scene_occupancy_repair"] = (
            _l6_roster._scene_occupancy_repair()["oneOf"][0]
        )
    required = [
        name for name in roster_branch["required"] if name in properties
    ]
    if "uav_visibility" in properties:
        required.append("uav_visibility")
    return _entity_branch(properties, required)


def _sumo_vehicle() -> dict:
    schema = _sumo_roster._sumo_vehicle()
    properties = schema["properties"]
    properties.update({
        "accel_mps2": _float_leaf(),
        # The current SUMO batch passes only instructions_from_explicit_plan
        # to run_traffic.  Its semantic controls are vehicle.moveToXY, and
        # its other instructions set neither aero.active_semantic_event nor
        # aero.semantic_vehicle_state.  The SUMO reader emits None for both
        # unset parameters; the converter and capture filter preserve them.
        # The separate build_semantic_vehicle_plans writer is not on this path.
        "active_semantic_event": {"type": "null"},
        "controlling_signal_state": {
            "enum": [
                "green", "mixed", "none", "off", "red", "red_yellow",
                "yellow",
            ],
        },
        "dimensions_m": {
            "type": "object", "additionalProperties": False,
            "required": ["height", "length", "width"],
            "properties": {
                "height": _float_leaf(),
                "length": _float_leaf(),
                "width": _float_leaf(),
            },
        },
        "required_lifecycle_preservation_policy": {
            "const": "required_sumo_vehicle_preserve_source_lifecycle_v1",
        },
        "semantic_vehicle_state": {"type": "null"},
        "signals": {"type": "integer"},
        "source_alpha": _float_leaf(),
        "source_next_time_s": _float_leaf(),
        "source_prev_time_s": _float_leaf(),
        "speed_mps": _float_leaf(),
        "sumo_angle_deg": _float_leaf(),
        "sumo_front_bumper_xy_m": _float2(),
        "sumo_xy_m": _float2(),
        "truth_front_bumper_enu_m": _float3(),
    })
    schema["required"] = schema["required"] + [
        "accel_mps2", "dimensions_m", "signals", "source_alpha",
        "source_next_time_s", "source_prev_time_s", "speed_mps",
        "sumo_angle_deg",
        "sumo_front_bumper_xy_m", "sumo_xy_m",
        "truth_front_bumper_enu_m",
    ]
    return schema


def _sumo_entity() -> dict:
    runtime_states = _l6_roster._runtime_state_properties()
    return _entity_branch(
        {
            **runtime_states,
            "background_role": {
                "enum": [
                    "sumo_background_traffic",
                    "sumo_semantic_event_vehicle",
                    "sumo_source_required_context_vehicle",
                ],
            },
            "background_vehicle": _sumo_roster._background_vehicle(),
            "capture_boundary_id": _identifier(),
            "ground_flow_contract": _sumo_roster._sumo_ground_flow_contract(),
            "source": {"const": "sumo_traci"},
            "state": _identifier(),
            "sumo_segment": _segment(_sumo_roster._SUMO_PHASE),
            "sumo_vehicle": _sumo_vehicle(),
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
        },
        [
            "background_role", "background_vehicle", "capture_boundary_id",
            "ground_flow_contract", "source", "state", "sumo_segment",
            "sumo_vehicle", "sumo_visibility",
        ],
    )


def _uav_global_flow_payload() -> dict:
    schema = _l6_roster._uav_global_flow_payload()
    schema["properties"].update({
        "active_event_ids": {"type": "array", "items": _identifier()},
        "source_alpha": _float_leaf(),
        "source_next_time_s": _float_leaf(),
        "source_prev_time_s": _float_leaf(),
    })
    schema["required"] = schema["required"] + [
        "active_event_ids", "source_alpha", "source_next_time_s",
        "source_prev_time_s",
    ]
    return schema


def _global_flow_entity() -> dict:
    roster_properties = _l6_roster._global_flow_uav_row()["properties"]
    return _entity_branch(
        {
            "background_role": roster_properties["background_role"],
            "mission_type": _identifier(),
            "motion_contract": roster_properties["motion_contract"],
            "planned_route_waypoints_enu_m": roster_properties[
                "planned_route_waypoints_enu_m"
            ],
            "semantic_role": roster_properties["semantic_role"],
            "source": roster_properties["source"],
            "task_id": roster_properties["task_id"],
            "uav_global_flow": _uav_global_flow_payload(),
            "uav_segment": roster_properties["uav_segment"],
            "uav_visibility": _uav_visibility(),
        },
        [
            "background_role", "motion_contract",
            "planned_route_waypoints_enu_m", "semantic_role", "source",
            "uav_global_flow", "uav_segment", "uav_visibility",
        ],
    )


def _global_pad_entity() -> dict:
    roster_properties = _l6_roster._global_pad_row()["properties"]
    return _entity_branch(
        {
            "background_role": roster_properties["background_role"],
            "motion_contract": roster_properties["motion_contract"],
            "semantic_scope": roster_properties["semantic_scope"],
            "source": roster_properties["source"],
            "uav_global_pad": roster_properties["uav_global_pad"],
        },
        [
            "background_role", "motion_contract", "semantic_scope", "source",
            "uav_global_pad",
        ],
    )


def _entities(*, sumo_enabled: bool, uav_enabled: bool) -> dict:
    source_branches = [
        _l6_roster._inspect_uav_row(),
        _l6_roster._mission_uav_row(),
        _l6_roster._observer_uav_row(),
        _l6_roster._semantic_facility_row(),
        _l6_roster._contract_facility_row(),
        _l6_roster._repaired_facility_row(),
        _l6_roster._pedestrian_row(),
        _l6_roster._corridor_row(),
        _l6_roster._gcs_anchor_row(),
    ]
    branches = [
        *([_sumo_entity()] if sumo_enabled else []),
        *(
            [_global_flow_entity(), _global_pad_entity()]
            if uav_enabled else []
        ),
        *(_source_truth_branch(branch) for branch in source_branches),
    ]
    return {
        "type": "array",
        "items": {"oneOf": branches},
    }


def _traffic_light_state() -> dict:
    link = {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "string"},
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "controlled_links", "next_switch_s", "phase_index", "program_id",
            "state",
        ],
        "properties": {
            "controlled_links": {
                "type": "array",
                "items": {"type": "array", "items": link},
            },
            "next_switch_s": _float_leaf(),
            "phase_index": {"type": "integer", "minimum": 0},
            "program_id": {"type": "string"},
            "state": _identifier(),
        },
    }


def _incident() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "accident_class", "affected_vehicle_ids", "anchor", "end_s",
            "episode_event_id", "episode_scenario_id", "incident_id",
            "injection_method", "intent", "intent_stage", "start_s",
        ],
        "properties": {
            "accident_class": _identifier(),
            "affected_vehicle_ids": {
                "type": "array", "items": _identifier(),
            },
            "anchor": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "geometry_source", "sumo_edge_id", "sumo_lane_id",
                    "truth_position_enu_m",
                ],
                "properties": {
                    "geometry_source": _identifier(),
                    "sumo_edge_id": _identifier(),
                    "sumo_lane_id": _identifier(),
                    "truth_position_enu_m": {
                        "oneOf": [_float3(), {"type": "null"}],
                    },
                },
            },
            "end_s": _float_leaf(),
            "episode_event_id": _identifier(),
            "episode_scenario_id": _identifier(),
            "incident_id": _identifier(),
            "injection_method": _identifier(),
            "intent": _identifier(),
            "intent_stage": _identifier(),
            "start_s": _float_leaf(),
            "traffic_light_id": _identifier(),
        },
        "allOf": [{
            "if": {
                "required": ["accident_class"],
                "properties": {
                    "accident_class": {"const": "traffic_light_all_red_fault"},
                },
            },
            "then": {"required": ["traffic_light_id"]},
        }],
    }


def _sumo_semantics() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "absolute_time_s", "active_incident_count", "active_incidents",
            "active_selected_vehicle_count", "enabled",
            "max_observation_distance_m",
            "retained_outside_boundary_sumo_vehicle_count",
            "runtime_boundary_visible_selected_vehicle_count", "scenario_id",
            "segment", "selected_vehicle_count",
            "semantic_lifecycle_outside_runtime_boundary_count",
            "source_alpha", "source_next_time_s", "source_prev_time_s",
            "source_vehicle_count", "traffic_light_count",
        ],
        "properties": {
            "absolute_time_s": _float_leaf(),
            "active_incident_count": {"type": "integer", "minimum": 0},
            "active_incidents": {"type": "array", "items": _incident()},
            "active_selected_vehicle_count": {
                "type": "integer", "minimum": 0,
            },
            "enabled": {"const": True},
            "max_observation_distance_m": {"const": 60.0},
            "retained_outside_boundary_sumo_vehicle_count": {
                "type": "integer", "minimum": 0,
            },
            "runtime_boundary_visible_selected_vehicle_count": {
                "type": "integer", "minimum": 0,
            },
            "scenario_id": _identifier(),
            "segment": _segment(_sumo_roster._SUMO_PHASE),
            "selected_vehicle_count": {"type": "integer", "minimum": 0},
            "semantic_lifecycle_outside_runtime_boundary_count": {
                "type": "integer", "minimum": 0,
            },
            "source_alpha": _float_leaf(),
            "source_next_time_s": _float_leaf(),
            "source_prev_time_s": _float_leaf(),
            "source_vehicle_count": {"type": "integer", "minimum": 0},
            "traffic_light_count": {"type": "integer", "minimum": 0},
        },
    }


def _uav_frame_flow() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "absolute_time_s", "active_selected_uav_count",
            "baseline_active_uavs", "camera_capture_roi_policy", "enabled",
            "minimum_active_uavs_required", "roi_capture_eligible_uav_count",
            "segment", "selected_uav_count", "source_alpha",
            "source_next_time_s", "source_prev_time_s", "source_uav_count",
            "task_count",
        ],
        "properties": {
            "absolute_time_s": _float_leaf(),
            "active_selected_uav_count": {
                "type": "integer", "minimum": 0,
            },
            "baseline_active_uavs": {"type": "integer", "minimum": 0},
            "camera_capture_roi_policy": {
                "const": "camera_only_while_uav_position_inside_capture_roi_polygon_v1",
            },
            "enabled": {"const": True},
            "minimum_active_uavs_required": {
                "type": "integer", "minimum": 0,
            },
            "roi_capture_eligible_uav_count": {
                "type": "integer", "minimum": 0,
            },
            "segment": _segment(_l6_roster._UAV_PHASE),
            "selected_uav_count": {"type": "integer", "minimum": 0},
            "source_alpha": _float_leaf(),
            "source_next_time_s": _float_leaf(),
            "source_prev_time_s": _float_leaf(),
            "source_uav_count": {"type": "integer", "minimum": 0},
            "task_count": {"type": "integer", "minimum": 0},
        },
    }


def _frame_root_schema(*, sumo_enabled: bool, uav_enabled: bool) -> dict:
    sumo_active_incidents = {
        "type": "array", "items": _incident(),
    }
    sumo_traffic_light_states = {
        "type": "object",
        "additionalProperties": _traffic_light_state(),
        "propertyNames": {
            "type": "string", "pattern": "^\\S+$",
        },
    }
    if not sumo_enabled:
        sumo_active_incidents["maxItems"] = 0
        sumo_traffic_light_states["maxProperties"] = 0
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "active_roi_id", "active_site_id", "capture_boundary_id", "dt_s",
            "entities", "entity_motion_state", "episode_id", "frame_id",
            "frame_seq", "inspect_observes_boundary", "map_id",
            "pad_boundary_policy", "render_mode", "roster_summary",
            "schema_name", "schema_version", "sim_time_s",
            "sumo_active_incidents", "sumo_segment", "sumo_semantics",
            "sumo_traffic_light_states", "tick", "tick_hz",
            "uav_crosses_boundary", "uav_global_flow", "uav_segment",
        ],
        "properties": {
            "active_roi_id": _identifier(),
            "active_site_id": _identifier(),
            "capture_boundary_id": _identifier(),
            "dt_s": {"const": 0.1},
            "entities": _entities(
                sumo_enabled=sumo_enabled,
                uav_enabled=uav_enabled,
            ),
            "entity_motion_state": {
                "type": "object",
                "additionalProperties": _identifier(),
                "propertyNames": {
                    "type": "string", "pattern": "^\\S+$",
                },
            },
            "episode_id": _identifier(),
            "frame_id": {
                "type": "string", "pattern": "^.+_tick_[0-9]+$",
            },
            "frame_seq": {"type": "integer", "minimum": 0},
            "inspect_observes_boundary": {"type": "boolean"},
            "map_id": _identifier(),
            "pad_boundary_policy": {
                "type": "object", "additionalProperties": False,
                "required": ["default", "inside_required_for"],
                "properties": {
                    "default": _identifier(),
                    "inside_required_for": {
                        "type": "array", "items": _identifier(),
                    },
                },
            },
            "render_mode": {"const": "ue_pie"},
            "roster_summary": {
                "type": "object", "additionalProperties": False,
                "required": ["by_category", "total"],
                "properties": {
                    "by_category": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "integer", "minimum": 0,
                        },
                        "propertyNames": {
                            "type": "string",
                            "pattern": "^[a-z][a-z0-9_]*$",
                        },
                    },
                    "total": {"type": "integer", "minimum": 0},
                },
            },
            "schema_name": {"const": "truth_frame"},
            "schema_version": {"const": "v1"},
            "sim_time_s": _float_leaf(),
            "sumo_active_incidents": sumo_active_incidents,
            "sumo_segment": (
                _segment(_sumo_roster._SUMO_PHASE)
                if sumo_enabled else {"type": "null"}
            ),
            "sumo_semantics": (
                _sumo_semantics() if sumo_enabled else {"type": "null"}
            ),
            "sumo_traffic_light_states": sumo_traffic_light_states,
            "tick": {"type": "integer", "minimum": 0},
            "tick_hz": {"const": 10},
            "uav_crosses_boundary": {"type": "boolean"},
            "uav_global_flow": {
                **(
                    _uav_frame_flow()
                    if uav_enabled else {"type": "null"}
                ),
            },
            "uav_segment": (
                _segment(_l6_roster._UAV_PHASE)
                if uav_enabled else {"type": "null"}
            ),
        },
    }


def formal_truth_frame_schema() -> dict:
    """Return the closed schema for one L6 ``truth_frames.jsonl`` row."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_truth_frame_l6.record.schema.json",
        "title": "AERO_WORLD formal_truth_frame L6 frame record schema",
        "oneOf": [
            _frame_root_schema(sumo_enabled=True, uav_enabled=True),
            _frame_root_schema(sumo_enabled=True, uav_enabled=False),
            _frame_root_schema(sumo_enabled=False, uav_enabled=True),
            _frame_root_schema(sumo_enabled=False, uav_enabled=False),
        ],
    }
