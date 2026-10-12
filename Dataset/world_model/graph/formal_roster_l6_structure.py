"""Closed whole-JSON grammar for L6 formal entity rosters.

``convert_to_render_ready.py`` writes ``{"entities": [...]}``; the capture
writer retains that JSON root while filtering entities.  The source reader
validates the unchanged root before projecting individual entities for the
field catalog.  Entity branches distinguish SUMO vehicles, global UAVs and
pads, scene-authored UAVs, facilities, pedestrians, corridors, and the L6-5
logical ground-control anchor.  Runtime-state subtrees use the executable
producer contract and engine source guard.

The schema constructors are pure and perform no data reads on import.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
    RUNTIME_STATE_FIELDS,
)
from Dataset.world_model.graph import formal_roster_l4_structure as _shared
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
    runtime_state_source_guard_rule,
)

__all__ = ["formal_roster_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    # Roster floats come from a json round-trip; the raw source type is
    # float (never int), so the leaf narrows the accepted raw type.
    return {"type": "number", "x-source-value-types": ["float"]}


def _int_leaf() -> dict:
    return {"type": "integer", "minimum": 0}


def _number3() -> dict:
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _waypoints() -> dict:
    # Producer pass-through of authored routes; the empty list is a
    # declared status (observed on facility rows).
    return {"type": "array", "items": _number3()}


def _segment(phase_name: dict) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "duration_s", "phase_name", "seed_index", "seed_label",
            "segment_end_s", "segment_start_s",
        ],
        "properties": {
            "duration_s": _float_leaf(),
            "phase_name": phase_name,
            "seed_index": {"type": "integer", "minimum": 0},
            "seed_label": {"type": "string", "pattern": "^seed[0-9]{2}$"},
            "segment_end_s": _float_leaf(),
            "segment_start_s": _float_leaf(),
        },
    }


_UAV_PHASE = {
    "enum": [
        "uav_demand_peak_hold",
        "uav_demand_ramp_up",
        "uav_demand_taper_with_persistent_patrol",
    ],
}


def _runtime_visibility() -> dict:
    # The source writer adds the first five fields; the capture writer may
    # add the two retention fields when it preserves a boundary entity.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "expanded_boundary_padding_m", "first_visible_tick",
            "last_visible_tick", "policy", "visible_tick_count",
        ],
        "properties": {
            "expanded_boundary_padding_m": _float_leaf(),
            "first_visible_tick": _int_leaf(),
            "last_visible_tick": _int_leaf(),
            "policy": {
                "const": "roi_polygon_expanded_60m_runtime_truth_crop_v1",
            },
            "visible_tick_count": _int_leaf(),
            "retained_outside_runtime_boundary": {"type": "boolean"},
            "retention_policy": _identifier(),
        },
    }


def _semantic_scope() -> dict:
    # entity_scope.validate_roster_facility_scope (called at :3260) +
    # facility_scope.semantic_scope_for_source; the L6 census shows
    # charging_station/communication_base_station/landing_pad.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["authority_id", "ontology_class_id", "scope_subtype",
                     "scope_type"],
        "properties": {
            "authority_id": {"const": "facility_scope_contract.v1"},
            "ontology_class_id": {
                "enum": [
                    "world:Barrier",
                    "world:ChargingStation",
                    "world:CommunicationStation",
                    "world:LandingPad",
                    "world:NoFlyZone",
                ],
            },
            "scope_subtype": {
                "enum": [
                    "barrier",
                    "charging_station",
                    "communication_base_station",
                    "landing_pad",
                    "no_fly_zone",
                ],
            },
            "scope_type": {"const": "facility"},
            "service_capacity": {"type": "integer", "minimum": 0},
        },
    }


def _semantic_event_contract() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["contract_scenario_id", "policy"],
        "properties": {
            "contract_scenario_id": _identifier(),
            "policy": {"const": "semantic_event_contract_v1"},
        },
    }


def _sensor_profile() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "camera_name", "capture_backend", "fixed_rotation_offset_deg",
            "height", "hfov_deg", "source", "width",
        ],
        "properties": {
            "camera_name": {"const": "bottom_center"},
            "capture_backend": {"const": "editor_hook"},
            "fixed_rotation_offset_deg": {
                "type": "object", "additionalProperties": False,
                "required": ["pitch_deg", "roll_deg", "yaw_deg"],
                "properties": {
                    "pitch_deg": _float_leaf(),
                    "roll_deg": _float_leaf(),
                    "yaw_deg": _float_leaf(),
                },
            },
            "height": {"type": "integer", "minimum": 0},
            "hfov_deg": _float_leaf(),
            "source": _identifier(),
            "width": {"type": "integer", "minimum": 0},
        },
    }


def _contract_inspect_uav() -> dict:
    # Scene-authored inspect contract; the route arrays carry real L6
    # waypoints in the L6-1_v1 census (non-empty), so they stay typed
    # waypoint arrays.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "corridor_policy", "corridor_role", "fixed_altitude_loop",
            "inspect_altitude_code", "inspect_altitude_m",
            "inspect_fov_coverage_required", "loop_route_enu_m",
            "min_path_length_m", "motion_policy",
            "observed_capture_boundary_id", "planned_route_enu_m", "policy",
            "repaired_path_length_m", "repaired_route_enu_m",
            "sensor_fov_deg", "sensor_fov_source", "sensor_profile",
        ],
        "properties": {
            "corridor_policy": {"const": "inspect_capture_corridor"},
            "corridor_role": {"const": "inspect_observer"},
            "fixed_altitude_loop": {"type": "boolean"},
            "inspect_altitude_code": _identifier(),
            "inspect_altitude_m": _float_leaf(),
            "inspect_fov_coverage_required": {"type": "boolean"},
            "loop_route_enu_m": _waypoints(),
            "min_path_length_m": _float_leaf(),
            "motion_policy": {
                "const": "fixed_altitude_loop_observes_capture_boundary",
            },
            "observed_capture_boundary_id": _identifier(),
            "planned_route_enu_m": _waypoints(),
            "policy": {"const": "semantic_event_contract_v1"},
            "repaired_path_length_m": _float_leaf(),
            "repaired_route_enu_m": _waypoints(),
            "sensor_fov_deg": _float_leaf(),
            "sensor_fov_source": {
                "const": "Config/LowAltitude/uav_sensor_profile.json#hfov_deg",
            },
            "sensor_profile": _sensor_profile(),
        },
    }


def _lifecycle() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "assigned_altitude_m", "auto_lifecycle_applied",
            "home_hover_enu_m", "home_pad_entity_id", "mission_start_enu_m",
            "requires_landing_or_terminal_resolution", "requires_takeoff",
            "waypoint_repair_applied",
        ],
        "properties": {
            "assigned_altitude_m": _float_leaf(),
            "auto_lifecycle_applied": {"type": "boolean"},
            "home_hover_enu_m": _number3(),
            "home_pad_entity_id": _identifier(),
            "mission_start_enu_m": _number3(),
            "requires_landing_or_terminal_resolution": {"type": "boolean"},
            "requires_takeoff": {"type": "boolean"},
            "waypoint_repair_applied": {"type": "boolean"},
        },
    }


def _observer_lifecycle() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "capture_boundary_id", "maximum_terminal_boundary_distance_m",
            "presence", "terminal_behavior",
        ],
        "properties": {
            "capture_boundary_id": _identifier(),
            "maximum_terminal_boundary_distance_m": _float_leaf(),
            "presence": {"const": "episode_full_duration"},
            "terminal_behavior": {
                "const": "airborne_hold_inside_runtime_boundary",
            },
        },
    }


def _uav_corridor() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["altitude_layers_m", "assigned_altitude_m",
                     "used_lateral_bypass"],
        "properties": {
            "altitude_layers_m": {"type": "array", "items": _float_leaf()},
            "assigned_altitude_m": _float_leaf(),
            "used_lateral_bypass": {"type": "boolean"},
        },
    }


def _pedestrian_ground_flow_contract() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "actor_kind", "loop_policy", "min_visible_motion_ratio",
            "min_xy_span_m", "planned_path_length_m", "policy",
            "required", "route_duration_ticks", "route_source", "speed_mps",
        ],
        "properties": {
            "actor_kind": {"const": "pedestrian"},
            "loop_policy": {"const": "single_pass_route_waypoints"},
            "min_visible_motion_ratio": _float_leaf(),
            "min_xy_span_m": _float_leaf(),
            "planned_path_length_m": _float_leaf(),
            "policy": {"const": "continuous_capture_ground_flow_v1"},
            "required": {"type": "boolean"},
            "route_duration_ticks": _int_leaf(),
            "route_source": {"const": "traffic_bundle_sidewalk"},
            "speed_mps": _float_leaf(),
        },
    }


def _uav_global_flow_payload() -> dict:
    # convert_to_render_ready.py :3379-3393 (writer) over the flow
    # planner's mission table; the enums are the observed planner
    # membership (logistics_delivery/pad_patrol in L6-1_v1, the relay and
    # inspect families observed on other profiles).
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "altitude_layer_m", "corridor_family", "corridor_id",
            "ground_reference_z_m", "mission_type", "origin_pad_id", "policy",
            "sample_period_s", "semantic_role", "target_cell_id",
            "target_pad_id", "task_id", "uav_id",
        ],
        "properties": {
            "altitude_layer_m": _float_leaf(),
            "corridor_family": {
                "enum": [
                    "high_altitude_sector_relay_loop",
                    "infrastructure_inspection_corridor",
                    "pad_patrol_local_loop",
                    "pad_to_pad_logistics_corridor",
                    "priority_incident_response_corridor",
                    "road_corridor_inspection_loop",
                ],
            },
            "corridor_id": _identifier(),
            "ground_reference_z_m": _float_leaf(),
            "mission_type": {
                "enum": [
                    "edge_compute_relay",
                    "incident_response_inspection",
                    "infrastructure_inspection",
                    "intersection_inspect",
                    "logistics_delivery",
                    "pad_patrol",
                ],
            },
            # The planner writes an empty string when a flow has no pad
            # endpoint (no-pad relay/inspect flows).
            "origin_pad_id": {"type": "string"},
            "policy": {"const": "donghu_global_uav_flow_truth_replay_v1"},
            "sample_period_s": _float_leaf(),
            "semantic_role": {
                "enum": [
                    "communication_relay_uav",
                    "corridor_inspect_uav",
                    "delivery_uav",
                    "incident_response_uav",
                    "pad_inspect_uav",
                    "temporary_infra_inspect_uav",
                ],
            },
            "target_cell_id": _identifier(),
            "target_pad_id": {"type": "string"},
            "task_id": _identifier(),
            "uav_id": _identifier(),
        },
    }


def _uav_motion_contract() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "actor_kind", "altitude_layer_m", "looping", "policy",
            "route_length_m", "route_source", "segment", "source",
            "speed_mps",
        ],
        "properties": {
            "actor_kind": {"const": "uav"},
            "altitude_layer_m": _float_leaf(),
            "looping": {"type": "boolean"},
            "policy": {"const": "uav_global_flow_truth_replay_v1"},
            "route_length_m": _float_leaf(),
            "route_source": {"const": "donghu_global_uav_flow"},
            "segment": _segment(_UAV_PHASE),
            "source": {"const": "uav_global_flow"},
            "speed_mps": _float_leaf(),
        },
    }


def _pad_motion_contract() -> dict:
    # convert_to_render_ready.py :3254-3258.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["actor_kind", "policy", "source"],
        "properties": {
            "actor_kind": {"const": "facility"},
            "policy": {"const": "static_facility_truth_frame_v1"},
            "source": {"const": "uav_global_flow"},
        },
    }


def _uav_global_pad() -> dict:
    # convert_to_render_ready.py :3244-3253 (writer) over the pad network
    # table; both phase weight vectors are three-wide (observed).
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "grid_cell_id", "lane_edge_id", "lane_s_m", "pad_id", "policy",
            "phase_destination_weights", "phase_origin_weights", "role",
        ],
        "properties": {
            "grid_cell_id": _identifier(),
            "lane_edge_id": _identifier(),
            "lane_s_m": _float_leaf(),
            "pad_id": _identifier(),
            "policy": {"const": "donghu_global_uav_pad_network_v1"},
            "phase_destination_weights": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": _float_leaf(),
            },
            "phase_origin_weights": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": _float_leaf(),
            },
            "role": {
                "enum": ["hub_charging_logistics",
                         "satellite_charging_logistics"],
            },
        },
    }


def _placement() -> dict:
    # The converter writes a repaired or an unrepaired facility placement.
    return {"oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": [
                "anchor_edge_id", "anchor_lane_s_m", "offset_from_curb_m",
                "placement_semantics", "resolved_lateral_from_center_m",
                "road_clearance_m", "scene_occupancy_authority",
            ],
            "properties": {
                "anchor_edge_id": _identifier(),
                "anchor_lane_s_m": _float_leaf(),
                "offset_from_curb_m": _float_leaf(),
                "placement_semantics": {"const": "service_facility_repair"},
                "resolved_lateral_from_center_m": _float_leaf(),
                "road_clearance_m": _float_leaf(),
                "scene_occupancy_authority": {
                    "const": "low_altitude_scene_occupancy_authority_v1",
                },
            },
        },
        {
            "type": "object", "additionalProperties": False,
            "required": [
                "blocking", "collision_policy", "placement_semantics",
                "road_clearance_m", "scene_occupancy_authority",
            ],
            "properties": {
                "blocking": {"const": False},
                "collision_policy": {
                    "const": "no_collision_visual_context_only",
                },
                "placement_semantics": {
                    "const": "service_facility_nonblocking_unrepaired",
                },
                "road_clearance_m": _float_leaf(),
                "scene_occupancy_authority": {
                    "const": "low_altitude_scene_occupancy_authority_v1",
                },
            },
        },
    ]}


def _scene_occupancy_repair() -> dict:
    # Successful repair has anchor fields; failed repair has status/reason.
    return {"oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": [
                "anchor_edge_id", "anchor_lane_s_m", "asset_id",
                "entity_id", "from_position_enu_m", "offset_from_curb_m",
                "policy", "required_clearance_m",
                "resolved_lateral_from_center_m", "road_clearance_m",
                "to_position_enu_m",
            ],
            "properties": {
                "anchor_edge_id": _identifier(),
                "anchor_lane_s_m": _float_leaf(),
                "asset_id": _identifier(),
                "entity_id": _identifier(),
                "fallback": {"const": "open_space_radial_search"},
                "from_position_enu_m": _number3(),
                "offset_from_curb_m": _float_leaf(),
                "policy": {
                    "const": "render_ready_service_facility_reanchor_v1",
                },
                "required_clearance_m": _float_leaf(),
                "resolved_lateral_from_center_m": _float_leaf(),
                "road_clearance_m": _float_leaf(),
                "to_position_enu_m": _number3(),
            },
        },
        {
            "type": "object", "additionalProperties": False,
            "required": [
                "asset_id", "entity_id", "from_position_enu_m", "policy",
                "reason", "required_clearance_m", "road_clearance_m",
                "status", "to_position_enu_m",
            ],
            "properties": {
                "asset_id": _identifier(),
                "entity_id": _identifier(),
                "from_position_enu_m": _number3(),
                "policy": {
                    "const": "render_ready_service_facility_reanchor_v1",
                },
                "reason": {"type": "string"},
                "required_clearance_m": _float_leaf(),
                "road_clearance_m": _float_leaf(),
                "status": {
                    "const": "unrepaired_nonblocking_service_facility",
                },
                "to_position_enu_m": _number3(),
            },
        },
    ]}


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
            {"type": "array", "minItems": 1,
             "items": runtime_enum_string_rule(members)},
        ]}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return runtime_enum_string_rule(
            tuple(sorted(RUNTIME_STATE_ENUM_FIELDS[(family, field)])))
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
        f"{family}.{field}")


def _runtime_state_family(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family]
                    | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        **runtime_state_source_guard_rule("engine_whole"),
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
    }


def _runtime_state_properties() -> dict:
    # Source-side preservation and the SUMO source merge accept each family
    # after validate_runtime_state_patch applies the executable contract.
    return {family: _runtime_state_family(family)
            for family in RUNTIME_STATE_FIELDS}


def _core_properties() -> dict:
    return {
        # scene-shared base members written by every roster builder
        # (:5031-5046 source-side, :4319-4334 SUMO, :3220-3241 pads,
        # :3354-3375 flows); logical_asset_id is identifier-typed because
        # the producer copies authored asset ids.
        "asset_id": _identifier(),
        "entity_id": _identifier(),
        "entity_category": _identifier(),
        "entity_kind": _identifier(),
        "entity_type": _identifier(),
        "initial_position_enu_m": _number3(),
        "initial_yaw_deg": _float_leaf(),
        "label_class": _identifier(),
        "logical_asset_id": _identifier(),
        "proxy_template_id": {"type": "string"},
        "roi_id": _identifier(),
        "site_id": _identifier(),
        # Authored tag list; the producer writes 2-4 entries per branch,
        # length not pinned.
        "tags": {"type": "array", "items": _identifier()},
    }


def _core_required() -> list:
    return [
        "asset_id", "entity_id", "entity_category", "entity_kind",
        "entity_type", "initial_position_enu_m", "initial_yaw_deg",
        "label_class", "logical_asset_id", "proxy_template_id", "roi_id",
        "site_id", "tags",
    ]


def _branch(properties: dict, required: list) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": _core_required() + required,
        "properties": {**_core_properties(), **properties},
    }


def _global_flow_uav_row() -> dict:
    return _branch(
        {
            "background_role": {"const": "donghu_global_uav_flow"},
            "ground_reference_source": {
                "enum": [
                    "uav_task_plan_global_ground_reference_v1",
                    "uav_task_plan_origin_pad_v1",
                ],
            },
            "ground_reference_z_m": _float_leaf(),
            "mode": {"const": "scene_sync"},
            "motion_contract": _uav_motion_contract(),
            "planned_route_waypoints_enu_m": _waypoints(),
            "semantic_role": _identifier(),
            "source": {"const": "uav_global_flow"},
            "task_id": _identifier(),
            "uav_global_flow": _uav_global_flow_payload(),
            "uav_id": _identifier(),
            "uav_segment": _segment(_UAV_PHASE),
        },
        [
            "background_role", "ground_reference_source",
            "ground_reference_z_m", "mode", "motion_contract",
            "planned_route_waypoints_enu_m", "semantic_role", "source",
            "task_id", "uav_global_flow", "uav_id", "uav_segment",
        ],
    )


def _global_pad_row() -> dict:
    return _branch(
        {
            "background_role": {"const": "global_uav_pad"},
            "mode": {"const": "scene_sync"},
            "motion_contract": _pad_motion_contract(),
            "semantic_scope": _semantic_scope(),
            "source": {"const": "uav_global_flow"},
            "uav_global_pad": _uav_global_pad(),
            "uav_pad_id": _identifier(),
        },
        [
            "background_role", "mode", "motion_contract", "semantic_scope",
            "source", "uav_pad_id", "uav_global_pad",
        ],
    )


def _scoped_uav_properties() -> dict:
    return {
        **_runtime_state_properties(),
        "activation_tick": _int_leaf(),
        "assigned_altitude_m": _float_leaf(),
        "category": {"const": "uav"},
        "communication_state": _runtime_state_family("communication_state"),
        "control_state": _runtime_state_family("control_state"),
        "full_episode_presence": {"type": "boolean"},
        "ground_reference_source": {
            "enum": [
                "home_hover_pose_origin_v1",
                "map_ground_reference_v1",
                "uav_task_plan_global_ground_reference_v1",
                "uav_task_plan_origin_pad_v1",
            ],
        },
        "ground_reference_z_m": _float_leaf(),
        "mode": {"const": "scene_sync"},
        "path_deviation_contract": {
            "type": "object", "additionalProperties": False,
            "required": ["minimum_route_distance_m", "predicate_id"],
            "properties": {
                "minimum_route_distance_m": _float_leaf(),
                "predicate_id": {
                    "const": "positioning.aircraft_deviates_from_planned_route",
                },
            },
        },
        "planned_route_waypoints_enu_m": _waypoints(),
        "role": _identifier(),
        "route_waypoints_enu_m": _waypoints(),
        "runtime_visibility": _runtime_visibility(),
        "semantic_role": _identifier(),
        "state_sequence": {"type": "array", "items": _identifier()},
        "task_id": _identifier(),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": {
            "enum": ["inspect_observer", "mission_uav", "observer"],
        },
    }


def _scoped_uav_required() -> list:
    return [
        "activation_tick", "assigned_altitude_m", "category",
        "ground_reference_source", "ground_reference_z_m", "mode",
        "planned_route_waypoints_enu_m", "role", "route_waypoints_enu_m",
        "runtime_visibility", "semantic_role", "state_sequence", "task_id",
        "uav_corridor", "uav_corridor_role",
    ]


def _inspect_uav_row() -> dict:
    return _branch(
        {
            **_scoped_uav_properties(),
            "contract_inspect_uav": _contract_inspect_uav(),
            "full_episode_presence": {"const": True},
            "inspect_altitude_code": {
                "type": "string", "pattern": "^I[0-9]+$",
            },
            "inspect_altitude_m": _float_leaf(),
            "inspect_fov_coverage_required": {"type": "boolean"},
            "min_path_length_m": _float_leaf(),
        },
        _scoped_uav_required() + [
            "contract_inspect_uav", "full_episode_presence",
            "inspect_altitude_code", "inspect_altitude_m",
            "inspect_fov_coverage_required", "min_path_length_m",
        ],
    )


def _mission_uav_row() -> dict:
    return _branch(
        {
            **_scoped_uav_properties(),
            "lifecycle": _lifecycle(),
        },
        _scoped_uav_required() + ["lifecycle"],
    )


def _observer_uav_row() -> dict:
    return _branch(
        {
            **_scoped_uav_properties(),
            "observer_lifecycle": _observer_lifecycle(),
        },
        _scoped_uav_required() + ["observer_lifecycle"],
    )


def _facility_properties() -> dict:
    return {
        **_runtime_state_properties(),
        "activation_tick": _int_leaf(),
        "category": {"const": "facility"},
        "mode": {"enum": ["metadata_only", "scene_sync"]},
        "role": {"const": "semantic_facility"},
        "route_waypoints_enu_m": _waypoints(),
        "runtime_visibility": _runtime_visibility(),
        "semantic_role": _identifier(),
        "state_sequence": {"type": "array", "items": _identifier()},
        "task_id": _identifier(),
    }


_FACILITY_REQUIRED = [
    "activation_tick", "category", "mode", "role", "route_waypoints_enu_m",
    "runtime_visibility", "semantic_role", "state_sequence", "task_id",
]


def _semantic_facility_row() -> dict:
    # Landing pad variant B (pad_home_*) and other scene-authored
    # facilities without contract keys.
    return _branch(
        {
            **_facility_properties(),
            "semantic_scope": _semantic_scope(),
        },
        _FACILITY_REQUIRED + ["semantic_scope"],
    )


def _contract_facility_row() -> dict:
    return _branch(
        {
            **_facility_properties(),
            "contract_facility": _semantic_event_contract(),
            "contract_scenario_id": _identifier(),
            "semantic_scope": _semantic_scope(),
        },
        _FACILITY_REQUIRED + ["contract_facility", "contract_scenario_id",
                              "semantic_scope"],
    )


def _repaired_facility_row() -> dict:
    # The repaired landing pads carry no contract keys; the repaired radio
    # towers do — both stay optional on this branch.
    return _branch(
        {
            **_facility_properties(),
            "contract_facility": _semantic_event_contract(),
            "contract_scenario_id": _identifier(),
            "placement": _placement(),
            "scene_occupancy_repair": _scene_occupancy_repair(),
            "semantic_scope": _semantic_scope(),
        },
        _FACILITY_REQUIRED + ["placement", "scene_occupancy_repair",
                              "semantic_scope"],
    )


def _pedestrian_row() -> dict:
    return _branch(
        {
            **_runtime_state_properties(),
            "activation_tick": _int_leaf(),
            "background_pedestrian": {
                "type": "object", "additionalProperties": False,
                "required": ["contract_scenario_id", "policy",
                             "semantic_role"],
                "properties": {
                    "contract_scenario_id": _identifier(),
                    "policy": {"const": "semantic_event_contract_v1"},
                    # Free string (spaces observed, e.g. "operator-area
                    # context").
                    "semantic_role": {"type": "string"},
                },
            },
            "background_role": {"const": "semantic_context"},
            "category": {"const": "pedestrian"},
            "contract_scenario_id": _identifier(),
            "ground_flow_contract": _pedestrian_ground_flow_contract(),
            "mode": {"const": "pedestrian_managed"},
            "role": {"const": "semantic_background_pedestrian"},
            "route_waypoints_enu_m": _waypoints(),
            "runtime_visibility": _runtime_visibility(),
            # Free string (spaces observed).
            "semantic_role": {"type": "string"},
            "state_sequence": {"type": "array", "items": _identifier()},
            "task_id": _identifier(),
        },
        [
            "activation_tick", "background_pedestrian", "background_role",
            "category", "contract_scenario_id", "ground_flow_contract",
            "mode", "role", "route_waypoints_enu_m", "runtime_visibility",
            "semantic_role", "state_sequence", "task_id",
        ],
    )


def _corridor_row() -> dict:
    return _branch(
        {
            **_runtime_state_properties(),
            "activation_tick": _int_leaf(),
            "category": {"const": "airspace_corridor"},
            "mode": {"const": "metadata_only"},
            "runtime_visibility": _runtime_visibility(),
        },
        ["activation_tick", "category", "mode", "runtime_visibility"],
    )


def _gcs_anchor_row() -> dict:
    # L6-5 scene authors a logical ground-station anchor with security
    # state; profile_for_entity keeps its authored ground_station kind.
    return _branch(
        {
            **_runtime_state_properties(),
            "activation_tick": _int_leaf(),
            "asset_id": {"const": "semantic.asset_anchor"},
            "category": {"const": "facility"},
            "entity_category": {"const": "facility"},
            "entity_kind": {"const": "ground_station.ground_control_station"},
            "entity_type": {"const": "facility.other"},
            "label_class": {"const": "other"},
            "logical_asset_id": {"const": "semantic.asset_anchor"},
            "mode": {"const": "logical_contract"},
            "proxy_template_id": {"const": ""},
            "role": {"const": "semantic_context"},
            "route_waypoints_enu_m": _waypoints(),
            "runtime_visibility": _runtime_visibility(),
            "semantic_role": _identifier(),
            "semantic_scope": {
                "type": "object", "additionalProperties": False,
                "required": ["authority_id", "ontology_class_id",
                             "scope_subtype", "scope_type"],
                "properties": {
                    "authority_id": {"const": "facility_scope_contract.v1"},
                    "ontology_class_id": {"const": "world:GroundControlStation"},
                    "scope_subtype": {"const": "ground_control_station"},
                    "scope_type": {"const": "facility"},
                },
            },
            "state_sequence": {"type": "array", "items": _identifier()},
            "task_id": _identifier(),
        },
        ["activation_tick", "category", "mode", "role",
         "route_waypoints_enu_m", "runtime_visibility", "security_state",
         "semantic_role", "semantic_scope", "state_sequence", "task_id"],
    )


def _entity_schema() -> dict:
    sumo = _shared._sumo_vehicle_row()
    sumo["properties"].update(_runtime_state_properties())
    return {
        "oneOf": [
            sumo,
            _global_flow_uav_row(),
            _global_pad_row(),
            _inspect_uav_row(),
            _mission_uav_row(),
            _observer_uav_row(),
            _semantic_facility_row(),
            _contract_facility_row(),
            _repaired_facility_row(),
            _pedestrian_row(),
            _corridor_row(),
            _gcs_anchor_row(),
        ],
    }


def formal_roster_schema() -> dict:
    """Return the closed whole-JSON L6 ``formal_roster`` schema."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_roster_l6.record.schema.json",
        "title": "AERO_WORLD formal_roster L6 JSON root record schema",
        "type": "object", "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {"type": "array", "minItems": 1,
                         "items": _entity_schema()},
        },
    }
