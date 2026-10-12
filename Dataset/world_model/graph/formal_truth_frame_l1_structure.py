"""Closed record structure for one formal truth frame.

Source family ``formal_truth_frame``; profiles L1-1_v1, L1-1_v2, L1-2_v1,
L1-3_v1, L1-3_v2, L1-4_v1, L1-4_v2.  Producer chain:
``Dataset/tools/convert_to_render_ready.py`` assembles each frame in the
per-tick loop (:5069-5409) and appends it to ``truth_frames``
(:5367-5409); ``Dataset/tools/filter_render_ready_truth_for_capture.py``
rewrites ``entities``/``roster_summary``/``entity_motion_state``/
``sumo_semantics.active_selected_vehicle_count``/
``uav_global_flow.active_selected_uav_count`` in place
(:437-456 ``update_frame_summaries``) and lists ``truth_frames.jsonl`` in
``REGENERATED_FILES``, so the frames at
``aw_data/render_ready_episodes_capture_filtered`` are the converter's
frame shape with the capture-filter's boundary-cropped entity lists.

Record identity: ``structure_sources.py`` does NOT list
``formal_truth_frame`` in ``_ENTITY_FILES`` (:34), so
``iter_source_records`` yields every JSONL row itself — one frame dict is
the record root.  ``field_paths.py`` declares the entity-keyed maps
``entity_motion_state`` and ``sumo_traffic_light_states`` (keys are
entity / traffic-light-controller identity, values are per-instance
data) and the ``entities`` entity list; this grammar therefore closes the
whole frame and binds the dynamic keys with anchored ``propertyNames``
patterns instead of enumerating instances.

Every frame writes the same 26 root keys (:5367-5409), so the root is one
closed object.  Entity rows use 15 closed branches: three SUMO
lifecycle/retention forms, global UAV and pad forms, and ten L1 source-row
forms.  A bounded scan of six seed00 files found 5,406 frame roots and
218,079 entity rows across 13 branches.  The two required-lifecycle SUMO
branches and the producer's nullable SUMO/UAV payload branches had no
instance in that scope; they remain explicit producer-authorized forms.

Producer evidence per branch (beyond the lines above):
- Shared truth pose: ``truth_pose`` (:924-938) — 7 fixed keys, the
  rotation zeros are written constants (:933-934).
- ``render_presence``: the fixed six-key visibility/submission block built
  per frame (:5126).
- ``runtime_visibility``: ``runtime_visibility_payload`` (:2939-2950).
  Retained SUMO rows form a separate closed branch that requires
  ``retained_outside_runtime_boundary`` and ``retention_policy``
  (:5253-5261); other branches do not declare those keys.
- ``uav_visibility``: ``uav_camera_capture_visibility_payload``
  (:2953-2973) over ``UAV_CAMERA_CAPTURE_ROI_POLICY`` (:161-163).
- ``annotations``: ``build_annotations`` (:1830-1863) + pedestrian
  ``activity_annotations`` (``Dataset/tools/pedestrian_activity_catalog.py``
  :105-118); 14 canonical activities (:41-74).  UAV annotations use
  free-form ``activity_type``/``animation_hint`` (:1818-1824).
- ``state``: source row state (:5150-5151), the sumo activity_type
  (:4403), or the UAV task state in the census.
- ``semantic_scope``: preserved facility scope (roster's
  ``_semantic_scope`` shape; ``validate_roster_facility_scope`` :3195).
- Road-authority keys inside ``sumo_vehicle``: the 16 fields from
  ``Dataset/tools/sumo_ground_flow/road_signal_context.py``
  (``static_sumo_vehicle_fields`` :189-254 plus
  ``enrich_sumo_vehicle`` :77-254 — populated-or-null triplets exactly
  when the lane has a signal connection; ``controlling_signal_state``
  is the live signal state or the const ``"none"`` (:30, :223)); the
  interpolation fields and ``truth_*`` geometry come from
  ``_interpolate_vehicle`` (``truth_integration.py`` :626-683) and
  ``projected_sumo_vehicle_position`` (:3451-3457).
- ``sumo_vehicle.semantic_metadata``: the 2-key identity block or the
  27-key background block (roster census; ``explicit_vehicle_plan``).
- ``ground_flow_contract``: the SUMO writer's five-key block
  (``actor_kind``, ``policy``, ``route_source``, ``sample_period_s``,
  ``segment``) or the distinct pedestrian block.
- ``uav_global_flow`` entity block: roster 15-key payload merged with
  ``active_event_ids`` + ``source_prev/next_time_s`` + ``source_alpha``
  (:3405-3411); ``active_event_ids`` is empty in the corpus
  (尚无实例验证 for nonempty members).
- ``uav_global_pad``: 8-key pad block (:3179-3188); the weight arrays
  hold 2-3 floats (census shapes [[2],[3]]).
- Frame ``sumo_segment``/``uav_segment``: the segment payload or None
  (:5388-5400); entity ``sumo_segment``/``uav_segment`` are always the
  payload dicts (:4415, :3413).
- ``sumo_active_incidents``: ``_compact_incident``
  (``truth_integration.py`` :574-599) — 13 required keys plus
  ``traffic_light_id`` exactly on ``traffic_light_all_red_fault``
  (:594-598); the same rows are embedded in
  ``sumo_semantics.active_incidents[]`` (:4508, :4523).
- ``sumo_traffic_light_states``: ``_traffic_light_states``
  (``truth_integration.py`` :560-571) — keys are SUMO controller ids
  (``n_00008`` form, instance-bound, not enumerated); the
  capture-filter rebuilds this map with cropped controller ids
  (:422-424), so its keys differ from ``entity_motion_state`` keys.
- ``uav_global_flow`` frame block: ``uav_frame_truth_payload``
  (:3425-3448) + ``roi_capture_eligible_uav_count``/
  ``camera_capture_roi_policy`` (:5352-5357).
- ``entity_motion_state`` / ``roster_summary``:
  ``truth_boundary_summary`` (:3105-3130; UAV categories only) and the
  per-frame counter (:5359, :5403-5406); the capture-filter rebuilds
  both from the cropped entity list (:437-456), including the empty
  frame with ``total: 0`` (尚无实例验证 for a cropped empty frame).
- ``state_revision`` is ``tick + 1`` (:5128, :3400, :4404, :3225) —
  a frame-relative rule, not a schema pin.

Runtime state: L1 rows in the bounded scope consume ``control_state`` and
``sensor_state`` only.  Each object is closed to
``Dataset/tools/runtime_state_contract.py`` and uses the shared
``structure_fragments`` rules for Python ``strip().lower()`` enum values
and nonblank metadata.  The ``engine_whole`` source guard is attached to
the consumed family object itself.

Publication exclusion: frames carry no digest/integrity/hash field
(census has none; ``is_publication_excluded_path`` has no
formal_truth_frame rule), so this module declares no excluded path.

Boundary: this module exports the pure schema constructor only.  It is
not registered in ``structure_closure.py`` (``formal_truth_frame`` is
absent from ``CLOSED_RECORD_FAMILIES``) and
``validate_source_record`` dispatches it as "validated by source
projection" until registration; registration/dispatch wiring belongs to
the integration stage.  The module scans no data at import and
compiles nothing at import.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
)
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
    runtime_state_source_guard_rule,
)

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
        "items": {"type": "number"},
    }


def _float3_array() -> dict:
    return {"type": "array", "items": _float3()}


def _nullable_float() -> dict:
    return {"oneOf": [_float_leaf(), {"type": "null"}]}


def _segment() -> dict:
    # SumoSegment.as_dict (truth_integration.py :113-121); the UAV segment
    # payload reuses the same six keys with the UAV phase names.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "duration_s", "phase_name", "seed_index", "seed_label",
            "segment_end_s", "segment_start_s",
        ],
        "properties": {
            "duration_s": _float_leaf(),
            "phase_name": _identifier(),
            "seed_index": {"type": "integer", "minimum": 0},
            "seed_label": {"type": "string", "pattern": "^seed[0-9]{2}$"},
            "segment_end_s": _float_leaf(),
            "segment_start_s": _float_leaf(),
        },
    }


def _vehicle_type() -> dict:
    # explicit_vehicle_plan writes the bare profile type or
    # "<type>@<sumo vehicle id>"; the joined half is a SUMO id.
    return {
        "type": "string",
        "pattern": "^(aero_passenger|aero_emergency)(@[^@\\s]+)?$",
    }


def _runtime_state_string_rule(members: tuple[str, ...]) -> dict:
    """Describe the raw strings accepted by ``value.strip().lower()``."""
    return runtime_enum_string_rule(members)


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
            _runtime_state_string_rule(members),
            {"type": "array", "minItems": 1,
             "items": _runtime_state_string_rule(members)},
        ]}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return _runtime_state_string_rule(
            tuple(sorted(RUNTIME_STATE_ENUM_FIELDS[(family, field)])))
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family]
                    | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _truth_pose() -> dict:
    # truth_pose (:924-938); the rotation zeros are written constants.
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
    # uav_camera_capture_visibility_payload (:2953-2973).
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


def _network_facet() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["latency_ms", "packet_loss", "status"],
        "properties": {
            "latency_ms": {"const": 0.0},
            "packet_loss": {"const": 0.0},
            "status": {"const": "nominal"},
        },
    }


def _entity_state_facets_activity() -> dict:
    # The 4 fixed activity keys (:1850-1855, :110-117); posture/social
    # stay free strings because the catalog values are not an L1
    # grammar-relevant enum beyond the catalog itself.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "activity_type", "animation_hint", "posture", "social_state",
        ],
        "properties": {
            "activity_type": {"type": "string", "minLength": 1},
            "animation_hint": {"type": "string", "minLength": 1},
            "posture": {"type": "string", "minLength": 1},
            "social_state": {"type": "string", "minLength": 1},
        },
    }


def _annotations_activity() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["activity", "network"],
        "properties": {
            "activity": _entity_state_facets_activity(),
            "network": _network_facet(),
        },
    }


def _annotations() -> dict:
    # build_annotations (:1830-1863) + activity_annotations
    # (pedestrian_activity_catalog.py :105-118).  activity_type stays
    # free-form: the UAV arm passes the task state through (:1818-1824)
    # and the pedestrian arm is catalog-normalized upstream.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["activity_type", "speed_mps", "state_facets"],
        "properties": {
            "activity_type": {"type": "string", "minLength": 1},
            "speed_mps": _float_leaf(),
            "state_facets": _annotations_activity(),
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
            "loop_route_enu_m": _float3_array(),
            "min_path_length_m": _float_leaf(),
            "motion_policy": {
                "const": "fixed_altitude_loop_observes_capture_boundary",
            },
            "observed_capture_boundary_id": _identifier(),
            "planned_route_enu_m": _float3_array(),
            "policy": {"const": "semantic_event_contract_v1"},
            "repaired_path_length_m": _float_leaf(),
            "repaired_route_enu_m": _float3_array(),
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
            "home_hover_enu_m": _float3(),
            "home_pad_entity_id": _identifier(),
            "mission_start_enu_m": _float3(),
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


def _scene_occupancy_repair() -> dict:
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
                "from_position_enu_m": _float3(),
                "offset_from_curb_m": _float_leaf(),
                "policy": {
                    "const": "render_ready_service_facility_reanchor_v1",
                },
                "required_clearance_m": _float_leaf(),
                "resolved_lateral_from_center_m": _float_leaf(),
                "road_clearance_m": _float_leaf(),
                "to_position_enu_m": _float3(),
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
                "from_position_enu_m": _float3(),
                "policy": {
                    "const": "render_ready_service_facility_reanchor_v1",
                },
                "reason": {"type": "string"},
                "required_clearance_m": _float_leaf(),
                "road_clearance_m": _float_leaf(),
                "status": {
                    "const": "unrepaired_nonblocking_service_facility",
                },
                "to_position_enu_m": _float3(),
            },
        },
    ]}


def _semantic_metadata() -> dict:
    # Two producer forms: the 2-key identity block and the 27-key
    # background block.  The semantic_* slots are left unwritten at
    # source level; None/""/[] are declared statuses.  Nonempty member
    # instances: 尚无实例验证.
    return {"oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": ["logical_asset_authority", "logical_asset_id"],
            "properties": {
                "logical_asset_authority": {
                    "const": "explicit_vehicle_plan_type",
                },
                "logical_asset_id": {
                    "enum": ["vehicle.emergency.suv.v1",
                             "vehicle.ground.boxcar.v1"],
                },
            },
        },
        {
            "type": "object", "additionalProperties": False,
            "required": [
                "direction_role", "expected_core_entry_tick",
                "expected_core_exit_tick", "explicit_vehicle_plan",
                "logical_asset_authority", "logical_asset_id",
                "protected_window_role", "release_tick", "required_edges",
                "role", "route_source", "seed_profile",
                "semantic_activation_depart_edge",
                "semantic_activation_depart_pos_m",
                "semantic_activation_policy", "semantic_add_tick",
                "semantic_bootstrap_depart_pos_m",
                "semantic_control_window_ticks", "semantic_lifecycle",
                "semantic_pre_activation_ticks", "source_entity_id",
                "spawn_phase", "traffic_coverage_scope", "traffic_role",
                "traffic_slot_index", "vehicle_id", "vehicle_source_policy",
            ],
            "properties": {
                "direction_role": {"enum": ["cross", "inbound", "outbound"]},
                "expected_core_entry_tick": {"type": "integer"},
                "expected_core_exit_tick": {"type": "integer"},
                "explicit_vehicle_plan": {"type": "boolean"},
                "logical_asset_authority": {
                    "const": "explicit_vehicle_plan_type",
                },
                "logical_asset_id": {
                    "enum": ["vehicle.emergency.suv.v1",
                             "vehicle.ground.boxcar.v1"],
                },
                "protected_window_role": {
                    "const": "background_forbidden_from_protected_edges",
                },
                "release_tick": {"type": "integer"},
                "required_edges": {"type": "array",
                                   "items": {"type": "string"}},
                "role": {"const": "background"},
                "route_source": {"const": "sumo_findRoute"},
                "seed_profile": {
                    "type": "object", "additionalProperties": False,
                    "required": ["direction_bias", "profile_id",
                                 "seed_index", "seed_label"],
                    "properties": {
                        "direction_bias": _identifier(),
                        "profile_id": _identifier(),
                        "seed_index": {"type": "integer", "minimum": 0},
                        "seed_label": {
                            "type": "string",
                            "pattern": "^seed[0-9]{2}$",
                        },
                    },
                },
                "semantic_activation_depart_edge": {"type": "string"},
                "semantic_activation_depart_pos_m": {"type": "null"},
                "semantic_activation_policy": {"type": "string"},
                "semantic_add_tick": {"type": "null"},
                "semantic_bootstrap_depart_pos_m": {"type": "null"},
                "semantic_control_window_ticks": {
                    "type": "array", "items": False,
                },
                "semantic_lifecycle": {"type": "null"},
                "semantic_pre_activation_ticks": {"type": "null"},
                "source_entity_id": {"type": "string"},
                "spawn_phase": {"const": "warmup_or_capture_release"},
                "traffic_coverage_scope": {
                    "enum": [
                        "accepted_background_replenishment_context_flow",
                        "extended_context_flow_protected_window",
                        "roi_background_route_crossing_flow",
                        "source_background_context_flow",
                    ],
                },
                "traffic_role": {
                    "const": "deterministic_background_vehicle",
                },
                "traffic_slot_index": {
                    "oneOf": [{"type": "integer"}, {"type": "null"}],
                },
                "vehicle_id": _identifier(),
                "vehicle_source_policy": {
                    "const": "explicit_episode_traffic_schedule_only_v2",
                },
            },
        },
    ]}


def _sumo_vehicle(*, required_lifecycle: bool) -> dict:
    schema = {
        "type": "object", "additionalProperties": False,
        "required": [
            "accel_mps2", "active_semantic_event", "allowed_speed_mps",
            "canonical_logical_asset_id", "center_lane_position_m",
            "control_role",
            "controlling_signal_id", "controlling_signal_ontology_class_id",
            "controlling_signal_state", "crossed_stop_line", "dimensions_m",
            "following_distance_rule_id",
            "following_distance_rule_ontology_class_id",
            "following_distance_rule_source_ref", "lane_id",
            "lane_instance_source_ref", "lane_ontology_class_id",
            "lane_position_m", "logical_asset_authority",
            "right_of_way_id", "right_of_way_ontology_class_id",
            "right_of_way_source_ref", "route_id", "semantic_episode_id",
            "semantic_metadata", "semantic_vehicle",
            "semantic_vehicle_state", "source_entity_id",
            "source_next_time_s", "source_prev_time_s", "source_alpha",
            "source_presence_required", "speed_limit_regulation_id",
            "speed_limit_regulation_ontology_class_id",
            "speed_limit_regulation_source_ref", "speed_mps", "signals",
            "stop_line_id", "stop_line_ontology_class_id",
            "stop_line_source_ref", "sumo_angle_deg", "sumo_edge_id",
            "sumo_front_bumper_xy_m", "sumo_lane_id",
            "sumo_position_reference", "sumo_xy_m", "truth_front_bumper_enu_m",
            "truth_position_reference", "vehicle_id", "vehicle_type",
        ],
        "properties": {
            "accel_mps2": _nullable_float(),
            "active_semantic_event": {"type": "null"},
            "allowed_speed_mps": _float_leaf(),
            "canonical_logical_asset_id": _identifier(),
            "center_lane_position_m": _nullable_float(),
            "control_role": {"const": "background"},
            "controlling_signal_id": {"type": "string"},
            "controlling_signal_ontology_class_id": {
                "oneOf": [{"const": "world:TrafficSignal"},
                          {"type": "null"}],
            },
            "controlling_signal_state": {"type": "string"},
            "crossed_stop_line": {"type": "boolean"},
            "dimensions_m": {
                "type": "object", "additionalProperties": False,
                "required": ["height", "length", "width"],
                "properties": {
                    "height": _float_leaf(),
                    "length": _float_leaf(),
                    "width": _float_leaf(),
                },
            },
            "following_distance_rule_id": _identifier(),
            "following_distance_rule_ontology_class_id": {
                "const": "world:FollowingDistanceRule",
            },
            "following_distance_rule_source_ref": _identifier(),
            "lane_id": _identifier(),
            "lane_instance_source_ref": _identifier(),
            "lane_ontology_class_id": {"const": "world:RoadLane"},
            "lane_position_m": _nullable_float(),
            "logical_asset_authority": {
                "const": "explicit_vehicle_plan_type",
            },
            "right_of_way_id": {
                "oneOf": [{"type": "string"}, {"type": "null"}],
            },
            "right_of_way_ontology_class_id": {
                "oneOf": [{"const": "world:RightOfWay"}, {"type": "null"}],
            },
            "right_of_way_source_ref": {
                "oneOf": [{"type": "string"}, {"type": "null"}],
            },
            "route_id": _identifier(),
            "semantic_episode_id": _identifier(),
            "semantic_metadata": _semantic_metadata(),
            "semantic_vehicle": {"type": "boolean"},
            "semantic_vehicle_state": {"type": "null"},
            "source_entity_id": {"type": "string"},
            "source_next_time_s": _float_leaf(),
            "source_prev_time_s": _float_leaf(),
            "source_alpha": _float_leaf(),
            "source_presence_required": {"type": "boolean"},
            "speed_limit_regulation_id": _identifier(),
            "speed_limit_regulation_ontology_class_id": {
                "const": "world:SpeedLimitRegulation",
            },
            "speed_limit_regulation_source_ref": _identifier(),
            "speed_mps": _nullable_float(),
            "signals": {"type": "integer", "minimum": 0},
            "stop_line_id": {
                "oneOf": [{"type": "string"}, {"type": "null"}],
            },
            "stop_line_ontology_class_id": {
                "oneOf": [{"const": "world:StopLine"}, {"type": "null"}],
            },
            "stop_line_source_ref": {
                "oneOf": [{"type": "string"}, {"type": "null"}],
            },
            "sumo_angle_deg": _nullable_float(),
            "sumo_edge_id": _identifier(),
            "sumo_front_bumper_xy_m": {
                "oneOf": [_float2(), {"type": "null"}],
            },
            "sumo_lane_id": _identifier(),
            "sumo_position_reference": {"const": "front_bumper_center"},
            "sumo_xy_m": {
                "oneOf": [_float2(), {"type": "null"}],
            },
            "truth_front_bumper_enu_m": _float3(),
            "truth_position_reference": {
                "const": "vehicle_body_center_from_front_bumper",
            },
            "vehicle_id": _identifier(),
            "vehicle_type": _vehicle_type(),
        },
    }
    if required_lifecycle:
        schema["required"].append("required_lifecycle_preservation_policy")
        schema["properties"]["required_lifecycle_preservation_policy"] = {
            "const": "required_sumo_vehicle_preserve_source_lifecycle_v1",
        }
    return schema


def _background_vehicle() -> dict:
    # Roster pass-through (:4409-4411); written keys per roster census.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "control_role", "policy", "semantic_episode_id",
            "semantic_vehicle", "source", "source_entity_id",
            "source_presence_required", "sumo_vehicle_id", "vehicle_type",
        ],
        "properties": {
            "control_role": {"const": "background"},
            "policy": {"const": "sumo_truth_frame_background_v1"},
            "semantic_episode_id": _identifier(),
            "semantic_vehicle": {"type": "boolean"},
            "source": {"const": "sumo_traci"},
            "source_entity_id": {"type": "string"},
            "source_presence_required": {"type": "boolean"},
            "sumo_vehicle_id": _identifier(),
            "vehicle_type": _vehicle_type(),
        },
    }


def _sumo_ground_flow_contract() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "actor_kind", "policy", "route_source", "sample_period_s",
            "segment",
        ],
        "properties": {
            "actor_kind": {"const": "vehicle"},
            "policy": {"const": "sumo_segment_truth_replay_v1"},
            "route_source": {
                "const": "sumo_traci_fcd_projected_to_ue_truth",
            },
            "sample_period_s": _float_leaf(),
            "segment": _segment(),
        },
    }


def _background_pedestrian() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["contract_scenario_id", "policy", "semantic_role"],
        "properties": {
            "contract_scenario_id": _identifier(),
            "policy": {"const": "semantic_event_contract_v1"},
            "semantic_role": _identifier(),
        },
    }


def _pedestrian_ground_flow_contract() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "actor_kind", "loop_policy", "min_visible_motion_ratio",
            "min_xy_span_m", "planned_path_length_m", "policy", "required",
            "route_duration_ticks", "route_source", "speed_mps",
        ],
        "properties": {
            "actor_kind": {"const": "pedestrian"},
            "loop_policy": {"const": "single_pass_route_waypoints"},
            "min_visible_motion_ratio": _float_leaf(),
            "min_xy_span_m": _float_leaf(),
            "planned_path_length_m": _float_leaf(),
            "policy": {"const": "continuous_capture_ground_flow_v1"},
            "required": {"type": "boolean"},
            "route_duration_ticks": {"type": "integer", "minimum": 0},
            "route_source": {"const": "traffic_bundle_sidewalk"},
            "speed_mps": _float_leaf(),
        },
    }


def _pad_motion_contract() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["actor_kind", "policy", "source"],
        "properties": {
            "actor_kind": {"const": "facility"},
            "policy": {"const": "static_facility_truth_frame_v1"},
            "source": {"const": "uav_global_flow"},
        },
    }


def _uav_motion_contract() -> dict:
    # Roster pass-through (:3412); census :4839-4847 (segment keys under
    # motion_contract.segment).
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
            "segment": _segment(),
            "source": {"const": "uav_global_flow"},
            "speed_mps": _float_leaf(),
        },
    }


def _uav_global_pad() -> dict:
    # uav_global_pad block (:3179-3188); the weight arrays hold 2-3
    # floats (census shapes [[2],[3]]).
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
                "type": "array", "items": _float_leaf(),
            },
            "phase_origin_weights": {
                "type": "array", "items": _float_leaf(),
            },
            "role": _identifier(),
        },
    }


def _uav_global_flow_payload() -> dict:
    # Roster 15-key payload (:3406) + the per-frame merge (:3407-3410);
    # active_event_ids is empty in the corpus (非空成员：尚无实例验证).
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "active_event_ids", "altitude_layer_m", "corridor_family",
            "corridor_id", "ground_reference_z_m", "mission_type",
            "origin_pad_id", "policy", "sample_period_s", "semantic_role",
            "source_alpha", "source_next_time_s", "source_prev_time_s",
            "target_cell_id", "target_pad_id", "task_id", "uav_id",
        ],
        "properties": {
            "active_event_ids": {"type": "array", "items": _identifier()},
            "altitude_layer_m": _float_leaf(),
            "corridor_family": _identifier(),
            "corridor_id": _identifier(),
            "ground_reference_z_m": _float_leaf(),
            "mission_type": _identifier(),
            "origin_pad_id": {"type": "string"},
            "policy": {"const": "donghu_global_uav_flow_truth_replay_v1"},
            "sample_period_s": _float_leaf(),
            "semantic_role": _identifier(),
            "source_alpha": _float_leaf(),
            "source_next_time_s": _float_leaf(),
            "source_prev_time_s": _float_leaf(),
            "target_cell_id": _identifier(),
            "target_pad_id": {"type": "string"},
            "task_id": _identifier(),
            "uav_id": _identifier(),
        },
    }


def _tls_state() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "controlled_links", "next_switch_s", "phase_index", "program_id",
            "state",
        ],
        "properties": {
            "controlled_links": {
                "type": "array",
                "items": {"type": "array",
                          "items": {"type": "array",
                                    "items": {"type": "string"}}},
            },
            "next_switch_s": _float_leaf(),
            "phase_index": {"type": "integer", "minimum": 0},
            "program_id": {"type": "string"},
            "state": {"type": "string", "minLength": 1},
        },
    }


def _incident_row() -> dict:
    # _compact_incident (truth_integration.py :574-599); traffic_light_id
    # exists exactly on the all-red-fault class (:594-598).
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "accident_class", "anchor", "affected_vehicle_ids",
            "end_s", "episode_event_id", "episode_scenario_id",
            "incident_id", "injection_method", "intent", "intent_stage",
            "start_s",
        ],
        "properties": {
            "accident_class": _identifier(),
            "affected_vehicle_ids": {"type": "array",
                                     "items": _identifier()},
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
    }


def _semantic_scope() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "authority_id", "ontology_class_id", "scope_subtype", "scope_type",
        ],
        "properties": {
            "authority_id": {"const": "facility_scope_contract.v1"},
            "ontology_class_id": {
                "enum": [
                    "world:Barrier", "world:ChargingStation",
                    "world:CommunicationStation", "world:LandingPad",
                    "world:NoFlyZone",
                ],
            },
            "scope_subtype": {
                "enum": [
                    "barrier", "charging_station",
                    "communication_base_station", "landing_pad",
                    "no_fly_zone",
                ],
            },
            "scope_type": {"const": "facility"},
            "service_capacity": {"type": "integer", "minimum": 0},
        },
    }


def _uav_corridor() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "altitude_layers_m", "assigned_altitude_m", "used_lateral_bypass",
        ],
        "properties": {
            "altitude_layers_m": {"type": "array", "items": _float_leaf()},
            "assigned_altitude_m": _float_leaf(),
            "used_lateral_bypass": {"type": "boolean"},
        },
    }


def _source_task_properties(*, category: str) -> dict:
    return {
        "activation_tick": {"type": "integer", "minimum": 0},
        "category": {"const": category},
        "role": _identifier(),
        "route_waypoints_enu_m": _float3_array(),
        "semantic_role": _identifier(),
        "state_sequence": {"type": "array", "items": _identifier()},
        "task_id": _identifier(),
    }


def _source_uav_properties() -> dict:
    return {
        **_source_task_properties(category="uav"),
        "assigned_altitude_m": _float_leaf(),
        "control_state": _runtime_state_family("control_state"),
        "ground_reference_source": _identifier(),
        "ground_reference_z_m": _float_leaf(),
        "planned_route_waypoints_enu_m": _float3_array(),
        "sensor_state": _runtime_state_family("sensor_state"),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": _identifier(),
    }


def _entity_core_properties() -> dict:
    # The 16 keys every writer places first (:5114-5133, :3383-3402,
    # :4389-4406, :3208-3226); ``state`` is written conditionally.
    return {
        "annotations": _annotations(),
        "entity_category": _identifier(),
        "entity_id": _identifier(),
        "entity_kind": _identifier(),
        "entity_type": _identifier(),
        "label_class": _identifier(),
        "logical_asset_id": _identifier(),
        "proxy_template_id": {"type": "string"},
        "render_presence": {
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
        },
        "roi_id": _identifier(),
        "runtime_visibility": _runtime_visibility(),
        "site_id": _identifier(),
        "state": _identifier(),
        "state_revision": {"type": "integer", "minimum": 1},
        "tags": {"type": "array", "items": _identifier()},
        "truth_pose": _truth_pose(),
        "uav_visibility": _uav_visibility(),
        "visual_revision": {"const": 1},
    }


def _frame_entity_branch(properties: dict, required: list | tuple) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": list(required),
        "properties": {**_entity_core_properties(), **properties},
    }


_ENTITY_CORE_REQUIRED = (
    "annotations", "entity_category", "entity_id", "entity_kind",
    "entity_type", "label_class", "logical_asset_id", "proxy_template_id",
    "render_presence", "roi_id", "runtime_visibility", "site_id",
    "state_revision", "tags", "truth_pose", "visual_revision",
)


_SOURCE_TASK_REQUIRED = (
    "activation_tick", "category", "role", "route_waypoints_enu_m",
    "semantic_role", "state_sequence", "task_id",
)
_SOURCE_UAV_REQUIRED = _SOURCE_TASK_REQUIRED + (
    "assigned_altitude_m", "ground_reference_source", "ground_reference_z_m",
    "planned_route_waypoints_enu_m", "uav_corridor", "uav_corridor_role",
    "uav_visibility",
)


def _sumo_row(*, required_lifecycle: bool, retained: bool) -> dict:
    if retained and not required_lifecycle:
        raise ValueError("retained SUMO rows require lifecycle preservation")
    return _frame_entity_branch(
        {
            "background_role": {"const": "sumo_background_traffic"},
            "background_vehicle": _background_vehicle(),
            "capture_boundary_id": _identifier(),
            "control_state": _runtime_state_family("control_state"),
            "ground_flow_contract": _sumo_ground_flow_contract(),
            "runtime_visibility": _sumo_runtime_visibility(retained=retained),
            "sensor_state": _runtime_state_family("sensor_state"),
            "source": {"const": "sumo_traci"},
            "sumo_segment": _segment(),
            "sumo_vehicle": _sumo_vehicle(
                required_lifecycle=required_lifecycle
            ),
            "sumo_visibility": {
                "type": "object", "additionalProperties": False,
                "required": ["inspect_observation_distance_m",
                             "selected_for_capture_truth"],
                "properties": {
                    "inspect_observation_distance_m": _float_leaf(),
                    "selected_for_capture_truth": {"const": True},
                },
            },
        },
        _ENTITY_CORE_REQUIRED + (
            "background_role", "background_vehicle", "capture_boundary_id",
            "ground_flow_contract", "source", "state", "sumo_segment",
            "sumo_vehicle", "sumo_visibility",
        ),
    )


def _uav_flow_row() -> dict:
    return _frame_entity_branch(
        {
            "background_role": {"const": "donghu_global_uav_flow"},
            "mission_type": _identifier(),
            "motion_contract": _uav_motion_contract(),
            "planned_route_waypoints_enu_m": _float3_array(),
            "semantic_role": _identifier(),
            "source": {"const": "uav_global_flow"},
            "task_id": _identifier(),
            "uav_global_flow": _uav_global_flow_payload(),
            "uav_segment": _segment(),
        },
        _ENTITY_CORE_REQUIRED + (
            "background_role", "motion_contract", "semantic_role", "source",
            "uav_global_flow", "uav_segment", "uav_visibility",
        ),
    )


def _pad_row() -> dict:
    return _frame_entity_branch(
        {
            "background_role": {"const": "global_uav_pad"},
            "motion_contract": _pad_motion_contract(),
            "semantic_scope": _semantic_scope(),
            "source": {"const": "uav_global_flow"},
            "uav_global_pad": _uav_global_pad(),
        },
        _ENTITY_CORE_REQUIRED + (
            "background_role", "motion_contract", "source",
            "uav_global_pad",
        ),
    )


def _inspect_uav_row() -> dict:
    return _frame_entity_branch(
        {
            **_source_uav_properties(),
            "contract_inspect_uav": _contract_inspect_uav(),
            "full_episode_presence": {"const": True},
            "inspect_altitude_code": _identifier(),
            "inspect_altitude_m": _float_leaf(),
            "inspect_fov_coverage_required": {"type": "boolean"},
            "min_path_length_m": _float_leaf(),
        },
        _ENTITY_CORE_REQUIRED + _SOURCE_UAV_REQUIRED + (
            "contract_inspect_uav", "full_episode_presence",
            "inspect_altitude_code", "inspect_altitude_m",
            "inspect_fov_coverage_required", "min_path_length_m",
        ),
    )


def _mission_uav_row() -> dict:
    return _frame_entity_branch(
        {**_source_uav_properties(), "lifecycle": _lifecycle()},
        _ENTITY_CORE_REQUIRED + _SOURCE_UAV_REQUIRED + ("lifecycle",),
    )


def _observer_uav_row() -> dict:
    return _frame_entity_branch(
        {
            **_source_uav_properties(),
            "observer_lifecycle": _observer_lifecycle(),
        },
        _ENTITY_CORE_REQUIRED + _SOURCE_UAV_REQUIRED + ("observer_lifecycle",),
    )


def _facility_properties() -> dict:
    return _source_task_properties(category="facility")


def _semantic_facility_row() -> dict:
    return _frame_entity_branch(
        {**_facility_properties(), "semantic_scope": _semantic_scope()},
        _ENTITY_CORE_REQUIRED + _SOURCE_TASK_REQUIRED + ("semantic_scope",),
    )


def _contract_facility_row() -> dict:
    return _frame_entity_branch(
        {
            **_facility_properties(),
            "contract_facility": _semantic_event_contract(),
            "contract_scenario_id": _identifier(),
            "semantic_scope": _semantic_scope(),
        },
        _ENTITY_CORE_REQUIRED + _SOURCE_TASK_REQUIRED + (
            "contract_facility", "contract_scenario_id", "semantic_scope",
        ),
    )


def _repaired_facility_row(*, contract: bool) -> dict:
    properties = {
        **_facility_properties(),
        "scene_occupancy_repair": _scene_occupancy_repair(),
        "semantic_scope": _semantic_scope(),
    }
    required = _SOURCE_TASK_REQUIRED + (
        "scene_occupancy_repair", "semantic_scope",
    )
    if contract:
        properties.update({
            "contract_facility": _semantic_event_contract(),
            "contract_scenario_id": _identifier(),
        })
        required += ("contract_facility", "contract_scenario_id")
    return _frame_entity_branch(properties, _ENTITY_CORE_REQUIRED + required)


def _pedestrian_row() -> dict:
    return _frame_entity_branch(
        {
            **_source_task_properties(category="pedestrian"),
            "background_pedestrian": _background_pedestrian(),
            "background_role": {"const": "semantic_context"},
            "contract_scenario_id": _identifier(),
            "ground_flow_contract": _pedestrian_ground_flow_contract(),
        },
        _ENTITY_CORE_REQUIRED + _SOURCE_TASK_REQUIRED + (
            "background_pedestrian", "background_role", "contract_scenario_id",
            "ground_flow_contract",
        ),
    )


def _corridor_row(*, shared: bool) -> dict:
    properties = {
        "activation_tick": {"type": "integer", "minimum": 0},
        "category": {"const": "airspace_corridor"},
    }
    required = ("activation_tick", "category")
    if shared:
        properties["semantic_role"] = _identifier()
        required += ("semantic_role",)
    return _frame_entity_branch(properties, _ENTITY_CORE_REQUIRED + required)


def _entities() -> dict:
    return {
        "type": "array",
        "items": {"oneOf": [
            _sumo_row(required_lifecycle=False, retained=False),
            _sumo_row(required_lifecycle=True, retained=False),
            _sumo_row(required_lifecycle=True, retained=True),
            _uav_flow_row(),
            _pad_row(),
            _inspect_uav_row(),
            _mission_uav_row(),
            _observer_uav_row(),
            _semantic_facility_row(),
            _contract_facility_row(),
            _repaired_facility_row(contract=False),
            _repaired_facility_row(contract=True),
            _pedestrian_row(),
            _corridor_row(shared=False),
            _corridor_row(shared=True),
        ]},
    }


def formal_truth_frame_schema() -> dict:
    """Return the closed record schema for one formal truth frame row.

    The record root is the whole JSONL row (``formal_truth_frame`` is
    not an entity file in ``structure_sources.py``).  All 26 root keys
    are written every frame (:5367-5409), so the root is one closed
    object.  The entity-keyed maps ``entity_motion_state`` and
    ``sumo_traffic_light_states`` keep instance binding through anchored
    ``propertyNames`` patterns; ``roster_summary.by_category`` keys are
    category tokens.  Entity rows split into 15 closed producer branches:
    three SUMO lifecycle/retention forms, global UAV and pad forms, and ten
    L1 source-row forms.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_truth_frame_l1.record.schema.json",
        "title": "AERO_WORLD formal_truth_frame L1 frame record schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "active_roi_id", "active_site_id", "capture_boundary_id",
            "dt_s", "entities", "episode_id", "entity_motion_state",
            "frame_id", "frame_seq", "inspect_observes_boundary", "map_id",
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
            "entities": _entities(),
            "episode_id": _identifier(),
            "entity_motion_state": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "propertyNames": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9_]+$",
                },
                "$comment": "truth_boundary_summary :3113-3123",
            },
            "frame_id": {
                "type": "string",
                "pattern": "^.+_tick_[0-9]+$",
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
            "sumo_active_incidents": {
                "type": "array", "items": _incident_row(),
            },
            "sumo_segment": {"oneOf": [_segment(), {"type": "null"}]},
            "sumo_semantics": {
                "oneOf": [
                    {
                        "type": "object", "additionalProperties": False,
                        "required": [
                            "absolute_time_s", "active_incident_count",
                            "active_incidents", "active_selected_vehicle_count",
                            "enabled", "max_observation_distance_m",
                            "retained_outside_boundary_sumo_vehicle_count",
                            "runtime_boundary_visible_selected_vehicle_count",
                            "scenario_id", "segment",
                            "semantic_lifecycle_outside_runtime_boundary_count",
                            "source_alpha", "source_next_time_s",
                            "source_prev_time_s", "source_vehicle_count",
                            "selected_vehicle_count", "traffic_light_count",
                        ],
                        "properties": {
                            "absolute_time_s": _float_leaf(),
                            "active_incident_count": {
                                "type": "integer", "minimum": 0,
                            },
                            "active_incidents": {
                                "type": "array", "items": _incident_row(),
                            },
                            "active_selected_vehicle_count": {
                                "type": "integer", "minimum": 0,
                            },
                            "enabled": {"const": True},
                            "max_observation_distance_m": {
                                "const": 60.0,
                            },
                            "retained_outside_boundary_sumo_vehicle_count": {
                                "type": "integer", "minimum": 0,
                            },
                            "runtime_boundary_visible_selected_vehicle_count": {
                                "type": "integer", "minimum": 0,
                            },
                            "scenario_id": _identifier(),
                            "segment": _segment(),
                            "semantic_lifecycle_outside_runtime_boundary_count": {
                                "type": "integer", "minimum": 0,
                            },
                            "source_alpha": _float_leaf(),
                            "source_next_time_s": _float_leaf(),
                            "source_prev_time_s": _float_leaf(),
                            "source_vehicle_count": {
                                "type": "integer", "minimum": 0,
                            },
                            "selected_vehicle_count": {
                                "type": "integer", "minimum": 0,
                            },
                            "traffic_light_count": {
                                "type": "integer", "minimum": 0,
                            },
                        },
                    },
                    {"type": "null"},
                ],
            },
            "sumo_traffic_light_states": {
                "type": "object",
                "additionalProperties": _tls_state(),
                "propertyNames": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9_]+$",
                },
                "$comment": "_traffic_light_states :560-571",
            },
            "tick": {"type": "integer", "minimum": 0},
            "tick_hz": {"const": 10},
            "uav_crosses_boundary": {"type": "boolean"},
            "uav_global_flow": {
                "oneOf": [{
                    "type": "object", "additionalProperties": False,
                    "required": [
                    "absolute_time_s", "active_selected_uav_count",
                    "baseline_active_uavs", "camera_capture_roi_policy", "enabled",
                    "minimum_active_uavs_required",
                    "roi_capture_eligible_uav_count", "segment",
                    "selected_uav_count", "source_alpha",
                    "source_next_time_s", "source_prev_time_s",
                    "source_uav_count", "task_count",
                    ],
                    "properties": {
                    "absolute_time_s": _float_leaf(),
                    "active_selected_uav_count": {
                        "type": "integer", "minimum": 0,
                    },
                    "baseline_active_uavs": {
                        "type": "integer", "minimum": 0,
                    },
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
                    "segment": _segment(),
                    "selected_uav_count": {"type": "integer", "minimum": 0},
                    "source_alpha": _float_leaf(),
                    "source_next_time_s": _float_leaf(),
                    "source_prev_time_s": _float_leaf(),
                    "source_uav_count": {"type": "integer", "minimum": 0},
                    "task_count": {"type": "integer", "minimum": 0},
                    },
                }, {"type": "null"}],
            },
            "uav_segment": {"oneOf": [_segment(), {"type": "null"}]},
        },
    }
