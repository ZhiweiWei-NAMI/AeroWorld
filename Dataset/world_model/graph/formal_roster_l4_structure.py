"""Closed record structure for one formal roster JSON root, L4 layer.

Source family ``formal_roster``; profiles L4-1_v1, L4-1_v2, L4-2_v1,
L4-2_v2, L4-3_v1, L4-3_v2, L4-3_v3, L4-4_v1, L4-4_v2, L4-5_v1, L4-5_v2, L4-5_v3,
L4-6_v1, L4-6_v2, L4-7_v1, L4-7_v2, L4-8_v1, L4-8_v2, L4-9_v1, L4-9_v2,
L4-10_v1, L4-10_v2, L4-11_v1, L4-11_v2 (24 profiles x 3 seeds = 72
declarations).  Producer chain: the scenario ``spec.py`` authors the scene
entities and ``initial_state``, ``Dataset/tools/batch_generate.py`` writes
hop-1 trajectory rows where every row is the sample keys plus
``preserved_fields_from(entity)`` (``_row_for_entity`` :2395-2415, merge
:2414; preserve list :215-256, ``preserved_fields_from`` :1737-1763),
``Dataset/tools/convert_to_render_ready.py`` writes the per-episode roster
entry — the 14-key base block (:5031-5046), the preserve merge (:5047),
the landing-pad kind fix (:5048-5060), the inspect-route overwrite
(:5061-5064), and ``runtime_visibility`` (:5107-5113) — and drops rows
without visible ticks (:5081-5086); SUMO roster entries are appended by
``build_sumo_vehicle_roster_entries`` (:4250-4406) and UAV global-flow
entries by the UAV roster builders; ``Dataset/tools/
filter_render_ready_truth_for_capture.py`` copies roster episodes through
to the transfer boundary unchanged, so every roster byte at
``aw_data/render_ready_episodes_capture_filtered`` is the converter's
output.

Record identity: ``structure_sources.py`` lists ``formal_roster`` in
``_ENTITY_FILES``; ``iter_source_records`` reads one whole JSON root per
file and validates that root before projection.  The converter writes
``{"entities": [...]}``.  ``_project_full_or_entity_root`` later projects
each entity with prefix tokens ``("entities", index)`` for the field
catalog.  The grammar root is the JSON object; the entity oneOf is the
schema of each ``entities[]`` item.

Branches — the grammar declares the keyset families the producer chain
writes at the L4 layer.  Every branch closes with
``additionalProperties: False`` and declares exactly its keyset, so the
marker keys (``sumo_vehicle_id`` + ``source`` const, ``spawn_policy`` +
``event_actor_motion_contract``, ``background_role`` const
``semantic_context``, ``uav_id``/``uav_pad_id``, ``contract_inspect_uav``,
``lifecycle``, ``observer_lifecycle``, ``background_pedestrian``,
``contract_facility``, ``semantic_scope``, the ``entity_category``/``category``
const pair on the logical-only anchors, ``semantic_role`` on corridors) leave
one matching branch per row.  ``placement`` + ``scene_occupancy_repair`` are
payload keys rather than markers: they appear on the repaired facility row and
on the repaired global pads, each of which is already selected by its own
marker.  Keysets verified against authored scene entities in
``Dataset/scenarios/L4_agents/interaction/L4-10_v1/spec.py``
(ambulance and civilian-yield event actors :38-126, semantic background
vehicle :343-391, background pedestrian :620-664, mission/inspect/observer
UAVs, pads, facilities, corridor rows through the spawn lists), and the
per-file roster census over all 72 L4 roots (5587 entities) now confirms the
enumeration and the pairwise branch-exclusivity property.

L4 deltas against the L1 sibling grammar
(``formal_roster_l1_structure.py``):
- Sumo vehicle rows: ``background_role`` is the converter's ternary
  (:4293-4299) — ``sumo_semantic_event_vehicle`` /
  ``sumo_source_required_context_vehicle`` / ``sumo_background_traffic`` —
  so the L1 const widens to that three-member enum; the block structure
  (``background_vehicle``, ``ground_flow_contract``, ``sumo_segment``,
  ``sumo_vehicle``, ``sumo_vehicle_id``, ``sumo_visibility``, optional
  runtime-state families) is unchanged.
- ``ground_flow_contract`` on ground vehicles carries the authored
  continuous-capture contract when the row keeps its scene identity
  (semantic background vehicle rows, e.g. L4-10_v1 ``bg_vehicle_*``:
  policy ``continuous_capture_ground_flow_v1``, ``route_source``
  ``sumo_net_directed_lane_geometry``, optional
  ``physical_lane_lateral_m``), while sumo rows keep the converter's
  ``sumo_segment_truth_replay_v1`` — the two forms never mix on one
  branch.  The pedestrian contract keeps the L1 shape verbatim
  (``route_source`` const ``traffic_bundle_sidewalk``).
- New family, event-actor vehicle rows: ``stage_event_actor``
  (``regenerate_boundary_scenarios.py`` :1411-1432) writes
  ``activation_tick`` = DEFAULT_DURATION_TICKS + 1, ``enabled`` (scene
  level, never preserved), ``spawn_policy`` = ``event_script_only`` and
  the 6-key base ``event_actor_motion_contract``; ``event_actor_entry``
  (:1435-1513) appends the 7 entry keys (:1482-1492) and the boundary
  annotation pass adds ``entry_capture_boundary_clearance_m``
  (EVENT_ACTOR_ENTRY_BOUNDARY_CLEARANCE_M :288) — the 14-key entry form
  observed on both L4-10_v1 event actors.  ``spawn_event_actor_at_dispatch``
  (:1516-1529) stops at the 6-key base form.  ``motion_contract`` and
  ``capture_boundary``/``capture_contract`` are NOT in the hop-1 preserve
  list (:215-256), so they never reach these rows; ``vehicle_state`` does
  (ambulance ``priority_request_acknowledged``, civilian ``yielding``).
  The entry variant is the observed form; the at-dispatch 6-key variant is
  declared from its writer and has no verified roster instance.
- New family, semantic background vehicle rows: authored background
  vehicles that keep their scene identity (not selected into SUMO — the
  converter only replaces rows for ``explicit_source_vehicle_ids``
  membership at :4990-4996) carry ``background_role`` =
  ``semantic_context``, ``background_vehicle`` (with the authored
  ``physical_lane_lateral_m`` and ``semantic_role``), the continuous
  ground-flow contract, and no ``source``/``sumo_*`` keys.
- Runtime-state families: the merge reaches rows through
  ``runtime_state_fields_from_sources`` (sumo entries :4398-4404) and the
  preserved scene fields; the chain carries ``control_state``,
  ``sensor_state`` and ``vehicle_state`` payloads (event actors author
  ``vehicle_state``, e.g. the L4-10_v1 ambulance
  ``priority_request_acknowledged`` :82).  Those three families are
  declared closed
  to the full executable producer contract in
  ``Dataset/tools/runtime_state_contract.py`` (booleans, enum members via
  the raw-value rule, command members, metadata strings; ``minProperties:
  1``) exactly as the L1 sibling closes ``control_state``/``sensor_state``.
  ``mission_state`` and ``incident_state`` reach the scoped UAV rows through
  the same merge (L4-11_v2 carries ``hazard_managed`` and L4-3_v1 carries
  ``physical_fault_active``).  Other runtime
  families without a verified L4 roster payload are not declared here.
- Logical-only anchors: ``profile_for_entity`` selects
  ``logical_only_profile`` for ``semantic.asset_anchor`` and keeps the
  authored ``facade_anchor`` category (L4-2_v1/v2); the same profile keeps
  ``semantic.spawn_zone`` and the authored ``crowd_anchor`` category for the
  L4-8_v1/v2 crowd anchors.  These rows have ``mode=logical_contract``, no
  ``semantic_scope``, and differ only in the ``entity_category``/``category``
  and asset/label consts, so they occupy two sibling branches.
- Pedestrians: authored background-contract rows keep
  ``background_pedestrian`` and ``ground_flow_contract``.  Their
  ``background_role`` is copied only when authored (L4-10_v1 has it;
  L4-3_v1 does not).  The L4-3_v1 ordinary pedestrians have neither
  contract and keep ``role=semantic_pedestrian``.  The ordinary landing
  actors a/b retain their authored ``pedestrian_state.evacuation_active``,
  and the L4-7_v1/v2 fallen actor adds ``incident_state.dispatch_active``
  beside ``pedestrian_state`` fallen/injured/health/posture.
- Mission UAV lifecycle: L4-4_v1 authors the optional boolean
  ``terminal_flight_state`` inside the otherwise closed lifecycle block.

Still declared from producer writers only, with no verified L4 roster record:
the at-dispatch event-actor arm and the unrepairable arm of ``placement`` +
``scene_occupancy_repair`` (a repaired facility row that carries the contract
keys).  The repairable arm is verified: ``convert_to_render_ready.py``
:1273-1301 writes the pair onto the re-anchored landing pads observed on
L4-4_v2 and L4-6_v1, and onto the re-anchored facilities.  The
``facility_state``/``navigation_state``/``security_state``/``communication_state``
runtime families have no observed L4 roster payload
and are deliberately undeclared.  The per-file L4 census and the pairwise
branch-exclusivity sweep over all 72 files run as host validation, never at
import time.

Publication exclusion: roster rows carry no digest/integrity/hash field
(``is_publication_excluded_path`` has no roster-family rule and no roster
key matches), so this module declares no excluded path.  Top-level
``metadata`` and per-entity metadata blocks do not exist on any observed
row.

Boundary: this module exports the pure schema constructor only.  It is
not registered in ``structure_closure.py`` (``formal_roster`` is absent
from ``CLOSED_RECORD_FAMILIES`` and from the ``validate_source_record``
dispatch); registration/dispatch wiring belongs to the integration stage.
The module scans no data at import and compiles nothing at import.
"""

from __future__ import annotations

import re

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
)

__all__ = ["formal_roster_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _number3() -> dict:
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _waypoints() -> dict:
    # Producer pass-through of the authored route; the converter copies the
    # authored list verbatim, so the empty list stays a declared status
    # (event actors author empty routes, e.g. L4-10_v1 ambulance).
    return {"type": "array", "items": _number3()}


def _nullable_string() -> dict:
    return {"oneOf": [{"type": "string"}, {"type": "null"}]}


def _nullable(const: str) -> dict:
    # Populated exactly when the lane has a signal connection
    # (road_signal_context.py :225-252), null otherwise.
    return {"oneOf": [{"const": const}, {"type": "null"}]}


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


_SUMO_PHASE = {
    "const": "episode_local_vehicle_authority_after_warmup",
}
_UAV_PHASE = {
    "enum": [
        "uav_demand_peak_hold",
        "uav_demand_ramp_up",
        "uav_demand_taper_with_persistent_patrol",
    ],
}
_KNOWN_SUMO_LOGICAL_ASSETS = (
    "vehicle.emergency.ambulance.v1",
    "vehicle.emergency.police_suv.v1",
    "vehicle.emergency.suv.v1",
    "vehicle.ground.boxcar.v1",
    "vehicle.service.box.v1",
)


def _vehicle_type() -> dict:
    # explicit_vehicle_plan writes the bare profile type or
    # "<type>@<sumo vehicle id>"; types follow the converter's
    # SUMO_TYPE_LOGICAL_ASSET_BY_TYPE_ID mapping.
    return {
        "type": "string",
        "pattern": (
            "^(aero_ambulance|aero_delivery|aero_emergency|"
            "aero_passenger|aero_police)(@[^@\\s]+)?$"
        ),
    }


def _runtime_visibility() -> dict:
    # convert_to_render_ready.runtime_visibility_payload (:3004), written
    # per entity at :5107-5113.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "expanded_boundary_padding_m", "first_visible_tick",
            "last_visible_tick", "policy", "visible_tick_count",
        ],
        "properties": {
            "expanded_boundary_padding_m": _float_leaf(),
            "first_visible_tick": {"type": "integer", "minimum": 0},
            "last_visible_tick": {"type": "integer", "minimum": 0},
            "policy": {
                "const": "roi_polygon_expanded_60m_runtime_truth_crop_v1",
            },
            "visible_tick_count": {"type": "integer", "minimum": 0},
        },
    }


def _semantic_scope() -> dict:
    # facility_scope.semantic_scope_for_source (:95-125, dict :150).
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
    # The scene-authored semantic-event contract; the three route arrays
    # stay producer-typed waypoint arrays (empty on the L1 corpus;
    # L4-10_v1 authors 22-point loop/planned/repaired routes :187-257).
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
            "terminal_flight_state": {"type": "boolean"},
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


def _seed_profile() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["direction_bias", "profile_id", "seed_index",
                     "seed_label"],
        "properties": {
            "direction_bias": _identifier(),
            "profile_id": _identifier(),
            "seed_index": {"type": "integer", "minimum": 0},
            "seed_label": {"type": "string", "pattern": "^seed[0-9]{2}$"},
        },
    }


def _sumo_semantic_metadata() -> dict:
    # Two producer forms: the 2-key identity block and the 27-key
    # background block (explicit_vehicle_plan.py).  The semantic_* slots
    # are left unwritten at roster level; None/""/[] are declared statuses.
    # build_sumo_vehicle_roster_entries overwrites logical_asset_id with
    # canonical_asset, whose domain is KNOWN_VEHICLE_LOGICAL_ASSETS in the
    # converter.  This includes the L4-11 emergency SUV background row.
    return {"oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": ["logical_asset_authority", "logical_asset_id"],
            "properties": {
                "logical_asset_authority": {
                    "const": "explicit_vehicle_plan_type",
                },
                "logical_asset_id": {"enum": list(_KNOWN_SUMO_LOGICAL_ASSETS)},
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
                "direction_role": {
                    "enum": ["cross", "inbound", "outbound"],
                },
                "expected_core_entry_tick": {"type": "integer"},
                "expected_core_exit_tick": {"type": "integer"},
                "explicit_vehicle_plan": {"type": "boolean"},
                "logical_asset_authority": {
                    "const": "explicit_vehicle_plan_type",
                },
                "logical_asset_id": {"enum": list(_KNOWN_SUMO_LOGICAL_ASSETS)},
                "protected_window_role": {
                    "const": "background_forbidden_from_protected_edges",
                },
                "release_tick": {"type": "integer"},
                "required_edges": {"type": "array",
                                   "items": {"type": "string"}},
                "role": {"const": "background"},
                "route_source": {"const": "sumo_findRoute"},
                "seed_profile": _seed_profile(),
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


def _sumo_vehicle() -> dict:
    # convert_to_render_ready.py :4336-4385; the 16 road-authority keys are
    # road_signal_context.static_sumo_vehicle_fields (:189-254).
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "active_semantic_event", "allowed_speed_mps",
            "canonical_logical_asset_id", "center_lane_position_m",
            "control_role", "controlling_signal_id",
            "controlling_signal_ontology_class_id",
            "controlling_signal_state", "crossed_stop_line",
            "following_distance_rule_id",
            "following_distance_rule_ontology_class_id",
            "following_distance_rule_source_ref", "lane_id",
            "lane_instance_source_ref", "lane_ontology_class_id",
            "lane_position_m", "logical_asset_authority",
            "right_of_way_id", "right_of_way_ontology_class_id",
            "right_of_way_source_ref", "route_id", "semantic_episode_id",
            "semantic_metadata", "semantic_vehicle", "semantic_vehicle_state",
            "source_entity_id", "source_presence_required",
            "speed_limit_regulation_id",
            "speed_limit_regulation_ontology_class_id",
            "speed_limit_regulation_source_ref", "stop_line_id",
            "stop_line_ontology_class_id", "stop_line_source_ref",
            "sumo_edge_id", "sumo_lane_id", "sumo_position_reference",
            "truth_position_reference", "vehicle_id", "vehicle_type",
        ],
        "properties": {
            # The planner leaves both semantic slots unwritten at roster
            # level; the converter copies them verbatim.
            "active_semantic_event": {"type": "null"},
            "allowed_speed_mps": _float_leaf(),
            "canonical_logical_asset_id": {
                "enum": list(_KNOWN_SUMO_LOGICAL_ASSETS),
            },
            "center_lane_position_m": _float_leaf(),
            "control_role": {"const": "background"},
            "controlling_signal_id": {"type": "string"},
            "controlling_signal_ontology_class_id": _nullable(
                "world:TrafficSignal"),
            "controlling_signal_state": {
                "const": "not_sampled_at_roster_level",
            },
            "crossed_stop_line": {"type": "boolean"},
            "following_distance_rule_id": _identifier(),
            "following_distance_rule_ontology_class_id": {
                "const": "world:FollowingDistanceRule",
            },
            "following_distance_rule_source_ref": _identifier(),
            "lane_id": _identifier(),
            "lane_instance_source_ref": _identifier(),
            "lane_ontology_class_id": {"const": "world:RoadLane"},
            "lane_position_m": _float_leaf(),
            "logical_asset_authority": {
                "const": "explicit_vehicle_plan_type",
            },
            "right_of_way_id": _nullable_string(),
            "right_of_way_ontology_class_id": _nullable("world:RightOfWay"),
            "right_of_way_source_ref": _nullable_string(),
            "route_id": _identifier(),
            "semantic_episode_id": _identifier(),
            "semantic_metadata": _sumo_semantic_metadata(),
            "semantic_vehicle": {"type": "boolean"},
            "semantic_vehicle_state": {"type": "null"},
            "source_entity_id": {"type": "string"},
            "source_presence_required": {"type": "boolean"},
            "speed_limit_regulation_id": _identifier(),
            "speed_limit_regulation_ontology_class_id": {
                "const": "world:SpeedLimitRegulation",
            },
            "speed_limit_regulation_source_ref": _identifier(),
            "stop_line_id": _nullable_string(),
            "stop_line_ontology_class_id": _nullable("world:StopLine"),
            "stop_line_source_ref": _nullable_string(),
            "sumo_edge_id": _identifier(),
            "sumo_lane_id": _identifier(),
            "sumo_position_reference": {"const": "front_bumper_center"},
            "truth_position_reference": {
                "const": "vehicle_body_center_from_front_bumper",
            },
            "vehicle_id": _identifier(),
            "vehicle_type": _vehicle_type(),
        },
    }


def _background_vehicle() -> dict:
    # Sumo rows: convert_to_render_ready.py :4337-4348.
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


def _authored_background_vehicle() -> dict:
    # Authored dict on semantic background vehicle rows (L4-10_v1 spec
    # :344-347): contract_scenario_id, physical_lane_lateral_m, policy,
    # semantic_role.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["contract_scenario_id", "policy"],
        "properties": {
            "contract_scenario_id": _identifier(),
            "physical_lane_lateral_m": _float_leaf(),
            "policy": {"const": "semantic_event_contract_v1"},
            "semantic_role": {"type": "string"},
        },
    }


def _sumo_ground_flow_contract() -> dict:
    # Sumo rows keep the converter contract verbatim
    # (convert_to_render_ready.py :4348); rows that keep their scene
    # identity carry the authored continuous-capture contract instead.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["actor_kind", "policy", "route_source",
                     "sample_period_s", "segment"],
        "properties": {
            "actor_kind": {"const": "vehicle"},
            "policy": {"const": "sumo_segment_truth_replay_v1"},
            "route_source": {
                "const": "sumo_traci_fcd_projected_to_ue_truth",
            },
            "sample_period_s": _float_leaf(),
            "segment": _segment(_SUMO_PHASE),
        },
    }


def _ground_vehicle_continuous_flow_contract() -> dict:
    # Authored continuous-capture contract on semantic background vehicle
    # rows (regenerate_boundary_scenarios.py :275-286 constants;
    # L4-10_v1 bg_vehicle :350-360).  Vehicles route on the SUMO net
    # directed lane geometry; the authored lateral offset is optional.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "actor_kind", "loop_policy", "min_visible_motion_ratio",
            "min_xy_span_m", "planned_path_length_m", "policy",
            "required", "route_duration_ticks", "route_source", "speed_mps",
        ],
        "properties": {
            "actor_kind": {"const": "vehicle"},
            "loop_policy": {"const": "single_pass_route_waypoints"},
            "min_visible_motion_ratio": _float_leaf(),
            "min_xy_span_m": _float_leaf(),
            "physical_lane_lateral_m": _float_leaf(),
            "planned_path_length_m": _float_leaf(),
            "policy": {"const": "continuous_capture_ground_flow_v1"},
            "required": {"type": "boolean"},
            "route_duration_ticks": {"type": "integer", "minimum": 0},
            "route_source": {"const": "sumo_net_directed_lane_geometry"},
            "speed_mps": _float_leaf(),
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
            "route_duration_ticks": {"type": "integer", "minimum": 0},
            "route_source": {"const": "traffic_bundle_sidewalk"},
            "speed_mps": _float_leaf(),
        },
    }


def _uav_global_flow_payload() -> dict:
    # convert_to_render_ready.py (writer) over the flow planner's mission
    # table; the enums are the observed planner membership.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "altitude_layer_m", "corridor_family", "corridor_id",
            "ground_reference_z_m", "mission_type", "origin_pad_id",
            "policy", "sample_period_s", "semantic_role", "target_cell_id",
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
            # endpoint.
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
    # convert_to_render_ready.py (writer) over the pad network table.
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
            "role": {
                "enum": ["hub_charging_logistics",
                         "satellite_charging_logistics"],
            },
        },
    }


def _placement() -> dict:
    # convert_to_render_ready.py repaired writer (:1365-1376) and the
    # unrepairable path (:1347-1360) — the second arm has no verified L4
    # roster instance this session.
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
        # Still-without-instance-verification: the unrepairable writer
        # path (:1347-1360).
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
    # Repaired dict: convert_to_render_ready.py :1304-1324; unrepaired
    # dict (:1325-1346 of the same pass) — no verified L4 instance of the
    # second arm this session.
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
        # Still-without-instance-verification.
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


def _case_exact_core(member: str) -> str:
    parts = []
    for character in member:
        if "a" <= character <= "z" or "A" <= character <= "Z":
            parts.append(f"[{character.lower()}{character.upper()}]")
        else:
            parts.append(re.escape(character))
    return "".join(parts)


def _runtime_state_string_rule(members: tuple[str, ...]) -> dict:
    cores = "|".join(_case_exact_core(member) for member in members)
    return {"type": "string", "pattern": f"^\\s*({cores})\\s*$"}


def _family_enum_members(family: str) -> tuple[str, ...]:
    for (enum_family, _), members in RUNTIME_STATE_ENUM_FIELDS.items():
        if enum_family == family:
            return tuple(sorted(members))
    raise ValueError(f"runtime state family has no governed enum set: {family}")


def _runtime_state_field_rule(family: str, field: str) -> dict:
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return {"type": "string", "pattern": "\\S"}
    if family == "incident_state" and field in {
        "hazard_concentration_ppm", "hazard_radius_m",
    }:
        # runtime_state_contract accepts nonnegative int/float for both.
        return {"type": "number", "minimum": 0}
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
    }


def _core_properties() -> dict:
    return {
        # The site/roi pair the L4 scene files declare; kept as
        # identifiers because the producer copies the authored config
        # value.
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
        # Authored tag list; length not pinned.
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


def _sumo_vehicle_row() -> dict:
    # build_sumo_vehicle_roster_entries (:4250-4406); background_role is
    # the converter ternary (:4293-4299): sumo_semantic_event_vehicle /
    # sumo_source_required_context_vehicle / sumo_background_traffic.
    return _branch(
        {
            "background_role": {
                "enum": [
                    "sumo_background_traffic",
                    "sumo_semantic_event_vehicle",
                    "sumo_source_required_context_vehicle",
                ],
            },
            "background_vehicle": _background_vehicle(),
            "control_state": _runtime_state_family("control_state"),
            "ground_flow_contract": _sumo_ground_flow_contract(),
            "mode": {"const": "scene_sync"},
            "sensor_state": _runtime_state_family("sensor_state"),
            "source": {"const": "sumo_traci"},
            "sumo_segment": _segment(_SUMO_PHASE),
            "sumo_vehicle": _sumo_vehicle(),
            "sumo_vehicle_id": _identifier(),
            "sumo_visibility": {
                "type": "object", "additionalProperties": False,
                "required": ["frames_seen_in_segment",
                             "min_observation_distance_m"],
                "properties": {
                    "frames_seen_in_segment": {
                        "type": "integer", "minimum": 0,
                    },
                    "min_observation_distance_m": _float_leaf(),
                },
            },
        },
        [
            "background_role", "background_vehicle", "ground_flow_contract",
            "mode", "source", "sumo_segment", "sumo_vehicle",
            "sumo_vehicle_id", "sumo_visibility",
        ],
    )


def _event_actor_entry_contract() -> dict:
    # stage_event_actor (:1411-1432) writes the 6 base keys,
    # event_actor_entry (:1482-1492) appends the 7 entry keys, and the
    # boundary annotation pass adds entry_capture_boundary_clearance_m
    # (EVENT_ACTOR_ENTRY_BOUNDARY_CLEARANCE_M :288) — the 14-key form
    # observed on both L4-10_v1 event actors (spec :42-55, :87-100).
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "arrival_tick", "duration_model", "entry_capture_boundary_clearance_m",
            "entry_event_id", "entry_speed_mps", "entry_tick",
            "minimum_visible_motion_ratio_before_event", "policy",
            "pre_activation_presence", "route_length_m", "schema",
            "semantic_event_id", "stationary_preplacement_allowed",
            "yaw_authority",
        ],
        "properties": {
            "arrival_tick": {"type": "integer", "minimum": 0},
            "duration_model": {
                "const": "sum_segment_ceil_distance_over_speed_ticks",
            },
            "entry_capture_boundary_clearance_m": _float_leaf(),
            "entry_event_id": _identifier(),
            "entry_speed_mps": _float_leaf(),
            "entry_tick": {"type": "integer", "minimum": 0},
            "minimum_visible_motion_ratio_before_event": _float_leaf(),
            "policy": {"const": "offstage_then_continuous_lane_entry"},
            "pre_activation_presence": {"const": "absent"},
            "route_length_m": _float_leaf(),
            "schema": {"const": "event_actor_motion_contract_v1"},
            "semantic_event_id": _identifier(),
            "stationary_preplacement_allowed": {"type": "boolean"},
            "yaw_authority": {"const": "scene_lane_then_velocity"},
        },
    }


def _event_actor_dispatch_contract() -> dict:
    # spawn_event_actor_at_dispatch (:1516-1529) stops at the 6-key base
    # form written by stage_event_actor (:1425-1432) with the dispatch
    # policy.  Still-without-instance-verification: no verified L4 roster
    # record carries this variant this session.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "policy", "pre_activation_presence", "schema",
            "semantic_event_id", "stationary_preplacement_allowed",
            "yaw_authority",
        ],
        "properties": {
            "policy": {"const": "offstage_until_semantic_dispatch"},
            "pre_activation_presence": {"const": "absent"},
            "schema": {"const": "event_actor_motion_contract_v1"},
            "semantic_event_id": _identifier(),
            "stationary_preplacement_allowed": {"type": "boolean"},
            "yaw_authority": {"const": "scene_lane_then_velocity"},
        },
    }


def _event_actor_vehicle_row() -> dict:
    # New L4 family: scripted event actors keep their scene identity
    # (is_script_controlled_vehicle excludes spawn_policy
    # "event_script_only" from SUMO, explicit_vehicle_plan.py :284-310).
    # The hop-1 preserve list (:215-256) carries role/semantic_role/
    # state_sequence/task_id from initial_state, the contract,
    # spawn_policy, activation_tick, route_waypoints_enu_m and the
    # vehicle_state family; it does NOT carry motion_contract or
    # capture_boundary/capture_contract.  ``enabled`` stays scene-level.
    def _arm(contract: dict, required_extra: list) -> dict:
        return _branch(
            {
                "activation_tick": {"type": "integer", "minimum": 0},
                "category": {"const": "vehicle"},
                "control_state": _runtime_state_family("control_state"),
                "deactivation_tick": {"type": "integer", "minimum": 0},
                "event_actor_motion_contract": contract,
                "mode": {"const": "scene_sync"},
                "role": _identifier(),
                "route_waypoints_enu_m": _waypoints(),
                "runtime_visibility": _runtime_visibility(),
                "semantic_role": _identifier(),
                "sensor_state": _runtime_state_family("sensor_state"),
                "spawn_policy": {"const": "event_script_only"},
                "state_sequence": {"type": "array", "items": _identifier()},
                "task_id": _identifier(),
                "vehicle_state": _runtime_state_family("vehicle_state"),
            },
            [
                "activation_tick", "category", "event_actor_motion_contract",
                "mode", "role", "route_waypoints_enu_m", "runtime_visibility",
                "semantic_role", "spawn_policy", "state_sequence", "task_id",
            ] + required_extra,
        )

    return {"oneOf": [
        _arm(_event_actor_entry_contract(), []),
        # Still-without-instance-verification: the at-dispatch variant.
        _arm(_event_actor_dispatch_contract(), []),
    ]}


def _semantic_background_vehicle_row() -> dict:
    # Authored background vehicles that keep their scene identity (the
    # converter replaces only explicit_source_vehicle_ids rows,
    # :4990-4996): background_role/background_vehicle/ground_flow_contract
    # from the preserve list, role/semantic_role/state_sequence/task_id
    # from initial_state, no source/sumo_* keys, no motion_contract.
    return _branch(
        {
            "activation_tick": {"type": "integer", "minimum": 0},
            "background_role": {"const": "semantic_context"},
            "background_vehicle": _authored_background_vehicle(),
            "category": {"const": "vehicle"},
            "deactivation_tick": {"type": "integer", "minimum": 0},
            "ground_flow_contract": _ground_vehicle_continuous_flow_contract(),
            "mode": {"const": "scene_sync"},
            "role": _identifier(),
            "route_waypoints_enu_m": _waypoints(),
            "runtime_visibility": _runtime_visibility(),
            "semantic_role": _identifier(),
            "state_sequence": {"type": "array", "items": _identifier()},
            "task_id": _identifier(),
            "vehicle_state": _runtime_state_family("vehicle_state"),
        },
        [
            "activation_tick", "background_role", "background_vehicle",
            "category", "ground_flow_contract", "mode", "role",
            "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
            "state_sequence", "task_id",
        ],
    )


def _global_flow_uav_row() -> dict:
    return _branch(
        {
            "background_role": {"const": "donghu_global_uav_flow"},
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
    # convert_to_render_ready.py :1273-1301 writes the repair pair onto the
    # roster entry it re-anchors (L4-4_v2 pads 13, L4-6_v1 pads 01/14).  The
    # unrepaired pads keep the same keyset without those two keys, so both
    # stay optional here exactly as on _repaired_facility_row.
    return _branch(
        {
            "background_role": {"const": "global_uav_pad"},
            "mode": {"const": "scene_sync"},
            "motion_contract": _pad_motion_contract(),
            "placement": _placement(),
            "scene_occupancy_repair": _scene_occupancy_repair(),
            "semantic_scope": _semantic_scope(),
            "source": {"const": "uav_global_flow"},
            "uav_global_pad": _uav_global_pad(),
            "uav_pad_id": _identifier(),
        },
        [
            "background_role", "mode", "motion_contract", "semantic_scope",
            "source", "uav_global_pad", "uav_pad_id",
        ],
    )


def _scoped_uav_properties() -> dict:
    return {
        "activation_tick": {"type": "integer", "minimum": 0},
        "assigned_altitude_m": _float_leaf(),
        "category": {"const": "uav"},
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
        "incident_state": _runtime_state_family("incident_state"),
        "mode": {"const": "scene_sync"},
        "mission_state": _runtime_state_family("mission_state"),
        "planned_route_waypoints_enu_m": _waypoints(),
        "role": _identifier(),
        "route_waypoints_enu_m": _waypoints(),
        "runtime_visibility": _runtime_visibility(),
        "semantic_role": _identifier(),
        "sensor_state": _runtime_state_family("sensor_state"),
        "state_sequence": {"type": "array", "items": _identifier()},
        "task_id": _identifier(),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": _identifier(),
    }


_SCOPED_UAV_REQUIRED = [
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
            "inspect_altitude_code": _identifier(),
            "inspect_altitude_m": _float_leaf(),
            "inspect_fov_coverage_required": {"type": "boolean"},
            "min_path_length_m": _float_leaf(),
        },
        _SCOPED_UAV_REQUIRED + [
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
        _SCOPED_UAV_REQUIRED + ["lifecycle"],
    )


def _observer_uav_row() -> dict:
    return _branch(
        {
            **_scoped_uav_properties(),
            "observer_lifecycle": _observer_lifecycle(),
        },
        _SCOPED_UAV_REQUIRED + ["observer_lifecycle"],
    )


def _facility_properties() -> dict:
    return {
        "activation_tick": {"type": "integer", "minimum": 0},
        "category": {"const": "facility"},
        # The converter copies the authored state list; mode comes from
        # ENTITY_PROFILES (:190-294): scene_sync on visible facilities,
        # metadata_only on the no-fly/hazard rows.
        "mode": {"enum": ["metadata_only", "scene_sync"]},
        "role": _identifier(),
        "route_waypoints_enu_m": _waypoints(),
        "runtime_visibility": _runtime_visibility(),
        "semantic_role": _identifier(),
        "state_sequence": {"type": "array", "items": _identifier()},
        "task_id": _identifier(),
    }


_FACILITY_REQUIRED = [
    "activation_tick", "category", "mode", "role",
    "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
    "state_sequence", "task_id",
]


def _semantic_facility_row() -> dict:
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


def _logical_facade_anchor_row() -> dict:
    # profile_for_entity uses logical_only_profile for semantic.asset_anchor
    # and retains the authored facade_anchor category (L4-2_v1/v2).
    return _branch(
        {
            "asset_id": {"const": "semantic.asset_anchor"},
            "entity_category": {"const": "facade_anchor"},
            "entity_kind": {"const": "facade_anchor.other"},
            "entity_type": {"const": "facade_anchor.other"},
            "label_class": {"const": "other"},
            "logical_asset_id": {"const": "semantic.asset_anchor"},
            "proxy_template_id": {"const": ""},
            "activation_tick": {"type": "integer", "minimum": 0},
            "category": {"const": "facade_anchor"},
            "mode": {"const": "logical_contract"},
            "role": {"const": "semantic_context"},
            "route_waypoints_enu_m": _waypoints(),
            "runtime_visibility": _runtime_visibility(),
            "semantic_role": _identifier(),
            "state_sequence": {"type": "array", "items": _identifier()},
            "task_id": _identifier(),
        },
        [
            "activation_tick", "category", "mode", "role",
            "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
            "state_sequence", "task_id",
        ],
    )


def _logical_crowd_anchor_row() -> dict:
    # profile_for_entity selects logical_only_profile for semantic.spawn_zone
    # and keeps the authored crowd_anchor category (L4-8_v1/v2), the same
    # logical-only shape as the facade anchor above.  Built from fresh leaves
    # only: no shared schema object is reused or mutated.
    return _branch(
        {
            "asset_id": {"const": "semantic.spawn_zone"},
            "entity_category": {"const": "crowd_anchor"},
            "entity_kind": {"const": "crowd_anchor.no_fly_zone"},
            "entity_type": {"const": "crowd_anchor.no_fly_zone"},
            "label_class": {"const": "no_fly_zone"},
            "logical_asset_id": {"const": "semantic.spawn_zone"},
            "proxy_template_id": {"const": ""},
            "activation_tick": {"type": "integer", "minimum": 0},
            "category": {"const": "crowd_anchor"},
            "mode": {"const": "logical_contract"},
            "role": {"const": "semantic_context"},
            "route_waypoints_enu_m": _waypoints(),
            "runtime_visibility": _runtime_visibility(),
            "semantic_role": _identifier(),
            "state_sequence": {"type": "array", "items": _identifier()},
            "task_id": _identifier(),
        },
        [
            "activation_tick", "category", "mode", "role",
            "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
            "state_sequence", "task_id",
        ],
    )


def _pedestrian_properties(role: str) -> dict:
    return {
        "activation_tick": {"type": "integer", "minimum": 0},
        "category": {"const": "pedestrian"},
        "mode": {"const": "pedestrian_managed"},
        "role": {"const": role},
        "route_waypoints_enu_m": _waypoints(),
        "runtime_visibility": _runtime_visibility(),
        "semantic_role": _identifier(),
        "state_sequence": {"type": "array", "items": _identifier()},
        "task_id": _identifier(),
    }


_PEDESTRIAN_REQUIRED = [
    "activation_tick", "category", "mode", "role",
    "route_waypoints_enu_m", "runtime_visibility", "semantic_role",
    "state_sequence", "task_id",
]


def _pedestrian_row() -> dict:
    # The background contract is authored at L4-3_v1 and L4-10_v1;
    # background_role is preserved only when the author supplied it.
    # The contract keeps route_source const traffic_bundle_sidewalk.
    return _branch(
        {
            **_pedestrian_properties("semantic_background_pedestrian"),
            "background_pedestrian": {
                "type": "object", "additionalProperties": False,
                "required": ["contract_scenario_id", "policy",
                             "semantic_role"],
                "properties": {
                    "contract_scenario_id": _identifier(),
                    "policy": {"const": "semantic_event_contract_v1"},
                    "semantic_role": _identifier(),
                },
            },
            "background_role": {"const": "semantic_context"},
            "contract_scenario_id": _identifier(),
            "ground_flow_contract": _pedestrian_ground_flow_contract(),
        },
        _PEDESTRIAN_REQUIRED + [
            "background_pedestrian", "contract_scenario_id",
            "ground_flow_contract",
        ],
    )


def _ordinary_pedestrian_row() -> dict:
    # L4-3_v1 authored cohort members 01..05 and landing actors a/b carry
    # the common pedestrian fields without a background contract.  The
    # L4-7_v1/v2 fallen actor adds an incident_state dispatch flag alongside
    # its pedestrian_state; both families stay optional so the plain cohort
    # and landing actors keep matching this one branch.
    return _branch(
        {
            **_pedestrian_properties("semantic_pedestrian"),
            "incident_state": _runtime_state_family("incident_state"),
            "pedestrian_state": _runtime_state_family("pedestrian_state"),
        },
        _PEDESTRIAN_REQUIRED,
    )


def _corridor_properties() -> dict:
    return {
        "activation_tick": {"type": "integer", "minimum": 0},
        "category": {"const": "airspace_corridor"},
        "mode": {"const": "metadata_only"},
        "runtime_visibility": _runtime_visibility(),
    }


def _corridor_row() -> dict:
    return _branch(_corridor_properties(),
                   ["activation_tick", "category", "mode",
                    "runtime_visibility"])


def _shared_corridor_row() -> dict:
    return _branch(
        {**_corridor_properties(), "semantic_role": _identifier()},
        ["activation_tick", "category", "mode", "runtime_visibility",
         "semantic_role"],
    )


def _formal_roster_entity_schema() -> dict:
    """Return the closed item schema for one L4 roster entity."""
    return {
        "oneOf": [
            _sumo_vehicle_row(),
            _event_actor_vehicle_row(),
            _semantic_background_vehicle_row(),
            _global_flow_uav_row(),
            _global_pad_row(),
            _inspect_uav_row(),
            _mission_uav_row(),
            _observer_uav_row(),
            _semantic_facility_row(),
            _contract_facility_row(),
            _repaired_facility_row(),
            _logical_facade_anchor_row(),
            _logical_crowd_anchor_row(),
            _pedestrian_row(),
            _ordinary_pedestrian_row(),
            _corridor_row(),
            _shared_corridor_row(),
        ],
    }


def formal_roster_schema() -> dict:
    """Return the closed JSON-root schema for L4 ``formal_roster``.

    The entity-item ``oneOf`` splits the record space the way the producer chain
    writes it at the L4 layer: one branch per keyset family (see the
    module docstring for the producer line references and the verified
    keysets).  Because every branch closes with
    ``additionalProperties: False`` and declares exactly its keyset, the
    marker keys make the branches pairwise exclusive — a row stripped of
    its branch marker or carrying a foreign payload key no longer
    validates.  The JSON root itself permits exactly the ``entities`` key.
    The per-file census and the pairwise sweep over all 72 declaration
    files are the host validation step.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_roster_l4.record.schema.json",
        "title": "AERO_WORLD formal_roster L4 JSON root record schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["entities"],
        "properties": {
            "entities": {
                "type": "array",
                "items": _formal_roster_entity_schema(),
            },
        },
    }
