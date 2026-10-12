"""Closed whole-JSON record structure for a formal L1 entity roster.

Source family ``formal_roster``; profiles L1-1_v1, L1-1_v2, L1-2_v1,
L1-3_v1, L1-3_v2, L1-4_v1, L1-4_v2.  Producer chain:
``Dataset/tools/convert_to_render_ready.py`` reads the source roster
(``read_source_roster`` :1549) and writes the frozen per-episode
``global_entity_roster.json`` (``write_json`` :5795);
``Dataset/tools/filter_render_ready_truth_for_capture.py`` copies roster
episodes through to the transfer boundary unchanged (:679-681, :781-786,
:821; the ``sumo_visibility`` slot stays in its preserve list :423), so
every roster byte at ``aw_data/render_ready_episodes_capture_filtered`` is
the converter's output.

Record identity: ``structure_sources.py`` lists ``formal_roster`` in
``_ENTITY_FILES`` (:34); ``iter_source_records`` yields the whole JSON root
per file.  The grammar validates that root before field projection.  Its
only root key is ``entities``; each array member uses the closed entity
branch schema below.

Entity items use closed ``oneOf`` branches for SUMO vehicles, global-flow
UAVs, pads, scoped UAVs, facilities, pedestrians and corridors.  Branch
markers include ``sumo_vehicle_id``, ``uav_id``, ``uav_pad_id``,
``contract_inspect_uav``, ``lifecycle``, ``observer_lifecycle``,
``background_pedestrian``, ``contract_facility``,
``scene_occupancy_repair`` and corridor ``semantic_role``.  The branch
objects reject foreign payload keys.

Producer evidence per branch:
- Sumo vehicle rows: ``build_sumo_vehicle_roster_entries`` (:4250) writes
  the entry at :4316-4394 — ``entity_id``/``sumo_vehicle_id`` from
  ``sumo_truth_entity_id``/``sumo_render_entity_id``
  (``Dataset/tools/sumo_ground_flow/truth_integration.py`` :66-77;
  ``Dataset/tools/sumo_ground_flow/explicit_vehicle_plan.py`` :2865 builds
  ``traffic_bg_<scenario>_<seed>_slotNN_NNN`` ids), ``background_vehicle``
  (:4337), ``ground_flow_contract`` (:4348, ``route_source``
  ``sumo_traci_fcd_projected_to_ue_truth``, ``sample_period_s`` from the
  sumo manifest), the shared ``sumo_segment`` payload, the 39-key
  ``sumo_vehicle`` block (:4336-4385) whose 16 road-authority keys come
  verbatim from ``Dataset/tools/sumo_ground_flow/road_signal_context.py::
  static_sumo_vehicle_fields`` (:189-254 — the lane/speed/following
  authority ids and refs are built from the net XML and are never empty;
  ``controlling_signal_id`` is the TLS id or the const ``"none"``
  (:29, :223) and ``controlling_signal_state`` the const
  ``not_sampled_at_roster_level`` (:228), ``crossed_stop_line`` the const
  ``False`` (:253); ``controlling_signal_ontology_class_id`` is
  ``world:TrafficSignal`` or null (:225-227) and the ``stop_line_*`` /
  ``right_of_way_*`` triplets are populated or null exactly when the lane
  has a signal connection (:229-252)) and ``sumo_visibility``
  (:4386-4394).  ``runtime_state_fields_from_sources`` merges source
  runtime state after the block (:4395-4406, :4556-4559), so the
  ``control_state``/``sensor_state`` families are optional members of this
  branch; the corpus writes none on these rows.
- Global-flow UAV rows: ``build_uav_roster_entries`` (:3299, entry
  :3356-3407) with ``uav_truth_entity_id = global_uav_<uav_id>``
  (``Dataset/tools/uav_global_flow/truth_integration.py`` :58-73),
  ``uav_global_flow`` payload (:3379), ``motion_contract`` (:3394),
  ``uav_segment`` (:3405) and ``ground_reference_z_m``/``source`` at
  :3406-3407.
- Global pad rows: ``build_uav_pad_roster_entries`` (:3208, entry
  :3222-3262) — ``uav_pad_truth_entity_id = global_<pad_id>``,
  ``background_role: "global_uav_pad"`` (:3243), the eight-key
  ``uav_global_pad`` block (:3244) and the static-facility
  ``motion_contract`` (:3254); ``semantic_scope`` comes from
  ``Dataset/tools/uav_global_flow/generate_uav_flow.py`` :429-432 through
  ``Dataset/semantic_truth/facility_scope.py::semantic_scope_for_source``
  (:95-125, dict at :150).
- Scoped UAV rows (inspect/mission/observer): the scene entity's preserved
  contract fields reach the roster verbatim —
  ``convert_to_render_ready.PRESERVED_ENTITY_FIELDS`` (:397-441) and
  ``preserved_fields_from`` (:1476-1530, which copies every present
  non-``None``/non-"" field including nested contract keys), with
  ``activation_tick`` written by ``Dataset/tools/batch_generate.py``
  (:2562; contract fields at ~:220-254).  ``contract_inspect_uav`` (the
  semantic-event contract the L1 scene files author), ``lifecycle``,
  ``observer_lifecycle`` and ``uav_corridor`` keep the authored keysets;
  ``state_sequence`` is the authored state list.
- Facility rows: semantic facilities carry the semantic-event keys plus
  ``semantic_scope``; contract facilities add ``contract_facility`` and a
  top-level ``contract_scenario_id`` (``_add_contract_facility``,
  ``Dataset/tools/regenerate_boundary_scenarios.py``); repaired facilities add ``placement``
  and ``scene_occupancy_repair`` written by the reanchor pass
  (``convert_to_render_ready.py`` :1295-1380): the repaired variant sets
  ``placement_semantics: "service_facility_repair"`` with the four anchor
  numbers (:1365-1376) and the repair dict at :1304-1324, while the
  unrepairable path writes ``placement_semantics:
  "service_facility_nonblocking_unrepaired"`` plus ``blocking``/
  ``collision_policy`` (:1347-1360) and a repair dict with
  ``status``/``reason`` (:1325-1346) — no L1 instance of that variant
  exists, so both ``placement`` and ``scene_occupancy_repair`` declare it
  as a second ``oneOf`` arm marked still-without-instance-verification.
  The repair dict also has a distinct successful arm with
  ``fallback: "open_space_radial_search"`` when the producer's radial
  search succeeds; that arm has no verified L1 instance.
  The six repaired landing pads carry no ``contract_facility``/
  ``contract_scenario_id``; the six repaired radio towers do, so those
  two keys stay optional on the repaired branch.
- Pedestrian rows: the pedestrian flow writer emits
  ``background_pedestrian`` (``policy: "semantic_event_contract_v1"`` plus
  the authored ``semantic_role``/``contract_scenario_id``) and a
  ``ground_flow_contract`` with the
  ``continuous_capture_ground_flow_v1`` members; the segment payload is
  shared below.
- Corridor rows: ``category: "airspace_corridor"`` plus
  ``activation_tick`` and ``runtime_visibility``
  (``runtime_visibility_payload`` :3004, written per entity at :5107-5113);
  the six shared cells add ``semantic_role``.

Shared payload closures:
- ``_segment``: ``seed_index``/``seed_label`` (``seed00|seed01|seed02``)/
  ``segment_start_s``/``segment_end_s``/``duration_s``/``phase_name`` —
  sumo rows const ``episode_local_vehicle_authority_after_warmup``; UAV
  rows enum the three ``uav_demand_*`` phases (772/153 observed).
- ``runtime_visibility``: the five-key crop payload with its policy const;
  417 instances, identical keyset.
- ``semantic_scope``: ``scope_type: "facility"``, the five
  ``scope_subtype``/``ontology_class_id`` pairs, ``authority_id:
  "facility_scope_contract.v1"``, optional ``service_capacity`` (75 with,
  33 without).
- ``uav_corridor``: ``altitude_layers_m`` is a producer-typed float array
  (length 3 in all 69 instances, not pinned), plus
  ``assigned_altitude_m``/``used_lateral_bypass``.
- ``contract_inspect_uav``: the 17-key semantic-event contract; the three
  route arrays (``loop_route_enu_m``/``repaired_route_enu_m``/
  ``planned_route_enu_m``) carry typed waypoints.  The checked seed00
  roots contain nonempty routes.
- ``sumo_vehicle.semantic_metadata``: two producer branches — the 2-key
  identity form (497 rows) and the 27-key background form (275 rows,
  ``explicit_vehicle_plan.py``) with the semantic slots the planner leaves
  unwritten at roster level (``None``/""/empty list are the declared
  structural statuses; ``traffic_slot_index`` is ``int|null``).
- ``vehicle_type`` (both in ``sumo_vehicle`` and ``background_vehicle``):
  ``aero_passenger|aero_emergency`` bare, or the same prefix joined with
  ``@`` and the SUMO vehicle id (760/12 prefixes, 467 joined forms).
- ``initial_position_enu_m`` is the float vec3 and ``initial_yaw_deg`` the
  float yaw the converter rounds (:4334); ``tags`` is the authored tag
  list (2-4 entries in the corpus); id fields stay
  nonempty-string identifiers — the documented id shapes (``bg_vehicle_*``,
  ``bg_ped_*``, ``corridor_*``, ``global_uav_*``, ``global_<pad>``,
  ``uav_l<n>_v<k>[_<role>]``, ``contract_facility_*``, ``nfz_*``) are
  producer evidence, not grammar patterns; no id regex is pinned.
- ``mode`` couples with the branch: ``scene_sync`` on all presence-bearing
  rows, ``metadata_only`` on corridors and the no-fly/hazard facility
  rows, ``pedestrian_managed`` on pedestrians.
- Runtime-state families: the roster merges source runtime state only
  through ``runtime_state_fields_from_sources`` (sumo entries :4395-4406/
  :4556-4559) and the preserved scene fields of the scoped UAVs; the L1
  scenes author only ``control_state`` booleans/enum members and
  ``sensor_state.intruder_detected``, so exactly those two families are
  declared, closed to the full executable producer contract in
  ``Dataset/tools/runtime_state_contract.py`` (booleans, enum members via
  the raw-value rule, command members, metadata strings; ``minProperties:
  1`` instead of enumerating observed keysets) exactly as
  ``arm_states_l1_structure.py`` closes them.  The other eight families
  have no L1 roster payload — not declared here.

Cross-family reuse for ``arm_branch_roster`` requires a separate source
check in G1/G2; this module declares only ``formal_roster`` L1.

Publication exclusion: roster rows carry no digest/integrity/hash field
(``is_publication_excluded_path`` has no roster-family rule and no roster
key matches), so this module declares no excluded path.  Top-level
``metadata`` and per-entity metadata blocks likewise do not exist on any
observed row.

Boundary: this module exports the pure schema constructor only.  It is not
registered in ``structure_closure.py`` (``formal_roster`` is absent from
``CLOSED_RECORD_FAMILIES`` :159 and from the ``validate_source_record``
dispatch :203); registration/dispatch wiring belongs to the integration
stage.  The module scans no data at import and compiles nothing at import.
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
    # (observed empty on the 99 facility rows).
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


def _vehicle_type() -> dict:
    # explicit_vehicle_plan writes the bare profile type or
    # "<type>@<sumo vehicle id>"; the joined half is a SUMO id.
    return {
        "type": "string",
        "pattern": "^(aero_passenger|aero_emergency)(@[^@\\s]+)?$",
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
    # The L1 scene-authored semantic-event contract; the three route arrays
    # retain their producer-typed waypoints, including nonempty routes.
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
                "direction_role": {
                    "enum": ["cross", "inbound", "outbound"],
                },
                "expected_core_entry_tick": {"type": "integer"},
                "expected_core_exit_tick": {"type": "integer"},
                "explicit_vehicle_plan": {"type": "boolean"},
                "logical_asset_authority": {
                    "const": "explicit_vehicle_plan_type",
                },
                "logical_asset_id": {"const": "vehicle.ground.boxcar.v1"},
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
                "enum": ["vehicle.emergency.suv.v1",
                         "vehicle.ground.boxcar.v1"],
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
    # convert_to_render_ready.py :4337-4348.
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
    # convert_to_render_ready.py :3379 (writer) over the flow planner's
    # mission table; the enums are the observed planner membership.
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
            # endpoint (18/27 rows without).
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
    # convert_to_render_ready.py :3244 (writer) over the pad network table.
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
    # convert_to_render_ready.py :1365-1376 (repaired, 12 instances) and
    # :1347-1360 (unrepairable path, no L1 instance).
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
        # Still-without-instance-verification: the unrepairable writer path
        # (:1347-1360), no L1 roster row carries it.
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
    # The reanchor and radial-search success paths share eleven fields.
    # The latter adds one fixed fallback marker. The unrepaired path has
    # status/reason instead; neither of those two paths has a verified L1
    # instance.
    anchored = {
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
    }
    radial = {
        **anchored,
        "required": [*anchored["required"], "fallback"],
        "properties": {
            **anchored["properties"],
            "fallback": {"const": "open_space_radial_search"},
        },
    }
    return {"oneOf": [
        anchored,
        radial,
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


def _runtime_state_string_rule(members: tuple[str, ...]) -> dict:
    """Raw-string rule for one enum or command value of the executable
    contract, delegating to the audited shared fragment.

    ``invalid_runtime_state_value_paths`` (runtime_state_contract.py:549-553,
    :576) validates enum and command members as ``value.strip().lower() in
    allowed`` while the written row keeps the raw string.  A JSON Schema
    ``enum`` of the canonical members would reject the non-canonical raw
    spellings the producer accepts, and JSON Schema patterns are ECMA-262,
    which has no case-insensitive flag — ``(?i:...)`` is a Python-only
    construct.  The fragment returns an anchored frame over an explicit
    Python-``strip()`` whitespace class around per-member ASCII case cores
    plus the Kelvin sign U+212A for the letter ``k``: the sole non-ASCII code
    point whose ``str.lower()`` is an ASCII letter on the pinned Unicode
    13.0.0 runtime, which the earlier ``\\s``-anchored local rule both
    mis-cased (per-member ``[aA]`` cores were right, but ``\\s`` diverges
    from ``str.strip()`` across engines) and left open to the ECMA/Python
    whitespace disagreement.  It keeps whitespace-only, blank,
    out-of-membership and internally-spaced values rejected, matching the
    producer's rejection of ``"   "``.

    Remaining integration dependency (owned by the producer contract, not by
    this grammar): any future Unicode-table change other than U+212A.  The
    fragment raises on a changed ``unicodedata.unidata_version`` so the rule
    cannot silently drift.
    """
    return runtime_enum_string_rule(members)


def _family_enum_members(family: str) -> tuple[str, ...]:
    for (enum_family, _), members in RUNTIME_STATE_ENUM_FIELDS.items():
        if enum_family == family:
            return tuple(sorted(members))
    raise ValueError(f"runtime state family has no governed enum set: {family}")


def _runtime_state_field_rule(family: str, field: str) -> dict:
    if field in RUNTIME_STATE_METADATA_FIELDS:
        # The producer rejects values that are blank after strip()
        # (runtime_state_contract.py:541-543); the raw value is written back,
        # so the accepted raw string is "non-empty after stripping".  The
        # shared fragment's ``nonblank_python_string_rule`` names Python
        # ``str.strip()``'s whitespace set exactly (unlike ``\\S`` under a
        # strict ECMA-262 engine, which would additionally reject U+FEFF);
        # ``minLength: 1`` would accept whitespace-only strings the producer
        # rejects.
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
    }


def _core_properties() -> dict:
    return {
        # The single site/roi pair the seven L1 scene files declare
        # (``site.intersection_a`` / ``roi.intersection_a.v1``); kept as
        # identifiers because the producer copies the authored config value.
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
        # Authored tag list; 2-4 entries in the corpus, length not pinned.
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
    return _branch(
        {
            "background_role": {"const": "sumo_background_traffic"},
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
        "mode": {"const": "scene_sync"},
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
        # The converter copies the authored state list; facilities author
        # ("available",) / ("online",) / ("idle",) in the corpus.
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


def _pedestrian_row() -> dict:
    return _branch(
        {
            "activation_tick": {"type": "integer", "minimum": 0},
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
            "category": {"const": "pedestrian"},
            "contract_scenario_id": _identifier(),
            "ground_flow_contract": _pedestrian_ground_flow_contract(),
            "mode": {"const": "pedestrian_managed"},
            "role": {"const": "semantic_background_pedestrian"},
            "route_waypoints_enu_m": _waypoints(),
            "runtime_visibility": _runtime_visibility(),
            "semantic_role": _identifier(),
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
    """Return the closed item schema for one L1 roster entity."""
    return {
        "oneOf": [
            _sumo_vehicle_row(),
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
            _shared_corridor_row(),
        ],
    }


def formal_roster_schema() -> dict:
    """Return the closed whole-JSON schema for an L1 formal roster."""
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_roster_l1.record.schema.json",
        "title": "AERO_WORLD formal_roster L1 JSON root record schema",
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
