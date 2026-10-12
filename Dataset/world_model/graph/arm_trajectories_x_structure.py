"""Closed record structure for X ARM raw trajectory rows.

Source family ``arm_trajectories``; profiles X1_rain_to_c2loss_to_forced_landing,
X2_gnss_spoof_to_geofence_violation, X3_pedestrian_fall_to_emergency_response,
X4_fog_to_uav_conflict, X5_comm_failure_to_pad_contention,
X6_crowd_evacuation_to_airspace_lockdown (``grp_cross_layer`` windows; 6
profiles x 3 seeds x 2-3 arms = 42 declared JSONL files, 342741 present rows).
Producer: ``Dataset/tools/x_arm_pipeline.py::run_arm`` replays the compiled
branch through ``Dataset/tools/l6_v2/engine.py::execute`` and writes every
engine row in ``[start_tick, end_tick]`` verbatim to
``raw/trajectories.jsonl`` (x_arm_pipeline.py:248, 251).  There is exactly one
record branch — the engine branch of ``EpisodeStateEngine._row_for_entity``
(``Dataset/tools/batch_generate.py:2395``): the fixed base keys
``tick/entity_id/label_class/asset_id/pos_enu/vel_mps/yaw_deg/state``,
``row_activity_payload``'s unconditional ``activity_type/animation_hint/
posture/social_state`` (batch_generate.py:1856), then
``preserved_fields_from(entity)`` (batch_generate.py:1737), which copies a
field only when its value is neither ``None`` nor ``""`` from the scene entity
or its nested ``initial_state``/``visual_state``.  Every X row therefore
carries the twelve always-written base keys, and every preserved key is a
conditional slot.

Entity roster: an X window records exactly the rows of
``config.scope.simulated_entity_ids`` — the branch scene minus the background
vehicles the pipeline relegates to immutable canonical records
(x_arm_pipeline.py:73-77, 84; all 42 configs verified against their written
rows).  ``label_class`` is ``label_for_declared_asset(logical_asset_id)``
(batch_generate.py:1524), so it closes to the eleven declared X assets:
``uav`` (uav.inspect.quad.v1, uav.airsim.cv_pawn.v1, uav.airsim.flying_pawn.v1),
``uav_corridor`` (semantic.uav_corridor.segment.v1), ``pedestrian``
(pedestrian.cityops.basic.v1), ``vehicle`` (vehicle.emergency.ambulance.v1 —
the other three X vehicle assets are excluded background), ``landing_pad``
(facility.landing_pad.visible.v1), ``radio_tower``
(facility.radio.base_tower.v1), ``charging_pile`` (facility.charger.cityops.v1),
``barrier`` (facility.barrier.basic), ``no_fly_zone`` (trigger.no_fly.box.v1),
``police_tape`` (prop.incident.police_tape.v1) and ``spawn_zone``
(semantic.spawn_zone).  All 342741 rows confirm the label set; the corridor
segments (118806 rows) carry only the base keys, and ``spawn_zone`` anchors
never carry ``role``/``route_waypoints_enu_m`` — both are declared shape
differences of the same engine branch, not separate record types.

Writer float casts: ``_row_for_entity`` writes ``pos_enu``/``vel_mps`` as
``[round(float(value), 6)]`` per component (batch_generate.py:2408-2409) and
``yaw_deg`` as ``round(float(...), 6)`` (batch_generate.py:2410); floats
serialize with no int fallback, so those leaves close to ``number`` with
``x-source-value-types: ["float"]`` — the corpus confirms float components on
all 342741 rows (1028223 pos + 1028223 vel members, 342741 yaw values).
``route_waypoints_enu_m``/``planned_route_waypoints_enu_m`` and the inspect
routes are verbatim scene pass-throughs whose authored members are all floats
(2766766 route members scanned: no int), so the pass-through arrays close to
three-float members with empty lists a declared status (136788 empty
``route_waypoints_enu_m`` rows).

Preserved slots on the 135 configured X entities (scene evidence over
``Dataset/scenarios/X_cross_layer/*/scene_setup.json``, confirmed row shape by
row shape): ``activation_tick/category/role/state_sequence/semantic_role/
task_id`` on every non-corridor entity; ``entity_kind/semantic_scope`` on
typed facilities; ``background_role/background_pedestrian`` on the stamped
crowd extras; ``uav_corridor/uav_corridor_role/assigned_altitude_m`` on
mission UAVs; ``lifecycle/ground_reference_z_m/ground_reference_source`` on
home-pad UAVs; ``observer_lifecycle`` on observation UAVs;
``contract_inspect_uav/inspect_altitude_code/inspect_altitude_m/
min_path_length_m/full_episode_presence`` on inspect UAVs; ``contract_facility`` on
contract-scoped facilities; ``ground_flow_contract/route_waypoints_enu_m`` on
moving ground actors; ``planned_route_waypoints_enu_m`` on corridor-planned
UAVs; ``spawn_policy`` on event actors.  ``boundary_crossing_injected`` and
``sensor_profile`` (inside ``contract_inspect_uav``) are scene-side only; the
engine never sees the first and the second rides inside its parent.

Repair decisions (this revision):
* Value-domain tightening to the executable producer contracts — the record
  grammar keeps raw written strings, but every vocabulary the producer itself
  enforces (pedestrian posture/social_state catalog, UAV corridor altitude
  layers, facility_scope authority, semantic-event policy stamps) is closed
  with ``enum``/``const`` so malformed rows fail at the structure layer
  instead of downstream.
* ``state``/``activity_type``/``animation_hint`` keep the writer's verbatim
  pass-through typing (``_state_label`` normalizes only pedestrian labels to
  the catalog; the UAV/facility states are scene-authored strings), while
  ``posture``/``social_state`` close to the catalog's value domains —
  ``row_activity_payload`` is the sole writer and the catalog is executable.
* ``route_waypoints_enu_m``/``planned_route_waypoints_enu_m`` and the inspect
  routes keep array member structure without a member-count bound: the corpus
  carries 136788 empty and multi-hundred-member routes, so only the triple
  shape is structural.
* Undeclared-but-producer-allowed slots: the ``security_state``/
  ``sensor_state``/``vehicle_state`` runtime families, the five preserved
  fields with no X row carrier in the 342741-row corpus scan
  (``background_vehicle``, ``contract_logical_sidecar``, ``task_kind``,
  ``task_state``, ``deactivation_tick``) and the top-level
  ``contract_scenario_id`` stamp stay outside the property set — declaring
  them would admit rows no X producer path writes.
  ``event_actor_motion_contract`` and ``path_deviation_contract`` ARE declared
  (they sit in ``PRESERVED_ENTITY_FIELDS``, so the producer admits them) and
  carry the 尚无实例验证 flag: no X scene, script or written row exercises
  them.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
)

__all__ = ["arm_trajectories_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# label_class values written by label_for_declared_asset over the eleven
# declared X logical assets (batch_generate.py:1521-1524).
_LABEL_CLASSES = (
    "barrier", "charging_pile", "landing_pad", "no_fly_zone", "pedestrian",
    "police_tape", "radio_tower", "spawn_zone", "uav", "uav_corridor",
    "vehicle",
)


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _writer_float3() -> dict:
    # ``_row_for_entity`` writes round(float(value), 6) per component
    # (batch_generate.py:2408-2410); float members serialize with no int
    # fallback, so the recorded source type is float on every component.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


def _number3() -> dict:
    # Verbatim scene pass-through member (route waypoints, inspect routes):
    # every authored X member is a float triple, but the pass-through keeps
    # the scene's positional triples without narrowing the JSON type further.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _waypoints(member: dict | None = None) -> dict:
    # Producer pass-through of the authored route; empty lists are a declared
    # structural status for this family (136788 rows carry one).
    return {"type": "array", "items": member or _number3()}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state patch field, from the executable
    producer contract (``validate_runtime_state_patch`` plus
    ``invalid_runtime_state_value_paths`` in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        # invalid_runtime_state_value_paths rejects blank/whitespace-only
        # metadata values ("\\S" scan), so the raw grammar keeps the same
        # non-blank shape.
        return {"type": "string", "pattern": r"\S"}
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        return {"oneOf": [
            {"type": "string", "minLength": 1},
            {"type": "array", "minItems": 1,
             "items": {"type": "string", "minLength": 1}},
        ]}
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list; only non-string members fail.
        # X rows carry ['uav_x5_priority'], ['uav_x5_priority',
        # 'uav_x5_second'] and the empty list.
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        return {"type": "string", "minLength": 1}
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        # Membership (case-insensitively) is checked by the producer contract;
        # the raw grammar keeps the written string.  X rows write
        # navigation_state.operational_state (spoofed/relocalizing/
        # geofence_alert/nominal) and pedestrian_state.posture/health.
        return {"type": "string", "minLength": 1}
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family_patch(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
    }


def _carried_runtime_state_families() -> dict:
    """Closed rule per executable family actually carried by X rows.

    The X producer admits any declared family (scene preservation plus
    ``set_runtime_state`` dispatch), so the seven carried families use the
    executable value contract; ``security_state``/``sensor_state``/
    ``vehicle_state`` have no X carrier and stay undeclared here.
    """
    carried = (
        "communication_state", "control_state", "facility_state",
        "incident_state", "mission_state", "navigation_state",
        "pedestrian_state",
    )
    return {family: _runtime_state_family_patch(family) for family in carried}


def _semantic_scope() -> dict:
    # Scene facility_scope contract.  service_capacity rides on the scoped
    # chargers and pads, not on the base stations, barriers and no-fly zones
    # (42153 five-key rows vs 17154 four-key rows), so it stays optional.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["authority_id", "ontology_class_id", "scope_subtype",
                     "scope_type"],
        "properties": {
            "authority_id": {"const": "facility_scope_contract.v1"},
            "ontology_class_id": _identifier(),
            "scope_subtype": _identifier(),
            "scope_type": {"const": "facility"},
            "service_capacity": {"type": "integer", "minimum": 0},
        },
    }


def _contract_facility() -> dict:
    # Two-key scene stamp copied verbatim onto contract-scoped facilities.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["policy", "contract_scenario_id"],
        "properties": {
            "policy": {"const": "semantic_event_contract_v1"},
            "contract_scenario_id": _identifier(),
        },
    }


def _background_pedestrian() -> dict:
    # Three-key scene stamp on every background pedestrian.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["policy", "semantic_role", "contract_scenario_id"],
        "properties": {
            "policy": {"const": "semantic_event_contract_v1"},
            "semantic_role": _identifier(),
            "contract_scenario_id": _identifier(),
        },
    }


def _ground_flow_contract() -> dict:
    # Continuous-capture ground-flow contract copied verbatim from the scene
    # (regenerate_boundary_scenarios._ground_flow_contract:3287): ten keys,
    # ``planned_path_length_m`` included on every contract the writer emits,
    # ``required`` hard-coded True (:3303).  ``physical_lane_lateral_m``
    # belongs to the excluded background vehicles and never reaches a row.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "actor_kind", "loop_policy", "min_visible_motion_ratio",
            "min_xy_span_m", "planned_path_length_m", "policy", "required",
            "route_duration_ticks", "route_source", "speed_mps",
        ],
        "properties": {
            "actor_kind": _identifier(),
            "loop_policy": {"const": "single_pass_route_waypoints"},
            "min_visible_motion_ratio": _float_leaf(),
            "min_xy_span_m": _float_leaf(),
            "planned_path_length_m": _float_leaf(),
            "policy": {"const": "continuous_capture_ground_flow_v1"},
            "required": {"const": True},
            "route_duration_ticks": {"type": "integer", "minimum": 0},
            "route_source": _identifier(),
            "speed_mps": _float_leaf(),
        },
    }


def _observer_lifecycle() -> dict:
    # Four-key observation-UAV lifecycle stamp
    # (regenerate_boundary_scenarios:4737).  The writer fixes three values
    # and takes ``capture_boundary_id`` from the scene's capture boundary.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["capture_boundary_id",
                     "maximum_terminal_boundary_distance_m", "presence",
                     "terminal_behavior"],
        "properties": {
            "capture_boundary_id": _identifier(),
            "maximum_terminal_boundary_distance_m": {"const": 18.0},
            "presence": {"const": "episode_full_duration"},
            "terminal_behavior": {
                "const": "airborne_hold_inside_runtime_boundary"},
        },
    }


def _rotation3() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["pitch_deg", "roll_deg", "yaw_deg"],
        "properties": {
            "pitch_deg": _float_leaf(),
            "roll_deg": _float_leaf(),
            "yaw_deg": _float_leaf(),
        },
    }


def _sensor_profile() -> dict:
    # Scene-authored camera profile for the inspect UAV, verbatim.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["camera_name", "capture_backend", "height",
                     "hfov_deg", "source", "width",
                     "fixed_rotation_offset_deg"],
        "properties": {
            "camera_name": _identifier(),
            "capture_backend": _identifier(),
            "height": {"type": "integer", "minimum": 1},
            "hfov_deg": _float_leaf(),
            "source": _identifier(),
            "width": {"type": "integer", "minimum": 1},
            "fixed_rotation_offset_deg": _rotation3(),
        },
    }


def _contract_inspect_uav() -> dict:
    # Inspect-UAV contract copied verbatim from the scene; the three route
    # arrays are producer pass-throughs of the corridor-planned triples (the
    # corpus route-member scan found no int members), so the member rule stays
    # the untyped triple and only non-emptiness is structural.
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
            "corridor_policy": _identifier(),
            "corridor_role": _identifier(),
            "fixed_altitude_loop": {"type": "boolean"},
            "inspect_altitude_code": _identifier(),
            "inspect_altitude_m": _float_leaf(),
            "inspect_fov_coverage_required": {"type": "boolean"},
            "loop_route_enu_m": {"type": "array", "minItems": 1,
                                 "items": _number3()},
            "min_path_length_m": _float_leaf(),
            "motion_policy": _identifier(),
            "observed_capture_boundary_id": _identifier(),
            "planned_route_enu_m": {"type": "array", "minItems": 1,
                                    "items": _number3()},
            "policy": {"const": "semantic_event_contract_v1"},
            "repaired_path_length_m": _float_leaf(),
            "repaired_route_enu_m": {"type": "array", "minItems": 1,
                                     "items": _number3()},
            "sensor_fov_deg": _float_leaf(),
            "sensor_fov_source": _identifier(),
            "sensor_profile": _sensor_profile(),
        },
    }


def _lifecycle() -> dict:
    # Eight-key object built from the scene lifecycle contract (writer at
    # regenerate_boundary_scenarios:7231) plus the altitude-repair update
    # (:8020); identical on all 18564 X lifecycle rows.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "assigned_altitude_m", "auto_lifecycle_applied", "home_hover_enu_m",
            "home_pad_entity_id", "mission_start_enu_m",
            "requires_landing_or_terminal_resolution", "requires_takeoff",
            "waypoint_repair_applied",
        ],
        "properties": {
            "assigned_altitude_m": _float_leaf(),
            "auto_lifecycle_applied": {"const": True},
            "home_hover_enu_m": _writer_float3(),
            "home_pad_entity_id": _identifier(),
            "mission_start_enu_m": _writer_float3(),
            "requires_landing_or_terminal_resolution": {"const": True},
            "requires_takeoff": {"const": True},
            "waypoint_repair_applied": {"const": True},
        },
    }


def _uav_corridor() -> dict:
    # Engine carries the scene's uav_corridor verbatim; the writer stamps
    # ``UAV_ALTITUDE_LAYERS_M`` (uav_corridor_planner.py:27) so the three
    # layers are the planner's (30.0, 50.0, 80.0) on every X row.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["altitude_layers_m", "assigned_altitude_m",
                     "used_lateral_bypass"],
        "properties": {
            "altitude_layers_m": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "prefixItems": [{"const": 30.0}, {"const": 50.0},
                                {"const": 80.0}],
                "items": False,
            },
            "assigned_altitude_m": _float_leaf(),
            "used_lateral_bypass": {"type": "boolean"},
        },
    }


def _event_actor_motion_contract() -> dict:
    """Event-actor motion contract closed per ``policy``.

    Producer basis: ``regenerate_boundary_scenarios.stage_event_actor`` stages
    the six shared keys under ``policy: offstage_until_semantic_dispatch`` and
    ``event_actor_entry`` updates the staged contract with the seven entry
    keys under ``policy: offstage_then_continuous_lane_entry``
    (regenerate_boundary_scenarios.py:1425-1432, 1481-1492 — the builder's
    ``entry_capture_boundary_clearance_m`` routes planning only and is not
    written into the contract).  No X scene, X event script or X writer path
    produces the field, so this slot has no X instance — 尚无实例验证.
    """
    shared = {
        "schema": {"const": "event_actor_motion_contract_v1"},
        "policy": _identifier(),
        "semantic_event_id": _identifier(),
        "pre_activation_presence": {"const": "absent"},
        "stationary_preplacement_allowed": {"const": False},
        "yaw_authority": {"const": "scene_lane_then_velocity"},
    }
    entry_keys = {
        "entry_event_id": _identifier(),
        "entry_tick": {"type": "integer", "minimum": 0},
        "arrival_tick": {"type": "integer", "minimum": 0},
        "route_length_m": _float_leaf(),
        "entry_speed_mps": _float_leaf(),
        "duration_model": {"const":
                           "sum_segment_ceil_distance_over_speed_ticks"},
        "minimum_visible_motion_ratio_before_event": {"const": 1.0},
    }
    entry = {**shared, **entry_keys}
    # Branch schemas carry their own closed boundaries; the wrapper declares
    # only the type and the policy-bound alternatives (a parent-level
    # additionalProperties: False would see no parent annotations and reject
    # every branch key).
    return {
        "type": "object",
        "oneOf": [
            {
                "type": "object", "additionalProperties": False,
                "required": sorted(shared),
                "properties": {**shared,
                               "policy": {"const": "offstage_until_semantic_dispatch"}},
            },
            {
                "type": "object", "additionalProperties": False,
                "required": sorted(entry),
                "properties": {**entry,
                               "policy": {"const": "offstage_then_continuous_lane_entry"}},
            },
        ],
    }


def _path_deviation_contract() -> dict:
    # The reduced scene contract shape (same two keys the L2-2 scene authors
    # and the preserved_fields_from copy admits).  No X authoring exists.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["minimum_route_distance_m", "predicate_id"],
        "properties": {
            "minimum_route_distance_m": _float_leaf(),
            "predicate_id": _identifier(),
        },
    }


def _row_properties() -> dict:
    families = _carried_runtime_state_families()
    return {
        # _row_for_entity base keys (always written).
        "tick": {"type": "integer", "minimum": 0},
        "entity_id": _identifier(),
        "label_class": {"type": "string", "minLength": 1,
                        "enum": list(_LABEL_CLASSES)},
        "asset_id": _identifier(),
        "pos_enu": _writer_float3(),
        "vel_mps": _writer_float3(),
        "yaw_deg": _float_leaf(),
        "state": _identifier(),
        "activity_type": _identifier(),
        "animation_hint": _identifier(),
        # row_activity_payload: pedestrian labels come from the pedestrian
        # activity catalog, non-pedestrian rows stamp standing/solo; every
        # row carries both keys.
        "posture": {"type": "string", "enum": ["standing", "fallen"]},
        "social_state": {"type": "string",
                         "enum": ["solo", "observer", "group"]},
        # Preserved slots: written only when the scene value is neither None
        # nor "" (preserved_fields_from), so each stays optional.
        "activation_tick": {"type": "integer", "minimum": 0},
        "assigned_altitude_m": _float_leaf(),
        "background_pedestrian": _background_pedestrian(),
        "background_role": _identifier(),
        "category": _identifier(),
        "contract_facility": _contract_facility(),
        "contract_inspect_uav": _contract_inspect_uav(),
        "entity_kind": _identifier(),
        # Scene authoring writes the boolean True on inspect UAVs
        # (scene_setup.json "full_episode_presence": true), preserved verbatim.
        "full_episode_presence": {"const": True},
        "ground_flow_contract": _ground_flow_contract(),
        "ground_reference_source": _identifier(),
        "ground_reference_z_m": _float_leaf(),
        "inspect_altitude_code": _identifier(),
        "inspect_altitude_m": _float_leaf(),
        "lifecycle": _lifecycle(),
        "min_path_length_m": _float_leaf(),
        "observer_lifecycle": _observer_lifecycle(),
        "planned_route_waypoints_enu_m": _waypoints(),
        "role": _identifier(),
        "route_waypoints_enu_m": _waypoints(),
        "semantic_role": _identifier(),
        "semantic_scope": _semantic_scope(),
        "spawn_policy": {"const": "event_script_only"},
        "state_sequence": {"type": "array", "minItems": 1,
                           "items": _identifier()},
        "task_id": _identifier(),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": _identifier(),
        # Producer-admitted preserved slots with no X authoring anywhere
        # (no X scene, no X event script, no X row) — 尚无实例验证.
        "event_actor_motion_contract": _event_actor_motion_contract(),
        "path_deviation_contract": _path_deviation_contract(),
        # Runtime families preserved_fields_from deep-copies from the scene
        # entity or deep-merges from a set_runtime_state patch; the seven
        # carried families are instance-verified, the rest has no X carrier.
        **families,
    }


def _row() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "activation_tick", "activity_type", "animation_hint", "asset_id",
            "category", "entity_id", "label_class", "pos_enu", "posture",
            "social_state", "state", "tick", "vel_mps", "yaw_deg",
        ],
        "properties": _row_properties(),
    }


def arm_trajectories_schema() -> dict:
    """Compile the closed X raw trajectory record schema.

    One record branch: the ``EpisodeStateEngine._row_for_entity`` shape
    ``x_arm_pipeline.run_arm`` writes verbatim for every X window.  Required
    are the fourteen always-written keys (the twelve ``_row_for_entity`` base
    keys plus ``activation_tick``/``category``, authored on every configured X
    entity); every other preserved slot is optional because
    ``preserved_fields_from`` drops ``None``/"" values.  The seven carried
    runtime-state families (``communication_state``, ``control_state``,
    ``facility_state``, ``incident_state``, ``mission_state``,
    ``navigation_state``, ``pedestrian_state``) are closed to the executable
    producer contract and instance-verified; ``security_state``,
    ``sensor_state`` and ``vehicle_state`` are producer-allowed without any X
    scene, script or row carrier and stay undeclared (the docstring reports
    them), while the ``event_actor_motion_contract`` and
    ``path_deviation_contract`` slots are declared from the producer's own
    contract shapes with no X instance — 尚无实例验证.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_trajectories_x.record.schema.json",
        "title": "X ARM raw trajectory record",
        **_row(),
    }
