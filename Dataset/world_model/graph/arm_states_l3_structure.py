"""Whole-record grammar for L3 ``arm_states`` JSONL rows.

``l3_v2.arm_runtime._run_arm`` writes its selected engine ``trajectory_rows``
unchanged to both ``raw/states.jsonl`` and ``raw/trajectories.jsonl``. The
engine writes twelve core fields and copies the scene's preserved fields;
``l3_v2.engine.execute`` also stamps ``pedestrian_state`` on every sampled
pedestrian. This module binds the fixed L3 scene identities to their carried
fields and keeps the executable runtime-state family rules on each family.
The L3 mission UAV's scene ``lifecycle`` survives ``preserved_fields_from``
and is part of the row. No registry or source I/O occurs at import.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
    _CONTROL_ENUMS,
    _MISSION_ENUMS,
    _SECURITY_ENUMS,
)
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
    runtime_state_source_guard_rule,
)

__all__ = ["arm_states_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# The fixed L3 scripts schedule pedestrian walking, waiting and evacuating.
# Background pedestrian routes exceed their 900-tick ground-flow contract,
# so their idle fallback never runs in this declared source range.
_POSTURE_SOCIAL_STATES = {
    "posture": ("standing",),
    "social_state": ("group", "solo"),
}

# Fixed L3 scene identities, catalog labels and engine window states.
_LABEL_CLASSES = (
    "barrier", "charging_pile", "construction_fence", "hazard_zone",
    "landing_pad", "no_fly_zone", "pedestrian", "radio_tower", "spawn_zone",
    "traffic_cone", "uav", "uav_corridor", "vehicle",
)
_ASSET_IDS = (
    "facility.barrier.basic", "facility.charger.cityops.v1",
    "facility.landing_pad.visible.v1", "facility.radio.base_tower.v1",
    "pedestrian.cityops.basic.v1", "prop.roadwork.barrier.v1",
    "prop.roadwork.construction_fence.v1", "prop.roadwork.traffic_cone.v1",
    "semantic.spawn_zone", "semantic.uav_corridor.segment.v1",
    "trigger.hazard.generic.box.v1", "trigger.no_fly.box.v1",
    "uav.airsim.cv_pawn.v1", "uav.inspect.quad.v1",
    "vehicle.emergency.ambulance.v1", "vehicle.emergency.suv.v1",
    "vehicle.ground.boxcar.v1", "vehicle.service.box.v1",
)
_CATEGORIES = (
    "airspace_corridor", "crowd_anchor", "facility", "pedestrian", "prop",
    "uav", "vehicle", "vehicle_anchor",
)
# Includes the verbatim non-pedestrian "moving" carrier state.
_ROW_STATES = (
    "active", "airborne", "available", "blocked", "corridor_observer",
    "cruise", "descending", "detour", "evacuating", "hazmat_arrival_reference",
    "inactive", "inspect_racetrack", "isolation_active", "moving", "online",
    "responder", "response", "return_to_pad", "safe_zone_reference",
    "semantic_corridor", "staged", "traffic_flow", "traffic_slow", "waiting",
    "walking",
)
# Same set with the pedestrian states replaced by their pedestrian_* hints.
_ANIMATION_HINTS = (
    "active", "airborne", "available", "blocked", "corridor_observer",
    "cruise", "descending", "detour", "hazmat_arrival_reference", "inactive",
    "inspect_racetrack", "isolation_active", "moving", "online",
    "pedestrian_evacuation_walk", "pedestrian_idle", "pedestrian_walk",
    "responder", "response", "return_to_pad", "safe_zone_reference",
    "semantic_corridor", "staged", "traffic_flow", "traffic_slow",
)
_ROLES = (
    "U_inspect", "mission_uav", "semantic_background_pedestrian",
    "semantic_background_vehicle", "semantic_context", "semantic_facility",
    "semantic_pedestrian", "semantic_vehicle",
)
_ENTITY_KINDS = (
    "facility.barrier", "facility.base_station", "facility.charger",
    "facility.landing_pad", "facility.no_fly_zone",
)
_UAV_CORRIDOR_ROLES = ("inspect_observer", "mission_uav", "observer")
_GROUND_REFERENCE_SOURCES = (
    "home_hover_pose_origin_v1", "map_ground_reference_v1",
)
_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _float3() -> dict:
    # ``_row_for_entity`` writes ``round(float(value), 6)`` per component, so
    # every raw member is a float source value; the JSON type stays ``number``.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


def _route_member() -> dict:
    # Scene routes pass through ``q(x, y, z)`` = ``round(float(...), 3)``
    # (regenerate_boundary_scenarios.py:627) and the corridor planner writes
    # floats, so every measured waypoint member is a float source value.  The
    # JSON type stays ``number`` and members stay positional triples.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


def _waypoints() -> dict:
    # Producer pass-through of the authored route; empty arrays are a
    # measured status for this slot (roadwork props stage empty routes).
    return {"type": "array", "items": _route_member()}


def _planned_route_waypoints() -> dict:
    # make_bundle stamps ``planned_route`` as a deepcopy of the UAV route and
    # raises on ``len(planned_route) < 2`` (7573-7579), so every UAV row
    # carries at least two waypoints.
    return {"type": "array", "minItems": 2, "items": _route_member()}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state family field, from the executable
    producer contract (``validate_runtime_state_patch`` plus
    ``invalid_runtime_state_value_paths`` in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return nonblank_python_string_rule()
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return runtime_enum_string_rule(RUNTIME_STATE_ENUM_FIELDS[family, field])
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        member = runtime_enum_string_rule(_FAMILY_COMMAND_ENUMS[family])
        return {"oneOf": [
            dict(member),
            {"type": "array", "minItems": 1, "items": member},
        ]}
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list; only non-string members fail.
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        return nonblank_python_string_rule()
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family_patch(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _runtime_state_families() -> dict:
    """Closed rule per executable family; ``set_runtime_state`` patches
    deep-merge and may create or extend any of them, so each stays an
    optional row slot.  The spot-read L3 rows carry ``pedestrian_state``
    (``evacuation_active``); ``control_state``, ``incident_state`` and
    ``mission_state`` are scene-carried in the L3 scenarios; the other six
    families have no L3 instance (尚无实例验证), and non-boolean members of
    the carried families are modeled from the executable contract, not from
    L3 rows."""
    return {family: _runtime_state_family_patch(family)
            for family in RUNTIME_STATE_FIELDS}


def _uav_corridor() -> dict:
    # _set_uav_altitude_metadata (7973) copies the scene's uav_corridor
    # verbatim; the planner fixes the ordered 30.0/50.0/80.0 layers.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["altitude_layers_m", "assigned_altitude_m", "used_lateral_bypass"],
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


_FACILITY_SCOPES = {
    "landing_pad": ("world:LandingPad", 1),
    "charging_station": ("world:ChargingStation", 1),
    "communication_base_station": ("world:CommunicationStation", None),
    "barrier": ("world:Barrier", None),
    "no_fly_zone": ("world:NoFlyZone", None),
}


def _semantic_scope(subtype: str | None = None) -> dict:
    """Frozen facility_scope_contract tuple; asset binding is at the row."""
    if subtype is None:
        return {"oneOf": [_semantic_scope(name) for name in _FACILITY_SCOPES]}
    ontology_class, capacity = _FACILITY_SCOPES[subtype]
    properties = {
        "authority_id": {"const": "facility_scope_contract.v1"},
        "ontology_class_id": {"const": ontology_class},
        "scope_subtype": {"const": subtype},
        "scope_type": {"const": "facility"},
    }
    if capacity is not None:
        properties["service_capacity"] = {"const": capacity}
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(properties), "properties": properties,
    }


def _lifecycle() -> dict:
    """Mission UAV stamp from add_uav_lifecycle and altitude repair."""
    properties = {
        "auto_lifecycle_applied": {"const": True},
        "home_pad_entity_id": _identifier(),
        "home_hover_enu_m": _float3(),
        "mission_start_enu_m": _float3(),
        "requires_takeoff": {"const": True},
        "requires_landing_or_terminal_resolution": {"const": True},
        "assigned_altitude_m": _float_leaf(),
        "waypoint_repair_applied": {"const": True},
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(properties), "properties": properties,
    }


def _sensor_profile() -> dict:
    # default_uav_sensor_profile (311) reads
    # Config/LowAltitude/uav_sensor_profile.json; the seven closed keys ride
    # on every contract_inspect_uav row.  The producer checks hfov_deg > 0
    # and a pitch <= -60 rotation offset; the raw grammar keeps the values.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "camera_name", "capture_backend", "fixed_rotation_offset_deg",
            "height", "hfov_deg", "source", "width",
        ],
        "properties": {
            "camera_name": _identifier(),
            "capture_backend": _identifier(),
            "fixed_rotation_offset_deg": {
                "type": "object", "additionalProperties": False,
                "required": ["pitch_deg", "roll_deg", "yaw_deg"],
                "properties": {
                    "pitch_deg": _float_leaf(),
                    "roll_deg": _float_leaf(),
                    "yaw_deg": _float_leaf(),
                },
            },
            "height": {"type": "integer", "minimum": 1},
            "hfov_deg": _float_leaf(),
            "source": _identifier(),
            "width": {"type": "integer", "minimum": 1},
        },
    }


def _contract_inspect_uav() -> dict:
    # _add_contract_inspect_uav (5140) plus the
    # apply_uav_corridor_contract/ensure_uav_routes_cross_capture_boundary
    # re-stamps; seventeen keys on every inspect_observer row.
    # planned_route_enu_m inherits the make_bundle non-emptiness; the loop
    # and repaired routes are re-stamps of the same contract routes.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "corridor_policy", "corridor_role", "fixed_altitude_loop",
            "inspect_altitude_code", "inspect_altitude_m",
            "inspect_fov_coverage_required", "loop_route_enu_m",
            "min_path_length_m", "motion_policy", "observed_capture_boundary_id",
            "planned_route_enu_m", "policy", "repaired_path_length_m",
            "repaired_route_enu_m", "sensor_fov_deg", "sensor_fov_source",
            "sensor_profile",
        ],
        "properties": {
            "corridor_policy": {"const": "inspect_capture_corridor"},
            "corridor_role": {"const": "inspect_observer"},
            "fixed_altitude_loop": {"type": "boolean"},
            "inspect_altitude_code": _identifier(),
            "inspect_altitude_m": _float_leaf(),
            "inspect_fov_coverage_required": {"type": "boolean"},
            "loop_route_enu_m": {"type": "array", "minItems": 1,
                                 "items": _route_member()},
            "min_path_length_m": _float_leaf(),
            "motion_policy": {
                "const": "fixed_altitude_loop_observes_capture_boundary"},
            "observed_capture_boundary_id": _identifier(),
            "planned_route_enu_m": {"type": "array", "minItems": 2,
                                    "items": _route_member()},
            "policy": {"const": "semantic_event_contract_v1"},
            "repaired_path_length_m": _float_leaf(),
            "repaired_route_enu_m": {"type": "array", "minItems": 1,
                                     "items": _route_member()},
            "sensor_fov_deg": _float_leaf(),
            "sensor_fov_source": {
                "const": "Config/LowAltitude/uav_sensor_profile.json#hfov_deg"},
            "sensor_profile": _sensor_profile(),
        },
    }


def _observer_lifecycle() -> dict:
    # ensure_uav_routes_cross_capture_boundary stamps the four keys on every
    # corridor-observer row (4737-4742).
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
                "const": "airborne_hold_inside_runtime_boundary"},
        },
    }


def _ground_flow_contract(actor: str | None = None) -> dict:
    # The common builder writes ten keys. Contract vehicles add an eleventh
    # lane offset and use SUMO lane geometry; pedestrians use sidewalk route.
    cases = []
    for actor_kind, route_source in (
        ("pedestrian", "traffic_bundle_sidewalk"),
        ("vehicle", "sumo_net_directed_lane_geometry"),
    ):
        properties = {
            "actor_kind": {"const": actor_kind},
            "loop_policy": {"const": "single_pass_route_waypoints"},
            "min_visible_motion_ratio": _float_leaf(),
            "min_xy_span_m": _float_leaf(),
            "planned_path_length_m": _float_leaf(),
            "policy": {"const": "continuous_capture_ground_flow_v1"},
            "required": {"const": True},
            "route_duration_ticks": {"type": "integer", "minimum": 0},
            "route_source": {"const": route_source},
            "speed_mps": _float_leaf(),
        }
        if actor_kind == "vehicle":
            properties["physical_lane_lateral_m"] = _float_leaf()
        cases.append({
            "type": "object", "additionalProperties": False,
            "required": sorted(properties), "properties": properties,
        })
    if actor == "pedestrian":
        return cases[0]
    if actor == "vehicle":
        return cases[1]
    return {"oneOf": cases}


def _background_pedestrian() -> dict:
    # _add_contract_pedestrian (3727) / semantic_state (2293): three keys.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["contract_scenario_id", "policy", "semantic_role"],
        "properties": {
            "contract_scenario_id": _identifier(),
            "policy": {"const": "semantic_event_contract_v1"},
            "semantic_role": _identifier(),
        },
    }


def _background_vehicle() -> dict:
    # _add_contract_vehicle (3668): the pedestrian keys plus the physical
    # lane offset, always written for vehicles.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "contract_scenario_id", "physical_lane_lateral_m", "policy",
            "semantic_role",
        ],
        "properties": {
            "contract_scenario_id": _identifier(),
            "physical_lane_lateral_m": _float_leaf(),
            "policy": {"const": "semantic_event_contract_v1"},
            "semantic_role": _identifier(),
        },
    }


def _contract_facility() -> dict:
    # _add_contract_facility (5215): exactly two keys on scoped facilities.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["contract_scenario_id", "policy"],
        "properties": {
            "contract_scenario_id": _identifier(),
            "policy": {"const": "semantic_event_contract_v1"},
        },
    }


def _event_actor_motion_contract() -> dict:
    """L3-1's event_actor_entry stamp; other policies have no L3 source."""
    properties = {
        "schema": {"const": "event_actor_motion_contract_v1"},
        "policy": {"const": "offstage_then_continuous_lane_entry"},
        "semantic_event_id": _identifier(),
        "pre_activation_presence": {"const": "absent"},
        "stationary_preplacement_allowed": {"const": False},
        "yaw_authority": {"const": "scene_lane_then_velocity"},
        "entry_event_id": _identifier(),
        "entry_tick": {"type": "integer", "minimum": 0},
        "arrival_tick": {"type": "integer", "minimum": 0},
        "route_length_m": _float_leaf(),
        "entry_speed_mps": _float_leaf(),
        "duration_model": {"const": "sum_segment_ceil_distance_over_speed_ticks"},
        "minimum_visible_motion_ratio_before_event": {"const": 1.0},
        "entry_capture_boundary_clearance_m": _float_leaf(),
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(properties), "properties": properties,
    }


def _row_properties() -> dict:
    families = _runtime_state_families()
    return {
        # _row_for_entity base keys (always written).  ``tick`` is the
        # engine.execute runs through end_tick=900 inclusive.
        "tick": {"type": "integer", "minimum": 0, "maximum": 900},
        "entity_id": _identifier(),
        "label_class": {"enum": list(_LABEL_CLASSES)},
        "asset_id": {"enum": list(_ASSET_IDS)},
        "pos_enu": _float3(),
        "vel_mps": _float3(),
        "yaw_deg": _float_leaf(),
        "state": {"enum": list(_ROW_STATES)},
        "activity_type": {"enum": list(_ROW_STATES)},
        "animation_hint": {"enum": list(_ANIMATION_HINTS)},
        "posture": {"enum": list(_POSTURE_SOCIAL_STATES["posture"])},
        "social_state": {"enum": list(_POSTURE_SOCIAL_STATES["social_state"])},
        # Preserved slots: written only when the scene value is neither None
        # nor "" (preserved_fields_from), so each stays optional.
        "activation_tick": {"type": "integer", "minimum": 0},
        # Authored scene value; the L3-2 NFZ deactivation is 520 (9243).
        "deactivation_tick": {"type": "integer", "minimum": 0},
        "assigned_altitude_m": _float_leaf(),
        "background_role": {"const": "semantic_context"},
        "background_pedestrian": _background_pedestrian(),
        "background_vehicle": _background_vehicle(),
        "category": {"enum": list(_CATEGORIES)},
        "contract_facility": _contract_facility(),
        "contract_inspect_uav": _contract_inspect_uav(),
        "entity_kind": {"enum": list(_ENTITY_KINDS)},
        "event_actor_motion_contract": _event_actor_motion_contract(),
        # Observer rows only; the stamp writes the boolean True (4737).
        "full_episode_presence": {"const": True},
        "ground_flow_contract": _ground_flow_contract(),
        "ground_reference_source": {"enum": list(_GROUND_REFERENCE_SOURCES)},
        "ground_reference_z_m": _float_leaf(),
        # EpisodeContract.inspect_code = f"I{int(round(altitude))}"
        # (semantic_event_contract.py); measured I18/I28 — an identifier,
        # not a const.
        "inspect_altitude_code": _identifier(),
        "lifecycle": _lifecycle(),
        "min_path_length_m": _float_leaf(),
        "observer_lifecycle": _observer_lifecycle(),
        "planned_route_waypoints_enu_m": _planned_route_waypoints(),
        "role": {"enum": list(_ROLES)},
        "route_waypoints_enu_m": _waypoints(),
        "semantic_role": _identifier(),
        "semantic_scope": _semantic_scope(),
        # stage_event_actor sentinel: "event_script_only" (1411).
        "spawn_policy": {"const": "event_script_only"},
        "state_sequence": {"type": "array", "minItems": 1,
                           "items": _identifier()},
        "task_id": _identifier(),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": {"enum": list(_UAV_CORRIDOR_ROLES)},
        # Runtime families preserved_fields_from deep-copies from the scene
        # entity or deep-merges from a set_runtime_state patch.
        "control_state": families["control_state"],
        "mission_state": families["mission_state"],
        "navigation_state": families["navigation_state"],
        "security_state": families["security_state"],
        "sensor_state": families["sensor_state"],
        "facility_state": families["facility_state"],
        "communication_state": families["communication_state"],
        "incident_state": families["incident_state"],
        "vehicle_state": families["vehicle_state"],
        "pedestrian_state": families["pedestrian_state"],
    }


_CORE_KEYS = frozenset((
    "tick", "entity_id", "label_class", "asset_id", "pos_enu", "vel_mps",
    "yaw_deg", "state", "activity_type", "animation_hint", "posture",
    "social_state",
))
_WORLD_KEYS = frozenset((
    "activation_tick", "category", "role", "route_waypoints_enu_m",
    "semantic_role", "state_sequence", "task_id",
))
_UAV_KEYS = frozenset((
    "assigned_altitude_m", "ground_reference_source",
    "ground_reference_z_m", "planned_route_waypoints_enu_m", "uav_corridor",
    "uav_corridor_role",
))


def _identity_branch(
    asset: str, label: str, category: str, role: str | None,
    *, extra: tuple[str, ...] = (), required: tuple[str, ...] = (),
    overrides: dict | None = None, world: bool = True,
) -> dict:
    """Restrict preserved keys to one fixed L3 scene identity."""
    allowed = set(_CORE_KEYS) | set(RUNTIME_STATE_FIELDS)
    allowed.update(_WORLD_KEYS if world else ("activation_tick", "category"))
    allowed.update(extra)
    properties = {
        "asset_id": {"const": asset},
        "label_class": {"const": label},
        "category": {"const": category},
    }
    if label == "pedestrian":
        properties.update({
            "state": {"enum": ["walking", "waiting", "evacuating"]},
            "activity_type": {"enum": ["walking", "waiting", "evacuating"]},
            "animation_hint": {"enum": [
                "pedestrian_walk", "pedestrian_idle",
                "pedestrian_evacuation_walk",
            ]},
            "posture": {"const": "standing"},
            "social_state": {"enum": ["solo", "group"]},
        })
    else:
        non_pedestrian_states = [
            state for state in _ROW_STATES if state in _ANIMATION_HINTS
        ]
        properties.update({
            "state": {"enum": non_pedestrian_states},
            "activity_type": {"enum": non_pedestrian_states},
            "animation_hint": {"enum": non_pedestrian_states},
            "posture": {"const": "standing"},
            "social_state": {"const": "solo"},
        })
    needed = set(_WORLD_KEYS if world else ("activation_tick", "category"))
    if role is not None:
        properties["role"] = {"const": role}
    if overrides:
        properties.update(overrides)
    needed.update(required)
    return {
        "type": "object",
        "propertyNames": {"enum": sorted(allowed)},
        "properties": properties,
        "required": sorted(needed),
    }


def _identity_branches() -> list[dict]:
    # Frozen L3 scene_setup identities plus the current 51 ARM mechanisms:
    # none of the authored spawn actions replaces its scene asset, and the
    # intervention injections only move, patch runtime state, or set a mode.
    branches = []
    for asset, label in (
        ("prop.roadwork.barrier.v1", "barrier"),
        ("prop.roadwork.construction_fence.v1", "construction_fence"),
        ("prop.roadwork.traffic_cone.v1", "traffic_cone"),
    ):
        branches.append(_identity_branch(
            asset, label, "prop", "semantic_context",
            extra=("spawn_policy",), required=("spawn_policy",),
        ))
    branches.append(_identity_branch(
        "semantic.uav_corridor.segment.v1", "uav_corridor",
        "airspace_corridor", None, world=False,
    ))
    for category in ("crowd_anchor", "vehicle_anchor"):
        branches.append(_identity_branch(
            "semantic.spawn_zone", "spawn_zone", category,
            "semantic_context",
        ))
    branches.append(_identity_branch(
        "pedestrian.cityops.basic.v1", "pedestrian", "pedestrian",
        "semantic_pedestrian", required=("pedestrian_state",),
    ))
    branches.append(_identity_branch(
        "pedestrian.cityops.basic.v1", "pedestrian", "pedestrian",
        "semantic_background_pedestrian",
        extra=("background_role", "background_pedestrian",
               "ground_flow_contract"),
        required=("background_role", "background_pedestrian",
                  "ground_flow_contract", "pedestrian_state"),
        overrides={"ground_flow_contract":
                   _ground_flow_contract("pedestrian")},
    ))
    for asset in (
        "vehicle.ground.boxcar.v1", "vehicle.emergency.suv.v1",
        "vehicle.service.box.v1",
    ):
        branches.append(_identity_branch(
            asset, "vehicle", "vehicle", "semantic_background_vehicle",
            extra=("background_role", "background_vehicle",
                   "ground_flow_contract"),
            required=("background_role", "background_vehicle",
                      "ground_flow_contract"),
            overrides={"ground_flow_contract": _ground_flow_contract("vehicle")},
        ))
    branches.append(_identity_branch(
        "vehicle.ground.boxcar.v1", "vehicle", "vehicle",
        "semantic_vehicle", extra=("spawn_policy", "event_actor_motion_contract"),
        required=("spawn_policy", "event_actor_motion_contract"),
    ))
    branches.append(_identity_branch(
        "vehicle.emergency.ambulance.v1", "vehicle", "vehicle",
        "semantic_vehicle", extra=("spawn_policy",),
        required=("spawn_policy",),
    ))
    for asset, label, subtype, kind in (
        ("facility.landing_pad.visible.v1", "landing_pad", "landing_pad",
         "facility.landing_pad"),
        ("facility.charger.cityops.v1", "charging_pile", "charging_station",
         "facility.charger"),
        ("facility.radio.base_tower.v1", "radio_tower",
         "communication_base_station", "facility.base_station"),
        ("facility.barrier.basic", "barrier", "barrier", "facility.barrier"),
    ):
        branches.append(_identity_branch(
            asset, label, "facility", "semantic_facility",
            extra=("semantic_scope", "entity_kind", "contract_facility"),
            required=("semantic_scope", "entity_kind"),
            overrides={"semantic_scope": _semantic_scope(subtype),
                       "entity_kind": {"const": kind}},
        ))
    for asset, label, extras in (
        ("trigger.no_fly.box.v1", "no_fly_zone", ("deactivation_tick",)),
        ("trigger.hazard.generic.box.v1", "hazard_zone", ()),
    ):
        branches.append(_identity_branch(
            asset, label, "facility", "semantic_context",
            extra=("semantic_scope", "entity_kind", "spawn_policy", *extras),
            required=("semantic_scope", "entity_kind", "spawn_policy",
                      "incident_state"),
            overrides={"semantic_scope": _semantic_scope("no_fly_zone"),
                       "entity_kind": {"const": "facility.no_fly_zone"}},
        ))
    uav_extra = tuple(_UAV_KEYS)
    uav_required = tuple(_UAV_KEYS)
    branches.append(_identity_branch(
        "uav.inspect.quad.v1", "uav", "uav", "mission_uav",
        extra=(*uav_extra, "lifecycle"),
        required=(*uav_required, "lifecycle"),
        overrides={"uav_corridor_role": {"const": "mission_uav"},
                   "ground_reference_source":
                   {"const": "home_hover_pose_origin_v1"}},
    ))
    branches.append(_identity_branch(
        "uav.inspect.quad.v1", "uav", "uav", "U_inspect",
        extra=(*uav_extra, "contract_inspect_uav", "full_episode_presence",
               "inspect_altitude_code", "min_path_length_m"),
        required=(*uav_required, "contract_inspect_uav",
                  "full_episode_presence", "inspect_altitude_code",
                  "min_path_length_m"),
        overrides={"uav_corridor_role": {"const": "inspect_observer"},
                   "ground_reference_source":
                   {"const": "map_ground_reference_v1"}},
    ))
    branches.append(_identity_branch(
        "uav.airsim.cv_pawn.v1", "uav", "uav", "mission_uav",
        extra=(*uav_extra, "observer_lifecycle"),
        required=(*uav_required, "observer_lifecycle"),
        overrides={"uav_corridor_role": {"const": "observer"},
                   "ground_reference_source":
                   {"const": "map_ground_reference_v1"}},
    ))
    return branches


def _row() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": sorted(_CORE_KEYS),
        "properties": _row_properties(),
        "allOf": [{"oneOf": _identity_branches()}],
    }


def arm_states_schema() -> dict:
    """Return the closed record schema for one ``arm_states`` L3 JSONL row.

    The writer emits one row shape with identity-specific preserved fields.
    There is no observation marker, tombstone, or values projection.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_states_l3.record.schema.json",
        "title": "arm_states L3 whole-record grammar (raw/states.jsonl row)",
        **_row(),
    }
