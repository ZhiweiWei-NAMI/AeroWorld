"""Closed whole-record grammar for the ``arm_trajectories`` source family,
L3 layer (profiles L3-1_v1, L3-2_v1, L3-2_v2, L3-3_v1, L3-3_v2).

Provenance chain: ``Dataset/tools/l3_v2/arm_runtime.run_arm`` slices
``EpisodeStateEngine.trajectory_rows`` to ``spec.entity_ids`` and the arm
window ``[start_tick, end_tick]`` (arm_runtime.py L123-127); each row is
``EpisodeStateEngine._row_for_entity`` (batch_generate.py:2395) plus the
``row_activity_payload`` activity triple (batch_generate.py:1856) plus
``preserved_fields_from`` (batch_generate.py:1737), which deep-copies the
scene entity's top-level and ``initial_state``/``visual_state`` slots and
drops ``None``/"" values, so every preserved slot stays optional.  L3 scene
and script shapes come from ``Dataset/tools/regenerate_boundary_scenarios.py``:
``stage_event_actor`` (1411), ``event_actor_entry`` (1435),
``spawn_event_actor_at_dispatch`` (1516), ``world_entity`` (2162),
``semantic_state`` (2293), ``lane_entity`` (2364), ``_ground_flow_contract``
(3287), ``_add_contract_vehicle`` (3668), ``_add_contract_pedestrian`` (3727),
the entry-clearance updater (4384-4428),
``ensure_uav_routes_cross_capture_boundary`` (4681) with the observer
lifecycle stamp (4737), ``_add_contract_inspect_uav`` (5140),
``_add_contract_facility`` (5215), ``add_uav_lifecycle`` (7148), the
make_bundle planned-route stamp (7573-7579), ``_set_uav_altitude_metadata``
(7973), ``apply_uav_corridor_contract`` (8207), the L3-2 NFZ
``deactivation_tick`` authoring (9243) and the L3-3_v1 ambulance staging
(9607).

Enum memberships below are the measured L3 corpus census (51 files, 243300
rows, scanned in the delivery session); the producer chain constrains the
same slots: ``row_activity_payload`` stamps the pedestrian activity catalog
verbatim for pedestrian carriers and mirrors ``state`` for every other
carrier, ``spawn_entity`` (batch_generate.py:2321) stages ``active_from``
and the ``visual_state.mode || entity.state || "spawned"`` activity, and
``stage_event_actor`` stages ``activation_tick=901`` with
``spawn_policy="event_script_only"`` — event actors keep the sentinel tick
inside the arm window, so ``activation_tick`` stays a row slot.
``planned_route_waypoints_enu_m`` is declared non-empty: the make_bundle
stamp deep-copies the UAV route and raises on fewer than two waypoints.
``moving`` stays in the ``state``/``activity_type`` enums verbatim (18
non-pedestrian carrier rows of ``blocked_car_l3_1``; the pedestrian branch
of ``row_activity_payload`` would normalize it to ``walking``).

Mission-UAV rows carry ``lifecycle`` verbatim through
``preserved_fields_from`` (``PRESERVED_ENTITY_FIELDS`` includes it):
``add_uav_lifecycle`` (7231) stamps the six contract keys and
``_set_uav_altitude_metadata`` (8020) adds ``assigned_altitude_m`` and
``waypoint_repair_applied``, so the eight-key object is a row slot closed
by ``_lifecycle()``.  Runtime-state enum and command values are raw
strings, but the producer contract bounds their domain:
``invalid_runtime_state_value_paths`` (runtime_state_contract.py:549-577)
accepts an enum member only when ``value.strip().lower()`` folds into the
governed per-field set of ``RUNTIME_STATE_ENUM_FIELDS``, and a command
member (``active_command``/``active_commands``/``commands``) only when
every element folds into the family's control/mission/security enum set.
The grammar keeps the written string and encodes the same fold through the
audited shared fragment ``structure_fragments.runtime_enum_string_rule``
(the rule the registered ``arm_states`` L1 grammar already uses): an
anchored frame over an explicit Python-``strip()`` whitespace class around
per-member ASCII case cores plus the Kelvin sign U+212A (the sole
non-ASCII code point whose ``str.lower()`` is an ASCII letter on the
pinned Unicode 13.0.0 runtime).  Metadata strings use the fragment's
``nonblank_python_string_rule`` (the producer's ``bool(value.strip())``).
Any future Unicode-table change other than U+212A stays an integration
dependency owned by the producer contract — the fragment raises on a
changed ``unicodedata.unidata_version`` so the rule cannot silently drift.
``ground_flow_contract.physical_lane_lateral_m`` follows the producer
branch keyed on ``actor_kind``: every vehicle writer stamps it
(``_add_contract_vehicle`` 3722, ``ensure_background_ground_flow_routes``
5705, the road-context filler 5962) and no pedestrian writer does
(``_add_contract_pedestrian`` 3807-3812,
``ensure_background_ground_flow_routes`` 5737); ``_ground_flow_contract``
encodes the branch as an ``if/then`` variant on ``actor_kind``.

Producer-allowed branches with no L3 instance carry 尚无实例验证 in the
matching helper comment: the six runtime families without an L3 patch
(``communication_state``, ``facility_state``, ``navigation_state``,
``security_state``, ``sensor_state``, ``vehicle_state``) and the
non-boolean members of the four carried families (modeled from the
executable contract, not from L3 rows), ``posture: fallen`` and
``social_state: observer`` (catalog-reachable via ``medical_incident`` /
``observing``, absent from every L3 scene and script), and the
``offstage_until_semantic_dispatch`` event-actor branch.  No row carries an
integrity or digest field, so no publication-exclusion path is required.
Registry wiring into ``structure_closure.py`` belongs to the integration
stage; this module only exports the pure schema constructor and scans no
data at import.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
)
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
)

__all__ = ["arm_trajectories_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# row_activity_payload stamps the pedestrian activity catalog verbatim on
# pedestrian rows and "standing"/"solo" on every other carrier.  "fallen"
# (medical_incident) and "observer" (observing) are catalog-reachable but
# appear in no L3 scene or script (尚无实例验证).
_POSTURE_SOCIAL_STATES = {
    "posture": ("fallen", "standing"),
    "social_state": ("group", "observer", "solo"),
}

# Measured closed sets; see module docstring for the producer anchors.
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
# 25 shared values; includes the verbatim non-pedestrian "moving" carrier
# state (18 rows) — see module docstring.
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
_GROUND_FLOW_ACTOR_KINDS = ("pedestrian", "vehicle")
_GROUND_FLOW_ROUTE_SOURCES = (
    "sumo_net_directed_lane_geometry", "traffic_bundle_sidewalk",
)


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
    # floats, so all 3749556 measured waypoint members are float source
    # values.  The JSON type stays ``number`` and members stay positional.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "prefixItems": [
            {"type": "number", "x-source-value-types": ["float"]},
            {"type": "number", "x-source-value-types": ["float"]},
            {"type": "number", "x-source-value-types": ["float"]},
        ],
        "items": False,
    }


def _waypoints() -> dict:
    # Producer pass-through of the authored route; empty arrays are a
    # measured status for this slot (corridor-adjacent carriers hold no
    # route).  ``planned_route_waypoints_enu_m`` is the non-empty variant.
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
        # The producer accepts the value only when ``bool(value.strip())``
        # (runtime_state_contract.py:541-543); the fragment names that exact
        # domain instead of the wider ``minLength: 1``.
        return nonblank_python_string_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        return _runtime_state_command_rule(family)
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list and gates each member only on
        # non-empty truthiness (590-593), so members keep ``minLength: 1``.
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        # Producer gate is ``bool(value.strip())`` (600-601).
        return nonblank_python_string_rule()
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return _runtime_state_enum_rule(family, field)
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_string_rule(family: str, field: str) -> dict:
    """Raw-string rule for one enum value of the executable contract.

    ``invalid_runtime_state_value_paths`` (runtime_state_contract.py:549-555)
    validates an enum member as ``value.strip().lower() in allowed`` while
    the written row keeps the raw string.  A JSON Schema ``enum`` of the
    canonical members would reject the non-canonical raw spellings the
    producer accepts, and JSON Schema patterns are ECMA-262, which has no
    case-insensitive flag — ``(?i:...)`` is a Python-only construct and must
    not be used.

    The rule delegates to the audited shared fragment
    ``structure_fragments.runtime_enum_string_rule`` (the same rule the
    registered ``arm_states`` L1 grammar uses): an anchored frame over an
    explicit Python-``strip()`` whitespace class around per-member ASCII
    case cores plus the Kelvin sign U+212A, the sole non-ASCII code point
    whose ``str.lower()`` is an ASCII letter on the pinned Unicode 13.0.0
    runtime.  It keeps whitespace-only, blank, out-of-membership and
    internally-spaced values rejected, matching the producer's rejection of
    ``"   "``.  The fragment raises on a changed ``unicodedata.unidata_version``
    so the rule cannot silently drift; any future Unicode-table change other
    than U+212A stays an integration dependency owned by the producer
    contract, not by this grammar.
    """
    return runtime_enum_string_rule(_family_enum_members(family, field))


def _runtime_state_enum_rule(family: str, field: str) -> dict:
    """Membership is bounded by the producer contract; the written raw
    spelling is kept through the fragment's case-folding alternation."""
    return _runtime_state_string_rule(family, field)


def _runtime_state_command_rule(family: str) -> dict:
    """Command members are governed by the same contract fold
    (runtime_state_contract.py:557-577): every element must fold into the
    family's control/mission/security enum set; the raw spelling is kept."""
    members = _family_enum_members(family)
    member_rule = _runtime_state_string_rule_from_members(members)
    return {"oneOf": [member_rule,
                      {"type": "array", "minItems": 1, "items": member_rule}]}


def _runtime_state_string_rule_from_members(members: tuple[str, ...]) -> dict:
    """Fragment-based raw-string rule for an explicit governed member set
    (the command projection of ``RUNTIME_STATE_ENUM_FIELDS``)."""
    return runtime_enum_string_rule(members)


def _family_enum_members(family: str, field: str | None = None) -> tuple[str, ...]:
    """The governed enum set for one family slot, in canonical sorted order,
    read from the executable contract that rejects values at write time.

    For a (family, field) enum slot this is the projected member set of
    ``RUNTIME_STATE_ENUM_FIELDS``.  For command slots (``active_command``/
    ``active_commands``/``commands``) the executable contract does not key a
    set — it selects the family's control/mission/security enum set at
    validation time (runtime_state_contract.py:569-575) — and that same set
    is projected per family on the family's enum fields, so an unspecified
    field returns the family's first projected set (identical across the
    family's enum fields)."""
    if field is None:
        for (enum_family, _), members in RUNTIME_STATE_ENUM_FIELDS.items():
            if enum_family == family:
                return tuple(sorted(members))
        raise ValueError(f"runtime state family has no governed enum set: {family}")
    members = RUNTIME_STATE_ENUM_FIELDS.get((family, field))
    if members is None:
        raise ValueError(
            f"runtime state field has no governed enum set: {family}.{field}")
    return tuple(sorted(members))


def _runtime_state_family_patch(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
    }


def _runtime_state_families() -> dict:
    """Closed rule per executable family; ``set_runtime_state`` patches
    deep-merge and may create or extend any of them, so each stays an
    optional row slot.  L3 rows carry ``control_state``, ``incident_state``,
    ``mission_state`` and ``pedestrian_state``; the other six families have
    no L3 instance (尚无实例验证), and non-boolean members of the carried
    families are modeled from the executable contract, not from L3 rows."""
    return {family: _runtime_state_family_patch(family)
            for family in RUNTIME_STATE_FIELDS}


def _lifecycle() -> dict:
    # add_uav_lifecycle (7148) writes the six contract keys and
    # _set_uav_altitude_metadata (7973) adds the altitude pair; eight keys on
    # all 11736 mission_uav rows, coordinate members are float triples.
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
            "auto_lifecycle_applied": {"type": "boolean"},
            "home_hover_enu_m": _float3(),
            "home_pad_entity_id": _identifier(),
            "mission_start_enu_m": _float3(),
            "requires_landing_or_terminal_resolution": {"type": "boolean"},
            "requires_takeoff": {"type": "boolean"},
            "waypoint_repair_applied": {"type": "boolean"},
        },
    }


def _uav_corridor() -> dict:
    # _set_uav_altitude_metadata (7973) copies the scene's uav_corridor
    # verbatim; the planner fixes three altitude layers (30.0/50.0/80.0),
    # carried as three float leaves, not a const tuple.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["altitude_layers_m", "assigned_altitude_m", "used_lateral_bypass"],
        "properties": {
            "altitude_layers_m": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": _float_leaf(),
            },
            "assigned_altitude_m": _float_leaf(),
            "used_lateral_bypass": {"type": "boolean"},
        },
    }


def _semantic_scope() -> dict:
    # Scene facility_scope contract (world_entity, 2162).  service_capacity
    # rides on the scoped landing pads and charging stations only, so it
    # stays optional (24450 five-key rows vs 19938 four-key rows).
    return {
        "type": "object", "additionalProperties": False,
        "required": ["authority_id", "ontology_class_id", "scope_subtype", "scope_type"],
        "properties": {
            "authority_id": {"const": "facility_scope_contract.v1"},
            "ontology_class_id": _identifier(),
            "scope_subtype": _identifier(),
            "scope_type": _identifier(),
            "service_capacity": {"type": "integer", "minimum": 0},
        },
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
    # re-stamps; seventeen keys on all 11736 inspect_observer rows.
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
    # corridor-observer row (4737).
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


def _ground_flow_contract() -> dict:
    # _ground_flow_contract (3287) writes ten keys; the eleventh key branches
    # on actor_kind in the producer: every vehicle writer stamps
    # physical_lane_lateral_m (_add_contract_vehicle 3722,
    # ensure_background_ground_flow_routes 5705, the road-context filler
    # 5962) and no pedestrian writer does (_add_contract_pedestrian
    # 3807-3812, ensure_background_ground_flow_routes 5737).  if/then encodes
    # that producer branch: vehicle rows require the offset, pedestrian rows
    # keep the ten-key shape.
    required = [
        "actor_kind", "loop_policy", "min_visible_motion_ratio",
        "min_xy_span_m", "planned_path_length_m", "policy", "required",
        "route_duration_ticks", "route_source", "speed_mps",
    ]
    properties = {
        "actor_kind": {"enum": list(_GROUND_FLOW_ACTOR_KINDS)},
        "loop_policy": {"const": "single_pass_route_waypoints"},
        "min_visible_motion_ratio": _float_leaf(),
        "min_xy_span_m": _float_leaf(),
        "physical_lane_lateral_m": _float_leaf(),
        # round(path_length_m(route_points), 3), or 0.0 for a
        # single-point route (3287); measured over 25 distinct values.
        "planned_path_length_m": _float_leaf(),
        "policy": {"const": "continuous_capture_ground_flow_v1"},
        "required": {"const": True},
        "route_duration_ticks": {"type": "integer", "minimum": 0},
        "route_source": {"enum": list(_GROUND_FLOW_ROUTE_SOURCES)},
        "speed_mps": _float_leaf(),
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": required,
        "properties": properties,
        "if": {"properties": {"actor_kind": {"const": "vehicle"}}},
        "then": {
            "type": "object", "additionalProperties": False,
            "required": [*required, "physical_lane_lateral_m"],
            "properties": properties,
        },
    }


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
    """Event-actor motion contract closed per ``policy``.  The producer
    builders write exactly two shapes: ``event_actor_entry`` stages the
    eight entry keys under ``policy: offstage_then_continuous_lane_entry``
    (the L3-1 roadwork vehicle, 498 rows) while
    ``spawn_event_actor_at_dispatch`` stages the six shared keys only under
    ``policy: offstage_until_semantic_dispatch`` (no L3 script action,
    尚无实例验证)."""
    shared = {
        "schema": {"const": "event_actor_motion_contract_v1"},
        "policy": _identifier(),
        "semantic_event_id": _identifier(),
        "pre_activation_presence": _identifier(),
        "stationary_preplacement_allowed": {"type": "boolean"},
        "yaw_authority": _identifier(),
    }
    entry_keys = {
        "entry_event_id": _identifier(),
        "entry_tick": {"type": "integer", "minimum": 0},
        "arrival_tick": {"type": "integer", "minimum": 0},
        "route_length_m": _float_leaf(),
        "entry_speed_mps": _float_leaf(),
        "duration_model": _identifier(),
        "minimum_visible_motion_ratio_before_event": _float_leaf(),
        "entry_capture_boundary_clearance_m": _float_leaf(),
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


def _row_properties() -> dict:
    families = _runtime_state_families()
    return {
        # _row_for_entity base keys (always written).
        "tick": {"type": "integer", "minimum": 0},
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
        # add_uav_lifecycle (7148) stamps the six contract keys on every
        # mission-lifecycle UAV and _set_uav_altitude_metadata (7973) adds
        # assigned_altitude_m + waypoint_repair_applied; preserved_fields_from
        # carries the eight-key object verbatim (PRESERVED_ENTITY_FIELDS
        # includes "lifecycle").  Observer UAVs, pads, facilities, corridors
        # and ground carriers never receive it (the observer stamp at 4737
        # writes observer_lifecycle instead), so the slot stays optional.
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


def _row() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "tick", "entity_id", "label_class", "asset_id", "pos_enu",
            "vel_mps", "yaw_deg", "state", "activity_type", "animation_hint",
            "posture", "social_state",
        ],
        "properties": _row_properties(),
    }


def arm_trajectories_schema() -> dict:
    """Compile the closed L3 raw trajectory record schema.

    One record branch: the ``EpisodeStateEngine._row_for_entity`` shape that
    ``l3_v2.arm_runtime`` writes for the factual and intervention arms of
    L3-1_v1/L3-2_v1/L3-2_v2/L3-3_v1/L3-3_v2.  Required are the twelve
    always-written base keys; every preserved slot is optional because
    ``preserved_fields_from`` drops ``None``/"" values.  The four L3-carried
    runtime families (``control_state``, ``incident_state``,
    ``mission_state``, ``pedestrian_state``) are measured; the remaining six
    families stay producer-allowed without an L3 instance (尚无实例验证).
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_trajectories_l3.record.schema.json",
        "title": "L3 ARM raw trajectory record",
        **_row(),
    }
