"""Closed record structure for X ARM raw state rows.

Source family ``arm_states``; profiles X1_rain_to_c2loss_to_forced_landing,
X2_gnss_spoof_to_geofence_violation, X3_pedestrian_fall_to_emergency_response,
X4_fog_to_uav_conflict, X5_comm_failure_to_pad_contention,
X6_crowd_evacuation_to_airspace_lockdown (``grp_cross_layer`` windows;
6 profiles x 3 seeds x 2-3 arms = 42 declared files, 342741 present rows).
Producer chain: ``Dataset/tools/x_arm_pipeline.py::run_arm`` replays the
compiled branch through ``Dataset/tools/l6_v2/engine.py::execute`` and writes
each engine trajectory row verbatim to ``raw/states.jsonl``
(``x_arm_pipeline.py`` line 251), so the row space is exactly
``EpisodeStateEngine._row_for_entity`` (``Dataset/tools/batch_generate.py``
line 2395): the fixed core keys, ``row_activity_payload`` (line 1856) for the
four activity keys, and ``preserved_fields_from`` (line 1737) over
``PRESERVED_ENTITY_FIELDS`` (line 215) for every stamp and runtime-state
family.

The record space partitions by ``label_class``, which fixes ``category``,
the role default, the state vocabulary and the stamp set at once, so the
root is a flat oneOf of complete-row branches (every sub-schema is
a whole-record branch with closed keys, so ``additionalProperties: False``
applies to the full row).  The splits mirror the producer writers in
``Dataset/tools/regenerate_boundary_scenarios.py``:

- ``uav`` splits three ways by ``role``: ``U_inspect`` rows (contract stamp
  ``contract_inspect_uav``, writer line 5173 with the route repair at line
  8330, ``full_episode_presence``, the fixed four-step sequence and corridor
  role ``inspect_observer``) versus ``mission_uav`` rows, which carry
  exactly one of ``lifecycle`` (home-pad missions, line 7236) or
  ``observer_lifecycle`` (corridor observers, line 4737).
- ``pedestrian`` splits by background contract and runtime-family presence:
  background-contract rows
  (``background_pedestrian`` line 3802 + ``background_role`` +
  ``ground_flow_contract`` line 3287 co-present, ``role`` fixed to
  ``semantic_background_pedestrian``, two-step walking sequence, nonempty
  route), plain rows (empty route) with neither or exactly one of
  ``incident_state`` and ``pedestrian_state``, and incident rows with both.
- ``landing_pad`` splits contract-scene rows (``contract_facility`` line
  5243, ``semantic_role`` fixed to the world_entity contract sentence,
  ``state`` ``available``) from runtime-side rows (``facility_state``
  required); ``radio_tower`` splits the same way over
  ``communication_state``.
- ``no_fly_zone``/``police_tape`` split by ``_mark_event_actor`` (line
  1416): activation 901 rows with ``spawn_policy``
  ``event_script_only`` versus tick-0 rows without the stamp.
- ``barrier``/``charging_pile`` always carry ``contract_facility``;
  ``spawn_zone``/``vehicle``/``uav_corridor`` are single-keyset branches
  (the corridor rows carry no ``role``/``task_id``/``state_sequence``/
  ``semantic_role`` and hold ``vel_mps`` at zero).

Key authority per class is the row builder plus the named writers above;
facility scope stamps are closed against
``Dataset/semantic_rules/profiles/facility_scope_contract.json`` through
``Dataset/semantic_truth/facility_scope.py::semantic_scope_for_source``
(each class's ``scope_subtype``, ``ontology_class_id`` and — for
charging_station/landing_pad — ``service_capacity`` come from that frozen
contract), labels and asset ids against
``Config/LowAltitude/asset_catalog.json`` through
``Dataset/tools/typed_entity_labels.py``.  ``row_activity_payload`` writes
``activity_type`` = the row state and a catalog animation label on every row
of every class (verified on all 342741 rows); the grammar closes
``state``/``activity_type`` to the same per-class vocabulary and keeps the
``animation_hint`` label as an identifier — the per-activity equality pair
is producer-closed by ``row_activity_payload`` + the catalog and is not
duplicated as if/then branches here.  Posture and ``social_state`` follow
the catalog mapping for the branch's state set (medical_incident is the
only fallen activity and only occurs on the incident branch).

All ten runtime-state families are accepted for every label class,
closed against the executable contract in
``Dataset/tools/runtime_state_contract.py`` (consumed fields, boolean/
enum/command value rules, facility requester lists, non-negative numbers,
the ``rule_id``/``mechanism_id`` metadata fields): ``set_runtime_state``
can address any entity, so every label class has a branch accepting each
family.  Per-field value rules mirror
``invalid_runtime_state_value_paths`` exactly: the
raw-string domain whose ``strip().lower()`` is a member of the family/field
enum set, non-whitespace-bearing metadata strings and facility allocation
owner strings, booleans, the facility and hazard number rules, and the raw
command string/list shape whose members obey the same stripped-lowercase
membership.  Enum and command member sets come from the current contract
tables, and the raw-string domain is stated by the shared dual-audited
fragments ``structure_fragments.runtime_enum_string_rule`` and
``structure_fragments.nonblank_python_string_rule``: case and surrounding
whitespace variants are producer-accepted raw forms, not substring junk.
No checked X producer input writes a command field (the X5/X6 scene
initial states and the X6 crowd_evacuation schedule author allocation and
boolean families only), so command branches remain 尚无实例验证.  Families
already present in X rows (``control_state``, ``mission_state``,
``navigation_state``, ``facility_state``, ``communication_state``,
``incident_state``, ``pedestrian_state``) are instance-verified.

``batch_generate.validate_runtime_state_patch`` recursively rejects strings
whose casefolded alphanumeric token names a forbidden semantic source field,
and strings that look like paths.  Each runtime family object uses the
``engine_whole`` source guard so this applies to metadata, facility requester
ids and allocation owner ids without restricting unrelated row strings.

Branches and forms with a producer basis but no current X instance are
declared and remain 尚无实例验证:

- the ``asset_anchor`` label: catalog-declared, written by no X producer
  path (0 of 342741 rows) — no branch exists for it;
- the ``security_state``, ``sensor_state`` and ``vehicle_state`` families
  (contract-declared, 0 rows) stay valid through the shared family rule on
  every branch;
- ``deactivation_tick`` (in ``PRESERVED_ENTITY_FIELDS``) appears on no X row
  because window rows are written only for entities active at the tick, so
  it is not declared; ``offstage`` state rows likewise never occur;
- ``contract_logical_sidecar`` (writer line 8136) and
  ``event_actor_motion_contract`` (line 1420) are preserved fields with no X
  instance and are not declared;
- ``spawn_policy`` on classes other than ``no_fly_zone``/``police_tape``
  and activation values other than 0/901 have no X instance; the event-actor
  split keeps its writer-closed two-form shape;
- ``observer_lifecycle`` rows at corridor layers 30.0/50.0 (all 10560 X rows
  hold 80.0) and ``lifecycle`` rows at layer 80.0 (all X rows hold 30/50):
  the declared altitude enums keep the producer's full layer table
  ``UAV_ALTITUDE_LAYERS_M`` (``uav_corridor_planner.py`` line 27) with the
  instance coverage noted here;
- top-level ``inspect_altitude_m`` appears on no X row (the altitude lives
  inside ``contract_inspect_uav``), so only ``inspect_altitude_code`` is
  declared.

These records carry no integrity metadata (no digest/hash/sha256 field on
any of the 342741 rows), so no publication-exclusion path is required.

The sibling roots ``arm_states_l4_structure.py`` and
``arm_states_l6_structure.py`` declare the L4/L6 producers' record spaces;
their stamp sets and couplings differ, so they are not reused as-is.
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
    runtime_state_source_guard_rule,
)

__all__ = ["arm_states_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# The facility contract stamp sentence (world_entity default semantic_role
# for contract-scene facility rows, verified constant on all stamped rows).
_FACILITY_CONTRACT_SENTENCE = (
    "visible facility context required by low-altitude event-chain contract"
)
_EVENT_SCRIPT_ONLY = "event_script_only"
_CONTRACT_SCENARIOS = (
    "X1_rain_to_c2loss_to_forced_landing",
    "X2_gnss_spoof_to_geofence_violation",
    "X3_pedestrian_fall_to_emergency_response",
    "X4_fog_to_uav_conflict",
    "X5_comm_failure_to_pad_contention",
    "X6_crowd_evacuation_to_airspace_lockdown",
)

# Per-class state vocabularies (sample_keyframes / row_activity_payload /
# _state_label outputs, verified on every row of the class).
_STATE_ENUMS = {
    "barrier": ("available",),
    "charging_pile": ("available",),
    "landing_pad": ("available", "reserved", "unavailable"),
    "no_fly_zone": ("active", "idle", "standdown"),
    "police_tape": ("standdown", "staged"),
    "radio_tower": ("failed", "link_lost", "online", "rain_stressed"),
    "spawn_zone": ("idle", "reroute_exit_reference", "safe_zone_reference"),
    "uav": (
        "airborne", "corridor_observer", "cruise", "descending", "detecting",
        "geofence_alert", "gnss_spoofed", "hold", "holding", "hover",
        "landed", "low_altitude_hold", "observation_hold", "takeoff",
        "visual_degraded",
    ),
    "uav_corridor": ("semantic_corridor",),
    "vehicle": ("response",),
}
# Pedestrian states written in X windows (row_activity_payload through the
# activity catalog); the catalog itself declares more members.
_PED_STATES = (
    "chatting", "evacuating", "medical_incident", "observing", "phone_call",
    "standing", "waiting", "walking",
)
# Per-class semantic_role vocabularies for the classes whose writer fixes
# them (world_entity defaults and state mirroring, verified per class).
_SEMANTIC_ROLE_ENUMS = {
    "no_fly_zone": ("idle",),
    "police_tape": ("staged",),
    "spawn_zone": ("idle", "reroute_exit_reference", "safe_zone_reference"),
    "vehicle": ("response",),
}


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _number3() -> dict:
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _waypoints(min_items: int, max_items: int | None = None) -> dict:
    rule: dict = {"type": "array", "minItems": min_items,
                  "items": _number3()}
    if max_items is not None:
        rule["maxItems"] = max_items
    return rule


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _string_enum(*values: str) -> dict:
    return {"type": "string", "enum": list(values)}


def _command_member_set(family: str) -> frozenset[str]:
    """Command member vocabulary for one family, from the current
    executable contract: ``invalid_runtime_state_value_paths`` (lines
    569-575) checks every stripped-lowercase member against the family's
    mode-enum set (_CONTROL_ENUMS / _MISSION_ENUMS / _SECURITY_ENUMS)."""
    enums = {
        "control_state": frozenset({
            "nominal", "normal", "none", "idle", "rth", "return_to_home",
            "alternate_rth", "alternate_return_to_home", "reroute",
            "alternate_reroute", "alternate_route", "altitude_correction",
            "altitude_corrective_maneuver", "deconfliction", "evasion",
            "pull_up", "safe_hold", "hold", "resequence", "divert",
            "diversion", "slowdown",
        }),
        "mission_state": frozenset({
            "nominal", "normal", "active", "none", "abort", "mission_abort",
        }),
        "security_state": frozenset({
            "nominal", "normal", "clear", "none", "gcs_compromised",
            "jamming", "unauthorized_command", "command_integrity_violation",
            "command_lockout",
        }),
    }
    return enums[family]


def _command_rule(family: str) -> dict:
    """Raw command shape: a non-whitespace-only string or a nonempty list of
    them (``bool(value.strip())`` plus stripped-lowercase membership, lines
    557-577).  The member rule is the shared fragment; case and surrounding
    whitespace variants are producer-accepted raw forms.  No checked X
    producer input writes a command field (尚无实例验证)."""
    member = runtime_enum_string_rule(_command_member_set(family))
    return {"oneOf": [
        member,
        {"type": "array", "minItems": 1, "items": member},
    ]}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state family field, from the executable
    producer contract (validate_runtime_state_patch plus
    invalid_runtime_state_value_paths in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        # invalid_runtime_state_value_paths (runtime_state_contract.py lines
        # 541-543) keeps a metadata value only when it is a string containing
        # at least one non-whitespace character.  No X row carries
        # rule_id/mechanism_id (尚无实例验证).
        return nonblank_python_string_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        return _command_rule(family)
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list; only non-string members fail.
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        # Same bool(value.strip()) check as the metadata values (lines
        # 596-602); the scene pads write the raw provenance string "none".
        return nonblank_python_string_rule()
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        # invalid_runtime_state_value_paths (lines 549-555) keeps the value
        # only when it is a string whose stripped lowercase form is a member
        # of the family/field enum set.  The shared fragment states exactly
        # that raw-string domain; case and surrounding whitespace variants
        # are producer-accepted raw forms, and unrelated substring strings
        # stay outside.  All current contract member sets are lowercase
        # ASCII identifiers, so no X instance is rejected by the tightening.
        allowed = RUNTIME_STATE_ENUM_FIELDS[(family, field)]
        return runtime_enum_string_rule(allowed)
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family_patch(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family]
                    | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _optional_runtime_families() -> dict:
    """All ten contract families as optional row members: set_runtime_state
    can address any entity, so every branch accepts every family."""
    return {family: _runtime_state_family_patch(family)
            for family in sorted(RUNTIME_STATE_FIELDS)}


def _semantic_scope(subtype: str, ontology_class: str,
                    capacity: bool) -> dict:
    """Facility scope stamp from semantic_scope_for_source over the frozen
    facility_scope_contract.json (service_capacity only for the
    charging_station and landing_pad subtypes; the contract value is 1)."""
    properties = {
        "authority_id": {"const": "facility_scope_contract.v1"},
        "ontology_class_id": {"const": ontology_class},
        "scope_subtype": {"const": subtype},
        "scope_type": {"const": "facility"},
    }
    required = ["authority_id", "ontology_class_id", "scope_subtype",
                "scope_type"]
    if capacity:
        properties["service_capacity"] = {"const": 1}
        required.append("service_capacity")
    return {
        "type": "object", "additionalProperties": False,
        "required": required, "properties": properties,
    }


def _contract_facility() -> dict:
    # contract_facility writer (regenerate_boundary_scenarios.py line 5243).
    return {
        "type": "object", "additionalProperties": False,
        "required": ["policy", "contract_scenario_id"],
        "properties": {
            "policy": {"const": "semantic_event_contract_v1"},
            "contract_scenario_id": _string_enum(*_CONTRACT_SCENARIOS),
        },
    }


def _background_pedestrian() -> dict:
    # background_pedestrian writer (regenerate_boundary_scenarios.py line
    # 3802): semantic_role is a producer-authored label mirrored by the
    # row-level semantic_role.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["policy", "semantic_role", "contract_scenario_id"],
        "properties": {
            "policy": {"const": "semantic_event_contract_v1"},
            "semantic_role": _identifier(),
            "contract_scenario_id": _string_enum(*_CONTRACT_SCENARIOS),
        },
    }


def _ground_flow_contract() -> dict:
    # _ground_flow_contract (regenerate_boundary_scenarios.py line 3287):
    # ten keys; every scalar is constant on X rows except
    # planned_path_length_m, and the contract carries no route array.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "policy", "actor_kind", "required", "min_visible_motion_ratio",
            "min_xy_span_m", "speed_mps", "route_duration_ticks",
            "loop_policy", "route_source", "planned_path_length_m",
        ],
        "properties": {
            "policy": {"const": "continuous_capture_ground_flow_v1"},
            "actor_kind": {"const": "pedestrian"},
            "required": {"const": True},
            "min_visible_motion_ratio": {"const": 0.85},
            "min_xy_span_m": {"const": 9.0},
            "speed_mps": {"const": 1.25},
            "route_duration_ticks": {"const": 900},
            "loop_policy": {"const": "single_pass_route_waypoints"},
            "route_source": {"const": "traffic_bundle_sidewalk"},
            "planned_path_length_m": _float_leaf(),
        },
    }


def _sensor_profile() -> dict:
    # default_uav_sensor_profile() over
    # Config/LowAltitude/uav_sensor_profile.json.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["source", "camera_name", "capture_backend", "width",
                     "height", "hfov_deg", "fixed_rotation_offset_deg"],
        "properties": {
            "source": {"const": "Config/LowAltitude/uav_sensor_profile.json"},
            "camera_name": {"const": "bottom_center"},
            "capture_backend": {"const": "editor_hook"},
            "width": {"const": 1280},
            "height": {"const": 720},
            "hfov_deg": {"const": 85.0},
            "fixed_rotation_offset_deg": {
                "type": "object", "additionalProperties": False,
                "required": ["pitch_deg", "yaw_deg", "roll_deg"],
                "properties": {
                    "pitch_deg": {"const": -90.0},
                    "yaw_deg": {"const": 0.0},
                    "roll_deg": {"const": 0.0},
                },
            },
        },
    }


def _contract_inspect_uav() -> dict:
    # contract_inspect_uav writer (regenerate_boundary_scenarios.py line
    # 5173) plus the route repair at line 8330: all 14562 rows carry the
    # same policy/motion/corridor/sensor scalars, the three route arrays are
    # equal length (11/16/36 on X rows), loop == repaired, and the code is
    # I{round(altitude)}.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "policy", "corridor_role", "corridor_policy", "motion_policy",
            "inspect_altitude_m", "inspect_altitude_code", "sensor_fov_deg",
            "sensor_fov_source", "sensor_profile",
            "observed_capture_boundary_id", "fixed_altitude_loop",
            "min_path_length_m", "inspect_fov_coverage_required",
            "planned_route_enu_m", "repaired_route_enu_m", "loop_route_enu_m",
            "repaired_path_length_m",
        ],
        "properties": {
            "policy": {"const": "semantic_event_contract_v1"},
            "corridor_role": {"const": "inspect_observer"},
            "corridor_policy": {"const": "inspect_capture_corridor"},
            "motion_policy": {
                "const": "fixed_altitude_loop_observes_capture_boundary"},
            "inspect_altitude_m": _float_leaf(),
            "inspect_altitude_code": _string_enum("I10", "I18", "I20", "I22"),
            "sensor_fov_deg": {"const": 85.0},
            "sensor_fov_source": {
                "const": "Config/LowAltitude/uav_sensor_profile.json"
                         "#hfov_deg"},
            "sensor_profile": _sensor_profile(),
            "observed_capture_boundary_id": _identifier(),
            "fixed_altitude_loop": {"const": True},
            "min_path_length_m": {"const": 80.0},
            "inspect_fov_coverage_required": {"const": True},
            "planned_route_enu_m": _waypoints(2),
            "repaired_route_enu_m": _waypoints(2),
            "loop_route_enu_m": _waypoints(2),
            "repaired_path_length_m": _float_leaf(),
        },
    }


def _lifecycle() -> dict:
    # lifecycle stamp (regenerate_boundary_scenarios.py line 7236 plus the
    # altitude repair): four true flags, the home pad id, two 3-vectors, the
    # assigned altitude and the repair marker — exactly eight keys, all
    # verified on the 18564 home-pad rows.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "auto_lifecycle_applied", "home_pad_entity_id",
            "home_hover_enu_m", "mission_start_enu_m", "requires_takeoff",
            "requires_landing_or_terminal_resolution", "assigned_altitude_m",
            "waypoint_repair_applied",
        ],
        "properties": {
            "auto_lifecycle_applied": {"const": True},
            "home_pad_entity_id": _identifier(),
            "home_hover_enu_m": _number3(),
            "mission_start_enu_m": _number3(),
            "requires_takeoff": {"const": True},
            "requires_landing_or_terminal_resolution": {"const": True},
            "assigned_altitude_m": _float_leaf(),
            "waypoint_repair_applied": {"const": True},
        },
    }


def _observer_lifecycle() -> dict:
    # observer_lifecycle stamp (regenerate_boundary_scenarios.py line 4737);
    # the boundary distance is the writer's 18.0 literal.
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "presence", "terminal_behavior",
            "maximum_terminal_boundary_distance_m", "capture_boundary_id",
        ],
        "properties": {
            "presence": {"const": "episode_full_duration"},
            "terminal_behavior": {
                "const": "airborne_hold_inside_runtime_boundary"},
            "maximum_terminal_boundary_distance_m": {"const": 18.0},
            "capture_boundary_id": _identifier(),
        },
    }


def _uav_corridor_stamp() -> dict:
    # _set_uav_altitude_metadata (regenerate_boundary_scenarios.py line
    # 7973): the layer list is UAV_ALTITUDE_LAYERS_M (uav_corridor_planner.py
    # line 27), constant on all X rows.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["assigned_altitude_m", "altitude_layers_m",
                     "used_lateral_bypass"],
        "properties": {
            "assigned_altitude_m": _float_leaf(),
            "altitude_layers_m": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": False,
                "prefixItems": [{"const": 30.0}, {"const": 50.0},
                                {"const": 80.0}],
            },
            "used_lateral_bypass": {"type": "boolean"},
        },
    }


_FACILITY_SHAPE = {
    # label_class -> (entity_kind, scope_subtype, ontology_class_id,
    # service_capacity, state const for contract-scene rows, plain
    # semantic_role).
    "barrier": ("facility.barrier", "barrier", "world:Barrier", False, None,
                None),
    "charging_pile": ("facility.charger", "charging_station",
                      "world:ChargingStation", True, None, None),
    "landing_pad": ("facility.landing_pad", "landing_pad", "world:LandingPad",
                    True, "available", "available"),
    "no_fly_zone": ("facility.no_fly_zone", "no_fly_zone", "world:NoFlyZone",
                    False, None, None),
    "radio_tower": ("facility.base_station", "communication_base_station",
                    "world:CommunicationStation", False, "online", "online"),
}
_CATEGORY = {
    "barrier": "facility", "charging_pile": "facility",
    "landing_pad": "facility", "no_fly_zone": "facility",
    "pedestrian": "pedestrian", "police_tape": "prop",
    "radio_tower": "facility", "spawn_zone": "crowd_anchor",
    "uav": "uav", "uav_corridor": "airspace_corridor", "vehicle": "vehicle",
}
_ASSET = {
    "barrier": "facility.barrier.basic",
    "charging_pile": "facility.charger.cityops.v1",
    "landing_pad": "facility.landing_pad.visible.v1",
    "no_fly_zone": "trigger.no_fly.box.v1",
    "pedestrian": "pedestrian.cityops.basic.v1",
    "police_tape": "prop.incident.police_tape.v1",
    "radio_tower": "facility.radio.base_tower.v1",
    "spawn_zone": "semantic.spawn_zone",
    "uav": ("uav.airsim.cv_pawn.v1", "uav.airsim.flying_pawn.v1",
            "uav.inspect.quad.v1"),
    "uav_corridor": "semantic.uav_corridor.segment.v1",
    "vehicle": "vehicle.emergency.ambulance.v1",
}
_ROLE_DEFAULT = {
    "barrier": "semantic_facility", "charging_pile": "semantic_facility",
    "landing_pad": "semantic_facility", "no_fly_zone": "semantic_context",
    "police_tape": "semantic_context", "radio_tower": "semantic_facility",
    "spawn_zone": "semantic_context", "uav_corridor": None,
    "uav": None, "vehicle": "semantic_vehicle", "pedestrian": None,
}


def _base_properties(label_class: str, *, posture: tuple[str, ...],
                     social: tuple[str, ...]) -> tuple[dict, list[str]]:
    """Core keys shared by every branch of one label class, plus the
    optional runtime families.  role/state_sequence/semantic_role/route are
    added here only for classes whose default role is fixed; the uav, uav
    corridor and pedestrian branches assemble their own variants."""
    properties: dict[str, dict] = {
        "tick": {"type": "integer", "minimum": 0},
        "entity_id": _identifier(),
        "label_class": {"const": label_class},
        "asset_id": ({"enum": list(_ASSET[label_class])}
                     if isinstance(_ASSET[label_class], tuple)
                     else {"const": _ASSET[label_class]}),
        "category": {"const": _CATEGORY[label_class]},
        "pos_enu": _number3(),
        "vel_mps": _number3(),
        "yaw_deg": _float_leaf(),
        "activation_tick": {"type": "integer", "minimum": 0},
        "activity_type": _state_rule(label_class),
        "animation_hint": _identifier(),
        "posture": ({"const": "standing"} if posture == ("standing",)
                    else _string_enum(*posture)),
        "social_state": ({"const": "solo"} if social == ("solo",)
                         else _string_enum(*social)),
        "state": _state_rule(label_class),
        **_optional_runtime_families(),
    }
    required = ["tick", "entity_id", "label_class", "asset_id", "category",
                "pos_enu", "vel_mps", "yaw_deg", "activation_tick",
                "activity_type", "animation_hint", "posture", "social_state",
                "state"]
    role = _ROLE_DEFAULT[label_class]
    if label_class != "uav_corridor":
        # _row_for_entity stamps task_id on every label but the corridor
        # rows (verified: present on all rows of the ten other classes).
        properties["task_id"] = _identifier()
        required.append("task_id")
    if role is not None:
        properties["role"] = {"const": role}
        properties["state_sequence"] = _state_sequence_rule(label_class)
        properties["semantic_role"] = _semantic_role_rule(label_class)
        # Every non-corridor, non-uav class writes an empty route list on X
        # rows (verified per class).
        properties["route_waypoints_enu_m"] = {"type": "array",
                                               "maxItems": 0, "items": False}
        required += ["role", "state_sequence", "semantic_role",
                     "route_waypoints_enu_m"]
    if label_class in _FACILITY_SHAPE:
        entity_kind, subtype, ontology, capacity, _, _ = _FACILITY_SHAPE[
            label_class]
        properties["entity_kind"] = {"const": entity_kind}
        properties["semantic_scope"] = _semantic_scope(subtype, ontology,
                                                       capacity)
        required += ["entity_kind", "semantic_scope"]
    return properties, required


def _state_rule(label_class: str) -> dict:
    if label_class == "pedestrian":
        return _string_enum(*_PED_STATES)
    return _string_enum(*_STATE_ENUMS[label_class])


def _state_sequence_rule(label_class: str) -> dict:
    """state_sequence for the fixed-role classes: single step, class state
    vocabulary (state_sequence[0] diverges from state only on the 901
    event-actor rows, whose items stay in the same vocabulary)."""
    return {
        "type": "array", "minItems": 1, "maxItems": 1,
        "items": _state_rule(label_class),
    }


def _semantic_role_rule(label_class: str) -> dict:
    """semantic_role for the fixed-role classes; the facility contract
    sentence rides on the contract-scene branches."""
    if label_class in ("barrier", "charging_pile"):
        return {"const": _FACILITY_CONTRACT_SENTENCE}
    if label_class == "radio_tower":
        return _string_enum("online", _FACILITY_CONTRACT_SENTENCE)
    if label_class == "landing_pad":
        return _string_enum("available", _FACILITY_CONTRACT_SENTENCE)
    return _string_enum(*_SEMANTIC_ROLE_ENUMS[label_class])


def _branch(label_class: str, *, posture: tuple[str, ...] = ("standing",),
            social: tuple[str, ...] = ("solo",),
            extra_properties: dict | None = None,
            extra_required: tuple[str, ...] = ()) -> dict:
    properties, required = _base_properties(label_class, posture=posture,
                                            social=social)
    if extra_properties:
        properties.update(extra_properties)
    return {
        "type": "object", "additionalProperties": False,
        "required": [*required, *extra_required],
        "properties": properties,
    }


def _facility_contract_branch(label_class: str) -> dict:
    """Contract-scene facility row: contract stamp present, semantic_role
    fixed to the contract sentence, state fixed to the stamped value."""
    _, _, _, _, state_const, _ = _FACILITY_SHAPE[label_class]
    extras: dict[str, dict] = {
        "contract_facility": _contract_facility(),
        "semantic_role": {"const": _FACILITY_CONTRACT_SENTENCE},
    }
    branch = _branch(label_class, extra_properties=extras,
                     extra_required=("contract_facility",))
    if state_const is not None:
        branch["properties"]["state"] = {"const": state_const}
        branch["properties"]["activity_type"] = {"const": state_const}
    return branch


def _facility_plain_branches(label_class: str) -> list[dict]:
    """Runtime-side facility rows as two complete, mutually exclusive root
    branches: rows carrying the allocation or link family (required, so the
    bare side cannot match) and rows carrying neither it nor the contract
    stamp — the writer leaves both off when no patch reached the facility
    in the window."""
    _, _, _, _, _, plain_role = _FACILITY_SHAPE[label_class]
    family = ("facility_state" if label_class == "landing_pad"
              else "communication_state")
    family_side = _branch(label_class, extra_properties={
        family: _runtime_state_family_patch(family),
        "semantic_role": {"const": plain_role},
    }, extra_required=(family,))
    # The bare side must exclude the family outright or the exactly-one
    # oneOf would accept family rows twice.
    bare_side = _branch(label_class, extra_properties={
        "semantic_role": {"const": plain_role},
    })
    bare_side["properties"].pop(family)
    return [family_side, bare_side]


def _event_actor_branches(label_class: str) -> list[dict]:
    """_mark_event_actor (regenerate_boundary_scenarios.py line 1416) sets
    activation_tick to 901 together with spawn_policy; the plain side keeps
    tick 0 and no stamp.  Two complete branches.  On X rows every
    ``police_tape`` row is stamped (180 of 180) and every ``no_fly_zone``
    standdown row is stamped (525 of 1701); the plain side is hit only by
    ``no_fly_zone`` idle rows and remains 尚无实例验证 for police_tape."""
    stamped = _branch(label_class, extra_properties={
        "activation_tick": {"const": 901},
        "spawn_policy": {"const": _EVENT_SCRIPT_ONLY},
    }, extra_required=("spawn_policy",))
    plain = _branch(label_class, extra_properties={
        "activation_tick": {"const": 0},
    })
    return [stamped, plain]


def _uav_branches() -> list[dict]:
    """Three complete uav branches: the U_inspect contract row and the two
    mission_uav lifecycle variants.  The altitude enums keep the producer's
    layer table; instance coverage per branch is noted in the docstring."""
    common, _ = _base_properties("uav", posture=("standing",),
                                 social=("solo",))
    inspect_properties = {
        **common,
        "role": {"const": "U_inspect"},
        "state_sequence": {"type": "array", "minItems": 4, "maxItems": 4,
                           "items": {"const": "inspect_racetrack"},
                           "prefixItems": [{"const": "takeoff"},
                                           {"const": "inspect_racetrack"},
                                           {"const": "orbit"},
                                           {"const": "land"}]},
        "semantic_role": {"const": "long-lived UAV inspect view replacing "
                                   "high overview"},
        "state": {"const": "inspect_racetrack"},
        "activity_type": {"const": "inspect_racetrack"},
        "animation_hint": {"const": "inspect_racetrack"},
        "ground_reference_z_m": {"const": 0.0},
        "ground_reference_source": {"const": "map_ground_reference_v1"},
        "planned_route_waypoints_enu_m": _waypoints(2),
        "route_waypoints_enu_m": _waypoints(2),
        "uav_corridor": _uav_corridor_stamp(),
        "uav_corridor_role": {"const": "inspect_observer"},
        "assigned_altitude_m": {"type": "number",
                                "enum": [10.0, 18.0, 20.0, 22.0],
                                "x-source-value-types": ["float"]},
        "min_path_length_m": {"const": 80.0},
        "inspect_altitude_code": _string_enum("I10", "I18", "I20", "I22"),
        "contract_inspect_uav": _contract_inspect_uav(),
        "full_episode_presence": {"const": True},
    }
    mission_keys = {
        "role": {"const": "mission_uav"},
        "ground_reference_z_m": _float_leaf(),
        "ground_reference_source": _string_enum("home_hover_pose_origin_v1",
                                                "map_ground_reference_v1"),
        "planned_route_waypoints_enu_m": _waypoints(2),
        "route_waypoints_enu_m": _waypoints(2),
        "uav_corridor": _uav_corridor_stamp(),
        # The producer's layer table; X instances: 80.0 on observers, 30/50
        # on home-pad missions (尚无实例验证 for the other combinations).
        "assigned_altitude_m": {"type": "number",
                                "enum": [30.0, 50.0, 80.0],
                                "x-source-value-types": ["float"]},
    }
    observer_properties = {
        **common, **mission_keys,
        "uav_corridor_role": {"const": "observer"},
        "state_sequence": {"type": "array", "minItems": 1, "maxItems": 1,
                           "items": {"const": "corridor_observer"}},
        "semantic_role": {"const": "corridor_observer"},
        "state": {"const": "corridor_observer"},
        "activity_type": {"const": "corridor_observer"},
        "animation_hint": {"const": "corridor_observer"},
        "observer_lifecycle": _observer_lifecycle(),
    }
    mission_states = [s for s in _STATE_ENUMS["uav"] if s != "corridor_observer"]
    mission_properties = {
        **common, **mission_keys,
        "uav_corridor_role": {"const": "mission_uav"},
        "state_sequence": {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": _string_enum("landing_request", "mission", "monitor",
                                  "patrol", "rain_patrol"),
        },
        "semantic_role": _string_enum("landing_request", "mission",
                                      "monitor", "patrol", "rain_patrol"),
        "state": _string_enum(*mission_states),
        "activity_type": _string_enum(*mission_states),
        "lifecycle": _lifecycle(),
    }
    branches = []
    for required_extra, properties in (
        (("contract_inspect_uav", "full_episode_presence",
          "inspect_altitude_code", "min_path_length_m"),
         inspect_properties),
        (("observer_lifecycle",), observer_properties),
        (("lifecycle",), mission_properties),
    ):
        branches.append({
            "type": "object", "additionalProperties": False,
            "required": [
                "tick", "entity_id", "label_class", "asset_id", "category",
                "pos_enu", "vel_mps", "yaw_deg", "activation_tick",
                "activity_type", "animation_hint", "posture", "social_state",
                "state", "task_id", "role", "state_sequence", "semantic_role",
                "ground_reference_z_m", "ground_reference_source",
                "planned_route_waypoints_enu_m", "route_waypoints_enu_m",
                "uav_corridor", "uav_corridor_role", "assigned_altitude_m",
                *required_extra,
            ],
            "properties": properties,
        })
    return branches


def _pedestrian_branches() -> list[dict]:
    """Background rows and four disjoint plain/incident family shapes.

    Posture and social_state follow the catalog mapping for each branch's
    state set.
    """
    bg = _branch("pedestrian", posture=("standing",), social=("group", "solo"),
                 extra_properties={
                     "background_pedestrian": _background_pedestrian(),
                     "background_role": {"const": "semantic_context"},
                     "ground_flow_contract": _ground_flow_contract(),
                     "role": {"const": "semantic_background_pedestrian"},
                     "semantic_role": _identifier(),
                     "state_sequence": {
                         "type": "array", "minItems": 2, "maxItems": 2,
                         "items": {"const": "walking"}},
                     "route_waypoints_enu_m": _waypoints(1),
                 },
                 extra_required=("background_pedestrian", "background_role",
                                 "ground_flow_contract", "role",
                                 "state_sequence", "semantic_role",
                                 "route_waypoints_enu_m"))
    plain = _branch("pedestrian", posture=("standing",),
                    social=("group", "observer", "solo"),
                    extra_properties={
                        "role": {"const": "semantic_pedestrian"},
                        "semantic_role": _string_enum("observing", "standing",
                                                      "waiting"),
                        "state_sequence": {
                            "type": "array", "minItems": 1, "maxItems": 1,
                            "items": _string_enum("observing", "standing",
                                                  "waiting")},
                        "route_waypoints_enu_m": {"type": "array",
                                                  "maxItems": 0,
                                                  "items": False},
                    },
                    extra_required=("role", "state_sequence", "semantic_role",
                                    "route_waypoints_enu_m"))
    # Partition plain rows by family presence.  Each single-family branch
    # requires its key and excludes the other; the bare branch excludes both.
    # The incident branch requires both keys.  X3's schedule stamps both on
    # ped_x3_fall from tick 0 and patches them on later incident events.
    plain["properties"].pop("incident_state")
    plain["properties"].pop("pedestrian_state")
    single_family = [{
        **plain,
        "required": [*plain["required"], family],
        "properties": {
            **plain["properties"],
            family: _runtime_state_family_patch(family),
        },
    } for family in ("incident_state", "pedestrian_state")]
    incident = _branch("pedestrian", posture=("fallen", "standing"),
                       social=("solo",),
                       extra_properties={
                           **plain["properties"],
                           "posture": _string_enum("fallen", "standing"),
                           "social_state": {"const": "solo"},
                           "state": _string_enum("medical_incident",
                                                 "walking", "waiting"),
                           "activity_type": _string_enum("medical_incident",
                                                         "walking",
                                                         "waiting"),
                       },
                       extra_required=("role", "state_sequence",
                                       "semantic_role",
                                       "route_waypoints_enu_m",
                                       "incident_state",
                                       "pedestrian_state"))
    return [bg, plain, *single_family, incident]


def _uav_corridor_branch() -> dict:
    """Corridor rows: no role/task_id/state_sequence/semantic_role/route,
    the fixed semantic_corridor activity block and zero velocity."""
    return _branch("uav_corridor", extra_properties={
        "state": {"const": "semantic_corridor"},
        "activity_type": {"const": "semantic_corridor"},
        "animation_hint": {"const": "semantic_corridor"},
        "vel_mps": {"type": "array", "minItems": 3, "maxItems": 3,
                    "items": {"const": 0.0}},
    })


def arm_states_schema() -> dict:
    """Return the closed record schema for one ``arm_states`` X JSONL row.

    The root is a flat oneOf of complete-row branches, mutually
    exclusive by ``label_class`` plus the branch-required stamps: the uav
    branches by ``role`` and the lifecycle/contract keys, the pedestrian
    branches by the background triple and the two family presence flags, the
    ``landing_pad``/``radio_tower`` branches by ``contract_facility`` versus
    the runtime-side family, and the ``no_fly_zone``/``police_tape`` pairs by
    ``spawn_policy`` plus the pinned activation tick.  Every label class
    accepts the ten runtime-state families.  The single-family pedestrian
    branches and the police_tape plain side stay 尚无实例验证, together
    with the other producer-allowed forms listed in the module docstring.
    The ``asset_anchor`` label has no branch because no X producer path
    writes it.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_states_x.record.schema.json",
        "title": "X ARM raw state record",
        "oneOf": [
            _facility_contract_branch("barrier"),
            _facility_contract_branch("charging_pile"),
            _facility_contract_branch("landing_pad"),
            *_facility_plain_branches("landing_pad"),
            *_event_actor_branches("no_fly_zone"),
            *_pedestrian_branches(),
            *_event_actor_branches("police_tape"),
            _facility_contract_branch("radio_tower"),
            *_facility_plain_branches("radio_tower"),
            _branch("spawn_zone"),
            *_uav_branches(),
            _uav_corridor_branch(),
            _branch("vehicle"),
        ],
    }
