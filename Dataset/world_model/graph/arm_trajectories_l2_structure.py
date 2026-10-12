"""Closed record structure for L2 ARM raw trajectory rows.

Source family ``arm_trajectories``; profiles L2-1_v1, L2-1_v2, L2-2_v1,
L2-2_v2, L2-3_v1, L2-3_v2, L2-4_v1, L2-4_v2, L2-5_v1.  Producer:
``Dataset/tools/l2_window.py::execute_engine`` runs
``EpisodeStateEngine`` from tick 0, and ``_execute`` writes every engine row
in ``[start_tick, end_tick]`` whose ``entity_id`` is in the declared study set
verbatim to ``raw/trajectories.jsonl`` via ``write_rows`` (``sorted-keys``
compact JSON).  L2 factual windows record engine rows, not frozen parent
capture rows, so there is exactly one record branch — the engine branch of
``EpisodeStateEngine._row_for_entity`` (``Dataset/tools/batch_generate.py``):
the fixed base keys ``tick/entity_id/label_class/asset_id/pos_enu/vel_mps/
yaw_deg/state``, ``row_activity_payload``'s unconditional ``activity_type/
animation_hint/posture/social_state`` (non-pedestrian stamp ``standing``/
``solo``), then ``preserved_fields_from(entity)``.

``preserved_fields_from`` copies a field only when its value is neither
``None`` nor ``""`` — from the entity, or from the entity's nested
``initial_state``/``visual_state`` — so every preserved key is a conditional
slot.  The engine entity carries ``PRESERVED_ENTITY_FIELDS`` copied from the
L2 scene setups, so the row space closes to what the nine L2 scene files
actually author on the configured study entities:

- mission UAVs (``category: uav``): ``activation_tick/category/role/
  state_sequence/semantic_role/task_id/route_waypoints_enu_m/
  planned_route_waypoints_enu_m/uav_corridor/uav_corridor_role/
  assigned_altitude_m/ground_reference_z_m/ground_reference_source/lifecycle``
  and, on the L2-2 GNSS-anomaly UAV, the scene's
  ``path_deviation_contract``;
- facilities with a typed scope (radio tower, charger, landing pad): the
  preserved base plus ``entity_kind/semantic_scope`` and the
  ``facility_state``/``communication_state`` families the scene authors;
- the traffic light (untyped facility prop): preserved base plus
  ``facility_state`` only;
- event actors (``spawn_policy: event_script_only`` vehicles): preserved base
  plus ``spawn_policy/event_actor_motion_contract`` and, on the police unit,
  the ``incident_state`` patch slot from the scene.

Runtime-state families: scene ``initial_state``/top-level values enter
through ``preserved_fields_from`` -> ``runtime_state_fields_from_sources``;
intervention and retained event-script ``set_runtime_state`` actions patch
the entity through ``_apply_pending_runtime_state_patches`` +
``deep_merge_mapping``, which can create a family the scene never declared
(navigation_state on the L2-4 divert UAV) or merge extra boolean keys into an
existing one (the four-key communication_state on the L2-1 recovery arm).
Each family is therefore an optional row slot closed against the executable
contract in ``Dataset/tools/runtime_state_contract.py``: the ten declared
families, their consumed fields, boolean/command value rules, facility
requester lists and non-negative numbers, plus the ``rule_id``/
``mechanism_id`` metadata strings.  The five families L2 sources actually
author (``communication_state``, ``control_state``, ``facility_state``,
``navigation_state``, ``incident_state``) are modeled from the current
production contract and scoped sources.  ``mission_state``, ``security_state``,
``sensor_state``, ``vehicle_state`` and ``pedestrian_state`` are
producer-allowed (the dispatcher accepts ``set_runtime_state`` rows for any
declared family) but no L2 scene, event script or arm configuration writes
them — 尚无实例验证.  Enum membership and case normalisation stay
producer-side; the raw grammar keeps the written string.

Producers never emit ``event_phases``/``sumo_*``/``uav_visibility``/pedestrian
payloads on these rows (L2 study entities never carry them and
``set_pedestrian_activity`` is absent from every L2 scene and script).  No
row carries an integrity or digest field, so no publication-exclusion path is
required.  ``activation_tick`` is the authored scene value (the police unit
keeps its ``event_script_only`` sentinel 901 inside a tick-341 window);
``route_waypoints_enu_m``/``planned_route_waypoints_enu_m`` are producer
pass-throughs of the authored arrays (empty lists are a declared status;
waypoint members stay positional scene triples — x/y authored floats, z
authored float, except the L2-2 planned route whose z the scene writes as
int).  Motion semantics, units and event truth are later-stage concerns.
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
    _CONTROL_ENUMS,
    _MISSION_ENUMS,
    _SECURITY_ENUMS,
)
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
    runtime_state_source_guard_rule,
)

__all__ = ["arm_trajectories_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# row_activity_payload stamps these two verbatim for every non-pedestrian
# row; L2 study entities are never pedestrians.
_POSTURE = "standing"
_SOCIAL_STATE = "solo"

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


def _waypoints(member: dict) -> dict:
    # Producer pass-through of the authored route; empty lists are a declared
    # structural status for this family.  Members are the scene's positional
    # x/y/z triples.
    return {"type": "array", "items": member}


def _planned_route_waypoints() -> dict:
    # Same producer pass-through; the L2-2 scene authors the planned route's
    # z component as int (the only L2 int waypoint member, 6012 of 189780
    # planned waypoints), while x/y stay float.
    member = {
        "type": "array", "minItems": 3, "maxItems": 3,
        "prefixItems": [
            {"type": "number", "x-source-value-types": ["float"]},
            {"type": "number", "x-source-value-types": ["float"]},
            {"type": "number"},
        ],
        "items": False,
    }
    return {"type": "array", "items": member}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state family field, from the executable
    producer contract (``validate_runtime_state_patch`` plus
    ``invalid_runtime_state_value_paths`` in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return nonblank_python_string_rule()
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
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return runtime_enum_string_rule(RUNTIME_STATE_ENUM_FIELDS[(family, field)])
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
    """Closed rule per executable family; deep-merge patches may create or
    extend any of them, so each stays an optional row slot."""
    return {family: _runtime_state_family_patch(family)
            for family in RUNTIME_STATE_FIELDS}


def _lifecycle() -> dict:
    # Eight-key object built from the scene lifecycle contract and the engine
    # auto-lifecycle stamps; identical on all 23250 UAV rows.
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
    # Engine carries the scene's uav_corridor verbatim; L2 scenes author
    # exactly three altitude layers, and 23250/23250 rows carry three floats.
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
    # Scene facility_scope contract.  service_capacity rides on the scoped
    # pads and chargers, not on the radio tower (1308 four-key rows vs
    # 14145 five-key rows), so it stays optional.
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


def _path_deviation_contract() -> dict:
    # L2-2 scene authoring copied verbatim onto the GNSS-anomaly UAV rows.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["minimum_route_distance_m", "predicate_id"],
        "properties": {
            "minimum_route_distance_m": _float_leaf(),
            "predicate_id": _identifier(),
        },
    }


def _event_actor_motion_contract() -> dict:
    """Event-actor motion contract closed per ``policy``.  The producer
    builders write exactly two shapes: ``regenerate_boundary_scenarios.event_actor_entry``
    updates the staged contract with the eight entry keys under
    ``policy: offstage_then_continuous_lane_entry``, while
    ``spawn_event_actor_at_dispatch`` stages the six shared keys only under
    ``policy: offstage_until_semantic_dispatch``; the L2-5 scene authors both
    verbatim and the rows confirm the key sets (civilian carriers always
    14 keys, the police unit always 6)."""
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
        "label_class": _identifier(),
        "asset_id": _identifier(),
        "pos_enu": _float3(),
        "vel_mps": _float3(),
        "yaw_deg": _float_leaf(),
        "state": _identifier(),
        "activity_type": _identifier(),
        "animation_hint": _identifier(),
        "posture": {"const": _POSTURE},
        "social_state": {"const": _SOCIAL_STATE},
        # Preserved slots: written only when the scene value is neither None
        # nor "" (preserved_fields_from), so each stays optional.
        "activation_tick": {"type": "integer", "minimum": 0},
        "assigned_altitude_m": _float_leaf(),
        "category": _identifier(),
        "entity_kind": _identifier(),
        "ground_reference_source": _identifier(),
        "ground_reference_z_m": _float_leaf(),
        "lifecycle": _lifecycle(),
        "planned_route_waypoints_enu_m": _planned_route_waypoints(),
        "role": _identifier(),
        "route_waypoints_enu_m": _waypoints(_float3()),
        "semantic_role": _identifier(),
        "semantic_scope": _semantic_scope(),
        "spawn_policy": _identifier(),
        "state_sequence": {"type": "array", "minItems": 1,
                           "items": _identifier()},
        "task_id": _identifier(),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": _identifier(),
        "path_deviation_contract": _path_deviation_contract(),
        "event_actor_motion_contract": _event_actor_motion_contract(),
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
    """Compile the closed L2 raw trajectory record schema.

    One record branch: the ``EpisodeStateEngine._row_for_entity`` shape
    ``l2_window._execute`` writes for both factual and intervention arms.
    Required are the twelve always-written base keys; every preserved slot is
    optional because ``preserved_fields_from`` drops ``None``/"" values.  The
    five L2-authored runtime families (``communication_state``,
    ``control_state``, ``facility_state``, ``navigation_state``,
    ``incident_state``) are modeled from the current production contract
    and scoped sources; ``mission_state``, ``security_state``, ``sensor_state``,
    ``vehicle_state`` and ``pedestrian_state`` remain producer-allowed
    without an L2 instance (尚无实例验证).
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_trajectories_l2.record.schema.json",
        "title": "L2 ARM raw trajectory record",
        **_row(),
    }
