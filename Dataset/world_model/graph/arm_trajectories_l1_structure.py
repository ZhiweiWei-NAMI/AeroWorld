"""Closed record structure for L1 ARM raw trajectory rows.

Source family ``arm_trajectories``; profiles L1-1_v1, L1-1_v2, L1-2_v1,
L1-3_v1, L1-3_v2, L1-4_v1, L1-4_v2 (84 declared JSONL files, 38256 present
rows; 42 factual + 42 intervention arms across 3 seeds per profile).
Producer: ``Dataset/tools/l1_window.py::execute_config`` resolves one
``trajectories`` list and writes the identical list object to
``raw/states.jsonl`` and ``raw/trajectories.jsonl`` (l1_window.py:457-458);
the two files are byte-identical on all 84 declared L1 windows (``cmp``:
compared=84, differing=0), so this schema and
``arm_states_l1_structure.arm_states_schema`` close the same record space
from the same producer anchor.  This module declares the
``arm_trajectories`` family contract with its own ``$id`` instead of
aliasing that module: sibling reuse below the layer boundary is
documentation, not an import dependency, and the registry delivers one
schema constructor per source family.

Rows come from exactly two record shapes (three producer forms).

Formal-capture form (``mode: factual_observation`` without pre-activation
fill, 42 of 42 factual windows): the frozen parent ``trajectories.jsonl`` of
``aw_data/render_ready_episodes_capture_filtered/<episode>/`` restricted to
``config["entity_ids"]`` and to ``[start_tick, end_tick]``.  That frozen file
is written by ``Dataset/tools/filter_render_ready_truth_for_capture.py::
trajectory_row_from_entity`` (:395): the fixed base keys
``tick/frame_id/sim_time_s/entity_id/label_class/asset_id/entity_category/
entity_kind/entity_type/pos_enu/vel_mps/yaw_deg/state/activity_type/category``
(``tick`` unconditionally, :398), then ``preserve_keys`` copies present
entity keys (:415-433), and a final projection drops ``None`` values.
``role/semantic_role/task_id`` are present on every scoped L1 entity and
therefore on every formal row.  ``pos_enu``/``vel_mps`` come from
``position_enu_from_entity`` (:129-141) and ``velocity_enu_from_entity``
(:197-205), which cast every component with ``float(...)`` before the row is
written, so both writer paths emit float source values only — the corpus
confirms all 38256 rows carry float components on both branches.  Those two
producer-cast vectors are declared with ``x-source-value-types: ["float"]``
per component; the JSON type stays ``number``.  Scene-authored route and
waypoint arrays are verbatim pass-throughs and stay permissive (the
structural catalog declares ``(None, 3)`` axes with int|float elements and
an allowed empty list); the same holds for the lifecycle pose vectors.

``uav_visibility`` is written for every ``uav`` truth entity by
``Dataset/tools/convert_to_render_ready.py::
uav_camera_capture_visibility_payload`` (:3018-3038) with the two producer
policy constants.  The payload writes ``roi_capture_distance_m`` as
``round(float(roi_distance_m), 6) if math.isfinite(float(roi_distance_m))
else None`` (:3030-3032), but the serializer's nonfinite branch is
unreachable for every record that reaches ARM raw: both converter paths
that call the payload gate the position on
``distance_to_polygon_xy(...) <= RUNTIME_BOUNDARY_PADDING_M`` (60.0, :158)
first — the local UAV path via ``source_entity_visible_at_tick``
(convert_to_render_ready.py:3619, applied per visible tick at :5138) and
the global UAV path via ``point_in_runtime_boundary_for_contract``
(:5386-5389) — and a non-finite distance cannot pass that comparison, so
``math.isfinite`` never fails at :3031 and the recorded value is always a
finite float (all ``uav_visibility`` leaves in the L1 factual corpus are
floats).  The leaf stays required with a float-only branch; finiteness is
guaranteed by the producer gate, not by the grammar.

Engine form (all 42 intervention windows, plus the pre-activation fill of a
factual window): ``EpisodeStateEngine._row_for_entity``
(Dataset/tools/batch_generate.py:2395) writes the fixed base keys
``tick/entity_id/label_class/asset_id/pos_enu/vel_mps/yaw_deg/state`` with
``pos_enu``/``vel_mps`` as ``[round(float(value), 6)]`` per component
(batch_generate.py:2408-2409), plus ``row_activity_payload``
(batch_generate.py:1856) unconditional
``activity_type/animation_hint/posture/social_state``;
``preserved_fields_from(entity)`` (batch_generate.py:2414) then adds a
preserved slot only for fields whose value is neither ``None`` nor ``""``.
The engine entity carries ``PRESERVED_ENTITY_FIELDS`` copied from the scene
setup (``build_entities``, batch_generate.py:1968), so the L1 branch closes
to the preserved slots the seven L1 scene files actually declare on the
configured ``entity_ids``:
``category/role/semantic_role/state_sequence/task_id/uav_corridor_role/
uav_corridor/assigned_altitude_m/ground_reference_z_m/ground_reference_source/
lifecycle/activation_tick/route_waypoints_enu_m/planned_route_waypoints_enu_m``,
the runtime-state families the scene entities declare (``control_state`` on
every mission UAV, ``sensor_state`` on the L1-3 primaries), and the audited
``event_actor_motion_contract`` slot below.  The remaining
``PRESERVED_ENTITY_FIELDS`` slots (``semantic_scope``/``entity_kind``/
``background_*``/``contract_*``/``inspect_*``/``full_episode_presence``/
``observer_lifecycle``/``deactivation_tick``/``spawn_policy``/
``path_deviation_contract``/``task_kind``/``task_state``) stay undeclared:
no L1 scene authors them on a scoped entity and no L1 writer path creates
them.

``event_actor_motion_contract``: ``PRESERVED_ENTITY_FIELDS``
(batch_generate.py:215-253) includes it and ``preserved_fields_from`` copies
a scene-authored value verbatim, so an L1 scene that authors the contract
reaches engine rows unchanged.  No current L1 producer path writes it: the
seven L1 scene files carry none, the L1 event scripts' ``set_visual_state``
payloads are ``mode``-only, and ``_handle_spawn_entity``
(batch_generate.py:2321) writes only
``pos/active_from/inactive_from/spawned/asset_id/schedules``.  The slot is
therefore declared from the producer's own contract shapes with no L1
instance (尚无实例验证).  Structural authority:
``regenerate_boundary_scenarios.stage_event_actor`` (:1425-1433) stages the
six shared keys and ``event_actor_entry`` updates the staged contract with
the eight entry keys, both stamping ``schema:
event_actor_motion_contract_v1``; the 23 scene instances confirm exactly the
two policy-keyed shapes closed here (the same two shapes
``arm_trajectories_l2_structure`` closes for the same producer builders).

Pre-activation lifecycle fill (l1_window.py:417-428): a factual window whose
scoped entity has no capture row before its own activation tick takes those
pre-spawn samples from the engine's own pre-action sampling and writes
``sorted(parent + covered)`` (l1_window.py:423-427).  The producer admits it and the writer emits it
to ``raw/trajectories.jsonl`` unchanged, so it is declared, but it carries no
current instance: all 84 ``objective/window_status.json`` files record
``lifecycle_coverage.pre_activation_samples_filled_from_engine: 0`` and no L1
``raw/trajectories.jsonl`` mixes the two record shapes (per-file census: 42
formal-only + 42 engine-only).

Runtime-state families: the L1 window dispatcher registers all eight
``DISPATCH`` handlers as source-script handlers (l1_window.py:224-225), the
seven L1 scene ``event_script.json`` files author 28 ``set_runtime_state``
actions (census 2+2+2+3+3+8+8 over L1-1_v1 .. L1-4_v2) applied by
``EpisodeStateEngine._apply_pending_runtime_state_patches`` and value-gated
by ``validate_runtime_state_patch`` (batch_generate.py:849), and the L1
entry plans under ``scripts/arms/L1-*/`` declare no direct actions (omission
plans with empty ``actions``).  Preserved scene families enter through
``runtime_state_fields_from_sources`` and a merged patch can create or
extend any declared family, so each of the ten families is an optional row
slot closed to the executable producer contract
(``Dataset/tools/runtime_state_contract.py``): consumed fields, boolean/
command/enum value rules, facility requester lists and non-negative numbers,
communication availability bounds, incident hazard bounds, plus the
``rule_id``/``mechanism_id`` metadata strings.  The L1 corpus carries
exactly ``control_state`` (24868 rows) and ``sensor_state`` (5844 rows);
their boolean members ``rth_active``, ``alternate_rth_active``,
``altitude_corrective_maneuver_active``, ``deconfliction_active``,
``evasion_active``, ``resequence_active``, ``divert_active`` and
``sensor_state.intruder_detected`` are instance-verified with both ``True``
and ``False`` values.  The non-boolean members the same handler accepts
(``control_state`` ``mode/active_mode/command_mode/active_command/
active_commands/commands`` and ``sensor_state``
``exposure_status/imaging_mode``, with the command fields as
string-or-string-list and enum membership checked producer-side) are
declared from the same executable contract with no L1 instance, as are the
other eight families — 尚无实例验证.

Boundary: an L1 window records only rows of the configured ``entity_ids``
(all mission UAVs in the current corpus, plus the L1-3 intruders).  SUMO and
global-flow payloads (``sumo_segment``, ``sumo_vehicle``, ``sumo_visibility``,
``uav_segment``, ``uav_global_flow``) are written by the frozen writer only
for those entity classes, carry no L1 instance, and their owned structures
belong to the layers that produce them; they are not declared here with
weakened placeholders.  Neither record form carries an integrity or digest
field, so no publication exclusion is required.  Registry wiring into
``structure_closure.py`` belongs to the integration stage; this module only
exports the pure schema constructor and scans no data at import.
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

# convert_to_render_ready.uav_camera_capture_visibility_payload writes these
# two policy stamps verbatim from module constants.
_CAPTURE_ROI_POLICY = "camera_only_while_uav_position_inside_capture_roi_polygon_v1"
_CAMERA_CALL_POLICY = "do_not_call_camera_when_roi_capture_eligible_is_false"

_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _writer_float3() -> dict:
    # Both writer paths cast every component with float(...) before writing —
    # engine: round(float(value), 6) (batch_generate.py:2408-2409); frozen:
    # position_enu_from_entity/velocity_enu_from_entity float casts
    # (filter_render_ready_truth_for_capture.py:129-141, 197-205).  The raw
    # corpus confirms float components on all 38256 rows.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


def _number3() -> dict:
    # Verbatim scene pass-through (lifecycle pose vectors); the structural
    # catalog keeps (None, 3) axes with int|float elements.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _waypoints() -> dict:
    # Producer pass-through of the authored route; empty lists are a declared
    # structural status for this family.  Members stay the scene's positional
    # triples without a narrowed source type.
    return {"type": "array", "items": _number3()}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state patch field, from the executable
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
    """Closed rule per executable family.

    The L1 window dispatcher admits ``set_runtime_state`` patches for any
    declared family (source-script handler, l1_window.py:59) and scene
    entities can preserve any family, so all ten families use the same
    executable value contract.  ``control_state`` and ``sensor_state`` are
    L1-verified for their boolean members; the non-boolean contract members
    and the other eight families have no L1 instance.
    """
    return {family: _runtime_state_family_patch(family)
            for family in RUNTIME_STATE_FIELDS}


def _event_actor_motion_contract() -> dict:
    """Event-actor motion contract closed per ``policy``.

    The producer builders write exactly two shapes:
    ``regenerate_boundary_scenarios.event_actor_entry`` updates the staged
    contract with the eight entry keys under
    ``policy: offstage_then_continuous_lane_entry``, while
    ``stage_event_actor`` stages the six shared keys only under
    ``policy: offstage_until_semantic_dispatch``; the 23 scene instances
    confirm the key sets.  No L1 scene or L1 writer path produces the field,
    so this slot has no L1 instance.
    """
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


def _uav_visibility() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "camera_call_policy", "capture_roi_policy",
            "inspect_observation_distance_m", "roi_capture_distance_m",
            "roi_capture_eligible", "selected_for_capture_truth",
        ],
        "properties": {
            "camera_call_policy": {"const": _CAMERA_CALL_POLICY},
            "capture_roi_policy": {"const": _CAPTURE_ROI_POLICY},
            "inspect_observation_distance_m": _float_leaf(),
            # uav_camera_capture_visibility_payload writes round(float(d), 6)
            # if math.isfinite(d) else None (:3030-3032), but both converter
            # paths gate the position on distance_to_polygon_xy(...) <= 60.0
            # before the payload call (local UAV: :3619 + :5138; global UAV:
            # :5386-5389), so a non-finite distance never reaches :3031 and
            # the null branch is unreachable for ARM sources.  Required
            # non-null float; the producer gate and validate_record's strict
            # JSON number check both reject non-finite values.
            "roi_capture_distance_m": _float_leaf(),
            "roi_capture_eligible": {"type": "boolean"},
            "selected_for_capture_truth": {"type": "boolean"},
        },
    }


def _lifecycle() -> dict:
    # Eight-key object built from the scene lifecycle contract and the engine
    # auto-lifecycle stamps; identical on all 19128 engine rows.
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
            "home_hover_enu_m": _number3(),
            "home_pad_entity_id": _identifier(),
            "mission_start_enu_m": _number3(),
            "requires_landing_or_terminal_resolution": {"type": "boolean"},
            "requires_takeoff": {"type": "boolean"},
            "waypoint_repair_applied": {"type": "boolean"},
        },
    }


def _uav_corridor() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["altitude_layers_m", "assigned_altitude_m", "used_lateral_bypass"],
        "properties": {
            "altitude_layers_m": {"type": "array", "items": _float_leaf()},
            "assigned_altitude_m": _float_leaf(),
            "used_lateral_bypass": {"type": "boolean"},
        },
    }


def _engine_row() -> dict:
    families = _runtime_state_families()
    properties = {
        "activity_type": _identifier(),
        "animation_hint": _identifier(),
        "asset_id": _identifier(),
        # Producer identity: engine rows come from the configured L1 scoped
        # entities — mission UAVs and the L1-3 intruders
        # (``uav_l1_<n>_v<k>[_<role>]`` / ``intruder_l1_<n>_v<k>``).
        "entity_id": {"type": "string", "minLength": 1,
                      "pattern": "^(uav|intruder)_l1_[0-9]_v[0-9](_[a-z]+)?$"},
        "label_class": _identifier(),
        "pos_enu": _writer_float3(),
        "posture": _identifier(),
        "social_state": _identifier(),
        "state": _identifier(),
        "tick": {"type": "integer", "minimum": 0},
        "vel_mps": _writer_float3(),
        "yaw_deg": _float_leaf(),
        # Preserved slot: written only when the scene value is neither None
        # nor "" (preserved_fields_from), so each stays optional.
        "activation_tick": {"type": "integer", "minimum": 0},
        "assigned_altitude_m": _float_leaf(),
        "category": _identifier(),
        "control_state": families["control_state"],
        "ground_reference_source": _identifier(),
        "ground_reference_z_m": _float_leaf(),
        "lifecycle": _lifecycle(),
        "planned_route_waypoints_enu_m": _waypoints(),
        "role": _identifier(),
        "route_waypoints_enu_m": _waypoints(),
        "semantic_role": _identifier(),
        "sensor_state": families["sensor_state"],
        # Preserved slot reached only through a scene-authored contract
        # value; no L1 scene carries it and no L1 writer path creates it.
        "event_actor_motion_contract": _event_actor_motion_contract(),
        # Runtime families preserved_fields_from deep-copies from the scene
        # entity and deep-merges from a source-script set_runtime_state
        # patch; control_state and sensor_state are L1-verified for their
        # boolean members, the rest has no L1 instance.
        "mission_state": families["mission_state"],
        "navigation_state": families["navigation_state"],
        "security_state": families["security_state"],
        "facility_state": families["facility_state"],
        "communication_state": families["communication_state"],
        "incident_state": families["incident_state"],
        "vehicle_state": families["vehicle_state"],
        "pedestrian_state": families["pedestrian_state"],
        "state_sequence": {"type": "array", "items": _identifier()},
        "task_id": _identifier(),
        "uav_corridor": _uav_corridor(),
        "uav_corridor_role": _identifier(),
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "activity_type", "animation_hint", "asset_id", "entity_id",
            "label_class", "pos_enu", "posture", "social_state", "state",
            "tick", "vel_mps", "yaw_deg",
        ],
        "properties": properties,
    }


def _formal_row() -> dict:
    families = _runtime_state_families()
    properties = {
        "activity_type": _identifier(),
        "asset_id": _identifier(),
        "category": _identifier(),
        "entity_category": _identifier(),
        # Producer identity: frozen rows belong to the configured scoped
        # entities — mission UAVs and the L1-3 intruders — whose ids are
        # ``uav_l1_<n>_v<k>[_<role>]`` or ``intruder_l1_<n>_v<k>``.
        "entity_id": {"type": "string", "minLength": 1,
                      "pattern": "^(uav|intruder)_l1_[0-9]_v[0-9](_[a-z]+)?$"},
        "entity_kind": _identifier(),
        "entity_type": _identifier(),
        # Per-episode frame key "<episode_id>_tick_<tick>" written by
        # convert_to_render_ready for every truth frame.
        "frame_id": {"type": "string", "minLength": 1, "pattern": "_tick_[0-9]+$"},
        "label_class": _identifier(),
        "pos_enu": _writer_float3(),
        "sim_time_s": _float_leaf(),
        "state": _identifier(),
        "tick": {"type": "integer", "minimum": 0},
        "vel_mps": _writer_float3(),
        "yaw_deg": _float_leaf(),
        # preserve_keys slots present on every scoped L1 entity.
        "role": _identifier(),
        "semantic_role": _identifier(),
        "task_id": _identifier(),
        "uav_visibility": _uav_visibility(),
        # Runtime families the frozen writer copies when the truth entity
        # carries them; control_state and sensor_state are L1-verified for
        # their boolean members, the rest has no L1 instance.
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
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "activity_type", "asset_id", "category", "entity_category",
            "entity_id", "entity_kind", "entity_type", "frame_id",
            "label_class", "pos_enu", "role", "semantic_role", "sim_time_s",
            "state", "task_id", "tick", "uav_visibility", "vel_mps",
            "yaw_deg",
        ],
        "properties": properties,
    }


def arm_trajectories_schema() -> dict:
    """Return the closed record schema for one ``arm_trajectories`` L1 row.

    The top-level ``oneOf`` splits the record space the way production does:
    the formal-capture branch requires ``tick`` and ``frame_id``
    (``..._tick_<n>``), ``entity_category``/``entity_kind``/``entity_type``,
    ``uav_visibility``, ``role``/``semantic_role``/``task_id`` and the scoped
    two-segment entity-id shape, while the engine branch requires
    ``posture/social_state/animation_hint`` and the scoped entity-id shape
    and forbids ``frame_id`` through ``additionalProperties`` closure.  The
    branches are mutually exclusive by those producer-identity keys; a formal
    row stripped of its branch markers no longer validates.  Branches with a
    producer basis but no current L1 instance are declared and remain
    unverified by instances (尚无实例验证): the pre-activation lifecycle fill
    of a factual window (all 84 ``objective/window_status.json`` files record
    ``pre_activation_samples_filled_from_engine: 0``), the eight non-L1
    runtime-state families (``mission_state``, ``navigation_state``,
    ``security_state``, ``facility_state``, ``communication_state``,
    ``incident_state``, ``vehicle_state``, ``pedestrian_state``), the
    non-boolean contract members of ``control_state``/``sensor_state``
    (``mode``, ``active_mode``, ``command_mode``, ``active_command``,
    ``active_commands``, ``commands``, ``exposure_status``, ``imaging_mode``
    — accepted by the same ``set_runtime_state`` handler and scene
    preservation path, with no L1 writer of them in the current corpus), the
    ``rule_id``/``mechanism_id`` metadata strings and the scene-authored
    ``event_actor_motion_contract`` slot.  Everything else is
    instance-verified against all 84 declared L1 records (38256 rows).
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_trajectories_l1.record.schema.json",
        "title": "L1 ARM raw trajectory record",
        "oneOf": [_engine_row(), _formal_row()],
    }
