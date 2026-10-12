"""Closed record structure for L1 ARM raw state rows.

Source family ``arm_states``; profiles L1-1_v1, L1-1_v2, L1-2_v1, L1-3_v1,
L1-3_v2, L1-4_v1, L1-4_v2.  Producer: ``Dataset/tools/l1_window.py::
execute_config`` writes one row list to ``raw/states.jsonl`` (and the same
list to ``raw/trajectories.jsonl``).  Rows come from exactly two branches.

Engine branch (``mode: intervention``, plus the pre-spawn lifecycle fill of
``mode: factual_observation``): ``EpisodeStateEngine._row_for_entity``
(Dataset/tools/batch_generate.py) writes the fixed base keys
``tick/entity_id/label_class/asset_id/pos_enu/vel_mps/yaw_deg/state`` plus
``row_activity_payload``'s unconditional ``activity_type/animation_hint/
posture/social_state``; ``preserved_fields_from(entity)`` then adds a
preserved slot only for fields whose value is neither ``None`` nor ``""``.
The engine entity carries ``PRESERVED_ENTITY_FIELDS`` copied from the scene
setup, so the L1 branch is closed to the preserved fields the seven L1
scene files actually declare on the configured ``entity_ids``:
``category/role/semantic_role/state_sequence/task_id/uav_corridor_role/
uav_corridor/assigned_altitude_m/ground_reference_z_m/ground_reference_source/
lifecycle/activation_tick/route_waypoints_enu_m/planned_route_waypoints_enu_m``
plus the runtime-state families the scene entity carries
(``control_state`` on the nine mission UAVs that declare it - the two
L1-4 tertiary UAVs and the two L1-3 intruders declare none - and
``sensor_state`` on the two L1-3 primaries; no other family appears in
any L1 scene or L1 source payload).  Route and waypoint arrays stay producer-typed pass-throughs (the
structural catalog declares ``(None, 3)`` axes with ``int|float`` elements
and an allowed empty list); pose and velocity vectors are the catalog's
three-component observed pose with ``int|float`` sources.

Formal branch (``mode: factual_observation`` window rows): the frozen parent
``trajectories.jsonl`` of
``aw_data/render_ready_episodes_capture_filtered/<episode>/`` restricted to
``config["entity_ids"]``.  That frozen file is written by
``Dataset/tools/filter_render_ready_truth_for_capture.py::
trajectory_row_from_entity``: the fixed base keys
``tick/frame_id/sim_time_s/entity_id/label_class/asset_id/entity_category/
entity_kind/entity_type/pos_enu/vel_mps/yaw_deg/state/activity_type/category``,
then ``preserve_keys`` copies present entity keys, and a final projection
drops ``None`` values.  ``role/semantic_role/task_id`` are present on every
scoped L1 entity and therefore on every formal row.  The writer also builds a
``source`` key from ``entity.get("source")``, but the source rows feeding L1
truth entities never carry that key, so the final projection drops it on
every L1 row; it stays undeclared.  ``uav_visibility`` is
written for every ``uav`` truth entity by
``Dataset/tools/convert_to_render_ready.py::uav_camera_capture_visibility_payload``
with the two producer policy constants; ``roi_capture_distance_m`` is a
finite number whenever that writer runs (see the uav_visibility note), so
this grammar keeps it required and non-null.

Runtime-state families: the L1 producer's own action plan dispatches only
``move_entity`` (all 39 action entries across the seven L1 entry families in
``scripts/arms/L1-*`` are ``move_entity``; the omission plans
``arm_factual``/``arm_avoidance_withheld`` declare ``actions: []`` — for
example L1-2_v1 ``prevent_altitude_excursion`` and L1-4_v1/v2
``diversion_withheld`` each plan one or more ``move_entity`` actions).  No
L1 entry declares ``set_runtime_state`` or ``spawn_entity``.  The source of
runtime patches is the seven scene-side ``event_script.json`` files
(``Dataset/scenarios/L1_airspace/interaction/<profile>/``): they author 28
``set_runtime_state`` actions (2+2+2+3+3+8+8 over L1-1_v1 .. L1-4_v2), applied by
``l1_window.run_engine`` through the registered source-script handler
``EpisodeStateEngine._handle_set_runtime_state`` and
``_apply_pending_runtime_state_patches``, value-gated by
``validate_runtime_state_patch``.  Every written patch value is merged into
the scene entity by ``deep_merge_mapping`` and reaches both branches
verbatim through ``preserved_fields_from`` /
``filter_render_ready_truth_for_capture.preserve_keys``.  The corpus writes
and carries exactly seven ``control_state`` booleans plus
``sensor_state.intruder_detected`` (``rth_active`` L1-1_v1,
``alternate_rth_active`` L1-1_v2,
``altitude_corrective_maneuver_active`` L1-2_v1,
``deconfliction_active``/``evasion_active`` L1-3_v1/L1-3_v2,
``resequence_active``/``divert_active`` L1-4_v1/L1-4_v2).  The instance
evidence is boolean-only, but it is not the acceptance boundary: the
handler/validator pair merges and preserves every value shape
``validate_runtime_state_patch`` accepts, so each L1-carried family is
closed to the full executable contract in ``runtime_state_contract.py`` -
the ``RUNTIME_STATE_BOOLEAN_FIELDS`` booleans, the
``RUNTIME_STATE_ENUM_FIELDS`` strings (``control_state.mode``/
``active_mode``/``command_mode``, ``sensor_state.exposure_status``/
``imaging_mode``), the ``RUNTIME_STATE_COMMAND_FIELDS`` members
(``active_command``/``active_commands``/``commands``, oneOf string or
nonempty string list) and the ``rule_id``/``mechanism_id`` metadata
strings.  ``invalid_runtime_state_value_paths`` validates enum and command
membership as ``value.strip().lower()`` against the same canonical members
and writes the raw value back; the grammar declares each member with the
audited shared raw-string fragment
``Dataset/world_model/graph/structure_fragments.py::runtime_enum_string_rule``
and the metadata strings with ``nonblank_python_string_rule``.  The shared
fragment is imported, not modified; its per-member core covers the Kelvin
sign U+212A (the sole non-ASCII code point whose ``str.lower()`` is an
ASCII letter on the pinned Unicode 13.0.0 runtime) and its explicit
whitespace class is exactly Python ``str.strip()``'s set, closing the
ASCII+``\\s`` approximation the double audit rejected.  The fragment pins
Unicode 13.0.0 and re-checks ``unicodedata.unidata_version`` on every
call; a runtime Unicode upgrade re-opens that review (remaining
integration dependency on the producer contract's own case folding).  The
non-boolean members of ``control_state``/``sensor_state`` and
the other eight families (``mission_state``, ``navigation_state``,
``security_state``, ``facility_state``, ``communication_state``,
``incident_state``, ``vehicle_state``, ``pedestrian_state``) have no L1
scene entity, no authored patch and no L1 payload instance - declared from
the same executable contract, still-without-instance-verification.

uav_visibility null reachability: the frozen writer's contract is built by
``convert_to_render_ready.capture_contract_from_source``, which calls
``source_boundary_polygon`` unconditionally; that builder raises
``RuntimeError("Source capture_boundary lacks polygon_enu_m")`` when the
authored polygon is empty, so a conversion with an empty polygon cannot
produce records at all.  The writer stores the validated non-empty polygon
into ``capture_boundary_polygon_enu_m``; ``distance_to_polygon_xy`` over a
non-empty polygon returns 0.0 inside or a finite minimum of finite segment
distances (``distance_to_segment_xy`` projects onto a bounded segment), so
``math.isfinite`` gates no reachable L1 row and the writer's own null branch
(static, below the finite gate) is dead for every record that reaches
``raw/states.jsonl``.  The filter-side mirror
(``filter_render_ready_truth_for_capture.capture_polygon_from_event_script``)
raises on an empty polygon before any row is written.  This closure is
evidenced on the seven fixed L1 sources (each authors a four-point capture
polygon); the grammar does not claim it for arbitrary inputs.  The frozen
``states.jsonl`` rows reaching this family are the same parent trajectory
rows (position/velocity read-only inspection); the finite-ROI statement is
inherited from that writer's gate, and the sampled factual rows confirm
float-only ``roi_capture_distance_m`` values.

Boundary: an L1 window records only rows of the configured ``entity_ids``
(all mission UAVs in the current corpus, plus the two L1-3 intruders).
SUMO and global-flow payloads (``sumo_segment``, ``sumo_vehicle``,
``sumo_visibility``, ``uav_segment``, ``uav_global_flow``) are written by
the frozen writer only for those entity classes, carry no L1 instance, and
their owned structures belong to the layers that produce them; they are not
declared here with weakened placeholders.  Neither branch carries an
integrity or digest field, so no publication exclusion is required.
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

# The command members are validated by the producer contract against the same
# per-family enum sets that govern the family's enum fields
# (``invalid_runtime_state_value_paths``: control_state -> _CONTROL_ENUMS,
# mission_state -> _MISSION_ENUMS, security_state -> _SECURITY_ENUMS).  The
# executable contract does not export those sets, but it does export them per
# family through ``RUNTIME_STATE_ENUM_FIELDS`` on the family's enum fields;
# this projection is the same frozenset object, materialized once here.

# convert_to_render_ready.uav_camera_capture_visibility_payload writes these
# two policy stamps verbatim from module constants.  The finite-ROI statement
# below is scoped to the seven fixed L1 sources: each of their seven
# event_script.json files authors capture_boundary.polygon_enu_m with exactly
# four points, and both polygon builders reject an empty polygon before any
# record is written; it is a property of those sources, not of arbitrary
# input.
_CAPTURE_ROI_POLICY = "camera_only_while_uav_position_inside_capture_roi_polygon_v1"
_CAMERA_CALL_POLICY = "do_not_call_camera_when_roi_capture_eligible_is_false"


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _family_enum_members(family: str) -> tuple[str, ...]:
    """The family's governed enum set, from the same executable contract that
    validates it at write time (``RUNTIME_STATE_ENUM_FIELDS`` projects it per
    family on that family's enum fields: control_state ``mode``/
    ``active_mode``/``command_mode`` -> the control enums; mission_state
    ``mode``/``status``/``phase`` -> the mission enums; security_state
    ``mode``/``status``/``threat``/``condition`` -> the security enums)."""
    for (enum_family, _), members in RUNTIME_STATE_ENUM_FIELDS.items():
        if enum_family == family:
            return tuple(sorted(members))
    raise ValueError(f"runtime state family has no governed enum set: {family}")


def _runtime_state_string_rule(members: tuple[str, ...]) -> dict:
    """Raw-string rule for one enum or command value of the executable
    contract, delegating to the audited shared fragment.

    ``invalid_runtime_state_value_paths`` (runtime_state_contract.py:549-553,
    :576) validates enum and command members as ``value.strip().lower() in
    allowed`` while the written row keeps the raw string.  A JSON Schema
    ``enum`` of the canonical members would reject the non-canonical raw
    spellings the producer accepts, and JSON Schema patterns are ECMA-262,
    which has no case-insensitive flag — ``(?i:...)`` is a Python-only
    construct and must not be used.

    The fragment (``structure_fragments.runtime_enum_string_rule``) returns an
    anchored frame over an explicit Python-``strip()`` whitespace class around
    per-member ASCII case cores plus the Kelvin sign U+212A for the letter
    ``k``: the sole non-ASCII code point whose ``str.lower()`` is an ASCII
    letter on the pinned Unicode 13.0.0 runtime, which the audited
    ASCII+``\\s`` approximation rejected.  It keeps whitespace-only, blank,
    out-of-membership and internally-spaced values rejected, matching the
    producer's rejection of ``"   "``.

    Remaining integration dependency (owned by the producer contract, not by
    this grammar): any future Unicode-table change other than U+212A.  The
    fragment raises on a changed ``unicodedata.unidata_version`` so the rule
    cannot silently drift.
    """
    return runtime_enum_string_rule(members)


def _command_rule(family: str) -> dict:
    members = _family_enum_members(family)
    return {"oneOf": [
        _runtime_state_string_rule(members),
        {"type": "array", "minItems": 1,
         "items": _runtime_state_string_rule(members)},
    ]}


def _number3() -> dict:
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _waypoints() -> dict:
    # Producer pass-through of the authored route; empty lists are a declared
    # structural status for this family.
    return {"type": "array", "items": _number3()}


def _runtime_state_family(family: str) -> dict:
    """Close one runtime-state family to the executable producer contract.

    ``validate_runtime_state_patch`` (batch_generate.py:849, mirrored by
    ``convert_to_render_ready.py:526``) governs family membership and every
    written value: metadata strings, the contract booleans, the
    ``RUNTIME_STATE_ENUM_FIELDS`` strings, the ``RUNTIME_STATE_COMMAND_FIELDS``
    string-or-flat-string-list members and the scalar facility,
    communication and incident fields.  A written family reaches the row
    verbatim through ``preserved_fields_from`` /
    ``filter_render_ready_truth_for_capture.preserve_keys`` and is merged by
    ``deep_merge_mapping``, so any nonempty subset of these members with the
    contract's value shapes is a producer-reachable row state; the grammar
    declares them optional with ``minProperties: 1`` instead of enumerating
    observed keysets.  The enum and command string members are declared with
    their governed membership as raw-value rules (see
    ``_runtime_state_string_rule``): the producer checks
    ``value.strip().lower()`` against the canonical members at write time and
    writes the raw value back, so the grammar accepts the raw spellings the
    producer accepts — per-member ASCII case variants, the Kelvin-sign ``k``
    variant and the exact Python-strip whitespace frame from the audited
    shared fragment — and keeps rejecting blank, whitespace-only and
    out-of-membership values.  Any Unicode-table change beyond the pinned
    13.0.0 is a remaining integration dependency on the producer contract
    itself; the shared fragment raises on such a change instead of drifting.
    """
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object", "additionalProperties": False, "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state patch field, from the executable
    producer contract (``validate_runtime_state_patch`` plus
    ``invalid_runtime_state_value_paths`` in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        # The producer rejects values that are blank after strip()
        # (runtime_state_contract.py:541-543); the raw value is written back,
        # so the accepted raw string is "non-empty after stripping".  The
        # shared fragment's ``nonblank_python_string_rule`` names Python
        # ``str.strip()``'s whitespace set exactly (unlike ``\\S`` under a
        # strict ECMA-262 engine, which would add U+FEFF); ``minLength: 1``
        # would accept whitespace-only strings the producer rejects.
        return nonblank_python_string_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        # Membership is validated by the same producer contract
        # (``invalid_runtime_state_value_paths`` checks each command member
        # against the family's governed enum set); declared here with the
        # same members rather than only the string shape.
        return _command_rule(family)
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list and rejects any member that is
        # not a non-empty string (runtime_state_contract.py:590-595); the
        # per-member rule below states exactly that.
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        # Same producer check as the metadata strings:
        # runtime_state_contract.py:596-601 rejects values that are blank
        # after strip(), and the raw value is written back, so the shared
        # fragment's "non-empty after stripping" rule is the exact contract
        # (``minLength: 1`` would accept whitespace-only strings the
        # producer rejects).
        return nonblank_python_string_rule()
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        # Membership is validated by the same producer contract
        # (``invalid_runtime_state_value_paths``, case-insensitively with
        # ``strip().lower()`` and the raw value written back); declared here
        # as the raw-value frame rule over the canonical members (see
        # ``_runtime_state_string_rule``).
        return _runtime_state_string_rule(
            tuple(sorted(RUNTIME_STATE_ENUM_FIELDS[(family, field)])))
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_families() -> dict:
    rules: dict[str, dict] = {}
    for family in RUNTIME_STATE_FIELDS:
        # One closure for all ten families: validate_runtime_state_patch and
        # invalid_runtime_state_value_paths accept the same shapes in L1 as
        # in the action rows of every other layer, and a source-script patch
        # merged into the scene entity reaches the L1 row verbatim.
        rules[family] = _runtime_state_family(family)
    return rules


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
            # distance_to_polygon_xy over the validated non-empty capture
            # polygon (empty raises before any record is written; every one
            # of the seven fixed L1 sources authors a four-point polygon)
            # returns 0.0 inside or a finite segment minimum, so round()
            # always yields a finite number here.
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
        "pos_enu": _number3(),
        "posture": _identifier(),
        "social_state": _identifier(),
        "state": _identifier(),
        "tick": {"type": "integer", "minimum": 0},
        "vel_mps": _number3(),
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
        # Runtime families preserved_fields_from deep-copies from the scene
        # entity and the applied source-script patches; each family is closed
        # to the same executable producer contract.
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
        "pos_enu": _number3(),
        "sim_time_s": _float_leaf(),
        "state": _identifier(),
        "tick": {"type": "integer", "minimum": 0},
        "vel_mps": _number3(),
        "yaw_deg": _float_leaf(),
        # preserve_keys slots present on every scoped L1 entity.
        "role": _identifier(),
        "semantic_role": _identifier(),
        "task_id": _identifier(),
        "uav_visibility": _uav_visibility(),
        # Runtime families the frozen writer copies when the truth entity
        # carries them; each is closed to the same executable producer
        # contract as the engine branch.
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
            "state", "task_id", "tick", "uav_visibility", "vel_mps", "yaw_deg",
        ],
        "properties": properties,
    }


def arm_states_schema() -> dict:
    """Return the closed record schema for one ``arm_states`` L1 JSONL row.

    The top-level ``oneOf`` splits the record space the way production does.
    The branches are mutually exclusive by producer-identity keys: the engine
    branch requires ``posture/social_state/animation_hint`` and the scoped
    entity-id shape and forbids ``frame_id``, while the formal branch requires
    ``frame_id`` (``..._tick_<n>``), ``entity_category``/``entity_kind``/
    ``entity_type``, ``uav_visibility``, ``role``/``semantic_role``/``task_id``
    and the scoped two-segment entity-id shape.  A formal row stripped of its
    branch markers no longer validates.

    Instance-verified branches (bounded row check behind this wording:
    12 rows from each of six L1 source files spanning the profiles
    L1-1_v1, L1-1_v2, L1-2_v1, L1-3_v1, L1-4_v1 and L1-4_v2 — 72 rows in
    total; L1-3_v2 was not among the read files, so no claim below rests on
    an L1-3_v2 row): the always-written keys and the preserved slots of both
    branches, the ``lifecycle``/``uav_corridor``/``uav_visibility``
    subobjects, the route/waypoint arrays, and the scripted ``control_state``
    booleans (``rth_active`` L1-1_v1, ``alternate_rth_active`` L1-1_v2,
    ``altitude_corrective_maneuver_active`` L1-2_v1,
    ``deconfliction_active``/``evasion_active`` L1-3_v1,
    ``resequence_active``/``divert_active`` L1-4_v1/L1-4_v2) plus
    ``sensor_state.intruder_detected`` on the sampled L1-3_v1 primary — each
    observed as a JSON boolean in those rows.  The six-file bound is the
    actual checked scope behind this wording, not a claim that all 84
    declared ``raw/states.jsonl`` files under the seven L1 profiles or every
    branch variant were instance-validated.

    Producer-evidenced but without a current L1 instance (declared from the
    executable contract, not verified against a row): every non-boolean
    member of ``control_state``/``sensor_state`` (the enum fields
    ``mode``/``active_mode``/``command_mode`` and
    ``exposure_status``/``imaging_mode``, the command members
    ``active_command``/``active_commands``/``commands`` and the
    ``rule_id``/``mechanism_id`` metadata strings), the eight runtime
    families without an L1 scene entity, authored patch or payload instance
    (``mission_state``, ``navigation_state``, ``security_state``,
    ``facility_state``, ``communication_state``, ``incident_state``,
    ``vehicle_state``, ``pedestrian_state``), and the observed boolean
    members' non-observed siblings inside ``control_state``/``sensor_state``.
    Unsampled-profile L1-3_v2 branches stay in this category: the
    ``Dataset/scenarios/L1_airspace/interaction/L1-3_v2/event_script.json``
    actions ``set_runtime_state_l1_3_v2_boundary_conflict_00`` and
    ``..._return_safe_airspace_00`` author ``control_state.deconfliction_active``/
    ``evasion_active`` and the ``approach_boundary`` action authors
    ``sensor_state.intruder_detected`` on ``uav_l1_3_v2_primary``, but no
    L1-3_v2 source row was read this round, so those values are
    script-evidenced, not row-observed, and are marked 尚无实例验证.
    A null ``uav_visibility.roi_capture_distance_m`` is no longer part of the
    record space.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_states_l1.record.schema.json",
        "title": "L1 ARM raw state record",
        "oneOf": [_engine_row(), _formal_row()],
    }
