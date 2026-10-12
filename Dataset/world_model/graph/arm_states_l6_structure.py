"""Closed record structure for L6 ARM raw state rows.

Source family ``arm_states``; profiles L6-1_v1, L6-1_v2, L6-2_v1, L6-2_v2,
L6-3_v1, L6-3_v2, L6-4_v1, L6-4_v2, L6-5_v1, L6-5_v2.  The only writer of
``raw/states.jsonl`` under ``arms/L6-*/`` is
``Dataset/tools/l6_v2/arm_runtime.py:266-269``: one fixed template per row —
``tick/entity_id/present/values/missing_reason`` with ``present`` hard-wired
to ``True``, ``missing_reason`` hard-wired to ``None``, and ``values`` built
as ``{f: r[f] for f in COMPARE_FIELDS if f in r}`` over the engine row.
The L6 row ``r`` comes from ``EpisodeStateEngine._row_for_entity``
(Dataset/tools/batch_generate.py:2395) via the ``Dataset/tools/l6_v2/engine.py``
replay, filtered to the scope ``Dataset/tools/l6_v2/scope.py::select_scope``
declares (aircraft, stations reached by actions, GCS anchors, home pads,
runtime-state targets — no pedestrian, vehicle or airspace-corridor entity
reaches this file).

The ``values`` key universe is exactly ``COMPARE_FIELDS``
(``Dataset/tools/l6_v2/common.py:22`` = ``SEMANTIC_FIELDS + LABEL_FIELDS``:
pos_enu, vel_mps, the five RUNTIME_FAMILIES, state, activity_type, posture,
yaw_deg), filtered to keys present on the row.  The engine row's fixed base
keys are tick/entity_id/pos_enu/vel_mps/yaw_deg/state plus the activity
payload (``row_activity_payload``, batch_generate.py:1856: activity_type,
animation_hint, posture, social_state) — animation_hint and social_state are
not in COMPARE_FIELDS and never reach this file.  ``label_class`` and
``asset_id`` are dropped by the same filter.  ``preserved_fields_from``
(batch_generate.py:1737) merges only ``PRESERVED_ENTITY_FIELDS`` plus the ten
runtime-state families, so no further COMPARE_FIELDS key can appear.

Runtime-state families on the row: ``RUNTIME_FAMILIES``
(common.py:18 — control_state, communication_state, navigation_state,
security_state, mission_state) reach the entity only as
``preserved_fields_from`` carriers (validated by
``validate_runtime_state_patch``, batch_generate.py:849) or as
``set_runtime_state`` patch payloads (validated the same way,
``_apply_pending_runtime_state_patches`` then deep-merges them,
batch_generate.py:2312/971), and are optional on ``values`` because a family
is present only when some carrier or patch writes it for that entity.  Over
the ten declared profiles, the current scene and script inputs write
control_state, communication_state, navigation_state and security_state.
The shared writer and executable patch contract also allow mission_state,
so it is declared as an optional family on ``values``.  It has no instance
in the selected L6 records and remains 尚无实例验证.  The other five contract families
(sensor/facility/incident/vehicle/pedestrian_state) are likewise outside
RUNTIME_FAMILIES, so the COMPARE_FIELDS filter can never copy them.

Each carried family closes to the executable patch contract:
``sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)``
with ``unconsumed_runtime_state_paths``
(Dataset/tools/runtime_state_contract.py:493) rejecting unknown keys and
``invalid_runtime_state_value_paths`` (:527) governing values — metadata
strings (:541-545), family booleans, contract enums as written strings
(:550-556, membership is ``value.strip().lower() in <set>``, encoded as an
audited raw-string fragment so the recorded spelling stays free),
command fields as one string or a nonempty flat string array (:557-571),
communication availability numbers in [0, 1] (:602-614).
Contract enums and commands are declared from the executable contract; the
bounded sample writes booleans plus ``mode`` (communication, enum set)
and ``operational_state`` (navigation, enum set).

Identity, labels, vectors: tick/entity_id identify the observation (tick is
an integer in [0, 900]: the engine is constructed with ``end_tick`` at most
the formal episode ceiling 900 and every declared L6 window is inside it);
pos_enu/vel_mps are three-component vectors whose members the engine writes
per component as ``round(float(value), 6)`` (batch_generate.py:2408-2409) —
floats that json.dumps serializes with no int fallback — so each component
closes to ``number`` with ``x-source-value-types: ["float"]``; finiteness is
re-verified read-side by ``source_projection.py::_arm_state_observation``.
``yaw_deg`` is ``round(float(...), 6)`` of ``last_yaw_by_entity``
(initialized from ``initial_yaw_deg`` or 0.0).  ``state`` is the keyframe
activity label (``_state_label``, batch_generate.py:1554); for the non-
pedestrian scope entities the payload sets ``activity_type`` to the same
label verbatim and ``posture`` to the literal ``"standing"``
(row_activity_payload, batch_generate.py:1866-1868), so both stay nonempty
strings rather than constants; ``get_activity(mode)`` in
``_handle_set_visual_state`` (batch_generate.py:2205) rejects an empty
authored mode before it can become a row label.

Presence branch: the record root closes to the exact keys the producer
writes.  ``arm_runtime.py:266-269`` writes ``present: True`` and
``missing_reason: None`` unconditionally on every row; no L6 producer writes
a ``present=False`` row.  The writer-side ``False`` markers exist only in
``arm_window_semantics.overlay_trajectories`` and
``batch_generate.truth_sample_summary`` — in-memory projections whose own
documentation keeps tombstones out of persisted trajectory artifacts — and
``Dataset/world_model/graph/arm_observation.py::record_observed`` (L6 branch,
arm_observation.py:50-61) is the read-side validator, not a second producer.
The ``present=False``/empty-``values`` combination is therefore read-side
scope, outside this raw grammar: this record grammar closes the root to the
single producer branch ``present=True``.  ``missing_reason`` stays closed to
null; a string reason would be an unclosed producer extension (the
string-valued form belongs to the L2 producer).  No L6 states row carries an
integrity, digest, or hash field at any depth — the only nested objects are
the runtime-state families closed below — so no publication exclusion path
is required and no global exclusion rule change is needed.

This module's root template (tick/entity_id/present/values/missing_reason)
differs from every other layer's arm_states grammar (L2 observed-marker,
L4 direct rows, L5 status pairs), so no cross-layer reuse applies.  Registry
wiring into ``structure_closure.py`` / ``validate_source_record`` belongs to
the integration stage; this module exports the pure schema constructor and
scans no data at import.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
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

# RUNTIME_FAMILIES in Dataset/tools/l6_v2/common.py:18 — the family members
# the COMPARE_FIELDS filter can copy onto values.  mission_state has no current L6
# carrier (no scene, no script, no dedicated entry writes it; no sampled row
# carries it), so it is declared contract-only below, like the contract enums
# and commands; sensor/facility/incident/vehicle/pedestrian_state are outside
# RUNTIME_FAMILIES and can never reach this file.
_CARRIED_FAMILIES = (
    "communication_state",
    "control_state",
    "mission_state",
    "navigation_state",
    "security_state",
)

# invalid_runtime_state_value_paths selects command-member domains by family
# name; imported from the executable contract so the grammar cannot drift
# from the producer's membership test.
_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}

# The exact root keys the single producer template writes
# (arm_runtime.py:266-269); every row carries all five.
_ROOT_KEYS = ("entity_id", "missing_reason", "present", "tick", "values")

# _GNSS_ENUMS / _COMMUNICATION_MODE_ENUMS are read from
# RUNTIME_STATE_ENUM_FIELDS below; only the command domains need explicit
# imports because the contract keys them by family, not by field.


def _nonempty_string() -> dict:
    return {"type": "string", "minLength": 1}


def _number3() -> dict:
    """One ENU position/velocity sample: three rounded float leaves.

    ``_row_for_entity`` writes ``round(float(value), 6)`` per member
    (batch_generate.py:2408-2409); float members serialize as JSON numbers
    with no int fallback, so the recorded source type is float on every
    component.
    """
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _metadata_rule() -> dict:
    """``rule_id`` / ``mechanism_id``: the executable contract admits the
    provenance pair beside consumed fields and rejects a value without a
    non-whitespace character (``not value.strip()``,
    runtime_state_contract.py:536-540).  No L6 carrier writes either member,
    so both are declared from the contract and stay 尚无实例验证."""
    return nonblank_python_string_rule()


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state patch field, from the executable
    patch contract (``validate_runtime_state_patch`` at
    batch_generate.py:849 delegating to
    ``invalid_runtime_state_value_paths`` in runtime_state_contract.py:527);
    raw value types are kept as the producer writes them."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return _metadata_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        # members = [value] if isinstance(value, str) else value: one string
        # or a nonempty flat sequence, each member inside the family domain
        # the contract selects.
        member = runtime_enum_string_rule(_FAMILY_COMMAND_ENUMS[family])
        return {"oneOf": [
            dict(member),
            {"type": "array", "minItems": 1, "items": member},
        ]}
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return runtime_enum_string_rule(RUNTIME_STATE_ENUM_FIELDS[(family, field)])
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family_rule(family: str) -> dict:
    """One recorded family value on the row.

    Production shape: ``deep_merge_mapping`` of validated carriers and
    patches onto the engine entity (batch_generate.py:971, applied by
    ``_apply_pending_runtime_state_patches`` at :2312), which the executable
    contract keeps a flat nonempty mapping of contract fields to scalars (a
    nested field mapping is rejected by ``unconsumed_runtime_state_paths``).
    Key closure is the contract itself, so an L6-authored contract field no
    current scene or entry has written stays declared, and an unknown key
    stays rejected.
    """
    fields = sorted(
        RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _values() -> dict:
    """The ``values`` object: fixed base keys present on every row plus the
    RUNTIME_FAMILIES members present on that row.

    Required six: pos_enu/vel_mps/yaw_deg/state come from
    ``_row_for_entity`` and activity_type/posture from the unconditional
    non-pedestrian ``row_activity_payload`` branch (the L6 scope admits no
    pedestrian row).  Each family is present only when a validated carrier
    or patch wrote it for that entity.  Key closure is COMPARE_FIELDS
    intersected with the engine-row key set — no other key can survive the
    writer filter.
    """
    properties = {
        "activity_type": _nonempty_string(),
        "pos_enu": _number3(),
        "posture": _nonempty_string(),
        "state": _nonempty_string(),
        "vel_mps": _number3(),
        "yaw_deg": _float_leaf(),
    }
    for family in _CARRIED_FAMILIES:
        properties[family] = _runtime_state_family_rule(family)
    return {
        "type": "object", "additionalProperties": False,
        "required": ["activity_type", "pos_enu", "posture", "state",
                     "vel_mps", "yaw_deg"],
        "properties": properties,
    }


def _root() -> dict:
    """The single root template of ``arm_runtime.py:266-269``.

    The writer produces exactly one shape, so the root closes as one object:
    tick/entity_id/missing_reason are written unconditionally on every row
    (:267-268 — tick and entity_id from the row, missing_reason as the
    literal ``None``), and present/values carry the only producer branch,
    the hard-wired ``present: True`` with the closed ``values`` object.  A
    string ``missing_reason`` would be an unclosed producer extension (the
    string-valued form belongs to the L2 producer)."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["entity_id", "missing_reason", "present", "tick",
                     "values"],
        "properties": {
            "entity_id": _nonempty_string(),
            # Written unconditionally as None by arm_runtime.py:268; no L6
            # branch produces a string reason.
            "missing_reason": {"type": "null"},
            "present": {"const": True},
            "tick": {"type": "integer", "minimum": 0, "maximum": 900},
            "values": _values(),
        },
    }


def arm_states_schema() -> dict:
    """Return the closed record schema for one ``arm_states`` L6 JSONL row.

    Single producer branch: ``arm_runtime.py:266-269`` writes present=True
    and missing_reason=None on every row, so the root closes to those
    constants plus the five-key template; the read-side False/tombstone
    combination is outside this raw grammar (see the module docstring).
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_states_l6.record.schema.json",
        "title": "L6 ARM raw state record",
        **_root(),
    }
