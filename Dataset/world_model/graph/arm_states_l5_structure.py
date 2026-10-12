"""Closed record structure for L5 ARM raw entity-state rows.

Source family ``arm_states``; the L5 profile set is L5-1_v1, L5-1_v2,
L5-1_v3, L5-2_v1, L5-2_v2, L5-3_v1, L5-3_v2, all carrying
``grp_l5_env_branch_v1`` arm groups (arm_factual plus arm_time_delay where a
delayed branch exists).  Producer: ``states_from_rows`` in
``Dataset/tools/l5_arm_common.py`` iterates ``range(start, end+1)`` x
``config['entity_ids']``; for a (tick, entity) missing from the filtered
trajectory index it writes the ``engine_inactive`` shape, otherwise the
``observed`` shape copied from the ``EpisodeStateEngine`` tick row built by
``Dataset/tools/batch_generate.py::_row_for_entity`` plus
``row_activity_payload`` and the runtime families merged by
``EpisodeStateEngine._apply_pending_runtime_state_patches``.
``l5_arm_pipeline.execute_config`` writes that list verbatim to
``raw/states.jsonl``; ``Dataset/tools/l5_family_delivery.py`` reads it back
through the same engine and never rewrites the file.

Branch-exact shapes.  ``observation_status`` splits the record space exactly
as production does, and each branch declares its own fixed fields:
``states_from_rows`` writes ``execution_active: True`` on every observed row
and ``execution_active: False`` plus the literal ``runtime_state: {}`` on
every engine_inactive row, so the branches carry const / empty-object rules
instead of one shared permissive rule.  The observed shape is
instance-verified against all 36 declared L5 sources (21238 rows); the
engine_inactive shape is declared from the producer code alone — the current
L5 corpus has no such row because every configured entity is active for its
whole window — and is 尚无实例验证.

Window bound.  ``load_config`` accepts only ``0 <= start < fork < end <=
900`` on the 5-tick grid and the writer enumerates every tick in
``range(start, end+1)``, so both branches declare ``tick`` as an integer in
[0, 900].  The grid bounds the configured window ends, not the enumerated
rows, so no ``multipleOf`` is declared.

``state``/``activity_type``: ``row_activity_payload`` sets ``activity_type``
to the keyframe state verbatim for the UAV entities configured in L5, and
the corpus (21238 rows) has ``state == activity_type`` on every row; the
declared rule keeps them as independent nonempty strings rather than one
constant, because the keyframe state is producer-chosen and the payload
entity keeps its own value.

Entity identifiers.  ``entity_id`` comes from ``config['entity_ids']``, whose
items ``rules/l5_arm_config.schema.json`` bound only to a nonempty string
(``{"type": "string", "minLength": 1}`` — no character pattern), and the
same string is interpolated verbatim into the ``source_ref`` fragment
(``f'raw/trajectories.jsonl#tick={tick}&entity={entity}'``).
``entity_id`` keeps the config contract's nonempty-string rule.
``source_ref`` is anchored at both ends and matches the producer fragment
exactly: the ``tick`` span is ``(?:0|[1-9][0-9]?|[1-8][0-9]{2}|900)`` — the
canonical decimal form (``0`` itself, otherwise no leading zeros) of the
enumerated ticks ``0..900`` — and the identifier span is ``[\s\S]+``, every
nonempty character sequence, newlines included.  ``.`` stops at a newline in
both Python ``re`` and ECMA-262, while the union class ``[\s\S]`` matches
every character under either engine's ``\s`` definition, so the span stays
as wide as the config contract; ``^``…``$`` pin both ends under JSON
Schema's search semantics (Python's ``$`` also matches before a trailing
newline, but the all-character span consumes that newline itself, so both
engines accept exactly the same set).  A wider ``tick`` span (``[0-9]+``),
an unanchored tail (``.*``) or an empty identifier span would each admit
refs the producer cannot write.

Runtime-state families.  ``states_from_rows`` copies the nine
``RUNTIME_FAMILIES`` members present on the trajectory row, and records
``unavailable_runtime_families`` as one comprehension over the same fixed
tuple — the written order is exactly the ``RUNTIME_FAMILIES`` tuple order
(control, mission, navigation, sensor, energy, battery, payload,
communication, positioning), members cannot repeat, and ``uniqueItems``
holds structurally.  The four families without an executable contract entry
(``energy_state``, ``battery_state``, ``payload_state``,
``positioning_state``) are declared from the writer: a set_runtime_state
payload carrying any of them is rejected by
``validate_runtime_state_patch`` (batch_generate.py:860-866 — unknown
families raise before a patch exists), and ``preserved_fields_from``
admits a family-named field only when it is a ``RUNTIME_STATE_FIELDS``
member routed through the same validator (batch_generate.py:1743-1747,
``PRESERVED_ENTITY_FIELDS`` ends with ``*RUNTIME_STATE_FIELDS``), so no
producer path can place one on a trajectory row.  Of the nine tuple
members, ``energy_state``/``battery_state``/``payload_state`` therefore
open every written array and ``positioning_state`` — the last tuple
member — closes it, with the absent contract members in tuple order
between; the written array is exactly
``[f for f in RUNTIME_FAMILIES if f not in present]`` for the row's
present contract families.  The grammar enumerates those 32 producer
outputs as fixed-length, per-position const branches, which encodes the
comprehension's order exactly instead of leaving it unconstrained.

A contract family value reaches the engine entity only through
``preserved_fields_from`` on the scene carrier or a validated
``set_runtime_state`` patch — both pass ``validate_runtime_state_patch``,
which admits exactly the ten ``RUNTIME_STATE_FIELDS`` families as nonempty
flat mappings whose fields the executable contract consumes (a nested field
mapping is rejected by ``unconsumed_runtime_state_paths``).  The recorded
``runtime_state`` properties close over the five contract families of
``RUNTIME_FAMILIES`` (control, mission, navigation, sensor, communication).
The L5 corpus records control, mission and navigation, boolean-valued; an
empty ``runtime_state`` is a valid observed value (6918 of 21238 rows carry
no family).

Value rules are the executable contract's own membership tests
(``invalid_runtime_state_value_paths`` in
``Dataset/tools/runtime_state_contract.py``), with raw types unchanged.
The member patterns are the audited shared fragments
``runtime_enum_string_rule`` / ``nonblank_python_string_rule`` of
``Dataset/world_model/graph/structure_fragments.py``, which encode the
producer's ``value.strip().lower()`` membership test and
``bool(value.strip())`` acceptance for the project runtime (CPython 3.10,
Unicode 13.0.0 — the corpus was generated by the mandated interpreter,
manifest ``runtime.python == "3.10.14"``): boolean members stay booleans;
enum members are strings whose stripped lowercase form is in the per-field
``RUNTIME_STATE_ENUM_FIELDS`` set; command members
(``active_command``/``active_commands``/``commands`` on control and mission)
are a string or a nonempty flat list of strings whose members pass the same
membership test against the family domain the contract selects
(``_CONTROL_ENUMS`` / ``_MISSION_ENUMS`` / ``_SECURITY_ENUMS``); the
provenance pair ``rule_id``/``mechanism_id`` keeps a non-whitespace
character.  The producer writes the raw value back unchanged
(``validate_runtime_state_patch`` deep-copies the payload in
``Dataset/tools/batch_generate.py`` after ``invalid_runtime_state_value_paths``
gates it), so the grammar admits the recorded spelling.  The residual
Unicode cases the fragments cannot express — a ``str.lower()`` mapping whose
result is not its own ASCII case pair under the reviewed 13.0.0 tables —
stay a producer-side integration dependency of
``invalid_runtime_state_value_paths`` and are not claimed closed here.  The
L5 authored inputs (the L5-1..L5-3 scenarios of
``Dataset/semantic_rules/profiles/epi_runtime_state_schedules/
l5_l6_x_runtime_state_schedule.json`` merged by
``Dataset/tools/regenerate_boundary_scenarios.py`` into the branch scene
carriers, plus the ``set_runtime_state`` patches of the L5 branch scripts)
write booleans only, so every contract-admitted enum, command and metadata
member is declared from the executable contract and is 尚无实例验证 on L5
records.

Dynamic keys: ``entity_id`` is the config contract's nonempty string;
``runtime_state`` family and field keys are closed over the executable
contract.  The pairing between ``runtime_state`` and
``unavailable_runtime_families`` (a disjoint cover of the nine families) and
the equality of ``tick``/``entity_id`` with the ``source_ref`` fragment span
two properties and are not expressible in JSON Schema; they remain
producer-side invariants of ``states_from_rows``.

Integrity metadata: raw ``states.jsonl`` rows carry no digest/hash/sha256
field at any depth — the top-level keyset is fixed on all 21238 rows and the
only nested objects are the runtime-state families closed above — so the
schema declares no publication-excluded path and no global exclusion rule
change is needed.

Action effects, weather semantics, coordinates, units and predicate truth
are later-stage concerns and are not encoded here.  Registry wiring into
``structure_closure.py`` and dispatch through ``structure_sources.py``
belong to the integration stage; this module only exports the pure schema
constructor, imports no data and scans nothing at import.
"""

from __future__ import annotations

import itertools

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

# RUNTIME_FAMILIES in Dataset/tools/l5_arm_common.py — the exact set
# states_from_rows may copy into runtime_state, and the exact member pool of
# unavailable_runtime_families: one comprehension over this tuple, so the
# written order is this tuple order and members cannot repeat.
_RUNTIME_FAMILIES = (
    "control_state", "mission_state", "navigation_state", "sensor_state",
    "energy_state", "battery_state", "payload_state", "communication_state",
    "positioning_state",
)
# The families the executable patch contract gives content to
# (RUNTIME_STATE_CONSUMED_FIELDS entries); a set_runtime_state payload with
# any other RUNTIME_FAMILIES member is rejected by
# validate_runtime_state_patch / unconsumed_runtime_state_paths, and no L5
# scene carrier carries one, so states_from_rows can never copy it.
# energy_state, battery_state, payload_state and positioning_state stay in
# RUNTIME_FAMILIES (and always in unavailable_runtime_families) but have no
# executable contract entry and no L5 producer path, so the recorded
# runtime_state properties close over the families below.
_FAMILIES_WITH_CONTRACT_CONTENT = tuple(
    family for family in _RUNTIME_FAMILIES
    if family in RUNTIME_STATE_CONSUMED_FIELDS
)
# The command-member domains invalid_runtime_state_value_paths selects by
# family name; imported from the executable contract so the grammar cannot
# drift from the producer's membership test.
_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}


def _nonempty_string() -> dict:
    return {"type": "string", "minLength": 1}


def _metadata_rule() -> dict:
    """``rule_id`` / ``mechanism_id``: the executable contract admits the
    provenance pair beside consumed fields and rejects a value without a
    non-whitespace character (``not value.strip()``).  The audited shared
    fragment ``nonblank_python_string_rule`` encodes that acceptance for the
    project runtime.  No L5 authored input writes either member, so both are
    declared from the contract and stay 尚无实例验证 on L5 records."""
    return nonblank_python_string_rule()


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state field, from the executable patch
    contract (``invalid_runtime_state_value_paths`` in
    ``Dataset/tools/runtime_state_contract.py``); raw value types are kept
    as the producer writes them."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return _metadata_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        # members = [value] if isinstance(value, str) else value: one string
        # or a nonempty flat sequence, each member inside the family domain
        # the contract selects.
        member_rule = runtime_enum_string_rule(_FAMILY_COMMAND_ENUMS[family])
        return {"oneOf": [
            dict(member_rule),
            {"type": "array", "minItems": 1, "items": member_rule},
        ]}
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return runtime_enum_string_rule(
            RUNTIME_STATE_ENUM_FIELDS[(family, field)])
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family_rule(family: str) -> dict:
    """One recorded family value on an observed row.

    Production shape: ``deep_merge_mapping`` of validated carriers and
    patches onto the engine entity, which the executable contract keeps a
    flat nonempty mapping of contract fields to scalars.  Key closure is the
    contract itself (consumed fields plus the admitted provenance pair), so
    an L5-authored contract field the current scripts have not yet written
    stays declared, and an unknown key stays rejected.
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


def _observed_runtime_state() -> dict:
    """``runtime_state`` on an observed row: the nine RUNTIME_FAMILIES
    members present on the trajectory row, keyed family -> nonempty flat
    contract mapping.  The empty object is a valid observed value (6918 of
    21238 rows carry no family)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {family: _runtime_state_family_rule(family)
                       for family in _FAMILIES_WITH_CONTRACT_CONTENT},
    }


def _engine_inactive_runtime_state() -> dict:
    """The literal ``{}`` ``states_from_rows`` writes on the engine_inactive
    branch (``l5_arm_common.py`` line 74); no family key can appear."""
    return {"type": "object", "additionalProperties": False,
            "maxProperties": 0}


def _unavailable_runtime_families() -> dict:
    """``unavailable_runtime_families``: the absent remainder of the nine
    ``RUNTIME_FAMILIES``, written by one comprehension over the fixed tuple
    (``l5_arm_common.py`` line 82) — order is the tuple order, members
    cannot repeat.

    The producer output is a deterministic function of the row's present
    contract families ``P``: the written array is exactly
    ``[f for f in RUNTIME_FAMILIES if f not in P]`` — the three open
    non-contract families (energy, battery, payload) first, the absent
    contract families in tuple order between, and ``positioning_state``
    (the last tuple member) always last.  Each of the 32 subsets ``P`` of
    the five contract families is producer-reachable (a validated patch or
    carrier can place any subset), so the rule enumerates all 32 written
    arrays as fixed-length per-position const branches.  This encodes the
    comprehension's order and the always-absent non-contract members
    exactly; ``additionalItems``-style rejection of a wrong order follows
    from every branch closing its own length with ``items: False``.
    """
    contract_families = _FAMILIES_WITH_CONTRACT_CONTENT
    written = {
        present: [f for f in _RUNTIME_FAMILIES if f not in present]
        for present in (
            frozenset(subset)
            for size in range(len(contract_families) + 1)
            for subset in itertools.combinations(contract_families, size)
        )
    }
    return {"oneOf": [
        {
            "type": "array",
            "minItems": len(order), "maxItems": len(order),
            "prefixItems": [{"const": family} for family in order],
            "items": False,
        }
        for order in written.values()
    ]}


def _observed_branch() -> dict:
    """The shape ``states_from_rows`` writes when the (tick, entity) pair is
    in the window-filtered trajectory index (``l5_arm_common.py`` lines
    77-83): ``execution_active`` is the literal True, and pos_enu, vel_mps,
    runtime_state, unavailable_runtime_families and source_ref are present
    on every row.  Instance-verified on all 36 declared L5 sources."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["tick", "entity_id", "observation_status",
                     "execution_active", "state", "activity_type",
                     "pos_enu", "vel_mps", "runtime_state",
                     "unavailable_runtime_families", "source_ref"],
        "properties": {
            "tick": {"type": "integer", "minimum": 0, "maximum": 900},
            # rules/l5_arm_config.schema.json bounds entity_ids items only to
            # a nonempty string (no character pattern); the identifier is
            # interpolated verbatim into the source_ref fragment, so both
            # keep the config contract's rule.
            "entity_id": _nonempty_string(),
            "observation_status": {"const": "observed"},
            "execution_active": {"const": True},
            "state": _nonempty_string(),
            "activity_type": _nonempty_string(),
            "pos_enu": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {"type": "number", "x-source-value-types": ["float"]},
            },
            "vel_mps": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {"type": "number", "x-source-value-types": ["float"]},
            },
            "runtime_state": _observed_runtime_state(),
            "unavailable_runtime_families": _unavailable_runtime_families(),
            # f'raw/trajectories.jsonl#tick={tick}&entity={entity}' with the
            # config contract's identifier, anchored at both ends:
            # - the tick span is the canonical decimal spelling of every
            #   tick states_from_rows enumerates (range(start, end+1) inside
            #   a window bounded by 0 <= start < fork < end <= 900): "0" or
            #   one/two digits without a leading zero for 1..99, three
            #   digits 100..899 without a leading zero, or the literal 900 —
            #   so no leading zeros, nothing below 0 or above 900, and no
            #   trailing text;
            # - the identifier span is [\s\S]+ — every nonempty character
            #   sequence, newlines included, because . does not match a
            #   newline in Python re or ECMA-262 and [\s\S] matches every
            #   character under both engines — keeping the span exactly as
            #   wide as the config contract's nonempty-string rule.
            # ^ and $ pin the whole value.  The equality between tick and
            # the fragment's tick span, and between entity_id and the
            # identifier span, is a cross-field producer invariant of
            # states_from_rows and is not expressible in JSON Schema.
            "source_ref": {
                "type": "string",
                "pattern": (
                    r"^raw/trajectories\.jsonl"
                    r"#tick=(?:0|[1-9][0-9]?|[1-8][0-9]{2}|900)"
                    r"&entity=[\s\S]+$"
                ),
            },
        },
    }


def _engine_inactive_branch() -> dict:
    """The exact literal shape ``states_from_rows`` writes for a configured
    (tick, entity) absent from the window-filtered trajectory index
    (``l5_arm_common.py`` lines 70-74): ``execution_active`` False, a
    one-element ``missing_source_record``, null state/activity_type and the
    literal empty ``runtime_state``.  Declared from the producer code; the
    current L5 corpus has no such row (all configured entities are active
    for their whole window), so this branch is 尚无实例验证."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["tick", "entity_id", "observation_status",
                     "execution_active", "missing_source_record", "state",
                     "activity_type", "runtime_state"],
        "properties": {
            "tick": {"type": "integer", "minimum": 0, "maximum": 900},
            "entity_id": _nonempty_string(),
            "observation_status": {"const": "engine_inactive"},
            "execution_active": {"const": False},
            "missing_source_record": {
                "type": "array", "minItems": 1, "maxItems": 1,
                "items": {"const": "raw/trajectories.jsonl"},
            },
            "state": {"type": "null"},
            "activity_type": {"type": "null"},
            "runtime_state": _engine_inactive_runtime_state(),
        },
    }


def arm_states_schema() -> dict:
    """Return the closed record schema for one ``arm_states`` L5 JSONL row.

    ``oneOf`` over the two producer shapes of ``states_from_rows``:
    ``observed`` (instance-verified on all 36 declared L5 sources) and
    ``engine_inactive`` (declared from the producer code, 尚无实例验证).
    Each branch carries its own fixed ``execution_active`` and
    ``runtime_state`` rule, ``tick`` is bounded by the configured window
    ceiling, the runtime-state value rules are the executable patch
    contract's own membership tests (the audited shared fragments of
    ``structure_fragments.py``), ``entity_id`` keeps the config contract's
    nonempty-string identifier, ``source_ref`` is anchored at both ends to
    the producer fragment — canonical tick spelling over 0..900 followed by
    a nonempty all-characters identifier span, newlines included — and
    ``unavailable_runtime_families`` fixes the always-absent non-contract
    families at their tuple positions.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_states_l5.record.schema.json",
        "title": "L5 ARM raw entity-state record",
        "oneOf": [_observed_branch(), _engine_inactive_branch()],
    }
