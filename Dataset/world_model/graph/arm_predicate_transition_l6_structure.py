"""Closed record structure for L6 ARM objective predicate transitions.

Source family ``arm_predicate_transitions`` (source_family ==
effective_family; one JSONL row per record), profiles L6-1_v1, L6-1_v2,
L6-2_v1, L6-2_v2, L6-3_v1, L6-3_v2, L6-4_v1, L6-4_v2, L6-5_v1, L6-5_v2.
Declared inventory (recount at generation time): 168 arm branches, 168
``objective/predicate_transitions.jsonl`` files present, 0 missing,
structure status ``field_scan_defined``; the family is declared per arm
branch with owner ``PredicateTransition``
(``Dataset/world_model/graph/field_catalog.py``).

Producer chain, verified in the current tree:

1. ``Dataset/tools/l6_v2/arm_runtime.py`` is the single writer anchor for
   all ten profiles: every L6 dedicated entry under ``scripts/arms/L6-*/``
   imports ``run_arm`` from it, and it writes the window-filtered rows
   verbatim through ``write_rows(out / 'objective' / f'{name}.jsonl',
   data)`` for ``name == 'predicate_transitions'`` (arm_runtime.py:281-282,
   common.py::write_rows).  ``Dataset/tools/l6_v2/observations.py`` selects
   the rows by ``start <= r['to_tick'] <= end`` (observations.py:43-44) and
   ``rewrite()`` prefixes only string values: refs whose ``#tick=N`` lies
   before the window become ``design/<group_key>/history/<arm>_<sha16>
   .json.gz#raw/...`` (observations.py:59-68); keys are never rewritten.
2. ``Dataset/tools/l6_v2/semantics.py`` builds the assertions.  The
   per-arm predicate universe is ``ArmSpec.predicate_ids()``
   (arm_runtime.py:111-117): ``BASE_PREDICATES`` (arm_runtime.py:45-50)
   plus ``PREDICATE_ADDITIONS['positioning.']`` when the primary is a
   ``positioning.`` id (arm_runtime.py:51-54) plus the arm's
   ``primary_predicate``.  The L6 primaries read from the dedicated
   entries are ``communication.link_unavailable`` (L6-1),
   ``communication.link_degraded`` (L6-2),
   ``positioning.gnss_spoofing_active`` (L6-3),
   ``security.communication_jamming_impact_active`` (L6-4) and
   ``security.ground_control_station_compromised`` (L6-5), so the
   profile union is exactly the fifteen ids in ``_ARM_PREDICATE_IDS``.
   ``binding_universe`` (semantics.py:200-235) takes each template's
   ``argument_roles`` from the same governed registry this module reads
   and synthesizes bindings from the L6 scope (aircraft/station/gcs ids,
   home pads, the GNSS service and spoofing signal instances); an
   unsupported role tuple raises there.  ``evaluate`` stamps
   ``tuple_type`` as ``'grounded_roles__' + '__'.join(role_keys)`` in
   ``argument_roles`` order and ``tuple_id`` as ``'arm_tuple:'`` plus the
   first 24 hex of ``digest([episode, predicate, binding])``
   (semantics.py:355-357; ``common.digest`` returns ``sha256:<64 hex>``,
   common.py:54-57), on formal-grid ticks only
   (``tick % PARAMS['formal_step_ticks'] == 0``, semantics.py:349-350).
3. ``Dataset/semantic_truth/minimal_semantics.py::build_transitions``
   pairs adjacent same-tuple assertions (minimal_semantics.py:238-295):
   only the exact boolean flips are kept — ``false->true`` rising,
   ``true->false`` falling — same-value and unknown pairs ``continue``
   (lines 262-267), and a non-adjacent pair becomes a
   ``continuity_break`` which the L6 caller rejects before any write
   (semantics.py:380-382, ``ValueError('non-adjacent formal predicate
   samples')``).  ``from_value``/``to_value`` are therefore the two-value
   string enums ``"true"``/``"false"`` on this layer, narrower than the
   registry-wide four-value vocabulary the rule digest's name recalls;
   L6 also asserts only ``"true"``/``"false"``/``"unknown"`` values
   (semantics.py:337-343), so ``"out_of_scope"``/``"unknown"`` can never
   sit in a kept flip.
4. ``Dataset/semantic_truth/minimal_semantics_adapter.py::
   _adapt_transitions`` (lines 868-929) fixes every key of the record:
   the five constant header fields (``schema_name`` ``"predicate_transition"``,
   ``SCHEMA_VERSION`` ``"3.0.0"`` from line 38, ``"L1"``/``"L0"``,
   ``"predicate_delta"``), ``transition_id`` = ``stable_identifier(
   "transition", episode, predicate, tuple, before_tick, after_tick,
   transition)`` = ``"transition:"`` + 24 hex (provenance.py:42-44),
   ``episode_id`` = ``config['episode_id'] + '::' + group_key + '/' +
   arm_id`` (semantics.py:251), ``bindings`` rebuilt from
   ``grounded_roles__<k1>__<k2>`` plus the participants, ``from_tick``/
   ``to_tick`` = ``before_tick``/``after_tick``, ``from_value``/
   ``to_value`` = the two boundary ``Truth.value`` strings, ``direction``,
   ``evidence_truth_ids`` = ``[before['truth_id'], after['truth_id']]``
   (``"truth:"`` + 24 hex ids of the two boundary truth rows),
   ``source_refs`` = ``sorted(set(before)|set(after))`` of the boundary
   refs, ``rule_digest`` = ``digest_object({'rule':
   'exact_adjacent_formal_step_four_value_transition',
   'predicate_id': ...})`` (``sha256:`` + 64 hex), ``template_parameters``
   = literal ``{}``.  Every L6 ref list starts with the trajectory ref
   ``raw/trajectories.jsonl#tick={tick}&entity={entity_id}``
   (semantics.py:344), so the merged list is nonempty and unique by
   construction.  ``semantics.py:389-392`` then stamps
   ``parent_episode_id`` = ``config['episode_id']`` = ``{profile}__{seed}``
   (arm_runtime.py:91-92) and ``arm_key`` = ``group_key + '/' + arm_id``
   where ``group_key = {profile}/{seed}/grp_{anchor}[_{suffix}]``
   (arm_runtime.py:95-97) onto every row.  The current adapter writes 21
   keys; bounded frozen L6-3/L6-4/L6-5 rows have the same envelope minus
   ``rule_digest`` (20 keys).  The available writer path does not explain
   this frozen difference, so the integrity field is optional and typed.

Branch structure (family convention, matching the formal twin
``Dataset/world_model/graph/predicate_structure.py`` lines 411-477 and the
L5 sibling ``arm_predicate_transition_l5_structure.py`` lines 316-331):
the root is the 20-key envelope with optional ``rule_digest`` and
``additionalProperties: false``; the first ``allOf`` branch carries one
``oneOf`` variant per predicate id — const ``predicate_id`` plus the
closed ``bindings`` object holding exactly that id's registry role keys
and its source-reference shape —
and the second pins the two boolean flips to the direction.  The root
``bindings`` property only declares the key for the root's closed
boundary; the exact closed shape comes from the variant, exactly as in
``predicate_structure.py::_base_properties`` line 287.  Role keys are read
live from ``Dataset/semantic_truth/core_semantic_registry.py``
(``get_core_predicate_templates`` — the same governed registry the
producer stamped into ``tuple_type``) and the builder fails closed on
registry drift instead of widening.

For the ten non-airspace predicates, each boundary truth row contributes
one extra ref and one trajectory ref.  With both ticks inside the window,
the adapter's ``sorted(set(...))`` orders the four raw refs as extra, extra,
trajectory, trajectory.  A transition whose ``to_tick`` is the window start
retains its preceding ``from_tick`` and the projection prefixes both earlier
refs with the same history path without re-sorting the array.  The written
order is history extra, raw extra, history trajectory, raw trajectory.
These are separate fixed-position array branches; airspace refs keep their
existing item rule.

Not re-decided here: predicate business truth; the tick arithmetic
(``to_tick - from_tick == formal_step`` and ``to_tick > from_tick`` span
properties and stay producer-side invariants of ``build_transitions``,
like the L5 grammar line 68-73); the per-row equality of the ref ticks
with ``from_tick``/``to_tick`` and of the ``arm_key`` segment layout with
``episode_id`` (multi-property/file spans, kept as producer invariants;
no character contract exists for profile/seed/arm ids beyond
non-emptiness); and semantic repair.

Evidence scope.  Six factual seed00 transition files, one each from
L6-1_v1, L6-2_v1, L6-3_v2, L6-4_v2, L6-5_v1 and L6-5_v2, contributed
35 rows (each file at most 12), covering 13 of the 15 predicate ids.
The two airspace predicates ``aircraft_airborne`` and
``aircraft_touchdown``, airspace rows with fewer than four refs, and the
history-rewritten ``design/...`` ref form remain 尚无实例验证 in that sample.
The off-grid ``raw/communication_ticks`` ref cannot enter formal
transitions and is not admitted.

Integration (not wired by this module): registry/dispatch ownership is
G1 (``structure_closure``); the suggested authority anchor is
``Dataset/tools/l6_v2/arm_runtime.py;Dataset/tools/l6_v2/semantics.py;
Dataset/semantic_truth/minimal_semantics_adapter.py:_adapt_transitions``.
Publication projection needs no new exclusion paths: this family's only
integrity-keyed field is the top-level ``rule_digest``, already covered by
the exact path ``("rule_digest",)`` in
``structure_schema._PUBLICATION_EXCLUDED_PATHS`` (line 27).
"""

from __future__ import annotations

from typing import Any

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# The L6-active predicate id union: BASE_PREDICATES and PREDICATE_ADDITIONS
# (Dataset/tools/l6_v2/arm_runtime.py:45-54) plus the L6 primaries verified
# from the dedicated entries under scripts/arms/L6-*/ (L6-1/L6-2/L6-3
# primaries are base or positioning-addition ids; L6-4 and L6-5 carry the
# two security ids).  This is the union across L6 arms; the set and
# the per-id role keys below are cross-checked against the registry at
# schema build time and the build fails closed on drift.
_ARM_PREDICATE_IDS = (
    "airspace.aircraft_airborne",
    "airspace.aircraft_at_home_pad",
    "airspace.aircraft_in_landing_zone",
    "airspace.aircraft_outside_assigned_altitude_band",
    "airspace.aircraft_touchdown",
    "communication.link_degraded",
    "communication.link_unavailable",
    "emergency.aircraft_in_return_to_home",
    "emergency.aircraft_in_safe_hold",
    "positioning.aircraft_deviates_from_planned_route",
    "positioning.gnss_quality_degraded",
    "positioning.gnss_spoofing_active",
    "positioning.navigation_error_exceeds_tolerance",
    "security.communication_jamming_impact_active",
    "security.ground_control_station_compromised",
)

_SCHEMA_VERSION = "3.0.0"  # minimal_semantics_adapter.SCHEMA_VERSION (line 38)
_HEX24 = "[0-9a-f]{24}"
_REF_TICK = r"(?:0|[1-9][0-9]?|[1-8][0-9]{2}|900)"
_HISTORY_PREFIX = (
    r"design/[^/]+/[^/]+/grp_[^/]+/history/"
    r"[^/]+_[0-9a-f]{16}\.json\.gz#"
)
_TRAJECTORY_BODY = rf"raw/trajectories\.jsonl#tick={_REF_TICK}&entity=[\s\S]+"
_COMMUNICATION_BODY = rf"raw/communication_state\.jsonl#tick={_REF_TICK}&entity=[\s\S]+"
_CONTROL_BODY = rf"raw/control_observations\.jsonl#tick={_REF_TICK}&entity=[\s\S]+"
_GNSS_BODY = (
    rf"raw/domain_observations\.jsonl#tick={_REF_TICK}"
    rf"&subject=[\s\S]+&family=gnss_navigation"
)
_SECURITY_BODY = (
    rf"raw/domain_observations\.jsonl#tick={_REF_TICK}"
    rf"&subject=[\s\S]+&family=security_command"
)
_FLIP_VALUES = ("true", "false")  # build_transitions keeps boolean flips only
_TRUTH_ID = {"type": "string", "pattern": f"^truth:{_HEX24}$"}
_EMPTY_OBJECT = {
    "type": "object",
    "additionalProperties": False,
    "maxProperties": 0,
}


def _identifier() -> dict[str, Any]:
    """Producer ids carry no character contract beyond non-emptiness."""
    return {"type": "string", "minLength": 1}


def _source_ref(body: str) -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": rf"^(?:{body}|{_HISTORY_PREFIX}{body})$",
    }


def _raw_ref(body: str) -> dict[str, Any]:
    return {"type": "string", "pattern": rf"^{body}$"}


def _history_ref(body: str) -> dict[str, Any]:
    return {"type": "string", "pattern": rf"^{_HISTORY_PREFIX}{body}$"}


def _four_refs(*members: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "array",
        "prefixItems": list(members),
        "items": False,
        "minItems": 4,
        "maxItems": 4,
        "uniqueItems": True,
    }


def _fixed_refs(first: str) -> dict[str, Any]:
    return {"oneOf": [
        _four_refs(
            _raw_ref(first), _raw_ref(first),
            _raw_ref(_TRAJECTORY_BODY), _raw_ref(_TRAJECTORY_BODY),
        ),
        _four_refs(
            _history_ref(first), _raw_ref(first),
            _history_ref(_TRAJECTORY_BODY), _raw_ref(_TRAJECTORY_BODY),
        ),
    ]}


def _source_refs_for(predicate_id: str) -> dict[str, Any]:
    if predicate_id.startswith("communication."):
        return _fixed_refs(_COMMUNICATION_BODY)
    if predicate_id.startswith("positioning.gnss") or predicate_id == "positioning.navigation_error_exceeds_tolerance":
        return _fixed_refs(_GNSS_BODY)
    if predicate_id.startswith("security."):
        return _fixed_refs(_SECURITY_BODY)
    if predicate_id == "positioning.aircraft_deviates_from_planned_route" or predicate_id.startswith("emergency."):
        return _fixed_refs(_CONTROL_BODY)
    if predicate_id.startswith("airspace."):
        needs_pad = predicate_id in (
            "airspace.aircraft_at_home_pad",
            "airspace.aircraft_in_landing_zone",
        )
        return {
            "type": "array",
            "items": _source_ref(_TRAJECTORY_BODY),
            "minItems": 4 if needs_pad else 2,
            "maxItems": 4,
            "uniqueItems": True,
        }
    raise ValueError(f"unregistered L6 transition predicate: {predicate_id}")


def _arm_predicate_roles() -> dict[str, tuple[str, ...]]:
    """Registry argument-role keys for the L6-active predicate ids.

    Reads the governed registry the producer itself stamps into
    ``tuple_type`` (semantics.py:355-356) and fails closed on any drift
    between ``_ARM_PREDICATE_IDS`` and the registry: unknown id, empty or
    malformed ``argument_roles``, or a non-string role key all raise.
    """

    templates = {str(t["id"]): t for t in get_core_predicate_templates()}
    roles: dict[str, tuple[str, ...]] = {}
    for predicate_id in _ARM_PREDICATE_IDS:
        template = templates.get(predicate_id)
        if template is None:
            raise ValueError(
                f"L6 arm predicate is absent from the core registry: {predicate_id!r}"
            )
        declared = template.get("argument_roles")
        if not isinstance(declared, list) or not declared:
            raise ValueError(f"{predicate_id!r}: predicate has no argument roles")
        keys: list[str] = []
        for role in declared:
            if not isinstance(role, dict) or not isinstance(role.get("key"), str) \
                    or not role["key"]:
                raise ValueError(f"{predicate_id!r}: predicate role key is invalid")
            keys.append(role["key"])
        roles[predicate_id] = tuple(keys)
    return roles


def _binding_variant(predicate_id: str, role_keys: tuple[str, ...]) -> dict[str, Any]:
    """One branch: exact predicate, role-key bindings and evidence refs.

    ``_bindings`` reconstructs the bindings from
    ``grounded_roles__<k1>__<k2>`` plus the participants, so the object's
    keys are exactly the template's ``argument_roles`` keys, in that
    order, and the values are the participant entity/service ids.  The
    variant carries no ``type``: it composes with the root object rule.
    """

    return {
        "properties": {
            "predicate_id": {"const": predicate_id},
            "bindings": {
                "type": "object",
                "additionalProperties": False,
                "required": list(role_keys),
                "properties": {key: _identifier() for key in role_keys},
            },
            "source_refs": _source_refs_for(predicate_id),
        },
        "required": ["predicate_id", "bindings", "source_refs"],
    }


def _direction_variant(direction: str, from_value: str, to_value: str) -> dict[str, Any]:
    """The exact boolean flip ``build_transitions`` emits for a direction."""

    return {
        "properties": {
            "direction": {"const": direction},
            "from_value": {"const": from_value},
            "to_value": {"const": to_value},
        },
        "required": ["direction", "from_value", "to_value"],
    }


def arm_predicate_transitions_schema() -> dict[str, Any]:
    """Return the closed Draft 2020-12 record schema for one L6 row.

    The root is the 20-key envelope plus optional ``rule_digest``; every
    envelope key is required and extra properties are closed.  The
    first ``allOf`` branch carries one ``oneOf`` variant per L6 predicate
    id (const predicate id plus closed registry role-key bindings and refs
    object; the root-level ``bindings`` property only declares the key for
    the root's closed boundary, the variant carries its closed shape), the
    second pins the ``direction``/``from_value``/``to_value`` pair to the
    two boolean flips ``build_transitions`` emits; continuity-break
    records cannot reach this file because the producer raises on them.
    Fresh containers are built on every call; nothing is shared with
    other modules.
    """

    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/"
               "arm_predicate_transitions_l6.record.schema.json",
        "title": "L6 ARM objective predicate transition record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_name", "schema_version", "annotation_layer",
            "source_layer", "materialization_policy", "transition_id",
            "episode_id", "predicate_id", "tuple_id", "bindings",
            "from_tick", "to_tick", "from_value", "to_value", "direction",
            "evidence_truth_ids", "source_refs",
            "template_parameters", "parent_episode_id", "arm_key",
        ],
        "properties": {
            "schema_name": {"const": "predicate_transition"},
            "schema_version": {"const": _SCHEMA_VERSION},
            "annotation_layer": {"const": "L1"},
            "source_layer": {"const": "L0"},
            "materialization_policy": {"const": "predicate_delta"},
            "transition_id": {
                "type": "string", "pattern": f"^transition:{_HEX24}$"},
            "episode_id": {
                "type": "string",
                "pattern": r"^(L6-[1-5]_v[12])__([^/]+)::\1/\2/grp_[^/]+/[^/]+$",
            },
            "predicate_id": {"enum": sorted(_ARM_PREDICATE_IDS)},
            "tuple_id": {
                "type": "string", "pattern": f"^arm_tuple:{_HEX24}$"},
            # Root-level base rule (as in predicate_structure.py
            # _base_properties): it declares the key for the root's
            # additionalProperties boundary; the exact closed shape per
            # predicate id comes from the first allOf oneOf branch below.
            "bindings": {"type": "object"},
            "from_tick": {"type": "integer", "minimum": 0,
                          "maximum": 895, "multipleOf": 5},
            "to_tick": {"type": "integer", "minimum": 5,
                        "maximum": 900, "multipleOf": 5},
            "from_value": {"enum": list(_FLIP_VALUES)},
            "to_value": {"enum": list(_FLIP_VALUES)},
            "direction": {"enum": ["rising", "falling"]},
            # [before["truth_id"], after["truth_id"]] (adapter line 910).
            "evidence_truth_ids": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "prefixItems": [dict(_TRUTH_ID), dict(_TRUTH_ID)],
                "items": False, "uniqueItems": True,
            },
            # The predicate variant closes the ref formats and lengths.
            # The adapter sorts and deduplicates boundary refs; history
            # projection may prefix a pre-window ref with its gzip path.
            # Ref tick and boundary tick equality remains producer-side.
            "source_refs": {
                "type": "array", "minItems": 2, "maxItems": 4,
                "uniqueItems": True, "items": {"type": "string"},
            },
            "rule_digest": {
                "type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            # The adapter's literal {} (adapter line 921), closed like
            # the formal twin's _empty_object.
            "template_parameters": dict(_EMPTY_OBJECT),
            "parent_episode_id": {
                "type": "string", "pattern": r"^L6-[1-5]_v[12]__[^/]+$"},
            "arm_key": {
                "type": "string",
                "pattern": r"^L6-[1-5]_v[12]/[^/]+/grp_[^/]+/[^/]+$",
            },
        },
        "allOf": [
            # One branch per predicate id: const predicate_id plus
            # exactly that id's registry binding key set.  The
            # predicate-id discrimination has to sit at this level — the
            # L6-4/L6-5 security ids share the {actor, station} shape
            # with the two communication ids, so a bindings-only oneOf
            # could not stay exclusive.
            {"oneOf": [_binding_variant(predicate_id, role_keys)
                       for predicate_id, role_keys
                       in sorted(_arm_predicate_roles().items())]},
            {"oneOf": [
                _direction_variant("rising", "false", "true"),
                _direction_variant("falling", "true", "false"),
            ]},
        ],
    }
