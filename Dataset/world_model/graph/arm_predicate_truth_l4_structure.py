"""Whole-row grammar for L4 ARM ``objective/predicate_truth.jsonl``.

``l4_window.execute_config`` writes window-cropped rows from one of two
producers.  Formal profiles run ``build_episode_objective_artifacts``; the
other eleven profiles use ``l4_business_semantics.evaluate``.  That business
producer appends the same row object to its dense tick and sampled truth
collections, so the L4 tick schema supplies its measured metric structures.

Frozen formal 3.0.0 rows have no update ticks and occur both without digests
and with all three digests.  The current formal adapter writes 3.1.0 with
all three digests and update ticks.  Frozen L4-9 business rows predate ``rule_digest``
and spell their binding as ``vehicles``; the current business writer emits
``rule_digest`` and spells every binding as ``entities``.  These are separate
closed roots, not independently optional fields.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from Dataset.tools.l4_business_semantics import contract as business_contract
from Dataset.tools.l4_chain_truth import L4_INCIDENT_FAMILY
from Dataset.world_model.graph.arm_predicate_ticks_l4_structure import (
    arm_predicate_truth_ticks_schema,
)
from Dataset.world_model.graph.predicate_structure import (
    formal_predicate_truth_schema,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_DIGEST_FIELDS = ("rule_digest", "parameter_digest", "input_digest")
_UPDATE_FIELDS = ("truth_state_update_tick", "evidence_update_tick")
# B1-L4: the only predicate leaf whose registered role class (LandingArea) is a
# base class that real evidence narrows to an actual subclass (LandingPad).
_LANDING_ZONE_PREDICATE = "airspace.aircraft_in_landing_zone"
_DECLARED_PROFILES = frozenset({
    "L4-1_v1", "L4-1_v2", "L4-2_v1", "L4-2_v2", "L4-3_v1",
    "L4-3_v2", "L4-3_v3", "L4-4_v1", "L4-4_v2", "L4-5_v1",
    "L4-5_v2", "L4-5_v3", "L4-6_v1", "L4-6_v2", "L4-7_v1",
    "L4-7_v2", "L4-8_v1", "L4-8_v2", "L4-9_v1", "L4-9_v2",
    "L4-10_v1", "L4-10_v2", "L4-11_v1", "L4-11_v2",
})
_BUSINESS_PROFILES = frozenset({
    "L4-4_v1", "L4-4_v2", "L4-5_v1", "L4-5_v2", "L4-5_v3",
    "L4-9_v1", "L4-9_v2", "L4-10_v1", "L4-10_v2",
    "L4-11_v1", "L4-11_v2",
})


def _episode_pattern(profiles: set[str] | frozenset[str]) -> str:
    if not profiles:
        raise ValueError("L4 predicate truth has no profiles for this branch")
    alternatives = "|".join(re.escape(name) for name in sorted(profiles))
    return rf"^(?:{alternatives})__seed(?:00|01|02)$"


def _formal_branch(formal: dict[str, Any], *, frozen: bool,
                   digests: bool,
                   profiles: frozenset[str]) -> dict[str, Any]:
    schema = deepcopy(formal)
    for annotation in ("$schema", "$id", "title"):
        schema.pop(annotation)
    properties = schema["properties"]
    required = schema["required"]
    properties["episode_id"] = {
        "type": "string", "pattern": _episode_pattern(profiles),
    }
    properties["truth_id"] = {
        "type": "string", "pattern": r"^truth:[0-9a-f]{24}$",
    }
    properties["tuple_id"] = {
        "type": "string", "pattern": r"^world_predicate_tuple:[0-9a-f]{24}$",
    }
    properties["tick"] = {
        "type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5,
    }
    properties["source_class"] = {"const": "deterministic_derived"}
    properties["mechanism_id"] = {"const": None}
    required.append("mechanism_id")
    if frozen:
        properties["schema_version"] = {"const": "3.0.0"}
        for field in _UPDATE_FIELDS:
            properties.pop(field)
            required.remove(field)
        if not digests:
            for field in _DIGEST_FIELDS:
                properties.pop(field)
                required.remove(field)
    else:
        for field in _UPDATE_FIELDS:
            properties[field] = {
                "type": "integer", "minimum": 0, "maximum": 900,
                "multipleOf": 5,
            }

    for predicate in schema["allOf"][0]["oneOf"]:
        observations = predicate["properties"]["evidence"]["properties"]["observations"]
        observations["minItems"] = 1
        observations["maxItems"] = 1
        value = observations["items"]["oneOf"][0]["properties"]["value"]
        if frozen:
            for field in _UPDATE_FIELDS:
                value["properties"].pop(field)
                value["required"].remove(field)
        else:
            for field in _UPDATE_FIELDS:
                value["properties"][field] = {
                    "type": "integer", "minimum": 0, "maximum": 900,
                    "multipleOf": 5,
                }
        # B1-L4: the landing_zone role class of aircraft_in_landing_zone is
        # registered as the base class LandingArea, so formal_predicate_truth_schema
        # pins binding_ontology_classes.landing_zone to the const world:LandingArea.
        # Real evidence (world_truth._resolve_actual_class) records the actual
        # subclass instead: frozen 3.0.0 no-digest rows declare only world:LandingPad;
        # frozen 3.0.0 digest rows declare world:LandingPad and world:LandingArea; the
        # current 3.1.0 resolver supports base LandingArea and subclass LandingPad.
        # Declare each branch from its own evidence/resolver set -- never widen to an
        # open union, and never pin the instance-less current branch to Pad alone.
        if predicate["properties"]["predicate_id"].get("const") == _LANDING_ZONE_PREDICATE:
            role_classes = value["properties"]["binding_ontology_classes"]["properties"]
            if digests:
                role_classes["landing_zone"] = {"enum": ["world:LandingPad", "world:LandingArea"]}
            else:
                role_classes["landing_zone"] = {"const": "world:LandingPad"}
    return schema


def _business_metrics_value_guard() -> dict[str, Any]:
    """An incomplete state sample has empty metrics and unknown truth."""
    return {"oneOf": [
        {
            "properties": {
                "metrics": {"type": "object", "maxProperties": 0},
                "value": {"const": "unknown"},
            },
            "required": ["metrics", "value"],
        },
        {
            "properties": {"metrics": {"type": "object", "minProperties": 1}},
            "required": ["metrics"],
        },
    ]}


def _business_branches(authority: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tick_schema = arm_predicate_truth_ticks_schema()
    for annotation in ("$schema", "$id", "title"):
        tick_schema.pop(annotation)
    if authority["capture_step_ticks"] != 5:
        raise ValueError("L4 business capture step changed; review truth ticks")
    if authority["authority_scope"] != tick_schema["properties"]["authority_scope"]["const"]:
        raise ValueError("L4 business authority scope differs from tick schema")
    rules = authority["profiles"]
    variants = tick_schema["allOf"][0]["oneOf"]
    by_ids = {
        frozenset(variant["properties"]["predicate_id"]["enum"]): variant
        for variant in variants
    }
    if len(by_ids) != len(variants):
        raise ValueError("L4 business predicate groups overlap")
    bindings = tick_schema["allOf"][1]["oneOf"]
    entities_binding = next(
        item for item in bindings
        if "entities" in item["properties"]["bindings"]["properties"]
    )
    vehicles_binding = next(
        item for item in bindings
        if "vehicles" in item["properties"]["bindings"]["properties"]
    )
    by_mechanism: dict[str, set[str]] = {}
    ids_by_mechanism: dict[str, frozenset[str]] = {}
    for profile, rule in rules.items():
        mechanism = rule["mechanism"]
        predicate_ids = frozenset(rule["predicate_ids"])
        if mechanism in ids_by_mechanism and ids_by_mechanism[mechanism] != predicate_ids:
            raise ValueError(f"{mechanism}: L4 business predicate IDs disagree")
        ids_by_mechanism[mechanism] = predicate_ids
        by_mechanism.setdefault(mechanism, set()).add(profile)
    if set(ids_by_mechanism.values()) != set(by_ids):
        raise ValueError("L4 business tick schema differs from current profile rules")

    mechanism_branches = []
    for mechanism, profiles in sorted(by_mechanism.items()):
        branch = deepcopy(by_ids[ids_by_mechanism[mechanism]])
        branch["properties"]["episode_id"] = {
            "type": "string", "pattern": _episode_pattern(profiles),
        }
        branch["required"].append("episode_id")
        mechanism_branches.append(branch)

    current = deepcopy(tick_schema)
    current["properties"]["tick"] = {
        "type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5,
    }
    current["required"].append("rule_digest")
    current["allOf"] = [
        {"oneOf": mechanism_branches}, deepcopy(entities_binding),
        _business_metrics_value_guard(),
    ]

    legacy = deepcopy(tick_schema)
    legacy["properties"].pop("rule_digest")
    legacy["properties"]["tick"] = deepcopy(current["properties"]["tick"])
    intersection = next(
        branch for branch in mechanism_branches
        if frozenset(branch["properties"]["predicate_id"]["enum"])
        == ids_by_mechanism["intersection"]
    )
    legacy["allOf"] = [
        deepcopy(intersection), deepcopy(vehicles_binding),
        _business_metrics_value_guard(),
    ]
    return current, legacy


def arm_predicate_truth_schema() -> dict[str, Any]:
    """Return the five closed L4 JSONL record envelopes."""
    if set(L4_INCIDENT_FAMILY) != _DECLARED_PROFILES:
        raise ValueError("L4 profile declaration changed; review this grammar")
    authority = business_contract()
    rules = authority["profiles"]
    if set(rules) != _BUSINESS_PROFILES:
        raise ValueError("L4 business profile rules changed; review this grammar")
    formal_profiles = frozenset(_DECLARED_PROFILES - _BUSINESS_PROFILES)
    formal = formal_predicate_truth_schema()
    current_business, legacy_business = _business_branches(authority)
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_predicate_truth_l4.record.schema.json",
        "title": "L4 ARM predicate truth JSONL record",
        "oneOf": [
            _formal_branch(formal, frozen=True, digests=False,
                           profiles=formal_profiles),
            _formal_branch(formal, frozen=True, digests=True,
                           profiles=formal_profiles),
            _formal_branch(formal, frozen=False, digests=True,
                           profiles=formal_profiles),
            current_business,
            legacy_business,
        ],
    }
