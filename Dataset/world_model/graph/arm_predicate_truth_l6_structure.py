"""Whole-row grammar for frozen L6 ARM ``objective/predicate_truth.jsonl``.

``l6_v2.semantics.evaluate`` constructs assertions from its fifteen declared
predicate ids.  ``minimal_semantics_adapter._adapt_assertions`` wraps one
observation per assertion; ``l6_v2.observations.window_semantics`` crops those
rows to the ARM window.  Frozen rows are version 3.0.0 without update ticks.
L6-1/2 rows carry three root digests; L6-3/4/5 rows carry none.

The current adapter asks ``ObservationRow.values`` for two update ticks that
the L6 writer does not supply, so its 3.1.0 path cannot currently emit a row.
The two frozen envelopes here are independent of that producer gap.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.world_model.graph.arm_predicate_ticks_l6_structure import (
    _L6_PREDICATES,
    _context_for,
)
from Dataset.world_model.graph.predicate_structure import (
    formal_predicate_truth_schema,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_DIGEST_FIELDS = ("rule_digest", "parameter_digest", "input_digest")
_UPDATE_FIELDS = ("truth_state_update_tick", "evidence_update_tick")
_DECLARED_PREDICATES = frozenset({
    "airspace.aircraft_airborne",
    "airspace.aircraft_touchdown",
    "airspace.aircraft_at_home_pad",
    "airspace.aircraft_in_landing_zone",
    "airspace.aircraft_outside_assigned_altitude_band",
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
})
_BASE_PREDICATES = (
    "airspace.aircraft_airborne",
    "airspace.aircraft_touchdown",
    "airspace.aircraft_at_home_pad",
    "airspace.aircraft_in_landing_zone",
    "airspace.aircraft_outside_assigned_altitude_band",
    "communication.link_degraded",
    "communication.link_unavailable",
    "emergency.aircraft_in_return_to_home",
    "emergency.aircraft_in_safe_hold",
    "positioning.aircraft_deviates_from_planned_route",
)
_GNSS_ADDITIONS = (
    "positioning.gnss_quality_degraded",
    "positioning.gnss_spoofing_active",
    "positioning.navigation_error_exceeds_tolerance",
)
_REF_TICK_ENTITY = r"#tick=[0-9]+&entity=[^\s/&]+"
_TRAJECTORY_REF = r"^raw/trajectories\.jsonl" + _REF_TICK_ENTITY + r"$"
_COMM_REF = r"^raw/communication_state\.jsonl" + _REF_TICK_ENTITY + r"$"
_CONTROL_REF = r"^raw/control_observations\.jsonl" + _REF_TICK_ENTITY + r"$"
_GNSS_REF = (
    r"^raw/domain_observations\.jsonl#tick=[0-9]+&subject=[^\s/&]+"
    r"&family=gnss_navigation$"
)
_SECURITY_REF = (
    r"^raw/domain_observations\.jsonl#tick=[0-9]+&subject=[^\s/&]+"
    r"&family=security_command$"
)


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    }


def _source_refs(predicate_id: str) -> dict[str, Any]:
    # semantics.evaluate starts with the primary trajectory reference.  Each
    # branch then writes at most one fixed-form additional reference.  An
    # airspace pad row and an assigned communication link can be absent.
    second: str
    optional = False
    if predicate_id.startswith("airspace."):
        second = _TRAJECTORY_REF
        optional = True
    elif predicate_id.startswith("communication."):
        second = _COMM_REF
        optional = True
    elif predicate_id.startswith("positioning.gnss") or predicate_id == \
            "positioning.navigation_error_exceeds_tolerance":
        second = _GNSS_REF
    elif predicate_id.startswith("security."):
        second = _SECURITY_REF
    elif predicate_id.startswith("emergency.") or predicate_id == \
            "positioning.aircraft_deviates_from_planned_route":
        second = _CONTROL_REF
    else:
        raise ValueError(f"L6 truth reference branch is unknown: {predicate_id}")
    return {
        "type": "array",
        "prefixItems": [
            {"type": "string", "pattern": _TRAJECTORY_REF},
            {"type": "string", "pattern": second},
        ],
        "items": False, "minItems": 1 if optional else 2, "maxItems": 2,
    }


def _observation_value(predicate_id: str, formal_value: dict[str, Any]) -> dict[str, Any]:
    """Add the L6 producer's empty-or-one-context payload to formal roles."""
    common = {
        "bindings": deepcopy(formal_value["properties"]["bindings"]),
        "binding_ontology_classes": deepcopy(
            formal_value["properties"]["binding_ontology_classes"]
        ),
        "world_truth_value": {
            "enum": ["true", "false", "unknown", "out_of_scope"],
        },
    }
    context = _context_for(predicate_id)
    alternatives = context.get("oneOf")
    if not isinstance(alternatives, list) or len(alternatives) != 2:
        raise ValueError(f"L6 context contract changed: {predicate_id}")
    branches = []
    for entry in alternatives:
        if entry.get("type") != "object" or entry.get("additionalProperties") is not False:
            raise ValueError(f"L6 context is not closed: {predicate_id}")
        if set(entry.get("required", ())) != set(entry.get("properties", ())):
            raise ValueError(f"L6 context fields became optional: {predicate_id}")
        branches.append(_object({**deepcopy(common), **deepcopy(entry["properties"])}))
    return {"oneOf": branches}


def _predicate_variant(formal_variant: dict[str, Any], predicate_id: str) -> dict[str, Any]:
    branch = deepcopy(formal_variant)
    evidence = branch["properties"]["evidence"]
    evidence["properties"]["source_refs"] = _source_refs(predicate_id)
    evidence["properties"]["missing_requirements"] = {
        "type": "array", "items": False, "minItems": 0, "maxItems": 0,
    }
    observations = evidence["properties"]["observations"]
    observations["minItems"] = 1
    observations["maxItems"] = 1
    observation = observations["items"]["oneOf"][0]
    formal_value = observation["properties"]["value"]
    observation["properties"]["value"] = _observation_value(
        predicate_id, formal_value,
    )
    return branch


def _frozen_branch(formal: dict[str, Any], *, profile_group: str,
                   predicate_ids: tuple[str, ...], digests: bool) -> dict[str, Any]:
    schema = deepcopy(formal)
    for annotation in ("$schema", "$id", "title"):
        schema.pop(annotation)
    properties = schema["properties"]
    required = schema["required"]
    identity = rf"(?:{profile_group})__seed(?:00|01|02)"
    arm_key = rf"(?:{profile_group})/seed(?:00|01|02)/grp_[a-z0-9_]+/arm_[a-z0-9_]+"
    properties["schema_version"] = {"const": "3.0.0"}
    properties["episode_id"] = {
        "type": "string", "pattern": rf"^{identity}::{arm_key}$",
    }
    properties["parent_episode_id"] = {
        "type": "string", "pattern": rf"^{identity}$",
    }
    properties["arm_key"] = {
        "type": "string", "pattern": rf"^{arm_key}$",
    }
    required.extend(("parent_episode_id", "arm_key", "mechanism_id"))
    properties["truth_id"] = {
        "type": "string", "pattern": r"^truth:[0-9a-f]{24}$",
    }
    properties["tuple_id"] = {
        "type": "string", "pattern": r"^arm_tuple:[0-9a-f]{24}$",
    }
    properties["tick"] = {
        "type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5,
    }
    properties["source_class"] = {"const": "simulated_derived"}
    properties["mechanism_id"] = {"const": None}
    for field in _UPDATE_FIELDS:
        properties.pop(field)
        required.remove(field)
    if not digests:
        for field in _DIGEST_FIELDS:
            properties.pop(field)
            required.remove(field)
    formal_variants = {
        variant["properties"]["predicate_id"]["const"]: variant
        for variant in schema["allOf"][0]["oneOf"]
    }
    if set(_L6_PREDICATES) != _DECLARED_PREDICATES or not _DECLARED_PREDICATES <= formal_variants.keys():
        raise ValueError("L6 predicate set differs from the declared producer or registry")
    schema["allOf"][0]["oneOf"] = [
        _predicate_variant(formal_variants[predicate_id], predicate_id)
        for predicate_id in predicate_ids
    ]
    return schema


def arm_predicate_truth_schema() -> dict[str, Any]:
    """Return the two closed, frozen L6 truth-record envelopes."""
    formal = formal_predicate_truth_schema()
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_predicate_truth_l6.record.schema.json",
        "title": "Frozen L6 ARM predicate truth JSONL record",
        "oneOf": [
            _frozen_branch(
                formal, profile_group=r"L6-[12]_v[12]",
                predicate_ids=_BASE_PREDICATES, digests=True,
            ),
            _frozen_branch(
                formal, profile_group=r"L6-3_v[12]",
                predicate_ids=(*_BASE_PREDICATES, *_GNSS_ADDITIONS),
                digests=False,
            ),
            _frozen_branch(
                formal, profile_group=r"L6-4_v[12]",
                predicate_ids=(*_BASE_PREDICATES,
                               "security.communication_jamming_impact_active"),
                digests=False,
            ),
            _frozen_branch(
                formal, profile_group=r"L6-5_v[12]",
                predicate_ids=(*_BASE_PREDICATES,
                               "security.ground_control_station_compromised"),
                digests=False,
            ),
        ],
    }
