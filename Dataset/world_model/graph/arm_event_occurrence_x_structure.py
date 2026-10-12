"""Closed whole-record grammar for X ARM event-occurrence JSONL rows.

The source and effective family are both ``arm_event_occurrences``.  The 42
current X entries cover the six cross-layer profiles listed in
``_X_PROFILES`` and are executed by ``Dataset/tools/x_arm_pipeline.py``.
``run_arm`` writes ``calculate``'s occurrence list directly to
``objective/event_occurrences.jsonl``.  It does not call the formal minimal
semantics adapter or ``arm_window_semantics``; those writers produce different
record envelopes and are outside this X family.

``calculate`` groups measured predicate rows on the five-tick grid and emits
an occurrence only for a false-to-true transition.  Every emitted root has
exactly seven keys: ``event_id``, ``tick``, ``predicate_id``, ``bindings``,
``kind``, ``ontology_event`` and ``source_refs``.  The kind is always
``measured_business_predicate_onset`` and ``ontology_event`` is always false.
The ten predicate/binding branches come from the producer-owned X predicate
tick schema.  Construction checks their exact order and patterns so a change
in that authority fails instead of silently attaching the wrong evidence
shape.

Evidence arrays are closed to the writer's positional layouts.  Weather
onsets cite one weather row; entity, motion, pedestrian and facility onsets
cite one trajectory row; geometry and separation onsets cite two trajectory
rows; no-fly-zone onsets add the configured scene entity; restricted-airspace
onsets cite the aircraft, region trajectory and scene entity.  None of the 42
fixed X scripts removes a restricted region; the X6 event-script-only region
cannot become active until its spawn row is present.  Equality between the
root tick/bindings and their text inside source references is a producer
invariant that JSON Schema cannot express.

These rows contain no integrity metadata.  ``event_id`` is the producer's
stable occurrence identity and remains part of the raw and published record;
no publication-exclusion path is needed.  This module performs no I/O and
does not register the grammar globally.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from Dataset.world_model.graph.arm_predicate_ticks_x_structure import (
    arm_predicate_truth_ticks_schema,
)


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_X_PROFILES = (
    "X1_rain_to_c2loss_to_forced_landing",
    "X2_gnss_spoof_to_geofence_violation",
    "X3_pedestrian_fall_to_emergency_response",
    "X4_fog_to_uav_conflict",
    "X5_comm_failure_to_pad_contention",
    "X6_crowd_evacuation_to_airspace_lockdown",
)
_PROFILE_ALTERNATION = "(?:" + "|".join(map(re.escape, _X_PROFILES)) + ")"
_TICK_TEXT = r"(?:[0-9]|[1-9][0-9]|[1-8][0-9]{2}|900)"
_ENTITY_TEXT = r"[A-Za-z0-9_.:-]+"

# ``arm_predicate_truth_ticks_schema`` currently returns these branches in
# producer call-site order.  The source-reference layouts below use the same
# order, so fail closed if that dependency changes.
_EXPECTED_PREDICATE_PATTERNS = (
    r"^environment\.(rain_active|fog_active)$",
    (
        r"^x\.runtime\.(communication_state|control_state|facility_state|"
        r"incident_state|mission_state|navigation_state)\."
        r"[a-z][a-z0-9_]*$"
    ),
    r"^x\.motion\.moving$",
    r"^x\.motion\.descending$",
    r"^x\.pedestrian\.fallen$",
    r"^x\.facility\.multiple_requests$",
    r"^x\.geometry\.[^\s]+$",
    r"^safety\.aircraft_pair_below_separation$",
    r"^emergency\.temporary_no_fly_zone_active$",
    r"^airspace\.aircraft_inside_restricted_airspace$",
)


def _reference(pattern: str) -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": pattern,
        "x-source-value-types": ["string"],
    }


def _fixed_references(*items: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": len(items),
        "maxItems": len(items),
        "prefixItems": [deepcopy(dict(item)) for item in items],
        "items": False,
        "uniqueItems": True,
    }


def _source_reference_rules() -> tuple[dict[str, Any], ...]:
    trajectory_entity = _reference(
        rf"^raw/trajectories\.jsonl#tick={_TICK_TEXT}&entity={_ENTITY_TEXT}$"
    )
    weather = _reference(rf"^raw/weather\.jsonl#tick={_TICK_TEXT}$")
    scene = _reference(
        rf"^interventions/{_PROFILE_ALTERNATION}/seed[0-9]{{2}}/"
        rf"grp_cross_layer/scene_setup\.json#entity={_ENTITY_TEXT}$"
    )
    return (
        _fixed_references(weather),
        _fixed_references(trajectory_entity),
        _fixed_references(trajectory_entity),
        _fixed_references(trajectory_entity),
        _fixed_references(trajectory_entity),
        _fixed_references(trajectory_entity),
        _fixed_references(trajectory_entity, trajectory_entity),
        _fixed_references(trajectory_entity, trajectory_entity),
        _fixed_references(trajectory_entity, scene),
        _fixed_references(trajectory_entity, trajectory_entity, scene),
    )


def _occurrence_branch(
    predicate_branch: Mapping[str, Any],
    source_refs: Mapping[str, Any],
    *,
    weather_profile_binding: bool,
) -> dict[str, Any]:
    properties = predicate_branch.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("X predicate branch lacks object properties")
    required = {"tick", "predicate_id", "bindings", "source_refs"}
    if not required.issubset(properties):
        missing = sorted(required.difference(properties))
        raise ValueError(f"X predicate branch lacks occurrence inputs: {missing}")

    tick = deepcopy(properties["tick"])
    tick["multipleOf"] = 5
    bindings = deepcopy(properties["bindings"])
    if weather_profile_binding:
        binding_properties = bindings.get("properties")
        if not isinstance(binding_properties, dict) or set(binding_properties) != {"region"}:
            raise ValueError("X weather predicate binding changed")
        binding_properties["region"] = {
            "type": "string",
            "enum": [profile + ".authored_weather_domain" for profile in _X_PROFILES],
            "x-source-value-types": ["string"],
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "event_id",
            "tick",
            "predicate_id",
            "bindings",
            "kind",
            "ontology_event",
            "source_refs",
        ],
        "properties": {
            "event_id": {
                "type": "string",
                "pattern": r"^xb:[0-9a-f]{64}$",
                "x-source-value-types": ["string"],
            },
            "tick": tick,
            "predicate_id": deepcopy(properties["predicate_id"]),
            "bindings": bindings,
            "kind": {"const": "measured_business_predicate_onset"},
            "ontology_event": {"const": False},
            "source_refs": deepcopy(dict(source_refs)),
        },
    }


def arm_event_occurrences_schema() -> dict[str, Any]:
    """Return the closed schema for one X ARM event-occurrence JSONL row."""

    predicate_schema = arm_predicate_truth_ticks_schema(layer="X")
    predicate_branches = predicate_schema.get("oneOf")
    if not isinstance(predicate_branches, list):
        raise ValueError("X predicate schema lacks its producer branches")
    observed_patterns: list[str] = []
    for branch in predicate_branches:
        if not isinstance(branch, Mapping):
            raise ValueError("X predicate schema contains a non-object branch")
        properties = branch.get("properties")
        predicate_id = properties.get("predicate_id") if isinstance(properties, Mapping) else None
        pattern = predicate_id.get("pattern") if isinstance(predicate_id, Mapping) else None
        if not isinstance(pattern, str):
            raise ValueError("X predicate branch lacks a predicate_id pattern")
        observed_patterns.append(pattern)
    if tuple(observed_patterns) != _EXPECTED_PREDICATE_PATTERNS:
        raise ValueError("X predicate branch authority changed; review occurrence grammar")

    reference_rules = _source_reference_rules()
    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_event_occurrence_x.record.schema.json"
        ),
        "title": "X ARM measured event occurrence record",
        "oneOf": [
            _occurrence_branch(
                branch,
                source_refs,
                weather_profile_binding=index == 0,
            )
            for index, (branch, source_refs) in enumerate(
                zip(predicate_branches, reference_rules, strict=True)
            )
        ],
    }


__all__ = ["arm_event_occurrences_schema"]
