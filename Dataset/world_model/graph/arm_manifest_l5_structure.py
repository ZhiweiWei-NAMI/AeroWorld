"""Closed root grammar for L5 ARM manifests.

``l5_arm_pipeline.execute_config`` writes the candidate root and
``l5_family_delivery.deliver_config`` adds the UE package fields.  Both
complete writer states are declared without reading data on import.
"""

from __future__ import annotations

from copy import deepcopy

__all__ = ["arm_manifest_schema"]

_SOURCE_KEYS = (
    "scene", "script", "trajectories", "weather", "roster",
    "event_realization", "event_trace", "episode_manifest",
    "scenario_plan", "event_occurrences", "event_outcomes",
    "predicate_transitions", "semantic_manifest", "runtime_schedule",
)
_COUNT_KEYS = (
    "trajectories", "states", "weather", "predicate_truth",
    "predicate_truth_ticks", "predicate_transitions", "event_occurrences",
    "event_outcomes", "unresolved_predicate_records",
)


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def _string() -> dict:
    return {"type": "string", "minLength": 1}


def _strings() -> dict:
    return {"type": "array", "items": _string()}


def _integer() -> dict:
    return {"type": "integer", "minimum": 0}


def _window(*, basis: bool = False) -> dict:
    fields = {"start_tick": _integer(), "fork_tick": _integer(),
              "end_tick": _integer()}
    if basis:
        fields["basis"] = _string()
    else:
        fields["phase"] = {"const": "after_state_sample_before_actions"}
        fields["intervention_order"] = {
            "const": "source_actions_then_intervention"}
    return _obj(fields)


def _provenance() -> dict:
    delay = _obj({
        "value": _integer(), "source": _string(), "rule": _string(),
        "digest_prefix": _string(), "grid": _string(),
        "recompute": _string(),
    })
    sources = _obj({
        "factual_root": _string(), "thresholds": _string(),
        "branch_definition": _string(), "runtime_schedule": _string(),
        "branch_environment": _string(), "delay_ticks": delay,
    }, ["factual_root", "thresholds", "branch_definition",
        "runtime_schedule", "branch_environment"])
    return _obj({
        "experiment_ref": _string(), "parameter_sources": sources,
        "window": _window(basis=True), "implementation": _string(),
    })


def _base() -> dict:
    properties = {
        "schema": {"const": "aeroworld_l5_candidate_window"},
        "version": {"const": 1}, "arm_id": _string(),
        "arm_key": {"type": "string", "pattern": r"^L5-[1-3]_v[123]/seed[0-9]{2}/grp_[a-z0-9_]+/[a-z0-9_]+$"},
        "episode_id": {"type": "string", "pattern": r"^L5-[1-3]_v[123]__seed[0-9]{2}$"},
        "seed": _integer(),
        "mode": {"enum": ["factual_observation", "intervention"]},
        "window": _window(), "source_version": _string(),
        "source_refs": _obj({key: _string() for key in _SOURCE_KEYS}),
        "configuration_ref": _string(), "experiment_ref": _string(),
        "provenance": _provenance(),
        "entity_ids": _strings(), "primary_entity_id": _string(),
        "predicate_ids": _strings(),
        "blocked_event_families": {"type": "array", "items": _obj({
            "event_family_id": _string(), "status": {"const": "blocked"},
            "reason": _string(), "missing": _strings(),
        })},
        "record_counts": _obj({key: _integer() for key in _COUNT_KEYS}),
        "declaration": {"type": "null"},
        "runtime": _obj({
            "python": _string(), "executable": _string(),
            "platform": _string(), "started_utc": _string(),
            "completed_utc": _string(), "random_seed": _integer(),
            "primary_engine_rng": _string(),
        }),
        "certification": _obj({
            "local_candidate_asset": _string(),
            "historical_counterfactual": _string(),
            "full_L0_producer": _string(), "UE_capture": _string(),
            "release": _string(),
        }),
        "limitations": _strings(), "artifacts": _strings(),
    }
    return _obj(properties)


def arm_manifest_schema() -> dict:
    """Return the closed L5 candidate and UE delivery root branches."""
    candidate = _base()
    delivered = deepcopy(candidate)
    for key in ("ue_package_ref", "ue_validation_ref"):
        delivered["properties"][key] = _string()
        delivered["required"].append(key)
    delivered["properties"]["ue_capture_status"] = {
        "const": "input_package_validated_capture_not_executed"}
    delivered["required"].append("ue_capture_status")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_manifest_l5.record.schema.json",
        "title": "AERO_WORLD arm_manifest L5 record schema",
        "oneOf": [candidate, delivered],
    }
