"""Closed event-outcome row grammar for L2 ARM windows.

``l2_arm_semantics.recompute`` selects formal event rules by family and sends
their outcomes through ``minimal_semantics_adapter._adapt_outcomes``.  The
adapter emits only ``pending`` and ``succeeded`` rows.  ``l2_window`` then
passes those rows directly to ``window_outcome.arm_window_outcome``, which
changes only the schema header before the rows are written.

Frozen L2 rows retain the earlier ``event_outcome`` header and the branch
identity fields ``parent_episode_id`` and ``arm_key``.  The current writer uses
the ``arm_window_event_outcome`` header and does not add those identity fields
to outcomes.  The two exact envelopes are separate branches over the same
registry-derived formal outcome variants.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
)
from Dataset.world_model.graph.event_structure import formal_event_outcome_schema


__all__ = [
    "arm_event_outcomes_schema",
    "build_standard_arm_event_outcomes_schema",
]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_L2_PROFILE_PATTERN = (
    r"(?:L2-1_v[12]|L2-2_v[12]|L2-3_v[12]|L2-4_v[12]|L2-5_v1)"
)
_SEED_PATTERN = r"(?:00|01|02)"
_ARM_SUFFIX_PATTERN = r"/[^/]+/arm_[a-z0-9_]+"

# ``scope_for`` can select these families over the declared L2 profiles.  The
# registry remains authoritative for their event types and terminal contracts.
_L2_FAMILIES = frozenset({
    "uav_takeoff",
    "uav_landing",
    "altitude_excursion",
    "uav_separation_conflict",
    "corridor_congestion",
    "structure_near_miss",
    "c2_degradation_loss",
    "gnss_signal_degradation",
    "charger_unavailable",
    "pad_contention",
    "signal_malfunction",
    "charging_queue_over_capacity",
})


def _event_types_for_families(families: frozenset[str]) -> frozenset[str]:
    event_types = frozenset(
        rule["event_type"]
        for rule in get_core_event_occurrence_types()
        if rule["event_family_id"] in families
    )
    if not event_types:
        raise ValueError("event-outcome family selection resolved no formal rules")
    return event_types


def _select_event_types(
    formal: dict[str, Any], allowed_event_types: frozenset[str]
) -> None:
    all_of = formal.get("allOf")
    if (
        type(all_of) is not list
        or len(all_of) != 1
        or type(all_of[0]) is not dict
        or type(all_of[0].get("oneOf")) is not list
    ):
        raise ValueError("formal event-outcome schema has no event-rule variants")

    selected = []
    for variant in all_of[0]["oneOf"]:
        if type(variant) is not dict:
            raise ValueError("formal event-outcome rule variant is not an object")
        event_type = (
            variant.get("properties", {})
            .get("event_type_id", {})
            .get("const")
        )
        if event_type in allowed_event_types:
            selected.append(variant)

    selected_types = {
        variant["properties"]["event_type_id"]["const"]
        for variant in selected
    }
    if (
        selected_types != allowed_event_types
        or len(selected) != len(allowed_event_types)
    ):
        missing = sorted(allowed_event_types - selected_types)
        raise ValueError(
            "formal event-outcome variants differ from the selected producer "
            f"rules: missing={missing}"
        )
    all_of[0]["oneOf"] = selected


def _header_envelope(
    formal: dict[str, Any],
    *,
    schema_name: str,
    schema_version: str,
    episode_id_pattern: str,
    extra_properties: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    schema = deepcopy(formal)
    for annotation in ("$schema", "$id", "title"):
        schema.pop(annotation, None)

    properties = schema.get("properties")
    required = schema.get("required")
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or type(properties) is not dict
        or type(required) is not list
        or set(required) != set(properties)
    ):
        raise ValueError("formal event-outcome schema has no closed root envelope")

    properties["schema_name"] = {"const": schema_name}
    properties["schema_version"] = {"const": schema_version}
    properties["episode_id"] = {
        "type": "string",
        "pattern": episode_id_pattern,
    }
    if extra_properties is not None:
        overlap = set(properties) & set(extra_properties)
        if overlap:
            raise ValueError(
                f"event-outcome envelope fields repeat formal fields: {sorted(overlap)}"
            )
        properties.update(deepcopy(dict(extra_properties)))
    schema["required"] = list(properties)
    return schema


def build_standard_arm_event_outcomes_schema(
    *,
    schema_id: str,
    title: str,
    description: str,
    episode_id_pattern: str,
    allowed_event_types: frozenset[str],
    frozen_extra_properties: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the two closed headers shared by the standard ARM outcome chain."""
    for name, value in (
        ("schema_id", schema_id),
        ("title", title),
        ("description", description),
        ("episode_id_pattern", episode_id_pattern),
    ):
        if type(value) is not str or not value:
            raise ValueError(f"{name} must be a nonempty string")
    if type(allowed_event_types) is not frozenset or not allowed_event_types:
        raise TypeError("allowed_event_types must be a nonempty frozenset")
    if any(type(value) is not str or not value for value in allowed_event_types):
        raise ValueError("allowed_event_types contains a non-string or empty value")

    formal = formal_event_outcome_schema()
    _select_event_types(formal, allowed_event_types)
    return {
        "$schema": _DRAFT,
        "$id": schema_id,
        "title": title,
        "description": description,
        "type": "object",
        "oneOf": [
            _header_envelope(
                formal,
                schema_name="event_outcome",
                schema_version="3.0.0",
                episode_id_pattern=episode_id_pattern,
                extra_properties=frozen_extra_properties,
            ),
            _header_envelope(
                formal,
                schema_name="arm_window_event_outcome",
                schema_version="1.0.0",
                episode_id_pattern=episode_id_pattern,
            ),
        ],
    }


def arm_event_outcomes_schema() -> dict[str, Any]:
    """Return the closed grammar for one L2 ARM event-outcome JSONL row."""
    episode_id_pattern = (
        "^"
        + _L2_PROFILE_PATTERN
        + "__seed"
        + _SEED_PATTERN
        + "::"
        + _L2_PROFILE_PATTERN
        + "/seed"
        + _SEED_PATTERN
        + _ARM_SUFFIX_PATTERN
        + "$"
    )
    frozen_extra_properties = {
        "parent_episode_id": {
            "type": "string",
            "pattern": (
                "^" + _L2_PROFILE_PATTERN + "__seed" + _SEED_PATTERN + "$"
            ),
        },
        "arm_key": {
            "type": "string",
            "pattern": (
                "^"
                + _L2_PROFILE_PATTERN
                + "/seed"
                + _SEED_PATTERN
                + _ARM_SUFFIX_PATTERN
                + "$"
            ),
        },
    }
    return build_standard_arm_event_outcomes_schema(
        schema_id=(
            "https://aeroworld.local/schema/"
            "arm_event_outcome.l2.record.schema.json"
        ),
        title="L2 ARM event outcome record",
        description=(
            "One unchanged objective/event_outcomes.jsonl row from a declared "
            "L2 ARM branch."
        ),
        episode_id_pattern=episode_id_pattern,
        allowed_event_types=_event_types_for_families(_L2_FAMILIES),
        frozen_extra_properties=frozen_extra_properties,
    )
