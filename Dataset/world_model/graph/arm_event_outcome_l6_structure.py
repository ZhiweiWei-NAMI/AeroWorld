"""Closed whole-row grammar for L6 ARM event outcomes.

The current L6 writer builds formal outcomes with ``_adapt_outcomes``, selects
the bounded event set, changes the envelope through ``arm_window_outcome``, and
writes the returned rows unchanged.  Frozen L6 rows predate that envelope and
retain the writer's former ``parent_episode_id`` and ``arm_key`` projection.

Both envelopes keep the registry-bound event, terminal-predicate, binding, and
evidence branches from the formal outcome grammar.  The L6 runtime selects the
seven event families declared below; the current composition can emit only the
``pending`` and ``succeeded`` statuses produced by ``_adapt_outcomes``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.world_model.graph.event_structure import formal_event_outcome_schema


__all__ = ["arm_event_outcomes_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_L6_EVENT_FAMILIES = frozenset(
    {
        "uav_takeoff",
        "uav_landing",
        "altitude_excursion",
        "c2_degradation_loss",
        "gnss_signal_degradation",
        "gnss_spoofing_effect",
        "jamming_gcs_intrusion",
    }
)
_PROFILE = r"(?:L6-[1-5]_v[12])"
_SEED = r"seed(?:00|01|02)"
_ARM_SEGMENT = r"[a-z0-9_]+"
_PARENT_EPISODE_PATTERN = rf"^{_PROFILE}__{_SEED}$"
_ARM_KEY_PATTERN = (
    rf"^{_PROFILE}/{_SEED}/grp_{_ARM_SEGMENT}/{_ARM_SEGMENT}$"
)
_EPISODE_PATTERN = (
    rf"^{_PROFILE}__{_SEED}::{_PROFILE}/{_SEED}/"
    rf"grp_{_ARM_SEGMENT}/{_ARM_SEGMENT}$"
)


def _l6_formal_schema() -> dict[str, Any]:
    """Select the formal event branches reachable from the L6 runtime."""

    schema = formal_event_outcome_schema()
    all_of = schema.get("allOf")
    if (
        type(all_of) is not list
        or len(all_of) != 1
        or type(all_of[0]) is not dict
        or type(all_of[0].get("oneOf")) is not list
    ):
        raise ValueError("formal event-outcome schema has no family selector")

    selected = []
    selected_families: set[str] = set()
    for branch in all_of[0]["oneOf"]:
        if type(branch) is not dict or type(branch.get("properties")) is not dict:
            raise ValueError("formal event-outcome family branch lacks properties")
        family_property = branch["properties"].get("event_family_id")
        if type(family_property) is not dict:
            raise ValueError("formal event-outcome branch lacks event_family_id")
        family = family_property.get("const")
        if family in _L6_EVENT_FAMILIES:
            selected.append(branch)
            selected_families.add(family)
    if selected_families != _L6_EVENT_FAMILIES:
        missing = sorted(_L6_EVENT_FAMILIES - selected_families)
        raise ValueError(f"formal authority lacks L6 event families: {missing}")
    all_of[0]["oneOf"] = selected
    return schema


def _envelope(formal: dict[str, Any], *, frozen: bool) -> dict[str, Any]:
    """Detach the formal schema and bind one exact L6 output envelope."""

    schema = deepcopy(formal)
    for annotation in ("$schema", "$id", "title", "description"):
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
        raise ValueError("formal event-outcome root is not closed and fully required")

    properties["episode_id"] = {
        "type": "string",
        "pattern": _EPISODE_PATTERN,
    }
    if frozen:
        properties["parent_episode_id"] = {
            "type": "string",
            "pattern": _PARENT_EPISODE_PATTERN,
        }
        properties["arm_key"] = {
            "type": "string",
            "pattern": _ARM_KEY_PATTERN,
        }
    else:
        properties["schema_name"] = {"const": "arm_window_event_outcome"}
        properties["schema_version"] = {"const": "1.0.0"}
    schema["required"] = list(properties)
    return schema


def arm_event_outcomes_schema() -> dict[str, Any]:
    """Return the closed grammar for one L6 event-outcomes JSONL row."""

    formal = _l6_formal_schema()
    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_event_outcome.l6.record.schema.json"
        ),
        "title": "L6 ARM event outcome record",
        "description": (
            "One objective/event_outcomes.jsonl row from an L6 ARM branch, "
            "using either the frozen formal envelope or the current bounded-window "
            "envelope."
        ),
        "oneOf": [
            _envelope(formal, frozen=True),
            _envelope(formal, frozen=False),
        ],
    }
