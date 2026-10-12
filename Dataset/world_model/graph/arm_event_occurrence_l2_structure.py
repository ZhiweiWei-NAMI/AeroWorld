"""Closed JSONL-row grammar for L2 ARM event occurrences.

``Dataset/tools/l2_arm_semantics.py::recompute`` builds the formal occurrence
root through ``minimal_semantics_adapter._adapt_occurrences`` and then stamps
``parent_episode_id`` and ``arm_key`` on every row.  ``l2_window._execute``
keeps events whose onset is in the window or whose lifecycle spans the window
boundary, without reshaping the occurrence record.

The shared formal schema owns the event-rule variants, participant bindings,
phase sequences, evidence arrays, and optional producer branches.  This module
selects the thirteen rule variants reachable from the predicates instantiated
by ``l2_arm_semantics.scope_for`` and adds the L2 ARM identity envelope.  The
identity patterns follow ``rules/l2_window.schema.json`` and the equality of
profile, seed, group, and arm across the three identity fields remains a
producer invariant.  L2 formal event ticks are bounded by the validated
0..900 window and the five-tick formal sampling step; an occurrence may still
start before its saved window when its lifecycle spans the boundary.

``rule_digest`` remains a required raw field from the formal schema.  Its exact
publication-exclusion path is the root path ``("rule_digest",)``; raw record
validation does not apply that projection.
"""

from __future__ import annotations

from typing import Any

from Dataset.world_model.graph.event_structure import (
    formal_event_occurrence_schema,
)

__all__ = ["arm_event_occurrences_schema"]

_L2_PROFILES = (
    "L2-1_v1",
    "L2-1_v2",
    "L2-2_v1",
    "L2-2_v2",
    "L2-3_v1",
    "L2-3_v2",
    "L2-4_v1",
    "L2-4_v2",
    "L2-5_v1",
)
_PROFILE_PATTERN = "(?:" + "|".join(_L2_PROFILES) + ")"
_SEED_PATTERN = "(?:00|01|02)"
_ARM_SUFFIX_PATTERN = r"/[^/]+/arm_[a-z0-9_]+"

# These are the formal rules whose trigger predicates ``scope_for`` can
# instantiate.  Keeping the list producer-based avoids treating an observed
# union from the current corpus as the rule authority.
_L2_RULE_IDS = frozenset({
    "event_family.uav_takeoff.0",
    "event_family.uav_landing.0",
    "event_family.altitude_excursion.0",
    "event_family.uav_separation_conflict.0",
    "event_family.corridor_congestion.0",
    "event_family.structure_near_miss.0",
    "event_family.c2_degradation_loss.0",
    "event_family.c2_degradation_loss.1",
    "event_family.gnss_signal_degradation.0",
    "event_family.charger_unavailable.0",
    "event_family.pad_contention.0",
    "event_family.signal_malfunction.0",
    "event_family.charging_queue_over_capacity.0",
})


def _tick() -> dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 0,
        "maximum": 900,
        "multipleOf": 5,
    }


def _select_l2_rule_variants(schema: dict[str, Any]) -> None:
    variants = schema["allOf"][0]["oneOf"]
    selected = [
        variant
        for variant in variants
        if variant["properties"]["rule_id"]["const"] in _L2_RULE_IDS
    ]
    selected_ids = {
        variant["properties"]["rule_id"]["const"] for variant in selected
    }
    if selected_ids != _L2_RULE_IDS:
        missing = sorted(_L2_RULE_IDS - selected_ids)
        raise ValueError(f"formal event schema lacks L2 rule variants: {missing}")
    schema["allOf"][0]["oneOf"] = selected


def _apply_l2_tick_contract(schema: dict[str, Any]) -> None:
    for name in ("trigger_tick", "detection_tick", "start_tick", "end_tick"):
        schema["properties"][name] = _tick()
    for variant in schema["allOf"][0]["oneOf"]:
        phase_sequences = variant["properties"]["event_phases"]["oneOf"]
        for sequence in phase_sequences:
            for phase in sequence["prefixItems"]:
                phase["properties"]["tick"] = _tick()


def arm_event_occurrences_schema() -> dict[str, Any]:
    """Return the closed schema for one L2 ARM occurrence JSONL row."""
    schema = formal_event_occurrence_schema()
    _select_l2_rule_variants(schema)
    _apply_l2_tick_contract(schema)

    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "arm_event_occurrence_l2.record.schema.json"
    )
    schema["title"] = "AERO_WORLD arm_event_occurrences L2 JSONL row schema"
    schema["description"] = (
        "Formal event occurrence with the required L2 ARM parent and branch "
        "identity envelope."
    )
    schema["properties"]["episode_id"] = {
        "type": "string",
        "pattern": (
            "^" + _PROFILE_PATTERN + "__seed" + _SEED_PATTERN + "::"
            + _PROFILE_PATTERN + "/seed" + _SEED_PATTERN
            + _ARM_SUFFIX_PATTERN + "$"
        ),
    }
    schema["properties"]["parent_episode_id"] = {
        "type": "string",
        "pattern": (
            "^" + _PROFILE_PATTERN + "__seed" + _SEED_PATTERN + "$"
        ),
    }
    schema["properties"]["arm_key"] = {
        "type": "string",
        "pattern": (
            "^" + _PROFILE_PATTERN + "/seed" + _SEED_PATTERN
            + _ARM_SUFFIX_PATTERN + "$"
        ),
    }
    schema["required"].extend(("parent_episode_id", "arm_key"))
    return schema
