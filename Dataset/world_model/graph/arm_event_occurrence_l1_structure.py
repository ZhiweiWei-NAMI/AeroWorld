"""Closed whole-record grammar for L1 ARM event-occurrence JSONL rows.

All 84 declared L1 ARM entry scripts call
``Dataset.tools.l1_window.execute_config``.  That writer obtains occurrence
rows from ``minimal_semantics_adapter._adapt_occurrences``, admits occurrences
whose ``trigger_tick`` is inside the ARM window, and removes only evidence
outside the window before writing ``objective/event_occurrences.jsonl``.
Because the adapter makes the onset phase tick equal to ``trigger_tick``, the
window keeps the formal root's nonempty phase and evidence containers.

The current L1 configurations select four registry-backed event rules:
restricted-region intrusion, altitude excursion, UAV separation conflict, and
corridor congestion.  The producer-owned formal schema contains other event
families, so this module starts from its detached schema and retains only those
four branches.  ``Dataset.tools.arm_window_semantics`` writes a different,
smaller occurrence envelope and is not called by any declared L1 entry script.

``event_subtype`` remains optional and ``mechanism_id`` remains string-or-null
because the direct adapter and formal schema permit those branches.  They were
not instantiated in the bounded six-file sample.  The required top-level
``rule_digest`` stays in the raw grammar; publication excludes only its exact
path ``("rule_digest",)`` in ``structure_schema``.
"""

from __future__ import annotations

from typing import Any

from Dataset.world_model.graph.event_structure import (
    formal_event_occurrence_schema,
)


_L1_RULE_IDS = (
    "event_family.restricted_region_intrusion.0",
    "event_family.altitude_excursion.0",
    "event_family.uav_separation_conflict.0",
    "event_family.corridor_congestion.0",
)


def _formal_rule_branches(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the formal alternatives by their required ``rule_id`` const."""

    all_of = schema.get("allOf")
    if type(all_of) is not list or len(all_of) != 1:
        raise ValueError("formal event occurrence schema has an unexpected allOf layout")
    selector = all_of[0]
    if type(selector) is not dict or type(selector.get("oneOf")) is not list:
        raise ValueError("formal event occurrence schema lacks its rule selector")

    result: dict[str, dict[str, Any]] = {}
    for branch in selector["oneOf"]:
        if type(branch) is not dict:
            raise ValueError("formal event occurrence rule branch is not an object")
        properties = branch.get("properties")
        if type(properties) is not dict:
            raise ValueError("formal event occurrence rule branch lacks properties")
        rule_property = properties.get("rule_id")
        if type(rule_property) is not dict:
            raise ValueError("formal event occurrence rule branch lacks rule_id")
        rule_id = rule_property.get("const")
        if type(rule_id) is not str or not rule_id:
            raise ValueError("formal event occurrence rule branch lacks a rule_id const")
        if rule_id in result:
            raise ValueError(f"formal event occurrence rule repeats: {rule_id}")
        result[rule_id] = branch
    return result


def arm_event_occurrences_schema() -> dict[str, Any]:
    """Return the closed Draft 2020-12 schema for one L1 ARM occurrence row."""

    schema = formal_event_occurrence_schema()
    by_rule = _formal_rule_branches(schema)
    missing = [rule_id for rule_id in _L1_RULE_IDS if rule_id not in by_rule]
    if missing:
        raise ValueError(f"L1 event occurrence rules are absent from authority: {missing}")

    schema["allOf"][0]["oneOf"] = [by_rule[rule_id] for rule_id in _L1_RULE_IDS]
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "arm_event_occurrence_l1.record.schema.json"
    )
    schema["title"] = "L1 ARM event occurrence record"
    schema["description"] = (
        "Closed whole-record grammar for one objective/event_occurrences.jsonl "
        "row written by the L1 ARM window producer. Four rule_id branches are "
        "selected from the current formal event-occurrence authority."
    )
    return schema


__all__ = ["arm_event_occurrences_schema"]
