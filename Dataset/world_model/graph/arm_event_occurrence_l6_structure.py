"""Closed whole-record structure for L6 ARM event occurrence JSONL rows.

The dedicated L6 ARM entries call ``l6_v2.arm_runtime.run_arm``. Its semantic
writer calls ``minimal_semantics_adapter._adapt_occurrences`` with rules from
the seven families selected by the current L6 primary predicates. Window
projection retains event rows and may rewrite pre-window raw references to a
history file. The separate ``arm_window_semantics`` writer is not on this path.

The current adapter writes ``rule_digest``. Some retained L6 rows predate that
field, so the raw field remains typed and optional. Its exact publication
exclusion path is already ``("rule_digest",)`` in ``structure_schema``.
"""

from __future__ import annotations

from typing import Any

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
)
from Dataset.tools.l6_v2.arm_runtime import BASE_FAMILIES, FAMILY_ADDITIONS
from Dataset.world_model.graph.event_structure import (
    formal_event_occurrence_schema,
)


_PRIMARY_PREDICATES = (
    "communication.link_unavailable",
    "communication.link_degraded",
    "positioning.gnss_spoofing_active",
    "security.communication_jamming_impact_active",
    "security.ground_control_station_compromised",
)
_PROFILE = r"L6-[1-5]_v[12]"
_SEED = r"seed[0-9]{2}"
_HEX24 = r"[0-9a-f]{24}"
_REF_TICK = r"(?:0|[1-9][0-9]?|[1-8][0-9]{2}|900)"
_HISTORY_PREFIX = (
    rf"design/{_PROFILE}/{_SEED}/grp_[^/]+/history/"
    r"[^/]+_[0-9a-f]{16}\.json\.gz#"
)
_TRAJECTORY_REF = rf"raw/trajectories\.jsonl#tick={_REF_TICK}&entity=[^&]+"
_COMMUNICATION_REF = rf"raw/communication_state\.jsonl#tick={_REF_TICK}&entity=[^&]+"
_GNSS_REF = (
    rf"raw/domain_observations\.jsonl#tick={_REF_TICK}"
    r"&subject=[^&]+&family=gnss_navigation"
)
_SECURITY_REF = (
    rf"raw/domain_observations\.jsonl#tick={_REF_TICK}"
    r"&subject=[^&]+&family=security_command"
)


def _active_rules() -> tuple[dict[str, Any], ...]:
    families = set(BASE_FAMILIES)
    for predicate_id in _PRIMARY_PREDICATES:
        for prefix, additions in FAMILY_ADDITIONS.items():
            if predicate_id.startswith(prefix):
                families.update(additions)
    rules = tuple(
        rule for rule in get_core_event_occurrence_types()
        if rule["event_family_id"] in families
    )
    if not rules:
        raise ValueError("L6 ARM primary predicates select no event rules")
    return rules


def _raw_reference_for_predicate(predicate_id: str) -> str:
    if predicate_id.startswith("airspace."):
        return _TRAJECTORY_REF
    if predicate_id.startswith("communication."):
        return _COMMUNICATION_REF
    if predicate_id.startswith("positioning.gnss"):
        return _GNSS_REF
    if predicate_id.startswith("security."):
        return _SECURITY_REF
    raise ValueError(f"L6 occurrence predicate has no raw source rule: {predicate_id}")


def _source_ref(body: str) -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": rf"^(?:{body}|{_HISTORY_PREFIX}{body})$",
    }


def _rule_source_refs(rule: dict[str, Any]) -> dict[str, Any]:
    predicate_ids = (
        rule["trigger_predicate_id"],
        *(condition["predicate_id"] for condition in rule["support_conditions"]),
        rule["terminal"]["predicate_id"],
    )
    bodies = [_TRAJECTORY_REF]
    for predicate_id in predicate_ids:
        body = _raw_reference_for_predicate(predicate_id)
        if body not in bodies:
            bodies.append(body)
    return {
        "type": "array",
        "items": {"oneOf": [_source_ref(body) for body in bodies]},
        "minItems": 1,
        "uniqueItems": True,
    }


def arm_event_occurrences_schema() -> dict[str, Any]:
    """Return the L6 schema for one complete occurrence JSONL row."""

    schema = formal_event_occurrence_schema()
    rules = _active_rules()
    branches = {
        branch["properties"]["rule_id"]["const"]: branch
        for branch in schema["allOf"][0]["oneOf"]
    }
    missing = [rule["rule_id"] for rule in rules if rule["rule_id"] not in branches]
    if missing:
        raise ValueError(f"L6 event rules are absent from formal structure: {missing}")
    schema["allOf"][0]["oneOf"] = [branches[rule["rule_id"]] for rule in rules]
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "arm_event_occurrence_l6.record.schema.json"
    )
    schema["title"] = "L6 ARM event occurrence record"

    properties = schema["properties"]
    properties.pop("event_subtype")  # build_events sets it to None; adapter omits it.
    properties["mechanism_id"] = {"const": None}  # L6 formal observations stamp None.
    properties["event_id"] = {
        "type": "string", "pattern": rf"^objective_event:{_HEX24}$"
    }
    properties["episode_id"] = {
        "type": "string",
        "pattern": (
            rf"^{_PROFILE}__{_SEED}::{_PROFILE}/{_SEED}/"
            r"grp_[^/]+/arm_[^/]+$"
        ),
    }
    properties["parent_episode_id"] = {
        "type": "string", "pattern": rf"^{_PROFILE}__{_SEED}$"
    }
    properties["arm_key"] = {
        "type": "string",
        "pattern": rf"^{_PROFILE}/{_SEED}/grp_[^/]+/arm_[^/]+$",
    }
    for name in ("trigger_tick", "detection_tick", "start_tick", "end_tick"):
        properties[name] = {
            "type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5
        }
    properties["supporting_truth_ids"]["items"] = {
        "type": "string", "pattern": rf"^truth:{_HEX24}$"
    }
    properties["supporting_transition_ids"]["items"] = {
        "type": "string", "pattern": rf"^transition:{_HEX24}$"
    }
    schema["required"].remove("rule_digest")
    schema["required"].extend(("parent_episode_id", "arm_key"))

    for rule in rules:
        variant = branches[rule["rule_id"]]
        variant_properties = variant["properties"]
        variant_properties["source_refs"] = _rule_source_refs(rule)
        for phase_array in variant_properties["event_phases"]["oneOf"]:
            for phase in phase_array.get("prefixItems", []):
                phase_properties = phase["properties"]
                phase_properties["predicate_tuple_id"] = {
                    "type": "string", "pattern": rf"^arm_tuple:{_HEX24}$"
                }
                phase_properties["transition_id"] = {
                    "type": "string", "pattern": rf"^transition:{_HEX24}$"
                }
                phase_properties["tick"] = {
                    "type": "integer", "minimum": 0, "maximum": 900,
                    "multipleOf": 5,
                }
    return schema


__all__ = ["arm_event_occurrences_schema"]
