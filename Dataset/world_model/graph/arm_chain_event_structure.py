"""Closed JSONL row schemas for the ARM chain/events source.

L1, L4 and L5 write node-like rows; L2 writes occurrence summaries.  The
L3, L6 and X producers write chain/nodes.jsonl instead, so they have no
chain/events record type.  ``effective_family`` is the discriminator returned
by ``sample_identity.arm_chain_row_family``; a schema covers one such type.
"""

from __future__ import annotations

from copy import deepcopy

from Dataset.world_model.graph.event_structure import formal_event_occurrence_schema


__all__ = ["arm_chain_events_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_KINDS = {
    "L1": frozenset({"predicate_transition", "objective_event", "event_recovery_confirmed"}),
    "L4": frozenset({"predicate_transition", "objective_event", "event_recovery_confirmed",
                     "event_response_confirmed", "local_business_observation"}),
    "L5": frozenset({"predicate_transition", "objective_event", "event_recovery_confirmed"}),
}


def _row(properties: dict, *, required: list[str] | None = None) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": properties,
            "required": list(properties) if required is None else required}


def _node(level: str, kind: str) -> dict:
    properties = {
        "node_id": {"type": "string", "minLength": 1},
        "kind": {"const": kind},
        "tick": {"type": "integer", "minimum": 0},
        "evidence_ref": {"type": "string", "minLength": 1},
    }
    if kind in {"event_recovery_confirmed", "event_response_confirmed"} and level in {"L1", "L4"}:
        properties["within_window"] = (
            {"type": "boolean"} if level == "L4" and kind == "event_recovery_confirmed"
            else {"const": True}
        )
    if kind == "local_business_observation":
        properties["evidence_ref"] = {"const": "objective/local_business_result.json"}
    else:
        file = ("predicate_transitions.jsonl" if kind == "predicate_transition" else
                "event_occurrences.jsonl" if kind == "objective_event" else
                "event_outcomes.jsonl")
        properties["evidence_ref"] = {"type": "string", "pattern": f"^objective/{file}#.+$"}
    return _row(properties)


def _l2_occurrence() -> dict:
    # l2_window.build_chain copies exactly these five fields from the formal
    # occurrence.  event_structure obtains the family, trigger and role keys
    # from the current executable event-rule registry.
    formal = formal_event_occurrence_schema()
    variants = formal["allOf"][0]["oneOf"]
    rows = []
    for variant in variants:
        source = variant["properties"]
        rows.append(_row({
            "node_id": {"type": "string", "minLength": 1},
            "family": deepcopy(source["event_family_id"]),
            "bindings": deepcopy(source["bindings"]),
            "trigger_tick": {"type": "integer", "minimum": 0},
            "trigger_predicate_id": deepcopy(source["trigger_predicate_id"]),
        }))
    return {"oneOf": rows}


def arm_chain_events_schema(level: str, effective_family: str) -> dict:
    """Return one closed chain/events row schema for a producer and row type."""
    if level == "L2":
        if effective_family != "arm_chain_events:l2_event_occurrence":
            raise ValueError(f"L2 chain/events has no row type {effective_family!r}")
        body = _l2_occurrence()
    elif level in _KINDS:
        kind = effective_family.removeprefix("arm_chain_events:")
        if effective_family != "arm_chain_events:" + kind or kind not in _KINDS[level]:
            raise ValueError(f"{level} chain/events has no row type {effective_family!r}")
        body = _node(level, kind)
    else:
        raise ValueError(f"{level!r} has no chain/events producer")
    return {"$schema": _DRAFT,
            "$id": f"https://aeroworld.local/schema/arm_chain_events_{level.lower()}_"
                   f"{effective_family.rsplit(':', 1)[-1]}.record.schema.json",
            "title": f"ARM {level} {effective_family} JSONL row", **body}
