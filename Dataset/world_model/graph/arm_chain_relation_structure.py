"""Closed JSONL row schemas for the seven ARM chain/relations producers."""

from __future__ import annotations


__all__ = ["arm_chain_relations_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_LEVELS = frozenset({"L1", "L2", "L3", "L4", "L5", "L6", "X"})
_EVIDENCE_TYPES = (
    "predicate_onset_evidence", "predicate_escalation_support_evidence",
    "predicate_terminal_evidence",
)


def _row(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def _str() -> dict:
    return {"type": "string", "minLength": 1}


def _formal() -> dict:
    base = {"source": _str(), "target": _str(), "tick": {"type": "integer", "minimum": 0},
            "causal_claim": {"const": False}}
    return {"oneOf": [
        _row({**base, "relation_type": {"const": kind}, "rule_id": _str()})
        for kind in _EVIDENCE_TYPES
    ] + [_row({**base, "relation_type": {"const": "has_observed_outcome"}})]}


def _l2() -> dict:
    base = {"from": _str(), "to": _str()}
    return {"oneOf": [
        _row({**base, "relation": {"const": "predicate_supports_event"},
              "evidence": {"const": "current expression recomputation"}}),
    ] + [
        _row({**base, "relation": {"const": "observed_outcome"},
              "status": {"const": status},
              "terminal_tick": ({"type": "integer", "minimum": 0}
                                if status == "succeeded" else {"type": "null"})})
        for status in ("pending", "succeeded", "right_censored")
    ]}


def _x() -> dict:
    base = {"source": _str(), "target": _str(), "causal_claim": {"const": False}}
    return {"oneOf": [
        _row({**base, "type": {"const": "script_dependency"},
              "source_ref": _str()}),
        _row({**base, "type": {"const": "observed_dispatch"}}),
        _row({**base, "type": {"const": "geometry_trigger_support"}}),
    ]}


def arm_chain_relations_schema(level: str) -> dict:
    """Return the closed relation row schema for one ARM producer level."""
    if level not in _LEVELS:
        raise ValueError(f"unsupported ARM chain relation level: {level!r}")
    body = _l2() if level == "L2" else _x() if level == "X" else _formal()
    return {"$schema": _DRAFT,
            "$id": f"https://aeroworld.local/schema/arm_chain_relations_{level.lower()}.record.schema.json",
            "title": f"ARM {level} chain relation JSONL row", **body}
