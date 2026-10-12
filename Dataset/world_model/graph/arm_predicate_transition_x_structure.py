"""Closed whole-record grammar for X ARM predicate transition rows.

``Dataset/tools/x_arm_pipeline.py::calculate`` groups dense predicate rows
on the five-tick grid and writes one eight-key transition when adjacent
values differ.  ``run_arm`` writes that list unchanged to
``objective/predicate_transitions.jsonl``.  The declared source family and
effective record type are both ``arm_predicate_transitions``; this X writer
does not call the formal transition adapter used by L1-L6.

The transition copies ``predicate_id`` and ``bindings`` from the later dense
row and uses its ``source_refs``.  Their ten mutually exclusive producer
branches are projected from the X dense-tick grammar, which describes the
same ``calculate`` emitters.  Values are the writer's strings ``true``,
``false`` and ``unknown``; all six unequal pairs are producer-reachable by
the comparison, although the inspected factual X rows contain only the two
boolean flips.  ``from_tick`` and ``to_tick`` are on the formal grid in
0..900.  Their ordering and adjacency, and the digest-based identifier's
equality to the row content, remain producer invariants across properties.

There are no integrity metadata fields or publication-exclusion paths in
this record.  Registry dispatch belongs to the integration stage.  Schema
construction has no data access or import-time scans.
"""

from __future__ import annotations

import copy
from typing import Any

from Dataset.world_model.graph.arm_predicate_ticks_x_structure import (
    arm_predicate_truth_ticks_schema,
)

__all__ = ["arm_predicate_transitions_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_VALUES = ("true", "false", "unknown")
_COPIED_FIELDS = ("predicate_id", "bindings", "source_refs")


def _predicate_variants() -> list[dict[str, Any]]:
    """Keep the later dense row's predicate, binding and ref branch."""
    dense_schema = arm_predicate_truth_ticks_schema("X")
    return [
        {
            "properties": {
                name: copy.deepcopy(branch["properties"][name])
                for name in _COPIED_FIELDS
            },
            "required": list(_COPIED_FIELDS),
        }
        for branch in dense_schema["oneOf"]
    ]


def _value_variants() -> list[dict[str, Any]]:
    return [
        {
            "properties": {
                "before": {"const": before},
                "after": {"const": after},
            },
            "required": ["before", "after"],
        }
        for before in _VALUES
        for after in _VALUES
        if before != after
    ]


def arm_predicate_transitions_schema() -> dict[str, Any]:
    """Return the closed Draft 2020-12 schema for one X transition row."""
    properties: dict[str, Any] = {
        "transition_id": {
            "type": "string", "pattern": "^xt:[0-9a-f]{64}$",
        },
        "predicate_id": {"type": "string", "minLength": 1},
        # The selected dense-row branch closes the binding keys.
        "bindings": {"type": "object"},
        "from_tick": {
            "type": "integer", "minimum": 0, "maximum": 895,
            "multipleOf": 5,
        },
        "to_tick": {
            "type": "integer", "minimum": 5, "maximum": 900,
            "multipleOf": 5,
        },
        "before": {"enum": list(_VALUES)},
        "after": {"enum": list(_VALUES)},
        # The selected dense-row branch supplies the ref format.
        "source_refs": {
            "type": "array", "minItems": 1,
            "items": {"type": "string"}, "uniqueItems": True,
        },
    }
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/"
               "arm_predicate_transitions_x.record.schema.json",
        "title": "X ARM predicate transition record",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
        "allOf": [
            {"oneOf": _predicate_variants()},
            {"oneOf": _value_variants()},
        ],
    }
