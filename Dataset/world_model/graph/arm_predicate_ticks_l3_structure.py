"""Closed record structure for L3 ARM dense predicate tick rows.

Source family ``arm_predicate_truth_ticks``; profiles L3-1_v1, L3-2_v1,
L3-2_v2, L3-3_v1, L3-3_v2.  Producer: ``Dataset/tools/l3_v2/arm_runtime.py``
slices ``computed["predicate_truth_ticks"]`` to the declared window and
persists it verbatim to ``objective/predicate_truth_ticks.jsonl``
(``_run_arm``, ``write_rows(out / "objective/predicate_truth_ticks.jsonl",
dense)``).  The rows are built by ``Dataset/tools/l3_v2/semantics.py``
(``evaluate``, lines 325-328) as a fixed six-key dict literal per formal
predicate truth row:

    {"tick": r["tick"], "predicate_id": r["predicate_id"],
     "bindings": r["bindings"], "value": r["value"],
     "truth_id": r["truth_id"], "sampling": "authoritative_capture_grid"}

so every row carries all six keys and no other key can appear: the writer is
this one comprehension and no L3 code path adds fields afterwards.  ``tick``,
``predicate_id``, ``bindings``, ``value`` and ``truth_id`` are copied from the
objective pipeline's predicate truth rows (``result["predicate_truth"]``),
produced by ``Dataset/semantic_truth/minimal_semantics_adapter.py``:

- ``predicate_id``: the engine vocabulary.  ``_assert_core_vocabulary``
  (lines 1733-1740) raises when any emitted predicate leaves the executable
  registry set, so the closed enum is exactly
  ``Dataset/semantic_truth/core_semantic_registry.get_executable_predicate_ids()``.
- ``tick``: the engine tick copied from the predicate truth row; the L3
  episode clock covers ticks 0 through 900 in steps of 5.
- ``bindings``: the L3 engine emits grounded assertions using each predicate
  template's ``argument_roles``.  The adapter forms a ``grounded_roles__``
  tuple from those keys and ``_bindings`` writes the same keys with nonempty
  participant identifiers.  Each predicate ID therefore selects one exact
  closed role map; names for other adapter tuple types are not L3 roles.
- ``value``: ``Truth`` (``Dataset/semantic_truth/minimal_semantics.py`` lines
  23-27) written through ``item.truth.value``; all four members are reachable
  and present in the L3 corpus (``true``, ``false``, ``unknown``,
  ``out_of_scope``).
- ``truth_id``: ``stable_identifier("truth", ...)`` from
  ``Dataset/semantic_truth/provenance.py`` (lines 42-44) — the literal prefix
  ``truth:`` plus 24 lowercase hex characters.

Other-layer dense-row shapes are deliberately not accepted: the
``arm_window_semantics.py`` branch-window writer (``schema_name`` /
``schema_version`` / ``annotation_layer`` / ``context`` /
``missing_source_record`` / ``source_refs`` ...), the ``l1_window.py`` L0
projection (``operands`` / ``source_refs`` / ``measurements`` /
``occupied_aircraft_ids`` ...) and the L6 ``context``/``missing_inputs``/
``sampling_scope`` rows are different producers, and the L4 business branch
objects (``metrics.speeds_mps`` entity-keyed map, ``context.weather.*``,
``measurements.requester_ids``) have no L3 writer.  The L3 grammar describes
only the six-key producer above; cross-layer reuse and multi-branch dispatch
belong to G1/G2.  The six keys carry no integrity-keyed field, so the
publication projection (``structure_schema._PUBLICATION_EXCLUDED_PATHS``)
excludes nothing from these records and raw validation needs no carve-out.

This module exports the predicate-row schema constructor and performs no
source scan at import time.
"""

from __future__ import annotations

from typing import Any

from Dataset.semantic_truth.core_semantic_registry import (
    load_core_semantic_registry,
)

__all__ = ["arm_predicate_truth_ticks_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# Truth membership is closed by the producer enum
# (Dataset/semantic_truth/minimal_semantics.py, Truth, lines 23-27).
_TRUTH_VALUES = ("true", "false", "unknown", "out_of_scope")

# stable_identifier("truth", ...) writes "truth:" + 24 lowercase hex digits.
_TRUTH_ID_PATTERN = "^truth:[0-9a-f]{24}$"


def _bindings_schema(roles: tuple[str, ...]) -> dict[str, Any]:
    """One predicate's exact argument-role map to participant IDs."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(roles),
        "properties": {
            role: {"type": "string", "minLength": 1} for role in roles
        },
    }


def _row_schema(predicate_ids: list[str], roles: tuple[str, ...]) -> dict[str, Any]:
    """One full row branch for predicates with the same argument roles."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "bindings", "predicate_id", "sampling", "tick", "truth_id", "value",
        ],
        "properties": {
            "tick": {
                "type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5,
            },
            "predicate_id": {"enum": predicate_ids},
            "bindings": _bindings_schema(roles),
            "value": {"enum": list(_TRUTH_VALUES)},
            "truth_id": {
                "type": "string",
                "pattern": _TRUTH_ID_PATTERN,
            },
            "sampling": {"const": "authoritative_capture_grid"},
        },
    }


def arm_predicate_truth_ticks_schema() -> dict[str, Any]:
    """Return the closed schema for one L3 dense predicate tick row.

    Predicate IDs and their exact argument roles are read from the executable
    registry at call time; no state is cached or shared with other modules.
    """

    registry = load_core_semantic_registry()
    templates = {template["id"]: template for template in registry["templates"]}
    role_groups: dict[tuple[str, ...], list[str]] = {}
    for predicate_id in registry["implementation"]["executable_predicate_ids"]:
        roles = tuple(
            role["key"] for role in templates[predicate_id]["argument_roles"]
        )
        role_groups.setdefault(roles, []).append(predicate_id)

    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_predicate_truth_ticks_l3.record.schema.json"
        ),
        "title": "L3 ARM dense predicate tick record",
        "oneOf": [
            _row_schema(predicate_ids, roles)
            for roles, predicate_ids in role_groups.items()
        ],
    }
