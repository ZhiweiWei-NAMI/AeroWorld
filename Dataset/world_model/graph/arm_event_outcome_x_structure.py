"""Closed whole-record grammar for X ARM event-outcome JSONL rows.

The source and effective family are both ``arm_event_outcomes``.  The six X
profiles use the same record roots because an outcome carries only the stable
event identity produced for its measured predicate onset.  ``run_arm`` writes
``calculate``'s outcome list directly to ``objective/event_outcomes.jsonl``;
the formal event adapter and window-outcome envelopes are not part of this
family.

``calculate`` emits two closed roots.  A recovered outcome has exactly
``event_id``, ``terminal_tick``, ``status`` and ``hold_samples``.  Its terminal
tick is on the five-tick formal grid, and ``hold_samples`` is exactly three
because the producer appends the row as soon as the configured recovery hold
is reached.  A right-censored outcome has exactly ``event_id``,
``terminal_tick`` and ``status``; its terminal tick is null and it has no hold
count.

These rows contain no integrity metadata.  ``event_id`` stays in the raw and
published record, so no publication-exclusion path is needed.  This module
performs no I/O and does not register the grammar globally.
"""

from __future__ import annotations

from typing import Any

__all__ = ["arm_event_outcomes_schema"]


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_EVENT_ID_PATTERN = r"^xb:[0-9a-f]{64}$"
_RECOVERY_HOLD_SAMPLES = 3


def _event_id() -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": _EVENT_ID_PATTERN,
        "x-source-value-types": ["string"],
    }


def _terminal_tick() -> dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 0,
        "maximum": 900,
        "multipleOf": 5,
        "x-source-value-types": ["int"],
    }


def _recovered_branch() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["event_id", "terminal_tick", "status", "hold_samples"],
        "properties": {
            "event_id": _event_id(),
            "terminal_tick": _terminal_tick(),
            "status": {"const": "recovered"},
            "hold_samples": {"const": _RECOVERY_HOLD_SAMPLES},
        },
    }


def _right_censored_branch() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["event_id", "terminal_tick", "status"],
        "properties": {
            "event_id": _event_id(),
            "terminal_tick": {"const": None},
            "status": {"const": "right_censored"},
        },
    }


def arm_event_outcomes_schema() -> dict[str, Any]:
    """Return the closed grammar for one X ARM event-outcome JSONL row."""

    return {
        "$schema": _DRAFT,
        "$id": (
            "https://aeroworld.local/schema/"
            "arm_event_outcome_x.record.schema.json"
        ),
        "title": "X ARM measured event outcome record",
        "oneOf": [_recovered_branch(), _right_censored_branch()],
    }
