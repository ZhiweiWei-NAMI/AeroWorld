"""Closed whole-record grammar for L1 ARM event-outcome JSONL rows.

All 84 declared L1 ARM entry scripts call
``Dataset.tools.l1_window.execute_config``.  The current writer obtains formal
outcomes from ``minimal_semantics_adapter._adapt_outcomes``, crops evidence to
the ARM window, and converts the envelope with ``arm_window_outcome``.  Its
window envelope adds the ``right_censored`` status and ``window_censor_tick``.

The bounded frozen L1 records retain the formal ``event_outcome`` 3.0.0
envelope and add the paired boolean fields ``terminal_within_window`` and
``censored_by_window``.  Frozen ``succeeded`` rows carry ``true``/``false``;
frozen ``pending`` rows carry ``false``/``true``.  The envelope selector below
keeps those recorded shapes separate from the current
``arm_window_event_outcome`` 1.0.0 writer shape.

L1 selects four registry-backed event families.  Their terminal predicates,
binding roles, lifecycle status, and evidence layout stay owned by the
detached formal outcome schema.  This module only selects those family
branches and closes the two L1 envelope shapes.
"""

from __future__ import annotations

from typing import Any

from Dataset.world_model.graph.event_structure import formal_event_outcome_schema


_L1_EVENT_FAMILIES = (
    "restricted_region_intrusion",
    "altitude_excursion",
    "uav_separation_conflict",
    "corridor_congestion",
)
_FROZEN_WINDOW_FIELDS = (
    "terminal_within_window",
    "censored_by_window",
)


def _formal_family_branches(
    schema: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index only the formal outcome alternatives selected by L1."""

    all_of = schema.get("allOf")
    if type(all_of) is not list or len(all_of) != 1:
        raise ValueError("formal event outcome schema has an unexpected allOf layout")
    selector = all_of[0]
    if type(selector) is not dict or type(selector.get("oneOf")) is not list:
        raise ValueError("formal event outcome schema lacks its family selector")

    selected: dict[str, dict[str, Any]] = {}
    for branch in selector["oneOf"]:
        if type(branch) is not dict or type(branch.get("properties")) is not dict:
            raise ValueError("formal event outcome family branch lacks properties")
        family_property = branch["properties"].get("event_family_id")
        if type(family_property) is not dict:
            raise ValueError("formal event outcome family branch lacks event_family_id")
        family = family_property.get("const")
        if family not in _L1_EVENT_FAMILIES:
            continue
        if family in selected:
            raise ValueError(f"formal L1 event outcome family repeats: {family}")
        selected[family] = branch
    return selected


def _add_right_censored_branch(branch: dict[str, Any]) -> None:
    """Add the window-only terminal branch to one detached formal family."""

    status_branches = branch.get("oneOf")
    if type(status_branches) is not list:
        raise ValueError("formal event outcome family lacks status branches")
    statuses: list[str] = []
    for status_branch in status_branches:
        if type(status_branch) is not dict:
            raise ValueError("formal event outcome status branch is not an object")
        properties = status_branch.get("properties")
        if type(properties) is not dict:
            raise ValueError("formal event outcome status branch lacks properties")
        status_property = properties.get("status")
        if type(status_property) is not dict or type(status_property.get("const")) is not str:
            raise ValueError("formal event outcome status branch lacks a status const")
        statuses.append(status_property["const"])
    if statuses != ["pending", "succeeded"]:
        raise ValueError(f"formal event outcome statuses changed: {statuses}")

    status_branches.append(
        {
            "properties": {
                "status": {"const": "right_censored"},
                "terminal_tick": {"const": None},
                "lifecycle_status": {"const": "observed_until_window_end"},
                "lifecycle_phase_evidence": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 0,
                    "items": False,
                },
            },
            "required": [
                "status",
                "terminal_tick",
                "lifecycle_status",
                "lifecycle_phase_evidence",
                "window_censor_tick",
            ],
        }
    )


def arm_event_outcomes_schema() -> dict[str, Any]:
    """Return the closed Draft 2020-12 schema for one L1 ARM outcome row."""

    schema = formal_event_outcome_schema()
    properties = schema.get("properties")
    required = schema.get("required")
    if (
        type(properties) is not dict
        or type(required) is not list
        or len(required) != len(set(required))
        or set(required) != set(properties)
        or schema.get("additionalProperties") is not False
    ):
        raise ValueError("formal event outcome root is no longer fully required and closed")

    by_family = _formal_family_branches(schema)
    missing = [family for family in _L1_EVENT_FAMILIES if family not in by_family]
    if missing:
        raise ValueError(f"L1 event outcome families are absent from authority: {missing}")
    selected = [by_family[family] for family in _L1_EVENT_FAMILIES]
    schema["allOf"][0]["oneOf"] = selected
    for branch in selected:
        _add_right_censored_branch(branch)

    base_property_count = len(required)
    properties["schema_name"] = {
        "enum": ["event_outcome", "arm_window_event_outcome"]
    }
    properties["schema_version"] = {"enum": ["3.0.0", "1.0.0"]}
    properties["status"] = {
        "enum": ["pending", "succeeded", "right_censored"]
    }
    properties["lifecycle_status"] = {
        "enum": [
            "terminal_predicate_hold",
            "recovery_predicate_hold",
            "terminal_transition_not_observed",
            "terminal_hold_not_observed",
            "observed_until_window_end",
        ]
    }
    properties["window_censor_tick"] = {"type": "integer", "minimum": 0}
    properties["terminal_within_window"] = {"type": "boolean"}
    properties["censored_by_window"] = {"type": "boolean"}

    schema["allOf"].append(
        {
            "oneOf": [
                {
                    "properties": {
                        "schema_name": {"const": "event_outcome"},
                        "schema_version": {"const": "3.0.0"},
                        "status": {"const": "pending"},
                        "terminal_within_window": {"const": False},
                        "censored_by_window": {"const": True},
                    },
                    "required": list(_FROZEN_WINDOW_FIELDS),
                    "minProperties": base_property_count + 2,
                    "maxProperties": base_property_count + 2,
                },
                {
                    "properties": {
                        "schema_name": {"const": "event_outcome"},
                        "schema_version": {"const": "3.0.0"},
                        "status": {"const": "succeeded"},
                        "terminal_within_window": {"const": True},
                        "censored_by_window": {"const": False},
                    },
                    "required": list(_FROZEN_WINDOW_FIELDS),
                    "minProperties": base_property_count + 2,
                    "maxProperties": base_property_count + 2,
                },
                {
                    "properties": {
                        "schema_name": {"const": "arm_window_event_outcome"},
                        "schema_version": {"const": "1.0.0"},
                        "status": {"enum": ["pending", "succeeded"]},
                    },
                    "minProperties": base_property_count,
                    "maxProperties": base_property_count,
                },
                {
                    "properties": {
                        "schema_name": {"const": "arm_window_event_outcome"},
                        "schema_version": {"const": "1.0.0"},
                        "status": {"const": "right_censored"},
                    },
                    "required": ["window_censor_tick"],
                    "minProperties": base_property_count + 1,
                    "maxProperties": base_property_count + 1,
                },
            ]
        }
    )
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "arm_event_outcome_l1.record.schema.json"
    )
    schema["title"] = "L1 ARM event outcome record"
    schema["description"] = (
        "Closed whole-record grammar for one L1 objective/event_outcomes.jsonl "
        "row. Four formal event-family branches are combined with the frozen "
        "3.0.0 envelope and the current 1.0.0 ARM-window envelope."
    )
    return schema


__all__ = ["arm_event_outcomes_schema"]
