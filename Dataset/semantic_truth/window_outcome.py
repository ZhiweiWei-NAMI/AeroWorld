"""Validate the outcome carried by one bounded ARM window."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


@lru_cache(maxsize=2)
def _validator(schema_name: str) -> Draft202012Validator:
    if schema_name not in {"arm_window_event_outcome", "l4_business_event_outcome"}:
        raise ValueError(f"unknown ARM outcome schema: {schema_name}")
    path = Path(__file__).resolve().parents[1] / f"semantic_rules/schema/{schema_name}.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def arm_window_outcome(row: dict[str, Any], *, start_tick: int, end_tick: int) -> dict[str, Any]:
    """Give a standard objective outcome its window contract and check its bounds."""
    if row.get("schema_name") != "event_outcome" or row.get("schema_version") != "3.0.0":
        raise ValueError(f"ARM outcome has a different source schema: {row.get('outcome_id')}")
    result = {**row, "schema_name": "arm_window_event_outcome", "schema_version": "1.0.0"}
    validate_arm_window_outcome(result, start_tick=start_tick, end_tick=end_tick)
    return result


def validate_arm_window_outcome(row: dict[str, Any], *, start_tick: int, end_tick: int) -> None:
    """Check a saved standard ARM outcome against its schema and window."""
    if (type(start_tick) is not int or type(end_tick) is not int
            or start_tick < 0 or start_tick > end_tick):
        raise ValueError("ARM outcome window bounds are invalid")
    status = row.get("status")
    if status == "succeeded" and (type(row.get("terminal_tick")) is not int
                                  or not start_tick <= row["terminal_tick"] <= end_tick):
        raise ValueError(f"ARM terminal is outside its window: {row.get('outcome_id')}")
    if status == "right_censored" and row.get("window_censor_tick") != end_tick:
        raise ValueError(f"ARM censor boundary differs from window: {row.get('outcome_id')}")
    errors = sorted(_validator("arm_window_event_outcome").iter_errors(row),
                    key=lambda item: tuple(str(x) for x in item.path))
    if errors:
        raise ValueError(f"ARM outcome violates its schema: {row.get('outcome_id')}: {errors[0].message}")


def validate_l4_business_outcome(row: dict[str, Any], *, start_tick: int, end_tick: int) -> None:
    """Keep local response outcomes separate from ontology event outcomes."""
    if (type(start_tick) is not int or type(end_tick) is not int
            or start_tick < 0 or start_tick > end_tick):
        raise ValueError("L4 business outcome window bounds are invalid")
    tick = row.get("terminal_tick")
    if tick is not None and (type(tick) is not int or not start_tick <= tick <= end_tick):
        raise ValueError(f"L4 business terminal is outside its window: {row.get('outcome_id')}")
    errors = sorted(_validator("l4_business_event_outcome").iter_errors(row),
                    key=lambda item: tuple(str(x) for x in item.path))
    if errors:
        raise ValueError(f"L4 business outcome violates its schema: {row.get('outcome_id')}: {errors[0].message}")
