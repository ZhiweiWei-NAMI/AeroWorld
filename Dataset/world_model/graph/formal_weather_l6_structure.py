"""Closed weather-row grammar for the L6 formal_weather_meta source family.

The L6 converter uses the same normalize_weather_row and expand_weather_rows
producer as L1.  Reuse its pure schema constructor so both levels keep the
same required, optional, range, and conditional-presence rules.
"""

from __future__ import annotations

from Dataset.world_model.graph.formal_weather_l1_structure import (
    formal_weather_meta_schema as _shared_weather_schema,
)

__all__ = ["formal_weather_meta_schema"]


def formal_weather_meta_schema() -> dict:
    """Return the closed schema for one L6 weather_meta.jsonl row."""
    schema = _shared_weather_schema()
    schema["$id"] = "https://aeroworld.local/schema/formal_weather_l6.record.schema.json"
    schema["title"] = "AERO_WORLD formal_weather_meta L6 per-tick weather row record schema"
    return schema
