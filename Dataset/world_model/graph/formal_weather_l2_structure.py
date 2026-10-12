"""Closed weather-row grammar for L2 ``formal_weather_meta`` declarations.

The current inventory declares three seeds each for L2-1_v1, L2-1_v2,
L2-2_v1, L2-2_v2, L2-3_v1, L2-3_v2, L2-4_v1, L2-4_v2, and L2-5_v1.
Each source is ``weather_meta.jsonl`` at the formal capture boundary; one
JSONL row is one record.  The converter's ``normalize_weather_row`` and
``expand_weather_rows`` build these rows, and the capture filter copies the
file unchanged.  L2 has no separate weather-row writer or key rule.

Reuse the reviewed L1 constructor for the complete producer contract:
required and optional fields, float source types, numeric ranges, the
``temperature_source``/``temperature_c`` dependency, and ticks 0..900.
The factory returns a fresh schema before this module assigns its L2 identity.
"""

from __future__ import annotations

from Dataset.world_model.graph.formal_weather_l1_structure import (
    formal_weather_meta_schema as _shared_weather_schema,
)

__all__ = ["formal_weather_meta_schema"]


def formal_weather_meta_schema() -> dict:
    """Return the closed schema for one L2 ``weather_meta.jsonl`` row."""
    schema = _shared_weather_schema()
    schema["$id"] = "https://aeroworld.local/schema/formal_weather_l2.record.schema.json"
    schema["title"] = "AERO_WORLD formal_weather_meta L2 weather row record schema"
    return schema
