"""Closed weather-row grammar for L3 ``formal_weather_meta`` declarations.

The current inventory declares three seeds each for L3-1_v1, L3-2_v1,
L3-2_v2, L3-3_v1, and L3-3_v2.  Each formal source is
``weather_meta.jsonl``; one JSONL row is one record.  The same converter
``normalize_weather_row`` and ``expand_weather_rows`` used for L1 build L3
rows, and the capture filter copies the file unchanged.  L3 has no separate
weather-row key, type, or range rule.

Reuse the reviewed L1 constructor for required and optional fields, float
source types, ranges, the ``temperature_source``/``temperature_c`` rule,
and ticks 0..900.  ``temperature_source`` is a producer-declared branch;
representative L3 rows do not establish an instance of it.  The factory
returns a fresh schema before this module assigns its L3 identity.
"""

from __future__ import annotations

from Dataset.world_model.graph.formal_weather_l1_structure import (
    formal_weather_meta_schema as _shared_weather_schema,
)

__all__ = ["formal_weather_meta_schema"]


def formal_weather_meta_schema() -> dict:
    """Return the closed schema for one L3 ``weather_meta.jsonl`` row."""
    schema = _shared_weather_schema()
    schema["$id"] = "https://aeroworld.local/schema/formal_weather_l3.record.schema.json"
    schema["title"] = "AERO_WORLD formal_weather_meta L3 weather row record schema"
    return schema
