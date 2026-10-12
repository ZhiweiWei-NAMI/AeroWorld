"""Closed weather-row grammar for X ``formal_weather_meta`` declarations.

The current inventory declares three seeds each for
X1_rain_to_c2loss_to_forced_landing,
X2_gnss_spoof_to_geofence_violation,
X3_pedestrian_fall_to_emergency_response,
X4_fog_to_uav_conflict,
X5_comm_failure_to_pad_contention, and
X6_crowd_evacuation_to_airspace_lockdown.  Each formal source is
``weather_meta.jsonl``; one JSONL row is one record.

The converter's ``normalize_weather_row`` and ``expand_weather_rows`` build
the rows for X as for L1, and the capture filter copies the file unchanged.
There is no X-specific weather key or range rule.  Reuse the reviewed L1
constructor for required and optional fields, float source types, ranges,
the ``temperature_source``/``temperature_c`` dependency, and ticks 0..900.
The sampled X rows contain ``dust`` and ``hazard_source_active``; they do not
establish an instance of the declared ``temperature_source`` branch.
The factory returns a fresh schema before this module assigns its X identity.
"""

from __future__ import annotations

from Dataset.world_model.graph.formal_weather_l1_structure import (
    formal_weather_meta_schema as _shared_weather_schema,
)

__all__ = ["formal_weather_meta_schema"]


def formal_weather_meta_schema() -> dict:
    """Return the closed schema for one X ``weather_meta.jsonl`` row."""
    schema = _shared_weather_schema()
    schema["$id"] = "https://aeroworld.local/schema/formal_weather_x.record.schema.json"
    schema["title"] = "AERO_WORLD formal_weather_meta X weather row record schema"
    return schema
