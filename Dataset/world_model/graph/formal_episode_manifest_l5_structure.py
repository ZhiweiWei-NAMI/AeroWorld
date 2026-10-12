"""Closed whole-JSON grammar for formal L5 episode manifests.

The nine L5 profiles use the manifest root built by batch_generate,
convert_to_render_ready, scene_occupancy, and the capture truth filter.
This layer adds planned ``weather_car_*`` SUMO vehicle ids to the shared
vehicle-key maps.  The converter emits a three-key required-vehicle
lifecycle block when no capture-required SUMO vehicle is selected, or a
five-key block with source and truth lifecycle maps when one is selected.
The maps' members follow the payloads of
``source_sumo_vehicle_lifecycle_by_vehicle_id`` and
``truth_sumo_vehicle_lifecycle_by_vehicle_id``.  Construction reads no data.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_episode_manifest_l4_structure as _shared

__all__ = ["formal_episode_manifest_schema"]

_L5_PROFILES = (
    "L5-1_v1", "L5-1_v2", "L5-1_v3", "L5-2_v1", "L5-2_v2",
    "L5-3_v1", "L5-3_v2", "L5-4_v1", "L5-5_v1",
)
_L5_VEHICLE_ID_PATTERN = (
    "^(?:bg_vehicle_[A-Za-z0-9_]+__seed[0-9]{2}|"
    "traffic_bg_[A-Za-z0-9_]+|"
    "weather_car_l5_(?:1_v[123]|2_v[12]|3_v[12]|4_v1|5_v1)"
    "__seed[0-9]{2})$"
)
_VEHICLE_SELECTION_MAPS = (
    "entity_ids", "min_distance_m_by_vehicle_id",
    "frames_seen_by_vehicle_id", "motion_span_m_by_vehicle_id",
    "max_speed_mps_by_vehicle_id",
)


def _extend_vehicle_keys(map_schema: dict) -> None:
    patterns = map_schema["patternProperties"]
    member = patterns.pop(_shared._PATTERN_VEHICLE_ID)
    patterns[_L5_VEHICLE_ID_PATTERN] = member


def formal_episode_manifest_schema() -> dict:
    """Return the closed whole-root schema for the nine L5 profiles."""
    schema = _shared.formal_episode_manifest_schema()
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "formal_episode_manifest_l5.record.schema.json"
    )
    schema["title"] = "AERO_WORLD formal_episode_manifest L5 record schema"
    schema["properties"]["scenario_id"] = {"enum": list(_L5_PROFILES)}

    sumo = schema["properties"]["sumo_traffic"]["properties"]
    selection = sumo["selection"]["properties"]
    for name in _VEHICLE_SELECTION_MAPS:
        _extend_vehicle_keys(selection[name])
    _extend_vehicle_keys(sumo["canonical_asset_records"])

    full_lifecycle = sumo["required_vehicle_lifecycle_policy"]["oneOf"][1]
    for name in ("source_lifecycle_by_vehicle_id", "truth_lifecycle_by_vehicle_id"):
        _extend_vehicle_keys(full_lifecycle["properties"][name])
    return schema
