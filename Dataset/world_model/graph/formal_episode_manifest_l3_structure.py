"""Closed whole-JSON grammar for formal L3 episode manifests.

All five L3 profiles use the manifest structure produced by batch_generate,
convert_to_render_ready, scene_occupancy, and the capture truth filter.  The
current converter writes no ``generation.source_contract_hash``; frozen L3
records carry it, so the raw field remains declared but optional.  The
optional root ``source_event_script_digest`` is retained by the shared
structure.  This constructor reads no source files.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_episode_manifest_l4_structure as _shared

__all__ = ["formal_episode_manifest_schema"]

_L3_PROFILES = (
    "L3-1_v1", "L3-2_v1", "L3-2_v2", "L3-3_v1", "L3-3_v2",
)


def formal_episode_manifest_schema() -> dict:
    """Return the closed whole-root schema for the five L3 profiles."""
    schema = _shared.formal_episode_manifest_schema()
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "formal_episode_manifest_l3.record.schema.json"
    )
    schema["title"] = "AERO_WORLD formal_episode_manifest L3 record schema"
    schema["properties"]["scenario_id"] = {"enum": list(_L3_PROFILES)}
    return schema
