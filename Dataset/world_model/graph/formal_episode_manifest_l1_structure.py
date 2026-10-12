"""Closed whole-JSON grammar for L1 formal episode manifests.

L1 uses the same manifest producers and nested structures as L4.  Frozen
L1 roots differ in profile identity and in whether generation carries
``source_contract_hash``: the L1-1/L1-2 records checked here omit it,
while L1-3/L1-4 records carry it.  The root
``source_event_script_digest`` is likewise retained when present.  Both
integrity paths remain in the raw schema and are already excluded by the
shared publication projection.  Constructors read no files.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_episode_manifest_l4_structure as _shared

__all__ = ["formal_episode_manifest_schema"]

_L1_PROFILES = (
    "L1-1_v1", "L1-1_v2", "L1-2_v1", "L1-3_v1", "L1-3_v2",
    "L1-4_v1", "L1-4_v2",
)


def formal_episode_manifest_schema() -> dict:
    """Return the closed whole-JSON L1 manifest schema."""
    schema = _shared.formal_episode_manifest_schema()
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        "formal_episode_manifest_l1.record.schema.json"
    )
    schema["title"] = "AERO_WORLD formal_episode_manifest L1 record schema"
    schema["properties"]["scenario_id"] = {"enum": list(_L1_PROFILES)}
    return schema
