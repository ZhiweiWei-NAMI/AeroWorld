"""Closed whole-JSON grammar for formal L2 episode manifests.

The capture-filtered manifests for L2-1 through L2-4 retain the producer
structure shared with L4.  The current converter writes no
``generation.source_contract_hash``, while frozen records may carry it, so
that integrity field is optional in the ordinary branch.  The root
``source_event_script_digest`` is optional in the shared structure.

``repair_l25_signal_truth.py:_stage_episode`` rewrites all three L2-5_v1
formal manifests: it replaces both generator fields and the entire
``generation`` object, and sets ``validation_summary.ok`` to null.  That
producer branch has its own closed root alternative.  Constructors perform
no source reads or compilation at import.
"""

from __future__ import annotations

from Dataset.world_model.graph import formal_episode_manifest_l4_structure as _shared

__all__ = ["formal_episode_manifest_schema"]

_ORDINARY_PROFILES = (
    "L2-1_v1", "L2-1_v2", "L2-2_v1", "L2-2_v2",
    "L2-3_v1", "L2-3_v2", "L2-4_v1", "L2-4_v2",
)
_REPAIRED_PROFILE = "L2-5_v1"
_REPAIR_GENERATOR = "Dataset/tools/repair_l25_signal_truth.py"


def formal_episode_manifest_schema() -> dict:
    """Return the closed whole-root schema for the nine L2 profiles."""
    ordinary = _shared.formal_episode_manifest_schema()
    ordinary["properties"]["scenario_id"] = {"enum": list(_ORDINARY_PROFILES)}
    repaired = _shared.formal_episode_manifest_schema()
    repaired["properties"]["scenario_id"] = {"const": _REPAIRED_PROFILE}
    repaired["properties"]["generator"] = {"const": _REPAIR_GENERATOR}
    repaired["properties"]["generation"] = {
        "type": "object",
        "properties": {
            "generator": {"const": _REPAIR_GENERATOR},
            "spatial_basis": {
                "const": "Retained formal truth_frames and trajectories for this episode"
            },
            "state_basis": {
                "const": "Dataset/scenarios/L2_infrastructure/interaction/"
                "L2-5_v1/event_script.json"
            },
            "scene_basis": {
                "const": "Dataset/scenarios/L2_infrastructure/interaction/"
                "L2-5_v1/scene_setup.json"
            },
        },
        "required": ["generator", "spatial_basis", "state_basis", "scene_basis"],
        "additionalProperties": False,
    }
    repaired["properties"]["validation_summary"]["properties"]["ok"] = {
        "type": "null"
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/"
        "formal_episode_manifest_l2.record.schema.json",
        "title": "AERO_WORLD formal_episode_manifest L2 record schema",
        "oneOf": [ordinary, repaired],
    }
