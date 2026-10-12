"""Closed record grammar for formal event-chain relation JSONL rows.

This module is the shared schema factory for source family
``formal_event_chain_relations`` at levels L1, L2, L3, L4, L5, L6, and X.
``Dataset.tools.pipeline.build_event_chains`` writes each mapping returned by
``Dataset.semantic_truth.event_chain.build_event_chain`` as one JSONL row.  Its
producer-owned Draft 2020-12 contract is
``Dataset/semantic_rules/schema/event_chain_relation.schema.json``.

The factory reads that native contract on each call, so a producer-contract
change cannot leave a silent embedded copy behind.  It rejects an unexpected
root or source-endpoint layout, then works on the newly decoded mapping.  The
native empty source-endpoint branch is already closed by ``maxProperties: 0``;
the local structure compiler additionally requires
``additionalProperties: false`` on every object branch, so the factory adds
that equivalent declaration before compilation.  No source data or schema is
read at module import.

The required ``record_digest`` and the three required digests under
``provenance`` remain in the raw grammar.  Publication projection separately
omits only the exact paths declared by ``structure_schema``; this factory does
not remove or weaken those fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


__all__ = ["formal_event_chain_relations_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_NATIVE_ID = "https://aeroworld.local/schema/event_chain_relation.schema.json"
_NATIVE_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "semantic_rules"
    / "schema"
    / "event_chain_relation.schema.json"
)
_LEVEL_PROFILES = {
    "L1": (
        "L1-1_v1", "L1-1_v2", "L1-2_v1", "L1-3_v1", "L1-3_v2",
        "L1-4_v1", "L1-4_v2",
    ),
    "L2": (
        "L2-1_v1", "L2-1_v2", "L2-2_v1", "L2-2_v2", "L2-3_v1",
        "L2-3_v2", "L2-4_v1", "L2-4_v2", "L2-5_v1",
    ),
    "L3": (
        "L3-1_v1", "L3-2_v1", "L3-2_v2", "L3-3_v1", "L3-3_v2",
    ),
    "L4": (
        "L4-10_v1", "L4-10_v2", "L4-11_v1", "L4-11_v2", "L4-1_v1",
        "L4-1_v2", "L4-2_v1", "L4-2_v2", "L4-3_v1", "L4-3_v2",
        "L4-3_v3", "L4-4_v1", "L4-4_v2", "L4-5_v1", "L4-5_v2",
        "L4-5_v3", "L4-6_v1", "L4-6_v2", "L4-7_v1", "L4-7_v2",
        "L4-8_v1", "L4-8_v2", "L4-9_v1", "L4-9_v2",
    ),
    "L5": (
        "L5-1_v1", "L5-1_v2", "L5-1_v3", "L5-2_v1", "L5-2_v2",
        "L5-3_v1", "L5-3_v2", "L5-4_v1", "L5-5_v1",
    ),
    "L6": (
        "L6-1_v1", "L6-1_v2", "L6-2_v1", "L6-2_v2", "L6-3_v1",
        "L6-3_v2", "L6-4_v1", "L6-4_v2", "L6-5_v1", "L6-5_v2",
    ),
    "X": (
        "X1_rain_to_c2loss_to_forced_landing",
        "X2_gnss_spoof_to_geofence_violation",
        "X3_pedestrian_fall_to_emergency_response",
        "X4_fog_to_uav_conflict",
        "X5_comm_failure_to_pad_contention",
        "X6_crowd_evacuation_to_airspace_lockdown",
    ),
}


def _current_producer_schema() -> dict[str, Any]:
    """Read and check the current producer-owned relation schema."""

    try:
        schema = json.loads(_NATIVE_SCHEMA.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read producer schema {_NATIVE_SCHEMA}: {exc}") from exc
    if (
        type(schema) is not dict
        or schema.get("$schema") != _DRAFT
        or schema.get("$id") != _NATIVE_ID
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        raise ValueError(f"producer event-chain relation schema root drifted: {_NATIVE_SCHEMA}")
    try:
        properties = schema["properties"]
        source_variants = schema["$defs"]["sourceEndpoint"]["oneOf"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"producer event-chain relation schema layout drifted: {_NATIVE_SCHEMA}"
        ) from exc
    if (
        type(properties) is not dict
        or type(source_variants) is not list
        or not source_variants
        or source_variants[0] != {"type": "object", "maxProperties": 0}
    ):
        raise ValueError(
            f"producer event-chain endpoint schema layout drifted: {_NATIVE_SCHEMA}"
        )
    source_variants[0]["additionalProperties"] = False
    return schema


def formal_event_chain_relations_schema(level: str) -> dict[str, Any]:
    """Return the current closed relation-row schema for one formal level."""

    try:
        profiles = _LEVEL_PROFILES[level]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported formal event-chain level: {level!r}") from exc

    schema = _current_producer_schema()
    schema["$id"] = (
        "https://aeroworld.local/schema/"
        f"formal_event_chain_relations_{level.lower()}.record.schema.json"
    )
    schema["title"] = f"AERO_WORLD formal_event_chain_relations {level} row schema"
    schema["properties"]["epi_id"] = {"enum": list(profiles)}
    schema["properties"]["episode_id"] = {
        "enum": [
            f"{profile}__seed{seed:02d}"
            for profile in profiles
            for seed in range(3)
        ]
    }
    return schema
