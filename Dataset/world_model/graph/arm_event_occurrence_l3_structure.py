"""Closed whole-row grammar for L3 ARM event-occurrence records.

Every bound L3 entrypoint runs through ``l3_ue_delivery`` and
``l3_v2.arm_runtime``.  The runtime selects lifecycle rows from the objective
pipeline's computed occurrences and writes those rows unchanged.  Their
participant bindings, phase sequences, evidence arrays, event-rule identity,
and fixed metadata therefore come from
``event_structure.formal_event_occurrence_schema`` and its executable
registry authorities.

Current production writes the root ``rule_digest``.  Stored L3 3.0.0 rows
also have a closed pre-digest envelope, so the two key sets are represented as
disjoint branches instead of making the integrity field loosely optional.
Raw validation retains and checks the digest when present; publication
projection separately excludes the exact root path ``("rule_digest",)``.

``arm_window_semantics`` has a smaller local occurrence format, but no current
L3 entrypoint calls that writer.  It is not an accepted fallback here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.world_model.graph.event_structure import formal_event_occurrence_schema


__all__ = ["arm_event_occurrences_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_EPISODE_ID_PATTERN = r"^(?:L3-1_v1|L3-2_v[12]|L3-3_v[12])__seed(?:00|01|02)$"
_RULE_DIGEST = "rule_digest"


def _formal_envelope(
    formal: dict[str, Any], *, with_rule_digest: bool
) -> dict[str, Any]:
    """Detach one formal envelope and apply the L3 source identity boundary."""

    schema = deepcopy(formal)
    for annotation in ("$schema", "$id", "title"):
        schema.pop(annotation, None)

    properties = schema.get("properties")
    required = schema.get("required")
    if type(properties) is not dict or type(required) is not list:
        raise ValueError("formal event-occurrence schema has no closed root envelope")
    if _RULE_DIGEST not in properties or _RULE_DIGEST not in required:
        raise ValueError("formal event-occurrence digest contract has changed")

    properties["episode_id"] = {
        "type": "string",
        "pattern": _EPISODE_ID_PATTERN,
    }
    if not with_rule_digest:
        properties.pop(_RULE_DIGEST)
        required.remove(_RULE_DIGEST)
    return schema


def arm_event_occurrences_schema() -> dict[str, Any]:
    """Return the closed grammar for one L3 event-occurrence JSONL row."""

    formal = formal_event_occurrence_schema()
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_event_occurrence.l3.record.schema.json",
        "title": "L3 ARM event occurrence record",
        "description": (
            "One unchanged objective/event_occurrences.jsonl row from a bound "
            "L3 ARM branch."
        ),
        "oneOf": [
            _formal_envelope(formal, with_rule_digest=False),
            _formal_envelope(formal, with_rule_digest=True),
        ],
    }
