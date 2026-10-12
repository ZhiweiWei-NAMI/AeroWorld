"""Closed occurrence row schema for the L5 ARM semantic producer.

``l5_arm_semantics.calculate`` passes only ``COMPUTED_FAMILIES`` rules to
``_adapt_occurrences``. Its pedestrian rule is passed only when the branch
configuration has semantic pedestrian IDs. The shared formal schema supplies
the complete row and phase structure; this module narrows its rule variants
to the producer's reachable set without changing that shared schema.
"""

from __future__ import annotations

from typing import Any

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
)
from Dataset.tools.l5_arm_semantics import COMPUTED_FAMILIES
from Dataset.world_model.graph.event_structure import (
    formal_event_occurrence_schema,
)


_PEDESTRIAN_FAMILY = "uav_pedestrian_near_miss"


def arm_event_occurrences_schema(*, has_semantic_pedestrians: bool) -> dict[str, Any]:
    """Return the L5 row schema for the branch's pedestrian configuration."""
    if type(has_semantic_pedestrians) is not bool:
        raise TypeError("has_semantic_pedestrians must be a bool")

    allowed_rule_ids = {
        rule["rule_id"]
        for rule in get_core_event_occurrence_types()
        if rule["event_family_id"] in COMPUTED_FAMILIES
        and (
            rule["event_family_id"] != _PEDESTRIAN_FAMILY
            or has_semantic_pedestrians
        )
    }
    schema = formal_event_occurrence_schema()
    variants = schema["allOf"][0]["oneOf"]
    selected = [
        variant
        for variant in variants
        if variant["properties"]["rule_id"]["const"] in allowed_rule_ids
    ]
    selected_ids = {
        variant["properties"]["rule_id"]["const"] for variant in selected
    }
    if selected_ids != allowed_rule_ids or len(selected) != len(allowed_rule_ids):
        raise ValueError("L5 event rules differ from the formal occurrence variants")

    schema["allOf"][0]["oneOf"] = selected
    schema["$id"] = "https://aeroworld.local/schema/event_occurrence.arm_l5.schema.json"
    schema["title"] = "L5 ARM event occurrence record"
    return schema
