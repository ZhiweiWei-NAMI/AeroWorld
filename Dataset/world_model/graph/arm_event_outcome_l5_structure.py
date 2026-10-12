"""Closed event-outcome row grammar for the L5 ARM semantic producer.

``l5_arm_semantics.calculate`` uses the standard formal outcome adapter and
window wrapper.  Its event-rule set is ``COMPUTED_FAMILIES``; the pedestrian
near-miss rule is reachable only when the branch declares semantic pedestrian
IDs.  The explicit factory argument preserves that source condition.

Frozen L5 rows retain the ``event_outcome`` version ``3.0.0`` header.  The
current writer changes it to ``arm_window_event_outcome`` version ``1.0.0``.
Neither envelope adds branch identity fields to an L5 outcome row.
"""

from __future__ import annotations

from typing import Any

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
)
from Dataset.tools.l5_arm_semantics import COMPUTED_FAMILIES
from Dataset.world_model.graph.arm_event_outcome_l2_structure import (
    build_standard_arm_event_outcomes_schema,
)


__all__ = ["arm_event_outcomes_schema"]

_PEDESTRIAN_FAMILY = "uav_pedestrian_near_miss"
_L5_PROFILE_PATTERN = r"(?:L5-1_v[123]|L5-2_v[12]|L5-3_v[12])"
_SEED_PATTERN = r"(?:00|01|02)"
_ARM_SUFFIX_PATTERN = r"/[^/]+/arm_[a-z0-9_]+"


def arm_event_outcomes_schema(
    *, has_semantic_pedestrians: bool
) -> dict[str, Any]:
    """Return the L5 row grammar for the branch's pedestrian configuration."""
    if type(has_semantic_pedestrians) is not bool:
        raise TypeError("has_semantic_pedestrians must be a bool")

    allowed_event_types = frozenset(
        rule["event_type"]
        for rule in get_core_event_occurrence_types()
        if rule["event_family_id"] in COMPUTED_FAMILIES
        and (
            rule["event_family_id"] != _PEDESTRIAN_FAMILY
            or has_semantic_pedestrians
        )
    )
    episode_id_pattern = (
        "^"
        + _L5_PROFILE_PATTERN
        + "__seed"
        + _SEED_PATTERN
        + "::"
        + _L5_PROFILE_PATTERN
        + "/seed"
        + _SEED_PATTERN
        + _ARM_SUFFIX_PATTERN
        + "$"
    )
    return build_standard_arm_event_outcomes_schema(
        schema_id=(
            "https://aeroworld.local/schema/"
            "arm_event_outcome.l5.record.schema.json"
        ),
        title="L5 ARM event outcome record",
        description=(
            "One unchanged objective/event_outcomes.jsonl row from a declared "
            "L5 ARM branch."
        ),
        episode_id_pattern=episode_id_pattern,
        allowed_event_types=allowed_event_types,
    )
