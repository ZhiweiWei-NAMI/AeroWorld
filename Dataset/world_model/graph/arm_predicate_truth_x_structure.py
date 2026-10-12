"""Closed X ARM ``objective/predicate_truth.jsonl`` row grammar.

``x_arm_pipeline.run_arm`` writes ``calculate``'s seven-key rows to the
dense tick file and writes the same row objects with ``tick % 5 == 0`` to
this sampled truth file.  The X tick schema owns their ten producer shapes;
this schema takes an independent copy and constrains only the sampled grid.

The source and effective family are ``arm_predicate_truth``.  The file path
carries the X profile because the JSONL record has no profile field.  These
rows have no integrity metadata.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.world_model.graph.arm_predicate_ticks_x_structure import (
    arm_predicate_truth_ticks_schema,
)


__all__ = ["arm_predicate_truth_schema"]


def arm_predicate_truth_schema() -> dict[str, Any]:
    """Return a detached schema for one sampled X predicate truth row."""
    schema = deepcopy(arm_predicate_truth_ticks_schema(layer="X"))
    schema["$id"] = (
        "https://aeroworld.local/schema/arm_predicate_truth_x.record.schema.json"
    )
    schema["title"] = "X ARM sampled predicate truth JSONL record"
    for branch in schema["oneOf"]:
        branch["properties"]["tick"].update({"maximum": 900, "multipleOf": 5})
    return schema
