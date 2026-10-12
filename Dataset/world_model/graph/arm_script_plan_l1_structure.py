"""Whole-root ARM UE script grammar for L1--L5 copied script inputs.

``arm_ue_export.export_arm`` copies the script argument to
``ue/event_script.json`` without editing it. L1--L4 pass the declared
scenario authoring script. L5 passes the derived design script, whose
producer and equality to its scenario input are checked by
``authoring_source_grammar``. The source path is explicit because each
profile has its own fixed script structure.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from Dataset.world_model.graph.authoring_structure import authoring_source_grammar

__all__ = ["arm_script_plan_schema"]


def arm_script_plan_schema(level: str, source_file: str | Path) -> dict:
    """Return a fresh closed JSON-root schema for one declared ARM script."""
    if level not in {"L1", "L2", "L3", "L4", "L5"}:
        raise ValueError(f"unsupported copied ARM script level: {level!r}")
    source = Path(source_file)
    if not any(part.startswith(f"{level}-") for part in source.parts):
        raise ValueError(f"source profile does not match {level}: {source_file}")
    family = "script_category_source" if level == "L5" else "formal_source_script"
    schema = authoring_source_grammar(family, source_file).schema
    if level == "L2" and "L2-5_v1" in source.parts:
        # The frozen L2-5_v1 UE roots place the runtime-state action before
        # the visual-state action in event 6; the current scenario authoring
        # source has the reverse order. Keep both exact positional structures.
        actions = schema["properties"]["events"]["prefixItems"][6]["properties"]["actions"]
        reordered = deepcopy(actions)
        reordered["prefixItems"][0], reordered["prefixItems"][1] = (
            reordered["prefixItems"][1], reordered["prefixItems"][0]
        )
        schema["properties"]["events"]["prefixItems"][6]["properties"]["actions"] = {
            "oneOf": [actions, reordered]
        }
    return schema
