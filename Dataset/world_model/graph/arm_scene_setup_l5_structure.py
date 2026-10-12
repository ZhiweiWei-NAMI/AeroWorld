"""L5 ARM derived-scene grammar, gated by the checked entry transform."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from Dataset.world_model.graph.arm_scene_setup_l1_structure import (
    _authored_scene_schema, _canonical_relative,
)
from Dataset.world_model.graph.authoring_structure import authoring_source_grammar


def arm_scene_setup_schema(source_file: str | Path) -> dict[str, Any]:
    relative = _canonical_relative(source_file)
    parts = relative.parts
    if (len(parts) != 5 or parts[0] != "design"
            or not re.fullmatch(r"L5-[0-9]+_v[0-9]+", parts[1])
            or not re.fullmatch(r"seed[0-9]{2}", parts[2])
            or not re.fullmatch(r"grp_[a-z0-9_]+", parts[3])
            or not re.fullmatch(r"arm_[a-z0-9_]+\.scene\.json", parts[4])):
        raise ValueError(f"{relative}: not a declared L5 branch scene")

    # The existing authoring gate verifies the manifest/config references,
    # exact entry AST, offset transform, and equality of both derived files.
    script = relative.with_name(relative.name.removesuffix(".scene.json") + ".script.json")
    authoring_source_grammar("script_category_source", script)
    authored = (f"Dataset/scenarios/L5_environment/interaction/"
                f"{parts[1]}/scene_setup.json")
    return _authored_scene_schema("L5", authored)
