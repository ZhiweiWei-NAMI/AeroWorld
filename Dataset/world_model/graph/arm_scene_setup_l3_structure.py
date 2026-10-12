"""L3 ARM scene grammar from the referenced formal authoring scene."""

from pathlib import Path
from typing import Any

from Dataset.world_model.graph.arm_scene_setup_l1_structure import _authored_scene_schema


def arm_scene_setup_schema(source_file: str | Path) -> dict[str, Any]:
    return _authored_scene_schema("L3", source_file)
