"""Whole-root ARM UE script grammar for dedicated X branch scripts.

``x_arm_pipeline.run_arm`` checks each intervention script against its
dedicated entrypoint's ``build_script(source)`` and passes that script to
``arm_ue_export.export_arm``. Build the same result from the declared scenario
source only when requested; importing this module scans no files.
"""

from __future__ import annotations

import json
from pathlib import Path

from Dataset.tools.x_arm_pipeline import entry_module
from Dataset.world_model.graph.contract import REPO
from Dataset.world_model.graph.structure_schema import _literal_schema

__all__ = ["arm_script_plan_schema"]

_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _source_path(path: str | Path, prefix: tuple[str, ...]) -> Path:
    supplied = Path(path)
    absolute = (supplied if supplied.is_absolute() else REPO / supplied).resolve(strict=True)
    relative = absolute.relative_to(REPO)
    if relative.parts[:len(prefix)] != prefix:
        raise ValueError(f"path is outside declared X producer: {path}")
    return absolute


def arm_script_plan_schema(source_file: str | Path, entrypoint: str | Path) -> dict:
    """Return the closed structure of the dedicated X producer's script."""
    source = _source_path(source_file, ("Dataset", "scenarios", "X_cross_layer"))
    entry = _source_path(entrypoint, ("scripts", "arms"))
    if source.name != "event_script.json" or not source.parent.name.startswith("X"):
        raise ValueError(f"not an X scenario script: {source_file}")
    if source.parent.name not in entry.relative_to(REPO).parts:
        raise ValueError(f"X entrypoint profile differs from source: {entrypoint}")
    authored = json.loads(source.read_text(encoding="utf-8"))
    produced = entry_module(entry).build_script(authored)
    if type(produced) is not dict:
        raise TypeError(f"X build_script did not produce a JSON object: {entrypoint}")
    return {"$schema": _DRAFT, **_literal_schema(produced)}
