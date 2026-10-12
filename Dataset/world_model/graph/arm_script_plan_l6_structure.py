"""Whole-root ARM UE script grammar for L6 cancellation and retiming.

``l6_v2.engine.execute`` starts from the profile's scenario script, removes
the explicitly cancelled events, and changes one existing integer trigger
field by an integer delta. The UE exporter copies that executed script.
The source script and cancellation/retiming declarations are explicit inputs;
the grammar keeps every remaining author's array position and nested field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Any

from Dataset.world_model.graph.contract import REPO
from Dataset.world_model.graph.structure_schema import compile_authoring_input

__all__ = ["arm_script_plan_schema"]


def arm_script_plan_schema(
    source_file: str | Path, *, cancel_event_ids: tuple[str, ...] = (),
    retime: Mapping[str, Any] | None = None,
) -> dict:
    """Return the authored script structure after the declared L6 edit."""
    source = Path(source_file)
    absolute = (source if source.is_absolute() else REPO / source).resolve(strict=True)
    relative = absolute.relative_to(REPO)
    if (relative.name != "event_script.json" or
            not any(part.startswith("L6-") for part in relative.parts) or
            relative.parts[:2] != ("Dataset", "scenarios")):
        raise ValueError(f"not an L6 scenario script: {source_file}")
    authored = json.loads(absolute.read_text(encoding="utf-8"))
    events = authored["events"]
    event_ids = [event["event_id"] for event in events]
    cancelled = set(cancel_event_ids)
    if len(cancelled) != len(cancel_event_ids) or cancelled - set(event_ids):
        raise ValueError(f"invalid L6 cancelled event ids: {cancel_event_ids!r}")
    if retime is not None:
        if set(retime) != {"trigger_id", "field", "delta"} or type(retime["delta"]) is not int:
            raise ValueError("L6 retime must declare trigger_id, field and integer delta")
        matching = [trigger for trigger in authored["triggers"]
                    if trigger["trigger_id"] == retime["trigger_id"]]
        if (len(matching) != 1 or retime["field"] not in matching[0] or
                type(matching[0][retime["field"]]) is not int):
            raise ValueError(f"L6 retime does not target an integer trigger field: {retime!r}")
    schema = compile_authoring_input(absolute).schema
    event_schema = schema["properties"]["events"]
    positions = [position for position, event_id in enumerate(event_ids)
                 if event_id not in cancelled]
    event_schema["prefixItems"] = [event_schema["prefixItems"][position]
                                   for position in positions]
    event_schema["minItems"] = len(positions)
    event_schema["maxItems"] = len(positions)
    return schema
