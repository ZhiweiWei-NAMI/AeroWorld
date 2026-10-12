"""Reconstruct a declared parent-plus-delta trajectory without inventing rows."""

from typing import Any

from Dataset.tools.structured_branch_source import DELTA_SCOPES


def compose_trajectory(
    parent: dict[str, dict[int, dict[str, Any]]],
    delta: dict[str, dict[int, dict[str, Any]]],
    lifecycle: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, dict[int, dict[str, Any]]]:
    if manifest.get("artifact_scope") not in DELTA_SCOPES:
        raise ValueError("branch must declare a supported parent-plus-delta scope")
    if manifest.get("delta_only") is not True:
        raise ValueError("branch must declare delta_only")
    fork, end = manifest["fork_tick"], manifest["window_end_tick"]
    result = {entity: dict(samples) for entity, samples in parent.items()}
    for entity, samples in delta.items():
        for tick, row in samples.items():
            if not fork <= tick <= end:
                raise ValueError("delta trajectory lies outside declared branch window")
            result.setdefault(entity, {})[tick] = row
    seen = set()
    for row in lifecycle:
        entity, tick, present = row["entity_id"], row["tick"], row["present"]
        if (entity, tick) in seen or not fork <= tick <= end or type(present) is not bool:
            raise ValueError("invalid lifecycle delta")
        seen.add((entity, tick))
        if present:
            if tick not in delta.get(entity, {}):
                raise ValueError("added entity has no measured delta trajectory")
        else:
            if tick in delta.get(entity, {}):
                raise ValueError("removed entity still has a delta trajectory")
            result.get(entity, {}).pop(tick, None)
    return {entity: samples for entity, samples in result.items() if samples}
