"""Choose a group-stable measured state subject with explicit role evidence."""

from typing import Any, Mapping, Sequence


def resolve_state_subject(
    target_rows: Sequence[Mapping[str, Any]],
    target_ids: set[str],
    parent_entity_ids: set[str],
    roster: Mapping[str, Any],
    *,
    truth_source: str,
    roster_source: str,
) -> dict[str, Any]:
    actor_ids = set()
    for row in target_rows:
        bindings = row.get("bindings")
        if isinstance(bindings, Mapping):
            actor = bindings.get("actor")
            if isinstance(actor, str) and actor:
                actor_ids.add(actor)
    if len(actor_ids) == 1:
        entity = next(iter(actor_ids))
        if entity in parent_entity_ids:
            return {"entity_id": entity, "role": "predicate_actor", "source": truth_source,
                    "source_field": "bindings.actor", "target_entity_ids": sorted(target_ids)}
    if len(target_ids) == 1:
        entity = next(iter(target_ids))
        if entity in parent_entity_ids:
            return {"entity_id": entity, "role": "predicate_target", "source": truth_source,
                    "source_field": "bindings", "target_entity_ids": sorted(target_ids)}
    # Inspect observers are explicitly authored to observe the episode's
    # capture boundary. They do not replace the semantic target or its truth.
    for role in ("inspect_observer", "observer"):
        entities = {item["entity_id"] for item in roster.get("entities", [])
                    if item.get("label_class") == "uav" and item.get("uav_corridor_role") == role}
        if len(entities) > 1:
            return {"entity_id": None, "reason": "AMBIGUOUS_DECLARED_OBSERVER",
                    "candidates": sorted(entities), "source": roster_source}
        if len(entities) == 1:
            entity = next(iter(entities))
            if entity not in parent_entity_ids:
                return {"entity_id": None, "reason": "DECLARED_OBSERVER_TRAJECTORY_MISSING",
                        "candidates": [entity], "source": roster_source}
            return {"entity_id": entity, "role": role, "source": roster_source,
                    "source_field": "entities[].uav_corridor_role",
                    "target_entity_ids": sorted(target_ids)}
    return {"entity_id": None, "reason": "STATE_SUBJECT_UNRESOLVED",
            "target_entity_ids": sorted(target_ids)}
