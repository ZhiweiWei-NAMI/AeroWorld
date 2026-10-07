"""Entity classification for the event and acceptance layers.

Predicate truth is computed for every in-scope entity.  This module decides only
whether an authored event may name an entity, and which entities the acceptance
layer treats as background presence.

Classification reads the authoritative record fields declared in
Dataset/semantic_rules/profiles/entity_scope_classes.json.  Two declarations make an
entity presence-only: an explicit background_role, and an authored UAV task role
(U_inspect / corridor observer) that the same contract lists as observation
presence.  An identifier is used solely to look the record up -- its text is never
inspected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "semantic_rules"
    / "profiles"
    / "entity_scope_classes.json"
)

_CONTRACT_CACHE: dict[str, Any] | None = None


def entity_scope_contract() -> dict[str, Any]:
    """Load the governed entity scope classification table."""
    global _CONTRACT_CACHE
    if _CONTRACT_CACHE is None:
        _CONTRACT_CACHE = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return _CONTRACT_CACHE


def roster_index(roster_payload: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Index a roster payload (or its entity list) by entity id."""
    if isinstance(roster_payload, Mapping):
        entities = roster_payload.get("entities")
        if entities is None:
            raise ValueError("roster payload lacks an entity list")
    else:
        entities = list(roster_payload)
    if not isinstance(entities, (list, tuple)) or not entities:
        raise ValueError("roster payload lacks a non-empty entity list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise ValueError("roster entity record must be an object")
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("roster entity record lacks entity_id")
        if entity_id in indexed:
            raise ValueError(f"duplicate entity_id in one roster: {entity_id}")
        indexed[entity_id] = entity
    return indexed


def role_is_actor(role_name: str) -> bool:
    """True when the role names the acting subject of an event."""
    kinds = entity_scope_contract()["role_role_kinds"]
    if role_name in kinds["actor_roles"]:
        return True
    if role_name in kinds["non_actor_roles"]:
        return False
    raise ValueError(
        f"event role {role_name!r} is not declared in entity_scope_classes; "
        "declare it under role_role_kinds before use"
    )


def undeclared_event_roles() -> dict[str, list[str]]:
    """Event-rule participant roles with no declaration in entity_scope_classes.

    The objective semantic contract binds participants by role name while this
    module owns the role classification.  The two drifted apart once already:
    ``region`` and ``crowd`` were bound by event rules but absent from
    ``role_role_kinds``, and because ``role_is_actor`` fails closed, every
    environment event raised instead of being built.  Callers that compile
    event rules should fail loudly on a non-empty result rather than discover
    the gap the first time a matching predicate fires.
    """
    declared = {
        *entity_scope_contract()["role_role_kinds"]["actor_roles"],
        *entity_scope_contract()["role_role_kinds"]["non_actor_roles"],
    }
    from Dataset.semantic_truth.core_semantic_registry import (
        get_core_event_occurrence_types,
    )

    offenders: dict[str, list[str]] = {}
    for rule in get_core_event_occurrence_types():
        for participant in rule.get("participant_roles") or []:
            role = str(participant.get("role") or "")
            if role and role not in declared:
                offenders.setdefault(role, []).append(str(rule.get("event_family_id") or rule.get("rule_id") or ""))
    return offenders


def is_background(entity_id: str, roster_by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    """True when the record declares this entity as presence, not an event actor.

    Two declarations qualify: an explicit governed background_role, or an authored
    UAV task role that the contract lists under non_bindable_uav_task_roles.  The
    task role is resolved from the entity's own declared role fields, so a scene
    that has already authored ``role: U_inspect`` needs no extra field.
    """
    if not roster_by_id:
        raise ValueError(
            "background classification requires the episode roster; "
            "pass roster_by_id built from global_entity_roster.json"
        )
    record = roster_by_id.get(entity_id)
    if record is None:
        return False
    contract = entity_scope_contract()
    if record.get("background_role") in contract["background_role_values"]:
        return True
    return _authored_presence_uav_task_role(record, contract)


def _authored_presence_uav_task_role(
    record: Mapping[str, Any], contract: Mapping[str, Any]
) -> bool:
    """True when the record's authored task role is declared observation presence."""
    presence_roles = contract.get("non_bindable_uav_task_roles")
    if not isinstance(presence_roles, (list, tuple)) or not presence_roles:
        return False
    if str(record.get("entity_category") or "") != "uav":
        return False
    return uav_task_role(record) in presence_roles


def participant_binding_allowed(
    entity_id: str, role_name: str, roster_by_id: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Decide whether an event rule may bind this entity in this role.

    A background entity is never bound.  A non-background entity is bound as an
    actor only when its category permits acting, and in a non-actor role only when
    it is real scene geometry.  An identifier absent from the roster is
    infrastructure and can fill non-actor roles only.
    """
    contract = entity_scope_contract()
    record = roster_by_id.get(entity_id)
    is_actor = role_is_actor(role_name)
    if record is None:
        return not is_actor
    if record.get("background_role") in contract["background_role_values"]:
        return False
    if _authored_presence_uav_task_role(record, contract):
        return False
    category = str(record.get("entity_category") or "")
    if is_actor:
        return category in contract["actor_categories"]
    return (
        category in contract["actor_categories"]
        or category in contract["non_actor_categories"]
    )


__all__ = [
    "CONTRACT_PATH",
    "entity_scope_contract",
    "is_background",
    "participant_binding_allowed",
    "role_is_actor",
    "undeclared_event_roles",
    "roster_index",
]


def uav_task_role(entity: Mapping[str, Any]) -> str:
    """Classify an authored UAV as inspect, observer or mission.

    Values come from the roster record fields declared under
    ``uav_role_classification``; the entity identifier is never inspected.
    """
    config = entity_scope_contract().get("uav_role_classification")
    if not isinstance(config, Mapping):
        raise ValueError("entity_scope_classes lacks uav_role_classification")
    initial_state = dict(entity.get("initial_state") or {})
    fallback = str(config["fallback_class"])
    for scope_class in config["evaluation_order"]:
        rule = config["class_rules"].get(scope_class) or {}
        if rule.get("fallback") is True:
            return str(scope_class)
        contract_field = rule.get("contract_field")
        if contract_field and entity.get(contract_field):
            return str(scope_class)
        for field, key in (
            ("role", "role_in"),
            ("uav_corridor_role", "uav_corridor_role_in"),
            ("semantic_role", "semantic_role_in"),
        ):
            allowed = rule.get(key)
            if not allowed:
                continue
            value = entity.get(field) or initial_state.get(field)
            if str(value or "") in allowed:
                return str(scope_class)
    return fallback
