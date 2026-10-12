"""Strict causal continuation helpers for action-branch scripts.

An intervention is evaluated from the immutable parent episode.  When a
causal source event is suppressed, events that explicitly depend on it must
also be disabled; otherwise a later scheduled action can recreate the very
state the intervention is meant to remove.  This module derives that closure
from the authored trigger graph and causal metadata only.  It never invents
an event, tick, entity, or action.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any, Mapping


class CausalContinuationError(RuntimeError):
    """The authored event graph is insufficient for a strict continuation."""


def _event_index(script: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    events: dict[str, Mapping[str, Any]] = {}
    for event in script.get("events") or ():
        if not isinstance(event, Mapping):
            continue
        event_id = str(event.get("event_id") or "")
        if event_id:
            events[event_id] = event
    return events


def _trigger_index(script: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    triggers: dict[str, Mapping[str, Any]] = {}
    for trigger in script.get("triggers") or ():
        if not isinstance(trigger, Mapping):
            continue
        trigger_id = str(trigger.get("trigger_id") or "")
        if trigger_id:
            triggers[trigger_id] = trigger
    return triggers


def _trigger_event_dependencies(
    trigger_id: str,
    triggers: Mapping[str, Mapping[str, Any]],
    resolving: frozenset[str] = frozenset(),
) -> set[str]:
    """Return event IDs required by one authored trigger."""

    if not trigger_id:
        return set()
    if trigger_id in resolving:
        raise CausalContinuationError(
            f"cyclic composite trigger dependency: {trigger_id!r}"
        )
    trigger = triggers.get(trigger_id)
    if trigger is None:
        raise CausalContinuationError(f"event references missing trigger: {trigger_id!r}")
    trigger_type = str(trigger.get("type") or "")
    if trigger_type in {"event_fired", "event_fired_after"}:
        event_id = str(trigger.get("event_id") or "")
        if not event_id:
            raise CausalContinuationError(
                f"{trigger_type} trigger lacks event_id: {trigger_id!r}"
            )
        return {event_id}
    if trigger_type != "composite":
        return set()
    children = trigger.get("children")
    if not isinstance(children, list) or not children:
        raise CausalContinuationError(
            f"composite trigger lacks children: {trigger_id!r}"
        )
    result: set[str] = set()
    for child in children:
        result.update(
            _trigger_event_dependencies(
                str(child or ""),
                triggers,
                resolving=resolving | {trigger_id},
            )
        )
    return result


def _stage_rank(event: Mapping[str, Any]) -> int | None:
    value = str(event.get("intent_stage") or "")
    match = re.match(r"^(\d+)(?:\.|$)", value)
    return int(match.group(1)) if match else None


def _static_trigger_tick(
    event: Mapping[str, Any], triggers: Mapping[str, Mapping[str, Any]]
) -> int | None:
    trigger = triggers.get(str(event.get("trigger_ref") or ""))
    if trigger is None or str(trigger.get("type") or "") != "tick":
        return None
    value = trigger.get("tick")
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _may_follow_anchor(
    event: Mapping[str, Any],
    anchor: Mapping[str, Any],
    triggers: Mapping[str, Mapping[str, Any]],
    *,
    fork_tick: int | None,
) -> bool:
    """Reject metadata links that point to an upstream/parallel event."""

    candidate_chain = str(event.get("causal_chain_id") or "")
    anchor_chain = str(anchor.get("causal_chain_id") or "")
    if candidate_chain and anchor_chain and candidate_chain != anchor_chain:
        return False

    candidate_tick = _static_trigger_tick(event, triggers)
    anchor_tick = _static_trigger_tick(anchor, triggers)
    lower_bound = fork_tick if fork_tick is not None else anchor_tick
    if candidate_tick is not None and lower_bound is not None and candidate_tick < lower_bound:
        return False

    candidate_stage = _stage_rank(event)
    anchor_stage = _stage_rank(anchor)
    # A predecessor-intent string by itself is not a causal edge.  When both
    # authored stages are available, only a strictly later stage proves that
    # the event is downstream.  Same-stage events are parallel context unless
    # they were already connected by an explicit trigger/event edge above.
    if candidate_stage is not None and anchor_stage is not None:
        return candidate_stage > anchor_stage
    # Some older authored chains omit stage numbers but carry concrete tick
    # triggers.  In that narrower case a later tick is the only admissible
    # ordering evidence.  With neither ordering signal, remain fail-closed.
    if candidate_tick is not None and anchor_tick is not None:
        return candidate_tick > anchor_tick
    return False


def causal_suppression_closure(
    script: Mapping[str, Any],
    anchor_event_id: str,
    *,
    fork_tick: int | None = None,
) -> list[str]:
    """Resolve the authored event bundle affected by source suppression.

    Explicit trigger dependencies and ``on_fire.emit_events`` edges are
    always followed.  A later event whose authored
    ``causal_predecessor_intent`` equals the intent of a suppressed event is
    included only when its chain/stage/tick metadata proves it is downstream.
    """

    events = _event_index(script)
    triggers = _trigger_index(script)
    anchor = events.get(str(anchor_event_id))
    if anchor is None:
        raise CausalContinuationError(f"anchor event is not authored: {anchor_event_id!r}")

    dependencies: dict[str, set[str]] = defaultdict(set)
    for event_id, event in events.items():
        dependencies[event_id].update(
            _trigger_event_dependencies(
                str(event.get("trigger_ref") or ""),
                triggers,
            )
        )
        predecessor_ids = event.get("causal_predecessor_event_ids")
        if predecessor_ids is not None and not isinstance(predecessor_ids, (list, tuple, set)):
            raise CausalContinuationError(
                f"causal_predecessor_event_ids must be a sequence: {event_id!r}"
            )
        dependencies[event_id].update(
            str(value)
            for value in predecessor_ids or ()
            if str(value)
        )
        missing_predecessors = sorted(
            value for value in dependencies[event_id] if value not in events
        )
        if missing_predecessors:
            raise CausalContinuationError(
                f"event references non-authored predecessor(s): {event_id!r}: "
                f"{missing_predecessors}"
            )
        on_fire = event.get("on_fire")
        if on_fire is not None and not isinstance(on_fire, Mapping):
            raise CausalContinuationError(
                f"on_fire must be an object: {event_id!r}"
            )
        for emitted_id in (on_fire or {}).get("emit_events") or ():
            emitted = str(emitted_id or "")
            if emitted and emitted not in events:
                raise CausalContinuationError(
                    f"event emits non-authored event: {event_id!r} -> {emitted!r}"
                )
            if emitted in events:
                dependencies[emitted].add(event_id)

    closure: set[str] = {str(anchor_event_id)}
    while True:
        before = len(closure)
        tokens: set[str] = set()
        for event_id in closure:
            event = events[event_id]
            for field in ("intent", "intent_stage", "event_id"):
                value = str(event.get(field) or "")
                if value:
                    tokens.add(value)
        for event_id, required in dependencies.items():
            if event_id in closure:
                continue
            if required & closure:
                closure.add(event_id)
                continue
            predecessor_intent = str(
                events[event_id].get("causal_predecessor_intent") or ""
            )
            if (
                predecessor_intent
                and predecessor_intent in tokens
                and _may_follow_anchor(
                    events[event_id],
                    anchor,
                    triggers,
                    fork_tick=fork_tick,
                )
            ):
                closure.add(event_id)
        if len(closure) == before:
            break
    return sorted(closure)


def suppress_event_triggers(
    script: Mapping[str, Any],
    event_ids: list[str] | tuple[str, ...] | set[str],
    *,
    never_tick: int,
) -> dict[str, dict[str, Any]]:
    """Disable exactly the supplied authored events and return originals."""

    events = _event_index(script)
    triggers = _trigger_index(script)
    originals: dict[str, dict[str, Any]] = {}
    for event_id in sorted({str(value) for value in event_ids if str(value)}):
        event = events.get(event_id)
        if event is None:
            raise CausalContinuationError(f"suppressed event is not authored: {event_id!r}")
        trigger_id = str(event.get("trigger_ref") or "")
        trigger = triggers.get(trigger_id)
        if not isinstance(trigger, dict):
            raise CausalContinuationError(
                f"suppressed event trigger is missing or immutable: {event_id!r}"
            )
        originals[event_id] = copy.deepcopy(dict(trigger))
        trigger.clear()
        trigger.update({"trigger_id": trigger_id, "type": "tick", "tick": int(never_tick)})
    return originals
