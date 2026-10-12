"""Deterministic projection of authored script predecessors onto runtime events.

This module is deliberately independent from the objective semantic pipeline.  A
script-declared predecessor is an authored control/narrative relation, not an L2
physical ``causes`` relation.  The functions below therefore consume already
loaded script, trace, and realization objects and only emit
``script_declared_causal`` or ``script_declared_context`` records.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from .provenance import digest_object, sha256_bytes


SCHEMA_VERSION = "2.0.0"
INTENT_STAGE_PATTERN = re.compile(r"^(\d+)\.(.+)$")
SOURCE_KINDS = {"script_event", "intent_stage_set", "context_anchor"}
RELATION_TYPES = {
    "script_declared_causal",
    "script_declared_context",
}
RUNTIME_PASS_VALIDATION_STATUSES = {
    "runtime_pass",
    "runtime_pass_context",
    "runtime_pass_co_triggered",
    "runtime_pass_declared_set",
}


class EventChainError(ValueError):
    """Base class for deterministic event-chain construction errors."""


class EventChainStructureError(EventChainError):
    """Raised when a script cannot identify one deterministic authored chain."""


@dataclass(frozen=True)
class EventChainBuildResult:
    manifest: Mapping[str, Any]
    relations: tuple[Mapping[str, Any], ...]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_identifier(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _require_text(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = _text(record.get(field))
    if not value:
        raise EventChainStructureError(f"{context}: {field} must be a non-empty string")
    return value


def _stage_number(value: str, *, context: str) -> int:
    match = INTENT_STAGE_PATTERN.fullmatch(value)
    if match is None:
        raise EventChainStructureError(
            f"{context}: intent_stage must use '<integer>.<name>': {value!r}"
        )
    return int(match.group(1))


def _digest_field(provenance: Mapping[str, Any], field: str) -> str:
    return _require_text(provenance, field, context="input provenance")


def _normalized_script(
    script: Mapping[str, Any], *, epi_id: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    scenario_id = _require_text(script, "scenario_id", context="event script")
    if scenario_id != epi_id:
        raise EventChainStructureError(
            f"event script scenario_id {scenario_id!r} differs from EPI {epi_id!r}"
        )
    raw_events = script.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise EventChainStructureError("event script: events must be a non-empty array")

    events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    chain_ids: set[str] = set()
    roots: list[str] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise EventChainStructureError(f"events[{index}] must be an object")
        event = dict(raw)
        context = f"events[{index}]"
        event_id = _require_text(event, "event_id", context=context)
        if event_id in seen_event_ids:
            raise EventChainStructureError(f"duplicate script event_id: {event_id}")
        seen_event_ids.add(event_id)
        _require_text(event, "intent", context=f"event {event_id}")
        stage = _require_text(event, "intent_stage", context=f"event {event_id}")
        _stage_number(stage, context=f"event {event_id}")
        chain_id = _require_text(event, "causal_chain_id", context=f"event {event_id}")
        chain_ids.add(chain_id)
        if "causal_predecessor_intent" not in event:
            raise EventChainStructureError(
                f"event {event_id}: causal_predecessor_intent field is absent"
            )
        predecessor = _text(event.get("causal_predecessor_intent"))
        if not predecessor:
            roots.append(event_id)
        mode = event.get("causal_predecessor_mode")
        declared_ids = event.get("causal_predecessor_event_ids")
        if (mode is None) != (declared_ids is None):
            raise EventChainStructureError(
                f"event {event_id}: causal_predecessor_event_ids and "
                "causal_predecessor_mode must be declared together"
            )
        if mode is not None:
            if mode != "all_of":
                raise EventChainStructureError(
                    f"event {event_id}: causal_predecessor_mode must be all_of"
                )
            if (
                not isinstance(declared_ids, list)
                or not declared_ids
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in declared_ids
                )
                or len({_text(item) for item in declared_ids}) != len(declared_ids)
            ):
                raise EventChainStructureError(
                    f"event {event_id}: causal_predecessor_event_ids must be a "
                    "non-empty unique string array"
                )
        temporal_policy = event.get("causal_temporal_policy")
        if temporal_policy not in (None, "same_tick_required"):
            raise EventChainStructureError(
                f"event {event_id}: unsupported causal_temporal_policy "
                f"{temporal_policy!r}"
            )
        if temporal_policy is not None and (
            not predecessor or predecessor == "context"
        ):
            raise EventChainStructureError(
                f"event {event_id}: causal_temporal_policy requires a non-context "
                "causal predecessor"
            )
        events.append(event)

    if len(chain_ids) != 1:
        raise EventChainStructureError(
            f"event script must declare exactly one causal chain, found {sorted(chain_ids)}"
        )
    if len(roots) != 1:
        raise EventChainStructureError(
            f"event script must declare exactly one root, found {roots}"
        )
    if roots[0] != _text(events[0].get("event_id")):
        raise EventChainStructureError(
            f"chain root must be events[0], found root {roots[0]!r}"
        )

    raw_triggers = script.get("triggers")
    if not isinstance(raw_triggers, list):
        raise EventChainStructureError("event script: triggers must be an array")
    triggers: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_triggers):
        if not isinstance(raw, Mapping):
            raise EventChainStructureError(f"triggers[{index}] must be an object")
        trigger = dict(raw)
        trigger_id = _require_text(trigger, "trigger_id", context=f"triggers[{index}]")
        if trigger_id in triggers:
            raise EventChainStructureError(f"duplicate trigger_id: {trigger_id}")
        triggers[trigger_id] = trigger
    for event in events:
        event_id = _text(event["event_id"])
        trigger_ref = _require_text(event, "trigger_ref", context=f"event {event_id}")
        if trigger_ref not in triggers:
            raise EventChainStructureError(
                f"event {event_id}: trigger_ref does not exist: {trigger_ref}"
            )
    event_positions = {
        _text(event.get("event_id")): index for index, event in enumerate(events)
    }
    for index, event in enumerate(events):
        event_id = _text(event.get("event_id"))
        dependency = _direct_scheduling_dependency(event, triggers)
        if dependency is not None and (
            dependency not in event_positions or event_positions[dependency] >= index
        ):
            raise EventChainStructureError(
                f"event {event_id}: scheduling dependency must reference an earlier "
                f"script event: {dependency}"
            )
    return events, triggers, next(iter(chain_ids))


def _script_event_endpoint(event: Mapping[str, Any]) -> dict[str, Any]:
    event_id = _text(event.get("event_id"))
    return {
        "kind": "script_event",
        "event_id": event_id,
        "intent": _text(event.get("intent")),
        "intent_stage": _text(event.get("intent_stage")),
        "candidate_event_ids": [event_id],
    }


def _direct_scheduling_dependency(
    event: Mapping[str, Any], triggers: Mapping[str, Mapping[str, Any]]
) -> str | None:
    trigger = triggers[_text(event.get("trigger_ref"))]
    if trigger.get("type") not in {"event_fired", "event_fired_after"}:
        return None
    value = _text(trigger.get("event_id"))
    return value or None


def resolve_source_endpoint(
    events: Sequence[Mapping[str, Any]],
    triggers: Mapping[str, Mapping[str, Any]],
    target_index: int,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Resolve one non-root predecessor without temporal or lexical guessing."""

    target = events[target_index]
    target_id = _text(target.get("event_id"))
    predecessor = _text(target.get("causal_predecessor_intent"))
    if not predecessor:
        raise EventChainStructureError(
            f"event {target_id}: roots do not have predecessor relation records"
        )
    prior = [
        event
        for event in events[:target_index]
        if _text(event.get("causal_chain_id")) == _text(target.get("causal_chain_id"))
    ]
    scheduling_id = _direct_scheduling_dependency(target, triggers)

    if predecessor == "context":
        if target.get("causal_predecessor_event_ids") is not None:
            raise EventChainStructureError(
                f"event {target_id}: context predecessors cannot declare event-set "
                "cardinality"
            )
        candidates = [
            event for event in prior if _text(event.get("intent")) == "context"
        ]
        candidate_ids = tuple(_text(event.get("event_id")) for event in candidates)
        if not candidates:
            return (
                {},
                {
                    "method": "source_declaration_missing",
                    "status": "source_declaration_missing",
                    "cardinality": "context",
                    "scheduling_dependency_event_id": scheduling_id,
                },
                (),
            )
        return (
            {
                "kind": "context_anchor",
                "intent": "context",
                "candidate_event_ids": list(candidate_ids),
                "candidate_intent_stages": [
                    _text(event.get("intent_stage")) for event in candidates
                ],
            },
            {
                "method": "context_anchor",
                "status": "resolved_context",
                "cardinality": "context",
                "scheduling_dependency_event_id": scheduling_id,
            },
            candidate_ids,
        )

    matching = [event for event in prior if _text(event.get("intent")) == predecessor]
    explicit = next(
        (event for event in matching if _text(event.get("event_id")) == scheduling_id),
        None,
    )
    if explicit is not None:
        if target.get("causal_predecessor_event_ids") is not None:
            raise EventChainStructureError(
                f"event {target_id}: cardinality declaration is invalid for an exact "
                "event-trigger predecessor"
            )
        event_id = _text(explicit.get("event_id"))
        return (
            _script_event_endpoint(explicit),
            {
                "method": "explicit_event_trigger",
                "status": "resolved_unique",
                "cardinality": "single",
                "scheduling_dependency_event_id": scheduling_id,
            },
            (event_id,),
        )
    if len(matching) == 1:
        if target.get("causal_predecessor_event_ids") is not None:
            raise EventChainStructureError(
                f"event {target_id}: cardinality declaration is invalid for a unique "
                "prior-intent predecessor"
            )
        event_id = _text(matching[0].get("event_id"))
        return (
            _script_event_endpoint(matching[0]),
            {
                "method": "unique_prior_intent",
                "status": "resolved_unique",
                "cardinality": "single",
                "scheduling_dependency_event_id": scheduling_id,
            },
            (event_id,),
        )
    if not matching:
        if target.get("causal_predecessor_event_ids") is not None:
            raise EventChainStructureError(
                f"event {target_id}: declared predecessor set has no matching prior intent"
            )
        return (
            {},
            {
                "method": "source_declaration_missing",
                "status": "source_declaration_missing",
                "cardinality": "unspecified",
                "scheduling_dependency_event_id": scheduling_id,
            },
            (),
        )

    by_stage: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in matching:
        stage = _text(candidate.get("intent_stage"))
        by_stage[
            _stage_number(stage, context=f"event {_text(candidate.get('event_id'))}")
        ].append(candidate)
    latest = by_stage[max(by_stage)]
    if len(latest) == 1:
        if target.get("causal_predecessor_event_ids") is not None:
            raise EventChainStructureError(
                f"event {target_id}: cardinality declaration is invalid for a unique "
                "latest-stage predecessor"
            )
        event_id = _text(latest[0].get("event_id"))
        return (
            _script_event_endpoint(latest[0]),
            {
                "method": "latest_prior_stage_unique",
                "status": "resolved_unique",
                "cardinality": "single",
                "scheduling_dependency_event_id": scheduling_id,
            },
            (event_id,),
        )

    candidate_ids = tuple(_text(event.get("event_id")) for event in latest)
    declared_ids = target.get("causal_predecessor_event_ids")
    mode = target.get("causal_predecessor_mode")
    status = "requires_cardinality_declaration"
    cardinality = "unspecified"
    output_ids = candidate_ids
    if declared_ids is not None:
        normalized_declared = tuple(_text(item) for item in declared_ids)
        if set(normalized_declared) != set(candidate_ids):
            raise EventChainStructureError(
                f"event {target_id}: declared predecessor set {normalized_declared} "
                f"differs from latest-stage candidates {candidate_ids}"
            )
        output_ids = normalized_declared
        status = "resolved_declared_set"
        cardinality = str(mode)
    latest_stage = _text(latest[0].get("intent_stage"))
    return (
        {
            "kind": "intent_stage_set",
            "intent": predecessor,
            "intent_stage": latest_stage,
            "candidate_event_ids": list(output_ids),
        },
        {
            "method": "intent_stage_set",
            "status": status,
            "cardinality": cardinality,
            "scheduling_dependency_event_id": scheduling_id,
        },
        output_ids,
    )


def _index_runtime_rows(
    rows: Sequence[Mapping[str, Any]], *, id_field: str, active_only: bool
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unidentified: list[int] = []
    for row_index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            unidentified.append(row_index)
            continue
        row = dict(raw)
        if active_only and row.get("status") != "active":
            continue
        event_id = _text(row.get(id_field))
        if not event_id:
            unidentified.append(row_index)
            continue
        index[event_id].append(row)
    return dict(index), unidentified


def _runtime_audit(
    events: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    realization_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    set[str],
]:
    trace_index, unidentified_trace = _index_runtime_rows(
        trace_rows, id_field="chain_id", active_only=True
    )
    realization_index, unidentified_realization = _index_runtime_rows(
        realization_rows, id_field="event_id", active_only=False
    )
    expected_ids = {_text(event.get("event_id")) for event in events}
    mismatches: list[dict[str, Any]] = []
    erroneous_event_ids: set[str] = set()
    action_failure_count = 0

    for event in events:
        event_id = _text(event.get("event_id"))
        trace_candidates = trace_index.get(event_id, [])
        realization_candidates = realization_index.get(event_id, [])
        if len(trace_candidates) == 1:
            trace = trace_candidates[0]
            for field in (
                "intent",
                "intent_stage",
                "causal_chain_id",
                "causal_predecessor_intent",
            ):
                expected = _text(event.get(field))
                actual = _text(trace.get(field))
                if expected != actual:
                    mismatches.append(
                        {
                            "event_id": event_id,
                            "source": "event_trace",
                            "field": field,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
                    erroneous_event_ids.add(event_id)
        if len(realization_candidates) == 1:
            realization = realization_candidates[0]
            expected_intent = _text(event.get("intent"))
            actual_intent = _text(realization.get("intent"))
            if expected_intent != actual_intent:
                mismatches.append(
                    {
                        "event_id": event_id,
                        "source": "event_realization",
                        "field": "intent",
                        "expected": expected_intent,
                        "actual": actual_intent,
                    }
                )
                erroneous_event_ids.add(event_id)
            for field in ("dispatch_tick", "result_tick", "evidence_tick"):
                if not isinstance(realization.get(field), int):
                    mismatches.append(
                        {
                            "event_id": event_id,
                            "source": "event_realization",
                            "field": field,
                            "expected": "integer",
                            "actual": type(realization.get(field)).__name__,
                        }
                    )
                    erroneous_event_ids.add(event_id)
                elif not 0 <= int(realization[field]) <= 900:
                    mismatches.append(
                        {
                            "event_id": event_id,
                            "source": "event_realization",
                            "field": field,
                            "expected": "integer in tick range 0..900",
                            "actual": realization[field],
                        }
                    )
                    erroneous_event_ids.add(event_id)
            if (
                isinstance(realization.get("dispatch_tick"), int)
                and int(realization["dispatch_tick"]) % 5 != 0
            ):
                mismatches.append(
                    {
                        "event_id": event_id,
                        "source": "event_realization",
                        "field": "dispatch_tick",
                        "expected": "5-tick capture grid",
                        "actual": realization["dispatch_tick"],
                    }
                )
                erroneous_event_ids.add(event_id)
            actions = realization.get("action_realizations")
            if not isinstance(actions, list):
                mismatches.append(
                    {
                        "event_id": event_id,
                        "source": "event_realization",
                        "field": "action_realizations",
                        "expected": "array",
                        "actual": type(actions).__name__,
                    }
                )
                erroneous_event_ids.add(event_id)
            else:
                failed = [action for action in actions if action.get("status") != "ok"]
                if failed:
                    action_failure_count += len(failed)
                    mismatches.append(
                        {
                            "event_id": event_id,
                            "source": "event_realization",
                            "field": "action_realizations.status",
                            "expected": "ok",
                            "actual": "non_ok",
                        }
                    )
                    erroneous_event_ids.add(event_id)
        if len(trace_candidates) == 1 and len(realization_candidates) == 1:
            trace_tick = trace_candidates[0].get("tick")
            dispatch_tick = realization_candidates[0].get("dispatch_tick")
            if not isinstance(trace_tick, int) or trace_tick != dispatch_tick:
                mismatches.append(
                    {
                        "event_id": event_id,
                        "source": "runtime_join",
                        "field": "trace_tick_equals_dispatch_tick",
                        "expected": dispatch_tick,
                        "actual": trace_tick,
                    }
                )
                erroneous_event_ids.add(event_id)

    missing_trace = sorted(
        event_id for event_id in expected_ids if not trace_index.get(event_id)
    )
    missing_realization = sorted(
        event_id for event_id in expected_ids if not realization_index.get(event_id)
    )
    duplicate_trace = {
        event_id: len(rows)
        for event_id, rows in sorted(trace_index.items())
        if event_id in expected_ids and len(rows) != 1
    }
    duplicate_realization = {
        event_id: len(rows)
        for event_id, rows in sorted(realization_index.items())
        if event_id in expected_ids and len(rows) != 1
    }
    extra_trace = sorted(set(trace_index) - expected_ids)
    extra_realization = sorted(set(realization_index) - expected_ids)
    runtime_ok = not any(
        (
            missing_trace,
            missing_realization,
            duplicate_trace,
            duplicate_realization,
            extra_trace,
            extra_realization,
            unidentified_trace,
            unidentified_realization,
            mismatches,
        )
    )
    audit = {
        "declared_event_count": len(events),
        "active_trace_row_count": sum(len(rows) for rows in trace_index.values()),
        "realization_row_count": sum(len(rows) for rows in realization_index.values()),
        "missing_trace_event_ids": missing_trace,
        "missing_realization_event_ids": missing_realization,
        "duplicate_trace_event_ids": duplicate_trace,
        "duplicate_realization_event_ids": duplicate_realization,
        "extra_trace_event_ids": extra_trace,
        "extra_realization_event_ids": extra_realization,
        "unidentified_trace_row_indexes": unidentified_trace,
        "unidentified_realization_row_indexes": unidentified_realization,
        "declaration_mismatches": mismatches,
        "action_failure_count": action_failure_count,
        "status": "PASS" if runtime_ok else "FAIL",
    }
    return audit, trace_index, realization_index, erroneous_event_ids


def _one_runtime_row(
    index: Mapping[str, Sequence[Mapping[str, Any]]], event_id: str
) -> Mapping[str, Any] | None:
    rows = index.get(event_id, ())
    return rows[0] if len(rows) == 1 else None


def _runtime_ref(base_ref: str, event_id: str) -> str:
    return f"{base_ref}#event_id={event_id}"


def _scheduled_root_provenance(
    *,
    target: Mapping[str, Any],
    source_ids: Sequence[str],
    trace_index: Mapping[str, Sequence[Mapping[str, Any]]],
    realization_index: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    target_id = _text(target.get("event_id"))
    target_row = _one_runtime_row(realization_index, target_id)
    target_trace = _one_runtime_row(trace_index, target_id)
    source_rows = [
        _one_runtime_row(realization_index, source_id) for source_id in source_ids
    ]
    source_traces = [
        _one_runtime_row(trace_index, source_id) for source_id in source_ids
    ]
    target_tick = (
        target_row.get("dispatch_tick")
        if target_row is not None and isinstance(target_row.get("dispatch_tick"), int)
        else None
    )
    source_ticks = [
        int(row["dispatch_tick"])
        for row in source_rows
        if row is not None and isinstance(row.get("dispatch_tick"), int)
    ]
    anchor_present = (
        target_tick is not None
        and target_trace is not None
        and bool(source_ids)
        and len(source_ticks) == len(source_ids)
        and all(trace is not None for trace in source_traces)
        and any(source_tick <= target_tick for source_tick in source_ticks)
    )
    return {
        "kind": "script_scheduled",
        "trigger_type": "tick",
        "trigger_ref": _text(target.get("trigger_ref")),
        "declared_predecessor_intent": _text(
            target.get("causal_predecessor_intent")
        ),
        "schedule_anchor": "present" if anchor_present else "missing",
    }


def _runtime_alignment(
    *,
    target: Mapping[str, Any],
    source_ids: Sequence[str],
    resolution: Mapping[str, Any],
    trace_index: Mapping[str, Sequence[Mapping[str, Any]]],
    realization_index: Mapping[str, Sequence[Mapping[str, Any]]],
    erroneous_event_ids: set[str],
    realization_ref: str,
) -> tuple[dict[str, Any], str, bool]:
    target_id = _text(target.get("event_id"))
    target_row = _one_runtime_row(realization_index, target_id)
    target_trace = _one_runtime_row(trace_index, target_id)
    source_rows = [
        _one_runtime_row(realization_index, source_id) for source_id in source_ids
    ]
    source_traces = [
        _one_runtime_row(trace_index, source_id) for source_id in source_ids
    ]
    target_tick = (
        target_row.get("dispatch_tick")
        if target_row is not None and isinstance(target_row.get("dispatch_tick"), int)
        else None
    )
    source_ticks = [
        int(row["dispatch_tick"])
        for row in source_rows
        if row is not None and isinstance(row.get("dispatch_tick"), int)
    ]
    deltas = (
        [target_tick - source_tick for source_tick in source_ticks]
        if target_tick is not None
        else []
    )
    source_status = str(resolution.get("status"))
    cardinality = str(resolution.get("cardinality"))
    if _text(target.get("causal_predecessor_intent")) == "context":
        policy = "any_anchor_at_or_before_target"
    elif cardinality == "unspecified":
        policy = "all_candidates_strictly_prior"
    elif target.get("causal_temporal_policy") == "same_tick_required":
        policy = "same_tick_required"
    else:
        policy = "strict_before"

    if target_row is None or target_trace is None:
        status = "target_not_fired"
    elif not source_ids:
        status = "source_declaration_missing"
    elif any(row is None for row in source_rows) or any(
        row is None for row in source_traces
    ):
        status = "source_not_fired"
    elif resolution.get("method") == "context_anchor":
        status = (
            "context_present"
            if any(source_tick <= target_tick for source_tick in source_ticks)
            else "context_missing"
        )
    elif any(source_tick > target_tick for source_tick in source_ticks):
        status = "inverted"
    elif any(source_tick == target_tick for source_tick in source_ticks):
        status = "co_triggered"
    elif resolution.get("method") == "intent_stage_set":
        status = "all_candidates_strictly_prior"
    else:
        status = "aligned_strict"

    declaration_consistent = not ({target_id, *source_ids} & erroneous_event_ids)
    if policy == "same_tick_required" and status != "co_triggered":
        declaration_consistent = False
    alignment = {
        "source_dispatch_ticks": source_ticks,
        "target_dispatch_tick": target_tick,
        "delta_ticks_min": min(deltas) if deltas else None,
        "delta_ticks_max": max(deltas) if deltas else None,
        "policy": policy,
        "status": status,
        "declaration_consistent": declaration_consistent,
        "source_realization_refs": [
            _runtime_ref(realization_ref, source_id)
            for source_id, row in zip(source_ids, source_rows)
            if row is not None
        ],
        "target_realization_ref": (
            _runtime_ref(realization_ref, target_id) if target_row is not None else None
        ),
    }
    runtime_strict = (
        source_status == "resolved_unique"
        and status == "aligned_strict"
        and declaration_consistent
    )
    if source_status in {
        "source_declaration_missing",
        "requires_cardinality_declaration",
    }:
        validation_status = "runtime_fail_source_resolution"
    elif target_row is None or target_trace is None:
        validation_status = "runtime_fail_target_not_fired"
    elif any(row is None for row in source_rows) or any(
        row is None for row in source_traces
    ):
        validation_status = "runtime_fail_source_not_fired"
    elif not declaration_consistent:
        validation_status = "runtime_fail_declaration_mismatch"
    elif status == "inverted":
        validation_status = "runtime_fail_inverted"
    elif status == "context_missing":
        validation_status = "runtime_fail_context_missing"
    elif status == "co_triggered" and policy != "same_tick_required":
        validation_status = "runtime_fail_co_triggered_not_allowed"
    elif status == "context_present":
        validation_status = "runtime_pass_context"
    elif status == "co_triggered":
        validation_status = "runtime_pass_co_triggered"
    elif source_status == "resolved_declared_set":
        validation_status = "runtime_pass_declared_set"
    else:
        validation_status = "runtime_pass"
    return alignment, validation_status, runtime_strict


def _record_digest(record: Mapping[str, Any], field: str) -> str:
    payload = dict(record)
    payload.pop(field, None)
    return digest_object(payload)


def build_event_chain(
    *,
    episode_id: str,
    epi_id: str,
    script: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
    realization_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> EventChainBuildResult:
    """Build a complete Stage-A audit projection from already loaded inputs."""

    if not _text(episode_id) or not _text(epi_id):
        raise EventChainStructureError("episode_id and epi_id must be non-empty")
    events, triggers, chain_id = _normalized_script(script, epi_id=epi_id)
    event_script_digest = _digest_field(provenance, "event_script_digest")
    event_trace_digest = _digest_field(provenance, "event_trace_digest")
    event_realization_digest = _digest_field(provenance, "event_realization_digest")
    event_script_ref = _require_text(
        provenance, "event_script_ref", context="input provenance"
    )
    event_trace_ref = _require_text(
        provenance, "event_trace_ref", context="input provenance"
    )
    event_realization_ref = _require_text(
        provenance, "event_realization_ref", context="input provenance"
    )
    source_seal = provenance.get("source_manifest_event_script_digest")
    if source_seal is not None:
        source_seal = _text(source_seal)
    digest_authority = "consumed_snapshot_unsealed"

    audit, trace_index, realization_index, erroneous_ids = _runtime_audit(
        events, trace_rows, realization_rows
    )

    relations: list[dict[str, Any]] = []
    scheduled_root_provenance: dict[str, dict[str, Any]] = {}
    for target_index, target in enumerate(events):
        predecessor = _text(target.get("causal_predecessor_intent"))
        if not predecessor:
            continue
        target_id = _text(target.get("event_id"))
        source_endpoint, resolution, source_ids = resolve_source_endpoint(
            events, triggers, target_index
        )
        trigger = triggers[_text(target.get("trigger_ref"))]
        scheduled_only = trigger.get("type") == "tick" and predecessor != "context"
        if scheduled_only:
            scheduled_root_provenance[target_id] = _scheduled_root_provenance(
                target=target,
                source_ids=source_ids,
                trace_index=trace_index,
                realization_index=realization_index,
            )
            continue
        alignment, validation_status, runtime_strict = _runtime_alignment(
            target=target,
            source_ids=source_ids,
            resolution=resolution,
            trace_index=trace_index,
            realization_index=realization_index,
            erroneous_event_ids=erroneous_ids,
            realization_ref=event_realization_ref,
        )
        if predecessor == "context":
            relation_type = "script_declared_context"
        else:
            relation_type = "script_declared_causal"
        relation: dict[str, Any] = {
            "schema_name": "event_chain_relation",
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "epi_id": epi_id,
            "causal_chain_id": chain_id,
            "relation_id": stable_identifier(
                "script_chain_relation", episode_id, chain_id, target_id
            ),
            "relation_type": relation_type,
            "declared_predecessor_intent": predecessor,
            "source_endpoint": source_endpoint,
            "target_endpoint": _script_event_endpoint(target),
            "source_resolution": resolution,
            "runtime_alignment": alignment,
            "validation_status": validation_status,
            "runtime_strict_edge_eligible": runtime_strict,
            "provenance": {
                "event_script_ref": f"{event_script_ref}#event={target_id}",
                "event_script_digest": event_script_digest,
                "event_trace_digest": event_trace_digest,
                "event_realization_digest": event_realization_digest,
            },
        }
        relation["record_digest"] = _record_digest(relation, "record_digest")
        relations.append(relation)

    target_ids = {
        _text(relation["target_endpoint"].get("event_id")) for relation in relations
    }
    roots: list[dict[str, Any]] = []
    for root in events:
        root_id = _text(root.get("event_id"))
        if root_id in target_ids:
            continue
        root_trace = _one_runtime_row(trace_index, root_id)
        root_realization = _one_runtime_row(realization_index, root_id)
        root_dispatch_tick = (
            root_realization.get("dispatch_tick")
            if root_realization is not None
            and isinstance(root_realization.get("dispatch_tick"), int)
            else None
        )
        root_status = (
            "fired"
            if root_trace is not None
            and root_realization is not None
            and root_id not in erroneous_ids
            else "runtime_invalid"
        )
        roots.append(
            {
                **_script_event_endpoint(root),
                "dispatch_tick": root_dispatch_tick,
                "runtime_status": root_status,
                "realization_ref": (
                    _runtime_ref(event_realization_ref, root_id)
                    if root_realization is not None
                    else None
                ),
                "scheduling_provenance": scheduled_root_provenance.get(
                    root_id, {"kind": "none_declared"}
                ),
            }
        )
    script_scheduled_root_count = sum(
        root["scheduling_provenance"]["kind"] == "script_scheduled"
        for root in roots
    )
    # Exogenous roots are exactly script-scheduled roots plus roots with no
    # declared predecessor; neither category contributes an incoming relation.
    exogenous_root_count = sum(
        root["scheduling_provenance"]["kind"]
        in {"script_scheduled", "none_declared"}
        for root in roots
    )
    source_resolution_counts = Counter(
        str(row["source_resolution"]["method"]) for row in relations
    )
    runtime_alignment_counts = Counter(
        str(row["runtime_alignment"]["status"]) for row in relations
    )
    validation_status_counts = Counter(
        str(row["validation_status"]) for row in relations
    )
    runtime_relation_pass = all(
        row["validation_status"] in RUNTIME_PASS_VALIDATION_STATUSES
        for row in relations
    )
    manifest: dict[str, Any] = {
        "schema_name": "event_chain_manifest",
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "epi_id": epi_id,
        "causal_chain_id": chain_id,
        "node_count": len(events),
        "relation_count": len(relations),
        "root_count": len(roots),
        "roots": roots,
        "runtime_audit": audit,
        "summary": {
            "source_resolution_counts": dict(sorted(source_resolution_counts.items())),
            "runtime_alignment_counts": dict(sorted(runtime_alignment_counts.items())),
            "validation_status_counts": dict(sorted(validation_status_counts.items())),
            "runtime_strict_edge_eligible_count": sum(
                bool(row["runtime_strict_edge_eligible"]) for row in relations
            ),
            "script_scheduled_root_count": script_scheduled_root_count,
            "exogenous_root_count": exogenous_root_count,
        },
        "input_provenance": {
            "event_script_ref": event_script_ref,
            "event_script_digest": event_script_digest,
            "event_script_digest_authority": digest_authority,
            "source_manifest_event_script_digest": source_seal,
            "event_trace_ref": event_trace_ref,
            "event_trace_digest": event_trace_digest,
            "event_realization_ref": event_realization_ref,
            "event_realization_digest": event_realization_digest,
        },
        "runtime_validation_status": (
            "PASS" if audit["status"] == "PASS" and runtime_relation_pass else "FAIL"
        ),
    }
    manifest["manifest_digest"] = _record_digest(manifest, "manifest_digest")
    errors = validate_event_chain_artifacts(manifest, relations)
    if errors:
        raise EventChainStructureError(
            "constructed event chain violates its own contract: " + "; ".join(errors)
        )
    return EventChainBuildResult(manifest, tuple(relations))


def _histogram(
    rows: Iterable[Mapping[str, Any]], path: tuple[str, ...]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _source_candidate_ids(source: Any) -> tuple[str, ...]:
    if not isinstance(source, Mapping):
        return ()
    candidates = source.get("candidate_event_ids")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return ()
    return tuple(_text(value) for value in candidates if _text(value))


def _stored_runtime_ticks(
    alignment: Mapping[str, Any],
) -> tuple[list[int], int | None, list[str]]:
    errors: list[str] = []
    source_ticks_raw = alignment.get("source_dispatch_ticks")
    if not isinstance(source_ticks_raw, list):
        errors.append("runtime_alignment.source_dispatch_ticks must be an array")
        source_ticks: list[int] = []
    else:
        source_ticks = []
        for index, value in enumerate(source_ticks_raw):
            if isinstance(value, int):
                source_ticks.append(value)
            else:
                errors.append(
                    "runtime_alignment.source_dispatch_ticks"
                    f"[{index}] must be an integer"
                )
    target_tick_raw = alignment.get("target_dispatch_tick")
    if target_tick_raw is None or isinstance(target_tick_raw, int):
        target_tick = target_tick_raw
    else:
        errors.append("runtime_alignment.target_dispatch_tick must be integer or null")
        target_tick = None
    return source_ticks, target_tick, errors


def _event_ids_from_audit_field(audit: Mapping[str, Any], field: str) -> set[str]:
    value = audit.get(field)
    if isinstance(value, Mapping):
        return {_text(key) for key in value if _text(key)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {_text(item) for item in value if _text(item)}
    return set()


def _runtime_unavailable_event_ids(audit: Mapping[str, Any]) -> set[str]:
    unavailable: set[str] = set()
    for field in (
        "missing_trace_event_ids",
        "missing_realization_event_ids",
        "duplicate_trace_event_ids",
        "duplicate_realization_event_ids",
    ):
        unavailable.update(_event_ids_from_audit_field(audit, field))
    return unavailable


def _recompute_stored_runtime_projection(
    relation: Mapping[str, Any], *, runtime_audit: Mapping[str, Any]
) -> tuple[dict[str, Any], str, bool, list[str]]:
    """Recompute artifact runtime classification from persisted audit fields."""

    errors: list[str] = []
    source = relation.get("source_endpoint")
    target = relation.get("target_endpoint")
    resolution = relation.get("source_resolution")
    alignment = relation.get("runtime_alignment")
    if not isinstance(target, Mapping):
        errors.append("target_endpoint must be an object")
        target = {}
    if not isinstance(resolution, Mapping):
        errors.append("source_resolution must be an object")
        resolution = {}
    if not isinstance(alignment, Mapping):
        errors.append("runtime_alignment must be an object")
        alignment = {}

    source_ids = _source_candidate_ids(source)
    source_ticks, target_tick, tick_errors = _stored_runtime_ticks(alignment)
    errors.extend(tick_errors)
    unavailable_ids = _runtime_unavailable_event_ids(runtime_audit)
    target_id = _text(target.get("event_id"))
    target_unavailable = target_tick is None or target_id in unavailable_ids
    source_unavailable = len(source_ticks) != len(source_ids) or any(
        source_id in unavailable_ids for source_id in source_ids
    )
    deltas = (
        [target_tick - source_tick for source_tick in source_ticks]
        if target_tick is not None
        else []
    )
    source_status = str(resolution.get("status"))
    cardinality = str(resolution.get("cardinality"))
    method = str(resolution.get("method"))
    policy = str(alignment.get("policy"))
    declaration_consistent = alignment.get("declaration_consistent") is True

    if relation.get("relation_type") == "script_declared_context":
        expected_policy = "any_anchor_at_or_before_target"
    elif cardinality == "unspecified":
        expected_policy = "all_candidates_strictly_prior"
    elif policy == "same_tick_required":
        expected_policy = "same_tick_required"
    else:
        expected_policy = "strict_before"
    if policy != expected_policy:
        errors.append(
            f"runtime_alignment.policy {policy!r} differs from recomputed "
            f"{expected_policy!r}"
        )

    if target_unavailable:
        status = "target_not_fired"
    elif not source_ids:
        status = "source_declaration_missing"
    elif source_unavailable:
        status = "source_not_fired"
    elif method == "context_anchor":
        status = (
            "context_present"
            if any(source_tick <= target_tick for source_tick in source_ticks)
            else "context_missing"
        )
    elif any(source_tick > target_tick for source_tick in source_ticks):
        status = "inverted"
    elif any(source_tick == target_tick for source_tick in source_ticks):
        status = "co_triggered"
    elif method == "intent_stage_set":
        status = "all_candidates_strictly_prior"
    else:
        status = "aligned_strict"

    runtime_strict = (
        source_status == "resolved_unique"
        and status == "aligned_strict"
        and declaration_consistent
    )
    if source_status in {
        "source_declaration_missing",
        "requires_cardinality_declaration",
    }:
        validation_status = "runtime_fail_source_resolution"
    elif target_unavailable:
        validation_status = "runtime_fail_target_not_fired"
    elif source_unavailable:
        validation_status = "runtime_fail_source_not_fired"
    elif not declaration_consistent:
        validation_status = "runtime_fail_declaration_mismatch"
    elif status == "inverted":
        validation_status = "runtime_fail_inverted"
    elif status == "context_missing":
        validation_status = "runtime_fail_context_missing"
    elif status == "co_triggered" and policy != "same_tick_required":
        validation_status = "runtime_fail_co_triggered_not_allowed"
    elif status == "context_present":
        validation_status = "runtime_pass_context"
    elif status == "co_triggered":
        validation_status = "runtime_pass_co_triggered"
    elif source_status == "resolved_declared_set":
        validation_status = "runtime_pass_declared_set"
    else:
        validation_status = "runtime_pass"

    recomputed_alignment = {
        "source_dispatch_ticks": source_ticks,
        "target_dispatch_tick": target_tick,
        "delta_ticks_min": min(deltas) if deltas else None,
        "delta_ticks_max": max(deltas) if deltas else None,
        "policy": policy,
        "status": status,
        "declaration_consistent": declaration_consistent,
    }
    return recomputed_alignment, validation_status, runtime_strict, errors


def validate_event_chain_artifacts(
    manifest: Mapping[str, Any], relations: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Validate cross-record invariants not expressible in JSON Schema."""

    errors: list[str] = []
    episode_id = _text(manifest.get("episode_id"))
    epi_id = _text(manifest.get("epi_id"))
    chain_id = _text(manifest.get("causal_chain_id"))
    relation_ids: set[str] = set()
    target_ids: set[str] = set()
    node_ids: set[str] = set()
    roots = manifest.get("roots")
    if not isinstance(roots, list) or not roots:
        errors.append("manifest must contain at least one root descriptor")
        roots = []
    root_ids: set[str] = set()
    for root in roots:
        if isinstance(root, Mapping):
            root_id = _text(root.get("event_id"))
            if root_id in root_ids:
                errors.append(f"manifest contains duplicate root event {root_id}")
            root_ids.add(root_id)
            node_ids.add(root_id)
        else:
            errors.append("manifest root descriptor must be an object")
    for index, relation in enumerate(relations):
        location = f"relations[{index}]"
        if relation.get("relation_type") not in RELATION_TYPES:
            errors.append(f"{location}: invalid authored relation type")
        if (
            relation.get("episode_id") != episode_id
            or relation.get("epi_id") != epi_id
            or relation.get("causal_chain_id") != chain_id
        ):
            errors.append(f"{location}: manifest identity mismatch")
        relation_id = _text(relation.get("relation_id"))
        if relation_id in relation_ids:
            errors.append(f"{location}: duplicate relation_id {relation_id}")
        relation_ids.add(relation_id)
        target = relation.get("target_endpoint")
        if not isinstance(target, Mapping):
            errors.append(f"{location}: target endpoint is absent")
            continue
        target_id = _text(target.get("event_id"))
        if target_id in target_ids:
            errors.append(f"{location}: duplicate target event {target_id}")
        target_ids.add(target_id)
        node_ids.add(target_id)
        source = relation.get("source_endpoint")
        if isinstance(source, Mapping):
            kind = source.get("kind")
            if kind is not None and kind not in SOURCE_KINDS:
                errors.append(f"{location}: invalid source endpoint kind")
            node_ids.update(
                _text(value) for value in source.get("candidate_event_ids", ())
            )
        runtime_strict = relation.get("runtime_strict_edge_eligible") is True
        if runtime_strict and not (
            relation.get("source_resolution", {}).get("status") == "resolved_unique"
            and relation.get("runtime_alignment", {}).get("status") == "aligned_strict"
            and relation.get("runtime_alignment", {}).get("declaration_consistent")
            is True
        ):
            errors.append(f"{location}: invalid runtime strict-edge eligibility")
        if relation.get("relation_type") == "script_declared_context":
            missing_context_source = (
                isinstance(source, Mapping)
                and not source
                and relation.get("source_resolution", {}).get("status")
                == "source_declaration_missing"
            )
            resolved_context_source = (
                isinstance(source, Mapping)
                and source.get("kind") == "context_anchor"
                and relation.get("source_resolution", {}).get("status")
                == "resolved_context"
            )
            if not (missing_context_source or resolved_context_source) or (
                relation.get("runtime_alignment", {}).get("policy")
                != "any_anchor_at_or_before_target"
            ):
                errors.append(
                    f"{location}: context relation has a non-context source or policy"
                )
        if (
            relation.get("runtime_alignment", {}).get("status") == "co_triggered"
            and runtime_strict
        ):
            errors.append(
                f"{location}: co-triggered relation entered strict runtime view"
            )
        if (
            relation.get("source_resolution", {}).get("status")
            == "requires_cardinality_declaration"
            and runtime_strict
        ):
            errors.append(
                f"{location}: unresolved stage set entered strict runtime view"
            )
        runtime_audit = manifest.get("runtime_audit")
        if not isinstance(runtime_audit, Mapping):
            runtime_audit = {}
        (
            recomputed_alignment,
            validation_status,
            recomputed_runtime_strict,
            recompute_errors,
        ) = _recompute_stored_runtime_projection(relation, runtime_audit=runtime_audit)
        errors.extend(f"{location}: {error}" for error in recompute_errors)
        alignment = relation.get("runtime_alignment")
        if isinstance(alignment, Mapping):
            for field, expected in recomputed_alignment.items():
                if alignment.get(field) != expected:
                    errors.append(
                        f"{location}: runtime_alignment.{field} differs from "
                        f"recomputed {expected!r}"
                    )
        if relation.get("validation_status") != validation_status:
            errors.append(
                f"{location}: validation_status differs from recomputed "
                f"{validation_status!r}"
            )
        if runtime_strict != recomputed_runtime_strict:
            errors.append(
                f"{location}: runtime_strict_edge_eligible differs from recomputed "
                f"{recomputed_runtime_strict!r}"
            )

    if manifest.get("relation_count") != len(relations):
        errors.append("manifest relation_count differs from relation rows")
    if manifest.get("root_count") != len(roots):
        errors.append("manifest root_count differs from root descriptors")
    if root_ids & target_ids:
        errors.append("one or more roots have an incoming relation")
    if manifest.get("node_count") != len(node_ids):
        errors.append(
            f"manifest node_count differs from reconstructed nodes: "
            f"{manifest.get('node_count')} != {len(node_ids)}"
        )
    if node_ids - target_ids != root_ids:
        errors.append("manifest roots differ from nodes without incoming relations")
    if set(node_ids) != set(target_ids) | root_ids:
        errors.append(
            "one or more source candidates are not declared root/target nodes"
        )
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping):
        errors.append("manifest summary is absent")
    else:
        if summary.get("source_resolution_counts") != _histogram(
            relations, ("source_resolution", "method")
        ):
            errors.append("source-resolution histogram mismatch")
        if summary.get("runtime_alignment_counts") != _histogram(
            relations, ("runtime_alignment", "status")
        ):
            errors.append("runtime-alignment histogram mismatch")
        if summary.get("validation_status_counts") != _histogram(
            relations, ("validation_status",)
        ):
            errors.append("validation-status histogram mismatch")
        if summary.get("runtime_strict_edge_eligible_count") != sum(
            relation.get("runtime_strict_edge_eligible") is True
            for relation in relations
        ):
            errors.append("runtime strict-edge count mismatch")
        script_scheduled_root_count = sum(
            isinstance(root, Mapping)
            and isinstance(root.get("scheduling_provenance"), Mapping)
            and root["scheduling_provenance"].get("kind") == "script_scheduled"
            for root in roots
        )
        # Exogenous roots use only the two non-causal scheduling provenance kinds.
        exogenous_root_count = sum(
            isinstance(root, Mapping)
            and isinstance(root.get("scheduling_provenance"), Mapping)
            and root["scheduling_provenance"].get("kind")
            in {"script_scheduled", "none_declared"}
            for root in roots
        )
        if summary.get("script_scheduled_root_count") != script_scheduled_root_count:
            errors.append("script-scheduled root count mismatch")
        if summary.get("exogenous_root_count") != exogenous_root_count:
            errors.append("exogenous root count mismatch")
    return errors


__all__ = [
    "EventChainBuildResult",
    "EventChainError",
    "EventChainStructureError",
    "RUNTIME_PASS_VALIDATION_STATUSES",
    "SCHEMA_VERSION",
    "build_event_chain",
    "canonical_json",
    "digest_object",
    "resolve_source_endpoint",
    "sha256_bytes",
    "stable_identifier",
    "validate_event_chain_artifacts",
]
