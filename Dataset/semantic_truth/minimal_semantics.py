#!/usr/bin/env python3
"""Ontology-authoritative L2 event predicate and occurrence engine.

The complete 79-predicate world matrix is evaluated by ``world_truth``.  This
engine consumes the smaller normalized tuple projection needed by objective
events, never authored event labels, and derives transitions and occurrences
from objective state changes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_ids


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    OUT_OF_SCOPE = "out_of_scope"


ALLOWED_SOURCE_CLASSES = {
    "engine_truth",
    "deterministic_derived",
    "simulated_derived",
    "visual_derived",
}
FORBIDDEN_KEYS = {
    "active_event_ids",
    "confirmedAtTick",
    "dynamic_labels",
    "event_realization",
    "event_trace",
    "expected_event",
    "expected_event_id",
    "level",
    "scenario_plan",
    "scenario_title",
    "semantic_role",
    "state_facets",
    "task_id",
    "variant",
}
ENGINE_PREDICATE_VOCABULARY = frozenset(get_core_predicate_ids())


@dataclass(frozen=True)
class ObservationRow:
    episode_id: str
    tick: int
    tuple_id: str
    tuple_type: str
    participants: tuple[str, ...]
    scope: bool
    source_class: str
    source_refs: tuple[str, ...]
    mechanism_id: str | None
    values: Mapping[str, Any]


@dataclass(frozen=True)
class Assertion:
    episode_id: str
    tick: int
    tuple_id: str
    tuple_type: str
    participants: tuple[str, ...]
    predicate_id: str
    truth: Truth
    source_class: str
    source_refs: tuple[str, ...]
    mechanism_id: str | None
    missing_source_record: tuple[str, ...] = ()


@dataclass(frozen=True)
class Transition:
    episode_id: str
    tuple_id: str
    tuple_type: str
    participants: tuple[str, ...]
    predicate_id: str
    before_tick: int
    after_tick: int
    before_truth: Truth
    after_truth: Truth
    transition: str
    mechanism_id: str | None


@dataclass(frozen=True)
class SupportConditionProof:
    predicate_id: str
    desired_truth: Truth
    tuple_id: str
    tuple_type: str
    participants: tuple[str, ...]
    assertions: tuple[Assertion, ...]

    @property
    def start_tick(self) -> int:
        return self.assertions[0].tick

    @property
    def end_tick(self) -> int:
        return self.assertions[-1].tick


@dataclass(frozen=True)
class EventOccurrence:
    event_id: str
    episode_id: str
    event_family_id: str
    event_type: str
    rule_id: str
    event_subtype: str | None
    tuple_id: str
    tuple_type: str
    participants: tuple[str, ...]
    trigger_tick: int
    support_transition_tick: int
    detection_tick: int
    trigger_predicates: tuple[str, ...]
    mechanism_id: str | None
    support_condition_proofs: tuple[SupportConditionProof, ...] = ()
    source_kind: str = "objective"


@dataclass(frozen=True)
class EventOutcome:
    event_id: str
    episode_id: str
    event_family_id: str
    event_type: str
    tuple_id: str
    terminal_predicate_id: str
    trigger_tick: int
    terminal_tick: int | None
    terminal_transition: Transition | None
    terminal_hold_assertions: tuple[Assertion, ...]
    outcome: str
    reason: str


def _stable_id(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_KEYS & {str(key) for key in value}
        if forbidden:
            raise ValueError(
                f"forbidden truth inputs reached the engine: {sorted(forbidden)}"
            )
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _assert_no_forbidden_keys(child)


def parse_row(raw: Mapping[str, Any]) -> ObservationRow:
    _assert_no_forbidden_keys(raw)
    required = {
        "episode_id",
        "tick",
        "tuple_id",
        "tuple_type",
        "participants",
        "scope",
        "source_class",
        "source_refs",
        "values",
    }
    absent = sorted(required - set(raw))
    if absent:
        raise ValueError(f"normalized row is missing fields: {absent}")
    if not isinstance(raw["episode_id"], str) or not raw["episode_id"]:
        raise ValueError("episode_id must be a non-empty string")
    if not isinstance(raw["tick"], int) or raw["tick"] < 0:
        raise ValueError("tick must be a non-negative integer")
    if not isinstance(raw["tuple_id"], str) or not isinstance(raw["tuple_type"], str):
        raise ValueError("tuple_id and tuple_type must be strings")
    participants = raw["participants"]
    if not isinstance(participants, list) or not all(
        isinstance(participant, str) and participant for participant in participants
    ):
        raise ValueError("participants must be non-empty strings")
    if raw["source_class"] not in ALLOWED_SOURCE_CLASSES:
        raise ValueError(f"unsupported source_class: {raw['source_class']!r}")
    if not isinstance(raw["source_refs"], list) or not all(
        isinstance(ref, str) for ref in raw["source_refs"]
    ):
        raise ValueError("source_refs must be strings")
    if not isinstance(raw["values"], Mapping):
        raise ValueError("values must be an object")
    mechanism_id = raw.get("mechanism_id")
    if mechanism_id is not None and not isinstance(mechanism_id, str):
        raise ValueError("mechanism_id must be a string or null")
    return ObservationRow(
        episode_id=raw["episode_id"],
        tick=raw["tick"],
        tuple_id=raw["tuple_id"],
        tuple_type=raw["tuple_type"],
        participants=tuple(participants),
        scope=bool(raw["scope"]),
        source_class=raw["source_class"],
        source_refs=tuple(sorted(set(raw["source_refs"]))),
        mechanism_id=mechanism_id,
        values=dict(raw["values"]),
    )


def validate_rows(rows: Sequence[ObservationRow], params: Mapping[str, Any]) -> None:
    step = params.get("formal_step_ticks")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("formal_step_ticks must be a positive integer")
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        if row.tick % step:
            raise ValueError(f"non-authoritative tick {row.tick}")
        key = (row.episode_id, row.tick, row.tuple_id)
        if key in seen:
            raise ValueError(f"duplicate normalized tuple row: {key}")
        seen.add(key)


def build_transitions(
    assertions: Iterable[Assertion], params: Mapping[str, Any]
) -> list[Transition]:
    step = int(params["formal_step_ticks"])
    grouped: dict[tuple[str, str, str], list[Assertion]] = defaultdict(list)
    for assertion in assertions:
        grouped[
            (assertion.episode_id, assertion.tuple_id, assertion.predicate_id)
        ].append(assertion)
    result: list[Transition] = []
    for series in grouped.values():
        series.sort(key=lambda item: item.tick)
        for before, after in zip(series, series[1:]):
            if (
                before.tuple_type != after.tuple_type
                or before.participants != after.participants
            ):
                raise ValueError(
                    "predicate tuple identity changed across ticks: "
                    f"{after.episode_id} {after.predicate_id} {after.tuple_id} "
                    f"{before.tick}->{after.tick}"
                )
            if after.tick - before.tick != step:
                kind = "continuity_break"
            elif before.truth == Truth.FALSE and after.truth == Truth.TRUE:
                kind = "rising"
            elif before.truth == Truth.TRUE and after.truth == Truth.FALSE:
                kind = "falling"
            else:
                continue
            result.append(
                Transition(
                    episode_id=after.episode_id,
                    tuple_id=after.tuple_id,
                    tuple_type=after.tuple_type,
                    participants=after.participants,
                    predicate_id=after.predicate_id,
                    before_tick=before.tick,
                    after_tick=after.tick,
                    before_truth=before.truth,
                    after_truth=after.truth,
                    transition=kind,
                    mechanism_id=(
                        before.mechanism_id
                        if before.mechanism_id == after.mechanism_id
                        else None
                    ),
                )
            )
    result.sort(
        key=lambda item: (
            item.episode_id,
            item.after_tick,
            item.predicate_id,
            item.tuple_id,
        )
    )
    return result


def _event(
    transition: Transition,
    rule: Mapping[str, Any],
    triggers: Sequence[str],
    *,
    detection_tick: int | None = None,
    support_condition_proofs: Sequence[SupportConditionProof] = (),
) -> EventOccurrence:
    confirmed_tick = transition.after_tick if detection_tick is None else detection_tick
    if confirmed_tick < transition.after_tick:
        raise ValueError("event detection cannot precede its objective onset")
    event_type = str(rule["event_type"])
    return EventOccurrence(
        event_id="objective_event:"
        + _stable_id(
            transition.episode_id,
            rule["rule_id"],
            event_type,
            transition.tuple_id,
            transition.after_tick,
        ),
        episode_id=transition.episode_id,
        event_family_id=str(rule["event_family_id"]),
        event_type=event_type,
        rule_id=str(rule["rule_id"]),
        event_subtype=None,
        tuple_id=transition.tuple_id,
        tuple_type=transition.tuple_type,
        participants=transition.participants,
        trigger_tick=transition.after_tick,
        support_transition_tick=transition.after_tick,
        detection_tick=confirmed_tick,
        trigger_predicates=tuple(triggers),
        mechanism_id=transition.mechanism_id,
        support_condition_proofs=tuple(support_condition_proofs),
    )


ROLES_BY_TUPLE_TYPE = {
    "aircraft": ("aircraft",),
    "aircraft_home_pad": ("aircraft", "home_pad"),
    "aircraft_landing_zone": ("aircraft", "landing_zone"),
    "aircraft_restricted_region": ("aircraft", "restricted_region"),
    "aircraft_pair": ("aircraft_a", "aircraft_b"),
    "aircraft_corridor": ("aircraft", "corridor"),
    "corridor": ("corridor",),
}
GROUNDED_ROLE_TUPLE_PREFIX = "grounded_roles__"


def _tuple_roles(tuple_type: str) -> tuple[str, ...]:
    if tuple_type.startswith(GROUNDED_ROLE_TUPLE_PREFIX):
        roles = tuple(
            role
            for role in tuple_type.removeprefix(GROUNDED_ROLE_TUPLE_PREFIX).split("__")
            if role
        )
        if roles:
            return roles
    roles = ROLES_BY_TUPLE_TYPE.get(tuple_type)
    if roles is None:
        raise ValueError(f"unknown tuple role mapping for tuple_type={tuple_type!r}")
    return roles


def _tuple_bindings(tuple_type: str, participants: Sequence[str]) -> dict[str, str]:
    roles = _tuple_roles(tuple_type)
    if len(roles) != len(participants):
        raise ValueError(
            f"tuple role count mismatch for {tuple_type!r}: "
            f"roles={roles}, participants={participants}"
        )
    return {role: str(participant) for role, participant in zip(roles, participants)}


def _required_join_roles(
    value: Mapping[str, Any],
    context: str,
) -> tuple[str, ...]:
    join_roles = value.get("join_roles")
    if (
        not isinstance(join_roles, Sequence)
        or isinstance(join_roles, (str, bytes))
        or any(not isinstance(role, str) or not role for role in join_roles)
    ):
        raise ValueError(f"{context} must declare join_roles")
    if not join_roles:
        raise ValueError(f"{context} must declare non-empty join_roles")
    return tuple(str(role) for role in join_roles)


def _bindings_match_on_roles(
    left: Mapping[str, str],
    right: Mapping[str, str],
    roles: Sequence[str],
) -> bool:
    return all(
        role in left and role in right and left[role] == right[role] for role in roles
    )


def _support_condition_proof(
    condition: Mapping[str, Any],
    transition: Transition,
    assertions_by_predicate_tick: Mapping[tuple[str, int, str], Sequence[Assertion]],
) -> SupportConditionProof | None:
    """Return one exact-tuple proof satisfying an event support guard.

    ``before`` is the state immediately before the trigger transition. ``after``
    starts exactly at the objective onset sample. ``hold_samples`` defaults to
    one. Every sample in a multi-sample hold must belong to the same complete
    predicate tuple; sharing only the declared join roles is insufficient.
    """

    hold_samples = condition.get("hold_samples", 1)
    if not isinstance(hold_samples, int) or isinstance(hold_samples, bool):
        raise ValueError(
            f"event support condition hold_samples must be an integer: {condition!r}"
        )
    if hold_samples < 1:
        raise ValueError(
            f"event support condition hold_samples must be positive: {condition!r}"
        )
    at = condition.get("at")
    if at == "before":
        if hold_samples != 1:
            raise ValueError(
                "a before-anchored event support condition is one exact sample"
            )
        candidate_ticks = (transition.before_tick,)
    elif at == "after":
        step = transition.after_tick - transition.before_tick
        if step <= 0:
            raise ValueError("event trigger transition must advance time")
        candidate_ticks = tuple(
            transition.after_tick + offset * step for offset in range(hold_samples)
        )
    else:
        raise ValueError(f"event support condition has invalid temporal anchor: {at!r}")
    desired = Truth(str(condition.get("value")))
    predicate_id = str(condition["predicate_id"])
    join_roles = _required_join_roles(condition, "event support condition")
    trigger_bindings = _tuple_bindings(transition.tuple_type, transition.participants)
    exact_assertions: dict[
        tuple[int, str, str, tuple[tuple[str, str], ...]], Assertion
    ] = {}
    identities: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    for tick in candidate_ticks:
        for item in assertions_by_predicate_tick.get(
            (transition.episode_id, tick, predicate_id), ()
        ):
            item_bindings = _tuple_bindings(item.tuple_type, item.participants)
            if not _bindings_match_on_roles(
                trigger_bindings, item_bindings, join_roles
            ):
                continue
            binding_items = tuple(sorted(item_bindings.items()))
            identity = (item.tuple_id, item.tuple_type, binding_items)
            key = (tick, *identity)
            if key in exact_assertions:
                raise ValueError(
                    "duplicate support assertion for exact predicate tuple: "
                    f"{transition.episode_id} {predicate_id} {item.tuple_id} {tick}"
                )
            exact_assertions[key] = item
            identities.add(identity)

    proofs: list[SupportConditionProof] = []
    for tuple_id, tuple_type, binding_items in sorted(identities):
        run = tuple(
            exact_assertions.get((tick, tuple_id, tuple_type, binding_items))
            for tick in candidate_ticks
        )
        if any(item is None or item.truth != desired for item in run):
            continue
        proven_assertions = tuple(item for item in run if item is not None)
        proofs.append(
            SupportConditionProof(
                predicate_id=predicate_id,
                desired_truth=desired,
                tuple_id=tuple_id,
                tuple_type=tuple_type,
                participants=proven_assertions[0].participants,
                assertions=proven_assertions,
            )
        )
    if not proofs:
        return None
    earliest_by_identity: dict[
        tuple[str, str, tuple[str, ...]], SupportConditionProof
    ] = {}
    for proof in sorted(
        proofs,
        key=lambda item: (
            item.end_tick,
            item.start_tick,
            item.tuple_id,
            item.tuple_type,
            item.participants,
        ),
    ):
        identity = (proof.tuple_id, proof.tuple_type, proof.participants)
        earliest_by_identity.setdefault(identity, proof)
    if len(earliest_by_identity) != 1:
        raise ValueError(
            "event support condition resolves to multiple exact predicate tuples: "
            f"{transition.episode_id} {predicate_id} "
            f"{sorted(identity[0] for identity in earliest_by_identity)}"
        )
    return next(iter(earliest_by_identity.values()))


# Entity classification for the event layer lives in entity_scope.py, which reads the
# governed roster fields.  Predicate truth is computed for every in-scope entity;
# classification decides only whether an authored event may name one.
from Dataset.semantic_truth.entity_scope import (  # noqa: E402
    entity_scope_contract,
    is_background,
    participant_binding_allowed,
    role_is_actor,
)


def _rule_allows_transition(
    rule: Mapping[str, Any],
    transition: Transition,
    roster_by_id: Mapping[str, Any],
) -> bool:
    roles = rule.get("participant_roles")
    if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
        raise ValueError("event rule must declare participant_roles")
    if not any(
        isinstance(role, Mapping) and role.get("background_allowed") is False
        for role in roles
    ):
        return True
    role_names = [str(role["role"]) for role in roles if isinstance(role, Mapping)]
    return all(
        any(
            participant_binding_allowed(participant, name, roster_by_id)
            for name in role_names
        )
        for participant in transition.participants
    )


def build_events(
    rows: Sequence[ObservationRow],
    assertions: Sequence[Assertion],
    transitions: Sequence[Transition],
    event_rules: Sequence[Mapping[str, Any]],
    *,
    roster_by_id: Mapping[str, Any] | None = None,
) -> list[EventOccurrence]:
    del rows
    roster = roster_by_id if roster_by_id is not None else {}
    by_predicate_tick: dict[tuple[str, int, str], list[Assertion]] = defaultdict(list)
    for item in assertions:
        by_predicate_tick[(item.episode_id, item.tick, item.predicate_id)].append(item)
    rules_by_trigger: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    rule_ids: set[str] = set()
    for rule in event_rules:
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError(f"event rule lacks rule_id: {rule!r}")
        if rule_id in rule_ids:
            raise ValueError(f"duplicate event rule_id: {rule_id}")
        rule_ids.add(rule_id)
        predicate_id = rule.get("trigger_predicate_id")
        direction = rule.get("trigger_direction")
        if not isinstance(predicate_id, str) or direction not in {"rising", "falling"}:
            raise ValueError(f"invalid declarative event rule: {rule!r}")
        rules_by_trigger[(predicate_id, str(direction))].append(rule)
    events: list[EventOccurrence] = []
    for transition in transitions:
        for rule in rules_by_trigger.get(
            (transition.predicate_id, transition.transition), ()
        ):
            if not _rule_allows_transition(rule, transition, roster):
                continue
            conditions = rule.get("support_conditions", ())
            if not isinstance(conditions, Sequence) or isinstance(
                conditions, (str, bytes)
            ):
                raise ValueError(
                    f"event rule support_conditions must be an array: {rule!r}"
                )
            support_proofs: list[SupportConditionProof] = []
            for condition in conditions:
                if not isinstance(condition, Mapping):
                    raise ValueError(
                        f"event rule support condition must be an object: {rule!r}"
                    )
                support_proof = _support_condition_proof(
                    condition, transition, by_predicate_tick
                )
                if support_proof is None:
                    break
                support_proofs.append(support_proof)
            else:
                events.append(
                    _event(
                        transition,
                        rule,
                        (
                            transition.predicate_id,
                            *(
                                str(condition["predicate_id"])
                                for condition in conditions
                            ),
                        ),
                        detection_tick=max(
                            (
                                transition.after_tick,
                                *(proof.end_tick for proof in support_proofs),
                            )
                        ),
                        support_condition_proofs=support_proofs,
                    )
                )
                continue
    unique = {event.event_id: event for event in events}
    if len(unique) != len(events):
        raise ValueError("distinct event proofs produced a duplicate event_id")
    return sorted(
        unique.values(),
        key=lambda event: (
            event.episode_id,
            event.trigger_tick,
            event.event_type,
            event.event_id,
        ),
    )


@dataclass(frozen=True)
class _HoldProof:
    assertions: tuple[Assertion, ...]

    @property
    def start_tick(self) -> int:
        return self.assertions[0].tick

    @property
    def end_tick(self) -> int:
        return self.assertions[-1].tick

    @property
    def tuple_id(self) -> str:
        return self.assertions[0].tuple_id


def _hold_proofs(
    assertions: Sequence[Assertion],
    event: EventOccurrence,
    predicate_id: str,
    desired: Truth,
    samples: int,
    step: int,
    join_roles: Sequence[str],
    *,
    before_tick: int | None = None,
) -> tuple[_HoldProof, ...]:
    if samples <= 0:
        raise ValueError("terminal hold samples must be positive")
    if step <= 0:
        raise ValueError("terminal hold step must be positive")
    event_bindings = _tuple_bindings(event.tuple_type, event.participants)
    candidates = [
        item
        for item in assertions
        if item.episode_id == event.episode_id
        and item.predicate_id == predicate_id
        and item.tick >= event.support_transition_tick
        and (before_tick is None or item.tick < before_tick)
        and _bindings_match_on_roles(
            event_bindings,
            _tuple_bindings(item.tuple_type, item.participants),
            join_roles,
        )
    ]
    grouped: dict[tuple[str, str, tuple[tuple[str, str], ...]], list[Assertion]] = (
        defaultdict(list)
    )
    for item in candidates:
        bindings = tuple(
            sorted(_tuple_bindings(item.tuple_type, item.participants).items())
        )
        grouped[(item.tuple_id, item.tuple_type, bindings)].append(item)

    proofs: list[_HoldProof] = []
    for group_key, series in grouped.items():
        series.sort(key=lambda item: item.tick)
        ticks = [item.tick for item in series]
        if len(ticks) != len(set(ticks)):
            raise ValueError(
                "duplicate terminal assertion for exact predicate tuple: "
                f"{event.episode_id} {predicate_id} {group_key[0]}"
            )
        run: list[Assertion] = []
        for item in series:
            if item.truth != desired:
                run = []
                continue
            if run and item.tick != run[-1].tick + step:
                run = []
            run.append(item)
            if len(run) == samples:
                proofs.append(_HoldProof(tuple(run)))
                break
    return tuple(
        sorted(
            proofs,
            key=lambda proof: (
                proof.end_tick,
                proof.start_tick,
                proof.tuple_id,
                proof.assertions[0].tuple_type,
                proof.assertions[0].participants,
            ),
        )
    )


def _first_hold_proof(
    assertions: Sequence[Assertion],
    event: EventOccurrence,
    predicate_id: str,
    desired: Truth,
    samples: int,
    step: int,
    join_roles: Sequence[str],
    *,
    before_tick: int | None = None,
) -> _HoldProof | None:
    proofs = _hold_proofs(
        assertions,
        event,
        predicate_id,
        desired,
        samples,
        step,
        join_roles,
        before_tick=before_tick,
    )
    return proofs[0] if proofs else None


def _first_hold(
    assertions: Sequence[Assertion],
    event: EventOccurrence,
    predicate_id: str,
    desired: Truth,
    samples: int,
    step: int,
    join_roles: Sequence[str],
    *,
    before_tick: int | None = None,
) -> int | None:
    proof = _first_hold_proof(
        assertions,
        event,
        predicate_id,
        desired,
        samples,
        step,
        join_roles,
        before_tick=before_tick,
    )
    return None if proof is None else proof.end_tick


def _next_same_trigger_tuple_onsets(
    events: Sequence[EventOccurrence],
) -> dict[str, int]:
    grouped: dict[
        tuple[str, str, str, str, tuple[str, ...], tuple[str, ...]],
        list[EventOccurrence],
    ] = defaultdict(list)
    for event in events:
        grouped[
            (
                event.episode_id,
                event.rule_id,
                event.tuple_id,
                event.tuple_type,
                event.participants,
                event.trigger_predicates,
            )
        ].append(event)

    result: dict[str, int] = {}
    for group in grouped.values():
        group.sort(key=lambda item: (item.trigger_tick, item.event_id))
        for current, next_event in zip(group, group[1:]):
            if next_event.trigger_tick > current.trigger_tick:
                result[current.event_id] = next_event.trigger_tick
    return result


def _terminal_transition(
    transitions: Sequence[Transition],
    proof: _HoldProof,
    predicate_id: str,
    desired: Truth,
    step: int,
) -> Transition | None:
    first = proof.assertions[0]
    expected_before = Truth.FALSE if desired == Truth.TRUE else Truth.TRUE
    expected_direction = "rising" if desired == Truth.TRUE else "falling"
    matches = [
        transition
        for transition in transitions
        if transition.episode_id == first.episode_id
        and transition.predicate_id == predicate_id
        and transition.tuple_id == first.tuple_id
        and transition.tuple_type == first.tuple_type
        and transition.participants == first.participants
        and transition.before_tick == proof.start_tick - step
        and transition.after_tick == proof.start_tick
        and transition.before_truth == expected_before
        and transition.after_truth == desired
        and transition.transition == expected_direction
    ]
    if len(matches) > 1:
        raise ValueError(
            "multiple exact transitions establish one terminal hold: "
            f"{first.episode_id} {predicate_id} {first.tuple_id} {proof.start_tick}"
        )
    return matches[0] if matches else None


def build_outcomes(
    events: Sequence[EventOccurrence],
    assertions: Sequence[Assertion],
    params: Mapping[str, Any],
    event_rules: Sequence[Mapping[str, Any]],
    *,
    transitions: Sequence[Transition],
) -> list[EventOutcome]:
    step = int(params["formal_step_ticks"])
    rules_by_id = {str(rule["rule_id"]): rule for rule in event_rules}
    next_same_onset_by_event_id = _next_same_trigger_tuple_onsets(events)
    result: list[EventOutcome] = []
    for event in events:
        rule = rules_by_id[event.rule_id]
        terminal = rule.get("terminal")
        if not isinstance(terminal, Mapping):
            raise ValueError(f"event rule lacks a terminal expression: {event.rule_id}")
        predicate_id = str(terminal["predicate_id"])
        desired = Truth(str(terminal["value"]))
        hold_parameter = str(terminal["hold_parameter"])
        samples = int(params[hold_parameter])
        hold_proofs = _hold_proofs(
            assertions,
            event,
            predicate_id,
            desired,
            samples,
            step,
            _required_join_roles(
                terminal,
                "event terminal predicate",
            ),
            before_tick=next_same_onset_by_event_id.get(event.event_id),
        )
        terminal_proofs: list[tuple[_HoldProof, Transition]] = []
        for candidate_proof in hold_proofs:
            candidate_transition = _terminal_transition(
                transitions,
                candidate_proof,
                predicate_id,
                desired,
                step,
            )
            if candidate_transition is not None:
                terminal_proofs.append((candidate_proof, candidate_transition))
        terminal_identities = {
            (
                proof.tuple_id,
                proof.assertions[0].tuple_type,
                proof.assertions[0].participants,
            )
            for proof, _transition in terminal_proofs
        }
        if len(terminal_identities) > 1:
            raise ValueError(
                "event terminal resolves to multiple exact predicate tuples: "
                f"{event.event_id} {predicate_id} "
                f"{sorted(identity[0] for identity in terminal_identities)}"
            )
        if terminal_proofs:
            hold_proof, terminal_transition = terminal_proofs[0]
        else:
            hold_proof = None
            terminal_transition = None
        succeeded = hold_proof is not None
        terminal_tick = hold_proof.end_tick if succeeded else None
        result.append(
            EventOutcome(
                event_id=event.event_id,
                episode_id=event.episode_id,
                event_family_id=event.event_family_id,
                event_type=event.event_type,
                tuple_id=event.tuple_id,
                terminal_predicate_id=predicate_id,
                trigger_tick=event.trigger_tick,
                terminal_tick=terminal_tick,
                terminal_transition=terminal_transition if succeeded else None,
                terminal_hold_assertions=(
                    hold_proof.assertions
                    if succeeded and hold_proof is not None
                    else ()
                ),
                outcome="success" if succeeded else "pending",
                reason=(
                    "terminal_predicate_hold"
                    if desired == Truth.TRUE
                    else "recovery_predicate_hold"
                )
                if succeeded
                else (
                    "terminal_transition_not_observed"
                    if hold_proofs
                    else "terminal_hold_not_observed"
                ),
            )
        )
    return result


__all__ = [
    "ALLOWED_SOURCE_CLASSES",
    "ENGINE_PREDICATE_VOCABULARY",
    "Assertion",
    "EventOccurrence",
    "EventOutcome",
    "ObservationRow",
    "SupportConditionProof",
    "Transition",
    "Truth",
    "build_events",
    "build_outcomes",
    "build_transitions",
    "parse_row",
    "validate_rows",
]
