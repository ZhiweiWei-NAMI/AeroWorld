"""Normalize the V3 event projection and adapt the L2 semantic engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
    get_core_predicate_ids,
    get_core_predicate_templates,
    get_executable_predicate_ids,
    get_governed_parameter_defaults,
)
from Dataset.semantic_truth.facility_scope import ontology_class_is_a
from Dataset.semantic_truth.entity_scope import (
    participant_binding_allowed as _participant_binding_allowed,
)
from Dataset.semantic_truth.minimal_semantics import (
    ENGINE_PREDICATE_VOCABULARY,
    Assertion,
    EventOccurrence,
    EventOutcome,
    ObservationRow,
    Transition,
    Truth,
    build_events,
    build_outcomes,
    build_transitions,
    parse_row,
    validate_rows,
)
from Dataset.semantic_truth.model import TickContext
from Dataset.semantic_truth.provenance import digest_object, stable_identifier


SCHEMA_VERSION = "3.0.0"
PREDICATE_TRUTH_SCHEMA_VERSION = "3.1.0"
FORBIDDEN_INPUT_KEYS = {
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
NORMALIZED_TUPLE_TYPES = frozenset(
    {
        "aircraft",
        "aircraft_home_pad",
        "aircraft_landing_zone",
        "aircraft_restricted_region",
        "aircraft_pair",
        "aircraft_corridor",
        "corridor",
        "world_scope_uav",
        "world_scope_vehicle",
        "world_scope_pedestrian",
        "world_scope_facility",
        "world_scope_compute_node",
        "world_scope_scene",
    }
)
GROUNDED_ROLE_TUPLE_PREFIX = "grounded_roles__"


@dataclass(frozen=True)
class MinimalSemanticResult:
    normalized_rows: tuple[ObservationRow, ...]
    semantic_state: tuple[dict[str, Any], ...]
    predicate_truth: tuple[dict[str, Any], ...]
    transitions: tuple[dict[str, Any], ...]
    continuity_breaks: tuple[dict[str, Any], ...]
    occurrences: tuple[dict[str, Any], ...]
    outcomes: tuple[dict[str, Any], ...]
    projection: Mapping[str, Any]


class _StateIndex:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
        self.by_tick_family: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
        for row in rows:
            tick = row.get("tick")
            family = row.get("observation_family")
            subject = row.get("subject_id")
            if (
                not isinstance(tick, int)
                or not isinstance(family, str)
                or not isinstance(subject, str)
            ):
                continue
            key = (tick, family, subject)
            if key in self.by_key:
                raise ValueError(f"duplicate predicate-contract state row: {key}")
            self.by_key[key] = row
            self.by_tick_family.setdefault((tick, family), []).append(row)

    def get(self, tick: int, family: str, subject: str) -> Mapping[str, Any] | None:
        return self.by_key.get((tick, family, subject))

    def rows(self, tick: int, family: str) -> list[Mapping[str, Any]]:
        return list(self.by_tick_family.get((tick, family), ()))


def run_minimal_semantic_engine(
    contexts: Sequence[TickContext],
    domain_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    world_truth_base: Mapping[str, Any],
    world_truth_deltas: Sequence[Mapping[str, Any]],
    *,
    input_digest: str,
    stage_projection_predicate_ids: Sequence[str] = (),
    roster_by_id: Mapping[str, Any] | None = None,
) -> MinimalSemanticResult:
    """Normalize state contracts, evaluate predicates, and adapt output records."""

    params = get_governed_parameter_defaults()
    governed = world_truth_base.get("governed_parameters")
    if isinstance(governed, Mapping):
        params.update(governed)
    event_rules = get_core_event_occurrence_types()
    raw_rows = build_normalized_rows(
        contexts,
        domain_rows,
        communication_rows,
        params=params,
    )
    rows = [parse_row(row) for row in raw_rows]
    validate_rows(rows, params)
    assertions: list[Assertion] = []
    world_rows, world_assertions, world_lineage = _build_world_event_projection(
        world_truth_base,
        world_truth_deltas,
        event_rules,
        stage_projection_predicate_ids=stage_projection_predicate_ids,
    )
    rows.extend(world_rows)
    assertions.extend(world_assertions)
    rows.sort(
        key=lambda row: (
            row.episode_id,
            row.tick,
            row.tuple_type,
            row.tuple_id,
        )
    )
    assertions.sort(
        key=lambda item: (
            item.episode_id,
            item.tick,
            item.tuple_type,
            item.tuple_id,
            item.predicate_id,
        )
    )
    _assert_core_vocabulary(assertions)
    engine_transitions = build_transitions(assertions, params)
    events = build_events(
        rows,
        assertions,
        engine_transitions,
        event_rules,
        roster_by_id=roster_by_id,
    )
    outcomes = build_outcomes(
        events,
        assertions,
        params,
        event_rules,
        transitions=engine_transitions,
    )
    parameter_digest = digest_object(params)
    truth = _adapt_assertions(assertions, rows, world_lineage)
    truth_by_key = {
        (row["episode_id"], row["tick"], row["tuple_id"], row["predicate_id"]): row
        for row in truth
    }
    adapted_transitions, breaks = _adapt_transitions(engine_transitions, truth_by_key)
    transition_by_engine_key = {
        (
            row["episode_id"],
            row["to_tick"],
            row["tuple_id"],
            row["predicate_id"],
            row["direction"],
        ): row
        for row in adapted_transitions
    }
    occurrences = _adapt_occurrences(
        events,
        rows,
        assertions,
        transition_by_engine_key,
        event_rules,
        outcomes,
        roster_by_id=roster_by_id,
    )
    adapted_outcomes = _adapt_outcomes(
        outcomes,
        occurrences,
        truth_by_key,
        formal_step=int(params["formal_step_ticks"]),
    )
    semantic_state = tuple(
        _adapt_normalized_state(row, input_digest, parameter_digest) for row in rows
    )
    core_ids = list(get_core_predicate_ids())
    event_count = len(get_core_event_occurrence_types())
    projection = {
        "schema_name": "minimal_semantic_projection",
        "schema_version": SCHEMA_VERSION,
        "catalog_id": "aeroworld.ontology_predicates.v3",
        "catalog_version": "3.0.0",
        "episode_id": rows[0].episode_id if rows else "",
        "selected_api_ids": core_ids,
        "executable_predicate_ids": list(get_executable_predicate_ids()),
        "catalog_gate": {
            "passed": True,
            "template_count": len(core_ids),
            "event_occurrence_type_count": event_count,
        },
        "normalized_tuple_type_count": len({row.tuple_type for row in rows}),
        "normalized_tuple_types": sorted({row.tuple_type for row in rows}),
        "normalized_row_count": len(rows),
        "runtime_predicate_ids": sorted({item.predicate_id for item in assertions}),
    }
    return MinimalSemanticResult(
        normalized_rows=tuple(rows),
        semantic_state=semantic_state,
        predicate_truth=tuple(truth),
        transitions=tuple(adapted_transitions),
        continuity_breaks=tuple(breaks),
        occurrences=tuple(occurrences),
        outcomes=tuple(adapted_outcomes),
        projection=projection,
    )


def _build_world_event_projection(
    base: Mapping[str, Any],
    deltas: Sequence[Mapping[str, Any]],
    event_rules: Sequence[Mapping[str, Any]],
    *,
    stage_projection_predicate_ids: Sequence[str] = (),
) -> tuple[
    list[ObservationRow],
    list[Assertion],
    dict[tuple[str, int, str, str], tuple[str, str]],
]:
    """Expand registry-bound event and explicit stage predicates for L2."""

    if base.get("representation") != "grounded_predicate_candidate_base":
        raise ValueError("world event projection requires grounded L1 base")
    required_ids = {str(rule["trigger_predicate_id"]) for rule in event_rules}
    required_ids.update(
        str(condition["predicate_id"])
        for rule in event_rules
        for condition in rule.get("support_conditions", ())
        if isinstance(condition, Mapping)
    )
    required_ids.update(
        str(rule["terminal"]["predicate_id"])
        for rule in event_rules
        if isinstance(rule.get("terminal"), Mapping)
    )
    required_ids.update(
        str(predicate_id) for predicate_id in stage_projection_predicate_ids
    )
    unknown_stage_ids = sorted(required_ids - set(get_core_predicate_ids()))
    if unknown_stage_ids:
        raise ValueError(
            f"stage projection references predicates outside the registry: {unknown_stage_ids}"
        )
    if base.get("schema_version") != "3.1.0" or any(
        delta.get("schema_version") != "3.1.0" for delta in deltas
    ):
        raise ValueError("grounded world-truth version mismatch")
    episode_id = str(base["episode_id"])
    templates = {
        str(template["id"]): template for template in get_core_predicate_templates()
    }
    states: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in base.get("initial_assertions", ()):
        if not isinstance(raw, Mapping):
            continue
        predicate_id = str(raw.get("predicate_id") or "")
        if predicate_id not in required_ids:
            continue
        key = (predicate_id, str(raw["tuple_id"]))
        states[key] = dict(raw)
    operations_by_tick: dict[int, list[Mapping[str, Any]]] = {}
    for delta in deltas:
        tick = delta.get("tick")
        if isinstance(tick, int):
            operations_by_tick[tick] = [
                operation
                for operation in delta.get("operations", ())
                if isinstance(operation, Mapping)
                and _operation_predicate_id(operation) in required_ids
            ]

    rows: list[ObservationRow] = []
    assertions: list[Assertion] = []
    lineage_by_truth_key: dict[tuple[str, int, str, str], tuple[str, str]] = {}
    for tick in range(0, 901, 5):
        for operation in operations_by_tick.get(tick, ()):
            operation_name = str(operation.get("operation") or "")
            if operation_name == "add_predicate_assertion":
                assertion = operation.get("assertion")
                if not isinstance(assertion, Mapping):
                    raise ValueError("grounded projection add lacks assertion")
                predicate_id = str(assertion.get("predicate_id") or "")
                if predicate_id not in required_ids:
                    continue
                key = (predicate_id, str(assertion["tuple_id"]))
                if key in states:
                    raise ValueError(
                        f"grounded projection duplicate add for tuple: {key}"
                    )
                states[key] = dict(assertion)
                continue
            predicate_id = str(operation.get("predicate_id") or "")
            if predicate_id not in required_ids:
                continue
            key = (predicate_id, str(operation["tuple_id"]))
            current = states.get(key)
            if current is None:
                raise ValueError(
                    f"grounded event projection lacks an applicable prior tuple: {key}"
                )
            if operation_name == "remove_predicate_assertion":
                if str(current["value"]) != str(operation["from_value"]) or dict(
                    current.get("bindings") or {}
                ) != dict(operation.get("bindings") or {}):
                    raise ValueError(
                        f"grounded projection remove precondition mismatch: {key}"
                    )
                del states[key]
                continue
            if operation_name == "refresh_predicate_evidence":
                if str(current["value"]) != str(operation["value"]):
                    raise ValueError(
                        f"grounded projection evidence refresh value mismatch for {key}"
                    )
                if dict(current.get("bindings") or {}) != dict(
                    operation.get("bindings") or {}
                ):
                    raise ValueError(
                        f"grounded projection evidence refresh binding mismatch for {key}"
                    )
                current.update(
                    missing_source_record=list(operation["missing_source_record"]),
                    observations=list(operation["observations"]),
                    source_refs=list(operation["source_refs"]),
                    evidence_update_tick=tick,
                )
                continue
            if operation_name != "set_predicate_value":
                raise ValueError(
                    f"unknown grounded event projection operation: {operation_name}"
                )
            from_value = str(operation["from_value"])
            to_value = str(operation["to_value"])
            if str(current["value"]) != from_value:
                raise ValueError(
                    f"grounded event projection transition mismatch for {key}: "
                    f"{current['value']} != {from_value}"
                )
            if dict(current.get("bindings") or {}) != dict(
                operation.get("bindings") or {}
            ):
                raise ValueError(
                    f"grounded event projection binding mismatch for {key}"
                )
            current.update(
                {
                    "value": to_value,
                    "missing_source_record": list(
                        operation.get("missing_source_record", ())
                    ),
                    "observations": list(operation.get("observations", ())),
                    "source_refs": list(operation.get("source_refs", ())),
                    "truth_state_update_tick": tick,
                    "evidence_update_tick": tick,
                }
            )
        for (predicate_id, tuple_id), state in sorted(states.items()):
            missing = tuple(
                sorted(
                    str(item)
                    for item in state.get("missing_source_record", ())
                    if isinstance(item, str)
                )
            )
            source_refs = tuple(
                sorted(
                    str(item)
                    for item in state.get("source_refs", ())
                    if isinstance(item, str)
                )
            )
            template = templates[predicate_id]
            roles = _template_roles(template)
            bindings = _state_bindings(state)
            participants = tuple(bindings[role] for role in roles)
            tuple_type = _grounded_tuple_type(roles)
            row = ObservationRow(
                episode_id=episode_id,
                tick=tick,
                tuple_id=tuple_id,
                tuple_type=tuple_type,
                participants=participants,
                scope=True,
                source_class="deterministic_derived",
                source_refs=source_refs,
                mechanism_id=None,
                values={
                    "world_truth_value": state["value"],
                    "world_observations": list(state.get("observations", ())),
                    "truth_state_update_tick": state["truth_state_update_tick"],
                    "evidence_update_tick": state["evidence_update_tick"],
                    "missing_source_record": list(missing),
                    "bindings": bindings,
                    "binding_ontology_classes": dict(
                        state.get("binding_ontology_classes") or {}
                    ),
                },
            )
            rows.append(row)
            state_input_digest = _state_lineage_digest(state, "input_digest")
            state_parameter_digest = _state_lineage_digest(state, "parameter_digest")
            assertions.append(
                Assertion(
                    episode_id=episode_id,
                    tick=tick,
                    tuple_id=tuple_id,
                    tuple_type=row.tuple_type,
                    participants=row.participants,
                    predicate_id=predicate_id,
                    truth=Truth(str(state["value"])),
                    source_class=row.source_class,
                    source_refs=source_refs,
                    mechanism_id=None,
                    missing_source_record=missing,
                )
            )
            lineage_by_truth_key[(episode_id, tick, tuple_id, predicate_id)] = (
                state_input_digest,
                state_parameter_digest,
            )
    return rows, assertions, lineage_by_truth_key


def _operation_predicate_id(operation: Mapping[str, Any]) -> str | None:
    if operation.get("operation") == "add_predicate_assertion":
        assertion = operation.get("assertion")
        if not isinstance(assertion, Mapping):
            raise ValueError("grounded projection add lacks assertion")
        predicate_id = assertion.get("predicate_id")
    else:
        predicate_id = operation.get("predicate_id")
    return str(predicate_id) if isinstance(predicate_id, str) else None


def _template_roles(template: Mapping[str, Any]) -> tuple[str, ...]:
    roles = tuple(
        str(role["key"])
        for role in template.get("argument_roles", ())
        if isinstance(role, Mapping)
    )
    if not roles:
        raise ValueError(f"{template.get('id')}: predicate lacks argument roles")
    return roles


def _state_bindings(state: Mapping[str, Any]) -> dict[str, str]:
    raw = state.get("bindings")
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"grounded assertion lacks bindings: {state.get('predicate_id')} "
            f"{state.get('tuple_id')}"
        )
    return {
        str(role): entity_id
        for role, entity_id in raw.items()
        if isinstance(entity_id, str) and entity_id
    }


def _state_lineage_digest(state: Mapping[str, Any], field: str) -> str:
    value = state.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"grounded assertion lacks {field}: "
            f"{state.get('predicate_id')} {state.get('tuple_id')}"
        )
    return value


def _grounded_tuple_type(roles: Sequence[str]) -> str:
    return GROUNDED_ROLE_TUPLE_PREFIX + "__".join(str(role) for role in roles)


def build_normalized_rows(
    contexts: Sequence[TickContext],
    domain_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    *,
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build L2 candidate tuples solely from the two state computers."""

    del communication_rows, params
    index = _StateIndex(domain_rows)
    rows: list[dict[str, Any]] = []
    for context in sorted(contexts, key=lambda item: item.tick):
        if not context.tick_present:
            continue
        tick = int(context.tick)
        for geometry in index.rows(tick, "predicate_contract_aircraft_geometry"):
            aircraft_id = str(geometry["subject_id"])
            plan = index.get(tick, "predicate_contract_aircraft_plan", aircraft_id)
            aircraft_values = _merged_values(geometry, plan)
            rows.append(
                _normalized_row(
                    context,
                    "aircraft",
                    (aircraft_id,),
                    aircraft_values,
                    (geometry, plan),
                )
            )
            home_pad_id = aircraft_values.get("home_pad_id")
            if isinstance(home_pad_id, str) and home_pad_id:
                rows.append(
                    _normalized_row(
                        context,
                        "aircraft_home_pad",
                        (aircraft_id, home_pad_id),
                        aircraft_values,
                        (geometry,),
                    )
                )
            landing_zone_id = aircraft_values.get("assigned_landing_zone_id")
            if isinstance(landing_zone_id, str) and landing_zone_id:
                rows.append(
                    _normalized_row(
                        context,
                        "aircraft_landing_zone",
                        (aircraft_id, landing_zone_id),
                        aircraft_values,
                        (geometry,),
                    )
                )

        for geometry in index.rows(tick, "predicate_contract_aircraft_region_geometry"):
            subject = str(geometry["subject_id"])
            participants = tuple(part for part in subject.split("|") if part)
            if len(participants) != 2:
                raise ValueError(f"invalid aircraft/region state subject: {subject}")
            aircraft_id, region_id = participants
            runtime_state = index.get(
                tick,
                "predicate_contract_region_runtime_state",
                region_id,
            )
            values = _merged_values(geometry, runtime_state)
            rows.append(
                _normalized_row(
                    context,
                    "aircraft_restricted_region",
                    (aircraft_id, region_id),
                    values,
                    (geometry, runtime_state),
                )
            )

        for pair in index.rows(tick, "predicate_contract_aircraft_pair_geometry"):
            subject = str(pair["subject_id"])
            participants = tuple(part for part in subject.split("|") if part)
            if len(participants) != 2:
                raise ValueError(f"invalid aircraft-pair state subject: {subject}")
            rows.append(
                _normalized_row(
                    context,
                    "aircraft_pair",
                    participants,
                    _merged_values(pair),
                    (pair,),
                )
            )

        for corridor in index.rows(tick, "predicate_contract_corridor_geometry"):
            corridor_id = str(corridor["subject_id"])
            rows.append(
                _normalized_row(
                    context,
                    "corridor",
                    (corridor_id,),
                    _merged_values(corridor),
                    (corridor,),
                )
            )
        for occupancy in index.rows(
            tick, "predicate_contract_aircraft_corridor_geometry"
        ):
            subject = str(occupancy["subject_id"])
            participants = tuple(part for part in subject.split("|") if part)
            if len(participants) != 2:
                raise ValueError(f"invalid aircraft/corridor state subject: {subject}")
            rows.append(
                _normalized_row(
                    context,
                    "aircraft_corridor",
                    participants,
                    _merged_values(occupancy),
                    (occupancy,),
                )
            )

    tuple_templates = {
        str(row["tuple_id"]): (
            str(row["tuple_type"]),
            tuple(str(participant) for participant in row["participants"]),
        )
        for row in rows
    }
    existing = {(int(row["tick"]), str(row["tuple_id"])) for row in rows}
    aircraft_participant_indexes = {
        "aircraft": (0,),
        "aircraft_home_pad": (0,),
        "aircraft_landing_zone": (0,),
        "aircraft_restricted_region": (0,),
        "aircraft_pair": (0, 1),
        "aircraft_corridor": (0,),
    }
    for context in sorted(contexts, key=lambda item: item.tick):
        if not context.tick_present:
            continue
        visible_ids = set(context.frame_entities)
        for tuple_id, (tuple_type, participants) in sorted(tuple_templates.items()):
            if (int(context.tick), tuple_id) in existing:
                continue
            participant_indexes = aircraft_participant_indexes.get(tuple_type)
            if participant_indexes is None:
                raise ValueError(
                    f"normalized tuple disappeared while its non-aircraft scope remained active: "
                    f"tick={context.tick}, tuple={tuple_id}"
                )
            aircraft_ids = tuple(participants[index] for index in participant_indexes)
            if all(aircraft_id in visible_ids for aircraft_id in aircraft_ids):
                raise ValueError(
                    f"normalized tuple missing for visible aircraft participants: "
                    f"tick={context.tick}, tuple={tuple_id}"
                )
            rows.append(
                _normalized_row(
                    context,
                    tuple_type,
                    participants,
                    {},
                    (),
                    scope=False,
                )
            )

    _assert_no_forbidden_keys(rows)
    unexpected = sorted({row["tuple_type"] for row in rows} - NORMALIZED_TUPLE_TYPES)
    if unexpected:
        raise ValueError(f"adapter emitted unsupported tuple types: {unexpected}")
    keys = [(row["episode_id"], row["tick"], row["tuple_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("adapter emitted duplicate normalized tuple rows")
    rows.sort(
        key=lambda row: (
            row["episode_id"],
            row["tick"],
            row["tuple_type"],
            row["tuple_id"],
        )
    )
    return rows


def _merged_values(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    missing: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        values = source.get("values")
        if isinstance(values, Mapping):
            overlap = set(result) & set(values)
            conflicts = sorted(key for key in overlap if result[key] != values[key])
            if conflicts:
                raise ValueError(f"conflicting state-computer values: {conflicts}")
            result.update(values)
        raw_missing = source.get("missing_source_record")
        if isinstance(raw_missing, Sequence) and not isinstance(
            raw_missing, (str, bytes)
        ):
            missing.update(
                str(field) for field in raw_missing if isinstance(field, str)
            )
    result["missing_source_record"] = sorted(missing)
    return result


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _clean_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_clean_value(child) for child in value]
    return value


def _normalized_row(
    context: TickContext,
    tuple_type: str,
    participants: Sequence[str],
    values: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any] | None],
    *,
    scope: bool = True,
) -> dict[str, Any]:
    source_rows = [source for source in sources if isinstance(source, Mapping)]
    refs = sorted(
        {
            str(ref)
            for source in source_rows
            for ref in source.get("source_refs", ())
            if isinstance(ref, str) and ref
        }
    )
    if not refs:
        refs = [f"{context.truth_frames_name}#tick={context.tick}"]
    source_class = (
        "simulated_derived"
        if any(
            source.get("source_class") == "simulated_derived" for source in source_rows
        )
        else "deterministic_derived"
    )
    participant_tuple = tuple(str(participant) for participant in participants)
    return {
        "episode_id": context.episode_id,
        "tick": int(context.tick),
        "tuple_id": f"{tuple_type}:{'|'.join(participant_tuple)}",
        "tuple_type": tuple_type,
        "participants": list(participant_tuple),
        "scope": scope,
        "source_class": source_class,
        "source_refs": refs,
        "mechanism_id": None,
        "values": _clean_value(values),
    }


def _adapt_normalized_state(
    row: ObservationRow, input_digest: str, parameter_digest: str
) -> dict[str, Any]:
    return {
        "schema_name": "semantic_state_observation",
        "schema_version": SCHEMA_VERSION,
        "observation_id": stable_identifier(
            "normalized_tuple", row.episode_id, row.tick, row.tuple_id
        ),
        "episode_id": row.episode_id,
        "tick": row.tick,
        "authoritative_tick": True,
        "api_id": f"normalized.{row.tuple_type}",
        "tuple_id": row.tuple_id,
        "bindings": _bindings(row.tuple_type, row.participants),
        "state_kind": "normalized_candidate_tuple",
        "unit": "structured",
        "value": dict(row.values),
        "source_class": row.source_class,
        "source_refs": list(row.source_refs),
        "quality": (
            "unknown_missing_source"
            if row.values.get("missing_source_record")
            else "projected"
        ),
        "unknown_reason": (
            "missing_source_record" if row.values.get("missing_source_record") else None
        ),
    }


def _adapt_assertions(
    assertions: Sequence[Assertion],
    rows: Sequence[ObservationRow],
    lineage_by_truth_key: Mapping[tuple[str, int, str, str], tuple[str, str]],
) -> list[dict[str, Any]]:
    row_by_id = {(row.episode_id, row.tick, row.tuple_id): row for row in rows}
    result: list[dict[str, Any]] = []
    for item in assertions:
        source = row_by_id[(item.episode_id, item.tick, item.tuple_id)]
        truth_key = (item.episode_id, item.tick, item.tuple_id, item.predicate_id)
        lineage = lineage_by_truth_key.get(truth_key)
        if lineage is None:
            raise ValueError(f"L2 predicate truth lacks exact L1 lineage: {truth_key}")
        input_digest, parameter_digest = lineage
        result.append(
            {
                "schema_name": "predicate_truth",
                "schema_version": PREDICATE_TRUTH_SCHEMA_VERSION,
                "annotation_layer": "L1",
                "source_layer": "L0",
                "materialization_policy": "on_demand_epi_predicate",
                "truth_id": _truth_id(
                    item.episode_id, item.tick, item.tuple_id, item.predicate_id
                ),
                "episode_id": item.episode_id,
                "tick": item.tick,
                "truth_state_update_tick": source.values[
                    "truth_state_update_tick"
                ],
                "evidence_update_tick": source.values["evidence_update_tick"],
                "authoritative_tick": True,
                "predicate_id": item.predicate_id,
                "tuple_id": item.tuple_id,
                "bindings": _bindings(item.tuple_type, item.participants),
                "value": item.truth.value,
                "evidence": {
                    "source_refs": list(item.source_refs),
                    "observations": [
                        {
                            "path": f"normalized.{item.tuple_type}",
                            "value": dict(source.values),
                        }
                    ],
                    "missing_requirements": list(item.missing_source_record),
                },
                "source_class": item.source_class,
                "mechanism_id": item.mechanism_id,
                "template_parameters": {},
            }
        )
    return result


def _adapt_transitions(
    transitions: Sequence[Transition],
    truth_by_key: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    breaks: list[dict[str, Any]] = []
    for item in transitions:
        before = truth_by_key[
            (item.episode_id, item.before_tick, item.tuple_id, item.predicate_id)
        ]
        after = truth_by_key[
            (item.episode_id, item.after_tick, item.tuple_id, item.predicate_id)
        ]
        transition_id = stable_identifier(
            "transition",
            item.episode_id,
            item.predicate_id,
            item.tuple_id,
            item.before_tick,
            item.after_tick,
            item.transition,
        )
        base = {
            "schema_name": (
                "predicate_continuity_break"
                if item.transition == "continuity_break"
                else "predicate_transition"
            ),
            "schema_version": SCHEMA_VERSION,
            "annotation_layer": "L1",
            "source_layer": "L0",
            "materialization_policy": "predicate_delta",
            "transition_id": transition_id,
            "episode_id": item.episode_id,
            "predicate_id": item.predicate_id,
            "tuple_id": item.tuple_id,
            "bindings": _bindings(item.tuple_type, item.participants),
            "from_tick": item.before_tick,
            "to_tick": item.after_tick,
            "from_value": item.before_truth.value,
            "to_value": item.after_truth.value,
            "direction": item.transition,
            "evidence_truth_ids": [before["truth_id"], after["truth_id"]],
            "source_refs": sorted(
                set(before["evidence"]["source_refs"])
                | set(after["evidence"]["source_refs"])
            ),
            "template_parameters": {},
        }
        if item.transition == "continuity_break":
            base["break_id"] = transition_id
            base["reason"] = "non_adjacent_formal_samples"
            breaks.append(base)
        else:
            converted.append(base)
    return converted, breaks


def _exact_predicate_role_binding(
    *,
    rows: Sequence[ObservationRow],
    episode_id: str,
    tuple_id: str,
    tuple_type: str,
    participants: Sequence[str],
    predicate_role: str,
    source_ontology_class: str,
    expected_ticks: Sequence[int],
    expected_truths: Sequence[Truth],
    context: str,
) -> str:
    predicate_bindings = _bindings(tuple_type, participants)
    if not predicate_role or predicate_role not in predicate_bindings:
        raise ValueError(f"{context}: declared predicate role is absent")
    if (
        len(rows) != len(expected_ticks)
        or len(rows) != len(expected_truths)
        or not rows
    ):
        raise ValueError(f"{context}: exact predicate evidence rows are missing")
    if not source_ontology_class or ":" in source_ontology_class:
        raise ValueError(f"{context}: source ontology class must be a catalog class id")
    expected_class_id = f"world:{source_ontology_class}"
    expected_participants = tuple(str(item) for item in participants)
    for row, expected_tick, expected_truth in zip(
        rows, expected_ticks, expected_truths
    ):
        if not row.scope:
            raise ValueError(f"{context}: predicate evidence is out of scope")
        if (
            row.episode_id != episode_id
            or row.tick != expected_tick
            or row.tuple_id != tuple_id
            or row.tuple_type != tuple_type
            or row.participants != expected_participants
        ):
            raise ValueError(f"{context}: predicate tuple evidence does not match")
        row_bindings = row.values.get("bindings")
        row_classes = row.values.get("binding_ontology_classes")
        if (
            not isinstance(row_bindings, Mapping)
            or dict(row_bindings) != dict(predicate_bindings)
            or not isinstance(row_classes, Mapping)
            or set(row_classes) != set(predicate_bindings)
        ):
            raise ValueError(f"{context}: predicate tuple evidence does not match")
        actual_class_id = row_classes.get(predicate_role)
        if not isinstance(actual_class_id, str) or not ontology_class_is_a(
            actual_class_id, expected_class_id
        ):
            raise ValueError(f"{context}: predicate role ontology class differs")
        if row.values.get("world_truth_value") != expected_truth.value:
            raise ValueError(f"{context}: predicate truth evidence differs")
    return str(predicate_bindings[predicate_role])


def _project_event_role_bindings(
    event: EventOccurrence,
    rule: Mapping[str, Any],
    onset_transition: Mapping[str, Any],
    row_by_id: Mapping[tuple[str, int, str], ObservationRow],
    roster_by_id: Mapping[str, Any],
) -> dict[str, str]:
    raw_sources = rule.get("event_role_sources")
    raw_participant_roles = rule.get("participant_roles")
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ValueError(f"{event.event_id}: event role sources are missing")
    if not isinstance(raw_participant_roles, Sequence) or isinstance(
        raw_participant_roles, (str, bytes)
    ):
        raise ValueError(f"{event.event_id}: event participant roles are missing")
    participant_roles = list(raw_participant_roles)
    required_participant_keys = {
        "role",
        "ontology_class",
        "source_kind",
        "background_allowed",
    }
    if any(
        not isinstance(item, Mapping) or set(item) != required_participant_keys
        for item in participant_roles
    ):
        raise ValueError(f"{event.event_id}: event participant role is invalid")
    declared_role_order = [str(item.get("role") or "") for item in participant_roles]
    if (
        any(not isinstance(role, str) or not role for role in raw_sources)
        or any(not role for role in declared_role_order)
        or declared_role_order != list(raw_sources)
    ):
        raise ValueError(
            f"{event.event_id}: event sources do not exactly cover ontology roles"
        )

    trigger_predicate_id = str(rule.get("trigger_predicate_id") or "")
    onset_bindings = _record_bindings(onset_transition)
    expected_onset_bindings = _bindings(event.tuple_type, event.participants)
    if (
        onset_transition.get("episode_id") != event.episode_id
        or onset_transition.get("predicate_id") != trigger_predicate_id
        or onset_transition.get("tuple_id") != event.tuple_id
        or onset_transition.get("to_tick") != event.trigger_tick
        or onset_bindings != expected_onset_bindings
    ):
        raise ValueError(f"{event.event_id}: trigger transition tuple differs")
    onset_row = row_by_id.get((event.episode_id, event.trigger_tick, event.tuple_id))
    if onset_row is None:
        raise ValueError(f"{event.event_id}: trigger predicate row is absent")
    raw_support_conditions = rule.get("support_conditions", ())
    if (
        not isinstance(raw_support_conditions, Sequence)
        or isinstance(raw_support_conditions, (str, bytes))
        or any(
            not isinstance(condition, Mapping) for condition in raw_support_conditions
        )
    ):
        raise ValueError(f"{event.event_id}: support conditions are invalid")
    support_predicate_ids = {
        str(condition.get("predicate_id") or "") for condition in raw_support_conditions
    }
    result: dict[str, str] = {}
    for participant, event_role in zip(participant_roles, declared_role_order):
        source = raw_sources[event_role]
        if not isinstance(source, Mapping):
            raise ValueError(f"{event.event_id}:{event_role}: source is not an object")
        kind = source.get("kind")
        if kind not in {"trigger_binding", "support_binding"}:
            raise ValueError(
                f"{event.event_id}:{event_role}: unsupported runtime source kind {kind!r}"
            )
        required_source_keys = {
            "event_role",
            "event_ontology_class",
            "kind",
            "predicate_id",
            "predicate_role",
            "source_ontology_class",
            "background_allowed",
        }
        if set(source) != required_source_keys:
            raise ValueError(
                f"{event.event_id}:{event_role}: event role source shape differs"
            )
        if (
            source.get("event_role") != event_role
            or participant.get("ontology_class") != source.get("event_ontology_class")
            or participant.get("source_kind") != kind
            or participant.get("background_allowed") != source.get("background_allowed")
        ):
            raise ValueError(
                f"{event.event_id}:{event_role}: participant/source declaration differs"
            )
        predicate_id = source.get("predicate_id")
        predicate_role = source.get("predicate_role")
        source_class = source.get("source_ontology_class")
        if not all(
            isinstance(value, str) and value
            for value in (predicate_id, predicate_role, source_class)
        ) or not isinstance(source.get("background_allowed"), bool):
            raise ValueError(
                f"{event.event_id}:{event_role}: event role source is incomplete"
            )
        if kind == "trigger_binding":
            if predicate_id != trigger_predicate_id:
                raise ValueError(
                    f"{event.event_id}:{event_role}: trigger source predicate differs"
                )
            entity_id = _exact_predicate_role_binding(
                rows=(onset_row,),
                episode_id=event.episode_id,
                tuple_id=event.tuple_id,
                tuple_type=event.tuple_type,
                participants=event.participants,
                predicate_role=predicate_role,
                source_ontology_class=source_class,
                expected_ticks=(event.trigger_tick,),
                expected_truths=(Truth(str(onset_transition["to_value"])),),
                context=f"{event.event_id}:{event_role}",
            )
        else:
            if predicate_id not in support_predicate_ids:
                raise ValueError(
                    f"{event.event_id}:{event_role}: support source predicate differs"
                )
            matching_proofs = [
                proof
                for proof in event.support_condition_proofs
                if proof.predicate_id == predicate_id
            ]
            if len(matching_proofs) != 1:
                raise ValueError(
                    f"{event.event_id}:{event_role}: support source is missing or ambiguous"
                )
            proof = matching_proofs[0]
            proof_rows: list[ObservationRow] = []
            for assertion in proof.assertions:
                proof_row = row_by_id.get(
                    (assertion.episode_id, assertion.tick, assertion.tuple_id)
                )
                if proof_row is None:
                    raise ValueError(
                        f"{event.event_id}:{event_role}: support predicate row is absent"
                    )
                proof_rows.append(proof_row)
            entity_id = _exact_predicate_role_binding(
                rows=proof_rows,
                episode_id=event.episode_id,
                tuple_id=proof.tuple_id,
                tuple_type=proof.tuple_type,
                participants=proof.participants,
                predicate_role=predicate_role,
                source_ontology_class=source_class,
                expected_ticks=tuple(assertion.tick for assertion in proof.assertions),
                expected_truths=tuple(
                    assertion.truth for assertion in proof.assertions
                ),
                context=f"{event.event_id}:{event_role}",
            )
        if source["background_allowed"] is False and not _participant_binding_allowed(
            entity_id, str(event_role), roster_by_id
        ):
            raise ValueError(
                f"{event.event_id}:{event_role}: participant scope class forbids "
                f"event binding for {entity_id}"
            )
        result[event_role] = entity_id
    return result


def _adapt_occurrences(
    events: Sequence[EventOccurrence],
    rows: Sequence[ObservationRow],
    assertions: Sequence[Assertion],
    transitions: Mapping[tuple[str, int, str, str, str], Mapping[str, Any]],
    event_rules: Sequence[Mapping[str, Any]],
    outcomes: Sequence[EventOutcome],
    roster_by_id: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    roster = roster_by_id or {}
    row_by_id: dict[tuple[str, int, str], ObservationRow] = {}
    for row in rows:
        key = (row.episode_id, row.tick, row.tuple_id)
        if key in row_by_id:
            raise ValueError(
                f"duplicate predicate row while adapting occurrences: {key}"
            )
        row_by_id[key] = row
    assertion_by_key: dict[tuple[str, int, str, str], Assertion] = {}
    for item in assertions:
        key = (item.episode_id, item.tick, item.tuple_id, item.predicate_id)
        if key in assertion_by_key:
            raise ValueError(f"duplicate assertion while adapting occurrences: {key}")
        assertion_by_key[key] = item
    rules_by_id = {str(rule["rule_id"]): rule for rule in event_rules}
    outcome_by_event_id: dict[str, EventOutcome] = {}
    for outcome in outcomes:
        if outcome.event_id in outcome_by_event_id:
            raise ValueError(f"duplicate outcome for event: {outcome.event_id}")
        outcome_by_event_id[outcome.event_id] = outcome
    result: list[dict[str, Any]] = []
    for event in events:
        rule = rules_by_id.get(event.rule_id)
        if rule is None:
            raise ValueError(f"event references absent rule: {event.rule_id}")
        trigger_predicate_id = str(rule["trigger_predicate_id"])
        trigger_direction = str(rule["trigger_direction"])
        support_tick = event.support_transition_tick
        onset_transition = transitions.get(
            (
                event.episode_id,
                support_tick,
                event.tuple_id,
                trigger_predicate_id,
                trigger_direction,
            )
        )
        if onset_transition is None:
            raise ValueError(
                f"event has no objective supporting transition: {event.event_id}"
            )
        if event.trigger_tick != support_tick:
            raise ValueError(
                f"event trigger_tick is not its objective onset: {event.event_id}"
            )
        trigger_bindings = _bindings(event.tuple_type, event.participants)
        supporting: list[Mapping[str, Any]] = [onset_transition]
        support_truth_ids = set(onset_transition["evidence_truth_ids"])
        source_refs = set(onset_transition.get("source_refs", ()))
        phase_entries = [
            _event_phase_entry("onset", onset_transition),
        ]
        support_conditions = rule.get("support_conditions", ())
        if not isinstance(support_conditions, Sequence) or isinstance(
            support_conditions, (str, bytes)
        ):
            raise ValueError(
                f"event rule support_conditions must be an array: {rule!r}"
            )
        if len(event.support_condition_proofs) != len(support_conditions):
            raise ValueError(
                f"event support condition proof arity mismatch: {event.event_id}"
            )
        for index, condition in enumerate(support_conditions):
            if not isinstance(condition, Mapping):
                raise ValueError(f"event support condition must be an object: {rule!r}")
            proof = event.support_condition_proofs[index]
            predicate_id = str(condition["predicate_id"])
            desired = Truth(str(condition["value"]))
            if proof.predicate_id != predicate_id:
                raise ValueError(
                    f"event support condition predicate mismatch: {event.event_id}"
                )
            if proof.desired_truth != desired:
                raise ValueError(
                    f"event support condition truth mismatch: {event.event_id}"
                )
            join_roles = _condition_join_roles(condition, event.event_id)
            hold_samples = condition.get("hold_samples", 1)
            if (
                not isinstance(hold_samples, int)
                or isinstance(hold_samples, bool)
                or hold_samples < 1
                or len(proof.assertions) != hold_samples
            ):
                raise ValueError(
                    f"event support condition hold proof arity mismatch: {event.event_id}"
                )
            proof_bindings = _bindings(proof.tuple_type, proof.participants)
            if not _bindings_match_on_roles(
                trigger_bindings, proof_bindings, join_roles
            ):
                raise ValueError(
                    f"event support proof violates declared join roles: {event.event_id}"
                )
            formal_step = int(onset_transition["to_tick"]) - int(
                onset_transition["from_tick"]
            )
            if formal_step <= 0:
                raise ValueError(
                    f"event onset transition does not advance time: {event.event_id}"
                )
            proof_ticks = tuple(item.tick for item in proof.assertions)
            if condition.get("at") == "before":
                expected_ticks = (int(onset_transition["from_tick"]),)
            elif condition.get("at") == "after" and proof_ticks:
                if proof_ticks[0] != int(onset_transition["to_tick"]):
                    raise ValueError(
                        f"event support proof has an invalid after anchor: {event.event_id}"
                    )
                expected_ticks = tuple(
                    proof_ticks[0] + offset * formal_step
                    for offset in range(hold_samples)
                )
            else:
                raise ValueError(
                    f"event support proof has an invalid temporal anchor: {event.event_id}"
                )
            if proof_ticks != expected_ticks:
                raise ValueError(
                    f"event support proof is not strictly consecutive: {event.event_id}"
                )
            for item in proof.assertions:
                if (
                    item.episode_id != event.episode_id
                    or item.predicate_id != predicate_id
                    or item.truth != desired
                    or item.tuple_id != proof.tuple_id
                    or item.tuple_type != proof.tuple_type
                    or item.participants != proof.participants
                ):
                    raise ValueError(
                        f"event support proof changes exact tuple identity: {event.event_id}"
                    )
                exact_item = assertion_by_key.get(
                    (item.episode_id, item.tick, item.tuple_id, item.predicate_id)
                )
                if exact_item != item:
                    raise ValueError(
                        f"event support proof assertion is absent from L1: {event.event_id}"
                    )
                support_truth_ids.add(
                    _truth_id(
                        item.episode_id,
                        item.tick,
                        item.tuple_id,
                        item.predicate_id,
                    )
                )
                source_refs.update(item.source_refs)
            if desired in {Truth.TRUE, Truth.FALSE}:
                direction = "rising" if desired == Truth.TRUE else "falling"
                support_transition = transitions.get(
                    (
                        event.episode_id,
                        proof.start_tick,
                        proof.tuple_id,
                        predicate_id,
                        direction,
                    )
                )
                if support_transition is not None:
                    expected_from = Truth.FALSE if desired == Truth.TRUE else Truth.TRUE
                    if (
                        dict(support_transition["bindings"]) != proof_bindings
                        or int(support_transition["from_tick"])
                        != proof.start_tick - formal_step
                        or int(support_transition["to_tick"]) != proof.start_tick
                        or Truth(str(support_transition["from_value"])) != expected_from
                        or Truth(str(support_transition["to_value"])) != desired
                        or str(support_transition["direction"]) != direction
                    ):
                        raise ValueError(
                            f"event support transition conflicts with exact proof: {event.event_id}"
                        )
                    if (
                        support_transition["transition_id"]
                        == onset_transition["transition_id"]
                    ):
                        raise ValueError(
                            "event support transition reuses onset transition: "
                            f"{event.event_id} {support_transition['transition_id']}"
                        )
                    _append_event_phase(
                        phase_entries,
                        supporting,
                        "escalation_support",
                        support_transition,
                        event.event_id,
                    )
                    support_truth_ids.update(support_transition["evidence_truth_ids"])
                    source_refs.update(support_transition.get("source_refs", ()))
        expected_detection_tick = max(
            (
                support_tick,
                *(proof.end_tick for proof in event.support_condition_proofs),
            )
        )
        if event.detection_tick != expected_detection_tick:
            raise ValueError(
                f"event detection_tick does not equal support confirmation: {event.event_id}"
            )
        event_bindings = _project_event_role_bindings(
            event,
            rule,
            onset_transition,
            row_by_id,
            roster_by_id or {},
        )
        for predicate_id in event.trigger_predicates:
            assertion = assertion_by_key.get(
                (
                    event.episode_id,
                    event.trigger_tick,
                    event.tuple_id,
                    predicate_id,
                )
            )
            if assertion is not None:
                source_refs.update(assertion.source_refs)
        row = row_by_id[(event.episode_id, event.trigger_tick, event.tuple_id)]
        source_refs.update(row.source_refs)
        outcome = outcome_by_event_id.get(event.event_id)
        if outcome is None:
            raise ValueError(f"event has no lifecycle outcome: {event.event_id}")
        if outcome.outcome == "success":
            terminal_engine_transition = outcome.terminal_transition
            if terminal_engine_transition is None:
                raise ValueError(
                    f"successful outcome lacks terminal transition: {event.event_id}"
                )
            terminal_transition = transitions.get(
                (
                    terminal_engine_transition.episode_id,
                    terminal_engine_transition.after_tick,
                    terminal_engine_transition.tuple_id,
                    terminal_engine_transition.predicate_id,
                    terminal_engine_transition.transition,
                )
            )
            if terminal_transition is None:
                raise ValueError(
                    f"outcome terminal transition is absent from L1: {event.event_id}"
                )
            if terminal_transition["transition_id"] not in {
                phase["transition_id"] for phase in phase_entries
            }:
                _append_event_phase(
                    phase_entries,
                    supporting,
                    "terminal",
                    terminal_transition,
                    event.event_id,
                )
            else:
                phase_entries.append(
                    _event_phase_entry("terminal", terminal_transition)
                )
            support_truth_ids.update(terminal_transition["evidence_truth_ids"])
            source_refs.update(terminal_transition.get("source_refs", ()))
        supporting_transition_ids = _phase_transition_ids(phase_entries)
        if supporting_transition_ids != tuple(
            str(transition["transition_id"]) for transition in supporting
        ):
            raise ValueError(
                "event phases and supporting transitions are not one-to-one: "
                f"{event.event_id}"
            )
        if not source_refs:
            raise ValueError(f"event has no objective source refs: {event.event_id}")
        occurrence = {
            "schema_name": "event_occurrence",
            "schema_version": SCHEMA_VERSION,
            "annotation_layer": "L2",
            "source_layer": "L1",
            "event_id": event.event_id,
            "episode_id": event.episode_id,
            "event_family_id": event.event_family_id,
            "event_type_id": event.event_type,
            "rule_id": event.rule_id,
            "trigger_tick": event.trigger_tick,
            "detection_tick": event.detection_tick,
            "start_tick": event.trigger_tick,
            "end_tick": event.detection_tick,
            "trigger_predicate_id": event.trigger_predicates[0],
            "bindings": event_bindings,
            "participant_roles": list(event_bindings),
            "supporting_truth_ids": sorted(support_truth_ids),
            "supporting_transition_ids": list(supporting_transition_ids),
            "source_refs": sorted(source_refs),
            "source_class": "deterministic_derived",
            "supporting_parameter_refs": (
                ["aeroworld.world_truth_predicates.v3#parameter=boundary_margin_m"]
                if event.event_family_id == "restricted_region_intrusion"
                else []
            ),
            "continuity_status": "strict_adjacent_transition",
            "event_level": 2,
            "event_phases": phase_entries,
            "mechanism_id": event.mechanism_id,
            "template_parameters": {},
        }
        if event.event_subtype is not None:
            occurrence["event_subtype"] = event.event_subtype
        result.append(occurrence)
    return result


def _condition_join_roles(
    condition: Mapping[str, Any],
    event_id: str,
) -> tuple[str, ...]:
    join_roles = condition.get("join_roles")
    if (
        not isinstance(join_roles, Sequence)
        or isinstance(join_roles, (str, bytes))
        or not join_roles
        or any(not isinstance(role, str) or not role for role in join_roles)
    ):
        raise ValueError(f"{event_id}: support condition must declare join_roles")
    return tuple(str(role) for role in join_roles)


def _bindings_match_on_roles(
    left: Mapping[str, str],
    right: Mapping[str, str],
    roles: Sequence[str],
) -> bool:
    return all(
        role in left and role in right and left[role] == right[role] for role in roles
    )


def _event_phase_entry(
    phase_kind: str,
    transition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "phase_kind": phase_kind,
        "predicate_id": transition["predicate_id"],
        "predicate_tuple_id": transition["tuple_id"],
        "predicate_bindings": dict(transition["bindings"]),
        "direction": transition["direction"],
        "tick": transition["to_tick"],
        "transition_id": transition["transition_id"],
    }


def _append_event_phase(
    phases: list[dict[str, Any]],
    supporting: list[Mapping[str, Any]],
    phase_kind: str,
    transition: Mapping[str, Any],
    event_id: str,
) -> None:
    transition_id = str(transition["transition_id"])
    existing = next(
        (phase for phase in phases if phase["transition_id"] == transition_id),
        None,
    )
    if existing is not None:
        raise ValueError(
            f"transition cannot be appended twice to event phases: {event_id} {transition_id}"
        )
    phases.append(_event_phase_entry(phase_kind, transition))
    supporting.append(transition)


def _phase_transition_ids(
    phases: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    transition_ids: list[str] = []
    phase_keys: set[tuple[str, str]] = set()
    phase_kinds_by_transition: dict[str, set[str]] = {}
    for phase in phases:
        transition_id = phase.get("transition_id")
        phase_kind = phase.get("phase_kind")
        if not isinstance(transition_id, str) or not transition_id:
            raise ValueError("event phase lacks a non-empty transition_id")
        if not isinstance(phase_kind, str) or not phase_kind:
            raise ValueError("event phase lacks a non-empty phase_kind")
        phase_key = (phase_kind, transition_id)
        if phase_key in phase_keys:
            raise ValueError("event phases contain a duplicate phase role")
        phase_keys.add(phase_key)
        phase_kinds_by_transition.setdefault(transition_id, set()).add(phase_kind)
        if transition_id not in transition_ids:
            transition_ids.append(transition_id)
    if any(
        kinds != {"onset", "terminal"}
        for kinds in phase_kinds_by_transition.values()
        if len(kinds) > 1
    ):
        raise ValueError(
            "one transition may serve multiple phases only as onset and terminal"
        )
    return tuple(transition_ids)


def _adapt_outcomes(
    outcomes: Sequence[EventOutcome],
    occurrences: Sequence[Mapping[str, Any]],
    truth_by_key: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    *,
    formal_step: int,
) -> list[dict[str, Any]]:
    if formal_step <= 0:
        raise ValueError("formal outcome step must be positive")
    occurrence_by_id = {str(row["event_id"]): row for row in occurrences}
    status_map = {"success": "succeeded", "pending": "pending"}
    result: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome.outcome not in status_map:
            raise ValueError(
                f"unsupported objective outcome status: {outcome.event_id} {outcome.outcome}"
            )
        occurrence = occurrence_by_id.get(outcome.event_id)
        if occurrence is None:
            raise ValueError(
                f"outcome references absent occurrence: {outcome.event_id}"
            )
        truth_ids = (
            set(occurrence.get("supporting_truth_ids", ()))
            if outcome.outcome != "pending"
            else set()
        )
        lifecycle_phase_evidence: list[dict[str, Any]] = []
        if outcome.outcome != "pending":
            if outcome.terminal_tick is None or outcome.terminal_transition is None:
                raise ValueError(
                    f"successful outcome lacks exact terminal proof: {outcome.event_id}"
                )
            hold_truth_rows: list[Mapping[str, Any]] = []
            for assertion in outcome.terminal_hold_assertions:
                terminal = truth_by_key.get(
                    (
                        assertion.episode_id,
                        assertion.tick,
                        assertion.tuple_id,
                        assertion.predicate_id,
                    )
                )
                if terminal is None:
                    raise ValueError(
                        "outcome terminal hold assertion is absent from L1: "
                        f"{outcome.event_id} {assertion.tick} {assertion.tuple_id}"
                    )
                truth_ids.add(str(terminal["truth_id"]))
                hold_truth_rows.append(terminal)
            if not hold_truth_rows:
                raise ValueError(
                    f"successful outcome lacks terminal hold truth: {outcome.event_id}"
                )
            hold_ticks = [int(row["tick"]) for row in hold_truth_rows]
            transition_step = (
                outcome.terminal_transition.after_tick
                - outcome.terminal_transition.before_tick
            )
            if transition_step != formal_step:
                raise ValueError(
                    f"outcome terminal transition is not formal-grid adjacent: {outcome.event_id}"
                )
            expected_hold_ticks = list(
                range(hold_ticks[0], hold_ticks[-1] + formal_step, formal_step)
            )
            if (
                hold_ticks != expected_hold_ticks
                or outcome.terminal_tick != hold_ticks[-1]
            ):
                raise ValueError(
                    f"outcome terminal hold is not exact-grid continuous: {outcome.event_id}"
                )
            first_hold = hold_truth_rows[0]
            if any(
                row["predicate_id"] != first_hold["predicate_id"]
                or row["tuple_id"] != first_hold["tuple_id"]
                or row["bindings"] != first_hold["bindings"]
                for row in hold_truth_rows
            ):
                raise ValueError(
                    f"outcome terminal hold changes exact tuple identity: {outcome.event_id}"
                )
            entry_transition_matches = [
                phase
                for phase in occurrence.get("event_phases", ())
                if phase.get("phase_kind") == "terminal"
                and phase.get("predicate_id")
                == outcome.terminal_transition.predicate_id
                and phase.get("predicate_tuple_id")
                == outcome.terminal_transition.tuple_id
                and phase.get("tick") == outcome.terminal_transition.after_tick
                and phase.get("direction") == outcome.terminal_transition.transition
            ]
            if len(entry_transition_matches) != 1:
                raise ValueError(
                    f"outcome terminal entry transition is not an exact occurrence phase: {outcome.event_id}"
                )
            lifecycle_phase_evidence.append(
                {
                    "phase_kind": "terminal_hold",
                    "predicate_id": first_hold["predicate_id"],
                    "predicate_tuple_id": first_hold["tuple_id"],
                    "predicate_bindings": dict(first_hold["bindings"]),
                    "value": first_hold["value"],
                    "start_tick": int(hold_truth_rows[0]["tick"]),
                    "end_tick": int(hold_truth_rows[-1]["tick"]),
                    "entry_transition_id": entry_transition_matches[0]["transition_id"],
                    "supporting_truth_ids": [
                        str(row["truth_id"]) for row in hold_truth_rows
                    ],
                }
            )
        result.append(
            {
                "schema_name": "event_outcome",
                "schema_version": SCHEMA_VERSION,
                "annotation_layer": "L2",
                "source_layer": "L1",
                "outcome_id": stable_identifier(
                    "outcome", outcome.event_id, outcome.outcome, outcome.terminal_tick
                ),
                "event_id": outcome.event_id,
                "episode_id": outcome.episode_id,
                "event_family_id": outcome.event_family_id,
                "event_type_id": outcome.event_type,
                "status": status_map[outcome.outcome],
                "terminal_tick": outcome.terminal_tick,
                "lifecycle_status": outcome.reason,
                "source_class": "deterministic_derived",
                "supporting_truth_ids": sorted(truth_ids),
                "lifecycle_phase_evidence": lifecycle_phase_evidence,
            }
        )
    return result


def _record_bindings(row: Mapping[str, Any]) -> dict[str, str]:
    raw = row.get("bindings")
    if not isinstance(raw, Mapping):
        return {}
    return dict(
        sorted(
            {
                str(role): entity_id
                for role, entity_id in raw.items()
                if isinstance(entity_id, str)
            }.items()
        )
    )


def _assert_core_vocabulary(assertions: Sequence[Assertion]) -> None:
    actual = {item.predicate_id for item in assertions}
    foreign = sorted(actual - ENGINE_PREDICATE_VOCABULARY)
    if foreign:
        raise ValueError(f"engine emitted predicates absent from domain TTL: {foreign}")
    non_executable = sorted(actual - set(get_executable_predicate_ids()))
    if non_executable:
        raise ValueError(f"engine emitted non-executable predicates: {non_executable}")


def _bindings(tuple_type: str, participants: Sequence[str]) -> dict[str, str]:
    roles_by_type = {
        "aircraft": ("aircraft",),
        "aircraft_home_pad": ("aircraft", "home_pad"),
        "aircraft_landing_zone": ("aircraft", "landing_zone"),
        "aircraft_restricted_region": ("aircraft", "restricted_region"),
        "aircraft_pair": ("aircraft_a", "aircraft_b"),
        "aircraft_corridor": ("aircraft", "corridor"),
        "corridor": ("corridor",),
        "world_scope_uav": ("aircraft",),
        "world_scope_vehicle": ("vehicle",),
        "world_scope_pedestrian": ("pedestrian",),
        "world_scope_facility": ("facility",),
        "world_scope_compute_node": ("compute_node",),
        "world_scope_scene": ("scene",),
    }
    if tuple_type.startswith(GROUNDED_ROLE_TUPLE_PREFIX):
        roles = tuple(
            role
            for role in tuple_type.removeprefix(GROUNDED_ROLE_TUPLE_PREFIX).split("__")
            if role
        )
    else:
        roles = roles_by_type[tuple_type]
    if len(roles) != len(participants):
        raise ValueError(
            f"tuple role count mismatch for {tuple_type}: "
            f"roles={roles}, participants={participants}"
        )
    return {role: str(participant) for role, participant in zip(roles, participants)}


def _truth_id(episode_id: str, tick: int, tuple_id: str, predicate_id: str) -> str:
    return stable_identifier("truth", episode_id, tick, tuple_id, predicate_id)


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_INPUT_KEYS & {str(key) for key in value}
        if forbidden:
            raise ValueError(
                f"forbidden authored/label fields reached the adapter: {sorted(forbidden)}"
            )
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _assert_no_forbidden_keys(child)


__all__ = [
    "MinimalSemanticResult",
    "NORMALIZED_TUPLE_TYPES",
    "build_normalized_rows",
    "run_minimal_semantic_engine",
]
