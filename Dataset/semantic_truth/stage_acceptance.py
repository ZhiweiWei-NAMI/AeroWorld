"""Exact, evidence-bound evaluation of EPI stage acceptance contracts.

The evaluator never interprets ``stage_goal`` text.  Every stage is selected by
``epi_id::stage_index`` and evaluated only through its explicit predicate,
event, or lifecycle clauses.  A stage that is explicitly outside the published
semantic scope remains ``NOT_APPLICABLE``; it is never counted as evidence and
is never confused with an executable stage whose evidence is ``UNKNOWN``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Mapping, Sequence

from .provenance import digest_object, unique_sorted
from Dataset.semantic_truth.entity_scope import is_background


SCHEMA_VERSION = "2.0.0"
PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"
SUPPORTED_EVENT_PHASES = {
    "onset",
    "escalation_support",
    "terminal",
}


class StageAcceptanceError(RuntimeError):
    """Raised when the stage contract or supplied lifecycle data is invalid."""


@dataclass(frozen=True)
class _EvidenceMatch:
    tick: int | None
    bindings: dict[str, str]
    record_ids: tuple[str, ...]
    evidence_types: frozenset[str]
    detail: dict[str, Any]


@dataclass(frozen=True)
class _ClauseEvaluation:
    status: str
    reason: str
    matches: tuple[_EvidenceMatch, ...]


def validate_stage_acceptance_contract(
    source_contract: Mapping[str, Any],
    stage_contract: Mapping[str, Any],
) -> None:
    """Require exact one-to-one preservation of all source contract stages."""

    if stage_contract.get("schema_name") != "epi_stage_acceptance_contract":
        raise StageAcceptanceError("invalid stage acceptance schema_name")
    if stage_contract.get("stage_key_format") != "{epi_id}::{stage_index}":
        raise StageAcceptanceError("invalid stage key format")
    stages = stage_contract.get("stages")
    epi_contracts = source_contract.get("epi_contracts")
    if not isinstance(stages, Mapping) or not isinstance(epi_contracts, Mapping):
        raise StageAcceptanceError("source and stage contracts must contain mappings")
    expected: dict[str, tuple[str, int, str, str]] = {}
    for epi_id, epi_contract in epi_contracts.items():
        if not isinstance(epi_contract, Mapping):
            raise StageAcceptanceError(f"invalid source EPI contract: {epi_id}")
        required_chain = epi_contract.get("required_chain")
        if not isinstance(required_chain, Sequence):
            raise StageAcceptanceError(f"source EPI lacks required_chain: {epi_id}")
        for stage_index, stage in enumerate(required_chain):
            if not isinstance(stage, Mapping):
                raise StageAcceptanceError(
                    f"invalid source stage: {epi_id}::{stage_index}"
                )
            key = f"{epi_id}::{stage_index}"
            expected[key] = (
                str(epi_id),
                stage_index,
                str(stage.get("stage_goal") or ""),
                str(stage.get("event_family_id") or ""),
            )
    if set(stages) != set(expected):
        raise StageAcceptanceError(
            "stage contract keys do not exactly match the source contract: "
            f"missing={sorted(set(expected) - set(stages))[:10]} "
            f"extra={sorted(set(stages) - set(expected))[:10]}"
        )
    for key, expected_fields in expected.items():
        stage = stages[key]
        if not isinstance(stage, Mapping):
            raise StageAcceptanceError(f"stage contract entry is not an object: {key}")
        actual_fields = (
            str(stage.get("epi_id") or ""),
            stage.get("stage_index"),
            str(stage.get("stage_goal") or ""),
            str(stage.get("event_family_id") or ""),
        )
        if actual_fields != expected_fields or stage.get("key") != key:
            raise StageAcceptanceError(f"stage contract drift at {key}")
        clauses = stage.get("acceptance_clauses")
        if not isinstance(clauses, Sequence) or isinstance(clauses, (str, bytes)):
            raise StageAcceptanceError(f"stage clauses must be an array: {key}")
        implementation_status = stage.get("implementation_status")
        if implementation_status == "not_applicable":
            if "binding_projection_from_previous" in stage:
                raise StageAcceptanceError(
                    f"not-applicable stage declares a binding projection: {key}"
                )
            if clauses or stage.get("required_evidence_types") != [
                "declared_not_applicable"
            ]:
                raise StageAcceptanceError(
                    f"not-applicable stage lacks an explicit declaration: {key}"
                )
            continue
        if implementation_status != "executable" or not clauses:
            raise StageAcceptanceError(f"executable stage has no exact clause: {key}")
        _validate_binding_projection(stage, stages, key)
        if any(not isinstance(clause, Mapping) for clause in clauses):
            raise StageAcceptanceError(f"stage has a non-object clause: {key}")
        for clause in clauses:
            _validate_clause_templates(clause, key)
    validation = stage_contract.get("validation")
    if not isinstance(validation, Mapping):
        raise StageAcceptanceError("stage contract lacks validation policy")
    if validation.get("runtime_keyword_matching_permitted") is not False:
        raise StageAcceptanceError("runtime keyword matching must be forbidden")
    if int(validation.get("expected_epi_count", -1)) != len(epi_contracts):
        raise StageAcceptanceError("stage contract EPI count drift")
    if int(validation.get("expected_stage_count", -1)) != len(expected):
        raise StageAcceptanceError("stage contract stage count drift")
    catalog = stage_contract.get("evidence_type_catalog")
    if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)):
        raise StageAcceptanceError("stage evidence catalog must be an array")
    used_evidence_types = {
        str(evidence_type)
        for stage in stages.values()
        for evidence_type in stage.get("required_evidence_types", ())
    }
    missing_evidence_types = sorted(used_evidence_types - set(catalog))
    if missing_evidence_types:
        raise StageAcceptanceError(
            "stage evidence catalog omits required evidence types: "
            f"{missing_evidence_types}"
        )


def evaluate_epi_stage_acceptance(
    source_contract: Mapping[str, Any],
    stage_contract: Mapping[str, Any],
    *,
    episode_id: str,
    epi_id: str,
    semantic_state: Sequence[Mapping[str, Any]],
    predicate_truth: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    roster_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate one episode against its exact stage records."""

    validate_stage_acceptance_contract(source_contract, stage_contract)
    source_epi = source_contract.get("epi_contracts", {}).get(epi_id)
    if not isinstance(source_epi, Mapping):
        return {
            "schema_name": "epi_closure",
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "epi_id": epi_id,
            "status": UNKNOWN,
            "event_count": len(occurrences),
            "stage_status_counts": {UNKNOWN: 1},
            "stage_contract_profile_id": stage_contract.get("profile_id"),
            "stage_contract_digest": digest_object(stage_contract),
            "stages": [],
            "unknown_reason": "episode_epi_id_not_declared_in_stage_contract",
        }

    index = _AcceptanceIndex(
        episode_id=episode_id,
        semantic_state=semantic_state,
        predicate_truth=predicate_truth,
        transitions=transitions,
        occurrences=occurrences,
        outcomes=outcomes,
        roster_by_id=roster_by_id,
    )
    stages_by_key = stage_contract["stages"]
    stage_count = len(source_epi.get("required_chain", ()))
    stage_sequence = [
        stages_by_key[f"{epi_id}::{stage_index}"] for stage_index in range(stage_count)
    ]
    evaluated = _evaluate_stage_chain(stage_sequence, index)
    counts = Counter(str(stage["stage_goal_status"]) for stage in evaluated)
    executable = [
        stage
        for stage in evaluated
        if stage.get("implementation_status") == "executable"
    ]
    executable_counts = Counter(str(stage["stage_goal_status"]) for stage in executable)
    if executable_counts.get(FAIL, 0):
        status = FAIL
    elif executable_counts.get(UNKNOWN, 0):
        status = UNKNOWN
    elif all(stage["stage_goal_status"] == PASS for stage in executable):
        status = PASS
    else:
        status = FAIL
    not_applicable_count = counts.get(NOT_APPLICABLE, 0)
    if not_applicable_count == 0:
        coverage_status = "COMPLETE"
    elif executable:
        coverage_status = "EXPLICIT_GAPS"
    else:
        coverage_status = "NO_EXECUTABLE_STAGES"
    return {
        "schema_name": "epi_closure",
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "epi_id": epi_id,
        "status": status,
        "event_count": len(occurrences),
        "stage_status_counts": dict(sorted(counts.items())),
        "executable_stage_status_counts": dict(sorted(executable_counts.items())),
        "executable_stage_count": len(executable),
        "not_applicable_stage_count": not_applicable_count,
        "semantic_coverage_status": coverage_status,
        "stage_contract_profile_id": stage_contract.get("profile_id"),
        "stage_contract_digest": digest_object(stage_contract),
        "runtime_keyword_matching_used": False,
        "stages": evaluated,
    }


class _AcceptanceIndex:
    def __init__(
        self,
        *,
        episode_id: str,
        semantic_state: Sequence[Mapping[str, Any]],
        predicate_truth: Sequence[Mapping[str, Any]],
        transitions: Sequence[Mapping[str, Any]],
        occurrences: Sequence[Mapping[str, Any]],
        outcomes: Sequence[Mapping[str, Any]],
        roster_by_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.episode_id = episode_id
        if not roster_by_id:
            raise StageAcceptanceError(
                "acceptance index requires the episode roster for background classification"
            )
        self.roster_by_id = roster_by_id
        self.semantic_state = [
            row for row in semantic_state if _episode_matches(row, episode_id)
        ]
        self.truth = [
            row for row in predicate_truth if _episode_matches(row, episode_id)
        ]
        self.transitions = [
            row for row in transitions if _episode_matches(row, episode_id)
        ]
        self.occurrences = [
            row
            for row in occurrences
            if _episode_matches(row, episode_id)
            and not any(
                is_background(entity_id, self.roster_by_id)
                for entity_id in _bindings(row).values()
            )
        ]
        self.outcomes = [row for row in outcomes if _episode_matches(row, episode_id)]
        self.truth_by_id = {
            str(row["truth_id"]): row
            for row in self.truth
            if isinstance(row.get("truth_id"), str)
        }
        self.truth_by_predicate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in self.truth:
            self.truth_by_predicate[str(row.get("predicate_id", ""))].append(row)
        self.transition_by_id = {
            str(row["transition_id"]): row
            for row in self.transitions
            if isinstance(row.get("transition_id"), str)
        }
        self.occurrence_by_id = {
            str(row["event_id"]): row
            for row in self.occurrences
            if isinstance(row.get("event_id"), str)
        }

    def predicate_is_unknown(self, predicate_id: str) -> bool:
        rows = [
            row
            for row in self.truth_by_predicate.get(predicate_id, ())
            if not any(
                is_background(entity_id, self.roster_by_id)
                for entity_id in _bindings(row).values()
            )
        ]
        return not rows or not any(
            row.get("value") in {"true", "false"}
            and not (row.get("evidence") or {}).get("missing_requirements")
            for row in rows
        )

    def has_numeric_state(self, tick: int, bindings: Mapping[str, str]) -> bool:
        bound_ids = set(bindings.values())
        for row in self.semantic_state:
            if row.get("tick") != tick:
                continue
            row_bindings = _bindings(row)
            if (
                bound_ids
                and row_bindings
                and not (bound_ids & set(row_bindings.values()))
            ):
                continue
            if _contains_number(
                {
                    "value": row.get("value"),
                    "source_values": row.get("source_values"),
                }
            ):
                return True
        return False


def _evaluate_stage_chain(
    stages: Sequence[Mapping[str, Any]],
    index: _AcceptanceIndex,
) -> list[dict[str, Any]]:
    evaluated, _complete = _search_stage_chain(stages, index, 0, None)
    return evaluated


def _search_stage_chain(
    stages: Sequence[Mapping[str, Any]],
    index: _AcceptanceIndex,
    stage_offset: int,
    previous: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], bool]:
    if stage_offset >= len(stages):
        return [], True

    stage = stages[stage_offset]
    if stage.get("implementation_status") == "not_applicable":
        declared = _not_applicable_stage_result(
            stage,
            observed_event_count=len(index.occurrences),
        )
        suffix, complete = _search_stage_chain(
            stages, index, stage_offset + 1, previous
        )
        return [declared, *suffix], complete
    candidates = _stage_pass_candidates(stage, index, previous)
    if not candidates:
        failed = _evaluate_stage(stage, index, previous)
        return [
            failed,
            *_evaluate_remaining_stages(stages, index, stage_offset + 1, failed),
        ], False

    best: list[dict[str, Any]] | None = None
    best_passed = -1
    for candidate in candidates:
        suffix, complete = _search_stage_chain(
            stages, index, stage_offset + 1, candidate
        )
        chain = [candidate, *suffix]
        if complete:
            return chain, True
        passed = sum(1 for item in chain if item.get("stage_goal_status") == PASS)
        if passed > best_passed:
            best = chain
            best_passed = passed
    return best or [], False


def _evaluate_remaining_stages(
    stages: Sequence[Mapping[str, Any]],
    index: _AcceptanceIndex,
    stage_offset: int,
    previous: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for stage in stages[stage_offset:]:
        if stage.get("implementation_status") == "not_applicable":
            evaluated.append(
                _not_applicable_stage_result(
                    stage,
                    observed_event_count=len(index.occurrences),
                )
            )
            continue
        result = _evaluate_stage(stage, index, previous)
        evaluated.append(result)
        previous = result
    return evaluated


def _not_applicable_stage_result(
    stage: Mapping[str, Any],
    *,
    observed_event_count: int,
) -> dict[str, Any]:
    return {
        "stage_key": str(stage["key"]),
        "stage_index": int(stage["stage_index"]),
        "stage_goal": str(stage["stage_goal"]),
        "event_family_id": str(stage["event_family_id"]),
        "implementation_status": "not_applicable",
        "stage_kind": "not_applicable",
        "stage_goal_status": NOT_APPLICABLE,
        "status_reason": "declared_not_applicable",
        "observed_event_count": observed_event_count,
        "supporting_record_ids": [],
        "supporting_ticks": [],
        "supporting_tick": None,
        "binding_sets": [],
        "selected_event_ids": [],
        "required_evidence_types": ["declared_not_applicable"],
        "observed_evidence_types": ["declared_not_applicable"],
        "missing_evidence_types": [],
        "clause_results": [],
        "not_applicable_reason": str(stage.get("not_applicable_reason") or ""),
        "closure_statement": str(stage.get("closure_statement") or ""),
        "temporal_order": dict(stage["temporal_order"]),
        "temporal_interpretation": "declared_not_applicable_no_physical_tick",
    }


def _evaluate_stage(
    stage: Mapping[str, Any],
    index: _AcceptanceIndex,
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    clauses = _clauses_for_stage(stage)
    clause_results = [_evaluate_clause(clause, index) for clause in clauses]
    required_evidence = {str(item) for item in stage.get("required_evidence_types", ())}
    minimum_bindings = int(stage.get("minimum_distinct_binding_sets", 1))
    policy = str(stage["clause_policy"])
    selected, evidence_status, reason = _select_stage_matches(
        policy,
        clause_results,
        minimum_bindings=minimum_bindings,
        previous=previous,
        event_family_id=str(stage["event_family_id"]),
        binding_projection=_binding_projection(stage),
    )
    return _stage_result_from_selection(
        stage,
        clauses,
        clause_results,
        selected,
        evidence_status,
        reason,
        required_evidence=required_evidence,
        previous=previous,
    )


def _stage_pass_candidates(
    stage: Mapping[str, Any],
    index: _AcceptanceIndex,
    previous: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    clauses = _clauses_for_stage(stage)
    clause_results = [_evaluate_clause(clause, index) for clause in clauses]
    required_evidence = {str(item) for item in stage.get("required_evidence_types", ())}
    minimum_bindings = int(stage.get("minimum_distinct_binding_sets", 1))
    selected_sets = _candidate_selected_match_sets(
        str(stage["clause_policy"]),
        clause_results,
        minimum_bindings=minimum_bindings,
        previous=previous,
        event_family_id=str(stage["event_family_id"]),
        binding_projection=_binding_projection(stage),
    )
    candidates: list[dict[str, Any]] = []
    for selected in selected_sets:
        result = _stage_result_from_selection(
            stage,
            clauses,
            clause_results,
            selected,
            PASS,
            "exact_contract_clause_satisfied",
            required_evidence=required_evidence,
            previous=previous,
        )
        if result["stage_goal_status"] == PASS:
            candidates.append(result)
    candidates.sort(key=_stage_result_sort_key)
    return candidates


def _stage_result_from_selection(
    stage: Mapping[str, Any],
    clauses: Sequence[Mapping[str, Any]],
    clause_results: Sequence[_ClauseEvaluation],
    selected: Sequence[_EvidenceMatch],
    status: str,
    reason: str,
    *,
    required_evidence: set[str],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    observed_evidence = (
        set().union(*(match.evidence_types for match in selected))
        if selected
        else set()
    )
    missing_evidence = sorted(required_evidence - observed_evidence)
    if status == PASS and missing_evidence:
        status = UNKNOWN
        reason = "required_evidence_missing"
    if (
        previous is not None
        and previous.get("stage_goal_status") != PASS
        and status == PASS
    ):
        status = UNKNOWN
        reason = "previous_stage_not_accepted"
    supporting_ticks = sorted(
        {match.tick for match in selected if isinstance(match.tick, int)}
    )
    support_tick = max(supporting_ticks) if supporting_ticks else None
    record_ids = unique_sorted(
        record_id for match in selected for record_id in match.record_ids
    )
    binding_sets = [
        dict(sorted(match.bindings.items())) for match in selected if match.bindings
    ]
    selected_event_ids = _selected_event_ids(selected)
    return {
        "stage_key": str(stage["key"]),
        "stage_index": int(stage["stage_index"]),
        "stage_goal": str(stage["stage_goal"]),
        "event_family_id": str(stage["event_family_id"]),
        "implementation_status": str(stage["implementation_status"]),
        "stage_kind": "+".join(sorted({str(c["acceptance_kind"]) for c in clauses})),
        "stage_goal_status": status,
        "status_reason": reason,
        "supporting_record_ids": list(record_ids),
        "supporting_ticks": supporting_ticks,
        "supporting_tick": support_tick,
        "binding_sets": _deduplicate_bindings(binding_sets),
        "selected_event_ids": sorted(selected_event_ids),
        "required_evidence_types": sorted(required_evidence),
        "observed_evidence_types": sorted(observed_evidence),
        "missing_evidence_types": missing_evidence,
        "clause_results": [
            {
                "clause": clause,
                "status": result.status,
                "reason": result.reason,
                "matching_record_ids": list(
                    unique_sorted(
                        record_id
                        for match in _selected_matches_for_clause(selected, result)
                        for record_id in match.record_ids
                    )
                ),
                "candidate_match_count": len(result.matches),
                "candidate_match_digest": digest_object(
                    [
                        {
                            "tick": match.tick,
                            "bindings": match.bindings,
                            "record_ids": match.record_ids,
                        }
                        for match in result.matches
                    ]
                ),
            }
            for clause, result in zip(clauses, clause_results)
        ],
        "temporal_order": dict(stage["temporal_order"]),
        "temporal_interpretation": "physical_evidence_tick_strictly_after_previous_stage",
    }


def _candidate_selected_match_sets(
    policy: str,
    clause_results: Sequence[_ClauseEvaluation],
    *,
    minimum_bindings: int,
    previous: Mapping[str, Any] | None,
    event_family_id: str,
    binding_projection: Mapping[str, str] | None,
) -> list[tuple[_EvidenceMatch, ...]]:
    previous_tick = (
        previous.get("supporting_tick") if isinstance(previous, Mapping) else None
    )

    def eligible(matches: Sequence[_EvidenceMatch]) -> list[_EvidenceMatch]:
        rows = list(matches)
        if not isinstance(previous_tick, int):
            return rows
        return [
            match
            for match in rows
            if isinstance(match.tick, int) and match.tick > previous_tick
        ]

    selected_sets: list[tuple[_EvidenceMatch, ...]] = []
    if policy == "all_of":
        per_clause: list[list[_EvidenceMatch]] = []
        for result in clause_results:
            if result.status != PASS:
                return []
            matches = eligible(result.matches)
            if not matches:
                return []
            per_clause.append(matches)
        for selected in product(*per_clause):
            selected_tuple = tuple(selected)
            if not _matches_pairwise_binding_compatible(selected_tuple):
                continue
            if not _selected_matches_chain_compatible(
                selected_tuple,
                previous,
                event_family_id,
                binding_projection,
            ):
                continue
            selected_sets.append(selected_tuple)
    elif policy in {"single", "any_of"}:
        passing = [
            match
            for result in clause_results
            if result.status == PASS
            for match in eligible(result.matches)
        ]
        passing = sorted(passing, key=_match_sort_key)
        if minimum_bindings <= 1:
            selected_sets = [
                (match,)
                for match in passing
                if _selected_matches_chain_compatible(
                    (match,),
                    previous,
                    event_family_id,
                    binding_projection,
                )
            ]
        else:
            for selected in combinations(passing, minimum_bindings):
                signatures = {
                    tuple(sorted(match.bindings.items()))
                    for match in selected
                    if match.bindings
                }
                if len(signatures) < minimum_bindings:
                    continue
                if not _selected_matches_chain_compatible(
                    selected,
                    previous,
                    event_family_id,
                    binding_projection,
                ):
                    continue
                selected_sets.append(tuple(selected))
    else:
        raise StageAcceptanceError(f"unsupported clause_policy: {policy}")
    by_key = {
        _selected_matches_sort_key(selected): selected for selected in selected_sets
    }
    return [by_key[key] for key in sorted(by_key)]


def _selected_matches_chain_compatible(
    selected: Sequence[_EvidenceMatch],
    previous: Mapping[str, Any] | None,
    event_family_id: str,
    binding_projection: Mapping[str, str] | None = None,
) -> bool:
    previous_tick = (
        previous.get("supporting_tick") if isinstance(previous, Mapping) else None
    )
    require_binding_continuity = _requires_temporal_binding_continuity(
        previous, event_family_id
    )
    if isinstance(previous_tick, int):
        if any(
            not isinstance(match.tick, int) or match.tick <= previous_tick
            for match in selected
        ):
            return False
        if require_binding_continuity and not _bindings_continue(
            previous.get("binding_sets", ()), selected, binding_projection
        ):
            return False
        if not _event_identity_continuous(previous, selected, event_family_id):
            return False
    return True


def _requires_temporal_binding_continuity(
    previous: Mapping[str, Any] | None,
    event_family_id: str,
) -> bool:
    return (
        isinstance(previous, Mapping)
        and previous.get("event_family_id") == event_family_id
        and bool(previous.get("binding_sets"))
    )


def _selected_event_ids(selected: Sequence[_EvidenceMatch]) -> set[str]:
    return {
        str(match.detail["event_id"])
        for match in selected
        if isinstance(match.detail.get("event_id"), str)
        and match.detail.get("event_id")
    }


def _event_identity_continuous(
    previous: Mapping[str, Any] | None,
    selected: Sequence[_EvidenceMatch],
    event_family_id: str,
) -> bool:
    if (
        not isinstance(previous, Mapping)
        or previous.get("event_family_id") != event_family_id
    ):
        return True
    prior_event_ids = {
        str(event_id)
        for event_id in previous.get("selected_event_ids", ())
        if isinstance(event_id, str) and event_id
    }
    current_event_ids = _selected_event_ids(selected)
    return (
        not prior_event_ids
        or not current_event_ids
        or bool(prior_event_ids & current_event_ids)
    )


def _matches_pairwise_binding_compatible(selected: Sequence[_EvidenceMatch]) -> bool:
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            if (
                left.bindings
                and right.bindings
                and not _bindings_compatible(left.bindings, right.bindings)
            ):
                return False
    return True


def _selected_matches_for_clause(
    selected: Sequence[_EvidenceMatch],
    result: _ClauseEvaluation,
) -> tuple[_EvidenceMatch, ...]:
    return tuple(
        match
        for match in selected
        if any(match == candidate for candidate in result.matches)
    )


def _selected_matches_sort_key(selected: Sequence[_EvidenceMatch]) -> tuple[Any, ...]:
    return tuple(_match_sort_key(match) for match in selected)


def _stage_result_sort_key(stage: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        stage.get("supporting_tick") is None,
        stage.get("supporting_tick")
        if isinstance(stage.get("supporting_tick"), int)
        else 10**12,
        tuple(tuple(sorted(item.items())) for item in stage.get("binding_sets", ())),
        tuple(stage.get("supporting_record_ids", ())),
    )


def _select_stage_matches(
    policy: str,
    clause_results: Sequence[_ClauseEvaluation],
    *,
    minimum_bindings: int,
    previous: Mapping[str, Any] | None,
    event_family_id: str,
    binding_projection: Mapping[str, str] | None,
) -> tuple[list[_EvidenceMatch], str, str]:
    previous_tick = (
        previous.get("supporting_tick") if isinstance(previous, Mapping) else None
    )

    def eligible(matches: Sequence[_EvidenceMatch]) -> list[_EvidenceMatch]:
        rows = list(matches)
        if not isinstance(previous_tick, int):
            return rows
        return [
            match
            for match in rows
            if isinstance(match.tick, int) and match.tick > previous_tick
        ]

    selected: list[_EvidenceMatch] = []
    if policy == "all_of":
        for result in clause_results:
            matches = eligible(result.matches)
            if result.status == FAIL:
                return [], FAIL, result.reason
            if result.status == UNKNOWN or not matches:
                return [], UNKNOWN, result.reason or "required_clause_unknown"
            selected.append(matches[0])
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                if (
                    left.bindings
                    and right.bindings
                    and not _bindings_compatible(
                        left.bindings,
                        right.bindings,
                    )
                ):
                    return [], FAIL, "all_of_clause_binding_discontinuity"
    elif policy in {"single", "any_of"}:
        passing = [
            match
            for result in clause_results
            if result.status == PASS
            for match in eligible(result.matches)
        ]
        passing.sort(key=_match_sort_key)
        for match in passing:
            signature = tuple(sorted(match.bindings.items()))
            if signature and any(
                tuple(sorted(item.bindings.items())) == signature for item in selected
            ):
                continue
            selected.append(match)
            if len(selected) >= minimum_bindings:
                break
        if len(selected) < minimum_bindings:
            if any(result.status == UNKNOWN for result in clause_results):
                return selected, UNKNOWN, "declared_or_observed_semantic_gap"
            failing = next(
                (result for result in clause_results if result.status == FAIL),
                None,
            )
            if failing is not None:
                return selected, FAIL, failing.reason
            if passing and previous is not None:
                return selected, FAIL, "temporal_order_or_binding_cardinality_not_met"
            return selected, FAIL, "exact_acceptance_record_not_observed"
    else:
        raise StageAcceptanceError(f"unsupported clause_policy: {policy}")

    if isinstance(previous_tick, int):
        if any(
            not isinstance(match.tick, int) or match.tick <= previous_tick
            for match in selected
        ):
            return [], FAIL, "physical_evidence_not_strictly_after_previous_stage"
        if _requires_temporal_binding_continuity(
            previous, event_family_id
        ) and not _bindings_continue(
            previous.get("binding_sets", ()), selected, binding_projection
        ):
            return [], FAIL, "participant_binding_discontinuity"
        if not _event_identity_continuous(previous, selected, event_family_id):
            return [], FAIL, "event_identity_discontinuity"
    return selected, PASS, "exact_contract_clause_satisfied"


def _evaluate_clause(
    clause: Mapping[str, Any], index: _AcceptanceIndex
) -> _ClauseEvaluation:
    if clause.get("api_availability") == "missing":
        return _ClauseEvaluation(
            UNKNOWN, str(clause.get("gap_reason") or "declared_api_gap"), ()
        )
    event_phase = clause.get("event_phase")
    if isinstance(event_phase, str) and event_phase not in SUPPORTED_EVENT_PHASES:
        return _ClauseEvaluation(FAIL, "unsupported_event_phase", ())
    kind = str(clause.get("acceptance_kind") or "")
    if kind == "predicate_transition":
        return _predicate_transition_clause(clause, index)
    if kind == "event_occurrence":
        return _event_occurrence_clause(clause, index)
    if kind == "event_outcome":
        return _event_outcome_clause(clause, index)
    raise StageAcceptanceError(f"unsupported acceptance_kind: {kind}")


def _predicate_transition_clause(
    clause: Mapping[str, Any],
    index: _AcceptanceIndex,
) -> _ClauseEvaluation:
    predicate_id = _resolve_clause_predicate(clause, "predicate_id")
    direction = str(clause["direction"])
    matches = [
        match
        for row in index.transitions
        if row.get("predicate_id") == predicate_id
        and row.get("direction") == direction
        and _record_template_matches(clause, row, index)
        for match in [_transition_evidence_match(row, index)]
        if match is not None
    ]
    if matches:
        return _ClauseEvaluation(
            PASS,
            "exact_predicate_transition_observed",
            tuple(sorted(matches, key=_match_sort_key)),
        )
    status = UNKNOWN if index.predicate_is_unknown(predicate_id) else FAIL
    return _ClauseEvaluation(
        status,
        "predicate_truth_unknown" if status == UNKNOWN else "transition_not_observed",
        (),
    )


def _event_occurrence_clause(
    clause: Mapping[str, Any],
    index: _AcceptanceIndex,
) -> _ClauseEvaluation:
    event_type = str(clause["event_type_id"])
    event_family_id = str(clause["event_family_id"])
    predicate_id = _resolve_clause_predicate(clause, "trigger_predicate_id")
    direction = str(clause["direction"])
    event_phase = str(clause.get("event_phase") or "")
    if event_phase not in SUPPORTED_EVENT_PHASES:
        return _ClauseEvaluation(FAIL, "unsupported_event_phase", ())
    matches: list[_EvidenceMatch] = []
    for event in index.occurrences:
        if (
            event.get("event_type_id") != event_type
            or event.get("event_family_id") != event_family_id
            or event.get("trigger_predicate_id") != predicate_id
            or not _record_template_matches(clause, event, index)
        ):
            continue
        transition_matches = _phase_transition_matches(
            event,
            event_phase=event_phase,
            predicate_id=predicate_id,
            direction=direction,
            index=index,
        )
        if not transition_matches:
            continue
        bindings = _bindings(event)
        for phase_match in transition_matches:
            evidence_types = set(phase_match.evidence_types)
            evidence_types.add("event_occurrence_record")
            if bindings:
                evidence_types.add("participant_identity")
            if event.get("source_refs"):
                evidence_types.add("source_refs")
            matches.append(
                _EvidenceMatch(
                    tick=phase_match.tick,
                    bindings=bindings,
                    record_ids=tuple(
                        unique_sorted([str(event["event_id"]), *phase_match.record_ids])
                    ),
                    evidence_types=frozenset(evidence_types),
                    detail={
                        "event_id": event["event_id"],
                        "event_type_id": event_type,
                        "event_phase": event_phase,
                        **phase_match.detail,
                    },
                )
            )
    if matches:
        return _ClauseEvaluation(
            PASS,
            "exact_event_occurrence_observed",
            tuple(sorted(matches, key=_match_sort_key)),
        )
    status = UNKNOWN if index.predicate_is_unknown(predicate_id) else FAIL
    return _ClauseEvaluation(
        status,
        "occurrence_trigger_unknown"
        if status == UNKNOWN
        else "event_occurrence_not_observed",
        (),
    )


def _event_outcome_clause(
    clause: Mapping[str, Any],
    index: _AcceptanceIndex,
) -> _ClauseEvaluation:
    event_type = str(clause["event_type_id"])
    terminal_status = str(clause["terminal_status"])
    required_predicates = [
        _resolve_predicate_template(str(item), _clause_template_parameters(clause))
        for item in clause["required_terminal_predicate_ids"]
    ]
    matches: list[_EvidenceMatch] = []
    for outcome in index.outcomes:
        if (
            outcome.get("event_type_id") != event_type
            or outcome.get("status") != terminal_status
        ):
            continue
        terminal_tick = outcome.get("terminal_tick")
        if not isinstance(terminal_tick, int):
            continue
        event = index.occurrence_by_id.get(str(outcome.get("event_id", "")))
        if event is None or event.get("event_type_id") != event_type:
            continue
        bindings = _bindings(event)
        outcome_truth_ids = _string_id_set(outcome.get("supporting_truth_ids"))
        terminal_phase_ids = _terminal_phase_transition_ids(event)
        lifecycle_entry_ids = _terminal_hold_entry_transition_ids(outcome)
        valid_lifecycle_entry_ids = tuple(
            sorted(terminal_phase_ids & lifecycle_entry_ids)
        )
        if not valid_lifecycle_entry_ids:
            continue
        terminal_truth: list[Mapping[str, Any]] = []
        for predicate_id in required_predicates:
            compatible = [
                row
                for row in index.truth_by_predicate.get(predicate_id, ())
                if row.get("tick") == terminal_tick
                and row.get("value") == "true"
                and str(row.get("truth_id", "")) in outcome_truth_ids
                and _bindings_compatible(bindings, _bindings(row))
            ]
            if not compatible:
                break
            terminal_truth.append(compatible[0])
        else:
            evidence_types = {
                "event_occurrence_record",
                "event_outcome_record",
                "predicate_truth_after",
                "tuple_binding_continuity",
            }
            if bindings:
                evidence_types.add("participant_identity")
            if all(
                (row.get("evidence") or {}).get("source_refs") for row in terminal_truth
            ):
                evidence_types.add("source_refs")
            if index.has_numeric_state(terminal_tick, bindings) or any(
                _contains_number((row.get("evidence") or {}).get("observations"))
                for row in terminal_truth
            ):
                evidence_types.add("terminal_numeric_state")
            matches.append(
                _EvidenceMatch(
                    tick=terminal_tick,
                    bindings=bindings,
                    record_ids=tuple(
                        unique_sorted(
                            [str(event["event_id"]), str(outcome["outcome_id"])]
                            + [str(row["truth_id"]) for row in terminal_truth]
                            + list(valid_lifecycle_entry_ids)
                        )
                    ),
                    evidence_types=frozenset(evidence_types),
                    detail={
                        "event_id": event["event_id"],
                        "outcome_id": outcome["outcome_id"],
                        "terminal_phase_transition_ids": valid_lifecycle_entry_ids,
                    },
                )
            )
    if matches:
        return _ClauseEvaluation(
            PASS,
            "exact_event_outcome_observed",
            tuple(sorted(matches, key=_match_sort_key)),
        )
    status = (
        UNKNOWN
        if any(index.predicate_is_unknown(item) for item in required_predicates)
        else FAIL
    )
    return _ClauseEvaluation(
        status,
        "terminal_predicate_unknown"
        if status == UNKNOWN
        else "event_outcome_not_observed",
        (),
    )


def _transition_evidence_match(
    transition: Mapping[str, Any],
    index: _AcceptanceIndex,
) -> _EvidenceMatch | None:
    evidence_ids = [
        str(item)
        for item in transition.get("evidence_truth_ids", ())
        if isinstance(item, str)
    ]
    if len(evidence_ids) != 2:
        return None
    before = index.truth_by_id.get(evidence_ids[0])
    after = index.truth_by_id.get(evidence_ids[1])
    if before is None or after is None:
        return None
    from_tick = transition.get("from_tick")
    to_tick = transition.get("to_tick")
    if (
        not isinstance(from_tick, int)
        or not isinstance(to_tick, int)
        or to_tick - from_tick != 5
    ):
        return None
    if before.get("tick") != from_tick or after.get("tick") != to_tick:
        return None
    if before.get("predicate_id") != transition.get("predicate_id") or after.get(
        "predicate_id"
    ) != transition.get("predicate_id"):
        return None
    if before.get("tuple_id") != transition.get("tuple_id") or after.get(
        "tuple_id"
    ) != transition.get("tuple_id"):
        return None
    expected = (
        ("false", "true")
        if transition.get("direction") == "rising"
        else ("true", "false")
    )
    if (before.get("value"), after.get("value")) != expected:
        return None
    bindings = _bindings(transition)
    if bindings != _bindings(before) or bindings != _bindings(after):
        return None
    evidence_types = {
        "predicate_truth_before",
        "predicate_truth_after",
        "strict_adjacent_transition",
        "tuple_binding_continuity",
    }
    if (before.get("evidence") or {}).get("source_refs") and (
        after.get("evidence") or {}
    ).get("source_refs"):
        evidence_types.add("source_refs")
    if index.has_numeric_state(from_tick, bindings) or _contains_number(
        (before.get("evidence") or {}).get("observations")
    ):
        evidence_types.add("numeric_state_before")
    if index.has_numeric_state(to_tick, bindings) or _contains_number(
        (after.get("evidence") or {}).get("observations")
    ):
        evidence_types.add("numeric_state_after")
    return _EvidenceMatch(
        tick=to_tick,
        bindings=bindings,
        record_ids=(str(transition["transition_id"]), *evidence_ids),
        evidence_types=frozenset(evidence_types),
        detail={"transition_id": transition["transition_id"]},
    )


def _phase_transition_matches(
    event: Mapping[str, Any],
    *,
    event_phase: str,
    predicate_id: str,
    direction: str,
    index: _AcceptanceIndex,
) -> list[_EvidenceMatch]:
    raw_phases = event.get("event_phases")
    raw_supporting_ids = event.get("supporting_transition_ids")
    if (
        not isinstance(raw_phases, Sequence)
        or isinstance(raw_phases, (str, bytes))
        or not isinstance(raw_supporting_ids, Sequence)
        or isinstance(raw_supporting_ids, (str, bytes))
    ):
        return []
    supporting_ids = {
        str(item) for item in raw_supporting_ids if isinstance(item, str) and item
    }
    matches: list[_EvidenceMatch] = []
    for phase in raw_phases:
        if not isinstance(phase, Mapping):
            continue
        if (
            phase.get("phase_kind") != event_phase
            or phase.get("predicate_id") != predicate_id
            or phase.get("direction") != direction
        ):
            continue
        transition_id = phase.get("transition_id")
        if not isinstance(transition_id, str) or transition_id not in supporting_ids:
            continue
        transition = index.transition_by_id.get(transition_id)
        if transition is None:
            continue
        if not _phase_matches_transition(phase, transition):
            continue
        match = _transition_evidence_match(transition, index)
        if match is not None:
            matches.append(match)
    return matches


def _string_id_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(item) for item in value if isinstance(item, str) and item}


def _terminal_phase_transition_ids(event: Mapping[str, Any]) -> set[str]:
    phases = event.get("event_phases")
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        return set()
    return {
        str(phase["transition_id"])
        for phase in phases
        if isinstance(phase, Mapping)
        and phase.get("phase_kind") == "terminal"
        and isinstance(phase.get("transition_id"), str)
        and phase["transition_id"]
    }


def _terminal_hold_entry_transition_ids(outcome: Mapping[str, Any]) -> set[str]:
    entries = outcome.get("lifecycle_phase_evidence")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return set()
    return {
        str(entry["entry_transition_id"])
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("phase_kind") == "terminal_hold"
        and isinstance(entry.get("entry_transition_id"), str)
        and entry["entry_transition_id"]
    }


def _phase_matches_transition(
    phase: Mapping[str, Any],
    transition: Mapping[str, Any],
) -> bool:
    return (
        phase.get("transition_id") == transition.get("transition_id")
        and phase.get("predicate_id") == transition.get("predicate_id")
        and phase.get("predicate_tuple_id") == transition.get("tuple_id")
        and phase.get("predicate_bindings") == _bindings(transition)
        and phase.get("direction") == transition.get("direction")
        and phase.get("tick") == transition.get("to_tick")
    )


def _bindings_continue(
    previous_sets: Any,
    matches: Sequence[_EvidenceMatch],
    projection: Mapping[str, str] | None,
) -> bool:
    if not isinstance(previous_sets, Sequence) or isinstance(
        previous_sets, (str, bytes)
    ):
        return True
    prior = [item for item in previous_sets if isinstance(item, Mapping)]
    if not prior:
        return True
    if projection is not None:
        current_roles = set(projection)
        for match in matches:
            if not current_roles <= set(match.bindings):
                return False
            if not any(
                all(
                    match.bindings[current_role] == previous.get(previous_role)
                    for current_role, previous_role in projection.items()
                )
                for previous in prior
            ):
                return False
        return True
    for match in matches:
        if not match.bindings or not any(
            match.bindings == previous for previous in prior
        ):
            return False
    return True


def _binding_projection(stage: Mapping[str, Any]) -> dict[str, str] | None:
    raw = stage.get("binding_projection_from_previous")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise StageAcceptanceError("binding projection must be an object")
    return {str(current): str(previous) for current, previous in raw.items()}


def _validate_binding_projection(
    stage: Mapping[str, Any],
    stages: Mapping[str, Any],
    stage_key: str,
) -> None:
    projection = _binding_projection(stage)
    if projection is None:
        return
    if (
        not projection
        or any(not current or not previous for current, previous in projection.items())
        or len(set(projection.values())) != len(projection)
    ):
        raise StageAcceptanceError(
            f"binding projection must be a non-empty one-to-one role map: {stage_key}"
        )
    stage_index = int(stage["stage_index"])
    if stage_index == 0:
        raise StageAcceptanceError(
            f"root stage cannot declare a binding projection: {stage_key}"
        )
    previous_key = f"{stage['epi_id']}::{stage_index - 1}"
    previous = stages.get(previous_key)
    if not isinstance(previous, Mapping) or previous.get(
        "event_family_id"
    ) != stage.get("event_family_id"):
        raise StageAcceptanceError(
            "binding projection requires a preceding stage from the same event "
            f"family: {stage_key}"
        )
    clauses = stage.get("acceptance_clauses", ())
    if len(clauses) != 1 or clauses[0].get("acceptance_kind") != "predicate_transition":
        raise StageAcceptanceError(
            f"binding projection is only valid for one predicate transition: {stage_key}"
        )


def _clauses_for_stage(stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    event_phase = stage.get("event_phase")
    clauses = [dict(clause) for clause in stage["acceptance_clauses"]]
    if isinstance(event_phase, str) and event_phase:
        for clause in clauses:
            clause.setdefault("event_phase", event_phase)
    return clauses


def _validate_clause_templates(clause: Mapping[str, Any], stage_key: str) -> None:
    parameters = _clause_template_parameters(clause)
    unsupported = sorted(set(parameters) - {"mode", "pair_type", "facility_kind"})
    if unsupported:
        raise StageAcceptanceError(
            f"unsupported template parameters at {stage_key}: {unsupported}"
        )
    for field in (
        "predicate_id",
        "trigger_predicate_id",
    ):
        if field in clause:
            _resolve_predicate_template(str(clause[field]), parameters)
    terminal_predicates = clause.get("required_terminal_predicate_ids", ())
    if isinstance(terminal_predicates, Sequence) and not isinstance(
        terminal_predicates, (str, bytes)
    ):
        for predicate_id in terminal_predicates:
            _resolve_predicate_template(str(predicate_id), parameters)


def _clause_template_parameters(clause: Mapping[str, Any]) -> dict[str, str]:
    raw = clause.get("template_parameters")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise StageAcceptanceError("template_parameters must be an object")
    result: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
            raise StageAcceptanceError(
                "template parameter names and values must be scalar"
            )
        result[key] = _normalize_template_value(key, value)
    return result


def _resolve_clause_predicate(clause: Mapping[str, Any], field: str) -> str:
    return _resolve_predicate_template(
        str(clause[field]), _clause_template_parameters(clause)
    )


def _resolve_predicate_template(
    predicate_id: str,
    parameters: Mapping[str, str],
) -> str:
    resolved = predicate_id
    for key, value in sorted(parameters.items()):
        resolved = resolved.replace("{" + key + "}", value)
    if "{" in resolved or "}" in resolved:
        raise StageAcceptanceError(
            f"predicate template lacks a concrete binding: {predicate_id}"
        )
    return resolved


def _record_template_matches(
    clause: Mapping[str, Any],
    row: Mapping[str, Any],
    index: _AcceptanceIndex,
) -> bool:
    expected = {
        key: value
        for key, value in _clause_template_parameters(clause).items()
        if key in {"pair_type", "facility_kind"}
    }
    if not expected:
        return True
    actual = _record_template_parameters(row, index)
    return all(actual.get(key) == value for key, value in expected.items())


def _record_template_parameters(
    row: Mapping[str, Any],
    index: _AcceptanceIndex,
) -> dict[str, str]:
    actual: dict[str, str] = {}
    raw = row.get("template_parameters")
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                actual[key] = _normalize_template_value(key, value)
    subtype = row.get("event_subtype")
    event_type = str(row.get("event_type_id") or "")
    if isinstance(subtype, str):
        if event_type in {
            "uav_separation_conflict",
            "uav_pedestrian_conflict",
            "uav_vehicle_conflict",
            "vehicle_pedestrian_collision",
            "vehicle_vehicle_collision",
        }:
            actual.setdefault(
                "pair_type", _normalize_template_value("pair_type", subtype)
            )
        elif event_type in {"charger_unavailable", "pad_contention"}:
            actual.setdefault(
                "facility_kind", _normalize_template_value("facility_kind", subtype)
            )
    support_truth_ids = (
        row.get("evidence_truth_ids") or row.get("supporting_truth_ids") or ()
    )
    if isinstance(support_truth_ids, Sequence) and not isinstance(
        support_truth_ids, (str, bytes)
    ):
        for truth_id in support_truth_ids:
            truth = index.truth_by_id.get(str(truth_id))
            if not truth:
                continue
            truth_parameters = truth.get("template_parameters")
            if isinstance(truth_parameters, Mapping):
                for key, value in truth_parameters.items():
                    if isinstance(key, str) and isinstance(
                        value, (str, int, float, bool)
                    ):
                        actual.setdefault(key, _normalize_template_value(key, value))
            for observation in (truth.get("evidence") or {}).get("observations", ()):
                if not isinstance(observation, Mapping):
                    continue
                values = observation.get("value")
                if not isinstance(values, Mapping):
                    continue
                for key in ("pair_type", "facility_kind"):
                    value = values.get(key)
                    if isinstance(value, str):
                        actual.setdefault(key, _normalize_template_value(key, value))
    supporting_transition_ids = row.get("supporting_transition_ids", ())
    if isinstance(supporting_transition_ids, Sequence) and not isinstance(
        supporting_transition_ids, (str, bytes)
    ):
        for transition_id in supporting_transition_ids:
            transition = index.transition_by_id.get(str(transition_id))
            if transition is None or transition is row:
                continue
            for key, value in _record_template_parameters(transition, index).items():
                actual.setdefault(key, value)
    return actual


def _normalize_template_value(key: str, value: Any) -> str:
    text = str(value or "").strip().lower()
    if key == "facility_kind":
        if text in {"landing_pad", "vertipad", "emergency_pad"}:
            return "pad"
        if text == "charging_station":
            return "charger"
    return text


def _bindings_compatible(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    return bool(left) and left == right


def _bindings(row: Mapping[str, Any]) -> dict[str, str]:
    raw = row.get("bindings")
    if not isinstance(raw, Mapping):
        return {}
    normalized: dict[str, str] = {}
    for role, entity_id in raw.items():
        if not isinstance(entity_id, str):
            continue
        normalized[str(role)] = entity_id
    return dict(sorted(normalized.items()))


def _deduplicate_bindings(values: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    by_key = {
        tuple(sorted(item.items())): dict(sorted(item.items())) for item in values
    }
    return [by_key[key] for key in sorted(by_key)]


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Mapping):
        return any(_contains_number(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_number(item) for item in value)
    return False


def _episode_matches(row: Mapping[str, Any], episode_id: str) -> bool:
    return row.get("episode_id") in {None, "", episode_id}


def _match_sort_key(match: _EvidenceMatch) -> tuple[Any, ...]:
    return (
        match.tick is None,
        match.tick if isinstance(match.tick, int) else 10**12,
        tuple(sorted(match.bindings.items())),
        match.record_ids,
    )


__all__ = [
    "StageAcceptanceError",
    "evaluate_epi_stage_acceptance",
    "validate_stage_acceptance_contract",
]
