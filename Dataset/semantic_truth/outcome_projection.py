"""Validate one event outcome before projecting a recovery chain node."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def recovery_tick_for_chain(outcome: Mapping[str, Any], *,
                            window_start_tick: int, window_end_tick: int) -> int | None:
    """Return an evidenced recovery tick; keep other valid outcomes as outcomes."""
    if (type(window_start_tick) is not int or type(window_end_tick) is not int or
            window_start_tick > window_end_tick):
        raise ValueError("event recovery projection has invalid window bounds")
    status = outcome.get("status")
    lifecycle = outcome.get("lifecycle_status")
    tick = outcome.get("terminal_tick")
    evidence = outcome.get("lifecycle_phase_evidence")
    if not isinstance(evidence, list):
        raise ValueError(f"event outcome lacks lifecycle evidence list: {outcome.get('outcome_id')}")
    if status == "pending":
        if (tick is not None or evidence or lifecycle not in {
                "terminal_hold_not_observed", "terminal_transition_not_observed"}):
            raise ValueError(f"pending event outcome has terminal evidence: {outcome.get('outcome_id')}")
        return None
    if status == "right_censored":
        if (tick is not None or evidence or lifecycle != "observed_until_window_end" or
                type(outcome.get("window_censor_tick")) is not int or
                outcome["window_censor_tick"] != window_end_tick):
            raise ValueError(f"right-censored event outcome has invalid window evidence: {outcome.get('outcome_id')}")
        return None
    if status != "succeeded" or lifecycle not in {
            "recovery_predicate_hold", "terminal_predicate_hold"}:
        raise ValueError(f"event outcome has unsupported terminal status: {outcome.get('outcome_id')}")
    if type(tick) is not int or not window_start_tick <= tick <= window_end_tick:
        raise ValueError(f"event outcome terminal tick is outside its window: {outcome.get('outcome_id')}")
    if len(evidence) != 1 or not isinstance(evidence[0], Mapping):
        raise ValueError(f"event outcome lacks one terminal hold proof: {outcome.get('outcome_id')}")
    proof = evidence[0]
    expected_value = "false" if lifecycle == "recovery_predicate_hold" else "true"
    proof_refs = proof.get("supporting_truth_ids")
    outcome_refs = outcome.get("supporting_truth_ids")
    if (proof.get("phase_kind") != "terminal_hold" or proof.get("value") != expected_value or
            type(proof.get("end_tick")) is not int or proof["end_tick"] != tick or
            not isinstance(outcome_refs, list) or not outcome_refs or
            any(not isinstance(ref, str) or not ref for ref in outcome_refs) or
            not isinstance(proof_refs, list) or not proof_refs or
            any(not isinstance(ref, str) or not ref for ref in proof_refs)):
        raise ValueError(f"event outcome terminal proof conflicts with status: {outcome.get('outcome_id')}")
    return tick if lifecycle == "recovery_predicate_hold" else None


def source_index(rows: Sequence[Mapping[str, Any]], id_key: str) -> dict[str, Mapping[str, Any]]:
    """Index full source records without silently replacing duplicate identities."""
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = row.get(id_key)
        if not isinstance(record_id, str) or not record_id or record_id in indexed:
            raise ValueError(f"invalid or duplicate {id_key}: {record_id!r}")
        indexed[record_id] = row
    return indexed


def validate_occurrence_sources(
    occurrence: Mapping[str, Any], *,
    truth_by_id: Mapping[str, Mapping[str, Any]],
    transition_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Resolve an occurrence's phase transitions and their measured truth."""
    event_id = occurrence.get("event_id")
    phases = occurrence.get("event_phases")
    truth_refs = occurrence.get("supporting_truth_ids")
    transition_refs = occurrence.get("supporting_transition_ids")
    if (not isinstance(phases, list) or not phases or
            not isinstance(truth_refs, list) or not truth_refs or
            not isinstance(transition_refs, list) or not transition_refs):
        raise ValueError(f"event occurrence lacks source evidence: {event_id}")
    onset_phases = [phase for phase in phases if phase.get("phase_kind") == "onset"]
    if (len(onset_phases) != 1 or
            type(occurrence.get("trigger_tick")) is not int or
            onset_phases[0].get("tick") != occurrence["trigger_tick"]):
        raise ValueError(f"event occurrence onset differs from trigger: {event_id}")
    phase_ids = [phase.get("transition_id") for phase in phases]
    phases_by_transition: dict[str, list[str]] = {}
    for phase in phases:
        if phase.get("phase_kind") not in {"onset", "escalation_support", "terminal"}:
            raise ValueError(f"event occurrence has unknown phase kind: {event_id}")
        phases_by_transition.setdefault(phase.get("transition_id"), []).append(
            phase.get("phase_kind"))
    if (any(len(kinds) > 1 and sorted(kinds) != ["onset", "terminal"]
            for kinds in phases_by_transition.values()) or
            set(phase_ids) != set(transition_refs) or
            len(transition_refs) != len(set(transition_refs)) or
            len(truth_refs) != len(set(truth_refs))):
        raise ValueError(f"event occurrence phase support differs from source: {event_id}")
    evidenced_truth: set[str] = set()
    for phase in phases:
        transition = transition_by_id.get(phase["transition_id"])
        if (transition is None or
                phase.get("predicate_id") != transition.get("predicate_id") or
                phase.get("predicate_tuple_id") != transition.get("tuple_id") or
                phase.get("predicate_bindings") != transition.get("bindings") or
                phase.get("direction") != transition.get("direction") or
                type(phase.get("tick")) is not int or
                phase["tick"] != transition.get("to_tick")):
            raise ValueError(f"event occurrence phase does not resolve: {event_id}")
        endpoint_refs = transition.get("evidence_truth_ids")
        if not isinstance(endpoint_refs, list) or len(endpoint_refs) != 2:
            raise ValueError(f"event transition lacks endpoint truth: {event_id}")
        for ref, tick_key, value_key in zip(endpoint_refs,
                                            ("from_tick", "to_tick"),
                                            ("from_value", "to_value")):
            truth = truth_by_id.get(ref)
            if (truth is None or truth.get("predicate_id") != transition["predicate_id"] or
                    truth.get("tuple_id") != transition["tuple_id"] or
                    truth.get("bindings") != transition["bindings"] or
                    truth.get("tick") != transition[tick_key] or
                    truth.get("value") != transition[value_key]):
                raise ValueError(f"event transition endpoint truth does not resolve: {event_id}: {ref}")
            evidenced_truth.add(ref)
    if set(truth_refs) != evidenced_truth:
        raise ValueError(f"event occurrence truth support differs from phases: {event_id}")


def validated_recovery_tick(
    outcome: Mapping[str, Any], *, occurrence: Mapping[str, Any] | None,
    truth_by_id: Mapping[str, Mapping[str, Any]],
    transition_by_id: Mapping[str, Mapping[str, Any]],
    window_start_tick: int, window_end_tick: int,
    formal_step_ticks: int = 5,
) -> int | None:
    """Project recovery only when the source event and hold proof resolve."""
    if occurrence is None or occurrence.get("event_id") != outcome.get("event_id"):
        raise ValueError(f"event outcome has no source occurrence: {outcome.get('outcome_id')}")
    if occurrence.get("event_family_id") != outcome.get("event_family_id"):
        raise ValueError(f"event outcome family differs from occurrence: {outcome.get('outcome_id')}")
    validate_occurrence_sources(
        occurrence, truth_by_id=truth_by_id, transition_by_id=transition_by_id)
    outcome_refs = outcome.get("supporting_truth_ids")
    if (not isinstance(outcome_refs, list) or
            len(outcome_refs) != len(set(outcome_refs)) or
            any(ref not in truth_by_id for ref in outcome_refs)):
        raise ValueError(f"event outcome has unresolved truth source: {outcome.get('outcome_id')}")
    tick = recovery_tick_for_chain(
        outcome, window_start_tick=window_start_tick, window_end_tick=window_end_tick)
    if tick is None:
        if outcome.get("status") == "pending":
            if outcome_refs or any(phase.get("phase_kind") == "terminal"
                                   for phase in occurrence["event_phases"]):
                raise ValueError(f"pending event claims terminal evidence: {outcome.get('outcome_id')}")
        return None
    if type(formal_step_ticks) is not int or formal_step_ticks <= 0:
        raise ValueError("event recovery projection has invalid formal step")
    if type(occurrence.get("trigger_tick")) is not int or occurrence["trigger_tick"] >= tick:
        raise ValueError(f"recovery does not follow event onset: {outcome.get('outcome_id')}")
    proof = outcome["lifecycle_phase_evidence"][0]
    start_tick = proof.get("start_tick")
    if type(start_tick) is not int or start_tick > tick or (tick - start_tick) % formal_step_ticks:
        raise ValueError(f"recovery hold interval is invalid: {outcome.get('outcome_id')}")
    terminal_phases = [phase for phase in occurrence.get("event_phases", [])
                       if phase.get("phase_kind") == "terminal"]
    entry_id = proof.get("entry_transition_id")
    transition = transition_by_id.get(entry_id)
    if (len(terminal_phases) != 1 or transition is None or
            terminal_phases[0].get("transition_id") != entry_id or
            transition.get("predicate_id") != proof.get("predicate_id") or
            transition.get("tuple_id") != proof.get("predicate_tuple_id") or
            transition.get("bindings") != proof.get("predicate_bindings") or
            transition.get("to_value") != "false" or
            type(transition.get("to_tick")) is not int or
            transition["to_tick"] != start_tick):
        raise ValueError(f"recovery transition proof does not resolve: {outcome.get('outcome_id')}")
    refs = proof["supporting_truth_ids"]
    expected_ticks = list(range(start_tick, tick + 1, formal_step_ticks))
    if len(refs) != len(expected_ticks) or len(refs) != len(set(refs)):
        raise ValueError(f"recovery hold lacks sampled truth coverage: {outcome.get('outcome_id')}")
    for truth_id, expected_tick in zip(refs, expected_ticks):
        truth = truth_by_id.get(truth_id)
        if (truth is None or truth.get("predicate_id") != proof.get("predicate_id") or
                truth.get("tuple_id") != proof.get("predicate_tuple_id") or
                truth.get("bindings") != proof.get("predicate_bindings") or
                truth.get("value") != "false" or
                type(truth.get("tick")) is not int or truth["tick"] != expected_tick):
            raise ValueError(f"recovery truth proof does not resolve: {outcome.get('outcome_id')}: {truth_id}")
    expected_refs = set(occurrence.get("supporting_truth_ids", [])) | set(refs)
    outcome_refs = outcome["supporting_truth_ids"]
    if (set(outcome_refs) != expected_refs or len(outcome_refs) != len(expected_refs) or
            any(truth_id not in truth_by_id for truth_id in outcome_refs)):
        raise ValueError(f"recovery outcome truth support differs from source: {outcome.get('outcome_id')}")
    return tick
