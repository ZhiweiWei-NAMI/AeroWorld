"""Resolve explicitly inherited L6 predicate evidence before graph writing."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from Dataset.world_model.graph.contract import REPO, iter_jsonl, read_json
from Dataset.world_model.graph.structure_schema import project_record_for_publication


def _indexed(rows: Any, id_field: str, source: Path) -> dict[str, tuple[int, dict]]:
    if not isinstance(rows, list):
        raise ValueError(f"{source}: historical {id_field} collection is not an array")
    result: dict[str, tuple[int, dict]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get(id_field), str) or not row[id_field]:
            raise ValueError(f"{source}: historical {id_field} missing at index {index}")
        native = row[id_field]
        if native in result:
            raise ValueError(f"{source}: repeated historical {id_field}: {native}")
        result[native] = index, row
    return result


def load_l6_history(
    branch_root: Path,
    manifest: dict,
    *,
    profile: str,
    seed: str,
    group: str,
    branch: str,
) -> dict[str, Any] | None:
    """Return the referenced pre-window closure; no graph output is opened here."""

    descriptor = (manifest.get("source_refs") or {}).get("semantic_history")
    chain_path = branch_root / "chain/nodes.jsonl"
    inherited_nodes = [
        row for _, row in iter_jsonl(chain_path)
        if row.get("kind") == "inherited_predicate_transition"
    ] if chain_path.exists() else []
    if descriptor is None:
        if inherited_nodes:
            raise ValueError(f"{branch_root}: inherited chain evidence lacks semantic_history")
        return None
    if not profile.startswith("L6-") or not isinstance(descriptor, dict) or set(descriptor) != {"path", "format", "scope"}:
        raise ValueError(f"{branch_root}: invalid L6 semantic_history descriptor")
    if descriptor["format"] != "gzip_json" or descriptor["scope"] != "pre_window_measured_evidence":
        raise ValueError(f"{branch_root}: unsupported semantic_history source")
    relative_path = Path(descriptor["path"])
    expected_parent = REPO / "design" / profile / seed / group / "history"
    history_path = (REPO / relative_path).resolve()
    if relative_path.is_absolute() or history_path.parent != expected_parent.resolve() or history_path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError(f"{branch_root}: semantic_history path is outside the declared history scope")
    if not history_path.is_file():
        raise ValueError(f"{branch_root}: semantic_history file is absent: {relative_path}")
    initial_path = branch_root / "objective/initial_events.json"
    initial = project_record_for_publication(read_json(initial_path))
    start = manifest["window"]["start_tick"]
    if initial.get("source") != descriptor or initial.get("tick") != start or initial.get("policy") != "inherit_original_onset_and_producer_memory":
        raise ValueError(f"{initial_path}: initial event inheritance differs from manifest")
    with gzip.open(history_path, "rt", encoding="utf-8") as stream:
        history = project_record_for_publication(json.load(stream))
    arm_key = f"{profile}/{seed}/{group}/{branch}"
    episode_id = manifest["episode_id"]
    scoped_episode = f"{episode_id}::{arm_key}"
    if (
        history.get("schema") != "l6_measured_pre_window_evidence"
        or history.get("episode_id") != episode_id
        or history.get("arm_key") != arm_key
        or history.get("end_exclusive_tick") != start
    ):
        raise ValueError(f"{history_path}: history identity or window boundary differs")
    objective = history.get("objective")
    if not isinstance(objective, dict):
        raise ValueError(f"{history_path}: historical objective is absent")
    truths = _indexed(objective.get("predicate_truth"), "truth_id", history_path)
    transitions = _indexed(objective.get("predicate_transitions"), "transition_id", history_path)
    for native, (_, row) in truths.items():
        if row.get("episode_id") != scoped_episode or type(row.get("tick")) is not int or not 0 <= row["tick"] < start:
            raise ValueError(f"{history_path}: historical truth scope differs: {native}")
    for native, (_, row) in transitions.items():
        if (row.get("episode_id") != scoped_episode or type(row.get("from_tick")) is not int
                or type(row.get("to_tick")) is not int or not 0 <= row["from_tick"] < row["to_tick"] < start):
            raise ValueError(f"{history_path}: historical transition scope differs: {native}")

    occurrence_path = branch_root / "objective/event_occurrences.jsonl"
    outcome_path = branch_root / "objective/event_outcomes.jsonl"
    local_truth_path = branch_root / "objective/predicate_truth.jsonl"
    local_transition_path = branch_root / "objective/predicate_transitions.jsonl"
    occurrences = [row for _, row in iter_jsonl(occurrence_path)] if occurrence_path.exists() else []
    outcomes = [row for _, row in iter_jsonl(outcome_path)] if outcome_path.exists() else []
    local_truth_ids = {row["truth_id"] for _, row in iter_jsonl(local_truth_path)} if local_truth_path.exists() else set()
    local_transition_ids = {row["transition_id"] for _, row in iter_jsonl(local_transition_path)} if local_transition_path.exists() else set()
    if local_truth_ids & truths.keys() or local_transition_ids & transitions.keys():
        raise ValueError(f"{branch_root}: local and inherited objective IDs overlap")
    occurrence_by_id = _indexed(occurrences, "event_id", occurrence_path)
    initial_events = _indexed(initial.get("events"), "event_id", initial_path)
    for native, (_, event) in initial_events.items():
        local = occurrence_by_id.get(native)
        if local is None or any(
            local[1].get(target) != event.get(source)
            for source, target in (("onset_tick", "trigger_tick"), ("bindings", "bindings"), ("event_family_id", "event_family_id"))
        ):
            raise ValueError(f"{initial_path}: inherited event differs from occurrence: {native}")
    initial_transitions = _indexed(initial.get("transitions"), "transition_id", initial_path)
    for native, (_, transition) in initial_transitions.items():
        if native not in transitions:
            raise ValueError(f"{initial_path}: inherited transition absent from history: {native}")
        archived = transitions[native][1]
        if {k: v for k, v in transition.items() if k != "source_refs"} != {k: v for k, v in archived.items() if k != "source_refs"}:
            raise ValueError(f"{initial_path}: inherited transition differs from history: {native}")
        if transition.get("source_refs") != [f"{relative_path.as_posix()}#{ref}" for ref in archived.get("source_refs", [])]:
            raise ValueError(f"{initial_path}: inherited transition source references differ: {native}")

    required_truth: set[str] = set()
    required_transition: set[str] = set(initial_transitions)
    for row in (*occurrences, *outcomes):
        required_truth.update(row.get("supporting_truth_ids") or [])
        required_transition.update(row.get("supporting_transition_ids") or [])
        for phase in (row.get("event_phases") or []) + (row.get("lifecycle_phase_evidence") or []):
            required_truth.update(phase.get("supporting_truth_ids") or [])
            if phase.get("transition_id"):
                required_transition.add(phase["transition_id"])
            if phase.get("entry_transition_id"):
                required_transition.add(phase["entry_transition_id"])
    inherited_node_ids = set()
    for row in inherited_nodes:
        native = row.get("node_id")
        if not isinstance(native, str) or native in inherited_node_ids or native not in transitions:
            raise ValueError(f"{chain_path}: invalid inherited transition node: {native}")
        archived = transitions[native][1]
        if row.get("evidence_ref") != f"{relative_path.as_posix()}#{native}" or row.get("tick") != archived["to_tick"]:
            raise ValueError(f"{chain_path}: inherited transition reference differs: {native}")
        inherited_node_ids.add(native)
        required_transition.add(native)
    selected_transitions = required_transition - local_transition_ids
    if selected_transitions != inherited_node_ids or not selected_transitions <= transitions.keys():
        raise ValueError(f"{branch_root}: inherited transition closure differs from chain nodes")
    inherited_relation_sources: set[str] = set()
    relation_path = branch_root / "chain/relations.jsonl"
    if selected_transitions and not relation_path.is_file():
        raise ValueError(f"{branch_root}: inherited transitions lack chain relations")
    if relation_path.is_file():
        for line, relation in iter_jsonl(relation_path):
            native = relation.get("source")
            if native not in selected_transitions:
                continue
            transition = transitions[native][1]
            event = initial_events.get(relation.get("target"))
            occurrence = occurrence_by_id.get(relation.get("target"))
            onset_phases = [phase for phase in occurrence[1].get("event_phases", [])
                            if phase.get("phase_kind") == "onset"] if occurrence else []
            if (relation.get("relation_type") != "predicate_onset_evidence"
                    or event is None or relation.get("tick") != transition["to_tick"]
                    or event[1].get("onset_tick") != transition["to_tick"]
                    or relation.get("causal_claim") is not False
                    or occurrence is None or len(onset_phases) != 1
                    or occurrence[1].get("trigger_predicate_id") != transition.get("predicate_id")
                    or occurrence[1].get("bindings") != transition.get("bindings")
                    or occurrence[1].get("rule_id") != relation.get("rule_id")
                    or transition.get("direction") != "rising"
                    or transition.get("from_value") != "false"
                    or transition.get("to_value") != "true"
                    or any(onset_phases[0].get(key) != expected for key, expected in (
                        ("transition_id", native), ("predicate_id", transition.get("predicate_id")),
                        ("predicate_tuple_id", transition.get("tuple_id")),
                        ("predicate_bindings", transition.get("bindings")),
                        ("direction", "rising"), ("tick", transition["to_tick"]),
                    ))):
                raise ValueError(f"{relation_path}:{line}: inherited onset relation differs from source")
            inherited_relation_sources.add(native)
    if inherited_relation_sources != selected_transitions:
        raise ValueError(f"{relation_path}: inherited transition has no onset relation")
    for native in selected_transitions:
        row = transitions[native][1]
        endpoint_ids = row.get("evidence_truth_ids")
        if not isinstance(endpoint_ids, list) or len(endpoint_ids) != 2 or len(set(endpoint_ids)) != 2:
            raise ValueError(f"{history_path}: transition has invalid evidence endpoints: {native}")
        for truth_id, tick, value in zip(endpoint_ids, (row["from_tick"], row["to_tick"]), (row["from_value"], row["to_value"])):
            endpoint = truths.get(truth_id)
            if endpoint is None or any(
                endpoint[1].get(key) != expected for key, expected in (
                    ("tick", tick), ("value", value), ("predicate_id", row["predicate_id"]),
                    ("tuple_id", row["tuple_id"]), ("bindings", row["bindings"]),
                )
            ):
                raise ValueError(f"{history_path}: transition endpoint differs: {native}:{truth_id}")
        required_truth.update(endpoint_ids)
    selected_truths = required_truth - local_truth_ids
    if not selected_truths <= truths.keys():
        raise ValueError(f"{branch_root}: inherited truth closure has missing IDs")
    return {
        "source_path": history_path,
        "truths": [(truths[native][0], truths[native][1]) for native in sorted(selected_truths)],
        "transitions": [(transitions[native][0], transitions[native][1]) for native in sorted(selected_transitions)],
    }


def qualify_history_refs(refs: Any, *, history_path: Path, start_tick: int) -> list[str]:
    """Point pre-window raw references into their declared gzip source."""
    if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
        raise ValueError(f"{history_path}: historical source references are malformed")
    archive_ref = history_path.relative_to(REPO).as_posix()
    qualified = []
    for ref in refs:
        if ref.startswith("raw/"):
            marker = "#tick="
            if marker not in ref:
                raise ValueError(f"{history_path}: historical raw reference lacks a tick: {ref}")
            tick_text = ref.split(marker, 1)[1].split("&", 1)[0]
            if not tick_text.isdigit() or int(tick_text) >= start_tick:
                raise ValueError(f"{history_path}: historical raw reference is outside its window: {ref}")
            qualified.append(f"{archive_ref}#{ref}")
        else:
            qualified.append(ref)
    return qualified
