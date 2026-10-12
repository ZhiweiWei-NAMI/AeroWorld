"""Locate saved event references without inventing their missing observations."""

from __future__ import annotations

from typing import Iterable

from Dataset.semantic_truth.minimal_semantics_adapter import _truth_id


def boundary_references(truths: Iterable[dict], events: list[dict],
                        outcomes: list[dict], start: int, end: int) -> dict[str, dict]:
    """Use the producer's ID formula only to locate references outside a window.

    A located reference is still unobserved. It is never a replacement assertion
    or evidence that the event happened. In-window missing references stay errors.
    """
    keys, present, requested, locations = set(), set(), set(), {}
    maximum_tick = end
    for row in truths:
        native = row.get("truth_id")
        if native is None or row.get("tuple_id") is None:
            continue
        present.add(native)
        for episode in {row["episode_id"], row.get("parent_episode_id")} - {None}:
            keys.add((episode, row["tuple_id"], row["predicate_id"]))
    for row in events + outcomes:
        requested.update(row.get("supporting_truth_ids", []))
        for phase in row.get("event_phases", []) + row.get("lifecycle_phase_evidence", []):
            requested.update(phase.get("supporting_truth_ids", []))
            tick = phase.get("tick", phase.get("start_tick"))
            if tick is not None:
                maximum_tick = max(maximum_tick, tick, phase.get("end_tick", tick))
            tuple_id, predicate = phase.get("predicate_tuple_id"), phase.get("predicate_id")
            if tuple_id is not None and predicate is not None:
                for episode in {row["episode_id"], row.get("parent_episode_id")} - {None}:
                    keys.add((episode, tuple_id, predicate))
            transition = phase.get("transition_id", phase.get("entry_transition_id"))
            if transition is not None and tick is not None and tick < start:
                locations[transition] = {"native_reference": transition, "tick": tick,
                    "availability": "before_saved_window", "record_kind": "transition",
                    "location_basis": "event_phase", "observed": False}
        terminal = row.get("terminal_tick")
        if terminal is not None:
            maximum_tick = max(maximum_tick, terminal)
    requested.difference_update(present)
    # Formal truth IDs use the exact five-tick semantic grid. Dense ARM values
    # have separate producers and are not inferred by this routine.
    for episode, tuple_id, predicate in sorted(keys):
        for tick in range(0, maximum_tick + 1, 5):
            if start <= tick <= end:
                continue
            native = _truth_id(episode, tick, tuple_id, predicate)
            if native in requested:
                candidate = {"native_reference": native, "tick": tick,
                    "availability": "before_saved_window" if tick < start else "after_saved_window",
                    "record_kind": "truth", "predicate_id": predicate, "tuple_id": tuple_id,
                    "location_basis": "Dataset/semantic_truth/minimal_semantics_adapter.py:_truth_id",
                    "observed": False}
                if native in locations and locations[native] != candidate:
                    raise ValueError(f"reference has conflicting locations: {native}")
                locations[native] = candidate
    return locations
