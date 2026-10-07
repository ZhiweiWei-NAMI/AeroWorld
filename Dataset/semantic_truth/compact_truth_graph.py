"""Build compact, provenance-closed semantic truth graphs.

The builder is deliberately downstream of predicate and objective-event
derivation.  It never reads scenario plans, authored event traces, semantic
roles, or simulator files.  Records are admitted only when their referenced
truth/event dependencies resolve inside the same episode.

Compactness is achieved by emitting graphs only for:

* the first valid Boolean assertion for each L2-scoped tuple;
* predicate transitions;
* predicate continuity breaks;
* event occurrence/outcome lifecycle records.

Persistent false/unknown assertions, unrelated entities, and unreferenced
numeric observations are not materialized.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from collections.abc import Mapping as _AbcMapping, Sequence as _AbcSequence

from .core_semantic_registry import get_core_predicate_ids
from .provenance import canonical_json, digest_object, stable_identifier, write_jsonl, without_integrity_metadata
from .runtime_schema import runtime_schema_errors


SCHEMA_NAME = "layered_semantic_event_graph"
SCHEMA_VERSION = "2.0.0"
BASE_GRAPH_SCHEMA_NAME = "semantic_graph_base"
DELTA_SCHEMA_NAME = "semantic_graph_delta"
FORMAL_TICK_STEP = 5
GRAPH_PREDICATE_VOCABULARY = frozenset(get_core_predicate_ids())

_FORBIDDEN_EVIDENCE_FRAGMENTS = (
    "event_trace",
    "event_realization",
    "authored_event_label",
    "authored_event",
    "expected_event",
    "semantic_role",
    "task_id",
    "source_event_script_path",
    "dynamic_labels",
    "scenario_plan_as_truth",
)
_RECORD_ID_KEYS = (
    "truth_id",
    "transition_id",
    "event_id",
    "outcome_id",
    "observation_id",
    "evidence_id",
    "numeric_evidence_id",
    "break_id",
    "record_id",
)


@dataclass(frozen=True)
class CompactTruthGraphBuildResult:
    """Replayable base/delta layers plus event-window graph projections."""

    base_graphs: tuple[dict[str, Any], ...]
    deltas: tuple[dict[str, Any], ...]
    graphs: tuple[dict[str, Any], ...]
    rejected_records: tuple[dict[str, Any], ...]


def build_empty_semantic_graph_base(episode_id: str) -> dict[str, Any]:
    """Return the canonical L2 base for an episode with no event ROI."""

    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("empty semantic graph base requires episode_id")
    payload: dict[str, Any] = {
        "schema_name": BASE_GRAPH_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "annotation_layer": "L2",
        "source_layer": "L1",
        "episode_id": episode_id,
        "tick": 0,
        "scope_policy": "roi_or_event_entities",
        "entities": [],
        "initial_assertions": [],
        "edges": [],
        "summary": {
            "entity_count": 0,
            "initial_assertion_count": 0,
            "edge_count": 0,
        },
    }
    payload["base_graph_digest"] = digest_object(payload)
    return payload


def build_compact_truth_graphs(
    predicate_truth: Sequence[Mapping[str, Any]],
    predicate_transitions: Sequence[Mapping[str, Any]],
    event_occurrences: Sequence[Mapping[str, Any]],
    event_outcomes: Sequence[Mapping[str, Any]],
    numeric_evidence: Sequence[Mapping[str, Any]] | None = None,
    continuity_breaks: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return stable compact graph rows or fail on any rejected input row."""

    result = build_compact_truth_graph_result(
        predicate_truth,
        predicate_transitions,
        event_occurrences,
        event_outcomes,
        numeric_evidence,
        continuity_breaks,
    )
    if result.rejected_records:
        first = result.rejected_records[0]
        raise ValueError(
            "compact graph input was rejected: "
            f"{first['record_kind']}:{first['record_id']}:{first['reason']}"
        )
    return list(result.graphs)


def build_compact_truth_graph_result(
    predicate_truth: Sequence[Mapping[str, Any]],
    predicate_transitions: Sequence[Mapping[str, Any]],
    event_occurrences: Sequence[Mapping[str, Any]],
    event_outcomes: Sequence[Mapping[str, Any]],
    numeric_evidence: Sequence[Mapping[str, Any]] | None = None,
    continuity_breaks: Sequence[Mapping[str, Any]] | None = None,
    *,
    continuity_source_truth: Sequence[Mapping[str, Any]] = (),
) -> CompactTruthGraphBuildResult:
    """Build graphs and expose records rejected by evidence-closure gates."""

    builder = _CompactGraphBuilder(
        predicate_truth=predicate_truth,
        predicate_transitions=predicate_transitions,
        event_occurrences=event_occurrences,
        event_outcomes=event_outcomes,
        numeric_evidence=numeric_evidence or (),
        continuity_breaks=continuity_breaks or (),
        continuity_source_truth=continuity_source_truth,
    )
    return builder.build()


def serialize_compact_truth_graph_jsonl(graphs: Iterable[Mapping[str, Any]]) -> str:
    """Serialize graph rows canonically, including a final newline."""

    rows = sorted((dict(row) for row in graphs), key=_graph_sort_key)
    if not rows:
        return ""
    return "".join(f"{canonical_json(without_integrity_metadata(row))}\n" for row in rows)


def serialize_semantic_graph_deltas_jsonl(
    deltas: Iterable[Mapping[str, Any]],
) -> str:
    """Serialize the ordered predicate delta stream canonically."""

    rows = sorted(
        (dict(row) for row in deltas),
        key=lambda row: (
            str(row.get("episode_id", "")),
            int(row.get("tick", -1)),
            str(row.get("delta_id", "")),
        ),
    )
    return "".join(f"{canonical_json(without_integrity_metadata(row))}\n" for row in rows)


def replay_predicate_state(
    base_graph: Mapping[str, Any],
    deltas: Sequence[Mapping[str, Any]],
    tick: int,
) -> dict[str, Any]:
    """Reconstruct the L1 predicate state at any tick from L2 base + deltas."""

    if base_graph.get("schema_name") != BASE_GRAPH_SCHEMA_NAME:
        raise ValueError("base graph has an unexpected schema")
    if not _is_int(tick) or int(tick) < int(base_graph.get("tick", 0)):
        raise ValueError("replay tick precedes the base graph")
    claimed_base_digest = base_graph.get("base_graph_digest")

    episode_id = str(base_graph["episode_id"])
    state: dict[tuple[str, str], dict[str, Any]] = {}
    for assertion in base_graph.get("initial_assertions", ()):
        if not isinstance(assertion, _AbcMapping):
            raise ValueError("base assertion must be an object")
        key = (str(assertion["predicate_id"]), str(assertion["tuple_id"]))
        if key in state:
            raise ValueError(f"duplicate base predicate tuple: {key}")
        state[key] = {
            "predicate_id": key[0],
            "tuple_id": key[1],
            "bindings": _bindings(assertion),
            "truth_value": str(assertion["truth_value"]),
            "last_update_tick": int(base_graph["tick"]),
            "source_record_id": str(assertion["assertion_id"]),
        }

    previous_delta_id: str | None = None
    for delta in sorted(
        deltas,
        key=lambda row: (int(row.get("tick", -1)), str(row.get("delta_id", ""))),
    ):
        if str(delta.get("episode_id")) != episode_id:
            raise ValueError("delta episode differs from the base graph")
        if delta.get("previous_delta_id") != previous_delta_id:
            raise ValueError("delta chain is not contiguous")
        previous_delta_id = str(delta["delta_id"])
        if int(delta["tick"]) > int(tick):
            continue
        for operation in delta.get("operations", ()):
            if not isinstance(operation, _AbcMapping):
                raise ValueError("delta operation must be an object")
            key = (
                str(operation["predicate_id"]),
                str(operation["tuple_id"]),
            )
            current = state.get(key)
            if operation.get("operation") == "add_predicate_truth":
                if current is not None:
                    raise ValueError(
                        f"delta adds a tuple already present in state: {key}"
                    )
                state[key] = {
                    "predicate_id": key[0],
                    "tuple_id": key[1],
                    "bindings": _bindings(operation),
                    "truth_value": str(operation["truth_value"]),
                    "last_update_tick": int(delta["tick"]),
                    "source_record_id": str(operation["source_truth_id"]),
                }
                continue
            if current is None:
                raise ValueError(
                    f"delta references a tuple absent from the base: {key}"
                )
            if operation.get("operation") == "set_predicate_truth":
                if current["truth_value"] != operation.get("from_value"):
                    raise ValueError(
                        f"delta from_value does not match replay state for {key}"
                    )
                value = str(operation["to_value"])
                source_record_id = str(operation["source_transition_id"])
            elif operation.get("operation") == "invalidate_predicate_truth":
                value = "unknown"
                source_record_id = str(operation["source_break_id"])
            elif operation.get("operation") == "restore_predicate_truth":
                if current["truth_value"] != "unknown":
                    raise ValueError(
                        f"delta restores a tuple without an unknown interval: {key}"
                    )
                value = str(operation["truth_value"])
                source_record_id = str(operation["source_truth_id"])
            else:
                raise ValueError(
                    f"unknown delta operation: {operation.get('operation')}"
                )
            current.update(
                truth_value=value,
                last_update_tick=int(delta["tick"]),
                source_record_id=source_record_id,
            )

    rows = sorted(
        state.values(),
        key=lambda row: (str(row["predicate_id"]), str(row["tuple_id"])),
    )
    payload: dict[str, Any] = {
        "schema_name": "semantic_graph_replay_state",
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "tick": int(tick),
        "base_graph_digest": claimed_base_digest,
        "predicate_states": rows,
    }
    payload["state_digest"] = digest_object(rows)
    return payload


def write_compact_truth_graph_jsonl(
    path: Path,
    graphs: Iterable[Mapping[str, Any]],
) -> None:
    """Write stable compact graph JSONL without creating unrelated folders."""

    write_jsonl(path, sorted((dict(row) for row in graphs), key=_graph_sort_key))


class _CompactGraphBuilder:
    def __init__(
        self,
        *,
        predicate_truth: Sequence[Mapping[str, Any]],
        predicate_transitions: Sequence[Mapping[str, Any]],
        event_occurrences: Sequence[Mapping[str, Any]],
        event_outcomes: Sequence[Mapping[str, Any]],
        numeric_evidence: Sequence[Mapping[str, Any]],
        continuity_breaks: Sequence[Mapping[str, Any]],
        continuity_source_truth: Sequence[Mapping[str, Any]],
    ) -> None:
        self.raw_truth = tuple(predicate_truth)
        self.raw_transitions = tuple(predicate_transitions)
        self.raw_events = tuple(event_occurrences)
        self.raw_outcomes = tuple(event_outcomes)
        self.raw_numeric_evidence = tuple(numeric_evidence)
        self.raw_breaks = tuple(continuity_breaks)
        self.raw_continuity_truth = tuple(continuity_source_truth)
        self.continuity_truth: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.rejections: list[dict[str, Any]] = []

        self.truth: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.transitions: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.events: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.outcomes: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.numeric: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.breaks: dict[tuple[str, str], Mapping[str, Any]] = {}

    def build(self) -> CompactTruthGraphBuildResult:
        self.numeric = self._index_numeric_evidence()
        self.truth = self._index_truth()
        self.continuity_truth = self._index_continuity_truth()
        self.transitions = self._index_transitions()
        self.events = self._index_events()
        self.outcomes = self._index_outcomes()
        self.breaks = self._index_breaks()

        scope_by_episode = self._l2_entity_scope()
        base_graphs = self._build_base_graphs(scope_by_episode)
        deltas = self._build_delta_stream(scope_by_episode, base_graphs)

        buckets: dict[tuple[str, int], _GraphBucket] = {}
        referenced_truth: set[tuple[str, str]] = set()

        for key, event in sorted(self.events.items()):
            bucket = self._bucket(buckets, key[0], int(event["trigger_tick"]))
            self._add_event_closure(bucket, event, referenced_truth)

        for key, outcome in sorted(self.outcomes.items()):
            event = self.events[(key[0], str(outcome["event_id"]))]
            anchor_tick = outcome.get("terminal_tick")
            if not _is_int(anchor_tick):
                anchor_tick = int(event["trigger_tick"])
            bucket = self._bucket(buckets, key[0], int(anchor_tick))
            self._add_lifecycle_closure(bucket, outcome, referenced_truth)

        first_true_by_tuple: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        for (episode_id, truth_id), truth in sorted(
            self.truth.items(), key=lambda item: _truth_sort_key(item[1])
        ):
            if truth.get("value") != "true":
                continue
            tuple_key = (
                episode_id,
                str(truth["predicate_id"]),
                str(truth["tuple_id"]),
            )
            first_true_by_tuple.setdefault(tuple_key, truth)
        for truth in first_true_by_tuple.values():
            key = (str(truth["episode_id"]), str(truth["truth_id"]))
            if key in referenced_truth:
                continue
            if not set(_bindings(truth).values()) & scope_by_episode.get(
                str(truth["episode_id"]), set()
            ):
                continue
            bucket = self._bucket(buckets, str(truth["episode_id"]), int(truth["tick"]))
            bucket.add_truth(truth, self.numeric)
            referenced_truth.add(key)

        for (_, _), continuity_break in sorted(self.breaks.items()):
            bucket = self._bucket(
                buckets,
                str(continuity_break["episode_id"]),
                int(continuity_break["to_tick"]),
            )
            self._add_break_closure(bucket, continuity_break, referenced_truth)

        base_by_episode = {str(base["episode_id"]): base for base in base_graphs}
        deltas_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for delta in deltas:
            deltas_by_episode[str(delta["episode_id"])].append(delta)
        graphs = tuple(
            self._materialize_event_projection(
                bucket.finish(),
                base_by_episode[str(bucket.episode_id)],
                deltas_by_episode[str(bucket.episode_id)],
            )
            for _, bucket in sorted(buckets.items(), key=lambda item: item[0])
            if bucket.has_semantic_content
        )
        rejected = tuple(
            sorted(
                self.rejections,
                key=lambda row: (
                    row.get("episode_id", ""),
                    row.get("record_kind", ""),
                    row.get("record_id", ""),
                    row.get("reason", ""),
                ),
            )
        )
        return CompactTruthGraphBuildResult(
            base_graphs=base_graphs,
            deltas=deltas,
            graphs=graphs,
            rejected_records=rejected,
        )

    def _l2_entity_scope(self) -> dict[str, set[str]]:
        """Close event participants over their selected L1 predicate tuples."""

        scope: dict[str, set[str]] = defaultdict(set)
        for episode_id, _ in self.truth:
            scope[episode_id]
        for (episode_id, _), event in self.events.items():
            scope[episode_id].update(_bindings(event).values())
        event_episode_ids = {episode_id for episode_id, _ in self.events}
        for (episode_id, _), truth in self.truth.items():
            if episode_id not in event_episode_ids and truth.get("value") == "true":
                scope[episode_id].update(_bindings(truth).values())
        for (episode_id, _), continuity_break in self.breaks.items():
            scope[episode_id].update(_bindings(continuity_break).values())
        changed = True
        rows = [*self.truth.values(), *self.transitions.values()]
        while changed:
            changed = False
            for row in rows:
                episode_id = str(row["episode_id"])
                bindings = set(_bindings(row).values())
                if not bindings or not (bindings & scope[episode_id]):
                    continue
                before = len(scope[episode_id])
                scope[episode_id].update(bindings)
                changed = changed or len(scope[episode_id]) != before
        return scope

    def _build_base_graphs(
        self,
        scope_by_episode: Mapping[str, set[str]],
    ) -> tuple[dict[str, Any], ...]:
        base_graphs: list[dict[str, Any]] = []
        for episode_id, entity_ids in sorted(scope_by_episode.items()):
            assertions: list[dict[str, Any]] = []
            for (_, _), row in sorted(
                self.truth.items(), key=lambda item: _truth_sort_key(item[1])
            ):
                if str(row["episode_id"]) != episode_id or int(row["tick"]) != 0:
                    continue
                bindings = _bindings(row)
                if not set(bindings.values()) & entity_ids:
                    continue
                assertions.append(
                    {
                        "assertion_id": str(row["truth_id"]),
                        "predicate_id": str(row["predicate_id"]),
                        "tuple_id": str(row["tuple_id"]),
                        "truth_value": str(row["value"]),
                        "bindings": bindings,
                        "source_refs": _record_source_refs(row),
                        "rule_digest": row.get("rule_digest"),
                        "parameter_digest": row.get("parameter_digest"),
                        "input_digest": row.get("input_digest"),
                    }
                )
            entities = [
                {
                    "node_id": stable_identifier(
                        "semantic_base_entity", episode_id, entity_id
                    ),
                    "kind": "entity",
                    "entity_id": entity_id,
                }
                for entity_id in sorted(entity_ids)
            ]
            edges = [
                {
                    "edge_id": stable_identifier(
                        "semantic_base_asserts_for",
                        episode_id,
                        assertion["assertion_id"],
                        role,
                        entity_id,
                    ),
                    "kind": "asserts_for",
                    "assertion_id": assertion["assertion_id"],
                    "role": role,
                    "entity_id": entity_id,
                }
                for assertion in assertions
                for role, entity_id in sorted(assertion["bindings"].items())
            ]
            payload: dict[str, Any] = {
                "schema_name": BASE_GRAPH_SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "annotation_layer": "L2",
                "source_layer": "L1",
                "episode_id": episode_id,
                "tick": 0,
                "scope_policy": "roi_or_event_entities",
                "entities": entities,
                "initial_assertions": assertions,
                "edges": edges,
                "summary": {
                    "entity_count": len(entities),
                    "initial_assertion_count": len(assertions),
                    "edge_count": len(edges),
                },
            }
            payload["base_graph_digest"] = digest_object(payload)
            base_graphs.append(payload)
        return tuple(base_graphs)

    def _build_delta_stream(
        self,
        scope_by_episode: Mapping[str, set[str]],
        base_graphs: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        base_digest = {
            str(base["episode_id"]): str(base["base_graph_digest"])
            for base in base_graphs
        }
        operations_by_tick: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(
            list
        )
        first_truth_by_tuple: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        # Unknown endpoints initialize four-valued replay state from their
        # actual record/tick. They never enter Boolean self.truth or assertions.
        replay_sources = dict(self.truth)
        replay_sources.update(self.continuity_truth)
        for (_, _), row in sorted(
            replay_sources.items(), key=lambda item: _truth_sort_key(item[1])
        ):
            episode_id = str(row["episode_id"])
            bindings = _bindings(row)
            if not set(bindings.values()) & scope_by_episode.get(episode_id, set()):
                continue
            key = (episode_id, str(row["predicate_id"]), str(row["tuple_id"]))
            first_truth_by_tuple.setdefault(key, row)
        for (episode_id, predicate_id, tuple_id), row in sorted(
            first_truth_by_tuple.items()
        ):
            tick = int(row["tick"])
            if tick == 0 and row['value'] in {'true', 'false'}:
                continue
            operations_by_tick[(episode_id, tick)].append(
                {
                    "operation": "add_predicate_truth",
                    "source_truth_id": str(row["truth_id"]),
                    "predicate_id": predicate_id,
                    "tuple_id": tuple_id,
                    "bindings": _bindings(row),
                    "truth_value": str(row["value"]),
                    "source_refs": _record_source_refs(row),
                }
            )
        for (episode_id, _), row in sorted(self.transitions.items()):
            bindings = _bindings(row)
            if not set(bindings.values()) & scope_by_episode.get(episode_id, set()):
                continue
            operations_by_tick[(episode_id, int(row["to_tick"]))].append(
                {
                    "operation": "set_predicate_truth",
                    "source_transition_id": str(row["transition_id"]),
                    "predicate_id": str(row["predicate_id"]),
                    "tuple_id": str(row["tuple_id"]),
                    "bindings": bindings,
                    "from_tick": int(row["from_tick"]),
                    "to_tick": int(row["to_tick"]),
                    "from_value": str(row["from_value"]),
                    "to_value": str(row["to_value"]),
                    "direction": str(row["direction"]),
                    "evidence_truth_ids": sorted(
                        _string_list(row.get("evidence_truth_ids"))
                    ),
                }
            )
        for (episode_id, _), row in sorted(self.breaks.items()):
            bindings = _bindings(row)
            if not set(bindings.values()) & scope_by_episode.get(episode_id, set()):
                continue
            predicate_id = str(row["predicate_id"])
            tuple_id = str(row["tuple_id"])
            gap_start_tick = int(row["from_tick"]) + FORMAL_TICK_STEP
            operations_by_tick[(episode_id, gap_start_tick)].append(
                {
                    "operation": "invalidate_predicate_truth",
                    "source_break_id": str(row["break_id"]),
                    "predicate_id": predicate_id,
                    "tuple_id": tuple_id,
                    "bindings": bindings,
                    "from_tick": int(row["from_tick"]),
                    "to_tick": int(row["to_tick"]),
                    "to_value": "unknown",
                    "reason": str(row["reason"]),
                }
            )
            restoration_rows = [
                truth
                for truth in self.truth.values()
                if str(truth["episode_id"]) == episode_id
                and str(truth["predicate_id"]) == predicate_id
                and str(truth["tuple_id"]) == tuple_id
                and int(truth["tick"]) == int(row["to_tick"])
                and _bindings(truth) == bindings
            ]
            if len(restoration_rows) > 1:
                self._reject(
                    row,
                    "predicate_continuity_break",
                    "ambiguous_break_reestablishment_truth",
                )
                continue
            if not restoration_rows:
                if row.get("to_value") in {"true", "false"}:
                    self._reject(
                        row,
                        "predicate_continuity_break",
                        "missing_break_reestablishment_truth",
                    )
                continue
            restoration = restoration_rows[0]
            operations_by_tick[(episode_id, int(row["to_tick"]))].append(
                {
                    "operation": "restore_predicate_truth",
                    "source_break_id": str(row["break_id"]),
                    "source_truth_id": str(restoration["truth_id"]),
                    "predicate_id": predicate_id,
                    "tuple_id": tuple_id,
                    "bindings": bindings,
                    "truth_value": str(restoration["value"]),
                    "reason": "reestablish_after_continuity_break",
                }
            )

        replay_values: dict[tuple[str, str, str], str] = {}
        for base in base_graphs:
            episode_id = str(base["episode_id"])
            for assertion in base.get("initial_assertions", ()):
                replay_values[
                    (
                        episode_id,
                        str(assertion["predicate_id"]),
                        str(assertion["tuple_id"]),
                    )
                ] = str(assertion["truth_value"])
        filtered_operations: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(
            list
        )
        for (episode_id, tick), operations in sorted(operations_by_tick.items()):
            operations.sort(
                key=lambda operation: (
                    _delta_operation_order(str(operation["operation"])),
                    str(operation["predicate_id"]),
                    str(operation["tuple_id"]),
                    str(
                        operation.get(
                            "source_transition_id",
                            operation.get("source_break_id", ""),
                        )
                    ),
                )
            )
            for operation in operations:
                state_key = (
                    episode_id,
                    str(operation["predicate_id"]),
                    str(operation["tuple_id"]),
                )
                current = replay_values.get(state_key)
                if operation["operation"] == "add_predicate_truth":
                    if current is not None:
                        continue
                    replay_values[state_key] = str(operation["truth_value"])
                    filtered_operations[(episode_id, tick)].append(operation)
                    continue
                if operation["operation"] == "restore_predicate_truth":
                    if current != "unknown":
                        source = self.breaks.get(
                            (episode_id, str(operation["source_break_id"]))
                        )
                        if source is not None:
                            self._reject(
                                source,
                                "predicate_continuity_break",
                                "continuity_restore_without_unknown_state",
                            )
                        continue
                    replay_values[state_key] = str(operation["truth_value"])
                    filtered_operations[(episode_id, tick)].append(operation)
                    continue
                if current is None:
                    if (
                        operation["operation"] == "set_predicate_truth"
                        and operation.get("from_value") == "out_of_scope"
                    ):
                        replay_values[state_key] = str(operation["to_value"])
                        filtered_operations[(episode_id, tick)].append(
                            {
                                "operation": "add_predicate_truth",
                                "source_transition_id": str(
                                    operation["source_transition_id"]
                                ),
                                "predicate_id": str(operation["predicate_id"]),
                                "tuple_id": str(operation["tuple_id"]),
                                "bindings": dict(operation["bindings"]),
                                "truth_value": str(operation["to_value"]),
                                "reason": "reenter_scope_after_out_of_scope",
                            }
                        )
                        continue
                    source_id = str(
                        operation.get(
                            "source_transition_id", operation.get("source_break_id", "")
                        )
                    )
                    source = self.transitions.get(
                        (episode_id, source_id)
                    ) or self.breaks.get((episode_id, source_id))
                    if source is not None:
                        self._reject(
                            source,
                            str(source.get("schema_name", "predicate_delta")),
                            "delta_tuple_has_no_prior_state",
                        )
                    continue
                if operation["operation"] == "set_predicate_truth":
                    if current != str(operation["from_value"]):
                        transition = self.transitions.get(
                            (episode_id, str(operation["source_transition_id"]))
                        )
                        if transition is not None:
                            self._reject(
                                transition,
                                "predicate_transition",
                                "transition_chain_discontinuous",
                            )
                        continue
                    replay_values[state_key] = str(operation["to_value"])
                else:
                    replay_values[state_key] = "unknown"
                filtered_operations[(episode_id, tick)].append(operation)
        operations_by_tick = filtered_operations

        result: list[dict[str, Any]] = []
        previous_by_episode: dict[str, str | None] = defaultdict(lambda: None)
        for (episode_id, tick), operations in sorted(operations_by_tick.items()):
            operations.sort(
                key=lambda row: (
                    _delta_operation_order(str(row["operation"])),
                    str(row["predicate_id"]),
                    str(row["tuple_id"]),
                    str(
                        row.get("source_transition_id", row.get("source_break_id", ""))
                    ),
                )
            )
            delta_id = stable_identifier(
                "semantic_graph_delta", episode_id, tick, operations
            )
            payload: dict[str, Any] = {
                "schema_name": DELTA_SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "annotation_layer": "L2",
                "source_layer": "L1",
                "episode_id": episode_id,
                "tick": tick,
                "delta_id": delta_id,
                "previous_delta_id": previous_by_episode[episode_id],
                "base_graph_digest": base_digest[episode_id],
                "operations": operations,
                "operation_count": len(operations),
            }
            payload["delta_digest"] = digest_object(payload)
            result.append(payload)
            previous_by_episode[episode_id] = delta_id
        return tuple(result)

    def _materialize_event_projection(
        self,
        graph: dict[str, Any],
        base_graph: Mapping[str, Any],
        deltas: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        replay = replay_predicate_state(base_graph, deltas, int(graph["anchor_tick"]))
        applied = [
            str(delta["delta_id"])
            for delta in deltas
            if int(delta["tick"]) <= int(graph["anchor_tick"])
        ]
        graph.pop("graph_digest", None)
        graph.update(
            {
                "annotation_layer": "L2",
                "source_layer": "L1",
                "scope_policy": "roi_or_event_entities",
                "representation": "event_window_projection_from_base_delta",
                "base_graph_digest": base_graph["base_graph_digest"],
                "applied_delta_ids": applied,
                "replayed_state_digest": replay["state_digest"],
            }
        )
        graph["graph_digest"] = digest_object(graph)
        return graph

    def _bucket(
        self,
        buckets: MutableMapping[tuple[str, int], "_GraphBucket"],
        episode_id: str,
        anchor_tick: int,
    ) -> "_GraphBucket":
        key = (episode_id, anchor_tick)
        if key not in buckets:
            buckets[key] = _GraphBucket(episode_id, anchor_tick)
        return buckets[key]

    def _reject(self, row: Mapping[str, Any], kind: str, reason: str) -> None:
        self.rejections.append(
            {
                "episode_id": str(row.get("episode_id", "")),
                "record_kind": kind,
                "record_id": _record_id(row) or "<missing>",
                "reason": reason,
                "record_digest": digest_object(dict(row)),
            }
        )

    def _index_numeric_evidence(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        accepted: dict[tuple[str, str], Mapping[str, Any]] = {}
        conflicted: set[tuple[str, str]] = set()
        for row in self.raw_numeric_evidence:
            episode_id = _episode_id(row)
            record_id = _record_id(row)
            if episode_id is None or record_id is None:
                self._reject(row, "numeric_evidence", "missing_episode_or_record_id")
                continue
            key = (episode_id, record_id)
            if _evidence_has_forbidden_content(row, include_entire_record=True):
                self._reject(
                    row, "numeric_evidence", "forbidden_authored_evidence_source"
                )
                continue
            if not _numeric_leaves(row):
                self._reject(row, "numeric_evidence", "missing_finite_numeric_value")
                continue
            if not _string_list(row.get("source_refs")):
                self._reject(row, "numeric_evidence", "missing_numeric_source_refs")
                continue
            if key in accepted and dict(accepted[key]) != dict(row):
                conflicted.add(key)
                self._reject(row, "numeric_evidence", "conflicting_duplicate_record_id")
                continue
            accepted[key] = row
        for key in conflicted:
            accepted.pop(key, None)
        return accepted

    def _index_truth(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        accepted: dict[tuple[str, str], Mapping[str, Any]] = {}
        conflicted: set[tuple[str, str]] = set()
        for row in self.raw_truth:
            episode_id = _episode_id(row)
            truth_id = _text(row.get("truth_id"))
            if episode_id is None or truth_id is None:
                self._reject(row, "predicate_truth", "missing_episode_or_truth_id")
                continue
            key = (episode_id, truth_id)
            reason = _truth_rejection_reason(row, self.numeric)
            if reason:
                self._reject(row, "predicate_truth", reason)
                continue
            if key in accepted and dict(accepted[key]) != dict(row):
                conflicted.add(key)
                self._reject(row, "predicate_truth", "conflicting_duplicate_truth_id")
                continue
            accepted[key] = row
        for key in conflicted:
            accepted.pop(key, None)
        return accepted

    def _index_transitions(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        accepted: dict[tuple[str, str], Mapping[str, Any]] = {}
        conflicted: set[tuple[str, str]] = set()
        for row in self.raw_transitions:
            episode_id = _episode_id(row)
            transition_id = _text(row.get("transition_id"))
            if episode_id is None or transition_id is None:
                self._reject(
                    row, "predicate_transition", "missing_episode_or_transition_id"
                )
                continue
            key = (episode_id, transition_id)
            reason = self._transition_rejection_reason(row)
            if reason:
                self._reject(row, "predicate_transition", reason)
                continue
            if key in accepted and dict(accepted[key]) != dict(row):
                conflicted.add(key)
                self._reject(
                    row, "predicate_transition", "conflicting_duplicate_transition_id"
                )
                continue
            accepted[key] = row
        for key in conflicted:
            accepted.pop(key, None)
        return accepted

    def _transition_rejection_reason(self, row: Mapping[str, Any]) -> str | None:
        if _evidence_has_forbidden_content(row):
            return "forbidden_authored_evidence_source"
        if not _string_list(row.get("source_refs")):
            return "missing_transition_source_refs"
        from_tick = row.get("from_tick")
        to_tick = row.get("to_tick")
        if (
            not _is_int(from_tick)
            or not _is_int(to_tick)
            or int(to_tick) - int(from_tick) != FORMAL_TICK_STEP
        ):
            return "invalid_transition_tick_window"
        from_value = row.get("from_value")
        to_value = row.get("to_value")
        direction = row.get("direction")
        if (from_value, to_value, direction) not in {
            ("false", "true", "rising"),
            ("true", "false", "falling"),
        }:
            return "invalid_boolean_transition_direction"
        evidence_ids = _string_list(row.get("evidence_truth_ids"))
        if len(evidence_ids) != 2:
            return "transition_requires_two_truth_records"
        episode_id = str(row["episode_id"])
        resolved = [self.truth.get((episode_id, truth_id)) for truth_id in evidence_ids]
        if any(item is None for item in resolved):
            return "missing_referenced_truth_record"
        assert all(item is not None for item in resolved)
        for tick, value in ((int(from_tick), from_value), (int(to_tick), to_value)):
            if not any(
                int(item["tick"]) == tick and item.get("value") == value
                for item in resolved
            ):
                return "transition_truth_endpoints_do_not_match"
        for item in resolved:
            if (
                str(item.get("predicate_id")) != str(row.get("predicate_id"))
                or str(item.get("tuple_id")) != str(row.get("tuple_id"))
                or _bindings(item) != _bindings(row)
            ):
                return "transition_truth_identity_mismatch"
        return None

    def _index_events(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        accepted: dict[tuple[str, str], Mapping[str, Any]] = {}
        conflicted: set[tuple[str, str]] = set()
        for row in self.raw_events:
            episode_id = _episode_id(row)
            event_id = _text(row.get("event_id"))
            if episode_id is None or event_id is None:
                self._reject(row, "event_occurrence", "missing_episode_or_event_id")
                continue
            key = (episode_id, event_id)
            reason = self._event_rejection_reason(row)
            if reason:
                self._reject(row, "event_occurrence", reason)
                continue
            if key in accepted and dict(accepted[key]) != dict(row):
                conflicted.add(key)
                self._reject(row, "event_occurrence", "conflicting_duplicate_event_id")
                continue
            accepted[key] = row
        for key in conflicted:
            accepted.pop(key, None)
        return accepted

    def _event_rejection_reason(self, row: Mapping[str, Any]) -> str | None:
        if runtime_schema_errors(row, "event_occurrence.schema.json"):
            return "event_occurrence_schema_violation"
        header_reason = _runtime_record_header_rejection_reason(
            row,
            schema_name="event_occurrence",
            annotation_layer="L2",
            source_layer="L1",
        )
        if header_reason is not None:
            return header_reason
        if _evidence_has_forbidden_content(row):
            return "forbidden_authored_evidence_source"
        if not _string_list(row.get("source_refs")):
            return "missing_event_source_refs"
        event_bindings = _binding_map(row.get("bindings"))
        if event_bindings is None:
            return "missing_event_bindings"
        participant_roles = _ordered_string_list(row.get("participant_roles"))
        if participant_roles is None or set(participant_roles) != set(event_bindings):
            return "event_participant_roles_differ_from_bindings"
        for key in ("trigger_tick", "detection_tick", "start_tick", "end_tick"):
            if not _is_int(row.get(key)):
                return "invalid_event_tick"
        if (
            int(row["start_tick"]) != int(row["trigger_tick"])
            or int(row["end_tick"]) != int(row["detection_tick"])
            or int(row["detection_tick"]) < int(row["trigger_tick"])
        ):
            return "invalid_event_window"
        truth_ids = _ordered_string_list(row.get("supporting_truth_ids"))
        transition_ids = _ordered_string_list(row.get("supporting_transition_ids"))
        if not truth_ids or not transition_ids:
            return "event_requires_truth_and_transition_support"
        episode_id = str(row["episode_id"])
        if any((episode_id, item) not in self.truth for item in truth_ids):
            return "missing_event_truth_support"
        if any((episode_id, item) not in self.transitions for item in transition_ids):
            return "missing_event_transition_support"
        raw_phases = row.get("event_phases")
        if (
            not isinstance(raw_phases, _AbcSequence)
            or isinstance(raw_phases, (str, bytes, bytearray))
            or not raw_phases
            or any(not isinstance(phase, _AbcMapping) for phase in raw_phases)
        ):
            return "event_requires_explicit_phases"
        phases = list(raw_phases)
        phase_keys: list[tuple[str, str]] = []
        phase_transition_ids: list[str] = []
        phase_kinds_by_transition: dict[str, set[str]] = defaultdict(set)
        onset_count = 0
        for phase in phases:
            phase_kind = _text(phase.get("phase_kind"))
            transition_id = _text(phase.get("transition_id"))
            if phase_kind not in {"onset", "escalation_support", "terminal"}:
                return "invalid_event_phase_kind"
            if transition_id is None:
                return "event_phase_lacks_transition"
            phase_key = (phase_kind, transition_id)
            if phase_key in phase_keys:
                return "duplicate_event_phase_role"
            phase_keys.append(phase_key)
            phase_kinds_by_transition[transition_id].add(phase_kind)
            if transition_id not in phase_transition_ids:
                phase_transition_ids.append(transition_id)
            transition = self.transitions.get((episode_id, transition_id))
            if transition is None:
                return "missing_event_transition_support"
            phase_bindings = _binding_map(phase.get("predicate_bindings"))
            if (
                _text(phase.get("predicate_id"))
                != _text(transition.get("predicate_id"))
                or _text(phase.get("predicate_tuple_id"))
                != _text(transition.get("tuple_id"))
                or phase_bindings is None
                or phase_bindings != _binding_map(transition.get("bindings"))
                or _text(phase.get("direction")) != _text(transition.get("direction"))
                or not _is_int(phase.get("tick"))
                or int(phase["tick"]) != int(transition["to_tick"])
            ):
                return "event_phase_metadata_mismatch"
            transition_truth_ids = _ordered_string_list(
                transition.get("evidence_truth_ids")
            )
            if transition_truth_ids is None or not set(transition_truth_ids) <= set(
                truth_ids
            ):
                return "event_phase_truth_support_mismatch"
            if phase_kind == "onset":
                onset_count += 1
                if int(phase["tick"]) != int(row["trigger_tick"]):
                    return "event_onset_tick_mismatch"
        if onset_count != 1:
            return "event_requires_exactly_one_onset_phase"
        if phase_transition_ids != transition_ids:
            return "event_phases_and_transition_support_differ"
        if any(
            kinds != {"onset", "terminal"}
            for kinds in phase_kinds_by_transition.values()
            if len(kinds) > 1
        ):
            return "transition_has_invalid_multiple_phase_roles"
        return None

    def _index_outcomes(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        accepted: dict[tuple[str, str], Mapping[str, Any]] = {}
        conflicted: set[tuple[str, str]] = set()
        for row in self.raw_outcomes:
            episode_id = _episode_id(row)
            outcome_id = _text(row.get("outcome_id"))
            if episode_id is None or outcome_id is None:
                self._reject(row, "event_outcome", "missing_episode_or_outcome_id")
                continue
            key = (episode_id, outcome_id)
            reason = self._outcome_rejection_reason(row)
            if reason:
                self._reject(row, "event_outcome", reason)
                continue
            if key in accepted and dict(accepted[key]) != dict(row):
                conflicted.add(key)
                self._reject(row, "event_outcome", "conflicting_duplicate_outcome_id")
                continue
            accepted[key] = row
        for key in conflicted:
            accepted.pop(key, None)
        return accepted

    def _outcome_rejection_reason(self, row: Mapping[str, Any]) -> str | None:
        if runtime_schema_errors(row, "event_outcome.schema.json"):
            return "event_outcome_schema_violation"
        header_reason = _runtime_record_header_rejection_reason(
            row,
            schema_name="event_outcome",
            annotation_layer="L2",
            source_layer="L1",
        )
        if header_reason is not None:
            return header_reason
        if _evidence_has_forbidden_content(row):
            return "forbidden_authored_evidence_source"
        episode_id = str(row.get("episode_id", ""))
        event_id = _text(row.get("event_id"))
        if event_id is None or (episode_id, event_id) not in self.events:
            return "missing_referenced_event"
        event = self.events[(episode_id, event_id)]
        if row.get("event_family_id") != event.get("event_family_id") or row.get(
            "event_type_id"
        ) != event.get("event_type_id"):
            return "outcome_event_identity_mismatch"
        status = _text(row.get("status"))
        truth_ids = _ordered_string_list(row.get("supporting_truth_ids"))
        if truth_ids is None:
            return "invalid_outcome_truth_support"
        terminal_tick = row.get("terminal_tick")
        lifecycle_evidence = row.get("lifecycle_phase_evidence")
        if (
            not isinstance(lifecycle_evidence, _AbcSequence)
            or isinstance(lifecycle_evidence, (str, bytes, bytearray))
            or any(not isinstance(item, _AbcMapping) for item in lifecycle_evidence)
        ):
            return "invalid_outcome_lifecycle_evidence"
        terminal_phases = [
            phase
            for phase in event.get("event_phases", ())
            if isinstance(phase, _AbcMapping) and phase.get("phase_kind") == "terminal"
        ]
        if status == "pending":
            if (
                terminal_tick is not None
                or truth_ids
                or lifecycle_evidence
                or terminal_phases
                or row.get("lifecycle_status")
                not in {
                    "terminal_transition_not_observed",
                    "terminal_hold_not_observed",
                }
            ):
                return "pending_outcome_must_not_claim_terminal_evidence"
            return None
        if status != "succeeded" or not _is_int(terminal_tick):
            return "succeeded_outcome_requires_terminal_tick"
        if int(terminal_tick) <= int(event["trigger_tick"]):
            return "terminal_tick_must_follow_event_start"
        if len(terminal_phases) != 1 or len(lifecycle_evidence) != 1:
            return "succeeded_outcome_requires_one_terminal_phase"
        evidence = lifecycle_evidence[0]
        hold_truth_ids = _ordered_string_list(evidence.get("supporting_truth_ids"))
        hold_bindings = _binding_map(evidence.get("predicate_bindings"))
        start_tick = evidence.get("start_tick")
        end_tick = evidence.get("end_tick")
        entry_transition_id = _text(evidence.get("entry_transition_id"))
        if (
            evidence.get("phase_kind") != "terminal_hold"
            or not hold_truth_ids
            or hold_bindings is None
            or not _is_int(start_tick)
            or not _is_int(end_tick)
            or int(start_tick) > int(end_tick)
            or (int(end_tick) - int(start_tick)) % FORMAL_TICK_STEP != 0
            or int(end_tick) != int(terminal_tick)
            or entry_transition_id is None
            or terminal_phases[0].get("transition_id") != entry_transition_id
        ):
            return "invalid_terminal_hold_evidence"
        transition = self.transitions.get((episode_id, entry_transition_id))
        if transition is None:
            return "missing_terminal_entry_transition"
        if (
            evidence.get("predicate_id") != transition.get("predicate_id")
            or evidence.get("predicate_tuple_id") != transition.get("tuple_id")
            or hold_bindings != _binding_map(transition.get("bindings"))
            or evidence.get("value") != transition.get("to_value")
            or int(start_tick) != int(transition["to_tick"])
        ):
            return "terminal_hold_identity_mismatch"
        expected_lifecycle_status = (
            "terminal_predicate_hold"
            if transition.get("to_value") == "true"
            else "recovery_predicate_hold"
        )
        if row.get("lifecycle_status") != expected_lifecycle_status:
            return "outcome_lifecycle_status_mismatch"
        expected_ticks = list(
            range(int(start_tick), int(end_tick) + 1, FORMAL_TICK_STEP)
        )
        if len(expected_ticks) != len(hold_truth_ids):
            return "terminal_hold_tick_coverage_mismatch"
        for expected_tick, truth_id in zip(expected_ticks, hold_truth_ids):
            truth = self.truth.get((episode_id, truth_id))
            if truth is None:
                return "missing_outcome_truth_support"
            if (
                truth.get("predicate_id") != evidence.get("predicate_id")
                or truth.get("tuple_id") != evidence.get("predicate_tuple_id")
                or _binding_map(truth.get("bindings")) != hold_bindings
                or truth.get("value") != evidence.get("value")
                or not _is_int(truth.get("tick"))
                or int(truth["tick"]) != expected_tick
            ):
                return "terminal_hold_truth_mismatch"
        expected_truth_ids = sorted(
            {
                *_ordered_string_list(event.get("supporting_truth_ids")),
                *hold_truth_ids,
            }
        )
        if truth_ids != expected_truth_ids:
            return "outcome_truth_support_is_not_exact"
        if any((episode_id, item) not in self.truth for item in truth_ids):
            return "missing_outcome_truth_support"
        return None

    def _index_continuity_truth(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        """Index recorded endpoints without asserting unknown/missing truth."""
        referenced = {
            (str(row['episode_id']), truth_id)
            for row in self.raw_breaks
            for truth_id in _string_list(row.get('evidence_truth_ids'))
        }
        accepted = {}
        conflicted = set()
        for row in self.raw_continuity_truth:
            key = (_episode_id(row), _text(row.get('truth_id')))
            if key not in referenced:
                continue
            if runtime_schema_errors(row, 'predicate_truth.schema.json'):
                self._reject(row, 'predicate_truth', 'continuity_source_schema_violation')
                continue
            if _evidence_has_forbidden_content(row):
                self._reject(row, 'predicate_truth', 'forbidden_continuity_evidence_source')
                continue
            if key in accepted and dict(accepted[key]) != dict(row):
                conflicted.add(key)
                self._reject(row, 'predicate_truth', 'conflicting_continuity_source_truth')
                continue
            accepted[key] = row
        for key in conflicted:
            accepted.pop(key, None)
        return accepted

    def _index_breaks(self) -> dict[tuple[str, str], Mapping[str, Any]]:
        accepted: dict[tuple[str, str], Mapping[str, Any]] = {}
        conflicted: set[tuple[str, str]] = set()
        for row in self.raw_breaks:
            episode_id = _episode_id(row)
            break_id = _text(row.get("break_id"))
            if episode_id is None or break_id is None:
                self._reject(
                    row, "predicate_continuity_break", "missing_episode_or_break_id"
                )
                continue
            key = (episode_id, break_id)
            reason = self._break_rejection_reason(row)
            if reason:
                self._reject(row, "predicate_continuity_break", reason)
                continue
            if key in accepted and dict(accepted[key]) != dict(row):
                conflicted.add(key)
                self._reject(
                    row, "predicate_continuity_break", "conflicting_duplicate_break_id"
                )
                continue
            accepted[key] = row
        for key in conflicted:
            accepted.pop(key, None)
        return accepted

    def _break_rejection_reason(self, row: Mapping[str, Any]) -> str | None:
        if _evidence_has_forbidden_content(row):
            return "forbidden_authored_evidence_source"
        if not _string_list(row.get("source_refs")):
            return "missing_break_source_refs"
        if (
            not _text(row.get("predicate_id"))
            or not _text(row.get("tuple_id"))
            or not _bindings(row)
        ):
            return "missing_break_predicate_identity_or_bindings"
        from_tick = row.get("from_tick")
        to_tick = row.get("to_tick")
        if (
            not _is_int(from_tick)
            or not _is_int(to_tick)
            or int(to_tick) <= int(from_tick)
        ):
            return "invalid_break_tick_window"
        if not _text(row.get("from_value")) or not _text(row.get("to_value")):
            return "missing_break_values"
        if not _text(row.get("reason")):
            return "missing_break_reason"
        evidence_ids = _ordered_string_list(row.get("evidence_truth_ids"))
        if evidence_ids is None or len(evidence_ids) != 2:
            return "break_requires_two_truth_record_refs"
        episode_id = str(row["episode_id"])
        for index, truth_id in enumerate(evidence_ids):
            truth = self.continuity_truth.get((episode_id, truth_id))
            if truth is None:
                truth = self.truth.get((episode_id, truth_id))
            if truth is None:
                return "missing_break_truth_support"
            endpoint = 'from' if index == 0 else 'to'
            if truth.get('tick') != row[f'{endpoint}_tick'] or truth.get('value') != row[f'{endpoint}_value']:
                return 'break_truth_endpoint_mismatch'
            if (
                str(truth.get("predicate_id")) != str(row.get("predicate_id"))
                or str(truth.get("tuple_id")) != str(row.get("tuple_id"))
                or _bindings(truth) != _bindings(row)
            ):
                return "break_truth_identity_mismatch"
        return None

    def _add_transition_closure(
        self,
        bucket: "_GraphBucket",
        transition: Mapping[str, Any],
        referenced_truth: set[tuple[str, str]],
    ) -> None:
        bucket.add_transition(transition)
        episode_id = str(transition["episode_id"])
        for truth_id in _string_list(transition.get("evidence_truth_ids")):
            truth = self.truth[(episode_id, truth_id)]
            referenced_truth.add((episode_id, truth_id))
            bucket._add_provenance(truth)
            if truth.get("value") == "true":
                bucket.add_truth(truth, self.numeric)
                bucket.add_support_edge(transition, truth)

    def _add_event_closure(
        self,
        bucket: "_GraphBucket",
        event: Mapping[str, Any],
        referenced_truth: set[tuple[str, str]],
    ) -> None:
        bucket.add_event(event)
        episode_id = str(event["episode_id"])
        for transition_id in _string_list(event.get("supporting_transition_ids")):
            transition = self.transitions[(episode_id, transition_id)]
            self._add_transition_closure(bucket, transition, referenced_truth)
            bucket.add_support_edge(event, transition)
        for truth_id in _string_list(event.get("supporting_truth_ids")):
            truth = self.truth[(episode_id, truth_id)]
            referenced_truth.add((episode_id, truth_id))
            bucket._add_provenance(truth)
            if truth.get("value") == "true":
                bucket.add_truth(truth, self.numeric)
                bucket.add_support_edge(event, truth)

    def _add_lifecycle_closure(
        self,
        bucket: "_GraphBucket",
        row: Mapping[str, Any],
        referenced_truth: set[tuple[str, str]],
    ) -> None:
        episode_id = str(row["episode_id"])
        event = self.events[(episode_id, str(row["event_id"]))]
        self._add_event_closure(bucket, event, referenced_truth)
        bucket.add_lifecycle(row, event)
        for transition_id in _string_list(row.get("supporting_transition_ids")):
            transition = self.transitions[(episode_id, transition_id)]
            self._add_transition_closure(bucket, transition, referenced_truth)
            bucket.add_support_edge(row, transition)
        for truth_id in _string_list(row.get("supporting_truth_ids")):
            truth = self.truth[(episode_id, truth_id)]
            referenced_truth.add((episode_id, truth_id))
            bucket._add_provenance(truth)
            if truth.get("value") == "true":
                bucket.add_truth(truth, self.numeric)
                bucket.add_support_edge(row, truth)

    def _add_break_closure(
        self,
        bucket: "_GraphBucket",
        continuity_break: Mapping[str, Any],
        referenced_truth: set[tuple[str, str]],
    ) -> None:
        bucket.add_break(continuity_break)
        episode_id = str(continuity_break["episode_id"])
        for truth_id in _string_list(continuity_break.get("evidence_truth_ids")):
            truth = self.continuity_truth.get((episode_id, truth_id))
            if truth is None:
                truth = self.truth.get((episode_id, truth_id))
            if truth is None:
                raise ValueError('Accepted continuity break lost its recorded endpoint')
            referenced_truth.add((episode_id, truth_id))
            bucket._add_provenance(truth)


class _GraphBucket:
    def __init__(self, episode_id: str, anchor_tick: int) -> None:
        self.episode_id = episode_id
        self.anchor_tick = anchor_tick
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.provenance: dict[str, dict[str, Any]] = {}

    @property
    def has_semantic_content(self) -> bool:
        return any(node["kind"] != "entity" for node in self.nodes.values())

    def add_truth(
        self,
        row: Mapping[str, Any],
        numeric_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> str:
        truth_id = str(row["truth_id"])
        node_id = _node_id(self.episode_id, "predicate_assertion", truth_id)
        provenance_ref = self._add_provenance(row)
        evidence = row.get("evidence", {})
        source_refs = (
            _string_list(evidence.get("source_refs"))
            if isinstance(evidence, _AbcMapping)
            else []
        )
        self.nodes[node_id] = {
            "node_id": node_id,
            "kind": "predicate_assertion",
            "source_record_id": truth_id,
            "predicate_id": str(row["predicate_id"]),
            "tuple_id": str(row["tuple_id"]),
            "tick": int(row["tick"]),
            "truth_value": "true",
            "bindings": _bindings(row),
            "evidence_refs": sorted(set(source_refs)),
            "provenance_refs": [provenance_ref],
        }
        self._add_binding_edges(node_id, row, "asserts_for")

        observations = (
            evidence.get("observations", ()) if isinstance(evidence, _AbcMapping) else ()
        )
        for observation in observations if isinstance(observations, _AbcSequence) else ():
            if not isinstance(observation, _AbcMapping):
                continue
            path = _text(observation.get("path"))
            value = observation.get("value")
            if path is None or not _is_finite_number(value):
                continue
            evidence_id = stable_identifier(
                "inline_numeric_evidence",
                self.episode_id,
                truth_id,
                path,
                value,
            )
            evidence_node_id = _node_id(
                self.episode_id, "numeric_evidence", evidence_id
            )
            self.nodes[evidence_node_id] = {
                "node_id": evidence_node_id,
                "kind": "numeric_evidence",
                "source_record_id": evidence_id,
                "metric_path": path,
                "tick": int(row["tick"]),
                "numeric_values": [{"path": path, "value": value}],
                "source_refs": sorted(set(source_refs)),
                "provenance_refs": [provenance_ref],
            }
            self._add_edge(
                "supported_by_numeric_evidence",
                node_id,
                evidence_node_id,
                path,
                (provenance_ref,),
            )

        for evidence_ref in source_refs:
            numeric_row = numeric_index.get((self.episode_id, evidence_ref))
            if numeric_row is None:
                continue
            numeric_node_id = self.add_numeric_evidence(numeric_row)
            self._add_edge(
                "supported_by_numeric_evidence",
                node_id,
                numeric_node_id,
                str(numeric_row.get("metric_id", "numeric_evidence")),
                (provenance_ref, self._record_provenance_ref(numeric_row)),
            )
        return node_id

    def add_numeric_evidence(self, row: Mapping[str, Any]) -> str:
        record_id = _record_id(row)
        assert record_id is not None
        node_id = _node_id(self.episode_id, "numeric_evidence", record_id)
        provenance_ref = self._add_provenance(row)
        leaves = [
            {"path": path, "value": value}
            for path, value in _numeric_leaves(row)
            if path not in {"tick"}
        ]
        self.nodes[node_id] = {
            "node_id": node_id,
            "kind": "numeric_evidence",
            "source_record_id": record_id,
            "metric_id": str(
                row.get("metric_id", row.get("observation_family", "numeric_evidence"))
            ),
            "tick": int(row["tick"]) if _is_int(row.get("tick")) else None,
            "subject_id": row.get("subject_id"),
            "object_id": row.get("object_id"),
            "unit": row.get("unit"),
            "numeric_values": leaves,
            "source_refs": sorted(set(_string_list(row.get("source_refs")))),
            "provenance_refs": [provenance_ref],
        }
        return node_id

    def add_transition(self, row: Mapping[str, Any]) -> str:
        transition_id = str(row["transition_id"])
        node_id = _node_id(self.episode_id, "predicate_transition", transition_id)
        provenance_ref = self._add_provenance(row)
        self.nodes[node_id] = {
            "node_id": node_id,
            "kind": "predicate_transition",
            "source_record_id": transition_id,
            "predicate_id": str(row["predicate_id"]),
            "tuple_id": str(row["tuple_id"]),
            "from_tick": int(row["from_tick"]),
            "to_tick": int(row["to_tick"]),
            "from_value": str(row["from_value"]),
            "to_value": str(row["to_value"]),
            "direction": str(row["direction"]),
            "bindings": _bindings(row),
            "evidence_truth_ids": sorted(_string_list(row.get("evidence_truth_ids"))),
            "provenance_refs": [provenance_ref],
        }
        self._add_binding_edges(node_id, row, "transitions_for")
        return node_id

    def add_break(self, row: Mapping[str, Any]) -> str:
        break_id = str(row["break_id"])
        node_id = _node_id(self.episode_id, "predicate_continuity_break", break_id)
        provenance_ref = self._add_provenance(row)
        self.nodes[node_id] = {
            "node_id": node_id,
            "kind": "predicate_continuity_break",
            "source_record_id": break_id,
            "break_id": break_id,
            "predicate_id": str(row["predicate_id"]),
            "tuple_id": str(row["tuple_id"]),
            "bindings": _bindings(row),
            "from_tick": int(row["from_tick"]),
            "to_tick": int(row["to_tick"]),
            "from_value": str(row["from_value"]),
            "to_value": str(row["to_value"]),
            "reason": str(row["reason"]),
            "evidence_truth_ids": sorted(_string_list(row.get("evidence_truth_ids"))),
            "source_refs": sorted(set(_string_list(row.get("source_refs")))),
            "provenance_refs": [provenance_ref],
        }
        self._add_binding_edges(node_id, row, "break_for")
        return node_id

    def add_event(self, row: Mapping[str, Any]) -> str:
        event_id = str(row["event_id"])
        node_id = _node_id(self.episode_id, "event_occurrence", event_id)
        provenance_ref = self._add_provenance(row)
        self.nodes[node_id] = {
            "node_id": node_id,
            "kind": "event_occurrence",
            "source_record_id": event_id,
            "event_type_id": str(row["event_type_id"]),
            "rule_id": str(row["rule_id"]),
            "trigger_tick": int(row["trigger_tick"]),
            "start_tick": int(row["start_tick"]),
            "end_tick": int(row["end_tick"]),
            "bindings": _bindings(row),
            "event_phases": [
                dict(phase)
                for phase in row.get("event_phases", ())
                if isinstance(phase, _AbcMapping)
            ],
            "provenance_refs": [provenance_ref],
        }
        self._add_binding_edges(node_id, row, "has_participant")
        self._materialize_handover_process(row)
        return node_id

    def _materialize_handover_process(self, row: Mapping[str, Any]) -> None:
        if row.get("event_type_id") != "communication.communication_handover_event":
            return
        bindings = _bindings(row)
        required_roles = {
            "handover",
            "session",
            "source_station",
            "target_station",
        }
        if set(bindings) != required_roles:
            raise ValueError(
                "communication handover event requires exact process bindings"
            )
        provenance_ref = self._record_provenance_ref(row)
        handover_node_id = self._add_ontology_individual(
            bindings["handover"],
            "world:Handover",
            row,
        )
        session_node_id = self._add_ontology_individual(
            bindings["session"],
            "world:CommunicationSession",
            row,
        )
        source_station_node_id = self._add_entity(bindings["source_station"], row)
        target_station_node_id = self._add_entity(bindings["target_station"], row)
        for property_id, source_node_id, target_node_id, source_id, target_id in (
            (
                "dom:handoverTransfersSession",
                handover_node_id,
                session_node_id,
                bindings["handover"],
                bindings["session"],
            ),
            (
                "dom:handoverSourceStation",
                handover_node_id,
                source_station_node_id,
                bindings["handover"],
                bindings["source_station"],
            ),
            (
                "dom:handoverTargetStation",
                handover_node_id,
                target_station_node_id,
                bindings["handover"],
                bindings["target_station"],
            ),
            (
                "dom:sessionUsesStation",
                session_node_id,
                target_station_node_id,
                bindings["session"],
                bindings["target_station"],
            ),
        ):
            self._add_edge(
                "ontology_relation",
                source_node_id,
                target_node_id,
                property_id,
                (provenance_ref,),
                ontology_property_id=property_id,
                source_individual_id=source_id,
                target_individual_id=target_id,
            )

    def _add_ontology_individual(
        self,
        individual_id: str,
        ontology_class_id: str,
        source_row: Mapping[str, Any],
    ) -> str:
        node_id = _node_id(self.episode_id, "ontology_individual", individual_id)
        provenance_ref = self._add_provenance(source_row)
        existing = self.nodes.get(node_id)
        if existing is None:
            self.nodes[node_id] = {
                "node_id": node_id,
                "kind": "ontology_individual",
                "individual_id": individual_id,
                "ontology_class_id": ontology_class_id,
                "episode_id": self.episode_id,
                "provenance_refs": [provenance_ref],
            }
        elif existing.get("ontology_class_id") != ontology_class_id:
            raise ValueError(
                f"ontology individual class conflict: {individual_id}"
            )
        else:
            existing["provenance_refs"] = sorted(
                set(existing.get("provenance_refs", ())) | {provenance_ref}
            )
        return node_id

    def add_lifecycle(self, row: Mapping[str, Any], event: Mapping[str, Any]) -> str:
        kind, record_id = _lifecycle_identity(row)
        if kind != "event_outcome":
            raise ValueError(f"unsupported event lifecycle node kind: {kind}")
        node_id = _node_id(self.episode_id, kind, record_id)
        provenance_ref = self._add_provenance(row, dependencies=(event,))
        node: dict[str, Any] = {
            "node_id": node_id,
            "kind": kind,
            "source_record_id": record_id,
            "event_id": str(row["event_id"]),
            "event_type_id": str(
                row.get("event_type_id", event.get("event_type_id", ""))
            ),
            "provenance_refs": [provenance_ref],
        }
        node.update(
            status=str(row.get("status", "")),
            terminal_tick=int(row["terminal_tick"])
            if _is_int(row.get("terminal_tick"))
            else None,
        )
        edge_kind = "outcome_of"
        self.nodes[node_id] = node
        event_node_id = self.add_event(event)
        self._add_edge(
            edge_kind,
            node_id,
            event_node_id,
            edge_kind,
            (provenance_ref, self._record_provenance_ref(event)),
        )
        self._add_binding_edges(node_id, event, "inherits_participant")
        return node_id

    def add_support_edge(
        self,
        semantic_row: Mapping[str, Any],
        evidence_row: Mapping[str, Any],
    ) -> None:
        semantic_kind, semantic_id = _semantic_identity(semantic_row)
        evidence_kind, evidence_id = _semantic_identity(evidence_row)
        semantic_node = _node_id(self.episode_id, semantic_kind, semantic_id)
        evidence_node = _node_id(self.episode_id, evidence_kind, evidence_id)
        self._add_edge(
            f"supported_by_{evidence_kind}",
            semantic_node,
            evidence_node,
            "supported_by",
            (
                self._record_provenance_ref(semantic_row),
                self._record_provenance_ref(evidence_row),
            ),
        )

    def _add_binding_edges(
        self, node_id: str, row: Mapping[str, Any], edge_kind: str
    ) -> None:
        provenance_ref = self._record_provenance_ref(row)
        for role, entity_id in sorted(_bindings(row).items()):
            entity_node_id = self._add_entity(entity_id, row)
            self._add_edge(
                edge_kind,
                node_id,
                entity_node_id,
                role,
                (provenance_ref,),
                role=role,
            )

    def _add_entity(self, entity_id: str, source_row: Mapping[str, Any]) -> str:
        node_id = _node_id(self.episode_id, "entity", entity_id)
        provenance_ref = self._add_provenance(source_row)
        existing = self.nodes.get(node_id)
        if existing is None:
            self.nodes[node_id] = {
                "node_id": node_id,
                "kind": "entity",
                "entity_id": entity_id,
                "episode_id": self.episode_id,
                "provenance_refs": [provenance_ref],
            }
        else:
            existing["provenance_refs"] = sorted(
                set(existing.get("provenance_refs", ())) | {provenance_ref}
            )
        return node_id

    def _add_edge(
        self,
        kind: str,
        source_node_id: str,
        target_node_id: str,
        label: str,
        provenance_refs: Sequence[str],
        **extra: Any,
    ) -> str:
        refs = sorted(set(provenance_refs))
        edge_id = stable_identifier(
            "compact_edge",
            self.episode_id,
            kind,
            source_node_id,
            target_node_id,
            label,
            refs,
            extra,
        )
        edge = {
            "edge_id": edge_id,
            "kind": kind,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "label": label,
            "provenance_refs": refs,
        }
        edge.update(extra)
        self.edges[edge_id] = edge
        return edge_id

    def _record_provenance_ref(self, row: Mapping[str, Any]) -> str:
        record_id = _record_id(row)
        if record_id is None:
            record_id = digest_object(dict(row))
        return stable_identifier(
            "compact_provenance",
            self.episode_id,
            str(row.get("schema_name", "record")),
            record_id,
        )

    def _add_provenance(
        self,
        row: Mapping[str, Any],
        *,
        dependencies: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        provenance_ref = self._record_provenance_ref(row)
        records = (row, *dependencies)
        source_refs: set[str] = set()
        evidence_refs: set[str] = set()
        rule_digests: set[str] = set()
        parameter_digests: set[str] = set()
        input_digests: set[str] = set()
        dependency_refs: set[str] = set()
        for record in records:
            source_refs.update(_record_source_refs(record))
            evidence_refs.update(_record_evidence_refs(record))
            for key, target in (
                ("rule_digest", rule_digests),
                ("parameter_digest", parameter_digests),
                ("input_digest", input_digests),
            ):
                value = record.get(key)
                if isinstance(value, str) and value:
                    target.add(str(value))
            if record is not row:
                dependency_refs.add(self._record_provenance_ref(record))
        entry = {
            "provenance_ref": provenance_ref,
            "source_record_id": _record_id(row) or "<digest-addressed>",
            "source_schema_name": str(row.get("schema_name", "record")),
            "record_digest": digest_object(dict(row)),
            "source_refs": sorted(source_refs),
            "evidence_refs": sorted(evidence_refs),
            "rule_digests": sorted(rule_digests),
            "parameter_digests": sorted(parameter_digests),
            "input_digests": sorted(input_digests),
            "dependency_provenance_refs": sorted(dependency_refs),
        }
        self.provenance[provenance_ref] = entry
        return provenance_ref

    def finish(self) -> dict[str, Any]:
        nodes = sorted(self.nodes.values(), key=_node_sort_key)
        edges = sorted(self.edges.values(), key=_edge_sort_key)
        provenance = sorted(
            self.provenance.values(), key=lambda row: row["provenance_ref"]
        )
        ticks: list[int] = []
        for node in nodes:
            for key in (
                "tick",
                "from_tick",
                "to_tick",
                "trigger_tick",
                "start_tick",
                "end_tick",
                "hold_end_tick",
                "terminal_tick",
            ):
                if _is_int(node.get(key)):
                    ticks.append(int(node[key]))
        if not ticks:
            ticks.append(self.anchor_tick)
        record_refs = sorted(entry["provenance_ref"] for entry in provenance)
        graph_id = stable_identifier(
            "compact_graph",
            self.episode_id,
            self.anchor_tick,
            min(ticks),
            max(ticks),
            record_refs,
        )
        payload: dict[str, Any] = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "anchor_tick": self.anchor_tick,
            "window": {
                "start_tick": min(ticks),
                "end_tick": max(ticks),
            },
            "graph_id": graph_id,
            "nodes": nodes,
            "edges": edges,
            "provenance": provenance,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "entity_count": sum(node["kind"] == "entity" for node in nodes),
                "true_assertion_count": sum(
                    node["kind"] == "predicate_assertion" for node in nodes
                ),
                "transition_count": sum(
                    node["kind"] == "predicate_transition" for node in nodes
                ),
                "continuity_break_count": sum(
                    node["kind"] == "predicate_continuity_break" for node in nodes
                ),
                "event_count": sum(
                    node["kind"] == "event_occurrence" for node in nodes
                ),
                "ontology_individual_count": sum(
                    node["kind"] == "ontology_individual" for node in nodes
                ),
                "ontology_relation_count": sum(
                    edge["kind"] == "ontology_relation" for edge in edges
                ),
                "numeric_evidence_count": sum(
                    node["kind"] == "numeric_evidence" for node in nodes
                ),
            },
        }
        payload["graph_digest"] = digest_object(payload)
        return payload


def _truth_rejection_reason(
    row: Mapping[str, Any],
    numeric_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str | None:
    if row.get("value") not in {"true", "false"}:
        return "non_boolean_truth_is_not_graph_evidence"
    if row.get("authoritative_tick") is not True or not _is_int(row.get("tick")):
        return "truth_is_not_at_an_authoritative_tick"
    if (
        not _text(row.get("predicate_id"))
        or not _text(row.get("tuple_id"))
        or not _bindings(row)
    ):
        return "missing_predicate_identity_or_bindings"
    if str(row["predicate_id"]) not in GRAPH_PREDICATE_VOCABULARY:
        return "predicate_outside_ontology_vocabulary"
    evidence = row.get("evidence")
    if not isinstance(evidence, _AbcMapping):
        return "missing_truth_evidence"
    if _string_list(evidence.get("missing_requirements")):
        return "truth_has_missing_evidence_requirements"
    source_refs = _string_list(evidence.get("source_refs"))
    if not source_refs:
        return "truth_has_no_source_refs"
    if _evidence_has_forbidden_content(row):
        return "forbidden_authored_evidence_source"
    episode_id = str(row["episode_id"])
    for source_ref in source_refs:
        numeric = numeric_index.get((episode_id, source_ref))
        if numeric is not None and not _numeric_leaves(numeric):
            return "referenced_numeric_evidence_is_invalid"
    return None


def _semantic_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    schema_name = str(row.get("schema_name", ""))
    if schema_name == "predicate_truth" or "truth_id" in row:
        return "predicate_assertion", str(row["truth_id"])
    if schema_name == "predicate_transition" or "transition_id" in row:
        return "predicate_transition", str(row["transition_id"])
    if schema_name == "predicate_continuity_break" or "break_id" in row:
        return "predicate_continuity_break", str(row["break_id"])
    if schema_name == "event_occurrence" or (
        "event_id" in row and "trigger_tick" in row and "rule_id" in row
    ):
        return "event_occurrence", str(row["event_id"])
    return _lifecycle_identity(row)


def _lifecycle_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    if "outcome_id" in row:
        return "event_outcome", str(row["outcome_id"])
    raise ValueError("record is not an event lifecycle record")


def _record_id(row: Mapping[str, Any]) -> str | None:
    kind_specific_key = {
        "predicate_truth": "truth_id",
        "predicate_transition": "transition_id",
        "predicate_continuity_break": "break_id",
        "event_occurrence": "event_id",
        "event_outcome": "outcome_id",
        "mechanism_observation": "observation_id",
    }.get(row.get("schema_name"))
    if kind_specific_key is not None:
        value = _text(row.get(kind_specific_key))
        if value is not None:
            return value
    for key in _RECORD_ID_KEYS:
        value = _text(row.get(key))
        if value is not None:
            return value
    return None


def _episode_id(row: Mapping[str, Any]) -> str | None:
    return _text(row.get("episode_id"))


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _bindings(row: Mapping[str, Any]) -> dict[str, str]:
    return _binding_map(row.get("bindings")) or {}


def _binding_map(value: Any) -> dict[str, str] | None:
    if not isinstance(value, _AbcMapping) or not value:
        return None
    if any(
        not isinstance(role, str)
        or not role
        or not isinstance(entity_id, str)
        or not entity_id
        for role, entity_id in value.items()
    ):
        return None
    return {role: value[role] for role in sorted(value)}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, _AbcSequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return sorted({str(item) for item in value if isinstance(item, str) and item})


def _ordered_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if any(not isinstance(item, str) or not item for item in value):
        return None
    if len(value) != len(set(value)):
        return None
    return list(value)


def _runtime_record_header_rejection_reason(
    row: Mapping[str, Any],
    *,
    schema_name: str,
    annotation_layer: str,
    source_layer: str,
) -> str | None:
    if row.get("schema_name") != schema_name:
        return "invalid_runtime_schema_name"
    if row.get("schema_version") != "3.0.0":
        return "invalid_runtime_schema_version"
    if row.get("annotation_layer") != annotation_layer:
        return "invalid_runtime_annotation_layer"
    if row.get("source_layer") != source_layer:
        return "invalid_runtime_source_layer"
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _delta_operation_order(operation: str) -> int:
    return {
        "add_predicate_truth": 0,
        "invalidate_predicate_truth": 1,
        "restore_predicate_truth": 2,
        "set_predicate_truth": 3,
    }[operation]


def _numeric_leaves(value: Any, path: str = "") -> list[tuple[str, int | float]]:
    leaves: list[tuple[str, int | float]] = []
    if _is_finite_number(value):
        leaves.append((path or "value", value))
    elif isinstance(value, _AbcMapping):
        for key in sorted(value, key=str):
            if str(key) in {
                "schema_version",
                "episode_id",
                "tick",
                "observation_id",
                "evidence_id",
                "numeric_evidence_id",
                "record_id",
            }:
                continue
            child_path = f"{path}.{key}" if path else str(key)
            leaves.extend(_numeric_leaves(value[key], child_path))
    elif isinstance(value, _AbcSequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            leaves.extend(_numeric_leaves(item, child_path))
    return leaves


def _evidence_has_forbidden_content(
    row: Mapping[str, Any],
    *,
    include_entire_record: bool = False,
) -> bool:
    if include_entire_record:
        payloads: Sequence[Any] = (row,)
    else:
        payloads = (
            row.get("evidence"),
            row.get("source_refs"),
            row.get("supporting_parameter_refs"),
            row.get("evidence_refs"),
            row.get("observation_ids"),
        )
    return any(_value_has_forbidden_fragment(payload) for payload in payloads)


def _value_has_forbidden_fragment(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower().replace("\\", "/")
        return any(fragment in lowered for fragment in _FORBIDDEN_EVIDENCE_FRAGMENTS)
    if isinstance(value, _AbcMapping):
        return any(
            _value_has_forbidden_fragment(key) or _value_has_forbidden_fragment(item)
            for key, item in value.items()
        )
    if isinstance(value, _AbcSequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_value_has_forbidden_fragment(item) for item in value)
    return False


def _record_source_refs(row: Mapping[str, Any]) -> list[str]:
    refs = set(_string_list(row.get("source_refs")))
    evidence = row.get("evidence")
    if isinstance(evidence, _AbcMapping):
        refs.update(_string_list(evidence.get("source_refs")))
    return sorted(refs)


def _record_evidence_refs(row: Mapping[str, Any]) -> list[str]:
    refs: set[str] = set()
    for key in (
        "evidence_truth_ids",
        "supporting_truth_ids",
        "supporting_transition_ids",
        "evidence_refs",
        "observation_ids",
        "supporting_parameter_refs",
    ):
        refs.update(_string_list(row.get(key)))
    evidence = row.get("evidence")
    if isinstance(evidence, _AbcMapping):
        refs.update(_string_list(evidence.get("source_refs")))
    return sorted(refs)


def _node_id(episode_id: str, kind: str, source_id: str) -> str:
    return stable_identifier("compact_node", episode_id, kind, source_id)


def _truth_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("episode_id", "")),
        int(row.get("tick", -1)),
        str(row.get("predicate_id", "")),
        str(row.get("tuple_id", "")),
        str(row.get("truth_id", "")),
    )


def _node_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (str(row.get("kind", "")), str(row.get("node_id", "")))


def _edge_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("kind", "")),
        str(row.get("source_node_id", "")),
        str(row.get("target_node_id", "")),
        str(row.get("edge_id", "")),
    )


def _graph_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("episode_id", "")),
        int(row.get("anchor_tick", -1)),
        str(row.get("graph_id", "")),
    )


__all__ = [
    "BASE_GRAPH_SCHEMA_NAME",
    "CompactTruthGraphBuildResult",
    "DELTA_SCHEMA_NAME",
    "GRAPH_PREDICATE_VOCABULARY",
    "build_compact_truth_graph_result",
    "build_compact_truth_graphs",
    "replay_predicate_state",
    "serialize_compact_truth_graph_jsonl",
    "serialize_semantic_graph_deltas_jsonl",
    "write_compact_truth_graph_jsonl",
]
