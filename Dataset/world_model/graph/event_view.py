"""Event windows with complete critical evidence and explicit history pages."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from Dataset.world_model.graph.records import load_scope
from Dataset.world_model.graph.contract import REPO, iter_jsonl


CRITICAL_TYPES = {"EventOccurrence", "EventOutcome", "EventPhase", "PredicateTransition",
                  "PredicateEpisodeOutcome", "CommandDispatch"}
PROOF_EDGES = {"event_participant", "event_outcome", "event_phase", "assertion_supports_event",
               "transition_supports_event", "assertion_supports_outcome", "transition_supports_outcome",
               "transition_from", "transition_to", "predicate_onset_assertion",
               "predicate_episode_outcome", "transition_supports_phase", "assertion_supports_phase"}
OUTGOING_PROOFS = {"event_participant", "event_outcome", "event_phase", "transition_to",
                   "predicate_onset_assertion", "predicate_episode_outcome"}
INCOMING_PROOFS = PROOF_EDGES - OUTGOING_PROOFS


def _decode(row: dict) -> dict:
    return {**{k: v for k, v in row.items() if k != "attrs_json"},
            "attrs": json.loads(row["attrs_json"])}


def _rows(partition: Path, *, kinds: set[str] | None = None, categories: set[str] | None = None,
          start: int | None = None, end: int | None = None):
    expression = ds.field("node_type").isin(sorted(kinds)) if kinds else None
    if categories is not None:
        category = ds.field("category_id").isin(sorted(categories))
        expression = category if expression is None else expression & category
    if start is not None:
        timed = (ds.field("tick") >= start) & (ds.field("tick") <= end)
        expression = timed if expression is None else expression & timed
    scanner = ds.dataset(partition / "nodes.parquet", format="parquet").scanner(
        filter=expression, batch_size=8192)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            yield _decode(row)


def event_window(partition: Path, *, event_id: str = "", entity_id: str = "",
                 start_tick: int | None = None, end_tick: int | None = None,
                 offset: int = 0, limit: int = 2000) -> dict:
    scope = load_scope(partition)
    identities = list(_rows(partition, kinds={"EntityIdentity"}))
    identity_by_id = {r["attrs"]["entity_id"]: r for r in identities}
    logical_objects = [r for r in _rows(partition, kinds={"ContextAnchor"}, categories={"compute_comm_supplement"})
                       if r["attrs"].get("source_object_field") is not None]
    identity_by_id.update((r["attrs"]["entity_id"], r) for r in logical_objects)
    critical = list(_rows(partition, kinds=CRITICAL_TYPES))
    # X's unregistered business onsets remain visibly predicate-level targets.
    critical.extend(r for r in _rows(partition, kinds={"ContextAnchor"}, categories={"measured_business_predicate_onset"})
                    if r["category_id"] == "measured_business_predicate_onset")
    events = sorted((r for r in critical if r["node_type"] == "EventOccurrence"
                     or r["category_id"] == "measured_business_predicate_onset"),
                    key=lambda r: (r["tick"], r["node_id"]))
    if event_id:
        focus = next((r for r in events if r["node_id"] == event_id), None)
        if focus is None:
            raise ValueError("requested event/onset does not exist in this sample")
    else:
        eligible = [r for r in events if not entity_id or entity_id in (r["attrs"].get("bindings") or {}).values()]
        within = [r for r in eligible if scope["start_tick"] <= r["tick"] <= scope["end_tick"]]
        actor_events = [r for r in within if r["node_type"] == "EventOccurrence" and any(
            identity_by_id.get(entity, {}).get("category_id") in {"uav", "vehicle", "pedestrian"}
            or identity_by_id.get(entity, {}).get("attrs", {}).get("ontology_class") == "world:Crowd"
            for entity in (r["attrs"].get("bindings") or {}).values())]
        focus = actor_events[0] if actor_events else within[0] if within else eligible[0] if eligible else None
    if start_tick is None:
        center = focus["tick"] if focus is not None else scope.get("fork_tick", scope["start_tick"])
        center = min(scope["end_tick"], max(scope["start_tick"], center))
        start_tick, end_tick = max(scope["start_tick"], center - 50), min(scope["end_tick"], center + 75)
    if end_tick is None or not scope["start_tick"] <= start_tick <= end_tick <= scope["end_tick"]:
        raise ValueError("event window exceeds saved sample scope")
    subjects = {entity_id} if entity_id else set((focus["attrs"].get("bindings") or {}).values()) if focus else set()
    if not subjects and scope["source_kind"] == "arm":
        manifest = json.loads((Path(scope["source_root"]) / "manifest.json").read_text())
        subjects.update((manifest.get("scope") or {}).get("entity_ids", ()))
        subjects.update((manifest.get("scope") or {}).get("simulated_entity_ids", ()))
    if subjects - identity_by_id.keys():
        raise ValueError(f"event subject lacks a saved identity: {sorted(subjects - identity_by_id.keys())}")
    subject_groups = {}
    for entity in sorted(subjects):
        a = identity_by_id[entity]["attrs"]
        if a.get("ontology_class") != "world:Crowd":
            continue
        pointer = a.get("identity_basis_path", "")
        match = re.fullmatch(r"line\[(\d+)\]\.values\.crowd_id", pointer)
        if match is None:
            raise ValueError(f"crowd membership has no recorded source location: {entity}")
        source = a["identity_basis"]
        line = int(match.group(1))
        row = next((r for n, r in iter_jsonl(REPO / source) if n == line), None)
        if row is None or row["values"]["crowd_id"] != entity:
            raise ValueError(f"crowd membership source differs: {source}:{line}")
        members = row["values"]["cohort_member_ids"]
        if row["values"]["cohort_member_count"] != len(members) or set(members) - identity_by_id.keys():
            raise ValueError(f"crowd membership identities are incomplete: {source}:{line}")
        subject_groups[entity] = {"members": members, "source_file": source, "source_line": line,
                                 "graph_identity": identity_by_id[entity]["node_id"]}
        subjects.update(members)
    in_window = [r for r in critical if r["tick"] is None or start_tick <= r["tick"] <= end_tick]
    selected = {}
    if focus is not None:
        selected[focus["node_id"]] = focus
    for row in in_window:
        attrs = row["attrs"]
        bound = set((attrs.get("bindings") or {}).values())
        target = attrs.get("target_entity_id")
        if (not subjects or bound & subjects or target in subjects
                or row["node_id"] == (focus or {}).get("node_id")):
            selected[row["node_id"]] = row
    # Event proofs can precede detection or lie beyond the requested drawing
    # window. Preserve them and label their own actual times.
    by_id = {r["node_id"]: r for r in critical}
    edges = []
    edge_filter = ds.field("edge_type").isin(sorted(PROOF_EDGES | {
        "identity_state", "same_entity_next", "same_predicate_next", "action_target", "RelativeState", "observed_dispatch"}))
    scanner = ds.dataset(partition / "edges.parquet", format="parquet").scanner(filter=edge_filter, batch_size=32768)
    for batch in scanner.to_batches():
        edges.extend(_decode(row) for row in batch.to_pylist())
    seeds = set(selected)
    for _ in range(4):
        added = {endpoint for edge in edges
                 if (edge["edge_type"] in OUTGOING_PROOFS and edge["source"] in seeds
                     or edge["edge_type"] in INCOMING_PROOFS and edge["target"] in seeds)
                 for endpoint in (edge["source"], edge["target"])}
        if added <= seeds:
            break
        seeds.update(added)
    selected.update((key, value) for key, value in by_id.items() if key in seeds)
    wanted_predicates = {r["category_id"] for r in selected.values() if r["node_type"] == "PredicateTransition"}
    if focus:
        from Dataset.world_model.graph.event_tasks import task_definitions
        definition = next((r for r in task_definitions() if r["id"] == focus["category_id"]), None)
        if definition:
            wanted_predicates.add(definition["trigger_predicate"])
            if definition["terminal_condition"] is not None:
                wanted_predicates.add(definition["terminal_condition"]["predicate_id"])
        if focus["node_type"] == "ContextAnchor":
            wanted_predicates.add(focus["attrs"]["predicate_id"])
    states, assertions, background = [], [], Counter()
    for row in _rows(partition, kinds={"EntityState", "PredicateAssertion"}, start=start_tick, end=end_tick):
        attrs = row["attrs"]
        if row["node_type"] == "EntityState":
            if attrs["entity_id"] in subjects:
                states.append(row)
            else:
                background["state_records"] += 1
        elif (set(attrs["bindings"].values()) & subjects and row["category_id"] in wanted_predicates):
            assertions.append(row)
        elif row["node_id"] not in seeds:
            background["predicate_records"] += 1
        if row["node_id"] in seeds:
            selected[row["node_id"]] = row
    selected.update((r["node_id"], r) for r in logical_objects if r["node_id"] in seeds)
    missing_proofs = seeds - selected.keys() - {r["node_id"] for r in identities}
    if missing_proofs:
        for row in _rows(partition, kinds={"PredicateAssertion", "EntityState"}):
            if row["node_id"] in missing_proofs:
                selected[row["node_id"]] = row
                missing_proofs.remove(row["node_id"])
            if not missing_proofs:
                break
    if missing_proofs:
        raise ValueError(f"event proof endpoints are missing: {sorted(missing_proofs)}")
    pages, counts = {}, {}
    for name, rows in (("states", states), ("assertions", assertions)):
        rows.sort(key=lambda r: (r["tick"], r["node_id"]))
        pages[name] = rows[offset:offset + limit]
        ticks = [r["tick"] for r in pages[name]]
        counts[name] = {"total": len(rows), "returned": len(pages[name]), "offset": offset,
                       "has_more": offset + limit < len(rows), "next_offset": offset + limit,
                       "covered_ticks": [min(ticks), max(ticks)] if ticks else None}
        selected.update((r["node_id"], r) for r in pages[name])
    for identity in identities:
        if identity["attrs"]["entity_id"] in subjects or identity["node_id"] in seeds:
            selected[identity["node_id"]] = identity
    ids = set(selected)
    # Preserve links across history-page boundaries. Include the other endpoint
    # as context without counting it twice in the paginated state/series rows.
    history_records = {r["node_id"]: r for r in (*states, *assertions)}
    eligible_ids = ids | history_records.keys()
    selected_edges = [e for e in edges
                      if e["source"] in eligible_ids and e["target"] in eligible_ids
                      and (e["source"] in ids or e["target"] in ids)]
    for edge in selected_edges:
        for endpoint in (edge["source"], edge["target"]):
            if endpoint not in selected:
                selected[endpoint] = history_records[endpoint]
    predicates = [{"record_id": r["node_id"], "predicate_id": r["category_id"], "tick": r["tick"],
                   "truth_status": r["attrs"]["truth_status"], "bindings": r["attrs"]["bindings"],
                   "source_file": r["source_file"], "source_line": r["source_line"]}
                  for r in selected.values() if r["node_type"] == "PredicateAssertion"]
    series = defaultdict(list)
    for row in pages["states"]:
        raw = row["attrs"]["raw"]
        position = (raw.get("truth_pose") or {}).get("position_enu_m", raw.get("pos_enu"))
        if not isinstance(position, list) or len(position) != 3:
            continue
        for axis, value in enumerate(position):
            series[row["attrs"]["entity_id"], axis].append({"tick": row["tick"], "value": value,
                "value_status": "present", "record_id": row["node_id"],
                "source_file": row["source_file"], "source_line": row["source_line"]})
    event_rows = [{**r, "record_id": r["node_id"]} for r in selected.values()
                  if r["node_type"] in CRITICAL_TYPES or r["category_id"] == "measured_business_predicate_onset"]
    return {"scope": scope, "window": {"start_tick": start_tick, "end_tick": end_tick},
        "focus": focus, "subjects": sorted(subjects), "subject_groups": subject_groups,
        "event_choices": [{"id": r["node_id"], "type": r["category_id"], "tick": r["tick"],
                           "bindings": r["attrs"].get("bindings"), "kind": r["node_type"]} for r in events],
        "entities": [{"node_id": r["node_id"], "entity_id": r["attrs"]["entity_id"],
            "entity_namespace": r["attrs"]["identity_namespace"], "category_id": r["category_id"],
            "identity_status": r["attrs"]["identity_status"]} for r in identities],
        "nodes": list(selected.values()), "edges": selected_edges, "predicates": predicates,
        "events": event_rows, "timepoints": sorted({r["tick"] for r in selected.values()
                    if r["tick"] is not None and start_tick <= r["tick"] <= end_tick}),
        "series": [{"id": f"position:{entity}:{axis}", "field_id": "position_enu_m",
            "label": f"位置 · {('东','北','上')[axis]}", "canonical_unit": "m", "entity_id": entity,
            "entity_namespace": "sample", "element_positions": [axis], "points": points}
            for (entity, axis), points in sorted(series.items())],
        "collection_counts": counts, "critical_evidence_complete": True,
        "collapsed_background": dict(background),
        "truncation": {"nodes": any(r["has_more"] for r in counts.values()), "edges": False,
                       "series": counts["states"]["has_more"], "predicates": counts["assertions"]["has_more"], "events": False}}
