"""Run current ontology event rules on measured X predicate evidence."""

import json
from collections import defaultdict

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types, get_core_predicate_templates,
    get_governed_parameter_defaults,
)
from Dataset.semantic_truth.minimal_semantics import (
    Assertion, ObservationRow, Transition, Truth, GROUNDED_ROLE_TUPLE_PREFIX,
    build_events, build_outcomes,
)
from Dataset.semantic_truth.minimal_semantics_adapter import _project_event_role_bindings


def materialize_x_events(graph, prefix: str) -> dict:
    """Derive only events whose complete onset proof exists in this window.

    Raw X business onsets remain predicate evidence. Formal event types come
    from the executable registry and its actual class/support/terminal rules.
    """
    templates = {r["id"]: r for r in get_core_predicate_templates()}
    rules = get_core_event_occurrence_types()
    assertions, transitions = [], []
    truth_nodes, transition_nodes, tuples = {}, {}, {}
    roster = {}
    for node in list(graph.node_index.values()):
        if node["node_type"] == "EntityIdentity":
            raw = node["attrs"].get("_participant_scope")
            if isinstance(raw, dict):
                roster[node["attrs"]["entity_id"]] = raw
        if node["node_type"] != "PredicateAssertion" or node["category_id"] not in templates:
            continue
        tick, attrs = node["tick"], node["attrs"]
        if tick % 5 or attrs.get("assertion_layer") not in {"sampled", "dense"}:
            continue
        pid, bindings = node["category_id"], attrs["bindings"]
        roles = tuple(r["key"] for r in templates[pid]["argument_roles"])
        if set(bindings) != set(roles):
            raise ValueError(f"X event input has an incomplete predicate binding: {node['node_id']}")
        participants = tuple(bindings[role] for role in roles)
        tuple_id = json.dumps([pid, bindings], sort_keys=True, separators=(",", ":"))
        key = pid, tuple_id, tick
        if key in truth_nodes:
            raise ValueError(f"X event input has ambiguous sampled evidence: {key}")
        tuples[pid, tuple_id] = roles, participants
        truth_nodes[key] = node
        assertions.append(Assertion(graph.scope["episode_id"], tick, tuple_id,
            GROUNDED_ROLE_TUPLE_PREFIX + "__".join(roles), participants, pid,
            Truth(attrs["truth_status"]), "deterministic_derived",
            (node["source_file"] + (f":{node['source_line']}" if node["source_line"] else ""),), None))
    for node in list(graph.node_index.values()):
        if node["node_type"] != "PredicateTransition" or node["category_id"] not in templates:
            continue
        pid, attrs = node["category_id"], node["attrs"]
        tuple_id = json.dumps([pid, attrs["bindings"]], sort_keys=True, separators=(",", ":"))
        before, after = attrs["from_tick"], attrs["to_tick"]
        if after - before != 5:
            continue
        if (pid, tuple_id, before) not in truth_nodes or (pid, tuple_id, after) not in truth_nodes:
            continue
        values = attrs["from_value"], attrs["to_value"]
        if values not in {("false", "true"), ("true", "false")}:
            continue
        roles, participants = tuples[pid, tuple_id]
        direction = "rising" if values[1] == "true" else "falling"
        transitions.append(Transition(graph.scope["episode_id"], tuple_id,
            GROUNDED_ROLE_TUPLE_PREFIX + "__".join(roles), participants, pid,
            before, after, Truth(values[0]), Truth(values[1]), direction, None))
        transition_nodes[pid, tuple_id, after] = node
    events = build_events([], assertions, transitions, rules, roster_by_id=roster)
    outcomes = build_outcomes(events, assertions, get_governed_parameter_defaults(),
                              rules, transitions=transitions)
    rules_by_id = {r["rule_id"]: r for r in rules}
    by_event = {r.event_id: r for r in outcomes}
    counts = defaultdict(int)
    identities = {n["attrs"]["entity_id"]: n for n in graph.node_index.values()
                  if n["node_type"] == "EntityIdentity"}
    rows_by_id = {}
    for assertion in assertions:
        node = truth_nodes[assertion.predicate_id, assertion.tuple_id, assertion.tick]
        rows_by_id[assertion.episode_id, assertion.tick, assertion.tuple_id] = ObservationRow(
            assertion.episode_id, assertion.tick, assertion.tuple_id, assertion.tuple_type,
            assertion.participants, assertion.truth != Truth.OUT_OF_SCOPE, assertion.source_class,
            assertion.source_refs, assertion.mechanism_id,
            {"bindings": node["attrs"]["bindings"],
             "binding_ontology_classes": node["attrs"]["binding_ontology_classes"],
             "world_truth_value": assertion.truth.value})
    for event in events:
        rule = rules_by_id[event.rule_id]
        trigger = rule["trigger_predicate_id"]
        support = transition_nodes[trigger, event.tuple_id, event.support_transition_tick]
        event_id = prefix + "registered_event:" + event.event_id
        bindings = _project_event_role_bindings(event, rule,
            {"episode_id": event.episode_id, "predicate_id": trigger, "tuple_id": event.tuple_id,
             "to_tick": event.trigger_tick, "to_value": support["attrs"]["to_value"],
             "bindings": support["attrs"]["bindings"]}, rows_by_id, roster)
        source, line = support["source_file"], support["source_line"]
        graph.add_node(event_id, "EventOccurrence", event.event_type, event.detection_tick,
            {"native_event_id": event.event_id, "event_family_id": event.event_family_id,
             "bindings": bindings, "rule_id": event.rule_id, "trigger_tick": event.trigger_tick,
             "detection_tick": event.detection_tick, "start_tick": event.trigger_tick,
             "end_tick": event.detection_tick, "derivation": "current_rule_on_measured_X_predicates",
             "rule_source": "Dataset/semantic_rules/profiles/event_family_authority_mapping.json"},
            source, line, rule_ref=event.rule_id)
        graph.add_edge("at_time", event_id, prefix + "t:" + str(event.detection_tick),
                       role="detection_tick", source_file=source, source_line=line)
        graph.add_edge("transition_supports_event", support["node_id"], event_id,
                       source_file=source, source_line=line, rule_ref=event.rule_id)
        proof_nodes = [truth_nodes[trigger, event.tuple_id, support["attrs"]["from_tick"]],
                       truth_nodes[trigger, event.tuple_id, support["attrs"]["to_tick"]]]
        proof_nodes += [truth_nodes[p.predicate_id, p.tuple_id, a.tick]
                        for p in event.support_condition_proofs for a in p.assertions]
        for proof in {n["node_id"]: n for n in proof_nodes}.values():
            graph.add_edge("assertion_supports_event", proof["node_id"], event_id,
                           source_file=proof["source_file"], source_line=proof["source_line"], rule_ref=event.rule_id)
        for role, entity in bindings.items():
            identity = identities.get(entity)
            if identity is None or identity["attrs"]["identity_status"] != "resolved":
                raise ValueError(f"registered X event participant is unresolved: {event_id}:{role}:{entity}")
            graph.add_edge("event_participant", event_id, identity["node_id"], role=role,
                           source_file=source, source_line=line)
        outcome = by_event[event.event_id]
        terminal = rule["terminal"]
        terminal_roles = tuple(terminal["join_roles"])
        relevant = [a for a in assertions if a.predicate_id == terminal["predicate_id"]
                    and all(dict(zip(tuples[a.predicate_id, a.tuple_id][0], a.participants)).get(r) == bindings.get(r)
                            for r in terminal_roles) and a.tick >= event.trigger_tick]
        end = graph.scope["end_tick"]
        known = {a.tick for a in relevant if a.truth in {Truth.TRUE, Truth.FALSE}}
        proof_end = outcome.terminal_tick if outcome.outcome == "success" else end
        missing = any(t not in known for t in range(event.trigger_tick, proof_end + 1, 5))
        status = "succeeded" if outcome.outcome == "success" else "unknown" if missing else "right_censored"
        outcome_id = event_id + ":outcome"
        graph.add_node(outcome_id, "EventOutcome", event.event_type, outcome.terminal_tick,
            {"native_event_id": event.event_id, "status": status,
             "terminal_predicate_id": outcome.terminal_predicate_id,
             "censor_tick": end if status == "right_censored" else None,
             "target_missing": missing, "derivation": "current_terminal_rule_on_measured_X_predicates"},
            source, line, rule_ref=event.rule_id)
        graph.add_edge("event_outcome", event_id, outcome_id, source_file=source, source_line=line)
        for assertion in outcome.terminal_hold_assertions:
            proof = truth_nodes[assertion.predicate_id, assertion.tuple_id, assertion.tick]
            graph.add_edge("assertion_supports_outcome", proof["node_id"], outcome_id,
                           source_file=proof["source_file"], source_line=proof["source_line"])
        counts[event.event_type] += 1
    return dict(counts)
