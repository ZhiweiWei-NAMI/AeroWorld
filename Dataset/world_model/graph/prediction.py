"""Prediction inputs assembled from observed prefixes, separate from supervision."""

from __future__ import annotations

import argparse
import copy
import json
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import pyarrow.dataset as ds
import torch
from torch_geometric.data import HeteroData

from Dataset.world_model.graph.contract import CONTRACT_VERSION, ENTITY_CATEGORIES
from Dataset.world_model.graph.records import load_scope
from Dataset.semantic_truth.formal_clock import FORMAL_TICK_HZ
from Dataset.world_model.graph.observation_logs import (
    CHANNELS as LOG_CHANNELS, KINDS as LOG_KINDS, RECEIVERS as LOG_RECEIVERS,
    FEATURES as LOG_FEATURES, FEATURE_UNITS as LOG_UNITS,
    ENUM_VALUES as LOG_ENUM_VALUES,
    MISSING_REASONS as LOG_MISSING_REASONS, REFERENCE_ROLES as LOG_REFERENCE_ROLES,
    PERSISTENT_KINDS as LOG_PERSISTENT_KINDS,
    POLICY as LOG_POLICY, LOG_VERSION, read_nodes as log_nodes,
)

INPUT_VERSION = "belief-input-5"
STATE_FEATURES = ("position_e_m", "position_n_m", "position_u_m",
                  "velocity_e_mps", "velocity_n_mps", "velocity_u_mps", "yaw_deg")
DEFAULT_OBSERVATION = "structured_logs"
OBSERVATIONS = (DEFAULT_OBSERVATION, "capture_filtered_logs", "structured_physics", "capture_filtered_physics")
OBSERVATION_POLICY = (
    "physical samples, observed cohort membership and mode-selected local monitor logs; simulator dispatches "
    "do not establish receiver knowledge and contribute no nodes, payloads, "
    "identities, candidates or edges; local logs are admitted only at their availability tick")
# Pairwise relation between two states observed at the same tick: the target
# minus the source, ENU metres and backward-difference m/s. Distances are plain
# Euclidean/horizontal/vertical; no rule-specific separation metric or radius.
RELATION_FEATURES = ("relative_position_e_m", "relative_position_n_m", "relative_position_u_m",
                     "euclidean_distance_m", "horizontal_distance_m", "vertical_distance_m",
                     "relative_velocity_e_mps", "relative_velocity_n_mps", "relative_velocity_u_mps")
RELATION_POLICY = ("all ordered pairs of entities observed at the latest observed tick of the prefix; "
                   "an entity not observed at that tick has no relation edge")


class PredictionGraph(HeteroData):
    """Offset subject references when observation graphs are batched."""

    def __inc__(self, key, value, store=None, *args, **kwargs):
        if key == "query_entity_index":
            return self["ObservedEntity"].num_nodes
        return super().__inc__(key, value, store, *args, **kwargs)


def observation_nodes(partition: Path, *, start: int | None = None,
                      end: int | None = None, observation: str = DEFAULT_OBSERVATION) -> list[dict]:
    if observation not in OBSERVATIONS:
        raise ValueError("undeclared observation condition")
    expression = ds.field("node_type") == "EntityState"
    expression |= ((ds.field("node_type") == "ContextAnchor")
                   & (ds.field("category_id") == "crowd_evacuation"))
    if start is not None:
        expression &= (ds.field("tick") >= start) & (ds.field("tick") <= end)
    table = ds.dataset(partition / "nodes.parquet", format="parquet").to_table(filter=expression)
    nodes = [{**r, "attrs": json.loads(r["attrs_json"])} for r in table.to_pylist()]
    if observation.endswith("_logs"):
        nodes.extend(log_nodes(partition, start=start, end=end))
    return nodes


def _vector(value: Any) -> list[float | None]:
    if value is None:
        return [None, None, None]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"physical vector has an undeclared shape: {value!r}")
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in value):
        raise ValueError(f"invalid physical vector: {value!r}")
    return list(value)


def _physical_state(node: dict) -> dict:
    a = node["attrs"]
    raw = a["raw"]
    pose = raw.get("truth_pose")
    if isinstance(pose, dict):
        position = pose.get("position_enu_m")
    elif "values" in raw:
        # L6 serializes state values under an explicit values object; an
        # unobserved record has values=null. Its trajectory copy is flat.
        values = raw["values"]
        if values is not None and not isinstance(values, dict):
            raise ValueError("physical state values have an undeclared shape")
        position = values.get("pos_enu") if values is not None else None
    else:
        position = raw.get("pos_enu")
    return {"entity_id": a["entity_id"], "tick": node["tick"],
            "entity_category": a["entity_category"],
            "values": [*_vector(position), None, None, None, None]}


def _capture_visible(raw: dict) -> bool:
    presence = raw.get("render_presence")
    if isinstance(presence, dict) and presence.get("offstage") is True:
        return False
    visibility = raw.get("uav_visibility")
    if isinstance(visibility, dict) and visibility.get("selected_for_capture_truth") is False:
        return False
    # This condition describes retained physical rows, not formal RGB visibility.
    return True


def _observed_state(node: dict, observation: str) -> dict | None:
    a = node["attrs"]
    if (a.get("state_layer") in ("frozen_parent_reference", "history")
            or a.get("observed") is False or node["tick"] % 5):
        return None
    if observation.startswith("capture_filtered_") and not _capture_visible(a["raw"]):
        return None
    if a["identity_status"] != "resolved":
        raise ValueError(f"observed input state has unresolved identity: {node['node_id']}")
    row = _physical_state(node)
    if row["entity_category"] not in ENTITY_CATEGORIES:
        raise ValueError(f"input category has no executable declaration: {row['entity_category']}")
    return row


def _observed_cohort(node: dict) -> tuple[str, list[str]] | None:
    a = node["attrs"]
    if a.get("observation_family") != "crowd_evacuation" or not isinstance(a.get("raw"), dict):
        return None
    values = a["raw"]["values"]
    members = values["cohort_member_ids"]
    if values["cohort_member_count"] != len(members):
        raise ValueError("observed cohort declaration has inconsistent membership")
    entity = values["crowd_id"]
    if entity is None and not members:
        return None  # Explicitly unavailable cohort, not an observed null entity.
    if not isinstance(entity, str) or not entity or any(not isinstance(m, str) or not m for m in members):
        raise ValueError("observed cohort has invalid identities")
    return entity, members


class ObservationCandidates:
    """Index physical and mode-selected log times; query each history window.

    Candidate selection shares the exact acceptance rules with isolate_records.
    Dispatches cannot introduce identities into either representation.
    """

    def __init__(self, nodes: Iterable[dict], observation: str):
        if observation not in OBSERVATIONS:
            raise ValueError("undeclared observation condition")
        states, cohorts, persistent = defaultdict(list), [], {}
        for node in nodes:
            tick = node["tick"]
            if type(tick) is not int:
                continue
            if node["node_type"] == "EntityState":
                row = _observed_state(node, observation)
                if row is not None:
                    states[row["entity_id"]].append(tick)
            elif node["node_type"] == "ContextAnchor":
                cohort = _observed_cohort(node)
                if cohort is not None:
                    cohorts.append((tick, *cohort))
            elif node["node_type"] == "ReceiverObservation" and observation.endswith("_logs"):
                log = node["attrs"]
                if log["availability_tick"] != tick or log["sample_tick"] > tick:
                    raise ValueError("monitor availability precedes its source sample")
                for references in log["references"].values():
                    for entity in references:
                        if log["kind"] in LOG_PERSISTENT_KINDS:
                            if entity not in persistent or tick < persistent[entity]:
                                persistent[entity] = tick
                        else:
                            states[entity].append(tick)
        self.state_ticks = {entity: sorted(set(ticks)) for entity, ticks in states.items()}
        self.cohorts = cohorts
        self.persistent = persistent

    def entities(self, start_tick: int, cutoff_tick: int) -> set[str]:
        if start_tick > cutoff_tick:
            raise ValueError("invalid observation window")
        known = {entity for entity, tick in self.persistent.items() if tick <= cutoff_tick}
        for entity, ticks in self.state_ticks.items():
            index = bisect_left(ticks, start_tick)
            if index < len(ticks) and ticks[index] <= cutoff_tick:
                known.add(entity)
        groups = {}
        for tick, entity, members in self.cohorts:
            if not start_tick <= tick <= cutoff_tick:
                continue
            if entity in groups and groups[entity] != members:
                raise ValueError("prefix cohort membership changes without an explicit rule")
            groups[entity] = members
        return known | {entity for entity, members in groups.items() if set(members) <= known}


def isolate_records(nodes: Iterable[dict], *, cutoff_tick: int, start_tick: int,
                    observation: str = DEFAULT_OBSERVATION, query: dict | None = None) -> dict:
    """No outcome, predicate label, untimed row, source summary or target edge is read.

    Category and candidate sets come only from accepted prefix states and
    observed cohort membership. Simulator dispatch is execution evidence, not
    receiver knowledge. Trace locations are not tensor inputs.
    """
    if observation not in OBSERVATIONS or start_tick > cutoff_tick:
        raise ValueError("invalid prediction input request")
    states, trace, cohorts, logs = [], {}, {}, []
    for node in nodes:
        tick = node["tick"]
        if type(tick) is not int or tick > cutoff_tick:
            continue
        kind, a = node["node_type"], node["attrs"]
        if tick < start_tick and not (kind == "ReceiverObservation" and observation.endswith("_logs")
                                     and a["kind"] in LOG_PERSISTENT_KINDS):
            continue
        if kind == "EntityState":
            row = _observed_state(node, observation)
            if row is None:
                continue
            row["id"] = "state:" + json.dumps([row["entity_id"], tick, row["entity_category"]], separators=(",", ":"))
            states.append(row)
            trace.setdefault(row["id"], []).append({"graph_node": node["node_id"], "source": node["source_file"], "line": node["source_line"]})
        elif kind == "ContextAnchor":
            cohort = _observed_cohort(node)
            if cohort is None:
                continue
            entity, members = cohort
            earlier = cohorts.get(entity)
            if earlier is not None and earlier != members:
                raise ValueError("prefix cohort membership changes without an explicit rule")
            cohorts[entity] = members
        elif kind == "ReceiverObservation" and observation.endswith("_logs"):
            if a["availability_tick"] != tick or a["sample_tick"] > tick:
                raise ValueError("monitor availability precedes its source sample")
            logs.append(copy.deepcopy(a))
        # CommandDispatch records simulator execution, including exogenous
        # motion, hidden runtime state and render annotations. Neither its
        # timestamp nor an ActionPlan establishes a receiver observation.
    states.sort(key=lambda r: (r["tick"], r["entity_id"], r["id"]))
    # Equivalent physical copies have one input slot. Their trace stays external.
    unique_states = {}
    for row in states:
        existing = unique_states.get(row["id"])
        if existing is not None and existing != row:
            raise ValueError(f"physical observation has conflicting source poses: {row['id']}")
        unique_states[row["id"]] = row
    states = list(unique_states.values())
    # Formal reported motion can use a forward difference at the first sample.
    # Input motion is instead calculated solely from already observed poses.
    previous = {}
    for state in states:
        earlier = previous.get(state["entity_id"])
        position = state["values"][:3]
        if earlier is not None and earlier["tick"] < state["tick"] and all(v is not None for v in position + earlier["values"][:3]):
            dt = (state["tick"] - earlier["tick"]) / FORMAL_TICK_HZ
            velocity = [(position[i] - earlier["values"][i]) / dt for i in range(3)]
            heading = math.degrees(math.atan2(velocity[1], velocity[0])) if velocity[0] != 0 or velocity[1] != 0 else None
        else:
            velocity, heading = [None, None, None], None
        state["values"][3:] = [*velocity, heading]
        previous[state["entity_id"]] = state
    log_entities = {entity for log in logs for references in log["references"].values() for entity in references}
    entities = sorted({r["entity_id"] for r in states} | log_entities)
    cohorts = {entity: members for entity, members in cohorts.items() if set(members) <= set(entities)}
    entities.extend(entity for entity in sorted(cohorts) if entity not in entities)
    if query is not None:
        if set(query) != {"task_id", "query", "bindings", "horizon_ticks"}:
            raise ValueError("prediction query must not contain supervision fields")
        if query["query"] not in {"future_onset", "current_hidden_predicate", "future_terminal"}:
            raise ValueError("undeclared prediction query")
        if (not isinstance(query["bindings"], dict)
                or any(not isinstance(role, str) or not isinstance(entity, str)
                       for role, entity in query["bindings"].items())):
            raise ValueError("prediction query has invalid bindings")
        if not set(query["bindings"].values()) <= set(entities):
            raise ValueError("prediction candidate is not known from the observed prefix")
    # Names often contain scenario intent. Their source trace belongs outside the
    # learner input, just like paths and supervision IDs.
    aliases = {entity: f"entity:{index}" for index, entity in enumerate(entities)}
    for row in states:
        old_id = row["id"]
        row["entity_id"] = aliases[row["entity_id"]]
        row["id"] = f"state:{row['entity_id']}@{row['tick']}"
        trace[row["id"]] = trace.pop(old_id)
    logs.sort(key=lambda log: (log["availability_tick"], log["kind"], log["observation_id"]))
    for index, log in enumerate(logs):
        log_id = f"log:{index}"
        trace[log_id] = {"source_observation": log.pop("observation_id"), "source_refs": log.pop("source_refs")}
        log["id"] = log_id
        log["references"] = {role: [aliases[entity] for entity in references]
                             for role, references in log["references"].items()}
    isolated_query = copy.deepcopy(query)
    if isolated_query is not None:
        isolated_query["bindings"] = {role: aliases[entity] for role, entity in query["bindings"].items()}
    trace["entity_references"] = {alias: entity for entity, alias in aliases.items()}
    return {"input": {"version": INPUT_VERSION, "cutoff_tick": cutoff_tick,
                      "history_start_tick": start_tick, "observation": observation,
                      "query": isolated_query, "entities": list(aliases.values()),
                      "observed_cohorts": {aliases[entity]: [aliases[member] for member in members]
                                           for entity, members in cohorts.items()},
                      "observation_policy": OBSERVATION_POLICY,
                      "log_policy": LOG_POLICY if observation.endswith("_logs") else None,
                      "log_version": LOG_VERSION if observation.endswith("_logs") else None,
                      "logs": logs, "states": states}, "trace": trace}


def read_prediction_view(partition: Path, cutoff_tick: int, *, history_ticks: int = 100,
                         observation: str = DEFAULT_OBSERVATION, query: dict | None = None) -> dict:
    scope = load_scope(partition)
    if scope["contract_version"] != CONTRACT_VERSION:
        raise ValueError("prediction input requires the current graph version")
    if not scope["start_tick"] <= cutoff_tick <= scope["end_tick"] or history_ticks < 0:
        raise ValueError("prediction cutoff exceeds actual sample")
    start = max(scope["start_tick"], cutoff_tick - history_ticks)
    return isolate_records(observation_nodes(partition, start=start, end=cutoff_tick, observation=observation),
        cutoff_tick=cutoff_tick, start_tick=start, observation=observation, query=query)


def to_pyg(view: dict) -> HeteroData:
    """Seven physical channels with masks; latest-state readout and same-tick
    relation edges follow RELATION_POLICY."""
    payload = view["input"]
    graph = PredictionGraph()
    identities = {entity: i for i, entity in enumerate(payload["entities"])}
    states = payload["states"]
    graph["ObservedEntity"].num_nodes = len(identities)
    graph["ObservedState"].x = torch.tensor(
        [[0. if v is None else v for v in r["values"]] for r in states], dtype=torch.float32).reshape(-1, len(STATE_FEATURES))
    graph["ObservedState"].value_present = torch.tensor(
        [[v is not None for v in r["values"]] for r in states], dtype=torch.bool).reshape(-1, len(STATE_FEATURES))
    graph["ObservedState"].tick = torch.tensor([r["tick"] for r in states], dtype=torch.long)
    graph["ObservedState"].category = torch.tensor([ENTITY_CATEGORIES.index(r["entity_category"]) for r in states], dtype=torch.long)
    owner = [(identities[r["entity_id"]], i) for i, r in enumerate(states)]
    graph["ObservedEntity", "has_observation", "ObservedState"].edge_index = torch.tensor(owner, dtype=torch.long).reshape(-1, 2).T.contiguous()
    previous, successors = {}, []
    for i, row in enumerate(states):
        entity = row["entity_id"]
        if entity in previous:
            earlier = previous[entity]
            if states[earlier]["tick"] < row["tick"]:
                successors.append((earlier, i))
        previous[entity] = i
    graph["ObservedState", "observed_successor", "ObservedState"].edge_index = torch.tensor(successors, dtype=torch.long).reshape(-1, 2).T.contiguous()
    # Readout: each entity's most recent observed state points to the entity.
    latest = {}
    for i, row in enumerate(states):
        if row["entity_id"] not in latest or states[latest[row["entity_id"]]]["tick"] < row["tick"]:
            latest[row["entity_id"]] = i
    readout = sorted((i, identities[entity]) for entity, i in latest.items())
    store = graph["ObservedState", "latest_observation_of", "ObservedEntity"]
    store.edge_index = torch.tensor(readout, dtype=torch.long).reshape(-1, 2).T.contiguous()
    store.observation_age_ticks = torch.tensor([payload["cutoff_tick"] - states[i]["tick"] for i, _ in readout], dtype=torch.long)
    relation_tick = max((r["tick"] for r in states), default=None)
    snapshot = [i for i, r in enumerate(states) if r["tick"] == relation_tick]
    pairs, values = [], []
    for source in snapshot:
        for target in snapshot:
            if source == target:
                continue
            a, b = states[source]["values"], states[target]["values"]
            position = [None if a[k] is None or b[k] is None else b[k] - a[k] for k in range(3)]
            velocity = [None if a[k] is None or b[k] is None else b[k] - a[k] for k in range(3, 6)]
            known = all(v is not None for v in position)
            distances = ([math.sqrt(sum(v * v for v in position)), math.hypot(position[0], position[1]), abs(position[2])]
                         if known else [None, None, None])
            pairs.append((source, target))
            values.append([*position, *distances, *velocity])
    store = graph["ObservedState", "observed_relative", "ObservedState"]
    store.edge_index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
    store.edge_attr = torch.tensor([[0. if v is None else v for v in row] for row in values],
                                   dtype=torch.float32).reshape(-1, len(RELATION_FEATURES))
    store.value_present = torch.tensor([[v is not None for v in row] for row in values],
                                       dtype=torch.bool).reshape(-1, len(RELATION_FEATURES))
    membership = [(identities[member], identities[group]) for group, members in payload["observed_cohorts"].items()
                  for member in members]
    graph["ObservedEntity", "observed_cohort_member", "ObservedEntity"].edge_index = torch.tensor(membership, dtype=torch.long).reshape(-1, 2).T.contiguous()
    logs = payload["logs"]
    graph["ObservedLog"].x = torch.tensor([[0. if row["values"].get(field) is None else row["values"][field]
        for field in LOG_FEATURES] for row in logs], dtype=torch.float32).reshape(-1, len(LOG_FEATURES))
    graph["ObservedLog"].value_present = torch.tensor([[row["values"].get(field) is not None
        for field in LOG_FEATURES] for row in logs], dtype=torch.bool).reshape(-1, len(LOG_FEATURES))
    graph["ObservedLog"].missing_reason = torch.tensor([[0 if field not in row["missing"] else
        1 + LOG_MISSING_REASONS.index(row["missing"][field]) for field in LOG_FEATURES]
        for row in logs], dtype=torch.long).reshape(-1, len(LOG_FEATURES))
    for key, vocabulary in (("channel", LOG_CHANNELS), ("kind", LOG_KINDS), ("receiver_kind", LOG_RECEIVERS)):
        graph["ObservedLog"][key] = torch.tensor([vocabulary.index(row[key]) for row in logs], dtype=torch.long)
    for key in ("sample_tick", "availability_tick"):
        graph["ObservedLog"][key] = torch.tensor([row[key] for row in logs], dtype=torch.long)
    graph["ObservedLog"].persistent = torch.tensor([row["kind"] in LOG_PERSISTENT_KINDS for row in logs], dtype=torch.bool)
    log_edges, log_roles = [], []
    for index, row in enumerate(logs):
        for role, references in row["references"].items():
            for entity in references:
                log_edges.append((identities[entity], index))
                log_roles.append(LOG_REFERENCE_ROLES.index(role))
    edge = graph["ObservedEntity", "mentioned_in_log", "ObservedLog"]
    edge.edge_index = torch.tensor(log_edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
    edge.role = torch.tensor(log_roles, dtype=torch.long)
    # Metadata containers are independent copies of module authority and of the input payload.
    graph.input_metadata = copy.deepcopy({key: payload[key] for key in (
        "version", "cutoff_tick", "history_start_tick", "observation", "observation_policy", "query")})
    graph.input_metadata.update(log_version=payload["log_version"], log_policy=payload["log_policy"],
        log_features=list(LOG_FEATURES), log_feature_units=copy.deepcopy(LOG_UNITS),
        log_enum_values=copy.deepcopy(LOG_ENUM_VALUES),
        log_channels=list(LOG_CHANNELS), log_kinds=list(LOG_KINDS), log_receivers=list(LOG_RECEIVERS),
        log_persistent_kinds=list(LOG_PERSISTENT_KINDS),
        log_reference_roles=list(LOG_REFERENCE_ROLES), log_missing_reasons=["not_missing_or_not_applicable", *LOG_MISSING_REASONS])
    graph.input_metadata["state_features"] = list(STATE_FEATURES)
    graph.input_metadata["relation_features"] = list(RELATION_FEATURES)
    graph.input_metadata["relation_policy"] = RELATION_POLICY
    graph.input_metadata["relation_tick"] = relation_tick
    graph.input_metadata["category_vocabulary"] = list(ENTITY_CATEGORIES)
    query = payload["query"]
    if query is not None:
        graph.query_roles = sorted(query["bindings"])
        graph.query_role_count = torch.tensor([len(graph.query_roles)], dtype=torch.long)
        graph.query_entity_index = torch.tensor([identities[query["bindings"][r]]
                                                 for r in graph.query_roles], dtype=torch.long)
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", type=Path)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--history", type=int, default=100)
    parser.add_argument("--observation", choices=OBSERVATIONS, default=OBSERVATIONS[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = read_prediction_view(args.partition, args.cutoff, history_ticks=args.history, observation=args.observation)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(to_pyg(result), args.output / "prediction_input.pt")
    (args.output / "input.json").write_text(json.dumps(result["input"], ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "source_trace.json").write_text(json.dumps(result["trace"], ensure_ascii=False) + "\n", encoding="utf-8")
