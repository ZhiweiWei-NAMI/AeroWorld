"""Small GRU over bound-entity state histories and the queried predicate truth."""
from __future__ import annotations

import bisect
from collections import defaultdict
import json
import math
import re
from types import SimpleNamespace

import torch
from torch import nn

from Dataset.semantic_simulation.compute_comm import _communication_predicate_values
from Dataset.semantic_truth.provenance import open_artifact_text
from Dataset.world_model.data.provenance import content_available_at, resolve_source_split
from Dataset.world_model.model.causal_input import COMM, COMM_KEYS, STATUS, cell
from .qwen_predicate import HORIZONS, PREDICATES, TRUTH, PredicateQueryEncoder, SharedProbabilityHeads

# Explicit, physically interpretable input fields. Missing values have a
# separate categorical status; numeric storage padding is never an observation.
NUMERIC_FIELDS = tuple((name, i) for name in ("pose.position_enu_m", "pose.velocity_enu_mps") for i in range(3)) + (
    ("pose.yaw_deg", None), ("communication.latency_ms", None), ("communication.packet_loss_ratio", None))
CATEGORICAL_FIELDS = (("communication.handover_active", (False, True)),
                      ("communication.quality_level", COMM["communication.quality_level"][3]))
CELL_WIDTH = len(NUMERIC_FIELDS) * (1 + len(STATUS)) + sum(len(v) + len(STATUS) for _, v in CATEGORICAL_FIELDS)


class TemporalEpisodeSource:
    """Index actual baseline inputs once per episode, then slice causal windows.

    The GRU never consumed SUMO/map graph encodings. Avoid constructing those
    unused objects for every cutoff. Only numeric/categorical fields used by the
    GRU, native roster order, and scoped predicate histories are retained here.
    Source indexing can scan an episode; window() exposes only <= cutoff data.
    """
    def __init__(self, episode, spec, root):
        self.episode = episode
        self.split = resolve_source_split(episode)
        if self.split not in ("TRAIN", "VALID"):
            raise ValueError("TEST input is prohibited")
        self.roster, self.states, self.bound_states = {}, {}, {}
        self.timelines = defaultdict(dict)
        self.bindings = {}
        fields = {name for name, _ in NUMERIC_FIELDS + CATEGORICAL_FIELDS}
        frame_or_entity = re.compile(r'"record_kind"\s*:\s*"(?:frame|entity)"')
        with (root / spec["canonical"]).open() as stream:
            for line in stream:
                # The large archive also holds observations, graph projections,
                # and raw capture payloads which are not features of this GRU.
                if not frame_or_entity.search(line):
                    continue
                row = json.loads(line)
                if row["episode_id"] != episode:
                    raise ValueError("cross-episode canonical frame")
                tick = row.get("tick")
                if not content_available_at(row, cutoff=0 if tick is None else tick):
                    continue
                if row["record_kind"] == "entity":
                    identity = row["fields"]["roster.entity_id"]
                    self.roster[identity] = len(self.roster)
                elif tick % 5 == 0:
                    self.states[tick, row["entity_id"]] = {k: v for k, v in row["fields"].items() if k in fields}
        if not self.roster or not self.states:
            raise ValueError("no authentic roster/frames in canonical source")

        def record(pid, bound, tick, value):
            key = pid, tuple(sorted(bound.items()))
            self.timelines[key][tick] = value
            self.bindings[key] = bound

        base_path = root / spec["graph_base"]
        base = json.loads(base_path.read_text())
        if base["episode_id"] != episode or base["tick"] != 0:
            raise ValueError("invalid graph base")
        for body in base["initial_assertions"]:
            pid = body["predicate_id"]
            if pid in PREDICATES and PREDICATES[pid] != "communication":
                record(pid, body["bindings"], 0, body["value"])
        with open_artifact_text(base_path.with_name("world_truth_graph_deltas.jsonl")) as stream:
            for line in stream:
                delta = json.loads(line)
                for op in delta["operations"]:
                    kind = op["operation"]
                    body = op["assertion"] if kind == "add_predicate_assertion" else op
                    pid = body["predicate_id"]
                    if pid not in PREDICATES or PREDICATES[pid] == "communication" or kind == "refresh_predicate_evidence":
                        continue
                    value = ("unknown" if kind == "remove_predicate_assertion" else
                             body["to_value"] if kind == "set_predicate_value" else body["value"])
                    record(pid, body["bindings"], delta["tick"], value)

        comm_path = root / spec["communication"]
        comm_pids = {pid for pid, family in PREDICATES.items() if family == "communication"}
        # Recorded-run predicates have precedence over today's producer version.
        with comm_path.with_name("predicate_truth.jsonl").open() as stream:
            for line in stream:
                if not any(pid in line for pid in comm_pids):
                    continue
                row = json.loads(line)
                if row["predicate_id"] in comm_pids:
                    record(row["predicate_id"], row["bindings"], row["tick"], row["value"])
        with comm_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                tick = row["tick"]
                if row["episode_id"] != episode or row["scope_active"] is not True:
                    raise ValueError("invalid communication sample")
                values = {name: ({"kind": "source_unknown"} if row[parent][leaf] == "unknown" else row[parent][leaf])
                          for name, (parent, leaf) in COMM_KEYS.items()}
                self.states[tick, row["entity_id"]].update(values)
                bound = {"communication.handover_in_progress": {"handover": row["handover_id"]},
                         "communication.data_flow_latency_above_threshold": {"flow": row["flow_id"]}}
                computed = _communication_predicate_values(row)
                for pid, bindings in bound.items():
                    key = pid, tuple(sorted(bindings.items()))
                    if tick not in self.timelines[key]:
                        record(pid, bindings, tick, computed[pid])
                for identity in (row["flow_id"], row["handover_id"]):
                    self.bound_states[tick, identity] = values
        # Communication samples cannot persist across a missing sampled tick.
        for key, samples in self.timelines.items():
            if PREDICATES[key[0]] == "communication":
                for tick in list(samples):
                    if tick + 5 not in samples:
                        samples[tick + 5] = "unknown"
        self.timelines = {key: sorted(samples.items()) for key, samples in self.timelines.items()}
        self.times = {key: [tick for tick, _ in values] for key, values in self.timelines.items()}

    def window(self, cutoff, rows):
        left = cutoff - 50
        keys = {(row["predicate_id"], tuple(sorted(row["bindings"].items()))) for row in rows}
        identities = {identity for key in keys for _, identity in key[1]}
        # Ordinals follow the native roster, then first causal tuple appearance;
        # they never depend on which queries share this window or their order.
        local = dict(self.roster)
        for key in sorted(self.timelines, key=lambda k: (self.times[k][0], k)):
            if self.times[key][0] <= cutoff:
                for _, identity in key[1]:
                    if identity not in local:
                        local[identity] = len(local)
        semantic = []
        for pid, bound in sorted(keys):
            key = pid, bound
            start = max(0, bisect.bisect_right(self.times[key], left) - 1)
            stop = bisect.bisect_right(self.times[key], cutoff)
            for tick, value in self.timelines[key][start:stop]:
                semantic.append({"tick": max(left, tick), "relation": {
                    "predicate_id": pid, "tuple_id": repr(bound), "bindings": dict(bound), "value": value}})
        ages = {}
        for key in keys:
            samples = self.timelines[key][:bisect.bisect_right(self.times[key], cutoff)]
            current = samples[-1][1]
            start = len(samples)-1
            while start > 0 and samples[start-1][1] == current:
                start -= 1
            ages[key] = ((cutoff-samples[start][0])/10., start == 0)
        return SimpleNamespace(episode=self.episode, cutoff=cutoff, window_start=left, split=self.split,
            tuple_state_ages=ages,
            states={(tick, identity): self.states[tick, identity] for tick in range(left, cutoff+1, 5)
                    for identity in identities if (tick, identity) in self.states},
            bound_communication_states={(tick, identity): self.bound_states[tick, identity]
                    for tick in range(left, cutoff+1, 5) for identity in identities if (tick, identity) in self.bound_states},
            semantic_records=semantic, local_identities=local, scales={}, fields=(), temporal_context=None)


def relation_timelines(index):
    """Read already-replayed load_index records; never read raw future deltas."""
    result = {}
    for record in sorted(index.semantic_records, key=lambda r: r["tick"]):
        if "relation" not in record:
            continue
        body = record["relation"]
        if body["predicate_id"] not in PREDICATES:
            continue
        key = body["predicate_id"], body["tuple_id"]
        operation = record.get("operation")
        if operation == "remove_predicate_assertion":
            truth = "unknown"
        elif operation == "set_predicate_value":
            truth = body["to_value"]
        else:
            truth = body["value"]
        if truth == "out_of_scope":
            truth = "unknown"
        if truth not in TRUTH:
            raise ValueError(f"unexpected formal truth {truth}")
        result.setdefault(key, []).append((record["tick"], truth, body["bindings"]))
    return result


def truth_at(timeline, tick):
    position = bisect.bisect_right([x[0] for x in timeline], tick) - 1
    return "unknown" if position < 0 else timeline[position][1]


def tuple_history_features(index, rows):
    """Direct formal truth history; age is measured from the full causal timeline.

    The age flag marks a spell already active at its first recorded observation.
    Neither cutoff truth nor age is taken from supervision labels.
    """
    timelines = relation_timelines(index)
    bound = {(pid, tuple(sorted(t[-1][2].items()))): t for (pid, _), t in timelines.items()}
    values = []
    for row in rows:
        key = row["predicate_id"], tuple(sorted(row["bindings"].items()))
        timeline = bound[key]
        sequence = [truth_at(timeline, tick) for tick in range(index.cutoff-50, index.cutoff+1, 5)]
        age, censored = index.tuple_state_ages[key]
        values.append([*(float(truth == value) for truth in sequence for value in TRUTH),
                       math.log1p(age), float(censored)])
    return torch.tensor(values, dtype=torch.float32)


def _state_vector(state, scales):
    values = []
    for name, component in NUMERIC_FIELDS:
        if state is None:
            kind, value = "absent", None
        elif name not in state:
            kind, value = "missing", None
        else:
            kind, value = cell(state[name], component)
        status = [float(kind == s) for s in STATUS]
        if kind == "present":
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid present state {name}: {value}")
            # Physical signed-log transform requires no heldout normalization.
            value = math.copysign(math.log1p(abs(value)), value)
        else:
            value = 0.  # Masked storage, with the actual non-present kind above.
        values.extend([value, *status])
    for name, categories in CATEGORICAL_FIELDS:
        if state is None:
            kind, value = "absent", None
        elif name not in state:
            kind, value = "missing", None
        else:
            kind, value = cell(state[name], None)
        if kind == "present" and value not in categories:
            raise ValueError(f"invalid category {name}: {value}")
        values.extend(float(kind == "present" and value == c) for c in categories)
        values.extend(float(kind == s) for s in STATUS)
    return values


def history_features(index, rows, query_encoder):
    """Eleven authentic samples in [cutoff-50, cutoff], with role order retained."""
    timelines = relation_timelines(index)
    # Native graph and communication exports can serialize tuple roles in
    # different orders, producing different opaque tuple IDs. Join the actual
    # predicate and role-bound identities; tuple IDs only identify output rows.
    bound_timelines = {}
    for (predicate, _), timeline in timelines.items():
        key = predicate, tuple(sorted(timeline[-1][2].items()))
        if key in bound_timelines:
            raise ValueError(f"duplicate formal timeline for bound predicate {key}")
        bound_timelines[key] = timeline
    max_roles = max(map(len, query_encoder.roles.values()))
    features = []
    for row in rows:
        key = row["predicate_id"], tuple(sorted(row["bindings"].items()))
        timeline = bound_timelines[key]
        current = truth_at(timeline, index.cutoff)
        if current != row["state_at_cutoff"]:
            raise ValueError(f"cutoff truth disagrees with formal source for {key}: {current} != {row['state_at_cutoff']}")
        ordered = query_encoder.ordered_bindings(row)
        sequence = []
        for tick in range(index.cutoff - 50, index.cutoff + 1, 5):
            vector = []
            for _, identity in ordered:
                state = index.states.get((tick, identity))
                if state is None and hasattr(index, "bound_communication_states"):
                    state = index.bound_communication_states.get((tick, identity))
                vector.extend(_state_vector(state, index.scales))
                vector.append(1.)  # Declared role is bound, even without a state frame.
            for _ in range(max_roles - len(ordered)):
                vector.extend([0.] * (CELL_WIDTH + 1))  # Unbound role padding.
            truth = truth_at(timeline, tick)
            vector.extend(float(truth == s) for s in TRUTH)
            vector.append((tick - index.cutoff) / 50)
            sequence.append(vector)
        features.append(sequence)
    return torch.tensor(features, dtype=torch.float32)


class TemporalBaseline(nn.Module):
    def __init__(self, width=128, event_types=()):
        super().__init__()
        self.query = PredicateQueryEncoder(width)
        max_roles = max(map(len, self.query.roles.values()))
        self.input_width = max_roles * (CELL_WIDTH + 1) + len(TRUTH) + 1
        self.gru = nn.GRU(self.input_width, width, batch_first=True)
        self.combine = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.heads = SharedProbabilityHeads(width, event_types)

    def forward(self, index, rows, features=None):
        features = history_features(index, rows, self.query) if features is None else features
        features = features.to(self.gru.weight_ih_l0.device)
        _, hidden = self.gru(features)
        query, _ = self.query(rows, index)
        output = self.heads(self.combine(torch.cat((hidden[-1], query), -1)))
        output["event_types"] = self.heads.event_types
        return output


class RuleTemporalBaseline(TemporalBaseline):
    """GRU with the same direct tuple context and mean sampled-rule operands as Qwen."""
    def __init__(self, rule_width, width=128, *, per_sample=False):
        super().__init__(width)
        self.per_sample = per_sample
        self.rule_context = nn.Sequential(nn.Linear(rule_width+35, width), nn.GELU(), nn.LayerNorm(width))
        self.combine = nn.Sequential(nn.Linear(width*3, width), nn.GELU(), nn.LayerNorm(width))
        nn.init.zeros_(self.heads.state.weight)
        nn.init.zeros_(self.heads.state.bias)
        nn.init.zeros_(self.heads.transition.weight)
        nn.init.zeros_(self.heads.transition.bias)

    def forward(self, index, rows, features):
        device = self.gru.weight_ih_l0.device
        _, hidden = self.gru(features["history"].to(device))
        query, _ = self.query(rows, index)
        if self.per_sample:
            # Same conditional correction/mean-probability operation as Qwen;
            # only the temporal representation and correction network differ.
            k = len(features["rule"])
            direct = features["tuple"].to(device)[None].expand(k, -1, -1)
            context = self.rule_context(torch.cat((direct, features["rule"].to(device)), -1))
            output = self.heads(self.combine(torch.cat((hidden[-1][None].expand(k, -1, -1),
                query[None].expand(k, -1, -1), context), -1)))
            output["state_logits"] = output["state_logits"] + torch.logit(features["state_prior"].to(device))
            output["transition_logits"] = output["transition_logits"] + features["mass_prior"].to(device).log()
            from .two_stage import aggregate_outputs
            return aggregate_outputs([{key: value[i] if isinstance(value, torch.Tensor) else value
                for key, value in {**output, "event_types": ()}.items()} for i in range(k)])
        context = self.rule_context(torch.cat((features["tuple"], features["rule"].mean(0)), -1).to(device))
        output = self.heads(self.combine(torch.cat((hidden[-1], query, context), -1)))
        output["state_logits"] = output["state_logits"] + torch.logit(features["state_prior"].mean(0).to(device))
        output["transition_logits"] = output["transition_logits"] + features["mass_prior"].mean(0).to(device).log()
        output["event_types"] = ()
        return output
