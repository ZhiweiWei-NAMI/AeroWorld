"""One parent episode's world streams, read from the producer sources (RESEARCH_PLAN §2).

Sources (all 210 parents carry them):
  render_ready_episodes_capture_filtered/<ep>/truth_frames.jsonl   entity motion and state, 10 Hz
  render_ready_episodes_capture_filtered/<ep>/event_trace.jsonl    scheduled activations
  objective_semantic_truth/<ep>/world_truth_graph_{base,deltas}    every predicate tuple, value and evidence, 0.5 s
  objective_semantic_truth/<ep>/{communication,compute}_state      link and compute processes, 0.5 s
  objective_semantic_truth/<ep>/event_occurrences, compute_events  objective and producer events

Quantized values are integers (value = k * eps).  Label validity: a communication
or compute predicate label is masked when its recorded value disagrees with the
producer rule applied to the recorded state at that tick (a producer version
mismatch); unknown and out_of_scope values are masked individually.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import orjson

from Dataset.semantic_truth.provenance import open_artifact_text
from Dataset.world_model.unified import schema as S


def rows(path: Path):
    with open_artifact_text(path) as handle:
        for line in handle:
            if line.strip():
                yield orjson.loads(line)


def get(row, path):
    for part in path.split("."):
        if not isinstance(row, dict) or part not in row:
            return None
        row = row[part]
    return row


def quantize(value, eps):
    """Integer quantum count, or None for non-numeric / non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return int(round(value / eps))


def categorical(value):
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass
class Stream:
    """A sampled field: tick -> value (int quanta for numeric fields, str for categorical)."""
    numeric: bool
    eps: float | None
    samples: dict = field(default_factory=dict)


@dataclass
class Relation:
    tuple_id: str
    predicate: str
    bindings: dict
    samples: dict = field(default_factory=dict)       # formal tick -> value str, or None when absent
    evidence: dict = field(default_factory=dict)      # path -> Stream (formal ticks while present)


@dataclass
class Event:
    tick: int
    type: str
    bindings: dict
    entities: frozenset
    source: str


@dataclass
class Scheduled:
    tick: int
    category: str
    chain: str
    title: str


@dataclass
class EpisodeSource:
    episode: str
    category: dict = field(default_factory=dict)        # entity -> category
    kind: dict = field(default_factory=dict)            # entity -> entity_type
    present: dict = field(default_factory=dict)         # entity -> sorted list of ticks
    position: dict = field(default_factory=dict)        # entity -> {tick: (x, y, z) cm}
    position_raw: dict = field(default_factory=dict)    # entity -> {tick: (x, y, z) m at storage precision} (labels only)
    target_raw: dict = field(default_factory=dict)      # (entity, source, field) -> {tick: float} (labels only)
    velocity: dict = field(default_factory=dict)        # entity -> {tick: (vx, vy, vz) cm/s}
    fields: dict = field(default_factory=dict)          # (entity, source, field) -> Stream; source in frame/comm/compute
    objects: dict = field(default_factory=dict)         # object id -> (kind, owner entity)
    relations: dict = field(default_factory=dict)       # tuple id -> Relation
    events: list = field(default_factory=list)
    scheduled: list = field(default_factory=list)
    field_index: dict = field(default_factory=dict)     # entity -> set of (entity, source, field) keys
    masked_labels: set = field(default_factory=set)          # (tuple id, tick) whose value disagrees with the producer rule
    audit: dict = field(default_factory=dict)

    def owner(self, ref):
        """Entity an identifier refers to: itself, an object's owner, or None."""
        if ref in self.category:
            return ref
        if ref in self.objects:
            return self.objects[ref][1]
        return None


def _leaves(row, prefix=""):
    for key, value in row.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _leaves(value, path+".")
        else:
            yield path, value


def _frame_fields(entity):
    out = {}
    for name in S.FRAME_FIELDS:
        value = get(entity, name)
        if value is not None:
            out[name] = categorical(value)
    for prefix in S.FRAME_PREFIXES:
        block = get(entity, prefix[:-1])
        if isinstance(block, dict):
            for path, value in _leaves(block, prefix):
                if isinstance(value, (bool, str)):
                    out[path] = categorical(value)
    return out


def load_frames(src: EpisodeSource, path: Path):
    for frame in rows(path):
        tick = frame["tick"]
        for e in frame["entities"]:
            eid = e["entity_id"]
            src.category.setdefault(eid, e["entity_category"])
            src.kind.setdefault(eid, e.get("entity_type") or e.get("entity_kind") or e["entity_category"])
            src.present.setdefault(eid, []).append(tick)
            pose = e.get("truth_pose") or {}
            p, v = pose.get("position_enu_m"), pose.get("velocity_enu_mps")
            if p is not None:
                src.position.setdefault(eid, {})[tick] = tuple(int(round(c/S.POS_EPS)) for c in p)
                src.position_raw.setdefault(eid, {})[tick] = tuple(float(c) for c in p)
            if v is not None:
                src.velocity.setdefault(eid, {})[tick] = tuple(int(round(c/S.VEL_EPS)) for c in v)
            for name, eps in S.FRAME_NUMERIC.items():
                q = quantize(get(e, name), eps)
                if q is not None:
                    src.fields.setdefault((eid, "frame", name), Stream(True, eps)).samples[tick] = q
            for name, value in _frame_fields(e).items():
                if value is not None:
                    src.fields.setdefault((eid, "frame", name), Stream(False, None)).samples[tick] = value


def _process_rows(src, path, numeric, cats, objects, source, key):
    for row in rows(path):
        tick, eid = row["tick"], row[key]
        for name, kind in objects.items():
            if row.get(name):
                src.objects.setdefault(row[name], (kind, eid))
        for name, eps in numeric.items():
            raw = get(row, name)
            q = quantize(raw, eps)
            if q is not None and any(name == f for _, f, _ in S.NUMERIC_TARGETS.values()):
                src.target_raw.setdefault((eid, source, name), {})[tick] = float(raw)
            stream = src.fields.setdefault((eid, source, name), Stream(True, eps))
            stream.samples[tick] = q if q is not None else ("unknown" if raw is not None else None)
        for name in cats:
            value = categorical(get(row, name))
            if value is not None:
                src.fields.setdefault((eid, source, name), Stream(False, None)).samples[tick] = value
        yield row


def load_world_graph(src: EpisodeSource, directory: Path):
    base = json.loads((directory/"world_truth_graph_base.json").read_text())
    if base["tick"] != 0 or base["governed_parameters"]["formal_step_ticks"] != S.FORMAL_STEP:
        raise ValueError(f"{src.episode}: unexpected world graph cadence")

    def evidence(rel, observations, tick):
        """An operation's observation list is the tuple's complete evidence snapshot at that tick:
        addresses it omits are cleared from that tick on."""
        listed = {o.get("path") for o in observations or ()}
        for path, stream in rel.evidence.items():
            if path not in listed:
                stream.samples[tick] = None
        for obs in observations or ():
            path, value = obs.get("path"), obs.get("value")
            if path is None or path in S.EVIDENCE_SKIP:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                eps = S.evidence_eps(path)
                stream = rel.evidence.setdefault(path, Stream(True, eps))
                stream.samples[tick] = quantize(value, eps)
            else:
                value = categorical(value)
                if value is not None:
                    rel.evidence.setdefault(path, Stream(False, None)).samples[tick] = value

    def assert_(body, tick):
        rel = src.relations.get(body["tuple_id"])
        if rel is None:
            rel = src.relations[body["tuple_id"]] = Relation(body["tuple_id"], body["predicate_id"], body["bindings"])
        rel.samples[tick] = body["value"]
        evidence(rel, body.get("observations"), tick)

    for body in base["initial_assertions"]:
        assert_(body, 0)
    for delta in rows(directory/"world_truth_graph_deltas.jsonl"):
        tick = delta["tick"]
        for op in delta["operations"]:
            kind = op["operation"]
            if kind == "add_predicate_assertion":
                assert_(op["assertion"], tick)
            elif kind == "remove_predicate_assertion":
                src.relations[op["tuple_id"]].samples[tick] = None
            elif kind == "set_predicate_value":
                rel = src.relations[op["tuple_id"]]
                rel.samples[tick] = op["to_value"]
                if "observations" in op:
                    evidence(rel, op["observations"], tick)
            elif kind == "refresh_predicate_evidence":
                evidence(src.relations[op["tuple_id"]], op.get("observations") or [], tick)
            else:
                raise ValueError(f"{src.episode}: unsupported graph operation {kind}")
    # Dense formal timelines: a value holds until the next change; absence is None.
    for rel in src.relations.values():
        dense, state = {}, None
        changes = rel.samples
        for tick in range(0, S.LAST_TICK+1, S.FORMAL_STEP):
            if tick in changes:
                state = changes[tick]
            if state is not None:
                dense[tick] = state
        rel.samples = dense
        for stream in rel.evidence.values():
            held, value = {}, None
            for tick in range(0, S.LAST_TICK+1, S.FORMAL_STEP):
                if tick in stream.samples:
                    value = stream.samples[tick]          # None: cleared by a snapshot without this address
                if tick not in dense:
                    value = None
                elif value is not None:
                    held[tick] = value
            stream.samples = held


def load_events(src: EpisodeSource, directory: Path, render: Path):
    def resolve(bindings):
        return frozenset(o for o in (src.owner(v) for v in bindings.values() if isinstance(v, str)) if o)
    for row in rows(directory/"event_occurrences.jsonl"):
        src.events.append(Event(row["detection_tick"], row["event_type_id"], row["bindings"], resolve(row["bindings"]), "objective"))
    for row in rows(directory/"compute_events.jsonl"):
        tick = row.get("trigger_tick", row["transition"]["to_tick"])
        src.events.append(Event(tick, row["event_type_id"], row["bindings"], resolve(row["bindings"]), "compute_comm"))
    src.events.sort(key=lambda e: (e.tick, e.type, json.dumps(e.bindings, sort_keys=True)))
    for row in rows(render/"event_trace.jsonl"):
        payload = row.get("payload") or {}
        src.scheduled.append(Scheduled(row["activated_tick"], payload.get("category") or "unspecified",
                                       row.get("chain_id", ""), payload.get("title", "")))
    src.scheduled.sort(key=lambda s: (s.tick, s.chain))


def check_validity(src: EpisodeSource, comm_rows, compute_rows):
    """Mask a communication/compute predicate family whose recorded values disagree with the producer rule."""
    from Dataset.semantic_simulation import compute_comm as producer
    index = {(r.predicate, json.dumps(r.bindings, sort_keys=True)): r for r in src.relations.values()}
    bind_comm = {
        "communication.channel_bandwidth_over_allocated": lambda r: {"allocation": r["channel_allocation_id"]},
        "communication.communication_session_interrupted": lambda r: {"session": r["session_id"]},
        "communication.data_flow_latency_above_threshold": lambda r: {"flow": r["flow_id"]},
        "communication.handover_in_progress": lambda r: {"handover": r["handover_id"]},
        "communication.link_degraded": lambda r: {"actor": r["entity_id"], "station": r["station_id"]},
        "communication.link_unavailable": lambda r: {"actor": r["entity_id"], "station": r["station_id"]},
        "communication.message_transmission_failed": lambda r: {"attempt": r["transmission_attempt_id"]},
        "communication.packet_loss_above_threshold": lambda r: {"flow": r["flow_id"]},
        "communication.retransmission_in_progress": lambda r: {"retransmission": r["retransmission_id"]}}
    bind_compute = {
        "compute.compute_node_unavailable": lambda r: {"node": r["node_id"]},
        "compute.compute_resource_over_allocated": lambda r: {"node": r["node_id"]},
        "compute.compute_task_deadline_missed": lambda r: {"task": r["task_id"]},
        "compute.compute_task_execution_failed": lambda r: {"task": r["task_id"]},
        "compute.compute_task_migrating": lambda r: {"task": r["task_id"]},
        "compute.compute_task_offloaded": lambda r: {"task": r["task_id"]},
        "compute.compute_task_queued": lambda r: {"queue": r["queue_id"], "task": r["task_id"]}}
    checked, mismatched = defaultdict(int), defaultdict(int)
    for state_rows, values, binders in ((comm_rows, producer._communication_predicate_values, bind_comm),
                                        (compute_rows, producer._compute_predicate_values, bind_compute)):
        for row in state_rows:
            derived = values(row)
            for pid, bind in binders.items():
                rel = index.get((pid, json.dumps(bind(row), sort_keys=True)))
                recorded = rel.samples.get(row["tick"]) if rel else None
                if recorded is None:
                    continue
                checked[pid] += 1
                if recorded != derived[pid]:
                    mismatched[pid] += 1
                    src.masked_labels.add((rel.tuple_id, row["tick"]))
    src.audit["producer_check"] = {pid: [mismatched[pid], checked[pid]] for pid in checked}


def load_episode(episode: str) -> EpisodeSource:
    src = EpisodeSource(episode)
    render = S.AW/"render_ready_episodes_capture_filtered"/episode
    objective = S.AW/"objective_semantic_truth"/episode
    load_frames(src, render/"truth_frames.jsonl")
    comm_rows = list(_process_rows(src, objective/"communication_state.jsonl", S.COMM_NUMERIC, S.COMM_CATEGORICAL,
                                   S.COMM_OBJECTS, "comm", "entity_id"))
    compute_rows = list(_process_rows(src, objective/"compute_state.jsonl", S.COMPUTE_NUMERIC, S.COMPUTE_CATEGORICAL,
                                      S.COMPUTE_OBJECTS, "compute", "entity_id"))
    for row in comm_rows:   # stations bind to their facility entity when one exists
        for ref in (row.get("station_id"), get(row, "handover.current_station_id")):
            if isinstance(ref, str) and ref not in src.objects:
                facility = ref.split(":", 1)[-1]
                src.objects[ref] = ("station", facility if facility in src.category else None)
    for key in src.fields:
        src.field_index.setdefault(key[0], set()).add(key)
    load_world_graph(src, objective)
    load_events(src, objective, render)
    check_validity(src, comm_rows, compute_rows)
    src.audit.update({"entities": len(src.category), "relations": len(src.relations), "events": len(src.events),
                      "scheduled": len(src.scheduled), "masked_labels": len(src.masked_labels)})
    return src
