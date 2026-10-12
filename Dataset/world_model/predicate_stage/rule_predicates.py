"""Scoped fixed rules over joint future states; no learned predicate classifier.

Geometry uses the same compiled boxes and pair volumes as the formal producer.
Communication uses its published row thresholds and four-valued semantics.
Unknown operands remain NaN internally and are never counted as false.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults
from Dataset.semantic_truth.episode_sources import source_episode_root
from Dataset.semantic_truth.provenance import open_artifact_text
from Dataset.semantic_simulation.predicate_state_computers import (
    _aircraft_pair_separation_models, _airspace_corridors, _corridor_active_at_tick,
    _index_entities, _resolve_scenario_json,
)

ROOT = Path(__file__).resolve().parents[3]
HORIZONS = (5, 10, 20, 30, 50)
OFFSETS = np.arange(0, 51, 5)
SEPARATION = "safety.aircraft_pair_below_separation"
CAPACITY = "airspace.airspace_corridor_over_capacity"
OCCUPANCY = "airspace.aircraft_occupies_airspace_corridor"
HANDOVER = "communication.handover_in_progress"
LATENCY = "communication.data_flow_latency_above_threshold"
DEGRADED = "communication.link_degraded"
UNAVAILABLE = "communication.link_unavailable"
SCOPED = (SEPARATION, CAPACITY, OCCUPANCY, HANDOVER, LATENCY, DEGRADED, UNAVAILABLE)
EXPORTABLE = (SEPARATION, CAPACITY, OCCUPANCY, HANDOVER, LATENCY)


def rows(path, needles=()):
    """Read logical JSONL (the formal truth file may be zstd despite its suffix)."""
    with open_artifact_text(Path(path)) as stream:
        for line in stream:
            if not needles or any(n in line for n in needles):
                yield json.loads(line)


def binding_key(bindings):
    return tuple(sorted(bindings.items()))


@dataclass
class FutureStates:
    """Typed graph node values: [sample, time, entity, xyz], shared across queries.

    Relations are the query bindings and RuleContext's static corridor/pair
    topology. Communication node values are [sample, time, entity].
    """
    entities: tuple[str, ...]
    ticks: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    communication: dict[str, np.ndarray]


class RuleContext:
    def __init__(self, episode, root=ROOT):
        self.episode = episode
        objective = root / "aw_data/objective_semantic_truth" / episode
        authority = json.loads((objective / "manifest.json").read_text())
        # L6 in particular has a newer executed source than render_ready. The
        # manifest, rather than a directory-name convention, binds formal truth.
        render = Path(authority["episode_root"])
        manifest = json.loads((render / "episode_manifest.json").read_text())
        scene_path = _resolve_scenario_json(manifest, "scene_setup.json")
        script_path = _resolve_scenario_json(manifest, "event_script.json")
        scene = _index_entities(json.loads(scene_path.read_text())["entities"], scene_path)
        roster_path = render / "global_entity_roster.json"
        roster = _index_entities(json.loads(roster_path.read_text())["entities"], roster_path)
        self.margin = float(get_governed_parameter_defaults()["separation_margin_m"])
        self.corridors = _airspace_corridors(scene, self.margin)
        self.pair_models = _aircraft_pair_separation_models(
            json.loads(script_path.read_text()), {**scene, **roster}, self.margin)
        self.uavs = {e for e, r in roster.items() if r.get("entity_category") == "uav"}
        self.positions_static = {e: r["initial_position_enu_m"] for e, r in roster.items()
                                 if "initial_position_enu_m" in r and
                                 (r.get("entity_category") == "facility" or "base_station" in str(r.get("entity_kind", "")))}
        self.stations = {e: p for e, p in self.positions_static.items()
                         if "base_station" in str(roster[e].get("entity_kind", ""))}
        self.comm = defaultdict(dict)
        # Load source states, never predicate labels, for causal lookups below.
        self.communication_source = root / "aw_data/compute_comm_supplement" / episode
        for row in rows(self.communication_source / "communication_state.jsonl"):
            self.comm[row["tick"]][row["entity_id"]] = row
        self.render, self.objective = render, objective
        self.source_descriptor = {"geometry_episode_root": str(render),
                                  "communication_source": str(self.communication_source)}
        templates = json.loads((root / "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json").read_text())
        self.events = defaultdict(list)
        for rule in templates["event_occurrence_types"]:
            if rule.get("trigger_predicate_id") in SCOPED and not rule["support_conditions"]:
                self.events[rule["trigger_predicate_id"]].append(
                    (rule["event_type"], rule["trigger_direction"]))

    def pose_sources(self, max_tick):
        """Authoritative UAV poses, world rows first, scenario-only rows second.

        Same source precedence as load_trajectory_vectors. The caller controls
        the time bound; inference requests cutoff, oracle requests cutoff+50.
        """
        paths = [self.render / "trajectories.jsonl"]
        source = source_episode_root(self.render)
        if source is not None:
            paths.append(source / "trajectories.jsonl")
        out = defaultdict(dict)
        for path in paths:
            for r in rows(path, ('"uav"',)):
                t, e = r["tick"], r["entity_id"]
                if t <= max_tick and e in self.uavs and e not in out[t]:
                    out[t][e] = r
        return out

    def communication_row(self, tick, bindings):
        """Resolve identities at this tick, including flow/handover owner nodes."""
        candidates = self.comm.get(tick, {})
        if "actor" in bindings:
            row = candidates.get(bindings["actor"])
            if row is None or ("station" in bindings and row["station_id"] != bindings["station"]):
                return None
            return row
        for role in ("flow", "handover"):
            if role in bindings:
                matches = [r for r in candidates.values() if r[role + "_id"] == bindings[role]]
                if len(matches) > 1:
                    raise ValueError(f"ambiguous {role} binding: {bindings}")
                return matches[0] if matches else None
        raise ValueError(f"unsupported communication binding: {bindings}")

    def formal_rows(self, ticks):
        wanted = set(ticks)
        # Replay the authoritative persistent graph; the published L1 projection
        # omits occupancy/flow/handover tuples in some episodes.
        with open_artifact_text(self.objective / "world_truth_graph_base.json") as stream:
            base = json.load(stream)
        current = {(r["predicate_id"], r["tuple_id"]): dict(r) for r in base["initial_assertions"]
                   if r["predicate_id"] in (SEPARATION, CAPACITY, OCCUPANCY)}
        if 0 in wanted:
            for r in current.values():
                yield {**r, "tick": 0}
        for delta in rows(self.objective / "world_truth_graph_deltas.jsonl"):
            t = delta["tick"]
            if t > max(wanted):
                break
            for op in delta["operations"]:
                body = op["assertion"] if op["operation"] == "add_predicate_assertion" else op
                if body["predicate_id"] not in (SEPARATION, CAPACITY, OCCUPANCY):
                    continue
                key = body["predicate_id"], body["tuple_id"]
                if op["operation"] == "add_predicate_assertion":
                    current[key] = dict(body)
                elif op["operation"] == "remove_predicate_assertion":
                    del current[key]
                elif op["operation"] == "set_predicate_value":
                    current[key]["value"] = op["to_value"]
                elif op["operation"] != "refresh_predicate_evidence":
                    raise ValueError(f"unsupported graph operation: {op['operation']}")
            if t in wanted:
                for r in current.values():
                    yield {**r, "tick": t}
        from Dataset.semantic_truth.provenance import stable_identifier
        from Dataset.semantic_simulation.compute_comm import _communication_predicate_values
        known = {}
        for r in rows(self.communication_source / "predicate_truth.jsonl", (HANDOVER, LATENCY, DEGRADED, UNAVAILABLE)):
            if r["predicate_id"] in (HANDOVER, LATENCY, DEGRADED, UNAVAILABLE) and r["tick"] in wanted:
                known[(r["predicate_id"], binding_key(r["bindings"]), r["tick"])] = r["value"]
        for t in sorted(wanted):
            for r in self.comm.get(t, {}).values():
                bindings = {HANDOVER: {"handover": r["handover_id"]}, LATENCY: {"flow": r["flow_id"]},
                            DEGRADED: {"actor": r["entity_id"], "station": r["station_id"]},
                            UNAVAILABLE: {"actor": r["entity_id"], "station": r["station_id"]}}
                computed = _communication_predicate_values(r)
                for pid, bound in bindings.items():
                    key = pid, binding_key(bound), t
                    # Task A keeps recorded producer truth; only a missing row is
                    # restored by running the actual producer on its source state.
                    value = known[key] if key in known else computed[pid]
                    yield {"tick": t, "predicate_id": pid, "bindings": bound, "value": value,
                           "tuple_id": stable_identifier("world_predicate_tuple", pid, list(bound.items()))}

    def true_states(self, ticks, poses):
        entities = tuple(sorted(self.uavs))
        p = np.full((1, len(ticks), len(entities), 3), np.nan)
        v = np.full_like(p, np.nan)
        comm = {k: np.full(p.shape[:-1], np.nan) for k in
                ("latency", "handover", "packet_loss", "unavailable", "latency_threshold", "loss_threshold", "requirement")}
        for ti, t in enumerate(ticks):
            for ei, e in enumerate(entities):
                if e in poses.get(t, {}):
                    row = poses[t][e]
                    p[0, ti, ei] = row["pos_enu"]
                    if row.get("vel_mps") is not None:
                        v[0, ti, ei] = row["vel_mps"]
                if e in self.comm.get(t, {}):
                    for k, val in communication_values(self.comm[t][e]).items():
                        comm[k][0, ti, ei] = val
        return FutureStates(entities, np.asarray(ticks), p, v, comm)


def numeric(value):
    return float(value) if type(value) in (int, float) and np.isfinite(value) else np.nan


def communication_values(row):
    quality = row["link_quality"]
    level = quality["quality_level"]
    down = 1. if level == "down" else (0. if level in {"excellent", "good", "fair", "poor"} else np.nan)
    active = row["handover"]["active"]
    requirement = row["requirement"]["status"]
    return {"latency": numeric(quality["latency_ms"]),
            "handover": float(active) if type(active) is bool else np.nan,
            "packet_loss": numeric(quality["packet_loss_ratio"]), "unavailable": down,
            "latency_threshold": numeric(row["quality_thresholds"]["latency_ms"]),
            "loss_threshold": numeric(row["quality_thresholds"]["packet_loss_ratio"]),
            "requirement": {"not_required": 0., "satisfied": 0., "source_required": 1.,
                            "missing_source_record": 2.}.get(requirement, np.nan)}


def compare(value, threshold):
    return np.where(np.isfinite(value) & np.isfinite(threshold), value > threshold, np.nan)


def evaluate(states, context, predicate_id, bindings, *, owner=None, fixed_scope=False):
    """Return [S,T] truth in {0,1,NaN}; inactive/missing scope stays unknown.

    fixed_scope is a declared prediction approximation: retain the cutoff link
    identity rather than looking at future station/flow/handover identities.
    """
    index = {e: i for i, e in enumerate(states.entities)}
    shape = states.position.shape[:2]
    def position(e):
        return states.position[:, :, index[e]] if e in index else np.full((*shape, 3), np.nan)
    if predicate_id == SEPARATION:
        a, b = bindings["first_aircraft"], bindings["second_aircraft"]
        delta = position(a) - position(b)
        model = context.pair_models.get(tuple(sorted((a, b))))
        distance = (np.linalg.norm(delta, axis=-1) if model is None else
                    np.maximum(np.linalg.norm(delta[..., :2], axis=-1) / model["horizontal_limit_m"],
                               np.abs(delta[..., 2]) / model["vertical_limit_m"]) * context.margin)
        return np.where(np.isfinite(distance), distance < context.margin, np.nan)
    if predicate_id in (CAPACITY, OCCUPANCY):
        corridor = context.corridors[bindings["corridor"]]
        members = ([index[e] for e in states.entities if e in context.uavs] if predicate_id == CAPACITY
                   else [index[bindings["actor"]]] if bindings["actor"] in index else [])
        if not members:
            return np.full(shape, np.nan)
        delta = states.position[:, :, members] - np.asarray(corridor["center_enu_m"])
        yaw = np.deg2rad(corridor["yaw_deg"])
        local = np.stack((np.cos(yaw)*delta[..., 0] + np.sin(yaw)*delta[..., 1],
                          -np.sin(yaw)*delta[..., 0] + np.cos(yaw)*delta[..., 1], delta[..., 2]), -1)
        present = np.isfinite(local).all(-1)
        inside = (np.abs(local) <= np.asarray(corridor["extent_m"]) + 1e-9).all(-1) & present
        # Formal occupancy counts the aircraft present in the source trajectory.
        value = inside.sum(-1) > corridor["capacity"] if predicate_id == CAPACITY else inside[..., 0]
        valid = np.ones(shape, bool) if predicate_id == CAPACITY else present[..., 0]
        valid &= np.array([_corridor_active_at_tick(corridor, int(t)) for t in states.ticks])[None]
        return np.where(valid, value, np.nan)
    if predicate_id not in SCOPED:
        raise ValueError(f"predicate outside Task C scope: {predicate_id}")
    if owner is None:
        resolved = [context.communication_row(int(t), bindings) for t in states.ticks]
        owners = {r["entity_id"] for r in resolved if r is not None}
        if not owners:
            return np.full(shape, np.nan)
        if len(owners) != 1:
            raise ValueError(f"communication identity changes owner: {bindings}")
        owner = owners.pop()
    if owner not in index:
        return np.full(shape, np.nan)
    c = {k: value[:, :, index[owner]] for k, value in states.communication.items()}
    latency = compare(c["latency"], c["latency_threshold"])
    latency = np.where(c["requirement"] == 1, np.nan, latency)
    unavailable = np.where(c["requirement"] == 1, np.nan, c["unavailable"])
    if predicate_id == HANDOVER:
        value = c["handover"]
    elif predicate_id == LATENCY:
        value = latency
    elif predicate_id == UNAVAILABLE:
        value = unavailable
    else:
        packet = compare(c["packet_loss"], c["loss_threshold"])
        violation = np.where((latency == 1) | (packet == 1), 1.,
                             np.where((latency == 0) & (packet == 0), 0., np.nan))
        value = np.where(unavailable == 1, 0., np.where(unavailable == 0, violation, np.nan))
        value = np.where(c["requirement"] != 0, np.nan, value)
    if not fixed_scope:
        active = [context.communication_row(int(t), bindings) is not None for t in states.ticks]
        value = np.where(np.asarray(active)[None], value, np.nan)
    return value


def probabilities(truth, ticks, cutoff, initial, event_rules=()):
    """Unconditional first-change mass, initialized by recorded cutoff truth.

    A cutoff rule/record disagreement is not a future transition. Keep the
    recorded state until the rule actually changes, then apply that rule state.
    Future probabilities, transitions and events all use this same sequence.
    """
    if initial not in ("true", "false"):
        raise ValueError("transition probability requires a known cutoff truth")
    if not np.isfinite(truth).all():
        raise ValueError("sampled rule has unknown operands; cannot emit invented probabilities")
    initial_value = initial == "true"
    ticks = np.asarray(ticks)
    if ticks[0] != cutoff or np.any(np.diff(ticks) <= 0):
        raise ValueError("rule sequence must start at cutoff and have increasing ticks")
    rule = truth.astype(bool)
    sequence = np.empty_like(rule)
    sequence[:, 0] = initial_value
    for i in range(1, len(ticks)):
        sequence[:, i] = np.where(rule[:, i] != rule[:, i-1], rule[:, i], sequence[:, i-1])
    future = sequence[:, ticks > cutoff]
    offsets = np.asarray(ticks)[np.asarray(ticks) > cutoff] - cutoff
    changed = future != initial_value
    first = np.where(changed.any(1), offsets[changed.argmax(1)], 51)
    edges = (0, *HORIZONS)
    hazard = [float(((first > lo) & (first <= hi)).mean()) for lo, hi in zip(edges, edges[1:])]
    result = {"p_future_true": {str(h): float(sequence[:, np.flatnonzero(ticks == cutoff+h)[0]].mean())
                               for h in HORIZONS},
              "p_transition": float(sum(hazard)), "transition_hazard": hazard}
    p_event = {}
    for event, direction in event_rules:
        change = ((~sequence[:, :-1] & sequence[:, 1:]) if direction == "rising" else
                  (sequence[:, :-1] & ~sequence[:, 1:]))
        p_event[event] = float(change.any(1).mean())
    if p_event:
        result["p_event"] = p_event
    return result
