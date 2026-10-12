"""Compact per-episode arrays, causal per-cell features and exact anchor labels.

Input for the small numerical state predictor (compact_model.py).  One episode
becomes dense ``[tick, entity]`` arrays read once from the canonical records;
features of a cell at cutoff ``c`` only read ticks ``<= c`` (targets ``c+h``
are labels only).  Anchors and atom labels follow causal_input.anchor_terms /
anchor_atom exactly: the latest present value at or before ``c``, pose
position advanced by the latest velocity; atom = target within the source
rounding bound, decided in integer micro-units.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from Dataset.world_model.data.provenance import resolve_source_split
from Dataset.world_model.model.sumo_temporal import _load_static

CATEGORIES = ("vehicle", "pedestrian", "uav", "facility", "airspace_corridor", "prop",
              "crowd_anchor", "facade_anchor", "vehicle_anchor", "other")
SIGNAL_STATES = ("none", "red", "yellow", "green", "other")
HORIZONS = (5, 10)
# Velocity-change lags (ticks) and past constant-velocity residual lags.
VELOCITY_LAGS = (1, 2, 5, 10, 20, 50)
RESIDUAL_LAGS = (5, 10, 20)
NEIGHBOURS = 4
MICRO = 10**6


def _present(value):
    if isinstance(value, dict):
        if value.get("kind", "present") != "present":
            return None
        value = value.get("value")
    return value


def _micro(values, context):
    out = []
    for v in values:
        scaled = float(v)*MICRO
        units = round(scaled)
        if abs(scaled-units) > 1e-3:
            raise ValueError(f"{context}: {v!r} is off the 1e-6 source grid")
        out.append(units)
    return out


@dataclass
class EpisodeArrays:
    episode: str
    split: str
    tick_seconds: float
    entities: list
    category: np.ndarray          # [E] index into CATEGORIES
    present: np.ndarray           # [T, E] bool
    position: np.ndarray          # [T, E, 3] int64 micro-metres
    velocity: np.ndarray          # [T, E, 3] int64 micro-m/s
    yaw: np.ndarray               # [T, E] float64 degrees (nan if absent)
    activity: np.ndarray          # [T, E] int (vocabulary index, -1 absent)
    activity_vocab: list
    # SUMO vehicle context (nan / -1 where not a SUMO vehicle sample)
    lane: np.ndarray              # [T, E] int lane index (-1)
    lane_vocab: list
    lane_position: np.ndarray     # [T, E] front-bumper lane position (m)
    lane_length: np.ndarray       # [T, E] (m)
    lane_speed: np.ndarray        # [T, E] lane speed limit (m/s)
    allowed_speed: np.ndarray     # [T, E]
    lane_internal: np.ndarray     # [T, E] 1 inside a junction
    signal: np.ndarray            # [T, E] int SIGNAL_STATES
    crossed_stop_line: np.ndarray  # [T, E] float
    length: np.ndarray            # [E] vehicle length (m), nan otherwise


def extract_episode(canonical: Path, episode: str) -> EpisodeArrays:
    frames, roster, tick_seconds = [], {}, set()
    with open(canonical) as stream:
        for line in stream:
            record = json.loads(line)
            kind = record["record_kind"]
            if kind == "entity":
                roster[record["fields"]["roster.entity_id"]] = record["fields"].get("roster.entity_category", "other")
            elif kind == "frame":
                fields = record["fields"]
                if "archive.scene.dt_s" in fields:
                    tick_seconds.add(float(fields["archive.scene.dt_s"]))
                if record.get("entity_id") is None:
                    continue
                truth = fields.get("archive.truth_frames")
                sumo = truth.get("sumo_vehicle") if isinstance(truth, dict) else None
                frames.append((record["tick"], record["entity_id"], _present(fields.get("pose.position_enu_m")),
                               _present(fields.get("pose.velocity_enu_mps")), _present(fields.get("pose.yaw_deg")),
                               fields.get("annotations.activity_type"), sumo))
    if len(tick_seconds) != 1:
        raise ValueError(f"{episode}: expected one tick duration")
    entities = sorted({f[1] for f in frames})
    column = {e: i for i, e in enumerate(entities)}
    T, E = max(f[0] for f in frames)+1, len(entities)
    present = np.zeros((T, E), bool)
    position = np.zeros((T, E, 3), np.int64)
    velocity = np.zeros((T, E, 3), np.int64)
    yaw = np.full((T, E), np.nan)
    activity = np.full((T, E), -1, np.int64)
    lane = np.full((T, E), -1, np.int64)
    floats = {name: np.full((T, E), np.nan) for name in ("lane_position", "lane_length", "lane_speed", "allowed_speed",
                                                          "lane_internal", "crossed_stop_line")}
    signal = np.zeros((T, E), np.int64)
    length = np.full(E, np.nan)
    activity_vocab, lane_vocab, maps = {}, {}, {}
    for tick, entity, p, v, y, act, sumo in frames:
        e = column[entity]
        if isinstance(p, list) and isinstance(v, list) and len(p) == 3 and len(v) == 3:
            present[tick, e] = True
            position[tick, e] = _micro(p, f"{episode}:{entity}@{tick} position")
            velocity[tick, e] = _micro(v, f"{episode}:{entity}@{tick} velocity")
        if y is not None:
            yaw[tick, e] = float(y)
        if isinstance(act, str):
            activity[tick, e] = activity_vocab.setdefault(act, len(activity_vocab))
        if isinstance(sumo, dict) and sumo.get("sumo_lane_id"):
            lane_id = sumo["sumo_lane_id"]
            ref = sumo.get("lane_instance_source_ref", "").split("#", 1)[0]
            if ref not in maps:
                maps[ref] = _load_static(Path(ref))[0]
            definition = maps[ref].get(lane_id, {}).get("fields", {})
            lane[tick, e] = lane_vocab.setdefault(lane_id, len(lane_vocab))
            floats["lane_position"][tick, e] = float(sumo.get("lane_position_m", np.nan))
            floats["lane_length"][tick, e] = float(definition.get("sumo.length_m", np.nan)) if isinstance(definition.get("sumo.length_m"), (int, float)) else np.nan
            floats["lane_speed"][tick, e] = float(definition.get("sumo.speed_mps", np.nan)) if isinstance(definition.get("sumo.speed_mps"), (int, float)) else np.nan
            floats["allowed_speed"][tick, e] = float(sumo.get("allowed_speed_mps", np.nan))
            floats["lane_internal"][tick, e] = float(definition.get("sumo.edge_function") == "internal")
            floats["crossed_stop_line"][tick, e] = float(bool(sumo.get("crossed_stop_line")))
            state = sumo.get("controlling_signal_state") or "none"
            signal[tick, e] = SIGNAL_STATES.index(state) if state in SIGNAL_STATES else SIGNAL_STATES.index("other")
            dims = sumo.get("dimensions_m") or {}
            if isinstance(dims.get("length"), (int, float)):
                length[e] = float(dims["length"])
    category = np.array([CATEGORIES.index(roster.get(e, "other")) if roster.get(e, "other") in CATEGORIES
                         else CATEGORIES.index("other") for e in entities])
    return EpisodeArrays(episode, resolve_source_split(episode), tick_seconds.pop(), entities, category, present,
                         position, velocity, yaw, activity, sorted(activity_vocab, key=activity_vocab.get),
                         lane, sorted(lane_vocab, key=lane_vocab.get), floats["lane_position"], floats["lane_length"],
                         floats["lane_speed"], floats["allowed_speed"], floats["lane_internal"], signal,
                         floats["crossed_stop_line"], length)


def save_episode(arrays: EpisodeArrays, path: Path) -> None:
    payload = {k: v for k, v in arrays.__dict__.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in arrays.__dict__.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **payload)


def load_episode(path: Path) -> EpisodeArrays:
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    return EpisodeArrays(**meta, **{k: data[k] for k in data.files if k != "meta"})


# ---------------------------------------------------------------- labels

def anchor_labels(arrays: EpisodeArrays, cutoff: int, entities: np.ndarray, horizon: int):
    """Anchor, truth, atom label and atom band for pose position/velocity components.

    Returns dict of ``[N, 6]`` arrays (components: position xyz, velocity xyz)
    and ``valid [N]`` (latest present within the 5 s history and present target).
    """
    T = arrays.present.shape[0]
    target = cutoff+horizon
    history = arrays.present[max(0, cutoff-50):cutoff+1, entities]
    has = history.any(0)
    latest = cutoff-np.argmax(history[::-1], axis=0)
    valid = has & (target < T)
    if target < T:
        valid &= arrays.present[target, entities]
    latest = np.where(has, latest, cutoff)
    k = target-latest                     # elapsed ticks
    P = arrays.position[latest, entities]
    V = arrays.velocity[latest, entities]
    tgt_p = arrays.position[min(target, T-1), entities]
    tgt_v = arrays.velocity[min(target, T-1), entities]
    dt = arrays.tick_seconds
    if not math.isclose(dt, .1):
        raise ValueError("the exact atom rule here assumes 0.1 s ticks")
    # |T - P - V k/10| <= 1 + k/20  <=>  |20 (T-P) - 2 k V| <= 20 + k   (micro-units)
    position_atom = np.abs(20*(tgt_p-P)-2*k[:, None]*V) <= (20+k)[:, None]
    velocity_atom = tgt_v == V
    anchor = np.concatenate([(P+V*k[:, None]*dt)/MICRO, V/MICRO], 1)
    truth = np.concatenate([tgt_p/MICRO, tgt_v/MICRO], 1)
    band = np.concatenate([np.repeat(((1+k*dt/2)*1e-6)[:, None], 3, 1), np.zeros((len(k), 3))], 1)
    return {"anchor": anchor, "truth": truth, "atom": np.concatenate([position_atom, velocity_atom], 1),
            "band": band, "elapsed": k*dt, "valid": valid}


# ---------------------------------------------------------------- features

def _wrap(angle):
    return (angle+180.) % 360.-180.


def cell_features(arrays: EpisodeArrays, cutoff: int, entities: np.ndarray):
    """Causal feature vector per (entity, cutoff); only ticks <= cutoff are read."""
    dt = arrays.tick_seconds
    T0 = cutoff
    present = arrays.present
    def at(array, lag):
        t = T0-lag
        if t < 0:
            return np.zeros_like(array[0, entities]), np.zeros(len(entities), bool)
        return array[t, entities], present[t, entities]
    v0 = arrays.velocity[T0, entities]/MICRO
    p0 = arrays.position[T0, entities]/MICRO
    ok0 = present[T0, entities]
    feats, names = [], []
    def add(name, values):
        values = np.asarray(values, np.float64)
        if values.ndim == 1:
            values = values[:, None]
        feats.append(values)
        names.extend(f"{name}[{i}]" for i in range(values.shape[1]))
    add("present_at_cutoff", ok0)
    add("velocity", v0)
    add("speed", np.linalg.norm(v0, axis=1))
    add("speed_xy", np.linalg.norm(v0[:, :2], axis=1))
    add("altitude", p0[:, 2])
    for lag in VELOCITY_LAGS:
        v, ok = at(arrays.velocity, lag)
        dv = np.where((ok & ok0)[:, None], (v0-v/MICRO)/(lag*dt), 0.)
        add(f"accel_lag{lag}", dv)
        add(f"accel_lag{lag}_ok", ok & ok0)
    for lag in RESIDUAL_LAGS:
        p, okp = at(arrays.position, lag)
        v, _ = at(arrays.velocity, lag)
        residual = np.where((okp & ok0)[:, None], p0-(p/MICRO+v/MICRO*lag*dt), 0.)
        add(f"cv_residual_lag{lag}", residual/(lag*dt))
    y0 = arrays.yaw[T0, entities]
    for lag in (1, 5, 10):
        y, _ = at(arrays.yaw, lag)
        rate = _wrap(y0-y)/(lag*dt)
        add(f"yaw_rate_lag{lag}", np.nan_to_num(rate))
    # Resolution-level persistence over the last second.
    window = slice(max(0, T0-10), T0+1)
    vel = arrays.velocity[window][:, entities]
    pos = arrays.position[window][:, entities]
    pres = present[window][:, entities]
    both = pres[1:] & pres[:-1]
    same_v = (vel[1:] == vel[:-1]).all(-1) & both
    step = np.abs(20*(pos[1:]-pos[:-1])-2*vel[:-1]) <= 21
    atom_p = step.all(-1) & both
    n = np.maximum(both.sum(0), 1)
    add("velocity_unchanged_rate_1s", same_v.sum(0)/n)
    add("position_atom_rate_1s", atom_p.sum(0)/n)
    changed = ~(arrays.velocity[1:T0+1, entities] == arrays.velocity[:T0, entities]).all(-1)
    last_change = np.where(changed.any(0), T0-1-np.argmax(changed[::-1], 0), -1)
    add("log_ticks_since_velocity_change", np.log1p(np.where(last_change >= 0, T0-last_change, 999)))
    add("history_ticks", np.log1p(present[max(0, T0-50):T0+1, entities].sum(0)))
    # SUMO lane / signal / leader context at the cutoff.
    lane_pos = arrays.lane_position[T0, entities]
    lane_len = arrays.lane_length[T0, entities]
    sumo = ~np.isnan(lane_pos)
    add("sumo", sumo)
    add("lane_remaining", np.nan_to_num(lane_len-lane_pos))
    add("lane_length", np.nan_to_num(lane_len))
    add("lane_speed", np.nan_to_num(arrays.lane_speed[T0, entities]))
    add("speed_over_limit", np.nan_to_num(np.linalg.norm(v0, axis=1)/arrays.allowed_speed[T0, entities]))
    add("lane_internal", np.nan_to_num(arrays.lane_internal[T0, entities]))
    add("crossed_stop_line", np.nan_to_num(arrays.crossed_stop_line[T0, entities]))
    add("signal", np.eye(len(SIGNAL_STATES))[arrays.signal[T0, entities]]*sumo[:, None])
    lane_changed = (arrays.lane[1:T0+1, entities] != arrays.lane[:T0, entities])
    last_lane = np.where(lane_changed.any(0), T0-1-np.argmax(lane_changed[::-1], 0), -1)
    add("log_ticks_since_lane_change", np.log1p(np.where(last_lane >= 0, T0-last_lane, 999))*sumo)
    signal_changed = (arrays.signal[1:T0+1, entities] != arrays.signal[:T0, entities])
    last_signal = np.where(signal_changed.any(0), T0-1-np.argmax(signal_changed[::-1], 0), -1)
    add("log_ticks_since_signal_change", np.log1p(np.where(last_signal >= 0, T0-last_signal, 999))*sumo)
    # Same-lane leader by front-bumper lane position (vehicles observed together).
    lanes = arrays.lane[T0]
    lead_gap = np.zeros(len(entities)); lead_dv = np.zeros(len(entities)); lead_acc = np.zeros(len(entities)); has_lead = np.zeros(len(entities))
    all_pos = arrays.lane_position[T0]
    all_v = np.linalg.norm(arrays.velocity[T0]/MICRO, axis=1)
    prev_v = np.linalg.norm(arrays.velocity[max(T0-5, 0)]/MICRO, axis=1)
    for i, e in enumerate(entities):
        if lanes[e] < 0:
            continue
        same = np.where((lanes == lanes[e]) & (all_pos > all_pos[e]) & present[T0])[0]
        if len(same):
            j = same[np.argmin(all_pos[same])]
            length_j = arrays.length[j] if not np.isnan(arrays.length[j]) else 5.
            lead_gap[i] = all_pos[j]-length_j-all_pos[e]
            lead_dv[i] = all_v[j]-all_v[e]
            lead_acc[i] = (all_v[j]-prev_v[j])/(5*dt)
            has_lead[i] = 1.
    add("has_leader", has_lead)
    add("leader_gap", np.clip(lead_gap, -10, 200))
    add("leader_relative_speed", lead_dv)
    add("leader_accel", lead_acc)
    add("leader_time_gap", np.clip(lead_gap/np.maximum(np.linalg.norm(v0, axis=1), .5), -10, 60)*has_lead)
    # Nearest other actors: relative position/velocity, distance-sorted.
    actors = np.where(present[T0] & np.isin(arrays.category, [CATEGORIES.index(c) for c in ("vehicle", "pedestrian", "uav")]))[0]
    P = arrays.position[T0]/MICRO; Vv = arrays.velocity[T0]/MICRO
    for slot in range(NEIGHBOURS):
        rel = np.zeros((len(entities), 7))
        feats.append(rel); names.extend(f"neighbour{slot}[{i}]" for i in range(7))
    block = feats[-NEIGHBOURS:]
    for i, e in enumerate(entities):
        others = actors[actors != e]
        if not len(others) or not present[T0, e]:
            continue
        d = np.linalg.norm(P[others]-P[e], axis=1)
        order = np.argsort(d)[:NEIGHBOURS]
        for slot, j in enumerate(others[order]):
            block[slot][i] = [*(P[j]-P[e]), *(Vv[j]-Vv[e]), np.log1p(d[order[slot]])]
    X = np.concatenate(feats, 1)
    return X.astype(np.float32), names
