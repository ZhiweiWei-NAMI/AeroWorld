"""Compact predictor of the primary UAV link fields: latency (ms) and handover_active.

Inputs at a cutoff ``c`` (a communication tick, multiple of 5) are causal: the
UAV's kinematic features (compact_data.cell_features), its latency/handover
history up to ``c``, the associated station at ``c`` and the static station
positions (roster ``initial_position_enu_m`` of facilities / base stations; they
reproduce the reported link distance to <1e-6 m).  Geometry for the target is
the constant-velocity extrapolation of the UAV only (no future information).
Latency uses the pose law (persistence atom + Gaussian mixture, unit = TRAIN
off-atom rate x elapsed); handover is Bernoulli.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from Dataset.world_model.model.compact_data import CATEGORIES, HORIZONS, MICRO, cell_features, load_episode

COMM_LAGS = (5, 10, 20, 50)
DWELL_CAP = 60


def station_positions(canonical):
    """(base stations, every facility/base-station position) from the static roster."""
    stations, positions = {}, {}
    with open(canonical) as stream:
        for line in stream:
            if '"record_kind": "entity"' not in line and '"record_kind":"entity"' not in line:
                continue
            record = json.loads(line)
            if record["record_kind"] != "entity":
                continue
            fields = record["fields"]
            roster = fields.get("archive.global_entity_roster") or {}
            kind = str(fields.get("roster.entity_kind", ""))
            if isinstance(roster, dict) and roster.get("initial_position_enu_m") and \
                    (fields.get("roster.entity_category") == "facility" or "base_station" in kind):
                position = [float(v) for v in roster["initial_position_enu_m"]]
                positions[fields["roster.entity_id"]] = position
                if "base_station" in kind:
                    stations[fields["roster.entity_id"]] = position
    return stations, positions


def comm_series(path, arrays):
    T, E = arrays.present.shape
    column = {e: i for i, e in enumerate(arrays.entities)}
    latency = np.full((T, E), np.nan); handover = np.full((T, E), -1, np.int64); station = np.full((T, E), "", dtype=object)
    # Link configuration reported with each sample (observable at its tick): effective range, handover margin and dwell.
    config = np.full((T, E, 3), np.nan)
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            e = column.get(row["entity_id"]); t = row["tick"]
            if e is None or t >= T:
                continue
            value = row["link_quality"]["latency_ms"]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                latency[t, e] = float(value)
            handover[t, e] = int(bool(row["handover"]["active"]))
            station[t, e] = row["handover"]["current_station_id"].split(":")[-1]
            policy = row.get("handover_policy") or {}
            for i, value in enumerate((row["link_quality"].get("effective_range_m"), policy.get("handover_margin_m"), policy.get("minimum_dwell_ticks"))):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    config[t, e, i] = float(value)
    return latency, handover, station, config


def _episode_cells(arrays_path, canonical, communication, cutoffs):
    a = load_episode(Path(arrays_path))
    stations, positions = station_positions(canonical)
    names_s = sorted(stations); S = np.array([stations[n] for n in names_s]) if names_s else np.zeros((0, 3))
    latency, handover, station, config = comm_series(communication, a)
    T = a.present.shape[0]
    uavs = np.where(a.category == CATEGORIES.index("uav"))[0]
    out, feature_names = [], None
    dt = a.tick_seconds
    for c in cutoffs:
        if c < 50 or c % 5 or c+min(HORIZONS) >= T:
            continue
        ents = uavs[~np.isnan(latency[c, uavs]) & a.present[c, uavs]]
        if not len(ents):
            continue
        X, names = cell_features(a, c, ents)
        P = a.position[c, ents]/MICRO; V = a.velocity[c, ents]/MICRO
        for hi, h in enumerate(HORIZONS):
            target = c+h
            if target >= T:
                continue
            ok = ~np.isnan(latency[target, ents]) & (handover[target, ents] >= 0)
            if not ok.any():
                continue
            feats, fnames = [], []
            def add(name, v):
                feats.append(np.asarray(v, np.float64).reshape(len(ents), -1)); fnames.extend(f"{name}[{i}]" for i in range(feats[-1].shape[1]))
            lat0 = latency[c, ents]
            add("latency", lat0)
            for lag in COMM_LAGS:
                prev = latency[c-lag, ents]
                add(f"latency_rate_lag{lag}", np.nan_to_num((lat0-prev)/(lag*dt))); add(f"latency_lag{lag}_ok", ~np.isnan(prev))
            add("handover_now", handover[c, ents] == 1)
            changed = np.zeros(len(ents)); since = np.full(len(ents), DWELL_CAP, float)
            for i, e in enumerate(ents):
                hist = [station[t, e] for t in range(max(0, c-DWELL_CAP), c+1, 5)]
                for back, (s0, s1) in enumerate(zip(hist[::-1][1:], hist[::-1][:-1])):
                    if s0 and s1 and s0 != s1:
                        since[i] = 5*(back+1); changed[i] = 1; break
            add("ticks_since_station_change", since/DWELL_CAP); add("station_changed_recently", changed)
            # Candidates: roster base stations; the associated station (observed at c) may be another facility.
            future = P+V*h*dt
            here = np.array([positions.get(station[c, e], [np.nan]*3) for e in ents])
            cur_now = np.linalg.norm(P-here, axis=1); cur_fut = np.linalg.norm(future-here, axis=1)
            if len(S):
                d_now = np.linalg.norm(P[:, None]-S[None], axis=-1); d_fut = np.linalg.norm(future[:, None]-S[None], axis=-1)
                same = np.array([[station[c, e] == n for n in names_s] for e in ents])
                other_now = np.where(same, np.inf, d_now).min(1); other_fut = np.where(same, np.inf, d_fut).min(1)
            else:
                other_now = other_fut = np.full(len(ents), np.inf)
            for name, v in (("station_distance", cur_now), ("station_distance_cv_target", cur_fut),
                            ("other_station_distance", other_now), ("other_station_distance_cv_target", other_fut)):
                add(name, np.nan_to_num(np.where(np.isinf(v), np.nan, v), nan=-1.))
            add("station_distance_change_cv", np.nan_to_num(cur_fut-cur_now))
            adv_now = np.nan_to_num(np.where(np.isinf(other_now), np.nan, cur_now-other_now), nan=-100.)
            adv_fut = np.nan_to_num(np.where(np.isinf(other_fut), np.nan, cur_fut-other_fut), nan=-100.)
            add("distance_advantage", np.clip(adv_now, -100, 100)); add("distance_advantage_cv_target", np.clip(adv_fut, -100, 100))
            add("stations", np.full(len(ents), len(S)+(1-np.isin([station[c, e] for e in ents], names_s))))
            # Link model (compute_comm.py): latency = 20 + 200 (1 - score), score = (1 - d/R) F with range R and a
            # factor F (weather x operation x availability) inferred from the cutoff sample; extrapolated with the
            # constant-velocity distance to the current station and, if the handover rule fires, to the best other one.
            R = config[c, ents, 0]; margin = config[c, ents, 1]; dwell = config[c, ents, 2]
            score0 = 1-(lat0-20.)/200.
            base = np.clip(1-cur_now/R, 1e-3, None)
            factor = np.clip(score0/base, 0, 2)
            phys_cur = 20.+200.*(1-np.clip((1-cur_fut/R), 0, 1)*factor)
            other_fut_f = np.where(np.isinf(other_fut), np.nan, other_fut)
            phys_other = 20.+200.*(1-np.clip((1-other_fut_f/R), 0, 1)*factor)
            rule = (np.nan_to_num(cur_fut-other_fut_f, nan=-1e9) > margin) & (since >= dwell)
            phys = np.where(rule, phys_other, phys_cur)
            add("link_factor", np.nan_to_num(factor)); add("effective_range", np.nan_to_num(R))
            add("latency_phys_target", np.nan_to_num(phys, nan=lat0)); add("latency_phys_delta", np.nan_to_num(phys-lat0))
            add("latency_phys_current_station", np.nan_to_num(phys_cur, nan=lat0))
            add("handover_rule", rule); add("advantage_minus_margin", np.clip(np.nan_to_num(adv_fut-margin, nan=-100.), -100, 100))
            add("dwell_minus_minimum", np.nan_to_num(since-dwell, nan=0.)/DWELL_CAP)
            Xc = np.concatenate([X]+feats, 1)
            feature_names = names+fnames
            truth = latency[target, ents]
            out.append({"X": Xc[ok].astype(np.float32), "horizon": np.full(ok.sum(), hi), "anchor": lat0[ok], "truth": truth[ok],
                        "atom": np.round(truth[ok]*MICRO) == np.round(lat0[ok]*MICRO), "elapsed": np.full(ok.sum(), h*dt),
                        "handover": handover[target, ents][ok].astype(np.float32), "cutoff": np.full(ok.sum(), c),
                        "physics": np.nan_to_num(phys, nan=lat0)[ok].astype(np.float64),
                        "entity": np.array(a.entities, dtype=object)[ents][ok], "episode": np.full(ok.sum(), a.episode, dtype=object)})
    cells = {k: np.concatenate([o[k] for o in out]) for k in out[0]} if out else None
    return cells, feature_names


def build_comm_cells(config, arrays_dir, episodes, cutoffs_of, workers=12):
    jobs = [(str(Path(arrays_dir)/f"{e}.npz"), config["episodes"][e]["canonical"], config["episodes"][e]["communication"], list(cutoffs_of(e)))
            for e in episodes]
    with ProcessPoolExecutor(workers) as pool:
        parts = [p for p in pool.map(_episode_cells, *zip(*jobs)) if p[0] is not None]
    names = parts[0][1]
    return {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}, names


class CompactCommPredictor(nn.Module):
    def __init__(self, features, hidden=128, depth=2, dropout=.1, mixture=3, micro=False):
        super().__init__()
        self.mixture = mixture
        self.horizon = nn.Embedding(len(HORIZONS), 8)
        self.input = nn.Linear(features+8, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2*hidden), nn.GELU(), nn.Dropout(dropout),
                                                   nn.Linear(2*hidden, hidden)) for _ in range(depth)])
        self.latency = nn.Linear(hidden, 1+3*mixture+int(micro))
        self.handover = nn.Linear(hidden, 1)
        nn.init.zeros_(self.latency.weight); nn.init.zeros_(self.latency.bias)

    def forward(self, x, horizon):
        h = self.input(torch.cat((x, self.horizon(horizon)), -1))
        for block in self.blocks:
            h = h+block(h)
        return self.latency(h).unsqueeze(1), self.handover(h).squeeze(-1)
