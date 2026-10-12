"""Batched compact-law sampling and causal, joint future-state assembly.

First-loop approximation (no retraining or future inputs): sample the trained
0.5/1 s marginals; join pose/velocity endpoints by cubic Hermite curves; after
1 s continue the sampled terminal velocity. Communication after 1 s follows
the link-distance model and its reported margin/dwell policy on that path.
The long-horizon extension is physical, not a learned 5 s marginal.

One entity-level atom quantile and mixture quantile are reused across components
and horizons. Component-specific probabilities/weights still determine the
indicator, preserving their distinct marginals. Position/velocity share the
same axis residual rank. Entities are conditionally independent by default;
shared_rho enables a common Gaussian residual factor for a sensitivity study.
Ensemble member identity is held across components, modalities and horizons.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
import torch

from Dataset.world_model.model.compact_data import CATEGORIES, HORIZONS, MICRO, cell_features
from Dataset.world_model.model.compact_model import CompactStatePredictor, FEATURE_CLIP, law_terms
from Dataset.world_model.model.compact_comm import CompactCommPredictor, COMM_LAGS, DWELL_CAP
from .rule_predicates import FutureStates, communication_values, numeric


def causal_pose_arrays(arrays, source_poses, max_tick):
    """Connect compact UAV rows to the formal trajectory authority.

    Copy only when a real pose AND velocity exist. Never fabricate a missing
    velocity, activity or yaw; feature masks retain absence. No source tick
    after max_tick is consulted. Formal source poses replace stale compact
    poses (e.g. the newer L6 executed episode); non-UAV context stays compact.
    """
    a = replace(arrays, **{k: getattr(arrays, k).copy() for k in
                          ("present", "position", "velocity", "yaw", "activity")},
                activity_vocab=list(arrays.activity_vocab))
    index = {e: i for i, e in enumerate(a.entities)}
    count = 0
    for t, entities in source_poses.items():
        if t > max_tick:
            continue
        for e, row in entities.items():
            if e not in index or row.get("vel_mps") is None:
                continue
            i = index[e]
            p = np.rint(np.asarray(row["pos_enu"]) * MICRO).astype(np.int64)
            v = np.rint(np.asarray(row["vel_mps"]) * MICRO).astype(np.int64)
            if a.present[t, i] and np.array_equal(a.position[t, i], p) and np.array_equal(a.velocity[t, i], v):
                continue
            a.position[t, i], a.velocity[t, i] = p, v
            a.present[t, i] = True
            if isinstance(row.get("yaw_deg"), (float, int)):
                a.yaw[t, i] = row["yaw_deg"]
            activity = row.get("activity_type", row.get("state"))
            if isinstance(activity, str) and activity:
                if activity not in a.activity_vocab:
                    a.activity_vocab.append(activity)
                a.activity[t, i] = a.activity_vocab.index(activity)
            count += 1
    return a, count


STATIC_REFERENCE_CATEGORIES = frozenset({"vehicle_anchor", "crowd_anchor"})


def communication_features(a, cutoff, entities, context, *, return_physics=False):
    """Inference-only counterpart of compact_comm._episode_cells.

    Does not call its training builder, which checks future target presence.
    Names/order are checked against each checkpoint at inference.
    """
    X, names = cell_features(a, cutoff, entities)
    observed = [context.comm[cutoff][a.entities[e]] for e in entities]
    lat0 = np.array([communication_values(r)["latency"] for r in observed])
    handover_now = np.array([communication_values(r)["handover"] for r in observed])
    if not np.isfinite(lat0).all() or not np.isfinite(handover_now).all():
        raise ValueError(f"missing observed latency/handover at {a.episode}:{cutoff}")
    station_ids = [r["handover"]["current_station_id"].split(":")[-1] for r in observed]
    station_names = sorted(context.stations)
    stations = np.array([context.stations[e] for e in station_names]).reshape(-1, 3)
    P, V = a.position[cutoff, entities] / MICRO, a.velocity[cutoff, entities] / MICRO
    outputs, anchors = [], []
    for h in HORIZONS:
        parts, fnames = [X], list(names)
        def add(name, value):
            value = np.asarray(value, np.float64).reshape(len(entities), -1)
            parts.append(value)
            fnames.extend(f"{name}[{i}]" for i in range(value.shape[1]))
        add("latency", lat0)
        for lag in COMM_LAGS:
            prev = np.array([communication_values(context.comm[cutoff-lag][a.entities[e]])["latency"]
                             if a.entities[e] in context.comm.get(cutoff-lag, {}) else np.nan for e in entities])
            # Zero plus an explicit availability mask is the checkpoint's feature encoding.
            add(f"latency_rate_lag{lag}", np.nan_to_num((lat0-prev)/(lag*a.tick_seconds)))
            add(f"latency_lag{lag}_ok", np.isfinite(prev))
        add("handover_now", handover_now)
        changed, since = np.zeros(len(entities)), np.full(len(entities), DWELL_CAP, float)
        for i, e in enumerate(entities):
            hist = [context.comm.get(t, {}).get(a.entities[e]) for t in range(max(0, cutoff-DWELL_CAP), cutoff+1, 5)]
            for back, (r0, r1) in enumerate(zip(hist[::-1][1:], hist[::-1][:-1])):
                if r0 is not None and r1 is not None and r0["handover"]["current_station_id"] != r1["handover"]["current_station_id"]:
                    since[i], changed[i] = 5*(back+1), 1.
                    break
        add("ticks_since_station_change", since/DWELL_CAP)
        add("station_changed_recently", changed)
        future = P + V*h*a.tick_seconds
        here = np.array([context.positions_static[e] for e in station_ids])
        now_d, future_d = np.linalg.norm(P-here, axis=1), np.linalg.norm(future-here, axis=1)
        if len(stations):
            same = np.array([[e == s for s in station_names] for e in station_ids])
            other_now = np.where(same, np.inf, np.linalg.norm(P[:, None]-stations[None], axis=-1)).min(1)
            other_future = np.where(same, np.inf, np.linalg.norm(future[:, None]-stations[None], axis=-1)).min(1)
        else:
            other_now = other_future = np.full(len(entities), np.inf)
        for name, value in (("station_distance", now_d), ("station_distance_cv_target", future_d),
                            ("other_station_distance", other_now), ("other_station_distance_cv_target", other_future)):
            add(name, np.nan_to_num(np.where(np.isinf(value), np.nan, value), nan=-1.))
        add("station_distance_change_cv", np.nan_to_num(future_d-now_d))
        adv_now = np.nan_to_num(np.where(np.isinf(other_now), np.nan, now_d-other_now), nan=-100.)
        adv_future = np.nan_to_num(np.where(np.isinf(other_future), np.nan, future_d-other_future), nan=-100.)
        add("distance_advantage", np.clip(adv_now, -100, 100))
        add("distance_advantage_cv_target", np.clip(adv_future, -100, 100))
        add("stations", len(stations)+(1-np.isin(station_ids, station_names)))
        config = np.array([[numeric(r["link_quality"]["effective_range_m"]),
                            numeric(r["handover_policy"]["handover_margin_m"]),
                            numeric(r["handover_policy"]["minimum_dwell_ticks"])] for r in observed])
        R, margin, dwell = config.T
        score0 = 1-(lat0-20.)/200.
        base = np.clip(1-now_d/R, 1e-3, None)
        factor = np.clip(score0/base, 0, 2)
        phys_cur = 20.+200.*(1-np.clip(1-future_d/R, 0, 1)*factor)
        other_fut_f = np.where(np.isinf(other_future), np.nan, other_future)
        phys_other = 20.+200.*(1-np.clip(1-other_fut_f/R, 0, 1)*factor)
        rule = (np.nan_to_num(future_d-other_fut_f, nan=-1e9) > margin) & (since >= dwell)
        phys = np.where(rule, phys_other, phys_cur)
        # These masks/sentinels reproduce the trained schema, not output states.
        # A forecast anchor itself must have real finite physical operands.
        if not np.isfinite(config).all() or np.any(R <= 0) or not np.isfinite([phys, phys_cur, factor]).all():
            raise ValueError(f"missing observed link geometry/configuration at {a.episode}:{cutoff}")
        add("link_factor", factor)
        add("effective_range", R)
        add("latency_phys_target", phys)
        add("latency_phys_delta", phys-lat0)
        add("latency_phys_current_station", phys_cur)
        add("handover_rule", rule)
        add("advantage_minus_margin", np.clip(np.nan_to_num(adv_future-margin, nan=-100.), -100, 100))
        add("dwell_minus_minimum", (since-dwell)/DWELL_CAP)
        outputs.append(np.concatenate(parts, 1).astype(np.float32))
        anchors.append(phys)
    result = (np.stack(outputs), fnames)
    return (*result, np.stack(anchors)) if return_physics else result


def link_rollout(a, cutoff, entities, context, position, offsets):
    """Causal station association and link physics at the producer's 5-tick cadence.

    The sampled path supplies distance. Weather, range and operational factors
    persist from the observed link; future station outages/logical declarations
    are not observable. A geometric handover is a one-communication-tick pulse.
    """
    S, T, N = position.shape[:3]
    latency = np.empty((S, T, N))
    handover = np.zeros((S, T, N), dtype=bool)
    names = sorted(context.stations)
    station_pos = np.array([context.stations[e] for e in names]).reshape(-1, 3)
    for i, e in enumerate(entities):
        row = context.comm[cutoff][a.entities[e]]
        quality, policy = row["link_quality"], row["handover_policy"]
        station = row["handover"]["current_station_id"].split(":")[-1]
        current = np.broadcast_to(context.positions_static[station], (S, 3)).copy()
        current_id = np.full(S, names.index(station) if station in names else -1)
        R, margin, dwell = (numeric(quality["effective_range_m"]),
                            numeric(policy["handover_margin_m"]), numeric(policy["minimum_dwell_ticks"]))
        factors = [numeric(quality[k]) for k in ("weather_attenuation", "operational_factor", "availability")]
        if not np.isfinite([R, margin, dwell, *factors]).all() or R <= 0:
            raise ValueError(f"missing cutoff link physics: {a.episode}:{cutoff}:{a.entities[e]}")
        last = policy["last_switch_tick"]
        if last == "none" or row["handover"]["basis"] == "geometric_station_switch":
            # The row's policy describes the decision before the producer
            # records this tick's initial association / completed switch.
            last_switch = np.full(S, float(cutoff))
        elif type(last) in (int, float) and np.isfinite(last) and last <= cutoff:
            last_switch = np.full(S, float(last))
        else:
            raise ValueError(f"invalid reported last switch: {last}")
        latency[:, 0, i] = communication_values(row)["latency"]
        active = communication_values(row)["handover"]
        if not np.isfinite(active):
            raise ValueError(f"missing observed handover: {a.episode}:{cutoff}:{a.entities[e]}")
        handover[:, 0, i] = bool(active)
        for ti, offset in enumerate(offsets[1:], 1):
            distance = np.linalg.norm(position[:, ti, i]-current, axis=-1)
            switched = np.zeros(S, bool)
            if offset % 5 == 0 and len(names):
                distances = np.linalg.norm(position[:, ti, i, None]-station_pos[None], axis=-1)
                # Exclude the currently associated base station from alternatives.
                distances = np.where(np.arange(len(names))[None] == current_id[:, None], np.inf, distances)
                candidate = distances.argmin(1)
                best = distances[np.arange(S), candidate]
                elapsed = cutoff+offset-last_switch
                switched = (distance-best > margin) & (elapsed >= dwell)
                current[switched] = station_pos[candidate[switched]]
                current_id[switched] = candidate[switched]
                last_switch[switched] = cutoff+offset
                distance = np.where(switched, best, distance)
            score = np.clip(np.clip(1-distance/R, 0, 1)*np.prod(factors), 0, 1)
            latency[:, ti, i] = 20.+200.*(1-score)
            handover[:, ti, i] = switched
    return latency, handover


class CompactSampler:
    def __init__(self, pose_checkpoints, comm_checkpoints=(), device="cpu", *,
                 handover_checkpoints=(), pose_pool=None, latency_shrinkage=.1):
        if device not in ("cpu", "cuda:4"):
            raise ValueError("Task C permits CPU or physical GPU 4 only")
        self.device = torch.device(device)
        self.pose = self._load(pose_checkpoints, False)
        self.comm = self._load(comm_checkpoints, True)
        self.handover = self._load(handover_checkpoints, True)
        pool_path = Path(pose_pool or "/mnt/data2/weizhiwei/AERO_WORLD/Dataset/world_model/runs/compact/logs/pool_lambda_micro.json")
        self.pose_pool = json.loads(pool_path.read_text())
        self.latency_shrinkage = float(latency_shrinkage)
        if not 0 <= self.latency_shrinkage <= 1:
            raise ValueError("latency shrinkage must be in [0,1]")
        if not self.pose:
            raise ValueError("at least one real pose checkpoint is required")
        if self.comm and len(self.comm) != len(self.pose):
            raise ValueError("pair pose/communication ensemble members explicitly (equal counts)")
        if len(self.handover) != len(self.comm):
            raise ValueError("separate latency and handover ensembles must have equal counts")
        if any(c["args"].get("physics_anchor") != "True" for _, c in self.comm):
            raise ValueError("latency checkpoints must use --physics-anchor")

    def _load(self, paths, comm):
        models = []
        for path in paths:
            c = torch.load(Path(path), map_location="cpu", weights_only=False)
            o = c["args"]
            if comm:
                m = CompactCommPredictor(len(c["names"]), int(o["hidden"]), 2, float(o["dropout"]), int(o["mixture"]),
                                         float(o.get("micro_scale", 0)) > 0)
            else:
                m = CompactStatePredictor(len(c["names"]), len(c["vocab"]), int(o["hidden"]), int(o["depth"]),
                                          float(o["dropout"]), int(o["mixture"]), float(o.get("micro_scale", 0)) > 0)
            m.load_state_dict(c["model"])
            m.to(self.device).eval()
            models.append((m, c))
        return models

    @torch.inference_mode()
    def infer(self, a, cutoff, context, *, comm_arrays=None):
        if cutoff < 50 or cutoff % 5:
            raise ValueError("cutoff must be a multiple of 5 with 50 ticks of history")
        # Current presence is causal. Future births/deaths are not observed.
        entities = np.flatnonzero(a.present[cutoff])
        if not len(entities):
            raise ValueError("no observed entities at cutoff")
        # L3's logical-contract semantic.spawn_zone records are fixed map
        # references, not moving vehicles/crowds. Their genuine geometry has a
        # point-mass law and does not need a DEV-fitted dynamic pooling weight.
        # Validate this against causal source observations, never future labels.
        static = {e for e in entities if CATEGORIES[a.category[e]] in STATIC_REFERENCE_CATEGORIES}
        for e in static:
            observed = a.present[:cutoff+1, e]
            if np.any(a.velocity[:cutoff+1, e][observed] != 0) or np.any(
                    a.position[:cutoff+1, e][observed] != a.position[cutoff, e]):
                raise ValueError(f"fixed reference geometry changed in causal source: {a.episode}:{a.entities[e]}@{cutoff}")
        X, names = cell_features(a, cutoff, entities)
        horizon = torch.as_tensor(np.repeat([0, 1], len(entities)), device=self.device)
        laws = []
        for model, c in self.pose:
            if names != c["names"]:
                raise ValueError("pose feature schema differs from checkpoint")
            feature = X.copy()
            for prefix in filter(None, c["args"]["drop_features"].split(",")):
                cols = [i for i, n in enumerate(names) if n.startswith(prefix)]
                feature[:, cols] = c["mean"][cols]
            x = np.clip((feature-c["mean"])/c["std"], -FEATURE_CLIP, FEATURE_CLIP)
            acts = [c["vocab"].get(a.activity_vocab[a.activity[cutoff, e]], len(c["vocab"]))
                    if a.activity[cutoff, e] >= 0 else len(c["vocab"]) for e in entities]
            raw = model(torch.as_tensor(np.tile(x, (2, 1)), device=self.device, dtype=torch.float32),
                        torch.as_tensor(np.tile(a.category[entities], 2), device=self.device),
                        torch.as_tensor(np.tile(acts, 2), device=self.device), horizon)
            elapsed = np.repeat(np.asarray(HORIZONS)*a.tick_seconds, len(entities))
            unit = np.tile(c["rates"][a.category[entities]], (2, 1))*elapsed[:, None]
            lp, _, lw, off, scale = law_terms(raw, float(c["args"]["scale_floor"]),
                                             torch.as_tensor(unit, device=self.device),
                                             float(c["args"].get("micro_scale", 0)))
            pooled = np.empty((2, len(entities), 6))
            for hi, h in enumerate(HORIZONS):
                for i, e in enumerate(entities):
                    category = CATEGORIES[a.category[e]]
                    for sl, family in ((slice(0, 3), "position"), (slice(3, 6), "velocity")):
                        key = f"{category}|{family}"
                        lam = (1. if e in static else float(self.pose_pool[f"{key}@{h}"] if
                               f"{key}@{h}" in self.pose_pool else self.pose_pool[key]))
                        if not 0 <= lam <= 1:
                            raise ValueError(f"invalid shrinkage: {key}@{h}={lam}")
                        pooled[hi, i, sl] = lam
            p = lp.exp().cpu().numpy().reshape(2, -1, 6)
            laws.append((pooled+(1-pooled)*p,
                         lw.exp().cpu().numpy().reshape(2, len(entities), 6, -1),
                         (off.cpu().numpy()*unit[..., None]).reshape(2, len(entities), 6, -1),
                         (scale.cpu().numpy()*unit[..., None]).reshape(2, len(entities), 6, -1)))
        comm_entities = np.array([e for e in entities if a.category[e] == CATEGORIES.index("uav") and
                                  a.entities[e] in context.comm.get(cutoff, {}) and
                                  np.isfinite(communication_values(context.comm[cutoff][a.entities[e]])["latency"])], dtype=int)
        comm_laws, handovers, physics = [], [], None
        if len(comm_entities) and self.comm:
            # Preserve the communication training source's history masks. The
            # trajectory authority may restore pose rows missing in that source;
            # those restored rows drive sampled geometry, not cached features.
            feature_arrays = a if comm_arrays is None else comm_arrays
            if feature_arrays.entities != a.entities:
                raise ValueError("communication/pose entity indexing differs")
            features, comm_names, physics = communication_features(feature_arrays, cutoff, comm_entities, context,
                                                                   return_physics=True)
            hi = torch.as_tensor(np.repeat([0, 1], len(comm_entities)), device=self.device)
            for model, c in self.comm:
                if comm_names != c["names"]:
                    raise ValueError("communication feature schema differs from checkpoint")
                x = np.clip((features.reshape(-1, len(comm_names))-c["mean"])/c["std"], -FEATURE_CLIP, FEATURE_CLIP)
                raw, _ = model(torch.as_tensor(x, device=self.device, dtype=torch.float32), hi)
                unit = np.repeat(np.array(HORIZONS)*a.tick_seconds*float(c["rate"]), len(comm_entities))[:, None, None]
                lp, _, lw, off, scale = law_terms(raw, float(c["args"]["scale_floor"]),
                                                 torch.as_tensor(unit[..., 0], device=self.device),
                                                 float(c["args"].get("micro_scale", 0)))
                p = lp.exp().cpu().numpy().reshape(2, -1, 1)
                comm_laws.append((self.latency_shrinkage+(1-self.latency_shrinkage)*p,
                                  lw.exp().cpu().numpy().reshape(2, len(comm_entities), 1, -1),
                                  (off.cpu().numpy()*unit).reshape(2, len(comm_entities), 1, -1),
                                  (scale.cpu().numpy()*unit).reshape(2, len(comm_entities), 1, -1)))
            for model, c in self.handover:
                if comm_names != c["names"]:
                    raise ValueError("handover feature schema differs from checkpoint")
                x = np.clip((features.reshape(-1, len(comm_names))-c["mean"])/c["std"], -FEATURE_CLIP, FEATURE_CLIP)
                _, logit = model(torch.as_tensor(x, device=self.device, dtype=torch.float32), hi)
                handovers.append(torch.sigmoid(logit).cpu().numpy().reshape(2, -1))
        return {"entities": entities, "pose_laws": tuple(np.stack([p[i] for p in laws]) for i in range(4)),
                "comm_entities": comm_entities, "comm_laws": tuple(np.stack([p[i] for p in comm_laws]) for i in range(4))
                if comm_laws else None, "handover": np.stack(handovers) if handovers else None,
                "latency_anchor": physics}

    def sample(self, a, cutoff, context, samples=128, seed=0, shared_rho=0., inferred=None, tick_step=5,
               *, comm_arrays=None):
        if samples < 1 or not 0 <= shared_rho <= 1 or tick_step not in (1, 5):
            raise ValueError("S >= 1, rho in [0,1], and tick_step in {1,5} required")
        law = self.infer(a, cutoff, context, comm_arrays=comm_arrays) if inferred is None else inferred
        entities = law["entities"]
        n = len(entities)
        rng = np.random.default_rng(seed)
        model = rng.integers(len(self.pose), size=(samples, n))
        atom_u = rng.random((samples, n, 1))
        mixture_u = rng.random((samples, n, 1))
        independent_z = rng.standard_normal((samples, n, 3))
        shared_z = rng.standard_normal((samples, 1, 3))
        z = np.sqrt(1-shared_rho)*independent_z + np.sqrt(shared_rho)*shared_z
        residual = draw_laws(law["pose_laws"], model, atom_u, mixture_u, np.concatenate((z, z), -1),
                             bands=np.array([[1.25e-6]*3+[0.]*3, [1.5e-6]*3+[0.]*3]))
        p0, v0 = a.position[cutoff, entities]/MICRO, a.velocity[cutoff, entities]/MICRO
        endpoint_p = p0[None, None] + v0[None, None]*np.array([.5, 1.])[None, :, None, None] + residual[..., :3]
        endpoint_v = v0[None, None] + residual[..., 3:]
        offsets = np.arange(0, 51, tick_step)
        position, velocity = hermite_tail(p0, v0, endpoint_p, endpoint_v, offsets*.1)
        comm = {k: np.full((samples, len(offsets), n), np.nan) for k in
                ("latency", "handover", "packet_loss", "unavailable", "latency_threshold", "loss_threshold", "requirement")}
        index = {e: i for i, e in enumerate(entities)}
        for e, i in index.items():
            if a.entities[e] in context.comm.get(cutoff, {}):
                for k, value in communication_values(context.comm[cutoff][a.entities[e]]).items():
                    comm[k][:, :, i] = value
        if law["comm_laws"] is not None:
            ids = np.array([index[e] for e in law["comm_entities"]])
            # A shared entity quantile links latency atom/mixture choice to pose;
            # latency's continuous residual has its own rank, held across horizons.
            zc = rng.standard_normal((samples, len(ids), 1))
            rc = draw_laws(law["comm_laws"], model[:, ids], atom_u[:, ids], mixture_u[:, ids], zc,
                           bands=np.full((2, 1), 1e-5))
            lat0 = comm["latency"][:, 0, ids]
            endpoint_lat = np.maximum(0., law["latency_anchor"][None]+rc[..., 0])
            hu = rng.random((samples, len(ids)))
            hp = law["handover"][model[:, ids, None], np.arange(2)[None, None, :], np.arange(len(ids))[None, :, None]]
            handover = (hu[..., None] < hp).transpose(0, 2, 1)
            physical_lat, physical_hand = link_rollout(a, cutoff, law["comm_entities"], context,
                                                      position[:, :, ids], offsets)
            for ti, offset in enumerate(offsets):
                if offset == 0:
                    continue
                if offset <= 5:
                    alpha = offset/5
                    lat = lat0*(1-alpha)+endpoint_lat[:, 0]*alpha
                    hand = handover[:, 0]
                elif offset <= 10:
                    alpha = (offset-5)/5
                    lat = endpoint_lat[:, 0]*(1-alpha)+endpoint_lat[:, 1]*alpha
                    hand = handover[:, 1]
                else:
                    lat = physical_lat[:, ti]
                    hand = physical_hand[:, ti]
                comm["latency"][:, ti, ids] = lat
                comm["handover"][:, ti, ids] = hand
        return FutureStates(tuple(a.entities[e] for e in entities), cutoff+offsets, position, velocity, comm)


def draw_laws(laws, model, atom_u, mixture_u, z, bands=None):
    """Draw physical residuals from atom + censored mixture, with a rank copula."""
    p, w, mu, sigma = laws
    S, N = model.shape
    # Advanced indices give [S,N,H,C,(K)]; transpose to [S,H,N,C,(K)].
    idx = (model[:, :, None], np.arange(2)[None, None], np.arange(N)[None, :, None])
    p = p[idx].transpose(0, 2, 1, 3)
    w, mu, sigma = [v[idx].transpose(0, 2, 1, 3, 4) for v in (w, mu, sigma)]
    band = np.zeros((2, p.shape[-1])) if bands is None else bands
    lower = ndtr((-band[None, :, None, :, None]-mu)/sigma)
    upper = ndtr((band[None, :, None, :, None]-mu)/sigma)
    retained = lower + (1-upper)
    conditional_w = w*retained
    conditional_w /= conditional_w.sum(-1, keepdims=True)
    component = (mixture_u[:, None, :, :, None] > conditional_w.cumsum(-1)).sum(-1)
    component = np.minimum(component, w.shape[-1]-1)
    pick = lambda value: np.take_along_axis(value, component[..., None], -1)[..., 0]
    lo, hi, mass = pick(lower), pick(upper), pick(retained)
    u = ndtr(z)[:, None]*mass
    u = np.where(u <= lo, u, hi+(u-lo))
    off = pick(mu)+pick(sigma)*ndtri(np.clip(u, np.finfo(float).eps, 1-np.finfo(float).eps))
    return np.where(atom_u[:, None] < p, 0., off)


def hermite_tail(p0, v0, endpoint_p, endpoint_v, times):
    """C1 path preserving both sampled pose and velocity endpoint marginals."""
    S, _, N, _ = endpoint_p.shape
    p = np.empty((S, len(times), N, 3))
    v = np.empty_like(p)
    for ti, t in enumerate(times):
        if t > 1.:
            p[:, ti] = endpoint_p[:, 1]+endpoint_v[:, 1]*(t-1.)
            v[:, ti] = endpoint_v[:, 1]
            continue
        if t <= .5:
            a, va, b, vb, u = p0, v0, endpoint_p[:, 0], endpoint_v[:, 0], t/.5
        else:
            a, va, b, vb, u = endpoint_p[:, 0], endpoint_v[:, 0], endpoint_p[:, 1], endpoint_v[:, 1], (t-.5)/.5
        p[:, ti] = (2*u**3-3*u**2+1)*a+(u**3-2*u**2+u)*.5*va+(-2*u**3+3*u**2)*b+(u**3-u**2)*.5*vb
        v[:, ti] = (6*u*u-6*u)/.5*a+(3*u*u-4*u+1)*va+(-6*u*u+6*u)/.5*b+(3*u*u-2*u)*vb
    return p, v
