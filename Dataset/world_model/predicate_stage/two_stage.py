"""Compact joint futures -> isolated Qwen continuations -> probability mixture.

Only oracle mode reads future source states. K has no learned parameters. The
Task C API is confined to CompactFutureAdapter. Predicted relations are computed
with the existing C2 rules, never read from future supervision.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from Dataset.world_model.data.provenance import resolve_source_split
from Dataset.world_model.model.compact_data import MICRO, load_episode
from Dataset.world_model.model.derivations import derive
from Dataset.world_model.model.causal_input import cell
from Dataset.world_model.model.run_episode import chronological_prefix, scalar_positions, typed_read
from .qwen_predicate import QwenPredicateModel, pool_prefix, fork_prefix_cache
from .rule_predicates import (FutureStates, RuleContext, evaluate as evaluate_rule,
                              probabilities as rule_probabilities, communication_values,
                              SEPARATION, CAPACITY, OCCUPANCY, LATENCY)
from .sampling import CompactSampler, causal_pose_arrays, STATIC_REFERENCE_CATEGORIES

COMPACT = Path(__file__).resolve().parents[1] / "runs" / "compact"
OFFSETS = tuple(range(5, 51, 5))


def window_seed(seed, episode, cutoff, draw=0):
    return int.from_bytes(hashlib.sha256(f"two-stage:{seed}:{episode}:{cutoff}:{draw}".encode()).digest()[:8], "little")


def causal_arrays(arrays, cutoff):
    """Expose no post-cutoff array cells, even to a future sampler implementation."""
    time_fields = ("present", "position", "velocity", "yaw", "activity", "lane", "lane_position",
                   "lane_length", "lane_speed", "allowed_speed", "lane_internal", "signal", "crossed_stop_line")
    return replace(arrays, **{name: getattr(arrays, name)[:cutoff+1] for name in time_fields})


class CompactFutureAdapter:
    """C2 source/checkpoint adapter; inference always runs on CPU beside Qwen.

    C2 landed before this implementation. Keep its separate latency/handover
    checkpoints and comm_arrays API confined here so sampler revisions do not
    change Qwen or its heads. Failed inference is never replaced by persistence.
    """
    def __init__(self, *, arrays=COMPACT / "arrays", members=5, pose_pool=None,
                 pose_checkpoints=None, latency_checkpoints=None, handover_checkpoints=None,
                 shared_rho=0., train_noise_scale=1.):
        self.arrays_path = Path(arrays)
        self.shared_rho = shared_rho
        self.train_noise_scale = train_noise_scale
        self.sources = {}
        self.pose_paths = pose_checkpoints or [COMPACT / f"runs/f{i}_mix3_aug_micro/model.pt" for i in range(1, members+1)]
        self.latency_paths = latency_checkpoints or [COMPACT / f"runs/comm4_e{i}/model.pt" for i in range(1, members+1)]
        self.handover_paths = handover_checkpoints or [COMPACT / f"runs/comm2_e{i}/model.pt" for i in range(1, members+1)]
        self.api = "C2"
        self.sampler = CompactSampler(self.pose_paths, self.latency_paths, device="cpu",
                                      handover_checkpoints=self.handover_paths, pose_pool=pose_pool)

    def source(self, episode):
        if resolve_source_split(episode) not in ("TRAIN", "VALID"):
            raise ValueError("TEST source access is prohibited")
        if episode not in self.sources:
            arrays = load_episode(self.arrays_path / f"{episode}.npz")
            context = RuleContext(episode)
            # Index states once. Only causal views below can enter sample/infer.
            poses = context.pose_sources(len(arrays.present)-1)
            self.sources[episode] = (arrays, context, poses)
        return self.sources[episode]

    def causal_source(self, episode, cutoff):
        raw, context, poses = self.source(episode)
        observed = copy.copy(context)
        observed.comm = {t: r for t, r in context.comm.items() if t <= cutoff}
        observed_poses = {t: r for t, r in poses.items() if t <= cutoff}
        comm_arrays = causal_arrays(raw, cutoff)
        pose_arrays, _ = causal_pose_arrays(comm_arrays, observed_poses, cutoff)
        return pose_arrays, observed, comm_arrays

    def sample(self, episode, cutoff, k, seed, *, training=False):
        a, context, comm_arrays = self.causal_source(episode, cutoff)
        law = self.sampler.infer(a, cutoff, context, comm_arrays=comm_arrays)
        if training and self.train_noise_scale != 1.:
            # Explicit sensitivity experiment, NOT a fitted held-out noise law.
            # Preserve discrete gate masses and anchors; scale continuous offsets
            # and standard deviations together, including their temporal coupling.
            law = dict(law)
            for name in ("pose_laws", "comm_laws"):
                if law[name] is not None:
                    p, weights, offsets, scales = law[name]
                    law[name] = (p, weights, offsets*self.train_noise_scale, scales*self.train_noise_scale)
        return self.sampler.sample(a, cutoff, context, samples=k, seed=seed,
                                   shared_rho=self.shared_rho, inferred=law, tick_step=1, comm_arrays=comm_arrays)

    def observed(self, episode, cutoff):
        arrays, context, _ = self.causal_source(episode, cutoff)
        ids = np.flatnonzero(arrays.present[cutoff])
        entities = tuple(arrays.entities[i] for i in ids)
        position = arrays.position[cutoff, ids][None, None]/MICRO
        velocity = arrays.velocity[cutoff, ids][None, None]/MICRO
        names = ("latency", "handover", "packet_loss", "unavailable", "latency_threshold", "loss_threshold", "requirement")
        comm = {name: np.full(position.shape[:-1], np.nan) for name in names}
        for i, entity in enumerate(entities):
            if entity in context.comm.get(cutoff, {}):
                for name, value in communication_values(context.comm[cutoff][entity]).items():
                    comm[name][0, 0, i] = value
        return FutureStates(entities, np.asarray([cutoff]), position, velocity, comm)

    def oracle(self, episode, cutoff):
        """Explicit ablation: real future primaries, same cutoff entity population.

        UAV source authority follows C2; other entities use compact source arrays.
        Unobserved/truncated future cells retain missing status, never persistence.
        """
        raw, context, poses = self.source(episode)
        a, _, _ = self.causal_source(episode, cutoff)
        ids = np.flatnonzero(a.present[cutoff])
        entities = tuple(a.entities[i] for i in ids)
        ticks = np.asarray([cutoff, *(cutoff+h for h in OFFSETS)])
        position = np.full((1, len(ticks), len(ids), 3), np.nan)
        velocity = np.full_like(position, np.nan)
        true_uav = context.true_states(ticks, {t: r for t, r in poses.items() if t <= cutoff+50})
        lookup = {e: i for i, e in enumerate(true_uav.entities)}
        comm = {name: np.full(position.shape[:-1], np.nan) for name in true_uav.communication}
        for j, (e, i) in enumerate(zip(entities, ids)):
            if e in lookup:
                u = lookup[e]
                position[:, :, j], velocity[:, :, j] = true_uav.position[:, :, u], true_uav.velocity[:, :, u]
                for name in comm:
                    comm[name][:, :, j] = true_uav.communication[name][:, :, u]
            else:
                for ti, tick in enumerate(ticks):
                    if tick < len(raw.present) and raw.present[tick, i]:
                        position[0, ti, j] = raw.position[tick, i]/MICRO
                        velocity[0, ti, j] = raw.velocity[tick, i]/MICRO
        return FutureStates(entities, ticks, position, velocity, comm)

    def describe(self):
        return {"api": self.api, "device": "cpu", "pose_checkpoints": list(map(str, self.pose_paths)),
                "latency_checkpoints": list(map(str, self.latency_paths)),
                "handover_checkpoints": list(map(str, self.handover_paths)),
                "shared_rho": self.shared_rho, "train_noise_scale": self.train_noise_scale,
                "static_reference_geometry": sorted(STATIC_REFERENCE_CATEGORIES),
                "static_reference_policy": "recorded fixed spawn-zone geometry; point mass, no fitted pooling coefficient",
                "train_noise_note": "continuous-law inflation sensitivity only; not held-out calibration"}


def assert_causal_index(index):
    if any(t > index.cutoff for t, _ in index.states) or any(r["tick"] > index.cutoff for r in index.semantic_records):
        raise ValueError("post-cutoff observed state in graph input")
    if index.target_ticks:
        raise ValueError("prediction index still carries target ticks")


def future_feedback(index, future, sample):
    """Use the exact _encode feedback schema and native primary derivations."""
    expected = [index.cutoff+h for h in OFFSETS]
    lookup = {int(t): i for i, t in enumerate(future.ticks)}
    if not set(expected) <= lookup.keys():
        raise ValueError("sampler omitted a required five-tick future state")
    if set(future.entities)-index.entities.keys():
        raise ValueError("sample contains entities absent from the causal graph roster")
    latest_yaw = {}
    for (tick, entity), state in sorted(index.states.items()):
        if "pose.yaw_deg" in state:
            kind, value = cell(state["pose.yaw_deg"], None)
            if kind == "present":
                latest_yaw[entity] = value

    def payload(value):
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            return {"kind": "source_unknown"}
        # A draw is a concrete state, not a marginal record probability. Its
        # field embedding must match an identical observed cell exactly.
        return {"kind": "present", "value": value}

    result = []
    for tick in expected:
        ti, values = lookup[tick], {}
        for j, entity in enumerate(future.entities):
            p, v = future.position[sample, ti, j], future.velocity[sample, ti, j]
            cells = {(family, c): payload(float(vector[c]))
                     for family, vector in (("pose.position_enu_m", p), ("pose.velocity_enu_mps", v))
                     for c in range(3)}
            primary = {}
            if np.isfinite(p).all():
                primary["pose.position_enu_m"] = p.tolist()
            if np.isfinite(v).all():
                primary["pose.velocity_enu_mps"] = v.tolist()
            if index.entities[entity]["roster.entity_category"] == "uav":
                latency = float(future.communication["latency"][sample, ti, j])
                handover = float(future.communication["handover"][sample, ti, j])
                cells["communication.latency_ms", None] = payload(latency)
                if math.isfinite(handover) and handover not in (0., 1.):
                    raise ValueError("sampled handover must be boolean, not a marginal probability")
                cells["communication.handover_active", None] = payload(bool(handover) if math.isfinite(handover) else handover)
                if math.isfinite(latency):
                    primary["communication.latency_ms"] = latency
            # No fabricated heading for a stationary entity without yaw history.
            can_derive_yaw = ("pose.velocity_enu_mps" not in primary or
                              np.hypot(v[0], v[1]) > 1e-5 or entity in latest_yaw)
            derivation_input = primary if can_derive_yaw else {k: val for k, val in primary.items() if k != "pose.velocity_enu_mps"}
            # B2 has no capture tokens; observer camera fields are not registered
            # graph fields in this checkout. Only registered state derivations
            # (speed, yaw and link fields) enter rollout feedback.
            derived = derive(derivation_input, {"latest_yaw": latest_yaw.get(entity), "observer": False})
            if not can_derive_yaw:
                derived["motion.speed_mps"] = float(np.linalg.norm(v))
                cells["pose.yaw_deg", None] = payload(float("nan"))
            if "pose.yaw_deg" in derived:
                latest_yaw[entity] = derived["pose.yaw_deg"]
            for family, value in derived.items():
                if isinstance(value, list):
                    cells.update({(family, c): payload(val) for c, val in enumerate(value)})
                else:
                    cells[family, None] = payload(value)
            values[entity] = cells
        result.append((tick, values))
    return tuple(result)


def select_structure(structure, mask):
    ids = torch.where(mask)[0]
    integer_ids = ids.cpu().tolist()
    return {**structure, "embeddings": structure["embeddings"][:, ids], "times": structure["times"][ids],
            "token_entities": [structure["token_entities"][i] for i in integer_ids],
            "token_fields": [structure["token_fields"][i] for i in integer_ids]}


def select_cache(cache, indices):
    """Select/fork ALL cache tensor families, preserving gradients and isolation."""
    result = copy.copy(cache)
    result.layers = []
    def clone(value, name=""):
        if isinstance(value, torch.Tensor):
            if name in ("keys", "values") or name.startswith(("conv_states", "recurrent_states")):
                return value.index_select(0, indices.to(value.device))
            return value.clone()
        if isinstance(value, dict):
            return {k: clone(v, f"{name}.{k}") for k, v in value.items()}
        return copy.copy(value)
    for layer in cache.layers:
        copied = copy.copy(layer)
        copied.__dict__ = {key: clone(value, key) for key, value in vars(layer).items()}
        result.layers.append(copied)
    return result


def aggregate_outputs(outputs):
    """Stable log of the mean probabilities; never average logits or losses."""
    state = torch.stack([o["state_logits"].float() for o in outputs])
    transition = torch.stack([o["transition_logits"].float() for o in outputs])
    log_k = math.log(len(outputs))
    positive = torch.logsumexp(F.logsigmoid(state), 0)-log_k
    negative = torch.logsumexp(F.logsigmoid(-state), 0)-log_k
    mass = torch.logsumexp(transition.log_softmax(-1), 0)-log_k
    event = None
    sample_event = None
    if outputs[0]["event_logits"] is not None:
        sample_event = torch.stack([o["event_logits"].float() for o in outputs])
        event = torch.logsumexp(F.logsigmoid(sample_event), 0)-torch.logsumexp(F.logsigmoid(-sample_event), 0)
    return {"state_logits": positive-negative, "transition_logits": mass, "event_logits": event,
            "event_types": outputs[0]["event_types"],
            "state_std": state.sigmoid().std(0, correction=0),
            "transition_std": transition.softmax(-1).std(0, correction=0),
            "event_std": None if sample_event is None else sample_event.sigmoid().std(0, correction=0),
            "sample_state": state.sigmoid(), "sample_transition": transition.softmax(-1)}


# Eight operand channels at cutoff and ten future ticks, each with a presence
# mask. Different predicate families explicitly mark unused channels absent.
RULE_WIDTH = 11*2 + 6 + 3 + 11*8*2


def rule_features(future, context, rows, tuple_features, *, strength=None):
    """C2-aligned rule sequence, first change, and authentic sampled operands.

    All rules run at 0.1 s cadence; Qwen sees 0.5 s relation/operand cadence.
    Recorded cutoff truth persists until the rule actually changes, as in C2.
    Jeffreys priors use the actual K, with no invented Monte Carlo observations.
    """
    k, _, _ = future.position.shape[:3]
    strength = k if strength is None else strength
    cutoff = int(future.ticks[0])
    tick_ids = [int(np.flatnonzero(future.ticks == cutoff+h)[0]) for h in (0, *OFFSETS)]
    entities = {e: i for i, e in enumerate(future.entities)}
    values, states, masses, cutoff_rules = [], [], [], []
    for qi, row in enumerate(rows):
        pid, bindings = row["predicate_id"], row["bindings"]
        owner = None
        if row["family"] == "communication":
            record = context.communication_row(cutoff, bindings)
            if record is None:
                raise ValueError(f"missing causal communication owner: {bindings}")
            owner = record["entity_id"]
        truth = evaluate_rule(future, context, pid, bindings, owner=owner, fixed_scope=True)
        if not np.isfinite(truth).all():
            raise ValueError(f"sample rule operands unavailable: {row['episode']}@{cutoff} {pid} {bindings}")
        initial_id = int(tuple_features[qi, 30:33].argmax())
        if initial_id == 2:
            raise ValueError("rule prior requires an observed cutoff truth")
        initial = ("false", "true")[initial_id]
        aligned = np.empty_like(truth)
        aligned[:, 0] = initial_id
        for ti in range(1, len(future.ticks)):
            aligned[:, ti] = np.where(truth[:, ti] != truth[:, ti-1], truth[:, ti], aligned[:, ti-1])
        state = np.stack([aligned[:, np.flatnonzero(future.ticks == cutoff+h)[0]] for h in (5, 10, 20, 30, 50)], -1)
        first = np.where((aligned != initial_id).any(1), future.ticks[(aligned != initial_id).argmax(1)]-cutoff, 51)
        bins = np.stack([(first > lo) & (first <= hi) for lo, hi in zip((0, 5, 10, 20, 30, 50), (5, 10, 20, 30, 50, 51))], -1).astype(float)
        # Independently use the imported public C2 probability implementation.
        reference = rule_probabilities(truth, future.ticks, cutoff, initial)
        if not np.allclose(state.mean(0), list(reference["p_future_true"].values()), atol=1e-12) or not np.allclose(bins[:, :5].mean(0), reference["transition_hazard"], atol=1e-12):
            raise AssertionError("per-sample features disagree with C2 sequence semantics")
        operands = np.full((k, len(future.ticks), 8), np.nan)
        if pid == SEPARATION:
            a, b = (entities[bindings[name]] for name in ("first_aircraft", "second_aircraft"))
            delta = future.position[:, :, a]-future.position[:, :, b]
            velocity = future.velocity[:, :, a]-future.velocity[:, :, b]
            horizontal = np.linalg.norm(delta[..., :2], axis=-1)
            vertical = np.abs(delta[..., 2])
            distance = np.linalg.norm(delta, axis=-1)
            model = context.pair_models.get(tuple(sorted((bindings["first_aircraft"], bindings["second_aircraft"]))))
            effective = distance if model is None else np.maximum(horizontal/model["horizontal_limit_m"], vertical/model["vertical_limit_m"])*context.margin
            operands[..., 0] = effective/context.margin
            operands[..., 1] = distance
            operands[..., 2] = context.margin
            operands[..., 3] = -(delta*velocity).sum(-1)/np.maximum(distance, 1e-9)
            operands[..., 4] = horizontal
            operands[..., 5] = vertical
            operands[..., 6] = context.margin if model is None else model["horizontal_limit_m"]
            operands[..., 7] = context.margin if model is None else model["vertical_limit_m"]
        elif pid in (CAPACITY, OCCUPANCY):
            corridor = context.corridors[bindings["corridor"]]
            members = [entities[e] for e in future.entities if e in context.uavs]
            delta = future.position[:, :, members]-np.asarray(corridor["center_enu_m"])
            yaw = np.deg2rad(corridor["yaw_deg"])
            local = np.stack((np.cos(yaw)*delta[..., 0]+np.sin(yaw)*delta[..., 1], -np.sin(yaw)*delta[..., 0]+np.cos(yaw)*delta[..., 1], delta[..., 2]), -1)
            inside = (np.abs(local) <= np.asarray(corridor["extent_m"])+1e-9).all(-1) & np.isfinite(local).all(-1)
            count = inside.sum(-1)
            operands[..., 0] = count/max(float(corridor["capacity"]), 1.)
            operands[..., 1] = count
            operands[..., 2] = corridor["capacity"]
            if pid == OCCUPANCY:
                actor = members.index(entities[bindings["actor"]])
                operands[..., 3:6] = local[:, :, actor]/np.asarray(corridor["extent_m"])
            operands[..., 6] = np.min(np.max(np.abs(local)/np.asarray(corridor["extent_m"]), -1), -1)
            operands[..., 7] = inside.sum(-1)
        else:
            ei = entities[owner]
            for oi, name in enumerate(("latency", "latency_threshold", "handover", "packet_loss", "loss_threshold", "unavailable", "requirement"), 1):
                operands[..., oi] = future.communication[name][:, :, ei]
            operands[..., 0] = operands[..., 1]/operands[..., 2]
        operand = operands[:, tick_ids]
        present = np.isfinite(operand)
        signed_log = np.sign(np.where(present, operand, 0.))*np.log1p(np.abs(np.where(present, operand, 0.)))
        cutoff_onehot = np.stack((truth[:, 0] == 0, truth[:, 0] == 1, ~np.isfinite(truth[:, 0])), -1)
        values.append(np.concatenate((truth[:, tick_ids], aligned[:, tick_ids], bins, cutoff_onehot,
                                      np.stack((signed_log, present), -1).reshape(k, -1)), -1))
        states.append((strength*state+.5)/(strength+1.))
        masses.append((strength*bins+.5)/(strength+3.))
        cutoff_rules.append(cutoff_onehot[0])
    return {"rule": torch.tensor(np.stack(values, 1), dtype=torch.float32),
            "state_prior": torch.tensor(np.stack(states, 1), dtype=torch.float32),
            "mass_prior": torch.tensor(np.stack(masses, 1), dtype=torch.float32),
            "cutoff_rule": torch.tensor(np.stack(cutoff_rules), dtype=torch.float32)}


class TwoStageModel(QwenPredicateModel):
    """Same learned parameters as B2; only runtime prefixes/mixtures differ.

    Graph history and Qwen history prefill are shared across samples. Each graph
    branch copies the incremental relation/temporal state, retains gradients to
    the shared history, and only encodes its own future records. Future prefills
    and query continuations have independent batch rows.
    """
    def __init__(self, *args, history_tokens=128, sample_batch_size=2, query_batch_size=8,
                 context_features=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < history_tokens < self.max_prefix_tokens or sample_batch_size < 1 or query_batch_size < 1:
            raise ValueError("need 0 < history_tokens < max_prefix_tokens and positive sample/query batch sizes")
        self.history_tokens = history_tokens
        self.sample_batch_size = sample_batch_size
        self.query_batch_size = query_batch_size
        self.history_cache = None
        self.context_features = context_features
        if context_features:
            width = self.query.template.embedding_dim
            self.direct_context = torch.nn.Sequential(torch.nn.Linear(RULE_WIDTH+39, width), torch.nn.GELU(),
                                                       torch.nn.Linear(width, width))
            torch.nn.init.zeros_(self.direct_context[-1].weight)
            torch.nn.init.zeros_(self.direct_context[-1].bias)
            self.direct_context.to(self.query.template.weight.device)

    def structures(self, index, feedbacks):
        if self.history_cache is not None and self.history_cache.pooled_only:
            return self.history_cache.pooled(index, feedbacks, self.history_tokens, self.max_prefix_tokens,
                                             lambda: self._structures(index, feedbacks))
        return self._structures(index, feedbacks)

    def _structures(self, index, feedbacks):
        assert_causal_index(index)
        if self.history_cache is not None:
            history, cache_hit = self.history_cache.load(index)
        else:
            history = self.graph.encode_history(index, None, reconstruct=False, return_state=feedbacks is not None)
            cache_hit = False
        if feedbacks is None:
            prefix = chronological_prefix(self.qwen, pool_prefix(history, self.history_tokens), {})
            return None, prefix, [{"raw_graph_slots": history["embeddings"].shape[1],
                                   "history_tokens": prefix["next_token"], "future_tokens": 0, "future_ticks": 0,
                                   "history_graph_cache_hit": cache_hit}]
        historical = chronological_prefix(self.qwen, pool_prefix(history, self.history_tokens), {})
        history_slots = history["embeddings"].shape[1]
        futures, diagnostics = [], []
        for feedback in feedbacks:
            future = (self.history_cache.load_future(index, feedback, history["continuation_state"])
                      if self.history_cache is not None else
                      self.graph.encode_future(index, feedback, history["continuation_state"]))
            # Preserve all ten future ticks even when the total prefix is pooled.
            ticks = sorted(set(future["times"].cpu().tolist()))
            budget = self.max_prefix_tokens-self.history_tokens
            if budget < len(ticks):
                raise ValueError("future token budget must cover all ten sampled ticks")
            parts = [pool_prefix(select_structure(future, future["times"] == t),
                                 budget//len(ticks)+int(j < budget % len(ticks))) for j, t in enumerate(ticks)]
            pooled = {**future, "embeddings": torch.cat([p["embeddings"] for p in parts], 1),
                      "times": torch.cat([p["times"] for p in parts]),
                      "token_entities": sum((p["token_entities"] for p in parts), []),
                      "token_fields": sum((p["token_fields"] for p in parts), [])}
            appended = chronological_prefix(self.qwen, pooled, {}, token_start=historical["next_token"],
                                              rope_start=historical["next_rope"])
            futures.append(appended)
            diagnostics.append({"raw_graph_slots": history_slots+future["embeddings"].shape[1],
                                "history_tokens": historical["next_token"],
                                "future_tokens": appended["embeddings"].shape[1],
                                "future_ticks": len(ticks), "history_graph_shared": True,
                                "history_graph_cache_hit": cache_hit})
        return historical, futures, diagnostics

    def prefill_samples(self, index, structures):
        # A frozen CPU graph can be prepared in parallel while the only Qwen
        # process uses GPU 3. Only compact graph prefixes cross devices.
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.query.template.weight.device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(move(item) for item in value)
            return value
        structures = move(structures)
        history, futures, diagnostics = structures
        if history is None:
            return [(self.prefill(index, futures), None)], diagnostics
        prefix = self.prefill(index, history)
        blocks = []
        for start in range(0, len(futures), self.sample_batch_size):
            group = futures[start:start+self.sample_batch_size]
            count = len(group)
            # Fixed physical batch shape also avoids bf16 GEMM shape drift.
            # Discard duplicated padding before mixing probabilities or losses.
            group = group+[group[-1]]*(self.sample_batch_size-count)
            if len({f["embeddings"].shape[1] for f in group}) != 1:
                raise ValueError("sample prefix lengths differ; do not pad recurrent histories silently")
            embeddings = torch.cat([f["embeddings"] for f in group], 0)
            positions = torch.cat([f["positions"] for f in group], 1)
            out = self.qwen(inputs_embeds=embeddings, position_ids=positions,
                            past_key_values=fork_prefix_cache(prefix["cache"], len(group)), use_cache=True)
            sample_prefixes = [{**f, "token_entities": prefix["token_entities"]+f["token_entities"],
                                "token_fields": prefix["token_fields"]+f["token_fields"],
                                "hidden": torch.cat((prefix["hidden"], out.last_hidden_state[i:i+1]), 1),
                                "padding": i >= count}
                               for i, f in enumerate(group)]
            blocks.append((out.past_key_values, sample_prefixes))
        return blocks, diagnostics

    def score_samples(self, index, rows, prepared, context=None):
        blocks, _ = prepared
        if not self.context_features and len(blocks) == 1 and blocks[0][1] is None:
            return aggregate_outputs([super().forward(index, rows, blocks[0][0])])
        if self.context_features and context is None:
            raise ValueError("Task E requires authentic tuple/rule features")
        if len(blocks) == 1 and blocks[0][1] is None:
            prefix = blocks[0][0]
            blocks = [(prefix["cache"], [{**prefix, "padding": False}])]
        per_sample = []
        sample_offset = 0
        for cache, prefixes in blocks:
            sample_outputs = [[] for _ in prefixes]
            for start in range(0, len(rows), self.query_batch_size):
                group = rows[start:start+self.query_batch_size]
                count = len(group)
                padded = group+[group[-1]]*(self.query_batch_size-count)
                vector, query = self.query(padded, index)
                vectors = []
                for i in range(len(prefixes)):
                    if self.context_features:
                        def pad(value):
                            v = value[start:start+count]
                            return torch.cat((v, v[-1:].expand(self.query_batch_size-count, -1)))
                        direct = pad(context["tuple"])
                        cutoff_rule = pad(context["cutoff_rule"])
                        if "rule" in context:
                            si = min(sample_offset+i, len(context["rule"])-1)
                            rule = pad(context["rule"][si])
                            available = torch.ones(len(padded), 1)
                        else:
                            rule = torch.zeros(len(padded), RULE_WIDTH)
                            available = torch.zeros(len(padded), 1)
                        direct = torch.cat((direct, cutoff_rule, rule, available), -1).to(vector.device)
                        vectors.append(vector+self.direct_context(direct))
                    else:
                        vectors.append(vector)
                tokens = self.query_token(torch.cat(vectors))[:, None].to(self.qwen.dtype)
                indices = torch.arange(len(prefixes), device=tokens.device).repeat_interleave(self.query_batch_size)
                positions = scalar_positions(1, prefixes[0]["next_token"], prefixes[0]["next_rope"], tokens.device)
                hidden = self.qwen(inputs_embeds=tokens, position_ids=positions.expand(-1, len(tokens), -1),
                                   past_key_values=select_cache(cache, indices), use_cache=True).last_hidden_state
                for i, prefix in enumerate(prefixes):
                    features = self.role_features(vectors[i][query["owners"]]+query["role_vectors"])
                    bound = {**query, "features": features,
                             "query_hidden": hidden[i*self.query_batch_size:(i+1)*self.query_batch_size, 0]}
                    out, _ = typed_read(self.readout, self.attention, bound, prefix["hidden"], prefix)
                    if self.context_features and "rule" in context:
                        si = min(sample_offset+i, len(context["rule"])-1)
                        out["state_logits"] = out["state_logits"][:count] + torch.logit(context["state_prior"][si, start:start+count].to(vector.device))
                        out["transition_logits"] = out["transition_logits"][:count] + context["mass_prior"][si, start:start+count].to(vector.device).log()
                    sample_outputs[i].append({key: None if value is None else value[:count]
                                              for key, value in out.items() if key != "event_types"})
            for prefix, outputs in zip(prefixes, sample_outputs):
                if prefix["padding"]:
                    continue
                per_sample.append({**{key: None if outputs[0][key] is None else torch.cat([o[key] for o in outputs])
                                      for key in outputs[0]}, "event_types": self.readout.heads.event_types})
            sample_offset += sum(not p["padding"] for p in prefixes)
        return aggregate_outputs(per_sample)

    def forward(self, index, rows, feedbacks=None, *, structures=None):
        prepared = self.prefill_samples(index, self.structures(index, feedbacks) if structures is None else structures)
        return self.score_samples(index, rows, prepared)
