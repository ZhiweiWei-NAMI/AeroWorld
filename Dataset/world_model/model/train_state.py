"""Multi-window training and evaluation of future state distributions.

The frozen Qwen reads the existing chronological prefix (typed graph slots and
RGB/depth/seg/LiDAR capture tokens).  The shared address-query attention and
typed distribution head predict one law per future (entity, field) cell for
every window, whatever its entity and query count.  Gaussian fields are fitted
by NLL against the actual future value; other fields keep their typed laws.
Modality decoding is outside this model.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from Dataset.world_model.data.observations import (load_observation_frame, observation_token_bindings,
                                                   source_entity_association)
from . import multimodal as mm
from .causal_input import (ATOM_SEMANTICS, STATUS, TypedGraphEncoder, cell, dumps, fit_observation_scales,
                           fit_state_normalization, labels, load_index, query_tensors, restrict_query_fields)
from .distributions import (GAUSSIAN_SCALE_FLOOR, SharedDistributionHead, anchor_mixture_summary, band_mass, cell_nll, law,
                            mixture_crps, predictions, typed_loss)
from .query_head import AddressQueryAttention, FeatureScale, fit_feature_scale
from .run_episode import (anchor_output, chronological_prefix, configure_qwen_checkpointing, device_tensors,
                          floats, modality_inputs, normalize_features, typed_read)

CAPTURE_STEP = 5


@dataclass
class Window:
    role: str
    episode: str
    observer: str
    cutoff: int
    history_ticks: tuple[int, ...]
    target_ticks: tuple[int, ...]
    index: object = None
    queries: dict = None
    truths: dict = None


def window_ticks(cutoff, history_s, tick_seconds, horizons):
    """Capture ticks inside the last ``history_s`` seconds, and the target ticks."""
    left = cutoff-round(history_s/tick_seconds)
    history = tuple(t for t in range(left, cutoff+1) if t > 0 and t % CAPTURE_STEP == 0)
    if not history or history[-1] != cutoff:
        raise ValueError(f"cutoff {cutoff} is not a capture tick with history inside [{left}, {cutoff}]")
    return history, tuple(cutoff+h for h in horizons)


def build_window(spec, episode, role, cutoff, history_s, tick_seconds, horizons, capture_history_s=None, keyframe_s=None):
    """Build one window's causal index (runs in a worker process).

    The state graph covers ``history_s``; capture tokens cover the last
    ``capture_history_s`` (default: the whole history).  ``keyframe_s`` turns
    on history compression (causal_input.compress_history).  The camera
    position is the observer UAV's pose, so capture frames are not read here;
    capture features come from the feature cache.
    """
    history, targets = window_ticks(cutoff, history_s if capture_history_s is None else capture_history_s, tick_seconds, horizons)
    index = load_index(Path(spec["canonical"]), Path(spec["communication"]), episode, cutoff, targets,
                       Path(spec["graph_base"]), history_s=history_s, observer_ids=(spec["observer"],), keyframe_s=keyframe_s,
                       sumo_net_xml=Path(spec["sumo_net_xml"]) if spec.get("sumo_net_xml") else None)
    # Per-record source addresses (raw canonical fields included) are export
    # provenance (run_episode.py), never read by training; hundreds of windows
    # would otherwise hold them in RAM.  Windows stay identified by episode/cutoff.
    index.source_bindings = []
    return Window(role, episode, spec["observer"], cutoff, history, targets, index)


def load_windows(config, tick_seconds, workers):
    jobs = []
    for role in ("train", "eval"):
        for group in config[role+"_windows"]:
            spec = config["episodes"][group["episode"]]
            for cutoff in group["cutoffs"]:
                jobs.append((spec, group["episode"], role, cutoff, config["history_s"], tick_seconds, tuple(config["horizon_ticks"]),
                             config.get("capture_history_s"), config.get("keyframe_s")))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(build_window, *zip(*jobs)))


class FeatureCache:
    """Frozen-encoder capture features, one file per (episode, observer, tick).

    Features depend on the Qwen vision tower, the codec teachers and the seeded
    UE class embedding; ``signature`` records those inputs and a mismatch is an
    error, never a silent reuse.
    """
    def __init__(self, root, signature):
        self.root, self.signature = Path(root), signature

    def path(self, episode, observer, tick):
        return self.root/episode/observer/f"tick_{tick:06d}.pt"

    def load(self, episode, observer, tick, device):
        payload = torch.load(self.path(episode, observer, tick), map_location=device, weights_only=False)
        if payload["signature"] != self.signature:
            raise ValueError(f"feature cache {self.path(episode, observer, tick)} was built with {payload['signature']}")
        return mm.SensorFeatures(**payload["features"])

    def save(self, episode, observer, tick, features):
        path = self.path(episode, observer, tick)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {name: getattr(features, name) for name in mm.SensorFeatures.__dataclass_fields__}
        values = {name: value.cpu() if isinstance(value, torch.Tensor) else value for name, value in values.items()}
        # Per-process name: concurrent runs may fill the same shared cache.
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        torch.save({"signature": self.signature, "features": values}, temporary)
        os.replace(temporary, path)


def prepare_features(config, windows, args, qwen, device):
    """Prepared capture features for every history tick, through the cache."""
    needed = sorted({(w.episode, w.observer, t) for w in windows for t in w.history_ticks})
    frames0 = {episode: load_observation_frame(spec["observations"], episode, min(t for e, _, t in needed if e == episode),
                                               spec["observer"], branch="historical", cutoff=10**9)
               for episode, spec in config["episodes"].items() if any(e == episode for e, _, _ in needed)}
    layouts = {episode: (frame.binding["lidar"], frame.binding["modality"]["lidar"]["sensor_id"]) for episode, frame in frames0.items()}
    settings = None
    for episode, layout in layouts.items():
        current = mm.configure_sensor_layout(*layout)
        if settings is not None and current != settings:
            raise ValueError(f"LiDAR layout of {episode} differs from the other episodes")
        settings = current
    ue_classes = sorted({int(c) for frame in frames0.values() for c in frame.binding["segmentation"]["class_by_id"]})
    signature = {"qwen": str(args.model), "codecs": str(args.codecs), "seed": args.feature_seed, "ue_class_ids": ue_classes}
    cache = FeatureCache(args.feature_cache, signature)
    missing = [key for key in needed if not cache.path(*key).exists()]
    if missing:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True, min_pixels=256*448, max_pixels=256*448)
        codecs = mm.SpatialCodecs(args.codecs, device)
        # The teacher UE embedding is randomly initialised at registration; the
        # seed fixes it so cached teacher features are reproducible.
        torch.manual_seed(args.feature_seed)
        codecs.register_ue_classes({str(c): None for c in ue_classes})
        started = time.perf_counter()
        for number, (episode, observer, tick) in enumerate(missing, 1):
            spec = config["episodes"][episode]
            frame = load_observation_frame(spec["observations"], episode, tick, observer, branch="historical", cutoff=tick)
            with torch.no_grad():
                cache.save(episode, observer, tick, mm.prepare_frame(frame, qwen, processor, device, codecs, settings))
            print(dumps({"phase": "feature_cache", "done": number, "missing": len(missing),
                         "elapsed_seconds": time.perf_counter()-started}), flush=True)
        del codecs
        torch.cuda.empty_cache()
    prepared, bindings = {}, {}
    for episode, observer, tick in needed:
        spec = config["episodes"][episode]
        frame = load_observation_frame(spec["observations"], episode, tick, observer, branch="historical", cutoff=tick)
        # Held on CPU; each window moves only its own captures to the GPU.
        prepared[episode, tick] = cache.load(episode, observer, tick, "cpu")
        bindings[episode, tick] = observation_token_bindings(frame, source_entity_association(frame))
    return prepared, bindings, settings, signature, len(missing)


def to_device(features, device):
    values = {name: getattr(features, name) for name in mm.SensorFeatures.__dataclass_fields__}
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value
    return mm.SensorFeatures(**{name: move(value) for name, value in values.items()})


def unbounded_record(row, address, f, mass, distribution, query, truth, i, window, tick):
    """Per-cell inputs of ``score_unbounded`` for one unbounded field; gate kept as log probabilities."""
    return {"group": f"{address.field_family}[{f.component}]@{tick-window.cutoff}", "episode": window.episode,
            "category": window.index.entities[address.entity_id]["roster.entity_category"],
            "atom": bool(truth["atom"][i]), "truth": float(truth["numeric"][i]), "anchor": float(query["anchor"][i]),
            "band": float(query["atom_band"][i]), "unit": float(query["forecast_std"][i]),
            "log_p": float(mass[0]), "log_off": float(mass[1]), "mu": float(distribution.loc), "sigma": float(distribution.scale)}


def score_unbounded(records):
    """Decomposed NLL and point/interval statistics for unbounded-field records.

    full = gate BCE + 1[off] x off-anchor Gaussian NLL (normalized over the band
    complement); moving = off-anchor gate NLL + off-anchor Gaussian NLL.  The
    untrained head (P(atom)=0.5, mean=anchor, std=(1+floor) x unit) is the
    reference law.  MAE uses the mixture median, RMSE the mixture mean.
    """
    def column(key, dtype=torch.float64):
        return torch.tensor([r[key] for r in records], dtype=dtype)
    atom, y, a, band, unit = column("atom", torch.bool), column("truth"), column("anchor"), column("band"), column("unit")
    def terms(log_p, log_off, mu, sigma):
        gaussian = Normal(mu, sigma)
        _, log_complement, _, _ = band_mass(gaussian, a, band)
        gate = torch.where(atom, -log_p, -log_off)
        continuous = -gaussian.log_prob(y)+log_complement
        p = log_p.exp()
        summary = anchor_mixture_summary(a, p, log_off.exp(), gaussian, band)
        q = summary["quantiles"]
        def covered(lo, hi):
            return torch.where(atom, (lo <= a) & (a <= hi), (lo <= y) & (y <= hi)).double()
        z = ((y-mu)/sigma).abs()
        crps = mixture_crps(a, p, mu, sigma, torch.where(atom, a, y))
        return {"gate": gate, "gaussian": continuous, "full": gate+torch.where(atom, 0., continuous), "crps": crps,
                "median_abs_error": (q[.5]-y).abs(), "mean_squared_error": (summary["mean"]-y)**2,
                "covered_50": covered(q[.25], q[.75]), "covered_90": covered(q[.05], q[.95]),
                "off_gaussian_covered_90": (z <= 1.6448536269514722).double(), "p": p}
    model = terms(column("log_p"), column("log_off"), column("mu"), column("sigma"))
    half = torch.full_like(a, math.log(.5))
    reference = terms(half, half, a, unit*(1+GAUSSIAN_SCALE_FLOOR))
    off = ~atom
    def mean(values, mask=None):
        values = values if mask is None else values[mask]
        return float(values.mean()) if len(values) else None
    def block(mask):
        n = int(mask.sum())
        if not n:
            return {"cells": 0}
        off_mask = mask & off
        out = {"cells": n, "off_anchor_cells": int(off_mask.sum()), "off_anchor_fraction": int(off_mask.sum())/n}
        for name, t in (("model", model), ("reference", reference)):
            out[name] = {"full_nll": mean(t["full"], mask), "gate_bce_all": mean(t["gate"], mask),
                         "off_anchor_gate_nll": mean(t["gate"], off_mask), "off_anchor_gaussian_nll": mean(t["gaussian"], off_mask),
                         "moving_nll": mean(t["full"], off_mask), "mae_mixture_median": mean(t["median_abs_error"], mask),
                         "rmse_mixture_mean": math.sqrt(mean(t["mean_squared_error"], mask)),
                         "coverage_50": mean(t["covered_50"], mask), "coverage_90": mean(t["covered_90"], mask),
                         "off_anchor_gaussian_coverage_90": mean(t["off_gaussian_covered_90"], off_mask),
                         "atom_brier": mean((t["p"]-atom.double())**2, mask), "crps": mean(t["crps"], mask)}
        # The deterministic anchor's CRPS is its absolute error (atoms scored at the anchor).
        anchor_crps = mean((a-torch.where(atom, a, y)).abs(), mask)
        out["anchor_point"] = {"mae": mean((a-y).abs(), mask), "rmse": math.sqrt(mean((a-y)**2, mask)), "crps": anchor_crps}
        out["crps_skill_vs_anchor"] = 1-out["model"]["crps"]/anchor_crps if anchor_crps else None
        return out
    groups = sorted({r["group"] for r in records})
    episodes = sorted({r["episode"] for r in records})
    group_index = [r["group"] for r in records]
    episode_index = [r["episode"] for r in records]
    return {"all": block(torch.ones_like(atom)),
            "by_field": {g: block(torch.tensor([x == g for x in group_index])) for g in groups},
            "by_episode": {e: block(torch.tensor([x == e for x in episode_index])) for e in episodes},
            "by_category": {c: block(torch.tensor([r["category"] == c for r in records]))
                            for c in sorted({r["category"] for r in records})}}


def evaluation(windows, compile_prefix, qwen, modules, device):
    """Unbounded fields: decomposed scores (``score_unbounded``); other laws: NLL and point error."""
    groups, rows_out, unbounded = {}, [], []
    def group(name):
        if name not in groups:
            groups[name] = {"nll": [], "anchor_nll": [], "abs_error": [], "anchor_abs_error": [], "persistence_abs_error": [],
                            "correct": [], "anchor_correct": [], "law": None}
        return groups[name]
    with torch.no_grad():
        for w in windows:
            prefix, _ = compile_prefix(w)
            hidden = qwen(inputs_embeds=prefix["embeddings"], position_ids=prefix["positions"], use_cache=False).last_hidden_state
            for tick in w.target_ticks:
                output, _ = typed_read(modules["typed"], modules["query"], w.queries[tick], hidden, prefix)
                # Per-cell laws and metrics run on CPU copies.
                cpu = lambda values: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in values.items()}
                query, truth, output = cpu(w.queries[tick]), cpu(w.truths[tick]), cpu(output)
                baseline = anchor_output(query, "cpu")
                rows, _ = predictions(output, query)
                for i, (address, f) in enumerate(zip(query["addresses"], query["fields"])):
                    row = {**rows[i], "window_cutoff": w.cutoff, "horizon_ticks": tick-w.cutoff, "truth": None}
                    present = bool(truth["status_mask"][i]) and int(truth["status"][i]) == 0
                    if present and f.logical_dtype == "real":
                        value = float(truth["numeric"][i])
                        row["truth"] = value
                        name, model_law, model_mass = law(output["parameters"][i], f, query["anchor"][i], query["forecast_std"][i], query["positive_prior"][i])
                        if name == "anchor_inflated_normal":
                            row["truth_atom"] = bool(truth["atom"][i])
                            unbounded.append(unbounded_record(row, address, f, model_mass, model_law, query, truth, i, w, tick))
                        else:
                            g = group(f"{address.field_family}{'' if f.component is None else f'[{f.component}]'}@{tick-w.cutoff}")
                            _, anchor_law, anchor_mass = law(baseline["parameters"][i], f, query["anchor"][i], query["forecast_std"][i], query["positive_prior"][i])
                            g["law"] = name
                            nll = lambda d, m: float(cell_nll(name, d, m, truth["numeric"][i].double(), f))
                            g["nll"].append(nll(model_law, model_mass)); g["anchor_nll"].append(nll(anchor_law, anchor_mass))
                            difference = (lambda x, y: abs((x-y+180.) % 360.-180.)) if f.circular else (lambda x, y: abs(x-y))
                            g["abs_error"].append(difference(row["conditional_point"], value))
                            g["anchor_abs_error"].append(difference(float(query["anchor"][i]), value))
                            if float(query["features"][i, 2]) > 0:
                                g["persistence_abs_error"].append(difference(latest_value(w.index, address, f), value))
                    elif present:
                        value = f.enum_values[int(truth["categorical"][i])]
                        row["truth"] = value
                        g = group(f"{address.field_family}{'' if f.component is None else f'[{f.component}]'}@{tick-w.cutoff}")
                        g["law"] = "categorical"
                        log_probabilities = output["classes"][i, :len(f.enum_values)].log_softmax(-1)
                        g["nll"].append(-float(log_probabilities[int(truth["categorical"][i])]))
                        g["correct"].append(row["conditional_point"] == value)
                        g["anchor_correct"].append(f.enum_values[int(query["category_prior"][i])] == value)
                    rows_out.append(row)
    metrics = {}
    for name, g in sorted(groups.items()):
        mean = lambda values: float(np.mean(values)) if values else None
        metrics[name] = {"law": g["law"], "cells": len(g["nll"]), "nll": mean(g["nll"])}
        if g["law"] == "categorical":
            metrics[name].update({"accuracy": mean(g["correct"]), "prior_accuracy": mean(g["anchor_correct"])})
        else:
            metrics[name].update({"anchor_nll": mean(g["anchor_nll"]), "mae": mean(g["abs_error"]),
                                  "anchor_mae": mean(g["anchor_abs_error"]), "persistence_mae": mean(g["persistence_abs_error"])})
    scores = score_unbounded(unbounded)
    summary = {"atom_semantics": ATOM_SEMANTICS, **{k: v for k, v in scores["all"].items()}, "windows": len(windows)}
    return {"summary": summary, "unbounded": scores, "fields": metrics}, rows_out


def gradient_norms(modules, lora=()):
    """Gradient norm per trainable branch (None when the branch received no gradient)."""
    branches = {"graph": list(modules["graph"].parameters()), "query": list(modules["query"].parameters()),
                "typed": list(modules["typed"].parameters()),
                **{f"projectors.{name}": list(child.parameters()) for name, child in modules["projectors"].named_children()}}
    if lora:
        branches["qwen_lora"] = [p for _, p in lora]
    result = {}
    for name, parameters in branches.items():
        grads = [p.grad.detach().float() for p in parameters if p.requires_grad and p.grad is not None]
        result[name] = float(torch.sqrt(sum(g.square().sum() for g in grads))) if grads else None
    return result


# Language-model projections of Qwen3.5: full attention (q/k/v/o), gated delta
# net (in_proj_qkv, in_proj_z, out_proj) and MLP.  Vision tower names differ.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj")


def lora_settings(args):
    return {"rank": args.lora_rank, "alpha": args.lora_alpha, "learning_rate": args.lora_lr,
            "targets": list(LORA_TARGETS) if args.lora_rank else [], "adapter_dtype": "float32"}


def attach_lora(qwen, args, checkpoint):
    """Inject LoRA adapters into the frozen Qwen language model; returns (name, parameter) pairs.

    B starts at zero, so the adapted model initially equals the frozen one.  A
    dedicated seed keeps head initialization and window order identical to a
    run without LoRA.  Adapters are float32 (PEFT casts inputs and results).
    """
    if not args.lora_rank:
        return []
    from peft import LoraConfig, inject_adapter_in_model
    torch.manual_seed(args.seed+1)
    inject_adapter_in_model(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0., bias="none",
                                       target_modules=list(LORA_TARGETS)), qwen.language_model)
    lora = [(name, p) for name, p in qwen.named_parameters() if "lora_" in name]
    if not lora or any(not p.requires_grad for _, p in lora) or \
            any(p.requires_grad for name, p in qwen.named_parameters() if "lora_" not in name):
        raise RuntimeError("LoRA injection must leave exactly the adapters trainable")
    for _, p in lora:
        p.data = p.data.float()
    if checkpoint is not None:
        saved = checkpoint["lora_parameters"]
        if set(saved) != {name for name, _ in lora}:
            raise ValueError("checkpoint LoRA adapters differ from the configured injection")
        for name, p in lora:
            p.data.copy_(saved[name])
    return lora


def source_provenance(config, windows, args):
    """Source commit, tracked and untracked working-tree state, config and split identity."""
    import hashlib
    import subprocess
    from Dataset.world_model.data.provenance import SOURCE_SPLITS
    root = Path(__file__).resolve().parents[3]
    def git(*command):
        return subprocess.run(["git", "-C", str(root), *command], capture_output=True, text=True, check=True).stdout
    return {"source_root": str(root), "commit": git("rev-parse", "HEAD").strip(),
            "tracked_changes": git("status", "--porcelain", "--untracked-files=no").splitlines(),
            "untracked_model_sources": git("ls-files", "--others", "--exclude-standard", "Dataset/world_model").splitlines(),
            "config_path": str(args.config), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "atom_semantics": ATOM_SEMANTICS, "temporal_input_version": "sumo_incremental_v1",
            "model_source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob("*.py"))},
            "static_maps": sorted({(w.index.temporal_context.metadata["net_xml"],
                                    w.index.temporal_context.metadata["map_version"]) for w in windows
                                   if w.index.temporal_context is not None}),
            "splits": {episode: SOURCE_SPLITS[episode] for episode in sorted({w.episode for w in windows})},
            "split_table_size": len(SOURCE_SPLITS)}


def latest_value(index, address, f):
    present = sorted((t, cell(v[f.family], f.component)) for (t, e), v in index.states.items()
                     if e == address.entity_id and t <= index.cutoff and f.family in v)
    return float([v for _, (kind, v) in present if kind == "present"][-1])


def run(args):
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    from .causal_input import read_tick_seconds
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        clocks = set(pool.map(read_tick_seconds, [Path(spec["canonical"]) for spec in config["episodes"].values()]))
    if len(clocks) != 1:
        raise ValueError("episodes declare different tick clocks")
    tick_seconds = clocks.pop()
    windows = load_windows(config, tick_seconds, args.workers)
    train = [w for w in windows if w.role == "train"]
    heldout = [w for w in windows if w.role == "eval"]
    for w in windows:
        expected = "TRAIN" if w.role == "train" else args.eval_split
        if w.index.split != expected:
            raise ValueError(f"{w.role} window {w.episode}@{w.cutoff} is {w.index.split}, expected {expected}")
    provenance = source_provenance(config, windows, args)
    print(dumps({"phase": "windows_loaded", "train": len(train), "eval": len(heldout),
                 "elapsed_seconds": time.perf_counter()-started}), flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if checkpoint is None:
        scales, forecast_scales, constant = fit_state_normalization([w.index for w in train], unit_grouping=args.unit_grouping)
        observation_scales = fit_observation_scales([w.index for w in train])
    else:
        scales, forecast_scales = checkpoint["state_scales"], checkpoint["forecast_scales"]
        constant, observation_scales = checkpoint["constant_targets"], checkpoint["graph_observation_scales"]
    for w in windows:
        w.index.scales, w.index.forecast_scales = scales, forecast_scales
        restrict_query_fields(w.index, constant)
        w.queries = {t: device_tensors(query_tensors(w.index, t), device) for t in w.target_ticks}
        if args.no_history_atoms:
            for query in w.queries.values():
                query["features"][:, -3:] = 0.
        w.truths = {t: device_tensors(labels(w.index, w.queries[t]), device) for t in w.target_ticks}
    from transformers import Qwen3_5Model
    qwen = Qwen3_5Model.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16,
                                        attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    checkpointed_layers = configure_qwen_checkpointing(qwen)
    if args.no_capture_tokens:
        # Capture caches, Sonata and feature scales are unused without capture tokens.
        prepared, token_bindings, settings, cache_signature, encoded = {}, {}, None, None, 0
        sonata, sonata_config = None, None
    else:
        prepared, token_bindings, settings, cache_signature, encoded = prepare_features(config, windows, args, qwen, device)
    # Capture features are cached; the prefix only needs the language model.
    qwen.visual = None
    if cuda:
        torch.cuda.empty_cache()
    print(dumps({"phase": "features_ready", "captures": len(prepared), "newly_encoded": encoded,
                 "elapsed_seconds": time.perf_counter()-started}), flush=True)
    if not args.no_capture_tokens:
        sonata, sonata_config = mm.load_sonata(args.sonata, device)
    # Module initialization starts from the same seed state in both arms.
    torch.manual_seed(args.seed)
    modules = nn.ModuleDict({"graph": TypedGraphEncoder(observation_scales, mode=args.input_mode),
                             "query": AddressQueryAttention(), "typed": SharedDistributionHead(),
                             "projectors": mm.ModalityProjectors()}).to(device)
    if args.no_capture_tokens:
        feature_scales = {}
        if checkpoint is not None:
            modules.load_state_dict(checkpoint["trained_parameters"], strict=True)
    elif checkpoint is None:
        # Feature scales see LiDAR through the initial adapter, as in the single-window run.
        teacher_geometry = copy.deepcopy(modules["projectors"].geometry_adapter).requires_grad_(False).eval()
        with torch.no_grad():
            train_ticks = sorted({(w.episode, t) for w in train for t in w.history_ticks})
            feature_scales = {"rgb": fit_feature_scale(torch.cat([prepared[key].rgb.to(device) for key in train_ticks]).float()),
                              "lidar": fit_feature_scale(torch.cat([mm.sonata_features(to_device(prepared[key], device), sonata, teacher_geometry)[0]
                                                                    for key in train_ticks]).float())}
    else:
        modules.load_state_dict(checkpoint["trained_parameters"], strict=True)
        feature_scales = {name: FeatureScale(value["mean"].to(device), value["std"].to(device))
                          for name, value in checkpoint["feature_scales"].items()}
    lora = attach_lora(qwen, args, checkpoint)
    print(dumps({"phase": "model_ready", "elapsed_seconds": time.perf_counter()-started,
                 "gpu_allocated_bytes": torch.cuda.memory_allocated() if cuda else 0}), flush=True)
    if not args.no_capture_tokens:
        shape = next(iter(prepared.values())).image_grid_thw[0].tolist()
        layouts = {"rgb": mm.image_grid(shape[1]//2, shape[2]//2).to(device),
                   "depth": mm.image_grid(mm.CODEC_ROWS, mm.CODEC_COLS).to(device), "seg": mm.image_grid(mm.CODEC_ROWS, mm.CODEC_COLS).to(device),
                   "lidar": mm.angular_grid(mm.COARSE_EL, mm.COARSE_AZ).to(device)}

    def compile_prefix(w):
        """Trainable prefix of one window: LiDAR adapter -> Sonata, projections, graph slots."""
        if args.no_capture_tokens:
            structure = modules["graph"].encode_history(w.index, None, reconstruct=args.reconstruction_weight > 0)
            return chronological_prefix(qwen, structure, {}), structure
        frames = {t: to_device(prepared[w.episode, t], device) for t in w.history_ticks}
        features = {}
        for t, frame in frames.items():
            lidar, _ = mm.sonata_features(frame, sonata, modules["projectors"].geometry_adapter)
            features[t] = normalize_features({"rgb": frame.rgb, "depth": frame.depth, "seg": frame.seg, "lidar": lidar}, feature_scales)
        contexts, blocks = modality_inputs(modules["projectors"], features, frames, w.history_ticks, layouts, w.index,
                                           feature_scales, {t: token_bindings[w.episode, t] for t in w.history_ticks})
        structure = modules["graph"].encode_history(w.index, contexts, reconstruct=args.reconstruction_weight > 0)
        prefix = chronological_prefix(qwen, structure, blocks)
        if prefix["embeddings"].shape[1] > qwen.config.text_config.max_position_embeddings:
            raise ValueError("multimodal prefix exceeds native Qwen context")
        return prefix, structure

    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    history, evaluations = [], []

    def save_checkpoint(step):
        payload = {"trained_parameters": {name: p.detach().cpu() for name, p in modules.named_parameters() if p.requires_grad},
                   "state_scales": scales, "forecast_scales": forecast_scales, "constant_targets": constant,
                   "graph_observation_scales": observation_scales,
                   "feature_scales": {name: {"mean": s.mean.cpu(), "std": s.std.cpu()} for name, s in feature_scales.items()},
                   "lora_parameters": {name: p.detach().cpu() for name, p in lora}, "lora": lora_settings(args),
                   "completed_steps": step, "source_config": run_config, "provenance": provenance}
        torch.save(payload, args.output/"head.pt.tmp"); os.replace(args.output/"head.pt.tmp", args.output/"head.pt")

    def evaluate(step):
        if args.steps:
            save_checkpoint(step)
        qwen.eval()
        result, rows = evaluation(heldout, compile_prefix, qwen, modules, device)
        result["step"] = step
        evaluations.append(result)
        print(dumps({"phase": "evaluation", "step": step, **result["summary"]}), flush=True)
        return rows

    head_parameters = [p for p in modules.parameters() if p.requires_grad]
    lora_parameters = [p for _, p in lora]
    active = head_parameters+lora_parameters
    optimizer = torch.optim.AdamW([{"params": head_parameters, "lr": args.lr}]
                                  +([{"params": lora_parameters, "lr": args.lora_lr}] if lora_parameters else []))
    order = random.Random(args.seed)
    queue = []
    rows = None
    for step in range(args.steps):
        modules.train(); qwen.train()
        optimizer.zero_grad(set_to_none=True)
        totals, timing = {}, {"prefix_seconds": 0., "qwen_forward_seconds": 0., "backward_seconds": 0.}
        def lap(name, since):
            if cuda:
                torch.cuda.synchronize()
            now = time.perf_counter()
            timing[name] += now-since
            return now
        for _ in range(args.accumulate):
            if not queue:
                queue = list(range(len(train))); order.shuffle(queue)
            w = train[queue.pop()]
            mark = time.perf_counter()
            prefix, structure = compile_prefix(w)
            mark = lap("prefix_seconds", mark)
            hidden = qwen(inputs_embeds=prefix["embeddings"], position_ids=prefix["positions"], use_cache=False).last_hidden_state
            mark = lap("qwen_forward_seconds", mark)
            losses = [typed_loss(typed_read(modules["typed"], modules["query"], w.queries[t], hidden, prefix)[0], w.queries[t], w.truths[t],
                                 crps_weight=args.crps_weight)
                      for t in w.target_ticks]
            terms = {name: torch.stack([loss[name] for loss in losses]).mean() for name in losses[0]}
            if args.reconstruction_weight > 0:
                terms.update({"reconstruct_"+name: value for name, value in structure["reconstruction_losses"].items() if name != "total"})
                terms["graph_reconstruction"] = structure["reconstruction_losses"]["total"]
            total = terms["typed"]+args.reconstruction_weight*terms.get("graph_reconstruction", 0.)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"non-finite objective on window {w.episode}@{w.cutoff}")
            (total/args.accumulate).backward()
            lap("backward_seconds", mark)
            for name, value in {**terms, "total": total}.items():
                totals[name] = totals.get(name, 0.)+float(value)/args.accumulate
        branch_gradients = gradient_norms(modules, lora)
        norm = nn.utils.clip_grad_norm_(active, args.grad_clip)
        optimizer.step()
        row = {"step": step+1, "window": f"{w.episode}@{w.cutoff}", **totals, **timing, "prefix_tokens": prefix["embeddings"].shape[1],
               "gradient_norm_before_clip": float(norm), "branch_gradient_norms": branch_gradients,
               "elapsed_seconds": time.perf_counter()-started,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0}
        history.append(row)
        print(dumps({"phase": "optimizer", **row}), flush=True)
        if args.eval_every and (step+1) % args.eval_every == 0 and step+1 < args.steps:
            evaluate(step+1)
    rows = evaluate(args.steps)
    with (args.output/"predictions_eval.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, default=str)+"\n")
    normalization = {f"{family}{'' if component is None else f'[{component}]'}":
                     {**({"input_mean": scales[family, component].mean, "input_std": scales[family, component].std}
                         if (family, component) in scales else {}), **value}
                     for (family, component), value in forecast_scales.items()}
    result = {"provenance": provenance, "config": config, "run": run_config, "tick_seconds": tick_seconds,
              "windows": {role: [{"episode": w.episode, "observer": w.observer, "cutoff": w.cutoff,
                                  "history_capture_ticks": list(w.history_ticks), "target_ticks": list(w.target_ticks),
                                  "entities": len(w.index.entities),
                                  "queries": {str(t): len(w.queries[t]["addresses"]) for t in w.target_ticks}}
                                 for w in windows if w.role == role] for role in ("train", "eval")},
              "normalization": normalization,
              "constant_targets": [{"field": family, "component": component,
                                    "reason": "no TRAIN target leaves the anchor atom (or every residual is zero); input only"}
                                   for family, component in constant],
              "feature_cache": {"root": str(args.feature_cache), "signature": cache_signature,
                                                                "newly_encoded_captures": encoded},
              "model": {"qwen_base_frozen": True, "lora": lora_settings(args), "qwen_checkpointed_blocks": checkpointed_layers,
                        "sonata": sonata_config, "trainable_parameters": sum(p.numel() for p in active),
                        "lora_parameters": sum(p.numel() for p in lora_parameters), "lidar_settings": settings,
                        "unbounded_law": f"anchor atom P = sigmoid(head); off-anchor Gaussian mean = anchor + head offset x unit, "
                                         f"std = unit x (softplus(head scale) + {GAUSSIAN_SCALE_FLOOR}); unit = TRAIN off-atom residual rate"
                                         f" x elapsed seconds; atom semantics {ATOM_SEMANTICS} (band scored as one censored outcome,"
                                         " Gaussian normalized over its complement)",
                        "loss": "typed NLL/CE (censored-band atom + Gaussian for unbounded real fields) + recording/status terms"
                                "+ reconstruction_weight x graph history reconstruction"},
              "training": {"steps": args.steps, "accumulate": args.accumulate, "loss_history": history},
              "evaluations": evaluations, "wall_seconds": time.perf_counter()-started,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0}
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False, default=str)+"\n")
    print(dumps({"phase": "complete", "output": str(args.output), "wall_seconds": result["wall_seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("Dataset/world_model/configs/state_distribution.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sonata", type=Path, default=Path("Dataset/world_model/models/sonata"))
    parser.add_argument("--codecs", type=Path, default=Path("Dataset/world_model/models"))
    parser.add_argument("--feature-cache", type=Path, default=Path("Dataset/world_model/runs/state_distribution/feature_cache"))
    parser.add_argument("--checkpoint", type=Path, help="evaluate a saved head.pt; combine with --steps 0")
    parser.add_argument("--input-mode", choices=("graph_full", "graph_summary", "record_ablation"), default="graph_full")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--no-capture-tokens", action="store_true",
                        help="ablation: omit RGB/depth/seg/LiDAR capture tokens and graph modality nodes; graph inputs unchanged")
    parser.add_argument("--no-history-atoms", action="store_true",
                        help="ablation: zero the three history anchor-atom query features (same architecture)")
    parser.add_argument("--reconstruction-weight", type=float, default=0.,
                        help="weight of the graph encoder's auxiliary history reconstruction; 0 skips it")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank on the Qwen language model; 0 keeps Qwen frozen")
    parser.add_argument("--lora-alpha", type=float, default=32.)
    parser.add_argument("--unit-grouping", choices=("pooled", "category"), default="pooled",
                        help="TRAIN residual unit pooled per field component, or per entity category too")
    parser.add_argument("--crps-weight", type=float, default=0.,
                        help="weight of the unit-scaled mixture CRPS added to the NLL objective")
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--eval-split", choices=("VALID", "TEST"), default="VALID",
                        help="declared split of evaluation windows; TEST is reserved for the final evaluation")
    parser.add_argument("--seed", type=int, default=20261008, help="model initialization and window order")
    parser.add_argument("--feature-seed", type=int, default=20261008,
                        help="seed of the cached teacher UE class embedding; part of the feature-cache signature")
    args = parser.parse_args()
    if args.checkpoint is not None and args.steps:
        parser.error("--checkpoint evaluates a saved head; pass --steps 0")
    run(args)


if __name__ == "__main__":
    main()
