"""Typed predicate queries over the existing causal graph and frozen Qwen base.

The five transition outputs are *unconditional bin masses*, as required by the
stage contract. A sixth softmax category is survival beyond tick +50. No text
generation, future state cells, tuple hashes, or episode names enter a query.
"""
from __future__ import annotations

import copy
import inspect
from importlib.metadata import version
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from Dataset.semantic_truth.core_semantic_registry import load_core_semantic_registry
from Dataset.world_model.model.causal_input import TypedGraphEncoder
from Dataset.world_model.model.query_head import AddressQueryAttention, QueryAddress
from Dataset.world_model.model.run_episode import (
    chronological_prefix, scalar_positions, typed_read,
)
from Dataset.world_model.model.train_state import attach_lora, lora_settings

HORIZONS = (5, 10, 20, 30, 50)
TRUTH = ("false", "true", "unknown")
PREDICATES = {
    "safety.aircraft_pair_below_separation": "separation",
    "airspace.airspace_corridor_over_capacity": "corridor",
    "airspace.aircraft_occupies_airspace_corridor": "corridor",
    "communication.handover_in_progress": "communication",
    "communication.data_flow_latency_above_threshold": "communication",
}
EXCLUDED_PREDICATES = frozenset({"communication.link_degraded", "communication.link_unavailable"})
DEFAULT_QWEN = "/mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35/hf/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def configure_qwen_kernels(device):
    """Use differentiable CUDA kernels for training and recurrent kernels for evaluation.

    FLA's fused recurrent kernel is inference-only. A one-token training query
    must use its chunk kernel so gradients also reach the copied prefix state.
    CPU execution explicitly uses the native Torch equations.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn")
    if not hasattr(hf, "_aeroworld_native_kernels"):
        hf._aeroworld_native_kernels = {name: inspect.unwrap(getattr(hf, name)) for name in names}
    native = hf._aeroworld_native_kernels
    if torch.device(device).type == "cpu":
        for name, function in native.items():
            setattr(hf, name, function)
        return {"linear_attention": "torch_cpu", "convolution": "torch_cpu"}
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from causal_conv1d import causal_conv1d_fn

    def dispatch(reference, training, inference):
        parameters = {fn: set(inspect.signature(fn).parameters) for fn in (reference, training, inference)}
        def call(*args, **kwargs):
            fn = reference if args[0].device.type == "cpu" else training if torch.is_grad_enabled() else inference
            return fn(*args, **{k: v for k, v in kwargs.items() if k in parameters[fn]})
        return call
    hf.torch_chunk_gated_delta_rule = dispatch(native[names[0]], chunk_gated_delta_rule, chunk_gated_delta_rule)
    hf.torch_recurrent_gated_delta_rule = dispatch(native[names[1]], chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
    hf.causal_conv1d_fn = dispatch(native[names[2]], causal_conv1d_fn, causal_conv1d_fn)
    return {"linear_attention": "fla", "fla_version": version("flash-linear-attention"),
            "training_decode": "differentiable_chunk", "inference_decode": "fused_recurrent",
            "convolution": "causal_conv1d", "causal_conv1d_version": version("causal-conv1d")}


class PredicateQueryEncoder(nn.Module):
    """Template + role/identity pairs + requested horizons; shared across tuples."""

    def __init__(self, width=128):
        super().__init__()
        templates = load_core_semantic_registry()["templates"]
        self.pred_ids = {name: i for i, name in enumerate(sorted(PREDICATES))}
        self.roles = {t["id"]: tuple(r["key"] for r in t["argument_roles"])
                      for t in templates if t["id"] in PREDICATES}
        if set(self.roles) != set(PREDICATES):
            raise ValueError("predicate contract differs from the semantic registry")
        self.role_ids = {name: i for i, name in enumerate(sorted({r for rs in self.roles.values() for r in rs}))}
        self.template = nn.Embedding(len(self.pred_ids), width)
        self.role = nn.Embedding(len(self.role_ids), width)
        self.bound_role = nn.Sequential(nn.Linear(width + 32, width), nn.GELU(), nn.Linear(width, width))
        self.horizon = nn.Linear(len(HORIZONS), width)
        self.norm = nn.LayerNorm(width)

    def ordered_bindings(self, row):
        roles = self.roles[row["predicate_id"]]
        if set(roles) != set(row["bindings"]):
            raise ValueError(f"role mismatch for {row['predicate_id']}: {row['bindings']}")
        return [(role, row["bindings"][role]) for role in roles]

    def forward(self, rows, index):
        device = self.template.weight.device
        vectors, owners, ordinals, role_vectors, addresses = [], [], [], [], []
        for n, row in enumerate(rows):
            parts = []
            for role, identity in self.ordered_bindings(row):
                ordinal = index.local_identities[identity]
                phase = (ordinal + 1) * torch.exp(-torch.arange(16, device=device) * math.log(10000) / 16)
                identity_vector = torch.cat((phase.sin(), phase.cos()))
                role_vector = self.role.weight[self.role_ids[role]]
                part = self.bound_role(torch.cat((role_vector, identity_vector)))
                parts.append(part)
                owners.append(n)
                ordinals.append(ordinal)
                role_vectors.append(part)
                addresses.append(QueryAddress(index.episode, identity, "relation."+row["predicate_id"], index.cutoff))
            horizons = torch.tensor(row["horizons"], device=device, dtype=torch.float32) / 50
            vectors.append(self.template.weight[self.pred_ids[row["predicate_id"]]]
                           + torch.stack(parts).mean(0) + self.horizon(horizons))
        return self.norm(torch.stack(vectors)), {
            "owners": torch.tensor(owners, device=device),
            "entity_ordinals": torch.tensor(ordinals, device=device),
            "addresses": addresses,
            "role_vectors": torch.stack(role_vectors),
        }


class SharedProbabilityHeads(nn.Module):
    """One state head per horizon and one normalized transition-time law."""

    def __init__(self, width, event_types=()):
        super().__init__()
        self.event_types = tuple(event_types)
        self.state = nn.Linear(width, len(HORIZONS))
        self.transition = nn.Linear(width, len(HORIZONS) + 1)
        self.event = nn.Linear(width, len(event_types)) if event_types else None

    def forward(self, hidden):
        return {"state_logits": self.state(hidden), "transition_logits": self.transition(hidden),
                "event_logits": self.event(hidden) if self.event is not None else None}


def probability_loss(output, rows):
    """Known-state BCE and interval-censored first-transition log likelihood.

    Unknown truth is not a negative. Partial censoring includes the unresolved
    time bin and all later bins; it never asserts survival through tick +50.
    """
    logits = output["state_logits"].float()
    locations, targets = [], []
    for i, row in enumerate(rows):
        for j, h in enumerate(HORIZONS):
            value = row["future_state"][str(h)]
            if value in ("false", "true"):
                locations.append((i, j))
                targets.append(float(value == "true"))
            elif value != "unknown":
                raise ValueError(f"undeclared future truth {value}")
    components = {}
    if locations:
        i, j = zip(*locations)
        components["state"] = F.binary_cross_entropy_with_logits(
            logits[list(i), list(j)], logits.new_tensor(targets))
    log_mass = output["transition_logits"].float().log_softmax(-1)
    transition_losses = []
    for i, row in enumerate(rows):
        if row["state_at_cutoff"] == "unknown":
            continue
        tick = row["first_transition_tick"]
        if tick is not None:
            elapsed = tick - row["cutoff"]
            if not 0 < elapsed <= 50:
                raise ValueError("first transition outside the prediction window")
            bin_id = next(j for j, h in enumerate(HORIZONS) if elapsed <= h)
            transition_losses.append(-log_mass[i, bin_id])
        else:
            observed = min(50, row["transition_censored_at"] - row["cutoff"])
            if observed < HORIZONS[0]:
                continue
            possible = [j for j, h in enumerate(HORIZONS) if h > observed] + [len(HORIZONS)]
            transition_losses.append(-torch.logsumexp(log_mass[i, possible], dim=0))
    if transition_losses:
        components["transition"] = torch.stack(transition_losses).mean()
    if output["event_logits"] is not None:
        event_types = output["event_types"]
        y, known = [], []
        for row in rows:
            present = {e["event_type_id"] for e in row["event_targets"]}
            end = row["source"].get("label_window_end_tick")
            complete = end is not None and end >= row["cutoff"] + 50
            y.append([float(event in present) for event in event_types])
            # An observed event is a known positive even in a truncated window.
            # A negative requires coverage through +50, not predicate censoring.
            known.append([event in present or complete for event in event_types])
        mask = torch.tensor(known, device=logits.device, dtype=torch.bool)
        if mask.any():
            element = F.binary_cross_entropy_with_logits(
                output["event_logits"].float(), logits.new_tensor(y), reduction="none")
            components["event"] = element[mask].mean()
    if not components:
        raise ValueError("this batch has no observed supervision for any enabled head")
    return sum(components.values()), components


def probabilities(output):
    state = output["state_logits"].float().sigmoid()
    mass = output["transition_logits"].float().softmax(-1)
    event = None if output["event_logits"] is None else output["event_logits"].float().sigmoid()
    return state, mass, event


def pool_prefix(structure, max_tokens):
    """Bound Qwen cost by pooling consecutive causal graph slots, not dropping history.

    A pooled slot is dated at its latest constituent and retains the union of
    exact entity/field registrations for the existing address attention.
    """
    count = structure["embeddings"].shape[1]
    if max_tokens <= 0 or count <= max_tokens:
        return structure
    order = torch.argsort(structure["times"], stable=True)
    chunks = torch.tensor_split(order, max_tokens)
    ids = [chunk.cpu().tolist() for chunk in chunks]
    return {**structure,
            "embeddings": torch.stack([structure["embeddings"][:, chunk].mean(1) for chunk in chunks], dim=1),
            "times": torch.stack([structure["times"][chunk].max() for chunk in chunks]),
            "token_entities": [sorted({e for i in chunk for e in structure["token_entities"][i]}) for chunk in ids],
            "token_fields": [sorted({f for i in chunk for f in structure["token_fields"][i]}) for chunk in ids]}


class PredicateReadout(nn.Module):
    def __init__(self, hidden, width, event_types):
        super().__init__()
        self.combine = nn.Sequential(nn.Linear(hidden * 3, width), nn.GELU(), nn.LayerNorm(width))
        self.heads = SharedProbabilityHeads(width, event_types)

    def forward(self, query, entity, field, global_context):
        owners = query["owners"]
        count = len(query["query_hidden"])
        denom = torch.bincount(owners, minlength=count).to(entity.dtype)[:, None]
        def reduce(value):
            return value.new_zeros(count, value.shape[-1]).index_add(0, owners, value) / denom
        hidden = self.combine(torch.cat((query["query_hidden"].float(), reduce(entity), reduce(field)), -1))
        output = self.heads(hidden)
        output["event_types"] = self.heads.event_types
        return output


class QwenPredicateModel(nn.Module):
    # A fixed decode shape also removes bf16 GEMM batch-size rounding changes.
    # Outer cohort/microbatch sizes may vary without changing this arithmetic.
    query_batch_size = 8

    def __init__(self, observation_scales, *, model_path=DEFAULT_QWEN, device="cuda:0", width=128,
                 seed=17, lora_lr=1e-4, max_prefix_tokens=192, graph_mode="graph_summary", event_types=()):
        super().__init__()
        from transformers import Qwen3_5Model
        self.kernel_config = configure_qwen_kernels(device)
        self.qwen = Qwen3_5Model.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        # The existing block-checkpoint wrapper replays a *mutated* cache during
        # backward. Do not use it with recurrent prefix caches. Prefix length and
        # query microbatches bound memory; all prefix gradients remain connected.
        self.qwen.requires_grad_(False)
        self.checkpointed_layers = 0
        self.qwen.visual = None  # No capture tokens in this loop.
        args = SimpleNamespace(lora_rank=16, lora_alpha=32., lora_lr=lora_lr, seed=seed)
        self.lora_config = lora_settings(args)
        self.lora_parameters = attach_lora(self.qwen, args, None)
        hidden = self.qwen.config.text_config.hidden_size
        self.graph = TypedGraphEncoder(observation_scales, hidden=hidden, mode=graph_mode)
        self.query = PredicateQueryEncoder(width)
        self.query_token = nn.Linear(width, hidden)
        self.role_features = nn.Linear(width, 15)
        # Match the graph encoder's registered relation.<predicate_id> labels.
        # Predicate order is unchanged, preserving checkpoint embedding rows.
        self.attention = AddressQueryAttention(hidden=hidden, width=width)
        self.attention.field_ids = {"relation."+name: ordinal for name, ordinal in self.query.pred_ids.items()}
        self.attention.field = nn.Embedding(len(self.query.pred_ids), width)
        self.readout = PredicateReadout(hidden, width, event_types)
        self.max_prefix_tokens = max_prefix_tokens
        self.to(device)

    def encode_prefix(self, index):
        if any(r["tick"] > index.cutoff for r in index.semantic_records):
            raise ValueError("future record in causal graph input")
        structure = self.graph.encode_history(index, None, reconstruct=False)
        return chronological_prefix(self.qwen, pool_prefix(structure, self.max_prefix_tokens), {})

    def prefill(self, index, prefix=None):
        """Compute one window prefix, including every full/recurrent cache state.

        Reuse only within an evaluation window or the same training forward:
        an optimizer update invalidates both this cache and the prefix hidden.
        """
        from transformers.cache_utils import DynamicCache, LinearAttentionLayer
        prefix = self.encode_prefix(index) if prefix is None else prefix
        cache = DynamicCache(config=self.qwen.config.text_config)
        for i, layer in enumerate(cache.layers):
            if isinstance(layer, LinearAttentionLayer):
                cache.layers[i] = functional_linear_cache(layer.number_of_states)
        result = self.qwen(inputs_embeds=prefix["embeddings"], position_ids=prefix["positions"],
                           past_key_values=cache, use_cache=True)
        return {**prefix, "cache": result.past_key_values, "hidden": result.last_hidden_state}

    def forward(self, index, rows, prefix=None):
        if not rows:
            raise ValueError("a predicate batch must contain a query")
        prefix = self.prefill(index, prefix) if prefix is None or "cache" not in prefix else prefix
        outputs = []
        for start in range(0, len(rows), self.query_batch_size):
            group = rows[start:start+self.query_batch_size]
            count = len(group)
            # Padding is an independent copy of an existing query, and is
            # discarded BEFORE loss/export. It adds no observed or target data.
            padded = group + [group[-1]] * (self.query_batch_size-count)
            output = self._query_continuations(index, padded, prefix)
            outputs.append({key: None if value is None else value[:count]
                            for key, value in output.items() if key != "event_types"})
        return {**{key: None if outputs[0][key] is None else torch.cat([out[key] for out in outputs])
                   for key in outputs[0]}, "event_types": self.readout.heads.event_types}

    def _query_continuations(self, index, rows, prefix):
        vector, query = self.query(rows, index)
        # Each query occupies its OWN sequence and the SAME next position. No
        # attention, convolution, or recurrent state can cross a batch boundary.
        tokens = self.query_token(vector)[:, None].to(self.qwen.dtype)
        positions = scalar_positions(1, prefix["next_token"], prefix["next_rope"], tokens.device)
        cache = fork_prefix_cache(prefix["cache"], len(rows))
        hidden = self.qwen(inputs_embeds=tokens, position_ids=positions.expand(-1, len(rows), -1),
                           past_key_values=cache, use_cache=True).last_hidden_state
        query["features"] = self.role_features(vector[query["owners"]] + query.pop("role_vectors"))
        query["query_hidden"] = hidden[:, 0]
        output, _ = typed_read(self.readout, self.attention, query, prefix["hidden"], prefix)
        return output

    def trainable_state(self):
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in names}


def functional_linear_cache(number_of_states):
    """The installed HF cache writes in place; training needs functional updates.

    record_past routes single-token convolution through the differentiable
    full-convolution path, instead of the inference-only in-place update kernel.
    The retained context is still bounded to the native convolution kernel size.
    """
    from transformers.cache_utils import LinearAttentionLayer

    class FunctionalLinearCache(LinearAttentionLayer):
        def __init__(self):
            super().__init__(number_of_states=number_of_states)
            self.record_past = True

        def update_conv_state(self, states, state_idx=0, conv_kernel_size=None, **kwargs):
            if not self.is_conv_states_initialized[state_idx]:
                self.lazy_initialization(conv_states=states, state_idx=state_idx,
                                         conv_kernel_size=conv_kernel_size)
            full = (torch.cat((self.conv_states[state_idx], states), -1)
                    if self.has_previous_state[state_idx] else states)
            size = self.conv_kernel_size[state_idx]
            self.conv_states[state_idx] = F.pad(full, (max(0, size-full.shape[-1]), 0))[..., -size:]
            self.has_previous_state[state_idx] = True
            return full

        def update_recurrent_state(self, states, state_idx=0, **kwargs):
            self.recurrent_states[state_idx] = states
            self.is_recurrent_states_initialized[state_idx] = True
            return states

    return FunctionalLinearCache()


def fork_prefix_cache(cache, batch_size):
    """Copy cache metadata AND all tensor families; retain autograd to the prefix.

    HF's batch_repeat_interleave handles K/V but the installed linear-attention
    cache lacks that method. Explicitly include conv/recurrent state dictionaries.
    No continuation may mutate the reusable prefix or another continuation.
    """
    result = copy.copy(cache)
    result.layers = []
    def clone(value):
        if isinstance(value, torch.Tensor):
            if value.ndim and value.shape[0] == 1:
                return value.repeat_interleave(batch_size, dim=0)
            return value.clone()
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        return copy.copy(value)
    for layer in cache.layers:
        copied = copy.copy(layer)
        copied.__dict__ = {key: clone(value) for key, value in vars(layer).items()}
        result.layers.append(copied)
    return result
