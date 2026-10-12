"""Shared dense trajectory forward over exact field/owner/rule incidence.

The optional PyG representation is an assembly artifact, not this neural backbone.
The eleven outcomes are first flip at ticks 5..50 and the tail T>50. UNKNOWN
has a separate channel and never becomes a false observation or a tail label.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .communication_cv_models import TrajectoryLogitHead


@dataclass(frozen=True)
class TrajectoryLayout:
    """Setup-time field types and owner-sharing topology, without entity embeddings.

    ``fields`` are the exact encoded field coordinates the layout was compiled
    over: canonical fields, or their declared per-component expansions. Empty
    only for layouts built directly from a cache vocabulary.
    """
    field_type_keys: tuple[tuple, ...]
    field_type_ids: tuple[int, ...]
    owner_incidence: tuple[tuple[bool, ...], ...]
    fields: tuple = ()

    @property
    def component_fields(self):
        """Encoded field coordinates; canonical fields carry no declared vector."""
        return self.fields

    @classmethod
    def vocabulary_from_field_groups(cls, field_groups):
        """Compile one exact vocabulary for grouped heterogeneous episode batches."""
        keys = []
        for fields in field_groups:
            for key in cls.from_fields(fields).field_type_keys:
                if key not in keys:
                    keys.append(key)
        if not keys:
            raise ValueError("global vocabulary requires declared episode fields")
        return tuple(keys)

    @classmethod
    def from_fields(cls, fields, *, field_type_keys=None):
        keys, ids, owners = list(field_type_keys) if field_type_keys is not None else [], [], []
        for field in fields:
            schema = field.schema
            owner_types = tuple(tuple(entity.ontology_classes) for entity in field.owner)
            key = (schema.field, schema.data_type, schema.unit, schema.source,
                   tuple(schema.categories), owner_types)
            if key not in keys:
                if field_type_keys is not None:
                    raise ValueError("field type is absent from the declared global vocabulary")
                keys.append(key)
            ids.append(keys.index(key))
            owners.append(frozenset((entity.entity_id, tuple(entity.ontology_classes)) for entity in field.owner))
        if not ids:
            raise ValueError("trajectory layout needs declared fields")
        adjacency = tuple(tuple(i == j or bool(left & right) for j, right in enumerate(owners))
                          for i, left in enumerate(owners))
        return cls(tuple(keys), tuple(ids), adjacency, tuple(fields))

    @classmethod
    def from_component_fields(cls, fields, *, field_type_keys=None):
        """Compile the encoded-component layout at setup time.

        Declared vector components are expanded exactly once here, before any
        numerical execution; scalar fields pass through unchanged. Each
        component keeps the original canonical identity, typed owner, roles and
        authority, so owner sharing and predicate incidence extend per encoded
        component and never per a guessed norm or code.
        """
        from .predicate_trajectory_data import expanded_field_coordinates
        return cls.from_fields(expanded_field_coordinates(fields),
                               field_type_keys=field_type_keys)


class SharedPredicateTrajectoryModel(nn.Module):
    """One field encoder and one trajectory head for arbitrary F, P and K.

    Dependency pooling uses exact predicate-to-field incidence. Owner pooling
    shares features among declared typed owners. No predicate identifier selects
    parameters. Metadata is compiled once, before numerical execution. A global vocabulary
    with per-batch field_type_ids and owner incidence lets the same parameters
    process grouped batches with different field/predicate/owner counts.
    """
    def __init__(self, layout: TrajectoryLayout, hidden_dim=32, field_dim=16):
        super().__init__()
        self.model_name = "shared_predicate_trajectory"
        self.layout = layout
        self.register_buffer("field_type_ids", torch.tensor(layout.field_type_ids, dtype=torch.long))
        owner = torch.tensor(layout.owner_incidence, dtype=torch.float64)
        self.register_buffer("owner_pool", owner / owner.sum(-1, keepdim=True))
        self.field_embedding = nn.Embedding(len(layout.field_type_keys), field_dim)
        self.field_encoder = nn.Sequential(nn.Linear(field_dim + 2, field_dim), nn.Tanh())
        self.owner_encoder = nn.Sequential(nn.Linear(2 * field_dim, field_dim), nn.Tanh())
        self.trajectory_head = TrajectoryLogitHead(tuple(f"shared_field_{i}" for i in range(field_dim)), hidden_dim)
        # Extend the existing shared branch GRU with a distinct rule-known channel.
        self.trajectory_head.branch_encoder = nn.GRU(field_dim + 2, hidden_dim, batch_first=True)

    def _fields(self, values, known, owner_field_incidence=None, field_type_ids=None):
        if known.dtype != torch.bool or known.shape != values.shape:
            raise ValueError("each field value requires a Boolean observation mask")
        ids = self.field_type_ids if field_type_ids is None else field_type_ids
        if ids.dtype != torch.long or ids.ndim != 1 or values.shape[-1] != len(ids):
            raise ValueError("field width differs from precompiled vocabulary IDs")
        values = torch.where(known, values, torch.zeros_like(values))
        metadata = self.field_embedding(ids)
        metadata = metadata.expand(*values.shape, metadata.shape[-1])
        local = self.field_encoder(torch.cat((values[..., None], known.to(values.dtype)[..., None], metadata), -1))
        if owner_field_incidence is None:
            shared = torch.einsum("fg,...gd->...fd", self.owner_pool.to(values.dtype), local)
        else:
            owner = owner_field_incidence.to(device=values.device, dtype=values.dtype)
            if owner.shape != (values.shape[0], values.shape[-1], values.shape[-1]) or bool((owner.sum(-1) == 0).any()):
                raise ValueError("typed owner incidence requires nonempty [B,F,F] rows")
            owner = owner / owner.sum(-1, keepdim=True)
            shared = torch.einsum("bfg,b...gd->b...fd", owner, local)
        return self.owner_encoder(torch.cat((local, shared), -1))

    def forward(self, history, sampled_states, sampled_truth, *, history_known,
                state_known, truth_known, applicable, predicate_field_incidence,
                prior_logits=None, owner_field_incidence=None, field_type_ids=None, return_branch_logs=False):
        """Return log PMF [B,P,11], or branch logs [B,P,K,11].

        prior_logits, when supplied, must be declared observed-history adapter
        output [B,P,11]. Branch probabilities are averaged after softmax.
        """
        if history.ndim != 3 or sampled_states.ndim != 4:
            raise ValueError("expected history[B,H,F] and sampled_states[B,K,11,F]")
        batch, branches, steps, fields = sampled_states.shape
        if steps != 11 or branches < 1 or history.shape[0] != batch or history.shape[-1] != fields:
            raise ValueError("state trajectories require anchor plus ten future steps")
        if sampled_truth.ndim != 4 or sampled_truth.dtype != torch.bool or sampled_truth.shape[:3] != (batch, branches, steps):
            raise ValueError("sampled truth requires Boolean [B,K,11,P]")
        predicates = sampled_truth.shape[-1]
        if truth_known.dtype != torch.bool or applicable.dtype != torch.bool or truth_known.shape != sampled_truth.shape or applicable.shape != sampled_truth.shape:
            raise ValueError("rule-known/applicable masks must align with rule truth")
        if predicate_field_incidence.dtype != torch.bool:
            raise ValueError("exact predicate-field incidence must be Boolean")
        incidence = predicate_field_incidence.to(dtype=history.dtype, device=history.device)
        if incidence.ndim == 2:
            incidence = incidence[None].expand(batch, -1, -1)
        if incidence.shape != (batch, predicates, fields) or bool((incidence.sum(-1) == 0).any()):
            raise ValueError("each predicate needs declared field incidence [P,F] or [B,P,F]")
        pool = incidence / incidence.sum(-1, keepdim=True)
        prefix = torch.einsum("bpf,bhfd->bphd", pool, self._fields(history, history_known, owner_field_incidence, field_type_ids))
        states = torch.einsum("bpf,bktfd->bpktd", pool, self._fields(sampled_states, state_known, owner_field_incidence, field_type_ids))
        known = (truth_known & applicable).permute(0, 3, 1, 2)
        trace = torch.where(known, sampled_truth.permute(0, 3, 1, 2), False)
        sequence = torch.cat((states, trace.to(states.dtype)[..., None], known.to(states.dtype)[..., None]), -1)
        head = self.trajectory_head
        _, prefix_hidden = head.history_encoder(prefix.reshape(batch * predicates, history.shape[1], -1))
        _, branch_hidden = head.branch_encoder(sequence.reshape(batch * predicates * branches, steps, -1))
        prefix_hidden = prefix_hidden[-1].reshape(batch, predicates, 1, -1).expand(-1, -1, branches, -1)
        context = head.context_encoder(torch.cat((prefix_hidden, branch_hidden[-1].reshape(batch, predicates, branches, -1)), -1))
        # A rule-derived outcome is identified only if the anchor and every
        # observation through its first flip (or all ten non-flips) are known.
        opposite = trace[..., 1:] != trace[..., :1]
        first = opposite.long().argmax(-1)
        has_flip = opposite.any(-1)
        bins = torch.where(has_flip, first, torch.full_like(first, 10))
        prefix_known = known[..., 1:].long().cumprod(-1).bool()
        identified = known[..., 0] & torch.where(has_flip, prefix_known.gather(-1, first[..., None]).squeeze(-1), prefix_known[..., -1])
        outcome = torch.nn.functional.one_hot(bins, 11).to(states.dtype) * identified[..., None]
        logits = head.outcome_skip(outcome) + head.residual(torch.cat((context, outcome), -1))
        if prior_logits is not None:
            if prior_logits.shape != (batch, predicates, 11):
                raise ValueError("declared history prior requires [B,P,11] logits")
            logits = logits + prior_logits[:, :, None]
        logs = torch.log_softmax(logits, -1)
        return logs if return_branch_logs else torch.logsumexp(logs, 2) - math.log(branches)
