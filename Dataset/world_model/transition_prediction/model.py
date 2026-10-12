"""Matched-information graph operators with a shared discrete hazard objective."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(n, d, out=None):
    return nn.Sequential(nn.Linear(n, d), nn.SiLU(), nn.Linear(d, d if out is None else out))


def masked_mean(value, present, dim):
    return (value * present.unsqueeze(-1)).sum(dim) / present.sum(dim).clamp_min(1).unsqueeze(-1)


def log_distribution(logits):
    log_survive = F.logsigmoid(-logits)
    prefix = torch.cat([torch.zeros_like(log_survive[:, :1]), log_survive.cumsum(-1)[:, :-1]], -1)
    logp = torch.cat([F.logsigmoid(logits) + prefix, log_survive.sum(-1, keepdim=True)], -1)
    if not torch.isfinite(logp).all():
        raise FloatingPointError("non-finite predictive distribution")
    if not torch.allclose(logp.exp().sum(-1), torch.ones(len(logits), dtype=logp.dtype, device=logp.device), atol=2e-6):
        raise ArithmeticError("hazard probabilities do not sum to one")
    return logp


def censored_nll(logp, event_observed, event_bin, observed_bins):
    """First-event likelihood; censoring after m whole bins scores survival S_m.

    Zero-bin censored rows carry no information and must receive zero training
    weight. The head and its eleven probability masses are unchanged.
    """
    if logp.ndim != 2 or logp.shape[1] != 11:
        raise ValueError("first-event distribution must have eleven masses")
    event = torch.as_tensor(event_observed, dtype=torch.bool, device=logp.device)
    bins = torch.as_tensor(event_bin, dtype=torch.long, device=logp.device)
    observed = torch.as_tensor(observed_bins, dtype=torch.long, device=logp.device)
    if any(value.shape != (len(logp),) for value in (event, bins, observed)):
        raise ValueError("censor metadata must match the query population")
    if ((observed < 0) | (observed > 10)).any():
        raise ValueError("observed_bins must be in 0..10")
    if ((bins[event] < 0) | (bins[event] >= 10)).any() or (bins[~event] != -1).any():
        raise ValueError("event_bin must be 0..9 for events and -1 otherwise")
    tail = torch.arange(11, device=logp.device)[None] >= observed[:, None]
    survival_loss = -torch.logsumexp(logp.masked_fill(~tail, -torch.inf), dim=-1)
    event_loss = -logp.gather(1, bins.clamp_min(0)[:, None]).squeeze(1)
    return torch.where(event, event_loss, survival_loss)


class PriorOffset(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(10))
        self.register_buffer("prior_logits", torch.zeros(len(vocab["predicates"]), 2, 10))

    def forward(self, x):
        query = x["query"]
        current = x["truth"][-1, query] - 1
        return log_distribution(self.prior_logits[x["predicate_type"][query], current] + self.offset)


def build_model(vocab, mode, width=24):
    return PriorOffset(vocab) if mode == "Offset" else SwitchModel(vocab, mode, width)


def incidence_to_entities(pred, role, pointers, valid, entity_count):
    t, p, d = pred.shape
    messages = pred[:, :, None, :] * torch.sigmoid(role)[None]
    messages = messages * valid[..., None]
    target = pointers.flatten()
    result = pred.new_zeros(t, entity_count, d)
    result.index_add_(1, target, messages.reshape(t, -1, d))
    count = pred.new_zeros(t, entity_count)
    count.index_add_(1, target, valid.reshape(t, -1).float())
    return result / count.clamp_min(1)[..., None]


class SpatialBlock(nn.Module):
    def __init__(self, d, attention=False):
        super().__init__()
        self.entity = mlp(2*d, d)
        self.predicate = mlp(2*d, d)
        self.enorm, self.pnorm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = attention
        if attention:
            self.qkv = nn.Linear(d, 3*d)
            self.pair_bias = mlp(7, d, 1)

    def forward(self, h, z, role, x, mode, incident_source=None):
        active = (x["truth"] >= 1) & (x["truth"] <= 3)
        valid = active[..., None] & x["role_present"][None]
        if mode == "M0":
            incoming = torch.zeros_like(h)
        else:
            incoming = incidence_to_entities(z if incident_source is None else incident_source, role, x["pointers"], valid, h.shape[1])
        if mode == "T":
            q, k, v = self.qkv(h + incoming).chunk(3, -1)
            state = x["physical"]
            difference = state[:, :, None, :6] - state[:, None, :, :6]
            pair_present = x["physical_present"][:, :, None] & x["physical_present"][:, None, :]
            pair = torch.cat([difference * pair_present[..., None], pair_present[..., None]], -1)
            score = q @ k.transpose(-1, -2) / math.sqrt(h.shape[-1]) + self.pair_bias(pair).squeeze(-1)
            score = score.masked_fill(~x["entity_visible"][:, None, :], -1e4)
            incoming = incoming + score.softmax(-1) @ v
        h = self.enorm(h + self.entity(torch.cat([h, incoming], -1)))
        h = h * x["entity_visible"][..., None]
        bound = h[:, x["pointers"]] * torch.sigmoid(role)[None]
        bound = masked_mean(bound, valid, 2)
        z = self.pnorm(z + self.predicate(torch.cat([z, bound], -1)))
        z = z * active[..., None]
        return h, z


class SwitchModel(nn.Module):
    def __init__(self, vocab, mode, width=24, parameter_target=60000):
        super().__init__()
        self.mode, self.width = mode, width
        d = width
        self.entity_type = nn.Embedding(len(vocab["categories"]), d)
        self.predicate_type = nn.Embedding(len(vocab["predicates"]), d)
        self.role_type = nn.Embedding(len(vocab["roles"]), d)
        self.entity_in = mlp(d+10, d)
        self.predicate_in = mlp(d+6, d)
        self.blocks = nn.ModuleList([SpatialBlock(d, mode == "T") for _ in range(2)])
        self.has_global = mode in ("M2", "T", "G", "G_no_history")
        self.query_in = mlp((3 if self.has_global else 2)*d, d)
        self.temporal = nn.GRU(d, d, batch_first=True)
        current = sum(p.numel() for p in self.parameters())
        hidden = max(d, (parameter_target - current - 10) // (d+11))
        self.head = mlp(d, hidden, 10)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("prior_logits", torch.zeros(len(vocab["predicates"]), 2, 10))

    def forward(self, x, return_embedding=False):
        if self.mode == "G_no_history":
            temporal_fields = ("physical", "physical_present", "entity_visible", "truth", "relative_age")
            x = {k: v[-1:] if k in temporal_fields else v for k, v in x.items()}
        t = x["truth"].shape[0]
        age = x["relative_age"][:, None, None]
        entity_type = self.entity_type(x["entity_type"])[None].expand(t, -1, -1)
        pred_type = self.predicate_type(x["predicate_type"])[None].expand(t, -1, -1)
        h = self.entity_in(torch.cat([x["physical"], x["physical_present"][..., None],
                                     age.expand(-1, entity_type.shape[1], -1), entity_type], -1))
        z = self.predicate_in(torch.cat([F.one_hot(x["truth"], 5).float(),
                                        age.expand(-1, pred_type.shape[1], -1), pred_type], -1))
        h = h * x["entity_visible"][..., None]
        active = (x["truth"] >= 1) & (x["truth"] <= 3)
        z = z * active[..., None]
        role = self.role_type(x["role_type"])
        z0 = z
        mode = "G" if self.mode == "G_no_history" else self.mode
        for block in self.blocks:
            if mode in ("M1", "M2"):
                # Fixed query-incident tuples at every layer; no multi-hop propagation.
                h, z = block(h, z, role, x, "M1", incident_source=z0)
            else:
                h, z = block(h, z, role, x, mode)
        valid = active[..., None] & x["role_present"][None]
        bound = masked_mean(h[:, x["pointers"]] * torch.sigmoid(role)[None], valid, 2)
        query = x["query"]
        parts = [z[:, query], bound[:, query]]
        if self.has_global:
            global_context = (masked_mean(h, x["entity_visible"], 1) + masked_mean(z, active, 1)) / 2
            parts.append(global_context[:, None].expand(-1, len(query), -1))
        sequence = self.query_in(torch.cat(parts, -1)).transpose(0, 1)
        _, hidden = self.temporal(sequence)
        embedding = hidden[-1]
        current = x["truth"][-1, query] - 1
        logits = self.head(embedding) + self.prior_logits[x["predicate_type"][query], current]
        # log P(first switch in k) and log P(no switch); no independent confidence head.
        logp = log_distribution(logits)
        return (logp, embedding) if return_embedding else logp
