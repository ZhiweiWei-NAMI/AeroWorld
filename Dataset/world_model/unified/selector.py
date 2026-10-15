"""0.8B selector backbone: the query-conditioned task prefix and record reading used by value_head.record_states.

The scoring head, metadata and KEEP/EDIT control modules remain only so earlier checkpoints (selector_init.pt) load
unchanged; the measured-value critic (value_head.py) reads the backbone's final hidden states.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
import re

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from Dataset.world_model.unified import assemble as A, frame_cache as FC, schema as S
from Dataset.world_model.unified.predictor import base_text_model, load_backbone
from Dataset.world_model.unified.selection import ScoringConfig

SEGMENT_BITS = (FC.RGB, FC.DEPTH, FC.SEG, FC.LIDAR, FC.LIDAR)
PLACEHOLDER = "<|image_pad|>"
FORMAT = "aeroworld-bundle-policy-v8"
RECORD_KINDS = ("origin", "knot", "frame_change", "comm_change", "compute_change", "rel_change",
                "evidence_change", "event", "scheduled", "leave", "capture", "other")
HORIZON = re.compile(r"^(.*)@\+([0-9.]+)s:$")


def grouped_queries(texts):
    """One line per target: "p2 position@+0.5s:" ... "@+5.0s:" become "p2 position@+0.5,1.0,2.0,3.0,5.0s:".
    Lossless: every horizon stays listed; targets keep their first-appearance order."""
    lines, horizons = [], {}
    for t in texts:
        m = HORIZON.match(t)
        if m is None:
            lines.append(t)
        elif m.group(1) in horizons:
            horizons[m.group(1)].append(m.group(2))
        else:
            horizons[m.group(1)] = [m.group(2)]
            lines.append(m.group(1))
    return [f"{t}@+{','.join(horizons[t])}s:" if t in horizons else t for t in lines]


def prefix_ids(pw, tok, query_ids, budget, actions_ids):
    """Shared context of every chunk: budget, entity roster, the full definition of every queried relation (heading
    with role bindings, predicate expression and thresholds), the query set and its fixed origin lines."""
    texts = grouped_queries([pw.w["queries"][i]["text"] for i in query_ids])
    fixed, _ = pw.fixed_keys(query_ids)
    fixed_text = "\n".join(f"{pw.units[k].head.split('|')[1]}:{pw.units[k].line}" for k in
                           sorted(fixed, key=lambda k: (A._head_order(pw.units[k].head), k)))
    roster = " ".join(f"{h.split('|', 1)[1]}={pw.w['headings'][h].split()[2]}" for h in
                      sorted((h for h in pw.w["headings"] if h.startswith("entity|")), key=A._head_order))
    defs, lines = A.predicate_definitions(), []
    for t in sorted(pw.queried_tuples(query_ids), key=lambda t: int(t[1:])):
        heading = pw.w["headings"][f"rel|{t}"]
        pred = heading.split(" = ", 1)[1].split("(", 1)[0]
        d = defs.get(pred)
        if d:
            th = ", ".join(f"{k}={v}" for k, v in d["thresholds"].items())
            heading += f"  [{d['expression']}" + (f"; {th}" if th else "") + "]"
        owners = pw.w["rel_entities"].get(t, [])
        lines.append(heading + (f"  entities: {','.join(owners)}" if owners else ""))
    return tok(f"select evidence (history records and sensor captures) for {budget} input positions\n"
               f"entities: {roster}\nqueried relations:\n" + "\n".join(lines) + "\nqueries:\n" + "\n".join(texts)
               + "\nfixed:\n" + fixed_text + "\n", add_special_tokens=False)["input_ids"] + list(actions_ids) + \
        tok("\nScore the candidates as alternatives, not as already selected evidence.\n",
            add_special_tokens=False)["input_ids"]



class Selector(nn.Module):
    def __init__(self, device="cuda", config=None):
        super().__init__()
        self.config = config or ScoringConfig()
        self.lm = load_backbone(S.QWEN_08B, device, self.config.lora_rank)
        self.text = base_text_model(self.lm)
        d = self.text.config.hidden_size
        self.budget = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        # Capture summary -> one input embedding at word-embedding scale.  Each modality segment of the 2,631-value
        # summary is normalized on its own; segments of modalities the capture does not have are zeroed and a
        # learned "absent" vector per missing modality is added, so an absent modality never reads as observed zeros.
        self.visual = nn.ModuleDict({"norm": nn.ModuleList(nn.LayerNorm(n) for _, n in FC.SEGMENTS),
                                     "proj": nn.Linear(FC.SUMMARY_DIM, d)})
        nn.init.normal_(self.visual["proj"].weight, std=0.02 / math.sqrt(FC.SUMMARY_DIM))
        nn.init.zeros_(self.visual["proj"].bias)
        self.visual_absent = nn.Parameter(torch.randn(len(SEGMENT_BITS), d) * 0.02)
        self.kind = nn.Parameter(torch.randn(len(RECORD_KINDS), d) * 0.02)          # text, capture
        layer = nn.TransformerEncoderLayer(d, 8, 2*d, dropout=0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, self.config.head_layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.normal_(self.out[1].weight, std=0.01)
        nn.init.constant_(self.out[1].bias, -2.)     # categorical preference, not a calibrated inclusion probability
        # Observed metadata is an input embedding, never a hand-scored fallback.
        self.metadata = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.metadata[-1].weight)
        nn.init.zeros_(self.metadata[-1].bias)
        self.controls = nn.Linear(d, 2)  # KEEP, EDIT from the contextual budget token
        nn.init.normal_(self.controls.weight, std=0.01)
        nn.init.constant_(self.controls.bias, -4.)
        self.to(device)
        self.device = torch.device(device)
        self.checkpointing = True
        self.checkpoint_every = int(os.environ.get("SELECTOR_CHECKPOINT_EVERY", "1"))
        if self.checkpoint_every < 0:
            raise ValueError("SELECTOR_CHECKPOINT_EVERY must be nonnegative (0 disables recomputation)")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def summary_embedding(self, vec, masks):
        """vec [n, SUMMARY_DIM], masks [n] modality bit masks -> [n, d]."""
        parts, absent, start = [], 0., 0
        for j, ((_, n), bit) in enumerate(zip(FC.SEGMENTS, SEGMENT_BITS)):
            present = ((masks & bit) > 0).float()[:, None]
            parts.append(self.visual["norm"][j](vec[:, start:start + n]) * present)
            absent = absent + (1. - present) * self.visual_absent[j]
            start += n
        return self.visual["proj"](torch.cat(parts, 1)) + absent

    def prefix_states(self, ids):
        """Run the shared prefix once; per-layer (kind, a, b) states every batch continues from
        (key/value for full attention, conv/recurrent state for linear attention; as Predictor.context_states)."""
        cfg = self.text.config
        x = self.text.embed_tokens(torch.tensor(ids, device=self.device))[None]
        pos = torch.arange(len(ids), device=self.device)[None, None].expand(3, 1, -1)
        pe = self.text.rotary_emb(x, pos)
        h, states = x, []
        for i, layer in enumerate(self.text.layers[: cfg.num_hidden_layers]):
            kind = cfg.layer_types[i]

            def run(h, cos, sin, layer=layer, i=i, kind=kind):
                tmp = DynamicCache(config=cfg)
                out = layer(h, position_embeddings=(cos, sin), attention_mask=None, position_ids=None,
                            past_key_values=tmp)
                L = tmp.layers[i]
                if kind == "full_attention":
                    return out, L.keys, L.values
                return out, L.conv_states[0], L.recurrent_states[0]
            if self.checkpointing and self.checkpoint_every and (i + 1) % self.checkpoint_every == 0 \
                    and torch.is_grad_enabled():
                h, a, b = checkpoint(run, h, pe[0], pe[1], use_reentrant=False)
            else:
                h, a, b = run(h, pe[0], pe[1])
            states.append((kind, a, b))
        return states

    def layer_cache(self, i, kind, a, b):
        """A one-layer cache holding the prefix state, with out-of-place updates so the state stays intact for
        autograd and for the next batch (as Predictor.fork_cache). Built inside each checkpointed call."""
        cache = DynamicCache(config=self.text.config)
        L = cache.layers[i]
        if kind == "full_attention":
            L.lazy_initialization(a, b)
            L.keys, L.values = a, b
        else:
            L.conv_states[0], L.recurrent_states[0] = a, b
            L.is_conv_states_initialized[0] = L.is_recurrent_states_initialized[0] = True
            L.has_previous_state[0] = True
            L.conv_kernel_size[0] = a.shape[-1]
            L.device, L.dtype = a.device, a.dtype
            L.update_conv_state = (lambda new, state_idx=0, _L=L, **kw: torch.cat([_L.conv_states[state_idx], new], -1))
            L.update_recurrent_state = (lambda new, state_idx=0, **kw: new)
        return cache

    def encode_chunk(self, states, start, ids, inserts=(), summaries=None):
        """Hidden states of one batch continuing from the prefix states (prefix length start)."""
        cfg = self.text.config
        x = self.text.embed_tokens(torch.tensor(ids, device=self.device))[None]
        if inserts:
            at = torch.tensor([i for i, _ in inserts], device=self.device)
            vec = torch.stack([summaries[k][0] for _, k in inserts]).to(self.device, torch.float32)
            masks = torch.tensor([summaries[k][1] for _, k in inserts], device=self.device)
            x = x.index_put((torch.zeros_like(at), at), self.summary_embedding(vec, masks).to(x.dtype))
        n = len(ids)
        pos = (start + torch.arange(n, device=self.device))[None, None].expand(3, 1, -1)
        pe = self.text.rotary_emb(x, pos)
        # Full attention sees the whole prefix and is causal within the batch. SDPA's own is_causal would align
        # the query block to the first keys, so the mask is explicit; linear attention takes no mask.
        mask = torch.ones(n, start + n, dtype=torch.bool, device=self.device).tril(diagonal=start)[None, None]
        h = x
        for i, layer in enumerate(self.text.layers[: cfg.num_hidden_layers]):
            kind, a, b = states[i]

            def run(h, cos, sin, a, b, layer=layer, i=i, kind=kind):
                return layer(h, position_embeddings=(cos, sin), position_ids=None,
                             attention_mask=mask if kind == "full_attention" else None,
                             past_key_values=self.layer_cache(i, kind, a, b))
            h = checkpoint(run, h, pe[0], pe[1], a, b, use_reentrant=False) \
                if self.checkpointing and self.checkpoint_every and (i + 1) % self.checkpoint_every == 0 \
                and torch.is_grad_enabled() else run(h, pe[0], pe[1], a, b)
        return self.text.norm(h)[0]

    def trainable_state(self):
        return {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad}

    def restore(self, weights):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(weights) != expected:
            raise ValueError(f"selector checkpoint keys differ: missing={sorted(expected - set(weights))[:5]}, "
                             f"unexpected={sorted(set(weights) - expected)[:5]}")
        self.load_state_dict(weights, strict=False)  # frozen base weights come from the pinned base model

    def save(self, path):
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        torch.save({"format": FORMAT, "config": asdict(self.config), "weights": self.trainable_state()}, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path, device="cuda"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format"] != FORMAT:
            raise ValueError("selector checkpoint has incompatible action semantics; initialize a new gain-editor experiment")
        model = cls(device=device, config=ScoringConfig(**state["config"]))
        model.restore(state["weights"])
        return model
