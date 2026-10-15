"""Qwen3.5-2B predictor over structured text and sensor embeddings (RESEARCH_PLAN §3.5, §4, §6).

Context = header + image packet + selected history.  RGB enters as the cached
post-merger visual tokens of the native path with M-RoPE positions from its
image_grid_thw; depth, segmentation and LiDAR region features enter through
learned projections with position indices tied to the same frame's grid.

Scoring = locally normalized prefix tree: at every internal node the LM-head
logits are restricted to the node's valid child tokens and normalized there;
a code's probability is the product along its path.

Marginal mode: the context runs once; every query continues from the shared
context state and sees only the context and its own code prefix.  Isolation in
the hybrid backbone is by batch rows, not by an attention mask: each query is
its own row whose full-attention layers attend to the context K/V (an expanded
view, concatenated with the row's own keys per layer, so one transient copy per
row of a microbatch) and whose linear-attention layers start from their own copy
of the context's convolution and recurrent states, so no information path
connects two queries.  Queries run in microbatches of QUERY_CHUNK rows (default 4); in
training each microbatch back-propagates into detached context-state leaves and
frees its graph before the next one, and the accumulated state gradients flow
through the context once.  Context states are recomputed for every item, never
reused across optimizer updates.
Joint mode: queries in time-major order in one sequence, each answer
conditioned on all earlier answers (teacher forcing).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

from Dataset.world_model.unified import schema as S
from Dataset.world_model.unified.frame_cache import FOUR_MODAL

def _stable_kernel_tuning():
    """Pin the accelerated l2norm configuration used by the frozen reward model.

    Real identical-input repeats selected BT=128/warps=4 versus BT=32/warps=1
    and returned different NLLs. Autotuning is not numerically invariant here.
    Keep Triton acceleration, but do not let a timing race change RL rewards.
    """
    try:
        import triton
        from fla.modules import l2norm as fla_l2norm
    except ImportError:
        return
    for name in ("l2norm_fwd_kernel", "l2norm_bwd_kernel"):
        kernel = getattr(fla_l2norm, name, None)
        if kernel is not None and hasattr(kernel, "key_idx"):
            kernel.configs = [triton.Config({'BT': 128}, num_warps=4)]
            kernel.key_idx = [kernel.arg_names.index("D")]
            if hasattr(kernel, "keys"):
                kernel.keys = ["D"]


_stable_kernel_tuning()

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj")


def load_backbone(path, device, lora_rank=16, dtype=torch.bfloat16):
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    full = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    lm = full.model.language_model
    lm.config.use_cache = True
    del full.model.visual
    lm.to(device)
    for p in lm.parameters():
        p.requires_grad_(False)
    if lora_rank:
        lm = get_peft_model(lm, LoraConfig(r=lora_rank, lora_alpha=2*lora_rank, lora_dropout=0.0,
                                           target_modules=list(LORA_TARGETS)))
    return lm


def base_text_model(lm):
    return lm.base_model.model if hasattr(lm, "base_model") else lm


class SensorAdapter(nn.Module):
    """Projections of cached depth/segmentation/LiDAR features into the LM hidden space.

    Four-modal captures give RGB, depth, segmentation and LiDAR tokens; RGB-only captures (the fixed overview
    camera) give RGB tokens only, with their own modality embedding (index 4) so the model can tell the cameras
    apart; absent modalities produce no positions at all."""
    def __init__(self, hidden=2048):
        super().__init__()
        self.depth = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.seg = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        # LiDAR region token: frozen Sonata regional feature (512), the seven region statistics and presence.
        self.lidar = nn.Sequential(nn.Linear(512 + 7 + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.lidar_norm = nn.LayerNorm(512)
        self.descriptor = nn.ModuleDict({m: nn.Linear(d, hidden) for m, d in (("rgb", 18), ("depth", 18), ("seg", 18), ("lidar", 19))})
        # Modality embeddings start at the scale of word embeddings: an all-zero input position would sit at the
        # RMSNorm singularity (gradient ~ 1/sqrt(eps)).  Projections start small so images perturb the LM gently.
        self.modality = nn.Parameter(torch.randn(5, hidden) * 0.02)       # rgb, depth, seg, lidar, rgb of a fixed camera
        for m in (self.depth[-1], self.seg[-1], self.lidar[-1], *self.descriptor.values()):
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)

    def forward(self, feats, dtype):
        """Embeddings [N, hidden] and their (row, col) grid coordinates on the RGB merged grid."""
        dev = self.modality.device
        t, h, w = (int(x) for x in feats["image_grid_thw"][0])
        hr, wr = h // 2, w // 2
        mods = int(feats.get("modalities", FOUR_MODAL))
        rgb = feats["rgb"].to(dev, torch.float32) + self.descriptor["rgb"](feats["descriptors"]["rgb"].to(dev)) + \
            self.modality[0 if mods == FOUR_MODAL else 4]
        rows = [torch.arange(hr).repeat_interleave(wr), torch.arange(wr).repeat(hr)]
        out, coords = [rgb], [torch.stack(rows, 1)]
        if mods != FOUR_MODAL:
            if mods != 1:
                raise ValueError(f"unsupported modality mask {mods}")
            return torch.cat(out).to(dtype), torch.cat(coords), max(hr, wr)

        def pooled(x, side):        # 14x14 cells -> 7x7 tokens
            g = int(math.isqrt(x.shape[0]))
            x = x.reshape(g, g, -1).permute(2, 0, 1)[None]
            return F.adaptive_avg_pool2d(x, side)[0].permute(1, 2, 0).reshape(side*side, -1)
        for i, m in enumerate(("depth", "seg")):
            x = pooled(feats[m].to(dev, torch.float32), 7)
            out.append(getattr(self, m)(x) + self.descriptor[m](feats["descriptors"][m].to(dev)) + self.modality[1+i])
            r = torch.arange(7)
            coords.append(torch.stack([(r.repeat_interleave(7) * hr) // 7, (r.repeat(7) * wr) // 7], 1))
        stats = feats["lidar_region_stats"].to(dev, torch.float32)
        # Region stats: mean xyz (m), range mean/std (m), hit fraction, occupied flag -> metres scaled by 1/100.
        stats = torch.cat([stats[:, :5] / 100., stats[:, 5:]], 1)
        sonata = self.lidar_norm(feats["lidar_sonata"].to(dev, torch.float32))
        present = feats["lidar_present"].to(dev, torch.float32)[:, None]
        region = torch.cat([sonata * present, stats, present], 1)
        out.append(self.lidar(region) + self.descriptor["lidar"](feats["descriptors"]["lidar"].to(dev)) + self.modality[3])
        n = stats.shape[0]
        gr, gc = 4, max(1, n // 4)
        idx = torch.arange(n)
        coords.append(torch.stack([((idx // gc) * hr) // gr, ((idx % gc) * wr) // gc], 1))
        return torch.cat(out).to(dtype), torch.cat(coords), max(hr, wr)


@dataclass
class Context:
    embeds: torch.Tensor           # [1, L, H]
    positions: torch.Tensor        # [3, 1, L]
    next_position: int
    length: int


class Predictor(nn.Module):
    QUERY_CHUNK = 4

    def __init__(self, path=S.QWEN_2B, device="cuda", lora_rank=16, checkpointing=True, query_chunk=QUERY_CHUNK):
        super().__init__()
        self.query_chunk = query_chunk
        self.lm = load_backbone(path, device, lora_rank)
        self.text = base_text_model(self.lm)
        self.cfg = self.text.config
        self.adapter = SensorAdapter(self.cfg.hidden_size).to(device)
        self.device = torch.device(device)
        self.checkpointing = checkpointing

    @property
    def embed(self):
        return self.text.embed_tokens

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- context ---------------------------------------------------------------------------
    PAD_TO = 64           # context lengths are bucketed (trailing blank lines) so kernels see few distinct shapes
    NEWLINE = 198         # token id of "\n" in the Qwen3.5 tokenizer

    def build_context(self, segments, visual_loader=None):
        embeds, positions, pos = [], [], 0
        dtype = self.embed.weight.dtype
        for kind, value in segments:
            if kind == "text":
                if not value:
                    continue
                ids = torch.tensor(value, device=self.device)
                embeds.append(self.embed(ids))
                p = torch.arange(pos, pos + len(value), device=self.device)
                positions.append(p[None].expand(3, -1))
                pos += len(value)
            else:
                feats = visual_loader(value)
                e, coords, advance = self.adapter(feats, dtype)
                embeds.append(e)
                coords = coords.to(self.device)
                positions.append(torch.stack([torch.full_like(coords[:, 0], pos), pos + coords[:, 0], pos + coords[:, 1]]))
                pos += advance
        n = sum(e.shape[0] for e in embeds)
        pad = (-n) % self.PAD_TO
        if pad:
            embeds.append(self.embed(torch.full((pad,), self.NEWLINE, device=self.device)))
            positions.append(torch.arange(pos, pos + pad, device=self.device)[None].expand(3, -1))
            pos += pad
        x = torch.cat(embeds)[None]
        return Context(x, torch.cat(positions, 1)[:, None], pos, x.shape[1])

    def _rotary(self, hidden, positions):
        return self.text.rotary_emb(hidden, positions)

    def context_states(self, ctx: Context):
        """Run the context once; return per-layer states needed by continuations (differentiable)."""
        hidden = ctx.embeds
        pe = self._rotary(hidden, ctx.positions)
        states = []
        for i, layer in enumerate(self.text.layers[: self.cfg.num_hidden_layers]):
            kind = self.cfg.layer_types[i]

            def run(h, cos, sin, layer=layer, i=i, kind=kind):
                tmp = DynamicCache(config=self.cfg)
                out = layer(h, position_embeddings=(cos, sin), attention_mask=None, position_ids=None,
                            past_key_values=tmp)
                L = tmp.layers[i]
                if kind == "full_attention":
                    return out, L.keys, L.values
                return out, L.conv_states[0], L.recurrent_states[0]
            if self.checkpointing and torch.is_grad_enabled():
                hidden, a, b = checkpoint(run, hidden, pe[0], pe[1], use_reentrant=False)
            else:
                hidden, a, b = run(hidden, pe[0], pe[1])
            states.append((kind, a, b))
        return states, hidden

    def fork_cache(self, states, batch, persistent=False):
        """Per-query copies of the context state.  persistent=True keeps the standard in-place cache updates
        (incremental decoding); otherwise updates are out of place so the shared state stays intact for autograd."""
        cache = DynamicCache(config=self.cfg)
        for i, (kind, a, b) in enumerate(states):
            L = cache.layers[i]
            if kind == "full_attention":
                L.lazy_initialization(a, b)
                L.keys, L.values = a.expand(batch, *a.shape[1:]), b.expand(batch, *b.shape[1:])
            else:
                conv, rec = a.repeat(batch, *([1]*(a.dim()-1))), b.repeat(batch, *([1]*(b.dim()-1)))
                L.conv_states[0], L.recurrent_states[0] = conv, rec
                L.is_conv_states_initialized[0] = L.is_recurrent_states_initialized[0] = True
                L.has_previous_state[0] = True
                L.conv_kernel_size[0] = conv.shape[-1]
                L.device, L.dtype = conv.device, conv.dtype
                if not persistent:
                    # Out-of-place updates: the forked state must stay intact for autograd and other queries.
                    L.update_conv_state = (lambda new, state_idx=0, _L=L, **kw: torch.cat([_L.conv_states[state_idx], new], -1))
                    L.update_recurrent_state = (lambda new, state_idx=0, **kw: new)
        return cache

    # ---- scoring ---------------------------------------------------------------------------
    def restricted_logits(self, hidden, candidates):
        """hidden [N, H]; candidates list of token-id lists -> list of logit vectors (restricted LM head)."""
        weight = self.lm.get_output_embeddings().weight if hasattr(self.lm, "get_output_embeddings") and \
            self.lm.get_output_embeddings() is not None else self.embed.weight
        flat = torch.tensor([t for c in candidates for t in c], device=hidden.device)
        owner = torch.tensor([i for i, c in enumerate(candidates) for _ in c], device=hidden.device)
        logits = (hidden[owner].float() * weight[flat].float()).sum(-1)
        return torch.split(logits, [len(c) for c in candidates])

    def continuation(self, ctx: Context, states, part):
        """Log-softmax step vectors of a batch of queries continuing from the shared context states."""
        n = max(len(ids) for ids, _ in part)
        n += (-n) % 8
        rows_total = len(part)
        part = part + [part[-1]] * ((-len(part)) % self.query_chunk)  # fixed batch shape; padded rows are ignored
        ids = torch.full((len(part), n), self.NEWLINE, dtype=torch.long, device=self.device)
        for j, (q, _) in enumerate(part):
            ids[j, :len(q)] = torch.tensor(q, device=self.device)
        x = self.embed(ids)
        pos = (ctx.next_position + torch.arange(n, device=self.device))[None, None].expand(3, len(part), n)
        cache = self.fork_cache(states, len(part))
        h = self.text(inputs_embeds=x, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
        rows, cands = [], []
        for j, (_, steps) in enumerate(part[:rows_total]):
            for off, cand in steps:
                rows.append(h[j, off])
                cands.append(cand)
        logits = self.restricted_logits(torch.stack(rows), cands)
        out, k = [], 0
        for _, steps in part[:rows_total]:
            out.append([F.log_softmax(logits[k+s], -1) for s in range(len(steps))])
            k += len(steps)
        return out

    def marginal(self, ctx: Context, queries, chunk=None):
        """queries: list of (input_ids, steps) where steps = [(offset, candidate ids)]; offset indexes input_ids.

        Returns the per-step log-softmax vectors (restricted, locally normalized) of every query.
        """
        states, _ = self.context_states(ctx)
        chunk = chunk or self.query_chunk
        out = []
        for c0 in range(0, len(queries), chunk):
            out += self.continuation(ctx, states, queries[c0:c0+chunk])
        return out

    def marginal_backward(self, ctx: Context, queries, loss_fn, chunk=None):
        """Marginal-mode training step with bounded memory.

        The context runs once; its states become detached leaves; every query chunk
        back-propagates into them (and the LoRA weights); finally the accumulated
        state gradients flow back through the context once.  ``loss_fn(i, logps)``
        returns the weighted loss of query i.  Returns the summed loss value.
        """
        states, _ = self.context_states(ctx)
        chunk = chunk or self.query_chunk
        leaves = [(k, a.detach().requires_grad_(), b.detach().requires_grad_()) for k, a, b in states]
        total = 0.
        for c0 in range(0, len(queries), chunk):
            lps = self.continuation(ctx, leaves, queries[c0:c0+chunk])
            loss = sum(loss_fn(c0 + j, lp) for j, lp in enumerate(lps))
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
                total += float(loss)
        outs, grads = [], []
        for (_, a, b), (_, la, lb) in zip(states, leaves):
            for t, lt in ((a, la), (b, lb)):
                if lt.grad is not None:
                    outs.append(t)
                    grads.append(lt.grad)
        if outs:
            torch.autograd.backward(outs, grads)
        return total

    def joint(self, ctx: Context, queries):
        """All queries in one sequence after the context (time-major order given by the caller)."""
        ids = [t for q, _ in queries for t in q]
        ids += [self.NEWLINE] * ((-(ctx.length + len(ids))) % self.PAD_TO)     # trailing pad, after every scored position
        x = torch.cat([ctx.embeds, self.embed(torch.tensor(ids, device=self.device))[None]], 1)
        pos = torch.cat([ctx.positions, (ctx.next_position + torch.arange(len(ids), device=self.device))[None, None].expand(3, 1, -1)], 2)
        h = self.forward_sequence(x, pos)[0]
        rows, cands, base = [], [], ctx.length
        for q, steps in queries:
            for off, cand in steps:
                rows.append(h[base + off])
                cands.append(cand)
            base += len(q)
        logits = self.restricted_logits(torch.stack(rows), cands)
        out, k = [], 0
        for _, steps in queries:
            out.append([F.log_softmax(logits[k+s], -1) for s in range(len(steps))])
            k += len(steps)
        return out

    @torch.no_grad()
    def sample_joint(self, ctx: Context, queries, k, vocab, generator=None):
        """K ancestral joint samples of the query block (joint mode, time-major order given by the caller).

        queries: [(text ids, [tree per component])].  Returns, per sample, a list of sampled codes per query
        (one tuple of tree symbols per component).  Every sampled answer is fed back before the next query.
        Samples are decoded one at a time so code depths may differ without filler tokens.
        """
        states, _ = self.context_states(ctx)
        samples = []
        for _ in range(k):
            cache = self.fork_cache(states, 1, persistent=True)
            pos = ctx.next_position

            def step(ids):
                nonlocal pos
                p = (pos + torch.arange(len(ids), device=self.device))[None, None].expand(3, 1, len(ids))
                h = self.text(inputs_embeds=self.embed(torch.tensor(ids, device=self.device))[None], position_ids=p,
                              past_key_values=cache, use_cache=True).last_hidden_state
                pos += len(ids)
                return h[0, -1]
            answers = []
            for text_ids, trees in queries:
                h = step(list(text_ids))
                codes = []
                for tree in trees:
                    prefix = ()
                    while prefix in tree.children:
                        cands = vocab.tokens(tree, prefix)
                        logits = self.restricted_logits(h[None], [cands])[0]
                        pick = int(torch.multinomial(F.softmax(logits, -1), 1, generator=generator))
                        prefix = prefix + (tree.children[prefix][pick],)
                        h = step([cands[pick]])
                    codes.append(prefix)
                answers.append(codes)
            samples.append(answers)
        return samples

    def forward_sequence(self, x, pos):
        hidden = x
        pe = self._rotary(hidden, pos)
        for i, layer in enumerate(self.text.layers[: self.cfg.num_hidden_layers]):
            def run(h, cos, sin, layer=layer):
                return layer(h, position_embeddings=(cos, sin), attention_mask=None, position_ids=None,
                             past_key_values=None)
            hidden = checkpoint(run, hidden, pe[0], pe[1], use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() \
                else run(hidden, pe[0], pe[1])
        return self.text.norm(hidden)
