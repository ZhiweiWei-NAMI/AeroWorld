"""Context selector: Qwen3.5-0.8B (same tokenizer as the predictor) over text records and captures (RESEARCH_PLAN §9 step 3).

Input.  Every chunk starts with the shared context: the optional budget, the roster of window entities (alias and
type), the full definition of every queried relation (heading with role bindings, the predicate's expression and
thresholds, and the entities it binds directly or through owned objects), the query set and the fixed origin
lines it makes mandatory, so a reference such as "v3" or "r7" means the same thing in every chunk whether or not
a candidate of it is present.  Then candidate units in the canonical order: captures first (oldest
first), then entities, relations, events and scheduled activations under their headings.  A text unit is its
native record line.  A capture is its metadata line (observer, modalities, time, camera position, height,
optical axis, field of view, sensor-token cost, the entities projected into the image with depth and image
position, and the relations binding them; addressed entities and queried relations first) followed by one
summary embedding of its frozen features (frame_cache.py) through a learned projection that marks absent
modalities, so the selector sees a compact description of every capture rather than its full embedding.

Decision.  The representation of a unit is the hidden state at its last token.  A shared head (a bidirectional
transformer over all unit representations of the window, with a text/capture type embedding, plus the budget)
gives p_keep, so decisions exchange information across chunks before scoring.  Histories longer than one chunk
are read in chunks; candidates are never pre-filtered.

Selection.  Units are taken in decreasing p_keep while they fit the optional budget (headings counted on
demand): p_keep estimates inclusion in good budget-feasible sets, so the ordering does not divide by cost
(a capture costs ~300 positions, a record ~15).  In a text-only stage the capture packet is fixed
(assemble.fixed_packet) and only text is selected in the remaining capacity.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from Dataset.world_model.unified import assemble as A, frame_cache as FC, schema as S
from Dataset.world_model.unified.predictor import base_text_model, load_backbone

CHUNK_TOKENS = 8192
SEGMENT_BITS = (FC.RGB, FC.DEPTH, FC.SEG, FC.LIDAR, FC.LIDAR)     # modality of each summary segment
PLACEHOLDER = "<|image_pad|>"      # its embedding is replaced by the projected capture summary


def reading_order(pw, unit_keys):
    """Units grouped under their headings in the serialization order."""
    by_head = {}
    for k in unit_keys:
        by_head.setdefault(pw.units[k].head, []).append(pw.units[k])
    out = []
    for head in sorted(by_head, key=A._head_order):
        out.append((head, sorted(by_head[head], key=lambda u: (u.kind != "origin", u.tick, u.key))))
    return out


def prefix_ids(pw, tok, query_ids, budget):
    """Shared context of every chunk: budget, entity roster, the full definition of every queried relation (heading
    with role bindings, predicate expression and thresholds), the query set and its fixed origin lines."""
    texts = [pw.w["queries"][i]["text"] for i in query_ids]
    fixed, _ = pw.fixed_keys(query_ids)
    fixed_text = "\n".join(f"{pw.units[k].head.split('|')[1]}:{pw.units[k].line}" for k in
                           sorted(fixed, key=lambda k: A._head_order(pw.units[k].head)))
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
               + "\nfixed:\n" + fixed_text + "\ncandidates:\n", add_special_tokens=False)["input_ids"]


def build_chunks(pw, tok, query_ids, budget, unit_keys, chunk_tokens=CHUNK_TOKENS):
    """[(token ids, [(unit key, index of its last token)], [(placeholder index, unit key)])] covering every unit once."""
    prefix = prefix_ids(pw, tok, query_ids, budget)
    ph = tok.convert_tokens_to_ids(PLACEHOLDER)
    chunks, ids, marks, inserts = [], list(prefix), [], []

    def unit_ids(u):
        ids = list(pw.selector_ids(u, query_ids))
        if u.kind == "capture" and u.summary is not None:
            return ids[:-1] + [ph] + ids[-1:], len(ids) - 1
        return ids, None
    for head, units in reading_order(pw, unit_keys):
        head_ids = list(pw.headings[head])
        first, _ = unit_ids(units[0])
        if len(ids) + len(head_ids) + len(first) > chunk_tokens and marks:
            chunks.append((ids, marks, inserts))
            ids, marks, inserts = list(prefix), [], []
        ids += head_ids
        for u in units:
            uids, at = unit_ids(u)
            if len(ids) + len(uids) > chunk_tokens and marks:
                chunks.append((ids, marks, inserts))
                ids, marks, inserts = list(prefix) + head_ids, [], []
            if at is not None:
                inserts.append((len(ids) + at, u.key))
            ids += uids
            marks.append((u.key, len(ids) - 1))
    if marks:
        chunks.append((ids, marks, inserts))
    return chunks


class Selector(nn.Module):
    def __init__(self, path=S.QWEN_08B, device="cuda", lora_rank=16, head_layers=2, checkpointing=True):
        super().__init__()
        self.lm = load_backbone(path, device, lora_rank)
        self.text = base_text_model(self.lm)
        d = self.text.config.hidden_size
        self.budget = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        # Capture summary -> one input embedding at word-embedding scale.  Each modality segment of the 2,631-value
        # summary is normalized on its own; segments of modalities the capture does not have are zeroed and a
        # learned "absent" vector per missing modality is added, so an absent modality never reads as observed zeros.
        self.visual = nn.ModuleDict({"norm": nn.ModuleList(nn.LayerNorm(n) for _, n in FC.SEGMENTS),
                                     "proj": nn.Linear(FC.SUMMARY_DIM, d)})
        nn.init.normal_(self.visual["proj"].weight, std=0.02 / math.sqrt(FC.SUMMARY_DIM))
        nn.init.zeros_(self.visual["proj"].bias)
        self.visual_absent = nn.Parameter(torch.randn(len(SEGMENT_BITS), d) * 0.02)
        self.kind = nn.Parameter(torch.randn(2, d) * 0.02)          # text, capture
        layer = nn.TransformerEncoderLayer(d, 8, 2*d, dropout=0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.normal_(self.out[1].weight, std=0.01)
        nn.init.constant_(self.out[1].bias, -2.)     # small initial keep probability: budgets keep few units
        self.to(device)
        self.device = torch.device(device)
        self.checkpointing = checkpointing

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

    def encode_chunk(self, ids, inserts=(), summaries=None):
        x = self.text.embed_tokens(torch.tensor(ids, device=self.device))[None]
        if inserts:
            at = torch.tensor([i for i, _ in inserts], device=self.device)
            vec = torch.stack([summaries[k][0] for _, k in inserts]).to(self.device, torch.float32)
            masks = torch.tensor([summaries[k][1] for _, k in inserts], device=self.device)
            x = x.index_put((torch.zeros_like(at), at), self.summary_embedding(vec, masks).to(x.dtype))
        pos = torch.arange(len(ids), device=self.device)[None, None].expand(3, 1, -1)
        pe = self.text.rotary_emb(x, pos)
        h = x
        for layer in self.text.layers[: self.text.config.num_hidden_layers]:
            def run(h, cos, sin, layer=layer):
                return layer(h, position_embeddings=(cos, sin), attention_mask=None, position_ids=None, past_key_values=None)
            h = checkpoint(run, h, pe[0], pe[1], use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() \
                else run(h, pe[0], pe[1])
        return self.text.norm(h)[0]

    def forward(self, chunks, budget, pw):
        """Logits of p_keep for every unit of the chunks (in chunk order) and their keys (chunks non-empty)."""
        summaries = {u.key: (u.summary, u.capture.modalities) for u in pw.captures() if u.summary is not None}
        reps, keys = [], []
        for ids, marks, inserts in chunks:
            h = self.encode_chunk(ids, inserts, summaries)
            idx = torch.tensor([m[1] for m in marks], device=self.device)
            reps.append(h[idx].float())
            keys += [m[0] for m in marks]
        kind = torch.tensor([int(pw.units[k].kind == "capture") for k in keys], device=self.device)
        units = torch.cat(reps)[None] / reps[0].new_tensor(self.text.config.hidden_size ** 0.5) + self.kind[kind][None]
        b = self.budget(torch.tensor([[math.log2(max(2, budget)) / 15.]], device=self.device))[None]
        z = self.head(torch.cat([b, units], 1))
        return self.out(z[0, 1:]).squeeze(-1), keys

    @torch.no_grad()
    def select(self, pw, tok, query_ids, budget, visual="learned"):
        """Kept set under the optional budget.  visual="learned": text and captures are selected together;
        visual="fixed": the capture packet is fixed (text-only stage) and only text is selected."""
        if visual == "fixed":
            kept, used = A.fixed_packet(pw, query_ids, budget)
            units = [u.key for u in pw.candidates(query_ids, "text")]
        else:
            kept, used = [], 0
            units = [u.key for u in pw.candidates(query_ids)]
        if not units:
            return kept, {}
        logits, keys = self(build_chunks(pw, tok, query_ids, budget, units), budget, pw)
        p = torch.sigmoid(logits).tolist()
        heads = set(pw.base_heads(query_ids)) | {pw.units[k].head for k in kept}
        for k, _ in sorted(zip(keys, p), key=lambda kp: (-kp[1], kp[0])):
            c = pw.unit_cost(pw.units[k], heads)
            if used + c <= budget:
                kept.append(k)
                used += c
                heads.add(pw.units[k].head)
        return kept, dict(zip(keys, p))


def soft_targets(record, tau=0.02):
    """Boltzmann weights over the evaluated within-budget sets and the induced inclusion probabilities."""
    sets = [s for s in record["evaluated_sets"] if s["cost"] <= record["budget"]]
    losses = torch.tensor([s["loss"] for s in sets], dtype=torch.float64)
    w = torch.softmax(-losses / tau, 0)
    pool = record["pool"]
    incl = {k: 0. for k in pool}
    for s, wi in zip(sets, w.tolist()):
        for i in s["keep"]:
            incl[pool[i]] += wi
    return incl, {"sets": len(sets), "max_weight": float(w.max()) if len(w) else None}


def selector_loss(logits, keys, targets):
    """Cross-entropy of p_keep against soft inclusion targets on the labelled (pool) units."""
    idx = [i for i, k in enumerate(keys) if k in targets]
    if not idx:
        return None
    y = torch.tensor([targets[keys[i]] for i in idx], device=logits.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits[idx], y)


def ranking_loss(logits, keys, effects, margin=0.005):
    """Pairwise logistic ranking of measured removal effects: a unit or block whose removal raises the
    predictor's loss more should score higher (block score = mean member logit).  Blocks include
    complementary groups (a capture with the records of the entities it shows)."""
    pos = {k: i for i, k in enumerate(keys)}
    scored = []
    for e in effects:
        idx = [pos[k] for k in e["members"] if k in pos]
        if idx:
            scored.append((logits[idx].mean(), e["delta"]))
    terms = [F.softplus(-(si - sj)) for (si, di), (sj, dj) in
             ((a, b) for a in scored for b in scored) if di > dj + margin]
    return torch.stack(terms).mean() if terms else None


def emit(args):
    """Selector choices on random windows of the training pool (input to predictor adaptation, §9 step 4)."""
    import json
    import random
    from Dataset.world_model.unified import data as D
    model = Selector()
    model.load_state_dict(torch.load(args.selector, map_location="cpu"), strict=False)
    model.eval()
    enc = D.Encoder()
    episodes = D.parents() if args.all_parents else D.splits()["fit"]
    rng = random.Random(args.seed)
    # One file per shard (out.NN) so parallel shards never interleave lines; train_predictor reads them all.
    out = open(f"{args.out}.{args.shard:02d}" if args.shards > 1 else args.out, "a")
    for i, episode in enumerate(D.cover_episodes(episodes, args.windows, rng)):
        origin, total = rng.choice(D.ORIGINS), rng.choice(args.budgets)
        qrng = random.Random(rng.random())
        if i % args.shards != args.shard:
            continue
        w, pw = enc.window(episode, origin)
        qids = pw.select_queries(qrng)
        budget = enc.optional_budget(w, pw, qids, total)
        if args.visual == "fixed":
            enc.set_relevance(w, pw, qids)
        kept, _ = model.select(pw, enc.tok, qids, budget, visual=args.visual)
        out.write(json.dumps({"episode": episode, "origin": origin, "budget": total, "optional_budget": budget,
                              "query_ids": qids, "inclusion": pw.inclusion, "kept": kept}) + "\n")
        out.flush()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="emit selector choices for predictor adaptation")
    p.add_argument("--selector", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=int, default=2000)
    p.add_argument("--budgets", type=int, nargs="+", default=[A.TOTAL_BUDGET], help="total budgets")
    p.add_argument("--visual", choices=("learned", "fixed"), default="learned")
    p.add_argument("--all-parents", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    emit(p.parse_args())
