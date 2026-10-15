"""0.8B value selector: frozen query-conditioned record states + a small head fit to measured 2B losses.

features: the 0.8B reads the task prefix (budget, roster, queried relations, queries, fixed block) and every
candidate record once (packed chunks continuing from the prefix state); the final hidden state at each record's
last token is cached.
fit: per-record values v_k from [state, metadata, window summary]; the additive set value -sum_{k in S} v_k is fit
to the within-window centered measured losses of the collected contexts (value.collect).
Deployment ranks label-free candidate contexts (both anchors and members of the training exploration family) by the
predicted F and keeps the best. (Greedily optimizing F itself was measured worse than recency: the critic's
record-level marginals do not transfer outside the sampled family.)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from Dataset.world_model.unified import data as D, windows as W
from Dataset.world_model.unified.value import META, candidate_table

CHUNK = 8192


def window_summary(pw, qids, budget, keys):
    text = sum(len(pw.units[k].ids) for k in keys if pw.units[k].kind != "capture")
    visual = sum(len(pw.units[k].ids) + pw.units[k].extra for k in keys if pw.units[k].kind == "capture")
    return [math.log1p(budget) / 10., math.log1p(text) / 13., math.log1p(visual) / 13., math.log1p(len(keys)) / 9.,
            budget / max(1., text + visual), len(qids) / 100.]


@torch.no_grad()
def record_states(selector, pw, tok, qids, budget, actions_ids, keys):
    """[n, d] final hidden state at each candidate's last token (bf16 backbone, fp16 output)."""
    from Dataset.world_model.unified.selector import PLACEHOLDER, prefix_ids
    selector.eval()
    prefix = prefix_ids(pw, tok, qids, budget, actions_ids)
    states = selector.prefix_states(prefix)
    placeholder = tok.convert_tokens_to_ids(PLACEHOLDER)
    out = torch.zeros(len(keys), selector.text.config.hidden_size, dtype=torch.float16)

    def flush(ids, marks, inserts, present):
        if not ids:
            return
        summaries = {k: (pw.units[k].summary, pw.units[k].capture.modalities) for _, k in inserts}
        h = selector.encode_chunk(states, len(prefix), ids, inserts, summaries)
        at = torch.tensor([j for _, j in marks], device=h.device)
        out[[i for i, _ in marks]] = h[at].half().cpu()

    ids, marks, inserts, present = [], [], [], set()
    for i, k in enumerate(keys):
        u = pw.units[k]
        rec = list(pw.selector_ids(u, qids))
        insert = None
        if u.kind == "capture":
            insert = len(rec) - 1
            rec = rec[:-1] + [placeholder] + rec[-1:]
        head = [] if u.head in present else list(pw.headings[u.head])
        if len(ids) + len(head) + len(rec) > CHUNK:
            flush(ids, marks, inserts, present)
            ids, marks, inserts, present = [], [], [], set()
            head = list(pw.headings[u.head])
        ids += head
        present.add(u.head)
        if insert is not None:
            inserts.append((len(ids) + insert, k))
        ids += rec
        marks.append((i, len(ids) - 1))
    flush(ids, marks, inserts, present)
    return out


def head_ids(meta):
    """Candidates are sorted by (head, newest first): a head starts where the within-head rank is zero."""
    meta = torch.as_tensor(meta)
    return torch.cumsum((meta[:, META.index("head_rank")] == 0).long(), 0) - 1


GROUPS = ("position", "latency_ms", "cpu_utilization_pct", "predicate", "first_change", "event", "global_event",
          "scheduled")


class Head(nn.Module):
    """Per record and target group g: strength b_kg >= 0, a head-weight logit and an additive term c_kg. The predicted
    NLL of group g for a kept set S is -F_g(S), F_g(S) = sum_h a_hg log(1 + sum_{k in S, head h} b_kg) +
    sum_{k in S} c_kg, a_hg = softplus(mean alpha_kg over the head's candidates): records of one entity/relation have
    diminishing returns. groups=1 fits the window loss directly; groups=len(GROUPS) fits every query group's NLL and
    predicts the window loss as their mean over the groups present (as the window loss and the DEV score weigh them)."""
    def __init__(self, d=1024, meta=len(META), summary=6, width=256, use_states=True, groups=1):
        super().__init__()
        self.use_states, self.groups = use_states, groups
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width)) if use_states else None
        self.mlp = nn.Sequential(nn.Linear(width * use_states + meta + summary, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3 * groups))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, states, meta, summary):
        parts = [meta, summary.expand(len(meta), -1)]
        if self.use_states:
            parts.insert(0, self.proj(states.float()))
        alpha, b, c = self.mlp(torch.cat(parts, 1)).view(len(meta), 3, self.groups).unbind(1)
        hid = head_ids(meta).to(meta.device)
        n = int(hid.max()) + 1
        count = torch.zeros(n, 1, device=meta.device).index_add(0, hid, torch.ones_like(alpha[:, :1]))
        a = nn.functional.softplus(torch.zeros(n, self.groups, device=meta.device).index_add(0, hid, alpha) / count)
        return a * 1e-2, nn.functional.softplus(b), c * 1e-3, hid


def set_value(scores, membership):
    """F [m, groups] for each row of membership [m, n] (0/1)."""
    a, b, c, hid = scores
    sums = torch.zeros(len(membership), len(a), b.shape[1], device=membership.device, dtype=b.dtype).index_add(
        1, hid, membership[:, :, None] * b[None])
    return (a[None] * torch.log1p(sums)).sum(1) + membership @ c


def total(values, present):
    """Window-level F: the mean over the groups present (a single-group head is the window loss itself)."""
    return values[:, 0] if values.shape[1] == 1 else values[:, present].mean(1)


def present_groups(pw, qids):
    """Target groups of the query set (label-free)."""
    groups = {pw.w["queries"][i]["group"] for i in qids}
    return torch.tensor([g in groups for g in GROUPS])


def predicted_values(keys, scores, options, present=None):
    index = {k: i for i, k in enumerate(keys)}
    rows = torch.zeros(len(options), len(keys), dtype=torch.float64)
    for m, kept in enumerate(options.values()):
        rows[m, [index[k] for k in kept]] = 1.
    values = 0.
    for a, b, c, hid in scores if isinstance(scores, list) else [scores]:
        values = values + total(set_value((a.double().cpu(), b.double().cpu(), c.double().cpu(), hid.cpu()), rows),
                                present)
    return dict(zip(options, (values / (len(scores) if isinstance(scores, list) else 1)).tolist()))


def proposals(pw, qids, budget, keys, meta, n=12):
    """Label-free candidate contexts for ranking: both anchors and n members of the training exploration family,
    seeded by the window and query set (deployment never sees labels)."""
    from Dataset.world_model.unified import assemble as A
    from Dataset.world_model.unified.value import random_contexts
    rng = random.Random(W.stable_hash("value-rank", pw.w["episode"], pw.w["origin"], tuple(qids)))
    out = {"recency": A.select(pw, qids, budget, "recency")[0], "relevance": A.select(pw, qids, budget, "relevance")[0]}
    out.update((name, kept) for name, kept, _ in random_contexts(pw, qids, budget, keys, meta, rng, n))
    return out


class ValueSelector:
    """Deployable selector: frozen 0.8B record states, an ensemble of fitted heads averaged in F, and ranking of
    `proposals` label-free candidate contexts (plus both anchors)."""
    def __init__(self, selector, heads, proposals=12):
        self.selector, self.heads, self.n = selector, heads, proposals

    @classmethod
    def load(cls, path, device="cuda"):
        from Dataset.world_model.unified.selector import Selector
        state = torch.load(path, map_location="cpu", weights_only=False)
        heads = []
        for weights in state["heads"] if "heads" in state else [state["head"]]:
            head = Head(**state["head_config"])
            head.load_state_dict(weights)
            heads.append(head.to(device).eval())
        selector = Selector.load(state["backbone"], device) if heads[0].use_states else None
        return cls(selector, heads, state.get("proposals", 12))

    @torch.no_grad()
    def values(self, pw, tok, qids, budget, actions_ids, keys, meta, states=None):
        """Score tuples of every head (record states are computed once)."""
        device = next(self.heads[0].parameters()).device
        if not self.heads[0].use_states:
            states = None
        elif states is None:
            states = record_states(self.selector, pw, tok, qids, budget, actions_ids, keys)
        states = None if states is None else states.to(device)
        summary = torch.tensor(window_summary(pw, qids, budget, keys), device=device)[None]
        return [head(states, torch.as_tensor(meta, device=device), summary) for head in self.heads]

    def eval(self):
        return self

    def select(self, pw, tok, qids, budget, actions_ids):
        keys, meta = candidate_table(pw, qids)
        options = proposals(pw, qids, budget, keys, meta, self.n)
        predicted = predicted_values(keys, self.values(pw, tok, qids, budget, actions_ids, keys, meta), options,
                                     present_groups(pw, qids))
        return options[max(predicted, key=predicted.get)], {"positions": len(keys)}


def load_selector(path):
    """The deployable selector of a fitted value-head checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "head_config" not in state:
        raise ValueError(f"{path} is not a value-selector checkpoint (fit one with run.py fit-selector)")
    return ValueSelector.load(path)


def save_features(path, pw, qids, budget, keys, meta, states):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"keys": keys, "states": states, "meta": torch.from_numpy(meta),
                "summary": torch.tensor(window_summary(pw, qids, budget, keys)),
                "costs": torch.tensor([len(pw.units[k].ids) + pw.units[k].extra for k in keys])}, tmp)
    tmp.replace(path)


def features(args):
    """Cache record states, metadata and window summary for every collected window of this shard."""
    from Dataset.world_model.unified.selector import Selector
    from Dataset.world_model.unified.supervise import scoring_window
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selector = Selector.load(args.backbone) if args.backbone != "none" else None
    enc = D.Encoder()
    rows = [json.loads(l) for f in sorted(glob.glob(args.samples)) for l in open(f)]
    for row in rows[args.shard::args.shards]:
        job = row["job"]
        path = out / f"{job['id']}.pt"
        if path.exists():
            continue
        sc = scoring_window(None, enc, job, record={"query_ids": row["query_ids"], "inclusion": row["inclusion"]})
        keys, meta = candidate_table(sc.pw, sc.qids)
        states = None if selector is None else \
            record_states(selector, sc.pw, enc.tok, sc.qids, sc.budget, sc.actions_ids, keys)
        save_features(path, sc.pw, sc.qids, sc.budget, keys, meta, states)
        print(json.dumps({"job": job["id"], "records": len(keys)}), flush=True)


def load_windows(samples, feature_dir, use_states, state_dir=None):
    windows = []
    for f in sorted(glob.glob(samples)):
        for line in open(f):
            row = json.loads(line)
            path = Path(feature_dir) / f"{row['job']['id']}.pt"
            if not path.exists():
                continue
            feat = torch.load(path, weights_only=False)
            if use_states and state_dir is not None:
                spath = Path(state_dir) / path.name
                if not spath.exists():
                    continue
                cached = torch.load(spath, weights_only=False)
                if cached["keys"] != feat["keys"]:
                    raise ValueError(f"state cache key order differs: {spath}")
                feat["states"] = cached["states"]
            index = {k: i for i, k in enumerate(feat["keys"])}
            sets = [[index[k] for k in c["kept"]] for c in row["contexts"]]
            membership = torch.zeros(len(sets), len(index))
            for m, s in enumerate(sets):
                membership[m, s] = 1.
            windows.append({"job": row["job"], "episode": row["job"]["episode"],
                            "states": feat["states"] if use_states else None, "meta": feat["meta"],
                            "summary": feat["summary"][None], "membership": membership,
                            "loss": torch.tensor([c["loss"] for c in row["contexts"]], dtype=torch.float64),
                            "group_loss": torch.tensor([[c["groups"].get(g, 0.) for g in GROUPS]
                                                        for c in row["contexts"]], dtype=torch.float64),
                            "present": torch.tensor([g in row["contexts"][0]["groups"] for g in GROUPS]),
                            "names": [c["name"] for c in row["contexts"]]})
    return windows


def window_loss(head, w, device):
    """Within-window centred squared error of the predicted window loss (groups=1) or of every present group's NLL
    (averaged over the groups present); returns it and the predicted window loss of each context."""
    scores = head(w["states"].to(device) if w["states"] is not None else None, w["meta"].to(device),
                  w["summary"].to(device))
    predicted = -set_value(scores, w["membership"].to(device))
    if head.groups == 1:
        target = w["loss"].to(device).float()[:, None]
    else:
        present = w["present"].to(device)
        target, predicted = w["group_loss"].to(device).float()[:, present], predicted[:, present]
    err = (target - target.mean(0)) - (predicted - predicted.mean(0))
    return err.square().mean(), predicted.mean(1)


def evaluate(head, windows, device, per_window=False):
    """Centered MSE, within-window pairwise order accuracy, and the measured loss of the best-predicted context
    minus each anchor (picking among the measured contexts of the window)."""
    head.eval()
    mse, agree, pairs, rows = 0., 0, 0, []
    with torch.no_grad():
        for w in windows:
            loss, predicted = window_loss(head, w, device)
            mse += float(loss)
            p, t = predicted.cpu().double(), w["loss"]
            d = (t[:, None] - t[None]).abs() > 1e-3
            agree += int((((p[:, None] - p[None]) > 0) == ((t[:, None] - t[None]) > 0))[d].sum()) // 2
            pairs += int(d.sum()) // 2
            rel, rec = float(t[w["names"].index("relevance")]), float(t[w["names"].index("recency")])
            pick = float(t[int(p.argmin())])
            rows.append({"job": w["job"]["id"], "episode": w["episode"], "names": w["names"],
                         "predicted": p.tolist(), "loss": t.tolist(), "pick_minus_relevance": pick - rel,
                         "pick_minus_recency": pick - rec, "oracle_minus_relevance": float(t.min()) - rel,
                         "picked": w["names"][int(p.argmin())]})
    head.train()
    out = {"mse": mse / max(1, len(windows)), "pair_accuracy": agree / max(1, pairs), "windows": len(windows)}
    for k in ("pick_minus_relevance", "pick_minus_recency", "oracle_minus_relevance"):
        out[k] = float(np.mean([r[k] for r in rows])) if rows else 0.
    return (out, rows) if per_window else out


def train_head(train, valid, config, args, device):
    torch.manual_seed(args.seed)
    head = Head(**config).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: .5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    log, order = [], list(train)
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(order)
        total = 0.
        for w in order:
            loss, _ = window_loss(head, w, device)
            opt.zero_grad()
            (loss * 1e4).backward()
            opt.step()
            sched.step()
            total += float(loss)
        row = {"epoch": epoch, "train_mse": total / max(1, len(order))}
        if valid and (epoch % 4 == 3 or epoch == args.epochs - 1):
            row["valid"] = evaluate(head, valid, device)
        log.append(row)
        print(json.dumps(row), flush=True)
    return head, log


def fit(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    states = (args.states or args.features).split(",")
    windows = [w for sm, fd, sd in zip(args.samples.split(","), args.features.split(","), states)
               for w in load_windows(sm, fd, not args.metadata_only, sd)]
    config = {"use_states": not args.metadata_only, "width": args.width, "groups": len(GROUPS) if args.per_group else 1,
              "d": windows[0]["states"].shape[1] if not args.metadata_only else 1}
    group = D.profile if args.cv_key == "profile" else (lambda e: e)
    fold = lambda e: W.stable_hash("value-holdout", group(e)) % args.folds
    cv = []
    for k in range(args.folds if args.folds > 1 else 0):
        train = [w for w in windows if fold(w["episode"]) != k]
        valid = [w for w in windows if fold(w["episode"]) == k]
        head, _ = train_head(train, valid, config, args, device)
        cv += evaluate(head, valid, device, per_window=True)[1]
        if args.single_fold:
            break
    summary = {}
    if cv:
        for key in ("pick_minus_relevance", "pick_minus_recency", "oracle_minus_relevance"):
            x = np.array([r[key] for r in cv])
            summary[key] = [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))]
        summary["windows"] = len(cv)
        summary["picked"] = {n: sum(r["picked"] == n for r in cv) for n in sorted({r["picked"] for r in cv})}
        print(json.dumps({"cross_validation": summary}), flush=True)
    heads, log = [], []
    for e in range(args.ensemble):
        args.seed += e
        head, part = train_head(windows, [], config, args, device)
        args.seed -= e
        heads.append({k: v.cpu() for k, v in head.state_dict().items()})
        log.append(part)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_config": config, "heads": heads, "proposals": args.proposals,
                "backbone": args.backbone, "log": log, "args": vars(args), "cross_validation": summary,
                "cv_rows": cv, "windows": [w["job"]["id"] for w in windows]}, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("features")
    f.add_argument("--samples", required=True); f.add_argument("--out", required=True)
    f.add_argument("--backbone", required=True)
    f.add_argument("--shard", type=int, default=0); f.add_argument("--shards", type=int, default=1)
    t = sub.add_parser("fit")
    t.add_argument("--samples", required=True); t.add_argument("--features", required=True)
    t.add_argument("--out", required=True); t.add_argument("--backbone", required=True)
    t.add_argument("--states", help="directory of cached 0.8B record states (default: --features)")
    # --samples, --features and --states take comma-separated lists aligned by collection round
    t.add_argument("--metadata-only", action="store_true"); t.add_argument("--folds", type=int, default=5)
    t.add_argument("--single-fold", action="store_true", help="validate on one fold only (quick look)")
    t.add_argument("--cv-key", choices=("profile", "parent"), default="profile",
                   help="hold out whole scenario profiles (as DEV does) or parent episodes")
    t.add_argument("--epochs", type=int, default=20); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=.01); t.add_argument("--width", type=int, default=256)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ensemble", type=int, default=1, help="heads (seeds) of the deployed selector")
    t.add_argument("--per-group", action="store_true", help="fit every query group's NLL (window loss = their mean)")
    t.add_argument("--proposals", type=int, default=12, help="random candidate contexts ranked at deployment")
    args = p.parse_args()
    features(args) if args.command == "features" else fit(args)


if __name__ == "__main__":
    main()
