"""Predictor losses, DEV statistics and checkpoint helpers shared by finetune_predictor, supervise and evaluate."""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from Dataset.world_model.unified import codes as K, data as D


def item_losses(model, item, accum, train=True, modes=("marginal", "joint")):
    """Weighted marginal + joint training loss, and per-group marginal NLL (RESEARCH_PLAN §6, §9)."""
    loader, encs = D.visual_loader(item), item["encs"]
    per_query, total = [None] * len(encs), 0.
    scale = 1. / (len(modes) * accum)
    if "marginal" in modes:
        w = D.query_weights([e.group for e in encs], [e.inclusion for e in encs])
        ctx = model.build_context(item["segments"], loader)

        def loss_fn(i, lp):
            loss = K.query_loss(lp, encs[i])
            per_query[i] = float(loss)
            return scale * w[i] * loss
        if train:
            total += model.marginal_backward(ctx, [(e.ids, e.steps) for e in encs], loss_fn)
        else:
            lps = model.marginal(ctx, [(e.ids, e.steps) for e in encs])
            total += float(sum(loss_fn(i, lp) for i, lp in enumerate(lps)))
    joint_nll = None
    if "joint" in modes and item["joint"]:
        order = item["joint"]
        w = D.query_weights([encs[j].group for j in order], [encs[j].inclusion for j in order])
        ctx = model.build_context(item["segments"], loader)
        lps = model.joint(ctx, [(encs[j].full_ids, encs[j].steps) for j in order])
        losses = [K.query_loss(lp, encs[j]) for lp, j in zip(lps, order)]
        loss = scale * sum(wi * l for wi, l in zip(w, losses))
        if train:
            loss.backward()
        total += float(loss)
        joint_nll = sum(float(l) for l in losses)
    groups = defaultdict(list)
    for e, loss in zip(encs, per_query):
        if loss is not None:
            groups[e.group].append(loss)
    return total, {g: sum(v)/len(v) for g, v in groups.items()}, joint_nll


def quick_eval(rows):
    """Profile-averaged group NLL, counts and window-clustered SEs (RESEARCH_PLAN §9.1).

    ARM members of one pair share a cluster. Each profile has equal weight within a target group; its NLL is
    the per-query mean. The selection score is the mean over target groups, as in CONTRACT §5.
    """
    per_window = defaultdict(lambda: defaultdict(dict))
    for window, sums, _ in rows:
        for g, (total, n) in sums.items():
            clusters = per_window[g][D.profile(window[0])]
            old_total, old_n = clusters.get(window, (0., 0))
            clusters[window] = old_total + total, old_n + n
    if not per_window:
        raise ValueError("DEV set contains no scored target groups")
    out, counts, se = {}, {}, {}
    for g, profiles in per_window.items():
        means, variance, queries, clusters = [], 0., 0, 0
        for windows in profiles.values():
            rs = list(windows.values())
            total, n = sum(r[0] for r in rs), sum(r[1] for r in rs)
            mean = total / n
            means.append(mean)
            queries += n
            clusters += len(rs)
            variance += sum((r[0] - mean * r[1]) ** 2 for r in rs) / n ** 2
        out[g], counts[g] = sum(means) / len(means), queries
        se[g] = math.sqrt(clusters / (clusters - 1) * variance) / len(means) if clusters > 1 else None
    return out, counts, se, sum(out.values()) / len(out), sum(r[2] for r in rows) / len(rows)


def trainable_state(model):
    """LoRA and adapter parameters for checkpoints and full resume state (RESEARCH_PLAN §9.1)."""
    return {name: prm.detach().cpu() for name, prm in model.named_parameters() if prm.requires_grad}


def restore_trainable(model, state, path):
    """Require all requested trainable weights; base weights stay frozen (RESEARCH_PLAN §9)."""
    expected = {n for n, prm in model.named_parameters() if prm.requires_grad}
    absent = expected - state.keys()
    if absent:
        raise ValueError(f"missing trainable weights in {path}: {sorted(absent)[:5]}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected keys in {path}: {result.unexpected_keys[:5]}")


def load_trainable(model, path):
    """Load the complete LoRA and sensor adapter against the pinned backbone."""
    restore_trainable(model, torch.load(path, map_location="cpu", weights_only=True), path)
