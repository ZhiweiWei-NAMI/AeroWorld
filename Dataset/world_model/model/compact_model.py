"""Small numerical state-distribution predictor and its forecasting baselines.

Per (entity, cutoff, horizon) the network reads the causal feature vector of
compact_data.cell_features plus category/activity/horizon embeddings and
returns, for each pose component (position xyz, velocity xyz), the same law as
the Qwen head: P(anchor atom) and a Gaussian off the atom with mean
``anchor + offset*unit`` and std ``unit*(softplus(raw+log(e-1)) + 0.05)``,
``unit`` = TRAIN off-atom residual rate of the entity category x elapsed time.
Training minimizes the censored NLL (+ optional unit-scaled CRPS).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from Dataset.world_model.model.compact_data import (CATEGORIES, HORIZONS, anchor_labels, cell_features, load_episode)
from Dataset.world_model.model.distributions import GAUSSIAN_SCALE_FLOOR, UNIT_SCALE_OFFSET

COMPONENTS = (("pose.position_enu_m", 0), ("pose.position_enu_m", 1), ("pose.position_enu_m", 2),
              ("pose.velocity_enu_mps", 0), ("pose.velocity_enu_mps", 1), ("pose.velocity_enu_mps", 2))
HISTORY_TICKS = 50
MIN_GROUP_RESIDUALS = 50


def _episode_cells(path, cutoffs, activity_vocab):
    a = load_episode(Path(path))
    local_to_global = np.array([activity_vocab.get(v, len(activity_vocab)) for v in a.activity_vocab]+[len(activity_vocab)])
    out = {k: [] for k in ("X", "category", "activity", "horizon", "anchor", "truth", "atom", "band", "elapsed",
                           "cutoff", "entity")}
    names = None
    T = a.present.shape[0]
    for c in cutoffs:
        if c < HISTORY_TICKS or c+min(HORIZONS) >= T:
            continue
        entities = np.where(a.present[c-HISTORY_TICKS:c+1].any(0))[0]
        if not len(entities):
            continue
        X, names = cell_features(a, c, entities)
        act = a.activity[c, entities]
        act = local_to_global[np.where(act >= 0, act, len(a.activity_vocab))]
        for hi, h in enumerate(HORIZONS):
            lab = anchor_labels(a, c, entities, h)
            keep = lab["valid"]
            if not keep.any():
                continue
            out["X"].append(X[keep]); out["category"].append(a.category[entities][keep]); out["activity"].append(act[keep])
            out["horizon"].append(np.full(keep.sum(), hi))
            for key in ("anchor", "truth", "atom", "band", "elapsed"):
                out[key].append(lab[key][keep])
            out["cutoff"].append(np.full(keep.sum(), c)); out["entity"].append(np.array(a.entities, dtype=object)[entities][keep])
    result = {k: np.concatenate(v) if v else None for k, v in out.items()}
    result["episode"] = np.full(len(result["horizon"]) if result["horizon"] is not None else 0, a.episode, dtype=object)
    return result, names


def build_cells(arrays_dir, episodes, cutoffs_of, activity_vocab, workers=12):
    """Rows for every (episode, cutoff, entity, horizon) with a present target."""
    jobs = [(str(Path(arrays_dir)/f"{e}.npz"), cutoffs_of(e), activity_vocab) for e in episodes]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_episode_cells, *zip(*jobs)))
    names = next(n for _, n in parts if n is not None)
    cells = {k: np.concatenate([p[k] for p, _ in parts if p[k] is not None]) for k in parts[0][0]}
    return cells, names


def activity_vocabulary(arrays_dir, episodes):
    vocab = {}
    for e in episodes:
        for v in load_episode(Path(arrays_dir)/f"{e}.npz").activity_vocab:
            vocab.setdefault(v, len(vocab))
    return vocab


def fit_units(cells):
    """TRAIN off-atom residual rate per (category, component); pooled fallback."""
    rates = np.zeros((len(CATEGORIES), len(COMPONENTS)))
    residual = (cells["truth"]-cells["anchor"])/cells["elapsed"][:, None]
    off = ~cells["atom"]
    for j in range(len(COMPONENTS)):
        pooled = math.sqrt(np.mean(residual[off[:, j], j]**2))
        for c in range(len(CATEGORIES)):
            mask = off[:, j] & (cells["category"] == c)
            rates[c, j] = math.sqrt(np.mean(residual[mask, j]**2)) if mask.sum() >= MIN_GROUP_RESIDUALS else pooled
    return rates


def symmetric_xy(rates):
    """Equal x/y units (the D4 augmentation maps x and y onto each other)."""
    rates = rates.copy()
    for a, b in ((0, 1), (3, 4)):
        rates[:, a] = rates[:, b] = np.sqrt((rates[:, a]**2+rates[:, b]**2)/2)
    return rates


def feature_symmetry(names):
    """Index pairs of horizontal (x, y) feature components and the yaw-rate features (odd under reflection)."""
    index = {n: i for i, n in enumerate(names)}
    pairs = []
    for n in names:
        if n.endswith("[0]") and n[:-3]+"[1]" in index and (n.startswith(("velocity", "accel_lag", "cv_residual_lag"))
                                                           and not n.endswith("_ok[0]")):
            pairs.append((index[n], index[n[:-3]+"[1]"]))
        if n.startswith("neighbour") and n.endswith(("[0]", "[3]")):
            base, k = n[:-3], int(n[-2])
            pairs.append((index[n], index[f"{base}[{k+1}]"]))
    odd = [i for n, i in index.items() if n.startswith("yaw_rate")]
    return np.array(pairs), np.array(odd)


def symmetric_scaling(X, names):
    """Feature mean/std with x/y pairs centred at 0 and sharing one std (rotation-consistent)."""
    mean, std = X.mean(0), X.std(0)+1e-6
    pairs, odd = feature_symmetry(names)
    for a, b in pairs:
        mean[a] = mean[b] = 0.
        std[a] = std[b] = np.sqrt((X[:, a].astype(np.float64)**2+X[:, b].astype(np.float64)**2).mean()/2)+1e-6
    mean[odd] = 0.
    return mean, std


def d4_transform(X, anchor, truth, atom, transform, names):
    """Apply one of 8 planar symmetries (rotation by k*90 deg, then optional y-reflection) per row.

    Exact on the 1e-6 grid: atoms map onto atoms (a 90-degree turn swaps the x
    and y atom labels; a reflection negates y).
    """
    k, reflect = transform % 4, transform >= 4
    c = np.array([1, 0, -1, 0])[k]; s_ = np.array([0, 1, 0, -1])[k]
    sign = np.where(reflect, -1., 1.)
    def rotate(x, y):
        return c*x-s_*y, sign*(s_*x+c*y)
    X = X.copy(); anchor = anchor.copy(); truth = truth.copy(); atom = atom.copy()
    pairs, odd = feature_symmetry(names)
    for a, b in pairs:
        X[:, a], X[:, b] = rotate(X[:, a], X[:, b])
    X[:, odd] *= sign[:, None]
    for a, b in ((0, 1), (3, 4)):
        anchor[:, a], anchor[:, b] = rotate(anchor[:, a], anchor[:, b])
        truth[:, a], truth[:, b] = rotate(truth[:, a], truth[:, b])
        swap = k % 2 == 1
        atom[swap, a], atom[swap, b] = atom[swap, b].copy(), atom[swap, a].copy()
    return X, anchor, truth, atom


def cell_units(cells, rates):
    return rates[cells["category"]]*cells["elapsed"][:, None]


class CompactStatePredictor(nn.Module):
    def __init__(self, features, activities, hidden=256, depth=3, dropout=0., mixture=1, micro=False):
        super().__init__()
        self.mixture, self.micro = mixture, micro
        self.category = nn.Embedding(len(CATEGORIES), 16)
        self.activity = nn.Embedding(activities+1, 16)
        self.horizon = nn.Embedding(len(HORIZONS), 8)
        self.input = nn.Linear(features+40, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2*hidden), nn.GELU(),
                                                   nn.Dropout(dropout), nn.Linear(2*hidden, hidden)) for _ in range(depth)])
        self.out = nn.Linear(hidden, len(COMPONENTS)*(1+3*mixture+int(micro)))
        # Zero output: P(atom)=0.5, components centred on the anchor (the untrained reference law).
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, category, activity, horizon):
        h = self.input(torch.cat((x, self.category(category), self.activity(activity), self.horizon(horizon)), -1))
        for block in self.blocks:
            h = h+block(h)
        return self.out(h).view(-1, len(COMPONENTS), 1+3*self.mixture+int(self.micro))


# |gate logit| <= GATE_LIMIT: P(atom) in [3.4e-4, 0.99966], so one surprising
# cell costs at most ~8 nats instead of an unbounded penalty.
GATE_LIMIT = 8.


def law_terms(raw, floor=GAUSSIAN_SCALE_FLOOR, unit=None, micro=0.):
    """Atom + K-Gaussian mixture in unit space.

    Returns (log P(atom), log P(off), log weights [..,K], offsets [..,K], stds
    [..,K]).  Component stds increase with k (each adds a softplus to the
    previous), so a wider "tail" component can hedge rare manoeuvres that a
    single Gaussian could only cover by widening every forecast.  With
    ``micro`` > 0 (physical units) the last raw entry weighs an extra
    zero-offset component of fixed std ``micro``: sub-millimetre deviations
    just outside the exact atom band then need no wide off-atom spread.
    """
    gate = GATE_LIMIT*torch.tanh(raw[..., 0]/GATE_LIMIT)
    body = raw[..., 1:-1] if micro else raw[..., 1:]
    params = body.unflatten(-1, (-1, 3))
    offsets, scale_raw, weight_raw = params.unbind(-1)
    increments = F.softplus(scale_raw+UNIT_SCALE_OFFSET)
    scales = torch.cumsum(increments, -1)+floor
    if micro:
        weight_raw = torch.cat((weight_raw, raw[..., -1:]), -1)
        offsets = torch.cat((offsets, torch.zeros_like(offsets[..., :1])), -1)
        scales = torch.cat((scales, (micro/unit).unsqueeze(-1).to(scales.dtype)), -1)
    return F.logsigmoid(gate), F.logsigmoid(-gate), torch.log_softmax(weight_raw, -1), offsets, scales


def _expected_abs(d, s):
    z = d/s
    return 2*s*torch.exp(-z*z/2)/math.sqrt(2*math.pi)+d*(2*torch.special.ndtr(z)-1)


def atom_mixture_crps(p, log_w, offsets, scales, value):
    """CRPS of p*delta_0 + (1-p)*sum_k w_k N(o_k, s_k) at ``value`` (unit space)."""
    q, w = 1-p, log_w.exp()
    y = value.unsqueeze(-1)
    first = p*value.abs()+q*(w*_expected_abs(offsets-y, scales)).sum(-1)
    cross = 2*p*q*(w*_expected_abs(offsets, scales)).sum(-1)
    pair = (w.unsqueeze(-1)*w.unsqueeze(-2)*_expected_abs(offsets.unsqueeze(-1)-offsets.unsqueeze(-2),
                                                            torch.sqrt(scales.unsqueeze(-1)**2+scales.unsqueeze(-2)**2))).sum((-1, -2))
    return first-.5*(cross+q*q*pair)


def compact_loss(raw, batch, crps_weight=0., floor=GAUSSIAN_SCALE_FLOOR, micro=0.):
    """Censored atom + mixture NLL (unit space; physical NLL = this + log unit on off cells) and unit-scaled CRPS."""
    log_p, log_off, log_w, offsets, scales = law_terms(raw, floor, batch["unit"], micro)
    z = batch["residual_units"].unsqueeze(-1)
    band = batch["band_units"].unsqueeze(-1)
    log_density = torch.logsumexp(log_w+Normal(offsets, scales).log_prob(z), -1)
    log_complement = torch.logsumexp(log_w+torch.logaddexp(torch.special.log_ndtr((-band-offsets)/scales),
                                                           torch.special.log_ndtr((offsets-band)/scales)), -1)
    off_nll = -log_off-log_density+log_complement
    nll = torch.where(batch["atom"], -log_p, off_nll)
    loss = nll.mean()
    terms = {"nll_units": loss.detach(), "gate_nll": torch.where(batch["atom"], -log_p, -log_off).mean().detach()}
    if crps_weight:
        target = torch.where(batch["atom"], torch.zeros_like(batch["residual_units"]), batch["residual_units"])
        crps = atom_mixture_crps(log_p.exp(), log_w, offsets, scales, target).mean()
        loss = loss+crps_weight*crps
        terms["crps_units"] = crps.detach()
    return loss, terms


FEATURE_CLIP = 5.


def batch_tensors(cells, rows, units, mean, std, device, transform=None, names=None):
    X, anchor, truth, atom = cells["X"][rows], cells["anchor"][rows], cells["truth"][rows], cells["atom"][rows]
    if transform is not None:
        X, anchor, truth, atom = d4_transform(X, anchor, truth, atom, transform, names)
    # Standardized and clipped: heavy-tailed context features cannot saturate the net.
    x = np.clip((X-mean)/std, -FEATURE_CLIP, FEATURE_CLIP)
    unit = units[rows]
    residual = (truth-anchor)/unit
    return {"x": torch.tensor(x, dtype=torch.float32, device=device),
            "category": torch.tensor(cells["category"][rows], device=device),
            "activity": torch.tensor(cells["activity"][rows], device=device),
            "horizon": torch.tensor(cells["horizon"][rows], device=device),
            "unit": torch.tensor(unit, dtype=torch.float32, device=device),
            "residual_units": torch.tensor(residual, dtype=torch.float32, device=device),
            "band_units": torch.tensor(cells["band"][rows]/unit, dtype=torch.float32, device=device),
            "atom": torch.tensor(atom, device=device)}


def mixture_median(anchor, p, w, mu, sigma, iterations=90):
    """Median of p*delta_anchor + (1-p)*sum_k w_k N(mu_k, sigma_k) (float64, bisection)."""
    def cdf(x):
        return p*(x >= anchor)+(1-p)*(w*torch.special.ndtr((x.unsqueeze(-1)-mu)/sigma)).sum(-1)
    below = p.new_zeros(p.shape)+(1-p)*(w*torch.special.ndtr((anchor.unsqueeze(-1)-mu)/sigma)).sum(-1)
    at_anchor = (below <= .5) & (below+p >= .5)
    lo = torch.minimum((mu-12*sigma).min(-1).values, anchor)-1.
    hi = torch.maximum((mu+12*sigma).max(-1).values, anchor)+1.
    for _ in range(iterations):
        mid = (lo+hi)/2
        upper = cdf(mid) >= .5
        hi, lo = torch.where(upper, mid, hi), torch.where(upper, lo, mid)
    return torch.where(at_anchor, anchor, hi)


def prediction_rows(cells, rows, terms, units, run_name):
    """Paired-evaluation rows for the given cells, float64 physical units; ``terms`` from law_terms."""
    log_p, _, log_w, offsets, scales = terms
    anchor = torch.tensor(cells["anchor"][rows], dtype=torch.float64)
    unit = torch.tensor(units[rows], dtype=torch.float64)
    p = torch.as_tensor(log_p, dtype=torch.float64).exp()
    w = torch.as_tensor(log_w, dtype=torch.float64).exp()
    mu = anchor.unsqueeze(-1)+torch.as_tensor(offsets, dtype=torch.float64)*unit.unsqueeze(-1)
    sigma = torch.as_tensor(scales, dtype=torch.float64)*unit.unsqueeze(-1)
    band = torch.tensor(cells["band"][rows], dtype=torch.float64)
    mean = p*anchor+(1-p)*(w*mu).sum(-1)
    median = mixture_median(anchor, p, w, mu, sigma)
    out = []
    for n, r in enumerate(rows):
        for j, (field, component) in enumerate(COMPONENTS):
            components = [[float(w[n, j, k]), float(mu[n, j, k]), float(sigma[n, j, k])] for k in range(w.shape[-1])]
            out.append({"episode": cells["episode"][r], "entity": cells["entity"][r], "field": field, "component": component,
                        "window_cutoff": int(cells["cutoff"][r]), "horizon_ticks": HORIZONS[cells["horizon"][r]],
                        "law": "anchor_inflated_normal", "model": run_name,
                        "parameters": {"anchor": float(anchor[n, j]), "anchor_probability": float(p[n, j]),
                                       "location": components[0][1], "scale": components[0][2], "components": components,
                                       "mixture_median": float(median[n, j]), "mixture_mean": float(mean[n, j]),
                                       "atom_band": float(band[n, j])},
                        "truth": float(cells["truth"][r, j]), "truth_atom": bool(cells["atom"][r, j])})
    return out
