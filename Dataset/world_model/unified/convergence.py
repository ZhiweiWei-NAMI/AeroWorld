"""Convergence assessment and checkpoint selection for predictor runs.

Plots the training loss (raw and EMA-smoothed, total and per target group) and the fixed DEV metrics
(per-group marginal NLL with window-clustered standard errors and the group-averaged NLL) against the
step, over one or several chained runs (a continuation run counts its steps after its initial checkpoint).

Selection: the saved checkpoint with the lowest group-averaged DEV NLL.  Converged means the best value
improved by less than --tolerance (relative) over the last --patience evaluations and no group's DEV NLL
improved by more than two standard errors over that span.

  python -m Dataset.world_model.unified.convergence --runs dev_p3:0 --name dev_p3
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Dataset.world_model.unified import schema as S

RUNS = S.WORKSPACE/"predictor"
GROUPS = ("position", "latency_ms", "cpu_utilization_pct", "predicate", "first_change", "event", "global_event",
          "scheduled")


def DEV_KEY(r):
    """Per-group DEV NLL of a log row ("dev_recency_1024" in logs before the 20K total budget), or None."""
    return r.get("dev_group_nll") or r.get("dev_recency_1024")


def load(specs):
    """specs 'name:offset[:last_step]' -> training rows, dev rows and checkpoints with global steps."""
    train, dev, ckpts = [], [], {}
    for spec in specs:
        name, offset, *last = spec.split(":")
        offset, last = int(offset), (int(last[0]) if last else None)
        for line in open(RUNS/name/"log.jsonl"):
            r = json.loads(line)
            if last is not None and r["step"] > last:
                continue
            if not DEV_KEY(r) and "loss" not in r:
                continue                    # early-stop and other marker rows
            g = dict(r, step=r["step"] + offset, run=name)
            (dev if DEV_KEY(r) else train).append(g)
        for p in sorted((RUNS/name).glob("step_*.pt")):
            step = int(p.stem.split("_")[1])
            if last is None or step <= last:
                ckpts[step + offset] = str(p)
    return train, dev, ckpts


def ema(xs, alpha=0.05):
    out, m = [], None
    for x in xs:
        m = x if m is None else (1 - alpha) * m + alpha * x
        out.append(m)
    return out


def weighted(r):
    return r.get("dev_weighted_nll") or sum(DEV_KEY(r).values()) / len(DEV_KEY(r))


def assess(dev, ckpts, tolerance, patience):
    scored = [(r["step"], weighted(r), r) for r in dev if r["step"] in ckpts]
    if not scored:
        return None
    best = min(scored, key=lambda x: x[1])
    converged = None
    if len(scored) > patience:
        window = scored[-(patience + 1):]
        rel = (window[0][1] - min(x[1] for x in window[1:])) / window[0][1]
        group_gains = {}
        for g in DEV_KEY(window[0][2]):
            first, last = DEV_KEY(window[0][2])[g], DEV_KEY(window[-1][2]).get(g, DEV_KEY(window[0][2])[g])
            se = (window[-1][2].get("dev_se") or {}).get(g)
            group_gains[g] = {"gain": first - last, "se": se, "significant": bool(se and first - last > 2 * se)}
        converged = {"relative_improvement_last_span": rel, "tolerance": tolerance, "span_evaluations": patience,
                     "groups": group_gains,
                     "converged": rel < tolerance and not any(v["significant"] for v in group_gains.values())}
    return {"selected_step": best[0], "selected_checkpoint": ckpts[best[0]], "selected_dev_weighted_nll": best[1],
            "evaluations": [(s, w) for s, w, _ in scored], "convergence": converged}


def plot(train, dev, out, title):
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    steps = [r["step"] for r in train]
    ax = axes[0, 0]
    ax.plot(steps, [r["loss"] for r in train], alpha=.25, lw=.6)
    ax.plot(steps, ema([r["loss"] for r in train]), lw=1.5)
    ax.set_title("training loss (weighted, marginal+joint)")
    ax = axes[0, 1]
    ds = [r["step"] for r in dev]
    ax.plot(ds, [weighted(r) for r in dev], "o-")
    ax.set_title("DEV group-averaged NLL")
    ax = axes[0, 2]
    js = [r.get("joint_nll_per_query") for r in train]
    ax.plot(steps, ema([j or 0 for j in js]), lw=1.5)
    ax.set_title("training joint NLL per query (EMA)")
    ax = axes[0, 3]
    ax.plot(steps, ema([r.get("with_images", 0) for r in train], .02), lw=1.5)
    ax.plot(ds, [r.get("dev_windows_with_images", 0) for r in dev], "o")
    ax.set_ylim(-.05, 1.05)
    ax.set_title("fraction of items with an image packet (train EMA, DEV)")
    for k, g in enumerate(GROUPS):
        ax = axes[1 + k // 4, k % 4]
        tv = [(r["step"], r[g]) for r in train if g in r]
        if tv:
            ax.plot([s for s, _ in tv], ema([v for _, v in tv]), lw=1.2, label="train (EMA)")
        dv = [(r["step"], DEV_KEY(r)[g], (r.get("dev_se") or {}).get(g)) for r in dev if g in DEV_KEY(r)]
        if dv:
            ax.errorbar([s for s, _, _ in dv], [v for _, v, _ in dv], yerr=[2 * (e or 0) for _, _, e in dv],
                        fmt="o-", capsize=3, label="DEV ±2 SE")
        ax.set_title(f"{g} NLL")
        ax.legend(fontsize=7)
    for a in axes.flat:
        a.set_xlabel("step")
        a.grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--name", required=True, help="run directory that receives the plot and selection")
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=2)
    args = p.parse_args()
    train, dev, ckpts = load(args.runs)
    out = RUNS/args.name
    plot(train, dev, out/"convergence.png", " + ".join(args.runs))
    result = assess(dev, ckpts, args.tolerance, args.patience)
    (out/"selection.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
