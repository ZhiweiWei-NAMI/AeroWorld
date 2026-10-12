"""Equal-weight mixture ensemble of compact-predictor prediction files (same cells, same order).

The ensemble law is the average of the members' laws: P(atom) is the mean
member P(atom); the off-atom part is the union of the members' Gaussian
components, weighted by each member's off-atom mass.  Median by bisection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _cdf(x, a, p, comps):
    return p*(x >= a)+(1-p)*sum(w*.5*math.erfc(-(x-m)/(s*math.sqrt(2))) for w, m, s in comps)


def median(a, p, comps):
    below = (1-p)*sum(w*.5*math.erfc(-(a-m)/(s*math.sqrt(2))) for w, m, s in comps)
    if below <= .5 <= below+p:
        return a
    lo = min([a]+[m-12*s for _, m, s in comps])-1; hi = max([a]+[m+12*s for _, m, s in comps])+1
    for _ in range(90):
        mid = (lo+hi)/2
        lo, hi = (lo, mid) if _cdf(mid, a, p, comps) >= .5 else (mid, hi)
    return hi


def combine(rows):
    first = rows[0]; P0 = first["parameters"]; a = P0["anchor"]
    p = sum(r["parameters"]["anchor_probability"] for r in rows)/len(rows)
    q = 1-p
    comps = []
    for r in rows:
        P = r["parameters"]; qi = 1-P["anchor_probability"]
        for w, m, s in P["components"]:
            weight = qi*w/len(rows)/q if q > 0 else 1/len(rows)/len(P["components"])
            if weight > 1e-12:
                comps.append([weight, m, s])
    mean = p*a+q*sum(w*m for w, m, _ in comps)
    out = dict(first)
    out["model"] = "ensemble"
    out["parameters"] = {**P0, "anchor_probability": p, "components": comps, "location": comps[0][1], "scale": comps[0][2],
                         "mixture_mean": mean, "mixture_median": median(a, p, comps)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("members", type=Path, nargs="+")
    args = parser.parse_args()
    streams = [open(m) for m in args.members]
    with open(args.output, "w") as out:
        for lines in zip(*streams):
            rows = [json.loads(l) for l in lines]
            key = {(r["episode"], r["entity"], r["field"], r["component"], r["window_cutoff"], r["horizon_ticks"]) for r in rows}
            if len(key) != 1:
                raise ValueError("ensemble members are not aligned")
            out.write(json.dumps(combine(rows))+"\n")


if __name__ == "__main__":
    main()
