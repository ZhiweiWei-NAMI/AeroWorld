"""Corpus passes: cached episode sources and the declared answer schema.

  --sources   read every parent's producer streams once (sources.load_episode) into
              runs/unified/sources/<episode>.pkl.zst with a per-episode audit
  --schema    derive schema-level facts from the cached sources and write
              configs/unified_schema.json:
                * event types and the participant categories they bind,
                * scheduled activation categories,
                * bounded numeric supports R_{f,c,h} (§4.3): per-category maximum
                  speed x horizon x 1.2 for position; the observed range for
                  latency and CPU utilisation (a physical, label-free bound per
                  field, not a per-window quantity).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
import pickle

import zstandard

from Dataset.world_model.data import capture_sources
from Dataset.world_model.unified import schema as S

SOURCES = S.WORKSPACE/"sources"
SCHEMA = S.ROOT/"Dataset/world_model/configs/unified_schema.json"
PREDICATES = S.ROOT/"Dataset/world_model/configs/unified_predicates.json"


def parents():
    return sorted(capture_sources.load()["parents"])


def save_source(episode):
    from Dataset.world_model.unified.sources import load_episode
    src = load_episode(episode)
    path = SOURCES/f"{episode}.pkl.zst"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(zstandard.ZstdCompressor(level=3).compress(pickle.dumps(src, protocol=5)))
    tmp.replace(path)
    return episode, src.audit


def load_source(episode):
    return pickle.loads(zstandard.ZstdDecompressor().decompress((SOURCES/f"{episode}.pkl.zst").read_bytes()))


def episode_facts(episode):
    src = load_source(episode)
    out = {"event_categories": defaultdict(set), "scheduled": set(), "speed": defaultdict(float),
           "range": defaultdict(lambda: [math.inf, -math.inf]), "events": defaultdict(int)}
    for ev in src.events:
        out["events"][ev.type] += 1
        for e in ev.entities:
            out["event_categories"][ev.type].add(src.category[e])
    for sc in src.scheduled:
        out["scheduled"].add(sc.category)
    for e, vel in src.velocity.items():
        cat = src.category[e]
        if cat in S.MOVING:
            for v in vel.values():
                out["speed"][cat] = max(out["speed"][cat], max(abs(c) for c in v) * S.VEL_EPS)
    for fam, (source, name, eps) in S.NUMERIC_TARGETS.items():
        if source == "frame":
            continue
        for (e, s, n), stream in src.fields.items():
            if s == source and n == name:
                for v in stream.samples.values():
                    if isinstance(v, int):
                        r = out["range"][fam]
                        r[0], r[1] = min(r[0], v*eps), max(r[1], v*eps)
    return episode, {k: dict(v) if isinstance(v, defaultdict) else v for k, v in out.items()}


def build_schema(facts):
    cats, sched, speed, rng, events = defaultdict(set), set(), defaultdict(float), {}, defaultdict(int)
    for _, f in facts:
        for t, c in f["event_categories"].items():
            cats[t] |= c
        for t, n in f["events"].items():
            events[t] += n
        sched |= f["scheduled"]
        for c, v in f["speed"].items():
            speed[c] = max(speed[c], v)
        for fam, (lo, hi) in f["range"].items():
            cur = rng.setdefault(fam, [math.inf, -math.inf])
            cur[0], cur[1] = min(cur[0], lo), max(cur[1], hi)
    support = {"position": {c: {str(h): round(1.2 * speed[c] * h * S.TICK_S + 0.5, 2) for h in S.HORIZONS}
                            for c in S.MOVING}}
    for fam, (lo, hi) in rng.items():
        support[fam] = {"*": {str(h): round(hi - lo, 4) for h in S.HORIZONS}}
    event_types = defaultdict(list)
    for t, c in sorted(cats.items()):
        for cat in sorted(c):
            event_types[cat].append(t)
    return {"rule": "support R_{f,c,h}: position = 1.2 x max per-axis speed of category c x horizon + 0.5 m; "
                    "latency/cpu = observed field range (residual cannot exceed it)",
            "max_axis_speed_mps": speed, "field_range": rng, "support": support,
            "event_types_by_category": event_types, "event_counts": events, "scheduled_categories": sorted(sched)}


def predicate_definitions():
    """Expression and governed threshold values of every registry predicate with a stated expression (the
    background block of a window lists the queried ones, assemble.py)."""
    from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates, get_governed_parameter_defaults
    defaults = get_governed_parameter_defaults()
    out = {}
    for t in get_core_predicate_templates():
        expr = t.get("expression")
        if not expr or "governed predicate value" in expr:
            continue        # communication/compute predicates are producer-governed values without an expression
        params = (t.get("state_contract") or {}).get("threshold_parameters") or []
        out[t["id"]] = {"expression": expr, "thresholds": {k: defaults[k] for k in params if k in defaults}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", action="store_true")
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--predicates", action="store_true", help="write configs/unified_predicates.json")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.sources:
        SOURCES.mkdir(parents=True, exist_ok=True)
        todo = [e for e in parents() if not (SOURCES/f"{e}.pkl.zst").exists()]
        audits = {}
        with ProcessPoolExecutor(args.workers) as pool:
            for episode, audit in pool.map(save_source, todo):
                audits[episode] = audit
                print(episode, json.dumps(audit)[:200], flush=True)
        audit_path = S.WORKSPACE/"audit/sources.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        prior = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        audit_path.write_text(json.dumps({**prior, **audits}, indent=1, default=list))
    if args.schema:
        with ProcessPoolExecutor(args.workers) as pool:
            facts = list(pool.map(episode_facts, parents()))
        SCHEMA.write_text(json.dumps(build_schema(facts), indent=1, default=sorted))
        print(SCHEMA.read_text()[:3000])
    if args.predicates:
        PREDICATES.write_text(json.dumps(predicate_definitions(), indent=1))


if __name__ == "__main__":
    main()
