"""Window construction: aliases, fixed block, candidate record groups, queries and labels (RESEARCH_PLAN §3-§5).

A window is the 5 s history (t0-50, t0] of one parent at origin t0.  Every
stream is compacted reconstructibly (compaction.py) and the reconstruction is
verified per window; failing streams keep all samples.  The result is stored
uncommitted to a query set: origin-state lines of the entities and tuples a
query set addresses become fixed at assembly time (assemble.py), all other
origin lines and every compacted history record are selection candidates.

Inputs contain only records at ticks <= t0.  Labels are read from ticks > t0.
"""
from __future__ import annotations

import hashlib
import math

from Dataset.world_model.unified import compaction as C
from Dataset.world_model.unified import schema as S

KNOWN = ("false", "true")


def decimals(eps):
    return max(0, -int(math.floor(math.log10(eps) + 1e-9)))


def num(k, eps):
    return f"{k*eps:.{decimals(eps)}f}"


def ts(tick, origin):
    return f"{(tick-origin)*S.TICK_S:.1f}".replace("-0.0", "0.0") if tick != origin else "0"


def vec(k, eps):
    return "[" + ",".join(num(c, eps) for c in k) + "]"


def stable_hash(*parts):
    return int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16)


def value_text(value, eps, al=None):
    if isinstance(value, int) and eps is not None:
        return num(value, eps)
    if al is not None and isinstance(value, str) and (value in al.src.objects or value in al.src.category):
        return al.ref(value)
    return str(value)


def family_of(episode):
    return episode.split("-")[0] if episode.startswith("L") else episode.split("_")[0]


class Aliases:
    def __init__(self, src, entities, origin):
        order = {c: i for i, c in enumerate(("uav", "vehicle", "pedestrian", "facility", "airspace_corridor"))}
        self.entity = {}
        counts = {}
        for e in sorted(entities, key=lambda e: (order.get(src.category[e], 9), src.category[e], e)):
            prefix = S.ALIAS_PREFIX.get(src.category[e], "e")
            counts[prefix] = counts.get(prefix, 0) + 1
            self.entity[e] = f"{prefix}{counts[prefix]}"
        self.src = src
        self.other, self.objects, self.counts = {}, {}, {}

    def ref(self, ref):
        """Alias of an entity, a unique owned object (owner.kindN, numbered by first reference) or a short name."""
        if ref in self.entity:
            return self.entity[ref]
        if ref in self.objects:
            return self.objects[ref]
        obj = self.src.objects.get(ref)
        if obj is not None:
            kind, owner = obj
            if owner in self.entity:
                if kind == "station":
                    return self.entity[owner]
                key = (owner, kind)
                self.counts[key] = self.counts.get(key, 0) + 1
                self.objects[ref] = f"{self.entity[owner]}.{kind}{self.counts[key]}"
                return self.objects[ref]
        if ref not in self.other:
            self.other[ref] = f"x{len(self.other)+1}"
        return self.other[ref]


def window_ticks(origin):
    return list(range(origin - S.HISTORY_TICKS, origin + 1))


def formal_ticks(origin):
    return list(range(origin - S.HISTORY_TICKS, origin + 1, S.FORMAL_STEP))


def last_at_or_before(samples, tick):
    best = None
    for t in range(tick, tick - S.HISTORY_TICKS - 1, -1):
        if samples.get(t) is not None:
            return t
    return best


def build_window(src, origin, visual=None, max_tuple_candidates=None):
    ticks, fticks = window_ticks(origin), formal_ticks(origin)
    start = ticks[0]
    in_window = {e for e, present in src.present.items() if any(start <= t <= origin for t in present)}
    present0 = {e for e in in_window if origin in set(src.present[e])}
    al = Aliases(src, in_window, origin)
    frame_o = None
    for e in sorted(present0, key=lambda e: al.entity[e]):
        if src.category[e] in S.MOVING and origin in src.position.get(e, {}):
            frame_o = tuple(round(c/100)*100 for c in src.position[e][origin])   # 1 m grid, in cm
            break
    frame_o = frame_o or (0, 0, 0)

    def rel_pos(p):
        return tuple(p[i] - frame_o[i] for i in range(3))

    groups, origin_lines, headings, failures = [], {}, {}, {}
    # ---- entities -------------------------------------------------------------------------
    for e in sorted(in_window, key=lambda e: al.entity[e]):
        a, cat = al.entity[e], src.category[e]
        ident = f"entity {a} {cat} type={src.kind[e]}"
        if cat not in S.MOVING and origin in src.position.get(e, {}):
            ident += f" p={vec(rel_pos(src.position[e][origin]), S.POS_EPS)}"
        headings[("entity", a)] = ident
        # Motion: velocity-hold knots.
        pos, vel = src.position.get(e, {}), src.velocity.get(e, {})
        if cat in S.MOVING and pos:
            knots, runs = C.motion_knots(pos, vel, ticks)
            if not C.verify_motion(pos, vel, ticks, knots, runs):
                failures[("motion", a)] = True
                knots = [(t, pos[t], vel[t]) for t in ticks if t in pos and t in vel]
            run_starts = {first for first, _ in runs}
            for t, p, v in knots:
                if t == origin:
                    continue
                enter = " enters" if t in run_starts and t != start else ""
                groups.append({"kind": "knot", "head": ("entity", a), "tick": t,
                               "line": f"  t={ts(t, origin)}{enter} p={vec(rel_pos(p), S.POS_EPS)} v={vec(v, S.VEL_EPS)}"})
            for first, last in runs:
                if last != origin:
                    groups.append({"kind": "leave", "head": ("entity", a), "tick": last+1,
                                   "line": f"  t={ts(last+1, origin)} leaves"})
        # Fields grouped per (source, tick); origin values collected per source.
        origin_vals = {"frame": [], "comm": [], "compute": []}
        changes = {}
        for source, step, grid in (("frame", 1, ticks), ("comm", S.FORMAL_STEP, fticks), ("compute", S.FORMAL_STEP, fticks)):
            names = sorted(n for (ee, s, n) in src.field_index.get(e, ()) if s == source)
            for name in names:
                stream = src.fields[(e, source, name)]
                samples = {t: stream.samples.get(t) for t in grid}
                if not any(v is not None for v in samples.values()):
                    continue
                eps = stream.eps
                recs = C.hold_last(samples, grid, step, origin)
                if not C.verify_hold(samples, grid, step, origin, recs):
                    failures[("field", a, source, name)] = True
                    recs = [("change", t, None, samples[t]) for t in grid if samples[t] is not None]
                v0 = samples.get(origin)
                if v0 is not None:
                    origin_vals[source].append((name, v0, eps))
                for r in recs:
                    sn = S.short(name)
                    if r[0] == "change":
                        txt = f"{sn} {value_text(r[2], eps, al) if r[2] is not None else '?'}->{value_text(r[3], eps, al)}"
                    elif r[0] == "appear":
                        txt = f"{sn} appears {value_text(r[2], eps, al)}"
                    else:
                        txt = f"{sn} ends {value_text(r[2], eps, al)}"
                    changes.setdefault((source, r[1]), []).append(txt)
        tags = {"frame": "", "comm": "link ", "compute": "compute "}
        for (source, t), items in sorted(changes.items()):
            groups.append({"kind": f"{source}_change", "head": ("entity", a), "tick": t,
                           "line": f"  t={ts(t, origin)} {tags[source]}" + "; ".join(items)})
        if e in present0:
            parts = []
            if cat in S.MOVING and origin in pos and origin in vel:
                parts.append(f"p={vec(rel_pos(pos[origin]), S.POS_EPS)} v={vec(vel[origin], S.VEL_EPS)}")
            flags = [S.short(n) for n, v, _ in origin_vals["frame"] if v == "true"]
            off = [S.short(n) for n, v, _ in origin_vals["frame"] if v == "false"]
            other = [f"{S.short(n)}={value_text(v, eps, al)}" for n, v, eps in origin_vals["frame"] if v not in ("true", "false")]
            parts += other
            # Recorded true and recorded false booleans are both listed: a boolean field in neither list was not
            # recorded at the origin (unknown), never an implicit false.
            if flags:
                parts.append("flags=[" + ",".join(flags) + "]")
            if off:
                parts.append("off=[" + ",".join(off) + "]")
            if parts:
                origin_lines[("entity", a, "frame")] = {"head": ("entity", a), "line": "  t=0 " + " ".join(parts)}
            for source in ("comm", "compute"):
                vals = origin_vals[source]
                if vals:
                    body = " ".join(f"{S.short(n)}={value_text(v, eps, al)}" for n, v, eps in vals)
                    origin_lines[("entity", a, source)] = {"head": ("entity", a), "line": f"  t=0 {tags[source]}{body}"}
    # ---- relations ------------------------------------------------------------------------
    rel_alias, rel_entities = {}, {}
    rels = [r for r in src.relations.values() if any(t in r.samples for t in fticks)]
    for i, rel in enumerate(sorted(rels, key=lambda r: (r.predicate, r.tuple_id))):
        alias = rel_alias[rel.tuple_id] = f"r{i+1}"
        args = ",".join(f"{role}={al.ref(v) if isinstance(v, str) else v}" for role, v in rel.bindings.items())
        headings[("rel", alias)] = f"{alias} = {rel.predicate}({args})"
        # Window entities a relation binds, directly or as the owner of a bound object (flow, task, queue, ...).
        rel_entities[alias] = sorted({al.entity[src.owner(v)] for v in rel.bindings.values()
                                      if isinstance(v, str) and src.owner(v) in al.entity})
        samples = {t: rel.samples.get(t) for t in fticks}
        recs = C.hold_last(samples, fticks, S.FORMAL_STEP, origin)
        if not C.verify_hold(samples, fticks, S.FORMAL_STEP, origin, recs):
            failures[("rel", alias)] = True
            recs = [("change", t, None, samples[t]) for t in fticks if samples[t] is not None]
        for r in recs:
            txt = {"change": lambda r: f"{r[2]}->{r[3]}", "appear": lambda r: f"appears {r[2]}",
                   "leave": lambda r: f"removed (was {r[2]})"}[r[0]](r)
            groups.append({"kind": "rel_change", "head": ("rel", alias), "tick": r[1],
                           "line": f"  {alias} t={ts(r[1], origin)} {txt}"})
        ev_changes, ev_origin = {}, []
        for path in sorted(rel.evidence):
            stream = rel.evidence[path]
            es = {t: stream.samples.get(t) for t in fticks}
            if not any(v is not None for v in es.values()):
                continue
            erecs = C.hold_last(es, fticks, S.FORMAL_STEP, origin)
            if not C.verify_hold(es, fticks, S.FORMAL_STEP, origin, erecs):
                failures[("evidence", alias, path)] = True
                erecs = [("change", t, None, es[t]) for t in fticks if es[t] is not None]
            if es.get(origin) is not None:
                ev_origin.append(f"{path}={value_text(es[origin], stream.eps, al)}")
            for r in erecs:
                if r[0] == "change":
                    txt = f"{path} {value_text(r[2], stream.eps, al) if r[2] is not None else '?'}->{value_text(r[3], stream.eps, al)}"
                elif r[0] == "appear":
                    txt = f"{path} appears {value_text(r[2], stream.eps, al)}"
                else:
                    txt = f"{path} ends {value_text(r[2], stream.eps, al)}"
                ev_changes.setdefault(r[1], []).append(txt)
        for t, items in sorted(ev_changes.items()):
            groups.append({"kind": "evidence_change", "head": ("rel", alias), "tick": t,
                           "line": f"  {alias} t={ts(t, origin)} " + "; ".join(items)})
        if samples.get(origin) is not None:
            origin_lines[("rel", alias)] = {"head": ("rel", alias), "line": f"  {alias} t=0 {samples[origin]}"}
            if ev_origin:
                origin_lines[("rel", alias, "evidence")] = {"head": ("rel", alias),
                                                           "line": f"  {alias} t=0 evidence " + "; ".join(ev_origin)}
    # ---- events and scheduled activations (history only) ----------------------------------
    for ev in src.events:
        if start < ev.tick <= origin:
            args = " ".join(f"{k}={al.ref(v) if isinstance(v, str) else v}" for k, v in ev.bindings.items())
            groups.append({"kind": "event", "head": ("events",), "tick": ev.tick,
                           "line": f"  t={ts(ev.tick, origin)} {ev.type} {args}"})
    for sc in src.scheduled:
        if start < sc.tick <= origin:
            groups.append({"kind": "scheduled", "head": ("scheduled",), "tick": sc.tick,
                           "line": f"  t={ts(sc.tick, origin)} {sc.category}: {sc.title}"})
    headings[("events",)] = "events"
    headings[("scheduled",)] = "scheduled"
    queries = build_queries(src, origin, al, rel_alias, present0)
    return {"episode": src.episode, "family": family_of(src.episode), "origin": origin,
            "frame_origin_cm": frame_o, "aliases": al.entity, "headings": {"|".join(k): v for k, v in headings.items()},
            "rel_entities": rel_entities,
            "origin_lines": {"|".join(k): {"head": "|".join(v["head"]), "line": v["line"]} for k, v in origin_lines.items()},
            "groups": [{**g, "head": "|".join(g["head"])} for g in groups],
            "queries": queries, "visual": visual or [],
            "compaction_failures": ["|".join(map(str, k)) for k in failures]}


# ---- queries and labels -----------------------------------------------------------------------
def flip_label(samples, masked, tuple_id, origin, v0):
    """First-change class (FLIP_LABELS index), or the compatible classes when the timeline has gaps.

    t_eq is the end of the observed run equal to v0 that starts at the origin.  If the next sample is observed
    and different, the class is exact.  Otherwise the first change lies in (t_eq, t_diff], where t_diff is the
    first later observed sample differing from v0 (a change must have happened by then), or after t_eq without
    upper bound when no later sample differs.
    """
    known = lambda t: samples.get(t) in KNOWN and (tuple_id, t) not in masked
    ticks = range(origin + S.FORMAL_STEP, origin + 51, S.FORMAL_STEP)
    t_eq = origin
    for t in ticks:
        if not known(t):
            break
        if samples[t] != v0:
            return time_bin(t - origin), None
        t_eq = t
    else:
        return 5, None
    t_diff = next((t for t in ticks if t > t_eq and known(t) and samples[t] != v0), None)
    lo = t_eq - origin
    hi = (t_diff - origin) if t_diff is not None else None
    compatible = [i for i, (a, b) in enumerate(S.FLIP_BINS) if b > lo and (hi is None or a < hi)]
    if hi is None:
        compatible.append(5)
    return (compatible[0], None) if len(compatible) == 1 else (None, compatible)


def time_bin(dt):
    return next(i for i, (a, b) in enumerate(S.FLIP_BINS) if a < dt <= b)


def build_queries(src, origin, al, rel_alias, present0):
    queries = []
    # Numeric states.
    for e in sorted(present0, key=lambda e: al.entity[e]):
        a, cat = al.entity[e], src.category[e]
        pos = src.position_raw.get(e, {})
        if cat in S.MOVING and origin in pos:
            for h in S.HORIZONS:
                t = origin + h
                # Residual of the stored values in quanta: |y - x0| < eps/2 is I0.
                label = [int(round((pos[t][i] - pos[origin][i]) / S.POS_EPS)) for i in range(3)] if t in pos else None
                queries.append({"group": "position", "entity": a, "category": cat, "horizon": h,
                                "text": f"{a} position@+{h*S.TICK_S:.1f}s:", "residual": label, "eps": S.POS_EPS})
        for fam, (source, name, eps) in S.NUMERIC_TARGETS.items():
            if source == "frame":
                continue
            raw = src.target_raw.get((e, source, name), {})
            if origin not in raw:
                continue
            x0 = raw[origin]
            for h in S.HORIZONS:
                y = raw.get(origin + h)
                queries.append({"group": fam, "entity": a, "category": cat, "horizon": h,
                                "text": f"{a} {'link' if source == 'comm' else 'compute'} {S.short(name)}@+{h*S.TICK_S:.1f}s:",
                                "residual": [int(round((y - x0) / eps))] if y is not None else None, "eps": eps})
    # Predicates: per-horizon state and first change.
    for tid, alias in rel_alias.items():
        rel = src.relations[tid]
        v0 = rel.samples.get(origin)
        # Label validity is per (tuple, tick): a horizon's value label needs only its own tick to be valid; the
        # first change additionally needs a valid origin state.
        origin_ok = v0 in KNOWN and (tid, origin) not in src.masked_labels
        hist = sum(1 for t in range(origin - S.HISTORY_TICKS + S.FORMAL_STEP, origin + 1, S.FORMAL_STEP)
                   if rel.samples.get(t) != rel.samples.get(t - S.FORMAL_STEP))
        base = {"tuple": alias, "predicate": rel.predicate, "family": rel.predicate.split(".")[0],
                "state": v0 if origin_ok else None, "history_changes": hist}
        horizon_queries = []
        for h in S.HORIZONS:
            v = rel.samples.get(origin + h)
            ok = v in KNOWN and (tid, origin + h) not in src.masked_labels
            horizon_queries.append({**base, "group": "predicate", "horizon": h,
                                    "text": f"{alias} value@+{h*S.TICK_S:.1f}s:", "label": KNOWN.index(v) if ok else None})
        # Every history tuple is queryable whatever its future labels: the query set (and with it the fixed lines
        # and the selector prefix) must not depend on which future labels happen to be valid.
        queries += horizon_queries
        if origin_ok:
            label, compatible = flip_label(rel.samples, src.masked_labels, tid, origin, v0)
            queries.append({**base, "group": "first_change", "text": f"{alias} first_change:", "label": label,
                            "compatible": compatible})
    # Events: first occurrence of each event type involving an entity, within (t0, t0+5 s].
    for e in sorted(present0, key=lambda e: al.entity[e]):
        a, cat = al.entity[e], src.category[e]
        for etype in EVENT_TYPES.get(cat, ()):
            hit = next((ev.tick for ev in src.events if ev.type == etype and e in ev.entities
                        and origin < ev.tick <= origin + 50), None)
            queries.append({"group": "event", "entity": a, "category": cat, "event_type": etype,
                            "text": f"{a} next {etype}:", "label": 0 if hit is None else 1 + time_bin(hit - origin)})
    # Events without entity participants (environment, ground, emergency): one window-level query.
    hit = next((ev for ev in src.events if ev.type in GLOBAL_EVENT_TYPES and origin < ev.tick <= origin + 50), None)
    queries.append({"group": "global_event", "text": "next area event:",
                    "label": None if hit is None else [hit.type, time_bin(hit.tick - origin)], "none": hit is None})
    nxt = next((sc for sc in src.scheduled if origin < sc.tick <= origin + 50), None)
    queries.append({"group": "scheduled", "text": "next scheduled activation:",
                    "label": None if nxt is None else [nxt.category, time_bin(nxt.tick - origin)],
                    "none": nxt is None})
    return queries


# Event types each participant category can take part in (schema-level, configs/unified_schema.json).
def _event_types():
    import json
    path = S.ROOT/"Dataset/world_model/configs/unified_schema.json"
    return json.loads(path.read_text())["event_types_by_category"] if path.exists() else {}


EVENT_TYPES: dict = _event_types()


def _global_event_types():
    import json
    path = S.ROOT/"Dataset/world_model/configs/unified_schema.json"
    if not path.exists():
        return ()
    raw = json.loads(path.read_text())
    bound = {t for types in raw["event_types_by_category"].values() for t in types}
    return tuple(sorted(set(raw["event_counts"]) - bound))


GLOBAL_EVENT_TYPES = _global_event_types()
