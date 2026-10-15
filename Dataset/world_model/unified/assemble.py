"""Query sets, units, the 20K total budget and deterministic serialization of a window (RESEARCH_PLAN §3.3-§3.5, §6).

Units.  Every compacted history record group, every origin-state line not fixed by the query set, and every
four-modal capture of every observer in the history window is a selection unit.  A text unit costs its tokens
plus, on demand, the heading it needs (an entity or relation definition not already present).  A capture costs
its frame header, its sensor tokens and a closing newline, plus the "images" heading once.

Budget.  TOTAL_BUDGET is the whole predictor sequence in the training mode: background, fixed block, selected
text and captures, the query block (the larger of the joint sequence of all answers and the longest marginal
continuation) and the padding of the context and of the joint sequence.  What remains after the background,
the fixed block, the queries and the padding is the optional capacity shared by text and captures; there is no
separate text, observer or frame limit.  When every candidate fits, every candidate is kept.

Text.  Background (window conventions, the queried predicates' definitions and thresholds, answer codes),
actions fixed at the origin (§8), captures oldest first (frame header, sensor tokens), entities (identity,
origin lines, kept records by time),
relations (definition, origin line, kept records), events, scheduled activations.  Kept groups are merged by
owner, then time; repeated headings are not repeated; no free-form rewriting.  Token ids are the concatenation
of per-line tokenizations, so position counts are exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import json
import random

from Dataset.world_model.unified import schema as S
from Dataset.world_model.unified.windows import stable_hash

TOTAL_BUDGET = 20000
PAD_TO = 64                        # predictor.Predictor.PAD_TO
NEWLINE = 198                      # "\n" in the Qwen3.5 tokenizer
PAD_RESERVE = 2 * (PAD_TO - 1)     # worst-case padding of the context and of the joint sequence
NO_ACTIONS_TEXT = "actions fixed at t=0\n  none supplied: no external actions or plans were given at the origin\n"

BACKGROUND = (
    "world window: history 5.0 s sampled at 0.1 s; times t in s relative to the prediction origin (t=0); inputs "
    "hold only t<=0\n"
    "frame: east-north-up metres relative to the window reference point o, z above local ground; p=position "
    "v=velocity\n"
    "quanta: p 0.01 m; v 0.01 m/s; link.latency_ms 0.1; compute.cpu_utilization_pct 0.01; evidence by unit suffix\n"
    "reconstruction: motion = velocity-hold knots (p_t = p_k + v_k (t - t_k)); fields, relations and evidence = "
    "hold-last with a->b changes; in a shown t=0 line, flags=[...] lists the entity's recorded true boolean fields "
    "and off=[...] its recorded false ones (a boolean in neither list is unknown); records not shown were omitted "
    "and are unknown\n")
IMAGES_NOTE = ("images: each capture gives observer, time t, camera position p and height agl (m), optical axis "
               "look (unit vector), field of view (deg) and its modalities, then the tokens of those modalities "
               "(rgb; depth, segmentation and lidar regions when listed)\n")
ANSWER_NOTES = {
    "numeric": "numeric answers code the change from t=0 in quanta (position: x, y, z in turn): first letter A = "
               "no change (under half a quantum), then intervals of growing size, negative ones first, then positive "
               "ones; second letter = sub-interval, smallest magnitude first\n",
    "predicate": "value@ answers: false or true\n",
    "time": "first_change: A (0,0.5] B (0.5,1] C (1,2] D (2,3] E (3,5] F >5 s; next <event>: A none, B-F the same "
            "bins up to 5 s\n",
    "categorical": "next area event / next scheduled activation: A none, else a type letter then a time-bin letter "
                   "(A (0,0.5] ... E (3,5] s)\n",
}
ANSWER_GROUPS = {"position": "numeric", "latency_ms": "numeric", "cpu_utilization_pct": "numeric",
                 "predicate": "predicate", "first_change": "time", "event": "time", "global_event": "categorical",
                 "scheduled": "categorical"}
QUERY_HEADER = "queries (answer with codes):"
GROUP_WEIGHTS = {"position": 1., "latency_ms": 1., "cpu_utilization_pct": 1., "predicate": 1., "first_change": 1.,
                 "event": 1., "scheduled": 1., "global_event": 1.}
CAPS = {"entities": 6, "link_entities": 2, "compute_entities": 2, "tuples": 12, "events": 8}


@functools.lru_cache(maxsize=1)
def predicate_definitions():
    path = S.ROOT/"Dataset/world_model/configs/unified_predicates.json"
    return json.loads(path.read_text()) if path.exists() else {}


@dataclass
class Unit:
    key: str
    head: str
    tick: int
    kind: str
    line: str
    ids: list = field(default_factory=list)
    extra: int = 0                 # positions besides ids: sensor tokens and the closing newline of a capture
    ref: dict | None = None        # capture reference (visual.load_features)
    summary: object = None         # frozen-feature summary vector of a capture (selector input)
    observer: str = ""
    capture: object = None         # visual.Capture of a capture unit (structured metadata in capture.info)
    relevance: float = 0.


class PreparedWindow:
    """A window with tokenized lines and capture units; serves query selection, unit costs and assembly.

    captures: [(visual.Capture with its metadata in .info, predictor frame header)].
    """

    def __init__(self, window, tokenizer, captures=()):
        self.w = window
        self.tok = tokenizer
        enc = lambda lines: tokenizer(list(lines), add_special_tokens=False)["input_ids"] if lines else []
        self._enc = enc
        self.action_ids = enc([NO_ACTIONS_TEXT])[0]
        self.headings = dict(zip(window["headings"], enc(v+"\n" for v in window["headings"].values())))
        units = []
        for i, g in enumerate(window["groups"]):
            units.append(Unit(f"g{i}", g["head"], g["tick"], g["kind"], g["line"]))
        for key, o in window["origin_lines"].items():
            units.append(Unit(f"o:{key}", o["head"], window["origin"], "origin", o["line"]))
        for u, ids in zip(units, enc(u.line+"\n" for u in units)):
            u.ids = ids
        caps = []
        for cap, header in captures:
            caps.append(Unit(cap.key, "images", cap.tick, "capture", header, extra=cap.positions + 1, ref=cap.ref(),
                             summary=cap.summary, observer=cap.observer, capture=cap))
        if caps:
            self.headings["images"] = enc(["images\n"])[0]
            for u, ids in zip(caps, enc(u.line for u in caps)):
                u.ids = ids
        self.units = {u.key: u for u in units + caps}
        self.inclusion = None
        self._background = {}
        self._selector_ids = {}

    # ---- queries ---------------------------------------------------------------------------
    def select_queries(self, rng: random.Random | None = None, caps=CAPS, oversample=None):
        """Structured query subset.  rng=None gives the deterministic evaluation set of the window.

        Training oversamples tuples whose value changed in the history (half of the tuple slots); evaluation
        (rng=None) samples tuples uniformly so scores refer to the natural distribution.
        """
        w, qs = self.w, self.w["queries"]
        oversample = (rng is not None) if oversample is None else oversample
        self.inclusion = {}
        seed = stable_hash(w["episode"], w["origin"])
        rng = rng or random.Random(seed)
        by = {}
        for i, q in enumerate(qs):
            by.setdefault(q["group"], []).append(i)
        chosen = []
        moving = sorted({qs[i]["entity"] for i in by.get("position", [])})
        ents = rng.sample(moving, min(caps["entities"], len(moving)))
        chosen += [i for i in by.get("position", []) if qs[i]["entity"] in ents]
        for g, cap in (("latency_ms", caps["link_entities"]), ("cpu_utilization_pct", caps["compute_entities"])):
            pool = sorted({qs[i]["entity"] for i in by.get(g, [])})
            pick = rng.sample(pool, min(cap, len(pool)))
            chosen += [i for i in by.get(g, []) if qs[i]["entity"] in pick]
        # Tuples with any query: per-horizon value queries exist even when the origin label (needed only by the first
        # change) is invalid.
        tuple_queries = by.get("predicate", []) + by.get("first_change", [])
        tuples = sorted({qs[i]["tuple"] for i in tuple_queries})
        changed = sorted({qs[i]["tuple"] for i in tuple_queries if qs[i]["history_changes"]})
        half = caps["tuples"] // 2 if oversample else 0
        pick = rng.sample(changed, min(half, len(changed)))
        rest = [t for t in tuples if t not in pick]
        extra = rng.sample(rest, min(caps["tuples"] - len(pick), len(rest)))
        # Inclusion probabilities of the two-stratum design, for inverse-probability loss weights (§6).
        p_changed = min(1., half / len(changed)) if changed else 0.
        p_rest = len(extra) / len(rest) if rest else 0.
        unchanged = set(tuples) - set(changed)
        for t in pick + extra:
            # A changed tuple not taken in the first stratum can still enter the second one.
            self.inclusion[t] = p_rest if t in unchanged else p_changed + (1 - p_changed) * p_rest
        pick += extra
        chosen += [i for g in ("predicate", "first_change") for i in by.get(g, []) if qs[i]["tuple"] in pick]
        events = by.get("event", [])
        chosen += rng.sample(events, min(caps["events"], len(events)))
        chosen += by.get("scheduled", []) + by.get("global_event", [])
        return sorted(set(chosen))

    def query_inclusion(self, i):
        """Inclusion probability of query i's tuple under the last select_queries call (1 for other groups)."""
        q = self.w["queries"][i]
        return self.inclusion.get(q.get("tuple"), 1.) if "tuple" in q else 1.

    def addressed_entities(self, query_ids):
        """Entity aliases the query set addresses: queried entities and the entities a queried relation binds,
        directly or as owners of bound objects (window rel_entities)."""
        out = set()
        for i in query_ids:
            q = self.w["queries"][i]
            if "tuple" in q:
                out.update(self.w["rel_entities"].get(q["tuple"], ()))
            elif "entity" in q:
                out.add(q["entity"])
        return out

    def fixed_keys(self, query_ids):
        """Origin-line units fixed by the query set (the addressed entities' and tuples' origin state)."""
        keys, heads = set(), set()
        for i in query_ids:
            q = self.w["queries"][i]
            if "tuple" in q:
                heads.add(f"rel|{q['tuple']}")
                keys.add(f"o:rel|{q['tuple']}")
                # The bound entities' current state (for a bound object, its owner's) is part of the queried
                # relation's origin state: the frame line, and the link/compute line of communication/compute
                # predicates.
                source = {"communication": "comm", "compute": "compute"}.get(q["predicate"].split(".")[0])
                for a in self.w["rel_entities"].get(q["tuple"], ()):
                    heads.add(f"entity|{a}")
                    keys.add(f"o:entity|{a}|frame")
                    if source:
                        keys.add(f"o:entity|{a}|{source}")
            elif "entity" in q:
                heads.add(f"entity|{q['entity']}")
                source = {"latency_ms": "comm", "cpu_utilization_pct": "compute"}.get(q["group"], "frame")
                keys.add(f"o:entity|{q['entity']}|{source}")
        return {k for k in keys if k in self.units}, heads

    def candidates(self, query_ids, kinds="all"):
        """Optional units: kinds = "all", "text" or "visual" (captures)."""
        fixed, _ = self.fixed_keys(query_ids)
        keep = {"all": lambda u: True, "text": lambda u: u.kind != "capture", "visual": lambda u: u.kind == "capture"}[kinds]
        return [u for k, u in self.units.items() if k not in fixed and keep(u)]

    def captures(self):
        return [u for u in self.units.values() if u.kind == "capture"]

    def selector_ids(self, unit, query_ids=()):
        """Token ids of a unit as the selector reads it: the text line, or for a capture its metadata line
        rendered for the query set (addressed entities and queried relations listed first)."""
        if unit.kind != "capture":
            return unit.ids
        key = (unit.key, tuple(query_ids))
        if key not in self._selector_ids:
            from Dataset.world_model.unified import visual as V
            line = V.metadata_line(unit.capture, self.addressed_entities(query_ids), self.queried_tuples(query_ids))
            self._selector_ids[key] = self._enc([line])[0] + [NEWLINE]
        return self._selector_ids[key]

    def queried_tuples(self, query_ids):
        return {self.w["queries"][i]["tuple"] for i in query_ids if "tuple" in self.w["queries"][i]}

    # ---- background and budget -------------------------------------------------------------
    def background_ids(self, query_ids):
        """Window conventions, the queried predicates' definitions and thresholds, and the answer-code notes of
        the query groups present (only what the window's inputs and queries use)."""
        qs = [self.w["queries"][i] for i in query_ids]
        preds = sorted({q["predicate"] for q in qs if "predicate" in q})
        notes = sorted({ANSWER_GROUPS[q["group"]] for q in qs}, key=list(ANSWER_NOTES).index)
        key = (tuple(preds), tuple(notes), bool(self.captures()))
        if key not in self._background:
            text = BACKGROUND + (IMAGES_NOTE if key[2] else "")
            defs = predicate_definitions()
            for p in preds:
                if p in defs:
                    th = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                                   for k, v in defs[p]["thresholds"].items())
                    text += f"predicate {p}: {defs[p]['expression']}" + (f" ({th})" if th else "") + "\n"
            text += "".join(ANSWER_NOTES[n] for n in notes)
            self._background[key] = self._enc([text])[0]
        return self._background[key]

    def fixed_cost(self, query_ids):
        """Mandatory context including the default origin-action block (RESEARCH_PLAN §6, §8)."""
        fixed, _ = self.fixed_keys(query_ids)
        return (len(self.background_ids(query_ids)) + len(self.action_ids)
                + sum(len(self.headings[h]) for h in self.base_heads(query_ids))
                + sum(len(self.units[k].ids) for k in fixed))

    def optional_budget(self, query_ids, n_query, total=TOTAL_BUDGET):
        """Positions left for selected text and captures under the total sequence budget."""
        return max(0, total - self.fixed_cost(query_ids) - n_query - PAD_RESERVE)

    # ---- costs -----------------------------------------------------------------------------
    def base_heads(self, query_ids):
        """Headings always present: identity of every entity present at the origin, plus query-addressed ones."""
        _, heads = self.fixed_keys(query_ids)
        return heads | {h for h in self.headings if h.startswith("entity|") and self._present(h)}

    def _present(self, head):
        a = head.split("|", 1)[1]
        return any(f"o:entity|{a}|{s}" in self.units for s in ("frame", "comm", "compute"))

    def unit_cost(self, unit, heads):
        return len(unit.ids) + unit.extra + (len(self.headings[unit.head]) if unit.head not in heads else 0)

    def set_cost(self, keys, query_ids):
        """Optional positions of a kept set (headings counted once)."""
        heads, used = set(self.base_heads(query_ids)), 0
        for k in sorted(keys, key=lambda k: (self.units[k].tick, k)):
            u = self.units[k]
            used += self.unit_cost(u, heads)
            heads.add(u.head)
        return used

    def greedy_fill(self, ordered_units, budget, query_ids):
        """Keep units in the given priority order while the optional budget allows (headings on demand)."""
        heads, kept, used = set(self.base_heads(query_ids)), [], 0
        for u in ordered_units:
            c = self.unit_cost(u, heads)
            if used + c <= budget:
                kept.append(u.key)
                used += c
                heads.add(u.head)
        return kept, used

    # ---- assembly --------------------------------------------------------------------------
    def assemble(self, kept, query_ids, actions_ids=None):
        """Context and position counts, with the origin-action block after background (RESEARCH_PLAN §6, §8).

        Returns segments [("text", ids) | ("image", capture ref)] and counts: N_background, N_fixed (background,
        actions, identity headings, fixed origin lines), N_text (optional text), N_frame_text (images heading, frame
        headers and closing newlines), N_visual (sensor tokens).
        """
        fixed, _ = self.fixed_keys(query_ids)
        kept = set(kept) | fixed
        heads = self.base_heads(query_ids)
        bg = self.background_ids(query_ids)
        actions = self.action_ids if actions_ids is None else actions_ids
        segments = [("text", list(bg)), ("text", list(actions))]
        counts = {"N_background": len(bg), "N_fixed": len(bg) + len(actions), "N_text": 0,
                  "N_frame_text": 0, "N_visual": 0,
                  "N_captures": 0}
        caps = sorted((self.units[k] for k in kept if self.units[k].kind == "capture"), key=lambda u: (u.tick, u.key))
        if caps:
            segments.append(("text", list(self.headings["images"])))
            counts["N_frame_text"] += len(self.headings["images"])
            for u in caps:
                segments += [("text", list(u.ids)), ("image", u.ref), ("text", [NEWLINE])]
                counts["N_frame_text"] += len(u.ids) + 1
                counts["N_visual"] += u.extra - 1
            counts["N_captures"] = len(caps)
        by_head = {}
        for k in kept:
            u = self.units[k]
            if u.kind != "capture":
                by_head.setdefault(u.head, []).append(u)
        text = []
        for head in sorted(self.headings, key=_head_order):
            if head == "images":
                continue
            units = by_head.get(head, [])
            if head not in heads and not units:
                continue
            ids = list(self.headings[head])
            text.append(ids)
            counts["N_fixed" if head in heads else "N_text"] += len(ids)
            for u in sorted(units, key=lambda u: (u.kind != "origin", u.tick, u.key)):
                text.append(u.ids)
                counts["N_fixed" if u.key in fixed else "N_text"] += len(u.ids)
        segments.append(("text", [t for ids in text for t in ids]))
        return segments, counts


def _head_order(head):
    kind, _, name = head.partition("|")
    rank = {"images": -1, "entity": 0, "rel": 1, "events": 2, "scheduled": 3}[kind]
    prefix = name.rstrip("0123456789")
    number = int(name[len(prefix):]) if name[len(prefix):].isdigit() else 0
    return rank, prefix, number, name


# ---- simple selection policies (baselines at equal budget) --------------------------------------
CHANGE_KINDS = ("rel_change", "event", "scheduled", "leave", "frame_change", "comm_change", "compute_change", "origin",
                "capture")
POLICIES = ("recency", "uniform_time", "change_point", "random", "relevance")
VISUAL_POLICIES = ("recency", "uniform_time", "relevance", "random")


def _order(units, policy, rng=None):
    if policy == "recency":
        return sorted(units, key=lambda u: (-u.tick, u.key))
    if policy == "uniform_time":
        by_tick = {}
        for u in sorted(units, key=lambda u: u.key):
            by_tick.setdefault(u.tick, []).append(u)
        ticks = sorted(by_tick)
        if not ticks:
            return []
        # Van der Corput order over the history's ticks spreads consecutive picks evenly in time.
        bits = max(1, (len(ticks) - 1).bit_length())
        spread = sorted(range(len(ticks)), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))
        out, depth = [], 0
        while len(out) < len(units):
            for i in spread:
                group = by_tick[ticks[i]]
                if depth < len(group):
                    out.append(group[depth])
            depth += 1
        return out
    if policy == "change_point":
        rank = {k: i for i, k in enumerate(CHANGE_KINDS)}
        return sorted(units, key=lambda u: (rank.get(u.kind, len(rank)), -u.tick, u.key))
    if policy == "random":
        rng = rng or random.Random(0)
        units = sorted(units, key=lambda u: u.key)
        rng.shuffle(units)
        return units
    if policy == "relevance":        # captures by geometric relevance (visual.relevance), then recency
        return sorted(units, key=lambda u: (-u.relevance, -u.tick, u.key))
    raise ValueError(policy)


def policy_order(pw: PreparedWindow, query_ids, policy, rng=None, kinds="all"):
    """One priority order over the optional units ("relevance": captures by geometric relevance, then the text
    by recency; as a baseline it is applied through select(), which gives captures half of the budget first)."""
    units = pw.candidates(query_ids, kinds)
    if policy == "relevance":
        return _order([u for u in units if u.kind == "capture"], "relevance") + \
            _order([u for u in units if u.kind != "capture"], "recency")
    return _order(units, policy, rng)


def mixed_fill(pw: PreparedWindow, query_ids, budget, text_policy, visual_policy, share, rng=None):
    """Kept set with a visual share: captures by visual_policy up to share x budget, then text by text_policy,
    then captures again in the remaining capacity (training reductions and the "relevance" baseline)."""
    caps = _order(pw.candidates(query_ids, "visual"), visual_policy, rng)
    text = _order(pw.candidates(query_ids, "text"), text_policy, rng)
    first, used = pw.greedy_fill(caps, int(share * budget), query_ids)
    heads = set(pw.base_heads(query_ids)) | {pw.units[k].head for k in first}
    kept, taken = list(first), set(first)
    for u in text + [c for c in caps if c.key not in taken]:
        c = pw.unit_cost(u, heads)
        if used + c <= budget:
            kept.append(u.key)
            used += c
            heads.add(u.head)
    return kept, used


def select(pw: PreparedWindow, query_ids, budget, policy, rng=None):
    """Kept set of a named baseline policy under the optional budget."""
    if policy == "relevance":
        return mixed_fill(pw, query_ids, budget, "recency", "relevance", 0.5)
    return pw.greedy_fill(policy_order(pw, query_ids, policy, rng), budget, query_ids)
