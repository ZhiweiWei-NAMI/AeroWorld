"""Answer domains as prefix trees of single tokens, interval codes and scores (RESEARCH_PLAN §4.2-§4.3).

Probability of a code: product along its path of the LM-head logits restricted
to the node's valid children and normalized there (locally normalized prefix
tree).  Numeric targets are residuals r = y - y_ref with y_ref = x0 (reference
R0, the current value), observed as the interval containing r.  Intervals are
in integer quanta: I0 = {0} (|r| < eps/2) and per sign log-spaced integer bins
up to the declared bounded support R_{f,c,h}.  A residual outside the support is
an audit error, never clipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

import numpy as np

from Dataset.world_model.unified import schema as S

FINE = 8          # sub-intervals per coarse interval
PER_SIGN = 64     # maximum intervals per sign


@dataclass
class Tree:
    """Prefix tree over abstract symbols; ``children[prefix]`` lists the valid next symbols."""
    children: dict = field(default_factory=dict)
    leaves: list = field(default_factory=list)       # complete codes (tuples of symbols)

    @staticmethod
    def from_codes(codes):
        t = Tree(leaves=[tuple(c) for c in codes])
        for c in t.leaves:
            for j in range(len(c)):
                kids = t.children.setdefault(c[:j], [])
                if c[j] not in kids:
                    kids.append(c[j])
        for c in t.leaves:      # no valid code is a proper prefix of another
            if c in t.children:
                raise ValueError(f"code {c} is a prefix of another code")
        return t


def integer_edges(r_quanta: int):
    """Upper edges u_1 < ... < u_M = r_quanta of the positive integer bins (u_{i-1}, u_i]."""
    edges = []
    for i in range(1, PER_SIGN + 1):
        v = math.ceil(0.5 * (r_quanta / 0.5) ** (i / PER_SIGN))
        v = max(v, edges[-1] + 1 if edges else 1)
        if v >= r_quanta:
            break
        edges.append(v)
    edges.append(r_quanta)
    return edges


@dataclass
class IntervalCode:
    """Residual partition of one numeric component at one support."""
    eps: float
    r_quanta: int
    edges: list = None

    def __post_init__(self):
        self.edges = integer_edges(self.r_quanta)
        m = len(self.edges)
        # Leaf symbols: ("0",) for I0; (sign, coarse, fine) for the others.
        self.codes, self.bins = [("z",)], [(0, 0)]
        for sign in ("-", "+"):
            for i in range(m):
                coarse, fine = divmod(i, FINE)
                self.codes.append((f"{sign}{coarse}", f"{fine}"))
                lo = (self.edges[i-1] + 1) if i else 1
                hi = self.edges[i]
                self.bins.append((-hi, -lo) if sign == "-" else (lo, hi))
        self.tree = Tree.from_codes(self.codes)

    def index(self, r: int):
        """Leaf index of an integer residual, or None when it lies outside the support."""
        if r == 0:
            return 0
        if abs(r) > self.r_quanta:
            return None
        i = int(np.searchsorted(self.edges, abs(r)))
        return 1 + i + (len(self.edges) if r > 0 else 0)

    def real_bounds(self):
        """Continuous interval (lo, hi] in field units of every leaf (quantum-centred)."""
        out = []
        for lo, hi in self.bins:
            out.append(((lo - .5) * self.eps, (hi + .5) * self.eps))
        return out


def boolean_tree():
    return Tree.from_codes([("false",), ("true",)])


def flat_tree(n):
    return Tree.from_codes([(str(i),) for i in range(n)])


def scheduled_tree(categories):
    return Tree.from_codes([("none",)] + [(c, str(b)) for c in categories for b in range(5)])


class Vocabulary:
    """Maps tree symbols to single LM tokens: booleans to ' false'/' true', the rest to the alphabet by position."""
    def __init__(self, tokenizer):
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
        self.alphabet = [enc(a)[0] for a in S.ALPHABET]
        self.boolean = {"false": enc(" false")[0], "true": enc(" true")[0]}
        assert all(len(enc(a)) == 1 for a in S.ALPHABET)

    def tokens(self, tree: Tree, prefix: tuple):
        """Token ids of the valid children of ``prefix`` (same order as tree.children)."""
        kids = tree.children[prefix]
        if set(kids) <= set(self.boolean):
            return [self.boolean[k] for k in kids]
        if len(kids) > len(self.alphabet):
            raise ValueError("node has more children than the alphabet")
        return self.alphabet[:len(kids)]

    def code_tokens(self, tree: Tree, code: tuple):
        """Token sequence of a complete code, with each step's child position."""
        out = []
        for j in range(len(code)):
            kids = tree.children[code[:j]]
            k = kids.index(code[j])
            out.append((self.tokens(tree, code[:j])[k], k))
        return out


class AnswerSchema:
    """Domains of every query group, built from configs/unified_schema.json."""
    def __init__(self, path=S.ROOT/"Dataset/world_model/configs/unified_schema.json"):
        self.raw = json.loads(path.read_text())
        self.scheduled = scheduled_tree(self.raw["scheduled_categories"])
        bound = {t for types in self.raw["event_types_by_category"].values() for t in types}
        self.global_event = scheduled_tree(sorted(set(self.raw["event_counts"]) - bound))
        self.boolean = boolean_tree()
        self.flip = flat_tree(6)
        self.event = flat_tree(6)
        self._intervals = {}

    def interval(self, family, category, horizon, eps):
        sup = self.raw["support"][family]
        r = sup.get(category, sup.get("*"))[str(horizon)]
        key = (family, category if category in sup else "*", horizon)
        if key not in self._intervals:
            self._intervals[key] = IntervalCode(eps, max(1, int(math.ceil(r / eps))))
        return self._intervals[key]

    def query_domain(self, q):
        """(tree(s), target code(s) or None, compatible codes for censoring or None).

        Numeric queries return one IntervalCode per component; the vector code is
        their concatenation (components in order, one prefix tree).
        """
        g = q["group"]
        if g in ("position", "latency_ms", "cpu_utilization_pct"):
            ic = self.interval(g, q.get("category"), q["horizon"], q["eps"])
            comps = 3 if g == "position" else 1
            if q["residual"] is None:
                return [ic]*comps, None, None
            idx = [ic.index(r) for r in q["residual"]]
            if any(i is None for i in idx):
                return [ic]*comps, "out_of_support", None
            return [ic]*comps, [ic.codes[i] for i in idx], None
        if g == "predicate":
            return [self.boolean], None if q["label"] is None else [("false", "true")[q["label"]:q["label"]+1]], None
        if g == "first_change":
            if q["label"] is not None:
                return [self.flip], [(str(q["label"]),)], None
            return [self.flip], None, [(str(i),) for i in q["compatible"]] if q.get("compatible") else None
        if g == "event":
            return [self.event], [(str(q["label"]),)], None
        if g in ("scheduled", "global_event"):
            tree = self.scheduled if g == "scheduled" else self.global_event
            if q["none"]:
                return [tree], [("none",)], None
            cat, b = q["label"]
            if (cat,) not in {c[:1] for c in tree.leaves}:
                return [tree], None, None
            return [tree], [(cat, str(b))], None
        raise ValueError(g)


# ---- continuous scores for interval forecasts -------------------------------------------------
def crps_piecewise_uniform(bounds, probs, y):
    """Exact CRPS of a piecewise-uniform density on bounded contiguous intervals against observation y."""
    order = np.argsort([b[0] for b in bounds])
    lo = np.array([bounds[i][0] for i in order], dtype=np.float64)
    hi = np.array([bounds[i][1] for i in order], dtype=np.float64)
    p = np.array([probs[i] for i in order], dtype=np.float64)
    cdf_lo = np.concatenate([[0.], np.cumsum(p)[:-1]])
    total = 0.
    for a, b, f0, m in zip(lo, hi, cdf_lo, p):
        # F(x) = f0 + m (x - a) / (b - a) on [a, b]; integrate (F - H(x - y))^2 exactly.
        w = b - a

        def seg(x0, x1, step):          # integral over [x0, x1] of (F(x) - step)^2
            if x1 <= x0:
                return 0.
            u0, u1 = f0 + m*(x0-a)/w - step, f0 + m*(x1-a)/w - step
            slope = m / w
            return (u1**3 - u0**3) / (3*slope) if slope > 0 else u0**2 * (x1-x0)
        cut = min(max(y, a), b)
        total += seg(a, cut, 0.) + seg(cut, b, 1.)
    # Outside the support F is 0 below and 1 above; the observation lies inside the support.
    if y < lo[0]:
        total += lo[0] - y
    if y > hi[-1]:
        total += y - hi[-1]
    return float(total)


def ranked_probability_score(probs, k):
    """RPS over ordered leaves (cumulative squared error), normalized by the number of thresholds."""
    cdf = np.cumsum(probs)[:-1]
    obs = (np.arange(len(probs)-1) >= k).astype(np.float64)
    return float(np.mean((cdf - obs)**2))


# ---- query encoding ---------------------------------------------------------------------------
@dataclass
class EncodedQuery:
    group: str
    ids: list                 # query text tokens followed by the teacher-forced code tokens (all but the last)
    steps: list               # [(offset into ids, candidate token ids)]
    targets: list | None      # child index per step (fully observed label)
    compatible: list | None   # censored single-level label: compatible child indices
    trees: list               # one tree per component
    path: list | None         # target code symbols per component
    full_ids: list | None = None   # ids plus the final code token (joint mode conditions later queries on it)
    inclusion: float = 1.          # inclusion probability of the query under the sampling design


def out_of_support(q, answers: AnswerSchema):
    """1 when a numeric label lies outside its declared support (an audit error, reported, never clipped)."""
    return int(answers.query_domain(q)[1] == "out_of_support")


def encode_query(q, answers: AnswerSchema, vocab: Vocabulary, tokenizer, text_cache=None):
    """Token layout of one query in marginal or joint mode, or None when its label is undefined."""
    trees, codes, compatible = answers.query_domain(q)
    if codes == "out_of_support":
        return None
    if codes is None and compatible is None:
        return None
    key = q["text"]
    if text_cache is not None and key in text_cache:
        text_ids = text_cache[key]
    else:
        text_ids = tokenizer("\n" + q["text"], add_special_tokens=False)["input_ids"]
        if text_cache is not None:
            text_cache[key] = text_ids
    if codes is None:     # censored: one level, compatible leaves
        tree = trees[0].tree if isinstance(trees[0], IntervalCode) else trees[0]
        kids = tree.children[()]
        return EncodedQuery(q["group"], list(text_ids), [(len(text_ids) - 1, vocab.tokens(tree, ()))], None,
                            [kids.index(c[0]) for c in compatible], trees, None)
    ids, steps, targets = list(text_ids), [], []
    for domain, code in zip(trees, codes):
        tree = domain.tree if isinstance(domain, IntervalCode) else domain
        for j in range(len(code)):
            prefix = code[:j]
            cands = vocab.tokens(tree, prefix)
            k = tree.children[prefix].index(code[j])
            steps.append((len(ids) - 1, cands))
            targets.append(k)
            ids.append(cands[k])
    full = list(ids)
    ids = ids[:-1]           # the last code token is predicted, never input
    return EncodedQuery(q["group"], ids, steps, targets, None, trees, list(codes), full)


def query_loss(logps, enc: EncodedQuery):
    """-log p(code) (product of local conditionals) or -log sum of compatible single-level masses."""
    import torch
    if enc.compatible is not None:
        return -torch.logsumexp(logps[0][enc.compatible], 0)
    return -sum(lp[t] for lp, t in zip(logps, enc.targets))
