"""Parent and ARM items under fixed TRAIN/DEV/TEST splits.

Text and captures share the capacity left after mandatory inputs and queries within the total budget.
"""
from __future__ import annotations

import functools
import json
import os
import random

from Dataset.world_model.unified import assemble as A, codes as K, schema as S, visual as V, windows as W
from Dataset.world_model.unified.build import load_source, parents

SPLITS = S.ROOT/"Dataset/world_model/configs/unified_splits.json"
ORIGINS = list(range(S.HISTORY_TICKS, S.LAST_TICK - 50 + 1, S.FORMAL_STEP))
DEV_PROFILES = ("L1-3_v1", "L2-3_v2", "L2-4_v2", "L3-2_v2", "L4-10_v2", "L4-3_v1", "L4-4_v1",
                "L4-5_v3", "L4-9_v1", "L5-1_v1", "L5-3_v1", "L6-2_v2", "L6-5_v1")
JOINT_TAIL = ("first_change", "event", "global_event", "scheduled")


def profile(episode):
    return episode.rsplit("__", 1)[0]


def template(episode):
    """Scenario template shared by its profile versions (RESEARCH_PLAN §8)."""
    return profile(episode).rsplit("_v", 1)[0]


def make_splits():
    """Keep the 13 DEV profiles; hold out TEST templates by family (RESEARCH_PLAN §8)."""
    episodes = parents()
    held = json.loads(SPLITS.read_text())["dev_profiles"] if SPLITS.exists() else list(DEV_PROFILES)
    dev_templates = {template(e) for e in episodes if profile(e) in held}
    by_family = {}
    for e in episodes:
        family = W.family_of(e)
        by_family.setdefault("X" if family.startswith("X") else family, set()).add(template(e))
    rng, test_templates = random.Random(20261012), []
    for fam in sorted(by_family):
        candidates = sorted(by_family[fam] - dev_templates)
        k = max(1, round(0.2 * len(by_family[fam])))
        test_templates += rng.sample(candidates, min(k, len(candidates)))
    dev = sorted(e for e in episodes if profile(e) in held)
    test = sorted(e for e in episodes if template(e) in test_templates)
    out = {"rule": "fixed DEV profiles; TEST templates outside DEV, 20% per family (X pooled), seed 20261012; "
                   "TRAIN contains the remaining parents; all seeds and arms follow their parent",
           "dev_profiles": sorted(held), "test_templates": sorted(test_templates),
           "train": sorted(set(episodes) - set(dev) - set(test)), "dev": dev, "test": test}
    tmp = SPLITS.with_name(f".{SPLITS.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(SPLITS)
    splits.cache_clear()
    return out


@functools.lru_cache(maxsize=1)
def splits():
    """Fixed three-way parent split; migrate the earlier profile-only file (RESEARCH_PLAN §8)."""
    if SPLITS.exists():
        out = json.loads(SPLITS.read_text())
        if all(k in out for k in ("train", "dev", "test", "test_templates")) and "fit" not in out:
            return out
    return make_splits()


def split_of(episode):
    """A parent and all its ARM members belong to one split (RESEARCH_PLAN §8)."""
    for name in ("train", "dev", "test"):
        if episode in splits()[name]:
            return name
    raise ValueError(f"parent is absent from the split: {episode}")


@functools.lru_cache(maxsize=8)
def cached_source(episode):
    return load_source(episode)


def query_positions(encs, joint):
    """Realized positions of the query block in the training mode: the joint sequence of every answer, or the
    longest marginal continuation (padded to 8) if longer (reporting only; budgets use query_reserve)."""
    joint_len = sum(len(encs[j].full_ids) for j in joint)
    marginal = max((len(e.ids) + (-len(e.ids)) % 8 for e in encs), default=0)
    return max(joint_len, marginal)


def code_depth(tree):
    return max(len(leaf) for leaf in tree.leaves)


def query_reserve(enc, queries, qids):
    """Label-free upper bound of the query block: every selected query with the longest code of its answer
    domain, in the joint sequence or as the longest marginal continuation.  It depends only on the query set,
    never on the realized answers (a zero residual codes shorter than a nonzero one), so identical histories
    always get the same optional capacity; it also bounds ancestral joint samples."""
    joint, marginal = 0, 0
    for i in qids:
        q = queries[i]
        key = q["text"]
        if key not in enc.text_cache:
            enc.text_cache[key] = enc.tok("\n" + q["text"], add_special_tokens=False)["input_ids"]
        trees = enc.answers.domains(q)
        n = len(enc.text_cache[key]) + sum(code_depth(t.tree if isinstance(t, K.IntervalCode) else t) for t in trees)
        joint += n
        marginal = max(marginal, n - 1 + (-(n - 1)) % 8)
    return max(joint, marginal)


def padded_total(counts, encs, joint):
    """Sequence positions the predictor actually builds (context and joint padding included)."""
    ctx = counts["N_fixed"] + counts["N_text"] + counts["N_frame_text"] + counts["N_visual"]
    ctx += (-ctx) % A.PAD_TO
    joint_len = sum(len(encs[j].full_ids) for j in joint)
    marginal = max((len(e.ids) + (-len(e.ids)) % 8 for e in encs), default=0)
    return ctx + max(joint_len + (-(ctx + joint_len)) % A.PAD_TO, marginal)


def visual_loader(item):
    return lambda ref: item["visual"][V.ref_key(ref)]


class Encoder:
    """Per-process tokenizer, answer schema and vocabulary."""
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(S.QWEN_2B)
        self.answers = K.AnswerSchema()
        self.vocab = K.Vocabulary(self.tok)
        self.text_cache = {}

    def window(self, episode, origin, images=True):
        src = cached_source(episode)
        w = W.build_window(src, origin)
        caps = []
        if images:
            for c in V.window_captures(episode, origin):
                caps.append((c, V.describe(c, src, w, W.ts)))
        return w, A.PreparedWindow(w, self.tok, caps)

    def set_relevance(self, w, pw, qids):
        """Geometric relevance of every capture for the query set (entities addressed directly or through a
        queried tuple's bindings weigh 1, other visible entities 0.1)."""
        alias_to_id = {a: e for e, a in w["aliases"].items()}
        addressed = {alias_to_id[a] for a in pw.addressed_entities(qids) if a in alias_to_id}
        src = cached_source(w["episode"])
        for u in pw.captures():
            u.relevance = V.relevance(u.capture, src, addressed)

    def encode_queries(self, w, pw, qids):
        encs, idx, out_of_support = [], [], 0
        for i in qids:
            q = w["queries"][i]
            e = K.encode_query(q, self.answers, self.vocab, self.tok, self.text_cache)
            if e is None:
                out_of_support += K.out_of_support(q, self.answers)
                continue
            e.inclusion = pw.query_inclusion(i) if getattr(pw, "inclusion", None) is not None else 1.
            encs.append(e)
            idx.append(i)
        return encs, idx, out_of_support

    def optional_budget(self, w, pw, qids, total=A.TOTAL_BUDGET):
        """Positions left for text and captures once the background, fixed block, label-free query reserve and
        padding of the query set are reserved within the total budget (ValueError if they alone exceed it)."""
        reserve = query_reserve(self, w["queries"], qids)
        if pw.fixed_cost(qids) + reserve + A.PAD_RESERVE > total:
            raise ValueError(f"mandatory inputs exceed the total budget {total}: {w['episode']}@{w['origin']}")
        return pw.optional_budget(qids, reserve, total)

    def reduce(self, pw, qids, budget, policy, rng=None):
        """Kept set of a policy under the optional budget.  policy: a baseline name (assemble.POLICIES), or
        {"text": p, "visual": p, "share": f} for a mixed training reduction."""
        if isinstance(policy, dict):
            if policy["share"] == 0:
                return pw.greedy_fill(A.policy_order(pw, qids, policy["text"], rng, kinds="text"), budget, qids)
            return A.mixed_fill(pw, qids, budget, policy["text"], policy["visual"], policy["share"], rng)
        return A.select(pw, qids, budget, policy, rng)

    def item(self, episode, origin, rng=None, budget=A.TOTAL_BUDGET, policy="recency", kept=None, query_ids=None,
             images=True, prepared=None, features=None, inclusion=None):
        """One item.  budget = total sequence positions; kept (optional) overrides the policy (selector choices,
        supervision sets).  features: a dict reused across items of one window (frozen-encoder outputs).
        inclusion: the query-tuple inclusion probabilities of explicitly given query_ids (selector exports); with
        query_ids from select_queries on the same prepared window, its recorded probabilities are used."""
        w, pw = prepared or self.window(episode, origin, images)
        if query_ids is None:
            qids = pw.select_queries(rng)
        else:
            qids = query_ids
            if inclusion is not None or prepared is None:
                pw.inclusion = inclusion
        encs, idx, out_of_support = self.encode_queries(w, pw, qids)
        joint = joint_order(w["queries"], idx, encs)
        n_query = query_positions(encs, joint)
        optional = self.optional_budget(w, pw, qids, budget)
        if kept is None:
            if policy == "relevance" or (isinstance(policy, dict) and policy["visual"] == "relevance"):
                self.set_relevance(w, pw, qids)
            kept, _ = self.reduce(pw, qids, optional, policy, random.Random(rng.random()) if rng else None)
        if not images:
            kept = [k for k in kept if pw.units[k].kind != "capture"]
        segments, counts = pw.assemble(kept, qids)
        features = {} if features is None else features
        visual = {}
        for kind, ref in segments:
            if kind == "image":
                key = V.ref_key(ref)
                if key not in features:
                    features[key] = V.load_features(ref)
                visual[key] = features[key]
        counts["N_query"] = n_query
        counts["N_query_reserved"] = query_reserve(self, w["queries"], qids)
        counts["N_total"] = padded_total(counts, encs, joint)
        if counts["N_total"] > budget:
            raise ValueError(f"sequence of {counts['N_total']} positions exceeds the total budget {budget}: "
                             f"{episode}@{origin}")
        counts["optional_budget"] = optional
        counts["optional_used"] = pw.set_cost([k for k in kept if k in pw.units], qids)
        counts["out_of_support"] = out_of_support      # audit: residual labels beyond the declared support
        # Compression denominators: every candidate text unit with its heading, and every candidate capture.
        counts["N_text_full"] = sum(len(u.ids) for u in pw.candidates(qids, "text")) + \
            sum(len(v) for k, v in pw.headings.items() if k not in pw.base_heads(qids) and k != "images")
        counts["N_captures_full"] = len(pw.captures())
        counts["N_visual_full"] = sum(u.extra - 1 for u in pw.captures())
        meta = [{k: w["queries"][i].get(k) for k in ("group", "tuple", "entity", "horizon", "state", "residual")}
                for i in idx]
        return {"episode": episode, "origin": origin, "budget": budget,
                "policy": policy if not isinstance(policy, dict) else "mix:" + json.dumps(policy, sort_keys=True),
                "segments": segments, "query_meta": meta,
                "visual_refs": [[r["observer"], r["tick"]] for k, r in segments if k == "image"],
                "visual": visual, "encs": encs, "query_index": idx, "joint": joint, "counts": counts,
                "kept": list(kept), "query_ids": list(qids), "groups": [e.group for e in encs]}


def joint_order(queries, idx, encs):
    """Time-major order: per-horizon targets by horizon, then entity/field order; interval targets last."""
    def key(j):
        q = queries[idx[j]]
        tail = q["group"] in JOINT_TAIL
        return (tail, q.get("horizon", 0), q.get("entity") or q.get("tuple") or "", q["group"], idx[j])
    return [j for j in sorted(range(len(encs)), key=key) if encs[j].full_ids is not None]


def cover_episodes(episodes, n, rng):
    """n episodes for n windows: every episode once (shuffled) before any repeats, then uniform draws."""
    eps = sorted(episodes)
    rng.shuffle(eps)
    return eps[:n] + [rng.choice(eps) for _ in range(n - len(eps))]


def query_weights(groups, inclusion=None, weights=A.GROUP_WEIGHTS):
    """Weight of every query: w_g times its inverse inclusion probability normalized within its group (§6:
    oversampled transitions are reweighted so probabilities stay calibrated), normalized over present groups."""
    inclusion = inclusion or [1.] * len(groups)
    mass = {}
    for g, p in zip(groups, inclusion):
        mass[g] = mass.get(g, 0.) + 1. / p
    total = sum(weights.get(g, 1.) for g in mass)
    return [weights.get(g, 1.) * (1. / p) / mass[g] / total for g, p in zip(groups, inclusion)]
