"""Frozen real-future set scoring and the frozen TRAIN/DEV job schedule.

Future labels enter only predictor losses and selector rewards.
"""
from __future__ import annotations

from collections import Counter
import random
import time

import torch

from Dataset.world_model.unified import arms, assemble as A, data as D
from Dataset.world_model.unified.predictor import Predictor
from Dataset.world_model.unified.train_predictor import item_losses, load_trainable


class SetScorer:
    """L_theta(S; I) for kept sets of one window with fixed queries; frozen features reused across sets."""
    def __init__(self, model, enc, episode, origin, rng, total, pair=None, which=None,
                 query_ids=None, inclusion=None):
        self.model, self.enc = model, enc
        self.episode, self.origin = episode, origin
        self.pair, self.which = pair, which
        self.total = total
        if pair is None:
            self.w, self.pw = enc.window(episode, origin)
            if query_ids is None:
                self.qids = self.pw.select_queries(rng)
            else:
                if inclusion is None:
                    raise ValueError("saved query selection requires its actual inclusion probabilities")
                self.qids = list(query_ids)
                self.pw.inclusion = inclusion
            self.budget = enc.optional_budget(self.w, self.pw, self.qids, total)
        else:
            self.w, self.pw, _ = arms.arm_window(enc, pair)
            self.qids = list(range(len(self.w["queries"])))
            self.budget = arms.optional_budget(enc, pair, total)
        enc.set_relevance(self.w, self.pw, self.qids)
        self.features, self.cache = {}, {}
        self.calls, self.seconds = 0, 0.0
        self.actions_ids = self.pw.action_ids if pair is None else self.w["arm_action_ids"][
            pair["arm_key"] if which == "arm" else pair["factual_key"]]

    @torch.no_grad()
    def __call__(self, kept):
        key = tuple(sorted(kept))
        if key not in self.cache:
            started = time.perf_counter()
            if self.pair is None:
                item = self.enc.item(self.episode, self.origin, kept=list(key), query_ids=self.qids, budget=self.total,
                                     prepared=(self.w, self.pw), features=self.features)
            else:
                item = arms.arm_item(self.enc, self.pair, self.which, self.total, "relevance", kept=list(key))
            if not item["encs"]:
                raise ValueError(f"no real future labels: {self.episode}@{self.origin}")
            total, groups, _ = item_losses(self.model, item, 1, train=False, modes=("marginal",))
            self.calls += 1
            self.seconds += time.perf_counter() - started
            self.cache[key] = {"loss": total, "groups": groups, "counts": item["counts"],
                               "query_counts": dict(Counter(e.group for e in item["encs"]))}
        return self.cache[key]

    def cost(self, keys):
        return self.pw.set_cost(keys, self.qids)


def scoring_window(model, enc, job, record=None):
    # Jobs already come from the frozen valid-pair schedule. Do not rebuild every
    # ARM's future labels just to construct this single requested context.
    key = job["arm_key"]
    pair = None if key is None else {"arm_key": key, "factual_key": arms._factual_key(key),
        "parent_episode": job["episode"], "origin": job["origin"], "group": key.rsplit("/", 1)[0]}
    rng = random.Random(job["seed"]) if job["seed"] is not None else None
    return SetScorer(model, enc, job["episode"], job["origin"], rng, A.TOTAL_BUDGET, pair, job["which"],
                     query_ids=record["query_ids"] if record is not None else None,
                     inclusion=record["inclusion"] if record is not None else None)


def make_jobs(split, windows, seed, arm_every=6):
    """One shared TRAIN schedule per round; fixed DEV parents plus all matched DEV arms."""
    rng = random.Random(seed)
    n_arm = windows // arm_every if split == "train" else 0
    parents = iter(D.cover_episodes(D.splits()[split], windows - n_arm, rng))
    by_profile = {}
    if n_arm:
        for members in arms.groups(split).values():
            by_profile.setdefault(members[0]["profile"], []).append(members)
        if not by_profile:
            raise ValueError("TRAIN has no ARM groups")
    jobs = []

    def add(episode, origin, pair=None, which=None):
        jobs.append({"id": len(jobs), "split": split, "episode": episode, "origin": origin,
                     "arm_key": pair["arm_key"] if pair is not None else None, "which": which,
                     "seed": rng.randrange(2**31) if split == "train" else None})

    for i in range(windows):
        if split == "train" and (i + 1) % arm_every == 0:
            members = rng.choice(by_profile[rng.choice(sorted(by_profile))])
            member = rng.randrange(len(members) + 1)
            pair = members[member - 1] if member else members[0]
            add(pair["parent_episode"], pair["origin"], pair, "arm" if member else "factual")
        else:
            episode = next(parents)
            add(episode, rng.choice(D.ORIGINS))
    if split != "train":
        for pair in arms.pairs(split):
            for which in ("factual", "arm"):
                add(pair["parent_episode"], pair["origin"], pair, which)
    return jobs


def frozen_predictor(path, config):
    """Frozen 2B scorer; query_chunk queries continue from the shared context state per forward."""
    model = Predictor(device="cuda", lora_rank=config["predictor_lora_rank"], query_chunk=config["predictor_query_chunk"])
    load_trainable(model, path)
    model.eval()
    model.requires_grad_(False)
    return model
