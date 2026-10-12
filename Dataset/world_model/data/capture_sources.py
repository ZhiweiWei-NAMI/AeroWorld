"""Authoritative capture version of every parent episode and counterfactual arm.

The dataset is 210 canonical parent episodes plus their arms.  37 parents were
recaptured in v14 (``capture_archives/parent_v14_36``); a recapture REPLACES the
episode's earlier capture in ``parent_complete_210``.  It is never an extra
scene, an extra training example or an independent split member, whatever its
file hash.  Exactly one capture is selected per parent:

  * the v14 recapture when one exists, otherwise the parent_complete_210 capture;
  * the superseded capture is not embedded, loaded or counted.

Episode ids are unchanged by the replacement, so split membership (resolved from
the episode's original identity) and arm -> parent references stay as they are.
Counterfactual arms carry their own capture; factual arms have no capture of
their own and are anchored to their parent's selected capture.

``configs/capture_sources.json`` is the committed mapping; ``--write`` rebuilds
it from the archive inventory (archive identities are kept after the raw
archives are deleted).
"""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT/"Dataset/world_model/configs/capture_sources.json"
INVENTORY = ROOT/"Dataset/world_model/runs/capture_embeddings/inventory.json"
ORIGINAL, RECAPTURE, ARMS = "parent_complete_210", "parent_v14_36", "arm_multimodal_375"
PARENTS = 210


def _identity(row):
    return {"archive_id": row["id"], "collection": row["collection"], "archive_rel": row["archive_rel"],
            "sha256": row["sha256"], "compressed_bytes": row["compressed_bytes"],
            "captures_4modal": row["captures_4modal"], "captures_rgb_only": row["captures_rgb_only"]}


def build(inventory, arms_index):
    from Dataset.world_model.data.provenance import SOURCE_SPLITS
    originals = {r["episode"]: r for r in inventory["archives"] if r["collection"] == ORIGINAL}
    recaptures = {r["episode"]: r for r in inventory["archives"] if r["collection"] == RECAPTURE}
    if len(originals) != PARENTS or not set(recaptures) <= set(originals):
        raise ValueError(f"expected {PARENTS} parents with recaptures among them; got {len(originals)} and "
                         f"{sorted(set(recaptures)-set(originals))}")
    parents = {}
    for episode in sorted(originals):
        replacement = recaptures.get(episode)
        parents[episode] = {"version": "v14_recapture" if replacement else "original",
                            "capture": _identity(replacement or originals[episode]),
                            "superseded": _identity(originals[episode]) if replacement else None,
                            "split": SOURCE_SPLITS.get(episode)}
    captured = {r["arm"]["key"]: r for r in inventory["archives"] if r["collection"] == ARMS}
    arms = {}
    for arm in arms_index:
        factual = arm["arm_id"] in ("arm_factual", "factual")
        parent = f'{arm["profile"]}__{arm["seed"]}'
        if parent not in parents:
            raise ValueError(f"arm {arm['key']} references unknown parent {parent}")
        if factual == (arm["key"] in captured):
            raise ValueError(f"arm {arm['key']}: factual={factual} but captured={arm['key'] in captured}")
        arms[arm["key"]] = {"parent_episode": parent, "factual": factual,
                            "capture": {"anchored_to_parent": parent} if factual else _identity(captured[arm["key"]])}
    if set(captured) - set(arms):
        raise ValueError(f"captured arms missing from arms/index.jsonl: {sorted(set(captured)-set(arms))}")
    versions = Counter(p["version"] for p in parents.values())
    return {"rule": "one capture per parent: the v14 recapture replaces the original; superseded captures are "
                    "never embedded, loaded or counted; factual arms anchor to the parent's selected capture",
            "counts": {"parents": len(parents), "parents_original_capture": versions["original"],
                       "parents_v14_recapture": versions["v14_recapture"],
                       "superseded_captures": versions["v14_recapture"],
                       "arms_counterfactual_captured": sum(not a["factual"] for a in arms.values()),
                       "arms_factual_anchored": sum(a["factual"] for a in arms.values())},
            "parents": parents, "arms": arms}


@functools.lru_cache(maxsize=1)
def load():
    return json.loads(SOURCES.read_text())


def parent_capture(episode):
    """Selected capture identity of a parent episode."""
    return load()["parents"][episode]["capture"]


def arm_capture(key):
    """Capture identity of an arm: its own (counterfactual) or its parent's selected capture (factual)."""
    arm = load()["arms"][key]
    return parent_capture(arm["parent_episode"]) if arm["factual"] else arm["capture"]


def selected_archive_ids():
    sources = load()
    return {p["capture"]["archive_id"] for p in sources["parents"].values()} | \
           {a["capture"]["archive_id"] for a in sources["arms"].values() if not a["factual"]}


def superseded_archive_ids():
    """Superseded archive id -> id of the recapture that replaces it."""
    return {p["superseded"]["archive_id"]: p["capture"]["archive_id"]
            for p in load()["parents"].values() if p["superseded"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--write", action="store_true", help="rebuild configs/capture_sources.json")
    args = parser.parse_args()
    if args.write:
        inventory = json.loads(args.inventory.read_text())
        arms_index = [json.loads(line) for line in (ROOT/"arms/index.jsonl").read_text().splitlines() if line.strip()]
        SOURCES.write_text(json.dumps(build(inventory, arms_index), indent=1, sort_keys=True)+"\n")
        load.cache_clear()
    print(json.dumps(load()["counts"]))


if __name__ == "__main__":
    main()
