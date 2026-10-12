"""Frozen Sonata regional LiDAR features for every parent four-modal capture (RESEARCH_PLAN §3.5).

The cached capture features keep the acquired LiDAR cloud as angular-cell points (coordinates, normals,
geometry channels, coarse region index), so the frozen Sonata encoder runs from the cache without the raw
archives.  Sonata's input features are [centred xyz, colour, normal]; LiDAR has no colour, so the colour
slots are fixed to zero (a colourless cloud) instead of the trainable slot adapter of the earlier pipeline,
which keeps the features frozen and cacheable.  Point features are pooled into the 32 support regions:
region [32, 512] (float16) and region presence [32].  A scan without returns gives zeros and no presence.

  CUDA_VISIBLE_DEVICES=5 python -m Dataset.world_model.unified.sonata_cache --shard 0 --shards 1
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader, Dataset

from Dataset.world_model.model import multimodal as mm
from Dataset.world_model.unified import schema as S

EMB = S.ROOT/"Dataset/world_model/runs/capture_embeddings"
OUT = EMB/"sonata_cache"
SONATA = S.ROOT/"Dataset/world_model/models/sonata"
SIGNATURE = {"encoder": "sonata_small (point_transformer_v3_s)", "colour_slots": "zeros", "grid_m": 0.1,
             "pooling": "mean over 32 coarse support regions"}


def frames():
    out = []
    for collection in ("parent_complete_210", "parent_v14_36"):
        for index in sorted((EMB/"indexes"/collection).glob("*.json")):
            for row in json.loads(index.read_text())["four_modal"]:
                out.append((collection, row["episode_id"], row["observer"], row["tick"]))
    return out


def out_path(collection, episode, observer, tick):
    return OUT/collection/episode/observer/f"tick_{tick:06d}.pt"


class Frames(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        collection, episode, observer, tick = self.items[i]
        f = torch.load(EMB/"feature_cache"/collection/episode/observer/f"tick_{tick:06d}.pt", map_location="cpu",
                       weights_only=False)["features"]
        return self.items[i], {k: f[k] for k in ("lidar_coordinates", "lidar_normals", "lidar_geometry_channels",
                                                 "coarse_cell_indices")}


class _Frame:
    def __init__(self, d, device):
        for k, v in d.items():
            setattr(self, k, v.to(device))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    todo = [x for i, x in enumerate(frames()) if i % args.shards == args.shard and not out_path(*x).exists()]
    sonata, _ = mm.load_sonata(SONATA, "cuda")
    colour = lambda g: torch.zeros(len(g), 3, device=g.device, dtype=g.dtype)
    loader = DataLoader(Frames(todo), batch_size=None, num_workers=args.workers, prefetch_factor=8)
    t0, done = time.time(), 0
    for key, d in loader:
        path = out_path(*key)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = d["lidar_coordinates"].shape[0]
        if n == 0:
            region, present = torch.zeros(32, 512), torch.zeros(32, dtype=torch.bool)
        else:
            with torch.no_grad():
                region, present = mm.sonata_features(_Frame(d, "cuda"), sonata, colour)
        tmp = path.with_suffix(".tmp")
        torch.save({"region": region.half().cpu(), "present": present.cpu(), "signature": SIGNATURE}, tmp)
        tmp.replace(path)
        done += 1
        if done % 2000 == 0:
            print(json.dumps({"done": done, "todo": len(todo), "seconds": round(time.time() - t0)}), flush=True)
    print(json.dumps({"done": done, "todo": len(todo), "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
