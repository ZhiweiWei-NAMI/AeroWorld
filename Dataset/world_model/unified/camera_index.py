"""Per-capture camera models of every parent four-modal capture, for the causal viewpoint ranking (visual.py).

From each capture's own binding (known at capture time): camera_to_world = body_to_world @ camera_to_body
(optical axes right/down/forward, world = map x/y/up) and the intrinsics of the encoded RGB image.
Output: runs/capture_embeddings/camera_index/<episode>.json  {observer: {tick: [12 pose values, fx, fy, cx, cy, w, h]}}

  python -m Dataset.world_model.unified.camera_index --workers 32
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import torch

from Dataset.world_model.unified import schema as S

EMB = S.ROOT/"Dataset/world_model/runs/capture_embeddings"
OUT = EMB/"camera_index"


def camera(path):
    b = torch.load(path, map_location="cpu", weights_only=False)["features"]["binding"]
    body = np.asarray(b["lidar_sensor_pose"]["body_to_world"], dtype=np.float64)
    phys = b["calibrated_descriptors"]["rgb"]["physical"]
    cam = body @ np.asarray(phys["camera_to_body"], dtype=np.float64)
    k = np.asarray(phys["intrinsic"], dtype=np.float64)
    return cam[:3].reshape(-1).tolist() + [k[0, 0], k[1, 1], k[0, 2], k[1, 2], phys["width"], phys["height"]]


def episode(args):
    collection, index = args
    raw = json.loads(index.read_text())
    out = defaultdict(dict)
    for row in raw["four_modal"]:
        path = EMB/"feature_cache"/collection/row["episode_id"]/row["observer"]/f"tick_{row['tick']:06d}.pt"
        out[row["observer"]][str(row["tick"])] = camera(path)
    ep = raw["four_modal"][0]["episode_id"] if raw["four_modal"] else index.stem
    (OUT/f"{ep}.json").write_text(json.dumps(out))
    return ep, sum(len(v) for v in out.values())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, i) for c in ("parent_complete_210", "parent_v14_36") for i in sorted((EMB/"indexes"/c).glob("*.json"))
            if not (OUT/i.name).exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for ep, n in pool.map(episode, jobs):
            print(ep, n, flush=True)


if __name__ == "__main__":
    main()
