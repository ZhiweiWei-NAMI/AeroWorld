"""Compact per-capture inputs of the predictor and the selector (RESEARCH_PLAN §3.5).

The capture feature cache keeps raw LiDAR rays and points (about 2 MB and 0.5 s per load), which is too slow
when every capture of a 5 s window is a candidate.  This pass keeps only what the models read:

* frame_cache/<collection>/<episode>/<observer>/tick_XXXXXX.pt (predictor): RGB post-merger tokens, depth and
  segmentation codec features, calibrated descriptors, the seven LiDAR region statistics and the frozen Sonata
  regional features (sonata_cache.py), in float16 where lossless enough for bf16 training.
* frame_meta/<episode>.pt (selector and window metadata), per observer and capture tick: height above ground,
  ground elevation under the camera, number of RGB tokens, the modality mask, and a summary vector of the frozen
  features [mean RGB token 2048 | mean depth 32 | mean segmentation 32 | presence-weighted mean Sonata 512 |
  mean region statistics 7] that the 0.8B selector receives through a learned projection.  Segments of
  modalities a capture does not have are zero and flagged absent by the mask (never read as observed zeros).

RGB-only captures (--rgb-only, after the four-modal pass): the fixed overview camera of every parent
("ground_a_overview_top", RGB every 0.5 s).  Its camera model comes from the per-tick capture sidecar kept in
capture_metadata/<collection>/<episode>.meta.tar.zst: a nadir pinhole (pitch -90 deg, roll 0) with square
pixels, intrinsics from the uncropped source field of view and the ROI crop of the source width.
The sidecar pose is in the truth frame (z above local ground; the capture placed the camera so its footprint at
z = 0 covers the ROI), so its ground elevation is 0.  The pass writes compact frames (RGB tokens and an 18-value
descriptor in the four-modal layout: normalized intrinsics, the camera-to-world rotation, zero translation),
adds the observer to frame_meta with modality mask RGB only, and adds its camera models to camera_index.

  python -m Dataset.world_model.unified.frame_cache --workers 32
  python -m Dataset.world_model.unified.frame_cache --rgb-only --workers 32
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import tarfile

import numpy as np
import torch
import zstandard

from Dataset.world_model.unified import schema as S

EMB = S.ROOT/"Dataset/world_model/runs/capture_embeddings"
FRAMES = EMB/"frame_cache"
META = EMB/"frame_meta"
SUMMARY_DIM = 2048 + 32 + 32 + 512 + 7
SEGMENTS = (("rgb", 2048), ("depth", 32), ("seg", 32), ("lidar_sonata", 512), ("lidar_stats", 7))
RGB, DEPTH, SEG, LIDAR = 1, 2, 4, 8               # modality mask bits
FOUR_MODAL = RGB | DEPTH | SEG | LIDAR
COLLECTIONS = ("parent_complete_210", "parent_v14_36")


def frame_path(collection, episode, observer, tick):
    return FRAMES/collection/episode/observer/f"tick_{tick:06d}.pt"


def compact(collection, episode, observer, tick):
    f = torch.load(EMB/"feature_cache"/collection/episode/observer/f"tick_{tick:06d}.pt", map_location="cpu",
                   weights_only=False)["features"]
    son = torch.load(EMB/"sonata_cache"/collection/episode/observer/f"tick_{tick:06d}.pt", map_location="cpu",
                     weights_only=False)
    b = f["binding"]
    frame = {"rgb": f["rgb"].half(), "image_grid_thw": f["image_grid_thw"], "descriptors": f["descriptors"],
             "depth": f["depth"].half(), "seg": f["seg"].half(), "lidar_region_stats": f["lidar_region_stats"].float(),
             "lidar_sonata": son["region"].half(), "lidar_present": son["present"].bool()}
    present = frame["lidar_present"].float()
    sonata = (son["region"].float() * present[:, None]).sum(0) / present.sum().clamp(min=1.)
    stats = f["lidar_region_stats"].float()
    stats = torch.cat([stats[:, :5] / 100., stats[:, 5:]], 1).mean(0)
    summary = torch.cat([f["rgb"].float().mean(0), f["depth"].float().mean(0), f["seg"].float().mean(0), sonata, stats])
    adj = b.get("capture_pose_adjustment") or {}
    ground = (b.get("ground_reference") or {}).get("projected_enu_m")
    meta = {"agl": float(adj.get("resolved_capture_altitude_agl_m", b["source_pose_agl_m"][2])),
            "ground_z": float(ground[2]) if ground else float(b["capture_position_world_m"][2] - b["source_pose_agl_m"][2]),
            "rgb_tokens": int(f["rgb"].shape[0])}
    return frame, summary.half(), meta


def episode(job):
    collection, index = job
    rows = json.loads(index.read_text())["four_modal"]
    if not rows:
        return index.stem, 0
    ep = rows[0]["episode_id"]
    by_obs = {}
    for row in sorted(rows, key=lambda r: (r["observer"], r["tick"])):
        obs, tick = row["observer"], row["tick"]
        frame, summary, meta = compact(collection, ep, obs, tick)
        path = frame_path(collection, ep, obs, tick)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(frame, tmp)
        tmp.replace(path)
        d = by_obs.setdefault(obs, {"ticks": [], "summary": [], "agl": [], "ground_z": [], "rgb_tokens": [],
                                    "modalities": []})
        d["modalities"].append(FOUR_MODAL)
        d["ticks"].append(tick)
        d["summary"].append(summary)
        for k in ("agl", "ground_z", "rgb_tokens"):
            d[k].append(meta[k])
    out = {"collection": collection, "episode": ep, "summary_dim": SUMMARY_DIM, "observers": {
        obs: {"ticks": torch.tensor(d["ticks"]), "summary": torch.stack(d["summary"]), "agl": torch.tensor(d["agl"]),
              "ground_z": torch.tensor(d["ground_z"]), "rgb_tokens": torch.tensor(d["rgb_tokens"]),
              "modalities": torch.tensor(d["modalities"])}
        for obs, d in by_obs.items()}}
    tmp = META/f"{ep}.tmp"
    torch.save(out, tmp)
    tmp.replace(META/f"{ep}.pt")
    return ep, len(rows)


def fixed_camera_model(side):
    """[12 pose values (camera_to_world, optical axes right/down/forward as columns), fx, fy, cx, cy, w, h] of a
    fixed nadir RGB camera from its capture sidecar."""
    rot = side["camera_rotation_deg"]
    if abs(rot["pitch_deg"] + 90.) > 1e-6 or abs(rot["roll_deg"]) > 1e-6:
        raise ValueError(f"fixed camera is not nadir: {rot}")
    # The engine's convention, as in the binding-derived UAV nadir camera models (camera_index.py) and checked by
    # projecting a sidecar's entity records onto its image: image up points along the yaw direction (east at
    # yaw 0, east-north-up axes) and image right 90 deg counter-clockwise from it (north at yaw 0).
    yaw = math.radians(rot["yaw_deg"])
    up = np.array([math.cos(yaw), math.sin(yaw), 0.])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.])
    pose = np.stack([right, -up, np.array([0., 0., -1.])], 1)
    pose = np.concatenate([pose, np.asarray(side["camera_pose_enu_m"], dtype=np.float64)[:, None]], 1)
    w, h = side["width"], side["height"]
    if side.get("roi_crop_applied"):
        sw, sh = side["roi_crop_source_width"], side["roi_crop_source_height"]
        u0, v0, u1, v1 = side["roi_crop_box_px"]
        if (u1 - u0, v1 - v0) != (w, h):
            raise ValueError("ROI crop was resized")
        f = (sw / 2.) / math.tan(math.radians(side["roi_crop_source_fov_degrees"]) / 2.)
        cx, cy = sw / 2. - u0, sh / 2. - v0
    else:
        f = (w / 2.) / math.tan(math.radians(side["fov_degrees"]) / 2.)
        cx, cy = w / 2., h / 2.
    return pose.reshape(-1).tolist() + [f, f, cx, cy, float(w), float(h)]


def fixed_descriptor(model):
    """18 values in the four-modal RGB descriptor layout: fx/w, fy/h, cx/w, cy/h, w/1024, h/1024, rotation (here
    camera-to-world, the camera has no body), translation / 200 (zero)."""
    fx, fy, cx, cy, w, h = model[12:]
    rot = np.asarray(model[:12]).reshape(3, 4)[:, :3]
    return torch.tensor([[fx / w, fy / h, cx / w, cy / h, w / 1024., h / 1024., *rot.reshape(-1), 0., 0., 0.]],
                        dtype=torch.float32)


def rgb_only_episode(job):
    """Compact frames, frame metadata and camera models of a parent's RGB-only captures."""
    collection, index = job
    rows = json.loads(index.read_text())["rgb_only"]
    if not rows:
        return index.stem, 0
    ep = rows[0]["binding"]["episode_id"]
    want = {r["binding"]["sidecar_member"]: r for r in rows}
    sides = {}
    with open(EMB/"capture_metadata"/collection/f"{ep}.meta.tar.zst", "rb") as fh:
        tar = tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(fh), mode="r|")
        for m in tar:
            if m.name in want:
                sides[m.name] = json.load(tar.extractfile(m))
    if set(want) - set(sides):
        raise FileNotFoundError(f"{ep}: {len(set(want) - set(sides))} RGB-only sidecars missing")
    cams = json.loads((EMB/"camera_index"/f"{ep}.json").read_text())
    meta = torch.load(META/f"{ep}.pt", weights_only=False)
    by_obs = {}
    for member, r in sorted(want.items(), key=lambda kv: (kv[1]["binding"]["camera_id"], kv[1]["tick"])):
        obs, tick = r["binding"]["camera_id"], r["tick"]
        if obs in meta["observers"] and obs not in by_obs:
            raise ValueError(f"{ep}: RGB-only camera {obs} collides with a four-modal observer")
        model = fixed_camera_model(sides[member])
        f = torch.load(r["path"], map_location="cpu", weights_only=False)["features"]
        frame = {"rgb": f["rgb"].half(), "image_grid_thw": f["image_grid_thw"],
                 "descriptors": {"rgb": fixed_descriptor(model)}, "modalities": RGB}
        path = frame_path(collection, ep, obs, tick)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(frame, tmp)
        tmp.replace(path)
        summary = torch.zeros(SUMMARY_DIM)
        summary[:2048] = f["rgb"].float().mean(0)
        d = by_obs.setdefault(obs, {"ticks": [], "summary": [], "agl": [], "ground_z": [], "rgb_tokens": [],
                                    "modalities": []})
        d["ticks"].append(tick)
        d["summary"].append(summary.half())
        d["agl"].append(float(model[11]))
        d["ground_z"].append(0.)
        d["rgb_tokens"].append(int(f["rgb"].shape[0]))
        d["modalities"].append(RGB)
        cams.setdefault(obs, {})[str(tick)] = model
    for obs, d in by_obs.items():
        meta["observers"][obs] = {"ticks": torch.tensor(d["ticks"]), "summary": torch.stack(d["summary"]),
                                  "agl": torch.tensor(d["agl"]), "ground_z": torch.tensor(d["ground_z"]),
                                  "rgb_tokens": torch.tensor(d["rgb_tokens"]), "modalities": torch.tensor(d["modalities"])}
    for o, d in meta["observers"].items():
        d.setdefault("modalities", torch.full((len(d["ticks"]),), FOUR_MODAL))
    # Camera models first: frame metadata with rgb_only=True is the completion marker of the episode.
    tmp = EMB/"camera_index"/f"{ep}.json.tmp"
    tmp.write_text(json.dumps(cams))
    tmp.replace(EMB/"camera_index"/f"{ep}.json")
    meta["rgb_only"] = True
    tmp = META/f"{ep}.tmp"
    torch.save(meta, tmp)
    tmp.replace(META/f"{ep}.pt")
    return ep, len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--rgb-only", action="store_true", help="add the RGB-only captures (after the four-modal pass)")
    p.add_argument("--episodes", nargs="*", default=None)
    args = p.parse_args()
    META.mkdir(parents=True, exist_ok=True)
    if args.rgb_only:
        jobs = []
        for c in COLLECTIONS:
            for index in sorted((EMB/"indexes"/c).glob("*.json")):
                if args.episodes is not None and index.stem not in args.episodes:
                    continue
                rows = json.loads(index.read_text())["rgb_only"]
                if rows and not torch.load(META/f"{rows[0]['binding']['episode_id']}.pt",
                                           weights_only=False).get("rgb_only"):
                    jobs.append((c, index))
        with ProcessPoolExecutor(args.workers) as pool:
            for ep, n in pool.map(rgb_only_episode, jobs):
                print(json.dumps({"episode": ep, "rgb_only_frames": n}), flush=True)
        return
    jobs = []
    for c in COLLECTIONS:
        for index in sorted((EMB/"indexes"/c).glob("*.json")):
            rows = json.loads(index.read_text())["four_modal"]
            if rows and not (META/f"{rows[0]['episode_id']}.pt").exists():
                jobs.append((c, index))
    with ProcessPoolExecutor(args.workers) as pool:
        for ep, n in pool.map(episode, jobs):
            print(json.dumps({"episode": ep, "frames": n}), flush=True)


if __name__ == "__main__":
    main()
