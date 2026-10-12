"""Captures of a window as selection candidates (RESEARCH_PLAN §3.5).

Candidates.  Every actual capture of every observer at a tick in the history window [t0 - 5 s, t0] is a
candidate unit: the four-modal UAV captures (RGB, depth, segmentation, LiDAR) and the RGB-only captures of the
fixed overview camera.  Nothing is pre-filtered by observer, modality, time or count: the selector (or, at equal
budget, a simple policy) decides what enters the predictor's 20K positions.  A capture costs the sensor tokens
of the modalities it actually has.

Metadata (known at capture time, from the capture's own binding or sidecar and the entity records up to the
capture tick): modalities, camera position in the window frame (east-north-up metres from the window reference
o, z above local ground like the entity records), height above ground, optical axis (unit vector), field of
view, and the entities projected into the image at the capture tick with their optical depth and image
position, plus the relations binding them (directly or through owned objects, window rel_entities).  The
metadata is stored structured and rendered for the query set: the entities and relations the query set
addresses are listed first, then the nearest others, up to MAX_SEEN / MAX_RELS.  The selector reads it with a
summary vector of the frozen features (frame_cache.py); the predictor reads a short frame header (observer,
time, pose, field of view, modalities) and the frame's sensor tokens.

Coordinates.  UAV camera models (camera_index.py) are in the render world, whose z is ground elevation plus
height above ground; entity records keep z above local ground, so entity points are lifted by the ground
elevation under the camera before projection (flat ground around the camera).  The fixed camera's sidecar pose
is in the truth frame (ground elevation 0).

Predictor features per capture: RGB post-merger tokens of the native path; for four-modal captures also depth
and segmentation codec features and LiDAR Sonata regional features [32, 512] with the seven region statistics
and presence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import json
import math

import numpy as np
import torch

from Dataset.world_model.data import capture_sources
from Dataset.world_model.unified import frame_cache as FC, schema as S

EMB = S.ROOT/"Dataset/world_model/runs/capture_embeddings"
FRAMES = EMB/"frame_cache"
META = EMB/"frame_meta"
DEPTH_SEG_TOKENS = 2 * 49          # depth and segmentation pooled to 7x7 each
LIDAR_TOKENS = 32
MAX_SEEN = 12                      # entities listed per capture in the selector metadata
MAX_RELS = 8
SCALE_M = 50.
MODALITY_NAMES = ((FC.RGB, "rgb"), (FC.DEPTH, "depth"), (FC.SEG, "seg"), (FC.LIDAR, "lidar"))


def modality_text(mask):
    return "+".join(n for bit, n in MODALITY_NAMES if mask & bit)


@functools.lru_cache(maxsize=32)
def capture_index(episode):
    """(collection, {observer: sorted ticks}) of every actual capture of the parent: four-modal UAV captures and
    RGB-only captures (observer = camera id)."""
    cap = capture_sources.parent_capture(episode)
    stem = cap["archive_id"].split("__", 1)[1]
    path = EMB/"indexes"/cap["collection"]/f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"no capture index for {episode}: {path}")
    raw = json.loads(path.read_text())
    out = {}
    for row in raw["four_modal"]:
        out.setdefault(row["observer"], []).append(row["tick"])
    for row in raw.get("rgb_only", ()):
        out.setdefault(row["binding"]["camera_id"], []).append(row["tick"])
    return cap["collection"], {o: sorted(t) for o, t in out.items()}


@functools.lru_cache(maxsize=32)
def cameras(episode):
    """observer -> {tick: camera model [12 pose values (3x4 camera_to_world), fx, fy, cx, cy, w, h]}."""
    raw = json.loads((EMB/"camera_index"/f"{episode}.json").read_text())
    return {obs: {int(t): np.asarray(v, dtype=np.float64) for t, v in by_tick.items()} for obs, by_tick in raw.items()}


@functools.lru_cache(maxsize=32)
def frame_meta(episode):
    """frame_cache.py metadata: observer -> {tick: row} with agl, ground_z, rgb_tokens, modalities and summary."""
    path = META/f"{episode}.pt"
    if not path.exists():
        return None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for obs, d in raw["observers"].items():
        mods = d.get("modalities")
        out[obs] = {int(t): {"agl": float(d["agl"][i]), "ground_z": float(d["ground_z"][i]),
                             "rgb_tokens": int(d["rgb_tokens"][i]), "summary": d["summary"][i],
                             "modalities": int(mods[i]) if mods is not None else FC.FOUR_MODAL}
                    for i, t in enumerate(d["ticks"].tolist())}
    return out


@dataclass
class Capture:
    collection: str
    episode: str
    observer: str
    tick: int
    model: np.ndarray
    agl: float
    ground_z: float
    rgb_tokens: int
    summary: torch.Tensor
    modalities: int = FC.FOUR_MODAL
    info: dict = field(default_factory=dict)       # window-specific metadata (describe)

    @property
    def key(self):
        return f"c:{self.observer}@{self.tick}"

    @property
    def positions(self):
        """LM positions of the frame's sensor tokens: RGB, + 2 x 49 pooled depth/segmentation, + 32 LiDAR regions."""
        return (self.rgb_tokens + (DEPTH_SEG_TOKENS if self.modalities & FC.DEPTH else 0)
                + (LIDAR_TOKENS if self.modalities & FC.LIDAR else 0))

    def ref(self):
        return {"collection": self.collection, "episode": self.episode, "observer": self.observer, "tick": self.tick}


def window_captures(episode, origin):
    """Every capture of every observer in [t0 - 5 s, t0], ordered by (tick, observer).

    Requires the camera index and the frame metadata of the parent (camera_index.py, frame_cache.py, including
    its --rgb-only pass): a capture without them raises instead of being dropped or costed with a guess.
    """
    collection, index = capture_index(episode)
    ticks = [(o, t) for o, ts in index.items() for t in ts if origin - S.HISTORY_TICKS <= t <= origin]
    if not ticks:
        return []
    cams, meta = cameras(episode), frame_meta(episode)
    if meta is None:
        raise FileNotFoundError(f"no frame metadata for {episode}: run frame_cache.py")
    out = []
    for observer, t in ticks:
        model = cams.get(observer, {}).get(t)
        m = meta.get(observer, {}).get(t)
        if model is None or m is None:
            raise KeyError(f"capture {episode} {observer}@{t} lacks a camera model or frame metadata")
        out.append(Capture(collection, episode, observer, t, model, m["agl"], m["ground_z"], m["rgb_tokens"],
                           m["summary"], m["modalities"]))
    return sorted(out, key=lambda c: (c.tick, c.observer))


def project(model, points):
    """Optical depth and normalized image coordinates of world points; inside = in front and in the image."""
    pose = model[:12].reshape(3, 4)
    fx, fy, cx, cy, w, h = model[12:]
    rot, t = pose[:, :3], pose[:, 3]
    cam = (points - t) @ rot                 # world -> optical (right, down, forward)
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return z, u / w, v / h, inside


def seen(cap, src):
    """[(entity id, depth m, u, v)] of entities present at the capture tick and inside the image, nearest first."""
    ents = [e for e, pos in src.position_raw.items() if cap.tick in pos and e != cap.observer]
    if not ents:
        return []
    pts = np.asarray([src.position_raw[e][cap.tick] for e in ents], dtype=np.float64)
    pts[:, 2] += cap.ground_z
    z, u, v, inside = project(cap.model, pts)
    return sorted(((e, float(z[i]), float(u[i]), float(v[i])) for i, e in enumerate(ents) if inside[i]),
                  key=lambda r: r[1])


def relevance(cap, src, addressed):
    """Geometric relevance at prediction time: sum over visible entities of weight / (1 + depth / 50 m), weight 1
    for entities the query set addresses (ids) and 0.1 otherwise."""
    return sum((1. if e in addressed else .1) / (1. + z / SCALE_M) for e, z, _, _ in seen(cap, src))


def observer_alias(observer, aliases, episode):
    """The observer's entity alias (UAV cameras), else cam<k> over the parent's non-entity observers."""
    if observer in aliases:
        return aliases[observer]
    others = sorted(o for o in capture_index(episode)[1] if o not in aliases)
    return f"cam{others.index(observer) + 1}"


def _f(x, nd):
    return f"{round(float(x), nd) + 0.:.{nd}f}"


def pose_text(cap, frame_origin_cm):
    """Camera position in the window frame, height, optical axis and field of view."""
    pose = cap.model[:12].reshape(3, 4)
    fx, fy, _, _, w, h = cap.model[12:]
    o = [c / 100. for c in frame_origin_cm]
    p = [pose[0, 3] - o[0], pose[1, 3] - o[1], pose[2, 3] - cap.ground_z - o[2]]
    look = pose[:, 2] / max(1e-9, np.linalg.norm(pose[:, 2]))
    hfov, vfov = (math.degrees(2 * math.atan(w / (2 * fx))), math.degrees(2 * math.atan(h / (2 * fy))))
    return (f"p=[{_f(p[0], 1)},{_f(p[1], 1)},{_f(p[2], 1)}] agl={_f(cap.agl, 1)} "
            f"look=[{_f(look[0], 2)},{_f(look[1], 2)},{_f(look[2], 2)}] fov={hfov:.0f}x{vfov:.0f}")


def describe(cap, src, window, ts):
    """Predictor frame header of a capture; stores the structured selector metadata in cap.info."""
    aliases = window["aliases"]
    obs = observer_alias(cap.observer, aliases, cap.episode)
    when = ts(cap.tick, window["origin"])
    pose = pose_text(cap, window["frame_origin_cm"])
    mods = modality_text(cap.modalities)
    vis = [(aliases[e], z, u, v) for e, z, u, v in seen(cap, src) if e in aliases]
    names = {a for a, *_ in vis}
    rels = [r for r, ents in window["rel_entities"].items() if names & set(ents)]
    cap.info = {"observer": obs, "when": when, "pose": pose, "modalities": mods, "seen": vis, "rels": rels}
    return f"image observer={obs} t={when} {pose} {mods}:"


def metadata_line(cap, addressed=frozenset(), queried_rels=frozenset()):
    """Selector metadata of a capture: addressed entities and queried relations first, then the nearest."""
    i = cap.info
    vis = sorted(i["seen"], key=lambda r: (r[0] not in addressed, r[1]))
    parts = [f"{a} d={z:.0f} uv=({u:.2f},{v:.2f})" for a, z, u, v in vis[:MAX_SEEN]]
    more = f" +{len(vis) - MAX_SEEN} more" if len(vis) > MAX_SEEN else ""
    rels = sorted(i["rels"], key=lambda r: (r not in queried_rels, int(r[1:])))
    rel_more = f" +{len(rels) - MAX_RELS}" if len(rels) > MAX_RELS else ""
    return (f"  t={i['when']} capture observer={i['observer']} {i['modalities']} {i['pose']} {cap.positions} tokens"
            f" sees {'; '.join(parts) if parts else 'none'}{more}"
            + (f" rels {','.join(rels[:MAX_RELS])}{rel_more}" if rels else ""))


def load_features(ref):
    """Predictor inputs of a capture: the compact frame cache (required for RGB-only captures), else the full
    four-modal feature cache plus Sonata."""
    path = FRAMES/ref["collection"]/ref["episode"]/ref["observer"]/f"tick_{ref['tick']:06d}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    f = torch.load(EMB/"feature_cache"/ref["collection"]/ref["episode"]/ref["observer"]/f"tick_{ref['tick']:06d}.pt",
                   map_location="cpu", weights_only=False)["features"]
    son = torch.load(EMB/"sonata_cache"/ref["collection"]/ref["episode"]/ref["observer"]/f"tick_{ref['tick']:06d}.pt",
                     map_location="cpu", weights_only=False)
    return {"rgb": f["rgb"], "image_grid_thw": f["image_grid_thw"], "descriptors": f["descriptors"],
            "depth": f["depth"], "seg": f["seg"], "lidar_region_stats": f["lidar_region_stats"],
            "lidar_sonata": son["region"].float(), "lidar_present": son["present"]}


def ref_key(ref):
    return (ref["episode"], ref["observer"], ref["tick"])
