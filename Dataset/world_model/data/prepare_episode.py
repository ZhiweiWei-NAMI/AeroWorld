"""Prepare one episode/observer for state-distribution windows.

Extracts the observer's capture view from the episode archive into scratch
space, runs the canonical importer on it, materializes every capture tick that
carries all four modalities, then removes the scratch copy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

import zstandard

from Dataset.world_model.data import capture_sources, import_episode
from Dataset.world_model.data.observations import CaptureArchive, MODALITIES, materialize_observations


def extract_view(archive, observer, scratch):
    """Extract the single view ending in ``__<observer>``; return (view root, view name, common ticks)."""
    view, ticks = None, {}
    with open(archive, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream, \
            tarfile.open(fileobj=stream, mode="r|") as members:
        for member in members:
            parts = member.name.split("/")
            if len(parts) < 4 or not parts[2].endswith("__"+observer):
                continue
            if view is not None and parts[2] != view:
                raise ValueError(f"observer {observer} resolves to capture views {view} and {parts[2]}")
            view = parts[2]
            members.extract(member, scratch, filter="data")
            match = re.fullmatch(r"tick_(\d{6})\.\w+", parts[-1]) if len(parts) == 5 else None
            if match:
                ticks.setdefault(parts[3], set()).add(int(match.group(1)))
    if view is None:
        raise ValueError(f"archive {archive} has no capture view for observer {observer}")
    if set(ticks) != set(MODALITIES):
        raise ValueError(f"view {view} carries modalities {sorted(ticks)}, expected {sorted(MODALITIES)}")
    common = sorted(t for t in set.intersection(*ticks.values()) if t > 0)
    return parts[0], parts[1], view, common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--observer", required=True)
    parser.add_argument("--output", type=Path, default=Path("Dataset/world_model/runs/state_distribution"))
    parser.add_argument("--scratch", type=Path, required=True, help="directory for the temporary view extraction")
    args = parser.parse_args()
    # The episode's one authoritative capture (a v14 recapture replaces the original).
    archive = Path(capture_sources.parent_capture(args.episode)["archive_rel"])
    inputs = args.output/"inputs"/args.episode
    observations = args.output/"observations"/args.episode/args.observer
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.scratch) as scratch:
        profile, seed, view, ticks = extract_view(archive, args.observer, scratch)
        import_episode._cli(["--episode", args.episode, "--view", view, "--observer", args.observer,
                             "--capture-root", str(Path(scratch)/profile/seed),
                             "--out-derived", str(inputs/"canonical.jsonl"), "--out-manifest", str(inputs/"import.json")])
    shutil.rmtree(observations, ignore_errors=True)
    source = CaptureArchive(archive, episode_id=args.episode, observer=args.observer, ticks=ticks)
    materialize_observations(source, observations, cutoff=ticks[-1])
    print(json.dumps({"episode": args.episode, "observer": args.observer, "view": view,
                      "canonical": str(inputs/"canonical.jsonl"), "observations": str(observations),
                      "capture_ticks": ticks}))


if __name__ == "__main__":
    main()
