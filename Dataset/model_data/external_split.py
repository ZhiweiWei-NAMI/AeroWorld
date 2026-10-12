"""Attach an explicit episode split to immutable, split-neutral graph bytes."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping

from .contract import ModelDataError


SPLITS = frozenset({"train", "validation", "test"})


def load_external_episode_splits(path: Path, episode_profiles: Mapping[str, str]) -> dict[str, str]:
    """Validate complete episode assignments and family isolation before loading windows."""
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    records = manifest.get("episodes")
    if not isinstance(records, list) or not records:
        raise ModelDataError("external split manifest has no episode assignments")
    assignments = {}
    grouped = defaultdict(set)
    for row in records:
        episode = row.get("episode_id")
        profile = row.get("profile_id")
        family = row.get("scenario_family_id")
        split = row.get("split")
        if any(not isinstance(value, str) or not value for value in (episode, profile, family)):
            raise ModelDataError("external split has missing episode/profile/family identity")
        if split not in SPLITS or episode in assignments:
            raise ModelDataError("external split has an invalid split or duplicate episode")
        assignments[episode] = split
        for kind, identity in (("profile", profile), ("family", family)):
            grouped[kind, identity].add(split)
        if episode in episode_profiles and episode_profiles[episode] != profile:
            raise ModelDataError(f"external profile differs from graph source: {episode}")
    if any(len(values) != 1 for values in grouped.values()):
        raise ModelDataError("external split crosses a scenario family or profile")
    missing = set(episode_profiles) - assignments.keys()
    if missing:
        raise ModelDataError(f"graph episode missing external assignment: {sorted(missing)[0]}")
    return {episode: assignments[episode] for episode in episode_profiles}
