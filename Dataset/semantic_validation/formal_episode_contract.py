"""Source-controlled identity contract for the formal 210 episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_EPISODE_MANIFEST = (
    PROJECT_ROOT
    / "Dataset"
    / "knowledge_graph"
    / "profiles"
    / "ontology_selection_manifest.json"
)
FORMAL_SEEDS = frozenset({"seed00", "seed01", "seed02"})
FORMAL_EPI_COUNT = 70
FORMAL_EPISODE_COUNT = 210


class FormalEpisodeContractError(ValueError):
    """Raised when the checked-in formal episode identity set is invalid."""


def formal_episode_ids() -> frozenset[str]:
    value = json.loads(FORMAL_EPISODE_MANIFEST.read_text(encoding="utf-8-sig"))
    rows = value.get("episodes")
    if (
        value.get("manifest_id") != "aeroworld_210_episode_stable_module_selection"
        or value.get("episode_count") != FORMAL_EPISODE_COUNT
        or not isinstance(rows, list)
        or len(rows) != FORMAL_EPISODE_COUNT
    ):
        raise FormalEpisodeContractError(
            "formal ontology-selection manifest does not declare exact 210 episodes"
        )
    episode_ids = [
        row.get("episode_id") if isinstance(row, dict) else None for row in rows
    ]
    if any(
        not isinstance(episode_id, str) or not episode_id for episode_id in episode_ids
    ):
        raise FormalEpisodeContractError("formal episode identity is missing")
    identities = frozenset(episode_ids)
    if len(identities) != FORMAL_EPISODE_COUNT:
        raise FormalEpisodeContractError("formal episode identities are not unique")
    epi_seeds: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "episode_id",
            "scenario_id",
            "seed",
        }:
            raise FormalEpisodeContractError(
                "formal episode rows must contain exact identity fields"
            )
        episode_id = row["episode_id"]
        scenario_id = row["scenario_id"]
        seed_index = row["seed"]
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or not isinstance(seed_index, int)
            or seed_index not in {0, 1, 2}
            or episode_id != f"{scenario_id}__seed{seed_index:02d}"
        ):
            raise FormalEpisodeContractError(
                f"formal episode identity fields diverge: {row}"
            )
        if "__" not in episode_id:
            raise FormalEpisodeContractError(
                f"formal episode lacks seed suffix: {episode_id}"
            )
        epi_id, seed = episode_id.rsplit("__", 1)
        epi_seeds.setdefault(epi_id, set()).add(seed)
    if len(epi_seeds) != FORMAL_EPI_COUNT or any(
        seeds != FORMAL_SEEDS for seeds in epi_seeds.values()
    ):
        raise FormalEpisodeContractError(
            "formal episode identities are not exact 70-EPI/3-seed closure"
        )
    return identities


def require_formal_episode_set(
    episode_ids: Iterable[str],
    *,
    label: str,
) -> frozenset[str]:
    observed = frozenset(episode_ids)
    expected = formal_episode_ids()
    if observed != expected:
        raise FormalEpisodeContractError(
            f"{label} differs from the source-controlled formal episode set: "
            f"missing={sorted(expected - observed)[:10]}, "
            f"extra={sorted(observed - expected)[:10]}"
        )
    return observed
