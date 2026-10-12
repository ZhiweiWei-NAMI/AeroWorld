"""Fail-closed loader for the exact ROI repair and UE recapture scope."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from Dataset.tools.roi_contract import (
    CAPTURE_STEP_TICKS,
    EXPECTED_CAPTURE_FRAME_COUNT,
    EXPECTED_TICK_END,
    EXPECTED_TICK_START,
    RELEASE_SEEDS,
    TARGET_RECAPTURE_SCENARIO_IDS,
    load_rebuild_contracts,
)


DEFAULT_SCOPE_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "roi_repair_recapture_scope.json"
)
DEFAULT_REBUILD_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "roi_rebuild_contracts.json"
)


@dataclass(frozen=True)
class TargetedUavObservation:
    profile: str
    seed: int
    tick: int
    actor: str
    target_class: str

    @property
    def episode_id(self) -> str:
        return f"{self.profile}__seed{self.seed:02d}"


@dataclass(frozen=True)
class RoiRepairScope:
    path: Path
    payload: Mapping[str, Any]
    profiles: tuple[str, ...]
    rebuilt_profiles: frozenset[str]
    preserved_profiles: frozenset[str]
    uav_observations: tuple[TargetedUavObservation, ...]

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{profile}__seed{seed:02d}"
            for profile in self.profiles
            for seed in RELEASE_SEEDS
        )

    @property
    def rebuilt_episode_ids(self) -> frozenset[str]:
        return frozenset(
            f"{profile}__seed{seed:02d}"
            for profile in self.rebuilt_profiles
            for seed in RELEASE_SEEDS
        )

    @property
    def preserved_episode_ids(self) -> frozenset[str]:
        return frozenset(self.episode_ids) - self.rebuilt_episode_ids


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _targeted_uav_observations(payload: Mapping[str, Any]) -> tuple[TargetedUavObservation, ...]:
    observations: list[TargetedUavObservation] = []
    inspect_targets = payload.get("inspect_black_frame_targets")
    if not isinstance(inspect_targets, Mapping):
        raise TypeError("inspect_black_frame_targets must be an object")
    for profile, raw_ticks in inspect_targets.items():
        if profile not in TARGET_RECAPTURE_SCENARIO_IDS:
            raise ValueError(f"unknown inspect target profile: {profile}")
        if not isinstance(raw_ticks, list) or not raw_ticks:
            raise ValueError(f"{profile}: inspect target ticks must be a non-empty list")
        actor = f"u_inspect_{profile.lower().replace('-', '_')}"
        for tick in raw_ticks:
            tick_value = int(tick)
            for seed in RELEASE_SEEDS:
                observations.append(
                    TargetedUavObservation(
                        profile=str(profile),
                        seed=seed,
                        tick=tick_value,
                        actor=actor,
                        target_class="inspect_uav",
                    )
                )

    secondary_targets = payload.get("secondary_black_frame_targets")
    if not isinstance(secondary_targets, list):
        raise TypeError("secondary_black_frame_targets must be a list")
    for index, raw_target in enumerate(secondary_targets):
        if not isinstance(raw_target, Mapping):
            raise TypeError(f"secondary_black_frame_targets[{index}] must be an object")
        profile = str(raw_target.get("profile") or "")
        actor = str(raw_target.get("actor") or "")
        raw_observations = raw_target.get("observations")
        if profile not in TARGET_RECAPTURE_SCENARIO_IDS or not actor:
            raise ValueError(f"secondary_black_frame_targets[{index}] has invalid identity")
        if not isinstance(raw_observations, list) or not raw_observations:
            raise ValueError(f"secondary_black_frame_targets[{index}] lacks observations")
        for raw_observation in raw_observations:
            if not isinstance(raw_observation, Mapping):
                raise TypeError(f"secondary_black_frame_targets[{index}] observation must be an object")
            observations.append(
                TargetedUavObservation(
                    profile=profile,
                    seed=int(raw_observation.get("seed")),
                    tick=int(raw_observation.get("tick")),
                    actor=actor,
                    target_class="secondary_uav",
                )
            )

    identities = [
        (item.profile, item.seed, item.tick, item.actor)
        for item in observations
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("targeted UAV observations contain duplicate actor/tick identities")
    for item in observations:
        if item.seed not in RELEASE_SEEDS:
            raise ValueError(f"{item}: seed is outside the release contract")
        if not EXPECTED_TICK_START <= item.tick <= EXPECTED_TICK_END:
            raise ValueError(f"{item}: tick is outside the release contract")
    inspect_count = sum(item.target_class == "inspect_uav" for item in observations)
    secondary_count = len(observations) - inspect_count
    if inspect_count != 69 or secondary_count != 15:
        raise ValueError(
            "targeted UAV contract must contain exactly 69 inspect and 15 secondary checks; "
            f"got {inspect_count} and {secondary_count}"
        )
    return tuple(observations)


def load_roi_repair_scope(
    path: Path = DEFAULT_SCOPE_PATH,
    *,
    rebuild_contract_path: Path = DEFAULT_REBUILD_CONTRACT_PATH,
) -> RoiRepairScope:
    resolved_path = path.resolve()
    payload = _read_object(resolved_path)
    if payload.get("schema_name") != "aeroworld_roi_repair_recapture_scope":
        raise ValueError(f"{resolved_path}: unexpected schema_name")
    if payload.get("schema_version") != "2.0.0":
        raise ValueError(f"{resolved_path}: unsupported schema_version")

    profiles = tuple(str(item) for item in payload.get("profiles") or ())
    if profiles != TARGET_RECAPTURE_SCENARIO_IDS:
        raise ValueError(
            f"{resolved_path}: profiles must exactly match TARGET_RECAPTURE_SCENARIO_IDS"
        )
    if int(payload.get("profile_count")) != len(profiles):
        raise ValueError(f"{resolved_path}: profile_count is inconsistent")
    if tuple(int(item) for item in payload.get("seeds") or ()) != RELEASE_SEEDS:
        raise ValueError(f"{resolved_path}: release seeds must be {RELEASE_SEEDS}")
    if int(payload.get("episode_count")) != len(profiles) * len(RELEASE_SEEDS):
        raise ValueError(f"{resolved_path}: episode_count is inconsistent")
    if [int(item) for item in payload.get("episode_ticks") or ()] != [
        EXPECTED_TICK_START,
        EXPECTED_TICK_END,
    ]:
        raise ValueError(f"{resolved_path}: episode tick contract diverges")
    if int(payload.get("capture_step_ticks")) != CAPTURE_STEP_TICKS:
        raise ValueError(f"{resolved_path}: capture step contract diverges")
    if int(payload.get("capture_frames_per_episode")) != EXPECTED_CAPTURE_FRAME_COUNT:
        raise ValueError(f"{resolved_path}: capture frame count diverges")

    roi_policy = payload.get("roi_policy")
    if not isinstance(roi_policy, Mapping):
        raise TypeError(f"{resolved_path}: roi_policy must be an object")
    rebuilt_profiles = frozenset(
        str(item) for item in roi_policy.get("full_chain_rebuild_profiles") or ()
    )
    contract_profiles = frozenset(load_rebuild_contracts(rebuild_contract_path))
    if rebuilt_profiles != contract_profiles:
        raise ValueError(
            f"{resolved_path}: rebuilt profiles {sorted(rebuilt_profiles)} diverge from "
            f"{rebuild_contract_path}: {sorted(contract_profiles)}"
        )
    preserved_profiles = frozenset(profiles) - rebuilt_profiles
    if int(roi_policy.get("full_chain_rebuild_profile_count")) != len(rebuilt_profiles):
        raise ValueError(f"{resolved_path}: rebuilt profile count is inconsistent")
    if int(roi_policy.get("full_chain_rebuild_episode_count")) != len(rebuilt_profiles) * 3:
        raise ValueError(f"{resolved_path}: rebuilt episode count is inconsistent")
    if int(roi_policy.get("unchanged_profile_count")) != len(preserved_profiles):
        raise ValueError(f"{resolved_path}: unchanged profile count is inconsistent")
    if int(roi_policy.get("unchanged_episode_count")) != len(preserved_profiles) * 3:
        raise ValueError(f"{resolved_path}: unchanged episode count is inconsistent")

    return RoiRepairScope(
        path=resolved_path,
        payload=payload,
        profiles=profiles,
        rebuilt_profiles=rebuilt_profiles,
        preserved_profiles=preserved_profiles,
        uav_observations=_targeted_uav_observations(payload),
    )
