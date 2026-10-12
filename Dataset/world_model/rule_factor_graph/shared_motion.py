"""Frozen TRAIN prefix innovations sampled once per actor and future branch.

This pilot has one UAV actor per communication scope. A donor is one joint
six-field mean motion innovation from an observed prefix. Repeated scopes for
the same actor/cutoff reuse the same draw; station
and flow identities do not enter its random seed or cache key.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .communication_data import CommunicationPredictionInput, CommunicationWindow, _frozen_array
from .contracts import BranchState, GroundingError, RoleBinding, StateKey, Variable, VariableKind


MOTION_FIELDS = ("truth_pose.position_enu_m.x", "truth_pose.position_enu_m.y",
                 "truth_pose.position_enu_m.z", "truth_pose.velocity_enu_mps.x",
                 "truth_pose.velocity_enu_mps.y", "truth_pose.velocity_enu_mps.z")
BRANCH_FEATURE_NAMES = (
    "horizon_fraction", "distance_over_nominal_range", "speed_over_10mps",
    "radial_speed_over_10mps", "relative_station_x_over_nominal_range",
    "relative_station_y_over_nominal_range", "relative_station_z_over_nominal_range",
    "velocity_x_over_10mps", "velocity_y_over_10mps", "velocity_z_over_10mps",
)


@dataclass(frozen=True)
class SharedMotionBranches:
    episode_id: str
    actor_id: str
    ticks: tuple[int, ...]
    sample_id: str
    states: np.ndarray
    branch_features: np.ndarray
    donor_ids: tuple[str, ...]
    source_refs: tuple[str, ...]

    def __post_init__(self):
        states = _frozen_array(self.states, np.float64)
        features = _frozen_array(self.branch_features, np.float64)
        if states.ndim != 4 or states.shape[1:] != (len(self.ticks), 1, 6):
            raise GroundingError("shared motion branches must have shape [K,T,1,6]")
        if features.shape != (len(states), len(self.ticks), len(BRANCH_FEATURE_NAMES)):
            raise GroundingError("branch feature shape differs from the declared motion fields")
        if len(self.donor_ids) != len(states) or not np.isfinite(states).all() or not np.isfinite(features).all():
            raise GroundingError("motion draws need complete finite states and one donor per branch")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "branch_features", features)

    def as_branch_state(self, branch: int) -> BranchState:
        """Expose existing phase1 typed state contract without learnable q values."""
        if not 0 <= branch < len(self.states):
            raise IndexError("motion branch index is out of range")
        bindings = (RoleBinding("actor", self.actor_id, "world:UnmannedAircraft"),)
        variables = tuple(Variable(
            StateKey(self.actor_id, field, self.episode_id, tick, f"{self.sample_id}:branch={branch}"),
            VariableKind.STATE, observed=False, role_bindings=bindings, source_refs=self.source_refs)
            for tick in self.ticks for field in MOTION_FIELDS)
        values = torch.as_tensor(self.states[branch].copy(), dtype=torch.float64).reshape(-1)
        return BranchState(variables, values, torch.ones_like(values, dtype=torch.bool), self.source_refs)

    def tensor_features(self, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
        return torch.as_tensor(self.branch_features.copy(), dtype=dtype, device=device).detach()


@dataclass(frozen=True)
class _MotionDonor:
    window_id: str
    component: str
    source_refs: tuple[str, ...]


class FrozenSharedMotionQ:
    """Offline TRAIN calibration; prediction reads an input and this frozen bank.

    q samples a joint six-field mean one-step prefix innovation, then repeats
    it recursively after constant-velocity updates over the finite horizon.
    This persistence is a declared proposal assumption. Calibration shares
    TRAIN data with potential fitting, explicitly
    reported in diagnostics. TRAIN queries exclude their entire component;
    held-out queries use TRAIN donors only. There is no communication state in
    the donor bank and no posterior update of its uniform branch weights.
    """
    def __init__(self):
        self._innovations: np.ndarray | None = None
        self._donors: tuple[_MotionDonor, ...] = ()
        self._history_steps: int | None = None
        self._future_steps: int | None = None
        self._cache: dict[tuple, tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray]] = {}
        self.diagnostics: dict = {}

    def fit(self, training: Sequence[CommunicationWindow]) -> FrozenSharedMotionQ:
        if self._innovations is not None:
            raise RuntimeError("q is frozen after one offline TRAIN fit")
        if not training or any(window.prediction.split != "train" for window in training):
            raise GroundingError("q calibration accepts a nonempty TRAIN-only window sequence")
        shapes = {(len(window.prediction.prefix), len(window.prediction.ticks) - 1) for window in training}
        if len(shapes) != 1:
            raise GroundingError("q calibration window lengths must agree")
        self._history_steps, self._future_steps = next(iter(shapes))
        seen = set()
        innovations = []
        donors = []
        duplicates = 0
        for window in sorted(training, key=lambda item: item.prediction.window_id):
            query = window.prediction
            if not np.isfinite(query.motion_history).all():
                raise GroundingError("complete donor prefix motion must be finite")
            # Full copies are removed by raw data equality, not digest comparison.
            key = (tuple(query.motion_history.ravel()), query.component)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            interval = (query.prefix[1].tick - query.prefix[0].tick) / query.tick_hz
            predicted_next = query.motion_history[:-1].copy()
            predicted_next[..., :3] += query.motion_history[:-1, :, 3:] * interval
            innovations.append((query.motion_history[1:] - predicted_next).mean(axis=0))
            donors.append(_MotionDonor(query.window_id, query.component, query.motion_source_refs))
        if not donors:
            raise GroundingError("TRAIN has no complete prefix motion donor windows")
        if len({donor.component for donor in donors}) < 2:
            raise GroundingError("TRAIN q calibration needs two components for own-component exclusion")
        self._innovations = _frozen_array(np.stack(innovations), np.float64)
        self._donors = tuple(donors)
        self.diagnostics = {
            "model": "recursive_constant_velocity_plus_uniform_joint_mean_prefix_innovation",
            "fit_split": "train", "calibration_shares_potential_training_split": True,
            "communication_state_in_bank": False, "future_state_in_bank": False,
            "motion_actor_type": "world:UnmannedAircraft",
            "training_windows": len(training), "unique_complete_donors": len(donors),
            "full_motion_prefix_copies_removed": duplicates,
            "components": sorted({donor.component for donor in donors}),
            "train_query_component_exclusion": True, "future_steps": self._future_steps,
            "sample_key": "episode/cutoff/actor/field/time/branch; station and flow excluded",
            "branch_weights": "uniform frozen proposal; no posterior reweighting",
            "innovation_persistence": "one joint six-field prefix innovation repeated over the finite horizon",
            "innovation_std": np.std(self._innovations, axis=0).ravel().tolist(),
            "claim": "finite empirical motion proposal; no exact continuous dynamics integration",
        }
        return self

    def sample(self, query: CommunicationPredictionInput, k: int, seed: int) -> SharedMotionBranches:
        if not isinstance(query, CommunicationPredictionInput):
            raise TypeError("q prediction accepts CommunicationPredictionInput only")
        if self._innovations is None:
            raise RuntimeError("fit the frozen offline TRAIN q bank first")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1 or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("k must be positive and seed must be a nonnegative integer")
        if (len(query.prefix), len(query.ticks) - 1) != (self._history_steps, self._future_steps):
            raise GroundingError("query lengths differ from the frozen q bank")
        key = (query.episode_id, query.cutoff_tick, query.owner.actor_id, k, seed, query.component, query.split)
        if key in self._cache:
            states, donor_ids, source_refs, prefix = self._cache[key]
            if not np.array_equal(prefix, query.motion_history):
                raise GroundingError("same actor/cutoff identity carries different motion prefixes")
        else:
            allowed = np.asarray([index for index, donor in enumerate(self._donors)
                                  if query.split != "train" or donor.component != query.component], dtype=np.int64)
            if not len(allowed):
                raise GroundingError("q query has no TRAIN donor outside its component")
            # SeedSequence consumes exact identity code points, avoiding process-randomized hash().
            identity = f"{query.episode_id}|{query.cutoff_tick}|{query.owner.actor_id}"
            generator = np.random.default_rng(np.random.SeedSequence([seed, *map(ord, identity)]))
            chosen = allowed[generator.integers(0, len(allowed), size=k)]
            intervals = np.diff(np.asarray(query.ticks, dtype=np.float64)) / query.tick_hz
            last = np.broadcast_to(query.motion_history[-1], (k, 1, 6)).copy()
            future = np.empty((k, len(intervals), 1, 6), dtype=np.float64)
            for step, interval in enumerate(intervals):
                last = last.copy()
                last[..., :3] += last[..., 3:] * interval
                last += self._innovations[chosen]
                future[:, step] = last
            anchor = np.broadcast_to(query.motion_history[-1], (k, 1, 1, 6))
            states = _frozen_array(np.concatenate((anchor, future), axis=1), np.float64)
            donor_ids = tuple(self._donors[index].window_id for index in chosen)
            source_refs = tuple(dict.fromkeys(ref for index in chosen for ref in self._donors[index].source_refs))
            self._cache[key] = (states, donor_ids, source_refs, query.motion_history)
        features = _branch_features(query, states)
        sample_id = f"q:{query.episode_id}:actor={query.owner.actor_id}:cutoff={query.cutoff_tick}:seed={seed}"
        return SharedMotionBranches(query.episode_id, query.owner.actor_id, query.ticks, sample_id,
                                    states, features, donor_ids, source_refs)

    def donor_manifest(self) -> list[dict]:
        if self._innovations is None:
            raise RuntimeError("fit q before exporting its donor manifest")
        return [{"window_id": donor.window_id, "component": donor.component,
                 "source_refs": list(donor.source_refs)} for donor in self._donors]


def _branch_features(query: CommunicationPredictionInput, states: np.ndarray) -> np.ndarray:
    actor = states[:, :, 0]
    relative = actor[..., :3] - query.station_position
    distance = np.linalg.norm(relative, axis=-1)
    speed = np.linalg.norm(actor[..., 3:], axis=-1)
    radial = np.zeros_like(distance)
    separated = distance > 0
    radial[separated] = np.sum(relative[separated] * actor[..., 3:][separated], axis=-1) / distance[separated]
    horizon = np.arange(len(query.ticks), dtype=np.float64) / (len(query.ticks) - 1)
    return np.concatenate((np.broadcast_to(horizon[None, :, None], (*distance.shape, 1)),
        (distance / query.nominal_range_m)[..., None], (speed / 10.0)[..., None],
        (radial / 10.0)[..., None], relative / query.nominal_range_m, actor[..., 3:] / 10.0), axis=-1)
