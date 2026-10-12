"""TRAIN-only motion proposal and shared, standardized six-component CV cache.

The proposal keeps ``FrozenSharedMotionQ``'s empirical donor law. TRAIN
queries exclude their component; held-out queries use all TRAIN components.
Prediction prefixes alone fit q and feature statistics. Future supervision has
no role in either operation. Arrays are saved once per fold and shared by all
four models and all optimizer seeds.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .communication_data import (CommunicationPredictionInput, CommunicationWindow,
                                 HISTORY_FEATURE_NAMES, _frozen_array)
from .contracts import GroundingError
from .shared_motion import (BRANCH_FEATURE_NAMES, FrozenSharedMotionQ,
                            SharedMotionBranches, _branch_features)


CACHE_FORMAT = "communication_component_cv_q_v1"
MEAN_CACHE_FORMAT = "communication_component_cv_q_mean_v1"
NORMALIZATION_ZERO_STD = 1e-12


class FrozenComponentMotionQ(FrozenSharedMotionQ):
    """Preserve q draws while avoiding a donor-bank scan for every query.

    The small draw cache bounds retained future states. Prefix identity checks
    remain valid after draw-cache eviction, and station/flow fields remain
    absent from the sample identity.
    """

    def __init__(self, *, max_cached_actor_draws: int = 256):
        if type(max_cached_actor_draws) is not int or max_cached_actor_draws < 1:
            raise ValueError("max_cached_actor_draws must be a positive integer")
        super().__init__()
        self._max_cached_actor_draws = max_cached_actor_draws
        self._cache = OrderedDict()
        self._allowed_by_component: dict[str, np.ndarray] = {}
        self._all_donors: np.ndarray | None = None
        self._actor_prefixes: dict[tuple, tuple] = {}

    def fit(self, training: Sequence[CommunicationWindow]) -> "FrozenComponentMotionQ":
        if any(not isinstance(window, CommunicationWindow) for window in training):
            raise TypeError("q fit requires CommunicationWindow entries; only prediction fields are read")
        super().fit(training)
        self._all_donors = np.arange(len(self._donors), dtype=np.int64)
        components = tuple(sorted({donor.component for donor in self._donors}))
        for component in components:
            allowed = np.asarray([index for index, donor in enumerate(self._donors)
                                  if donor.component != component], dtype=np.int64)
            if not len(allowed):
                raise GroundingError(f"TRAIN component {component} has no external donor support")
            allowed.setflags(write=False)
            self._allowed_by_component[component] = allowed
        self._all_donors.setflags(write=False)
        self.diagnostics.update({
            "component_donor_counts": {component: sum(donor.component == component for donor in self._donors)
                                       for component in components},
            "train_allowed_donor_counts": {component: len(self._allowed_by_component[component])
                                           for component in components},
            "proposal_component_mismatch": "TRAIN excludes one of five TRAIN groups; held-out uses all five",
            "draw_cache_capacity_actor_keys": self._max_cached_actor_draws,
            "zero_support_policy": "raise; no donor backoff",
        })
        return self

    def sample(self, query: CommunicationPredictionInput, k: int, seed: int) -> SharedMotionBranches:
        if not isinstance(query, CommunicationPredictionInput):
            raise TypeError("q prediction accepts CommunicationPredictionInput only")
        if self._innovations is None or self._all_donors is None:
            raise RuntimeError("fit the frozen offline TRAIN q bank first")
        if type(k) is not int or k < 1 or type(seed) is not int or seed < 0:
            raise ValueError("k must be positive and seed must be a nonnegative integer")
        if (len(query.prefix), len(query.ticks) - 1) != (self._history_steps, self._future_steps):
            raise GroundingError("query lengths differ from the frozen q bank")
        if query.split == "train":
            if query.component not in self._allowed_by_component:
                raise GroundingError("TRAIN query component was not fitted by q")
            allowed = self._allowed_by_component[query.component]
        else:
            if query.component in self._allowed_by_component:
                raise GroundingError("held-out query component occurs in the TRAIN donor bank")
            allowed = self._all_donors
        identity_key = (query.episode_id, query.cutoff_tick, query.owner.actor_id)
        signature = (query.component, query.split, query.ticks, query.tick_hz)
        if identity_key in self._actor_prefixes:
            previous_prefix, previous_signature = self._actor_prefixes[identity_key]
            if signature != previous_signature or not np.array_equal(previous_prefix, query.motion_history):
                raise GroundingError("same actor/cutoff identity carries inconsistent prefix or fold metadata")
        else:
            self._actor_prefixes[identity_key] = (query.motion_history, signature)
        key = (*identity_key, k, seed, query.component, query.split)
        if key in self._cache:
            states, donor_ids, source_refs = self._cache.pop(key)
            self._cache[key] = (states, donor_ids, source_refs)
        else:
            # Keep the exact seed, donor ordering and recurrence of the original q.
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
            self._cache[key] = (states, donor_ids, source_refs)
            if len(self._cache) > self._max_cached_actor_draws:
                self._cache.popitem(last=False)
        features = _branch_features(query, states)
        sample_id = f"q:{query.episode_id}:actor={query.owner.actor_id}:cutoff={query.cutoff_tick}:seed={seed}"
        return SharedMotionBranches(query.episode_id, query.owner.actor_id, query.ticks, sample_id,
                                    states, features, donor_ids, source_refs)


def _moments(rows: np.ndarray, indices: np.ndarray, *, batch_size: int = 256):
    """Merge TRAIN block moments without materializing all training branches."""
    count = 0
    mean = np.zeros(rows.shape[-1], dtype=np.float64)
    square_sum = np.zeros_like(mean)
    for start in range(0, len(indices), batch_size):
        block = np.asarray(rows[indices[start:start + batch_size]], dtype=np.float64).reshape(-1, rows.shape[-1])
        if not np.isfinite(block).all():
            raise GroundingError("normalization TRAIN features must be finite")
        block_mean = block.mean(axis=0)
        block_square_sum = ((block - block_mean) ** 2).sum(axis=0)
        updated_count = count + len(block)
        delta = block_mean - mean
        square_sum += block_square_sum + delta ** 2 * (count * len(block) / updated_count)
        mean += delta * (len(block) / updated_count)
        count = updated_count
    if not count:
        raise GroundingError("normalization requires nonempty TRAIN observations")
    std = np.sqrt(square_sum / count)
    constant = std <= NORMALIZATION_ZERO_STD
    scale = np.where(constant, 1.0, std)
    return _frozen_array(mean), _frozen_array(scale), _frozen_array(std), _frozen_array(constant, bool), count


@dataclass(frozen=True)
class FrozenTrainingNormalizer:
    history_mean: np.ndarray
    history_scale: np.ndarray
    branch_mean: np.ndarray
    branch_scale: np.ndarray
    diagnostics: dict

    @classmethod
    def fit(cls, training: Sequence[CommunicationWindow], branch_features: np.ndarray) -> "FrozenTrainingNormalizer":
        """Fit on raw training history and its already sampled raw q features."""
        if not training or any(window.prediction.split != "train" for window in training):
            raise GroundingError("normalizer fit accepts a nonempty TRAIN-only sequence")
        histories = np.stack([window.prediction.history for window in training])
        if branch_features.shape[0] != len(training) or branch_features.ndim != 4:
            raise GroundingError("normalizer needs one raw q branch-feature row per TRAIN window")
        return cls._fit_arrays(histories, branch_features, np.arange(len(training)), training)

    @classmethod
    def _fit_arrays(cls, histories, branches, training_indices, training):
        hm, hs, hstd, hc, hn = _moments(histories, training_indices)
        bm, bs, bstd, bc, bn = _moments(branches, training_indices)
        if len(hm) != len(HISTORY_FEATURE_NAMES) or len(bm) != len(BRANCH_FEATURE_NAMES):
            raise GroundingError("normalizer dimensions differ from the declared feature order")
        diagnostics = {
            "fit_split": "train", "training_windows": len(training),
            "training_components": sorted({window.prediction.component for window in training}),
            "history_feature_names": list(HISTORY_FEATURE_NAMES),
            "branch_feature_names": list(BRANCH_FEATURE_NAMES),
            "history_observations": hn, "branch_observations": bn,
            "history_raw_std": hstd.tolist(), "branch_raw_std": bstd.tolist(),
            "history_constant_features": [name for name, flag in zip(HISTORY_FEATURE_NAMES, hc) if flag],
            "branch_constant_features": [name for name, flag in zip(BRANCH_FEATURE_NAMES, bc) if flag],
            "constant_feature_scale": 1.0, "zero_std_threshold": NORMALIZATION_ZERO_STD,
            "population_std": True, "future_supervision_used": False,
            "training_zero_event_information_windows_included": True,
            "weighting": "equal prefix observations; equal TRAIN-query branch/time observations",
        }
        return cls(hm, hs, bm, bs, diagnostics)

    def transform_history(self, history):
        result = (np.asarray(history, dtype=np.float64) - self.history_mean) / self.history_scale
        if not np.isfinite(result).all():
            raise GroundingError("standardized history must be finite")
        return result

    def transform_branches(self, features):
        result = (np.asarray(features, dtype=np.float64) - self.branch_mean) / self.branch_scale
        if not np.isfinite(result).all():
            raise GroundingError("standardized branch features must be finite")
        return result

    def manifest(self):
        return {"history_mean": self.history_mean.tolist(), "history_scale": self.history_scale.tolist(),
                "branch_mean": self.branch_mean.tolist(), "branch_scale": self.branch_scale.tolist(),
                "diagnostics": self.diagnostics}

    @classmethod
    def from_manifest(cls, value):
        arrays = [_frozen_array(value[name], np.float64) for name in
                  ("history_mean", "history_scale", "branch_mean", "branch_scale")]
        if tuple(array.shape for array in arrays) != ((16,), (16,), (10,), (10,)):
            raise GroundingError("saved normalizer has wrong feature dimensions")
        if any(not np.isfinite(array).all() for array in arrays) or any((arrays[index] <= 0).any() for index in (1, 3)):
            raise GroundingError("saved normalizer has nonfinite values or nonpositive scales")
        if value["diagnostics"]["fit_split"] != "train":
            raise GroundingError("saved normalizer was not fitted on TRAIN")
        return cls(*arrays, value["diagnostics"])


def _prediction_manifest(query: CommunicationPredictionInput):
    return {"window_id": query.window_id, "episode_id": query.episode_id, "family": query.family,
            "component": query.component, "split": query.split, "cutoff_tick": query.cutoff_tick,
            "ticks": list(query.ticks), "actor_id": query.owner.actor_id,
            "owner_id": query.owner.owner_id, "station_position": query.station_position.tolist(),
            "nominal_range_m": query.nominal_range_m, "tick_hz": query.tick_hz,
            "initial_predicates": list(query.initial_predicates), "initial_joint_state": query.initial_joint_state,
            "profile_id": query.profile_id, "profile_source_path": query.profile_source_path,
            "communication_prefix_sources": [f"{obs.source_path}#line={obs.source_line}" for obs in query.prefix],
            "motion_source_refs": list(query.motion_source_refs), "station_source_ref": query.station_source_ref}


@dataclass(frozen=True)
class FoldMotionCache:
    cache_dir: Path
    window_ids: tuple[str, ...]
    history: np.ndarray
    branch_features: np.ndarray
    motion_states: np.ndarray
    donor_indices: np.ndarray
    normalizer: FrozenTrainingNormalizer
    manifest: dict

    @property
    def index_by_window_id(self):
        return {window_id: index for index, window_id in enumerate(self.window_ids)}

    def batch(self, indices):
        selected = np.asarray(indices)
        if not np.issubdtype(selected.dtype, np.integer):
            raise TypeError("cache batch indices must have an integer dtype")
        selected = selected.astype(np.int64, copy=False)
        if selected.ndim != 1 or not len(selected) or (selected < 0).any() or (selected >= len(self.window_ids)).any():
            raise ValueError("cache batch indices must be a nonempty valid one-dimensional index array")
        return np.asarray(self.history[selected]), np.asarray(self.branch_features[selected])

    def donor_records(self, index: int):
        if type(index) is not int or not 0 <= index < len(self.window_ids):
            raise IndexError("cache donor row index is out of range")
        donors = self.manifest["donors"]
        return tuple(donors[int(value)] for value in self.donor_indices[index])


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def build_fold_cache(training: Sequence[CommunicationWindow], queries: Sequence[CommunicationWindow],
                     cache_dir: Path | str, *, k: int = 8, q_seed: int = 0) -> FoldMotionCache:
    """Write one immutable fold cache, or verify and reuse its completed cache.

    Existing incomplete directories are errors. Their evidence is left intact
    for explicit diagnosis. No failed fit or empty donor pool changes the law.
    """
    training, queries = tuple(training), tuple(queries)
    if not training or not queries or any(not isinstance(window, CommunicationWindow) for window in (*training, *queries)):
        raise TypeError("fold cache needs nonempty CommunicationWindow training and query sequences")
    if type(k) is not int or k < 1 or type(q_seed) is not int or q_seed < 0:
        raise ValueError("k must be positive and q_seed must be a nonnegative integer")
    if any(window.prediction.split != "train" for window in training):
        raise GroundingError("cache calibration is TRAIN-only")
    query_ids = tuple(window.prediction.window_id for window in queries)
    training_ids = tuple(window.prediction.window_id for window in training)
    if len(set(query_ids)) != len(query_ids) or len(set(training_ids)) != len(training_ids):
        raise GroundingError("fold cache window identities must be unique")
    by_id = {window.prediction.window_id: window for window in queries}
    if set(training_ids) != {window.prediction.window_id for window in queries if window.prediction.split == "train"}:
        raise GroundingError("q fitting windows must exactly equal the cache TRAIN queries")
    for window in training:
        query = by_id[window.prediction.window_id].prediction
        if (_prediction_manifest(query) != _prediction_manifest(window.prediction)
                or not np.array_equal(query.history, window.prediction.history)
                or not np.array_equal(query.motion_history, window.prediction.motion_history)):
            raise GroundingError("TRAIN fit input differs from its cache query prefix")
    train_components = {window.prediction.component for window in training}
    if any(window.prediction.split not in ("train", "test") for window in queries):
        raise GroundingError("six-component fold cache queries must be TRAIN or held-out TEST")
    if any(window.prediction.component in train_components for window in queries if window.prediction.split == "test"):
        raise GroundingError("held-out component occurs in cache TRAIN inputs")
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        result = load_fold_cache(cache_dir, expected_windows=queries)
        if result.manifest["k"] != k or result.manifest["q_seed"] != q_seed:
            raise GroundingError("existing cache K or qseed differs from the requested fixed proposal")
        return result
    q = FrozenComponentMotionQ().fit(training)
    shapes = {(len(window.prediction.prefix), len(window.prediction.ticks)) for window in queries}
    if len(shapes) != 1:
        raise GroundingError("fold query prefix and horizon dimensions must agree")
    history_steps, time_steps = next(iter(shapes))
    n = len(queries)
    cache_dir.mkdir(parents=True)
    history = np.lib.format.open_memmap(cache_dir / "history.npy", mode="w+", dtype=np.float64,
                                       shape=(n, history_steps, len(HISTORY_FEATURE_NAMES)))
    branches = np.lib.format.open_memmap(cache_dir / "branch_features.npy", mode="w+", dtype=np.float64,
                                        shape=(n, k, time_steps, len(BRANCH_FEATURE_NAMES)))
    states = np.lib.format.open_memmap(cache_dir / "motion_states.npy", mode="w+", dtype=np.float64,
                                      shape=(n, k, time_steps, 1, 6))
    prefixes = np.lib.format.open_memmap(cache_dir / "motion_prefixes.npy", mode="w+", dtype=np.float64,
                                        shape=(n, history_steps, 1, 6))
    donor_indices = np.lib.format.open_memmap(cache_dir / "donor_indices.npy", mode="w+", dtype=np.int64,
                                             shape=(n, k))
    donors = q.donor_manifest()
    donor_by_id = {row["window_id"]: index for index, row in enumerate(donors)}
    donor_components = tuple(row["component"] for row in donors)
    components = tuple(sorted(train_components))
    actual_counts = {component: {donor_component: 0 for donor_component in components}
                     for component in sorted({window.prediction.component for window in queries})}
    for index, window in enumerate(queries):
        query = window.prediction
        draw = q.sample(query, k, q_seed)
        history[index] = query.history
        branches[index] = draw.branch_features
        states[index] = draw.states
        prefixes[index] = query.motion_history
        chosen = [donor_by_id[donor_id] for donor_id in draw.donor_ids]
        donor_indices[index] = chosen
        for donor_index in chosen:
            donor_component = donor_components[donor_index]
            if query.split == "train" and donor_component == query.component:
                raise GroundingError("q used an own-component TRAIN donor")
            actual_counts[query.component][donor_component] += 1
    training_indices = np.asarray([index for index, window in enumerate(queries)
                                   if window.prediction.split == "train"], dtype=np.int64)
    normalizer = FrozenTrainingNormalizer._fit_arrays(history, branches, training_indices, training)
    for start in range(0, n, 256):
        history[start:start + 256] = normalizer.transform_history(history[start:start + 256])
        branches[start:start + 256] = normalizer.transform_branches(branches[start:start + 256])
    for value in (history, branches, states, prefixes, donor_indices):
        value.flush()
    np.save(cache_dir / "donor_innovations.npy", q._innovations)
    manifest = {
        "format": CACHE_FORMAT, "completed": True, "k": k, "q_seed": q_seed,
        "window_ids": list(query_ids), "training_window_ids": list(training_ids),
        "training_components": list(components),
        "query_inputs": [_prediction_manifest(window.prediction) for window in queries],
        "donors": donors, "calibration": q.diagnostics, "normalizer": normalizer.manifest(),
        "actual_draw_counts_by_query_component_and_donor_component": actual_counts,
        "donor_indices_file": "donor_indices.npy", "donor_source_refs": "donors[donor_indices[row,branch]].source_refs",
        "arrays_standardized": ["history.npy", "branch_features.npy"],
        "physical_motion_states_standardized": False, "all_models_and_training_seeds_share_cache": True,
        "future_supervision_used_for_q_or_normalization": False,
        "prediction_prefixes_saved_for_resume_identity_check": True,
    }
    _write_json(cache_dir / "manifest.json", manifest)
    del history, branches, states, prefixes, donor_indices
    return load_fold_cache(cache_dir, expected_windows=queries)


def load_fold_cache(cache_dir: Path | str, *, expected_windows: Sequence[CommunicationWindow] | None = None) -> FoldMotionCache:
    """Read a completed cache; with current windows, recheck prefixes and q fit."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["format"] != CACHE_FORMAT or manifest["completed"] is not True:
        raise GroundingError("fold q cache format is unsupported or its build is incomplete")
    normalizer = FrozenTrainingNormalizer.from_manifest(manifest["normalizer"])
    ids = tuple(manifest["window_ids"])
    if not ids or len(set(ids)) != len(ids) or len(manifest["query_inputs"]) != len(ids):
        raise GroundingError("saved cache identities are empty, repeated or inconsistent")
    arrays = {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("history", "branch_features", "motion_states", "motion_prefixes", "donor_indices")}
    n, k = len(ids), manifest["k"]
    history_steps = len(manifest["query_inputs"][0]["communication_prefix_sources"])
    time_steps = len(manifest["query_inputs"][0]["ticks"])
    shapes = {"history": (n, history_steps, 16), "branch_features": (n, k, time_steps, 10),
              "motion_states": (n, k, time_steps, 1, 6), "motion_prefixes": (n, history_steps, 1, 6),
              "donor_indices": (n, k)}
    for name, array in arrays.items():
        if array.shape != shapes[name] or array.dtype != (np.dtype(np.int64) if name == "donor_indices" else np.dtype(np.float64)):
            raise GroundingError(f"saved cache {name} shape or dtype differs from its contract")
        if not np.isfinite(array).all():
            raise GroundingError(f"saved cache {name} has nonfinite values")
    donor_count = len(manifest["donors"])
    if not donor_count or (arrays["donor_indices"] < 0).any() or (arrays["donor_indices"] >= donor_count).any():
        raise GroundingError("saved cache donor indices have empty or invalid support")
    if expected_windows is not None:
        expected_windows = tuple(expected_windows)
        if tuple(window.prediction.window_id for window in expected_windows) != ids:
            raise GroundingError("current fold query order or identities differ from the saved cache")
        training = tuple(window for window in expected_windows if window.prediction.split == "train")
        if set(window.prediction.window_id for window in training) != set(manifest["training_window_ids"]):
            raise GroundingError("current TRAIN identities differ from the saved cache")
        q = FrozenComponentMotionQ().fit(training)
        saved_innovations = np.load(cache_dir / "donor_innovations.npy", allow_pickle=False)
        if q.donor_manifest() != manifest["donors"] or not np.array_equal(q._innovations, saved_innovations):
            raise GroundingError("current TRAIN q bank differs from the saved cache")
        for index, window in enumerate(expected_windows):
            query = window.prediction
            if (_prediction_manifest(query) != manifest["query_inputs"][index]
                    or not np.array_equal(query.motion_history, arrays["motion_prefixes"][index])
                    or not np.array_equal(normalizer.transform_history(query.history), arrays["history"][index])):
                raise GroundingError("current prediction prefix or metadata differs from the saved cache")
            donor_components = [manifest["donors"][int(value)]["component"] for value in arrays["donor_indices"][index]]
            if query.split == "train" and query.component in donor_components:
                raise GroundingError("saved TRAIN cache contains an own-component donor")
            if query.split == "test" and query.component in manifest["training_components"]:
                raise GroundingError("saved held-out component occurs in the TRAIN donor bank")
            draw = q.sample(query, manifest["k"], manifest["q_seed"])
            saved_donor_ids = tuple(manifest["donors"][int(value)]["window_id"]
                                    for value in arrays["donor_indices"][index])
            if (saved_donor_ids != draw.donor_ids
                    or not np.array_equal(draw.states, arrays["motion_states"][index])
                    or not np.array_equal(normalizer.transform_branches(draw.branch_features),
                                          arrays["branch_features"][index])):
                raise GroundingError("saved q donor sequence, states or branch features differ from deterministic replay")
        train_indices = np.asarray([index for index, window in enumerate(expected_windows)
                                    if window.prediction.split == "train"], dtype=np.int64)
        for name, feature_names, constant_names in (
                ("history", HISTORY_FEATURE_NAMES, normalizer.diagnostics["history_constant_features"]),
                ("branch_features", BRANCH_FEATURE_NAMES, normalizer.diagnostics["branch_constant_features"])):
            mean, _, std, _, _ = _moments(arrays[name], train_indices)
            varying = np.asarray([feature not in constant_names for feature in feature_names], dtype=bool)
            if (np.max(np.abs(mean)) > 1e-7 or np.max(np.abs(std[varying] - 1.0), initial=0.0) > 1e-7
                    or np.max(std[~varying], initial=0.0) > 1e-7):
                raise GroundingError("saved standardized TRAIN feature moments differ from zero mean/unit variance")
    return FoldMotionCache(cache_dir, ids, arrays["history"], arrays["branch_features"], arrays["motion_states"],
                           arrays["donor_indices"], normalizer, manifest)


def conditional_mean_motion_center(query: CommunicationPredictionInput, mean_innovation: np.ndarray, *, k: int = 8):
    """Return exact conditional E_q[state], then f(E_q[state]), copied K times.

    Each B8 branch draws one uniform eligible donor's joint innovation and
    repeats it over all future steps. The update is linear, so expectation
    obeys the same recursion with the full eligible bank's mean innovation.
    At dt=.5: E[v_t]=v0+t*mu_v and
    E[p_t]=p0+.5*t*v0+t*mu_p+.25*t*(t-1)*mu_v. Floating-point execution uses
    the original repeated update order. Neither future labels nor K sampled
    paths determine this mean; nonlinear features are computed after it.
    """
    if not isinstance(query, CommunicationPredictionInput):
        raise TypeError("conditional motion mean accepts CommunicationPredictionInput only")
    if type(k) is not int or k < 1:
        raise ValueError("center branch count must be a positive integer")
    mean_innovation = np.asarray(mean_innovation)
    if mean_innovation.shape != (1, 6) or mean_innovation.dtype != np.dtype(np.float64) or not np.isfinite(mean_innovation).all():
        raise GroundingError("conditional innovation mean must be finite float64[1,6]")
    intervals = np.diff(np.asarray(query.ticks, dtype=np.float64)) / query.tick_hz
    last = np.broadcast_to(query.motion_history[-1], (k, 1, 6)).copy()
    states = np.empty((k, len(query.ticks), 1, 6), dtype=np.float64)
    states[:, 0] = last
    for step, interval in enumerate(intervals, start=1):
        last = last.copy()
        last[..., :3] += last[..., 3:] * interval
        last += mean_innovation
        states[:, step] = last
    return _frozen_array(states), _frozen_array(_branch_features(query, states))


def conditional_motion_mean_innovations(b8_cache: FoldMotionCache):
    """Mean each exact conditional donor law, with no component reweighting.

    TRAIN keys name the excluded query component. ``heldout_all_train`` uses
    every frozen TRAIN donor. Component weights therefore follow donor counts.
    """
    innovations = np.load(b8_cache.cache_dir / "donor_innovations.npy", allow_pickle=False)
    donors = b8_cache.manifest["donors"]
    if (innovations.shape != (len(donors), 1, 6) or innovations.dtype != np.dtype(np.float64)
            or not len(donors) or not np.isfinite(innovations).all()):
        raise GroundingError("B8 frozen joint innovation bank is empty or malformed")
    components = np.asarray([row["component"] for row in donors])
    means = {"heldout_all_train": _frozen_array(innovations.mean(axis=0))}
    for component in b8_cache.manifest["training_components"]:
        allowed = components != component
        if not allowed.any():
            raise GroundingError("conditional motion mean has no donor outside the TRAIN query component")
        means[component] = _frozen_array(innovations[allowed].mean(axis=0))
    return means


def _query_motion_mean(query, means):
    if query.split == "train":
        if query.component not in means:
            raise GroundingError("conditional mean TRAIN query component is absent from its frozen bank")
        return means[query.component]
    if query.component in means:
        raise GroundingError("conditional mean held-out component occurs in the frozen TRAIN bank")
    return means["heldout_all_train"]


@dataclass(frozen=True)
class ConditionalMeanCache:
    """Bmean q-mean inputs; history and normalizer are immutable B8 objects."""
    cache_dir: Path
    window_ids: tuple[str, ...]
    history: np.ndarray
    branch_features: np.ndarray
    motion_states: np.ndarray
    normalizer: FrozenTrainingNormalizer
    manifest: dict

    @property
    def index_by_window_id(self):
        return {window_id: index for index, window_id in enumerate(self.window_ids)}

    def batch(self, indices):
        selected = np.asarray(indices)
        if not np.issubdtype(selected.dtype, np.integer):
            raise TypeError("cache batch indices must have an integer dtype")
        selected = selected.astype(np.int64, copy=False)
        if selected.ndim != 1 or not len(selected) or (selected < 0).any() or (selected >= len(self.window_ids)).any():
            raise ValueError("cache batch indices must be a nonempty valid one-dimensional index array")
        return np.asarray(self.history[selected]), np.asarray(self.branch_features[selected])


def _validate_mean_source(queries, b8_cache):
    if not queries or any(not isinstance(window, CommunicationWindow) for window in queries):
        raise TypeError("Bmean cache requires nonempty CommunicationWindow queries")
    if not isinstance(b8_cache, FoldMotionCache) or b8_cache.manifest["format"] != CACHE_FORMAT:
        raise TypeError("Bmean cache requires a validated B8 FoldMotionCache source")
    if b8_cache.manifest["k"] != 8 or b8_cache.manifest["q_seed"] != 0:
        raise GroundingError("Bmean comparison requires the frozen K8/qseed0 B8 source")
    if tuple(window.prediction.window_id for window in queries) != b8_cache.window_ids:
        raise GroundingError("Bmean query identities or order differ from B8")
    if b8_cache.normalizer.manifest() != b8_cache.manifest["normalizer"]:
        raise GroundingError("B8 normalizer object differs from its frozen source manifest")
    prefixes = np.load(b8_cache.cache_dir / "motion_prefixes.npy", mmap_mode="r", allow_pickle=False)
    for index, window in enumerate(queries):
        query = window.prediction
        if (_prediction_manifest(query) != b8_cache.manifest["query_inputs"][index]
                or not np.array_equal(query.motion_history, prefixes[index])
                or not np.array_equal(b8_cache.normalizer.transform_history(query.history), b8_cache.history[index])):
            raise GroundingError("Bmean current prediction prefix differs from its frozen B8 source")


def build_mean_fold_cache(queries: Sequence[CommunicationWindow], b8_cache_dir: Path | str,
                        cache_dir: Path | str) -> ConditionalMeanCache:
    """Derive Bmean inputs in a separate directory using frozen B8 statistics.

    B8 acceptance runs once here. Bmean neither fits q nor refits normalization.
    Its eight paths are identical copies of the exact conditional q mean.
    """
    queries = tuple(queries)
    b8_cache = load_fold_cache(b8_cache_dir, expected_windows=queries)
    _validate_mean_source(queries, b8_cache)
    means = conditional_motion_mean_innovations(b8_cache)
    cache_dir = Path(cache_dir)
    if cache_dir.resolve() == b8_cache.cache_dir.resolve():
        raise GroundingError("Bmean cache directory must differ from its immutable B8 source")
    if cache_dir.exists():
        return load_mean_fold_cache(cache_dir, expected_windows=queries, b8_cache=b8_cache)
    n, k, time_steps, _ = b8_cache.branch_features.shape
    cache_dir.mkdir(parents=True)
    branches = np.lib.format.open_memmap(cache_dir / "branch_features.npy", mode="w+", dtype=np.float64,
                                        shape=(n, time_steps, len(BRANCH_FEATURE_NAMES)))
    states = np.lib.format.open_memmap(cache_dir / "motion_states.npy", mode="w+", dtype=np.float64,
                                      shape=(n, time_steps, 1, 6))
    for index, window in enumerate(queries):
        center, raw_features = conditional_mean_motion_center(window.prediction,
            _query_motion_mean(window.prediction, means), k=1)
        states[index] = center[0]
        branches[index] = b8_cache.normalizer.transform_branches(raw_features)[0]
    states.flush()
    branches.flush()
    manifest = {
        "format": MEAN_CACHE_FORMAT, "completed": True, "variant": "Bmean", "k": k,
        "window_ids": list(b8_cache.window_ids),
        "source_b8_cache_dir": str(b8_cache.cache_dir.resolve()),
        "normalizer": b8_cache.normalizer.manifest(), "normalizer_refitted": False,
        "history_source": "source_b8_cache_dir/history.npy", "history_unchanged": True,
        "center": "exact conditional E_q[state]: recursively position += velocity*dt, then add full eligible donor mean innovation",
        "center_source_refs": "source_b8_cache_dir/manifest.json: query_inputs[row].motion_source_refs[-1]",
        "innovation_source": "source_b8_cache_dir/donor_innovations.npy; donors and components in source manifest",
        "conditional_mean_innovations": {key: value.tolist() for key, value in means.items()},
        "conditional_donor_weights": "uniform over distinct full-motion-prefix/component rows; TRAIN excludes own component; heldout uses all TRAIN",
        "donor_draw_persistence": "one joint innovation sampled per B8 branch, repeated at each future step",
        "branch_feature_names": list(BRANCH_FEATURE_NAMES),
        "branch_features": "f(E_q[physical state]) via existing _branch_features, then frozen B8 TRAIN normalizer",
        "residual_donor_draws_used": False, "conditional_donor_mean_added": True,
        "future_supervision_used": False, "identical_center_copies": k,
        "stored_center_paths": 1, "runtime_branch_layout": "broadcast stored single path to K8",
        "contrast": "same conditional physical-state expectation; Bmean removes sampled innovation dispersion",
        "nonlinear_feature_order": "f(E_q[state]), not E_q[f(state)] and not a K8 sample average",
    }
    _write_json(cache_dir / "manifest.json", manifest)
    del branches, states
    return load_mean_fold_cache(cache_dir, expected_windows=queries, b8_cache=b8_cache)


def load_mean_fold_cache(cache_dir: Path | str, *, expected_windows: Sequence[CommunicationWindow],
                       b8_cache: FoldMotionCache | None = None) -> ConditionalMeanCache:
    """Verify conditional q-mean Bmean rows and frozen B8 normalization reuse.

    A supplied B8 source must already have passed ``load_fold_cache`` with
    current windows. This avoids repeating B8 acceptance when building Bmean.
    """
    cache_dir = Path(cache_dir)
    queries = tuple(expected_windows)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["format"] != MEAN_CACHE_FORMAT or manifest["completed"] is not True or manifest["variant"] != "Bmean":
        raise GroundingError("Bmean cache is unsupported or incomplete")
    source = Path(manifest["source_b8_cache_dir"])
    if b8_cache is None:
        b8_cache = load_fold_cache(source, expected_windows=queries)
    if b8_cache.cache_dir.resolve() != source.resolve():
        raise GroundingError("Bmean source cache path differs from the supplied B8 cache")
    _validate_mean_source(queries, b8_cache)
    means = conditional_motion_mean_innovations(b8_cache)
    if (manifest["normalizer"] != b8_cache.normalizer.manifest() or manifest["normalizer_refitted"] is not False
            or tuple(manifest["window_ids"]) != b8_cache.window_ids or manifest["k"] != 8
            or manifest["conditional_mean_innovations"] != {key: value.tolist() for key, value in means.items()}):
        raise GroundingError("Bmean normalizer, K or query identities differ from frozen B8")
    branches = np.load(cache_dir / "branch_features.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(cache_dir / "motion_states.npy", mmap_mode="r", allow_pickle=False)
    n, k, time_steps, feature_dim = b8_cache.branch_features.shape
    if (manifest["stored_center_paths"] != 1 or branches.shape != (n, time_steps, feature_dim)
            or states.shape != (n, time_steps, 1, 6)
            or branches.dtype != np.dtype(np.float64) or states.dtype != np.dtype(np.float64)
            or not np.isfinite(branches).all() or not np.isfinite(states).all()):
        raise GroundingError("Bmean center arrays have wrong shapes, dtype or nonfinite values")
    for index, window in enumerate(queries):
        center, raw_features = conditional_mean_motion_center(window.prediction,
            _query_motion_mean(window.prediction, means), k=1)
        if (not np.array_equal(center[0], states[index])
                or not np.array_equal(b8_cache.normalizer.transform_branches(raw_features)[0], branches[index])):
            raise GroundingError("Bmean cached states or features differ from conditional q-mean replay")
    runtime_branches = np.broadcast_to(branches[:, None], (n, k, time_steps, feature_dim))
    runtime_states = np.broadcast_to(states[:, None], (n, k, time_steps, 1, 6))
    return ConditionalMeanCache(cache_dir, b8_cache.window_ids, b8_cache.history, runtime_branches, runtime_states,
                                    b8_cache.normalizer, manifest)
