"""Fixed nearest-32 prefix conditioning for the empirical residual block bank.

The base sampler owns TRAIN-only fitting, full-copy collapse, whole-component
exclusion, and supported motion/radial pools. This sampler ranks that admissible
pool by 15 prefix-only kinematic features. Fit adds at most 120 bytes per unique
donor for standardized features; each query costs O(P*15 + P*log(P) + K*120),
where P is the supported coarse pool size. Neighbor count is fixed, not tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import STEP_SECONDS, WindowDataset
from .sampling import (EmpiricalResidualSampler, InsufficientDonorsError, MODES, _conditions,
                       _validate_history, _validate_rows, constant_velocity_center)


NEIGHBOR_COUNT = 32
ACCELERATION_HISTORY_OFFSET = 4
FEATURE_NAMES = (
    "first_vx", "first_vy", "first_vz", "second_vx", "second_vy", "second_vz",
    "first_ax", "first_ay", "first_az", "second_ax", "second_ay", "second_az",
    "relative_x_over_H", "relative_y_over_H", "relative_z_over_V",
)


def kinematic_prefix_features(history: np.ndarray, rule_parameters: np.ndarray) -> np.ndarray:
    """Ordered velocities, two-second accelerations, and scaled relative position."""
    history = _validate_history(history)
    parameters = np.asarray(rule_parameters, dtype=np.float64)
    if parameters.shape != (len(history), 4) or not np.isfinite(parameters).all():
        raise ValueError("rule_parameters must contain finite [N,4] cutoff-known parameters")
    if not np.isin(parameters[:, 0], (0, 1)).all() or (parameters[:, 1:] <= 0).any():
        raise ValueError("rule_parameters need a valid metric and positive H/V/margin")
    last = history[:, -1]
    velocity = last[..., 3:].reshape(len(history), 6)
    earlier_velocity = history[:, -1 - ACCELERATION_HISTORY_OFFSET, :, 3:]
    acceleration = ((last[..., 3:] - earlier_velocity)
                    / (ACCELERATION_HISTORY_OFFSET * STEP_SECONDS)).reshape(len(history), 6)
    relative_position = last[:, 1, :3] - last[:, 0, :3]
    limits = parameters[:, [1, 1, 2]]
    features = np.concatenate((velocity, acceleration, relative_position / limits), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("kinematic prefix features must be finite")
    return features


class KinematicResidualSampler(EmpiricalResidualSampler):
    """Draw whole residual blocks uniformly from the nearest admissible donors."""

    def __init__(self, min_support: int = 2) -> None:
        super().__init__(min_support=min_support)
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.active_feature_indices_: np.ndarray | None = None
        self._standardized_donor_features: np.ndarray | None = None

    def fit(self, training: WindowDataset) -> KinematicResidualSampler:
        """Normalize only the unique complete TRAIN representatives of the base bank."""
        self.feature_mean_ = None
        self.feature_scale_ = None
        self.active_feature_indices_ = None
        self._standardized_donor_features = None
        super().fit(training)
        positions = {row_id: index for index, row_id in enumerate(training.rows["row_id"])}
        representatives = np.asarray([positions[donor.row_id] for donor in self._donors],
                                     dtype=np.int64)
        features = kinematic_prefix_features(training.history[representatives],
                                             training.rule_parameters[representatives])
        active = np.flatnonzero(features.max(axis=0) != features.min(axis=0))
        mean = features.mean(axis=0)
        scale = features.std(axis=0, ddof=0)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("TRAIN normalization statistics must be finite")
        if (scale[active] <= 0).any():
            raise ValueError("nonconstant TRAIN feature has zero normalization scale")
        self.feature_mean_ = mean.copy()
        self.feature_scale_ = scale.copy()
        self.active_feature_indices_ = active.copy()
        self._standardized_donor_features = (features[:, active] - mean[active]) / scale[active]
        self.diagnostics.update({
            "sampler": "kinematic_nearest_32",
            "neighbor_count": NEIGHBOR_COUNT,
            "feature_names": list(FEATURE_NAMES),
            "acceleration_interval_seconds": ACCELERATION_HISTORY_OFFSET * STEP_SECONDS,
            "normalization_split": "train",
            "normalization_donors": len(representatives),
            "normalization_donor_row_ids": [donor.row_id for donor in self._donors],
            "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(),
            "active_feature_indices": active.tolist(),
            "constant_feature_indices_removed": np.flatnonzero(
                features.max(axis=0) == features.min(axis=0)).tolist(),
            "standardized_feature_bank_bytes": int(self._standardized_donor_features.nbytes),
            "distance": "Euclidean over active TRAIN-standardized prefix features",
            "distance_tie_break": "donor row_id ascending",
        })
        return self

    def _nearest_pool(self, pool: np.ndarray, query_feature: np.ndarray):
        if (self._standardized_donor_features is None or self.feature_mean_ is None
                or self.feature_scale_ is None or self.active_feature_indices_ is None):
            raise RuntimeError("fit the TRAIN kinematic donor bank before residual sampling")
        active = self.active_feature_indices_
        standardized = (query_feature[active] - self.feature_mean_[active]) / self.feature_scale_[active]
        difference = self._standardized_donor_features[pool] - standardized
        squared_distance = np.sum(difference * difference, axis=1)
        if not np.isfinite(squared_distance).all():
            raise ValueError("query kinematic distances must be finite")
        row_ids = np.asarray([self._donors[int(index)].row_id for index in pool])
        ranking = np.lexsort((row_ids, squared_distance))[:NEIGHBOR_COUNT]
        if len(ranking) < self.min_support:
            raise InsufficientDonorsError(
                f"nearest-{NEIGHBOR_COUNT} pool has {len(ranking)} donors; "
                f"min_support={self.min_support}")
        return pool[ranking], np.sqrt(squared_distance[ranking])

    def query_pool(self, dataset: WindowDataset, index: int) -> tuple[np.ndarray, str, int]:
        """Return the exact admissible nearest pool for independent belief diagnostics."""
        history = _validate_history(dataset.history[index:index + 1])
        rows = dataset.rows.iloc[[index]]
        _validate_rows(rows, len(history), fit=False)
        feature = kinematic_prefix_features(history, dataset.rule_parameters[index:index + 1])[0]
        row = rows.iloc[0]
        coarse, level, excluded = super()._pool(
            _conditions(history)[0], int(row["family_component"]), row["split"], row["row_id"])
        nearest, _ = self._nearest_pool(coarse, feature)
        return nearest, level, excluded

    def sample(self, history: np.ndarray, rows: pd.DataFrame, k: int, seed: int,
               mode: str = "joint", *, rule_parameters: np.ndarray) -> np.ndarray:
        """Return [N,K,10,2,6]; explicit cutoff-known static parameters are required."""
        history = _validate_history(history)
        _validate_rows(rows, len(history), fit=False)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        features = kinematic_prefix_features(history, rule_parameters)
        self.last_sample_metadata = pd.DataFrame()
        if mode == "center":
            samples = super().sample(history, rows, k=k, seed=seed, mode="center")
            self.last_sample_metadata["coarse_pool_count"] = 0
            self.last_sample_metadata["coarse_pool_components"] = [[] for _ in range(len(rows))]
            self.last_sample_metadata["nearest_pool_count"] = 0
            self.last_sample_metadata["nearest_pool_components"] = [[] for _ in range(len(rows))]
            self.last_sample_metadata["nearest_donor_row_ids"] = [[] for _ in range(len(rows))]
            self.last_sample_metadata["nearest_distances"] = [[] for _ in range(len(rows))]
            self.last_sample_metadata["active_feature_indices"] = [
                [] if self.active_feature_indices_ is None else self.active_feature_indices_.tolist()
                for _ in range(len(rows))]
            return samples
        if self._residuals is None or self._standardized_donor_features is None:
            raise RuntimeError("fit the TRAIN kinematic donor bank before residual sampling")

        center = constant_velocity_center(history)
        samples = np.repeat(center[:, None], int(k), axis=1)
        rng = np.random.default_rng(int(seed))
        conditions = _conditions(history)
        metadata = []
        for index, (_, row) in enumerate(rows.iterrows()):
            coarse_pool, level, excluded = super()._pool(
                conditions[index], int(row["family_component"]), row["split"], row["row_id"])
            pool, distances = self._nearest_pool(coarse_pool, features[index])
            if mode == "joint":
                selected = rng.choice(pool, size=int(k), replace=True)
                samples[index] += self._residuals[selected]
            else:
                selected = rng.choice(pool, size=(int(k), 2, 6), replace=True)
                for entity in range(2):
                    for attribute in range(6):
                        samples[index, :, :, entity, attribute] += self._residuals[
                            selected[:, entity, attribute], :, entity, attribute]
            coarse_components = sorted({self._donors[int(i)].family_component for i in coarse_pool})
            nearest_components = sorted({self._donors[int(i)].family_component for i in pool})
            metadata.append({
                "row_id": row["row_id"], "mode": mode, "level": level,
                "query_family_component": int(row["family_component"]),
                "excluded_component_donor_count": excluded,
                "coarse_pool_count": len(coarse_pool),
                "coarse_pool_components": coarse_components,
                "nearest_pool_count": len(pool),
                "nearest_pool_components": nearest_components,
                "support_count": len(pool),
                "support_component_count": len(nearest_components),
                "condition": conditions[index].tolist(),
                "nearest_donor_row_ids": [self._donors[int(i)].row_id for i in pool],
                "nearest_distances": distances.tolist(),
                "active_feature_indices": self.active_feature_indices_.tolist(),
                "donor_row_ids": np.asarray([
                    self._donors[int(donor)].row_id for donor in selected.flat
                ], dtype=object).reshape(selected.shape).tolist(),
            })
        self.last_sample_metadata = pd.DataFrame(metadata)
        return samples

    def sample_dataset(self, dataset: WindowDataset, k: int, seed: int,
                       mode: str = "joint") -> np.ndarray:
        """Read query history, metadata, and cutoff-known rules; never query futures."""
        return self.sample(dataset.history, dataset.rows, k=k, seed=seed, mode=mode,
                           rule_parameters=dataset.rule_parameters)
