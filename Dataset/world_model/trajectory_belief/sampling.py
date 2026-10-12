"""Train-only empirical residual trajectories for the fixed UAV pair contract.

Fit and query conditioning use complete prefixes. A joint draw selects one
observed ``[10, 2, 6]`` future residual block, preserving the donor's time,
entity, and attribute dependence. No learned noise or future labels enter the
sampler. Fit costs O(N log N + N * (21 + 10) * 2 * 6); residual storage is
960 * N bytes in float64. Sampling writes O(Q * K * 10 * 2 * 6) values and
scans at most N donors per query to exclude its family component and choose a
supported condition group.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults
from .contracts import FUTURE_STEPS, HISTORY_STEPS, STEP_SECONDS, WindowDataset


HISTORY_SHAPE = (HISTORY_STEPS, 2, 6)
FUTURE_SHAPE = (FUTURE_STEPS, 2, 6)
CONDITION_EPSILON = 1e-6
STATIONARY_SPEED_MPS = float(
    get_governed_parameter_defaults()["aircraft_stationary_speed_threshold_mps"]
)
MODES = ("joint", "independent", "center")


class InsufficientDonorsError(ValueError):
    """A query cannot use the required number of distinct training donors."""


@dataclass(frozen=True)
class _Donor:
    row_id: str
    family_component: int
    row_ids: tuple[str, ...]


def constant_velocity_center(history: np.ndarray) -> np.ndarray:
    """Project last observed position and persist last observed velocity."""
    history = _validate_history(history)
    offsets = np.arange(1, FUTURE_SHAPE[0] + 1, dtype=np.float64) * STEP_SECONDS
    center = np.broadcast_to(history[:, -1, None, :, :],
                             (len(history),) + FUTURE_SHAPE).copy()
    center[..., :3] += history[:, -1, None, :, 3:] * offsets[None, :, None, None]
    return center


def _validate_history(history: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 4 or history.shape[1:] != HISTORY_SHAPE:
        raise ValueError(f"history must have shape [N,{HISTORY_SHAPE}], got {history.shape}")
    if not np.isfinite(history).all():
        raise ValueError("history must contain complete finite prefix states")
    return history


def _validate_rows(rows: pd.DataFrame, n: int, *, fit: bool) -> None:
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")
    if len(rows) != n:
        raise ValueError("rows and history lengths differ")
    required = {"row_id", "family", "family_component", "split"}
    if fit:
        required.update(("state_complete", "full_copy_group", "full_copy_representative",
                         "current"))
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"missing sampler metadata columns: {sorted(missing)}")
    for field in ("row_id", "family", "split"):
        if rows[field].isna().any():
            raise ValueError(f"{field} must be present for every row")
        if not rows[field].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"{field} must contain nonempty strings")
    if not rows["row_id"].is_unique:
        raise ValueError("query and donor row_id values must be unique")
    if not rows["split"].isin(("train", "validation", "test")).all():
        raise ValueError("split must be train, validation, or test")
    if fit and not rows["split"].eq("train").all():
        raise ValueError("sampler fit accepts TRAIN rows only")
    for field in ("family_component",) + (("full_copy_group",) if fit else ()):
        if not pd.api.types.is_integer_dtype(rows[field].dtype) or rows[field].isna().any():
            raise TypeError(f"{field} must contain integer group identifiers")
        if (rows[field] < 0).any():
            raise ValueError(f"{field} must contain nonnegative group identifiers")
    if fit and not rows["current"].isin((0, 1)).all():
        raise ValueError("current must contain observed binary prefix states")


def _conditions(history: np.ndarray) -> np.ndarray:
    """Per-role motion and pair radial motion, from the last prefix state."""
    last = history[:, -1]
    velocity = last[..., 3:]
    moving = np.linalg.norm(velocity, axis=-1) > STATIONARY_SPEED_MPS
    relative_position = last[:, 1, :3] - last[:, 0, :3]
    relative_velocity = velocity[:, 1] - velocity[:, 0]
    distance = np.linalg.norm(relative_position, axis=-1)
    radial = np.zeros(len(history), dtype=np.float64)
    separated = distance > CONDITION_EPSILON
    radial[separated] = np.sum(relative_position[separated] * relative_velocity[separated],
                               axis=-1) / distance[separated]
    radial_state = np.where(radial < -CONDITION_EPSILON, -1,
                            np.where(radial > CONDITION_EPSILON, 1, 0))
    return np.column_stack((moving.astype(np.int8), radial_state)).astype(np.int8)


class EmpiricalResidualSampler:
    """Conditioned empirical block bank with declared sparse-group backoff.

    The first version supports the fixed ordered pair (UAV, UAV). Conditions
    are ``(first moving, second moving, radial motion)``. Deterministic support
    backoff drops radial motion, then drops motion, retaining this pair type.
    ``min_support`` counts unique complete trajectory records after removing
    the training query's entire family component and declared full copies.
    Draws are with replacement; K does not require K different training records.
    """

    def __init__(self, min_support: int = 2) -> None:
        if isinstance(min_support, bool) or not isinstance(min_support, (int, np.integer)):
            raise TypeError("min_support must be an integer")
        if min_support < 1:
            raise ValueError("min_support must be at least 1")
        self.min_support = int(min_support)
        self._residuals: np.ndarray | None = None
        self._conditions: np.ndarray | None = None
        self._donors: tuple[_Donor, ...] = ()
        self.last_sample_metadata = pd.DataFrame()
        self.diagnostics: dict[str, object] = {}

    def fit(self, training: WindowDataset) -> EmpiricalResidualSampler:
        """Fit q(S|H) using only complete TRAIN future windows."""
        history = _validate_history(training.history)
        rows = training.rows.reset_index(drop=True)
        _validate_rows(rows, len(history), fit=True)
        future = np.asarray(training.future, dtype=np.float64)
        present = np.asarray(training.future_present)
        if future.shape != (len(history),) + FUTURE_SHAPE:
            raise ValueError(f"future must have shape [N,{FUTURE_SHAPE}]")
        if present.shape != (len(history), FUTURE_SHAPE[0], FUTURE_SHAPE[1]):
            raise ValueError("future_present must have shape [N,10,2]")
        if present.dtype != np.bool_:
            raise TypeError("future_present must contain booleans")
        if rows["state_complete"].dtype != np.bool_:
            raise TypeError("state_complete must contain booleans")
        if rows["full_copy_representative"].dtype != np.bool_:
            raise TypeError("full_copy_representative must contain booleans")
        complete = present.all(axis=(1, 2))
        if not np.array_equal(complete, rows["state_complete"].to_numpy()):
            raise ValueError("state_complete disagrees with future_present")
        if not complete.any():
            raise InsufficientDonorsError("TRAIN has no complete future windows")
        if not np.isfinite(future[complete]).all():
            raise ValueError("complete TRAIN future windows must contain finite states")

        parameters = np.asarray(training.rule_parameters, dtype=np.float64)
        if parameters.shape != (len(history), 4) or not np.isfinite(parameters).all():
            raise ValueError("rule_parameters must contain finite [N,4] static parameters")
        # The data contract assigns shared-prefix family components before
        # splitting. Its declared full-copy groups retain exactly one donor.
        # Donor order depends on stable row IDs, not the caller's index.
        eligible = np.flatnonzero(complete)
        eligible = eligible[np.argsort(rows.iloc[eligible]["row_id"].to_numpy(), kind="stable")]
        donors = []
        representatives = []
        complete_rows = rows.iloc[eligible]
        for _, source_rows in complete_rows.groupby("full_copy_group", sort=False):
            chosen = source_rows.index[source_rows["full_copy_representative"]]
            if len(chosen) != 1:
                raise ValueError("each complete full_copy_group needs exactly one representative")
            if source_rows["family_component"].nunique() != 1:
                raise ValueError("full_copy_group crosses independent family components")
            representative = int(chosen[0])
            for source in source_rows.index:
                source = int(source)
                direct = (np.array_equal(history[source], history[representative])
                          and np.array_equal(future[source], future[representative])
                          and np.array_equal(present[source], present[representative]))
                reversed_pair = (
                    np.array_equal(history[source, :, ::-1], history[representative])
                    and np.array_equal(future[source, :, ::-1], future[representative])
                    and np.array_equal(present[source, :, ::-1], present[representative]))
                same_static = (
                    np.array_equal(parameters[source], parameters[representative])
                    and rows.iloc[source]["current"] == rows.iloc[representative]["current"])
                if not ((direct or reversed_pair) and same_static):
                    raise ValueError("full_copy_group contains different trajectory records")
            representatives.append(representative)
            donors.append(_Donor(rows.iloc[representative]["row_id"],
                                 int(source_rows["family_component"].iloc[0]),
                                 tuple(source_rows["row_id"].tolist())))
        ordering = np.argsort([donor.row_id for donor in donors], kind="stable")
        representatives = np.asarray(representatives, dtype=np.int64)[ordering]
        donors = [donors[index] for index in ordering]
        donor_history = history[representatives]
        self._residuals = future[representatives] - constant_velocity_center(donor_history)
        self._conditions = _conditions(donor_history)
        self._donors = tuple(donors)
        self.last_sample_metadata = pd.DataFrame()
        self.diagnostics = {
            "pair_type": ["UAV", "UAV"],
            "fit_split": "train",
            "training_rows": len(rows),
            "complete_rows": int(complete.sum()),
            "incomplete_rows_excluded": int((~complete).sum()),
            "unique_donors": len(donors),
            "full_copy_duplicates_removed": len(eligible) - len(donors),
            "duplicate_definition": "float64 history/future/present/static/current; pair reversal",
            "family_components": int(rows["family_component"].nunique()),
            "min_support": self.min_support,
            "stationary_speed_mps": STATIONARY_SPEED_MPS,
            "condition_epsilon": CONDITION_EPSILON,
            "condition_levels": ["motion_and_radial", "motion", "type"],
            "residual_bank_bytes": int(self._residuals.nbytes),
        }
        return self

    def _pool(self, condition: np.ndarray, family_component: int, split: str,
              row_id: str) -> tuple[np.ndarray, str, int]:
        if self._residuals is None or self._conditions is None:
            raise RuntimeError("fit the TRAIN donor bank before sampling residuals")
        allowed = np.ones(len(self._donors), dtype=bool)
        if split == "train":
            allowed = np.array([family_component != donor.family_component
                                for donor in self._donors])
        excluded = int((~allowed).sum())
        conditions = self._conditions
        levels = (
            ("motion_and_radial", (conditions == condition).all(axis=1)),
            ("motion", (conditions[:, :2] == condition[:2]).all(axis=1)),
            ("type", np.ones(len(self._donors), dtype=bool)),
        )
        for level, matches in levels:
            pool = np.flatnonzero(allowed & matches)
            if len(pool) >= self.min_support:
                return pool, level, excluded
        raise InsufficientDonorsError(
            f"row {row_id!r} has {int(allowed.sum())} admissible unique TRAIN donors "
            f"after family-component exclusion; min_support={self.min_support}"
        )

    def sample(self, history: np.ndarray, rows: pd.DataFrame, k: int, seed: int,
               mode: str = "joint") -> np.ndarray:
        """Return [N,K,10,2,6]; query futures and labels are never read."""
        history = _validate_history(history)
        _validate_rows(rows, len(history), fit=False)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.last_sample_metadata = pd.DataFrame()
        center = constant_velocity_center(history)
        samples = np.repeat(center[:, None], int(k), axis=1)
        if mode == "center":
            self.last_sample_metadata = pd.DataFrame({
                "row_id": rows["row_id"].to_numpy(), "mode": mode,
                "level": "center", "support_count": 0,
                "excluded_component_donor_count": 0,
                "query_family_component": rows["family_component"].to_numpy(),
                "support_component_count": 0,
                "condition": _conditions(history).tolist(),
                "donor_row_ids": [[] for _ in range(len(history))],
            })
            return samples
        if self._residuals is None:
            raise RuntimeError("fit the TRAIN donor bank before sampling residuals")

        rng = np.random.default_rng(int(seed))
        query_conditions = _conditions(history)
        metadata = []
        for index, (_, row) in enumerate(rows.iterrows()):
            pool, level, excluded = self._pool(query_conditions[index], row["family_component"],
                                                row["split"], row["row_id"])
            if mode == "joint":
                selected = rng.choice(pool, size=int(k), replace=True)
                samples[index] += self._residuals[selected]
            else:
                # A draw is independent across the 12 entity/attribute slots,
                # while all ten time steps of a slot come from one donor.
                selected = rng.choice(pool, size=(int(k), 2, 6), replace=True)
                for entity in range(2):
                    for attribute in range(6):
                        samples[index, :, :, entity, attribute] += self._residuals[
                            selected[:, entity, attribute], :, entity, attribute]
            metadata.append({
                "row_id": row["row_id"], "mode": mode, "level": level,
                "support_count": len(pool), "excluded_component_donor_count": excluded,
                "query_family_component": int(row["family_component"]),
                "support_component_count": len({self._donors[int(i)].family_component
                                                for i in pool}),
                "condition": query_conditions[index].tolist(),
                "donor_row_ids": np.asarray([
                    self._donors[int(donor)].row_id for donor in selected.flat
                ], dtype=object).reshape(selected.shape).tolist(),
            })
        self.last_sample_metadata = pd.DataFrame(metadata)
        return samples

    def sample_dataset(self, dataset: WindowDataset, k: int, seed: int,
                       mode: str = "joint") -> np.ndarray:
        """Sample a dataset prefix without accessing its future or labels."""
        return self.sample(dataset.history, dataset.rows, k=k, seed=seed, mode=mode)
