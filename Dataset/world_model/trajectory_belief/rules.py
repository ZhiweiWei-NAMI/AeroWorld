"""Exact aircraft-pair geometry and first-opposite observation-grid projection."""
from __future__ import annotations

import numpy as np

from .contracts import FUTURE_STEPS


def evaluate_pair(trajectory, rule_parameters):
    """Return strict separation violations for [..., 2, 6] physical states.

    Parameters [N,4] broadcast over the trajectory's middle sample/time axes.
    The producer's anisotropic equivalent distance is
    max(horizontal/H, vertical/V) * margin, followed by distance < margin.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    parameters = np.asarray(rule_parameters, dtype=np.float64)
    if trajectory.shape[-2:] != (2, 6) or parameters.shape[-1:] != (4,):
        raise ValueError("rule requires trajectory [...,2,6] and parameters [...,4]")
    if not np.isfinite(trajectory).all() or not np.isfinite(parameters).all():
        raise ValueError("rule evaluation requires complete finite states and parameters")
    if not np.isin(parameters[..., 0], [0, 1]).all() or (parameters[..., 1:] <= 0).any():
        raise ValueError("invalid separation metric or limits")
    delta = trajectory[..., 0, :3] - trajectory[..., 1, :3]
    euclidean = np.linalg.norm(delta, axis=-1)
    horizontal = np.linalg.norm(delta[..., :2], axis=-1)
    vertical = np.abs(delta[..., 2])
    expanded = parameters.reshape(parameters.shape[:-1] + (1,) * (euclidean.ndim - parameters.ndim + 1) + (4,))
    equivalent = np.maximum(horizontal / expanded[..., 1], vertical / expanded[..., 2]) * expanded[..., 3]
    distance = np.where(expanded[..., 0] == 1, equivalent, euclidean)
    return distance < expanded[..., 3]


def first_opposite_distribution(current, future_binary):
    """Average complete sampled trajectories into 10 right-endpoint bins + none."""
    future = np.asarray(future_binary)
    current = np.asarray(current)
    if future.ndim != 3 or future.shape[2] != FUTURE_STEPS or future.shape[1] == 0:
        raise ValueError("future_binary must have shape [N,K,10] with K>0")
    if current.shape != (future.shape[0],) or not np.isin(current, [0, 1]).all():
        raise ValueError("current must be binary [N]")
    if not np.isin(future, [0, 1]).all():
        raise ValueError("future truth must be binary")
    opposite = future.astype(bool) != current[:, None, None].astype(bool)
    first = np.where(opposite.any(axis=-1), opposite.argmax(axis=-1), FUTURE_STEPS)
    return np.eye(FUTURE_STEPS + 1, dtype=np.float64)[first].mean(axis=1)
