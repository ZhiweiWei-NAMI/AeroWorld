"""Origin-visible risk sets and interval-censored semantic change targets."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SemanticEventTargets:
    at_risk: np.ndarray
    observed: np.ndarray
    event_type: np.ndarray
    lower_bin: np.ndarray
    upper_bin: np.ndarray
    observed_bins: np.ndarray


def semantic_change_targets(history: np.ndarray, future: np.ndarray) -> SemanticEventTargets:
    """First false-to-true or true-to-false change, excluding availability edits.

    Truth codes are false=0,true=1,unknown=2,out_of_scope=3,inactive=4.
    Loss of known scope right-censors immediately; a later known row never
    bridges an unknown interval into a fabricated exact-frame transition.
    Event timing is the capture interval ending at the changed future bin.
    """
    if history.ndim != 2 or future.ndim != 2 or history.shape[1] != future.shape[1]:
        raise ValueError("semantic targets require [time, binding] matrices")
    if history.shape[0] < 1 or future.shape[0] < 1:
        raise ValueError("semantic target history and future cannot be empty")
    if not np.isin(history, [0, 1, 2, 3, 4]).all() or not np.isin(future, [0, 1, 2, 3, 4]).all():
        raise ValueError("invalid semantic truth code")
    bindings = history.shape[1]
    at_risk = np.isin(history[-1], [0, 1])
    observed = np.zeros(bindings, dtype=bool)
    event_type = np.full(bindings, -1, dtype=np.int64)
    lower = np.full(bindings, -1, dtype=np.int64)
    upper = np.full(bindings, -1, dtype=np.int64)
    observed_bins = np.zeros(bindings, dtype=np.int64)
    for binding in np.flatnonzero(at_risk):
        initial = history[-1, binding]
        for step, value in enumerate(future[:, binding]):
            if value not in (0, 1):
                break
            observed_bins[binding] += 1
            upper[binding] = step
            if value != initial:
                observed[binding] = True
                event_type[binding] = 0 if value == 1 else 1
                lower[binding] = step
                break
    return SemanticEventTargets(at_risk, observed, event_type, lower, upper, observed_bins)
