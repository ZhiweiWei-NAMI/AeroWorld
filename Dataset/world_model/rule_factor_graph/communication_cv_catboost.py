"""Native GPU censored hazards from common numeric prefix and estimatedQmean.

No joint event projection. Native weighted Logloss learns each marginal hazard.
"""
from __future__ import annotations
from collections import Counter
import time
import numpy as np
import torch
from .contracts import GroundingError
from .events import EventConstraint

CATBOOST_VERSION = "1.2.10"
HEADS = ("L", "P")


def risk_targets(event: EventConstraint):
    """Return observed risk-bin IDs and binary labels; censor zero has no rows."""
    if not isinstance(event, EventConstraint):
        raise TypeError("risk targets require an EventConstraint")
    if event.kind == "first_flip":
        if not 1 <= event.step <= 10:
            raise GroundingError("first-flip risk time must lie in 1..10")
        target = np.zeros(event.step, dtype=np.int64)
        target[-1] = 1
    elif event.kind == "right_censored":
        if not 0 <= event.step <= 10:
            raise GroundingError("observed censor time must lie in 0..10")
        target = np.zeros(event.step, dtype=np.int64)
    else:
        target = np.zeros(10, dtype=np.int64)
    return np.arange(1, len(target) + 1, dtype=np.int64), target



def probability_columns_to_event_logs(probabilities):
    """Convert P(survive),P(first flip|at risk) directly to an 11-way log PMF.

    Both GPU Probability columns are used directly. No subtraction from one
    or epsilon clipping conceals an endpoint. Such outputs require explicit
    diagnosis before a different GPU prediction representation is introduced.
    """
    if not isinstance(probabilities, torch.Tensor) or not probabilities.is_floating_point():
        raise TypeError("hazard probability-law construction requires a floating-point tensor")
    values = probabilities.to(dtype=torch.float64)
    if values.ndim != 3 or values.shape[1:] != (10, 2) or not bool(torch.isfinite(values).all()):
        raise GroundingError("GPU binary hazard probabilities must be finite [B,10,2]")
    if bool(((values <= 0) | (values >= 1)).any()):
        raise FloatingPointError("CatBoost GPU Probability produced an exact 0/1 or invalid endpoint; no clipping or CPU fallback")
    if not torch.allclose(values.sum(-1), torch.ones_like(values[..., 0]), rtol=0.0, atol=1e-8):
        raise GroundingError("GPU binary class probabilities do not sum to one")
    logs = torch.log(values)
    survived = values.new_zeros(len(values))
    event_logs = []
    # Explicit ten-bin prefixes support the runner's deterministic CUDA mode.
    for bin_index in range(10):
        event_logs.append(survived + logs[:, bin_index, 1])
        survived = survived + logs[:, bin_index, 0]
    event_logs.append(survived)
    return torch.stack(event_logs, dim=1)


class CatBoostHazardModel:
    model_name = "CatBoost"

    def __init__(self, heads, metadata):
        self.heads, self.metadata = heads, metadata

    @classmethod
    def fit(cls, features, events, components, *, seed, iterations=2000):
        from catboost import CatBoostClassifier, Pool, __version__
        if __version__ != CATBOOST_VERSION:
            raise RuntimeError(f"CatBoost version mismatch: {__version__}")
        informative = [i for i, row in enumerate(events) if any(e.kind != "right_censored" or e.step > 0 for e in row.values())]
        counts = Counter(components[i] for i in informative)
        groups = len(counts)
        config = dict(loss_function="Logloss", task_type="GPU", devices="0", depth=6,
                      iterations=iterations, learning_rate=.03, l2_leaf_reg=3.0,
                      random_seed=seed, thread_count=2, gpu_ram_part=.35,
                      allow_writing_files=False, verbose=False)
        heads, diagnostics = {}, {}
        for name in HEADS:
            indices, bins, labels, weights = [], [], [], []
            for i in informative:
                risk, targets = risk_targets(events[i][name])
                indices.extend([i]*len(risk)); bins.extend(risk); labels.extend(targets)
                weights.extend([1/(groups*counts[components[i]])]*len(risk))
            x = np.concatenate((features[np.asarray(indices)], np.asarray(bins)[:, None]), -1).astype(np.float32)
            labels, weights = np.asarray(labels), np.asarray(weights)
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError(f"CatBoost {name} requires both hazard classes")
            weights *= len(weights)/weights.sum()
            model = CatBoostClassifier(**config)
            started = time.perf_counter()
            model.fit(Pool(x, labels, weight=weights))
            losses = model.get_evals_result()["learn"]["Logloss"]
            diagnostics[name] = dict(risk_rows=len(labels), positives=int(labels.sum()),
                                     trees=int(model.tree_count_), seconds=time.perf_counter()-started,
                                     loss_first=losses[0], loss_last=losses[-1],
                                     actual_native_params=model.get_all_params())
            heads[name] = model
            print(f"CatBoost {name} seed={seed} completed trees={model.tree_count_}", flush=True)
        return cls(heads, dict(configuration=config, feature_count=features.shape[1]+1,
                               heads=diagnostics, version=__version__,
                               objective="component-balanced window weighted censored risk terms SUM"))

    def predict_log_pmf(self, features, *, device):
        rows = np.concatenate((np.repeat(features, 10, axis=0),
                               np.tile(np.arange(1, 11), len(features))[:, None]), -1).astype(np.float32)
        output = []
        for name in HEADS:
            probability = self.heads[name].predict(rows, prediction_type="Probability", task_type="GPU", thread_count=2)
            values = torch.as_tensor(np.asarray(probability).reshape(len(features), 10, 2), dtype=torch.float64, device=device)
            output.append(probability_columns_to_event_logs(values))
        return torch.stack(output, 1)
