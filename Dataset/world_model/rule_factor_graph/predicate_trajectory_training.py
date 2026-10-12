"""Callable assembly/cache loaders and masked cumulative training operations.

Loading never generates trajectories or touches TEST. Numerical operations read
fixed tensors and precompiled incidence; no provenance or registry scans occur in
forward/loss/training. Calling train_step is an explicit optimization action.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .predicate_trajectory_models import TrajectoryLayout

FORWARD_KEYS = ("history", "sampled_states", "sampled_truth", "history_known", "state_known",
                "truth_known", "applicable", "predicate_field_incidence")
BOOLEAN_KEYS = frozenset(("sampled_truth", "history_known", "state_known", "truth_known",
                          "applicable", "target_known", "compatible", "predicate_field_incidence"))


@dataclass(frozen=True)
class PreparedTrajectorySplit:
    tensors: Mapping[str, torch.Tensor]
    layout: TrajectoryLayout
    split: str
    metadata: object

    @property
    def size(self):
        return self.tensors["history"].shape[0]

    def batch(self, indices):
        """Slice only the batch axis; incidence is fixed setup-time metadata."""
        return {key: value if (key == "predicate_field_incidence" and value.ndim == 2) or key == "field_type_ids" else value[indices]
                for key, value in self.tensors.items()}


def episode_component_layout(episode, *, field_type_keys=None):
    """Compile one episode's encoded-component layout from its declared fields.

    Schema reuse and instance topology stay separate. The field-type vocabulary
    (schema reuse) is shared across episodes; the owner incidence (instance
    topology) compiles from this episode's exact typed owners alone: the
    declared (entity id, ontology classes) identities in each field
    coordinate. The contract declares no epoch, generation or version fields,
    and none are invented here; two typed owners are the same instance exactly
    when their declared identity is equal. Two episodes that declare the same
    local field schemas over different typed owners therefore keep different
    owner incidence, and a layout is never reused from another episode's
    fields or local ids.
    """
    from .predicate_trajectory_data import expanded_field_coordinates
    return TrajectoryLayout.from_fields(expanded_field_coordinates(episode.fields),
                                        field_type_keys=field_type_keys)


def prepare_assembled_batch(batch, *, split, device, targets=None, target_known=None,
                            compatible=None, prior_logits=None, dtype=torch.float64, field_type_keys=None):
    """Convert an assembly batch once, with explicitly supplied supervision.

    Training supervision [B,P,10] is observed future cumulative evidence, distinct
    from sampled fixed-rule truth. compatible [B,P,11] represents event censoring.
    Encoded component fields carry the layout; declared vector components are
    already expanded by the assembly projection, so the training tensors are the
    component tensors exactly. Each episode compiles its own encoded-component
    layout and owner incidence from its declared fields against the shared
    vocabulary (episode_component_layout): identical field types with different
    typed-owner topology keep their distinct per-episode incidence instead of
    inheriting the first episode's topology.
    """
    if split not in ("train", "validation"):
        raise ValueError("only fixed TRAIN and VALID may be prepared here")
    mapping = batch.as_mapping()
    if any(episode.split != split for episode in batch.episodes):
        raise ValueError("assembled episodes disagree with the requested fixed split")
    layouts = tuple(episode_component_layout(episode, field_type_keys=field_type_keys)
                    for episode in batch.episodes)
    layout = layouts[0]
    if any(item.field_type_keys != layout.field_type_keys or item.field_type_ids != layout.field_type_ids for item in layouts):
        raise ValueError("assembled batch contains different field type layouts")
    tensors = {key: torch.as_tensor(np.array(mapping[key], copy=True),
                dtype=torch.bool if key in BOOLEAN_KEYS else dtype, device=device)
               for key in FORWARD_KEYS}
    tensors["field_type_ids"] = torch.tensor(layout.field_type_ids, dtype=torch.long, device=device)
    tensors["owner_field_incidence"] = torch.tensor([item.owner_incidence for item in layouts], dtype=torch.bool, device=device)
    for key, value in (("targets", targets), ("target_known", target_known),
                       ("compatible", compatible), ("prior_logits", prior_logits)):
        if value is not None:
            tensors[key] = torch.as_tensor(np.array(value, copy=True),
                              dtype=torch.bool if key in BOOLEAN_KEYS else dtype, device=device)
    if (targets is None) != (target_known is None):
        raise ValueError("cumulative supervision needs targets and known masks together")
    if targets is not None:
        shape = (tensors["history"].shape[0], tensors["sampled_truth"].shape[-1], 10)
        if tensors["targets"].shape != shape or tensors["target_known"].shape != shape:
            raise ValueError("cumulative targets must align with [B,P,10]")
    return PreparedTrajectorySplit(tensors, layout, split, batch)


def cumulative_loss(log_pmf, targets, known, mode="hybrid"):
    """Hybrid: known-CDF square1..9 + .5 known-BCE10; controls cover1..10."""
    if log_pmf.shape[-1] != 11 or targets.shape != (*log_pmf.shape[:-1], 10) or known.shape != targets.shape or known.dtype != torch.bool:
        raise ValueError("loss requires log PMF[B,P,11], targets/known[B,P,10]")
    log_cdf = torch.logcumsumexp(log_pmf, -1)[..., :10]
    log_survival = torch.logcumsumexp(log_pmf.flip(-1), -1).flip(-1)[..., 1:]
    cdf = (log_cdf - torch.logaddexp(log_cdf, log_survival)).exp()
    square_mask, bce_mask = known.clone(), known.clone()
    if mode == "hybrid":
        square_mask[..., 9] = False
        bce_mask[..., :9] = False
    elif mode == "rps":
        bce_mask[:] = False
    elif mode == "cumulative_bce":
        square_mask[:] = False
    else:
        raise ValueError(f"undeclared cumulative loss: {mode}")
    zero = log_pmf.exp().sum() * 0
    squared = (cdf[square_mask] - targets[square_mask]).square().mean() if bool(square_mask.any()) else zero
    bce = zero
    if bool(bce_mask.any()):
        selected = torch.where(targets[bce_mask].bool(), log_cdf[bce_mask], log_survival[bce_mask])
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("known cumulative target has invalid probability")
        bce = -selected.mean()
    return squared + .5 * bce if mode == "hybrid" else squared + bce


def censored_nll(log_pmf, compatible):
    """Event likelihood sums only observation-compatible first-flip bins."""
    if compatible.dtype != torch.bool or compatible.shape != log_pmf.shape or not bool(compatible.any(-1).all()):
        raise ValueError("censored event evidence needs nonempty Boolean support[B,P,11]")
    return -torch.logsumexp(log_pmf.masked_fill(~compatible, -torch.inf), -1).mean()


def predict_batch(model, batch):
    kwargs = {key: batch[key] for key in FORWARD_KEYS}
    for key in ("prior_logits", "owner_field_incidence", "field_type_ids"):
        if key in batch:
            kwargs[key] = batch[key]
    return model(**kwargs)


def train_step(model, optimizer, batch, *, mode="hybrid"):
    """One requested optimizer update; all loss arms use the same forward/batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_pmf = predict_batch(model, batch)
    loss = cumulative_loss(log_pmf, batch["targets"], batch["target_known"], mode)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite cumulative objective")
    loss.backward()
    optimizer.step()
    return loss.detach()


def _event_support(rows, predicates):
    compatible = np.zeros((len(rows), predicates, 11), dtype=bool)
    for i, row in enumerate(rows):
        if len(row) != predicates:
            raise ValueError("event constraints disagree with predicate axis")
        for p, (kind, step) in enumerate(row):
            if kind == "first_flip" and 1 <= step <= 10:
                compatible[i, p, step - 1] = True
            elif kind == "right_censored" and 0 <= step <= 10:
                compatible[i, p, step:] = True
            elif kind == "none":
                compatible[i, p, 10] = True
            else:
                raise ValueError(f"invalid cached event constraint {kind}, {step}")
    return compatible


def load_lp_numeric_cache(path: str | Path, *, device, dtype=torch.float64,
                          observed_history_prior_adapter=None, field_type_keys=None):
    """Existing L/P NPZ vertical slice through the generic forward contract.

    The cache provides actual numeric sampled states and Boolean fixed-rule
    traces. Its published complete numeric values are explicitly known. Optional
    prior adapter receives only prefix history/raw_history/governed thresholds;
    it returns declared [B,P,11] logits. No default prior is invented.
    """
    from .communication_cv_data import receiver_numeric_fields
    from .communication_data import PREDICATES
    specs = receiver_numeric_fields()
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        ordered = []
        for key in metadata["numeric_keys"]:
            matches = [s for s in specs if f"{s.artifact}#{s.field_path}" == key]
            if len(matches) != 1:
                raise ValueError(f"cache field has no exact contract: {key}")
            ordered.append(matches[0])
        # Owner roles, not per-entity embeddings. These cached receiver fields
        # share the same declared actor/station pair within each window.
        keys = tuple((s.field_id, s.data_type, s.unit, s.artifact, (), s.owner_keys) for s in ordered)
        vocabulary = keys if field_type_keys is None else tuple(field_type_keys)
        if any(key not in vocabulary for key in keys):
            raise ValueError("L/P cache field type is absent from declared global vocabulary")
        layout = TrajectoryLayout(vocabulary, tuple(vocabulary.index(key) for key in keys),
                     tuple(tuple(True for _ in keys) for _ in keys))
        incidence = np.asarray([[pid in spec.predicate_ids for spec in ordered] for pid in PREDICATES], dtype=bool)
        for split in ("train", "validation"):
            rows = metadata["splits"][split]
            if any(q["split"] != split for q in rows["queries"]):
                raise ValueError("cache query belongs to a different fixed split")
            h = np.array(archive[f"{split}_numeric_history"], copy=True)
            states = np.array(archive[f"{split}_numeric_branches"], copy=True)
            truth = np.array(archive[f"{split}_predicate_trajectories"], copy=True)
            if truth.dtype != np.bool_ or not np.isfinite(h).all() or not np.isfinite(states).all():
                raise ValueError("L/P cache lacks complete declared numeric/Boolean inputs")
            mapping = dict(history=h, sampled_states=states, sampled_truth=truth,
                history_known=np.ones(h.shape, dtype=bool), state_known=np.ones(states.shape, dtype=bool),
                truth_known=np.ones(truth.shape, dtype=bool), applicable=np.ones(truth.shape, dtype=bool),
                predicate_field_incidence=incidence, targets=archive[f"{split}_targets"],
                target_known=archive[f"{split}_known"], compatible=_event_support(rows["constraints"], len(PREDICATES)))
            tensors = {key: torch.as_tensor(np.array(value, copy=True),
                         dtype=torch.bool if key in BOOLEAN_KEYS else dtype, device=device)
                       for key, value in mapping.items()}
            tensors["field_type_ids"] = torch.tensor(layout.field_type_ids, dtype=torch.long, device=device)
            tensors["owner_field_incidence"] = torch.tensor([layout.owner_incidence] * len(h), dtype=torch.bool, device=device)
            if observed_history_prior_adapter is not None:
                raw = torch.as_tensor(np.array(archive[f"{split}_raw_history"], copy=True), dtype=dtype, device=device)
                thresholds = torch.as_tensor(np.array(archive[f"{split}_governed_thresholds"], copy=True), dtype=dtype, device=device)
                with torch.no_grad():
                    prior = observed_history_prior_adapter(tensors["history"], raw, thresholds)
                if prior.shape != (len(h), len(PREDICATES), 11):
                    raise ValueError("declared L/P history prior has wrong shape")
                tensors["prior_logits"] = prior.detach()
            result[split] = PreparedTrajectorySplit(tensors, layout, split, rows)
    return result


def train_prepared_split(model, optimizer, prepared: PreparedTrajectorySplit,
                         batch_indices, *, mode="hybrid"):
    """Train an explicit fixed TRAIN schedule; never fit on VALID or TEST.

    The caller declares the sampling schedule, seeds and update budget. This
    entrypoint makes no schedule selection, early stopping or performance tuning.
    """
    if prepared.split != "train":
        raise ValueError("optimizer updates require the fixed TRAIN cohort")
    indices = torch.as_tensor(batch_indices, dtype=torch.long,
                              device=prepared.tensors["history"].device)
    if indices.ndim != 2 or indices.shape[1] == 0 or bool((indices < 0).any()) or bool((indices >= prepared.size).any()):
        raise ValueError("declared training schedule needs valid [updates,batch] indices")
    losses = [train_step(model, optimizer, prepared.batch(row), mode=mode) for row in indices]
    return torch.stack(losses) if losses else prepared.tensors["history"].new_empty((0,))
