"""Exact mixed A/Q-chain inference and L/P supervision.

The six states are (A,Q) = 00, 01, 02, 10, 11, 12. Their L/P
projections repeat across A; projected predicates are not separate states.
All partitions accumulate in float64 and preserve gradients to input scores.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .events import EventConstraint
from .exact_inference import ZeroSupportError, _safe_logsumexp, infer_chain
from .mixed_graph import joint_assignments

PREDICATE_NAMES = ("L", "P")
MAX_BATCH_SIZE = 64
MAX_BRANCHES = 16
MAX_TIME_STEPS = 11
JOINT_STATES = 6


@dataclass(frozen=True)
class MixedChainResult:
    branch_logZ: torch.Tensor
    branch_node_marginals: torch.Tensor
    branch_edge_marginals: torch.Tensor
    joint_state_marginals: torch.Tensor
    predicate_marginals: torch.Tensor
    predicate_joint_marginals: torch.Tensor
    availability_marginals: torch.Tensor
    quality_marginals: torch.Tensor
    first_flip_pmf: torch.Tensor | None
    joint_first_flip_pmf: torch.Tensor | None


@dataclass(frozen=True)
class MixedObservationLoss:
    event_nll: torch.Tensor
    joint_nll: torch.Tensor

    @property
    def availability_given_event_nll(self) -> torch.Tensor:
        return self.joint_nll - self.event_nll

    def objective(self, auxiliary_weight: float = 1.0) -> torch.Tensor:
        if not isinstance(auxiliary_weight, (int, float)) or not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be finite and nonnegative")
        if auxiliary_weight == 1:
            return self.joint_nll
        if auxiliary_weight == 0:
            return self.event_nll
        return self.event_nll + auxiliary_weight * self.availability_given_event_nll


@dataclass(frozen=True)
class MixedEvidenceUpdate:
    prior: MixedChainResult
    posterior: MixedChainResult
    evidence_log_probability: torch.Tensor
    diagnostic_only: bool = True


def _validate_batch(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
) -> tuple[int, int, int]:
    if not all(isinstance(value, torch.Tensor) for value in (unary, transitions, support)):
        raise TypeError("mixed chain inputs must be torch tensors")
    if unary.ndim != 4 or unary.shape[-1] != JOINT_STATES:
        raise ValueError("mixed unary must have shape [B, K, T, 6]")
    batch, branches, time_steps, _ = unary.shape
    if not 1 <= batch <= MAX_BATCH_SIZE or not 1 <= branches <= MAX_BRANCHES:
        raise ValueError(f"mixed budgets require B<= {MAX_BATCH_SIZE} and K<= {MAX_BRANCHES}, both positive")
    if not 1 <= time_steps <= MAX_TIME_STEPS:
        raise ValueError(f"mixed time budget is 1..{MAX_TIME_STEPS}")
    if transitions.shape != (batch, branches, time_steps - 1, JOINT_STATES, JOINT_STATES):
        raise ValueError("mixed transitions must have shape [B, K, T-1, 6, 6]")
    if support.shape != unary.shape or support.dtype != torch.bool:
        raise ValueError("mixed support must be Boolean with shape [B, K, T, 6]")
    if not unary.is_floating_point() or not transitions.is_floating_point() or unary.dtype != transitions.dtype:
        raise TypeError("mixed scores must share a floating point dtype")
    if unary.device != transitions.device or unary.device != support.device:
        raise ValueError("mixed inputs must share a device")
    if bool((torch.isnan(unary) | torch.isposinf(unary)).any()) or bool((torch.isnan(transitions) | torch.isposinf(transitions)).any()):
        raise ValueError("mixed scores contain NaN or positive infinity")
    return batch, branches, time_steps


def _as_single_batch(unary, transitions, support):
    if not all(isinstance(value, torch.Tensor) for value in (unary, transitions, support)):
        raise TypeError("mixed chain inputs must be torch tensors")
    if unary.ndim != 3:
        raise ValueError("mixed unary must have shape [K, T, 6]")
    batch_inputs = (unary[None], transitions[None], support[None])
    _validate_batch(*batch_inputs)
    return batch_inputs


def _center_batch_scores(unary, transitions, support):
    """Vectorize the phase-one fixed-precision and complete-path contract."""
    unary = unary.to(dtype=torch.float64)
    transitions = transitions.to(dtype=torch.float64)
    active = support & torch.isfinite(unary)
    finite_edges = torch.isfinite(transitions)
    time_steps = unary.shape[2]
    forward = [active[:, :, 0]]
    for time in range(1, time_steps):
        forward.append(active[:, :, time] & (forward[-1][..., :, None] & finite_edges[:, :, time - 1]).any(dim=-2))
    backward = [torch.zeros_like(active[:, :, 0]) for _ in range(time_steps)]
    backward[-1] = active[:, :, -1]
    for time in range(time_steps - 2, -1, -1):
        backward[time] = active[:, :, time] & (finite_edges[:, :, time] & backward[time + 1][..., None, :]).any(dim=-1)
    viable = torch.stack(forward, dim=2) & torch.stack(backward, dim=2)
    masked_unary = torch.where(viable, unary, torch.full_like(unary, -torch.inf))
    maxima = masked_unary.amax(dim=-1)
    unary_offsets = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima)).detach()
    centered_unary = torch.where(torch.isfinite(masked_unary), masked_unary - unary_offsets[..., None], masked_unary)
    viable_edges = viable[:, :, :-1, :, None] & viable[:, :, 1:, None, :] & finite_edges
    masked_transitions = torch.where(viable_edges, transitions, torch.full_like(transitions, -torch.inf))
    maxima = masked_transitions.amax(dim=(-1, -2))
    edge_offsets = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima)).detach()
    centered_transitions = torch.where(
        torch.isfinite(masked_transitions), masked_transitions - edge_offsets[..., None, None], masked_transitions,
    )
    return centered_unary, centered_transitions, unary_offsets.sum(dim=-1) + edge_offsets.sum(dim=-1)


def _batch_log_partition(unary, transitions, support):
    masked = torch.where(support, unary, torch.full_like(unary, -torch.inf))
    alpha = masked[:, :, 0]
    for time in range(1, unary.shape[2]):
        alpha = masked[:, :, time] + _safe_logsumexp(alpha[..., :, None] + transitions[:, :, time - 1], dim=-2)
    log_partition = _safe_logsumexp(alpha, dim=-1)
    if unary.shape[2] == 1:
        log_partition = log_partition + transitions.sum(dim=(-1, -2, -3))
    return log_partition


def mixed_batch_log_partition(unary, transitions, support) -> torch.Tensor:
    """Return [B,K] full log partitions; every prior branch must be supported."""
    _validate_batch(unary, transitions, support)
    centered_unary, centered_transitions, offset = _center_batch_scores(unary, transitions, support)
    partition = _batch_log_partition(centered_unary, centered_transitions, support)
    if bool(torch.isneginf(partition).any()):
        raise ZeroSupportError("mixed prior contains a zero-mass branch")
    restored = partition + offset
    if not bool(torch.isfinite(restored).all()):
        raise ValueError("mixed full partition overflowed")
    return restored


def _validate_constraints(constraints, batch):
    if not isinstance(constraints, Sequence) or len(constraints) != batch:
        raise ValueError("event_constraints must contain one mapping per batch item")
    for row in constraints:
        if not isinstance(row, Mapping):
            raise TypeError("each mixed event constraint must be a mapping")
        for predicate, event in row.items():
            if predicate not in PREDICATE_NAMES:
                raise ValueError("mixed event constraints use only L and P")
            if not isinstance(event, EventConstraint):
                raise TypeError("mixed event observations require EventConstraint")


def _resolve_initial_values(support, initial_values, constraints):
    batch = support.shape[0]
    projection = joint_assignments(device=support.device)[:, 2:].bool()
    if initial_values is None:
        anchor = support[:, :, 0]
        zero = (anchor[..., None] & ~projection[None, None]).any(dim=(1, 2))
        one = (anchor[..., None] & projection[None, None]).any(dim=(1, 2))
        if bool((~(zero | one)).any()):
            raise ZeroSupportError("mixed chain has zero initial support")
        values = torch.where(zero & one, -1, one.long())
    else:
        if not isinstance(initial_values, torch.Tensor) or initial_values.shape != (batch, 2):
            raise ValueError("initial_values must be an integer or Boolean [B,2] tensor")
        if initial_values.is_floating_point() or initial_values.is_complex() or initial_values.device != support.device:
            raise TypeError("initial_values must be discrete and on the chain device")
        if not bool(((initial_values == -1) | (initial_values == 0) | (initial_values == 1)).all()):
            raise ValueError("initial truth uses 0/1, with -1 explicitly unknown")
        values = initial_values.long()
    for row, observations in enumerate(constraints):
        for predicate in observations:
            column = PREDICATE_NAMES.index(predicate)
            if bool(values[row, column] < 0):
                raise ValueError(f"unknown initial truth for queried predicate {predicate}")
    return values


def _condition_anchor(support, values):
    projection = joint_assignments(device=support.device)[:, 2:]
    anchor_mask = ((values[:, None, :] < 0) | (projection[None] == values[:, None, :])).all(dim=-1)
    return torch.cat((support[:, :, :1] & anchor_mask[:, None, None, :], support[:, :, 1:]), dim=2)


def _prepare_batch(unary, transitions, support, constraints, initial_values):
    batch, _, _ = _validate_batch(unary, transitions, support)
    _validate_constraints(constraints, batch)
    values = _resolve_initial_values(support, initial_values, constraints)
    conditioned = _condition_anchor(support, values)
    centered_unary, centered_transitions, _ = _center_batch_scores(unary, transitions, conditioned)
    denominator = _batch_log_partition(centered_unary, centered_transitions, conditioned)
    if bool(torch.isneginf(denominator).any()):
        raise ZeroSupportError("mixed prior conditioning contains a zero-mass branch")
    return centered_unary, centered_transitions, conditioned, values, denominator


def _event_mask(support, values, constraints):
    mask = torch.ones((support.shape[0], support.shape[2], JOINT_STATES), device=support.device, dtype=torch.bool)
    projection = joint_assignments(device=support.device)[:, 2:]
    horizon = support.shape[2] - 1
    for row, observations in enumerate(constraints):
        for predicate, event in observations.items():
            column = PREDICATE_NAMES.index(predicate)
            same = projection[:, column] == values[row, column]
            if event.kind == "first_flip":
                if not 1 <= event.step <= horizon:
                    raise ValueError("mixed first_flip step must lie in 1..T-1")
                mask[row, :event.step] &= same[None]
                mask[row, event.step] &= ~same
            elif event.kind == "right_censored":
                if not 0 <= event.step <= horizon:
                    raise ValueError("mixed right_censored step must lie in 0..T-1")
                mask[row, :event.step + 1] &= same[None]
            else:
                mask[row] &= same[None]
    return support & mask[:, None]


def _branch_log_probability(unary, transitions, observed_support, denominator):
    numerator = _batch_log_partition(unary, transitions, observed_support)
    return torch.where(torch.isfinite(numerator), numerator - denominator, numerator)


def _uniform_mixture_log_probability(branch_logs):
    return _safe_logsumexp(branch_logs, dim=-1) - math.log(branch_logs.shape[-1])


def _single_initial_values(initial_values, device):
    if initial_values is None:
        return None
    if isinstance(initial_values, torch.Tensor):
        if initial_values.shape != (2,):
            raise ValueError("single-window initial truth must have shape [2]")
        return initial_values[None]
    if not isinstance(initial_values, Sequence) or len(initial_values) != 2:
        raise ValueError("single-window initial truth must have two L/P values")
    if any(value is not None and (not isinstance(value, (bool, int)) or value not in (-1, 0, 1))
           for value in initial_values):
        raise ValueError("initial truth must be Boolean, 0/1, or explicitly unknown")
    return torch.tensor([[(-1 if value is None else int(value)) for value in initial_values]], device=device)


def mixed_branch_event_log_probabilities(
    unary, transitions, support, constraints: Mapping[str, EventConstraint],
    *, initial_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [K] branch-normalized event likelihoods, without motion updates."""
    batch_inputs = _as_single_batch(unary, transitions, support)
    values = _single_initial_values(initial_values, support.device)
    centered_unary, centered_transitions, conditioned, values, denominator = _prepare_batch(
        *batch_inputs, [constraints], values,
    )
    observed = _event_mask(conditioned, values, [constraints])
    return _branch_log_probability(centered_unary, centered_transitions, observed, denominator)[0]


def mixed_event_log_probability(*args, **kwargs) -> torch.Tensor:
    return _uniform_mixture_log_probability(mixed_branch_event_log_probabilities(*args, **kwargs))


def _discrete_evidence(values, known, shape, device, field):
    if values is None and known is None:
        return None
    if not isinstance(values, torch.Tensor) or not isinstance(known, torch.Tensor):
        raise TypeError(f"{field} values and known mask must both be tensors")
    if values.shape != shape or known.shape != shape or known.dtype != torch.bool:
        raise ValueError(f"{field} values/mask must have shape {shape}, mask Boolean")
    if values.device != device or known.device != device or values.is_floating_point() or values.is_complex():
        raise TypeError(f"{field} evidence must be discrete on the chain device")
    if not bool(((values == -1) | (values == 0) | (values == 1)).all()):
        raise ValueError(f"{field} values must be 0/1 or explicitly unknown (-1)")
    if bool((known & (values < 0)).any()):
        raise ValueError(f"known {field} evidence cannot have an unknown value")
    return values.long(), known


def _availability_support(support, availability_values, availability_known):
    evidence = _discrete_evidence(
        availability_values, availability_known, (support.shape[0], support.shape[2]),
        support.device, "availability",
    )
    if evidence is None:
        return support
    values, known = evidence
    availability = joint_assignments(device=support.device)[:, 0]
    compatible = (~known[:, None, :, None]) | (availability[None, None, None] == values[:, None, :, None])
    return support & compatible


def mixed_observation_nll(
    unary, transitions, support, event_constraints: Sequence[Mapping[str, EventConstraint]],
    *, initial_values: torch.Tensor | None = None,
    availability_values: torch.Tensor | None = None,
    availability_known: torch.Tensor | None = None,
) -> MixedObservationLoss:
    """Return separate event and joint event-plus-A NLLs for [B,K,T,6].

    Future availability labels constrain the numerator only. Branch
    probabilities are normalized before uniform mixing. At auxiliary weight
    one, the objective is the joint observation NLL, not two marginal losses.
    """
    centered_unary, centered_transitions, conditioned, values, denominator = _prepare_batch(
        unary, transitions, support, event_constraints, initial_values,
    )
    event_support = _event_mask(conditioned, values, event_constraints)
    event_logs = _branch_log_probability(centered_unary, centered_transitions, event_support, denominator)
    joint_support = _availability_support(event_support, availability_values, availability_known)
    joint_logs = _branch_log_probability(centered_unary, centered_transitions, joint_support, denominator)
    return MixedObservationLoss(-_uniform_mixture_log_probability(event_logs), -_uniform_mixture_log_probability(joint_logs))


def mixed_joint_first_flip_pmf(unary, transitions, support, *, initial_values=None) -> torch.Tensor:
    """Return [T,T] joint L/P first-flip PMF, with the none outcome last."""
    batch_inputs = _as_single_batch(unary, transitions, support)
    time_steps = unary.shape[1]
    initial = _single_initial_values(initial_values, support.device)
    required = {predicate: EventConstraint("none") for predicate in PREDICATE_NAMES}
    centered_unary, centered_transitions, conditioned, values, denominator = _prepare_batch(
        *batch_inputs, [required], initial,
    )
    outcomes = [EventConstraint("first_flip", step) for step in range(1, time_steps)] + [EventConstraint("none")]
    queries = [{"L": left, "P": right} for left in outcomes for right in outcomes]
    probabilities = []
    for start in range(0, len(queries), MAX_BATCH_SIZE):
        chunk = queries[start:start + MAX_BATCH_SIZE]
        count = len(chunk)
        chunk_support = conditioned.expand(count, -1, -1, -1)
        observed = _event_mask(chunk_support, values.expand(count, -1), chunk)
        branch_logs = _branch_log_probability(
            centered_unary.expand(count, -1, -1, -1),
            centered_transitions.expand(count, -1, -1, -1, -1), observed,
            denominator.expand(count, -1),
        )
        probabilities.append(torch.exp(_uniform_mixture_log_probability(branch_logs)))
    return torch.cat(probabilities).reshape(time_steps, time_steps)


def mixed_first_flip_pmf(*args, **kwargs) -> torch.Tensor:
    joint = mixed_joint_first_flip_pmf(*args, **kwargs)
    return torch.stack((joint.sum(dim=1), joint.sum(dim=0)))


def infer_mixed_chain(
    unary, transitions, support, *, initial_values=None, include_events: bool = True,
) -> MixedChainResult:
    """Infer each branch, then average its normalized joint distribution."""
    batch_inputs = _as_single_batch(unary, transitions, support)
    if not isinstance(include_events, bool):
        raise TypeError("include_events must be Boolean")
    if initial_values is not None:
        values = _resolve_initial_values(batch_inputs[2], _single_initial_values(initial_values, support.device), [{}])
        support = _condition_anchor(batch_inputs[2], values)[0]
    results = [infer_chain(unary[branch], transitions[branch], support[branch]) for branch in range(unary.shape[0])]
    branch_nodes = torch.stack([result.node_marginals for result in results])
    branch_edges = torch.stack([result.edge_marginals for result in results])
    nodes = branch_nodes.mean(dim=0)
    states = joint_assignments(device=unary.device)
    predicate_joint = torch.stack([
        nodes[:, (states[:, 2] == left) & (states[:, 3] == right)].sum(dim=-1)
        for left in (0, 1) for right in (0, 1)
    ], dim=-1).reshape(unary.shape[1], 2, 2)
    availability = torch.stack([nodes[:, states[:, 0] == value].sum(dim=-1) for value in (0, 1)], dim=-1)
    quality = torch.stack([nodes[:, states[:, 1] == value].sum(dim=-1) for value in range(3)], dim=-1)
    joint_event = mixed_joint_first_flip_pmf(unary, transitions, support) if include_events else None
    marginal_event = None if joint_event is None else torch.stack((joint_event.sum(dim=1), joint_event.sum(dim=0)))
    return MixedChainResult(
        torch.stack([result.logZ for result in results]), branch_nodes, branch_edges,
        nodes, nodes @ states[:, 2:].to(dtype=torch.float64), predicate_joint,
        availability, quality, marginal_event, joint_event,
    )


def condition_on_mixed_evidence(
    unary, transitions, support, *, availability_values=None, availability_known=None,
    predicate_values=None, predicate_known=None,
) -> MixedEvidenceUpdate:
    """Perform a single-branch hypothetical update; forecast samples stay fixed.

    Only entries marked known are observed. This diagnostic does not train a
    motion posterior or authorize using future labels as prediction inputs.
    """
    batch_inputs = _as_single_batch(unary, transitions, support)
    if unary.shape[0] != 1:
        raise ValueError("hypothetical evidence diagnostics require exactly one branch")
    if availability_values is None and predicate_values is None:
        raise ValueError("an evidence diagnostic requires explicit observations")
    centered_unary, centered_transitions, conditioned, _, denominator = _prepare_batch(
        *batch_inputs, [{}], None,
    )
    observed = _availability_support(
        conditioned,
        None if availability_values is None else availability_values[None],
        None if availability_known is None else availability_known[None],
    )
    evidence = _discrete_evidence(
        predicate_values, predicate_known, (unary.shape[1], 2), unary.device, "predicate",
    )
    if evidence is not None:
        values, known = evidence
        projection = joint_assignments(device=unary.device)[:, 2:]
        compatible = ((~known[:, None]) | (projection[None] == values[:, None])).all(dim=-1)
        observed = observed & compatible[None, None]
    log_probability = _branch_log_probability(centered_unary, centered_transitions, observed, denominator)[0, 0]
    if bool(torch.isneginf(log_probability)):
        raise ZeroSupportError("hypothetical evidence has zero supported probability")
    prior = infer_mixed_chain(unary, transitions, support, include_events=False)
    posterior = infer_mixed_chain(unary, transitions, observed[0], include_events=False)
    return MixedEvidenceUpdate(prior, posterior, log_probability)
