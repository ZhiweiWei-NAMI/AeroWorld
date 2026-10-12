"""Batched exact shared-regime chains and joint censored-event likelihoods.

Scores enter one partition per sampled actor trajectory. Each branch is
normalized before the frozen, uniformly sampled q branches are averaged.
Future labels affect an observation numerator only, never the prior support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch

from .events import EventConstraint
from .exact_inference import ZeroSupportError, _safe_logsumexp
from .corridor_state_space import PREDICATE_NAMES, predicate_assignments, state_assignments, contradiction_states, speed_conflict_states


@dataclass(frozen=True)
class CorridorChainResult:
    branch_logZ: torch.Tensor
    branch_node_marginals: torch.Tensor
    branch_edge_marginals: torch.Tensor
    joint_state_marginals: torch.Tensor
    latent_marginals: torch.Tensor
    predicate_marginals: torch.Tensor
    speed_conflict_mass: torch.Tensor
    contradiction_mass: torch.Tensor
    first_flip_pmf: torch.Tensor | None


@dataclass(frozen=True)
class CorridorObservationLoss:
    event_nll: torch.Tensor
    joint_nll: torch.Tensor
    branch_event_log_probability: torch.Tensor
    branch_responsibility: torch.Tensor
    event_observed: torch.Tensor
    joint_observed: torch.Tensor


def _validate_batch(unary, transitions, support, *, separate_b):
    states = 16 if separate_b else 8
    if not isinstance(separate_b, bool):
        raise TypeError("separate_b must be Boolean")
    if not all(isinstance(x, torch.Tensor) for x in (unary, transitions, support)):
        raise TypeError("corridor chain inputs must be torch tensors")
    if unary.ndim != 4 or unary.shape[-1] != states:
        raise ValueError(f"corridor unary must have shape [B,K,T,{states}]")
    batch, branches, times, _ = unary.shape
    if batch < 1 or branches < 1 or not 1 <= times <= 11:
        raise ValueError("corridor chains require positive B,K and T in 1..11")
    if transitions.shape != (batch, branches, times-1, states, states):
        raise ValueError("corridor transitions must have shape [B,K,T-1,S,S]")
    if support.shape != unary.shape or support.dtype != torch.bool:
        raise ValueError("corridor support must be Boolean with unary shape")
    if not unary.is_floating_point() or not transitions.is_floating_point() or unary.dtype != transitions.dtype:
        raise TypeError("corridor scores must share a floating dtype")
    if unary.device != transitions.device or unary.device != support.device:
        raise ValueError("corridor scores and support must share a device")
    for scores in (unary, transitions):
        if bool((torch.isnan(scores) | torch.isposinf(scores)).any()):
            raise ValueError("corridor scores contain NaN or positive infinity")
    return batch, branches, times, states


def _center_scores(unary, transitions, support):
    unary, transitions = unary.double(), transitions.double()
    active, finite_edges = support & torch.isfinite(unary), torch.isfinite(transitions)
    times = unary.shape[2]
    forward = [active[:, :, 0]]
    for time in range(1, times):
        forward.append(active[:, :, time] & (forward[-1][..., :, None] & finite_edges[:, :, time-1]).any(-2))
    backward = [torch.zeros_like(active[:, :, 0]) for _ in range(times)]
    backward[-1] = active[:, :, -1]
    for time in range(times-2, -1, -1):
        backward[time] = active[:, :, time] & (finite_edges[:, :, time] & backward[time+1][..., None, :]).any(-1)
    viable = torch.stack(forward, 2) & torch.stack(backward, 2)
    masked = torch.where(viable, unary, torch.full_like(unary, -torch.inf))
    maxima = masked.amax(-1)
    node_offsets = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima)).detach()
    centered = torch.where(torch.isfinite(masked), masked-node_offsets[..., None], masked)
    edge_viable = viable[:, :, :-1, :, None] & viable[:, :, 1:, None, :] & finite_edges
    masked_edges = torch.where(edge_viable, transitions, torch.full_like(transitions, -torch.inf))
    maxima = masked_edges.amax((-1, -2))
    edge_offsets = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima)).detach()
    centered_edges = torch.where(torch.isfinite(masked_edges), masked_edges-edge_offsets[..., None, None], masked_edges)
    return centered, centered_edges, node_offsets.sum(-1)+edge_offsets.sum(-1)


def _forward(unary, transitions, support):
    masked = torch.where(support, unary, torch.full_like(unary, -torch.inf))
    alpha = [masked[:, :, 0]]
    for time in range(1, unary.shape[2]):
        alpha.append(masked[:, :, time] + _safe_logsumexp(alpha[-1][..., :, None]+transitions[:, :, time-1], -2))
    partition = _safe_logsumexp(alpha[-1], -1)
    if unary.shape[2] == 1:
        partition = partition + transitions.sum((-1, -2, -3))
    return masked, alpha, partition


def corridor_batch_log_partition(unary, transitions, support, *, separate_b=False):
    """Return [B,K] logZ with float64 accumulation and intact autograd."""
    _validate_batch(unary, transitions, support, separate_b=separate_b)
    centered, edges, offset = _center_scores(unary, transitions, support)
    _, _, partition = _forward(centered, edges, support)
    if bool(torch.isneginf(partition).any()):
        raise ZeroSupportError("corridor prior contains a zero-mass branch")
    restored = partition+offset
    if not bool(torch.isfinite(restored).all()):
        raise ValueError("corridor partition arithmetic overflowed")
    return restored


def _validate_constraints(constraints, batch):
    if not isinstance(constraints, Sequence) or len(constraints) != batch:
        raise ValueError("event_constraints require one mapping per batch item")
    for row in constraints:
        if not isinstance(row, Mapping):
            raise TypeError("each event constraint row must be a mapping")
        if any(key not in PREDICATE_NAMES for key in row):
            raise ValueError("corridor event names use O1,M1,S1,O2,M2,S2")
        if any(not isinstance(value, EventConstraint) for value in row.values()):
            raise TypeError("corridor observations must use EventConstraint")


def _initial_values(support, projection, initial_values, constraints):
    batch = support.shape[0]
    if initial_values is None:
        anchor = support[:, :, 0]
        zero = (anchor[..., None] & ~projection.bool()[None, None]).any((1, 2))
        one = (anchor[..., None] & projection.bool()[None, None]).any((1, 2))
        if bool((~(zero | one)).any()):
            raise ZeroSupportError("corridor chain has zero initial support")
        values = torch.where(zero & one, -1, one.long())
    else:
        if not isinstance(initial_values, torch.Tensor) or initial_values.shape != (batch, 6):
            raise ValueError("initial_values must be discrete [B,6]")
        if initial_values.is_floating_point() or initial_values.is_complex() or initial_values.device != support.device:
            raise TypeError("initial_values must be discrete on the chain device")
        if not bool(((initial_values == -1) | (initial_values == 0) | (initial_values == 1)).all()):
            raise ValueError("initial_values use 0/1, with -1 explicitly unknown")
        values = initial_values.long()
    for row, observations in enumerate(constraints):
        for predicate in observations:
            if values[row, PREDICATE_NAMES.index(predicate)] < 0:
                raise ValueError(f"initial truth for queried predicate {predicate} is unknown")
    return values


def _prepare(unary, transitions, support, constraints, initial_values, separate_b):
    batch, _, _, _ = _validate_batch(unary, transitions, support, separate_b=separate_b)
    _validate_constraints(constraints, batch)
    projection = predicate_assignments(separate_b=separate_b, device=unary.device)
    values = _initial_values(support, projection, initial_values, constraints)
    anchor_mask = ((values[:, None] < 0) | (projection[None] == values[:, None])).all(-1)
    conditioned = torch.cat((support[:, :, :1] & anchor_mask[:, None, None], support[:, :, 1:]), 2)
    centered, edges, offset = _center_scores(unary, transitions, conditioned)
    _, _, denominator = _forward(centered, edges, conditioned)
    if bool(torch.isneginf(denominator).any()):
        raise ZeroSupportError("corridor prefix conditioning contains a zero-mass branch")
    return centered, edges, conditioned, projection, values, denominator, offset


def _event_support(support, projection, values, constraints):
    batch, _, times, states = support.shape
    mask = torch.ones((batch, times, states), dtype=torch.bool, device=support.device)
    observed = torch.zeros(batch, dtype=torch.bool, device=support.device)
    for row, observations in enumerate(constraints):
        for predicate, event in observations.items():
            same = projection[:, PREDICATE_NAMES.index(predicate)] == values[row, PREDICATE_NAMES.index(predicate)]
            if event.kind == "first_flip":
                if not 1 <= event.step < times:
                    raise ValueError("first_flip step must lie in 1..T-1")
                mask[row, :event.step] &= same[None]
                mask[row, event.step] &= ~same
                observed[row] = True
            elif event.kind == "right_censored":
                if not 0 <= event.step < times:
                    raise ValueError("right_censored step must lie in 0..T-1")
                mask[row, :event.step+1] &= same[None]
                observed[row] |= event.step > 0
            else:
                mask[row] &= same[None]
                observed[row] |= times > 1
    return support & mask[:, None], observed


def _branch_log_probability(unary, transitions, observed_support, denominator):
    _, _, numerator = _forward(unary, transitions, observed_support)
    return torch.where(torch.isfinite(numerator), numerator-denominator, numerator)


def _mixture_log_probability(branch_logs):
    return _safe_logsumexp(branch_logs, -1)-math.log(branch_logs.shape[-1])


def _latent_support(support, latent_values, latent_known, separate_b):
    if latent_values is None and latent_known is None:
        return support, torch.zeros(support.shape[0], dtype=torch.bool, device=support.device)
    dimensions = 4 if separate_b else 3
    shape = (support.shape[0], support.shape[2], dimensions)
    if not isinstance(latent_values, torch.Tensor) or not isinstance(latent_known, torch.Tensor):
        raise TypeError("latent labels and known mask must both be tensors")
    if latent_values.shape != shape or latent_known.shape != shape or latent_known.dtype != torch.bool:
        raise ValueError(f"latent values/mask must have shape {shape}, with Boolean mask")
    if latent_values.device != support.device or latent_known.device != support.device:
        raise ValueError("latent supervision must share the chain device")
    if latent_values.is_floating_point() or latent_values.is_complex():
        raise TypeError("latent labels must be discrete")
    if not bool(((latent_values == -1) | (latent_values == 0) | (latent_values == 1)).all()):
        raise ValueError("latent labels use 0/1, with -1 explicitly unknown")
    if bool((latent_known & (latent_values < 0)).any()):
        raise ValueError("known latent labels cannot carry unknown values")
    states = state_assignments(separate_b=separate_b, device=support.device)
    compatible = ((~latent_known[:, :, None]) | (latent_values[:, :, None] == states[None, None])).all(-1)
    return support & compatible[:, None], latent_known[:, 1:].any((1, 2))


def corridor_observation_nll(
    unary, transitions, support, event_constraints, *, initial_values=None,
    latent_values=None, latent_known=None, separate_b=False,
) -> CorridorObservationLoss:
    """One joint observed-event NLL per window, optionally with partial labels.

    For event A and branch k, ell_k=logZ_A,k-logZ_k, and
    NLL=-logsumexp_k(ell_k)+log K. Thus dNLL/dtheta is the
    event-responsibility-weighted E_full[dscore]-E_event[dscore].
    This is never the mean branch NLL or a sum of marginal predicate NLLs.
    Zero observed future bins are explicitly flagged for caller exclusion.
    """
    centered, edges, conditioned, projection, values, denominator, _ = _prepare(
        unary, transitions, support, event_constraints, initial_values, separate_b)
    event_support, event_observed = _event_support(conditioned, projection, values, event_constraints)
    event_logs = _branch_log_probability(centered, edges, event_support, denominator)
    joint_support, latent_observed = _latent_support(event_support, latent_values, latent_known, separate_b)
    joint_logs = _branch_log_probability(centered, edges, joint_support, denominator)
    norm = _safe_logsumexp(event_logs, -1)
    # For an impossible observation, no branch owns posterior mass; NLL is inf.
    finite_norm = torch.where(torch.isfinite(norm), norm, torch.zeros_like(norm))
    responsibility = torch.where(torch.isfinite(norm[:, None]), (event_logs-finite_norm[:, None]).exp(), torch.zeros_like(event_logs))
    return CorridorObservationLoss(-_mixture_log_probability(event_logs), -_mixture_log_probability(joint_logs),
                                   event_logs, responsibility, event_observed, event_observed | latent_observed)


def corridor_first_flip_pmf(unary, transitions, support, *, initial_values=None, separate_b=False):
    """Return [B,6,T], steps1..T-1 followed by the no-flip outcome."""
    batch = unary.shape[0]
    required = [{name: EventConstraint("none") for name in PREDICATE_NAMES} for _ in range(batch)]
    centered, edges, conditioned, projection, values, denominator, _ = _prepare(
        unary, transitions, support, required, initial_values, separate_b)
    rows = []
    outcomes = [EventConstraint("first_flip", step) for step in range(1, unary.shape[2])] + [EventConstraint("none")]
    for predicate in PREDICATE_NAMES:
        probabilities = []
        for outcome in outcomes:
            observations = [{predicate: outcome} for _ in range(batch)]
            observed, _ = _event_support(conditioned, projection, values, observations)
            branch_logs = _branch_log_probability(centered, edges, observed, denominator)
            probabilities.append(_mixture_log_probability(branch_logs).exp())
        rows.append(torch.stack(probabilities, -1))
    return torch.stack(rows, 1)


def infer_corridor_batch(unary, transitions, support, *, initial_values=None, include_events=True, separate_b=False):
    """Infer branch chains, then average normalized distributions under frozen q."""
    if not isinstance(include_events, bool):
        raise TypeError("include_events must be Boolean")
    batch = unary.shape[0]
    centered, edges, conditioned, projection, _, denominator, offset = _prepare(
        unary, transitions, support, [{} for _ in range(batch)], initial_values, separate_b)
    masked, alpha, logz = _forward(centered, edges, conditioned)
    times, states = unary.shape[2:]
    beta = [torch.zeros_like(centered[:, :, 0]) for _ in range(times)]
    for time in range(times-2, -1, -1):
        beta[time] = _safe_logsumexp(edges[:, :, time]+masked[:, :, time+1, None, :]+beta[time+1][..., None, :], -1)
    branch_nodes = torch.stack([(alpha[time]+beta[time]-logz[..., None]).exp() for time in range(times)], 2)
    if times > 1:
        branch_edges = torch.stack([(alpha[time][..., :, None]+edges[:, :, time]
                                     +masked[:, :, time+1, None, :]+beta[time+1][..., None, :]
                                     -logz[..., None, None]).exp() for time in range(times-1)], 2)
    else:
        branch_edges = centered.new_empty((batch, unary.shape[1], 0, states, states))
    nodes = branch_nodes.mean(1)
    latent = state_assignments(separate_b=separate_b, device=unary.device)
    latent_marginals = torch.stack([torch.stack([nodes[..., latent[:, column] == value].sum(-1) for value in (0, 1)], -1)
                                    for column in range(latent.shape[-1])], -2)
    predicate_marginals = nodes @ projection.double()
    contradiction = nodes[..., contradiction_states(separate_b=separate_b, device=unary.device)].sum(-1)
    speed_conflict = nodes[..., speed_conflict_states(separate_b=separate_b, device=unary.device)].sum(-1)
    event_pmf = corridor_first_flip_pmf(unary, transitions, conditioned, initial_values=initial_values,
                                        separate_b=separate_b) if include_events else None
    return CorridorChainResult(denominator+offset, branch_nodes, branch_edges, nodes, latent_marginals,
                               predicate_marginals, speed_conflict, contradiction, event_pmf)
