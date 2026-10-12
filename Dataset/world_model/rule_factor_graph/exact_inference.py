"""Differentiable exact chain inference with fixed float64 accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rule_potentials import MAX_CHAIN_STEPS, MAX_JOINT_STATES


class ZeroSupportError(ValueError):
    """A conditioning support admits no positive-mass chain trajectory."""


@dataclass(frozen=True)
class ChainResult:
    logZ: torch.Tensor
    node_marginals: torch.Tensor
    edge_marginals: torch.Tensor


def _validate_chain_inputs(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
) -> tuple[int, int]:
    if not all(isinstance(value, torch.Tensor) for value in (unary, transitions, support)):
        raise TypeError("unary, transitions, and support must be torch tensors")
    if unary.ndim != 2:
        raise ValueError("unary must have shape [T, S]")
    time_steps, states = unary.shape
    if not 1 <= time_steps <= MAX_CHAIN_STEPS:
        raise ValueError(f"chain time budget is 1..{MAX_CHAIN_STEPS}")
    if not 1 <= states <= MAX_JOINT_STATES:
        raise ValueError(f"joint state budget is 1..{MAX_JOINT_STATES}")
    if transitions.shape != (time_steps - 1, states, states):
        raise ValueError("transitions must have shape [T-1, S, S]")
    if support.shape != unary.shape or support.dtype != torch.bool:
        raise ValueError("support must be a Boolean tensor with shape [T, S]")
    if not unary.is_floating_point() or not transitions.is_floating_point():
        raise TypeError("chain scores must have floating point dtypes")
    if unary.dtype != transitions.dtype:
        raise TypeError("unary and transitions must have the same dtype")
    if unary.device != transitions.device or unary.device != support.device:
        raise ValueError("all chain inputs must be on the same device")
    for name, scores in (("unary", unary), ("transitions", transitions)):
        if bool((torch.isnan(scores) | torch.isposinf(scores)).any()):
            raise ValueError(f"{name} contains NaN or positive infinity")
    return time_steps, states


def _safe_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Keep all-unreachable reductions constant so their gradients are zero."""
    if bool((torch.isnan(values) | torch.isposinf(values)).any()):
        raise ValueError("chain score arithmetic produced NaN or positive infinity")
    reachable = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe_values = torch.where(reachable, values, torch.zeros_like(values))
    reduced = torch.logsumexp(safe_values, dim=dim)
    return torch.where(reachable.squeeze(dim), reduced, torch.full_like(reduced, -torch.inf))


def _center_scores(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate in float64 and center scores on complete feasible paths.

    The same centered scores must be reused by an event numerator and its
    denominator. Restoring offsets before subtracting partitions loses the
    normalized event probability's precision for large float32 constants.
    Casts retain autograd to the input dtype. Fixed float64 accumulation is
    part of the bounded solver's contract, including float32 score inputs.
    """
    unary = unary.to(dtype=torch.float64)
    transitions = transitions.to(dtype=torch.float64)
    active = support & torch.isfinite(unary)
    finite_edges = torch.isfinite(transitions)
    forward = [active[0]]
    for time in range(1, unary.shape[0]):
        forward.append(active[time] & (forward[-1][:, None] & finite_edges[time - 1]).any(dim=0))
    backward = [torch.zeros_like(active[0]) for _ in range(unary.shape[0])]
    backward[-1] = active[-1]
    for time in range(unary.shape[0] - 2, -1, -1):
        backward[time] = active[time] & (finite_edges[time] & backward[time + 1][None, :]).any(dim=1)
    viable_nodes = torch.stack(forward) & torch.stack(backward)
    masked_unary = torch.where(viable_nodes, unary, torch.full_like(unary, -torch.inf))
    unary_maxima = masked_unary.amax(dim=1)
    unary_offsets = torch.where(torch.isfinite(unary_maxima), unary_maxima, torch.zeros_like(unary_maxima)).detach()
    centered_unary = torch.where(
        torch.isfinite(masked_unary), masked_unary - unary_offsets[:, None], masked_unary,
    )
    edge_support = viable_nodes[:-1, :, None] & viable_nodes[1:, None, :] & finite_edges
    masked_transitions = torch.where(edge_support, transitions, torch.full_like(transitions, -torch.inf))
    transition_maxima = masked_transitions.amax(dim=(1, 2))
    transition_offsets = torch.where(
        torch.isfinite(transition_maxima), transition_maxima, torch.zeros_like(transition_maxima),
    ).detach()
    centered_transitions = torch.where(
        torch.isfinite(masked_transitions),
        masked_transitions - transition_offsets[:, None, None], masked_transitions,
    )
    return centered_unary, centered_transitions, unary_offsets.sum() + transition_offsets.sum()


def _forward(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    masked = torch.where(support, unary, torch.full_like(unary, -torch.inf))
    alpha = [masked[0]]
    for time in range(1, unary.shape[0]):
        incoming = alpha[-1][:, None] + transitions[time - 1]
        alpha.append(masked[time] + _safe_logsumexp(incoming, dim=0))
    log_partition = _safe_logsumexp(alpha[-1], dim=0)
    if unary.shape[0] == 1:
        log_partition = log_partition + transitions.sum()
    if bool(torch.isnan(log_partition) | torch.isposinf(log_partition)):
        raise ValueError("chain score arithmetic produced a non-finite partition")
    return masked, alpha, log_partition


def _log_partition(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
    *, allow_zero_mass: bool = False,
) -> torch.Tensor:
    _validate_chain_inputs(unary, transitions, support)
    _, _, log_partition = _forward(unary, transitions, support)
    if not allow_zero_mass and bool(torch.isneginf(log_partition)):
        raise ZeroSupportError("chain has zero total supported mass")
    return log_partition


def infer_chain(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
) -> ChainResult:
    """Return float64 log partition, node marginals, and edge marginals.

    Scores use the convention sum_t unary[t, x_t] plus
    sum_t transitions[t, x_t, x_{t+1}]. Unsupported states have zero mass.
    """
    time_steps, states = _validate_chain_inputs(unary, transitions, support)
    centered_unary, centered_transitions, offset = _center_scores(unary, transitions, support)
    masked, alpha, log_partition = _forward(centered_unary, centered_transitions, support)
    if bool(torch.isneginf(log_partition)):
        raise ZeroSupportError("chain has zero total supported mass")
    beta = [torch.zeros_like(unary[0]) for _ in range(time_steps)]
    for time in range(time_steps - 2, -1, -1):
        outgoing = centered_transitions[time] + masked[time + 1][None, :] + beta[time + 1][None, :]
        beta[time] = _safe_logsumexp(outgoing, dim=1)
    nodes = torch.stack([
        torch.exp(alpha[time] + beta[time] - log_partition)
        for time in range(time_steps)
    ])
    if time_steps == 1:
        edges = centered_unary.new_empty((0, states, states))
    else:
        edges = torch.stack([
            torch.exp(alpha[time][:, None] + centered_transitions[time]
                      + masked[time + 1][None, :] + beta[time + 1][None, :] - log_partition)
            for time in range(time_steps - 1)
        ])
    if not bool(torch.isfinite(nodes).all() & torch.isfinite(edges).all()):
        raise ValueError("chain score arithmetic produced non-finite marginals")
    restored_partition = log_partition + offset
    if not bool(torch.isfinite(restored_partition)):
        raise ValueError("chain score arithmetic overflowed the full log partition")
    return ChainResult(restored_partition, nodes, edges)


def infer_grounded_chain(
    unary: torch.Tensor, transitions: torch.Tensor, grounded_slices,
    assignments: torch.Tensor | None = None,
) -> ChainResult:
    """Validate grounded dependencies and compile rules before scoring a chain."""
    from .contracts import GroundedSlice
    from .grounding import validate_grounded_slice
    from .rule_potentials import compile_support

    slices = (grounded_slices,) if isinstance(grounded_slices, GroundedSlice) else tuple(grounded_slices)
    if not 1 <= len(slices) <= MAX_CHAIN_STEPS:
        raise ValueError(f"chain time budget is 1..{MAX_CHAIN_STEPS}")
    for item in slices:
        validate_grounded_slice(item)
    identity = lambda item: (item.episode_id, item.branch_id, item.actor_id, item.corridor_id,
                             item.owner_id, item.actor_class, item.corridor_class, item.tau_mps)
    if any(identity(item) != identity(slices[0]) for item in slices[1:]):
        raise ValueError("grounded chain changes episode, branch, owner, role class, or threshold")
    if any(right.tick - left.tick != 5 for left, right in zip(slices, slices[1:])):
        raise ValueError("grounded corridor chain must use ordered five-tick half-second steps")
    support = compile_support(slices, assignments).to(device=unary.device)
    return infer_chain(unary, transitions, support)
