"""First-flip and censoring probabilities from the joint chain partition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .exact_inference import ZeroSupportError, _center_scores, _log_partition, _validate_chain_inputs
from .rule_potentials import PREDICATE_NAMES, validate_assignments

EVENT_STEP_SECONDS = 0.5
MAX_EVENT_BINS = 10


@dataclass(frozen=True)
class EventConstraint:
    """An event uses grid steps, with t0 at step zero and flips after t0."""

    kind: str
    step: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("first_flip", "right_censored", "none"):
            raise ValueError("event kind must be first_flip, right_censored, or none")
        if self.kind == "none":
            if self.step is not None:
                raise ValueError("the none event does not take a step")
        elif isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TypeError("first_flip and right_censored require an integer step")


def _initial_values(
    support: torch.Tensor, assignments: torch.Tensor,
    initial_values: Sequence[bool | None] | None,
    queried: Sequence[int],
) -> tuple[bool | None, bool | None, bool | None]:
    if initial_values is not None:
        if len(initial_values) != 3:
            raise ValueError("initial_values must use O, M, S order")
        values = tuple(initial_values)
        if any(value is not None and not isinstance(value, bool) for value in values):
            raise TypeError("initial values must be Boolean or None")
    else:
        if not bool(support[0].any()):
            raise ZeroSupportError("chain has zero initial support")
        values = tuple(
            bool(assignments[support[0], column][0])
            if bool((assignments[support[0], column] == assignments[support[0], column][0]).all())
            else None
            for column in range(3)
        )
    for column in queried:
        if values[column] is None:
            raise ValueError(f"unknown initial truth for queried predicate {PREDICATE_NAMES[column]}")
    return values


def _condition_anchor(
    support: torch.Tensor, assignments: torch.Tensor,
    values: Sequence[bool | None],
) -> torch.Tensor:
    anchor = support[0].clone()
    for column, value in enumerate(values):
        if value is not None:
            anchor = anchor & (assignments[:, column] == value)
    return torch.cat((anchor[None, :], support[1:]), dim=0)


def _event_support(
    support: torch.Tensor, assignments: torch.Tensor,
    constraints: Mapping[str, EventConstraint], values: Sequence[bool | None],
) -> torch.Tensor:
    constrained = support.clone()
    horizon = support.shape[0] - 1
    for predicate, observation in constraints.items():
        if predicate not in PREDICATE_NAMES:
            raise ValueError(f"unknown predicate {predicate!r}; use O, M, or S")
        if not isinstance(observation, EventConstraint):
            raise TypeError("event evidence must use EventConstraint objects")
        column = PREDICATE_NAMES.index(predicate)
        same = assignments[:, column] == values[column]
        if observation.kind == "first_flip":
            step = observation.step
            if not 1 <= step <= horizon:
                raise ValueError("first_flip step must lie in 1..T-1")
            constrained[:step] = constrained[:step] & same[None, :]
            constrained[step] = constrained[step] & ~same
        elif observation.kind == "right_censored":
            step = observation.step
            if not 0 <= step <= horizon:
                raise ValueError("right_censored step must lie in 0..T-1")
            constrained[:step + 1] = constrained[:step + 1] & same[None, :]
        else:
            constrained = constrained & same[None, :]
    return constrained


def _prepare(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
    assignments: torch.Tensor | None, initial_values: Sequence[bool | None] | None,
    queried: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, tuple[bool | None, bool | None, bool | None],
           torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_chain_inputs(unary, transitions, support)
    states = validate_assignments(assignments).to(device=unary.device).bool()
    if states.shape[0] != unary.shape[1]:
        raise ValueError("assignment count must equal the chain state count")
    values = _initial_values(support, states, initial_values, queried)
    conditioned = _condition_anchor(support, states, values)
    centered_unary, centered_transitions, _ = _center_scores(unary, transitions, conditioned)
    denominator = _log_partition(centered_unary, centered_transitions, conditioned)
    return states, conditioned, values, denominator, centered_unary, centered_transitions


def event_log_probability(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
    constraints: Mapping[str, EventConstraint],
    assignments: torch.Tensor | None = None,
    *, initial_values: Sequence[bool | None] | None = None,
) -> torch.Tensor:
    """Compute one joint event log probability, including joint censoring."""
    if not isinstance(constraints, Mapping) or not constraints:
        raise ValueError("constraints must be a nonempty predicate-to-event mapping")
    if any(predicate not in PREDICATE_NAMES for predicate in constraints):
        raise ValueError("event constraints must use O, M, or S predicate names")
    queried = tuple(PREDICATE_NAMES.index(predicate) for predicate in constraints)
    states, conditioned, values, denominator, centered_unary, centered_transitions = _prepare(
        unary, transitions, support, assignments, initial_values, queried,
    )
    observed = _event_support(conditioned, states, constraints, values)
    numerator = _log_partition(centered_unary, centered_transitions, observed, allow_zero_mass=True)
    return torch.where(torch.isfinite(numerator), numerator - denominator, numerator)


def event_nll(*args, **kwargs) -> torch.Tensor:
    return -event_log_probability(*args, **kwargs)


def first_flip_pmf(
    unary: torch.Tensor, transitions: torch.Tensor, support: torch.Tensor,
    assignments: torch.Tensor | None = None,
    *, initial_values: Sequence[bool | None] | None = None,
) -> torch.Tensor:
    """Return [3, T] marginal PMFs: steps 1..T-1, then no flip.

    T=11 represents ten half-second intervals and a final none outcome. A
    first flip constrains the prefix only; later returns remain permitted.
    """
    states, conditioned, values, denominator, centered_unary, centered_transitions = _prepare(
        unary, transitions, support, assignments, initial_values, (0, 1, 2),
    )
    rows = []
    for predicate in PREDICATE_NAMES:
        probabilities = []
        outcomes = [EventConstraint("first_flip", step) for step in range(1, unary.shape[0])]
        outcomes.append(EventConstraint("none"))
        for outcome in outcomes:
            observed = _event_support(conditioned, states, {predicate: outcome}, values)
            numerator = _log_partition(centered_unary, centered_transitions, observed, allow_zero_mass=True)
            log_probability = torch.where(torch.isfinite(numerator), numerator - denominator, numerator)
            probabilities.append(torch.exp(log_probability))
        rows.append(torch.stack(probabilities))
    return torch.stack(rows)


def _normalized_weights(
    branches: int, reference: torch.Tensor, weights: torch.Tensor | None,
) -> torch.Tensor:
    if weights is None:
        return reference.new_full((branches,), 1.0 / branches)
    if not isinstance(weights, torch.Tensor) or weights.shape != (branches,):
        raise ValueError("mixture weights must have shape [K]")
    if not weights.is_floating_point() or weights.device != reference.device:
        raise ValueError("mixture weights must be floating tensors on the branch device")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("mixture weights must be finite and nonnegative")
    total = weights.sum()
    if not bool(torch.isfinite(total)) or not bool(total > 0):
        raise ValueError("mixture weights must have positive finite total mass")
    return weights / total


def mixture_first_flip_pmf(
    branch_pmfs: torch.Tensor, weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average normalized branch PMFs before applying an event loss."""
    if not isinstance(branch_pmfs, torch.Tensor) or branch_pmfs.ndim != 3:
        raise ValueError("branch PMFs must have shape [K, 3, T]")
    branches, predicates, outcomes = branch_pmfs.shape
    if branches < 1 or predicates != 3 or not 1 <= outcomes <= MAX_EVENT_BINS + 1:
        raise ValueError("branch PMFs must have nonempty bounded shape [K, 3, T]")
    if not branch_pmfs.is_floating_point() or not bool(torch.isfinite(branch_pmfs).all()):
        raise ValueError("branch PMFs must be finite floating probabilities")
    if bool((branch_pmfs < 0).any()):
        raise ValueError("branch PMFs cannot contain negative probabilities")
    totals = branch_pmfs.sum(dim=-1)
    if not torch.allclose(totals, torch.ones_like(totals), rtol=1e-5, atol=1e-7):
        raise ValueError("every branch predicate PMF must already be normalized")
    normalized_weights = _normalized_weights(branches, branch_pmfs, weights)
    return (normalized_weights[:, None, None] * branch_pmfs).sum(dim=0)


def mixture_event_nll(
    branch_log_probabilities: torch.Tensor, weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Take NLL after mixing each branch's normalized joint event probability."""
    if (not isinstance(branch_log_probabilities, torch.Tensor)
            or branch_log_probabilities.ndim != 1 or branch_log_probabilities.shape[0] < 1):
        raise ValueError("branch event log probabilities must have nonempty shape [K]")
    if not branch_log_probabilities.is_floating_point():
        raise TypeError("branch event log probabilities must be floating point")
    if bool((torch.isnan(branch_log_probabilities) | torch.isposinf(branch_log_probabilities)).any()):
        raise ValueError("branch event log probabilities contain NaN or positive infinity")
    if bool((branch_log_probabilities > 1e-6).any()):
        raise ValueError("normalized event log probabilities cannot be positive")
    normalized_weights = _normalized_weights(
        branch_log_probabilities.shape[0], branch_log_probabilities, weights,
    )
    positive = normalized_weights > 0
    safe_weights = torch.where(positive, normalized_weights, torch.ones_like(normalized_weights))
    log_weights = torch.where(positive, torch.log(safe_weights), torch.full_like(safe_weights, -torch.inf))
    from .exact_inference import _safe_logsumexp

    return -_safe_logsumexp(log_weights + branch_log_probabilities, dim=0)
