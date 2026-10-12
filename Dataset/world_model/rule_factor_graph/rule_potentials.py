"""Hard support for the corridor occupancy, movement, and stationary rules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

import torch

from .contracts import GroundedSlice, UnsupportedGroundingError

PREDICATE_NAMES = ("O", "M", "S")
MAX_CHAIN_STEPS = 11
MAX_JOINT_STATES = 8
DEFAULT_ASSIGNMENTS = torch.tensor(list(product((0, 1), repeat=3)), dtype=torch.long)


def validate_assignments(assignments: torch.Tensor | None = None) -> torch.Tensor:
    """Return a discrete, unique state table in O, M, S column order."""
    if assignments is None:
        assignments = DEFAULT_ASSIGNMENTS.clone()
    if not isinstance(assignments, torch.Tensor):
        raise TypeError("assignments must be a torch.Tensor")
    if assignments.ndim != 2 or assignments.shape[1] != len(PREDICATE_NAMES):
        raise ValueError("assignments must have shape [S, 3] in O, M, S order")
    if not 1 <= assignments.shape[0] <= MAX_JOINT_STATES:
        raise ValueError(f"joint state budget is 1..{MAX_JOINT_STATES}")
    if assignments.is_floating_point() or assignments.is_complex():
        raise TypeError("assignments must have a Boolean or integer dtype")
    if not bool(((assignments == 0) | (assignments == 1)).all()):
        raise ValueError("assignments must contain only Boolean values")
    if torch.unique(assignments, dim=0).shape[0] != assignments.shape[0]:
        raise ValueError("joint assignments must be unique")
    return assignments


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return number


def compile_support(
    grounded_slice: GroundedSlice | Sequence[GroundedSlice],
    assignments: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compile [T, S] support by completing unknown state fields existentially.

    O = inside, M = inside and speed > tau, S = inside and speed <= tau.
    Unknown inputs retain every compatible legal joint assignment. Observed
    predicate evidence intersects that support; contradictory evidence fails.
    """
    states = validate_assignments(assignments).bool()
    if isinstance(grounded_slice, GroundedSlice):
        slices = (grounded_slice,)
    elif isinstance(grounded_slice, Sequence):
        slices = tuple(grounded_slice)
    else:
        raise TypeError("grounded_slice must be a GroundedSlice or a sequence of slices")
    if not 1 <= len(slices) <= MAX_CHAIN_STEPS:
        raise ValueError(f"chain time budget is 1..{MAX_CHAIN_STEPS}")
    occupancy, moving, stationary = states.unbind(dim=1)
    legal = ((~occupancy & ~moving & ~stationary)
             | (occupancy & moving & ~stationary)
             | (occupancy & ~moving & stationary))
    rows = []
    for item in slices:
        if not isinstance(item, GroundedSlice):
            raise TypeError("every grounded slice must be a GroundedSlice")
        if not isinstance(item.applicable, bool):
            raise TypeError("applicable must be Boolean")
        if not item.applicable:
            raise UnsupportedGroundingError("out_of_scope corridor rule slice")
        tau = _finite_nonnegative(item.tau_mps, "tau_mps")
        if item.inside is not None and not isinstance(item.inside, bool):
            raise TypeError("inside must be Boolean or explicitly unknown (None)")
        speed = None if item.speed_mps is None else _finite_nonnegative(item.speed_mps, "speed_mps")
        if len(item.predicate_values) != 3 or len(item.predicate_observed) != 3:
            raise ValueError("predicate evidence must use O, M, S order")
        row = legal.clone()
        if item.inside is not None:
            row = row & (occupancy == item.inside)
        if speed is not None:
            is_moving = speed > tau
            row = row & (moving == (occupancy & is_moving))
            row = row & (stationary == (occupancy & (not is_moving)))
        for column, (value, observed) in enumerate(zip(item.predicate_values, item.predicate_observed)):
            if not isinstance(observed, bool):
                raise TypeError("predicate observation flags must be Boolean")
            if value is not None and not isinstance(value, bool):
                raise TypeError("predicate evidence must be Boolean or None")
            if observed != (value is not None):
                raise ValueError("known predicate evidence and observation flags disagree")
            if observed:
                row = row & (states[:, column] == value)
        if not bool(row.any()):
            raise ValueError(f"zero grounded support at tick {item.tick}")
        rows.append(row)
    return torch.stack(rows)
