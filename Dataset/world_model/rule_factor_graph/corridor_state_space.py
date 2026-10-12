"""Latent corridor regimes and exact support for two authored ENU boxes.

The shared state order is lexicographic (B, I1, I2). Predicate projection
is deliberately many-to-one: both all-outside B values remain latent states.
Geometry follows predicate_state_computers._point_in_oriented_corridor_box:
extent_m is a half extent, yaw rotates ENU into local XY, and faces are
closed after expanding each half extent by 1e-9 metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from collections.abc import Sequence

import torch

from .contracts import StateKey

PREDICATE_NAMES = ("O1", "M1", "S1", "O2", "M2", "S2")
GEOMETRY_TOLERANCE_M = 1e-9
GEOMETRY_SOURCE = "Dataset/semantic_simulation/predicate_state_computers.py#_point_in_oriented_corridor_box"


@dataclass(frozen=True)
class CorridorBox:
    corridor_id: str
    center_enu_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    yaw_deg: float
    source_ref: str

    def __post_init__(self):
        if not isinstance(self.corridor_id, str) or not self.corridor_id:
            raise ValueError("corridor_id must be a nonempty identity")
        if not isinstance(self.source_ref, str) or not self.source_ref:
            raise ValueError("corridor box needs its static source reference")
        if len(self.center_enu_m) != 3 or len(self.extent_m) != 3:
            raise ValueError("corridor center and half extent must have three ENU coordinates")
        if any(not math.isfinite(float(x)) for x in (*self.center_enu_m, *self.extent_m, self.yaw_deg)):
            raise ValueError("corridor geometry must be finite")
        if any(float(x) <= 0 for x in self.extent_m):
            raise ValueError("corridor half extents must be positive")

    @property
    def half_extent_with_tolerance(self):
        return tuple(float(x) + GEOMETRY_TOLERANCE_M for x in self.extent_m)

    @property
    def xy_axes(self):
        cardinal = float(self.yaw_deg) % 360
        if cardinal in (0.0, 90.0, 180.0, 270.0):
            return {0.0: ((1.0, 0.0), (0.0, 1.0)),
                    90.0: ((0.0, 1.0), (-1.0, 0.0)),
                    180.0: ((-1.0, 0.0), (0.0, -1.0)),
                    270.0: ((0.0, -1.0), (1.0, 0.0))}[cardinal]
        angle = math.radians(float(self.yaw_deg))
        return ((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle)))


def state_assignments(*, separate_b: bool = False, device=None) -> torch.Tensor:
    """Return [8,3] B,I1,I2 or diagnostic [16,4] B1,B2,I1,I2."""
    if not isinstance(separate_b, bool):
        raise TypeError("separate_b must be Boolean")
    return torch.tensor(list(product((0, 1), repeat=4 if separate_b else 3)),
                        dtype=torch.long, device=device)


def predicate_assignments(*, separate_b: bool = False, device=None) -> torch.Tensor:
    """Return deterministic O1,M1,S1,O2,M2,S2 without deduplicating."""
    states = state_assignments(separate_b=separate_b, device=device)
    if separate_b:
        b1, b2, i1, i2 = states.unbind(-1)
    else:
        b1, i1, i2 = states.unbind(-1)
        b2 = b1
    return torch.stack((i1, i1 * b1, i1 * (1-b1), i2, i2 * b2, i2 * (1-b2)), -1)


def _validate_boxes(boxes: Sequence[CorridorBox]) -> tuple[CorridorBox, CorridorBox]:
    if not isinstance(boxes, Sequence) or len(boxes) != 2 or any(not isinstance(x, CorridorBox) for x in boxes):
        raise TypeError("exact corridor support requires two CorridorBox objects")
    if boxes[0].corridor_id == boxes[1].corridor_id:
        raise ValueError("the joint corridor case requires two distinct corridor owners")
    return boxes[0], boxes[1]


def _effective_corners(box: CorridorBox):
    axis_x, axis_y = box.xy_axes
    half = box.half_extent_with_tolerance
    return tuple((box.center_enu_m[0] + sx*half[0]*axis_x[0] + sy*half[1]*axis_y[0],
                  box.center_enu_m[1] + sx*half[0]*axis_x[1] + sy*half[1]*axis_y[1],
                  box.center_enu_m[2] + sz*half[2])
                 for sx, sy, sz in product((-1, 1), repeat=3))


def _contains_box(container: CorridorBox, contained: CorridorBox):
    # Analytic vertex extrema avoid rotated-corner roundtrip error. Identical
    # orientation uses its exact half extents rather than dot(axis,axis).
    outer_half, inner_half = container.half_extent_with_tolerance, contained.half_extent_with_tolerance
    delta = tuple(contained.center_enu_m[j]-container.center_enu_m[j] for j in range(3))
    if abs(delta[2])+inner_half[2] > outer_half[2]:
        return False
    axes, inner_axes = container.xy_axes, contained.xy_axes
    same_orientation = float(container.yaw_deg) % 360 == float(contained.yaw_deg) % 360
    for j, axis in enumerate(axes):
        shift = abs(math.fsum(delta[k]*axis[k] for k in range(2)))
        radius = inner_half[j] if same_orientation else math.fsum(
            inner_half[k]*abs(math.fsum(inner_axes[k][l]*axis[l] for l in range(2))) for k in range(2))
        if shift+radius > outer_half[j]:
            return False
    return True


def _boxes_intersect(first: CorridorBox, second: CorridorBox):
    first_half, second_half = first.half_extent_with_tolerance, second.half_extent_with_tolerance
    delta = tuple(second.center_enu_m[j] - first.center_enu_m[j] for j in range(3))
    if abs(delta[2]) > first_half[2] + second_half[2]:
        return False
    first_axes, second_axes = first.xy_axes, second.xy_axes
    # Separating axis theorem: the two local axes of each rectangle suffice.
    for axis in (*first_axes, *second_axes):
        center_distance = abs(sum(delta[j] * axis[j] for j in range(2)))
        radius_first = sum(first_half[k]*abs(sum(first_axes[k][j]*axis[j] for j in range(2))) for k in range(2))
        radius_second = sum(second_half[k]*abs(sum(second_axes[k][j]*axis[j] for j in range(2))) for k in range(2))
        if center_distance > radius_first + radius_second:
            return False
    return True


def geometry_membership_support(first: CorridorBox, second: CorridorBox, *, device=None) -> torch.Tensor:
    """Return feasible [I1,I2] pairs in unconstrained ENU space.

    00 is feasible outside the finite union. 11 requires closed intersection;
    10 and 01 require the corresponding set differences to be nonempty.
    No unknown scene domain constraints are invented.
    """
    _validate_boxes((first, second))
    return torch.tensor([[True, not _contains_box(first, second)],
                         [not _contains_box(second, first), _boxes_intersect(first, second)]],
                        dtype=torch.bool, device=device)


def geometry_state_support(boxes: Sequence[CorridorBox], *, separate_b=False, device=None) -> torch.Tensor:
    first, second = _validate_boxes(boxes)
    membership = geometry_membership_support(first, second, device=device)
    states = state_assignments(separate_b=separate_b, device=device)
    return membership[states[:, -2], states[:, -1]]


def inside_points(points: torch.Tensor, box: CorridorBox) -> torch.Tensor:
    """Producer-equivalent membership for points[...,3] in world ENU metres."""
    if not isinstance(points, torch.Tensor) or points.shape[-1:] != (3,) or not points.is_floating_point():
        raise TypeError("points must be a floating tensor ending in ENU dimension three")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("point coordinates must be finite")
    points = points.double()
    delta = points - points.new_tensor(box.center_enu_m)
    angle = math.radians(float(box.yaw_deg))
    axes = points.new_tensor(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))
    local = torch.cat((delta[..., :2] @ axes.T, delta[..., 2:]), -1)
    return (local.abs() <= points.new_tensor(box.half_extent_with_tolerance)).all(-1)


def signed_distance_points(points: torch.Tensor, box: CorridorBox) -> torch.Tensor:
    """Signed Euclidean distance to the tolerance-expanded closed box surface."""
    if not isinstance(points, torch.Tensor) or points.shape[-1:] != (3,) or not points.is_floating_point():
        raise TypeError("points must be a floating tensor ending in ENU dimension three")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("point coordinates must be finite")
    points = points.double()
    delta = points - points.new_tensor(box.center_enu_m)
    angle = math.radians(float(box.yaw_deg))
    axes = points.new_tensor(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))
    local_xy = delta[..., :2] @ axes.T
    local = torch.cat((local_xy, delta[..., 2:]), -1)
    q = local.abs() - points.new_tensor(box.half_extent_with_tolerance)
    return torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(-1).clamp_max(0)


def build_corridor_support(
    boxes: Sequence[CorridorBox], anchor_latent: torch.Tensor, *, branches: int,
    time_steps: int = 11, separate_b: bool = False,
) -> torch.Tensor:
    """Static geometry everywhere; real-prefix anchor is the sole truth clamp.

    Anchor is discrete [batch,3/4], with -1 explicitly unknown. The function
    accepts no sampled future path and no future truth arrays.
    """
    dim = 4 if separate_b else 3
    if not isinstance(anchor_latent, torch.Tensor) or anchor_latent.ndim != 2 or anchor_latent.shape[-1] != dim:
        raise ValueError(f"anchor_latent must have shape [B,{dim}]")
    if anchor_latent.is_floating_point() or anchor_latent.is_complex():
        raise TypeError("anchor latent values must be discrete")
    if not bool(((anchor_latent == -1) | (anchor_latent == 0) | (anchor_latent == 1)).all()):
        raise ValueError("anchor values must be 0/1, with -1 explicitly unknown")
    if type(branches) is not int or branches < 1 or type(time_steps) is not int or not 1 <= time_steps <= 11:
        raise ValueError("positive branch count and 1..11 time steps are required")
    states = state_assignments(separate_b=separate_b, device=anchor_latent.device)
    geometry = geometry_state_support(boxes, separate_b=separate_b, device=anchor_latent.device)
    anchor = ((anchor_latent[:, None] < 0) | (states[None] == anchor_latent[:, None])).all(-1) & geometry
    if not bool(anchor.any(-1).all()):
        raise ValueError("observed prefix anchor conflicts with static corridor geometry")
    future = geometry[None, None, None].expand(anchor_latent.shape[0], branches, time_steps-1, -1)
    return torch.cat((anchor[:, None, None].expand(-1, branches, 1, -1), future), 2)


def latent_state_keys(*, episode_id: str, actor_id: str, corridor_ids: Sequence[str], tick: int, branch_id: str):
    """Typed owners: one actor B and two actor|corridor I states."""
    if len(corridor_ids) != 2 or corridor_ids[0] == corridor_ids[1]:
        raise ValueError("two distinct corridor IDs are required")
    return (StateKey(actor_id, "B", episode_id, tick, branch_id),
            *(StateKey(f"{actor_id}|{corridor_id}", "I", episode_id, tick, branch_id) for corridor_id in corridor_ids))


def contradiction_states(*, separate_b=False, device=None):
    """Both-inside states asserting two different speed regimes for one actor."""
    states = state_assignments(separate_b=separate_b, device=device)
    if not separate_b:
        return torch.zeros(states.shape[0], dtype=torch.bool, device=device)
    return (states[:, 0] != states[:, 1]) & states[:, 2].bool() & states[:, 3].bool()


def speed_conflict_states(*, separate_b=False, device=None):
    """All latent assignments giving the same actor two distinct B values."""
    states = state_assignments(separate_b=separate_b, device=device)
    if not separate_b:
        return torch.zeros(states.shape[0], dtype=torch.bool, device=device)
    return states[:, 0] != states[:, 1]
