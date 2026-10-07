"""Shared executable camera-footprint contract for fixed-altitude inspection.

The scenario generator, source validator, and render-ready converter must use
the same geometry.  A route observes a boundary only when every polygon corner
and its center lies in the oriented ground footprint of at least one route
segment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


MIN_DOWNWARD_PITCH_DEG = -60.0


def camera_footprint_half_extents_m(
    sensor_profile: Mapping[str, Any],
    altitude_m: float,
) -> tuple[float, float]:
    """Return horizontal and vertical half-extents of the nadir footprint."""

    hfov_deg = float(
        sensor_profile.get("hfov_deg")
        or sensor_profile.get("FOV_Degrees")
        or sensor_profile.get("fov_degrees")
        or 0.0
    )
    width = float(sensor_profile.get("width") or 0.0)
    height = float(sensor_profile.get("height") or 0.0)
    rotation = dict(sensor_profile.get("fixed_rotation_offset_deg") or {})
    pitch_deg = float(rotation.get("pitch_deg", 0.0))
    altitude = float(altitude_m)
    if (
        hfov_deg <= 0.0
        or hfov_deg >= 180.0
        or width <= 0.0
        or height <= 0.0
        or altitude <= 0.0
        or pitch_deg > MIN_DOWNWARD_PITCH_DEG
    ):
        raise ValueError(
            "inspect sensor contract requires positive altitude/resolution, "
            "0<hfov<180, and a downward pitch of at most -60 degrees"
        )
    half_width_m = math.tan(math.radians(hfov_deg * 0.5)) * altitude
    half_height_m = half_width_m * height / width
    return half_width_m, half_height_m


def boundary_observation_samples(
    polygon: Sequence[Sequence[float]],
) -> list[list[float]]:
    if len(polygon) < 3:
        raise ValueError("capture polygon must contain at least three points")
    samples = [[float(point[0]), float(point[1])] for point in polygon]
    samples.append(
        [
            sum(point[0] for point in samples) / len(samples),
            sum(point[1] for point in samples) / len(samples),
        ]
    )
    return samples


def point_in_oriented_frustum_footprint_xy(
    point: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    half_width_m: float,
    half_height_m: float,
) -> bool:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(point[0]), float(point[1])
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return abs(px - ax) <= half_width_m and abs(py - ay) <= half_height_m
    ux = dx / length
    uy = dy / length
    rel_x = px - ax
    rel_y = py - ay
    along = rel_x * ux + rel_y * uy
    cross = abs(-rel_x * uy + rel_y * ux)
    return (
        -half_height_m <= along <= length + half_height_m
        and cross <= half_width_m
    )


def route_observes_samples(
    route: Sequence[Sequence[float]],
    samples: Sequence[Sequence[float]],
    sensor_profile: Mapping[str, Any],
    altitude_m: float,
) -> bool:
    if len(route) < 2 or not samples:
        return False
    try:
        half_width_m, half_height_m = camera_footprint_half_extents_m(
            sensor_profile,
            altitude_m,
        )
    except ValueError:
        return False
    segments = list(zip(route, route[1:]))
    return all(
        any(
            point_in_oriented_frustum_footprint_xy(
                sample,
                a,
                b,
                half_width_m,
                half_height_m,
            )
            for a, b in segments
        )
        for sample in samples
    )


def route_observes_boundary(
    route: Sequence[Sequence[float]],
    polygon: Sequence[Sequence[float]],
    inspect_contract: Mapping[str, Any],
) -> bool:
    if not route or not polygon:
        return False
    profile = dict(inspect_contract.get("sensor_profile") or {})
    altitude_m = float(
        inspect_contract.get("inspect_altitude_m")
        or (route[0][2] if len(route[0]) >= 3 else 0.0)
    )
    try:
        samples = boundary_observation_samples(polygon)
    except ValueError:
        return False
    return route_observes_samples(route, samples, profile, altitude_m)
