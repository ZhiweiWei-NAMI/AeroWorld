"""Mechanism-level observations derived from authoritative truth state.

This module is the narrow state projection layer used before semantic event
rules.  It converts per-tick truth-frame state into stable, traceable
observations without reading scheduled events, event traces, dynamic labels, or
authored semantic roles as facts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .model import TickContext
from .provenance import stable_identifier


GROUND_Z_THRESHOLD_M = 1.5
UAV_PAD_HOVER_Z_THRESHOLD_M = 4.0
VERTICAL_RATE_THRESHOLD_MPS = 0.2
STATIONARY_SPEED_THRESHOLD_MPS = 0.2
TTC_CLOSING_EPSILON_MPS = 1e-6

ENTITY_CATEGORIES_FOR_PAIRS = {"uav", "vehicle", "pedestrian"}

FORBIDDEN_FACT_KEYS = {
    "active_event_ids",
    "dynamic_labels",
    "event_trace",
    "event_realization",
    "scenario_plan",
    "semantic_role",
    "state_sequence",
}

WEATHER_ROOT_KEYS = {
    "weather",
    "weather_state",
    "environment",
    "environment_state",
    "environmental_state",
    "meteorology",
    "atmosphere",
}


def build_mechanism_observations(
    contexts: Sequence[TickContext],
    *,
    frames_by_tick: Mapping[int, Mapping[str, Any]] | None = None,
    include_pairs: bool = True,
    include_weather: bool = True,
    pair_candidate_radius_m: float | None = 120.0,
) -> list[dict[str, Any]]:
    """Build deterministic mechanism observations for a sequence of ticks."""

    rows: list[dict[str, Any]] = []
    for context in contexts:
        frame = frames_by_tick.get(context.tick) if frames_by_tick is not None else None
        rows.extend(
            observe_tick(
                context,
                frame=frame,
                include_pairs=include_pairs,
                pair_candidate_radius_m=pair_candidate_radius_m,
            )
        )
        if include_weather:
            rows.extend(observe_weather(context, frame=frame))
    rows.sort(key=_sort_key)
    return rows


def observe_tick(
    context: TickContext,
    *,
    frame: Mapping[str, Any] | None = None,
    include_pairs: bool = True,
    pair_candidate_radius_m: float | None = 120.0,
) -> list[dict[str, Any]]:
    """Build all per-entity and optional pair observations for one tick."""

    rows: list[dict[str, Any]] = []
    # Per-tick observables apply only to entities physically represented at
    # that tick, plus explicit static geometry.  ``context.entities`` also
    # contains the episode-wide roster; expanding every roster member at every
    # tick creates unknown rows for vehicles that have not entered (or have
    # already left) the captured world.
    entity_ids = sorted(set(context.frame_entities) | set(context.static_entities))
    entity_observations: dict[str, dict[str, Any]] = {}
    for entity_id in entity_ids:
        entity = context.entities[entity_id]
        rows.extend(_observe_entity(context, entity_id, entity))
        entity_observations[entity_id] = {
            "position": _vector_at(entity, ("truth_pose", "position_enu_m"), 3),
            "velocity": _vector_at(entity, ("truth_pose", "velocity_enu_mps"), 3),
            "category": _entity_category(entity),
        }
    rows.extend(_observe_nearest_pad_geometry(context, entity_observations))
    rows.extend(_observe_airspace_and_structure_geometry(context, entity_observations))
    if include_pairs:
        pair_ids = [
            entity_id
            for entity_id, values in sorted(entity_observations.items())
            if values["category"] in ENTITY_CATEGORIES_FOR_PAIRS
            and entity_id in context.frame_entities
        ]
        for index, first_id in enumerate(pair_ids):
            for second_id in pair_ids[index + 1 :]:
                if not _pair_is_candidate(
                    entity_observations[first_id],
                    entity_observations[second_id],
                    pair_candidate_radius_m,
                ):
                    continue
                rows.extend(
                    _observe_pair(
                        context,
                        frame,
                        first_id,
                        second_id,
                        entity_observations[first_id],
                        entity_observations[second_id],
                    )
                )
    rows.sort(key=_sort_key)
    return rows


def _observe_nearest_pad_geometry(
    context: TickContext,
    entity_observations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure UAV position relative to the nearest physical pad.

    The UAV pose origin in the current AeroWorld truth export can remain several
    metres above the pad surface while the vehicle is visibly grounded.  This
    layer therefore records the raw relative altitude and leaves the governed
    tolerance decision to the predicate layer; it never treats absolute world Z
    as pad membership.
    """

    pads: list[tuple[str, tuple[float, float, float]]] = []
    for entity_id, values in sorted(entity_observations.items()):
        entity = context.entities.get(entity_id, {})
        category = str(values.get("category") or "").lower()
        position = values.get("position")
        if position is None:
            continue
        semantic_scope = entity.get("semantic_scope")
        if (
            category == "facility"
            and isinstance(semantic_scope, Mapping)
            and semantic_scope.get("scope_type") == "facility"
            and semantic_scope.get("scope_subtype") == "landing_pad"
        ):
            pads.append((entity_id, position))

    rows: list[dict[str, Any]] = []
    for uav_id, values in sorted(entity_observations.items()):
        if values.get("category") != "uav":
            continue
        uav_position = values.get("position")
        if uav_position is None:
            reason = "missing_uav_position_for_pad_geometry"
            rows.extend(
                _unknown_entity_observation(context, metric, unit, uav_id, reason)
                for metric, unit in (
                    ("uav.nearest_pad_id", "id"),
                    ("uav.pad_distance_xy_m", "m"),
                    ("uav.pad_relative_altitude_m", "m"),
                    ("uav.pad_distance_3d_m", "m"),
                )
            )
            continue
        if not pads:
            reason = "missing_physical_pad_geometry"
            rows.extend(
                _unknown_entity_observation(context, metric, unit, uav_id, reason)
                for metric, unit in (
                    ("uav.nearest_pad_id", "id"),
                    ("uav.pad_distance_xy_m", "m"),
                    ("uav.pad_relative_altitude_m", "m"),
                    ("uav.pad_distance_3d_m", "m"),
                )
            )
            continue

        pad_id, pad_position = min(
            pads,
            key=lambda item: (
                math.hypot(uav_position[0] - item[1][0], uav_position[1] - item[1][1]),
                item[0],
            ),
        )
        dx = uav_position[0] - pad_position[0]
        dy = uav_position[1] - pad_position[1]
        dz = uav_position[2] - pad_position[2]
        refs = (
            f"{_entity_ref(context, uav_id)}.truth_pose.position_enu_m",
            f"{_entity_ref(context, pad_id)}.truth_pose.position_enu_m",
        )
        for metric_id, value, unit in (
            ("uav.nearest_pad_id", pad_id, "id"),
            ("uav.pad_distance_xy_m", math.hypot(dx, dy), "m"),
            ("uav.pad_relative_altitude_m", dz, "m"),
            ("uav.pad_distance_3d_m", math.sqrt(dx * dx + dy * dy + dz * dz), "m"),
        ):
            rows.append(
                _observation(
                    context,
                    metric_id,
                    value,
                    unit=unit,
                    subject_id=uav_id,
                    object_id=pad_id,
                    source_class="derived_from_observed",
                    source_refs=refs,
                    quality="derived",
                )
            )
    return rows


def _pair_is_candidate(
    first_values: Mapping[str, Any],
    second_values: Mapping[str, Any],
    radius_m: float | None,
) -> bool:
    """Keep only physically plausible interaction pairs.

    Pair absence outside the governed candidate radius means out of candidate
    scope, not an assertion that every distant pair has a known false relation.
    This prevents quadratic truth-store growth from background traffic.
    """

    first_category = first_values.get("category")
    second_category = second_values.get("category")
    categories = frozenset((first_category, second_category))
    if categories not in {
        frozenset(("uav",)),
        frozenset(("vehicle",)),
        frozenset(("uav", "vehicle")),
        frozenset(("uav", "pedestrian")),
        frozenset(("vehicle", "pedestrian")),
    }:
        return False
    first_position = first_values.get("position")
    second_position = second_values.get("position")
    if first_position is None or second_position is None:
        return False
    if radius_m is None:
        return True
    if radius_m <= 0:
        raise ValueError("pair_candidate_radius_m must be positive or None")
    return math.dist(first_position[:3], second_position[:3]) <= radius_m


def _observe_airspace_and_structure_geometry(
    context: TickContext,
    entity_observations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure UAV relationships to explicit regions, corridors, and facades."""

    uavs = [
        (entity_id, values)
        for entity_id, values in sorted(entity_observations.items())
        if values.get("category") == "uav" and values.get("position") is not None
    ]
    rows: list[dict[str, Any]] = []
    corridor_occupants: dict[str, list[str]] = {}
    corridor_network_occupants: set[str] = set()
    for object_id, entity in sorted(context.entities.items()):
        geometry = entity.get("geometry")
        if not isinstance(geometry, Mapping):
            continue
        geometry_ref = f"{context.static_name}#entity={object_id}.geometry"
        constraint_kind = str(geometry.get("constraint_kind") or "")
        logical_asset_id = str(entity.get("logical_asset_id") or "")
        entity_category = str(entity.get("category") or entity.get("entity_category") or "")
        is_region = (
            entity_category == "airspace_constraint"
            or constraint_kind.startswith("airspace.")
            or logical_asset_id == "trigger.no_fly.box.v1"
        )
        is_corridor = entity_category == "airspace_corridor"
        is_facade = geometry.get("geometry_kind") == "facade_anchor"
        configured_active_value = geometry.get("active")
        configured_active = (
            configured_active_value
            if isinstance(configured_active_value, bool)
            else None
        )
        active: bool | None = configured_active
        activity_source_refs: tuple[str, ...] = (f"{geometry_ref}.active",)
        activity_source_class = "derived_from_observed"
        region_unknown_reason: str | None = None
        if is_region:
            # The strict objective input intentionally removes authored
            # activity labels.  Region activation therefore uses an explicit
            # physical state token when one survives, otherwise it falls back
            # to the configured activation/deactivation schedule embedded in
            # the tick-specific geometry.  Same-tick entity presence is still
            # required, so static geometry alone cannot assert an active
            # region.  This keeps permanent geofences observable without
            # treating an animation label such as ``idle`` as semantic truth.
            active = None
            frame_ref = f"{context.truth_frames_name}#tick={context.tick}&entity={object_id}"
            activity_source_refs = (f"{geometry_ref}.active", frame_ref)
            frame_entity = context.frame_entities.get(object_id)
            if frame_entity is None:
                region_unknown_reason = "missing_same_tick_region_entity"
            else:
                constraint_state = frame_entity.get("constraint_state")
                structured_active = (
                    constraint_state.get("active")
                    if isinstance(constraint_state, Mapping)
                    else None
                )
                observed_state = str(frame_entity.get("state") or "").strip().lower()
                if isinstance(structured_active, bool):
                    active = structured_active
                    activity_source_refs = (f"{frame_ref}.constraint_state.active",)
                    activity_source_class = "observed_simulator_truth"
                elif observed_state in {"active", "deployed", "lockdown", "isolating"}:
                    active = True
                    activity_source_refs = (f"{frame_ref}.state",)
                    activity_source_class = "observed_simulator_truth"
                elif observed_state in {
                    "offstage",
                    "standdown",
                    "inactive",
                    "removed",
                    "cleared",
                }:
                    active = False
                    activity_source_refs = (f"{frame_ref}.state",)
                    activity_source_class = "observed_simulator_truth"
                else:
                    active = configured_active
                    if configured_active is None:
                        region_unknown_reason = "missing_configured_region_activation"

        if is_region:
            rows.append(
                _observation(
                    context,
                    "airspace.region_active",
                    active,
                    unit="bool",
                    subject_id=object_id,
                    source_class=activity_source_class,
                    source_refs=activity_source_refs,
                    quality=(
                        "observed"
                        if active is not None and activity_source_class == "observed_simulator_truth"
                        else "derived"
                        if active is not None
                        else "unknown"
                    ),
                    unknown_reason=region_unknown_reason,
                )
            )
        if (is_region or is_corridor) and active:
            polygon = _polygon_xy(geometry.get("polygon_enu_m"))
            base_z = _numeric_at(geometry, ("base_z_m",))
            height = _numeric_at(geometry, ("height_m",))
            if polygon is None or base_z is None or height is None:
                continue
            corridor_occupants.setdefault(object_id, [])
            for uav_id, values in uavs:
                position = values["position"]
                velocity = values.get("velocity")
                inside_xy = _point_in_polygon(position[0], position[1], polygon)
                inside_z = base_z <= position[2] <= base_z + height
                boundary_distance, boundary_point = _distance_to_polygon_boundary(
                    position[0], position[1], polygon
                )
                signed_distance = -boundary_distance if inside_xy and inside_z else boundary_distance
                pair_id = f"{uav_id}|{object_id}"
                refs = (
                    f"{_entity_ref(context, uav_id)}.truth_pose.position_enu_m",
                    geometry_ref,
                )
                if is_region:
                    if velocity is None or boundary_distance <= 1e-9:
                        closing_speed: float | None = None
                    else:
                        toward_x = (boundary_point[0] - position[0]) / boundary_distance
                        toward_y = (boundary_point[1] - position[1]) / boundary_distance
                        closing_speed = velocity[0] * toward_x + velocity[1] * toward_y
                    for metric_id, value, unit in (
                        ("airspace.distance_to_boundary_m", boundary_distance, "m"),
                        ("airspace.signed_distance_to_boundary_m", signed_distance, "m"),
                        ("airspace.boundary_closing_speed_mps", closing_speed, "m/s"),
                        ("region.inside_active_restricted_region", inside_xy and inside_z, "bool"),
                    ):
                        rows.append(
                            _observation(
                                context,
                                metric_id,
                                value,
                                unit=unit,
                                subject_id=pair_id,
                                object_id=object_id,
                                source_class="derived_from_observed",
                                source_refs=refs,
                                quality="derived" if value is not None else "unknown",
                                unknown_reason=(
                                    "missing_uav_velocity_for_boundary_closing_speed"
                                    if value is None
                                    else None
                                ),
                            )
                        )
                if is_corridor and inside_xy and inside_z:
                    corridor_occupants[object_id].append(uav_id)
                    corridor_network_occupants.add(uav_id)

        if is_facade and active:
            point = _vector_from_value(geometry.get("point_enu_m"), 3)
            normal = _vector_from_value(geometry.get("outward_normal_enu"), 3)
            if point is None or normal is None:
                continue
            stand_off = _numeric_at(geometry, ("stand_off_m",)) or 0.0
            wall_point = tuple(point[index] - normal[index] * stand_off for index in range(3))
            for uav_id, values in uavs:
                position = values["position"]
                velocity = values.get("velocity")
                delta = tuple(position[index] - point[index] for index in range(3))
                anchor_distance = _norm(delta)
                wall_delta = tuple(position[index] - wall_point[index] for index in range(3))
                plane_clearance = sum(wall_delta[index] * normal[index] for index in range(3))
                if velocity is None or anchor_distance <= 1e-9:
                    closing_speed = None
                    ttc = None
                else:
                    closing_speed = -sum(delta[index] * velocity[index] for index in range(3)) / anchor_distance
                    ttc = anchor_distance / closing_speed if closing_speed > TTC_CLOSING_EPSILON_MPS else None
                pair_id = f"{uav_id}|{object_id}"
                refs = (
                    f"{_entity_ref(context, uav_id)}.truth_pose.position_enu_m|truth_pose.velocity_enu_mps",
                    geometry_ref,
                )
                for metric_id, value, unit, reason in (
                    ("structure.anchor_distance_3d_m", anchor_distance, "m", None),
                    ("structure.signed_plane_clearance_m", plane_clearance, "m", None),
                    (
                        "structure.relative_closing_speed_mps",
                        closing_speed,
                        "m/s",
                        "missing_uav_velocity_for_structure_closing_speed",
                    ),
                    (
                        "structure.time_to_conflict_s",
                        ttc,
                        "s",
                        "structure_not_closing_or_velocity_missing",
                    ),
                ):
                    rows.append(
                        _observation(
                            context,
                            metric_id,
                            value,
                            unit=unit,
                            subject_id=pair_id,
                            object_id=object_id,
                            source_class="derived_from_observed",
                            source_refs=refs,
                            quality="derived" if value is not None else "not_applicable",
                            unknown_reason=reason if value is None else None,
                        )
                    )

    for corridor_id, occupants in sorted(corridor_occupants.items()):
        rows.append(
            _observation(
                context,
                "corridor.occupancy_count",
                len(occupants),
                unit="count",
                subject_id=corridor_id,
                source_class="derived_from_observed",
                source_refs=(
                    f"{context.static_name}#entity={corridor_id}.geometry",
                    *(
                        f"{_entity_ref(context, uav_id)}.truth_pose.position_enu_m"
                        for uav_id in sorted(occupants)
                    ),
                ),
                quality="derived",
            )
        )
        rows.append(
            _observation(
                context,
                "corridor.occupant_ids",
                sorted(occupants),
                unit="ids",
                subject_id=corridor_id,
                source_class="derived_from_observed",
                source_refs=(f"{context.static_name}#entity={corridor_id}.geometry",),
                quality="derived",
            )
        )
    if corridor_occupants:
        aggregate_refs = tuple(
            f"{context.static_name}#entity={corridor_id}.geometry"
            for corridor_id in sorted(corridor_occupants)
        )
        rows.append(
            _observation(
                context,
                "corridor.occupancy_count",
                len(corridor_network_occupants),
                unit="count",
                subject_id="corridor_network",
                source_class="derived_from_observed",
                source_refs=aggregate_refs,
                quality="derived",
            )
        )
        rows.append(
            _observation(
                context,
                "corridor.occupant_ids",
                sorted(corridor_network_occupants),
                unit="ids",
                subject_id="corridor_network",
                source_class="derived_from_observed",
                source_refs=aggregate_refs,
                quality="derived",
            )
        )
    return rows


def _polygon_xy(value: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    points: list[tuple[float, float]] = []
    for point in value:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) < 2
            or not isinstance(point[0], (int, float))
            or not isinstance(point[1], (int, float))
        ):
            return None
        points.append((float(point[0]), float(point[1])))
    return tuple(points)


def _vector_from_value(value: Any, size: int) -> tuple[float, ...] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < size
        or not all(isinstance(item, (int, float)) for item in value[:size])
    ):
        return None
    return tuple(float(item) for item in value[:size])


def _point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            crossing_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _distance_to_polygon_boundary(
    x: float,
    y: float,
    polygon: Sequence[tuple[float, float]],
) -> tuple[float, tuple[float, float]]:
    best_distance = math.inf
    best_point = polygon[0]
    previous = polygon[-1]
    for current in polygon:
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 0.0:
            candidate = previous
        else:
            fraction = max(0.0, min(1.0, ((x - previous[0]) * dx + (y - previous[1]) * dy) / length_squared))
            candidate = (previous[0] + fraction * dx, previous[1] + fraction * dy)
        distance = math.hypot(x - candidate[0], y - candidate[1])
        if distance < best_distance:
            best_distance = distance
            best_point = candidate
        previous = current
    return best_distance, best_point


def observe_weather(
    context: TickContext,
    *,
    frame: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract numeric weather/environment observations from the truth frame."""

    if not context.tick_present:
        return [
            _observation(
                context,
                "weather.present",
                False,
                unit="bool",
                subject_id="environment",
                source_class="unknown",
                source_refs=(f"{context.truth_frames_name}#tick={context.tick}",),
                quality="unknown",
                unknown_reason="missing_authoritative_tick",
            )
        ]
    frame = frame or {}
    weather_source_by_root = frame.get("__weather_source_by_root")
    if not isinstance(weather_source_by_root, Mapping):
        weather_source_by_root = {}
    rows: list[dict[str, Any]] = []
    for root_key in sorted(WEATHER_ROOT_KEYS):
        root_value = frame.get(root_key)
        if not isinstance(root_value, Mapping):
            continue
        source_name = weather_source_by_root.get(root_key, context.truth_frames_name)
        if not isinstance(source_name, str) or not source_name:
            source_name = context.truth_frames_name
        for path, value in _iter_numeric_leaves(root_value):
            metric = f"weather.{root_key}.{path}"
            rows.append(
                _observation(
                    context,
                    metric,
                    value,
                    unit=_weather_unit(path),
                    subject_id="environment",
                    source_class="observed_simulator_truth",
                    source_refs=(f"{source_name}#tick={context.tick}.{root_key}.{path}",),
                    quality="observed",
                )
            )
    if rows:
        rows.sort(key=_sort_key)
        return rows
    return [
        _observation(
            context,
            "weather.numeric_state_available",
            None,
            unit="bool",
            subject_id="environment",
            source_class="unknown",
            source_refs=(f"{context.truth_frames_name}#tick={context.tick}",),
            quality="unknown",
            unknown_reason="no_numeric_weather_state_in_truth_frame",
        )
    ]


def _observe_entity(
    context: TickContext,
    entity_id: str,
    entity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entity_ref = _entity_ref(context, entity_id)
    category = _entity_category(entity)
    present = context.tick_present and entity_id in context.frame_entities
    rows.append(
        _observation(
            context,
            "entity.present",
            present,
            unit="bool",
            subject_id=entity_id,
            source_class="derived_from_observed",
            source_refs=(entity_ref,),
            quality="derived" if context.tick_present else "unknown",
            unknown_reason=None if context.tick_present else "missing_authoritative_tick",
        )
    )
    rows.append(
        _observation(
            context,
            "entity.category",
            category,
            unit="id",
            subject_id=entity_id,
            source_class="observed_simulator_truth" if category else "unknown",
            source_refs=(entity_ref,),
            quality="observed" if category else "unknown",
            unknown_reason=None if category else "missing_entity_category",
        )
    )
    activity_state = _get(entity, ("annotations", "state_facets", "activity", "activity_type"))
    if not isinstance(activity_state, str) or not activity_state:
        activity_state = _get(entity, ("annotations", "activity_type"))
    rows.append(
        _observation(
            context,
            "entity.activity_state",
            activity_state if isinstance(activity_state, str) and activity_state else None,
            unit="categorical",
            subject_id=entity_id,
            source_class=(
                "observed_simulator_truth"
                if isinstance(activity_state, str) and activity_state
                else "unknown"
            ),
            source_refs=(f"{entity_ref}.annotations.state_facets.activity.activity_type",),
            quality=(
                "observed"
                if isinstance(activity_state, str) and activity_state
                else "unknown"
            ),
            unknown_reason=(
                None
                if isinstance(activity_state, str) and activity_state
                else "missing_activity_state"
            ),
        )
    )

    position = _vector_at(entity, ("truth_pose", "position_enu_m"), 3)
    velocity = _vector_at(entity, ("truth_pose", "velocity_enu_mps"), 3)
    if position is None:
        rows.append(
            _unknown_entity_observation(
                context,
                "kinematics.position_enu_m",
                "m",
                entity_id,
                "missing_truth_pose_position_enu_m",
            )
        )
    else:
        rows.append(
            _observation(
                context,
                "kinematics.position_enu_m",
                list(position),
                unit="m",
                subject_id=entity_id,
                source_class="observed_simulator_truth",
                source_refs=(f"{entity_ref}.truth_pose.position_enu_m",),
                quality="observed",
            )
        )

    if velocity is None:
        rows.append(
            _unknown_entity_observation(
                context,
                "kinematics.velocity_enu_mps",
                "m/s",
                entity_id,
                "missing_truth_pose_velocity_enu_mps",
            )
        )
        rows.append(
            _unknown_entity_observation(
                context,
                "kinematics.speed_mps",
                "m/s",
                entity_id,
                "missing_truth_pose_velocity_enu_mps",
            )
        )
    else:
        speed = _norm(velocity)
        rows.append(
            _observation(
                context,
                "kinematics.velocity_enu_mps",
                list(velocity),
                unit="m/s",
                subject_id=entity_id,
                source_class="observed_simulator_truth",
                source_refs=(f"{entity_ref}.truth_pose.velocity_enu_mps",),
                quality="observed",
            )
        )
        rows.append(
            _observation(
                context,
                "kinematics.speed_mps",
                speed,
                unit="m/s",
                subject_id=entity_id,
                source_class="derived_from_observed",
                source_refs=(f"{entity_ref}.truth_pose.velocity_enu_mps",),
                quality="derived",
            )
        )
        rows.append(
            _observation(
                context,
                "kinematics.stationary",
                speed <= STATIONARY_SPEED_THRESHOLD_MPS,
                unit="bool",
                subject_id=entity_id,
                source_class="derived_from_observed",
                source_refs=(f"{entity_ref}.truth_pose.velocity_enu_mps",),
                quality="derived",
            )
        )

    if position is None:
        rows.extend(
            _unknown_entity_observation(context, metric, "bool", entity_id, "missing_truth_pose_position_enu_m")
            for metric in ("kinematics.grounded", "kinematics.airborne")
        )
    else:
        z_m = position[2]
        ground_threshold_m = (
            UAV_PAD_HOVER_Z_THRESHOLD_M if category == "uav" else GROUND_Z_THRESHOLD_M
        )
        rows.append(
            _observation(
                context,
                "kinematics.altitude_m",
                z_m,
                unit="m",
                subject_id=entity_id,
                source_class="derived_from_observed",
                source_refs=(f"{entity_ref}.truth_pose.position_enu_m",),
                quality="derived",
            )
        )
        rows.append(
            _observation(
                context,
                "kinematics.ground_threshold_m",
                ground_threshold_m,
                unit="m",
                subject_id=entity_id,
                source_class="deterministic_simulation",
                source_refs=(
                    "mechanism_observables.UAV_PAD_HOVER_Z_THRESHOLD_M"
                    if category == "uav"
                    else "mechanism_observables.GROUND_Z_THRESHOLD_M",
                ),
                quality="deterministic_model_parameter",
            )
        )
        grounded = z_m <= ground_threshold_m
        rows.append(
            _observation(
                context,
                "kinematics.grounded",
                grounded,
                unit="bool",
                subject_id=entity_id,
                source_class="derived_from_observed",
                source_refs=(f"{entity_ref}.truth_pose.position_enu_m",),
                quality="derived",
            )
        )
        rows.append(
            _observation(
                context,
                "kinematics.airborne",
                not grounded,
                unit="bool",
                subject_id=entity_id,
                source_class="derived_from_observed",
                source_refs=(f"{entity_ref}.truth_pose.position_enu_m",),
                quality="derived",
            )
        )

    if velocity is None:
        rows.extend(
            _unknown_entity_observation(context, metric, "bool", entity_id, "missing_truth_pose_velocity_enu_mps")
            for metric in ("kinematics.ascending", "kinematics.descending")
        )
    else:
        vz_mps = velocity[2]
        rows.append(
            _observation(
                context,
                "kinematics.vertical_speed_mps",
                vz_mps,
                unit="m/s",
                subject_id=entity_id,
                source_class="derived_from_observed",
                source_refs=(f"{entity_ref}.truth_pose.velocity_enu_mps",),
                quality="derived",
            )
        )
        rows.append(
            _observation(
                context,
                "kinematics.ascending",
                vz_mps > VERTICAL_RATE_THRESHOLD_MPS,
                unit="bool",
                subject_id=entity_id,
                source_class="derived_from_observed",
                source_refs=(f"{entity_ref}.truth_pose.velocity_enu_mps",),
                quality="derived",
            )
        )
        rows.append(
            _observation(
                context,
                "kinematics.descending",
                vz_mps < -VERTICAL_RATE_THRESHOLD_MPS,
                unit="bool",
                subject_id=entity_id,
                source_class="derived_from_observed",
                source_refs=(f"{entity_ref}.truth_pose.velocity_enu_mps",),
                quality="derived",
            )
        )

    if category == "pedestrian":
        rows.extend(_observe_pedestrian(context, entity_id, entity))
    elif category == "vehicle":
        rows.extend(_observe_vehicle(context, entity_id, entity))
    elif category == "uav":
        rows.extend(_observe_uav(context, entity_id, entity, position))
    return rows


def _observe_pedestrian(
    context: TickContext,
    entity_id: str,
    entity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entity_ref = _entity_ref(context, entity_id)
    rows: list[dict[str, Any]] = []
    activity = _get(entity, ("annotations", "state_facets", "activity", "activity_type"))
    if not isinstance(activity, str) or not activity:
        activity = _get(entity, ("annotations", "activity_type"))
    posture = _get(entity, ("annotations", "state_facets", "activity", "posture"))
    for metric, value, path in (
        ("pedestrian.activity", activity, "annotations.state_facets.activity.activity_type"),
        ("pedestrian.posture", posture, "annotations.state_facets.activity.posture"),
    ):
        if isinstance(value, str) and value:
            rows.append(
                _observation(
                    context,
                    metric,
                    value,
                    unit="categorical",
                    subject_id=entity_id,
                    source_class="observed_simulator_truth",
                    source_refs=(f"{entity_ref}.{path}",),
                    quality="observed",
                )
            )
        else:
            rows.append(
                _unknown_entity_observation(
                    context,
                    metric,
                    "categorical",
                    entity_id,
                    f"missing_{metric.replace('.', '_')}",
                )
            )
    return rows


def _observe_vehicle(
    context: TickContext,
    entity_id: str,
    entity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entity_ref = _entity_ref(context, entity_id)
    rows: list[dict[str, Any]] = []
    lane_id = _get(entity, ("sumo_vehicle", "sumo_lane_id"))
    if isinstance(lane_id, str) and lane_id:
        rows.append(
            _observation(
                context,
                "vehicle.sumo_lane_id",
                lane_id,
                unit="id",
                subject_id=entity_id,
                source_class="observed_simulator_truth",
                source_refs=(f"{entity_ref}.sumo_vehicle.sumo_lane_id",),
                quality="observed",
            )
        )
    else:
        rows.append(
            _unknown_entity_observation(
                context,
                "vehicle.sumo_lane_id",
                "id",
                entity_id,
                "missing_sumo_vehicle_sumo_lane_id",
            )
        )
    speed = _numeric_at(entity, ("sumo_vehicle", "speed_mps"))
    if speed is None:
        rows.append(
            _unknown_entity_observation(
                context,
                "vehicle.sumo_speed_mps",
                "m/s",
                entity_id,
                "missing_sumo_vehicle_speed_mps",
            )
        )
    else:
        rows.append(
            _observation(
                context,
                "vehicle.sumo_speed_mps",
                speed,
                unit="m/s",
                subject_id=entity_id,
                source_class="observed_simulator_truth",
                source_refs=(f"{entity_ref}.sumo_vehicle.speed_mps",),
                quality="observed",
            )
        )
    return rows


def _observe_uav(
    context: TickContext,
    entity_id: str,
    entity: Mapping[str, Any],
    position: tuple[float, float, float] | None,
) -> list[dict[str, Any]]:
    entity_ref = _entity_ref(context, entity_id)
    rows: list[dict[str, Any]] = []
    corridor_id = _get(entity, ("uav_global_flow", "corridor_id"))
    if isinstance(corridor_id, str) and corridor_id:
        rows.append(
            _observation(
                context,
                "uav.corridor_id",
                corridor_id,
                unit="id",
                subject_id=entity_id,
                source_class="observed_simulator_truth",
                source_refs=(f"{entity_ref}.uav_global_flow.corridor_id",),
                quality="observed",
            )
        )
    else:
        rows.append(
            _unknown_entity_observation(
                context,
                "uav.corridor_id",
                "id",
                entity_id,
                "missing_uav_global_flow_corridor_id",
            )
        )
    assigned_altitude, assigned_altitude_path = _first_numeric_with_path(
        entity,
        (
            ("uav_global_flow", "altitude_layer_m"),
            ("uav_corridor", "assigned_altitude_m"),
            ("assigned_altitude_m",),
        ),
    )
    if assigned_altitude is None:
        rows.append(
            _unknown_entity_observation(
                context,
                "uav.assigned_altitude_m",
                "m",
                entity_id,
                "missing_uav_assigned_altitude",
            )
        )
    else:
        rows.append(
            _observation(
                context,
                "uav.assigned_altitude_m",
                assigned_altitude,
                unit="m",
                subject_id=entity_id,
                    source_class="observed_simulator_truth",
                    source_refs=(f"{entity_ref}.{'.'.join(assigned_altitude_path)}",),
                    quality="observed",
                )
            )
        if position is not None:
            rows.append(
                _observation(
                    context,
                    "uav.assigned_altitude_error_m",
                    position[2] - assigned_altitude,
                    unit="m",
                    subject_id=entity_id,
                    source_class="derived_from_observed",
                    source_refs=(
                        f"{entity_ref}.truth_pose.position_enu_m",
                        f"{entity_ref}.{'.'.join(assigned_altitude_path)}",
                    ),
                    quality="derived",
                )
            )
    return rows


def _observe_pair(
    context: TickContext,
    frame: Mapping[str, Any] | None,
    first_id: str,
    second_id: str,
    first_values: Mapping[str, Any],
    second_values: Mapping[str, Any],
) -> list[dict[str, Any]]:
    first_position = first_values["position"]
    second_position = second_values["position"]
    first_velocity = first_values["velocity"]
    second_velocity = second_values["velocity"]
    pair_id = f"{first_id}|{second_id}"
    pair_refs = (_entity_ref(context, first_id), _entity_ref(context, second_id))
    if first_position is None or second_position is None:
        return [
            _unknown_pair_observation(context, metric, unit, pair_id, "missing_pair_position")
            for metric, unit in (
                ("pair.xy_distance_m", "m"),
                ("pair.xyz_distance_m", "m"),
                ("pair.relative_closing_speed_mps", "m/s"),
                ("pair.ttc_s", "s"),
                ("pair.collision_contact", "bool"),
            )
        ]

    delta = tuple(second_position[index] - first_position[index] for index in range(3))
    xy_distance = math.hypot(delta[0], delta[1])
    xyz_distance = _norm(delta)
    rows = [
        _observation(
            context,
            "pair.xy_distance_m",
            xy_distance,
            unit="m",
            subject_id=pair_id,
            object_id=second_id,
            source_class="derived_from_observed",
            source_refs=tuple(f"{ref}.truth_pose.position_enu_m" for ref in pair_refs),
            quality="derived",
        ),
        _observation(
            context,
            "pair.xyz_distance_m",
            xyz_distance,
            unit="m",
            subject_id=pair_id,
            object_id=second_id,
            source_class="derived_from_observed",
            source_refs=tuple(f"{ref}.truth_pose.position_enu_m" for ref in pair_refs),
            quality="derived",
        ),
        _observation(
            context,
            "pair.horizontal_separation_m",
            xy_distance,
            unit="m",
            subject_id=pair_id,
            object_id=second_id,
            source_class="derived_from_observed",
            source_refs=tuple(f"{ref}.truth_pose.position_enu_m" for ref in pair_refs),
            quality="derived",
        ),
        _observation(
            context,
            "pair.vertical_separation_m",
            abs(delta[2]),
            unit="m",
            subject_id=pair_id,
            object_id=second_id,
            source_class="derived_from_observed",
            source_refs=tuple(f"{ref}.truth_pose.position_enu_m" for ref in pair_refs),
            quality="derived",
        ),
    ]
    if first_velocity is None or second_velocity is None:
        rows.append(
            _unknown_pair_observation(
                context,
                "pair.relative_closing_speed_mps",
                "m/s",
                pair_id,
                "missing_pair_velocity",
            )
        )
        rows.append(
            _unknown_pair_observation(
                context,
                "pair.ttc_s",
                "s",
                pair_id,
                "missing_pair_velocity",
            )
        )
    else:
        relative_velocity = tuple(second_velocity[index] - first_velocity[index] for index in range(3))
        closing_speed = 0.0
        if xyz_distance > 0.0:
            closing_speed = -sum(delta[index] * relative_velocity[index] for index in range(3)) / xyz_distance
        rows.append(
            _observation(
                context,
                "pair.relative_closing_speed_mps",
                closing_speed,
                unit="m/s",
                subject_id=pair_id,
                object_id=second_id,
                source_class="derived_from_observed",
                source_refs=tuple(
                    ref + ".truth_pose.position_enu_m|truth_pose.velocity_enu_mps"
                    for ref in pair_refs
                ),
                quality="derived",
            )
        )
        if closing_speed > TTC_CLOSING_EPSILON_MPS:
            ttc_value: float | None = xyz_distance / closing_speed
            ttc_quality = "derived"
            unknown_reason = None
        else:
            ttc_value = None
            ttc_quality = "not_applicable"
            unknown_reason = "pair_not_closing"
        rows.append(
            _observation(
                context,
                "pair.ttc_s",
                ttc_value,
                unit="s",
                subject_id=pair_id,
                object_id=second_id,
                source_class="derived_from_observed",
                source_refs=tuple(
                    ref + ".truth_pose.position_enu_m|truth_pose.velocity_enu_mps"
                    for ref in pair_refs
                ),
                quality=ttc_quality,
                unknown_reason=unknown_reason,
            )
        )

    contact_value, contact_refs = _contact_evidence(context, frame, first_id, second_id)
    if contact_value is None:
        rows.append(
            _unknown_pair_observation(
                context,
                "pair.collision_contact",
                "bool",
                pair_id,
                "missing_authoritative_contact_or_extent_evidence",
            )
        )
    else:
        rows.append(
            _observation(
                context,
                "pair.collision_contact",
                contact_value,
                unit="bool",
                subject_id=pair_id,
                object_id=second_id,
                source_class="observed_simulator_truth",
                source_refs=tuple(contact_refs),
                quality="observed",
            )
        )
    return rows


def _contact_evidence(
    context: TickContext,
    frame: Mapping[str, Any] | None,
    first_id: str,
    second_id: str,
) -> tuple[bool | None, tuple[str, ...]]:
    frame = frame or {}
    candidates = (
        frame.get("contacts"),
        frame.get("collisions"),
        frame.get("contact_pairs"),
        frame.get("collision_pairs"),
    )
    unordered = {first_id, second_id}
    for collection_index, collection in enumerate(candidates):
        if not isinstance(collection, list):
            continue
        for item_index, item in enumerate(collection):
            if not isinstance(item, Mapping):
                continue
            ids = _pair_ids(item)
            if ids == unordered:
                value = item.get("contact", item.get("colliding", item.get("collision", True)))
                if isinstance(value, bool):
                    return (
                        value,
                        (
                            f"{context.truth_frames_name}#tick={context.tick}.contact_collection[{collection_index}][{item_index}]",
                        ),
                    )
    return None, ()


def _pair_ids(item: Mapping[str, Any]) -> set[str]:
    for keys in (("entity_a", "entity_b"), ("first_entity_id", "second_entity_id"), ("a", "b")):
        first = item.get(keys[0])
        second = item.get(keys[1])
        if isinstance(first, str) and isinstance(second, str):
            return {first, second}
    ids = item.get("entities")
    if isinstance(ids, list) and len(ids) == 2 and all(isinstance(value, str) for value in ids):
        return set(ids)
    return set()


def _observation(
    context: TickContext,
    metric_id: str,
    value: Any,
    *,
    unit: str,
    subject_id: str,
    source_class: str,
    source_refs: Sequence[str],
    quality: str,
    object_id: str | None = None,
    unknown_reason: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema_name": "mechanism_observation",
        "schema_version": "1.0.0",
        "observation_id": stable_identifier(
            "mechanism_obs",
            context.episode_id,
            context.tick,
            metric_id,
            subject_id,
            object_id or "",
        ),
        "episode_id": context.episode_id,
        "tick": context.tick,
        "metric_id": metric_id,
        "subject_id": subject_id,
        "object_id": object_id,
        "value": value,
        "unit": unit,
        "source_class": source_class,
        "source_refs": sorted(set(source_refs)),
        "quality": quality,
        "unknown_reason": unknown_reason,
    }
    return row


def _unknown_entity_observation(
    context: TickContext,
    metric_id: str,
    unit: str,
    entity_id: str,
    unknown_reason: str,
) -> dict[str, Any]:
    return _observation(
        context,
        metric_id,
        None,
        unit=unit,
        subject_id=entity_id,
        source_class="unknown",
        source_refs=(_entity_ref(context, entity_id),),
        quality="unknown",
        unknown_reason=unknown_reason,
    )


def _unknown_pair_observation(
    context: TickContext,
    metric_id: str,
    unit: str,
    pair_id: str,
    unknown_reason: str,
) -> dict[str, Any]:
    first_id, second_id = pair_id.split("|", 1)
    return _observation(
        context,
        metric_id,
        None,
        unit=unit,
        subject_id=pair_id,
        object_id=second_id,
        source_class="unknown",
        source_refs=(_entity_ref(context, first_id), _entity_ref(context, second_id)),
        quality="unknown",
        unknown_reason=unknown_reason,
    )


def _entity_ref(context: TickContext, entity_id: str) -> str:
    if entity_id in context.frame_entities:
        return f"{context.truth_frames_name}#tick={context.tick}&entity={entity_id}"
    if entity_id in context.static_entities:
        return f"{context.static_name}#entity={entity_id}"
    return f"{context.roster_name}#entity={entity_id}"


def _entity_category(entity: Mapping[str, Any]) -> str | None:
    value = entity.get("entity_category", entity.get("category"))
    return value if isinstance(value, str) else None


def _get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping):
            return None
        if part in FORBIDDEN_FACT_KEYS:
            return None
        current = current.get(part)
    return current


def _numeric_at(value: Mapping[str, Any], path: Sequence[str]) -> float | None:
    candidate = _get(value, path)
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return None
    if not math.isfinite(float(candidate)):
        return None
    return float(candidate)


def _first_numeric_with_path(
    value: Mapping[str, Any],
    paths: Iterable[Sequence[str]],
) -> tuple[float | None, Sequence[str]]:
    for path in paths:
        candidate = _numeric_at(value, path)
        if candidate is not None:
            return candidate, path
    return None, ()


def _vector_at(
    value: Mapping[str, Any],
    path: Sequence[str],
    minimum_length: int,
) -> tuple[float, ...] | None:
    candidate = _get(value, path)
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes, bytearray)):
        return None
    if len(candidate) < minimum_length:
        return None
    result: list[float] = []
    for item in candidate[:minimum_length]:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        result.append(number)
    return tuple(result)


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _iter_numeric_leaves(value: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, float]]:
    for key, item in sorted(value.items()):
        if key in FORBIDDEN_FACT_KEYS:
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            yield from _iter_numeric_leaves(item, path)
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
            yield path, float(item)


def _weather_unit(path: str) -> str:
    lower = path.lower()
    if "visibility" in lower:
        return "m"
    if "temperature" in lower or lower.endswith("temp_c"):
        return "degC"
    if "wind" in lower and ("speed" in lower or lower.endswith("mps")):
        return "m/s"
    if "rain" in lower or "precipitation" in lower:
        return "mm/h"
    if "humidity" in lower:
        return "percent"
    return "scalar"


def _sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["tick"],
        row["metric_id"],
        row["subject_id"],
        row.get("object_id") or "",
        row["observation_id"],
    )
