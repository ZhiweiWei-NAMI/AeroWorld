"""Observable state completion for contract-facing semantic predicates.

This module derives supplemental state snapshots only from sampled truth-frame
state.  It does not read authored event traces, event realizations, dynamic
labels, scenario plans, task ids, or semantic roles.  Missing numeric inputs
produce ``unknown`` for the affected tick instead of intent-derived truth.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from Dataset.semantic_truth.provenance import (
    digest_object,
    load_json,
    read_jsonl,
    stable_identifier,
)

SCHEMA_VERSION = "1.0.0"
RULE_VERSION = "1.2.0"
UNKNOWN = "unknown"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_INPUT_NAMES = {
    "dynamic_labels.jsonl",
    "event_realization.jsonl",
    "event_trace.jsonl",
    "scenario_plan.json",
}

_UAV_CATEGORIES = {"aircraft_uav", "drone", "uav", "unmanned_aerial_vehicle"}
_PEDESTRIAN_CATEGORIES = {"human", "pedestrian", "person", "walker"}
_VEHICLE_CATEGORIES = {
    "ambulance",
    "car",
    "ego_vehicle",
    "ground_vehicle",
    "traffic_vehicle",
    "vehicle",
}
_FACILITY_CATEGORIES = {
    "backup_charger",
    "charger",
    "charging_pad",
    "charging_station",
    "facility",
    "landing_facility",
    "landing_pad",
    "pad",
}
_PAD_CATEGORIES = {"charging_pad", "landing_facility", "landing_pad", "pad"}

_MISSION_BOOL_FIELDS = (
    "active",
    "unsafe",
    "termination_active",
    "safe_altitude_reached",
    "inspection_complete",
    "hazard_managed",
)
_NAVIGATION_BOOL_FIELDS = (
    "visual_navigation_degraded",
    "visual_navigation_failed",
    "visual_navigation_recovered",
    "degraded",
    "route_degraded",
    "route_uncertain",
    "route_recovered",
    "replan_active",
    "alternate_charger_selected",
    "correction_active",
)
_COMMUNICATION_BOOL_FIELDS = (
    "handover_active",
    "backup_link_active",
    "link_restored",
    "channel_recovered",
)
_INCIDENT_BOOL_FIELDS = (
    "detection_active",
    "dispatch_active",
    "responder_arrived",
    "handoff_complete",
    "hazmat_resolved",
    "isolation_active",
    "manual_control_active",
    "temporary_lockdown_active",
    "lockdown_clear",
    "requires_reroute",
    "hazard_source_active",
    "hazard_leak_active",
    "hazard_spread_active",
    "landing_failed",
)
_CONTROL_BOOL_FIELDS = (
    "structure_evasion",
    "structure_recovery",
    "safe_hold",
    "altitude_corrective_maneuver",
    "deconfliction_active",
    "pull_up_active",
    "diversion_active",
    "rth_active",
    "reroute_active",
    "slowdown_active",
)
_SECURITY_BOOL_FIELDS = (
    "lockout_active",
    "alternate_channel_active",
    "secure_recovery_active",
)


@dataclass(frozen=True)
class ObservableStateParameters:
    gust_window_ticks: int = 15
    gust_delta_on_mps: float = 3.0
    gust_delta_off_mps: float = 1.0
    heavy_rain_on: float = 0.65
    heavy_rain_off: float = 0.45
    visibility_low_on_m: float = 250.0
    visibility_low_off_m: float = 350.0
    wetness_on: float = 0.60
    wetness_off: float = 0.40
    braking_accel_mps2: float = -1.5
    braking_speed_drop_mps: float = 0.05
    retreat_distance_delta_m: float = 0.75
    retreat_hold_ticks: int = 10
    pad_eta_window_ticks: int = 20
    pad_default_capacity: int = 1
    lockdown_radius_m: float = 12.0
    lockdown_height_m: float = 120.0
    lockdown_control_patient_max_distance_m: float = 20.0
    hazmat_safe_radius_margin_m: float = 5.0
    hazmat_safe_dwell_ticks: int = 10
    hazmat_ambulance_perimeter_tolerance_m: float = 6.0
    hazmat_arrival_dwell_ticks: int = 10
    hazmat_handoff_dwell_ticks: int = 15


def _load_static_lockdown_entities(
    episode_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load only lawful static physical placement metadata.

    The manifest is used solely to locate ``scene_setup.json``.  Event scripts,
    event traces, scenario plans, authored stage labels, and initial semantic
    state are never read.  Dynamic presence and activation still come from the
    same-tick truth-frame entity.
    """

    manifest_path = episode_root / "episode_manifest.json"
    if not manifest_path.is_file():
        return {}, {}
    manifest = load_json(manifest_path)
    source_value = manifest.get("source_scene_setup_path")
    if not isinstance(source_value, str) or not source_value.strip():
        return {}, {}
    source_path = Path(source_value)
    candidates = (
        [source_path]
        if source_path.is_absolute()
        else [episode_root / source_path, PROJECT_ROOT / source_path]
    )
    allowed_roots = (episode_root.resolve(), PROJECT_ROOT.resolve())
    resolved: Path | None = None
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_file() or candidate.name != "scene_setup.json":
            continue
        if not any(
            candidate == root or root in candidate.parents for root in allowed_roots
        ):
            continue
        resolved = candidate
        break
    if resolved is None:
        return {}, {}
    payload = load_json(resolved)
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        return {}, {}
    entities: dict[str, dict[str, Any]] = {}
    refs: dict[str, str] = {}
    for raw in raw_entities:
        if not isinstance(raw, Mapping):
            continue
        entity_id = raw.get("entity_id")
        placement = raw.get("placement")
        if not isinstance(entity_id, str) or not isinstance(placement, Mapping):
            continue
        entities[entity_id] = {
            key: raw.get(key)
            for key in (
                "entity_id",
                "category",
                "entity_category",
                "entity_kind",
                "entity_type",
                "logical_asset_id",
                "placement_mode",
            )
            if raw.get(key) is not None
        }
        entities[entity_id]["placement"] = dict(placement)
        refs[entity_id] = (
            f"{source_value.replace(chr(92), '/')}#entity={entity_id}#path=placement"
        )
    return entities, refs


def build_observable_state_rows(
    episode_root: Path,
    *,
    parameters: ObservableStateParameters | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic observable-state snapshots for one episode."""

    episode_root = episode_root.resolve()
    truth_path = episode_root / "truth_frames.jsonl"
    weather_path = episode_root / "weather_meta.jsonl"
    if not truth_path.is_file():
        raise FileNotFoundError(f"truth_frames.jsonl is missing: {truth_path}")
    if not weather_path.is_file():
        raise FileNotFoundError(f"weather_meta.jsonl is missing: {weather_path}")
    params = _parameters(parameters)
    frames = sorted(read_jsonl(truth_path), key=lambda item: int(item.get("tick", -1)))
    weather_by_tick: dict[int, Mapping[str, Any]] = {}
    for weather in read_jsonl(weather_path):
        weather_tick = weather.get("tick")
        if not isinstance(weather_tick, int):
            raise ValueError("weather_meta.jsonl row lacks integer tick")
        if weather_tick in weather_by_tick:
            raise ValueError(f"duplicate weather row at tick {weather_tick}")
        weather_by_tick[weather_tick] = weather
    episode_id = _episode_id(episode_root, frames)
    static_lockdown_entities, static_lockdown_refs = _load_static_lockdown_entities(
        episode_root
    )
    weather_latches: dict[str, bool] = {}
    wind_history: deque[tuple[int, float]] = deque()
    previous_entities: dict[str, Mapping[str, Any]] = {}
    previous_entity_ticks: dict[str, int] = {}
    previous_pedestrian_positions: dict[str, tuple[float, float, float]] = {}
    previous_pedestrian_ticks: dict[str, int] = {}
    retreat_ticks: dict[tuple[str, str], int] = defaultdict(int)
    previous_pair_distance: dict[tuple[str, str], float] = {}
    hazmat_response_state: dict[str, Any] = {
        "ped_safe_ticks": defaultdict(int),
        "dispatch_observed": False,
        "arrival_ticks": 0,
        "handoff_ticks": 0,
        "hazard_center": None,
        "hazard_radius_m": UNKNOWN,
        "isolation_observed": False,
        "target_pedestrian_ids": None,
        "previous_ambulance_present": None,
        "previous_ambulance_distance_m": None,
        "last_tick": None,
    }
    rows: list[dict[str, Any]] = []

    for frame in frames:
        tick = int(frame["tick"])
        if tick not in weather_by_tick:
            raise ValueError(f"weather_meta.jsonl lacks tick {tick}")
        observed_frame = dict(frame)
        observed_frame["environment"] = dict(weather_by_tick[tick])
        entities = [
            entity
            for entity in observed_frame.get("entities", ())
            if isinstance(entity, Mapping)
        ]
        rows.extend(
            _environment_rows(
                episode_id,
                tick,
                observed_frame,
                entities,
                params,
                weather_latches,
                wind_history,
            )
        )
        rows.extend(
            _hazmat_response_rows(
                episode_id,
                tick,
                observed_frame,
                entities,
                params,
                hazmat_response_state,
            )
        )
        rows.extend(
            _lockdown_region_rows(
                episode_id,
                tick,
                observed_frame,
                entities,
                params,
                static_lockdown_entities,
                static_lockdown_refs,
            )
        )
        rows.extend(_sensor_rows(episode_id, tick, entities, params))
        rows.extend(_structured_runtime_rows(episode_id, tick, entities, params))
        vehicle_rows, vehicle_by_id = _vehicle_rows(
            episode_id,
            tick,
            entities,
            previous_entities,
            previous_entity_ticks,
            params,
        )
        rows.extend(vehicle_rows)
        rows.extend(
            _pedestrian_rows(
                episode_id,
                tick,
                entities,
                vehicle_by_id,
                params,
                previous_pair_distance,
                retreat_ticks,
                previous_pedestrian_positions,
                previous_pedestrian_ticks,
            )
        )
        rows.extend(_facility_rows(episode_id, tick, entities, params))
        previous_entities = {
            str(entity.get("entity_id")): entity
            for entity in entities
            if isinstance(entity.get("entity_id"), str)
        }
        previous_entity_ticks = {
            str(entity.get("entity_id")): tick
            for entity in entities
            if isinstance(entity.get("entity_id"), str)
        }

    rows.sort(
        key=lambda row: (
            row["tick"],
            row["observation_family"],
            row["subject_id"],
            row["observation_id"],
        )
    )
    return rows


def _environment_rows(
    episode_id: str,
    tick: int,
    frame: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    params: ObservableStateParameters,
    latches: dict[str, bool],
    wind_history: deque[tuple[int, float]],
) -> list[dict[str, Any]]:
    weather = _first_mapping(
        frame, ("weather", "environment", "weather_state", "environment_state")
    )
    rain = _number(
        _first_value(
            weather,
            ("rain", "rain_rate", "precipitation_rate", "precipitation_rate_sim"),
        )
    )
    visibility = _number(
        _first_value(weather, ("visibility_m", "visibility", "visibility_distance_m"))
    )
    wetness = _number(
        _first_value(weather, ("road_wetness", "wetness", "surface_wetness"))
    )
    wind = _number(
        _first_value(weather, ("wind_speed_mps", "crosswind_speed_mps", "wind_speed"))
    )
    gust = UNKNOWN
    if wind is not None:
        wind_history.append((tick, wind))
        while wind_history and tick - wind_history[0][0] > params.gust_window_ticks:
            wind_history.popleft()
        baseline = min(value for _, value in wind_history)
        delta = wind - baseline
        gust = _hysteresis(
            "gust",
            delta,
            params.gust_delta_on_mps,
            params.gust_delta_off_mps,
            latches,
            high_is_active=True,
        )

    values = {
        "gust_active": gust,
        "heavy_rain_active": _hysteresis(
            "heavy_rain",
            rain,
            params.heavy_rain_on,
            params.heavy_rain_off,
            latches,
            high_is_active=True,
        ),
        "visibility_low": _hysteresis(
            "visibility_low",
            visibility,
            params.visibility_low_on_m,
            params.visibility_low_off_m,
            latches,
            high_is_active=False,
        ),
        "rain_rate": rain if rain is not None else UNKNOWN,
        "visibility_m": visibility if visibility is not None else UNKNOWN,
        "wind_speed_mps": wind if wind is not None else UNKNOWN,
        "wind_direction_deg": _number_or_unknown(
            _first_value(weather, ("wind_direction_deg", "wind_heading_deg"))
        ),
        "temperature_c": _number_or_unknown(
            _first_value(weather, ("temperature_c", "air_temperature_c"))
        ),
        "illumination_lux": _number_or_unknown(
            _first_value(weather, ("illumination_lux", "ambient_illumination_lux"))
        ),
        "hazard_source_active": _first_known(
            _first_exact_bool(weather.get("hazard_source_active")),
        ),
        "hazard_concentration_ppm": _first_known(
            _number_or_unknown(weather.get("hazard_concentration_ppm")),
        ),
        "hazard_radius_m": _first_known(
            _number_or_unknown(weather.get("hazard_radius_m")),
        ),
        "gust_window_ticks": params.gust_window_ticks,
    }
    ground_values = {
        "wet_road_active": _hysteresis(
            "wet_road",
            wetness,
            params.wetness_on,
            params.wetness_off,
            latches,
            high_is_active=True,
        ),
        "surface_wetness": wetness if wetness is not None else UNKNOWN,
    }
    return [
        _row(
            episode_id,
            tick,
            "observable_environment",
            "environment",
            {"weather_region": "environment"},
            values,
            ["weather_meta.jsonl#tick=%d" % tick],
            parameters=params,
        ),
        _row(
            episode_id,
            tick,
            "observable_ground_surface",
            "road_network",
            {"surface": "road_network"},
            ground_values,
            ["weather_meta.jsonl#tick=%d#field=wetness" % tick],
            parameters=params,
        ),
    ]


def _hazmat_response_rows(
    episode_id: str,
    tick: int,
    frame: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    params: ObservableStateParameters,
    response_state: dict[str, Any],
) -> list[dict[str, Any]]:
    weather = _first_mapping(
        frame, ("weather", "environment", "weather_state", "environment_state")
    )
    hazard_source_active = _first_exact_bool(weather.get("hazard_source_active"))
    hazard_concentration = _number_or_unknown(weather.get("hazard_concentration_ppm"))
    hazard_radius = _number_or_unknown(weather.get("hazard_radius_m"))
    hazard_active = _hazmat_numeric_active(
        hazard_source_active, hazard_concentration, hazard_radius
    )
    hazard_cleared = (
        hazard_source_active is False
        and hazard_concentration == 0.0
        and hazard_radius == 0.0
    )
    if isinstance(hazard_radius, (int, float)) and hazard_radius > 0.0:
        response_state["hazard_radius_m"] = hazard_radius

    hazard_controls = [
        entity for entity in entities if _is_hazmat_hazard_or_cordon_entity(entity)
    ]
    active_controls = [
        entity for entity in hazard_controls if _hazmat_entity_deployed(entity)
    ]
    for entity in [*active_controls, *hazard_controls]:
        center = _hazmat_hazard_center(entity, allow_static=entity in active_controls)
        if center is not None:
            response_state["hazard_center"] = center
            break

    last_tick = response_state.get("last_tick")
    elapsed_ticks = max(1, tick - int(last_tick)) if isinstance(last_tick, int) else 1
    response_state["last_tick"] = tick

    effective_radius = _number(response_state.get("hazard_radius_m"))
    hazard_center = response_state.get("hazard_center")
    if active_controls and hazard_center is not None and hazard_active is True:
        isolation_active: bool | str = True
    elif hazard_active == UNKNOWN:
        isolation_active = UNKNOWN
    else:
        isolation_active = False
    if isolation_active is True:
        response_state["isolation_observed"] = True

    target_pedestrians = _hazmat_target_pedestrians(entities)
    all_targets_safe: bool | str = False
    if response_state.get("target_pedestrian_ids") is None:
        initial_target_ids = [
            str(entity.get("entity_id") or "")
            for entity in target_pedestrians
            if str(entity.get("entity_id") or "")
        ]
        if len(initial_target_ids) >= 2:
            response_state["target_pedestrian_ids"] = sorted(initial_target_ids[:2])
    target_ids = response_state.get("target_pedestrian_ids")
    if isinstance(target_ids, list):
        target_by_id = {
            str(entity.get("entity_id") or ""): entity for entity in target_pedestrians
        }
        target_pedestrians = [
            target_by_id[target_id]
            for target_id in target_ids
            if target_id in target_by_id
        ]
        if len(target_pedestrians) < len(target_ids):
            all_targets_safe = UNKNOWN
    missing_target_ids = (
        [
            target_id
            for target_id in target_ids
            if target_id
            not in {str(entity.get("entity_id") or "") for entity in target_pedestrians}
        ]
        if isinstance(target_ids, list)
        else UNKNOWN
    )
    pedestrian_diagnostics: dict[str, dict[str, Any]] = {}
    if (
        all_targets_safe != UNKNOWN
        and len(target_pedestrians) >= 2
        and isinstance(hazard_center, tuple)
        and effective_radius is not None
    ):
        ped_safe_ticks = response_state["ped_safe_ticks"]
        safe_ids: list[str] = []
        for pedestrian in target_pedestrians:
            pedestrian_id = str(pedestrian.get("entity_id") or UNKNOWN)
            position = _hazmat_dynamic_position(pedestrian)
            present = position is not None and _hazmat_dynamic_entity_present(
                pedestrian
            )
            if not present or position is None:
                ped_safe_ticks[pedestrian_id] = 0
                pedestrian_diagnostics[pedestrian_id] = {
                    "present": False,
                    "distance_m": UNKNOWN,
                    "safe_dwell_ticks": 0,
                    "safe": UNKNOWN,
                }
                all_targets_safe = UNKNOWN
                continue
            distance = _distance_xy(position, hazard_center)
            if distance >= effective_radius + float(params.hazmat_safe_radius_margin_m):
                ped_safe_ticks[pedestrian_id] += elapsed_ticks
            else:
                ped_safe_ticks[pedestrian_id] = 0
            safe = ped_safe_ticks[pedestrian_id] >= params.hazmat_safe_dwell_ticks
            pedestrian_diagnostics[pedestrian_id] = {
                "present": True,
                "distance_m": distance,
                "safe_dwell_ticks": ped_safe_ticks[pedestrian_id],
                "safe": safe,
            }
            if safe:
                safe_ids.append(pedestrian_id)
        if all_targets_safe != UNKNOWN:
            all_targets_safe = len(safe_ids) == len(target_pedestrians)
    elif all_targets_safe != UNKNOWN:
        all_targets_safe = UNKNOWN

    ambulance = _hazmat_ambulance_entity(entities)
    ambulance_position = (
        _hazmat_dynamic_position(ambulance) if ambulance is not None else None
    )
    ambulance_visible = (
        ambulance is not None
        and ambulance_position is not None
        and _hazmat_dynamic_entity_present(ambulance)
    )
    ambulance_distance: float | None = None
    if ambulance is not None and isinstance(hazard_center, tuple):
        if ambulance_position is not None:
            ambulance_distance = _distance_xy(ambulance_position, hazard_center)
    previous_distance = _number(response_state.get("previous_ambulance_distance_m"))
    closing_after_isolation = False
    if ambulance_visible:
        speed = _speed(ambulance)
        previous_present = response_state.get("previous_ambulance_present")
        appeared_after_isolation = (
            response_state.get("isolation_observed") is True
            and previous_present is False
        )
        closing_after_isolation = (
            response_state.get("isolation_observed") is True
            and ambulance_distance is not None
            and previous_distance is not None
            and ambulance_distance < previous_distance - 0.05
        )
        if (
            appeared_after_isolation
            or closing_after_isolation
            or (
                response_state.get("isolation_observed") is True
                and speed is not None
                and speed > 0.05
            )
        ):
            response_state["dispatch_observed"] = True
    response_state["previous_ambulance_present"] = bool(ambulance_visible)
    response_state["previous_ambulance_distance_m"] = (
        ambulance_distance if ambulance_distance is not None else UNKNOWN
    )

    arrival_condition: bool | str = False
    if (
        response_state.get("dispatch_observed") is True
        and ambulance_visible
        and ambulance is not None
        and isinstance(hazard_center, tuple)
        and effective_radius is not None
    ):
        if ambulance_distance is not None:
            tolerance = float(params.hazmat_ambulance_perimeter_tolerance_m)
            arrival_condition = (
                ambulance_distance <= effective_radius + tolerance
                and ambulance_distance >= effective_radius
            )
    elif ambulance is not None and (
        hazard_active == UNKNOWN
        or not isinstance(hazard_center, tuple)
        or effective_radius is None
    ):
        arrival_condition = UNKNOWN
    if arrival_condition is True:
        response_state["arrival_ticks"] += elapsed_ticks
    else:
        response_state["arrival_ticks"] = 0
        response_state["handoff_ticks"] = 0
    responder_arrived: bool | str = (
        UNKNOWN
        if arrival_condition == UNKNOWN
        else response_state["arrival_ticks"] >= params.hazmat_arrival_dwell_ticks
    )
    if responder_arrived is True:
        response_state["handoff_ticks"] += elapsed_ticks
    handoff_complete: bool | str = (
        response_state["handoff_ticks"] >= params.hazmat_handoff_dwell_ticks
    )
    hazmat_resolved: bool | str = (
        True
        if (
            hazard_cleared
            and response_state.get("isolation_observed") is True
            and all_targets_safe is True
            and responder_arrived is True
            and handoff_complete is True
        )
        else UNKNOWN
        if hazard_active == UNKNOWN or all_targets_safe == UNKNOWN
        else False
    )
    source_refs = [f"weather_meta.jsonl#tick={tick}#path=hazmat"]
    source_refs.extend(
        f"truth_frames.jsonl#tick={tick}#entity={entity.get('entity_id') or UNKNOWN}#path=pose_state"
        for entity in [
            *hazard_controls,
            *target_pedestrians,
            *([ambulance] if ambulance is not None else []),
        ]
    )
    target_ids_value = target_ids if isinstance(target_ids, list) else UNKNOWN
    ambulance_id = (
        str(ambulance.get("entity_id") or UNKNOWN) if ambulance is not None else UNKNOWN
    )
    return [
        _row(
            episode_id,
            tick,
            "hazmat_response_state",
            "environment",
            {"weather_region": "environment"},
            {
                "hazard_source_active": hazard_source_active,
                "hazard_concentration_ppm": hazard_concentration,
                "hazard_radius_m": hazard_radius,
                "hazard_center_enu_m": list(hazard_center)
                if isinstance(hazard_center, tuple)
                else UNKNOWN,
                "hazard_active": hazard_active,
                "isolation_active": isolation_active,
                "isolation_observed": response_state.get("isolation_observed") is True,
                "target_pedestrian_ids": target_ids_value,
                "missing_target_pedestrian_ids": missing_target_ids,
                "target_pedestrian_distances_m": {
                    key: value["distance_m"]
                    for key, value in pedestrian_diagnostics.items()
                },
                "target_pedestrian_safe_dwell_ticks": {
                    key: value["safe_dwell_ticks"]
                    for key, value in pedestrian_diagnostics.items()
                },
                "all_targets_safe": all_targets_safe,
                "ambulance_id": ambulance_id,
                "ambulance_present": bool(ambulance_visible),
                "ambulance_distance_m": ambulance_distance
                if ambulance_distance is not None
                else UNKNOWN,
                "ambulance_previous_distance_m": previous_distance
                if previous_distance is not None
                else UNKNOWN,
                "ambulance_closing_after_isolation": closing_after_isolation,
                "dispatch_observed": response_state.get("dispatch_observed") is True,
                "arrival_condition": arrival_condition,
                "arrival_dwell_ticks": response_state["arrival_ticks"],
                "arrival_required_dwell_ticks": params.hazmat_arrival_dwell_ticks,
                "responder_arrived": responder_arrived,
                "handoff_dwell_ticks": response_state["handoff_ticks"],
                "handoff_required_dwell_ticks": params.hazmat_handoff_dwell_ticks,
                "handoff_complete": handoff_complete,
                "safe_required_dwell_ticks": params.hazmat_safe_dwell_ticks,
                "safe_radius_margin_m": params.hazmat_safe_radius_margin_m,
                "ambulance_perimeter_tolerance_m": params.hazmat_ambulance_perimeter_tolerance_m,
                "hazmat_resolved": hazmat_resolved,
            },
            source_refs,
            parameters=params,
        )
    ]


def _is_hazmat_hazard_or_cordon_entity(entity: Mapping[str, Any]) -> bool:
    identity = _entity_identity_text(entity)
    category = _category(entity)
    if category in {"pedestrian", "vehicle", "uav"} and not any(
        token in identity
        for token in ("cordon", "police_tape", "barrier", "isolation_perimeter")
    ):
        return False
    return any(
        token in identity
        for token in (
            "hazard",
            "cordon",
            "isolation",
            "police_tape",
            "barrier",
        )
    )


def _hazmat_numeric_active(
    source_active: bool | str,
    concentration_ppm: float | str,
    radius_m: float | str,
) -> bool | str:
    concentration_known = isinstance(concentration_ppm, (int, float))
    radius_known = isinstance(radius_m, (int, float))
    if (
        source_active is True
        or (concentration_known and float(concentration_ppm) > 0.0)
        or (radius_known and float(radius_m) > 0.0)
    ):
        return True
    if source_active is False and concentration_ppm == 0.0 and radius_m == 0.0:
        return False
    return UNKNOWN


def _hazmat_target_pedestrians(
    entities: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    pedestrians = [
        entity
        for entity in entities
        if _category(entity) == "pedestrian"
        or _exact_string(entity.get("label_class")) == "pedestrian"
    ]
    hazmat_pedestrians = [
        entity for entity in pedestrians if "hazmat" in _entity_identity_text(entity)
    ]
    # Fail closed instead of binding arbitrary background pedestrians into a
    # hazmat response.  Target identity must be present in observed entity
    # metadata; otherwise the response row remains unknown.
    return sorted(
        hazmat_pedestrians,
        key=lambda entity: str(entity.get("entity_id") or ""),
    )


def _hazmat_ambulance_entity(
    entities: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    ambulances = [
        entity for entity in entities if "ambulance" in _entity_identity_text(entity)
    ]
    if not ambulances:
        return None
    return sorted(ambulances, key=lambda entity: str(entity.get("entity_id") or ""))[0]


def _hazmat_entity_deployed(entity: Mapping[str, Any]) -> bool:
    return _constraint_active(entity) is True


def _hazmat_dynamic_entity_present(entity: Mapping[str, Any]) -> bool:
    render_presence = entity.get("render_presence")
    if not isinstance(render_presence, Mapping):
        return False
    offstage = render_presence.get("offstage")
    visibility = _exact_string(render_presence.get("visibility_state"))
    submission = _exact_string(render_presence.get("submission_state"))
    return (
        offstage is False
        and visibility not in {"hidden", "not_visible", "absent"}
        and submission not in {"withheld", "not_submitted", "absent"}
    )


def _constraint_active(entity: Mapping[str, Any]) -> bool | str:
    incident_state = entity.get("incident_state")
    if isinstance(incident_state, Mapping):
        temporary_lockdown_active = _first_exact_bool(
            incident_state.get("temporary_lockdown_active")
        )
        if isinstance(temporary_lockdown_active, bool):
            return temporary_lockdown_active
    state = entity.get("constraint_state")
    if not isinstance(state, Mapping):
        return UNKNOWN
    return _first_exact_bool(state.get("active"))


def _hazmat_dynamic_position(
    entity: Mapping[str, Any] | None,
) -> tuple[float, float, float] | None:
    if entity is None:
        return None
    position = _position(entity)
    if position is not None:
        return position
    value = entity.get("pos_enu")
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 3
    ):
        nums = [_number(item) for item in value[:3]]
        if all(item is not None for item in nums):
            return (float(nums[0]), float(nums[1]), float(nums[2]))
    return None


def _hazmat_hazard_center(
    entity: Mapping[str, Any],
    *,
    allow_static: bool,
) -> tuple[float, float, float] | None:
    position = _hazmat_dynamic_position(entity)
    if position is not None:
        return position
    if not allow_static:
        return None
    placement = entity.get("placement")
    if isinstance(placement, Mapping):
        value = placement.get("center_enu_m") or placement.get(
            "resolved_position_enu_m"
        )
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) >= 3
        ):
            nums = [_number(item) for item in value[:3]]
            if all(item is not None for item in nums):
                return (float(nums[0]), float(nums[1]), float(nums[2]))
    return None


def _entity_identity_text(entity: Mapping[str, Any]) -> str:
    parts = [
        entity.get("entity_id"),
        entity.get("asset_id"),
        entity.get("logical_asset_id"),
        entity.get("category"),
        entity.get("entity_category"),
        entity.get("label_class"),
    ]
    return " ".join(str(part or "") for part in parts).lower()


def _lockdown_region_rows(
    episode_id: str,
    tick: int,
    frame: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    params: ObservableStateParameters,
    static_entities: Mapping[str, Mapping[str, Any]],
    static_refs: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Bind lockdown control state to a physical, tick-stable region.

    A controller flag alone is insufficient: activation also requires a
    deployed barrier/cordon (or an explicit region geometry) with a usable
    pose.  When a fallen pedestrian is nearby, that identity and distance are
    retained so later causal projection can link the isolation response to the
    same medical incident without consulting authored event identifiers.
    """

    region_rows: list[dict[str, Any]] = []
    for entity in entities:
        entity_id = str(entity.get("entity_id") or UNKNOWN)
        static_entity = static_entities.get(entity_id, {})
        if not _is_lockdown_region_entity(entity, static_entity):
            continue
        active = _constraint_active(entity)
        clear: bool | str = not active if isinstance(active, bool) else UNKNOWN
        region_state = (
            "active" if active is True else "inactive" if active is False else UNKNOWN
        )
        geometry = _lockdown_geometry_from_entity(entity, static_entity)
        region_source_refs = [
            f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=truth_pose",
            f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=constraint_state.active",
            f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=incident_state.temporary_lockdown_active",
        ]
        if entity_id in static_refs:
            region_source_refs.append(static_refs[entity_id])
        region_rows.append(
            _row(
                episode_id,
                tick,
                "lockdown_region_state",
                entity_id,
                {"region": entity_id},
                {
                    "emergency_isolation_zone_id": entity_id,
                    "emergency_isolation_zone_ontology_class_id": (
                        "world:EmergencyIsolationZone"
                    ),
                    "temporary_lockdown_active": active,
                    "lockdown_clear": clear,
                    "geometry": geometry,
                    "region_scope_id": entity_id,
                    "region_state": region_state,
                    "controller_ids": [entity_id],
                    "deployed_controller_ids": [entity_id] if active is True else [],
                    "deployed_controller_count": 1 if active is True else 0,
                    "associated_patient_id": UNKNOWN,
                    "associated_patient_distance_m": UNKNOWN,
                    "lockdown_radius_m": UNKNOWN,
                    "patient_association_max_distance_m": float(
                        params.lockdown_control_patient_max_distance_m
                    ),
                },
                region_source_refs,
                source_class="derived_from_observed",
                parameters=params,
            )
        )

    controls: list[
        tuple[Mapping[str, Any], bool | str, bool | str, Mapping[str, Any]]
    ] = []
    source_refs: list[str] = []
    for entity in entities:
        # A governed airspace region is already represented by its own
        # observed geometry/state row above.  It must not also be interpreted
        # as a ground isolation controller, otherwise one physical NFZ would
        # create a second synthetic buffer region.
        static_entity = static_entities.get(str(entity.get("entity_id") or ""), {})
        if _is_lockdown_region_entity(entity, static_entity):
            continue
        if not _is_ground_lockdown_control(entity):
            continue
        entity_id = str(entity.get("entity_id") or UNKNOWN)
        deployed = _lockdown_control_deployed(entity)
        cleared = _lockdown_control_clear(entity)
        incident_state = _top_level_state(entity, "incident_state") or {}
        controls.append((entity, deployed, cleared, incident_state))
        source_refs.extend(
            [
                f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=truth_pose",
                f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=state",
                f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=render_presence",
            ]
        )
        if incident_state:
            source_refs.append(
                f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=incident_state"
            )
    if not controls:
        return region_rows

    active_controls: list[Mapping[str, Any]] = []
    activation_unknown = False
    known_activation_values = False
    for entity, deployed, _, _ in controls:
        if deployed != UNKNOWN:
            known_activation_values = True
        if deployed is True:
            active_controls.append(entity)
        elif deployed == UNKNOWN:
            activation_unknown = True
    if active_controls:
        lockdown_active: bool | str = True
    elif activation_unknown or not known_activation_values:
        lockdown_active = UNKNOWN
    else:
        lockdown_active = False

    clear_values = [cleared for _, _, cleared, _ in controls]
    physical_clear = _aggregate_exact_bool(clear_values)
    lockdown_clear: bool | str = (
        False
        if lockdown_active is True
        else physical_clear
        if physical_clear != UNKNOWN
        else False
        if lockdown_active is False
        else UNKNOWN
    )

    positioned_controls = [
        (entity, _position(entity))
        for entity in (active_controls or [item[0] for item in controls])
        if _position(entity) is not None
    ]
    center: tuple[float, float, float] | None = None
    if positioned_controls:
        positions = [
            position for _, position in positioned_controls if position is not None
        ]
        center = (
            sum(position[0] for position in positions) / len(positions),
            sum(position[1] for position in positions) / len(positions),
            min(position[2] for position in positions),
        )
    geometry: Mapping[str, Any] | str = UNKNOWN
    if center is not None:
        radius = float(params.lockdown_radius_m)
        geometry = {
            "geometry_kind": "polygon_prism",
            "polygon_enu_m": [
                [center[0] - radius, center[1] - radius],
                [center[0] + radius, center[1] - radius],
                [center[0] + radius, center[1] + radius],
                [center[0] - radius, center[1] + radius],
            ],
            "base_z_m": center[2],
            "height_m": float(params.lockdown_height_m),
            "construction": "governed_buffer_around_observed_isolation_control",
        }

    associated_patient_id: str | None = None
    associated_patient_distance_m: float | None = None
    if lockdown_active is True and center is not None:
        for entity in entities:
            if _category(entity) != "pedestrian":
                continue
            pedestrian_state = _top_level_state(entity, "pedestrian_state") or {}
            fallen = _first_exact_bool(
                pedestrian_state.get("fallen"),
                pedestrian_state.get("injured"),
            )
            posture = _exact_string(pedestrian_state.get("posture"))
            if fallen is not True and posture not in {"fallen", "lying", "prone"}:
                continue
            position = _position(entity)
            if position is None:
                continue
            distance = math.dist(center[:2], position[:2])
            if distance > float(params.lockdown_control_patient_max_distance_m):
                continue
            entity_id = str(entity.get("entity_id") or UNKNOWN)
            if (
                associated_patient_distance_m is None
                or distance < associated_patient_distance_m
                or (
                    math.isclose(distance, associated_patient_distance_m)
                    and entity_id < str(associated_patient_id)
                )
            ):
                associated_patient_id = entity_id
                associated_patient_distance_m = distance
    if associated_patient_id is not None:
        source_refs.extend(
            [
                f"truth_frames.jsonl#tick={tick}#entity={associated_patient_id}#path=pedestrian_state",
                f"truth_frames.jsonl#tick={tick}#entity={associated_patient_id}#path=truth_pose",
            ]
        )

    controller_ids = sorted(
        str(entity.get("entity_id") or UNKNOWN) for entity, _, _, _ in controls
    )
    region_id = stable_identifier(
        "emergency_isolation_zone", episode_id, controller_ids
    )
    deployed_ids = sorted(
        str(entity.get("entity_id") or UNKNOWN)
        for entity, deployed, _, _ in controls
        if deployed is True
    )
    barrier_row = _row(
        episode_id,
        tick,
        "lockdown_region_state",
        region_id,
        {"region": region_id},
        {
            "emergency_isolation_zone_id": region_id,
            "emergency_isolation_zone_ontology_class_id": (
                "world:EmergencyIsolationZone"
            ),
            "temporary_lockdown_active": lockdown_active,
            "lockdown_clear": lockdown_clear,
            "geometry": geometry,
            "region_scope_id": region_id,
            "controller_ids": controller_ids,
            "deployed_controller_ids": deployed_ids,
            "deployed_controller_count": len(deployed_ids),
            "associated_patient_id": associated_patient_id or UNKNOWN,
            "associated_patient_distance_m": (
                associated_patient_distance_m
                if associated_patient_distance_m is not None
                else UNKNOWN
            ),
            "lockdown_radius_m": float(params.lockdown_radius_m),
            "patient_association_max_distance_m": float(
                params.lockdown_control_patient_max_distance_m
            ),
        },
        sorted(set(source_refs)),
        source_class="simulated_derived",
        parameters=params,
    )
    return [*region_rows, barrier_row]


def _is_lockdown_region_entity(
    entity: Mapping[str, Any],
    static_entity: Mapping[str, Any] | None = None,
) -> bool:
    static_entity = static_entity or {}
    category = str(
        entity.get("entity_category")
        or entity.get("category")
        or static_entity.get("entity_category")
        or static_entity.get("category")
        or ""
    ).lower()
    if category != "facility" and "airspace_constraint" not in category:
        return False
    identity = " ".join(
        str(entity.get(key) or static_entity.get(key) or "")
        for key in (
            "entity_category",
            "category",
            "entity_kind",
            "entity_type",
            "logical_asset_id",
            "proxy_template_id",
            "placement_mode",
        )
    ).lower()
    return bool(
        "airspace_constraint" in identity
        or "no_fly" in identity
        or "no-fly" in identity
        or "nfz" in identity
    )


def _lockdown_geometry_from_entity(
    entity: Mapping[str, Any],
    static_entity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | str:
    static_entity = static_entity or {}
    geometry = entity.get("geometry") or static_entity.get("geometry")
    if isinstance(geometry, Mapping):
        return dict(geometry)
    return UNKNOWN


def _lockdown_control_deployed(entity: Mapping[str, Any]) -> bool | str:
    if _position(entity) is None:
        return UNKNOWN
    governed_active = _constraint_active(entity)
    if isinstance(governed_active, bool):
        return governed_active
    state = _exact_string(entity.get("state"))
    if state in {
        "offstage",
        "inactive",
        "hidden",
        "stowed",
        "removed",
        "standdown",
        "demobilized",
        "cleared",
    }:
        return False
    render_presence = (
        entity.get("render_presence")
        if isinstance(entity.get("render_presence"), Mapping)
        else {}
    )
    if render_presence.get("offstage") is True:
        return False
    if _exact_string(render_presence.get("submission_state")) in {
        "do_not_submit",
        "excluded",
        "hidden",
    }:
        return False
    identity = " ".join(
        str(entity.get(key) or "")
        for key in ("entity_kind", "entity_type", "logical_asset_id", "label_class")
    ).lower()
    tags = entity.get("tags")
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
        identity = f"{identity} {' '.join(str(tag) for tag in tags)}".lower()
    physical_control = any(
        token in identity
        for token in ("police_tape", "barrier", "cordon", "roadblock", "isolation")
    ) or isinstance(entity.get("geometry"), Mapping)
    if not physical_control:
        return UNKNOWN
    if state in {"staged", "deployed", "active", "isolating", "lockdown"}:
        return True
    visibility = _exact_string(render_presence.get("visibility_state"))
    if state is None and visibility in {"visible", "rendered"}:
        return True
    return UNKNOWN


def _lockdown_control_clear(entity: Mapping[str, Any]) -> bool | str:
    governed_active = _constraint_active(entity)
    if isinstance(governed_active, bool):
        return not governed_active
    state = _exact_string(entity.get("state"))
    if state in {
        "offstage",
        "inactive",
        "hidden",
        "stowed",
        "removed",
        "standdown",
        "demobilized",
        "cleared",
    }:
        return True
    if state in {"staged", "deployed", "active", "isolating", "lockdown"}:
        return False
    render_presence = (
        entity.get("render_presence")
        if isinstance(entity.get("render_presence"), Mapping)
        else {}
    )
    if render_presence.get("offstage") is True:
        return True
    if render_presence.get("offstage") is False:
        return False
    return UNKNOWN


def _is_ground_lockdown_control(entity: Mapping[str, Any]) -> bool:
    if _position(entity) is None:
        return False
    identity = " ".join(
        str(entity.get(key) or "")
        for key in (
            "entity_id",
            "entity_category",
            "entity_kind",
            "entity_type",
            "logical_asset_id",
            "proxy_template_id",
        )
    ).lower()
    normalized = identity.replace("-", "_").replace(" ", "_")
    return any(
        token in normalized
        for token in (
            "police_tape",
            "incident_cordon",
            "isolation_cordon",
            "isolation_perimeter",
            "incident_perimeter",
            "emergency_roadblock",
        )
    )


def _sensor_rows(
    episode_id: str,
    tick: int,
    entities: Sequence[Mapping[str, Any]],
    params: ObservableStateParameters,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in entities:
        sensor_state = _top_level_state(entity, "sensor_state")
        if _category(entity) != "uav" and sensor_state is None:
            continue
        entity_id = str(entity.get("entity_id") or UNKNOWN)
        sensor_state = sensor_state or {}
        underexposed = _first_exact_bool(
            sensor_state.get("image_underexposed"),
            sensor_state.get("underexposed"),
        )
        if underexposed == UNKNOWN:
            exposure_status = _exact_string(sensor_state.get("exposure_status"))
            if exposure_status is not None:
                underexposed = exposure_status == "underexposed"
        infrared_active = _first_exact_bool(
            sensor_state.get("infrared_mode_active"),
            sensor_state.get("ir_active"),
        )
        if infrared_active == UNKNOWN:
            imaging_mode = _exact_string(sensor_state.get("imaging_mode"))
            if imaging_mode is not None:
                infrared_active = imaging_mode in {
                    "infrared",
                    "thermal",
                    "night_vision",
                }
        values = {
            "underexposed": _exact_state_bool(sensor_state, "underexposed"),
            "infrared": _exact_state_bool(sensor_state, "infrared"),
            "exposure_recovered": _exact_state_bool(sensor_state, "exposure_recovered"),
            "sensor_fault": _exact_state_bool(sensor_state, "sensor_fault"),
            "intruder_detected": _exact_state_bool(sensor_state, "intruder_detected"),
            "image_underexposed": underexposed,
            "infrared_mode_active": infrared_active,
            "exposure_status": _exact_string(sensor_state.get("exposure_status"))
            or UNKNOWN,
            "imaging_mode": _exact_string(sensor_state.get("imaging_mode")) or UNKNOWN,
        }
        rows.append(
            _row(
                episode_id,
                tick,
                "observable_sensor_state",
                entity_id,
                {"uav": entity_id},
                values,
                [
                    f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=sensor_state"
                ],
                parameters=params,
            )
        )
    return rows


def _vehicle_rows(
    episode_id: str,
    tick: int,
    entities: Sequence[Mapping[str, Any]],
    previous_entities: Mapping[str, Mapping[str, Any]],
    previous_entity_ticks: Mapping[str, int],
    params: ObservableStateParameters,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    rows: list[dict[str, Any]] = []
    vehicles: dict[str, Mapping[str, Any]] = {}
    for entity in entities:
        if _category(entity) != "vehicle":
            continue
        entity_id = str(entity.get("entity_id") or UNKNOWN)
        vehicles[entity_id] = entity
        speed = _speed(entity)
        previous = previous_entities.get(entity_id)
        previous_speed = _speed(previous) if previous is not None else None
        accel = _accel(entity)
        previous_tick = previous_entity_ticks.get(entity_id)
        elapsed_ticks = (
            max(1, tick - previous_tick) if previous_tick is not None else None
        )
        if (
            accel is None
            and speed is not None
            and previous_speed is not None
            and elapsed_ticks is not None
        ):
            accel = (speed - previous_speed) / (elapsed_ticks / 10.0)
        vehicle_state_mapping = _top_level_state(entity, "vehicle_state")
        vehicle_state = vehicle_state_mapping or {}
        exact_braking = _exact_state_bool(vehicle_state, "braking")
        exact_emergency_stop = _exact_state_bool(vehicle_state, "emergency_stop")
        braking = _first_exact_bool(
            vehicle_state.get("brake_active"),
            vehicle_state.get("braking"),
            vehicle_state.get("emergency_braking_active"),
        )
        if (
            braking == UNKNOWN
            and speed is not None
            and previous_speed is not None
            and accel is not None
        ):
            braking = (
                accel <= params.braking_accel_mps2
                and previous_speed - speed >= params.braking_speed_drop_mps
            )
        values = {
            "braking": exact_braking,
            "yielding": _exact_state_bool(vehicle_state, "yielding"),
            "emergency_stop": exact_emergency_stop,
            "collision_contact": _exact_state_bool(vehicle_state, "collision_contact"),
            "near_miss_classified": _exact_state_bool(
                vehicle_state, "near_miss_classified"
            ),
            "minimal_risk_maneuver_active": _exact_state_bool(
                vehicle_state, "minimal_risk_maneuver_active"
            ),
            "stopped_safe": _exact_state_bool(vehicle_state, "stopped_safe"),
            "warning_active": _exact_state_bool(vehicle_state, "warning_active"),
            "safe_stop_failed": _exact_state_bool(vehicle_state, "safe_stop_failed"),
            "sensor_fault": _exact_state_bool(vehicle_state, "sensor_fault"),
            "ambulance_passage_completed": _exact_state_bool(
                vehicle_state, "ambulance_passage_completed"
            ),
            "vehicle_braking": braking,
            "vehicle_emergency_stop_active": exact_emergency_stop,
            "speed_mps": speed if speed is not None else UNKNOWN,
            "accel_mps2": accel if accel is not None else UNKNOWN,
        }
        rows.append(
            _row(
                episode_id,
                tick,
                "observable_vehicle_response",
                entity_id,
                {"vehicle": entity_id},
                values,
                [f"truth_frames.jsonl#tick={tick}#entity={entity_id}"],
                parameters=params,
            )
        )
    return rows, vehicles


def _pedestrian_rows(
    episode_id: str,
    tick: int,
    entities: Sequence[Mapping[str, Any]],
    vehicles: Mapping[str, Mapping[str, Any]],
    params: ObservableStateParameters,
    previous_pair_distance: dict[tuple[str, str], float],
    retreat_ticks: dict[tuple[str, str], int],
    previous_pedestrian_positions: dict[str, tuple[float, float, float]],
    previous_pedestrian_ticks: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pedestrian in entities:
        if _category(pedestrian) != "pedestrian":
            continue
        pedestrian_id = str(pedestrian.get("entity_id") or UNKNOWN)
        pedestrian_pos = _position(pedestrian)
        best_vehicle_id = UNKNOWN
        current_distance: float | None = None
        if pedestrian_pos is not None:
            for vehicle_id, vehicle in vehicles.items():
                vehicle_pos = _position(vehicle)
                if vehicle_pos is None:
                    continue
                distance = _distance_xy(pedestrian_pos, vehicle_pos)
                if current_distance is None or distance < current_distance:
                    best_vehicle_id, current_distance = vehicle_id, distance
        key = (pedestrian_id, best_vehicle_id)
        previous_distance = previous_pair_distance.get(key)
        pedestrian_state_mapping = _top_level_state(pedestrian, "pedestrian_state")
        pedestrian_state = pedestrian_state_mapping or {}
        exact_retreating = _exact_state_bool(pedestrian_state, "retreating")
        retreating = _first_exact_bool(pedestrian_state.get("retreating"))
        previous_pedestrian_position = previous_pedestrian_positions.get(pedestrian_id)
        previous_tick = previous_pedestrian_ticks.get(pedestrian_id)
        if (
            retreating == UNKNOWN
            and current_distance is not None
            and previous_distance is not None
            and pedestrian_pos is not None
            and previous_pedestrian_position is not None
            and best_vehicle_id in vehicles
        ):
            vehicle_pos = _position(vehicles[best_vehicle_id])
            moved_away_m = _displacement_away_from(
                previous_pedestrian_position,
                pedestrian_pos,
                vehicle_pos,
            )
            if (
                moved_away_m is not None
                and moved_away_m >= params.retreat_distance_delta_m
            ):
                retreat_ticks[key] += (
                    max(1, tick - previous_tick) if previous_tick is not None else 0
                )
            else:
                retreat_ticks[key] = 0
            retreating = retreat_ticks[key] >= params.retreat_hold_ticks
        if current_distance is not None:
            previous_pair_distance[key] = current_distance
        if pedestrian_pos is not None:
            previous_pedestrian_positions[pedestrian_id] = pedestrian_pos
            previous_pedestrian_ticks[pedestrian_id] = tick
        values = {
            "posture": _exact_state_string(pedestrian_state, "posture"),
            "health": _exact_state_string(pedestrian_state, "health"),
            "fallen": _exact_state_bool(pedestrian_state, "fallen"),
            "injured": _exact_state_bool(pedestrian_state, "injured"),
            "retreating": exact_retreating,
            "jaywalking": _exact_state_bool(pedestrian_state, "jaywalking"),
            "evacuation_active": _exact_state_bool(
                pedestrian_state, "evacuation_active"
            ),
            "safe_zone_reached": _exact_state_bool(
                pedestrian_state, "safe_zone_reached"
            ),
            "pedestrian_retreating": retreating,
            "risk_vehicle_id": best_vehicle_id,
            "risk_distance_m": (
                current_distance if current_distance is not None else UNKNOWN
            ),
        }
        rows.append(
            _row(
                episode_id,
                tick,
                "observable_pedestrian_response",
                pedestrian_id,
                {"pedestrian": pedestrian_id, "vehicle": best_vehicle_id},
                values,
                [f"truth_frames.jsonl#tick={tick}#entity={pedestrian_id}"],
                parameters=params,
            )
        )
    return rows


def _facility_rows(
    episode_id: str,
    tick: int,
    entities: Sequence[Mapping[str, Any]],
    params: ObservableStateParameters,
) -> list[dict[str, Any]]:
    pads = [entity for entity in entities if _is_pad(entity)]
    uavs = [entity for entity in entities if _category(entity) == "uav"]
    rows: list[dict[str, Any]] = []
    for pad in pads:
        pad_id = str(pad.get("entity_id") or UNKNOWN)
        pad_pos = _position(pad)
        facility_state = _structured_state(pad, "facility_state", "pad_state")
        capacity_value = _number(
            facility_state.get("capacity")
            if "capacity" in facility_state
            else _lookup_nested(pad, ("capacity", "pad_capacity"))
        )
        capacity = (
            int(capacity_value)
            if capacity_value is not None and capacity_value >= 0
            else None
        )
        requester_ids = _string_list(facility_state.get("requester_ids"))
        request_count_value = _number(facility_state.get("request_count"))
        request_count = (
            int(request_count_value)
            if request_count_value is not None and request_count_value >= 0
            else len(requester_ids)
            if requester_ids is not None
            else None
        )
        eta_pairs: list[tuple[str, float]] = []
        approaching: list[str] = []
        holding: list[str] = []
        eta_inputs_complete = pad_pos is not None
        control_modes_complete = True
        for uav in uavs:
            uav_id = str(uav.get("entity_id") or UNKNOWN)
            control_mode = _control_mode(uav)
            if control_mode == UNKNOWN:
                control_modes_complete = False
            if control_mode in {"hold", "safe_hold", "divert", "diversion"}:
                holding.append(uav_id)
            uav_pos = _position(uav)
            speed = _speed(uav)
            if pad_pos is None or uav_pos is None or speed is None:
                eta_inputs_complete = False
                continue
            if speed <= 0:
                continue
            eta = _distance_xy(pad_pos, uav_pos) / speed
            if eta <= params.pad_eta_window_ticks:
                eta_pairs.append((uav_id, eta))
                if control_mode not in {"hold", "safe_hold", "divert", "diversion"}:
                    approaching.append(uav_id)
        simultaneous_approaches = (
            len(eta_pairs) > capacity
            and _eta_overlap(eta_pairs, params.pad_eta_window_ticks)
            if capacity is not None and eta_inputs_complete
            else UNKNOWN
        )
        simultaneous_requests = (
            request_count > capacity
            if request_count is not None and capacity is not None
            else UNKNOWN
        )
        exact_priority = _exact_state_bool(facility_state, "priority_granted")
        priority = _first_exact_bool(facility_state.get("priority_granted"))
        priority_granted_to = _known_identifier(
            facility_state.get("priority_granted_to")
            or facility_state.get("arbitration_winner_id")
        )
        if priority == UNKNOWN and priority_granted_to is not None:
            priority = True
        values = {
            "simultaneous_pad_requests": simultaneous_requests,
            "simultaneous_pad_approaches": simultaneous_approaches,
            "pad_priority_granted": priority,
            "contention": _exact_state_bool(facility_state, "contention"),
            "allocation_failed": _exact_state_bool(facility_state, "allocation_failed"),
            "allocation_stale": _exact_state_bool(facility_state, "allocation_stale"),
            "arbitration_active": _exact_state_bool(
                facility_state, "arbitration_active"
            ),
            "priority_granted": exact_priority,
            "priority_granted_to": priority_granted_to or UNKNOWN,
            "backup_charger_accepted": _exact_state_bool(
                facility_state, "backup_charger_accepted"
            ),
            "availability": _exact_state_string(facility_state, "availability"),
            "fault": _exact_state_scalar(facility_state, "fault"),
            "reserved": _exact_state_bool(facility_state, "reserved"),
            "inference_class": "physical_approach_inference",
            "capacity": capacity if capacity is not None else UNKNOWN,
            "request_count": request_count if request_count is not None else UNKNOWN,
            "requester_ids": requester_ids if requester_ids is not None else UNKNOWN,
            "approaching_uav_ids": (
                sorted(approaching)
                if eta_inputs_complete and control_modes_complete
                else UNKNOWN
            ),
            "holding_or_diverting_uav_ids": (
                sorted(holding) if control_modes_complete else UNKNOWN
            ),
            "eta_inputs_complete": eta_inputs_complete,
            "control_modes_complete": control_modes_complete,
            "eta_window_ticks": params.pad_eta_window_ticks,
        }
        rows.append(
            _row(
                episode_id,
                tick,
                "observable_pad_facility",
                pad_id,
                {"pad": pad_id},
                values,
                [f"truth_frames.jsonl#tick={tick}#entity={pad_id}"],
                source_class="mixed_observed_and_physical_inference",
                parameters=params,
            )
        )
    return rows


def _structured_runtime_rows(
    episode_id: str,
    tick: int,
    entities: Sequence[Mapping[str, Any]],
    params: ObservableStateParameters,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs: tuple[tuple[str, str, tuple[str, ...], set[str]], ...] = (
        ("mission_state", "observable_mission_state", _MISSION_BOOL_FIELDS, {"uav"}),
        (
            "navigation_state",
            "observable_navigation_state",
            _NAVIGATION_BOOL_FIELDS,
            {"uav"},
        ),
        (
            "communication_state",
            "observable_communication_state",
            _COMMUNICATION_BOOL_FIELDS,
            {"uav", "vehicle"},
        ),
        (
            "incident_state",
            "observable_incident_state",
            _INCIDENT_BOOL_FIELDS,
            {"uav", "vehicle", "pedestrian", "facility"},
        ),
        (
            "control_state",
            "observable_control_state",
            _CONTROL_BOOL_FIELDS,
            {"uav", "vehicle"},
        ),
        (
            "security_state",
            "observable_security_state",
            _SECURITY_BOOL_FIELDS,
            {"uav", "vehicle"},
        ),
    )
    for entity in entities:
        entity_id = str(entity.get("entity_id") or UNKNOWN)
        category = _category(entity)
        for state_key, family, bool_fields, expected_categories in specs:
            state = _top_level_state(entity, state_key)
            if state is None and category not in expected_categories:
                continue
            values = {
                field: _exact_state_bool(state or {}, field) for field in bool_fields
            }
            rows.append(
                _row(
                    episode_id,
                    tick,
                    family,
                    entity_id,
                    {"entity": entity_id},
                    values,
                    [
                        f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path={state_key}"
                    ],
                    parameters=params,
                )
            )
        incident_state = _top_level_state(entity, "incident_state")
        if incident_state is not None:
            rows.append(
                _row(
                    episode_id,
                    tick,
                    "lockdown_control",
                    entity_id,
                    {"entity": entity_id},
                    {
                        "clear": _exact_state_bool(
                            incident_state,
                            "lockdown_clear",
                        ),
                        "requires_reroute": _exact_state_bool(
                            incident_state,
                            "requires_reroute",
                        ),
                    },
                    [
                        f"truth_frames.jsonl#tick={tick}#entity={entity_id}#path=incident_state"
                    ],
                    parameters=params,
                )
            )
    return rows


def _row(
    episode_id: str,
    tick: int,
    family: str,
    subject_id: str,
    bindings: Mapping[str, str],
    values: Mapping[str, Any],
    source_refs: Sequence[str],
    *,
    source_class: str = "derived_from_observed",
    parameters: ObservableStateParameters | None = None,
) -> dict[str, Any]:
    missing_inputs = sorted(key for key, value in values.items() if value == UNKNOWN)
    parameter_digest = digest_object(
        {
            "rule_version": RULE_VERSION,
            "family": family,
            "parameters": asdict(parameters) if parameters is not None else {},
        }
    )
    return {
        "schema_name": "domain_state_observation",
        "schema_version": SCHEMA_VERSION,
        "observation_id": stable_identifier(
            "observable_state",
            episode_id,
            tick,
            family,
            subject_id,
            dict(sorted(bindings.items())),
            values,
        ),
        "episode_id": episode_id,
        "tick": tick,
        "observation_family": family,
        "family": family,
        "subject_id": subject_id,
        "bindings": dict(sorted(bindings.items())),
        "source_class": source_class,
        "rule_id": f"observable_state_completion.{family}",
        "rule_version": RULE_VERSION,
        "parameter_digest": parameter_digest,
        "input_digest": digest_object(
            {"source_refs": sorted(source_refs), "values": values}
        ),
        "values": dict(values),
        "source_refs": sorted(set(source_refs)),
        "missing_inputs": missing_inputs,
    }


def _parameters(
    parameters: ObservableStateParameters | Mapping[str, Any] | None,
) -> ObservableStateParameters:
    if parameters is None:
        return ObservableStateParameters()
    if isinstance(parameters, ObservableStateParameters):
        return parameters
    current = ObservableStateParameters()
    for key, value in parameters.items():
        if hasattr(current, str(key)):
            current = replace(current, **{str(key): value})
    return current


def _episode_id(episode_root: Path, frames: Sequence[Mapping[str, Any]]) -> str:
    for frame in frames:
        value = frame.get("episode_id")
        if isinstance(value, str) and value:
            return value
    return episode_root.name


def _category(entity: Mapping[str, Any]) -> str:
    for key in ("entity_category", "category", "entity_kind", "entity_type"):
        raw = _exact_string(entity.get(key))
        if raw in _UAV_CATEGORIES:
            return "uav"
        if raw in _PEDESTRIAN_CATEGORIES:
            return "pedestrian"
        if raw in _VEHICLE_CATEGORIES:
            return "vehicle"
        if raw in _FACILITY_CATEGORIES:
            return "facility"
    return UNKNOWN


def _is_pad(entity: Mapping[str, Any]) -> bool:
    return any(
        _exact_string(entity.get(key)) in _PAD_CATEGORIES
        for key in ("entity_category", "category", "entity_kind", "entity_type")
    )


def _structured_state(entity: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    return _top_level_state(entity, *keys) or {}


def _top_level_state(entity: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    for key in keys:
        value = entity.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _first_exact_bool(*values: Any) -> bool | str:
    for value in values:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
    return UNKNOWN


def _first_known(*values: Any) -> Any:
    for value in values:
        if value != UNKNOWN:
            return value
    return UNKNOWN


def _aggregate_exact_bool(values: Sequence[Any]) -> bool | str:
    known = [value for value in values if value != UNKNOWN]
    if any(value is True for value in known):
        return True
    if known and all(value is False for value in known):
        return False
    return UNKNOWN


def _exact_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized or normalized in {"unknown", "none", "null"}:
        return None
    return normalized


def _exact_state_bool(state: Mapping[str, Any], key: str) -> bool | str:
    if key not in state:
        return UNKNOWN
    return _first_exact_bool(state.get(key))


def _exact_state_string(state: Mapping[str, Any], key: str) -> str:
    if key not in state:
        return UNKNOWN
    return _exact_string(state.get(key)) or UNKNOWN


def _exact_state_scalar(state: Mapping[str, Any], key: str) -> bool | float | str:
    if key not in state:
        return UNKNOWN
    value = state.get(key)
    if isinstance(value, bool):
        return value
    number = _number(value)
    if number is not None:
        return number
    return _exact_string(value) or UNKNOWN


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result = [
        item for item in (_known_identifier(raw) for raw in value) if item is not None
    ]
    return sorted(set(result))


def _known_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"unknown", "none", "null", "false"}:
        return None
    return stripped


def _control_mode(entity: Mapping[str, Any]) -> str | None:
    state = _structured_state(entity, "control_state")
    return _exact_string(state.get("mode"))


def _first_mapping(
    mapping: Mapping[str, Any], keys: Sequence[str]
) -> Mapping[str, Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return value
    return mapping


def _first_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _lookup_nested(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    annotations = mapping.get("annotations")
    if isinstance(annotations, Mapping):
        for key in keys:
            if key in annotations:
                return annotations[key]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _number_or_unknown(value: Any) -> float | str:
    number = _number(value)
    return number if number is not None else UNKNOWN


def _position(entity: Mapping[str, Any]) -> tuple[float, float, float] | None:
    pose = (
        entity.get("truth_pose")
        if isinstance(entity.get("truth_pose"), Mapping)
        else entity
    )
    value = pose.get("position_enu_m") or pose.get("position")
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 3
    ):
        nums = [_number(item) for item in value[:3]]
        if all(item is not None for item in nums):
            return (float(nums[0]), float(nums[1]), float(nums[2]))
    return None


def _speed(entity: Mapping[str, Any] | None) -> float | None:
    if entity is None:
        return None
    annotations = (
        entity.get("annotations")
        if isinstance(entity.get("annotations"), Mapping)
        else {}
    )
    direct = _number(
        annotations.get("speed_mps")
        if "speed_mps" in annotations
        else entity.get("speed_mps")
    )
    if direct is not None:
        return direct
    pose = (
        entity.get("truth_pose")
        if isinstance(entity.get("truth_pose"), Mapping)
        else {}
    )
    velocity = pose.get("velocity_enu_mps")
    if (
        isinstance(velocity, Sequence)
        and not isinstance(velocity, (str, bytes))
        and len(velocity) >= 2
    ):
        nums = [_number(item) for item in velocity[:3]]
        if all(item is not None for item in nums):
            return math.sqrt(sum(float(item) * float(item) for item in nums))
    return None


def _accel(entity: Mapping[str, Any]) -> float | None:
    annotations = (
        entity.get("annotations")
        if isinstance(entity.get("annotations"), Mapping)
        else {}
    )
    vehicle_state = _structured_state(entity, "vehicle_state")
    sumo_vehicle = (
        entity.get("sumo_vehicle")
        if isinstance(entity.get("sumo_vehicle"), Mapping)
        else {}
    )
    for value in (
        vehicle_state.get("accel_mps2"),
        sumo_vehicle.get("accel_mps2"),
        annotations.get("accel_mps2"),
        entity.get("accel_mps2"),
    ):
        number = _number(value)
        if number is not None:
            return number
    return None


def _distance_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _displacement_away_from(
    previous: tuple[float, float, float],
    current: tuple[float, float, float],
    hazard: tuple[float, float, float] | None,
) -> float | None:
    if hazard is None:
        return None
    away_x = current[0] - hazard[0]
    away_y = current[1] - hazard[1]
    norm = math.hypot(away_x, away_y)
    if norm <= 1e-9:
        return None
    displacement_x = current[0] - previous[0]
    displacement_y = current[1] - previous[1]
    return (displacement_x * away_x + displacement_y * away_y) / norm


def _hysteresis(
    key: str,
    value: float | None,
    on: float,
    off: float,
    latches: dict[str, bool],
    *,
    high_is_active: bool,
) -> bool | str:
    if value is None:
        return UNKNOWN
    active = bool(latches.get(key, False))
    if high_is_active:
        active = True if value >= on else False if value <= off else active
    else:
        active = True if value <= on else False if value >= off else active
    latches[key] = active
    return active


def _eta_overlap(eta_pairs: Sequence[tuple[str, float]], window: int) -> bool:
    if len(eta_pairs) < 2:
        return False
    values = sorted(eta for _, eta in eta_pairs)
    return values[-1] - values[0] <= window
