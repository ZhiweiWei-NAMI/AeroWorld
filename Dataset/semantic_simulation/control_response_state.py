"""Observed control-response state for objective semantic projection.

The rows produced here are state observations, not event answers.  The module
uses only sampled truth-frame structured state mappings, route geometry, and
already-governed communication/domain rows supplied by callers.  It does not
read authored event traces, scenario plans, dynamic labels, expected labels,
activity prose, or EPI names to set a semantic value.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Dataset.semantic_truth.provenance import (
    digest_object,
    read_jsonl,
    stable_identifier,
)


SCHEMA_VERSION = "1.0.0"
RULE_VERSION = "1.4.0"
UNKNOWN = "unknown"
CONTROL_FAMILY = "control_response_state"

SUPPORTED_API_VALUE_KEYS = {
    "communication.backup_link_active": "backup_link_active",
    "communication.station_unavailable": "station_unavailable",
    "control.alternate_reroute_active": "alternate_reroute_active",
    "control.alternate_rth_active": "alternate_rth_active",
    "control.altitude_corrective_maneuver_active": "altitude_corrective_maneuver_active",
    "control.deconfliction_active": "deconfliction_active",
    "control.diversion_active": "diversion_active",
    "control.divert_active": "divert_active",
    "control.evasion_active": "evasion_active",
    "control.hold_active": "hold_active",
    "control.pull_up_active": "pull_up_active",
    "control.reroute_active": "reroute_active",
    "control.resequence_active": "resequence_active",
    "control.rth_active": "rth_active",
    "control.safe_hold_active": "safe_hold_active",
    "control.slowdown_active": "slowdown_active",
    "mission.abort_active": "mission_abort_active",
    "navigation.path_deviation": "path_deviation",
    "navigation.route_recovered": "route_recovered",
    "navigation.sustained_drift": "sustained_drift",
    "security.command_integrity_violation": "command_integrity_violation",
    "security.command_lockout_active": "command_lockout",
    "security.gcs_compromised": "gcs_compromised",
    "security.jamming": "jamming_active",
    "security.unauthorized_command": "unauthorized_command",
}

FORBIDDEN_INPUTS = {
    "dynamic_labels.jsonl",
    "event_realization.jsonl",
    "event_trace.jsonl",
    "expected_event",
    "scenario_plan.json",
}

PATH_DEVIATION_THRESHOLD_M = 15.0
ROUTE_RECOVERY_THRESHOLD_M = 5.0
SUSTAINED_DRIFT_TICKS = 10
ALTITUDE_BAND_M = 5.0
SLOWDOWN_DELTA_MPS = 0.5
RTH_LEFT_HOME_DISTANCE_M = 15.0
RTH_HOME_ARRIVAL_DISTANCE_M = 8.0
RTH_ARRIVAL_SPEED_MPS = 0.75
RTH_MIN_HOMEWARD_PROGRESS_M = 1.0
RTH_MIN_HOMEWARD_PROGRESS_SAMPLES = 2
RTH_AWAY_MOVEMENT_TOLERANCE_M = 2.0
RTH_FORMAL_SAMPLE_STEP_TICKS = 5


class ControlResponseStateError(RuntimeError):
    """Raised when control-response state cannot be derived safely."""


@dataclass(frozen=True)
class _EntitySample:
    tick: int
    entity_id: str
    category: str
    kind: str
    control_state: Mapping[str, Any]
    mission_state: Mapping[str, Any]
    security_state: Mapping[str, Any]
    communication_state: Mapping[str, Any]
    position: tuple[float, float, float] | None
    velocity: tuple[float, float, float] | None
    speed_mps: float | None
    assigned_altitude_m: float | None
    route_waypoints: tuple[tuple[float, float, float], ...]


CONTROL_MODE_VALUES = {
    "rth": "rth_active",
    "return_to_home": "rth_active",
    "alternate_rth": "alternate_rth_active",
    "alternate_return_to_home": "alternate_rth_active",
    "reroute": "reroute_active",
    "alternate_reroute": "alternate_reroute_active",
    "alternate_route": "alternate_reroute_active",
    "altitude_correction": "altitude_corrective_maneuver_active",
    "altitude_corrective_maneuver": "altitude_corrective_maneuver_active",
    "deconfliction": "deconfliction_active",
    "evasion": "evasion_active",
    "pull_up": "pull_up_active",
    "safe_hold": "safe_hold_active",
    "hold": "hold_active",
    "resequence": "resequence_active",
    "divert": "divert_active",
    "diversion": "diversion_active",
    "slowdown": "slowdown_active",
}

MISSION_MODE_VALUES = {
    "abort": "mission_abort_active",
    "mission_abort": "mission_abort_active",
}

SECURITY_MODE_VALUES = {
    "gcs_compromised": "gcs_compromised",
    "jamming": "jamming_active",
    "unauthorized_command": "unauthorized_command",
    "command_integrity_violation": "command_integrity_violation",
    "command_lockout": "command_lockout",
}

INACTIVE_CONTROL_MODES = {"nominal", "normal", "none", "idle"}
INACTIVE_MISSION_MODES = {"nominal", "normal", "active", "none"}
INACTIVE_SECURITY_MODES = {"nominal", "normal", "clear", "none"}
INACTIVE_LINK_MODES = {"primary", "nominal", "normal", "main"}
ACTIVE_BACKUP_LINK_MODES = {"backup", "backup_link", "alternate_link", "handover"}
AVAILABLE_STATES = {"available", "online", "up", "nominal", "normal"}
UNAVAILABLE_STATES = {"unavailable", "offline", "down", "failed", "lost"}


def build_control_response_state_rows(
    episode_root: Path,
    *,
    communication_rows: Sequence[Mapping[str, Any]] = (),
    domain_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build deterministic control-response state observations for one episode."""

    episode_root = episode_root.resolve()
    truth_path = episode_root / "truth_frames.jsonl"
    manifest_path = episode_root / "episode_manifest.json"
    if not truth_path.is_file():
        raise ControlResponseStateError(f"truth_frames.jsonl is missing: {truth_path}")
    manifest = _load_manifest(manifest_path)
    episode_id = str(manifest.get("episode_id") or episode_root.name)
    if episode_id != episode_root.name:
        raise ControlResponseStateError(
            f"episode manifest id {episode_id} does not match directory {episode_root.name}"
        )

    communication_index = _index_by_tick_entity(
        communication_rows, episode_id=episode_id
    )
    domain_index = _index_by_tick_entity(domain_rows, episode_id=episode_id)
    previous_by_entity: dict[str, _EntitySample] = {}
    state_by_entity: dict[str, dict[str, Any]] = defaultdict(dict)
    rows: list[dict[str, Any]] = []

    for frame in sorted(read_jsonl(truth_path), key=lambda item: int(item["tick"])):
        tick = frame.get("tick")
        if not isinstance(tick, int):
            raise ControlResponseStateError(
                f"{truth_path}: truth frame lacks integer tick"
            )
        samples = [
            _sample_entity(tick, entity)
            for entity in frame.get("entities", ())
            if isinstance(entity, Mapping)
        ]
        for sample in samples:
            if sample.category not in {
                "uav",
                "ground_station",
                "communication_station",
            }:
                continue
            previous = previous_by_entity.get(sample.entity_id)
            communication = communication_index.get((tick, sample.entity_id), ())
            domain = domain_index.get((tick, sample.entity_id), ())
            values = _derive_values(
                sample,
                previous,
                state_by_entity[sample.entity_id],
                communication,
                domain,
            )
            rows.append(
                _observation_row(
                    episode_id=episode_id,
                    tick=tick,
                    subject_id=sample.entity_id,
                    subject_kind=sample.category or sample.kind or UNKNOWN,
                    values=values,
                    input_digest=digest_object(
                        {
                            "truth": _sample_digest_projection(sample),
                            "communication": [
                                _row_digest_projection(row) for row in communication
                            ],
                            "domain": [_row_digest_projection(row) for row in domain],
                        }
                    ),
                    source_refs=_source_refs(tick, sample, communication, domain),
                )
            )
            previous_by_entity[sample.entity_id] = sample
    rows.sort(key=lambda row: (row["tick"], row["subject_id"], row["observation_id"]))
    return rows


def _derive_values(
    sample: _EntitySample,
    previous: _EntitySample | None,
    state: dict[str, Any],
    communication: Sequence[Mapping[str, Any]],
    domain: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    communication_values = tuple(_values(row) for row in communication)
    domain_values = tuple(_values(row) for row in domain)
    domain_control_states = tuple(
        _mapping(values.get("control_state")) for values in domain_values
    )
    domain_mission_states = tuple(
        _mapping(values.get("mission_state")) for values in domain_values
    )
    domain_security_states = tuple(
        _mapping(values.get("security_state")) for values in domain_values
    )
    domain_communication_states = tuple(
        _mapping(values.get("communication_state")) for values in domain_values
    )
    domain_control_api_states = tuple(
        _exact_api_fields(values, "control.") for values in domain_values
    )
    domain_mission_api_states = tuple(
        _exact_api_fields(values, "mission.") for values in domain_values
    )
    domain_security_api_states = tuple(
        _exact_api_fields(values, "security.") for values in domain_values
    )
    domain_communication_api_states = tuple(
        _exact_api_fields(values, "communication.") for values in domain_values
    )
    row_communication_states = tuple(
        _mapping(values.get("communication_state")) for values in communication_values
    )
    row_security_states = tuple(
        _mapping(values.get("security_state")) for values in communication_values
    )

    # Control, mission, and security truth may only come from their explicitly
    # typed structured-state families.  A generic domain row can contain keys
    # such as ``status`` or ``mode`` for an unrelated observation family; using
    # the entire values mapping here would let that unrelated row masquerade as
    # a controller response.
    control_sources = (
        sample.control_state,
        *domain_control_states,
        *domain_control_api_states,
    )
    mission_sources = (
        sample.mission_state,
        *domain_mission_states,
        *domain_mission_api_states,
    )
    security_sources = (
        sample.security_state,
        *domain_security_states,
        *row_security_states,
        *domain_security_api_states,
    )
    communication_sources = (
        sample.communication_state,
        *row_communication_states,
        *communication_values,
        *domain_communication_states,
        *domain_communication_api_states,
    )
    route_distance = _route_distance(sample)
    previous_route_distance = (
        _route_distance(previous) if previous is not None else None
    )
    home_distance = _home_distance(sample)
    previous_home_distance = _home_distance(previous) if previous is not None else None
    structured_rth_active = _control_bool("rth_active", control_sources)
    kinematic_rth_active = _kinematic_rth_bool(
        sample,
        state,
        home_distance=home_distance,
    )
    rth_active = structured_rth_active
    path_deviation = (
        route_distance > PATH_DEVIATION_THRESHOLD_M
        if route_distance is not None
        else UNKNOWN
    )
    if path_deviation is True:
        state["path_deviation_seen"] = True
        elapsed_ticks = (
            max(1, sample.tick - previous.tick) if previous is not None else 0
        )
        state["sustained_drift_ticks"] = (
            int(state.get("sustained_drift_ticks", 0)) + elapsed_ticks
        )
    elif path_deviation is False:
        state["sustained_drift_ticks"] = 0
    if not state.get("path_deviation_seen", False):
        route_recovered: bool | str = UNKNOWN
    elif route_distance is None:
        route_recovered = UNKNOWN
    else:
        route_recovered = route_distance <= ROUTE_RECOVERY_THRESHOLD_M
    sustained_drift = (
        int(state.get("sustained_drift_ticks", 0)) >= SUSTAINED_DRIFT_TICKS
        if path_deviation != UNKNOWN
        else UNKNOWN
    )

    speed_delta = (
        previous.speed_mps - sample.speed_mps
        if previous is not None
        and previous.speed_mps is not None
        and sample.speed_mps is not None
        else None
    )
    altitude_error = (
        sample.position[2] - sample.assigned_altitude_m
        if sample.position is not None and sample.assigned_altitude_m is not None
        else None
    )
    vertical_speed = sample.velocity[2] if sample.velocity is not None else None

    return {
        "supported_api_value_keys": dict(sorted(SUPPORTED_API_VALUE_KEYS.items())),
        "rth_active": rth_active,
        "kinematic_rth_active": kinematic_rth_active,
        "alternate_rth_active": _control_bool("alternate_rth_active", control_sources),
        "reroute_active": _control_bool("reroute_active", control_sources),
        "alternate_reroute_active": _control_bool(
            "alternate_reroute_active", control_sources
        ),
        "altitude_corrective_maneuver_active": _first_known(
            _control_bool("altitude_corrective_maneuver_active", control_sources),
            _altitude_correction(altitude_error, vertical_speed),
        ),
        "deconfliction_active": _control_bool("deconfliction_active", control_sources),
        "evasion_active": _control_bool("evasion_active", control_sources),
        "pull_up_active": _control_bool("pull_up_active", control_sources),
        "safe_hold_active": _control_bool("safe_hold_active", control_sources),
        "hold_active": _control_bool("hold_active", control_sources),
        "resequence_active": _control_bool("resequence_active", control_sources),
        "divert_active": _control_bool("divert_active", control_sources),
        "diversion_active": _control_bool("diversion_active", control_sources),
        "slowdown_active": _first_bool(
            _control_bool("slowdown_active", control_sources),
            speed_delta >= SLOWDOWN_DELTA_MPS if speed_delta is not None else UNKNOWN,
        ),
        "mission_abort_active": _mission_bool("mission_abort_active", mission_sources),
        "station_unavailable": _communication_unavailable(communication_sources),
        "backup_link_active": _backup_link_active(communication_sources),
        "path_deviation": path_deviation,
        "route_recovered": route_recovered,
        "sustained_drift": sustained_drift,
        "route_distance_m": _round(route_distance),
        "previous_route_distance_m": _round(previous_route_distance),
        "home_distance_m": _round(home_distance),
        "previous_home_distance_m": _round(previous_home_distance),
        "speed_delta_mps": _round(speed_delta),
        "altitude_error_m": _round(altitude_error),
        "gcs_compromised": _security_bool("gcs_compromised", security_sources),
        "jamming_active": _security_bool("jamming_active", security_sources),
        "unauthorized_command": _security_bool(
            "unauthorized_command", security_sources
        ),
        "command_integrity_violation": _security_bool(
            "command_integrity_violation", security_sources
        ),
        "command_lockout": _security_bool("command_lockout", security_sources),
    }


def _observation_row(
    *,
    episode_id: str,
    tick: int,
    subject_id: str,
    subject_kind: str,
    values: Mapping[str, Any],
    input_digest: str,
    source_refs: Sequence[str],
) -> dict[str, Any]:
    parameter_digest = digest_object(
        {
            "path_deviation_threshold_m": PATH_DEVIATION_THRESHOLD_M,
            "route_recovery_threshold_m": ROUTE_RECOVERY_THRESHOLD_M,
            "sustained_drift_ticks": SUSTAINED_DRIFT_TICKS,
            "altitude_band_m": ALTITUDE_BAND_M,
            "slowdown_delta_mps": SLOWDOWN_DELTA_MPS,
            "rth_left_home_distance_m": RTH_LEFT_HOME_DISTANCE_M,
            "rth_home_arrival_distance_m": RTH_HOME_ARRIVAL_DISTANCE_M,
            "rth_arrival_speed_mps": RTH_ARRIVAL_SPEED_MPS,
            "rth_min_homeward_progress_m": RTH_MIN_HOMEWARD_PROGRESS_M,
            "rth_min_homeward_progress_samples": RTH_MIN_HOMEWARD_PROGRESS_SAMPLES,
            "rth_away_movement_tolerance_m": RTH_AWAY_MOVEMENT_TOLERANCE_M,
            "rth_formal_sample_step_ticks": RTH_FORMAL_SAMPLE_STEP_TICKS,
            "supported_api_value_keys": SUPPORTED_API_VALUE_KEYS,
        }
    )
    observation_id = stable_identifier(
        "domain_state_observation",
        CONTROL_FAMILY,
        episode_id,
        tick,
        subject_id,
        values,
        parameter_digest,
    )
    return {
        "schema_name": "domain_state_observation",
        "schema_version": SCHEMA_VERSION,
        "observation_id": observation_id,
        "episode_id": episode_id,
        "tick": tick,
        "observation_family": CONTROL_FAMILY,
        "subject_id": subject_id,
        "subject_kind": subject_kind,
        "source_class": "derived_from_observed",
        "model_id": "control_response_state.structured_truth_route_comm_kinematic_rth_v5",
        "model_version": RULE_VERSION,
        "rule_id": "control_response_state.structured_truth_route_communication_mapping",
        "rule_version": RULE_VERSION,
        "input_digest": input_digest,
        "parameter_digest": parameter_digest,
        "values": dict(values),
        "source_refs": sorted(set(source_refs)),
    }


def _sample_entity(tick: int, entity: Mapping[str, Any]) -> _EntitySample:
    pose = (
        entity.get("truth_pose")
        if isinstance(entity.get("truth_pose"), Mapping)
        else {}
    )
    annotations = (
        entity.get("annotations")
        if isinstance(entity.get("annotations"), Mapping)
        else {}
    )
    position = _vector3(pose.get("position_enu_m"))
    velocity = _vector3(pose.get("velocity_enu_mps"))
    speed = _number(annotations.get("speed_mps"))
    if speed is None and velocity is not None:
        speed = math.sqrt(sum(item * item for item in velocity))
    return _EntitySample(
        tick=tick,
        entity_id=str(entity.get("entity_id") or UNKNOWN),
        category=str(
            entity.get("entity_category") or entity.get("category") or UNKNOWN
        ),
        kind=str(entity.get("entity_kind") or entity.get("entity_type") or UNKNOWN),
        control_state=_mapping(entity.get("control_state")),
        mission_state=_mapping(entity.get("mission_state")),
        security_state=_mapping(entity.get("security_state")),
        communication_state=_mapping(entity.get("communication_state")),
        position=position,
        velocity=velocity,
        speed_mps=speed,
        assigned_altitude_m=_assigned_altitude(entity),
        route_waypoints=tuple(
            item
            for item in (
                _vector3(value)
                for value in entity.get("planned_route_waypoints_enu_m", ())
            )
            if item is not None
        ),
    )


def _assigned_altitude(entity: Mapping[str, Any]) -> float | None:
    direct = _number(entity.get("assigned_altitude_m"))
    if direct is not None:
        return direct
    corridor = entity.get("uav_corridor")
    if isinstance(corridor, Mapping):
        return _number(corridor.get("assigned_altitude_m"))
    return None


def _route_distance(sample: _EntitySample | None) -> float | None:
    if sample is None or sample.position is None or len(sample.route_waypoints) < 2:
        return None
    point = sample.position
    return min(
        _point_segment_distance_xy(point, start, end)
        for start, end in zip(sample.route_waypoints[:-1], sample.route_waypoints[1:])
    )


def _home_distance(sample: _EntitySample | None) -> float | None:
    if sample is None or sample.position is None or not sample.route_waypoints:
        return None
    home = sample.route_waypoints[0]
    return math.dist((sample.position[0], sample.position[1]), (home[0], home[1]))


def _kinematic_rth_bool(
    sample: _EntitySample,
    state: dict[str, Any],
    *,
    home_distance: float | None,
) -> bool | str:
    if home_distance is None:
        state["kinematic_rth_homeward_progress_samples"] = 0
        state.pop("kinematic_rth_reference_tick", None)
        state.pop("kinematic_rth_reference_home_distance", None)
        state["kinematic_rth_arrived_home"] = False
        return UNKNOWN

    if home_distance >= RTH_LEFT_HOME_DISTANCE_M:
        state["kinematic_rth_left_home"] = True
        state["kinematic_rth_arrived_home"] = False

    if (
        state.get("kinematic_rth_arrived_home") is True
        and home_distance <= RTH_HOME_ARRIVAL_DISTANCE_M
    ):
        state["kinematic_rth_active"] = False
        state["kinematic_rth_left_home"] = False
        state["kinematic_rth_homeward_progress_samples"] = 0
        state["kinematic_rth_reference_tick"] = sample.tick
        state["kinematic_rth_reference_home_distance"] = home_distance
        return False

    was_active = state.get("kinematic_rth_active") is True
    if (
        was_active
        and home_distance <= RTH_HOME_ARRIVAL_DISTANCE_M
        and sample.speed_mps is not None
        and sample.speed_mps <= RTH_ARRIVAL_SPEED_MPS
    ):
        state["kinematic_rth_active"] = False
        state["kinematic_rth_left_home"] = False
        state["kinematic_rth_arrived_home"] = True
        state["kinematic_rth_homeward_progress_samples"] = 0
        state["kinematic_rth_reference_tick"] = sample.tick
        state["kinematic_rth_reference_home_distance"] = home_distance
        return False

    reference_tick = state.get("kinematic_rth_reference_tick")
    reference_home_distance = state.get("kinematic_rth_reference_home_distance")
    if not isinstance(reference_tick, int) or not isinstance(
        reference_home_distance, (int, float)
    ):
        state["kinematic_rth_reference_tick"] = sample.tick
        state["kinematic_rth_reference_home_distance"] = home_distance
        state["kinematic_rth_homeward_progress_samples"] = 0
        return True if was_active else UNKNOWN

    if not state.get("kinematic_rth_left_home", False):
        state["kinematic_rth_reference_tick"] = sample.tick
        state["kinematic_rth_reference_home_distance"] = home_distance
        state["kinematic_rth_homeward_progress_samples"] = 0
        return True if was_active else UNKNOWN

    elapsed_ticks = sample.tick - reference_tick
    if elapsed_ticks < RTH_FORMAL_SAMPLE_STEP_TICKS:
        return True if was_active else UNKNOWN

    progress_m = float(reference_home_distance) - home_distance
    state["kinematic_rth_reference_tick"] = sample.tick
    state["kinematic_rth_reference_home_distance"] = home_distance
    if progress_m >= RTH_MIN_HOMEWARD_PROGRESS_M:
        progress_samples = (
            int(state.get("kinematic_rth_homeward_progress_samples", 0)) + 1
        )
        state["kinematic_rth_homeward_progress_samples"] = progress_samples
        if progress_samples >= RTH_MIN_HOMEWARD_PROGRESS_SAMPLES:
            state["kinematic_rth_active"] = True
            return True
        state["kinematic_rth_active"] = False
        return True if was_active else False

    state["kinematic_rth_homeward_progress_samples"] = 0
    if progress_m <= -RTH_AWAY_MOVEMENT_TOLERANCE_M:
        state["kinematic_rth_active"] = False
        return False
    state["kinematic_rth_active"] = False
    return False


def _point_segment_distance_xy(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    px, py = point[0], point[1]
    sx, sy = start[0], start[1]
    ex, ey = end[0], end[1]
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return math.dist((px, py), (sx, sy))
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    return math.dist((px, py), (sx + t * dx, sy + t * dy))


def _index_by_tick_entity(
    rows: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
) -> dict[tuple[int, str], tuple[Mapping[str, Any], ...]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("episode_id") not in {None, "", episode_id}:
            continue
        tick = row.get("tick")
        entity_id = row.get("entity_id") or row.get("subject_id")
        if isinstance(tick, int) and isinstance(entity_id, str):
            grouped[(tick, entity_id)].append(row)
    return {
        key: tuple(
            sorted(
                values,
                key=lambda row: (
                    str(
                        row.get("observation_family")
                        or row.get("record_family")
                        or row.get("schema_name")
                        or ""
                    ),
                    str(row.get("observation_id") or row.get("message_id") or ""),
                ),
            )
        )
        for key, values in grouped.items()
    }


def _control_bool(output_key: str, sources: Iterable[Mapping[str, Any]]) -> bool | str:
    return _structured_state_bool(
        output_key,
        sources,
        enum_values=CONTROL_MODE_VALUES,
        inactive_modes=INACTIVE_CONTROL_MODES,
        enum_fields=("mode", "active_mode", "command_mode"),
    )


def _mission_bool(output_key: str, sources: Iterable[Mapping[str, Any]]) -> bool | str:
    return _structured_state_bool(
        output_key,
        sources,
        enum_values=MISSION_MODE_VALUES,
        inactive_modes=INACTIVE_MISSION_MODES,
        enum_fields=("mode", "status", "phase"),
    )


def _security_bool(output_key: str, sources: Iterable[Mapping[str, Any]]) -> bool | str:
    return _structured_state_bool(
        output_key,
        sources,
        enum_values=SECURITY_MODE_VALUES,
        inactive_modes=INACTIVE_SECURITY_MODES,
        enum_fields=("mode", "status", "threat", "condition"),
    )


def _structured_state_bool(
    output_key: str,
    sources: Iterable[Mapping[str, Any]],
    *,
    enum_values: Mapping[str, str],
    inactive_modes: set[str],
    enum_fields: Sequence[str],
) -> bool | str:
    for source in sources:
        explicit = _source_bool(
            source,
            output_key,
            _api_key(output_key),
        )
        if explicit != UNKNOWN:
            return explicit
        command = _active_command_bool(output_key, source, enum_values)
        enum = _mode_bool(output_key, source, enum_values, inactive_modes, enum_fields)
        if command is True or enum is True:
            return True
        if command is False or enum is False:
            return False
    return UNKNOWN


def _communication_unavailable(sources: Iterable[Mapping[str, Any]]) -> bool | str:
    for source in sources:
        explicit = _source_bool(
            source,
            "station_unavailable",
            "communication.station_unavailable",
            "communication_unavailable",
        )
        if explicit != UNKNOWN:
            return explicit
        for key in ("available", "link_available", "station_available"):
            if isinstance(source.get(key), bool):
                return not source[key]
        for key in ("availability", "link_availability"):
            value = _number(source.get(key))
            if value is not None:
                return value <= 0.0
        for key in (
            "status",
            "station_status",
            "link_status",
            "availability_state",
            "link_state",
        ):
            enum = _canonical_enum(source.get(key))
            if enum in UNAVAILABLE_STATES:
                return True
            if enum in AVAILABLE_STATES:
                return False
    return UNKNOWN


def _backup_link_active(sources: Iterable[Mapping[str, Any]]) -> bool | str:
    for source in sources:
        explicit = _source_bool(
            source,
            "backup_link_active",
            "communication.backup_link_active",
            "backup_active",
            "alternate_link_active",
            "handover_active",
        )
        if explicit != UNKNOWN:
            return explicit
        for key in ("mode", "link_mode", "channel_mode", "handover_mode"):
            enum = _canonical_enum(source.get(key))
            if enum in ACTIVE_BACKUP_LINK_MODES:
                return True
            if enum in INACTIVE_LINK_MODES:
                return False
    return UNKNOWN


def _altitude_correction(
    altitude_error: float | None,
    vertical_speed: float | None,
) -> bool | str:
    if altitude_error is None or vertical_speed is None:
        return UNKNOWN
    if abs(altitude_error) <= ALTITUDE_BAND_M:
        return False
    return (altitude_error > 0 and vertical_speed < 0) or (
        altitude_error < 0 and vertical_speed > 0
    )


def _source_bool(source: Mapping[str, Any], *keys: str | None) -> bool | str:
    for key in keys:
        if key is not None and isinstance(source.get(key), bool):
            return source[key]
    return UNKNOWN


def _active_command_bool(
    output_key: str,
    source: Mapping[str, Any],
    enum_values: Mapping[str, str],
) -> bool | str:
    command_keys = ("active_commands", "active_command", "commands")
    for key in command_keys:
        if key not in source:
            continue
        members = _command_members(source[key])
        if members is None:
            return UNKNOWN
        return bool(members & _target_members(output_key, enum_values))
    return UNKNOWN


def _mode_bool(
    output_key: str,
    source: Mapping[str, Any],
    enum_values: Mapping[str, str],
    inactive_modes: set[str],
    enum_fields: Sequence[str],
) -> bool | str:
    for key in enum_fields:
        if key not in source:
            continue
        enum = _canonical_enum(source[key])
        if enum in enum_values:
            return enum_values[enum] == output_key
        if enum in inactive_modes:
            return False
    return UNKNOWN


def _target_members(output_key: str, enum_values: Mapping[str, str]) -> set[str]:
    targets = {_canonical_enum(output_key)}
    api_key = _api_key(output_key)
    if api_key is not None:
        targets.add(_canonical_enum(api_key))
    targets.update(enum for enum, target in enum_values.items() if target == output_key)
    return {target for target in targets if target is not None}


def _command_members(value: Any) -> set[str] | None:
    if isinstance(value, str):
        member = _canonical_enum(value)
        return {member} if member is not None else None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        members = {_canonical_enum(item) for item in value}
        return {member for member in members if member is not None}
    return None


def _canonical_enum(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def _api_key(output_key: str) -> str | None:
    for api_key, value_key in SUPPORTED_API_VALUE_KEYS.items():
        if value_key == output_key:
            return api_key
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _exact_api_fields(values: Mapping[str, Any], namespace: str) -> Mapping[str, Any]:
    """Retain only declared dotted API fields from a generic domain row."""

    return {
        key: values[key]
        for key in sorted(values)
        if key.startswith(namespace) and key in SUPPORTED_API_VALUE_KEYS
    }


def _first_known(*values: Any) -> bool | str:
    for value in values:
        if value != UNKNOWN:
            return value
    return UNKNOWN


def _first_bool(*values: Any) -> bool | str:
    saw_false = False
    for value in values:
        if value is True:
            return True
        if value is False:
            saw_false = True
    return False if saw_false else UNKNOWN


def _values(row: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    values = row.get("values")
    if isinstance(values, Mapping):
        return values
    return row


def _source_refs(
    tick: int,
    sample: _EntitySample,
    communication: Sequence[Mapping[str, Any]],
    domain: Sequence[Mapping[str, Any]],
) -> list[str]:
    refs = [f"truth_frames.jsonl#tick={tick}#entity={sample.entity_id}"]
    for rows in (communication, domain):
        for row in rows:
            row_refs = [
                str(item)
                for item in row.get("source_refs", ())
                if isinstance(item, str)
            ]
            refs.extend(row_refs)
    return refs


def _sample_digest_projection(sample: _EntitySample) -> dict[str, Any]:
    return {
        "tick": sample.tick,
        "entity_id": sample.entity_id,
        "category": sample.category,
        "kind": sample.kind,
        "control_state": _structured_mapping_digest(sample.control_state),
        "mission_state": _structured_mapping_digest(sample.mission_state),
        "security_state": _structured_mapping_digest(sample.security_state),
        "communication_state": _structured_mapping_digest(sample.communication_state),
        "position": sample.position,
        "velocity": sample.velocity,
        "speed_mps": sample.speed_mps,
        "assigned_altitude_m": sample.assigned_altitude_m,
        "route_waypoints": sample.route_waypoints,
    }


def _row_digest_projection(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    entity_id = row.get("entity_id") or row.get("subject_id")
    return {
        "tick": row.get("tick") if isinstance(row.get("tick"), int) else UNKNOWN,
        "entity_id": entity_id if isinstance(entity_id, str) else UNKNOWN,
        "values": _structured_mapping_digest(_values(row)),
    }


def _structured_mapping_digest(values: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = _digest_allowed_keys()
    digest: dict[str, Any] = {}
    for key, value in sorted(values.items()):
        if key in {
            "control_state",
            "mission_state",
            "security_state",
            "communication_state",
        } and isinstance(value, Mapping):
            nested = _structured_mapping_digest(value)
            if nested:
                digest[key] = nested
            continue
        if key not in allowed_keys:
            continue
        projected = _digest_value(value)
        if projected != UNKNOWN:
            digest[key] = projected
    return digest


def _digest_allowed_keys() -> set[str]:
    return {
        *SUPPORTED_API_VALUE_KEYS.keys(),
        *SUPPORTED_API_VALUE_KEYS.values(),
        "active_command",
        "active_commands",
        "active_mode",
        "alternate_link_active",
        "availability",
        "availability_state",
        "available",
        "backup_active",
        "channel_mode",
        "command_mode",
        "commands",
        "communication_unavailable",
        "condition",
        "handover_active",
        "handover_mode",
        "link_availability",
        "link_available",
        "link_mode",
        "link_state",
        "mode",
        "phase",
        "station_available",
        "station_status",
        "status",
        "threat",
    }


def _digest_value(value: Any) -> Any:
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        projected = [_digest_value(item) for item in value]
        return [item for item in projected if item != UNKNOWN]
    return UNKNOWN


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ControlResponseStateError(f"{path} must contain a JSON object")
    return value


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 3
    ):
        return None
    numbers = [_number(item) for item in value[:3]]
    if any(item is None for item in numbers):
        return None
    return (float(numbers[0]), float(numbers[1]), float(numbers[2]))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | str:
    return round(float(value), 6) if value is not None else UNKNOWN


__all__ = [
    "CONTROL_FAMILY",
    "FORBIDDEN_INPUTS",
    "SUPPORTED_API_VALUE_KEYS",
    "ControlResponseStateError",
    "build_control_response_state_rows",
]
