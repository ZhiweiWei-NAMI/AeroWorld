"""Deterministic domain-state supplement for low-altitude semantic rules.

This module produces state observations that downstream semantic rules can
turn into predicate truth and event transitions. It deliberately does not read
authored event traces, event realizations, dynamic labels, or expected event
labels, and it does not emit event occurrences.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Dataset.semantic_truth.provenance import (
    canonical_json,
    digest_file,
    digest_object,
    read_jsonl,
    stable_identifier,
)
from Dataset.semantic_truth.facility_scope import (
    FacilityScopeError,
    validate_roster_facility_scope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "1.2.0"
RULE_VERSION = "1.14.0"
SOURCE_CLASSES = {
    "observed_simulator_truth",
    "derived_from_observed",
    "simulated_derived",
    "deterministic_simulation",
    "standard_referenced_parameter",
    "unknown",
}
UNKNOWN = "unknown"
CHARGING_NONACTIVE_UAV_STATES = frozenset({"landed", "charging_hold"})
CHARGING_NONACTIVE_FACILITY_STATES = frozenset(
    {"available", "reserved", "unavailable"}
)
OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES = (
    "constraint_state",
    "control_state",
    "mission_state",
    "security_state",
    "sensor_state",
    "facility_state",
    "communication_state",
    "incident_state",
    "vehicle_state",
    "pedestrian_state",
    "navigation_state",
)
OBJECTIVE_FORBIDDEN_ENTITY_KEYS = {
    "active_event_id",
    "active_event_ids",
    "active_event_label",
    "active_event_labels",
    "activity_label",
    "activity_labels",
    "activity_state",
    "activity_type",
    "dynamic_label",
    "dynamic_labels",
    "expected_event",
    "posture",
    "scenario_plan",
    "semantic_role",
    "state_facets",
    "task_id",
}
OBJECTIVE_FORBIDDEN_ANNOTATION_KEYS = {
    "activity",
    "activity_type",
    "posture",
    "state_facets",
}


class DomainStateSimulationError(ValueError):
    """Raised when profile, inputs, or output rows fail closed."""


@dataclass(frozen=True)
class DomainEpisodeInputs:
    episode_id: str
    episode_root: Path
    manifest_projection: dict[str, Any]
    roster_entities: dict[str, dict[str, Any]]
    frames: list[dict[str, Any]]
    charging_truth_by_tick: dict[int, dict[str, dict[str, Any]]]
    queue_window_by_tick: dict[int, dict[str, Any]]
    weather_by_tick: dict[int, dict[str, Any]]
    scene_setup: dict[str, Any] | None
    charging_service_plan: dict[str, Any]
    preflight_uavs_by_tick: dict[int, list[dict[str, Any]]]
    preflight_gaps_by_tick: dict[int, dict[str, str]]
    preflight_source_path: str | None
    input_files: dict[str, Path]
    input_digest: str


@dataclass(frozen=True)
class DomainEpisodeArtifacts:
    episode_id: str
    output_dir: Path
    observations: tuple[dict[str, Any], ...]
    files: dict[str, str]
    manifest: dict[str, Any]
    summary: dict[str, Any]
    input_digest: str


def load_domain_state_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise DomainStateSimulationError(f"{path} must contain a JSON object")
    validate_profile(profile)
    return profile


def validate_profile(profile: Mapping[str, Any]) -> None:
    required = [
        "schema_name",
        "schema_version",
        "profile_id",
        "default_input_root",
        "default_output_root",
        "model",
        "inputs",
        "authoritative_tick_policy",
        "source_policy",
        "parameter_governance",
        "domain_models",
        "observation_families",
        "validation",
    ]
    missing = [key for key in required if key not in profile]
    if missing:
        raise DomainStateSimulationError(f"profile lacks required keys: {missing}")
    if profile["schema_name"] != "domain_state_supplement_profile":
        raise DomainStateSimulationError(
            "profile schema_name is not domain_state_supplement_profile"
        )
    if (
        profile["model"].get("seed_namespace")
        != "domain_state_structured_input_digest_v2"
    ):
        raise DomainStateSimulationError(
            "profile model.seed_namespace must be domain_state_structured_input_digest_v2"
        )
    inputs = profile["inputs"]
    for key, expected in (
        ("episode_manifest", "episode_manifest.json"),
        ("entity_roster", "global_entity_roster.json"),
        ("truth_frames", "truth_frames.jsonl"),
        ("weather", "weather_meta.jsonl"),
        ("scene_setup", "scene_setup.json"),
    ):
        if inputs.get(key) != expected:
            raise DomainStateSimulationError(f"profile inputs.{key} must be {expected}")
    charging_input = str(inputs.get("charging_service_plan") or "")
    if not charging_input.endswith("{episode_id}/charging_service_plan.json"):
        raise DomainStateSimulationError(
            "profile inputs.charging_service_plan must be an episode-keyed plan path"
        )
    forbidden = set(profile["source_policy"].get("forbidden_inputs") or [])
    for forbidden_name in (
        "event_trace.jsonl",
        "event_realization.jsonl",
        "dynamic_labels.jsonl",
        "scenario_plan.json",
        "expected_event",
        "semantic_role",
        "task_id",
        "source_event_script_path",
        "n_events",
    ):
        if forbidden_name not in forbidden:
            raise DomainStateSimulationError(f"profile must forbid {forbidden_name}")
    allowed_manifest = set(
        profile["source_policy"].get("allowed_manifest_fields") or []
    )
    if allowed_manifest != {
        "episode_id", "seed", "map_id", "generation.source_episode_dir"
    }:
        raise DomainStateSimulationError(
            "profile must restrict manifest facts to episode identity and the declared source episode path"
        )
    governance = profile["parameter_governance"]
    if governance.get("review_status") != "reviewed_for_controlled_simulation":
        raise DomainStateSimulationError("profile parameters must be reviewed")
    if governance.get("source_class") != "deterministic_simulation":
        raise DomainStateSimulationError(
            "profile parameter_governance source_class must be deterministic_simulation"
        )
    if governance.get("measurement_provenance") != {
        "sensor_logs_present": False,
        "security_logs_present": False,
        "payload_logs_present": False,
        "facility_logs_present": False,
    }:
        raise DomainStateSimulationError(
            "profile parameter_governance.measurement_provenance is incomplete"
        )
    allowed_statuses = {
        "standard_referenced_parameter",
        "deterministic_simulation",
        "simulated_derived",
    }
    groups = governance.get("parameter_groups")
    if not isinstance(groups, Mapping) or any(
        not isinstance(group, Mapping) or group.get("status") not in allowed_statuses
        for group in groups.values()
    ):
        raise DomainStateSimulationError(
            "domain parameter group statuses violate the V7 authority vocabulary"
        )
    if not profile.get("observation_families"):
        raise DomainStateSimulationError(
            "profile observation_families must be non-empty"
        )
    gnss = profile["domain_models"].get("gnss")
    service = gnss.get("service_instance") if isinstance(gnss, Mapping) else None
    if not isinstance(service, Mapping) or set(service) != {
        "service_id",
        "ontology_class_id",
        "authority",
    }:
        raise DomainStateSimulationError(
            "domain_models.gnss.service_instance must be an exact modeled-individual declaration"
        )
    if (
        not isinstance(service["service_id"], str)
        or not service["service_id"]
        or service["ontology_class_id"] != "world:GnssService"
        or service["authority"] != "domain_state_supplement_profile"
    ):
        raise DomainStateSimulationError(
            "domain_models.gnss.service_instance is not the governed GNSS service"
        )
    spoof_signal = (
        gnss.get("spoofing_signal_instance") if isinstance(gnss, Mapping) else None
    )
    if not isinstance(spoof_signal, Mapping) or set(spoof_signal) != {
        "service_id",
        "ontology_class_id",
        "authority",
    }:
        raise DomainStateSimulationError(
            "domain_models.gnss.spoofing_signal_instance must be an exact "
            "modeled-individual declaration"
        )
    if (
        not isinstance(spoof_signal["service_id"], str)
        or not spoof_signal["service_id"]
        or spoof_signal["ontology_class_id"] != "world:SpoofingSignal"
        or spoof_signal["authority"] != "domain_state_supplement_profile"
    ):
        raise DomainStateSimulationError(
            "domain_models.gnss.spoofing_signal_instance is not the governed "
            "spoofing signal individual"
        )


def load_episode_inputs(
    episode_root: Path,
    profile: Mapping[str, Any],
    *,
    charging_service_plan_path: Path | None = None,
) -> DomainEpisodeInputs:
    inputs = profile["inputs"]
    paths = {
        "episode_manifest": episode_root / inputs["episode_manifest"],
        "entity_roster": episode_root / inputs["entity_roster"],
        "truth_frames": episode_root / inputs["truth_frames"],
        "weather": episode_root / inputs["weather"],
        "charging_service_plan": resolve_declared_input_path(
            episode_root,
            str(inputs["charging_service_plan"]).format(episode_id=episode_root.name),
        ),
    }
    if charging_service_plan_path is not None:
        paths["charging_service_plan"] = charging_service_plan_path
    for name in (
        "episode_manifest",
        "entity_roster",
        "truth_frames",
        "weather",
        "charging_service_plan",
    ):
        if not paths[name].is_file():
            raise DomainStateSimulationError(
                f"required {name} input is missing: {paths[name]}"
            )

    manifest = _load_json(paths["episode_manifest"])
    episode_id = manifest.get("episode_id")
    seed = manifest.get("seed")
    map_id = manifest.get("map_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise DomainStateSimulationError(
            f"{paths['episode_manifest']}: episode_id must be a non-empty string"
        )
    if episode_id != episode_root.name:
        raise DomainStateSimulationError(
            f"episode manifest id {episode_id} does not match directory {episode_root.name}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise DomainStateSimulationError(
            f"{paths['episode_manifest']}: seed must be a non-negative integer"
        )
    if not isinstance(map_id, str) or not map_id:
        raise DomainStateSimulationError(
            f"{paths['episode_manifest']}: map_id must be a non-empty string"
        )
    generation = manifest.get("generation")
    if generation is not None and not isinstance(generation, Mapping):
        raise DomainStateSimulationError(
            f"{paths['episode_manifest']}: generation must be an object"
        )
    source_episode_dir = (
        generation.get("source_episode_dir")
        if isinstance(generation, Mapping) else None
    )
    if source_episode_dir is not None and not isinstance(source_episode_dir, str):
        raise DomainStateSimulationError(
            f"{paths['episode_manifest']}: generation.source_episode_dir must be a path"
        )
    manifest_projection = {
        "episode_id": episode_id,
        "seed": seed,
        "map_id": map_id,
        "generation": {"source_episode_dir": source_episode_dir},
        "uav_global_flow": copy.deepcopy(manifest.get("uav_global_flow")),
    }
    roster = _load_json(paths["entity_roster"])
    roster_entities = _index_entities(
        roster.get("entities"), str(paths["entity_roster"])
    )

    tick_policy = profile["authoritative_tick_policy"]
    expected_ticks = list(
        range(
            int(tick_policy["start"]),
            int(tick_policy["end"]) + 1,
            int(tick_policy["step"]),
        )
    )
    wanted_ticks = set(expected_ticks)
    frames: list[dict[str, Any]] = []
    charging_truth_by_tick: dict[int, dict[str, dict[str, Any]]] = {}
    queue_window_by_tick: dict[int, dict[str, Any]] = {}
    seen_ticks: set[int] = set()
    for raw_row in read_jsonl(paths["truth_frames"]):
        tick = raw_row.get("tick")
        if not isinstance(tick, int):
            raise DomainStateSimulationError(
                f"{paths['truth_frames']}: truth frame lacks integer tick"
            )
        if int(tick_policy["start"]) <= tick <= int(tick_policy["end"]):
            if tick in charging_truth_by_tick:
                raise DomainStateSimulationError(
                    f"{paths['truth_frames']}: duplicate charging truth tick {tick}"
                )
            charging_truth_by_tick[tick] = _charging_truth_entities(
                raw_row, roster_entities, episode_id
            )
        if tick not in wanted_ticks:
            continue
        row = sanitize_objective_input(raw_row)
        if row.get("episode_id") != episode_id:
            raise DomainStateSimulationError(
                f"{paths['truth_frames']}: tick {tick} episode_id mismatch"
            )
        if "sumo_active_incidents" not in row:
            raise DomainStateSimulationError(
                f"{paths['truth_frames']}: tick {tick} lacks sumo_active_incidents"
            )
        row["sumo_active_incidents"] = _sumo_incident_projection(
            row["sumo_active_incidents"]
        )
        _validate_frame_roster(row, roster_entities)
        if tick in wanted_ticks:
            vehicles_by_lane: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for entity in _current_entity_views(row):
                if _entity_category(entity) != "vehicle":
                    continue
                lane_id = _nested_value(entity, ("sumo_vehicle", "sumo_lane_id"))
                if isinstance(lane_id, str) and lane_id:
                    vehicles_by_lane[lane_id].append(entity)
            stopped_by_lane = {
                lane_id: sorted(
                    str(entity["entity_id"])
                    for entity in lane_vehicles
                    if _speed(entity) is not None
                    and (_speed(entity) or 0.0)
                    <= float(
                        profile["domain_models"]["ground_traffic"][
                            "queue_speed_threshold_mps"
                        ]
                    )
                )
                for lane_id, lane_vehicles in vehicles_by_lane.items()
            }
            queue_lane_id, stopped_ids = max(
                stopped_by_lane.items(),
                key=lambda item: (len(item[1]), item[0]),
                default=(UNKNOWN, []),
            )
            queue_window_by_tick[tick] = {
                "queue_vehicle_count": len(stopped_ids),
                "queued_vehicle_ids": stopped_ids,
                "queue_lane_id": queue_lane_id,
                "stopped_vehicle_count_by_lane": {
                    lane_id: len(entity_ids)
                    for lane_id, entity_ids in sorted(stopped_by_lane.items())
                },
                "peak_source_tick": tick,
                "window_start_tick": tick,
                "window_end_tick": tick,
            }
        if tick in seen_ticks:
            raise DomainStateSimulationError(
                f"{paths['truth_frames']}: duplicate sampled tick {tick}"
            )
        seen_ticks.add(tick)
        frames.append(copy.deepcopy(row))
    frames.sort(key=lambda row: int(row["tick"]))
    missing_ticks = [tick for tick in expected_ticks if tick not in seen_ticks]
    if missing_ticks:
        raise DomainStateSimulationError(
            f"{paths['truth_frames']}: missing authoritative ticks {missing_ticks}"
        )
    missing_charge_ticks = set(
        range(int(tick_policy["start"]), int(tick_policy["end"]) + 1)
    ) - set(charging_truth_by_tick)
    if missing_charge_ticks:
        raise DomainStateSimulationError(
            f"{paths['truth_frames']}: missing charging truth ticks {sorted(missing_charge_ticks)}"
        )

    weather_by_tick: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(paths["weather"]):
        tick = row.get("tick")
        if not isinstance(tick, int):
            raise DomainStateSimulationError(
                f"{paths['weather']}: invalid weather tick {tick!r}"
            )
        if tick not in wanted_ticks:
            continue
        if tick in weather_by_tick:
            raise DomainStateSimulationError(
                f"{paths['weather']}: duplicate weather tick {tick}"
            )
        weather_by_tick[tick] = copy.deepcopy(row)
    missing_weather_ticks = [
        tick for tick in expected_ticks if tick not in weather_by_tick
    ]
    if missing_weather_ticks:
        raise DomainStateSimulationError(
            f"{paths['weather']}: missing authoritative ticks {missing_weather_ticks}"
        )

    charging_service_plan = _load_json(paths["charging_service_plan"])
    _validate_charging_service_plan(charging_service_plan, episode_id)
    input_files = {name: path for name, path in paths.items() if path.is_file()}
    scene_setup = _load_scene_setup(episode_root, profile, input_files)
    preflight_uavs_by_tick, preflight_gaps_by_tick, preflight_source_path = (
        _load_pad_preflight_uavs(
            episode_id=episode_id,
            source_episode_dir=source_episode_dir,
            roster_entities=roster_entities,
            frames=frames,
            wanted_ticks=wanted_ticks,
            input_files=input_files,
        )
    )

    model_projection = _model_input_projection(
        manifest_projection=manifest_projection,
        roster_entities=roster_entities,
        frames=frames,
        queue_window_by_tick=queue_window_by_tick,
        weather_by_tick=weather_by_tick,
        scene_setup=scene_setup,
        charging_service_plan=charging_service_plan,
        preflight_uavs_by_tick=preflight_uavs_by_tick,
        preflight_gaps_by_tick=preflight_gaps_by_tick,
    )
    return DomainEpisodeInputs(
        episode_id=episode_id,
        episode_root=episode_root,
        manifest_projection=manifest_projection,
        roster_entities=roster_entities,
        frames=frames,
        charging_truth_by_tick=charging_truth_by_tick,
        queue_window_by_tick=queue_window_by_tick,
        weather_by_tick=weather_by_tick,
        scene_setup=scene_setup,
        charging_service_plan=charging_service_plan,
        preflight_uavs_by_tick=preflight_uavs_by_tick,
        preflight_gaps_by_tick=preflight_gaps_by_tick,
        preflight_source_path=preflight_source_path,
        input_files=input_files,
        input_digest=digest_object(model_projection),
    )


def _load_pad_preflight_uavs(
    *,
    episode_id: str,
    source_episode_dir: str | None,
    roster_entities: Mapping[str, Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    wanted_ticks: set[int],
    input_files: dict[str, Path],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, str]], str | None]:
    """Read declared pad and full-episode observer poses before ROI visibility."""
    candidate_ticks: dict[str, set[int]] = {}
    full_episode_observers: set[str] = set()
    for entity_id, entity in roster_entities.items():
        if _entity_category(entity) != "uav":
            continue
        home_pad = _nested_value(entity, ("lifecycle", "home_pad_entity_id"))
        observer_lifecycle = _nested_mapping(entity, ("observer_lifecycle",))
        full_episode_observer = (
            observer_lifecycle is not None
            and observer_lifecycle.get("presence") == "episode_full_duration"
        )
        visibility = _nested_mapping(entity, ("runtime_visibility",))
        first_visible = visibility.get("first_visible_tick") if visibility else None
        if first_visible is None or not (
            (isinstance(home_pad, str) and home_pad) or full_episode_observer
        ):
            continue
        activation = entity.get("activation_tick")
        if (
            not isinstance(first_visible, int)
            or isinstance(first_visible, bool)
            or not isinstance(activation, int)
            or isinstance(activation, bool)
        ):
            raise DomainStateSimulationError(
                f"{episode_id}: invalid visibility or activation tick for {entity_id}"
            )
        ticks = {tick for tick in wanted_ticks if activation <= tick < first_visible}
        if ticks:
            candidate_ticks[entity_id] = ticks
            if full_episode_observer:
                full_episode_observers.add(entity_id)
    if not candidate_ticks:
        return {}, {}, None

    gaps: dict[int, dict[str, str]] = defaultdict(dict)
    if not source_episode_dir:
        for entity_id, ticks in candidate_ticks.items():
            for tick in ticks:
                gaps[tick][entity_id] = "source_episode_dir_missing"
        return {}, dict(gaps), None

    declared = Path(source_episode_dir)
    source_root = (PROJECT_ROOT / declared).resolve()
    if (
        declared.is_absolute()
        or not source_root.is_relative_to(PROJECT_ROOT.resolve())
        or source_root.name != episode_id
    ):
        raise DomainStateSimulationError(
            f"{episode_id}: invalid declared source episode path {source_episode_dir!r}"
        )
    source_file = source_root / "trajectories.jsonl"
    source_ref = str(source_file.relative_to(PROJECT_ROOT))
    source_manifest = source_root / "episode_manifest.json"
    if not source_file.is_file() or not source_manifest.is_file():
        for entity_id, ticks in candidate_ticks.items():
            for tick in ticks:
                gaps[tick][entity_id] = f"declared_source_artifact_missing:{source_ref}"
        return {}, dict(gaps), None
    if _load_json(source_manifest).get("episode_id") != episode_id:
        raise DomainStateSimulationError(
            f"{source_manifest}: source episode identity differs from {episode_id}"
        )

    input_files["pad_preflight_trajectories"] = source_file
    input_files["pad_preflight_source_manifest"] = source_manifest
    formal_positions: dict[tuple[int, str], list[float] | None] = {}
    semantic_only_positions: set[tuple[int, str]] = set()
    for frame in frames:
        tick = int(frame["tick"])
        for entity in _current_entity_views(frame):
            entity_id = str(entity["entity_id"])
            if entity_id in candidate_ticks:
                formal_positions[tick, entity_id] = _position(entity)
                presence = entity.get("render_presence")
                pose = entity.get("truth_pose")
                if (isinstance(presence, Mapping) and isinstance(pose, Mapping)
                        and presence.get("offstage") is True
                        and presence.get("submission_state") == "semantic_truth_only"
                        and pose.get("authority_owner") == "all_entity_trajectory_truth"):
                    semantic_only_positions.add((tick, entity_id))
    raw_rows: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in read_jsonl(source_file):
        entity_id = row.get("entity_id")
        if not isinstance(entity_id, str) or entity_id not in candidate_ticks:
            continue
        tick = row.get("tick")
        if not isinstance(tick, int) or isinstance(tick, bool):
            raise DomainStateSimulationError(
                f"{source_ref}: invalid tick for {entity_id}"
            )
        if tick not in wanted_ticks:
            continue
        if row.get("label_class") != "uav":
            raise DomainStateSimulationError(
                f"{source_ref}: {entity_id}@{tick} is not a UAV trajectory"
            )
        key = (tick, entity_id)
        if key in raw_rows:
            raise DomainStateSimulationError(
                f"{source_ref}: duplicate UAV trajectory {entity_id}@{tick}"
            )
        raw_rows[key] = row

    for key, formal_position in formal_positions.items():
        raw = raw_rows.get(key)
        raw_position = _source_trajectory_position(raw)
        if raw_position is None or formal_position is None or raw_position != formal_position:
            raise DomainStateSimulationError(
                f"{source_ref}: source/formal UAV pose differs at {key[1]}@{key[0]}"
            )

    preflight: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entity_id, ticks in candidate_ticks.items():
        for tick in ticks:
            key = (tick, entity_id)
            if key in formal_positions:
                # The L0 trajectory producer can already have projected this
                # offstage state into the semantic frame. Its source pose was
                # compared above; it must not be counted again as preflight.
                if key in semantic_only_positions:
                    continue
                raise DomainStateSimulationError(
                    f"{episode_id}: {entity_id}@{tick} appears before declared first visible tick"
                )
            raw = raw_rows.get(key)
            position = _source_trajectory_position(raw)
            valid_source_state = raw is not None and (
                raw.get("state") == "corridor_observer"
                if entity_id in full_episode_observers
                else raw.get("state") == "preflight_on_pad"
            )
            if raw is None or position is None or not valid_source_state:
                gaps[tick][entity_id] = (
                    "preflight_source_row_missing_or_unresolved:" + source_ref
                )
                continue
            source_uav = {
                    "entity_id": entity_id,
                    "truth_pose": {
                        "position_enu_m": position,
                        "velocity_enu_mps": _source_trajectory_velocity(raw),
                    },
                }
            if entity_id in full_episode_observers:
                if raw.get("activation_tick") != roster_entities[entity_id]["activation_tick"]:
                    raise DomainStateSimulationError(
                        f"{source_ref}: observer activation differs for {entity_id}"
                    )
            if "route_waypoints_enu_m" in raw:
                source_uav["route_waypoints_enu_m"] = copy.deepcopy(raw["route_waypoints_enu_m"])
            preflight[tick].append(source_uav)
    return dict(preflight), dict(gaps), source_ref


def _source_trajectory_position(row: Mapping[str, Any] | None) -> list[float] | None:
    position = row.get("pos_enu") if isinstance(row, Mapping) else None
    if (
        not isinstance(position, list)
        or len(position) != 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in position
        )
    ):
        return None
    return [float(value) for value in position]


def _source_trajectory_velocity(row: Mapping[str, Any] | None) -> list[float] | None:
    velocity = row.get("vel_mps") if isinstance(row, Mapping) else None
    if (
        not isinstance(velocity, list)
        or len(velocity) != 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in velocity
        )
    ):
        return None
    return [float(value) for value in velocity]


def build_episode_artifacts(
    episode_root: Path,
    output_dir: Path,
    profile_path: Path,
    *,
    communication_rows: Sequence[Mapping[str, Any]] = (),
    charging_service_plan_path: Path | None = None,
) -> DomainEpisodeArtifacts:
    profile = load_domain_state_profile(profile_path)
    inputs = load_episode_inputs(episode_root, profile, charging_service_plan_path=charging_service_plan_path)
    domain_input_digest, parameter_digest, seed_digest = _run_digests(
        inputs, profile, profile_path, communication_rows
    )
    common = {
        "episode_id": inputs.episode_id,
        "profile_id": profile["profile_id"],
        "profile_version": profile["schema_version"],
        "model_id": profile["model"]["model_id"],
        "model_version": profile["model"]["model_version"],
        "input_digest": domain_input_digest,
        "parameter_digest": parameter_digest,
        "seed_digest": seed_digest,
    }

    rows = _build_observation_rows(
        inputs,
        profile,
        common,
        communication_rows=communication_rows,
    )
    rows.sort(
        key=lambda row: (
            int(row["tick"]),
            str(row["observation_family"]),
            str(row["subject_id"]),
        )
    )
    validate_output_rows(
        rows, profile, expected_ticks=[int(frame["tick"]) for frame in inputs.frames]
    )
    summary = _build_summary(inputs, profile, rows)
    files = {
        "domain_state_observations.jsonl": _jsonl_text(rows),
        "summary.json": _json_text(summary),
    }
    manifest = _build_manifest(inputs, profile, common, output_dir, files)
    files["domain_state_manifest.json"] = _json_text(manifest)
    return DomainEpisodeArtifacts(
        episode_id=inputs.episode_id,
        output_dir=output_dir,
        observations=tuple(copy.deepcopy(rows)),
        files=files,
        manifest=manifest,
        summary=summary,
        input_digest=domain_input_digest,
    )


def domain_seed_digest(
    episode_root: Path,
    profile_path: Path,
    *,
    communication_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Recreate the governed GNSS seed from the current source inputs in memory."""
    profile = load_domain_state_profile(profile_path)
    inputs = load_episode_inputs(episode_root, profile)
    return _run_digests(inputs, profile, profile_path, communication_rows)[2]


def _run_digests(
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    profile_path: Path,
    communication_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    profile_digest = digest_file(profile_path)
    domain_input_digest = digest_object(
        {
            "episode_inputs": inputs.input_digest,
            "communication_rows": digest_object(communication_rows),
        }
    )
    parameter_digest = digest_object(
        {
            "domain_models": profile["domain_models"],
            "parameter_governance": profile["parameter_governance"],
            "observation_families": profile["observation_families"],
        }
    )
    seed_digest = digest_object(
        {
            "profile_id": profile["profile_id"],
            "profile_digest": profile_digest,
            "input_digest": domain_input_digest,
            "seed_namespace": profile["model"]["seed_namespace"],
        }
    )
    return domain_input_digest, parameter_digest, seed_digest


def build_domain_state_rows(
    episode_root: Path,
    profile_path: Path,
    *,
    communication_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build deterministic domain observations in memory for semantic pipelines."""

    artifacts = build_episode_artifacts(
        episode_root=episode_root,
        output_dir=episode_root,
        profile_path=profile_path,
        communication_rows=communication_rows,
    )
    return [copy.deepcopy(row) for row in artifacts.observations]


def sanitize_objective_input(value: Any) -> Any:
    """Return the sole accepted runtime view for objective semantic models.

    Numeric truth, SUMO truth, and explicitly typed runtime-state families are
    retained. Authored activity, generic state, and scenario-intent fields are
    removed without projecting them into a typed family.
    """

    return _sanitize_objective_value(value, path=())


def write_episode_outputs(artifacts: DomainEpisodeArtifacts) -> None:
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, Path]] = []
    try:
        for file_name, text in artifacts.files.items():
            final_path = artifacts.output_dir / file_name
            temporary_path = artifacts.output_dir / (file_name + ".new")
            temporary_path.write_text(text, encoding="utf-8", newline="\n")
            pending.append((temporary_path, final_path))
        for temporary_path, final_path in pending:
            os.replace(temporary_path, final_path)
    finally:
        for temporary_path, _ in pending:
            temporary_path.unlink(missing_ok=True)


def check_episode_outputs(artifacts: DomainEpisodeArtifacts) -> list[str]:
    mismatches: list[str] = []
    for file_name, expected in sorted(artifacts.files.items()):
        path = artifacts.output_dir / file_name
        if not path.is_file():
            mismatches.append(f"missing:{file_name}")
            continue
        actual = path.read_text(encoding="utf-8-sig")
        if actual != expected:
            mismatches.append(f"drift:{file_name}")
    return mismatches


def validate_output_rows(
    rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    *,
    expected_ticks: Sequence[int],
) -> None:
    if not rows:
        raise DomainStateSimulationError(
            "domain_state_observations.jsonl would be empty"
        )
    expected_families = set(profile["observation_families"])
    observed_families = {str(row.get("observation_family")) for row in rows}
    missing_families = expected_families - observed_families
    if missing_families:
        raise DomainStateSimulationError(
            f"missing observation families: {sorted(missing_families)}"
        )
    observed_ticks = {
        int(row["tick"]) for row in rows if isinstance(row.get("tick"), int)
    }
    missing_ticks = set(expected_ticks) - observed_ticks
    if missing_ticks:
        raise DomainStateSimulationError(
            f"missing observations for ticks: {sorted(missing_ticks)}"
        )
    for index, row in enumerate(rows):
        for field in (
            "observation_id",
            "schema_name",
            "schema_version",
            "episode_id",
            "tick",
            "observation_family",
            "subject_id",
            "subject_category",
            "source_class",
            "rule_id",
            "rule_version",
            "model_id",
            "model_version",
            "source_refs",
            "values",
            "quality",
        ):
            if field not in row:
                raise DomainStateSimulationError(f"row {index} lacks {field}")
        if row["source_class"] not in SOURCE_CLASSES:
            raise DomainStateSimulationError(
                f"row {index} has invalid source_class {row['source_class']}"
            )
        refs = row["source_refs"]
        if not isinstance(refs, list):
            raise DomainStateSimulationError(f"row {index} source_refs must be a list")
        forbidden_ref_tokens = (
            "event_trace",
            "event_realization",
            "dynamic_labels",
            "scenario_plan",
        )
        if any(
            any(token in str(ref) for token in forbidden_ref_tokens) for ref in refs
        ):
            raise DomainStateSimulationError(
                f"row {index} uses a forbidden source reference"
            )


def _build_observation_rows(
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    *,
    communication_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_positions: dict[str, list[float]] = {}
    previous_gnss_error: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    static_duration: dict[str, int] = defaultdict(int)
    fall_persistence: dict[str, int] = defaultdict(int)
    detection_dwell: dict[str, int] = defaultdict(int)
    responder_dwell: dict[str, int] = defaultdict(int)
    previous_responder_distance: dict[str, float] = {}
    dispatch_latched: dict[str, bool] = defaultdict(bool)
    handoff_latched: dict[str, bool] = defaultdict(bool)
    red_duration: dict[str, int] = defaultdict(int)
    crowd_initial: dict[str, Any] = {}
    security_lockout_duration: dict[str, int] = defaultdict(int)
    previous_swing_angle: dict[str, float] = defaultdict(float)
    battery_soc_by_uav: dict[str, float | None] = {}
    last_energy_tick_by_uav: dict[str, int] = {}
    preflight_energy_gap_by_uav: dict[str, str] = {}
    safe_stop_dwell: dict[str, int] = defaultdict(int)
    safe_stop_fault_start: dict[str, int] = {}
    ambulance_priority_state: dict[str, Any] = {}
    forced_landing_state: dict[str, dict[str, Any]] = defaultdict(dict)
    communication_by_tick_entity = {
        (int(row["tick"]), str(row["entity_id"])): row
        for row in communication_rows
        if row.get("schema_name") == "communication_state"
        and isinstance(row.get("tick"), int)
        and isinstance(row.get("entity_id"), str)
        and str(row.get("episode_id", inputs.episode_id)) == inputs.episode_id
    }
    incident_anchor_segment_ids = _episode_incident_anchor_segment_ids(inputs)
    charging_activity_by_tick, charging_issues_by_tick = _charging_activity_by_tick(
        inputs, profile
    )
    from Dataset.semantic_simulation.p09_core_sources import domain_global_energy
    global_energy_checkpoints, global_energy_gaps, global_lifetimes = domain_global_energy(inputs, profile)

    for frame in sorted(inputs.frames, key=lambda item: int(item["tick"])):
        tick = int(frame["tick"])
        weather = inputs.weather_by_tick.get(tick)
        entities = _current_entities(frame)
        pads = _facility_entities(entities, {"landing_pad"})
        facilities = _facility_entities(entities)
        uavs = [entity for entity in entities if _entity_category(entity) == "uav"]
        pedestrians = [
            entity for entity in entities if _entity_category(entity) == "pedestrian"
        ]
        vehicles = [
            entity for entity in entities if _entity_category(entity) == "vehicle"
        ]
        props = [entity for entity in entities if _entity_category(entity) == "prop"]
        traffic_lights = [
            entity for entity in entities if _entity_category(entity) == "traffic_light"
        ]
        ground_stations = _facility_entities(entities, {"ground_control_station"})

        rows.extend(
            _uav_rows(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                uavs=uavs,
                pads=pads,
                facilities=facilities,
                previous_positions=previous_positions,
                previous_gnss_error=previous_gnss_error,
            )
        )
        rows.extend(
            _forced_landing_rows(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                frame=frame,
                uavs=uavs,
                pedestrians=pedestrians,
                vehicles=vehicles,
                previous_positions=previous_positions,
                communication_by_tick_entity=communication_by_tick_entity,
                state_by_uav=forced_landing_state,
            )
        )
        facility_rows = _facility_rows(
            inputs=inputs,
            profile=profile,
            common=common,
            tick=tick,
            facilities=facilities,
            uavs=uavs,
        )
        rows.extend(facility_rows)
        visible_uav_ids = {str(uav["entity_id"]) for uav in uavs}
        previous_tick = max(
            int(profile["authoritative_tick_policy"]["start"]),
            tick - int(profile["authoritative_tick_policy"]["step"]),
        )
        for source_tick in range(previous_tick, tick + 1):
            for uav_id, state in charging_activity_by_tick[source_tick].items():
                if state == UNKNOWN and uav_id not in visible_uav_ids:
                    battery_soc_by_uav[uav_id] = None
        rows.extend(
            _traffic_rows(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                frame=frame,
                vehicles=vehicles,
                traffic_lights=traffic_lights,
                props=props,
                red_duration=red_duration,
                queue_window=inputs.queue_window_by_tick.get(tick),
                scene_setup=inputs.scene_setup,
                incident_anchor_segment_ids=incident_anchor_segment_ids,
            )
        )
        medical_rows = _medical_rows(
            inputs=inputs,
            profile=profile,
            common=common,
            tick=tick,
            pedestrians=pedestrians,
            uavs=uavs,
            responders=_responder_entities(entities),
            static_duration=static_duration,
            fall_persistence=fall_persistence,
            detection_dwell=detection_dwell,
            responder_dwell=responder_dwell,
            previous_responder_distance=previous_responder_distance,
            dispatch_latched=dispatch_latched,
            handoff_latched=handoff_latched,
        )
        rows.extend(medical_rows)
        rows.append(
            _crowd_row(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                pedestrians=pedestrians,
                previous_positions=previous_positions,
                crowd_initial=crowd_initial,
            )
        )
        rows.extend(
            _security_rows(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                uavs=uavs,
                facilities=ground_stations,
                communication_by_tick_entity=communication_by_tick_entity,
                lockout_duration=security_lockout_duration,
            )
        )
        for uav_id, reason in inputs.preflight_gaps_by_tick.get(tick, {}).items():
            battery_soc_by_uav[uav_id] = None
            preflight_energy_gap_by_uav[uav_id] = reason
        preflight_uavs = inputs.preflight_uavs_by_tick.get(tick, [])
        if preflight_uavs:
            preflight_rows = _payload_rows(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                weather=weather,
                uavs=preflight_uavs,
                previous_swing_angle=previous_swing_angle,
                battery_soc_by_uav=battery_soc_by_uav,
                preflight_energy_gap_by_uav=preflight_energy_gap_by_uav,
                charging_activity_by_tick=charging_activity_by_tick,
                charging_issues_by_tick=charging_issues_by_tick,
                global_energy_checkpoints=global_energy_checkpoints,
                global_energy_gaps=global_energy_gaps,
                global_lifetimes=global_lifetimes,
                last_energy_tick_by_uav=last_energy_tick_by_uav,
            )
            if not isinstance(inputs.preflight_source_path, str):
                raise DomainStateSimulationError("preflight poses lack their declared trajectory source")
            for preflight_row in preflight_rows:
                preflight_row["source_refs"].append(inputs.preflight_source_path)
                preflight_row["quality"]["world_state_source_status"] = "available"
                preflight_row["quality"]["render_roi_presence"] = "observer_unseen"
            rows.extend(preflight_rows)
        rows.extend(
            _payload_rows(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                weather=weather,
                uavs=uavs,
                previous_swing_angle=previous_swing_angle,
                battery_soc_by_uav=battery_soc_by_uav,
                preflight_energy_gap_by_uav=preflight_energy_gap_by_uav,
                charging_activity_by_tick=charging_activity_by_tick,
                charging_issues_by_tick=charging_issues_by_tick,
                global_energy_checkpoints=global_energy_checkpoints,
                global_energy_gaps=global_energy_gaps,
                global_lifetimes=global_lifetimes,
                last_energy_tick_by_uav=last_energy_tick_by_uav,
            )
        )
        rows.extend(
            _av_safe_stop_rows(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                frame=frame,
                vehicles=vehicles,
                safe_stop_dwell=safe_stop_dwell,
                fault_start_tick=safe_stop_fault_start,
            )
        )
        rows.append(
            _ambulance_priority_row(
                inputs=inputs,
                profile=profile,
                common=common,
                tick=tick,
                frame=frame,
                vehicles=vehicles,
                priority_state=ambulance_priority_state,
                medical_rows=medical_rows,
            )
        )

        for entity in entities:
            position = _position(entity)
            if position is not None:
                previous_positions[str(entity["entity_id"])] = position
    rows.sort(
        key=lambda row: (
            int(row["tick"]),
            str(row["observation_family"]),
            str(row["subject_id"]),
        )
    )
    return rows


def _uav_rows(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    uavs: Sequence[Mapping[str, Any]],
    pads: Sequence[Mapping[str, Any]],
    facilities: Sequence[Mapping[str, Any]],
    previous_positions: Mapping[str, list[float]],
    previous_gnss_error: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model = profile["domain_models"]
    uav_params = model["uav_motion"]
    gnss_params = model["gnss"]
    for uav in uavs:
        uav_id = str(uav["entity_id"])
        position = _position(uav)
        velocity = _velocity(uav)
        speed = _speed(uav)
        nearest_pad, nearest_pad_distance_3d = _nearest_entity(position, pads)
        nearest_pad_position = _position(nearest_pad) if nearest_pad else None
        pad_delta = _delta_vector(position, nearest_pad_position)
        pad_xy_distance = _xy_norm(pad_delta)
        pad_altitude_delta = pad_delta[2] if isinstance(pad_delta, list) else None
        altitude_m = position[2] if position else None
        vertical_speed = velocity[2] if velocity else None
        previous_position = previous_positions.get(uav_id)
        tick_step = int(profile["authoritative_tick_policy"]["step"])
        tick_hz = float(uav_params.get("tick_hz", 10.0))
        if (
            position is not None
            and isinstance(previous_position, list)
            and len(previous_position) >= 3
            and tick_step > 0
            and tick_hz > 0.0
            and (vertical_speed is None or abs(float(vertical_speed)) < 1e-9)
        ):
            elapsed_s = float(tick_step) / tick_hz
            inferred_vertical_speed = (
                float(position[2]) - float(previous_position[2])
            ) / elapsed_s
            if abs(inferred_vertical_speed) > 1e-9:
                vertical_speed = inferred_vertical_speed
        if position is None or nearest_pad_position is None:
            at_pad: bool | str = UNKNOWN
        else:
            at_pad = _bool_or_unknown(
                pad_xy_distance is not None
                and pad_xy_distance <= float(uav_params["pad_radius_m"])
                and pad_altitude_delta is not None
                and abs(pad_altitude_delta)
                <= float(uav_params["pad_altitude_tolerance_m"])
            )
        if position is None or at_pad == UNKNOWN:
            airborne: bool | str = UNKNOWN
        elif at_pad is True:
            airborne = False
        else:
            airborne = _bool_or_unknown(
                altitude_m is not None
                and altitude_m > float(uav_params["airborne_altitude_m"])
            )
        motion_values = {
            "position_enu_m": _round_list(position),
            "velocity_enu_mps": _round_list(velocity),
            "speed_mps": _round(speed),
            "altitude_m": _round(altitude_m),
            "vertical_speed_mps": _round(vertical_speed),
            "airborne": airborne,
            "at_pad": at_pad,
            "nearest_pad_id": nearest_pad.get("entity_id") if nearest_pad else UNKNOWN,
            "nearest_pad_delta_enu_m": _round_list(pad_delta),
            "nearest_pad_xy_distance_m": _round(pad_xy_distance),
            "nearest_pad_altitude_delta_m": _round(pad_altitude_delta),
            "nearest_pad_distance_m": _round(nearest_pad_distance_3d),
        }
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "uav_motion",
                uav_id,
                "uav",
                "derived_from_observed",
                "domain_state.uav.motion_kinematics",
                RULE_VERSION,
                motion_values,
                source_refs=["truth_frames.jsonl", "global_entity_roster.json"],
            )
        )

        gnss_values, gnss_quality, gnss_source = _gnss_values(
            uav=uav,
            uav_id=uav_id,
            position=position,
            previous_error_range=previous_gnss_error[uav_id],
            params=gnss_params,
            tick=tick,
            seed_digest=common["seed_digest"],
        )
        previous_gnss_error[uav_id] = gnss_values.pop("_next_error_range_m")
        service_instance = gnss_params["service_instance"]
        gnss_values.update(
            {
                "gnss_service_id": service_instance["service_id"],
                "gnss_service_ontology_class_id": service_instance["ontology_class_id"],
                "gnss_service_source_ref": (
                    "domain_state_supplement_profile.json"
                    "#domain_models.gnss.service_instance"
                ),
            }
        )
        spoof_signal = gnss_params["spoofing_signal_instance"]
        gnss_values.update(
            {
                "spoofing_signal_id": spoof_signal["service_id"],
                "spoofing_signal_ontology_class_id": spoof_signal["ontology_class_id"],
                "spoofing_signal_source_ref": (
                    "domain_state_supplement_profile.json"
                    "#domain_models.gnss.spoofing_signal_instance"
                ),
            }
        )
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "gnss_navigation",
                uav_id,
                "uav",
                gnss_source,
                "domain_state.gnss.reported_pose_error",
                RULE_VERSION,
                gnss_values,
                quality=gnss_quality,
                source_refs=["truth_frames.jsonl", "weather_meta.jsonl",
                             "Dataset/semantic_rules/profiles/domain_state_supplement_profile.json"],
            )
        )
    return rows


def _forced_landing_rows(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    frame: Mapping[str, Any],
    uavs: Sequence[Mapping[str, Any]],
    pedestrians: Sequence[Mapping[str, Any]],
    vehicles: Sequence[Mapping[str, Any]],
    previous_positions: Mapping[str, list[float]],
    communication_by_tick_entity: Mapping[tuple[int, str], Mapping[str, Any]],
    state_by_uav: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive the governed fault/failsafe-to-touchdown control state.

    The state machine consumes only physical truth-frame kinematics, route
    endpoint geometry, ground occupancy, and the governed communication model.
    It never reads authored event traces or expected scenario labels.
    """

    params = profile["domain_models"]["forced_landing"]
    tick_step = int(profile["authoritative_tick_policy"]["step"])
    tick_hz = float(params.get("tick_hz", 10.0))
    active_incidents = frame.get("sumo_active_incidents")
    rows: list[dict[str, Any]] = []

    for uav in uavs:
        uav_id = str(uav["entity_id"])
        state = state_by_uav[uav_id]
        position = _position(uav)
        previous_position = previous_positions.get(uav_id)
        velocity = _velocity(uav)
        speed = _speed(uav)
        vertical_speed = (
            velocity[2] if isinstance(velocity, list) and len(velocity) >= 3 else None
        )
        if (
            position is not None
            and isinstance(previous_position, list)
            and len(previous_position) >= 3
            and tick_step > 0
            and tick_hz > 0.0
            and (vertical_speed is None or abs(float(vertical_speed)) < 1e-9)
        ):
            inferred = (float(position[2]) - float(previous_position[2])) / (
                float(tick_step) / tick_hz
            )
            if abs(inferred) > 1e-9:
                vertical_speed = inferred

        physical_fault_active = _physical_fault_active(uav, active_incidents, uav_id)

        communication = communication_by_tick_entity.get((tick, uav_id))
        link_unavailable: bool | str = UNKNOWN
        heartbeat_age_ms: Any = UNKNOWN
        station_id: Any = UNKNOWN
        if communication is not None:
            link_quality = communication.get("link_quality")
            if isinstance(link_quality, Mapping):
                availability = _number(link_quality.get("availability"))
                quality_level = str(link_quality.get("quality_level") or "").lower()
                if quality_level in {"lost", "unavailable", "down"} or (
                    availability is not None
                    and availability <= float(params["link_unavailable_max_availability"])
                ):
                    link_unavailable = True
                elif availability is not None and quality_level in {
                    "excellent", "good", "fair", "poor"
                }:
                    link_unavailable = False
                heartbeat_age_ms = communication.get("heartbeat_age_ms", UNKNOWN)
            station_id = communication.get("station_id", UNKNOWN)

        if link_unavailable is True:
            state.setdefault("link_unavailable_since_tick", tick)
        elif link_unavailable is False and not state.get("failsafe_latched", False):
            state.pop("link_unavailable_since_tick", None)
        unavailable_since = state.get("link_unavailable_since_tick")
        unavailable_duration_ticks = (
            tick - int(unavailable_since)
            if isinstance(unavailable_since, int) and link_unavailable is True
            else 0
            if link_unavailable is False
            else UNKNOWN
        )
        if (
            link_unavailable is True
            and isinstance(unavailable_duration_ticks, int)
            and unavailable_duration_ticks
            >= int(params["failsafe_activation_hold_ticks"])
        ):
            state["failsafe_latched"] = True
        failsafe_active = bool(state.get("failsafe_latched", False))

        fault_or_failsafe_active: bool | str
        if physical_fault_active is True or failsafe_active is True:
            fault_or_failsafe_active = True
        elif physical_fault_active == UNKNOWN:
            fault_or_failsafe_active = UNKNOWN
        else:
            fault_or_failsafe_active = False
        if not isinstance(vertical_speed, (int, float)) or isinstance(
            vertical_speed, bool
        ):
            forced_descent: bool | str = UNKNOWN
        elif fault_or_failsafe_active == UNKNOWN:
            forced_descent = UNKNOWN
        else:
            forced_descent = bool(
                fault_or_failsafe_active
                and float(vertical_speed)
                <= float(params["forced_descent_threshold_mps"])
            )
        if forced_descent is True and physical_fault_active is True and failsafe_active:
            descent_driver = "physical_fault_and_communication_failsafe"
        elif forced_descent is True and physical_fault_active is True:
            descent_driver = "physical_fault"
        elif forced_descent is True and failsafe_active:
            descent_driver = "communication_failsafe"
        elif forced_descent == UNKNOWN:
            descent_driver = UNKNOWN
        else:
            descent_driver = "none"
        if forced_descent is True:
            state["descent_seen"] = True
            state.setdefault("forced_descent_start_tick", tick)
            state.setdefault("forced_descent_initiating_driver", descent_driver)

        endpoint, approach = _emergency_route_endpoint(uav, params)
        geometry_valid = endpoint is not None and approach is not None
        landing_zone_id = (
            stable_identifier(
                "derived_emergency_landing_zone", inputs.episode_id, uav_id, endpoint
            )
            if endpoint is not None
            else UNKNOWN
        )
        descent_start_tick = state.get("forced_descent_start_tick")
        landing_zone_established = bool(
            geometry_valid
            and isinstance(descent_start_tick, int)
            and tick > descent_start_tick
        )
        xy_distance_to_zone = (
            math.dist(position[:2], endpoint[:2])
            if position is not None and endpoint is not None
            else None
        )
        altitude_delta_to_zone = (
            float(position[2]) - float(endpoint[2])
            if position is not None and endpoint is not None
            else None
        )
        in_landing_zone = bool(
            landing_zone_established
            and xy_distance_to_zone is not None
            and xy_distance_to_zone <= float(params["landing_zone_radius_m"])
            and altitude_delta_to_zone is not None
            and abs(altitude_delta_to_zone)
            <= float(params["landing_zone_altitude_tolerance_m"])
        )

        ground_occupants: list[str] = []
        if endpoint is not None:
            for entity in [*pedestrians, *vehicles]:
                entity_position = _position(entity)
                if entity_position is None:
                    continue
                if math.dist(entity_position[:2], endpoint[:2]) <= float(
                    params["ground_clearance_radius_m"]
                ):
                    ground_occupants.append(str(entity["entity_id"]))
        ground_zone_cleared = bool(landing_zone_established and not ground_occupants)

        touchdown_candidate = bool(
            state.get("descent_seen", False)
            and in_landing_zone
            and ground_zone_cleared
            and isinstance(speed, (int, float))
            and float(speed) <= float(params["touchdown_speed_threshold_mps"])
            and (
                vertical_speed is None
                or abs(float(vertical_speed))
                <= float(params["touchdown_vertical_speed_abs_mps"])
            )
        )
        if touchdown_candidate:
            state["touchdown_dwell_ticks"] = (
                int(state.get("touchdown_dwell_ticks", 0)) + tick_step
            )
        else:
            state["touchdown_dwell_ticks"] = 0
        touchdown = bool(
            state.get("touchdown_latched", False)
            or int(state.get("touchdown_dwell_ticks", 0))
            >= int(params["touchdown_required_dwell_ticks"])
        )
        if touchdown:
            state["touchdown_latched"] = True
        # A reported control/activity label is corroborating state only.  The
        # formal landing outcome is latched exclusively after numeric position,
        # speed, vertical-speed, dwell, and ground-clearance checks establish a
        # touchdown.  This prevents a scenario-authored label from bypassing the
        # physical terminal condition.
        landed = bool(touchdown)
        if landed:
            state["landed_latched"] = True
        if state.get("landed_latched", False):
            landed = True
            state["failsafe_latched"] = False
        airborne_override: Any = False if landed else UNKNOWN

        source_refs = [f"truth_frames.jsonl#tick={tick}#entity={uav_id}"]
        if communication is not None:
            source_refs.append(f"communication_state.jsonl#tick={tick}#entity={uav_id}")
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "forced_landing_state",
                uav_id,
                "uav",
                "simulated_derived"
                if communication is not None
                else "derived_from_observed",
                "domain_state.uav.fault_failsafe_forced_descent_touchdown",
                RULE_VERSION,
                {
                    "physical_fault_active": physical_fault_active,
                    "fault_active": physical_fault_active,
                    "position_enu_m": _round_list(position),
                    "altitude_m": _round(position[2])
                    if position is not None
                    else UNKNOWN,
                    "speed_mps": _round(speed),
                    "link_unavailable": link_unavailable,
                    "link_unavailable_since_tick": unavailable_since
                    if isinstance(unavailable_since, int)
                    else UNKNOWN,
                    "link_unavailable_duration_ticks": unavailable_duration_ticks,
                    "heartbeat_age_ms": heartbeat_age_ms,
                    "communication_station_id": station_id,
                    "failsafe_active": failsafe_active,
                    "vertical_speed_mps": _round(vertical_speed),
                    "forced_descent": forced_descent,
                    "descent_seen": bool(state.get("descent_seen", False)),
                    "descent_driver": descent_driver,
                    "forced_descent_initiating_driver": state.get(
                        "forced_descent_initiating_driver", UNKNOWN
                    ),
                    "causal_communication_entity_id": (
                        uav_id
                        if "communication_failsafe" in descent_driver
                        else UNKNOWN
                    ),
                    "causal_communication_station_id": (
                        station_id
                        if "communication_failsafe" in descent_driver
                        else UNKNOWN
                    ),
                    "forced_descent_start_tick": descent_start_tick
                    if isinstance(descent_start_tick, int)
                    else UNKNOWN,
                    "landing_zone_id": landing_zone_id,
                    "landing_zone_center_enu_m": _round_list(endpoint),
                    "landing_zone_approach_enu_m": _round_list(approach),
                    "emergency_landing_zone_valid": landing_zone_established,
                    "ground_zone_cleared": ground_zone_cleared,
                    "ground_zone_occupant_ids": sorted(ground_occupants)[
                        : int(params["max_ids_per_observation"])
                    ],
                    "xy_distance_to_landing_zone_m": _round(xy_distance_to_zone),
                    "altitude_delta_to_landing_zone_m": _round(altitude_delta_to_zone),
                    "in_landing_zone": in_landing_zone,
                    "touchdown_candidate": touchdown_candidate,
                    "touchdown_dwell_ticks": int(state.get("touchdown_dwell_ticks", 0)),
                    "touchdown": touchdown,
                    "landed": landed,
                    "airborne_override": airborne_override,
                },
                source_refs=source_refs,
            )
        )
    return rows


def _emergency_route_endpoint(
    uav: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[list[float] | None, list[float] | None]:
    raw_waypoints = uav.get("route_waypoints_enu_m")
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) < 2:
        return None, None
    waypoints = [_position_from_value(item) for item in raw_waypoints]
    valid = [item for item in waypoints if item is not None]
    if len(valid) < 2:
        return None, None
    endpoint = valid[-1]
    approach = valid[-2]
    if float(endpoint[2]) > float(params["landing_endpoint_max_altitude_m"]):
        return None, None
    if math.dist(endpoint[:2], approach[:2]) > float(
        params["landing_endpoint_alignment_m"]
    ):
        return None, None
    if float(approach[2]) - float(endpoint[2]) < float(
        params["landing_endpoint_min_descent_m"]
    ):
        return None, None
    return endpoint, approach


def _position_from_value(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value[:3]
    ):
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def _route_length_m(value: Any) -> float | None:
    if not isinstance(value, list):
        return None
    points = [_position_from_value(item) for item in value]
    valid = [point for point in points if point is not None]
    if len(valid) < 2 or len(valid) != len(points):
        return None
    return sum(math.dist(start, end) for start, end in zip(valid[:-1], valid[1:]))


def _physical_fault_active(
    entity: Mapping[str, Any],
    active_incidents: Any,
    entity_id: str,
) -> bool | str:
    incident_state = _nested_mapping(entity, ("incident_state",))
    if incident_state is not None:
        direct = _state_bool(
            incident_state,
            (
                "physical_fault_active",
                "fault_active",
                "uav_fault",
                "propulsion_fault",
                "motor_fault",
                "emergency_fault",
            ),
        )
        if direct != UNKNOWN:
            return direct
        fault_type = _exact_enum(
            incident_state,
            ("incident_type", "fault_type", "failure_type", "physical_state"),
            {
                "uav_fault",
                "propulsion_fault",
                "motor_fault",
                "flight_fault",
                "emergency_fault",
            },
        )
        if fault_type != UNKNOWN:
            return True
    incident_value = _active_incident_targets_entity_exact(
        active_incidents,
        entity_id,
        {
            "uav_fault",
            "propulsion_fault",
            "motor_fault",
            "flight_fault",
            "emergency_fault",
        },
    )
    if incident_value != UNKNOWN:
        return incident_value
    return UNKNOWN


def _active_incident_targets_entity_exact(
    active_incidents: Any,
    entity_id: str,
    accepted_types: set[str],
) -> bool | str:
    if not isinstance(active_incidents, list):
        return UNKNOWN
    matched_known_type = False
    for incident in active_incidents:
        if not isinstance(incident, Mapping):
            continue
        incident_type = _exact_enum(
            incident,
            ("accident_class",),
            accepted_types,
        )
        if incident_type == UNKNOWN:
            continue
        matched_known_type = True
        affected = (
            incident.get("affected_vehicle_ids")
            or incident.get("affected_entity_ids")
            or incident.get("entity_ids")
        )
        if isinstance(affected, list) and entity_id not in {
            str(item) for item in affected
        }:
            continue
        return True
    return False if matched_known_type else UNKNOWN


def _facility_rows(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    facilities: Sequence[Mapping[str, Any]],
    uavs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = profile["domain_models"]["pad_facility"]
    for facility in facilities:
        facility_id = str(facility["entity_id"])
        semantic_scope = validate_roster_facility_scope(facility)
        subtype = str(semantic_scope["scope_subtype"])
        facility_state = _nested_mapping(facility, ("facility_state",)) or {}
        fault_active = _state_bool(facility_state, ("fault",))
        reserved = _state_bool(facility_state, ("reserved",))
        charging_state: dict[str, Any] | None = None
        if subtype == "charging_station":
            charging_state = _charging_service_state(
                inputs=inputs,
                facility=facility,
                tick=tick,
            )
            availability = str(charging_state["availability"])
            if fault_active is True:
                availability = "unavailable"
        else:
            availability = _exact_enum(
                facility_state,
                ("availability",),
                {"available", "unavailable", "reserved", "fault", "failed"},
            )
            if availability in {"fault", "failed"}:
                availability = "unavailable"
            elif availability == UNKNOWN and fault_active is True:
                availability = "unavailable"
        explicit_requesters = facility_state.get("requester_ids")
        if subtype == "charging_station":
            assert charging_state is not None
            requesters = list(charging_state["requester_ids"])
        elif isinstance(explicit_requesters, list) and all(
            isinstance(item, str) for item in explicit_requesters
        ):
            requesters = sorted(set(explicit_requesters))
        elif subtype == "landing_pad":
            requesters = _uav_landing_requesters(facility, uavs)
        else:
            requesters = []
        occupancy_uavs = [*uavs, *inputs.preflight_uavs_by_tick.get(tick, [])]
        occupiers = (
            _uav_pad_occupiers(facility, occupancy_uavs, params)
            if subtype == "landing_pad"
            else []
        )
        uav_occupancy_gaps: list[str] = []
        if subtype == "landing_pad":
            if _position(facility) is None:
                uav_occupancy_gaps.append(f"landing_pad_position_missing:{facility_id}")
            uav_occupancy_gaps.extend(
                f"uav_position_missing:{uav['entity_id']}"
                for uav in occupancy_uavs
                if _position(uav) is None
            )
            for uav_id, reason in inputs.preflight_gaps_by_tick.get(tick, {}).items():
                uav_occupancy_gaps.append(f"{uav_id}:{reason}")
        state_capacity = _number(facility_state.get("capacity"))
        contract_capacity = semantic_scope.get("service_capacity")
        if (
            charging_state is not None
            and int(charging_state["capacity"]) != contract_capacity
        ):
            raise DomainStateSimulationError(
                f"{inputs.episode_id}:{facility_id}@{tick} charging plan capacity "
                f"{charging_state['capacity']} conflicts with asset contract {contract_capacity}"
            )
        if state_capacity is not None and state_capacity != contract_capacity:
            raise DomainStateSimulationError(
                f"{inputs.episode_id}:{facility_id}@{tick} facility_state.capacity "
                f"{state_capacity} conflicts with asset contract {contract_capacity}"
            )
        capacity = (
            int(contract_capacity) if isinstance(contract_capacity, int) else None
        )
        explicit_contention = _state_bool(facility_state, ("contention",))
        contention = len(requesters) > capacity if isinstance(capacity, int) else None
        if (
            explicit_contention != UNKNOWN
            and contention is not None
            and explicit_contention is not contention
        ):
            raise DomainStateSimulationError(
                f"{inputs.episode_id}:{facility_id}@{tick} explicit contention "
                "conflicts with requester_count > capacity"
            )
        if reserved is True:
            reservation_state: Any = "reserved"
        elif reserved is False:
            reservation_state = "unreserved"
        else:
            reservation_state = UNKNOWN
        values: dict[str, Any] = {
            "facility_kind": _string_or_unknown(facility.get("entity_kind")),
            "facility_subtype": subtype,
            "ontology_class_id": semantic_scope["ontology_class_id"],
            "availability": availability,
            "fault_active": fault_active,
            "reserved": reserved,
            "reservation_state": reservation_state,
            "position_enu_m": _round_list(_position(facility)),
        }
        if subtype == "charging_station":
            assert charging_state is not None
            service_aircraft_id = (
                str(charging_state["service_aircraft_id"])
                if availability == "available"
                else "none"
            )
            values.update(
                requester_ids=sorted(requesters),
                capacity=capacity,
                contention=contention,
                service_aircraft_id=service_aircraft_id,
                queue_depth=max(0, len(requesters) - int(capacity or 0)),
                service_plan_id=charging_state["service_plan_id"],
            )
            if service_aircraft_id != "none":
                service_uav = next(
                    (
                        uav
                        for uav in uavs
                        if str(uav["entity_id"]) == service_aircraft_id
                    ),
                    None,
                )
                contact_distance = _facility_service_distance(
                    facility,
                    service_uav,
                )
                contact_active = (
                    contact_distance <= float(params["charging_contact_radius_m"])
                    if contact_distance is not None
                    else UNKNOWN
                )
                values.update(
                    service_contact_active=contact_active,
                    service_contact_distance_m=_round(contact_distance),
                )
            else:
                values.update(
                    service_contact_active=False,
                    service_contact_distance_m=UNKNOWN,
                )
        elif subtype == "landing_pad":
            roster_uav_ids = sorted(
                str(entity_id)
                for entity_id, entity in inputs.roster_entities.items()
                if _entity_category(entity) == "uav"
            )
            requester_ids = set(requesters)
            values.update(
                requester_ids=sorted(requesters),
                occupier_ids=sorted(occupiers) if not uav_occupancy_gaps else UNKNOWN,
                occupier_ids_complete=not uav_occupancy_gaps,
                occupied=True if occupiers else UNKNOWN,
                capacity=capacity,
                contention=contention,
                contention_pairs=[
                    {
                        "pad_id": facility_id,
                        "pad_ontology_class_id": "world:LandingPad",
                        "first_aircraft_id": first_aircraft_id,
                        "first_aircraft_ontology_class_id": "world:UnmannedAircraft",
                        "second_aircraft_id": second_aircraft_id,
                        "second_aircraft_ontology_class_id": "world:UnmannedAircraft",
                        "pair_contention": bool(
                            contention
                            and first_aircraft_id in requester_ids
                            and second_aircraft_id in requester_ids
                        ),
                    }
                    for first_aircraft_id, second_aircraft_id in combinations(
                        roster_uav_ids, 2
                    )
                ],
            )
            if uav_occupancy_gaps:
                values["known_occupier_ids"] = sorted(occupiers)
        source_refs = (
            [
                "truth_frames.jsonl",
                "global_entity_roster.json",
                f"aw_data/charging_supplement/{inputs.episode_id}/charging_service_plan.json",
            ]
            if subtype == "charging_station"
            else ["truth_frames.jsonl", "global_entity_roster.json"]
        )
        if subtype == "landing_pad" and inputs.preflight_uavs_by_tick.get(tick):
            if inputs.preflight_source_path is None:
                raise DomainStateSimulationError(
                    f"{inputs.episode_id}: preflight UAVs lack source trajectory reference"
                )
            source_refs.extend(
                f"{inputs.preflight_source_path}#tick={tick}&entity={uav['entity_id']}"
                for uav in inputs.preflight_uavs_by_tick[tick]
            )
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "pad_facility",
                facility_id,
                "facility",
                "simulated_derived",
                "domain_state.facility.typed_service_state",
                RULE_VERSION,
                values,
                quality={
                    "status": "unknown",
                    "missing_inputs": sorted(
                        [*uav_occupancy_gaps, "pad_blocker_geometry_contract"]
                    ),
                }
                if subtype == "landing_pad"
                else (
                    {
                        "status": "unknown",
                        "missing_inputs": [
                            "facility_or_selected_uav_position_enu_m"
                        ],
                    }
                    if subtype == "charging_station"
                    and values["service_contact_active"] == UNKNOWN
                    else None
                ),
                source_refs=source_refs,
            )
        )
    return rows


def _facility_service_distance(
    facility: Mapping[str, Any],
    uav: Mapping[str, Any] | None,
) -> float | None:
    facility_position = _position(facility)
    uav_position = _position(uav) if isinstance(uav, Mapping) else None
    if facility_position is None or uav_position is None:
        return None
    return math.dist(facility_position, uav_position)


def _traffic_rows(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    frame: Mapping[str, Any],
    vehicles: Sequence[Mapping[str, Any]],
    traffic_lights: Sequence[Mapping[str, Any]],
    props: Sequence[Mapping[str, Any]],
    red_duration: dict[str, int],
    queue_window: Mapping[str, Any] | None,
    scene_setup: Mapping[str, Any] | None,
    incident_anchor_segment_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = profile["domain_models"]["ground_traffic"]
    sumo_traffic_lights = frame.get("sumo_traffic_light_states")
    active_incidents = frame.get("sumo_active_incidents")
    stopped_by_lane: dict[str, list[str]] = defaultdict(list)
    vehicles_by_lane: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for vehicle in vehicles:
        lane_id = _nested_value(vehicle, ("sumo_vehicle", "sumo_lane_id"))
        if isinstance(lane_id, str) and lane_id:
            vehicles_by_lane[lane_id].append(vehicle)
        if (
            isinstance(lane_id, str)
            and lane_id
            and _speed(vehicle) is not None
            and (_speed(vehicle) or 0.0) <= float(params["queue_speed_threshold_mps"])
        ):
            stopped_by_lane[lane_id].append(str(vehicle["entity_id"]))
    queue_lane_id, stopped_at_tick = max(
        (
            (lane_id, sorted(entity_ids))
            for lane_id, entity_ids in stopped_by_lane.items()
        ),
        key=lambda item: (len(item[1]), item[0]),
        default=(UNKNOWN, []),
    )
    stopped = (
        [str(entity_id) for entity_id in queue_window["queued_vehicle_ids"]]
        if isinstance(queue_window, Mapping)
        and isinstance(queue_window.get("queued_vehicle_ids"), list)
        else stopped_at_tick
    )
    signal_entities = [
        entity
        for entity in traffic_lights
        if _entity_category(entity) == "traffic_light"
    ]
    signal_states: dict[str, str] = {}
    for entity in signal_entities:
        signal_id = str(entity["entity_id"])
        facility_state = _nested_mapping(entity, ("facility_state",)) or {}
        structured_fault = _state_bool(facility_state, ("fault",))
        declared_state = entity.get("state")
        if structured_fault is True and declared_state == "all_red_fault":
            signal_states[signal_id] = "all_red_fault"
        elif structured_fault is False and declared_state in {
            "green_cycle", "nominal", "healthy", "sumo_cycle",
        }:
            signal_states[signal_id] = "nominal"
        else:
            signal_states[signal_id] = UNKNOWN
    incident_fault = _incident_has_class(
        active_incidents,
        {"traffic_light_all_red_fault"},
    )
    if not isinstance(sumo_traffic_lights, Mapping):
        raise DomainStateSimulationError("sampled frame lacks SUMO controller states")
    targeted_controllers = {
        incident["traffic_light_id"]
        for incident in active_incidents
        if incident["accident_class"] == "traffic_light_all_red_fault"
    }
    for target in targeted_controllers:
        if target not in sumo_traffic_lights:
            raise DomainStateSimulationError(f"active all-red incident target has no current SUMO state: {target}")
    for signal_id, state in sumo_traffic_lights.items():
        if str(signal_id) in signal_states:
            raise DomainStateSimulationError(f"scene and SUMO controller keys collide: {signal_id}")
        signal_text = state.get("state") if isinstance(state, Mapping) else None
        if not isinstance(signal_text, str) or not signal_text or any(
            ch not in "rgyou" for ch in signal_text.lower()
        ):
            classified = UNKNOWN
        elif signal_id in targeted_controllers:
            classified = "all_red_fault" if set(signal_text.lower()) == {"r"} else UNKNOWN
        else:
            classified = "sumo_cycle"
        signal_states[str(signal_id)] = classified
    controller_health = "unknown"
    if any(state == "all_red_fault" for state in signal_states.values()):
        controller_health = "all_red_fault"
    elif signal_states and not incident_fault and UNKNOWN not in signal_states.values():
        controller_health = "nominal"
    tick_step = int(profile["authoritative_tick_policy"]["step"])
    for missing_id in red_duration.keys() - signal_states.keys():
        del red_duration[missing_id]
    duration_values: dict[str, int | str] = {}
    for signal_id, state in signal_states.items():
        if state == "all_red_fault":
            red_duration[signal_id] += tick_step
            duration_values[signal_id] = red_duration[signal_id]
        elif state == UNKNOWN:
            red_duration.pop(signal_id, None)
            duration_values[signal_id] = UNKNOWN
        else:
            red_duration[signal_id] = 0
            duration_values[signal_id] = 0
    mean_speed = _mean([_speed(vehicle) for vehicle in vehicles])
    values = {
        "traffic_signal_system_id": stable_identifier(
            "traffic_signal_system", inputs.episode_id
        ),
        "traffic_signal_ids": sorted(signal_states),
        "controller_count": len(signal_states) if signal_states else UNKNOWN,
        "controller_health": controller_health,
        "malfunction": controller_health == "all_red_fault"
        if controller_health != UNKNOWN
        else UNKNOWN,
        "red_duration_ticks_by_controller": dict(sorted(duration_values.items())),
        "max_red_duration_ticks": (UNKNOWN if not duration_values or
                                   UNKNOWN in duration_values.values() else
                                   max(duration_values.values())),
        "queue_vehicle_count": len(stopped),
        "queue_lane_id": (
            queue_window.get("queue_lane_id", UNKNOWN)
            if isinstance(queue_window, Mapping)
            else queue_lane_id
        ),
        "queue_lane_ontology_class_id": "world:RoadLane",
        "queue_mean_speed_mps": _round(
            _mean(
                [
                    _speed(vehicle)
                    for vehicle in vehicles
                    if str(vehicle["entity_id"]) in set(stopped)
                ]
            )
        ),
        "network_mean_speed_mps": _round(mean_speed),
        "queued_vehicle_ids": sorted(stopped)[: int(params["max_ids_per_observation"])],
        "queue_sampling_window": (
            dict(queue_window) if isinstance(queue_window, Mapping) else UNKNOWN
        ),
    }
    source_class = (
        "derived_from_observed"
        if signal_states or isinstance(sumo_traffic_lights, Mapping)
        else "unknown"
    )
    rows.append(
        _observation_row(
            inputs,
            common,
            tick,
            "signal_queue",
            "traffic_signal_system",
            "traffic_control",
            source_class,
            "domain_state.traffic.signal_health_and_queue",
            RULE_VERSION,
            values,
            source_refs=["truth_frames.jsonl"],
        )
    )
    for lane_id, lane_vehicles in sorted(vehicles_by_lane.items()):
        queued_ids = sorted(stopped_by_lane.get(lane_id, ()))
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "signal_queue_lane_state",
                lane_id,
                "road_lane",
                "observed_simulator_truth",
                "domain_state.traffic.sumo_lane_queue",
                RULE_VERSION,
                {
                    "queue_lane_id": lane_id,
                    "queue_lane_ontology_class_id": "world:RoadLane",
                    "queue_vehicle_count": len(queued_ids),
                    "queued_vehicle_ids": queued_ids,
                    "lane_vehicle_count": len(lane_vehicles),
                },
                source_refs=["truth_frames.jsonl#entities[].sumo_vehicle.sumo_lane_id"],
            )
        )
    for signal_id, signal_state in sorted(signal_states.items()):
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "traffic_signal_state",
                signal_id,
                "traffic_signal",
                source_class,
                "domain_state.traffic.signal_controller_state",
                RULE_VERSION,
                {
                    "signal_id": signal_id,
                    "signal_ontology_class_id": "world:TrafficSignal",
                    "controller_state": signal_state,
                    "controller_health": (
                        "all_red_fault"
                        if signal_state == "all_red_fault"
                        else UNKNOWN if signal_state == UNKNOWN else "nominal"
                    ),
                },
                source_refs=["truth_frames.jsonl"],
            )
        )

    barrier_props = _barrier_props(props)
    barrier_segment_ids = _barrier_segment_ids(scene_setup)
    uncertain_barrier_segment_ids = {
        segment_id for prop in barrier_props
        if _barrier_active(prop) is not False
        and (segment_id := barrier_segment_ids.get(str(prop["entity_id"]))) is not None
    }
    closure_scope_known = False
    if isinstance(active_incidents, list):
        closed = [
            copy.deepcopy(item)
            for item in active_incidents
            if isinstance(item, Mapping) and _incident_closes_road(item)
        ]
        closure_scope_known = all(
            isinstance(item.get("anchor"), Mapping)
            and isinstance(item["anchor"].get("sumo_edge_id"), str)
            and bool(item["anchor"]["sumo_edge_id"])
            for item in closed
        )
        closed_refs = [
            _string_or_unknown(
                item.get("incident_id") or item.get("id"), f"incident_{index}"
            )
            for index, item in enumerate(closed)
        ]
        affected_vehicle_ids = sorted(_affected_vehicle_ids(active_incidents, vehicles))
        barrier_active = any(_barrier_active(prop) is True for prop in barrier_props)
        closure_values = {
            "road_closed": True if closed else UNKNOWN if uncertain_barrier_segment_ids else False,
            "barrier_active": barrier_active,
            "barrier_prop_ids": sorted(
                str(prop["entity_id"]) for prop in barrier_props
            ),
            "active_closure_count": len(closed),
            "closed_segment_refs": closed_refs,
            # A vehicle affected by a closure may occupy another segment.
            # A present barrier is not measured lane obstruction. Only the
            # explicit incident anchor identifies a known closed RoadSegment.
            "closed_road_segment_ids": sorted({
                anchor["sumo_edge_id"] for item in closed
                if isinstance((anchor := item.get("anchor")), Mapping)
                and isinstance(anchor.get("sumo_edge_id"), str)
                and anchor["sumo_edge_id"]
            }),
            "affected_vehicle_ids": affected_vehicle_ids[
                : int(params["max_ids_per_observation"])
            ],
            "affected_vehicle_count": len(affected_vehicle_ids),
            "route_deviation_vehicle_count": _route_deviation_vehicle_count(
                vehicles, active_incidents
            ),
            "detour_candidate_vehicle_count": len(affected_vehicle_ids),
        }
        source_class = (
            "observed_simulator_truth"
            if closed or barrier_active
            else "derived_from_observed"
        )
    else:
        barrier_active = any(_barrier_active(prop) is True for prop in barrier_props)
        closure_values = {
            "road_closed": UNKNOWN,
            "barrier_active": barrier_active if barrier_props else UNKNOWN,
            "barrier_prop_ids": sorted(
                str(prop["entity_id"]) for prop in barrier_props
            ),
            "active_closure_count": UNKNOWN,
            "closed_segment_refs": [],
            "closed_road_segment_ids": [],
            "affected_vehicle_ids": [],
            "affected_vehicle_count": UNKNOWN,
            "route_deviation_vehicle_count": UNKNOWN,
            "detour_candidate_vehicle_count": UNKNOWN,
        }
        source_class = "derived_from_observed" if barrier_props else "unknown"
    rows.append(
        _observation_row(
            inputs,
            common,
            tick,
            "road_closure",
            "road_network",
            "road_network",
            source_class,
            "domain_state.traffic.road_barrier_closure",
            RULE_VERSION,
            closure_values,
            source_refs=["truth_frames.jsonl"],
        )
    )
    closed_segment_ids = frozenset(
        str(segment_id) for segment_id in closure_values["closed_road_segment_ids"]
    )
    observed_segment_ids = {
        str(edge_id)
        for vehicle in vehicles
        if isinstance(
            (edge_id := _nested_value(vehicle, ("sumo_vehicle", "sumo_edge_id"))),
            str,
        )
        and edge_id
    }
    planned_segment_ids = (
        set(barrier_segment_ids.values()) | incident_anchor_segment_ids
    )
    for segment_id in sorted(observed_segment_ids | set(closed_segment_ids)):
        planned_segment_ids.add(segment_id)
    for segment_id in sorted(planned_segment_ids):
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "road_segment_state",
                segment_id,
                "road_segment",
                source_class,
                "domain_state.traffic.sumo_edge_closure",
                RULE_VERSION,
                {
                    "road_segment_id": segment_id,
                    "road_segment_ontology_class_id": "world:RoadSegment",
                    "road_closed": (
                        True if segment_id in closed_segment_ids
                        else UNKNOWN if not closure_scope_known
                        or segment_id in uncertain_barrier_segment_ids
                        else False
                    ),
                },
                source_refs=[
                    "truth_frames.jsonl#entities[].sumo_vehicle.sumo_edge_id",
                    *(
                        ["scene_setup.json#road_segment_anchors"]
                        if segment_id in set(barrier_segment_ids.values())
                        else []
                    ),
                ],
            )
        )
    return rows


def _medical_rows(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    pedestrians: Sequence[Mapping[str, Any]],
    uavs: Sequence[Mapping[str, Any]],
    responders: Sequence[Mapping[str, Any]],
    static_duration: dict[str, int],
    fall_persistence: dict[str, int],
    detection_dwell: dict[str, int],
    responder_dwell: dict[str, int],
    previous_responder_distance: dict[str, float],
    dispatch_latched: dict[str, bool],
    handoff_latched: dict[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = profile["domain_models"]["medical_response"]
    tick_step = int(profile["authoritative_tick_policy"]["step"])
    for pedestrian in pedestrians:
        pedestrian_id = str(pedestrian["entity_id"])
        speed = _speed(pedestrian)
        is_static: bool | str = (
            speed <= float(params["static_speed_threshold_mps"])
            if speed is not None
            else UNKNOWN
        )
        pedestrian_state = _nested_mapping(pedestrian, ("pedestrian_state",)) or {}
        structured_fallen = _state_bool(pedestrian_state, ("fallen",))
        structured_injured = _state_bool(pedestrian_state, ("injured",))
        structured_posture = _exact_enum(
            pedestrian_state,
            ("posture",),
            {"standing", "fallen", "lying", "prone"},
        )
        structured_health = _exact_enum(
            pedestrian_state,
            ("health",),
            {"nominal", "injured"},
        )
        posture = structured_posture
        if structured_fallen != UNKNOWN:
            fallen: bool | str = structured_fallen
        elif structured_injured is True or structured_posture in {
            "fallen",
            "lying",
            "prone",
        }:
            fallen = True
        elif structured_injured is False and structured_posture == "standing":
            fallen = False
        else:
            fallen = UNKNOWN
        position = _position(pedestrian)
        if fallen is True and is_static is True and position is not None:
            static_duration[pedestrian_id] += tick_step
        else:
            # Fall confirmation requires an uninterrupted low-speed, posed
            # interval after the fall evidence begins.  Pre-fall waiting time,
            # high-speed motion, and missing pose cannot contribute.
            static_duration[pedestrian_id] = 0
        health_state = (
            structured_health
            if structured_health != UNKNOWN
            else "injured"
            if structured_injured is True
            else "nominal"
            if structured_injured is False
            else UNKNOWN
        )
        if fallen is True:
            fall_persistence[pedestrian_id] += tick_step
        elif fallen is False:
            fall_persistence[pedestrian_id] = 0
        if fallen is False or is_static is False:
            confirmed_fall: bool | str = False
        elif fallen is True and is_static is True and position is not None:
            confirmed_fall = bool(
                fall_persistence[pedestrian_id]
                >= int(params["fall_confirmation_ticks"])
                and static_duration[pedestrian_id]
                >= int(params["fall_confirmation_ticks"])
            )
        else:
            confirmed_fall = UNKNOWN
        nearest_uav, uav_distance = _nearest_entity(position, uavs)
        within_detection_radius = (
            confirmed_fall is True
            and position is not None
            and uav_distance is not None
            and uav_distance <= float(params["uav_detection_radius_m"])
        )
        if within_detection_radius:
            detection_dwell[pedestrian_id] += tick_step
        else:
            # Detection requires one uninterrupted confirmed-fall/UAV geometry
            # window.  Fall persistence is deliberately not reused here: a UAV
            # entering after a long unattended fall must still satisfy its own
            # observation dwell.
            detection_dwell[pedestrian_id] = 0
        if confirmed_fall is False:
            physical_detection: bool | str = False
        elif confirmed_fall is not True or position is None:
            physical_detection = UNKNOWN
        else:
            physical_detection = within_detection_radius and detection_dwell[
                pedestrian_id
            ] >= int(params["uav_detection_required_ticks"])
        # Detection is derived only from the confirmed physical fall, the
        # designated nearest UAV's observed 3-D range, and an uninterrupted
        # dwell.  Authored incident labels cannot assert this predicate.
        detection_active = physical_detection
        nearest_responder, responder_distance = _nearest_entity(position, responders)
        previous_distance = previous_responder_distance.get(pedestrian_id)
        responder_deployed = responder_distance is not None
        responder_approaching = (
            responder_distance is not None
            and previous_distance is not None
            and responder_distance
            < previous_distance - float(params["responder_approach_min_delta_m"])
        )
        first_responder_deployment = (
            responder_distance is not None and previous_distance is None
        )
        if responder_distance is None:
            previous_responder_distance.pop(pedestrian_id, None)
        else:
            previous_responder_distance[pedestrian_id] = responder_distance
        physical_dispatch_request = bool(
            responder_approaching or first_responder_deployment
        )
        if confirmed_fall is True and detection_active is True:
            if physical_dispatch_request:
                dispatch_latched[pedestrian_id] = True
            if responder_distance is None:
                physical_dispatch: bool | str = UNKNOWN
                dispatch_active: bool | str = UNKNOWN
            else:
                physical_dispatch = bool(dispatch_latched[pedestrian_id])
                dispatch_active = physical_dispatch
        elif confirmed_fall is False or detection_active is False:
            dispatch_latched[pedestrian_id] = False
            physical_dispatch = False
            dispatch_active = False
        else:
            physical_dispatch = UNKNOWN
            dispatch_active = UNKNOWN
        # "Responder arrival" means arrival for this dispatched incident, not an
        # ambulance that happened to be nearby beforehand.  It therefore requires
        # the physical fall + dispatch + same-tick responder distance chain;
        # structured incident flags never substitute for physical geometry.
        if responder_distance is not None:
            physical_arrived: bool | str = bool(
                fallen is True
                and dispatch_active is True
                and responder_distance <= float(params["responder_arrival_radius_m"])
            )
            arrived = physical_arrived
        elif fallen is False or dispatch_active is False:
            physical_arrived = UNKNOWN
            arrived = False
        else:
            physical_arrived = UNKNOWN
            arrived = UNKNOWN
        tick_step = int(profile["authoritative_tick_policy"]["step"])
        if arrived is True and confirmed_fall is True:
            responder_dwell[pedestrian_id] += tick_step
        else:
            responder_dwell[pedestrian_id] = 0
        if responder_dwell[pedestrian_id] >= int(
            params["handoff_required_dwell_ticks"]
        ):
            handoff_latched[pedestrian_id] = True
        eta_s = UNKNOWN
        if responder_distance is not None:
            responder_nominal_speed = float(params["responder_nominal_speed_mps"])
            eta_s = (
                responder_distance / responder_nominal_speed
                if responder_nominal_speed > 0
                else UNKNOWN
            )
        if fallen is False:
            handoff_latched[pedestrian_id] = False
        # Handoff is exclusively a physical arrival dwell.  Authored incident
        # flags cannot replace a missing responder pose or motion history.
        if responder_distance is not None:
            handoff_complete: bool | str = bool(handoff_latched[pedestrian_id])
        elif fallen is False or dispatch_active is False:
            handoff_complete = False
        else:
            handoff_complete = UNKNOWN
        # Resolution is a terminal latch produced only by the physical response
        # chain.  In particular, mission_state.medical_resolved and incident
        # flags are not inputs: without a responder pose, verified arrival, and
        # the configured handoff dwell, resolution remains false/unknown.
        if handoff_latched[pedestrian_id]:
            medical_resolved: bool | str = True
        elif fallen is False or confirmed_fall is False or dispatch_active is False:
            medical_resolved = False
        elif responder_distance is None:
            medical_resolved = UNKNOWN
        else:
            medical_resolved = False
        values = {
            "posture": posture,
            "health_state": health_state,
            "speed_mps": _round(speed),
            "fallen_or_injured": fallen,
            "fall_persistence_ticks": fall_persistence[pedestrian_id],
            "confirmed_fall": confirmed_fall,
            "static_duration_ticks": static_duration[pedestrian_id],
            "nearest_uav_id": nearest_uav.get("entity_id") if nearest_uav else UNKNOWN,
            "nearest_uav_distance_m": _round(uav_distance),
            "detection_dwell_ticks": detection_dwell[pedestrian_id],
            "detection_active": detection_active,
            "nearest_responder_id": nearest_responder.get("entity_id")
            if nearest_responder
            else UNKNOWN,
            "nearest_responder_distance_m": _round(responder_distance),
            "responder_eta_s": _round(eta_s),
            "responder_deployed": responder_deployed,
            "responder_geometry_available": responder_distance is not None,
            "first_responder_deployment": first_responder_deployment,
            "responder_approaching": responder_approaching,
            "physical_dispatch_trigger": physical_dispatch_request,
            "physical_dispatch_latched": bool(dispatch_latched[pedestrian_id]),
            "dispatch_active": dispatch_active,
            "responder_arrived": arrived,
            "responder_arrival_dwell_ticks": responder_dwell[pedestrian_id],
            "handoff_dwell_ticks": responder_dwell[pedestrian_id],
            "handoff_complete": handoff_complete,
            "medical_resolved": medical_resolved,
            "response_rule_parameters": {
                "fall_confirmation_ticks": int(params["fall_confirmation_ticks"]),
                "uav_detection_radius_m": float(params["uav_detection_radius_m"]),
                "uav_detection_required_ticks": int(
                    params["uav_detection_required_ticks"]
                ),
                "responder_arrival_radius_m": float(
                    params["responder_arrival_radius_m"]
                ),
                "responder_approach_min_delta_m": float(
                    params["responder_approach_min_delta_m"]
                ),
                "handoff_required_dwell_ticks": int(
                    params["handoff_required_dwell_ticks"]
                ),
            },
        }
        source_refs = [
            f"truth_frames.jsonl#tick={tick}#entity={pedestrian_id}#path=truth_pose",
            f"truth_frames.jsonl#tick={tick}#entity={pedestrian_id}#path=pedestrian_state",
        ]
        nearest_uav_id = nearest_uav.get("entity_id") if nearest_uav else None
        if isinstance(nearest_uav_id, str):
            source_refs.append(
                f"truth_frames.jsonl#tick={tick}#entity={nearest_uav_id}#path=truth_pose"
            )
        nearest_responder_id = (
            nearest_responder.get("entity_id") if nearest_responder else None
        )
        if isinstance(nearest_responder_id, str):
            source_refs.append(
                f"truth_frames.jsonl#tick={tick}#entity={nearest_responder_id}#path=truth_pose"
            )
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "medical_response",
                pedestrian_id,
                "pedestrian",
                "derived_from_observed",
                "domain_state.medical.fall_static_responder_distance",
                RULE_VERSION,
                values,
                source_refs=source_refs,
            )
        )
    return rows


def _crowd_row(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    pedestrians: Sequence[Mapping[str, Any]],
    previous_positions: Mapping[str, list[float]],
    crowd_initial: dict[str, Any],
) -> dict[str, Any]:
    params = profile["domain_models"]["crowd"]
    if "cohort_member_ids" not in crowd_initial:
        explicit_members = [
            pedestrian
            for pedestrian in pedestrians
            if not _is_background_pedestrian(pedestrian)
        ]
        crowd_initial["cohort_member_ids"] = sorted(
            str(pedestrian["entity_id"]) for pedestrian in explicit_members
        )
        crowd_initial["cohort_definition_source"] = (
            "observed_non_background_pedestrian_roster_at_first_authoritative_tick"
            if explicit_members
            else "unavailable"
        )
        crowd_initial["initial_positions"] = {
            str(pedestrian["entity_id"]): _position(pedestrian)
            for pedestrian in explicit_members
            if _position(pedestrian) is not None
        }
        initial_positions = list(crowd_initial["initial_positions"].values())
        if initial_positions:
            crowd_initial["hazard_reference_enu_m"] = _centroid(initial_positions)
        crowd_initial["safe_zone_condition_dwell_ticks"] = 0
        crowd_initial["last_tick"] = tick
    cohort_ids = tuple(str(item) for item in crowd_initial["cohort_member_ids"])
    pedestrians_by_id = {
        str(pedestrian["entity_id"]): pedestrian for pedestrian in pedestrians
    }
    cohort_pedestrians = [
        pedestrians_by_id[member_id]
        for member_id in cohort_ids
        if member_id in pedestrians_by_id
    ]
    missing_cohort_ids = sorted(set(cohort_ids) - set(pedestrians_by_id))
    moving = [
        pedestrian
        for pedestrian in cohort_pedestrians
        if _speed(pedestrian) is not None and (_speed(pedestrian) or 0.0) > 0.2
    ]
    positions = [_position(pedestrian) for pedestrian in cohort_pedestrians]
    known_positions = [position for position in positions if position is not None]
    all_cohort_positions_observed = (
        bool(cohort_ids)
        and not missing_cohort_ids
        and len(known_positions) == len(cohort_ids)
    )
    centroid = _centroid(known_positions)
    hazard_reference = crowd_initial.get("hazard_reference_enu_m")
    safe_zone_count: Any = UNKNOWN
    hazard_zone_count: Any = UNKNOWN
    displacement_by_pedestrian: dict[str, Any] = {}
    continuous_movers = 0
    if hazard_reference is not None:
        hazard_radius = float(params["crowd_hazard_radius_m"])
        safe_radius = float(params["safe_zone_distance_m"])
        hazard_zone_count = sum(
            1
            for position in known_positions
            if math.dist(position[:2], hazard_reference[:2]) <= hazard_radius
        )
        safe_zone_count = sum(
            1
            for position in known_positions
            if math.dist(position[:2], hazard_reference[:2]) >= safe_radius
        )
    for pedestrian in cohort_pedestrians:
        pedestrian_id = str(pedestrian["entity_id"])
        position = _position(pedestrian)
        initial = (crowd_initial.get("initial_positions") or {}).get(pedestrian_id)
        displacement = (
            math.dist(position[:2], initial[:2])
            if position is not None and initial is not None
            else None
        )
        displacement_by_pedestrian[pedestrian_id] = _round(displacement)
        previous = previous_positions.get(pedestrian_id)
        speed = _speed(pedestrian)
        if (
            previous is not None
            and position is not None
            and math.dist(previous[:2], position[:2])
            >= float(params["continuous_movement_min_delta_m"])
            and speed is not None
            and speed >= float(params["evacuation_speed_threshold_mps"])
        ):
            continuous_movers += 1
    structured_evacuation_values = [
        _state_bool(
            _nested_mapping(pedestrian, ("pedestrian_state",)) or {},
            ("evacuation_active",),
        )
        for pedestrian in cohort_pedestrians
    ]
    structured_safe_zone_values = [
        _state_bool(
            _nested_mapping(pedestrian, ("pedestrian_state",)) or {},
            ("safe_zone_reached",),
        )
        for pedestrian in cohort_pedestrians
    ]
    structured_evacuation_count = sum(
        1 for value in structured_evacuation_values if value is True
    )
    evacuation_active = _truth_or(*structured_evacuation_values)
    safe_zone_required_count = (
        max(
            1,
            math.ceil(len(cohort_ids) * float(params["safe_zone_reached_fraction"])),
        )
        if cohort_ids
        else 0
    )
    if hazard_reference is None or not cohort_ids:
        physical_safe_zone_condition: bool | str = UNKNOWN
    elif not all_cohort_positions_observed:
        physical_safe_zone_condition = UNKNOWN
    else:
        physical_safe_zone_condition = safe_zone_count >= safe_zone_required_count
    previous_tick = crowd_initial.get("last_tick")
    elapsed_ticks = (
        max(0, tick - int(previous_tick)) if isinstance(previous_tick, int) else 0
    )
    if physical_safe_zone_condition is True:
        crowd_initial["safe_zone_condition_dwell_ticks"] = (
            int(crowd_initial.get("safe_zone_condition_dwell_ticks", 0)) + elapsed_ticks
        )
    else:
        # False, missing members, or unknown positions all break continuity.
        crowd_initial["safe_zone_condition_dwell_ticks"] = 0
    crowd_initial["last_tick"] = tick
    safe_zone_condition_dwell_ticks = int(
        crowd_initial.get("safe_zone_condition_dwell_ticks", 0)
    )
    required_safe_zone_dwell_ticks = int(params["safe_zone_required_dwell_ticks"])
    physical_safe_zone_reached: bool | str = (
        safe_zone_condition_dwell_ticks >= required_safe_zone_dwell_ticks
        if physical_safe_zone_condition is True
        else False
        if physical_safe_zone_condition is False
        else UNKNOWN
    )
    structured_safe_zone_reached: bool | str = (
        all(value is True for value in structured_safe_zone_values)
        if structured_safe_zone_values
        and all(value != UNKNOWN for value in structured_safe_zone_values)
        else UNKNOWN
    )
    # A frozen physical cohort is authoritative. Missing cohort members or
    # unobserved positions remain unknown; typed state never hides that gap.
    safe_zone_reached = (
        physical_safe_zone_reached if cohort_ids else structured_safe_zone_reached
    )
    values = {
        "crowd_id": (
            stable_identifier("crowd", inputs.episode_id, cohort_ids)
            if cohort_ids
            else None
        ),
        "crowd_ontology_class_id": "world:Crowd",
        "pedestrian_count": len(cohort_ids),
        "observed_pedestrian_count_total": len(pedestrians),
        "cohort_member_ids": list(cohort_ids),
        "cohort_member_count": len(cohort_ids),
        "observed_cohort_member_count": len(cohort_pedestrians),
        "missing_cohort_member_ids": missing_cohort_ids,
        "cohort_definition_source": crowd_initial.get(
            "cohort_definition_source", UNKNOWN
        ),
        "moving_count": len(moving),
        "evacuating_state_count": structured_evacuation_count,
        "structured_evacuation_state_count": structured_evacuation_count,
        "continuous_movement_count": continuous_movers,
        "evacuation_active": evacuation_active,
        "crowd_centroid_enu_m": _round_list(centroid),
        "hazard_reference_enu_m": _round_list(hazard_reference),
        "hazard_zone_count": hazard_zone_count,
        "safe_zone_count": safe_zone_count,
        "safe_zone_required_count": safe_zone_required_count,
        "safe_zone_reached_fraction": float(params["safe_zone_reached_fraction"]),
        "safe_zone_distance_m": float(params["safe_zone_distance_m"]),
        "safe_zone_condition_dwell_ticks": safe_zone_condition_dwell_ticks,
        "safe_zone_required_dwell_ticks": required_safe_zone_dwell_ticks,
        "safe_zone_reached": safe_zone_reached,
        "structured_safe_zone_reached": structured_safe_zone_reached,
        "pedestrian_displacement_m": dict(sorted(displacement_by_pedestrian.items())),
        "safe_zone_geometry_source": "fixed_initial_hazard_reference"
        if hazard_reference is not None
        else UNKNOWN,
    }
    return _observation_row(
        inputs,
        common,
        tick,
        "crowd_evacuation",
        "pedestrian_crowd",
        "crowd",
        "derived_from_observed" if cohort_ids else "unknown",
        "domain_state.crowd.hazard_and_safe_zone_counts",
        RULE_VERSION,
        values,
        source_refs=["truth_frames.jsonl"],
    )


def _is_background_pedestrian(entity: Mapping[str, Any]) -> bool:
    return isinstance(entity.get("background_pedestrian"), Mapping)


def _security_rows(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    uavs: Sequence[Mapping[str, Any]],
    facilities: Sequence[Mapping[str, Any]],
    communication_by_tick_entity: Mapping[tuple[int, str], Mapping[str, Any]],
    lockout_duration: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tick_step = int(profile["authoritative_tick_policy"]["step"])
    for entity, subject_category, rule_id in (
        *(
            (
                station,
                "ground_station",
                "domain_state.security.gcs_auth_command_integrity",
            )
            for station in facilities
        ),
        *((uav, "uav", "domain_state.security.auth_command_integrity") for uav in uavs),
    ):
        entity_id = str(entity["entity_id"])
        communication = communication_by_tick_entity.get((tick, entity_id))
        communication_station_id = _explicit_identifier_or_none(
            communication.get("station_id")
            if subject_category == "uav" and communication is not None
            else None
        )
        structured = _nested_mapping(entity, ("security_state",)) or {}
        operational_state = _exact_enum(
            structured,
            ("mode", "status", "threat", "condition"),
            {
                "nominal",
                "normal",
                "clear",
                "none",
                "gcs_compromised",
                "jamming",
                "unauthorized_command",
                "command_integrity_violation",
                "command_lockout",
            },
        )

        compromised = _state_bool(structured, ("gcs_compromised",))
        unauthorized = _state_bool(structured, ("unauthorized_command",))
        integrity_violation = _state_bool(
            structured,
            ("command_integrity_violation",),
        )
        jamming = _state_bool(structured, ("jamming_active",))
        if (
            subject_category == "uav"
            and jamming is True
            and communication_station_id is None
        ):
            raise DomainStateSimulationError(
                f"jamming-active aircraft {entity_id!r} at tick {tick} lacks an "
                "explicit communication-station identity"
            )
        locked = _state_bool(
            structured,
            ("lockout_active", "command_lockout"),
        )
        if locked is True:
            lockout_duration[entity_id] += tick_step
        elif locked is False:
            lockout_duration[entity_id] = 0

        # Command authorization and station compromise are separate observations.
        explicit_compromise = compromised
        if explicit_compromise is True:
            auth_score = 0.25
        elif explicit_compromise is False:
            auth_score = 1.0
        else:
            auth_score = None
        if integrity_violation is True:
            command_score = 0.1
        elif integrity_violation is False:
            command_score = 1.0
        else:
            command_score = None
        auth_state = _auth_state_from_score(auth_score)
        command_integrity: bool | str = (
            command_score >= 0.8 if command_score is not None else UNKNOWN
        )
        command_source_valid: bool | str = (
            not unauthorized if isinstance(unauthorized, bool) else UNKNOWN
        )
        spectrum_interference_ratio: float | str = (
            1.0 if jamming is True else 0.0 if jamming is False else UNKNOWN
        )
        has_structured_evidence = any(
            value != UNKNOWN
            for value in (
                operational_state,
                compromised,
                unauthorized,
                integrity_violation,
                jamming,
                locked,
            )
        )
        source_class = (
            "observed_simulator_truth" if has_structured_evidence else "unknown"
        )
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "security_command",
                entity_id,
                subject_category,
                source_class,
                rule_id,
                RULE_VERSION,
                {
                    "actor_id": entity_id if subject_category == "uav" else None,
                    "actor_ontology_class_id": (
                        "world:UnmannedAircraft" if subject_category == "uav" else None
                    ),
                    "ground_control_station_id": (
                        entity_id if subject_category == "ground_station" else None
                    ),
                    "ground_control_station_ontology_class_id": (
                        "world:GroundControlStation"
                        if subject_category == "ground_station"
                        else None
                    ),
                    "communication_station_id": communication_station_id,
                    "communication_station_ontology_class_id": (
                        "world:CommunicationStation"
                        if isinstance(communication_station_id, str)
                        and communication_station_id
                        else None
                    ),
                    "operational_state": operational_state,
                    "auth_score": _round(auth_score),
                    "auth_state": auth_state,
                    "command_integrity_score": _round(command_score),
                    "command_integrity": command_integrity,
                    "command_source_valid": command_source_valid,
                    "jamming_indicator": jamming,
                    "spectrum_interference_ratio": spectrum_interference_ratio,
                    "lockout_active": locked,
                    "lockout_duration_ticks": (
                        lockout_duration[entity_id] if locked != UNKNOWN else UNKNOWN
                    ),
                    "missing_inputs": []
                    if source_class != "unknown"
                    else ["truth_frames.entities[].security_state"],
                },
                source_refs=[
                    "truth_frames.jsonl",
                    *(
                        ["communication_state.jsonl" f"#tick={tick}#entity={entity_id}"]
                        if communication is not None
                        else []
                    ),
                ],
            )
        )
    return rows


def _payload_rows(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    weather: Mapping[str, Any] | None,
    uavs: Sequence[Mapping[str, Any]],
    previous_swing_angle: dict[str, float],
    battery_soc_by_uav: dict[str, float | None],
    preflight_energy_gap_by_uav: Mapping[str, str],
    charging_activity_by_tick: Mapping[int, Mapping[str, bool | str]],
    charging_issues_by_tick: Mapping[int, Mapping[str, str]],
    global_energy_checkpoints: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
    global_energy_gaps: Mapping[str, str] | None = None,
    global_lifetimes: Mapping[str, Mapping[str, Any]] | None = None,
    last_energy_tick_by_uav: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = profile["domain_models"]["payload_energy"]
    for uav in uavs:
        uav_id = str(uav["entity_id"])
        control_state = _nested_mapping(uav, ("control_state",)) or {}
        attitude_unstable = _state_bool(control_state, ("attitude_unstable",))
        operational_state = _exact_enum(
            control_state,
            ("operational_state", "flight_mode", "status"),
            {
                "nominal",
                "grounded",
                "airborne",
                "hovering",
                "landing",
                "failsafe",
                "attitude_unstable",
            },
        )
        wind = (
            _number(weather.get("wind_speed")) if isinstance(weather, Mapping) else None
        )
        speed = _speed(uav)
        if wind is not None and speed is not None:
            swing_angle = min(
                float(params["max_payload_swing_deg"]),
                wind * float(params["wind_to_swing_deg_per_mps"])
                + speed * float(params["speed_to_swing_deg_per_mps"]),
            )
            if attitude_unstable is True:
                swing_angle = min(
                    float(params["max_payload_swing_deg"]),
                    swing_angle + float(params["attitude_unstable_swing_bonus_deg"]),
                )
            swing_state: Any = (
                "high"
                if swing_angle >= float(params["swing_high_threshold_deg"])
                else "nominal"
            )
            missing: list[str] = []
        else:
            swing_angle = UNKNOWN
            swing_state = UNKNOWN
            missing = [
                "weather_meta.wind_speed"
                if wind is None
                else "truth_frames.entities[].speed_mps"
            ]
        tick_step = int(profile["authoritative_tick_policy"]["step"])
        if isinstance(swing_angle, (int, float)):
            swing_rate = (
                float(swing_angle) - float(previous_swing_angle[uav_id])
            ) / max(1, tick_step)
            previous_swing_angle[uav_id] = float(swing_angle)
        else:
            swing_rate = UNKNOWN
        observed_temperature = (
            _number(weather.get("temperature_c"))
            if isinstance(weather, Mapping)
            else None
        )
        interval_start = max(
            int(profile["authoritative_tick_policy"]["start"]), tick - tick_step
        )
        previous_soc = battery_soc_by_uav.get(uav_id)
        checkpoint = (global_energy_checkpoints or {}).get(uav_id, {}).get(tick)
        world_lifetime = (global_lifetimes or {}).get(uav_id)
        initial_source_step: Mapping[str, Any] | None = None
        last_energy_tick = (last_energy_tick_by_uav or {}).get(uav_id)
        if checkpoint is not None and (uav_id not in battery_soc_by_uav or
                (last_energy_tick is not None and last_energy_tick < interval_start)):
            # Resume at the beginning of this interval, including the hidden
            # world history. A birth inside the first interval has no preceding
            # grid checkpoint; that exact partial source step is already done.
            interval_checkpoint = (global_energy_checkpoints or {})[uav_id].get(interval_start)
            if interval_checkpoint is None:
                interval_checkpoint = checkpoint
                initial_source_step = checkpoint
                interval_start = tick
            previous_soc = float(interval_checkpoint["state_of_charge_ratio"])
            battery_soc_by_uav[uav_id] = previous_soc
        if uav_id not in battery_soc_by_uav:
            roster_entity = inputs.roster_entities.get(uav_id)
            activation_tick = (
                roster_entity.get("activation_tick")
                if isinstance(roster_entity, Mapping)
                else None
            )
            if world_lifetime is not None:
                activation_tick = world_lifetime["first_active_grid_tick"]
                source_gap = (global_energy_gaps or {}).get(uav_id)
                if source_gap is not None:
                    missing.append("global_energy_source:"+source_gap)
            if type(activation_tick) is not int:
                missing.append("global_entity_roster.entities[].activation_tick")
            elif activation_tick > tick:
                raise DomainStateSimulationError(
                    f"{inputs.episode_id}: UAV {uav_id} precedes activation at tick {tick}"
                )
            elif activation_tick < interval_start:
                missing.append("pre_first_visible_energy_history_unavailable")
            else:
                previous_soc = float(params["initial_soc_ratio"])
                interval_start = activation_tick
        if previous_soc is None:
            missing.append("prior_energy_state_unresolved")
            preflight_gap = preflight_energy_gap_by_uav.get(uav_id)
            if preflight_gap is not None:
                missing.append(f"preflight_source:{preflight_gap}")
        elapsed_ticks = tick - interval_start
        charging_active = charging_activity_by_tick[tick].get(uav_id, False)
        interval_charging = [
            charging_activity_by_tick[source_tick].get(uav_id, False)
            for source_tick in range(interval_start, tick)
        ]
        charge_issues = {
            charging_issues_by_tick[source_tick][uav_id]
            for source_tick in range(interval_start, tick + 1)
            if uav_id in charging_issues_by_tick[source_tick]
        }
        missing.extend(sorted(charge_issues))
        charged = (
            UNKNOWN
            if UNKNOWN in interval_charging
            else float(params["soc_charge_per_tick"])
            * sum(state is True for state in interval_charging)
        )
        if observed_temperature is None:
            temperature = UNKNOWN
            soc = UNKNOWN
            consumed = UNKNOWN
            derating = UNKNOWN
            range_state = UNKNOWN
            predicted_range_m = UNKNOWN
            planned_route_distance_m = _route_length_m(uav.get("planned_route_waypoints_enu_m") if world_lifetime is not None else uav.get("route_waypoints_enu_m"))
            range_insufficient = UNKNOWN
            missing.append("weather_meta.temperature_c_or_uav_battery_derating_state")
            battery_soc_by_uav[uav_id] = None
        else:
            temperature = observed_temperature
            derating = min(
                float(params["max_power_derating_ratio"]),
                max(
                    0.0,
                    (temperature - float(params["derating_start_temperature_c"]))
                    * float(params["temperature_to_derating_ratio"]),
                ),
            )
            speed_factor = (
                1.0 + min(
                    float(params["maximum_speed_energy_factor"]),
                    max(0.0, speed)
                    * float(params["speed_energy_factor_per_mps"]),
                )
                if speed is not None
                else UNKNOWN
            )
            temperature_energy_factor = 1.0 + min(
                float(params["maximum_derating_energy_factor"]),
                derating * float(params["derating_energy_factor_per_ratio"]),
            )
            consumed = (
                float(params["soc_consumption_per_tick"])
                * elapsed_ticks
                * speed_factor
                * temperature_energy_factor
                if speed is not None
                else UNKNOWN
            )
            if speed is None:
                missing.append("truth_frames.entities[].truth_pose.velocity_enu_mps")
            soc = (
                min(
                    1.0,
                    max(
                        float(params["minimum_soc"]),
                        previous_soc - consumed + charged,
                    ),
                )
                if previous_soc is not None
                and isinstance(consumed, float)
                and isinstance(charged, float)
                else UNKNOWN
            )
            battery_soc_by_uav[uav_id] = soc if isinstance(soc, float) else None
            if initial_source_step is not None:
                consumed = float(initial_source_step["energy_consumed_ratio"])
                charged = float(initial_source_step["energy_charged_ratio"])
            range_state = "derated" if derating > 0 else "nominal"
            range_scale = max(
                float(params["minimum_range_scale"]),
                1.0 - derating * float(params["derating_to_range_loss_factor"]),
            )
            predicted_range_m = (
                float(params["nominal_range_m"]) * soc * range_scale
                if isinstance(soc, float)
                else UNKNOWN
            )
            planned_route_distance_m = _route_length_m(uav.get("planned_route_waypoints_enu_m") if world_lifetime is not None else uav.get("route_waypoints_enu_m"))
            range_insufficient = (
                predicted_range_m
                < planned_route_distance_m + float(params["required_route_reserve_m"])
                if planned_route_distance_m is not None
                and isinstance(predicted_range_m, float)
                else UNKNOWN
            )
        row_source_class = (
            "simulated_derived"
            if isinstance(swing_angle, (int, float))
            or isinstance(temperature, (int, float))
            or isinstance(derating, (int, float))
            else "unknown"
        )
        if last_energy_tick_by_uav is not None:
            last_energy_tick_by_uav[uav_id] = tick
        rows.append(
            _observation_row(
                inputs,
                common,
                tick,
                "payload_energy",
                str(uav["entity_id"]),
                "uav",
                row_source_class,
                "domain_state.payload.wind_swing_derating_range",
                RULE_VERSION,
                {
                    "operational_state": operational_state,
                    "wind_speed_mps": _round(wind),
                    "speed_mps": _round(speed),
                    "payload_swing_angle_deg": _round(swing_angle),
                    "payload_swing_rate_deg_per_tick": _round(swing_rate),
                    "payload_swing_state": swing_state,
                    "temperature_c": _round(temperature),
                    "state_of_charge_ratio": _round(soc),
                    "energy_consumed_ratio": _round(consumed),
                    "energy_charged_ratio": _round(charged),
                    "temperature_energy_factor": _round(
                        temperature_energy_factor
                        if observed_temperature is not None
                        else UNKNOWN
                    ),
                    "charging_active": charging_active,
                    "power_derating_ratio": _round(derating),
                    "range_state": range_state,
                    "predicted_range_m": _round(predicted_range_m),
                    "planned_route_distance_m": _round(planned_route_distance_m),
                    "range_insufficient": range_insufficient,
                    "missing_inputs": sorted(set(missing)),
                    **({"energy_history_source": "global_task_plan_episode_weather_and_complete_charging_plan",
                        "world_birth_episode_s": world_lifetime["episode_birth_s"],
                        "energy_history_source_gap": (global_energy_gaps or {}).get(uav_id),
                        "battery_telemetry_measured": False} if world_lifetime is not None else {}),
                    **({"world_history_consumption_since_birth_ratio": _round(checkpoint["consumption_since_world_birth_ratio"])} if checkpoint is not None else {}),
                },
                quality={
                    "status": "unknown",
                    "missing_inputs": sorted(set(missing)),
                }
                if missing
                else None,
                source_refs=[
                    "truth_frames.jsonl",
                    "weather_meta.jsonl",
                    f"aw_data/charging_supplement/{inputs.episode_id}/charging_service_plan.json",
                    *(["aw_data/uav_outputs/donghu_uav_flow_270s/uav_task_plan.json",
                       "Dataset/semantic_simulation/p09_core_sources.py#world_energy_history"] if world_lifetime is not None else []),
                ],
            )
        )
    return rows


def _av_safe_stop_rows(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    frame: Mapping[str, Any],
    vehicles: Sequence[Mapping[str, Any]],
    safe_stop_dwell: dict[str, int],
    fault_start_tick: dict[str, int],
) -> list[dict[str, Any]]:
    params = profile["domain_models"]["av_safe_stop"]
    active_incidents = frame.get("sumo_active_incidents")
    incident_sensor_fault = _incident_has_class(
        active_incidents,
        {"av_sensor_fault_stop"},
    )
    explicit_fault_ids = sorted(
        str(vehicle["entity_id"])
        for vehicle in vehicles
        if _state_bool(
            _nested_mapping(vehicle, ("vehicle_state",)) or {},
            ("sensor_fault",),
        )
        is True
    )
    sensor_fault = _truth_or(incident_sensor_fault, bool(explicit_fault_ids))
    affected_ids = (
        sorted(_affected_vehicle_ids(active_incidents, vehicles))
        if incident_sensor_fault
        else explicit_fault_ids
    )
    candidate_vehicles = [
        vehicle
        for vehicle in vehicles
        if str(vehicle["entity_id"]) in set(affected_ids)
    ]
    stopped_ids = [
        str(vehicle["entity_id"])
        for vehicle in candidate_vehicles
        if (
            _speed(vehicle) is not None
            and (_speed(vehicle) or 0.0) <= float(params["stopped_speed_threshold_mps"])
        )
    ]
    tick_step = int(profile["authoritative_tick_policy"]["step"])
    subject_id = "av_safe_stop_system"
    if sensor_fault is True and subject_id not in fault_start_tick:
        fault_start_tick[subject_id] = tick
    if sensor_fault is False:
        fault_start_tick.pop(subject_id, None)
    physical_mrm_active: bool | str = (
        sensor_fault is True
        and subject_id in fault_start_tick
        and tick > fault_start_tick[subject_id]
        if sensor_fault != UNKNOWN
        else UNKNOWN
    )
    explicit_mrm_active = _any_exact_state_bool(
        candidate_vehicles,
        "vehicle_state",
        ("minimal_risk_maneuver_active",),
    )
    mrm_active = _truth_or(physical_mrm_active, explicit_mrm_active)
    if mrm_active is True and stopped_ids:
        safe_stop_dwell[subject_id] += tick_step
    elif mrm_active is False or not stopped_ids:
        safe_stop_dwell[subject_id] = 0
    physical_warning_effective: bool | str = (
        safe_stop_dwell[subject_id] >= int(params["warning_effective_dwell_ticks"])
        if mrm_active is True
        else False
        if mrm_active is False
        else UNKNOWN
    )
    warning_effective = _truth_or(
        physical_warning_effective,
        _any_exact_state_bool(candidate_vehicles, "vehicle_state", ("warning_active",)),
    )
    explicit_stopped_safe = _any_exact_state_bool(
        candidate_vehicles,
        "vehicle_state",
        ("stopped_safe",),
    )
    physical_stopped_safe: bool | str = (
        bool(stopped_ids) and warning_effective is True
        if mrm_active is True
        else False
        if mrm_active is False
        else UNKNOWN
    )
    stopped_safe = _truth_or(physical_stopped_safe, explicit_stopped_safe)
    safe_stop_failed = _any_exact_state_bool(
        candidate_vehicles,
        "vehicle_state",
        ("safe_stop_failed",),
    )
    return [
        _observation_row(
            inputs,
            common,
            tick,
            "av_safe_stop",
            subject_id,
            "vehicle_safety",
            "derived_from_observed"
            if isinstance(active_incidents, list)
            else "unknown",
            "domain_state.vehicle.av_sensor_fault_mrm_safe_stop",
            RULE_VERSION,
            {
                "sensor_fault_active": sensor_fault,
                "fault_start_tick": fault_start_tick.get(subject_id, UNKNOWN),
                "mrm_active": mrm_active,
                "affected_vehicle_ids": affected_ids[
                    : int(params["max_ids_per_observation"])
                ],
                "affected_vehicle_count": len(affected_ids),
                "stopped_vehicle_ids": sorted(stopped_ids)[
                    : int(params["max_ids_per_observation"])
                ],
                "stopped_vehicle_count": len(stopped_ids),
                "stop_dwell_ticks": safe_stop_dwell[subject_id],
                "warning_effective": warning_effective,
                "stopped_safe": stopped_safe,
                "safe_stop_failed": safe_stop_failed,
                "mean_affected_speed_mps": _round(
                    _mean([_speed(vehicle) for vehicle in candidate_vehicles])
                ),
                "mean_affected_accel_mps2": _round(
                    _mean([_vehicle_accel(vehicle) for vehicle in candidate_vehicles])
                ),
            },
            source_refs=["truth_frames.jsonl"],
        )
    ]


def _ambulance_priority_row(
    *,
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    tick: int,
    frame: Mapping[str, Any],
    vehicles: Sequence[Mapping[str, Any]],
    priority_state: dict[str, Any],
    medical_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    params = profile["domain_models"]["ambulance_priority"]
    active_incidents = frame.get("sumo_active_incidents")
    incident_priority_active = _incident_has_class(
        active_incidents,
        {"emergency_vehicle_priority"},
    )
    ambulances = sorted(
        [_vehicle for _vehicle in vehicles if _is_ambulance(_vehicle)],
        key=lambda vehicle: str(vehicle.get("entity_id") or ""),
    )
    civilians = [_vehicle for _vehicle in vehicles if _vehicle not in ambulances]
    primary_ambulance = ambulances[0] if len(ambulances) == 1 else None
    ambulance_position = _position(primary_ambulance) if primary_ambulance else None
    ambulance_edge = (
        _nested_value(primary_ambulance, ("sumo_vehicle", "sumo_edge_id"))
        if primary_ambulance
        else None
    )
    ambulance_lane = (
        _nested_value(primary_ambulance, ("sumo_vehicle", "sumo_lane_id"))
        if primary_ambulance
        else None
    )
    primary_ambulance_id = (
        str(primary_ambulance.get("entity_id")) if primary_ambulance else None
    )
    primary_ambulance_speed = _speed(primary_ambulance) if primary_ambulance else None
    primary_ambulance_state = (
        _nested_mapping(primary_ambulance, ("vehicle_state",))
        if primary_ambulance
        else None
    ) or {}
    passage_completed = _state_bool(
        primary_ambulance_state,
        ("ambulance_passage_completed",),
    )
    matching_medical_rows: list[Mapping[str, Any]] = []
    for row in medical_rows:
        values = row.get("values")
        if not isinstance(values, Mapping):
            continue
        if (
            values.get("confirmed_fall") is not True
            or values.get("dispatch_active") is not True
        ):
            continue
        if (
            primary_ambulance_id is None
            or values.get("nearest_responder_id") != primary_ambulance_id
        ):
            continue
        matching_medical_rows.append(row)
    matching_medical_rows.sort(key=lambda row: str(row.get("subject_id") or ""))
    medical_subject_id = (
        str(matching_medical_rows[0].get("subject_id"))
        if matching_medical_rows
        else None
    )
    medical_values = (
        matching_medical_rows[0].get("values")
        if matching_medical_rows
        and isinstance(matching_medical_rows[0].get("values"), Mapping)
        else {}
    )
    response_motion_observed = bool(
        medical_values.get("first_responder_deployment") is True
        or medical_values.get("responder_approaching") is True
        or (
            primary_ambulance_speed is not None
            and primary_ambulance_speed
            >= float(params.get("response_priority_min_speed_mps", 0.5))
        )
    )
    response_priority_active = bool(
        matching_medical_rows
        and (
            response_motion_observed
            or (
                priority_state.get("response_priority_latched") is True
                and priority_state.get("medical_subject_id") == medical_subject_id
            )
        )
    )
    priority_requested = bool(incident_priority_active or response_priority_active)
    priority_active = bool(priority_requested and passage_completed is not True)
    priority_basis = (
        "passage_completed"
        if passage_completed is True
        else "sumo_priority_incident"
        if incident_priority_active
        else "medical_dispatch_and_responder_motion"
        if response_priority_active
        else "inactive"
    )
    if priority_active and not priority_state.get("active"):
        priority_state.clear()
        priority_state.update(
            {
                "active": True,
                "response_priority_latched": response_priority_active,
                "priority_basis": priority_basis,
                "medical_subject_id": medical_subject_id or UNKNOWN,
                "activation_tick": tick,
                "start_position_enu_m": copy.deepcopy(ambulance_position),
                "priority_edge_id": ambulance_edge or UNKNOWN,
                "priority_lane_id": ambulance_lane or UNKNOWN,
                "yield_observed": False,
                "clearance_reached": False,
            }
        )
    elif not priority_active:
        priority_state.clear()

    nearby_civilians: list[Mapping[str, Any]] = []
    if ambulance_position is not None:
        for vehicle in civilians:
            position = _position(vehicle)
            if position is None:
                continue
            if math.dist(ambulance_position[:2], position[:2]) <= float(
                params["yield_observation_radius_m"]
            ):
                nearby_civilians.append(vehicle)
    yielding = []
    for vehicle in nearby_civilians:
        speed = _speed(vehicle)
        accel = _vehicle_accel(vehicle)
        if priority_active and (
            (speed is not None and speed <= float(params["yield_speed_threshold_mps"]))
            or (
                accel is not None
                and accel <= float(params["yield_deceleration_threshold_mps2"])
            )
        ):
            yielding.append(str(vehicle["entity_id"]))
    if yielding and priority_state.get("active"):
        priority_state["yield_observed"] = True

    civilian_positions = [
        _position(vehicle)
        for vehicle in nearby_civilians
        if _position(vehicle) is not None
    ]
    clearance_gap = UNKNOWN
    if ambulance_position is not None and civilian_positions:
        clearance_gap = min(
            math.dist(ambulance_position[:2], position[:2])
            for position in civilian_positions
        )
    start_position = priority_state.get("start_position_enu_m")
    travel_distance = (
        math.dist(ambulance_position[:2], start_position[:2])
        if ambulance_position is not None and isinstance(start_position, list)
        else None
    )
    if (
        priority_state.get("active")
        and priority_state.get("yield_observed")
        and travel_distance is not None
        and travel_distance >= float(params["clearance_travel_distance_m"])
    ):
        priority_state["clearance_reached"] = True
    clearance_reached = bool(priority_state.get("clearance_reached", False))
    return _observation_row(
        inputs,
        common,
        tick,
        "ambulance_priority",
        "ambulance_priority_system",
        "traffic_priority",
        "derived_from_observed"
        if isinstance(active_incidents, list) or ambulances
        else "unknown",
        "domain_state.traffic.ambulance_priority_yield_clearance",
        RULE_VERSION,
        {
            "priority_active": priority_active,
            "incident_priority_active": incident_priority_active,
            "response_priority_active": response_priority_active,
            "ambulance_passage_completed": passage_completed,
            "priority_basis": priority_state.get("priority_basis", priority_basis),
            "medical_subject_id": priority_state.get(
                "medical_subject_id", medical_subject_id or UNKNOWN
            ),
            "ambulance_vehicle_ids": sorted(
                str(vehicle["entity_id"]) for vehicle in ambulances
            ),
            "primary_ambulance_id": primary_ambulance.get("entity_id")
            if primary_ambulance
            else UNKNOWN,
            "priority_activation_tick": priority_state.get("activation_tick", UNKNOWN),
            "priority_edge_id": priority_state.get("priority_edge_id", UNKNOWN),
            "priority_lane_id": priority_state.get("priority_lane_id", UNKNOWN),
            "ambulance_in_priority_lane": bool(priority_active and ambulance_lane),
            "nearby_civilian_vehicle_ids": sorted(
                str(vehicle["entity_id"]) for vehicle in nearby_civilians
            ),
            "yielding_vehicle_ids": sorted(yielding)[
                : int(params["max_ids_per_observation"])
            ],
            "yielding_vehicle_count": len(yielding),
            "yield_observed_latched": bool(priority_state.get("yield_observed", False)),
            "clearance_gap_m": _round(clearance_gap),
            "ambulance_travel_since_priority_m": _round(travel_distance),
            "clearance_reached": clearance_reached,
            "ambulance_mean_speed_mps": _round(
                _mean([_speed(vehicle) for vehicle in ambulances])
            ),
            "civilian_mean_speed_mps": _round(
                _mean([_speed(vehicle) for vehicle in civilians])
            ),
        },
        source_refs=["truth_frames.jsonl"],
    )


def _gnss_values(
    *,
    uav: Mapping[str, Any],
    uav_id: str,
    position: list[float] | None,
    previous_error_range: tuple[float, float],
    params: Mapping[str, Any],
    tick: int,
    seed_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    raw_navigation = uav.get("navigation_state")
    if raw_navigation is not None and not isinstance(raw_navigation, Mapping):
        raise DomainStateSimulationError("GNSS navigation_state must be an object or null")
    navigation_state = raw_navigation
    if len(previous_error_range) != 2 or not all(
        type(value) in (int, float) and math.isfinite(value) and value >= 0
        for value in previous_error_range
    ) or previous_error_range[0] > previous_error_range[1]:
        raise DomainStateSimulationError("GNSS previous error range is invalid")
    missing_inputs: list[str] = []
    if position is None:
        missing_inputs.append("truth_frames.entities[].truth_pose.position_enu_m")
    if navigation_state is None:
        missing_inputs.append("truth_frames.entities[].navigation_state")
    navigation = navigation_state or {}
    input_keys = {
        "multipath_warning": ("multipath_warning",),
        "visual_relocalization": ("visual_relocalization", "visual_relocalization_active"),
        "gnss_spoofed": ("gnss_spoofed", "spoofing_active"),
        "geofence_alert": ("geofence_alert", "geofence_violation"),
        "mission_recovered": ("mission_recovered", "relocalization_complete"),
    }
    source_flags: dict[str, bool | str] = {}
    for field, keys in input_keys.items():
        declared = [navigation[key] for key in keys if key in navigation]
        if any(not (type(value) is bool or value is None or value == UNKNOWN)
               for value in declared):
            raise DomainStateSimulationError(f"GNSS {field} has an invalid source value")
        known = {value for value in declared if type(value) is bool}
        if len(known) > 1:
            raise DomainStateSimulationError(f"GNSS {field} aliases conflict")
        source_flags[field] = next(iter(known)) if known else UNKNOWN
        if source_flags[field] == UNKNOWN and navigation_state is not None:
            missing_inputs.append("truth_frames.entities[].navigation_state." + "|".join(keys))
    operational_keys = ("operational_state", "gnss_mode", "navigation_mode", "recovery_state")
    allowed_operational_states = {
        "nominal", "multipath_degraded", "spoofed", "geofence_alert",
        "relocalizing", "recovered",
    }
    known_operational_states: set[str] = set()
    for key in operational_keys:
        if key not in navigation or navigation[key] is None or navigation[key] == UNKNOWN:
            continue
        if not isinstance(navigation[key], str):
            raise DomainStateSimulationError(f"GNSS {key} has a non-string source value")
        state = navigation[key].strip().lower()
        if state not in allowed_operational_states:
            raise DomainStateSimulationError(f"GNSS {key} has an unsupported source state: {state}")
        known_operational_states.add(state)
    if len(known_operational_states) > 1:
        raise DomainStateSimulationError("GNSS operational-state aliases conflict")
    operational_state = next(iter(known_operational_states)) if known_operational_states else UNKNOWN
    if operational_state == UNKNOWN and navigation_state is not None:
        missing_inputs.append("truth_frames.entities[].navigation_state." + "|".join(operational_keys))
    base_error = float(params["base_error_m"])
    decay = float(params["error_recovery_decay"])
    if not 0 <= decay <= 1:
        raise DomainStateSimulationError("GNSS recovery decay must be between zero and one")
    flag_names = tuple(input_keys)
    choices = (
        (source_flags[name],) if type(source_flags[name]) is bool else (False, True)
        for name in flag_names
    )
    source_angle = _unit_interval("gnss_error_angle", seed_digest, uav_id, tick) * math.tau
    spoof_angle = float(params["spoof_offset_direction_deg"]) / 180.0 * math.pi
    candidates: list[tuple[float, tuple[float, float, float], str, str]] = []
    for flags in product(*choices):
        multipath_active, visual_relocalization, spoofed, geofence_alert, mission_recovered = flags
        target_error = base_error
        fsm_state = "nominal"
        if multipath_active:
            target_error = max(target_error, float(params["multipath_error_m"]))
            fsm_state = "multipath_degraded"
        if spoofed:
            target_error = max(target_error, float(params["spoofed_error_m"]))
            fsm_state = "spoofed"
        effective_geofence = geofence_alert or (
            spoofed and target_error >= float(params["geofence_alert_error_m"])
        )
        if effective_geofence:
            target_error = max(target_error, float(params["geofence_alert_error_m"]))
            fsm_state = "geofence_alert"
        if visual_relocalization:
            target_error = min(float(params["visual_relocalization_error_m"]),
                               max(base_error, target_error))
            fsm_state = "relocalizing"
        if mission_recovered:
            target_error = base_error
            fsm_state = "recovered"
        angle = spoof_angle if spoofed or effective_geofence else source_angle
        for previous_mag in set(previous_error_range):
            current_error = target_error
            if current_error < previous_mag and not spoofed and not effective_geofence:
                current_error = previous_mag * decay + current_error * (1.0 - decay)
            vector = (math.cos(angle) * current_error,
                      math.sin(angle) * current_error, 0.0)
            if current_error <= float(params["nominal_error_threshold_m"]):
                quality = "nominal"
            elif current_error <= float(params["degraded_error_threshold_m"]):
                quality = "degraded"
            else:
                quality = "spoofing_suspect"
            candidates.append((current_error, vector, quality, fsm_state))
    error_values = [candidate[0] for candidate in candidates]
    vector_values = [candidate[1] for candidate in candidates]
    quality_values = [candidate[2] for candidate in candidates]
    fsm_values = [candidate[3] for candidate in candidates]
    error = error_values[0] if all(value == error_values[0] for value in error_values) else None
    vector = vector_values[0] if all(value == vector_values[0] for value in vector_values) else None
    quality = quality_values[0] if all(value == quality_values[0] for value in quality_values) else UNKNOWN
    fsm_state = fsm_values[0] if all(value == fsm_values[0] for value in fsm_values) else UNKNOWN
    reported = ([position[index] + vector[index] for index in range(3)]
                if position is not None and vector is not None else None)
    resolved = not missing_inputs and error is not None and vector is not None and quality != UNKNOWN and fsm_state != UNKNOWN
    return (
        {
            "quality_level": quality,
            "truth_position_enu_m": _round_list(position),
            "reported_position_enu_m": _round_list(reported),
            "position_error_m": _round(error),
            "drift_offset_enu_m": _round_list(list(vector)) if vector is not None else UNKNOWN,
            "operational_state": operational_state,
            "multipath_warning": source_flags["multipath_warning"],
            "visual_relocalization": source_flags["visual_relocalization"],
            "gnss_spoofed": source_flags["gnss_spoofed"],
            "geofence_alert": source_flags["geofence_alert"],
            "mission_recovered": source_flags["mission_recovered"],
            "fsm_state": fsm_state,
            "recovery_decay": decay,
            "_next_error_range_m": (min(error_values), max(error_values)),
            "missing_inputs": missing_inputs,
        },
        {
            "status": "complete" if resolved else "unknown",
            "numeric_fsm_applied": error is not None,
            "rain_visibility_not_used_as_event_driver": True,
        },
        "simulated_derived" if error is not None else "unknown",
    )


def _observation_row(
    inputs: DomainEpisodeInputs,
    common: Mapping[str, Any],
    tick: int,
    family: str,
    subject_id: str,
    subject_category: str,
    source_class: str,
    rule_id: str,
    rule_version: str,
    values: Mapping[str, Any],
    *,
    quality: Mapping[str, Any] | None = None,
    source_refs: Sequence[str],
) -> dict[str, Any]:
    row = {
        "schema_name": "domain_state_observation",
        "schema_version": SCHEMA_VERSION,
        "episode_id": inputs.episode_id,
        "tick": tick,
        "observation_family": family,
        "subject_id": subject_id,
        "subject_category": subject_category,
        "source_class": source_class,
        "rule_id": rule_id,
        "rule_version": rule_version,
        "model_id": common["model_id"],
        "model_version": common["model_version"],
        "source_refs": list(source_refs),
        "values": copy.deepcopy(dict(values)),
        "quality": dict(quality or {"status": "complete"}),
    }
    row["observation_id"] = stable_identifier(
        "domain_state_observation",
        row["episode_id"],
        row["tick"],
        row["observation_family"],
        row["subject_id"],
        row["values"],
    )
    return row


def _build_summary(
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_family = Counter(str(row["observation_family"]) for row in rows)
    by_source = Counter(str(row["source_class"]) for row in rows)
    unknown_by_family: dict[str, int] = defaultdict(int)
    for row in rows:
        if _contains_unknown(row.get("values")):
            unknown_by_family[str(row["observation_family"])] += 1
    return {
        "schema_name": "domain_state_supplement_summary",
        "schema_version": SCHEMA_VERSION,
        "episode_id": inputs.episode_id,
        "profile_id": profile["profile_id"],
        "observed_tick_count": len(inputs.frames),
        "record_counts": {"domain_state_observations": len(rows)},
        "observation_family_counts": dict(sorted(by_family.items())),
        "source_class_counts": dict(sorted(by_source.items())),
        "unknown_value_counts_by_family": dict(sorted(unknown_by_family.items())),
        "forbidden_inputs_used": [],
        "parameter_governance": profile["parameter_governance"],
    }


def _build_manifest(
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    output_dir: Path,
    files: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_name": "domain_state_supplement_manifest",
        "schema_version": SCHEMA_VERSION,
        "episode_id": inputs.episode_id,
        "profile_id": profile["profile_id"],
        "model_id": common["model_id"],
        "model_version": common["model_version"],
        "source_policy": profile["source_policy"],
        "parameter_governance": profile["parameter_governance"],
        "input_files": {
            name: {"path": str(path)}
            for name, path in sorted(inputs.input_files.items())
        },
        "output_dir": str(output_dir),
        "artifacts": {
            name: {
                "path": name,
                "bytes": len(text.encode("utf-8")),
            }
            for name, text in sorted(files.items())
        },
    }


def _model_input_projection(
    *,
    manifest_projection: Mapping[str, Any],
    roster_entities: Mapping[str, Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    queue_window_by_tick: Mapping[int, Mapping[str, Any]],
    weather_by_tick: Mapping[int, Mapping[str, Any]],
    scene_setup: Mapping[str, Any] | None,
    charging_service_plan: Mapping[str, Any],
    preflight_uavs_by_tick: Mapping[int, Sequence[Mapping[str, Any]]],
    preflight_gaps_by_tick: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    roster_projection = [
        {
            "entity_id": entity_id,
            "entity_category": _entity_category(entity),
            "home_pad_entity_id": _nested_value(
                entity, ("lifecycle", "home_pad_entity_id")
            ),
            "first_visible_tick": _nested_value(
                entity, ("runtime_visibility", "first_visible_tick")
            ),
            "activation_tick": entity.get("activation_tick"),
        }
        for entity_id, entity in sorted(roster_entities.items())
    ]
    frame_projection: list[dict[str, Any]] = []
    for frame in sorted(frames, key=lambda row: int(row["tick"])):
        entities = []
        for entity in sorted(
            _current_entities(frame), key=lambda row: str(row["entity_id"])
        ):
            category = _entity_category(entity)
            if category not in {
                "uav",
                "vehicle",
                "pedestrian",
                "facility",
                "prop",
                "traffic_light",
                "ground_station",
                "crowd_anchor",
            }:
                continue
            projected = {
                "entity_id": str(entity["entity_id"]),
                "entity_category": category,
                "state": entity.get("state"),
                "entity_kind": _string_or_unknown(entity.get("entity_kind")),
                "logical_asset_id": entity.get("logical_asset_id"),
                "semantic_scope": copy.deepcopy(entity.get("semantic_scope")),
                "position_enu_m": _position(entity),
                "velocity_enu_mps": _velocity(entity),
                "sumo_vehicle": _sumo_projection(entity.get("sumo_vehicle")),
                "uav_global_flow": _uav_projection(entity.get("uav_global_flow")),
                "lifecycle": _lifecycle_projection(entity.get("lifecycle")),
                "route_waypoints_enu_m": copy.deepcopy(
                    entity.get("route_waypoints_enu_m")
                ),
                "background_pedestrian": copy.deepcopy(
                    entity.get("background_pedestrian")
                ),
            }
            for family in OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES:
                if family in entity:
                    projected[family] = sanitize_objective_input(entity[family])
            entities.append(projected)
        frame_projection.append(
            {
                "tick": int(frame["tick"]),
                "traffic_light_states": frame.get("sumo_traffic_light_states"),
                "sumo_active_incidents": _sumo_incident_projection(
                    frame.get("sumo_active_incidents")
                ),
                "entities": entities,
            }
        )
    return {
        "manifest": dict(manifest_projection),
        "roster_entities": roster_projection,
        "truth_frames": frame_projection,
        "queue_sampling_windows": [
            {"tick": tick, **dict(row)}
            for tick, row in sorted(queue_window_by_tick.items())
        ],
        "weather": [
            {
                "tick": tick,
                "condition": row.get("condition"),
                "rain": row.get("rain"),
                "wetness": row.get("wetness"),
                "fog_density": row.get("fog_density"),
                "dust": row.get("dust"),
                "wind_speed": row.get("wind_speed"),
                "visibility_m": row.get("visibility_m"),
                "temperature_c": row.get("temperature_c"),
            }
            for tick, row in sorted(weather_by_tick.items())
        ],
        "scene_setup_geometry": _scene_setup_projection(scene_setup),
        "charging_service_plan": copy.deepcopy(dict(charging_service_plan)),
        "pad_preflight_uavs": [
            {
                "tick": tick,
                "uavs": [
                    {
                        "entity_id": uav["entity_id"],
                        "truth_pose": {
                            "position_enu_m": copy.deepcopy(
                                uav["truth_pose"]["position_enu_m"]
                            )
                        },
                    }
                    for uav in uavs
                ],
            }
            for tick, uavs in sorted(preflight_uavs_by_tick.items())
        ],
        "pad_preflight_gaps": [
            {"tick": tick, "missing": dict(sorted(gaps.items()))}
            for tick, gaps in sorted(preflight_gaps_by_tick.items())
        ],
    }


def _load_scene_setup(
    episode_root: Path,
    profile: Mapping[str, Any],
    input_files: dict[str, Path],
) -> dict[str, Any] | None:
    scene_setup_file = episode_root / str(
        profile["inputs"].get("scene_setup", "scene_setup.json")
    )
    if scene_setup_file.is_file():
        input_files["scene_setup"] = scene_setup_file
        value = _load_json(scene_setup_file)
        return _scene_setup_projection(value)
    return None


def resolve_declared_input_path(
    episode_root: Path,
    declared: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    path = Path(declared)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"aw_data", "Dataset"}:
        return project_root / path
    return episode_root / path


def _scene_setup_projection(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    road_segment_anchors = _scene_setup_road_segment_anchors(value)
    return {"road_segment_anchors": road_segment_anchors}


def _scene_setup_road_segment_anchors(
    scene_setup: Mapping[str, Any],
) -> list[dict[str, str]]:
    anchors: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for collection_key in ("entities", "initial_entities"):
        entities = scene_setup.get(collection_key)
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, Mapping) or not _is_barrier_entity(entity):
                continue
            entity_id = entity.get("entity_id")
            placement = entity.get("placement")
            if not isinstance(entity_id, str) or not isinstance(placement, Mapping):
                continue
            edge_id = placement.get("edge_id")
            if not isinstance(edge_id, str) or not edge_id:
                continue
            key = (entity_id, edge_id)
            if key in seen:
                continue
            seen.add(key)
            anchors.append(
                {
                    "entity_id": entity_id,
                    "road_segment_id": edge_id,
                    "road_segment_ontology_class_id": "world:RoadSegment",
                }
            )
    anchors.sort(key=lambda item: (item["road_segment_id"], item["entity_id"]))
    return anchors


def _current_entities(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [copy.deepcopy(dict(entity)) for entity in _current_entity_views(frame)]


def _current_entity_views(frame: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entities = frame.get("entities")
    if not isinstance(entities, list):
        raise DomainStateSimulationError(
            f"truth frame tick {frame.get('tick')} lacks entities array"
        )
    result: list[Mapping[str, Any]] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            raise DomainStateSimulationError(
                f"truth frame tick {frame.get('tick')} entity {index} is not an object"
            )
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise DomainStateSimulationError(
                f"truth frame tick {frame.get('tick')} entity {index} lacks entity_id"
            )
        result.append(entity)
    return result


def _validate_frame_roster(
    frame: Mapping[str, Any],
    roster_entities: Mapping[str, Mapping[str, Any]],
) -> None:
    tick = frame.get("tick")
    seen: set[str] = set()
    for entity in _current_entity_views(frame):
        entity_id = str(entity["entity_id"])
        if entity_id in seen:
            raise DomainStateSimulationError(
                f"truth frame tick {tick} duplicates entity_id {entity_id}"
            )
        seen.add(entity_id)
        roster_entity = roster_entities.get(entity_id)
        if roster_entity is None:
            raise DomainStateSimulationError(
                f"truth frame tick {tick} contains entity absent from roster: {entity_id}"
            )
        frame_category = entity.get("entity_category")
        roster_category = roster_entity.get("entity_category")
        if (
            not isinstance(frame_category, str)
            or not frame_category
            or frame_category != roster_category
        ):
            raise DomainStateSimulationError(
                f"truth frame tick {tick} category differs from roster for {entity_id}"
            )


def _index_entities(values: Any, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise DomainStateSimulationError(f"{source}: entities must be an array")
    result: dict[str, dict[str, Any]] = {}
    for index, entity in enumerate(values):
        if not isinstance(entity, Mapping):
            raise DomainStateSimulationError(
                f"{source}: entity {index} must be an object"
            )
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise DomainStateSimulationError(
                f"{source}: entity {index} lacks entity_id"
            )
        category = entity.get("entity_category")
        if not isinstance(category, str) or not category:
            raise DomainStateSimulationError(
                f"{source}: entity {index} lacks entity_category"
            )
        if entity_id in result:
            raise DomainStateSimulationError(
                f"{source}: duplicate entity_id {entity_id}"
            )
        result[entity_id] = copy.deepcopy(dict(entity))
    return result


def _facility_entities(
    entities: Sequence[Mapping[str, Any]],
    subtypes: set[str] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entity in entities:
        if _entity_category(entity) not in {"facility", "ground_station"}:
            continue
        try:
            semantic_scope = validate_roster_facility_scope(entity)
        except FacilityScopeError as exc:
            raise DomainStateSimulationError(str(exc)) from exc
        subtype = str(semantic_scope["scope_subtype"])
        if subtypes is not None and subtype not in subtypes:
            continue
        result.append(copy.deepcopy(dict(entity)))
    return result


def _responder_entities(entities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(entity))
        for entity in entities
        if _entity_category(entity) == "vehicle" and _is_ambulance(entity)
    ]


def _uav_landing_requesters(
    pad: Mapping[str, Any],
    uavs: Sequence[Mapping[str, Any]],
) -> list[str]:
    pad_id = str(pad["entity_id"])
    pad_aliases = {pad_id}
    uav_pad_id = pad.get("uav_pad_id")
    if isinstance(uav_pad_id, str) and uav_pad_id:
        pad_aliases.add(uav_pad_id)
    requesters: list[str] = []
    for uav in uavs:
        lifecycle_pad = _nested_value(uav, ("lifecycle", "home_pad_entity_id"))
        flow = _nested_mapping(uav, ("uav_global_flow",)) or {}
        assigned_pad_ids = {
            str(value)
            for value in (
                lifecycle_pad,
                flow.get("origin_pad_id"),
                flow.get("target_pad_id"),
            )
            if isinstance(value, str) and value
        }
        if assigned_pad_ids & pad_aliases:
            requesters.append(str(uav["entity_id"]))
    return sorted(set(requesters))


def _validate_charging_service_plan(
    plan: Mapping[str, Any],
    episode_id: str,
) -> None:
    required_plan = {
        "schema_name",
        "schema_version",
        "model_id",
        "model_version",
        "facilities",
    }
    if set(plan) != required_plan:
        raise DomainStateSimulationError(
            f"{episode_id}: charging service plan must have exactly {sorted(required_plan)}"
        )
    if (
        plan["schema_name"] != "charging_service_plan"
        or plan["schema_version"] != "1.0.0"
        or not isinstance(plan["model_id"], str)
        or not plan["model_id"]
        or not isinstance(plan["model_version"], str)
        or not plan["model_version"]
        or not isinstance(plan["facilities"], list)
    ):
        raise DomainStateSimulationError(
            f"{episode_id}: charging service plan header is invalid"
        )
    seen_facilities: set[str] = set()
    for index, facility in enumerate(plan["facilities"]):
        required_facility = {
            "facility_id",
            "capacity",
            "availability_schedule",
            "requests",
        }
        if not isinstance(facility, Mapping) or set(facility) != required_facility:
            raise DomainStateSimulationError(
                f"{episode_id}: charging facility plan {index} has invalid fields"
            )
        facility_id = facility["facility_id"]
        capacity = facility["capacity"]
        if (
            not isinstance(facility_id, str)
            or not facility_id
            or facility_id in seen_facilities
            or not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity < 0
        ):
            raise DomainStateSimulationError(
                f"{episode_id}: charging facility plan {index} has invalid identity or capacity"
            )
        seen_facilities.add(facility_id)
        schedule = facility["availability_schedule"]
        if not isinstance(schedule, list) or not schedule:
            raise DomainStateSimulationError(
                f"{episode_id}: charger {facility_id} availability schedule is empty"
            )
        for interval in schedule:
            if (
                not isinstance(interval, Mapping)
                or set(interval) != {"start_tick", "end_tick", "status"}
                or not isinstance(interval["start_tick"], int)
                or isinstance(interval["start_tick"], bool)
                or not isinstance(interval["end_tick"], int)
                or isinstance(interval["end_tick"], bool)
                or interval["start_tick"] > interval["end_tick"]
                or interval["status"] not in {"available", "unavailable"}
            ):
                raise DomainStateSimulationError(
                    f"{episode_id}: charger {facility_id} availability interval is invalid"
                )
        requests = facility["requests"]
        if not isinstance(requests, list):
            raise DomainStateSimulationError(
                f"{episode_id}: charger {facility_id} requests must be an array"
            )
        seen_requests: set[tuple[str, int, int]] = set()
        for request in requests:
            if (
                not isinstance(request, Mapping)
                or set(request) != {"uav_id", "arrival_tick", "departure_tick"}
                or not isinstance(request["uav_id"], str)
                or not request["uav_id"]
                or not isinstance(request["arrival_tick"], int)
                or isinstance(request["arrival_tick"], bool)
                or not isinstance(request["departure_tick"], int)
                or isinstance(request["departure_tick"], bool)
                or request["arrival_tick"] > request["departure_tick"]
            ):
                raise DomainStateSimulationError(
                    f"{episode_id}: charger {facility_id} request is invalid"
                )
            signature = (
                request["uav_id"],
                request["arrival_tick"],
                request["departure_tick"],
            )
            if signature in seen_requests:
                raise DomainStateSimulationError(
                    f"{episode_id}: charger {facility_id} duplicates a request"
                )
            seen_requests.add(signature)


def _charging_service_state(
    *,
    inputs: DomainEpisodeInputs,
    facility: Mapping[str, Any],
    tick: int,
) -> dict[str, Any]:
    plan = inputs.charging_service_plan
    facility_id = str(facility["entity_id"])
    matches = [row for row in plan["facilities"] if row["facility_id"] == facility_id]
    if len(matches) != 1:
        raise DomainStateSimulationError(
            f"{inputs.episode_id}: charger {facility_id} has {len(matches)} service plans"
        )
    facility_plan = matches[0]
    availability_matches: list[str] = []
    schedule = facility_plan["availability_schedule"]
    for interval in schedule:
        if interval["start_tick"] <= tick <= interval["end_tick"]:
            availability_matches.append(interval["status"])
    if len(availability_matches) != 1:
        raise DomainStateSimulationError(
            f"charger {facility_id} has {len(availability_matches)} availability states at tick {tick}"
        )
    requests = facility_plan["requests"]
    active_requests: list[tuple[int, str]] = []
    for request in requests:
        uav_id = request["uav_id"]
        arrival_tick = request["arrival_tick"]
        departure_tick = request["departure_tick"]
        if arrival_tick <= tick <= departure_tick:
            roster_entity = inputs.roster_entities.get(uav_id)
            if roster_entity is None or _entity_category(roster_entity) != "uav":
                raise DomainStateSimulationError(
                    f"charger {facility_id} has active request from unregistered UAV "
                    f"{uav_id} at tick {tick}"
                )
            active_requests.append((arrival_tick, uav_id))
    active_requests.sort()
    requesters = [uav_id for _arrival, uav_id in active_requests]
    availability = availability_matches[0]
    capacity = facility_plan["capacity"]
    service_aircraft_id = (
        requesters[0]
        if availability == "available" and capacity > 0 and requesters
        else "none"
    )
    return {
        "availability": availability,
        "requester_ids": requesters,
        "service_aircraft_id": service_aircraft_id,
        "capacity": capacity,
        "service_plan_id": stable_identifier(
            "charging_service_plan", inputs.episode_id, facility_id
        ),
    }


def _charging_truth_entities(
    frame: Mapping[str, Any],
    roster_entities: Mapping[str, Mapping[str, Any]],
    episode_id: str,
) -> dict[str, dict[str, Any]]:
    """Extract only the operational evidence needed for charging before generic-state removal."""

    entities = frame.get("entities")
    if not isinstance(entities, list):
        raise DomainStateSimulationError(f"{episode_id}: charging truth frame lacks entities")
    result: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise DomainStateSimulationError(f"{episode_id}: malformed charging truth entity")
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str):
            continue
        roster_entity = roster_entities.get(entity_id)
        if roster_entity is None:
            continue
        roster_category = _entity_category(roster_entity)
        if roster_category == "uav":
            pass
        elif roster_category == "facility":
            scope = _nested_mapping(roster_entity, ("semantic_scope",))
            if scope is None or scope.get("scope_subtype") != "charging_station":
                continue
        else:
            continue
        if _entity_category(entity) != roster_category:
            raise DomainStateSimulationError(
                f"{episode_id}: charging truth category conflicts with roster for {entity_id}"
            )
        if entity_id in result:
            raise DomainStateSimulationError(
                f"{episode_id}: charging truth duplicates {entity_id}"
            )
        raw_state = entity.get("state")
        if raw_state is not None and not isinstance(raw_state, str):
            raise DomainStateSimulationError(
                f"{episode_id}: charging truth state is malformed for {entity_id}"
            )
        facility_state = _nested_mapping(entity, ("facility_state",)) or {}
        result[entity_id] = {
            "category": roster_category,
            "state": raw_state.strip().lower() if raw_state else UNKNOWN,
            "position": _position(entity),
            "fault": _state_bool(facility_state, ("fault",)),
        }
    return result


def _charging_activity_by_tick(
    inputs: DomainEpisodeInputs,
    profile: Mapping[str, Any],
) -> tuple[dict[int, dict[str, bool | str]], dict[int, dict[str, str]]]:
    """Resolve plan, contact, and formal operational states at each source tick."""

    tick_policy = profile["authoritative_tick_policy"]
    start = int(tick_policy["start"])
    step = int(tick_policy["step"])
    radius = float(
        profile["domain_models"]["pad_facility"]["charging_contact_radius_m"]
    )
    statuses: dict[int, dict[str, bool | str]] = {}
    issues: dict[int, dict[str, str]] = {}
    facility_ids = [
        str(row["facility_id"])
        for row in inputs.charging_service_plan["facilities"]
    ]
    for tick, truth_entities in sorted(inputs.charging_truth_by_tick.items()):
        # Plans are defined on the sampling grid; formal operational states are per tick.
        plan_tick = start + ((tick - start) // step) * step
        active: dict[str, bool | str] = {}
        tick_issues: dict[str, str] = {}
        candidates: dict[str, list[tuple[str, bool | str, str | None]]] = defaultdict(list)
        for facility_id in facility_ids:
            plan_state = _charging_service_state(
                inputs=inputs,
                facility={"entity_id": facility_id},
                tick=plan_tick,
            )
            uav_id = str(plan_state["service_aircraft_id"])
            if uav_id == "none":
                continue
            facility_truth = truth_entities.get(facility_id)
            uav_truth = truth_entities.get(uav_id)
            facility_position = (
                facility_truth["position"] if facility_truth is not None else None
            )
            uav_position = uav_truth["position"] if uav_truth is not None else None
            distance = (
                math.dist(facility_position, uav_position)
                if facility_position is not None and uav_position is not None
                else None
            )
            uav_state = uav_truth["state"] if uav_truth is not None else UNKNOWN
            facility_state = (
                facility_truth["state"] if facility_truth is not None else UNKNOWN
            )
            issue = None
            # Requests express candidate service, not simultaneous physical
            # occupancy. First resolve contact and the station's actual fault.
            if distance is not None and distance > radius:
                if uav_state == "charging" and facility_state == "serving":
                    status = UNKNOWN
                    issue = "charging_state_contact_conflict"
                else:
                    status = False
            elif facility_truth is not None and facility_truth["fault"] is True:
                if uav_state == "charging":
                    status = UNKNOWN
                    issue = "charging_fault_state_conflict"
                else:
                    status = False
            elif uav_state == "charging" and facility_state == "serving":
                if distance is None:
                    status = UNKNOWN
                    issue = "charging_contact_position_missing"
                else:
                    status = True
            elif (
                uav_state in CHARGING_NONACTIVE_UAV_STATES
                and facility_state in CHARGING_NONACTIVE_FACILITY_STATES
            ):
                status = False
            else:
                status = UNKNOWN
                if uav_state == UNKNOWN or facility_state == UNKNOWN:
                    issue = "charging_state_missing"
                elif uav_state == "charging" or facility_state == "serving":
                    issue = "charging_state_pair_conflict"
                else:
                    issue = "charging_state_unclassified"
            candidates[uav_id].append((facility_id, status, issue))
        for uav_id, claims in candidates.items():
            physically_active = [facility_id for facility_id, status, _ in claims if status is True]
            if len(physically_active) > 1:
                raise DomainStateSimulationError(
                    f"{inputs.episode_id}: UAV {uav_id} physically charging at multiple facilities "
                    f"at tick {tick}: {physically_active}"
                )
            active[uav_id] = (True if physically_active else
                              UNKNOWN if any(status == UNKNOWN for _, status, _ in claims) else False)
            unresolved = [f"{facility_id}:{issue}" for facility_id, _, issue in claims if issue is not None]
            if unresolved:
                tick_issues[uav_id] = ";".join(unresolved)
        for entity_id, truth in truth_entities.items():
            if (
                truth["category"] == "uav"
                and truth["state"] == "charging"
                and entity_id not in active
            ):
                active[entity_id] = UNKNOWN
                tick_issues[entity_id] = "charging_plan_binding_missing"
        statuses[tick] = active
        issues[tick] = tick_issues
    return statuses, issues


def _uav_pad_occupiers(
    pad: Mapping[str, Any],
    uavs: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> list[str]:
    pad_pos = _position(pad)
    if pad_pos is None:
        return []
    occupiers = []
    pad_radius = float(params["pad_occupancy_radius_m"])
    altitude_tolerance = float(params["pad_occupancy_altitude_m"])
    for uav in uavs:
        position = _position(uav)
        if position is None:
            continue
        if (
            math.dist(position[:2], pad_pos[:2]) <= pad_radius
            and abs(position[2] - pad_pos[2]) <= altitude_tolerance
        ):
            occupiers.append(str(uav["entity_id"]))
    return sorted(set(occupiers))


def _nearest_entity(
    position: list[float] | None,
    entities: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, float | None]:
    if position is None:
        return (None, None)
    nearest: dict[str, Any] | None = None
    best_distance: float | None = None
    for entity in entities:
        other = _position(entity)
        if other is None:
            continue
        distance = math.dist(position[:3], other[:3])
        if best_distance is None or distance < best_distance:
            nearest = copy.deepcopy(dict(entity))
            best_distance = distance
    return nearest, best_distance


def _traffic_light_malfunction(state: Any) -> bool:
    if not isinstance(state, Mapping):
        return True
    signal = str(state.get("state") or "")
    return (
        signal == ""
        or set(signal.lower()) <= {"r"}
        or any(ch not in "rgyou" for ch in signal.lower())
    )


def _incident_has_class(active_incidents: Any, accepted_classes: set[str]) -> bool:
    if not isinstance(active_incidents, list):
        return False
    return any(
        isinstance(incident, Mapping)
        and incident.get("accident_class") in accepted_classes
        for incident in active_incidents
    )


def _episode_incident_anchor_segment_ids(inputs: DomainEpisodeInputs) -> set[str]:
    """Return SUMO road-segment anchors declared by road-closing incidents."""
    segment_ids: set[str] = set()
    for frame in inputs.frames:
        active_incidents = frame.get("sumo_active_incidents")
        if not isinstance(active_incidents, list):
            continue
        for item in active_incidents:
            if not isinstance(item, Mapping) or not _incident_closes_road(item):
                continue
            anchor = item.get("anchor")
            if not isinstance(anchor, Mapping):
                continue
            edge_id = anchor.get("sumo_edge_id")
            if isinstance(edge_id, str) and edge_id:
                segment_ids.add(edge_id)
    return segment_ids


def _incident_closes_road(item: Mapping[str, Any]) -> bool:
    return item.get("accident_class") in {
        "lane_closure_roadwork",
        "hazmat_isolation_zone",
    }


def _barrier_props(props: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for prop in props:
        if _is_barrier_entity(prop):
            result.append(copy.deepcopy(dict(prop)))
    return result


def _is_barrier_entity(entity: Mapping[str, Any]) -> bool:
    return entity.get("logical_asset_id") in {
        "prop.roadwork.barrier.v1",
        "prop.roadwork.construction_fence.v1",
        "prop.roadwork.traffic_cone.v1",
    }


def _barrier_active(entity: Mapping[str, Any]) -> bool | str:
    for family in ("facility_state", "control_state", "constraint_state"):
        state = _nested_mapping(entity, (family,)) or {}
        active = _truth_or(
            _state_bool(state, ("active",)),
            _truth_or(
                _state_bool(state, ("deployed",)),
                _state_bool(state, ("staged",)),
            ),
        )
        if active != UNKNOWN:
            return active
    return UNKNOWN


def _barrier_segment_ids(scene_setup: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(scene_setup, Mapping):
        return {}
    anchors = scene_setup.get("road_segment_anchors")
    if not isinstance(anchors, list):
        # Explicit topology from entity placement.edge_id (scene contract), not names.
        anchors = _scene_setup_road_segment_anchors(scene_setup)
    result: dict[str, str] = {}
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            continue
        entity_id = anchor.get("entity_id")
        segment_id = anchor.get("road_segment_id")
        if isinstance(entity_id, str) and isinstance(segment_id, str) and segment_id:
            result[entity_id] = segment_id
    return result


def _affected_vehicle_ids(
    active_incidents: Any, vehicles: Sequence[Mapping[str, Any]]
) -> set[str]:
    if not isinstance(active_incidents, list):
        return set()
    declared_ids = {
        item
        for incident in active_incidents
        if isinstance(incident, Mapping)
        and isinstance(incident.get("affected_vehicle_ids"), list)
        for item in incident["affected_vehicle_ids"]
        if isinstance(item, str) and item
    }
    return {
        str(vehicle["entity_id"])
        for vehicle in vehicles
        if str(vehicle["entity_id"]) in declared_ids
        or _nested_value(vehicle, ("sumo_vehicle", "vehicle_id")) in declared_ids
    }


def _route_deviation_vehicle_count(
    vehicles: Sequence[Mapping[str, Any]], active_incidents: Any
) -> int:
    if not isinstance(active_incidents, list):
        return 0
    return sum(
        1
        for vehicle in vehicles
        if _nested_value(
            vehicle,
            ("sumo_vehicle", "semantic_metadata", "route_deviation_active"),
        )
        is True
    )


def _vehicle_accel(vehicle: Mapping[str, Any]) -> float | None:
    return _number(_nested_value(vehicle, ("sumo_vehicle", "accel_mps2")))


def _is_ambulance(vehicle: Mapping[str, Any]) -> bool:
    logical_asset_id = vehicle.get("logical_asset_id")
    canonical_logical_asset_id = _nested_value(
        vehicle, ("sumo_vehicle", "canonical_logical_asset_id")
    )
    vehicle_type = _nested_value(vehicle, ("sumo_vehicle", "vehicle_type"))
    return (
        logical_asset_id == "vehicle.emergency.ambulance.v1"
        or canonical_logical_asset_id == "vehicle.emergency.ambulance.v1"
        or vehicle_type == "aero_emergency"
    )


def _auth_state_from_score(score: float | None) -> str:
    if score is None:
        return UNKNOWN
    if score >= 0.8:
        return "authenticated"
    if score >= 0.5:
        return "locked_or_recovering"
    return "compromised"


def _delta_vector(
    a: Sequence[float] | None, b: Sequence[float] | None
) -> list[float] | None:
    if a is None or b is None:
        return None
    return [float(a[index]) - float(b[index]) for index in range(3)]


def _xy_norm(value: Sequence[float] | None) -> float | None:
    if value is None:
        return None
    return math.sqrt(float(value[0]) ** 2 + float(value[1]) ** 2)


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [_number(value) for value in values]
    clean = [value for value in numbers if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _position(entity: Mapping[str, Any]) -> list[float] | None:
    pose = entity.get("truth_pose")
    position = pose.get("position_enu_m") if isinstance(pose, Mapping) else None
    if (
        isinstance(position, list)
        and len(position) >= 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in position[:3]
        )
    ):
        return [float(position[0]), float(position[1]), float(position[2])]
    return None


def _velocity(entity: Mapping[str, Any]) -> list[float] | None:
    pose = entity.get("truth_pose")
    velocity = pose.get("velocity_enu_mps") if isinstance(pose, Mapping) else None
    if (
        isinstance(velocity, list)
        and len(velocity) >= 3
        and all(isinstance(value, (int, float)) for value in velocity[:3])
    ):
        return [float(velocity[0]), float(velocity[1]), float(velocity[2])]
    return None


def _speed(entity: Mapping[str, Any]) -> float | None:
    number = _number(_nested_value(entity, ("sumo_vehicle", "speed_mps")))
    if number is not None:
        return number
    velocity = _velocity(entity)
    if velocity is not None:
        return math.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
    return None


def _entity_category(entity: Mapping[str, Any]) -> str:
    return str(entity.get("entity_category") or UNKNOWN)


def _nested_value(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _nested_mapping(
    value: Mapping[str, Any], path: Sequence[str]
) -> Mapping[str, Any] | None:
    current = _nested_value(value, path)
    return current if isinstance(current, Mapping) else None


def _state_bool(state: Mapping[str, Any], keys: Sequence[str]) -> bool | str:
    for key in keys:
        value = state.get(key)
        if isinstance(value, bool):
            return value
    return UNKNOWN


def _truth_or(*values: Any) -> bool | str:
    """OR independent optional evidence without letting a missing source poison false."""

    saw_false = False
    for value in values:
        if value is True:
            return True
        if value is False:
            saw_false = True
    return False if saw_false else UNKNOWN


def _any_exact_state_bool(
    entities: Sequence[Mapping[str, Any]],
    family: str,
    keys: Sequence[str],
) -> bool | str:
    saw_known = False
    for entity in entities:
        state = _nested_mapping(entity, (family,))
        if state is None:
            continue
        value = _state_bool(state, keys)
        if value is True:
            return True
        if value is False:
            saw_known = True
    return False if saw_known else UNKNOWN


def _exact_enum(
    state: Mapping[str, Any],
    keys: Sequence[str],
    accepted: set[str],
) -> str:
    for key in keys:
        value = state.get(key)
        if not isinstance(value, str):
            continue
        lowered = value.strip().lower()
        if lowered in accepted:
            return lowered
    return UNKNOWN


def _sumo_projection(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = (
        "sumo_edge_id",
        "sumo_lane_id",
        "vehicle_id",
        "vehicle_type",
        "canonical_logical_asset_id",
        "lane_position_m",
        "speed_mps",
        "accel_mps2",
        "signals",
        "right_of_way_id",
        "right_of_way_ontology_class_id",
        "following_distance_m",
    )
    result = {key: copy.deepcopy(value.get(key)) for key in allowed if key in value}
    semantic_metadata = value.get("semantic_metadata")
    if (
        isinstance(semantic_metadata, Mapping)
        and "route_deviation_active" in semantic_metadata
    ):
        result["semantic_metadata"] = {
            "route_deviation_active": semantic_metadata["route_deviation_active"]
        }
    return result


def _sumo_incident_projection(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise DomainStateSimulationError("sumo_active_incidents must be an array")
    rows: list[dict[str, Any]] = []
    seen_incident_ids: set[str] = set()
    for index, incident in enumerate(value):
        if not isinstance(incident, Mapping):
            raise DomainStateSimulationError(
                f"sumo_active_incidents[{index}] must be an object"
            )
        incident_id = incident.get("incident_id")
        accident_class = incident.get("accident_class")
        affected = incident.get("affected_vehicle_ids")
        if (
            not isinstance(incident_id, str)
            or not incident_id
            or incident_id in seen_incident_ids
        ):
            raise DomainStateSimulationError(
                f"sumo_active_incidents[{index}] has invalid or duplicate incident_id"
            )
        if not isinstance(accident_class, str) or not accident_class:
            raise DomainStateSimulationError(
                f"sumo_active_incidents[{index}] lacks accident_class"
            )
        if (
            not isinstance(affected, list)
            or any(not isinstance(item, str) or not item for item in affected)
            or len(affected) != len(set(affected))
        ):
            raise DomainStateSimulationError(
                f"sumo_active_incidents[{index}] has invalid affected_vehicle_ids"
            )
        seen_incident_ids.add(incident_id)
        projected: dict[str, Any] = {
            "incident_id": incident_id,
            "accident_class": accident_class,
            "affected_vehicle_ids": sorted(affected),
        }
        if accident_class == "traffic_light_all_red_fault":
            target = incident.get("traffic_light_id")
            anchor = incident.get("anchor")
            source = anchor.get("geometry_source") if isinstance(anchor, Mapping) else None
            anchored_target = (source.removeprefix("sumo_traffic_light:")
                               if isinstance(source, str) and source.startswith("sumo_traffic_light:") else None)
            if target is not None and anchored_target is not None and target != anchored_target:
                raise DomainStateSimulationError(
                    f"sumo_active_incidents[{index}] has conflicting controller references"
                )
            if target is None:
                target = anchored_target
            if not isinstance(target, str) or not target:
                raise DomainStateSimulationError(
                    f"sumo_active_incidents[{index}] lacks all-red controller target"
                )
            projected["traffic_light_id"] = target
        anchor = incident.get("anchor")
        if isinstance(anchor, Mapping):
            anchor_edge_id = anchor.get("sumo_edge_id")
            anchor_lane_id = anchor.get("sumo_lane_id")
            if isinstance(anchor_edge_id, str) and anchor_edge_id:
                projected["anchor"] = {
                    "sumo_edge_id": anchor_edge_id,
                    "sumo_lane_id": (
                        anchor_lane_id
                        if isinstance(anchor_lane_id, str) and anchor_lane_id
                        else None
                    ),
                }
        rows.append(projected)
    rows.sort(
        key=lambda row: (
            str(row["incident_id"] or ""),
            str(row["accident_class"] or ""),
        )
    )
    return rows


def _uav_projection(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = (
        "mission_type",
        "corridor_id",
        "altitude_layer_m",
        "origin_pad_id",
        "target_pad_id",
        "target_cell_id",
    )
    return {key: copy.deepcopy(value.get(key)) for key in allowed if key in value}


def _lifecycle_projection(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = ("home_pad_entity_id",)
    return {key: copy.deepcopy(value.get(key)) for key in allowed if key in value}


def _centroid(positions: Sequence[list[float]]) -> list[float] | None:
    if not positions:
        return None
    return [
        sum(position[index] for position in positions) / len(positions)
        for index in range(3)
    ]


def _contains_unknown(value: Any) -> bool:
    if value == UNKNOWN:
        return True
    if isinstance(value, Mapping):
        return any(_contains_unknown(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unknown(item) for item in value)
    return False


def _sanitize_objective_value(value: Any, *, path: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if _objective_strict_drops_key(lowered, path):
                continue
            result[key] = _sanitize_objective_value(item, path=(*path, lowered))
        return result
    if isinstance(value, list):
        return [_sanitize_objective_value(item, path=path) for item in value]
    return copy.deepcopy(value)


def _objective_strict_drops_key(key: str, path: tuple[str, ...]) -> bool:
    if (
        path
        and path[-1] == "entities"
        and key in OBJECTIVE_FORBIDDEN_ENTITY_KEYS | {"state"}
    ):
        return True
    if key in {
        "active_event_id",
        "active_event_ids",
        "active_event_label",
        "active_event_labels",
        "dynamic_label",
        "dynamic_labels",
        "expected_event",
        "scenario_plan",
        "semantic_role",
        "task_id",
        "activity_label",
        "activity_labels",
        "activity_state",
        "activity_type",
        "state_facets",
    }:
        return True
    if (
        path
        and path[-1] == "annotations"
        and key in OBJECTIVE_FORBIDDEN_ANNOTATION_KEYS
    ):
        return True
    if key in OBJECTIVE_FORBIDDEN_ENTITY_KEYS and "annotations" in path:
        return True
    return False


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DomainStateSimulationError(f"{path} must contain a JSON object")
    return value


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _string_or_unknown(value: Any, default: str = UNKNOWN) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _explicit_identifier_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.casefold() in {"unknown", "none", "null", "n/a"}:
        return None
    return normalized


def _bool_or_unknown(value: bool | None) -> bool | str:
    if value is None:
        return UNKNOWN
    return bool(value)


def _round(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    if value is None:
        return UNKNOWN
    return value


def _round_list(values: Sequence[float] | None) -> list[float] | str:
    if values is None:
        return UNKNOWN
    return [_round(value) for value in values]


def _unit_interval(*parts: Any) -> float:
    digest = digest_object(parts).split(":", 1)[1]
    return int(digest[:16], 16) / float(16**16 - 1)
