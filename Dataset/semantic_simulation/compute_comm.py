"""Profile-driven compute and communication supplement simulation.

The supplement reads only objective episode inputs: manifest, roster, truth
frames, and weather. Authored event labels and event traces are intentionally
outside this module's input contract.
"""

from __future__ import annotations

import copy
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Dataset.semantic_truth.provenance import (
    canonical_json,
    digest_file,
    digest_object,
    read_jsonl,
    stable_identifier,
)
from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_SCHEMA_VERSION = "2.0.0"
NUMERIC_TOLERANCE = 1e-5
SOURCE_CLASSES = {
    "observed_simulator_truth",
    "rule_inferred",
    "simulated_derived",
    "standard_referenced_parameter",
    "deterministic_simulation",
}
PARAMETER_GROUP_STATUSES = {
    "standard_referenced_parameter",
    "deterministic_simulation",
    "simulated_derived",
}
TRUTH_VALUES = {"true", "false", "unknown", "out_of_scope"}
ONTOLOGY_PREDICATE_ROLE_CLASSES = {
    str(template["id"]): {
        str(role["key"]): (
            str(role["class"])
            if ":" in str(role["class"])
            else f"world:{role['class']}"
        )
        for role in template["argument_roles"]
    }
    for template in get_core_predicate_templates()
    if str(template["id"]).startswith(("compute.", "communication."))
}
ONTOLOGY_PREDICATE_ROLE_ORDER = {
    predicate_id: tuple(role_classes)
    for predicate_id, role_classes in ONTOLOGY_PREDICATE_ROLE_CLASSES.items()
}
ONTOLOGY_PREDICATE_ROLES = {
    predicate_id: set(role_classes)
    for predicate_id, role_classes in ONTOLOGY_PREDICATE_ROLE_CLASSES.items()
}
ONTOLOGY_EVENT_ROLES = {
    "compute.compute_resource_saturation_event": {"node", "resource"},
    "compute.compute_task_execution_failure_event": {
        "execution",
        "failure_state",
        "node",
        "task",
    },
    "communication.message_retransmission_event": {
        "message",
        "original_attempt",
        "retransmission",
    },
    "communication.communication_handover_event": {
        "handover",
        "session",
        "source_station",
        "target_station",
    },
    "communication.c2_link_degradation_event": {"actor", "station"},
    "communication.c2_link_loss_event": {"actor", "station"},
}
COMPUTE_PREDICATE_IDS = tuple(
    sorted(
        identifier
        for identifier in ONTOLOGY_PREDICATE_ROLES
        if identifier.startswith("compute.")
    )
)
COMMUNICATION_PREDICATE_IDS = tuple(
    sorted(
        identifier
        for identifier in ONTOLOGY_PREDICATE_ROLES
        if identifier.startswith("communication.")
    )
)


class ComputeCommSimulationError(ValueError):
    """Raised when simulation inputs, profile, or outputs fail closed."""


@dataclass(frozen=True)
class EpisodeInputs:
    episode_id: str
    episode_root: Path
    manifest: dict[str, Any]
    tick_hz: float
    tick_hz_source_ref: str
    roster_entities: dict[str, dict[str, Any]]
    frames: list[dict[str, Any]]
    weather_by_tick: dict[int, dict[str, Any]]
    input_files: dict[str, Path]
    input_digests: dict[str, str]
    input_digest: str
    raw_input_digest: str


@dataclass(frozen=True)
class EpisodeArtifacts:
    episode_id: str
    output_dir: Path
    files: dict[str, str]
    simulation_manifest: dict[str, Any]
    summary: dict[str, Any]


def load_compute_comm_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise ComputeCommSimulationError(f"{path} must contain a JSON object")
    validate_profile(profile)
    return profile


def validate_profile(profile: Mapping[str, Any]) -> None:
    required = [
        "schema_name",
        "schema_version",
        "profile_id",
        "model",
        "default_input_root",
        "default_output_root",
        "inputs",
        "authoritative_tick_policy",
        "source_policy",
        "parameter_governance",
        "compute",
        "communication",
        "predicate_rules",
        "event_rules",
        "validation",
    ]
    missing = [key for key in required if key not in profile]
    if missing:
        raise ComputeCommSimulationError(f"profile lacks required keys: {missing}")
    if profile["schema_name"] != "compute_comm_supplement_profile":
        raise ComputeCommSimulationError(
            "profile schema_name is not compute_comm_supplement_profile"
        )
    if (
        profile["model"].get("seed_namespace")
        != "profile_whitelisted_model_input_digest_v3"
    ):
        raise ComputeCommSimulationError(
            "profile model.seed_namespace must be profile_whitelisted_model_input_digest_v3"
        )
    inputs = profile["inputs"]
    for key, expected in (
        ("episode_manifest", "episode_manifest.json"),
        ("entity_roster", "global_entity_roster.json"),
        ("truth_frames", "truth_frames.jsonl"),
        ("weather", "weather_meta.jsonl"),
    ):
        if inputs.get(key) != expected:
            raise ComputeCommSimulationError(f"profile inputs.{key} must be {expected}")
    forbidden = set(profile["source_policy"].get("forbidden_inputs") or [])
    for forbidden_name in (
        "event_trace.jsonl",
        "event_realization.jsonl",
        "dynamic_labels.jsonl",
        "scenario_plan.json",
        "source_event_script_path",
        "n_events",
    ):
        if forbidden_name not in forbidden:
            raise ComputeCommSimulationError(f"profile must forbid {forbidden_name}")
    if not profile["source_policy"].get("model_input_fields"):
        raise ComputeCommSimulationError(
            "profile source_policy.model_input_fields must be explicit"
        )
    if not profile["source_policy"].get("forbidden_manifest_fields"):
        raise ComputeCommSimulationError(
            "profile source_policy.forbidden_manifest_fields must be explicit"
        )
    governance = profile["parameter_governance"]
    if not isinstance(governance, Mapping):
        raise ComputeCommSimulationError(
            "profile parameter_governance must be an object"
        )
    if governance.get("review_status") != "reviewed_for_controlled_simulation":
        raise ComputeCommSimulationError(
            "profile parameters must be reviewed for controlled simulation"
        )
    measurement_provenance = governance.get("measurement_provenance")
    if measurement_provenance != {
        "hardware_logs_present": False,
        "network_logs_present": False,
    }:
        raise ComputeCommSimulationError(
            "profile parameter_governance.measurement_provenance must record log availability"
        )
    if governance.get("source_class") != "standard_referenced_parameter":
        raise ComputeCommSimulationError(
            "profile parameter_governance source_class must be standard_referenced_parameter"
        )
    references = governance.get("references")
    if not isinstance(references, list) or not references:
        raise ComputeCommSimulationError(
            "standard_referenced_parameter governance requires a non-empty references list"
        )
    for index, item in enumerate(references):
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("document_id"), str)
            or not isinstance(item.get("anchors"), list)
            or not item["anchors"]
        ):
            raise ComputeCommSimulationError(
                f"profile parameter_governance.references[{index}] must declare "
                "document_id and a non-empty anchors list"
            )
        for anchor in item["anchors"]:
            if (not isinstance(anchor, Mapping) or
                    str(anchor.get("parameter", "")).endswith("channel_bandwidth_mbps")):
                raise ComputeCommSimulationError(
                    "channel bandwidth is a profile-declared per-flow cap, not an external numeric anchor"
                )
    bandwidth_path = "communication.station_profiles.base_station.channel_bandwidth_mbps"
    bandwidth_model = governance.get("bandwidth_model")
    station_profiles = profile.get("communication", {}).get("station_profiles", {})
    station_profile = station_profiles.get("base_station", {}) if isinstance(station_profiles, Mapping) else {}
    bandwidth_value = station_profile.get("channel_bandwidth_mbps") if isinstance(station_profile, Mapping) else None
    if (not isinstance(bandwidth_model, Mapping) or
            bandwidth_model.get("parameter") != bandwidth_path or
            bandwidth_model.get("scope") != "per_flow_nominal_cap" or
            bandwidth_model.get("external_numeric_basis") != "unverified" or
            not isinstance(bandwidth_model.get("mapping"), str) or
            not bandwidth_model["mapping"] or
            type(bandwidth_value) not in (int, float) or
            not math.isfinite(bandwidth_value) or bandwidth_value < 0 or
            type(bandwidth_model.get("value")) not in (int, float) or
            not math.isfinite(bandwidth_model["value"]) or
            bandwidth_model["value"] != bandwidth_value):
        raise ComputeCommSimulationError(
            "profile bandwidth model must match its finite per-flow station input"
        )
    forbidden_claims = {str(item) for item in governance.get("forbidden_claims") or []}
    if {
        "measured_hardware_performance",
        "measured_network_performance",
        "real_world_generalization",
    } - forbidden_claims:
        raise ComputeCommSimulationError(
            "profile parameter_governance lacks required claim boundaries"
        )
    required_groups = {
        "compute_capacity",
        "compute_task_demand",
        "communication_channel",
        "station_operational_state",
        "weather_attenuation",
        "retransmission",
        "seed_variation",
        "event_labels",
    }
    groups = governance.get("parameter_groups")
    if not isinstance(groups, Mapping) or set(groups) != required_groups:
        raise ComputeCommSimulationError(
            "profile parameter_governance must cover every simulated parameter group"
        )
    group_statuses: set[str] = set()
    for group_name in required_groups:
        group = groups.get(group_name)
        if not isinstance(group, Mapping):
            raise ComputeCommSimulationError(
                f"profile parameter group must be an object: {group_name}"
            )
        group_statuses.add(str(group.get("status")))
    if not group_statuses.issubset(PARAMETER_GROUP_STATUSES):
        raise ComputeCommSimulationError(
            "profile parameter group status must be one of "
            f"{sorted(PARAMETER_GROUP_STATUSES)}"
        )
    if "standard_referenced_parameter" not in group_statuses:
        raise ComputeCommSimulationError(
            "standard_referenced_parameter governance requires at least one "
            "parameter group with status standard_referenced_parameter"
        )
    if set(profile["predicate_rules"]) != set(ONTOLOGY_PREDICATE_ROLES):
        raise ComputeCommSimulationError(
            "profile predicate_rules must exactly match the ontology-aligned compute/communication predicate set"
        )
    event_types = {
        str(item.get("event_type_id"))
        for item in profile["event_rules"].values()
        if isinstance(item, Mapping)
    }
    if not event_types or not event_types.issubset(ONTOLOGY_EVENT_ROLES):
        raise ComputeCommSimulationError(
            "profile event rules contain a non-ontology event type"
        )
    for predicate_id, event_rule in profile["event_rules"].items():
        if predicate_id not in ONTOLOGY_PREDICATE_ROLES:
            raise ComputeCommSimulationError(
                f"event rule uses an unknown source predicate: {predicate_id}"
            )
        if not isinstance(event_rule, Mapping):
            raise ComputeCommSimulationError(
                f"event rule must be an object: {predicate_id}"
            )
        event_type = str(event_rule.get("event_type_id"))
        participant_bindings = event_rule.get("participant_bindings")
        if not isinstance(participant_bindings, Mapping):
            raise ComputeCommSimulationError(
                f"event rule lacks participant_bindings: {predicate_id}"
            )
        expected_roles = ONTOLOGY_EVENT_ROLES[event_type]
        if set(participant_bindings) != expected_roles:
            raise ComputeCommSimulationError(
                f"event {event_type} participant roles must exactly match ontology roles: "
                f"expected={sorted(expected_roles)}, actual={sorted(participant_bindings)}"
            )
    for predicate_id, predicate_rule in profile["predicate_rules"].items():
        if not isinstance(predicate_rule, Mapping):
            raise ComputeCommSimulationError(
                f"predicate rule must be an object: {predicate_id}"
            )
        role_bindings = predicate_rule.get("role_bindings")
        if not isinstance(role_bindings, list) or not all(
            isinstance(item, str) for item in role_bindings
        ):
            raise ComputeCommSimulationError(
                f"predicate rule lacks exact role_bindings: {predicate_id}"
            )
        if set(role_bindings) != ONTOLOGY_PREDICATE_ROLES[predicate_id]:
            raise ComputeCommSimulationError(
                f"predicate {predicate_id} role_bindings must exactly match ontology roles: "
                f"expected={sorted(ONTOLOGY_PREDICATE_ROLES[predicate_id])}, actual={sorted(role_bindings)}"
            )
    for section_name in ("compute", "communication"):
        if not isinstance(profile.get(section_name), Mapping):
            raise ComputeCommSimulationError(
                f"profile {section_name} must be an object"
            )
    if (
        not isinstance(profile["predicate_rules"], Mapping)
        or not profile["predicate_rules"]
    ):
        raise ComputeCommSimulationError(
            "profile predicate_rules must be a non-empty object"
        )
    if not isinstance(profile["event_rules"], Mapping) or not profile["event_rules"]:
        raise ComputeCommSimulationError(
            "profile event_rules must be a non-empty object"
        )


def _resolve_tick_hz(
    manifest: Mapping[str, Any],
    tick_policy: Mapping[str, Any],
) -> tuple[float, str]:
    """Resolve the authoritative episode clock without inventing a default.

    A profile- or manifest-level ``tick_hz`` is preferred.  Formal render-ready
    manifests currently expose the same clock through their tick/simulation-time
    endpoints, so that mapping is also an authoritative manifest source.
    """

    for value, source_ref in (
        (tick_policy.get("tick_hz"), "profile:authoritative_tick_policy.tick_hz"),
        (manifest.get("tick_hz"), "episode_manifest.json#tick_hz"),
    ):
        tick_hz = _number(value)
        if tick_hz is not None:
            if tick_hz <= 0:
                raise ComputeCommSimulationError(
                    "authoritative tick_hz must be positive"
                )
            return float(tick_hz), source_ref

    time_range = manifest.get("time_range")
    if isinstance(time_range, Mapping):
        tick_start = _number(time_range.get("tick_start"))
        tick_end = _number(time_range.get("tick_end"))
        time_start = _number(time_range.get("sim_time_start"))
        time_end = _number(time_range.get("sim_time_end"))
        if all(
            value is not None for value in (tick_start, tick_end, time_start, time_end)
        ):
            elapsed_ticks = float(tick_end) - float(tick_start)
            elapsed_seconds = float(time_end) - float(time_start)
            if elapsed_ticks > 0 and elapsed_seconds > 0:
                return (
                    elapsed_ticks / elapsed_seconds,
                    "episode_manifest.json#time_range",
                )

    raise ComputeCommSimulationError(
        "authoritative tick_hz is missing; provide profile/manifest tick_hz or "
        "manifest time_range tick/simulation-time endpoints"
    )


def _model_input_projection(
    *,
    tick_hz: float,
    roster_entities: Mapping[str, Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    weather_by_tick: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the exact, label-free data projection allowed to affect simulation.

    The raw files remain part of provenance, but authored event metadata, scenario
    plans, semantic roles, task ids, and manifest labels are deliberately absent.
    """

    roster_projection: list[dict[str, Any]] = []
    for entity_id, entity in sorted(roster_entities.items()):
        roster_projection.append(
            {
                "entity_id": entity_id,
                "entity_category": _entity_category(entity),
                "entity_kind": _string_or_unknown(entity.get("entity_kind")),
                "position_enu_m": _position(entity),
            }
        )

    frame_projection: list[dict[str, Any]] = []
    for frame in sorted(frames, key=lambda item: int(item["tick"])):
        entities: list[dict[str, Any]] = []
        for entity in sorted(
            _current_entities(frame), key=lambda item: str(item["entity_id"])
        ):
            category = _entity_category(entity)
            is_station = _is_station_entity(entity)
            if category not in {"uav", "vehicle"} and not is_station:
                continue
            pose = entity.get("truth_pose")
            velocity = (
                pose.get("velocity_enu_mps") if isinstance(pose, Mapping) else None
            )
            annotations = entity.get("annotations")
            sumo_vehicle = entity.get("sumo_vehicle")
            projected_entity = {
                "entity_id": str(entity["entity_id"]),
                "entity_category": category,
                "entity_kind": _string_or_unknown(entity.get("entity_kind")),
                "position_enu_m": _position(entity),
                "communication_state": _structured_communication_input(entity),
                "security_state": _structured_security_input(entity),
            }
            if not is_station:
                projected_entity.update(
                    {
                        "velocity_enu_mps": copy.deepcopy(velocity),
                        "annotation_speed_mps": (
                            annotations.get("speed_mps")
                            if isinstance(annotations, Mapping)
                            else None
                        ),
                        "sumo_speed_mps": (
                            sumo_vehicle.get("speed_mps")
                            if isinstance(sumo_vehicle, Mapping)
                            else None
                        ),
                    }
                )
            entities.append(projected_entity)
        frame_projection.append({"tick": int(frame["tick"]), "entities": entities})

    weather_projection = [
        {
            "tick": tick,
            "condition": row.get("condition"),
            "rain": row.get("rain"),
            "fog_density": row.get("fog_density"),
            "dust": row.get("dust"),
            "wind_speed": row.get("wind_speed"),
        }
        for tick, row in sorted(weather_by_tick.items())
    ]
    return {
        "clock": {"tick_hz": _round(tick_hz)},
        "roster_entities": roster_projection,
        "truth_frames": frame_projection,
        "weather": weather_projection,
    }


def load_episode_inputs(
    episode_root: Path, profile: Mapping[str, Any]
) -> EpisodeInputs:
    inputs = profile["inputs"]
    paths = {
        "episode_manifest": episode_root / inputs["episode_manifest"],
        "entity_roster": episode_root / inputs["entity_roster"],
        "truth_frames": episode_root / inputs["truth_frames"],
        "weather": episode_root / inputs["weather"],
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ComputeCommSimulationError(
                f"required {name} input is missing: {path}"
            )

    manifest = _load_json(paths["episode_manifest"])
    roster = _load_json(paths["entity_roster"])
    episode_id = _string_or_unknown(manifest.get("episode_id"))
    if episode_id == "unknown":
        episode_id = episode_root.name
    if episode_id != episode_root.name:
        raise ComputeCommSimulationError(
            f"episode manifest id {episode_id} does not match directory {episode_root.name}"
        )
    tick_hz, tick_hz_source_ref = _resolve_tick_hz(
        manifest,
        profile["authoritative_tick_policy"],
    )
    roster_entities = _index_entities(
        roster.get("entities"), str(paths["entity_roster"])
    )
    tick_policy = profile["authoritative_tick_policy"]
    wanted_ticks = set(
        range(
            int(tick_policy["start"]),
            int(tick_policy["end"]) + 1,
            int(tick_policy["step"]),
        )
    )
    frames: list[dict[str, Any]] = []
    seen_ticks: set[int] = set()
    for row in read_jsonl(paths["truth_frames"]):
        tick = row.get("tick")
        if not isinstance(tick, int):
            raise ComputeCommSimulationError(
                f"{paths['truth_frames']}: truth frame lacks integer tick"
            )
        if tick not in wanted_ticks:
            continue
        if tick in seen_ticks:
            raise ComputeCommSimulationError(
                f"{paths['truth_frames']}: duplicate sampled tick {tick}"
            )
        seen_ticks.add(tick)
        frames.append(_compact_compute_comm_frame(row))
    frames.sort(key=lambda row: int(row["tick"]))
    if not frames:
        raise ComputeCommSimulationError(
            f"{paths['truth_frames']}: no sampled authoritative ticks found"
        )
    missing_ticks = sorted(wanted_ticks - seen_ticks)
    if missing_ticks:
        raise ComputeCommSimulationError(
            f"{paths['truth_frames']}: missing authoritative ticks {missing_ticks}"
        )
    for frame in frames:
        frame_ids: set[str] = set()
        for entity in frame["entities"]:
            entity_id = str(entity["entity_id"])
            if entity_id in frame_ids:
                raise ComputeCommSimulationError(
                    f"{paths['truth_frames']}: duplicate entity {entity_id} at tick {frame['tick']}"
                )
            frame_ids.add(entity_id)
            category = _entity_category(entity)
            roster_entity = roster_entities.get(entity_id)
            roster_category = (
                _entity_category(roster_entity) if roster_entity is not None else None
            )
            if (
                category in {"uav", "vehicle"}
                or roster_category in {"uav", "vehicle"}
            ) and roster_category != category:
                raise ComputeCommSimulationError(
                    f"{paths['truth_frames']}: compute entity {entity_id} lacks matching roster category at tick {frame['tick']}"
                )

    weather_by_tick: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(paths["weather"]):
        tick = row.get("tick")
        if isinstance(tick, int) and tick in wanted_ticks:
            weather_by_tick[tick] = copy.deepcopy(row)

    digests = {name: digest_file(path) for name, path in sorted(paths.items())}
    model_input = _model_input_projection(
        tick_hz=tick_hz,
        roster_entities=roster_entities,
        frames=frames,
        weather_by_tick=weather_by_tick,
    )
    return EpisodeInputs(
        episode_id=episode_id,
        episode_root=episode_root,
        manifest=manifest,
        tick_hz=tick_hz,
        tick_hz_source_ref=tick_hz_source_ref,
        roster_entities=roster_entities,
        frames=frames,
        weather_by_tick=weather_by_tick,
        input_files=paths,
        input_digests=digests,
        input_digest=digest_object(model_input),
        raw_input_digest=digest_object(digests),
    )


def _compact_compute_comm_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only physical/runtime fields consumed by the mechanism model."""

    result: dict[str, Any] = {"tick": int(frame["tick"]), "entities": []}
    for entity in _current_entities(frame):
        compact: dict[str, Any] = {
            "entity_id": str(entity["entity_id"]),
            "entity_category": _entity_category(entity),
            "entity_kind": _string_or_unknown(entity.get("entity_kind")),
        }
        pose = entity.get("truth_pose")
        if isinstance(pose, Mapping):
            compact["truth_pose"] = {
                key: copy.deepcopy(pose[key])
                for key in ("position_enu_m", "velocity_enu_mps")
                if key in pose
            }
        sumo = entity.get("sumo_vehicle")
        if isinstance(sumo, Mapping) and "speed_mps" in sumo:
            compact["sumo_vehicle"] = {"speed_mps": sumo["speed_mps"]}
        annotations = entity.get("annotations")
        if isinstance(annotations, Mapping) and "speed_mps" in annotations:
            compact["annotations"] = {"speed_mps": annotations["speed_mps"]}
        for family in ("communication_state", "security_state"):
            value = entity.get(family)
            if isinstance(value, Mapping):
                compact[family] = copy.deepcopy(dict(value))
        result["entities"].append(compact)
    result["entities"].sort(key=lambda entity: str(entity["entity_id"]))
    return result


def build_episode_artifacts(
    episode_root: Path,
    output_dir: Path,
    profile_path: Path,
) -> EpisodeArtifacts:
    profile = load_compute_comm_profile(profile_path)
    inputs = load_episode_inputs(episode_root, profile)
    common, profile_digest = _simulation_common(inputs, profile, profile_path)

    compute_rows = _build_compute_rows(inputs, profile, common)
    communication_rows = _build_communication_rows(inputs, profile, common)
    coverage_rows = _build_process_coverage_rows(
        inputs,
        compute_rows,
        communication_rows,
        common,
    )
    predicate_rows = _build_predicate_rows(
        compute_rows,
        communication_rows,
        profile,
        common,
    )
    predicate_matrix_rows = _build_predicate_matrix_rows(
        inputs,
        compute_rows,
        communication_rows,
        common,
    )
    event_rows = _build_event_rows(predicate_rows, profile, common)
    provenance_rows = _build_provenance_rows(
        inputs, profile_path, profile, common, profile_digest
    )

    validate_output_rows(
        compute_rows=compute_rows,
        communication_rows=communication_rows,
        predicate_rows=predicate_rows,
        predicate_matrix_rows=predicate_matrix_rows,
        event_rows=event_rows,
        provenance_rows=provenance_rows,
        tick_step=int(profile["authoritative_tick_policy"]["step"]),
        coverage_rows=coverage_rows,
        expected_ticks=[int(frame["tick"]) for frame in inputs.frames],
    )

    summary = _build_summary(
        inputs,
        profile,
        compute_rows,
        communication_rows,
        coverage_rows,
        predicate_rows,
        event_rows,
        predicate_matrix_rows=predicate_matrix_rows,
    )
    files: dict[str, str] = {}
    files["compute_state.jsonl"] = _jsonl_text(_artifact_rows(compute_rows))
    del compute_rows
    files["communication_state.jsonl"] = _jsonl_text(_artifact_rows(communication_rows))
    del communication_rows
    files["process_coverage.jsonl"] = _jsonl_text(_artifact_rows(coverage_rows))
    del coverage_rows
    files["predicate_truth.jsonl"] = _jsonl_text(_artifact_rows(predicate_rows))
    del predicate_rows
    files["predicate_truth_matrix.jsonl"] = _jsonl_text(_artifact_rows(predicate_matrix_rows))
    del predicate_matrix_rows
    files["events.jsonl"] = _jsonl_text(_artifact_rows(event_rows))
    del event_rows
    files["provenance.jsonl"] = _jsonl_text(_artifact_rows(provenance_rows))
    del provenance_rows
    files["summary.json"] = _json_text(summary)
    simulation_manifest = _build_simulation_manifest(
        inputs,
        profile,
        common,
        output_dir,
        {name: text for name, text in files.items()},
    )
    files["simulation_manifest.json"] = _json_text(simulation_manifest)
    return EpisodeArtifacts(
        episode_id=inputs.episode_id,
        output_dir=output_dir,
        files=files,
        simulation_manifest=simulation_manifest,
        summary=summary,
    )


def build_communication_state_rows(
    episode_root: Path,
    profile_path: Path,
) -> list[dict[str, Any]]:
    """Build only UAV communication mechanism rows for objective semantics.

    The objective event pipeline does not need the much larger vehicle compute
    workload or the duplicate communication predicates/events emitted by this
    supplement.  This focused entry point keeps the final truth corpus compact
    while using the exact same governed channel model as the full supplement.
    """

    profile = load_compute_comm_profile(profile_path)
    inputs = load_episode_inputs(episode_root, profile)
    common, _ = _simulation_common(inputs, profile, profile_path)
    rows = _build_communication_rows(inputs, profile, common)
    for row in rows:
        _validate_provenance_fields(row)
        _validate_communication_capacity(row)
    return rows


def build_objective_world_inputs(
    episode_root: Path,
    profile_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build full compute/communication state and grounded predicate rows."""

    profile = load_compute_comm_profile(profile_path)
    inputs = load_episode_inputs(episode_root, profile)
    common, _ = _simulation_common(inputs, profile, profile_path)
    compute_rows = _build_compute_rows(inputs, profile, common)
    communication_rows = _build_communication_rows(inputs, profile, common)
    predicate_rows = _build_predicate_rows(
        compute_rows,
        communication_rows,
        profile,
        common,
    )
    grounded_predicate_rows = [
        row for row in predicate_rows if row.get("binding_status") == "complete"
    ]
    return compute_rows, communication_rows, grounded_predicate_rows


def _simulation_common(
    inputs: EpisodeInputs,
    profile: Mapping[str, Any],
    profile_path: Path,
) -> tuple[dict[str, Any], str]:
    """Return deterministic shared provenance for full and focused builds."""

    profile_digest = digest_file(profile_path)
    seed_digest = digest_object(
        {
            "profile_id": profile["profile_id"],
            "profile_digest": profile_digest,
            "input_digest": inputs.input_digest,
            "seed_namespace": profile["model"].get("seed_namespace"),
        }
    )
    parameter_digest = digest_object(
        {
            "compute": profile["compute"],
            "communication": profile["communication"],
            "parameter_governance": profile["parameter_governance"],
            "predicate_rules": profile["predicate_rules"],
            "event_rules": profile["event_rules"],
        }
    )
    common = {
        "episode_id": inputs.episode_id,
        "profile_id": profile["profile_id"],
        "profile_version": profile["schema_version"],
        "model_id": profile["model"]["model_id"],
        "model_version": profile["model"]["model_version"],
        "input_digest": inputs.input_digest,
        "parameter_digest": parameter_digest,
        "seed_digest": seed_digest,
    }
    return common, profile_digest


def write_episode_outputs(artifacts: EpisodeArtifacts) -> None:
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (
        "compute_state.jsonl",
        "communication_state.jsonl",
        "process_coverage.jsonl",
        "predicate_truth.jsonl",
        "predicate_truth_matrix.jsonl",
        "events.jsonl",
        "provenance.jsonl",
        "summary.json",
        "simulation_manifest.json",
    ):
        (artifacts.output_dir / file_name).write_text(
            artifacts.files[file_name],
            encoding="utf-8",
            newline="\n",
        )


def check_episode_outputs(artifacts: EpisodeArtifacts) -> list[str]:
    mismatches: list[str] = []
    for file_name, expected in sorted(artifacts.files.items()):
        path = artifacts.output_dir / file_name
        if not path.is_file():
            mismatches.append(f"missing: {path}")
            continue
        actual = path.read_text(encoding="utf-8-sig")
        if actual != expected:
            mismatches.append(f"content mismatch: {path}")
    return mismatches


def _build_compute_rows(
    inputs: EpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    compute_profile = profile["compute"]
    node_profiles = compute_profile.get("node_profiles") or {}
    task_profiles = compute_profile.get("task_profiles") or {}
    max_queue_depth_default = _number(compute_profile.get("max_queue_depth_default"))
    overcapacity_threshold = _number(
        compute_profile.get("overcapacity_utilization_threshold")
    )
    queue_started_tick_by_entity: dict[str, int] = {}
    for frame in inputs.frames:
        tick = int(frame["tick"])
        compute_entities = sorted(
            (
                entity
                for entity in _current_entities(frame)
                if _entity_category(entity) in {"uav", "vehicle"}
            ),
            key=lambda item: str(item.get("entity_id") or ""),
        )
        rank_by_entity: dict[str, int] = {}
        for category in ("uav", "vehicle"):
            category_entities = [
                entity
                for entity in compute_entities
                if _entity_category(entity) == category
            ]
            rank_by_entity.update(
                {
                    str(entity.get("entity_id")): rank
                    for rank, entity in enumerate(category_entities)
                }
            )
        current_entity_ids = {str(entity["entity_id"]) for entity in compute_entities}
        for entity_id in queue_started_tick_by_entity.keys() - current_entity_ids:
            del queue_started_tick_by_entity[entity_id]

        frame_records: list[dict[str, Any]] = []
        for entity in compute_entities:
            category = _entity_category(entity)
            entity_id = _string_or_unknown(entity.get("entity_id"))
            node_profile = node_profiles.get(category)
            task_profile = task_profiles.get(category)
            missing_parameters: list[str] = []
            if not isinstance(node_profile, Mapping):
                missing_parameters.append(f"compute.node_profiles.{category}")
                node_profile = {}
            if not isinstance(task_profile, Mapping):
                missing_parameters.append(f"compute.task_profiles.{category}")
                task_profile = {}
            declared_capacity = {
                "cpu_cores": _number(node_profile.get("cpu_cores")),
                "gpu_units": _number(node_profile.get("gpu_units")),
                "memory_mb": _number(node_profile.get("memory_mb")),
                "max_queue_depth": _number(
                    node_profile.get("max_queue_depth"), max_queue_depth_default
                ),
            }
            missing_parameters.extend(
                f"compute.node_profiles.{category}.{resource}"
                for resource in ("cpu_cores", "gpu_units", "memory_mb")
                if declared_capacity[resource] is None
            )
            node_available = _compute_node_available(
                compute_profile,
                tick=tick,
                category=category,
                entity_rank=rank_by_entity[entity_id],
            )
            capacity = dict(declared_capacity)
            if not node_available:
                for key in ("cpu_cores", "gpu_units", "memory_mb"):
                    if capacity[key] is not None:
                        capacity[key] = 0.0
            demand = _compute_demand(entity, frame, task_profile, common["seed_digest"])
            missing_parameters.extend(demand.pop("missing_parameters"))
            allocation, balance = _allocate_compute(demand, capacity)
            frame_records.append(
                {
                    "entity": entity,
                    "entity_id": entity_id,
                    "category": category,
                    "task_profile": task_profile,
                    "missing_parameters": missing_parameters,
                    "declared_capacity": declared_capacity,
                    "capacity": capacity,
                    "node_available": node_available,
                    "demand": demand,
                    "allocation": allocation,
                    "balance": balance,
                }
            )

        residual_cpu_by_node = {
            f"compute_node:{record['entity_id']}": max(
                0.0,
                (_number(record["capacity"].get("cpu_cores"), 0.0) or 0.0)
                - (_number(record["allocation"].get("cpu_cores"), 0.0) or 0.0),
            )
            for record in frame_records
        }
        inbound_cpu_by_node = {
            f"compute_node:{record['entity_id']}": 0.0 for record in frame_records
        }
        for record in frame_records:
            balance = record["balance"]
            queued_cpu = _number(balance.get("cpu_queued"), 0.0) or 0.0
            target_node_id = _select_compute_offload_target(
                source_entity_id=record["entity_id"],
                tick=tick,
                frame_records=frame_records,
                residual_cpu_by_node=residual_cpu_by_node,
                seed_digest=common["seed_digest"],
            )
            accepted_cpu = 0.0
            if queued_cpu > 0.0 and target_node_id is not None:
                accepted_cpu = min(queued_cpu, residual_cpu_by_node[target_node_id])
                residual_cpu_by_node[target_node_id] -= accepted_cpu
                inbound_cpu_by_node[target_node_id] += accepted_cpu
            remaining_cpu = max(0.0, queued_cpu - accepted_cpu)
            balance["cpu_offloaded"] = accepted_cpu
            balance["cpu_queued"] = remaining_cpu
            balance["queue_depth"] = _compute_queue_depth(balance)
            record["offload"] = _compute_offload(
                source_entity_id=record["entity_id"],
                requested_cpu_cores=queued_cpu,
                accepted_cpu_cores=accepted_cpu,
                target_node_id=target_node_id,
            )

        for record in frame_records:
            entity = record["entity"]
            entity_id = record["entity_id"]
            category = record["category"]
            task_profile = record["task_profile"]
            capacity = record["capacity"]
            declared_capacity = record["declared_capacity"]
            demand = record["demand"]
            allocation = record["allocation"]
            balance = record["balance"]
            missing_parameters = record["missing_parameters"]
            node_available = record["node_available"]
            missing_source_record = bool(missing_parameters)
            queue_depth = int(balance["queue_depth"] or 0)
            if missing_source_record:
                queue_started_tick_by_entity.pop(entity_id, None)
            elif queue_depth > 0:
                if entity_id not in queue_started_tick_by_entity:
                    queue_started_tick_by_entity[entity_id] = tick
            else:
                queue_started_tick_by_entity.pop(entity_id, None)
            deadline = _deadline(
                task_profile,
                tick=tick,
                queue_started_tick=queue_started_tick_by_entity.get(entity_id),
                missing_source_record=missing_source_record,
            )
            failed_by_queue = (
                capacity["max_queue_depth"] is not None
                and queue_depth > capacity["max_queue_depth"]
            )
            over_capacity = (
                overcapacity_threshold is not None
                and balance["cpu_utilization_pct"] is not None
                and balance["cpu_utilization_pct"] > overcapacity_threshold * 100.0
            )
            if missing_source_record:
                execution_state = "unknown"
                failure_mode = "missing_source_record"
            elif failed_by_queue:
                execution_state = "failed"
                failure_mode = "capacity_queue_overflow"
            elif not node_available:
                execution_state = (
                    "migrating" if record["offload"]["enabled"] else "failed"
                )
                failure_mode = "node_unavailable"
            elif queue_depth:
                execution_state = "queued"
                failure_mode = "capacity_pressure"
            else:
                execution_state = (
                    "running_offloaded" if record["offload"]["enabled"] else "running"
                )
                failure_mode = "capacity_pressure" if over_capacity else "none"
            migration_active = bool(
                not node_available
                and record["offload"]["enabled"]
                and record["offload"]["target_node_id"] != "none"
            )
            row = _common_row(
                common,
                source_class="simulated_derived",
                rule_id="compute_comm_supplement.compute_state",
                rule_version=profile["compute"]["rule_version"],
                source_refs=[
                    f"{inputs.input_files['truth_frames'].name}#tick={tick}#entity={entity_id}",
                    f"profile:{Path(profile['schema_name']).name}#compute.{category}",
                ],
            )
            row.update(
                {
                    "schema_name": "compute_state",
                    "schema_version": SCHEMA_VERSION,
                    "tick": tick,
                    "node_id": f"compute_node:{entity_id}",
                    "queue_id": f"compute_queue:{entity_id}",
                    "execution_id": stable_identifier(
                        "task_execution", inputs.episode_id, entity_id, tick
                    ),
                    "allocation_id": stable_identifier(
                        "resource_allocation", inputs.episode_id, entity_id, tick
                    ),
                    "resource_id": stable_identifier(
                        "compute_resource", inputs.episode_id, entity_id, "aggregate"
                    ),
                    "failure_state_id": stable_identifier(
                        "failure_state",
                        inputs.episode_id,
                        entity_id,
                        failure_mode,
                    ),
                    "offload_id": stable_identifier(
                        "task_offload", inputs.episode_id, entity_id, tick
                    ),
                    "migration_id": stable_identifier(
                        "task_migration", inputs.episode_id, entity_id, tick
                    ),
                    "entity_id": entity_id,
                    "node_kind": f"{category}_edge_node",
                    "task_id": stable_identifier(
                        "compute_task", inputs.episode_id, entity_id
                    ),
                    "task_kind": _string_or_unknown(
                        task_profile.get("task_kind"), "unknown"
                    ),
                    "execution_state": execution_state,
                    "cpu_utilization_pct": _round(balance["cpu_utilization_pct"]),
                    "gpu_utilization_pct": _round(balance["gpu_utilization_pct"]),
                    "memory_used_mb": _round(allocation["memory_mb"]),
                    "queue_depth": queue_depth
                    if not missing_source_record
                    else "unknown",
                    "declared_capacity": _numeric_or_unknown_dict(declared_capacity),
                    "capacity": _numeric_or_unknown_dict(capacity),
                    "resource_request": _numeric_or_unknown_dict(demand),
                    "allocation": _numeric_or_unknown_dict(allocation),
                    "resource_balance": _numeric_or_unknown_dict(balance),
                    "inbound_allocation": {
                        "cpu_cores": _round(
                            inbound_cpu_by_node[f"compute_node:{entity_id}"]
                        )
                    },
                    "node_availability": {
                        "available": node_available,
                        "source": "deterministic_maintenance_schedule",
                    },
                    "offload": record["offload"],
                    "migration": {
                        "active": migration_active,
                        "source_node_id": f"compute_node:{entity_id}",
                        "target_node_id": record["offload"]["target_node_id"]
                        if migration_active
                        else "none",
                        "transferred_cpu_cores": record["offload"]["accepted_cpu_cores"]
                        if migration_active
                        else 0.0,
                    },
                    "failure": {
                        "mode": failure_mode,
                        "requirement_status": "missing_source_record"
                        if missing_source_record
                        else "not_required",
                        "missing_parameters": sorted(missing_parameters),
                    },
                    "deadline": deadline,
                }
            )
            rows.append(row)
    rows.sort(key=lambda row: (row["tick"], row["entity_id"], row["node_id"]))
    return rows


def _simulated_station_from_facility_anchor(
    roster_entities: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically synthesize one communication station from a facility.

    Episodes without an authored communication base station still need a
    communication anchor for UAV communication predicates.  The historical
    supplement used the lexicographically first positioned facility as that
    anchor (``simulated_communication_station:<facility_id>``); keep the same
    deterministic policy.
    """
    facilities = sorted(
        (
            dict(entity)
            for entity in roster_entities.values()
            if str(entity.get("entity_category") or "") == "facility"
            and _position(entity) is not None
        ),
        key=lambda item: str(item.get("entity_id") or ""),
    )
    if not facilities:
        return []
    anchor = facilities[0]
    station_id = (
        f"simulated_communication_station:"
        f"{str(anchor.get('entity_id') or '')}"
    )
    return [
        {
            "entity_id": station_id,
            "entity_category": "communication_station",
            "entity_kind": "facility.base_station",
            "initial_position_enu_m": _position(anchor),
            "station_source": "simulated_from_facility_anchor",
        }
    ]


def _build_communication_rows(
    inputs: EpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comm_profile = profile["communication"]
    flow_profiles = comm_profile.get("flow_profiles") or {}
    station_profiles = comm_profile.get("station_profiles") or {}
    channel_profile = comm_profile.get("channel") or {}
    handover_policy = comm_profile.get("handover_policy") or {}
    handover_margin_m = _number(handover_policy.get("handover_margin_m"), 0.0) or 0.0
    minimum_dwell_ticks = int(
        _number(handover_policy.get("minimum_dwell_ticks"), 0.0) or 0.0
    )
    base_station_profile = station_profiles.get("base_station")
    previous_station_by_entity: dict[str, str] = {}
    previous_station_mode_by_id: dict[str, str] = {}
    previous_declared_handover_by_entity: dict[str, bool | None] = {}
    logical_handover_by_entity: dict[str, tuple[str, str]] = {}
    last_switch_tick_by_entity: dict[str, int] = {}
    last_successful_heartbeat_tick_by_entity: dict[str, int] = {}
    has_observed_station_in_roster = any(
        _is_station_entity(entity) for entity in inputs.roster_entities.values()
    )
    for frame in inputs.frames:
        tick = int(frame["tick"])
        weather = inputs.weather_by_tick.get(tick)
        frame_entities = _current_entities(frame)
        stations = _station_entities(frame_entities)
        if not stations and not has_observed_station_in_roster:
            stations = _simulated_station_from_facility_anchor(
                inputs.roster_entities
            )
        station_modes = {
            str(station["entity_id"]): str(
                _structured_communication_input(station).get("mode", "online")
            )
            for station in stations
        }
        station_entered_backup_link = {
            station_id
            for station_id, mode in station_modes.items()
            if mode == "backup_link"
            and previous_station_mode_by_id.get(station_id) not in {
                None,
                "backup_link",
            }
        }
        station_ids = set(station_modes)
        for entity in frame_entities:
            category = _entity_category(entity)
            if category != "uav":
                continue
            entity_id = _string_or_unknown(entity.get("entity_id"))
            flow_profile = flow_profiles.get(category)
            station_profile = base_station_profile
            missing_parameters: list[str] = []
            if not isinstance(flow_profile, Mapping):
                missing_parameters.append(f"communication.flow_profiles.{category}")
                flow_profile = {}
            if not isinstance(station_profile, Mapping):
                missing_parameters.append("communication.station_profiles.base_station")
                station_profile = {}
            if not stations:
                missing_parameters.append("communication.station_roster.base_station")
            if weather is None:
                missing_parameters.append("inputs.weather.tick")
            else:
                for weather_field in ("rain", "fog_density", "dust"):
                    if _number(weather.get(weather_field)) is None:
                        missing_parameters.append(f"weather_meta.{weather_field}")
            for parameter_name in (
                "latency_threshold_ms",
                "packet_loss_threshold_ratio",
            ):
                if _number(flow_profile.get(parameter_name)) is None:
                    missing_parameters.append(
                        f"communication.flow_profiles.{category}.{parameter_name}"
                    )
            for parameter_name in (
                "nominal_latency_ms",
                "worst_case_latency_ms",
                "maximum_packet_loss_ratio",
            ):
                if _number(channel_profile.get(parameter_name)) is None:
                    missing_parameters.append(f"communication.channel.{parameter_name}")
            candidate_station, candidate_distance_m = _select_station(entity, stations)
            station, distance_m, handover_evidence = _apply_handover_policy(
                entity=entity,
                candidate_station=candidate_station,
                candidate_distance_m=candidate_distance_m,
                stations=stations,
                previous_station_id=previous_station_by_entity.get(entity_id),
                last_switch_tick=last_switch_tick_by_entity.get(entity_id),
                tick=tick,
                margin_m=handover_margin_m,
                minimum_dwell_ticks=minimum_dwell_ticks,
            )
            station_id = station.get("entity_id") if station else "unknown"
            station_source = (
                str(station.get("station_source"))
                if station is not None
                else "unavailable"
            )
            if station is None or distance_m is None:
                missing_parameters.append("communication.current_station_distance")
            station_operational_state = _station_operational_state(
                station,
                entity,
                station_profile,
            )
            missing_parameters.extend(
                station_operational_state.pop("missing_parameters")
            )
            station_state_source_refs = station_operational_state.pop("source_refs")
            quality = _link_quality(
                distance_m,
                weather,
                channel_profile,
                station_profile,
                station_operational_state,
            )
            heartbeat_age_ms = _heartbeat_age_ms(
                entity_id=entity_id,
                tick=tick,
                tick_hz=inputs.tick_hz,
                quality=quality,
                last_success_tick_by_entity=last_successful_heartbeat_tick_by_entity,
            )
            bandwidth_request = _number(flow_profile.get("bandwidth_mbps"))
            nominal_bandwidth_capacity = _number(
                station_profile.get("channel_bandwidth_mbps")
            )
            capacity_factor = _number(station_operational_state.get("capacity_factor"))
            bandwidth_capacity = (
                None
                if nominal_bandwidth_capacity is None or capacity_factor is None
                else max(0.0, nominal_bandwidth_capacity * capacity_factor)
            )
            if bandwidth_request is None:
                missing_parameters.append(
                    f"communication.flow_profiles.{category}.bandwidth_mbps"
                )
            if nominal_bandwidth_capacity is None:
                missing_parameters.append(
                    "communication.station_profiles.base_station.channel_bandwidth_mbps"
                )
            allocated_bandwidth = (
                None
                if bandwidth_request is None or bandwidth_capacity is None
                else min(bandwidth_request, bandwidth_capacity)
            )
            queued_bandwidth = (
                None
                if bandwidth_request is None or allocated_bandwidth is None
                else max(0.0, bandwidth_request - allocated_bandwidth)
            )
            previous_station = previous_station_by_entity.get(entity_id)
            geometric_handover_active = (
                False
                if station_id == "unknown" or previous_station in (None, station_id)
                else True
            )
            declared_handover_active = _structured_boolean(
                _structured_communication_input(entity),
                "handover_active",
            )
            declared_handover_rising = (
                declared_handover_active is True
                and previous_declared_handover_by_entity.get(entity_id) is not True
            )
            backup_station_id = station_profile.get("backup_station_id")
            logical_handover_started = (
                station_id in station_entered_backup_link
                and declared_handover_rising
            )
            if logical_handover_started:
                if (
                    not isinstance(backup_station_id, str)
                    or not backup_station_id
                    or backup_station_id == station_id
                    or backup_station_id not in station_ids
                ):
                    missing_parameters.append(
                        "communication.station_profiles.base_station.backup_station_id"
                    )
                else:
                    logical_handover_by_entity[entity_id] = (
                        str(station_id),
                        backup_station_id,
                    )
            if declared_handover_active is not True:
                logical_handover_by_entity.pop(entity_id, None)
            logical_handover = logical_handover_by_entity.get(entity_id)
            logical_handover_active = (
                declared_handover_active is True and logical_handover is not None
            )
            handover_active = geometric_handover_active or logical_handover_active
            previous_declared_handover_by_entity[entity_id] = (
                declared_handover_active
            )
            if station_id != "unknown":
                if previous_station not in (None, station_id):
                    last_switch_tick_by_entity[entity_id] = tick
                elif entity_id not in last_switch_tick_by_entity:
                    last_switch_tick_by_entity[entity_id] = tick
                previous_station_by_entity[entity_id] = str(station_id)
            retransmission_count = _retransmission_count(
                quality,
                weather,
                channel_profile,
                tick=tick,
            )
            source_required = bool(missing_parameters)
            row = _common_row(
                common,
                source_class="simulated_derived",
                rule_id="compute_comm_supplement.communication_state",
                rule_version=profile["communication"]["rule_version"],
                source_refs=[
                    f"{inputs.input_files['truth_frames'].name}#tick={tick}#entity={entity_id}",
                    f"{inputs.input_files['weather'].name}#tick={tick}",
                    inputs.tick_hz_source_ref,
                    (
                        "global_entity_roster.json#facility_anchor"
                        if station_source == "simulated_from_facility_anchor"
                        else f"{inputs.input_files['truth_frames'].name}#tick={tick}#station={station_id}"
                    ),
                    *[
                        f"{inputs.input_files['truth_frames'].name}#tick={tick}#{source_ref}"
                        for source_ref in station_state_source_refs
                    ],
                ],
            )
            row.update(
                {
                    "schema_name": "communication_state",
                    "schema_version": SCHEMA_VERSION,
                    "tick": tick,
                    "station_id": station_id,
                    "station_source": station_source,
                    "session_id": stable_identifier(
                        "session", inputs.episode_id, entity_id, station_id
                    ),
                    "flow_id": stable_identifier(
                        "flow", inputs.episode_id, entity_id, station_id
                    ),
                    "message_id": stable_identifier(
                        "message",
                        inputs.episode_id,
                        entity_id,
                        tick,
                        common["seed_digest"],
                    ),
                    "channel_allocation_id": stable_identifier(
                        "channel_allocation",
                        inputs.episode_id,
                        entity_id,
                        station_id,
                    ),
                    "transmission_attempt_id": stable_identifier(
                        "transmission_attempt", inputs.episode_id, entity_id, tick
                    ),
                    "handover_id": stable_identifier(
                        "handover", inputs.episode_id, entity_id
                    ),
                    "retransmission_id": stable_identifier(
                        "retransmission", inputs.episode_id, entity_id, station_id
                    ),
                    "entity_id": entity_id,
                    "scope_active": True,
                    "scope_activity_authority": "truth_frames.entities",
                    "heartbeat_age_ms": heartbeat_age_ms,
                    "channel": {
                        "kind": _string_or_unknown(
                            channel_profile.get("kind"), "unknown"
                        ),
                        "capacity_mbps": _round(bandwidth_capacity),
                        "nominal_capacity_mbps": _round(nominal_bandwidth_capacity),
                        "allocated_bandwidth_mbps": _round(allocated_bandwidth),
                        "queued_bandwidth_mbps": _round(queued_bandwidth),
                    },
                    "bandwidth": {
                        "requested_mbps": _round(bandwidth_request),
                        "allocated_mbps": _round(allocated_bandwidth),
                        "dropped_mbps": _round(queued_bandwidth),
                    },
                    "route": {
                        "hops": [entity_id, station_id]
                        if station_id != "unknown"
                        else [entity_id, "unknown"],
                        "distance_m": _round(distance_m),
                        "requirement_status": "source_required"
                        if station_id == "unknown"
                        else "not_required",
                    },
                    "retransmission": {
                        "count": retransmission_count
                        if not source_required
                        else "unknown",
                        "reason": "unknown"
                        if source_required
                        else ("link_quality" if retransmission_count else "none"),
                    },
                    "handover": {
                        "active": handover_active if not source_required else "unknown",
                        "previous_station_id": (
                            logical_handover[0]
                            if logical_handover_active
                            else previous_station or "none"
                        ),
                        "current_station_id": (
                            logical_handover[1]
                            if logical_handover_active
                            else station_id
                        ),
                        "basis": (
                            "governed_station_backup_transition"
                            if logical_handover_active
                            else (
                                "geometric_station_switch"
                                if geometric_handover_active
                                else "none"
                            )
                        ),
                        "association_source": (
                            "communication.station_profiles.base_station."
                            "backup_station_id"
                            if logical_handover_active
                            else "none"
                        ),
                    },
                    "handover_policy": handover_evidence,
                    "station_operational_state": station_operational_state,
                    "link_quality": quality,
                    "quality_thresholds": {
                        "latency_ms": _round(
                            _number(flow_profile.get("latency_threshold_ms"))
                        ),
                        "packet_loss_ratio": _round(
                            _number(flow_profile.get("packet_loss_threshold_ratio"))
                        ),
                    },
                    "requirement": {
                        "status": "source_required"
                        if source_required
                        else "not_required",
                        "missing_parameters": sorted(set(missing_parameters)),
                    },
                }
            )
            rows.append(row)
        previous_station_mode_by_id = station_modes
    rows.sort(key=lambda row: (row["tick"], row["entity_id"], row["station_id"]))
    return rows


def _build_process_coverage_rows(
    inputs: EpisodeInputs,
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Record candidate-scope coverage for every authoritative supplement tick.

    A zero-candidate tick is an explicit, valid scope result.  It must not be
    represented by a fabricated compute node or communication session.
    """

    compute_counts = Counter(int(row["tick"]) for row in compute_rows)
    communication_counts = Counter(int(row["tick"]) for row in communication_rows)
    rows: list[dict[str, Any]] = []
    for frame in inputs.frames:
        tick = int(frame["tick"])
        compute_count = int(compute_counts.get(tick, 0))
        communication_count = int(communication_counts.get(tick, 0))
        row = _common_row(
            common,
            source_class="simulated_derived",
            rule_id="compute_comm_supplement.process_scope_coverage",
            rule_version=SCHEMA_VERSION,
            source_refs=[f"truth_frames.jsonl#tick={tick}"],
        )
        row.update(
            {
                "schema_name": "compute_comm_process_coverage",
                "schema_version": SCHEMA_VERSION,
                "coverage_id": stable_identifier(
                    "process_coverage",
                    inputs.episode_id,
                    tick,
                    compute_count,
                    communication_count,
                ),
                "tick": tick,
                "compute": {
                    "candidate_count": compute_count,
                    "coverage_status": "covered_with_candidates"
                    if compute_count
                    else "covered_zero_candidates",
                },
                "communication": {
                    "candidate_count": communication_count,
                    "coverage_status": "covered_with_candidates"
                    if communication_count
                    else "covered_zero_candidates",
                },
            }
        )
        rows.append(row)
    return rows


def _build_predicate_rows(
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    *,
    retained_predicate_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in compute_rows:
        candidates = [
            _predicate_row(
                common,
                profile,
                row,
                "compute.compute_node_unavailable",
                {"node": row["node_id"]},
                _truth_from_node_unavailable(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "compute.compute_resource_over_allocated",
                {"node": row["node_id"]},
                _truth_from_compute_capacity(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "compute.compute_task_deadline_missed",
                {"task": row["task_id"]},
                _truth_from_deadline_missed(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "compute.compute_task_execution_failed",
                {"task": row["task_id"]},
                _truth_from_failure(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "compute.compute_task_migrating",
                {"task": row["task_id"]},
                _truth_from_migration(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "compute.compute_task_offloaded",
                {"task": row["task_id"]},
                _truth_from_offload(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "compute.compute_task_queued",
                {"queue": row["queue_id"], "task": row["task_id"]},
                _truth_from_queue(row),
            ),
        ]
        rows.extend(
            candidate
            for candidate in candidates
            if retained_predicate_ids is None
            or candidate["predicate_id"] in retained_predicate_ids
        )
    for row in communication_rows:
        candidates = [
            _predicate_row(
                common,
                profile,
                row,
                "communication.channel_bandwidth_over_allocated",
                {"allocation": row["channel_allocation_id"]},
                _truth_from_bandwidth_overallocated(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "communication.communication_session_interrupted",
                {"session": row["session_id"]},
                _truth_from_link_unavailable(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "communication.data_flow_latency_above_threshold",
                {"flow": row["flow_id"]},
                _truth_from_latency(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "communication.handover_in_progress",
                {"handover": row["handover_id"]},
                _truth_from_handover(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "communication.link_degraded",
                {"actor": row["entity_id"], "station": row["station_id"]},
                _truth_from_link_degraded(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "communication.link_unavailable",
                {"actor": row["entity_id"], "station": row["station_id"]},
                _truth_from_link_unavailable(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "communication.message_transmission_failed",
                {"attempt": row["transmission_attempt_id"]},
                _truth_from_link_unavailable(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "communication.packet_loss_above_threshold",
                {"flow": row["flow_id"]},
                _truth_from_packet_loss(row),
            ),
            _predicate_row(
                common,
                profile,
                row,
                "communication.retransmission_in_progress",
                {"retransmission": row["retransmission_id"]},
                _truth_from_retransmission(row),
            ),
        ]
        rows.extend(
            candidate
            for candidate in candidates
            if retained_predicate_ids is None
            or candidate["predicate_id"] in retained_predicate_ids
        )
    rows.sort(
        key=lambda row: (
            row["tick"],
            row["predicate_id"],
            canonical_json(row["bindings"]),
        )
    )
    return rows


def _compute_predicate_values(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "compute.compute_node_unavailable": _truth_from_node_unavailable(row),
        "compute.compute_resource_over_allocated": _truth_from_compute_capacity(row),
        "compute.compute_task_deadline_missed": _truth_from_deadline_missed(row),
        "compute.compute_task_execution_failed": _truth_from_failure(row),
        "compute.compute_task_migrating": _truth_from_migration(row),
        "compute.compute_task_offloaded": _truth_from_offload(row),
        "compute.compute_task_queued": _truth_from_queue(row),
    }


def _communication_predicate_values(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "communication.channel_bandwidth_over_allocated": _truth_from_bandwidth_overallocated(
            row
        ),
        "communication.communication_session_interrupted": _truth_from_link_unavailable(
            row
        ),
        "communication.data_flow_latency_above_threshold": _truth_from_latency(row),
        "communication.handover_in_progress": _truth_from_handover(row),
        "communication.link_degraded": _truth_from_link_degraded(row),
        "communication.link_unavailable": _truth_from_link_unavailable(row),
        "communication.message_transmission_failed": _truth_from_link_unavailable(row),
        "communication.packet_loss_above_threshold": _truth_from_packet_loss(row),
        "communication.retransmission_in_progress": _truth_from_retransmission(row),
    }


def _build_predicate_matrix_rows(
    inputs: EpisodeInputs,
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compactly materialize every governed scope/predicate/tick value."""

    compute_values: dict[tuple[int, str], dict[str, str]] = {}
    for row in compute_rows:
        key = (int(row["tick"]), str(row["node_id"]))
        if key in compute_values:
            raise ComputeCommSimulationError(
                f"duplicate compute state for matrix scope: {key[1]}@{key[0]}"
            )
        compute_values[key] = _compute_predicate_values(row)
    communication_values = {
        (int(row["tick"]), str(row["entity_id"])): _communication_predicate_values(row)
        for row in communication_rows
    }
    return _build_predicate_matrix_rows_from_values(
        inputs,
        compute_values,
        communication_values,
        common,
    )


def _build_predicate_matrix_rows_from_values(
    inputs: EpisodeInputs,
    compute_values: Mapping[tuple[int, str], Mapping[str, str]],
    communication_values: Mapping[tuple[int, str], Mapping[str, str]],
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Materialize the compact full matrix from precomputed direct values."""

    compute_entities = [
        entity_id
        for entity_id, entity in sorted(inputs.roster_entities.items())
        if _entity_category(entity) in {"uav", "vehicle"}
    ]
    uav_entities = [
        entity_id
        for entity_id, entity in sorted(inputs.roster_entities.items())
        if _entity_category(entity) == "uav"
    ]
    rows: list[dict[str, Any]] = []
    for frame in inputs.frames:
        tick = int(frame["tick"])
        active_ids = {str(entity["entity_id"]) for entity in _current_entities(frame)}
        scope_values: list[dict[str, Any]] = []
        for entity_id in compute_entities:
            node_id = f"compute_node:{entity_id}"
            key = (tick, node_id)
            scope_active = entity_id in active_ids
            if scope_active:
                values = compute_values.get(key)
                if values is None or set(values) != set(COMPUTE_PREDICATE_IDS):
                    raise ComputeCommSimulationError(
                        f"active compute scope lacks direct values: {node_id}@{tick}"
                    )
            else:
                if key in compute_values:
                    raise ComputeCommSimulationError(
                        f"inactive compute scope has a state row: {node_id}@{tick}"
                    )
                values = {
                    predicate_id: "out_of_scope" for predicate_id in COMPUTE_PREDICATE_IDS
                }
            scope_values.append(
                {
                    "scope_type": "compute_node",
                    "scope_entity_id": node_id,
                    "scope_active": scope_active,
                    "workload_active": scope_active,
                    "predicate_values": dict(sorted(values.items())),
                }
            )
        for entity_id in uav_entities:
            scope_active = entity_id in active_ids
            values = (
                communication_values.get((tick, entity_id), {})
                if scope_active
                else {
                    predicate_id: "out_of_scope"
                    for predicate_id in COMMUNICATION_PREDICATE_IDS
                }
            )
            if scope_active and set(values) != set(COMMUNICATION_PREDICATE_IDS):
                raise ComputeCommSimulationError(
                    f"active communication scope lacks a state row: {entity_id}@{tick}"
                )
            if scope_active and set(values.values()) & {"unknown"}:
                missing = sorted(
                    predicate_id
                    for predicate_id, value in values.items()
                    if value == "unknown"
                )
                raise ComputeCommSimulationError(
                    f"active communication scope lacks direct values: "
                    f"{entity_id}@{tick}: {missing}"
                )
            scope_values.append(
                {
                    "scope_type": "uav",
                    "scope_entity_id": entity_id,
                    "scope_active": scope_active,
                    "workload_active": scope_active,
                    "predicate_values": dict(sorted(values.items())),
                }
            )
        scope_values.sort(
            key=lambda value: (str(value["scope_type"]), str(value["scope_entity_id"]))
        )
        rows.append(_predicate_matrix_tick_row(common, tick, scope_values))
    expected_scope_rows = len(inputs.frames) * (
        len(compute_entities) + len(uav_entities)
    )
    actual_scope_rows = sum(len(row["scope_values"]) for row in rows)
    if actual_scope_rows != expected_scope_rows:
        raise ComputeCommSimulationError(
            "predicate matrix scope count mismatch: "
            f"{actual_scope_rows} != {expected_scope_rows}"
        )
    return rows


def _predicate_matrix_tick_row(
    common: Mapping[str, Any],
    tick: int,
    scope_values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    invalid = sorted(
        {
            value
            for scope in scope_values
            for value in scope["predicate_values"].values()
        }
        - TRUTH_VALUES
    )
    if invalid:
        raise ComputeCommSimulationError(
            f"predicate matrix has invalid truth values: {invalid}"
        )
    row = _common_row(
        common,
        source_class="simulated_derived",
        rule_id="compute_comm_supplement.full_predicate_matrix",
        rule_version=SCHEMA_VERSION,
        source_refs=[f"truth_frames.jsonl#tick={tick}", "global_entity_roster.json"],
    )
    row.update(
        {
            "schema_name": "compute_comm_predicate_truth_matrix",
            "schema_version": SCHEMA_VERSION,
            "matrix_row_id": stable_identifier(
                "compute_comm_predicate_matrix",
                common["episode_id"],
                tick,
                scope_values,
            ),
            "tick": tick,
            "scope_values": list(scope_values),
        }
    )
    return row


def _build_event_rows(
    predicate_rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    event_map = profile["event_rules"]
    tick_step = int(profile["authoritative_tick_policy"]["step"])
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predicate_rows:
        grouped[(str(row["predicate_id"]), canonical_json(row["bindings"]))].append(row)
    events: list[dict[str, Any]] = []
    for (predicate_id, _bindings_key), rows in grouped.items():
        event_rule = event_map.get(predicate_id)
        if not isinstance(event_rule, Mapping):
            continue
        rows_sorted = sorted(rows, key=lambda row: int(row["tick"]))
        for previous, current in zip(rows_sorted, rows_sorted[1:]):
            if int(current["tick"]) - int(previous["tick"]) != tick_step:
                continue
            if previous["value"] != "false" or current["value"] != "true":
                continue
            event_type = event_rule["event_type_id"]
            event_bindings = _event_bindings(event_type, event_rule, current)
            if set(event_bindings) != ONTOLOGY_EVENT_ROLES[event_type]:
                raise ComputeCommSimulationError(
                    f"event {event_type} bindings do not match ontology roles: {sorted(event_bindings)}"
                )
            event_id = stable_identifier(
                "event",
                common["episode_id"],
                event_type,
                current["tick"],
                current["predicate_truth_id"],
            )
            row = _common_row(
                common,
                source_class="simulated_derived",
                rule_id=event_rule["rule_id"],
                rule_version=event_rule["rule_version"],
                source_refs=[
                    previous["predicate_truth_id"],
                    current["predicate_truth_id"],
                ],
            )
            row.update(
                {
                    "schema_name": "compute_comm_event",
                    "schema_version": SCHEMA_VERSION,
                    "event_id": event_id,
                    "event_type_id": event_type,
                    "derivation_kind": "predicate_rising",
                    "trigger_tick": current["tick"],
                    "source_predicate_id": predicate_id,
                    "transition": {
                        "from_tick": previous["tick"],
                        "to_tick": current["tick"],
                        "from_value": previous["value"],
                        "to_value": current["value"],
                    },
                    "bindings": event_bindings,
                }
            )
            events.append(row)
    events.sort(
        key=lambda row: (row["trigger_tick"], row["event_type_id"], row["event_id"])
    )
    return events


def _build_provenance_rows(
    inputs: EpisodeInputs,
    profile_path: Path,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    profile_digest: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in sorted(inputs.input_files.items()):
        row = _common_row(
            common,
            source_class="observed_simulator_truth",
            rule_id="compute_comm_supplement.input_source",
            rule_version=SCHEMA_VERSION,
            source_refs=[path.name],
        )
        row.update(
            {
                "schema_name": "compute_comm_provenance",
                "schema_version": SCHEMA_VERSION,
                "provenance_id": stable_identifier(
                    "provenance", inputs.episode_id, name, inputs.input_digests[name]
                ),
                "record_kind": "input_file",
                "source_name": name,
                "source_path": str(path),
                "source_digest": inputs.input_digests[name],
            }
        )
        rows.append(row)
    parameter_row = _common_row(
        common,
        source_class=str(profile["parameter_governance"]["source_class"]),
        rule_id="compute_comm_supplement.profile_parameters",
        rule_version=SCHEMA_VERSION,
        source_refs=[str(profile_path)],
    )
    parameter_row.update(
        {
            "schema_name": "compute_comm_provenance",
            "schema_version": SCHEMA_VERSION,
            "provenance_id": stable_identifier(
                "provenance", inputs.episode_id, "profile", profile_digest
            ),
            "record_kind": "profile",
            "source_name": profile["profile_id"],
            "source_path": str(profile_path),
            "source_digest": profile_digest,
        }
    )
    rows.append(parameter_row)
    seed_row = _common_row(
        common,
        source_class="simulated_derived",
        rule_id="compute_comm_supplement.seed_derivation",
        rule_version=SCHEMA_VERSION,
        source_refs=[*(str(path) for path in inputs.input_files.values()), str(profile_path)],
    )
    seed_row.update(
        {
            "schema_name": "compute_comm_provenance",
            "schema_version": SCHEMA_VERSION,
            "provenance_id": stable_identifier(
                "provenance", inputs.episode_id, "seed", common["seed_digest"]
            ),
            "record_kind": "seed",
            "source_name": "deterministic_seed",
            "source_path": "none",
            "source_digest": common["seed_digest"],
        }
    )
    rows.append(seed_row)
    return rows


def validate_output_rows(
    *,
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    predicate_rows: Sequence[Mapping[str, Any]],
    predicate_matrix_rows: Sequence[Mapping[str, Any]] = (),
    event_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    tick_step: int,
    coverage_rows: Sequence[Mapping[str, Any]] = (),
    expected_ticks: Sequence[int] = (),
) -> None:
    for row in (
        list(compute_rows)
        + list(communication_rows)
        + list(coverage_rows)
        + list(predicate_rows)
        + list(predicate_matrix_rows)
        + list(event_rows)
        + list(provenance_rows)
    ):
        _validate_provenance_fields(row)
    for row in compute_rows:
        _validate_compute_capacity(row)
    compute_by_tick_node = {
        (int(row["tick"]), str(row["node_id"])): row for row in compute_rows
    }
    accepted_by_target: dict[tuple[int, str], float] = defaultdict(float)
    for row in compute_rows:
        offload = row["offload"]
        if not bool(offload.get("enabled")):
            continue
        key = (int(row["tick"]), str(offload["target_node_id"]))
        if key not in compute_by_tick_node:
            raise ComputeCommSimulationError(
                f"offload target is not a real compute node at tick {row['tick']}: {key[1]}"
            )
        accepted_by_target[key] += float(offload["accepted_cpu_cores"])
    for key, target in compute_by_tick_node.items():
        inbound = float(target.get("inbound_allocation", {}).get("cpu_cores") or 0.0)
        if abs(inbound - accepted_by_target.get(key, 0.0)) > NUMERIC_TOLERANCE:
            raise ComputeCommSimulationError(
                f"offload transfer does not balance at tick {key[0]} for {key[1]}"
            )
    for row in communication_rows:
        _validate_communication_capacity(row)
    if coverage_rows:
        coverage_ticks = [int(row["tick"]) for row in coverage_rows]
        if coverage_ticks != sorted(set(coverage_ticks)):
            raise ComputeCommSimulationError(
                "process coverage ticks must be unique and sorted"
            )
        if expected_ticks and coverage_ticks != list(expected_ticks):
            raise ComputeCommSimulationError(
                "process coverage ticks do not match the authoritative tick sequence"
            )
        compute_counts = Counter(int(row["tick"]) for row in compute_rows)
        communication_counts = Counter(int(row["tick"]) for row in communication_rows)
        for row in coverage_rows:
            tick = int(row["tick"])
            for name, counts in (
                ("compute", compute_counts),
                ("communication", communication_counts),
            ):
                scope = row.get(name)
                if not isinstance(scope, Mapping):
                    raise ComputeCommSimulationError(
                        f"process coverage lacks {name} scope at tick {tick}"
                    )
                actual_count = int(scope.get("candidate_count") or 0)
                expected_count = int(counts.get(tick, 0))
                if actual_count != expected_count:
                    raise ComputeCommSimulationError(
                        f"process coverage candidate count mismatch for {name} at tick {tick}"
                    )
                expected_status = (
                    "covered_with_candidates"
                    if expected_count
                    else "covered_zero_candidates"
                )
                if str(scope.get("coverage_status")) != expected_status:
                    raise ComputeCommSimulationError(
                        f"process coverage status mismatch for {name} at tick {tick}"
                    )
    for row in predicate_rows:
        if row.get("value") not in TRUTH_VALUES:
            raise ComputeCommSimulationError(
                f"invalid four-valued predicate truth: {row.get('value')}"
            )
    for row in predicate_matrix_rows:
        scopes = row.get("scope_values")
        if not isinstance(scopes, list):
            raise ComputeCommSimulationError(
                "predicate truth matrix lacks scope_values"
            )
        for scope in scopes:
            values = (
                scope.get("predicate_values") if isinstance(scope, Mapping) else None
            )
            if not isinstance(values, Mapping) or set(values.values()) - TRUTH_VALUES:
                raise ComputeCommSimulationError(
                    f"invalid predicate truth matrix row: {row.get('matrix_row_id')}"
                )
    predicate_by_id = {row["predicate_truth_id"]: row for row in predicate_rows}
    for event in event_rows:
        transition = event.get("transition") or {}
        if (
            transition.get("from_value") != "false"
            or transition.get("to_value") != "true"
        ):
            raise ComputeCommSimulationError(
                f"invalid event transition: {event.get('event_id')}"
            )
        if int(transition["to_tick"]) - int(transition["from_tick"]) != tick_step:
            raise ComputeCommSimulationError(
                f"non-adjacent event transition: {event.get('event_id')}"
            )
        for source_ref in event.get("source_refs", []):
            if (
                str(source_ref).startswith("predicate_truth:")
                and source_ref not in predicate_by_id
            ):
                raise ComputeCommSimulationError(
                    f"event cites missing predicate truth: {source_ref}"
                )


def _validate_provenance_fields(row: Mapping[str, Any]) -> None:
    for key in (
        "source_class",
        "rule_id",
        "rule_version",
        "model_id",
        "model_version",
        "input_digest",
        "parameter_digest",
        "seed_digest",
        "source_refs",
    ):
        if key not in row:
            raise ComputeCommSimulationError(
                f"row lacks provenance field {key}: {row.get('schema_name')}"
            )
    if row["source_class"] not in SOURCE_CLASSES:
        raise ComputeCommSimulationError(f"invalid source_class: {row['source_class']}")


def _validate_compute_capacity(row: Mapping[str, Any]) -> None:
    capacity = row["capacity"]
    allocation = row["allocation"]
    balance = row["resource_balance"]
    inbound_cpu = (
        _number(row.get("inbound_allocation", {}).get("cpu_cores"), 0.0) or 0.0
    )
    for key in ("cpu_cores", "gpu_units", "memory_mb"):
        cap = capacity.get(key)
        used = allocation.get(key)
        if key == "cpu_cores" and isinstance(used, (int, float)):
            used = float(used) + inbound_cpu
        if (
            isinstance(cap, (int, float))
            and isinstance(used, (int, float))
            and used - cap > NUMERIC_TOLERANCE
        ):
            raise ComputeCommSimulationError(
                f"compute allocation exceeds capacity for {row['node_id']}:{key}"
            )
    for key in ("cpu", "gpu", "memory"):
        requested = balance.get(f"{key}_requested")
        allocated = balance.get(f"{key}_allocated")
        queued = balance.get(f"{key}_queued")
        offloaded = balance.get(f"{key}_offloaded", 0.0)
        if all(
            isinstance(value, (int, float))
            for value in (requested, allocated, queued, offloaded)
        ):
            if (
                abs(
                    float(requested)
                    - float(allocated)
                    - float(queued)
                    - float(offloaded)
                )
                > NUMERIC_TOLERANCE
            ):
                raise ComputeCommSimulationError(
                    f"compute conservation failed for {row['node_id']}:{key}"
                )
    offload = row.get("offload")
    if not isinstance(offload, Mapping):
        raise ComputeCommSimulationError(
            f"compute row lacks offload record for {row['node_id']}"
        )
    accepted = _number(offload.get("accepted_cpu_cores"), 0.0) or 0.0
    if bool(offload.get("enabled")) != (accepted > 0.0):
        raise ComputeCommSimulationError(
            f"offload enablement mismatch for {row['node_id']}"
        )
    if accepted > 0.0 and str(offload.get("target_node_id")) == "none":
        raise ComputeCommSimulationError(
            f"offload lacks real target for {row['node_id']}"
        )


def _validate_communication_capacity(row: Mapping[str, Any]) -> None:
    heartbeat_age_ms = row.get("heartbeat_age_ms")
    if heartbeat_age_ms != "unknown" and (
        not isinstance(heartbeat_age_ms, (int, float)) or heartbeat_age_ms < 0
    ):
        raise ComputeCommSimulationError(
            f"invalid heartbeat_age_ms for {row.get('entity_id', 'unknown')}"
        )
    channel = row["channel"]
    capacity = channel.get("capacity_mbps")
    allocated = channel.get("allocated_bandwidth_mbps")
    if (
        isinstance(capacity, (int, float))
        and isinstance(allocated, (int, float))
        and allocated - capacity > 1e-6
    ):
        raise ComputeCommSimulationError(
            f"communication allocation exceeds channel capacity for {row['flow_id']}"
        )
    requested = row["bandwidth"].get("requested_mbps")
    dropped = row["bandwidth"].get("dropped_mbps")
    if all(
        isinstance(value, (int, float)) for value in (requested, allocated, dropped)
    ):
        if abs(float(requested) - float(allocated) - float(dropped)) > 1e-6:
            raise ComputeCommSimulationError(
                f"communication bandwidth conservation failed for {row['flow_id']}"
            )


def _common_row(
    common: Mapping[str, Any],
    *,
    source_class: str,
    rule_id: str,
    rule_version: str,
    source_refs: Sequence[str],
) -> dict[str, Any]:
    return {
        "episode_id": common["episode_id"],
        "source_class": source_class,
        "rule_id": rule_id,
        "rule_version": rule_version,
        "model_id": common["model_id"],
        "model_version": common["model_version"],
        "input_digest": common["input_digest"],
        "parameter_digest": common["parameter_digest"],
        "seed_digest": common["seed_digest"],
        "source_refs": sorted(str(ref) for ref in source_refs),
    }


def _predicate_row(
    common: Mapping[str, Any],
    profile: Mapping[str, Any],
    source_row: Mapping[str, Any],
    predicate_id: str,
    bindings: Mapping[str, str],
    value: str,
) -> dict[str, Any]:
    expected_roles = ONTOLOGY_PREDICATE_ROLES.get(predicate_id)
    if expected_roles is None:
        raise ComputeCommSimulationError(
            f"predicate id is absent from the ontology-aligned registry: {predicate_id}"
        )
    if set(bindings) != expected_roles:
        raise ComputeCommSimulationError(
            f"predicate {predicate_id} bindings do not match ontology roles: "
            f"expected={sorted(expected_roles)}, actual={sorted(bindings)}"
        )
    rule = profile["predicate_rules"][predicate_id]
    predicate_truth_id = stable_identifier(
        "predicate_truth",
        common["episode_id"],
        predicate_id,
        source_row["tick"],
        bindings,
        value,
    )
    row = _common_row(
        common,
        source_class="simulated_derived",
        rule_id=rule["rule_id"],
        rule_version=rule["rule_version"],
        source_refs=[source_row.get("node_id") or source_row.get("flow_id")],
    )
    row.update(
        {
            "schema_name": "compute_comm_predicate_truth",
            "schema_version": SCHEMA_VERSION,
            "predicate_truth_id": predicate_truth_id,
            "tick": source_row["tick"],
            "predicate_id": predicate_id,
            "bindings": {
                role: str(bindings[role])
                for role in ONTOLOGY_PREDICATE_ROLE_ORDER[predicate_id]
            },
            "binding_ontology_classes": dict(
                ONTOLOGY_PREDICATE_ROLE_CLASSES[predicate_id]
            ),
            "event_participants": _event_participants(source_row),
            "binding_status": (
                "incomplete_binding"
                if any(str(item) == "unknown" for item in bindings.values())
                else "complete"
            ),
            "value": value,
            "requirement_status": _source_requirement_status(source_row),
        }
    )
    return row


def _truth_from_compute_capacity(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) in {"source_required", "missing_source_record"}:
        return "unknown"
    comparisons: list[bool | None] = []
    for resource in ("cpu_cores", "gpu_units", "memory_mb"):
        requested = _number(row["resource_request"][resource])
        capacity = _number(row["capacity"][resource])
        comparisons.append(
            requested > capacity
            if requested is not None and capacity is not None
            else None
        )
    if any(value is True for value in comparisons):
        return "true"
    return "unknown" if None in comparisons else "false"


def _truth_from_node_unavailable(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) == "missing_source_record":
        return "unknown"
    available = row.get("node_availability", {}).get("available")
    if isinstance(available, bool):
        return "false" if available else "true"
    return "unknown"


def _truth_from_deadline_missed(row: Mapping[str, Any]) -> str:
    status = row["deadline"].get("status")
    if status not in {"inactive", "within_budget", "overdue", "unknown"}:
        raise ComputeCommSimulationError(f"undeclared queue-segment deadline status: {status!r}")
    # Queue waiting does not establish a ComputeTask arrival or completion.
    # The task-level predicate requires that evidence and remains unknown.
    return "unknown"


def _truth_from_failure(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) in {"source_required", "missing_source_record"}:
        return "unknown"
    state = row["execution_state"]
    if state == "failed":
        return "true"
    if state in {"running", "running_offloaded", "queued", "migrating"}:
        return "false"
    if state == "unknown":
        return "unknown"
    raise ComputeCommSimulationError(f"undeclared compute execution state: {state!r}")


def _truth_from_migration(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) == "missing_source_record":
        return "unknown"
    active = row.get("migration", {}).get("active")
    if isinstance(active, bool):
        return "true" if active else "false"
    return "unknown"


def _truth_from_offload(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) == "source_required":
        return "unknown"
    return "true" if bool(row["offload"].get("enabled")) else "false"


def _truth_from_queue(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) in {"source_required", "missing_source_record"}:
        return "unknown"
    depth = row.get("queue_depth")
    if type(depth) is not int:
        return "unknown"
    return "true" if depth > 0 else "false"


def _truth_from_bandwidth_overallocated(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) == "source_required":
        return "unknown"
    dropped = row["bandwidth"].get("dropped_mbps")
    return "true" if isinstance(dropped, (int, float)) and dropped > 0 else "false"


def _truth_from_link_degraded(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) in {"source_required", "missing_source_record"}:
        return "unknown"
    unavailable = _truth_from_link_unavailable(row)
    if unavailable == "true":
        return "false"
    if unavailable == "unknown":
        return "unknown"
    violations = (_truth_from_latency(row), _truth_from_packet_loss(row))
    if "true" in violations:
        return "true"
    return "unknown" if "unknown" in violations else "false"


def _truth_from_link_unavailable(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) == "source_required":
        return "unknown"
    level = row["link_quality"].get("quality_level")
    if level == "down":
        return "true"
    if level in {"excellent", "good", "fair", "poor"}:
        return "false"
    return "unknown"


def _truth_from_latency(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) == "source_required":
        return "unknown"
    value = row["link_quality"].get("latency_ms")
    threshold = row["quality_thresholds"].get("latency_ms")
    if not isinstance(value, (int, float)) or not isinstance(threshold, (int, float)):
        return "unknown"
    return "true" if value > threshold else "false"


def _truth_from_packet_loss(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) == "source_required":
        return "unknown"
    value = row["link_quality"].get("packet_loss_ratio")
    threshold = row["quality_thresholds"].get("packet_loss_ratio")
    if not isinstance(value, (int, float)) or not isinstance(threshold, (int, float)):
        return "unknown"
    return "true" if value > threshold else "false"


def _truth_from_retransmission(row: Mapping[str, Any]) -> str:
    if _source_requirement_status(row) == "source_required":
        return "unknown"
    count = row["retransmission"].get("count")
    if not isinstance(count, int):
        return "unknown"
    return "true" if count > 0 else "false"


def _truth_from_handover(row: Mapping[str, Any]) -> str:
    active = row["handover"].get("active")
    if active == "unknown":
        return "unknown"
    return "true" if bool(active) else "false"


def _source_requirement_status(row: Mapping[str, Any]) -> str:
    if "requirement" in row:
        return str(row["requirement"].get("status"))
    if "failure" in row:
        return str(row["failure"].get("requirement_status"))
    return "not_required"


def _event_participants(source_row: Mapping[str, Any]) -> dict[str, str]:
    if source_row.get("schema_name") == "compute_state":
        return {
            "node": str(source_row["node_id"]),
            "resource": str(source_row["resource_id"]),
            "execution": str(source_row["execution_id"]),
            "task": str(source_row["task_id"]),
            "failure_state": str(source_row["failure_state_id"]),
        }
    if source_row.get("schema_name") == "communication_state":
        handover = source_row.get("handover") or {}
        return {
            "actor": str(source_row["entity_id"]),
            "station": str(source_row["station_id"]),
            "session": str(source_row["session_id"]),
            "flow": str(source_row["flow_id"]),
            "handover": str(source_row["handover_id"]),
            "source_station": str(handover.get("previous_station_id") or "none"),
            "target_station": str(
                handover.get("current_station_id") or source_row["station_id"]
            ),
            "retransmission": str(source_row["retransmission_id"]),
            "original_attempt": str(source_row["transmission_attempt_id"]),
            "message": str(source_row["message_id"]),
            "attempt": str(source_row["transmission_attempt_id"]),
            "allocation": str(source_row["channel_allocation_id"]),
        }
    return {}


def _event_bindings(
    event_type: str,
    event_rule: Mapping[str, Any],
    predicate_row: Mapping[str, Any],
) -> dict[str, str]:
    participant_bindings = event_rule.get("participant_bindings")
    if not isinstance(participant_bindings, Mapping):
        raise ComputeCommSimulationError(
            f"event {event_type} lacks participant_bindings"
        )
    participants = predicate_row.get("event_participants")
    if not isinstance(participants, Mapping):
        raise ComputeCommSimulationError(
            f"predicate row lacks event participants for {event_type}"
        )
    bindings: dict[str, str] = {}
    for role, participant_key in participant_bindings.items():
        if not isinstance(participant_key, str) or participant_key not in participants:
            raise ComputeCommSimulationError(
                f"event {event_type} participant {role} cannot be resolved from {participant_key}"
            )
        bindings[str(role)] = str(participants[participant_key])
    return bindings


def _compute_demand(
    entity: Mapping[str, Any],
    frame: Mapping[str, Any],
    task_profile: Mapping[str, Any],
    _seed_digest: str,
) -> dict[str, Any]:
    missing: list[str] = []
    base_cpu = _number(task_profile.get("cpu_cores"))
    base_gpu = _number(task_profile.get("gpu_units"))
    base_memory = _number(task_profile.get("memory_mb"))
    cpu_per_mps = _number(task_profile.get("cpu_cores_per_mps"), 0.0)
    variation = _number(task_profile.get("deterministic_variation"), 0.0)
    for key, value in (
        ("cpu_cores", base_cpu),
        ("gpu_units", base_gpu),
        ("memory_mb", base_memory),
    ):
        if value is None:
            missing.append(f"task_profiles.{_entity_category(entity)}.{key}")
    speed = _entity_speed(entity)
    if speed is None:
        speed = 0.0
    tick = int(frame.get("tick") or 0)
    load_multiplier = 1.0
    load_windows = task_profile.get("load_windows")
    if not isinstance(load_windows, list):
        missing.append(f"task_profiles.{_entity_category(entity)}.load_windows")
        load_windows = []
    for window in load_windows:
        if not isinstance(window, Mapping):
            missing.append(f"task_profiles.{_entity_category(entity)}.load_windows[]")
            continue
        start_tick = _number(window.get("start_tick"))
        end_tick = _number(window.get("end_tick"))
        multiplier = _number(window.get("demand_multiplier"))
        if start_tick is None or end_tick is None or multiplier is None:
            missing.append(f"task_profiles.{_entity_category(entity)}.load_windows[]")
            continue
        if int(start_tick) <= tick <= int(end_tick):
            load_multiplier *= multiplier
    scale = (
        1.0
        + (
            2.0
            * _unit_interval(
                "compute_current_state_v1",
                _entity_category(entity),
                frame.get("tick"),
                _round(speed),
            )
            - 1.0
        )
        * variation
    )
    return {
        "cpu_cores": None
        if base_cpu is None
        else max(0.0, (base_cpu + cpu_per_mps * speed) * scale * load_multiplier),
        "gpu_units": None
        if base_gpu is None
        else max(0.0, base_gpu * scale * load_multiplier),
        "memory_mb": None
        if base_memory is None
        else max(0.0, base_memory * scale * load_multiplier),
        "missing_parameters": missing,
    }


def _allocate_compute(
    demand: Mapping[str, Any],
    capacity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    allocation: dict[str, Any] = {}
    balance: dict[str, Any] = {}
    for demand_key, balance_key in (
        ("cpu_cores", "cpu"),
        ("gpu_units", "gpu"),
        ("memory_mb", "memory"),
    ):
        requested = _number(demand.get(demand_key))
        cap = _number(capacity.get(demand_key))
        if requested is None or cap is None:
            allocation[demand_key] = None
            balance[f"{balance_key}_requested"] = None
            balance[f"{balance_key}_allocated"] = None
            balance[f"{balance_key}_queued"] = None
            continue
        allocated = min(requested, cap)
        queued = max(0.0, requested - allocated)
        allocation[demand_key] = allocated
        balance[f"{balance_key}_requested"] = requested
        balance[f"{balance_key}_allocated"] = allocated
        balance[f"{balance_key}_queued"] = queued
    cpu_cap = _number(capacity.get("cpu_cores"))
    gpu_cap = _number(capacity.get("gpu_units"))
    mem_cap = _number(capacity.get("memory_mb"))
    balance["cpu_utilization_pct"] = _percent(allocation.get("cpu_cores"), cpu_cap)
    balance["gpu_utilization_pct"] = _percent(allocation.get("gpu_units"), gpu_cap)
    balance["memory_utilization_pct"] = _percent(allocation.get("memory_mb"), mem_cap)
    balance["queue_depth"] = _compute_queue_depth(balance)
    return allocation, balance


def _compute_queue_depth(balance: Mapping[str, Any]) -> int | None:
    queued = [
        _number(balance[f"{resource}_queued"])
        for resource in ("cpu", "gpu", "memory")
    ]
    if any(value is None for value in queued):
        return None
    cpu_queued, gpu_queued, memory_queued = queued
    # One aggregate task must wait for any unmet GPU or memory demand; those
    # resource quantities are not task counts. Keep the existing CPU depth.
    return max(
        int(math.ceil(cpu_queued)),
        int(gpu_queued > 0.0),
        int(memory_queued > 0.0),
    )


def _compute_node_available(
    compute_profile: Mapping[str, Any],
    *,
    tick: int,
    category: str,
    entity_rank: int,
) -> bool:
    schedule = compute_profile.get("node_unavailability_schedule")
    if not isinstance(schedule, list):
        raise ComputeCommSimulationError(
            "compute.node_unavailability_schedule must be an explicit list"
        )
    for interval in schedule:
        if not isinstance(interval, Mapping):
            raise ComputeCommSimulationError(
                "compute.node_unavailability_schedule entries must be objects"
            )
        required = {"entity_category", "entity_rank", "start_tick", "end_tick"}
        if set(interval) != required:
            raise ComputeCommSimulationError(
                "compute.node_unavailability_schedule entries must have exactly "
                f"{sorted(required)}"
            )
        if (
            str(interval["entity_category"]) == category
            and int(interval["entity_rank"]) == entity_rank
            and int(interval["start_tick"]) <= tick <= int(interval["end_tick"])
        ):
            return False
    return True


def _select_compute_offload_target(
    *,
    source_entity_id: str,
    tick: int,
    frame_records: Sequence[Mapping[str, Any]],
    residual_cpu_by_node: Mapping[str, float],
    seed_digest: str,
) -> str | None:
    candidates = sorted(
        f"compute_node:{record['entity_id']}"
        for record in frame_records
        if record["entity_id"] != source_entity_id
        and bool(record["node_available"])
        and residual_cpu_by_node.get(f"compute_node:{record['entity_id']}", 0.0) > 0.0
    )
    if not candidates:
        return None
    offset = int(
        _unit_interval(
            "compute_offload_target_v1",
            seed_digest,
            source_entity_id,
            tick,
        )
        * len(candidates)
    )
    return candidates[min(offset, len(candidates) - 1)]


def _compute_offload(
    *,
    source_entity_id: str,
    requested_cpu_cores: float,
    accepted_cpu_cores: float,
    target_node_id: str | None,
) -> dict[str, Any]:
    enabled = accepted_cpu_cores > 0.0 and target_node_id is not None
    return {
        "enabled": enabled,
        "source_node_id": f"compute_node:{source_entity_id}",
        "target_node_id": target_node_id if enabled else "none",
        "requested_cpu_cores": _round(requested_cpu_cores),
        "accepted_cpu_cores": _round(accepted_cpu_cores),
        "reason": "local_capacity_pressure"
        if requested_cpu_cores > 0.0
        else "not_required",
    }


def _deadline(
    task_profile: Mapping[str, Any],
    *,
    tick: int,
    queue_started_tick: int | None,
    missing_source_record: bool,
) -> dict[str, Any]:
    slack = task_profile.get("deadline_slack_ticks")
    if missing_source_record or type(slack) is not int or slack < 0:
        return {
            "queue_started_tick": "unknown",
            "deadline_tick": "unknown",
            "slack_ticks": "unknown",
            "status": "unknown",
            "requirement_status": "missing_source_record",
        }
    if queue_started_tick is None:
        return {
            "queue_started_tick": None,
            "deadline_tick": None,
            "slack_ticks": slack,
            "status": "inactive",
            "requirement_status": "not_applicable",
        }
    deadline_tick = queue_started_tick + slack
    return {
        "queue_started_tick": queue_started_tick,
        "deadline_tick": deadline_tick,
        "slack_ticks": slack,
        "status": "overdue" if tick >= deadline_tick else "within_budget",
        "requirement_status": "not_required",
    }


def _station_entities(
    entities: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_entities = [copy.deepcopy(dict(entity)) for entity in entities]
    stations: list[dict[str, Any]] = []
    for entity in source_entities:
        if _is_station_entity(entity):
            entity["station_source"] = "observed_simulator_truth"
            stations.append(entity)
    stations.sort(key=lambda item: str(item.get("entity_id") or ""))
    return stations


def _is_station_entity(entity: Mapping[str, Any]) -> bool:
    kind = str(entity.get("entity_kind") or "")
    category = str(entity.get("entity_category") or "")
    return (
        "base_station" in kind
        or "communication" in kind
        or category == "communication_station"
    )


def _structured_communication_input(entity: Mapping[str, Any] | None) -> dict[str, Any]:
    return _structured_state_subset(
        entity,
        "communication_state",
        {
            "station_unavailable",
            "communication_unavailable",
            "available",
            "link_available",
            "availability",
            "mode",
            "handover_active",
        },
    )


def _structured_security_input(entity: Mapping[str, Any] | None) -> dict[str, Any]:
    return _structured_state_subset(
        entity,
        "security_state",
        {
            "jamming",
            "jamming_active",
            "gcs_compromised",
        },
    )


def _structured_state_subset(
    entity: Mapping[str, Any] | None,
    family: str,
    allowed_fields: set[str],
) -> dict[str, Any]:
    if entity is None:
        return {}
    state = entity.get(family)
    if not isinstance(state, Mapping):
        return {}
    result: dict[str, Any] = {}
    for field in sorted(allowed_fields):
        value = state.get(field)
        if isinstance(value, bool):
            result[field] = value
        elif field == "mode" and isinstance(value, str) and value:
            result[field] = value
        elif field == "availability":
            number = _number(value)
            if number is not None:
                result[field] = max(0.0, min(1.0, number))
    return result


def _structured_boolean(
    state: Mapping[str, Any],
    field: str,
) -> bool | None:
    value = state.get(field)
    return value if isinstance(value, bool) else None


def _structured_availability(state: Mapping[str, Any]) -> float | None:
    value = _number(state.get("availability"))
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def _station_operational_state(
    station: Mapping[str, Any] | None,
    actor: Mapping[str, Any] | None,
    station_profile: Mapping[str, Any],
) -> dict[str, Any]:
    operational_states = station_profile.get("operational_states")
    missing_parameters: list[str] = []
    if station is None:
        missing_parameters.append("truth_frames.entities.station")
    station_communication = _structured_communication_input(station)
    observed_mode = station_communication.get("mode", "online")
    if not isinstance(operational_states, Mapping):
        missing_parameters.append(
            "communication.station_profiles.base_station.operational_states"
        )
        baseline_profile: Mapping[str, Any] = {}
    else:
        selected = operational_states.get(observed_mode)
        if not isinstance(selected, Mapping):
            missing_parameters.append(
                f"communication.station_profiles.base_station.operational_states.{observed_mode}"
            )
            baseline_profile = {}
        else:
            baseline_profile = selected
    structured_sources = [
        ("station", station_communication, _structured_security_input(station)),
        (
            "actor",
            _structured_communication_input(actor),
            _structured_security_input(actor),
        ),
    ]
    source_refs: list[str] = []
    reasons: list[str] = []
    has_structured_state = False
    unavailable = False
    degraded = False
    explicit_availability_values: list[float] = []
    for role, communication_state, security_state in structured_sources:
        for field in sorted(communication_state):
            source_refs.append(f"{role}.communication_state.{field}")
        for field in sorted(security_state):
            source_refs.append(f"{role}.security_state.{field}")
        if communication_state or security_state:
            has_structured_state = True
        for field in ("station_unavailable", "communication_unavailable"):
            state_value = _structured_boolean(communication_state, field)
            if state_value is True:
                unavailable = True
                reasons.append(f"{role}.communication_state.{field}")
        for field in ("available", "link_available"):
            state_value = _structured_boolean(communication_state, field)
            if state_value is False:
                unavailable = True
                reasons.append(f"{role}.communication_state.{field}=false")
        availability_value = _structured_availability(communication_state)
        if availability_value is not None:
            explicit_availability_values.append(availability_value)
            if availability_value <= 0.0:
                unavailable = True
                reasons.append(f"{role}.communication_state.availability=0")
            elif availability_value < 1.0:
                degraded = True
                reasons.append(f"{role}.communication_state.availability<1")
        for field in ("jamming", "jamming_active"):
            state_value = _structured_boolean(security_state, field)
            if state_value is True:
                degraded = True
                reasons.append(f"{role}.security_state.{field}")
        if _structured_boolean(security_state, "gcs_compromised") is True:
            unavailable = True
            reasons.append(f"{role}.security_state.gcs_compromised")
    state_mode = "unknown"
    values: dict[str, Any] = {
        "activity_type": state_mode,
        "state_mode": state_mode,
        "state_basis": "structured_communication_security_state",
        "state_reasons": sorted(set(reasons)),
        "source_class": str(station.get("station_source"))
        if station is not None
        else "unavailable",
    }
    for key in (
        "operational_factor",
        "capacity_factor",
        "range_factor",
        "availability",
    ):
        value = _number(baseline_profile.get(key))
        if value is None:
            missing_parameters.append(
                "communication.station_profiles.base_station.operational_states."
                f"{observed_mode}.{key}"
            )
            values[key] = "unknown"
        else:
            values[key] = max(0.0, min(1.0, value))
    if has_structured_state and not missing_parameters:
        if unavailable:
            state_mode = "structured_unavailable"
            for key in (
                "operational_factor",
                "capacity_factor",
                "range_factor",
                "availability",
            ):
                values[key] = 0.0
        else:
            if explicit_availability_values:
                minimum_availability = min(explicit_availability_values)
                values["availability"] = min(
                    values["availability"], minimum_availability
                )
            if degraded:
                state_mode = "structured_degraded"
                values["operational_factor"] = min(values["operational_factor"], 0.22)
                values["capacity_factor"] = min(values["capacity_factor"], 0.6)
                values["range_factor"] = min(values["range_factor"], 0.75)
                values["availability"] = min(values["availability"], 0.85)
            elif not explicit_availability_values:
                state_mode = "observed_" + str(observed_mode)
            else:
                state_mode = "structured_nominal"
    elif not missing_parameters:
        state_mode = f"profile_nominal_{observed_mode}"
        values["state_basis"] = "governed_profile_no_runtime_override"
    values["activity_type"] = state_mode
    values["state_mode"] = state_mode
    values["missing_parameters"] = sorted(set(missing_parameters))
    values["source_refs"] = sorted(set(source_refs))
    return values


def _select_station(
    entity: Mapping[str, Any],
    stations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, float | None]:
    entity_pos = _position(entity)
    if entity_pos is None:
        return (None, None)
    best_station: dict[str, Any] | None = None
    best_distance: float | None = None
    for station in stations:
        station_pos = _position(station)
        if station_pos is None:
            continue
        distance = math.dist(entity_pos[:3], station_pos[:3])
        if best_distance is None or distance < best_distance:
            best_station = copy.deepcopy(dict(station))
            best_distance = distance
    return best_station, best_distance


def _apply_handover_policy(
    *,
    entity: Mapping[str, Any],
    candidate_station: Mapping[str, Any] | None,
    candidate_distance_m: float | None,
    stations: Sequence[Mapping[str, Any]],
    previous_station_id: str | None,
    last_switch_tick: int | None,
    tick: int,
    margin_m: float,
    minimum_dwell_ticks: int,
) -> tuple[dict[str, Any] | None, float | None, dict[str, Any]]:
    if (
        candidate_station is None
        or candidate_distance_m is None
        or not previous_station_id
    ):
        return (
            copy.deepcopy(dict(candidate_station))
            if candidate_station is not None
            else None,
            candidate_distance_m,
            {
                "handover_margin_m": _round(margin_m),
                "minimum_dwell_ticks": minimum_dwell_ticks,
                "last_switch_tick": last_switch_tick
                if last_switch_tick is not None
                else "none",
                "candidate_station_id": (
                    candidate_station.get("entity_id")
                    if candidate_station is not None
                    else "unknown"
                ),
                "candidate_distance_m": _round(candidate_distance_m),
                "current_station_distance_m": "unknown",
                "distance_advantage_m": "unknown",
                "dwell_elapsed_ticks": "unknown",
                "switch_allowed": False,
                "decision": "initial_station_or_unavailable",
            },
        )
    current_station = next(
        (
            station
            for station in stations
            if station.get("entity_id") == previous_station_id
        ),
        None,
    )
    entity_pos = _position(entity)
    station_pos = _position(current_station) if current_station is not None else None
    if entity_pos is None or station_pos is None:
        return (
            copy.deepcopy(dict(candidate_station)),
            candidate_distance_m,
            {
                "handover_margin_m": _round(margin_m),
                "minimum_dwell_ticks": minimum_dwell_ticks,
                "last_switch_tick": last_switch_tick
                if last_switch_tick is not None
                else "none",
                "candidate_station_id": candidate_station.get("entity_id"),
                "candidate_distance_m": _round(candidate_distance_m),
                "current_station_distance_m": "unknown",
                "distance_advantage_m": "unknown",
                "dwell_elapsed_ticks": "unknown",
                "switch_allowed": True,
                "decision": "current_station_unavailable",
            },
        )
    current_distance_m = math.dist(entity_pos[:3], station_pos[:3])
    dwell_elapsed = tick - (last_switch_tick if last_switch_tick is not None else tick)
    distance_advantage_m = current_distance_m - candidate_distance_m
    candidate_id = str(candidate_station.get("entity_id"))
    switch_allowed = (
        candidate_id != previous_station_id
        and distance_advantage_m > margin_m
        and dwell_elapsed >= minimum_dwell_ticks
    )
    selected_station = candidate_station if switch_allowed else current_station
    selected_distance = candidate_distance_m if switch_allowed else current_distance_m
    return (
        copy.deepcopy(dict(selected_station)),
        selected_distance,
        {
            "handover_margin_m": _round(margin_m),
            "minimum_dwell_ticks": minimum_dwell_ticks,
            "last_switch_tick": last_switch_tick
            if last_switch_tick is not None
            else "none",
            "candidate_station_id": candidate_id,
            "candidate_distance_m": _round(candidate_distance_m),
            "current_station_distance_m": _round(current_distance_m),
            "distance_advantage_m": _round(distance_advantage_m),
            "dwell_elapsed_ticks": dwell_elapsed,
            "switch_allowed": switch_allowed,
            "decision": "switch" if switch_allowed else "retain_current_station",
        },
    )


def _heartbeat_age_ms(
    *,
    entity_id: str,
    tick: int,
    tick_hz: float,
    quality: Mapping[str, Any],
    last_success_tick_by_entity: dict[str, int],
) -> float | str:
    """Derive heartbeat age from the numeric link state for one entity.

    A known usable link represents a successful heartbeat at the current tick
    and resets the age to zero.  A known unavailable link advances from the
    entity's last successful tick using the authoritative episode clock.  If no
    successful heartbeat has ever been observed, the age remains unknown.
    """

    if entity_id == "unknown":
        return "unknown"

    level = str(quality.get("quality_level") or "unknown")
    availability = _number(quality.get("availability"))
    link_successful = (
        level in {"excellent", "good", "fair", "poor"}
        and availability is not None
        and availability > 0
    )
    link_unavailable = level == "down" or (
        availability is not None and availability <= 0
    )

    if link_successful:
        last_success_tick_by_entity[entity_id] = tick
        return 0.0
    if not link_unavailable:
        return "unknown"

    last_success_tick = last_success_tick_by_entity.get(entity_id)
    if last_success_tick is None:
        return "unknown"
    elapsed_ticks = tick - last_success_tick
    if elapsed_ticks < 0:
        raise ComputeCommSimulationError(
            f"heartbeat ticks are not monotonic for {entity_id}: "
            f"last_success={last_success_tick}, current={tick}"
        )
    return _round(elapsed_ticks * 1000.0 / tick_hz)


def _link_quality(
    distance_m: float | None,
    weather: Mapping[str, Any] | None,
    channel_profile: Mapping[str, Any],
    station_profile: Mapping[str, Any],
    station_operational_state: Mapping[str, Any],
) -> dict[str, Any]:
    operational_factor = _number(station_operational_state.get("operational_factor"))
    capacity_factor = _number(station_operational_state.get("capacity_factor"))
    range_factor = _number(station_operational_state.get("range_factor"))
    availability = _number(station_operational_state.get("availability"))
    if distance_m is None or weather is None or any(
        _number(weather.get(field)) is None
        for field in ("rain", "fog_density", "dust")
    ):
        return {
            "quality_score": "unknown",
            "quality_level": "unknown",
            "latency_ms": "unknown",
            "packet_loss_ratio": "unknown",
            "distance_m": _round(distance_m),
            "effective_range_m": "unknown",
            "weather_attenuation": "unknown",
            "operational_factor": _round(operational_factor),
            "capacity_factor": _round(capacity_factor),
            "range_factor": _round(range_factor),
            "availability": _round(availability),
        }
    nominal_range = _number(station_profile.get("nominal_range_m"))
    if (
        nominal_range is None
        or nominal_range <= 0
        or operational_factor is None
        or capacity_factor is None
        or range_factor is None
        or availability is None
    ):
        return {
            "quality_score": "unknown",
            "quality_level": "unknown",
            "latency_ms": "unknown",
            "packet_loss_ratio": "unknown",
            "distance_m": _round(distance_m),
            "effective_range_m": "unknown",
            "weather_attenuation": "unknown",
            "operational_factor": _round(operational_factor),
            "capacity_factor": _round(capacity_factor),
            "range_factor": _round(range_factor),
            "availability": _round(availability),
        }
    effective_range = nominal_range * max(0.0, range_factor)
    weather_attenuation = _weather_attenuation(weather, channel_profile)
    if effective_range <= 0 or operational_factor <= 0 or availability <= 0:
        range_score = 0.0
        score = 0.0
    else:
        range_score = max(0.0, min(1.0, 1.0 - distance_m / effective_range))
        score = max(
            0.0,
            min(
                1.0,
                range_score * weather_attenuation * operational_factor * availability,
            ),
        )
    if score >= 0.75:
        level = "excellent"
    elif score >= 0.5:
        level = "good"
    elif score >= 0.25:
        level = "fair"
    elif score > 0:
        level = "poor"
    else:
        level = "down"
    nominal_latency = _number(channel_profile.get("nominal_latency_ms"), 20.0) or 20.0
    worst_latency = (
        _number(channel_profile.get("worst_case_latency_ms"), 220.0) or 220.0
    )
    maximum_packet_loss = (
        _number(channel_profile.get("maximum_packet_loss_ratio"), 0.25) or 0.25
    )
    latency_ms = nominal_latency + (1.0 - score) * max(
        0.0, worst_latency - nominal_latency
    )
    packet_loss_ratio = max(0.0, min(1.0, (1.0 - score) ** 2 * maximum_packet_loss))
    return {
        "quality_score": _round(score),
        "quality_level": level,
        "latency_ms": _round(latency_ms),
        "packet_loss_ratio": _round(packet_loss_ratio),
        "distance_m": _round(distance_m),
        "effective_range_m": _round(effective_range),
        "weather_attenuation": _round(weather_attenuation),
        "operational_factor": _round(operational_factor),
        "capacity_factor": _round(capacity_factor),
        "range_factor": _round(range_factor),
        "availability": _round(availability),
    }


def _weather_attenuation(
    weather: Mapping[str, Any], channel_profile: Mapping[str, Any]
) -> float:
    rain = _number(weather.get("rain"))
    fog = _number(weather.get("fog_density"))
    dust = _number(weather.get("dust"))
    if rain is None or fog is None or dust is None:
        raise ComputeCommSimulationError("weather attenuation requires rain, fog_density, and dust")
    coeff = channel_profile.get("weather_attenuation") or {}
    rain_coeff = _number(coeff.get("rain"), 0.0) or 0.0
    fog_coeff = _number(coeff.get("fog_density"), 0.0) or 0.0
    dust_coeff = _number(coeff.get("dust"), 0.0) or 0.0
    return max(0.05, 1.0 - rain * rain_coeff - fog * fog_coeff - dust * dust_coeff)


def _retransmission_count(
    quality: Mapping[str, Any],
    weather: Mapping[str, Any] | None,
    channel_profile: Mapping[str, Any],
    *,
    tick: int,
) -> int:
    level = str(quality.get("quality_level"))
    if level in {"excellent", "good"}:
        base_count = 0
    elif level == "fair":
        base_count = 1
    elif level == "poor":
        base_count = 2
    elif level == "down":
        base_count = 4
    else:
        return 0 if weather is not None else 0
    packet_loss_ratio = _number(quality.get("packet_loss_ratio"), 0.0) or 0.0
    floor_probability = (
        _number(
            channel_profile.get("deterministic_retransmission_probability_floor"),
            0.0,
        )
        or 0.0
    )
    probability = max(0.0, min(1.0, max(packet_loss_ratio, floor_probability)))
    sample = _unit_interval(
        "deterministic_transport_retransmission_v1",
        tick,
        _round(quality.get("quality_score")),
        _round(quality.get("latency_ms")),
        _round(quality.get("packet_loss_ratio")),
        _round(quality.get("distance_m")),
    )
    return max(base_count, 1 if sample < probability else 0)


def _build_summary(
    inputs: EpisodeInputs,
    profile: Mapping[str, Any],
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    predicate_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    predicate_matrix_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ticks = list(
        range(
            int(profile["authoritative_tick_policy"]["start"]),
            int(profile["authoritative_tick_policy"]["end"]) + 1,
            int(profile["authoritative_tick_policy"]["step"]),
        )
    )
    observed_ticks = sorted({int(frame["tick"]) for frame in inputs.frames})
    missing_ticks = [tick for tick in expected_ticks if tick not in set(observed_ticks)]
    compute_candidate_ticks = {int(row["tick"]) for row in compute_rows}
    communication_candidate_ticks = {int(row["tick"]) for row in communication_rows}
    coverage_ticks = [int(row["tick"]) for row in coverage_rows]
    return {
        "schema_name": "compute_comm_supplement_summary",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "episode_id": inputs.episode_id,
        "profile_id": profile["profile_id"],
        "source_observed_tick_count": len(observed_ticks),
        "observed_ticks": observed_ticks,
        "process_coverage_ticks": coverage_ticks,
        "process_coverage_complete": coverage_ticks == expected_ticks,
        "missing_authoritative_ticks": missing_ticks,
        "compute_zero_candidate_ticks": [
            tick for tick in observed_ticks if tick not in compute_candidate_ticks
        ],
        "communication_zero_candidate_ticks": [
            tick for tick in observed_ticks if tick not in communication_candidate_ticks
        ],
        "record_counts": {
            "compute_state": len(compute_rows),
            "communication_state": len(communication_rows),
            "process_coverage": len(coverage_rows),
            "predicate_truth": len(predicate_rows),
            "predicate_truth_matrix": len(predicate_matrix_rows),
            "events": len(event_rows),
        },
        "predicate_matrix_value_counts": dict(
            Counter(
                str(value)
                for row in predicate_matrix_rows
                for scope in row["scope_values"]
                for value in scope["predicate_values"].values()
            )
        ),
        "truth_value_counts": dict(
            Counter(str(row["value"]) for row in predicate_rows)
        ),
        "source_required_counts": {
            "compute_state": sum(
                row["failure"]["requirement_status"] == "source_required"
                for row in compute_rows
            ),
            "communication_state": sum(
                row["requirement"]["status"] == "source_required"
                for row in communication_rows
            ),
            "predicate_truth": sum(
                row["requirement_status"] == "source_required" for row in predicate_rows
            ),
        },
        "forbidden_inputs_used": [],
        "parameter_governance": profile["parameter_governance"],
    }


def _build_simulation_manifest(
    inputs: EpisodeInputs,
    profile: Mapping[str, Any],
    common: Mapping[str, Any],
    output_dir: Path,
    file_texts: Mapping[str, str],
) -> dict[str, Any]:
    artifacts = {
        name: {
            "path": name,
            "bytes": len(text.encode("utf-8")),
        }
        for name, text in sorted(file_texts.items())
    }
    return {
        "schema_name": "compute_comm_supplement_manifest",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
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
        "artifacts": artifacts,
    }


def _current_entities(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    entities = frame.get("entities")
    if not isinstance(entities, list):
        raise ComputeCommSimulationError(
            f"truth frame tick {frame.get('tick')} lacks entities array"
        )
    result: list[dict[str, Any]] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            raise ComputeCommSimulationError(
                f"truth frame tick {frame.get('tick')} entity {index} is not an object"
            )
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ComputeCommSimulationError(
                f"truth frame tick {frame.get('tick')} entity {index} lacks entity_id"
            )
        result.append(copy.deepcopy(dict(entity)))
    return result


def _index_entities(values: Any, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise ComputeCommSimulationError(f"{source}: entities must be an array")
    result: dict[str, dict[str, Any]] = {}
    for index, entity in enumerate(values):
        if not isinstance(entity, Mapping):
            raise ComputeCommSimulationError(
                f"{source}: entity {index} must be an object"
            )
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ComputeCommSimulationError(
                f"{source}: entity {index} lacks entity_id"
            )
        if entity_id in result:
            raise ComputeCommSimulationError(
                f"{source}: duplicate entity_id {entity_id}"
            )
        result[entity_id] = copy.deepcopy(dict(entity))
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ComputeCommSimulationError(f"{path} must contain a JSON object")
    return value


def _artifact_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    """Keep simulation seeds in memory while publishing path-based provenance."""
    excluded = {"input_digest", "raw_input_digest", "parameter_digest", "seed_digest", "source_digest"}
    for row in rows:
        public = {key: value for key, value in row.items() if key not in excluded}
        public["schema_version"] = ARTIFACT_SCHEMA_VERSION
        yield public


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    for row in rows:
        buffer.write(canonical_json(row))
        buffer.write("\n")
    return buffer.getvalue()


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )




def _entity_category(entity: Mapping[str, Any]) -> str:
    return str(entity.get("entity_category") or entity.get("category") or "unknown")


def _entity_speed(entity: Mapping[str, Any]) -> float | None:
    for path in (
        ("sumo_vehicle", "speed_mps"),
        ("annotations", "speed_mps"),
    ):
        current: Any = entity
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        number = _number(current)
        if number is not None:
            return number
    pose = entity.get("truth_pose")
    if isinstance(pose, Mapping):
        velocity = pose.get("velocity_enu_mps")
        if (
            isinstance(velocity, list)
            and len(velocity) >= 2
            and all(isinstance(v, (int, float)) for v in velocity[:2])
        ):
            return math.sqrt(float(velocity[0]) ** 2 + float(velocity[1]) ** 2)
    return None


def _position(entity: Mapping[str, Any]) -> list[float] | None:
    pose = entity.get("truth_pose")
    position = (
        pose.get("position_enu_m")
        if isinstance(pose, Mapping)
        else entity.get("initial_position_enu_m")
    )
    if (
        isinstance(position, list)
        and len(position) >= 3
        and all(isinstance(value, (int, float)) for value in position[:3])
    ):
        return [float(position[0]), float(position[1]), float(position[2])]
    return None


def _string_or_unknown(value: Any, default: str = "unknown") -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _round(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    if value is None:
        return "unknown"
    return value


def _numeric_or_unknown_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _round(value)
        for key, value in sorted(values.items())
        if key != "missing_parameters"
    }


def _percent(used: Any, capacity: Any) -> float | None:
    used_number = _number(used)
    capacity_number = _number(capacity)
    if used_number is None or capacity_number is None or capacity_number <= 0:
        return None
    return used_number / capacity_number * 100.0


def _unit_interval(*parts: Any) -> float:
    digest = digest_object(parts).split(":", 1)[1]
    return int(digest[:16], 16) / float(16**16 - 1)
