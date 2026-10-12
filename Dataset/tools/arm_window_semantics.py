#!/usr/bin/env python3
"""Window-local Arm predicate/event evaluation from reconstructed state.

An Arm owns only its post-fork delta.  This module overlays that delta on the
read-only parent capture, re-evaluates the family's target predicate on the
capture grid, and derives occurrence/outcome from measured rising edges.
It does not invent ticks, copy parent labels as Arm truth, or claim G10.
"""

from __future__ import annotations

import gzip
import functools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
    get_governed_parameter_defaults,
)
from Dataset.semantic_truth.world_truth import _evaluate_expression, _expression_truth_value, _missing_value
from Dataset.semantic_truth.provenance import canonical_json, digest_object, open_artifact_text
from Dataset.tools.causal_query_bindings import (
    FIXED_GEOMETRY_PREDICATES, bind_causal_query, scene_geometry_classes,
)
from Dataset.semantic_simulation.domain_state import (
    UNKNOWN as DOMAIN_UNKNOWN,
    _gnss_values,
    domain_seed_digest,
    load_domain_state_profile,
)
from Dataset.semantic_simulation.compute_comm import build_communication_state_rows
from Dataset.semantic_simulation.predicate_state_computers import (
    _airspace_corridors,
    _corridor_active_at_tick,
    _point_in_polygon_xy,
    _point_in_oriented_corridor_box,
    _restricted_regions,
    _signed_distance_to_restricted_prism,
    restricted_region_activity_by_tick,
    PredicateStateComputerError,
    resolve_aircraft_ground_references,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    PROJECT_ROOT / "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json"
)
FAMILY_MAP_PATH = (
    PROJECT_ROOT / "Dataset/semantic_rules/profiles/event_family_authority_mapping.json"
)
OBJECTIVE_ROOT = PROJECT_ROOT / "aw_data/objective_semantic_truth"
CAPTURE_ROOT = PROJECT_ROOT / "aw_data/render_ready_episodes_capture_filtered"
CAPTURE_STEP = 5
DOMAIN_PROFILE_PATH = (
    PROJECT_ROOT
    / "Dataset/semantic_rules/profiles/domain_state_supplement_profile.json"
)
DOMAIN_SUPPLEMENT_ROOT = PROJECT_ROOT / "aw_data/domain_state_supplement"
COMPUTE_COMM_ROOT = PROJECT_ROOT / "aw_data/compute_comm_supplement"
COMPUTE_COMM_PROFILE_PATH = (
    PROJECT_ROOT / "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json"
)

GEOMETRY_PREDICATES = {
    "airspace.aircraft_airborne",
    "airspace.aircraft_touchdown",
    "airspace.aircraft_at_home_pad",
    "airspace.aircraft_in_landing_zone",
    "airspace.aircraft_inside_restricted_airspace",
    "airspace.aircraft_outside_assigned_altitude_band",
    "safety.aircraft_pair_below_separation",
    "safety.aircraft_near_pedestrian",
    "safety.aircraft_near_ground_vehicle",
    "safety.aircraft_near_building_structure",
    "safety.vehicle_pedestrian_conflict",
    "facility.landing_pad_contention",
    "airspace.airspace_corridor_over_capacity",
}
WEATHER_PREDICATES = {
    "environment.rain_active",
    "environment.fog_active",
    "environment.wind_speed_above_threshold",
    "environment.visibility_below_threshold",
    "environment.low_illumination_active",
    "environment.temperature_outside_operating_range",
}
DOMAIN_PREDICATES = {
    "positioning.gnss_quality_degraded",
    "positioning.gnss_spoofing_active",
    "security.communication_jamming_impact_active",
    "security.ground_control_station_compromised",
    "communication.link_unavailable",
    "communication.link_degraded",
    "facility.charging_station_unavailable",
    "ground.traffic_signal_malfunction_active",
    "ground.road_segment_closed",
    "safety.crowd_evacuation_active",
}


class WindowSemanticsError(RuntimeError):
    """Window-local evaluation cannot proceed without inventing authority."""


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WindowSemanticsError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(dict(value)) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    reader = gzip.open(path, "rt", encoding="utf-8-sig") if path.suffix == ".gz" else open_artifact_text(path, encoding="utf-8-sig")
    with reader as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _read_jsonl_window(path: Path, start_tick: int, end_tick: int) -> list[dict[str, Any]]:
    """Read only the authoritative temporal slice needed by one Arm.

    Parent episodes are formal 0..900 streams.  Loading the complete stream
    for every one of thousands of Arms wastes I/O and memory while adding no
    semantic information; rows outside the local fork window cannot affect
    this evaluator.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    reader = gzip.open(path, "rt", encoding="utf-8-sig") if path.suffix == ".gz" else open_artifact_text(path, encoding="utf-8-sig")
    with reader as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                continue
            raw_tick = value.get("tick")
            if raw_tick is None or isinstance(raw_tick, bool):
                continue
            try:
                tick = int(raw_tick)
            except (TypeError, ValueError):
                continue
            if start_tick <= tick <= end_tick:
                rows.append(value)
    return rows


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    coords = [_number(item) for item in value[:3]]
    if any(item is None for item in coords):
        return None
    return [float(item) for item in coords]  # type: ignore[misc]


def _speed(row: Mapping[str, Any]) -> float | None:
    """Return the measured speed, keeping an absent velocity unresolved."""

    vel = _vector3(row.get("vel_mps"))
    if vel is None:
        return None
    return math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)


def _missing_source_record(source: str, tick: int | None, field: str) -> str:
    """Build a traceable missing-source record without inventing a tick."""

    suffix = f"#tick={int(tick)}" if tick is not None else ""
    return f"missing_source_record:{source}{suffix}&field={field}"


def _missing_observations(
    source: str,
    tick: int | None,
    *fields: str,
) -> dict[str, Any]:
    records = [_missing_source_record(source, tick, field) for field in fields]
    return {"missing_source_record": records}


RESTRICTED_PREDICATE = "airspace.aircraft_inside_restricted_airspace"


def _restricted_template() -> Mapping[str, Any]:
    template = next(
        (item for item in get_core_predicate_templates() if item["id"] == RESTRICTED_PREDICATE),
        None,
    )
    if template is None:
        raise WindowSemanticsError("restricted predicate missing from the live registry")
    if template["argument_roles"] != [
        {"key": "actor", "class": "UnmannedAircraft"},
        {"key": "restricted_region", "class": "RestrictedAirspaceRegion"},
    ]:
        raise WindowSemanticsError("restricted identity disagrees with governed roles")
    required = [str(item["field"]) for item in template["state_contract"]["required_fields"]]
    if required != [
        "geometry.restricted_region_active",
        "geometry.minimum_restricted_boundary_distance_m",
    ]:
        raise WindowSemanticsError("restricted required fields disagree with the governed contract")
    return template


def _static_restricted_region_activity(
    region: Mapping[str, Any], tick: int
) -> bool | None:
    """Same static contract as restricted_region_activity_by_tick (exclusive deactivation).

    event_script_only regions require measured branch activity and return None here.
    """
    if region.get("spawn_policy") == "event_script_only":
        return None
    try:
        activation_tick = int(region.get("activation_tick", 0))
    except (TypeError, ValueError):
        return None
    deactivation_tick = region.get("deactivation_tick")
    return tick >= activation_tick and (
        not isinstance(deactivation_tick, int) or tick < deactivation_tick
    )


def _scene_restricted_regions(scene: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entities = {
        str(item.get("entity_id") or ""): item
        for item in scene.get("entities") or []
        if isinstance(item, Mapping) and item.get("entity_id")
    }
    return _restricted_regions(entities)


def _window_restricted_region_activity(
    scene: Mapping[str, Any], ticks: Sequence[int]
) -> dict[int, dict[str, bool]]:
    """Static per-region activity for window ticks; omits dynamic regions."""
    regions = _scene_restricted_regions(scene)
    by_tick: dict[int, dict[str, bool]] = {}
    for tick in ticks:
        row: dict[str, bool] = {}
        for region_id, region in regions.items():
            activity = _static_restricted_region_activity(region, tick)
            if activity is None:
                continue
            row[region_id] = activity
        by_tick[int(tick)] = row
    return by_tick


def _branch_owned_truth_frames(arm_root: Path | None) -> Path | None:
    if arm_root is None:
        return None
    for candidate in (Path(arm_root) / "raw" / "truth_frames.jsonl", Path(arm_root) / "truth_frames.jsonl"):
        if candidate.is_file():
            return candidate
    return None


def _restricted_activity_for_window(
    scene: Mapping[str, Any],
    ticks: Sequence[int],
    arm_root: Path | None = None,
) -> dict[int, dict[str, bool]]:
    """Prefer complete branch-owned truth_frames; never pad or use parent post-fork frames."""
    frames = _branch_owned_truth_frames(arm_root)
    if frames is not None:
        try:
            full = restricted_region_activity_by_tick(frames, _scene_restricted_regions(scene))
        except PredicateStateComputerError as exc:
            raise WindowSemanticsError(
                f"branch-owned truth_frames are not a complete activity authority: {exc}"
            ) from exc
        return {int(tick): dict(full[tick]) for tick in ticks if tick in full}
    return _window_restricted_region_activity(scene, ticks)


def _typed_uav_actor_id(
    actor_ids: Sequence[str],
    entity_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    typed = [
        entity_id
        for entity_id in actor_ids
        if _is_typed_uav(entity_metadata.get(entity_id))
    ]
    if len(set(typed)) == 1:
        return typed[0]
    scene_uavs = sorted(
        entity_id
        for entity_id, entity in entity_metadata.items()
        if _is_typed_uav(entity)
    )
    if len(scene_uavs) == 1:
        return scene_uavs[0]
    return ""


def _eval_restricted_airspace(
    *,
    tick: int,
    actor_ids: Sequence[str],
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    params: Mapping[str, Any],
    entities: Mapping[str, Mapping[str, Any]],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    binding_hint: Mapping[str, str] | None,
    region_activity: Mapping[str, bool] | None,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Governed pair: hinted actor+region; required fields then shared expression."""
    template = _restricted_template()
    hint = dict(binding_hint or {})
    hinted_actor = str(hint.get("actor") or "")
    hinted_region = str(hint.get("restricted_region") or "")
    actor_id = hinted_actor or _typed_uav_actor_id(actor_ids, entity_metadata)
    bindings: dict[str, str] = {}
    if actor_id:
        bindings["actor"] = actor_id
    if hinted_region:
        bindings["restricted_region"] = hinted_region
    if not actor_id:
        return (
            "unknown",
            _missing_observations("branch/arm_record.json", tick, "typed_aircraft_actor_id"),
            bindings,
        )
    row = rows.get((tick, actor_id))
    if not _is_typed_uav(entity_metadata.get(actor_id) or entities.get(actor_id)):
        return (
            "unknown",
            _missing_observations("branch/arm_record.json", tick, "typed_aircraft_actor_id"),
            bindings,
        )
    if not hinted_region:
        return (
            "unknown",
            _missing_observations("scene_setup.json", tick, "restricted_region"),
            bindings,
        )
    regions = _restricted_regions(entities)
    if hinted_region not in regions:
        return (
            "unknown",
            _missing_observations(
                "scene_setup.json", tick, f"restricted_region={hinted_region}"
            ),
            bindings,
        )
    region = regions[hinted_region]
    active: bool | None = None
    if region_activity is not None and hinted_region in region_activity:
        measured = region_activity[hinted_region]
        if type(measured) is not bool:
            active = None
        else:
            active = measured
    else:
        active = _static_restricted_region_activity(region, tick)
    distance: float | None = None
    if row is None or not _row_present(row):
        if _explicit_absent_row(row):
            return "out_of_scope", {"scope.present": False}, bindings
    else:
        pos = _vector3(row.get("pos_enu"))
        if pos is not None:
            measured_distance = _signed_distance_to_restricted_prism(pos, region)
            if type(measured_distance) in {int, float} and math.isfinite(measured_distance):
                distance = float(measured_distance)
    operands: dict[str, Any] = {}
    if type(active) is bool:
        operands["geometry.restricted_region_active"] = active
    if distance is not None:
        operands["geometry.minimum_restricted_boundary_distance_m"] = distance
    missing: list[str] = []
    observations: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    for field in template["state_contract"]["required_fields"]:
        path = str(field["field"])
        value = operands.get(path)
        if _missing_value(value) or path not in operands:
            missing.append(_missing_source_record(
                "branch_region_activity" if path.endswith("restricted_region_active") else "trajectories.jsonl",
                tick,
                path,
            ))
        else:
            observations[path] = value
            geometry[path.split(".", 1)[-1]] = value
    if missing:
        return "unknown", {"missing_source_record": missing}, bindings
    evaluated = _evaluate_expression(
        template["evaluation_spec"], {"geometry": geometry}, {}, params
    )
    value = _expression_truth_value(evaluated)
    return value, observations, bindings


def _entity_label(
    row: Mapping[str, Any] | None,
    scene_entity: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve a role label from runtime state or authoritative scene metadata."""

    for source in (row or {}, scene_entity or {}):
        for key in (
            "label_class",
            "ontology_class",
            "entity_kind",
            "entity_type",
            "entity_category",
            "category",
        ):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip().lower()
    return None


def _row_present(row: Mapping[str, Any] | None) -> bool:
    """Whether a reconstructed runtime row denotes a live entity."""

    return isinstance(row, Mapping) and row.get("present") is not False


def _explicit_absent_row(row: Mapping[str, Any] | None) -> bool:
    """True only for an authoritative per-tick tombstone (present is False).

    A missing row is not absence evidence. Presence at another tick in the
    window (including future ticks) must not be used to infer out_of_scope.
    """

    return isinstance(row, Mapping) and row.get("present") is False


def _typed_category(entity: Mapping[str, Any] | None) -> str:
    """Exact scene/roster category token. Never read label_class or id spelling."""

    if not isinstance(entity, Mapping):
        return ""
    for key in ("entity_category", "category"):
        value = entity.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return ""


def _is_typed_uav(entity: Mapping[str, Any] | None) -> bool:
    return _typed_category(entity) == "uav"


def _is_typed_vehicle(entity: Mapping[str, Any] | None) -> bool:
    return _typed_category(entity) == "vehicle"


def _is_typed_pedestrian(entity: Mapping[str, Any] | None) -> bool:
    return _typed_category(entity) == "pedestrian"


def _is_typed_structure(entity: Mapping[str, Any] | None) -> bool:
    if _typed_category(entity) in {"building", "structure", "facade_anchor"}:
        return True
    if not isinstance(entity, Mapping):
        return False
    kind = str(entity.get("entity_kind") or entity.get("entity_type") or "")
    return kind in {"facade_anchor", "building_anchor", "structure_anchor"}


def _is_landing_pad(entity: Mapping[str, Any] | None) -> bool:
    if not isinstance(entity, Mapping):
        return False
    scope = entity.get("semantic_scope")
    if (
        isinstance(scope, Mapping)
        and scope.get("scope_type") == "facility"
        and scope.get("scope_subtype") == "landing_pad"
    ):
        return True
    kind = str(entity.get("entity_kind") or entity.get("entity_type") or "")
    return kind == "facility.landing_pad"


def _is_typed_facility(entity: Mapping[str, Any] | None) -> bool:
    return _typed_category(entity) == "facility"


def _xy_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _digest(*parts: Any) -> str:
    return digest_object(parts).split(":", 1)[-1][:24]


def _project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_family_rules() -> dict[str, list[dict[str, Any]]]:
    registry = _json(REGISTRY_PATH)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in registry.get("event_occurrence_types") or []:
        if not isinstance(row, Mapping):
            continue
        family = str(row.get("event_family_id") or "")
        if family:
            by_family.setdefault(family, []).append(dict(row))
    return by_family


def family_rule(family: str, rule_id: str | None, predicate_id: str) -> dict[str, Any]:
    rules = load_family_rules().get(family) or []
    if rule_id:
        for row in rules:
            if (
                str(row.get("rule_id") or "") == rule_id
                and str(row.get("trigger_predicate_id") or "") == predicate_id
            ):
                return row
    for row in rules:
        if str(row.get("trigger_predicate_id") or "") == predicate_id:
            return row
    if len(rules) == 1:
        return rules[0]
    if not rules:
        raise WindowSemanticsError(f"no registry event rule for family {family}")
    return rules[0]


def capture_ticks(fork_tick: int, end_tick: int) -> list[int]:
    start = fork_tick if fork_tick % CAPTURE_STEP == 0 else fork_tick + (CAPTURE_STEP - fork_tick % CAPTURE_STEP)
    return list(range(start, end_tick + 1, CAPTURE_STEP))


def overlay_trajectories(
    parent_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    *,
    fork_tick: int,
    end_tick: int,
    lifecycle_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Overlay a compact branch delta on its immutable parent prefix.

    A delta stream can legitimately contain no trajectory row when an Arm
    suppresses a spawn/remove action.  In that case the absence itself is
    evidence and must not leave the parent entity visible in the reconstructed
    branch.  ``lifecycle_rows`` is produced by the real branch engine and
    carries only measured presence changes.  Tombstones are kept in-memory so
    predicate evaluators can return ``out_of_scope`` without putting a
    non-physical row into the persisted trajectory artifact.
    """
    merged: dict[tuple[int, str], dict[str, Any]] = {}
    for row in parent_rows:
        try:
            raw_tick = row.get("tick")
            if isinstance(raw_tick, bool) or raw_tick is None:
                continue
            tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        if fork_tick <= tick <= end_tick:
            entity_id = str(row.get("entity_id") or "")
            if entity_id:
                merged[(tick, entity_id)] = dict(row)
    for row in lifecycle_rows or ():
        try:
            raw_tick = row.get("tick")
            if isinstance(raw_tick, bool) or raw_tick is None:
                continue
            tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        if not (fork_tick <= tick <= end_tick):
            continue
        entity_id = str(row.get("entity_id") or "")
        if not entity_id:
            continue
        present = row.get("present")
        if present is False:
            # Authoritative absence: never leave a parent-filled pose for this
            # tick. Delta may later overwrite if the engine reintroduces the entity.
            merged[(tick, entity_id)] = {
                "tick": tick,
                "entity_id": entity_id,
                "present": False,
                "lifecycle_source": str(
                    row.get("source") or "engine_trajectory_presence_diff"
                ),
            }
    for row in delta_rows:
        try:
            raw_tick = row.get("tick")
            if isinstance(raw_tick, bool) or raw_tick is None:
                continue
            tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        if fork_tick <= tick <= end_tick:
            entity_id = str(row.get("entity_id") or "")
            if entity_id:
                merged[(tick, entity_id)] = dict(row)
    return merged


def overlay_weather(
    parent_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    *,
    fork_tick: int,
    end_tick: int,
) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in parent_rows:
        try:
            raw_tick = row.get("tick")
            if isinstance(raw_tick, bool) or raw_tick is None:
                continue
            tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        if fork_tick <= tick <= end_tick:
            merged[tick] = dict(row)
    for row in delta_rows:
        try:
            raw_tick = row.get("tick")
            if isinstance(raw_tick, bool) or raw_tick is None:
                continue
            tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        if fork_tick <= tick <= end_tick:
            merged[tick] = dict(row)
    return merged


def resolve_actor_ids(
    group: Mapping[str, Any],
    arm_record: Mapping[str, Any],
    script: Mapping[str, Any] | None = None,
) -> list[str]:
    ids: list[str] = []
    arm = arm_record.get("decision_arm") or {}
    for step in list(arm.get("mechanism_steps") or []):
        if not isinstance(step, Mapping):
            continue
        for key in ("actor_id", "entity_id"):
            value = str(step.get(key) or "")
            if value and value not in ids:
                ids.append(value)
    if script:
        anchor_id = str(group.get("anchor_event_id") or "")
        for event in script.get("events") or []:
            if not isinstance(event, Mapping) or str(event.get("event_id") or "") != anchor_id:
                continue
            for action in event.get("actions") or []:
                if not isinstance(action, Mapping):
                    continue
                value = str(action.get("entity_id") or "")
                if value and value not in ids:
                    ids.append(value)
    return ids


def _scene_pads(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    pads: list[dict[str, Any]] = []
    for entity in scene.get("entities") or []:
        if not isinstance(entity, Mapping):
            continue
        if not _is_landing_pad(entity):
            continue
        placement = entity.get("placement")
        pos = _vector3(
            entity.get("position_enu_m")
            or entity.get("pos_enu")
            or entity.get("location_enu_m")
            or (placement.get("position_enu_m") if isinstance(placement, Mapping) else None)
            or (placement.get("resolved_position_enu_m") if isinstance(placement, Mapping) else None)
        )
        if pos is not None:
            pads.append({"entity_id": str(entity.get("entity_id") or ""), "pos_enu": pos})
    return pads


def _ground_z(
    actor_id: str,
    tick: int,
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    pads: Sequence[Mapping[str, Any]],
    fork_tick: int,
) -> float | None:
    if pads:
        row = rows.get((tick, actor_id))
        pos = _vector3((row or {}).get("pos_enu"))
        if pos is not None:
            nearest = min(pads, key=lambda pad: _xy_distance(pos, pad["pos_enu"]))
            return float(nearest["pos_enu"][2])
    return None


def _assigned_altitude_m(row: Mapping[str, Any], scene_entity: Mapping[str, Any] | None) -> float | None:
    for source in (row, scene_entity or {}):
        for key in ("assigned_altitude_m", "cruise_altitude_m", "target_altitude_m"):
            number = _number(source.get(key))
            if number is not None:
                return number
        for nested_key in (
            "mission_state",
            "navigation_state",
            "initial_state",
            "lifecycle",
            "uav_corridor",
        ):
            nested = source.get(nested_key)
            if isinstance(nested, Mapping):
                for key in ("assigned_altitude_m", "cruise_altitude_m", "target_altitude_m"):
                    number = _number(nested.get(key))
                    if number is not None:
                        return number
    return None


def _script_proximity_pairs(
    script: Mapping[str, Any] | None,
    *,
    entity_metadata: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str, str | None, str | None, Mapping[str, Any]]]:
    """Return typed, authored proximity pairs from the scenario script."""

    result: list[tuple[str, str, str | None, str | None, Mapping[str, Any]]] = []
    if not isinstance(script, Mapping):
        return result
    for trigger in script.get("triggers") or []:
        if not isinstance(trigger, Mapping) or trigger.get("type") != "entity_proximity":
            continue
        first = str(trigger.get("entity_a") or "")
        second = str(trigger.get("entity_b") or "")
        if not first or not second:
            continue
        result.append((first, second, entity_metadata.get(first), entity_metadata.get(second), trigger))
    return result


def _resolve_script_pair(
    pairs: Sequence[tuple[str, str, Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any]]],
    *,
    actor_ids: Sequence[str],
    first_role: str,
    second_role: str,
) -> tuple[str, str] | None:
    """Resolve one exact typed pair, requiring authored actor overlap."""

    actor_set = set(map(str, actor_ids))

    def matches(entity: Mapping[str, Any] | None, role: str) -> bool:
        return (
            _is_typed_uav(entity)
            if role == "uav"
            else _is_typed_pedestrian(entity)
            if role == "pedestrian"
            else _is_typed_vehicle(entity)
            if role == "vehicle"
            else _is_typed_structure(entity)
            if role == "structure"
            else False
        )

    candidates: set[tuple[str, str]] = set()
    for first, second, first_entity, second_entity, _ in pairs:
        if actor_set and actor_set.isdisjoint({first, second}):
            continue
        if matches(first_entity, first_role) and matches(second_entity, second_role):
            candidates.add(
                tuple(sorted((first, second)))
                if first_role == second_role
                else (first, second)
            )
        if first_role != second_role and matches(second_entity, first_role) and matches(first_entity, second_role):
            candidates.add((second, first))
    return next(iter(candidates)) if len(candidates) == 1 else None


def eval_geometry(
    predicate_id: str,
    *,
    tick: int,
    actor_ids: Sequence[str],
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    scene: Mapping[str, Any],
    params: Mapping[str, Any],
    fork_tick: int,
    script: Mapping[str, Any] | None = None,
    binding_hint: Mapping[str, str] | None = None,
    local_ground_by_tick: Mapping[int, Mapping[str, float]] | None = None,
    region_activity: Mapping[str, bool] | None = None,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    pads = _scene_pads(scene)
    entities = {
        str(item.get("entity_id") or ""): item
        for item in scene.get("entities") or []
        if isinstance(item, Mapping) and item.get("entity_id")
    }
    entity_metadata: dict[str, Mapping[str, Any]] = dict(entities)
    for (sample_tick, entity_id), row in sorted(rows.items()):
        if entity_id in entity_metadata or not _row_present(row):
            continue
        if _typed_category(row):
            entity_metadata[entity_id] = row
    script_pairs = _script_proximity_pairs(
        script, entity_metadata=entity_metadata
    )
    observations: dict[str, Any] = {}
    bindings: dict[str, str] = dict(binding_hint or {})
    if predicate_id == RESTRICTED_PREDICATE:
        return _eval_restricted_airspace(
            tick=tick,
            actor_ids=actor_ids,
            rows=rows,
            params=params,
            entities=entities,
            entity_metadata=entity_metadata,
            binding_hint=binding_hint,
            region_activity=region_activity,
        )
    if predicate_id in {
        "airspace.aircraft_airborne",
        "airspace.aircraft_touchdown",
        "airspace.aircraft_at_home_pad",
        "airspace.aircraft_in_landing_zone",
        "airspace.aircraft_outside_assigned_altitude_band",
    }:
        typed_actor_ids = [
            entity_id
            for entity_id in actor_ids
            if _is_typed_uav(entity_metadata.get(entity_id) or entities.get(entity_id))
        ]
        actor_id = typed_actor_ids[0] if len(set(typed_actor_ids)) == 1 else ""
        if not actor_id:
            scene_uavs = sorted(
                entity_id
                for entity_id, entity in entity_metadata.items()
                if _is_typed_uav(entity)
            )
            if len(scene_uavs) == 1:
                actor_id = scene_uavs[0]
        if not actor_id:
            return "unknown", _missing_observations("branch/arm_record.json", tick, "typed_aircraft_actor_id"), {}
        aircraft_role = "actor" if predicate_id == "airspace.aircraft_outside_assigned_altitude_band" else "aircraft"
        bindings = {aircraft_role: actor_id}
        row = rows.get((tick, actor_id))
        if row is None or not _row_present(row):
            if _explicit_absent_row(row):
                return "out_of_scope", {"scope.present": False}, bindings
            return "unknown", _missing_observations("trajectories.jsonl", tick, f"entity_id={actor_id}"), bindings
        if not _is_typed_uav(entities.get(actor_id) or entity_metadata.get(actor_id)):
            return "unknown", _missing_observations("scene_setup.json", tick, f"entity_id={actor_id}:entity_category"), bindings
        pos = _vector3(row.get("pos_enu"))
        if pos is None:
            return "unknown", _missing_observations("trajectories.jsonl", tick, f"entity_id={actor_id}:pos_enu"), bindings
        if predicate_id == "airspace.aircraft_outside_assigned_altitude_band":
            assigned = _assigned_altitude_m(row, entities.get(actor_id))
            if assigned is None:
                return "unknown", _missing_observations("trajectories.jsonl", tick, f"entity_id={actor_id}:assigned_altitude_m"), bindings
            delta = abs(float(pos[2]) - assigned)
            observations = {"geometry.position_z_m": float(pos[2]), "geometry.assigned_altitude_m": assigned,
                            "geometry.altitude_error_m": delta}
            return ("true" if delta > float(params["altitude_tolerance_m"]) else "false"), observations, bindings
        ground_z = (
            (local_ground_by_tick or {}).get(tick, {}).get(actor_id)
            if local_ground_by_tick is not None
            else None
        )
        if ground_z is None:
            ground_z = _ground_z(actor_id, tick, rows, pads, fork_tick)
        if ground_z is None:
            return "unknown", _missing_observations("trajectories.jsonl", tick, f"entity_id={actor_id}:ground_reference_z_m"), bindings
        z_agl = float(pos[2]) - ground_z
        speed = _speed(row)
        observations = {
            "geometry.position_z_m": float(pos[2]),
            "geometry.local_ground_reference_z_m": ground_z,
            "geometry.z_agl_m": z_agl,
            "geometry.speed_mps": speed,
        }
        if predicate_id == "airspace.aircraft_airborne":
            return ("true" if z_agl > float(params["airborne_threshold_m"]) else "false"), observations, bindings
        if predicate_id == "airspace.aircraft_touchdown":
            if speed is None:
                return "unknown", _missing_observations("trajectories.jsonl", tick, f"entity_id={actor_id}:vel_mps"), bindings
            value = (
                z_agl <= float(params["touchdown_threshold_m"])
                and speed <= float(params["aircraft_stationary_speed_threshold_mps"])
            )
            return ("true" if value else "false"), observations, bindings
        if predicate_id == "airspace.aircraft_at_home_pad":
            if not pads:
                return "unknown", _missing_observations("scene_setup.json", tick, "landing_pad"), bindings
            nearest = min(pads, key=lambda pad: _xy_distance(pos, pad["pos_enu"]))
            dist = _xy_distance(pos, nearest["pos_enu"])
            observations["geometry.xy_distance_to_home_pad_m"] = dist
            observations["geometry.home_pad_id"] = nearest["entity_id"]
            bindings["home_pad"] = nearest["entity_id"]
            return ("true" if dist <= float(params["home_pad_radius_m"]) else "false"), observations, bindings
        if predicate_id == "airspace.aircraft_in_landing_zone":
            if not pads:
                return "unknown", _missing_observations("scene_setup.json", tick, "landing_zone"), bindings
            nearest = min(pads, key=lambda pad: _xy_distance(pos, pad["pos_enu"]))
            dist = _xy_distance(pos, nearest["pos_enu"])
            observations["geometry.xy_distance_to_landing_zone_m"] = dist
            bindings["landing_zone"] = nearest["entity_id"]
            return ("true" if dist <= float(params["landing_zone_radius_m"]) else "false"), observations, bindings
    pair_contracts = {
        "safety.aircraft_pair_below_separation": ("first_aircraft", "second_aircraft", _is_typed_uav, _is_typed_uav, "separation_margin_m"),
        "safety.aircraft_near_pedestrian": ("aircraft", "pedestrian", _is_typed_uav, _is_typed_pedestrian, "safety_near_pedestrian_m"),
        "safety.aircraft_near_ground_vehicle": ("aircraft", "vehicle", _is_typed_uav, _is_typed_vehicle, "safety_near_vehicle_m"),
        "safety.aircraft_near_building_structure": ("aircraft", "structure", _is_typed_uav, _is_typed_structure, "safety_near_structure_m"),
        "safety.vehicle_pedestrian_conflict": ("vehicle", "pedestrian", _is_typed_vehicle, _is_typed_pedestrian, "vehicle_pedestrian_conflict_m"),
    }
    if predicate_id in pair_contracts:
        first_role, second_role, first_type, second_type, threshold = pair_contracts[predicate_id]
        if set(bindings) != {first_role, second_role}:
            return "unknown", _missing_observations("fork_causal_context", tick, "complete_typed_pair"), bindings
        for role, type_check in ((first_role, first_type), (second_role, second_type)):
            if not type_check(entities.get(bindings[role]) or entity_metadata.get(bindings[role])):
                return "unknown", _missing_observations("scene_setup.json", tick, role + ":ontology_class"), bindings
        absent = [identity for identity in bindings.values() if _explicit_absent_row(rows.get((tick, identity)))]
        if absent:
            return "out_of_scope", {"scope.present": False, "scope.absent_entity_ids": absent}, bindings
        points = []
        for role, type_check in ((first_role, first_type), (second_role, second_type)):
            identity = bindings[role]
            declared = entities.get(identity) or entity_metadata.get(identity)
            current = rows.get((tick, identity))
            if current is None and role == "structure":
                # A declared static building's placement is time-independent.
                placement = declared.get("placement", {})
                position = _vector3(declared.get("position_enu_m") or placement.get("resolved_position_enu_m") or placement.get("position_enu_m"))
            else:
                position = _vector3(current.get("pos_enu")) if isinstance(current, Mapping) else None
            if position is None:
                return "unknown", _missing_observations("trajectories.jsonl", tick, role + ":pos_enu"), bindings
            points.append(position)
        first, second = points
        horizontal = _xy_distance(first, second)
        vertical = abs(first[2]-second[2])
        distance = math.hypot(horizontal, vertical)
        margin = float(params[threshold])
        observations = {"geometry.horizontal_distance_m": horizontal, "geometry.vertical_distance_m": vertical}
        if predicate_id == "safety.aircraft_pair_below_separation":
            observations.update({"geometry.pair_distance_m": distance,
                                 "geometry.separation_margin_m": margin,
                                 "geometry.separation_model": "euclidean_3d"})
        elif predicate_id == "safety.vehicle_pedestrian_conflict":
            distance = horizontal
            observations["derived.nearest_pedestrian_distance_m"] = distance
        else:
            observations["geometry.nearest_distance_m"] = distance
        return ("true" if distance < margin else "false"), observations, bindings
    if predicate_id == "facility.landing_pad_contention":
        if not pads:
            return "unknown", _missing_observations("scene_setup.json", tick, "landing_pad"), {}
        occupants: dict[str, list[str]] = {}
        missing_labels: list[str] = []
        for (sample_tick, entity_id), row in rows.items():
            if sample_tick != tick or not _row_present(row):
                continue
            if not _is_typed_uav(entities.get(entity_id) or entity_metadata.get(entity_id)):
                if not _typed_category(entities.get(entity_id) or entity_metadata.get(entity_id)):
                    missing_labels.append(entity_id)
                continue
            pos = _vector3(row.get("pos_enu"))
            if pos is None:
                return "unknown", _missing_observations("trajectories.jsonl", tick, f"entity_id={entity_id}:pos_enu"), {}
            nearest = min(pads, key=lambda pad: _xy_distance(pos, pad["pos_enu"]))
            if _xy_distance(pos, nearest["pos_enu"]) <= float(params["home_pad_radius_m"]):
                occupants.setdefault(nearest["entity_id"], []).append(entity_id)
        contended = next((pad_id for pad_id, uavs in occupants.items() if len(uavs) >= 2), None)
        if contended is None:
            if missing_labels:
                return "unknown", _missing_observations(
                    "trajectories.jsonl",
                    tick,
                    *(f"entity_id={item}:label_class" for item in missing_labels),
                ), {}
            return "false", {"occupants": occupants}, {}
        uavs = occupants[contended]
        bindings = {"pad": contended, "first_aircraft": uavs[0], "second_aircraft": uavs[1]}
        return "true", {"occupants": occupants}, bindings
    if predicate_id == "airspace.airspace_corridor_over_capacity":
        separation_margin = float(params["separation_margin_m"])
        corridors = _airspace_corridors(entities, separation_margin)
        active = {
            corridor_id: corridor
            for corridor_id, corridor in corridors.items()
            if _corridor_active_at_tick(corridor, tick)
        }
        if not active:
            return "false", {
                "scene.maximum_corridor_occupancy_count": 0,
                "scene.maximum_corridor_capacity": 0,
            }, {}
        uav_positions = {}
        for (sample_tick, entity_id), row in rows.items():
            if sample_tick != tick or not _row_present(row):
                continue
            scene_entity = entities.get(entity_id) or {}
            if str(scene_entity.get("category") or "") != "uav":
                continue
            position = _vector3(row.get("pos_enu"))
            if position is None:
                continue
            uav_positions[entity_id] = position
        occupancy = {
            corridor_id: sum(
                _point_in_oriented_corridor_box(position, corridor)
                for position in uav_positions.values()
            )
            for corridor_id, corridor in active.items()
        }
        selected_id = max(
            active,
            key=lambda corridor_id: (
                occupancy[corridor_id] - int(active[corridor_id]["capacity"]),
                occupancy[corridor_id],
                corridor_id,
            ),
        )
        selected = active[selected_id]
        count = occupancy[selected_id]
        capacity = int(selected["capacity"])
        observations = {
            "scene.maximum_corridor_occupancy_count": count,
            "scene.maximum_corridor_capacity": capacity,
            "scene.corridor_occupancy_by_id": dict(sorted(occupancy.items())),
        }
        return (
            "true" if count > capacity else "false",
            observations,
            {"corridor": selected_id},
        )
    return "unknown", _missing_observations("predicate_registry", tick, f"unsupported:{predicate_id}"), {}


def _strict_weather_scalar(value: Any) -> float | None:
    if type(value) in (int, float) and type(value) is not bool and math.isfinite(value):
        return float(value)
    return None


def eval_weather(
    predicate_id: str,
    weather: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    tick: int | None = None,
) -> tuple[str, dict[str, Any]]:
    rain = _strict_weather_scalar(weather.get("rain"))
    fog = _strict_weather_scalar(weather.get("fog_density"))
    wind = _strict_weather_scalar(weather.get("wind_speed"))
    vis = _strict_weather_scalar(weather.get("visibility_m"))
    lux = _strict_weather_scalar(weather.get("illumination_lux"))
    temp = _strict_weather_scalar(weather.get("temperature_c"))
    observations = {
        "weather.rain": rain,
        "weather.fog_density": fog,
        "weather.wind_speed": wind,
        "weather.visibility_m": vis,
        "weather.illumination_lux": lux,
        "weather.temperature_c": temp,
    }
    if predicate_id == "environment.rain_active":
        if rain is None:
            observations.update(_missing_observations("weather_meta.jsonl", tick, "weather.rain"))
            return "unknown", observations
        return ("true" if rain > float(params["rain_active_threshold"]) else "false"), observations
    if predicate_id == "environment.fog_active":
        if fog is None:
            observations.update(_missing_observations("weather_meta.jsonl", tick, "weather.fog_density"))
            return "unknown", observations
        return ("true" if fog > float(params["fog_density_threshold"]) else "false"), observations
    if predicate_id == "environment.wind_speed_above_threshold":
        if wind is None:
            observations.update(_missing_observations("weather_meta.jsonl", tick, "weather.wind_speed"))
            return "unknown", observations
        return ("true" if wind > float(params["wind_speed_threshold_mps"]) else "false"), observations
    if predicate_id == "environment.visibility_below_threshold":
        if vis is None:
            observations.update(_missing_observations("weather_meta.jsonl", tick, "weather.visibility_m"))
            return "unknown", observations
        return ("true" if vis < float(params["visibility_threshold_m"]) else "false"), observations
    if predicate_id == "environment.low_illumination_active":
        if lux is None:
            observations.update(_missing_observations("weather_meta.jsonl", tick, "weather.illumination_lux"))
            return "unknown", observations
        return ("true" if lux < float(params["low_illumination_threshold_lux"]) else "false"), observations
    if predicate_id == "environment.temperature_outside_operating_range":
        if temp is None:
            observations.update(_missing_observations("weather_meta.jsonl", tick, "weather.temperature_c"))
            return "unknown", observations
        outside = temp < float(params["temperature_min_c"]) or temp > float(params["temperature_max_c"])
        return ("true" if outside else "false"), observations
    observations.update(_missing_observations("predicate_registry", tick, f"unsupported:{predicate_id}"))
    return "unknown", observations


def _nested_get(row: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        cursor: Any = row
        ok = True
        for part in path.split("."):
            if isinstance(cursor, Mapping) and part in cursor:
                cursor = cursor[part]
            else:
                ok = False
                break
        if ok:
            return cursor
    return None


def _nested_get_with_path(
    row: Mapping[str, Any], *paths: str
) -> tuple[Any, str | None]:
    for path in paths:
        if path in row:
            return row[path], path
        cursor: Any = row
        ok = True
        for part in path.split("."):
            if isinstance(cursor, Mapping) and part in cursor:
                cursor = cursor[part]
            else:
                ok = False
                break
        if ok:
            return cursor, path
    return None, None


# Governed controller_health codes for traffic signals (template equals all_red_fault).
_SIGNAL_CONTROLLER_HEALTH_TRUE = frozenset({"all_red_fault"})
_SIGNAL_CONTROLLER_HEALTH_FALSE = frozenset({"nominal", "sumo_cycle", "healthy"})
_SIGNAL_CONTROLLER_HEALTH_KNOWN = (
    _SIGNAL_CONTROLLER_HEALTH_TRUE | _SIGNAL_CONTROLLER_HEALTH_FALSE
)


def _normalize_signal_controller_health(raw: Any) -> str | None:
    """Map a candidate raw value to a known controller_health code, else None.

    Boolean facility_state.fault is never a controller_health code.
    """

    if isinstance(raw, bool):
        return None
    if not isinstance(raw, str):
        return None
    text = raw
    return text if text in _SIGNAL_CONTROLLER_HEALTH_KNOWN else None


def resolve_signal_controller_health(
    row: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str]:
    """Resolve governed controller_health from a typed signal runtime row.

    Authority order (string health codes only):
    1. domain.signal_queue.controller_health
    2. state (when it is a known health code such as all_red_fault)
    3. activity_type (when it is a known health code)

    Ignored: facility_state.fault (bool), semantic_role (e.g. green_cycle).
    Conflicting authorities => (None, None, \"conflict\").
    Absent => (None, None, \"missing\").
    """

    if not isinstance(row, Mapping):
        return None, None, "missing"
    authorities: list[tuple[str, str]] = []
    nested, nested_path = _nested_get_with_path(row, "domain.signal_queue.controller_health")
    code = _normalize_signal_controller_health(nested)
    if nested_path is not None and code is None:
        return None, nested_path, "conflict"
    if code is not None:
        authorities.append(("domain.signal_queue.controller_health", code))
    for key in ("state", "activity_type"):
        code = _normalize_signal_controller_health(row.get(key))
        if code is not None:
            authorities.append((key, code))
    if not authorities:
        return None, None, "missing"
    # Conflict only when authorities disagree on truth polarity (true vs false),
    # not when multiple normal codes (nominal vs sumo_cycle) both evaluate false.
    polarities = {
        evaluate_signal_controller_health_code(code) for _, code in authorities
    }
    polarities.discard(None)
    if len(polarities) > 1:
        return None, None, "conflict"
    # Prefer explicit domain.controller_health, else first health-coded field.
    path, code = authorities[0]
    return code, path, "ok"


def evaluate_signal_controller_health_code(code: str | None) -> str | None:
    """Equals template: all_red_fault => true; known normal codes => false; else unknown."""

    if code is None:
        return None
    if code in _SIGNAL_CONTROLLER_HEALTH_TRUE:
        return "true"
    if code in _SIGNAL_CONTROLLER_HEALTH_FALSE:
        return "false"
    return None


@functools.lru_cache(maxsize=256)
def _parent_gnss_error_before_fork(
    episode_id: str, actor_id: str, fork_tick: int
) -> tuple[list[float] | None, int | None]:
    path = DOMAIN_SUPPLEMENT_ROOT / episode_id / "domain_state_observations.jsonl"
    if not path.is_file():
        return None, None
    selected: tuple[int, list[float]] | None = None
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tick = int(row.get("tick", -1))
            if tick >= fork_tick:
                break
            if (
                row.get("observation_family") != "gnss_navigation"
                or str(row.get("subject_id") or "") != actor_id
            ):
                continue
            values = row.get("values")
            error = _vector3(
                values.get("drift_offset_enu_m")
                if isinstance(values, Mapping)
                else None
            )
            if error is not None and (selected is None or tick > selected[0]):
                selected = (tick, error)
    return (selected[1], selected[0]) if selected is not None else (None, None)


@functools.lru_cache(maxsize=256)
def _parent_domain_observations(episode_id: str) -> tuple[dict[tuple[int, str], dict[str, Any]], ...]:
    """Load measured parent domain rows as an immutable splice base.

    These rows are generated by the governed domain-state simulator from the
    parent L0 truth.  They are usable only when the Arm has no measured state
    or lifecycle delta; an Arm that changes runtime/geometry state must be
    re-evaluated from its own evidence and remains unresolved if the required
    field is absent.
    """

    path = DOMAIN_SUPPLEMENT_ROOT / episode_id / "domain_state_observations.jsonl"
    if not path.is_file():
        return ({},)
    result: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    continue
                tick = row.get("tick")
                family = str(row.get("observation_family") or "")
                subject = str(row.get("subject_id") or "")
                if isinstance(tick, bool) or tick is None or not family or not subject:
                    continue
                try:
                    result[(int(tick), f"{family}:{subject}")] = dict(row)
                except (TypeError, ValueError):
                    continue
    except (OSError, json.JSONDecodeError):
        return ({},)
    return (result,)


def eval_parent_domain(
    predicate_id: str,
    *,
    tick: int,
    actor_ids: Sequence[str],
    episode_id: str,
    binding_hint: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, str]] | None:
    """Return an unchanged-parent domain observation, if one is authoritative."""

    family_by_predicate = {
        "positioning.gnss_quality_degraded": "gnss_navigation",
        "positioning.gnss_spoofing_active": "gnss_navigation",
        "security.communication_jamming_impact_active": "security_command",
        "security.ground_control_station_compromised": "security_command",
        "communication.link_unavailable": "communication_link",
        "communication.link_degraded": "communication_link",
        "facility.charging_station_unavailable": "pad_facility",
        "ground.traffic_signal_malfunction_active": "signal_queue",
        # Grounding_spec candidate_authority is road_segment_state (RoadSegment),
        # not the road_closure/road_network aggregate.
        "ground.road_segment_closed": "road_segment_state",
        "safety.crowd_evacuation_active": "crowd_evacuation",
    }
    family = family_by_predicate.get(predicate_id)
    if family is None:
        return None
    bindings = dict(binding_hint or {})
    subject = ""
    if predicate_id in {
        "positioning.gnss_quality_degraded",
        "positioning.gnss_spoofing_active",
    }:
        subject = str(bindings.get("actor") or (actor_ids[0] if actor_ids else ""))
        if subject:
            bindings.setdefault("actor", subject)
    elif predicate_id in {
        "security.ground_control_station_compromised",
        "facility.charging_station_unavailable",
        "ground.traffic_signal_malfunction_active",
    }:
        subject = str(
            bindings.get("station")
            or bindings.get("signal")
            or (actor_ids[0] if actor_ids else "")
        )
        if subject:
            bindings.setdefault(
                "station" if predicate_id != "ground.traffic_signal_malfunction_active" else "signal",
                subject,
            )
    elif predicate_id == "ground.road_segment_closed":
        # Role class RoadSegment — road_network aggregate is not role-valid.
        subject = str(bindings.get("road_segment") or "")
        if subject == "road_network":
            return None
        if subject:
            bindings["road_segment"] = subject
    elif predicate_id == "safety.crowd_evacuation_active":
        subject = "pedestrian_crowd"
        bindings.setdefault("crowd", subject)
    else:
        subject = str(bindings.get("actor") or (actor_ids[0] if actor_ids else ""))
    if not subject:
        return None
    rows = _parent_domain_observations(episode_id)[0]
    row = rows.get((tick, f"{family}:{subject}"))
    # Crowd identifiers are generated from the measured cohort and therefore
    # are not stable literals such as ``pedestrian_crowd``.  If the expected
    # subject is absent, accept a unique measured crowd row for this tick;
    # multiple competing rows remain unresolved rather than being guessed.
    if row is None and family == "crowd_evacuation":
        candidates = [
            value
            for (row_tick, row_key), value in rows.items()
            if row_tick == tick
            and row_key.startswith(f"{family}:")
            and isinstance(value.get("values"), Mapping)
            and "evacuation_active" in value["values"]
        ]
        if len(candidates) == 1:
            row = candidates[0]
            subject = str(row.get("subject_id") or subject)
            bindings["crowd"] = subject
    # Signal domain rows are often keyed by traffic_signal_system while the
    # branch binds a typed traffic_light id. Accept a unique covering row.
    if row is None and family == "signal_queue" and subject:
        candidates = []
        for (row_tick, row_key), value in rows.items():
            if row_tick != tick or not str(row_key).startswith(f"{family}:"):
                continue
            if not isinstance(value.get("values"), Mapping):
                continue
            vals = value["values"]
            if "controller_health" not in vals:
                continue
            ids = {str(item) for item in (vals.get("traffic_signal_ids") or []) if str(item)}
            if subject in ids or str(value.get("subject_id") or "") == subject:
                candidates.append(value)
        if len(candidates) == 1:
            row = candidates[0]
        elif len(candidates) > 1:
            return None
    if row is None:
        return None
    values = row.get("values")
    if not isinstance(values, Mapping):
        return None
    field_by_predicate = {
        "positioning.gnss_quality_degraded": ("quality_level", {"degraded", "spoofing_suspect"}),
        "positioning.gnss_spoofing_active": ("gnss_spoofed", {True, "true", "spoofed"}),
        "security.communication_jamming_impact_active": ("jamming_indicator", {True, "true", "jammed", "jamming"}),
        "security.ground_control_station_compromised": ("auth_state", {"compromised", "unauthenticated", "lockout"}),
        "communication.link_unavailable": ("link_unavailable", {True, "true", "unavailable", "down"}),
        "communication.link_degraded": ("link_degraded", {True, "true", "degraded", "lossy"}),
        # Governed template: equals availability == "unavailable" (not fault/failed).
        "facility.charging_station_unavailable": ("availability", {"unavailable"}),
        # Governed template: equals controller_health == "all_red_fault".
        "ground.traffic_signal_malfunction_active": ("controller_health", {"all_red_fault"}),
        "ground.road_segment_closed": ("road_closed", {True, "true"}),
        "safety.crowd_evacuation_active": ("evacuation_active", {True, "true"}),
    }
    field, active_values = field_by_predicate[predicate_id]

    raw = values.get(field)
    if raw is None:
        return None
    equals_predicates = {
        "facility.charging_station_unavailable",
        "ground.traffic_signal_malfunction_active",
    }
    if predicate_id in equals_predicates:
        # Governed equals templates: only exact string members are active.
        if not isinstance(raw, str):
            return None
        text = raw.strip().lower()
        active = text in {str(item).lower() for item in active_values}
        known = text in {
            "available",
            "unavailable",
            "reserved",
            "fault",
            "failed",
            "nominal",
            "sumo_cycle",
            "all_red_fault",
            "healthy",
        }
        if not known:
            return None
    elif isinstance(raw, bool):
        active = raw
        known = True
    elif isinstance(raw, str):
        active = raw.strip().lower() in {str(item).lower() for item in active_values}
        known = raw.strip().lower() in {
            "true", "false", "nominal", "normal", "good", "degraded", "spoofing_suspect",
            "spoofed", "available", "unavailable", "fault", "failed", "all_red_fault",
            "up", "down", "online", "offline", "clear", "compromised", "authenticated",
            "unauthenticated", "lockout", "jammed", "jamming", "lossy", "recovered",
        }
        if not known:
            return None
    else:
        try:
            active = raw in active_values
        except TypeError:
            active = False
    observations = {
        f"parent.domain.{family}.{field}": raw,
        "parent.domain.source_class": row.get("source_class"),
        "parent.domain.observation_id": row.get("observation_id"),
        "parent.domain.source_refs": list(row.get("source_refs") or []),
    }
    return ("true" if active else "false"), observations, bindings


@functools.lru_cache(maxsize=256)
def _parent_compute_comm_rows(episode_id: str) -> tuple[dict[tuple[int, str], dict[str, Any]], ...]:
    """Index the measured parent communication simulator output by tick/actor."""

    path = COMPUTE_COMM_ROOT / episode_id / "communication_state.jsonl"
    if not path.is_file():
        return ({},)
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    continue
                tick = value.get("tick")
                actor = str(value.get("entity_id") or "")
                if isinstance(tick, bool) or tick is None or not actor:
                    continue
                try:
                    rows[(int(tick), actor)] = dict(value)
                except (TypeError, ValueError):
                    continue
    except (OSError, json.JSONDecodeError):
        return ({},)
    return (rows,)


def eval_parent_compute_comm(
    predicate_id: str,
    *,
    tick: int,
    actor_ids: Sequence[str],
    episode_id: str,
    binding_hint: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, str]] | None:
    """Evaluate a link predicate from the governed parent communication rows."""

    if predicate_id not in {"communication.link_unavailable", "communication.link_degraded"}:
        return None
    bindings = dict(binding_hint or {})
    hinted_actor = str(bindings.get("actor") or "")
    hinted_station = str(bindings.get("station") or "")
    indexed = _parent_compute_comm_rows(episode_id)[0]
    # Candidate mechanism steps sometimes list the station before the UAV.
    # Resolve the actor from measured communication rows rather than relying
    # on list order; a row is accepted only when its station agrees with an
    # already-bound station.
    actor_candidates = [hinted_actor] if hinted_actor else []
    actor_candidates.extend(
        str(value) for value in actor_ids
        if str(value) and str(value) not in actor_candidates
    )
    actor = ""
    row: Mapping[str, Any] | None = None
    for candidate in actor_candidates:
        candidate_row = indexed.get((tick, candidate))
        if candidate_row is None:
            continue
        station_candidate = str(candidate_row.get("station_id") or "")
        if hinted_station and station_candidate and hinted_station != station_candidate:
            continue
        if not isinstance(candidate_row.get("link_quality"), Mapping):
            continue
        actor = candidate
        row = candidate_row
        break
    if not actor or row is None:
        return None
    station = str(row.get("station_id") or "")
    if station:
        bindings["station"] = station
    bindings["actor"] = actor
    requirement = row.get("requirement")
    if isinstance(requirement, Mapping) and requirement.get("status") == "source_required":
        return (
            "unknown",
            _missing_observations(
                "aw_data/compute_comm_supplement/communication_state.jsonl",
                tick,
                "required_communication_source",
            ),
            bindings,
        )
    quality = row.get("link_quality")
    if not isinstance(quality, Mapping):
        return None
    level = quality.get("quality_level")
    if not isinstance(level, str):
        return None
    level = level.strip().lower()
    if predicate_id == "communication.link_degraded":
        if level == "poor":
            value = "true"
        elif level in {"excellent", "good", "fair", "down"}:
            value = "false"
        else:
            return None
    else:
        if level == "down":
            value = "true"
        elif level in {"excellent", "good", "fair", "poor"}:
            value = "false"
        else:
            return None
    observations = {
        "parent.compute_comm.quality_level": level,
        "parent.compute_comm.link_quality": dict(quality),
        "parent.compute_comm.station_id": station,
        "parent.compute_comm.source_class": row.get("source_class"),
        "parent.compute_comm.source_refs": list(row.get("source_refs") or []),
        "parent.compute_comm.rule_id": row.get("rule_id"),
    }
    return value, observations, bindings


@functools.lru_cache(maxsize=210)
def _parent_domain_seed_digest(episode_id: str) -> str:
    parent_root = CAPTURE_ROOT / episode_id
    if not parent_root.is_dir():
        raise FileNotFoundError(f"GNSS parent capture episode missing: {parent_root}")
    communication_rows = build_communication_state_rows(
        parent_root.resolve(), COMPUTE_COMM_PROFILE_PATH
    )
    return domain_seed_digest(
        parent_root.resolve(), DOMAIN_PROFILE_PATH,
        communication_rows=communication_rows,
    )


def precompute_gnss_quality(
    *,
    episode_id: str,
    actor_id: str,
    ticks: Sequence[int],
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    scene: Mapping[str, Any],
    binding_hint: Mapping[str, str],
    fork_tick: int,
) -> dict[int, tuple[str, dict[str, Any], dict[str, str]]]:
    """Run the governed GNSS FSM over the local branch capture grid."""

    profile = load_domain_state_profile(DOMAIN_PROFILE_PATH)
    params = profile["domain_models"]["gnss"]
    manifest_path = DOMAIN_SUPPLEMENT_ROOT / episode_id / "domain_state_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != "1.1.0" or manifest.get("episode_id") != episode_id:
        raise ValueError(f"GNSS parent domain supplement version or identity differs: {manifest_path}")
    seed_digest = _parent_domain_seed_digest(episode_id)
    previous_error, parent_tick = _parent_gnss_error_before_fork(
        episode_id, actor_id, fork_tick
    )
    result: dict[int, tuple[str, dict[str, Any], dict[str, str]]] = {}
    if previous_error is None:
        # The governed GNSS model has an explicit base-error parameter.  When
        # the parent supplement has no resolvable navigation state before the
        # fork, initialise the local FSM at that documented base rather than
        # copying an unknown parent label.  This is valid only if the branch
        # supplies a concrete navigation_state at its first local sample.
        first_row = next(
            (
                rows.get((tick, actor_id))
                for tick in ticks
                if _row_present(rows.get((tick, actor_id)))
            ),
            None,
        )
        if isinstance(first_row, Mapping) and isinstance(
            first_row.get("navigation_state"), Mapping
        ):
            previous_error = [0.0, 0.0, 0.0]
            parent_tick = fork_tick
    if previous_error is None or parent_tick is None:
        for tick in ticks:
            result[tick] = (
                "unknown",
                _missing_observations(
                    "aw_data/domain_state_supplement",
                    tick,
                    "pre_fork_gnss_fsm_state",
                ),
                dict(binding_hint),
            )
        return result
    scene_entities = {
        str(item.get("entity_id") or ""): dict(item)
        for item in scene.get("entities") or []
        if isinstance(item, Mapping) and item.get("entity_id")
    }
    service = params["service_instance"]
    bindings = dict(binding_hint)
    bindings.setdefault("actor", actor_id)
    bindings.setdefault("service", str(service["service_id"]))
    for tick in ticks:
        actor = rows.get((tick, actor_id))
        if actor is None or not _row_present(actor):
            absent = _explicit_absent_row(actor)
            result[tick] = (
                "out_of_scope" if absent else "unknown",
                {"scope.present": False}
                if absent
                else _missing_observations(
                    "trajectories.jsonl", tick, f"entity_id={actor_id}"
                ),
                bindings,
            )
            continue
        position = _vector3(actor.get("pos_enu"))
        facilities: list[Mapping[str, Any]] = []
        for entity_id, entity in scene_entities.items():
            live = rows.get((tick, entity_id))
            candidate = dict(entity)
            if isinstance(live, Mapping) and _row_present(live):
                candidate.update(dict(live))
            if _is_typed_facility(entity):
                facilities.append(candidate)
        values, quality, source_class = _gnss_values(
            uav=actor,
            uav_id=actor_id,
            position=position,
            previous_error=previous_error,
            facilities=facilities,
            params=params,
            tick=tick,
            seed_digest=seed_digest,
        )
        next_error = _vector3(values.pop("_next_error_vector_m", None))
        if next_error is not None:
            previous_error = next_error
        quality_level = values.get("quality_level")
        if quality_level == DOMAIN_UNKNOWN:
            value = "unknown"
            observations = {
                "domain.gnss_navigation": values,
                **_missing_observations(
                    "trajectories.jsonl", tick, "navigation_state"
                ),
            }
        else:
            value = (
                "true"
                if quality_level in {"degraded", "spoofing_suspect"}
                else "false"
            )
            observations = {
                "domain.gnss_navigation": values,
                "domain.gnss_navigation.quality": quality,
                "domain.gnss_navigation.source_class": source_class,
                "domain.gnss_navigation.parent_fsm_tick": parent_tick,
            }
        result[tick] = value, observations, bindings
    return result


def eval_domain(
    predicate_id: str,
    *,
    tick: int,
    actor_ids: Sequence[str],
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    parent_values: Mapping[int, str],
    arm_kind: str,
    operation: str,
    realization_tick: int | None,
    delay_ticks: int,
    fork_tick: int,
    binding_hint: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Evaluate a domain predicate from the current overlaid runtime state.

    ``parent_values`` and the operation metadata remain in the signature for
    callers that already provide them, but they are intentionally not truth
    inputs.  A missing runtime field is unresolved rather than inherited from
    a parent objective label or synthesized from an arm operation.
    """

    actor_id = actor_ids[0] if actor_ids else ""
    bindings = dict(binding_hint or {})
    if predicate_id == "ground.road_segment_closed":
        subject = bindings.get("road_segment")
        road_row = rows.get((tick, subject)) if subject else None
        if not isinstance(road_row, Mapping):
            return "unknown", _missing_observations("own_branch_road_segment_state", tick, "road_closed"), bindings
        state = road_row.get("road_segment_state")
        if (not isinstance(state, Mapping) or state.get("road_segment_id") != subject
                or state.get("ontology_class_id") != "world:RoadSegment"):
            return "unknown", _missing_observations("own_branch_road_segment_state", tick, "typed_RoadSegment"), bindings
        raw = state.get("road_closed")
        if type(raw) is not bool:
            return "unknown", _missing_observations("own_branch_road_segment_state", tick, "road_closed"), bindings
        return "true" if raw else "false", {"domain.road_closure.road_closed": raw}, bindings
    field_paths: dict[str, tuple[str, ...]] = {
        "positioning.gnss_spoofing_active": (
            "navigation_state.gnss_spoofed",
            "navigation_state.spoofing_active",
        ),
        "positioning.gnss_quality_degraded": (
            "navigation_state.gnss_mode",
            "navigation_state.degraded",
        ),
        "security.communication_jamming_impact_active": (
            "security_state.jamming_active",
            "security_state.threat",
        ),
        "security.ground_control_station_compromised": (
            "security_state.auth_state",
            "security_state.gcs_compromised",
        ),
        "communication.link_unavailable": (
            "communication_state.communication_unavailable",
            "communication_state.link_available",
            "communication_state.availability",
        ),
        "communication.link_degraded": (
            "communication_state.mode",
            "communication_state.link_state",
            "communication_state.link_status",
        ),
        "facility.charging_station_unavailable": (
            "facility_state.availability",
        ),
        "ground.traffic_signal_malfunction_active": (
            # Governed field is controller_health string codes only.
            # facility_state.fault (bool) and semantic_role are not substitutes.
            "domain.signal_queue.controller_health",
        ),
        "ground.road_segment_closed": (
            "domain.road_closure.road_closed",
            "incident_state.road_closed",
            "incident_state.isolation_active",
            "vehicle_state.road_closed",
        ),
        "safety.crowd_evacuation_active": (
            "domain.crowd_evacuation.evacuation_active",
            "pedestrian_state.evacuation_active",
            "incident_state.evacuation_active",
        ),
    }
    paths = field_paths.get(predicate_id)
    if paths is None:
        return "unknown", _missing_observations("predicate_registry", tick, f"unsupported:{predicate_id}"), bindings

    preferred_ids = list(dict.fromkeys([*map(str, actor_ids), *bindings.values()]))
    candidates = sorted(
        (
            entity_id,
            candidate,
        )
        for (sample_tick, entity_id), candidate in rows.items()
        if sample_tick == tick
    )
    matching = [
        (entity_id, candidate)
        for entity_id, candidate in candidates
        if _nested_get_with_path(candidate, *paths)[1] is not None
    ]
    preferred_matching = [
        item for item in matching if item[0] in set(preferred_ids)
    ]
    if predicate_id in {"ground.traffic_signal_malfunction_active", "facility.charging_station_unavailable"}:
        # The query subject is fixed at the decision. Other actors (including
        # other healthy signals) cannot supply its state or replace a missing
        # observation. Actor order and spelling have no semantic authority.
        role = "signal" if predicate_id == "ground.traffic_signal_malfunction_active" else "station"
        subject = bindings.get(role)
        selected = next((item for item in candidates if item[0] == subject), None)
    else:
        selected = preferred_matching[0] if preferred_matching else matching[0] if len(matching) == 1 else None
    row: Mapping[str, Any] | None = selected[1] if selected else None
    if selected is not None:
        actor_id = selected[0]
    if not bindings and selected is not None:
        role = {
            "facility.charging_station_unavailable": "station",
            "ground.traffic_signal_malfunction_active": "signal",
            "ground.road_segment_closed": "road_segment",
            "safety.crowd_evacuation_active": "crowd",
            "security.ground_control_station_compromised": "station",
        }.get(predicate_id, "actor")
        bindings[role] = actor_id

    observations: dict[str, Any] = {}
    if row is None:
        field = paths[0]
        return "unknown", _missing_observations("trajectories.jsonl", tick, field), bindings
    if _explicit_absent_row(row):
        return "out_of_scope", {"scope.present": False}, bindings
    if _entity_label(row) is None:
        return "unknown", _missing_observations("trajectories.jsonl", tick, "label_class"), bindings

    # Signal malfunction: map controller_health faithfully; never treat bool fault
    # or semantic_role=green_cycle as the governed health code.
    if predicate_id == "ground.traffic_signal_malfunction_active":
        code, health_path, health_status = resolve_signal_controller_health(row)
        observations["domain.signal_queue.controller_health"] = code
        observations["signal.controller_health_source"] = health_path
        observations["signal.controller_health_status"] = health_status
        if "facility_state" in row and isinstance(row.get("facility_state"), Mapping):
            observations["runtime.facility_state.fault"] = row["facility_state"].get("fault")
        if row.get("semantic_role") is not None:
            observations["runtime.semantic_role_ignored"] = row.get("semantic_role")
        if health_status == "conflict":
            return "unknown", {
                **observations,
                **_missing_observations(
                    "trajectories.jsonl",
                    tick,
                    "controller_health_authority_conflict",
                ),
            }, bindings
        runtime_value = evaluate_signal_controller_health_code(code)
        if runtime_value in {"true", "false"}:
            return runtime_value, observations, bindings
        return "unknown", {
            **observations,
            **_missing_observations(
                "trajectories.jsonl",
                tick,
                "domain.signal_queue.controller_health",
            ),
        }, bindings

    raw, path = _nested_get_with_path(row, *paths)
    observation_key = {
        "positioning.gnss_spoofing_active": "domain.gnss_navigation.gnss_spoofed",
        "positioning.gnss_quality_degraded": "domain.gnss_navigation.quality_level",
        "security.communication_jamming_impact_active": "domain.security_command.jamming_indicator",
        "security.ground_control_station_compromised": "domain.security_command.auth_state",
        "communication.link_unavailable": "communication.link_unavailable",
        "communication.link_degraded": "communication.link_degraded",
        "facility.charging_station_unavailable": "domain.pad_facility.availability",
        "ground.traffic_signal_malfunction_active": "domain.signal_queue.controller_health",
        "ground.road_segment_closed": "domain.road_closure.road_closed",
        "safety.crowd_evacuation_active": "domain.crowd_evacuation.evacuation_active",
    }[predicate_id]
    observations[observation_key] = raw
    if path is None:
        return "unknown", {
            **observations,
            **_missing_observations("trajectories.jsonl", tick, paths[0]),
        }, bindings

    def from_runtime(value: Any) -> str | None:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = _number(value)
            if number is not None:
                return "true" if number != 0.0 else "false"
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {
                "true",
                "degraded",
                "down",
                "unavailable",
                "spoofing_suspect",
                "spoofed",
                "jammed",
                "jamming",
                "compromised",
                "fault",
                "all_red_fault",
                "lossy",
            }:
                return "true"
            if text in {
                "false",
                "nominal",
                "ok",
                "available",
                "up",
                "clear",
                "healthy",
                "authenticated",
                "normal",
                "online",
                "backup_link",
                "reserved",
                "staged",
                "recovered",
            }:
                return "false"
        return None

    runtime_value: str | None
    if predicate_id == "communication.link_degraded" and isinstance(raw, str):
        text = raw.strip().lower()
        runtime_value = (
            "true"
            if text in {"degraded", "poor", "lossy"}
            else "false"
            if text in {"online", "backup_link", "available", "nominal", "down"}
            else None
        )
    elif predicate_id == "communication.link_unavailable" and path == "communication_state.link_available":
        available = from_runtime(raw)
        runtime_value = None if available is None else ("false" if available == "true" else "true")
    elif predicate_id == "communication.link_unavailable" and path == "communication_state.availability":
        availability = _number(raw)
        runtime_value = None if availability is None else ("true" if availability <= 0.0 else "false")
    elif predicate_id == "communication.link_unavailable" and isinstance(raw, str):
        text = raw.strip().lower()
        runtime_value = (
            "true"
            if text in {"unavailable", "down", "offline", "lost"}
            else "false"
            if text in {"online", "available", "degraded", "backup_link"}
            else None
        )
    elif predicate_id == "positioning.gnss_quality_degraded" and isinstance(raw, str):
        text = raw.strip().lower()
        runtime_value = (
            "true"
            if text in {"degraded", "spoofing_suspect"}
            else "false"
            if text in {"nominal", "good", "recovered", "normal"}
            else None
        )
    elif predicate_id == "facility.charging_station_unavailable":
        # Template op=equals value=unavailable. Fault/failed are not true.
        if isinstance(raw, str):
            text = raw
            runtime_value = (
                "true"
                if text == "unavailable"
                else "false"
                if text in {"available", "reserved", "fault", "failed"}
                else None
            )
        else:
            runtime_value = None
    elif predicate_id == "ground.traffic_signal_malfunction_active":
        # Template op=equals value=all_red_fault. Generic fault/True are not true.
        if isinstance(raw, str):
            text = raw.strip().lower()
            runtime_value = (
                "true"
                if text == "all_red_fault"
                else "false"
                if text in {"nominal", "sumo_cycle", "fault", "healthy"}
                else None
            )
        elif isinstance(raw, bool):
            runtime_value = "false"
        else:
            runtime_value = None
    else:
        runtime_value = from_runtime(raw)
    if runtime_value in {"true", "false"}:
        if predicate_id in {"communication.link_degraded", "communication.link_unavailable"} and not {"actor", "station"}.issubset(bindings):
            return "unknown", {
                **observations,
                **_missing_observations(
                    "parent_objective/event_occurrences.jsonl",
                    tick,
                    "typed_actor_station_binding",
                ),
            }, bindings
        return runtime_value, observations, bindings
    return "unknown", {
        **observations,
        **_missing_observations("trajectories.jsonl", tick, path),
    }, bindings


_DOMAIN_STATE_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "positioning.gnss_quality_degraded": (("navigation_state", "gnss_mode"), ("navigation_state", "degraded")),
    "positioning.gnss_spoofing_active": (("navigation_state", "gnss_spoofed"), ("navigation_state", "spoofing_active")),
    "security.communication_jamming_impact_active": (("security_state", "jamming_active"), ("security_state", "threat")),
    "security.ground_control_station_compromised": (("security_state", "gcs_compromised"), ("security_state", "auth_state")),
    "communication.link_unavailable": (("communication_state", "communication_unavailable"), ("communication_state", "link_available"), ("communication_state", "availability")),
    "communication.link_degraded": (("communication_state", "mode"), ("communication_state", "link_state"), ("communication_state", "link_status")),
    "facility.charging_station_unavailable": (("facility_state", "availability"), ("facility_state", "fault")),
    "ground.traffic_signal_malfunction_active": (
        ("domain", "signal_queue", "controller_health"),
        ("facility_state", "fault"),
        ("vehicle_state", "signal_malfunction"),
        ("incident_state", "signal_malfunction"),
        ("incident_state", "fault_active"),
    ),
    "ground.road_segment_closed": (
        ("domain", "road_closure", "road_closed"),
        ("incident_state", "road_closed"),
        ("incident_state", "isolation_active"),
        ("vehicle_state", "road_closed"),
    ),
    "safety.crowd_evacuation_active": (("domain", "crowd_evacuation", "evacuation_active"), ("pedestrian_state", "evacuation_active"), ("incident_state", "evacuation_active")),
}


def _branch_has_relevant_domain_delta(
    predicate_id: str,
    delta_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    *,
    actor_ids: Sequence[str],
    binding_hint: Mapping[str, str],
) -> bool:
    """Decide whether parent domain fallback must be disabled.

    A movement-only delta for an unrelated UAV does not invalidate an
    unchanged parent road/crowd/facility observation.  Conversely, a row that
    actually carries the predicate's runtime/domain field makes the branch
    authoritative.  A lifecycle delta alone only changes entity presence; it
    is not evidence that an incident/facility/domain state changed, so parent
    domain truth remains a valid immutable base in that case.
    """

    paths = _DOMAIN_STATE_PATHS.get(predicate_id, ())
    if not paths:
        return bool(delta_rows)
    relevant_ids = {str(value) for value in actor_ids if str(value)}
    relevant_ids.update(str(value) for value in binding_hint.values() if str(value))
    for row in delta_rows:
        entity_id = str(row.get("entity_id") or "")
        if relevant_ids and entity_id not in relevant_ids:
            continue
        for path in paths:
            cursor: Any = row
            present = True
            for key in path:
                if not isinstance(cursor, Mapping) or key not in cursor:
                    present = False
                    break
                cursor = cursor[key]
            if present and cursor is not None:
                return True
    return False


def parent_predicate_values(
    parent_objective: Path,
    predicate_id: str,
    actor_ids: Sequence[str],
) -> dict[int, str]:
    values: dict[int, str] = {}
    path = parent_objective / "predicate_truth.jsonl"
    if not path.is_file():
        gz = parent_objective / "predicate_truth.jsonl.gz"
        path = gz if gz.is_file() else path
    if not path.is_file():
        return values
    wanted = set(actor_ids)
    for row in _read_jsonl(path):
        if str(row.get("predicate_id") or "") != predicate_id:
            continue
        bindings = row.get("bindings") or {}
        if wanted and isinstance(bindings, Mapping):
            bound = {str(value) for value in bindings.values()}
            if bound.isdisjoint(wanted):
                continue
        tick = int(row.get("tick", -1))
        value = str(row.get("value") or "")
        if value in {"true", "false", "unknown", "out_of_scope"}:
            values[tick] = value
    return values


def parent_template_row(
    parent_objective: Path,
    predicate_id: str,
    actor_ids: Sequence[str],
) -> dict[str, Any] | None:
    path = parent_objective / "predicate_truth.jsonl"
    if not path.is_file():
        gz = parent_objective / "predicate_truth.jsonl.gz"
        path = gz if gz.is_file() else path
    if not path.is_file():
        return None
    wanted = set(actor_ids)
    fallback = None
    for row in _read_jsonl(path):
        if str(row.get("predicate_id") or "") != predicate_id:
            continue
        fallback = row
        bindings = row.get("bindings") or {}
        if wanted and isinstance(bindings, Mapping):
            bound = {str(value) for value in bindings.values()}
            if not bound.isdisjoint(wanted) and str(row.get("value") or "") != "out_of_scope":
                return row
    return fallback


def parent_event_binding_hint(
    parent_objective: Path,
    *,
    family: str,
    predicate_id: str,
    rule_id: str,
    actor_ids: Sequence[str],
    fork_tick: int,
    end_tick: int,
) -> dict[str, str]:
    """Resolve typed participant identity from small parent event evidence.

    Only role bindings are reused; parent truth values and result labels are
    never inputs to branch predicate evaluation.
    """

    path = parent_objective / "event_occurrences.jsonl"
    if not path.is_file():
        return {}
    actor_set = set(map(str, actor_ids))
    candidates: list[tuple[int, int, dict[str, str]]] = []
    for row in _read_jsonl(path):
        if str(row.get("event_family_id") or "") != family:
            continue
        observed_predicate = str(
            row.get("trigger_predicate_id") or row.get("predicate_id") or ""
        )
        if observed_predicate and observed_predicate != predicate_id:
            continue
        observed_rule = str(row.get("rule_id") or "")
        if observed_rule and rule_id and observed_rule != rule_id:
            continue
        tick = next(
            (
                int(row[key])
                for key in (
                    "evidence_tick",
                    "trigger_tick",
                    "detection_tick",
                    "start_tick",
                )
                if row.get(key) is not None
            ),
            None,
        )
        if tick is None or tick < fork_tick - CAPTURE_STEP or tick > end_tick:
            continue
        raw_bindings = row.get("bindings")
        if not isinstance(raw_bindings, Mapping):
            continue
        bindings = {
            str(key): str(value)
            for key, value in raw_bindings.items()
            if str(key) and str(value)
        }
        if not bindings:
            continue
        overlap = len(actor_set.intersection(bindings.values()))
        candidates.append((-overlap, abs(tick - fork_tick), bindings))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (item[0], item[1], sorted(item[2].items())))
    best_score = candidates[0][:2]
    best = {
        tuple(sorted(bindings.items())): bindings
        for overlap, distance, bindings in candidates
        if (overlap, distance) == best_score
    }
    return next(iter(best.values())) if len(best) == 1 else {}


def _scene_entities_by_id(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("entity_id") or ""): item
        for item in scene.get("entities") or []
        if isinstance(item, Mapping) and item.get("entity_id")
    }


def _placement_edge_id(entity: Mapping[str, Any] | None) -> str:
    if not isinstance(entity, Mapping):
        return ""
    placement = entity.get("placement") if isinstance(entity.get("placement"), Mapping) else {}
    return str(placement.get("edge_id") or "")


def _placement_semantics(entity: Mapping[str, Any] | None) -> str:
    if not isinstance(entity, Mapping):
        return ""
    placement = entity.get("placement") if isinstance(entity.get("placement"), Mapping) else {}
    return str(placement.get("placement_semantics") or "")


def _script_event_map(script: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("event_id") or ""): item
        for item in script.get("events") or []
        if isinstance(item, Mapping) and item.get("event_id")
    }


def _events_by_intent(script: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    by_intent: dict[str, list[Mapping[str, Any]]] = {}
    for event in script.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        intent = str(event.get("intent") or "")
        if not intent:
            log_event = event.get("log_event") if isinstance(event.get("log_event"), Mapping) else {}
            intent = str(log_event.get("intent") or "")
        if intent:
            by_intent.setdefault(intent, []).append(event)
    return by_intent


def _charger_availability_entity_ids(event: Mapping[str, Any], scene_entities: Mapping[str, Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    for action in event.get("actions") or []:
        if not isinstance(action, Mapping):
            continue
        eid = str(action.get("entity_id") or action.get("actor_id") or "")
        if not eid:
            continue
        patch = action.get("state_patch") if isinstance(action.get("state_patch"), Mapping) else {}
        facility = patch.get("facility_state") if isinstance(patch.get("facility_state"), Mapping) else {}
        if "availability" not in facility:
            continue
        ent = scene_entities.get(eid)
        kind = str((ent or {}).get("entity_kind") or "").strip().lower()
        if kind != "facility.charger":
            continue
        if eid not in out:
            out.append(eid)
    return out


def fork_time_domain_binding_hint(
    *,
    script: Mapping[str, Any] | None,
    scene: Mapping[str, Any],
    group: Mapping[str, Any],
    predicate_id: str,
    fork_tick: int,
    parent_realizations: Sequence[Mapping[str, Any]],
    parent_objective: Path | None = None,
) -> dict[str, str]:
    """Fork-time typed role bindings for domain predicates.

    Uses authored script/scene causal predecessor contracts available at fork.
    Never scans future parent events, never uses saved predicate_truth values.
    Ambiguity yields empty hint (caller must leave unknown).
    """

    if not isinstance(script, Mapping):
        return {}
    scene_entities = _scene_entities_by_id(scene)
    anchor_id = str(group.get("anchor_event_id") or "")
    events = _script_event_map(script)
    anchor = events.get(anchor_id)

    if predicate_id == "facility.charging_station_unavailable":
        if anchor is None:
            return {}
        # The anchor is the prospective request. Earlier events, however, must
        # have an actual completed observation available at the decision.
        available = {
            row["event_id"]
            for row in parent_realizations
            if isinstance(row.get("event_id"), str)
            and type(row.get("dispatch_tick")) is int
            and type(row.get("result_tick")) is int
            and row["dispatch_tick"] <= fork_tick
            and row["result_tick"] <= fork_tick
        }
        triggers = {row["trigger_id"]: row for row in script.get("triggers", [])}
        by_intent = _events_by_intent(script)
        current = anchor
        visited = set()
        while True:
            event_id = current["event_id"]
            if event_id in visited:
                raise WindowSemanticsError("cycle in charger causal predecessor context")
            visited.add(event_id)
            stations = _charger_availability_entity_ids(current, scene_entities)
            if stations:
                return {"station": stations[0]} if len(stations) == 1 else {}
            trigger = triggers.get(current.get("trigger_ref"), {})
            if trigger.get("type") == "event_fired_after":
                predecessor = trigger["event_id"]
                candidates = [events[predecessor]] if predecessor in available and predecessor in events else []
            else:
                intent = current.get("causal_predecessor_intent")
                candidates = [event for event in by_intent.get(intent, [])
                              if event["event_id"] in available and event["event_id"] not in visited]
            if len(candidates) != 1:
                return {}
            current = candidates[0]

    if predicate_id == "ground.road_segment_closed":
        if anchor is None:
            return {}
        available = {r["event_id"] for r in parent_realizations
                     if isinstance(r.get("event_id"), str)
                     and type(r.get("dispatch_tick")) is int
                     and type(r.get("result_tick")) is int
                     and r["dispatch_tick"] <= fork_tick and r["result_tick"] <= fork_tick}
        triggers = {t["trigger_id"]: t for t in script["triggers"]}
        current = anchor
        visited = set()
        while current is not None:
            identity = current["event_id"]
            if identity in visited:
                raise WindowSemanticsError("road query causal cycle")
            visited.add(identity)
            targets = list(current.get("log_event", {}).get("target_ids", []))
            for action in current.get("actions", []):
                for field in ("entity_id", "actor_id", "target_id"):
                    if isinstance(action.get(field), str):
                        targets.append(action[field])
            segments = set()
            for target in targets:
                entity = scene_entities.get(target)
                if not isinstance(entity, Mapping):
                    continue
                placement = entity.get("placement")
                if not isinstance(placement, Mapping):
                    continue
                # A mobile entity's initial placement is not its road at fork.
                # Only an explicit stationary lane-scope declaration is usable.
                if _typed_category(entity) in {"vehicle", "uav", "pedestrian"}:
                    continue
                if placement.get("placement_semantics") not in {"lane_center", "travel_lane"}:
                    continue
                edge = placement.get("edge_id")
                if isinstance(edge, str) and edge.strip():
                    segments.add(edge)
            if segments:
                return {"road_segment": next(iter(segments))} if len(segments) == 1 else {}
            trigger = triggers.get(current.get("trigger_ref"), {})
            if trigger.get("type") == "event_fired_after":
                predecessor = trigger["event_id"]
                candidates = [events[predecessor]] if predecessor in available and predecessor in events else []
            else:
                intent = current.get("causal_predecessor_intent")
                candidates = [event for event in events.values() if intent
                              and event.get("intent") == intent
                              and event["event_id"] in available
                              and event["event_id"] not in visited]
            current = candidates[0] if len(candidates) == 1 else None
        return {}

    if predicate_id == "ground.traffic_signal_malfunction_active":
        if anchor is None:
            return {}
        subjects: list[str] = []
        for action in anchor.get("actions") or []:
            if not isinstance(action, Mapping):
                continue
            eid = str(action.get("entity_id") or action.get("actor_id") or "")
            ent = scene_entities.get(eid)
            if not eid or not isinstance(ent, Mapping):
                continue
            if str(ent.get("category") or "").lower() == "traffic_light":
                subjects.append(eid)
        log_event = anchor.get("log_event") if isinstance(anchor.get("log_event"), Mapping) else {}
        for tid in log_event.get("target_ids") or []:
            ent = scene_entities.get(str(tid))
            if isinstance(ent, Mapping) and str(ent.get("category") or "").lower() == "traffic_light":
                subjects.append(str(tid))
        subjects = list(dict.fromkeys(subjects))
        if len(subjects) == 1:
            return {"signal": subjects[0]}
        return {}

    return {}


def anchor_script_binding_hint(
    script: Mapping[str, Any] | None,
    *,
    anchor_event_id: str,
    scene: Mapping[str, Any],
    predicate_id: str,
) -> dict[str, str]:
    """Resolve typed target roles from the authoritative anchor event.

    Some candidate Arms describe only the intervention actor (for example,
    the restricted-zone facility) and omit the predicate's aircraft
    argument.  The event script's explicit ``log_event.target_ids`` and
    action entity IDs are authoritative source fields, so they may supply a
    binding when they identify exactly one UAV and one restricted region.  No
    candidate label or nearest-entity heuristic is used; ambiguity remains
    unresolved and fail-closed.
    """

    if predicate_id != "airspace.aircraft_inside_restricted_airspace" or not isinstance(script, Mapping):
        return {}
    event = next(
        (
            item
            for item in script.get("events") or []
            if isinstance(item, Mapping)
            and str(item.get("event_id") or "") == str(anchor_event_id)
        ),
        None,
    )
    if event is None:
        return {}
    target_ids: list[str] = []
    for source in (event, event.get("log_event") if isinstance(event.get("log_event"), Mapping) else {}):
        values = source.get("target_ids") if isinstance(source, Mapping) else None
        if isinstance(values, list):
            target_ids.extend(str(value) for value in values if str(value))
    for action in event.get("actions") or []:
        if not isinstance(action, Mapping):
            continue
        for key in ("entity_id", "actor_id", "target_id"):
            value = str(action.get(key) or "")
            if value:
                target_ids.append(value)
    ordered_ids = list(dict.fromkeys(target_ids))
    scene_entities = {
        str(item.get("entity_id") or ""): item
        for item in scene.get("entities") or []
        if isinstance(item, Mapping) and item.get("entity_id")
    }
    regions = _restricted_regions(scene_entities)
    uavs = sorted(
        entity_id
        for entity_id in ordered_ids
        if _is_typed_uav(scene_entities.get(entity_id))
    )
    region_ids = sorted(entity_id for entity_id in ordered_ids if entity_id in regions)
    if len(uavs) == 1 and len(region_ids) == 1:
        return {"actor": uavs[0], "restricted_region": region_ids[0]}
    return {}


def _branch_identity(
    group: Mapping[str, Any], arm_record: Mapping[str, Any]
) -> dict[str, str]:
    """Resolve the canonical five-part identity from candidate authority."""

    episode_id = str(group.get("episode_id") or "")
    profile_id = str(
        arm_record.get("profile_id")
        or group.get("scenario_id")
        or (episode_id.rsplit("__seed", 1)[0] if "__seed" in episode_id else "")
    )
    anchor = arm_record.get("anchor")
    if not isinstance(anchor, Mapping):
        anchor = {}
    seed_id = str(
        anchor.get("seed_id")
        or group.get("seed_id")
        or (
            f"seed{episode_id.rsplit('__seed', 1)[1]}"
            if "__seed" in episode_id
            else ""
        )
    )
    branch_group_id = str(group.get("branch_group_id") or "")
    group_id = str(group.get("group_id") or "")
    prefix = f"{episode_id}__"
    if not group_id and branch_group_id.startswith(prefix):
        group_id = branch_group_id[len(prefix) :]
    arm_id = str(
        arm_record.get("arm_id")
        or (arm_record.get("decision_arm") or {}).get("arm_id")
        or ""
    )
    if not all((profile_id, episode_id, seed_id, group_id, arm_id)):
        raise WindowSemanticsError(
            "candidate authority cannot form profile/episode/seed/group/arm identity"
        )
    return {
        "profile_id": profile_id,
        "episode_id": episode_id,
        "seed_id": seed_id,
        "group_id": group_id,
        "arm_id": arm_id,
        "branch_key": "/".join(
            (profile_id, episode_id, seed_id, group_id, arm_id)
        ),
    }


def _write_window_graph_deltas(
    *,
    objective: Path,
    identity: Mapping[str, str],
    family: str,
    predicate_id: str,
    fork_tick: int,
    end_tick: int,
    parent_objective: Path,
    predicate_rows: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist evidence-backed target graph deltas, never inferred causality.

    The parent Episode remains an immutable base reference.  The branch files
    contain only target predicate/event records measured inside this Arm's
    causal window.  ``script_scheduled`` is deliberately absent: scheduling
    provenance is not a causal edge.
    """

    branch_key = identity["branch_key"]
    world_rows: list[dict[str, Any]] = []
    for row in predicate_rows:
        world_rows.append(
            {
                "schema_name": "aeroworld_branch_world_truth_graph_delta",
                "schema_version": "1.0.0",
                "branch_key": branch_key,
                "episode_id": identity["episode_id"],
                "tick": int(row["tick"]),
                "operation": "upsert_target_predicate_truth",
                "predicate_id": predicate_id,
                "bindings": dict(row.get("bindings") or {}),
                "truth_value": row.get("value"),
                "source_truth_id": row.get("truth_id"),
                "tuple_id": row.get("tuple_id"),
                "source_refs": list(row.get("source_refs") or []),
                "missing_source_record": list(
                    row.get("missing_source_record") or []
                ),
            }
        )

    semantic_rows: list[dict[str, Any]] = []
    for row in transitions:
        semantic_rows.append(
            {
                "schema_name": "aeroworld_branch_semantic_graph_delta",
                "schema_version": "1.0.0",
                "branch_key": branch_key,
                "episode_id": identity["episode_id"],
                "tick": int(row["to_tick"]),
                "operation": "add_target_predicate_transition",
                "predicate_id": predicate_id,
                "source_transition_id": row.get("transition_id"),
                "from_tick": row.get("from_tick"),
                "to_tick": row.get("to_tick"),
                "from_value": row.get("from_value"),
                "to_value": row.get("to_value"),
                "direction": row.get("direction"),
                "bindings": dict(row.get("bindings") or {}),
                "evidence_truth_ids": list(row.get("evidence_truth_ids") or []),
            }
        )
    for row in occurrences:
        semantic_rows.append(
            {
                "schema_name": "aeroworld_branch_semantic_graph_delta",
                "schema_version": "1.0.0",
                "branch_key": branch_key,
                "episode_id": identity["episode_id"],
                "tick": int(row["evidence_tick"]),
                "operation": "add_target_event_occurrence",
                "event_family_id": family,
                "predicate_id": predicate_id,
                "event_id": row.get("event_id"),
                "event_type_id": row.get("event_type_id"),
                "rule_id": row.get("rule_id"),
                "bindings": dict(row.get("bindings") or {}),
                "supporting_transition_ids": list(
                    row.get("supporting_transition_ids") or []
                ),
                "source_refs": list(row.get("source_refs") or []),
            }
        )
    for row in outcomes:
        evidence_tick = row.get("evidence_tick")
        semantic_rows.append(
            {
                "schema_name": "aeroworld_branch_semantic_graph_delta",
                "schema_version": "1.0.0",
                "branch_key": branch_key,
                "episode_id": identity["episode_id"],
                "tick": int(evidence_tick) if evidence_tick is not None else None,
                "operation": "add_target_event_outcome",
                "event_family_id": family,
                "predicate_id": predicate_id,
                "event_id": row.get("event_id"),
                "outcome_id": row.get("outcome_id"),
                "status": row.get("status"),
                "lifecycle_status": row.get("lifecycle_status"),
                "missing_source_record": list(
                    row.get("missing_source_record") or []
                ),
            }
        )

    world_path = objective / "world_truth_graph_deltas.jsonl"
    semantic_path = objective / "semantic_graph_deltas.jsonl"
    _write_jsonl(world_path, world_rows)
    _write_jsonl(semantic_path, semantic_rows)

    parent_graph_bases: dict[str, str] = {}
    for name in ("world_truth_graph_base.json", "semantic_graph_base.json"):
        source = parent_objective / name
        if source.is_file():
            parent_graph_bases[name] = _project_relative(source)
    return {
        "schema_name": "aeroworld_branch_graph_index",
        "schema_version": "1.0.0",
        **dict(identity),
        "event_family_id": family,
        "predicate_id": predicate_id,
        "fork_tick": fork_tick,
        "window_end_tick": end_tick,
        "window": {"fork_tick": fork_tick, "window_end_tick": end_tick},
        "artifact_scope": "parent_graph_reference_plus_target_window_delta",
        "causal_relation_policy": "measured_evidence_only",
        "script_scheduled_is_causal_relation": False,
        "parent_graph_bases": parent_graph_bases,
        "delta_files": {
            "world_truth_graph_deltas": {
                "path": world_path.name,
                "source_present": bool(world_rows),
                "row_count": len(world_rows),
            },
            "semantic_graph_deltas": {
                "path": semantic_path.name,
                "source_present": bool(semantic_rows),
                "row_count": len(semantic_rows),
            },
        },
        "world_truth_delta_rows": len(world_rows),
        "semantic_delta_rows": len(semantic_rows),
        "g10_not_claimed": True,
    }


def _write_branch_plot(
    path: Path,
    *,
    branch_key: str,
    predicate_id: str,
    fork_tick: int,
    end_tick: int,
    predicate_rows: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    entry_active: bool,
) -> None:
    """Render the Arm's measured target truth; no parent plot is reused."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {
        "false": 0.0,
        "true": 1.0,
        "unknown": -0.15,
        "out_of_scope": -0.3,
    }
    ticks = [int(row["tick"]) for row in predicate_rows]
    ys = [values.get(str(row.get("value")), -0.15) for row in predicate_rows]
    colors = [
        "#147d64"
        if y == 1.0
        else "#3f5f88"
        if y == 0.0
        else "#8c3b72"
        if y == -0.3
        else "#c23b22"
        for y in ys
    ]
    fig, ax = plt.subplots(figsize=(9.6, 3.2), constrained_layout=True)
    if ticks:
        ax.step(ticks, ys, where="post", color="#283646", linewidth=1.4)
        ax.scatter(ticks, ys, c=colors, s=22, zorder=3)
    ax.axvline(fork_tick, color="#8c3b72", linestyle="--", linewidth=1.2, label="fork")
    for row in occurrences:
        tick = row.get("evidence_tick")
        if tick is not None:
            ax.axvline(int(tick), color="#d98f00", linewidth=1.0, alpha=0.9)
    if entry_active:
        ax.text(
            fork_tick,
            1.08,
            "parent-active at fork",
            fontsize=8,
            color="#147d64",
            ha="left",
        )
    ax.set_xlim(fork_tick, max(fork_tick + 1, end_tick))
    ax.set_ylim(-0.42, 1.2)
    ax.set_yticks(
        [-0.3, -0.15, 0.0, 1.0],
        ["out_of_scope", "unknown", "false", "true"],
    )
    ax.set_xlabel("engine tick")
    ax.set_ylabel("target truth")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"{branch_key}\n{predicate_id}", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_arm_window(
    *,
    group: Mapping[str, Any],
    arm_record: Mapping[str, Any],
    arm_root: Path,
    parent_capture: Path,
    parent_objective: Path,
    scene: Mapping[str, Any],
    script: Mapping[str, Any] | None = None,
    require_anchor_realization: bool = False,
) -> dict[str, Any]:
    identity = _branch_identity(group, arm_record)
    family = str(group.get("event_family_id") or "")
    predicate_id = str(group.get("target_predicate_id") or "")
    rule_id = str(group.get("rule_id") or "")
    episode_id = identity["episode_id"]
    fork_tick = int(group["fork_tick"])
    end_tick = int(group["window_end_tick"])
    policy = arm_record.get("window_policy")
    if policy is not None:
        if (policy["fork_tick"] != fork_tick or policy["end_tick"] != end_tick
                or policy["chosen_before_rollout"] is not True
                or policy["same_for_all_arms"] is not True
                or policy["actual_result_can_change_end"] is not False):
            raise WindowSemanticsError("output window differs from fixed candidate window")
    arm_id = identity["arm_id"]
    arm_kind = str(arm_record.get("arm_kind") or (arm_record.get("decision_arm") or {}).get("arm_kind") or "")
    operation = str((arm_record.get("decision_arm") or {}).get("event_operation") or "")
    if not predicate_id or not family:
        raise WindowSemanticsError("group lacks target_predicate_id or event_family_id")
    params = get_governed_parameter_defaults()
    restricted_activity_by_tick: dict[int, dict[str, bool]] | None = None
    rule = family_rule(family, rule_id, predicate_id)
    event_type = str(rule.get("event_type") or "")
    ticks = capture_ticks(fork_tick, end_tick)
    if not ticks:
        raise WindowSemanticsError("window has no capture ticks")
    if predicate_id == RESTRICTED_PREDICATE:
        activity_ticks = list(ticks)
        previous_capture = fork_tick - CAPTURE_STEP
        if previous_capture >= 0 and previous_capture not in activity_ticks:
            activity_ticks.insert(0, previous_capture)
        restricted_activity_by_tick = _restricted_activity_for_window(
            scene, activity_ticks, arm_root
        )
    def _arm_jsonl(*parts: str) -> list[dict[str, Any]]:
        for base in (arm_root / "raw", arm_root):
            path = base.joinpath(*parts)
            candidates = (path, path.with_name(path.name + ".gz"))
            for candidate in candidates:
                rows = _read_jsonl(candidate)
                if rows or candidate.is_file():
                    return rows
        return []

    flight_ground_predicates = {"airspace.aircraft_airborne", "airspace.aircraft_touchdown"}
    parent_read_start = 0 if predicate_id in flight_ground_predicates else max(0, fork_tick - CAPTURE_STEP)
    parent_traj = _read_jsonl_window(
        parent_capture / "trajectories.jsonl", parent_read_start, end_tick
    )
    delta_traj = _arm_jsonl("trajectories.jsonl")
    lifecycle_delta = _arm_jsonl("entity_lifecycle_delta.jsonl")
    branch_realizations = _arm_jsonl("event_realization.jsonl")
    # This is resolved after typed bindings are recovered below.  A generic
    # movement delta is not, by itself, a change to a domain predicate.
    branch_has_state_delta = False
    parent_weather = _read_jsonl_window(
        parent_capture / "weather_meta.jsonl", parent_read_start, end_tick
    )
    delta_weather = _arm_jsonl("weather_meta.jsonl")
    ledger_rows = _arm_jsonl("branch_event_ledger.jsonl")
    ledger = ledger_rows[0] if ledger_rows else {}
    delay_ticks = int(ledger.get("delay_ticks") or 0)
    rows = overlay_trajectories(
        parent_traj,
        delta_traj,
        fork_tick=fork_tick,
        end_tick=end_tick,
        lifecycle_rows=lifecycle_delta,
    )
    weather = overlay_weather(parent_weather, delta_weather, fork_tick=fork_tick, end_tick=end_tick)
    local_ground_by_tick: Mapping[int, Mapping[str, float]] | None = None
    if predicate_id in {
        "airspace.aircraft_airborne",
        "airspace.aircraft_touchdown",
    }:
        roster_value = _json(parent_capture / "global_entity_roster.json")
        roster_by_id = {
            str(item.get("entity_id") or ""): dict(item)
            for item in roster_value.get("entities") or []
            if isinstance(item, Mapping) and item.get("entity_id")
        }
        positions_by_tick: dict[int, dict[str, tuple[float, float, float]]] = {}
        velocities_by_tick: dict[int, dict[str, tuple[float, float, float]]] = {}
        activities_by_tick: dict[int, dict[str, str]] = {}
        # Ground calibration includes observed parent history, while every
        # post-fork sample uses this arm's overlay. The previous capture used
        # for event edges must use this same causal ground authority.
        ground_rows = {(row['tick'], row['entity_id']): row for row in parent_traj if row['tick'] < fork_tick}
        ground_rows.update(rows)
        for (sample_tick, entity_id), row in ground_rows.items():
            position = _vector3(row.get("pos_enu"))
            velocity = _vector3(row.get("vel_mps"))
            if position is not None:
                positions_by_tick.setdefault(sample_tick, {})[entity_id] = tuple(position)
            if velocity is not None:
                velocities_by_tick.setdefault(sample_tick, {})[entity_id] = tuple(velocity)
            activity = str(row.get("activity_type") or row.get("state") or "")
            if activity:
                activities_by_tick.setdefault(sample_tick, {})[entity_id] = activity
        scene_by_id = {
            str(item.get("entity_id") or ""): dict(item)
            for item in scene.get("entities") or []
            if isinstance(item, Mapping) and item.get("entity_id")
        }
        try:
            ground = resolve_aircraft_ground_references(
                roster_by_id,
                positions_by_tick,
                velocities_by_tick,
                activities_by_tick,
                scene_by_id=scene_by_id,
                episode_id=episode_id,
            )
        except PredicateStateComputerError as exc:
            raise WindowSemanticsError(
                f"canonical aircraft ground reference failed: {exc}"
            ) from exc
        local_ground_by_tick = ground.local_ground_reference_by_tick
    actor_ids = resolve_actor_ids(group, arm_record, script)
    window_policy = arm_record.get("window_policy")
    if not isinstance(window_policy, Mapping):
        window_policy = {}
    # Domain predicates: fork-time typed script/causal predecessor only.
    # Never parent_event_binding_hint over [fork, window_end] (future leakage).
    domain_fork_time_predicates = {
        "facility.charging_station_unavailable",
        "ground.road_segment_closed",
        "ground.traffic_signal_malfunction_active",
    }
    binding_evidence = None
    if predicate_id in FIXED_GEOMETRY_PREDICATES:
        template = next(item for item in get_core_predicate_templates() if item['id'] == predicate_id)
        binding_evidence = bind_causal_query(
            template=template, script=script,
            anchor_event_id=str(group['anchor_event_id']), fork_tick=fork_tick,
            entity_classes=scene_geometry_classes(scene),
            realizations=_read_jsonl(parent_capture / 'event_realization.jsonl'),
        )
        binding_hint = binding_evidence['bindings']
        actor_ids = list(binding_hint.values())
        _write_json(arm_root / 'objective/query_binding.json', binding_evidence)
    elif predicate_id in WEATHER_PREDICATES:
        manifest_path = parent_capture / "episode_manifest.json"
        manifest = {}
        if manifest_path.is_file():
            manifest = _json(manifest_path)
        raw_capture_id = manifest.get("capture_boundary_id")
        if (
            isinstance(raw_capture_id, str)
            and raw_capture_id.strip()
        ):
            capture_boundary_id = raw_capture_id.strip()
            binding_hint = {"region": capture_boundary_id}
            actor_ids = [capture_boundary_id]
            binding_evidence = {
                "status": "resolved",
                "bindings": {"region": capture_boundary_id},
                "source": "episode_manifest.json#capture_boundary_id",
                "candidates": {"world:OperationalRegion": [capture_boundary_id]},
                "reason": "manifest_capture_boundary_identity",
            }
        else:
            binding_hint = {}
            actor_ids = []
            binding_evidence = {
                "status": "unknown",
                "bindings": {},
                "source": "episode_manifest.json#capture_boundary_id",
                "candidates": {"world:OperationalRegion": []},
                "reason": "missing_capture_boundary_id",
            }
        _write_json(arm_root / "objective/query_binding.json", binding_evidence)
    elif predicate_id in domain_fork_time_predicates:
        binding_hint = fork_time_domain_binding_hint(
            script=script,
            scene=scene,
            group=group,
            predicate_id=predicate_id,
            fork_tick=fork_tick,
            parent_realizations=_read_jsonl(parent_capture / "event_realization.jsonl"),
            parent_objective=parent_objective,
        )
    else:
        binding_search_end = min(
            900,
            max(end_tick, int(window_policy.get("end_tick") or end_tick)),
        )
        binding_hint = parent_event_binding_hint(
            parent_objective,
            family=family,
            predicate_id=predicate_id,
            rule_id=str(rule.get("rule_id") or rule_id),
            actor_ids=actor_ids,
            fork_tick=fork_tick,
            end_tick=binding_search_end,
        )
    script_hint = anchor_script_binding_hint(
        script,
        anchor_event_id=str(group.get("anchor_event_id") or ""),
        scene=scene,
        predicate_id=predicate_id,
    )
    for role, entity_id in script_hint.items():
        existing = binding_hint.get(role)
        if existing and existing != entity_id:
            raise WindowSemanticsError(
                f"authoritative binding conflict for {role}: {existing!r} vs {entity_id!r}"
            )
        binding_hint[role] = entity_id
    for entity_id in binding_hint.values():
        if entity_id not in actor_ids:
            actor_ids.append(entity_id)
    branch_has_state_delta = _branch_has_relevant_domain_delta(
        predicate_id,
        delta_traj,
        lifecycle_delta,
        actor_ids=actor_ids,
        binding_hint=binding_hint,
    )
    anchor_id = str(group.get("anchor_event_id") or "")
    anchor_realized = any(
        isinstance(item, Mapping)
        and str(item.get("event_id") or "") == anchor_id
        and str(item.get("status") or "ok") not in {"error", "failed"}
        for item in branch_realizations
    )
    gnss_quality_by_tick: dict[
        int, tuple[str, dict[str, Any], dict[str, str]]
    ] = {}
    if predicate_id == "positioning.gnss_quality_degraded":
        gnss_actor = str(binding_hint.get("actor") or "")
        if not gnss_actor:
            scene_by_id = {
                str(item.get("entity_id") or ""): item
                for item in scene.get("entities") or []
                if isinstance(item, Mapping) and item.get("entity_id")
            }
            typed_uavs = [
                entity_id
                for entity_id in actor_ids
                if _is_typed_uav(scene_by_id.get(entity_id))
            ]
            if len(set(typed_uavs)) == 1:
                gnss_actor = typed_uavs[0]
        if gnss_actor:
            gnss_quality_by_tick = precompute_gnss_quality(
                episode_id=episode_id,
                actor_id=gnss_actor,
                ticks=ticks,
                rows=rows,
                scene=scene,
                binding_hint=binding_hint,
                fork_tick=fork_tick,
            )

    def evaluate_state(
        sample_tick: int,
        state_rows: Mapping[tuple[int, str], Mapping[str, Any]],
        state_weather: Mapping[int, Mapping[str, Any]],
        *,
        state_arm_kind: str = arm_kind,
        state_operation: str = operation,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        if predicate_id in GEOMETRY_PREDICATES:
            if binding_evidence is not None and binding_evidence['status'] != 'resolved':
                return 'unknown', _missing_observations('fork_causal_context', sample_tick, binding_evidence['reason']), {}
            return eval_geometry(
                predicate_id,
                tick=sample_tick,
                actor_ids=actor_ids,
                rows=state_rows,
                scene=scene,
                params=params,
                fork_tick=fork_tick,
                script=script,
                binding_hint=binding_hint,
                local_ground_by_tick=local_ground_by_tick,
                region_activity=(
                    (restricted_activity_by_tick or {}).get(sample_tick)
                    if predicate_id == RESTRICTED_PREDICATE
                    else None
                ),
            )
        if predicate_id in WEATHER_PREDICATES:
            value, observations = eval_weather(
                predicate_id,
                state_weather.get(sample_tick) or {},
                params,
                tick=sample_tick,
            )
            region_id = binding_hint.get("region")
            if not region_id:
                observations = {
                    **observations,
                    **_missing_observations("episode_manifest.json", sample_tick, "capture_boundary_id"),
                }
                return "unknown", observations, {}
            return value, observations, {"region": region_id}
        if predicate_id in DOMAIN_PREDICATES:
            role = {
                "facility.charging_station_unavailable": "station",
                "ground.traffic_signal_malfunction_active": "signal",
                "ground.road_segment_closed": "road_segment",
            }.get(predicate_id)
            if role is not None and not binding_hint.get(role):
                return "unknown", _missing_observations("fork_causal_context", sample_tick, role), {}
            if predicate_id == "positioning.gnss_quality_degraded" and sample_tick in gnss_quality_by_tick:
                return gnss_quality_by_tick[sample_tick]
            domain_binding_hint = dict(binding_hint or {})
            if predicate_id in {"communication.link_unavailable", "communication.link_degraded"}:
                # The branch runtime row carries the measured link state, but
                # it may omit the station role.  Resolve that role from the
                # immutable parent communication record for the same actor
                # and tick; only identity is inherited here, never the
                # parent's truth value.
                parent_comm = eval_parent_compute_comm(
                    predicate_id,
                    tick=sample_tick,
                    actor_ids=actor_ids,
                    episode_id=episode_id,
                    binding_hint=domain_binding_hint,
                )
                if parent_comm is not None:
                    _, _, parent_bindings = parent_comm
                    for key in ("actor", "station"):
                        if parent_bindings.get(key) and not domain_binding_hint.get(key):
                            domain_binding_hint[key] = parent_bindings[key]
            # An empty branch state delta means the real engine reproduced the
            # parent state for this window.  In that narrow case the governed
            # parent domain observation is an immutable base reference; it is
            # never used when the branch has measured state/lifecycle changes.
            if not branch_has_state_delta and predicate_id not in {
                "facility.charging_station_unavailable", "ground.traffic_signal_malfunction_active",
                "ground.road_segment_closed",
            }:
                inherited = eval_parent_domain(
                    predicate_id,
                    tick=sample_tick,
                    actor_ids=actor_ids,
                    episode_id=episode_id,
                    binding_hint=binding_hint,
                )
                if inherited is not None:
                    return inherited
                inherited_comm = eval_parent_compute_comm(
                    predicate_id,
                    tick=sample_tick,
                    actor_ids=actor_ids,
                    episode_id=episode_id,
                    binding_hint=binding_hint,
                )
                if inherited_comm is not None:
                    return inherited_comm
            return eval_domain(
                predicate_id,
                tick=sample_tick,
                actor_ids=actor_ids,
                rows=state_rows,
                parent_values={},
                arm_kind=state_arm_kind,
                operation=state_operation,
                realization_tick=None,
                delay_ticks=delay_ticks,
                fork_tick=fork_tick,
                binding_hint=domain_binding_hint,
            )
        return "unknown", _missing_observations("predicate_registry", sample_tick, f"unsupported:{predicate_id}"), {}

    predicate_rows: list[dict[str, Any]] = []
    last_bindings: dict[str, str] = {}
    for tick in ticks:
        value, observations, bindings = evaluate_state(tick, rows, weather)
        last_bindings = bindings or last_bindings
        source_kind = (
            "branch_delta_geometry_with_parent_overlay"
            if predicate_id in GEOMETRY_PREDICATES
            else "branch_delta_weather_with_parent_overlay"
            if predicate_id in WEATHER_PREDICATES
            else "branch_runtime_with_parent_overlay"
        )
        missing_source_record = list(observations.get("missing_source_record") or [])
        if value == "unknown" and not missing_source_record:
            missing_source_record = [
                _missing_source_record("predicate_truth", tick, "predicate_evaluation")
            ]
            observations = {**observations, "missing_source_record": missing_source_record}
        source_file = "weather_meta.jsonl" if predicate_id in WEATHER_PREDICATES else "trajectories.jsonl"
        row = {
            "schema_name": "predicate_truth",
            "schema_version": "3.0.0",
            "annotation_layer": "L1",
            "source_class": "deterministic_derived",
            "source_layer": "L0",
            "materialization_policy": "branch_window_state_recompute",
            "authoritative_tick": True,
            "episode_id": episode_id,
            "tick": tick,
            "predicate_id": predicate_id,
            "bindings": bindings,
            "value": value,
            "missing_source_record": missing_source_record,
            "mechanism_id": str(rule.get("rule_id") or rule_id),
            "source_refs": [
                f"raw/{source_file}#tick={tick}",
                f"{_project_relative(parent_capture)}/{source_file}#tick={tick}",
            ],
            "truth_id": f"truth:{_digest(episode_id, arm_id, predicate_id, tick, bindings, value)}",
            "tuple_id": f"world_predicate_tuple:{_digest(predicate_id, bindings)}",
            "evidence": {
                "missing_source_record": missing_source_record,
                "missing_requirements": missing_source_record,
                "observations": [
                    {
                        "path": "normalized.grounded_window_state",
                        "value": {
                            "bindings": bindings,
                            "world_observations": [
                                {"path": key, "value": item} for key, item in observations.items()
                            ],
                            "world_truth_value": value,
                        },
                    }
                ],
            },
            "input_digest": digest_object(
                {
                    "source_kind": source_kind,
                    "episode_id": episode_id,
                    "tick": tick,
                    "bindings": bindings,
                    "observations": observations,
                }
            ),
        }
        predicate_rows.append(row)

    parent_capture_ticks: set[int] = set()
    for source_row in parent_traj:
        raw_tick = source_row.get("tick")
        if isinstance(raw_tick, bool) or raw_tick is None:
            continue
        try:
            source_tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        if source_tick % CAPTURE_STEP == 0:
            parent_capture_ticks.add(source_tick)
    for source_row in parent_weather:
        raw_tick = source_row.get("tick")
        if isinstance(raw_tick, bool) or raw_tick is None:
            continue
        try:
            source_tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        if source_tick % CAPTURE_STEP == 0:
            parent_capture_ticks.add(source_tick)
    previous_parent_tick = max(
        (source_tick for source_tick in parent_capture_ticks if source_tick < fork_tick),
        default=None,
    )
    previous_reference = None
    if previous_parent_tick is not None:
        pre_rows = overlay_trajectories(
            parent_traj,
            [],
            fork_tick=previous_parent_tick,
            end_tick=previous_parent_tick,
        )
        pre_weather = overlay_weather(
            parent_weather,
            [],
            fork_tick=previous_parent_tick,
            end_tick=previous_parent_tick,
        )
        pre_value, pre_observations, pre_bindings = evaluate_state(
            previous_parent_tick,
            pre_rows,
            pre_weather,
            state_arm_kind="factual",
            state_operation="",
        )
        pre_missing = list(pre_observations.get("missing_source_record") or [])
        last_bindings = last_bindings or pre_bindings
        previous_reference = {
            "tick": previous_parent_tick,
            "value": pre_value,
            "missing_source_record": pre_missing,
            "truth_id": f"parent_truth_ref:{_digest(episode_id, predicate_id, previous_parent_tick)}",
        }
    chain = ([previous_reference] if previous_reference is not None else []) + predicate_rows
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(chain, chain[1:]):
        if previous is None:
            continue
        from_value = str(previous.get("value"))
        to_value = str(current.get("value"))
        if from_value == to_value:
            continue
        if from_value == "out_of_scope" or to_value == "out_of_scope":
            continue
        direction = (
            "rising"
            if from_value == "false" and to_value == "true"
            else "falling"
            if from_value == "true" and to_value == "false"
            else "change"
        )
        transitions.append(
            {
                "schema_name": "predicate_transition",
                "schema_version": "3.0.0",
                "annotation_layer": "L1",
                "source_layer": "L0",
                "episode_id": episode_id,
                "predicate_id": predicate_id,
                "bindings": current.get("bindings") or last_bindings,
                "from_tick": int(previous["tick"]),
                "to_tick": int(current["tick"]),
                "from_value": from_value,
                "to_value": to_value,
                "direction": direction,
                "transition_id": f"transition:{_digest(episode_id, arm_id, predicate_id, previous['tick'], current['tick'], from_value, to_value)}",
                "tuple_id": current.get("tuple_id"),
                "evidence_truth_ids": [str(previous.get("truth_id")), str(current.get("truth_id"))],
                "materialization_policy": "branch_window_adjacent_capture_grid",
            }
        )
    rising = next(
        (
            item
            for item in transitions
            if item.get("direction") == "rising" and int(item["to_tick"]) >= fork_tick
        ),
        None,
    )
    falling = next(
        (
            item
            for item in transitions
            if item.get("direction") == "falling"
            and int(item["to_tick"]) >= fork_tick
            and (rising is None or int(item["to_tick"]) > int(rising["to_tick"]))
        ),
        None,
    )
    entry_active = bool(
        previous_reference
        and previous_reference.get("value") == "true"
        and predicate_rows
        and predicate_rows[0].get("value") == "true"
    )
    occurrence_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    unknown_source_records = sorted(
        {
            str(record)
            for row in predicate_rows
            for record in row.get("missing_source_record") or []
            if record
        }
    )
    if (
        previous_reference
        and previous_reference.get("value") == "unknown"
        and predicate_rows
        and predicate_rows[0].get("value") == "true"
    ):
        unknown_source_records = sorted(
            set(unknown_source_records)
            | {
                str(record)
                for record in previous_reference.get("missing_source_record") or []
                if record
            }
        )
    elif previous_reference is None and predicate_rows and predicate_rows[0].get("value") == "true":
        unknown_source_records = sorted(
            set(unknown_source_records)
            | {
                _missing_source_record(
                    "parent_capture/trajectories.jsonl",
                    None,
                    "pre_window_predicate_state",
                )
            }
        )
    if rising is not None and (not require_anchor_realization or anchor_realized):
        trigger_tick = int(rising["to_tick"])
        unknown_after_rising = any(
            row.get("value") == "unknown" and int(row["tick"]) >= trigger_tick
            for row in predicate_rows
        )
        terminal_tick = (
            int(falling["to_tick"])
            if falling is not None
            else None
            if unknown_after_rising
            else int(end_tick)
        )
        event_id = f"objective_event:{_digest(episode_id, arm_id, family, trigger_tick)}"
        occurrence_rows.append(
            {
                "schema_name": "event_occurrence",
                "schema_version": "3.0.0",
                "annotation_layer": "L2",
                "source_layer": "L1",
                "source_class": "deterministic_derived",
                "episode_id": episode_id,
                "event_family_id": family,
                "event_type_id": event_type,
                "rule_id": str(rule.get("rule_id") or rule_id),
                "trigger_predicate_id": predicate_id,
                "bindings": rising.get("bindings") or last_bindings,
                "event_id": event_id,
                "detection_tick": trigger_tick,
                "start_tick": trigger_tick,
                "trigger_tick": trigger_tick,
                "end_tick": terminal_tick,
                "evidence_tick": trigger_tick,
                "continuity_status": "strict_adjacent_capture_transition",
                "supporting_transition_ids": [str(rising["transition_id"])],
                "source_refs": [
                    f"objective/predicate_transitions.jsonl#transition={rising['transition_id']}",
                    f"raw/event_realization.jsonl#event={anchor_id}",
                ],
            }
        )
        outcome_rows.append(
            {
                "schema_name": "event_outcome",
                "schema_version": "3.0.0",
                "episode_id": episode_id,
                "event_family_id": family,
                "event_type_id": event_type,
                "event_id": event_id,
                "outcome_id": f"outcome:{_digest(event_id, terminal_tick)}",
                "status": "unknown" if terminal_tick is None else "succeeded",
                "lifecycle_status": (
                    "terminal_predicate_observed"
                    if falling is not None
                    else "terminal_predicate_unknown_in_window"
                    if terminal_tick is None
                    else "window_endpoint_observed"
                ),
                "terminal_tick": terminal_tick,
                "evidence_tick": terminal_tick,
                "missing_source_record": unknown_source_records if terminal_tick is None else [],
            }
        )
    elif entry_active and (not require_anchor_realization or anchor_realized):
        # The target state was already active in the immutable parent at the
        # branch boundary.  This is not a new Arm-local occurrence and must
        # never be relabelled as one.  Persist the continuation outcome with
        # an explicit parent tick reference so the delta can be spliced
        # without copying parent event rows.
        unknown_while_active = any(
            row.get("value") == "unknown" for row in predicate_rows
        )
        terminal_tick = (
            int(falling["to_tick"])
            if falling is not None
            else None
            if unknown_while_active
            else int(end_tick)
        )
        inherited_id = (
            f"objective_event_parent_active:{_digest(episode_id, arm_id, family, fork_tick)}"
        )
        outcome_rows.append(
            {
                "schema_name": "event_outcome",
                "schema_version": "3.0.0",
                "episode_id": episode_id,
                "event_family_id": family,
                "event_type_id": event_type,
                "event_id": inherited_id,
                "outcome_id": f"outcome:{_digest(inherited_id, terminal_tick)}",
                "status": "unknown" if terminal_tick is None else "succeeded",
                "lifecycle_status": (
                    "parent_active_recovery_observed"
                    if falling is not None
                    else "parent_active_state_unknown_in_window"
                    if terminal_tick is None
                    else "parent_active_through_window_endpoint"
                ),
                "terminal_tick": terminal_tick,
                "evidence_tick": terminal_tick,
                "entry_evidence": {
                    "parent_tick": int(previous_reference["tick"]),
                    "parent_value": "true",
                    "branch_tick": int(predicate_rows[0]["tick"]),
                    "branch_value": "true",
                    "parent_truth_ref": previous_reference.get("truth_id"),
                    "branch_truth_id": predicate_rows[0].get("truth_id"),
                },
                "missing_source_record": (
                    unknown_source_records if terminal_tick is None else []
                ),
            }
        )
    else:
        absence_id = f"objective_event_absence:{_digest(episode_id, arm_id, family, end_tick)}"
        event_unknown = bool(unknown_source_records)
        outcome_rows.append(
            {
                "schema_name": "event_outcome",
                "schema_version": "3.0.0",
                "episode_id": episode_id,
                "event_family_id": family,
                "event_type_id": event_type,
                "event_id": absence_id,
                "outcome_id": f"outcome:{_digest(absence_id, end_tick)}",
                "status": "unknown" if event_unknown else "not_observed",
                "lifecycle_status": (
                    "target_predicate_unknown_in_window"
                    if event_unknown
                    else "target_predicate_never_rose_in_window"
                ),
                "terminal_tick": None if event_unknown else int(end_tick),
                "evidence_tick": None if event_unknown else int(end_tick),
                "missing_source_record": unknown_source_records if event_unknown else [],
            }
        )
    objective = arm_root / "objective"
    objective.mkdir(parents=True, exist_ok=True)
    _write_jsonl(objective / "predicate_truth.jsonl", predicate_rows)
    _write_jsonl(objective / "predicate_transitions.jsonl", transitions)
    _write_jsonl(objective / "event_occurrences.jsonl", occurrence_rows)
    _write_jsonl(objective / "event_outcomes.jsonl", outcome_rows)
    true_ticks = [int(row["tick"]) for row in predicate_rows if row.get("value") == "true"]
    summary = {
        "schema_name": "aeroworld_arm_window_semantics",
        "schema_version": "1.0.0",
        "episode_id": episode_id,
        "arm_id": arm_id,
        "arm_kind": arm_kind,
        "event_family_id": family,
        "predicate_id": predicate_id,
        "event_type_id": event_type,
        "fork_tick": fork_tick,
        "window_end_tick": end_tick,
        "actor_ids": actor_ids,
        "predicate_row_count": len(predicate_rows),
        "predicate_true_count": len(true_ticks),
        "predicate_true_ticks": true_ticks,
        "first_true_tick": true_ticks[0] if true_ticks else None,
        "predicate_unknown_count": sum(
            row.get("value") == "unknown" for row in predicate_rows
        ),
        "predicate_out_of_scope_count": sum(
            row.get("value") == "out_of_scope" for row in predicate_rows
        ),
        "missing_source_record": unknown_source_records,
        "transition_count": len(transitions),
        "rising_tick": int(rising["to_tick"]) if rising is not None else None,
        "event_occurrence_count": len(occurrence_rows),
        "event_observed": bool(occurrence_rows) or entry_active,
        "event_window_state": (
            "local_onset_observed"
            if occurrence_rows
            else "active_at_window_entry"
            if entry_active
            else "unknown"
            if unknown_source_records
            else "not_observed"
        ),
        "event_status": (
            "observed"
            if (occurrence_rows or entry_active)
            and not any(row.get("status") == "unknown" for row in outcome_rows)
            else "unknown"
            if unknown_source_records
            else "not_observed"
        ),
        "g10_not_claimed": True,
        "training_eligible": False,
        "source": "window_local_state_recompute",
    }
    _write_json(objective / "window_semantics.json", summary)
    _write_branch_plot(
        arm_root / "branch_plot.png",
        branch_key=identity["branch_key"],
        predicate_id=predicate_id,
        fork_tick=fork_tick,
        end_tick=end_tick,
        predicate_rows=predicate_rows,
        occurrences=occurrence_rows,
        entry_active=entry_active,
    )
    graph_index = _write_window_graph_deltas(
        objective=objective,
        identity=identity,
        family=family,
        predicate_id=predicate_id,
        fork_tick=fork_tick,
        end_tick=end_tick,
        parent_objective=parent_objective,
        predicate_rows=predicate_rows,
        transitions=transitions,
        occurrences=occurrence_rows,
        outcomes=outcome_rows,
    )
    graph_index.update(
        {
            "window_semantics": True,
            "objective_skipped": False,
            "event_occurrence_count": len(occurrence_rows),
        }
    )
    _write_json(objective / "graph_index.json", graph_index)

    parent_reference = {
        "schema_name": "aeroworld_branch_parent_reference",
        "schema_version": "1.0.0",
        **identity,
        "objective_root": _project_relative(parent_objective),
        "capture_episode_root": _project_relative(parent_capture),
        "parent_tick_ref": {
            "episode_id": episode_id,
            "tick": fork_tick,
            "previous_capture_tick": previous_parent_tick,
        },
        "window": {"fork_tick": fork_tick, "window_end_tick": end_tick},
        "splice_policy": "parent_exact_tick_rows_overlaid_by_branch_delta",
        "parent_truth_copied": False,
    }
    _write_json(objective / "parent_reference.json", parent_reference)

    expected_ticks = capture_ticks(fork_tick, end_tick)
    closure_findings: list[dict[str, Any]] = []
    observed_ticks = [int(row["tick"]) for row in predicate_rows]
    if observed_ticks != expected_ticks:
        closure_findings.append(
            {
                "code": "predicate_capture_grid_mismatch",
                "expected_ticks": expected_ticks,
                "observed_ticks": observed_ticks,
            }
        )
    if summary["predicate_unknown_count"]:
        closure_findings.append(
            {
                "code": "unresolved_predicate_truth",
                "count": summary["predicate_unknown_count"],
                "missing_source_record": unknown_source_records,
            }
        )
    if len(outcome_rows) != 1:
        closure_findings.append(
            {"code": "event_outcome_cardinality", "count": len(outcome_rows)}
        )
    elif outcome_rows[0].get("evidence_tick") is None:
        closure_findings.append(
            {
                "code": "event_outcome_missing_evidence_tick",
                "missing_source_record": list(
                    outcome_rows[0].get("missing_source_record") or []
                ),
            }
        )
    if str(ledger.get("status") or "") not in {"applied", "executed", "reused"}:
        closure_findings.append(
            {
                "code": "branch_ledger_not_applied",
                "status": ledger.get("status"),
            }
        )
    if not event_type or not str(rule.get("rule_id") or rule_id):
        closure_findings.append(
            {
                "code": "unresolved_event_rule",
                "event_type_id": event_type,
                "rule_id": str(rule.get("rule_id") or rule_id),
            }
        )
    closure = {
        "schema_name": "aeroworld_arm_semantic_window_closure",
        "schema_version": "1.0.0",
        **identity,
        "event_family_id": family,
        "predicate_id": predicate_id,
        "rule_id": str(rule.get("rule_id") or rule_id),
        "event_type_id": event_type,
        "window": {"fork_tick": fork_tick, "window_end_tick": end_tick},
        "status": "PASS" if not closure_findings else "FAIL",
        "findings": closure_findings,
        "scope": "arm_local_semantic_window_only",
        "g7_group_comparison_required": True,
        "g10_not_claimed": True,
        "training_eligible": False,
    }
    _write_json(objective / "window_closure.json", closure)

    artifact_rows = {
        "predicate_truth.jsonl": len(predicate_rows),
        "predicate_transitions.jsonl": len(transitions),
        "event_occurrences.jsonl": len(occurrence_rows),
        "event_outcomes.jsonl": len(outcome_rows),
        "world_truth_graph_deltas.jsonl": graph_index["world_truth_delta_rows"],
        "semantic_graph_deltas.jsonl": graph_index["semantic_delta_rows"],
    }
    objective_manifest = {
        "schema_name": "aeroworld_arm_window_objective_manifest",
        "schema_version": "1.0.0",
        **identity,
        "event_family_id": family,
        "predicate_id": predicate_id,
        "rule_id": str(rule.get("rule_id") or rule_id),
        "event_type_id": event_type,
        "window": {"fork_tick": fork_tick, "window_end_tick": end_tick},
        "artifact_scope": "parent_reference_plus_local_semantic_delta",
        "artifacts": artifact_rows,
        "branch_plot": "../branch_plot.png",
        "parent_reference": "parent_reference.json",
        "graph_index": "graph_index.json",
        "closure": {
            "path": "window_closure.json",
            "scope": closure["scope"],
            "status": closure["status"],
        },
        "g10_not_claimed": True,
        "training_eligible": False,
    }
    _write_json(objective / "manifest.json", objective_manifest)
    _write_json(objective / "objective_index.json", objective_manifest)
    return summary
