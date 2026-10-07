"""Materialize authoritative L0 scope and SUMO road state for semantics.

Render-ready truth intentionally filters off-ROI traffic.  Predicate truth may
not inherit that rendering filter, so the strict semantic pipeline reconstructs
all selected SUMO vehicle lifecycles from the episode-local TraCI output here.
The formal render-ready source files are never modified.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from Dataset.semantic_simulation.domain_state import (
    OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES,
)
from Dataset.semantic_truth.l0_state_profile import (
    DEFAULT_L0_STATE_PROFILE_PATH,
    load_l0_state_profile,
)
from Dataset.semantic_truth.provenance import digest_file, digest_object, read_jsonl
from Dataset.tools.sumo_ground_flow.road_signal_context import (
    DEFAULT_ROAD_DERIVED_NET_XML,
    RoadSignalContext,
)
from Dataset.semantic_truth.episode_sources import source_sumo_frames
from Dataset.semantic_truth.facility_scope import ontology_class_is_a


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_SOURCE_TICKS = tuple(range(0, 901))
FORMAL_SEMANTIC_TICKS = frozenset(range(0, 901, 5))
ORIENTATION_INVARIANT_STATIC_CATEGORIES = frozenset(
    {
        "airspace_constraint",
        "airspace_corridor",
        "crowd_anchor",
        "facade_anchor",
        "facility",
        "ground_station",
        "hazard_zone",
        "prop",
        "traffic_light",
        "vehicle_anchor",
    }
)


class L0SupplementError(ValueError):
    """Raised when an authoritative L0 source cannot be reconstructed."""


def build_l0_source_availability(
    episode_root: Path, *, net_xml: Path = DEFAULT_ROAD_DERIVED_NET_XML,
) -> dict[str, Any]:
    """Derive the predicate source contract from the current episode inputs."""

    episode_root = Path(episode_root).resolve()
    manifest = _load_object(episode_root / "episode_manifest.json")
    if manifest.get("episode_id") != episode_root.name:
        raise L0SupplementError(
            f"manifest episode_id differs from directory {episode_root.name!r}"
        )
    roster_path = episode_root / "global_entity_roster.json"
    roster = _load_object(roster_path)
    roster_by_id = _index_roster(roster.get("entities"), roster_path)
    return _build_source_availability(
        episode_root=episode_root,
        manifest=manifest,
        roster_by_id=roster_by_id,
        road_context=RoadSignalContext.load(net_xml),
        profile=load_l0_state_profile(),
    )


def materialize_episode_l0_state(
    episode_root: Path, *, net_xml: Path = DEFAULT_ROAD_DERIVED_NET_XML,
) -> dict[str, Any]:
    """Add full SUMO lifecycles, road rules, and explicit scope activity."""

    episode_root = Path(episode_root).resolve()
    manifest_path = episode_root / "episode_manifest.json"
    roster_path = episode_root / "global_entity_roster.json"
    frames_path = episode_root / "truth_frames.jsonl"
    trajectories_path = episode_root / "trajectories.jsonl"
    weather_path = episode_root / "weather_meta.jsonl"
    for path in (
        manifest_path,
        roster_path,
        frames_path,
        trajectories_path,
        weather_path,
    ):
        if not path.is_file():
            raise L0SupplementError(f"required strict L0 input is missing: {path}")

    manifest = _load_object(manifest_path)
    episode_id = str(manifest.get("episode_id") or episode_root.name)
    if episode_id != episode_root.name:
        raise L0SupplementError(
            f"manifest episode_id {episode_id!r} differs from {episode_root.name!r}"
        )
    scenario_id = str(manifest.get("scenario_id") or episode_id.split("__seed", 1)[0])
    profile = load_l0_state_profile()
    crosswalk_by_pedestrian = _scene_crosswalk_bindings(manifest)
    roster = _load_object(roster_path)
    roster_by_id = _index_roster(roster.get("entities"), roster_path)
    vehicle_entity_by_sumo_id = _sumo_vehicle_roster_index(roster_by_id)
    trajectories_by_tick = _index_formal_trajectories(
        trajectories_path,
        roster_by_id,
        frozenset(vehicle_entity_by_sumo_id.values()),
    )

    frames = sorted(read_jsonl(frames_path), key=lambda row: int(row.get("tick", -1)))
    if [row.get("tick") for row in frames] != list(FORMAL_SOURCE_TICKS):
        raise L0SupplementError("strict truth frames must contain exact ticks 0..900")
    first_frame = frames[0]
    tick_hz = _finite_number(first_frame.get("tick_hz"))
    segment = first_frame.get("sumo_segment")
    segment_start_s = (
        _finite_number(segment.get("segment_start_s"))
        if isinstance(segment, Mapping)
        else None
    )
    if tick_hz is None or segment_start_s is None:
        raise L0SupplementError(
            "L0 SUMO authority requires truth_frames tick_hz and sumo_segment.start"
        )

    raw_frames: dict[int, dict[str, Any]] = {}
    raw_frames_path = source_sumo_frames(episode_root)
    if not raw_frames_path.is_file():
        raise L0SupplementError(
            f"episode-local SUMO authority is missing: {raw_frames_path}"
        )
    raw_frames = _load_episode_sumo_window(
        raw_frames_path,
        segment_start_s=float(segment_start_s),
        tick_hz=float(tick_hz),
        selected_vehicle_ids=frozenset(vehicle_entity_by_sumo_id),
    )

    road_context = RoadSignalContext.load(net_xml)
    weather_temperature_count = _materialize_weather_temperature(
        weather_path,
        profile,
    )
    previous_vehicle_by_entity: dict[str, Mapping[str, Any]] = {}
    previous_signals: Mapping[str, Mapping[str, Any]] | None = None
    active_vehicle_tick_count = 0
    active_uav_tick_count = 0
    trajectory_entity_tick_count = 0
    regulation_field_count = 0
    raw_vehicle_ids_seen: set[str] = set()
    pending_crossing_by_entity: dict[str, dict[str, Any]] = {}
    render_vehicle_tick_removed_count = 0
    render_vehicle_tick_pose_reconciled_count = 0
    render_vehicle_authority_conflicts: list[dict[str, Any]] = []
    runtime_state_field_normalization_count = 0

    for frame in frames:
        tick = int(frame["tick"])
        entities = frame.get("entities")
        if not isinstance(entities, list):
            raise L0SupplementError(f"truth frame {tick} lacks entities array")
        if tick in FORMAL_SEMANTIC_TICKS:
            existing_entity_ids = {
                str(entity.get("entity_id"))
                for entity in entities
                if isinstance(entity, Mapping)
                and isinstance(entity.get("entity_id"), str)
            }
            for entity_id, trajectory in trajectories_by_tick.get(tick, {}).items():
                if entity_id in existing_entity_ids:
                    continue
                entities.append(
                    _semantic_only_trajectory_entity(
                        roster_by_id[entity_id],
                        trajectory,
                        tick=tick,
                    )
                )
                existing_entity_ids.add(entity_id)
                trajectory_entity_tick_count += 1
        for entity in entities:
            runtime_state_field_normalization_count += (
                _normalize_l0_runtime_state_fields(entity, tick=tick)
            )
            entity_id = entity.get("entity_id")
            if isinstance(entity_id, str) and entity_id in crosswalk_by_pedestrian:
                crosswalk = crosswalk_by_pedestrian[entity_id]
                pedestrian_state = entity.setdefault("pedestrian_state", {})
                if not isinstance(pedestrian_state, dict):
                    raise L0SupplementError(
                        f"tick {tick}: {entity_id!r}.pedestrian_state must be an object"
                    )
                existing_crosswalk = pedestrian_state.get("crosswalk_id")
                if existing_crosswalk not in (None, crosswalk["crosswalk_id"]):
                    raise L0SupplementError(
                        f"tick {tick}: {entity_id!r} crosswalk identity conflicts with scene authority"
                    )
                existing_class = pedestrian_state.get("crosswalk_ontology_class_id")
                if existing_class is not None and (
                    not isinstance(existing_class, str)
                    or not ontology_class_is_a(existing_class, crosswalk["crosswalk_ontology_class_id"])
                ):
                    raise L0SupplementError(
                        f"tick {tick}: {entity_id!r} crosswalk class conflicts with crossing authority"
                    )
                pedestrian_state.update(crosswalk)
                _materialize_crosswalk_geometry(
                    entity, road_context=road_context, tick=tick,
                )
        entity_by_id = {
            str(entity["entity_id"]): entity
            for entity in entities
            if isinstance(entity, dict) and isinstance(entity.get("entity_id"), str)
        }
        if len(entity_by_id) != len(entities):
            raise L0SupplementError(
                f"truth frame {tick} has invalid/duplicate entities"
            )

        raw_frame = raw_frames.get(tick, {"vehicles": [], "traffic_lights": []})
        signal_states = _traffic_light_states(raw_frame)
        raw_vehicle_by_id = {
            str(vehicle["vehicle_id"]): vehicle
            for vehicle in raw_frame.get("vehicles", ())
            if isinstance(vehicle, Mapping)
            and isinstance(vehicle.get("vehicle_id"), str)
            and vehicle.get("vehicle_id") in vehicle_entity_by_sumo_id
        }
        raw_vehicle_ids_seen.update(raw_vehicle_by_id)

        removed = _remove_render_sumo_vehicles_absent_from_authority(
            entities,
            raw_vehicle_ids=frozenset(raw_vehicle_by_id),
            vehicle_entity_by_sumo_id=vehicle_entity_by_sumo_id,
            tick=tick,
        )
        render_vehicle_tick_removed_count += len(removed)
        render_vehicle_authority_conflicts.extend(
            {
                "conflict": "render_presence_without_episode_local_sumo_presence",
                "entity_id": entity_id,
                "sumo_vehicle_id": sumo_id,
                "tick": tick,
            }
            for entity_id, sumo_id in removed
        )
        entity_by_id = {
            str(entity["entity_id"]): entity
            for entity in entities
            if isinstance(entity, dict) and isinstance(entity.get("entity_id"), str)
        }

        active_vehicle_ids: list[str] = []
        for sumo_id, vehicle in sorted(raw_vehicle_by_id.items()):
            entity_id = vehicle_entity_by_sumo_id[sumo_id]
            roster_entity = roster_by_id[entity_id]
            sumo_state = _sumo_vehicle_state(roster_entity, vehicle)
            previous = previous_vehicle_by_entity.get(entity_id)
            road_fields = road_context.enrich_sumo_vehicle(
                sumo_state,
                signal_states,
                previous_sumo_vehicle=previous,
                previous_traffic_light_states=previous_signals,
            )
            sumo_state.update(road_fields)
            if road_fields["crossed_stop_line"] is True:
                pending_crossing_by_entity[entity_id] = copy.deepcopy(road_fields)
            previous_vehicle_by_entity[entity_id] = copy.deepcopy(sumo_state)
            entity = entity_by_id.get(entity_id)
            if entity is None:
                entity = _semantic_only_vehicle_entity(
                    roster_entity,
                    vehicle,
                    tick=tick,
                )
                entities.append(entity)
                entity_by_id[entity_id] = entity
            else:
                conflict = _apply_episode_local_sumo_pose_authority(
                    entity,
                    roster_entity,
                    vehicle,
                    tick=tick,
                )
                render_vehicle_tick_pose_reconciled_count += 1
                if conflict:
                    render_vehicle_authority_conflicts.append(
                        {
                            "conflict": "render_pose_or_lane_differs_from_episode_local_sumo",
                            "entity_id": entity_id,
                            "sumo_vehicle_id": sumo_id,
                            "tick": tick,
                        }
                    )
            if tick in FORMAL_SEMANTIC_TICKS:
                interval_crossing = pending_crossing_by_entity.pop(entity_id, None)
                if interval_crossing is not None:
                    sumo_state.update(
                        {
                            "controlling_signal_id": interval_crossing[
                                "controlling_signal_id"
                            ],
                            "controlling_signal_state": interval_crossing[
                                "controlling_signal_state"
                            ],
                            "crossed_stop_line": True,
                        }
                    )
            entity["sumo_vehicle"] = sumo_state
            entity["l0_scope_state"] = {
                "active": True,
                "authority": "episode_local_sumo_traci",
                "source_tick": tick,
            }
            if tick not in FORMAL_SEMANTIC_TICKS:
                continue
            active_vehicle_ids.append(entity_id)
            active_vehicle_tick_count += 1
            regulation_field_count += len(road_fields)

        if tick not in FORMAL_SEMANTIC_TICKS:
            previous_signals = signal_states
            continue

        _materialize_following_vehicle_relations(
            entity_by_id,
            active_vehicle_ids,
            tick=tick,
        )

        active_uav_ids = sorted(
            entity_id
            for entity_id, entity in entity_by_id.items()
            if _entity_category(entity) == "uav"
        )
        for entity_id in active_uav_ids:
            entity_by_id[entity_id]["l0_scope_state"] = {
                "active": True,
                "authority": "truth_frames.entities",
                "source_tick": tick,
            }
        active_uav_tick_count += len(active_uav_ids)
        frame["sumo_traffic_light_states"] = signal_states
        frame["l0_scope_activity"] = {
            "active_uav_ids": active_uav_ids,
            "active_vehicle_ids": sorted(active_vehicle_ids),
            "inactive_uav_ids": sorted(
                entity_id
                for entity_id, roster_entity in roster_by_id.items()
                if _entity_category(roster_entity) == "uav"
                and entity_id not in active_uav_ids
            ),
            "inactive_vehicle_ids": sorted(
                entity_id
                for entity_id, roster_entity in roster_by_id.items()
                if _entity_category(roster_entity) == "vehicle"
                and entity_id not in active_vehicle_ids
            ),
            "authority": "truth_frames_plus_episode_local_sumo_traci",
        }
        entities.sort(key=lambda entity: str(entity.get("entity_id") or ""))
        previous_signals = signal_states

    never_active_roster_vehicles = sorted(
        set(vehicle_entity_by_sumo_id) - raw_vehicle_ids_seen
    )

    source_availability = _build_source_availability(
        episode_root=episode_root,
        manifest=manifest,
        roster_by_id=roster_by_id,
        road_context=road_context,
        profile=profile,
    )
    availability_path = episode_root / "l0_predicate_source_availability.json"
    availability_path.write_text(
        json.dumps(
            source_availability,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_jsonl(frames_path, frames)
    return {
        "schema_name": "l0_state_materialization_summary",
        "schema_version": "1.0.0",
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "formal_source_tick_count": len(FORMAL_SOURCE_TICKS),
        "formal_semantic_tick_count": len(FORMAL_SEMANTIC_TICKS),
        "active_vehicle_tick_count": active_vehicle_tick_count,
        "active_uav_tick_count": active_uav_tick_count,
        "trajectory_entity_tick_count": trajectory_entity_tick_count,
        "regulation_field_count": regulation_field_count,
        "weather_temperature_count": weather_temperature_count,
        "source_unavailability_count": len(source_availability["entries"]),
        "sumo_roster_vehicle_count": len(vehicle_entity_by_sumo_id),
        "sumo_vehicle_ids_seen": len(raw_vehicle_ids_seen),
        "sumo_roster_never_active_count": len(never_active_roster_vehicles),
        "sumo_roster_never_active_digest": digest_object(never_active_roster_vehicles),
        "render_vehicle_tick_removed_count": render_vehicle_tick_removed_count,
        "render_vehicle_tick_pose_reconciled_count": (
            render_vehicle_tick_pose_reconciled_count
        ),
        "render_vehicle_authority_conflict_count": len(
            render_vehicle_authority_conflicts
        ),
        "render_vehicle_authority_conflict_digest": digest_object(
            render_vehicle_authority_conflicts
        ),
        "runtime_state_field_normalization_count": (
            runtime_state_field_normalization_count
        ),
        "road_lane_count": len(road_context.lanes),
        "road_traffic_light_controller_count": road_context.traffic_light_controller_count,
        "road_signalized_junction_count": road_context.signalized_junction_count,
        "road_crossing_count": road_context.crossing_count,
        "l0_state_profile_id": profile["profile_id"],
        "l0_state_profile_sha256": digest_file(DEFAULT_L0_STATE_PROFILE_PATH),
        "sumo_authority_digest": (
            digest_file(raw_frames_path)
            if raw_frames_path.is_file()
            else "not_applicable"
        ),
        "state_digest": digest_object(
            {
                "active_vehicle_tick_count": active_vehicle_tick_count,
                "active_uav_tick_count": active_uav_tick_count,
                "trajectory_entity_tick_count": trajectory_entity_tick_count,
                "regulation_field_count": regulation_field_count,
                "weather_temperature_count": weather_temperature_count,
                "render_vehicle_tick_removed_count": (
                    render_vehicle_tick_removed_count
                ),
                "render_vehicle_tick_pose_reconciled_count": (
                    render_vehicle_tick_pose_reconciled_count
                ),
                "render_vehicle_authority_conflicts": (
                    render_vehicle_authority_conflicts
                ),
                "sumo_roster_never_active_vehicle_ids": (never_active_roster_vehicles),
                "runtime_state_field_normalization_count": (
                    runtime_state_field_normalization_count
                ),
                "source_unavailability_entries": source_availability["entries"],
            }
        ),
    }


def _materialize_weather_temperature(
    path: Path,
    profile: Mapping[str, Any],
) -> int:
    rows = sorted(read_jsonl(path), key=lambda row: int(row.get("tick", -1)))
    if [row.get("tick") for row in rows] != list(FORMAL_SOURCE_TICKS):
        raise L0SupplementError(
            "strict weather metadata must contain exact ticks 0..900"
        )
    temperature_by_condition = profile["weather_temperature_c_by_condition"]
    materialized = 0
    for row in rows:
        existing = _finite_number(row.get("temperature_c"))
        if existing is not None:
            continue
        condition = str(row.get("condition") or "").strip().lower()
        temperature = temperature_by_condition.get(condition)
        if temperature is None:
            raise L0SupplementError(
                f"weather condition {condition!r} lacks a governed temperature"
            )
        row["temperature_c"] = float(temperature)
        row["temperature_source"] = (
            "Dataset/semantic_rules/profiles/l0_state_supplement_profile.json"
            f"#weather_temperature_c_by_condition.{condition}"
        )
        materialized += 1
    _write_jsonl(path, rows)
    return materialized


def _build_source_availability(
    *,
    episode_root: Path,
    manifest: Mapping[str, Any],
    roster_by_id: Mapping[str, Mapping[str, Any]],
    road_context: RoadSignalContext,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    contracts = profile["source_unavailability_contracts"]
    entries: list[dict[str, Any]] = []

    def append_contract(contract_id: str, entity_ids: list[str]) -> None:
        contract = contracts[contract_id]
        for entity_id in sorted(entity_ids):
            for predicate_id in sorted(contract["predicate_ids"]):
                entries.append(
                    {
                        "contract_id": contract_id,
                        "predicate_id": predicate_id,
                        "scope_type": contract["scope_type"],
                        "scope_entity_id": entity_id,
                        "executable": False,
                        "non_executable_reason": contract["non_executable_reason"],
                        "closure_statement": contract["closure_statement"],
                        "source_ref": (
                            "Dataset/semantic_rules/profiles/"
                            "l0_state_supplement_profile.json"
                            f"#source_unavailability_contracts.{contract_id}"
                        ),
                    }
                )

    if road_context.crossing_count == 0:
        append_contract(
            "pedestrian_crosswalk_geometry",
            [
                entity_id
                for entity_id, entity in roster_by_id.items()
                if _entity_category(entity) == "pedestrian"
            ],
        )
    uav_ids = [
        entity_id
        for entity_id, entity in roster_by_id.items()
        if _entity_category(entity) == "uav"
    ]
    if not _episode_has_bound_building_geometry(episode_root, manifest):
        append_contract("uav_building_geometry", uav_ids)
    append_contract(
        "uav_pad_assignment",
        [
            entity_id
            for entity_id in uav_ids
            if not _uav_has_pad_assignment(roster_by_id[entity_id])
        ],
    )
    entries.sort(
        key=lambda row: (
            str(row["scope_type"]),
            str(row["scope_entity_id"]),
            str(row["predicate_id"]),
        )
    )
    return {
        "schema_name": "l0_predicate_source_availability",
        "schema_version": "1.0.0",
        "episode_id": episode_root.name,
        "profile_id": profile["profile_id"],
        "road_crossing_count": road_context.crossing_count,
        "entries": entries,
    }


def _uav_has_pad_assignment(entity: Mapping[str, Any]) -> bool:
    lifecycle = entity.get("lifecycle")
    if isinstance(lifecycle, Mapping):
        home_pad = lifecycle.get("home_pad_entity_id")
        if isinstance(home_pad, str) and home_pad:
            return True
    global_flow = entity.get("uav_global_flow")
    return bool(
        isinstance(global_flow, Mapping)
        and any(
            isinstance(global_flow.get(field), str) and global_flow.get(field)
            for field in ("origin_pad_id", "target_pad_id")
        )
    )


def _episode_has_bound_building_geometry(
    episode_root: Path,
    manifest: Mapping[str, Any],
) -> bool:
    candidates = [episode_root / "semantic_static_geometry.json"]
    source = manifest.get("source_scene_setup_path")
    if isinstance(source, str) and source:
        candidates.append(PROJECT_ROOT / source.replace("\\", "/"))
    for path in candidates:
        if not path.is_file():
            continue
        value = _load_object(path)
        raw_entities = value.get("entities") or value.get("static_entities") or []
        entities = (
            list(raw_entities.values())
            if isinstance(raw_entities, Mapping)
            else raw_entities
        )
        if not isinstance(entities, list):
            raise L0SupplementError(
                f"{path}: building geometry entities must be an array"
            )
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            placement = entity.get("placement")
            category = str(
                entity.get("category") or entity.get("entity_category") or ""
            )
            if category in {"facade_anchor", "building", "structure"}:
                return True
            if isinstance(placement, Mapping) and any(
                key in placement for key in ("building_id", "building_source")
            ):
                return True
    return False


def _index_formal_trajectories(
    path: Path,
    roster_by_id: Mapping[str, Mapping[str, Any]],
    sumo_entity_ids: frozenset[str],
) -> dict[int, dict[str, Mapping[str, Any]]]:
    """Index non-SUMO entity state on semantic ticks from all-entity truth."""

    result: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in read_jsonl(path):
        tick = row.get("tick")
        entity_id = row.get("entity_id")
        if tick not in FORMAL_SEMANTIC_TICKS:
            continue
        if not isinstance(entity_id, str) or entity_id not in roster_by_id:
            raise L0SupplementError(
                f"{path}: formal trajectory references an absent roster entity: {entity_id!r}"
            )
        if entity_id in sumo_entity_ids:
            continue
        rows_at_tick = result.setdefault(int(tick), {})
        if entity_id in rows_at_tick:
            raise L0SupplementError(
                f"{path}: duplicate formal trajectory row {(tick, entity_id)}"
            )
        rows_at_tick[entity_id] = row
    _annotate_temporal_yaw_authority(result, roster_by_id)
    return result


def _annotate_temporal_yaw_authority(
    rows_by_tick: Mapping[int, Mapping[str, Mapping[str, Any]]],
    roster_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    rows_by_entity: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for tick, rows in sorted(rows_by_tick.items()):
        for entity_id, raw_row in sorted(rows.items()):
            if not isinstance(raw_row, dict):
                raise L0SupplementError(
                    f"trajectory row {(tick, entity_id)} is not mutable"
                )
            rows_by_entity.setdefault(entity_id, []).append((tick, raw_row))

    for entity_id, timed_rows in sorted(rows_by_entity.items()):
        roster_entity = roster_by_id[entity_id]
        for _tick, row in timed_rows:
            yaw = _finite_number(row.get("yaw_deg"))
            source = "trajectory.yaw_deg"
            if yaw is None:
                position = _vector3(row.get("pos_enu"))
                velocity = _vector3(row.get("vel_mps"))
                if position is not None and velocity is not None:
                    yaw = _derive_trajectory_yaw(position, velocity, roster_entity)
                    source = "velocity_or_planned_route_geometry"
            if yaw is not None:
                row["_l0_resolved_yaw_deg"] = yaw
                row["_l0_resolved_yaw_source"] = source

        for index, (_tick, row) in enumerate(timed_rows):
            if _finite_number(row.get("_l0_resolved_yaw_deg")) is not None:
                continue
            position = _vector3(row.get("pos_enu"))
            if position is None:
                continue
            adjacent_positions: list[tuple[list[float], list[float]]] = []
            if index > 0:
                previous = _vector3(timed_rows[index - 1][1].get("pos_enu"))
                if previous is not None:
                    adjacent_positions.append((previous, position))
            if index + 1 < len(timed_rows):
                following = _vector3(timed_rows[index + 1][1].get("pos_enu"))
                if following is not None:
                    adjacent_positions.append((position, following))
            for origin, target in adjacent_positions:
                dx = target[0] - origin[0]
                dy = target[1] - origin[1]
                if math.hypot(dx, dy) <= 1e-6:
                    continue
                row["_l0_resolved_yaw_deg"] = math.degrees(math.atan2(dy, dx))
                row["_l0_resolved_yaw_source"] = "semantic_trajectory_displacement"
                break

        prior_yaw: float | None = None
        for _tick, row in timed_rows:
            yaw = _finite_number(row.get("_l0_resolved_yaw_deg"))
            if yaw is not None:
                prior_yaw = yaw
                continue
            if prior_yaw is not None:
                row["_l0_resolved_yaw_deg"] = prior_yaw
                row["_l0_resolved_yaw_source"] = "prior_semantic_trajectory_heading"

        subsequent_yaw: float | None = None
        for _tick, row in reversed(timed_rows):
            yaw = _finite_number(row.get("_l0_resolved_yaw_deg"))
            if yaw is not None:
                subsequent_yaw = yaw
                continue
            if subsequent_yaw is not None:
                row["_l0_resolved_yaw_deg"] = subsequent_yaw
                row["_l0_resolved_yaw_source"] = (
                    "subsequent_semantic_trajectory_heading"
                )

        if all(
            _finite_number(row.get("_l0_resolved_yaw_deg")) is None
            for _tick, row in timed_rows
        ) and _is_orientation_invariant_stationary_vehicle(
            timed_rows,
            roster_entity,
        ):
            for _tick, row in timed_rows:
                row["_l0_resolved_yaw_deg"] = 0.0
                row["_l0_resolved_yaw_source"] = (
                    "stationary_point_trajectory_orientation_contract"
                )


def _is_orientation_invariant_stationary_vehicle(
    timed_rows: list[tuple[int, dict[str, Any]]],
    roster_entity: Mapping[str, Any],
) -> bool:
    if _entity_category(roster_entity) != "vehicle" or not timed_rows:
        return False
    positions = [_vector3(row.get("pos_enu")) for _tick, row in timed_rows]
    velocities = [_vector3(row.get("vel_mps")) for _tick, row in timed_rows]
    if any(position is None for position in positions) or any(
        velocity is None for velocity in velocities
    ):
        return False
    origin = positions[0]
    if origin is None:
        return False
    return all(
        position is not None and math.dist(origin, position) <= 1e-6
        for position in positions
    ) and all(
        velocity is not None
        and math.sqrt(sum(component * component for component in velocity)) <= 1e-6
        for velocity in velocities
    )


def _semantic_only_trajectory_entity(
    roster_entity: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    tick: int,
) -> dict[str, Any]:
    """Project all-entity trajectory truth back into a semantic-only frame cell."""

    position = _vector3(trajectory.get("pos_enu"))
    velocity = _vector3(trajectory.get("vel_mps"))
    yaw = _finite_number(trajectory.get("_l0_resolved_yaw_deg"))
    yaw_source = str(trajectory.get("_l0_resolved_yaw_source") or "trajectory.yaw_deg")
    if yaw is None:
        yaw = _finite_number(trajectory.get("yaw_deg"))
    category = _entity_category(trajectory) or _entity_category(roster_entity)
    if position is None or velocity is None:
        raise L0SupplementError(
            f"trajectory entity {trajectory.get('entity_id')!r}@{tick} lacks pose truth"
        )
    if yaw is None:
        yaw = _derive_trajectory_yaw(position, velocity, roster_entity)
        yaw_source = "velocity_or_planned_route_geometry"
    if yaw is None and category in ORIENTATION_INVARIANT_STATIC_CATEGORIES:
        yaw = 0.0
        yaw_source = "orientation_invariant_static_entity_contract"
    if yaw is None:
        raise L0SupplementError(
            f"trajectory entity {trajectory.get('entity_id')!r}@{tick} lacks yaw authority"
        )
    entity = copy.deepcopy(dict(roster_entity))
    entity_id = str(roster_entity["entity_id"])
    entity.update(
        {
            "entity_id": entity_id,
            "entity_category": category,
            "category": category,
            "label_class": str(
                trajectory.get("label_class")
                or roster_entity.get("label_class")
                or category
            ),
            "entity_kind": str(
                trajectory.get("entity_kind")
                or roster_entity.get("entity_kind")
                or category
            ),
            "entity_type": str(
                trajectory.get("entity_type")
                or roster_entity.get("entity_type")
                or category
            ),
            "truth_pose": {
                "authority_mode": "authoritative_input",
                "authority_owner": "all_entity_trajectory_truth",
                "coordinate_contract_id": "coord.external_enu_m.v1",
                "position_enu_m": position,
                "rotation_deg": {
                    "pitch_deg": 0.0,
                    "roll_deg": 0.0,
                    "yaw_deg": yaw,
                },
                "yaw_source": yaw_source,
                "velocity_enu_mps": velocity,
            },
            "render_presence": {
                "global_roster": True,
                "offstage": True,
                "offstage_reason": "outside_render_runtime_boundary",
                "roi_membership": [],
                "submission_state": "semantic_truth_only",
                "visibility_state": "not_submitted",
            },
            "source": str(trajectory.get("source") or "all_entity_trajectory_truth"),
            "state_revision": tick + 1,
        }
    )
    for family in OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES:
        value = trajectory.get(family)
        if isinstance(value, Mapping):
            entity[family] = copy.deepcopy(dict(value))
    return entity


def _derive_trajectory_yaw(
    position: list[float],
    velocity: list[float],
    roster_entity: Mapping[str, Any],
) -> float | None:
    speed_xy = math.hypot(velocity[0], velocity[1])
    if speed_xy > 1e-6:
        return math.degrees(math.atan2(velocity[1], velocity[0]))
    route = roster_entity.get("route_waypoints_enu_m")
    if not isinstance(route, list):
        return None
    for waypoint in route:
        target = _vector3(waypoint)
        if target is None:
            continue
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        if math.hypot(dx, dy) > 1e-6:
            return math.degrees(math.atan2(dy, dx))
    return None


def _load_episode_sumo_window(
    path: Path,
    *,
    segment_start_s: float,
    tick_hz: float,
    selected_vehicle_ids: frozenset[str],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    segment_end_s = segment_start_s + FORMAL_SOURCE_TICKS[-1] / tick_hz
    tolerance_s = 0.5 / tick_hz
    for row in read_jsonl(path):
        sim_time_s = _finite_number(row.get("sim_time_s"))
        if sim_time_s is None:
            raise L0SupplementError(f"{path}: SUMO frame lacks sim_time_s")
        if sim_time_s < segment_start_s - tolerance_s:
            continue
        if sim_time_s > segment_end_s + tolerance_s:
            break
        tick_float = (sim_time_s - segment_start_s) * tick_hz
        tick = int(round(tick_float))
        if tick not in FORMAL_SOURCE_TICKS or not math.isclose(
            tick_float, tick, rel_tol=0.0, abs_tol=1e-5
        ):
            continue
        if tick in result:
            raise L0SupplementError(f"{path}: duplicate episode-local SUMO tick {tick}")
        result[tick] = {
            "vehicles": [
                _compact_raw_vehicle(vehicle)
                for vehicle in row.get("vehicles", ())
                if isinstance(vehicle, Mapping)
                and vehicle.get("vehicle_id") in selected_vehicle_ids
            ],
            "traffic_lights": [
                {
                    "tls_id": item.get("tls_id"),
                    "state": item.get("state"),
                    "phase_index": item.get("phase_index"),
                    "program_id": item.get("program_id"),
                    "next_switch_s": item.get("next_switch_s"),
                    "controlled_links": copy.deepcopy(
                        item.get("controlled_links") or []
                    ),
                }
                for item in row.get("traffic_lights", ())
                if isinstance(item, Mapping)
            ],
        }
    missing = sorted(set(FORMAL_SOURCE_TICKS) - set(result))
    if missing:
        raise L0SupplementError(
            f"{path}: SUMO authority window is missing local ticks {missing[:20]}"
        )
    return result


def _traffic_light_states(frame: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in frame.get("traffic_lights", ()):
        if not isinstance(item, Mapping):
            continue
        tls_id = item.get("tls_id")
        state = item.get("state")
        if not isinstance(tls_id, str) or not tls_id or not isinstance(state, str):
            raise L0SupplementError("SUMO traffic-light row lacks tls_id/state")
        result[tls_id] = {
            "state": state,
            "phase_index": int(item.get("phase_index") or 0),
            "program_id": str(item.get("program_id") or ""),
            "next_switch_s": round(float(item.get("next_switch_s") or 0.0), 6),
            "controlled_links": copy.deepcopy(item.get("controlled_links") or []),
        }
    return dict(sorted(result.items()))


def _sumo_vehicle_roster_index(
    roster_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for entity_id, entity in sorted(roster_by_id.items()):
        if _entity_category(entity) != "vehicle":
            continue
        sumo = entity.get("sumo_vehicle")
        if not isinstance(sumo, Mapping):
            continue
        vehicle_id = sumo.get("vehicle_id")
        if not isinstance(vehicle_id, str) or not vehicle_id:
            raise L0SupplementError(
                f"vehicle roster entity {entity_id!r} lacks SUMO authority identity"
            )
        if vehicle_id in result:
            raise L0SupplementError(f"duplicate roster SUMO vehicle id {vehicle_id!r}")
        result[vehicle_id] = entity_id
    return result


def _sumo_vehicle_state(
    roster_entity: Mapping[str, Any],
    vehicle: Mapping[str, Any],
) -> dict[str, Any]:
    roster_state = roster_entity.get("sumo_vehicle")
    result: dict[str, Any] = {}
    if isinstance(roster_state, Mapping):
        for key in (
            "vehicle_id",
            "source_entity_id",
            "semantic_episode_id",
            "semantic_vehicle",
            "canonical_logical_asset_id",
            "logical_asset_authority",
            "vehicle_type",
            "route_id",
            "control_role",
        ):
            if key in roster_state:
                result[key] = copy.deepcopy(roster_state[key])
    for key in (
        "vehicle_id",
        "source_entity_id",
        "semantic_episode_id",
        "semantic_vehicle",
        "active_semantic_event",
        "semantic_vehicle_state",
        "vehicle_type",
        "route_id",
        "sumo_edge_id",
        "sumo_lane_id",
        "lane_position_m",
        "center_lane_position_m",
        "sumo_xy_m",
        "sumo_position_reference",
        "sumo_front_bumper_xy_m",
        "truth_front_bumper_enu_m",
        "truth_position_reference",
        "sumo_angle_deg",
        "speed_mps",
        "accel_mps2",
        "signals",
        "dimensions_m",
        "control_role",
    ):
        if key in vehicle:
            result[key] = copy.deepcopy(vehicle[key])
    return result


def _compact_raw_vehicle(vehicle: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "vehicle_id",
        "source_entity_id",
        "semantic_episode_id",
        "semantic_vehicle",
        "active_semantic_event",
        "semantic_vehicle_state",
        "vehicle_type",
        "route_id",
        "sumo_edge_id",
        "sumo_lane_id",
        "lane_position_m",
        "center_lane_position_m",
        "sumo_xy_m",
        "sumo_position_reference",
        "sumo_front_bumper_xy_m",
        "truth_front_bumper_enu_m",
        "truth_position_enu_m",
        "truth_position_reference",
        "truth_yaw_deg",
        "sumo_angle_deg",
        "speed_mps",
        "accel_mps2",
        "signals",
        "dimensions_m",
        "control_role",
    )
    return {key: copy.deepcopy(vehicle[key]) for key in keys if key in vehicle}


def _remove_render_sumo_vehicles_absent_from_authority(
    entities: list[dict[str, Any]],
    *,
    raw_vehicle_ids: frozenset[str],
    vehicle_entity_by_sumo_id: Mapping[str, str],
    tick: int,
) -> list[tuple[str, str]]:
    """Remove render-only SUMO rows contradicted by episode-local TraCI.

    The render-ready frame is a spatially filtered delivery product.  It is not
    allowed to override the episode-local SUMO lifecycle authority used by L0.
    Unknown SUMO identities still fail closed; only known roster vehicles that
    are explicitly absent from the authoritative tick are removed.
    """

    retained: list[dict[str, Any]] = []
    removed: list[tuple[str, str]] = []
    for entity in entities:
        sumo = entity.get("sumo_vehicle")
        if not isinstance(sumo, Mapping):
            retained.append(entity)
            continue
        sumo_id = sumo.get("vehicle_id")
        entity_id = entity.get("entity_id")
        if not isinstance(sumo_id, str) or not sumo_id:
            raise L0SupplementError(
                f"tick {tick}: render SUMO entity {entity_id!r} lacks vehicle_id"
            )
        expected_entity_id = vehicle_entity_by_sumo_id.get(sumo_id)
        if expected_entity_id is None:
            raise L0SupplementError(
                f"tick {tick}: render truth contains unregistered SUMO vehicle "
                f"{sumo_id!r}"
            )
        if entity_id != expected_entity_id:
            raise L0SupplementError(
                f"tick {tick}: SUMO vehicle {sumo_id!r} is bound to render entity "
                f"{entity_id!r}, expected {expected_entity_id!r}"
            )
        if sumo_id not in raw_vehicle_ids:
            removed.append((str(entity_id), sumo_id))
            continue
        retained.append(entity)
    entities[:] = retained
    return removed


def _apply_episode_local_sumo_pose_authority(
    entity: dict[str, Any],
    roster_entity: Mapping[str, Any],
    vehicle: Mapping[str, Any],
    *,
    tick: int,
) -> bool:
    """Replace a render-derived vehicle pose with episode-local TraCI truth.

    Returns whether the render pose or lane disagreed with the authoritative
    record.  The boolean is persisted only as an aggregate audit count/digest;
    downstream predicate evaluation always receives the authoritative pose.
    """

    authoritative = _semantic_only_vehicle_entity(
        roster_entity,
        vehicle,
        tick=tick,
    )
    old_position = _entity_position(entity)
    new_position = _entity_position(authoritative)
    old_sumo = entity.get("sumo_vehicle")
    old_lane = old_sumo.get("sumo_lane_id") if isinstance(old_sumo, Mapping) else None
    new_lane = vehicle.get("sumo_lane_id")
    conflict = (
        old_position is None
        or new_position is None
        or math.dist(old_position, new_position) > 1e-6
        or old_lane != new_lane
    )
    entity["truth_pose"] = authoritative["truth_pose"]
    entity["source"] = authoritative["source"]
    entity["state_revision"] = authoritative["state_revision"]
    return conflict


def _normalize_l0_runtime_state_fields(
    entity: dict[str, Any],
    *,
    tick: int,
) -> int:
    """Replace non-authoritative runtime keys with predicate-contract keys."""

    control = entity.get("control_state")
    if not isinstance(control, dict) or "safe_hold" not in control:
        return 0
    safe_hold = control.pop("safe_hold")
    if not isinstance(safe_hold, bool):
        raise L0SupplementError(
            f"tick {tick}: {entity.get('entity_id')!r}.control_state.safe_hold "
            "must be boolean"
        )
    canonical = control.get("safe_hold_active")
    if canonical is not None and canonical != safe_hold:
        raise L0SupplementError(
            f"tick {tick}: {entity.get('entity_id')!r} has conflicting "
            "safe_hold and safe_hold_active values"
        )
    control["safe_hold_active"] = safe_hold
    return 1


def _materialize_following_vehicle_relations(
    entity_by_id: Mapping[str, dict[str, Any]],
    active_vehicle_ids: list[str],
    *,
    tick: int,
) -> None:
    """Bind each active vehicle to its immediate same-lane leader.

    SUMO lane identity and longitudinal position are the sole ordering
    authority.  The relation is absent for the front-most vehicle; no fake
    leader or rule instance is emitted.
    """

    by_lane: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}
    for entity_id in active_vehicle_ids:
        entity = entity_by_id.get(entity_id)
        sumo = entity.get("sumo_vehicle") if isinstance(entity, Mapping) else None
        if not isinstance(sumo, dict):
            raise L0SupplementError(
                f"tick {tick}: active vehicle {entity_id!r} lacks SUMO state"
            )
        lane_id = sumo.get("sumo_lane_id")
        position = _finite_number(
            sumo.get("center_lane_position_m", sumo.get("lane_position_m"))
        )
        if not isinstance(lane_id, str) or not lane_id or position is None:
            raise L0SupplementError(
                f"tick {tick}: active vehicle {entity_id!r} lacks lane ordering authority"
            )
        by_lane.setdefault(lane_id, []).append((position, entity_id, sumo))

    for lane_id, vehicles in sorted(by_lane.items()):
        ordered = sorted(vehicles, key=lambda item: (item[0], item[1]))
        for follower, leader in zip(ordered, ordered[1:]):
            follower_position, follower_id, follower_state = follower
            leader_position, leader_id, _leader_state = leader
            follower_state.update(
                {
                    "leading_vehicle_id": leader_id,
                    "leading_vehicle_ontology_class_id": "world:GroundVehicle",
                    "following_distance_m": max(
                        0.0, leader_position - follower_position
                    ),
                    "following_distance_rule_id": (
                        f"sumo_following_distance_rule:{lane_id}"
                    ),
                    "following_distance_rule_ontology_class_id": (
                        "world:FollowingDistanceRule"
                    ),
                    "following_distance_source_ref": (
                        "episode_local_sumo_traci"
                        f"#tick={tick}&lane={lane_id}&following={follower_id}"
                        f"&leading={leader_id}"
                    ),
                }
            )


def _scene_crosswalk_bindings(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    source = manifest.get("source_scene_setup_path")
    if not isinstance(source, str) or not source:
        raise L0SupplementError("episode manifest lacks source_scene_setup_path")
    scene_path = (PROJECT_ROOT / source).resolve()
    if not scene_path.is_file() or PROJECT_ROOT not in scene_path.parents:
        raise L0SupplementError(f"invalid scene setup authority: {scene_path}")
    scene = _load_object(scene_path)
    result: dict[str, dict[str, str]] = {}
    for entity in scene.get("entities", ()):
        if not isinstance(entity, Mapping) or _entity_category(entity) != "pedestrian":
            continue
        if entity.get("placement_mode") != "crosswalk_anchor":
            continue
        entity_id = entity.get("entity_id")
        placement = entity.get("placement")
        crosswalk_id = (
            placement.get("crosswalk_id") if isinstance(placement, Mapping) else None
        )
        if not isinstance(entity_id, str) or not entity_id:
            raise L0SupplementError(
                f"{scene_path}: crosswalk pedestrian lacks entity_id"
            )
        if not isinstance(crosswalk_id, str) or not crosswalk_id:
            raise L0SupplementError(
                f"{scene_path}: crosswalk_anchor {entity_id!r} lacks crosswalk_id"
            )
        if entity_id in result:
            raise L0SupplementError(
                f"{scene_path}: duplicate crosswalk binding for {entity_id!r}"
            )
        result[entity_id] = {
            "crosswalk_id": crosswalk_id,
            "crosswalk_ontology_class_id": "world:Crosswalk",
            "crosswalk_source_ref": (
                f"{source}#entity={entity_id}&placement.crosswalk_id"
            ),
        }
    return result


def _materialize_crosswalk_geometry(
    entity: dict[str, Any], *, road_context: RoadSignalContext, tick: int,
) -> None:
    """Produce crosswalk occupancy from road geometry and observed pose.

    An authored anchor supplies identity only. A network with no crossings
    remains explicitly unavailable through the source availability contract.
    Missing pose is an unknown observation; a named crossing absent from a
    nonempty authority is an identity error.
    """
    state = entity["pedestrian_state"]
    entity_id = entity["entity_id"]
    crosswalk_id = state["crosswalk_id"]
    existing = state.get("in_crosswalk")
    if existing is not None and type(existing) is not bool:
        raise L0SupplementError(
            f"tick {tick}: {entity_id!r}.in_crosswalk must be Boolean or null"
        )
    pose = entity.get("truth_pose")
    if pose is not None and not isinstance(pose, Mapping):
        raise L0SupplementError(
            f"tick {tick}: {entity_id!r}.truth_pose must be an object or null"
        )
    raw_position = pose.get("position_enu_m") if isinstance(pose, Mapping) else None
    position = None if raw_position is None else _vector3(raw_position)
    if raw_position is not None and position is None:
        raise L0SupplementError(
            f"tick {tick}: {entity_id!r}.truth_pose.position_enu_m "
            "must contain exactly three finite metre coordinates or be null"
        )
    if road_context.crossing_count == 0:
        entity["crosswalk_geometry_evidence"] = {
            "status": "crossing_geometry_unavailable",
            "source_ref": str(road_context.net_xml),
        }
        return
    crossing = road_context.crosswalks.get(crosswalk_id)
    if crossing is None:
        raise L0SupplementError(
            f"tick {tick}: {entity_id!r} names crosswalk {crosswalk_id!r} "
            f"absent from {road_context.net_xml}"
        )
    computed = None if position is None else crossing.contains_xy(position)
    if existing is not None and existing != computed:
        raise L0SupplementError(
            f"tick {tick}: {entity_id!r}.in_crosswalk conflicts with "
            "authoritative crossing geometry and observed pose"
        )
    state["in_crosswalk"] = computed
    entity["crosswalk_geometry_evidence"] = {
        "status": "pose_missing" if position is None else "observed_geometry",
        "source_refs": [
            crossing.source_ref,
            f"truth_frames.jsonl#tick={tick}&entity={entity_id}&truth_pose.position_enu_m",
        ],
        "producer": "Dataset/semantic_truth/l0_supplement.py#_materialize_crosswalk_geometry",
        "producer_version": "1.0.0",
        "semantic_contract_id": "crosswalk_closed_lane_footprint",
        "semantic_contract_version": "1.0.0",
        "coordinate_route": "sumo_xy_identity_truth_enu_m",
    }


def _entity_position(entity: Mapping[str, Any]) -> list[float] | None:
    pose = entity.get("truth_pose")
    return _vector3(pose.get("position_enu_m")) if isinstance(pose, Mapping) else None


def _semantic_only_vehicle_entity(
    roster_entity: Mapping[str, Any],
    vehicle: Mapping[str, Any],
    *,
    tick: int,
) -> dict[str, Any]:
    position = _vector3(vehicle.get("truth_position_enu_m"))
    velocity = _vector3(vehicle.get("velocity_enu_mps"))
    yaw = _finite_number(vehicle.get("truth_yaw_deg"))
    if velocity is None and yaw is not None:
        speed = _finite_number(vehicle.get("speed_mps"))
        if speed is not None:
            yaw_radians = math.radians(yaw)
            velocity = [
                speed * math.cos(yaw_radians),
                speed * math.sin(yaw_radians),
                0.0,
            ]
    if position is None or velocity is None or yaw is None:
        raise L0SupplementError(
            f"SUMO vehicle {vehicle.get('vehicle_id')!r}@{tick} lacks pose truth"
        )
    return {
        "entity_id": str(roster_entity["entity_id"]),
        "entity_category": "vehicle",
        "entity_kind": str(roster_entity.get("entity_kind") or "vehicle.car"),
        "entity_type": str(roster_entity.get("entity_type") or "vehicle.car"),
        "label_class": "vehicle",
        "logical_asset_id": roster_entity.get("logical_asset_id"),
        "truth_pose": {
            "authority_mode": "authoritative_input",
            "authority_owner": "episode_local_sumo_traci",
            "coordinate_contract_id": "coord.external_enu_m.v1",
            "position_enu_m": position,
            "rotation_deg": {"pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": yaw},
            "velocity_enu_mps": velocity,
        },
        "render_presence": {
            "global_roster": True,
            "offstage": True,
            "offstage_reason": "outside_render_runtime_boundary",
            "roi_membership": [],
            "submission_state": "semantic_truth_only",
            "visibility_state": "not_submitted",
        },
        "source": "sumo_traci",
        "state_revision": tick + 1,
    }


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise L0SupplementError(f"{path}: root must be an object")
    return value


def _index_roster(values: Any, path: Path) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, list):
        raise L0SupplementError(f"{path}: entities must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for entity in values:
        if not isinstance(entity, Mapping):
            raise L0SupplementError(f"{path}: roster entry must be an object")
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id or entity_id in result:
            raise L0SupplementError(f"{path}: invalid/duplicate roster entity id")
        result[entity_id] = entity
    return result


def _entity_category(entity: Mapping[str, Any]) -> str:
    value = entity.get("entity_category") or entity.get("category")
    return str(value).lower() if isinstance(value, str) else ""


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    numbers = [_finite_number(item) for item in value[:3]]
    if any(item is None for item in numbers):
        return None
    return [float(item) for item in numbers if item is not None]


__all__ = ["L0SupplementError", "materialize_episode_l0_state"]
