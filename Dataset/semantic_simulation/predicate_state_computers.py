"""Generic runtime-state, plan-window, and geometry predicate inputs.

These computers emit state only.  They never emit predicate truth or event
occurrences, and they never read event traces, event realizations, or dynamic
labels.  Runtime region activity comes from authoritative truth frames; scene
setup contributes geometry only.  The plan-window computer compiles the other
scheduled control/configuration windows.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from Dataset.semantic_truth.provenance import (
    digest_file,
    digest_object,
    read_jsonl,
    stable_identifier,
)
from Dataset.semantic_truth.core_semantic_registry import (
    get_governed_parameter_defaults,
)
from Dataset.semantic_truth.episode_sources import source_episode_root
from Dataset.semantic_truth.l0_state_profile import (
    DEFAULT_L0_STATE_PROFILE_PATH,
    load_l0_state_profile,
    runtime_baseline_for_category,
)
from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_FIELDS,
    invalid_runtime_state_value_paths,
    unconsumed_runtime_state_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_TICKS = tuple(range(0, 901, 5))
FORMAL_STEP_TICKS = 5
RUNTIME_STATE_PROFILE_ROOT = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "epi_runtime_state_schedules"
)
GLOBAL_UAV_TASK_PLAN_PATH = (
    PROJECT_ROOT
    / "aw_data"
    / "uav_outputs"
    / "donghu_uav_flow_270s"
    / "uav_task_plan.json"
)
AIRCRAFT_STATIONARY_SPEED_THRESHOLD_MPS = float(
    get_governed_parameter_defaults()["aircraft_stationary_speed_threshold_mps"]
)
AIRCRAFT_PAD_CONTACT_ACTIVITIES = frozenset({"preflight_on_pad", "landed", "touchdown"})
PROTECTED_CONTAINMENT_CONTRACT = {
    "contract_id": "protected_closed_prism_containment",
    "version": "1.0.0",
    "decision_authority": "user_explicit_closed_prism_selection",
    "xy_boundary": "closed",
    "z_boundary": "closed",
    "numeric_roundoff_policy": "32_coordinate_scale_ulps_m_no_physical_buffer",
    "diagnostic_boundary_meaning": "computed_float64_zero_residual_not_mathematical_exactness",
    "field_sources": {
        "geometry.inside_protected_airspace": "record:values.inside_protected_airspace",
    },
    "yaml_field_sources": {
        "geometry.inside_protected_airspace": "record:values.inside_protected_airspace",
    },
    "field_units": {"geometry.inside_protected_airspace": "bool"},
    "role_sources": {
        "aircraft": "values.aircraft_id",
        "protected_region": "values.protected_region_id",
    },
}


class PredicateStateComputerError(ValueError):
    """Raised when an authoritative state-computer input is malformed."""


@dataclass(frozen=True)
class GeometricStateResult:
    rows: tuple[dict[str, Any], ...]
    positions_by_tick: Mapping[int, Mapping[str, tuple[float, float, float]]]
    velocities_by_tick: Mapping[int, Mapping[str, tuple[float, float, float]]]
    roster_by_id: Mapping[str, Mapping[str, Any]]
    regions: Mapping[str, Mapping[str, Any]]
    restricted_region_activity_by_tick: Mapping[int, Mapping[str, bool]]
    protected_regions: Mapping[str, Mapping[str, Any]]
    corridors: Mapping[str, Mapping[str, Any]]
    input_digest: str


@dataclass(frozen=True)
class AircraftGroundReferenceResult:
    """Canonical local ground/contact model for aircraft predicate geometry."""

    uav_ids: tuple[str, ...]
    pads: Mapping[str, Mapping[str, Any]]
    global_uav_pads: Mapping[str, Mapping[str, Any]]
    global_ground_reference_z_m: float
    landing_zone_radius_m: float
    home_by_uav: Mapping[
        str,
        tuple[
            str | None,
            tuple[float, float, float] | None,
            float,
            str | None,
            tuple[float, float, float] | None,
        ],
    ]
    pad_contact_references_by_tick: Mapping[int, Mapping[str, float]]
    local_ground_reference_by_tick: Mapping[int, Mapping[str, float]]


def resolve_aircraft_ground_references(
    roster_by_id: Mapping[str, Mapping[str, Any]],
    positions_by_tick: Mapping[int, Mapping[str, tuple[float, float, float]]],
    velocities_by_tick: Mapping[int, Mapping[str, tuple[float, float, float]]],
    activities_by_tick: Mapping[int, Mapping[str, str]],
    *,
    scene_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    episode_id: str,
) -> AircraftGroundReferenceResult:
    """Resolve the single governed local-ground authority used by L1 and V8.

    ``ground_reference_z_m`` is the aircraft pose-origin contact height at its
    home location, not a globally valid terrain plane.  Away from home, the
    local reference is the assigned landing contact, an observed pad contact,
    or the governed world ground surface.
    """

    scene_entities = scene_by_id or {}
    governed_parameters = get_governed_parameter_defaults()
    landing_zone_radius_m = float(governed_parameters["landing_zone_radius_m"])
    uav_ids = tuple(
        sorted(
            entity_id
            for entity_id, entity in roster_by_id.items()
            if _entity_category(entity) == "uav"
        )
    )
    pads: dict[str, Mapping[str, Any]] = {
        entity_id: entity
        for entity_id, entity in roster_by_id.items()
        if isinstance(entity.get("semantic_scope"), Mapping)
        and entity["semantic_scope"].get("scope_type") == "facility"
        and entity["semantic_scope"].get("scope_subtype") == "landing_pad"
    }
    global_uav_pads = _load_global_uav_pads(GLOBAL_UAV_TASK_PLAN_PATH)
    global_ground_reference_z_m = _load_global_uav_ground_reference_z(
        GLOBAL_UAV_TASK_PLAN_PATH
    )
    pads.update(global_uav_pads)

    home_by_uav: dict[
        str,
        tuple[
            str | None,
            tuple[float, float, float] | None,
            float,
            str | None,
            tuple[float, float, float] | None,
        ],
    ] = {}
    for uav_id in uav_ids:
        roster = roster_by_id[uav_id]
        scene_uav = scene_entities.get(uav_id, {})
        lifecycle = (
            roster.get("lifecycle")
            if isinstance(roster.get("lifecycle"), Mapping)
            else {}
        )
        raw_home_pad_id = lifecycle.get("home_pad_entity_id")
        home_pad_id = (
            str(raw_home_pad_id)
            if isinstance(raw_home_pad_id, str) and raw_home_pad_id
            else None
        )
        global_flow = (
            roster.get("uav_global_flow")
            if isinstance(roster.get("uav_global_flow"), Mapping)
            else {}
        )
        if home_pad_id is None:
            origin_pad_id = global_flow.get("origin_pad_id")
            if isinstance(origin_pad_id, str) and origin_pad_id:
                home_pad_id = origin_pad_id
        home_pad_pose = (
            _entity_position(pads.get(home_pad_id, {})) if home_pad_id else None
        )
        home_hover = _vector3(lifecycle.get("home_hover_enu_m"))
        explicit_ground_z = _number(
            roster.get("ground_reference_z_m")
            if roster.get("ground_reference_z_m") is not None
            else scene_uav.get("ground_reference_z_m")
        )
        flow_ground_z = _number(global_flow.get("ground_reference_z_m"))
        if explicit_ground_z is None:
            if home_hover is not None:
                explicit_ground_z = home_hover[2]
            elif home_pad_pose is not None:
                explicit_ground_z = home_pad_pose[2]
            elif flow_ground_z is not None:
                explicit_ground_z = flow_ground_z
            else:
                explicit_ground_z = global_ground_reference_z_m
        if explicit_ground_z is None:
            raise PredicateStateComputerError(
                f"{episode_id}:{uav_id}: authoritative ground_reference_z_m is required"
            )
        takeoff_ground_z = float(explicit_ground_z)
        target_pad_id = global_flow.get("target_pad_id")
        target_pad_pose = (
            _entity_position(pads.get(target_pad_id, {}))
            if isinstance(target_pad_id, str) and target_pad_id
            else None
        )
        if target_pad_pose is not None:
            landing_zone_id = str(target_pad_id)
            landing_pose = target_pad_pose
        else:
            landing_zone_id, landing_pose = _assigned_landing_zone(
                uav_id,
                roster,
                home_pad_id,
                home_pad_pose,
                home_hover,
            )
        home_by_uav[uav_id] = (
            home_pad_id,
            home_pad_pose,
            takeoff_ground_z,
            landing_zone_id,
            landing_pose,
        )

    pad_contact_references_by_tick = _landing_pad_contact_references_by_tick(
        pads,
        positions_by_tick,
        velocities_by_tick,
        activities_by_tick,
        uav_ids,
        landing_zone_radius_m,
    )
    local_ground_reference_by_tick: dict[int, dict[str, float]] = {}
    for tick, tick_positions in positions_by_tick.items():
        for uav_id in uav_ids:
            position = tick_positions.get(uav_id)
            if position is None:
                continue
            (
                _home_pad_id,
                home_pad_pose,
                takeoff_ground_z,
                _landing_zone_id,
                landing_pose,
            ) = home_by_uav[uav_id]
            local_ground_reference_by_tick.setdefault(tick, {})[uav_id] = (
                _local_aircraft_ground_reference_z(
                    position,
                    landing_pose,
                    home_pad_pose,
                    takeoff_ground_z,
                    pads,
                    pad_contact_references_by_tick[tick],
                    landing_zone_radius_m,
                    (
                        home_pad_pose[2]
                        if home_pad_pose is not None
                        else global_ground_reference_z_m
                    ),
                )
            )

    return AircraftGroundReferenceResult(
        uav_ids=uav_ids,
        pads=pads,
        global_uav_pads=global_uav_pads,
        global_ground_reference_z_m=global_ground_reference_z_m,
        landing_zone_radius_m=landing_zone_radius_m,
        home_by_uav=home_by_uav,
        pad_contact_references_by_tick=pad_contact_references_by_tick,
        local_ground_reference_by_tick=local_ground_reference_by_tick,
    )


class GeometricStateComputer:
    """Derive reusable aircraft, region, pair, altitude, and corridor geometry."""

    def __init__(self, episode_root: Path) -> None:
        self.episode_root = episode_root.resolve()
        self.manifest_path = self.episode_root / "episode_manifest.json"
        self.roster_path = self.episode_root / "global_entity_roster.json"
        self.trajectories_path = self.episode_root / "trajectories.jsonl"
        self.truth_frames_path = self.episode_root / "truth_frames.jsonl"
        scenario_root = source_episode_root(self.episode_root)
        self.scenario_trajectories_path = scenario_root / 'trajectories.jsonl' if scenario_root is not None else None
        for path in (
            self.manifest_path,
            self.roster_path,
            self.trajectories_path,
            self.truth_frames_path,
        ):
            if not path.is_file():
                raise PredicateStateComputerError(
                    f"required geometric input is missing: {path}"
                )

    def compute(self) -> GeometricStateResult:
        manifest = _load_object(self.manifest_path)
        episode_id = str(manifest.get("episode_id") or self.episode_root.name)
        if episode_id != self.episode_root.name:
            raise PredicateStateComputerError(
                f"manifest episode_id {episode_id!r} does not match {self.episode_root.name!r}"
            )
        roster_value = _load_object(self.roster_path)
        roster_by_id = _index_entities(roster_value.get("entities"), self.roster_path)
        scene_path = _resolve_scenario_json(manifest, "scene_setup.json")
        scene = _load_object(scene_path)
        scene_by_id = _index_entities(scene.get("entities"), scene_path)
        positions, velocities, activities = load_trajectory_vectors(
            self.trajectories_path,
            self.scenario_trajectories_path,
        )
        regions = _restricted_regions(scene_by_id)
        region_activity_by_tick = restricted_region_activity_by_tick(
            self.truth_frames_path,
            regions,
        )
        protected_regions = _protected_regions(scene_by_id)
        frame_entity_ids = ({
            tick: set(entities)
            for _, tick, entities in _indexed_truth_frame_entities(self.truth_frames_path, episode_id)
            if tick in FORMAL_TICKS
        } if protected_regions else {})
        governed_parameters = get_governed_parameter_defaults()
        separation_margin_m = float(governed_parameters["separation_margin_m"])
        corridors = _airspace_corridors(scene_by_id, separation_margin_m)
        event_script_path = _resolve_scenario_json(manifest, "event_script.json")
        event_script = _load_object(event_script_path)
        entity_metadata = dict(scene_by_id)
        entity_metadata.update(roster_by_id)
        separation_models = _aircraft_pair_separation_models(
            event_script,
            entity_metadata,
            separation_margin_m,
        )
        aircraft_ground = resolve_aircraft_ground_references(
            roster_by_id,
            positions,
            velocities,
            activities,
            scene_by_id=scene_by_id,
            episode_id=episode_id,
        )
        uav_ids = aircraft_ground.uav_ids
        home_by_uav = aircraft_ground.home_by_uav
        landing_zone_radius_m = aircraft_ground.landing_zone_radius_m
        rows: list[dict[str, Any]] = []
        input_digest = digest_object(
            {
                "episode_id": episode_id,
                "trajectory_pose_count": sum(
                    len(items) for items in positions.values()
                ),
                "trajectory_state_digest": _trajectory_state_digest(
                    positions,
                    velocities,
                ),
                "trajectory_overlap_authority": "render-ready world source",
                "uav_ids": uav_ids,
                "region_geometry": regions,
                "restricted_region_activity_digest": digest_object(
                    region_activity_by_tick
                ),
                "protected_region_geometry": protected_regions,
                "corridor_geometry": corridors,
                "aircraft_pair_separation_models": _separation_model_records(
                    separation_models
                ),
                "corridor_capacity_model": {
                    "model": "independent_cross_section_lanes",
                    "minimum_center_separation_m": separation_margin_m,
                },
                "scene_setup": str(scene_path.relative_to(PROJECT_ROOT)),
                "global_uav_task_plan": digest_file(GLOBAL_UAV_TASK_PLAN_PATH),
                "global_uav_pad_geometry": aircraft_ground.global_uav_pads,
            }
        )

        for tick in FORMAL_TICKS:
            tick_positions = positions.get(tick, {})
            tick_velocities = velocities.get(tick, {})
            present_uavs = [uav_id for uav_id in uav_ids if uav_id in tick_positions]
            active_corridors = {
                corridor_id: corridor
                for corridor_id, corridor in corridors.items()
                if _corridor_active_at_tick(corridor, tick)
            }
            occupants_by_corridor = {
                corridor_id: [
                    uav_id
                    for uav_id in present_uavs
                    if _point_in_oriented_corridor_box(tick_positions[uav_id], corridor)
                ]
                for corridor_id, corridor in sorted(active_corridors.items())
            }
            for uav_id in present_uavs:
                position = tick_positions[uav_id]
                velocity = tick_velocities.get(uav_id)
                roster = roster_by_id[uav_id]
                (
                    home_pad_id,
                    pad_pose,
                    takeoff_ground_z,
                    landing_zone_id,
                    landing_pose,
                ) = home_by_uav[uav_id]
                xy_home = (
                    _distance_xy(position, pad_pose) if pad_pose is not None else None
                )
                xy_landing = (
                    _distance_xy(position, landing_pose)
                    if landing_pose is not None
                    else None
                )
                local_ground_z = aircraft_ground.local_ground_reference_by_tick[tick][
                    uav_id
                ]
                z_agl = (
                    position[2] - local_ground_z if local_ground_z is not None else None
                )
                assigned_altitude = _number(roster.get("assigned_altitude_m"))
                if assigned_altitude is None and isinstance(
                    roster.get("uav_corridor"), Mapping
                ):
                    assigned_altitude = _number(
                        roster["uav_corridor"].get("assigned_altitude_m")
                    )
                if assigned_altitude is None and isinstance(
                    roster.get("uav_global_flow"), Mapping
                ):
                    assigned_altitude = _number(
                        roster["uav_global_flow"].get("altitude_layer_m")
                    )
                if assigned_altitude is None and isinstance(
                    roster.get("motion_contract"), Mapping
                ):
                    assigned_altitude = _number(
                        roster["motion_contract"].get("altitude_layer_m")
                    )
                containing_corridor_ids = sorted(
                    corridor_id
                    for corridor_id, occupants in occupants_by_corridor.items()
                    if uav_id in occupants
                )
                values = {
                    "position_enu_m": list(position),
                    "velocity_enu_mps": list(velocity)
                    if velocity is not None
                    else None,
                    "speed_mps": math.dist((0.0, 0.0, 0.0), velocity)
                    if velocity is not None
                    else None,
                    "z_agl_m": z_agl,
                    "local_ground_reference_z_m": local_ground_z,
                    "takeoff_ground_z_m": takeoff_ground_z,
                    "home_pad_id": home_pad_id,
                    "home_pad_ontology_class_id": (
                        "world:LandingPad" if home_pad_id is not None else None
                    ),
                    "home_pad_pose_enu_m": list(pad_pose)
                    if pad_pose is not None
                    else None,
                    "xy_distance_to_home_pad_m": xy_home,
                    "assigned_landing_zone_id": landing_zone_id,
                    "assigned_landing_zone_ontology_class_id": (
                        "world:LandingArea"
                        if isinstance(landing_zone_id, str)
                        and landing_zone_id.startswith("assigned_landing_zone:")
                        else "world:LandingPad"
                        if landing_zone_id is not None
                        else None
                    ),
                    "assigned_landing_zone_pose_enu_m": (
                        list(landing_pose) if landing_pose is not None else None
                    ),
                    "xy_distance_to_assigned_landing_zone_m": xy_landing,
                    "assigned_altitude_m": assigned_altitude,
                    "altitude_deviation_m": abs(position[2] - assigned_altitude)
                    if assigned_altitude is not None
                    else None,
                    "aircraft_inside_corridor": bool(containing_corridor_ids),
                    "containing_corridor_ids": containing_corridor_ids,
                    "entity_role": _entity_role(roster),
                }
                rows.append(
                    _state_row(
                        episode_id,
                        tick,
                        "predicate_contract_aircraft_geometry",
                        uav_id,
                        "uav",
                        values,
                        _missing(
                            values,
                            (
                                "position_enu_m",
                                "z_agl_m",
                                "local_ground_reference_z_m",
                                "assigned_landing_zone_id",
                            ),
                        ),
                        [
                            f"trajectories.jsonl#tick={tick}&entity={uav_id}",
                            f"global_entity_roster.json#entity={uav_id}",
                        ],
                        input_digest,
                        "geometric_computer.aircraft_state",
                        parameter_digest=digest_object(
                            {
                                "landing_zone_radius_m": landing_zone_radius_m,
                                "local_ground_policy": (
                                    "assigned_landing_contact_else_nearest_contacted_pad_else_enu_ground_surface"
                                ),
                            }
                        ),
                    )
                )

                for region_id, region in sorted(regions.items()):
                    polygon = region["polygon_enu_m"]
                    inside_xy = _point_in_polygon_xy(position, polygon)
                    inside_z = (
                        float(region["base_z_m"])
                        <= position[2]
                        <= float(region["top_z_m"])
                    )
                    horizontal_boundary_distance = _distance_to_polygon_xy(
                        position, polygon
                    )
                    signed_distance = _signed_distance_to_restricted_prism(position, region)
                    boundary_distance = abs(signed_distance)
                    region_values = {
                        "aircraft_id": uav_id,
                        "aircraft_ontology_class_id": "world:UnmannedAircraft",
                        "restricted_region_id": region_id,
                        "restricted_region_ontology_class_id": (
                            "world:RestrictedAirspaceRegion"
                        ),
                        "restricted_region_active": (
                            region_activity_by_tick[tick][region_id]
                        ),
                        "boundary_polygon_enu_m": [list(point) for point in polygon],
                        "distance_to_boundary_m": boundary_distance,
                        "horizontal_distance_to_boundary_m": horizontal_boundary_distance,
                        "signed_distance_to_boundary_m": signed_distance,
                        "inside_restricted_region": inside_xy and inside_z,
                    }
                    rows.append(
                        _state_row(
                            episode_id,
                            tick,
                            "predicate_contract_aircraft_region_geometry",
                            f"{uav_id}|{region_id}",
                            "uav_region",
                            region_values,
                            _missing(
                                region_values,
                                ("boundary_polygon_enu_m", "distance_to_boundary_m"),
                            ),
                            [
                                f"trajectories.jsonl#tick={tick}&entity={uav_id}",
                                f"{scene_path.relative_to(PROJECT_ROOT)}#entity={region_id}",
                                f"truth_frames.jsonl#tick={tick}&entity={region_id}",
                            ],
                            input_digest,
                            "geometric_computer.restricted_region_distance",
                            rule_version="4.0.0",
                            source_semantic_contract={
                                "contract_id": "restricted_prism_signed_distance",
                                "version": "1.0.0",
                                "field_sources": {
                                    "geometry.restricted_region_active": "record:values.restricted_region_active",
                                    "geometry.minimum_restricted_boundary_distance_m": "record:values.signed_distance_to_boundary_m",
                                },
                                "yaml_field_sources": {
                                    "geometry.restricted_region_active": "record:values.restricted_region_active",
                                    "geometry.minimum_restricted_boundary_distance_m": "record:values.signed_distance_to_boundary_m",
                                },
                                "field_units": {
                                    "geometry.restricted_region_active": "bool",
                                    "geometry.minimum_restricted_boundary_distance_m": "m",
                                },
                                "role_sources": {
                                    "actor": "values.aircraft_id",
                                    "restricted_region": "values.restricted_region_id",
                                },
                            },
                        )
                    )

                for region_id, region in sorted(protected_regions.items()):
                    if uav_id not in frame_entity_ids[tick]:
                        continue
                    inside_protected, containment = _protected_prism_membership(position, region)
                    protected_values = {
                        "aircraft_id": uav_id,
                        "aircraft_ontology_class_id": "world:UnmannedAircraft",
                        "protected_region_id": region_id,
                        "protected_region_ontology_class_id": "world:ProtectedAirspace",
                        "inside_protected_airspace": inside_protected,
                        "protected_containment": containment,
                        "distance_to_protected_airspace_m": (
                            _distance_to_restricted_prism(position, region)
                        ),
                    }
                    rows.append(
                        _state_row(
                            episode_id,
                            tick,
                            "predicate_contract_aircraft_protected_region_geometry",
                            f"{uav_id}|{region_id}",
                            "uav_protected_region",
                            protected_values,
                            [],
                            [
                                f"trajectories.jsonl#tick={tick}&entity={uav_id}",
                                f"{scene_path.relative_to(PROJECT_ROOT)}#entity={region_id}",
                            ],
                            input_digest,
                            "geometric_computer.protected_region_containment",
                            rule_version="4.0.0",
                            source_semantic_contract=PROTECTED_CONTAINMENT_CONTRACT,
                        )
                    )

                for corridor_id, corridor in sorted(active_corridors.items()):
                    inside_corridor = uav_id in occupants_by_corridor[corridor_id]
                    corridor_pair_values = {
                        "aircraft_id": uav_id,
                        "aircraft_ontology_class_id": "world:UnmannedAircraft",
                        "corridor_id": corridor_id,
                        "corridor_ontology_class_id": "world:AirspaceCorridor",
                        "aircraft_inside_corridor": inside_corridor,
                        "speed_mps": values["speed_mps"],
                    }
                    rows.append(
                        _state_row(
                            episode_id,
                            tick,
                            "predicate_contract_aircraft_corridor_geometry",
                            f"{uav_id}|{corridor_id}",
                            "uav_corridor",
                            corridor_pair_values,
                            _missing(corridor_pair_values, ("speed_mps",)),
                            [
                                f"trajectories.jsonl#tick={tick}&entity={uav_id}",
                                f"{scene_path.relative_to(PROJECT_ROOT)}#entity={corridor_id}",
                            ],
                            input_digest,
                            "geometric_computer.aircraft_corridor_membership",
                        )
                    )

            for uav_id in sorted((set(uav_ids) & frame_entity_ids.get(tick, set())) - set(present_uavs)):
                for region_id in sorted(protected_regions):
                    missing_pose_ref = f"trajectories.jsonl#tick={tick}&entity={uav_id}&field=position_enu_m"
                    rows.append(_state_row(
                        episode_id, tick, "predicate_contract_aircraft_protected_region_geometry",
                        f"{uav_id}|{region_id}", "uav_protected_region",
                        {
                            "aircraft_id": uav_id,
                            "aircraft_ontology_class_id": "world:UnmannedAircraft",
                            "protected_region_id": region_id,
                            "protected_region_ontology_class_id": "world:ProtectedAirspace",
                            "inside_protected_airspace": None,
                            "distance_to_protected_airspace_m": None,
                            "protected_containment": {"classification": "unknown_missing_pose"},
                        },
                        [f"missing_source_record:{missing_pose_ref}"],
                        [missing_pose_ref, f"truth_frames.jsonl#tick={tick}&entity={uav_id}",
                         f"{scene_path.relative_to(PROJECT_ROOT)}#entity={region_id}"],
                        input_digest, "geometric_computer.protected_region_containment",
                        rule_version="4.0.0", source_semantic_contract=PROTECTED_CONTAINMENT_CONTRACT,
                    ))

            for first, second in combinations(present_uavs, 2):
                euclidean_distance = math.dist(
                    tick_positions[first], tick_positions[second]
                )
                horizontal_distance = math.dist(
                    tick_positions[first][:2], tick_positions[second][:2]
                )
                vertical_distance = abs(
                    tick_positions[first][2] - tick_positions[second][2]
                )
                separation_model = separation_models.get(
                    tuple(sorted((first, second)))
                )
                distance = _equivalent_aircraft_separation_distance(
                    euclidean_distance=euclidean_distance,
                    horizontal_distance=horizontal_distance,
                    vertical_distance=vertical_distance,
                    model=separation_model,
                    separation_margin_m=separation_margin_m,
                )
                values = {
                    "first_aircraft_id": first,
                    "first_aircraft_ontology_class_id": "world:UnmannedAircraft",
                    "second_aircraft_id": second,
                    "second_aircraft_ontology_class_id": "world:UnmannedAircraft",
                    "pair_entity_ids": [first, second],
                    "pair_distance_m": distance,
                    "pair_euclidean_distance_m": euclidean_distance,
                    "pair_horizontal_distance_m": horizontal_distance,
                    "pair_vertical_distance_m": vertical_distance,
                    "pair_distance_metric": (
                        "anisotropic_safety_volume_equivalent_distance"
                        if separation_model is not None
                        else "euclidean_3d"
                    ),
                }
                if separation_model is not None:
                    values.update(
                        {
                            "pair_horizontal_limit_m": separation_model[
                                "horizontal_limit_m"
                            ],
                            "pair_vertical_limit_m": separation_model[
                                "vertical_limit_m"
                            ],
                        }
                    )
                rows.append(
                    _state_row(
                        episode_id,
                        tick,
                        "predicate_contract_aircraft_pair_geometry",
                        f"{first}|{second}",
                        "uav_pair",
                        values,
                        [],
                        [
                            f"trajectories.jsonl#tick={tick}&entity={first}",
                            f"trajectories.jsonl#tick={tick}&entity={second}",
                        ],
                        input_digest,
                        "geometric_computer.aircraft_pair_distance",
                        parameter_digest=digest_object(
                            {
                                "separation_margin_m": separation_margin_m,
                                "model": _separation_model_record(
                                    separation_model
                                ),
                            }
                        ),
                    )
                )

            structure_ids = sorted(
                entity_id
                for entity_id, entity in entity_metadata.items()
                if _is_building_structure(entity_id, entity)
            )
            present_vehicles = sorted(
                entity_id
                for entity_id in tick_positions
                if _entity_category(entity_metadata.get(entity_id, {})) == "vehicle"
            )
            present_pedestrians = sorted(
                entity_id
                for entity_id in tick_positions
                if _entity_category(entity_metadata.get(entity_id, {})) == "pedestrian"
            )
            _append_full_pair_geometry_rows(
                rows=rows,
                episode_id=episode_id,
                tick=tick,
                tick_positions=tick_positions,
                entity_metadata=entity_metadata,
                uav_ids=present_uavs,
                vehicle_ids=present_vehicles,
                pedestrian_ids=present_pedestrians,
                structure_ids=structure_ids,
                event_script_path=event_script_path,
                input_digest=input_digest,
            )
            _append_agent_proximity_rows(
                rows=rows,
                episode_id=episode_id,
                tick=tick,
                tick_positions=tick_positions,
                roster_by_id=roster_by_id,
                entity_metadata=entity_metadata,
                structure_ids=structure_ids,
                separation_margin_m=separation_margin_m,
                event_script_path=event_script_path,
                input_digest=input_digest,
                separation_models=separation_models,
            )

            for corridor_id, corridor in sorted(active_corridors.items()):
                occupants = sorted(occupants_by_corridor[corridor_id])
                corridor_values = {
                    "corridor_id": corridor_id,
                    "corridor_ontology_class_id": "world:AirspaceCorridor",
                    "corridor_occupancy_count": len(occupants),
                    "corridor_capacity": int(corridor["capacity"]),
                    "occupying_aircraft_ids": occupants,
                    "corridor_center_enu_m": list(corridor["center_enu_m"]),
                    "corridor_extent_m": list(corridor["extent_m"]),
                    "corridor_yaw_deg": float(corridor["yaw_deg"]),
                    "corridor_cross_section_size_m": list(
                        corridor["cross_section_size_m"]
                    ),
                    "minimum_center_separation_m": separation_margin_m,
                    "capacity_model": "independent_cross_section_lanes",
                }
                rows.append(
                    _state_row(
                        episode_id,
                        tick,
                        "predicate_contract_corridor_geometry",
                        corridor_id,
                        "airspace_corridor",
                        corridor_values,
                        [],
                        [
                            f"{scene_path.relative_to(PROJECT_ROOT)}#entity={corridor_id}",
                            *(
                                f"trajectories.jsonl#tick={tick}&entity={uav_id}"
                                for uav_id in occupants
                            ),
                        ],
                        input_digest,
                        "geometric_computer.corridor_occupancy_and_capacity",
                        parameter_digest=str(corridor["parameter_digest"]),
                    )
                )

        rows.sort(
            key=lambda row: (
                int(row["tick"]),
                str(row["observation_family"]),
                str(row["subject_id"]),
            )
        )
        return GeometricStateResult(
            rows=tuple(rows),
            positions_by_tick=positions,
            velocities_by_tick=velocities,
            roster_by_id=roster_by_id,
            regions=regions,
            restricted_region_activity_by_tick=region_activity_by_tick,
            protected_regions=protected_regions,
            corridors=corridors,
            input_digest=input_digest,
        )


class PlanWindowComputer:
    """Compile region activation, cooperation, and control-mode plan windows."""

    def __init__(self, episode_root: Path, geometry: GeometricStateResult) -> None:
        self.episode_root = episode_root.resolve()
        self.geometry = geometry
        self.manifest_path = self.episode_root / "episode_manifest.json"

    def materialize_runtime_state_truth(
        self,
        truth_frames_path: Path | None = None,
    ) -> dict[str, Any]:
        """Apply the executed runtime-state plan to an in-memory L0 episode.

        Runtime-state schedules are simulator control inputs.  Their event
        triggers are independently resolved from numeric trajectories and
        weather; event traces, realizations, labels, and expected events are
        never read.  The method is intended for the objective pipeline's
        temporary strict episode root and never mutates the formal source
        episode.
        """

        manifest = _load_object(self.manifest_path)
        episode_id = str(manifest.get("episode_id") or self.episode_root.name)
        epi_id = episode_id.split("__seed", 1)[0]
        schedule = _load_runtime_state_schedule(epi_id)
        event_script_path = _resolve_scenario_json(manifest, "event_script.json")
        script = _load_object(event_script_path)
        events = {
            str(event["event_id"]): event
            for event in script.get("events", ())
            if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
        }
        runtime_event_ids = set(schedule["event_state_transitions"])
        if not runtime_event_ids <= set(events):
            raise PredicateStateComputerError(
                f"runtime schedule events are absent from {epi_id}: "
                f"{sorted(runtime_event_ids - set(events))}"
            )
        weather_path = self.episode_root / "weather_meta.jsonl"
        if not weather_path.is_file():
            raise PredicateStateComputerError(
                f"runtime plan weather input is missing: {weather_path}"
            )
        weather_by_tick = _weather_by_tick(weather_path)
        fire_ticks = _resolve_event_fire_ticks(
            script,
            self.geometry.positions_by_tick,
            self.geometry.regions,
            required_event_ids=runtime_event_ids,
            weather_by_tick=weather_by_tick,
        )
        unresolved_event_ids = sorted(runtime_event_ids - set(fire_ticks))
        l0_profile = load_l0_state_profile()
        state_by_entity: dict[str, dict[str, Any]] = {}
        for entity_id, entity in sorted(self.geometry.roster_by_id.items()):
            baseline = runtime_baseline_for_category(
                l0_profile,
                _entity_category(entity),
            )
            if baseline:
                state_by_entity[entity_id] = baseline
        for entity_id, initial_state in schedule["initial_entity_states"].items():
            _deep_merge(state_by_entity.setdefault(entity_id, {}), initial_state)
        patches_by_tick: dict[int, list[tuple[str, dict[str, Any], str]]] = {}
        applied_action_ids: list[str] = []
        pending_outside_window: list[dict[str, Any]] = []
        for event_id, raw_transitions in sorted(
            schedule["event_state_transitions"].items()
        ):
            if event_id not in fire_ticks:
                continue
            for transition_index, transition in enumerate(raw_transitions):
                entity_id = str(transition["entity_id"])
                patch = deepcopy(transition["state_patch"])
                delay = int(transition["delay_ticks"])
                effective_tick = fire_ticks[event_id] + max(1, delay)
                action_id = f"{schedule['profile_id']}:{event_id}:{transition_index}"
                if effective_tick > 900:
                    pending_outside_window.append({"event_id": event_id, "entity_id": entity_id,
                        "action_id": action_id, "effective_tick": effective_tick, "window_end_tick": 900,
                        "source_script": str(event_script_path.relative_to(PROJECT_ROOT)),
                        "status": "scheduled_beyond_observation_window"})
                    continue
                patches_by_tick.setdefault(effective_tick, []).append(
                    (entity_id, patch, action_id)
                )

        target_path = (
            truth_frames_path or self.episode_root / "truth_frames.jsonl"
        ).resolve()
        if not target_path.is_file():
            raise PredicateStateComputerError(
                f"runtime plan truth input is missing: {target_path}"
            )
        frames = sorted(
            read_jsonl(target_path), key=lambda row: int(row.get("tick", -1))
        )
        expected_ticks = list(range(0, 901))
        observed_ticks = [row.get("tick") for row in frames]
        if observed_ticks != expected_ticks:
            raise PredicateStateComputerError(
                "runtime-state materialization requires exact tick 0..900 truth frames"
            )
        roster_ids = set(self.geometry.roster_by_id)
        unknown_entities = sorted(set(state_by_entity) - roster_ids)
        if unknown_entities:
            raise PredicateStateComputerError(
                f"runtime schedule references entities outside the roster: {unknown_entities}"
            )
        scheduled_entities = {
            entity_id
            for items in patches_by_tick.values()
            for entity_id, _patch, _action_id in items
        } | {item["entity_id"] for item in pending_outside_window}
        unknown_entities = sorted(scheduled_entities - roster_ids)
        if unknown_entities:
            raise PredicateStateComputerError(
                f"runtime actions reference entities outside the roster: {unknown_entities}"
            )

        materialized_entity_ticks = 0
        for frame in frames:
            tick = int(frame["tick"])
            for entity_id, patch, action_id in patches_by_tick.get(tick, ()):
                current = state_by_entity.setdefault(entity_id, {})
                _deep_merge(current, patch)
                applied_action_ids.append(action_id)
            entities = frame.get("entities")
            if not isinstance(entities, list):
                raise PredicateStateComputerError(
                    f"{target_path}: tick {tick} entities must be an array"
                )
            entity_by_id = {
                str(entity.get("entity_id")): entity
                for entity in entities
                if isinstance(entity, dict) and isinstance(entity.get("entity_id"), str)
            }
            for entity_id, planned_state in sorted(state_by_entity.items()):
                entity = entity_by_id.get(entity_id)
                if entity is None:
                    continue
                for family, family_state in planned_state.items():
                    existing = entity.get(family)
                    if existing is None:
                        existing = {}
                    if not isinstance(existing, dict):
                        raise PredicateStateComputerError(
                            f"{target_path}: {entity_id}.{family} is not an object"
                        )
                    _merge_missing_runtime_state(
                        existing,
                        family_state,
                        family=family,
                    )
                    entity[family] = existing
                materialized_entity_ticks += 1

        target_path.write_text(
            "".join(
                json.dumps(
                    frame,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
                for frame in frames
            ),
            encoding="utf-8",
            newline="\n",
        )
        summary = {
            "episode_id": episode_id,
            "runtime_schedule_profile_id": schedule["profile_id"],
            "runtime_schedule_source": str(
                schedule["source_path"].relative_to(PROJECT_ROOT)
            ),
            "l0_state_profile_id": l0_profile["profile_id"],
            "runtime_event_count": len(runtime_event_ids),
            "resolved_runtime_event_count": len(runtime_event_ids)
            - len(unresolved_event_ids),
            "unresolved_runtime_event_ids": unresolved_event_ids,
            "applied_runtime_action_count": len(applied_action_ids),
            "materialized_entity_tick_count": materialized_entity_ticks,
            "event_fire_ticks": dict(sorted(fire_ticks.items())),
        }
        if pending_outside_window:
            summary["pending_outside_window"] = pending_outside_window
        return summary

    def compute(self) -> tuple[dict[str, Any], ...]:
        manifest = _load_object(self.manifest_path)
        episode_id = str(manifest.get("episode_id") or self.episode_root.name)
        event_script_path = _resolve_scenario_json(manifest, "event_script.json")
        script = _load_object(event_script_path)
        runtime_schedule = _load_runtime_state_schedule(
            episode_id.split("__seed", 1)[0]
        )
        lockdown_state_governed = _runtime_schedule_writes_field(
            runtime_schedule,
            "incident_state",
            "temporary_lockdown_active",
        )
        boundary_margin_m = _boundary_margin_from_script(script)
        events = [
            event for event in script.get("events", []) if isinstance(event, Mapping)
        ]
        cooperation = _observed_aircraft_cooperation(
            self.episode_root / "truth_frames.jsonl",
            tuple(
                entity_id
                for entity_id, roster in self.geometry.roster_by_id.items()
                if _entity_category(roster) == "uav"
            ),
            episode_id=episode_id,
        )
        rth_event_ids = {
            str(event.get("event_id") or "")
            for event in events
            if _event_activates_rth(event)
        }
        rth_aircraft_ids = {
            aircraft_id
            for event in events
            if str(event.get("event_id") or "") in rth_event_ids
            for aircraft_id in _moved_entity_ids(event)
        }
        landing_event_ids = {
            str(event.get("event_id") or "")
            for event in events
            if "landing" in str(event.get("intent") or "").lower()
            and set(_moved_entity_ids(event)) & rth_aircraft_ids
        }
        fire_ticks = _resolve_event_fire_ticks(
            script,
            self.geometry.positions_by_tick,
            self.geometry.regions,
            required_event_ids=rth_event_ids | landing_event_ids,
        )
        rth_windows = _control_windows(events, fire_ticks, rth_event_ids)
        landing_starts = _landing_start_ticks(events, fire_ticks)
        input_digest = digest_object(
            {
                "event_script": str(event_script_path.relative_to(PROJECT_ROOT)),
                "event_fire_ticks": fire_ticks,
                "boundary_margin_m": boundary_margin_m,
                "region_ids": sorted(self.geometry.regions),
                "geometry_input_digest": self.geometry.input_digest,
                "runtime_schedule": digest_file(runtime_schedule["source_path"]),
                "lockdown_state_governed": lockdown_state_governed,
            }
        )
        rows: list[dict[str, Any]] = []
        boundary_distances: dict[tuple[int, str], list[float]] = {}
        for row in self.geometry.rows:
            if (
                row.get("observation_family")
                != "predicate_contract_aircraft_region_geometry"
            ):
                continue
            subject = str(row.get("subject_id") or "")
            uav_id, separator, region_id = subject.partition("|")
            row_values = row.get("values")
            distance = _number(
                row_values.get("signed_distance_to_boundary_m")
                if isinstance(row_values, Mapping)
                else None
            )
            if (
                separator
                and distance is not None
                and self.geometry.restricted_region_activity_by_tick[int(row["tick"])][region_id]
            ):
                boundary_distances.setdefault((int(row["tick"]), uav_id), []).append(
                    distance
                )

        for tick in FORMAL_TICKS:
            for region_id, region in sorted(self.geometry.regions.items()):
                active = self.geometry.restricted_region_activity_by_tick[tick][
                    region_id
                ]
                rows.append(
                    _state_row(
                        episode_id,
                        tick,
                        "predicate_contract_region_runtime_state",
                        region_id,
                        "restricted_region",
                        {
                            "no_fly_zone_id": region_id,
                            "no_fly_zone_ontology_class_id": "world:NoFlyZone",
                            "restricted_region_active": active,
                            "boundary_margin_m": boundary_margin_m,
                        },
                        [],
                        [
                            f"{event_script_path.relative_to(PROJECT_ROOT)}#region={region_id}",
                            f"truth_frames.jsonl#tick={tick}&entity={region_id}",
                            f"{event_script_path.relative_to(PROJECT_ROOT)}#parameters.conflict_distance_m",
                        ],
                        input_digest,
                        "runtime_region_state_computer.region_activity",
                        parameter_digest=digest_object(
                            {
                                "boundary_margin_m": boundary_margin_m,
                                "source": "parameters.conflict_distance_m",
                            }
                        ),
                    )
                )

            tick_positions = self.geometry.positions_by_tick.get(tick, {})
            for uav_id, roster in sorted(self.geometry.roster_by_id.items()):
                if _entity_category(roster) != "uav":
                    continue
                active_regions = [
                    (region_id, region)
                    for region_id, region in sorted(self.geometry.regions.items())
                    if self.geometry.restricted_region_activity_by_tick[tick][region_id]
                ]
                active_region_distances = boundary_distances.get((tick, uav_id), ())
                effective_values = {
                    "restricted_region_active": bool(active_regions),
                    "minimum_restricted_boundary_distance_m": (
                        0.0
                        if not active_regions
                        else min(active_region_distances)
                        if active_region_distances
                        else None
                    ),
                    "boundary_margin_m": boundary_margin_m,
                    "active_restricted_region_ids": [
                        region_id for region_id, _ in active_regions
                    ],
                }
                rows.append(
                    _state_row(
                        episode_id,
                        tick,
                        "predicate_contract_restricted_airspace_state",
                        uav_id,
                        "uav",
                        effective_values,
                        _missing(
                            effective_values,
                            (
                                "restricted_region_active",
                                "minimum_restricted_boundary_distance_m",
                                "boundary_margin_m",
                            ),
                        ),
                        [
                            f"trajectories.jsonl#tick={tick}&entity={uav_id}",
                            f"{event_script_path.relative_to(PROJECT_ROOT)}#parameters.conflict_distance_m",
                            *(
                                f"{event_script_path.relative_to(PROJECT_ROOT)}#region={region_id}"
                                for region_id in sorted(self.geometry.regions)
                            ),
                        ],
                        input_digest,
                        "plan_window_computer.restricted_airspace_effective_range",
                        parameter_digest=digest_object(
                            {
                                "boundary_margin_m": boundary_margin_m,
                                "source": "parameters.conflict_distance_m",
                            }
                        ),
                    )
                )
                if uav_id not in tick_positions:
                    continue
                windows = rth_windows.get(uav_id, ())
                rth_active = any(
                    start
                    <= tick
                    < landing_starts.get(uav_id, FORMAL_TICKS[-1] + FORMAL_STEP_TICKS)
                    for start in windows
                )
                role = _entity_role(roster)
                violating_regions = [
                    region_id
                    for region_id, region in active_regions
                    if _inside_restricted_at_tick(
                        tick,
                        uav_id,
                        region,
                        self.geometry.positions_by_tick,
                    )
                ]
                values = {
                    "rth_mode_active": rth_active,
                    "entity_role": role,
                    "cooperative_flag": cooperation[(tick, uav_id)]["value"],
                    "restricted_region_id": violating_regions[0]
                    if violating_regions
                    else None,
                }
                rows.append(
                    _state_row(
                        episode_id,
                        tick,
                        "predicate_contract_aircraft_plan",
                        uav_id,
                        "uav",
                        values,
                        _missing(
                            values,
                            ("rth_mode_active", "entity_role", "cooperative_flag"),
                        ),
                        [
                            cooperation[(tick, uav_id)]["source_ref"],
                            f"{event_script_path.relative_to(PROJECT_ROOT)}#aircraft={uav_id}"
                        ],
                        input_digest,
                        "plan_window_computer.aircraft_control_and_compliance",
                        rule_version="4.0.0",
                        source_semantic_contract={
                            "contract_id": "observed_aircraft_cooperation",
                            "version": "1.0.0",
                            "field_sources": {
                                "plan.cooperative_flag": "record:values.cooperative_flag",
                            },
                            "yaml_field_sources": {
                                "plan.cooperative_flag": "context:plan.cooperative_flag",
                            },
                            "field_units": {"plan.cooperative_flag": "bool"},
                            "role_sources": {"aircraft": "subject_id"},
                            "observed_input_sources": {
                                "plan.cooperative_flag": "truth_frames.entities[].control_state.cooperative_flag",
                            },
                            "authored_value_contract": "strict_bool_when_present",
                            "observation_missing_policy": "absent_or_null_is_unknown",
                        },
                        input_observations=[cooperation[(tick, uav_id)]],
                    )
                )

        rows.sort(
            key=lambda row: (
                int(row["tick"]),
                str(row["observation_family"]),
                str(row["subject_id"]),
            )
        )
        return tuple(rows)


def _indexed_truth_frame_entities(truth_frames_path: Path, episode_id: str):
    """Validate exact episode/time/entity identity before reading observed state."""
    frames = list(read_jsonl(truth_frames_path))
    ticks = [frame.get("tick") for frame in frames]
    if any(type(tick) is not int for tick in ticks) or ticks not in (
        list(range(0, 901)), list(FORMAL_TICKS)
    ):
        raise PredicateStateComputerError(
            f"{truth_frames_path}: observed state requires exact ordered source ticks "
            "0..900 or the exact 5-tick semantic projection"
        )
    for row_index, frame in enumerate(frames, 1):
        if frame.get("schema_name") != "truth_frame" or frame.get("episode_id") != episode_id:
            raise PredicateStateComputerError(
                f"{truth_frames_path}: observed source schema or episode identity mismatch"
            )
        tick = int(frame["tick"])
        entities = frame.get("entities")
        if not isinstance(entities, list):
            raise PredicateStateComputerError(
                f"{truth_frames_path}: tick {tick} entities must be an array"
            )
        indexed: dict[str, Mapping[str, Any]] = {}
        for entity in entities:
            if not isinstance(entity, Mapping):
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} entity must be an object"
                )
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id:
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} entity lacks entity_id"
                )
            if entity_id in indexed:
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} duplicates {entity_id}"
                )
            indexed[entity_id] = entity
        yield row_index, tick, indexed


def _observed_aircraft_cooperation(
    truth_frames_path: Path,
    aircraft_ids: Sequence[str],
    *,
    episode_id: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Read optional observed booleans without deriving missing evidence."""
    relative_path = str(truth_frames_path.resolve().relative_to(PROJECT_ROOT))
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row_index, tick, indexed in _indexed_truth_frame_entities(truth_frames_path, episode_id):
        for aircraft_id in aircraft_ids:
            entity = indexed.get(aircraft_id)
            value: bool | None = None
            status = "missing_entity" if entity is None else "missing_field"
            if entity is not None:
                control = entity.get("control_state")
                if control is not None and not isinstance(control, Mapping):
                    raise PredicateStateComputerError(
                        f"{truth_frames_path}: {aircraft_id}@{tick} control_state must be an object"
                    )
                if isinstance(control, Mapping) and "cooperative_flag" in control:
                    raw_value = control["cooperative_flag"]
                    if raw_value is not None and not isinstance(raw_value, bool):
                        raise PredicateStateComputerError(
                            f"{truth_frames_path}: {aircraft_id}@{tick} "
                            "control_state.cooperative_flag must be a bool or null"
                        )
                    value = raw_value
                    status = "null" if raw_value is None else "observed"
            result[(tick, aircraft_id)] = {
                "source_ref": (
                    f"{relative_path}#row={row_index}&tick={tick}&entity={aircraft_id}"
                    "&field=control_state.cooperative_flag"
                ),
                "path": relative_path,
                "tick": tick,
                "entity_id": aircraft_id,
                "field": "control_state.cooperative_flag",
                "input_selector": "truth_frames.entities[].control_state.cooperative_flag",
                "episode_id": episode_id,
                "value": value,
                "status": status,
            }
    return result


def _state_row(
    episode_id: str,
    tick: int,
    family: str,
    subject_id: str,
    subject_kind: str,
    values: Mapping[str, Any],
    missing_source_record: Sequence[str],
    source_refs: Sequence[str],
    input_digest: str,
    rule_id: str,
    *,
    parameter_digest: str = "registry/governed_defaults",
    rule_version: str = "3.0.0",
    source_semantic_contract: Mapping[str, Any] | None = None,
    input_observations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if missing_source_record and not source_refs:
        raise PredicateStateComputerError(
            f"{family}:{subject_id}@{tick} cannot identify its missing source"
        )
    normalized_missing = [
        item
        if str(item).startswith("missing_source_record:")
        else f"missing_source_record:{source_refs[0]}&field={item}"
        for item in missing_source_record
    ]
    observation_id = stable_identifier(
        "predicate_contract_state", episode_id, tick, family, subject_id, values
    )
    row = {
        "schema_name": "domain_state_observation",
        "schema_version": "3.0.0",
        "annotation_layer": "L0",
        "observation_id": observation_id,
        "episode_id": episode_id,
        "tick": tick,
        "observation_family": family,
        "subject_id": subject_id,
        "subject_kind": subject_kind,
        "subject_category": subject_kind,
        "source_class": "derived_from_observed",
        "rule_id": rule_id,
        "rule_version": rule_version,
        "model_id": rule_id.split(".", 1)[0],
        "model_version": rule_version,
        "input_digest": input_digest,
        "parameter_digest": parameter_digest,
        "seed_digest": input_digest,
        "source_refs": sorted(set(source_refs)),
        "values": dict(values),
        "missing_source_record": normalized_missing,
        "quality": "complete" if not normalized_missing else "unknown_missing_source",
    }
    if source_semantic_contract is not None:
        row["source_semantic_contract"] = deepcopy(dict(source_semantic_contract))
    if input_observations:
        row["input_observations"] = deepcopy(list(input_observations))
    return row


def _boundary_margin_from_script(script: Mapping[str, Any]) -> float:
    boundary_margin_m = float(get_governed_parameter_defaults()["boundary_margin_m"])
    parameters = script.get("parameters")
    configured = (
        _number(parameters.get("conflict_distance_m"))
        if isinstance(parameters, Mapping)
        else None
    )
    if boundary_margin_m <= 0.0:
        raise PredicateStateComputerError(
            "governed boundary_margin_m must be a positive finite number"
        )
    if configured is not None and configured != boundary_margin_m:
        raise PredicateStateComputerError(
            "parameters.conflict_distance_m differs from governed boundary_margin_m"
        )
    triggers = {
        str(trigger.get("trigger_id")): trigger
        for trigger in script.get("triggers", ())
        if isinstance(trigger, Mapping) and isinstance(trigger.get("trigger_id"), str)
    }
    boundary_events = [
        event
        for event in script.get("events", ())
        if isinstance(event, Mapping) and event.get("event_id") == "boundary_conflict"
    ]
    if not boundary_events:
        return boundary_margin_m
    if len(boundary_events) != 1:
        raise PredicateStateComputerError(
            "event script must declare exactly one boundary_conflict event"
        )
    trigger = triggers.get(str(boundary_events[0].get("trigger_ref") or ""))
    if isinstance(trigger, Mapping) and trigger.get("type") == "tick":
        return boundary_margin_m
    if not isinstance(trigger, Mapping) or trigger.get("type") != "entity_proximity":
        raise PredicateStateComputerError(
            "boundary_conflict must reference an entity_proximity trigger"
        )
    trigger_distance_m = _number(trigger.get("distance_m"))
    separation_margin_m = float(
        get_governed_parameter_defaults()["separation_margin_m"]
    )
    if trigger_distance_m is None:
        raise PredicateStateComputerError(
            "boundary_conflict trigger lacks a numeric distance_m"
        )
    declared_response_distance_m = (
        _number(parameters.get("boundary_response_distance_m"))
        if isinstance(parameters, Mapping)
        else None
    )
    if declared_response_distance_m is not None:
        # A scenario may move its response trigger inside the governed conflict
        # envelope without relaxing the envelope itself.  boundary_margin_m stays the
        # predicate and ground-clearance threshold; only the reaction point shrinks,
        # and it is bounded by the governed response distance and by boundary_margin_m.
        governed_response_distance_m = float(
            get_governed_parameter_defaults()["boundary_response_distance_m"]
        )
        if not math.isclose(
            declared_response_distance_m,
            governed_response_distance_m,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise PredicateStateComputerError(
                "parameters.boundary_response_distance_m differs from the governed "
                "boundary_response_distance_m"
            )
        if not 0.0 < declared_response_distance_m <= boundary_margin_m:
            raise PredicateStateComputerError(
                "parameters.boundary_response_distance_m must lie in "
                "(0, governed boundary_margin_m]"
            )
        if not math.isclose(
            trigger_distance_m,
            declared_response_distance_m,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise PredicateStateComputerError(
                "boundary_conflict trigger distance must equal the scenario's declared "
                "boundary_response_distance_m"
            )
        for field in (
            "boundary_response_distance_basis",
            "boundary_response_distance_scope",
        ):
            value = parameters.get(field)
            if not isinstance(value, str) or not value.strip():
                raise PredicateStateComputerError(
                    "parameters.boundary_response_distance_m requires a non-empty "
                    f"{field}"
                )
        return boundary_margin_m
    if not (
        math.isclose(trigger_distance_m, boundary_margin_m, rel_tol=0.0, abs_tol=1e-9)
        or math.isclose(
            trigger_distance_m, separation_margin_m, rel_tol=0.0, abs_tol=1e-9
        )
    ):
        raise PredicateStateComputerError(
            "boundary_conflict trigger distance must equal governed boundary_margin_m "
            "or governed separation_margin_m"
        )
    return boundary_margin_m


def _resolve_event_fire_ticks(
    script: Mapping[str, Any],
    positions: Mapping[int, Mapping[str, tuple[float, float, float]]],
    regions: Mapping[str, Mapping[str, Any]],
    *,
    required_event_ids: set[str],
    weather_by_tick: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    triggers = {
        str(trigger["trigger_id"]): trigger
        for trigger in script.get("triggers", [])
        if isinstance(trigger, Mapping) and isinstance(trigger.get("trigger_id"), str)
    }
    events = {
        str(event["event_id"]): event
        for event in script.get("events", [])
        if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
    }
    missing_required = sorted(required_event_ids - set(events))
    if missing_required:
        raise PredicateStateComputerError(
            f"plan-window events are absent from the script: {missing_required}"
        )
    unresolved = _event_dependency_closure(required_event_ids, events, triggers)
    fire_ticks: dict[str, int] = {}
    while unresolved:
        progress = False
        for event_id in sorted(unresolved):
            event = events[event_id]
            trigger = triggers.get(str(event.get("trigger_ref") or ""))
            if trigger is None:
                raise PredicateStateComputerError(
                    f"event {event_id} references an absent trigger"
                )
            tick = _trigger_tick(
                trigger,
                fire_ticks,
                positions,
                regions,
                triggers=triggers,
                weather_by_tick=weather_by_tick or {},
            )
            if tick is None:
                continue
            fire_ticks[event_id] = _formal_tick_ceiling(tick)
            unresolved.remove(event_id)
            progress = True
            break
        if not progress:
            break
    return fire_ticks


def _event_dependency_closure(
    required_event_ids: set[str],
    events: Mapping[str, Mapping[str, Any]],
    triggers: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Return only the authored event chain needed by plan-window predicates."""

    result = set(required_event_ids)
    pending = list(sorted(required_event_ids))
    while pending:
        event_id = pending.pop()
        event = events[event_id]
        trigger = triggers.get(str(event.get("trigger_ref") or ""))
        if trigger is None:
            raise PredicateStateComputerError(
                f"event {event_id} references an absent trigger"
            )
        for dependency in sorted(_trigger_event_dependencies(trigger, triggers)):
            if dependency not in events:
                raise PredicateStateComputerError(
                    f"event {event_id} depends on absent event {dependency!r}"
                )
            if dependency not in result:
                result.add(dependency)
                pending.append(dependency)
    return result


def _trigger_event_dependencies(
    trigger: Mapping[str, Any],
    triggers: Mapping[str, Mapping[str, Any]],
    *,
    resolving_trigger_ids: frozenset[str] = frozenset(),
) -> set[str]:
    trigger_type = str(trigger.get("type") or "")
    if trigger_type in {"event_fired", "event_fired_after"}:
        event_id = str(trigger.get("event_id") or "")
        if not event_id:
            raise PredicateStateComputerError(
                f"{trigger_type} trigger lacks an event_id"
            )
        return {event_id}
    if trigger_type != "composite":
        return set()
    trigger_id = str(trigger.get("trigger_id") or "")
    if not trigger_id or trigger_id in resolving_trigger_ids:
        raise PredicateStateComputerError(
            f"cyclic or anonymous composite trigger: {trigger_id!r}"
        )
    children = trigger.get("children")
    if not isinstance(children, list) or not children:
        raise PredicateStateComputerError(
            f"composite trigger {trigger_id} lacks children"
        )
    dependencies: set[str] = set()
    for child_id in children:
        child = triggers.get(str(child_id))
        if child is None:
            raise PredicateStateComputerError(
                f"composite trigger {trigger_id} references absent child {child_id!r}"
            )
        dependencies.update(
            _trigger_event_dependencies(
                child,
                triggers,
                resolving_trigger_ids=resolving_trigger_ids | {trigger_id},
            )
        )
    return dependencies


def _trigger_tick(
    trigger: Mapping[str, Any],
    fire_ticks: Mapping[str, int],
    positions: Mapping[int, Mapping[str, tuple[float, float, float]]],
    regions: Mapping[str, Mapping[str, Any]],
    *,
    triggers: Mapping[str, Mapping[str, Any]] | None = None,
    weather_by_tick: Mapping[int, Mapping[str, Any]] | None = None,
    resolving_trigger_ids: frozenset[str] = frozenset(),
) -> int | None:
    trigger_type = str(trigger.get("type") or "")
    if trigger_type == "tick":
        tick = trigger.get("tick")
        if not isinstance(tick, int):
            raise PredicateStateComputerError("tick trigger lacks an integer tick")
        return tick
    if trigger_type in {"event_fired", "event_fired_after"}:
        event_id = str(trigger.get("event_id") or "")
        if event_id not in fire_ticks:
            return None
        delay = 0 if trigger_type == "event_fired" else trigger.get("delay_ticks")
        if isinstance(delay, bool) or not isinstance(delay, int):
            raise PredicateStateComputerError(
                "event_fired_after trigger lacks delay_ticks"
            )
        return fire_ticks[event_id] + delay
    if trigger_type == "weather_state":
        return _weather_trigger_tick(trigger, weather_by_tick or {})
    if trigger_type == "composite":
        trigger_id = str(trigger.get("trigger_id") or "")
        if not trigger_id or trigger_id in resolving_trigger_ids:
            raise PredicateStateComputerError(
                f"cyclic or anonymous composite trigger: {trigger_id!r}"
            )
        trigger_index = triggers or {}
        children = trigger.get("children")
        if not isinstance(children, list) or not children:
            raise PredicateStateComputerError(
                f"composite trigger {trigger_id} lacks children"
            )
        child_ticks: list[int] = []
        for child_id in children:
            child = trigger_index.get(str(child_id))
            if child is None:
                raise PredicateStateComputerError(
                    f"composite trigger {trigger_id} references absent child {child_id!r}"
                )
            child_tick = _trigger_tick(
                child,
                fire_ticks,
                positions,
                regions,
                triggers=trigger_index,
                weather_by_tick=weather_by_tick or {},
                resolving_trigger_ids=resolving_trigger_ids | {trigger_id},
            )
            if child_tick is None:
                return None
            child_ticks.append(child_tick)
        operator = str(trigger.get("operator") or "").upper()
        if operator == "AND":
            return max(child_ticks)
        if operator == "OR":
            return min(child_ticks)
        raise PredicateStateComputerError(
            f"composite trigger {trigger_id} has invalid operator {operator!r}"
        )
    if trigger_type != "entity_proximity":
        raise PredicateStateComputerError(
            f"unsupported plan trigger type: {trigger_type!r}"
        )
    first = str(trigger.get("entity_a") or "")
    second = str(trigger.get("entity_b") or "")
    threshold = _number(trigger.get("distance_m"))
    minimum = trigger.get("min_true_ticks", 1)
    metric = str(trigger.get("metric") or "xy")
    operator = str(trigger.get("operator") or "lte")
    if (
        not first
        or not second
        or threshold is None
        or not isinstance(minimum, int)
        or minimum < 1
    ):
        raise PredicateStateComputerError("entity_proximity trigger is incomplete")
    if metric not in {"xy", "3d", "xy_plus_z"} or operator not in {"lte", "lt"}:
        raise PredicateStateComputerError(
            f"unsupported proximity metric/operator: {metric}/{operator}"
        )
    run = 0
    previous_tick: int | None = None
    for tick in sorted(positions):
        current = positions[tick]
        dynamic_entity = (
            first if second in regions else second if first in regions else None
        )
        if dynamic_entity is not None:
            if dynamic_entity not in current:
                run = 0
                previous_tick = tick
                continue
            region_id = second if second in regions else first
            if metric == "xy_plus_z":
                horizontal_limit = _number(trigger.get("horizontal_distance_m"))
                vertical_limit = _number(trigger.get("vertical_distance_m"))
                if horizontal_limit is None or vertical_limit is None:
                    raise PredicateStateComputerError(
                        "xy_plus_z proximity requires horizontal_distance_m and vertical_distance_m"
                    )
                point = current[dynamic_entity]
                region = regions[region_id]
                horizontal = _distance_to_polygon_xy(point, region["polygon_enu_m"])
                base_z = float(region["base_z_m"])
                top_z = float(region["top_z_m"])
                vertical = max(base_z - point[2], point[2] - top_z, 0.0)
                satisfied = (
                    horizontal <= horizontal_limit and vertical <= vertical_limit
                    if operator == "lte"
                    else horizontal < horizontal_limit and vertical < vertical_limit
                )
                distance = None
            else:
                distance = (
                    _distance_to_polygon_xy(
                        current[dynamic_entity], regions[region_id]["polygon_enu_m"]
                    )
                    if metric == "xy"
                    else _distance_to_restricted_prism(
                        current[dynamic_entity], regions[region_id]
                    )
                )
        elif first in current and second in current:
            if metric == "xy_plus_z":
                horizontal_limit = _number(trigger.get("horizontal_distance_m"))
                vertical_limit = _number(trigger.get("vertical_distance_m"))
                if horizontal_limit is None or vertical_limit is None:
                    raise PredicateStateComputerError(
                        "xy_plus_z proximity requires horizontal_distance_m and vertical_distance_m"
                    )
                horizontal = math.dist(current[first][:2], current[second][:2])
                vertical = abs(current[first][2] - current[second][2])
                satisfied = (
                    horizontal <= horizontal_limit and vertical <= vertical_limit
                    if operator == "lte"
                    else horizontal < horizontal_limit and vertical < vertical_limit
                )
                distance = None
            else:
                dimensions = 2 if metric == "xy" else 3
                distance = math.dist(
                    current[first][:dimensions], current[second][:dimensions]
                )
        else:
            run = 0
            previous_tick = tick
            continue
        if previous_tick is not None and tick != previous_tick + 1:
            run = 0
        if metric != "xy_plus_z":
            satisfied = (
                distance <= threshold if operator == "lte" else distance < threshold
            )
        run = run + 1 if satisfied else 0
        if run >= minimum:
            return tick
        previous_tick = tick
    return None


def _weather_trigger_tick(
    trigger: Mapping[str, Any],
    weather_by_tick: Mapping[int, Mapping[str, Any]],
) -> int | None:
    parameter = str(trigger.get("parameter") or "")
    fields_by_parameter = {
        "fog": ("fog_density",),
        "rain": ("rain",),
        "visibility": ("visibility_m",),
        "wind_speed": ("wind_speed",),
        "temperature": ("temperature_c",),
        "illumination": ("illumination_lux",),
    }
    fields = fields_by_parameter.get(parameter, (parameter,))
    threshold = _number(trigger.get("value"))
    # ``sustain_ticks`` keeps a trigger active *after* its raw condition
    # becomes false in EventScriptInterpreter.  It is not a precondition for
    # the first activation.  The offline resolver only needs the activation
    # tick, so retain the field validation without using it as a raw-condition
    # run length.
    sustain_ticks = trigger.get("sustain_ticks", 0)
    min_true_ticks = trigger.get("min_true_ticks")
    operator = str(trigger.get("operator") or "")
    if (
        not parameter
        or threshold is None
        or isinstance(sustain_ticks, bool)
        or not isinstance(sustain_ticks, int)
        or sustain_ticks < 0
        or operator not in {"gte", "gt", "lte", "lt", "eq"}
    ):
        raise PredicateStateComputerError("weather_state trigger is incomplete")
    if min_true_ticks is not None and (
        isinstance(min_true_ticks, bool)
        or not isinstance(min_true_ticks, int)
        or min_true_ticks < 1
    ):
        raise PredicateStateComputerError(
            "weather_state trigger has invalid min_true_ticks"
        )
    comparisons = {
        "gte": lambda value: value >= threshold,
        "gt": lambda value: value > threshold,
        "lte": lambda value: value <= threshold,
        "lt": lambda value: value < threshold,
        "eq": lambda value: math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12),
    }
    run = 0
    previous_tick: int | None = None
    for tick, row in sorted(weather_by_tick.items()):
        value = next(
            (
                number
                for field in fields
                if (number := _number(row.get(field))) is not None
            ),
            None,
        )
        if previous_tick is not None and tick != previous_tick + 1:
            run = 0
        raw_condition_active = value is not None and comparisons[operator](value)
        run = run + 1 if raw_condition_active else 0
        if raw_condition_active and (
            min_true_ticks is None or run >= min_true_ticks
        ):
            return tick
        previous_tick = tick
    return None


def _weather_by_tick(path: Path) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in read_jsonl(path):
        tick = row.get("tick")
        if not isinstance(tick, int):
            raise PredicateStateComputerError(f"{path}: weather row lacks integer tick")
        if tick in result:
            raise PredicateStateComputerError(f"{path}: duplicate weather tick {tick}")
        result[tick] = row
    if sorted(result) != list(range(0, 901)):
        raise PredicateStateComputerError(
            f"{path}: weather ticks must be exactly 0..900"
        )
    return result


def _load_runtime_state_schedule(epi_id: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for path in sorted(RUNTIME_STATE_PROFILE_ROOT.glob("*.json")):
        profile = _load_object(path)
        if profile.get("schema_name") != "epi_runtime_state_schedule":
            raise PredicateStateComputerError(
                f"{path}: unexpected runtime-state schedule schema"
            )
        profile_id = str(profile.get("profile_id") or "")
        scenarios = profile.get("scenarios")
        if not profile_id or not isinstance(scenarios, list):
            raise PredicateStateComputerError(
                f"{path}: runtime-state schedule header is incomplete"
            )
        for scenario in scenarios:
            if (
                not isinstance(scenario, Mapping)
                or scenario.get("scenario_id") != epi_id
            ):
                continue
            initial = scenario.get("initial_entity_states")
            transitions = scenario.get("event_state_transitions")
            if not isinstance(initial, Mapping) or not isinstance(transitions, Mapping):
                raise PredicateStateComputerError(
                    f"{path}: {epi_id} runtime-state schedule is incomplete"
                )
            normalized_initial = {
                str(entity_id): _validated_runtime_state_patch(
                    state,
                    context=f"{path}#{epi_id}.initial_entity_states[{entity_id}]",
                )
                for entity_id, state in initial.items()
            }
            normalized_transitions: dict[str, list[dict[str, Any]]] = {}
            for event_id, raw_items in transitions.items():
                if not isinstance(raw_items, list):
                    raise PredicateStateComputerError(
                        f"{path}: {epi_id}.{event_id} transitions must be an array"
                    )
                normalized_transitions[str(event_id)] = []
                for index, raw in enumerate(raw_items):
                    if not isinstance(raw, Mapping):
                        raise PredicateStateComputerError(
                            f"{path}: {epi_id}.{event_id}[{index}] must be an object"
                        )
                    entity_id = str(raw.get("entity_id") or "")
                    delay = raw.get("delay_ticks", 0)
                    if (
                        not entity_id
                        or isinstance(delay, bool)
                        or not isinstance(delay, int)
                        or delay < 0
                    ):
                        raise PredicateStateComputerError(
                            f"{path}: {epi_id}.{event_id}[{index}] is incomplete"
                        )
                    normalized_transitions[str(event_id)].append(
                        {
                            "entity_id": entity_id,
                            "delay_ticks": delay,
                            "state_patch": _validated_runtime_state_patch(
                                raw.get("state_patch"),
                                context=f"{path}#{epi_id}.{event_id}[{index}]",
                            ),
                        }
                    )
            matches.append(
                {
                    "profile_id": profile_id,
                    "source_path": path.resolve(),
                    "initial_entity_states": normalized_initial,
                    "event_state_transitions": normalized_transitions,
                }
            )
    if len(matches) != 1:
        raise PredicateStateComputerError(
            f"runtime-state schedule authority must contain exactly one {epi_id}: {len(matches)}"
        )
    return matches[0]


def _validated_runtime_state_patch(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise PredicateStateComputerError(
            f"{context}: runtime-state patch must be a non-empty object"
        )
    unknown_families = sorted(set(map(str, value)) - set(RUNTIME_STATE_FIELDS))
    unconsumed = unconsumed_runtime_state_paths(value)
    invalid = invalid_runtime_state_value_paths(value)
    if unknown_families or unconsumed or invalid:
        raise PredicateStateComputerError(
            f"{context}: invalid runtime-state patch; unknown_families={unknown_families}, "
            f"unconsumed={unconsumed}, invalid_values={invalid}"
        )
    result: dict[str, Any] = {}
    for family, raw_state in value.items():
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise PredicateStateComputerError(
                f"{context}: runtime-state family {family} must be non-empty"
            )
        result[str(family)] = deepcopy(dict(raw_state))
    return result


def _deep_merge(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        key_text = str(key)
        if isinstance(value, Mapping) and isinstance(target.get(key_text), dict):
            _deep_merge(target[key_text], value)
        else:
            target[key_text] = deepcopy(value)


def _runtime_schedule_writes_field(
    schedule: Mapping[str, Any],
    family: str,
    field: str,
) -> bool:
    for state in schedule.get("initial_entity_states", {}).values():
        if (
            isinstance(state, Mapping)
            and isinstance(state.get(family), Mapping)
            and field in state[family]
        ):
            return True
    for transitions in schedule.get("event_state_transitions", {}).values():
        for transition in transitions:
            patch = (
                transition.get("state_patch")
                if isinstance(transition, Mapping)
                else None
            )
            if (
                isinstance(patch, Mapping)
                and isinstance(patch.get(family), Mapping)
                and field in patch[family]
            ):
                return True
    return False


def _merge_missing_runtime_state(
    target: dict[str, Any],
    planned: Mapping[str, Any],
    *, family: str,
) -> None:
    # These pairs are read as the same GNSS input by _gnss_values. A nominal
    # baseline must not create a second, contradictory value beside an actual
    # runtime field. Both explicitly supplied values remain for conflict checks.
    navigation_groups = (
        ("visual_relocalization", "visual_relocalization_active"),
        ("gnss_spoofed", "spoofing_active"),
        ("geofence_alert", "geofence_violation"),
        ("mission_recovered", "relocalization_complete"),
        ("operational_state", "gnss_mode", "navigation_mode", "recovery_state"),
    ) if family == "navigation_state" else ()
    for key, value in planned.items():
        key_text = str(key)
        if any(key_text in group and any(member in target for member in group)
               for group in navigation_groups):
            continue
        if key_text not in target:
            target[key_text] = deepcopy(value)


def _control_windows(
    events: Sequence[Mapping[str, Any]],
    fire_ticks: Mapping[str, int],
    event_ids: set[str],
) -> dict[str, tuple[int, ...]]:
    starts: dict[str, list[int]] = {}
    for event in events:
        event_id = str(event.get("event_id") or "")
        if event_id not in event_ids:
            continue
        start = fire_ticks.get(event_id)
        if start is None:
            continue
        for entity_id in _moved_entity_ids(event):
            starts.setdefault(entity_id, []).append(start)
    return {entity_id: tuple(sorted(values)) for entity_id, values in starts.items()}


def _event_activates_rth(event: Mapping[str, Any]) -> bool:
    for action in event.get("actions", ()):
        if not isinstance(action, Mapping):
            continue
        visual_state = action.get("visual_state")
        if (
            isinstance(visual_state, Mapping)
            and visual_state.get("mode") == "return_to_home"
        ):
            return True
        state_patch = action.get("state_patch")
        control_state = (
            state_patch.get("control_state")
            if isinstance(state_patch, Mapping)
            else None
        )
        if (
            isinstance(control_state, Mapping)
            and control_state.get("rth_active") is True
        ):
            return True
    return False


def _landing_start_ticks(
    events: Sequence[Mapping[str, Any]], fire_ticks: Mapping[str, int]
) -> dict[str, int]:
    result: dict[str, int] = {}
    for event in events:
        if "landing" not in str(event.get("intent") or "").lower():
            continue
        start = fire_ticks.get(str(event.get("event_id") or ""))
        if start is None:
            continue
        for entity_id in _moved_entity_ids(event):
            result[entity_id] = min(start, result.get(entity_id, start))
    return result


def _moved_entity_ids(event: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(action["entity_id"])
                for action in event.get("actions", [])
                if isinstance(action, Mapping)
                and action.get("type") == "move_entity"
                and isinstance(action.get("entity_id"), str)
            }
        )
    )


def _inside_restricted_at_tick(
    tick: int,
    uav_id: str,
    region: Mapping[str, Any],
    positions: Mapping[int, Mapping[str, tuple[float, float, float]]],
) -> bool:
    position = positions.get(tick, {}).get(uav_id)
    if position is None:
        return False
    return _point_in_polygon_xy(position, region["polygon_enu_m"]) and float(
        region["base_z_m"]
    ) <= position[2] <= float(region["top_z_m"])


# Producer/registry authorities only (no entity_id / substring guessing):
# - roster/scene category facade_anchor (world_core_catalog evidence)
# - input_adapter entity_kind structure.facade_anchor + placement_mode facade_anchor
# - ontology_class_id world:BuildingStructure (templates/pair geometry/reference_policy)
# - world:Building / world:BuildingFacade = world-namespace forms of catalog class
#   ids Building / BuildingFacade (ontology_iri .../world#). Unprefixed catalog
#   ids are not emitted as ontology_class_id by producers and are not accepted.
_BUILDING_STRUCTURE_CATEGORIES = frozenset({"facade_anchor"})
_BUILDING_STRUCTURE_KINDS = frozenset({"structure.facade_anchor"})
_BUILDING_STRUCTURE_PLACEMENT_MODES = frozenset({"facade_anchor"})
_BUILDING_STRUCTURE_ONTOLOGY_CLASS_IDS = frozenset(
    {
        "world:BuildingStructure",
        "world:Building",
        "world:BuildingFacade",
    }
)


def _is_building_structure(entity_id: str, entity: Mapping[str, Any]) -> bool:
    """Return whether an entity is a typed building/facade structure.

    Authority is exact category / entity_kind / placement_mode / ontology class
    only. Entity-id spelling and substring token matching are not used: missing
    typed fields mean unknown (not a structure), never a name-based guess.
    """

    del entity_id  # identifier spelling is never structure authority
    category = str(entity.get("entity_category") or entity.get("category") or "")
    if category in _BUILDING_STRUCTURE_CATEGORIES:
        return True
    kind = str(entity.get("entity_kind") or entity.get("entity_type") or "")
    if kind in _BUILDING_STRUCTURE_KINDS:
        return True
    placement_mode = entity.get("placement_mode")
    if (
        isinstance(placement_mode, str)
        and placement_mode in _BUILDING_STRUCTURE_PLACEMENT_MODES
    ):
        return True
    ontology_ids: list[str] = []
    direct = entity.get("ontology_class_id")
    if isinstance(direct, str) and direct:
        ontology_ids.append(direct)
    scope = entity.get("semantic_scope")
    if isinstance(scope, Mapping):
        scoped = scope.get("ontology_class_id")
        if isinstance(scoped, str) and scoped:
            ontology_ids.append(scoped)
    classes = entity.get("ontology_class_ids")
    if isinstance(classes, (list, tuple)):
        ontology_ids.extend(
            value for value in classes if isinstance(value, str) and value
        )
    return any(value in _BUILDING_STRUCTURE_ONTOLOGY_CLASS_IDS for value in ontology_ids)


def _aircraft_pair_separation_models(
    script: Mapping[str, Any],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    separation_margin_m: float,
) -> dict[tuple[str, str], dict[str, float | str]]:
    """Compile anisotropic flight-separation volumes into one governed metric.

    The ontology predicate keeps its frozen scalar comparison.  An authored
    ``xy_plus_z`` proximity trigger between two UAVs declares a horizontal and a
    vertical limit; mapping that cylinder to an equivalent scalar distance
    ``max(horizontal/H, vertical/V) * separation_margin_m`` makes the scalar
    comparison exact: the equivalent distance is below the governed margin iff
    both configured component limits are satisfied.
    """

    result: dict[tuple[str, str], dict[str, float | str]] = {}
    for trigger in script.get("triggers", ()):
        if (
            not isinstance(trigger, Mapping)
            or trigger.get("type") != "entity_proximity"
        ):
            continue
        first = str(trigger.get("entity_a") or "")
        second = str(trigger.get("entity_b") or "")
        if (
            _entity_category(entity_metadata.get(first, {})) != "uav"
            or _entity_category(entity_metadata.get(second, {})) != "uav"
            or trigger.get("metric") != "xy_plus_z"
        ):
            continue
        horizontal_limit = _number(trigger.get("horizontal_distance_m"))
        vertical_limit = _number(trigger.get("vertical_distance_m"))
        if (
            horizontal_limit is None
            or vertical_limit is None
            or horizontal_limit <= 0.0
            or vertical_limit <= 0.0
        ):
            raise PredicateStateComputerError(
                f"aircraft separation trigger {trigger.get('trigger_id')} lacks positive limits"
            )
        key = tuple(sorted((first, second)))
        model: dict[str, float | str] = {
            "metric": "anisotropic_safety_volume_equivalent_distance",
            "horizontal_limit_m": horizontal_limit,
            "vertical_limit_m": vertical_limit,
            "separation_margin_m": separation_margin_m,
            "trigger_id": str(trigger.get("trigger_id") or ""),
        }
        if key in result and result[key] != model:
            raise PredicateStateComputerError(
                f"aircraft pair {key} has conflicting separation models"
            )
        result[key] = model
    return result


def _separation_model_record(
    model: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The single compiled model for one pair, or the Euclidean fallback marker."""
    if model is None:
        return {"metric": "euclidean_3d"}
    return {
        "metric": str(model["metric"]),
        "horizontal_limit_m": float(model["horizontal_limit_m"]),
        "vertical_limit_m": float(model["vertical_limit_m"]),
        "trigger_id": str(model.get("trigger_id") or ""),
    }


def _separation_model_records(
    models: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"pair_entity_ids": list(pair), **dict(model)}
        for pair, model in sorted(models.items())
    ]


def _equivalent_aircraft_separation_distance(
    *,
    euclidean_distance: float,
    horizontal_distance: float,
    vertical_distance: float,
    model: Mapping[str, Any] | None,
    separation_margin_m: float,
) -> float:
    """Map a pair geometry to the scalar the frozen separation predicate compares.

    No compiled model means the pair has no authored ``xy_plus_z`` volume, so the
    Euclidean 3-D distance stays the metric.  With a model, the equivalent scalar
    is below ``separation_margin_m`` exactly when both component limits hold.
    """

    if model is None:
        return euclidean_distance
    horizontal_limit = float(model["horizontal_limit_m"])
    vertical_limit = float(model["vertical_limit_m"])
    return (
        max(horizontal_distance / horizontal_limit, vertical_distance / vertical_limit)
        * separation_margin_m
    )


def _position_at_tick_or_static(
    entity_id: str,
    tick_positions: Mapping[str, tuple[float, float, float]],
    entity_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, float] | None:
    """Tick sample first; static metadata pose only for non-tick entities.

    Mobile agent proximity subjects must not use this fallback for "current"
    geometry: callers that emit dynamic nearest_* for uav/vehicle/pedestrian
    require ``entity_id in tick_positions`` and read that sample directly.
    Static building/facade partners may still resolve via metadata pose.
    """

    return tick_positions.get(entity_id) or _entity_position(
        entity_metadata.get(entity_id, {})
    )


def _append_agent_proximity_rows(
    *,
    rows: list[dict[str, Any]],
    episode_id: str,
    tick: int,
    tick_positions: Mapping[str, tuple[float, float, float]],
    roster_by_id: Mapping[str, Mapping[str, Any]],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    structure_ids: Sequence[str],
    separation_margin_m: float,
    event_script_path: Path,
    input_digest: str,
    separation_models: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    """Emit agent proximity only for mobiles with a real tick sample.

    Roster still supplies typed category for subjects that are present. Absent
    mobiles (roster-only / future-first-pose) do not publish known nearest_*
    geometry and do not cite a missing trajectories.jsonl tick row.
    """

    proximity_subjects = sorted(
        entity_id
        for entity_id, entity in roster_by_id.items()
        if _entity_category(entity) in {"uav", "vehicle", "pedestrian"}
    )
    for subject_id in proximity_subjects:
        if subject_id not in tick_positions:
            continue
        subject_position = tick_positions[subject_id]
        subject_category = _entity_category(roster_by_id[subject_id])
        proximity_values = _agent_proximity_values(
            subject_id=subject_id,
            subject_position=subject_position,
            tick_positions=tick_positions,
            entity_metadata=entity_metadata,
            structure_ids=structure_ids,
            separation_models=separation_models,
            separation_margin_m=separation_margin_m,
        )
        required_fields = {
            "uav": (
                "nearest_aircraft_distance_m",
                "nearest_ground_vehicle_distance_m",
                "nearest_pedestrian_distance_m",
                "nearest_population_distance_m",
                "nearest_building_distance_m",
            ),
            "vehicle": ("nearest_pedestrian_distance_m",),
            "pedestrian": ("nearest_vehicle_distance_m",),
        }[subject_category]
        rows.append(
            _state_row(
                episode_id,
                tick,
                "predicate_contract_agent_proximity_geometry",
                subject_id,
                subject_category,
                proximity_values,
                _missing(proximity_values, required_fields),
                [
                    f"trajectories.jsonl#tick={tick}&entity={subject_id}",
                    str(event_script_path.relative_to(PROJECT_ROOT)),
                ],
                input_digest,
                "geometric_computer.agent_proximity",
                parameter_digest=digest_object(
                    {
                        "separation_margin_m": separation_margin_m,
                        "aircraft_pair_separation_models": _separation_model_records(
                            separation_models
                        ),
                        "distance_basis": "original_scenario_trajectory_then_world_supplement",
                    }
                ),
            )
        )


def _agent_proximity_values(
    *,
    subject_id: str,
    subject_position: tuple[float, float, float] | None,
    tick_positions: Mapping[str, tuple[float, float, float]],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    structure_ids: Sequence[str],
    separation_models: Mapping[tuple[str, str], Mapping[str, Any]],
    separation_margin_m: float,
) -> dict[str, Any]:
    fields = (
        "nearest_aircraft",
        "nearest_ground_vehicle",
        "nearest_vehicle",
        "nearest_pedestrian",
        "nearest_population",
        "nearest_building",
    )
    result: dict[str, Any] = {f"{field}_distance_m": None for field in fields}
    result.update({f"{field}_id": None for field in fields})
    if subject_position is None:
        return result

    candidate_ids = sorted(set(tick_positions) | set(structure_ids))
    category_sets = {
        "nearest_aircraft": {"uav"},
        "nearest_ground_vehicle": {"vehicle"},
        "nearest_vehicle": {"vehicle"},
        "nearest_pedestrian": {"pedestrian"},
        "nearest_population": {"vehicle", "pedestrian"},
    }
    for field, wanted_categories in category_sets.items():
        candidates: list[tuple[float, str]] = []
        for other_id in candidate_ids:
            if other_id == subject_id:
                continue
            metadata = entity_metadata.get(other_id, {})
            if _entity_category(metadata) not in wanted_categories:
                continue
            other_position = _position_at_tick_or_static(
                other_id,
                tick_positions,
                entity_metadata,
            )
            if other_position is None:
                continue
            euclidean = math.dist(subject_position, other_position)
            distance = euclidean
            if field == "nearest_aircraft":
                horizontal = math.dist(subject_position[:2], other_position[:2])
                vertical = abs(subject_position[2] - other_position[2])
                distance = _equivalent_aircraft_separation_distance(
                    euclidean_distance=euclidean,
                    horizontal_distance=horizontal,
                    vertical_distance=vertical,
                    model=separation_models.get(
                        tuple(sorted((subject_id, other_id)))
                    ),
                    separation_margin_m=separation_margin_m,
                )
            candidates.append((distance, other_id))
        if candidates:
            distance, other_id = min(candidates)
            result[f"{field}_distance_m"] = distance
            result[f"{field}_id"] = other_id

    structures: list[tuple[float, str]] = []
    for structure_id in structure_ids:
        if structure_id == subject_id:
            continue
        position = _position_at_tick_or_static(
            structure_id,
            tick_positions,
            entity_metadata,
        )
        if position is not None:
            structures.append((math.dist(subject_position, position), structure_id))
    if structures:
        distance, structure_id = min(structures)
        result["nearest_building_distance_m"] = distance
        result["nearest_building_id"] = structure_id
    return result


def _append_full_pair_geometry_rows(
    *,
    rows: list[dict[str, Any]],
    episode_id: str,
    tick: int,
    tick_positions: Mapping[str, tuple[float, float, float]],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    uav_ids: Sequence[str],
    vehicle_ids: Sequence[str],
    pedestrian_ids: Sequence[str],
    structure_ids: Sequence[str],
    event_script_path: Path,
    input_digest: str,
) -> None:
    """Emit every grounded heterogeneous safety relation for one tick.

    Nearest-neighbour aggregates are intentionally not used: each assertion
    has its own role-complete candidate row and pair-specific distance.
    """

    def append_pair(
        family: str,
        subject_kind: str,
        first_role: str,
        first_id: str,
        second_role: str,
        second_id: str,
        *,
        extra_values: Mapping[str, Any] | None = None,
        second_source_ref: str | None = None,
        parameter_digest: str = "registry/governed_defaults",
    ) -> None:
        first_position = _position_at_tick_or_static(
            first_id, tick_positions, entity_metadata
        )
        second_position = _position_at_tick_or_static(
            second_id, tick_positions, entity_metadata
        )
        if first_position is None or second_position is None:
            raise PredicateStateComputerError(
                f"{family}:{first_id}|{second_id}@{tick} lacks pair geometry"
            )
        values = {
            f"{first_role}_id": first_id,
            f"{second_role}_id": second_id,
            "distance_m": math.dist(first_position, second_position),
        }
        role_classes = {
            "aircraft": "world:UnmannedAircraft",
            "vehicle": "world:GroundVehicle",
            "pedestrian": "world:Pedestrian",
            "structure": "world:BuildingStructure",
        }
        values[f"{first_role}_ontology_class_id"] = role_classes[first_role]
        values[f"{second_role}_ontology_class_id"] = role_classes[second_role]
        values.update(dict(extra_values or {}))
        rows.append(
            _state_row(
                episode_id,
                tick,
                family,
                f"{first_id}|{second_id}",
                subject_kind,
                values,
                [],
                [
                    f"trajectories.jsonl#tick={tick}&entity={first_id}",
                    second_source_ref
                    or f"trajectories.jsonl#tick={tick}&entity={second_id}",
                ],
                input_digest,
                "geometric_computer.full_grounded_pair_distance",
                parameter_digest=parameter_digest,
            )
        )

    for aircraft_id in sorted(uav_ids):
        for structure_id in sorted(structure_ids):
            append_pair(
                "predicate_contract_aircraft_structure_geometry",
                "uav_structure",
                "aircraft",
                aircraft_id,
                "structure",
                structure_id,
                second_source_ref=(
                    f"{event_script_path.relative_to(PROJECT_ROOT)}"
                    f"#building_structure={structure_id}"
                ),
            )
        for vehicle_id in sorted(vehicle_ids):
            append_pair(
                "predicate_contract_aircraft_vehicle_geometry",
                "uav_vehicle",
                "aircraft",
                aircraft_id,
                "vehicle",
                vehicle_id,
            )
        for pedestrian_id in sorted(pedestrian_ids):
            append_pair(
                "predicate_contract_aircraft_pedestrian_geometry",
                "uav_pedestrian",
                "aircraft",
                aircraft_id,
                "pedestrian",
                pedestrian_id,
            )

    for vehicle_id in sorted(vehicle_ids):
        for pedestrian_id in sorted(pedestrian_ids):
            append_pair(
                "predicate_contract_vehicle_pedestrian_geometry",
                "vehicle_pedestrian",
                "vehicle",
                vehicle_id,
                "pedestrian",
                pedestrian_id,
            )


def load_trajectory_vectors(
    world_path: Path,
    scenario_path: Path | None,
) -> tuple[
    dict[int, dict[str, tuple[float, float, float]]],
    dict[int, dict[str, tuple[float, float, float]]],
    dict[int, dict[str, str]],
]:
    positions: dict[int, dict[str, tuple[float, float, float]]] = {}
    velocities: dict[int, dict[str, tuple[float, float, float]]] = {}
    activities: dict[int, dict[str, str]] = {}
    selected: set[tuple[int, str]] = set()
    paths = (world_path,) if scenario_path is None else (world_path, scenario_path)
    for path in paths:
        seen: set[tuple[int, str]] = set()
        for row in read_jsonl(path):
            tick = row.get("tick")
            entity_id = row.get("entity_id")
            position = _vector3(row.get("pos_enu"))
            velocity = _vector3(row.get("vel_mps"))
            if (
                not isinstance(tick, int)
                or not isinstance(entity_id, str)
                or not entity_id
            ):
                raise PredicateStateComputerError(
                    f"{path}: trajectory row lacks integer tick or entity_id"
                )
            if position is None:
                raise PredicateStateComputerError(
                    f"{path}: trajectory row lacks numeric pos_enu for {(tick, entity_id)}"
                )
            key = (tick, entity_id)
            if key in seen:
                raise PredicateStateComputerError(
                    f"{path}: duplicate trajectory row {key}"
                )
            seen.add(key)
            if key in selected:
                continue
            selected.add(key)
            positions.setdefault(tick, {})[entity_id] = position
            if velocity is not None:
                velocities.setdefault(tick, {})[entity_id] = velocity
            activity = str(row.get("activity_type") or row.get("state") or "").strip()
            if activity:
                activities.setdefault(tick, {})[entity_id] = activity
    if not positions:
        raise PredicateStateComputerError(
            f"no numeric trajectory poses in {world_path}"
        )
    return positions, velocities, activities


def _trajectory_state_digest(
    positions: Mapping[int, Mapping[str, tuple[float, float, float]]],
    velocities: Mapping[int, Mapping[str, tuple[float, float, float]]],
) -> str:
    return digest_object(
        [
            {
                "tick": tick,
                "entity_id": entity_id,
                "position_enu_m": list(position),
                "velocity_enu_mps": (
                    list(velocities[tick][entity_id])
                    if entity_id in velocities.get(tick, {})
                    else None
                ),
            }
            for tick, tick_positions in sorted(positions.items())
            for entity_id, position in sorted(tick_positions.items())
        ]
    )


def _restricted_regions(
    scene_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entity_id, entity in scene_by_id.items():
        placement = (
            entity.get("placement")
            if isinstance(entity.get("placement"), Mapping)
            else {}
        )
        polygon = placement.get("polygon_enu_m")
        category = str(entity.get("category") or "").lower()
        asset = str(entity.get("logical_asset_id") or "").lower()
        semantic_scope = entity.get("semantic_scope")
        if not isinstance(semantic_scope, Mapping):
            semantic_scope = {}
        is_authoritative_no_fly_zone = (
            entity.get("category") == "facility"
            and entity.get("entity_kind") == "facility.no_fly_zone"
            and semantic_scope.get("scope_type") == "facility"
            and semantic_scope.get("scope_subtype") == "no_fly_zone"
            and semantic_scope.get("ontology_class_id") == "world:NoFlyZone"
            and semantic_scope.get("authority_id") == "facility_scope_contract.v1"
        )
        if (
            category != "airspace_constraint"
            and asset != "trigger.no_fly.box.v1"
            and not is_authoritative_no_fly_zone
        ):
            continue
        if isinstance(polygon, list) and len(polygon) >= 3:
            points = tuple(_vector2(point) for point in polygon)
            if any(point is None for point in points):
                raise PredicateStateComputerError(
                    f"region {entity_id} has a non-numeric polygon"
                )
            base = _number(placement.get("base_z_m"))
            height = _number(placement.get("height_m"))
            if base is None or height is None or height <= 0:
                raise PredicateStateComputerError(
                    f"region {entity_id} lacks a valid vertical prism"
                )
        elif entity.get("placement_mode") == "box_volume":
            center = _vector3(placement.get("center_enu_m"))
            extent = _vector3(placement.get("extent_m"))
            if (
                center is None
                or extent is None
                or any(value <= 0.0 for value in extent)
            ):
                raise PredicateStateComputerError(
                    f"region {entity_id} lacks numeric box center/extent geometry"
                )
            points = (
                (center[0] - extent[0], center[1] - extent[1]),
                (center[0] + extent[0], center[1] - extent[1]),
                (center[0] + extent[0], center[1] + extent[1]),
                (center[0] - extent[0], center[1] + extent[1]),
            )
            base = center[2] - extent[2]
            height = 2.0 * extent[2]
        else:
            raise PredicateStateComputerError(
                f"restricted region {entity_id} lacks polygon or box-volume geometry"
            )
        result[entity_id] = {
            "polygon_enu_m": tuple(point for point in points if point is not None),
            "base_z_m": base,
            "top_z_m": base + height,
            "activation_tick": int(entity.get("activation_tick", 0)),
            "deactivation_tick": entity.get("deactivation_tick"),
            "spawn_policy": entity.get("spawn_policy"),
            "logical_asset_id": entity.get("logical_asset_id"),
        }
    return result


def restricted_region_activity_by_tick(
    truth_frames_path: Path,
    regions: Mapping[str, Mapping[str, Any]],
) -> dict[int, dict[str, bool]]:
    rows = list(read_jsonl(truth_frames_path))
    observed_ticks = [row.get("tick") for row in rows]
    if observed_ticks not in (list(range(0, 901)), list(FORMAL_TICKS)):
        raise PredicateStateComputerError(
            "restricted-region runtime truth requires exact ordered source ticks "
            "0..900 or the exact 5-tick semantic projection"
        )

    result: dict[int, dict[str, bool]] = {}
    dynamic_region_ids = {
        region_id
        for region_id, region in regions.items()
        if region.get("spawn_policy") == "event_script_only"
    }
    presence_ticks = {region_id: set() for region_id in dynamic_region_ids}
    dense_activity = {region_id: [] for region_id in dynamic_region_ids}
    for row in rows:
        tick = int(row["tick"])
        entities = row.get("entities")
        if not isinstance(entities, list):
            raise PredicateStateComputerError(
                f"{truth_frames_path}: tick {tick} entities must be an array"
            )
        entity_by_id: dict[str, Mapping[str, Any]] = {}
        for entity in entities:
            if not isinstance(entity, Mapping):
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} entity must be an object"
                )
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id:
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} entity lacks entity_id"
                )
            if entity_id in entity_by_id:
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} duplicates {entity_id}"
                )
            entity_by_id[entity_id] = entity

        tick_activity: dict[str, bool] = {}
        projected_activity = row.get("restricted_region_runtime_activity")
        if projected_activity is not None:
            if not isinstance(projected_activity, Mapping) or any(
                not isinstance(entity_id, str) or not isinstance(active, bool)
                for entity_id, active in projected_activity.items()
            ):
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} has invalid projected restricted-region activity"
                )
            unknown_projected_ids = sorted(set(projected_activity) - set(regions))
            if unknown_projected_ids:
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} projects unknown restricted regions {unknown_projected_ids}"
                )
        for region_id, region in sorted(regions.items()):
            if region.get("spawn_policy") != "event_script_only":
                activation_tick = int(region.get("activation_tick", 0))
                deactivation_tick = region.get("deactivation_tick")
                tick_activity[region_id] = tick >= activation_tick and (
                    not isinstance(deactivation_tick, int) or tick < deactivation_tick
                )
                continue
            if projected_activity is not None:
                if region_id in projected_activity:
                    presence_ticks[region_id].add(tick)
                tick_activity[region_id] = bool(
                    projected_activity.get(region_id, False)
                )
                dense_activity[region_id].append(tick_activity[region_id])
                continue
            entity = entity_by_id.get(region_id)
            if entity is None:
                tick_activity[region_id] = False
                dense_activity[region_id].append(False)
                continue
            presence_ticks[region_id].add(tick)
            if entity.get("logical_asset_id") != region.get("logical_asset_id"):
                raise PredicateStateComputerError(
                    f"{truth_frames_path}: tick {tick} region {region_id} has wrong asset identity"
                )
            tick_activity[region_id] = restricted_region_runtime_active(entity)
            dense_activity[region_id].append(tick_activity[region_id])
        if tick in FORMAL_TICKS:
            result[tick] = tick_activity
    if set(result) != set(FORMAL_TICKS):
        raise PredicateStateComputerError(
            "restricted-region runtime truth lacks the complete formal grid"
        )
    for region_id in sorted(dynamic_region_ids):
        observed_presence = presence_ticks[region_id]
        if not observed_presence:
            raise PredicateStateComputerError(
                f"dynamic restricted region {region_id} never appears in runtime truth"
            )
        first_presence_tick = min(observed_presence)
        expected_presence = {
            int(tick) for tick in observed_ticks if int(tick) >= first_presence_tick
        }
        if observed_presence != expected_presence:
            raise PredicateStateComputerError(
                f"dynamic restricted region {region_id} has a runtime-truth presence gap"
            )
        if observed_ticks == list(range(0, 901)):
            values = dense_activity[region_id]
            for boundary_tick in range(FORMAL_STEP_TICKS, 901, FORMAL_STEP_TICKS):
                changes = sum(
                    values[tick] != values[tick - 1]
                    for tick in range(
                        boundary_tick - FORMAL_STEP_TICKS + 1,
                        boundary_tick + 1,
                    )
                )
                if changes > 1:
                    raise PredicateStateComputerError(
                        f"dynamic restricted region {region_id} changes activity "
                        f"more than once in capture interval ending {boundary_tick}"
                    )
    return result


def restricted_region_runtime_active(entity: Mapping[str, Any]) -> bool:
    """Runtime activation of an authored trigger volume.

    Authority order matches the rest of the runtime-state layer:

    1. the typed runtime family the event script actually writes
       (`incident_state.temporary_lockdown_active`, then `constraint_state.active`);
    2. the authored activity label the producer projects into `annotations`, for
       frames that carry a label but no typed state.

    A frame that carries neither is an input error, never an implicit false.
    """
    incident_state = entity.get("incident_state")
    if isinstance(incident_state, Mapping):
        declared = incident_state.get("temporary_lockdown_active")
        if isinstance(declared, bool):
            return declared
    constraint_state = entity.get("constraint_state")
    if isinstance(constraint_state, Mapping):
        declared = constraint_state.get("active")
        if isinstance(declared, bool):
            return declared
    annotations = entity.get("annotations")
    activity_type = (
        annotations.get("activity_type") if isinstance(annotations, Mapping) else None
    )
    if activity_type == "active":
        return True
    if activity_type in {"idle", "inactive", "standdown"}:
        return False
    # A trigger volume that is present but carries no activation statement is
    # neither active nor provably inactive; surface it instead of guessing.
    state = entity.get("state")
    if isinstance(state, str):
        return state in {"isolation_active", "lockdown", "deployed", "isolating"}
    raise PredicateStateComputerError(
        f"restricted region has unsupported runtime activity {activity_type!r} "
        f"(no typed incident_state/constraint_state either)"
    )


def _protected_regions(
    scene_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load only explicitly typed protected-airspace scene entities.

    An episode with no such entity has an empty candidate set.  An explicitly
    typed entity without a complete prism is an input error; it is never
    collapsed to an episode-wide false value.
    """

    result: dict[str, dict[str, Any]] = {}
    for entity_id, entity in sorted(scene_by_id.items()):
        semantic_scope = (
            entity.get("semantic_scope")
            if isinstance(entity.get("semantic_scope"), Mapping)
            else {}
        )
        identity_text = " ".join(
            str(value).casefold()
            for value in (
                entity.get("category"),
                entity.get("entity_category"),
                entity.get("entity_kind"),
                entity.get("entity_type"),
                entity.get("logical_asset_id"),
                semantic_scope.get("ontology_class_id"),
            )
            if isinstance(value, str)
        )
        if not any(
            token in identity_text
            for token in ("protectedairspace", "protected_airspace")
        ):
            continue
        placement = (
            entity.get("placement")
            if isinstance(entity.get("placement"), Mapping)
            else {}
        )
        polygon = placement.get("polygon_enu_m")
        if isinstance(polygon, list) and len(polygon) >= 3:
            points = tuple(_vector2(point) for point in polygon)
            if any(point is None for point in points):
                raise PredicateStateComputerError(
                    f"protected region {entity_id} has a non-numeric polygon"
                )
            base = _number(placement.get("base_z_m"))
            height = _number(placement.get("height_m"))
            if base is None or height is None or height <= 0.0:
                raise PredicateStateComputerError(
                    f"protected region {entity_id} lacks a valid vertical prism"
                )
        elif entity.get("placement_mode") == "box_volume":
            center = _vector3(placement.get("center_enu_m"))
            extent = _vector3(placement.get("extent_m"))
            if (
                center is None
                or extent is None
                or any(value <= 0.0 for value in extent)
            ):
                raise PredicateStateComputerError(
                    f"protected region {entity_id} lacks numeric box geometry"
                )
            points = (
                (center[0] - extent[0], center[1] - extent[1]),
                (center[0] + extent[0], center[1] - extent[1]),
                (center[0] + extent[0], center[1] + extent[1]),
                (center[0] - extent[0], center[1] + extent[1]),
            )
            base = center[2] - extent[2]
            height = 2.0 * extent[2]
        else:
            raise PredicateStateComputerError(
                f"protected region {entity_id} lacks polygon or box-volume geometry"
            )
        result[entity_id] = {
            "polygon_enu_m": tuple(point for point in points if point is not None),
            "base_z_m": base,
            "top_z_m": base + height,
        }
    return result


def _airspace_corridors(
    scene_by_id: Mapping[str, Mapping[str, Any]],
    separation_margin_m: float,
) -> dict[str, dict[str, Any]]:
    """Compile the authored corridor boxes into a geometric capacity model.

    Corridor capacity is the number of independent lateral/vertical lanes that
    can maintain the governed aircraft center-separation margin.  Longitudinal
    length is deliberately excluded: one segment is one reservation cell.
    """

    if not math.isfinite(separation_margin_m) or separation_margin_m <= 0.0:
        raise PredicateStateComputerError(
            "separation_margin_m must be a positive finite number"
        )
    result: dict[str, dict[str, Any]] = {}
    for entity_id, entity in sorted(scene_by_id.items()):
        if _entity_category(entity) != "airspace_corridor":
            continue
        if entity.get("placement_mode") != "box_volume":
            raise PredicateStateComputerError(
                f"corridor {entity_id} must use box_volume placement"
            )
        placement = entity.get("placement")
        if not isinstance(placement, Mapping):
            raise PredicateStateComputerError(f"corridor {entity_id} lacks placement")
        center = _vector3(placement.get("center_enu_m"))
        extent = _vector3(placement.get("extent_m"))
        rotation = placement.get("rotation_deg")
        yaw_deg = (
            _number(rotation.get("yaw_deg")) if isinstance(rotation, Mapping) else None
        )
        if center is None or extent is None or any(value <= 0.0 for value in extent):
            raise PredicateStateComputerError(
                f"corridor {entity_id} lacks a valid center/extent"
            )
        if yaw_deg is None:
            raise PredicateStateComputerError(
                f"corridor {entity_id} lacks a finite yaw_deg"
            )
        activation_tick = entity.get("activation_tick", 0)
        deactivation_tick = entity.get("deactivation_tick")
        enabled = entity.get("enabled", True)
        if not isinstance(activation_tick, int):
            raise PredicateStateComputerError(
                f"corridor {entity_id} activation_tick must be an integer"
            )
        if deactivation_tick is not None and not isinstance(deactivation_tick, int):
            raise PredicateStateComputerError(
                f"corridor {entity_id} deactivation_tick must be an integer"
            )
        if not isinstance(enabled, bool):
            raise PredicateStateComputerError(
                f"corridor {entity_id} enabled must be boolean"
            )
        cross_section_size = (2.0 * extent[1], 2.0 * extent[2])
        lateral_slots = max(
            1, math.floor(cross_section_size[0] / separation_margin_m) + 1
        )
        vertical_slots = max(
            1, math.floor(cross_section_size[1] / separation_margin_m) + 1
        )
        capacity = lateral_slots * vertical_slots
        parameter_payload = {
            "corridor_id": entity_id,
            "center_enu_m": list(center),
            "extent_m": list(extent),
            "yaw_deg": yaw_deg,
            "cross_section_size_m": list(cross_section_size),
            "minimum_center_separation_m": separation_margin_m,
            "lateral_lane_count": lateral_slots,
            "vertical_lane_count": vertical_slots,
            "capacity": capacity,
            "model": "independent_cross_section_lanes",
        }
        result[entity_id] = {
            **parameter_payload,
            "center_enu_m": center,
            "extent_m": extent,
            "cross_section_size_m": cross_section_size,
            "activation_tick": activation_tick,
            "deactivation_tick": deactivation_tick,
            "enabled": enabled,
            "parameter_digest": digest_object(parameter_payload),
        }
    return result


def _corridor_active_at_tick(corridor: Mapping[str, Any], tick: int) -> bool:
    deactivation_tick = corridor.get("deactivation_tick")
    return (
        bool(corridor["enabled"])
        and tick >= int(corridor["activation_tick"])
        and (deactivation_tick is None or tick < int(deactivation_tick))
    )


def _point_in_oriented_corridor_box(
    point: Sequence[float], corridor: Mapping[str, Any]
) -> bool:
    center = corridor["center_enu_m"]
    extent = corridor["extent_m"]
    yaw_rad = math.radians(float(corridor["yaw_deg"]))
    delta_x = float(point[0]) - float(center[0])
    delta_y = float(point[1]) - float(center[1])
    delta_z = float(point[2]) - float(center[2])
    local_x = math.cos(yaw_rad) * delta_x + math.sin(yaw_rad) * delta_y
    local_y = -math.sin(yaw_rad) * delta_x + math.cos(yaw_rad) * delta_y
    tolerance_m = 1e-9
    return (
        abs(local_x) <= float(extent[0]) + tolerance_m
        and abs(local_y) <= float(extent[1]) + tolerance_m
        and abs(delta_z) <= float(extent[2]) + tolerance_m
    )


def _assigned_landing_zone(
    uav_id: str,
    episode_uav: Mapping[str, Any],
    home_pad_id: str | None,
    home_pad_pose: tuple[float, float, float] | None,
    home_hover: tuple[float, float, float] | None,
) -> tuple[str | None, tuple[float, float, float] | None]:
    """Resolve the physical terminal landing area from the authoritative route.

    UAV pose origins sit above a pad/ground surface.  ``home_hover`` therefore
    supplies the grounded pose-origin reference at the home pad.  An off-pad
    terminal waypoint no higher than that grounded reference plus the governed
    airborne clearance is an assigned emergency landing area.  High terminal
    mission waypoints are not reclassified as landing zones.
    """

    home_reference = home_hover
    if home_reference is None and home_pad_pose is not None:
        home_reference = home_pad_pose
    if home_pad_id is None or home_reference is None:
        return None, None
    route = episode_uav.get("route_waypoints_enu_m")
    terminal = None
    if isinstance(route, Sequence) and not isinstance(route, (str, bytes)):
        for waypoint in reversed(route):
            terminal = _vector3(waypoint)
            if terminal is not None:
                break
    maximum_landing_reference_z = home_reference[2] + float(
        get_governed_parameter_defaults()["airborne_threshold_m"]
    )
    if terminal is None or terminal[2] > maximum_landing_reference_z:
        return home_pad_id, home_reference
    if _distance_xy(terminal, home_reference) <= 1e-9:
        return home_pad_id, terminal
    return f"assigned_landing_zone:{uav_id}", terminal


def _landing_pad_contact_references_by_tick(
    pads: Mapping[str, Mapping[str, Any]],
    positions: Mapping[int, Mapping[str, tuple[float, float, float]]],
    velocities: Mapping[int, Mapping[str, tuple[float, float, float]]],
    activities: Mapping[int, Mapping[str, str]],
    uav_ids: Sequence[str],
    landing_zone_radius_m: float,
) -> dict[int, dict[str, float]]:
    """Contact calibration from observed prefixes, never later landing poses.

    A landing contact can update the local pose-origin reference only from its
    observation tick onward. Missing contact samples retain declared pad geometry.
    """
    contacts: dict[str, float] = {}
    references_by_tick: dict[int, dict[str, float]] = {}
    known_uav_ids = set(uav_ids)
    for tick in sorted(positions):
        for uav_id, position in positions[tick].items():
            if uav_id not in known_uav_ids:
                continue
            if (
                activities.get(tick, {}).get(uav_id)
                not in AIRCRAFT_PAD_CONTACT_ACTIVITIES
            ):
                continue
            velocity = velocities.get(tick, {}).get(uav_id)
            speed = (
                math.dist((0.0, 0.0, 0.0), velocity) if velocity is not None else None
            )
            if speed is None or speed > AIRCRAFT_STATIONARY_SPEED_THRESHOLD_MPS:
                continue
            for pad_id, pad in pads.items():
                pad_position = _entity_position(pad)
                if pad_position is None:
                    continue
                if _distance_xy(position, pad_position) <= landing_zone_radius_m:
                    height = float(position[2])
                    contacts[pad_id] = min(contacts[pad_id], height) if pad_id in contacts else height
        references_by_tick[tick] = {
            pad_id: contacts[pad_id] if pad_id in contacts else float(_entity_position(pad)[2])
            for pad_id, pad in pads.items()
            if _entity_position(pad) is not None
        }
    return references_by_tick


def _local_pad_contact_reference_z(
    position: Sequence[float],
    pads: Mapping[str, Mapping[str, Any]],
    contact_references: Mapping[str, float],
    landing_zone_radius_m: float,
    ground_surface_z: float,
) -> float:
    candidates = [
        (distance, contact_references[pad_id])
        for pad_id, pad in pads.items()
        if pad_id in contact_references
        if (pad_position := _entity_position(pad)) is not None
        if (distance := _distance_xy(position, pad_position)) <= landing_zone_radius_m
    ]
    return min(candidates)[1] if candidates else ground_surface_z


def _local_aircraft_ground_reference_z(
    position: Sequence[float],
    assigned_landing_pose: Sequence[float] | None,
    home_pad_pose: Sequence[float] | None,
    home_contact_reference_z: float | None,
    pads: Mapping[str, Mapping[str, Any]],
    contact_references: Mapping[str, float],
    landing_zone_radius_m: float,
    ground_surface_z: float,
) -> float:
    if (
        assigned_landing_pose is not None
        and _distance_xy(position, assigned_landing_pose) <= landing_zone_radius_m
    ):
        return float(assigned_landing_pose[2])
    if (
        home_pad_pose is not None
        and home_contact_reference_z is not None
        and _distance_xy(position, home_pad_pose) <= landing_zone_radius_m
    ):
        # A pad asset pose is a ground-surface coordinate, while each UAV pose
        # uses its own authoritative contact-height origin.  Shared-pad contact
        # samples from another aircraft must not overwrite this UAV's home
        # contact reference.
        return float(home_contact_reference_z)
    return _local_pad_contact_reference_z(
        position,
        pads,
        contact_references,
        landing_zone_radius_m,
        ground_surface_z,
    )


def _resolve_scenario_json(manifest: Mapping[str, Any], filename: str) -> Path:
    episode_id = manifest.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise PredicateStateComputerError("episode manifest lacks episode_id")
    declared_field = {
        "scene_setup.json": "source_scene_setup_path",
        "event_script.json": "source_event_script_path",
    }.get(filename)
    if declared_field is not None and declared_field in manifest:
        declared = manifest[declared_field]
        if not isinstance(declared, str) or not declared.strip():
            raise PredicateStateComputerError(
                f"episode manifest {declared_field} must be a nonempty path"
            )
        path = (PROJECT_ROOT / declared).resolve()
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise PredicateStateComputerError(
                f"episode manifest {declared_field} escapes the project"
            ) from exc
        if path.name != filename or not path.is_file():
            raise PredicateStateComputerError(
                f"episode manifest {declared_field} is not an existing {filename}: {path}"
            )
        return path
    epi_id = episode_id.split("__seed", 1)[0]
    matches = sorted(
        path.resolve()
        for path in (PROJECT_ROOT / "Dataset" / "scenarios").glob(
            f"**/{epi_id}/{filename}"
        )
        if path.is_file()
    )
    if len(matches) != 1:
        raise PredicateStateComputerError(
            f"scenario authority must contain exactly one {filename} for {epi_id}: {matches}"
        )
    return matches[0]


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise PredicateStateComputerError(f"{path} must contain a JSON object")
    return value


def _index_entities(value: Any, path: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise PredicateStateComputerError(f"{path}: entities must be an array")
    result: dict[str, dict[str, Any]] = {}
    for entity in value:
        if not isinstance(entity, Mapping) or not isinstance(
            entity.get("entity_id"), str
        ):
            raise PredicateStateComputerError(f"{path}: entity lacks entity_id")
        entity_id = str(entity["entity_id"])
        if entity_id in result:
            raise PredicateStateComputerError(
                f"{path}: duplicate entity_id {entity_id}"
            )
        result[entity_id] = dict(entity)
    return result


def _protected_prism_membership(
    position: Sequence[float], region: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Closed protected prism with a separately reported numerical roundoff band.

    The allowance is measured in metres: 32 ulps at the input coordinate scale
    cover subtraction, segment projection/distance and rounded rotated inputs.
    It is a floating-point allowance, not a geometric or physical safety margin.
    ``computed_boundary`` means a zero float64 residual or endpoint comparison;
    it does not prove exact membership from arbitrary-precision coordinates.
    """
    polygon = region["polygon_enu_m"]
    base, top = float(region["base_z_m"]), float(region["top_z_m"])
    scale = max(1.0, *(abs(float(value)) for value in position[:3]),
                *(abs(float(value)) for point in polygon for value in point[:2]),
                abs(base), abs(top))
    tolerance = 32.0 * math.ulp(scale)
    distance_xy = _distance_to_polygon_xy(position, polygon)
    strict_xy = _point_in_polygon_xy(position, polygon)
    if distance_xy == 0.0:
        xy_class = "computed_boundary"
    elif distance_xy <= tolerance:
        xy_class = "roundoff_band"
    else:
        xy_class = "strict_interior" if strict_xy else "outside"
    z = float(position[2])
    z_distance = min(abs(z - base), abs(z - top))
    strict_z = base < z < top
    if z == base or z == top:
        z_class = "computed_boundary"
    elif z_distance <= tolerance:
        z_class = "roundoff_band"
    else:
        z_class = "strict_interior" if strict_z else "outside"
    inside = xy_class != "outside" and z_class != "outside"
    classification = ("outside" if not inside else "roundoff_band"
                      if "roundoff_band" in (xy_class, z_class) else "computed_boundary"
                      if "computed_boundary" in (xy_class, z_class) else "strict_interior")
    return inside, {
        "classification": classification,
        "xy_classification": xy_class,
        "z_classification": z_class,
        "tolerance_m": tolerance,
        "coordinate_scale_m": scale,
        "xy_boundary_distance_m": distance_xy,
        "z_face_distance_m": z_distance,
        "computed_xy_inside_or_boundary": strict_xy or distance_xy == 0.0,
        "computed_z_inside_or_boundary": base <= z <= top,
    }


def _point_in_polygon_xy(
    point: Sequence[float], polygon: Sequence[Sequence[float]]
) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(current[0]), float(current[1])
        x2, y2 = float(previous[0]), float(previous[1])
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _distance_to_polygon_xy(
    point: Sequence[float], polygon: Sequence[Sequence[float]]
) -> float:
    return min(
        _point_segment_distance_xy(point, polygon[index - 1], polygon[index])
        for index in range(len(polygon))
    )


def _distance_to_restricted_prism(
    point: Sequence[float], region: Mapping[str, Any]
) -> float:
    polygon = region["polygon_enu_m"]
    horizontal = _distance_to_polygon_xy(point, polygon)
    inside_xy = _point_in_polygon_xy(point, polygon)
    z = float(point[2])
    base_z = float(region["base_z_m"])
    top_z = float(region["top_z_m"])
    if base_z <= z <= top_z:
        if inside_xy:
            return min(horizontal, z - base_z, top_z - z)
        return horizontal
    vertical = base_z - z if z < base_z else z - top_z
    if inside_xy:
        return vertical
    return math.hypot(horizontal, vertical)


def _signed_distance_to_restricted_prism(
    point: Sequence[float], region: Mapping[str, Any]
) -> float:
    """Return negative distance inside the prism and positive clearance outside."""
    distance = _distance_to_restricted_prism(point, region)
    inside = (
        _point_in_polygon_xy(point, region["polygon_enu_m"])
        and float(region["base_z_m"]) <= float(point[2]) <= float(region["top_z_m"])
    )
    return -distance if inside else distance


def _point_segment_distance_xy(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    denominator = dx * dx + dy * dy
    fraction = (
        0.0 if denominator == 0.0 else ((px - sx) * dx + (py - sy) * dy) / denominator
    )
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(px - (sx + fraction * dx), py - (sy + fraction * dy))


def _entity_position(entity: Mapping[str, Any]) -> tuple[float, float, float] | None:
    return (
        _vector3(entity.get("initial_position_enu_m"))
        or _vector3(entity.get("position_enu_m"))
        or _vector3(
            (entity.get("truth_pose") or {}).get("position_enu_m")
            if isinstance(entity.get("truth_pose"), Mapping)
            else None
        )
    )


def _load_global_uav_pads(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise PredicateStateComputerError(
            f"authoritative global UAV task plan is missing: {path}"
        )
    value = _load_object(path)
    raw_pads = value.get("pads")
    if not isinstance(raw_pads, list) or not raw_pads:
        raise PredicateStateComputerError(f"{path}: pads must be a non-empty array")
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_pads:
        if not isinstance(raw, Mapping):
            raise PredicateStateComputerError(f"{path}: pad row must be an object")
        pad_id = raw.get("pad_id")
        position = _vector3(raw.get("position_enu_m"))
        if (
            not isinstance(pad_id, str)
            or not pad_id
            or pad_id in result
            or position is None
        ):
            raise PredicateStateComputerError(f"{path}: invalid or duplicate UAV pad")
        result[pad_id] = {
            "entity_id": pad_id,
            "entity_category": "facility",
            "entity_kind": "facility.landing_pad",
            "semantic_scope": {
                "scope_type": "facility",
                "scope_subtype": "landing_pad",
                "ontology_class_id": "world:LandingPad",
                "authority_id": "facility_scope_contract.v1",
                "service_capacity": 1,
            },
            "initial_position_enu_m": list(position),
            "source": "donghu_global_uav_flow_task_plan",
        }
    return result


def _load_global_uav_ground_reference_z(path: Path) -> float:
    value = _load_object(path)
    raw = value.get("ground_reference_z_m")
    parsed = _number(raw)
    if parsed is None:
        raise PredicateStateComputerError(
            f"{path}: ground_reference_z_m is required for global UAV actors without pads"
        )
    return float(parsed)


def _entity_category(entity: Mapping[str, Any]) -> str:
    return str(entity.get("entity_category") or entity.get("category") or "").lower()


def _entity_kind(entity: Mapping[str, Any]) -> str:
    return str(entity.get("entity_kind") or entity.get("entity_type") or "").lower()


def _entity_role(entity: Mapping[str, Any]) -> str:
    return str(entity.get("role") or entity.get("uav_corridor_role") or "unknown")


def _distance_xy(first: Sequence[float], second: Sequence[float]) -> float:
    return math.dist(
        (float(first[0]), float(first[1])), (float(second[0]), float(second[1]))
    )


def _vector2(value: Any) -> tuple[float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 2
    ):
        return None
    numbers = (_number(value[0]), _number(value[1]))
    if any(item is None for item in numbers):
        return None
    return float(numbers[0]), float(numbers[1])


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 3
    ):
        return None
    numbers = (_number(value[0]), _number(value[1]), _number(value[2]))
    if any(item is None for item in numbers):
        return None
    return float(numbers[0]), float(numbers[1]), float(numbers[2])


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _missing(values: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    return sorted(field for field in fields if values.get(field) is None)


def _formal_tick_ceiling(tick: int) -> int:
    return ((tick + FORMAL_STEP_TICKS - 1) // FORMAL_STEP_TICKS) * FORMAL_STEP_TICKS


__all__ = [
    "GeometricStateComputer",
    "GeometricStateResult",
    "PlanWindowComputer",
    "PredicateStateComputerError",
]
