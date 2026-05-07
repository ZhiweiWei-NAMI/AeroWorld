#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = SCRIPT_DIR.parent
PLUGIN_ROOT = SCRIPT_DIR.parents[2]
SHARED_SCRIPTS_ROOT = PLUGIN_ROOT / "Scripts"
if str(SHARED_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_ROOT))

from donghu_core.coordinate_service import CoordinateTransformConfig as SharedCoordinateTransformConfig
from donghu_core.discovery import project_root_from, resolve_map_package, resolve_scenario_root, resolve_seed_package_root
from donghu_core.interfaces import ScenarioPackage
from donghu_core.traffic_topology_service import TrafficTopologyService
from donghu_core.weather_service import WeatherService

SCENARIO_DIR_NAME = "donghu_dense_uav_rain_fall"
CANONICAL_SCENARIO_SPEC_PATH = f"Plugins/SumoImporter/Scenarios/{SCENARIO_DIR_NAME}/spec/scenario_spec.json"
SPEC_PATH = SCENARIO_DIR / "spec" / "scenario_spec.json"
DEFAULT_OUTPUT_ROOT = SCENARIO_DIR
DEMO_CAPTURE_PLAN_RELATIVE_PATH = Path("artifacts") / "capture" / "demo_capture_plan.json"
DEMO_EPISODE_RELATIVE_DIR = Path("artifacts") / "episodes"


def _resolve_project_path(value: Any, *, base_dir: Path | None = None) -> Path:
    project_root = project_root_from(Path(__file__))
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (project_root / path).resolve()


def _repo_relative_string(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _safe_tick(sim_time_s: float, tick_hz: int) -> int:
    return int(round(sim_time_s * float(tick_hz)))


def _frame_id(episode_id: str, tick: int) -> str:
    return f"{episode_id}:tick:{tick}"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def _lerp_vec3(a: list[float], b: list[float], alpha: float) -> list[float]:
    return [_lerp(float(a[index]), float(b[index]), alpha) for index in range(3)]


def _polyline_length_m(points: list[list[float]]) -> float:
    return sum(_distance_3d(a, b) for a, b in zip(points, points[1:]))


def _polyline_point_at_fraction(points: list[list[float]], fraction: float) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    if len(points) == 1:
        return list(points[0])
    total_m = _polyline_length_m(points)
    if total_m <= 1e-6:
        return list(points[0])
    target_m = _clamp(float(fraction), 0.0, 1.0) * total_m
    traversed_m = 0.0
    for current, next_point in zip(points, points[1:]):
        segment_m = _distance_3d(current, next_point)
        if segment_m <= 1e-6:
            continue
        if traversed_m + segment_m >= target_m:
            alpha = (target_m - traversed_m) / segment_m
            return _lerp_vec3(current, next_point, _clamp(alpha, 0.0, 1.0))
        traversed_m += segment_m
    return list(points[-1])


def _distance_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _distance_3d(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def _heading_yaw_deg(velocity_enu_mps: list[float], fallback_yaw_deg: float = 0.0) -> float:
    if abs(float(velocity_enu_mps[0])) <= 1e-6 and abs(float(velocity_enu_mps[1])) <= 1e-6:
        return float(fallback_yaw_deg)
    return math.degrees(math.atan2(float(velocity_enu_mps[1]), float(velocity_enu_mps[0])))


def _look_at_rotation_deg(position_enu_m: list[float], target_enu_m: list[float]) -> dict[str, float]:
    delta = [
        float(target_enu_m[0]) - float(position_enu_m[0]),
        float(target_enu_m[1]) - float(position_enu_m[1]),
        float(target_enu_m[2]) - float(position_enu_m[2]),
    ]
    horizontal = max(1e-6, math.hypot(delta[0], delta[1]))
    return {
        "pitch_deg": math.degrees(math.atan2(delta[2], horizontal)),
        "yaw_deg": math.degrees(math.atan2(delta[1], delta[0])),
        "roll_deg": 0.0,
    }


def _truth_pose(position_enu_m: list[float], rotation_deg: dict[str, float], velocity_enu_mps: list[float]) -> dict[str, Any]:
    return {
        "authority_mode": "authoritative_input",
        "authority_owner": "aeroworld_backend",
        "coordinate_contract_id": "coord.external_enu_m.v1",
        "position_enu_m": [round(float(value), 6) for value in position_enu_m],
        "rotation_deg": {
            "pitch_deg": round(float(rotation_deg.get("pitch_deg", 0.0)), 6),
            "roll_deg": round(float(rotation_deg.get("roll_deg", 0.0)), 6),
            "yaw_deg": round(float(rotation_deg.get("yaw_deg", 0.0)), 6),
        },
        "velocity_enu_mps": [round(float(value), 6) for value in velocity_enu_mps],
    }


def _render_presence(*, active: bool, roi_id: str, offstage_reason: str = "") -> dict[str, Any]:
    return {
        "global_roster": True,
        "offstage": not active,
        "offstage_reason": "none" if active else offstage_reason,
        "roi_membership": [roi_id],
        "submission_state": "submit_to_ue" if active else "retain_offstage",
        "visibility_state": "visible" if active else "offstage",
    }


def _weather_payload(*, condition: str) -> dict[str, Any]:
    if condition == "rain":
        return {
            "condition": "rain",
            "rain": 0.7,
            "wetness": 0.8,
            "fog_density": 0.1,
            "dust": 0.0,
            "visibility_m": 2000.0,
            "visibility": 0.6,
            "wind_speed": 6.0,
            "surface_state_a": "wet",
            "surface_friction_scale_a": 0.72,
        }
    return {
        "condition": "clear",
        "rain": 0.0,
        "wetness": 0.0,
        "fog_density": 0.0,
        "dust": 0.0,
        "visibility_m": 6000.0,
        "visibility": 0.95,
        "wind_speed": 2.5,
        "surface_state_a": "dry",
        "surface_friction_scale_a": 1.0,
    }


def _activity_fields(activity_type: str, *, animation_hint: str, posture: str, social_state: str) -> dict[str, Any]:
    return {
        "activity_type": activity_type,
        "animation_hint": animation_hint,
        "posture": posture,
        "social_state": social_state,
    }


def _static_count_by_category(roster_entities: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entity in roster_entities:
        category = str(entity.get("entity_category") or "")
        counts[category] = counts.get(category, 0) + 1
    return counts


def _site_count(roster_entities: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entity in roster_entities:
        site_id = str(entity.get("site_id") or "")
        counts[site_id] = counts.get(site_id, 0) + 1
    return counts


@dataclass(frozen=True)
class CoordinateTransformConfig:
    enabled: bool
    translation_enu_m: tuple[float, float, float]
    axis_mapping: str
    yaw_deg: float
    scale_enu: tuple[float, float, float]

    @classmethod
    def from_root_config(cls, root_config: dict[str, Any]) -> "CoordinateTransformConfig":
        config = dict(root_config.get("coordinate_transform") or {})
        raw_translation = list(config.get("translation_enu_m") or [0.0, 0.0, 0.0])
        raw_scale = list(config.get("scale_enu") or [1.0, 1.0, 1.0])
        return cls(
            enabled=bool(config.get("enabled", False)),
            translation_enu_m=(
                float(raw_translation[0] if len(raw_translation) > 0 else 0.0),
                float(raw_translation[1] if len(raw_translation) > 1 else 0.0),
                float(raw_translation[2] if len(raw_translation) > 2 else 0.0),
            ),
            axis_mapping=str(config.get("axis_mapping", "XY_To_XY") or "XY_To_XY"),
            yaw_deg=float(config.get("yaw_deg", 0.0)),
            scale_enu=(
                float(raw_scale[0] if len(raw_scale) > 0 else 1.0),
                float(raw_scale[1] if len(raw_scale) > 1 else 1.0),
                float(raw_scale[2] if len(raw_scale) > 2 else 1.0),
            ),
        )

    def _apply_axis_mapping(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        mapping = self.axis_mapping.strip() or "XY_To_XY"
        if mapping == "XY_To_XNegY":
            return x, -y, z
        if mapping == "XY_To_YX":
            return y, x, z
        if mapping == "XY_To_YNegX":
            return y, -x, z
        return x, y, z

    def _rotate_xy(self, x: float, y: float) -> tuple[float, float]:
        if not self.enabled or abs(self.yaw_deg) <= 1e-6:
            return x, y
        yaw_rad = math.radians(self.yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        return x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw

    def apply_vector(self, value_enu_m: list[float]) -> list[float]:
        x = float(value_enu_m[0]) * self.scale_enu[0]
        y = float(value_enu_m[1]) * self.scale_enu[1]
        z = float(value_enu_m[2]) * self.scale_enu[2]
        if self.enabled:
            x, y, z = self._apply_axis_mapping(x, y, z)
            x, y = self._rotate_xy(x, y)
        return [x, y, z]

    def apply_position(self, position_enu_m: list[float]) -> list[float]:
        x, y, z = self.apply_vector(position_enu_m)
        if self.enabled:
            x += self.translation_enu_m[0]
            y += self.translation_enu_m[1]
            z += self.translation_enu_m[2]
        return [x, y, z]

    def apply_yaw_deg(self, yaw_deg: float) -> float:
        if not self.enabled:
            return float(yaw_deg)
        yaw_rad = math.radians(float(yaw_deg))
        forward_x, forward_y, _ = self.apply_vector([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
        return math.degrees(math.atan2(forward_y, forward_x))


@dataclass(frozen=True)
class LaneSamplePoint:
    s_m: float
    position_enu_m: tuple[float, float, float]
    yaw_deg: float
    lane_index: int


@dataclass(frozen=True)
class WorldLaneRouteSpec:
    route_id: str
    edge_id: str
    lane_id: str
    vehicle_ids: tuple[str, ...]
    speed_mps: float
    initial_offsets_m: tuple[float, ...]
    lateral_offset_m: float = 0.0
    loop: bool = True


@dataclass(frozen=True)
class UavRuntimeState:
    active: bool
    position_enu_m: list[float]
    velocity_enu_mps: list[float]
    activity_type: str
    animation_hint: str
    mission_id: str
    offstage_reason: str = ""
    fallback_yaw_deg: float = 0.0


@dataclass(frozen=True)
class VehicleLaneSpec:
    lane_id: str
    approach_id: str
    axis: str
    fixed_coord_m: float
    start_coord_m: float
    end_coord_m: float
    speed_mps: float
    min_gap_m: float
    vehicle_ids: tuple[str, ...]
    initial_offsets_m: tuple[float, ...]
    barrier_specs: tuple[tuple[str, float], ...]

    @property
    def direction_sign(self) -> float:
        return 1.0 if self.end_coord_m >= self.start_coord_m else -1.0

    @property
    def route_length_m(self) -> float:
        return abs(float(self.end_coord_m) - float(self.start_coord_m))

    def position_from_progress(self, progress_m: float, altitude_m: float) -> list[float]:
        coord_value = float(self.start_coord_m) + self.direction_sign * float(progress_m)
        if self.axis == "x":
            return [coord_value, float(self.fixed_coord_m), altitude_m]
        return [float(self.fixed_coord_m), coord_value, altitude_m]

    def stopline_progress_m(self, world_coord_m: float) -> float:
        if self.direction_sign > 0.0:
            return float(world_coord_m) - float(self.start_coord_m)
        return float(self.start_coord_m) - float(world_coord_m)


@dataclass(frozen=True)
class PedestrianPlan:
    entity_id: str
    variant_id: str
    path_id: str
    start_tick: int
    end_tick: int
    start_position_enu_m: list[float]
    end_position_enu_m: list[float]
    active_window_end_tick: int
    hold_position_enu_m: list[float]
    hold_activity_type: str
    hold_posture: str
    hold_social_state: str
    hold_animation_hint: str
    umbrella_after_rain: bool = False
    path_waypoints_enu_m: tuple[tuple[float, float, float], ...] = ()

    @property
    def route_points_enu_m(self) -> list[list[float]]:
        raw_points = [list(point) for point in self.path_waypoints_enu_m]
        if not raw_points:
            raw_points = [list(self.start_position_enu_m), list(self.end_position_enu_m)]
        if _distance_3d(raw_points[0], self.start_position_enu_m) > 1e-6:
            raw_points.insert(0, list(self.start_position_enu_m))
        if _distance_3d(raw_points[-1], self.end_position_enu_m) > 1e-6:
            raw_points.append(list(self.end_position_enu_m))
        deduped: list[list[float]] = []
        for point in raw_points:
            if not deduped or _distance_3d(deduped[-1], point) > 1e-6:
                deduped.append(list(point))
        return deduped

    @property
    def route_yaw_deg(self) -> float:
        route = self.route_points_enu_m
        start = route[0]
        end = route[-1]
        if len(route) >= 2:
            start = route[-2]
            end = route[-1]
        delta = [
            float(end[0]) - float(start[0]),
            float(end[1]) - float(start[1]),
            0.0,
        ]
        return _heading_yaw_deg(delta, 0.0)


class DenseUavEpisodeGenerator:
    def __init__(self, spec_path: Path, output_root: Path) -> None:
        self.spec_path = spec_path.resolve()
        self.spec = _load_json(self.spec_path)
        self.output_root = output_root.resolve()
        self.project_root = project_root_from(self.spec_path)
        self.scenario_root = resolve_scenario_root(self.project_root, scenario_dir=SCENARIO_DIR_NAME)
        self.seed_package_root = resolve_seed_package_root(self.project_root)

        self.seed_episode_dir = _resolve_project_path(self.spec["seed_episode_dir"], base_dir=self.spec_path.parent)
        self.seed_capture_config_path = _resolve_project_path(self.spec["seed_capture_config"], base_dir=self.spec_path.parent)
        self.seed_scenario_plan = _load_json(self.seed_episode_dir / "scenario_plan.json")
        self.seed_capture_config = _load_json(self.seed_capture_config_path)
        self.seed_roster_doc = _load_json(self.seed_episode_dir / "global_entity_roster.json")
        self.seed_truth_frames = _load_jsonl(self.seed_episode_dir / "truth_frames.jsonl")
        self.seed_weather_meta = _load_jsonl(self.seed_episode_dir / "weather_meta.jsonl")

        self.episode_id = str(self.spec["output_episode_id"])
        self.scenario_id = str(self.spec["scenario_id"])
        self.scenario_name = str(self.spec["scenario_name"])
        self.tick_hz = int(self.spec["tick_hz"])
        self.dt_s = 1.0 / float(self.tick_hz)
        self.duration_s = float(self.spec["duration_s"])
        self.max_tick = _safe_tick(self.duration_s, self.tick_hz)
        self.site_id = str(self.spec["site_id"])
        self.roi_id = str(self.spec["roi_id"])
        self.site_ground_elevation_m = float(self.spec.get("site_ground_elevation_m", 0.0))
        self.uav_cruise_height_agl_m = float(self.spec.get("uav_cruise_height_agl_m", 50.0))
        self.uav_transient_height_agl_m = float(self.spec.get("uav_transient_height_agl_m", self.uav_cruise_height_agl_m + 2.0))
        self.uav_inspection_height_agl_m = float(self.spec.get("uav_inspection_height_agl_m", 14.0))
        self.camera_high_height_agl_m = float(self.spec.get("camera_high_height_agl_m", 80.0))
        self.camera_low_height_agl_m = float(self.spec.get("camera_low_height_agl_m", 8.0))
        self.center_enu_m = [float(value) for value in self.spec["intersection_center_enu_m"]]
        self.fall_target_enu_m = [float(value) for value in self.spec["fall_target_enu_m"]]
        self.fall_facing_yaw_deg = float(self.spec.get("fall_facing_yaw_deg", 0.0))
        self.phase_windows = dict(self.spec["phase_windows_s"])

        self.rain_start_tick = _safe_tick(float(self.phase_windows["rain"][0]), self.tick_hz)
        self.divert_start_tick = _safe_tick(float(self.phase_windows["divert_start"]), self.tick_hz)
        self.divert_end_tick = _safe_tick(float(self.phase_windows["divert_end"]), self.tick_hz)
        self.fall_tick = _safe_tick(float(self.phase_windows["fall_trigger"]), self.tick_hz)
        self.inspection_tick = _safe_tick(float(self.phase_windows["inspection_start"]), self.tick_hz)

        self.seed_frames_by_tick = {int(frame["tick"]): frame for frame in self.seed_truth_frames}
        self.frame_template = copy.deepcopy(self.seed_frames_by_tick[min(self.seed_frames_by_tick)])
        self.coordinate_transform = SharedCoordinateTransformConfig.from_root_config(self.seed_capture_config)
        self.map_id = str(self.frame_template.get("map_id") or self.seed_capture_config.get("map_id") or "donghu_road_topo")
        self.map_package = resolve_map_package(self.project_root, map_id=self.map_id)
        self.traffic_topology_service = TrafficTopologyService.from_bundle_dir(self.map_package.traffic_bundle_dir)
        self.lane_center_samples_path = self.traffic_topology_service.bundle_paths.lane_center_samples_path
        self.road_geojson_path = self.map_package.source_geojson["road"]
        self.weather_service = WeatherService.from_profiles_path(self.map_package.weather_profiles_path)
        self.lane_samples_by_route = self.traffic_topology_service.load_lane_samples_by_route()
        self.road_props_by_edge = self._load_road_properties()

        self.base_roster_entities = list(self.seed_roster_doc.get("entities", []))
        self.runtime_support_entities: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []
        self.dynamic_labels: list[dict[str, Any]] = []

        self.local_uav_id = "drone_demo_a_021"
        self.inspection_uav_id = "drone_demo_a_023"
        self.transient_uav_id = "drone_demo_a_024"
        self.pre_rain_patrol_uav_ids = [f"drone_demo_a_{index:03d}" for index in range(1, 25)]
        self.departing_uav_ids = [
            entity_id
            for entity_id in self.pre_rain_patrol_uav_ids
            if entity_id not in {self.local_uav_id}
        ]

        self.vehicle_ids = [f"vehicle_a_{index:03d}" for index in range(1, 25)]
        self.pedestrian_ids = [f"pedestrian_a_{index:03d}" for index in range(1, 13)]
        self.vehicle_altitude_m = self.site_ground_elevation_m
        self.pedestrian_altitude_m = self.site_ground_elevation_m

        self.ped_variant_by_id = self._build_ped_variant_index()
        self.vehicle_routes = self._build_vehicle_routes()
        self.pedestrian_plans = self._build_pedestrian_plans()
        self.lower_crosswalk_ped_ids = {
            "pedestrian_a_001",
            "pedestrian_a_002",
            "pedestrian_a_003",
            "pedestrian_a_004",
            "pedestrian_a_005",
            "pedestrian_a_006",
        }
        self.upper_crosswalk_ped_ids = {
            "pedestrian_a_007",
            "pedestrian_a_008",
            "pedestrian_a_009",
            "pedestrian_a_010",
        }
        self._fall_heading_yaw_deg_cache: float | None = None
        self.center_world_enu_m = self._world_position(self.center_enu_m)
        self.fall_target_world_enu_m = self._world_position(list(self.fall_target_enu_m))
        self.pedestrian_state_cache = self._build_pedestrian_state_cache()
        self.vehicle_state_cache = self._build_vehicle_state_cache()

        self.demo_roster_entities = self._build_demo_roster()
        self.category_counts = _static_count_by_category(self.demo_roster_entities)
        self.site_counts = _site_count(self.demo_roster_entities)

    def _absolute_altitude_m(self, height_agl_m: float) -> float:
        return round(self.site_ground_elevation_m + float(height_agl_m), 3)

    def _world_position(self, local_position_enu_m: list[float]) -> list[float]:
        return self.coordinate_transform.apply_position(list(local_position_enu_m))

    def _world_vector(self, local_vector_enu_m: list[float]) -> list[float]:
        return self.coordinate_transform.apply_vector(list(local_vector_enu_m))

    def _world_yaw_deg(self, local_yaw_deg: float) -> float:
        return self.coordinate_transform.apply_yaw_deg(float(local_yaw_deg))

    def _load_lane_samples(self) -> dict[tuple[str, str], list[LaneSamplePoint]]:
        samples_by_route: dict[tuple[str, str], list[LaneSamplePoint]] = {}
        with self.lane_center_samples_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                edge_id = str(row.get("edge_id") or "").strip()
                lane_id = str(row.get("lane_id") or "").strip()
                if not edge_id or not lane_id:
                    continue
                samples_by_route.setdefault((edge_id, lane_id), []).append(
                    LaneSamplePoint(
                        s_m=float(row.get("s_m") or 0.0),
                        position_enu_m=(
                            float(row.get("x_m") or 0.0),
                            float(row.get("y_m") or 0.0),
                            float(row.get("z_m") or 0.0),
                        ),
                        yaw_deg=float(row.get("yaw_deg") or 0.0),
                        lane_index=int(row.get("lane_index") or 0),
                    )
                )
        for key in list(samples_by_route):
            samples_by_route[key] = sorted(samples_by_route[key], key=lambda item: item.s_m)
        return samples_by_route

    def _load_road_properties(self) -> dict[str, dict[str, float]]:
        obj = json.loads(self.road_geojson_path.read_text(encoding="utf-8-sig"))
        props_by_edge: dict[str, dict[str, float]] = {}
        for feature in obj.get("features", []):
            props = dict(feature.get("properties") or {})
            edge_id = f"cg_edge_{int(props.get('id'))}"
            props_by_edge[edge_id] = {
                "lanes": float(props.get("lanes") or 1.0),
                "width_m": float(props.get("width") or 3.0),
            }
        return props_by_edge

    def _build_vehicle_routes(self) -> list[WorldLaneRouteSpec]:
        return [
            WorldLaneRouteSpec(
                route_id="route.vehicle.world.lower_west_east",
                edge_id="cg_edge_253",
                lane_id="cg_edge_253_0",
                vehicle_ids=("vehicle_a_001", "vehicle_a_002", "vehicle_a_003", "vehicle_a_004", "vehicle_a_005"),
                speed_mps=1.4,
                initial_offsets_m=(0.0, 14.0, 28.0, 42.0, 56.0),
                lateral_offset_m=-5.25,
            ),
            WorldLaneRouteSpec(
                route_id="route.vehicle.world.mid_west_east",
                edge_id="cg_edge_664",
                lane_id="cg_edge_664_0",
                vehicle_ids=("vehicle_a_006", "vehicle_a_007", "vehicle_a_008", "vehicle_a_009", "vehicle_a_010"),
                speed_mps=1.2,
                initial_offsets_m=(0.0, 13.0, 26.0, 39.0, 52.0),
                lateral_offset_m=-1.75,
            ),
            WorldLaneRouteSpec(
                route_id="route.vehicle.world.upper_west_east",
                edge_id="cg_edge_438",
                lane_id="cg_edge_438_0",
                vehicle_ids=("vehicle_a_011", "vehicle_a_012", "vehicle_a_013", "vehicle_a_014"),
                speed_mps=0.9,
                initial_offsets_m=(0.0, 12.0, 24.0, 36.0),
                lateral_offset_m=1.75,
            ),
            WorldLaneRouteSpec(
                route_id="route.vehicle.world.west_south_north",
                edge_id="cg_edge_583",
                lane_id="cg_edge_583_0",
                vehicle_ids=("vehicle_a_015", "vehicle_a_016", "vehicle_a_017", "vehicle_a_018", "vehicle_a_019"),
                speed_mps=1.1,
                initial_offsets_m=(0.0, 14.0, 28.0, 42.0, 56.0),
                lateral_offset_m=-3.5,
            ),
            WorldLaneRouteSpec(
                route_id="route.vehicle.world.east_south_north",
                edge_id="cg_edge_185",
                lane_id="cg_edge_185_0",
                vehicle_ids=("vehicle_a_020", "vehicle_a_021", "vehicle_a_022", "vehicle_a_023", "vehicle_a_024"),
                speed_mps=1.0,
                initial_offsets_m=(12.0, 26.0, 40.0, 54.0, 68.0),
                lateral_offset_m=3.5,
            ),
        ]

    def _route_point_at_distance(self, route: WorldLaneRouteSpec, distance_m: float) -> tuple[list[float], float]:
        samples = self.lane_samples_by_route[(route.edge_id, route.lane_id)]
        road_props = dict(self.road_props_by_edge.get(route.edge_id) or {})
        lane_count = max(1, int(round(float(road_props.get("lanes") or 1.0))))
        effective_lateral_offset_m = float(route.lateral_offset_m) if lane_count > 1 else 0.0
        route_length_m = float(samples[-1].s_m)
        if route.loop and route_length_m > 1e-6:
            resolved_distance_m = float(distance_m) % route_length_m
        else:
            resolved_distance_m = _clamp(float(distance_m), 0.0, route_length_m)
        if resolved_distance_m <= samples[0].s_m:
            position = list(samples[0].position_enu_m)
            yaw_deg = float(samples[0].yaw_deg)
            if abs(effective_lateral_offset_m) > 1e-6:
                yaw_rad = math.radians(yaw_deg)
                left_normal_xy = (-math.sin(yaw_rad), math.cos(yaw_rad))
                position[0] += left_normal_xy[0] * effective_lateral_offset_m
                position[1] += left_normal_xy[1] * effective_lateral_offset_m
            return position, yaw_deg
        for current_sample, next_sample in zip(samples, samples[1:]):
            if current_sample.s_m <= resolved_distance_m <= next_sample.s_m:
                span_m = max(1e-6, next_sample.s_m - current_sample.s_m)
                alpha = (resolved_distance_m - current_sample.s_m) / span_m
                position = _lerp_vec3(list(current_sample.position_enu_m), list(next_sample.position_enu_m), alpha)
                velocity_hint = [
                    float(next_sample.position_enu_m[index]) - float(current_sample.position_enu_m[index])
                    for index in range(3)
                ]
                yaw_deg = _heading_yaw_deg(velocity_hint, current_sample.yaw_deg)
                if abs(effective_lateral_offset_m) > 1e-6:
                    yaw_rad = math.radians(yaw_deg)
                    left_normal_xy = (-math.sin(yaw_rad), math.cos(yaw_rad))
                    position[0] += left_normal_xy[0] * effective_lateral_offset_m
                    position[1] += left_normal_xy[1] * effective_lateral_offset_m
                return position, yaw_deg
        position = list(samples[-1].position_enu_m)
        yaw_deg = float(samples[-1].yaw_deg)
        if abs(effective_lateral_offset_m) > 1e-6:
            yaw_rad = math.radians(yaw_deg)
            left_normal_xy = (-math.sin(yaw_rad), math.cos(yaw_rad))
            position[0] += left_normal_xy[0] * effective_lateral_offset_m
            position[1] += left_normal_xy[1] * effective_lateral_offset_m
        return position, yaw_deg

    def _build_ped_variant_index(self) -> dict[str, str]:
        return {
            "pedestrian_a_001": "adult_female_commuter",
            "pedestrian_a_002": "adult_male_commuter",
            "pedestrian_a_003": "adult_female_commuter",
            "pedestrian_a_004": "child_crossing",
            "pedestrian_a_005": "elder_observer",
            "pedestrian_a_006": "adult_male_commuter",
            "pedestrian_a_007": "adult_female_commuter",
            "pedestrian_a_008": "child_crossing",
            "pedestrian_a_009": "adult_female_commuter",
            "pedestrian_a_010": "adult_male_commuter",
            "pedestrian_a_011": "elder_observer",
            "pedestrian_a_012": "adult_female_commuter",
        }

    def _build_vehicle_lanes(self) -> list[VehicleLaneSpec]:
        return [
            VehicleLaneSpec(
                lane_id="lane.vehicle.a.eastbound.lower",
                approach_id="eastbound_lower",
                axis="x",
                fixed_coord_m=12.4,
                start_coord_m=-140.0,
                end_coord_m=240.0,
                speed_mps=2.0,
                min_gap_m=14.0,
                vehicle_ids=("vehicle_a_001", "vehicle_a_002", "vehicle_a_003", "vehicle_a_004"),
                initial_offsets_m=(170.0, 130.0, 90.0, 50.0),
                barrier_specs=(),
            ),
            VehicleLaneSpec(
                lane_id="lane.vehicle.a.eastbound.middle",
                approach_id="eastbound_middle",
                axis="x",
                fixed_coord_m=16.4,
                start_coord_m=-140.0,
                end_coord_m=240.0,
                speed_mps=1.9,
                min_gap_m=14.0,
                vehicle_ids=("vehicle_a_005", "vehicle_a_006", "vehicle_a_007", "vehicle_a_008"),
                initial_offsets_m=(180.0, 140.0, 100.0, 60.0),
                barrier_specs=(),
            ),
            VehicleLaneSpec(
                lane_id="lane.vehicle.a.westbound.lower",
                approach_id="westbound_lower",
                axis="x",
                fixed_coord_m=22.4,
                start_coord_m=240.0,
                end_coord_m=-140.0,
                speed_mps=1.95,
                min_gap_m=14.0,
                vehicle_ids=("vehicle_a_009", "vehicle_a_010", "vehicle_a_011", "vehicle_a_012"),
                initial_offsets_m=(170.0, 130.0, 90.0, 50.0),
                barrier_specs=(),
            ),
            VehicleLaneSpec(
                lane_id="lane.vehicle.a.westbound.upper",
                approach_id="westbound_upper",
                axis="x",
                fixed_coord_m=34.8,
                start_coord_m=240.0,
                end_coord_m=-140.0,
                speed_mps=1.85,
                min_gap_m=14.0,
                vehicle_ids=("vehicle_a_013", "vehicle_a_014", "vehicle_a_015", "vehicle_a_016"),
                initial_offsets_m=(180.0, 140.0, 100.0, 60.0),
                barrier_specs=(),
            ),
            VehicleLaneSpec(
                lane_id="lane.vehicle.a.eastbound.outer",
                approach_id="eastbound_outer",
                axis="x",
                fixed_coord_m=8.4,
                start_coord_m=-140.0,
                end_coord_m=240.0,
                speed_mps=2.05,
                min_gap_m=14.0,
                vehicle_ids=("vehicle_a_017", "vehicle_a_018", "vehicle_a_019", "vehicle_a_020"),
                initial_offsets_m=(175.0, 135.0, 95.0, 55.0),
                barrier_specs=(),
            ),
            VehicleLaneSpec(
                lane_id="lane.vehicle.a.westbound.outer",
                approach_id="westbound_outer",
                axis="x",
                fixed_coord_m=38.8,
                start_coord_m=240.0,
                end_coord_m=-140.0,
                speed_mps=1.8,
                min_gap_m=14.0,
                vehicle_ids=("vehicle_a_021", "vehicle_a_022", "vehicle_a_023", "vehicle_a_024"),
                initial_offsets_m=(175.0, 135.0, 95.0, 55.0),
                barrier_specs=(),
            ),
        ]

    def _build_pedestrian_plans(self) -> dict[str, PedestrianPlan]:
        z = self.pedestrian_altitude_m
        return {
            "pedestrian_a_001": PedestrianPlan(
                entity_id="pedestrian_a_001",
                variant_id=self.ped_variant_by_id["pedestrian_a_001"],
                path_id="route.pedestrian.a.lower_01",
                start_tick=_safe_tick(5.0, self.tick_hz),
                end_tick=_safe_tick(28.0, self.tick_hz),
                start_position_enu_m=[30.0, 18.2, z],
                end_position_enu_m=[60.5, 18.2, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[62.0, 18.2, z],
                hold_activity_type="observing",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_observe",
                umbrella_after_rain=True,
            ),
            "pedestrian_a_002": PedestrianPlan(
                entity_id="pedestrian_a_002",
                variant_id=self.ped_variant_by_id["pedestrian_a_002"],
                path_id="route.pedestrian.a.lower_02",
                start_tick=_safe_tick(11.0, self.tick_hz),
                end_tick=_safe_tick(33.0, self.tick_hz),
                start_position_enu_m=[28.0, 19.1, z],
                end_position_enu_m=[58.5, 19.1, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[60.0, 19.1, z],
                hold_activity_type="waiting",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_idle",
                umbrella_after_rain=False,
            ),
            "pedestrian_a_003": PedestrianPlan(
                entity_id="pedestrian_a_003",
                variant_id=self.ped_variant_by_id["pedestrian_a_003"],
                path_id="route.pedestrian.a.lower_03",
                start_tick=_safe_tick(17.0, self.tick_hz),
                end_tick=_safe_tick(39.0, self.tick_hz),
                start_position_enu_m=[30.0, 20.0, z],
                end_position_enu_m=[60.0, 20.0, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[61.5, 20.0, z],
                hold_activity_type="observing",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_observe",
                umbrella_after_rain=True,
            ),
            "pedestrian_a_004": PedestrianPlan(
                entity_id="pedestrian_a_004",
                variant_id=self.ped_variant_by_id["pedestrian_a_004"],
                path_id="route.pedestrian.a.lower_04",
                start_tick=_safe_tick(24.0, self.tick_hz),
                end_tick=_safe_tick(43.0, self.tick_hz),
                start_position_enu_m=[27.5, 20.9, z],
                end_position_enu_m=[55.5, 20.9, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[56.8, 20.9, z],
                hold_activity_type="waiting",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_idle",
                umbrella_after_rain=False,
            ),
            "pedestrian_a_005": PedestrianPlan(
                entity_id="pedestrian_a_005",
                variant_id=self.ped_variant_by_id["pedestrian_a_005"],
                path_id="route.pedestrian.a.lower_incident",
                start_tick=self.rain_start_tick,
                end_tick=self.fall_tick,
                start_position_enu_m=[28.0, 24.8, z],
                end_position_enu_m=list(self.fall_target_enu_m),
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=list(self.fall_target_enu_m),
                hold_activity_type="medical_incident",
                hold_posture="fallen",
                hold_social_state="distress",
                hold_animation_hint="pedestrian_fall",
                umbrella_after_rain=False,
            ),
            "pedestrian_a_006": PedestrianPlan(
                entity_id="pedestrian_a_006",
                variant_id=self.ped_variant_by_id["pedestrian_a_006"],
                path_id="route.pedestrian.a.lower_05",
                start_tick=_safe_tick(9.0, self.tick_hz),
                end_tick=_safe_tick(31.0, self.tick_hz),
                start_position_enu_m=[29.0, 21.8, z],
                end_position_enu_m=[58.0, 21.8, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[59.0, 21.8, z],
                hold_activity_type="observing",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_observe",
                umbrella_after_rain=False,
            ),
            "pedestrian_a_007": PedestrianPlan(
                entity_id="pedestrian_a_007",
                variant_id=self.ped_variant_by_id["pedestrian_a_007"],
                path_id="route.pedestrian.a.upper_01",
                start_tick=_safe_tick(8.0, self.tick_hz),
                end_tick=_safe_tick(27.0, self.tick_hz),
                start_position_enu_m=[57.0, 27.4, z],
                end_position_enu_m=[31.0, 27.4, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[29.5, 27.4, z],
                hold_activity_type="observing",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_observe",
                umbrella_after_rain=True,
            ),
            "pedestrian_a_008": PedestrianPlan(
                entity_id="pedestrian_a_008",
                variant_id=self.ped_variant_by_id["pedestrian_a_008"],
                path_id="route.pedestrian.a.upper_02",
                start_tick=_safe_tick(16.0, self.tick_hz),
                end_tick=_safe_tick(34.0, self.tick_hz),
                start_position_enu_m=[55.5, 28.9, z],
                end_position_enu_m=[32.0, 28.9, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[30.0, 28.9, z],
                hold_activity_type="waiting",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_idle",
                umbrella_after_rain=False,
            ),
            "pedestrian_a_009": PedestrianPlan(
                entity_id="pedestrian_a_009",
                variant_id=self.ped_variant_by_id["pedestrian_a_009"],
                path_id="route.pedestrian.a.upper_03",
                start_tick=_safe_tick(23.0, self.tick_hz),
                end_tick=_safe_tick(42.0, self.tick_hz),
                start_position_enu_m=[56.5, 30.4, z],
                end_position_enu_m=[30.0, 30.4, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[28.5, 30.4, z],
                hold_activity_type="observing",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_observe",
                umbrella_after_rain=True,
            ),
            "pedestrian_a_010": PedestrianPlan(
                entity_id="pedestrian_a_010",
                variant_id=self.ped_variant_by_id["pedestrian_a_010"],
                path_id="route.pedestrian.a.upper_04",
                start_tick=_safe_tick(31.0, self.tick_hz),
                end_tick=_safe_tick(48.0, self.tick_hz),
                start_position_enu_m=[54.5, 31.9, z],
                end_position_enu_m=[31.5, 31.9, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[29.8, 31.9, z],
                hold_activity_type="waiting",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_idle",
                umbrella_after_rain=False,
            ),
            "pedestrian_a_011": PedestrianPlan(
                entity_id="pedestrian_a_011",
                variant_id=self.ped_variant_by_id["pedestrian_a_011"],
                path_id="route.pedestrian.a.sidewalk_north",
                start_tick=_safe_tick(0.0, self.tick_hz),
                end_tick=_safe_tick(22.0, self.tick_hz),
                start_position_enu_m=[31.0, 31.8, z],
                end_position_enu_m=[43.0, 31.8, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[44.0, 31.8, z],
                hold_activity_type="observing",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_observe",
                umbrella_after_rain=True,
            ),
            "pedestrian_a_012": PedestrianPlan(
                entity_id="pedestrian_a_012",
                variant_id=self.ped_variant_by_id["pedestrian_a_012"],
                path_id="route.pedestrian.a.sidewalk_south",
                start_tick=_safe_tick(0.0, self.tick_hz),
                end_tick=_safe_tick(22.0, self.tick_hz),
                start_position_enu_m=[60.0, 6.8, z],
                end_position_enu_m=[48.0, 6.8, z],
                active_window_end_tick=self.max_tick,
                hold_position_enu_m=[47.0, 6.8, z],
                hold_activity_type="observing",
                hold_posture="standing",
                hold_social_state="calm",
                hold_animation_hint="pedestrian_observe",
                umbrella_after_rain=True,
            ),
        }

    def _build_demo_roster(self) -> list[dict[str, Any]]:
        roster_index = {
            str(entity.get("entity_id") or ""): copy.deepcopy(entity)
            for entity in self.base_roster_entities
            if str(entity.get("site_id") or "") == self.site_id
        }
        entities: list[dict[str, Any]] = []

        for entity_id in self.vehicle_ids:
            base = copy.deepcopy(roster_index.get(entity_id) or {})
            if not base:
                base = {
                    "entity_id": entity_id,
                    "entity_category": "vehicle",
                    "entity_kind": "vehicle.car",
                    "proxy_template_id": "vehicle.sedan",
                    "site_id": self.site_id,
                    "tags": ["site_a", "vehicle"],
                }
            base["entity_id"] = entity_id
            base["entity_category"] = "vehicle"
            base["entity_kind"] = str(base.get("entity_kind") or base.get("entity_type") or "vehicle.car")
            base["proxy_template_id"] = str(base.get("proxy_template_id") or "vehicle.sedan")
            base["site_id"] = self.site_id
            base["tags"] = ["site_a", "vehicle"]
            if entity_id in self.vehicle_state_cache.get(0, {}):
                base["position_enu_m"] = list(self.vehicle_state_cache[0][entity_id]["position_enu_m"])
                base["yaw_deg"] = float(self.vehicle_state_cache[0][entity_id]["yaw_deg"])
            entities.append(base)

        for entity_id in self.pedestrian_ids:
            base = copy.deepcopy(roster_index.get(entity_id) or {})
            if not base:
                base = {
                    "entity_id": entity_id,
                    "entity_category": "pedestrian",
                    "entity_kind": "pedestrian.person",
                    "proxy_template_id": "human.walker",
                    "site_id": self.site_id,
                    "tags": ["site_a", "pedestrian"],
                }
            base["entity_id"] = entity_id
            base["entity_category"] = "pedestrian"
            base["entity_kind"] = str(base.get("entity_kind") or base.get("entity_type") or "pedestrian.person")
            base["proxy_template_id"] = "human.walker"
            base["site_id"] = self.site_id
            base["tags"] = ["site_a", "pedestrian"]
            base["variant_id"] = self.ped_variant_by_id[entity_id]
            if entity_id in self.pedestrian_state_cache.get(0, {}):
                ped_state = self.pedestrian_state_cache[0][entity_id]
                base["position_enu_m"] = self._world_position(list(ped_state["position_enu_m"]))
                base["yaw_deg"] = self._world_yaw_deg(float(ped_state["yaw_deg"]))
            entities.append(base)

        for index in range(1, 25):
            entity_id = f"drone_demo_a_{index:03d}"
            entities.append(
                {
                    "entity_category": "uav",
                    "entity_id": entity_id,
                    "entity_type": "uav.drone",
                    "facets": {},
                    "policies": [],
                    "position_enu_m": self._world_position(list(self._uav_state_at_tick(entity_id, 0).position_enu_m)),
                    "procedures": [],
                    "proxy_template_id": "drone.quadrotor",
                    "site_id": self.site_id,
                    "tags": ["site_a", "uav", "demo_dense"],
                    "yaw_deg": self._world_yaw_deg(_heading_yaw_deg(self._uav_state_at_tick(entity_id, 0).velocity_enu_mps, self._uav_state_at_tick(entity_id, 0).fallback_yaw_deg)),
                }
            )
        return entities

    def _active_weather(self, tick: int) -> dict[str, Any]:
        condition = "rain" if int(tick) >= self.rain_start_tick else "clear"
        return self.weather_service.payload_for_condition(condition)

    def _pedestrian_linear_state(self, plan: PedestrianPlan, tick: int) -> dict[str, Any]:
        tick_int = int(tick)
        route_yaw_deg = plan.route_yaw_deg
        if tick_int < plan.start_tick:
            return {
                "position_enu_m": list(plan.start_position_enu_m),
                "velocity_enu_mps": [0.0, 0.0, 0.0],
                "yaw_deg": route_yaw_deg,
                "activity_type": "waiting",
                "animation_hint": "pedestrian_idle",
                "posture": "standing",
                "social_state": "calm",
                "path_id": plan.path_id,
                "variant_id": plan.variant_id,
            }
        if tick_int <= plan.end_tick:
            alpha = float(tick_int - plan.start_tick) / max(1.0, float(plan.end_tick - plan.start_tick))
            next_alpha = float(min(plan.end_tick, tick_int + 1) - plan.start_tick) / max(1.0, float(plan.end_tick - plan.start_tick))
            route = plan.route_points_enu_m
            position = _polyline_point_at_fraction(route, alpha)
            next_position = _polyline_point_at_fraction(route, next_alpha)
            velocity = [
                (float(next_position[index]) - float(position[index])) / self.dt_s
                for index in range(3)
            ]
            return {
                "position_enu_m": position,
                "velocity_enu_mps": velocity,
                "yaw_deg": _heading_yaw_deg(velocity, route_yaw_deg),
                "activity_type": "walking",
                "animation_hint": "pedestrian_walk",
                "posture": "standing",
                "social_state": "calm",
                "path_id": plan.path_id,
                "variant_id": plan.variant_id,
            }
        return {
            "position_enu_m": list(plan.hold_position_enu_m),
            "velocity_enu_mps": [0.0, 0.0, 0.0],
            "yaw_deg": route_yaw_deg,
            "activity_type": plan.hold_activity_type,
            "animation_hint": plan.hold_animation_hint,
            "posture": plan.hold_posture,
            "social_state": plan.hold_social_state,
            "path_id": plan.path_id,
            "variant_id": plan.variant_id,
        }

    def _fall_pedestrian_yaw_deg(self) -> float:
        if self._fall_heading_yaw_deg_cache is not None:
            return self._fall_heading_yaw_deg_cache
        self._fall_heading_yaw_deg_cache = self.fall_facing_yaw_deg
        return self._fall_heading_yaw_deg_cache

    def _pedestrian_state_at_tick(self, entity_id: str, tick: int) -> dict[str, Any]:
        plan = self.pedestrian_plans[entity_id]
        state = self._pedestrian_linear_state(plan, tick)
        if entity_id == "pedestrian_a_005" and int(tick) >= self.fall_tick:
            state["position_enu_m"] = list(self.fall_target_enu_m)
            state["velocity_enu_mps"] = [0.0, 0.0, 0.0]
            state["yaw_deg"] = self._fall_pedestrian_yaw_deg()
            state["activity_type"] = "medical_incident"
            state["animation_hint"] = "pedestrian_fall"
            state["posture"] = "fallen"
            state["social_state"] = "distress"
        return state

    def _build_pedestrian_state_cache(self) -> dict[int, dict[str, dict[str, Any]]]:
        cache: dict[int, dict[str, dict[str, Any]]] = {}
        for tick in range(0, self.max_tick + 1):
            cache[tick] = {
                entity_id: self._pedestrian_state_at_tick(entity_id, tick)
                for entity_id in self.pedestrian_ids
            }
        return cache

    def _crosswalk_barrier_state(self, tick: int) -> dict[str, bool]:
        lower_active = False
        upper_active = False
        current_states = self.pedestrian_state_cache[int(tick)]
        for entity_id in self.lower_crosswalk_ped_ids:
            state = current_states[entity_id]
            position = state["position_enu_m"]
            activity_type = str(state["activity_type"])
            if activity_type in {"walking", "medical_incident"} and 32.0 <= float(position[0]) <= 58.5:
                lower_active = True
        for entity_id in self.upper_crosswalk_ped_ids:
            state = current_states[entity_id]
            position = state["position_enu_m"]
            if str(state["activity_type"]) == "walking" and 31.0 <= float(position[0]) <= 57.0:
                upper_active = True
        return {
            "lower_crosswalk": lower_active,
            "upper_crosswalk": upper_active,
        }

    def _build_vehicle_state_cache(self) -> dict[int, dict[str, dict[str, Any]]]:
        cache: dict[int, dict[str, dict[str, Any]]] = {}
        for tick in range(0, self.max_tick + 1):
            tick_cache: dict[str, dict[str, Any]] = {}
            for route in self.vehicle_routes:
                for vehicle_id, initial_offset in zip(route.vehicle_ids, route.initial_offsets_m):
                    current_distance_m = float(initial_offset) + float(route.speed_mps) * float(tick) * self.dt_s
                    next_distance_m = float(initial_offset) + float(route.speed_mps) * float(min(self.max_tick, tick + 1)) * self.dt_s
                    position, yaw_deg = self._route_point_at_distance(route, current_distance_m)
                    next_position, _ = self._route_point_at_distance(route, next_distance_m)
                    velocity_enu_mps = [
                        (float(next_position[index]) - float(position[index])) / self.dt_s
                        for index in range(3)
                    ]
                    tick_cache[vehicle_id] = {
                        "position_enu_m": position,
                        "velocity_enu_mps": velocity_enu_mps,
                        "yaw_deg": _heading_yaw_deg(velocity_enu_mps, yaw_deg),
                        "lane_id": route.lane_id,
                        "approach_id": route.route_id,
                        "activity_type": "driving",
                        "lights_on": tick >= self.rain_start_tick,
                    }
            cache[tick] = tick_cache
        return cache

    def _patrol_uav_route(self, entity_id: str) -> dict[str, Any]:
        numeric_id = int(entity_id.rsplit("_", 1)[-1])
        if entity_id == self.local_uav_id:
            return {
                "kind": "local_orbit",
                "speed_mps": 0.0,
                "altitude_m": self._absolute_altitude_m(self.uav_cruise_height_agl_m - 1.0),
            }

        eastbound_ids = [f"drone_demo_a_{index:03d}" for index in range(1, 7)]
        westbound_ids = [f"drone_demo_a_{index:03d}" for index in range(7, 13)]
        northbound_ids = [f"drone_demo_a_{index:03d}" for index in range(13, 19)]
        southbound_ids = ["drone_demo_a_019", "drone_demo_a_020", "drone_demo_a_022", "drone_demo_a_023", "drone_demo_a_024"]

        if entity_id in eastbound_ids:
            idx = eastbound_ids.index(entity_id)
            return {
                "kind": "linear",
                "start": [-180.0, 8.0 + idx * 4.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m + (idx % 3) * 2.0)],
                "end": [280.0, 8.0 + idx * 4.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m + (idx % 3) * 2.0)],
                "initial_offset_m": 260.0 - idx * 40.0,
                "speed_mps": 3.2,
            }
        if entity_id in westbound_ids:
            idx = westbound_ids.index(entity_id)
            return {
                "kind": "linear",
                "start": [280.0, 10.0 + idx * 4.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m + 1.0 + (idx % 2) * 2.0)],
                "end": [-180.0, 10.0 + idx * 4.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m + 1.0 + (idx % 2) * 2.0)],
                "initial_offset_m": 260.0 - idx * 40.0,
                "speed_mps": 3.25,
            }
        if entity_id in northbound_ids:
            idx = northbound_ids.index(entity_id)
            return {
                "kind": "linear",
                "start": [26.0 + idx * 7.0, -180.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m + (idx % 2) * 3.0)],
                "end": [26.0 + idx * 7.0, 240.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m + (idx % 2) * 3.0)],
                "initial_offset_m": 240.0 - idx * 34.0,
                "speed_mps": 3.0,
            }
        if entity_id in southbound_ids:
            idx = southbound_ids.index(entity_id)
            return {
                "kind": "linear",
                "start": [34.0 + idx * 8.0, 240.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m - 1.0 + (idx % 3) * 2.0)],
                "end": [34.0 + idx * 8.0, -180.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m - 1.0 + (idx % 3) * 2.0)],
                "initial_offset_m": 220.0 - idx * 38.0,
                "speed_mps": 3.1,
            }
        raise RuntimeError(f"Unhandled patrol route for UAV {entity_id}")

    def _linear_route_position(self, *, start: list[float], end: list[float], speed_mps: float, tick: int, initial_offset_m: float) -> tuple[list[float], list[float], float]:
        route_length = _distance_3d(start, end)
        if route_length <= 1e-6:
            return list(start), [0.0, 0.0, 0.0], 0.0
        direction = [
            (float(end[index]) - float(start[index])) / route_length
            for index in range(3)
        ]
        progress_m = min(route_length, float(initial_offset_m) + float(speed_mps) * float(tick) * self.dt_s)
        next_progress_m = min(route_length, float(initial_offset_m) + float(speed_mps) * float(min(self.max_tick, tick + 1)) * self.dt_s)
        position = [
            float(start[index]) + direction[index] * progress_m
            for index in range(3)
        ]
        next_position = [
            float(start[index]) + direction[index] * next_progress_m
            for index in range(3)
        ]
        velocity = [
            (float(next_position[index]) - float(position[index])) / self.dt_s
            for index in range(3)
        ]
        return position, velocity, _heading_yaw_deg(velocity, 0.0)

    def _uav_divert_target(self, entity_id: str) -> list[float]:
        numeric_id = int(entity_id.rsplit("_", 1)[-1])
        return [
            112.0 + float((numeric_id % 4) * 10.0),
            6.0 + float((numeric_id % 5) * 6.0),
            self._absolute_altitude_m(self.uav_cruise_height_agl_m + float(numeric_id % 3)),
        ]

    def _local_uav_position(self, tick: int) -> tuple[list[float], list[float], float]:
        time_s = float(tick) * self.dt_s
        if tick < self.divert_start_tick:
            angle = 0.28 * time_s
            position = [
                float(self.center_enu_m[0]) + 12.0 * math.cos(angle),
                float(self.center_enu_m[1]) + 7.0 * math.sin(angle),
                self._absolute_altitude_m(self.uav_cruise_height_agl_m - 1.0) + 1.2 * math.sin(angle * 0.7),
            ]
        else:
            angle = 0.18 * time_s + 0.8
            position = [
                66.0 + 5.0 * math.cos(angle),
                28.0 + 3.0 * math.sin(angle),
                self._absolute_altitude_m(self.uav_cruise_height_agl_m - 2.0) + 0.8 * math.sin(angle),
            ]
        next_position, _, _ = self._local_uav_position(min(self.max_tick, tick + 1)) if tick < self.max_tick else (position, [0.0, 0.0, 0.0], 0.0)
        velocity = [
            (float(next_position[index]) - float(position[index])) / self.dt_s
            for index in range(3)
        ]
        return position, velocity, _heading_yaw_deg(velocity, 90.0)

    def _inspection_uav_position(self, tick: int) -> tuple[list[float], list[float], float]:
        arrival_tick = self.inspection_tick + _safe_tick(5.0, self.tick_hz)
        descend_tick = arrival_tick + _safe_tick(3.0, self.tick_hz)
        hold_origin = [118.0, -28.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m + 4.0)]
        high_hover = [float(self.fall_target_enu_m[0]), float(self.fall_target_enu_m[1]), self._absolute_altitude_m(self.uav_cruise_height_agl_m)]
        low_hover = [float(self.fall_target_enu_m[0]), float(self.fall_target_enu_m[1]), self._absolute_altitude_m(self.uav_inspection_height_agl_m)]
        if tick <= arrival_tick:
            alpha = _clamp(float(tick - self.inspection_tick) / max(1.0, float(arrival_tick - self.inspection_tick)), 0.0, 1.0)
            position = _lerp_vec3(hold_origin, high_hover, alpha)
        elif tick <= descend_tick:
            alpha = _clamp(float(tick - arrival_tick) / max(1.0, float(descend_tick - arrival_tick)), 0.0, 1.0)
            position = _lerp_vec3(high_hover, low_hover, alpha)
        else:
            # Keep the inspection UAV directly above the fallen pedestrian once it
            # finishes descending so the final inspection frames remain exactly aligned.
            position = list(low_hover)
        if tick >= self.max_tick:
            next_position = list(position)
        else:
            next_position, _, _ = self._inspection_uav_position(min(self.max_tick, tick + 1))
        velocity = [
            (float(next_position[index]) - float(position[index])) / self.dt_s
            for index in range(3)
        ]
        if math.hypot(float(velocity[0]), float(velocity[1])) <= 1e-6:
            yaw_deg = _look_at_rotation_deg(position, list(self.fall_target_enu_m))["yaw_deg"]
        else:
            yaw_deg = _heading_yaw_deg(velocity, 0.0)
        return position, velocity, yaw_deg

    def _transient_uav_position(self, tick: int) -> tuple[list[float], list[float], float]:
        transient_start_tick = self.inspection_tick + _safe_tick(5.0, self.tick_hz)
        transient_end_tick = transient_start_tick + _safe_tick(12.0, self.tick_hz)
        start = [12.0, 31.0, self._absolute_altitude_m(self.uav_transient_height_agl_m)]
        end = [84.0, 31.0, self._absolute_altitude_m(self.uav_transient_height_agl_m)]
        if tick < transient_start_tick:
            position = [12.0, 40.0, self._absolute_altitude_m(self.uav_transient_height_agl_m)]
            velocity = [0.0, 0.0, 0.0]
            return position, velocity, 0.0
        alpha = _clamp(float(tick - transient_start_tick) / max(1.0, float(transient_end_tick - transient_start_tick)), 0.0, 1.0)
        position = _lerp_vec3(start, end, alpha)
        next_alpha = _clamp(float(min(self.max_tick, tick + 1) - transient_start_tick) / max(1.0, float(transient_end_tick - transient_start_tick)), 0.0, 1.0)
        next_position = _lerp_vec3(start, end, next_alpha)
        velocity = [
            (float(next_position[index]) - float(position[index])) / self.dt_s
            for index in range(3)
        ]
        return position, velocity, _heading_yaw_deg(velocity, 0.0)

    def _uav_state_at_tick(self, entity_id: str, tick: int) -> UavRuntimeState:
        tick_int = int(tick)
        if entity_id == self.local_uav_id:
            position, velocity, yaw_deg = self._local_uav_position(tick_int)
            return UavRuntimeState(
                active=True,
                position_enu_m=position,
                velocity_enu_mps=velocity,
                activity_type="monitoring" if tick_int >= self.divert_start_tick else "patrolling",
                animation_hint="uav_hover",
                mission_id="route_a_residual_monitoring",
                fallback_yaw_deg=yaw_deg,
            )

        if entity_id == self.inspection_uav_id:
            if tick_int < self.divert_start_tick:
                route = self._patrol_uav_route(entity_id)
                position, velocity, yaw_deg = self._linear_route_position(
                    start=list(route["start"]),
                    end=list(route["end"]),
                    speed_mps=float(route["speed_mps"]),
                    tick=tick_int,
                    initial_offset_m=float(route["initial_offset_m"]),
                )
                return UavRuntimeState(
                    active=True,
                    position_enu_m=position,
                    velocity_enu_mps=velocity,
                    activity_type="patrolling",
                    animation_hint="uav_hover",
                    mission_id="route_a_dense_patrol",
                    fallback_yaw_deg=yaw_deg,
                )
            if tick_int < self.inspection_tick:
                hold_position = [118.0, -28.0, self._absolute_altitude_m(self.uav_cruise_height_agl_m + 4.0)]
                return UavRuntimeState(
                    active=False,
                    position_enu_m=hold_position,
                    velocity_enu_mps=[0.0, 0.0, 0.0],
                    activity_type="holding",
                    animation_hint="uav_hover",
                    mission_id="inspection_elder_fall",
                    offstage_reason="inspection_uav_waiting_for_dispatch",
                    fallback_yaw_deg=135.0,
                )
            position, velocity, yaw_deg = self._inspection_uav_position(tick_int)
            return UavRuntimeState(
                active=True,
                position_enu_m=position,
                velocity_enu_mps=velocity,
                activity_type="inspection",
                animation_hint="uav_hover",
                mission_id="inspection_elder_fall",
                fallback_yaw_deg=yaw_deg,
            )

        if entity_id == self.transient_uav_id:
            if tick_int < self.divert_start_tick:
                route = self._patrol_uav_route(entity_id)
                position, velocity, yaw_deg = self._linear_route_position(
                    start=list(route["start"]),
                    end=list(route["end"]),
                    speed_mps=float(route["speed_mps"]),
                    tick=tick_int,
                    initial_offset_m=float(route["initial_offset_m"]),
                )
                return UavRuntimeState(
                    active=True,
                    position_enu_m=position,
                    velocity_enu_mps=velocity,
                    activity_type="patrolling",
                    animation_hint="uav_hover",
                    mission_id="route_a_dense_patrol",
                    fallback_yaw_deg=yaw_deg,
                )
            transient_start_tick = self.inspection_tick + _safe_tick(5.0, self.tick_hz)
            transient_end_tick = transient_start_tick + _safe_tick(12.0, self.tick_hz)
            if tick_int < transient_start_tick:
                return UavRuntimeState(
                    active=False,
                    position_enu_m=[12.0, 40.0, self._absolute_altitude_m(self.uav_transient_height_agl_m)],
                    velocity_enu_mps=[0.0, 0.0, 0.0],
                    activity_type="holding",
                    animation_hint="uav_hover",
                    mission_id="route_a_transient_pass",
                    offstage_reason="transient_uav_outside_pass_window",
                    fallback_yaw_deg=0.0,
                )
            if tick_int <= transient_end_tick:
                position, velocity, yaw_deg = self._transient_uav_position(tick_int)
                return UavRuntimeState(
                    active=True,
                    position_enu_m=position,
                    velocity_enu_mps=velocity,
                    activity_type="transit",
                    animation_hint="uav_hover",
                    mission_id="route_a_transient_pass",
                    fallback_yaw_deg=yaw_deg,
                )
            return UavRuntimeState(
                active=False,
                position_enu_m=[84.0, 31.0, self._absolute_altitude_m(self.uav_transient_height_agl_m)],
                velocity_enu_mps=[0.0, 0.0, 0.0],
                activity_type="transit",
                animation_hint="uav_hover",
                mission_id="route_a_transient_pass",
                offstage_reason="transient_uav_outside_pass_window",
                fallback_yaw_deg=0.0,
            )

        route = self._patrol_uav_route(entity_id)
        if tick_int < self.divert_start_tick:
            position, velocity, yaw_deg = self._linear_route_position(
                start=list(route["start"]),
                end=list(route["end"]),
                speed_mps=float(route["speed_mps"]),
                tick=tick_int,
                initial_offset_m=float(route["initial_offset_m"]),
            )
            return UavRuntimeState(
                active=True,
                position_enu_m=position,
                velocity_enu_mps=velocity,
                activity_type="patrolling",
                animation_hint="uav_hover",
                mission_id="route_a_dense_patrol",
                fallback_yaw_deg=yaw_deg,
            )
        if tick_int < self.divert_end_tick:
            start_position, _, _ = self._linear_route_position(
                start=list(route["start"]),
                end=list(route["end"]),
                speed_mps=float(route["speed_mps"]),
                tick=self.divert_start_tick,
                initial_offset_m=float(route["initial_offset_m"]),
            )
            alpha = _clamp(float(tick_int - self.divert_start_tick) / max(1.0, float(self.divert_end_tick - self.divert_start_tick)), 0.0, 1.0)
            target_position = self._uav_divert_target(entity_id)
            position = _lerp_vec3(start_position, target_position, alpha)
            next_alpha = _clamp(float(min(self.max_tick, tick_int + 1) - self.divert_start_tick) / max(1.0, float(self.divert_end_tick - self.divert_start_tick)), 0.0, 1.0)
            next_position = _lerp_vec3(start_position, target_position, next_alpha)
            velocity = [
                (float(next_position[index]) - float(position[index])) / self.dt_s
                for index in range(3)
            ]
            return UavRuntimeState(
                active=True,
                position_enu_m=position,
                velocity_enu_mps=velocity,
                activity_type="diverting",
                animation_hint="uav_hover",
                mission_id="divert_to_adjacent_intersection",
                fallback_yaw_deg=_heading_yaw_deg(velocity, 0.0),
            )
        target_position = self._uav_divert_target(entity_id)
        return UavRuntimeState(
            active=False,
            position_enu_m=target_position,
            velocity_enu_mps=[0.0, 0.0, 0.0],
            activity_type="diverted",
            animation_hint="uav_hover",
            mission_id="divert_to_adjacent_intersection",
            offstage_reason="diverted_from_demo_roi_after_rain",
            fallback_yaw_deg=0.0,
        )

    def _weather_annotation_for_entity(self, entity: dict[str, Any], weather_payload: dict[str, Any]) -> None:
        annotations = dict(entity.get("annotations") or {})
        annotations["weather"] = {
            "condition": weather_payload["condition"],
            "visibility": weather_payload["visibility"],
            "wind_speed": weather_payload["wind_speed"],
            "surface_state_a": weather_payload["surface_state_a"],
            "surface_friction_scale_a": weather_payload["surface_friction_scale_a"],
        }
        entity["annotations"] = annotations

    def _make_vehicle_entity(self, entity_id: str, tick: int, weather_payload: dict[str, Any]) -> dict[str, Any]:
        roster_entry = next(item for item in self.demo_roster_entities if item["entity_id"] == entity_id)
        entity = copy.deepcopy(roster_entry)
        state = self.vehicle_state_cache[int(tick)][entity_id]
        speed_mps = math.sqrt(sum(float(value) ** 2 for value in state["velocity_enu_mps"]))
        annotations = {
            "approach_id": state["approach_id"],
            "lane_id": state["lane_id"],
            "speed_mps": round(speed_mps, 4),
            "lights_on": bool(state["lights_on"]),
            "state_facets": {
                "activity": _activity_fields(
                    state["activity_type"],
                    animation_hint="vehicle_move" if speed_mps > 0.05 else "vehicle_wait",
                    posture="driving",
                    social_state="calm",
                ),
                "network": {
                    "bandwidth_load": 0.15,
                    "blind_spot": False,
                    "connected": True,
                    "coverage_score": 0.88,
                    "handover_event": None,
                    "latency_ms": 8.0,
                    "serving_bs_id": "base_station_alpha",
                },
            },
            "weather": {
                "condition": weather_payload["condition"],
                "visibility": weather_payload["visibility"],
                "wind_speed": weather_payload["wind_speed"],
                "surface_state_a": weather_payload["surface_state_a"],
                "surface_friction_scale_a": weather_payload["surface_friction_scale_a"],
            },
        }
        entity["annotations"] = annotations
        entity["render_presence"] = _render_presence(active=True, roi_id=self.roi_id)
        entity["truth_pose"] = _truth_pose(
            state["position_enu_m"],
            {"pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": state["yaw_deg"]},
            state["velocity_enu_mps"],
        )
        entity["state_revision"] = int(tick) + 1
        entity["visual_revision"] = 1
        return entity

    def _make_pedestrian_entity(self, entity_id: str, tick: int, weather_payload: dict[str, Any]) -> dict[str, Any]:
        roster_entry = next(item for item in self.demo_roster_entities if item["entity_id"] == entity_id)
        entity = copy.deepcopy(roster_entry)
        state = self.pedestrian_state_cache[int(tick)][entity_id]
        world_position_enu_m = self._world_position(list(state["position_enu_m"]))
        world_velocity_enu_mps = self._world_vector(list(state["velocity_enu_mps"]))
        world_yaw_deg = self._world_yaw_deg(float(state["yaw_deg"]))
        speed_mps = math.sqrt(sum(float(value) ** 2 for value in state["velocity_enu_mps"]))
        annotations = {
            "activity_type": state["activity_type"],
            "path_id": state["path_id"],
            "speed_mps": round(speed_mps, 4) if state["activity_type"] != "medical_incident" else 0.0,
            "state_facets": {
                "activity": _activity_fields(
                    state["activity_type"],
                    animation_hint=state["animation_hint"],
                    posture=state["posture"],
                    social_state=state["social_state"],
                ),
            },
            "weather": {
                "condition": weather_payload["condition"],
                "visibility": weather_payload["visibility"],
                "wind_speed": weather_payload["wind_speed"],
                "surface_state_a": weather_payload["surface_state_a"],
                "surface_friction_scale_a": weather_payload["surface_friction_scale_a"],
            },
        }
        if self.pedestrian_plans[entity_id].umbrella_after_rain and int(tick) >= self.rain_start_tick:
            entity_tags = list(entity.get("tags") or [])
            if "umbrella" not in entity_tags:
                entity_tags.append("umbrella")
            entity["tags"] = entity_tags
            annotations["rain_pedestrian_mode"] = "umbrella"
        entity["annotations"] = annotations
        entity["render_presence"] = _render_presence(active=True, roi_id=self.roi_id)
        entity["truth_pose"] = _truth_pose(
            world_position_enu_m,
            {"pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": world_yaw_deg},
            world_velocity_enu_mps,
        )
        entity["variant_id"] = self.ped_variant_by_id[entity_id]
        entity["state_revision"] = int(tick) + 1
        entity["visual_revision"] = 1
        return entity

    def _make_uav_entity(self, entity_id: str, tick: int, weather_payload: dict[str, Any]) -> dict[str, Any]:
        state = self._uav_state_at_tick(entity_id, tick)
        world_position_enu_m = self._world_position(list(state.position_enu_m))
        world_velocity_enu_mps = self._world_vector(list(state.velocity_enu_mps))
        world_yaw_deg = self._world_yaw_deg(_heading_yaw_deg(state.velocity_enu_mps, state.fallback_yaw_deg))
        network_state = {
            "bandwidth_load": 0.12 if state.activity_type == "inspection" else 0.08,
            "blind_spot": False,
            "connected": True,
            "coverage_score": 0.92 if state.active else 0.72,
            "handover_event": None,
            "latency_ms": 14.0,
            "serving_bs_id": "base_station_alpha",
        }
        annotations = {
            "speed_mps": round(math.sqrt(sum(float(value) ** 2 for value in state.velocity_enu_mps)), 4),
            "state_facets": {
                "activity": _activity_fields(
                    state.activity_type,
                    animation_hint=state.animation_hint,
                    posture="hover",
                    social_state="neutral",
                ),
                "assignment": {
                    "assignee_role": "inspection" if state.activity_type == "inspection" else "patrol",
                    "home_target": list(self.center_world_enu_m),
                    "mission_id": state.mission_id,
                    "observation_target": list(self.fall_target_world_enu_m),
                    "priority": 2 if state.activity_type == "inspection" else 1,
                },
                "network": network_state,
                "operational": {
                    "mode": "enabled" if state.active else "standby",
                    "reason": "" if state.active else state.offstage_reason,
                },
            },
            "weather": {
                "condition": weather_payload["condition"],
                "visibility": weather_payload["visibility"],
                "wind_speed": weather_payload["wind_speed"],
            },
        }
        return {
            "annotations": annotations,
            "entity_category": "uav",
            "entity_id": entity_id,
            "entity_kind": "uav.drone",
            "proxy_template_id": "drone.quadrotor",
            "render_presence": _render_presence(
                active=state.active,
                roi_id=self.roi_id,
                offstage_reason=state.offstage_reason or "uav_outside_demo_window",
            ),
            "site_id": self.site_id,
            "state_revision": int(tick) + 1,
            "tags": ["site_a", "uav", "demo_dense"],
            "truth_pose": _truth_pose(
                world_position_enu_m,
                {"pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": world_yaw_deg},
                world_velocity_enu_mps,
            ),
            "visual_revision": 1,
        }

    def _build_frame_entities(self, tick: int, weather_payload: dict[str, Any]) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        for entity_id in self.vehicle_ids:
            entities.append(self._make_vehicle_entity(entity_id, tick, weather_payload))
        for entity_id in self.pedestrian_ids:
            entities.append(self._make_pedestrian_entity(entity_id, tick, weather_payload))
        for entity_id in self.pre_rain_patrol_uav_ids:
            entities.append(self._make_uav_entity(entity_id, tick, weather_payload))
        return entities

    def _build_roster_summary(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        active_roi_entity_count = 0
        offstage_retained_count = 0
        for entity in entities:
            render_presence = dict(entity.get("render_presence") or {})
            if bool(render_presence.get("offstage", False)):
                offstage_retained_count += 1
            elif str(render_presence.get("submission_state") or "") == "submit_to_ue":
                active_roi_entity_count += 1
        return {
            "active_roi_entity_count": active_roi_entity_count,
            "category_counts_global": dict(self.category_counts),
            "global_entity_count": len(self.demo_roster_entities),
            "intersection_a_counts": {
                "pedestrian": 12,
                "uav": 24,
                "vehicle": 24,
            },
            "intersection_a_minimum_contract_satisfied": True,
            "offstage_retained_count": offstage_retained_count,
            "site_counts_global": dict(self.site_counts),
        }

    def _build_truth_frames(self) -> list[dict[str, Any]]:
        bounds = dict(
            (self.seed_scenario_plan.get("compiled_plan_summary") or {})
            .get("roi_windows", {})
            .get(self.roi_id, {})
        )
        truth_frames: list[dict[str, Any]] = []
        for tick in range(0, self.max_tick + 1):
            seed_frame = copy.deepcopy(self.frame_template)
            weather_payload = self._active_weather(tick)
            frame_entities = self._build_frame_entities(tick, weather_payload)
            seed_frame["episode_id"] = self.episode_id
            seed_frame["frame_id"] = _frame_id(self.episode_id, tick)
            seed_frame["frame_seq"] = tick
            seed_frame["tick"] = tick
            seed_frame["sim_time_s"] = round(float(tick) * self.dt_s, 3)
            seed_frame["tick_hz"] = self.tick_hz
            seed_frame["dt_s"] = self.dt_s
            seed_frame["schema_name"] = "truth_frame"
            seed_frame["schema_version"] = "v1"
            seed_frame["map_id"] = "donghu_road_topo"
            seed_frame["render_mode"] = "a_only"
            seed_frame["active_roi_id"] = self.roi_id
            seed_frame["active_roi_bounds_enu_m"] = copy.deepcopy(bounds)
            seed_frame["entities"] = frame_entities
            seed_frame["roster_summary"] = self._build_roster_summary(frame_entities)
            truth_frames.append(seed_frame)
        return truth_frames

    def _trajectory_row(self, entity: dict[str, Any], tick: int, sim_time_s: float, frame_id: str) -> dict[str, Any]:
        annotations = dict(entity.get("annotations") or {})
        truth_pose = dict(entity.get("truth_pose") or {})
        activity = dict((annotations.get("state_facets") or {}).get("activity") or {})
        position = [float(value) for value in truth_pose.get("position_enu_m") or [0.0, 0.0, 0.0]]
        velocity = [float(value) for value in truth_pose.get("velocity_enu_mps") or [0.0, 0.0, 0.0]]
        speed_mps = round(math.sqrt(sum(float(value) ** 2 for value in velocity)), 4)
        return {
            "agent_id": str(entity.get("entity_id") or ""),
            "category": str(entity.get("entity_category") or ""),
            "dynamic": str(entity.get("entity_category") or "") in {"vehicle", "pedestrian", "uav"},
            "entity_id": str(entity.get("entity_id") or ""),
            "entity_kind": str(entity.get("entity_kind") or entity.get("entity_type") or ""),
            "entity_category": str(entity.get("entity_category") or ""),
            "episode_id": self.episode_id,
            "frame_id": frame_id,
            "frame_seq": tick,
            "linked_processes": [],
            "linked_tasks": [],
            "network_state": copy.deepcopy(((annotations.get("state_facets") or {}).get("network") or {})),
            "node_id": str(entity.get("entity_id") or ""),
            "node_type": str(entity.get("entity_kind") or entity.get("entity_type") or ""),
            "position": list(position),
            "position_enu_m": list(position),
            "position_source": "truth_frame",
            "resolve_status": "pending",
            "resolved_frame_id": frame_id,
            "resolved_pose": {},
            "sample_id": f"trajectory:{tick}:{entity.get('entity_id')}",
            "sensor_id": "",
            "sim_time_s": sim_time_s,
            "tick": tick,
            "truth_frame_id": frame_id,
            "truth_pose": copy.deepcopy(truth_pose),
            "velocity_enu_mps": list(velocity),
            "speed_mps": speed_mps,
            "action": str(activity.get("animation_hint") or activity.get("activity_type") or annotations.get("activity_type") or "idle"),
            "posture": str(activity.get("posture") or "standing"),
        }

    def _build_trajectories(self, truth_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for frame in truth_frames:
            tick = int(frame["tick"])
            sim_time_s = float(frame["sim_time_s"])
            frame_id = str(frame["frame_id"])
            for entity in frame.get("entities", []):
                rows.append(self._trajectory_row(entity, tick, sim_time_s, frame_id))
        return rows

    def _build_weather_meta(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for tick in range(0, self.max_tick + 1):
            sim_time_s = round(float(tick) * self.dt_s, 3)
            payload = self._active_weather(tick)
            rows.append(
                {
                    "activated_tick": self.rain_start_tick if tick >= self.rain_start_tick else None,
                    "agent_id": "",
                    "condition": payload["condition"],
                    "episode_id": self.episode_id,
                    "frame_id": _frame_id(self.episode_id, tick),
                    "fog_density": payload.get("fog_density"),
                    "rain": payload.get("rain"),
                    "sample_id": f"weather:{tick}:weather_global",
                    "schema_name": "weather_meta",
                    "schema_version": "v1",
                    "sensor_id": "",
                    "sim_time_s": sim_time_s,
                    "source_event_id": "evt_demo_weather_rain_start" if tick >= self.rain_start_tick else "",
                    "source_kind": "scheduled" if tick >= self.rain_start_tick else "",
                    "source_tick": self.rain_start_tick if tick >= self.rain_start_tick else None,
                    "tick": tick,
                    "visibility_m": payload["visibility"],
                    "weather_entity_id": "weather_global",
                    "wetness": payload.get("wetness"),
                    "wind_mps": payload["wind_speed"],
                }
            )
        return rows

    def _add_event_and_label(
        self,
        *,
        tick: int,
        chain_id: str,
        instance_id: str,
        topic: str,
        title: str,
        category: str,
        overlay: str,
        severity: str,
        target_ids: list[str],
        target_id: str,
        target_kind: str,
        metadata: dict[str, Any],
    ) -> None:
        frame_id = _frame_id(self.episode_id, tick)
        sim_time_s = round(float(tick) * self.dt_s, 3)
        self.event_rows.append(
            {
                "activated_frame_id": frame_id,
                "activated_tick": tick,
                "agent_id": "",
                "causal_delay_ticks": 0,
                "chain_id": chain_id,
                "depth": 0,
                "effect_refs": [],
                "episode_id": self.episode_id,
                "frame_id": frame_id,
                "instance_id": instance_id,
                "metadata": metadata,
                "parent_event_id": "",
                "payload": {
                    "activated_tick": tick,
                    "category": category,
                    "causal_delay_ticks": 0,
                    "duration_ticks": 0,
                    "end_tick": tick,
                    "event_id": topic,
                    "phase": metadata.get("phase", ""),
                    "roi_id": metadata.get("roi_id", self.roi_id),
                    "sequence_no": len(self.event_rows) + 1,
                    "source_kind": "scheduled",
                    "source_tick": tick,
                    "source_topic": topic,
                    "start_tick": tick,
                    "title": title,
                },
                "published_event_refs": [],
                "recovered_frame_id": "",
                "recovered_tick": None,
                "render_hints": {
                    "overlay": overlay,
                    "severity": severity,
                },
                "sample_id": f"event_trace:{tick}:{instance_id}",
                "schema_name": "event_trace",
                "schema_version": "v1",
                "scope": {
                    "bbox": [],
                    "center": list(self.center_world_enu_m),
                    "entities": list(target_ids[:4]),
                    "fields": [],
                    "kind": "entity",
                    "relations": [],
                    "target_id": target_id,
                    "world_features": [],
                },
                "semantic_class": "state_event",
                "sensor_id": "",
                "sim_time_s": sim_time_s,
                "source_event_id": topic,
                "source_frame_id": frame_id,
                "source_kind": "scheduled",
                "source_tick": tick,
                "source_topic": topic,
                "state_diff_refs": [],
                "status": "active",
                "target_ids": list(target_ids),
                "tick": tick,
                "topic": topic,
            }
        )
        self.dynamic_labels.append(
            {
                "agent_id": "",
                "chain_id": chain_id,
                "effect_id": f"effect_{instance_id}",
                "episode_id": self.episode_id,
                "event_id": instance_id,
                "facet": metadata.get("facet", category),
                "frame_id": frame_id,
                "overlay": overlay,
                "render_hints": {
                    "overlay": overlay,
                    "severity": severity,
                },
                "sample_id": f"label:{tick}:{instance_id}",
                "sensor_id": "",
                "severity": severity,
                "sim_time_s": sim_time_s,
                "target_id": target_id,
                "target_kind": target_kind,
                "tick": tick,
            }
        )

    def _build_event_artifacts(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.event_rows.clear()
        self.dynamic_labels.clear()

        self._add_event_and_label(
            tick=self.rain_start_tick,
            chain_id="event_chain.donghu.demo_dense_uav.weather.rain.start.v1",
            instance_id="evtinst_demo_weather_rain_start",
            topic="evt_demo_weather_rain_start",
            title="Rain starts over intersection A",
            category="weather",
            overlay="weather",
            severity="warning",
            target_ids=["weather_global"],
            target_id="weather_global",
            target_kind="entity",
            metadata={
                "phase": "rain_start",
                "roi_id": self.roi_id,
                "facet": "weather",
            },
        )

        diverted_ids = [entity_id for entity_id in self.departing_uav_ids if entity_id != self.local_uav_id]
        self._add_event_and_label(
            tick=self.divert_start_tick,
            chain_id="event_chain.donghu.demo_dense_uav.uav.divert.local.v1",
            instance_id="evtinst_demo_uav_divert_local",
            topic="evt_demo_uav_divert_local",
            title="Most patrol UAVs divert away after rain",
            category="uav_mission",
            overlay="uav_mission",
            severity="warning",
            target_ids=diverted_ids,
            target_id=self.local_uav_id,
            target_kind="entity",
            metadata={
                "phase": "uav_divert",
                "roi_id": self.roi_id,
                "facet": "assignment",
            },
        )

        self._add_event_and_label(
            tick=self.fall_tick,
            chain_id="event_chain.donghu.demo_dense_uav.pedestrian.fall.triggered.a.v1",
            instance_id="evtinst_demo_pedestrian_fall",
            topic="evt_demo_pedestrian_fall_triggered_a",
            title="Elder pedestrian falls at the lower crosswalk",
            category="pedestrian",
            overlay="pedestrian",
            severity="warning",
            target_ids=["pedestrian_a_005"],
            target_id="pedestrian_a_005",
            target_kind="entity",
            metadata={
                "phase": "pedestrian_fall",
                "roi_id": self.roi_id,
                "facet": "activity",
            },
        )

        self._add_event_and_label(
            tick=self.inspection_tick,
            chain_id="event_chain.donghu.demo_dense_uav.uav.dispatch.inspection.a.v1",
            instance_id="evtinst_demo_uav_dispatch_inspection",
            topic="evt_demo_uav_dispatch_inspection_a",
            title="Inspection UAV arrives over the fall location",
            category="uav_mission",
            overlay="uav_mission",
            severity="info",
            target_ids=[self.inspection_uav_id, "pedestrian_a_005"],
            target_id=self.inspection_uav_id,
            target_kind="entity",
            metadata={
                "phase": "inspection",
                "roi_id": self.roi_id,
                "facet": "assignment",
            },
        )

        return list(self.event_rows), list(self.dynamic_labels)

    def _build_capture_plan(self) -> dict[str, Any]:
        plan = dict(self.spec.get("capture_plan") or {})
        high_camera = dict(plan.get("high_overview") or {})
        ground_camera = dict(plan.get("ground_event") or {})
        high_position = [
            float(high_camera["position_enu_m"][0]),
            float(high_camera["position_enu_m"][1]),
            self._absolute_altitude_m(self.camera_high_height_agl_m),
        ]
        high_look_at = [
            float(high_camera["look_at_enu_m"][0]),
            float(high_camera["look_at_enu_m"][1]),
            self.site_ground_elevation_m,
        ]
        ground_position = [
            float(ground_camera["position_enu_m"][0]),
            float(ground_camera["position_enu_m"][1]),
            self._absolute_altitude_m(self.camera_low_height_agl_m),
        ]
        ground_look_at = [
            float(ground_camera["look_at_enu_m"][0]),
            float(ground_camera["look_at_enu_m"][1]),
            self.site_ground_elevation_m,
        ]
        high_position = self._world_position(high_position)
        high_look_at = self._world_position(high_look_at)
        ground_position = self._world_position(ground_position)
        ground_look_at = self._world_position(ground_look_at)
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "episode_dir": _repo_relative_string(self.output_root / DEMO_EPISODE_RELATIVE_DIR / self.episode_id, project_root=self.project_root),
            "site_id": self.site_id,
            "roi_id": self.roi_id,
            "capture_interval_s": float(plan["capture_interval_s"]),
            "phase_script": {
                "0-45s": "dense_uav_patrol_clear_weather",
                "45-50s": "rain_start_with_full_airspace",
                "50-65s": "uav_diversion_with_residual_local_monitor",
                "60s": "elder_fall_triggered",
                "65-90s": "inspection_uav_arrives_with_reduced_density",
            },
            "cameras": [
                {
                    "camera_id": str(high_camera["camera_id"]),
                    "role": "high_overview",
                    "position_enu_m": high_position,
                    "look_at_enu_m": high_look_at,
                    "rotation_deg": {
                        "pitch_deg": -90.0,
                        "yaw_deg": 0.0,
                        "roll_deg": 0.0,
                    },
                    "fov_degrees": float(high_camera["fov_degrees"]),
                    "width": int(high_camera["width"]),
                    "height": int(high_camera["height"]),
                },
                {
                    "camera_id": str(ground_camera["camera_id"]),
                    "role": "ground_event",
                    "position_enu_m": ground_position,
                    "look_at_enu_m": ground_look_at,
                    "rotation_deg": _look_at_rotation_deg(ground_position, ground_look_at),
                    "fov_degrees": float(ground_camera["fov_degrees"]),
                    "width": int(ground_camera["width"]),
                    "height": int(ground_camera["height"]),
                },
            ],
        }

    def _build_scenario_plan(self) -> dict[str, Any]:
        scenario_plan = copy.deepcopy(self.seed_scenario_plan)
        compiled_summary = dict(scenario_plan.get("compiled_plan_summary") or {})
        compiled_summary["category_counts"] = dict(self.category_counts)
        compiled_summary["global_entity_count"] = len(self.demo_roster_entities)
        compiled_summary["incident_event_count"] = 1
        compiled_summary["weather_event_count"] = 1
        compiled_summary["intersection_a_counts"] = {
            "pedestrian": 12,
            "uav": 24,
            "vehicle": 24,
        }
        compiled_summary["site_counts"] = dict(self.site_counts)
        site_contracts = dict(compiled_summary.get("site_contracts") or {})
        if self.site_id in site_contracts:
            site_contracts[self.site_id] = dict(site_contracts[self.site_id])
            site_contracts[self.site_id]["minimum_counts"] = {
                "pedestrian": 12,
                "uav": 24,
                "vehicle": 24,
            }
        compiled_summary["site_contracts"] = site_contracts
        compiled_summary["path"] = CANONICAL_SCENARIO_SPEC_PATH
        compiled_summary["plan_id"] = self.scenario_id
        scenario_plan["available"] = True
        scenario_plan["episode_id"] = self.episode_id
        scenario_plan["scenario_id"] = self.scenario_id
        scenario_plan["scenario_name"] = self.scenario_name
        scenario_plan["compiled_plan_summary"] = compiled_summary
        scenario_plan["summary"] = copy.deepcopy(compiled_summary)
        scenario_plan["global_entity_roster"] = copy.deepcopy(self.demo_roster_entities)
        scenario_plan["schema_name"] = "scenario_plan"
        scenario_plan["schema_version"] = "v1"
        scenario_plan["runtime_support_entities"] = copy.deepcopy(self.runtime_support_entities)
        scenario_plan["scenario_plan"] = dict(scenario_plan.get("scenario_plan") or {})
        scenario_plan["scenario_plan"]["plan_id"] = self.scenario_id
        scenario_plan["scenario_plan"]["path"] = CANONICAL_SCENARIO_SPEC_PATH
        scenario_plan["scenario_plan"]["site_contracts"] = copy.deepcopy(site_contracts)
        scenario_plan["scenario_plan"]["summary"] = copy.deepcopy(compiled_summary)
        return scenario_plan

    def _assemble_manifest(
        self,
        *,
        episode_dir: Path,
        truth_frames: list[dict[str, Any]],
        event_trace: list[dict[str, Any]],
        trajectories: list[dict[str, Any]],
        weather_meta: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ticks = [int(item.get("tick") or 0) for item in truth_frames]
        times = [float(item.get("sim_time_s") or 0.0) for item in truth_frames]
        dynamic_ids = {
            str(item.get("entity_id") or "")
            for item in trajectories
            if bool(item.get("dynamic"))
        }
        all_ids = {str(item.get("entity_id") or "") for item in trajectories if str(item.get("entity_id") or "")}
        artifacts = {
            "scenario_plan": _repo_relative_string(episode_dir / "scenario_plan.json", project_root=self.project_root),
            "global_entity_roster": _repo_relative_string(episode_dir / "global_entity_roster.json", project_root=self.project_root),
            "truth_frames": _repo_relative_string(episode_dir / "truth_frames.jsonl", project_root=self.project_root),
            "event_trace": _repo_relative_string(episode_dir / "event_trace.jsonl", project_root=self.project_root),
            "trajectories": _repo_relative_string(episode_dir / "trajectories.jsonl", project_root=self.project_root),
            "weather_meta": _repo_relative_string(episode_dir / "weather_meta.jsonl", project_root=self.project_root),
            "episode_manifest": _repo_relative_string(episode_dir / "episode_manifest.json", project_root=self.project_root),
        }
        record_counts = {
            "scenario_plan": 1,
            "global_entity_roster": len(self.demo_roster_entities),
            "truth_frames": len(truth_frames),
            "event_trace": len(event_trace),
            "trajectories": len(trajectories),
            "weather_meta": len(weather_meta),
            "episode_manifest": 1,
        }
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "generation": {
                "generator": _repo_relative_string(SCRIPT_DIR / "build.py", project_root=self.project_root),
                "spec_path": _repo_relative_string(self.spec_path, project_root=self.project_root),
                "seed_episode_dir": _repo_relative_string(self.seed_episode_dir, project_root=self.project_root),
            },
            "record_counts": record_counts,
            "canonical_record_counts": copy.deepcopy(record_counts),
            "node_counts": {
                "all_nodes": len(all_ids),
                "dynamic_nodes": len(dynamic_ids),
                "static_nodes": max(0, len(all_ids) - len(dynamic_ids)),
            },
            "artifacts": copy.deepcopy(artifacts),
            "canonical_artifacts": copy.deepcopy(artifacts),
            "time_range": {
                "tick_start": min(ticks) if ticks else 0,
                "tick_end": max(ticks) if ticks else 0,
                "sim_time_start": min(times) if times else 0.0,
                "sim_time_end": max(times) if times else 0.0,
            },
            "validation_summary": {
                "ok": True,
                "errors": [],
                "warnings": [],
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def generate(self) -> dict[str, Path]:
        episode_dir = self.output_root / DEMO_EPISODE_RELATIVE_DIR / self.episode_id
        capture_plan_path = self.output_root / DEMO_CAPTURE_PLAN_RELATIVE_PATH

        scenario_plan = self._build_scenario_plan()
        truth_frames = self._build_truth_frames()
        trajectories = self._build_trajectories(truth_frames)
        weather_meta = self._build_weather_meta()
        event_trace, dynamic_labels = self._build_event_artifacts()
        capture_plan = self._build_capture_plan()
        manifest = self._assemble_manifest(
            episode_dir=episode_dir,
            truth_frames=truth_frames,
            event_trace=event_trace,
            trajectories=trajectories,
            weather_meta=weather_meta,
        )

        _write_json(episode_dir / "scenario_plan.json", scenario_plan)
        _write_json(episode_dir / "global_entity_roster.json", {"entities": self.demo_roster_entities})
        _write_jsonl(episode_dir / "truth_frames.jsonl", truth_frames)
        _write_jsonl(episode_dir / "event_trace.jsonl", event_trace)
        _write_jsonl(episode_dir / "trajectories.jsonl", trajectories)
        _write_jsonl(episode_dir / "weather_meta.jsonl", weather_meta)
        _write_jsonl(episode_dir / "dynamic_labels.jsonl", dynamic_labels)
        _write_json(episode_dir / "episode_manifest.json", manifest)
        _write_json(
            episode_dir / "scenario_package.json",
            {
                "scenario_id": self.scenario_id,
                "episode_id": self.episode_id,
                "root_dir": _repo_relative_string(episode_dir, project_root=self.project_root),
                "truth_frames": _repo_relative_string(episode_dir / "truth_frames.jsonl", project_root=self.project_root),
                "weather_meta": _repo_relative_string(episode_dir / "weather_meta.jsonl", project_root=self.project_root),
                "scenario_plan": _repo_relative_string(episode_dir / "scenario_plan.json", project_root=self.project_root),
                "capture_plan": _repo_relative_string(capture_plan_path, project_root=self.project_root),
                "episode_manifest": _repo_relative_string(episode_dir / "episode_manifest.json", project_root=self.project_root),
            },
        )
        _write_json(capture_plan_path, capture_plan)

        return {
            "episode_dir": episode_dir,
            "capture_plan_path": capture_plan_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dense-UAV truth-only Donghu demo episode.")
    parser.add_argument("--spec", default=str(SPEC_PATH), help="Spec JSON path")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = DenseUavEpisodeGenerator(Path(args.spec), Path(args.output_root))
    outputs = generator.generate()
    print(
        json.dumps(
            {
                "episode_dir": str(outputs["episode_dir"]),
                "capture_plan_path": str(outputs["capture_plan_path"]),
                "episode_id": generator.episode_id,
                "scenario_id": generator.scenario_id,
                "tick_hz": generator.tick_hz,
                "duration_s": generator.duration_s,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
