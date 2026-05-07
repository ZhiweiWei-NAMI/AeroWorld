#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "episode_render_host_config.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aero_sim_client import AeroSimClient  # noqa: E402
from editor_hook_client import FixedWorldCaptureEditorHook  # noqa: E402
from donghu_core.capture_orchestrator import CaptureOrchestrator  # noqa: E402
from donghu_core.discovery import project_root_from, resolve_map_package  # noqa: E402
from donghu_core.event_script_interpreter import EventScriptInterpreter  # noqa: E402
from donghu_core.interfaces import RuntimeFrameRecord  # noqa: E402
from donghu_core.pedestrian_pose_service import PedestrianPoseService  # noqa: E402
from donghu_core.uav_execution_service import UavExecutionService  # noqa: E402
from donghu_core.weather_service import WeatherService  # noqa: E402


REQUIRED_CAPABILITIES = {
    "simAeroDescribeCapabilities",
    "simAeroLoadContext",
    "simAeroApplyFrame",
    "simAeroPollFeedback",
    "simAeroPedSpawn",
    "simAeroPedReset",
    "simAeroPedObserve",
    "simAeroPedSetTarget",
    "simAeroPedCommitCross",
    "simAeroPedPlayAnimation",
    "simAeroPedSpawnCrowd",
    "simAeroPedClearCrowd",
    "simAeroPedStop",
    "simAeroPedSetVariant",
    "simAeroPedRelease",
    "simAeroSpawnAsset",
    "simAeroMoveAsset",
    "simAeroRemoveAsset",
    "simAeroCaptureWorldCamera",
    "simAeroApplyWeather",
    "simAeroCreateRuntimeMultirotor",
    "simAeroMoveRuntimeMultirotor",
    "simAeroGetRuntimeMultirotorStatus",
    "simAeroRemoveRuntimeVehicle",
    "simAeroGetRuntimeVehiclePose",
}

SEMANTIC_SEGMENTATION_CLASSES: tuple[dict[str, Any], ...] = (
    {
        "class_id": 1,
        "class_name": "city_base_background",
        "actor_regex": r".*(BP_CityBaseGenerator0|BP_CityBaseGenerator_C.*|BP_CityBaseGenerator.*).*",
        "component_regex": r".*(BP_CityBaseGenerator0|BP_CityBaseGenerator_C.*|BP_CityBaseGenerator.*).*",
        "canonical_actor_label": "BP_CityBaseGenerator0",
        "required_for_static_audit": True,
        "category": "static_city_base",
    },
    {
        "class_id": 2,
        "class_name": "building_style1",
        "actor_regex": r".*BP_Archi_Style1_C.*",
        "component_regex": r".*BP_Archi_Style1_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 3,
        "class_name": "building_style3",
        "actor_regex": r".*BP_Archi_Style3_C.*",
        "component_regex": r".*BP_Archi_Style3_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 4,
        "class_name": "building_style4",
        "actor_regex": r".*BP_Archi_Style4_C.*",
        "component_regex": r".*BP_Archi_Style4_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 5,
        "class_name": "building_style05",
        "actor_regex": r".*BP_Archi_Style05_C.*",
        "component_regex": r".*BP_Archi_Style05_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 6,
        "class_name": "building_roof",
        "actor_regex": r".*BP_Archi_Roof_C.*",
        "component_regex": r".*BP_Archi_Roof_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 7,
        "class_name": "building_pitched_roof",
        "actor_regex": r".*BP_Archi_PitchedRoof_C.*",
        "component_regex": r".*BP_Archi_PitchedRoof_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 20,
        "class_name": "uav",
        "actor_regex": r".*(CaptureUAV_0|Quadrotor|RuntimeMultirotor|uav).*",
        "component_regex": r".*(CaptureUAV_0|Quadrotor|RuntimeMultirotor|uav).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 21,
        "class_name": "vehicle",
        "actor_regex": r".*(Vehicle|BoxCar|SUV|Ambulance|Police).*",
        "component_regex": r".*(Vehicle|BoxCar|SUV|Ambulance|Police).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 22,
        "class_name": "pedestrian",
        "actor_regex": r".*(Pedestrian|ped_).*",
        "component_regex": r".*(Pedestrian|ped_).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 23,
        "class_name": "roadwork_prop",
        "actor_regex": r".*(Roadwork|ConstructionFence|TrafficCone|Barrier).*",
        "component_regex": r".*(Roadwork|ConstructionFence|TrafficCone|Barrier).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 24,
        "class_name": "traffic_control",
        "actor_regex": r".*(TrafficControl|Signal|PoliceSign|PoliceTape).*",
        "component_regex": r".*(TrafficControl|Signal|PoliceSign|PoliceTape).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 25,
        "class_name": "facility",
        "actor_regex": r".*(LandingPad|Charger|BaseTower|Facility).*",
        "component_regex": r".*(LandingPad|Charger|BaseTower|Facility).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 26,
        "class_name": "hazard_trigger",
        "actor_regex": r".*(NoFly|Hazard|Trigger).*",
        "component_regex": r".*(NoFly|Hazard|Trigger).*",
        "required_for_static_audit": False,
        "category": "optional_rendered_trigger",
    },
    {
        "class_id": 27,
        "class_name": "service_misc_prop",
        "actor_regex": r".*(DeliveryBag|Backpack|Phone|Umbrella|Service).*",
        "component_regex": r".*(DeliveryBag|Backpack|Phone|Umbrella|Service).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
)


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    site_id: str
    roi_id: str
    tick_start: int
    tick_end: int


@dataclass(frozen=True)
class CoordinateTransform:
    enabled: bool
    translation_enu_m: tuple[float, float, float]
    axis_mapping: str
    yaw_deg: float
    scale_enu: tuple[float, float, float]

    @staticmethod
    def _read_vector(config: dict[str, Any], field_name: str, default: Sequence[float]) -> tuple[float, float, float]:
        raw = config.get(field_name, default)
        values = list(raw if isinstance(raw, Sequence) else default)
        return (
            float(values[0] if len(values) > 0 else default[0]),
            float(values[1] if len(values) > 1 else default[1]),
            float(values[2] if len(values) > 2 else default[2]),
        )

    @classmethod
    def from_config(cls, root_config: dict[str, Any]) -> "CoordinateTransform":
        config = dict(root_config.get("coordinate_transform") or {})
        return cls(
            enabled=bool(config.get("enabled", False)),
            translation_enu_m=cls._read_vector(config, "translation_enu_m", (0.0, 0.0, 0.0)),
            axis_mapping=str(config.get("axis_mapping", "XY_To_XY") or "XY_To_XY"),
            yaw_deg=float(config.get("yaw_deg", 0.0)),
            scale_enu=cls._read_vector(config, "scale_enu", (1.0, 1.0, 1.0)),
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

    def _transform_vector_components(self, value_enu: Sequence[float]) -> tuple[float, float, float]:
        x = float(value_enu[0] if len(value_enu) > 0 else 0.0) * self.scale_enu[0]
        y = float(value_enu[1] if len(value_enu) > 1 else 0.0) * self.scale_enu[1]
        z = float(value_enu[2] if len(value_enu) > 2 else 0.0) * self.scale_enu[2]
        if self.enabled:
            x, y, z = self._apply_axis_mapping(x, y, z)
            x, y = self._rotate_xy(x, y)
        return x, y, z

    def apply_position(self, position_enu_m: Sequence[float]) -> list[float]:
        x, y, z = self._transform_vector_components(position_enu_m)
        if self.enabled:
            x += self.translation_enu_m[0]
            y += self.translation_enu_m[1]
            z += self.translation_enu_m[2]
        return [x, y, z]

    def _inverse_axis_mapping(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        mapping = self.axis_mapping.strip() or "XY_To_XY"
        if mapping == "XY_To_XNegY":
            return x, -y, z
        if mapping == "XY_To_YX":
            return y, x, z
        if mapping == "XY_To_YNegX":
            return -y, x, z
        return x, y, z

    def inverse_vector(self, value_enu: Sequence[float]) -> list[float]:
        x = float(value_enu[0] if len(value_enu) > 0 else 0.0)
        y = float(value_enu[1] if len(value_enu) > 1 else 0.0)
        z = float(value_enu[2] if len(value_enu) > 2 else 0.0)
        if self.enabled and abs(self.yaw_deg) > 1e-6:
            yaw_rad = math.radians(-self.yaw_deg)
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            x, y = x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw
        if self.enabled:
            x, y, z = self._inverse_axis_mapping(x, y, z)
        return [
            x / (self.scale_enu[0] or 1.0),
            y / (self.scale_enu[1] or 1.0),
            z / (self.scale_enu[2] or 1.0),
        ]

    def inverse_position(self, position_enu_m: Sequence[float]) -> list[float]:
        x = float(position_enu_m[0] if len(position_enu_m) > 0 else 0.0)
        y = float(position_enu_m[1] if len(position_enu_m) > 1 else 0.0)
        z = float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0)
        if self.enabled:
            x -= self.translation_enu_m[0]
            y -= self.translation_enu_m[1]
            z -= self.translation_enu_m[2]
        return self.inverse_vector((x, y, z))

    def apply_rotation(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        result = {
            "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", 0.0))),
            "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", 0.0))),
            "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", 0.0))),
        }
        if self.enabled:
            yaw_rad = math.radians(result["yaw_deg"])
            forward_x, forward_y, _ = self._transform_vector_components((math.cos(yaw_rad), math.sin(yaw_rad), 0.0))
            if abs(forward_x) > 1e-6 or abs(forward_y) > 1e-6:
                result["yaw_deg"] = math.degrees(math.atan2(forward_y, forward_x))
        return result

    def inverse_rotation(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        result = {
            "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", 0.0))),
            "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", 0.0))),
            "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", 0.0))),
        }
        if self.enabled:
            yaw_rad = math.radians(result["yaw_deg"])
            local_x, local_y, _ = self.inverse_vector((math.cos(yaw_rad), math.sin(yaw_rad), 0.0))
            if abs(local_x) > 1e-6 or abs(local_y) > 1e-6:
                result["yaw_deg"] = math.degrees(math.atan2(local_y, local_x))
        return result

    def apply_vector(self, value_enu: Sequence[float]) -> list[float]:
        x, y, z = self._transform_vector_components(value_enu)
        return [x, y, z]

    def describe(self) -> str:
        return (
            f"enabled={self.enabled} "
            f"translation_enu_m={[self.translation_enu_m[0], self.translation_enu_m[1], self.translation_enu_m[2]]} "
            f"axis_mapping={self.axis_mapping} "
            f"yaw_deg={self.yaw_deg} "
            f"scale_enu={[self.scale_enu[0], self.scale_enu[1], self.scale_enu[2]]}"
        )


@dataclass(frozen=True)
class RoadLaneSample:
    edge_id: str
    lane_id: str
    s_m: float
    position_enu_m: tuple[float, float, float]
    yaw_deg: float


@dataclass(frozen=True)
class RoadTopologySnapResult:
    edge_id: str
    lane_id: str
    s_m: float
    position_enu_m: tuple[float, float, float]
    yaw_deg: float
    distance_m: float
    heading_error_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "lane_id": self.lane_id,
            "s_m": self.s_m,
            "position_enu_m": [self.position_enu_m[0], self.position_enu_m[1], self.position_enu_m[2]],
            "yaw_deg": self.yaw_deg,
            "distance_m": self.distance_m,
            "heading_error_deg": self.heading_error_deg,
        }


@dataclass(frozen=True)
class RoadSegment:
    road_id: int
    lanes: int
    width_m: float
    start_enu_m: tuple[float, float, float]
    end_enu_m: tuple[float, float, float]


@dataclass(frozen=True)
class RoadGeometryProjectionResult:
    road_id: int
    lanes: int
    width_m: float
    projected_enu_m: tuple[float, float, float]
    heading_deg: float
    signed_lateral_m: float
    distance_m: float
    left_normal_xy: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "road_id": self.road_id,
            "edge_id": f"cg_edge_{self.road_id}",
            "lanes": self.lanes,
            "width_m": self.width_m,
            "projected_enu_m": [self.projected_enu_m[0], self.projected_enu_m[1], self.projected_enu_m[2]],
            "heading_deg": self.heading_deg,
            "signed_lateral_m": self.signed_lateral_m,
            "distance_m": self.distance_m,
            "left_normal_xy": [self.left_normal_xy[0], self.left_normal_xy[1]],
        }


class RoadGeometryIndex:
    def __init__(
        self,
        road_geojson_path: Path,
        *,
        lane_center_samples_path: Path | None = None,
        cell_size_m: float = 25.0,
    ) -> None:
        self.road_geojson_path = road_geojson_path
        self.lane_center_samples_path = lane_center_samples_path
        self.cell_size_m = max(1.0, float(cell_size_m))
        self.road_props_by_id: dict[int, dict[str, Any]] = {}
        self.segments: list[RoadSegment] = []
        self.grid: dict[tuple[int, int], list[int]] = {}
        self._load()

    def _cell_coords(self, x_m: float, y_m: float) -> tuple[int, int]:
        return int(math.floor(x_m / self.cell_size_m)), int(math.floor(y_m / self.cell_size_m))

    @staticmethod
    def _road_id_from_edge_id(edge_id: str) -> int | None:
        clean = str(edge_id or "")
        if clean.startswith("cg_edge_"):
            clean = clean[len("cg_edge_") :]
        if "_" in clean:
            clean = clean.split("_", 1)[0]
        try:
            return int(clean)
        except ValueError:
            return None

    def _append_segment(self, road_id: int, lanes: int, width_m: float, start: tuple[float, float, float], end: tuple[float, float, float]) -> None:
        if math.hypot(end[0] - start[0], end[1] - start[1]) <= 1e-3:
            return
        segment = RoadSegment(
            road_id=road_id,
            lanes=lanes,
            width_m=width_m,
            start_enu_m=start,
            end_enu_m=end,
        )
        segment_index = len(self.segments)
        self.segments.append(segment)
        min_x = min(start[0], end[0])
        max_x = max(start[0], end[0])
        min_y = min(start[1], end[1])
        max_y = max(start[1], end[1])
        min_cell_x, min_cell_y = self._cell_coords(min_x, min_y)
        max_cell_x, max_cell_y = self._cell_coords(max_x, max_y)
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                self.grid.setdefault((cell_x, cell_y), []).append(segment_index)

    def _load(self) -> None:
        if not self.road_geojson_path.exists():
            raise RuntimeError(f"road.geojson was not found: {self.road_geojson_path}")
        root = json.loads(self.road_geojson_path.read_text(encoding="utf-8-sig"))
        for feature in root.get("features", []):
            props = dict(feature.get("properties") or {})
            road_id = int(props.get("id") or -1)
            if road_id < 0:
                continue
            lanes = max(1, int(props.get("lanes") or 1))
            width_m = max(3.0, float(props.get("width") or (lanes * 3.5)))
            self.road_props_by_id[road_id] = {
                "road_id": road_id,
                "lanes": lanes,
                "width_m": width_m,
                "direction": props.get("direction"),
                "func_level": props.get("func_level"),
            }

        if self.lane_center_samples_path is not None and self.lane_center_samples_path.exists():
            grouped_samples: dict[tuple[int, str], list[tuple[float, tuple[float, float, float]]]] = {}
            with self.lane_center_samples_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    road_id = self._road_id_from_edge_id(str(row.get("edge_id") or ""))
                    if road_id is None:
                        continue
                    lane_id = str(row.get("lane_id") or "")
                    grouped_samples.setdefault((road_id, lane_id), []).append(
                        (
                            float(row.get("s_m") or 0.0),
                            (
                                float(row.get("x_m") or 0.0),
                                float(row.get("y_m") or 0.0),
                                float(row.get("z_m") or 0.0),
                            ),
                        )
                    )
            for (road_id, _lane_id), rows in grouped_samples.items():
                meta = self.road_props_by_id.get(road_id) or {"lanes": 1, "width_m": 3.5}
                rows = sorted(rows, key=lambda item: item[0])
                for (_s0, start), (_s1, end) in zip(rows, rows[1:]):
                    self._append_segment(road_id, int(meta["lanes"]), float(meta["width_m"]), start, end)
        else:
            for feature in root.get("features", []):
                props = dict(feature.get("properties") or {})
                geometry = dict(feature.get("geometry") or {})
                road_id = int(props.get("id") or -1)
                if road_id < 0:
                    continue
                lanes = max(1, int(props.get("lanes") or 1))
                width_m = max(3.0, float(props.get("width") or (lanes * 3.5)))
                coords_groups: list[list[Any]] = []
                geom_type = str(geometry.get("type") or "")
                if geom_type == "LineString":
                    coords_groups = [list(geometry.get("coordinates") or [])]
                elif geom_type == "MultiLineString":
                    coords_groups = [list(group) for group in (geometry.get("coordinates") or [])]
                for coords in coords_groups:
                    for start_raw, end_raw in zip(coords, coords[1:]):
                        start = (
                            float(start_raw[0]),
                            float(start_raw[1]),
                            float(start_raw[2] if len(start_raw) > 2 else 0.0),
                        )
                        end = (
                            float(end_raw[0]),
                            float(end_raw[1]),
                            float(end_raw[2] if len(end_raw) > 2 else 0.0),
                        )
                        self._append_segment(road_id, lanes, width_m, start, end)

        if not self.segments:
            raise RuntimeError(
                f"No usable road line segments were parsed from {self.lane_center_samples_path or self.road_geojson_path}"
            )

    def describe(self) -> str:
        return f"segments={len(self.segments)} roads={len(self.road_props_by_id)} path={self.road_geojson_path}"

    def edge_metadata(self, edge_id: str) -> dict[str, Any] | None:
        road_id = self._road_id_from_edge_id(edge_id)
        if road_id is None:
            return None
        meta = self.road_props_by_id.get(road_id)
        return dict(meta) if meta is not None else None

    def _candidate_segment_indexes(self, x_m: float, y_m: float, radius_m: float) -> list[int]:
        cx, cy = self._cell_coords(x_m, y_m)
        cell_radius = max(1, int(math.ceil(radius_m / self.cell_size_m)))
        indexes: list[int] = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                indexes.extend(self.grid.get((cx + dx, cy + dy), []))
        return indexes

    def project_point(self, position_enu_m: Sequence[float], *, radius_m: float) -> RoadGeometryProjectionResult | None:
        query_x = float(position_enu_m[0])
        query_y = float(position_enu_m[1])
        candidate_indexes = self._candidate_segment_indexes(query_x, query_y, radius_m)
        if not candidate_indexes:
            return None

        best_result: RoadGeometryProjectionResult | None = None
        for segment_index in candidate_indexes:
            segment = self.segments[segment_index]
            start_x, start_y, start_z = segment.start_enu_m
            end_x, end_y, end_z = segment.end_enu_m
            seg_dx = end_x - start_x
            seg_dy = end_y - start_y
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len_sq <= 1e-6:
                continue
            t = clamp(((query_x - start_x) * seg_dx + (query_y - start_y) * seg_dy) / seg_len_sq, 0.0, 1.0)
            proj_x = start_x + t * seg_dx
            proj_y = start_y + t * seg_dy
            proj_z = start_z + t * (end_z - start_z)
            distance_m = math.hypot(query_x - proj_x, query_y - proj_y)
            if distance_m > radius_m:
                continue
            seg_len = math.sqrt(seg_len_sq)
            left_normal_xy = (-seg_dy / seg_len, seg_dx / seg_len)
            signed_lateral_m = (query_x - proj_x) * left_normal_xy[0] + (query_y - proj_y) * left_normal_xy[1]
            result = RoadGeometryProjectionResult(
                road_id=segment.road_id,
                lanes=segment.lanes,
                width_m=segment.width_m,
                projected_enu_m=(proj_x, proj_y, proj_z),
                heading_deg=heading_deg_from_vector(seg_dx, seg_dy),
                signed_lateral_m=signed_lateral_m,
                distance_m=distance_m,
                left_normal_xy=left_normal_xy,
            )
            if best_result is None or result.distance_m < best_result.distance_m:
                best_result = result
        return best_result


class TrafficBundleRoadSnapper:
    def __init__(self, root_config: dict[str, Any], resolve_path: Any) -> None:
        config = dict(root_config.get("road_topology_snap") or {})
        self.enabled = bool(config.get("enabled", False))
        self.categories = {
            str(value).strip().lower()
            for value in (config.get("categories") or ["vehicle"])
            if str(value).strip()
        }
        self.use_sample_z = bool(config.get("use_sample_z", True))
        self.use_sample_yaw = bool(config.get("use_sample_yaw", True))
        self.cell_size_m = max(1.0, float(config.get("cell_size_m", 20.0)))
        self.search_radius_m = max(1.0, float(config.get("search_radius_m", 35.0)))
        self.fallback_radius_m = max(self.search_radius_m, float(config.get("fallback_radius_m", 90.0)))
        self.heading_weight_m_per_deg = max(0.0, float(config.get("heading_weight_m_per_deg", 0.06)))
        self.same_edge_bonus_m = max(0.0, float(config.get("same_edge_bonus_m", 1.5)))
        self.same_lane_bonus_m = max(0.0, float(config.get("same_lane_bonus_m", 3.0)))
        self.lane_center_samples_path = resolve_path(
            config.get("lane_center_samples_path") or "Saved/SUMO/traffic_bundle/lane_center_samples.csv"
        )

        self.samples: list[RoadLaneSample] = []
        self.grid: dict[tuple[int, int], list[int]] = {}
        self.last_match_by_entity: dict[str, RoadTopologySnapResult] = {}

        if self.enabled:
            self._load_samples()

    def _load_samples(self) -> None:
        if not self.lane_center_samples_path.exists():
            raise RuntimeError(
                f"road_topology_snap is enabled but lane_center_samples.csv was not found: {self.lane_center_samples_path}"
            )

        with self.lane_center_samples_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    sample = RoadLaneSample(
                        edge_id=str(row.get("edge_id") or ""),
                        lane_id=str(row.get("lane_id") or ""),
                        s_m=float(row.get("s_m") or 0.0),
                        position_enu_m=(
                            float(row.get("x_m") or 0.0),
                            float(row.get("y_m") or 0.0),
                            float(row.get("z_m") or 0.0),
                        ),
                        yaw_deg=float(row.get("yaw_deg") or 0.0),
                    )
                except Exception:
                    continue
                index = len(self.samples)
                self.samples.append(sample)
                self.grid.setdefault(self._cell_coords(sample.position_enu_m[0], sample.position_enu_m[1]), []).append(index)

        if not self.samples:
            raise RuntimeError(
                f"road_topology_snap is enabled but no valid samples were parsed from {self.lane_center_samples_path}"
            )

    def describe(self) -> str:
        return (
            f"enabled={self.enabled} "
            f"categories={sorted(self.categories)} "
            f"samples={len(self.samples)} "
            f"path={self.lane_center_samples_path}"
        )

    def clear_state(self) -> None:
        self.last_match_by_entity.clear()

    def should_snap(self, entity: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        category = str(entity.get("entity_category") or "").strip().lower()
        return category in self.categories

    def _cell_coords(self, x_m: float, y_m: float) -> tuple[int, int]:
        return int(math.floor(x_m / self.cell_size_m)), int(math.floor(y_m / self.cell_size_m))

    def _candidate_indexes(self, x_m: float, y_m: float, radius_m: float) -> list[int]:
        cx, cy = self._cell_coords(x_m, y_m)
        cell_radius = max(1, int(math.ceil(radius_m / self.cell_size_m)))
        candidates: list[int] = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                candidates.extend(self.grid.get((cx + dx, cy + dy), []))
        return candidates

    @staticmethod
    def _heading_error_deg(a_deg: float, b_deg: float) -> float:
        delta = (a_deg - b_deg + 180.0) % 360.0 - 180.0
        return abs(delta)

    def snap(
        self,
        *,
        entity_id: str,
        position_enu_m: Sequence[float],
        rotation_deg: dict[str, Any],
    ) -> RoadTopologySnapResult | None:
        if not self.enabled:
            return None

        query_x = float(position_enu_m[0])
        query_y = float(position_enu_m[1])
        query_yaw_deg = float(rotation_deg.get("yaw_deg", 0.0))
        previous = self.last_match_by_entity.get(entity_id)
        candidate_indexes = self._candidate_indexes(query_x, query_y, self.search_radius_m)
        if not candidate_indexes:
            candidate_indexes = self._candidate_indexes(query_x, query_y, self.fallback_radius_m)
        if not candidate_indexes:
            return None

        best_score: float | None = None
        best_result: RoadTopologySnapResult | None = None
        for sample_index in candidate_indexes:
            sample = self.samples[sample_index]
            dx = sample.position_enu_m[0] - query_x
            dy = sample.position_enu_m[1] - query_y
            distance_m = math.hypot(dx, dy)
            if distance_m > self.fallback_radius_m:
                continue
            heading_error_deg = self._heading_error_deg(sample.yaw_deg, query_yaw_deg)
            score = distance_m + self.heading_weight_m_per_deg * heading_error_deg
            if previous is not None:
                if sample.lane_id == previous.lane_id:
                    score -= self.same_lane_bonus_m
                elif sample.edge_id == previous.edge_id:
                    score -= self.same_edge_bonus_m
            if best_score is not None and score >= best_score:
                continue
            best_score = score
            best_result = RoadTopologySnapResult(
                edge_id=sample.edge_id,
                lane_id=sample.lane_id,
                s_m=sample.s_m,
                position_enu_m=sample.position_enu_m,
                yaw_deg=sample.yaw_deg,
                distance_m=distance_m,
                heading_error_deg=heading_error_deg,
            )

        if best_result is not None:
            self.last_match_by_entity[entity_id] = best_result
        return best_result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def safe_frame_id(frame_id: str) -> str:
    return safe_name(frame_id)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def left_normal_xy_from_yaw_deg(yaw_deg: float) -> tuple[float, float]:
    yaw_rad = math.radians(yaw_deg)
    return -math.sin(yaw_rad), math.cos(yaw_rad)


def heading_deg_from_vector(x_m: float, y_m: float, *, default_deg: float = 0.0) -> float:
    if abs(x_m) <= 1e-6 and abs(y_m) <= 1e-6:
        return default_deg
    return math.degrees(math.atan2(y_m, x_m))


def rotation_dict_from_truth(entity: dict[str, Any]) -> dict[str, float]:
    rotation = ((entity.get("truth_pose") or {}).get("rotation_deg") or {})
    return {
        "pitch_deg": float(rotation.get("pitch_deg", rotation.get("pitch", 0.0))),
        "yaw_deg": float(rotation.get("yaw_deg", rotation.get("yaw", 0.0))),
        "roll_deg": float(rotation.get("roll_deg", rotation.get("roll", 0.0))),
    }


def position_enu_from_truth(entity: dict[str, Any]) -> list[float]:
    pose = entity.get("truth_pose") or {}
    position = pose.get("position_enu_m") or pose.get("position_m") or [0.0, 0.0, 0.0]
    return [float(position[0]), float(position[1]), float(position[2] if len(position) > 2 else 0.0)]


def velocity_enu_from_truth(entity: dict[str, Any]) -> list[float]:
    pose = entity.get("truth_pose") or {}
    velocity = pose.get("velocity_enu_mps") or [0.0, 0.0, 0.0]
    return [float(velocity[0]), float(velocity[1]), float(velocity[2] if len(velocity) > 2 else 0.0)]


def distance_m(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def truth_submission_state(entity: dict[str, Any]) -> str:
    return str(((entity.get("render_presence") or {}).get("submission_state")) or "")


def visibility_state(entity: dict[str, Any]) -> str:
    return str(((entity.get("render_presence") or {}).get("visibility_state")) or "visible")


def normalize_activity_type(entity: dict[str, Any]) -> str:
    annotations = entity.get("annotations") or {}
    activity = ((annotations.get("state_facets") or {}).get("activity") or {}).get("activity_type")
    if not activity:
        activity = annotations.get("activity_type")
    return str(activity or "idle").strip().lower()


def is_fallen_activity(entity: dict[str, Any]) -> bool:
    annotations = entity.get("annotations") or {}
    activity = ((annotations.get("state_facets") or {}).get("activity") or {})
    posture = str(activity.get("posture") or "").strip().lower()
    animation_hint = str(activity.get("animation_hint") or "").strip().lower()
    if posture == "fallen":
        return True
    if animation_hint == "pedestrian_fall":
        return True
    return normalize_activity_type(entity) in {"medical_incident", "fall_flat"}


def build_airsim_pose(airsim_module: Any, position_enu_m: Sequence[float], rotation_deg: dict[str, Any]) -> Any:
    pitch_rad = math.radians(float(rotation_deg.get("pitch_deg", 0.0)))
    roll_rad = math.radians(float(rotation_deg.get("roll_deg", 0.0)))
    yaw_rad = math.radians(float(rotation_deg.get("yaw_deg", 0.0)))

    if hasattr(airsim_module, "to_quaternion"):
        orientation = airsim_module.to_quaternion(pitch_rad, roll_rad, yaw_rad)
    elif hasattr(airsim_module, "utils") and hasattr(airsim_module.utils, "euler_to_quaternion"):
        # cosysairsim 3.3 exposes utils.euler_to_quaternion(roll, pitch, yaw)
        # instead of the older top-level to_quaternion helper.
        orientation = airsim_module.utils.euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
    else:
        raise RuntimeError(
            "The active cosysairsim package does not expose to_quaternion or utils.euler_to_quaternion."
        )

    return airsim_module.Pose(
        airsim_module.Vector3r(float(position_enu_m[0]), float(position_enu_m[1]), float(-position_enu_m[2])),
        orientation,
    )


class TemplateResolver:
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._rules = list(config.get("rules", []))
        self._category_defaults = dict(config.get("category_defaults", {}))
        self._ped_cfg = dict(config.get("pedestrian_defaults", {}))
        self._uav_cfg = dict(config.get("uav_defaults", {}))

    def _rule_matches(self, rule: dict[str, Any], entity: dict[str, Any], roster_entry: dict[str, Any]) -> bool:
        match = dict(rule.get("match", {}))
        if not match:
            return True

        tags = set(entity.get("tags") or roster_entry.get("tags") or [])
        checks: list[tuple[Any, Any]] = [
            (entity.get("entity_id"), match.get("entity_id")),
            (entity.get("entity_kind") or entity.get("entity_type"), match.get("entity_kind")),
            (entity.get("entity_type") or entity.get("entity_kind"), match.get("entity_type")),
            (entity.get("entity_category"), match.get("entity_category")),
            (entity.get("site_id") or roster_entry.get("site_id"), match.get("site_id")),
            (entity.get("proxy_template_id") or roster_entry.get("proxy_template_id"), match.get("proxy_template_id")),
        ]
        for actual, expected in checks:
            if expected is not None and str(actual or "") != str(expected):
                return False
        tags_any = match.get("tags_any")
        if tags_any and not any(str(tag) in tags for tag in tags_any):
            return False
        return True

    def _choose_ped_variant(self, entity_id: str) -> str:
        variants = list(self._ped_cfg.get("variant_cycle", []))
        default_variant = str(self._ped_cfg.get("variant_id", "adult_male_commuter"))
        if not variants:
            return default_variant
        checksum = sum(ord(ch) for ch in entity_id)
        return str(variants[checksum % len(variants)])

    @staticmethod
    def _explicit_ped_variant(entity: dict[str, Any], roster_entry: dict[str, Any]) -> str:
        for source in (entity, roster_entry):
            candidate = str((source or {}).get("variant_id") or "").strip()
            if candidate:
                return candidate
        return ""

    def resolve(self, entity: dict[str, Any], roster_entry: dict[str, Any]) -> dict[str, Any]:
        for rule in self._rules:
            if self._rule_matches(rule, entity, roster_entry):
                resolved = dict(rule)
                resolved.pop("match", None)
                resolved["resolved_via"] = "rule"
                if resolved.get("mode") == "pedestrian_managed" and not resolved.get("variant_id"):
                    resolved["variant_id"] = self._explicit_ped_variant(entity, roster_entry) or self._choose_ped_variant(
                        str(entity.get("entity_id", ""))
                    )
                if resolved.get("mode") == "runtime_multirotor" and not resolved.get("vehicle_name"):
                    resolved["vehicle_name"] = str(entity.get("entity_id", ""))
                return resolved

        category = str(entity.get("entity_category") or roster_entry.get("entity_category") or "")
        resolved = dict(self._category_defaults.get(category, {}))
        if not resolved:
            resolved = {"mode": "metadata_only"}
        resolved["resolved_via"] = "category_default"
        if resolved.get("mode") == "pedestrian_managed":
            merged = dict(self._ped_cfg)
            merged.update(resolved)
            merged["variant_id"] = (
                merged.get("variant_id")
                or self._explicit_ped_variant(entity, roster_entry)
                or self._choose_ped_variant(str(entity.get("entity_id", "")))
            )
            merged["resolved_via"] = "pedestrian_default"
            return merged
        if resolved.get("mode") == "runtime_multirotor":
            merged = dict(self._uav_cfg)
            merged.update(resolved)
            merged["vehicle_name"] = str(merged.get("vehicle_name") or entity.get("entity_id", ""))
            merged["resolved_via"] = "uav_default"
            return merged
        return resolved


class EpisodeRenderHost:
    def __init__(self, config_path: Path, args: argparse.Namespace) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        self.args = args
        self.project_root = project_root_from(self.config_path)

        self.episode_dir = self._resolve_path(self.config.get("episode_dir"))
        self.output_dir = self._resolve_path(args.output_dir or self.config.get("output_dir"))
        self.template_resolver_path = self._resolve_path(self.config.get("template_resolver_path"))
        self.capture_presets_path = self._resolve_path(self.config.get("capture_presets_path"))

        self.episode_manifest = load_json(self.episode_dir / "episode_manifest.json")
        roster_root = load_json(self.episode_dir / "global_entity_roster.json")
        self.global_roster = list(roster_root.get("entities", []))
        self.roster_by_id = {str(entity["entity_id"]): entity for entity in self.global_roster}

        self.truth_frames = load_jsonl(self.episode_dir / "truth_frames.jsonl")
        self.frames_by_tick = {int(frame["tick"]): frame for frame in self.truth_frames}
        self.sorted_ticks = sorted(self.frames_by_tick)
        if not self.sorted_ticks:
            raise RuntimeError(f"No truth frames found in {self.episode_dir}")

        self.weather_rows = load_jsonl(self.episode_dir / "weather_meta.jsonl")
        self.weather_by_tick = {int(row["tick"]): row for row in self.weather_rows}
        self.weather_ticks = sorted(self.weather_by_tick)

        first_frame = self.frames_by_tick[self.sorted_ticks[0]]
        self.episode_id = str(first_frame.get("episode_id") or self.episode_manifest.get("episode_id") or "")
        self.map_id = str(args.map_id or self.config.get("map_id") or first_frame.get("map_id") or "")
        if not self.map_id:
            raise RuntimeError("Unable to determine map_id from config or truth frames.")
        time_range = dict(self.episode_manifest.get("time_range") or {})
        self.episode_duration_s = float(
            time_range.get("sim_time_end")
            or first_frame.get("sim_time_s")
            or self.frames_by_tick[self.sorted_ticks[-1]].get("sim_time_s")
            or 0.0
        )

        self.scenario_plan = load_json(self.episode_dir / "scenario_plan.json")
        self.capture_presets = load_json(self.capture_presets_path)
        self.map_package = resolve_map_package(self.project_root, self.map_id)
        self.weather_service = WeatherService.from_profiles_path(self.map_package.weather_profiles_path)
        self.template_resolver_doc = load_json(self.template_resolver_path)
        self.template_resolver = TemplateResolver(self.template_resolver_doc)
        self.coordinate_transform = CoordinateTransform.from_config(self.config)
        self.truth_frame_coordinate_space = str(self.config.get("truth_frame_coordinate_space", "local_enu") or "local_enu").strip().lower()
        if self.truth_frame_coordinate_space not in {"local_enu", "map_enu", "world_enu", "transformed_enu"}:
            raise RuntimeError(
                "truth_frame_coordinate_space must be one of: local_enu, map_enu, world_enu, transformed_enu"
            )
        self.road_topology_snapper = TrafficBundleRoadSnapper(self.config, self._resolve_path)
        self.ground_reference_cfg = dict(self.config.get("ground_reference") or {})
        self.entity_rotation_offset_cfg = dict(self.config.get("entity_rotation_offsets_deg") or {})
        self.road_geometry_cfg = dict(self.config.get("road_geometry") or {})
        self.vehicle_lane_offsets_cfg = dict(self.config.get("vehicle_lane_offsets") or {})
        self.pedestrian_roadside_cfg = dict(self.config.get("pedestrian_roadside_projection") or {})
        self.entity_overlap_cfg = dict(self.config.get("entity_overlap_filter") or {})
        default_pie_cleanup_cfg: dict[str, Any] = {
            "enabled": True,
            "destroy_actor_class_names": [
                "BP_AW_UAV_Inspection_Quad_01_C",
                "BP_AW_Pedestrian_CityOps_01_C",
            ],
            "destroy_actor_name_prefixes": [
                "drone_demo_",
            ],
            "preserve_actor_names": [
                "CaptureUAV_0",
            ],
        }
        configured_pie_cleanup = dict(self.config.get("pie_scene_cleanup") or {})
        for key, value in configured_pie_cleanup.items():
            default_pie_cleanup_cfg[key] = value
        self.pie_scene_cleanup_cfg = default_pie_cleanup_cfg
        self.road_geometry: RoadGeometryIndex | None = None
        road_geometry_enabled = bool(self.vehicle_lane_offsets_cfg.get("enabled", False)) or bool(
            self.pedestrian_roadside_cfg.get("enabled", False)
        )
        if road_geometry_enabled:
            road_geojson_path = self._resolve_path(
                self.road_geometry_cfg.get("road_geojson_path") or f"Content/Maps/{self.map_id}/road/road.geojson"
            )
            lane_center_samples_path = self._resolve_path(
                self.road_geometry_cfg.get("lane_center_samples_path")
                or (self.config.get("road_topology_snap") or {}).get("lane_center_samples_path")
                or "Saved/SUMO/traffic_bundle/lane_center_samples.csv"
            )
            self.road_geometry = RoadGeometryIndex(
                road_geojson_path,
                lane_center_samples_path=lane_center_samples_path,
                cell_size_m=float(self.road_geometry_cfg.get("cell_size_m", 25.0)),
            )
        self.vehicle_lane_slot_by_entity = self._build_vehicle_lane_slot_index()

        self.world_origin_cm = [0.0, 0.0, 0.0]
        self.local_frame = "UNKNOWN"
        self.geo_reference: dict[str, Any] = {}
        self.feedback_watermark_tick = -1
        self.last_weather_signature: dict[str, Any] | None = None
        self.ground_projection_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self.ground_projection_warning_keys: set[str] = set()
        self.capture_warmup_complete = False
        self.capture_orchestrator = CaptureOrchestrator()
        pedestrian_pose_cfg = dict(self.config.get("pedestrian_pose") or {})
        self.pedestrian_pose_service = PedestrianPoseService(
            min_speed_mps=float(pedestrian_pose_cfg.get("min_speed_mps", 0.15)),
            locomotion_yaw_offset_deg=float(pedestrian_pose_cfg.get("locomotion_yaw_offset_deg", 180.0)),
        )
        self.uav_execution_service = UavExecutionService(
            arrival_tolerance_m=float((self.config.get("runtime_uav") or {}).get("arrival_tolerance_m", 1.5)),
            hover_before_capture=False,
        )

        self.scene_active_ids: set[str] = set()
        self.ped_active_ids: set[str] = set()
        self.ped_last_activity: dict[str, str] = {}
        self.ped_last_variant: dict[str, str] = {}
        self.uav_active_by_entity: dict[str, str] = {}
        self.uav_last_command_target_by_entity: dict[str, list[float]] = {}
        self.event_controlled_entity_ids: set[str] = set()

        self.all_scene_sync_ids, self.all_ped_ids, self.all_uav_vehicle_names = self._discover_entity_modes()
        self.batch_plans = self._build_batches()

        self.airsim: Any | None = None
        self.client: AeroSimClient | None = None
        self.fixed_world_capture_hook: FixedWorldCaptureEditorHook | None = None
        self.ground_camera_asset_ids: dict[tuple[str, str], str] = {}
        self.prepared_capture_output_dirs: set[Path] = set()
        self.crowd_group_ids: set[str] = set()
        self.runtime_uav_direct_rpc_enabled = True
        self.runtime_uav_direct_rpc_disable_reason = ""
        self.runtime_uav_debug_cfg = dict(self.config.get("runtime_uav_debug") or {})
        self.runtime_uav_debug_entity_ids = {
            str(value).strip()
            for value in (self.runtime_uav_debug_cfg.get("entity_ids") or [])
            if str(value).strip()
        }
        role_filters = {
            str(value).strip().lower()
            for value in (getattr(self.args, "camera_role", None) or [])
            if str(value).strip()
        }
        if "all" in role_filters:
            role_filters.clear()
        self.capture_role_filters = role_filters
        self.capture_camera_filters = {
            str(value).strip()
            for value in (getattr(self.args, "camera_id", None) or [])
            if str(value).strip()
        }
        self.capture_modality_filters = {
            str(value).strip()
            for value in (getattr(self.args, "modality", None) or [])
            if str(value).strip()
        }
        self.uav_capture_backend = str(getattr(self.args, "uav_capture_backend", "airsim_native") or "airsim_native").strip().lower()
        self.segmentation_backend = str(
            getattr(self.args, "segmentation_backend", "ue_custom_stencil") or "ue_custom_stencil"
        ).strip().lower()
        semantic_rules_arg = str(getattr(self.args, "semantic_rules_path", "") or "").strip()
        self.semantic_rules_path = (
            self._resolve_path(semantic_rules_arg)
            if semantic_rules_arg
            else PROJECT_ROOT / "Config" / "LowAltitude" / "semantic_stencil_rules.json"
        )
        self.semantic_class_by_id = self._load_semantic_class_by_id(self.semantic_rules_path)
        self.airsim_capture_vehicle = str(getattr(self.args, "airsim_capture_vehicle", "CaptureUAV_0") or "CaptureUAV_0").strip()
        self.requested_airsim_capture_entity = str(getattr(self.args, "airsim_capture_entity", "") or "").strip()
        self.requested_capture_view_id = str(getattr(self.args, "capture_view_id", "") or "").strip()
        self.active_airsim_capture_entity_id = ""
        self.active_capture_view_id = ""
        self.airsim_capture_vehicle_ready = False
        self.airsim_capture_ned_origin_world_cm: list[float] | None = None
        self.airsim_segmentation_ready = False
        self.airsim_segmentation_registry_payload: dict[str, Any] | None = None
        self.event_weather_overlay: dict[str, Any] = {}
        self.event_scene_setup: dict[str, Any] = {}
        self.event_entity_assets: dict[str, str] = {}
        self.event_entity_initial_positions: dict[str, list[float]] = {}
        self.event_pedestrian_activity_state: dict[str, str] = {}
        self.event_capture_bbox_enu_m: list[float] | None = None

        # Optional event script interpreter
        event_script_path_str = self.config.get("event_script_path")
        if event_script_path_str:
            script_path = self._resolve_path(event_script_path_str)
            scene_setup_path = script_path.with_name("scene_setup.json")
            if scene_setup_path.exists():
                self.event_scene_setup = json.loads(scene_setup_path.read_text(encoding="utf-8"))
                for entity in self.event_scene_setup.get("entities") or []:
                    entity_id = str(entity.get("entity_id") or entity.get("instance_id") or "")
                    if not entity_id:
                        continue
                    self.event_entity_assets[entity_id] = str(entity.get("logical_asset_id") or "")
                    placement = dict(entity.get("placement") or {})
                    position = (
                        placement.get("resolved_position_enu_m")
                        or placement.get("position_enu_m")
                        or placement.get("center_enu_m")
                    )
                    if isinstance(position, list) and len(position) >= 2:
                        self.event_entity_initial_positions[entity_id] = [
                            float(position[0]),
                            float(position[1]),
                            float(position[2] if len(position) > 2 else 0.0),
                        ]
            script_params = dict(self.config.get("event_script_parameters") or {})
            self.event_interpreter = EventScriptInterpreter(
                script_path, parameters=script_params, episode_id=self.episode_id
            )
            self._register_event_action_handlers()
        else:
            self.event_interpreter = None

    def _resolve_path(self, value: Any) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _discover_entity_modes(self) -> tuple[set[str], set[str], set[str]]:
        scene_ids: set[str] = set()
        ped_ids: set[str] = set()
        uav_vehicle_names: set[str] = set()
        for entity in self.global_roster:
            entity_id = str(entity.get("entity_id", ""))
            resolved = self.template_resolver.resolve(entity, entity)
            mode = str(resolved.get("mode", "metadata_only"))
            if mode == "scene_sync":
                scene_ids.add(entity_id)
            elif mode == "pedestrian_managed":
                ped_ids.add(entity_id)
            elif mode == "runtime_multirotor":
                uav_vehicle_names.add(str(resolved.get("vehicle_name") or entity_id))
        return scene_ids, ped_ids, uav_vehicle_names

    @staticmethod
    def _cluster_scalar_values(values: Sequence[float], threshold_m: float) -> list[float]:
        if not values:
            return []
        sorted_values = sorted(float(value) for value in values)
        clusters: list[list[float]] = [[sorted_values[0]]]
        for value in sorted_values[1:]:
            if abs(value - clusters[-1][-1]) <= threshold_m:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        return [sum(cluster) / len(cluster) for cluster in clusters]

    def _vehicle_lane_group_key(self, entity: dict[str, Any]) -> tuple[str, str] | None:
        if str(entity.get("entity_category") or "").strip().lower() != "vehicle":
            return None
        annotations = dict(entity.get("annotations") or {})
        group_id = str(annotations.get("approach_id") or annotations.get("lane_id") or "")
        if not group_id:
            return None
        return str(entity.get("site_id", "")), group_id

    def _build_vehicle_lane_slot_index(self) -> dict[str, dict[str, Any]]:
        if not bool(self.vehicle_lane_offsets_cfg.get("enabled", False)):
            return {}
        threshold_m = max(0.5, float(self.vehicle_lane_offsets_cfg.get("cluster_threshold_m", 2.25)))
        duplicate_threshold_m = max(0.1, float(self.vehicle_lane_offsets_cfg.get("duplicate_threshold_m", 0.75)))
        first_frame = self.frames_by_tick[self.sorted_ticks[0]]
        grouped: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
        site_rows: dict[str, list[tuple[str, float, float]]] = {}
        for entity in first_frame.get("entities", []):
            group_key = self._vehicle_lane_group_key(entity)
            if group_key is None:
                continue
            raw_position = position_enu_from_truth(entity)
            row = (
                str(entity.get("entity_id", "")),
                float(raw_position[0]),
                float(raw_position[1]),
            )
            grouped.setdefault(group_key, []).append(row)
            site_rows.setdefault(str(entity.get("site_id", "")), []).append(row)

        lane_slots: dict[str, dict[str, Any]] = {}
        site_axis_by_site: dict[str, str] = {}
        site_centers_by_site: dict[str, list[float]] = {}
        site_centerline_value_by_site: dict[str, float] = {}
        side_rank_by_site_and_band: dict[tuple[str, int], tuple[int, int, int]] = {}

        for site_id, rows in site_rows.items():
            xs = [row[1] for row in rows]
            ys = [row[2] for row in rows]
            use_y_axis = (max(ys) - min(ys)) <= (max(xs) - min(xs))
            axis = "y" if use_y_axis else "x"
            centers = self._cluster_scalar_values([row[2] if use_y_axis else row[1] for row in rows], threshold_m)
            if not centers:
                continue
            site_axis_by_site[site_id] = axis
            site_centers_by_site[site_id] = list(centers)
            if len(centers) % 2 == 0:
                centerline_value = 0.5 * (centers[len(centers) // 2 - 1] + centers[len(centers) // 2])
            else:
                centerline_value = centers[len(centers) // 2]
            site_centerline_value_by_site[site_id] = centerline_value

            positive_bands = [index for index, value in enumerate(centers) if value < centerline_value - 1e-6]
            negative_bands = [index for index, value in enumerate(centers) if value > centerline_value + 1e-6]
            center_bands = [index for index, value in enumerate(centers) if abs(value - centerline_value) <= 1e-6]

            positive_bands.sort(key=lambda index: abs(centers[index] - centerline_value))
            negative_bands.sort(key=lambda index: abs(centers[index] - centerline_value))
            for rank, band_index in enumerate(positive_bands):
                side_rank_by_site_and_band[(site_id, band_index)] = (1, rank, len(positive_bands))
            for rank, band_index in enumerate(negative_bands):
                side_rank_by_site_and_band[(site_id, band_index)] = (-1, rank, len(negative_bands))
            for rank, band_index in enumerate(center_bands):
                side_rank_by_site_and_band[(site_id, band_index)] = (0, rank, len(center_bands))

        for group_key, rows in grouped.items():
            site_id = str(group_key[0])
            axis = site_axis_by_site.get(site_id)
            centers = site_centers_by_site.get(site_id)
            centerline_value = site_centerline_value_by_site.get(site_id)
            if not axis or not centers or centerline_value is None:
                continue
            use_y_axis = axis == "y"
            for entity_id, x_m, y_m in rows:
                lane_value = y_m if use_y_axis else x_m
                band_index = min(range(len(centers)), key=lambda index: abs(centers[index] - lane_value))
                side_sign, side_rank, side_band_count = side_rank_by_site_and_band.get((site_id, band_index), (0, 0, 1))
                lane_slots[entity_id] = {
                    "group_key": list(group_key),
                    "site_id": site_id,
                    "axis": axis,
                    "band_index": band_index,
                    "band_count": len(centers),
                    "source_lane_value_m": lane_value,
                    "band_centers_m": list(centers),
                    "source_longitudinal_value_m": x_m if use_y_axis else y_m,
                    "site_centerline_value_m": centerline_value,
                    "side_sign": side_sign,
                    "side_rank": side_rank,
                    "side_band_count": side_band_count,
                }
            for band_index in range(len(centers)):
                band_rows = [
                    (entity_id, float(slot["source_longitudinal_value_m"]))
                    for entity_id, slot in lane_slots.items()
                    if list(group_key) == list(slot.get("group_key") or []) and int(slot.get("band_index", -1)) == band_index
                ]
                band_rows.sort(key=lambda item: item[1])
                duplicate_groups: list[list[tuple[str, float]]] = []
                for item in band_rows:
                    if not duplicate_groups or abs(item[1] - duplicate_groups[-1][-1][1]) > duplicate_threshold_m:
                        duplicate_groups.append([item])
                    else:
                        duplicate_groups[-1].append(item)
                for duplicate_group in duplicate_groups:
                    duplicate_count = len(duplicate_group)
                    for duplicate_index, (entity_id, _longitudinal_value_m) in enumerate(duplicate_group):
                        lane_slots[entity_id]["duplicate_index"] = duplicate_index
                        lane_slots[entity_id]["duplicate_count"] = duplicate_count
        return lane_slots

    def _vehicle_lane_offset_details(
        self,
        entity: dict[str, Any],
        *,
        yaw_deg: float,
        edge_id: str,
    ) -> dict[str, Any] | None:
        if not bool(self.vehicle_lane_offsets_cfg.get("enabled", False)):
            return None
        entity_id = str(entity.get("entity_id", ""))
        slot = self.vehicle_lane_slot_by_entity.get(entity_id)
        if not slot:
            return None
        band_count = max(1, int(slot.get("band_count") or 1))
        lane_width_m = max(0.5, float(self.vehicle_lane_offsets_cfg.get("lane_width_m", 3.5)))
        road_meta = self.road_geometry.edge_metadata(edge_id) if self.road_geometry is not None else None
        road_lanes = max(1, int((road_meta or {}).get("lanes") or band_count or 1))
        road_width_m = float((road_meta or {}).get("width_m") or 0.0)
        if road_width_m <= 0.0:
            road_width_m = lane_width_m * float(road_lanes)
        lane_spacing_m = max(0.5, road_width_m / float(road_lanes))

        physical_lane_centers_m = [
            ((float(index) + 0.5) - 0.5 * float(road_lanes)) * lane_spacing_m
            for index in range(road_lanes)
        ]
        negative_lane_centers_m = sorted(
            [value for value in physical_lane_centers_m if value < -1e-6],
            key=lambda value: abs(value),
        )
        positive_lane_centers_m = sorted(
            [value for value in physical_lane_centers_m if value > 1e-6],
            key=lambda value: abs(value),
        )
        center_lane_centers_m = [value for value in physical_lane_centers_m if abs(value) <= 1e-6]

        side_sign = int(slot.get("side_sign") or 0)
        side_rank = max(0, int(slot.get("side_rank") or 0))
        side_band_count = max(1, int(slot.get("side_band_count") or 1))
        source_lane_value_m = float(slot.get("source_lane_value_m") or 0.0)
        site_centerline_value_m = float(slot.get("site_centerline_value_m") or 0.0)
        source_delta_from_centerline_m = source_lane_value_m - site_centerline_value_m

        selected_lane_index = -1
        lateral_offset_m = 0.0
        target_lane_centers_m: Sequence[float] = []

        def _mapped_lane_index(source_rank: int, *, source_count: int, target_count: int) -> int:
            if target_count <= 0:
                return -1
            if target_count == 1 or source_count <= 1:
                return 0
            mapped_index = int(round(float(source_rank) * float(target_count - 1) / float(source_count - 1)))
            return max(0, min(target_count - 1, mapped_index))

        def _pick_side_lane_center(centers: Sequence[float]) -> tuple[float, int]:
            if not centers:
                return 0.0, -1
            mapped_index = _mapped_lane_index(side_rank, source_count=side_band_count, target_count=len(centers))
            return float(centers[mapped_index]), mapped_index

        if side_sign < 0:
            target_lane_centers_m = negative_lane_centers_m
            lateral_offset_m, selected_lane_index = _pick_side_lane_center(negative_lane_centers_m)
        elif side_sign > 0:
            target_lane_centers_m = positive_lane_centers_m
            lateral_offset_m, selected_lane_index = _pick_side_lane_center(positive_lane_centers_m)
        elif center_lane_centers_m:
            target_lane_centers_m = center_lane_centers_m
            lateral_offset_m = float(center_lane_centers_m[0])
            selected_lane_index = 0
        elif source_delta_from_centerline_m < -1e-6:
            target_lane_centers_m = negative_lane_centers_m
            lateral_offset_m, selected_lane_index = _pick_side_lane_center(negative_lane_centers_m)
        elif source_delta_from_centerline_m > 1e-6:
            target_lane_centers_m = positive_lane_centers_m
            lateral_offset_m, selected_lane_index = _pick_side_lane_center(positive_lane_centers_m)

        duplicate_count = int(slot.get("duplicate_count") or 1)
        duplicate_index = int(slot.get("duplicate_index") or 0)
        queue_spacing_m = max(1.0, float(self.vehicle_lane_offsets_cfg.get("queue_spacing_m", 4.75)))
        centered_duplicate_index = float(duplicate_index) - 0.5 * float(max(0, duplicate_count - 1))
        longitudinal_offset_m = centered_duplicate_index * queue_spacing_m

        collapsed_band_count = 1
        collapsed_band_index = 0
        collapsed_band_queue_offset_m = 0.0
        if selected_lane_index >= 0 and len(target_lane_centers_m) > 0 and side_band_count > 1:
            collapsed_side_ranks = [
                rank
                for rank in range(side_band_count)
                if _mapped_lane_index(rank, source_count=side_band_count, target_count=len(target_lane_centers_m)) == selected_lane_index
            ]
            collapsed_band_count = max(1, len(collapsed_side_ranks))
            if collapsed_band_count > 1 and side_rank in collapsed_side_ranks:
                collapsed_band_index = collapsed_side_ranks.index(side_rank)
                centered_collapsed_band_index = float(collapsed_band_index) - 0.5 * float(collapsed_band_count - 1)
                collapsed_band_queue_offset_m = centered_collapsed_band_index * queue_spacing_m
                longitudinal_offset_m += collapsed_band_queue_offset_m

        if abs(lateral_offset_m) <= 1e-6 and abs(longitudinal_offset_m) <= 1e-6:
            return None
        left_normal_xy = left_normal_xy_from_yaw_deg(yaw_deg)
        forward_xy = (math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)))
        return {
            "lane_offset_m": lateral_offset_m,
            "longitudinal_offset_m": longitudinal_offset_m,
            "left_normal_xy": [left_normal_xy[0], left_normal_xy[1]],
            "forward_xy": [forward_xy[0], forward_xy[1]],
            "band_index": int(slot["band_index"]),
            "band_count": band_count,
            "lane_width_m": lane_width_m,
            "road_lanes": road_lanes,
            "road_width_m": road_width_m,
            "lane_spacing_m": lane_spacing_m,
            "side_sign": side_sign,
            "side_rank": side_rank,
            "side_band_count": side_band_count,
            "selected_lane_index": selected_lane_index,
            "collapsed_band_index": collapsed_band_index,
            "collapsed_band_count": collapsed_band_count,
            "collapsed_band_queue_offset_m": collapsed_band_queue_offset_m,
            "source_lane_axis": str(slot.get("axis") or ""),
            "source_lane_value_m": source_lane_value_m,
            "source_delta_from_centerline_m": source_delta_from_centerline_m,
            "duplicate_index": duplicate_index,
            "duplicate_count": duplicate_count,
            "queue_spacing_m": queue_spacing_m,
        }

    def _roadside_side_hint(self, entity: dict[str, Any], signed_lateral_m: float) -> float:
        entity_kind = str(entity.get("entity_kind") or "").strip().lower()
        entity_id = str(entity.get("entity_id") or "").strip().lower()
        if entity_kind == "traffic_light.signal":
            if entity_id.endswith("_ns"):
                return -1.0
            if entity_id.endswith("_ew"):
                return 1.0

        annotations = dict(entity.get("annotations") or {})
        path_id = str(annotations.get("path_id") or "").strip().lower()
        explicit_hints = dict(self.pedestrian_roadside_cfg.get("path_side_hints") or {})
        if path_id in explicit_hints:
            return 1.0 if float(explicit_hints[path_id]) >= 0.0 else -1.0
        if abs(signed_lateral_m) > 0.5:
            return 1.0 if signed_lateral_m >= 0.0 else -1.0
        if any(token in path_id for token in ("upper", "north", "east")):
            return 1.0
        if any(token in path_id for token in ("lower", "south", "west")):
            return -1.0
        checksum = sum(ord(ch) for ch in entity_id)
        return 1.0 if checksum % 2 == 0 else -1.0

    def _roadside_adjustment(
        self,
        entity: dict[str, Any],
        position_enu_m: Sequence[float],
    ) -> dict[str, Any] | None:
        entity_category = str(entity.get("entity_category") or "").strip().lower()
        entity_kind = str(entity.get("entity_kind") or "").strip().lower()
        if self.road_geometry is None:
            return None

        # Fallen pedestrians must stay at their truth position — do not project to roadside.
        if entity_category == "pedestrian" and is_fallen_activity(entity):
            return None

        adjustment_kind = ""
        keepout_margin_m = 0.0
        search_radius_m = 0.0
        if entity_category == "pedestrian" and bool(self.pedestrian_roadside_cfg.get("enabled", False)):
            adjustment_kind = "pedestrian_roadside_projection"
            search_radius_m = max(1.0, float(self.pedestrian_roadside_cfg.get("search_radius_m", 20.0)))
            keepout_margin_m = max(0.5, float(self.pedestrian_roadside_cfg.get("keepout_margin_m", 1.5)))
        elif entity_kind == "traffic_light.signal":
            adjustment_kind = "signal_roadside_projection"
            search_radius_m = max(1.0, float(self.pedestrian_roadside_cfg.get("search_radius_m", 20.0)))
            keepout_margin_m = max(0.5, float(self.pedestrian_roadside_cfg.get("keepout_margin_m", 1.5)))
        else:
            return None

        projection = self.road_geometry.project_point(position_enu_m, radius_m=search_radius_m)
        if projection is None:
            return None

        lane_based_clearance_m = max(0.5, projection.width_m / max(1, int(projection.lanes)))
        min_clearance_m = lane_based_clearance_m * 2.0
        clearance_m = max(1.0, 0.5 * projection.width_m + keepout_margin_m, min_clearance_m)
        side_sign = self._roadside_side_hint(entity, projection.signed_lateral_m)
        if abs(projection.signed_lateral_m) >= clearance_m and (
            projection.signed_lateral_m == 0.0 or (projection.signed_lateral_m > 0.0) == (side_sign > 0.0)
        ):
            return {
                "position_enu_m": [float(position_enu_m[0]), float(position_enu_m[1]), float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0)],
                "details": {
                    "kind": adjustment_kind,
                    **projection.to_dict(),
                    "clearance_m": clearance_m,
                    "minimum_two_lane_clearance_m": min_clearance_m,
                    "moved": False,
                },
            }

        target_x = projection.projected_enu_m[0] + projection.left_normal_xy[0] * clearance_m * side_sign
        target_y = projection.projected_enu_m[1] + projection.left_normal_xy[1] * clearance_m * side_sign
        return {
            "position_enu_m": [target_x, target_y, float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0)],
            "details": {
                "kind": adjustment_kind,
                **projection.to_dict(),
                "clearance_m": clearance_m,
                "minimum_two_lane_clearance_m": min_clearance_m,
                "moved": True,
                "target_side_sign": side_sign,
            },
        }

    def _build_batches(self) -> list[BatchPlan]:
        summary = dict(self.scenario_plan.get("compiled_plan_summary") or {})
        site_contracts = dict(summary.get("site_contracts", {}))
        strategy = dict(self.config.get("batch_strategy", {}))
        tick_window_size = int(strategy.get("tick_window_size") or 0)
        enabled_sites = set(strategy.get("sites") or site_contracts.keys() or ["default"])
        start_tick = int(self.args.start_tick if self.args.start_tick is not None else self.sorted_ticks[0])
        end_tick = int(self.args.end_tick if self.args.end_tick is not None else self.sorted_ticks[-1])
        selected_ticks = [tick for tick in self.sorted_ticks if start_tick <= tick <= end_tick]
        if not selected_ticks:
            raise RuntimeError(f"No truth frames available in requested tick range {start_tick}..{end_tick}")

        windows: list[list[int]] = []
        if tick_window_size > 0:
            for index in range(0, len(selected_ticks), tick_window_size):
                windows.append(selected_ticks[index : index + tick_window_size])
        else:
            windows.append(selected_ticks)

        if not site_contracts:
            site_contracts = {"default": {"roi_id": str(self.frames_by_tick[selected_ticks[0]].get("active_roi_id", ""))}}

        plans: list[BatchPlan] = []
        for site_id, contract in site_contracts.items():
            if site_id not in enabled_sites:
                continue
            if self.args.site and self.args.site != site_id:
                continue
            roi_id = str(contract.get("roi_id") or contract.get("suggested_roi_id") or "")
            for window_ticks in windows:
                batch_id = (
                    f"{safe_name(site_id)}__ticks_{window_ticks[0]:06d}_{window_ticks[-1]:06d}"
                    if tick_window_size > 0
                    else f"{safe_name(site_id)}"
                )
                plans.append(BatchPlan(batch_id=batch_id, site_id=str(site_id), roi_id=roi_id, tick_start=int(window_ticks[0]), tick_end=int(window_ticks[-1])))

        if self.args.batch_id:
            plans = [plan for plan in plans if plan.batch_id == self.args.batch_id]
        if self.args.max_batches > 0:
            plans = plans[: self.args.max_batches]
        if not plans:
            raise RuntimeError("No batch plans matched the current filters.")
        return plans

    def _import_airsim(self) -> Any:
        if self.airsim is not None:
            return self.airsim
        try:
            import cosysairsim as airsim  # type: ignore
        except Exception as exc:
            raise RuntimeError("cosysairsim import failed. Install it in the active Python environment.") from exc
        self.airsim = airsim
        return airsim

    def _retry(self, label: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        retries = int((self.config.get("timeouts") or {}).get("rpc_retry_count", 2))
        delay_s = float((self.config.get("timeouts") or {}).get("rpc_retry_delay_s", 1.0))
        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if attempt >= retries:
                    raise
                print(f"[EpisodeHost] {label} failed (attempt {attempt + 1}/{retries + 1}): {exc}")
                time.sleep(delay_s)
        raise RuntimeError(f"{label} failed without an exception")

    def _airsim_capture_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("airsim_capture") or {})

    def _airsim_capture_enabled(self) -> bool:
        return self.uav_capture_backend == "airsim_native" and self._capture_role_enabled("uav")

    def _airsim_capture_pose_tolerance_m(self) -> float:
        return max(0.05, float(self._airsim_capture_cfg().get("pose_tolerance_m", 2.0)))

    def _airsim_capture_park_pose(self) -> tuple[list[float], dict[str, float]]:
        cfg = self._airsim_capture_cfg()
        position = cfg.get("park_position_enu_m") or [0.0, 0.0, 20.0]
        rotation = cfg.get("park_rotation_deg") or {"pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0}
        return (
            [float(position[0]), float(position[1]), float(position[2] if len(position) > 2 else 20.0)],
            {
                "pitch_deg": float(rotation.get("pitch_deg", rotation.get("pitch", 0.0))),
                "yaw_deg": float(rotation.get("yaw_deg", rotation.get("yaw", 0.0))),
                "roll_deg": float(rotation.get("roll_deg", rotation.get("roll", 0.0))),
            },
        )

    def _ensure_airsim_capture_vehicle(self) -> None:
        if not self._airsim_capture_enabled():
            return
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        if self.airsim_capture_vehicle_ready:
            return
        try:
            settings_text = self._retry("getSettingsString", self.client.get_settings_string)
            settings = json.loads(settings_text)
        except Exception as exc:
            print(f"[EpisodeHost] AirSim settings inspection warning: {exc}")
            settings = {}
        view_mode = str(settings.get("ViewMode") or "").strip()
        if view_mode.lower() == "nodisplay":
            raise RuntimeError(
                "AirSim native image capture cannot run with ViewMode='NoDisplay'; "
                "CameraDirector disables world rendering and PIP camera capture updates in this mode. "
                "Change Huawei Share AirSim settings.json ViewMode to 'FlyWithMe' and re-enter PIE."
            )
        if not self.airsim_capture_vehicle:
            raise RuntimeError("--airsim-capture-vehicle cannot be empty in airsim_native mode.")
        park_position, park_rotation = self._airsim_capture_park_pose()
        vehicles = set(self._retry("list_vehicles", self.client.list_vehicles))
        if self.airsim_capture_vehicle not in vehicles:
            added = self._retry(
                "simAddVehicle",
                self.client.add_vehicle,
                self.airsim_capture_vehicle,
                vehicle_type="SimpleFlight",
                position_enu_m=park_position,
                rotation_deg=park_rotation,
            )
            if not added:
                raise RuntimeError(f"simAddVehicle failed for capture vehicle '{self.airsim_capture_vehicle}'.")
        self._retry("enableApiControl", self.client.enable_api_control, True, self.airsim_capture_vehicle)
        self._retry("armDisarm", self.client.arm_disarm, True, self.airsim_capture_vehicle)
        self._measure_airsim_capture_ned_origin_world_cm()
        self._pin_airsim_capture_vehicle(park_position, park_rotation, context="capture vehicle park")
        self.airsim_capture_vehicle_ready = True

    def _measure_airsim_capture_ned_origin_world_cm(self) -> list[float]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        self._retry(
            "simSetKinematicsNedOriginProbe",
            self.client.set_vehicle_pose_ned,
            self.airsim_capture_vehicle,
            position_ned_m=[0.0, 0.0, 0.0],
            rotation_deg={"pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0},
            ignore_collision=True,
        )
        probe, _ = self._probe_runtime_uav_actor(self.airsim_capture_vehicle)
        actor = dict(probe.get("actor") or {})
        world_cm = actor.get("position_world_cm")
        if not isinstance(world_cm, Sequence) or isinstance(world_cm, (str, bytes)) or len(world_cm) < 3:
            raise RuntimeError(
                f"Unable to measure AirSim NED origin for '{self.airsim_capture_vehicle}': actor probe={probe}"
            )
        self.airsim_capture_ned_origin_world_cm = [
            float(world_cm[0]),
            float(world_cm[1]),
            float(world_cm[2]),
        ]
        print(
            "[EpisodeHost] AirSim capture NED origin "
            f"vehicle={self.airsim_capture_vehicle} world_cm={self.airsim_capture_ned_origin_world_cm}"
        )
        return list(self.airsim_capture_ned_origin_world_cm)

    def _capture_anchor_world_cm(self, position_enu_m: Sequence[float], rotation_deg: dict[str, Any], *, context: str) -> list[float]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        target = [float(position_enu_m[0]), float(position_enu_m[1]), float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0)]
        rotation = {
            "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", 0.0))),
            "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", 0.0))),
            "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", 0.0))),
        }
        asset_id = safe_name(f"coord_anchor.{self.airsim_capture_vehicle}")
        payload = {
            "asset_id": asset_id,
            "entity_id": asset_id,
            "logical_asset_id": "semantic.asset_anchor",
            "proxy_template_id": "semantic.asset_anchor",
            "pose_enu_m": {"position_enu_m": target, "rotation_deg": rotation},
            "position_enu_m": target,
            "rotation_deg": rotation,
            "tags": ["coord_anchor", "airsim_capture", safe_name(context)],
        }
        response = self._retry("spawn_capture_coord_anchor", self.client.spawn_asset, payload, map_id=self.map_id)
        response_payload = dict(response.get("payload") or {})
        raw_world = response_payload.get("position_world_cm")
        if isinstance(raw_world, dict):
            world_cm = [raw_world.get("x"), raw_world.get("y"), raw_world.get("z")]
        else:
            world_cm = raw_world
        if not isinstance(world_cm, Sequence) or isinstance(world_cm, (str, bytes)) or len(world_cm) < 3:
            raise RuntimeError(
                f"Capture anchor did not return position_world_cm during {context}: {response}"
            )
        return [float(world_cm[0]), float(world_cm[1]), float(world_cm[2])]

    def _world_cm_to_airsim_ned_m(self, world_cm: Sequence[float]) -> list[float]:
        origin = self.airsim_capture_ned_origin_world_cm
        if not isinstance(origin, Sequence) or len(origin) < 3:
            origin = self._measure_airsim_capture_ned_origin_world_cm()
        return [
            (float(world_cm[0]) - float(origin[0])) / 100.0,
            (float(world_cm[1]) - float(origin[1])) / 100.0,
            -(float(world_cm[2]) - float(origin[2])) / 100.0,
        ]

    def _pin_airsim_capture_vehicle(
        self,
        position_enu_m: Sequence[float],
        rotation_deg: dict[str, Any],
        *,
        context: str,
    ) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        target = [float(position_enu_m[0]), float(position_enu_m[1]), float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0)]
        rotation = {
            "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", 0.0))),
            "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", 0.0))),
            "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", 0.0))),
        }
        target_world_cm = self._capture_anchor_world_cm(target, rotation, context=context)
        target_ned_m = self._world_cm_to_airsim_ned_m(target_world_cm)
        self._retry(
            "simSetKinematicsNed",
            self.client.set_vehicle_pose_ned,
            self.airsim_capture_vehicle,
            position_ned_m=target_ned_m,
            rotation_deg=rotation,
            ignore_collision=True,
        )
        pose = self._retry("simGetVehiclePoseNed", self.client.get_vehicle_pose_ned, self.airsim_capture_vehicle)
        actual_ned = list((pose or {}).get("position_ned_m") or [])
        error = distance_m(target_ned_m, actual_ned)
        if error is None or error > self._airsim_capture_pose_tolerance_m():
            raise RuntimeError(
                f"AirSim capture vehicle pose error during {context}: "
                f"target_ned_m={target_ned_m} actual_ned_m={actual_ned} error_m={error}"
            )
        return {
            "vehicle_name": self.airsim_capture_vehicle,
            "requested_position_enu_m": target,
            "requested_position_world_cm": target_world_cm,
            "requested_position_ned_m": target_ned_m,
            "requested_rotation_deg": rotation,
            "pose": pose,
            "pose_error_m": float(error),
            "capture_pose_mode": "semantic_anchor_world_cm_to_airsim_ned_simSetKinematics",
            "context": context,
        }

    def connect(self) -> None:
        self._import_airsim()
        rpc_timeout_s = float((self.config.get("timeouts") or {}).get("airsim_timeout_s", 120.0))
        self.client = AeroSimClient(host=self.args.host, port=self.args.port, timeout_value=rpc_timeout_s, auto_connect=True)
        caps = self._retry("describe_capabilities", self.client.describe_capabilities)
        operations = set((caps.get("payload") or {}).get("operations") or [])
        missing = sorted(REQUIRED_CAPABILITIES - operations)
        if missing:
            raise RuntimeError(f"Bridge is missing required capabilities: {', '.join(missing)}")

        ctx = self._retry("load_context", self.client.load_context, self.map_id)
        payload = dict(ctx.get("payload") or {})
        self.world_origin_cm = [float(value) for value in (payload.get("world_origin_cm") or [0.0, 0.0, 0.0])]
        self.local_frame = str(payload.get("local_frame") or "UNKNOWN")
        self.geo_reference = dict(payload.get("geo_reference") or {})
        self._validate_template_coverage()
        print(
            f"[EpisodeHost] Connected to {self.args.host}:{self.args.port} "
            f"map={self.map_id} local_frame={self.local_frame} world_origin_cm={self.world_origin_cm}"
        )
        print(
            f"[EpisodeHost] Coordinate transform {self.coordinate_transform.describe()} "
            f"truth_frame_coordinate_space={self.truth_frame_coordinate_space}"
        )
        print(f"[EpisodeHost] Road topology snap {self.road_topology_snapper.describe()}")
        if self.road_geometry is not None:
            print(f"[EpisodeHost] Road geometry {self.road_geometry.describe()}")
        if bool(self.vehicle_lane_offsets_cfg.get('enabled', False)):
            print(
                "[EpisodeHost] Vehicle lane offsets "
                f"lane_width_m={float(self.vehicle_lane_offsets_cfg.get('lane_width_m', 3.5))} "
                f"indexed_entities={len(self.vehicle_lane_slot_by_entity)}"
            )
        if bool(self.pedestrian_roadside_cfg.get('enabled', False)):
            print(
                "[EpisodeHost] Pedestrian roadside projection "
                f"search_radius_m={float(self.pedestrian_roadside_cfg.get('search_radius_m', 20.0))} "
                f"keepout_margin_m={float(self.pedestrian_roadside_cfg.get('keepout_margin_m', 1.5))}"
            )
        print(
            "[EpisodeHost] Ground reference "
            f"uav_ground_relative={bool(self.ground_reference_cfg.get('uav_ground_relative', False))} "
            f"ground_camera_ground_relative={bool(self.ground_reference_cfg.get('ground_camera_ground_relative', False))}"
        )
        if not bool(getattr(self.args, "segmentation_registry_audit_only", False)):
            self._ensure_airsim_capture_vehicle()

    def _transform_position_enu(self, position_enu_m: Sequence[float]) -> list[float]:
        return self.coordinate_transform.apply_position(position_enu_m)

    def _inverse_transform_position_enu(self, position_enu_m: Sequence[float]) -> list[float]:
        return self.coordinate_transform.inverse_position(position_enu_m)

    def _transform_rotation_deg(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        return self.coordinate_transform.apply_rotation(rotation_deg)

    def _inverse_transform_rotation_deg(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        return self.coordinate_transform.inverse_rotation(rotation_deg)

    def _transform_vector_enu(self, value_enu: Sequence[float]) -> list[float]:
        return self.coordinate_transform.apply_vector(value_enu)

    def _truth_frame_uses_map_enu(self) -> bool:
        return self.truth_frame_coordinate_space in {"map_enu", "world_enu", "transformed_enu"}

    def _transform_truth_position_enu(self, position_enu_m: Sequence[float]) -> list[float]:
        if self._truth_frame_uses_map_enu():
            return [
                float(position_enu_m[0] if len(position_enu_m) > 0 else 0.0),
                float(position_enu_m[1] if len(position_enu_m) > 1 else 0.0),
                float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0),
            ]
        return self._transform_position_enu(position_enu_m)

    def _transform_truth_rotation_deg(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        raw = {
            "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", 0.0))),
            "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", 0.0))),
            "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", 0.0))),
        }
        if self._truth_frame_uses_map_enu():
            return raw
        return self.coordinate_transform.apply_rotation(raw)

    def _transform_truth_vector_enu(self, value_enu: Sequence[float]) -> list[float]:
        if self._truth_frame_uses_map_enu():
            return [
                float(value_enu[0] if len(value_enu) > 0 else 0.0),
                float(value_enu[1] if len(value_enu) > 1 else 0.0),
                float(value_enu[2] if len(value_enu) > 2 else 0.0),
            ]
        return self._transform_vector_enu(value_enu)

    def _apply_frame_position_enu(self, position_enu_m: Sequence[float]) -> list[float]:
        if self._truth_frame_uses_map_enu():
            return self._inverse_transform_position_enu(position_enu_m)
        return [
            float(position_enu_m[0] if len(position_enu_m) > 0 else 0.0),
            float(position_enu_m[1] if len(position_enu_m) > 1 else 0.0),
            float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0),
        ]

    def _apply_frame_rotation_deg(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        if self._truth_frame_uses_map_enu():
            return self._inverse_transform_rotation_deg(rotation_deg)
        return dict(rotation_deg)

    def _transformed_entity_pose(self, entity: dict[str, Any]) -> tuple[list[float], dict[str, float]]:
        position_enu_m = self._transform_truth_position_enu(position_enu_from_truth(entity))
        rotation_deg = self._transform_truth_rotation_deg(rotation_dict_from_truth(entity))
        yaw_offset_deg = self._entity_yaw_offset_deg(entity)
        if abs(yaw_offset_deg) > 1e-6:
            rotation_deg = dict(rotation_deg)
            rotation_deg["yaw_deg"] = float(rotation_deg.get("yaw_deg", 0.0)) + yaw_offset_deg
        return position_enu_m, rotation_deg

    def _resolve_entity_pose(
        self,
        entity: dict[str, Any],
    ) -> tuple[list[float], dict[str, float], dict[str, Any] | None]:
        position_enu_m, rotation_deg = self._transformed_entity_pose(entity)
        transformed_position_enu_m = list(position_enu_m)
        transformed_rotation_deg = dict(rotation_deg)
        snap_details: dict[str, Any] | None = None
        if self.road_topology_snapper.should_snap(entity):
            snap_result = self.road_topology_snapper.snap(
                entity_id=str(entity.get("entity_id", "")),
                position_enu_m=position_enu_m,
                rotation_deg=rotation_deg,
            )
            if snap_result is not None:
                position_enu_m = [
                    float(snap_result.position_enu_m[0]),
                    float(snap_result.position_enu_m[1]),
                    float(snap_result.position_enu_m[2] if self.road_topology_snapper.use_sample_z else position_enu_m[2]),
                ]
                if self.road_topology_snapper.use_sample_yaw:
                    rotation_deg = dict(rotation_deg)
                    rotation_deg["yaw_deg"] = float(snap_result.yaw_deg)
                snap_details = {
                    "kind": "road_topology_snap",
                    **snap_result.to_dict(),
                }
                lane_offset_details = self._vehicle_lane_offset_details(
                    entity,
                    yaw_deg=float(rotation_deg.get("yaw_deg", 0.0)),
                    edge_id=snap_result.edge_id,
                )
                if lane_offset_details is not None:
                    left_normal_xy = lane_offset_details["left_normal_xy"]
                    position_enu_m[0] += float(left_normal_xy[0]) * float(lane_offset_details["lane_offset_m"])
                    position_enu_m[1] += float(left_normal_xy[1]) * float(lane_offset_details["lane_offset_m"])
                    forward_xy = lane_offset_details["forward_xy"]
                    position_enu_m[0] += float(forward_xy[0]) * float(lane_offset_details["longitudinal_offset_m"])
                    position_enu_m[1] += float(forward_xy[1]) * float(lane_offset_details["longitudinal_offset_m"])
                    snap_details.update(lane_offset_details)
        else:
            roadside_adjustment = self._roadside_adjustment(entity, position_enu_m)
            if roadside_adjustment is not None:
                position_enu_m = list(roadside_adjustment["position_enu_m"])
                snap_details = dict(roadside_adjustment["details"])
        if snap_details is not None:
            snap_details["transformed_position_enu_m"] = list(transformed_position_enu_m)
            snap_details["transformed_rotation_deg"] = dict(transformed_rotation_deg)
        return position_enu_m, rotation_deg, snap_details

    def _entity_position_enu(self, entity: dict[str, Any]) -> list[float]:
        position_enu_m, _, _ = self._resolve_entity_pose(entity)
        return position_enu_m

    def _entity_yaw_offset_deg(self, entity: dict[str, Any]) -> float:
        category = str(entity.get("entity_category") or "").strip().lower()
        if category == "vehicle":
            return float(self.entity_rotation_offset_cfg.get("vehicle_yaw_deg", 0.0))
        if category == "pedestrian":
            return float(self.entity_rotation_offset_cfg.get("pedestrian_yaw_deg", 0.0))
        if category == "uav":
            return float(self.entity_rotation_offset_cfg.get("uav_yaw_deg", 0.0))
        return float(self.entity_rotation_offset_cfg.get("default_yaw_deg", 0.0))

    def _entity_rotation_deg(self, entity: dict[str, Any]) -> dict[str, float]:
        _, rotation_deg, _ = self._resolve_entity_pose(entity)
        return rotation_deg

    def _ground_relative_position(
        self,
        position_enu_m: Sequence[float],
        *,
        enabled: bool,
        cache_namespace: str = "",
        use_cache: bool = False,
    ) -> list[float]:
        resolved = [float(position_enu_m[0]), float(position_enu_m[1]), float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0)]
        if not enabled or self.client is None:
            return resolved

        ground_details = self._project_ground_details(
            resolved,
            cache_namespace=cache_namespace,
            use_cache=use_cache,
        )
        ground_relative = (ground_details or {}).get("ground_relative_enu_m")
        if not isinstance(ground_relative, Sequence) or isinstance(ground_relative, (str, bytes)) or len(ground_relative) < 3:
            raise RuntimeError(
                "Ground-relative UAV command requested, but simAeroProjectGround did not return "
                f"ground_relative_enu_m for position_enu_m={resolved}"
            )
        return [float(ground_relative[0]), float(ground_relative[1]), float(ground_relative[2])]

    def _runtime_uav_command_position_enu(self, entity_id: str, position_enu_m: Sequence[float]) -> list[float]:
        return self._ground_relative_position(
            position_enu_m,
            enabled=bool(self.ground_reference_cfg.get("uav_ground_relative", False)),
            cache_namespace=f"uav_script:{entity_id}",
            use_cache=True,
        )

    def _project_ground_details(
        self,
        position_enu_m: Sequence[float],
        *,
        cache_namespace: str = "",
        use_cache: bool = False,
    ) -> dict[str, Any] | None:
        resolved = [float(position_enu_m[0]), float(position_enu_m[1]), float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0)]
        if self.client is None:
            return None

        cache_key = (cache_namespace, int(round(resolved[0] * 100.0)), int(round(resolved[1] * 100.0)))
        payload = self.ground_projection_cache.get(cache_key) if use_cache else None
        if payload is None:
            response = self._retry("project_ground", self.client.project_ground, point_enu_m=resolved, map_id=self.map_id)
            payload = dict(response.get("payload") or {})
            if use_cache:
                self.ground_projection_cache[cache_key] = dict(payload)

        projected_raw = payload.get("projected_enu_m") or resolved
        surface_normal_raw = payload.get("surface_normal_enu") or [0.0, 0.0, 1.0]
        projected_enu_m = [
            float(projected_raw[0] if len(projected_raw) > 0 else resolved[0]),
            float(projected_raw[1] if len(projected_raw) > 1 else resolved[1]),
            float(projected_raw[2] if len(projected_raw) > 2 else resolved[2]),
        ]
        surface_normal_enu = [
            float(surface_normal_raw[0] if len(surface_normal_raw) > 0 else 0.0),
            float(surface_normal_raw[1] if len(surface_normal_raw) > 1 else 0.0),
            float(surface_normal_raw[2] if len(surface_normal_raw) > 2 else 1.0),
        ]
        anchor_id = str(payload.get("anchor_id") or "")
        ground_resolved = bool(anchor_id) or abs(projected_enu_m[2] - resolved[2]) > 1e-3
        fallback_payload: dict[str, Any] | None = None
        if not ground_resolved and self.road_geometry is not None:
            search_radius_m = float(self.ground_reference_cfg.get("traffic_bundle_fallback_radius_m", 25.0))
            projection = self.road_geometry.project_point(resolved, radius_m=search_radius_m)
            if projection is not None:
                projected_enu_m = [
                    float(projection.projected_enu_m[0]),
                    float(projection.projected_enu_m[1]),
                    float(projection.projected_enu_m[2]),
                ]
                surface_normal_enu = [0.0, 0.0, 1.0]
                ground_resolved = True
                fallback_payload = {
                    "source": "traffic_bundle_lane_center_samples",
                    "search_radius_m": search_radius_m,
                    "projection": projection.to_dict(),
                }
        return {
            "input_enu_m": resolved,
            "projected_enu_m": projected_enu_m,
            "surface_normal_enu": surface_normal_enu,
            "anchor_id": anchor_id,
            "ground_resolved": ground_resolved,
            "ground_projection_fallback": fallback_payload,
            "ground_relative_enu_m": (
                [resolved[0], resolved[1], projected_enu_m[2] + resolved[2]]
                if ground_resolved
                else list(resolved)
            ),
        }

    def dump_coordinate_preview(self, *, include_ground: bool = False) -> None:
        tick = self.args.coord_preview_tick if self.args.coord_preview_tick is not None else self.sorted_ticks[0]
        if tick not in self.frames_by_tick:
            raise RuntimeError(f"Tick {tick} is not available in truth_frames.jsonl")

        limit = max(1, int(self.args.coord_preview_limit or 10))
        frame = self.frames_by_tick[tick]
        preview_entities = list(frame.get("entities", []))
        if self.args.site:
            preview_entities = [
                entity for entity in preview_entities if str(entity.get("site_id", "")) == str(self.args.site)
            ]
        if not preview_entities:
            requested_site = str(self.args.site or "").strip()
            if requested_site:
                raise RuntimeError(f"No entities found for site '{requested_site}' at tick {tick}.")
            raise RuntimeError(f"No entities found at tick {tick}.")
        print(
            f"[EpisodeHost] Coordinate preview tick={tick} frame_id={frame.get('frame_id', '')} "
            f"transform={self.coordinate_transform.describe()} "
            f"truth_frame_coordinate_space={self.truth_frame_coordinate_space}"
        )
        for entity in preview_entities[:limit]:
            entity_id = str(entity.get("entity_id", ""))
            raw_position = position_enu_from_truth(entity)
            raw_rotation = rotation_dict_from_truth(entity)
            transformed_position, transformed_rotation = self._transformed_entity_pose(entity)
            resolved_position, resolved_rotation, snap_details = self._resolve_entity_pose(entity)
            resolution = self._entity_resolution(entity)
            message = (
                f"[EpisodeHost] entity={entity_id} mode={resolution.get('mode', '')} "
                f"raw_pos={raw_position} transformed_pos={transformed_position} resolved_pos={resolved_position} "
                f"raw_rot={raw_rotation} transformed_rot={transformed_rotation} resolved_rot={resolved_rotation}"
            )
            if snap_details is not None:
                detail_kind = str(snap_details.get("kind") or "")
                if detail_kind == "road_topology_snap":
                    message += (
                        f" road_edge={snap_details['edge_id']} lane={snap_details['lane_id']} "
                        f"road_s_m={snap_details['s_m']:.1f} road_dist_m={snap_details['distance_m']:.2f} "
                        f"road_heading_err_deg={snap_details['heading_error_deg']:.1f}"
                    )
                    if snap_details.get("lane_offset_m") is not None:
                        message += (
                            f" lane_offset_m={float(snap_details['lane_offset_m']):.2f} "
                            f"band={int(snap_details['band_index']) + 1}/{int(snap_details['band_count'])}"
                        )
                    if snap_details.get("longitudinal_offset_m") is not None and abs(float(snap_details["longitudinal_offset_m"])) > 1e-6:
                        message += (
                            f" queue_offset_m={float(snap_details['longitudinal_offset_m']):.2f} "
                            f"dup={int(snap_details['duplicate_index']) + 1}/{int(snap_details['duplicate_count'])}"
                        )
                        if int(snap_details.get("collapsed_band_count") or 1) > 1:
                            message += (
                                f" collapsed_lane={int(snap_details['collapsed_band_index']) + 1}/"
                                f"{int(snap_details['collapsed_band_count'])}"
                            )
                elif detail_kind == "pedestrian_roadside_projection":
                    message += (
                        f" roadside_edge={snap_details['edge_id']} road_dist_m={snap_details['distance_m']:.2f} "
                        f"road_width_m={snap_details['width_m']:.1f} clearance_m={snap_details['clearance_m']:.2f} "
                        f"moved={bool(snap_details.get('moved', False))}"
                    )
            if include_ground:
                ground_details = self._project_ground_details(resolved_position)
                if ground_details is not None:
                    message += (
                        f" ground_projected_pos={ground_details['projected_enu_m']} "
                        f"ground_relative_pos={ground_details['ground_relative_enu_m']} "
                        f"ground_resolved={ground_details['ground_resolved']} "
                        f"anchor_id={ground_details['anchor_id'] or '<none>'} "
                        f"surface_normal={ground_details['surface_normal_enu']}"
                    )
            print(message)

        site_id = str(preview_entities[0].get("site_id", "")) if preview_entities else ""
        for preset in self._site_ground_presets(self.args.site or site_id)[:limit]:
            camera_id = str(preset.get("camera_id", preset.get("camera_name", "external")))
            raw_position = [float(value) for value in preset.get("position_enu_m") or [0.0, 0.0, 0.0]]
            resolved_position = self._camera_position_from_preset(preset)
            raw_rotation = dict(preset.get("rotation_deg") or {})
            resolved_rotation = self._camera_rotation_from_preset(preset)
            message = (
                f"[EpisodeHost] camera={camera_id} raw_pos={raw_position} resolved_pos={resolved_position} "
                f"raw_rot={raw_rotation} resolved_rot={resolved_rotation}"
            )
            if include_ground:
                ground_details = self._project_ground_details(resolved_position)
                if ground_details is not None:
                    message += (
                        f" ground_projected_pos={ground_details['projected_enu_m']} "
                        f"ground_relative_pos={ground_details['ground_relative_enu_m']} "
                        f"ground_resolved={ground_details['ground_resolved']} "
                        f"anchor_id={ground_details['anchor_id'] or '<none>'} "
                        f"surface_normal={ground_details['surface_normal_enu']}"
                    )
            print(message)

    def _validate_template_coverage(self) -> None:
        unresolved: list[str] = []
        for entity in self.global_roster:
            mode = str(self.template_resolver.resolve(entity, entity).get("mode", ""))
            if mode not in {"scene_sync", "pedestrian_managed", "runtime_multirotor", "metadata_only"}:
                unresolved.append(str(entity.get("entity_id", "")))
        if unresolved:
            raise RuntimeError(f"Template resolver returned unsupported mode for entities: {', '.join(unresolved)}")

    def _soft_reset_tracking(self) -> None:
        self.scene_active_ids.clear()
        self.ped_active_ids.clear()
        self.ped_last_activity.clear()
        self.ped_last_variant.clear()
        self.uav_active_by_entity.clear()
        self.uav_last_command_target_by_entity.clear()
        self.crowd_group_ids.clear()
        self.event_controlled_entity_ids.clear()
        self.road_topology_snapper.clear_state()
        self.last_weather_signature = None
        self.capture_warmup_complete = False

    def _best_effort(self, label: str, func: Any, *args: Any, **kwargs: Any) -> Any | None:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(f"[EpisodeHost] {label} warning: {exc}")
            return None

    def _weather_for_tick(self, tick: int) -> dict[str, Any]:
        if tick in self.weather_by_tick:
            return dict(self.weather_by_tick[tick])
        if not self.weather_ticks:
            return {}
        index = bisect.bisect_right(self.weather_ticks, tick) - 1
        return dict(self.weather_by_tick[self.weather_ticks[index]]) if index >= 0 else {}

    def _build_weather_payload(self, weather_row: dict[str, Any]) -> dict[str, Any]:
        payload = self.weather_service.payload_for_row(weather_row)
        capture_profiles = dict(self.capture_presets.get("weather_profiles") or {})
        condition = str(payload.get("condition") or weather_row.get("condition") or "clear")
        if condition in capture_profiles:
            payload = {**dict(capture_profiles.get(condition) or {}), **payload}
        return payload

    def _apply_weather_if_needed(self, tick: int) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        payload = self._build_weather_payload(self._weather_for_tick(tick))
        if self.event_weather_overlay:
            payload.update(self.event_weather_overlay)
        if payload != self.last_weather_signature:
            try:
                self._retry("apply_weather", self.client.apply_weather, payload, map_id=self.map_id)
            except Exception as exc:
                print(f"[EpisodeHost] apply_weather RPC failed, falling back to editor hook: {exc}")
                hook = self._fixed_world_capture_hook()
                hook.apply_weather(
                    map_id=self.map_id,
                    payload=payload,
                    request_id=f"weather_tick_{int(tick):06d}",
                )
            self.last_weather_signature = dict(payload)
        return payload

    def _entity_resolution(self, entity: dict[str, Any]) -> dict[str, Any]:
        roster_entry = self.roster_by_id.get(str(entity.get("entity_id")), {})
        return self.template_resolver.resolve(entity, roster_entry)

    def _scene_visual_state(self, entity: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
        annotations = dict(entity.get("annotations") or {})
        state_facets = dict(annotations.get("state_facets") or {})
        signal_state = dict(annotations.get("signal") or state_facets.get("signal_control") or {})
        activity_state = dict(state_facets.get("activity") or {})
        result: dict[str, Any] = {
            "mode": "hidden" if visibility_state(entity).lower() in {"hidden", "invisible"} else "visible"
        }
        if resolution.get("variant_id"):
            result["variant_id"] = str(resolution.get("variant_id"))
        montage_tag = activity_state.get("animation_hint") or activity_state.get("activity_type") or annotations.get("activity_type")
        if montage_tag:
            result["montage_tag"] = str(montage_tag)
        if signal_state.get("phase"):
            result["material_variant"] = str(signal_state.get("phase"))
        if annotations.get("lights_on") is not None:
            result["lights_on"] = bool(annotations.get("lights_on"))
        elif signal_state.get("phase") is not None:
            result["lights_on"] = str(signal_state.get("phase")).lower() not in {"off", "red"}
        return result

    def _scene_sync_item(
        self,
        entity: dict[str, Any],
        resolution: dict[str, Any],
        *,
        position_enu_m: Sequence[float] | None = None,
        rotation_deg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if position_enu_m is None or rotation_deg is None:
            position_enu_m, rotation_deg, _ = self._resolve_entity_pose(entity)
        return {
            "entity_id": str(entity["entity_id"]),
            "proxy_template_id": str(resolution["logical_asset_id"]),
            "pose_enu_m": {
                "position_enu_m": [float(value) for value in position_enu_m],
                "rotation_deg": dict(rotation_deg),
            },
            "tags": list(entity.get("tags") or self.roster_by_id.get(str(entity["entity_id"]), {}).get("tags") or []),
            "visual_state": self._scene_visual_state(entity, resolution),
        }

    def _entity_record(self, entity: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
        annotations = dict(entity.get("annotations") or {})
        signal_phase = None
        if isinstance(annotations.get("signal"), dict):
            signal_phase = annotations["signal"].get("phase")
        raw_position_enu_m = position_enu_from_truth(entity)
        raw_rotation_deg = rotation_dict_from_truth(entity)
        transformed_position_enu_m, transformed_rotation_deg = self._transformed_entity_pose(entity)
        resolved_position_enu_m, resolved_rotation_deg, snap_details = self._resolve_entity_pose(entity)
        return RuntimeFrameRecord(
            entity_id=str(entity.get("entity_id", "")),
            entity_category=str(entity.get("entity_category", "")),
            position_enu_m=list(resolved_position_enu_m),
            rotation_deg=dict(resolved_rotation_deg),
            velocity_enu_mps=list(self._transform_truth_vector_enu(velocity_enu_from_truth(entity))),
            submission_state=truth_submission_state(entity),
            visibility_state=visibility_state(entity),
            source_pose={
                "position_enu_m": list(raw_position_enu_m),
                "rotation_deg": dict(raw_rotation_deg),
            },
            resolved_pose={
                "position_enu_m": list(resolved_position_enu_m),
                "rotation_deg": dict(resolved_rotation_deg),
            },
            transformed_pose={
                "position_enu_m": list(transformed_position_enu_m),
                "rotation_deg": dict(transformed_rotation_deg),
            },
            extra={
                "entity_kind": str(entity.get("entity_kind") or entity.get("entity_type") or ""),
                "site_id": str(entity.get("site_id", "")),
                "proxy_template_id": str(entity.get("proxy_template_id") or ""),
                "logical_asset_id": str(resolution.get("logical_asset_id") or ""),
                "mode": str(resolution.get("mode", "metadata_only")),
                "source_position_enu_m": raw_position_enu_m,
                "source_rotation_deg": raw_rotation_deg,
                "transformed_position_enu_m": transformed_position_enu_m,
                "transformed_rotation_deg": transformed_rotation_deg,
                "activity_type": normalize_activity_type(entity),
                "signal_phase": signal_phase,
                "pose_adjustment": snap_details,
                "road_topology_snap": snap_details if str((snap_details or {}).get("kind") or "") == "road_topology_snap" else None,
            },
        ).to_dict()

    def visible_site_entity_rows(
        self,
        frame: dict[str, Any],
        *,
        site_id: str,
        entity_categories: Sequence[str] = ("vehicle", "pedestrian"),
        include_overlap_entity_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        allowed_categories = {str(category).strip().lower() for category in entity_categories}
        allowed_overlap_ids = {str(entity_id).strip() for entity_id in (include_overlap_entity_ids or set()) if str(entity_id).strip()}
        rows: list[dict[str, Any]] = []
        scene_candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        ped_candidates: list[tuple[int, str, dict[str, Any], dict[str, Any], dict[str, Any], bool]] = []
        for entity in frame.get("entities", []):
            if site_id and str(entity.get("site_id") or "") != site_id:
                continue
            category = str(entity.get("entity_category") or "").strip().lower()
            if allowed_categories and category not in allowed_categories:
                continue
            render_presence = entity.get("render_presence") or {}
            if str(render_presence.get("submission_state") or "") != "submit_to_ue":
                continue
            if str(render_presence.get("visibility_state") or "visible").lower() == "hidden":
                continue
            resolution = self._entity_resolution(entity)
            record = self._entity_record(entity, resolution)
            entity_id = str(entity.get("entity_id") or "")
            mode = str(resolution.get("mode", ""))
            if mode == "scene_sync":
                scene_candidates.append((entity, resolution, record))
                continue
            if mode == "pedestrian_managed":
                activity_type = normalize_activity_type(entity)
                activity_rule = self._ped_activity_rule(activity_type)
                preserve_pose = bool(activity_rule.get("freeze_pose_while_active", False))
                priority_rank = 0 if (preserve_pose or is_fallen_activity(entity)) else 1
                ped_candidates.append((priority_rank, entity_id, entity, resolution, record, preserve_pose))
                continue

            if entity_id in allowed_overlap_ids or not record.get("overlap_filtered"):
                rows.append({"entity": entity, "record": record})

        accepted_scene_records: list[dict[str, Any]] = []
        for entity, _, record in scene_candidates:
            entity_id = str(entity.get("entity_id") or "")
            overlap = self._vehicle_overlap_reason(record, accepted_scene_records)
            if overlap is not None and entity_id not in allowed_overlap_ids:
                kept_record, reason, dist_m = overlap
                record["overlap_filtered"] = {
                    "reason": reason,
                    "distance_m": dist_m,
                    "kept_entity_id": str(kept_record.get("entity_id") or ""),
                }
                continue
            accepted_scene_records.append(record)
            rows.append({"entity": entity, "record": record})

        accepted_ped_records: list[dict[str, Any]] = []
        ped_candidates.sort(key=lambda row: (row[0], row[1]))
        for _, entity_id, entity, _, record, preserve_pose in ped_candidates:
            overlap = self._pedestrian_overlap_reason(record, accepted_ped_records)
            if overlap is not None and not preserve_pose and entity_id not in allowed_overlap_ids:
                kept_record, reason, dist_m = overlap
                record["overlap_filtered"] = {
                    "reason": reason,
                    "distance_m": dist_m,
                    "kept_entity_id": str(kept_record.get("entity_id") or ""),
                }
                continue
            accepted_ped_records.append(record)
            rows.append({"entity": entity, "record": record})
        return rows

    def apply_frame_capture_state(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, _, entity_records = self._apply_scene_frame(frame)
        self._sync_pedestrians(frame)
        return entity_records

    def _record_xy_distance_m(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        position_a = list(a.get("position_enu_m") or [0.0, 0.0, 0.0])
        position_b = list(b.get("position_enu_m") or [0.0, 0.0, 0.0])
        return math.hypot(float(position_a[0]) - float(position_b[0]), float(position_a[1]) - float(position_b[1]))

    def _vehicle_overlap_reason(
        self,
        record: dict[str, Any],
        accepted_records: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], str, float] | None:
        if not bool(self.entity_overlap_cfg.get("enabled", False)):
            return None
        if str(record.get("entity_category") or "").strip().lower() != "vehicle":
            return None

        hard_overlap_m = max(0.25, float(self.entity_overlap_cfg.get("vehicle_hard_overlap_m", 2.75)))
        same_lane_gap_m = max(hard_overlap_m, float(self.entity_overlap_cfg.get("vehicle_same_lane_gap_m", 6.0)))
        lane_offset_tolerance_m = max(0.25, float(self.entity_overlap_cfg.get("vehicle_lane_offset_tolerance_m", 0.9)))
        snap = dict(record.get("road_topology_snap") or {})
        edge_id = str(snap.get("edge_id") or "")
        lane_offset_m = snap.get("lane_offset_m")
        s_m = snap.get("s_m")

        for existing in accepted_records:
            if str(existing.get("entity_category") or "").strip().lower() != "vehicle":
                continue
            dist_m = self._record_xy_distance_m(record, existing)
            if dist_m < hard_overlap_m:
                return existing, f"xy_overlap<{hard_overlap_m:.2f}m", dist_m

            existing_snap = dict(existing.get("road_topology_snap") or {})
            existing_edge_id = str(existing_snap.get("edge_id") or "")
            existing_lane_offset_m = existing_snap.get("lane_offset_m")
            existing_s_m = existing_snap.get("s_m")
            if (
                edge_id
                and existing_edge_id == edge_id
                and lane_offset_m is not None
                and existing_lane_offset_m is not None
                and s_m is not None
                and existing_s_m is not None
                and abs(float(lane_offset_m) - float(existing_lane_offset_m)) <= lane_offset_tolerance_m
                and abs(float(s_m) - float(existing_s_m)) < same_lane_gap_m
            ):
                return existing, f"same_lane_gap<{same_lane_gap_m:.2f}m", dist_m
        return None

    def _pedestrian_overlap_reason(
        self,
        record: dict[str, Any],
        accepted_records: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Any], str, float] | None:
        if not bool(self.entity_overlap_cfg.get("enabled", False)):
            return None
        if str(record.get("entity_category") or "").strip().lower() != "pedestrian":
            return None

        min_distance_m = max(0.1, float(self.entity_overlap_cfg.get("pedestrian_min_distance_m", 0.75)))
        for existing in accepted_records:
            if str(existing.get("entity_category") or "").strip().lower() != "pedestrian":
                continue
            dist_m = self._record_xy_distance_m(record, existing)
            if dist_m < min_distance_m:
                return existing, f"xy_overlap<{min_distance_m:.2f}m", dist_m
        return None

    def hard_reset_world_state(self) -> None:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        print("[EpisodeHost] Resetting previously spawned scene_sync, pedestrians, and runtime UAVs.")
        if self.all_scene_sync_ids:
            payload = {
                "tick": 0,
                "frame_id": 0,
                "sample_seq": 0,
                "sim_time_s": 0.0,
                "episode_id": self.episode_id,
                "removes": [{"entity_id": entity_id} for entity_id in sorted(self.all_scene_sync_ids)],
            }
            try:
                self._retry("reset_apply_frame", self.client.apply_frame, payload, map_id=self.map_id)
            except Exception as exc:
                print(f"[EpisodeHost] reset_apply_frame warning: {exc}")
        extra_remove_ids = [
            str(value).strip()
            for value in (dict(self.config.get("scene_reset") or {}).get("force_remove_entity_ids") or [])
            if str(value).strip()
        ]
        if extra_remove_ids:
            payload = {
                "tick": 0,
                "frame_id": 0,
                "sample_seq": 0,
                "sim_time_s": 0.0,
                "episode_id": self.episode_id or "forced_cleanup",
                "removes": [{"entity_id": entity_id} for entity_id in sorted(set(extra_remove_ids))],
            }
            try:
                self._retry("forced_reset_apply_frame", self.client.apply_frame, payload, map_id=self.map_id)
            except Exception as exc:
                print(f"[EpisodeHost] forced_reset_apply_frame warning: {exc}")
        for ped_id in sorted(self.all_ped_ids):
            self._best_effort("ped_release", self.client.ped_release, ped_id, map_id=self.map_id)
        for group_id in sorted(self.crowd_group_ids):
            self._best_effort("ped_clear_crowd", self.client.ped_clear_crowd, group_id, map_id=self.map_id)
        for vehicle_name in sorted(self.all_uav_vehicle_names):
            if self._runtime_uav_use_editor_hook():
                self._best_effort(
                    "remove_runtime_vehicle_editor_hook",
                    self._runtime_uav_editor_hook().remove_runtime_vehicle,
                    map_id=self.map_id,
                    vehicle_name=vehicle_name,
                )
                continue
            try:
                self.client.remove_runtime_vehicle(vehicle_name, map_id=self.map_id)
            except Exception as exc:
                print(f"[EpisodeHost] remove_runtime_vehicle reset warning for {vehicle_name}: {exc}")
                self._best_effort(
                    "remove_runtime_vehicle_editor_hook",
                    self._runtime_uav_editor_hook().remove_runtime_vehicle,
                    map_id=self.map_id,
                    vehicle_name=vehicle_name,
                )
        destroy_payload = self._force_destroy_runtime_vehicle_actors(self.all_uav_vehicle_names)
        if destroy_payload:
            destroyed_rows = list(destroy_payload.get("destroyed") or [])
            if destroyed_rows:
                destroyed_names = ", ".join(sorted({str(row.get("vehicle_name") or "") for row in destroyed_rows if str(row.get("vehicle_name") or "").strip()}))
                print(f"[EpisodeHost] Force-destroyed lingering runtime UAV actors: {destroyed_names}")
        self._soft_reset_tracking()

    def _apply_scene_frame(self, frame: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        frame_records: list[dict[str, Any]] = []
        current_scene_ids: set[str] = set()
        spawns: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        removes: list[dict[str, Any]] = []
        scene_candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

        for entity in frame.get("entities", []):
            resolution = self._entity_resolution(entity)
            record = self._entity_record(entity, resolution)
            frame_records.append(record)
            if str(resolution.get("mode", "")) != "scene_sync":
                continue
            if truth_submission_state(entity) != "submit_to_ue":
                continue
            scene_candidates.append((entity, resolution, record))

        accepted_scene_records: list[dict[str, Any]] = []
        for entity, resolution, record in scene_candidates:
            entity_id = str(entity["entity_id"])
            overlap = self._vehicle_overlap_reason(record, accepted_scene_records)
            if overlap is not None:
                kept_record, reason, dist_m = overlap
                record["overlap_filtered"] = {
                    "reason": reason,
                    "distance_m": dist_m,
                    "kept_entity_id": str(kept_record.get("entity_id") or ""),
                }
                print(
                    f"[EpisodeHost] Filtering scene entity {entity_id} due to overlap with "
                    f"{record['overlap_filtered']['kept_entity_id']} reason={reason} distance_m={dist_m:.2f}"
                )
                continue

            accepted_scene_records.append(record)
            current_scene_ids.add(entity_id)
            item = self._scene_sync_item(
                entity,
                resolution,
                position_enu_m=self._apply_frame_position_enu(record["position_enu_m"]),
                rotation_deg=self._apply_frame_rotation_deg(record["rotation_deg"]),
            )
            if entity_id in self.scene_active_ids:
                updates.append(item)
            else:
                spawns.append(item)

        for entity_id in sorted(self.scene_active_ids - current_scene_ids):
            removes.append({"entity_id": entity_id})

        if spawns or updates or removes:
            numeric_frame_id = int(frame.get("frame_seq") or frame.get("tick") or 0)
            payload = {
                "tick": int(frame["tick"]),
                "frame_id": numeric_frame_id,
                "sample_seq": numeric_frame_id,
                "sim_time_s": float(frame.get("sim_time_s", 0.0)),
                "episode_id": self.episode_id,
                "spawns": spawns,
                "updates": updates,
                "removes": removes,
            }
            self._retry("apply_frame", self.client.apply_frame, payload, map_id=self.map_id)

        self.scene_active_ids = current_scene_ids
        return spawns, updates, removes, frame_records

    def _ped_activity_action(self, ped_id: str, activity_type: str) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        action = self._ped_activity_rule(activity_type)
        op_list = list(action.get("ops") or [])
        if not op_list:
            op_list = [dict(action)]

        results: list[dict[str, Any]] = []
        for raw_step in op_list:
            step = dict(raw_step or {})
            op = str(step.get("op", "none"))
            if op == "ped_observe":
                results.append(
                    self._retry(
                        "ped_observe",
                        self.client.ped_observe,
                        ped_id,
                        start_section=str(step.get("start_section", "")),
                        map_id=self.map_id,
                    )
                )
                continue
            if op == "ped_stop":
                results.append(self._retry("ped_stop", self.client.ped_stop, ped_id, map_id=self.map_id))
                continue
            if op == "ped_play_animation":
                results.append(
                    self._retry(
                        "ped_play_animation",
                        self.client.ped_play_animation,
                        ped_id,
                        str(step["animation_asset_path"]),
                        start_section=str(step.get("start_section", "")),
                        play_rate=float(step.get("play_rate", 1.0)),
                        loop_count=int(step.get("loop_count", 1)),
                        map_id=self.map_id,
                    )
                )
                continue
            results.append({"status": "skipped", "op": op})

        if len(results) == 1:
            return results[0]
        return {"status": "multi", "steps": results}

    def _ped_activity_rule(self, activity_type: str) -> dict[str, Any]:
        activity_map = dict((self.template_resolver_doc.get("pedestrian_defaults") or {}).get("activity_actions", {}))
        return dict(activity_map.get(activity_type) or activity_map.get("default") or {"op": "none"})

    def _sync_pedestrians(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        current_ids: set[str] = set()
        results: dict[str, Any] = {}
        accepted_ped_records: list[dict[str, Any]] = []
        ped_rows: list[tuple[int, str, dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any], bool]] = []
        for entity in frame.get("entities", []):
            resolution = self._entity_resolution(entity)
            if str(resolution.get("mode", "")) != "pedestrian_managed":
                continue
            if truth_submission_state(entity) != "submit_to_ue":
                continue
            ped_id = str(entity["entity_id"])
            record = self._entity_record(entity, resolution)
            activity_type = normalize_activity_type(entity)
            activity_rule = self._ped_activity_rule(activity_type)
            preserve_pose = bool(activity_rule.get("freeze_pose_while_active", False))
            priority_rank = 0 if (preserve_pose or is_fallen_activity(entity)) else 1
            ped_rows.append((priority_rank, ped_id, entity, resolution, record, activity_type, activity_rule, preserve_pose))

        ped_rows.sort(key=lambda row: (row[0], row[1]))
        for _, ped_id, entity, resolution, record, activity_type, activity_rule, preserve_pose in ped_rows:
            skip_overlap = preserve_pose or is_fallen_activity(entity)
            overlap = self._pedestrian_overlap_reason(record, accepted_ped_records)
            if overlap is not None and not skip_overlap:
                kept_record, reason, dist_m = overlap
                results[ped_id] = {
                    "filtered_overlap": {
                        "reason": reason,
                        "distance_m": dist_m,
                        "kept_entity_id": str(kept_record.get("entity_id") or ""),
                    }
                }
                print(
                    f"[EpisodeHost] Filtering pedestrian {ped_id} due to overlap with "
                    f"{results[ped_id]['filtered_overlap']['kept_entity_id']} reason={reason} distance_m={dist_m:.2f}"
                )
                continue

            accepted_ped_records.append(record)
            current_ids.add(ped_id)
            position_enu_m = list(record["position_enu_m"])
            if activity_rule.get("ground_lift_m") is not None:
                position_enu_m[2] += float(activity_rule.get("ground_lift_m") or 0.0)
            variant_id = str(resolution.get("variant_id") or "adult_male_commuter")
            should_preserve_pose = (
                preserve_pose
                and ped_id in self.ped_active_ids
                and self.ped_last_activity.get(ped_id) == activity_type
            )
            rotation_deg = self.pedestrian_pose_service.resolve_rotation(
                entity_id=ped_id,
                position_enu_m=position_enu_m,
                velocity_enu_mps=list(record.get("velocity_enu_mps") or [0.0, 0.0, 0.0]),
                base_rotation_deg=dict(record["rotation_deg"]),
                activity_type=activity_type,
                freeze_pose=should_preserve_pose,
            )

            # Keep truth XY authoritative for pedestrians; C++ ground snapping only resolves
            # terrain Z when preserve_xy is enabled.
            snap_to_ground = bool(activity_rule.get("snap_to_ground", True))
            velocity = list(record.get("velocity_enu_mps") or [0.0, 0.0, 0.0])
            speed_mps = math.sqrt(sum(float(value) * float(value) for value in velocity[:2]))
            frame_pose = not preserve_pose and not is_fallen_activity(entity)
            locomotion_activity = activity_type in {"walking", "crossing", "evacuating", "texting_walk"}
            moving_locomotion = locomotion_activity and speed_mps > 0.05
            walking = frame_pose and moving_locomotion
            effective_activity_type = activity_type
            effective_activity_rule = activity_rule
            if frame_pose and locomotion_activity and not moving_locomotion:
                effective_activity_type = "stopped"
                effective_activity_rule = self._ped_activity_rule(effective_activity_type)
            pose_sync_performed = False

            if ped_id not in self.ped_active_ids:
                spawn_response = self._retry(
                    "ped_spawn",
                    self.client.ped_spawn,
                    ped_id,
                    position_enu_m=position_enu_m,
                    yaw_deg=float(rotation_deg["yaw_deg"]),
                    variant_id=variant_id,
                    snap_to_ground=snap_to_ground,
                    preserve_xy=True,
                    map_id=self.map_id,
                )
                results[ped_id] = {"spawn": spawn_response.get("payload", {})}
                pose_sync_performed = True
                if frame_pose:
                    frame_pose_response = self._retry(
                        "ped_frame_pose",
                        self.client.ped_reset,
                        ped_id,
                        position_enu_m=position_enu_m,
                        yaw_deg=float(rotation_deg["yaw_deg"]),
                        snap_to_ground=snap_to_ground,
                        preserve_xy=True,
                        frame_pose=True,
                        walking=walking,
                        speed_cm_per_sec=speed_mps * 100.0,
                        map_id=self.map_id,
                    )
                    results.setdefault(ped_id, {})["frame_pose"] = frame_pose_response.get("payload", {})
            elif should_preserve_pose:
                results[ped_id] = {
                    "reset": {
                        "status": "skipped",
                        "reason": "freeze_pose_while_active",
                    }
                }
                pose_sync_performed = False
            else:
                reset_response = self._retry(
                    "ped_frame_pose" if frame_pose else "ped_reset",
                    self.client.ped_reset,
                    ped_id,
                    position_enu_m=position_enu_m,
                    yaw_deg=float(rotation_deg["yaw_deg"]),
                    snap_to_ground=snap_to_ground,
                    preserve_xy=True,
                    frame_pose=frame_pose,
                    walking=walking,
                    speed_cm_per_sec=speed_mps * 100.0,
                    map_id=self.map_id,
                )
                results[ped_id] = {("frame_pose" if frame_pose else "reset"): reset_response.get("payload", {})}
                pose_sync_performed = True

            if self.ped_last_variant.get(ped_id) != variant_id:
                variant_response = self._retry("ped_set_variant", self.client.ped_set_variant, ped_id, variant_id, map_id=self.map_id)
                results.setdefault(ped_id, {})["variant"] = variant_response.get("payload", {})
                self.ped_last_variant[ped_id] = variant_id

            should_reapply_activity = bool(effective_activity_rule.get("reapply_after_pose_sync", False)) and pose_sync_performed
            if self.ped_last_activity.get(ped_id) != effective_activity_type or should_reapply_activity:
                results.setdefault(ped_id, {})["activity"] = self._ped_activity_action(ped_id, effective_activity_type)
                self.ped_last_activity[ped_id] = effective_activity_type

        for ped_id in sorted(self.ped_active_ids - current_ids):
            try:
                self._retry("ped_release", self.client.ped_release, ped_id, map_id=self.map_id)
            except Exception as exc:
                print(f"[EpisodeHost] ped_release warning for {ped_id}: {exc}")
            self.ped_last_activity.pop(ped_id, None)
            self.ped_last_variant.pop(ped_id, None)

        self.ped_active_ids = current_ids
        return results

    def _wait_for_uav_status(self, vehicle_name: str, target_enu_m: Sequence[float]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        timeout_cfg = dict(self.config.get("timeouts") or {})
        timeout_s = float(timeout_cfg.get("uav_move_timeout_s", 10.0))
        if timeout_s <= 0.0:
            pose_payload: dict[str, Any] = {}
            pose_warning = ""
            try:
                pose_response = self._retry("get_runtime_vehicle_pose", self.client.get_runtime_vehicle_pose, vehicle_name, map_id=self.map_id)
                pose_payload = dict(pose_response.get("payload") or {})
            except Exception as exc:
                pose_warning = str(exc)
            result = self.uav_execution_service.build_wait_status(
                vehicle_name=vehicle_name,
                status_payload={"state": "skipped", "reason": "uav status polling disabled"},
                pose_payload=pose_payload,
                target_enu_m=target_enu_m,
                timed_out=False,
                warning=pose_warning,
            )
            if pose_warning:
                result["pose_warning"] = pose_warning
            return result
        poll_interval_s = float(timeout_cfg.get("uav_status_poll_interval_s", 0.25))
        deadline = time.perf_counter() + timeout_s
        status_payload: dict[str, Any] = {}
        timed_out = False

        while time.perf_counter() < deadline:
            status_response = self._retry("get_runtime_multirotor_status", self.client.get_runtime_multirotor_status, vehicle_name, map_id=self.map_id)
            status_payload = dict(status_response.get("payload") or {})
            state = str(status_payload.get("state", "idle")).lower()
            if state in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(poll_interval_s)
        else:
            timed_out = True

        pose_response = self._retry("get_runtime_vehicle_pose", self.client.get_runtime_vehicle_pose, vehicle_name, map_id=self.map_id)
        pose_payload = dict(pose_response.get("payload") or {})
        return self.uav_execution_service.build_wait_status(
            vehicle_name=vehicle_name,
            status_payload=status_payload,
            pose_payload=pose_payload,
            target_enu_m=target_enu_m,
            timed_out=timed_out,
        )

    def _uav_status_snapshot(self, vehicle_name: str, target_enu_m: Sequence[float] | None = None) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        status_payload: dict[str, Any] = {}
        pose_payload: dict[str, Any] = {}
        warning_parts: list[str] = []
        try:
            status_response = self._retry(
                "get_runtime_multirotor_status",
                self.client.get_runtime_multirotor_status,
                vehicle_name,
                map_id=self.map_id,
            )
            status_payload = dict(status_response.get("payload") or {})
        except Exception as exc:
            warning_parts.append(f"status={exc}")
        try:
            pose_response = self._retry(
                "get_runtime_vehicle_pose",
                self.client.get_runtime_vehicle_pose,
                vehicle_name,
                map_id=self.map_id,
            )
            pose_payload = dict(pose_response.get("payload") or {})
        except Exception as exc:
            warning_parts.append(f"pose={exc}")

        resolved_target = target_enu_m
        pose_position = pose_payload.get("position_enu_m")
        if resolved_target is None and isinstance(pose_position, Sequence) and not isinstance(pose_position, (str, bytes)):
            resolved_target = [
                float(pose_position[0]),
                float(pose_position[1]),
                float(pose_position[2] if len(pose_position) > 2 else 0.0),
            ]
        if resolved_target is None:
            resolved_target = [0.0, 0.0, 0.0]

        result = self.uav_execution_service.build_wait_status(
            vehicle_name=vehicle_name,
            status_payload=status_payload,
            pose_payload=pose_payload,
            target_enu_m=resolved_target,
            timed_out=False,
            warning="; ".join(warning_parts),
        )
        if warning_parts:
            result["pose_warning"] = "; ".join(warning_parts)
        return result

    @staticmethod
    def _ensure_uav_status_ok(wait_status: dict[str, Any], *, vehicle_name: str, context: str) -> None:
        status_payload = dict(wait_status.get("status") or {})
        state = str(status_payload.get("state") or status_payload.get("status") or "").strip().lower()
        still_running = state in {"running", "moving", "in_progress", "pending", "active"}
        if bool(wait_status.get("timed_out")) and still_running:
            wait_status["timed_out_accepted_running"] = True
            wait_status["warning"] = "UAV command was still running at poll timeout; continuing capture."
            return
        if bool(wait_status.get("timed_out")) or state in {"failed", "cancelled", "timeout", "missing"}:
            raise RuntimeError(
                f"Runtime UAV command failed for {vehicle_name} during {context}: "
                f"state={state or '<empty>'} error={status_payload.get('error') or status_payload.get('message') or ''}"
            )

    def _ensure_uav_pose_ok(self, wait_status: dict[str, Any], *, vehicle_name: str, context: str) -> None:
        if bool(wait_status.get("timed_out")):
            raise RuntimeError(f"Runtime UAV pose wait timed out for {vehicle_name} during {context}")
        position_error_m = wait_status.get("position_error_m")
        if position_error_m is None:
            raise RuntimeError(f"Runtime UAV pose is unavailable for {vehicle_name} during {context}")
        tolerance_m = self._runtime_uav_position_tolerance_m()
        if float(position_error_m) > tolerance_m:
            raise RuntimeError(
                f"Runtime UAV pose error for {vehicle_name} during {context}: "
                f"{float(position_error_m):.3f} m > tolerance {tolerance_m:.3f} m"
            )

    def _wait_for_uav_pose(self, vehicle_name: str, target_enu_m: Sequence[float], *, timeout_s: float | None = None) -> dict[str, Any]:
        timeout_cfg = self._runtime_uav_timeout_cfg()
        pose_timeout_s = float(timeout_s if timeout_s is not None else timeout_cfg.get("uav_spawn_pose_timeout_s", 3.0))
        poll_interval_s = float(timeout_cfg.get("uav_status_poll_interval_s", 0.25))
        deadline = time.perf_counter() + max(0.0, pose_timeout_s)
        last_status = self._uav_status_snapshot(vehicle_name, target_enu_m)
        while True:
            position_error_m = last_status.get("position_error_m")
            if position_error_m is not None and float(position_error_m) <= self._runtime_uav_position_tolerance_m():
                return last_status
            if pose_timeout_s <= 0.0 or time.perf_counter() >= deadline:
                last_status["timed_out"] = bool(position_error_m is None)
                return last_status
            time.sleep(max(0.01, poll_interval_s))
            last_status = self._uav_status_snapshot(vehicle_name, target_enu_m)

    def _sync_uavs(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        current_entities: set[str] = set()
        results: dict[str, Any] = {}
        for entity in frame.get("entities", []):
            resolution = self._entity_resolution(entity)
            if str(resolution.get("mode", "")) != "runtime_multirotor":
                continue
            if truth_submission_state(entity) != "submit_to_ue":
                continue

            entity_id = str(entity["entity_id"])
            vehicle_name = str(resolution.get("vehicle_name") or entity_id)
            if self._airsim_capture_enabled() and entity_id == self.active_airsim_capture_entity_id:
                self._ensure_airsim_capture_vehicle()
                target_enu_m = self._entity_position_enu(entity)
                target_enu_m = self._ground_relative_position(
                    target_enu_m,
                    enabled=bool(self.ground_reference_cfg.get("uav_ground_relative", False)),
                    cache_namespace=f"capture_uav:{self.airsim_capture_vehicle}",
                    use_cache=True,
                )
                rotation_deg = self._entity_rotation_deg(entity)
                wait_status = self._pin_airsim_capture_vehicle(
                    target_enu_m,
                    rotation_deg,
                    context=f"truth-frame sync {entity_id}",
                )
                wait_status["path_used"] = "airsim_native_capture_vehicle"
                wait_status["source_entity_id"] = entity_id
                wait_status["replaces_runtime_multirotor"] = True
                wait_status["status"] = {
                    "state": "ok",
                    "reason": "capture_entity_pinned_to_airsim_vehicle",
                }
                wait_status["capture_gate"] = {
                    "wait_for_arrival": False,
                    "hover_before_capture": False,
                    "arrival_tolerance_m": self._airsim_capture_pose_tolerance_m(),
                    "degraded": False,
                }
                current_entities.add(entity_id)
                results[entity_id] = wait_status
                continue
            if entity_id in self.event_controlled_entity_ids and entity_id in self.uav_active_by_entity:
                vehicle_name = self.uav_active_by_entity.get(entity_id, vehicle_name)
                current_entities.add(entity_id)
                wait_status = self._uav_status_snapshot(vehicle_name)
                wait_status["path_used"] = "direct_rpc"
                wait_status["sync_skipped"] = "event_script_controlled"
                results[entity_id] = wait_status
                continue

            target_enu_m = self._entity_position_enu(entity)
            target_enu_m = self._ground_relative_position(
                target_enu_m,
                enabled=bool(self.ground_reference_cfg.get("uav_ground_relative", False)),
                cache_namespace=f"uav:{vehicle_name}",
                use_cache=True,
            )
            if self._runtime_uav_use_editor_hook() and not self._runtime_uav_visible_in_ground_views(
                str(entity.get("site_id") or self.args.site or ""),
                target_enu_m,
            ):
                continue
            current_entities.add(entity_id)
            rotation_deg = self._entity_rotation_deg(entity)
            velocity_hint = float(((entity.get("annotations") or {}).get("speed_mps")) or resolution.get("velocity_mps") or 5.0)

            if entity_id not in self.uav_active_by_entity:
                used_editor_hook = False
                if not self._runtime_uav_use_editor_hook():
                    try:
                        create_response = self._retry(
                            "create_runtime_multirotor",
                            self.client.create_runtime_multirotor,
                            vehicle_name,
                            position_enu_m=target_enu_m,
                            rotation_deg=rotation_deg,
                            map_id=self.map_id,
                        )
                        move_response = {
                            "payload": {
                                "vehicle_name": vehicle_name,
                                "state": "skipped",
                                "reason": "initial_create_exact_pose",
                                "target_enu_m": [float(value) for value in target_enu_m],
                                "velocity_mps": max(0.5, velocity_hint),
                                "synthetic": True,
                            }
                        }
                    except Exception as exc:
                        self._disable_runtime_uav_direct_rpc(str(exc))
                        print(f"[EpisodeHost] runtime UAV RPC failed for {vehicle_name}; falling back to editor hook: {exc}")
                        create_response, move_response = self._recreate_editor_hook_runtime_uav(
                            vehicle_name=vehicle_name,
                            target_enu_m=target_enu_m,
                            rotation_deg=rotation_deg,
                            velocity_mps=max(0.5, velocity_hint),
                        )
                        used_editor_hook = True
                else:
                    create_response, move_response = self._recreate_editor_hook_runtime_uav(
                        vehicle_name=vehicle_name,
                        target_enu_m=target_enu_m,
                        rotation_deg=rotation_deg,
                        velocity_mps=max(0.5, velocity_hint),
                    )
                    used_editor_hook = True
                if used_editor_hook:
                    try:
                        wait_status = self._wait_for_editor_hook_uav_status(vehicle_name, target_enu_m)
                    except Exception as exc:
                        print(f"[EpisodeHost] UAV actor-probe warning for {vehicle_name} after create: {exc}")
                        wait_status = {
                            "vehicle_name": vehicle_name,
                            "status": {"state": "unknown", "warning": str(exc), "reason": "editor_hook_actor_probe"},
                            "pose": {},
                            "timed_out": False,
                            "position_error_m": None,
                        }
                    wait_status["create"] = create_response.get("payload", {})
                    wait_status["move"] = move_response.get("payload", {})
                    wait_status["path_used"] = "editor_hook"
                    results[entity_id] = wait_status
                else:
                    try:
                        wait_status = self._wait_for_uav_status(vehicle_name, target_enu_m)
                    except Exception as exc:
                        print(f"[EpisodeHost] UAV status warning for {vehicle_name} after create: {exc}")
                        wait_status = self._uav_status_snapshot(vehicle_name, target_enu_m)
                        wait_status["pose_warning"] = str(exc)
                    wait_status["create"] = create_response.get("payload", {})
                    wait_status["move"] = move_response.get("payload", {})
                    wait_status["path_used"] = "direct_rpc"
                    wait_status["truth_sync_pose_deviation_accepted"] = True
                    results[entity_id] = wait_status
                    self.uav_last_command_target_by_entity[entity_id] = list(target_enu_m)
            else:
                used_editor_hook = False
                previous_target = self.uav_last_command_target_by_entity.get(entity_id)
                previous_error_m = distance_m(previous_target, target_enu_m)
                if previous_error_m is not None and previous_error_m <= 0.25:
                    wait_status = self._uav_status_snapshot(vehicle_name, target_enu_m)
                    position_error_m = wait_status.get("position_error_m")
                    if position_error_m is not None and float(position_error_m) <= self._runtime_uav_position_tolerance_m():
                        wait_status["move"] = {
                            "status": "skipped",
                            "reason": "target_unchanged",
                            "previous_error_m": previous_error_m,
                        }
                        wait_status["path_used"] = "direct_rpc" if not self._runtime_uav_use_editor_hook() else "editor_hook"
                        results[entity_id] = wait_status
                        self.uav_active_by_entity[entity_id] = vehicle_name
                        continue
                    wait_status["move"] = {
                        "status": "skipped",
                        "reason": "truth_sync_target_unchanged_pose_deviation_accepted",
                        "previous_error_m": previous_error_m,
                        "position_error_m": position_error_m,
                    }
                    wait_status["truth_sync_pose_deviation_accepted"] = True
                    wait_status["path_used"] = "direct_rpc" if not self._runtime_uav_use_editor_hook() else "editor_hook"
                    results[entity_id] = wait_status
                    self.uav_active_by_entity[entity_id] = vehicle_name
                    continue

                if not self._runtime_uav_use_editor_hook():
                    try:
                        move_response = self._retry(
                            "move_runtime_multirotor",
                            self.client.move_runtime_multirotor,
                            vehicle_name,
                            target_enu_m=target_enu_m,
                            velocity_mps=max(0.5, velocity_hint),
                            map_id=self.map_id,
                        )
                    except Exception as exc:
                        self._disable_runtime_uav_direct_rpc(str(exc))
                        print(f"[EpisodeHost] runtime UAV move RPC failed for {vehicle_name}; falling back to editor hook: {exc}")
                        create_response, move_response = self._recreate_editor_hook_runtime_uav(
                            vehicle_name=vehicle_name,
                            target_enu_m=target_enu_m,
                            rotation_deg=rotation_deg,
                            velocity_mps=max(0.5, velocity_hint),
                        )
                        used_editor_hook = True
                else:
                    create_response, move_response = self._recreate_editor_hook_runtime_uav(
                        vehicle_name=vehicle_name,
                        target_enu_m=target_enu_m,
                        rotation_deg=rotation_deg,
                        velocity_mps=max(0.5, velocity_hint),
                    )
                    used_editor_hook = True
                if used_editor_hook:
                    try:
                        wait_status = self._wait_for_editor_hook_uav_status(vehicle_name, target_enu_m)
                    except Exception as exc:
                        print(f"[EpisodeHost] UAV actor-probe warning for {vehicle_name} after move: {exc}")
                        wait_status = {
                            "vehicle_name": vehicle_name,
                            "status": {"state": "unknown", "warning": str(exc), "reason": "editor_hook_actor_probe"},
                            "pose": {},
                            "timed_out": False,
                            "position_error_m": None,
                        }
                    wait_status["create"] = create_response.get("payload", {})
                    wait_status["move"] = move_response.get("payload", {})
                    wait_status["path_used"] = "editor_hook"
                    results[entity_id] = wait_status
                else:
                    try:
                        wait_status = self._wait_for_uav_status(vehicle_name, target_enu_m)
                        self._ensure_uav_status_ok(wait_status, vehicle_name=vehicle_name, context="truth-frame sync")
                    except Exception as exc:
                        print(f"[EpisodeHost] UAV status warning for {vehicle_name} after move: {exc}")
                        raise
                    wait_status["move"] = move_response.get("payload", {})
                    wait_status["path_used"] = "direct_rpc"
                    results[entity_id] = wait_status
                    self.uav_last_command_target_by_entity[entity_id] = list(target_enu_m)

            self.uav_active_by_entity[entity_id] = vehicle_name

        for entity_id in sorted(set(self.uav_active_by_entity) - current_entities):
            vehicle_name = self.uav_active_by_entity.pop(entity_id)
            self.uav_last_command_target_by_entity.pop(entity_id, None)
            if not self._runtime_uav_use_editor_hook():
                try:
                    self._retry("remove_runtime_vehicle", self.client.remove_runtime_vehicle, vehicle_name, map_id=self.map_id)
                    continue
                except Exception as exc:
                    self._disable_runtime_uav_direct_rpc(str(exc))
                    print(f"[EpisodeHost] remove_runtime_vehicle RPC failed for {vehicle_name}; falling back to editor hook: {exc}")
            try:
                self._runtime_uav_editor_hook().remove_runtime_vehicle(map_id=self.map_id, vehicle_name=vehicle_name)
            except Exception as hook_exc:
                print(f"[EpisodeHost] remove_runtime_vehicle warning for {vehicle_name}: {hook_exc}")
            self._force_destroy_runtime_vehicle_actors([vehicle_name])

        return results

    def _poll_feedback(self) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        response = self._retry(
            "poll_feedback",
            self.client.poll_feedback,
            since_tick=self.feedback_watermark_tick if self.feedback_watermark_tick >= 0 else None,
            map_id=self.map_id,
        )
        payload = dict(response.get("payload") or {})
        upto_tick = payload.get("upto_tick")
        if isinstance(upto_tick, (int, float)):
            self.feedback_watermark_tick = int(upto_tick)
        return payload

    def _site_ground_presets(self, site_id: str) -> list[dict[str, Any]]:
        ground = dict(self.capture_presets.get("ground_cameras") or {})
        presets = [dict(item) for item in (ground.get(site_id) or ground.get("default") or [])]
        return [self._scene_aligned_ground_preset(preset) for preset in presets]

    @staticmethod
    def _maybe_vector3(value: Any) -> list[float] | None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
            return None
        try:
            return [
                float(value[0]),
                float(value[1]),
                float(value[2] if len(value) > 2 else 0.0),
            ]
        except (TypeError, ValueError):
            return None

    def _scenario_capture_bbox_enu_m(self) -> list[float] | None:
        if self.event_capture_bbox_enu_m is not None:
            return list(self.event_capture_bbox_enu_m)

        points: list[list[float]] = []

        def add_point(value: Any, *, transform_script_position: bool = False) -> None:
            point = self._maybe_vector3(value)
            if point is None:
                return
            if transform_script_position:
                try:
                    point = self._script_transform_position(point)
                except Exception:
                    pass
            points.append(point)

        summary = dict(self.scenario_plan.get("compiled_plan_summary") or {})
        roi_windows = dict(summary.get("roi_windows") or {})
        for value in roi_windows.values():
            if not isinstance(value, dict):
                continue
            bbox = list(value.get("bbox_enu_m") or [])
            if len(bbox) >= 4:
                points.append([float(bbox[0]), float(bbox[1]), 0.0])
                points.append([float(bbox[2]), float(bbox[3]), 0.0])

        for entity in self.global_roster:
            add_point(entity.get("initial_position_enu_m"))

        for entity in self.event_scene_setup.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            placement = dict(entity.get("placement") or {})
            for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
                add_point(placement.get(key))
            for waypoint in entity.get("route_waypoints_enu_m") or []:
                add_point(waypoint)
            for vertex in placement.get("polygon_enu_m") or []:
                add_point(vertex)

        event_script = dict(getattr(self.event_interpreter, "script", {}) or {})
        for event in event_script.get("events") or []:
            if not isinstance(event, dict):
                continue
            for action in event.get("actions") or []:
                if not isinstance(action, dict):
                    continue
                add_point(action.get("position_enu_m"), transform_script_position=True)
                add_point(action.get("spawn_origin_enu_m"), transform_script_position=True)
                for waypoint in action.get("waypoints_enu_m") or action.get("waypoints") or []:
                    add_point(waypoint, transform_script_position=True)

        if not points:
            return None

        min_x = min(float(point[0]) for point in points)
        min_y = min(float(point[1]) for point in points)
        max_x = max(float(point[0]) for point in points)
        max_y = max(float(point[1]) for point in points)
        max_z = max(float(point[2] if len(point) > 2 else 0.0) for point in points)
        margin_m = 8.0
        self.event_capture_bbox_enu_m = [
            min_x - margin_m,
            min_y - margin_m,
            max_x + margin_m,
            max_y + margin_m,
            max_z,
        ]
        return list(self.event_capture_bbox_enu_m)

    def _scene_aligned_ground_preset(self, preset: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(preset.get("camera_id", preset.get("camera_name", "")))
        if not camera_id.lower().endswith("overview_top"):
            return preset

        bbox = self._scenario_capture_bbox_enu_m()
        if not bbox or len(bbox) < 4:
            return preset

        min_x = float(bbox[0])
        min_y = float(bbox[1])
        max_x = float(bbox[2])
        max_y = float(bbox[3])
        max_z = float(bbox[4] if len(bbox) > 4 else 0.0)
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        span_x_m = max(max_x - min_x, 1.0)
        span_y_m = max(max_y - min_y, 1.0)

        width = max(1, int(preset.get("width") or 1280))
        height = max(1, int(preset.get("height") or 720))
        fov_degrees = max(float(preset.get("fov_degrees") or 70.0), 105.0)
        hfov_rad = math.radians(fov_degrees)
        vfov_rad = 2.0 * math.atan(math.tan(hfov_rad * 0.5) / (float(width) / float(height)))
        required_dz_m = max(
            span_x_m / max(0.1, 2.0 * math.tan(hfov_rad * 0.5)),
            span_y_m / max(0.1, 2.0 * math.tan(vfov_rad * 0.5)),
        )
        altitude_m = max(max_z + required_dz_m * 1.2, max_z + 25.0, 45.0)
        altitude_m = min(altitude_m, 160.0)

        aligned = dict(preset)
        aligned["position_enu_m"] = [round(center_x, 3), round(center_y, 3), round(altitude_m, 3)]
        aligned["rotation_deg"] = {"pitch_deg": -90.0, "yaw_deg": 0.0, "roll_deg": 0.0}
        aligned["fov_degrees"] = fov_degrees
        aligned["coordinate_space"] = "map_enu"
        aligned["scene_aligned"] = True
        aligned["scene_bbox_enu_m"] = [round(min_x, 3), round(min_y, 3), round(max_x, 3), round(max_y, 3)]
        aligned["scene_bbox_source"] = "scenario_plan_scene_setup_event_script"
        return aligned

    def _camera_position_from_preset(self, preset: dict[str, Any]) -> list[float]:
        raw_position = [float(value) for value in (preset.get("position_enu_m") or [0.0, 0.0, 0.0])]
        coordinate_space = str(preset.get("coordinate_space") or "").strip().lower()
        if coordinate_space in {"map_enu", "world_enu", "transformed_enu"}:
            return raw_position
        return self._transform_position_enu(raw_position)

    def _camera_rotation_from_preset(self, preset: dict[str, Any]) -> dict[str, float]:
        raw_rotation = dict(preset.get("rotation_deg") or {})
        coordinate_space = str(preset.get("coordinate_space") or "").strip().lower()
        if coordinate_space in {"map_enu", "world_enu", "transformed_enu"}:
            return {
                "pitch_deg": float(raw_rotation.get("pitch_deg", raw_rotation.get("pitch", 0.0))),
                "yaw_deg": float(raw_rotation.get("yaw_deg", raw_rotation.get("yaw", 0.0))),
                "roll_deg": float(raw_rotation.get("roll_deg", raw_rotation.get("roll", 0.0))),
            }
        return self._transform_rotation_deg(raw_rotation)

    def _runtime_uav_visibility_filter_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("runtime_uav_visibility_filter") or {})

    def _ground_camera_contains_position(self, preset: dict[str, Any], position_enu_m: Sequence[float]) -> bool:
        camera_position = self._camera_position_from_preset(preset)
        if len(camera_position) < 3 or len(position_enu_m) < 3:
            return True
        rotation = dict(preset.get("rotation_deg") or {})
        pitch_deg = float(rotation.get("pitch_deg", 0.0))
        if abs(pitch_deg + 90.0) > 1.0:
            return True

        camera_height_m = float(camera_position[2])
        target_height_m = float(position_enu_m[2])
        dz = camera_height_m - target_height_m
        if dz <= 0.0:
            return False

        width = max(1, int(preset.get("width") or 1920))
        height = max(1, int(preset.get("height") or 1080))
        hfov_rad = math.radians(float(preset.get("fov_degrees") or 60.0))
        vfov_rad = 2.0 * math.atan(math.tan(hfov_rad * 0.5) / (float(width) / float(height)))
        margin_m = max(0.0, float(self._runtime_uav_visibility_filter_cfg().get("margin_m", 0.0)))

        dx = float(position_enu_m[0]) - float(camera_position[0])
        dy = float(position_enu_m[1]) - float(camera_position[1])
        half_width_m = dz * math.tan(hfov_rad * 0.5) + margin_m
        half_height_m = dz * math.tan(vfov_rad * 0.5) + margin_m
        return abs(dx) <= half_width_m and abs(dy) <= half_height_m

    def _runtime_uav_visible_in_ground_views(self, site_id: str, position_enu_m: Sequence[float]) -> bool:
        visibility_cfg = self._runtime_uav_visibility_filter_cfg()
        if not bool(visibility_cfg.get("enabled", False)):
            return True
        presets = self._site_ground_presets(self.args.site or site_id)
        if not presets:
            return True
        return any(self._ground_camera_contains_position(preset, position_enu_m) for preset in presets)

    def _uav_camera_presets(self) -> list[dict[str, Any]]:
        uav = dict(self.capture_presets.get("uav_cameras") or {})
        return [dict(item) for item in (uav.get("default") or [])]

    def _frame_active_runtime_uavs(self, frame: dict[str, Any], *, site_id: str = "") -> list[str]:
        ids: list[str] = []
        for entity in frame.get("entities") or []:
            entity_id = str(entity.get("entity_id") or "")
            if not entity_id:
                continue
            if str(entity.get("entity_category") or "").strip().lower() != "uav":
                continue
            if site_id and str(entity.get("site_id") or "") not in {"", site_id}:
                continue
            if truth_submission_state(entity) != "submit_to_ue":
                continue
            if str(self._entity_resolution(entity).get("mode") or "") != "runtime_multirotor":
                continue
            ids.append(entity_id)
        return sorted(set(ids))

    def _select_airsim_capture_entity(self, batch: BatchPlan, capture_ticks: set[int]) -> None:
        self.active_airsim_capture_entity_id = ""
        self.active_capture_view_id = ""
        if not self._airsim_capture_enabled():
            return
        explicit = self.requested_airsim_capture_entity
        candidate_ticks = sorted(capture_ticks) if capture_ticks else [batch.tick_start]
        for tick in candidate_ticks:
            frame = self.frames_by_tick.get(int(tick))
            if not frame:
                continue
            active_ids = self._frame_active_runtime_uavs(frame, site_id=batch.site_id)
            if explicit:
                if explicit in active_ids:
                    self.active_airsim_capture_entity_id = explicit
                    break
            elif active_ids:
                self.active_airsim_capture_entity_id = active_ids[0]
                break
        if explicit and not self.active_airsim_capture_entity_id:
            raise RuntimeError(
                f"Requested --airsim-capture-entity '{explicit}' is not an active runtime UAV in batch {batch.batch_id}."
            )
        if not self.active_airsim_capture_entity_id:
            raise RuntimeError(f"AirSim native UAV capture requested but no active runtime UAV exists in batch {batch.batch_id}.")
        self.active_capture_view_id = self.requested_capture_view_id or f"uav_view_000__{safe_name(self.active_airsim_capture_entity_id)}"
        print(
            "[EpisodeHost] AirSim native capture source "
            f"entity={self.active_airsim_capture_entity_id} vehicle={self.airsim_capture_vehicle} "
            f"view_id={self.active_capture_view_id}"
        )

    def _capture_role_enabled(self, role: str) -> bool:
        role_name = str(role).strip().lower()
        return not self.capture_role_filters or role_name in self.capture_role_filters

    def _capture_camera_enabled(self, *candidate_ids: str) -> bool:
        if not self.capture_camera_filters:
            return True
        candidates = {str(value).strip() for value in candidate_ids if str(value).strip()}
        return bool(candidates & self.capture_camera_filters)

    def _filtered_modalities(self, modality_ids: Sequence[str]) -> list[str]:
        ordered = [str(value).strip() for value in modality_ids if str(value).strip()]
        if not self.capture_modality_filters:
            return ordered
        return [value for value in ordered if value in self.capture_modality_filters]

    def _fixed_world_capture_hook(self) -> FixedWorldCaptureEditorHook:
        if self.fixed_world_capture_hook is not None:
            return self.fixed_world_capture_hook
        timeout_cfg = dict(self.config.get("timeouts") or {})
        self.fixed_world_capture_hook = FixedWorldCaptureEditorHook(
            project_root=PROJECT_ROOT,
            discovery_timeout_s=float(timeout_cfg.get("editor_hook_discovery_timeout_s", 10.0)),
            capture_timeout_s=float(timeout_cfg.get("editor_hook_capture_timeout_s", 15.0)),
        )
        return self.fixed_world_capture_hook

    def _runtime_uav_editor_hook(self) -> FixedWorldCaptureEditorHook:
        return self._fixed_world_capture_hook()

    def _disable_runtime_uav_direct_rpc(self, reason: str) -> None:
        normalized_reason = " ".join(str(reason).split())
        if self.runtime_uav_direct_rpc_enabled:
            print(
                "[EpisodeHost] Disabling direct runtime UAV RPC for this session; "
                f"forcing editor hook fallback. reason={normalized_reason}"
            )
        self.runtime_uav_direct_rpc_enabled = False
        self.runtime_uav_direct_rpc_disable_reason = normalized_reason

    def _runtime_uav_use_editor_hook(self) -> bool:
        return not self.runtime_uav_direct_rpc_enabled

    def _runtime_uav_timeout_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("timeouts") or {})

    def _runtime_uav_position_tolerance_m(self) -> float:
        timeout_cfg = self._runtime_uav_timeout_cfg()
        return max(0.1, float(timeout_cfg.get("uav_position_tolerance_m", 2.0)))

    def _force_destroy_runtime_vehicle_actors(self, vehicle_names: Sequence[str]) -> dict[str, Any] | None:
        names = sorted({str(value).strip() for value in vehicle_names if str(value).strip()})
        if not names:
            return None
        return self._best_effort(
            "destroy_runtime_vehicle_actors",
            self._runtime_uav_editor_hook().destroy_runtime_vehicle_actors,
            vehicle_names=names,
        )

    def _runtime_uav_should_debug(self, entity_id: str) -> bool:
        if not self.runtime_uav_debug_entity_ids:
            return True
        return str(entity_id).strip() in self.runtime_uav_debug_entity_ids

    @staticmethod
    def _runtime_uav_path_used(status_entry: dict[str, Any]) -> str:
        status_reason = str((status_entry.get("status") or {}).get("reason") or "").strip().lower()
        if "editor_hook" in status_reason:
            return "editor_hook"
        for key in ("create", "move"):
            via = str((status_entry.get(key) or {}).get("via") or "").strip().lower()
            if via:
                return via
        return "direct_rpc"

    def _runtime_uav_command_debug(
        self,
        entity: dict[str, Any],
        *,
        vehicle_name: str,
        command_target_enu_m: Sequence[float],
        command_rotation_deg: dict[str, Any],
        command_velocity_mps: float,
        ground_details: dict[str, Any] | None,
        snap_details: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw_truth_position_enu_m = position_enu_from_truth(entity)
        raw_truth_rotation_deg = rotation_dict_from_truth(entity)
        transformed_position_enu_m, transformed_rotation_deg = self._transformed_entity_pose(entity)
        resolved_position_enu_m, resolved_rotation_deg, _ = self._resolve_entity_pose(entity)
        return {
            "vehicle_name": vehicle_name,
            "raw_truth_position_enu_m": raw_truth_position_enu_m,
            "raw_truth_rotation_deg": raw_truth_rotation_deg,
            "transformed_target_enu_m": transformed_position_enu_m,
            "transformed_rotation_deg": transformed_rotation_deg,
            "resolved_target_enu_m": resolved_position_enu_m,
            "resolved_rotation_deg": resolved_rotation_deg,
            "command_payload_target_enu_m": [float(value) for value in command_target_enu_m],
            "command_payload_rotation_deg": {
                "pitch_deg": float(command_rotation_deg.get("pitch_deg", 0.0)),
                "yaw_deg": float(command_rotation_deg.get("yaw_deg", 0.0)),
                "roll_deg": float(command_rotation_deg.get("roll_deg", 0.0)),
            },
            "command_velocity_mps": float(command_velocity_mps),
            "uav_ground_relative_enabled": bool(self.ground_reference_cfg.get("uav_ground_relative", False)),
            "ground_projection": dict(ground_details or {}),
            "pose_adjustment": dict(snap_details or {}),
        }

    def _probe_runtime_uav_actors(
        self,
        by_entity: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not by_entity:
            return {
                "vehicles": {},
                "probe_meta": {"vehicle_count": 0, "pawn_count": 0, "flying_pawns": []},
            }

        vehicle_names = sorted(
            {
                str(entry.get("vehicle_name") or "")
                for entry in by_entity.values()
                if str(entry.get("vehicle_name") or "").strip()
            }
        )
        if not vehicle_names:
            return {
                "vehicles": {},
                "probe_meta": {"vehicle_count": 0, "pawn_count": 0, "flying_pawns": []},
            }

        try:
            payload = self._runtime_uav_editor_hook().inspect_runtime_vehicle_actors(
                vehicle_names=vehicle_names,
                world_origin_cm=[float(value) for value in self.world_origin_cm],
            )
        except Exception as exc:
            error_text = str(exc)
            vehicles = {
                vehicle_name: {
                    "found": False,
                    "actor": None,
                    "name_candidates": [],
                    "probe_error": error_text,
                }
                for vehicle_name in vehicle_names
            }
            return {
                "vehicles": vehicles,
                "probe_meta": {"vehicle_count": len(vehicle_names), "pawn_count": 0, "flying_pawns": [], "probe_error": error_text},
            }

        vehicles = dict(payload.get("vehicles") or {})
        return {
            "vehicles": vehicles,
            "probe_meta": {
                "vehicle_count": int(payload.get("vehicle_count") or len(vehicle_names)),
                "pawn_count": int(payload.get("pawn_count") or 0),
                "flying_pawns": list(payload.get("flying_pawns") or []),
            },
        }

    def _probe_runtime_uav_actor(self, vehicle_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        probe_payload = self._probe_runtime_uav_actors({"__probe__": {"vehicle_name": vehicle_name}})
        return (
            dict((probe_payload.get("vehicles") or {}).get(vehicle_name) or {}),
            dict(probe_payload.get("probe_meta") or {}),
        )

    def _wait_for_editor_hook_uav_status(self, vehicle_name: str, target_enu_m: Sequence[float]) -> dict[str, Any]:
        timeout_cfg = self._runtime_uav_timeout_cfg()
        timeout_s = float(timeout_cfg.get("uav_move_timeout_s", 10.0))
        poll_interval_s = float(timeout_cfg.get("uav_status_poll_interval_s", 0.25))
        tolerance_m = self._runtime_uav_position_tolerance_m()
        deadline = time.perf_counter() + max(0.0, timeout_s)

        last_probe: dict[str, Any] = {}
        last_probe_meta: dict[str, Any] = {}
        last_error_m: float | None = None
        timed_out = False
        state = "probe_only"

        while True:
            last_probe, last_probe_meta = self._probe_runtime_uav_actor(vehicle_name)
            actor_row = dict(last_probe.get("actor") or {})
            actor_position_enu_m = actor_row.get("position_enu_m")
            last_error_m = distance_m(
                target_enu_m,
                actor_position_enu_m if isinstance(actor_position_enu_m, list) else None,
            )
            if bool(last_probe.get("found")) and last_error_m is not None and last_error_m <= tolerance_m:
                state = "succeeded"
                break
            if timeout_s <= 0.0:
                state = "probe_only"
                break
            if time.perf_counter() >= deadline:
                timed_out = True
                state = "timeout" if bool(last_probe.get("found")) else "missing"
                break
            time.sleep(poll_interval_s)

        actor_row = dict(last_probe.get("actor") or {})
        actor_position_enu_m = actor_row.get("position_enu_m")
        actor_rotation_deg = actor_row.get("rotation_deg")
        pose_payload: dict[str, Any] = {}
        if isinstance(actor_position_enu_m, list):
            pose_payload["position_enu_m"] = list(actor_position_enu_m)
        if isinstance(actor_rotation_deg, dict):
            pose_payload["rotation_deg"] = dict(actor_rotation_deg)

        probe_warning = str(last_probe.get("probe_error") or "")
        status_payload: dict[str, Any] = {
            "state": state,
            "reason": "editor_hook_actor_probe",
            "position_tolerance_m": tolerance_m,
        }
        if "position_enu_m" in pose_payload:
            status_payload["current_enu_m"] = list(pose_payload["position_enu_m"])
        if probe_warning:
            status_payload["warning"] = probe_warning

        result = self.uav_execution_service.build_wait_status(
            vehicle_name=vehicle_name,
            status_payload=status_payload,
            pose_payload=pose_payload,
            target_enu_m=target_enu_m,
            timed_out=timed_out,
            warning=probe_warning,
            reason="editor_hook_actor_probe",
        )
        result["pose_warning"] = probe_warning
        result["actor_probe"] = last_probe
        result["probe_meta"] = last_probe_meta
        result["position_error_m"] = last_error_m
        return result

    def _recreate_editor_hook_runtime_uav(
        self,
        *,
        vehicle_name: str,
        target_enu_m: Sequence[float],
        rotation_deg: dict[str, Any],
        velocity_mps: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        hook = self._runtime_uav_editor_hook()
        self._best_effort(
            "remove_runtime_vehicle_editor_hook",
            hook.remove_runtime_vehicle,
            map_id=self.map_id,
            vehicle_name=vehicle_name,
        )
        self._force_destroy_runtime_vehicle_actors([vehicle_name])
        create_response = hook.create_runtime_multirotor(
            map_id=self.map_id,
            vehicle_name=vehicle_name,
            position_enu_m=[float(value) for value in target_enu_m],
            rotation_deg={key: float(value) for key, value in rotation_deg.items()},
        )
        move_response = {
            "payload": {
                "vehicle_name": vehicle_name,
                "state": "skipped",
                "reason": "editor_hook_recreate_exact_pose",
                "target_enu_m": [float(value) for value in target_enu_m],
                "velocity_mps": float(velocity_mps),
                "via": "editor_hook_recreate",
                "synthetic": True,
            }
        }
        return create_response, move_response

    def _collect_runtime_uav_debug(
        self,
        frame: dict[str, Any],
        uav_status: dict[str, Any],
    ) -> dict[str, Any]:
        debug_by_entity: dict[str, dict[str, Any]] = {}
        vehicle_name_to_entity_id: dict[str, str] = {}

        for entity in frame.get("entities", []):
            resolution = self._entity_resolution(entity)
            if str(resolution.get("mode", "")) != "runtime_multirotor":
                continue
            if truth_submission_state(entity) != "submit_to_ue":
                continue

            entity_id = str(entity.get("entity_id") or "")
            if not self._runtime_uav_should_debug(entity_id):
                continue

            status_entry = dict(uav_status.get(entity_id) or {})
            if not status_entry:
                continue
            vehicle_name = str(status_entry.get("vehicle_name") or resolution.get("vehicle_name") or entity_id)
            transformed_position_enu_m, transformed_rotation_deg = self._transformed_entity_pose(entity)
            resolved_position_enu_m, resolved_rotation_deg, snap_details = self._resolve_entity_pose(entity)
            ground_details = self._project_ground_details(
                resolved_position_enu_m,
                cache_namespace=f"uav:{vehicle_name}",
                use_cache=True,
            ) if bool(self.ground_reference_cfg.get("uav_ground_relative", False)) else None
            command_target_enu_m = list((ground_details or {}).get("ground_relative_enu_m") or resolved_position_enu_m)
            command_velocity_mps = float(((entity.get("annotations") or {}).get("speed_mps")) or resolution.get("velocity_mps") or 5.0)

            rpc_status_payload = dict(status_entry.get("status") or {})
            rpc_pose_payload = dict(status_entry.get("pose") or {})
            rpc_status_current_enu_m = rpc_status_payload.get("current_enu_m")
            rpc_pose_enu_m = rpc_pose_payload.get("position_enu_m")

            debug_entry = self._runtime_uav_command_debug(
                entity,
                vehicle_name=vehicle_name,
                command_target_enu_m=command_target_enu_m,
                command_rotation_deg=resolved_rotation_deg,
                command_velocity_mps=command_velocity_mps,
                ground_details=ground_details,
                snap_details=snap_details,
            )
            debug_entry.update(
                {
                    "path_used": self._runtime_uav_path_used(status_entry),
                    "rpc_create_payload": dict(status_entry.get("create") or {}),
                    "rpc_move_payload": dict(status_entry.get("move") or {}),
                    "rpc_status_payload": rpc_status_payload,
                    "rpc_pose_payload": rpc_pose_payload,
                    "rpc_status_current_enu_m": list(rpc_status_current_enu_m) if isinstance(rpc_status_current_enu_m, list) else rpc_status_current_enu_m,
                    "rpc_pose_enu_m": list(rpc_pose_enu_m) if isinstance(rpc_pose_enu_m, list) else rpc_pose_enu_m,
                    "position_error_m": status_entry.get("position_error_m"),
                    "timed_out": bool(status_entry.get("timed_out", False)),
                    "pose_warning": str(status_entry.get("pose_warning") or ""),
                }
            )
            debug_by_entity[entity_id] = debug_entry
            vehicle_name_to_entity_id[vehicle_name] = entity_id

        probe_payload = self._probe_runtime_uav_actors(debug_by_entity)
        vehicle_probe_map = dict(probe_payload.get("vehicles") or {})
        probe_meta = dict(probe_payload.get("probe_meta") or {})

        for vehicle_name, entity_id in vehicle_name_to_entity_id.items():
            debug_entry = debug_by_entity.get(entity_id)
            if debug_entry is None:
                continue
            actor_probe = dict(vehicle_probe_map.get(vehicle_name) or {})
            actor_row = dict(actor_probe.get("actor") or {})
            actor_position_enu_m = actor_row.get("position_enu_m")
            actor_rotation_deg = actor_row.get("rotation_deg")
            debug_entry["actor_probe"] = actor_probe
            debug_entry["actor_position_enu_m"] = list(actor_position_enu_m) if isinstance(actor_position_enu_m, list) else actor_position_enu_m
            debug_entry["actor_rotation_deg"] = dict(actor_rotation_deg or {})
            debug_entry["actor_vs_command_error_m"] = distance_m(
                debug_entry.get("command_payload_target_enu_m"),
                actor_position_enu_m if isinstance(actor_position_enu_m, list) else None,
            )
            debug_entry["actor_vs_truth_error_m"] = distance_m(
                debug_entry.get("raw_truth_position_enu_m"),
                actor_position_enu_m if isinstance(actor_position_enu_m, list) else None,
            )
            debug_entry["rpc_pose_vs_actor_error_m"] = distance_m(
                debug_entry.get("rpc_pose_enu_m") if isinstance(debug_entry.get("rpc_pose_enu_m"), list) else None,
                actor_position_enu_m if isinstance(actor_position_enu_m, list) else None,
            )
            debug_entry["rpc_status_vs_actor_error_m"] = distance_m(
                debug_entry.get("rpc_status_current_enu_m") if isinstance(debug_entry.get("rpc_status_current_enu_m"), list) else None,
                actor_position_enu_m if isinstance(actor_position_enu_m, list) else None,
            )

        return {
            "vehicles": debug_by_entity,
            "probe_meta": probe_meta,
        }

    def _cleanup_pie_world_actors(self) -> None:
        cfg = dict(self.pie_scene_cleanup_cfg or {})
        if not bool(cfg.get("enabled", False)):
            return
        destroy_class_names = [str(value).strip() for value in (cfg.get("destroy_actor_class_names") or []) if str(value).strip()]
        destroy_class_prefixes = [str(value).strip() for value in (cfg.get("destroy_actor_class_prefixes") or []) if str(value).strip()]
        destroy_name_prefixes = [str(value).strip().lower() for value in (cfg.get("destroy_actor_name_prefixes") or []) if str(value).strip()]
        preserve_names = {
            str(value).strip().lower()
            for value in (cfg.get("preserve_actor_names") or [])
            if str(value).strip()
        }
        if self.airsim_capture_vehicle:
            preserve_names.add(str(self.airsim_capture_vehicle).strip().lower())
        class_prefixes = [str(value).strip() for value in (cfg.get("hide_actor_class_prefixes") or []) if str(value).strip()]
        name_keywords = [str(value).strip().lower() for value in (cfg.get("hide_actor_name_keywords") or []) if str(value).strip()]
        skip_prefixes = [str(value).strip() for value in (cfg.get("skip_actor_class_prefixes") or []) if str(value).strip()]
        if not destroy_class_names and not destroy_class_prefixes and not destroy_name_prefixes and not class_prefixes and not name_keywords:
            return

        request = {
            "destroy_class_names": destroy_class_names,
            "destroy_class_prefixes": destroy_class_prefixes,
            "destroy_name_prefixes": destroy_name_prefixes,
            "preserve_names": sorted(preserve_names),
            "class_prefixes": class_prefixes,
            "name_keywords": name_keywords,
            "skip_prefixes": skip_prefixes,
        }
        request_text = json.dumps(request, separators=(",", ":"), ensure_ascii=True)
        python_command = f"""
import json
import unreal

cfg = json.loads({request_text!r})
worlds = unreal.EditorLevelLibrary.get_pie_worlds(False)
if not worlds:
    raise RuntimeError("No PIE world available for scene cleanup.")
world = worlds[0]
actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
destroyed = []
hidden = []
for actor in actors:
    cls = actor.get_class().get_name()
    if any(cls.startswith(prefix) for prefix in cfg.get("skip_prefixes", [])):
        continue
    name = actor.get_name()
    label = ""
    try:
        label = actor.get_actor_label()
    except Exception:
        label = ""
    identity = set([name.lower(), label.lower()])
    if identity & set(cfg.get("preserve_names", [])):
        continue
    hay = (name + " " + label + " " + cls).lower()
    destroy_match = cls in set(cfg.get("destroy_class_names", []))
    if not destroy_match:
        destroy_match = any(cls.startswith(prefix) for prefix in cfg.get("destroy_class_prefixes", []))
    if not destroy_match:
        destroy_match = any(name.lower().startswith(prefix) or label.lower().startswith(prefix) for prefix in cfg.get("destroy_name_prefixes", []))
    if destroy_match:
        try:
            actor.destroy_actor()
            destroyed.append({{"name": name, "label": label, "class": cls}})
        except Exception:
            pass
        continue
    matched = any(cls.startswith(prefix) for prefix in cfg.get("class_prefixes", []))
    if not matched:
        matched = any(keyword in hay for keyword in cfg.get("name_keywords", []))
    if not matched:
        continue
    try:
        actor.set_actor_hidden_in_game(True)
    except Exception:
        pass
    try:
        actor.set_actor_enable_collision(False)
    except Exception:
        pass
    try:
        actor.set_actor_tick_enabled(False)
    except Exception:
        pass
    hidden.append({{"name": name, "label": label, "class": cls}})
for item in destroyed[:100]:
    print("PIE_SCENE_DESTROY", item)
print("PIE_SCENE_DESTROY_COUNT", len(destroyed))
for item in hidden[:100]:
    print("PIE_SCENE_CLEANUP", item)
print("PIE_SCENE_CLEANUP_COUNT", len(hidden))
"""
        result = self._fixed_world_capture_hook().remote.run_python(
            python_command,
            unattended=False,
            raise_on_failure=True,
        )
        for item in result.get("output") or []:
            text = str(item.get("output") or "").rstrip()
            if text:
                print(f"[EpisodeHost] {text}")

    def _ground_camera_asset_id(self, site_id: str, camera_id: str) -> str:
        key = (str(site_id), str(camera_id))
        if key not in self.ground_camera_asset_ids:
            self.ground_camera_asset_ids[key] = safe_name(f"fixed_world_camera.{site_id}.{camera_id}")
        return self.ground_camera_asset_ids[key]

    def _prepare_capture_output_dir(self, output_dir: Path) -> None:
        resolved_output = output_dir.resolve()
        resolved_root = self.output_dir.resolve()
        try:
            resolved_output.relative_to(resolved_root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to clean capture output outside output_dir: {resolved_output}") from exc
        if resolved_output in self.prepared_capture_output_dirs:
            ensure_dir(resolved_output)
            return
        if resolved_output.exists():
            shutil.rmtree(resolved_output)
        ensure_dir(resolved_output)
        self.prepared_capture_output_dirs.add(resolved_output)

    def _relative_to_output_root(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.output_dir.resolve()))
        except ValueError:
            return str(path)

    def _capture_storage_payload(
        self,
        *,
        batch: BatchPlan,
        view_id: str,
        modality: str,
        modality_output_dir: Path,
        primary_output_path: Path,
        camera_role: str,
    ) -> dict[str, Any]:
        normalized_modality = safe_name(str(modality or "rgb").strip().lower())
        return {
            "storage_layout_version": "capture_storage_v1",
            "storage_rule": "<episode_output_root>/<batch_id>/<capture_view_id>/<modality>/<frame_stem>.<ext>",
            "episode_output_root": str(self.output_dir),
            "batch_id": batch.batch_id,
            "batch_output_dir": str(self.output_dir / batch.batch_id),
            "capture_view_id": view_id,
            "capture_view_output_dir": str(self.output_dir / batch.batch_id / safe_name(view_id)),
            "camera_role": camera_role,
            "channel_id": normalized_modality,
            "modality": normalized_modality,
            "modality_output_dir": str(modality_output_dir),
            "primary_output_path": str(primary_output_path),
            "relative_modality_output_dir": self._relative_to_output_root(modality_output_dir),
            "relative_primary_output_path": self._relative_to_output_root(primary_output_path),
            "deterministic_overwrite_scope": "modality_output_dir",
            "single_camera_single_modality_capture": True,
        }

    @staticmethod
    def semantic_segmentation_classes() -> list[dict[str, Any]]:
        return [dict(item) for item in SEMANTIC_SEGMENTATION_CLASSES]

    @staticmethod
    def _load_semantic_class_by_id(rules_path: Path) -> dict[str, str]:
        fallback = {
            "0": "ignore",
            "1": "city_base_background",
            "2": "building",
            "3": "vegetation",
            "4": "water",
            "5": "vehicle",
            "6": "pedestrian",
            "7": "drone",
            "8": "obstacle",
            "9": "traffic_control",
            "10": "facility",
            "11": "hazard_trigger",
        }
        try:
            root = json.loads(Path(rules_path).read_text(encoding="utf-8-sig"))
        except Exception:
            return fallback
        classes = dict(root.get("classes") or {})
        if not classes:
            return fallback
        result: dict[str, str] = {}
        for class_name, class_id in classes.items():
            try:
                result[str(int(class_id))] = str(class_name)
            except Exception:
                continue
        return result or fallback

    @staticmethod
    def _semantic_class_by_name() -> dict[str, dict[str, Any]]:
        return {str(item["class_name"]): dict(item) for item in SEMANTIC_SEGMENTATION_CLASSES}

    @staticmethod
    def _semantic_color_from_map(color_map: Any, class_id: int) -> list[int] | None:
        if not isinstance(color_map, Sequence) or isinstance(color_map, (str, bytes)):
            return None
        if class_id < 0 or class_id >= len(color_map):
            return None
        value = color_map[class_id]
        if isinstance(value, dict):
            raw = [value.get("r", value.get("R", 0)), value.get("g", value.get("G", 0)), value.get("b", value.get("B", 0))]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raw = list(value)[:3]
        else:
            return None
        if len(raw) < 3:
            return None
        result: list[int] = []
        for channel in raw[:3]:
            number = float(channel)
            if 0.0 <= number <= 1.0:
                number *= 255.0
            result.append(max(0, min(255, int(round(number)))))
        return result

    @staticmethod
    def _semantic_assign_class_for_actor_name(name: str) -> str:
        haystack = str(name or "")
        for item in SEMANTIC_SEGMENTATION_CLASSES:
            if re.fullmatch(str(item["actor_regex"]), haystack, flags=re.IGNORECASE):
                return str(item["class_name"])
        return ""

    def _register_pie_semantic_segmentation_actors(self, *, allow_mutation: bool = False) -> dict[str, Any]:
        classes_for_remote = [
            {
                "class_name": str(item["class_name"]),
                "actor_regex": str(item["actor_regex"]),
                "canonical_actor_label": str(item.get("canonical_actor_label") or ""),
            }
            for item in SEMANTIC_SEGMENTATION_CLASSES
        ]
        mutation_enabled = bool(allow_mutation)
        mutation_enabled_literal = "True" if mutation_enabled else "False"
        script = f"""
import json
import re
import unreal

classes = json.loads({json.dumps(json.dumps(classes_for_remote, ensure_ascii=True))})
mutation_enabled = {mutation_enabled_literal}
payload = {{
    "world": "",
    "actor_count": 0,
    "sim_mode_found": False,
    "sim_mode_name": "",
    "mutation_enabled": mutation_enabled,
    "mutation_policy": "disabled_by_default_after_oom_crash" if not mutation_enabled else "explicitly_enabled_unsafe_actor_registration",
    "registered_actor_count": 0,
    "classes": {{}},
    "errors": [],
}}
for item in classes:
    payload["classes"][item["class_name"]] = {{
        "actor_match_count": 0,
        "registered_actor_count": 0,
    }}

try:
    world = unreal.EditorLevelLibrary.get_game_world()
    payload["world"] = "game_world"
except Exception as exc:
    world = None
    payload["errors"].append("get_game_world: " + str(exc))

actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor) if world else []
payload["actor_count"] = len(actors)
sim_mode = None
for actor in actors:
    cls = actor.get_class().get_name()
    name = actor.get_name()
    label = actor.get_actor_label()
    if "SimMode" in cls or "SimMode" in name or "SimMode" in label:
        sim_mode = actor
        payload["sim_mode_found"] = True
        payload["sim_mode_name"] = name + "|" + label + "|" + cls
        break

if sim_mode is None:
    payload["errors"].append("SimModeWorldMultiRotor actor not found")
else:
    compiled = [(item["class_name"], item["actor_regex"], re.compile(item["actor_regex"], re.IGNORECASE)) for item in classes]
    for actor in actors:
        name = actor.get_name()
        label = actor.get_actor_label()
        cls = actor.get_class().get_name()
        values = [name, label, cls, name + "|" + label + "|" + cls]
        for class_name, actor_regex, pattern in compiled:
            if any(pattern.fullmatch(value) for value in values):
                row = payload["classes"][class_name]
                row["actor_match_count"] += 1
                if mutation_enabled:
                    try:
                        if sim_mode.add_new_actor_to_instance_segmentation(actor, False):
                            row["registered_actor_count"] += 1
                            payload["registered_actor_count"] += 1
                    except Exception as exc:
                        payload["errors"].append("register " + name + ": " + str(exc))
                break
    if mutation_enabled:
        try:
            sim_mode.force_update_instance_segmentation()
            payload["force_update_instance_segmentation"] = True
        except Exception as exc:
            payload["force_update_instance_segmentation"] = False
            payload["errors"].append("force_update_instance_segmentation: " + str(exc))
    else:
        payload["force_update_instance_segmentation"] = False
        payload["errors"].append("PIE actor mutation skipped: unsafe AirSim annotation registration can exhaust UE memory")

print("AEROWORLD_SEMANTIC_REGISTRY_JSON_BEGIN" + json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "AEROWORLD_SEMANTIC_REGISTRY_JSON_END")
"""
        result = self._fixed_world_capture_hook().remote.run_python(
            script,
            unattended=False,
            raise_on_failure=True,
        )
        output_text = "".join(str(item.get("output", "")) for item in result.get("output") or [] if isinstance(item, dict))
        match = re.search(
            r"AEROWORLD_SEMANTIC_REGISTRY_JSON_BEGIN(.*?)AEROWORLD_SEMANTIC_REGISTRY_JSON_END",
            output_text,
            flags=re.S,
        )
        if not match:
            raise RuntimeError(f"Unable to parse semantic registry remote output: {output_text[:2000]}")
        return json.loads(match.group(1))

    def _semantic_component_matches(self, component_names: list[str]) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for item in SEMANTIC_SEGMENTATION_CLASSES:
            class_name = str(item["class_name"])
            component_regex = str(item["component_regex"])
            compiled = re.compile(component_regex, flags=re.IGNORECASE)
            matches = [name for name in component_names if compiled.fullmatch(str(name))]
            rows[class_name] = {
                "class_id": int(item["class_id"]),
                "class_name": class_name,
                "category": str(item.get("category") or ""),
                "actor_regex": str(item["actor_regex"]),
                "component_regex": component_regex,
                "required_for_static_audit": bool(item.get("required_for_static_audit", False)),
                "matched_component_count": int(len(matches)),
                "matched_component_sample": matches[:20],
            }
        return rows

    def _configure_semantic_segmentation_registry(self, entity_records: list[dict[str, Any]]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        audit_only = bool(getattr(self.args, "segmentation_registry_audit_only", False))
        if (
            not audit_only
            and self.airsim_segmentation_ready
            and self.airsim_segmentation_registry_payload is not None
        ):
            payload = dict(self.airsim_segmentation_registry_payload)
            payload["registry_reused"] = True
            payload["entity_records_considered"] = int(len(entity_records))
            return payload
        allow_actor_registration = bool(
            getattr(self.args, "enable_unsafe_pie_segmentation_actor_registration", False)
        )
        actor_payload = self._register_pie_semantic_segmentation_actors(
            allow_mutation=allow_actor_registration
        )
        if audit_only:
            component_rows = self._semantic_component_matches([])
            configured_rows: list[dict[str, Any]] = []
            for item in SEMANTIC_SEGMENTATION_CLASSES:
                class_name = str(item["class_name"])
                row = dict(component_rows[class_name])
                row["actor_match_count"] = int(
                    ((actor_payload.get("classes") or {}).get(class_name) or {}).get("actor_match_count") or 0
                )
                row["registered_actor_count"] = int(
                    ((actor_payload.get("classes") or {}).get(class_name) or {}).get("registered_actor_count") or 0
                )
                row["canonical_actor_label"] = str(item.get("canonical_actor_label") or "")
                row["actor_sample"] = []
                row["color_rgb"] = None
                row["set_segmentation_object_id_result"] = False
                row["set_segmentation_object_id_skipped"] = "audit_only_no_airsim_component_query_or_mutation"
                configured_rows.append(row)
            return {
                "segmentation_kind": "airsim_semantic_class_id_color",
                "semantic_segmentation_claim": True,
                "registry_version": "semantic_class_registry_v1",
                "registry_authority": "ue_actor_counts_only_audit_no_airsim_mutation",
                "registry_reused": False,
                "audit_only": True,
                "component_query_skipped": True,
                "component_query_skip_reason": (
                    "simListInstanceSegmentationObjects can add AirSim annotation objects; "
                    "audit-only must not mutate PIE state"
                ),
                "unsafe_pie_actor_registration_enabled": allow_actor_registration,
                "unsafe_pie_actor_registration_risk": (
                    "disabled by default because prior logs show AirSim annotation actor registration "
                    "on DynamicMeshComponent/CityBaseMeshComponent can exhaust UE memory"
                ),
                "hazard_trigger_pixel_policy": "only_rendered_trigger_or_hazard_proxies_can_appear_in_pixels",
                "city_base_policy": "BP_CityBaseGenerator is a merged road_terrain_ground_water background class; no material split is claimed",
                "ignored_actor_classes": ["SumoRoadNetworkActor"],
                "entity_records_considered": int(len(entity_records)),
                "pie_actor_registration": actor_payload,
                "registered_instance_segmentation_object_count": 0,
                "registered_instance_segmentation_object_sample": [],
                "semantic_classes": configured_rows,
                "semantic_class_by_id": {str(row["class_id"]): row["class_name"] for row in configured_rows},
                "airsim_segmentation_color_map": [],
            }
        component_names = self._retry(
            "simListInstanceSegmentationObjects",
            self.client.list_instance_segmentation_objects,
        )
        component_names = [str(value) for value in component_names]
        component_rows = self._semantic_component_matches(component_names)
        color_map: list[Any] = []
        try:
            color_map = self._retry("simGetSegmentationColorMap", self.client.get_segmentation_color_map)
        except Exception as exc:
            print(f"[EpisodeHost] AirSim segmentation color map warning: {exc}")
        configured_rows: list[dict[str, Any]] = []
        for item in SEMANTIC_SEGMENTATION_CLASSES:
            class_name = str(item["class_name"])
            class_id = int(item["class_id"])
            row = dict(component_rows[class_name])
            row["actor_match_count"] = int(
                ((actor_payload.get("classes") or {}).get(class_name) or {}).get("actor_match_count") or 0
            )
            row["registered_actor_count"] = int(
                ((actor_payload.get("classes") or {}).get(class_name) or {}).get("registered_actor_count") or 0
            )
            row["canonical_actor_label"] = str(item.get("canonical_actor_label") or "")
            row["actor_sample"] = list(((actor_payload.get("classes") or {}).get(class_name) or {}).get("actor_sample") or [])
            row["color_rgb"] = self._semantic_color_from_map(color_map, class_id)
            row["set_segmentation_object_id_result"] = False
            row["set_segmentation_object_id_skipped"] = ""
            configured_rows.append(row)

        required_failures = [
            row
            for row in configured_rows
            if row.get("required_for_static_audit")
            and row.get("actor_match_count", 0) > 0
            and row.get("matched_component_count", 0) <= 0
        ]
        building_rows = [row for row in configured_rows if str(row.get("category")) == "building"]
        building_actor_matches = sum(int(row.get("actor_match_count") or 0) for row in building_rows)
        building_component_matches = sum(int(row.get("matched_component_count") or 0) for row in building_rows)
        if required_failures:
            raise RuntimeError(
                "Semantic segmentation required static classes had actors but no AirSim components: "
                + json.dumps(required_failures, ensure_ascii=False)
            )
        if building_actor_matches > 0 and building_component_matches <= 0:
            raise RuntimeError(
                "Semantic segmentation found building actors but no AirSim building components after PIE registration."
            )
        for item in SEMANTIC_SEGMENTATION_CLASSES:
            class_name = str(item["class_name"])
            class_id = int(item["class_id"])
            row = next(value for value in configured_rows if str(value.get("class_name")) == class_name)
            if row["matched_component_count"] <= 0:
                row["set_segmentation_object_id_skipped"] = "no_matching_airsim_components"
                continue
            if audit_only:
                row["set_segmentation_object_id_skipped"] = "audit_only_no_mutation"
                continue
            row["set_segmentation_object_id_result"] = bool(
                self._retry(
                    "simSetSegmentationObjectID",
                    self.client.set_segmentation_object_id,
                    str(item["component_regex"]),
                    class_id,
                    is_name_regex=True,
                )
            )
        static_set_failures = [
            row
            for row in configured_rows
            if row.get("matched_component_count", 0) > 0
            and (
                row.get("required_for_static_audit")
                or str(row.get("category")) == "building"
            )
            and not bool(row.get("set_segmentation_object_id_result"))
        ]
        if static_set_failures:
            raise RuntimeError(
                "Semantic segmentation failed to assign AirSim IDs for required static/building classes: "
                + json.dumps(static_set_failures, ensure_ascii=False)
            )
        payload = {
            "segmentation_kind": "airsim_semantic_class_id_color",
            "semantic_segmentation_claim": True,
            "registry_version": "semantic_class_registry_v1",
            "registry_authority": (
                "ue_pie_actor_registration_then_airsim_component_regex_ids"
                if allow_actor_registration
                else "airsim_existing_component_regex_ids_only"
            ),
            "unsafe_pie_actor_registration_enabled": allow_actor_registration,
            "unsafe_pie_actor_registration_risk": (
                "disabled by default because prior logs show AirSim annotation actor registration "
                "on DynamicMeshComponent/CityBaseMeshComponent can exhaust UE memory"
            ),
            "hazard_trigger_pixel_policy": "only_rendered_trigger_or_hazard_proxies_can_appear_in_pixels",
            "city_base_policy": "BP_CityBaseGenerator is a merged road_terrain_ground_water background class; no material split is claimed",
            "ignored_actor_classes": ["SumoRoadNetworkActor"],
            "entity_records_considered": int(len(entity_records)),
            "pie_actor_registration": actor_payload,
            "registered_instance_segmentation_object_count": int(len(component_names)),
            "registered_instance_segmentation_object_sample": component_names[:80],
            "semantic_classes": configured_rows,
            "semantic_class_by_id": {str(row["class_id"]): row["class_name"] for row in configured_rows},
            "airsim_segmentation_color_map": color_map,
        }
        if not audit_only:
            self.airsim_segmentation_ready = True
            self.airsim_segmentation_registry_payload = dict(payload)
        return payload

    @staticmethod
    def _semantic_pixel_counts(raw_rgb: Any, registry_payload: dict[str, Any]) -> dict[str, Any]:
        import numpy as np  # type: ignore

        flat_rgb = raw_rgb.reshape((-1, 3))
        colors, counts = np.unique(flat_rgb, axis=0, return_counts=True)
        raw_counts = [
            {
                "color_rgb": [int(channel) for channel in colors[index].tolist()],
                "pixel_count": int(counts[index]),
            }
            for index in np.argsort(counts)[::-1]
        ]
        class_pixel_counts: dict[str, int] = {}
        color_to_class: dict[tuple[int, int, int], str] = {}
        for row in registry_payload.get("semantic_classes") or []:
            color = row.get("color_rgb")
            if isinstance(color, Sequence) and not isinstance(color, (str, bytes)) and len(color) >= 3:
                color_to_class[tuple(int(channel) for channel in list(color)[:3])] = str(row.get("class_name") or "")
                class_pixel_counts[str(row.get("class_name") or "")] = 0
        unknown_pixel_count = 0
        for color_row in raw_counts:
            color_tuple = tuple(int(channel) for channel in color_row["color_rgb"])
            class_name = color_to_class.get(color_tuple)
            if class_name:
                class_pixel_counts[class_name] = class_pixel_counts.get(class_name, 0) + int(color_row["pixel_count"])
            else:
                unknown_pixel_count += int(color_row["pixel_count"])
        return {
            "semantic_raw_unique_color_count": int(len(raw_counts)),
            "semantic_raw_top_colors": raw_counts[:40],
            "class_pixel_counts": class_pixel_counts,
            "unknown_semantic_color_pixel_count": int(unknown_pixel_count),
            "known_semantic_color_pixel_count": int(flat_rgb.shape[0] - unknown_pixel_count),
        }

    def _write_capture_storage_manifest(self) -> Path:
        path = self.output_dir / "capture_storage_manifest.json"
        ensure_dir(path.parent)
        payload = {
            "$schema": "aeroworld_capture_storage_manifest_v1",
            "episode_id": self.episode_id,
            "output_root": str(self.output_dir),
            "storage_layout_version": "capture_storage_v1",
            "storage_rule": "<episode_output_root>/<batch_id>/<capture_view_id>/<modality>/<frame_stem>.<ext>",
            "determinism_contract": {
                "timestamp_or_version_directories": False,
                "rerun_overwrites_same_modality_directory": True,
                "batch_id_from_site_contract": True,
                "capture_view_id_required_for_auditable_multi_uav_runs": True,
            },
            "capture_route_contract": {
                "ground_backend": "editor_hook_fixed_world_camera",
                "ground_modalities": ["rgb"],
                "uav_backend": "airsim_native_uav_camera",
                "uav_capture_vehicle": self.airsim_capture_vehicle,
                "uav_modalities": ["rgb", "depth", "seg"],
                "single_camera_single_modality_per_run": True,
                "uav_editor_hook_fallback_enabled": False,
                "python_segmentation_id_assignment_enabled": False,
                "segmentation_backend": self.segmentation_backend,
                "segmentation_kind": "ue_custom_stencil_class_id_u8",
                "semantic_segmentation_claim": True,
                "semantic_rules_path": str(self.semantic_rules_path),
                "semantic_class_by_id": self.semantic_class_by_id,
            },
            "requested_filters": {
                "camera_role": str(getattr(self.args, "camera_role", "") or ""),
                "camera_id": list(getattr(self.args, "camera_id", []) or []),
                "modality": list(getattr(self.args, "modality", []) or []),
                "airsim_capture_entity": str(getattr(self.args, "airsim_capture_entity", "") or ""),
                "capture_view_id": str(getattr(self.args, "capture_view_id", "") or ""),
                "segmentation_backend": self.segmentation_backend,
                "semantic_rules_path": str(self.semantic_rules_path),
                "batch_id": str(getattr(self.args, "batch_id", "") or ""),
                "capture_tick": [int(value) for value in (getattr(self.args, "capture_tick", []) or [])],
            },
            "modalities": {
                "rgb": {
                    "primary_format": "png",
                    "route": "AirSim ImageType.Scene for UAV; editor hook fixed-world RGB for ground",
                },
                "depth": {
                    "primary_format": "npy_float32_m",
                    "route": "AirSim ImageType.DepthPerspective for UAV only",
                    "debug_preview_optional": bool(getattr(self.args, "write_depth_preview", False)),
                },
                "seg": {
                    "primary_format": "png_uint8_class_id",
                    "audit_format": "json_semantic_stencil_audit",
                    "route": "UE CustomDepth/CustomStencil fixed-world capture for UAV segmentation; AirSim native segmentation is not used by default",
                },
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _editor_hook_output_format(modality_id: str) -> str:
        normalized = str(modality_id or "rgb").strip().lower()
        if normalized != "rgb":
            raise RuntimeError(
                "Editor-hook fixed-world capture is restricted to rgb. "
                f"Unsupported modality: {modality_id}"
            )
        return "png"

    @staticmethod
    def _depth_stats_for_npy(image_path: Path) -> dict[str, Any]:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            return {"depth_unit_m": True, "depth_stats_error": f"numpy import failed: {exc}"}

        try:
            values = np.load(image_path)
        except Exception as exc:
            return {"depth_unit_m": True, "depth_stats_error": f"depth npy load failed: {exc}"}

        arr = np.asarray(values, dtype=np.float32)
        valid_mask = np.isfinite(arr) & (arr > 0.0)
        valid_count = int(valid_mask.sum())
        invalid_count = int(arr.size - valid_count)
        if valid_count > 0:
            depth_min_m = float(arr[valid_mask].min())
            depth_max_m = float(arr[valid_mask].max())
        else:
            depth_min_m = 0.0
            depth_max_m = 0.0
        return {
            "depth_unit_m": True,
            "depth_dtype": "float32",
            "depth_shape": [int(value) for value in arr.shape],
            "depth_valid_count": valid_count,
            "depth_invalid_count": invalid_count,
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
        }

    @staticmethod
    def _semantic_stencil_pixel_counts(image_path: Path) -> dict[str, Any]:
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            return {"class_histogram_error": f"Pillow/numpy import failed: {exc}"}

        try:
            image = Image.open(image_path).convert("L")
            values = np.asarray(image, dtype=np.uint8)
        except Exception as exc:
            return {"class_histogram_error": f"semantic stencil PNG load failed: {exc}"}

        unique, counts = np.unique(values, return_counts=True)
        histogram = {str(int(class_id)): int(count) for class_id, count in zip(unique, counts)}
        return {
            "class_histogram": histogram,
            "ignore_pixel_count": int(histogram.get("0", 0)),
            "non_ignore_pixel_count": int(values.size - int(histogram.get("0", 0))),
            "semantic_unique_class_ids": [int(value) for value in unique.tolist()],
        }

    def _write_fixed_world_capture_output(
        self,
        batch: BatchPlan,
        frame: dict[str, Any],
        camera_id: str,
        modality_id: str,
        image_path: Path,
        common_sidecar: dict[str, Any],
        *,
        camera_role: str = "ground",
        width: int,
        height: int,
        fov_degrees: float,
    ) -> None:
        modalities = dict(self.capture_presets.get("modalities") or {})
        modality = dict(modalities.get(modality_id) or {})
        sidecar_path = image_path.with_suffix(".json")
        normalized_modality = str(modality_id or "rgb").strip().lower()
        image_data_format = "float32" if normalized_modality == "depth" else "uint8"
        output_format = self._editor_hook_output_format(normalized_modality)
        sidecar = dict(common_sidecar)
        sidecar.update(
            {
                "camera_id": camera_id,
                "camera_role": camera_role,
                "modality": normalized_modality,
                "image_type": modality.get("image_type", "Scene"),
                "pixels_as_float": bool(modality.get("pixels_as_float", normalized_modality == "depth")),
                "compress": bool(modality.get("compress", normalized_modality != "depth")),
                "image_data_format": image_data_format,
                "output_format": output_format,
                "image_path": str(image_path),
                "output_path": str(image_path),
                "capture_error": None,
                "capture_backend": str(common_sidecar.get("capture_backend") or "editor_hook_fixed_world_camera"),
                "width": int(width),
                "height": int(height),
                "fov_degrees": float(fov_degrees),
            }
        )
        sidecar.update(
            self._capture_storage_payload(
                batch=batch,
                view_id=camera_id,
                modality=normalized_modality,
                modality_output_dir=image_path.parent,
                primary_output_path=image_path,
                camera_role=camera_role,
            )
        )
        if normalized_modality == "depth":
            sidecar.update(self._depth_stats_for_npy(image_path))
        self.capture_orchestrator.write_sidecar(sidecar_path, sidecar)

    def _write_ue_stencil_capture_output(
        self,
        batch: BatchPlan,
        frame: dict[str, Any],
        *,
        view_id: str,
        camera_id: str,
        camera_name: str,
        image_path: Path,
        common_sidecar: dict[str, Any],
        width: int,
        height: int,
        fov_degrees: float,
        semantic_audit_path: Path,
    ) -> None:
        sidecar_path = image_path.with_suffix(".json")
        sidecar = dict(common_sidecar)
        sidecar.update(
            {
                "camera_id": camera_id,
                "camera_role": "uav",
                "camera_name": camera_name,
                "modality": "seg",
                "image_type": "UECustomStencil",
                "pixels_as_float": False,
                "compress": True,
                "image_path": str(image_path),
                "output_path": str(image_path),
                "capture_backend": "ue_custom_stencil_fixed_world_camera",
                "capture_view_id": view_id,
                "segmentation_backend": "ue_custom_stencil",
                "segmentation_kind": "ue_custom_stencil_class_id_u8",
                "semantic_segmentation_claim": True,
                "semantic_rules_path": str(self.semantic_rules_path),
                "semantic_audit_path": str(semantic_audit_path),
                "semantic_class_by_id": dict(self.semantic_class_by_id),
                "width": int(width),
                "height": int(height),
                "fov_degrees": float(fov_degrees),
                "output_format": "png_uint8_class_id",
            }
        )
        sidecar.update(
            self._capture_storage_payload(
                batch=batch,
                view_id=view_id,
                modality="seg",
                modality_output_dir=image_path.parent,
                primary_output_path=image_path,
                camera_role="uav",
            )
        )
        sidecar.update(self._semantic_stencil_pixel_counts(image_path))
        self.capture_orchestrator.write_sidecar(sidecar_path, sidecar)

    @staticmethod
    def _airsim_image_bytes(response: Any) -> bytes:
        raw = getattr(response, "image_data_uint8", b"")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        try:
            return bytes(int(value) & 0xFF for value in raw)
        except TypeError as exc:
            raise RuntimeError("AirSim image response does not contain uint8 image bytes") from exc

    @staticmethod
    def _airsim_response_pose_payload(response: Any) -> dict[str, Any]:
        position = getattr(response, "camera_position", None)
        orientation = getattr(response, "camera_orientation", None)
        return {
            "camera_position_ned_m": [
                float(getattr(position, "x_val", 0.0)),
                float(getattr(position, "y_val", 0.0)),
                float(getattr(position, "z_val", 0.0)),
            ],
            "camera_position_enu_m": [
                float(getattr(position, "x_val", 0.0)),
                float(getattr(position, "y_val", 0.0)),
                float(-getattr(position, "z_val", 0.0)),
            ],
            "camera_orientation": {
                "x_val": float(getattr(orientation, "x_val", 0.0)),
                "y_val": float(getattr(orientation, "y_val", 0.0)),
                "z_val": float(getattr(orientation, "z_val", 0.0)),
                "w_val": float(getattr(orientation, "w_val", 1.0)),
            },
        }

    def _write_airsim_native_capture_output(
        self,
        batch: BatchPlan,
        frame: dict[str, Any],
        *,
        view_id: str,
        camera_id: str,
        camera_name: str,
        modality_id: str,
        response: Any,
        common_sidecar: dict[str, Any],
        width: int,
        height: int,
        fov_degrees: float,
        segmentation_payload: dict[str, Any] | None = None,
    ) -> None:
        modalities = dict(self.capture_presets.get("modalities") or {})
        modality = dict(modalities.get(modality_id) or {})
        normalized_modality = str(modality_id or "rgb").strip().lower()
        output_dir = self.output_dir / batch.batch_id / safe_name(view_id) / safe_name(normalized_modality)
        self._prepare_capture_output_dir(output_dir)
        frame_stem = self.capture_orchestrator.frame_stem(frame)
        extension = str(modality.get("extension") or ("npy" if normalized_modality == "depth" else "png"))
        image_path = output_dir / f"{frame_stem}.{extension}"
        depth_preview_path: Path | None = None
        seg_raw_path: Path | None = None
        semantic_pixel_payload: dict[str, Any] | None = None
        if normalized_modality == "depth":
            try:
                import numpy as np  # type: ignore
            except Exception as exc:
                raise RuntimeError("numpy is required to write AirSim depth output") from exc
            values = np.asarray(list(getattr(response, "image_data_float", []) or []), dtype=np.float32)
            response_width = int(getattr(response, "width", 0) or width)
            response_height = int(getattr(response, "height", 0) or height)
            expected_size = response_width * response_height
            if expected_size <= 0 or values.size != expected_size:
                raise RuntimeError(
                    f"AirSim depth response size mismatch: values={values.size} width={response_width} height={response_height}"
                )
            depth_image = values.reshape((response_height, response_width))
            np.save(image_path, depth_image)
            if bool(getattr(self.args, "write_depth_preview", False)):
                try:
                    from PIL import Image  # type: ignore
                except Exception as exc:
                    raise RuntimeError("Pillow is required to write AirSim depth preview output") from exc
                finite = np.isfinite(depth_image)
                if finite.any():
                    finite_values = depth_image[finite]
                    near = float(np.percentile(finite_values, 2.0))
                    far = float(np.percentile(finite_values, 98.0))
                    if far <= near:
                        far = near + 1.0
                    preview = np.clip((depth_image - near) / (far - near), 0.0, 1.0)
                    preview[~finite] = 0.0
                    preview_u8 = (preview * 255.0).astype(np.uint8)
                else:
                    preview_u8 = np.zeros_like(depth_image, dtype=np.uint8)
                depth_preview_path = output_dir / f"{frame_stem}__depth_preview.png"
                Image.fromarray(preview_u8, mode="L").save(depth_preview_path)
        else:
            data = self._airsim_image_bytes(response)
            if not data:
                raise RuntimeError(f"AirSim {normalized_modality} response is empty for {camera_id}")
            response_width = int(getattr(response, "width", 0) or width)
            response_height = int(getattr(response, "height", 0) or height)
            response_compress = bool(getattr(response, "compress", modality.get("compress", normalized_modality != "depth")))
            if normalized_modality == "seg":
                try:
                    import numpy as np  # type: ignore
                    from PIL import Image  # type: ignore
                except Exception as exc:
                    raise RuntimeError("Pillow and numpy are required to write AirSim semantic segmentation output") from exc
                if response_compress:
                    from io import BytesIO

                    raw_image = Image.open(BytesIO(data)).convert("RGB")
                    response_width, response_height = raw_image.size
                else:
                    expected_rgb_size = response_width * response_height * 3
                    if response_width <= 0 or response_height <= 0 or len(data) != expected_rgb_size:
                        raise RuntimeError(
                            f"AirSim raw {normalized_modality} response size mismatch: bytes={len(data)} "
                            f"width={response_width} height={response_height} expected={expected_rgb_size}"
                        )
                    raw_image = Image.frombytes("RGB", (response_width, response_height), data)
                seg_raw_path = output_dir / f"{frame_stem}__airsim_raw.png"
                raw_image.save(seg_raw_path)
                raw_rgb = np.asarray(raw_image.convert("RGB"), dtype=np.uint8)
                semantic_pixel_payload = self._semantic_pixel_counts(raw_rgb, dict(segmentation_payload or {}))
                raw_image.save(image_path)
            elif response_compress:
                image_path.write_bytes(data)
            else:
                expected_rgb_size = response_width * response_height * 3
                if response_width <= 0 or response_height <= 0 or len(data) != expected_rgb_size:
                    raise RuntimeError(
                        f"AirSim raw {normalized_modality} response size mismatch: bytes={len(data)} "
                        f"width={response_width} height={response_height} expected={expected_rgb_size}"
                    )
                try:
                    from PIL import Image  # type: ignore
                except Exception as exc:
                    raise RuntimeError("Pillow is required to encode uncompressed AirSim RGB/seg output") from exc
                Image.frombytes("RGB", (response_width, response_height), data).save(image_path)

        sidecar_path = image_path.with_suffix(".json")
        sidecar = dict(common_sidecar)
        response_pose = self._airsim_response_pose_payload(response)
        sidecar.update(
            {
                "camera_id": camera_id,
                "camera_role": "uav",
                "camera_name": camera_name,
                "modality": normalized_modality,
                "image_type": "Segmentation" if normalized_modality == "seg" else modality.get("image_type", "Scene"),
                "pixels_as_float": bool(modality.get("pixels_as_float", normalized_modality == "depth")),
                "compress": bool(modality.get("compress", normalized_modality != "depth")),
                "image_path": str(image_path),
                "output_path": str(image_path),
                "capture_backend": "airsim_native_uav_camera",
                "capture_view_id": view_id,
                "airsim_capture_vehicle": str(common_sidecar.get("capture_vehicle_name") or common_sidecar.get("vehicle_name") or ""),
                "width": int(getattr(response, "width", 0) or width),
                "height": int(getattr(response, "height", 0) or height),
                "fov_degrees": float(fov_degrees),
                "airsim_response": {
                    "time_stamp": int(getattr(response, "time_stamp", 0) or 0),
                    "message": str(getattr(response, "message", "") or ""),
                    "image_type": int(getattr(response, "image_type", 0) or 0),
                    "pixels_as_float": bool(getattr(response, "pixels_as_float", False)),
                    "compress": bool(getattr(response, "compress", False)),
                    "encoded_from_uncompressed_rgb": bool(
                        normalized_modality != "depth"
                        and not bool(getattr(response, "compress", modality.get("compress", normalized_modality != "depth")))
                    ),
                    **response_pose,
                },
            }
        )
        sidecar.update(
            self._capture_storage_payload(
                batch=batch,
                view_id=view_id,
                modality=normalized_modality,
                modality_output_dir=output_dir,
                primary_output_path=image_path,
                camera_role="uav",
            )
        )
        if normalized_modality == "depth":
            sidecar.update(self._depth_stats_for_npy(image_path))
            sidecar["output_format"] = "npy_float32_m"
            if depth_preview_path is not None:
                sidecar["depth_preview_path"] = str(depth_preview_path)
                sidecar["depth_preview_for_debug_only"] = True
        elif normalized_modality == "seg":
            sidecar.update(dict(segmentation_payload or {}))
            sidecar.update(dict(semantic_pixel_payload or {}))
            sidecar["output_format"] = "png_airsim_semantic_class_id_color"
            sidecar["raw_airsim_seg_path"] = str(seg_raw_path) if seg_raw_path is not None else ""
            sidecar["semantic_seg_path"] = str(image_path)
        else:
            sidecar["output_format"] = "png"
        self.capture_orchestrator.write_sidecar(sidecar_path, sidecar)

    @staticmethod
    def _add_rotation_offsets(rotation_deg: dict[str, Any], offset_deg: dict[str, Any] | None) -> dict[str, float]:
        offset = dict(offset_deg or {})
        return {
            "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", 0.0)))
            + float(offset.get("pitch_deg", offset.get("pitch", 0.0))),
            "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", 0.0)))
            + float(offset.get("yaw_deg", offset.get("yaw", 0.0))),
            "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", 0.0)))
            + float(offset.get("roll_deg", offset.get("roll", 0.0))),
        }

    def _uav_pose_for_capture(
        self,
        entity: dict[str, Any],
        vehicle_status: dict[str, Any],
    ) -> tuple[list[float], dict[str, float]]:
        try:
            truth_position, truth_rotation = self._transformed_entity_pose(entity)
            if len(truth_position) >= 3:
                return (
                    [float(truth_position[0]), float(truth_position[1]), float(truth_position[2])],
                    self._add_rotation_offsets(dict(truth_rotation), None),
                )
        except Exception:
            pass

        pose = dict(vehicle_status.get("pose") or {})
        status = dict(vehicle_status.get("status") or {})
        position = pose.get("position_enu_m") or status.get("current_enu_m")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) < 2:
            position = self._entity_position_enu(entity)
        rotation = pose.get("rotation_deg")
        if not isinstance(rotation, dict):
            rotation = self._entity_rotation_deg(entity)
        return (
            [float(position[0]), float(position[1]), float(position[2] if len(position) > 2 else 0.0)],
            self._add_rotation_offsets(dict(rotation), None),
        )

    def _capture_uav_airsim_native_modality(
        self,
        batch: BatchPlan,
        frame: dict[str, Any],
        *,
        modality_id: str,
        entity_id: str,
        entity: dict[str, Any],
        vehicle_status: dict[str, Any],
        camera_id: str,
        camera_name: str,
        preset: dict[str, Any],
        weather_payload: dict[str, Any],
        entity_records: list[dict[str, Any]],
        feedback_payload: dict[str, Any],
        uav_debug: dict[str, Any],
    ) -> None:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        if entity_id != self.active_airsim_capture_entity_id:
            raise RuntimeError(
                f"AirSim native capture can only capture active source '{self.active_airsim_capture_entity_id}', got '{entity_id}'."
            )
        self._ensure_airsim_capture_vehicle()
        uav_position_enu_m, uav_rotation_deg = self._uav_pose_for_capture(entity, vehicle_status)
        source_uav_position_enu_m = list(uav_position_enu_m)
        uav_position_enu_m = self._ground_relative_position(
            uav_position_enu_m,
            enabled=bool(self.ground_reference_cfg.get("uav_ground_relative", False)),
            cache_namespace=f"capture_pre:{entity_id}",
            use_cache=True,
        )
        capture_pose = self._pin_airsim_capture_vehicle(
            uav_position_enu_m,
            uav_rotation_deg,
            context=f"pre-capture {entity_id} tick {int(frame['tick'])}",
        )
        settle_s = float((self.config.get("timeouts") or {}).get("camera_settle_s", 0.25))
        if settle_s > 0.0:
            time.sleep(settle_s)
        fov_degrees = float(preset.get("fov_degrees") or 85.0)
        camera_offset_body_m = [
            float(value)
            for value in (preset.get("camera_offset_body_ned_m") or preset.get("camera_offset_body_m") or [0.0, 0.0, 0.0])
        ][:3]
        while len(camera_offset_body_m) < 3:
            camera_offset_body_m.append(0.0)
        camera_rotation_body_deg = dict(preset.get("fixed_rotation_offset_deg") or {})
        if not camera_rotation_body_deg:
            camera_rotation_body_deg = {"pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0}
        set_runtime_camera_pose = bool(preset.get("set_runtime_camera_pose", False))
        if set_runtime_camera_pose:
            camera_pose_frame = str(preset.get("camera_pose_frame") or preset.get("camera_pose_coordinate_frame") or "ned").strip().lower()
            if camera_pose_frame not in {"ned", "enu"}:
                raise RuntimeError(f"Unsupported AirSim camera pose frame '{camera_pose_frame}' for preset {preset}")
            set_camera_pose_fn = self.client.set_camera_pose_ned if camera_pose_frame == "ned" else self.client.set_camera_pose
            set_camera_pose_kwargs = (
                {"position_ned_m": camera_offset_body_m}
                if camera_pose_frame == "ned"
                else {"position_enu_m": camera_offset_body_m}
            )
            self._retry(
                "simSetCameraPose",
                set_camera_pose_fn,
                self.airsim_capture_vehicle,
                camera_name,
                rotation_deg=camera_rotation_body_deg,
                **set_camera_pose_kwargs,
            )
        self._retry("simSetCameraFov", self.client.set_camera_fov, self.airsim_capture_vehicle, camera_name, fov_degrees)
        camera_info_before_capture = self._retry(
            "simGetCameraInfo",
            self.client.get_camera_info,
            self.airsim_capture_vehicle,
            camera_name,
        )

        normalized_modality = str(modality_id or "rgb").strip().lower()
        modalities = dict(self.capture_presets.get("modalities") or {})
        modality = dict(modalities.get(normalized_modality) or {})
        if normalized_modality not in {"rgb", "depth", "seg"}:
            raise RuntimeError(f"AirSim native UAV capture does not support modality '{modality_id}'.")
        camera_suffix = safe_name(str(preset.get("camera_id_suffix", camera_name)))
        view_id = self.requested_capture_view_id or f"{safe_name(self.active_capture_view_id)}__{camera_suffix}"
        if normalized_modality == "seg" and self.segmentation_backend == "ue_custom_stencil":
            hook = self._fixed_world_capture_hook()
            output_dir = self.output_dir / batch.batch_id / safe_name(view_id) / "seg"
            self._prepare_capture_output_dir(output_dir)
            frame_stem = self.capture_orchestrator.frame_stem(frame)
            image_path = output_dir / f"{frame_stem}.png"
            semantic_audit_path = output_dir / f"{frame_stem}__semantic_stencil_audit.json"
            camera_world_rotation_deg = self._add_rotation_offsets(uav_rotation_deg, camera_rotation_body_deg)
            semantic_camera_asset_id = safe_name(f"fixed_world_camera.semantic.{view_id}")
            self.ground_camera_asset_ids[("uav_semantic_stencil", view_id)] = semantic_camera_asset_id
            hook.ensure_fixed_world_camera(
                map_id=self.map_id,
                asset_id=semantic_camera_asset_id,
                logical_asset_id="camera.fixed_world_capture.semantic_stencil.v1",
                position_enu_m=uav_position_enu_m,
                rotation_deg=camera_world_rotation_deg,
            )
            hook.capture_modality(
                map_id=self.map_id,
                asset_id=semantic_camera_asset_id,
                modality="seg",
                output_path=image_path,
                width=int(preset.get("width") or 1280),
                height=int(preset.get("height") or 720),
                fov_degrees=fov_degrees,
                semantic_rules_path=self.semantic_rules_path,
                semantic_audit_path=semantic_audit_path,
            )
            sidecar = {
                "episode_id": self.episode_id,
                "frame_id": str(frame.get("frame_id", "")),
                "frame_seq": int(frame.get("frame_seq") or frame.get("tick") or 0),
                "tick": int(frame["tick"]),
                "sim_time_s": float(frame.get("sim_time_s", 0.0)),
                "map_id": self.map_id,
                "batch_id": batch.batch_id,
                "site_id": batch.site_id,
                "roi_id": batch.roi_id,
                "weather": weather_payload,
                "feedback": feedback_payload,
                "uav_runtime": vehicle_status,
                "uav_debug": uav_debug,
                "uav_entity_id": entity_id,
                "source_uav_entity_id": entity_id,
                "capture_vehicle_name": self.airsim_capture_vehicle,
                "vehicle_name": self.airsim_capture_vehicle,
                "camera_name": camera_name,
                "requested_camera_pose_body_m": camera_offset_body_m,
                "requested_camera_rotation_body_deg": camera_rotation_body_deg,
                "set_runtime_camera_pose": set_runtime_camera_pose,
                "camera_pose_frame": str(preset.get("camera_pose_frame") or preset.get("camera_pose_coordinate_frame") or "ned").strip().lower(),
                "camera_info_before_capture": camera_info_before_capture,
                "source_uav_pose_enu_m": source_uav_position_enu_m,
                "expected_uav_pose_enu_m": uav_position_enu_m,
                "expected_uav_rotation_deg": uav_rotation_deg,
                "requested_capture_pose_enu_m": capture_pose.get("requested_position_enu_m"),
                "requested_capture_rotation_deg": capture_pose.get("requested_rotation_deg"),
                "airsim_pose_before_capture": capture_pose.get("pose"),
                "pose_error_m": capture_pose.get("pose_error_m"),
                "capture_pose_mode": capture_pose.get("capture_pose_mode"),
                "ue_stencil_camera_asset_id": semantic_camera_asset_id,
                "ue_stencil_camera_position_enu_m": uav_position_enu_m,
                "ue_stencil_camera_rotation_deg": camera_world_rotation_deg,
                "capture_alignment_key": f"{self.episode_id}:{batch.batch_id}:{int(frame['tick'])}:{view_id}",
                "capture_alignment_source": "deterministic_episode_frame",
                "capture_backend": "ue_custom_stencil_fixed_world_camera",
                "capture_view_id": view_id,
                "entity_records": entity_records,
                "roster_summary": frame.get("roster_summary", {}),
            }
            self._write_ue_stencil_capture_output(
                batch,
                frame,
                view_id=view_id,
                camera_id=camera_id,
                camera_name=camera_name,
                image_path=image_path,
                common_sidecar=sidecar,
                width=int(preset.get("width") or 1280),
                height=int(preset.get("height") or 720),
                fov_degrees=fov_degrees,
                semantic_audit_path=semantic_audit_path,
            )
            return
        segmentation_payload = None
        if normalized_modality == "seg":
            segmentation_payload = self._configure_semantic_segmentation_registry(entity_records)
        response = self._retry(
            "simGetImages",
            self.client.capture_vehicle_image,
            self.airsim_capture_vehicle,
            camera_name=camera_name,
            image_type=("Segmentation" if normalized_modality == "seg" else str(modality.get("image_type") or "Scene")),
            pixels_as_float=bool(modality.get("pixels_as_float", normalized_modality == "depth")),
            compress=bool(modality.get("compress", normalized_modality != "depth")),
            annotation_name=str(modality.get("annotation_name") or ""),
        )
        sidecar = {
            "episode_id": self.episode_id,
            "frame_id": str(frame.get("frame_id", "")),
            "frame_seq": int(frame.get("frame_seq") or frame.get("tick") or 0),
            "tick": int(frame["tick"]),
            "sim_time_s": float(frame.get("sim_time_s", 0.0)),
            "map_id": self.map_id,
            "batch_id": batch.batch_id,
            "site_id": batch.site_id,
            "roi_id": batch.roi_id,
            "weather": weather_payload,
            "feedback": feedback_payload,
            "uav_runtime": vehicle_status,
            "uav_debug": uav_debug,
            "uav_entity_id": entity_id,
            "source_uav_entity_id": entity_id,
            "capture_vehicle_name": self.airsim_capture_vehicle,
            "vehicle_name": self.airsim_capture_vehicle,
            "camera_name": camera_name,
            "requested_camera_pose_body_m": camera_offset_body_m,
            "requested_camera_rotation_body_deg": camera_rotation_body_deg,
            "set_runtime_camera_pose": set_runtime_camera_pose,
            "camera_pose_frame": str(preset.get("camera_pose_frame") or preset.get("camera_pose_coordinate_frame") or "ned").strip().lower(),
            "camera_info_before_capture": camera_info_before_capture,
            "source_uav_pose_enu_m": source_uav_position_enu_m,
            "expected_uav_pose_enu_m": uav_position_enu_m,
            "expected_uav_rotation_deg": uav_rotation_deg,
            "requested_capture_pose_enu_m": capture_pose.get("requested_position_enu_m"),
            "requested_capture_rotation_deg": capture_pose.get("requested_rotation_deg"),
            "airsim_pose_before_capture": capture_pose.get("pose"),
            "pose_error_m": capture_pose.get("pose_error_m"),
            "capture_pose_mode": capture_pose.get("capture_pose_mode"),
            "capture_alignment_key": f"{self.episode_id}:{batch.batch_id}:{int(frame['tick'])}:{view_id}",
            "capture_alignment_source": "deterministic_episode_frame",
            "capture_backend": "airsim_native_uav_camera",
            "capture_view_id": view_id,
            "entity_records": entity_records,
            "roster_summary": frame.get("roster_summary", {}),
        }
        self._write_airsim_native_capture_output(
            batch,
            frame,
            view_id=view_id,
            camera_id=camera_id,
            camera_name=camera_name,
            modality_id=normalized_modality,
            response=response,
            common_sidecar=sidecar,
            width=int(preset.get("width") or 1280),
            height=int(preset.get("height") or 720),
            fov_degrees=fov_degrees,
            segmentation_payload=segmentation_payload,
        )

    def _capture_ground_views(
        self,
        batch: BatchPlan,
        frame: dict[str, Any],
        weather_payload: dict[str, Any],
        entity_records: list[dict[str, Any]],
        feedback_payload: dict[str, Any],
        uav_status: dict[str, Any],
        uav_debug: dict[str, Any],
    ) -> None:
        if not self._capture_role_enabled("ground"):
            return
        hook = self._fixed_world_capture_hook()
        modalities = dict(self.capture_presets.get("modalities") or {})
        for preset in self._site_ground_presets(batch.site_id):
            camera_name = str(preset.get("camera_name", "external"))
            camera_id = str(preset.get("camera_id", camera_name))
            if not self._capture_camera_enabled(camera_id, camera_name):
                continue
            modality_ids = self._filtered_modalities(
                list(preset.get("modalities") or self.capture_presets.get("default_modalities") or ["rgb"])
            )
            if not modality_ids:
                continue
            unsupported_modalities = [modality_id for modality_id in modality_ids if str(modality_id).strip().lower() != "rgb"]
            if unsupported_modalities:
                raise RuntimeError(
                    "Fixed world capture currently supports ground modality 'rgb' only. "
                    f"Unsupported modalities for camera '{camera_id}': {unsupported_modalities}"
                )
            raw_camera_position_enu_m = [float(value) for value in (preset.get("position_enu_m") or [0.0, 0.0, 0.0])]
            resolved_camera_position_enu_m = self._camera_position_from_preset(preset)
            resolved_camera_position_enu_m = self._ground_relative_position(
                resolved_camera_position_enu_m,
                enabled=bool(self.ground_reference_cfg.get("ground_camera_ground_relative", False)),
                cache_namespace=f"ground_camera:{batch.site_id}:{camera_id}",
                use_cache=True,
            )
            raw_camera_rotation_deg = dict(preset.get("rotation_deg") or {})
            coordinate_space = str(preset.get("coordinate_space") or "").strip().lower()
            raw_pitch_deg = float(raw_camera_rotation_deg.get("pitch_deg", raw_camera_rotation_deg.get("pitch", 0.0)))
            if coordinate_space in {"map_enu", "world_enu", "transformed_enu"}:
                resolved_camera_rotation_deg = self._camera_rotation_from_preset(preset)
            elif abs(abs(raw_pitch_deg) - 90.0) <= 1e-3:
                # For nadir fixed-world cameras, position transform is enough. Applying the
                # scenario yaw here can collapse into a rolled UE rotator and shift a target
                # directly under the camera away from the image center.
                resolved_camera_rotation_deg = {
                    "pitch_deg": raw_pitch_deg,
                    "yaw_deg": float(raw_camera_rotation_deg.get("yaw_deg", raw_camera_rotation_deg.get("yaw", 0.0))),
                    "roll_deg": float(raw_camera_rotation_deg.get("roll_deg", raw_camera_rotation_deg.get("roll", 0.0))),
                }
            else:
                resolved_camera_rotation_deg = self._transform_rotation_deg(raw_camera_rotation_deg)
            fov_degrees = float(preset.get("fov_degrees") or 70.0)
            asset_id = self._ground_camera_asset_id(batch.site_id, camera_id)
            hook.ensure_fixed_world_camera(
                map_id=self.map_id,
                asset_id=asset_id,
                logical_asset_id="camera.fixed_world_capture.rgb.v1",
                position_enu_m=resolved_camera_position_enu_m,
                rotation_deg=resolved_camera_rotation_deg,
            )
            frame_stem = self.capture_orchestrator.frame_stem(frame)
            output_dir = self.output_dir / batch.batch_id / safe_name(camera_id) / "rgb"
            self._prepare_capture_output_dir(output_dir)
            image_path = output_dir / f"{frame_stem}.{str((modalities.get('rgb') or {}).get('extension', 'png'))}"
            hook.capture_modality(
                map_id=self.map_id,
                asset_id=asset_id,
                modality="rgb",
                output_path=image_path,
                width=int(preset.get("width") or 1280),
                height=int(preset.get("height") or 720),
                fov_degrees=fov_degrees,
            )
            sidecar = {
                "episode_id": self.episode_id,
                "frame_id": str(frame.get("frame_id", "")),
                "frame_seq": int(frame.get("frame_seq") or frame.get("tick") or 0),
                "tick": int(frame["tick"]),
                "sim_time_s": float(frame.get("sim_time_s", 0.0)),
                "map_id": self.map_id,
                "batch_id": batch.batch_id,
                "site_id": batch.site_id,
                "roi_id": batch.roi_id,
                "weather": weather_payload,
                "feedback": feedback_payload,
                "uav_runtime": uav_status,
                "uav_debug": uav_debug,
                "camera_pose_enu_m": resolved_camera_position_enu_m,
                "camera_rotation_deg": resolved_camera_rotation_deg,
                "source_camera_pose_enu_m": raw_camera_position_enu_m,
                "source_camera_rotation_deg": raw_camera_rotation_deg,
                "capture_backend": "editor_hook_fixed_world_camera",
                "camera_name": camera_name,
                "asset_id": asset_id,
                "entity_records": entity_records,
                "roster_summary": frame.get("roster_summary", {}),
            }
            self._write_fixed_world_capture_output(
                batch,
                frame,
                camera_id,
                "rgb",
                image_path,
                sidecar,
                camera_role="ground",
                width=int(preset.get("width") or 1280),
                height=int(preset.get("height") or 720),
                fov_degrees=fov_degrees,
            )

    def _capture_uav_views(
        self,
        batch: BatchPlan,
        frame: dict[str, Any],
        weather_payload: dict[str, Any],
        entity_records: list[dict[str, Any]],
        feedback_payload: dict[str, Any],
        uav_status: dict[str, Any],
        uav_debug: dict[str, Any],
    ) -> None:
        if not self._capture_role_enabled("uav"):
            return
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        uav_capture_status = dict(uav_status)
        if not uav_capture_status:
            expected_uavs = [
                str(entity.get("entity_id") or "")
                for entity in frame.get("entities") or []
                if str(entity.get("entity_category") or "").strip().lower() == "uav"
                and str(self._entity_resolution(entity).get("mode") or "") == "runtime_multirotor"
                and truth_submission_state(entity) == "submit_to_ue"
            ]
            expected_uavs = [entity_id for entity_id in expected_uavs if entity_id]
            if expected_uavs:
                raise RuntimeError(
                    "UAV capture requested but no runtime multirotor status was produced for: "
                    + ", ".join(sorted(expected_uavs))
                )
            return

        frame_entities_by_id = {
            str(entity.get("entity_id") or ""): entity
            for entity in frame.get("entities") or []
            if str(entity.get("entity_id") or "").strip()
        }

        if self._airsim_capture_enabled():
            entity_id = self.active_airsim_capture_entity_id
            if not entity_id:
                raise RuntimeError("AirSim native UAV capture has no active source entity.")
            if entity_id not in uav_capture_status:
                raise RuntimeError(
                    f"AirSim native UAV capture source '{entity_id}' has no capture status at tick {int(frame['tick'])}."
                )
            entity = frame_entities_by_id.get(entity_id) or self.roster_by_id.get(entity_id) or {}
            vehicle_status = dict(uav_capture_status.get(entity_id) or {})
            jobs: list[tuple[str, str, str, dict[str, Any]]] = []
            for preset in self._uav_camera_presets():
                camera_name = str(preset.get("camera_name", "front_center"))
                camera_id = f"{safe_name(entity_id)}__{safe_name(str(preset.get('camera_id_suffix', camera_name)))}"
                if not self._capture_camera_enabled(camera_id, str(preset.get("camera_id_suffix", "")), camera_name, entity_id):
                    continue
                modality_ids = self._filtered_modalities(
                    list(preset.get("modalities") or self.capture_presets.get("default_modalities") or ["rgb", "depth", "seg"])
                )
                for modality_id in modality_ids:
                    jobs.append((camera_id, camera_name, str(modality_id), preset))
            if not jobs:
                raise RuntimeError(
                    "AirSim native UAV capture found no camera/modality job. "
                    "Check --camera-id and --modality filters."
                )
            if len(jobs) != 1:
                raise RuntimeError(
                    "AirSim native UAV capture requires exactly one camera and one modality per run. "
                    f"Matched jobs: {[{'camera_id': job[0], 'camera_name': job[1], 'modality': job[2]} for job in jobs]}"
                )
            camera_id, camera_name, modality_id, preset = jobs[0]
            self._capture_uav_airsim_native_modality(
                batch,
                frame,
                modality_id=modality_id,
                entity_id=entity_id,
                entity=entity,
                vehicle_status=vehicle_status,
                camera_id=camera_id,
                camera_name=camera_name,
                preset=preset,
                weather_payload=weather_payload,
                entity_records=entity_records,
                feedback_payload=feedback_payload,
                uav_debug=uav_debug,
            )
            return

        raise RuntimeError(
            "UAV image capture requires --uav-capture-backend airsim_native. "
            "The editor-hook UAV capture fallback is disabled so long runs cannot silently switch capture routes."
        )

    def _batch_ticks(self, batch: BatchPlan) -> list[int]:
        ticks = [tick for tick in self.sorted_ticks if batch.tick_start <= tick <= batch.tick_end]
        stride = max(1, int(getattr(self.args, "simulation_tick_stride", 1) or 1))
        return ticks[::stride]

    def _batch_capture_tick_set(self, batch: BatchPlan) -> set[int]:
        explicit_capture_ticks = [
            int(value)
            for value in (getattr(self.args, "capture_tick", None) or [])
            if batch.tick_start <= int(value) <= batch.tick_end
        ]
        if explicit_capture_ticks:
            return set(explicit_capture_ticks)
        ticks = [tick for tick in self.sorted_ticks if batch.tick_start <= tick <= batch.tick_end]
        stride = max(1, int(getattr(self.args, "tick_stride", 1) or 1))
        return set(ticks[::stride])

    # ------------------------------------------------------------------
    # Event script action handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_script_vector3(value: Any, field_name: str) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{field_name} must be a numeric vector")
        values = list(value)
        if len(values) < 2:
            raise ValueError(f"{field_name} requires at least x/y components")
        return [
            float(values[0]),
            float(values[1]),
            float(values[2] if len(values) > 2 else 0.0),
        ]

    @staticmethod
    def _yaw_from_points(start_enu_m: Sequence[float], end_enu_m: Sequence[float]) -> float:
        dx = float(end_enu_m[0]) - float(start_enu_m[0])
        dy = float(end_enu_m[1]) - float(start_enu_m[1])
        return heading_deg_from_vector(dx, dy)

    def _script_transform_position(self, position_enu_m: Sequence[float], field_name: str = "position_enu_m") -> list[float]:
        position = self._coerce_script_vector3(position_enu_m, field_name)
        if self._truth_frame_uses_map_enu():
            return position
        return self._transform_position_enu(position)

    def _script_transform_waypoints(self, raw_waypoints: Any) -> list[list[float]]:
        if not isinstance(raw_waypoints, Sequence) or isinstance(raw_waypoints, (str, bytes)):
            raise ValueError("waypoints_enu_m must be a list of numeric vectors")
        return [
            self._script_transform_position(raw_waypoint, "waypoints_enu_m")
            for raw_waypoint in raw_waypoints
        ]

    def _script_transform_rotation(self, action: dict[str, Any], *, fallback_yaw_deg: float = 0.0) -> dict[str, float]:
        raw_rotation = dict(
            action.get("rotation_deg")
            or {
                "pitch_deg": 0.0,
                "yaw_deg": fallback_yaw_deg,
                "roll_deg": 0.0,
            }
        )
        if self._truth_frame_uses_map_enu():
            return raw_rotation
        return self._transform_rotation_deg(raw_rotation)

    @staticmethod
    def _script_asset_instance_id(action: dict[str, Any], entity_id: str) -> str:
        return str(
            action.get("asset_instance_id")
            or action.get("instance_id")
            or action.get("asset_id_override")
            or entity_id
        )

    def _script_entity_logical_asset_id(self, entity_id: str, action: dict[str, Any] | None = None) -> str:
        action = action or {}
        direct = str(
            action.get("logical_asset_id")
            or action.get("asset_id")
            or action.get("proxy_template_id")
            or ""
        )
        if direct:
            return direct
        if entity_id in self.event_entity_assets:
            return self.event_entity_assets[entity_id]
        roster_entry = self.roster_by_id.get(entity_id, {})
        resolved = self.template_resolver.resolve(roster_entry, roster_entry) if roster_entry else {}
        return str(
            resolved.get("logical_asset_id")
            or roster_entry.get("logical_asset_id")
            or roster_entry.get("proxy_template_id")
            or ""
        )

    @staticmethod
    def _script_asset_kind(logical_asset_id: str) -> str:
        if logical_asset_id.startswith("pedestrian."):
            return "pedestrian"
        if logical_asset_id.startswith("uav."):
            return "uav"
        if logical_asset_id.startswith("vehicle."):
            return "vehicle"
        return "asset"

    def _register_event_action_handlers(self) -> None:
        interp = self.event_interpreter
        interp.register_handler("set_weather", self._script_set_weather)
        interp.register_handler("spawn_entity", self._script_spawn_entity)
        interp.register_handler("move_entity", self._script_move_entity)
        interp.register_handler("remove_entity", self._script_remove_entity)
        interp.register_handler("play_animation", self._script_play_animation)
        interp.register_handler("set_pedestrian_activity", self._script_set_pedestrian_activity)
        interp.register_handler("spawn_crowd", self._script_spawn_crowd)
        interp.register_handler("clear_crowd", self._script_clear_crowd)
        interp.register_handler("set_visual_state", self._script_set_visual_state)
        interp.register_handler("capture_screenshot", self._script_capture_screenshot)

    def _script_set_weather(self, action: dict[str, Any]) -> dict[str, Any]:
        profile = str(action.get("profile", "clear"))
        payload = self.weather_service.payload_for_condition(profile)
        overrides = dict(action.get("overrides") or {})
        if overrides:
            for key in ("rain", "fog", "fog_density", "visibility_m", "wind_speed", "wetness", "dust"):
                if key in overrides:
                    if key == "visibility_m":
                        mapped = "visibility"
                    elif key == "fog":
                        mapped = "fog_density"
                    else:
                        mapped = key
                    payload[mapped] = overrides[key]
        self.client.apply_weather(payload, map_id=self.map_id)
        self.last_weather_signature = dict(payload)
        self.event_weather_overlay = dict(payload)
        self.event_weather_overlay["visibility_m"] = payload.get("visibility", payload.get("visibility_m", 0.0))
        self.event_weather_overlay["fog"] = payload.get("fog_density", payload.get("fog", 0.0))
        return {"status": "ok", "profile": profile}

    def _script_spawn_entity(self, action: dict[str, Any]) -> dict[str, Any]:
        logical_asset_id = str(
            action.get("logical_asset_id")
            or action.get("asset_id")
            or action.get("proxy_template_id")
            or ""
        )
        if not logical_asset_id:
            raise ValueError("spawn_entity requires asset_id or logical_asset_id")
        entity_id = str(action.get("entity_id", ""))
        if not entity_id:
            import uuid as _uuid
            entity_id = f"event_{_uuid.uuid4().hex[:8]}"
        asset_instance_id = self._script_asset_instance_id(action, entity_id)
        raw_position = [
            float(action["position_enu_m"][0]),
            float(action["position_enu_m"][1]),
            float(action["position_enu_m"][2] if len(action["position_enu_m"]) > 2 else 0.0),
        ]
        position = self._script_transform_position(action["position_enu_m"])
        rotation = self._script_transform_rotation(action)
        self.event_entity_assets[entity_id] = logical_asset_id
        self.event_entity_initial_positions[entity_id] = list(raw_position)
        self.event_controlled_entity_ids.add(entity_id)
        if self.event_interpreter is not None:
            self.event_interpreter.update_entity_state(entity_id, raw_position, rotation_deg=dict(action.get("rotation_deg") or {}))
        asset_kind = self._script_asset_kind(logical_asset_id)
        if asset_kind == "pedestrian":
            response = self.client.ped_spawn(
                entity_id,
                position_enu_m=position,
                yaw_deg=float(rotation.get("yaw_deg", 0.0)),
                map_id=self.map_id,
            )
            self.ped_active_ids.add(entity_id)
            return {"status": "ok", "entity_id": entity_id, "ped_id": entity_id, "response": response}
        if asset_kind == "uav":
            command_position = self._runtime_uav_command_position_enu(entity_id, position)
            response = self.client.create_runtime_multirotor(
                asset_instance_id,
                position_enu_m=command_position,
                rotation_deg=rotation,
                map_id=self.map_id,
            )
            self.uav_active_by_entity[entity_id] = asset_instance_id
            self.uav_last_command_target_by_entity[entity_id] = list(command_position)
            return {
                "status": "ok",
                "entity_id": entity_id,
                "vehicle_name": asset_instance_id,
                "command_position_enu_m": command_position,
                "response": response,
            }
        payload: dict[str, Any] = {
            "asset_id": asset_instance_id,
            "entity_id": entity_id,
            "logical_asset_id": logical_asset_id,
            "proxy_template_id": logical_asset_id,
            "pose_enu_m": {"position_enu_m": position, "rotation_deg": rotation},
            "position_enu_m": position,
            "rotation_deg": rotation,
            "tags": list(action.get("tags") or []),
            "visual_state": dict(action.get("visual_state") or {}),
        }
        response = self.client.spawn_asset(payload, map_id=self.map_id)
        return {"status": "ok", "entity_id": entity_id, "asset_id": asset_instance_id, "response": response}

    def _script_move_entity(self, action: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(action["entity_id"])
        asset_instance_id = self._script_asset_instance_id(action, entity_id)
        logical_asset_id = self._script_entity_logical_asset_id(entity_id, action)
        asset_kind = self._script_asset_kind(logical_asset_id)
        transformed_waypoints: list[list[float]] = []
        raw_position_for_interpreter: list[float] | None = None
        if action.get("waypoints_enu_m") is not None or action.get("waypoints") is not None:
            raw_waypoints = action.get("waypoints_enu_m", action.get("waypoints")) or []
            transformed_waypoints = self._script_transform_waypoints(raw_waypoints)
            if not transformed_waypoints:
                return {"status": "skipped", "reason": "no waypoints", "entity_id": entity_id}
            position = list(transformed_waypoints[-1])
            raw_last = raw_waypoints[-1]
            raw_position_for_interpreter = [
                float(raw_last[0]),
                float(raw_last[1]),
                float(raw_last[2] if len(raw_last) > 2 else 0.0),
            ]
            fallback_yaw = (
                self._yaw_from_points(transformed_waypoints[-2], transformed_waypoints[-1])
                if len(transformed_waypoints) >= 2
                else None
            )
        elif "position_enu_m" in action:
            position = self._script_transform_position(action["position_enu_m"])
            raw_position_for_interpreter = [
                float(action["position_enu_m"][0]),
                float(action["position_enu_m"][1]),
                float(action["position_enu_m"][2] if len(action["position_enu_m"]) > 2 else 0.0),
            ]
            fallback_yaw = None
        else:
            raise ValueError("move_entity requires position_enu_m or waypoints_enu_m")
        if "rotation_deg" in action:
            rotation = self._script_transform_rotation(action)
        elif fallback_yaw is not None:
            rotation = {"pitch_deg": 0.0, "yaw_deg": fallback_yaw, "roll_deg": 0.0}
        else:
            rotation = None
        pose_enu_m: dict[str, Any] = {"position_enu_m": position}
        if rotation is not None:
            pose_enu_m["rotation_deg"] = rotation
        payload: dict[str, Any] = {
            "asset_id": asset_instance_id,
            "entity_id": entity_id,
            "pose_enu_m": pose_enu_m,
            "position_enu_m": position,
            "tags": list(action.get("tags") or []),
            "visual_state": dict(action.get("visual_state") or {}),
        }
        if rotation is not None:
            payload["rotation_deg"] = rotation
        if transformed_waypoints:
            payload["waypoints_enu_m"] = transformed_waypoints
            payload["velocity_mps"] = float(action.get("velocity_mps", 5.0))
        if raw_position_for_interpreter is not None:
            self.event_entity_initial_positions[entity_id] = list(raw_position_for_interpreter)
            self.event_controlled_entity_ids.add(entity_id)
            if self.event_interpreter is not None:
                self.event_interpreter.update_entity_state(entity_id, raw_position_for_interpreter, rotation_deg=dict(action.get("rotation_deg") or {}))
        if asset_kind == "pedestrian":
            self.ped_active_ids.add(entity_id)
            return {
                "status": "ok",
                "entity_id": entity_id,
                "ped_id": entity_id,
                "command": "truth_frame_pose_sync",
                "target_enu_m": [float(value) for value in position],
                "reason": "pedestrian movement is applied deterministically from truth_frames before capture",
            }
        if asset_kind == "uav":
            vehicle_name = self.uav_active_by_entity.get(entity_id, asset_instance_id)
            command_position = self._runtime_uav_command_position_enu(entity_id, position)
            velocity_mps = float(action.get("velocity_mps", 5.0))
            if self._airsim_capture_enabled() and entity_id == self.active_airsim_capture_entity_id:
                self._ensure_airsim_capture_vehicle()
                command_rotation = rotation or self._entity_rotation_deg({"entity_id": entity_id})
                wait_status = self._pin_airsim_capture_vehicle(
                    command_position,
                    command_rotation,
                    context=f"event action {action.get('action_id') or '<unnamed>'}",
                )
                response = {
                    "payload": {
                        "vehicle_name": self.airsim_capture_vehicle,
                        "source_entity_id": entity_id,
                        "state": "ok",
                        "reason": "capture_entity_event_move_pinned_to_airsim_vehicle",
                        "target_enu_m": [float(value) for value in command_position],
                        "rotation_deg": dict(command_rotation or {}),
                        "velocity_mps": velocity_mps,
                        "synthetic": True,
                    }
                }
                self.uav_active_by_entity[entity_id] = self.airsim_capture_vehicle
                self.uav_last_command_target_by_entity[entity_id] = list(command_position)
                wait_status["path_used"] = "airsim_native_capture_vehicle"
                wait_status["source_entity_id"] = entity_id
                wait_status["replaces_runtime_multirotor"] = True
                return {
                    "status": "ok",
                    "entity_id": entity_id,
                    "vehicle_name": self.airsim_capture_vehicle,
                    "command_position_enu_m": command_position,
                    "response": response,
                    "wait_status": wait_status,
                }
            previous_target = self.uav_last_command_target_by_entity.get(entity_id)
            previous_error_m = distance_m(previous_target, command_position)
            if previous_error_m is not None and previous_error_m <= 0.25:
                response = {
                    "payload": {
                        "vehicle_name": vehicle_name,
                        "state": "skipped",
                        "reason": "event_target_unchanged",
                        "target_enu_m": [float(value) for value in command_position],
                        "previous_error_m": previous_error_m,
                        "synthetic": True,
                    }
                }
                wait_status = self._uav_status_snapshot(vehicle_name, command_position)
            else:
                response = self.client.move_runtime_multirotor(
                    vehicle_name,
                    target_enu_m=command_position,
                    velocity_mps=velocity_mps,
                    map_id=self.map_id,
                )
                wait_status = self._wait_for_uav_status(vehicle_name, command_position)
                self._ensure_uav_status_ok(wait_status, vehicle_name=vehicle_name, context=f"event action {action.get('action_id') or '<unnamed>'}")
            self.uav_active_by_entity[entity_id] = vehicle_name
            self.uav_last_command_target_by_entity[entity_id] = list(command_position)
            return {
                "status": "ok",
                "entity_id": entity_id,
                "vehicle_name": vehicle_name,
                "command_position_enu_m": command_position,
                "response": response,
                "wait_status": wait_status,
            }
        response = self.client.move_asset(payload, map_id=self.map_id)
        return {"status": "ok", "entity_id": entity_id, "asset_id": asset_instance_id, "response": response}

    def _script_remove_entity(self, action: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(action["entity_id"])
        logical_asset_id = self._script_entity_logical_asset_id(entity_id, action)
        asset_kind = self._script_asset_kind(logical_asset_id)
        if asset_kind == "pedestrian":
            response = self.client.ped_release(entity_id, map_id=self.map_id)
            self.event_controlled_entity_ids.discard(entity_id)
            self.ped_active_ids.discard(entity_id)
            return {"status": "ok", "entity_id": entity_id, "response": response}
        if asset_kind == "uav":
            vehicle_name = self.uav_active_by_entity.pop(entity_id, entity_id)
            response = self.client.remove_runtime_vehicle(vehicle_name, map_id=self.map_id)
            self.event_controlled_entity_ids.discard(entity_id)
            self.uav_last_command_target_by_entity.pop(entity_id, None)
            return {"status": "ok", "entity_id": entity_id, "vehicle_name": vehicle_name, "response": response}
        response = self.client.remove_asset(entity_id, map_id=self.map_id)
        self.event_controlled_entity_ids.discard(entity_id)
        return {"status": "ok", "entity_id": entity_id, "response": response}

    def _script_play_animation(self, action: dict[str, Any]) -> dict[str, Any]:
        ped_id = str(action["ped_id"])
        anim_path = str(action["animation_path"])
        response = self.client.ped_play_animation(
            ped_id,
            anim_path,
            start_section=str(action.get("start_section", "")),
            play_rate=float(action.get("play_rate", 1.0)),
            loop_count=max(1, int(action.get("loop_count", 1))),
            map_id=self.map_id,
        )
        return {"status": "ok", "ped_id": ped_id, "response": response}

    def _script_set_pedestrian_activity(self, action: dict[str, Any]) -> dict[str, Any]:
        ped_id = str(action.get("entity_id") or action.get("ped_id") or "").strip()
        activity_type = str(action.get("activity_type") or "").strip().lower()
        if not ped_id:
            raise ValueError("set_pedestrian_activity requires entity_id")
        if not activity_type:
            raise ValueError("set_pedestrian_activity requires activity_type")
        self.event_pedestrian_activity_state[ped_id] = activity_type
        if self.event_interpreter is not None:
            self.event_interpreter.update_entity_activity(ped_id, activity_type)
        return {
            "status": "ok",
            "ped_id": ped_id,
            "activity_type": activity_type,
            "command": "python_activity_state_only",
        }

    def _script_spawn_crowd(self, action: dict[str, Any]) -> dict[str, Any]:
        group_id = str(action["group_id"])
        count = int(action["count"])
        origin = self._script_transform_position(action["spawn_origin_enu_m"], "spawn_origin_enu_m")
        extent = list(action.get("spawn_box_extent_cm", [500.0, 500.0, 0.0]))
        response = self.client.ped_spawn_crowd(
            group_id,
            count,
            spawn_origin_enu_m=origin,
            seed=int(action.get("seed", 0)),
            spawn_box_extent_cm=[float(extent[0]), float(extent[1]), float(extent[2]) if len(extent) > 2 else 0.0],
            appearance_pool_path=str(action.get("appearance_pool_path", "")),
            role_profile_path=str(action.get("role_profile_path", "")),
            map_id=self.map_id,
        )
        self.crowd_group_ids.add(group_id)
        return {"status": "ok", "group_id": group_id, "response": response}

    def _script_clear_crowd(self, action: dict[str, Any]) -> dict[str, Any]:
        group_id = str(action["group_id"])
        response = self.client.ped_clear_crowd(group_id, map_id=self.map_id)
        self.crowd_group_ids.discard(group_id)
        return {"status": "ok", "group_id": group_id, "response": response}

    def _script_set_visual_state(self, action: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(action["entity_id"])
        visual_state: dict[str, Any] = dict(action.get("visual_state") or {})
        for key in ("mode", "lights_on", "material_variant", "montage_tag"):
            if key in action:
                visual_state[key] = action[key]
        logical_asset_id = self._script_entity_logical_asset_id(entity_id, action)
        if self._script_asset_kind(logical_asset_id) == "uav":
            vehicle_name = self.uav_active_by_entity.get(entity_id)
            if not vehicle_name:
                raise RuntimeError(f"set_visual_state for UAV '{entity_id}' requires an active runtime multirotor")
            mode = str(visual_state.get("mode") or "").strip().lower()
            if mode == "hover":
                raw_position = self.event_entity_initial_positions.get(entity_id)
                if not raw_position:
                    pose_response = self.client.get_runtime_vehicle_pose(vehicle_name, map_id=self.map_id)
                    pose_payload = dict(pose_response.get("payload") or {})
                    pose_position = pose_payload.get("position_enu_m")
                    if not isinstance(pose_position, Sequence) or isinstance(pose_position, (str, bytes)):
                        raise RuntimeError(f"Cannot resolve current runtime UAV pose for hover: {entity_id}")
                    command_position = [
                        float(pose_position[0]),
                        float(pose_position[1]),
                        float(pose_position[2] if len(pose_position) > 2 else 0.0),
                    ]
                else:
                    command_position = self._runtime_uav_command_position_enu(entity_id, raw_position)
                wait_status = self._uav_status_snapshot(vehicle_name, command_position)
                self.event_controlled_entity_ids.add(entity_id)
                self.uav_last_command_target_by_entity[entity_id] = list(command_position)
                return {
                    "status": "ok",
                    "entity_id": entity_id,
                    "vehicle_name": vehicle_name,
                    "visual_state": visual_state,
                    "command": "runtime_multirotor_hold_state",
                    "command_position_enu_m": command_position,
                    "response": {"status": "skipped", "reason": "hover_does_not_issue_second_move"},
                    "wait_status": wait_status,
                }
            return {
                "status": "ok",
                "entity_id": entity_id,
                "vehicle_name": vehicle_name,
                "visual_state": visual_state,
                "runtime_uav_state_only": True,
            }
        payload: dict[str, Any] = {
            "asset_id": self._script_asset_instance_id(action, entity_id),
            "entity_id": entity_id,
            "visual_state": visual_state,
        }
        response = self.client.move_asset(payload, map_id=self.map_id)
        return {"status": "ok", "entity_id": entity_id, "visual_state": visual_state, "response": response}

    def _find_capture_preset_by_id(self, camera_id: str) -> dict[str, Any] | None:
        ground_cameras = self.capture_presets.get("ground_cameras") or {}
        if isinstance(ground_cameras, dict):
            direct = ground_cameras.get(camera_id)
            if isinstance(direct, dict):
                return dict(direct)
            for value in ground_cameras.values():
                if isinstance(value, list):
                    for preset in value:
                        if isinstance(preset, dict) and str(preset.get("camera_id", preset.get("camera_name", ""))) == camera_id:
                            return dict(preset)
                elif isinstance(value, dict) and str(value.get("camera_id", value.get("camera_name", ""))) == camera_id:
                    return dict(value)
        for site_cfg in (self.capture_presets.get("sites") or {}).values():
            for preset in site_cfg.get("ground_cameras") or []:
                if str(preset.get("camera_id", preset.get("camera_name", ""))) == camera_id:
                    return dict(preset)
        for camera in self.event_scene_setup.get("cameras") or []:
            if not isinstance(camera, dict):
                continue
            if str(camera.get("camera_id", camera.get("camera_name", ""))) != camera_id:
                continue
            placement = dict(camera.get("placement") or {})
            position = placement.get("resolved_position_enu_m") or placement.get("position_enu_m")
            if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) < 3:
                return None
            preset = {
                "camera_id": camera_id,
                "camera_name": camera_id,
                "position_enu_m": [float(position[0]), float(position[1]), float(position[2])],
                "rotation_deg": dict(placement.get("rotation_deg") or camera.get("rotation_deg") or {}),
                "fov_degrees": float(camera.get("fov_deg", camera.get("fov_degrees", 90.0))),
                "coordinate_space": "map_enu",
                "resolution": {
                    "width": int(camera.get("width", 1280)),
                    "height": int(camera.get("height", 720)),
                },
            }
            return preset
        return None

    def _script_capture_screenshot(self, action: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(action.get("camera_id", "event_camera"))
        preset = self._find_capture_preset_by_id(camera_id)
        if preset is None:
            return {"status": "skipped", "reason": f"camera preset '{camera_id}' not found"}

        position_enu_m = self._script_transform_position(preset["position_enu_m"])
        rotation_deg = self._transform_rotation_deg(dict(preset.get("rotation_deg") or {}))
        fov = float(preset.get("fov", preset.get("fov_degrees", 90.0)))
        resolution = preset.get("resolution", {"width": preset.get("width", 1920), "height": preset.get("height", 1080)})

        asset_key = ("event", camera_id)
        asset_id = self.ground_camera_asset_ids.get(asset_key)
        if asset_id is None:
            asset_id = safe_name(f"event_camera.{camera_id}")
            spawn_payload = {
                "asset_id": asset_id,
                "logical_asset_id": "camera.fixed_world_capture.rgb.v1",
                "pose_enu_m": {"position_enu_m": position_enu_m, "rotation_deg": rotation_deg},
            }
            spawn_resp = self.client.spawn_asset(spawn_payload, map_id=self.map_id)
            asset_id = str((spawn_resp.get("payload") or {}).get("asset_id") or asset_id)
            self.ground_camera_asset_ids[asset_key] = asset_id

        if asset_id:
            capture_payload = {
                "asset_id": asset_id,
                "width": int(resolution.get("width", 1920)),
                "height": int(resolution.get("height", 1080)),
                "fov": fov,
            }
            self.client.capture_world_camera(capture_payload, map_id=self.map_id)

        return {"status": "ok", "camera_id": camera_id, "asset_id": asset_id or ""}

    def run_batch(self, batch: BatchPlan) -> None:
        batch_ticks = self._batch_ticks(batch)
        capture_ticks = self._batch_capture_tick_set(batch)
        self._select_airsim_capture_entity(batch, capture_ticks)
        print(
            f"[EpisodeHost] Starting batch {batch.batch_id} "
            f"site={batch.site_id} roi={batch.roi_id or '<none>'} "
            f"ticks={batch.tick_start}..{batch.tick_end} "
            f"({len(batch_ticks)} simulation ticks, {len(capture_ticks)} capture frames)"
        )
        self._soft_reset_tracking()
        self.last_weather_signature = None
        for tick in batch_ticks:
            frame = self.frames_by_tick[tick]
            weather_payload = self._apply_weather_if_needed(tick)
            _, _, _, entity_records = self._apply_scene_frame(frame)
            _ = self._sync_pedestrians(frame)
            uav_status = self._sync_uavs(frame)
            uav_debug = self._collect_runtime_uav_debug(frame, uav_status)

            # --- Event script interpreter ---
            if self.event_interpreter is not None:
                self.event_interpreter.update_weather_state(weather_payload)
                for entity_id, position in self.event_entity_initial_positions.items():
                    if entity_id not in self.event_interpreter.entity_states:
                        self.event_interpreter.update_entity_state(entity_id, position)
                for rec in entity_records:
                    rec_entity_id = str(rec.get("entity_id", ""))
                    if rec_entity_id in self.event_controlled_entity_ids:
                        continue
                    self.event_interpreter.update_entity_state(
                        rec_entity_id,
                        rec.get("position_enu_m", [0.0, 0.0, 0.0]),
                        rotation_deg=rec.get("rotation_deg"),
                        velocity_enu_mps=rec.get("velocity_enu_mps"),
                    )
                event_results = self.event_interpreter.tick(tick)
                if event_results:
                    print(f"[EpisodeHost] Event actions at tick {tick}: {len(event_results)} action(s)")
                    failed_event_actions: list[dict[str, Any]] = []
                    for entry in event_results:
                        result = dict(entry.get("result") or {})
                        status = str(result.get("status") or "").strip().lower()
                        if status in {"error", "failed", "skipped"}:
                            failed_event_actions.append(entry)
                            continue
                        wait_status = result.get("wait_status")
                        if isinstance(wait_status, dict):
                            wait_payload = dict(wait_status.get("status") or {})
                            wait_state = str(wait_payload.get("state") or wait_payload.get("status") or "").strip().lower()
                            timed_out_accepted = bool(wait_status.get("timed_out_accepted_running"))
                            if (bool(wait_status.get("timed_out")) and not timed_out_accepted) or wait_state in {"failed", "cancelled", "timeout", "missing"}:
                                failed_event_actions.append(entry)
                    if failed_event_actions:
                        summary = [
                            {
                                "event_id": item.get("event_id", ""),
                                "action_id": item.get("action_id", ""),
                                "type": item.get("type", ""),
                                "result": item.get("result", {}),
                            }
                            for item in failed_event_actions[:5]
                        ]
                        raise RuntimeError(
                            f"Event action failure at tick {tick}: "
                            + json.dumps(summary, ensure_ascii=False, default=str)
                        )

            if tick in capture_ticks:
                feedback_payload = self._poll_feedback()
                self._capture_ground_views(batch, frame, weather_payload, entity_records, feedback_payload, uav_status, uav_debug)
                self._capture_uav_views(batch, frame, weather_payload, entity_records, feedback_payload, uav_status, uav_debug)

    def run(self) -> None:
        if self.client is None:
            self.connect()
        ensure_dir(self.output_dir)
        manifest_path = self._write_capture_storage_manifest()
        print(f"[EpisodeHost] Capture storage manifest written: {manifest_path}")
        if bool(getattr(self.args, "semantic_stencil_audit_only", False)):
            audit_path = self.output_dir / "semantic_stencil_audit.json"
            self._fixed_world_capture_hook().semantic_stencil_audit(
                map_id=self.map_id,
                semantic_rules_path=self.semantic_rules_path,
                semantic_audit_path=audit_path,
                assign=False,
            )
            print(f"[EpisodeHost] Semantic stencil audit written: {audit_path}")
            return
        if bool(getattr(self.args, "segmentation_registry_audit_only", False)):
            payload = self._configure_semantic_segmentation_registry([])
            audit_path = self.output_dir / "semantic_segmentation_registry_audit.json"
            audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[EpisodeHost] Semantic segmentation registry audit written: {audit_path}")
            return
        self._cleanup_pie_world_actors()
        self.hard_reset_world_state()
        try:
            for index, batch in enumerate(self.batch_plans):
                self.run_batch(batch)
                if index + 1 < len(self.batch_plans):
                    self.hard_reset_world_state()

            # Export event trace from script interpreter
            if self.event_interpreter is not None:
                from donghu_core.artifact_writer import write_jsonl as _write_jsonl_util
                event_log = self.event_interpreter.get_event_log()
                if event_log:
                    event_path = Path(self.output_dir) / "event_trace.jsonl"
                    _write_jsonl_util(event_path, event_log)
                    print(f"[EpisodeHost] Event trace written: {event_path} ({len(event_log)} events)")
        finally:
            if self.client is not None:
                for asset_id in sorted(set(self.ground_camera_asset_ids.values())):
                    try:
                        self._retry("remove_asset", self.client.remove_asset, asset_id, map_id=self.map_id)
                    except Exception as exc:
                        print(f"[EpisodeHost] remove_asset warning for {asset_id}: {exc}")
            self.hard_reset_world_state()
            if self.fixed_world_capture_hook is not None:
                self.fixed_world_capture_hook.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Episode render host for AeroSimHost PIE playback. "
            "Manual prerequisite: open the project, load the correct level, and enter PIE before running this script."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Host config JSON path")
    parser.add_argument("--host", default="127.0.0.1", help="AirSim RPC host")
    parser.add_argument("--port", type=int, default=41451, help="AirSim RPC port")
    parser.add_argument("--map_id", default="", help="Override map_id from config/truth")
    parser.add_argument("--output_dir", default="", help="Override output directory")
    parser.add_argument("--site", default="", help="Run only one site contract")
    parser.add_argument("--batch_id", default="", help="Run only one batch_id")
    parser.add_argument("--start_tick", type=int, default=None, help="Override batch start tick filter")
    parser.add_argument("--end_tick", type=int, default=None, help="Override batch end tick filter")
    parser.add_argument("--max_batches", type=int, default=0, help="Run only the first N matching batches")
    parser.add_argument("--tick_stride", type=int, default=1, help="Capture every Nth tick. Simulation still updates every tick by default")
    parser.add_argument("--simulation_tick_stride", type=int, default=1, help="Advance runtime state every Nth tick; keep 1 for smooth actor motion")
    parser.add_argument(
        "--capture_tick",
        action="append",
        type=int,
        default=[],
        help="Capture only this absolute tick while still simulating the full requested tick range. Repeatable.",
    )
    parser.add_argument(
        "--camera-role",
        action="append",
        choices=["all", "ground", "uav"],
        default=[],
        help="Capture only this camera role. Repeatable. Default captures all roles.",
    )
    parser.add_argument(
        "--camera-id",
        action="append",
        default=[],
        help="Capture only this camera id/name/suffix/entity. Repeatable.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=[],
        help="Capture only this modality id, e.g. rgb, depth, or seg. Repeatable.",
    )
    parser.add_argument(
        "--uav-capture-backend",
        choices=["editor_hook", "airsim_native"],
        default="airsim_native",
        help="Backend for UAV camera captures. Ground cameras still use editor hook RGB.",
    )
    parser.add_argument(
        "--segmentation-backend",
        choices=["ue_custom_stencil", "airsim_native"],
        default="ue_custom_stencil",
        help="Backend for UAV seg captures. Default uses UE CustomDepth/CustomStencil instead of AirSim ImageType.Segmentation.",
    )
    parser.add_argument(
        "--semantic-rules-path",
        default="",
        help="Semantic CustomStencil rules JSON. Default: Config/LowAltitude/semantic_stencil_rules.json.",
    )
    parser.add_argument(
        "--airsim-capture-vehicle",
        default="CaptureUAV_0",
        help="Single AirSim vehicle used as the native UAV capture platform.",
    )
    parser.add_argument(
        "--airsim-capture-entity",
        default="",
        help="Runtime UAV entity id to replace with the AirSim capture platform. Default selects the first active UAV.",
    )
    parser.add_argument(
        "--capture-view-id",
        default="",
        help="Stable deterministic view id used as the UAV capture output subdirectory.",
    )
    parser.add_argument(
        "--write-depth-preview",
        action="store_true",
        help="Write an extra 8-bit PNG preview next to depth .npy for smoke/debug only. Formal dataset generation should leave this off.",
    )
    parser.add_argument(
        "--segmentation-registry-audit-only",
        action="store_true",
        help="Deprecated AirSim segmentation registry audit path. Prefer --semantic-stencil-audit-only.",
    )
    parser.add_argument(
        "--semantic-stencil-audit-only",
        action="store_true",
        help="Write a read-only UE CustomStencil semantic component audit JSON and exit without capturing images.",
    )
    parser.add_argument(
        "--enable-unsafe-pie-segmentation-actor-registration",
        action="store_true",
        help=(
            "Unsafe diagnostic switch: call AirSim SimMode add_new_actor_to_instance_segmentation for matched PIE actors. "
            "Disabled by default because prior UE logs show this path can exhaust editor memory."
        ),
    )
    parser.add_argument("--dry_run_coords", action="store_true", help="Print raw/transformed ENU coordinates and exit")
    parser.add_argument("--preview_ground", action="store_true", help="Connect to PIE, print transformed + ground-projected ENU coordinates, and exit")
    parser.add_argument("--coord_preview_tick", type=int, default=None, help="Tick used by --dry_run_coords")
    parser.add_argument("--coord_preview_limit", type=int, default=10, help="Entity/camera preview count for --dry_run_coords")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = EpisodeRenderHost(Path(args.config), args)
    if args.preview_ground:
        host.connect()
        host.dump_coordinate_preview(include_ground=True)
        return
    if args.dry_run_coords:
        host.dump_coordinate_preview()
        return
    host.run()


if __name__ == "__main__":
    main()
