#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
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


def get_image_type_enum(airsim_module: Any, name: str) -> Any:
    clean = (name or "Scene").strip()
    if hasattr(airsim_module.ImageType, clean):
        return getattr(airsim_module.ImageType, clean)
    if clean.isdigit():
        return int(clean)
    raise ValueError(f"Unsupported AirSim image type: {name}")


def vector3r_to_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "x_val"):
        return [float(value.x_val), float(value.y_val), float(value.z_val)]
    if hasattr(value, "x"):
        return [float(value.x()), float(value.y()), float(value.z())]
    if isinstance(value, Sequence):
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def quaternion_to_dict(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if hasattr(value, "w_val"):
        return {
            "w": float(value.w_val),
            "x": float(value.x_val),
            "y": float(value.y_val),
            "z": float(value.z_val),
        }
    if hasattr(value, "w"):
        return {
            "w": float(value.w()),
            "x": float(value.x()),
            "y": float(value.y()),
            "z": float(value.z()),
        }
    return None


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
        self.pie_scene_cleanup_cfg = dict(self.config.get("pie_scene_cleanup") or {})
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
        self.pedestrian_pose_service = PedestrianPoseService()
        self.uav_execution_service = UavExecutionService(
            arrival_tolerance_m=float((self.config.get("runtime_uav") or {}).get("arrival_tolerance_m", 1.5)),
            hover_before_capture=False,
        )

        self.scene_active_ids: set[str] = set()
        self.ped_active_ids: set[str] = set()
        self.ped_last_activity: dict[str, str] = {}
        self.ped_last_variant: dict[str, str] = {}
        self.uav_active_by_entity: dict[str, str] = {}

        self.all_scene_sync_ids, self.all_ped_ids, self.all_uav_vehicle_names = self._discover_entity_modes()
        self.batch_plans = self._build_batches()

        self.airsim: Any | None = None
        self.client: AeroSimClient | None = None
        self.fixed_world_capture_hook: FixedWorldCaptureEditorHook | None = None
        self.ground_camera_asset_ids: dict[tuple[str, str], str] = {}
        self.runtime_uav_direct_rpc_enabled = True
        self.runtime_uav_direct_rpc_disable_reason = ""
        self.runtime_uav_debug_cfg = dict(self.config.get("runtime_uav_debug") or {})
        self.runtime_uav_debug_entity_ids = {
            str(value).strip()
            for value in (self.runtime_uav_debug_cfg.get("entity_ids") or [])
            if str(value).strip()
        }
        self.event_weather_overlay: dict[str, Any] = {}
        self.event_scene_setup: dict[str, Any] = {}
        self.event_entity_assets: dict[str, str] = {}
        self.event_entity_initial_positions: dict[str, list[float]] = {}

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

    def _transform_position_enu(self, position_enu_m: Sequence[float]) -> list[float]:
        return self.coordinate_transform.apply_position(position_enu_m)

    def _transform_rotation_deg(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        return self.coordinate_transform.apply_rotation(rotation_deg)

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
        return list((ground_details or {}).get("ground_relative_enu_m") or resolved)

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
            try:
                response = self._retry("project_ground", self.client.project_ground, point_enu_m=resolved, map_id=self.map_id)
                payload = dict(response.get("payload") or {})
                if use_cache:
                    self.ground_projection_cache[cache_key] = dict(payload)
            except Exception as exc:
                warning_key = cache_namespace or f"{cache_key[1]}:{cache_key[2]}"
                if warning_key not in self.ground_projection_warning_keys:
                    print(
                        "[EpisodeHost] project_ground warning "
                        f"namespace={cache_namespace or '<none>'} "
                        f"position_enu_m={resolved} "
                        f"error={exc} "
                        "falling back to the original ENU position."
                    )
                    self.ground_projection_warning_keys.add(warning_key)
                fallback = {
                    "input_enu_m": resolved,
                    "projected_enu_m": list(resolved),
                    "ground_relative_enu_m": list(resolved),
                    "surface_normal_enu": [0.0, 0.0, 1.0],
                    "anchor_id": "",
                    "ground_resolved": False,
                    "fallback_reason": str(exc),
                }
                if use_cache:
                    self.ground_projection_cache[cache_key] = dict(fallback)
                return fallback

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
        return {
            "input_enu_m": resolved,
            "projected_enu_m": projected_enu_m,
            "surface_normal_enu": surface_normal_enu,
            "anchor_id": anchor_id,
            "ground_resolved": ground_resolved,
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
            resolved_position = self._transform_position_enu(raw_position)
            raw_rotation = dict(preset.get("rotation_deg") or {})
            resolved_rotation = self._transform_rotation_deg(raw_rotation)
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
                self._disable_runtime_uav_direct_rpc(str(exc))
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
                position_enu_m=record["position_enu_m"],
                rotation_deg=record["rotation_deg"],
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

            # Keep truth XY authoritative for pedestrians. Bridge-level ground snapping can
            # reproject to semantic road anchors and drift fallen pedestrians away from the
            # incident point we intentionally aligned the camera to.
            snap_to_ground = bool(activity_rule.get("snap_to_ground", False))
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
                    map_id=self.map_id,
                )
                results[ped_id] = {"spawn": spawn_response.get("payload", {})}
                pose_sync_performed = True
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
                    "ped_reset",
                    self.client.ped_reset,
                    ped_id,
                    position_enu_m=position_enu_m,
                    yaw_deg=float(rotation_deg["yaw_deg"]),
                    snap_to_ground=snap_to_ground,
                    map_id=self.map_id,
                )
                results[ped_id] = {"reset": reset_response.get("payload", {})}
                pose_sync_performed = True

            if self.ped_last_variant.get(ped_id) != variant_id:
                variant_response = self._retry("ped_set_variant", self.client.ped_set_variant, ped_id, variant_id, map_id=self.map_id)
                results.setdefault(ped_id, {})["variant"] = variant_response.get("payload", {})
                self.ped_last_variant[ped_id] = variant_id

            should_reapply_activity = bool(activity_rule.get("reapply_after_pose_sync", False)) and pose_sync_performed
            if self.ped_last_activity.get(ped_id) != activity_type or should_reapply_activity:
                results.setdefault(ped_id, {})["activity"] = self._ped_activity_action(ped_id, activity_type)
                self.ped_last_activity[ped_id] = activity_type

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
            target_enu_m = self._entity_position_enu(entity)
            target_enu_m = self._ground_relative_position(
                target_enu_m,
                enabled=bool(self.ground_reference_cfg.get("uav_ground_relative", False)),
                cache_namespace=f"uav:{vehicle_name}",
                use_cache=False,
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
                        wait_status = {
                            "vehicle_name": vehicle_name,
                            "status": {"state": "unknown", "warning": str(exc)},
                            "pose": {},
                            "timed_out": False,
                            "position_error_m": None,
                        }
                    wait_status["create"] = create_response.get("payload", {})
                    wait_status["move"] = move_response.get("payload", {})
                    wait_status["path_used"] = "direct_rpc"
                    results[entity_id] = wait_status
            else:
                used_editor_hook = False
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
                    except Exception as exc:
                        print(f"[EpisodeHost] UAV status warning for {vehicle_name} after move: {exc}")
                        wait_status = {
                            "vehicle_name": vehicle_name,
                            "status": {"state": "unknown", "warning": str(exc)},
                            "pose": {},
                            "timed_out": False,
                            "position_error_m": None,
                        }
                    wait_status["move"] = move_response.get("payload", {})
                    wait_status["path_used"] = "direct_rpc"
                    results[entity_id] = wait_status

            self.uav_active_by_entity[entity_id] = vehicle_name

        for entity_id in sorted(set(self.uav_active_by_entity) - current_entities):
            vehicle_name = self.uav_active_by_entity.pop(entity_id)
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
        return [dict(item) for item in (ground.get(site_id) or ground.get("default") or [])]

    def _runtime_uav_visibility_filter_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("runtime_uav_visibility_filter") or {})

    def _ground_camera_contains_position(self, preset: dict[str, Any], position_enu_m: Sequence[float]) -> bool:
        camera_position = list(preset.get("position_enu_m") or [])
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
                use_cache=False,
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
        class_prefixes = [str(value).strip() for value in (cfg.get("hide_actor_class_prefixes") or []) if str(value).strip()]
        name_keywords = [str(value).strip().lower() for value in (cfg.get("hide_actor_name_keywords") or []) if str(value).strip()]
        skip_prefixes = [str(value).strip() for value in (cfg.get("skip_actor_class_prefixes") or []) if str(value).strip()]
        if not class_prefixes and not name_keywords:
            return

        request = {
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
    hay = (name + " " + label + " " + cls).lower()
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

    def _capture_requests(self, camera_name: str, modality_ids: Sequence[str]) -> list[Any]:
        airsim = self._import_airsim()
        modalities = dict(self.capture_presets.get("modalities") or {})
        requests: list[Any] = []
        for modality_id in modality_ids:
            modality = dict(modalities[modality_id])
            requests.append(
                airsim.ImageRequest(
                    camera_name,
                    get_image_type_enum(airsim, str(modality["image_type"])),
                    bool(modality.get("pixels_as_float", False)),
                    bool(modality.get("compress", True)),
                )
            )
        return requests

    def _set_camera_fov(self, camera_name: str, fov_degrees: float, vehicle_name: str = "") -> None:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        raw_client = self.client.client
        if hasattr(raw_client, "simSetCameraFov"):
            raw_client.simSetCameraFov(camera_name, float(fov_degrees), vehicle_name)
            return
        if hasattr(raw_client, "simSetCameraFoV"):
            raw_client.simSetCameraFoV(camera_name, float(fov_degrees), vehicle_name)
            return
        raise RuntimeError("The active AirSim Python client does not expose simSetCameraFov/FoV.")

    def _capture_images(self, raw_client: Any, requests: Sequence[Any], *, vehicle_name: str, label: str) -> Sequence[Any]:
        timeout_cfg = dict(self.config.get("timeouts") or {})
        capture_warmup_s = float(timeout_cfg.get("capture_warmup_s", 8.0))
        camera_settle_s = float(timeout_cfg.get("camera_settle_s", 0.35))
        retries = int(timeout_cfg.get("image_retry_count", 2))
        retry_delay_s = float(timeout_cfg.get("image_retry_delay_s", 5.0))

        if not self.capture_warmup_complete and capture_warmup_s > 0.0:
            print(f"[EpisodeHost] Waiting {capture_warmup_s:.1f}s before first image capture to let PIE rendering settle.")
            time.sleep(capture_warmup_s)
            self.capture_warmup_complete = True

        for attempt in range(retries + 1):
            if camera_settle_s > 0.0:
                time.sleep(camera_settle_s)
            try:
                return raw_client.simGetImages(requests, vehicle_name=vehicle_name)
            except Exception as exc:
                if attempt >= retries:
                    raise
                print(f"[EpisodeHost] {label} failed (attempt {attempt + 1}/{retries + 1}): {exc}")
                time.sleep(retry_delay_s)
        raise RuntimeError(f"{label} failed without an exception")

    def _write_capture_set(self, batch: BatchPlan, frame: dict[str, Any], camera_id: str, camera_role: str, modality_ids: Sequence[str], responses: Sequence[Any], common_sidecar: dict[str, Any]) -> None:
        modalities = dict(self.capture_presets.get("modalities") or {})
        frame_stem = self.capture_orchestrator.frame_stem(frame)
        for index, modality_id in enumerate(modality_ids):
            modality = dict(modalities[modality_id])
            output_dir = self.capture_orchestrator.modality_output_dir(
                self.output_dir,
                batch.batch_id,
                safe_name(camera_id),
                safe_name(modality_id),
            )
            image_path = output_dir / f"{frame_stem}.{str(modality.get('extension', 'png'))}"
            sidecar_path = output_dir / f"{frame_stem}.json"

            response = responses[index] if index < len(responses) else None
            capture_error = None
            image_data_format = "uint8"
            if response is None:
                capture_error = "missing response"
            else:
                width = int(getattr(response, "width", 0) or 0)
                height = int(getattr(response, "height", 0) or 0)
                if bool(modality.get("pixels_as_float", False)):
                    image_data_format = "float32"
                    payload = getattr(response, "image_data_float", None)
                    if payload and width > 0 and height > 0:
                        try:
                            import numpy as np  # type: ignore
                        except Exception as exc:
                            capture_error = f"numpy import failed for float image write: {exc}"
                        else:
                            values = np.asarray(payload, dtype=np.float32)
                            if values.size != width * height:
                                capture_error = (
                                    f"unexpected float image payload length: "
                                    f"expected {width * height}, got {values.size}"
                                )
                            else:
                                np.save(image_path, values.reshape((height, width)))
                    else:
                        capture_error = str(getattr(response, "message", "") or "empty float image payload")
                else:
                    payload = getattr(response, "image_data_uint8", None)
                    if payload:
                        image_path.write_bytes(bytes(payload))
                    else:
                        capture_error = str(getattr(response, "message", "") or "empty image payload")

            sidecar = dict(common_sidecar)
            sidecar.update(
                {
                    "camera_id": camera_id,
                    "camera_role": camera_role,
                    "modality": modality_id,
                    "image_type": modality.get("image_type"),
                    "pixels_as_float": bool(modality.get("pixels_as_float", False)),
                    "compress": bool(modality.get("compress", True)),
                    "image_data_format": image_data_format,
                    "image_path": str(image_path),
                    "output_path": str(image_path),
                    "capture_error": capture_error,
                }
            )
            if response is not None:
                sidecar["airsim_camera_position"] = vector3r_to_list(getattr(response, "camera_position", None))
                sidecar["airsim_camera_orientation"] = quaternion_to_dict(getattr(response, "camera_orientation", None))
                sidecar["width"] = int(getattr(response, "width", 0) or 0)
                sidecar["height"] = int(getattr(response, "height", 0) or 0)
                sidecar["response_message"] = str(getattr(response, "message", "") or "")
            self.capture_orchestrator.write_sidecar(sidecar_path, sidecar)

    def _write_fixed_world_capture_output(
        self,
        batch: BatchPlan,
        frame: dict[str, Any],
        camera_id: str,
        modality_id: str,
        image_path: Path,
        common_sidecar: dict[str, Any],
        *,
        width: int,
        height: int,
        fov_degrees: float,
    ) -> None:
        modalities = dict(self.capture_presets.get("modalities") or {})
        modality = dict(modalities.get(modality_id) or {})
        sidecar_path = image_path.with_suffix(".json")
        sidecar = dict(common_sidecar)
        sidecar.update(
            {
                "camera_id": camera_id,
                "camera_role": "ground",
                "modality": modality_id,
                "image_type": modality.get("image_type", "Scene"),
                "image_path": str(image_path),
                "output_path": str(image_path),
                "capture_error": None,
                "capture_backend": "editor_hook_fixed_world_camera",
                "width": int(width),
                "height": int(height),
                "fov_degrees": float(fov_degrees),
            }
        )
        self.capture_orchestrator.write_sidecar(sidecar_path, sidecar)

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
        hook = self._fixed_world_capture_hook()
        modalities = dict(self.capture_presets.get("modalities") or {})
        for preset in self._site_ground_presets(batch.site_id):
            camera_name = str(preset.get("camera_name", "external"))
            camera_id = str(preset.get("camera_id", camera_name))
            modality_ids = list(preset.get("modalities") or self.capture_presets.get("default_modalities") or ["rgb"])
            unsupported_modalities = [modality_id for modality_id in modality_ids if str(modality_id).strip().lower() != "rgb"]
            if unsupported_modalities:
                raise RuntimeError(
                    "Fixed world capture currently supports ground modality 'rgb' only. "
                    f"Unsupported modalities for camera '{camera_id}': {unsupported_modalities}"
                )
            raw_camera_position_enu_m = [float(value) for value in (preset.get("position_enu_m") or [0.0, 0.0, 0.0])]
            resolved_camera_position_enu_m = self._transform_position_enu(raw_camera_position_enu_m)
            resolved_camera_position_enu_m = self._ground_relative_position(
                resolved_camera_position_enu_m,
                enabled=bool(self.ground_reference_cfg.get("ground_camera_ground_relative", False)),
                cache_namespace=f"ground_camera:{batch.site_id}:{camera_id}",
                use_cache=True,
            )
            raw_camera_rotation_deg = dict(preset.get("rotation_deg") or {})
            raw_pitch_deg = float(raw_camera_rotation_deg.get("pitch_deg", raw_camera_rotation_deg.get("pitch", 0.0)))
            if abs(abs(raw_pitch_deg) - 90.0) <= 1e-3:
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
            ensure_dir(output_dir)
            image_path = output_dir / f"{frame_stem}.{str((modalities.get('rgb') or {}).get('extension', 'png'))}"
            hook.capture_rgb(
                map_id=self.map_id,
                asset_id=asset_id,
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
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        raw_client = self.client.client
        for entity_id, vehicle_status in sorted(uav_status.items()):
            entity = self.roster_by_id.get(entity_id) or {}
            if batch.site_id and str(entity.get("site_id", "")) not in {"", batch.site_id}:
                continue
            vehicle_name = str(vehicle_status.get("vehicle_name") or self.uav_active_by_entity.get(entity_id) or entity_id)
            for preset in self._uav_camera_presets():
                camera_name = str(preset.get("camera_name", "front_center"))
                modality_ids = list(preset.get("modalities") or self.capture_presets.get("default_modalities") or ["rgb", "depth", "seg"])
                responses = self._capture_images(
                    raw_client,
                    self._capture_requests(camera_name, modality_ids),
                    vehicle_name=vehicle_name,
                    label=f"simGetImages uav '{vehicle_name}' camera '{camera_name}'",
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
                    "vehicle_name": vehicle_name,
                    "camera_name": camera_name,
                    "entity_records": entity_records,
                    "roster_summary": frame.get("roster_summary", {}),
                }
                camera_id = f"{safe_name(entity_id)}__{safe_name(str(preset.get('camera_id_suffix', camera_name)))}"
                self._write_capture_set(batch, frame, camera_id, "uav", modality_ids, responses, sidecar)

    def _batch_ticks(self, batch: BatchPlan) -> list[int]:
        ticks = [tick for tick in self.sorted_ticks if batch.tick_start <= tick <= batch.tick_end]
        stride = max(1, int(getattr(self.args, "tick_stride", 1) or 1))
        return ticks[::stride]

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
        return self._transform_position_enu(self._coerce_script_vector3(position_enu_m, field_name))

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
        position = self._script_transform_position(action["position_enu_m"])
        rotation = self._script_transform_rotation(action)
        self.event_entity_assets[entity_id] = logical_asset_id
        self.event_entity_initial_positions[entity_id] = list(position)
        asset_kind = self._script_asset_kind(logical_asset_id)
        if asset_kind == "pedestrian":
            response = self.client.ped_spawn(
                entity_id,
                position_enu_m=position,
                yaw_deg=float(rotation.get("yaw_deg", 0.0)),
                map_id=self.map_id,
            )
            return {"status": "ok", "entity_id": entity_id, "ped_id": entity_id, "response": response}
        if asset_kind == "uav":
            response = self.client.create_runtime_multirotor(
                asset_instance_id,
                position_enu_m=position,
                rotation_deg=rotation,
                map_id=self.map_id,
            )
            self.uav_active_by_entity[entity_id] = asset_instance_id
            return {"status": "ok", "entity_id": entity_id, "vehicle_name": asset_instance_id, "response": response}
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
        if action.get("waypoints_enu_m") is not None or action.get("waypoints") is not None:
            transformed_waypoints = self._script_transform_waypoints(action.get("waypoints_enu_m", action.get("waypoints")))
            if not transformed_waypoints:
                return {"status": "skipped", "reason": "no waypoints", "entity_id": entity_id}
            position = list(transformed_waypoints[-1])
            fallback_yaw = (
                self._yaw_from_points(transformed_waypoints[-2], transformed_waypoints[-1])
                if len(transformed_waypoints) >= 2
                else None
            )
        elif "position_enu_m" in action:
            position = self._script_transform_position(action["position_enu_m"])
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
        if asset_kind == "pedestrian":
            response = self.client.ped_set_target(
                entity_id,
                target_enu_m=position,
                speed_cm_per_sec=float(action.get("velocity_mps", 1.5)) * 100.0,
                map_id=self.map_id,
            )
            return {"status": "ok", "entity_id": entity_id, "ped_id": entity_id, "response": response}
        if asset_kind == "uav":
            vehicle_name = self.uav_active_by_entity.get(entity_id, asset_instance_id)
            response = self.client.move_runtime_multirotor(
                vehicle_name,
                target_enu_m=position,
                velocity_mps=float(action.get("velocity_mps", 5.0)),
                map_id=self.map_id,
            )
            self.uav_active_by_entity[entity_id] = vehicle_name
            return {"status": "ok", "entity_id": entity_id, "vehicle_name": vehicle_name, "response": response}
        response = self.client.move_asset(payload, map_id=self.map_id)
        return {"status": "ok", "entity_id": entity_id, "asset_id": asset_instance_id, "response": response}

    def _script_remove_entity(self, action: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(action["entity_id"])
        logical_asset_id = self._script_entity_logical_asset_id(entity_id, action)
        asset_kind = self._script_asset_kind(logical_asset_id)
        if asset_kind == "pedestrian":
            response = self.client.ped_release(entity_id, map_id=self.map_id)
            return {"status": "ok", "entity_id": entity_id, "response": response}
        if asset_kind == "uav":
            vehicle_name = self.uav_active_by_entity.pop(entity_id, entity_id)
            response = self.client.remove_runtime_vehicle(vehicle_name, map_id=self.map_id)
            return {"status": "ok", "entity_id": entity_id, "vehicle_name": vehicle_name, "response": response}
        response = self.client.remove_asset(entity_id, map_id=self.map_id)
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
        return {"status": "ok", "group_id": group_id, "response": response}

    def _script_clear_crowd(self, action: dict[str, Any]) -> dict[str, Any]:
        group_id = str(action["group_id"])
        response = self.client.ped_clear_crowd(group_id, map_id=self.map_id)
        return {"status": "ok", "group_id": group_id, "response": response}

    def _script_set_visual_state(self, action: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(action["entity_id"])
        visual_state: dict[str, Any] = dict(action.get("visual_state") or {})
        for key in ("mode", "lights_on", "material_variant", "montage_tag"):
            if key in action:
                visual_state[key] = action[key]
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
        print(
            f"[EpisodeHost] Starting batch {batch.batch_id} "
            f"site={batch.site_id} roi={batch.roi_id or '<none>'} "
            f"ticks={batch.tick_start}..{batch.tick_end} ({len(batch_ticks)} frames)"
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
                    self.event_interpreter.update_entity_state(
                        str(rec.get("entity_id", "")),
                        rec.get("position_enu_m", [0.0, 0.0, 0.0]),
                        rotation_deg=rec.get("rotation_deg"),
                        velocity_enu_mps=rec.get("velocity_enu_mps"),
                    )
                event_results = self.event_interpreter.tick(tick)
                if event_results:
                    print(f"[EpisodeHost] Event actions at tick {tick}: {len(event_results)} action(s)")

            feedback_payload = self._poll_feedback()
            self._capture_ground_views(batch, frame, weather_payload, entity_records, feedback_payload, uav_status, uav_debug)
            self._capture_uav_views(batch, frame, weather_payload, entity_records, feedback_payload, uav_status, uav_debug)

    def run(self) -> None:
        if self.client is None:
            self.connect()
        ensure_dir(self.output_dir)
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
    parser.add_argument("--tick_stride", type=int, default=1, help="Process every Nth tick. Use 5 for 0.5s sampling at 10Hz")
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
