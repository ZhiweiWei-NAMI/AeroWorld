#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import bisect
import ctypes
import csv
import hashlib
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


def short_stable_name(value: str, prefix: str) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}{digest}"


def simple_capture_view_dir_name(view_id: str) -> str:
    match = re.search(r"uav_view_(\d+)", str(view_id))
    if match:
        return f"v{int(match.group(1)):03d}"
    return short_stable_name(str(view_id), "v")


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
        if category == "uav":
            merged = dict(self._uav_cfg)
            merged.update(resolved)
            merged["mode"] = "scene_sync"
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
                "SkyAtmosphere",
                "SkyLight",
            ],
            "sanitize_engine_sky_dome": True,
            "disable_sky_atmosphere_editor_notifications": True,
            "sky_dome_actor_keywords": [
                "sm_skysphere",
                "bp_sky_sphere",
            ],
            "sky_dome_component_path_keywords": [
                "enginesky/sm_skysphere",
                "enginesky/m_simpleskydome",
                "enginesky/skyatmosphere_materialskydome",
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
        self.scene_active_ids: set[str] = set()
        self.ped_active_ids: set[str] = set()
        self.ped_last_activity: dict[str, str] = {}
        self.ped_last_variant: dict[str, str] = {}
        self.uav_active_by_entity: dict[str, str] = {}
        self.uav_last_command_target_by_entity: dict[str, list[float]] = {}
        self.event_controlled_entity_ids: set[str] = set()

        self.all_scene_sync_ids, self.all_ped_ids = self._discover_entity_modes()
        self.batch_plans = self._build_batches()

        self.airsim: Any | None = None
        self.client: AeroSimClient | None = None
        self.fixed_world_capture_hook: FixedWorldCaptureEditorHook | None = None
        self.ground_camera_asset_ids: dict[tuple[str, str], str] = {}
        self.prepared_capture_output_dirs: set[Path] = set()
        self.crowd_group_ids: set[str] = set()
        self.asset_catalog_reload_attempted = False
        uav_scene_control_cfg = dict(self.config.get("uav_scene_control") or {})
        uav_scene_backend_arg = str(getattr(self.args, "uav_scene_control_backend", "") or "").strip().lower()
        configured_uav_scene_backend = str(uav_scene_control_cfg.get("backend") or "").strip().lower()
        requested_uav_scene_backend = uav_scene_backend_arg or str(
            configured_uav_scene_backend or "truth_frame_scene_sync"
        ).strip().lower()
        if requested_uav_scene_backend not in {"truth_frame_scene_sync"}:
            raise RuntimeError(
                "Formal dataset capture supports only truth_frame_scene_sync for non-capture UAVs; "
                f"got '{requested_uav_scene_backend}'. AirSim controls only the rotating capture vehicle."
            )
        self.uav_scene_control_backend = "truth_frame_scene_sync"
        self.uav_debug_cfg = dict(self.config.get("uav_debug") or {})
        self.uav_debug_entity_ids = {
            str(value).strip()
            for value in (self.uav_debug_cfg.get("entity_ids") or [])
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
        if self.segmentation_backend == "ue_custom_stencil" and not self.semantic_class_by_id:
            raise RuntimeError(
                "UE CustomStencil segmentation requires a semantic rules JSON with a non-empty classes object: "
                f"{self.semantic_rules_path}"
            )
        if self.segmentation_backend != "ue_custom_stencil":
            raise RuntimeError(
                "Formal UAV segmentation only supports UE CustomStencil. "
                "Use Plugins/SumoImporter/Scripts/dev_checks/airsim_segmentation_registry_audit.py "
                "for the read-only legacy AirSim registry diagnostic."
            )
        event_semantic_overlay_cfg = dict(self.config.get("event_semantic_overlays") or {})
        self.event_semantic_overlay_cfg = event_semantic_overlay_cfg
        self.logical_region_primary_segmentation_enabled = bool(
            event_semantic_overlay_cfg.get("logical_region_primary_segmentation_enabled", False)
        )
        self.airsim_capture_vehicle = str(getattr(self.args, "airsim_capture_vehicle", "CaptureUAV_0") or "CaptureUAV_0").strip()
        self.requested_airsim_capture_entity = str(getattr(self.args, "airsim_capture_entity", "") or "").strip()
        self.requested_capture_view_id = str(getattr(self.args, "capture_view_id", "") or "").strip()
        self.active_airsim_capture_entity_id = ""
        self.active_capture_view_id = ""
        self.airsim_capture_vehicle_ready = False
        self.airsim_capture_ned_origin_world_cm: list[float] | None = None
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
            else:
                spec_scene_setup = self._load_scene_setup_from_spec(script_path.with_name("spec.py"))
                if spec_scene_setup:
                    self.event_scene_setup = spec_scene_setup
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
        self.event_semantic_asset_ids = set()
        for entity in self._event_semantic_entities():
            logical_asset_id = str(entity.get("logical_asset_id") or "")
            entity_id = str(entity.get("entity_id") or entity.get("instance_id") or "")
            if logical_asset_id.startswith("trigger.") and entity_id:
                self.event_semantic_asset_ids.add(safe_name(f"event_semantic.NoFly.Trigger.{entity_id}"))
            else:
                self.event_semantic_asset_ids.add(self._event_semantic_asset_id(entity))
        self.event_semantic_coordinate_audit: list[dict[str, Any]] = []
        self.event_semantic_proxy_capture_targets: list[dict[str, Any]] = []
        self.event_semantic_proxy_sanitizer_result: dict[str, Any] = {}
        self.last_airsim_proxy_capture_exclusion_result: dict[str, Any] = {}
        self.static_map_coordinate_audit = self._load_static_map_coordinate_audit()

    def _resolve_path(self, value: Any) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @staticmethod
    def _load_scene_setup_from_spec(spec_path: Path) -> dict[str, Any]:
        if not spec_path.exists():
            return {}
        try:
            module = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))
            for node in module.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "SCENE_SETUP":
                            value = ast.literal_eval(node.value)
                            return dict(value) if isinstance(value, dict) else {}
        except Exception as exc:
            print(f"[EpisodeHost] scene setup spec parse warning for {spec_path}: {exc}")
        return {}

    @staticmethod
    def _coerce_audit_vector3(value: Any) -> list[float] | None:
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

    def _coordinate_audit_entry(
        self,
        *,
        entity_id: str,
        logical_asset_id: str,
        source_coordinate_space: str,
        raw_position_enu_m: Sequence[float],
        coordinate_transform_applied: bool,
        object_role: str,
    ) -> dict[str, Any]:
        raw = [
            float(raw_position_enu_m[0] if len(raw_position_enu_m) > 0 else 0.0),
            float(raw_position_enu_m[1] if len(raw_position_enu_m) > 1 else 0.0),
            float(raw_position_enu_m[2] if len(raw_position_enu_m) > 2 else 0.0),
        ]
        resolved = self._transform_position_enu(raw) if coordinate_transform_applied else list(raw)
        expected_world_cm = [
            float(self.world_origin_cm[0]) + float(resolved[0]) * 100.0,
            float(self.world_origin_cm[1]) + float(resolved[1]) * 100.0,
            float(self.world_origin_cm[2]) + float(resolved[2]) * 100.0,
        ]
        return {
            "entity_id": entity_id,
            "logical_asset_id": logical_asset_id,
            "object_role": object_role,
            "source_coordinate_space": source_coordinate_space,
            "raw_position_enu_m": raw,
            "resolved_map_enu_m": resolved,
            "expected_ue_world_cm": expected_world_cm,
            "observed_ue_world_cm": None,
            "coordinate_transform_applied": bool(coordinate_transform_applied),
        }

    def _coordinate_space_contract(self) -> dict[str, Any]:
        return {
            "truth_frame_coordinate_space": str(self.config.get("truth_frame_coordinate_space") or "local_enu"),
            "map_enu_m": "episode truth frames, event script, and global roster coordinates; UE cm = map_enu_m * 100 + world_origin_cm",
            "traffic_bundle_lane_samples_m": "lane_center_samples.csv and road-geometry lane offsets are resolved in map ENU meters",
            "scene_sync_pose_enu_m": "scene_sync spawns/updates receive resolved map ENU meters without inverse coordinate transforms",
            "local_enu_m": "only transformed when explicitly configured as local coordinates",
            "map_static_local_enu_m": "map scenario_objects fixtures; values like [14,16,8]m intentionally resolve to [1400,1600,800]cm",
            "ue_world_cm": "observed Unreal actor/component location only, never the default input format",
        }

    def _event_semantic_entities(self) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        for entity in self.event_scene_setup.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            logical_asset_id = str(entity.get("logical_asset_id") or "")
            if (
                logical_asset_id.startswith("trigger.")
                or logical_asset_id == "facility.landing_pad.visible.v1"
                or logical_asset_id.startswith("semantic.uav_corridor.")
            ):
                entities.append(dict(entity))
        return entities

    @staticmethod
    def _event_semantic_asset_id(entity: dict[str, Any]) -> str:
        entity_id = str(entity.get("entity_id") or entity.get("instance_id") or "event_semantic")
        return safe_name(f"event_semantic.{entity_id}")

    def _event_semantic_position_enu_m(self, entity: dict[str, Any]) -> list[float] | None:
        placement = dict(entity.get("placement") or {})
        for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
            value = self._coerce_audit_vector3(placement.get(key))
            if value is not None:
                return value
        return self._coerce_audit_vector3(entity.get("position_enu_m"))

    def _trigger_box_proxy_logical_asset_id(self, placement: dict[str, Any]) -> str:
        extent = self._coerce_audit_vector3(placement.get("extent_m") or placement.get("size_m"))
        if extent is None:
            return "semantic.trigger_box.extent_14_10_14.v1"
        key = "_".join(str(int(round(float(value)))) for value in extent)
        return f"semantic.trigger_box.extent_{key}.v1"

    def _trigger_semantic_proxy_logical_asset_id(self, entity: dict[str, Any]) -> str:
        placement = dict(entity.get("placement") or {})
        if self._is_polygon_prism_trigger(entity):
            return "semantic.trigger_box.extent_14_10_14.v1"
        return self._trigger_box_proxy_logical_asset_id(placement)

    @staticmethod
    def _is_polygon_prism_trigger(entity: dict[str, Any]) -> bool:
        placement = dict(entity.get("placement") or {})
        placement_mode = str(entity.get("placement_mode") or placement.get("placement_mode") or "")
        if placement_mode.lower() != "polygon_prism":
            return False
        polygon = placement.get("polygon_enu_m")
        return isinstance(polygon, Sequence) and not isinstance(polygon, (str, bytes)) and len(polygon) >= 3

    @staticmethod
    def _is_logical_region_semantic(entity: dict[str, Any]) -> bool:
        logical_asset_id = str(entity.get("logical_asset_id") or "")
        return logical_asset_id.startswith("trigger.") or logical_asset_id.startswith("semantic.uav_corridor.")

    def _event_semantic_logical_region_policy(self) -> dict[str, Any]:
        return {
            "logical_region_primary_segmentation_enabled": bool(self.logical_region_primary_segmentation_enabled),
            "default_policy": (
                "custom_stencil_primary_seg_overlay"
                if self.logical_region_primary_segmentation_enabled
                else "sidecar_meta_only"
            ),
            "reason": (
                "No-fly, hazard, and UAV-corridor regions are logical scene semantics. "
                "By default they remain in sidecar/meta JSON so the primary segmentation PNG "
                "describes RGB-visible geometry only."
            ),
        }

    @staticmethod
    def _json_safe_copy(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False))

    def _logical_region_source_geometry(
        self,
        entity: dict[str, Any],
        *,
        position_enu_m: Sequence[float],
        rotation_deg: dict[str, Any],
        placement: dict[str, Any],
        placement_mode: str,
    ) -> dict[str, Any]:
        geometry: dict[str, Any] = {
            "logical_asset_id": str(entity.get("logical_asset_id") or ""),
            "placement_mode": str(placement_mode or entity.get("placement_mode") or ""),
            "source_position_enu_m": [float(value) for value in position_enu_m],
            "source_rotation_deg": self._json_safe_copy(rotation_deg),
            "placement": self._json_safe_copy(placement),
        }
        for key in (
            "center_enu_m",
            "resolved_position_enu_m",
            "position_enu_m",
            "extent_m",
            "size_m",
            "scale_xyz",
            "polygon_enu_m",
            "segment_start_enu_m",
            "segment_end_enu_m",
            "altitude_layer_m",
            "lateral_offset_m",
        ):
            if key in placement:
                geometry[key] = self._json_safe_copy(placement.get(key))
        extent = self._coerce_audit_vector3(placement.get("extent_m"))
        size = self._coerce_audit_vector3(placement.get("size_m") or placement.get("scale_xyz"))
        if extent is not None:
            geometry["logical_extent_m"] = extent
            geometry["logical_size_m"] = [float(extent[0]) * 2.0, float(extent[1]) * 2.0, float(extent[2]) * 2.0]
        elif size is not None:
            geometry["logical_size_m"] = size
            geometry["logical_extent_m"] = [float(size[0]) * 0.5, float(size[1]) * 0.5, float(size[2]) * 0.5]
        geometry["primary_segmentation_representation"] = (
            "custom_stencil_primary_seg_overlay"
            if self.logical_region_primary_segmentation_enabled
            else "sidecar_meta_only_not_rasterized"
        )
        return geometry

    def _load_static_map_coordinate_audit(self) -> list[dict[str, Any]]:
        scenario_objects_path = (
            PROJECT_ROOT / "Config" / "LowAltitude" / "Maps" / self.map_id / "scenario_objects.json"
        )
        if not scenario_objects_path.exists():
            return []
        try:
            payload = json.loads(scenario_objects_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[EpisodeHost] static map coordinate audit warning for {scenario_objects_path}: {exc}")
            return []
        audit: list[dict[str, Any]] = []
        for item in payload.get("objects") or []:
            if not isinstance(item, dict):
                continue
            entity_id = str(
                item.get("object_id") or item.get("entity_id") or item.get("instance_id") or item.get("asset_id") or ""
            )
            logical_asset_id = str(item.get("logical_asset_id") or item.get("asset_id") or "")
            position = None
            placement = dict(item.get("placement") or {})
            for key in ("center_enu_m", "position_enu_m", "resolved_position_enu_m"):
                position = self._coerce_audit_vector3(placement.get(key) or item.get(key))
                if position is not None:
                    break
            if not entity_id or position is None:
                continue
            audit.append(
                self._coordinate_audit_entry(
                    entity_id=entity_id,
                    logical_asset_id=logical_asset_id,
                    source_coordinate_space="map_static_local_enu_m",
                    raw_position_enu_m=position,
                    coordinate_transform_applied=False,
                    object_role="map_static_fixture",
                )
            )
        return audit

    def _spawn_event_semantic_objects(self) -> None:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        if not self.asset_catalog_reload_attempted:
            self.asset_catalog_reload_attempted = True
            try:
                self.client.reload_config("asset_catalog", map_id=self.map_id)
                print("[EpisodeHost] Reloaded asset_catalog for event semantic proxy templates.")
            except Exception as exc:
                print(f"[EpisodeHost] asset_catalog reload warning before semantic proxy spawn: {exc}")
        self.event_semantic_coordinate_audit = []
        self.event_semantic_proxy_capture_targets = []
        self.event_semantic_proxy_sanitizer_result = {}
        self.last_airsim_proxy_capture_exclusion_result = {}
        for entity in self._event_semantic_entities():
            entity_id = str(entity.get("entity_id") or entity.get("instance_id") or "")
            logical_asset_id = str(entity.get("logical_asset_id") or "")
            is_logical_region = self._is_logical_region_semantic(entity)
            position = self._event_semantic_position_enu_m(entity)
            if not entity_id or not logical_asset_id or position is None:
                continue
            target_semantic_class = ""
            target_stencil_id: int | None = None
            asset_id = self._event_semantic_asset_id(entity)
            placement = dict(entity.get("placement") or {})
            placement_mode = str(entity.get("placement_mode") or placement.get("placement_mode") or "")
            rotation = dict(entity.get("rotation_deg") or placement.get("rotation_deg") or {})
            coordinate_audit = self._coordinate_audit_entry(
                entity_id=entity_id,
                logical_asset_id=logical_asset_id,
                source_coordinate_space="map_enu_m",
                raw_position_enu_m=position,
                coordinate_transform_applied=False,
                object_role="event_semantic_proxy",
            )
            if is_logical_region:
                coordinate_audit["logical_region_source_geometry"] = self._logical_region_source_geometry(
                    entity,
                    position_enu_m=position,
                    rotation_deg=rotation,
                    placement=placement,
                    placement_mode=placement_mode,
                )
            try:
                is_polygon_prism_trigger = logical_asset_id.startswith("trigger.") and self._is_polygon_prism_trigger(entity)
                spawn_logical_asset_id = (
                    self._trigger_semantic_proxy_logical_asset_id(entity)
                    if logical_asset_id.startswith("trigger.")
                    else logical_asset_id
                )
                spawn_asset_id = (
                    safe_name(f"event_semantic.NoFly.Trigger.{entity_id}")
                    if logical_asset_id.startswith("trigger.")
                    else asset_id
                )
                proxy_position = list(position)
                proxy_ground_projection: dict[str, Any] = {}
                try:
                    proxy_ground_projection = self._project_ground_details(
                        position,
                        cache_namespace=f"event_semantic_proxy:{entity_id}",
                        use_cache=True,
                    ) or {}
                    projected = proxy_ground_projection.get("projected_enu_m")
                    if isinstance(projected, list) and len(projected) >= 3:
                        proxy_position = [float(position[0]), float(position[1]), float(projected[2]) + 0.2]
                except Exception as exc:
                    coordinate_audit["render_proxy_ground_projection_error"] = str(exc)

                payload: dict[str, Any] = {
                        "asset_id": spawn_asset_id,
                        "entity_id": entity_id,
                        "logical_asset_id": spawn_logical_asset_id,
                        "proxy_template_id": spawn_logical_asset_id,
                        "pose_enu_m": {"position_enu_m": proxy_position, "rotation_deg": rotation},
                        "position_enu_m": proxy_position,
                        "rotation_deg": rotation,
                        "placement_mode": placement_mode,
                        "placement": placement,
                        "tags": ["event_semantic", "semantic_capture", logical_asset_id, entity_id],
                        "visual_state": {
                            "mode": "visible",
                            "semantic_source": "episode_scene_setup",
                        },
                        "source_coordinate_space": "map_enu_m",
                        "coordinate_audit": coordinate_audit,
                    }
                scale_xyz = placement.get("scale_xyz") or placement.get("size_m")
                if logical_asset_id.startswith("semantic.uav_corridor."):
                    target_semantic_class = "uav_corridor"
                    target_stencil_id = self._semantic_class_id(target_semantic_class, 12)
                    proxy_scale = list(scale_xyz or [1.0, 1.0, 1.0])
                    while len(proxy_scale) < 3:
                        proxy_scale.append(1.0)
                    proxy_scale[2] = min(max(float(proxy_scale[2] or 0.05), 0.05), 0.08)
                    payload["scale_xyz"] = proxy_scale
                    payload["custom_stencil_only"] = bool(self.logical_region_primary_segmentation_enabled)
                    payload["tags"].extend(["UAVCorridor", "HighAltitudeCorridor", "uav_corridor"])
                    payload["visual_state"]["semantic_class"] = "uav_corridor"
                if logical_asset_id.startswith("trigger."):
                    target_semantic_class = "hazard_trigger"
                    target_stencil_id = self._semantic_class_id(target_semantic_class, 11)
                    payload["custom_stencil_only"] = bool(self.logical_region_primary_segmentation_enabled)
                    payload["original_logical_asset_id"] = logical_asset_id
                    payload["trigger_semantic_proxy_kind"] = "polygon_prism_aabb_overlay" if is_polygon_prism_trigger else "box_overlay"
                    payload["trigger_extent_m"] = list(placement.get("extent_m") or placement.get("size_m") or [])
                    trigger_extent = self._coerce_audit_vector3(placement.get("extent_m") or placement.get("size_m"))
                    if trigger_extent is not None:
                        payload["scale_xyz"] = [max(0.05, float(trigger_extent[0]) * 2.0), max(0.05, float(trigger_extent[1]) * 2.0), 0.08]
                    elif is_polygon_prism_trigger:
                        polygon = placement.get("polygon_enu_m")
                        if isinstance(polygon, Sequence) and not isinstance(polygon, (str, bytes)):
                            xs: list[float] = []
                            ys: list[float] = []
                            for vertex in polygon:
                                if isinstance(vertex, Sequence) and not isinstance(vertex, (str, bytes)) and len(vertex) >= 2:
                                    xs.append(float(vertex[0]))
                                    ys.append(float(vertex[1]))
                            if xs and ys:
                                payload["scale_xyz"] = [
                                    max(0.05, max(xs) - min(xs)),
                                    max(0.05, max(ys) - min(ys)),
                                    0.08,
                                ]
                                coordinate_audit["polygon_prism_semantic_proxy_bounds_enu_m"] = {
                                    "min": [min(xs), min(ys)],
                                    "max": [max(xs), max(ys)],
                                }
                        coordinate_audit["polygon_prism_runtime_trigger_logical_asset_id"] = logical_asset_id
                        coordinate_audit["polygon_prism_runtime_placement_mode"] = placement_mode
                        coordinate_audit["polygon_prism_semantic_proxy_note"] = (
                            "semantic proxy is a thin CustomStencil overlay; runtime trigger_zone keeps polygon_enu_m in placement"
                        )
                    payload["visual_state"]["semantic_class"] = "hazard_trigger"
                if is_logical_region:
                    primary_segmentation_includes_logical_region = bool(payload.get("custom_stencil_only"))
                    logical_region_policy = (
                        "custom_stencil_primary_seg_overlay"
                        if primary_segmentation_includes_logical_region
                        else "sidecar_meta_only"
                    )
                    payload["logical_region_label_policy"] = logical_region_policy
                    coordinate_audit["logical_region_label_policy"] = logical_region_policy
                    coordinate_audit["primary_segmentation_includes_logical_region"] = (
                        primary_segmentation_includes_logical_region
                    )
                coordinate_audit["render_proxy_position_enu_m"] = proxy_position
                if proxy_ground_projection:
                    coordinate_audit["render_proxy_ground_projection"] = proxy_ground_projection
                if payload.get("scale_xyz") is not None:
                    coordinate_audit["render_proxy_scale_xyz"] = list(payload.get("scale_xyz") or [])
                coordinate_audit["spawn_logical_asset_id"] = spawn_logical_asset_id
                coordinate_audit["spawn_asset_id"] = spawn_asset_id
                skip_logical_region_spawn = is_logical_region and not bool(payload.get("custom_stencil_only"))
                if skip_logical_region_spawn:
                    coordinate_audit["spawn_response"] = {
                        "status": "skipped",
                        "reason": "logical_region_sidecar_meta_only",
                        "actor_name": "",
                    }
                    coordinate_audit["sidecar_region_position_enu_m"] = proxy_position
                else:
                    response = self.client.spawn_asset(payload, map_id=self.map_id)
                    coordinate_audit["spawn_response"] = response.get("payload", response) if isinstance(response, dict) else response
                    spawn_response = coordinate_audit["spawn_response"]
                    actor_name = str(spawn_response.get("actor_name") or "") if isinstance(spawn_response, dict) else ""
                    if (
                        bool(payload.get("custom_stencil_only"))
                        and actor_name
                        and target_semantic_class
                        and target_stencil_id is not None
                    ):
                        target = {
                            "actor_name": actor_name,
                            "entity_id": entity_id,
                            "logical_asset_id": logical_asset_id,
                            "spawn_logical_asset_id": spawn_logical_asset_id,
                            "spawn_asset_id": spawn_asset_id,
                            "semantic_class": target_semantic_class,
                            "stencil_id": int(target_stencil_id),
                        }
                        self.event_semantic_proxy_capture_targets.append(target)
                        coordinate_audit["capture_proxy_sanitizer_target"] = target
            except Exception as exc:
                coordinate_audit["spawn_error"] = str(exc)
                print(f"[EpisodeHost] event semantic proxy warning for {entity_id}: {exc}")
            self.event_semantic_coordinate_audit.append(coordinate_audit)
            self.event_semantic_asset_ids.add(str(coordinate_audit.get("spawn_asset_id") or asset_id))
        if self.event_semantic_coordinate_audit:
            print(
                "[EpisodeHost] Processed event semantic regions: "
                + ", ".join(entry["entity_id"] for entry in self.event_semantic_coordinate_audit)
            )
        self._sanitize_event_semantic_proxy_components()

    def _sanitize_event_semantic_proxy_components(self) -> dict[str, Any]:
        targets = [dict(target) for target in self.event_semantic_proxy_capture_targets if target.get("actor_name")]
        if not targets:
            self.event_semantic_proxy_sanitizer_result = {
                "status": "skipped",
                "reason": "no_custom_stencil_only_event_semantic_proxies",
                "target_count": 0,
            }
            return self.event_semantic_proxy_sanitizer_result

        compact_targets: list[dict[str, Any]] = []
        for target in targets:
            actor_name = str(target.get("actor_name") or "")
            if not actor_name:
                continue
            try:
                stencil_id = int(target.get("stencil_id") or 0)
            except Exception:
                stencil_id = 0
            compact_targets.append(
                {
                    "a": actor_name,
                    "c": str(target.get("semantic_class") or ""),
                    "l": str(target.get("spawn_logical_asset_id") or ""),
                    "s": stencil_id,
                }
            )

        request = {"t": compact_targets}
        request_text = json.dumps(request, separators=(",", ":"), ensure_ascii=True)
        python_command = f"""
import json,unreal
c=json.loads({request_text!r})
ws=unreal.EditorLevelLibrary.get_pie_worlds(False)
if not ws: raise RuntimeError("No PIE world available for event semantic proxy sanitizer.")
w=ws[0]
targets={{str(t.get("a") or ""):t for t in c.get("t",[]) if str(t.get("a") or "")}}
def addtags(o,prop,vals):
    try: arr=list(o.get_editor_property(prop) or [])
    except Exception:
        try: arr=list(getattr(o,prop) or [])
        except Exception: return False
    have={{str(x) for x in arr}}; changed=False
    for v in vals:
        sv=str(v)
        if not sv or sv in have: continue
        try: arr.append(unreal.Name(sv))
        except Exception: arr.append(sv)
        have.add(sv); changed=True
    if changed:
        try: o.set_editor_property(prop,arr)
        except Exception:
            try: setattr(o,prop,arr)
            except Exception: return False
    return True
def setp(o,k,v):
    try: o.set_editor_property(k,v); return True
    except Exception: return False
rows=[]; missing=[]
for a in unreal.GameplayStatics.get_all_actors_of_class(w,unreal.Actor):
    n=a.get_name()
    t=targets.get(n)
    if not t: continue
    cls=str(t.get("c") or "")
    sid=int(t.get("s") or 0)
    tags=["event_semantic","semantic_capture",cls]
    if cls=="uav_corridor": tags+=["UAVCorridor","HighAltitudeCorridor","uav_corridor"]
    if cls=="hazard_trigger": tags+=["NoFly","Hazard","Trigger","Geofence","hazard_trigger"]
    runtime_trigger=str(t.get("l") or "").lower().startswith("trigger.")
    addtags(a,"tags",tags)
    if not runtime_trigger:
        try: a.set_actor_enable_collision(False)
        except Exception: pass
        try: a.set_actor_tick_enabled(False)
        except Exception: pass
    try: comps=list(a.get_components_by_class(unreal.PrimitiveComponent))
    except Exception: comps=[]
    for x in comps:
        addtags(x,"component_tags",tags)
        setp(x,"render_in_main_pass",False)
        setp(x,"render_in_depth_pass",False)
        setp(x,"visible_in_scene_capture_only",False)
        setp(x,"hidden_in_scene_capture",False)
        try: x.set_render_custom_depth(True)
        except Exception: setp(x,"render_custom_depth",True)
        try: x.set_custom_depth_stencil_value(sid)
        except Exception: setp(x,"custom_depth_stencil_value",sid)
        if not runtime_trigger:
            try: x.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
            except Exception: pass
        for shadow_prop in ("cast_shadow","cast_dynamic_shadow","cast_static_shadow","cast_contact_shadow","cast_hidden_shadow","affect_dynamic_indirect_lighting","affect_distance_field_lighting"):
            setp(x,shadow_prop,False)
        try: x.set_cast_shadow(False)
        except Exception: pass
        try: x.mark_render_state_dirty()
        except Exception: pass
    rows.append(n)
for n in sorted(set(targets.keys())-set(rows)): missing.append(n)
print("EVENT_SEMANTIC_PROXY_SANITIZE_COUNT",len(rows))
print("EVENT_SEMANTIC_PROXY_SANITIZE_MISSING_COUNT",len(missing))
"""
        result = self._fixed_world_capture_hook().remote.run_python(
            python_command,
            unattended=False,
            raise_on_failure=True,
        )
        output_lines: list[str] = []
        for item in result.get("output") or []:
            text = str(item.get("output") or "").rstrip()
            if text:
                output_lines.append(text)
                print(f"[EpisodeHost] {text}")
        def _parse_count_line(prefix: str) -> int | None:
            for line in output_lines:
                if line.startswith(prefix):
                    try:
                        return int(line[len(prefix) :].strip())
                    except Exception:
                        return None
            return None

        sanitized_count = _parse_count_line("EVENT_SEMANTIC_PROXY_SANITIZE_COUNT")
        missing_count = _parse_count_line("EVENT_SEMANTIC_PROXY_SANITIZE_MISSING_COUNT")
        if sanitized_count is None:
            sanitized_count = sum(1 for line in output_lines if line.startswith("EVENT_SEMANTIC_PROXY_SANITIZE "))
        if missing_count is None:
            missing_count = sum(1 for line in output_lines if line.startswith("EVENT_SEMANTIC_PROXY_SANITIZE_MISSING "))
        self.event_semantic_proxy_sanitizer_result = {
            "status": "ok" if missing_count == 0 else "partial",
            "target_count": len(targets),
            "sanitized_actor_count": sanitized_count,
            "missing_actor_count": missing_count,
            "targets": targets,
            "remote_output": output_lines,
        }
        if missing_count > 0 or sanitized_count != len(targets):
            raise RuntimeError(
                "Failed to sanitize all custom-stencil-only event semantic proxies. "
                f"details={self.event_semantic_proxy_sanitizer_result}"
            )
        return self.event_semantic_proxy_sanitizer_result

    @staticmethod
    def _airsim_capture_component_names_for_image_type(image_type: str) -> list[str]:
        normalized = str(image_type or "Scene").strip().lower()
        mapping = {
            "scene": ["SceneCaptureComponent"],
            "depthperspective": ["DepthPerspectiveCaptureComponent"],
            "depthplanar": ["DepthPlanarCaptureComponent"],
            "depthvis": ["DepthVisCaptureComponent"],
            "lighting": ["LightingCaptureComponent"],
        }
        return list(mapping.get(normalized) or ["SceneCaptureComponent"])

    def _apply_airsim_semantic_proxy_capture_exclusion(
        self,
        *,
        camera_name: str,
        modality_id: str,
        image_type: str,
    ) -> dict[str, Any]:
        targets = [dict(target) for target in self.event_semantic_proxy_capture_targets if target.get("actor_name")]
        if not targets:
            result = {
                "status": "skipped",
                "reason": "no_custom_stencil_only_event_semantic_proxies",
                "target_count": 0,
            }
            self.last_airsim_proxy_capture_exclusion_result = result
            return result
        if str(modality_id or "").strip().lower() == "seg":
            result = {
                "status": "skipped",
                "reason": "ue_custom_stencil_segmentation_must_keep_proxies_visible_to_custom_depth",
                "target_count": len(targets),
            }
            self.last_airsim_proxy_capture_exclusion_result = result
            return result

        component_names = self._airsim_capture_component_names_for_image_type(image_type)
        self.last_airsim_proxy_capture_exclusion_result = {
            "status": "ok",
            "method": "proxy_primitive_render_flags",
            "contract": "render_in_main_pass=false, render_in_depth_pass=false, and shadow casting disabled hide proxies from AirSim Scene/Depth/RGB shadows; render_custom_depth=true and stencil ids keep thin ground-overlay proxies available to UE CustomStencil segmentation",
            "target_count": len(targets),
            "proxy_actor_names": [str(target.get("actor_name") or "") for target in targets],
            "camera_name": str(camera_name or ""),
            "modality": str(modality_id or ""),
            "image_type": str(image_type or ""),
            "airsim_capture_component_names_not_mutated": component_names,
            "pipcamera_hidden_lists_mutated": False,
            "sanitizer_status": dict(self.event_semantic_proxy_sanitizer_result or {}),
        }
        return self.last_airsim_proxy_capture_exclusion_result

    def _discover_entity_modes(self) -> tuple[set[str], set[str]]:
        scene_ids: set[str] = set()
        ped_ids: set[str] = set()
        for entity in self.global_roster:
            entity_id = str(entity.get("entity_id", ""))
            resolved = self.template_resolver.resolve(entity, entity)
            mode = str(resolved.get("mode", "metadata_only"))
            if mode == "scene_sync":
                scene_ids.add(entity_id)
            elif mode == "pedestrian_managed":
                ped_ids.add(entity_id)
        return scene_ids, ped_ids

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

        # Fallen pedestrians must stay at their truth position 鈥?do not project to roadside.
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

    @staticmethod
    def _free_memory_gb() -> float | None:
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return float(status.ullAvailPhys) / (1024.0 ** 3)
        except Exception:
            return None

    def _guard_runtime_resources(self, *, context: str) -> None:
        min_free_memory_gb = float(getattr(self.args, "min_free_memory_gb", 0.0) or 0.0)
        if min_free_memory_gb > 0.0:
            free_memory_gb = self._free_memory_gb()
            if free_memory_gb is not None and free_memory_gb < min_free_memory_gb:
                raise RuntimeError(
                    f"Resource guard failed before {context}: free_memory_gb={free_memory_gb:.2f} "
                    f"< min_free_memory_gb={min_free_memory_gb:.2f}. PIE is left running for inspection."
                )

        min_output_free_disk_gb = float(getattr(self.args, "min_output_free_disk_gb", 0.0) or 0.0)
        if min_output_free_disk_gb > 0.0:
            target = self.output_dir
            existing = target if target.exists() else target.parent
            while not existing.exists() and existing != existing.parent:
                existing = existing.parent
            if existing.exists():
                usage = shutil.disk_usage(existing)
                free_disk_gb = float(usage.free) / (1024.0 ** 3)
                if free_disk_gb < min_output_free_disk_gb:
                    raise RuntimeError(
                        f"Resource guard failed before {context}: output_free_disk_gb={free_disk_gb:.2f} "
                        f"< min_output_free_disk_gb={min_output_free_disk_gb:.2f} at {existing}. "
                        "PIE is left running for inspection."
                    )

    def _airsim_capture_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("airsim_capture") or {})

    def _airsim_capture_enabled(self) -> bool:
        return self.uav_capture_backend == "airsim_native" and self._capture_role_enabled("uav")

    def _airsim_capture_pose_tolerance_m(self) -> float:
        return max(0.05, float(self._airsim_capture_cfg().get("pose_tolerance_m", 2.0)))

    def _guard_airsim_native_capture_rpc(self) -> set[str]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        try:
            self._retry("airsim_native_preflight_ping", self.client.client.ping)
            vehicles = set(self._retry("airsim_native_preflight_list_vehicles", self.client.list_vehicles))
            _ = self._retry("airsim_native_preflight_getSettingsString", self.client.get_settings_string)
            return vehicles
        except Exception as exc:
            raise RuntimeError(
                "AirSim native capture RPC guard failed before UAV capture. "
                f"host={self.args.host} port={self.args.port} vehicle={self.airsim_capture_vehicle!r}. "
                "Keep UE/PIE open for inspection; re-enter PIE or restore AirSim RPC before retrying."
            ) from exc

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
        settings_text = self._retry("getSettingsString", self.client.get_settings_string)
        settings = json.loads(settings_text)
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
        vehicles = self._guard_airsim_native_capture_rpc()
        if self.airsim_capture_vehicle not in vehicles:
            raise RuntimeError(
                f"AirSim capture vehicle '{self.airsim_capture_vehicle}' is not registered in the current PIE "
                "vehicle list. This capture vehicle must be created from Huawei Share AirSim settings.json when "
                "entering PIE so its configured bottom_center camera exists; runtime simAddVehicle is not allowed "
                "for the formal UAV capture path."
            )
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
        probe, _ = self._probe_vehicle_actor(self.airsim_capture_vehicle)
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
        target = [float(position_enu_m[0]), float(position_enu_m[1]), float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0)]
        return [
            float(self.world_origin_cm[0]) + target[0] * 100.0,
            float(self.world_origin_cm[1]) + target[1] * 100.0,
            float(self.world_origin_cm[2]) + target[2] * 100.0,
        ]

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
        # Scene-sync records are already resolved to map ENU; C++ converts them to UE cm.
        return [
            float(position_enu_m[0] if len(position_enu_m) > 0 else 0.0),
            float(position_enu_m[1] if len(position_enu_m) > 1 else 0.0),
            float(position_enu_m[2] if len(position_enu_m) > 2 else 0.0),
        ]

    def _apply_frame_rotation_deg(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        # Rotation follows the same scene-sync contract as position.
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
            if mode not in {"scene_sync", "pedestrian_managed", "metadata_only"}:
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
        item = {
            "entity_id": str(entity["entity_id"]),
            "proxy_template_id": str(resolution["logical_asset_id"]),
            "pose_enu_m": {
                "position_enu_m": [float(value) for value in position_enu_m],
                "rotation_deg": dict(rotation_deg),
            },
            "tags": list(entity.get("tags") or self.roster_by_id.get(str(entity["entity_id"]), {}).get("tags") or []),
            "visual_state": self._scene_visual_state(entity, resolution),
        }
        placement = entity.get("placement")
        placement_mode = str(entity.get("placement_mode") or "")
        if placement_mode:
            item["placement_mode"] = placement_mode
        if isinstance(placement, dict):
            item["placement"] = dict(placement)
        return item

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
        print("[EpisodeHost] Resetting previously spawned scene_sync entities and pedestrians.")
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
        semantic_remove_ids = sorted(set(extra_remove_ids) | set(self.event_semantic_asset_ids))
        if semantic_remove_ids:
            payload = {
                "tick": 0,
                "frame_id": 0,
                "sample_seq": 0,
                "sim_time_s": 0.0,
                "episode_id": self.episode_id or "forced_cleanup",
                "removes": [{"entity_id": entity_id} for entity_id in semantic_remove_ids],
            }
            try:
                self._retry("forced_reset_apply_frame", self.client.apply_frame, payload, map_id=self.map_id)
            except Exception as exc:
                print(f"[EpisodeHost] forced_reset_apply_frame warning: {exc}")
            for asset_id in semantic_remove_ids:
                self._best_effort("remove_semantic_asset", self.client.remove_asset, asset_id, map_id=self.map_id)
        for ped_id in sorted(self.all_ped_ids):
            self._best_effort("ped_release", self.client.ped_release, ped_id, map_id=self.map_id)
        for group_id in sorted(self.crowd_group_ids):
            self._best_effort("ped_clear_crowd", self.client.ped_clear_crowd, group_id, map_id=self.map_id)
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
            if (
                self._airsim_capture_enabled()
                and entity_id == self.active_airsim_capture_entity_id
                and str(entity.get("entity_category") or "").strip().lower() == "uav"
            ):
                continue
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
            pose_update_performed = False

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
                pose_update_performed = True
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
                pose_update_performed = False
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
                pose_update_performed = True

            if self.ped_last_variant.get(ped_id) != variant_id:
                variant_response = self._retry("ped_set_variant", self.client.ped_set_variant, ped_id, variant_id, map_id=self.map_id)
                results.setdefault(ped_id, {})["variant"] = variant_response.get("payload", {})
                self.ped_last_variant[ped_id] = variant_id

            should_reapply_activity = bool(effective_activity_rule.get("reapply_after_pose_update", False)) and pose_update_performed
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

    def _sync_uavs(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Host is not connected.")
        results: dict[str, Any] = {}
        for entity in frame.get("entities", []):
            if str(entity.get("entity_category") or "").strip().lower() != "uav":
                continue
            if truth_submission_state(entity) != "submit_to_ue":
                continue

            entity_id = str(entity["entity_id"])
            vehicle_name = self.airsim_capture_vehicle if (
                self._airsim_capture_enabled() and entity_id == self.active_airsim_capture_entity_id
            ) else entity_id
            if self._airsim_capture_enabled() and entity_id == self.active_airsim_capture_entity_id:
                target_enu_m, rotation_deg = self._uav_pose_for_capture(entity, {})
                wait_status = self._truth_frame_uav_status(
                    entity_id=entity_id,
                    vehicle_name=self.airsim_capture_vehicle,
                    target_enu_m=target_enu_m,
                    rotation_deg=rotation_deg,
                    operation="capture_source_truth_frame",
                )
                wait_status["capture_vehicle_name"] = self.airsim_capture_vehicle
                wait_status["source_entity_id"] = entity_id
                wait_status["replaces_scene_uav_for_capture"] = True
                wait_status["capture_pose_mode"] = "deferred_capture_tick_pin"
                wait_status["path_used"] = "truth_frame_capture_source"
                wait_status["status"]["reason"] = "capture_source_truth_frame_deferred_to_capture_tick_pin"
                wait_status["capture_gate"] = {
                    "wait_for_arrival": False,
                    "hover_before_capture": False,
                    "arrival_tolerance_m": self._airsim_capture_pose_tolerance_m(),
                    "degraded": False,
                }
                results[entity_id] = wait_status
                self.uav_active_by_entity[entity_id] = self.airsim_capture_vehicle
                self.uav_last_command_target_by_entity[entity_id] = list(target_enu_m)
                continue

            target_enu_m, rotation_deg = self._uav_pose_for_capture(entity, {})
            wait_status = self._truth_frame_uav_status(
                entity_id=entity_id,
                vehicle_name=vehicle_name,
                target_enu_m=target_enu_m,
                rotation_deg=rotation_deg,
                operation="scene_sync_truth_frame",
            )
            results[entity_id] = wait_status
            self.uav_active_by_entity[entity_id] = vehicle_name
            self.uav_last_command_target_by_entity[entity_id] = list(target_enu_m)

        current_entities = set(results)
        for entity_id in sorted(set(self.uav_active_by_entity) - current_entities):
            self.uav_active_by_entity.pop(entity_id, None)
            self.uav_last_command_target_by_entity.pop(entity_id, None)

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

    def _uav_camera_presets(self) -> list[dict[str, Any]]:
        uav = dict(self.capture_presets.get("uav_cameras") or {})
        return [dict(item) for item in (uav.get("default") or [])]

    def _frame_active_uavs(self, frame: dict[str, Any], *, site_id: str = "") -> list[str]:
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
            ids.append(entity_id)
        return sorted(set(ids))

    def _select_airsim_capture_entity(self, batch: BatchPlan, capture_ticks: set[int]) -> None:
        self.active_airsim_capture_entity_id = ""
        self.active_capture_view_id = ""
        if not self._airsim_capture_enabled():
            return
        explicit = self.requested_airsim_capture_entity
        if not explicit:
            raise RuntimeError(
                "AirSim native UAV capture requires exactly one explicit --airsim-capture-entity. "
                "High-overview/fixed-world captures should run without camera-role uav."
            )
        if not self.requested_capture_view_id:
            raise RuntimeError(
                "AirSim native UAV capture requires explicit stable --capture-view-id. "
                "No deterministic fallback capture view id is allowed."
            )
        candidate_ticks = sorted(capture_ticks) if capture_ticks else [batch.tick_start]
        for tick in candidate_ticks:
            frame = self.frames_by_tick.get(int(tick))
            if not frame:
                continue
            active_ids = self._frame_active_uavs(frame, site_id=batch.site_id)
            if explicit in active_ids:
                self.active_airsim_capture_entity_id = explicit
                break
        if explicit and not self.active_airsim_capture_entity_id:
            raise RuntimeError(
                f"Requested --airsim-capture-entity '{explicit}' is not an active UAV in batch {batch.batch_id}."
            )
        if not self.active_airsim_capture_entity_id:
            raise RuntimeError(f"AirSim native UAV capture requested but no active UAV exists in batch {batch.batch_id}.")
        self.active_capture_view_id = self.requested_capture_view_id
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

    def _vehicle_actor_probe_hook(self) -> FixedWorldCaptureEditorHook:
        return self._fixed_world_capture_hook()

    def _truth_frame_uav_status(
        self,
        *,
        entity_id: str,
        vehicle_name: str,
        target_enu_m: Sequence[float],
        rotation_deg: dict[str, Any] | None = None,
        operation: str,
    ) -> dict[str, Any]:
        target = [float(value) for value in target_enu_m]
        rotation = dict(rotation_deg or {})
        return {
            "vehicle_name": vehicle_name,
            "source_entity_id": entity_id,
            "status": {
                "state": "ok",
                "reason": f"{operation}_uav_truth_frame_scene_sync",
                "position_enu_m": target,
                "target_enu_m": target,
                "distance_m": 0.0,
                "tolerance_m": self._uav_position_tolerance_m(),
            },
            "pose": {
                "position_enu_m": target,
                "rotation_deg": rotation,
            },
            "move": {
                "ok": True,
                "via": "truth_frame_scene_sync",
                "operation": operation,
                "target_enu_m": target,
                "rotation_deg": rotation,
            },
            "path_used": "truth_frame_scene_sync",
            "truth_frame_scene_sync": True,
            "timed_out": False,
            "position_error_m": 0.0,
            "capture_gate": {
                "wait_for_arrival": False,
                "hover_before_capture": False,
                "arrival_tolerance_m": self._uav_position_tolerance_m(),
                "degraded": False,
            },
        }

    def _uav_timeout_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("timeouts") or {})

    def _uav_position_tolerance_m(self) -> float:
        timeout_cfg = self._uav_timeout_cfg()
        return max(0.1, float(timeout_cfg.get("uav_position_tolerance_m", 2.0)))

    def _uav_should_debug(self, entity_id: str) -> bool:
        if not self.uav_debug_entity_ids:
            return True
        return str(entity_id).strip() in self.uav_debug_entity_ids

    def _uav_command_debug(
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

    def _probe_vehicle_actors(
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
            payload = self._vehicle_actor_probe_hook().inspect_capture_vehicle_actors(
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

    def _probe_vehicle_actor(self, vehicle_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        probe_payload = self._probe_vehicle_actors({"__probe__": {"vehicle_name": vehicle_name}})
        return (
            dict((probe_payload.get("vehicles") or {}).get(vehicle_name) or {}),
            dict(probe_payload.get("probe_meta") or {}),
        )

    def _collect_uav_debug(
        self,
        frame: dict[str, Any],
        uav_status: dict[str, Any],
    ) -> dict[str, Any]:
        debug_by_entity: dict[str, dict[str, Any]] = {}

        for entity in frame.get("entities", []):
            resolution = self._entity_resolution(entity)
            if str(entity.get("entity_category") or "").strip().lower() != "uav":
                continue
            if truth_submission_state(entity) != "submit_to_ue":
                continue

            entity_id = str(entity.get("entity_id") or "")
            if not self._uav_should_debug(entity_id):
                continue

            status_entry = dict(uav_status.get(entity_id) or {})
            if not status_entry:
                continue
            vehicle_name = str(status_entry.get("vehicle_name") or entity_id)
            transformed_position_enu_m, transformed_rotation_deg = self._transformed_entity_pose(entity)
            resolved_position_enu_m, resolved_rotation_deg, snap_details = self._resolve_entity_pose(entity)
            ground_details = self._project_ground_details(
                resolved_position_enu_m,
                cache_namespace=f"uav:{vehicle_name}",
                use_cache=True,
            ) if bool(self.ground_reference_cfg.get("uav_ground_relative", False)) else None
            command_target_enu_m = list((ground_details or {}).get("ground_relative_enu_m") or resolved_position_enu_m)
            command_velocity_mps = float(((entity.get("annotations") or {}).get("speed_mps")) or resolution.get("velocity_mps") or 5.0)

            status_payload = dict(status_entry.get("status") or {})
            pose_payload = dict(status_entry.get("pose") or {})
            pose_enu_m = pose_payload.get("position_enu_m")

            debug_entry = self._uav_command_debug(
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
                    "path_used": str(status_entry.get("path_used") or "truth_frame_scene_sync"),
                    "status_payload": status_payload,
                    "pose_payload": pose_payload,
                    "pose_enu_m": list(pose_enu_m) if isinstance(pose_enu_m, list) else pose_enu_m,
                    "position_error_m": status_entry.get("position_error_m"),
                    "timed_out": bool(status_entry.get("timed_out", False)),
                    "pose_warning": str(status_entry.get("pose_warning") or ""),
                }
            )
            debug_by_entity[entity_id] = debug_entry

        return {
            "vehicles": debug_by_entity,
            "probe_meta": {"probe_disabled": True, "reason": "uav_truth_frame_scene_sync"},
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
        preserve_names.update({"captureuav_0", "skyatmosphere", "skyatmosphere_0", "skylight", "skylight_0"})
        class_prefixes = [str(value).strip() for value in (cfg.get("hide_actor_class_prefixes") or []) if str(value).strip()]
        name_keywords = [str(value).strip().lower() for value in (cfg.get("hide_actor_name_keywords") or []) if str(value).strip()]
        skip_prefixes = [str(value).strip() for value in (cfg.get("skip_actor_class_prefixes") or []) if str(value).strip()]
        sanitize_engine_sky_dome = bool(cfg.get("sanitize_engine_sky_dome", False))
        disable_sky_atmosphere_editor_notifications = bool(cfg.get("disable_sky_atmosphere_editor_notifications", False))
        sky_dome_actor_keywords = [
            str(value).strip().lower()
            for value in (cfg.get("sky_dome_actor_keywords") or [])
            if str(value).strip()
        ]
        sky_dome_component_path_keywords = [
            str(value).strip().lower()
            for value in (cfg.get("sky_dome_component_path_keywords") or [])
            if str(value).strip()
        ]
        if (
            not destroy_class_names
            and not destroy_class_prefixes
            and not destroy_name_prefixes
            and not class_prefixes
            and not name_keywords
            and not sanitize_engine_sky_dome
            and not disable_sky_atmosphere_editor_notifications
        ):
            return

        request = {
            "destroy_class_names": destroy_class_names,
            "destroy_class_prefixes": destroy_class_prefixes,
            "destroy_name_prefixes": destroy_name_prefixes,
            "preserve_names": sorted(preserve_names),
            "class_prefixes": class_prefixes,
            "name_keywords": name_keywords,
            "skip_prefixes": skip_prefixes,
            "sanitize_engine_sky_dome": sanitize_engine_sky_dome,
            "disable_sky_atmosphere_editor_notifications": disable_sky_atmosphere_editor_notifications,
            "sky_dome_actor_keywords": sky_dome_actor_keywords,
            "sky_dome_component_path_keywords": sky_dome_component_path_keywords,
        }
        request_text = json.dumps(request, separators=(",", ":"), ensure_ascii=True)
        python_command = f"""
import json,unreal
c=json.loads({request_text!r})
ws=unreal.EditorLevelLibrary.get_pie_worlds(False)
if not ws: raise RuntimeError("No PIE world available for scene cleanup.")
w=ws[0]
if c.get("disable_sky_atmosphere_editor_notifications",False):
    try:
        unreal.SystemLibrary.execute_console_command(w,"r.SkyAtmosphere.EditorNotifications 0")
        print("PIE_SKY_ATMOSPHERE_EDITOR_NOTIFICATIONS_DISABLED")
    except Exception as e: print("PIE_SKY_ATMOSPHERE_EDITOR_NOTIFICATIONS_DISABLE_FAILED",str(e))
D=[];H=[];S=[];P=set(c.get("preserve_names",[]));AK=list(c.get("sky_dome_actor_keywords",[]));PK=list(c.get("sky_dome_component_path_keywords",[]))
def ident(a,cl):
    n=a.get_name();l=""
    try: l=a.get_actor_label()
    except Exception: pass
    return n,l,set([n.lower(),l.lower()]),(n+" "+l+" "+cl).lower()
def op(o):
    if o is None: return ""
    for q in ("get_path_name","get_name"):
        try:
            f=getattr(o,q,None);v=f() if callable(f) else ""
            if v: return str(v)
        except Exception: pass
    return str(o)
def paths(x):
    r=[]
    try:
        m=x.get_static_mesh()
        if m: r.append(op(m))
    except Exception: pass
    try: n=int(x.get_num_materials() or 0)
    except Exception: n=0
    for i in range(n):
        try:
            m=x.get_material(i)
            if m: r.append(op(m))
        except Exception: pass
    try: mats=x.get_editor_property("override_materials")
    except Exception: mats=[]
    for m in mats or []:
        if m: r.append(op(m))
    return " ".join(r).replace("\\\\","/").lower()
def setp(x,k,v):
    try: x.set_editor_property(k,v); return True
    except Exception: return False
def sc(x):
    ok=False
    try: x.set_visibility(False,True); ok=True
    except Exception: ok=setp(x,"visible",False) or ok
    try: x.set_hidden_in_game(True,True); ok=True
    except Exception: ok=setp(x,"hidden_in_game",True) or ok
    for k,v in (("render_in_main_pass",False),("render_custom_depth",False),("render_custom_depth_pass",False),("hidden_in_scene_capture",True)): ok=setp(x,k,v) or ok
    try: x.set_render_custom_depth(False); ok=True
    except Exception: pass
    try: x.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION); ok=True
    except Exception: pass
    return ok
for a in unreal.GameplayStatics.get_all_actors_of_class(w,unreal.Actor):
    cl=a.get_class().get_name()
    if any(cl.startswith(p) for p in c.get("skip_prefixes",[])): continue
    n,l,ids,hay=ident(a,cl)
    if ids & P: continue
    dm=cl in set(c.get("destroy_class_names",[])) or any(cl.startswith(p) for p in c.get("destroy_class_prefixes",[])) or any(n.lower().startswith(p) or l.lower().startswith(p) for p in c.get("destroy_name_prefixes",[]))
    if dm:
        try: a.destroy_actor(); D.append(n+"|"+l+"|"+cl)
        except Exception: pass
        continue
    if c.get("sanitize_engine_sky_dome",False):
        try: comps=list(a.get_components_by_class(unreal.PrimitiveComponent))
        except Exception: comps=[]
        tl=[n.lower(),l.lower(),cl.lower()]
        tb=[x[:-2] if x.endswith("_c") else x for x in tl]
        am=any(k in tl or k in tb for k in AK); ms=[x for x in comps if any(k in paths(x) for k in PK)]
        if am or ms:
            if am:
                for fn,args in ((a.set_actor_hidden_in_game,(True,)),(a.set_actor_enable_collision,(False,)),(a.set_actor_tick_enabled,(False,))):
                    try: fn(*args)
                    except Exception: pass
            cn=[]
            for x in (comps if am else ms):
                if sc(x):
                    try: cn.append(x.get_name())
                    except Exception: cn.append(str(x))
            S.append(n+"|"+l+"|"+cl+"|"+("actor_keyword" if am else "component_path")+"|"+",".join(cn))
            continue
    mt=any(cl.startswith(p) for p in c.get("class_prefixes",[])) or any(k in hay for k in c.get("name_keywords",[]))
    if not mt: continue
    for fn,args in ((a.set_actor_hidden_in_game,(True,)),(a.set_actor_enable_collision,(False,)),(a.set_actor_tick_enabled,(False,))):
        try: fn(*args)
        except Exception: pass
    H.append(n+"|"+l+"|"+cl)
for x in D[:100]: print("PIE_SCENE_DESTROY",x)
print("PIE_SCENE_DESTROY_COUNT",len(D))
for x in H[:100]: print("PIE_SCENE_CLEANUP",x)
print("PIE_SCENE_CLEANUP_COUNT",len(H))
for x in S[:100]: print("PIE_SKY_DOME_SANITIZE",x)
print("PIE_SKY_DOME_SANITIZE_COUNT",len(S))
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
        if bool(getattr(self.args, "preserve_capture_output_dir", False)):
            ensure_dir(resolved_output)
            self.prepared_capture_output_dirs.add(resolved_output)
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

    def _uav_capture_view_output_dir(self, view_id: str) -> Path:
        safe_view_id = safe_name(view_id)
        simple_view_id = simple_capture_view_dir_name(view_id)
        if safe_name(self.output_dir.name) in {safe_view_id, simple_view_id}:
            return self.output_dir
        return self.output_dir / safe_view_id

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
        capture_view_output_dir = modality_output_dir.parent
        view_dir_names = {safe_name(view_id), simple_capture_view_dir_name(view_id)}
        batch_output_dir = capture_view_output_dir.parent if capture_view_output_dir.name in view_dir_names else self.output_dir
        if camera_role != "uav":
            batch_output_dir = self.output_dir / batch.batch_id
        return {
            "storage_layout_version": "capture_storage_v1",
            "storage_rule": "Primary image paths use short stable names; episode/view/tick details live in this sidecar.",
            "episode_output_root": str(self.output_dir),
            "batch_id": batch.batch_id,
            "batch_output_dir": str(batch_output_dir),
            "capture_view_id": view_id,
            "capture_view_output_dir": str(capture_view_output_dir),
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
    def _load_semantic_class_by_id(rules_path: Path) -> dict[str, str]:
        try:
            root = json.loads(Path(rules_path).read_text(encoding="utf-8-sig"))
        except Exception:
            return {}
        classes = dict(root.get("classes") or {})
        if not classes:
            return {}
        result: dict[str, str] = {}
        for class_name, class_id in classes.items():
            try:
                result[str(int(class_id))] = str(class_name)
            except Exception:
                continue
        return result

    def _semantic_class_id(self, class_name: str, default: int) -> int:
        normalized = str(class_name or "").strip().lower()
        for raw_class_id, configured_name in self.semantic_class_by_id.items():
            if str(configured_name or "").strip().lower() != normalized:
                continue
            try:
                return int(raw_class_id)
            except (TypeError, ValueError):
                break
        return int(default)

    def _write_capture_storage_manifest(self) -> Path:
        path = self.output_dir / "capture_storage_manifest.json"
        ensure_dir(path.parent)
        payload = {
            "$schema": "aeroworld_capture_storage_manifest_v1",
            "episode_id": self.episode_id,
            "output_root": str(self.output_dir),
            "storage_layout_version": "capture_storage_v1",
            "storage_rule": "Primary image paths stay short; complex episode/view identifiers live in sidecars and this manifest.",
            "simple_path_contract": {
                "formal_default_output_root": "F:/aw_cap",
                "uav_route": "F:/aw_cap/uav/eNN/vNN/<modality>/tick_NNNNNN.<ext>",
                "high_overview_route": "F:/aw_cap/hi/eNN/tick_NNNNNN.<ext>",
                "meta_route": "F:/aw_cap/_meta",
                "complex_identifiers_live_in_sidecars": True,
            },
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
                "uav_scene_control_backend": self.uav_scene_control_backend,
                "non_capture_uav_control": "truth_frame_scene_sync",
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
                    "route": "UE CustomDepth/CustomStencil fixed-world capture for RGB-visible UAV segmentation; AirSim native segmentation is not used by default",
                },
            },
            "coordinate_space_contract": self._coordinate_space_contract(),
            "event_semantic_objects": {
                "source": "episode_scene_setup",
                "logical_region_visual_proxies_spawned": bool(self.logical_region_primary_segmentation_enabled),
                "visible_event_fixture_proxies_spawned": True,
                "logical_region_label_policy": self._event_semantic_logical_region_policy(),
                "trigger_proxies_custom_stencil_only": bool(self.logical_region_primary_segmentation_enabled),
                "rgb_visibility_contract": (
                    "Logical no-fly, hazard, and UAV-corridor regions default to sidecar/meta-only. "
                    "They are not injected into the primary segmentation PNG unless "
                    "event_semantic_overlays.logical_region_primary_segmentation_enabled=true."
                ),
                "coordinate_audit": self.event_semantic_coordinate_audit,
            },
            "static_map_objects": {
                "source_coordinate_space": "map_static_local_enu_m",
                "coordinate_audit": self.static_map_coordinate_audit,
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

    def _semantic_stencil_pixel_counts(self, image_path: Path) -> dict[str, Any]:
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            return {"class_histogram_error": f"Pillow/numpy import failed: {exc}"}

        try:
            image = Image.open(image_path)
            image_mode = str(image.mode)
            values = np.asarray(image.convert("L"), dtype=np.uint8)
        except Exception as exc:
            return {"class_histogram_error": f"semantic stencil PNG load failed: {exc}"}

        unique, counts = np.unique(values, return_counts=True)
        histogram = {str(int(class_id)): int(count) for class_id, count in zip(unique, counts)}
        allowed_class_ids: set[int] = set()
        for raw_class_id in self.semantic_class_by_id.keys():
            try:
                allowed_class_ids.add(int(raw_class_id))
            except (TypeError, ValueError):
                continue
        unknown_class_ids = [int(value) for value in unique.tolist() if int(value) not in allowed_class_ids]
        invalid_pixel_count = int(sum(histogram.get(str(class_id), 0) for class_id in unknown_class_ids))
        return {
            "semantic_png_mode": image_mode,
            "class_histogram": histogram,
            "ignore_pixel_count": int(histogram.get("0", 0)),
            "non_ignore_pixel_count": int(values.size - int(histogram.get("0", 0))),
            "semantic_unique_class_ids": [int(value) for value in unique.tolist()],
            "unknown_semantic_class_ids": unknown_class_ids,
            "invalid_semantic_class_id_pixel_count": invalid_pixel_count,
            "unknown_semantic_color_pixel_count": invalid_pixel_count,
        }

    @staticmethod
    def _write_semantic_palette_preview(image_path: Path) -> dict[str, Any]:
        palette_path = image_path.with_name(f"{image_path.stem}__palette.png")
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            return {"semantic_palette_preview_error": f"Pillow/numpy import failed: {exc}"}

        try:
            values = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
            palette = np.zeros((256, 3), dtype=np.uint8)
            palette[0] = [0, 0, 0]
            palette[1] = [140, 140, 140]
            palette[2] = [40, 120, 255]
            palette[3] = [30, 180, 80]
            palette[4] = [0, 200, 220]
            palette[5] = [230, 60, 50]
            palette[6] = [220, 60, 220]
            palette[7] = [255, 220, 40]
            palette[8] = [255, 140, 30]
            palette[9] = [150, 80, 255]
            palette[10] = [245, 245, 245]
            palette[11] = [255, 120, 170]
            palette[12] = [80, 255, 210]
            Image.fromarray(palette[values], mode="RGB").save(palette_path)
        except Exception as exc:
            return {"semantic_palette_preview_error": f"semantic palette preview write failed: {exc}"}

        return {
            "semantic_palette_preview_path": str(palette_path),
            "palette_preview_kind": "rgb_visualization_only_not_training_label",
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
        sidecar.update(self._write_semantic_palette_preview(image_path))
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
    ) -> None:
        modalities = dict(self.capture_presets.get("modalities") or {})
        modality = dict(modalities.get(modality_id) or {})
        normalized_modality = str(modality_id or "rgb").strip().lower()
        if normalized_modality == "seg":
            raise RuntimeError(
                "AirSim native segmentation output is forbidden by the semantic capture contract. "
                "UAV seg must use UE CustomStencil."
            )
        output_dir = self._uav_capture_view_output_dir(view_id) / safe_name(normalized_modality)
        self._prepare_capture_output_dir(output_dir)
        frame_stem = self.capture_orchestrator.frame_stem(frame)
        extension = str(modality.get("extension") or ("npy" if normalized_modality == "depth" else "png"))
        image_path = output_dir / f"{frame_stem}.{extension}"
        depth_preview_path: Path | None = None
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
            if response_compress:
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
                "image_type": modality.get("image_type", "Scene"),
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
        pose = entity.get("truth_pose")
        if not isinstance(pose, dict):
            raise RuntimeError(
                f"UAV capture entity '{entity.get('entity_id', '')}' must provide truth_pose.position_enu_m[3]."
            )
        position = pose.get("position_enu_m") or pose.get("position_m")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) < 3:
            raise RuntimeError(
                f"UAV capture entity '{entity.get('entity_id', '')}' must provide truth_pose.position_enu_m[3]."
            )
        truth_position, truth_rotation = self._transformed_entity_pose(entity)
        return (
            [float(truth_position[0]), float(truth_position[1]), float(truth_position[2])],
            self._add_rotation_offsets(dict(truth_rotation), None),
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
        uav_position_enu_m, uav_rotation_deg = self._uav_pose_for_capture(entity, vehicle_status)
        source_uav_position_enu_m = list(uav_position_enu_m)
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
        normalized_modality = str(modality_id or "rgb").strip().lower()
        modalities = dict(self.capture_presets.get("modalities") or {})
        modality = dict(modalities.get(normalized_modality) or {})
        if normalized_modality not in {"rgb", "depth", "seg"}:
            raise RuntimeError(f"AirSim native UAV capture does not support modality '{modality_id}'.")
        uses_ue_custom_stencil = normalized_modality == "seg"
        if uses_ue_custom_stencil:
            capture_pose = {
                "requested_position_enu_m": [float(value) for value in uav_position_enu_m],
                "requested_rotation_deg": dict(uav_rotation_deg),
                "pose": {},
                "pose_error_m": 0.0,
                "capture_pose_mode": "ue_custom_stencil_truth_pose_no_airsim_pin",
            }
            camera_info_before_capture = {}
        else:
            self._ensure_airsim_capture_vehicle()
            capture_pose = self._pin_airsim_capture_vehicle(
                uav_position_enu_m,
                uav_rotation_deg,
                context=f"pre-capture {entity_id} tick {int(frame['tick'])}",
            )
            settle_s = float((self.config.get("timeouts") or {}).get("camera_settle_s", 0.25))
            if settle_s > 0.0:
                time.sleep(settle_s)
        set_capture_camera_pose = bool(preset.get("set_capture_camera_pose", False))
        if set_capture_camera_pose and not uses_ue_custom_stencil:
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
        if not uses_ue_custom_stencil:
            camera_info_before_capture = self._retry(
                "simGetCameraInfo",
                self.client.get_camera_info,
                self.airsim_capture_vehicle,
                camera_name,
            )
            observed_fov = float(camera_info_before_capture.get("fov_degrees", 0.0))
            if not math.isfinite(observed_fov) or abs(observed_fov - fov_degrees) > 0.5:
                raise RuntimeError(
                    "AirSim capture camera FOV does not match the formal capture preset. "
                    f"vehicle={self.airsim_capture_vehicle!r} camera={camera_name!r} "
                    f"expected_fov_degrees={fov_degrees} observed_fov_degrees={observed_fov}. "
                    "Fix Huawei Share AirSim settings.json and re-enter PIE; runtime simSetCameraFov is not part of "
                    "the UAV capture contract."
                )
            image_type = str(modality.get("image_type") or "Scene")
            airsim_proxy_capture_exclusion = self._apply_airsim_semantic_proxy_capture_exclusion(
                camera_name=camera_name,
                modality_id=normalized_modality,
                image_type=image_type,
            )
            if (
                int(airsim_proxy_capture_exclusion.get("target_count") or 0) > 0
                and str(airsim_proxy_capture_exclusion.get("status") or "").lower() != "ok"
            ):
                raise RuntimeError(
                    "Unable to exclude custom-stencil-only semantic proxies from AirSim capture. "
                    f"details={airsim_proxy_capture_exclusion}"
                )
        else:
            airsim_proxy_capture_exclusion = {
                "status": "skipped",
                "reason": (
                    "logical_region_overlays_are_sidecar_meta_only"
                    if not self.logical_region_primary_segmentation_enabled
                    else "ue_custom_stencil_segmentation_must_keep_proxies_visible_to_custom_depth"
                ),
                "target_count": len(self.event_semantic_proxy_capture_targets),
            }
        camera_suffix = safe_name(str(preset.get("camera_id_suffix", camera_name)))
        if not self.requested_capture_view_id:
            raise RuntimeError(
                "AirSim native UAV capture requires explicit stable --capture-view-id. "
                "No deterministic fallback capture view id is allowed."
            )
        view_id = self.requested_capture_view_id
        if uses_ue_custom_stencil:
            hook = self._fixed_world_capture_hook()
            output_dir = self._uav_capture_view_output_dir(view_id) / "seg"
            self._prepare_capture_output_dir(output_dir)
            frame_stem = self.capture_orchestrator.frame_stem(frame)
            image_path = output_dir / f"{frame_stem}.png"
            semantic_audit_path = output_dir / f"{frame_stem}__seg_audit.json"
            camera_world_rotation_deg = self._add_rotation_offsets(uav_rotation_deg, camera_rotation_body_deg)
            semantic_camera_asset_id = safe_name(f"fixed_world_camera.semantic.{view_id}")
            self.ground_camera_asset_ids[("uav_semantic_stencil", view_id)] = semantic_camera_asset_id
            hook.ensure_fixed_world_camera(
                map_id=self.map_id,
                asset_id=semantic_camera_asset_id,
                logical_asset_id="camera.fixed_world_capture.rgb.v1",
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
                "logical_event_id": self.episode_id,
                "logical_sample_id": f"{self.episode_id}:tick{int(frame['tick']):06d}:{view_id}",
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
                "uav_scene_status": vehicle_status,
                "uav_debug": uav_debug,
                "uav_entity_id": entity_id,
                "source_uav_entity_id": entity_id,
                "capture_vehicle_name": self.airsim_capture_vehicle,
                "vehicle_name": self.airsim_capture_vehicle,
                "camera_name": camera_name,
                "requested_camera_pose_body_m": camera_offset_body_m,
                "requested_camera_rotation_body_deg": camera_rotation_body_deg,
                "set_capture_camera_pose": set_capture_camera_pose,
                "camera_pose_frame": str(preset.get("camera_pose_frame") or preset.get("camera_pose_coordinate_frame") or "ned").strip().lower(),
                "camera_info_before_capture": camera_info_before_capture,
                "source_uav_pose_enu_m": source_uav_position_enu_m,
                "expected_uav_pose_enu_m": uav_position_enu_m,
                "expected_uav_rotation_deg": uav_rotation_deg,
                "capture_source_coordinate_audit": self._coordinate_audit_entry(
                    entity_id=entity_id,
                    logical_asset_id=str(self._entity_resolution(entity).get("logical_asset_id") or ""),
                    source_coordinate_space="map_enu_m" if self._truth_frame_uses_map_enu() else "local_enu_m",
                    raw_position_enu_m=position_enu_from_truth(entity),
                    coordinate_transform_applied=not self._truth_frame_uses_map_enu(),
                    object_role="capture_source_uav",
                ),
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
                "uav_scene_control_backend": self.uav_scene_control_backend,
                "coordinate_space_contract": self._coordinate_space_contract(),
                "event_semantic_logical_region_policy": self._event_semantic_logical_region_policy(),
                "event_semantic_objects": self.event_semantic_coordinate_audit,
                "event_semantic_proxy_sanitizer": self.event_semantic_proxy_sanitizer_result,
                "airsim_proxy_capture_exclusion": airsim_proxy_capture_exclusion,
                "static_map_objects": self.static_map_coordinate_audit,
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
        if normalized_modality == "seg":
            raise RuntimeError("UAV seg must use UE CustomStencil and cannot reach AirSim native capture.")
        response = self._retry(
            "simGetImages",
            self.client.capture_vehicle_image,
            self.airsim_capture_vehicle,
            camera_name=camera_name,
            image_type=str(modality.get("image_type") or "Scene"),
            pixels_as_float=bool(modality.get("pixels_as_float", normalized_modality == "depth")),
            compress=bool(modality.get("compress", normalized_modality != "depth")),
            annotation_name=str(modality.get("annotation_name") or ""),
        )
        sidecar = {
            "episode_id": self.episode_id,
            "logical_event_id": self.episode_id,
            "logical_sample_id": f"{self.episode_id}:tick{int(frame['tick']):06d}:{view_id}",
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
            "uav_scene_status": vehicle_status,
            "uav_debug": uav_debug,
            "uav_entity_id": entity_id,
            "source_uav_entity_id": entity_id,
            "capture_vehicle_name": self.airsim_capture_vehicle,
            "vehicle_name": self.airsim_capture_vehicle,
            "camera_name": camera_name,
            "requested_camera_pose_body_m": camera_offset_body_m,
            "requested_camera_rotation_body_deg": camera_rotation_body_deg,
            "set_capture_camera_pose": set_capture_camera_pose,
            "camera_pose_frame": str(preset.get("camera_pose_frame") or preset.get("camera_pose_coordinate_frame") or "ned").strip().lower(),
            "camera_info_before_capture": camera_info_before_capture,
            "source_uav_pose_enu_m": source_uav_position_enu_m,
            "expected_uav_pose_enu_m": uav_position_enu_m,
            "expected_uav_rotation_deg": uav_rotation_deg,
            "capture_source_coordinate_audit": self._coordinate_audit_entry(
                entity_id=entity_id,
                logical_asset_id=str(self._entity_resolution(entity).get("logical_asset_id") or ""),
                source_coordinate_space="map_enu_m" if self._truth_frame_uses_map_enu() else "local_enu_m",
                raw_position_enu_m=position_enu_from_truth(entity),
                coordinate_transform_applied=not self._truth_frame_uses_map_enu(),
                object_role="capture_source_uav",
            ),
            "requested_capture_pose_enu_m": capture_pose.get("requested_position_enu_m"),
            "requested_capture_rotation_deg": capture_pose.get("requested_rotation_deg"),
            "airsim_pose_before_capture": capture_pose.get("pose"),
            "pose_error_m": capture_pose.get("pose_error_m"),
            "capture_pose_mode": capture_pose.get("capture_pose_mode"),
            "capture_alignment_key": f"{self.episode_id}:{batch.batch_id}:{int(frame['tick'])}:{view_id}",
            "capture_alignment_source": "deterministic_episode_frame",
            "capture_backend": "airsim_native_uav_camera",
            "uav_scene_control_backend": self.uav_scene_control_backend,
            "coordinate_space_contract": self._coordinate_space_contract(),
            "event_semantic_logical_region_policy": self._event_semantic_logical_region_policy(),
            "event_semantic_objects": self.event_semantic_coordinate_audit,
            "event_semantic_proxy_sanitizer": self.event_semantic_proxy_sanitizer_result,
            "airsim_proxy_capture_exclusion": airsim_proxy_capture_exclusion,
            "static_map_objects": self.static_map_coordinate_audit,
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
                "uav_scene_status": uav_status,
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
                and truth_submission_state(entity) == "submit_to_ue"
            ]
            expected_uavs = [entity_id for entity_id in expected_uavs if entity_id]
            if expected_uavs:
                raise RuntimeError(
                    "UAV capture requested but no truth-frame UAV status was produced for: "
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
            matched_cameras = {(job[0], job[1]) for job in jobs}
            if len(matched_cameras) != 1:
                raise RuntimeError(
                    "AirSim native UAV capture requires exactly one camera per run. "
                    f"Matched jobs: {[{'camera_id': job[0], 'camera_name': job[1], 'modality': job[2]} for job in jobs]}"
                )
            for camera_id, camera_name, modality_id, preset in jobs:
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
            ticks = sorted(set(explicit_capture_ticks))
            max_capture_frames = int(getattr(self.args, "max_capture_frames", 0) or 0)
            return set(ticks[:max_capture_frames] if max_capture_frames > 0 else ticks)
        ticks = [tick for tick in self.sorted_ticks if batch.tick_start <= tick <= batch.tick_end]
        stride = max(1, int(getattr(self.args, "tick_stride", 1) or 1))
        capture_ticks = ticks[::stride]
        max_capture_frames = int(getattr(self.args, "max_capture_frames", 0) or 0)
        if max_capture_frames > 0:
            capture_ticks = capture_ticks[:max_capture_frames]
        return set(capture_ticks)

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

    def _script_target_asset_instance_id(
        self,
        action: dict[str, Any],
        entity_id: str,
        logical_asset_id: str = "",
    ) -> str:
        explicit = str(
            action.get("asset_instance_id")
            or action.get("instance_id")
            or action.get("asset_id_override")
            or action.get("target_asset_id")
            or ""
        ).strip()
        if explicit:
            return explicit
        logical_asset_id = logical_asset_id or self._script_entity_logical_asset_id(entity_id, action)
        if logical_asset_id.startswith("trigger."):
            return safe_name(f"event_semantic.NoFly.Trigger.{entity_id}")
        if (
            logical_asset_id == "facility.landing_pad.visible.v1"
            or logical_asset_id.startswith("semantic.uav_corridor.")
        ):
            return safe_name(f"event_semantic.{entity_id}")
        return entity_id

    @staticmethod
    def _script_scale_xyz(value: Any, field_name: str) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{field_name} must be a numeric vector")
        values = list(value)
        if len(values) < 2:
            raise ValueError(f"{field_name} requires at least x/y components")
        while len(values) < 3:
            values.append(1.0)
        return [float(values[0]), float(values[1]), float(values[2])]

    def _script_transform_animation_keyframes(self, raw_keyframes: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_keyframes, Sequence) or isinstance(raw_keyframes, (str, bytes)):
            raise ValueError("animate_entity_transform requires a keyframes list")
        keyframes: list[dict[str, Any]] = []
        for index, raw_keyframe in enumerate(raw_keyframes):
            if not isinstance(raw_keyframe, dict):
                raise ValueError(f"keyframes[{index}] must be an object")
            keyframe = dict(raw_keyframe)
            if "position_enu_m" in keyframe:
                keyframe["position_enu_m"] = self._script_transform_position(
                    keyframe["position_enu_m"],
                    f"keyframes[{index}].position_enu_m",
                )
            if "rotation_deg" in keyframe:
                keyframe["rotation_deg"] = self._script_transform_rotation({"rotation_deg": keyframe["rotation_deg"]})
            if "scale_xyz" in keyframe:
                keyframe["scale_xyz"] = self._script_scale_xyz(keyframe["scale_xyz"], f"keyframes[{index}].scale_xyz")
            pose = keyframe.get("pose_enu_m")
            if isinstance(pose, dict):
                pose_payload = dict(pose)
                if "position_enu_m" in pose_payload:
                    pose_payload["position_enu_m"] = self._script_transform_position(
                        pose_payload["position_enu_m"],
                        f"keyframes[{index}].pose_enu_m.position_enu_m",
                    )
                if "rotation_deg" in pose_payload:
                    pose_payload["rotation_deg"] = self._script_transform_rotation(
                        {"rotation_deg": pose_payload["rotation_deg"]}
                    )
                keyframe["pose_enu_m"] = pose_payload
            keyframes.append(keyframe)
        if not keyframes:
            raise ValueError("animate_entity_transform requires at least one keyframe")
        return keyframes

    def _register_event_action_handlers(self) -> None:
        interp = self.event_interpreter
        interp.register_handler("set_weather", self._script_set_weather)
        interp.register_handler("spawn_entity", self._script_spawn_entity)
        interp.register_handler("move_entity", self._script_move_entity)
        interp.register_handler("remove_entity", self._script_remove_entity)
        interp.register_handler("animate_entity_transform", self._script_animate_entity_transform)
        interp.register_handler("set_facility_state", self._script_set_facility_state)
        interp.register_handler("set_logical_boundary_state", self._script_set_logical_boundary_state)
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
        if asset_kind == "uav" and self.uav_scene_control_backend == "truth_frame_scene_sync":
            if self._airsim_capture_enabled() and entity_id == self.active_airsim_capture_entity_id:
                self.uav_active_by_entity[entity_id] = self.airsim_capture_vehicle
            self.uav_last_command_target_by_entity[entity_id] = list(position)
            return {
                "status": "ok",
                "entity_id": entity_id,
                "asset_id": asset_instance_id,
                "response": {
                    "payload": {
                        "state": "skipped",
                        "reason": "uav_spawn_deferred_to_truth_frame_scene_sync",
                        "target_enu_m": [float(value) for value in position],
                    }
                },
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
        if asset_kind == "uav":
            self.uav_active_by_entity[entity_id] = asset_instance_id
            self.uav_last_command_target_by_entity[entity_id] = list(position)
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
                "command": "truth_frame_position_update",
                "target_enu_m": [float(value) for value in position],
                "reason": "pedestrian movement is applied deterministically from truth_frames before capture",
            }
        if asset_kind == "uav":
            if self._airsim_capture_enabled() and entity_id == self.active_airsim_capture_entity_id:
                command_rotation = rotation or self._entity_rotation_deg({"entity_id": entity_id})
                wait_status = self._truth_frame_uav_status(
                    entity_id=entity_id,
                    vehicle_name=self.airsim_capture_vehicle,
                    target_enu_m=position,
                    rotation_deg=command_rotation,
                    operation="capture_source_event_move",
                )
                wait_status["capture_vehicle_name"] = self.airsim_capture_vehicle
                wait_status["capture_pose_mode"] = "deferred_capture_tick_pin"
                wait_status["replaces_scene_uav_for_capture"] = True
                response = {
                    "payload": {
                        "vehicle_name": self.airsim_capture_vehicle,
                        "source_entity_id": entity_id,
                        "state": "ok",
                        "reason": "capture_source_event_move_deferred_to_capture_tick_pin",
                        "target_enu_m": [float(value) for value in position],
                        "rotation_deg": dict(command_rotation or {}),
                        "velocity_mps": float(action.get("velocity_mps", 5.0)),
                        "synthetic": True,
                    }
                }
                self.uav_active_by_entity[entity_id] = self.airsim_capture_vehicle
                self.uav_last_command_target_by_entity[entity_id] = list(position)
                wait_status["path_used"] = "truth_frame_capture_source"
                wait_status["source_entity_id"] = entity_id
                return {
                    "status": "ok",
                    "entity_id": entity_id,
                    "vehicle_name": self.airsim_capture_vehicle,
                    "command_position_enu_m": position,
                    "response": response,
                    "wait_status": wait_status,
                }
            if self.uav_scene_control_backend == "truth_frame_scene_sync":
                command_rotation = rotation or self._entity_rotation_deg({"entity_id": entity_id})
                wait_status = self._truth_frame_uav_status(
                    entity_id=entity_id,
                    vehicle_name=entity_id,
                    target_enu_m=position,
                    rotation_deg=command_rotation,
                    operation="event_move_deferred_to_truth_frame",
                )
                wait_status["status"]["reason"] = "event_uav_move_deferred_to_truth_frame_scene_sync"
                wait_status["move"]["reason"] = "non_capture_uav_pose_is_authoritative_in_truth_frames"
                self.uav_last_command_target_by_entity[entity_id] = list(position)
                return {
                    "status": "ok",
                    "entity_id": entity_id,
                    "asset_id": asset_instance_id,
                    "command_position_enu_m": position,
                    "response": {
                        "payload": {
                            "state": "skipped",
                            "reason": "uav_move_deferred_to_truth_frame_scene_sync",
                        }
                    },
                    "wait_status": wait_status,
                }
            response = self.client.move_asset(payload, map_id=self.map_id)
            self.uav_active_by_entity[entity_id] = asset_instance_id
            self.uav_last_command_target_by_entity[entity_id] = list(position)
            return {"status": "ok", "entity_id": entity_id, "asset_id": asset_instance_id, "response": response}
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
            if self._airsim_capture_enabled() and entity_id == self.active_airsim_capture_entity_id:
                self.uav_active_by_entity.pop(entity_id, None)
                self.event_controlled_entity_ids.discard(entity_id)
                self.uav_last_command_target_by_entity.pop(entity_id, None)
                return {
                    "status": "ok",
                    "entity_id": entity_id,
                    "capture_vehicle_name": self.airsim_capture_vehicle,
                    "response": {
                        "payload": {
                            "state": "skipped",
                            "reason": "capture_vehicle_is_retained_and_reused",
                        }
                    },
                }
            asset_id = self.uav_active_by_entity.pop(entity_id, entity_id)
            if self.uav_scene_control_backend == "truth_frame_scene_sync":
                self.event_controlled_entity_ids.discard(entity_id)
                self.uav_last_command_target_by_entity.pop(entity_id, None)
                return {
                    "status": "ok",
                    "entity_id": entity_id,
                    "asset_id": asset_id,
                    "response": {
                        "payload": {
                            "state": "skipped",
                            "reason": "uav_remove_deferred_to_truth_frame_scene_sync",
                        }
                    },
                }
            response = self.client.remove_asset(asset_id, map_id=self.map_id)
            self.event_controlled_entity_ids.discard(entity_id)
            self.uav_last_command_target_by_entity.pop(entity_id, None)
            return {"status": "ok", "entity_id": entity_id, "asset_id": asset_id, "response": response}
        response = self.client.remove_asset(entity_id, map_id=self.map_id)
        self.event_controlled_entity_ids.discard(entity_id)
        return {"status": "ok", "entity_id": entity_id, "response": response}

    def _script_animate_entity_transform(self, action: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(action.get("entity_id") or "").strip()
        if not entity_id:
            raise ValueError("animate_entity_transform requires entity_id")
        logical_asset_id = self._script_entity_logical_asset_id(entity_id, action)
        animation = dict(action.get("transform_animation") or {})
        raw_keyframes = action.get("keyframes", animation.get("keyframes"))
        keyframes = self._script_transform_animation_keyframes(raw_keyframes)
        animation["keyframes"] = keyframes
        for key in ("duration_ticks", "duration_s", "easing", "loop", "relative"):
            if key in action:
                animation[key] = action[key]
        visual_state = dict(action.get("visual_state") or {})
        for key in ("state", "mode", "material_variant", "pulse_rate_hz", "custom_stencil_class"):
            if key in action:
                visual_state[key] = action[key]
        visual_state.setdefault("mode", "transform_animation")
        payload: dict[str, Any] = {
            "asset_id": self._script_target_asset_instance_id(action, entity_id, logical_asset_id),
            "entity_id": entity_id,
            "visual_state": visual_state,
            "transform_animation": animation,
            "keyframes": keyframes,
        }
        if "tags" in action:
            payload["tags"] = list(action.get("tags") or [])
        final_keyframe = keyframes[-1]
        if "scale_xyz" in final_keyframe:
            payload["scale_xyz"] = list(final_keyframe["scale_xyz"])
        response = self.client.move_asset(payload, map_id=self.map_id)
        return {
            "status": "ok",
            "entity_id": entity_id,
            "asset_id": payload["asset_id"],
            "keyframe_count": len(keyframes),
            "response": response,
        }

    def _script_set_facility_state(self, action: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(action.get("entity_id") or "").strip()
        if not entity_id:
            raise ValueError("set_facility_state requires entity_id")
        logical_asset_id = self._script_entity_logical_asset_id(entity_id, action)
        visual_state = dict(action.get("visual_state") or {})
        for key in (
            "state",
            "lights_on",
            "beacon_phase",
            "door_state",
            "material_variant",
            "mode",
            "pulse_rate_hz",
        ):
            if key in action:
                visual_state[key] = action[key]
        facility_state = str(action.get("state") or visual_state.get("state") or "").strip()
        if facility_state:
            visual_state["facility_state"] = facility_state
        payload: dict[str, Any] = {
            "asset_id": self._script_target_asset_instance_id(action, entity_id, logical_asset_id),
            "entity_id": entity_id,
            "visual_state": visual_state,
            "facility_state": facility_state,
        }
        for key in ("lights_on", "beacon_phase", "door_state", "material_variant"):
            if key in visual_state:
                payload[key] = visual_state[key]
        response = self.client.move_asset(payload, map_id=self.map_id)
        return {
            "status": "ok",
            "entity_id": entity_id,
            "asset_id": payload["asset_id"],
            "facility_state": facility_state,
            "response": response,
        }

    def _script_set_logical_boundary_state(self, action: dict[str, Any]) -> dict[str, Any]:
        entity_id = str(action.get("entity_id") or "").strip()
        if not entity_id:
            raise ValueError("set_logical_boundary_state requires entity_id")
        logical_asset_id = self._script_entity_logical_asset_id(entity_id, action)
        visual_state = dict(action.get("visual_state") or {})
        for key in (
            "state",
            "pulse_rate_hz",
            "custom_stencil_class",
            "material_variant",
            "mode",
            "lights_on",
        ):
            if key in action:
                visual_state[key] = action[key]
        boundary_state = str(action.get("state") or visual_state.get("state") or "").strip()
        if boundary_state:
            visual_state["logical_boundary_state"] = boundary_state
        payload: dict[str, Any] = {
            "asset_id": self._script_target_asset_instance_id(action, entity_id, logical_asset_id),
            "entity_id": entity_id,
            "visual_state": visual_state,
            "logical_boundary_state": boundary_state,
        }
        for key in ("pulse_rate_hz", "custom_stencil_class", "material_variant", "lights_on"):
            if key in visual_state:
                payload[key] = visual_state[key]
        response = self.client.move_asset(payload, map_id=self.map_id)
        return {
            "status": "ok",
            "entity_id": entity_id,
            "asset_id": payload["asset_id"],
            "logical_boundary_state": boundary_state,
            "response": response,
        }

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
        activity_type = str(
            action.get("activity_type")
            or action.get("activity")
            or action.get("state")
            or action.get("mode")
            or ""
        ).strip().lower()
        if not ped_id:
            raise ValueError("set_pedestrian_activity requires entity_id")
        if not activity_type:
            raise ValueError("set_pedestrian_activity requires activity_type")
        self.event_pedestrian_activity_state[ped_id] = activity_type
        if self.event_interpreter is not None:
            self.event_interpreter.update_entity_activity(ped_id, activity_type)
        response: dict[str, Any] = {"status": "skipped", "reason": "pedestrian_not_active"}
        if ped_id in self.ped_active_ids:
            response = self._ped_activity_action(ped_id, activity_type)
            self.ped_last_activity[ped_id] = activity_type
        return {
            "status": "ok",
            "ped_id": ped_id,
            "activity_type": activity_type,
            "command": "pedestrian_activity_state",
            "response": response,
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
            payload = {
                "asset_id": self._script_target_asset_instance_id(action, entity_id, logical_asset_id),
                "entity_id": entity_id,
                "visual_state": visual_state,
            }
            if self.uav_scene_control_backend == "truth_frame_scene_sync":
                return {
                    "status": "ok",
                    "entity_id": entity_id,
                    "asset_id": payload["asset_id"],
                    "visual_state": visual_state,
                    "response": {
                        "payload": {
                            "state": "skipped",
                            "reason": "uav_visual_state_deferred_to_truth_frame_scene_sync",
                        }
                    },
                }
            response = self.client.move_asset(payload, map_id=self.map_id)
            return {"status": "ok", "entity_id": entity_id, "asset_id": payload["asset_id"], "visual_state": visual_state, "response": response}
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
            uav_debug = self._collect_uav_debug(frame, uav_status)

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
                self._guard_runtime_resources(context=f"capture tick {tick}")
                feedback_payload = self._poll_feedback()
                self._capture_ground_views(batch, frame, weather_payload, entity_records, feedback_payload, uav_status, uav_debug)
                self._capture_uav_views(batch, frame, weather_payload, entity_records, feedback_payload, uav_status, uav_debug)

    def run(self) -> None:
        if self.client is None:
            self.connect()
        ensure_dir(self.output_dir)
        self._guard_runtime_resources(context="capture startup")
        manifest_path = self._write_capture_storage_manifest()
        print(f"[EpisodeHost] Capture storage manifest written: {manifest_path}")
        if bool(getattr(self.args, "semantic_stencil_audit_only", False)):
            audit_path = self.output_dir / "semantic_seg_audit.json"
            self._fixed_world_capture_hook().semantic_stencil_audit(
                map_id=self.map_id,
                semantic_rules_path=self.semantic_rules_path,
                semantic_audit_path=audit_path,
                assign=False,
            )
            print(f"[EpisodeHost] Semantic stencil audit written: {audit_path}")
            return
        self._cleanup_pie_world_actors()
        self.hard_reset_world_state()
        self._spawn_event_semantic_objects()
        manifest_path = self._write_capture_storage_manifest()
        print(f"[EpisodeHost] Capture storage manifest updated after semantic proxy spawn: {manifest_path}")
        try:
            for index, batch in enumerate(self.batch_plans):
                self.run_batch(batch)
                if index + 1 < len(self.batch_plans):
                    self.hard_reset_world_state()
                    self._spawn_event_semantic_objects()

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
    parser.add_argument("--max_capture_frames", type=int, default=0, help="Hard cap on captured frames per batch after tick filters")
    parser.add_argument("--min_free_memory_gb", type=float, default=3.0, help="Abort before capture when available system RAM is below this threshold")
    parser.add_argument("--min_output_free_disk_gb", type=float, default=10.0, help="Abort before capture when output drive free space is below this threshold")
    parser.add_argument(
        "--preserve_capture_output_dir",
        action="store_true",
        help="Do not clear modality output directories before writing; callers must clean deterministic targets first.",
    )
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
        "--uav-scene-control-backend",
        choices=["truth_frame_scene_sync"],
        default="",
        help=(
            "Formal non-capture UAV scene-control backend. AirSim controls only the rotating capture vehicle."
        ),
    )
    parser.add_argument(
        "--segmentation-backend",
        choices=["ue_custom_stencil"],
        default="ue_custom_stencil",
        help="Backend for UAV seg captures. Formal captures only support UE CustomDepth/CustomStencil.",
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
        help=(
            "Scene UAV entity id to replace with the AirSim capture platform. "
            "Required when camera-role includes uav; high-overview/fixed-world captures do not use it."
        ),
    )
    parser.add_argument(
        "--capture-view-id",
        default="",
        help="Stable deterministic view id used as the UAV capture output subdirectory.",
    )
    parser.add_argument(
        "--write-depth-preview",
        action="store_true",
        default=True,
        help="Write an extra 8-bit PNG preview next to depth .npy for persistent review and presentation artifacts.",
    )
    parser.add_argument(
        "--no-write-depth-preview",
        dest="write_depth_preview",
        action="store_false",
        help="Disable depth preview PNG output and keep only the depth .npy plus sidecar.",
    )
    parser.add_argument(
        "--semantic-stencil-audit-only",
        action="store_true",
        help="Write a read-only UE CustomStencil semantic component audit JSON and exit without capturing images.",
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
