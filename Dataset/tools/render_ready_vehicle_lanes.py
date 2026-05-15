from __future__ import annotations

import bisect
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


VEHICLE_LANE_CENTER_TOLERANCE_M = 0.35
VEHICLE_CENTERLINE_TOLERANCE_M = 0.60
VEHICLE_LANE_GRID_CELL_M = 25.0


@dataclass(frozen=True)
class VehicleLaneSample:
    edge_id: str
    lane_id: str
    lane_index: int
    s_m: float
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass(frozen=True)
class VehicleRoadLaneMetadata:
    lanes: int
    width_m: float


def normalize_vector3(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = list(default)
    return [
        float(values[0] if len(values) > 0 else default[0]),
        float(values[1] if len(values) > 1 else default[1]),
        float(values[2] if len(values) > 2 else default[2]),
    ]


def vehicle_lane_normal(sample: VehicleLaneSample) -> tuple[float, float]:
    yaw_rad = math.radians(sample.yaw_deg)
    return -math.sin(yaw_rad), math.cos(yaw_rad)


def vehicle_lateral_from_lane(sample: VehicleLaneSample, position_enu_m: Sequence[float]) -> float:
    nx, ny = vehicle_lane_normal(sample)
    return (float(position_enu_m[0]) - sample.x_m) * nx + (float(position_enu_m[1]) - sample.y_m) * ny


def parse_lane_index_from_lane_id(lane_id: str | None) -> int | None:
    if not lane_id:
        return None
    suffix = str(lane_id).rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def edge_id_from_lane_id(lane_id: str | None) -> str:
    if not lane_id:
        return ""
    text = str(lane_id)
    suffix = text.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return text[: -(len(suffix) + 1)]
    return text


class VehicleLaneProjector:
    """Applies lane-derived vehicle XY correction to final truth, not UE render offsets."""

    def __init__(self, project_root: Path, map_id: str) -> None:
        lane_samples_csv = (
            project_root
            / "Config"
            / "LowAltitude"
            / "Maps"
            / map_id
            / "traffic_bundle"
            / "lane_center_samples.csv"
        )
        road_geojson = project_root / "Content" / "Maps" / map_id / "road" / "road.geojson"
        self.road_metadata = self._load_road_metadata(road_geojson)
        self.samples: list[VehicleLaneSample] = []
        self.by_edge: dict[str, list[VehicleLaneSample]] = defaultdict(list)
        self._grid: dict[tuple[int, int], list[VehicleLaneSample]] = defaultdict(list)
        with lane_samples_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"edge_id", "lane_id", "lane_index", "s_m", "x_m", "y_m", "yaw_deg"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{lane_samples_csv} missing required columns: {sorted(missing)}")
            for row in reader:
                sample = VehicleLaneSample(
                    edge_id=str(row["edge_id"]),
                    lane_id=str(row["lane_id"]),
                    lane_index=int(row["lane_index"]),
                    s_m=float(row["s_m"]),
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    yaw_deg=float(row.get("yaw_deg") or 0.0),
                )
                self.samples.append(sample)
                self.by_edge[sample.edge_id].append(sample)
                self._grid[self._grid_key(sample.x_m, sample.y_m)].append(sample)
        for edge_samples in self.by_edge.values():
            edge_samples.sort(key=lambda item: item.s_m)
        if not self.samples:
            raise ValueError(f"{lane_samples_csv} contains no lane samples")

    @staticmethod
    def _load_road_metadata(path: Path) -> dict[str, VehicleRoadLaneMetadata]:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        result: dict[str, VehicleRoadLaneMetadata] = {}
        for feature in payload.get("features") or []:
            properties = dict(feature.get("properties") or {})
            road_id = properties.get("id")
            if road_id is None:
                continue
            try:
                lanes = max(1, int(float(properties.get("lanes") or 1)))
                width_m = max(0.0, float(properties.get("width") or 0.0))
            except (TypeError, ValueError):
                continue
            result[f"cg_edge_{road_id}"] = VehicleRoadLaneMetadata(lanes=lanes, width_m=width_m)
        return result

    @staticmethod
    def _grid_key(x_m: float, y_m: float) -> tuple[int, int]:
        return (
            int(math.floor(float(x_m) / VEHICLE_LANE_GRID_CELL_M)),
            int(math.floor(float(y_m) / VEHICLE_LANE_GRID_CELL_M)),
        )

    def _nearest_sample(self, position_enu_m: Sequence[float]) -> VehicleLaneSample:
        x_m = float(position_enu_m[0])
        y_m = float(position_enu_m[1])
        cx, cy = self._grid_key(x_m, y_m)
        best: VehicleLaneSample | None = None
        best_d2 = float("inf")
        for radius in range(0, 5):
            for gx in range(cx - radius, cx + radius + 1):
                for gy in range(cy - radius, cy + radius + 1):
                    for sample in self._grid.get((gx, gy), []):
                        d2 = (sample.x_m - x_m) ** 2 + (sample.y_m - y_m) ** 2
                        if d2 < best_d2:
                            best = sample
                            best_d2 = d2
            if best is not None and best_d2 <= ((radius + 0.5) * VEHICLE_LANE_GRID_CELL_M) ** 2:
                break
        if best is not None:
            return best
        return min(self.samples, key=lambda item: (item.x_m - x_m) ** 2 + (item.y_m - y_m) ** 2)

    def _sample_for_vehicle(
        self,
        position_enu_m: Sequence[float],
        *,
        edge_id: str | None = None,
        lane_position_m: Any = None,
        lane_id: str | None = None,
    ) -> VehicleLaneSample:
        resolved_edge_id = str(edge_id or "").strip() or edge_id_from_lane_id(lane_id)
        edge_samples = self.by_edge.get(resolved_edge_id)
        if edge_samples and lane_position_m not in (None, ""):
            try:
                s_m = float(lane_position_m)
            except (TypeError, ValueError):
                s_m = float("nan")
            if math.isfinite(s_m):
                values = [sample.s_m for sample in edge_samples]
                index = bisect.bisect_left(values, s_m)
                candidates = []
                if index < len(edge_samples):
                    candidates.append(edge_samples[index])
                if index > 0:
                    candidates.append(edge_samples[index - 1])
                if candidates:
                    return min(candidates, key=lambda item: abs(item.s_m - s_m))
        return self._nearest_sample(position_enu_m)

    def _physical_offsets(self, edge_id: str) -> list[float]:
        metadata = self.road_metadata.get(edge_id)
        if metadata is None or metadata.lanes <= 1:
            return [0.0]
        lane_spacing_m = metadata.width_m / float(metadata.lanes) if metadata.width_m > 0.0 else 3.0
        return [
            ((float(index) + 0.5) - 0.5 * float(metadata.lanes)) * lane_spacing_m
            for index in range(metadata.lanes)
        ]

    def _target_offset(self, sample: VehicleLaneSample, offsets: list[float], yaw_deg: float, lane_id: str | None) -> float:
        lane_index = parse_lane_index_from_lane_id(lane_id)
        align = math.cos(math.radians(float(yaw_deg) - sample.yaw_deg))
        if align >= 0.0:
            side_offsets = [value for value in offsets if value < -1e-6]
        else:
            side_offsets = [value for value in offsets if value > 1e-6]
        if not side_offsets:
            side_offsets = [value for value in offsets if abs(value) > 1e-6]
        if not side_offsets:
            return 0.0
        side_offsets = sorted(side_offsets, key=lambda value: abs(value))
        if lane_index is not None and 0 <= lane_index < len(side_offsets):
            return side_offsets[lane_index]
        return side_offsets[0]

    def project_vehicle_position(
        self,
        position_enu_m: Sequence[float],
        *,
        yaw_deg: float = 0.0,
        edge_id: str | None = None,
        lane_position_m: Any = None,
        lane_id: str | None = None,
    ) -> list[float]:
        position = normalize_vector3(position_enu_m)
        sample = self._sample_for_vehicle(position, edge_id=edge_id, lane_position_m=lane_position_m, lane_id=lane_id)
        offsets = self._physical_offsets(sample.edge_id)
        if not offsets or all(abs(value) <= 1e-6 for value in offsets):
            return position
        lateral = vehicle_lateral_from_lane(sample, position)
        if min(abs(lateral - value) for value in offsets) <= VEHICLE_LANE_CENTER_TOLERANCE_M:
            return position
        if abs(lateral) > VEHICLE_CENTERLINE_TOLERANCE_M:
            return position
        target = self._target_offset(sample, offsets, yaw_deg, lane_id)
        nx, ny = vehicle_lane_normal(sample)
        return [sample.x_m + target * nx, sample.y_m + target * ny, position[2]]
