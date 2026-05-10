"""Deterministic UAV high-altitude corridor planning helpers.

The scenario generator uses these helpers as a hard gate before writing
event scripts.  Building GeoJSON coordinates are transformed through the
same map fit used by pedestrian/ground validation; no caller should interpret
source lon/lat as ENU directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from shapely.geometry import LineString, Point

import map_spatial_index as msi
from map_spatial_index import MapSpatialIndex, offset_from_lane


UAV_CORRIDOR_LOGICAL_ASSET_ID = "semantic.uav_corridor.segment.v1"
UAV_CORRIDOR_CLASS_NAME = "uav_corridor"
UAV_ALTITUDE_LAYERS_M = (30.0, 50.0, 80.0)


def _round3(value: float) -> float:
    return round(float(value), 3)


def _point3(value: Sequence[float]) -> list[float]:
    return [_round3(value[0]), _round3(value[1]), _round3(value[2] if len(value) > 2 else 0.0)]


def _dist_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


@dataclass(frozen=True)
class BuildingObstacle:
    building_id: str
    height_m: float
    geometry: Any


@dataclass(frozen=True)
class CorridorSlot:
    slot_id: str
    edge_id: str
    start_enu_m: list[float]
    end_enu_m: list[float]
    mid_enu_m: list[float]
    yaw_deg: float
    altitude_m: float
    lateral_offset_m: float
    length_m: float
    width_m: float
    height_m: float

    def point_at(self, t: float) -> list[float]:
        clamped = max(0.0, min(1.0, float(t)))
        return [
            _round3(float(self.start_enu_m[0]) + (float(self.end_enu_m[0]) - float(self.start_enu_m[0])) * clamped),
            _round3(float(self.start_enu_m[1]) + (float(self.end_enu_m[1]) - float(self.start_enu_m[1])) * clamped),
            _round3(self.altitude_m),
        ]

    def nearest_point(self, point: Sequence[float]) -> list[float]:
        ax, ay = float(self.start_enu_m[0]), float(self.start_enu_m[1])
        bx, by = float(self.end_enu_m[0]), float(self.end_enu_m[1])
        px, py = float(point[0]), float(point[1])
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-6:
            return list(self.start_enu_m)
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        return self.point_at(t)


class BuildingObstacleIndex:
    def __init__(
        self,
        spatial: MapSpatialIndex,
        *,
        horizontal_clearance_m: float = 6.0,
        vertical_clearance_m: float = 5.0,
        fallback_height_m: float = 18.0,
        floor_height_m: float = 3.2,
    ) -> None:
        self.spatial = spatial
        self.horizontal_clearance_m = float(horizontal_clearance_m)
        self.vertical_clearance_m = float(vertical_clearance_m)
        self.fallback_height_m = float(fallback_height_m)
        self.floor_height_m = float(floor_height_m)
        self.obstacles = self._load_buildings()

    def _building_height_m(self, feature: dict[str, Any]) -> float:
        props = feature.get("properties") or {}
        values: list[float] = []
        for key in ("height", "altitude"):
            raw = props.get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                values.append(value)
        try:
            floor_height = float(props.get("floor")) * self.floor_height_m
            if floor_height > 0.0:
                values.append(floor_height)
        except (TypeError, ValueError):
            pass
        geom = feature.get("geometry") or {}
        coords = str(geom.get("coordinates") or "")
        # Some source polygons carry a z coordinate per vertex.  Treat it as a
        # weak height hint when no explicit height is present.
        if values:
            return max(values)
        if coords:
            return self.fallback_height_m
        return self.fallback_height_m

    def _load_buildings(self) -> list[BuildingObstacle]:
        path = self.spatial.source_geojson_paths["building"]
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        result: list[BuildingObstacle] = []
        for index, feature in enumerate(data.get("features") or []):
            geometry = msi._transform_geojson_geometry(  # noqa: SLF001 - shared internal map fit helper.
                dict(feature.get("geometry") or {}),
                self.spatial.geojson_center_xy_m,
                self.spatial.geojson_fit,
            )
            if geometry is None or geometry.is_empty:
                continue
            props = feature.get("properties") or {}
            building_id = str(props.get("id_origin") or props.get("id") or index)
            result.append(
                BuildingObstacle(
                    building_id=building_id,
                    height_m=self._building_height_m(feature),
                    geometry=geometry,
                )
            )
        if not result:
            raise RuntimeError(f"No building obstacles could be loaded from {path}")
        return result

    def point_collision(self, point_enu_m: Sequence[float]) -> BuildingObstacle | None:
        point = Point(float(point_enu_m[0]), float(point_enu_m[1]))
        z_m = float(point_enu_m[2] if len(point_enu_m) > 2 else 0.0)
        for obstacle in self.obstacles:
            if z_m <= obstacle.height_m + self.vertical_clearance_m and obstacle.geometry.buffer(self.horizontal_clearance_m).covers(point):
                return obstacle
        return None

    def segment_collision(self, a_enu_m: Sequence[float], b_enu_m: Sequence[float]) -> BuildingObstacle | None:
        line = LineString([(float(a_enu_m[0]), float(a_enu_m[1])), (float(b_enu_m[0]), float(b_enu_m[1]))])
        z_min = min(float(a_enu_m[2] if len(a_enu_m) > 2 else 0.0), float(b_enu_m[2] if len(b_enu_m) > 2 else 0.0))
        for obstacle in self.obstacles:
            if z_min <= obstacle.height_m + self.vertical_clearance_m and line.intersects(obstacle.geometry.buffer(self.horizontal_clearance_m)):
                return obstacle
        return None

    def point_clear(self, point_enu_m: Sequence[float]) -> bool:
        return self.point_collision(point_enu_m) is None

    def segment_clear(self, a_enu_m: Sequence[float], b_enu_m: Sequence[float]) -> bool:
        return self.segment_collision(a_enu_m, b_enu_m) is None

    def air_point_clear(self, point_enu_m: Sequence[float]) -> bool:
        point = Point(float(point_enu_m[0]), float(point_enu_m[1]))
        if not self.spatial.bounds_union.covers(point):
            return False
        return self.point_clear(point_enu_m)

    def air_segment_clear(self, a_enu_m: Sequence[float], b_enu_m: Sequence[float]) -> bool:
        line = LineString([(float(a_enu_m[0]), float(a_enu_m[1])), (float(b_enu_m[0]), float(b_enu_m[1]))])
        if not self.spatial.bounds_union.covers(line):
            return False
        return self.air_point_clear(a_enu_m) and self.air_point_clear(b_enu_m) and self.segment_clear(a_enu_m, b_enu_m)

    def route_clear(self, route_enu_m: Sequence[Sequence[float]]) -> bool:
        points = [_point3(item) for item in route_enu_m]
        if any(not self.point_clear(point) for point in points):
            return False
        return all(self.segment_clear(a, b) for a, b in zip(points, points[1:]))

    def _point_at_altitude(self, point_enu_m: Sequence[float], altitude_m: float) -> list[float]:
        return [_round3(point_enu_m[0]), _round3(point_enu_m[1]), _round3(altitude_m)]

    def repair_point(self, point_enu_m: Sequence[float], altitude_m: float) -> list[float]:
        base = self._point_at_altitude(point_enu_m, altitude_m)
        if self.air_point_clear(base):
            return base
        for radius_m in (6.0, 10.0, 16.0, 24.0, 36.0, 54.0, 80.0, 120.0, 180.0):
            for index in range(16):
                angle = (math.tau * index) / 16.0
                candidate = [
                    _round3(float(point_enu_m[0]) + math.cos(angle) * radius_m),
                    _round3(float(point_enu_m[1]) + math.sin(angle) * radius_m),
                    _round3(altitude_m),
                ]
                if self.air_point_clear(candidate):
                    return candidate
        raise RuntimeError(f"Unable to repair UAV waypoint near {list(point_enu_m)} at {altitude_m}m")

    def repair_segment(self, a_enu_m: Sequence[float], b_enu_m: Sequence[float]) -> list[list[float]]:
        a = _point3(a_enu_m)
        b = _point3(b_enu_m)
        if self.air_segment_clear(a, b):
            return [b]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return [b] if self.air_point_clear(b) else []
        nx = -dy / length
        ny = dx / length
        for offset_m in (12.0, 20.0, 32.0, 48.0, 72.0, 108.0, 160.0, 240.0):
            for sign in (1.0, -1.0):
                mid = [
                    _round3((a[0] + b[0]) * 0.5 + nx * offset_m * sign),
                    _round3((a[1] + b[1]) * 0.5 + ny * offset_m * sign),
                    _round3((a[2] + b[2]) * 0.5),
                ]
                if self.air_point_clear(mid) and self.air_segment_clear(a, mid) and self.air_segment_clear(mid, b):
                    return [mid, b]
                a_side = [_round3(a[0] + nx * offset_m * sign), _round3(a[1] + ny * offset_m * sign), a[2]]
                b_side = [_round3(b[0] + nx * offset_m * sign), _round3(b[1] + ny * offset_m * sign), b[2]]
                if (
                    self.air_point_clear(a_side)
                    and self.air_point_clear(b_side)
                    and self.air_segment_clear(a, a_side)
                    and self.air_segment_clear(a_side, b_side)
                    and self.air_segment_clear(b_side, b)
                ):
                    return [a_side, b_side, b]
        return []

    def route_clear_at_altitude(self, route_enu_m: Sequence[Sequence[float]], altitude_m: float) -> bool:
        raw_points = [self._point_at_altitude(point, altitude_m) for point in route_enu_m if len(point) >= 2]
        return bool(raw_points) and all(self.air_point_clear(point) for point in raw_points) and all(
            self.air_segment_clear(a, b) for a, b in zip(raw_points, raw_points[1:])
        )

    def repair_route_at_altitude(self, route_enu_m: Sequence[Sequence[float]], altitude_m: float) -> list[list[float]]:
        raw_points = [self._point_at_altitude(point, altitude_m) for point in route_enu_m if len(point) >= 2]
        if not raw_points:
            return []
        current = self.repair_point(raw_points[0], altitude_m)
        repaired = [current]
        for raw_target in raw_points[1:]:
            target = self.repair_point(raw_target, altitude_m)
            leg = self.repair_segment(current, target)
            if not leg:
                raise RuntimeError(f"Unable to repair UAV segment {current}->{target} at {altitude_m}m")
            for point in leg:
                if _dist_xy(point, repaired[-1]) > 0.05 or abs(point[2] - repaired[-1][2]) > 0.05:
                    repaired.append(point)
            current = repaired[-1]
        if not all(self.air_segment_clear(a, b) for a, b in zip(repaired, repaired[1:])):
            raise RuntimeError(f"Repaired UAV route still intersects an obstacle at {altitude_m}m")
        return repaired


class HighAltitudeCorridorPlanner:
    def __init__(
        self,
        spatial: MapSpatialIndex,
        buildings: BuildingObstacleIndex | None = None,
        *,
        corridor_width_m: float = 8.0,
        corridor_height_m: float = 8.0,
    ) -> None:
        self.spatial = spatial
        self.buildings = buildings or BuildingObstacleIndex(spatial)
        self.corridor_width_m = float(corridor_width_m)
        self.corridor_height_m = float(corridor_height_m)

    @staticmethod
    def _scene_center(points: Sequence[Sequence[float]]) -> list[float]:
        if not points:
            return [7000.0, 6200.0, 0.0]
        return [
            sum(float(point[0]) for point in points) / len(points),
            sum(float(point[1]) for point in points) / len(points),
            0.0,
        ]

    def _line_clear(self, a_enu_m: Sequence[float], b_enu_m: Sequence[float]) -> bool:
        line = LineString([(float(a_enu_m[0]), float(a_enu_m[1])), (float(b_enu_m[0]), float(b_enu_m[1]))])
        if not self.spatial.bounds_union.covers(line):
            return False
        if line.intersects(self.spatial.water_union):
            return False
        return self.buildings.point_clear(a_enu_m) and self.buildings.point_clear(b_enu_m) and self.buildings.segment_clear(a_enu_m, b_enu_m)

    def find_slots(
        self,
        reference_points_enu_m: Sequence[Sequence[float]],
        *,
        count: int = 5,
        slot_prefix: str = "uav_corridor",
    ) -> list[CorridorSlot]:
        center = self._scene_center(reference_points_enu_m)
        nearby_samples = sorted(
            self.spatial.lanes.samples,
            key=lambda sample: (sample.x_m - center[0]) ** 2 + (sample.y_m - center[1]) ** 2,
        )[:80]
        ordered_samples = []
        seen_edges: set[str] = set()
        for sample in nearby_samples:
            if sample.edge_id not in seen_edges:
                seen_edges.add(sample.edge_id)
                ordered_samples.append(sample)
        ordered_samples.extend(nearby_samples)

        lateral_offsets = (0.0, 12.0, -12.0, 24.0, -24.0, 36.0, -36.0, 48.0, -48.0, 8.0, -8.0, 16.0, -16.0)
        altitudes = (42.0, 48.0, 54.0, 60.0, 72.0, 90.0, 110.0, 130.0, 160.0)
        spans = (60.0, 40.0, 80.0, 100.0, 30.0)
        biases = (0.0, -20.0, 20.0, -40.0, 40.0)
        slots: list[CorridorSlot] = []
        for sample in ordered_samples:
            min_s, max_s = self.spatial.lanes.edge_s_bounds(sample.edge_id)
            for span_m in spans:
                for bias_m in biases:
                    s0 = max(min_s, min(max_s, sample.s_m - span_m + bias_m))
                    s1 = max(min_s, min(max_s, sample.s_m + span_m + bias_m))
                    if abs(s1 - s0) < 20.0:
                        continue
                    start_sample = self.spatial.lanes.resolve_edge_s(sample.edge_id, s0)
                    end_sample = self.spatial.lanes.resolve_edge_s(sample.edge_id, s1)
                    for altitude_m in altitudes:
                        for lateral_m in lateral_offsets:
                            start = offset_from_lane(start_sample, lateral_m, altitude_m)
                            end = offset_from_lane(end_sample, lateral_m, altitude_m)
                            if not self._line_clear(start, end):
                                continue
                            mid = [
                                _round3((start[0] + end[0]) * 0.5),
                                _round3((start[1] + end[1]) * 0.5),
                                _round3(altitude_m),
                            ]
                            too_close = False
                            for slot in slots:
                                if abs(mid[2] - slot.mid_enu_m[2]) >= 6.0:
                                    continue
                                if (
                                    _dist_xy(mid, slot.mid_enu_m) < self.corridor_width_m + 2.0
                                    or _dist_xy(start, slot.start_enu_m) < self.corridor_width_m + 2.0
                                    or _dist_xy(end, slot.end_enu_m) < self.corridor_width_m + 2.0
                                ):
                                    too_close = True
                                    break
                            if too_close:
                                continue
                            yaw_deg = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
                            length_m = max(1.0, _dist_xy(start, end))
                            slots.append(
                                CorridorSlot(
                                    slot_id=f"{slot_prefix}_{len(slots):02d}",
                                    edge_id=sample.edge_id,
                                    start_enu_m=start,
                                    end_enu_m=end,
                                    mid_enu_m=mid,
                                    yaw_deg=round(yaw_deg, 3),
                                    altitude_m=altitude_m,
                                    lateral_offset_m=float(lateral_m),
                                    length_m=round(length_m, 3),
                                    width_m=self.corridor_width_m,
                                    height_m=self.corridor_height_m,
                                )
                            )
                            if len(slots) >= count:
                                return slots
        raise RuntimeError(f"Unable to find {count} building-clear UAV corridor slots near {center}")

    def corridor_scene_entity(self, scenario_id: str, slot: CorridorSlot) -> tuple[dict[str, Any], dict[str, Any]]:
        entity_id = f"corridor_{scenario_id.lower().replace('-', '_')}_{slot.slot_id}"
        rotation = {"pitch_deg": 0.0, "yaw_deg": slot.yaw_deg, "roll_deg": 0.0}
        spec = {
            "entity_id": entity_id,
            "asset_id": UAV_CORRIDOR_LOGICAL_ASSET_ID,
            "initial_pos_enu": list(slot.mid_enu_m),
            "initial_rotation_deg": [0.0, 0.0, slot.yaw_deg],
            "movement_waypoints": [],
            "visual_state": {"mode": "semantic_corridor"},
        }
        scene = {
            "entity_id": entity_id,
            "logical_asset_id": UAV_CORRIDOR_LOGICAL_ASSET_ID,
            "category": "airspace_corridor",
            "placement_mode": "box_volume",
            "placement": {
                "center_enu_m": list(slot.mid_enu_m),
                "resolved_position_enu_m": list(slot.mid_enu_m),
                "extent_m": [round(slot.length_m * 0.5, 3), round(slot.width_m * 0.5, 3), round(slot.height_m * 0.5, 3)],
                "size_m": [round(slot.length_m, 3), round(slot.width_m, 3), round(slot.height_m, 3)],
                "rotation_deg": rotation,
                "scale_xyz": [round(slot.length_m, 3), round(slot.width_m, 3), round(slot.height_m, 3)],
                "corridor_slot_id": slot.slot_id,
                "edge_id": slot.edge_id,
                "altitude_layer_m": slot.altitude_m,
                "lateral_offset_m": slot.lateral_offset_m,
            },
            "initial_state": {
                "mode": "semantic_corridor",
                "semantic_class": UAV_CORRIDOR_CLASS_NAME,
                "custom_stencil_only": True,
            },
            "query_tags": ["UAVCorridor", "HighAltitudeCorridor", "event_semantic"],
            "activation_tick": 0,
            "enabled": True,
        }
        return spec, scene
