"""Shared road semantic rules for SUMO traffic and truth validation.

This module is the single policy entry for scene-scoped road attributes that
must affect vehicle generation or validation.  It currently exports static
road-closure semantics inferred from road-blocking facilities/props placed on
RoadWay lane centers.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = ROOT / "Dataset" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from map_spatial_index import LANE_HALF_WIDTH_M, PEDESTRIAN_ROAD_BUFFER_M, LaneResolver, LaneSample  # noqa: E402


DEFAULT_TRAFFIC_BUNDLE = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
DEFAULT_LANE_CENTER_SAMPLES = DEFAULT_TRAFFIC_BUNDLE / "lane_center_samples.csv"
DEFAULT_LANE_META_CSV = DEFAULT_TRAFFIC_BUNDLE / "lane_meta.csv"

ROAD_BLOCKING_ASSETS = frozenset(
    {
        "facility.barrier.basic",
        "prop.roadwork.barrier.v1",
        "prop.roadwork.construction_fence.v1",
        "prop.roadwork.traffic_cone.v1",
    }
)
ROAD_BLOCKING_ASSET_RADIUS_M = {
    "facility.barrier.basic": 1.6,
    "prop.roadwork.barrier.v1": 1.6,
    "prop.roadwork.construction_fence.v1": 2.2,
    "prop.roadwork.traffic_cone.v1": 0.9,
}
ROAD_CLOSURE_CLEARANCE_M = LANE_HALF_WIDTH_M + PEDESTRIAN_ROAD_BUFFER_M
ROAD_CLOSURE_POLICY = "static_road_blocking_asset_closes_matched_lane_edges_v1"
ROAD_SEMANTICS_AUTHORITY = "road_semantic_rules_v1"


@dataclass(frozen=True)
class LaneAttributes:
    edge_id: str
    lane_id: str
    source_road_id: str
    direction_role: str
    physical_lane_index: int
    lane_width_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "lane_id": self.lane_id,
            "source_road_id": self.source_road_id,
            "direction_role": self.direction_role,
            "physical_lane_index": self.physical_lane_index,
            "lane_width_m": round(float(self.lane_width_m), 6),
        }


@dataclass(frozen=True)
class RoadClosureRecord:
    entity_id: str
    asset_id: str
    position_enu_m: tuple[float, float, float]
    nearest_lane_clearance_m: float
    edge_id: str
    lane_id: str
    source_road_id: str
    direction_role: str
    physical_lane_index: int
    matched_lane_distance_m: float
    matched_sample_s_m: float
    policy: str = ROAD_CLOSURE_POLICY

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "asset_id": self.asset_id,
            "position_enu_m": [round(float(value), 3) for value in self.position_enu_m],
            "nearest_lane_clearance_m": round(float(self.nearest_lane_clearance_m), 3),
            "edge_id": self.edge_id,
            "lane_id": self.lane_id,
            "source_road_id": self.source_road_id,
            "direction_role": self.direction_role,
            "physical_lane_index": int(self.physical_lane_index),
            "matched_lane_distance_m": round(float(self.matched_lane_distance_m), 3),
            "matched_sample_s_m": round(float(self.matched_sample_s_m), 3),
            "policy": self.policy,
        }


@dataclass(frozen=True)
class SceneRoadSemantics:
    episode_id: str
    closure_records: tuple[RoadClosureRecord, ...]
    off_road_blocking_assets: tuple[dict[str, Any], ...]
    lane_attributes_by_edge: dict[str, LaneAttributes]

    @property
    def closed_edges(self) -> set[str]:
        return {record.edge_id for record in self.closure_records if record.edge_id}

    @property
    def closed_lanes(self) -> set[str]:
        return {record.lane_id for record in self.closure_records if record.lane_id}

    @property
    def closed_source_roads(self) -> set[str]:
        return {record.source_road_id for record in self.closure_records if record.source_road_id}

    def is_edge_closed(self, edge_id: Any) -> bool:
        return str(edge_id or "") in self.closed_edges

    def is_lane_closed(self, lane_id: Any) -> bool:
        lane_text = str(lane_id or "")
        if lane_text in self.closed_lanes:
            return True
        if lane_text.endswith("_0") and lane_text[:-2] in self.closed_edges:
            return True
        return False

    def as_dict(self) -> dict[str, Any]:
        records = [record.as_dict() for record in self.closure_records]
        return {
            "authority": ROAD_SEMANTICS_AUTHORITY,
            "policy": ROAD_CLOSURE_POLICY,
            "episode_id": self.episode_id,
            "road_blocking_assets": sorted(ROAD_BLOCKING_ASSETS),
            "road_closure_clearance_m": round(float(ROAD_CLOSURE_CLEARANCE_M), 3),
            "road_blocking_asset_radius_m": {
                key: round(float(value), 3) for key, value in sorted(ROAD_BLOCKING_ASSET_RADIUS_M.items())
            },
            "closed_edge_count": len(self.closed_edges),
            "closed_lane_count": len(self.closed_lanes),
            "closed_source_road_count": len(self.closed_source_roads),
            "closed_edges": sorted(self.closed_edges),
            "closed_lanes": sorted(self.closed_lanes),
            "closed_source_roads": sorted(self.closed_source_roads),
            "closures": records,
            "off_road_blocking_assets": list(self.off_road_blocking_assets),
        }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def asset_id(entity: dict[str, Any]) -> str:
    return str(entity.get("logical_asset_id") or entity.get("asset_id") or entity.get("proxy_template_id") or "")


def entity_id(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_id") or entity.get("id") or "")


def entity_position(entity: dict[str, Any]) -> list[float] | None:
    truth_pose = entity.get("truth_pose")
    if isinstance(truth_pose, dict):
        point = _point(truth_pose.get("position_enu_m"))
        if point is not None:
            return point
    for key in ("position_enu_m", "initial_position_enu_m", "initial_pos_enu", "resolved_position_enu_m"):
        point = _point(entity.get(key))
        if point is not None:
            return point
    placement = entity.get("placement")
    if isinstance(placement, dict):
        for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
            point = _point(placement.get(key))
            if point is not None:
                return point
    return None


def load_lane_attributes(lane_meta_csv: Path = DEFAULT_LANE_META_CSV) -> dict[str, LaneAttributes]:
    rows: dict[str, LaneAttributes] = {}
    if not Path(lane_meta_csv).exists():
        return rows
    with Path(lane_meta_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            edge_id = str(row.get("edge_id") or "")
            direction_role = str(row.get("direction_role") or "")
            if not edge_id.startswith("cg_edge_") or direction_role not in {"f", "r"}:
                continue
            try:
                physical_lane_index = int(float(row.get("physical_lane_index") or 0))
            except (TypeError, ValueError):
                physical_lane_index = 0
            try:
                lane_width_m = float(row.get("lane_width_m") or 0.0)
            except (TypeError, ValueError):
                lane_width_m = 0.0
            rows[edge_id] = LaneAttributes(
                edge_id=edge_id,
                lane_id=str(row.get("lane_id") or f"{edge_id}_0"),
                source_road_id=str(row.get("source_road_id") or ""),
                direction_role=direction_role,
                physical_lane_index=physical_lane_index,
                lane_width_m=lane_width_m,
            )
    return rows


def resolve_episode_road_semantics(
    episode_dir: Path,
    *,
    episode_id: str | None = None,
    lane_center_samples_csv: Path = DEFAULT_LANE_CENTER_SAMPLES,
    lane_meta_csv: Path = DEFAULT_LANE_META_CSV,
) -> SceneRoadSemantics:
    episode_dir = Path(episode_dir)
    roster_path = episode_dir / "global_entity_roster.json"
    if not roster_path.exists():
        return SceneRoadSemantics(
            episode_id=episode_id or episode_dir.name,
            closure_records=(),
            off_road_blocking_assets=(),
            lane_attributes_by_edge=load_lane_attributes(lane_meta_csv),
        )
    roster = read_json(roster_path)
    entities = roster.get("entities") if isinstance(roster, dict) else roster
    if not isinstance(entities, list):
        entities = []
    return resolve_scene_road_semantics(
        [entity for entity in entities if isinstance(entity, dict)],
        episode_id=episode_id or episode_dir.name,
        lane_center_samples_csv=lane_center_samples_csv,
        lane_meta_csv=lane_meta_csv,
    )


def resolve_scene_road_semantics(
    scene_entities: Sequence[dict[str, Any]],
    *,
    episode_id: str,
    lane_center_samples_csv: Path = DEFAULT_LANE_CENTER_SAMPLES,
    lane_meta_csv: Path = DEFAULT_LANE_META_CSV,
) -> SceneRoadSemantics:
    blocking_entities = [
        entity
        for entity in scene_entities
        if asset_id(entity) in ROAD_BLOCKING_ASSETS and entity_position(entity) is not None
    ]
    if not blocking_entities:
        return SceneRoadSemantics(
            episode_id=str(episode_id),
            closure_records=(),
            off_road_blocking_assets=(),
            lane_attributes_by_edge={},
        )
    lane_attributes = load_lane_attributes(lane_meta_csv)
    if not Path(lane_center_samples_csv).exists() or not lane_attributes:
        return SceneRoadSemantics(
            episode_id=str(episode_id),
            closure_records=(),
            off_road_blocking_assets=(),
            lane_attributes_by_edge=lane_attributes,
        )
    resolver = LaneResolver(Path(lane_center_samples_csv))
    records: list[RoadClosureRecord] = []
    off_road: list[dict[str, Any]] = []
    for entity in blocking_entities:
        asset = asset_id(entity)
        pos = entity_position(entity)
        if pos is None:
            continue
        edge_distances = _nearest_edge_distances(
            resolver=resolver,
            lane_attributes=lane_attributes,
            position_enu_m=pos,
        )
        if not edge_distances:
            continue
        nearest_distance = edge_distances[0][0]
        if nearest_distance > ROAD_CLOSURE_CLEARANCE_M + 1e-6:
            off_road.append(
                {
                    "entity_id": entity_id(entity),
                    "asset_id": asset,
                    "position_enu_m": [round(float(value), 3) for value in _point3(pos)],
                    "nearest_lane_clearance_m": round(float(nearest_distance), 3),
                    "classification": "off_road_blocking_context",
                }
            )
            continue
        closure_distance_m = ROAD_CLOSURE_CLEARANCE_M + float(ROAD_BLOCKING_ASSET_RADIUS_M.get(asset, 1.5))
        for distance_m, sample in edge_distances:
            if distance_m > closure_distance_m + 1e-6:
                break
            attrs = lane_attributes.get(sample.edge_id)
            if attrs is None:
                continue
            records.append(
                RoadClosureRecord(
                    entity_id=entity_id(entity),
                    asset_id=asset,
                    position_enu_m=tuple(_point3(pos)),
                    nearest_lane_clearance_m=float(nearest_distance),
                    edge_id=sample.edge_id,
                    lane_id=attrs.lane_id or sample.lane_id,
                    source_road_id=attrs.source_road_id,
                    direction_role=attrs.direction_role,
                    physical_lane_index=attrs.physical_lane_index,
                    matched_lane_distance_m=float(distance_m),
                    matched_sample_s_m=float(sample.s_m),
                )
            )
    deduped = _dedupe_closure_records(records)
    return SceneRoadSemantics(
        episode_id=str(episode_id),
        closure_records=tuple(deduped),
        off_road_blocking_assets=tuple(off_road),
        lane_attributes_by_edge=lane_attributes,
    )


def vehicle_record_uses_closed_road(entity: dict[str, Any], semantics: SceneRoadSemantics) -> bool:
    if not semantics.closed_edges and not semantics.closed_lanes:
        return False
    sumo_vehicle = entity.get("sumo_vehicle")
    if isinstance(sumo_vehicle, dict):
        edge_id = str(sumo_vehicle.get("sumo_edge_id") or "")
        lane_id = str(sumo_vehicle.get("sumo_lane_id") or "")
    else:
        edge_id = str(entity.get("sumo_edge_id") or "")
        lane_id = str(entity.get("sumo_lane_id") or "")
    if edge_id and semantics.is_edge_closed(edge_id):
        return True
    if lane_id and semantics.is_lane_closed(lane_id):
        return True
    return False


def _point(value: Any) -> list[float] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        try:
            return [
                float(value[0]),
                float(value[1]),
                float(value[2] if len(value) > 2 else 0.0),
            ]
        except (TypeError, ValueError):
            return None
    return None


def _point3(value: Sequence[float]) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2] if len(value) > 2 else 0.0)]


def _distance_xy(point: Sequence[float], sample: LaneSample) -> float:
    return math.hypot(float(point[0]) - sample.x_m, float(point[1]) - sample.y_m)


def _nearest_edge_distances(
    *,
    resolver: LaneResolver,
    lane_attributes: dict[str, LaneAttributes],
    position_enu_m: Sequence[float],
) -> list[tuple[float, LaneSample]]:
    best_by_edge: dict[str, tuple[float, LaneSample]] = {}
    valid_edges = set(lane_attributes)
    for sample in resolver.samples:
        if sample.edge_id not in valid_edges:
            continue
        distance = _distance_xy(position_enu_m, sample)
        previous = best_by_edge.get(sample.edge_id)
        if previous is None or distance < previous[0]:
            best_by_edge[sample.edge_id] = (distance, sample)
    values = list(best_by_edge.values())
    values.sort(key=lambda item: (round(float(item[0]), 6), item[1].edge_id, item[1].s_m))
    return values


def _dedupe_closure_records(records: Sequence[RoadClosureRecord]) -> list[RoadClosureRecord]:
    best: dict[tuple[str, str, str], RoadClosureRecord] = {}
    for record in records:
        key = (record.entity_id, record.edge_id, record.lane_id)
        previous = best.get(key)
        if previous is None or record.matched_lane_distance_m < previous.matched_lane_distance_m:
            best[key] = record
    return sorted(
        best.values(),
        key=lambda item: (
            item.entity_id,
            item.source_road_id,
            item.direction_role,
            item.physical_lane_index,
            item.edge_id,
        ),
    )
