"""Scene occupancy authority for low-altitude PVU/facility episodes.

This module is intentionally shared by render-ready conversion, capture-filter
sync, and manual audits. It treats the RoadWay mesh-aligned lane graph as the
vehicle travel authority, while service facilities must live outside RoadWay
and lane buffers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from map_spatial_index import MapSpatialIndex, PEDESTRIAN_ROAD_BUFFER_M  # noqa: E402
from sumo_ground_flow.road_semantic_rules import (  # noqa: E402
    ROAD_BLOCKING_ASSETS,
    ROAD_CLOSURE_CLEARANCE_M,
    resolve_episode_road_semantics,
    vehicle_record_uses_closed_road,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAFFIC_BUNDLE = ROOT / "Config" / "LowAltitude" / "Maps" / "donghu_road_topo" / "traffic_bundle"
LANE_META_CSV = DEFAULT_TRAFFIC_BUNDLE / "lane_meta.csv"

LANE_HALF_WIDTH_M = 1.9
SERVICE_FACILITY_ASSETS = {
    "facility.charger.cityops.v1",
    "facility.landing_pad.visible.v1",
    "facility.radio.base_tower.v1",
}
LOGICAL_NONPHYSICAL_PREFIXES = (
    "semantic.",
    "trigger.",
)
SERVICE_MIN_CENTER_CLEARANCE_M = {
    "facility.landing_pad.visible.v1": LANE_HALF_WIDTH_M + 5.6,
    "facility.charger.cityops.v1": LANE_HALF_WIDTH_M + 3.6,
    "facility.radio.base_tower.v1": LANE_HALF_WIDTH_M + 4.5,
}
SERVICE_RADIUS_M = {
    "facility.landing_pad.visible.v1": 4.2,
    "facility.charger.cityops.v1": 1.8,
    "facility.radio.base_tower.v1": 2.8,
    "facility.barrier.basic": 1.6,
    "prop.roadwork.barrier.v1": 1.6,
    "prop.roadwork.construction_fence.v1": 2.2,
    "prop.roadwork.traffic_cone.v1": 0.9,
}
PVU_RADIUS_M = {
    "pedestrian": 0.55,
    "vehicle": 2.8,
    "uav": 1.4,
}
MAX_REPORTED_ISSUES = 80
MULTILANE_MIN_GROUP_SAMPLES = 80
MULTILANE_DOMINANT_SHARE_ERROR = 0.92
MULTILANE_DOMINANT_SHARE_WARNING = 0.86


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _asset_id(entity: dict[str, Any]) -> str:
    return str(entity.get("logical_asset_id") or entity.get("asset_id") or entity.get("proxy_template_id") or "")


def _entity_id(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_id") or entity.get("id") or "")


def _category(entity: dict[str, Any]) -> str:
    return str(
        entity.get("entity_category")
        or entity.get("category")
        or entity.get("entity_kind")
        or entity.get("entity_type")
        or ""
    )


def _point(value: Any) -> list[float] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1]), float(value[2] if len(value) > 2 else 0.0)]
        except (TypeError, ValueError):
            return None
    return None


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


def _xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _is_logical_nonphysical(asset_id: str) -> bool:
    return asset_id.startswith(LOGICAL_NONPHYSICAL_PREFIXES)


def _is_pvu(entity: dict[str, Any]) -> bool:
    category = _category(entity).lower()
    asset = _asset_id(entity)
    return (
        category in {"pedestrian", "vehicle", "uav"}
        or asset.startswith("pedestrian.")
        or asset.startswith("vehicle.")
        or asset.startswith("uav.")
    )


def _pvu_kind(entity: dict[str, Any]) -> str:
    category = _category(entity).lower()
    asset = _asset_id(entity)
    if "vehicle" in category or asset.startswith("vehicle."):
        return "vehicle"
    if "uav" in category or asset.startswith("uav."):
        return "uav"
    return "pedestrian"


def _radius_for_entity(entity: dict[str, Any]) -> float:
    asset = _asset_id(entity)
    category = _pvu_kind(entity) if _is_pvu(entity) else _category(entity)
    if category == "vehicle":
        sumo_vehicle = entity.get("sumo_vehicle")
        if isinstance(sumo_vehicle, dict):
            dims = sumo_vehicle.get("dimensions_m")
            if isinstance(dims, dict):
                try:
                    length = float(dims.get("length") or 5.0)
                    width = float(dims.get("width") or 1.8)
                    return 0.5 * math.hypot(length, width)
                except (TypeError, ValueError):
                    pass
    if category in PVU_RADIUS_M:
        return PVU_RADIUS_M[category]
    return SERVICE_RADIUS_M.get(asset, 1.5)


def _is_intentional_road_block(entity: dict[str, Any]) -> bool:
    occupancy = entity.get("scene_occupancy")
    if isinstance(occupancy, dict) and occupancy.get("intentional_road_blocking") is True:
        return True
    road_blocking = entity.get("road_blocking")
    if isinstance(road_blocking, dict) and road_blocking.get("intentional") is True:
        return True
    state = entity.get("initial_state")
    if isinstance(state, dict) and state.get("intentional_road_blocking") is True:
        return True
    return False


def _lane_edge_id_from_vehicle(vehicle: dict[str, Any]) -> str:
    sumo_vehicle = vehicle.get("sumo_vehicle")
    if isinstance(sumo_vehicle, dict):
        edge_id = str(sumo_vehicle.get("sumo_edge_id") or "")
        if edge_id:
            return edge_id
        lane_id = str(sumo_vehicle.get("sumo_lane_id") or "")
        if lane_id.endswith("_0"):
            return lane_id[:-2]
        return lane_id
    lane_id = str(vehicle.get("sumo_edge_id") or vehicle.get("sumo_lane_id") or "")
    if lane_id.endswith("_0"):
        return lane_id[:-2]
    return lane_id


def _is_sumo_background_vehicle(entity: dict[str, Any]) -> bool:
    if _pvu_kind(entity) != "vehicle":
        return False
    if str(entity.get("source") or "") != "sumo_traci":
        return False
    sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
    if bool(sumo_vehicle.get("semantic_vehicle")):
        return False
    if str(entity.get("background_role") or "") == "sumo_background_traffic":
        return True
    if str(sumo_vehicle.get("control_role") or "") == "background":
        return True
    metadata = dict(sumo_vehicle.get("semantic_metadata") or {})
    return str(metadata.get("role") or "") == "background" or str(metadata.get("traffic_role") or "").startswith(
        "deterministic_background"
    )


def load_lane_meta(lane_meta_csv: Path = LANE_META_CSV) -> dict[str, dict[str, Any]]:
    if not lane_meta_csv.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with lane_meta_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            edge_id = str(row.get("edge_id") or "")
            if not edge_id.startswith("cg_edge_"):
                continue
            direction_role = str(row.get("direction_role") or "")
            if direction_role not in {"f", "r"}:
                continue
            try:
                physical_lane_index = int(float(row.get("physical_lane_index") or 0))
            except (TypeError, ValueError):
                physical_lane_index = 0
            rows[edge_id] = {
                "source_road_id": str(row.get("source_road_id") or ""),
                "direction_role": direction_role,
                "physical_lane_index": physical_lane_index,
                "lane_width_m": float(row.get("lane_width_m") or 0.0),
            }
    return rows


def _audit_static_placement(
    roster_entities: list[dict[str, Any]],
    *,
    spatial: MapSpatialIndex,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    manifest_entities: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for entity in roster_entities:
        entity_id = _entity_id(entity)
        asset = _asset_id(entity)
        category = _category(entity)
        pos = entity_position(entity)
        if pos is None:
            continue
        clearance_m = round(float(spatial.nearest_lane_clearance(pos)), 3)
        role = "logical_nonphysical" if _is_logical_nonphysical(asset) else "physical"
        occupant = {
            "entity_id": entity_id,
            "asset_id": asset,
            "category": category,
            "role": role,
            "position_enu_m": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
            "nearest_lane_clearance_m": clearance_m,
        }
        if asset in SERVICE_FACILITY_ASSETS:
            required = SERVICE_MIN_CENTER_CLEARANCE_M[asset]
            point_errors = spatial.validation_errors_for_point(
                pos,
                context=f"service facility {entity_id}",
                allow_road=False,
                allow_green=True,
                road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
            )
            if point_errors:
                errors.extend(point_errors)
            if clearance_m < required:
                warnings.append(
                    f"service facility {entity_id} ({asset}) violates lane clearance: "
                    f"{clearance_m:.3f}m < {required:.3f}m"
                )
            occupant.update(
                {
                    "semantic_layer": "facility",
                    "blocking": False,
                    "service_facility": True,
                    "required_lane_clearance_m": round(required, 3),
                }
            )
        elif asset in ROAD_BLOCKING_ASSETS:
            on_road = clearance_m < ROAD_CLOSURE_CLEARANCE_M
            occupant.update(
                {
                    "semantic_layer": "facility",
                    "blocking": True,
                    "intentional_road_blocking": _is_intentional_road_block(entity),
                }
            )
            if on_road and not _is_intentional_road_block(entity):
                warnings.append(
                    f"road-blocking asset {entity_id} ({asset}) is on RoadWay without explicit closure metadata; "
                    "shared road_semantic_rules will infer a scene-scoped road closure"
                )
        manifest_entities.append(occupant)
    return manifest_entities, errors, warnings


def _static_physical_assets(roster_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entity in roster_entities:
        asset = _asset_id(entity)
        if asset in SERVICE_FACILITY_ASSETS or asset in ROAD_BLOCKING_ASSETS:
            pos = entity_position(entity)
            if pos is not None:
                out.append(entity)
    return out


def _audit_pvu_static_overlap(
    frames_path: Path,
    static_assets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if not frames_path.exists() or not static_assets:
        return [], [], []
    errors: list[str] = []
    warnings: list[str] = []
    overlaps: list[dict[str, Any]] = []
    static_specs: list[tuple[dict[str, Any], list[float], float]] = []
    for static in static_assets:
        pos = entity_position(static)
        if pos is None:
            continue
        static_specs.append((static, pos, _radius_for_entity(static)))
    checked_records = 0
    for frame in iter_jsonl(frames_path):
        tick = int(frame.get("tick") or 0)
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict) or not _is_pvu(entity):
                continue
            pvu_pos = entity_position(entity)
            if pvu_pos is None:
                continue
            pvu_kind = _pvu_kind(entity)
            if pvu_kind == "uav" and float(pvu_pos[2]) > 8.0:
                continue
            pvu_radius = _radius_for_entity(entity)
            for static, static_pos, static_radius in static_specs:
                static_asset = _asset_id(static)
                if pvu_kind == "uav" and static_asset == "facility.landing_pad.visible.v1":
                    continue
                dist = _xy_distance(pvu_pos, static_pos)
                threshold = pvu_radius + static_radius
                if dist + 1e-6 >= threshold:
                    continue
                overlap = {
                    "tick": tick,
                    "pvu_entity_id": _entity_id(entity),
                    "pvu_kind": pvu_kind,
                    "static_entity_id": _entity_id(static),
                    "static_asset_id": static_asset,
                    "distance_m": round(dist, 3),
                    "threshold_m": round(threshold, 3),
                }
                overlaps.append(overlap)
                message = (
                    f"PVU/static overlap at tick {tick}: {_entity_id(entity)} vs {_entity_id(static)} "
                    f"({dist:.3f}m < {threshold:.3f}m)"
                )
                if static_asset in SERVICE_FACILITY_ASSETS:
                    if len(warnings) < MAX_REPORTED_ISSUES:
                        warnings.append(f"nonblocking service facility {message}")
                elif static_asset in ROAD_BLOCKING_ASSETS:
                    if len(warnings) < MAX_REPORTED_ISSUES:
                        warnings.append(f"intentional road-blocking event prop {message}")
                else:
                    errors.append(message)
                    if len(errors) >= MAX_REPORTED_ISSUES:
                        return overlaps, errors, warnings
            checked_records += 1
    if checked_records == 0:
        warnings.append("no PVU records found for static overlap audit")
    return overlaps, errors, warnings


def _audit_road_semantic_vehicle_violations(
    frames_path: Path,
    road_semantics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not frames_path.exists():
        return [], []
    closed_edges = {str(edge_id) for edge_id in road_semantics.get("closed_edges") or [] if str(edge_id)}
    closed_lanes = {str(lane_id) for lane_id in road_semantics.get("closed_lanes") or [] if str(lane_id)}
    if not closed_edges and not closed_lanes:
        return [], []
    violations: list[dict[str, Any]] = []
    errors: list[str] = []
    for frame in iter_jsonl(frames_path):
        tick = int(frame.get("tick") or 0)
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            if not _is_sumo_background_vehicle(entity):
                continue
            if not vehicle_record_uses_closed_road(entity, _RoadSemanticsView(closed_edges, closed_lanes)):
                continue
            sumo_vehicle = entity.get("sumo_vehicle")
            if isinstance(sumo_vehicle, dict):
                edge_id = str(sumo_vehicle.get("sumo_edge_id") or "")
                lane_id = str(sumo_vehicle.get("sumo_lane_id") or "")
            else:
                edge_id = str(entity.get("sumo_edge_id") or "")
                lane_id = str(entity.get("sumo_lane_id") or "")
            violation = {
                "tick": tick,
                "vehicle_id": _entity_id(entity),
                "edge_id": edge_id,
                "lane_id": lane_id,
            }
            violations.append(violation)
            if len(errors) < MAX_REPORTED_ISSUES:
                errors.append(
                    f"vehicle {_entity_id(entity)} uses road-closure edge/lane at tick {tick}: "
                    f"edge={edge_id} lane={lane_id}"
                )
    return violations, errors


class _RoadSemanticsView:
    def __init__(self, closed_edges: set[str], closed_lanes: set[str]) -> None:
        self.closed_edges = closed_edges
        self.closed_lanes = closed_lanes

    def is_edge_closed(self, edge_id: Any) -> bool:
        return str(edge_id or "") in self.closed_edges

    def is_lane_closed(self, lane_id: Any) -> bool:
        lane_text = str(lane_id or "")
        if lane_text in self.closed_lanes:
            return True
        if lane_text.endswith("_0") and lane_text[:-2] in self.closed_edges:
            return True
        return False


def _audit_multilane_utilization(
    frames_path: Path,
    *,
    lane_meta: dict[str, dict[str, Any]],
    underuse_blocking: bool = True,
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_groups: dict[tuple[str, str], set[int]] = defaultdict(set)
    for meta in lane_meta.values():
        key = (str(meta["source_road_id"]), str(meta["direction_role"]))
        expected_groups[key].add(int(meta["physical_lane_index"]))

    counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    if frames_path.exists():
        for frame in iter_jsonl(frames_path):
            for entity in frame.get("entities") or []:
                if not isinstance(entity, dict):
                    continue
                if _pvu_kind(entity) != "vehicle":
                    continue
                edge_id = _lane_edge_id_from_vehicle(entity)
                meta = lane_meta.get(edge_id)
                if not meta:
                    continue
                key = (str(meta["source_road_id"]), str(meta["direction_role"]))
                counts[key][int(meta["physical_lane_index"])] += 1

    groups: list[dict[str, Any]] = []
    for key, expected_lanes in sorted(expected_groups.items(), key=lambda item: (int(item[0][0] or 0), item[0][1])):
        if len(expected_lanes) < 2:
            continue
        observed = counts.get(key, Counter())
        total = sum(observed.values())
        if total <= 0:
            continue
        dominant = max(observed.values()) / float(total)
        group_record = {
            "source_road_id": key[0],
            "direction_role": key[1],
            "expected_physical_lanes": sorted(expected_lanes),
            "observed_lane_sample_counts": {str(k): int(v) for k, v in sorted(observed.items())},
            "total_vehicle_samples": int(total),
            "used_physical_lane_count": len(observed),
            "dominant_lane_share": round(dominant, 4),
        }
        groups.append(group_record)
        if total < MULTILANE_MIN_GROUP_SAMPLES:
            continue
        if len(observed) < min(2, len(expected_lanes)):
            message = (
                f"multi-lane underuse road={key[0]} direction={key[1]}: "
                f"used {len(observed)}/{len(expected_lanes)} lanes over {total} samples"
            )
            if underuse_blocking and len(expected_lanes) >= 3:
                errors.append(message)
            else:
                warnings.append(message)
        elif dominant >= MULTILANE_DOMINANT_SHARE_ERROR:
            message = f"multi-lane imbalance road={key[0]} direction={key[1]}: dominant share {dominant:.3f}"
            warnings.append(message)
        elif dominant >= MULTILANE_DOMINANT_SHARE_WARNING:
            warnings.append(
                f"multi-lane skew road={key[0]} direction={key[1]}: dominant share {dominant:.3f}"
            )
    summary = {
        "policy": "roi_multilane_vehicle_sample_distribution_v2",
        "underuse_blocking": bool(underuse_blocking),
        "groups_checked": len(groups),
        "groups": groups,
    }
    return summary, errors, warnings


def audit_episode_dir(
    episode_dir: Path,
    *,
    spatial: MapSpatialIndex | None = None,
    lane_meta_csv: Path = LANE_META_CSV,
    write_manifest: bool = False,
) -> dict[str, Any]:
    episode_dir = Path(episode_dir)
    roster_path = episode_dir / "global_entity_roster.json"
    frames_path = episode_dir / "truth_frames.jsonl"
    if not roster_path.exists():
        raise FileNotFoundError(f"missing global_entity_roster.json: {episode_dir}")
    roster_root = read_json(roster_path)
    roster_entities = [entity for entity in roster_root.get("entities") or [] if isinstance(entity, dict)]
    manifest_path = episode_dir / "episode_manifest.json"
    manifest_payload = read_json(manifest_path) if manifest_path.exists() else {}
    source_vehicle_authority = dict(manifest_payload.get("source_vehicle_authority") or {})
    sumo_only_vehicle_authority = (
        source_vehicle_authority.get("enabled") is True
        and str(source_vehicle_authority.get("policy") or "").strip()
        == "sumo_only_vehicle_authority_replaces_source_vehicle_truth_v2"
    )
    spatial = spatial or MapSpatialIndex.default(ROOT)
    lane_meta = load_lane_meta(lane_meta_csv)
    road_semantics_obj = resolve_episode_road_semantics(
        episode_dir,
        lane_meta_csv=lane_meta_csv,
        lane_center_samples_csv=Path(lane_meta_csv).with_name("lane_center_samples.csv"),
    )
    road_semantics = road_semantics_obj.as_dict()

    static_manifest, static_errors, static_warnings = _audit_static_placement(roster_entities, spatial=spatial)
    closure_records_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for closure in road_semantics.get("closures") or []:
        if isinstance(closure, dict):
            closure_records_by_entity[str(closure.get("entity_id") or "")].append(closure)
    for occupant in static_manifest:
        records = closure_records_by_entity.get(str(occupant.get("entity_id") or ""))
        if not records:
            continue
        occupant["road_closure"] = {
            "authority": road_semantics.get("authority"),
            "policy": road_semantics.get("policy"),
            "closed_edges": sorted({str(record.get("edge_id") or "") for record in records if record.get("edge_id")}),
            "closed_lanes": sorted({str(record.get("lane_id") or "") for record in records if record.get("lane_id")}),
            "closed_source_roads": sorted(
                {str(record.get("source_road_id") or "") for record in records if record.get("source_road_id")}
            ),
            "records": records,
        }
    overlaps, overlap_errors, overlap_warnings = _audit_pvu_static_overlap(
        frames_path,
        _static_physical_assets(roster_entities),
    )
    closure_violations, closure_errors = _audit_road_semantic_vehicle_violations(
        frames_path,
        road_semantics,
    )
    multilane_summary, multilane_errors, multilane_warnings = _audit_multilane_utilization(
        frames_path,
        lane_meta=lane_meta,
        underuse_blocking=not sumo_only_vehicle_authority,
    )

    errors = [*static_errors, *overlap_errors, *closure_errors, *multilane_errors]
    warnings = [*static_warnings, *overlap_warnings, *multilane_warnings]
    category_counts = Counter(_category(entity) for entity in roster_entities)
    manifest = {
        "schema_name": "scene_occupancy_manifest",
        "schema_version": "v1",
        "episode_id": episode_dir.name,
        "authority": "low_altitude_scene_occupancy_authority_v1",
        "geometry_authority": "road_geojson_topology_plus_ue_roadway_mesh",
        "lane_geometry_source": "ue_roadway_mesh",
        "inputs": {
            "episode_dir": _repo_relative(episode_dir),
            "lane_meta_csv": _repo_relative(lane_meta_csv),
            "map_spatial_index": "MapSpatialIndex.default",
        },
        "entity_counts_by_category": dict(sorted(category_counts.items())),
        "static_occupants": static_manifest,
        "pvu_static_overlaps": overlaps,
        "road_semantics": road_semantics,
        "road_closure_vehicle_violations": closure_violations,
        "multi_lane_utilization": multilane_summary,
        "validation_summary": {
            "ok": not errors,
            "errors": errors[:MAX_REPORTED_ISSUES],
            "warnings": warnings[:MAX_REPORTED_ISSUES],
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
    }
    if write_manifest:
        write_json(episode_dir / "scene_occupancy_manifest.json", manifest)
    return manifest


def update_episode_manifests_with_scene_occupancy(
    episode_dir: Path,
    scene_occupancy: dict[str, Any],
) -> None:
    artifact_path = _repo_relative(Path(episode_dir) / "scene_occupancy_manifest.json")
    for filename in ("episode_manifest.json", "scenario_plan.json"):
        path = Path(episode_dir) / filename
        if not path.exists():
            continue
        payload = read_json(path)
        payload["scene_occupancy"] = {
            "authority": scene_occupancy.get("authority"),
            "geometry_authority": scene_occupancy.get("geometry_authority"),
            "lane_geometry_source": scene_occupancy.get("lane_geometry_source"),
            "validation_summary": scene_occupancy.get("validation_summary"),
            "artifact": artifact_path,
        }
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, dict):
            artifacts["scene_occupancy_manifest"] = artifact_path
        canonical = payload.get("canonical_artifacts")
        if isinstance(canonical, dict):
            canonical["scene_occupancy_manifest"] = artifact_path
        export_contract = payload.get("export_contract")
        if isinstance(export_contract, dict):
            contract_artifacts = export_contract.setdefault("artifacts", {})
            if isinstance(contract_artifacts, dict):
                contract_artifacts["scene_occupancy_manifest"] = "scene_occupancy_manifest.json"
        write_json(path, payload)


def audit_and_attach_episode(episode_dir: Path, *, write_manifest: bool = True) -> dict[str, Any]:
    scene_occupancy = audit_episode_dir(episode_dir, write_manifest=write_manifest)
    if write_manifest:
        update_episode_manifests_with_scene_occupancy(episode_dir, scene_occupancy)
    summary = dict(scene_occupancy.get("validation_summary") or {})
    if not summary.get("ok"):
        errors = summary.get("errors") or []
        raise RuntimeError(
            f"{Path(episode_dir).name}: scene occupancy audit failed with "
            f"{summary.get('error_count', len(errors))} error(s): {errors[:6]}"
        )
    return scene_occupancy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit low-altitude scene occupancy for render-ready episodes.")
    parser.add_argument("--input-root", type=Path, default=None, help="Episode root containing render-ready directories.")
    parser.add_argument("--episode", action="append", default=[], help="Episode directory name under --input-root.")
    parser.add_argument("--episode-dir", action="append", type=Path, default=[], help="Explicit episode directory.")
    parser.add_argument("--write-manifest", action="store_true", help="Write scene_occupancy_manifest.json and attach it.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    episode_dirs: list[Path] = []
    if args.episode_dir:
        episode_dirs.extend(Path(path) for path in args.episode_dir)
    if args.input_root:
        input_root = Path(args.input_root)
        if args.episode:
            episode_dirs.extend(input_root / name for name in args.episode)
        else:
            episode_dirs.extend(
                path for path in sorted(input_root.iterdir()) if path.is_dir() and (path / "truth_frames.jsonl").exists()
            )
    if not episode_dirs:
        raise SystemExit("Provide --input-root or --episode-dir.")

    results: list[dict[str, Any]] = []
    ok = True
    for episode_dir in episode_dirs:
        manifest = audit_episode_dir(episode_dir, write_manifest=args.write_manifest)
        if args.write_manifest:
            update_episode_manifests_with_scene_occupancy(episode_dir, manifest)
        summary = dict(manifest.get("validation_summary") or {})
        ok = ok and bool(summary.get("ok"))
        results.append(
            {
                "episode": Path(episode_dir).name,
                "ok": bool(summary.get("ok")),
                "error_count": int(summary.get("error_count") or 0),
                "warning_count": int(summary.get("warning_count") or 0),
                "errors": list(summary.get("errors") or [])[:8],
                "warnings": list(summary.get("warnings") or [])[:8],
            }
        )
    print(json.dumps({"ok": ok, "episodes": results}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
