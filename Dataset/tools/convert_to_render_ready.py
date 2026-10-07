from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import orjson
except (
    ModuleNotFoundError
):  # pragma: no cover - optional speedup for trajectory exports.
    orjson = None

from pedestrian_activity_catalog import activity_annotations, normalize_activity_type
from runtime_state_contract import (
    RUNTIME_STATE_FIELDS,
    invalid_runtime_state_value_paths,
    unconsumed_runtime_state_paths,
)
from inspect_observation_contract import (
    point_in_oriented_frustum_footprint_xy,
    route_observes_boundary as inspect_route_observes_boundary,
)
from Dataset.semantic_truth.core_semantic_registry import (
    get_governed_parameter_defaults,
)
from Dataset.semantic_truth.facility_scope import validate_roster_facility_scope
from Dataset.semantic_truth.entity_scope import roster_index, uav_task_role
from Dataset.semantic_truth.formal_clock import FORMAL_TICK_HZ, validate_formal_frame_clock
from Dataset.tools.runtime_state_source_guard import forbidden_runtime_state_paths
from Dataset.tools.weather_fields import WEATHER_ALIASES, WEATHER_FRACTION_FIELDS, WEATHER_SOURCE_FIELDS

try:
    from scene_occupancy import (
        SERVICE_FACILITY_ASSETS,
        SERVICE_MIN_CENTER_CLEARANCE_M,
        audit_and_attach_episode,
    )
except (
    ModuleNotFoundError
):  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.scene_occupancy import (
        SERVICE_FACILITY_ASSETS,
        SERVICE_MIN_CENTER_CLEARANCE_M,
        audit_and_attach_episode,
    )

try:
    from map_spatial_index import (
        LANE_HALF_WIDTH_M,
        PEDESTRIAN_ROAD_BUFFER_M,
        MapSpatialIndex,
    )
except (
    ModuleNotFoundError
):  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.map_spatial_index import (
        LANE_HALF_WIDTH_M,
        PEDESTRIAN_ROAD_BUFFER_M,
        MapSpatialIndex,
    )

try:
    from filter_render_ready_truth_for_capture import (
        filter_episode as filter_capture_visible_episode,
    )
except (
    ModuleNotFoundError
):  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.filter_render_ready_truth_for_capture import (
        filter_episode as filter_capture_visible_episode,
    )

try:
    from uav_global_flow.truth_integration import (
        DEFAULT_UAV_OUTPUT_DIR,
        UavGlobalFlowDataset,
        UavSegment,
        UavSelection,
        load_uav_global_flow_dataset,
        uav_pad_truth_entity_id,
    )
except (
    ModuleNotFoundError
):  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.uav_global_flow.truth_integration import (
        DEFAULT_UAV_OUTPUT_DIR,
        UavGlobalFlowDataset,
        UavSegment,
        UavSelection,
        load_uav_global_flow_dataset,
        uav_pad_truth_entity_id,
    )

try:
    from sumo_ground_flow.truth_integration import (
        DEFAULT_SUMO_OUTPUT_DIR,
        SumoTrafficDataset,
        VehicleSelection,
        VisibilityGeometry,
        load_sumo_traffic_dataset,
    )
    from sumo_ground_flow.explicit_vehicle_plan import (
        VEHICLE_SOURCE_POLICY,
        load_explicit_vehicle_plan,
        planned_script_controlled_source_vehicle_ids,
        planned_source_vehicle_ids,
        planned_vehicle_ids,
        validate_source_presence_manifest_contract,
    )
    from sumo_ground_flow.road_signal_context import RoadSignalContext
except (
    ModuleNotFoundError
):  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.sumo_ground_flow.truth_integration import (
        DEFAULT_SUMO_OUTPUT_DIR,
        SumoTrafficDataset,
        VehicleSelection,
        VisibilityGeometry,
        load_sumo_traffic_dataset,
    )
    from Dataset.tools.sumo_ground_flow.explicit_vehicle_plan import (
        VEHICLE_SOURCE_POLICY,
        load_explicit_vehicle_plan,
        planned_script_controlled_source_vehicle_ids,
        planned_source_vehicle_ids,
        planned_vehicle_ids,
        validate_source_presence_manifest_contract,
    )
    from Dataset.tools.sumo_ground_flow.road_signal_context import RoadSignalContext


DEFAULT_MAP_ID = "donghu_road_topo"
DEFAULT_SITE_ID = "site.intersection_a"
DEFAULT_ROI_ID = "roi.intersection_a.v1"
DEFAULT_TICK_HZ = FORMAL_TICK_HZ
CAPTURE_TICK_STEP = 5
DEFAULT_DURATION_TICKS = 900
DEFAULT_CAPTURE_FILTER_OUTPUT_ROOT = Path(
    "Dataset/render_ready_episodes_capture_filtered"
)
UAV_GLOBAL_FLOW_SOURCE = "uav_global_flow"
DYNAMIC_ENTITY_CATEGORIES = {"pedestrian", "vehicle", "uav"}
SERVICE_FACILITY_REPAIR_POLICY = "render_ready_service_facility_reanchor_v1"
PEDESTRIAN_VEHICLE_DYNAMIC_CLEARANCE_MIN_M = 4.0
TERMINAL_REALIZATION_TOLERANCE_M = 0.75
TERMINAL_REALIZATION_SPEED_MAX_MPS = float(
    get_governed_parameter_defaults()["aircraft_stationary_speed_threshold_mps"]
)
MOVE_REALIZATION_EPS_M = 0.25
MOVE_REALIZATION_SPEED_EPS_MPS = 0.1
RUNTIME_BOUNDARY_PADDING_M = 60.0
RUNTIME_SPATIAL_CROP_POLICY = "roi_polygon_expanded_60m_runtime_truth_crop_v1"
UAV_CAMERA_CAPTURE_ROI_POLICY = (
    "camera_only_while_uav_position_inside_capture_roi_polygon_v1"
)
SOURCE_VEHICLE_AUTHORITY_POLICY = (
    "sumo_only_vehicle_authority_replaces_source_vehicle_truth_v2"
)
SUMO_CANONICAL_VEHICLE_ASSET_POLICY = (
    "canonical_sumo_vehicle_asset_once_per_lifecycle_v1"
)
REQUIRED_SUMO_LIFECYCLE_PRESERVATION_POLICY = (
    "required_sumo_vehicle_preserve_source_lifecycle_v1"
)
RENDER_TRUTH_PROXIMITY_TRIGGER_REALIZATION_POLICY = (
    "render_truth_proximity_trigger_replaces_source_dispatch_v1"
)
RENDER_TRUTH_EVENT_FIRED_AFTER_CASCADE_POLICY = (
    "render_truth_event_fired_after_cascade_v1"
)
RENDER_TRUTH_REPLACED_VEHICLE_SNAPSHOT_POLICY = (
    "sumo_replaced_source_vehicle_source_snapshot_alignment_v1"
)

try:
    from render_ready_vehicle_lanes import VehicleLaneProjector
except (
    ModuleNotFoundError
):  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.render_ready_vehicle_lanes import VehicleLaneProjector


ENTITY_PROFILES: dict[str, dict[str, str]] = {
    "uav": {
        "entity_category": "uav",
        "entity_kind": "uav.drone",
        "proxy_template_id": "drone.quadrotor",
        "logical_asset_id": "uav.inspect.quad.v1",
        "mode": "scene_sync",
    },
    "vehicle": {
        "entity_category": "vehicle",
        "entity_kind": "vehicle.car",
        "proxy_template_id": "vehicle.sedan",
        "logical_asset_id": "vehicle.emergency.suv.v1",
        "mode": "scene_sync",
    },
    "pedestrian": {
        "entity_category": "pedestrian",
        "entity_kind": "pedestrian.person",
        "proxy_template_id": "human.walker",
        "logical_asset_id": "pedestrian.cityops.basic.v1",
        "mode": "pedestrian_managed",
    },
    "radio_tower": {
        "entity_category": "facility",
        "entity_kind": "facility.base_station",
        "proxy_template_id": "proxy.facility_base_station",
        "logical_asset_id": "facility.radio.base_tower.v1",
        "mode": "scene_sync",
    },
    "landing_pad": {
        "entity_category": "facility",
        "entity_kind": "facility.landing_pad",
        "proxy_template_id": "proxy.facility_landing_pad",
        "logical_asset_id": "facility.landing_pad.visible.v1",
        "mode": "scene_sync",
    },
    "traffic_light": {
        "entity_category": "traffic_light",
        "entity_kind": "traffic_light.signal",
        "proxy_template_id": "proxy.traffic_light_signal",
        "logical_asset_id": "prop.traffic_control.signal_light.v1",
        "mode": "scene_sync",
    },
    "charging_pile": {
        "entity_category": "facility",
        "entity_kind": "facility.charger",
        "proxy_template_id": "proxy.facility_charger",
        "logical_asset_id": "facility.charger.cityops.v1",
        "mode": "scene_sync",
    },
    "barrier": {
        "entity_category": "facility",
        "entity_kind": "facility.barrier",
        "proxy_template_id": "proxy.facility_barrier",
        "logical_asset_id": "facility.barrier.basic",
        "mode": "scene_sync",
    },
    "roadwork_barrier": {
        "entity_category": "prop",
        "entity_kind": "prop.roadwork_barrier",
        "proxy_template_id": "prop.roadwork.barrier.v1",
        "logical_asset_id": "prop.roadwork.barrier.v1",
        "mode": "scene_sync",
    },
    "traffic_cone": {
        "entity_category": "prop",
        "entity_kind": "prop.traffic_cone",
        "proxy_template_id": "prop.roadwork.traffic_cone.v1",
        "logical_asset_id": "prop.roadwork.traffic_cone.v1",
        "mode": "scene_sync",
    },
    "construction_fence": {
        "entity_category": "prop",
        "entity_kind": "prop.construction_fence",
        "proxy_template_id": "prop.roadwork.construction_fence.v1",
        "logical_asset_id": "prop.roadwork.construction_fence.v1",
        "mode": "scene_sync",
    },
    "police_tape": {
        "entity_category": "prop",
        "entity_kind": "prop.police_tape",
        "proxy_template_id": "prop.incident.police_tape.v1",
        "logical_asset_id": "prop.incident.police_tape.v1",
        "mode": "scene_sync",
    },
    "delivery_bag": {
        "entity_category": "prop",
        "entity_kind": "prop.delivery_bag",
        "proxy_template_id": "prop.service.delivery_bag.v1",
        "logical_asset_id": "prop.service.delivery_bag.v1",
        "mode": "scene_sync",
    },
    "beacon": {
        "entity_category": "facility",
        "entity_kind": "facility.beacon",
        "proxy_template_id": "proxy.traffic_light_signal",
        "logical_asset_id": "prop.traffic_control.signal_light.v1",
        "mode": "scene_sync",
    },
    "signal": {
        "entity_category": "traffic_light",
        "entity_kind": "traffic_light.signal",
        "proxy_template_id": "proxy.traffic_light_signal",
        "logical_asset_id": "prop.traffic_control.signal_light.v1",
        "mode": "scene_sync",
    },
    "hazmat": {
        "entity_category": "facility",
        "entity_kind": "facility.hazmat_proxy",
        "proxy_template_id": "proxy.hazmat_trigger_box",
        "logical_asset_id": "semantic.trigger_box.extent_12_10_15.v1",
        "mode": "scene_sync",
    },
    "hazard_trigger": {
        "entity_category": "facility",
        "entity_kind": "facility.hazmat_proxy",
        "proxy_template_id": "proxy.hazmat_trigger_box",
        "logical_asset_id": "semantic.trigger_box.extent_12_10_15.v1",
        "mode": "scene_sync",
    },
    "no_fly_zone": {
        "entity_category": "facility",
        "entity_kind": "facility.no_fly_zone",
        "proxy_template_id": "proxy.facility_observation_point",
        "logical_asset_id": "",
        "mode": "metadata_only",
    },
    "hazard_zone": {
        "entity_category": "facility",
        "entity_kind": "facility.hazard_zone",
        "proxy_template_id": "",
        "logical_asset_id": "trigger.hazard.generic.box.v1",
        "mode": "metadata_only",
    },
    "uav_corridor": {
        "entity_category": "airspace_corridor",
        "entity_kind": "airspace_corridor.uav_corridor",
        "proxy_template_id": "semantic.uav_corridor.segment.v1",
        "logical_asset_id": "semantic.uav_corridor.segment.v1",
        "mode": "metadata_only",
    },
}

ENTITY_PROFILES_BY_ASSET_ID: dict[str, dict[str, str]] = {
    "trigger.hazard.generic.box.v1": ENTITY_PROFILES["hazard_zone"],
    "facility.charging_pile.basic": ENTITY_PROFILES["charging_pile"],
    "facility.charger.cityops.v1": ENTITY_PROFILES["charging_pile"],
    "facility.landing_pad.visible.v1": ENTITY_PROFILES["landing_pad"],
    "facility.radio.base_tower.v1": ENTITY_PROFILES["radio_tower"],
    "facility.barrier.basic": ENTITY_PROFILES["barrier"],
    "prop.roadwork.barrier.v1": ENTITY_PROFILES["roadwork_barrier"],
    "prop.roadwork.construction_fence.v1": ENTITY_PROFILES["construction_fence"],
    "prop.roadwork.traffic_cone.v1": ENTITY_PROFILES["traffic_cone"],
    "prop.incident.police_tape.v1": ENTITY_PROFILES["police_tape"],
    "prop.service.delivery_bag.v1": ENTITY_PROFILES["delivery_bag"],
    "prop.traffic_control.signal_light.v1": ENTITY_PROFILES["traffic_light"],
    "semantic.trigger_box.extent_12_10_15.v1": ENTITY_PROFILES["hazard_trigger"],
    "semantic.trigger_box.extent_12_9_15.v1": ENTITY_PROFILES["hazard_trigger"],
    "semantic.trigger_box.extent_13_10_4.v1": ENTITY_PROFILES["hazard_trigger"],
    "semantic.trigger_box.extent_14_10_14.v1": ENTITY_PROFILES["hazard_trigger"],
}
ENTITY_CATEGORY_OVERRIDES: dict[str, str] = {
    "facility.landing_pad.visible.v1": "facility",
    "facility.charger.cityops.v1": "facility",
    "facility.radio.base_tower.v1": "facility",
    "facility.barrier.basic": "facility",
}
LOGICAL_ONLY_ASSET_IDS = {
    "semantic.landing_pad",
    "semantic.spawn_zone",
    "semantic.asset_anchor",
}
KNOWN_VEHICLE_LOGICAL_ASSETS = {
    "vehicle.emergency.ambulance.v1",
    "vehicle.emergency.police_suv.v1",
    "vehicle.emergency.suv.v1",
    "vehicle.ground.boxcar.v1",
    "vehicle.service.box.v1",
}
SUMO_TYPE_LOGICAL_ASSET_BY_TYPE_ID = {
    "aero_ambulance": "vehicle.emergency.ambulance.v1",
    "aero_delivery": "vehicle.service.box.v1",
    "aero_emergency": "vehicle.emergency.suv.v1",
    "aero_passenger": "vehicle.ground.boxcar.v1",
    "aero_police": "vehicle.emergency.police_suv.v1",
}
VISIBLE_FACILITY_ASSET_PREFIXES = (
    "facility.",
    "prop.traffic_control.",
    "prop.roadwork.",
    "semantic.trigger_box.",
)
PRESERVED_ENTITY_FIELDS = (
    "semantic_scope",
    "entity_kind",
    "task_id",
    "role",
    "state_sequence",
    "semantic_role",
    "background_role",
    "contract_scenario_id",
    "contract_inspect_uav",
    "capture_boundary",
    "capture_boundary_id",
    "capture_contract",
    "background_vehicle",
    "background_pedestrian",
    "ground_flow_contract",
    "event_actor_motion_contract",
    "contract_facility",
    "contract_logical_sidecar",
    "motion_contract",
    "uav_corridor_role",
    "uav_corridor",
    "inspect_capture_corridor",
    "uav_boundary_crossing_required",
    "uav_crosses_boundary",
    "inspect_observes_boundary",
    "mission_boundary_crossing",
    "pad_boundary_policy",
    "inspect_fov_coverage_required",
    "assigned_altitude_m",
    "inspect_altitude_code",
    "inspect_altitude_m",
    "min_path_length_m",
    "full_episode_presence",
    "observer_lifecycle",
    "ground_reference_z_m",
    "ground_reference_source",
    "lifecycle",
    "activation_tick",
    "deactivation_tick",
    "spawn_policy",
    "category",
    "route_waypoints_enu_m",
    "planned_route_waypoints_enu_m",
    "path_deviation_contract",
    "task_kind",
    "task_state",
    *RUNTIME_STATE_FIELDS,
)
PRESERVED_EVENT_FIELDS = (
    "task_id",
    "role",
    "state_sequence",
    "semantic_role",
    "background_role",
    "contract_scenario_id",
    "chain_id",
    "instance_id",
    "scope",
    "source_kind",
    "source_frame_id",
    "published_event_refs",
    "state_diff_refs",
    "intent",
    "intent_stage",
    "causal_chain_id",
    "causal_predecessor_intent",
    "target_roles",
    "capture_boundary_id",
    "uav_boundary_crossing_required",
    "inspect_fov_coverage_required",
    "pad_boundary_policy",
    "validation_event_type",
    "validation_reason",
    "validation_skip_checks",
)


def validate_runtime_state_patch(
    payload: Any,
    *,
    context: str = "runtime_state_patch",
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context}: runtime state patch must be a mapping")
    forbidden = sorted(set(forbidden_runtime_state_paths(payload, policy="engine_whole")))
    if forbidden:
        raise RuntimeError(
            f"{context}: runtime state payload contains forbidden semantic fields or paths: {forbidden}"
        )
    unknown = sorted(
        str(key) for key in payload if str(key) not in RUNTIME_STATE_FIELDS
    )
    if unknown:
        raise RuntimeError(f"{context}: unsupported runtime state families: {unknown}")
    unconsumed = unconsumed_runtime_state_paths(payload)
    if unconsumed:
        raise RuntimeError(
            f"{context}: runtime state payload contains fields that no formal state-to-predicate rule consumes: "
            f"{unconsumed}"
        )
    invalid_values = invalid_runtime_state_value_paths(payload)
    if invalid_values:
        raise RuntimeError(
            f"{context}: runtime state payload contains values outside the governed field contract: "
            f"{invalid_values}"
        )
    patch: dict[str, dict[str, Any]] = {}
    for family in RUNTIME_STATE_FIELDS:
        if family not in payload:
            continue
        family_value = payload[family]
        if not isinstance(family_value, Mapping):
            raise RuntimeError(f"{context}: {family} must be a mapping")
        if not family_value:
            raise RuntimeError(f"{context}: {family} must not be an empty family patch")
        patch[family] = copy.deepcopy(dict(family_value))
    if not patch:
        raise RuntimeError(
            f"{context}: runtime state patch must include at least one allowed state family"
        )
    return patch


def _runtime_state_patches_from_carrier(
    source: Mapping[str, Any],
    *,
    context: str,
) -> list[dict[str, dict[str, Any]]]:
    candidates: list[tuple[str, Any]] = []
    direct = {
        family: source[family] for family in RUNTIME_STATE_FIELDS if family in source
    }
    if direct:
        candidates.append(("top_level", direct))
    for nested_key in ("initial_state", "visual_state"):
        if nested_key not in source:
            continue
        nested = source.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        nested_direct = {
            family: nested[family]
            for family in RUNTIME_STATE_FIELDS
            if family in nested
        }
        if nested_direct:
            candidates.append((nested_key, nested_direct))
        if "runtime_state" in nested:
            candidates.append(
                (f"{nested_key}.runtime_state", nested.get("runtime_state"))
            )
    if "runtime_state" in source:
        candidates.append(("runtime_state", source.get("runtime_state")))
    return [
        validate_runtime_state_patch(payload, context=f"{context}.{location}")
        for location, payload in candidates
    ]


def runtime_state_fields_from_sources(
    *sources: Mapping[str, Any],
    context: str = "runtime_state_source",
) -> dict[str, dict[str, Any]]:
    preserved: dict[str, dict[str, Any]] = {}
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            continue
        patches = _runtime_state_patches_from_carrier(
            source,
            context=f"{context}[{source_index}]",
        )
        for patch in patches:
            for family, family_value in patch.items():
                if family not in preserved:
                    preserved[family] = copy.deepcopy(family_value)
    return preserved


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(
                    orjson.loads(stripped)
                    if orjson is not None
                    else json.loads(stripped.decode("utf-8"))
                )
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL row: {exc}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dumps_jsonl_bytes(payload: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def truth_entity_to_trajectory_row(
    frame: dict[str, Any], entity: dict[str, Any]
) -> dict[str, Any]:
    pose = dict(entity.get("truth_pose") or {})
    position = normalize_vector3(pose.get("position_enu_m"))
    velocity = normalize_vector3(pose.get("velocity_enu_mps"))
    annotations = dict(entity.get("annotations") or {})
    activity = dict(dict(annotations.get("state_facets") or {}).get("activity") or {})
    row: dict[str, Any] = {
        "tick": int(
            frame.get("tick")
            if frame.get("tick") is not None
            else frame.get("frame_seq", 0)
        ),
        "frame_id": frame.get("frame_id"),
        "sim_time_s": frame.get("sim_time_s"),
        "entity_id": entity.get("entity_id"),
        "label_class": entity.get("label_class"),
        "asset_id": entity.get("logical_asset_id") or entity.get("asset_id"),
        "entity_category": entity.get("entity_category"),
        "entity_kind": entity.get("entity_kind"),
        "entity_type": entity.get("entity_type"),
        "pos_enu": position,
        "vel_mps": velocity,
        "yaw_deg": dict(pose.get("rotation_deg") or {}).get("yaw_deg"),
        "state": entity.get("state") or annotations.get("activity_type"),
        "activity_type": annotations.get("activity_type"),
        "animation_hint": activity.get("animation_hint"),
        "posture": activity.get("posture"),
        "social_state": activity.get("social_state"),
        "source": entity.get("source"),
        "category": entity.get("entity_category"),
    }
    preserve_keys = (
        "semantic_scope",
        "entity_kind",
        "task_id",
        "role",
        "semantic_role",
        "background_role",
        "capture_boundary_id",
        "route_waypoints_enu_m",
        "planned_route_waypoints_enu_m",
        "path_deviation_contract",
        "background_vehicle",
        "background_pedestrian",
        "sumo_segment",
        "sumo_vehicle",
        "sumo_visibility",
        "uav_segment",
        "uav_global_flow",
        "uav_global_pad",
        "contract_facility",
        *RUNTIME_STATE_FIELDS,
    )
    runtime_fields = runtime_state_fields_from_sources(
        entity,
        context=f"truth_entity_to_trajectory_row[{entity.get('entity_id') or '<unknown>'}]",
    )
    for key in preserve_keys:
        if key in RUNTIME_STATE_FIELDS:
            if key in runtime_fields:
                row[key] = copy.deepcopy(runtime_fields[key])
        elif key in entity:
            row[key] = copy.deepcopy(entity[key])
    return {key: value for key, value in row.items() if value is not None}


def write_truth_trajectories(path: Path, truth_frames: Sequence[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as handle:
        for frame in truth_frames:
            for entity in frame.get("entities") or []:
                if not isinstance(entity, dict):
                    continue
                handle.write(
                    dumps_jsonl_bytes(truth_entity_to_trajectory_row(frame, entity))
                )
                count += 1
    return count


def is_sumo_vehicle_entity(entity: dict[str, Any]) -> bool:
    return (
        str(entity.get("entity_category") or entity.get("label_class") or "")
        == "vehicle"
        and str(entity.get("source") or "") == "sumo_traci"
    )


def is_capture_required_sumo_vehicle_entity(entity: dict[str, Any]) -> bool:
    if not is_sumo_vehicle_entity(entity):
        return False
    sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
    return bool(
        sumo_vehicle.get("semantic_vehicle")
        or sumo_vehicle.get("source_presence_required")
    )


def capture_required_sumo_vehicle_ids_from_roster(
    roster_entities: Sequence[dict[str, Any]],
) -> set[str]:
    vehicle_ids: set[str] = set()
    for entity in roster_entities:
        if not isinstance(entity, dict) or not is_capture_required_sumo_vehicle_entity(
            entity
        ):
            continue
        sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
        vehicle_id = str(
            entity.get("sumo_vehicle_id") or sumo_vehicle.get("vehicle_id") or ""
        )
        if vehicle_id:
            vehicle_ids.add(vehicle_id)
    return vehicle_ids


def assert_sumo_vehicle_logical_assets_stable(
    episode_id: str, truth_frames: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    assets_by_entity: dict[str, set[str]] = defaultdict(set)
    changes_by_entity: dict[str, list[tuple[int, str]]] = defaultdict(list)
    last_asset_by_entity: dict[str, str] = {}
    for frame in truth_frames:
        tick = int(frame.get("tick") or frame.get("frame_seq") or 0)
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict) or not is_sumo_vehicle_entity(entity):
                continue
            entity_id = str(entity.get("entity_id") or "")
            if not entity_id:
                continue
            asset = clean_vehicle_logical_asset(
                entity.get("logical_asset_id") or entity.get("asset_id")
            )
            assets_by_entity[entity_id].add(asset)
            if last_asset_by_entity.get(entity_id) != asset:
                changes_by_entity[entity_id].append((tick, asset))
                last_asset_by_entity[entity_id] = asset
    drift = {
        entity_id: {
            "assets": sorted(assets),
            "changes": changes_by_entity.get(entity_id, [])[:12],
        }
        for entity_id, assets in sorted(assets_by_entity.items())
        if len(assets) > 1
    }
    if drift:
        raise RuntimeError(
            f"{episode_id}: SUMO vehicle logical_asset_id drift detected: {drift}"
        )
    return {
        "policy": SUMO_CANONICAL_VEHICLE_ASSET_POLICY,
        "checked_sumo_vehicle_entity_count": len(assets_by_entity),
        "logical_asset_id_drift": 0,
    }


def source_sumo_vehicle_lifecycle_by_vehicle_id(
    *,
    sumo_dataset: SumoTrafficDataset,
    sumo_segment: Any,
    vehicle_ids: set[str],
    tick_hz: int,
) -> dict[str, dict[str, Any]]:
    lifecycle: dict[str, dict[str, Any]] = {}
    if not vehicle_ids:
        return lifecycle
    start_s = float(sumo_segment.segment_start_s)
    end_s = float(sumo_segment.segment_end_s)
    for frame in sumo_dataset.frames:
        time_s = float(frame.get("sim_time_s") or 0.0)
        if time_s < start_s - 1e-9 or time_s > end_s + 1e-9:
            continue
        local_tick = int(round((time_s - start_s) * float(tick_hz)))
        for vehicle in frame.get("vehicles") or []:
            vehicle_id = str(vehicle.get("vehicle_id") or "")
            if vehicle_id not in vehicle_ids:
                continue
            payload = lifecycle.setdefault(
                vehicle_id,
                {
                    "vehicle_id": vehicle_id,
                    "first_tick": local_tick,
                    "last_tick": local_tick,
                    "source_frame_count": 0,
                },
            )
            payload["first_tick"] = min(int(payload["first_tick"]), local_tick)
            payload["last_tick"] = max(int(payload["last_tick"]), local_tick)
            payload["source_frame_count"] = int(payload["source_frame_count"]) + 1
    return lifecycle


def truth_sumo_vehicle_lifecycle_by_vehicle_id(
    *,
    truth_frames: Sequence[dict[str, Any]],
    vehicle_ids: set[str],
) -> dict[str, dict[str, Any]]:
    lifecycle: dict[str, dict[str, Any]] = {}
    if not vehicle_ids:
        return lifecycle
    for frame in truth_frames:
        tick = int(frame.get("tick") or frame.get("frame_seq") or 0)
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict) or not is_sumo_vehicle_entity(entity):
                continue
            sumo_vehicle = dict(entity.get("sumo_vehicle") or {})
            vehicle_id = str(sumo_vehicle.get("vehicle_id") or "")
            if vehicle_id not in vehicle_ids:
                continue
            payload = lifecycle.setdefault(
                vehicle_id,
                {
                    "vehicle_id": vehicle_id,
                    "entity_id": str(entity.get("entity_id") or ""),
                    "first_tick": tick,
                    "last_tick": tick,
                    "truth_frame_count": 0,
                    "retained_outside_runtime_boundary_count": 0,
                },
            )
            payload["first_tick"] = min(int(payload["first_tick"]), tick)
            payload["last_tick"] = max(int(payload["last_tick"]), tick)
            payload["truth_frame_count"] = int(payload["truth_frame_count"]) + 1
            runtime_visibility = dict(entity.get("runtime_visibility") or {})
            if runtime_visibility.get("retained_outside_runtime_boundary"):
                payload["retained_outside_runtime_boundary_count"] = (
                    int(payload["retained_outside_runtime_boundary_count"]) + 1
                )
    return lifecycle


def assert_required_sumo_vehicle_lifecycle_preserved(
    *,
    episode_id: str,
    sumo_dataset: SumoTrafficDataset,
    sumo_segment: Any,
    required_vehicle_ids: set[str],
    truth_frames: Sequence[dict[str, Any]],
    tick_hz: int,
) -> dict[str, Any]:
    source_lifecycle = source_sumo_vehicle_lifecycle_by_vehicle_id(
        sumo_dataset=sumo_dataset,
        sumo_segment=sumo_segment,
        vehicle_ids=required_vehicle_ids,
        tick_hz=tick_hz,
    )
    truth_lifecycle = truth_sumo_vehicle_lifecycle_by_vehicle_id(
        truth_frames=truth_frames,
        vehicle_ids=required_vehicle_ids,
    )
    errors: list[str] = []
    for vehicle_id in sorted(required_vehicle_ids):
        source = source_lifecycle.get(vehicle_id)
        truth = truth_lifecycle.get(vehicle_id)
        if not source:
            errors.append(f"{vehicle_id}: missing source SUMO lifecycle")
            continue
        if not truth:
            errors.append(
                f"{vehicle_id}: missing truth lifecycle for source ticks "
                f"{source['first_tick']}..{source['last_tick']}"
            )
            continue
        if int(truth["first_tick"]) > int(source["first_tick"]) or int(
            truth["last_tick"]
        ) < int(source["last_tick"]):
            errors.append(
                f"{vehicle_id}: truth lifecycle {truth['first_tick']}..{truth['last_tick']} "
                f"does not cover source {source['first_tick']}..{source['last_tick']}"
            )
    if errors:
        raise RuntimeError(
            f"{episode_id}: required SUMO vehicle lifecycle was clipped: {errors[:20]}"
        )
    return {
        "policy": REQUIRED_SUMO_LIFECYCLE_PRESERVATION_POLICY,
        "required_vehicle_count": len(required_vehicle_ids),
        "source_lifecycle_by_vehicle_id": source_lifecycle,
        "truth_lifecycle_by_vehicle_id": truth_lifecycle,
        "clipped_required_vehicle_count": 0,
    }


def normalize_vector3(
    value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)
) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = list(default)
    return [
        float(values[0] if len(values) > 0 else default[0]),
        float(values[1] if len(values) > 1 else default[1]),
        float(values[2] if len(values) > 2 else default[2]),
    ]


def distance3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a[index]) - float(b[index])) ** 2 for index in range(3))
    )


def heading_deg_from_velocity(
    velocity_enu_mps: Sequence[float], fallback_deg: float = 0.0
) -> float:
    vx = float(velocity_enu_mps[0] if len(velocity_enu_mps) > 0 else 0.0)
    vy = float(velocity_enu_mps[1] if len(velocity_enu_mps) > 1 else 0.0)
    if abs(vx) <= 1e-6 and abs(vy) <= 1e-6:
        return fallback_deg
    return math.degrees(math.atan2(vy, vx))


def source_yaw_degrees(
    source: Mapping[str, Any], field: str, *, context: str
) -> float | None:
    """Read recorded yaw without treating zero as an absent measurement."""
    value = source.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{field}: yaw must be a finite number")
    yaw = float(value)
    if not math.isfinite(yaw):
        raise ValueError(f"{context}.{field}: yaw must be a finite number")
    return yaw


def scenario_initial_yaw_degrees(
    *,
    entity_id: str,
    first_row: Mapping[str, Any],
    source_entry: Mapping[str, Any],
    scene_entities: Mapping[str, dict[str, Any]],
) -> float:
    for source, field, context in (
        (first_row, "yaw_deg", f"trajectory:{entity_id}"),
        (source_entry, "initial_yaw_deg", f"roster:{entity_id}"),
    ):
        yaw = source_yaw_degrees(source, field, context=context)
        if yaw is not None:
            return yaw
    scene_entity = scene_entities.get(entity_id)
    if isinstance(scene_entity, dict):
        placement = scene_entity.get("placement")
        rotation = placement.get("rotation_deg") if isinstance(placement, dict) else None
        if isinstance(rotation, dict):
            yaw = source_yaw_degrees(
                rotation, "yaw_deg", context=f"scene:{entity_id}.placement.rotation_deg"
            )
            if yaw is not None:
                return yaw
    raise ValueError(f"{entity_id}: initial UAV yaw is unrecorded in trajectory, roster and scene placement")


def truth_pose(
    position_enu_m: Sequence[float], yaw_deg: float, velocity_enu_mps: Sequence[float]
) -> dict[str, Any]:
    return {
        "authority_mode": "authoritative_input",
        "authority_owner": "dataset_converter",
        "coordinate_contract_id": "coord.external_enu_m.v1",
        "position_enu_m": [round(float(value), 6) for value in position_enu_m[:3]],
        "rotation_deg": {
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "yaw_deg": round(float(yaw_deg), 6),
        },
        "velocity_enu_mps": [round(float(value), 6) for value in velocity_enu_mps[:3]],
    }


def _truth_entity_position(entity: dict[str, Any]) -> list[float]:
    pose = dict(entity.get("truth_pose") or {})
    return normalize_vector3(pose.get("position_enu_m"))


def _truth_entity_yaw(entity: dict[str, Any]) -> float:
    pose = dict(entity.get("truth_pose") or {})
    rotation = dict(pose.get("rotation_deg") or {})
    try:
        return float(rotation.get("yaw_deg") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _set_truth_entity_motion(
    entity: dict[str, Any], velocity: Sequence[float], yaw_deg: float
) -> None:
    pose = dict(entity.get("truth_pose") or {})
    rotation = dict(pose.get("rotation_deg") or {})
    rotation["yaw_deg"] = round(float(yaw_deg), 6)
    pose["rotation_deg"] = rotation
    pose["velocity_enu_mps"] = [round(float(value), 6) for value in list(velocity)[:3]]
    entity["truth_pose"] = pose
    annotations = entity.get("annotations")
    if isinstance(annotations, dict):
        speed_mps = math.sqrt(sum(float(value) ** 2 for value in list(velocity)[:3]))
        annotations["speed_mps"] = round(speed_mps, 4)


def align_dynamic_entity_motion_to_final_positions(
    truth_frames: list[dict[str, Any]], tick_hz: int
) -> None:
    tracks: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for frame_index, frame in enumerate(truth_frames):
        tick = int(frame.get("tick", frame_index))
        for entity in frame.get("entities") or []:
            category = str(
                entity.get("entity_category") or entity.get("label_class") or ""
            )
            if category not in {"pedestrian", "vehicle", "uav"}:
                continue
            if str(entity.get("source") or "") == "sumo_traci":
                continue
            entity_id = str(entity.get("entity_id") or "")
            if entity_id:
                tracks[entity_id].append((frame_index, tick, entity))

    for samples in tracks.values():
        if not samples:
            continue
        last_yaw = _truth_entity_yaw(samples[0][2])
        for index, (_, tick, entity) in enumerate(samples):
            recorded_global_yaw = entity.pop("_recorded_global_source_yaw_deg", None)
            position = _truth_entity_position(entity)
            if index > 0:
                _, previous_tick, previous_entity = samples[index - 1]
                previous_position = _truth_entity_position(previous_entity)
                dt_s = max(1e-6, float(tick - previous_tick) / float(tick_hz))
                delta = [position[i] - previous_position[i] for i in range(3)]
            elif len(samples) > 1:
                _, next_tick, next_entity = samples[index + 1]
                next_position = _truth_entity_position(next_entity)
                dt_s = max(1e-6, float(next_tick - tick) / float(tick_hz))
                delta = [next_position[i] - position[i] for i in range(3)]
            else:
                delta = [0.0, 0.0, 0.0]
                dt_s = 1.0 / float(tick_hz)

            velocity = [delta[i] / dt_s for i in range(3)]
            if math.hypot(velocity[0], velocity[1]) > 1e-5:
                last_yaw = math.degrees(math.atan2(velocity[1], velocity[0]))
            elif str(entity.get("source") or "") == UAV_GLOBAL_FLOW_SOURCE:
                recorded_yaw = source_yaw_degrees(
                    {"yaw_deg": recorded_global_yaw}, "yaw_deg",
                    context=f"global_uav:{entity['entity_id']}",
                )
                if recorded_yaw is not None:
                    last_yaw = recorded_yaw
            _set_truth_entity_motion(entity, velocity, last_yaw)


def assert_dynamic_roster_entities_have_truth(
    *,
    episode_id: str,
    roster_entities: Sequence[dict[str, Any]],
    truth_frames: Sequence[dict[str, Any]],
) -> None:
    dynamic_roster_ids = {
        str(entity.get("entity_id") or "")
        for entity in roster_entities
        if isinstance(entity, dict)
        and str(entity.get("entity_id") or "")
        and str(
            entity.get("entity_category") or entity.get("label_class") or ""
        ).lower()
        in DYNAMIC_ENTITY_CATEGORIES
    }
    truth_dynamic_ids: set[str] = set()
    for frame in truth_frames:
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            category = str(
                entity.get("entity_category") or entity.get("label_class") or ""
            ).lower()
            if category not in DYNAMIC_ENTITY_CATEGORIES:
                continue
            entity_id = str(entity.get("entity_id") or "")
            if entity_id:
                truth_dynamic_ids.add(entity_id)
    missing = sorted(dynamic_roster_ids - truth_dynamic_ids)
    if missing:
        preview = ", ".join(missing[:20])
        raise RuntimeError(
            f"{episode_id}: dynamic roster entities have no truth frame after runtime crop: {preview}"
        )


def _entity_asset_id(entity: dict[str, Any]) -> str:
    return str(
        entity.get("logical_asset_id")
        or entity.get("asset_id")
        or entity.get("proxy_template_id")
        or ""
    )


def _entity_position_for_repair(entity: dict[str, Any]) -> list[float] | None:
    truth_pose = entity.get("truth_pose")
    if isinstance(truth_pose, dict) and truth_pose.get("position_enu_m") is not None:
        return normalize_vector3(truth_pose.get("position_enu_m"))
    for key in (
        "initial_position_enu_m",
        "position_enu_m",
        "initial_pos_enu",
        "resolved_position_enu_m",
    ):
        if entity.get(key) is not None:
            return normalize_vector3(entity.get(key))
    placement = entity.get("placement")
    if isinstance(placement, dict):
        for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
            if placement.get(key) is not None:
                return normalize_vector3(placement.get(key))
    return None


def _update_position_fields(
    payload: dict[str, Any], position_enu_m: Sequence[float]
) -> None:
    position = [
        round(float(position_enu_m[0]), 6),
        round(float(position_enu_m[1]), 6),
        round(float(position_enu_m[2]), 6),
    ]
    for key in (
        "initial_position_enu_m",
        "position_enu_m",
        "initial_pos_enu",
        "resolved_position_enu_m",
    ):
        if key in payload:
            payload[key] = list(position)
    truth_pose_payload = payload.get("truth_pose")
    if isinstance(truth_pose_payload, dict) and "position_enu_m" in truth_pose_payload:
        truth_pose_payload["position_enu_m"] = list(position)
    placement = payload.get("placement")
    if isinstance(placement, dict):
        for key in ("resolved_position_enu_m", "position_enu_m", "center_enu_m"):
            if key in placement:
                placement[key] = list(position)


def _plan_service_facility_repair(
    spatial: MapSpatialIndex,
    *,
    entity_id: str,
    asset_id: str,
    desired_position_enu_m: Sequence[float],
) -> tuple[list[float], dict[str, Any]]:
    min_clearance_m = float(
        SERVICE_MIN_CENTER_CLEARANCE_M.get(asset_id, LANE_HALF_WIDTH_M + 4.5)
    )
    base_offset_m = max(1.2, min_clearance_m - LANE_HALF_WIDTH_M + 0.35)
    errors: list[str] = []
    for extra_m in (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 26.0):
        offset_m = round(base_offset_m + extra_m, 3)
        try:
            anchor = spatial.plan_sidewalk_anchor(
                desired_position_enu_m,
                offset_from_curb_m=offset_m,
                allow_green=True,
                placement_semantics="service_facility_repair",
            )
        except Exception as exc:
            errors.append(str(exc))
            continue
        candidate = list(anchor.position_enu_m)
        clearance_m = float(spatial.nearest_lane_clearance(candidate))
        point_errors = spatial.validation_errors_for_point(
            candidate,
            context=f"{entity_id} repaired service facility",
            allow_road=False,
            allow_green=True,
            road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
        )
        if point_errors or clearance_m + 1e-6 < min_clearance_m:
            errors.extend(point_errors[:2])
            if clearance_m + 1e-6 < min_clearance_m:
                errors.append(
                    f"lane clearance {clearance_m:.3f}m < {min_clearance_m:.3f}m"
                )
            continue
        repair = {
            "policy": SERVICE_FACILITY_REPAIR_POLICY,
            "entity_id": entity_id,
            "asset_id": asset_id,
            "from_position_enu_m": [
                round(float(value), 6) for value in list(desired_position_enu_m)[:3]
            ],
            "to_position_enu_m": [round(float(value), 6) for value in candidate[:3]],
            "road_clearance_m": round(clearance_m, 3),
            "required_clearance_m": round(min_clearance_m, 3),
            "anchor_edge_id": anchor.sample.edge_id,
            "anchor_lane_s_m": round(float(anchor.sample.s_m), 3),
            "offset_from_curb_m": round(float(anchor.offset_from_curb_m), 3),
            "resolved_lateral_from_center_m": round(
                float(anchor.resolved_lateral_from_center_m), 3
            ),
        }
        return candidate, repair

    for radius_m in (10.0, 14.0, 18.0, 24.0, 32.0, 44.0, 60.0, 80.0):
        for angle_index in range(24):
            angle = (2.0 * math.pi * float(angle_index)) / 24.0
            candidate = [
                float(desired_position_enu_m[0]) + math.cos(angle) * radius_m,
                float(desired_position_enu_m[1]) + math.sin(angle) * radius_m,
                float(
                    desired_position_enu_m[2]
                    if len(desired_position_enu_m) > 2
                    else 0.0
                ),
            ]
            clearance_m = float(spatial.nearest_lane_clearance(candidate))
            point_errors = spatial.validation_errors_for_point(
                candidate,
                context=f"{entity_id} repaired service facility open-space",
                allow_road=False,
                allow_green=True,
                road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
            )
            if point_errors or clearance_m + 1e-6 < min_clearance_m:
                continue
            nearest_sample = spatial.lanes.nearest(candidate)
            repair = {
                "policy": SERVICE_FACILITY_REPAIR_POLICY,
                "entity_id": entity_id,
                "asset_id": asset_id,
                "from_position_enu_m": [
                    round(float(value), 6) for value in list(desired_position_enu_m)[:3]
                ],
                "to_position_enu_m": [
                    round(float(value), 6) for value in candidate[:3]
                ],
                "road_clearance_m": round(clearance_m, 3),
                "required_clearance_m": round(min_clearance_m, 3),
                "anchor_edge_id": nearest_sample.edge_id,
                "anchor_lane_s_m": round(float(nearest_sample.s_m), 3),
                "offset_from_curb_m": round(float(radius_m), 3),
                "resolved_lateral_from_center_m": round(float(clearance_m), 3),
                "fallback": "open_space_radial_search",
            }
            return candidate, repair
    raise RuntimeError(
        f"Unable to repair service facility placement for {entity_id} asset={asset_id}; "
        f"first_errors={errors[:6]}"
    )


def repair_service_facility_placements(
    *,
    all_roster_entities: list[dict[str, Any]],
    truth_frames: list[dict[str, Any]],
    project_root: Path,
) -> list[dict[str, Any]]:
    spatial = MapSpatialIndex.default(project_root)
    repairs: list[dict[str, Any]] = []
    repaired_positions: dict[str, list[float]] = {}
    for roster_entry in all_roster_entities:
        entity_id = str(roster_entry.get("entity_id") or "")
        asset_id = _entity_asset_id(roster_entry)
        category = str(
            roster_entry.get("entity_category") or roster_entry.get("category") or ""
        )
        if (
            not entity_id
            or asset_id not in SERVICE_FACILITY_ASSETS
            or category != "facility"
        ):
            continue
        position = _entity_position_for_repair(roster_entry)
        if position is None:
            continue
        min_clearance_m = float(
            SERVICE_MIN_CENTER_CLEARANCE_M.get(asset_id, LANE_HALF_WIDTH_M + 4.5)
        )
        point_errors = spatial.validation_errors_for_point(
            position,
            context=f"{entity_id} service facility",
            allow_road=False,
            allow_green=True,
            road_buffer_m=PEDESTRIAN_ROAD_BUFFER_M,
        )
        clearance_m = float(spatial.nearest_lane_clearance(position))
        if not point_errors and clearance_m + 1e-6 >= min_clearance_m:
            continue
        try:
            repaired_position, repair = _plan_service_facility_repair(
                spatial,
                entity_id=entity_id,
                asset_id=asset_id,
                desired_position_enu_m=position,
            )
        except RuntimeError as exc:
            repair = {
                "policy": SERVICE_FACILITY_REPAIR_POLICY,
                "entity_id": entity_id,
                "asset_id": asset_id,
                "from_position_enu_m": [
                    round(float(value), 6) for value in list(position)[:3]
                ],
                "to_position_enu_m": [
                    round(float(value), 6) for value in list(position)[:3]
                ],
                "status": "unrepaired_nonblocking_service_facility",
                "reason": str(exc)[:500],
                "required_clearance_m": round(min_clearance_m, 3),
                "road_clearance_m": round(clearance_m, 3),
            }
            placement = roster_entry.setdefault("placement", {})
            if isinstance(placement, dict):
                placement["scene_occupancy_authority"] = (
                    "low_altitude_scene_occupancy_authority_v1"
                )
                placement["placement_semantics"] = (
                    "service_facility_nonblocking_unrepaired"
                )
                placement["blocking"] = False
                placement["collision_policy"] = "no_collision_visual_context_only"
                placement["road_clearance_m"] = repair["road_clearance_m"]
            roster_entry["scene_occupancy_repair"] = copy.deepcopy(repair)
            repairs.append(repair)
            continue
        _update_position_fields(roster_entry, repaired_position)
        placement = roster_entry.setdefault("placement", {})
        if isinstance(placement, dict):
            placement["scene_occupancy_authority"] = (
                "low_altitude_scene_occupancy_authority_v1"
            )
            placement["placement_semantics"] = "service_facility_repair"
            placement["road_clearance_m"] = repair["road_clearance_m"]
            placement["anchor_edge_id"] = repair["anchor_edge_id"]
            placement["anchor_lane_s_m"] = repair["anchor_lane_s_m"]
            placement["offset_from_curb_m"] = repair["offset_from_curb_m"]
            placement["resolved_lateral_from_center_m"] = repair[
                "resolved_lateral_from_center_m"
            ]
        roster_entry["scene_occupancy_repair"] = copy.deepcopy(repair)
        repaired_positions[entity_id] = repaired_position
        repairs.append(repair)

    if not repaired_positions:
        return repairs

    for frame in truth_frames:
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("entity_id") or "")
            repaired_position = repaired_positions.get(entity_id)
            if repaired_position is None:
                continue
            _update_position_fields(entity, repaired_position)
            entity["scene_occupancy_repair"] = copy.deepcopy(
                next((item for item in repairs if item["entity_id"] == entity_id), {})
            )
    return repairs


def render_presence(roi_id: str) -> dict[str, Any]:
    return {
        "global_roster": True,
        "offstage": False,
        "offstage_reason": "none",
        "roi_membership": [roi_id],
        "submission_state": "submit_to_ue",
        "visibility_state": "visible",
    }


def logical_only_profile(label_class: str) -> dict[str, str]:
    return {
        "entity_category": "other",
        "entity_kind": f"other.{label_class or 'logical'}",
        "proxy_template_id": "",
        "logical_asset_id": "",
        "mode": "logical_contract",
    }


def profile_for_label(label_class: str) -> dict[str, str]:
    profile = ENTITY_PROFILES.get(label_class)
    if not profile:
        raise RuntimeError(
            f"Missing deterministic entity profile for label_class={label_class or '<empty>'}"
        )
    return dict(profile)


def profile_for_entity(
    source_entry: dict[str, Any], first_row: dict[str, Any]
) -> dict[str, str]:
    label_class = str(
        source_entry.get("label_class") or first_row.get("label_class") or ""
    )
    asset_id = str(
        source_entry.get("logical_asset_id")
        or source_entry.get("asset_id")
        or first_row.get("logical_asset_id")
        or first_row.get("asset_id")
        or ""
    ).strip()
    if asset_id in LOGICAL_ONLY_ASSET_IDS:
        profile = logical_only_profile(label_class)
        profile["entity_category"] = str(
            source_entry.get("category") or first_row.get("category") or "facility"
        )
        profile["entity_kind"] = (
            f"{profile['entity_category']}.{label_class or 'logical'}"
        )
        profile["logical_asset_id"] = asset_id
        return profile
    if asset_id in ENTITY_PROFILES_BY_ASSET_ID:
        return dict(ENTITY_PROFILES_BY_ASSET_ID[asset_id])
    profile = profile_for_label(label_class)
    if asset_id in ENTITY_CATEGORY_OVERRIDES:
        category = ENTITY_CATEGORY_OVERRIDES[asset_id]
        profile["entity_category"] = category
        profile["entity_kind"] = f"{category}.{label_class or 'logical'}"
        if asset_id.startswith("facility."):
            profile["mode"] = "scene_sync"
    if (
        asset_id
        and asset_id.startswith(VISIBLE_FACILITY_ASSET_PREFIXES)
        and str(profile.get("mode") or "") != "scene_sync"
    ):
        raise RuntimeError(
            "Missing deterministic visual facility profile: "
            f"entity_id={source_entry.get('entity_id') or first_row.get('entity_id') or '<unknown>'} "
            f"label_class={label_class or '<empty>'} logical_asset_id={asset_id}"
        )
    return profile


def logical_asset_for(source_entry: dict[str, Any], profile: dict[str, str]) -> str:
    asset_id = str(
        source_entry.get("logical_asset_id") or source_entry.get("asset_id") or ""
    ).strip()
    if asset_id and asset_id.lower() not in {"unknown", "none", "null"}:
        return asset_id
    return str(profile.get("logical_asset_id") or "")


def value_is_present(value: Any) -> bool:
    return value is not None and value != ""


def preserved_fields_from(
    *sources: dict[str, Any],
    context: str = "preserved_entity_fields",
) -> dict[str, Any]:
    preserved: dict[str, Any] = {}
    runtime_fields = runtime_state_fields_from_sources(*sources, context=context)
    for field in PRESERVED_ENTITY_FIELDS:
        if field in RUNTIME_STATE_FIELDS:
            if field in runtime_fields:
                preserved[field] = copy.deepcopy(runtime_fields[field])
            continue
        for source in sources:
            if field in source and value_is_present(source[field]):
                preserved[field] = copy.deepcopy(source[field])
                break
            nested_state = source.get("initial_state")
            if (
                isinstance(nested_state, dict)
                and field in nested_state
                and value_is_present(nested_state[field])
            ):
                preserved[field] = copy.deepcopy(nested_state[field])
                break
            nested_visual_state = source.get("visual_state")
            if (
                isinstance(nested_visual_state, dict)
                and field in nested_visual_state
                and value_is_present(nested_visual_state[field])
            ):
                preserved[field] = copy.deepcopy(nested_visual_state[field])
                break
            for nested_key in (
                "background_vehicle",
                "background_pedestrian",
                "capture_contract",
                "contract_facility",
                "contract_inspect_uav",
                "contract_logical_sidecar",
                "inspect_capture_corridor",
                "motion_contract",
                "uav_corridor",
            ):
                nested = source.get(nested_key)
                if (
                    isinstance(nested, dict)
                    and field in nested
                    and value_is_present(nested[field])
                ):
                    preserved[field] = copy.deepcopy(nested[field])
                    break
            if field in preserved:
                break
    return preserved


def preserved_event_fields(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row.get("payload") or {})
    metadata = dict(row.get("metadata") or {})
    preserved: dict[str, Any] = {}
    for field in PRESERVED_EVENT_FIELDS:
        if field in row and row[field] not in (None, ""):
            preserved[field] = copy.deepcopy(row[field])
        elif field in payload and payload[field] not in (None, ""):
            preserved[field] = copy.deepcopy(payload[field])
        elif field in metadata and metadata[field] not in (None, ""):
            preserved[field] = copy.deepcopy(metadata[field])
    for nested_key in ("scope", "render_hints"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            preserved[nested_key] = copy.deepcopy(nested)
    return preserved


def read_source_roster(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing required source global_entity_roster.json: {path}")
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate source roster key in {path}: {key}")
            value[key] = item
        return value

    with path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream, object_pairs_hook=unique_object)
    if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
        return {entity_id: dict(entity) for entity_id, entity in roster_index(payload).items()}
    if isinstance(payload, dict):
        indexed: dict[str, dict[str, Any]] = {}
        for entity_id, value in payload.items():
            if not entity_id or not isinstance(value, dict):
                raise ValueError(f"invalid source roster entity {entity_id!r} in {path}")
            declared_id = value.get("entity_id")
            if declared_id is not None and declared_id != entity_id:
                raise ValueError(f"source roster ID conflict in {path}: {entity_id} != {declared_id!r}")
            indexed[entity_id] = dict(value)
        return indexed
    raise ValueError(f"Unsupported roster format: {path}")


def read_source_sumo_semantic_vehicle_plan(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported SUMO semantic vehicle plan format: {path}")
    vehicles = payload.get("vehicles")
    if vehicles is None:
        return {}
    if not isinstance(vehicles, dict):
        raise ValueError(
            f"Unsupported SUMO semantic vehicle plan vehicles format: {path}"
        )
    return {
        str(entity_id): dict(value)
        for entity_id, value in vehicles.items()
        if isinstance(value, dict)
    }


def rows_by_entity(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        entity_id = str(row.get("entity_id") or "")
        if entity_id:
            grouped[entity_id].append(dict(row))
    for entity_rows in grouped.values():
        entity_rows.sort(key=lambda row: int(row.get("tick", 0)))
    return dict(grouped)


def load_trajectory_groups(path: Path) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_tick = 0
    if not path.exists():
        return {}, max_tick
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL row: {exc}"
                ) from exc
            try:
                tick = int(row.get("tick", 0) or 0)
            except (TypeError, ValueError):
                tick = 0
            max_tick = max(max_tick, tick)
            entity_id = str(row.get("entity_id") or "")
            if entity_id:
                grouped[entity_id].append(row)
    for entity_rows in grouped.values():
        entity_rows.sort(key=lambda row: int(row.get("tick", 0)))
    return dict(grouped), max_tick


def sample_row_at_tick(
    entity_rows: Sequence[dict[str, Any]], tick: int, tick_hz: int
) -> dict[str, Any]:
    if not entity_rows:
        raise ValueError("cannot sample an empty entity trajectory")
    ticks = [int(row.get("tick", 0)) for row in entity_rows]
    index = bisect.bisect_right(ticks, tick)
    if index <= 0:
        row = dict(entity_rows[0])
        row["tick"] = tick
        return row
    if index >= len(entity_rows):
        row = dict(entity_rows[-1])
        row["tick"] = tick
        return row

    prev_row = entity_rows[index - 1]
    next_row = entity_rows[index]
    prev_tick = int(prev_row.get("tick", 0))
    next_tick = int(next_row.get("tick", prev_tick))
    span = max(1, next_tick - prev_tick)
    alpha = (tick - prev_tick) / float(span)

    prev_pos = normalize_vector3(prev_row.get("pos_enu"))
    next_pos = normalize_vector3(next_row.get("pos_enu"))
    position = [prev_pos[i] + (next_pos[i] - prev_pos[i]) * alpha for i in range(3)]

    prev_vel = normalize_vector3(prev_row.get("vel_mps"))
    next_vel = normalize_vector3(next_row.get("vel_mps"))
    velocity = [prev_vel[i] + (next_vel[i] - prev_vel[i]) * alpha for i in range(3)]
    if all(abs(value) <= 1e-6 for value in velocity) and span > 0:
        dt_s = span / float(max(1, tick_hz))
        velocity = [(next_pos[i] - prev_pos[i]) / dt_s for i in range(3)]

    row = dict(prev_row if alpha < 0.5 else next_row)
    row["tick"] = tick
    row["pos_enu"] = position
    row["vel_mps"] = velocity
    return row


def is_background_ground_flow_actor(entry: dict[str, Any]) -> bool:
    role = str(entry.get("role") or "")
    background_role = str(entry.get("background_role") or "")
    category = str(entry.get("entity_category") or entry.get("label_class") or "")
    contract = dict(entry.get("ground_flow_contract") or {})
    return (
        bool(contract)
        and category in {"pedestrian", "vehicle"}
        and (
            role in {"semantic_background_pedestrian", "semantic_background_vehicle"}
            or background_role == "semantic_context"
        )
    )


def visible_until_tick_for_ground_flow(
    entity_rows: Sequence[dict[str, Any]],
    tick_hz: int,
    source_entry: dict[str, Any] | None = None,
) -> int:
    max_tick = max((int(row.get("tick", 0)) for row in entity_rows), default=-1)
    contract = dict((source_entry or {}).get("ground_flow_contract") or {})
    if bool(contract.get("required")):
        try:
            contract_end_tick = int(contract.get("route_duration_ticks") or max_tick)
        except (TypeError, ValueError):
            contract_end_tick = max_tick
        if contract_end_tick >= 0:
            return min(max_tick, contract_end_tick)

    last_moving_tick = -1
    previous_position: list[float] | None = None
    for row in entity_rows:
        tick = int(row.get("tick", 0))
        position = normalize_vector3(row.get("pos_enu"))
        velocity = normalize_vector3(row.get("vel_mps"))
        moved_by_position = (
            previous_position is not None
            and math.hypot(
                position[0] - previous_position[0], position[1] - previous_position[1]
            )
            > 0.02
        )
        moving_by_velocity = math.hypot(velocity[0], velocity[1]) > 0.05
        if moved_by_position or moving_by_velocity:
            last_moving_tick = tick
        previous_position = position
    if last_moving_tick < 0:
        return -1
    return min(max_tick, last_moving_tick + max(1, int(tick_hz)))


def first_moving_tick(
    entity_rows: Sequence[dict[str, Any]], *, min_displacement_m: float = 0.5
) -> int:
    initial_position: list[float] | None = None
    for row in entity_rows:
        tick = int(row.get("tick", 0))
        position = normalize_vector3(row.get("pos_enu"))
        if initial_position is None:
            initial_position = position
        if (
            math.sqrt(
                sum(
                    (position[index] - initial_position[index]) ** 2
                    for index in range(3)
                )
            )
            > min_displacement_m
        ):
            return tick
    return -1


def uav_requires_motion_before_visibility(roster_entry: Mapping[str, Any]) -> bool:
    """Full-episode inspect contracts remain visible during stationary pre-roll."""

    observer_lifecycle = roster_entry.get("observer_lifecycle")
    return not (
        roster_entry.get("full_episode_presence") is True
        or (
            isinstance(observer_lifecycle, Mapping)
            and observer_lifecycle.get("presence") == "episode_full_duration"
        )
    )


def normalized_position_for_render(row: dict[str, Any]) -> list[float]:
    return normalize_vector3(row.get("pos_enu"))


def source_position_for_runtime(row: dict[str, Any], category: str) -> list[float]:
    position = normalized_position_for_render(row)
    if str(category) == "vehicle":
        position = source_vehicle_truth_position(position)
    return position


def event_text(row: dict[str, Any]) -> str:
    payload = dict(row.get("payload") or {})
    parts = [
        row.get("topic"),
        row.get("source_event_id"),
        row.get("chain_id"),
        payload.get("title"),
        payload.get("category"),
    ]
    return " ".join(str(part or "") for part in parts).lower()


def target_ids_from_event(row: dict[str, Any]) -> list[str]:
    targets = row.get("target_ids")
    if isinstance(targets, list):
        return [str(value) for value in targets if str(value)]
    scope = dict(row.get("scope") or {})
    scoped = scope.get("entities")
    if isinstance(scoped, list):
        return [str(value) for value in scoped if str(value)]
    target = scope.get("target_id")
    return [str(target)] if target else []


def _event_id_from_trace(row: dict[str, Any], scenario_id: str) -> str:
    raw = str(
        row.get("source_event_id") or row.get("topic") or row.get("event_id") or ""
    )
    prefix = f"evt_{scenario_id}_"
    return raw[len(prefix) :] if raw.startswith(prefix) else raw


def _event_id_from_realization(row: dict[str, Any], scenario_id: str) -> str:
    raw = str(row.get("event_id") or row.get("topic") or "")
    prefix = f"evt_{scenario_id}_"
    return raw[len(prefix) :] if raw.startswith(prefix) else raw


def indexed_realizations(
    event_realization_rows: Sequence[dict[str, Any]],
    scenario_id: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in event_realization_rows:
        for key in (
            _event_id_from_realization(row, scenario_id),
            str(row.get("event_id") or ""),
            str(row.get("topic") or ""),
        ):
            if key:
                result[key] = row
    return result


def realization_int(row: dict[str, Any] | None, key: str, default: int) -> int:
    if not row:
        return default
    try:
        return int(row.get(key) if row.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def activity_for_sample(
    *,
    entity_id: str,
    category: str,
    state: str,
    row_activity_type: str = "",
    velocity_enu_mps: Sequence[float],
    semantic_idle_when_stationary: bool = False,
) -> str:
    # Generated trajectory activity is authoritative; speed and state only
    # determine activity when the source row has no explicit value.
    speed_xy = math.hypot(float(velocity_enu_mps[0]), float(velocity_enu_mps[1]))
    state_text = str(state or "").strip().lower()
    if category == "pedestrian":
        row_activity = str(row_activity_type or "").strip().lower()
        if (
            semantic_idle_when_stationary
            and speed_xy <= 0.15
            and row_activity
            in {"walking", "crossing", "evacuating", "texting_walk", "moving"}
        ):
            return (
                "phone_call"
                if sum(ord(ch) for ch in entity_id) % 2 == 0
                else "chatting"
            )
        if row_activity:
            return normalize_activity_type(row_activity, moving=speed_xy > 0.15)
    if category == "pedestrian":
        row_activity = state_text if state_text not in {"moving", "idle"} else ""
        if (
            semantic_idle_when_stationary
            and speed_xy <= 0.15
            and row_activity
            in {"walking", "crossing", "evacuating", "texting_walk", "moving"}
        ):
            return (
                "phone_call"
                if sum(ord(ch) for ch in entity_id) % 2 == 0
                else "chatting"
            )
        if row_activity:
            return normalize_activity_type(row_activity, moving=speed_xy > 0.15)
        return "walking" if speed_xy > 0.15 or state_text == "moving" else "waiting"
    if category == "uav":
        row_activity = str(row_activity_type or "").strip()
        if row_activity and row_activity not in {"moving", "idle"}:
            return row_activity
        if state_text and state_text not in {"moving", "idle"}:
            return state_text
        return "flight" if speed_xy > 0.1 or state_text == "moving" else "idle"
    if category == "vehicle":
        return "moving" if speed_xy > 0.15 or state_text == "moving" else "idle"
    return state_text or "idle"


def build_annotations(
    activity_type: str, row: dict[str, Any], category: str
) -> dict[str, Any]:
    speed_mps = math.sqrt(
        sum(float(value) ** 2 for value in normalize_vector3(row.get("vel_mps")))
    )
    if category == "pedestrian":
        annotations = activity_annotations(activity_type, speed_mps=speed_mps)
        annotations["state_facets"]["network"] = {
            "status": "nominal",
            "latency_ms": 0.0,
            "packet_loss": 0.0,
        }
        return annotations
    posture = "standing"
    animation_hint = activity_type
    annotations: dict[str, Any] = {
        "activity_type": activity_type,
        "speed_mps": round(speed_mps, 4),
        "state_facets": {
            "activity": {
                "activity_type": activity_type,
                "animation_hint": animation_hint,
                "posture": posture,
                "social_state": "solo",
            },
            "network": {
                "status": "nominal",
                "latency_ms": 0.0,
                "packet_loss": 0.0,
            },
        },
    }
    return annotations


def normalize_weather_row(row: dict[str, Any], tick: int) -> dict[str, Any]:
    unknown = set(row) - WEATHER_SOURCE_FIELDS
    if unknown:
        raise ValueError(f"weather tick {tick}: unknown source keys {sorted(unknown)}")
    for alias, canonical in WEATHER_ALIASES.items():
        if alias in row and canonical in row and row[alias] != row[canonical]:
            raise ValueError(f"weather tick {tick}: conflicting {alias} and {canonical}")

    def required_number(field: str) -> float:
        value = row.get(field)
        if value is None:
            for alias, canonical in WEATHER_ALIASES.items():
                if canonical == field and alias in row:
                    value = row[alias]
                    break
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"weather tick {tick}: missing or invalid {field}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"weather tick {tick}: non-finite {field}")
        return result

    for alias in WEATHER_ALIASES:
        if alias in row:
            required_number(alias)

    condition = row.get("condition")
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError(f"weather tick {tick}: missing or invalid condition")
    normalized = {
        "tick": int(tick),
        "condition": condition.strip().lower(),
        "rain": required_number("rain"),
        "wetness": required_number("wetness"),
        "fog_density": required_number("fog_density"),
        "wind_speed": required_number("wind_speed"),
        "visibility_m": required_number("visibility_m"),
    }
    if "dust" in row:
        normalized["dust"] = required_number("dust")
    for field in WEATHER_FRACTION_FIELDS:
        if not 0.0 <= normalized[field] <= 1.0:
            raise ValueError(f"weather tick {tick}: {field} must be within [0, 1]")
    # These are physical simulator-state channels, not authored event labels.
    # Keep them absent when the source generator did not provide them so the
    # semantic layer fails closed instead of fabricating a nominal value.
    optional_numeric_fields = (
        "wind_direction_deg",
        "temperature_c",
        "illumination_lux",
        "hazard_concentration_ppm",
        "hazard_radius_m",
    )
    for field in optional_numeric_fields:
        if field in row:
            normalized[field] = required_number(field)
    if "hazard_source_active" in row:
        if not isinstance(row["hazard_source_active"], bool):
            raise ValueError(f"weather tick {tick}: invalid hazard_source_active")
        normalized["hazard_source_active"] = row["hazard_source_active"]
    if "temperature_source" in row:
        source = row["temperature_source"]
        if not isinstance(source, str) or not source.strip() or "temperature_c" not in row:
            raise ValueError(f"weather tick {tick}: invalid temperature_source")
        normalized["temperature_source"] = source
    return normalized


def expand_weather_rows(
    source_rows: Sequence[dict[str, Any]], ticks: Sequence[int]
) -> list[dict[str, Any]]:
    if not ticks:
        return []
    if not source_rows:
        raise ValueError("source weather rows are missing")
    rows_by_tick: dict[int, dict[str, Any]] = {}
    for row in source_rows:
        row_tick = row.get("tick")
        if isinstance(row_tick, bool) or not isinstance(row_tick, int):
            raise ValueError(f"source weather row has invalid tick: {row_tick!r}")
        if row_tick in rows_by_tick:
            raise ValueError(f"duplicate source weather tick: {row_tick}")
        rows_by_tick[row_tick] = dict(row)
    result: list[dict[str, Any]] = []
    for tick in ticks:
        if tick not in rows_by_tick:
            raise ValueError(f"source weather row is missing at tick {tick}")
        row = rows_by_tick[tick]
        result.append(normalize_weather_row(row, tick))
    return result


def build_dynamic_labels(
    event_rows: Sequence[dict[str, Any]],
    episode_id: str,
    *,
    scenario_id: str = "",
    event_realization_rows: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    realization_by_id = indexed_realizations(event_realization_rows, scenario_id)
    for index, row in enumerate(event_rows):
        trace_tick = int(row.get("tick", row.get("activated_tick", 0)) or 0)
        realization = realization_by_id.get(_event_id_from_trace(row, scenario_id))
        tick = realization_int(realization, "result_tick", trace_tick)
        evidence_tick = realization_int(realization, "evidence_tick", tick)
        label = {
            "schema_name": "dynamic_label",
            "schema_version": "v1",
            "episode_id": episode_id,
            "label_id": str(
                row.get("sample_id") or row.get("instance_id") or f"label_{index:04d}"
            ),
            "tick": tick,
            "frame_id": str(row.get("frame_id") or f"tick:{tick}"),
            "dispatch_tick": trace_tick,
            "result_tick": tick,
            "evidence_tick": evidence_tick,
            "source_event_id": str(row.get("source_event_id") or ""),
            "topic": str(row.get("topic") or ""),
            "semantic_class": str(row.get("semantic_class") or "state_event"),
            "target_ids": target_ids_from_event(row),
            "render_hints": dict(row.get("render_hints") or {}),
            "payload": dict(row.get("payload") or {}),
        }
        label.update(preserved_event_fields(row))
        labels.append(label)
    return labels


def ceil_to_capture_tick(tick: int, step: int = CAPTURE_TICK_STEP) -> int:
    return int(math.ceil(float(tick) / float(step)) * step)


def truth_samples_by_entity(
    truth_frames: Sequence[dict[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    samples: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for frame in truth_frames:
        try:
            tick = int(
                frame.get("tick")
                if frame.get("tick") is not None
                else frame.get("frame_seq", 0)
            )
        except (TypeError, ValueError):
            continue
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("entity_id") or "")
            if entity_id:
                samples[entity_id][tick] = entity
    return samples


def truth_entity_speed_mps(entity: dict[str, Any] | None) -> float:
    if not entity:
        return 0.0
    pose = dict(entity.get("truth_pose") or {})
    velocity = normalize_vector3(pose.get("velocity_enu_mps"))
    return math.sqrt(sum(float(value) ** 2 for value in velocity))


def render_truth_sample_summary(
    entity: dict[str, Any] | None, tick: int
) -> dict[str, Any]:
    if not entity:
        return {"present": False}
    annotations = dict(entity.get("annotations") or {})
    return {
        "present": True,
        "tick": int(tick),
        "position_enu_m": dict(entity.get("truth_pose") or {}).get("position_enu_m"),
        "velocity_enu_mps": dict(entity.get("truth_pose") or {}).get(
            "velocity_enu_mps"
        ),
        "state": entity.get("state"),
        "activity_type": annotations.get("activity_type")
        or dict(dict(annotations.get("state_facets") or {}).get("activity") or {}).get(
            "activity_type"
        ),
        "label_class": entity.get("label_class"),
        "entity_category": entity.get("entity_category"),
        "source": entity.get("source"),
    }


def first_render_motion_tick(
    samples: dict[int, dict[str, Any]],
    tick: int,
    baseline_pos: Sequence[float] | None,
) -> int | None:
    if baseline_pos is None:
        for candidate in sorted(samples):
            if candidate >= tick:
                baseline_pos = _truth_entity_position(samples[candidate])
                break
    if baseline_pos is None:
        return None
    for candidate in sorted(samples):
        if candidate < tick:
            continue
        entity = samples[candidate]
        if (
            distance3(_truth_entity_position(entity), baseline_pos)
            > MOVE_REALIZATION_EPS_M
        ):
            return candidate
        if truth_entity_speed_mps(entity) > MOVE_REALIZATION_SPEED_EPS_MPS:
            return candidate
    return None


def first_render_capture_motion_tick(
    samples: dict[int, dict[str, Any]],
    tick: int,
    baseline_pos: Sequence[float] | None,
) -> int | None:
    motion_tick = first_render_motion_tick(samples, tick, baseline_pos)
    if motion_tick is None:
        return None
    for candidate in sorted(samples):
        if candidate >= motion_tick and candidate % CAPTURE_TICK_STEP == 0:
            return candidate
    return None


def render_terminal_reached_tick(
    samples: dict[int, dict[str, Any]],
    tick: int,
    terminal: Sequence[float],
) -> int | None:
    terminal_pos = normalize_vector3(terminal)
    for candidate in sorted(samples):
        if candidate < tick:
            continue
        if (
            distance3(_truth_entity_position(samples[candidate]), terminal_pos)
            <= TERMINAL_REALIZATION_TOLERANCE_M
            and truth_entity_speed_mps(samples[candidate])
            <= TERMINAL_REALIZATION_SPEED_MAX_MPS
        ):
            return candidate
    return None


def first_grid_tick_at_or_after(tick: int, step: int = CAPTURE_TICK_STEP) -> int:
    """Return the smallest capture-grid tick >= tick (interpreter dispatch rule)."""
    tick = max(0, int(tick))
    if step <= 1:
        return tick
    remainder = tick % step
    if remainder == 0:
        return tick
    return tick + (step - remainder)


def render_proximity_condition_at_tick(
    samples_by_entity: dict[str, dict[int, dict[str, Any]]],
    entity_a: str,
    entity_b: str,
    tick: int,
    *,
    distance_m: float,
    operator: str = "lte",
    metric: str = "3d",
) -> bool:
    """Evaluate one entity_proximity trigger sample against render truth.

    Mirrors Plugins/SumoImporter/Scripts/donghu_core/event_script_interpreter.py
    `_eval_proximity_trigger`: a missing render entity makes the condition
    false (the SUMO authority does not place the vehicle at that tick).
    """
    a = samples_by_entity.get(entity_a, {}).get(tick)
    b = samples_by_entity.get(entity_b, {}).get(tick)
    if a is None or b is None:
        return False
    pos_a = _truth_entity_position(a)
    pos_b = _truth_entity_position(b)
    metric_key = str(metric or "xy").casefold()
    if metric_key == "xy_plus_z":
        horizontal = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
        vertical = abs(pos_a[2] - pos_b[2])
        horizontal_limit = float(distance_m)
        vertical_limit = float(distance_m)
        if vertical_limit <= 0:
            return horizontal <= horizontal_limit
        return horizontal <= horizontal_limit and vertical <= vertical_limit
    if metric_key == "3d":
        dist = distance3(pos_a, pos_b)
    else:
        dist = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
    op = str(operator or "lte").casefold()
    if op == "lt":
        return dist < distance_m
    if op == "lte":
        return dist <= distance_m
    if op == "gt":
        return dist > distance_m
    if op == "gte":
        return dist >= distance_m
    return False


def render_truth_proximity_fire_tick(
    samples_by_entity: dict[str, dict[int, dict[str, Any]]],
    trigger: Mapping[str, Any],
    *,
    duration_ticks: int = DEFAULT_DURATION_TICKS,
    grid_step: int = CAPTURE_TICK_STEP,
) -> int | None:
    """Re-time an entity_proximity trigger from render truth.

    Replicates the event interpreter contract: the raw distance condition must
    hold for ``min_true_ticks`` consecutive ticks and the event may only
    dispatch on the capture grid. Returns the first such grid tick, or None
    when render truth never satisfies the trigger.
    """
    entity_a = str(trigger.get("entity_a") or "")
    entity_b = str(trigger.get("entity_b") or "")
    if not entity_a or not entity_b:
        return None
    distance_m = float(trigger.get("distance_m") or 0.0)
    operator = str(trigger.get("operator") or "lte")
    metric = str(trigger.get("metric") or "xy")
    min_true = max(1, int(trigger.get("min_true_ticks") or 1))
    consecutive = 0
    for tick in range(0, max(0, duration_ticks) + 1):
        if render_proximity_condition_at_tick(
            samples_by_entity,
            entity_a,
            entity_b,
            tick,
            distance_m=distance_m,
            operator=operator,
            metric=metric,
        ):
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= min_true and tick % grid_step == 0:
            return tick
    return None


def render_truth_retimed_dispatch_ticks(
    event_realization_rows: Sequence[dict[str, Any]],
    samples_by_entity: dict[str, dict[int, dict[str, Any]]],
    event_script: Mapping[str, Any] | None,
    sumo_replaced_source_vehicle_ids: set[str],
    *,
    episode_id: str,
    duration_ticks: int = DEFAULT_DURATION_TICKS,
    grid_step: int = CAPTURE_TICK_STEP,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Compute render-truth dispatch ticks for events tied to SUMO-replaced
    vehicles.

    * ``entity_proximity`` triggers whose pair involves a SUMO-replaced source
      vehicle are re-timed from render truth (``render_truth_proximity_fire_tick``).
    * ``event_fired_after`` descendants cascade by the same dispatch shift.

    Returns ``(payloads_by_event_id, retimed_dispatch_by_event_id)``. A source
    event whose render truth never satisfies the trigger is contradictory and
    aborts reconciliation.
    """
    if not event_script or not sumo_replaced_source_vehicle_ids:
        return {}, {}
    triggers = {
        str(trigger.get("trigger_id") or ""): trigger
        for trigger in event_script.get("triggers") or []
        if isinstance(trigger, dict) and trigger.get("trigger_id")
    }
    event_to_trigger_ref = {
        str(event.get("event_id") or ""): str(event.get("trigger_ref") or "")
        for event in event_script.get("events") or []
        if isinstance(event, dict) and event.get("event_id")
    }
    retimed: dict[str, int] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for row in event_realization_rows:
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        trigger_ref = event_to_trigger_ref.get(event_id, "")
        trigger = triggers.get(trigger_ref)
        if not isinstance(trigger, dict):
            continue
        old_dispatch = realization_int(row, "dispatch_tick", 0)
        trigger_type = str(trigger.get("type") or "")
        if trigger_type == "entity_proximity":
            involved = {
                str(trigger.get("entity_a") or ""),
                str(trigger.get("entity_b") or ""),
            }
            if not (involved & set(sumo_replaced_source_vehicle_ids)):
                continue
            new_dispatch = render_truth_proximity_fire_tick(
                samples_by_entity,
                trigger,
                duration_ticks=duration_ticks,
                grid_step=grid_step,
            )
            if new_dispatch is None:
                raise ValueError(
                    "render truth contradicts source entity_proximity event: "
                    f"episode_id={episode_id!r} event_id={event_id!r} "
                    f"trigger_id={trigger_ref!r} "
                    f"window_ticks=0..{max(0, int(duration_ticks))}; "
                    "render truth never satisfies the trigger"
                )
            payloads[event_id] = {
                "event_id": event_id,
                "trigger_id": trigger_ref,
                "policy": RENDER_TRUTH_PROXIMITY_TRIGGER_REALIZATION_POLICY,
                "source_dispatch_tick": int(old_dispatch),
                "render_dispatch_tick": int(new_dispatch),
            }
            retimed[event_id] = int(new_dispatch)
        elif trigger_type == "event_fired_after":
            parent_id = str(trigger.get("event_id") or "")
            parent_new = retimed.get(parent_id)
            if parent_new is None:
                continue
            delay_ticks = max(0, int(trigger.get("delay_ticks") or 0))
            new_dispatch = min(
                duration_ticks,
                first_grid_tick_at_or_after(
                    parent_new + delay_ticks, grid_step
                ),
            )
            payloads[event_id] = {
                "event_id": event_id,
                "trigger_id": trigger_ref,
                "policy": RENDER_TRUTH_EVENT_FIRED_AFTER_CASCADE_POLICY,
                "parent_event_id": parent_id,
                "source_dispatch_tick": int(old_dispatch),
                "render_dispatch_tick": int(new_dispatch),
            }
            retimed[event_id] = int(new_dispatch)
    return payloads, retimed


def _shift_action_tick(
    action: dict[str, Any], key: str, delta: int, duration_ticks: int
) -> None:
    """Shift one action tick field by ``delta``, clamped to [0, duration]."""
    if delta == 0:
        return
    value = action.get(key)
    if value is None:
        return
    try:
        shifted = int(value) + delta
    except (TypeError, ValueError):
        return
    action[key] = max(0, min(duration_ticks, shifted))


def align_replaced_vehicle_source_snapshots(
    row: dict[str, Any],
    samples_by_entity: dict[str, dict[int, dict[str, Any]]],
    sumo_replaced_source_vehicle_ids: set[str],
) -> None:
    """Rewrite source_truth_snapshots positions of SUMO-replaced vehicles with
    render truth so no tick carries two positions for the same entity.

    The event realization is consumed after SUMO authority replacement, so the
    authoritative position of a replaced vehicle is the SUMO render truth, not
    the script-level source trajectory.
    """
    if not sumo_replaced_source_vehicle_ids:
        return
    snapshots = row.get("source_truth_snapshots_by_tick")
    if not isinstance(snapshots, dict):
        return
    for tick_key, tick_snapshot in snapshots.items():
        if not isinstance(tick_snapshot, dict):
            continue
        try:
            tick = int(tick_key)
        except (TypeError, ValueError):
            continue
        for entity_id, snapshot in tick_snapshot.items():
            if entity_id not in sumo_replaced_source_vehicle_ids:
                continue
            if not isinstance(snapshot, dict):
                continue
            render_entity = samples_by_entity.get(entity_id, {}).get(tick)
            if render_entity is None:
                # SUMO lifecycle does not place the vehicle at this tick; the
                # authoritative snapshot is absence, not the script position.
                tick_snapshot[entity_id] = {
                    **snapshot,
                    "present": False,
                    "position_enu_m": None,
                    "velocity_enu_mps": None,
                    "state": None,
                    "activity_type": None,
                    "source": "sumo_traci",
                    "sumo_authority_replaced": True,
                    "snapshot_policy": RENDER_TRUTH_REPLACED_VEHICLE_SNAPSHOT_POLICY,
                }
                continue
            summary = render_truth_sample_summary(render_entity, tick)
            tick_snapshot[entity_id] = {
                **snapshot,
                "present": True,
                "tick": tick,
                "position_enu_m": summary.get("position_enu_m"),
                "velocity_enu_mps": summary.get("velocity_enu_mps"),
                "state": summary.get("state"),
                "activity_type": summary.get("activity_type"),
                "label_class": summary.get("label_class"),
                "entity_category": summary.get("entity_category"),
                "source": "sumo_traci",
                "sumo_authority_replaced": True,
                "snapshot_policy": RENDER_TRUTH_REPLACED_VEHICLE_SNAPSHOT_POLICY,
            }


def reconcile_event_realizations_to_render_truth(
    event_realization_rows: Sequence[dict[str, Any]],
    truth_frames: Sequence[dict[str, Any]],
    *,
    event_script: Mapping[str, Any] | None = None,
    sumo_replaced_source_vehicle_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    samples_by_entity = truth_samples_by_entity(truth_frames)
    replaced_ids = set(sumo_replaced_source_vehicle_ids or set())
    retimed_payloads, retimed_ticks = render_truth_retimed_dispatch_ticks(
        event_realization_rows,
        samples_by_entity,
        event_script,
        replaced_ids,
        episode_id=str(truth_frames[0]["episode_id"]),
    )
    reconciled_rows: list[dict[str, Any]] = []
    for row in event_realization_rows:
        updated = copy.deepcopy(row)
        dispatch_tick = realization_int(updated, "dispatch_tick", 0)
        event_id = str(updated.get("event_id") or "")
        retimed_payload = retimed_payloads.get(event_id)
        retimed_dispatch = retimed_ticks.get(event_id)
        retime_delta = 0
        if retimed_dispatch is not None and retimed_dispatch != dispatch_tick:
            retime_delta = int(retimed_dispatch) - int(dispatch_tick)
            dispatch_tick = int(retimed_dispatch)
            updated["dispatch_tick"] = dispatch_tick
            for action in updated.get("action_realizations") or []:
                if not isinstance(action, dict):
                    continue
                _shift_action_tick(
                    action, "dispatch_tick", retime_delta, DEFAULT_DURATION_TICKS
                )
                _shift_action_tick(
                    action, "scheduled_tick", retime_delta, DEFAULT_DURATION_TICKS
                )
                if str(action.get("action_type") or "") != "move_entity":
                    _shift_action_tick(
                        action, "result_tick", retime_delta, DEFAULT_DURATION_TICKS
                    )
                    _shift_action_tick(
                        action, "evidence_tick", retime_delta, DEFAULT_DURATION_TICKS
                    )
        action_result_ticks: list[int] = []
        action_evidence_ticks: list[int] = []
        changes: list[dict[str, Any]] = []
        for action in updated.get("action_realizations") or []:
            if not isinstance(action, dict):
                continue
            entity_id = str(action.get("entity_id") or "")
            if not entity_id:
                continue
            samples = samples_by_entity.get(entity_id, {})
            action_dispatch = realization_int(action, "dispatch_tick", dispatch_tick)
            if str(action.get("action_type") or "") == "move_entity":
                baseline = (
                    _truth_entity_position(samples[action_dispatch])
                    if action_dispatch in samples
                    else None
                )
                motion_tick = first_render_motion_tick(
                    samples, action_dispatch, baseline
                )
                capture_motion_tick = first_render_capture_motion_tick(
                    samples, action_dispatch, baseline
                )
                if motion_tick is not None:
                    action["first_motion_tick"] = int(motion_tick)
                if capture_motion_tick is not None:
                    action["first_capture_motion_tick"] = int(capture_motion_tick)
                terminal = action.get("terminal_enu_m")
                terminal_required = bool(action.get("terminal_required"))
                if (
                    terminal_required
                    and isinstance(terminal, Sequence)
                    and not isinstance(terminal, (str, bytes))
                ):
                    terminal_tick = render_terminal_reached_tick(
                        samples, action_dispatch, terminal
                    )
                    if terminal_tick is not None:
                        old_result = action.get("result_tick")
                        old_terminal = action.get("terminal_tick")
                        action["terminal_tick"] = int(terminal_tick)
                        action["result_tick"] = int(terminal_tick)
                        action["evidence_tick"] = min(
                            DEFAULT_DURATION_TICKS,
                            ceil_to_capture_tick(int(terminal_tick)),
                        )
                        if (
                            old_result != action["result_tick"]
                            or old_terminal != action["terminal_tick"]
                        ):
                            changes.append(
                                {
                                    "action_id": action.get("action_id"),
                                    "entity_id": entity_id,
                                    "old_result_tick": old_result,
                                    "old_terminal_tick": old_terminal,
                                    "render_result_tick": action["result_tick"],
                                    "render_terminal_tick": action["terminal_tick"],
                                }
                            )
                elif motion_tick is not None:
                    action["result_tick"] = int(motion_tick)
                    action["evidence_tick"] = min(
                        DEFAULT_DURATION_TICKS, ceil_to_capture_tick(int(motion_tick))
                    )
            if action.get("result_tick") is not None:
                action_result_ticks.append(int(action["result_tick"]))
            if action.get("evidence_tick") is not None:
                action_evidence_ticks.append(int(action["evidence_tick"]))
        if action_result_ticks:
            updated["result_tick"] = max(action_result_ticks)
        if action_evidence_ticks:
            updated["evidence_tick"] = min(
                DEFAULT_DURATION_TICKS, max(action_evidence_ticks)
            )
        if (
            retimed_payload is not None
            and not action_result_ticks
            and not action_evidence_ticks
        ):
            # Events whose actions are all metadata-only (e.g. capture_screenshot
            # with an empty entity_id) are skipped by the action loop; their
            # result/evidence must follow the render-truth dispatch tick.
            updated["result_tick"] = dispatch_tick
            updated["evidence_tick"] = dispatch_tick
        evidence_tick = realization_int(
            updated,
            "evidence_tick",
            realization_int(updated, "result_tick", dispatch_tick),
        )
        updated["before_tick"] = max(0, evidence_tick - CAPTURE_TICK_STEP)
        updated["after_tick"] = min(
            DEFAULT_DURATION_TICKS, evidence_tick + CAPTURE_TICK_STEP
        )
        snapshot_ticks = [
            int(updated["before_tick"]),
            evidence_tick,
            int(updated["after_tick"]),
        ]
        target_ids = [
            str(entity_id)
            for entity_id in updated.get("target_ids") or []
            if str(entity_id)
        ]
        render_snapshots = {
            str(tick): {
                entity_id: render_truth_sample_summary(
                    samples_by_entity.get(entity_id, {}).get(tick), tick
                )
                for entity_id in target_ids
            }
            for tick in snapshot_ticks
        }
        updated["render_truth_snapshots_by_tick"] = render_snapshots
        if retimed_payload is not None:
            # The event moved to a render-truth dispatch window; rebuild the
            # source snapshot window at the new ticks from render truth so the
            # realization never carries a pre-SUMO position inside its window.
            source_snapshots = copy.deepcopy(render_snapshots)
            for tick_snapshot in source_snapshots.values():
                for entity_id, snapshot in tick_snapshot.items():
                    if (
                        entity_id in replaced_ids
                        and isinstance(snapshot, dict)
                        and snapshot.get("present")
                    ):
                        snapshot["sumo_authority_replaced"] = True
                        snapshot["snapshot_policy"] = (
                            RENDER_TRUTH_REPLACED_VEHICLE_SNAPSHOT_POLICY
                        )
            updated["source_truth_snapshots_by_tick"] = source_snapshots
        if changes:
            updated["render_truth_reconciliation"] = {
                "policy": "render_authoritative_event_realization_after_sumo_uav_replacement_v1",
                "changes": changes,
            }
            updated["basis"] = (
                "render_ready truth_frames after SUMO/UAV authority reconciliation"
            )
        if retimed_payload is not None:
            updated["render_truth_trigger_reconciliation"] = retimed_payload
            updated["basis"] = (
                "render_ready truth_frames after SUMO authority trigger reconciliation"
            )
        align_replaced_vehicle_source_snapshots(
            updated, samples_by_entity, replaced_ids
        )
        reconciled_rows.append(updated)
    return reconciled_rows


def repo_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace(
            "\\", "/"
        )
    except ValueError:
        return str(path.resolve())


def has_sumo_dataset_files(path: Path) -> bool:
    return all(
        (Path(path) / name).exists()
        for name in (
            "sumo_traffic_frames.jsonl",
            "sumo_traffic_manifest.json",
            "sumo_incident_plan.json",
        )
    )


def explicit_vehicle_ids_by_traffic_role(
    plan: dict[str, Any], traffic_role: str
) -> set[str]:
    return {
        str(vehicle.get("vehicle_id") or "")
        for vehicle in plan.get("vehicles") or []
        if str(vehicle.get("vehicle_id") or "")
        and str(vehicle.get("traffic_role") or "") == str(traffic_role)
    }


def resolve_episode_sumo_output_dir(base_dir: Path, episode_id: str) -> Path:
    root = Path(base_dir)
    episode_dir = root if root.name == str(episode_id) else root / str(episode_id)
    if has_sumo_dataset_files(episode_dir):
        return episode_dir
    missing = [
        str(episode_dir / name)
        for name in (
            "sumo_traffic_frames.jsonl",
            "sumo_traffic_manifest.json",
            "sumo_incident_plan.json",
        )
        if not (episode_dir / name).exists()
    ]
    raise FileNotFoundError(
        f"{episode_id}: missing episode-local vehicle SUMO authority output under {episode_dir}; "
        f"run sumo_ground_flow.run_traffic --episode {episode_id} first. Missing: {missing}"
    )


def load_sumo_dataset_for_episode(
    *,
    base_dir: Path,
    episode_id: str,
    preloaded_sumo_dataset: SumoTrafficDataset | None,
) -> SumoTrafficDataset:
    resolved_dir = resolve_episode_sumo_output_dir(base_dir, episode_id)
    if resolved_dir == Path(base_dir) and preloaded_sumo_dataset is not None:
        return preloaded_sumo_dataset
    return load_sumo_traffic_dataset(resolved_dir, use_cache=False)


def resolve_manifest_path(
    value: Any, source_episode_dir: Path, project_root: Path
) -> Path | None:
    if not value:
        return None
    raw_path = Path(str(value))
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [project_root / raw_path, source_episode_dir / raw_path]
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_json_or_none(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return load_json(path)


def scene_setup_entities(scene_setup: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for entity in scene_setup.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("entity_id") or "").strip()
        if entity_id:
            entities[entity_id] = dict(entity)
    return entities


def inspect_route_from_source(
    inspect_entity: dict[str, Any], inspect_contract: dict[str, Any]
) -> list[list[float]]:
    route = None
    for key in ("loop_route_enu_m", "repaired_route_enu_m", "planned_route_enu_m"):
        candidate = inspect_contract.get(key)
        if isinstance(candidate, list):
            route = candidate
            break
    if not isinstance(route, list):
        raise RuntimeError(
            f"Missing inspect loop route on contract {inspect_entity.get('entity_id')!r}"
        )
    route_points = [
        point
        for point in route
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes))
    ]
    if len(route_points) < 4:
        raise RuntimeError(
            f"Inspect loop route is too short on entity {inspect_entity.get('entity_id')!r}"
        )
    normalized = [[float(value) for value in point[:3]] for point in route_points]
    first = normalized[0]
    last = normalized[-1]
    if (
        math.sqrt(
            (first[0] - last[0]) ** 2
            + (first[1] - last[1]) ** 2
            + (first[2] - last[2]) ** 2
        )
        > 1.0
    ):
        raise RuntimeError(
            f"Inspect route must be a closed loop on entity {inspect_entity.get('entity_id')!r}"
        )
    return normalized


def inspect_entity_from_source(entities: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inspect_entities = [
        entity
        for entity in entities.values()
        if str((entity.get("initial_state") or {}).get("role") or "") == "U_inspect"
        or str(entity.get("uav_corridor_role") or "") == "inspect_observer"
        or isinstance(entity.get("contract_inspect_uav"), dict)
    ]
    if len(inspect_entities) != 1:
        raise RuntimeError(
            f"Expected exactly one inspect entity, found {len(inspect_entities)}"
        )
    return inspect_entities[0]


def source_boundary_payload(event_script: dict[str, Any]) -> dict[str, Any]:
    params = dict(event_script.get("parameters") or {})
    contract = dict(params.get("semantic_event_contract") or {})
    boundary = {
        **dict(contract.get("capture_boundary") or {}),
        **dict(params.get("capture_boundary") or {}),
    }
    if not boundary:
        raise RuntimeError("Missing capture_boundary in source event_script.json")
    return boundary


def source_capture_boundary_id(event_script: dict[str, Any]) -> str:
    boundary = source_boundary_payload(event_script)
    boundary_id = str(
        boundary.get("boundary_id") or boundary.get("source_entity_id") or ""
    ).strip()
    if not boundary_id:
        raise RuntimeError(
            "Missing capture_boundary.boundary_id in source event_script.json"
        )
    return boundary_id


def source_pad_boundary_policy(event_script: dict[str, Any]) -> Any:
    params = dict(event_script.get("parameters") or {})
    contract = dict(params.get("semantic_event_contract") or {})
    policy = contract.get("pad_boundary_policy")
    if policy not in (None, ""):
        return policy
    boundary = source_boundary_payload(event_script)
    policy = boundary.get("pad_boundary_policy")
    if policy in (None, ""):
        raise RuntimeError("Missing pad_boundary_policy in source event_script.json")
    return policy


def point_in_polygon_xy(
    point: Sequence[float], polygon: Sequence[Sequence[float]]
) -> bool:
    x = float(point[0])
    y = float(point[1])
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = float(polygon[index][0]), float(polygon[index][1])
        x2, y2 = (
            float(polygon[(index + 1) % count][0]),
            float(polygon[(index + 1) % count][1]),
        )
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def distance_to_segment_xy(
    point: Sequence[float], a: Sequence[float], b: Sequence[float]
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def distance_to_polygon_xy(
    point: Sequence[float], polygon: Sequence[Sequence[float]]
) -> float:
    if not polygon:
        return float("inf")
    if point_in_polygon_xy(point, polygon):
        return 0.0
    return min(
        distance_to_segment_xy(
            point, polygon[index], polygon[(index + 1) % len(polygon)]
        )
        for index in range(len(polygon))
    )


def point_in_capture_roi_xy(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
    *,
    edge_epsilon_m: float = 1e-6,
) -> bool:
    return distance_to_polygon_xy(point, polygon) <= float(edge_epsilon_m)


def point_in_runtime_boundary_xy(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
    *,
    padding_m: float = RUNTIME_BOUNDARY_PADDING_M,
) -> bool:
    return distance_to_polygon_xy(point, polygon) <= float(padding_m)


def runtime_boundary_bbox_enu_m(
    source_contract: dict[str, Any], *, padding_m: float = RUNTIME_BOUNDARY_PADDING_M
) -> list[float]:
    polygon = source_contract.get("capture_boundary_polygon_enu_m") or []
    xs = [
        float(point[0])
        for point in polygon
        if isinstance(point, Sequence)
        and not isinstance(point, (str, bytes))
        and len(point) >= 2
    ]
    ys = [
        float(point[1])
        for point in polygon
        if isinstance(point, Sequence)
        and not isinstance(point, (str, bytes))
        and len(point) >= 2
    ]
    if not xs or not ys:
        return []
    return [
        round(min(xs) - float(padding_m), 3),
        round(min(ys) - float(padding_m), 3),
        round(max(xs) + float(padding_m), 3),
        round(max(ys) + float(padding_m), 3),
    ]


def segment_intersects_polygon_xy(
    a: Sequence[float], b: Sequence[float], polygon: Sequence[Sequence[float]]
) -> bool:
    def ccw(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> bool:
        return (float(p3[1]) - float(p1[1])) * (float(p2[0]) - float(p1[0])) > (
            float(p2[1]) - float(p1[1])
        ) * (float(p3[0]) - float(p1[0]))

    if point_in_polygon_xy(a, polygon) or point_in_polygon_xy(b, polygon):
        return True
    for index in range(len(polygon)):
        c = polygon[index]
        d = polygon[(index + 1) % len(polygon)]
        if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
            return True
    return False


def route_crosses_boundary(
    route: Sequence[Sequence[float]], polygon: Sequence[Sequence[float]]
) -> bool:
    if not route or not polygon:
        return False
    if any(point_in_polygon_xy(point, polygon) for point in route):
        return True
    return any(
        segment_intersects_polygon_xy(a, b, polygon) for a, b in zip(route, route[1:])
    )


def source_boundary_polygon(event_script: dict[str, Any]) -> list[list[float]]:
    boundary = source_boundary_payload(event_script)
    polygon = []
    for point in boundary.get("polygon_enu_m") or []:
        if (
            isinstance(point, Sequence)
            and not isinstance(point, (str, bytes))
            and len(point) >= 2
        ):
            polygon.append([float(point[0]), float(point[1])])
    if not polygon:
        raise RuntimeError("Source capture_boundary lacks polygon_enu_m")
    return polygon


def point_in_interest_xy(
    point: Sequence[float], visibility: VisibilityGeometry
) -> bool:
    return float(visibility.observation_distance_m(point)) <= float(
        visibility.padding_m
    )


def runtime_boundary_distance_m(
    position: Sequence[float], source_contract: dict[str, Any]
) -> float:
    return distance_to_polygon_xy(
        position, source_contract.get("capture_boundary_polygon_enu_m") or []
    )


def point_in_runtime_boundary_for_contract(
    position: Sequence[float], source_contract: dict[str, Any]
) -> bool:
    return (
        runtime_boundary_distance_m(position, source_contract)
        <= RUNTIME_BOUNDARY_PADDING_M
    )


def runtime_visibility_payload(
    position: Sequence[float], source_contract: dict[str, Any]
) -> dict[str, Any]:
    distance_m = runtime_boundary_distance_m(position, source_contract)
    return {
        "policy": RUNTIME_SPATIAL_CROP_POLICY,
        "expanded_boundary_padding_m": RUNTIME_BOUNDARY_PADDING_M,
        "distance_to_capture_boundary_m": round(float(distance_m), 6)
        if math.isfinite(float(distance_m))
        else None,
        "runtime_visible": bool(distance_m <= RUNTIME_BOUNDARY_PADDING_M),
    }


def uav_camera_capture_visibility_payload(
    position: Sequence[float],
    *,
    source_contract: dict[str, Any],
    interest_visibility: VisibilityGeometry,
) -> dict[str, Any]:
    capture_polygon = source_contract.get("capture_boundary_polygon_enu_m") or []
    roi_distance_m = distance_to_polygon_xy(position, capture_polygon)
    roi_capture_eligible = point_in_capture_roi_xy(position, capture_polygon)
    return {
        "inspect_observation_distance_m": round(
            float(interest_visibility.observation_distance_m(position)), 6
        ),
        "roi_capture_distance_m": round(float(roi_distance_m), 6)
        if math.isfinite(float(roi_distance_m))
        else None,
        "roi_capture_eligible": bool(roi_capture_eligible),
        "selected_for_capture_truth": bool(roi_capture_eligible),
        "capture_roi_policy": UAV_CAMERA_CAPTURE_ROI_POLICY,
        "camera_call_policy": "do_not_call_camera_when_roi_capture_eligible_is_false",
    }


def dist_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def source_uav_role(entity: dict[str, Any]) -> str:
    """UAV task role from the governed uav_role_classification table."""
    return uav_task_role(entity)


def trajectory_route_for_entity(
    grouped_rows: dict[str, list[dict[str, Any]]], entity_id: str
) -> list[list[float]]:
    route = []
    for row in grouped_rows.get(entity_id, []):
        position = normalize_vector3(row.get("pos_enu"))
        route.append(position)
    return route


def source_uavs_cross_boundary(
    scene_setup: dict[str, Any],
    grouped_rows: dict[str, list[dict[str, Any]]],
    polygon: list[list[float]],
) -> bool:
    uavs = [
        entity
        for entity in scene_setup.get("entities") or []
        if str(entity.get("logical_asset_id") or "").startswith("uav.")
        and source_uav_role(entity) != "inspect"
    ]
    if not uavs:
        return False
    for entity in uavs:
        entity_id = str(entity.get("entity_id") or "")
        route = trajectory_route_for_entity(grouped_rows, entity_id)
        if not route_crosses_boundary(route, polygon):
            return False
    return True


def inspect_frustum_observes_boundary(
    route: list[list[float]],
    polygon: list[list[float]],
    inspect_contract: dict[str, Any],
) -> bool:
    return inspect_route_observes_boundary(route, polygon, inspect_contract)


def capture_contract_from_source(
    *,
    source_episode_dir: Path,
    manifest: dict[str, Any],
    scene_setup: dict[str, Any],
    project_root: Path,
    scenario_id: str,
    grouped_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    event_script_path = resolve_manifest_path(
        manifest.get("source_event_script_path"), source_episode_dir, project_root
    )
    event_script = load_json_or_none(event_script_path)
    if not isinstance(event_script, dict):
        raise RuntimeError(
            f"Missing event_script.json for source episode {source_episode_dir}"
        )

    entities = scene_setup_entities(scene_setup)
    inspect_entity = inspect_entity_from_source(entities)
    inspect_contract = dict(inspect_entity.get("contract_inspect_uav") or {})
    params = dict(event_script.get("parameters") or {})
    semantic_contract = dict(params.get("semantic_event_contract") or {})
    if semantic_contract.get("uav_boundary_crossing_required") is not True:
        raise RuntimeError(
            "Source semantic_event_contract must require UAV boundary crossing"
        )
    if semantic_contract.get("inspect_fov_coverage_required") is not True:
        raise RuntimeError(
            "Source semantic_event_contract must require inspect FoV coverage"
        )
    boundary = source_boundary_payload(event_script)
    if str(boundary.get("geometry_source") or "") != "event_entity":
        raise RuntimeError(
            "Source capture_boundary.geometry_source must be event_entity"
        )
    boundary_entity_id = str(
        boundary.get("source_entity_id") or boundary.get("boundary_id") or ""
    )
    if boundary_entity_id and not any(
        str(entity.get("entity_id") or "") == boundary_entity_id
        for entity in entities.values()
    ):
        raise RuntimeError(
            f"Source capture boundary entity is not declared in scene_setup: {boundary_entity_id}"
        )
    polygon = source_boundary_polygon(event_script)
    inspect_route = inspect_route_from_source(inspect_entity, inspect_contract)
    inspect_altitude_m = float(
        inspect_contract.get("inspect_altitude_m") or inspect_route[0][2]
    )
    if any(
        abs(float(point[2]) - inspect_altitude_m) > 0.001 for point in inspect_route
    ):
        raise RuntimeError("Source inspect route must stay at fixed inspect altitude")
    uav_crosses_boundary = source_uavs_cross_boundary(
        scene_setup, grouped_rows, polygon
    )
    inspect_observes_boundary = inspect_frustum_observes_boundary(
        inspect_route, polygon, inspect_contract
    )

    return {
        "capture_boundary_id": source_capture_boundary_id(event_script),
        "capture_boundary_polygon_enu_m": [list(point) for point in polygon],
        "uav_crosses_boundary": uav_crosses_boundary,
        "inspect_observes_boundary": inspect_observes_boundary,
        "pad_boundary_policy": source_pad_boundary_policy(event_script),
        "inspect_entity_id": str(inspect_entity.get("entity_id") or "").strip(),
        "inspect_route_enu_m": inspect_route,
        "inspect_contract": inspect_contract,
    }


def truth_boundary_summary(
    entities: Sequence[dict[str, Any]],
    *,
    capture_boundary_id: str,
    uav_crosses_boundary: bool,
    inspect_observes_boundary: bool,
    pad_boundary_policy: Any,
) -> dict[str, Any]:
    motion_state: dict[str, str] = {}
    for entity in sorted(entities, key=lambda item: str(item.get("entity_id") or "")):
        entity_id = str(entity.get("entity_id") or "")
        state = str(
            entity.get("state")
            or entity.get("task_state")
            or entity.get("role")
            or "idle"
        )
        if str(entity.get("entity_category") or "").lower() == "uav":
            motion_state[entity_id] = state
    return {
        "capture_boundary_id": capture_boundary_id,
        "uav_crosses_boundary": bool(uav_crosses_boundary),
        "inspect_observes_boundary": bool(inspect_observes_boundary),
        "pad_boundary_policy": pad_boundary_policy,
        "entity_motion_state": motion_state,
    }


def logical_asset_for_global_uav(uav: dict[str, Any]) -> str:
    mission_type = str(uav.get("mission_type") or "").lower()
    semantic_role = str(uav.get("semantic_role") or "").lower()
    if "relay" in mission_type or "relay" in semantic_role:
        return "uav.relay.quad.v1"
    if "delivery" in mission_type:
        return "uav.delivery.quad.v1"
    return "uav.inspect.quad.v1"


def build_uav_pad_roster_entries(
    *,
    uav_dataset: UavGlobalFlowDataset,
    site_id: str,
    roi_id: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for pad in uav_dataset.pads:
        pad_id = str(pad.get("pad_id") or "")
        if not pad_id:
            continue
        position = normalize_vector3(pad.get("position_enu_m"))
        entry = {
            "entity_id": uav_pad_truth_entity_id(pad_id),
            "uav_pad_id": pad_id,
            "label_class": "landing_pad",
            "asset_id": "facility.landing_pad.visible.v1",
            "site_id": site_id,
            "roi_id": roi_id,
            "entity_category": "facility",
            "entity_kind": "facility.landing_pad",
            "entity_type": "facility.landing_pad",
            "proxy_template_id": "proxy.facility_landing_pad",
            "logical_asset_id": "facility.landing_pad.visible.v1",
            "semantic_scope": dict(pad.get("semantic_scope") or {}),
            "mode": "scene_sync",
            "initial_position_enu_m": position,
            "initial_yaw_deg": 0.0,
            "tags": [
                "facility",
                "landing_pad",
                "uav_global_flow",
                "charging_logistics_pad",
            ],
            "source": UAV_GLOBAL_FLOW_SOURCE,
            "background_role": "global_uav_pad",
            "uav_global_pad": {
                "policy": "donghu_global_uav_pad_network_v1",
                "pad_id": pad_id,
                "role": pad.get("role"),
                "grid_cell_id": pad.get("grid_cell_id"),
                "lane_edge_id": pad.get("lane_edge_id"),
                "lane_s_m": pad.get("lane_s_m"),
                "phase_origin_weights": pad.get("phase_origin_weights"),
                "phase_destination_weights": pad.get("phase_destination_weights"),
            },
            "motion_contract": {
                "policy": "static_facility_truth_frame_v1",
                "actor_kind": "facility",
                "source": UAV_GLOBAL_FLOW_SOURCE,
            },
        }
        validate_roster_facility_scope(entry)
        entries.append(entry)
    return entries


def uav_pad_truth_entity(
    *,
    roster_entry: dict[str, Any],
    tick: int,
    site_id: str,
    roi_id: str,
) -> dict[str, Any]:
    position = normalize_vector3(roster_entry.get("initial_position_enu_m"))
    return {
        "entity_id": roster_entry["entity_id"],
        "entity_category": "facility",
        "entity_kind": "facility.landing_pad",
        "entity_type": "facility.landing_pad",
        "label_class": "landing_pad",
        "site_id": site_id,
        "roi_id": roi_id,
        "proxy_template_id": "proxy.facility_landing_pad",
        "logical_asset_id": "facility.landing_pad.visible.v1",
        "semantic_scope": copy.deepcopy(roster_entry["semantic_scope"]),
        "tags": list(roster_entry.get("tags") or []),
        "truth_pose": truth_pose(position, 0.0, [0.0, 0.0, 0.0]),
        "render_presence": render_presence(roi_id),
        "annotations": build_annotations(
            "static", {"vel_mps": [0.0, 0.0, 0.0]}, "facility"
        ),
        "state_revision": int(tick) + 1,
        "visual_revision": 1,
        "source": UAV_GLOBAL_FLOW_SOURCE,
        "background_role": "global_uav_pad",
        "uav_global_pad": copy.deepcopy(roster_entry.get("uav_global_pad") or {}),
        "motion_contract": copy.deepcopy(roster_entry.get("motion_contract") or {}),
    }


def build_uav_roster_entries(
    *,
    uav_dataset: UavGlobalFlowDataset,
    uav_segment: UavSegment,
    uav_selection: UavSelection,
    site_id: str,
    roi_id: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    segment_payload = uav_segment.as_dict()
    pads_by_id = {
        str(pad.get("pad_id") or ""): pad
        for pad in uav_dataset.pads
        if str(pad.get("pad_id") or "")
    }
    global_ground_reference_raw = uav_dataset.task_plan.get("ground_reference_z_m")
    try:
        global_ground_reference_z_m = float(global_ground_reference_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "global UAV task plan must declare numeric ground_reference_z_m"
        ) from exc
    for uav_id in uav_selection.uav_ids:
        first_record = uav_dataset.first_uav_record_in_segment(uav_segment, uav_id)
        if first_record is None:
            continue
        task_id = str(
            first_record.get("task_id") or uav_selection.task_ids.get(uav_id, "")
        )
        task = dict(uav_dataset.tasks_by_id.get(task_id) or {})
        lifetime = uav_dataset.task_lifetime(task_id, uav_segment)
        planned_route = task.get("route_waypoints_enu_m")
        if not isinstance(planned_route, list) or len(planned_route) < 2:
            raise RuntimeError(
                f"global UAV task {task_id} lacks an explicit planned route"
            )
        origin_pad_id = str(
            first_record.get("origin_pad_id") or task.get("origin_pad_id") or ""
        )
        origin_pad = pads_by_id.get(origin_pad_id)
        origin_pad_position = (
            origin_pad.get("position_enu_m") if isinstance(origin_pad, dict) else None
        )
        if isinstance(origin_pad_position, list) and len(origin_pad_position) >= 3:
            ground_reference_z_m = float(origin_pad_position[2])
            ground_reference_source = "uav_task_plan_origin_pad_v1"
        else:
            ground_reference_z_m = global_ground_reference_z_m
            ground_reference_source = "uav_task_plan_global_ground_reference_v1"
        position = normalize_vector3(first_record.get("position_enu_m"))
        velocity = normalize_vector3(first_record.get("velocity_enu_mps"))
        yaw_deg = float(
            first_record.get("yaw_deg") or heading_deg_from_velocity(velocity)
        )
        logical_asset_id = logical_asset_for_global_uav(first_record)
        entries.append(
            {
                "entity_id": uav_selection.entity_ids[uav_id],
                "uav_id": uav_id,
                "task_id": task_id,
                "activation_tick": lifetime["first_active_grid_tick"],
                "world_lifetime": lifetime,
                "label_class": "uav",
                "asset_id": logical_asset_id,
                "site_id": site_id,
                "roi_id": roi_id,
                "entity_category": "uav",
                "entity_kind": "uav.drone",
                "entity_type": "uav.drone",
                "proxy_template_id": "drone.quadrotor",
                "logical_asset_id": logical_asset_id,
                "mode": "scene_sync",
                "initial_position_enu_m": position,
                "initial_yaw_deg": round(yaw_deg, 6),
                "tags": [
                    "uav",
                    "uav_global_flow",
                    str(first_record.get("mission_type") or "mission"),
                    str(first_record.get("semantic_role") or "global_uav"),
                ],
                "source": UAV_GLOBAL_FLOW_SOURCE,
                "background_role": "donghu_global_uav_flow",
                "semantic_role": first_record.get("semantic_role"),
                "uav_global_flow": {
                    "policy": "donghu_global_uav_flow_truth_replay_v1",
                    "uav_id": uav_id,
                    "task_id": task_id,
                    "mission_type": first_record.get("mission_type"),
                    "semantic_role": first_record.get("semantic_role"),
                    "corridor_family": first_record.get("corridor_family"),
                    "corridor_id": first_record.get("corridor_id"),
                    "altitude_layer_m": first_record.get("altitude_layer_m"),
                    "origin_pad_id": origin_pad_id,
                    "ground_reference_z_m": ground_reference_z_m,
                    "target_pad_id": first_record.get("target_pad_id"),
                    "target_cell_id": first_record.get("target_cell_id"),
                    "sample_period_s": uav_dataset.manifest.get("sample_period_s"),
                    "start_s": task["start_s"],
                    "end_s": task["end_s"],
                    "terminal_hold_s": task["terminal_hold_s"],
                },
                "motion_contract": {
                    "policy": "uav_global_flow_truth_replay_v1",
                    "actor_kind": "uav",
                    "route_source": "donghu_global_uav_flow",
                    "source": UAV_GLOBAL_FLOW_SOURCE,
                    "segment": segment_payload,
                    "looping": task.get("looping"),
                    "speed_mps": task.get("speed_mps"),
                    "route_length_m": task.get("route_length_m"),
                    "altitude_layer_m": first_record.get("altitude_layer_m"),
                },
                "uav_segment": segment_payload,
                "ground_reference_z_m": ground_reference_z_m,
                "ground_reference_source": ground_reference_source,
                "planned_route_waypoints_enu_m": [
                    normalize_vector3(point) for point in planned_route
                ],
            }
        )
    return entries


def uav_global_truth_entity(
    *,
    uav: dict[str, Any],
    roster_entry: dict[str, Any],
    tick: int,
    site_id: str,
    roi_id: str,
) -> dict[str, Any]:
    position = normalize_vector3(uav.get("position_enu_m"))
    velocity = normalize_vector3(uav.get("velocity_enu_mps"))
    yaw_deg = source_yaw_degrees(
        uav, "yaw_deg", context=f"global_uav:{roster_entry['entity_id']}"
    )
    recorded_yaw_deg = yaw_deg
    if yaw_deg is None:
        initial_yaw = source_yaw_degrees(
            roster_entry, "initial_yaw_deg", context=f"roster:{roster_entry['entity_id']}"
        )
        if initial_yaw is not None:
            yaw_deg = heading_deg_from_velocity(velocity, fallback_deg=initial_yaw)
        elif abs(velocity[0]) > 1e-6 or abs(velocity[1]) > 1e-6:
            yaw_deg = math.degrees(math.atan2(velocity[1], velocity[0]))
        else:
            raise ValueError(f"{roster_entry['entity_id']}: stationary global UAV yaw is unrecorded")
    speed = math.sqrt(sum(float(value) ** 2 for value in velocity))
    planned_route = roster_entry.get("planned_route_waypoints_enu_m") or []
    terminal_waypoint = (
        normalize_vector3(planned_route[-1])
        if isinstance(planned_route, list) and planned_route
        else None
    )
    terminal_contact = (
        terminal_waypoint is not None
        and speed <= 0.1
        and math.dist(position, terminal_waypoint) <= 0.5
    )
    activity_row = {
        "vel_mps": velocity,
        "state": "idle" if terminal_contact else "moving",
    }
    entity = {
        "entity_id": roster_entry["entity_id"],
        "entity_category": "uav",
        "entity_kind": "uav.drone",
        "entity_type": "uav.drone",
        "label_class": "uav",
        "site_id": site_id,
        "roi_id": roi_id,
        "proxy_template_id": "drone.quadrotor",
        "logical_asset_id": roster_entry.get("logical_asset_id")
        or logical_asset_for_global_uav(uav),
        "tags": list(roster_entry.get("tags") or []),
        "truth_pose": truth_pose(position, yaw_deg, velocity),
        "render_presence": render_presence(roi_id),
        "annotations": build_annotations(
            "landed" if terminal_contact else "flight", activity_row, "uav"
        ),
        "state_revision": int(tick) + 1,
        "visual_revision": 1,
        "source": UAV_GLOBAL_FLOW_SOURCE,
        "background_role": "donghu_global_uav_flow",
        "semantic_role": uav.get("semantic_role") or roster_entry.get("semantic_role"),
        "uav_global_flow": {
            **copy.deepcopy(roster_entry.get("uav_global_flow") or {}),
            "active_event_ids": list(uav.get("active_event_ids") or []),
            "source_prev_time_s": uav.get("source_prev_time_s"),
            "source_next_time_s": uav.get("source_next_time_s"),
            "source_alpha": uav.get("source_alpha"),
        },
        "motion_contract": copy.deepcopy(roster_entry.get("motion_contract") or {}),
        "uav_segment": copy.deepcopy(roster_entry.get("uav_segment") or {}),
        "planned_route_waypoints_enu_m": copy.deepcopy(
            roster_entry["planned_route_waypoints_enu_m"]
        ),
    }
    if uav.get("mission_type") not in (None, ""):
        entity["mission_type"] = uav.get("mission_type")
    if uav.get("task_id") not in (None, ""):
        entity["task_id"] = uav.get("task_id")
    if recorded_yaw_deg is not None:
        entity["_recorded_global_source_yaw_deg"] = recorded_yaw_deg
    return entity


def uav_frame_truth_payload(
    *,
    uav_dataset: UavGlobalFlowDataset,
    uav_segment: UavSegment,
    uav_selection: UavSelection,
    uav_sample: dict[str, Any],
    active_uav_count: int,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "segment": uav_segment.as_dict(),
        "absolute_time_s": uav_sample["absolute_time_s"],
        "source_prev_time_s": uav_sample["source_prev_time_s"],
        "source_next_time_s": uav_sample["source_next_time_s"],
        "source_alpha": uav_sample["source_alpha"],
        "source_uav_count": int(uav_sample.get("source_uav_count") or 0),
        "active_selected_uav_count": int(active_uav_count),
        "selected_uav_count": int(uav_selection.selected_count),
        "minimum_active_uavs_required": uav_dataset.manifest.get(
            "minimum_active_uavs_required"
        ),
        "baseline_active_uavs": uav_dataset.manifest.get("baseline_active_uavs"),
        "task_count": uav_dataset.manifest.get("task_count"),
    }


def projected_sumo_vehicle_position(
    vehicle: dict[str, Any], vehicle_lane_projector: VehicleLaneProjector
) -> list[float]:
    # Road-derived SUMO uses one physical lane per edge; truth_position_enu_m is already
    # the UE truth-space physical lane coordinate.  Applying the legacy traffic-bundle
    # lateral correction here would double-shift vehicles toward sidewalks.
    return normalize_vector3(vehicle.get("truth_position_enu_m"))


def road_derived_sumo_vehicle_yaw_deg(
    vehicle: dict[str, Any],
    position_enu_m: Sequence[float],
    vehicle_lane_projector: VehicleLaneProjector,
) -> float:
    del position_enu_m, vehicle_lane_projector
    velocity = normalize_vector3(vehicle.get("velocity_enu_mps"))
    # SUMO truth exports vehicle body center position and the body-center lane
    # tangent yaw.  Reprojecting yaw from traffic-bundle samples here can rotate
    # vehicles differently from the overlap guard and from SUMO truth itself.
    return float(vehicle.get("truth_yaw_deg") or heading_deg_from_velocity(velocity))


def source_vehicle_truth_position(position: Sequence[float]) -> list[float]:
    # Dataset source vehicles are authored with physical lane offsets already applied.
    return normalize_vector3(position)


def source_vehicle_authority_payload(
    *,
    enabled: bool,
    replaced_ids: set[str],
    source_roster: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts_by_role: Counter[str] = Counter()
    for entity_id in replaced_ids:
        entry = dict(source_roster.get(entity_id) or {})
        role = str(entry.get("role") or "<none>")
        counts_by_role[role] += 1
    return {
        "enabled": bool(enabled),
        "policy": SOURCE_VEHICLE_AUTHORITY_POLICY,
        "authority": "episode_local_sumo_traci_after_warmup",
        "reason": (
            "source episode vehicle truth is only a semantic script input; every ground vehicle rendered "
            "for capture must come from the episode-local SUMO run after the warm-up window"
        ),
        "replaced_source_vehicle_count": int(len(replaced_ids)),
        "replaced_source_vehicle_ids": sorted(replaced_ids),
        "replaced_source_vehicle_counts_by_role": dict(sorted(counts_by_role.items())),
        "event_reference_policy": "source vehicle event ids are validation references only; render-ready vehicle actors must come from SUMO",
    }


def runtime_spatial_crop_payload(source_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": True,
        "policy": RUNTIME_SPATIAL_CROP_POLICY,
        "capture_boundary_id": source_contract["capture_boundary_id"],
        "expanded_boundary_padding_m": RUNTIME_BOUNDARY_PADDING_M,
        "bbox_enu_m": runtime_boundary_bbox_enu_m(source_contract),
        "dynamic_frame_rule": "write_only_ticks_inside_expanded_runtime_boundary",
        "uav_camera_rule": UAV_CAMERA_CAPTURE_ROI_POLICY,
    }


def source_entity_visible_at_tick(
    *,
    roster_entry: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
    tick: int,
    tick_hz: int,
    source_contract: dict[str, Any],
    visible_until_tick_by_background_ground_flow: dict[str, int],
    first_moving_tick_by_local_uav: dict[str, int],
) -> bool:
    entity_id = str(roster_entry["entity_id"])
    activation_tick = int(roster_entry.get("activation_tick") or 0)
    first_source_tick = int(grouped[entity_id][0]["tick"])
    if tick < max(activation_tick, first_source_tick):
        return False
    deactivation_tick = roster_entry.get("deactivation_tick")
    if deactivation_tick is not None and tick >= int(deactivation_tick):
        return False
    if entity_id in visible_until_tick_by_background_ground_flow:
        visible_until_tick = int(
            visible_until_tick_by_background_ground_flow[entity_id]
        )
        if visible_until_tick < 0 or tick > visible_until_tick:
            return False
    profile = profile_for_entity(roster_entry, grouped[entity_id][0])
    category = str(profile["entity_category"])
    if category == "uav" and entity_id in first_moving_tick_by_local_uav:
        first_motion_tick = int(first_moving_tick_by_local_uav[entity_id])
        if first_motion_tick < 0:
            return False
        if first_motion_tick > 0 and tick < first_motion_tick:
            return False
    row = sample_row_at_tick(grouped[entity_id], tick, tick_hz)
    source_state = str(row.get("state") or "").casefold()
    source_activity = str(row.get("activity_type") or "").casefold()
    if source_state == "offstage" or source_activity == "offstage":
        return False
    position = source_position_for_runtime(row, category)
    return point_in_runtime_boundary_for_contract(position, source_contract)


def source_runtime_visible_ticks_by_entity(
    *,
    roster_entities: Sequence[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    ticks: Sequence[int],
    tick_hz: int,
    source_contract: dict[str, Any],
) -> dict[str, set[int]]:
    visible_until_tick_by_background_ground_flow = {
        str(entry["entity_id"]): visible_until_tick_for_ground_flow(
            grouped[str(entry["entity_id"])], tick_hz, entry
        )
        for entry in roster_entities
        if str(entry.get("entity_id") or "") in grouped
        and is_background_ground_flow_actor(entry)
    }
    first_moving_tick_by_local_uav = {
        str(entry["entity_id"]): first_moving_tick(grouped[str(entry["entity_id"])])
        for entry in roster_entities
        if str(entry.get("entity_id") or "") in grouped
        and str(entry.get("entity_category") or entry.get("label_class") or "") == "uav"
        and uav_requires_motion_before_visibility(entry)
    }
    visible_ticks_by_entity: dict[str, set[int]] = {}
    for entry in roster_entities:
        entity_id = str(entry.get("entity_id") or "")
        if entity_id not in grouped:
            continue
        visible_ticks_by_entity[entity_id] = {
            int(tick)
            for tick in ticks
            if source_entity_visible_at_tick(
                roster_entry=entry,
                grouped=grouped,
                tick=int(tick),
                tick_hz=tick_hz,
                source_contract=source_contract,
                visible_until_tick_by_background_ground_flow=visible_until_tick_by_background_ground_flow,
                first_moving_tick_by_local_uav=first_moving_tick_by_local_uav,
            )
        }
    return visible_ticks_by_entity


def retarget_source_roster_initial_pose(
    *,
    roster_entry: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
    first_visible_tick: int,
    tick_hz: int,
    fallback_yaw_deg: float,
) -> tuple[list[float], list[float], float, dict[str, Any]]:
    entity_id = str(roster_entry["entity_id"])
    row = sample_row_at_tick(grouped[entity_id], int(first_visible_tick), tick_hz)
    profile = profile_for_entity(roster_entry, row)
    category = str(profile["entity_category"])
    position = source_position_for_runtime(row, category)
    velocity = normalize_vector3(row.get("vel_mps"))
    yaw = heading_deg_from_velocity(velocity, fallback_deg=fallback_yaw_deg)
    roster_entry["initial_position_enu_m"] = position
    roster_entry["initial_yaw_deg"] = round(float(yaw), 6)
    return position, velocity, yaw, row


def iter_event_entity_references(
    event_rows: Sequence[dict[str, Any]],
) -> Iterable[tuple[int, str, dict[str, Any]]]:
    for row in event_rows:
        try:
            tick = int(
                row.get("tick")
                if row.get("tick") is not None
                else row.get("activated_tick") or 0
            )
        except Exception:
            tick = 0
        candidates: list[Any] = []
        candidates.append(row.get("target_id"))
        candidates.extend(list(row.get("target_ids") or []))
        scope = dict(row.get("scope") or {})
        candidates.append(scope.get("target_id"))
        candidates.extend(list(scope.get("entities") or []))
        payload = dict(row.get("payload") or {})
        candidates.append(payload.get("target_id"))
        candidates.extend(list(payload.get("target_ids") or []))
        for value in candidates:
            entity_id = str(value or "").strip()
            if entity_id:
                yield tick, entity_id, row


def event_pad_boundary_policy(row: dict[str, Any]) -> Any:
    for payload in (
        row,
        dict(row.get("metadata") or {}),
        dict(row.get("payload") or {}),
    ):
        policy = payload.get("pad_boundary_policy")
        if policy not in (None, ""):
            return policy
    return {}


def event_target_roles(row: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for payload in (
        row,
        dict(row.get("metadata") or {}),
        dict(row.get("payload") or {}),
    ):
        for role in payload.get("target_roles") or []:
            role_text = str(role or "").strip().lower()
            if role_text:
                roles.add(role_text)
    return roles


def source_entity_is_landing_pad(
    entity_id: str, source_roster: dict[str, dict[str, Any]]
) -> bool:
    entry = dict(source_roster.get(entity_id) or {})
    semantic_scope = entry.get("semantic_scope")
    return bool(
        isinstance(semantic_scope, dict)
        and semantic_scope.get("scope_type") == "facility"
        and semantic_scope.get("scope_subtype") == "landing_pad"
    )


def event_allows_external_landing_pad(
    row: dict[str, Any], entity_id: str, source_roster: dict[str, dict[str, Any]]
) -> bool:
    if not source_entity_is_landing_pad(entity_id, source_roster):
        return False
    roles = event_target_roles(row)
    if roles and "landing_pad" not in roles:
        return False
    policy = event_pad_boundary_policy(row)
    if isinstance(policy, str):
        default_policy = policy
        inside_required_for: list[Any] = []
    elif isinstance(policy, dict):
        default_policy = str(policy.get("default") or "")
        inside_required_for = list(policy.get("inside_required_for") or [])
    else:
        default_policy = ""
        inside_required_for = []
    required = {
        str(value or "").strip().lower()
        for value in inside_required_for
        if str(value or "").strip()
    }
    if entity_id.lower() in required or "landing_pad" in required or "all" in required:
        return False
    return default_policy.lower() == "outside_allowed"


def validate_event_references_survive_runtime_crop(
    *,
    source_episode_name: str,
    event_rows: Sequence[dict[str, Any]],
    source_roster: dict[str, dict[str, Any]],
    visible_ticks_by_entity: dict[str, set[int]],
    allowed_missing_entity_ids: set[str] | None = None,
) -> None:
    missing: list[str] = []
    seen: set[tuple[int, str]] = set()
    allowed_missing = set(allowed_missing_entity_ids or set())
    for tick, entity_id, row in iter_event_entity_references(event_rows):
        if (tick, entity_id) in seen:
            continue
        seen.add((tick, entity_id))
        if entity_id not in source_roster:
            continue
        if entity_id in allowed_missing:
            continue
        if not visible_ticks_by_entity.get(entity_id):
            if event_allows_external_landing_pad(row, entity_id, source_roster):
                continue
            missing.append(f"{entity_id}@tick{tick}")
    if missing:
        preview = ", ".join(missing[:20])
        raise RuntimeError(
            f"{source_episode_name}: event semantic entity references have no expanded-boundary runtime truth: {preview}"
        )


def subset_vehicle_selection_for_runtime(
    selection: VehicleSelection,
    visible_vehicle_ids: set[str],
    required_lifecycle_vehicle_ids: set[str],
) -> VehicleSelection:
    selected_ids = visible_vehicle_ids | required_lifecycle_vehicle_ids
    selected = tuple(
        vehicle_id for vehicle_id in selection.vehicle_ids if vehicle_id in selected_ids
    )
    moving_ids = {
        vehicle_id
        for vehicle_id in selected
        if selection.motion_span_m_by_vehicle_id.get(vehicle_id, 0.0) >= 1e-6
        or selection.max_speed_mps_by_vehicle_id.get(vehicle_id, 0.0) >= 1e-6
    }
    return VehicleSelection(
        vehicle_ids=selected,
        entity_ids={
            vehicle_id: selection.entity_ids[vehicle_id] for vehicle_id in selected
        },
        min_distance_m_by_vehicle_id={
            vehicle_id: selection.min_distance_m_by_vehicle_id.get(vehicle_id, 0.0)
            for vehicle_id in selected
        },
        frames_seen_by_vehicle_id={
            vehicle_id: selection.frames_seen_by_vehicle_id.get(vehicle_id, 0)
            for vehicle_id in selected
        },
        motion_span_m_by_vehicle_id={
            vehicle_id: selection.motion_span_m_by_vehicle_id.get(vehicle_id, 0.0)
            for vehicle_id in selected
        },
        max_speed_mps_by_vehicle_id={
            vehicle_id: selection.max_speed_mps_by_vehicle_id.get(vehicle_id, 0.0)
            for vehicle_id in selected
        },
        scenario_vehicle_ids=tuple(
            vehicle_id
            for vehicle_id in selection.scenario_vehicle_ids
            if vehicle_id in selected_ids
        ),
        candidate_count=selection.candidate_count,
        moving_candidate_count=len(moving_ids),
        selected_count=len(selected),
        min_visible_vehicle_target=selection.min_visible_vehicle_target,
        max_visible_vehicle_target=selection.max_visible_vehicle_target,
        expanded_to_nearest=selection.expanded_to_nearest,
    )


def runtime_visible_sumo_vehicle_ids(
    *,
    sumo_dataset: SumoTrafficDataset,
    sumo_segment: Any,
    sumo_selection: VehicleSelection,
    vehicle_lane_projector: VehicleLaneProjector,
    source_contract: dict[str, Any],
    ticks: Sequence[int],
    tick_hz: int,
    scenario_id: str,
) -> set[str]:
    visible: set[str] = set()
    for tick in ticks:
        sample = sumo_dataset.sample(
            segment=sumo_segment,
            episode_sim_time_s=int(tick) / float(tick_hz),
            selected_vehicle_ids=sumo_selection.vehicle_ids,
            scenario_id=scenario_id,
        )
        for vehicle in sample.get("vehicles") or []:
            vehicle_id = str(vehicle.get("vehicle_id") or "")
            if not vehicle_id or vehicle_id in visible:
                continue
            position = projected_sumo_vehicle_position(vehicle, vehicle_lane_projector)
            if point_in_runtime_boundary_for_contract(position, source_contract):
                visible.add(vehicle_id)
    return visible


def subset_uav_selection_for_runtime(
    selection: UavSelection, visible_uav_ids: set[str]
) -> UavSelection:
    selected = tuple(
        uav_id for uav_id in selection.uav_ids if uav_id in visible_uav_ids
    )
    return UavSelection(
        uav_ids=selected,
        entity_ids={uav_id: selection.entity_ids[uav_id] for uav_id in selected},
        task_ids={uav_id: selection.task_ids.get(uav_id, "") for uav_id in selected},
        mission_type_by_uav_id={
            uav_id: selection.mission_type_by_uav_id.get(uav_id, "")
            for uav_id in selected
        },
        selected_count=len(selected),
        active_count_min=0,
        active_count_max=selection.active_count_max,
        active_count_mean=selection.active_count_mean,
        min_distance_m_by_uav_id={
            uav_id: selection.min_distance_m_by_uav_id.get(uav_id, 0.0)
            for uav_id in selected
        },
        frames_seen_by_uav_id={
            uav_id: selection.frames_seen_by_uav_id.get(uav_id, 0)
            for uav_id in selected
        },
        motion_span_m_by_uav_id={
            uav_id: selection.motion_span_m_by_uav_id.get(uav_id, 0.0)
            for uav_id in selected
        },
        max_speed_mps_by_uav_id={
            uav_id: selection.max_speed_mps_by_uav_id.get(uav_id, 0.0)
            for uav_id in selected
        },
        candidate_count=selection.candidate_count,
        observable_candidate_count=len(selected),
        visibility_padding_m=RUNTIME_BOUNDARY_PADDING_M,
    )


def runtime_visible_global_uav_ids(
    *,
    uav_dataset: UavGlobalFlowDataset,
    uav_segment: UavSegment,
    uav_selection: UavSelection,
    source_contract: dict[str, Any],
    ticks: Sequence[int],
    tick_hz: int,
) -> set[str]:
    visible: set[str] = set()
    for tick in ticks:
        sample = uav_dataset.sample(
            segment=uav_segment,
            episode_sim_time_s=int(tick) / float(tick_hz),
            selected_uav_ids=uav_selection.uav_ids,
        )
        for uav in sample.get("uavs") or []:
            uav_id = str(uav.get("uav_id") or "")
            if not uav_id or uav_id in visible:
                continue
            position = normalize_vector3(uav.get("position_enu_m"))
            if point_in_runtime_boundary_for_contract(position, source_contract):
                visible.add(uav_id)
    return visible


def filter_runtime_visible_uav_pad_roster_entries(
    entries: Sequence[dict[str, Any]],
    source_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        position = normalize_vector3(entry.get("initial_position_enu_m"))
        if point_in_runtime_boundary_for_contract(position, source_contract):
            filtered.append(dict(entry))
    return filtered


def clean_vehicle_logical_asset(value: Any) -> str:
    asset = str(value or "").strip()
    if not asset or asset.lower() in {"unknown", "none", "null"}:
        return ""
    if asset not in KNOWN_VEHICLE_LOGICAL_ASSETS:
        raise RuntimeError(
            f"Unknown vehicle logical_asset_id {asset!r}; refusing fallback asset selection"
        )
    return asset


def logical_asset_for_exact_sumo_type(type_id: Any, *, context: str) -> str:
    normalized = str(type_id or "").strip().lower()
    if not normalized:
        raise RuntimeError(
            f"{context}: missing SUMO type_id; refusing fallback asset selection"
        )
    if "@" in normalized:
        normalized = normalized.split("@", 1)[0].strip()
    asset = SUMO_TYPE_LOGICAL_ASSET_BY_TYPE_ID.get(normalized)
    if not asset:
        raise RuntimeError(
            f"{context}: unknown SUMO type_id {type_id!r}; refusing fallback asset selection"
        )
    return asset


def sumo_vehicle_profile_for_asset(logical_asset_id: str) -> dict[str, str]:
    asset = clean_vehicle_logical_asset(logical_asset_id)
    profile = dict(ENTITY_PROFILES["vehicle"])
    profile["logical_asset_id"] = asset
    return profile


def source_roster_vehicle_asset(
    source_roster: Mapping[str, dict[str, Any]], source_entity_id: str
) -> str:
    if not source_entity_id:
        return ""
    source_entry = dict(source_roster.get(source_entity_id) or {})
    if not source_entry:
        return ""
    return clean_vehicle_logical_asset(
        source_entry.get("logical_asset_id") or source_entry.get("asset_id")
    )


def source_semantic_plan_vehicle_asset(
    *,
    source_semantic_vehicle_plan: Mapping[str, dict[str, Any]],
    source_entity_id: str,
    context: str,
) -> dict[str, Any]:
    if not source_entity_id:
        return {}
    source_entry = dict(source_semantic_vehicle_plan.get(source_entity_id) or {})
    if not source_entry:
        return {}
    sumo_vehicle = dict(source_entry.get("sumo_vehicle") or {})
    type_id = str(sumo_vehicle.get("type_id") or "").strip()
    plan_asset = clean_vehicle_logical_asset(
        source_entry.get("logical_asset_id") or source_entry.get("asset_id")
    )
    if type_id:
        type_asset = logical_asset_for_exact_sumo_type(
            type_id,
            context=f"{context}:source_sumo_semantic_vehicle_plan:{source_entity_id}",
        )
        if plan_asset and plan_asset != type_asset:
            raise RuntimeError(
                f"{context}: source semantic vehicle plan {source_entity_id!r} "
                f"asset {plan_asset!r} conflicts with SUMO type {type_id!r} "
                f"asset {type_asset!r}"
            )
        return {
            "logical_asset_id": type_asset,
            "authority": "source_sumo_semantic_vehicle_plan_type",
            "source_sumo_semantic_type_id": type_id,
        }
    if plan_asset:
        return {
            "logical_asset_id": plan_asset,
            "authority": "source_sumo_semantic_vehicle_plan_asset",
            "source_sumo_semantic_type_id": "",
        }
    raise RuntimeError(
        f"{context}: source SUMO semantic vehicle plan entry {source_entity_id!r} has "
        "no exact type_id or logical_asset_id; refusing fallback asset selection"
    )


def validate_explicit_vehicle_source_identity(
    *,
    episode_id: str,
    seed_index: int,
    explicit_vehicle_plan: Mapping[str, Any],
    source_roster: Mapping[str, dict[str, Any]],
) -> None:
    if explicit_vehicle_plan.get("episode_id") != episode_id:
        raise RuntimeError(f"{episode_id}: explicit SUMO plan episode identity differs")
    expected_ids = {
        f"{source_id}__seed{seed_index:02d}": source_id
        for source_id in source_roster
    }
    seen_vehicle_ids: set[str] = set()
    for vehicle in explicit_vehicle_plan.get("vehicles") or []:
        vehicle_id = str(vehicle.get("vehicle_id") or "")
        source_id = str(vehicle.get("source_entity_id") or "")
        if not vehicle_id or vehicle_id in seen_vehicle_ids:
            raise RuntimeError(
                f"{episode_id}: missing or duplicate explicit SUMO vehicle ID {vehicle_id!r}"
            )
        seen_vehicle_ids.add(vehicle_id)
        if not isinstance(vehicle.get("source_presence_required"), bool):
            raise RuntimeError(
                f"{episode_id}:{vehicle_id}: source_presence_required must be boolean"
            )
        if vehicle["source_presence_required"] and not source_id:
            raise RuntimeError(
                f"{episode_id}:{vehicle_id}: source presence requires source_entity_id"
            )
        if source_id:
            if source_id not in source_roster:
                raise RuntimeError(
                    f"{episode_id}:{vehicle_id}: unknown source entity {source_id!r}"
                )
            expected_id = f"{source_id}__seed{seed_index:02d}"
            if vehicle_id != expected_id:
                raise RuntimeError(
                    f"{episode_id}:{vehicle_id}: source entity {source_id!r} "
                    f"requires vehicle ID {expected_id!r}"
                )
        elif vehicle_id in expected_ids:
            raise RuntimeError(
                f"{episode_id}:{vehicle_id}: source entity reference is missing"
            )


def canonical_sumo_vehicle_asset_records(
    *,
    episode_id: str,
    sumo_dataset: SumoTrafficDataset,
    sumo_segment: Any,
    sumo_selection: VehicleSelection,
    explicit_vehicle_plan: dict[str, Any] | None,
    source_semantic_vehicle_plan: Mapping[str, dict[str, Any]],
    source_roster: Mapping[str, dict[str, Any]],
    source_trajectories: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    explicit_by_vehicle_id = {
        str(vehicle.get("vehicle_id") or ""): dict(vehicle)
        for vehicle in (explicit_vehicle_plan or {}).get("vehicles") or []
        if str(vehicle.get("vehicle_id") or "")
    }
    records: dict[str, dict[str, Any]] = {}
    for vehicle_id in sumo_selection.vehicle_ids:
        first_record = sumo_dataset.first_vehicle_record_in_segment(
            sumo_segment, vehicle_id
        )
        if not first_record:
            raise RuntimeError(
                f"{episode_id}:{vehicle_id}: selected SUMO vehicle lacks a source frame"
            )
        explicit_vehicle = explicit_by_vehicle_id.get(vehicle_id, {})
        explicit_source_id = str(explicit_vehicle.get("source_entity_id") or "").strip()
        sumo_source_id = str(first_record.get("source_entity_id") or "").strip()
        if explicit_source_id and sumo_source_id and explicit_source_id != sumo_source_id:
            raise RuntimeError(
                f"{episode_id}:{vehicle_id}: explicit plan source entity "
                f"{explicit_source_id!r} conflicts with SUMO record {sumo_source_id!r}"
            )
        source_entity_id = str(
            explicit_source_id or sumo_source_id
        ).strip()
        context = f"{episode_id}:{vehicle_id}"
        explicit_type_id = str(explicit_vehicle.get("type_id") or "").strip()
        first_record_type_id = str(first_record.get("vehicle_type") or "").strip()
        if not first_record_type_id:
            raise RuntimeError(
                f"{episode_id}:{vehicle_id}: SUMO source frame lacks vehicle_type"
            )
        source_asset = source_roster_vehicle_asset(source_roster, source_entity_id)
        source_semantic_record = source_semantic_plan_vehicle_asset(
            source_semantic_vehicle_plan=source_semantic_vehicle_plan,
            source_entity_id=source_entity_id,
            context=context,
        )
        if source_semantic_record:
            logical_asset_id = str(source_semantic_record["logical_asset_id"])
            authority = str(source_semantic_record["authority"])
            type_id = str(
                source_semantic_record.get("source_sumo_semantic_type_id")
                or explicit_type_id
                or first_record_type_id
            )
        elif explicit_vehicle:
            type_id = explicit_type_id
            logical_asset_id = logical_asset_for_exact_sumo_type(
                type_id,
                context=f"{context}:explicit_vehicle_plan",
            )
            authority = "explicit_vehicle_plan_type"
        elif source_asset:
            type_id = first_record_type_id
            logical_asset_id = source_asset
            authority = "source_roster"
        else:
            type_id = first_record_type_id
            logical_asset_id = logical_asset_for_exact_sumo_type(
                type_id,
                context=f"{context}:sumo_vehicle_type",
            )
            authority = "sumo_vehicle_type"
        if source_entity_id:
            source_rows = source_trajectories.get(source_entity_id)
            if not source_rows:
                raise RuntimeError(
                    f"{context}: source entity {source_entity_id!r} lacks trajectory truth"
                )
            trajectory_assets = {
                str(row.get("asset_id") or "").strip() for row in source_rows
            }
            if len(trajectory_assets) != 1 or not next(iter(trajectory_assets)):
                raise RuntimeError(
                    f"{context}: source entity {source_entity_id!r} has inconsistent trajectory assets: "
                    f"{sorted(trajectory_assets)}"
                )
            trajectory_asset = next(iter(trajectory_assets))
            if trajectory_asset != logical_asset_id:
                raise RuntimeError(
                    f"{context}: derived logical asset {logical_asset_id!r} conflicts with "
                    f"source trajectory asset {trajectory_asset!r} for {source_entity_id!r}"
                )
        if source_asset and source_asset != logical_asset_id:
            raise RuntimeError(
                f"{context}: source roster asset {source_asset!r} conflicts "
                f"with derived asset {logical_asset_id!r} from {authority}"
            )
        if explicit_type_id:
            explicit_type_asset = logical_asset_for_exact_sumo_type(
                explicit_type_id,
                context=f"{context}:explicit_vehicle_plan_conflict_check",
            )
            if explicit_type_asset != logical_asset_id:
                raise RuntimeError(
                    f"{context}: explicit SUMO plan type {explicit_type_id!r} "
                    f"asset {explicit_type_asset!r} conflicts with derived "
                    f"asset {logical_asset_id!r} from {authority}"
                )
        if first_record_type_id:
            first_record_type_asset = logical_asset_for_exact_sumo_type(
                first_record_type_id,
                context=f"{context}:sumo_vehicle_type_conflict_check",
            )
            if first_record_type_asset != logical_asset_id:
                raise RuntimeError(
                    f"{context}: SUMO frame type {first_record_type_id!r} "
                    f"asset {first_record_type_asset!r} conflicts with derived "
                    f"asset {logical_asset_id!r} from {authority}"
                )
        records[vehicle_id] = {
            "logical_asset_id": logical_asset_id,
            "authority": authority,
            "source_entity_id": source_entity_id,
            "source_presence_required": explicit_vehicle.get("source_presence_required")
            is True,
            "sumo_type_id": str(type_id or ""),
            "source_roster_logical_asset_id": source_asset,
            "explicit_type_id": explicit_type_id,
            "sumo_record_type_id": first_record_type_id,
        }
    return records


def semantic_vehicle_activity_type(vehicle: dict[str, Any]) -> str:
    state = dict(vehicle.get("semantic_vehicle_state") or {})
    mode = str(state.get("mode") or "").strip()
    if mode:
        return mode
    if str(vehicle.get("control_role") or "") == "incident_controlled":
        return "incident_controlled"
    return "moving"


def build_sumo_vehicle_roster_entries(
    *,
    sumo_dataset: SumoTrafficDataset,
    sumo_segment: Any,
    sumo_selection: VehicleSelection,
    canonical_asset_records: Mapping[str, dict[str, Any]],
    vehicle_lane_projector: VehicleLaneProjector,
    road_signal_context: RoadSignalContext,
    site_id: str,
    roi_id: str,
    source_roster: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    segment_payload = sumo_segment.as_dict()
    for vehicle_id in sumo_selection.vehicle_ids:
        first_record = sumo_dataset.first_vehicle_record_in_segment(
            sumo_segment, vehicle_id
        )
        if not first_record:
            continue
        canonical_asset_record = dict(canonical_asset_records.get(vehicle_id) or {})
        canonical_asset = clean_vehicle_logical_asset(
            canonical_asset_record.get("logical_asset_id")
        )
        if not canonical_asset:
            raise RuntimeError(
                f"{vehicle_id}: missing canonical SUMO vehicle logical_asset_id"
            )
        profile = sumo_vehicle_profile_for_asset(canonical_asset)
        position = projected_sumo_vehicle_position(first_record, vehicle_lane_projector)
        yaw_deg = road_derived_sumo_vehicle_yaw_deg(
            first_record, position, vehicle_lane_projector
        )
        entity_id = sumo_selection.entity_ids[vehicle_id]
        source_entity_id = str(
            canonical_asset_record.get("source_entity_id")
            or first_record.get("source_entity_id")
            or ""
        )
        semantic_vehicle = bool(first_record.get("semantic_vehicle"))
        source_presence_required = (
            canonical_asset_record.get("source_presence_required") is True
        )
        background_role = (
            "sumo_semantic_event_vehicle"
            if semantic_vehicle
            else "sumo_source_required_context_vehicle"
            if source_presence_required
            else "sumo_background_traffic"
        )
        tags = [
            "vehicle",
            "sumo_traci",
            (
                "semantic_event_vehicle"
                if semantic_vehicle
                else "source_required_context_vehicle"
                if source_presence_required
                else "background_traffic"
            ),
        ]
        semantic_metadata = dict(first_record.get("semantic_metadata") or {})
        semantic_metadata["logical_asset_id"] = canonical_asset
        semantic_metadata["logical_asset_authority"] = str(
            canonical_asset_record.get("authority") or ""
        )
        static_road_fields = road_signal_context.static_sumo_vehicle_fields(
            first_record
        )
        entry = {
            "entity_id": entity_id,
            "sumo_vehicle_id": vehicle_id,
            "label_class": "vehicle",
            "asset_id": profile["logical_asset_id"],
            "site_id": site_id,
            "roi_id": roi_id,
            "entity_category": "vehicle",
            "entity_kind": "vehicle.car",
            "entity_type": "vehicle.car",
            "proxy_template_id": profile["proxy_template_id"],
            "logical_asset_id": profile["logical_asset_id"],
            "mode": "scene_sync",
            "initial_position_enu_m": position,
            "initial_yaw_deg": round(yaw_deg, 6),
            "tags": tags,
            "source": "sumo_traci",
            "background_role": background_role,
            "background_vehicle": {
                "policy": "sumo_truth_frame_background_v1",
                "source": "sumo_traci",
                "sumo_vehicle_id": vehicle_id,
                "vehicle_type": first_record.get("vehicle_type"),
                "control_role": first_record.get("control_role"),
                "semantic_vehicle": semantic_vehicle,
                "source_presence_required": source_presence_required,
                "source_entity_id": source_entity_id,
                "semantic_episode_id": first_record.get("semantic_episode_id"),
            },
            "ground_flow_contract": {
                "policy": "sumo_segment_truth_replay_v1",
                "actor_kind": "vehicle",
                "route_source": "sumo_traci_fcd_projected_to_ue_truth",
                "sample_period_s": sumo_dataset.manifest.get("sample_period_s"),
                "segment": segment_payload,
            },
            "sumo_segment": segment_payload,
            "sumo_vehicle": {
                "vehicle_id": vehicle_id,
                "source_entity_id": source_entity_id,
                "semantic_episode_id": first_record.get("semantic_episode_id"),
                "semantic_vehicle": semantic_vehicle,
                "source_presence_required": source_presence_required,
                "semantic_metadata": semantic_metadata,
                "canonical_logical_asset_id": canonical_asset,
                "logical_asset_authority": str(
                    canonical_asset_record.get("authority") or ""
                ),
                "active_semantic_event": copy.deepcopy(
                    first_record.get("active_semantic_event")
                ),
                "semantic_vehicle_state": copy.deepcopy(
                    first_record.get("semantic_vehicle_state")
                ),
                "vehicle_type": first_record.get("vehicle_type"),
                "route_id": first_record.get("route_id"),
                "sumo_edge_id": first_record.get("sumo_edge_id"),
                "sumo_lane_id": first_record.get("sumo_lane_id"),
                "lane_position_m": first_record.get("lane_position_m"),
                "center_lane_position_m": first_record.get("center_lane_position_m"),
                "sumo_position_reference": first_record.get("sumo_position_reference"),
                "truth_position_reference": first_record.get(
                    "truth_position_reference"
                ),
                "control_role": first_record.get("control_role"),
                **static_road_fields,
            },
            "sumo_visibility": {
                "min_observation_distance_m": round(
                    float(
                        sumo_selection.min_distance_m_by_vehicle_id.get(vehicle_id, 0.0)
                    ),
                    6,
                ),
                "frames_seen_in_segment": int(
                    sumo_selection.frames_seen_by_vehicle_id.get(vehicle_id, 0)
                ),
            },
        }
        if source_entity_id:
            entry.update(
                runtime_state_fields_from_sources(
                    dict((source_roster or {}).get(source_entity_id) or {}),
                    context=f"sumo_roster[{vehicle_id}].source_entity[{source_entity_id}]",
                )
            )
        entries.append(entry)
    return entries


def sumo_vehicle_truth_entity(
    *,
    vehicle: dict[str, Any],
    roster_entry: dict[str, Any],
    vehicle_lane_projector: VehicleLaneProjector,
    tick: int,
    site_id: str,
    roi_id: str,
    capture_boundary_id: str,
    observation_distance_m: float,
    source_runtime_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_asset = clean_vehicle_logical_asset(
        roster_entry.get("logical_asset_id") or roster_entry.get("asset_id")
    )
    if not canonical_asset:
        raise RuntimeError(
            f"{roster_entry.get('entity_id')}: missing canonical SUMO vehicle logical_asset_id"
        )
    profile = sumo_vehicle_profile_for_asset(canonical_asset)
    position = projected_sumo_vehicle_position(vehicle, vehicle_lane_projector)
    velocity = normalize_vector3(vehicle.get("velocity_enu_mps"))
    yaw_deg = road_derived_sumo_vehicle_yaw_deg(
        vehicle, position, vehicle_lane_projector
    )
    activity_type = semantic_vehicle_activity_type(vehicle)
    row_for_annotations = {"vel_mps": velocity}
    semantic_metadata = dict(vehicle.get("semantic_metadata") or {})
    roster_sumo_vehicle = dict(roster_entry.get("sumo_vehicle") or {})
    roster_semantic_metadata = dict(roster_sumo_vehicle.get("semantic_metadata") or {})
    source_entity_id = str(
        roster_sumo_vehicle.get("source_entity_id")
        or vehicle.get("source_entity_id")
        or ""
    )
    semantic_vehicle = bool(roster_sumo_vehicle.get("semantic_vehicle"))
    source_presence_required = (
        roster_sumo_vehicle.get("source_presence_required") is True
    )
    semantic_metadata["logical_asset_id"] = canonical_asset
    semantic_metadata["logical_asset_authority"] = str(
        roster_sumo_vehicle.get("logical_asset_authority")
        or roster_semantic_metadata.get("logical_asset_authority")
        or ""
    )
    entity = {
        "entity_id": str(roster_entry["entity_id"]),
        "entity_category": "vehicle",
        "entity_kind": "vehicle.car",
        "entity_type": "vehicle.car",
        "label_class": "vehicle",
        "site_id": site_id,
        "roi_id": roi_id,
        "proxy_template_id": profile["proxy_template_id"],
        "logical_asset_id": profile["logical_asset_id"],
        "tags": list(roster_entry.get("tags") or []),
        "truth_pose": truth_pose(position, yaw_deg, velocity),
        "render_presence": render_presence(roi_id),
        "annotations": build_annotations(activity_type, row_for_annotations, "vehicle"),
        "state": activity_type,
        "state_revision": int(tick) + 1,
        "visual_revision": 1,
        "source": "sumo_traci",
        "capture_boundary_id": capture_boundary_id,
        "background_role": roster_entry.get("background_role"),
        "background_vehicle": copy.deepcopy(
            roster_entry.get("background_vehicle") or {}
        ),
        "ground_flow_contract": copy.deepcopy(
            roster_entry.get("ground_flow_contract") or {}
        ),
        "sumo_segment": copy.deepcopy(roster_entry.get("sumo_segment") or {}),
        "sumo_vehicle": {
            "vehicle_id": vehicle.get("vehicle_id"),
            "source_entity_id": source_entity_id,
            "semantic_episode_id": vehicle.get("semantic_episode_id"),
            "semantic_vehicle": semantic_vehicle,
            "source_presence_required": source_presence_required,
            "semantic_metadata": semantic_metadata,
            "canonical_logical_asset_id": canonical_asset,
            "logical_asset_authority": semantic_metadata.get("logical_asset_authority"),
            "active_semantic_event": copy.deepcopy(
                vehicle.get("active_semantic_event")
            ),
            "semantic_vehicle_state": copy.deepcopy(
                vehicle.get("semantic_vehicle_state")
            ),
            "vehicle_type": vehicle.get("vehicle_type"),
            "route_id": vehicle.get("route_id"),
            "sumo_edge_id": vehicle.get("sumo_edge_id"),
            "sumo_lane_id": vehicle.get("sumo_lane_id"),
            "lane_position_m": vehicle.get("lane_position_m"),
            "center_lane_position_m": vehicle.get("center_lane_position_m"),
            "sumo_xy_m": vehicle.get("sumo_xy_m"),
            "sumo_position_reference": vehicle.get("sumo_position_reference"),
            "sumo_front_bumper_xy_m": vehicle.get("sumo_front_bumper_xy_m"),
            "truth_front_bumper_enu_m": vehicle.get("truth_front_bumper_enu_m"),
            "truth_position_reference": vehicle.get("truth_position_reference"),
            "sumo_angle_deg": vehicle.get("sumo_angle_deg"),
            "speed_mps": vehicle.get("speed_mps"),
            "accel_mps2": vehicle.get("accel_mps2"),
            "signals": vehicle.get("signals"),
            "dimensions_m": vehicle.get("dimensions_m"),
            "control_role": vehicle.get("control_role"),
            "lane_id": vehicle.get("lane_id"),
            "lane_ontology_class_id": vehicle.get("lane_ontology_class_id"),
            "lane_instance_source_ref": vehicle.get("lane_instance_source_ref"),
            "allowed_speed_mps": vehicle.get("allowed_speed_mps"),
            "speed_limit_regulation_id": vehicle.get("speed_limit_regulation_id"),
            "speed_limit_regulation_ontology_class_id": vehicle.get(
                "speed_limit_regulation_ontology_class_id"
            ),
            "speed_limit_regulation_source_ref": vehicle.get(
                "speed_limit_regulation_source_ref"
            ),
            "following_distance_rule_id": vehicle.get("following_distance_rule_id"),
            "following_distance_rule_ontology_class_id": vehicle.get(
                "following_distance_rule_ontology_class_id"
            ),
            "following_distance_rule_source_ref": vehicle.get(
                "following_distance_rule_source_ref"
            ),
            "controlling_signal_id": vehicle.get("controlling_signal_id"),
            "controlling_signal_ontology_class_id": vehicle.get(
                "controlling_signal_ontology_class_id"
            ),
            "controlling_signal_state": vehicle.get("controlling_signal_state"),
            "stop_line_id": vehicle.get("stop_line_id"),
            "stop_line_ontology_class_id": vehicle.get("stop_line_ontology_class_id"),
            "stop_line_source_ref": vehicle.get("stop_line_source_ref"),
            "right_of_way_id": vehicle.get("right_of_way_id"),
            "right_of_way_ontology_class_id": vehicle.get(
                "right_of_way_ontology_class_id"
            ),
            "right_of_way_source_ref": vehicle.get("right_of_way_source_ref"),
            "crossed_stop_line": vehicle.get("crossed_stop_line"),
            "source_prev_time_s": vehicle.get("source_prev_time_s"),
            "source_next_time_s": vehicle.get("source_next_time_s"),
            "source_alpha": vehicle.get("source_alpha"),
        },
        "sumo_visibility": {
            "inspect_observation_distance_m": round(float(observation_distance_m), 6),
            "selected_for_capture_truth": True,
        },
    }
    if source_runtime_state:
        entity.update(
            runtime_state_fields_from_sources(
                source_runtime_state,
                context=f"sumo_truth[{entity['entity_id']}].source_runtime_state",
            )
        )
    return entity


def sumo_frame_truth_payload(
    *,
    sumo_dataset: SumoTrafficDataset,
    sumo_segment: Any,
    sumo_selection: VehicleSelection,
    sumo_sample: dict[str, Any],
    scenario_id: str,
) -> dict[str, Any]:
    traffic_lights = dict(sumo_sample.get("traffic_lights") or {})
    active_incidents = list(sumo_sample.get("active_incidents") or [])
    return {
        "enabled": True,
        "segment": sumo_segment.as_dict(),
        "scenario_id": scenario_id,
        "absolute_time_s": sumo_sample["absolute_time_s"],
        "source_prev_time_s": sumo_sample["source_prev_time_s"],
        "source_next_time_s": sumo_sample["source_next_time_s"],
        "source_alpha": sumo_sample["source_alpha"],
        "source_vehicle_count": int(sumo_sample.get("source_vehicle_count") or 0),
        "selected_vehicle_count": int(sumo_selection.selected_count),
        "active_selected_vehicle_count": len(sumo_sample.get("vehicles") or []),
        "traffic_light_count": len(traffic_lights),
        "traffic_light_states": traffic_lights,
        "active_incidents": active_incidents,
        "active_incident_count": len(active_incidents),
    }


def _validate_cached_formal_clock(output_episode_dir: Path, episode_id: str) -> None:
    frame_path = output_episode_dir / "truth_frames.jsonl"
    count = 0
    with frame_path.open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"cached formal frame is invalid JSON: {frame_path}:{line}") from exc
            if not isinstance(frame, dict):
                raise ValueError(f"cached formal frame is not an object: {frame_path}:{line}")
            if frame.get("episode_id") != episode_id:
                raise ValueError(f"cached formal frame belongs to another episode: {frame_path}:{line}")
            validate_formal_frame_clock(frame, expected_tick=count,
                                        source=f"{frame_path}:{line}")
            count += 1
    if count != DEFAULT_DURATION_TICKS + 1:
        raise ValueError(f"cached formal frame count differs from contract: {frame_path}: {count}")
    plan_path = output_episode_dir / "scenario_plan.json"
    plan = load_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError(f"cached formal scenario plan is not an object: {plan_path}")
    runtime = plan.get("runtime_contract")
    if (plan.get("episode_id") != episode_id or not isinstance(runtime, dict) or
            type(runtime.get("tick_hz")) is not int or
            type(runtime.get("tick_start")) is not int or
            type(runtime.get("tick_end")) is not int or
            type(runtime.get("dt_s")) not in (int, float) or
            runtime != {"tick_hz": DEFAULT_TICK_HZ, "dt_s": 1 / DEFAULT_TICK_HZ,
                        "tick_start": 0, "tick_end": DEFAULT_DURATION_TICKS}):
        raise ValueError(f"cached formal runtime clock differs from frames: {plan_path}")


def convert_episode(
    source_episode_dir: Path,
    output_episode_dir: Path,
    *,
    project_root: Path,
    map_id: str = DEFAULT_MAP_ID,
    site_id: str = DEFAULT_SITE_ID,
    roi_id: str = DEFAULT_ROI_ID,
    tick_hz: int = DEFAULT_TICK_HZ,
    overwrite: bool = False,
    sumo_output_dir: Path | None = DEFAULT_SUMO_OUTPUT_DIR,
    enable_sumo_traffic: bool = True,
    preloaded_sumo_dataset: SumoTrafficDataset | None = None,
    uav_output_dir: Path | None = DEFAULT_UAV_OUTPUT_DIR,
    enable_uav_global_flow: bool = True,
    preloaded_uav_dataset: UavGlobalFlowDataset | None = None,
) -> dict[str, Any]:
    if type(tick_hz) is not int or tick_hz != DEFAULT_TICK_HZ:
        raise ValueError(f"formal episode clock requires {DEFAULT_TICK_HZ} Hz, got {tick_hz!r}")
    source_episode_dir = source_episode_dir.resolve()
    output_episode_dir = output_episode_dir.resolve()
    manifest = load_json(source_episode_dir / "episode_manifest.json")
    episode_id = str(manifest.get("episode_id") or source_episode_dir.name)
    if output_episode_dir.exists() and not overwrite:
        required = [
            "truth_frames.jsonl",
            "scenario_plan.json",
            "global_entity_roster.json",
        ]
        if all((output_episode_dir / name).exists() for name in required):
            _validate_cached_formal_clock(output_episode_dir, episode_id)
            return {
                "episode_dir": str(output_episode_dir),
                "skipped": True,
                "reason": "render-ready outputs already exist",
            }

    scenario_id = str(manifest.get("scenario_id") or source_episode_dir.name)
    duration_ticks = int(manifest.get("duration_ticks") or 0)
    grouped, max_traj_tick = load_trajectory_groups(
        source_episode_dir / "trajectories.jsonl"
    )
    event_rows = load_jsonl(source_episode_dir / "event_trace.jsonl")
    event_realization_path = source_episode_dir / "event_realization.jsonl"
    if not event_realization_path.exists():
        raise RuntimeError(
            f"{source_episode_dir.name}: missing authoritative event_realization.jsonl; regenerate Dataset/episodes with batch_generate.py"
        )
    event_realization_rows = load_jsonl(event_realization_path)
    if len(event_realization_rows) != len(event_rows):
        raise RuntimeError(
            f"{source_episode_dir.name}: event_realization count {len(event_realization_rows)} "
            f"does not match event_trace count {len(event_rows)}"
        )
    source_weather_rows = load_jsonl(source_episode_dir / "weather_meta.jsonl")
    source_roster = read_source_roster(source_episode_dir / "global_entity_roster.json")
    source_sumo_semantic_vehicle_plan = read_source_sumo_semantic_vehicle_plan(
        source_episode_dir / "sumo_semantic_vehicle_plan.json"
    )
    event_script_path = resolve_manifest_path(
        manifest.get("source_event_script_path"), source_episode_dir, project_root
    )
    event_script = load_json_or_none(event_script_path)
    if not isinstance(event_script, dict):
        raise RuntimeError(f"{source_episode_dir.name}: source event_script is missing")

    if not grouped:
        raise RuntimeError(f"No trajectory rows found in {source_episode_dir}")
    missing_roster = sorted(
        entity_id for entity_id in grouped if entity_id not in source_roster
    )
    if missing_roster:
        raise RuntimeError(
            f"{source_episode_dir.name}: trajectory entities missing from source roster: {missing_roster}"
        )
    if duration_ticks <= 0:
        duration_ticks = max_traj_tick
    ticks = list(range(0, duration_ticks + 1))
    scene_setup_path = resolve_manifest_path(
        manifest.get("source_scene_setup_path"), source_episode_dir, project_root
    )
    scene_setup = load_json_or_none(scene_setup_path)
    if not isinstance(scene_setup, dict):
        raise RuntimeError(f"Missing scene_setup.json for source episode {source_episode_dir}")
    source_scene_entities = scene_setup_entities(scene_setup)
    source_contract = capture_contract_from_source(
        source_episode_dir=source_episode_dir,
        manifest=manifest,
        scene_setup=scene_setup,
        project_root=project_root,
        scenario_id=scenario_id,
        grouped_rows=grouped,
    )
    if not source_contract["uav_crosses_boundary"]:
        raise RuntimeError(
            f"{source_episode_dir.name}: source trajectories do not prove all mission/observer UAVs cross capture boundary"
        )
    if not source_contract["inspect_observes_boundary"]:
        raise RuntimeError(
            f"{source_episode_dir.name}: source inspect route/sensor profile does not prove boundary observation"
        )
    interest_visibility = VisibilityGeometry.from_contract(source_contract)
    runtime_spatial_crop = runtime_spatial_crop_payload(source_contract)
    vehicle_lane_projector = VehicleLaneProjector(project_root, map_id)
    road_signal_context = RoadSignalContext.load()
    source_background_vehicle_clearance_filter = {
        "enabled": False,
        "policy": "validator_reports_pedestrian_vehicle_clearance_no_converter_deletion_v1",
        "reason": "generation_truth_must_be_fixed_upstream_instead_of_filtering_source_background_vehicles",
    }

    sumo_dataset: SumoTrafficDataset | None = None
    sumo_segment: Any | None = None
    sumo_visibility: VisibilityGeometry | None = None
    sumo_selection: VehicleSelection | None = None
    sumo_roster_entities: list[dict[str, Any]] = []
    sumo_roster_by_vehicle_id: dict[str, dict[str, Any]] = {}
    capture_required_sumo_vehicle_ids: set[str] = set()
    canonical_sumo_asset_records: dict[str, dict[str, Any]] = {}
    sumo_pedestrian_clearance_filter: dict[str, Any] = {"enabled": False}
    sumo_explicit_selection_audit: dict[str, Any] = {}
    explicit_vehicle_plan: dict[str, Any] | None = None
    explicit_vehicle_ids: set[str] = set()
    explicit_source_vehicle_ids: set[str] = set()
    if enable_sumo_traffic:
        if sumo_output_dir is None:
            raise RuntimeError(
                "SUMO traffic integration is enabled but no SUMO output directory was provided"
            )
        explicit_vehicle_plan = load_explicit_vehicle_plan(source_episode_dir)
        seed_index = manifest.get("seed")
        if isinstance(seed_index, bool) or not isinstance(seed_index, int):
            raise RuntimeError(
                f"{source_episode_dir.name}: source episode lacks integer seed"
            )
        validate_explicit_vehicle_source_identity(
            episode_id=episode_id,
            seed_index=seed_index,
            explicit_vehicle_plan=explicit_vehicle_plan,
            source_roster=source_roster,
        )
        explicit_vehicle_ids = planned_vehicle_ids(explicit_vehicle_plan)
        explicit_source_vehicle_ids = planned_source_vehicle_ids(
            explicit_vehicle_plan
        ) - planned_script_controlled_source_vehicle_ids(explicit_vehicle_plan)
        explicit_semantic_vehicle_ids = explicit_vehicle_ids_by_traffic_role(
            explicit_vehicle_plan, "semantic_vehicle"
        )
        explicit_source_presence_required_ids = {
            str(vehicle.get("vehicle_id") or "")
            for vehicle in explicit_vehicle_plan.get("vehicles") or []
            if vehicle.get("source_presence_required") is True
            and str(vehicle.get("vehicle_id") or "")
        }
        required_capture_vehicle_ids = (
            explicit_semantic_vehicle_ids | explicit_source_presence_required_ids
        )
        if not explicit_vehicle_ids:
            raise RuntimeError(
                f"{source_episode_dir.name}: explicit SUMO vehicle plan has no vehicles"
            )
        sumo_dataset = load_sumo_dataset_for_episode(
            base_dir=Path(sumo_output_dir),
            episode_id=episode_id,
            preloaded_sumo_dataset=preloaded_sumo_dataset,
        )
        validate_source_presence_manifest_contract(
            explicit_vehicle_plan,
            sumo_dataset.manifest,
        )
        sumo_segment = sumo_dataset.segment_for_episode(episode_id, manifest)
        if abs(float(duration_ticks) / float(tick_hz) - sumo_segment.duration_s) > 1e-6:
            raise RuntimeError(
                f"{source_episode_dir.name}: episode duration must be {sumo_segment.duration_s:.1f}s "
                f"to bind the episode-local vehicle SUMO capture window; found {duration_ticks / float(tick_hz):.3f}s"
            )
        if (
            str(sumo_dataset.manifest.get("vehicle_source_policy") or "")
            != VEHICLE_SOURCE_POLICY
        ):
            raise RuntimeError(
                f"{source_episode_dir.name}: SUMO manifest vehicle_source_policy must be "
                f"{VEHICLE_SOURCE_POLICY!r}, found {sumo_dataset.manifest.get('vehicle_source_policy')!r}"
            )
        sumo_visibility = interest_visibility
        sumo_selection = sumo_dataset.select_segment_vehicles(
            segment=sumo_segment,
            scenario_id=scenario_id,
            episode_id=episode_id,
        )
        seen_vehicle_ids = set(sumo_selection.vehicle_ids)
        extra_vehicle_ids = sorted(seen_vehicle_ids - explicit_vehicle_ids)
        missing_required_vehicle_ids = sorted(
            required_capture_vehicle_ids - seen_vehicle_ids
        )
        missing_background_vehicle_ids = sorted(
            (explicit_vehicle_ids - seen_vehicle_ids) - required_capture_vehicle_ids
        )
        if extra_vehicle_ids or missing_required_vehicle_ids:
            raise RuntimeError(
                f"{source_episode_dir.name}: SUMO segment vehicle roster must contain only explicit-plan vehicles "
                f"and all source-required/event-controlled vehicles; extra={extra_vehicle_ids[:12]} "
                f"missing_required={missing_required_vehicle_ids[:12]} "
                f"missing_background={missing_background_vehicle_ids[:12]}"
            )
        runtime_visible_ids = runtime_visible_sumo_vehicle_ids(
            sumo_dataset=sumo_dataset,
            sumo_segment=sumo_segment,
            sumo_selection=sumo_selection,
            vehicle_lane_projector=vehicle_lane_projector,
            source_contract=source_contract,
            ticks=ticks,
            tick_hz=tick_hz,
            scenario_id=scenario_id,
        )
        # Source-required vehicles are lifecycle truth, not ROI background. They
        # remain in the render-ready source package even when their geometry is
        # outside the runtime crop; capture filtering is validated separately.
        runtime_visible_ids.update(required_capture_vehicle_ids)
        selected_explicit_vehicle_ids = seen_vehicle_ids & explicit_vehicle_ids
        missing_visible_required_vehicle_ids = sorted(
            required_capture_vehicle_ids - runtime_visible_ids
        )
        missing_visible_background_vehicle_ids = sorted(
            (selected_explicit_vehicle_ids - runtime_visible_ids)
            - required_capture_vehicle_ids
        )
        extra_visible_vehicle_ids = sorted(runtime_visible_ids - explicit_vehicle_ids)
        if missing_visible_required_vehicle_ids:
            raise RuntimeError(
                f"{source_episode_dir.name}: render-ready runtime crop must preserve source-required and "
                f"event-controlled explicit-plan vehicles "
                f"and reject extras; extra={extra_visible_vehicle_ids[:12]} "
                f"missing_required={missing_visible_required_vehicle_ids[:12]} "
                f"missing_background={missing_visible_background_vehicle_ids[:12]}"
            )
        sumo_selection = subset_vehicle_selection_for_runtime(
            sumo_selection,
            runtime_visible_ids,
            required_capture_vehicle_ids,
        )
        sumo_explicit_selection_audit = {
            "policy": "semantic_hard_sumo_authoritative_background_tolerant_selection_v1",
            "planned_vehicle_count": len(explicit_vehicle_ids),
            "planned_semantic_vehicle_count": len(explicit_semantic_vehicle_ids),
            "planned_source_presence_required_vehicle_count": len(
                explicit_source_presence_required_ids
            ),
            "planned_capture_required_vehicle_count": len(required_capture_vehicle_ids),
            "realized_vehicle_count": len(seen_vehicle_ids),
            "selected_runtime_visible_vehicle_count": len(sumo_selection.vehicle_ids),
            "extra_realized_background_vehicle_count": len(extra_vehicle_ids),
            "extra_realized_background_vehicle_ids": extra_vehicle_ids[:100],
            "missing_background_vehicle_count": len(missing_background_vehicle_ids),
            "missing_background_vehicle_ids": missing_background_vehicle_ids[:100],
            "extra_visible_background_vehicle_count": len(extra_visible_vehicle_ids),
            "extra_visible_background_vehicle_ids": extra_visible_vehicle_ids[:100],
            "missing_visible_background_vehicle_count": len(
                missing_visible_background_vehicle_ids
            ),
            "missing_visible_background_vehicle_ids": missing_visible_background_vehicle_ids[
                :100
            ],
        }
        sumo_pedestrian_clearance_filter = {
            "enabled": False,
            "policy": "validator_reports_pedestrian_vehicle_clearance_no_sumo_selection_deletion_v1",
            "reason": "SUMO vehicle truth is cropped only by expanded runtime boundary; generator/validator owns conflict fixes",
        }
        canonical_sumo_asset_records = canonical_sumo_vehicle_asset_records(
            episode_id=episode_id,
            sumo_dataset=sumo_dataset,
            sumo_segment=sumo_segment,
            sumo_selection=sumo_selection,
            explicit_vehicle_plan=explicit_vehicle_plan,
            source_semantic_vehicle_plan=source_sumo_semantic_vehicle_plan,
            source_roster=source_roster,
            source_trajectories=grouped,
        )
        sumo_roster_entities = build_sumo_vehicle_roster_entries(
            sumo_dataset=sumo_dataset,
            sumo_segment=sumo_segment,
            sumo_selection=sumo_selection,
            canonical_asset_records=canonical_sumo_asset_records,
            vehicle_lane_projector=vehicle_lane_projector,
            road_signal_context=road_signal_context,
            site_id=site_id,
            roi_id=roi_id,
            source_roster=source_roster,
        )
        sumo_roster_by_vehicle_id = {
            str(entry["sumo_vehicle_id"]): entry for entry in sumo_roster_entities
        }
        capture_required_sumo_vehicle_ids = (
            capture_required_sumo_vehicle_ids_from_roster(sumo_roster_entities)
        )
        if len(sumo_roster_entities) != sumo_selection.selected_count:
            raise RuntimeError(
                f"{source_episode_dir.name}: selected SUMO vehicles missing first records "
                f"({len(sumo_roster_entities)} of {sumo_selection.selected_count})"
            )

    uav_dataset: UavGlobalFlowDataset | None = None
    uav_segment: UavSegment | None = None
    uav_selection: UavSelection | None = None
    uav_roster_entities: list[dict[str, Any]] = []
    uav_roster_by_uav_id: dict[str, dict[str, Any]] = {}
    uav_pad_roster_entities: list[dict[str, Any]] = []
    if enable_uav_global_flow:
        if uav_output_dir is None:
            raise RuntimeError(
                "UAV global flow integration is enabled but no UAV output directory was provided"
            )
        uav_dataset = preloaded_uav_dataset or load_uav_global_flow_dataset(
            Path(uav_output_dir)
        )
        uav_segment = uav_dataset.segment_for_episode(episode_id, manifest)
        if abs(float(duration_ticks) / float(tick_hz) - uav_segment.duration_s) > 1e-6:
            raise RuntimeError(
                f"{source_episode_dir.name}: episode duration must be {uav_segment.duration_s:.1f}s "
                f"to bind UAV seed segments; found {duration_ticks / float(tick_hz):.3f}s"
            )
        uav_selection = uav_dataset.select_segment_uavs(segment=uav_segment)
        uav_selection = subset_uav_selection_for_runtime(
            uav_selection,
            runtime_visible_global_uav_ids(
                uav_dataset=uav_dataset,
                uav_segment=uav_segment,
                uav_selection=uav_selection,
                source_contract=source_contract,
                ticks=ticks,
                tick_hz=tick_hz,
            ),
        )
        uav_roster_entities = build_uav_roster_entries(
            uav_dataset=uav_dataset,
            uav_segment=uav_segment,
            uav_selection=uav_selection,
            site_id=site_id,
            roi_id=roi_id,
        )
        uav_roster_by_uav_id = {
            str(entry["uav_id"]): entry for entry in uav_roster_entities
        }
        uav_pad_roster_entities = build_uav_pad_roster_entries(
            uav_dataset=uav_dataset,
            site_id=site_id,
            roi_id=roi_id,
        )
        uav_pad_roster_entities = filter_runtime_visible_uav_pad_roster_entries(
            uav_pad_roster_entities,
            source_contract,
        )
        if len(uav_roster_entities) != uav_selection.selected_count:
            raise RuntimeError(
                f"{source_episode_dir.name}: selected UAVs missing first records "
                f"({len(uav_roster_entities)} of {uav_selection.selected_count})"
            )

    roster_entities: list[dict[str, Any]] = []
    first_samples: dict[str, dict[str, Any]] = {}
    last_yaw_by_entity: dict[str, float] = {}
    sumo_replaced_source_vehicle_ids: set[str] = set()

    for entity_id in sorted(grouped):
        source_entry = dict(source_roster[entity_id])
        label_class = str(
            source_entry.get("label_class")
            or grouped[entity_id][0].get("label_class")
            or ""
        )
        first_row = sample_row_at_tick(grouped[entity_id], ticks[0], tick_hz)
        profile = profile_for_entity(source_entry, first_row)
        if (
            enable_sumo_traffic
            and str(profile.get("entity_category") or "") == "vehicle"
            and entity_id in explicit_source_vehicle_ids
        ):
            sumo_replaced_source_vehicle_ids.add(entity_id)
            continue
        source_asset_id = str(
            source_entry.get("asset_id")
            or source_entry.get("logical_asset_id")
            or profile.get("logical_asset_id")
            or ""
        )
        if not source_asset_id:
            raise RuntimeError(
                f"{source_episode_dir.name}: source roster entry lacks asset id: {entity_id}"
            )
        first_position = normalized_position_for_render(first_row)
        first_velocity = normalize_vector3(first_row.get("vel_mps"))
        if str(profile.get("entity_category") or "") == "uav":
            source_initial_yaw = scenario_initial_yaw_degrees(
                entity_id=entity_id,
                first_row=first_row,
                source_entry=source_entry,
                scene_entities=source_scene_entities,
            )
        else:
            source_initial_yaw = float(
                first_row.get("yaw_deg", source_entry.get("initial_yaw_deg", 0.0))
                or 0.0
            )
        first_yaw = heading_deg_from_velocity(
            first_velocity,
            fallback_deg=source_initial_yaw,
        )
        if str(profile.get("entity_category") or "") == "vehicle":
            first_position = source_vehicle_truth_position(first_position)
        first_samples[entity_id] = {
            "row": first_row,
            "position_enu_m": first_position,
            "velocity_enu_mps": first_velocity,
            "yaw_deg": first_yaw,
            "label_class": label_class,
            "profile": profile,
        }
        last_yaw_by_entity[entity_id] = first_yaw
        roster_entry = {
            "entity_id": entity_id,
            "label_class": label_class,
            "asset_id": source_asset_id,
            "site_id": site_id,
            "roi_id": roi_id,
            "entity_category": profile["entity_category"],
            "entity_kind": profile["entity_kind"],
            "entity_type": profile["entity_kind"],
            "proxy_template_id": profile["proxy_template_id"],
            "logical_asset_id": logical_asset_for(source_entry, profile),
            "mode": profile["mode"],
            "initial_position_enu_m": first_position,
            "initial_yaw_deg": round(first_yaw, 6),
            "tags": [profile["entity_category"], label_class],
        }
        roster_entry.update(preserved_fields_from(source_entry, first_row))
        if (
            str(roster_entry.get("role") or "") == "semantic_facility"
            and profile["entity_category"] == "facility"
        ):
            roster_entry["entity_category"] = "facility"
            roster_asset_id = str(
                roster_entry.get("asset_id") or source_entry.get("asset_id") or ""
            )
            roster_entry["entity_kind"] = (
                "facility.landing_pad"
                if roster_asset_id.startswith("facility.landing_pad")
                else roster_entry["entity_kind"]
            )
        if entity_id == source_contract["inspect_entity_id"]:
            roster_entry["route_waypoints_enu_m"] = [
                list(point) for point in source_contract["inspect_route_enu_m"]
            ]
        roster_entities.append(roster_entry)

    source_visible_ticks_by_entity = source_runtime_visible_ticks_by_entity(
        roster_entities=roster_entities,
        grouped=grouped,
        ticks=ticks,
        tick_hz=tick_hz,
        source_contract=source_contract,
    )
    validate_event_references_survive_runtime_crop(
        source_episode_name=source_episode_dir.name,
        event_rows=event_rows,
        source_roster=source_roster,
        visible_ticks_by_entity=source_visible_ticks_by_entity,
        allowed_missing_entity_ids=sumo_replaced_source_vehicle_ids,
    )
    cropped_source_roster_entities: list[dict[str, Any]] = []
    for roster_entry in roster_entities:
        entity_id = str(roster_entry["entity_id"])
        visible_ticks = source_visible_ticks_by_entity.get(entity_id, set())
        if not visible_ticks:
            continue
        first_visible_tick = min(visible_ticks)
        fallback_yaw = float(last_yaw_by_entity.get(entity_id, 0.0))
        first_position, first_velocity, first_yaw, first_row = (
            retarget_source_roster_initial_pose(
                roster_entry=roster_entry,
                grouped=grouped,
                first_visible_tick=first_visible_tick,
                tick_hz=tick_hz,
                fallback_yaw_deg=fallback_yaw,
            )
        )
        first_samples[entity_id] = {
            "row": first_row,
            "position_enu_m": first_position,
            "velocity_enu_mps": first_velocity,
            "yaw_deg": first_yaw,
            "label_class": roster_entry.get("label_class"),
            "profile": profile_for_entity(roster_entry, first_row),
        }
        last_yaw_by_entity[entity_id] = first_yaw
        roster_entry["runtime_visibility"] = {
            "policy": RUNTIME_SPATIAL_CROP_POLICY,
            "first_visible_tick": int(first_visible_tick),
            "last_visible_tick": int(max(visible_ticks)),
            "visible_tick_count": int(len(visible_ticks)),
            "expanded_boundary_padding_m": RUNTIME_BOUNDARY_PADDING_M,
        }
        cropped_source_roster_entities.append(roster_entry)
    roster_entities = cropped_source_roster_entities

    truth_frames: list[dict[str, Any]] = []
    visible_until_tick_by_background_ground_flow = {
        str(entry["entity_id"]): visible_until_tick_for_ground_flow(
            grouped[str(entry["entity_id"])], tick_hz, entry
        )
        for entry in roster_entities
        if is_background_ground_flow_actor(entry)
    }
    first_moving_tick_by_local_uav = {
        str(entry["entity_id"]): first_moving_tick(grouped[str(entry["entity_id"])])
        for entry in roster_entities
        if str(entry.get("entity_category") or entry.get("label_class") or "") == "uav"
        and uav_requires_motion_before_visibility(entry)
    }
    previous_sumo_vehicle_by_id: dict[str, dict[str, Any]] = {}
    previous_sumo_vehicle_tick_by_id: dict[str, int] = {}
    previous_sumo_traffic_light_states: Mapping[str, Mapping[str, Any]] | None = None
    for tick in ticks:
        entities: list[dict[str, Any]] = []
        for roster_entry in roster_entities:
            entity_id = str(roster_entry["entity_id"])
            if tick not in source_visible_ticks_by_entity.get(entity_id, set()):
                continue
            if entity_id in visible_until_tick_by_background_ground_flow:
                visible_until_tick = int(
                    visible_until_tick_by_background_ground_flow[entity_id]
                )
                if visible_until_tick < 0 or tick > visible_until_tick:
                    continue
            profile = profile_for_entity(roster_entry, grouped[entity_id][0])
            category = str(profile["entity_category"])
            if category == "uav" and entity_id in first_moving_tick_by_local_uav:
                first_motion_tick = int(first_moving_tick_by_local_uav[entity_id])
                if first_motion_tick < 0:
                    continue
                if first_motion_tick > 0 and tick < first_motion_tick:
                    continue
            row = sample_row_at_tick(grouped[entity_id], tick, tick_hz)
            position = source_position_for_runtime(row, category)
            velocity = normalize_vector3(row.get("vel_mps"))
            yaw = heading_deg_from_velocity(
                velocity,
                fallback_deg=float(
                    row.get(
                        "yaw_deg",
                        last_yaw_by_entity.get(entity_id, 0.0),
                    )
                    or 0.0
                ),
            )
            if math.hypot(velocity[0], velocity[1]) > 1e-4:
                last_yaw_by_entity[entity_id] = yaw
            activity_type = activity_for_sample(
                entity_id=entity_id,
                category=category,
                state=str(row.get("state") or ""),
                row_activity_type=str(row.get("activity_type") or ""),
                velocity_enu_mps=velocity,
                semantic_idle_when_stationary=is_background_ground_flow_actor(
                    roster_entry
                ),
            )
            entity = {
                "entity_id": entity_id,
                "entity_category": category,
                "entity_kind": profile["entity_kind"],
                "entity_type": profile["entity_kind"],
                "label_class": str(roster_entry.get("label_class") or ""),
                "site_id": site_id,
                "roi_id": roi_id,
                "proxy_template_id": profile["proxy_template_id"],
                "logical_asset_id": logical_asset_for(roster_entry, profile),
                "tags": list(roster_entry.get("tags") or []),
                "truth_pose": truth_pose(position, yaw, velocity),
                "render_presence": render_presence(roi_id),
                "annotations": build_annotations(activity_type, row, category),
                "state_revision": int(tick) + 1,
                "visual_revision": 1,
            }
            entity["runtime_visibility"] = runtime_visibility_payload(
                position, source_contract
            )
            entity.update(preserved_fields_from(row, roster_entry))
            if category == "uav":
                entity["uav_visibility"] = uav_camera_capture_visibility_payload(
                    position,
                    source_contract=source_contract,
                    interest_visibility=interest_visibility,
                )
            if entity_id == source_contract["inspect_entity_id"]:
                entity["route_waypoints_enu_m"] = [
                    list(point) for point in source_contract["inspect_route_enu_m"]
                ]
            if (
                str(entity.get("role") or "") == "semantic_facility"
                and category == "facility"
            ):
                entity["entity_category"] = "facility"
            if row.get("state") not in (None, ""):
                entity["state"] = row.get("state")
            entities.append(entity)

        sumo_truth_payload: dict[str, Any] | None = None
        if (
            sumo_dataset is not None
            and sumo_segment is not None
            and sumo_selection is not None
            and sumo_visibility is not None
        ):
            sumo_sample = sumo_dataset.sample(
                segment=sumo_segment,
                episode_sim_time_s=tick / float(tick_hz),
                selected_vehicle_ids=sumo_selection.vehicle_ids,
                scenario_id=scenario_id,
            )
            active_sumo_vehicle_count = 0
            runtime_boundary_visible_sumo_vehicle_count = 0
            semantic_lifecycle_outside_boundary_count = 0
            retained_outside_boundary_sumo_vehicle_count = 0
            current_sumo_traffic_light_states = dict(
                sumo_sample.get("traffic_lights") or {}
            )
            for vehicle in sumo_sample.get("vehicles") or []:
                vehicle_id = str(vehicle.get("vehicle_id") or "")
                roster_entry = sumo_roster_by_vehicle_id.get(vehicle_id)
                if not roster_entry:
                    continue
                previous_vehicle = (
                    previous_sumo_vehicle_by_id.get(vehicle_id)
                    if previous_sumo_vehicle_tick_by_id.get(vehicle_id) == tick - 1
                    else None
                )
                vehicle.update(
                    road_signal_context.enrich_sumo_vehicle(
                        vehicle,
                        current_sumo_traffic_light_states,
                        previous_sumo_vehicle=previous_vehicle,
                        previous_traffic_light_states=(
                            previous_sumo_traffic_light_states
                            if previous_vehicle is not None
                            else None
                        ),
                    )
                )
                previous_sumo_vehicle_by_id[vehicle_id] = copy.deepcopy(vehicle)
                previous_sumo_vehicle_tick_by_id[vehicle_id] = tick
                position = projected_sumo_vehicle_position(
                    vehicle, vehicle_lane_projector
                )
                inside_runtime_boundary = point_in_runtime_boundary_for_contract(
                    position, source_contract
                )
                is_capture_required_sumo_vehicle = (
                    vehicle_id in capture_required_sumo_vehicle_ids
                )
                if inside_runtime_boundary:
                    runtime_boundary_visible_sumo_vehicle_count += 1
                elif not is_capture_required_sumo_vehicle:
                    continue
                else:
                    retained_outside_boundary_sumo_vehicle_count += 1
                observation_distance_m = sumo_visibility.observation_distance_m(
                    position
                )
                roster_sumo_vehicle = dict(roster_entry.get("sumo_vehicle") or {})
                source_entity_id = str(
                    roster_sumo_vehicle.get("source_entity_id") or ""
                )
                source_runtime_state: dict[str, dict[str, Any]] = {}
                if source_entity_id:
                    source_rows = grouped.get(source_entity_id)
                    if not source_rows:
                        raise RuntimeError(
                            f"{source_episode_dir.name}: SUMO replacement {vehicle_id} references "
                            f"source entity {source_entity_id!r} without trajectory truth"
                        )
                    source_runtime_state = runtime_state_fields_from_sources(
                        sample_row_at_tick(source_rows, tick, tick_hz),
                        context=(
                            f"{source_episode_dir.name}.sumo_replacement[{vehicle_id}]"
                            f".source_entity[{source_entity_id}].tick[{tick}]"
                        ),
                    )
                entity = sumo_vehicle_truth_entity(
                    vehicle=vehicle,
                    roster_entry=roster_entry,
                    vehicle_lane_projector=vehicle_lane_projector,
                    tick=tick,
                    site_id=site_id,
                    roi_id=roi_id,
                    capture_boundary_id=source_contract["capture_boundary_id"],
                    observation_distance_m=observation_distance_m,
                    source_runtime_state=source_runtime_state,
                )
                entity["runtime_visibility"] = runtime_visibility_payload(
                    position, source_contract
                )
                if is_capture_required_sumo_vehicle:
                    entity["sumo_vehicle"]["required_lifecycle_preservation_policy"] = (
                        REQUIRED_SUMO_LIFECYCLE_PRESERVATION_POLICY
                    )
                if not inside_runtime_boundary:
                    entity["runtime_visibility"][
                        "retained_outside_runtime_boundary"
                    ] = True
                    entity["runtime_visibility"]["retention_policy"] = (
                        REQUIRED_SUMO_LIFECYCLE_PRESERVATION_POLICY
                        if is_capture_required_sumo_vehicle
                        else "sumo_background_traffic_retained_outside_runtime_boundary_v1"
                    )
                    semantic_lifecycle_outside_boundary_count += 1
                active_sumo_vehicle_count += 1
                entities.append(entity)
            previous_sumo_traffic_light_states = current_sumo_traffic_light_states
            sumo_truth_payload = sumo_frame_truth_payload(
                sumo_dataset=sumo_dataset,
                sumo_segment=sumo_segment,
                sumo_selection=sumo_selection,
                sumo_sample=sumo_sample,
                scenario_id=scenario_id,
            )
            sumo_truth_payload["active_selected_vehicle_count"] = (
                active_sumo_vehicle_count
            )
            sumo_truth_payload["runtime_boundary_visible_selected_vehicle_count"] = (
                runtime_boundary_visible_sumo_vehicle_count
            )
            sumo_truth_payload["semantic_lifecycle_outside_runtime_boundary_count"] = (
                semantic_lifecycle_outside_boundary_count
            )
            sumo_truth_payload["retained_outside_boundary_sumo_vehicle_count"] = (
                retained_outside_boundary_sumo_vehicle_count
            )
            sumo_truth_payload["max_observation_distance_m"] = sumo_visibility.padding_m
            sumo_semantics_payload = copy.deepcopy(sumo_truth_payload)
            sumo_semantics_payload.pop("traffic_light_states", None)
        else:
            sumo_semantics_payload = None

        uav_truth_payload: dict[str, Any] | None = None
        if (
            uav_dataset is not None
            and uav_segment is not None
            and uav_selection is not None
        ):
            for pad_entry in uav_pad_roster_entities:
                pad_entity = uav_pad_truth_entity(
                    roster_entry=pad_entry,
                    tick=tick,
                    site_id=site_id,
                    roi_id=roi_id,
                )
                pad_entity["runtime_visibility"] = runtime_visibility_payload(
                    normalize_vector3(pad_entry.get("initial_position_enu_m")),
                    source_contract,
                )
                entities.append(pad_entity)
            uav_sample = uav_dataset.sample(
                segment=uav_segment,
                episode_sim_time_s=tick / float(tick_hz),
                selected_uav_ids=uav_selection.uav_ids,
            )
            active_uav_count = 0
            roi_capture_eligible_uav_count = 0
            for uav in uav_sample.get("uavs") or []:
                uav_id = str(uav.get("uav_id") or "")
                roster_entry = uav_roster_by_uav_id.get(uav_id)
                if not roster_entry:
                    continue
                position = normalize_vector3(uav.get("position_enu_m"))
                if not point_in_runtime_boundary_for_contract(
                    position, source_contract
                ):
                    continue
                active_uav_count += 1
                entity = uav_global_truth_entity(
                    uav=uav,
                    roster_entry=roster_entry,
                    tick=tick,
                    site_id=site_id,
                    roi_id=roi_id,
                )
                entity["uav_visibility"] = uav_camera_capture_visibility_payload(
                    position,
                    source_contract=source_contract,
                    interest_visibility=interest_visibility,
                )
                entity["runtime_visibility"] = runtime_visibility_payload(
                    position, source_contract
                )
                if bool(entity["uav_visibility"].get("roi_capture_eligible")):
                    roi_capture_eligible_uav_count += 1
                entities.append(entity)
            uav_truth_payload = uav_frame_truth_payload(
                uav_dataset=uav_dataset,
                uav_segment=uav_segment,
                uav_selection=uav_selection,
                uav_sample=uav_sample,
                active_uav_count=active_uav_count,
            )
            uav_truth_payload["roi_capture_eligible_uav_count"] = int(
                roi_capture_eligible_uav_count
            )
            uav_truth_payload["camera_capture_roi_policy"] = (
                UAV_CAMERA_CAPTURE_ROI_POLICY
            )

        counts = Counter(str(entity["entity_category"]) for entity in entities)
        boundary_summary = truth_boundary_summary(
            entities,
            capture_boundary_id=source_contract["capture_boundary_id"],
            uav_crosses_boundary=source_contract["uav_crosses_boundary"],
            inspect_observes_boundary=source_contract["inspect_observes_boundary"],
            pad_boundary_policy=source_contract["pad_boundary_policy"],
        )
        truth_frames.append(
            {
                "schema_name": "truth_frame",
                "schema_version": "v1",
                "episode_id": episode_id,
                "frame_id": f"{episode_id}_tick_{tick}",
                "frame_seq": tick,
                "tick": tick,
                "tick_hz": tick_hz,
                "dt_s": round(1.0 / float(tick_hz), 6),
                "sim_time_s": round(tick / float(tick_hz), 6),
                "map_id": map_id,
                "render_mode": "ue_pie",
                "active_site_id": site_id,
                "active_roi_id": roi_id,
                "capture_boundary_id": boundary_summary["capture_boundary_id"],
                "uav_crosses_boundary": boundary_summary["uav_crosses_boundary"],
                "inspect_observes_boundary": boundary_summary[
                    "inspect_observes_boundary"
                ],
                "pad_boundary_policy": boundary_summary["pad_boundary_policy"],
                "sumo_segment": sumo_truth_payload["segment"]
                if sumo_truth_payload
                else None,
                "sumo_semantics": sumo_semantics_payload,
                "sumo_active_incidents": sumo_truth_payload["active_incidents"]
                if sumo_truth_payload
                else [],
                "sumo_traffic_light_states": sumo_truth_payload["traffic_light_states"]
                if sumo_truth_payload
                else {},
                "uav_segment": uav_truth_payload["segment"]
                if uav_truth_payload
                else None,
                "uav_global_flow": uav_truth_payload,
                "entity_motion_state": boundary_summary["entity_motion_state"],
                "roster_summary": {
                    "total": len(entities),
                    "by_category": dict(sorted(counts.items())),
                },
                "entities": entities,
            }
        )

    align_dynamic_entity_motion_to_final_positions(truth_frames, tick_hz)
    event_realization_rows = reconcile_event_realizations_to_render_truth(
        event_realization_rows,
        truth_frames,
        event_script=event_script,
        sumo_replaced_source_vehicle_ids=sumo_replaced_source_vehicle_ids,
    )

    output_episode_dir.mkdir(parents=True, exist_ok=True)
    source_event_trace_path = source_episode_dir / "event_trace.jsonl"
    if source_event_trace_path.exists():
        shutil.copy2(source_event_trace_path, output_episode_dir / "event_trace.jsonl")
    write_jsonl(output_episode_dir / "event_realization.jsonl", event_realization_rows)

    all_candidate_roster_entities = [
        *roster_entities,
        *sumo_roster_entities,
        *uav_pad_roster_entities,
        *uav_roster_entities,
    ]
    all_roster_entities = list(all_candidate_roster_entities)
    assert_dynamic_roster_entities_have_truth(
        episode_id=episode_id,
        roster_entities=all_roster_entities,
        truth_frames=truth_frames,
    )
    service_facility_repairs = repair_service_facility_placements(
        all_roster_entities=all_roster_entities,
        truth_frames=truth_frames,
        project_root=project_root,
    )
    sumo_asset_lifecycle = assert_sumo_vehicle_logical_assets_stable(
        episode_id, truth_frames
    )
    required_sumo_lifecycle: dict[str, Any] = {
        "policy": REQUIRED_SUMO_LIFECYCLE_PRESERVATION_POLICY,
        "required_vehicle_count": 0,
        "clipped_required_vehicle_count": 0,
    }
    if (
        sumo_dataset is not None
        and sumo_segment is not None
        and capture_required_sumo_vehicle_ids
    ):
        required_sumo_lifecycle = assert_required_sumo_vehicle_lifecycle_preserved(
            episode_id=episode_id,
            sumo_dataset=sumo_dataset,
            sumo_segment=sumo_segment,
            required_vehicle_ids=capture_required_sumo_vehicle_ids,
            truth_frames=truth_frames,
            tick_hz=tick_hz,
        )
    source_vehicle_authority = source_vehicle_authority_payload(
        enabled=enable_sumo_traffic,
        replaced_ids=sumo_replaced_source_vehicle_ids,
        source_roster=source_roster,
    )
    if explicit_vehicle_plan is not None:
        source_vehicle_authority["vehicle_source_policy"] = VEHICLE_SOURCE_POLICY
        source_vehicle_authority["explicit_vehicle_plan"] = {
            "schema": explicit_vehicle_plan.get("schema"),
            "episode_id": explicit_vehicle_plan.get("episode_id"),
            "vehicle_count": len(explicit_vehicle_ids),
            "vehicle_ids": sorted(explicit_vehicle_ids),
            "selection_audit": copy.deepcopy(sumo_explicit_selection_audit),
            "seed_profile": explicit_vehicle_plan.get("seed_profile"),
            "traffic_profile": explicit_vehicle_plan.get("traffic_profile"),
        }
    weather_rows = expand_weather_rows(source_weather_rows, ticks)
    dynamic_labels = build_dynamic_labels(
        event_rows,
        episode_id,
        scenario_id=scenario_id,
        event_realization_rows=event_realization_rows,
    )
    entity_counts = Counter(
        str(entity.get("entity_category") or "") for entity in all_roster_entities
    )
    roi_window = {
        "roi_id": roi_id,
        "site_id": site_id,
        "tick_start": ticks[0],
        "tick_end": ticks[-1],
        "bbox_enu_m": list(runtime_spatial_crop["bbox_enu_m"]),
        "bbox_source": RUNTIME_SPATIAL_CROP_POLICY,
        "expanded_boundary_padding_m": RUNTIME_BOUNDARY_PADDING_M,
    }
    site_contract = {
        "site_id": site_id,
        "roi_id": roi_id,
        "suggested_roi_id": roi_id,
        "tick_start": ticks[0],
        "tick_end": ticks[-1],
        "entity_count": len(all_roster_entities),
        "capture_boundary_id": source_contract["capture_boundary_id"],
        "uav_crosses_boundary": source_contract["uav_crosses_boundary"],
        "inspect_observes_boundary": source_contract["inspect_observes_boundary"],
        "pad_boundary_policy": source_contract["pad_boundary_policy"],
        "source_background_vehicle_clearance_filter": copy.deepcopy(
            source_background_vehicle_clearance_filter
        ),
        "source_vehicle_authority": copy.deepcopy(source_vehicle_authority),
        "runtime_spatial_crop": copy.deepcopy(runtime_spatial_crop),
    }
    compiled_summary = {
        "site_contracts": {site_id: site_contract},
        "roi_windows": {roi_id: roi_window},
        "entity_counts_by_category": dict(sorted(entity_counts.items())),
        "event_count": len(event_rows),
        "event_realization_count": len(event_realization_rows),
        "capture_boundary_id": source_contract["capture_boundary_id"],
        "uav_crosses_boundary": source_contract["uav_crosses_boundary"],
        "inspect_observes_boundary": source_contract["inspect_observes_boundary"],
        "pad_boundary_policy": source_contract["pad_boundary_policy"],
        "source_vehicle_authority": copy.deepcopy(source_vehicle_authority),
        "runtime_spatial_crop": copy.deepcopy(runtime_spatial_crop),
    }
    if service_facility_repairs:
        compiled_summary["scene_occupancy_repairs"] = {
            "policy": SERVICE_FACILITY_REPAIR_POLICY,
            "count": len(service_facility_repairs),
            "repairs": copy.deepcopy(service_facility_repairs),
        }
    if (
        sumo_dataset is not None
        and sumo_segment is not None
        and sumo_selection is not None
        and sumo_visibility is not None
    ):
        compiled_summary["sumo_traffic"] = {
            "enabled": True,
            "source": sumo_dataset.source_summary(),
            "segment": sumo_segment.as_dict(),
            "visibility_geometry": sumo_visibility.as_dict(),
            "selection": sumo_selection.as_dict(),
            "explicit_vehicle_plan": copy.deepcopy(
                source_vehicle_authority.get("explicit_vehicle_plan") or {}
            ),
            "explicit_selection_audit": copy.deepcopy(sumo_explicit_selection_audit),
            "canonical_asset_policy": copy.deepcopy(sumo_asset_lifecycle),
            "canonical_asset_records": copy.deepcopy(canonical_sumo_asset_records),
            "required_vehicle_lifecycle_policy": copy.deepcopy(required_sumo_lifecycle),
            "runtime_spatial_crop": copy.deepcopy(runtime_spatial_crop),
            "pedestrian_vehicle_clearance_filter": copy.deepcopy(
                sumo_pedestrian_clearance_filter
            ),
            "scenario_incidents": [
                {
                    "incident_id": item.get("incident_id"),
                    "episode_event_id": item.get("episode_event_id"),
                    "accident_class": item.get("accident_class"),
                    "start_s": item.get("start_s"),
                    "end_s": item.get("end_s"),
                    "injection_method": item.get("injection_method"),
                }
                for item in sumo_dataset.scenario_incidents(scenario_id)
            ],
        }
    if (
        uav_dataset is not None
        and uav_segment is not None
        and uav_selection is not None
    ):
        compiled_summary["uav_global_flow"] = {
            "enabled": True,
            "source": uav_dataset.source_summary(),
            "segment": uav_segment.as_dict(),
            "selection": uav_selection.as_dict(),
            "pad_count": len(uav_pad_roster_entities),
            "runtime_spatial_crop": copy.deepcopy(runtime_spatial_crop),
            "camera_capture_roi_policy": UAV_CAMERA_CAPTURE_ROI_POLICY,
        }
    scenario_plan = {
        "schema_name": "scenario_plan",
        "schema_version": "v1",
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "map_id": map_id,
        "capture_boundary_id": source_contract["capture_boundary_id"],
        "uav_crosses_boundary": source_contract["uav_crosses_boundary"],
        "inspect_observes_boundary": source_contract["inspect_observes_boundary"],
        "pad_boundary_policy": source_contract["pad_boundary_policy"],
        "source_background_vehicle_clearance_filter": copy.deepcopy(
            source_background_vehicle_clearance_filter
        ),
        "source_vehicle_authority": copy.deepcopy(source_vehicle_authority),
        "runtime_spatial_crop": copy.deepcopy(runtime_spatial_crop),
        "sumo_traffic": copy.deepcopy(
            compiled_summary.get("sumo_traffic") or {"enabled": False}
        ),
        "uav_global_flow": copy.deepcopy(
            compiled_summary.get("uav_global_flow") or {"enabled": False}
        ),
        "runtime_contract": {
            "tick_hz": tick_hz,
            "dt_s": round(1.0 / float(tick_hz), 6),
            "tick_start": ticks[0],
            "tick_end": ticks[-1],
        },
        "compiled_plan_summary": compiled_summary,
        "global_entity_roster": all_roster_entities,
        "export_contract": {
            "artifacts": {
                "scenario_plan": "scenario_plan.json",
                "global_entity_roster": "global_entity_roster.json",
                "truth_frames": "truth_frames.jsonl",
                "event_trace": "event_trace.jsonl",
                "event_realization": "event_realization.jsonl",
                "trajectories": "trajectories.jsonl",
                "weather_meta": "weather_meta.jsonl",
                "dynamic_labels": "dynamic_labels.jsonl",
            }
        },
        "scenario_plan": {
            "plan_id": scenario_id,
            "site_contracts": {site_id: site_contract},
            "summary": compiled_summary,
        },
    }
    render_trajectory_count = sum(
        len(frame.get("entities") or []) for frame in truth_frames
    )
    record_counts = {
        "scenario_plan": 1,
        "global_entity_roster": len(all_roster_entities),
        "truth_frames": len(truth_frames),
        "event_trace": len(event_rows),
        "event_realization": len(event_realization_rows),
        "trajectories": render_trajectory_count,
        "weather_meta": len(weather_rows),
        "dynamic_labels": len(dynamic_labels),
        "episode_manifest": 1,
    }
    artifacts = {
        "scenario_plan": repo_relative(
            output_episode_dir / "scenario_plan.json", project_root
        ),
        "global_entity_roster": repo_relative(
            output_episode_dir / "global_entity_roster.json", project_root
        ),
        "truth_frames": repo_relative(
            output_episode_dir / "truth_frames.jsonl", project_root
        ),
        "event_trace": repo_relative(
            output_episode_dir / "event_trace.jsonl", project_root
        ),
        "event_realization": repo_relative(
            output_episode_dir / "event_realization.jsonl", project_root
        ),
        "trajectories": repo_relative(
            output_episode_dir / "trajectories.jsonl", project_root
        ),
        "weather_meta": repo_relative(
            output_episode_dir / "weather_meta.jsonl", project_root
        ),
        "dynamic_labels": repo_relative(
            output_episode_dir / "dynamic_labels.jsonl", project_root
        ),
        "episode_manifest": repo_relative(
            output_episode_dir / "episode_manifest.json", project_root
        ),
    }
    render_manifest = copy.deepcopy(manifest)
    render_manifest.update(
        {
            "episode_id": episode_id,
            "scenario_id": scenario_id,
            "map_id": map_id,
            "capture_boundary_id": source_contract["capture_boundary_id"],
            "uav_crosses_boundary": source_contract["uav_crosses_boundary"],
            "inspect_observes_boundary": source_contract["inspect_observes_boundary"],
            "pad_boundary_policy": source_contract["pad_boundary_policy"],
            "runtime_spatial_crop": copy.deepcopy(runtime_spatial_crop),
            "source_vehicle_authority": copy.deepcopy(source_vehicle_authority),
            "sumo_traffic": copy.deepcopy(
                compiled_summary.get("sumo_traffic") or {"enabled": False}
            ),
            "uav_global_flow": copy.deepcopy(
                compiled_summary.get("uav_global_flow") or {"enabled": False}
            ),
            "generation": {
                "generator": "Dataset/tools/convert_to_render_ready.py",
                "source_episode_dir": repo_relative(source_episode_dir, project_root),
            },
            "record_counts": record_counts,
            "canonical_record_counts": copy.deepcopy(record_counts),
            "node_counts": {
                "all_nodes": len(all_roster_entities),
                "dynamic_nodes": sum(
                    1
                    for entity in all_roster_entities
                    if str(
                        entity.get("entity_category") or entity.get("label_class") or ""
                    ).lower()
                    in DYNAMIC_ENTITY_CATEGORIES
                ),
                "static_nodes": sum(
                    1
                    for entity in all_roster_entities
                    if str(
                        entity.get("entity_category") or entity.get("label_class") or ""
                    ).lower()
                    not in DYNAMIC_ENTITY_CATEGORIES
                ),
            },
            "artifacts": artifacts,
            "canonical_artifacts": copy.deepcopy(artifacts),
            "time_range": {
                "tick_start": ticks[0],
                "tick_end": ticks[-1],
                "sim_time_start": 0.0,
                "sim_time_end": round(ticks[-1] / float(tick_hz), 6),
            },
            "validation_summary": {"ok": True, "errors": [], "warnings": []},
        }
    )

    write_json(
        output_episode_dir / "global_entity_roster.json",
        {"entities": all_roster_entities},
    )
    write_jsonl(output_episode_dir / "truth_frames.jsonl", truth_frames)
    write_jsonl(output_episode_dir / "weather_meta.jsonl", weather_rows)
    write_jsonl(output_episode_dir / "dynamic_labels.jsonl", dynamic_labels)
    written_trajectory_count = write_truth_trajectories(
        output_episode_dir / "trajectories.jsonl", truth_frames
    )
    if written_trajectory_count != render_trajectory_count:
        raise RuntimeError(
            f"{source_episode_dir.name}: wrote {written_trajectory_count} trajectory rows, expected {render_trajectory_count}"
        )
    write_json(output_episode_dir / "scenario_plan.json", scenario_plan)
    write_json(output_episode_dir / "episode_manifest.json", render_manifest)
    scene_occupancy = audit_and_attach_episode(output_episode_dir, write_manifest=True)
    write_json(
        output_episode_dir / "scenario_package.json",
        {
            "scenario_id": scenario_id,
            "episode_id": episode_id,
            "root_dir": repo_relative(output_episode_dir, project_root),
            "truth_frames": repo_relative(
                output_episode_dir / "truth_frames.jsonl", project_root
            ),
            "weather_meta": repo_relative(
                output_episode_dir / "weather_meta.jsonl", project_root
            ),
            "scenario_plan": repo_relative(
                output_episode_dir / "scenario_plan.json", project_root
            ),
            "capture_plan": "",
            "episode_manifest": repo_relative(
                output_episode_dir / "episode_manifest.json", project_root
            ),
            "scene_occupancy_manifest": repo_relative(
                output_episode_dir / "scene_occupancy_manifest.json", project_root
            ),
        },
    )
    return {
        "episode_id": episode_id,
        "source_episode_dir": str(source_episode_dir),
        "episode_dir": str(output_episode_dir),
        "record_counts": record_counts,
        "scene_occupancy": scene_occupancy.get("validation_summary"),
        "skipped": False,
    }


def default_output_dir(source_episode_dir: Path, output_root: Path) -> Path:
    return output_root / source_episode_dir.name


def add_capture_visible_filter_result(
    result: dict[str, Any],
    *,
    capture_filter_output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    render_ready_episode_dir = Path(str(result["episode_dir"]))
    filter_result = filter_capture_visible_episode(
        render_ready_episode_dir,
        capture_filter_output_root,
        overwrite=overwrite,
    )
    enriched = dict(result)
    enriched["capture_filtered_episode_dir"] = str(
        (capture_filter_output_root / render_ready_episode_dir.name).resolve()
    )
    enriched["capture_filter"] = filter_result
    return enriched


def convert_episode_process(
    source_episode_dir: Path,
    *,
    output_root: Path,
    capture_filter_output_root: Path,
    project_root: Path,
    map_id: str,
    site_id: str,
    roi_id: str,
    tick_hz: int,
    overwrite: bool,
    sumo_output_dir: Path,
    uav_output_dir: Path,
) -> dict[str, Any]:
    result = convert_episode(
        source_episode_dir,
        default_output_dir(source_episode_dir, output_root),
        project_root=project_root,
        map_id=map_id,
        site_id=site_id,
        roi_id=roi_id,
        tick_hz=tick_hz,
        overwrite=overwrite,
        sumo_output_dir=sumo_output_dir,
        enable_sumo_traffic=True,
        preloaded_sumo_dataset=None,
        uav_output_dir=uav_output_dir,
        enable_uav_global_flow=True,
        preloaded_uav_dataset=None,
    )
    return add_capture_visible_filter_result(
        result,
        capture_filter_output_root=capture_filter_output_root,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Dataset episodes into episode_render_host-ready packages."
    )
    parser.add_argument(
        "--episode",
        type=Path,
        action="append",
        help="One Dataset episode directory to convert; may be repeated",
    )
    parser.add_argument(
        "--episodes-root",
        type=Path,
        default=Path("Dataset/episodes"),
        help="Source Dataset episodes root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Dataset/render_ready_episodes"),
        help="Render-ready output root",
    )
    parser.add_argument(
        "--capture-filter-output-root",
        type=Path,
        default=DEFAULT_CAPTURE_FILTER_OUTPUT_ROOT,
        help="Formal capture sync output root. This is mandatory for formal capture and is written on every conversion without corrective filtering.",
    )
    parser.add_argument("--map-id", default=DEFAULT_MAP_ID)
    parser.add_argument("--site-id", default=DEFAULT_SITE_ID)
    parser.add_argument("--roi-id", default=DEFAULT_ROI_ID)
    parser.add_argument("--tick-hz", type=int, default=DEFAULT_TICK_HZ)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every directory under --episodes-root",
    )
    parser.add_argument("--sumo-output-dir", type=Path, default=DEFAULT_SUMO_OUTPUT_DIR)
    parser.add_argument("--uav-output-dir", type=Path, default=DEFAULT_UAV_OUTPUT_DIR)
    parser.add_argument("--disable-uav-global-flow", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel episode conversion workers. Use 1 for deterministic single-threaded output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tick_hz != DEFAULT_TICK_HZ:
        raise SystemExit(f"formal episode clock requires --tick-hz {DEFAULT_TICK_HZ}")
    project_root = Path(__file__).resolve().parents[2]
    capture_filter_output_root = Path(args.capture_filter_output_root).resolve()
    if Path(args.output_root).resolve() == capture_filter_output_root:
        raise SystemExit(
            "--output-root and --capture-filter-output-root must be different roots."
        )
    if args.episode:
        episodes = list(args.episode)
    elif args.all:
        episodes = sorted(
            path for path in args.episodes_root.iterdir() if path.is_dir()
        )
    else:
        raise SystemExit("Specify --episode or --all.")

    worker_count = max(1, int(args.workers or 1))
    preloaded_sumo_dataset: SumoTrafficDataset | None = None
    preloaded_uav_dataset: UavGlobalFlowDataset | None = None
    if worker_count == 1:
        if has_sumo_dataset_files(Path(args.sumo_output_dir)):
            preloaded_sumo_dataset = load_sumo_traffic_dataset(
                Path(args.sumo_output_dir)
            )
        if not bool(args.disable_uav_global_flow):
            preloaded_uav_dataset = load_uav_global_flow_dataset(
                Path(args.uav_output_dir)
            )

    def _convert_one(source_episode_dir: Path) -> dict[str, Any]:
        result = convert_episode(
            source_episode_dir,
            default_output_dir(source_episode_dir, args.output_root),
            project_root=project_root,
            map_id=args.map_id,
            site_id=args.site_id,
            roi_id=args.roi_id,
            tick_hz=args.tick_hz,
            overwrite=bool(args.overwrite),
            sumo_output_dir=args.sumo_output_dir,
            enable_sumo_traffic=True,
            preloaded_sumo_dataset=preloaded_sumo_dataset,
            uav_output_dir=args.uav_output_dir,
            enable_uav_global_flow=not bool(args.disable_uav_global_flow),
            preloaded_uav_dataset=preloaded_uav_dataset,
        )
        return add_capture_visible_filter_result(
            result,
            capture_filter_output_root=capture_filter_output_root,
            overwrite=bool(args.overwrite),
        )

    results: list[dict[str, Any]] = []
    if worker_count == 1 or len(episodes) <= 1:
        for source_episode_dir in episodes:
            result = _convert_one(source_episode_dir)
            results.append(result)
            status = "skipped" if result.get("skipped") else "converted"
            print(
                f"[convert_to_render_ready] {status}: {source_episode_dir} -> {result['episode_dir']} "
                f"-> {result['capture_filtered_episode_dir']}"
            )
    else:
        if bool(args.disable_uav_global_flow):
            raise SystemExit(
                "Parallel formal conversion requires the authoritative UAV global flow."
            )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_episode = {
                executor.submit(
                    convert_episode_process,
                    episode,
                    output_root=Path(args.output_root),
                    capture_filter_output_root=capture_filter_output_root,
                    project_root=project_root,
                    map_id=str(args.map_id),
                    site_id=str(args.site_id),
                    roi_id=str(args.roi_id),
                    tick_hz=args.tick_hz,
                    overwrite=bool(args.overwrite),
                    sumo_output_dir=Path(args.sumo_output_dir),
                    uav_output_dir=Path(args.uav_output_dir),
                ): episode
                for episode in episodes
            }
            for future in as_completed(future_to_episode):
                source_episode_dir = future_to_episode[future]
                result = future.result()
                results.append(result)
                status = "skipped" if result.get("skipped") else "converted"
                print(
                    f"[convert_to_render_ready] {status}: {source_episode_dir} -> {result['episode_dir']} "
                    f"-> {result['capture_filtered_episode_dir']}"
                )

        results.sort(key=lambda item: str(item.get("episode_dir") or ""))

    print(
        json.dumps(
            {"count": len(results), "results": results}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
