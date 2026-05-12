from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from pedestrian_activity_catalog import activity_annotations, normalize_activity_type
from semantic_event_contract import contract_payload, get_contract


DEFAULT_MAP_ID = "donghu_road_topo"
DEFAULT_SITE_ID = "site.intersection_a"
DEFAULT_ROI_ID = "roi.intersection_a.v1"
DEFAULT_TICK_HZ = 10


ENTITY_PROFILES: dict[str, dict[str, str]] = {
    "uav": {
        "entity_category": "uav",
        "entity_kind": "uav.drone",
        "proxy_template_id": "drone.quadrotor",
        "logical_asset_id": "uav.inspect.quad.v1",
        "mode": "runtime_multirotor",
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
    "uav_corridor": {
        "entity_category": "airspace_corridor",
        "entity_kind": "airspace_corridor.uav_corridor",
        "proxy_template_id": "semantic.uav_corridor.segment.v1",
        "logical_asset_id": "semantic.uav_corridor.segment.v1",
        "mode": "metadata_only",
    },
}

ENTITY_PROFILES_BY_ASSET_ID: dict[str, dict[str, str]] = {
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
VISIBLE_FACILITY_ASSET_PREFIXES = (
    "facility.",
    "prop.traffic_control.",
    "prop.roadwork.",
    "semantic.trigger_box.",
)
PRESERVED_ENTITY_FIELDS = (
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
    "lifecycle",
    "activation_tick",
    "spawn_policy",
    "category",
    "route_waypoints_enu_m",
    "task_kind",
    "task_state",
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
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def heading_deg_from_velocity(velocity_enu_mps: Sequence[float], fallback_deg: float = 0.0) -> float:
    vx = float(velocity_enu_mps[0] if len(velocity_enu_mps) > 0 else 0.0)
    vy = float(velocity_enu_mps[1] if len(velocity_enu_mps) > 1 else 0.0)
    if abs(vx) <= 1e-6 and abs(vy) <= 1e-6:
        return fallback_deg
    return math.degrees(math.atan2(vy, vx))


def truth_pose(position_enu_m: Sequence[float], yaw_deg: float, velocity_enu_mps: Sequence[float]) -> dict[str, Any]:
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
        raise RuntimeError(f"Missing deterministic entity profile for label_class={label_class or '<empty>'}")
    return dict(profile)


def profile_for_entity(source_entry: dict[str, Any], first_row: dict[str, Any]) -> dict[str, str]:
    label_class = str(source_entry.get("label_class") or first_row.get("label_class") or "")
    asset_id = str(
        source_entry.get("logical_asset_id")
        or source_entry.get("asset_id")
        or first_row.get("logical_asset_id")
        or first_row.get("asset_id")
        or ""
    ).strip()
    if asset_id in LOGICAL_ONLY_ASSET_IDS:
        profile = logical_only_profile(label_class)
        profile["entity_category"] = str(source_entry.get("category") or first_row.get("category") or "facility")
        profile["entity_kind"] = f"{profile['entity_category']}.{label_class or 'logical'}"
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
    asset_id = str(source_entry.get("logical_asset_id") or source_entry.get("asset_id") or "").strip()
    if asset_id and asset_id.lower() not in {"unknown", "none", "null"}:
        return asset_id
    return str(profile.get("logical_asset_id") or "")


def value_is_present(value: Any) -> bool:
    return value is not None and value != ""


def preserved_fields_from(*sources: dict[str, Any]) -> dict[str, Any]:
    preserved: dict[str, Any] = {}
    for field in PRESERVED_ENTITY_FIELDS:
        for source in sources:
            if field in source and value_is_present(source[field]):
                preserved[field] = copy.deepcopy(source[field])
                break
            nested_state = source.get("initial_state")
            if isinstance(nested_state, dict) and field in nested_state and value_is_present(nested_state[field]):
                preserved[field] = copy.deepcopy(nested_state[field])
                break
            nested_visual_state = source.get("visual_state")
            if isinstance(nested_visual_state, dict) and field in nested_visual_state and value_is_present(nested_visual_state[field]):
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
                if isinstance(nested, dict) and field in nested and value_is_present(nested[field]):
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
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
        return {str(entity.get("entity_id")): dict(entity) for entity in payload["entities"]}
    if isinstance(payload, dict):
        return {str(entity_id): dict(value) for entity_id, value in payload.items() if isinstance(value, dict)}
    raise ValueError(f"Unsupported roster format: {path}")


def rows_by_entity(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        entity_id = str(row.get("entity_id") or "")
        if entity_id:
            grouped[entity_id].append(dict(row))
    for entity_rows in grouped.values():
        entity_rows.sort(key=lambda row: int(row.get("tick", 0)))
    return dict(grouped)


def sample_row_at_tick(entity_rows: Sequence[dict[str, Any]], tick: int, tick_hz: int) -> dict[str, Any]:
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


def normalized_position_for_render(row: dict[str, Any]) -> list[float]:
    return normalize_vector3(row.get("pos_enu"))


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


def build_activity_overrides(event_rows: Sequence[dict[str, Any]], roster: dict[str, dict[str, Any]]) -> dict[str, list[tuple[int, str]]]:
    overrides: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in event_rows:
        tick = int(row.get("tick", row.get("activated_tick", 0)) or 0)
        text = event_text(row)
        targets = target_ids_from_event(row)
        for entity_id in targets:
            label = str((roster.get(entity_id) or {}).get("label_class") or "")
            if label == "pedestrian" and any(token in text for token in ("fall", "fallen", "medical", "injury", "摔", "倒")):
                overrides[entity_id].append((tick, "medical_incident"))
            elif label == "pedestrian" and any(token in text for token in ("evac", "疏散")):
                overrides[entity_id].append((tick, "walking"))
            elif label == "vehicle" and any(token in text for token in ("stop", "blocked", "breakdown", "停车", "抛锚")):
                overrides[entity_id].append((tick, "stopped"))
    for values in overrides.values():
        values.sort(key=lambda item: item[0])
    return dict(overrides)


def activity_for_sample(
    *,
    entity_id: str,
    category: str,
    tick: int,
    state: str,
    row_activity_type: str = "",
    velocity_enu_mps: Sequence[float],
    overrides: dict[str, list[tuple[int, str]]],
) -> str:
    # The batch generator writes explicit pedestrian activity fields when the
    # source scenario carries semantic activity actions. Prefer those over text
    # inference so render-ready output remains driven by generated truth data.
    # Non-pedestrian rows keep the historical speed/state fallback below.
    speed_xy = math.hypot(float(velocity_enu_mps[0]), float(velocity_enu_mps[1]))
    state_text = str(state or "").strip().lower()
    if category == "pedestrian":
        row_activity = str(row_activity_type or "").strip().lower()
        if row_activity:
            return normalize_activity_type(row_activity, moving=speed_xy > 0.15)
    activity = ""
    for override_tick, override_activity in overrides.get(entity_id, []):
        if tick >= override_tick:
            activity = override_activity
        else:
            break
    if activity:
        return activity
    if category == "pedestrian":
        row_activity = state_text if state_text not in {"moving", "idle"} else ""
        if row_activity:
            return normalize_activity_type(row_activity, moving=speed_xy > 0.15)
        return "walking" if speed_xy > 0.15 or state_text == "moving" else "waiting"
    if category == "uav":
        return "flight" if speed_xy > 0.1 or state_text == "moving" else "idle"
    if category == "vehicle":
        return "moving" if speed_xy > 0.15 or state_text == "moving" else "idle"
    return state_text or "idle"


def build_annotations(activity_type: str, row: dict[str, Any], category: str) -> dict[str, Any]:
    speed_mps = math.sqrt(sum(float(value) ** 2 for value in normalize_vector3(row.get("vel_mps"))))
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
    if category == "traffic_light":
        annotations["signal"] = {"phase": "green"}
    return annotations


def normalize_weather_row(row: dict[str, Any], tick: int) -> dict[str, Any]:
    rain = float(row.get("rain", 0.0) or 0.0)
    fog_density = float(row.get("fog_density", row.get("fog", 0.0)) or 0.0)
    condition = str(row.get("condition") or "").strip().lower()
    if not condition:
        condition = "rain" if rain >= 0.3 else ("fog" if fog_density >= 0.3 else "clear")
    return {
        "tick": int(tick),
        "condition": condition,
        "rain": rain,
        "wetness": float(row.get("wetness", min(1.0, rain)) or 0.0),
        "fog_density": fog_density,
        "dust": float(row.get("dust", 0.0) or 0.0),
        "wind_speed": float(row.get("wind_speed", row.get("wind_mps", 2.0)) or 0.0),
        "visibility_m": float(row.get("visibility_m", 20000.0) or 20000.0),
    }


def expand_weather_rows(source_rows: Sequence[dict[str, Any]], ticks: Sequence[int]) -> list[dict[str, Any]]:
    if not ticks:
        return []
    rows_by_tick = {int(row.get("tick", 0)): dict(row) for row in source_rows}
    source_ticks = sorted(rows_by_tick)
    result: list[dict[str, Any]] = []
    for tick in ticks:
        if tick in rows_by_tick:
            row = rows_by_tick[tick]
        elif source_ticks:
            index = bisect.bisect_right(source_ticks, tick) - 1
            row = rows_by_tick[source_ticks[index]] if index >= 0 else rows_by_tick[source_ticks[0]]
        else:
            row = {}
        result.append(normalize_weather_row(row, tick))
    return result


def build_dynamic_labels(event_rows: Sequence[dict[str, Any]], episode_id: str) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for index, row in enumerate(event_rows):
        tick = int(row.get("tick", row.get("activated_tick", 0)) or 0)
        label = {
            "schema_name": "dynamic_label",
            "schema_version": "v1",
            "episode_id": episode_id,
            "label_id": str(row.get("sample_id") or row.get("instance_id") or f"label_{index:04d}"),
            "tick": tick,
            "frame_id": str(row.get("frame_id") or f"tick:{tick}"),
            "source_event_id": str(row.get("source_event_id") or ""),
            "topic": str(row.get("topic") or ""),
            "semantic_class": str(row.get("semantic_class") or "state_event"),
            "target_ids": target_ids_from_event(row),
            "render_hints": dict(row.get("render_hints") or {}),
            "payload": dict(row.get("payload") or {}),
        }
        label.update(preserved_event_fields(row))
        labels.append(
            label
        )
    return labels


def repo_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_manifest_path(value: Any, source_episode_dir: Path, project_root: Path) -> Path | None:
    if not value:
        return None
    raw_path = Path(str(value))
    candidates = [raw_path] if raw_path.is_absolute() else [project_root / raw_path, source_episode_dir / raw_path]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_json_or_none(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return load_json(path)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_source_contract_hash(
    source_episode_dir: Path,
    manifest: dict[str, Any],
    *,
    project_root: Path,
    scenario_id: str,
) -> str:
    scene_setup_path = resolve_manifest_path(manifest.get("source_scene_setup_path"), source_episode_dir, project_root)
    event_script_path = resolve_manifest_path(manifest.get("source_event_script_path"), source_episode_dir, project_root)
    try:
        semantic_contract = contract_payload(get_contract(scenario_id))
    except KeyError:
        semantic_contract = None
    return sha256_json(
        {
            "scene_setup": load_json_or_none(scene_setup_path),
            "event_script": load_json_or_none(event_script_path),
            "semantic_event_contract": semantic_contract,
        }
    )


def scene_setup_entities(scene_setup: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for entity in scene_setup.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("entity_id") or "").strip()
        if entity_id:
            entities[entity_id] = dict(entity)
    return entities


def inspect_route_from_source(inspect_entity: dict[str, Any], inspect_contract: dict[str, Any]) -> list[list[float]]:
    route = None
    for key in ("loop_route_enu_m", "repaired_route_enu_m", "planned_route_enu_m"):
        candidate = inspect_contract.get(key)
        if isinstance(candidate, list):
            route = candidate
            break
    if not isinstance(route, list):
        raise RuntimeError(f"Missing inspect loop route on contract {inspect_entity.get('entity_id')!r}")
    route_points = [point for point in route if isinstance(point, Sequence) and not isinstance(point, (str, bytes))]
    if len(route_points) < 4:
        raise RuntimeError(f"Inspect loop route is too short on entity {inspect_entity.get('entity_id')!r}")
    normalized = [[float(value) for value in point[:3]] for point in route_points]
    first = normalized[0]
    last = normalized[-1]
    if math.sqrt((first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2 + (first[2] - last[2]) ** 2) > 1.0:
        raise RuntimeError(f"Inspect route must be a closed loop on entity {inspect_entity.get('entity_id')!r}")
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
        raise RuntimeError(f"Expected exactly one inspect entity, found {len(inspect_entities)}")
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
    boundary_id = str(boundary.get("boundary_id") or boundary.get("source_entity_id") or "").strip()
    if not boundary_id:
        raise RuntimeError("Missing capture_boundary.boundary_id in source event_script.json")
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


def point_in_polygon_xy(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    x = float(point[0])
    y = float(point[1])
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = float(polygon[index][0]), float(polygon[index][1])
        x2, y2 = float(polygon[(index + 1) % count][0]), float(polygon[(index + 1) % count][1])
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def segment_intersects_polygon_xy(a: Sequence[float], b: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    def ccw(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> bool:
        return (float(p3[1]) - float(p1[1])) * (float(p2[0]) - float(p1[0])) > (float(p2[1]) - float(p1[1])) * (float(p3[0]) - float(p1[0]))

    if point_in_polygon_xy(a, polygon) or point_in_polygon_xy(b, polygon):
        return True
    for index in range(len(polygon)):
        c = polygon[index]
        d = polygon[(index + 1) % len(polygon)]
        if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
            return True
    return False


def route_crosses_boundary(route: Sequence[Sequence[float]], polygon: Sequence[Sequence[float]]) -> bool:
    if not route or not polygon:
        return False
    if any(point_in_polygon_xy(point, polygon) for point in route):
        return True
    return any(segment_intersects_polygon_xy(a, b, polygon) for a, b in zip(route, route[1:]))


def point_in_oriented_frustum_footprint_xy(
    point: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    half_width_m: float,
    half_height_m: float,
) -> bool:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(point[0]), float(point[1])
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return abs(px - ax) <= half_width_m and abs(py - ay) <= half_height_m
    ux = dx / length
    uy = dy / length
    rel_x = px - ax
    rel_y = py - ay
    along = rel_x * ux + rel_y * uy
    cross = abs(-rel_x * uy + rel_y * ux)
    return -half_height_m <= along <= length + half_height_m and cross <= half_width_m


def source_boundary_polygon(event_script: dict[str, Any]) -> list[list[float]]:
    boundary = source_boundary_payload(event_script)
    polygon = []
    for point in boundary.get("polygon_enu_m") or []:
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 2:
            polygon.append([float(point[0]), float(point[1])])
    if not polygon:
        raise RuntimeError("Source capture_boundary lacks polygon_enu_m")
    return polygon


def source_uav_role(entity: dict[str, Any]) -> str:
    initial_state = dict(entity.get("initial_state") or {})
    if str(initial_state.get("role") or "") == "U_inspect":
        return "inspect"
    corridor_role = str(entity.get("uav_corridor_role") or initial_state.get("uav_corridor_role") or "")
    if corridor_role == "inspect_observer":
        return "inspect"
    if corridor_role == "observer" or "observer" in str(initial_state.get("semantic_role") or ""):
        return "observer"
    return "mission"


def trajectory_route_for_entity(grouped_rows: dict[str, list[dict[str, Any]]], entity_id: str) -> list[list[float]]:
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
        if str(entity.get("logical_asset_id") or "").startswith("uav.") and source_uav_role(entity) != "inspect"
    ]
    if not uavs:
        return False
    for entity in uavs:
        entity_id = str(entity.get("entity_id") or "")
        route = trajectory_route_for_entity(grouped_rows, entity_id)
        if not route_crosses_boundary(route, polygon):
            return False
    return True


def inspect_frustum_observes_boundary(route: list[list[float]], polygon: list[list[float]], inspect_contract: dict[str, Any]) -> bool:
    profile = dict(inspect_contract.get("sensor_profile") or {})
    hfov_deg = float(profile.get("hfov_deg") or profile.get("FOV_Degrees") or profile.get("fov_degrees") or 0.0)
    width = float(profile.get("width") or 0.0)
    height = float(profile.get("height") or 0.0)
    rotation = dict(profile.get("fixed_rotation_offset_deg") or {})
    pitch_deg = float(rotation.get("pitch_deg", 0.0))
    if not route or not polygon or hfov_deg <= 0.0 or width <= 0.0 or height <= 0.0 or pitch_deg > -60.0:
        return False
    altitude_m = float(inspect_contract.get("inspect_altitude_m") or route[0][2])
    half_width_m = math.tan(math.radians(hfov_deg / 2.0)) * altitude_m
    vfov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg / 2.0)) * (height / width)))
    half_height_m = math.tan(math.radians(vfov_deg / 2.0)) * altitude_m
    samples = list(polygon)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    samples.append([sum(xs) / len(xs), sum(ys) / len(ys)])
    segments = list(zip(route, route[1:]))
    return all(
        any(point_in_oriented_frustum_footprint_xy(sample, a, b, half_width_m, half_height_m) for a, b in segments)
        for sample in samples
    )


def capture_contract_from_source(
    *,
    source_episode_dir: Path,
    manifest: dict[str, Any],
    project_root: Path,
    scenario_id: str,
    grouped_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scene_setup_path = resolve_manifest_path(manifest.get("source_scene_setup_path"), source_episode_dir, project_root)
    event_script_path = resolve_manifest_path(manifest.get("source_event_script_path"), source_episode_dir, project_root)
    scene_setup = load_json_or_none(scene_setup_path)
    event_script = load_json_or_none(event_script_path)
    if not isinstance(scene_setup, dict):
        raise RuntimeError(f"Missing scene_setup.json for source episode {source_episode_dir}")
    if not isinstance(event_script, dict):
        raise RuntimeError(f"Missing event_script.json for source episode {source_episode_dir}")

    entities = scene_setup_entities(scene_setup)
    inspect_entity = inspect_entity_from_source(entities)
    inspect_contract = dict(inspect_entity.get("contract_inspect_uav") or {})
    params = dict(event_script.get("parameters") or {})
    semantic_contract = dict(params.get("semantic_event_contract") or {})
    if semantic_contract.get("uav_boundary_crossing_required") is not True:
        raise RuntimeError("Source semantic_event_contract must require UAV boundary crossing")
    if semantic_contract.get("inspect_fov_coverage_required") is not True:
        raise RuntimeError("Source semantic_event_contract must require inspect FoV coverage")
    boundary = source_boundary_payload(event_script)
    if str(boundary.get("geometry_source") or "") != "event_entity":
        raise RuntimeError("Source capture_boundary.geometry_source must be event_entity")
    boundary_entity_id = str(boundary.get("source_entity_id") or boundary.get("boundary_id") or "")
    if boundary_entity_id and not any(str(entity.get("entity_id") or "") == boundary_entity_id for entity in entities.values()):
        raise RuntimeError(f"Source capture boundary entity is not declared in scene_setup: {boundary_entity_id}")
    polygon = source_boundary_polygon(event_script)
    inspect_route = inspect_route_from_source(inspect_entity, inspect_contract)
    inspect_altitude_m = float(inspect_contract.get("inspect_altitude_m") or inspect_route[0][2])
    if any(abs(float(point[2]) - inspect_altitude_m) > 0.001 for point in inspect_route):
        raise RuntimeError("Source inspect route must stay at fixed inspect altitude")
    uav_crosses_boundary = source_uavs_cross_boundary(scene_setup, grouped_rows, polygon)
    inspect_observes_boundary = inspect_frustum_observes_boundary(inspect_route, polygon, inspect_contract)

    return {
        "capture_boundary_id": source_capture_boundary_id(event_script),
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
        state = str(entity.get("state") or entity.get("task_state") or entity.get("role") or "idle")
        if str(entity.get("entity_category") or "").lower() == "uav":
            motion_state[entity_id] = state
    return {
        "capture_boundary_id": capture_boundary_id,
        "uav_crosses_boundary": bool(uav_crosses_boundary),
        "inspect_observes_boundary": bool(inspect_observes_boundary),
        "pad_boundary_policy": pad_boundary_policy,
        "entity_motion_state": motion_state,
    }


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
) -> dict[str, Any]:
    source_episode_dir = source_episode_dir.resolve()
    output_episode_dir = output_episode_dir.resolve()
    if output_episode_dir.exists() and not overwrite:
        required = ["truth_frames.jsonl", "scenario_plan.json", "global_entity_roster.json"]
        if all((output_episode_dir / name).exists() for name in required):
            return {
                "episode_dir": str(output_episode_dir),
                "skipped": True,
                "reason": "render-ready outputs already exist",
            }

    manifest = load_json(source_episode_dir / "episode_manifest.json")
    episode_id = str(manifest.get("episode_id") or source_episode_dir.name)
    scenario_id = str(manifest.get("scenario_id") or source_episode_dir.name)
    duration_ticks = int(manifest.get("duration_ticks") or 0)
    source_contract_hash = stable_source_contract_hash(
        source_episode_dir,
        manifest,
        project_root=project_root,
        scenario_id=scenario_id,
    )

    trajectory_rows = load_jsonl(source_episode_dir / "trajectories.jsonl")
    event_rows = load_jsonl(source_episode_dir / "event_trace.jsonl")
    source_weather_rows = load_jsonl(source_episode_dir / "weather_meta.jsonl")
    source_roster = read_source_roster(source_episode_dir / "global_entity_roster.json")

    grouped = rows_by_entity(trajectory_rows)
    if not grouped:
        raise RuntimeError(f"No trajectory rows found in {source_episode_dir}")
    missing_roster = sorted(entity_id for entity_id in grouped if entity_id not in source_roster)
    if missing_roster:
        raise RuntimeError(f"{source_episode_dir.name}: trajectory entities missing from source roster: {missing_roster}")
    max_traj_tick = max(int(row.get("tick", 0)) for row in trajectory_rows)
    if duration_ticks <= 0:
        duration_ticks = max_traj_tick
    ticks = list(range(0, duration_ticks + 1))
    source_contract = capture_contract_from_source(
        source_episode_dir=source_episode_dir,
        manifest=manifest,
        project_root=project_root,
        scenario_id=scenario_id,
        grouped_rows=grouped,
    )
    if not source_contract["uav_crosses_boundary"]:
        raise RuntimeError(f"{source_episode_dir.name}: source trajectories do not prove all mission/observer UAVs cross capture boundary")
    if not source_contract["inspect_observes_boundary"]:
        raise RuntimeError(f"{source_episode_dir.name}: source inspect route/sensor profile does not prove boundary observation")

    activity_overrides = build_activity_overrides(event_rows, source_roster)
    roster_entities: list[dict[str, Any]] = []
    first_samples: dict[str, dict[str, Any]] = {}
    last_yaw_by_entity: dict[str, float] = {}

    for entity_id in sorted(grouped):
        source_entry = dict(source_roster[entity_id])
        label_class = str(source_entry.get("label_class") or grouped[entity_id][0].get("label_class") or "")
        first_row = sample_row_at_tick(grouped[entity_id], ticks[0], tick_hz)
        profile = profile_for_entity(source_entry, first_row)
        source_asset_id = str(
            source_entry.get("asset_id")
            or source_entry.get("logical_asset_id")
            or profile.get("logical_asset_id")
            or ""
        )
        if not source_asset_id:
            raise RuntimeError(f"{source_episode_dir.name}: source roster entry lacks asset id: {entity_id}")
        first_position = normalized_position_for_render(first_row)
        first_velocity = normalize_vector3(first_row.get("vel_mps"))
        first_yaw = heading_deg_from_velocity(first_velocity)
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
        if str(roster_entry.get("role") or "") == "semantic_facility" and profile["entity_category"] == "facility":
            roster_entry["entity_category"] = "facility"
            roster_asset_id = str(roster_entry.get("asset_id") or source_entry.get("asset_id") or "")
            roster_entry["entity_kind"] = "facility.landing_pad" if roster_asset_id.startswith("facility.landing_pad") else roster_entry["entity_kind"]
        if entity_id == source_contract["inspect_entity_id"]:
            roster_entry["route_waypoints_enu_m"] = [list(point) for point in source_contract["inspect_route_enu_m"]]
        roster_entities.append(roster_entry)

    truth_frames: list[dict[str, Any]] = []
    for tick in ticks:
        entities: list[dict[str, Any]] = []
        for roster_entry in roster_entities:
            entity_id = str(roster_entry["entity_id"])
            profile = profile_for_entity(roster_entry, grouped[entity_id][0])
            category = str(profile["entity_category"])
            row = sample_row_at_tick(grouped[entity_id], tick, tick_hz)
            position = normalized_position_for_render(row)
            velocity = normalize_vector3(row.get("vel_mps"))
            yaw = heading_deg_from_velocity(velocity, fallback_deg=last_yaw_by_entity.get(entity_id, 0.0))
            if math.hypot(velocity[0], velocity[1]) > 1e-4:
                last_yaw_by_entity[entity_id] = yaw
            activity_type = activity_for_sample(
                entity_id=entity_id,
                category=category,
                tick=tick,
                state=str(row.get("state") or ""),
                row_activity_type=str(row.get("activity_type") or ""),
                velocity_enu_mps=velocity,
                overrides=activity_overrides,
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
            entity.update(preserved_fields_from(row, roster_entry))
            if entity_id == source_contract["inspect_entity_id"]:
                entity["route_waypoints_enu_m"] = [list(point) for point in source_contract["inspect_route_enu_m"]]
            if str(entity.get("role") or "") == "semantic_facility" and category == "facility":
                entity["entity_category"] = "facility"
            if row.get("state") not in (None, ""):
                entity["state"] = row.get("state")
            entities.append(entity)

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
                "inspect_observes_boundary": boundary_summary["inspect_observes_boundary"],
                "pad_boundary_policy": boundary_summary["pad_boundary_policy"],
                "entity_motion_state": boundary_summary["entity_motion_state"],
                "roster_summary": {
                    "total": len(entities),
                    "by_category": dict(sorted(counts.items())),
                },
                "entities": entities,
            }
        )

    output_episode_dir.mkdir(parents=True, exist_ok=True)
    for name in ("trajectories.jsonl", "event_trace.jsonl"):
        source_path = source_episode_dir / name
        if source_path.exists():
            shutil.copy2(source_path, output_episode_dir / name)

    weather_rows = expand_weather_rows(source_weather_rows, ticks)
    dynamic_labels = build_dynamic_labels(event_rows, episode_id)
    entity_counts = Counter(str(entity.get("entity_category") or "") for entity in roster_entities)
    xs = [
        float(entity["truth_pose"]["position_enu_m"][0])
        for frame in truth_frames
        for entity in frame["entities"]
    ]
    ys = [
        float(entity["truth_pose"]["position_enu_m"][1])
        for frame in truth_frames
        for entity in frame["entities"]
    ]
    margin_m = 25.0
    roi_window = {
        "roi_id": roi_id,
        "site_id": site_id,
        "tick_start": ticks[0],
        "tick_end": ticks[-1],
        "bbox_enu_m": [
            round(min(xs) - margin_m, 3),
            round(min(ys) - margin_m, 3),
            round(max(xs) + margin_m, 3),
            round(max(ys) + margin_m, 3),
        ] if xs and ys else [],
    }
    site_contract = {
        "site_id": site_id,
        "roi_id": roi_id,
        "suggested_roi_id": roi_id,
        "tick_start": ticks[0],
        "tick_end": ticks[-1],
        "entity_count": len(roster_entities),
        "capture_boundary_id": source_contract["capture_boundary_id"],
        "uav_crosses_boundary": source_contract["uav_crosses_boundary"],
        "inspect_observes_boundary": source_contract["inspect_observes_boundary"],
        "pad_boundary_policy": source_contract["pad_boundary_policy"],
    }
    compiled_summary = {
        "site_contracts": {site_id: site_contract},
        "roi_windows": {roi_id: roi_window},
        "entity_counts_by_category": dict(sorted(entity_counts.items())),
        "event_count": len(event_rows),
        "capture_boundary_id": source_contract["capture_boundary_id"],
        "uav_crosses_boundary": source_contract["uav_crosses_boundary"],
        "inspect_observes_boundary": source_contract["inspect_observes_boundary"],
        "pad_boundary_policy": source_contract["pad_boundary_policy"],
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
        "runtime_contract": {
            "tick_hz": tick_hz,
            "dt_s": round(1.0 / float(tick_hz), 6),
            "tick_start": ticks[0],
            "tick_end": ticks[-1],
        },
        "compiled_plan_summary": compiled_summary,
        "global_entity_roster": roster_entities,
        "export_contract": {
            "artifacts": {
                "scenario_plan": "scenario_plan.json",
                "global_entity_roster": "global_entity_roster.json",
                "truth_frames": "truth_frames.jsonl",
                "event_trace": "event_trace.jsonl",
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
    record_counts = {
        "scenario_plan": 1,
        "global_entity_roster": len(roster_entities),
        "truth_frames": len(truth_frames),
        "event_trace": len(event_rows),
        "trajectories": len(trajectory_rows),
        "weather_meta": len(weather_rows),
        "dynamic_labels": len(dynamic_labels),
        "episode_manifest": 1,
    }
    artifacts = {
        "scenario_plan": repo_relative(output_episode_dir / "scenario_plan.json", project_root),
        "global_entity_roster": repo_relative(output_episode_dir / "global_entity_roster.json", project_root),
        "truth_frames": repo_relative(output_episode_dir / "truth_frames.jsonl", project_root),
        "event_trace": repo_relative(output_episode_dir / "event_trace.jsonl", project_root),
        "trajectories": repo_relative(output_episode_dir / "trajectories.jsonl", project_root),
        "weather_meta": repo_relative(output_episode_dir / "weather_meta.jsonl", project_root),
        "dynamic_labels": repo_relative(output_episode_dir / "dynamic_labels.jsonl", project_root),
        "episode_manifest": repo_relative(output_episode_dir / "episode_manifest.json", project_root),
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
            "generation": {
                "generator": "Dataset/tools/convert_to_render_ready.py",
                "source_episode_dir": repo_relative(source_episode_dir, project_root),
                "source_contract_hash": source_contract_hash,
            },
            "record_counts": record_counts,
            "canonical_record_counts": copy.deepcopy(record_counts),
            "node_counts": {
                "all_nodes": len(roster_entities),
                "dynamic_nodes": len(grouped),
                "static_nodes": max(0, len(roster_entities) - len(grouped)),
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

    write_json(output_episode_dir / "global_entity_roster.json", {"entities": roster_entities})
    write_jsonl(output_episode_dir / "truth_frames.jsonl", truth_frames)
    write_jsonl(output_episode_dir / "weather_meta.jsonl", weather_rows)
    write_jsonl(output_episode_dir / "dynamic_labels.jsonl", dynamic_labels)
    write_json(output_episode_dir / "scenario_plan.json", scenario_plan)
    write_json(output_episode_dir / "episode_manifest.json", render_manifest)
    write_json(
        output_episode_dir / "scenario_package.json",
        {
            "scenario_id": scenario_id,
            "episode_id": episode_id,
            "root_dir": repo_relative(output_episode_dir, project_root),
            "truth_frames": repo_relative(output_episode_dir / "truth_frames.jsonl", project_root),
            "weather_meta": repo_relative(output_episode_dir / "weather_meta.jsonl", project_root),
            "scenario_plan": repo_relative(output_episode_dir / "scenario_plan.json", project_root),
            "capture_plan": "",
            "episode_manifest": repo_relative(output_episode_dir / "episode_manifest.json", project_root),
        },
    )
    return {
        "episode_id": episode_id,
        "source_episode_dir": str(source_episode_dir),
        "episode_dir": str(output_episode_dir),
        "record_counts": record_counts,
        "skipped": False,
    }


def default_output_dir(source_episode_dir: Path, output_root: Path) -> Path:
    return output_root / source_episode_dir.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Dataset episodes into episode_render_host-ready packages.")
    parser.add_argument("--episode", type=Path, help="One Dataset episode directory to convert")
    parser.add_argument("--episodes-root", type=Path, default=Path("Dataset/episodes"), help="Source Dataset episodes root")
    parser.add_argument("--output-root", type=Path, default=Path("Dataset/render_ready_episodes"), help="Render-ready output root")
    parser.add_argument("--map-id", default=DEFAULT_MAP_ID)
    parser.add_argument("--site-id", default=DEFAULT_SITE_ID)
    parser.add_argument("--roi-id", default=DEFAULT_ROI_ID)
    parser.add_argument("--tick-hz", type=int, default=DEFAULT_TICK_HZ)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--all", action="store_true", help="Convert every directory under --episodes-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.episode:
        episodes = [args.episode]
    elif args.all:
        episodes = sorted(path for path in args.episodes_root.iterdir() if path.is_dir())
    else:
        raise SystemExit("Specify --episode or --all.")

    results: list[dict[str, Any]] = []
    for source_episode_dir in episodes:
        result = convert_episode(
            source_episode_dir,
            default_output_dir(source_episode_dir, args.output_root),
            project_root=project_root,
            map_id=args.map_id,
            site_id=args.site_id,
            roi_id=args.roi_id,
            tick_hz=max(1, int(args.tick_hz)),
            overwrite=bool(args.overwrite),
        )
        results.append(result)
        status = "skipped" if result.get("skipped") else "converted"
        print(f"[convert_to_render_ready] {status}: {source_episode_dir} -> {result['episode_dir']}")

    print(json.dumps({"count": len(results), "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
