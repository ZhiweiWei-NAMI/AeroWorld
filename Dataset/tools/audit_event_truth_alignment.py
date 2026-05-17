#!/usr/bin/env python3
"""Audit event_script events against render-ready truth frames.

This validator is intentionally event-centric.  It checks every event instance
in every selected episode and verifies that the event trace, trigger condition,
target entities, and truth-frame motion/state evidence agree deterministically.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover - optional speedup.
    orjson = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
TICK_HZ = 10
PVU_CATEGORIES = {"pedestrian", "vehicle", "uav"}
RUNTIME_SPATIAL_CROP_POLICY = "roi_polygon_expanded_60m_runtime_truth_crop_v1"
TRACE_ONLY_ACTIONS = {"capture_screenshot"}
WAYPOINT_TOLERANCE_M = 5.0
SPAWN_TOLERANCE_M = 3.0
PROXIMITY_TOLERANCE_M = 1.0
STATE_WINDOW_TICKS = 60
MOVE_WINDOW_EXTRA_TICKS = 40
LIFECYCLE_INTENTS = {
    "uav_takes_off_from_visible_home_pad_and_enters_mission_airspace",
    "landing_or_terminal_resolution",
}


def pad_boundary_policy_from_event(event: dict[str, Any], row: dict[str, Any]) -> Any:
    log_event = dict(event.get("log_event") or {})
    for payload in (
        event,
        log_event,
        row,
        dict(row.get("metadata") or {}),
        dict(row.get("payload") or {}),
    ):
        policy = payload.get("pad_boundary_policy")
        if policy not in (None, ""):
            return policy
    return {}


def event_target_roles(event: dict[str, Any], row: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    log_event = dict(event.get("log_event") or {})
    for payload in (
        event,
        log_event,
        row,
        dict(row.get("metadata") or {}),
        dict(row.get("payload") or {}),
    ):
        for role in payload.get("target_roles") or []:
            role_text = str(role or "").strip().lower()
            if role_text:
                roles.add(role_text)
    return roles


def target_id_is_landing_pad(target_id: str) -> bool:
    lowered = target_id.lower()
    return lowered.startswith("pad_") or "landing_pad" in lowered or "landing-pad" in lowered


def event_allows_external_landing_pad_target(event: dict[str, Any], row: dict[str, Any], target_id: str) -> bool:
    if not target_id_is_landing_pad(target_id):
        return False
    roles = event_target_roles(event, row)
    if roles and "landing_pad" not in roles:
        return False
    policy = pad_boundary_policy_from_event(event, row)
    if isinstance(policy, str):
        default_policy = policy
        inside_required_for: list[Any] = []
    elif isinstance(policy, dict):
        default_policy = str(policy.get("default") or "")
        inside_required_for = list(policy.get("inside_required_for") or [])
    else:
        default_policy = ""
        inside_required_for = []
    required = {str(value or "").strip().lower() for value in inside_required_for if str(value or "").strip()}
    if target_id.lower() in required or "landing_pad" in required or "all" in required:
        return False
    return default_policy.lower() == "outside_allowed"


if str(ROOT / "Dataset" / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "Dataset" / "tools"))

from semantic_event_contract import get_contract, required_intent_sequence_matches  # noqa: E402


@dataclass(frozen=True)
class EntitySample:
    tick: int
    position: tuple[float, float, float]
    state: str
    activity_type: str
    category: str
    evidence_values: tuple[str, ...]


@dataclass
class EpisodeTruth:
    episode_id: str
    scenario_id: str
    duration_ticks: int
    frames_seen: set[int]
    entities_by_id: dict[str, dict[int, EntitySample]]
    categories_any: set[str]
    categories_first: set[str]
    frame_categories: dict[int, set[str]]

    def entity_ticks(self, entity_id: str) -> list[int]:
        return sorted(self.entities_by_id.get(entity_id, {}))

    def sample(self, entity_id: str, tick: int) -> EntitySample | None:
        return self.entities_by_id.get(entity_id, {}).get(tick)

    def samples_in_window(self, entity_id: str, start_tick: int, end_tick: int) -> list[EntitySample]:
        rows = self.entities_by_id.get(entity_id, {})
        if not rows:
            return []
        return [rows[tick] for tick in sorted(rows) if start_tick <= tick <= end_tick]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def loads_jsonl(raw: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8-sig"))


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, loads_jsonl(stripped)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc


def pos3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2] if len(value) > 2 else 0.0),
        )
    except (TypeError, ValueError):
        return None


def truth_position(entity: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = entity.get("truth_pose") or {}
    return pos3(pose.get("position_enu_m") or pose.get("position_m") or entity.get("pos_enu"))


def entity_category(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_category") or entity.get("category") or entity.get("label_class") or "").strip().lower()


def entity_activity(entity: dict[str, Any]) -> str:
    annotations = entity.get("annotations") or {}
    activity = annotations.get("activity_type")
    if activity:
        return str(activity)
    facets = annotations.get("state_facets") or {}
    activity_facet = facets.get("activity") or {}
    return str(activity_facet.get("activity_type") or "")


def normalize_evidence_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def entity_evidence_values(entity: dict[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()

    def add(value: Any) -> None:
        normalized = normalize_evidence_text(value)
        if normalized:
            values.add(normalized)

    for key in (
        "state",
        "role",
        "semantic_role",
        "background_role",
        "task_state",
        "task_kind",
        "source",
        "category",
        "entity_category",
        "label_class",
    ):
        add(entity.get(key))
    for value in entity.get("state_sequence") or []:
        add(value)
    for value in entity.get("tags") or []:
        add(value)
    annotations = entity.get("annotations") or {}
    add(annotations.get("activity_type"))
    facets = annotations.get("state_facets") or {}
    if isinstance(facets, dict):
        for facet in facets.values():
            if isinstance(facet, dict):
                for value in facet.values():
                    add(value)
    return tuple(sorted(values))


def expected_value_present(expected: str, samples: Sequence[EntitySample]) -> bool:
    expected_norm = normalize_evidence_text(expected)
    if not expected_norm:
        return True
    for sample in samples:
        if expected_norm == normalize_evidence_text(sample.state):
            return True
        if expected_norm == normalize_evidence_text(sample.activity_type):
            return True
        if expected_norm in sample.evidence_values:
            return True
    return False


def distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def distance_xyz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(distance_xyz(a, b) for a, b in zip(points, points[1:]))


def sample_path_length(samples: Sequence[EntitySample]) -> float:
    return path_length([sample.position for sample in samples])


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


def distance_point_to_segment_xy(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
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


def distance_to_polygon_xy(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    if not polygon:
        return float("inf")
    if point_in_polygon_xy(point, polygon):
        return 0.0
    return min(
        distance_point_to_segment_xy(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def segment_intersects_polygon_xy(a: Sequence[float], b: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    if not polygon:
        return False
    if point_in_polygon_xy(a, polygon) or point_in_polygon_xy(b, polygon):
        return True
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    for index in range(len(polygon)):
        cx, cy = float(polygon[index][0]), float(polygon[index][1])
        dx, dy = float(polygon[(index + 1) % len(polygon)][0]), float(polygon[(index + 1) % len(polygon)][1])
        denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
        if abs(denom) <= 1e-12:
            continue
        t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom
        u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return True
    return False


def route_crosses_polygon(route: Sequence[Sequence[float]], polygon: Sequence[Sequence[float]]) -> bool:
    if not route or not polygon:
        return False
    if any(point_in_polygon_xy(point, polygon) for point in route):
        return True
    return any(segment_intersects_polygon_xy(a, b, polygon) for a, b in zip(route, route[1:]))


def event_topic(event: dict[str, Any]) -> str:
    log_event = event.get("log_event") or {}
    return str(log_event.get("topic") or event.get("topic") or event.get("event_id") or "")


def intent_stage_index(event: dict[str, Any]) -> int | None:
    stage = str(event.get("intent_stage") or "")
    prefix = stage.split(".", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def is_lifecycle_intent(event: dict[str, Any]) -> bool:
    return str(event.get("intent") or "") in LIFECYCLE_INTENTS


def row_topic(row: dict[str, Any]) -> str:
    return str(row.get("topic") or row.get("source_event_id") or row.get("event_id") or "")


def event_targets(event: dict[str, Any], trigger: dict[str, Any] | None = None) -> set[str]:
    targets: set[str] = set()
    log_event = event.get("log_event") or {}
    for value in log_event.get("target_ids") or []:
        if value:
            targets.add(str(value))
    for action in event.get("actions") or []:
        entity_id = action.get("entity_id") or action.get("ped_id")
        if entity_id:
            targets.add(str(entity_id))
    if trigger:
        for key in ("entity_a", "entity_b"):
            if trigger.get(key):
                targets.add(str(trigger[key]))
    return targets


def resolve_manifest_path(value: Any, episode_dir: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise RuntimeError(f"{episode_dir.name}: manifest path is empty")
    path = Path(raw)
    if path.is_absolute():
        return path
    for candidate in ((ROOT / path).resolve(), (episode_dir / path).resolve()):
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


def capture_boundary(script: dict[str, Any]) -> tuple[str, list[tuple[float, float]]]:
    params = script.get("parameters") or {}
    contract = params.get("semantic_event_contract") or {}
    boundary = contract.get("capture_boundary") or {}
    boundary_id = str(boundary.get("boundary_id") or "")
    polygon: list[tuple[float, float]] = []
    for raw in boundary.get("polygon_enu_m") or []:
        point = pos3(raw)
        if point is None and isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
            try:
                point = (float(raw[0]), float(raw[1]), 0.0)
            except (TypeError, ValueError):
                point = None
        if point is not None:
            polygon.append((point[0], point[1]))
    return boundary_id, polygon


def load_truth(episode_dir: Path, scenario_id: str, duration_ticks: int) -> EpisodeTruth:
    truth_path = episode_dir / "truth_frames.jsonl"
    entities_by_id: dict[str, dict[int, EntitySample]] = defaultdict(dict)
    frames_seen: set[int] = set()
    categories_any: set[str] = set()
    categories_first: set[str] = set()
    frame_categories: dict[int, set[str]] = {}
    episode_id = episode_dir.name
    for line_number, frame in iter_jsonl(truth_path):
        tick = int(frame.get("tick") if frame.get("tick") is not None else frame.get("frame_seq", line_number - 1))
        frames_seen.add(tick)
        cats: set[str] = set()
        for entity in frame.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("entity_id") or "")
            position = truth_position(entity)
            if not entity_id or position is None:
                continue
            category = entity_category(entity)
            sample = EntitySample(
                tick=tick,
                position=position,
                state=str(entity.get("state") or ""),
                activity_type=entity_activity(entity),
                category=category,
                evidence_values=entity_evidence_values(entity),
            )
            entities_by_id[entity_id][tick] = sample
            if category:
                cats.add(category)
        frame_categories[tick] = cats
        categories_any.update(cats)
        if tick == 0:
            categories_first = set(cats)
    return EpisodeTruth(
        episode_id=episode_id,
        scenario_id=scenario_id,
        duration_ticks=duration_ticks,
        frames_seen=frames_seen,
        entities_by_id=dict(entities_by_id),
        categories_any=categories_any,
        categories_first=categories_first,
        frame_categories=frame_categories,
    )


def load_weather_by_tick(episode_dir: Path) -> dict[int, dict[str, Any]]:
    path = episode_dir / "weather_meta.jsonl"
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for _, row in iter_jsonl(path):
        tick = row.get("tick")
        if tick is None:
            continue
        rows[int(tick)] = row
    return rows


def value_from_weather(row: dict[str, Any], parameter: str) -> float | None:
    for container in (row, row.get("weather") or {}, row.get("weather_state") or {}, row.get("state") or {}):
        if not isinstance(container, dict):
            continue
        if parameter in container:
            try:
                return float(container[parameter])
            except (TypeError, ValueError):
                return None
        if parameter == "fog" and "fog_density" in container:
            try:
                return float(container["fog_density"])
            except (TypeError, ValueError):
                return None
    profile = str(row.get("profile") or row.get("weather_profile") or row.get("condition") or "").lower()
    if profile == parameter:
        return 1.0
    return None


def condition_ok(value: float | None, operator: str, threshold: float) -> bool:
    if value is None:
        return False
    if operator == "gte":
        return value >= threshold
    if operator == "gt":
        return value > threshold
    if operator == "lte":
        return value <= threshold
    if operator == "lt":
        return value < threshold
    if operator in {"eq", "=="}:
        return abs(value - threshold) <= 1e-9
    return False


def entity_distance_at_tick(
    truth: EpisodeTruth,
    entity_a: str,
    entity_b: str,
    tick: int,
    capture_boundary_id: str,
    capture_polygon_enu_m: Sequence[Sequence[float]],
    metric: str = "xy",
) -> float | None:
    sample_a = truth.sample(entity_a, tick)
    if sample_a is None:
        return None
    if entity_b == capture_boundary_id and capture_polygon_enu_m:
        return distance_to_polygon_xy(sample_a.position, capture_polygon_enu_m)
    sample_b = truth.sample(entity_b, tick)
    if sample_b is None:
        if entity_a == capture_boundary_id and capture_polygon_enu_m:
            return distance_to_polygon_xy(sample_b.position, capture_polygon_enu_m) if sample_b else None
        return None
    if metric == "3d" or metric == "xyz":
        return distance_xyz(sample_a.position, sample_b.position)
    return distance_xy(sample_a.position, sample_b.position)


def min_distance_to_point(samples: Sequence[EntitySample], point: Sequence[float]) -> float:
    if not samples:
        return float("inf")
    return min(distance_xyz(sample.position, point) for sample in samples)


def nearest_tick_to_point(samples: Sequence[EntitySample], point: Sequence[float]) -> int | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: distance_xyz(sample.position, point)).tick


def action_window(
    event_tick: int,
    action: dict[str, Any],
    next_same_entity_tick: int | None,
    duration_ticks: int,
) -> tuple[int, int]:
    waypoints = [point for point in (pos3(raw) for raw in action.get("waypoints_enu_m") or []) if point]
    velocity = float(action.get("velocity_mps") or 0.0)
    expected = 0
    if len(waypoints) >= 2 and velocity > 0.0:
        expected = int(math.ceil(path_length(waypoints) / velocity * TICK_HZ))
    # The deterministic episode generator records a tick's truth rows before it
    # executes actions for events firing on that same tick.  The first
    # machine-verifiable effect can therefore be either the event tick or the
    # following tick depending on the upstream source.
    start_tick = min(duration_ticks, max(0, event_tick))
    end_tick = event_tick + max(expected + MOVE_WINDOW_EXTRA_TICKS, STATE_WINDOW_TICKS)
    if next_same_entity_tick is not None:
        end_tick = min(end_tick, max(event_tick, next_same_entity_tick - 1))
    return start_tick, min(duration_ticks, end_tick)


def next_event_tick_for_entity(
    events: Sequence[dict[str, Any]],
    trace_by_event_id: dict[str, dict[str, Any]],
    event_index: int,
    entity_id: str,
    current_tick: int,
) -> int | None:
    candidates: list[int] = []
    for later in events:
        has_entity = any(str(action.get("entity_id") or action.get("ped_id") or "") == entity_id for action in later.get("actions") or [])
        if not has_entity:
            continue
        row = trace_by_event_id.get(str(later.get("event_id") or ""))
        if row and row.get("tick") is not None:
            tick = int(row["tick"])
            if tick > current_tick:
                candidates.append(tick)
    return min(candidates) if candidates else None


def validate_trigger(
    trigger: dict[str, Any],
    trace_tick: int,
    trace_by_event_id: dict[str, dict[str, Any]],
    truth: EpisodeTruth,
    weather_by_tick: dict[int, dict[str, Any]],
    capture_boundary_id: str,
    capture_polygon_enu_m: Sequence[Sequence[float]],
) -> list[str]:
    issues: list[str] = []
    trigger_type = str(trigger.get("type") or "")
    if trigger_type == "tick":
        expected_tick = int(trigger.get("tick") or 0)
        if trace_tick != expected_tick:
            issues.append(f"tick trigger mismatch: trace_tick={trace_tick} expected={expected_tick}")
    elif trigger_type == "event_fired_after":
        predecessor_id = str(trigger.get("event_id") or "")
        predecessor = trace_by_event_id.get(predecessor_id)
        delay_ticks = int(trigger.get("delay_ticks") or 0)
        if predecessor is None:
            issues.append(f"event_fired_after predecessor missing from trace: {predecessor_id}")
        else:
            expected_tick = int(predecessor.get("tick") or predecessor.get("activated_tick") or 0) + delay_ticks
            if trace_tick != expected_tick:
                issues.append(f"event_fired_after tick mismatch: trace_tick={trace_tick} expected={expected_tick}")
    elif trigger_type == "entity_proximity":
        entity_a = str(trigger.get("entity_a") or "")
        entity_b = str(trigger.get("entity_b") or "")
        threshold = float(trigger.get("distance_m") or 0.0)
        operator = str(trigger.get("operator") or "lte")
        metric = str(trigger.get("metric") or "xy").lower()
        sustain = max(1, int(trigger.get("min_true_ticks") or 1))
        start = max(0, trace_tick - sustain + 1)
        missing_ticks: list[int] = []
        failed_ticks: list[tuple[int, float | None]] = []
        for tick in range(start, trace_tick + 1):
            dist = entity_distance_at_tick(truth, entity_a, entity_b, tick, capture_boundary_id, capture_polygon_enu_m, metric)
            if dist is None:
                missing_ticks.append(tick)
            elif not condition_ok(dist, operator, threshold + PROXIMITY_TOLERANCE_M):
                failed_ticks.append((tick, dist))
        if missing_ticks:
            issues.append(f"entity_proximity target missing at ticks {missing_ticks[:5]} for {entity_a}/{entity_b}")
        if failed_ticks:
            preview = ", ".join(f"{tick}:{dist:.2f}" for tick, dist in failed_ticks[:5] if dist is not None)
            issues.append(f"entity_proximity condition false for {entity_a}/{entity_b}, threshold={threshold:.2f}, samples={preview}")
    elif trigger_type == "weather_state":
        parameter = str(trigger.get("parameter") or "")
        operator = str(trigger.get("operator") or "gte")
        threshold = float(trigger.get("value") or 0.0)
        sustain = max(1, int(trigger.get("sustain_ticks") or 1))
        start = trace_tick
        end = min(truth.duration_ticks, trace_tick + sustain - 1)
        failed: list[str] = []
        for tick in range(start, end + 1):
            row = weather_by_tick.get(tick) or {}
            value = value_from_weather(row, parameter)
            if not condition_ok(value, operator, threshold):
                failed.append(f"{tick}:{value}")
        if failed:
            issues.append(f"weather_state condition false for {parameter} {operator} {threshold}: {failed[:5]}")
    elif trigger_type == "composite":
        # Composite timing is validated through the fired event trace and the
        # child triggers that also appear on concrete events.  The event trace is
        # still the deterministic source of the composite activation tick.
        if trace_tick not in truth.frames_seen:
            issues.append(f"composite event tick has no truth frame: {trace_tick}")
    else:
        issues.append(f"unsupported trigger type: {trigger_type}")
    return issues


def validate_action(
    action: dict[str, Any],
    event: dict[str, Any],
    event_index: int,
    events: Sequence[dict[str, Any]],
    trace_tick: int,
    trace_by_event_id: dict[str, dict[str, Any]],
    truth: EpisodeTruth,
    weather_by_tick: dict[int, dict[str, Any]],
    runtime_spatial_crop_enabled: bool = False,
) -> list[str]:
    issues: list[str] = []
    action_type = str(action.get("type") or "")
    action_id = str(action.get("action_id") or action_type)
    if action_type in TRACE_ONLY_ACTIONS:
        return issues
    if action_type == "move_entity":
        entity_id = str(action.get("entity_id") or action.get("ped_id") or "")
        waypoints = [point for point in (pos3(raw) for raw in action.get("waypoints_enu_m") or []) if point]
        if not entity_id:
            return [f"{action_id}: move_entity missing entity_id"]
        if len(waypoints) < 1:
            return [f"{action_id}: move_entity has no valid waypoints"]
        next_tick = next_event_tick_for_entity(events, trace_by_event_id, event_index, entity_id, trace_tick)
        start_tick, end_tick = action_window(trace_tick, action, next_tick, truth.duration_ticks)
        samples = truth.samples_in_window(entity_id, start_tick, end_tick)
        if not samples:
            if runtime_spatial_crop_enabled and truth.entity_ticks(entity_id):
                return issues
            if is_lifecycle_intent(event) and truth.entity_ticks(entity_id):
                return issues
            return [f"{action_id}: target {entity_id} absent from truth window [{start_tick}, {end_tick}]"]
        velocity = float(action.get("velocity_mps") or 0.0)
        expected_ticks = int(math.ceil(path_length(waypoints) / velocity * TICK_HZ)) if velocity > 0.0 else 0
        interrupted = next_tick is not None and expected_ticks > 0 and next_tick < trace_tick + expected_ticks
        expected_path_length = path_length([samples[0].position, *waypoints]) if waypoints else 0.0
        observed_path_length = sample_path_length(samples)
        if expected_path_length > 2.0 and observed_path_length < 1.0 and not is_lifecycle_intent(event):
            issues.append(f"{action_id}: {entity_id} has no measurable motion in truth window [{start_tick}, {end_tick}]")
        terminal = waypoints[-1]
        if terminal[2] <= 8.0 and not interrupted and not is_lifecycle_intent(event):
            broad_samples = truth.samples_in_window(entity_id, trace_tick, truth.duration_ticks)
            if broad_samples and min(sample.position[2] for sample in broad_samples) > max(8.0, terminal[2] + 3.0):
                issues.append(f"{action_id}: {entity_id} never reaches landing altitude near {terminal[2]:.2f}m")
        expected_activity = str(action.get("activity_type") or "")
        if expected_activity and not expected_value_present(expected_activity, samples):
            issues.append(f"{action_id}: {entity_id} missing activity_type={expected_activity} in truth window")
    elif action_type == "set_visual_state":
        entity_id = str(action.get("entity_id") or "")
        mode = str((action.get("visual_state") or {}).get("mode") or "")
        if not entity_id:
            return [f"{action_id}: set_visual_state missing entity_id"]
        next_tick = next_event_tick_for_entity(events, trace_by_event_id, event_index, entity_id, trace_tick)
        end_tick = min(truth.duration_ticks, next_tick - 1 if next_tick is not None else trace_tick + STATE_WINDOW_TICKS)
        samples = truth.samples_in_window(entity_id, trace_tick, end_tick)
        if not samples:
            if not truth.entity_ticks(entity_id) and not runtime_spatial_crop_enabled:
                issues.append(f"{action_id}: target {entity_id} absent from truth state window [{trace_tick}, {end_tick}]")
        elif mode:
            if not expected_value_present(mode, samples):
                issues.append(f"{action_id}: visual mode={mode} not reflected by state/activity in truth window")
    elif action_type == "set_pedestrian_activity":
        entity_id = str(action.get("entity_id") or "")
        activity = str(action.get("activity_type") or "")
        next_tick = next_event_tick_for_entity(events, trace_by_event_id, event_index, entity_id, trace_tick)
        end_tick = min(truth.duration_ticks, next_tick - 1 if next_tick is not None else trace_tick + STATE_WINDOW_TICKS * 4)
        samples = truth.samples_in_window(entity_id, trace_tick, end_tick)
        if not samples:
            if not (runtime_spatial_crop_enabled and truth.entity_ticks(entity_id)):
                issues.append(f"{action_id}: target {entity_id} absent from truth activity window")
        elif activity and not expected_value_present(activity, samples):
            issues.append(f"{action_id}: activity_type={activity} not reflected in truth")
    elif action_type == "spawn_entity":
        entity_id = str(action.get("entity_id") or "")
        position = pos3(action.get("position_enu_m"))
        samples = truth.samples_in_window(entity_id, trace_tick, min(truth.duration_ticks, trace_tick + STATE_WINDOW_TICKS))
        if not samples:
            if not (runtime_spatial_crop_enabled and truth.entity_ticks(entity_id)):
                issues.append(f"{action_id}: spawned entity {entity_id} absent from truth")
        elif position is not None:
            distance = min_distance_to_point(samples, position)
            if distance > SPAWN_TOLERANCE_M:
                issues.append(f"{action_id}: spawned entity {entity_id} position mismatch {distance:.2f}m")
    elif action_type == "set_weather":
        profile = str(action.get("profile") or "")
        overrides = dict(action.get("overrides") or {})
        candidate_ticks = [tick for tick in range(trace_tick, min(truth.duration_ticks, trace_tick + 2) + 1)]
        if profile and profile != "clear":
            if not any((value_from_weather(weather_by_tick.get(tick) or {}, profile) or 0.0) > 0.0 for tick in candidate_ticks):
                issues.append(f"{action_id}: weather profile={profile} not reflected at tick {trace_tick}")
        for key, raw_value in overrides.items():
            if not isinstance(raw_value, (int, float)):
                continue
            parameter = "fog" if key == "fog_density" else str(key)
            matched_value: float | None = None
            for tick in candidate_ticks:
                value = value_from_weather(weather_by_tick.get(tick) or {}, parameter)
                if value is None:
                    continue
                if abs(value - float(raw_value)) <= max(0.05, abs(float(raw_value)) * 0.1):
                    matched_value = value
                    break
                if matched_value is None:
                    matched_value = value
            if matched_value is None:
                issues.append(f"{action_id}: weather override {key} missing at tick {trace_tick}")
            elif abs(matched_value - float(raw_value)) > max(0.05, abs(float(raw_value)) * 0.1):
                issues.append(f"{action_id}: weather override {key}={raw_value} not reflected, found {matched_value}")
    else:
        issues.append(f"{action_id}: unsupported action type {action_type}")
    return issues


def validate_intent_semantics(
    event: dict[str, Any],
    trace_tick: int,
    truth: EpisodeTruth,
    capture_boundary_id: str,
    capture_polygon_enu_m: Sequence[Sequence[float]],
) -> list[str]:
    issues: list[str] = []
    intent = str(event.get("intent") or "")
    log_event = event.get("log_event") or {}
    target_ids = [str(value) for value in log_event.get("target_ids") or [] if value]
    if intent in {"airspace_boundary_conflict", "intruder_conflict", "temporary_no_fly_zone_declared_mid_operation"}:
        moving_targets = [
            entity_id
            for entity_id in target_ids
            if entity_id != capture_boundary_id and any(
                sample.category == "uav" for sample in truth.entities_by_id.get(entity_id, {}).values()
            )
        ]
        if capture_polygon_enu_m and moving_targets:
            start = max(0, trace_tick - 20)
            end = min(truth.duration_ticks, trace_tick + 40)
            for entity_id in moving_targets:
                samples = truth.samples_in_window(entity_id, start, end)
                if not samples:
                    issues.append(f"{intent}: target {entity_id} absent near boundary event")
                    continue
                min_dist = min(distance_to_polygon_xy(sample.position, capture_polygon_enu_m) for sample in samples)
                if min_dist > 25.0:
                    issues.append(f"{intent}: target {entity_id} never approaches capture boundary, min_dist={min_dist:.2f}m")
    if intent in {"landing_or_terminal_resolution", "forced_landing"}:
        move_actions = [action for action in event.get("actions") or [] if action.get("type") == "move_entity"]
        for action in move_actions:
            waypoints = [point for point in (pos3(raw) for raw in action.get("waypoints_enu_m") or []) if point]
            if not waypoints:
                continue
            terminal = waypoints[-1]
            if terminal[2] > 8.0 and "forced" not in intent:
                issues.append(f"{intent}: terminal waypoint altitude {terminal[2]:.2f}m is too high for landing resolution")
    if intent == "pad_contention":
        pad_targets = [entity_id for entity_id in target_ids if "pad" in entity_id.lower()]
        if capture_polygon_enu_m and pad_targets:
            for pad_id in pad_targets:
                samples = truth.samples_in_window(pad_id, trace_tick, min(truth.duration_ticks, trace_tick + STATE_WINDOW_TICKS))
                if not samples:
                    issues.append(f"pad_contention: target pad {pad_id} absent from truth")
                    continue
                if not any(point_in_polygon_xy(sample.position, capture_polygon_enu_m) for sample in samples):
                    issues.append(f"pad_contention: target pad {pad_id} is not inside capture boundary")
    return issues


def validate_episode(episode_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest = read_json(episode_dir / "episode_manifest.json")
    scenario_id = str(manifest.get("scenario_id") or episode_dir.name.split("__seed")[0])
    duration_ticks = int(manifest.get("duration_ticks") or 900)
    runtime_spatial_crop_enabled = (
        str((manifest.get("runtime_spatial_crop") or {}).get("policy") or "") == RUNTIME_SPATIAL_CROP_POLICY
    )
    script_path = resolve_manifest_path(manifest.get("source_event_script_path"), episode_dir)
    script = read_json(script_path)
    events = list(script.get("events") or [])
    triggers = {str(trigger.get("trigger_id") or ""): trigger for trigger in script.get("triggers") or []}
    capture_boundary_id, capture_polygon_enu_m = capture_boundary(script)

    truth = load_truth(episode_dir, scenario_id, duration_ticks)
    weather_by_tick = load_weather_by_tick(episode_dir)
    trace_rows = [row for _, row in iter_jsonl(episode_dir / "event_trace.jsonl")]
    trace_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        trace_by_topic[row_topic(row)].append(row)
    trace_by_event_id: dict[str, dict[str, Any]] = {}
    event_to_trace: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event.get("event_id") or "")
        topic = event_topic(event)
        rows = trace_by_topic.get(topic) or trace_by_topic.get(f"evt_{scenario_id}_{event_id}") or trace_by_topic.get(event_id) or []
        if len(rows) != 1:
            errors.append(f"{event_id}: expected one event_trace row for topic={topic}, found {len(rows)}")
            continue
        row = rows[0]
        event_to_trace[event_id] = row
        trace_by_event_id[event_id] = row

    ordered_trace_rows = sorted(event_to_trace.values(), key=lambda row: (int(row.get("tick") or 0), row_topic(row)))
    try:
        contract = get_contract(scenario_id)
        ok, matched_topics = required_intent_sequence_matches(contract.required_intents, ordered_trace_rows)
        if not ok:
            errors.append(f"required intent order mismatch: required={contract.required_intents} matched={matched_topics}")
    except Exception as exc:
        warnings.append(f"required intent order skipped: {exc}")

    staged_events: list[tuple[int, str, int]] = []
    for event in events:
        stage_index = intent_stage_index(event)
        if stage_index is None or stage_index <= 0:
            continue
        row = event_to_trace.get(str(event.get("event_id") or ""))
        if row is None or row.get("tick") is None:
            continue
        staged_events.append((stage_index, str(event.get("event_id") or ""), int(row["tick"])))
    for left_index, (left_stage, left_event_id, left_tick) in enumerate(staged_events):
        for right_stage, right_event_id, right_tick in staged_events[left_index + 1 :]:
            if right_stage > left_stage and right_tick < left_tick:
                errors.append(
                    f"intent stage order mismatch: {right_event_id} stage={right_stage} tick={right_tick} "
                    f"fires before {left_event_id} stage={left_stage} tick={left_tick}"
                )
                break

    roster_entities = read_json(episode_dir / "global_entity_roster.json").get("entities") or []
    roster_ids = {str(entity.get("entity_id") or "") for entity in roster_entities if isinstance(entity, dict)}
    truth_ids = set(truth.entities_by_id)

    checked_events = 0
    checked_actions = 0
    intent_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    for event_index, event in enumerate(events):
        event_id = str(event.get("event_id") or "")
        row = event_to_trace.get(event_id)
        if row is None:
            continue
        checked_events += 1
        intent = str(event.get("intent") or "")
        intent_counter[intent] += 1
        trace_tick = int(row.get("tick") if row.get("tick") is not None else row.get("activated_tick", -1))
        if trace_tick not in truth.frames_seen:
            errors.append(f"{event_id}: event_trace tick {trace_tick} has no truth frame")
        for key in ("intent", "intent_stage", "causal_chain_id", "causal_predecessor_intent"):
            expected = str(event.get(key) or "")
            actual = str(row.get(key) or (row.get("metadata") or {}).get(key) or (row.get("payload") or {}).get(key) or "")
            if expected and actual != expected:
                errors.append(f"{event_id}: trace {key} mismatch expected={expected} actual={actual}")
        log_event = event.get("log_event") or {}
        log_targets = [str(value) for value in log_event.get("target_ids") or [] if value]
        trace_targets = {str(value) for value in row.get("target_ids") or []}
        if not trace_targets:
            trace_targets = {str(value) for value in ((row.get("scope") or {}).get("entities") or [])}
        missing_trace_targets = sorted(set(log_targets) - trace_targets)
        if missing_trace_targets:
            errors.append(f"{event_id}: trace missing target_ids {missing_trace_targets}")

        trigger = triggers.get(str(event.get("trigger_ref") or ""))
        if not trigger:
            errors.append(f"{event_id}: missing trigger_ref {event.get('trigger_ref')}")
        else:
            for issue in validate_trigger(
                trigger,
                trace_tick,
                trace_by_event_id,
                truth,
                weather_by_tick,
                capture_boundary_id,
                capture_polygon_enu_m,
            ):
                errors.append(f"{event_id}: {issue}")

        for target_id in event_targets(event, trigger):
            if target_id == capture_boundary_id:
                continue
            if target_id not in roster_ids and target_id not in truth_ids:
                if event_allows_external_landing_pad_target(event, row, target_id):
                    continue
                errors.append(f"{event_id}: target {target_id} missing from roster/truth")
                continue

        for action in event.get("actions") or []:
            action_counter[str(action.get("type") or "")] += 1
            checked_actions += 1
            for issue in validate_action(
                action,
                event,
                event_index,
                events,
                trace_tick,
                trace_by_event_id,
                truth,
                weather_by_tick,
                runtime_spatial_crop_enabled=runtime_spatial_crop_enabled,
            ):
                errors.append(f"{event_id}: {issue}")

        for issue in validate_intent_semantics(event, trace_tick, truth, capture_boundary_id, capture_polygon_enu_m):
            errors.append(f"{event_id}: {issue}")

    if not runtime_spatial_crop_enabled:
        missing_categories = sorted(PVU_CATEGORIES - truth.categories_any)
        if missing_categories:
            errors.append(f"episode truth missing PVU categories over capture window: {missing_categories}")
        if "facility" not in truth.categories_first:
            errors.append("first truth frame missing facility")
        if "airspace_corridor" not in truth.categories_first:
            errors.append("first truth frame missing airspace_corridor")

    return {
        "episode_id": episode_dir.name,
        "scenario_id": scenario_id,
        "events_checked": checked_events,
        "events_expected": len(events),
        "actions_checked": checked_actions,
        "frames": len(truth.frames_seen),
        "errors": errors,
        "warnings": warnings,
        "intent_counts": dict(sorted(intent_counter.items())),
        "action_counts": dict(sorted(action_counter.items())),
    }


def parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                names.append(item)
    return names


def selected_episode_dirs(input_root: Path, names: list[str], limit: int) -> list[Path]:
    all_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "truth_frames.jsonl").exists())
    if names:
        by_name = {path.name: path for path in all_dirs}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise SystemExit(f"Unknown episode(s): {', '.join(missing)}")
        all_dirs = [by_name[name] for name in names]
    if limit > 0:
        all_dirs = all_dirs[:limit]
    return all_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit event_script events against render-ready truth frames.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--max-errors", type=int, default=200)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    episode_dirs = selected_episode_dirs(input_root, parse_episode_names(args.episode), int(args.limit or 0))
    if not episode_dirs:
        raise SystemExit("No episodes selected")

    results: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        results.append(validate_episode(episode_dir))

    total_errors = sum(len(result["errors"]) for result in results)
    total_warnings = sum(len(result["warnings"]) for result in results)
    total_events = sum(int(result["events_checked"]) for result in results)
    total_actions = sum(int(result["actions_checked"]) for result in results)
    episodes_with_errors = [result["episode_id"] for result in results if result["errors"]]
    summary = {
        "ok": total_errors == 0,
        "input_root": str(input_root),
        "episodes_checked": len(results),
        "events_checked": total_events,
        "actions_checked": total_actions,
        "errors": total_errors,
        "warnings": total_warnings,
        "episodes_with_errors": episodes_with_errors,
    }
    report = {"summary": summary, "results": results}
    if args.report_json is not None:
        report_path = args.report_json.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if total_errors:
        printed = 0
        for result in results:
            for error in result["errors"]:
                print(f"ERROR {result['episode_id']}: {error}")
                printed += 1
                if printed >= int(args.max_errors):
                    remaining = total_errors - printed
                    if remaining > 0:
                        print(f"... {remaining} additional errors")
                    return 1
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
