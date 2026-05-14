"""Integrate Donghu global UAV flow samples into render-ready truth frames."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UAV_OUTPUT_DIR = ROOT / "Dataset" / "uav_outputs" / "donghu_uav_flow_270s"
SEGMENT_DURATION_S = 90.0
SEGMENT_COUNT = 3


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _round_vector(values: Sequence[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _position3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        z = float(value[2]) if len(value) >= 3 else 0.0
        return float(value[0]), float(value[1]), z
    except (TypeError, ValueError):
        return None


def _position2(value: Any) -> tuple[float, float] | None:
    position = _position3(value)
    if position is None:
        return None
    return position[0], position[1]


def _clean_entity_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def uav_truth_entity_id(uav_id: str) -> str:
    return f"global_uav_{_clean_entity_token(uav_id)}"


def uav_pad_truth_entity_id(pad_id: str) -> str:
    return f"global_{_clean_entity_token(pad_id)}"


def infer_seed_index(episode_id: str, manifest: dict[str, Any]) -> int:
    raw_seed = manifest.get("seed")
    if raw_seed not in (None, ""):
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{episode_id}: invalid manifest seed {raw_seed!r}") from exc
    else:
        match = re.search(r"__seed(\d+)$", episode_id)
        if not match:
            raise ValueError(f"{episode_id}: cannot infer seed index from episode id")
        seed = int(match.group(1))
    if seed < 0 or seed >= SEGMENT_COUNT:
        raise ValueError(f"{episode_id}: UAV flow integration requires seed00, seed01, or seed02; found seed{seed:02d}")
    return seed


@dataclass(frozen=True)
class UavSegment:
    seed_index: int
    segment_start_s: float
    segment_end_s: float
    duration_s: float
    phase_name: str

    @property
    def seed_label(self) -> str:
        return f"seed{self.seed_index:02d}"

    def absolute_time_s(self, episode_sim_time_s: float) -> float:
        local_time = max(0.0, min(self.duration_s, float(episode_sim_time_s)))
        return self.segment_start_s + local_time

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_index": int(self.seed_index),
            "seed_label": self.seed_label,
            "segment_start_s": round(float(self.segment_start_s), 6),
            "segment_end_s": round(float(self.segment_end_s), 6),
            "duration_s": round(float(self.duration_s), 6),
            "phase_name": self.phase_name,
        }


def segment_for_seed(seed_index: int) -> UavSegment:
    phase_names = {
        0: "uav_demand_ramp_up",
        1: "uav_demand_peak_hold",
        2: "uav_demand_taper_with_persistent_patrol",
    }
    start_s = float(seed_index) * SEGMENT_DURATION_S
    return UavSegment(
        seed_index=seed_index,
        segment_start_s=start_s,
        segment_end_s=start_s + SEGMENT_DURATION_S,
        duration_s=SEGMENT_DURATION_S,
        phase_name=phase_names[seed_index],
    )


@dataclass(frozen=True)
class UavSelection:
    uav_ids: tuple[str, ...]
    entity_ids: dict[str, str]
    task_ids: dict[str, str]
    mission_type_by_uav_id: dict[str, str]
    selected_count: int
    active_count_min: int
    active_count_max: int
    active_count_mean: float
    min_distance_m_by_uav_id: dict[str, float] = field(default_factory=dict)
    frames_seen_by_uav_id: dict[str, int] = field(default_factory=dict)
    motion_span_m_by_uav_id: dict[str, float] = field(default_factory=dict)
    max_speed_mps_by_uav_id: dict[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    observable_candidate_count: int = 0
    visibility_padding_m: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "uav_ids": list(self.uav_ids),
            "entity_ids": dict(self.entity_ids),
            "task_ids": dict(self.task_ids),
            "mission_type_by_uav_id": dict(self.mission_type_by_uav_id),
            "selected_count": int(self.selected_count),
            "active_count_min": int(self.active_count_min),
            "active_count_max": int(self.active_count_max),
            "active_count_mean": round(float(self.active_count_mean), 6),
            "candidate_count": int(self.candidate_count),
            "observable_candidate_count": int(self.observable_candidate_count),
            "visibility_padding_m": round(float(self.visibility_padding_m), 6),
            "min_distance_m_by_uav_id": {
                key: round(float(value), 6) for key, value in sorted(self.min_distance_m_by_uav_id.items())
            },
            "frames_seen_by_uav_id": dict(sorted(self.frames_seen_by_uav_id.items())),
            "motion_span_m_by_uav_id": {
                key: round(float(value), 6) for key, value in sorted(self.motion_span_m_by_uav_id.items())
            },
            "max_speed_mps_by_uav_id": {
                key: round(float(value), 6) for key, value in sorted(self.max_speed_mps_by_uav_id.items())
            },
        }


class UavGlobalFlowDataset:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.frames_path = self.output_dir / "uav_traffic_frames.jsonl"
        self.manifest_path = self.output_dir / "uav_flow_manifest.json"
        self.task_plan_path = self.output_dir / "uav_task_plan.json"
        missing = [path for path in (self.frames_path, self.manifest_path, self.task_plan_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing UAV global flow output files: {missing}")
        self.manifest = _read_json(self.manifest_path)
        self.task_plan = _read_json(self.task_plan_path)
        self.frames = sorted(_read_jsonl(self.frames_path), key=lambda row: float(row.get("sim_time_s") or 0.0))
        if not self.frames:
            raise RuntimeError(f"No UAV traffic frames in {self.frames_path}")
        self.times = [float(row.get("sim_time_s") or 0.0) for row in self.frames]
        self.tasks = [dict(item) for item in self.task_plan.get("tasks") or []]
        self.tasks_by_id = {str(item.get("task_id") or ""): item for item in self.tasks}
        self.pads = [dict(item) for item in self.task_plan.get("pads") or []]

    def segment_for_episode(self, episode_id: str, manifest: dict[str, Any]) -> UavSegment:
        return segment_for_seed(infer_seed_index(episode_id, manifest))

    def _frame_pair(self, absolute_time_s: float) -> tuple[dict[str, Any], dict[str, Any], float]:
        time_s = max(self.times[0], min(self.times[-1], float(absolute_time_s)))
        index = bisect.bisect_right(self.times, time_s)
        if index <= 0:
            return self.frames[0], self.frames[0], 0.0
        if index >= len(self.frames):
            return self.frames[-1], self.frames[-1], 0.0
        prev_frame = self.frames[index - 1]
        next_frame = self.frames[index]
        prev_time = float(prev_frame.get("sim_time_s") or 0.0)
        next_time = float(next_frame.get("sim_time_s") or prev_time)
        if abs(time_s - prev_time) <= 1e-9:
            return prev_frame, prev_frame, 0.0
        span = max(1e-9, next_time - prev_time)
        alpha = max(0.0, min(1.0, (time_s - prev_time) / span))
        return prev_frame, next_frame, alpha

    def sample(
        self,
        *,
        segment: UavSegment,
        episode_sim_time_s: float,
        selected_uav_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        absolute_time_s = segment.absolute_time_s(episode_sim_time_s)
        prev_frame, next_frame, alpha = self._frame_pair(absolute_time_s)
        prev_time = float(prev_frame.get("sim_time_s") or absolute_time_s)
        next_time = float(next_frame.get("sim_time_s") or prev_time)
        selected = set(selected_uav_ids or [])
        prev_by_id = {str(item.get("uav_id") or ""): item for item in prev_frame.get("uavs") or []}
        next_by_id = {str(item.get("uav_id") or ""): item for item in next_frame.get("uavs") or []}
        uav_ids = sorted(set(prev_by_id) | set(next_by_id))
        if selected:
            uav_ids = [uav_id for uav_id in uav_ids if uav_id in selected]
        uavs: list[dict[str, Any]] = []
        for uav_id in uav_ids:
            record = _interpolate_uav(prev_by_id.get(uav_id), next_by_id.get(uav_id), alpha, prev_time, next_time)
            if record is not None:
                uavs.append(record)
        return {
            "absolute_time_s": round(absolute_time_s, 6),
            "source_prev_time_s": round(prev_time, 6),
            "source_next_time_s": round(next_time, 6),
            "source_alpha": round(alpha, 6),
            "source_uav_count": len(set(prev_by_id) | set(next_by_id)),
            "uavs": uavs,
        }

    def first_uav_record_in_segment(self, segment: UavSegment, uav_id: str) -> dict[str, Any] | None:
        for frame in self.frames:
            time_s = float(frame.get("sim_time_s") or 0.0)
            if time_s < segment.segment_start_s or time_s > segment.segment_end_s:
                continue
            for uav in frame.get("uavs") or []:
                if str(uav.get("uav_id") or "") == uav_id:
                    return dict(uav)
        return None

    def select_segment_uavs(self, segment: UavSegment) -> UavSelection:
        uav_ids: set[str] = set()
        task_ids: dict[str, str] = {}
        mission_type_by_uav_id: dict[str, str] = {}
        active_counts: list[int] = []
        for frame in self.frames:
            time_s = float(frame.get("sim_time_s") or 0.0)
            if time_s < segment.segment_start_s or time_s > segment.segment_end_s:
                continue
            frame_ids: set[str] = set()
            for uav in frame.get("uavs") or []:
                uav_id = str(uav.get("uav_id") or "")
                if not uav_id:
                    continue
                frame_ids.add(uav_id)
                uav_ids.add(uav_id)
                task_ids[uav_id] = str(uav.get("task_id") or "")
                mission_type_by_uav_id[uav_id] = str(uav.get("mission_type") or "")
            active_counts.append(len(frame_ids))
        ordered = tuple(sorted(uav_ids))
        return UavSelection(
            uav_ids=ordered,
            entity_ids={uav_id: uav_truth_entity_id(uav_id) for uav_id in ordered},
            task_ids={uav_id: task_ids.get(uav_id, "") for uav_id in ordered},
            mission_type_by_uav_id={uav_id: mission_type_by_uav_id.get(uav_id, "") for uav_id in ordered},
            selected_count=len(ordered),
            active_count_min=min(active_counts) if active_counts else 0,
            active_count_max=max(active_counts) if active_counts else 0,
            active_count_mean=(sum(active_counts) / len(active_counts)) if active_counts else 0.0,
            candidate_count=len(ordered),
            observable_candidate_count=len(ordered),
        )

    def select_visible_uavs(self, *, segment: UavSegment, visibility: Any) -> UavSelection:
        min_distance_by_id: dict[str, float] = {}
        frames_seen_by_id: dict[str, int] = {}
        bounds_by_id: dict[str, list[float]] = {}
        max_speed_by_id: dict[str, float] = {}
        task_ids: dict[str, str] = {}
        mission_type_by_uav_id: dict[str, str] = {}
        observable_ids: set[str] = set()
        observable_ids_by_frame: list[set[str]] = []
        padding_m = float(getattr(visibility, "padding_m", 0.0) or 0.0)
        for frame in self.frames:
            time_s = float(frame.get("sim_time_s") or 0.0)
            if time_s < segment.segment_start_s or time_s > segment.segment_end_s:
                continue
            frame_observable_ids: set[str] = set()
            for uav in frame.get("uavs") or []:
                uav_id = str(uav.get("uav_id") or "")
                if not uav_id:
                    continue
                point = _position2(uav.get("position_enu_m"))
                if point is None:
                    continue
                distance_m = float(visibility.observation_distance_m(point))
                min_distance_by_id[uav_id] = min(distance_m, min_distance_by_id.get(uav_id, float("inf")))
                frames_seen_by_id[uav_id] = frames_seen_by_id.get(uav_id, 0) + 1
                if uav_id not in bounds_by_id:
                    bounds_by_id[uav_id] = [point[0], point[0], point[1], point[1]]
                else:
                    bounds = bounds_by_id[uav_id]
                    bounds[0] = min(bounds[0], point[0])
                    bounds[1] = max(bounds[1], point[0])
                    bounds[2] = min(bounds[2], point[1])
                    bounds[3] = max(bounds[3], point[1])
                max_speed_by_id[uav_id] = max(max_speed_by_id.get(uav_id, 0.0), float(uav.get("speed_mps") or 0.0))
                task_ids[uav_id] = str(uav.get("task_id") or "")
                mission_type_by_uav_id[uav_id] = str(uav.get("mission_type") or "")
                if distance_m <= padding_m:
                    observable_ids.add(uav_id)
                    frame_observable_ids.add(uav_id)
            observable_ids_by_frame.append(frame_observable_ids)

        ordered = tuple(sorted(observable_ids))
        selected_set = set(ordered)
        active_counts = [len(frame_ids & selected_set) for frame_ids in observable_ids_by_frame]
        motion_span_by_id = {
            uav_id: math.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2])
            for uav_id, bounds in bounds_by_id.items()
        }
        return UavSelection(
            uav_ids=ordered,
            entity_ids={uav_id: uav_truth_entity_id(uav_id) for uav_id in ordered},
            task_ids={uav_id: task_ids.get(uav_id, "") for uav_id in ordered},
            mission_type_by_uav_id={uav_id: mission_type_by_uav_id.get(uav_id, "") for uav_id in ordered},
            selected_count=len(ordered),
            active_count_min=min(active_counts) if active_counts else 0,
            active_count_max=max(active_counts) if active_counts else 0,
            active_count_mean=(sum(active_counts) / len(active_counts)) if active_counts else 0.0,
            min_distance_m_by_uav_id={uav_id: min_distance_by_id.get(uav_id, float("inf")) for uav_id in ordered},
            frames_seen_by_uav_id={uav_id: frames_seen_by_id.get(uav_id, 0) for uav_id in ordered},
            motion_span_m_by_uav_id={uav_id: motion_span_by_id.get(uav_id, 0.0) for uav_id in ordered},
            max_speed_mps_by_uav_id={uav_id: max_speed_by_id.get(uav_id, 0.0) for uav_id in ordered},
            candidate_count=len(min_distance_by_id),
            observable_candidate_count=len(observable_ids),
            visibility_padding_m=padding_m,
        )

    def source_summary(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "frames": str(self.frames_path),
            "manifest": str(self.manifest_path),
            "task_plan": str(self.task_plan_path),
            "map_id": self.manifest.get("map_id"),
            "duration_s": self.manifest.get("duration_s"),
            "sample_period_s": self.manifest.get("sample_period_s"),
            "sample_count": self.manifest.get("sample_count"),
            "minimum_active_uavs_required": self.manifest.get("minimum_active_uavs_required"),
            "baseline_active_uavs": self.manifest.get("baseline_active_uavs"),
            "task_count": self.manifest.get("task_count"),
            "task_count_by_type": self.manifest.get("task_count_by_type"),
        }


def _lerp_angle_deg(a: float, b: float, alpha: float) -> float:
    delta = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    value = float(a) + delta * float(alpha)
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    return value


def _velocity_from_yaw(speed_mps: float, yaw_deg: float, z_mps: float = 0.0) -> list[float]:
    radians = math.radians(float(yaw_deg))
    return [
        float(speed_mps) * math.cos(radians),
        float(speed_mps) * math.sin(radians),
        float(z_mps),
    ]


def _interpolate_uav(
    prev_uav: dict[str, Any] | None,
    next_uav: dict[str, Any] | None,
    alpha: float,
    prev_time_s: float,
    next_time_s: float,
) -> dict[str, Any] | None:
    if prev_uav is None and next_uav is None:
        return None
    if prev_uav is None:
        if alpha < 0.5:
            return None
        source = dict(next_uav or {})
        position = _position3(source.get("position_enu_m")) or (0.0, 0.0, 0.0)
        yaw = float(source.get("yaw_deg") or 0.0)
        velocity = _velocity_from_yaw(float(source.get("speed_mps") or 0.0), yaw)
    elif next_uav is None:
        if alpha >= 0.5:
            return None
        source = dict(prev_uav)
        position = _position3(source.get("position_enu_m")) or (0.0, 0.0, 0.0)
        yaw = float(source.get("yaw_deg") or 0.0)
        velocity = _velocity_from_yaw(float(source.get("speed_mps") or 0.0), yaw)
    elif prev_uav is next_uav:
        source = dict(prev_uav)
        position = _position3(source.get("position_enu_m")) or (0.0, 0.0, 0.0)
        yaw = float(source.get("yaw_deg") or 0.0)
        velocity = _velocity_from_yaw(float(source.get("speed_mps") or 0.0), yaw)
    else:
        source = dict(prev_uav if alpha < 0.5 else next_uav)
        prev_pos = _position3(prev_uav.get("position_enu_m")) or (0.0, 0.0, 0.0)
        next_pos = _position3(next_uav.get("position_enu_m")) or prev_pos
        position = tuple(prev_pos[i] + (next_pos[i] - prev_pos[i]) * float(alpha) for i in range(3))
        dt = max(1e-9, float(next_time_s) - float(prev_time_s))
        velocity = [(next_pos[i] - prev_pos[i]) / dt for i in range(3)]
        if math.hypot(velocity[0], velocity[1]) > 1e-5:
            yaw = math.degrees(math.atan2(velocity[1], velocity[0]))
        else:
            yaw = _lerp_angle_deg(float(prev_uav.get("yaw_deg") or 0.0), float(next_uav.get("yaw_deg") or 0.0), alpha)
    source["position_enu_m"] = _round_vector(position)
    source["yaw_deg"] = round(float(yaw), 6)
    source["velocity_enu_mps"] = _round_vector(velocity)
    source["source_prev_time_s"] = round(float(prev_time_s), 6)
    source["source_next_time_s"] = round(float(next_time_s), 6)
    source["source_alpha"] = round(float(alpha), 6)
    return source


_DATASET_CACHE: dict[Path, UavGlobalFlowDataset] = {}


def load_uav_global_flow_dataset(output_dir: Path = DEFAULT_UAV_OUTPUT_DIR) -> UavGlobalFlowDataset:
    resolved = output_dir.resolve()
    dataset = _DATASET_CACHE.get(resolved)
    if dataset is None:
        dataset = UavGlobalFlowDataset(resolved)
        _DATASET_CACHE[resolved] = dataset
    return dataset
