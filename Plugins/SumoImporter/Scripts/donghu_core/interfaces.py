from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MapPackage:
    map_id: str
    root_dir: Path
    map_context_path: Path
    traffic_bundle_dir: Path
    ped_nav_bundle_path: Path
    asset_catalog_path: Path
    weather_profiles_path: Path
    scenario_objects_runtime_path: Path
    scenario_objects_source_path: Path
    source_geojson: dict[str, Path] = field(default_factory=dict)
    spatial_index_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "map_id": self.map_id,
            "root_dir": str(self.root_dir),
            "map_context": str(self.map_context_path),
            "traffic_bundle_dir": str(self.traffic_bundle_dir),
            "ped_nav_bundle": str(self.ped_nav_bundle_path),
            "asset_catalog": str(self.asset_catalog_path),
            "weather_profiles": str(self.weather_profiles_path),
            "scenario_objects_runtime": str(self.scenario_objects_runtime_path),
            "scenario_objects_source": str(self.scenario_objects_source_path),
            "source_geojson": {key: str(value) for key, value in sorted(self.source_geojson.items())},
        }
        if self.spatial_index_path is not None:
            payload["spatial_index"] = str(self.spatial_index_path)
        return payload


@dataclass(frozen=True)
class ScenarioPackage:
    scenario_id: str
    episode_id: str
    root_dir: Path
    truth_frames_path: Path
    weather_meta_path: Path
    scenario_plan_path: Path
    capture_plan_path: Path
    episode_manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "episode_id": self.episode_id,
            "root_dir": str(self.root_dir),
            "truth_frames": str(self.truth_frames_path),
            "weather_meta": str(self.weather_meta_path),
            "scenario_plan": str(self.scenario_plan_path),
            "capture_plan": str(self.capture_plan_path),
            "episode_manifest": str(self.episode_manifest_path),
        }


@dataclass(frozen=True)
class RuntimeFrameRecord:
    entity_id: str
    entity_category: str
    position_enu_m: list[float]
    rotation_deg: dict[str, float]
    velocity_enu_mps: list[float]
    submission_state: str
    visibility_state: str
    source_pose: dict[str, Any]
    resolved_pose: dict[str, Any]
    transformed_pose: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "entity_id": self.entity_id,
            "entity_category": self.entity_category,
            "position_enu_m": [float(value) for value in self.position_enu_m],
            "rotation_deg": {key: float(value) for key, value in self.rotation_deg.items()},
            "velocity_enu_mps": [float(value) for value in self.velocity_enu_mps],
            "submission_state": self.submission_state,
            "visibility_state": self.visibility_state,
            "source_pose": self.source_pose,
            "resolved_pose": self.resolved_pose,
        }
        if self.transformed_pose is not None:
            payload["transformed_pose"] = self.transformed_pose
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class CaptureSidecar:
    tick: int
    frame_id: str
    camera_id: str
    camera_pose_enu_m: list[float]
    weather: dict[str, Any]
    uav_runtime: dict[str, Any]
    entity_records: list[dict[str, Any]]
    capture_backend: str
    output_path: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "tick": int(self.tick),
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "camera_pose_enu_m": [float(value) for value in self.camera_pose_enu_m],
            "weather": dict(self.weather),
            "uav_runtime": dict(self.uav_runtime),
            "entity_records": list(self.entity_records),
            "capture_backend": self.capture_backend,
            "output_path": self.output_path,
        }
        payload.update(self.extra)
        return payload
