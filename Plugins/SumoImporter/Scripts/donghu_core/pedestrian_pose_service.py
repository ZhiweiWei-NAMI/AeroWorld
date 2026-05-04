from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def normalize_activity_type(entity: dict[str, Any]) -> str:
    annotations = dict(entity.get("annotations") or {})
    state_facets = dict(annotations.get("state_facets") or {})
    activity = dict(state_facets.get("activity") or {})
    value = (
        activity.get("activity_type")
        or annotations.get("activity_type")
        or entity.get("activity_type")
        or "idle"
    )
    return str(value).strip().lower()


@dataclass
class PedestrianPoseService:
    min_speed_mps: float = 0.15
    last_rotation_by_entity: dict[str, dict[str, float]] = field(default_factory=dict)

    def resolve_rotation(
        self,
        *,
        entity_id: str,
        position_enu_m: list[float],
        velocity_enu_mps: list[float],
        base_rotation_deg: dict[str, Any],
        activity_type: str,
        freeze_pose: bool = False,
    ) -> dict[str, float]:
        previous_rotation = dict(self.last_rotation_by_entity.get(entity_id) or {})
        result = {
            "pitch_deg": float(base_rotation_deg.get("pitch_deg", base_rotation_deg.get("pitch", 0.0))),
            "yaw_deg": float(base_rotation_deg.get("yaw_deg", base_rotation_deg.get("yaw", 0.0))),
            "roll_deg": float(base_rotation_deg.get("roll_deg", base_rotation_deg.get("roll", 0.0))),
        }
        if freeze_pose and previous_rotation:
            return previous_rotation

        speed_xy = math.hypot(float(velocity_enu_mps[0]), float(velocity_enu_mps[1]))
        if speed_xy >= self.min_speed_mps:
            result["yaw_deg"] = math.degrees(math.atan2(float(velocity_enu_mps[1]), float(velocity_enu_mps[0])))
        elif previous_rotation:
            result["yaw_deg"] = float(previous_rotation.get("yaw_deg", result["yaw_deg"]))
        elif activity_type in {"medical_incident", "fall_flat"}:
            result["yaw_deg"] = float(base_rotation_deg.get("yaw_deg", result["yaw_deg"]))

        self.last_rotation_by_entity[entity_id] = dict(result)
        return result
