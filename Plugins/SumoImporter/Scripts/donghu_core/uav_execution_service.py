from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


def distance_m(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return None
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


@dataclass
class UavExecutionService:
    arrival_tolerance_m: float = 1.5
    hover_before_capture: bool = False

    def build_wait_status(
        self,
        *,
        vehicle_name: str,
        status_payload: dict[str, Any],
        pose_payload: dict[str, Any],
        target_enu_m: Sequence[float],
        timed_out: bool,
        warning: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        status = dict(status_payload or {})
        if warning and not status.get("warning"):
            status["warning"] = warning
        if reason and not status.get("reason"):
            status["reason"] = reason
        error_m = distance_m(target_enu_m, pose_payload.get("position_enu_m"))
        degraded = bool(timed_out) or (error_m is not None and error_m > float(self.arrival_tolerance_m))
        return {
            "vehicle_name": vehicle_name,
            "status": status,
            "pose": dict(pose_payload or {}),
            "timed_out": bool(timed_out),
            "position_error_m": error_m,
            "capture_gate": {
                "wait_for_arrival": True,
                "hover_before_capture": bool(self.hover_before_capture),
                "arrival_tolerance_m": float(self.arrival_tolerance_m),
                "degraded": degraded,
            },
        }
