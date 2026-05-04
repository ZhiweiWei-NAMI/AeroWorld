from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CoordinateTransform:
    enabled: bool
    translation_enu_m: tuple[float, float, float]
    axis_mapping: str
    yaw_deg: float
    scale_enu: tuple[float, float, float]

    @staticmethod
    def _read_vector(config: dict[str, Any], field_name: str, default: Sequence[float]) -> tuple[float, float, float]:
        raw = config.get(field_name, default)
        values = list(raw if isinstance(raw, Sequence) else default)
        return (
            float(values[0] if len(values) > 0 else default[0]),
            float(values[1] if len(values) > 1 else default[1]),
            float(values[2] if len(values) > 2 else default[2]),
        )

    @classmethod
    def from_root_config(cls, root_config: dict[str, Any]) -> "CoordinateTransform":
        config = dict(root_config.get("coordinate_transform") or {})
        return cls(
            enabled=bool(config.get("enabled", False)),
            translation_enu_m=cls._read_vector(config, "translation_enu_m", (0.0, 0.0, 0.0)),
            axis_mapping=str(config.get("axis_mapping", "XY_To_XY") or "XY_To_XY"),
            yaw_deg=float(config.get("yaw_deg", 0.0)),
            scale_enu=cls._read_vector(config, "scale_enu", (1.0, 1.0, 1.0)),
        )

    def _apply_axis_mapping(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        mapping = self.axis_mapping.strip() or "XY_To_XY"
        if mapping == "XY_To_XNegY":
            return x, -y, z
        if mapping == "XY_To_YX":
            return y, x, z
        if mapping == "XY_To_YNegX":
            return y, -x, z
        return x, y, z

    def _rotate_xy(self, x: float, y: float) -> tuple[float, float]:
        if not self.enabled or abs(self.yaw_deg) <= 1e-6:
            return x, y
        yaw_rad = math.radians(self.yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        return x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw

    def _transform_vector_components(self, value_enu: Sequence[float]) -> tuple[float, float, float]:
        x = float(value_enu[0] if len(value_enu) > 0 else 0.0) * self.scale_enu[0]
        y = float(value_enu[1] if len(value_enu) > 1 else 0.0) * self.scale_enu[1]
        z = float(value_enu[2] if len(value_enu) > 2 else 0.0) * self.scale_enu[2]
        if self.enabled:
            x, y, z = self._apply_axis_mapping(x, y, z)
            x, y = self._rotate_xy(x, y)
        return x, y, z

    def apply_position(self, position_enu_m: Sequence[float]) -> list[float]:
        x, y, z = self._transform_vector_components(position_enu_m)
        if self.enabled:
            x += self.translation_enu_m[0]
            y += self.translation_enu_m[1]
            z += self.translation_enu_m[2]
        return [x, y, z]

    def apply_vector(self, value_enu: Sequence[float]) -> list[float]:
        x, y, z = self._transform_vector_components(value_enu)
        return [x, y, z]

    def apply_rotation(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        result = {
            "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", 0.0))),
            "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", 0.0))),
            "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", 0.0))),
        }
        if self.enabled:
            yaw_rad = math.radians(result["yaw_deg"])
            forward_x, forward_y, _ = self._transform_vector_components((math.cos(yaw_rad), math.sin(yaw_rad), 0.0))
            if abs(forward_x) > 1e-6 or abs(forward_y) > 1e-6:
                result["yaw_deg"] = math.degrees(math.atan2(forward_y, forward_x))
        return result

    def apply_yaw_deg(self, yaw_deg: float) -> float:
        if not self.enabled:
            return float(yaw_deg)
        yaw_rad = math.radians(float(yaw_deg))
        forward_x, forward_y, _ = self.apply_vector([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
        return math.degrees(math.atan2(forward_y, forward_x))

    def describe(self) -> str:
        return (
            f"enabled={self.enabled} "
            f"translation_enu_m={[self.translation_enu_m[0], self.translation_enu_m[1], self.translation_enu_m[2]]} "
            f"axis_mapping={self.axis_mapping} "
            f"yaw_deg={self.yaw_deg} "
            f"scale_enu={[self.scale_enu[0], self.scale_enu[1], self.scale_enu[2]]}"
        )


CoordinateTransformConfig = CoordinateTransform


@dataclass
class CoordinateService:
    transform: CoordinateTransform

    @classmethod
    def from_root_config(cls, root_config: dict[str, Any]) -> "CoordinateService":
        return cls(transform=CoordinateTransform.from_root_config(root_config))

    def apply_position(self, position_enu_m: Sequence[float]) -> list[float]:
        return self.transform.apply_position(position_enu_m)

    def apply_vector(self, value_enu: Sequence[float]) -> list[float]:
        return self.transform.apply_vector(value_enu)

    def apply_rotation(self, rotation_deg: dict[str, Any]) -> dict[str, float]:
        return self.transform.apply_rotation(rotation_deg)

    def apply_yaw_deg(self, yaw_deg: float) -> float:
        return self.transform.apply_yaw_deg(yaw_deg)
