from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_writer import load_json


DEFAULT_WEATHER_PROFILES: dict[str, dict[str, Any]] = {
    "clear": {
        "condition": "clear",
        "rain": 0.0,
        "wetness": 0.0,
        "fog_density": 0.0,
        "dust": 0.0,
        "visibility_m": 6000.0,
        "visibility": 0.95,
        "wind_speed": 2.5,
        "surface_state_a": "dry",
        "surface_friction_scale_a": 1.0,
    },
    "rain": {
        "condition": "rain",
        "rain": 0.7,
        "wetness": 0.8,
        "fog_density": 0.1,
        "dust": 0.0,
        "visibility_m": 2000.0,
        "visibility": 0.6,
        "wind_speed": 6.0,
        "surface_state_a": "wet",
        "surface_friction_scale_a": 0.72,
    },
    "fog": {
        "condition": "fog",
        "rain": 0.0,
        "wetness": 0.2,
        "fog_density": 0.55,
        "dust": 0.0,
        "visibility_m": 500.0,
        "visibility": 0.35,
        "wind_speed": 1.5,
        "surface_state_a": "damp",
        "surface_friction_scale_a": 0.85,
    },
    "storm": {
        "condition": "storm",
        "rain": 1.0,
        "wetness": 1.0,
        "fog_density": 0.35,
        "dust": 0.15,
        "visibility_m": 300.0,
        "visibility": 0.2,
        "wind_speed": 9.0,
        "surface_state_a": "storm",
        "surface_friction_scale_a": 0.6,
    },
}


@dataclass
class WeatherService:
    profiles: dict[str, dict[str, Any]]

    @classmethod
    def from_profiles_path(cls, path: Path) -> "WeatherService":
        payload = load_json(path)
        profiles = dict(payload.get("profiles") or {})
        merged: dict[str, dict[str, Any]] = {}
        for key, value in DEFAULT_WEATHER_PROFILES.items():
            merged[key] = dict(value)
        for key, value in profiles.items():
            merged[str(key)] = {**merged.get(str(key), {}), **dict(value)}
        return cls(profiles=merged)

    def payload_for_condition(self, condition: str) -> dict[str, Any]:
        key = str(condition or "clear").strip().lower() or "clear"
        profile = dict(self.profiles.get(key) or self.profiles.get("clear") or DEFAULT_WEATHER_PROFILES["clear"])
        profile.setdefault("condition", key)
        return profile

    def payload_for_row(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = self.payload_for_condition(str(row.get("condition") or "clear"))
        if row.get("visibility_m") is not None:
            payload["visibility_m"] = float(row["visibility_m"])
        if row.get("rain") is not None:
            payload["rain"] = float(row["rain"])
        if row.get("wetness") is not None:
            payload["wetness"] = float(row["wetness"])
        if row.get("fog_density") is not None:
            payload["fog_density"] = float(row["fog_density"])
        if row.get("dust") is not None:
            payload["dust"] = float(row["dust"])
        if row.get("wind_speed") is not None:
            payload["wind_speed"] = float(row["wind_speed"])
        if row.get("wind_mps") is not None:
            payload["wind_mps"] = float(row["wind_mps"])
        if row.get("wind_vector_enu_mps") is not None:
            payload["wind_vector_enu_mps"] = list(row["wind_vector_enu_mps"])
        return payload
