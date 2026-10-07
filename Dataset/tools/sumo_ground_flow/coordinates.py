"""Coordinate conversion from SUMO net/FCD space into the UE traffic-bundle frame."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

try:
    from map_spatial_index import (  # type: ignore
        DEFAULT_MAP_PACKAGE,
        ROOT,
        GeoJsonBundleFit,
    )
except ModuleNotFoundError:  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.map_spatial_index import (  # type: ignore
        DEFAULT_MAP_PACKAGE,
        ROOT,
        GeoJsonBundleFit,
    )


class SumoCoordinateError(RuntimeError):
    """Raised when SUMO coordinates cannot be mapped into the truth frame."""


ROAD_DERIVED_COORDINATE_ROUTE = "sumo_xy_identity_truth_enu_m"


def _candidate_sumo_tools_dirs(explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidates.append(Path(sumo_home) / "tools")
    return candidates


def sumo_executable_from_environment(binary_name: str) -> Path:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise SumoCoordinateError(
            f"SUMO_HOME is required to resolve the {binary_name} executable"
        )
    executable = Path(sumo_home) / "bin" / binary_name
    if not executable.is_file():
        raise SumoCoordinateError(f"SUMO executable does not exist: {executable}")
    return executable


def ensure_sumo_tools_path(sumo_tools_dir: Path | None = None) -> None:
    for candidate in _candidate_sumo_tools_dirs(sumo_tools_dir):
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
            return
    raise SumoCoordinateError("SUMO tools directory not found; set SUMO_HOME or pass sumo_tools_dir")


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class MapCoordinateSources:
    road_geojson: Path
    bounds_geojson: Path
    lane_center_samples_csv: Path
    road_center_samples_csv: Path
    road_geojson_transform_json: Path

    @classmethod
    def from_map_package(cls, map_package: Path = DEFAULT_MAP_PACKAGE) -> "MapCoordinateSources":
        package = json.loads(Path(map_package).read_text(encoding="utf-8-sig"))
        source_geojson = dict(package.get("source_geojson") or {})
        traffic_bundle_dir = _resolve_project_path(str(package.get("traffic_bundle_dir") or ""))
        return cls(
            road_geojson=_resolve_project_path(str(source_geojson.get("road") or "")),
            bounds_geojson=_resolve_project_path(str(source_geojson.get("bounds") or "")),
            lane_center_samples_csv=traffic_bundle_dir / "lane_center_samples.csv",
            road_center_samples_csv=traffic_bundle_dir / "road_center_samples.csv",
            road_geojson_transform_json=traffic_bundle_dir / "road_geojson_transform.json",
        )


@dataclass(frozen=True)
class SumoTruthCoordinateMapper:
    """Maps formal road-derived SUMO XY directly into UE truth XY."""

    net: Any
    bounds_center_mercator_m: tuple[float, float]
    geojson_bundle_fit: GeoJsonBundleFit
    sources: MapCoordinateSources
    coordinate_route: str = ROAD_DERIVED_COORDINATE_ROUTE

    @classmethod
    def default(
        cls,
        net_xml: Path,
        *,
        map_package: Path = DEFAULT_MAP_PACKAGE,
        sumo_tools_dir: Path | None = None,
        coordinate_route: str | None = None,
    ) -> "SumoTruthCoordinateMapper":
        ensure_sumo_tools_path(sumo_tools_dir)
        import sumolib  # type: ignore

        sources = MapCoordinateSources.from_map_package(map_package)
        route = coordinate_route or ROAD_DERIVED_COORDINATE_ROUTE
        if route != ROAD_DERIVED_COORDINATE_ROUTE:
            raise SumoCoordinateError(
                f"Formal traffic supports only {ROAD_DERIVED_COORDINATE_ROUTE}; got {route}"
            )
        if not _looks_like_road_derived_net(Path(net_xml)):
            raise SumoCoordinateError(f"Formal traffic only accepts road_derived.net.xml under road_derived/: {net_xml}")
        missing = [path for path in (Path(net_xml), sources.road_geojson_transform_json) if not path.exists()]
        if missing:
            raise SumoCoordinateError(f"Missing coordinate source files: {missing}")
        fit = _identity_fit_from_transform(sources.road_geojson_transform_json)
        return cls(
            net=sumolib.net.readNet(str(net_xml)),
            bounds_center_mercator_m=(0.0, 0.0),
            geojson_bundle_fit=fit,
            sources=sources,
            coordinate_route=route,
        )

    def sumo_xy_to_truth_xy(self, x_m: float, y_m: float) -> tuple[float, float]:
        return float(x_m), float(y_m)

    def sumo_shape_to_truth_xy(self, points_xy: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        return tuple(self.sumo_xy_to_truth_xy(float(x_m), float(y_m)) for x_m, y_m in points_xy)

    @property
    def fit_summary(self) -> dict[str, Any]:
        fit = self.geojson_bundle_fit
        return {
            "source": self.coordinate_route,
            "matrix": fit.matrix,
            "mean_error_m": round(fit.mean_error_m, 9),
            "max_error_m": round(fit.max_error_m, 9),
            "pair_count": fit.pair_count,
        }


def _looks_like_road_derived_net(net_xml: Path) -> bool:
    normalized = str(net_xml).replace("\\", "/").lower()
    return "/road_derived/" in normalized and Path(net_xml).name.lower() == "road_derived.net.xml"


def _identity_fit_from_transform(transform_json: Path) -> GeoJsonBundleFit:
    if transform_json.exists():
        payload = json.loads(transform_json.read_text(encoding="utf-8-sig"))
        return GeoJsonBundleFit(
            matrix=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
            mean_error_m=float(payload.get("mean_error_m") or 0.0),
            max_error_m=float(payload.get("max_error_m") or 0.0),
            pair_count=int(payload.get("pair_count") or 0),
        )
    return GeoJsonBundleFit(matrix=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)), mean_error_m=0.0, max_error_m=0.0, pair_count=0)
