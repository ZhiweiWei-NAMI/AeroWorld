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
        _bounds_center_mercator,
        _geojson_coord_to_local_xy,
        fit_geojson_to_bundle,
    )
except ModuleNotFoundError:  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.map_spatial_index import (  # type: ignore
        DEFAULT_MAP_PACKAGE,
        ROOT,
        GeoJsonBundleFit,
        _bounds_center_mercator,
        _geojson_coord_to_local_xy,
        fit_geojson_to_bundle,
    )


class SumoCoordinateError(RuntimeError):
    """Raised when SUMO coordinates cannot be mapped into the truth frame."""


def _candidate_sumo_tools_dirs(explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidates.append(Path(sumo_home) / "tools")
    candidates.append(Path("E:/sumo-1.8.0/tools"))
    return candidates


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

    @classmethod
    def from_map_package(cls, map_package: Path = DEFAULT_MAP_PACKAGE) -> "MapCoordinateSources":
        package = json.loads(Path(map_package).read_text(encoding="utf-8-sig"))
        source_geojson = dict(package.get("source_geojson") or {})
        traffic_bundle_dir = _resolve_project_path(str(package.get("traffic_bundle_dir") or ""))
        return cls(
            road_geojson=_resolve_project_path(str(source_geojson.get("road") or "")),
            bounds_geojson=_resolve_project_path(str(source_geojson.get("bounds") or "")),
            lane_center_samples_csv=traffic_bundle_dir / "lane_center_samples.csv",
        )


@dataclass(frozen=True)
class SumoTruthCoordinateMapper:
    """Maps SUMO local XY through lon/lat and GeoJSON fitting into UE truth XY."""

    net: Any
    bounds_center_mercator_m: tuple[float, float]
    geojson_bundle_fit: GeoJsonBundleFit
    sources: MapCoordinateSources

    @classmethod
    def default(
        cls,
        net_xml: Path,
        *,
        map_package: Path = DEFAULT_MAP_PACKAGE,
        sumo_tools_dir: Path | None = None,
    ) -> "SumoTruthCoordinateMapper":
        ensure_sumo_tools_path(sumo_tools_dir)
        import sumolib  # type: ignore

        sources = MapCoordinateSources.from_map_package(map_package)
        missing = [
            path
            for path in (Path(net_xml), sources.road_geojson, sources.bounds_geojson, sources.lane_center_samples_csv)
            if not path.exists()
        ]
        if missing:
            raise SumoCoordinateError(f"Missing coordinate source files: {missing}")
        fit = fit_geojson_to_bundle(
            road_geojson_path=sources.road_geojson,
            bounds_geojson_path=sources.bounds_geojson,
            lane_center_samples_csv=sources.lane_center_samples_csv,
        )
        return cls(
            net=sumolib.net.readNet(str(net_xml)),
            bounds_center_mercator_m=_bounds_center_mercator(sources.bounds_geojson),
            geojson_bundle_fit=fit,
            sources=sources,
        )

    def sumo_xy_to_lonlat(self, x_m: float, y_m: float) -> tuple[float, float]:
        lon, lat = self.net.convertXY2LonLat(float(x_m), float(y_m))
        return float(lon), float(lat)

    def lonlat_to_truth_xy(self, lon_deg: float, lat_deg: float) -> tuple[float, float]:
        local_xy = _geojson_coord_to_local_xy((float(lon_deg), float(lat_deg)), self.bounds_center_mercator_m)
        return self.geojson_bundle_fit.transform_local_xy(local_xy[0], local_xy[1])

    def sumo_xy_to_truth_xy(self, x_m: float, y_m: float) -> tuple[float, float]:
        lon, lat = self.sumo_xy_to_lonlat(x_m, y_m)
        return self.lonlat_to_truth_xy(lon, lat)

    def sumo_shape_to_truth_xy(self, points_xy: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        return tuple(self.sumo_xy_to_truth_xy(float(x_m), float(y_m)) for x_m, y_m in points_xy)

    @property
    def fit_summary(self) -> dict[str, Any]:
        fit = self.geojson_bundle_fit
        return {
            "source": "sumo_xy_to_lonlat_to_geojson_bundle_fit",
            "matrix": fit.matrix,
            "mean_error_m": round(fit.mean_error_m, 9),
            "max_error_m": round(fit.max_error_m, 9),
            "pair_count": fit.pair_count,
        }
