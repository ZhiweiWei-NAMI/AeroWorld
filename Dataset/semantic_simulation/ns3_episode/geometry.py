"""Original building polygon extrusions in the frozen traffic-bundle ENU frame.

Only XY has a frozen coordinate authority. Floor-derived height and ground
ENU=0 are explicit model assumptions. GeoJSON Z is retained as raw evidence,
never interpreted as established ground elevation. No planner buffer is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.strtree import STRtree

from Dataset.tools.map_spatial_index import (
    _bounds_center_mercator, _load_geojson_bundle_fit, _transform_coordinate,
)


@dataclass(frozen=True)
class BuildingExtrusion:
    feature_index: int
    building_id: str
    footprint: Polygon | MultiPolygon
    base_enu_m: float
    height_m: float
    evidence: dict

    @property
    def roof_enu_m(self):
        return self.base_enu_m + self.height_m


def _components(geometry):
    if geometry.geom_type in ("LineString", "Point"):
        yield geometry
    elif geometry.geom_type in ("MultiLineString", "MultiPoint", "GeometryCollection"):
        for component in geometry.geoms:
            yield from _components(component)
    elif not geometry.is_empty:
        raise ValueError(f"unexpected planar segment intersection: {geometry.geom_type}")


class BuildingGeometryIndex:
    """Spatial candidates are indexed; obstruction uses exact polygon and Z tests."""
    def __init__(self, buildings, *, metadata):
        self.buildings = tuple(buildings)
        if not self.buildings:
            raise ValueError("building geometry requires authored features")
        self.metadata = metadata
        self._tree = STRtree([building.footprint for building in self.buildings])

    def segment_intersections(self, tx_enu_m, rx_enu_m):
        """Return closed-solid intersections with intervals on segment t in [0,1].

        Boundary/roof tangency counts as contact. Holes remain unobstructed
        volumes; MultiPolygon parts are tested without replacing them by boxes.
        """
        if len(tx_enu_m) != 3 or len(rx_enu_m) != 3:
            raise ValueError("Tx/Rx require three finite ENU coordinates")
        tx, rx = tuple(map(float, tx_enu_m)), tuple(map(float, rx_enu_m))
        if not all(math.isfinite(value) for value in tx + rx):
            raise ValueError("Tx/Rx coordinates must be finite")
        dx, dy, dz = tuple(right - left for left, right in zip(tx, rx))
        xy_norm = dx * dx + dy * dy
        planar = Point(tx[:2]) if xy_norm == 0 else LineString((tx[:2], rx[:2]))
        hits = []
        for index in sorted(int(value) for value in self._tree.query(planar, predicate="intersects")):
            building = self.buildings[index]
            if dz == 0:
                if not building.base_enu_m <= tx[2] <= building.roof_enu_m:
                    continue
                z_interval = (0.0, 1.0)
            else:
                limits = sorted(((building.base_enu_m - tx[2]) / dz, (building.roof_enu_m - tx[2]) / dz))
                z_interval = max(0.0, limits[0]), min(1.0, limits[1])
                if z_interval[0] > z_interval[1]:
                    continue
            if xy_norm == 0:
                xy_intervals = [(0.0, 1.0)]
            else:
                xy_intervals = []
                for component in _components(building.footprint.intersection(planar)):
                    parameters = [((x - tx[0]) * dx + (y - tx[1]) * dy) / xy_norm for x, y in component.coords]
                    if parameters:
                        xy_intervals.append((max(0.0, min(parameters)), min(1.0, max(parameters))))
            intervals = []
            for lower, upper in sorted(xy_intervals):
                lower, upper = max(lower, z_interval[0]), min(upper, z_interval[1])
                if lower <= upper:
                    if intervals and lower <= intervals[-1][1]:
                        intervals[-1][1] = max(intervals[-1][1], upper)
                    else:
                        intervals.append([lower, upper])
            if intervals:
                hits.append({"building_id": building.building_id, "feature_index": building.feature_index,
                    "segment_parameter_intervals": intervals, "base_enu_m": building.base_enu_m,
                    "roof_enu_m": building.roof_enu_m, "height_status": building.evidence["height_status"],
                    "base_status": building.evidence["base_status"]})
        return hits

    def classify_segment(self, tx_enu_m, rx_enu_m):
        hits = self.segment_intersections(tx_enu_m, rx_enu_m)
        return {"geometric_state": "NLOS" if hits else "LOS", "intersections": hits,
                "vertical_authority": "modeled_extrusion_unmeasured_heights_and_ground",
                "intersection_convention": "closed_original_polygon_extrusion_contact"}

    def export_for_ns3(self):
        """Plain dictionary of exact ENU rings and explicit extrusion assumptions."""
        rows = []
        for building in self.buildings:
            parts = (building.footprint,) if building.footprint.geom_type == "Polygon" else building.footprint.geoms
            polygons = [{"exterior_enu_xy_m": [list(point) for point in part.exterior.coords],
                         "holes_enu_xy_m": [[list(point) for point in ring.coords] for ring in part.interiors]} for part in parts]
            rows.append({"feature_index": building.feature_index, "building_id": building.building_id,
                "polygons": polygons, "base_enu_m": building.base_enu_m,
                "roof_enu_m": building.roof_enu_m, "height_m": building.height_m,
                "evidence": building.evidence})
        return {"schema_version": "p01.building_extrusions.v1", "coordinate_contract": "coord.external_enu_m.v1",
                "units": {"geometry": "m"}, "metadata": self.metadata, "buildings": rows}


def load_building_geometry(root, *, config_path="design/p01_ns3/geometry_R2.json"):
    """Read frozen inputs and build the mathematical spatial index once."""
    root = Path(root)
    with (root / config_path).open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    geometry = config["geometry"]
    with (root / geometry["map_package_ref"]).open(encoding="utf-8-sig") as stream:
        package = json.load(stream)
    center = _bounds_center_mercator(root / package["source_geojson"]["bounds"])
    fit = _load_geojson_bundle_fit(root / geometry["coordinate_transform_ref"])
    source_ref = package["source_geojson"]["building"]
    with (root / source_ref).open(encoding="utf-8-sig") as stream:
        features = json.load(stream)["features"]
    vertical = geometry["vertical_model"]
    floor_height, missing_height, base = vertical["floor_height_m"], vertical["missing_floor_height_m"], vertical["base_enu_m"]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (floor_height, missing_height, base)) or min(floor_height, missing_height) <= 0:
        raise ValueError("height assumptions must be explicit finite positive dimensions")
    buildings = []
    for index, feature in enumerate(features):
        raw, properties = feature["geometry"], feature["properties"]
        if raw["type"] == "Polygon":
            groups = [raw["coordinates"]]
        elif raw["type"] == "MultiPolygon":
            groups = raw["coordinates"]
        else:
            raise ValueError("building feature must be Polygon or MultiPolygon")
        parts = []
        z_values = []
        for rings in groups:
            transformed = [[_transform_coordinate(point, center, fit) for point in ring] for ring in rings]
            part = Polygon(transformed[0], transformed[1:])
            if part.is_empty or not part.is_valid:
                raise ValueError(f"invalid original building polygon at feature {index}")
            parts.append(part)
            z_values.extend(point[2] for ring in rings for point in ring if len(point) > 2)
        footprint = parts[0] if raw["type"] == "Polygon" else MultiPolygon(parts)
        if not footprint.is_valid:
            raise ValueError(f"invalid original building MultiPolygon at feature {index}")
        floor = properties["floor"]
        if floor is not None and (not isinstance(floor, (int, float)) or isinstance(floor, bool) or not math.isfinite(floor)):
            raise ValueError(f"floor count is malformed at feature {index}")
        has_floor = floor is not None and floor > 0
        height = floor * floor_height if has_floor else missing_height
        evidence = {"source_ref": source_ref, "source_properties": dict(properties),
            "source_floor": floor, "source_height": properties["height"], "source_altitude": properties["altitude"],
            "source_geojson_z_range": [min(z_values), max(z_values)] if z_values else None,
            "source_geojson_z_status": "unestablished_vertical_reference_not_used_as_ground",
            "height_status": "floor_times_height_assumption" if has_floor else "missing_or_nonpositive_floor_height_assumption",
            "height_source_unit_authority": "not_established_source_height_not_admitted",
            "floor_height_m": floor_height, "missing_floor_height_m": missing_height,
            "base_status": "assumed_flat_ground_missing_terrain_authority", "base_enu_m": base}
        buildings.append(BuildingExtrusion(index, str(properties["id"]), footprint, base, height, evidence))
    metadata = {"source_ref": source_ref, "map_package_ref": geometry["map_package_ref"],
        "coordinate_transform_ref": geometry["coordinate_transform_ref"],
        "coordinate_helper_ref": "Dataset/tools/map_spatial_index.py",
        "vertical_model": vertical, "horizontal_inflation_m": 0.0, "vertical_clearance_inflation_m": 0.0,
        "segment_model": "original_polygon_extrusion_including_holes_and_multipolygon",
        "shadowing_integration": config["propagation_prior"]["shadowing_integration"]}
    return BuildingGeometryIndex(buildings, metadata=metadata)
