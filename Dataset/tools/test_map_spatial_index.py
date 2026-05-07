from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from map_spatial_index import MapSpatialIndex, dist_xy  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


class MapSpatialIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spatial = MapSpatialIndex.default(ROOT)

    def test_geojson_fit_matches_traffic_bundle(self) -> None:
        self.assertGreaterEqual(self.spatial.geojson_fit.pair_count, 800)
        self.assertLess(self.spatial.geojson_fit.max_error_m, 0.05)
        self.assertEqual(self.spatial.source_geojson_paths["bounds"].name, "bounds.geojson")
        self.assertEqual(self.spatial.source_geojson_paths["block"].name, "block.geojson")

    def test_green_policy_reports_green_without_allowance(self) -> None:
        green_area = (
            self.spatial.green_union
            .intersection(self.spatial.bounds_union)
            .difference(self.spatial.water_union)
            .difference(self.spatial.building_union)
        )
        self.assertFalse(green_area.is_empty)
        point = green_area.representative_point()
        pos = [point.x, point.y, 0.0]

        disallowed = self.spatial.validation_errors_for_point(
            pos,
            context="green policy test",
            allow_road=True,
            allow_green=False,
        )
        allowed = self.spatial.validation_errors_for_point(
            pos,
            context="green policy test",
            allow_road=True,
            allow_green=True,
        )

        self.assertTrue(any("green.geojson" in error for error in disallowed))
        self.assertFalse(any("green.geojson" in error for error in allowed))

    def test_sidewalk_route_is_continuous_and_valid(self) -> None:
        for sample in self.spatial.lanes.samples[::25]:
            try:
                start = self.spatial.plan_sidewalk_anchor(
                    [sample.x_m, sample.y_m, 0.0],
                    edge_id_hint=sample.edge_id,
                    s_hint=sample.s_m,
                    allow_green=False,
                )
                end_sample = self.spatial.lanes.resolve_edge_s(sample.edge_id, sample.s_m + 25.0)
                end = self.spatial.plan_sidewalk_anchor(
                    [end_sample.x_m, end_sample.y_m, 0.0],
                    edge_id_hint=end_sample.edge_id,
                    s_hint=end_sample.s_m,
                    allow_green=False,
                )
                route = self.spatial.plan_sidewalk_route(
                    [start.position_enu_m, end.position_enu_m],
                    allow_green=False,
                    context="unit sidewalk route",
                )
            except Exception:
                continue

            self.assertGreaterEqual(len(route), 2)
            self.assertEqual(route[0], start.position_enu_m)
            self.assertEqual(route[-1], end.position_enu_m)
            for a, b in zip(route, route[1:]):
                self.assertLessEqual(dist_xy(a, b), 18.001)
                self.spatial.validate_segment(
                    a,
                    b,
                    context="unit sidewalk route segment",
                    allow_road=False,
                    allow_green=False,
                )
            return
        self.fail("No legal sidewalk route candidate found")

    def test_crossing_route_allows_only_controlled_road_segment(self) -> None:
        for sample in self.spatial.lanes.samples[::25]:
            try:
                crossing = self.spatial.plan_crossing_route([sample.x_m, sample.y_m, 0.0])
            except Exception:
                continue

            self.spatial.validate_point(
                crossing.start_position_enu_m,
                context="unit crossing start",
                allow_road=False,
                allow_green=False,
            )
            self.spatial.validate_segment(
                crossing.start_position_enu_m,
                crossing.roadway_center_position_enu_m,
                context="unit crossing segment",
                allow_road=True,
                allow_green=False,
            )
            self.spatial.validate_segment(
                crossing.roadway_center_position_enu_m,
                crossing.opposite_curb_position_enu_m,
                context="unit crossing segment",
                allow_road=True,
                allow_green=False,
            )
            return
        self.fail("No legal crossing route candidate found")

    def test_crowd_spawn_envelope_is_valid(self) -> None:
        for sample in self.spatial.lanes.samples[::25]:
            desired = [sample.x_m, sample.y_m, 0.0]
            extent_cm = [600.0, 400.0, 0.0]
            try:
                origin = self.spatial.plan_crowd_zone(desired, extent_cm, allow_green=True)
                self.spatial.validate_spawn_envelope(
                    origin,
                    extent_cm,
                    context="unit crowd zone",
                    allow_green=True,
                )
            except Exception:
                continue
            return
        self.fail("No legal crowd spawn envelope candidate found")


class DensePedestrianPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_path = ROOT / "Plugins" / "SumoImporter" / "Scenarios" / "donghu_dense_uav_rain_fall" / "scripts" / "build.py"
        spec = importlib.util.spec_from_file_location("dense_demo_build", build_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {build_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["dense_demo_build"] = module
        spec.loader.exec_module(module)
        cls.build = module

    def test_pedestrian_plan_polyline_interpolates_by_arclength(self) -> None:
        plan = self.build.PedestrianPlan(
            entity_id="pedestrian_test",
            variant_id="adult",
            path_id="route.test",
            start_tick=0,
            end_tick=10,
            start_position_enu_m=[0.0, 0.0, 0.0],
            end_position_enu_m=[10.0, 10.0, 0.0],
            active_window_end_tick=20,
            hold_position_enu_m=[10.0, 10.0, 0.0],
            hold_activity_type="waiting",
            hold_posture="standing",
            hold_social_state="calm",
            hold_animation_hint="pedestrian_idle",
            path_waypoints_enu_m=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)),
        )

        midpoint = self.build._polyline_point_at_fraction(plan.route_points_enu_m, 0.5)

        self.assertEqual(midpoint, [10.0, 0.0, 0.0])
        self.assertAlmostEqual(plan.route_yaw_deg, 90.0)


if __name__ == "__main__":
    unittest.main()
