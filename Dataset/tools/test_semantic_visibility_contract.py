from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
PLUGIN_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "Plugins" / "SumoImporter" / "Scripts"
sys.path.insert(0, str(PLUGIN_SCRIPT_DIR))

from episode_render_host import CoordinateTransform, EpisodeRenderHost, RoadTopologySnapResult  # noqa: E402
from verify_semantic_visibility_contract import (  # noqa: E402
    find_truth_candidates,
    validate_sample_row,
)
from run_semantic_70_dual_view_tick100 import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SUMMARY_PATH,
    is_f_drive_path,
    output_dir,
    uav_output_dir,
)


CLASS_NAME_TO_ID = {
    "ignore": 0,
    "city_base_background": 1,
    "building": 2,
    "vehicle": 5,
    "pedestrian": 6,
    "drone": 7,
    "hazard_trigger": 11,
    "uav_corridor": 12,
}


def minimal_render_host(*, truth_frame_coordinate_space: str = "map_enu") -> EpisodeRenderHost:
    host = object.__new__(EpisodeRenderHost)
    host.truth_frame_coordinate_space = truth_frame_coordinate_space
    host.coordinate_transform = CoordinateTransform(
        enabled=True,
        translation_enu_m=(7665.869094001044, 7316.608123103693, 0.0),
        axis_mapping="XY_To_XY",
        yaw_deg=245.0,
        scale_enu=(1.0, 1.0, 1.0),
    )
    return host


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def truth_entity(entity_id: str, category: str, label: str, position: list[float]) -> dict:
    return {
        "entity_id": entity_id,
        "entity_category": category,
        "entity_kind": f"{category}.{label}",
        "entity_type": f"{category}.{label}",
        "label_class": label,
        "logical_asset_id": label,
        "tags": [label],
        "truth_pose": {"position_enu_m": position},
        "render_presence": {"submission_state": "submit_to_ue", "visibility_state": "visible"},
    }


def write_truth_episode(root: Path, episode: str, entities: list[dict]) -> None:
    episode_dir = root / episode
    episode_dir.mkdir(parents=True, exist_ok=True)
    frame = {"tick": 100, "entities": entities}
    (episode_dir / "truth_frames.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")


class SemanticVisibilityTruthSearchTest(unittest.TestCase):
    def test_search_reports_missing_classes_without_camera_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_truth_episode(
                root,
                "episode_missing_ground_actors",
                [
                    truth_entity("capture_uav", "uav", "uav", [0.0, 0.0, 80.0]),
                    truth_entity("other_uav", "uav", "uav", [10.0, 0.0, 50.0]),
                    truth_entity("nfz", "facility", "no_fly_zone", [15.0, 0.0, 28.0]),
                    truth_entity("corridor", "airspace_corridor", "uav_corridor", [20.0, 0.0, 80.0]),
                ],
            )

            candidates = find_truth_candidates(
                root,
                required_classes=["drone", "hazard_trigger", "uav_corridor", "pedestrian", "vehicle"],
                altitude_min_m=75.0,
                altitude_max_m=85.0,
                fov_degrees=85.0,
            )

            self.assertEqual(len(candidates), 1)
            self.assertIn("pedestrian", candidates[0].missing_semantic_classes)
            self.assertIn("vehicle", candidates[0].missing_semantic_classes)
            self.assertFalse(candidates[0].satisfies_requirements)

    def test_search_accepts_complete_existing_nadir_footprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_truth_episode(
                root,
                "episode_complete",
                [
                    truth_entity("capture_uav", "uav", "uav", [0.0, 0.0, 80.0]),
                    truth_entity("other_uav", "uav", "uav", [10.0, 0.0, 50.0]),
                    truth_entity("nfz", "facility", "no_fly_zone", [15.0, 0.0, 28.0]),
                    truth_entity("corridor", "airspace_corridor", "uav_corridor", [20.0, 0.0, 80.0]),
                    truth_entity("ped_01", "pedestrian", "pedestrian", [25.0, 0.0, 0.0]),
                    truth_entity("vehicle_01", "vehicle", "vehicle", [30.0, 0.0, 0.0]),
                ],
            )

            candidates = find_truth_candidates(
                root,
                required_classes=["drone", "hazard_trigger", "uav_corridor", "pedestrian", "vehicle"],
                altitude_min_m=75.0,
                altitude_max_m=85.0,
                fov_degrees=85.0,
            )

            self.assertEqual(len(candidates), 1)
            self.assertTrue(candidates[0].satisfies_requirements)
            self.assertEqual(candidates[0].missing_semantic_classes, ())


class SceneSyncCoordinateContractTest(unittest.TestCase):
    def test_map_enu_scene_sync_payload_preserves_resolved_truth_xy(self) -> None:
        host = minimal_render_host(truth_frame_coordinate_space="map_enu")
        position = [7126.023, 6496.149, 6.84]
        rotation = {"pitch_deg": 0.0, "yaw_deg": -56.566, "roll_deg": 0.0}

        self.assertEqual(host._apply_frame_position_enu(position), position)
        self.assertEqual(host._apply_frame_rotation_deg(rotation), rotation)

        inverse_position = host.coordinate_transform.inverse_position(position)
        self.assertAlmostEqual(inverse_position[0], 971.737, places=3)
        self.assertAlmostEqual(inverse_position[1], -142.526, places=3)

    def test_traffic_bundle_snap_and_lane_offset_remain_map_enu_for_scene_sync(self) -> None:
        class FakeRoadTopologySnapper:
            use_sample_z = True
            use_sample_yaw = True

            def should_snap(self, entity: dict) -> bool:
                return entity.get("entity_category") == "vehicle"

            def snap(
                self,
                *,
                entity_id: str,
                position_enu_m: list[float],
                rotation_deg: dict[str, float],
            ) -> RoadTopologySnapResult:
                self.query_position_enu_m = list(position_enu_m)
                self.query_rotation_deg = dict(rotation_deg)
                return RoadTopologySnapResult(
                    edge_id="cg_edge_50",
                    lane_id="cg_edge_50_0",
                    s_m=118.0,
                    position_enu_m=(7126.023, 6496.149, 6.84),
                    yaw_deg=0.0,
                    distance_m=0.0,
                    heading_error_deg=0.0,
                )

        class FakeRoadGeometry:
            def edge_metadata(self, edge_id: str) -> dict:
                self.edge_id = edge_id
                return {"lanes": 2, "width_m": 7.0}

        host = minimal_render_host(truth_frame_coordinate_space="map_enu")
        host.entity_rotation_offset_cfg = {"vehicle_yaw_deg": 0.0}
        host.road_topology_snapper = FakeRoadTopologySnapper()
        host.road_geometry = FakeRoadGeometry()
        host.vehicle_lane_offsets_cfg = {
            "enabled": True,
            "lane_width_m": 3.5,
            "queue_spacing_m": 4.75,
        }
        host.vehicle_lane_slot_by_entity = {
            "vehicle_01": {
                "group_key": ["site.intersection_a", "cg_edge_50"],
                "site_id": "site.intersection_a",
                "axis": "y",
                "band_index": 1,
                "band_count": 2,
                "source_lane_value_m": 6496.149,
                "site_centerline_value_m": 6494.399,
                "side_sign": 1,
                "side_rank": 0,
                "side_band_count": 1,
                "duplicate_index": 0,
                "duplicate_count": 1,
            }
        }

        entity = truth_entity("vehicle_01", "vehicle", "vehicle", [7126.023, 6496.149, 6.84])
        entity["truth_pose"]["rotation_deg"] = {"pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0}

        resolved_position, resolved_rotation, snap_details = host._resolve_entity_pose(entity)

        self.assertEqual(host.road_topology_snapper.query_position_enu_m, [7126.023, 6496.149, 6.84])
        self.assertEqual(host.road_geometry.edge_id, "cg_edge_50")
        self.assertAlmostEqual(resolved_position[0], 7126.023, places=3)
        self.assertAlmostEqual(resolved_position[1], 6497.899, places=3)
        self.assertAlmostEqual(resolved_position[2], 6.84, places=3)
        self.assertEqual(resolved_rotation["yaw_deg"], 0.0)
        self.assertIsNotNone(snap_details)
        self.assertEqual(snap_details["road_lanes"], 2)
        self.assertEqual(snap_details["road_width_m"], 7.0)
        self.assertEqual(snap_details["lane_spacing_m"], 3.5)
        self.assertEqual(host._apply_frame_position_enu(resolved_position), resolved_position)


class FormalCapturePathContractTest(unittest.TestCase):
    def test_default_formal_outputs_are_on_f_drive(self) -> None:
        self.assertTrue(is_f_drive_path(DEFAULT_OUTPUT_ROOT))
        self.assertTrue(is_f_drive_path(DEFAULT_SUMMARY_PATH))

    def test_primary_output_dirs_are_short_and_stable(self) -> None:
        root = Path("F:/aw_cap")
        episode = "X6_crowd_evacuation_to_airspace_lockdown__seed00"
        view_id = "uav_view_000__uav_observer_x6_crowd_evacuation_to_airspace_lockdown_3"

        self.assertEqual(output_dir(root, 69, episode, "high_overview"), root / "hi" / "e69")
        self.assertEqual(uav_output_dir(root, 69, episode, view_id), root / "uav" / "e69" / "v000")


class SemanticVisibilityOutputContractTest(unittest.TestCase):
    def test_validate_sample_row_accepts_semantic_only_proxy_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "sample" / "rgb" / "tick_000100.png"
            seg_path = root / "sample" / "seg" / "tick_000100.png"
            rgb_path.parent.mkdir(parents=True)
            seg_path.parent.mkdir(parents=True)
            Image.new("RGB", (3, 2), color=(12, 34, 56)).save(rgb_path)
            seg = Image.new("L", (3, 2))
            seg.putdata([1, 5, 6, 7, 11, 12])
            seg.save(seg_path)
            common_sidecar = {
                "episode_id": "episode_complete",
                "tick": 100,
                "camera_name": "bottom_center",
                "requested_camera_rotation_body_deg": {"pitch_deg": -90.0, "yaw_deg": 0.0, "roll_deg": 0.0},
                "event_semantic_objects": [
                    {
                        "entity_id": "nfz",
                        "logical_asset_id": "trigger.no_fly.box.v1",
                        "spawn_logical_asset_id": "semantic.trigger_box.extent_14_10_14.v1",
                    },
                    {
                        "entity_id": "corridor",
                        "logical_asset_id": "semantic.uav_corridor.segment.v1",
                    },
                ],
            }
            write_json(
                rgb_path.with_suffix(".json"),
                {
                    **common_sidecar,
                    "capture_backend": "airsim_native_uav_camera",
                    "event_semantic_proxy_sanitizer": {
                        "status": "ok",
                        "target_count": 2,
                        "sanitized_actor_count": 2,
                        "missing_actor_count": 0,
                    },
                    "airsim_proxy_capture_exclusion": {
                        "status": "ok",
                        "method": "proxy_primitive_render_flags",
                        "pipcamera_hidden_lists_mutated": False,
                    },
                },
            )
            write_json(
                seg_path.with_suffix(".json"),
                {
                    **common_sidecar,
                    "capture_backend": "ue_custom_stencil_fixed_world_camera",
                    "segmentation_kind": "ue_custom_stencil_class_id_u8",
                    "class_histogram": {"1": 1, "5": 1, "6": 1, "7": 1, "11": 1, "12": 1},
                },
            )
            row = {
                "episode": "episode_complete",
                "tick": "100",
                "capture_entity_id": "capture_uav",
                "capture_view_id": "uav_view_000__capture_uav",
                "modality_outputs": json.dumps(
                    {
                        "rgb": {"path": str(rgb_path), "sidecar": str(rgb_path.with_suffix(".json"))},
                        "seg": {
                            "png": str(seg_path),
                            "sidecar": str(seg_path.with_suffix(".json")),
                            "histogram": {"1": 1, "5": 1, "6": 1, "7": 1, "11": 1, "12": 1},
                        },
                    }
                ),
            }

            errors, details = validate_sample_row(
                row,
                class_name_to_id=CLASS_NAME_TO_ID,
                required_seg_classes=["city_base_background", "drone", "hazard_trigger", "uav_corridor", "pedestrian", "vehicle"],
                logical_classes=["hazard_trigger", "uav_corridor"],
                min_pixels_per_class=1,
                project_root=root,
            )

            self.assertEqual(errors, [])
            self.assertEqual(details["seg_histogram"]["11"], 1)
            self.assertEqual(details["seg_histogram"]["12"], 1)


if __name__ == "__main__":
    unittest.main()
