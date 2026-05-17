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
from run_semantic_event_chain_every10 import (  # noqa: E402
    DEFAULT_CAPTURE_PRESETS,
    DEFAULT_OUTPUT_ROOT as EVENT_CHAIN_DEFAULT_OUTPUT_ROOT,
    DEFAULT_SUMMARY_PATH as EVENT_CHAIN_DEFAULT_SUMMARY_PATH,
    EpisodePlan,
    event_chain_capture_ticks,
    event_chain_output_dir,
    event_chain_uav_output_dir,
    filter_event_chain_capture_presets,
    is_f_drive_path,
    scene_uav_active_ticks_by_entity,
    validate_contract,
    write_guarded_config,
)
from convert_to_render_ready import (  # noqa: E402
    RUNTIME_BOUNDARY_PADDING_M,
    UavSelection,
    point_in_capture_roi_xy,
    point_in_runtime_boundary_xy,
    source_runtime_visible_ticks_by_entity,
    subset_uav_selection_for_runtime,
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
            preserve_truth_xy = True

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
        self.assertFalse(hasattr(host.road_geometry, "edge_id"))
        self.assertAlmostEqual(resolved_position[0], 7126.023, places=3)
        self.assertAlmostEqual(resolved_position[1], 6496.149, places=3)
        self.assertAlmostEqual(resolved_position[2], 6.84, places=3)
        self.assertEqual(resolved_rotation["yaw_deg"], 0.0)
        self.assertIsNotNone(snap_details)
        self.assertTrue(snap_details["preserve_truth_xy"])
        self.assertEqual(host._apply_frame_position_enu(resolved_position), resolved_position)


class EventChainCaptureContractTest(unittest.TestCase):
    def test_default_formal_outputs_are_on_f_drive(self) -> None:
        self.assertTrue(is_f_drive_path(EVENT_CHAIN_DEFAULT_OUTPUT_ROOT))
        self.assertTrue(is_f_drive_path(EVENT_CHAIN_DEFAULT_SUMMARY_PATH))

    def test_event_chain_outputs_are_short_and_stable(self) -> None:
        root = Path("F:/aw_cap")
        plan = EpisodePlan(
            index=69,
            episode_dir=Path("X6_crowd_evacuation_to_airspace_lockdown__seed00"),
            scenario_id="X6_crowd_evacuation_to_airspace_lockdown",
            seed_label="seed00",
            site_id="site.intersection_a",
            high_camera_id="site.intersection_a_overview_top",
            capture_ticks=[],
            uav_active_ticks={},
        )
        view_id = "uav_view_012__uav_observer_x6_crowd_evacuation_to_airspace_lockdown_3"

        self.assertEqual(event_chain_output_dir(root, plan, "high_overview_rgb"), root / plan.scenario_id / "seed00" / "high")
        self.assertEqual(event_chain_uav_output_dir(root, plan, "uav_observer", view_id), root / plan.scenario_id / "seed00" / view_id)

    def test_every10_tick_selection_and_scene_uav_active_ticks_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "episode_event_chain"
            episode_dir.mkdir()
            write_json(
                episode_dir / "global_entity_roster.json",
                {
                    "entities": [
                        {"entity_id": "uav_alpha", "entity_category": "uav", "mode": "scene_sync"},
                        {"entity_id": "uav_static", "entity_category": "uav", "mode": "metadata_only"},
                    ]
                },
            )
            frames = []
            for tick in (0, 10, 20):
                entities = []
                if tick != 10:
                    entities.append(
                        {
                            "entity_id": "uav_alpha",
                            "entity_category": "uav",
                            "render_presence": {"submission_state": "submit_to_ue", "visibility_state": "visible"},
                        }
                    )
                frames.append(json.dumps({"tick": tick, "entities": entities}))
            (episode_dir / "truth_frames.jsonl").write_text("\n".join(frames) + "\n", encoding="utf-8")

            ticks = event_chain_capture_ticks(episode_dir, tick_start=0, tick_end=20, tick_step=10, strict=True)
            active = scene_uav_active_ticks_by_entity(episode_dir, ticks)

            self.assertEqual(ticks, [0, 10, 20])
            self.assertEqual(active["uav_alpha"], [0, 20])
            self.assertNotIn("uav_static", active)
            with self.assertRaises(RuntimeError):
                event_chain_capture_ticks(episode_dir, tick_start=0, tick_end=30, tick_step=10, strict=True)

    def test_scene_uav_active_ticks_stop_when_roi_capture_eligible_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "episode_event_chain"
            episode_dir.mkdir()
            write_json(
                episode_dir / "global_entity_roster.json",
                {"entities": [{"entity_id": "uav_alpha", "entity_category": "uav", "mode": "scene_sync"}]},
            )
            frames = []
            for tick, eligible in ((0, True), (5, False), (10, True), (15, False)):
                frames.append(
                    json.dumps(
                        {
                            "tick": tick,
                            "entities": [
                                {
                                    "entity_id": "uav_alpha",
                                    "entity_category": "uav",
                                    "render_presence": {
                                        "submission_state": "submit_to_ue",
                                        "visibility_state": "visible",
                                    },
                                    "uav_visibility": {
                                        "roi_capture_eligible": eligible,
                                        "selected_for_capture_truth": eligible,
                                    },
                                }
                            ],
                        }
                    )
                )
            (episode_dir / "truth_frames.jsonl").write_text("\n".join(frames) + "\n", encoding="utf-8")

            ticks = event_chain_capture_ticks(episode_dir, tick_start=0, tick_end=15, tick_step=5, strict=True)
            active = scene_uav_active_ticks_by_entity(episode_dir, ticks)

            self.assertEqual(active["uav_alpha"], [0, 10])

    def test_event_chain_capture_presets_keep_only_high_rgb_overviews(self) -> None:
        presets = filter_event_chain_capture_presets(DEFAULT_CAPTURE_PRESETS)
        for cameras in presets["ground_cameras"].values():
            self.assertGreaterEqual(len(cameras), 1)
            for camera in cameras:
                self.assertTrue(str(camera.get("camera_id", "")).endswith("overview_top"))
                self.assertEqual(camera.get("modalities"), ["rgb"])

    def test_formal_event_chain_paths_are_exact_contract_defaults(self) -> None:
        class Args:
            output_root = EVENT_CHAIN_DEFAULT_OUTPUT_ROOT
            summary = EVENT_CHAIN_DEFAULT_SUMMARY_PATH
            segmentation_backend = "ue_custom_stencil"
            uav_capture_backend = "editor_hook"
            uav_modalities = ["rgb", "depth", "seg"]
            tick_start = 0
            tick_step = 5
            allow_nonstandard_tick_step = False
            max_private_memory_gb = 20.0
            max_working_set_gb = 20.0
            host_run_timeout_s = 300.0
            capture_ticks_per_host_run = 16
            allow_single_host_full_chain = False

        contract = {
            "defaults": {
                "output_root": "F:/aw_cap",
                "summary": "F:/aw_cap_summary.csv",
                "max_private_memory_gb": 20.0,
                "max_working_set_gb": 20.0,
                "host_run_timeout_s": 300.0,
            },
            "must_follow": {"output_root_must_be_f_drive_root": True},
        }

        validate_contract(Args, contract)
        Args.output_root = Path("F:/not_aw_cap")
        with self.assertRaises(RuntimeError):
            validate_contract(Args, contract)

    def test_guarded_config_forces_single_batch_window(self) -> None:
        class Args:
            output_root: Path
            editor_hook_capture_timeout_s = 90.0
            uav_scene_control_backend = "truth_frame_scene_sync"
            tick_start = 0
            tick_end = 900
            tick_step = 5
            simulation_tick_stride = 1
            capture_ticks_per_host_run = 16
            uav_modalities = ["rgb", "depth", "seg"]
            uav_capture_backend = "editor_hook"
            scene_sync_batch_size = 96
            scene_sync_delay_s = 0.0
            write_depth_preview = False
            write_semantic_audit = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode_dir = root / "episode"
            episode_dir.mkdir()
            write_json(
                episode_dir / "render_host_config.json",
                {
                    "batch_strategy": {"sites": ["site.intersection_a"], "tick_window_size": 12},
                    "timeouts": {},
                },
            )
            Args.output_root = root / "out"
            presets_path = root / "presets.json"
            write_json(presets_path, {"ground_cameras": {}})
            plan = EpisodePlan(
                index=0,
                episode_dir=episode_dir,
                scenario_id="episode",
                seed_label="seed00",
                site_id="site.intersection_a",
                high_camera_id="site.intersection_a_overview_top",
                capture_ticks=[],
                uav_active_ticks={},
            )

            guarded = write_guarded_config(Args, plan, presets_path)
            payload = json.loads(guarded.read_text(encoding="utf-8"))

            self.assertEqual(payload["batch_strategy"]["tick_window_size"], 0)


class RenderReadySpatialCropContractTest(unittest.TestCase):
    def test_runtime_boundary_expands_roi_but_camera_gate_does_not(self) -> None:
        polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

        self.assertTrue(point_in_capture_roi_xy([5.0, 5.0, 0.0], polygon))
        self.assertTrue(point_in_runtime_boundary_xy([5.0, 5.0, 0.0], polygon))
        self.assertFalse(point_in_capture_roi_xy([65.0, 5.0, 0.0], polygon))
        self.assertTrue(point_in_runtime_boundary_xy([65.0, 5.0, 0.0], polygon))
        self.assertFalse(point_in_runtime_boundary_xy([10.0 + RUNTIME_BOUNDARY_PADDING_M + 0.1, 5.0, 0.0], polygon))

    def test_source_runtime_visible_ticks_crop_enter_and_exit(self) -> None:
        source_contract = {
            "capture_boundary_id": "roi.test",
            "capture_boundary_polygon_enu_m": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        }
        roster = [
            {
                "entity_id": "ped_enter_exit",
                "label_class": "pedestrian",
                "asset_id": "pedestrian.cityops.basic.v1",
                "entity_category": "pedestrian",
                "role": "semantic_background_pedestrian",
                "ground_flow_contract": {"route_duration_ticks": 15},
            }
        ]
        grouped = {
            "ped_enter_exit": [
                {"entity_id": "ped_enter_exit", "tick": 0, "label_class": "pedestrian", "pos_enu": [101.0, 5.0, 0.0], "vel_mps": [0.0, 0.0, 0.0]},
                {"entity_id": "ped_enter_exit", "tick": 5, "label_class": "pedestrian", "pos_enu": [65.0, 5.0, 0.0], "vel_mps": [-1.0, 0.0, 0.0]},
                {"entity_id": "ped_enter_exit", "tick": 10, "label_class": "pedestrian", "pos_enu": [5.0, 5.0, 0.0], "vel_mps": [-1.0, 0.0, 0.0]},
                {"entity_id": "ped_enter_exit", "tick": 15, "label_class": "pedestrian", "pos_enu": [70.2, 5.0, 0.0], "vel_mps": [1.0, 0.0, 0.0]},
            ]
        }

        visible = source_runtime_visible_ticks_by_entity(
            roster_entities=roster,
            grouped=grouped,
            ticks=[0, 5, 10, 15],
            tick_hz=10,
            source_contract=source_contract,
        )

        self.assertEqual(visible["ped_enter_exit"], {5, 10})

    def test_global_uav_selection_is_reduced_to_runtime_visible_ids(self) -> None:
        selection = UavSelection(
            uav_ids=("uav_outside", "uav_inside"),
            entity_ids={"uav_outside": "global_uav_outside", "uav_inside": "global_uav_inside"},
            task_ids={"uav_outside": "task_outside", "uav_inside": "task_inside"},
            mission_type_by_uav_id={"uav_outside": "delivery", "uav_inside": "delivery"},
            selected_count=2,
            active_count_min=2,
            active_count_max=2,
            active_count_mean=2.0,
            candidate_count=2,
            observable_candidate_count=2,
        )

        reduced = subset_uav_selection_for_runtime(selection, {"uav_inside"})

        self.assertEqual(reduced.uav_ids, ("uav_inside",))
        self.assertEqual(reduced.selected_count, 1)
        self.assertEqual(reduced.entity_ids, {"uav_inside": "global_uav_inside"})


class SemanticVisibilityOutputContractTest(unittest.TestCase):
    def test_validate_sample_row_accepts_sidecar_only_logical_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "sample" / "rgb" / "tick_000100.png"
            seg_path = root / "sample" / "seg" / "tick_000100.png"
            rgb_path.parent.mkdir(parents=True)
            seg_path.parent.mkdir(parents=True)
            Image.new("RGB", (3, 2), color=(12, 34, 56)).save(rgb_path)
            seg = Image.new("L", (3, 2))
            seg.putdata([1, 5, 6, 7, 2, 1])
            seg.save(seg_path)
            common_sidecar = {
                "episode_id": "episode_complete",
                "tick": 100,
                "camera_name": "bottom_center",
                "requested_camera_rotation_body_deg": {"pitch_deg": -90.0, "yaw_deg": 0.0, "roll_deg": 0.0},
                "event_semantic_logical_region_policy": {
                    "logical_region_primary_segmentation_enabled": False,
                    "default_policy": "sidecar_meta_only",
                },
                "event_semantic_objects": [
                    {
                        "entity_id": "nfz",
                        "logical_asset_id": "trigger.no_fly.box.v1",
                        "spawn_logical_asset_id": "semantic.trigger_box.extent_14_10_14.v1",
                        "logical_region_label_policy": "sidecar_meta_only",
                        "primary_segmentation_includes_logical_region": False,
                    },
                    {
                        "entity_id": "corridor",
                        "logical_asset_id": "semantic.uav_corridor.segment.v1",
                        "logical_region_label_policy": "sidecar_meta_only",
                        "primary_segmentation_includes_logical_region": False,
                    },
                ],
            }
            write_json(
                rgb_path.with_suffix(".json"),
                {
                    **common_sidecar,
                    "capture_backend": "airsim_native_uav_camera",
                    "event_semantic_proxy_sanitizer": {
                        "status": "skipped",
                        "target_count": 0,
                        "sanitized_actor_count": 0,
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
                    "class_histogram": {"1": 2, "2": 1, "5": 1, "6": 1, "7": 1},
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
                            "histogram": {"1": 2, "2": 1, "5": 1, "6": 1, "7": 1},
                        },
                    }
                ),
            }

            errors, details = validate_sample_row(
                row,
                class_name_to_id=CLASS_NAME_TO_ID,
                required_seg_classes=["city_base_background", "drone", "pedestrian", "vehicle"],
                logical_classes=["hazard_trigger", "uav_corridor"],
                min_pixels_per_class=1,
                project_root=root,
            )

            self.assertEqual(errors, [])
            self.assertNotIn("11", details["seg_histogram"])
            self.assertNotIn("12", details["seg_histogram"])

            bad_seg_path = root / "sample" / "seg" / "tick_000101.png"
            bad_seg = Image.new("L", (3, 2))
            bad_seg.putdata([1, 5, 6, 7, 11, 12])
            bad_seg.save(bad_seg_path)
            write_json(
                bad_seg_path.with_suffix(".json"),
                {
                    **common_sidecar,
                    "capture_backend": "ue_custom_stencil_fixed_world_camera",
                    "segmentation_kind": "ue_custom_stencil_class_id_u8",
                    "class_histogram": {"1": 2, "2": 1, "5": 1, "6": 1, "7": 1},
                },
            )
            bad_row = dict(row)
            bad_outputs = json.loads(bad_row["modality_outputs"])
            bad_outputs["seg"] = {
                "png": str(bad_seg_path),
                "sidecar": str(bad_seg_path.with_suffix(".json")),
                "histogram": {"1": 2, "2": 1, "5": 1, "6": 1, "7": 1},
            }
            bad_row["modality_outputs"] = json.dumps(bad_outputs)

            bad_errors, _ = validate_sample_row(
                bad_row,
                class_name_to_id=CLASS_NAME_TO_ID,
                required_seg_classes=["city_base_background", "drone", "pedestrian", "vehicle"],
                logical_classes=["hazard_trigger", "uav_corridor"],
                min_pixels_per_class=1,
                project_root=root,
            )

            self.assertTrue(any("hazard_trigger" in error and "primary seg histogram" in error for error in bad_errors))
            self.assertTrue(any("uav_corridor" in error and "primary seg histogram" in error for error in bad_errors))


if __name__ == "__main__":
    unittest.main()
