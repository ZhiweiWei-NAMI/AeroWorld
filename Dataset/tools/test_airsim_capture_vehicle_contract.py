from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PLUGIN_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "Plugins" / "SumoImporter" / "Scripts"
sys.path.insert(0, str(PLUGIN_SCRIPT_DIR))

from episode_render_host import BatchPlan, EpisodeRenderHost  # noqa: E402


class AirSimCaptureVehicleContractTest(unittest.TestCase):
    def test_missing_capture_vehicle_fails_without_runtime_add(self) -> None:
        class InnerClient:
            def ping(self) -> bool:
                return True

        class Client:
            def __init__(self) -> None:
                self.client = InnerClient()
                self.add_vehicle_called = False

            def get_settings_string(self) -> str:
                return json.dumps({"ViewMode": "FlyWithMe"})

            def list_vehicles(self) -> list[str]:
                return []

            def add_vehicle(self, *_args: object, **_kwargs: object) -> bool:
                self.add_vehicle_called = True
                return True

        class Args:
            host = "127.0.0.1"
            port = 41451

        fake_client = Client()
        host = object.__new__(EpisodeRenderHost)
        host.args = Args()
        host.client = fake_client
        host.config = {"timeouts": {"rpc_retry_count": 0}}
        host.uav_capture_backend = "airsim_native"
        host.capture_role_filters = set()
        host.airsim_capture_vehicle = "CaptureUAV_0"
        host.airsim_capture_vehicle_ready = False

        with self.assertRaisesRegex(RuntimeError, "runtime simAddVehicle is not allowed"):
            host._ensure_airsim_capture_vehicle()

        self.assertFalse(fake_client.add_vehicle_called)

    def test_uav_capture_pose_uses_truth_even_when_status_disagrees(self) -> None:
        host = object.__new__(EpisodeRenderHost)
        host.truth_frame_coordinate_space = "map_enu"
        host.entity_rotation_offset_cfg = {}
        entity = {
            "entity_id": "uav_truth",
            "entity_category": "uav",
            "truth_pose": {
                "position_enu_m": [10.0, 20.0, 30.0],
                "rotation_deg": {"pitch_deg": 1.0, "yaw_deg": 2.0, "roll_deg": 3.0},
            },
        }
        stale_status = {
            "pose": {
                "position_enu_m": [1000.0, 2000.0, 3000.0],
                "rotation_deg": {"pitch_deg": 10.0, "yaw_deg": 20.0, "roll_deg": 30.0},
            }
        }

        position, rotation = host._uav_pose_for_capture(entity, stale_status)

        self.assertEqual(position, [10.0, 20.0, 30.0])
        self.assertEqual(rotation, {"pitch_deg": 1.0, "yaw_deg": 2.0, "roll_deg": 3.0})

    def test_uav_capture_pose_requires_truth_position(self) -> None:
        host = object.__new__(EpisodeRenderHost)
        host.truth_frame_coordinate_space = "map_enu"
        host.entity_rotation_offset_cfg = {}

        with self.assertRaisesRegex(RuntimeError, "truth_pose.position_enu_m"):
            host._uav_pose_for_capture({"entity_id": "missing_truth", "entity_category": "uav"}, {})

    def test_uav_native_capture_does_not_mutate_runtime_camera(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.capture_calls: list[dict[str, object]] = []

            def get_camera_info(self, vehicle_name: str, camera_name: str) -> dict[str, object]:
                return {
                    "vehicle_name": vehicle_name,
                    "camera_name": camera_name,
                    "fov_degrees": 85.0,
                    "position_ned_m": [0.0, 0.0, 0.0],
                    "orientation": {"x_val": 0.0, "y_val": 0.0, "z_val": 0.0, "w_val": 1.0},
                }

            def capture_vehicle_image(self, vehicle_name: str, **kwargs: object) -> object:
                self.capture_calls.append({"vehicle_name": vehicle_name, **kwargs})
                return SimpleNamespace(width=1280, height=720)

        fake_client = Client()
        written: dict[str, object] = {}
        host = object.__new__(EpisodeRenderHost)
        host.client = fake_client
        host.config = {"timeouts": {"rpc_retry_count": 0, "camera_settle_s": 0.0}}
        host.active_airsim_capture_entity_id = "uav_truth"
        host.airsim_capture_vehicle = "CaptureUAV_0"
        host.capture_presets = {"modalities": {"rgb": {"image_type": "Scene", "pixels_as_float": False, "compress": True}}}
        host.requested_capture_view_id = "view.truth.000"
        host.truth_frame_coordinate_space = "map_enu"
        host.entity_rotation_offset_cfg = {}
        host.logical_region_primary_segmentation_enabled = False
        host.event_semantic_proxy_capture_targets = []
        host.episode_id = "episode"
        host.map_id = "map"
        host.uav_scene_control_backend = "truth_frame_scene_sync"
        host.event_semantic_coordinate_audit = []
        host.event_semantic_proxy_sanitizer_result = {}
        host.static_map_coordinate_audit = []
        host._ensure_airsim_capture_vehicle = lambda: None
        host._pin_airsim_capture_vehicle = lambda position, rotation, context: {
            "requested_position_enu_m": list(position),
            "requested_rotation_deg": dict(rotation),
            "pose": {"position_ned_m": [1.0, 2.0, -3.0]},
            "pose_error_m": 0.0,
            "capture_pose_mode": "test_truth_pin",
        }
        host._apply_airsim_semantic_proxy_capture_exclusion = lambda **_kwargs: {"status": "ok", "target_count": 0}
        host._entity_resolution = lambda _entity: {}
        host._truth_frame_uses_map_enu = lambda: True
        host._coordinate_audit_entry = lambda **_kwargs: {}
        host._coordinate_space_contract = lambda: {}
        host._event_semantic_logical_region_policy = lambda: {}
        host._write_airsim_native_capture_output = lambda *args, **kwargs: written.update(kwargs)

        host._capture_uav_airsim_native_modality(
            BatchPlan(batch_id="batch", site_id="site", roi_id="roi", tick_start=10, tick_end=10),
            {"tick": 10, "frame_id": "f10", "frame_seq": 10, "sim_time_s": 1.0, "roster_summary": {}},
            modality_id="rgb",
            entity_id="uav_truth",
            entity={
                "entity_id": "uav_truth",
                "entity_category": "uav",
                "truth_pose": {
                    "position_enu_m": [10.0, 20.0, 30.0],
                    "rotation_deg": {"pitch_deg": 1.0, "yaw_deg": 2.0, "roll_deg": 3.0},
                },
            },
            vehicle_status={},
            camera_id="uav_truth__nadir_down",
            camera_name="bottom_center",
            preset={
                "camera_id_suffix": "nadir_down",
                "camera_name": "bottom_center",
                "fov_degrees": 85.0,
                "width": 1280,
                "height": 720,
                "set_capture_camera_pose": False,
                "camera_pose_frame": "ned",
                "camera_offset_body_ned_m": [0.0, 0.0, 0.0],
                "fixed_rotation_offset_deg": {"pitch_deg": -90.0, "yaw_deg": 0.0, "roll_deg": 0.0},
            },
            weather_payload={},
            entity_records=[],
            feedback_payload={},
            uav_debug={},
        )

        self.assertEqual(len(fake_client.capture_calls), 1)
        self.assertEqual(dict(written["common_sidecar"])["camera_info_before_capture"]["fov_degrees"], 85.0)


if __name__ == "__main__":
    unittest.main()
