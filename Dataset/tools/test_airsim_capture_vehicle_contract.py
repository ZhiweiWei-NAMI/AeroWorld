from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PLUGIN_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "Plugins" / "SumoImporter" / "Scripts"
sys.path.insert(0, str(PLUGIN_SCRIPT_DIR))

from episode_render_host import EpisodeRenderHost  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
