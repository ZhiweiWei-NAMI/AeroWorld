from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Plugins" / "SumoImporter" / "Scripts"))

from batch_generate import keyframes_for, sample_keyframes  # noqa: E402
from donghu_core.event_script_interpreter import EventScriptInterpreter  # noqa: E402
from pedestrian_activity_catalog import (  # noqa: E402
    get_activity,
    normalize_activity_type,
    validate_local_animation_assets,
)


ROOT = Path(__file__).resolve().parents[2]


class PedestrianActivityCatalogTest(unittest.TestCase):
    def test_catalog_animation_assets_exist_locally(self) -> None:
        self.assertEqual(validate_local_animation_assets(ROOT), [])

    def test_unknown_activity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_activity_type("story_only_activity")

    def test_activity_movement_flags_are_explicit(self) -> None:
        self.assertTrue(get_activity("crossing", moving=True).moving)
        self.assertTrue(get_activity("texting_walk", moving=True).moving)
        self.assertFalse(get_activity("phone_call").moving)
        self.assertFalse(get_activity("medical_incident").moving)


class PedestrianActivityTimelineTest(unittest.TestCase):
    def test_move_uses_activity_then_stationary_post_activity(self) -> None:
        frames = keyframes_for(
            [0.0, 0.0, 0.0],
            [
                {
                    "tick": 10,
                    "waypoints_enu_m": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    "velocity_mps": 1.0,
                    "activity_type": "crossing",
                    "post_activity_type": "observing",
                }
            ],
            "waiting",
            pedestrian=True,
        )

        mid_pos, mid_vel, mid_state = sample_keyframes(frames, 20)
        end_pos, end_vel, end_state = sample_keyframes(frames, 30)

        self.assertEqual(mid_state, "crossing")
        self.assertGreater(mid_vel[0], 0.0)
        self.assertEqual(end_pos, [2.0, 0.0, 0.0])
        self.assertEqual(end_vel, [1.0, 0.0, 0.0])
        self.assertEqual(end_state, "observing")

    def test_activity_schedule_preserves_waiting_position(self) -> None:
        frames = keyframes_for(
            [1.0, 2.0, 0.0],
            [{"type": "activity", "tick": 5, "activity_type": "phone_call"}],
            "waiting",
            pedestrian=True,
        )

        pos, vel, state = sample_keyframes(frames, 10)

        self.assertEqual(pos, [1.0, 2.0, 0.0])
        self.assertEqual(vel, [0.0, 0.0, 0.0])
        self.assertEqual(state, "phone_call")


class EventScriptActivityStateTest(unittest.TestCase):
    def test_interpreter_tracks_activity_without_new_rpc_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "event_script.json"
            script_path.write_text(
                json.dumps(
                    {
                        "schema_name": "event_script_v1",
                        "scenario_id": "activity_unit",
                        "triggers": [],
                        "events": [],
                    }
                ),
                encoding="utf-8",
            )
            interpreter = EventScriptInterpreter(script_path)
            interpreter.update_entity_state("ped_01", [0.0, 0.0, 0.0])
            interpreter.update_entity_activity("ped_01", "phone_call")

            self.assertEqual(interpreter.get_entity_activity("ped_01"), "phone_call")
            self.assertEqual(interpreter.entity_states["ped_01"].activity_type, "phone_call")


if __name__ == "__main__":
    unittest.main()
