from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Plugins" / "SumoImporter" / "Scripts"))

from batch_generate import build_entities, generate_trajectories, keyframes_for, sample_keyframes  # noqa: E402
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


class TrajectoryVariationTest(unittest.TestCase):
    def _rows_by_entity(self, rows: list[dict], entity_id: str) -> list[dict]:
        return [row for row in rows if row["entity_id"] == entity_id]

    def _first_moving_row(self, rows: list[dict]) -> dict | None:
        for row in rows:
            vx, vy, vz = row["vel_mps"]
            if abs(vx) + abs(vy) + abs(vz) > 1e-6:
                return row
        return None

    def _motion_rows(self, rows: list[dict]) -> list[dict]:
        return [row for row in rows if abs(row["vel_mps"][0]) + abs(row["vel_mps"][1]) + abs(row["vel_mps"][2]) > 1e-6]

    def _rounded_velocity_vectors(self, rows: list[dict]) -> set[tuple[float, float, float]]:
        return {
            (
                round(float(row["vel_mps"][0]), 6),
                round(float(row["vel_mps"][1]), 6),
                round(float(row["vel_mps"][2]), 6),
            )
            for row in rows
        }

    def _default_ped_scene(self) -> dict:
        return {
            "entities": [
                {
                    "entity_id": "ped_01",
                    "logical_asset_id": "pedestrian.adult_male.business_attire",
                    "category": "pedestrian",
                    "placement": {"resolved_position_enu_m": [0.0, 0.0, 0.0]},
                    "initial_state": {"activity_type": "waiting"},
                },
                {
                    "entity_id": "ped_02",
                    "logical_asset_id": "pedestrian.adult_female.casual",
                    "category": "pedestrian",
                    "placement": {"resolved_position_enu_m": [0.0, 1.0, 0.0]},
                    "initial_state": {"activity_type": "waiting"},
                },
            ]
        }

    def test_generate_trajectories_without_variation_is_unchanged(self) -> None:
        scene_setup = self._default_ped_scene()
        script = {
            "scenario_id": "baseline_no_variation",
            "parameters": {"duration_ticks": 220},
            "triggers": [{"trigger_id": "t0", "type": "tick", "tick": 0}],
            "events": [
                {
                    "event_id": "e0",
                    "trigger_ref": "t0",
                    "actions": [
                        {
                            "action_id": "move_main",
                            "type": "move_entity",
                            "entity_id": "ped_01",
                            "waypoints_enu_m": [[10.0, 0.0, 0.0]],
                            "velocity_mps": 1.0,
                            "activity_type": "crossing",
                            "post_activity_type": "waiting",
                        },
                        {
                            "action_id": "move_main",
                            "type": "move_entity",
                            "entity_id": "ped_02",
                            "waypoints_enu_m": [[10.0, 1.0, 0.0]],
                            "velocity_mps": 1.0,
                            "activity_type": "crossing",
                            "post_activity_type": "waiting",
                        },
                    ],
                }
            ],
        }
        rows = generate_trajectories(scene_setup, script, 120)
        ped1 = self._rows_by_entity(rows, "ped_01")
        ped2 = self._rows_by_entity(rows, "ped_02")

        ped1_motion = self._motion_rows(ped1)
        ped2_motion = self._motion_rows(ped2)
        self.assertTrue(ped1_motion)
        self.assertTrue(ped2_motion)
        self.assertAlmostEqual(ped1_motion[0]["vel_mps"][0], 1.0, places=6)
        self.assertAlmostEqual(ped2_motion[0]["vel_mps"][0], 1.0, places=6)
        self.assertAlmostEqual(ped1_motion[0]["vel_mps"][1], 0.0, places=6)
        self.assertAlmostEqual(ped2_motion[0]["vel_mps"][1], 0.0, places=6)
        self.assertEqual(self._first_moving_row(ped1)["tick"], 10)
        self.assertEqual(self._first_moving_row(ped2)["tick"], 10)
        self.assertEqual(ped1[-1]["pos_enu"], [10.0, 0.0, 0.0])
        self.assertEqual(ped2[-1]["pos_enu"], [10.0, 1.0, 0.0])

    def test_pedestrian_variation_is_deterministic_and_stably_offset(self) -> None:
        scene_setup = {
            "entities": [
                {
                    "entity_id": "ped_b",
                    "logical_asset_id": "pedestrian.adult_male.business_attire",
                    "category": "pedestrian",
                    "placement": {"resolved_position_enu_m": [0.0, 0.0, 0.0]},
                    "initial_state": {"activity_type": "waiting"},
                },
                {
                    "entity_id": "ped_x",
                    "logical_asset_id": "pedestrian.adult_female.casual",
                    "category": "pedestrian",
                    "placement": {"resolved_position_enu_m": [0.0, 1.0, 0.0]},
                    "initial_state": {"activity_type": "waiting"},
                },
            ]
        }
        script = {
            "scenario_id": "ped_var_case",
            "parameters": {"duration_ticks": 120},
            "triggers": [{"trigger_id": "t0", "type": "tick", "tick": 0}],
            "events": [
                {
                    "event_id": "e0",
                    "trigger_ref": "t0",
                    "actions": [
                        {
                            "action_id": "move_main",
                            "type": "move_entity",
                            "entity_id": "ped_b",
                            "waypoints_enu_m": [[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                            "velocity_mps": 1.0,
                            "activity_type": "crossing",
                            "post_activity_type": "waiting",
                            "trajectory_variation": {
                                "max_tick_offset_ticks": 30,
                                "velocity_jitter_ratio": 0.2,
                                "lateral_offset_m": 0.6,
                            },
                        },
                        {
                            "action_id": "move_main",
                            "type": "move_entity",
                            "entity_id": "ped_x",
                            "waypoints_enu_m": [[5.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
                            "velocity_mps": 1.0,
                            "activity_type": "crossing",
                            "post_activity_type": "waiting",
                            "trajectory_variation": {
                                "max_tick_offset_ticks": 30,
                                "velocity_jitter_ratio": 0.2,
                                "lateral_offset_m": 0.6,
                            },
                        },
                    ],
                }
            ],
        }

        entities = build_entities(scene_setup, script)
        self.assertEqual(entities["ped_b"]["schedules"][0]["waypoints_enu_m"][-1], [10.0, 0.0, 0.0])
        self.assertEqual(entities["ped_x"]["schedules"][0]["waypoints_enu_m"][-1], [10.0, 1.0, 0.0])

        rows1 = generate_trajectories(scene_setup, script, 220)
        rows2 = generate_trajectories(scene_setup, script, 220)
        self.assertEqual(rows1, rows2)

        ped1 = self._rows_by_entity(rows1, "ped_b")
        ped2 = self._rows_by_entity(rows1, "ped_x")
        ped1_first = self._first_moving_row(ped1)
        ped2_first = self._first_moving_row(ped2)
        self.assertIsNotNone(ped1_first)
        self.assertIsNotNone(ped2_first)
        self.assertNotEqual(ped1_first["tick"], ped2_first["tick"])
        self.assertNotEqual(ped1_first["vel_mps"][0], ped2_first["vel_mps"][0])

        ped1_motion = self._motion_rows(ped1)
        ped2_motion = self._motion_rows(ped2)
        self.assertTrue(ped1_motion and ped2_motion)
        self.assertLessEqual(len(self._rounded_velocity_vectors(ped1_motion)), 2)
        self.assertLessEqual(len(self._rounded_velocity_vectors(ped2_motion)), 2)
        self.assertNotAlmostEqual(ped1_motion[0]["pos_enu"][1], 0.0, places=6)
        self.assertNotAlmostEqual(ped2_motion[0]["pos_enu"][1], 1.0, places=6)
        self.assertLess(max(abs(row["pos_enu"][1]) for row in ped1_motion), 0.6)
        self.assertLess(max(abs(row["pos_enu"][1] - 1.0) for row in ped2_motion), 0.6)
        self.assertEqual(ped1[-1]["pos_enu"], [10.0, 0.0, 0.0])
        self.assertEqual(ped2[-1]["pos_enu"], [10.0, 1.0, 0.0])

    def test_vehicle_variation_changes_headway_and_speed_with_small_lateral(self) -> None:
        scene_setup = {
            "entities": [
                {
                    "entity_id": "car_01",
                    "logical_asset_id": "vehicle.sedan.standard",
                    "category": "vehicle",
                    "placement": {"resolved_position_enu_m": [0.0, 0.0, 0.0]},
                    "initial_state": {"mode": "idle"},
                },
                {
                    "entity_id": "car_02",
                    "logical_asset_id": "vehicle.sedan.standard",
                    "category": "vehicle",
                    "placement": {"resolved_position_enu_m": [0.0, 2.0, 0.0]},
                    "initial_state": {"mode": "idle"},
                },
            ]
        }
        script = {
            "scenario_id": "veh_var_case",
            "parameters": {"duration_ticks": 120},
            "triggers": [{"trigger_id": "t0", "type": "tick", "tick": 0}],
            "events": [
                {
                    "event_id": "e0",
                    "trigger_ref": "t0",
                    "actions": [
                        {
                            "action_id": "move_vehicle",
                            "type": "move_entity",
                            "entity_id": "car_01",
                            "waypoints_enu_m": [[10.0, 0.0, 0.0]],
                            "velocity_mps": 4.0,
                            "trajectory_variation": {
                                "max_tick_offset_ticks": 50,
                                "velocity_jitter_ratio": 0.15,
                                "lateral_offset_m": 0.5,
                            },
                        },
                        {
                            "action_id": "move_vehicle",
                            "type": "move_entity",
                            "entity_id": "car_02",
                            "waypoints_enu_m": [[10.0, 2.0, 0.0]],
                            "velocity_mps": 4.0,
                            "trajectory_variation": {
                                "max_tick_offset_ticks": 50,
                                "velocity_jitter_ratio": 0.15,
                                "lateral_offset_m": 0.5,
                            },
                        },
                    ],
                }
            ],
        }
        rows = generate_trajectories(scene_setup, script, 120)
        veh_a = self._rows_by_entity(rows, "car_01")
        veh_b = self._rows_by_entity(rows, "car_02")
        veh_a_first = self._first_moving_row(veh_a)
        veh_b_first = self._first_moving_row(veh_b)
        self.assertIsNotNone(veh_a_first)
        self.assertIsNotNone(veh_b_first)
        self.assertNotEqual(veh_a_first["tick"], veh_b_first["tick"])
        self.assertNotEqual(veh_a_first["vel_mps"][0], veh_b_first["vel_mps"][0])
        self.assertLess(abs(veh_a[-1]["pos_enu"][1] - 0.0), 0.2)
        self.assertLess(abs(veh_b[-1]["pos_enu"][1] - 2.0), 0.2)

    def test_density_profile_adds_group_variation_without_frame_jitter(self) -> None:
        scene_setup = self._default_ped_scene()
        script = {
            "scenario_id": "density_var_case",
            "parameters": {"duration_ticks": 140, "density_profile": "dense"},
            "triggers": [{"trigger_id": "t0", "type": "tick", "tick": 0}],
            "events": [
                {
                    "event_id": "e0",
                    "trigger_ref": "t0",
                    "actions": [
                        {
                            "action_id": "dense_walk",
                            "type": "move_entity",
                            "entity_id": "ped_01",
                            "waypoints_enu_m": [[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                            "velocity_mps": 1.2,
                            "activity_type": "crossing",
                            "post_activity_type": "waiting",
                        },
                        {
                            "action_id": "dense_walk",
                            "type": "move_entity",
                            "entity_id": "ped_02",
                            "waypoints_enu_m": [[5.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
                            "velocity_mps": 1.2,
                            "activity_type": "crossing",
                            "post_activity_type": "waiting",
                        },
                    ],
                }
            ],
        }

        entities = build_entities(scene_setup, script)
        ped1_schedule = entities["ped_01"]["schedules"][0]
        ped2_schedule = entities["ped_02"]["schedules"][0]
        self.assertNotEqual(ped1_schedule["tick"], ped2_schedule["tick"])
        self.assertNotEqual(ped1_schedule["velocity_mps"], ped2_schedule["velocity_mps"])
        self.assertEqual(ped1_schedule["waypoints_enu_m"][-1], [10.0, 0.0, 0.0])
        self.assertEqual(ped2_schedule["waypoints_enu_m"][-1], [10.0, 1.0, 0.0])

        rows = generate_trajectories(scene_setup, script, 140)
        ped1 = self._rows_by_entity(rows, "ped_01")
        ped2 = self._rows_by_entity(rows, "ped_02")
        ped1_first = self._first_moving_row(ped1)
        ped2_first = self._first_moving_row(ped2)

        self.assertIsNotNone(ped1_first)
        self.assertIsNotNone(ped2_first)
        self.assertNotEqual(ped1_first["vel_mps"][0], ped2_first["vel_mps"][0])
        self.assertEqual(ped1[-1]["pos_enu"], [10.0, 0.0, 0.0])
        self.assertEqual(ped2[-1]["pos_enu"], [10.0, 1.0, 0.0])

        for entity_rows, base_y in ((ped1, 0.0), (ped2, 1.0)):
            motion_rows = self._motion_rows(entity_rows)
            self.assertTrue(motion_rows)
            self.assertLessEqual(len(self._rounded_velocity_vectors(motion_rows)), 2)
            y_offsets = [abs(row["pos_enu"][1] - base_y) for row in motion_rows]
            self.assertLess(max(y_offsets), 0.13)


if __name__ == "__main__":
    unittest.main()
