from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_generate import apply_ambient_ground_motion, keep_ground_entities_moving, route_motion_schedule  # noqa: E402


class GroundFlowGenerationTest(unittest.TestCase):
    def test_continuous_ground_flow_schedule_preserves_route_without_bounce(self) -> None:
        route = [[100.0, 0.0, 0.0], [180.0, 20.0, 0.0]]
        schedules = route_motion_schedule(
            scenario_id="S",
            entity_id="bg_vehicle",
            label_class="vehicle",
            initial_pos=[0.0, 0.0, 0.0],
            route_waypoints=route,
            initial_state="traffic_flow",
            ground_flow_contract={
                "policy": "continuous_capture_ground_flow_v1",
                "speed_mps": 6.0,
                "route_duration_ticks": 900,
            },
        )

        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0]["waypoints_enu_m"], route)

    def test_continuous_ground_flow_rows_are_not_overwritten_by_ambient_motion(self) -> None:
        rows = [
            {
                "entity_id": "bg_vehicle",
                "label_class": "vehicle",
                "tick": tick,
                "pos_enu": [float(tick), 10.0, 0.0],
                "vel_mps": [1.0, 0.0, 0.0],
            }
            for tick in range(3)
        ]
        original_rows = copy.deepcopy(rows)
        entities = {
            "bg_vehicle": {
                "entity_id": "bg_vehicle",
                "label_class": "vehicle",
                "background_vehicle": {"policy": "semantic_event_contract_v1"},
                "ground_flow_contract": {"policy": "continuous_capture_ground_flow_v1"},
            }
        }

        apply_ambient_ground_motion(rows, entities, duration_ticks=900)
        keep_ground_entities_moving(rows, entities, duration_ticks=900)

        self.assertEqual(rows, original_rows)


if __name__ == "__main__":
    unittest.main()
