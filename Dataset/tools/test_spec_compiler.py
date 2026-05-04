from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spec_compiler import (  # noqa: E402
    ActionSpec,
    EventStepSpec,
    ScenarioSpec,
    SpecCompiler,
    make_composite_trigger,
    make_event_fired_trigger,
    make_proximity_trigger,
    make_tick_trigger,
    make_weather_trigger,
)


class SpecCompilerTest(unittest.TestCase):
    def compile(self, steps: list[EventStepSpec], parameters: dict | None = None) -> dict:
        spec = ScenarioSpec(
            scenario_id="test_scenario",
            category="test",
            description="unit test",
            parameters=parameters or {},
            event_chain=steps,
        )
        return SpecCompiler().compile(spec)

    def test_compile_simple_tick_trigger(self) -> None:
        compiled = self.compile([
            EventStepSpec(
                event_id="evt_start",
                trigger=make_tick_trigger(10),
                actions=[ActionSpec("capture_screenshot", {"camera_id": "overview"})],
            )
        ])

        self.assertEqual(compiled["triggers"][0]["type"], "tick")
        self.assertEqual(compiled["triggers"][0]["tick"], 10)
        self.assertEqual(compiled["events"][0]["trigger_ref"], "trig_evt_start")

    def test_compile_event_fired_chain(self) -> None:
        compiled = self.compile([
            EventStepSpec(
                event_id="evt_a",
                trigger=make_tick_trigger(1),
                on_fire_emit=["evt_b"],
            ),
            EventStepSpec(
                event_id="evt_b",
                trigger=make_event_fired_trigger("evt_a"),
            ),
        ])

        self.assertEqual(compiled["events"][0]["on_fire"]["emit_events"], ["evt_b"])
        self.assertIn({"trigger_id": "trig_after_evt_a", "type": "event_fired", "event_id": "evt_a"}, compiled["triggers"])

    def test_compile_weather_state_trigger(self) -> None:
        compiled = self.compile([
            EventStepSpec(
                event_id="evt_weather",
                trigger=make_weather_trigger("rain", "gte", 0.7, sustain_ticks=5),
            )
        ])

        trigger = compiled["triggers"][0]
        self.assertEqual(trigger["type"], "weather_state")
        self.assertEqual(trigger["parameter"], "rain")
        self.assertEqual(trigger["operator"], "gte")
        self.assertEqual(trigger["value"], 0.7)
        self.assertEqual(trigger["sustain_ticks"], 5)

    def test_compile_entity_proximity_trigger(self) -> None:
        compiled = self.compile([
            EventStepSpec(
                event_id="evt_near",
                trigger=make_proximity_trigger("uav_1", "ped_1", 8.0, min_true_ticks=2),
            )
        ])

        trigger = compiled["triggers"][0]
        self.assertEqual(trigger["type"], "entity_proximity")
        self.assertEqual(trigger["entity_a"], "uav_1")
        self.assertEqual(trigger["entity_b"], "ped_1")
        self.assertEqual(trigger["distance_m"], 8.0)
        self.assertEqual(trigger["min_true_ticks"], 2)

    def test_compile_composite_and_trigger(self) -> None:
        compiled = self.compile([
            EventStepSpec(event_id="evt_tick", trigger=make_tick_trigger(5)),
            EventStepSpec(event_id="evt_weather", trigger=make_weather_trigger("fog", "gte", 0.5)),
            EventStepSpec(
                event_id="evt_composite",
                trigger=make_composite_trigger("AND", ["evt_tick", "evt_weather"]),
            ),
        ])

        composite = next(t for t in compiled["triggers"] if t["type"] == "composite")
        self.assertEqual(composite["operator"], "AND")
        self.assertEqual(composite["children"], ["trig_evt_tick", "trig_evt_weather"])

    def test_compile_multiple_actions(self) -> None:
        compiled = self.compile([
            EventStepSpec(
                event_id="evt_actions",
                trigger=make_tick_trigger(1),
                actions=[
                    ActionSpec("set_weather", {"profile": "rain"}),
                    ActionSpec("move_entity", {"entity_id": "uav", "waypoints_enu_m": [[0, 0, 10], [1, 1, 10]]}),
                    ActionSpec("capture_screenshot", {"camera_id": "overview"}),
                ],
            )
        ])

        self.assertEqual([a["type"] for a in compiled["events"][0]["actions"]], [
            "set_weather",
            "move_entity",
            "capture_screenshot",
        ])

    def test_param_resolution(self) -> None:
        compiled = self.compile(
            [
                EventStepSpec(
                    event_id="evt_param",
                    trigger=make_tick_trigger(1),
                    actions=[
                        ActionSpec(
                            "move_entity",
                            {
                                "entity_id": "$param.uav_id",
                                "label": "prefix $param.uav_id suffix",
                                "waypoints_enu_m": ["$param.start_pos", ["$param.literal"]],
                    },
                )
            ],
            log_topic="evt_param_topic",
            log_category="uav_mission",
            log_title="Param topic",
            log_target_ids=["$param.uav_id"],
        )
            ],
            parameters={
                "uav_id": "drone_alpha",
                "start_pos": [1.0, 2.0, 3.0],
                "literal": 'quoted "value"',
            },
        )

        action = compiled["events"][0]["actions"][0]
        self.assertEqual(action["entity_id"], "drone_alpha")
        self.assertEqual(action["label"], "prefix $param.uav_id suffix")
        self.assertEqual(action["waypoints_enu_m"][0], [1.0, 2.0, 3.0])
        self.assertEqual(action["waypoints_enu_m"][1][0], 'quoted "value"')
        self.assertEqual(compiled["events"][0]["log_event"]["target_ids"], ["drone_alpha"])

    def test_trigger_deduplication(self) -> None:
        compiled = self.compile([
            EventStepSpec(event_id="evt_a", trigger=make_tick_trigger(10)),
            EventStepSpec(event_id="evt_b", trigger=make_tick_trigger(10)),
        ])

        self.assertEqual(len(compiled["triggers"]), 1)
        self.assertEqual(compiled["events"][0]["trigger_ref"], compiled["events"][1]["trigger_ref"])

    def test_output_schema_valid(self) -> None:
        compiled = self.compile([
            EventStepSpec(
                event_id="evt_schema",
                trigger=make_tick_trigger(1),
                actions=[ActionSpec("capture_screenshot", {"camera_id": "overview"})],
                log_topic="evt_schema_topic",
                log_category="uav_mission",
                log_title="Schema test",
            )
        ])

        json.dumps(compiled)
        self.assertEqual(compiled["$schema"], "event_script_v1")
        self.assertIsInstance(compiled["triggers"], list)
        self.assertIsInstance(compiled["events"], list)
        self.assertTrue(all("trigger_id" in t and "type" in t for t in compiled["triggers"]))
        self.assertTrue(all("event_id" in e and "trigger_ref" in e and "actions" in e for e in compiled["events"]))


if __name__ == "__main__":
    unittest.main()
