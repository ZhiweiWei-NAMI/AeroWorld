#!/usr/bin/env python3
"""Unit tests for EventScriptInterpreter (no UE required)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPTS = SCRIPT_DIR.parents[0] / "Plugins" / "SumoImporter" / "Scripts"
if str(REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS))

from donghu_core.event_script_interpreter import EventScriptInterpreter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(triggers: list[dict], events: list[dict], **kwargs: Any) -> Path:
    """Write a minimal event script to a temp file and return its Path."""
    doc: dict[str, Any] = {
        "$schema": "event_script_v1",
        "scenario_id": "test.scenario",
        "triggers": triggers,
        "events": events,
    }
    doc.update(kwargs)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(doc, tmp)
    tmp.close()
    return Path(tmp.name)


def _spy_handler() -> tuple[list[dict], Any]:
    """Return (calls_list, handler) where handler appends every action dict to calls_list."""
    calls: list[dict] = []

    def handler(action: dict) -> dict:
        calls.append(dict(action))
        return {"status": "ok"}

    return calls, handler


# ---------------------------------------------------------------------------
# Tests: tick trigger
# ---------------------------------------------------------------------------


def test_tick_trigger_fires_at_exact_tick() -> None:
    script = _make_script(
        triggers=[{"trigger_id": "t100", "type": "tick", "tick": 100}],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t100",
                "max_fire_count": 1,
                "actions": [{"action_id": "act1", "type": "log", "msg": "fired"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("log", handler)

    # Before tick 100, nothing fires
    r99 = interp.tick(99)
    assert len(r99) == 0
    assert len(calls) == 0

    # At tick 100, it fires
    r100 = interp.tick(100)
    assert len(r100) == 1
    assert r100[0]["type"] == "log"
    assert len(calls) == 1
    assert calls[0]["msg"] == "fired"

    # tick 101 — should NOT fire again (max_fire_count=1)
    r101 = interp.tick(101)
    assert len(r101) == 0
    assert len(calls) == 1


def test_tick_trigger_max_fire_count_zero() -> None:
    """max_fire_count=0 means unlimited fires."""
    script = _make_script(
        triggers=[{"trigger_id": "t0", "type": "tick", "tick": 0}],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t0",
                "max_fire_count": 0,
                "actions": [{"type": "log"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("log", handler)

    assert len(interp.tick(0)) == 1
    assert len(interp.tick(1)) == 1
    assert len(interp.tick(2)) == 1
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Tests: weather_state trigger
# ---------------------------------------------------------------------------


def test_weather_state_trigger_activates() -> None:
    script = _make_script(
        triggers=[
            {
                "trigger_id": "rain_heavy",
                "type": "weather_state",
                "parameter": "rain",
                "operator": "gte",
                "value": 0.5,
            }
        ],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "rain_heavy",
                "max_fire_count": 1,
                "actions": [{"type": "log", "msg": "rain"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("log", handler)

    # No rain yet
    interp.update_weather_state({"rain": 0.0})
    assert len(interp.tick(0)) == 0

    # Light rain — not enough
    interp.update_weather_state({"rain": 0.3})
    assert len(interp.tick(1)) == 0

    # Heavy rain — fires
    interp.update_weather_state({"rain": 0.7})
    assert len(interp.tick(2)) == 1
    assert calls[0]["msg"] == "rain"


def test_weather_state_sustain_ticks() -> None:
    script = _make_script(
        triggers=[
            {
                "trigger_id": "rain",
                "type": "weather_state",
                "parameter": "rain",
                "operator": "gte",
                "value": 0.5,
                "sustain_ticks": 5,
            }
        ],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "rain",
                "max_fire_count": 1,
                "actions": [{"type": "log"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("log", handler)

    # Activate
    interp.update_weather_state({"rain": 0.8})
    assert interp.tick(0)
    assert interp.trigger_states["rain"].is_active

    # Condition drops but sustain keeps it active
    interp.update_weather_state({"rain": 0.0})
    for t in range(1, 6):
        interp.tick(t)
        assert interp.trigger_states["rain"].is_active, f"sustain should hold at tick {t}"

    # After sustain window, inactive
    interp.tick(7)
    assert not interp.trigger_states["rain"].is_active


# ---------------------------------------------------------------------------
# Tests: entity_proximity trigger
# ---------------------------------------------------------------------------


def test_entity_proximity_trigger() -> None:
    script = _make_script(
        triggers=[
            {
                "trigger_id": "near",
                "type": "entity_proximity",
                "entity_a": "drone",
                "entity_b": "ped",
                "distance_m": 10.0,
                "operator": "lte",
                "min_true_ticks": 2,
            }
        ],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "near",
                "max_fire_count": 1,
                "actions": [{"type": "alert"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("alert", handler)

    interp.update_entity_state("drone", (0, 0, 0))
    interp.update_entity_state("ped", (100, 0, 0))  # 100m apart

    assert len(interp.tick(0)) == 0  # too far

    interp.update_entity_state("ped", (5, 0, 0))  # 5m apart
    assert len(interp.tick(1)) == 0  # min_true_ticks=2, first tick
    assert len(interp.tick(2)) == 1  # second consecutive tick, fires
    assert calls[0]["type"] == "alert"


# ---------------------------------------------------------------------------
# Tests: event_fired trigger (chaining)
# ---------------------------------------------------------------------------


def test_event_fired_trigger_chain() -> None:
    script = _make_script(
        triggers=[
            {"trigger_id": "t100", "type": "tick", "tick": 100},
            {"trigger_id": "e1_done", "type": "event_fired", "event_id": "e1"},
        ],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t100",
                "max_fire_count": 1,
                "actions": [{"type": "step1"}],
            },
            {
                "event_id": "e2",
                "trigger_ref": "e1_done",
                "max_fire_count": 1,
                "actions": [{"type": "step2"}],
            },
        ],
    )
    interp = EventScriptInterpreter(script)
    calls1, h1 = _spy_handler()
    calls2, h2 = _spy_handler()
    interp.register_handler("step1", h1)
    interp.register_handler("step2", h2)

    assert len(interp.tick(100)) == 1  # e1 fires (tick trigger)
    assert len(calls1) == 1

    # On the next tick, event_fired trigger becomes active, e2 fires
    assert len(interp.tick(101)) == 1  # e2 fires (event_fired trigger)
    assert len(calls2) == 1


# ---------------------------------------------------------------------------
# Tests: composite trigger (AND / OR)
# ---------------------------------------------------------------------------


def test_composite_and_trigger() -> None:
    script = _make_script(
        triggers=[
            {"trigger_id": "t100", "type": "tick", "tick": 100},
            {
                "trigger_id": "rain",
                "type": "weather_state",
                "parameter": "rain",
                "operator": "gte",
                "value": 0.5,
            },
            {
                "trigger_id": "tick_and_rain",
                "type": "composite",
                "operator": "AND",
                "children": ["t100", "rain"],
            },
        ],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "tick_and_rain",
                "max_fire_count": 1,
                "actions": [{"type": "both"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("both", handler)

    # tick >= 100 but no rain
    interp.tick(100)
    assert interp.trigger_states["t100"].is_active
    assert not interp.trigger_states["rain"].is_active
    assert not interp.trigger_states["tick_and_rain"].is_active
    assert len(calls) == 0

    # rain arrives
    interp.update_weather_state({"rain": 0.7})
    interp.tick(101)
    assert interp.trigger_states["tick_and_rain"].is_active
    assert len(calls) == 1
    assert calls[0]["type"] == "both"


def test_composite_or_trigger() -> None:
    script = _make_script(
        triggers=[
            {"trigger_id": "t100", "type": "tick", "tick": 100},
            {"trigger_id": "t200", "type": "tick", "tick": 200},
            {
                "trigger_id": "either",
                "type": "composite",
                "operator": "OR",
                "children": ["t100", "t200"],
            },
        ],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "either",
                "max_fire_count": 1,
                "actions": [{"type": "fired"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("fired", handler)

    # OR triggers fire at first child condition met
    assert len(interp.tick(100)) == 1  # t100 activates, so "either" is true
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Tests: require_conditions
# ---------------------------------------------------------------------------


def test_require_conditions_blocks_event() -> None:
    script = _make_script(
        triggers=[
            {"trigger_id": "t100", "type": "tick", "tick": 100},
            {
                "trigger_id": "rain",
                "type": "weather_state",
                "parameter": "rain",
                "operator": "gte",
                "value": 0.5,
            },
        ],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t100",
                "max_fire_count": 1,
                "require_conditions": ["rain"],
                "actions": [{"type": "conditional"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("conditional", handler)

    # Tick trigger active, but require_conditions (rain) not met
    interp.tick(100)
    assert len(calls) == 0

    # Now rain is active too
    interp.update_weather_state({"rain": 0.8})
    interp.tick(101)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Tests: emit_events (on_fire chaining)
# ---------------------------------------------------------------------------


def test_emit_events_chains_to_downstream() -> None:
    script = _make_script(
        triggers=[
            {"trigger_id": "t100", "type": "tick", "tick": 100},
        ],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t100",
                "max_fire_count": 1,
                "actions": [{"type": "first"}],
                "on_fire": {"emit_events": ["e2"]},
            },
            {
                "event_id": "e2",
                "max_fire_count": 1,
                "actions": [{"type": "second"}],
            },
        ],
    )
    interp = EventScriptInterpreter(script)
    calls1, h1 = _spy_handler()
    calls2, h2 = _spy_handler()
    interp.register_handler("first", h1)
    interp.register_handler("second", h2)

    r = interp.tick(100)
    assert len(r) == 2  # both e1 and e2 fire
    assert len(calls1) == 1
    assert len(calls2) == 1
    assert calls1[0]["type"] == "first"
    assert calls2[0]["type"] == "second"


# ---------------------------------------------------------------------------
# Tests: cooldown_ticks
# ---------------------------------------------------------------------------


def test_cooldown_prevents_refire() -> None:
    script = _make_script(
        triggers=[{"trigger_id": "always", "type": "tick", "tick": 0}],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "always",
                "max_fire_count": 0,
                "cooldown_ticks": 5,
                "actions": [{"type": "log"}],
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("log", handler)

    r0 = interp.tick(0)
    assert len(r0) == 1  # fires at tick 0

    for t in range(1, 5):
        r = interp.tick(t)
        assert len(r) == 0, f"should be on cooldown at tick {t}"

    r6 = interp.tick(6)
    assert len(r6) == 1  # cooldown expired


# ---------------------------------------------------------------------------
# Tests: $param resolution
# ---------------------------------------------------------------------------


def test_param_resolution() -> None:
    script = _make_script(
        triggers=[{"trigger_id": "t1", "type": "tick", "tick": "$param.start_tick"}],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t1",
                "max_fire_count": 1,
                "actions": [
                    {
                        "type": "spawn",
                        "position": "$param.spawn_pos",
                        "count": "$param.num_entities",
                    }
                ],
            }
        ],
        parameters={"start_tick": 300, "spawn_pos": [10, 20, 0], "num_entities": 5},
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("spawn", handler)

    # Should NOT fire at tick 100 (resolved start_tick = 300)
    assert len(interp.tick(100)) == 0
    assert len(interp.tick(299)) == 0

    # Fires at tick 300
    r = interp.tick(300)
    assert len(r) == 1
    assert calls[0]["position"] == [10, 20, 0]
    assert calls[0]["count"] == 5


def test_param_override() -> None:
    """User-supplied parameters override script defaults."""
    script = _make_script(
        triggers=[{"trigger_id": "t1", "type": "tick", "tick": "$param.delay"}],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t1",
                "max_fire_count": 1,
                "actions": [{"type": "log"}],
            }
        ],
        parameters={"delay": 500},
    )
    # Override delay to 10
    interp = EventScriptInterpreter(script, parameters={"delay": 10})
    calls, handler = _spy_handler()
    interp.register_handler("log", handler)

    assert len(interp.tick(10)) == 1
    assert len(interp.tick(11)) == 0  # max_fire_count=1


# ---------------------------------------------------------------------------
# Tests: priority ordering
# ---------------------------------------------------------------------------


def test_priority_ordering_same_tick() -> None:
    """Events on the same tick fire in priority order (lowest first)."""
    script = _make_script(
        triggers=[{"trigger_id": "t0", "type": "tick", "tick": 0}],
        events=[
            {
                "event_id": "e_low",
                "trigger_ref": "t0",
                "priority": 10,
                "max_fire_count": 1,
                "actions": [{"type": "low"}],
            },
            {
                "event_id": "e_high",
                "trigger_ref": "t0",
                "priority": 1,
                "max_fire_count": 1,
                "actions": [{"type": "high"}],
            },
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("low", handler)
    interp.register_handler("high", handler)

    interp.tick(0)
    # Both should fire, and e_high (priority=1) should fire before e_low (priority=10)
    assert len(calls) == 2
    assert calls[0]["type"] == "high"
    assert calls[1]["type"] == "low"


# ---------------------------------------------------------------------------
# Tests: action_sequences
# ---------------------------------------------------------------------------


def test_action_sequence_expansion() -> None:
    script = _make_script(
        triggers=[{"trigger_id": "t0", "type": "tick", "tick": 0}],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t0",
                "max_fire_count": 1,
                "actions": [{"sequence_ref": "greet"}],
            }
        ],
        action_sequences={
            "greet": {
                "steps": [
                    {"type": "hello"},
                    {"type": "world"},
                ]
            }
        },
    )
    interp = EventScriptInterpreter(script)
    # The interpreter sees sequence type; host must handle it
    calls, handler = _spy_handler()
    interp.register_handler("sequence", handler)

    r = interp.tick(0)
    assert len(r) == 1
    # The handler receives the expanded sequence
    assert calls[0]["type"] == "sequence"
    assert len(calls[0]["steps"]) == 2


# ---------------------------------------------------------------------------
# Tests: event_log output
# ---------------------------------------------------------------------------


def test_event_log_produces_valid_rows() -> None:
    script = _make_script(
        triggers=[{"trigger_id": "t100", "type": "tick", "tick": 100}],
        events=[
            {
                "event_id": "rain_event",
                "trigger_ref": "t100",
                "max_fire_count": 1,
                "actions": [{"type": "set_weather", "profile": "rain"}],
                "log_event": {
                    "topic": "evt_rain_start",
                    "category": "weather",
                    "title": "Rain starts",
                    "severity": "warning",
                    "overlay": "weather",
                    "target_ids": ["weather_global"],
                },
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    calls, handler = _spy_handler()
    interp.register_handler("set_weather", handler)

    interp.tick(100)
    log = interp.get_event_log()
    assert len(log) == 1
    row = log[0]
    assert row["topic"] == "evt_rain_start"
    assert row["tick"] == 100
    assert row["schema_name"] == "event_trace"
    assert row["schema_version"] == "v1"
    assert row["payload"]["category"] == "weather"
    assert row["render_hints"]["severity"] == "warning"
    assert "weather_global" in row["target_ids"]


# ---------------------------------------------------------------------------
# Tests: get_event_log returns copy
# ---------------------------------------------------------------------------


def test_get_event_log_returns_copy() -> None:
    script = _make_script(
        triggers=[{"trigger_id": "t0", "type": "tick", "tick": 0}],
        events=[
            {
                "event_id": "e1",
                "trigger_ref": "t0",
                "max_fire_count": 1,
                "actions": [{"type": "log"}],
                "log_event": {"topic": "evt", "target_ids": []},
            }
        ],
    )
    interp = EventScriptInterpreter(script)
    interp.register_handler("log", lambda a: {"ok": True})
    interp.tick(0)

    log1 = interp.get_event_log()
    log1.clear()
    log2 = interp.get_event_log()
    assert len(log2) == 1, "get_event_log should return a copy"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import traceback

    tests = [
        test_tick_trigger_fires_at_exact_tick,
        test_tick_trigger_max_fire_count_zero,
        test_weather_state_trigger_activates,
        test_weather_state_sustain_ticks,
        test_entity_proximity_trigger,
        test_event_fired_trigger_chain,
        test_composite_and_trigger,
        test_composite_or_trigger,
        test_require_conditions_blocks_event,
        test_emit_events_chains_to_downstream,
        test_cooldown_prevents_refire,
        test_param_resolution,
        test_param_override,
        test_priority_ordering_same_tick,
        test_action_sequence_expansion,
        test_event_log_produces_valid_rows,
        test_get_event_log_returns_copy,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"  PASS  {test_fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {test_fn.__name__}")
            traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
