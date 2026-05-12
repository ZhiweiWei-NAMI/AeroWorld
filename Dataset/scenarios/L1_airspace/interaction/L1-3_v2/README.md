# L1-3_v2: Intruder ignores warnings, forced interdiction

- **Event Type**: L1-3 — Non-Cooperative UAV Intrusion
- **ODD Layer**: L1 (L1)
- **Mechanism**: violation
- **SORA SAIL**: IV-V
- **CAAC Reference**: CAAC-10 (unauthorized entry)
- **Severity**: critical
- **Belcastro Domain**: UTM:Airspace Integration Failures

## Causal Chain
intruder detection → threat assessment → evasive maneuver / forced landing → airspace lock

## Contract
- Physically animated semantic chain with one clear key event and continuous interaction from start to terminal state.
- Background vehicles and pedestrians are semantic actors, not decoration; their motion and roles remain part of the episode.
- `U_inspect` is a long-lived moving inspect-view substitute, not a static hover, and must stay in motion across the episode.

## Entities
cooperative_uav, intruder_uav, no_fly_zone

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
