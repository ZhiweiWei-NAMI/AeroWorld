# L1-2_v1: Gradual drift below assigned corridor

- **Event Type**: L1-2 — UAV Altitude Deviation
- **ODD Layer**: L1 (L1)
- **Mechanism**: violation
- **SORA SAIL**: II-III
- **CAAC Reference**: Non-emergency (altitude deviation)
- **Severity**: minor
- **Belcastro Domain**: Operations:Procedural Deviations

## Causal Chain
altitude control error → corridor deviation → conflict alert → corrective maneuver

## Contract
- Physically animated semantic chain with one clear key event and continuous interaction from start to terminal state.
- Background vehicles and pedestrians are semantic actors, not decoration; their motion and roles remain part of the episode.
- `U_inspect` is a long-lived moving inspect-view substitute, not a static hover, and must stay in motion across the episode.

## Entities
uav, uav_2

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
