# L2-3_v1: Charger occupied, successful reallocation

- **Event Type**: L2-3 — Charging Station Unavailable
- **ODD Layer**: L2 (L2)
- **Mechanism**: failure
- **SORA SAIL**: II
- **CAAC Reference**: Non-emergency (facility issue)
- **Severity**: minor
- **Belcastro Domain**: Operations:Procedural Deviations (mission planning)

## Causal Chain
charger occupied/failed → charging request denied → search for alternative → battery critical → emergency divert

## Contract
- Physically animated semantic chain with one clear key event and continuous interaction from start to terminal state.
- Background vehicles and pedestrians are semantic actors, not decoration; their motion and roles remain part of the episode.
- `U_inspect` is a long-lived moving inspect-view substitute, not a static hover, and must stay in motion across the episode.

## Entities
uav, charging_station_1, charging_station_2

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
