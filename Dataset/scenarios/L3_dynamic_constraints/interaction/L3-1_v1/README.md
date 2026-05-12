# L3-1_v1: Partial lane closure with detour

- **Event Type**: L3-1 — Road Construction Detour
- **ODD Layer**: L3 (L3)
- **Mechanism**: operational
- **SORA SAIL**: I
- **CAAC Reference**: Non-emergency (temporary modification)
- **Severity**: minor
- **Belcastro Domain**: Operations:Terrain/Obstacle Collision (temporary obstacle)

## Causal Chain
construction setup → lane closure → vehicle detour → traffic congestion → delayed emergency response

## Contract
- Physically animated semantic chain with one clear key event and continuous interaction from start to terminal state.
- Background vehicles and pedestrians are semantic actors, not decoration; their motion and roles remain part of the episode.
- `U_inspect` is a long-lived moving inspect-view substitute, not a static hover, and must stay in motion across the episode.

## Entities
barrier, vehicle_1, vehicle_2, construction_cone

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
