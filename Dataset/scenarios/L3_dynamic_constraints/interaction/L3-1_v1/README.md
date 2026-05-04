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
