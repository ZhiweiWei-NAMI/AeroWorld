# L4-10_v2: Fire truck wrong-way through blocked intersection

- **Event Type**: L4-10 — Emergency Vehicle Priority Passage
- **ODD Layer**: L4 (L4)
- **Mechanism**: operational
- **SORA SAIL**: II
- **CAAC Reference**: Non-emergency (emergency vehicle operations)
- **Severity**: major
- **Belcastro Domain**: Ground domain (beyond Belcastro, CAAC cross-validated)

## Causal Chain
emergency call → dispatch → route planning → traffic yielding → corridor clearing → UAV airspace priority → arrival

## Entities
emergency_vehicle, vehicle_1, vehicle_2, uav

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
