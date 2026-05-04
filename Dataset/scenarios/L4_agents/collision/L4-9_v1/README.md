# L4-9_v1: Minor rear-end collision, partial blockage

- **Event Type**: L4-9 — Ground Vehicle Collision
- **ODD Layer**: L4 (L4)
- **Mechanism**: collision
- **SORA SAIL**: I
- **CAAC Reference**: Non-emergency (traffic accident)
- **Severity**: minor
- **Belcastro Domain**: Ground domain (beyond Belcastro, CAAC cross-validated)

## Causal Chain
signal violation / brake failure → collision → road blockage → traffic buildup → emergency dispatch → UAV overhead assessment

## Entities
vehicle_1, vehicle_2, uav, ambulance

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
