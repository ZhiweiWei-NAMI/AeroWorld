# L2-5_v1: Signal all-red, vehicles stop, UAV monitors

- **Event Type**: L2-5 — Traffic Signal Malfunction
- **ODD Layer**: L2 (L2)
- **Mechanism**: operational
- **SORA SAIL**: I-II
- **CAAC Reference**: Non-emergency (traffic infrastructure)
- **Severity**: minor
- **Belcastro Domain**: Ground domain (beyond Belcastro scope, CAAC cross-validated)

## Causal Chain
signal fault → traffic flow breakdown → vehicle queue → potential conflict → UAV overhead monitoring

## Entities
traffic_light, vehicle_1, vehicle_2, uav

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
