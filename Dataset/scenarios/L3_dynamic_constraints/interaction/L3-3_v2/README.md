# L3-3_v2: Large evacuation zone, mass rerouting

- **Event Type**: L3-3 — Emergency Isolation Zone
- **ODD Layer**: L3 (L3)
- **Mechanism**: operational
- **SORA SAIL**: III-IV
- **CAAC Reference**: CAAC-12 (emergency response activation)
- **Severity**: critical
- **Belcastro Domain**: Operations:Emergency and Contingency Management

## Causal Chain
incident (fire/leak/crash) → hazard zone defined → perimeter broadcast → agents reroute → emergency responders enter

## Entities
hazard_zone, uav, vehicle, pedestrian

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
