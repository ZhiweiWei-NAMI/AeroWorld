# L4-6_v1: Pedestrian jaywalk, vehicle emergency stops

- **Event Type**: L4-6 — Pedestrian-Vehicle Conflict
- **ODD Layer**: L4 (L4)
- **Mechanism**: collision
- **SORA SAIL**: I-II
- **CAAC Reference**: CAAC-14 related (injury from vehicle)
- **Severity**: major
- **Belcastro Domain**: Ground domain (beyond Belcastro, CAAC-14 validated)

## Causal Chain
pedestrian enters roadway → vehicle approaches → conflict detection → emergency brake → near-miss / minor impact

## Entities
pedestrian, vehicle, traffic_light

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
