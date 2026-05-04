# L2-4_v2: Emergency UAV preempts scheduled landing

- **Event Type**: L2-4 — Landing Pad Contention
- **ODD Layer**: L2 (L2)
- **Mechanism**: failure
- **SORA SAIL**: III
- **CAAC Reference**: CAAC-12 related (emergency response at landing site)
- **Severity**: major
- **Belcastro Domain**: Operations:Emergency and Contingency Management

## Causal Chain
simultaneous landing requests → pad contention → priority assessment → one holds, one lands

## Entities
uav_1, uav_2, landing_pad

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
