# L4-7_v2: Sudden cardiac event, passerby alerts, UAV confirms

- **Event Type**: L4-7 — Pedestrian Fall / Medical Emergency
- **ODD Layer**: L4 (L4)
- **Mechanism**: operational
- **SORA SAIL**: I
- **CAAC Reference**: CAAC-14 related (serious injury)
- **Severity**: critical
- **Belcastro Domain**: Ground domain (beyond Belcastro, CAAC-14 validated)

## Causal Chain
pedestrian falls → UAV overhead detects → alert generation → emergency dispatch → responder arrival → assistance

## Entities
pedestrian, uav, ambulance

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
