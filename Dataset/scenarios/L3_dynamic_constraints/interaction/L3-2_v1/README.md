# L3-2_v1: TFR with 60s warning, orderly exit

- **Event Type**: L3-2 — Temporary No-Fly Zone Activation
- **ODD Layer**: L3 (L3)
- **Mechanism**: violation
- **SORA SAIL**: IV
- **CAAC Reference**: CAAC-10 (entering restricted zones)
- **Severity**: major
- **Belcastro Domain**: UTM:Geofence Violations

## Causal Chain
ground incident → TFR activation → zone broadcast → UAVs within zone → forced exit or immediate landing

## Entities
no_fly_zone, uav_1, uav_2

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
