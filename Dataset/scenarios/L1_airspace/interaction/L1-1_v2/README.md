# L1-1_v2: Boundary oscillation, repeated entry/exit

- **Event Type**: L1-1 — UAV Geofence Violation
- **ODD Layer**: L1 (L1)
- **Mechanism**: violation
- **SORA SAIL**: III-IV
- **CAAC Reference**: CAAC-10 (entering prohibited/danger/restricted zones)
- **Severity**: minor
- **Belcastro Domain**: UTM:Geofence Violations

## Causal Chain
NAV error → position drift → geofence boundary crossing → RTH activation → mission abort

## Entities
uav, no_fly_zone, landing_pad

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
