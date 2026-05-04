# L4-3_v2: Uncontrolled descent near pedestrians

- **Event Type**: L4-3 — UAV Forced Landing in Populated Area
- **ODD Layer**: L4 (L4)
- **Mechanism**: failure
- **SORA SAIL**: VI
- **CAAC Reference**: CAAC-2 (crash or forced landing), CAAC-14 (death or serious injury)
- **Severity**: critical
- **Belcastro Domain**: Vehicle:Propulsion System Failures + Vehicle:Power System Failures

## Causal Chain
motor failure → altitude loss → forced landing site selection → descent → ground impact → pedestrian avoidance

## Entities
uav, pedestrian, landing_zone

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
