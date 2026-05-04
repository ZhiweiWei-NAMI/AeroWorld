# L1-3_v1: Single intruder, detected and avoided

- **Event Type**: L1-3 — Non-Cooperative UAV Intrusion
- **ODD Layer**: L1 (L1)
- **Mechanism**: violation
- **SORA SAIL**: IV-V
- **CAAC Reference**: CAAC-10 (unauthorized entry)
- **Severity**: major
- **Belcastro Domain**: UTM:Airspace Integration Failures

## Causal Chain
intruder detection → threat assessment → evasive maneuver / forced landing → airspace lock

## Entities
cooperative_uav, intruder_uav, no_fly_zone

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
