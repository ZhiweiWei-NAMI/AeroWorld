# L4-1_v1: Near-miss, successful avoidance

- **Event Type**: L4-1 — UAV-UAV Airspace Conflict
- **ODD Layer**: L4 (L4)
- **Mechanism**: collision
- **SORA SAIL**: V-VI
- **CAAC Reference**: CAAC-2 (mid-air collision or near-miss)
- **Severity**: major
- **Belcastro Domain**: Vehicle:Loss of Control + UTM:Traffic Management Coordination Failures

## Causal Chain
trajectory prediction → conflict detection → alert generation → evasive maneuver → separation restored

## Entities
uav_1, uav_2

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
