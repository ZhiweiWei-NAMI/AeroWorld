# L1-1_v1: Single boundary breach, immediate RTH

- **Event Type**: L1-1 — UAV Geofence Violation
- **ODD Layer**: L1 (L1)
- **Mechanism**: violation
- **SORA SAIL**: III-IV
- **CAAC Reference**: CAAC-10 (entering prohibited/danger/restricted zones)
- **Severity**: major
- **Belcastro Domain**: UTM:Geofence Violations

## Causal Chain
NAV error → position drift → geofence boundary crossing → RTH activation → mission abort

## Contract
- Physically animated semantic chain with one clear key event and continuous interaction from start to terminal state.
- Background vehicles and pedestrians are semantic actors, not decoration; their motion and roles remain part of the episode.
- `U_inspect` is a long-lived moving inspect-view substitute, not a static hover, and must stay in motion across the episode.

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
