# L6-1_v1: Link loss, successful RTH and landing

- **Event Type**: L6-1 — C2 Link Complete Loss
- **ODD Layer**: L6 (L6)
- **Mechanism**: failure
- **SORA SAIL**: V
- **CAAC Reference**: CAAC-9 (C2 link loss exceeding 30 seconds)
- **Severity**: major
- **Belcastro Domain**: Vehicle:Communication System Failures + GCS:Command and Control Link Loss

## Causal Chain
C2 signal loss → lost-link timer start → autonomous mode → pre-programmed RTH → attempt reconnection → land if no recovery

## Entities
uav, radio_tower, landing_pad

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
