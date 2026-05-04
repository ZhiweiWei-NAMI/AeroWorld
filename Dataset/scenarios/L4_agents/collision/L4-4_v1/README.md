# L4-4_v1: Low-altitude UAV strikes moving vehicle

- **Event Type**: L4-4 — UAV-Ground Vehicle Collision
- **ODD Layer**: L4 (L4)
- **Mechanism**: collision
- **SORA SAIL**: V-VI
- **CAAC Reference**: CAAC-13 (collision with vehicles)
- **Severity**: critical
- **Belcastro Domain**: Vehicle:Loss of Control

## Causal Chain
UAV low altitude / descent → vehicle in path → impact → vehicle damage / traffic disruption → emergency response

## Entities
uav, vehicle

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
