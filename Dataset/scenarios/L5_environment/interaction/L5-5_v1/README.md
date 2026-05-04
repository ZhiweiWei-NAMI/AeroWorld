# L5-5_v1: High temperature, battery derating, shortened mission

- **Event Type**: L5-5 — Extreme Temperature Event
- **ODD Layer**: L5 (L5)
- **Mechanism**: environmental
- **SORA SAIL**: II
- **CAAC Reference**: CAAC-8 related (battery/power failure due to temperature)
- **Severity**: minor
- **Belcastro Domain**: Operations:Environmental/Weather Hazards

## Causal Chain
temperature extreme → battery internal resistance change → reduced endurance → early RTH → mission incomplete

## Entities
uav

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
