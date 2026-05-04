# L5-1_v2: Sudden heavy downpour, immediate UAV grounding

- **Event Type**: L5-1 — Rain Transition (Gradual / Sudden)
- **ODD Layer**: L5 (L5)
- **Mechanism**: environmental
- **SORA SAIL**: II-IV
- **CAAC Reference**: CAAC-4 related (weather-induced obstacle strike)
- **Severity**: major
- **Belcastro Domain**: Operations:Environmental/Weather Hazards

## Causal Chain
rain onset → visibility decrease → UAV divert/RTH → ground wetness → pedestrian slip risk → traffic slowdown

## Entities
uav, pedestrian, vehicle

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
