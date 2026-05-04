# L5-3_v1: Steady crosswind, UAV compensates with drift

- **Event Type**: L5-3 — High Wind / Gust Event
- **ODD Layer**: L5 (L5)
- **Mechanism**: environmental
- **SORA SAIL**: III-IV
- **CAAC Reference**: CAAC-3 related (wind-induced loss of control)
- **Severity**: minor
- **Belcastro Domain**: Operations:Environmental/Weather Hazards

## Causal Chain
wind increase → UAV attitude deviation → stability compensation → exceed control authority → emergency descent

## Entities
uav, uav_with_payload

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
