# L1-4_v1: Moderate congestion, queued resolution

- **Event Type**: L1-4 — Airspace Corridor Congestion
- **ODD Layer**: L1 (L1)
- **Mechanism**: violation
- **SORA SAIL**: II
- **CAAC Reference**: Non-emergency (traffic management)
- **Severity**: minor
- **Belcastro Domain**: UTM:Scalability Issues

## Causal Chain
demand spike → corridor saturation → queue formation → priority arbitration → sequential passage

## Entities
uav_1, uav_2, uav_3, corridor_zone

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
