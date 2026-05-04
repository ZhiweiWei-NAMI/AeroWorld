# L4-2_v2: Actual building strike, UAV destroyed

- **Event Type**: L4-2 — UAV Building/Structure Strike
- **ODD Layer**: L4 (L4)
- **Mechanism**: collision
- **SORA SAIL**: V-VI
- **CAAC Reference**: CAAC-4 (striking obstacles), CAAC-13 (collision with facilities)
- **Severity**: critical
- **Belcastro Domain**: Vehicle:Loss of Control + Operations:Terrain/Obstacle Collision

## Causal Chain
navigation error / control failure → trajectory toward building → last-moment avoidance fails → impact → debris

## Entities
uav, building_structure

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
