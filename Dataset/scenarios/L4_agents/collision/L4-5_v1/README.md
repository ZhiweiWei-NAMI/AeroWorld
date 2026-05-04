# L4-5_v1: Near-miss, pedestrian startled

- **Event Type**: L4-5 — UAV-Pedestrian Near-Miss/Interaction
- **ODD Layer**: L4 (L4)
- **Mechanism**: collision
- **SORA SAIL**: V-VI
- **CAAC Reference**: CAAC-13 (collision with personnel), CAAC-14 (injury)
- **Severity**: major
- **Belcastro Domain**: Operations:Terrain/Obstacle Collision (people on ground)

## Causal Chain
UAV trajectory deviation → pedestrian in path → proximity alert → evasive action or impact → injury assessment

## Entities
uav, pedestrian

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
