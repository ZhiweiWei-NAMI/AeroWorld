# L5-4_v1: Dusk transition, gradual sensor adaptation

- **Event Type**: L5-4 — Lighting Condition Change
- **ODD Layer**: L5 (L5)
- **Mechanism**: environmental
- **SORA SAIL**: II
- **CAAC Reference**: CAAC-4 related (lighting-induced obstacle strike)
- **Severity**: minor
- **Belcastro Domain**: Operations:Night Operations Hazards

## Causal Chain
lighting change → camera exposure adjustment → temporary blindness → reduced detection range → slower speed

## Entities
uav, tunnel_entrance

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
