# L5-2_v1: Gradual fog, visibility slowly decreasing

- **Event Type**: L5-2 — Fog Onset / Low Visibility
- **ODD Layer**: L5 (L5)
- **Mechanism**: environmental
- **SORA SAIL**: III-IV
- **CAAC Reference**: CAAC-4 related (visibility-induced obstacle strike)
- **Severity**: major
- **Belcastro Domain**: Operations:Environmental/Weather Hazards

## Causal Chain
fog formation → visibility drop → visual sensors degraded → instrument navigation → speed reduction → potential obstacle hazard

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
