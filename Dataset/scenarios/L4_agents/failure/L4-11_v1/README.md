# L4-11_v1: AV sensor failure, safe stop in lane

- **Event Type**: L4-11 — Vehicle Breakdown Blocking Road
- **ODD Layer**: L4 (L4)
- **Mechanism**: failure
- **SORA SAIL**: I
- **CAAC Reference**: Non-emergency (vehicle failure)
- **Severity**: minor
- **Belcastro Domain**: Ground domain (beyond Belcastro, CAAC cross-validated)

## Causal Chain
sensor fault / fuel exhaustion → vehicle stops → lane blocked → following vehicles brake → traffic wave → UAV detects and reports

## Entities
disabled_vehicle, following_vehicle, uav

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
