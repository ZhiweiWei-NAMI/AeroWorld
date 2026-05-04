# L4-8_v1: Crowd gathering at metro exit, UAV monitors

- **Event Type**: L4-8 — Crowd Gathering / Evacuation
- **ODD Layer**: L4 (L4)
- **Mechanism**: operational
- **SORA SAIL**: I-II
- **CAAC Reference**: Non-emergency (public safety)
- **Severity**: minor
- **Belcastro Domain**: Ground domain (beyond Belcastro, CAAC cross-validated)

## Causal Chain
trigger event → crowd formation/dispersal → density change → ground risk reassessment → UAV operational adjustment

## Entities
crowd, uav, hazard_source

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
