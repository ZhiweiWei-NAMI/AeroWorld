# L6-2_v1: Moderate degradation, operator compensates

- **Event Type**: L6-2 — C2 Link Intermittent Degradation
- **ODD Layer**: L6 (L6)
- **Mechanism**: failure
- **SORA SAIL**: III-IV
- **CAAC Reference**: Non-emergency (link quality degradation)
- **Severity**: minor
- **Belcastro Domain**: Vehicle:Communication System Failures

## Causal Chain
signal interference → increased latency → control lag → trajectory oscillation → operator switches to higher power mode

## Entities
uav, radio_tower

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
