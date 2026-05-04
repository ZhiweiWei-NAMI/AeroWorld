# L6-2_v2: Severe degradation near lost-link threshold

- **Event Type**: L6-2 — C2 Link Intermittent Degradation
- **ODD Layer**: L6 (L6)
- **Mechanism**: failure
- **SORA SAIL**: III-IV
- **CAAC Reference**: Non-emergency (link quality degradation)
- **Severity**: major
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
