# L6-5_v2: Undetected breach, malicious commands executed

- **Event Type**: L6-5 — Ground Control Station Compromise
- **ODD Layer**: L6 (L6)
- **Mechanism**: failure
- **SORA SAIL**: V-VI
- **CAAC Reference**: Non-emergency (cybersecurity incident)
- **Severity**: critical
- **Belcastro Domain**: GCS:Cybersecurity Threats

## Causal Chain
GCS breach → unauthorized commands → anomalous UAV behavior → operator override fails → safety lockout activates

## Entities
uav, gcs_operator

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
