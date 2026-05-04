# L6-3_v1: Subtle drift spoofing, gradual deviation

- **Event Type**: L6-3 — GNSS Spoofing Attack
- **ODD Layer**: L6 (L6)
- **Mechanism**: failure
- **SORA SAIL**: V-VI
- **CAAC Reference**: CAAC-8 related (navigation system failure due to external attack)
- **Severity**: critical
- **Belcastro Domain**: UTM:Security Threats + Vehicle:Navigation and Guidance Errors

## Causal Chain
spoofing signal → false position fix → autopilot follows false track → geofence approach → anomaly detection → countermeasure

## Entities
uav, no_fly_zone

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
