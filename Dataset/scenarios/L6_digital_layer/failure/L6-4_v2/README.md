# L6-4_v2: Wideband jamming, all UAVs in area affected

- **Event Type**: L6-4 — Communication Jamming
- **ODD Layer**: L6 (L6)
- **Mechanism**: failure
- **SORA SAIL**: V
- **CAAC Reference**: Non-emergency (communication interference)
- **Severity**: critical
- **Belcastro Domain**: UTM:Security Threats (jamming)

## Causal Chain
jammer activated → noise floor increase → SNR degradation → multiple UAVs affected → frequency hopping → some links recovered

## Entities
uav_1, uav_2, radio_tower

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
