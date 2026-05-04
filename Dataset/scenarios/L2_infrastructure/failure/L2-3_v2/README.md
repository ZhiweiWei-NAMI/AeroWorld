# L2-3_v2: No charger available, emergency landing

- **Event Type**: L2-3 — Charging Station Unavailable
- **ODD Layer**: L2 (L2)
- **Mechanism**: failure
- **SORA SAIL**: II
- **CAAC Reference**: Non-emergency (facility issue)
- **Severity**: major
- **Belcastro Domain**: Operations:Procedural Deviations (mission planning)

## Causal Chain
charger occupied/failed → charging request denied → search for alternative → battery critical → emergency divert

## Entities
uav, charging_station_1, charging_station_2

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
