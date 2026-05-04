# L2-2_v2: Sudden GNSS denial (urban interference)

- **Event Type**: L2-2 — GNSS Signal Degradation
- **ODD Layer**: L2 (L2)
- **Mechanism**: failure
- **SORA SAIL**: III-IV
- **CAAC Reference**: CAAC-4 related (obstacle strike due to nav error)
- **Severity**: major
- **Belcastro Domain**: Vehicle:Navigation and Guidance Errors

## Causal Chain
urban canyon multipath → GNSS error growth → position uncertainty → degraded navigation → potential geofence breach

## Entities
uav, buildings

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
