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

## Contract
- Physically animated semantic chain with one clear key event and continuous interaction from start to terminal state.
- Background vehicles and pedestrians are semantic actors, not decoration; their motion and roles remain part of the episode.
- `U_inspect` is a long-lived moving inspect-view substitute, not a static hover, and must stay in motion across the episode.

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
