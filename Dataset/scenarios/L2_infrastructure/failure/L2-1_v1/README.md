# L2-1_v1: Single station failure, graceful degradation

- **Event Type**: L2-1 — Communication Station Failure
- **ODD Layer**: L2 (L2)
- **Mechanism**: failure
- **SORA SAIL**: III
- **CAAC Reference**: Non-emergency (infrastructure failure)
- **Severity**: minor
- **Belcastro Domain**: UTM:UTM Communication Infrastructure Issues

## Causal Chain
base station power loss → C2 signal degradation → UAV switches to backup link → reduced operational bandwidth

## Contract
- Physically animated semantic chain with one clear key event and continuous interaction from start to terminal state.
- Background vehicles and pedestrians are semantic actors, not decoration; their motion and roles remain part of the episode.
- `U_inspect` is a long-lived moving inspect-view substitute, not a static hover, and must stay in motion across the episode.

## Entities
radio_tower, uav, backup_station

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
