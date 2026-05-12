# L1-4_v2: Severe congestion, one UAV forced to divert

- **Event Type**: L1-4 — Airspace Corridor Congestion
- **ODD Layer**: L1 (L1)
- **Mechanism**: violation
- **SORA SAIL**: II
- **CAAC Reference**: Non-emergency (traffic management)
- **Severity**: major
- **Belcastro Domain**: UTM:Scalability Issues

## Causal Chain
demand spike → corridor saturation → queue formation → priority arbitration → sequential passage

## Contract
- Physically animated semantic chain with one clear key event and continuous interaction from start to terminal state.
- Background vehicles and pedestrians are semantic actors, not decoration; their motion and roles remain part of the episode.
- `U_inspect` is a long-lived moving inspect-view substitute, not a static hover, and must stay in motion across the episode.

## Entities
uav_1, uav_2, uav_3, corridor_zone

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
