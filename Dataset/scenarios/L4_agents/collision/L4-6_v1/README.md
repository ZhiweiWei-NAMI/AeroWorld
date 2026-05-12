# L4-6_v1: physically animated semantic chain with a clear key event and continuous interaction

- **Layer**: L4
- **Contract**: U/V/P/F/L = 3/2/3/1/2
- **Inspect**: I10, long-lived U_inspect, moving inspect-view substitute, not static hover
- **Weather**: clear

## Chain
jaywalk > vehicle brake > retreat

## Actors
Entities: semantic UAVs, background vehicles/pedestrians where present, and scenario-specific facilities/logical actors.
- Background vehicle semantics: braking/yielding vehicles
- Background pedestrian semantics: jaywalker + waiting peds
- Every episode is a physically animated semantic chain with continuous interaction from the first key event through recovery/landing/resolution.

## Files
- `event_script.json`
- `scene_setup.json`
- `spec.py`
