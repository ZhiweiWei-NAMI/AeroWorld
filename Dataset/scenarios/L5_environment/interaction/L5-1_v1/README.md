# L5-1_v1: physically animated semantic chain with weather-linked continuous interaction

- **Layer**: L5
- **Contract**: U/V/P/F/L = 3/4/6/2/6
- **Inspect**: I22, long-lived U_inspect, moving inspect-view substitute, not static hover
- **Weather**: rain

## Chain
rain > slowdown > recovery

## Actors
Entities: semantic UAVs, semantic background vehicles/pedestrians, and weather-visible facilities/logical actors.
- Background vehicle semantics: rain-slow traffic
- Background pedestrian semantics: seek shelter/walk slower
- Every episode is a physically animated semantic chain with continuous interaction from the first key event through recovery/landing/resolution.

## Files
- `event_script.json`
- `scene_setup.json`
- `spec.py`
