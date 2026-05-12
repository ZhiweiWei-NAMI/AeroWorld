# X4_fog_to_uav_conflict

- Scenario: Fog to multi-UAV conflict cross-layer chain
- Key event: `fog > UAV conflict > evasion`
- Contract: `UAV 3 / vehicle 3 / pedestrian 4 / facility 2 / logical 10`
- Inspect: `U_inspect`, `I22`, long-lived inspect-view substitute, must move for the full episode, min path length 80 m, orbit/racetrack motion
- Semantic actors: background vehicles and pedestrians are semantic actors, not decoration
- Background V/P: vehicles `low-visibility traffic`; pedestrians `fog-context pedestrians`
- Weather: `fog`
- Chain: physically animated semantic chain with continuous interaction across airspace, infrastructure, and ground layers
- Files: `event_script.json`, `scene_setup.json`, `spec.py`
