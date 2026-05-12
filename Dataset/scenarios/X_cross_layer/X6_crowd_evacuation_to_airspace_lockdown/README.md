# X6_crowd_evacuation_to_airspace_lockdown

- Scenario: Crowd evacuation to airspace lockdown chain
- Key event: `crowd evac > NFZ lockdown`
- Contract: `UAV 3 / vehicle 3 / pedestrian 10 / facility 3 / logical 12`
- Inspect: `U_inspect`, `I10`, long-lived inspect-view substitute, must move for the full episode, min path length 80 m, orbit/racetrack motion
- Semantic actors: background vehicles and pedestrians are semantic actors, not decoration
- Background V/P: vehicles `perimeter vehicles move/hold`; pedestrians `evacuation to safe zone`
- Weather: `clear/light smoke`
- Chain: physically animated semantic chain with continuous interaction across airspace, infrastructure, and ground layers
- Files: `event_script.json`, `scene_setup.json`, `spec.py`
