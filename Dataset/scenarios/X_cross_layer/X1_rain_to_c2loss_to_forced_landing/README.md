# X1_rain_to_c2loss_to_forced_landing

- Scenario: Rain to C2 loss to forced landing cross-layer chain
- Key event: `rain + C2 loss > forced landing`
- Contract: `UAV 3 / vehicle 3 / pedestrian 10 / facility 3 / logical 8`
- Inspect: `U_inspect`, `I10`, long-lived inspect-view substitute, must move for the full episode, min path length 80 m, orbit/racetrack motion
- Semantic actors: background vehicles and pedestrians are semantic actors, not decoration
- Background V/P: vehicles `emergency response/hold`; pedestrians `crowd evade/recover`
- Weather: `rain`
- Chain: physically animated semantic chain with continuous interaction across airspace, infrastructure, and ground layers
- Files: `event_script.json`, `scene_setup.json`, `spec.py`
