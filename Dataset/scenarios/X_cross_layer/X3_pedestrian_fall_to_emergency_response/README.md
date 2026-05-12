# X3_pedestrian_fall_to_emergency_response

- Scenario: Pedestrian fall to emergency response chain
- Key event: `fall > responder > handoff`
- Contract: `UAV 3 / vehicle 3 / pedestrian 6 / facility 2 / logical 6`
- Inspect: `U_inspect`, `I10`, long-lived inspect-view substitute, must move for the full episode, min path length 80 m, orbit/racetrack motion
- Semantic actors: background vehicles and pedestrians are semantic actors, not decoration
- Background V/P: vehicles `ambulance/yield chain`; pedestrians `fallen + bystanders`
- Weather: `clear`
- Chain: physically animated semantic chain with continuous interaction across airspace, infrastructure, and ground layers
- Files: `event_script.json`, `scene_setup.json`, `spec.py`
