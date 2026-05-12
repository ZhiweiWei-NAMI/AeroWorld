# X2_gnss_spoof_to_geofence_violation

- Scenario: GNSS spoofing to geofence violation chain
- Key event: `spoof > NFZ violation alert`
- Contract: `UAV 3 / vehicle 3 / pedestrian 4 / facility 2 / logical 8`
- Inspect: `U_inspect`, `I22`, long-lived inspect-view substitute, must move for the full episode, min path length 80 m, orbit/racetrack motion
- Semantic actors: background vehicles and pedestrians are semantic actors, not decoration
- Background V/P: vehicles `street reference and risk context`; pedestrians `occupied zone context`
- Weather: `clear`
- Chain: physically animated semantic chain with continuous interaction across airspace, infrastructure, and ground layers
- Files: `event_script.json`, `scene_setup.json`, `spec.py`
