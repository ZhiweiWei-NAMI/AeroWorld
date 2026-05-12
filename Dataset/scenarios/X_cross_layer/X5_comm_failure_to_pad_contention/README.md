# X5_comm_failure_to_pad_contention

- Scenario: Communication failure to pad contention chain
- Key event: `station fail > pad contention`
- Contract: `UAV 3 / vehicle 3 / pedestrian 4 / facility 5 / logical 10`
- Inspect: `U_inspect`, `I18`, long-lived inspect-view substitute, must move for the full episode, min path length 80 m, orbit/racetrack motion
- Semantic actors: background vehicles and pedestrians are semantic actors, not decoration
- Background V/P: vehicles `facility access traffic`; pedestrians `waiting near facility`
- Weather: `clear`
- Chain: physically animated semantic chain with continuous interaction across airspace, infrastructure, and ground layers
- Files: `event_script.json`, `scene_setup.json`, `spec.py`
