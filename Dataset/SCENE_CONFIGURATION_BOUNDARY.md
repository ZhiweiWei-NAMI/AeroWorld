# Scene Configuration Boundary

This document defines the canonical scene configuration boundary for the 70-episode low-altitude semantic event-chain dataset.

## Source of Truth

- `Dataset/tools/semantic_event_contract.py`
- `Dataset/tools/regenerate_boundary_scenarios.py`

## Required Scene Semantics

- Every scene entity is intentional and grounded.
- Background vehicles and pedestrians are semantic actors, not decoration.
- Logical actors, facilities, and context anchors are part of the scene contract.
- `U_inspect` is a long-lived semantic UAV role with full-episode presence.
- L1 `U_inspect` altitude code: `I28`
- L2 `U_inspect` altitude code: `I18`

## Exact Contract Shape

- L1 scenes require at least 3 UAV, 2 vehicles, and 2 pedestrians.
- L1-4 episodes require 4 UAV.
- L2 scenes require at least 3 UAV, 2 vehicles, and 2 pedestrians.
- Every episode has exact per-scenario counts from the contract table.
- Every episode carries one required semantic event and a deterministic background semantics policy.

## Placement and Motion

- Scene entities use concrete placement modes such as `world_pose`, `lane_anchor`, `sidewalk_anchor`, `facade_anchor`, `pad_anchor`, `box_volume`, and `polygon_prism`.
- Coordinates and offsets must flow through the shared SUMO/GeoJSON coordinate services. Do not duplicate coordinate conversion code.
- Landing pads and other facilities need visual-origin offsets when their asset origin differs from the semantic anchor, following the same policy as pedestrians.
- Vehicles need lane-derived lateral and longitudinal offsets from traffic-bundle lane metadata and lane samples.
- Vehicle and pedestrian background actors must have task/state semantics in scene setup.
- `U_inspect` must use orbit/racetrack motion, not static hover.
- No compatibility fallback is allowed for scene boundary definitions.

## File Boundary

- `scene_setup.json` defines entities, placements, validation rules, cameras, and weather profile.
- `event_script.json` defines triggers, actions, and the event chain.
- `spec.py` is the compiled source scenario spec.
