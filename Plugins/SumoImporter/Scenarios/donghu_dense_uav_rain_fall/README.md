# Donghu Dense UAV Rain Fall

This is the canonical scenario directory for the current Donghu fall workflow.
It participates in the canonical low-altitude semantic event-chain pipeline and the render-ready capture host.

## Layout

- `spec/`
  Scenario source-of-truth inputs, including `scenario_spec.json`
- `scripts/`
  Scenario-local build entrypoints
- `artifacts/`
  Generated `ScenarioPackage` outputs, including truth frames, render-ready rosters, dynamic labels, capture plans, and manifests
- `notes/`
  Scenario-local notes and migration details

## Main Entry

- `scripts/build.py`

## Current Episode

- `scenario_id`
  `scenario.donghu_demo_dense_uav_rain_fall.v1`
- `episode_id`
  `episode_demo_dense_uav_rain_fall_90s`

## Boundary

This directory owns the scenario package and scenario-local build logic.
Reusable Python services stay in `Plugins/SumoImporter/Scripts/donghu_core/`.
Render-ready outputs preserve engine-emitted task metadata such as `task_id`, `role`, and `state_sequence` when present.
Formal capture tasks must provide stable `--airsim-capture-entity`, `--capture-view-id`, `--uav-capture-backend editor_hook`, and exactly one modality.
Background vehicles and pedestrians are semantic actors, not decoration; relevant entities must have physical motion.
