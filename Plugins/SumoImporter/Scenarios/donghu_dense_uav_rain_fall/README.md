# Donghu Dense UAV Rain Fall

This is the canonical scenario directory for the current Donghu fall workflow.

## Layout

- `spec/`
  Scenario source-of-truth inputs, including `scenario_spec.json`
- `scripts/`
  Scenario-local build entrypoints
- `artifacts/`
  Generated `ScenarioPackage` outputs
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
