# Scenario Layout

All future scenario families should live under:

`Plugins/SumoImporter/Scenarios/<scenario_id>/`

The only supported low-altitude pipeline is:

`spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`

Capture tasks are valid only when they use stable `--airsim-capture-entity` and `--capture-view-id`.
Background vehicles and pedestrians are semantic actors, not decoration, and relevant entities must have physical motion.

## Recommended Structure

- `spec/`
  Source-of-truth scenario definitions and authored metadata.

- `scripts/`
  Scenario-specific generators, validators, or import helpers.

- `artifacts/`
  Generated `ScenarioPackage` outputs:
  `truth_frames.jsonl`
  `render_ready_roster.jsonl`
  `dynamic_labels.jsonl`
  `weather_meta.jsonl`
  `scenario_plan.json`
  `capture_plan.json`
  `episode_manifest.json`

- `notes/`
  Review notes, assumptions, and manual operating guidance.

## 30-Scene Scaling Rule

If we build 30 scenes, each scene gets its own folder under `Scenarios`.
Do not mix multiple incident families into one script directory.

## Current Status

The current Donghu fall demo already lives in:

`Plugins/SumoImporter/Scenarios/donghu_dense_uav_rain_fall/`

Use that scenario as the working reference for future scenes.
