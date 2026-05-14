# SumoImporter

This is the main plugin root for external Python development.

If the work starts from Python, scenario generation, map-source materials, the canonical low-altitude semantic event-chain, capture orchestration, or postprocess, start here first.

## Read Order

1. `README.md`
   Read this file first for ownership, directory conventions, and runtime flow.
2. `DEVELOPMENT_LAYOUT.md`
   Read this for concrete placement rules across plugins.
3. `Scripts/episode_render_host.py`
   Main runtime executor that consumes `ScenarioPackage` outputs and drives UE/AirSim.
4. `Scenarios/donghu_dense_uav_rain_fall/scripts/build.py`
   Current working example of a canonical scenario generator.
5. `Demos/multiview/build_multiview_demo_assets.py`
   Current postprocess entrypoint for GIF/timeline generation.
6. `Scripts/donghu_core/`
   Shared Python service layer. Reusable logic should go here, not be duplicated in CLIs.

The only supported low-altitude pipeline is:

`spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`

No alternate capture path is supported.

## Design Idea

The intended architecture is split into four layers:

- `Map Build`
  Upstream map-source materials and derived topology inputs.
- `Scenario Package`
  Per-scenario truth, semantic actor rosters, weather, plans, render-ready sidecars, and manifests.
- `Runtime Executor`
  UE/AirSim execution driven from package outputs.
- `Postprocess`
  GIFs, timelines, and presentation artifacts built from runtime outputs.

Only the runtime layer should talk directly to the UE bridge.
Scenario generation should produce data packages, not directly mutate the world.

## What External Python Should Read

- `Scripts/aero_sim_client.py`
  Python wrapper around the UE bridge RPC surface.
- `Scripts/episode_render_host_config.json`
  Main runtime config for package playback and capture.
  UAV host runs require explicit `--airsim-capture-entity` and `--capture-view-id`; the event-chain runner creates those runs by rotating across every active scene UAV.
- `Config/LowAltitude/Maps/<map_id>/map_package.json`
  Current summary of runtime map config and plugin-owned map-source inputs.
- `Scripts/episode_template_resolver.json`
  Entity-mode resolution between truth entities and runtime behavior.
- `Scripts/episode_capture_presets.json`
  Camera and modality defaults. These do not replace explicit capture entity/view ids.

## Main Runtime Flow

1. Build or update a `ScenarioPackage`.
2. Run `Scripts/episode_render_host.py`.
3. The executor loads truth frames, weather, scenario plan, and capture plan.
   The executor also loads semantic actor rosters and dynamic labels when they are present in the package.
4. The executor loads UE runtime context for `map_id`.
5. Per tick:
   apply weather
   apply scene-sync semantic actors
   update truth-driven semantic actors, including background vehicles and pedestrians
   enforce physical motion for all relevant entities
  sync scene UAVs from truth frames
  trigger all cameras once for that tick by pinning `CaptureUAV_0` to the selected scene UAV
   write sidecars and images
6. Run `Demos/multiview/build_multiview_demo_assets.py` for GIF/timeline outputs.

Capture tasks are valid only when they provide stable `--airsim-capture-entity` and `--capture-view-id`; missing values fail closed.

## APIs That Matter

- UE bridge API is surfaced through `Scripts/aero_sim_client.py`.
- The authoritative UE-side contract is documented in:
  `../AeroSimHost/AERO_API_REFERENCE.md`

For Python developers, the most important RPC families are:

- `load_context`
- `apply_frame`
- `poll_feedback`
- `project_ground`
- pedestrian RPCs such as `ped_spawn`, `ped_reset`, `ped_play_animation`
- `apply_weather`
- `capture_world_camera`

Background vehicles and pedestrians are semantic actors, not decoration, and their motion and state belong in the generated package outputs.

## Directory Rules

- Long-lived Python entrypoints:
  `Scripts/`
- Shared Python service logic:
  `Scripts/donghu_core/`
- Temporary check scripts:
  `Scripts/dev_checks/`
- Plugin-owned map-source inputs:
  `Maps/<map_id>/source/`
- Future scenario families:
  `Scenarios/<scenario_id>/`

## If We Build 30 Scenarios

Each scenario family gets its own folder:

- `Scenarios/<scenario_id>/spec/`
- `Scenarios/<scenario_id>/scripts/`
- `Scenarios/<scenario_id>/artifacts/`
- `Scenarios/<scenario_id>/notes/`

Do not keep adding new scenario families under `Scripts/`.
The current Donghu fall demo already lives in `Scenarios/donghu_dense_uav_rain_fall/` and should be treated as the canonical reference layout.

## Where To Put Materials

- Python-side scenario definitions, event chains, capture plans, and generated package artifacts:
  keep in `SumoImporter`
- UE assets such as flower pots, umbrellas, barricades, cones, and other reusable props:
  put in `../AeroWorldContent`

See:

- `DEVELOPMENT_LAYOUT.md`
- `Maps/donghu_road_topo/README.md`
- `Scenarios/README.md`
- `../AeroWorldContent/SCENARIO_ASSET_LAYOUT.md`
