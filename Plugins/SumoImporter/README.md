# SumoImporter

This is the main plugin root for external Python development.

If the work starts from Python, scenario generation, map-source materials, capture orchestration, or postprocess, start here first.

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

## Design Idea

The intended architecture is split into four layers:

- `Map Build`
  Upstream map-source materials and derived topology inputs.
- `Scenario Package`
  Per-scenario truth, weather, plans, and manifests.
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
- `Config/LowAltitude/Maps/<map_id>/map_package.json`
  Current summary of runtime map config and plugin-owned map-source inputs.
- `Scripts/episode_template_resolver.json`
  Entity-mode resolution between truth entities and runtime behavior.
- `Scripts/episode_capture_presets.json`
  Camera and modality defaults.

## Main Runtime Flow

1. Build or update a `ScenarioPackage`.
2. Run `Scripts/episode_render_host.py`.
3. The executor loads truth frames, weather, scenario plan, and capture plan.
4. The executor loads UE runtime context for `map_id`.
5. Per tick:
   apply weather
   apply scene-sync ground actors
   update truth-driven pedestrians
   update runtime UAVs
   trigger all cameras once for that tick
   write sidecars and images
6. Run `Demos/multiview/build_multiview_demo_assets.py` for GIF/timeline outputs.

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
- runtime UAV RPCs such as `create_runtime_multirotor`, `move_runtime_multirotor`, `get_runtime_multirotor_status`
- `apply_weather`
- `capture_world_camera`

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
