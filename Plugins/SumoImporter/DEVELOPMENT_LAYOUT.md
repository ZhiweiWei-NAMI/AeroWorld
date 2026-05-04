# Development Layout

This project now has a stricter ownership split.

## Key Plugins

- `Plugins/AeroSimHost`
  Runtime bridge, scene sync, asset placement, weather, and UE-facing APIs.
  If a change affects RPC contracts or UE-side world loading, it belongs here.

- `Plugins/PedestrianRuntime`
  UE-side pedestrian runtime behavior.
  If a change affects managed pedestrian spawning, animation, grounding, or runtime behavior, it belongs here.

- `Plugins/SumoImporter`
  External Python entrypoints, scenario generation, map-source materials, and offline tooling.
  If a change is driven from Python or needs to scale to many scenarios, it should usually start here.

- `Plugins/AeroWorldContent`
  UE assets and content that support scenarios.
  Reusable props, cameras, data assets, and other authored content belong here.

## External Python

- Long-lived CLI tools belong in `Plugins/SumoImporter/Scripts/`.
  Examples: `episode_render_host.py`, `build_multiview_demo_assets.py`, `play_multiview_demo_cli.py`.

- Shared Python services belong in `Plugins/SumoImporter/Scripts/donghu_core/`.
  Keep reusable logic here instead of re-copying it across scripts.

- Ad hoc debugging and one-off inspection scripts belong in `Plugins/SumoImporter/Scripts/dev_checks/`.
  Do not leave these in the repo root.

## Maps

- Plugin-owned map-source inputs belong in `Plugins/SumoImporter/Maps/<map_id>/source/`.
  This is where `map.net.xml`, `map.osm`, and temporary conversion outputs should live.

- UE runtime map config stays in `Config/LowAltitude/Maps/<map_id>/`.
  `AeroSimHost` currently resolves `map_context.json`, `scenario_objects.json`, and `ped_nav_semantic.bundle.json` from there, so these files should remain in `Config` until the UE-side loader is redesigned.

## Scenarios

- New scenario families should live in `Plugins/SumoImporter/Scenarios/<scenario_id>/`.

- Recommended per-scenario layout:
  `spec/`
  Scenario spec JSON and small authored metadata.
  `scripts/`
  Scenario-specific generators or conversion helpers.
  `artifacts/`
  Generated `ScenarioPackage` outputs such as `truth_frames.jsonl`, `weather_meta.jsonl`, `capture_plan.json`, and manifests.
  `notes/`
  Human-readable assumptions, contracts, and review notes.

- The current `donghu_dense_uav_rain_fall` directory under `Scenarios/` is the canonical reference scenario.
  New scenes should follow the same layout rather than creating new scenario roots under `Scripts/`.

## Scenario Materials

- Reusable UE assets such as flower pots, umbrellas, cones, warning signs, barricades, and other incident props belong in `Plugins/AeroWorldContent/Content/Props/ScenarioCommon/`.

- Scene-specific assets that only make sense for one incident family belong in `Plugins/AeroWorldContent/Content/Props/ScenarioSpecific/<scenario_id>/`.

- Pure data for those scenes does not go in `AeroWorldContent`.
  JSON specs, placement data, event chains, capture plans, and Python-side materials belong in `SumoImporter`.
