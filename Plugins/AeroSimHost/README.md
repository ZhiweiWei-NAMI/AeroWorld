# AeroSimHost

This plugin owns the UE runtime bridge and the authoritative UE-side API surface.

If Python code needs to drive UE, spawn/move assets, query grounding, control pedestrians, apply weather, or support fixed-world image capture, the contract eventually resolves here. Formal capture is invoked through the UE editor hook from `SumoImporter`.

## Read Order

1. `README.md`
   Responsibilities and boundaries.
2. `AERO_API_REFERENCE.md`
   Authoritative contract for Python callers.
3. `Source/AeroBridgeRuntime/Private/AeroBridgeWorldSubsystem.cpp`
   Main RPC implementation and runtime load flow.
4. `Source/AeroAssetPlacement/Private/AeroAssetPlacementSubsystem.cpp`
   Asset catalog and scenario object loading.
5. `Source/AeroPedNavSemantic/Private/AeroPedNavSemanticSubsystem.cpp`
   Pedestrian semantic navigation and ground projection support.
6. `Source/AeroWeatherRender/Private/AeroWeatherRenderSubsystem.cpp`
   Weather application path.

## Design Idea

- UE runtime config is currently loaded from `Config/LowAltitude`.
- Map-specific runtime files are loaded from `Config/LowAltitude/Maps/<map_id>/`.
- Python should not guess UE internals.
  It should call bridge APIs and trust the returned resolved pose.

## Runtime Load Logic

`simAeroLoadContext` loads:

- `Config/LowAltitude/asset_catalog.json`
- `Config/LowAltitude/weather_render_profiles.json`
- `Config/LowAltitude/Maps/<map_id>/map_context.json`
- `Config/LowAltitude/Maps/<map_id>/scenario_objects.json`
- `Config/LowAltitude/Maps/<map_id>/ped_nav_semantic.source.json`
- `Config/LowAltitude/Maps/<map_id>/ped_nav_semantic.bundle.json`

This is why those files currently stay in `Config`, even if source map materials are owned by `SumoImporter`.

## Key API Families

- context and config:
  `simAeroDescribeCapabilities`
  `simAeroLoadContext`
  `simAeroReloadConfig`
- world updates:
  `simAeroApplyFrame`
  `simAeroPollFeedback`
- pedestrian runtime:
  `simAeroPedSpawn`
  `simAeroPedReset`
  `simAeroPedObserve`
  `simAeroPedPlayAnimation`
  `simAeroPedStop`
  `simAeroPedSetVariant`
  `simAeroPedRelease`
- asset placement:
  `simAeroSpawnAsset`
  `simAeroMoveAsset`
  `simAeroRemoveAsset`
  `simAeroQueryNearest`
- geometry and grounding:
  `simAeroProjectGround`
  `simAeroQueryPedPath`
  `simAeroQueryPedAnchor`
- weather:
  `simAeroApplyWeather`
- capture:
  `simAeroCaptureWorldCamera`
  This is the UE fixed-world capture API surface. Formal UAV image collection should reach it through the editor hook, not AirSim native camera capture.

## Boundary

- If the change is about API behavior, runtime loading, or resolved poses, change `AeroSimHost`.
- If the change is about Python orchestration or scenario packaging, change `SumoImporter` first.
