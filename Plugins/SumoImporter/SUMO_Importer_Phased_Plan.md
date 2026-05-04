# SUMO Importer Phased Plan (MVP First)

## 1) Intent Lock
- Primary goal: implement the shortest usable chain in UE Editor.
- Chain: `net.xml -> parse -> SUMO->UE transform -> spawn lane splines/junction debug -> lane query API`.
- Current non-goals: `UFactory`, `Interchange`, mesh beautification, terrain fitting, OSM/SUMO dual-source alignment.

## 2) Decision-Complete Defaults
- Plugin topology: `Runtime + Editor` modules.
- Editor entry: one toolbar button + file dialog.
- Import scope: `function != internal` edges only.
- Re-import behavior: replace previous fixed network actor.
- Coordinate default: `SUMO X->UE Y`, `SUMO Y->UE X`, `m->cm`.
- Query API surface: both C++ and Blueprint.
- Sampling policy: `s` out-of-range uses clamp + warning semantics.
- Query space: return world transform.

## 3) Implementation Phases

### Phase A - Skeleton
- Create plugin descriptor and module skeleton.
- Add build dependencies for XML parsing, editor menus, and file dialog.
- Register plugin in project descriptor.
- Exit criteria:
  - Plugin can be discovered by UE.
  - Runtime and editor modules are loadable.

### Phase B - Semantic Parser + Transform
- Implement SUMO data types for location/edge/lane/junction/connection.
- Implement XML parser (`FXmlFile`) and shape parser (`x,y` and `x,y,z`).
- Implement import filtering (`internal` excluded by default).
- Implement deterministic coordinate transformer with configurable mapping.
- Exit criteria:
  - Parser can load a `.net.xml` into typed data.
  - Deterministic transform is reusable and parameterized.

### Phase C - Scene Builder + Query
- Implement `ASumoRoadNetworkActor` as the runtime container.
- Spawn one spline component per lane.
- Spawn junction debug splines from junction polygons.
- Build query indexes:
  - `laneId -> lane handle`
  - `edgeId -> lane handles`
  - `edge+lane -> successor connections`
- Implement API:
  - `FindLaneById`
  - `FindLanesByEdge`
  - `SampleTransformByEdgeLane`
  - `GetSuccessors`
- Exit criteria:
  - Lanes and junction debug geometry are generated.
  - Query APIs return stable results for downstream systems.

### Phase D - Editor Workflow
- Add LevelEditor toolbar button `Import SUMO Net`.
- Open file dialog and choose `net.xml`.
- Execute parse + build pipeline in current editor world.
- Output import summary and errors via log/dialog.
- Exit criteria:
  - Non-programmer can click button and import successfully.
  - Repeat imports do not accumulate duplicated networks.

### Phase E - Sanity Validation
- Validate shell/runtime assumptions.
- Attempt to locate local UE build tools for compile verification.
- If build tool unavailable, run strict logic/static review and record limits.
- Exit criteria:
  - Environment facts captured.
  - Remaining validation work clearly handed off.

## 4) Acceptance Checklist
- Functional:
  - `net.xml` can be imported from editor button.
  - Lane splines generated from lane shapes.
  - Junction debug geometry generated.
  - `edge/lane/s -> world transform` works.
  - Connection successors query works.
- Engineering:
  - Parser/transform/builder are separated.
  - Defaults are deterministic and centralized.
  - Re-import path is predictable.
  - Logs include import summary counts.

## 5) Execution Review Log (Updated per Stage)

### Review A - Skeleton Completed
- Status: done.
- Implemented:
  - `SumoImporter.uplugin`
  - `Source/SumoImporter/*` module skeleton
  - `Source/SumoImporterEditor/*` module skeleton
  - Project enablement entry in `DynamicCityCreatorEx.uproject`
- Notes:
  - Dual-module structure is in place for runtime/editor separation.

### Review B - Parser + Transform Completed
- Status: done.
- Implemented:
  - `FSumoParseOptions`, `FSumoNetData`, lane/junction/connection structs.
  - `FSumoNetParser` with edge/lane/junction/connection extraction.
  - Lane shape parser and warning aggregation.
  - `FSumoCoordinateTransformer` with axis mapping + scale + yaw + translation.
- Notes:
  - Import defaults currently exclude `internal` edges.

### Review C - Builder + Query Completed
- Status: done.
- Implemented:
  - `ASumoRoadNetworkActor` network container and spline generation.
  - Junction debug polygon splines.
  - Runtime indexes and query APIs.
  - Successor mapping from SUMO connections.
- Notes:
  - Re-import-safe actor replacement uses actor tags.

### Review D - Editor Entry Completed
- Status: done.
- Implemented:
  - Toolbar command registration.
  - File dialog selection.
  - Parse/build orchestration in editor world.
  - Import summary dialog + log output.
- Notes:
  - Workflow now matches `click button -> import net.xml`.

### Review E - Environment/Validation Check Completed
- Status: done with constraints.
- Verified facts:
  - Shell: PowerShell `5.1.22621.2506`.
  - Active code page: `936`.
  - No local `Engine/Build/BatchFiles/Build.bat` under project tree.
  - `UnrealBuildTool` command is not available in PATH.
- Result:
  - Full local compile was not executable in this environment.
  - Per request, implementation proceeded and was logically cross-checked.

### Review F - Compile Fixes (UE 5.2)
- Status: done.
- Implemented:
  - Fixed UE5 `FVector` double parsing mismatch in junction coordinate parsing.
  - Replaced unsupported `UToolMenus::IsAvailable()` with `UToolMenus::IsToolMenuUIEnabled()` guard.
- Notes:
  - Changes are API-compatible with the current `E:\UE_5.2` toolchain.

### Review G - Editor Button Visibility
- Status: done.
- Implemented:
  - Registered `Import SUMO Net` in both toolbar and `Tools` main menu.
  - Removed startup conditional guard so menu registration callback always binds.
  - Added explicit startup log `SumoImporterEditor menus registered.` for load verification.
- Notes:
  - If toolbar entry is collapsed/hidden, `Tools -> Import SUMO Net` remains available.

### Review H - Editable Transform for Imported Network
- Status: done.
- Implemented:
  - Set `ASumoRoadNetworkActor` root component mobility to `Movable`.
  - Set generated lane/junction spline components mobility to `Movable`.
- Notes:
  - Imported network actor can now be manually rotated/moved in editor transform tools.

### Review I - CityGenerator Road Alignment Path
- Status: done.
- Implemented:
  - Added `CityRoadGeoJsonParser` to read CityGenerator cache `road.geojson`.
  - Added WebMercator (`EPSG:3857`) conversion from GeoJSON lon/lat.
  - Added optional center extraction from sibling `bounds.geojson` (`properties.bbox`).
  - Extended import dialog to support both `*.net.xml` and `*.geojson`.
  - For GeoJSON import, switched default axis mapping to `XY_To_XY`.
- Notes:
  - This path generates splines from the same road source category CityGenerator uses, improving visual alignment debugging.

## 6) Remaining Test Steps for You
- Open UE project and ensure plugin `SumoImporter` is enabled.
- Click toolbar button `Import SUMO Net`.
- Select `Plugins/SumoImporter/map.net.xml` (or your target net file).
- Validate:
  - One network actor is generated/replaced.
  - Lane spline count and junction debug visibility.
  - Query API results in Blueprint/C++ test calls.
