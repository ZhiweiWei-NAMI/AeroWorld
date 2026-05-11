# Handover: Semantic UAV Capture

Date: 2026-05-10
Workspace: `E:\DynamicCityCreatorSamples`

## Hard Rules

- Reuse the existing PIE session and AirSim RPC at `127.0.0.1:41451`.
- Do not close UE/PIE unless C++ must be rebuilt or the user explicitly asks.
- Do not use `rg`; this environment reports Access denied. Use PowerShell `Get-ChildItem`, `Select-String`, and `Get-Content`.
- Never mutate AirSim `APIPCamera`, `BP_PIPCamera`, `PIPCamera`, `SceneCaptureComponent`, `HiddenActors`, or `HiddenComponents` from Python/editor remote execution.
- Formal UAV segmentation must use UE fixed-world CustomStencil, not AirSim native segmentation.
- Primary UAV segmentation is for RGB-visible geometry. Logical no-fly/hazard/corridor regions stay in sidecar/meta JSON by default.
- Formal image output defaults must stay under `F:\`:
  - image root: `F:\aw_cap`
  - summary: `F:\aw_cap_summary.csv`
- Primary image paths and filenames must stay simple. Use paths like `F:\aw_cap\uav\e69\v000\rgb\tick_000000.png`; write long episode IDs, capture entity IDs, alignment keys, coordinates, audits, and contracts into sidecar/meta JSON files.

## Code-First Contract

- Every behavior change must be traced through the Python chain first.
- Preferred Python sequence: `Dataset/tools/spec_compiler.py` -> `Dataset/tools/regenerate_boundary_scenarios.py` -> `Dataset/tools/batch_generate.py` -> `Dataset/tools/convert_to_render_ready.py` -> `Dataset/tools/run_semantic_event_chain_every10.py` -> `Plugins/SumoImporter/Scripts/episode_render_host.py` -> validators.
- Do not reach for C++ unless the Python chain cannot express the change.
- Every layer must keep baseline background human, vehicle, and UAV context actors.
- Every event must be a captureable chain, not a static label.

## Crash Boundary

The previous crash was likely caused by Python/editor remote execution mutating AirSim PIPCamera hidden lists. The stack was:

```text
APIPCamera::updateInstanceSegmentationAnnotation()
ASimModeBase::updateInstanceSegmentationAnnotation()
ASimModeBase::AddNewActorToInstanceSegmentation()
RegisterActorWithAirSimInstanceSegmentation()
UAeroAssetPlacementSubsystem::SpawnActorForInstance()
```

The failing C++ path touched `captures_[Scene]->HiddenComponents` while an `APIPCamera` had `captures_.Num() == 0`. Do not revisit that Python hidden-list strategy.

## Current Coordinate Contract

- `traffic_bundle/lane_center_samples.csv` coordinates are map ENU meters.
- Scenario truth frames, event scripts, and global rosters are map ENU meters.
- `TrafficBundleRoadSnapper` and `vehicle_lane_offsets` remain in map ENU.
- `road.geojson` supplies road `lanes` and `width`; `lane_width_m=3.5` is only fallback/config.
- `simAeroApplyFrame` receives resolved map ENU unchanged.
- C++ converts map ENU meters to UE centimeters as `world_origin_cm + PositionEnuM * 100`.
- Do not apply the old inverse local transform before scene sync. The old wrong vehicle coordinate was around `[971.737, -142.526]`; the correct X6 vehicle coordinates are around `[7126.023, 6496.149]` and `[7115.287, 6510.563]`.

## Current Code State

- `Config\LowAltitude\semantic_stencil_rules.json`
  - Drone priority is now above pedestrian. This fixed X6 UAV names containing `crowd_evacuation` being misclassified as pedestrian.
- `Dataset\tools\run_semantic_event_chain_every10.py`
  - Default output root is `F:\aw_cap`.
  - Default summary is `F:\aw_cap_summary.csv`.
  - Formal runner rejects non-`F:` output/summary paths when the contract requires it.
  - Runner output dirs are short: `uav\eNN\vNN`, `hi\eNN`, `_meta\configs`.
  - Palette preview now colors class `12` (`uav_corridor`) instead of leaving it black.
- `Plugins\SumoImporter\Scripts\episode_render_host.py`
  - Recognizes simple per-view dirs like `v000`, so it does not append long capture view IDs.
  - Storage sidecars/manifests state that complex identifiers live in metadata.
  - Logical no-fly/hazard/corridor regions are processed into sidecar coordinate audit by default, but their thin Cube overlay actors are not spawned unless `event_semantic_overlays.logical_region_primary_segmentation_enabled=true`.
  - Sidecar audit keeps the original logical placement geometry in `logical_region_source_geometry`; `render_proxy_*` fields are only the optional thin overlay representation.
- `Plugins\SumoImporter\Scripts\episode_render_host_config.json`
  - `event_semantic_overlays.logical_region_primary_segmentation_enabled=false` is the default.
- `Config\LowAltitude\semantic_capture_runtime_contract.json`
  - Defaults and must-follow rules now lock the `F:` output/simple path contract and the sidecar-only logical-region contract.
- `Dataset\tools\verify_semantic_visibility_contract.py`
  - Required primary seg classes are RGB-visible classes only: background, drone, pedestrian, and vehicle.
  - The verifier reads the primary `seg.png`, cross-checks summary/sidecar histograms, and fails if sidecar-only logical classes `11`/`12` appear in the primary segmentation histogram.
- `Plugins\SumoImporter\Scripts\donghu_core\capture_orchestrator.py`
  - Frame stem is short: `tick_000000`.

## Latest Verified Smoke

Historical command before the sidecar-only logical-region contract:

```powershell
python Dataset\tools\run_semantic_event_chain_every10.py --episodes-root Dataset\render_ready_episodes --output-root F:\aw_cap --summary F:\aw_cap_summary.csv --start-index 69 --limit 1 --skip-high-overview --host 127.0.0.1 --port 41451 --airsim-capture-entity uav_observer_x6_crowd_evacuation_to_airspace_lockdown_3
```

Result:

```text
hist={"0":27,"1":686464,"11":20766,"12":30240,"2":182655,"5":1020,"6":195,"7":233}
```

Verifier:

```powershell
python Dataset\tools\verify_semantic_visibility_contract.py --mode verify-summary --summary Saved\AirSim\x6_smoke_drone_rule_fix_summary.csv
```

This smoke is now intentionally obsolete for primary segmentation: classes `11` and `12` belonged to invisible logical overlays, not RGB-visible objects.

## Visual Interpretation

- RGB now shows vehicles.
- The earlier pink irregular region was class `11` (`hazard_trigger`), produced by `nfz_x6_lockdown` as an invisible CustomDepth-only logical proxy, not by a building.
- Class `12` is `uav_corridor`. Like class `11`, it is a logical scene region and should not appear in the primary training `seg.png` while sidecar-only mode is active.
- The logical box/corridor details remain in sidecar/meta JSON: entity id, source logical asset, ENU position, proxy scale, policy, and coordinate audit.

## Data Layer Notes

Subagent read-only analysis found many episodes without full background context. Priority scenes for background context additions were:

```text
X1, X6, L4-8_v1/v2, L4-3_v1/v2/v3, L4-5_v1/v2/v3
```

The background-context generation path must continue to satisfy overlay/collision clearance rules before rendering, and the chain must stay captureable end to end.

## Checks To Rerun After Edits

```powershell
python -m json.tool Config\LowAltitude\semantic_stencil_rules.json
python -m json.tool Config\LowAltitude\semantic_capture_runtime_contract.json
python -m json.tool Plugins\SumoImporter\Scripts\episode_render_host_config.json
python -m py_compile Plugins\SumoImporter\Scripts\episode_render_host.py Dataset\tools\run_semantic_event_chain_every10.py Dataset\tools\test_semantic_visibility_contract.py Dataset\tools\verify_semantic_visibility_contract.py Dataset\tools\verify_semantic_uav_logical_samples.py
python -m unittest Dataset.tools.test_semantic_visibility_contract
```

For a new formal smoke, omit `--output-root` and `--summary` to use the required defaults:

```powershell
python Dataset\tools\run_semantic_event_chain_every10.py --episodes-root Dataset\render_ready_episodes --start-index 69 --limit 1 --skip-high-overview --host 127.0.0.1 --port 41451 --airsim-capture-entity uav_observer_x6_crowd_evacuation_to_airspace_lockdown_3
```

Expected default outputs are under `F:\aw_cap`, with summary at `F:\aw_cap_summary.csv`.
