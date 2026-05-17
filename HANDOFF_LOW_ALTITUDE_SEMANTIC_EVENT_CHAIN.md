# Low Altitude Semantic Event Chain Handoff

Authoritative runtime note for the AeroWorld low-altitude semantic capture set. Read this before generation, validation, or capture.

## Core Contract

- Formal set: 70 scenarios x 3 seeds = 210 capture episodes.
- Formal capture ticks: `0..900`, every `5` ticks.
- Formal input root: `Dataset/render_ready_episodes_capture_filtered`.
- Formal output root: `F:\aw_cap`.
- Formal summary: `F:\aw_cap_summary.csv`.
- Python: `E:\conda\envs\aeroagentsim\python.exe`.
- Shell search: use PowerShell native commands, not `rg`.
- UE memory guard: 18GB working set/private memory.

## Capture Rules

- Reuse existing UE PIE and AirSim RPC `127.0.0.1:41451` whenever they exist.
- Do not close UE/PIE for normal episode, chunk, view, modality, or memory-guard failures.
- If UE/PIE is absent because the editor exited, the formal start script may start UE and enter PIE once.
- Long-run failures keep UE open for inspection.
- Formal images use UE editor-hook fixed-world capture, not AirSim native camera capture.
- AirSim is only the RPC/compatibility bridge.
- UAV capture is one scene UAV and one modality per host run.
- Each episode is complete only after high overview plus every active ROI UAV view has all required modalities.
- UAV editor-hook capture must reuse `fixed_world_camera.uav.shared_capture`.

## Debug Checklist

- Coordinates: use shared map/coordinate services and truth-frame contracts; do not add one-off math.
- SUMO vehicles: positions are selected from SUMO truth and lane metadata; render-side overlap or lane-offset filters must stay disabled in formal config.
- Pedestrians: source trajectories must remain semantically valid, continuous, and visible for their declared `ground_flow_contract.route_duration_ticks`.
- UAVs: pad selection drives task/order routing; pad changes require UAV truth regeneration.
- Motion: dynamic pedestrian/vehicle/UAV records must be continuous in time, with yaw aligned to final rendered movement direction.
- Semantics: event intent order, causal predecessors, target roles, facilities, corridors, pads, and logical regions must match the scenario contract.
- Visibility: `Dataset/render_ready_episodes_capture_filtered` is the formal capture root only; it must pass full render-ready truth through without deleting dynamic P/V/U. Visibility mistakes must fail validators and be fixed in generation/conversion.
- If an actor vanishes in capture, compare the capture sidecar with render-ready truth. If they match, fix generation/conversion truth, not UE.
- High overview should cover only the theoretical intersection capture boundary, using bbox + FOV geometry. It must not use global trajectory extents that reveal unrendered outside areas.

## Start Script

Use this script for formal capture:

```powershell
cd E:\DynamicCityCreatorSamples
.\Dataset\tools\start_formal_capture.ps1
```

Behavior:

- If formal capture is already running, it exits without starting a duplicate.
- If UE/PIE/RPC is ready, it reuses the current session.
- If UE is not running, it starts UE once with Huawei Share AirSim settings and enters PIE once.
- If UE is open but PIE/RPC is not ready, it sends one PIE hotkey and verifies RPC.
- If the supervisor fails, it leaves UE/PIE open and exits nonzero.
- It never kills or restarts an existing UE process.

## Main Entry Points

- `Dataset/tools/start_formal_capture.ps1`: safe formal launch wrapper.
- `Dataset/tools/supervise_event_chain_capture.py`: long-run formal supervisor.
- `Dataset/tools/run_semantic_event_chain_every10.py`: formal per-episode/per-view runner; the filename is historical.
- `Plugins/SumoImporter/Scripts/episode_render_host.py`: UE playback/capture host.
- `Plugins/SumoImporter/Scripts/editor_hook_client.py`: UE editor-hook fixed-world capture client.
- `Plugins/SumoImporter/Scripts/episode_capture_presets.json`: camera and modality presets.
- `Config/LowAltitude/semantic_capture_runtime_contract.json`: machine-readable runtime guardrail.

## Validation

Run validators before formal capture after any truth-generation change:

```powershell
cd E:\DynamicCityCreatorSamples
& E:\conda\envs\aeroagentsim\python.exe .\Dataset\tools\validate_render_ready_truth_light.py --input-root .\Dataset\render_ready_episodes --truth-check-mode manifest --workers 4
& E:\conda\envs\aeroagentsim\python.exe .\Dataset\tools\validate_capture_filtered_truth.py --input-root .\Dataset\render_ready_episodes_capture_filtered --source-root .\Dataset\render_ready_episodes
& E:\conda\envs\aeroagentsim\python.exe .\Dataset\tools\audit_event_truth_alignment.py --input-root .\Dataset\render_ready_episodes_capture_filtered
```

## Do Not Use

- Do not use AirSim native image capture for formal RGB/depth/seg.
- Do not write formal capture under `Saved/AirSim`.
- Do not create timestamp/version output roots.
- Do not use `Dataset/render_ready_episodes` directly for UE capture.
- Do not use external auto-restart wrappers that kill/restart UE on memory guard failures.
