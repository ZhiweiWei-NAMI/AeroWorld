# Low-Altitude Semantic Event-Chain Dataset

Canonical dataset root for the AeroWorld low-altitude semantic event-chain capture set. Read `../HANDOFF_LOW_ALTITUDE_SEMANTIC_EVENT_CHAIN.md` before generation, validation, or capture.

## Scope

- Formal set: 70 scenarios x 3 seeds = 210 episodes.
- Formal ticks: `0..900`, sampled every `5` ticks.
- Formal capture input: `Dataset/render_ready_episodes_capture_filtered`.
- Formal capture output: `F:\aw_cap`.
- Formal summary: `F:\aw_cap_summary.csv`.
- Python: `E:\conda\envs\aeroagentsim\python.exe`.

## Main Pipeline

`spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`

`convert_to_render_ready.py` writes both `Dataset/render_ready_episodes` and the formal filtered root `Dataset/render_ready_episodes_capture_filtered`. UE capture must consume the filtered root.

## Runtime Rules

- Reuse existing UE PIE and AirSim RPC `127.0.0.1:41451`.
- Do not close UE/PIE for normal episode, chunk, view, modality, or memory-guard failures.
- Start UE/PIE only when no PIE session exists, using `Dataset/tools/start_formal_capture.ps1`.
- Formal image capture uses the UE editor-hook fixed-world camera, not AirSim native camera capture.
- Formal capture is one UAV view and one modality per host run; each episode is complete only after high overview plus every active ROI UAV view has all required modalities.
- High overview covers only the theoretical intersection capture boundary, computed from boundary bbox and FoV. It must not use global trajectory extents.

## Debug Priorities

- Coordinate conversion must use shared map/coordinate services and truth-frame contracts.
- SUMO vehicles should come from SUMO truth and lane metadata; formal render config must not rely on overlay, projection, or lane-offset filters to repair truth.
- Pedestrians and vehicles must be continuous in time and yaw-aligned to rendered movement direction.
- Required ground-flow pedestrians stay visible for their declared `ground_flow_contract.route_duration_ticks`.
- UAV pads drive order and route generation; pad changes require regenerated UAV truth.
- Event semantics must preserve intent order, causal predecessors, target roles, facilities, corridors, pads, and logical regions.

## Start Script

```powershell
cd E:\DynamicCityCreatorSamples
.\Dataset\tools\start_formal_capture.ps1
```

The script reuses ready PIE/RPC, refuses duplicate formal capture, starts UE only if UnrealEditor is absent, enters PIE only if UE is open but not in PIE, and leaves UE/PIE open on failures.

## Canonical Artifacts

- `scene_setup.json`
- `event_script.json`
- `episode_manifest.json`
- `scenario_plan.json`
- `global_entity_roster.json`
- `truth_frames.jsonl`
- `event_trace.jsonl`
- `dynamic_labels.jsonl`
- `trajectories.jsonl`
- `weather_meta.jsonl`
