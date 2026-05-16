# Low Altitude Semantic Event Chain Handoff

Read this file first. It is the current project handoff and overrides older local notes unless code or the user explicitly says otherwise.

## Agent Gate

Any agent that reads this file must first report in the chat before running capture, generation, or cleanup. The report must:

1. Repeat the core runtime rules in the "Core Rules" section.
2. State the intended project entrypoint and the modules it will inspect.
3. Confirm it will use `E:\conda\envs\aeroagentsim\python.exe` and PowerShell native commands, not `rg`.
4. Confirm it will not close UE/PIE unless C++ rebuild or the user explicitly requests it.
5. Wait for the user's approval before continuing execution.

## Current Markdown Audit

The old handover file and several README files were stale. The stale items were: `Saved/AirSim/...` output roots, AirSim-native UAV image capture examples, one-channel wording that did not match the runner, 20GB memory guard defaults, and old wording that made AirSim sound like the image-capture backend. Current docs must point here instead of repeating long runtime recipes.

## Documentation Policy

Only four Markdown documents are intentionally kept:

- `AGENTS.md`: short mandatory rules for agents entering this workspace.
- `HANDOFF_LOW_ALTITUDE_SEMANTIC_EVENT_CHAIN.md`: authoritative project handoff and runtime contract for humans/agents.
- `PROJECT_STRUCTURE.md`: compact repository map.
- `Dataset/README.md`: compact dataset entrypoint.

All scenario README files, plugin README files, historical audits, phased plans, runbooks, generated API notes, and one-off reading guides were removed. For episode details, use `scene_setup.json`, `event_script.json`, `scenario_plan.json`, `episode_manifest.json`, `truth_frames.jsonl`, and validators instead of Markdown summaries.

## Core Rules

1. Reuse the existing UE PIE session and AirSim RPC `127.0.0.1:41451`. Do not close UE/PIE for normal episode or chunk transitions.
2. Closing UE/PIE is allowed only for C++ rebuilds or when the user explicitly asks. Long-run failure should keep the editor open for inspection.
3. Memory guard is 18GB for UE working set and private memory. After each episode or host chunk, clear world state, remove temporary capture actors, collect PIE garbage, and let the Python child exit. Clearing does not mean closing UE.
4. Every formal episode is 900 ticks, starting at tick `0` and ending at tick `900`. Formal capture interval is every `5` ticks.
5. Coordinate conversion must use the shared coordinate services and truth-frame contract. Do not create one-off coordinate math in scripts.
6. Every code/documentation change ends with a commit that states the behavioral difference, followed by an automatic review of the diff.
7. Formal capture is single UAV and single channel per host run. Channels are `rgb`, `depth`, and `seg`; an episode is complete only after all active UAVs that enter the ROI have their viewpoints captured.
8. Each region/episode must include a high overview capture for global visual inspection.
9. Image capture uses UE editor-hook fixed-world capture, not AirSim native camera capture. AirSim remains an RPC bridge and compatibility surface, not the formal image backend.
10. UAV editor-hook capture reuses one shared fixed-world capture camera, currently `fixed_world_camera.uav.shared_capture`. Do not create one fixed camera per UAV view.
11. Asset coordinate references come from the SUMO traffic bundle and GeoJSON sources, through the same coordinate interface. Landing pads and other facilities need offsets like pedestrians; vehicles need lane-derived offsets from lane information.
12. Truth-frame generation is decoupled from UE rendering. Generated truth must pass validators before capture. Each episode truth contains only vehicles, pedestrians, and UAVs relevant to the ROI, plus required semantic facilities/logical context.
13. Every episode must have a purposeful intent, background context, and a special event chain. The episode root metadata must include the event description, time, location, and event-chain steps.
14. Capture outputs live on `F:`. Formal root is `F:\aw_cap`; summary is `F:\aw_cap_summary.csv`. Do not write formal capture under `Saved/AirSim`.
15. Unless the event explicitly requires a collision, vehicles, pedestrians, and UAVs should avoid collisions. Event collisions must be authored as event actors, not accidental background overlaps.
16. Slow is acceptable. Cleanup, object generation, scene sync, and every-5-tick capture may use conservative chunk sizes and delays to avoid memory pressure.
17. Prefer user-opened PIE. If automation must enter PIE, first address asset-rendering/visualization risk, use the UE window handle to click PIE when needed, and continue only after screenshot or visual confirmation.
18. Keep output roots deterministic. No version or timestamp directories. Retry failed episodes with the same root and `--overwrite` or explicit cleanup.
19. AirSim settings live under Huawei Share.
20. Do not use `rg` in this environment; use PowerShell native search commands.

## Main Entry Points

- `Dataset/tools/supervise_event_chain_capture.py`
  Long-run supervisor for the formal 210 episode capture set. It keeps output on `F:\aw_cap`, reuses PIE when available, runs high overview plus all active UAV views, and falls back to slower profiles under memory pressure.

- `Dataset/tools/run_semantic_event_chain_every10.py`
  Formal event-chain runner. The filename is historical; current defaults are tick `0..900` with `--tick-step 5`, `--simulation-tick-stride 5`, 18GB guards, `--uav-capture-backend editor_hook`, and one channel per UAV host run.

- `Plugins/SumoImporter/Scripts/episode_render_host.py`
  UE playback host. It consumes one render-ready episode config, applies truth-frame scene state, runs event actions, invokes UE hook capture, writes images and sidecars, then resets world state and collects PIE garbage.

- `Plugins/SumoImporter/Scripts/editor_hook_client.py`
  UE Python remote-execution client and fixed-world capture hook. This is the correct image-capture path for formal RGB/depth/seg outputs.

- `Config/LowAltitude/semantic_capture_runtime_contract.json`
  Runtime contract defaults and invariant flags. This is the machine-readable guardrail for memory, output root, modality, and backend policy.

- `Plugins/SumoImporter/Scripts/episode_capture_presets.json`
  Camera and modality presets. UAV presets must stay `editor_hook` and should not drift back to `airsim_native`.

## Generation And Validation Pipeline

Canonical data flow:

```text
semantic_event_contract.py
-> regenerate_boundary_scenarios.py
-> batch_generate.py
-> convert_to_render_ready.py  (always writes Dataset/render_ready_episodes_capture_filtered)
-> run_semantic_event_chain_every10.py
-> episode_render_host.py
-> validators
```

Important validators and auditors:

- `Dataset/tools/validate_render_ready_truth_light.py`
- `Dataset/tools/audit_event_truth_alignment.py`
- `Dataset/tools/verify_semantic_visibility_contract.py`
- `Dataset/tools/sumo_ground_flow/validate_truth_integration.py`
- `Dataset/tools/sumo_ground_flow/validate_traffic_output.py`
- `Dataset/tools/uav_global_flow/validate_uav_flow.py`

## Coordinate And Map Ownership

- SUMO and GeoJSON sources are owned by `Plugins/SumoImporter/Maps/<map_id>/source/` and discovered through `donghu_core/discovery.py`.
- Runtime map config remains in `Config/LowAltitude/Maps/<map_id>/`.
- SUMO coordinate mapping is in `Dataset/tools/sumo_ground_flow/coordinates.py`.
- Shared Python service boundaries are under `Plugins/SumoImporter/Scripts/donghu_core/`.
- UE host placement and lane/ground snapping logic is in `Plugins/SumoImporter/Scripts/episode_render_host.py`.
- Vehicle offsets should come from lane metadata and lane samples, not hard-coded visual nudges.
- Facility offsets should follow the same policy as pedestrian/facility placement, not raw anchor placement when the asset visual origin needs correction.

## Formal Capture Command

Use the supervisor after the user approves:

```powershell
cd E:\DynamicCityCreatorSamples
& E:\conda\envs\aeroagentsim\python.exe .\Dataset\tools\supervise_event_chain_capture.py
```

For a read-only plan without touching UE:

```powershell
cd E:\DynamicCityCreatorSamples
& E:\conda\envs\aeroagentsim\python.exe .\Dataset\tools\run_semantic_event_chain_every10.py --plan-only --limit 1
```

For a single formal episode after approval:

```powershell
cd E:\DynamicCityCreatorSamples
& E:\conda\envs\aeroagentsim\python.exe .\Dataset\tools\run_semantic_event_chain_every10.py --start-index 0 --limit 1 --append-summary --resume-completed-ok
```

## Do Not Use

- Do not use `Saved/AirSim/semantic_70events_rgb_depth_seg_tick100` as formal output.
- Do not call UAV capture with `--uav-capture-backend airsim_native` for formal data.
- Do not use `Dataset/render_ready_episodes` directly as UE capture input. Use `Dataset/render_ready_episodes_capture_filtered`.
- Do not resurrect `Plugins/SumoImporter/Scripts/editor_start_pie.py`; manual PIE is preferred and automated PIE must be visually verified.
- Do not create timestamped/versioned output folders.
- Do not duplicate coordinate conversion code.

## Open Risks To Watch

- The runner name still says `every10`; treat this as historical naming only.
- UE automation can produce material/rendering surprises. Prefer user-opened PIE or verify screenshots before capture.
- Scenario README files were removed to prevent stale branch guidance. For formal episode execution, trust `scenario_plan.json`, `episode_manifest.json`, `event_script.json`, `truth_frames.jsonl`, and the validators.
