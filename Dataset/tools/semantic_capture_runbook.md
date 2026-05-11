# Semantic Capture Long-Run Runbook

This file is the durable operating contract for semantic dataset capture. Read it before running long jobs.

## Goal

The goal is deterministic UAV dataset sampling under one event episode: each selected UAV/view must produce `rgb`, `depth`, and `seg` for the same event, truth-frame tick, capture entity, and capture view. The three modalities are separate physical host processes, but they form one logical sample identified by `logical_sample_id = <episode>:tick<tick>:<capture_view_id>`.

## Must-Follow Rules

- Reuse the already running UE PIE session and AirSim RPC at `127.0.0.1:41451`.
- Do not close UE/PIE unless C++ must be rebuilt or the user explicitly asks.
- Keep UE/PIE open on failures so logs, memory state, and scene state can be inspected.
- Use deterministic output roots. Do not create timestamp or versioned output directories for normal runs.
- Do not jump directly to a high tick for UAV capture. Simulate from tick 0 to the capture tick.
- High-overview semantic capture does not require `--airsim-capture-entity`.
- UAV host runs must pass exactly one `--airsim-capture-entity`.
- Each UAV host run may replace at most one scene UAV with `CaptureUAV_0`.
- Formal `pose_sync` runs must not create non-capture UAVs through editor-hook `create_runtime_multirotor_json`. Non-capture UAVs use the legacy AirSim/Aero runtime multirotor RPC path; RPC failure is a nonfatal visibility/sync warning and must not spawn `BP_AW_UAV_Inspection_Quad_01_C`.
- Each image host process captures exactly one modality. RGB, depth, and seg for the same UAV view are three sequential host processes with the same tick, capture entity, capture view id, and deterministic pose.
- If an episode contains multiple runtime UAVs, the outer runner must either capture only one UAV or run separate host passes, one per UAV.
- `CaptureUAV_0` is the only AirSim RGB/depth capture platform.
- Formal segmentation uses UE CustomStencil only. AirSim native segmentation is not a formal dataset output.
- Every seg output must include a uint8 class-id PNG, sidecar JSON, palette preview PNG, and class histogram.
- UAV RGB/depth/seg for one logical pass must share the same `capture_alignment_key` after the three single-modality host processes finish.
- UAV RGB/depth/seg for one logical pass must share the same `logical_sample_id`, `episode_id`, tick, `capture_view_id`, and `source_uav_entity_id`.
- Different UAVs or different views in the same event must use different `capture_view_id` values while keeping the same event episode and capture tick.
- Sidecars and manifests must include coordinate audit records for captured/spawned semantic entities.
- The default memory guard is 20 GB private memory. If it trips, stop the runner and keep UE open for deterministic resume.

## Coordinate Spaces

- `map_enu_m`: episode truth frames, event script positions, and global roster positions. L1-3 `nfz_l1_3_v1` at `[7089.5, 6238.0, 28.0]m` should resolve to about `[708950, 623800, 2800]cm` in PIE before world-origin offset.
- `local_enu_m`: only transformed when explicitly marked local.
- `map_static_local_enu_m`: static map fixtures in `Config/LowAltitude/Maps/.../scenario_objects.json`. The no-fly fixture at `[14, 16, 8]m` resolving to `[1400, 1600, 800]cm` is expected and is not the L1-3 event NFZ.
- `ue_world_cm`: observed UE actor/component location used for audit only.

Event-level semantic objects such as `nfz_l1_3_v1`, `pad_home_intruder_l1_3_v1`, and `pad_home_uav_l1_3_v1_primary` come from the episode `SCENE_SETUP` and are spawned as visible semantic proxies in `map_enu_m` coordinates. Static map boxes keep their existing static fixture meaning.

## Default Formal Run

- Runner: `Dataset/tools/run_semantic_event_chain_every10.py`
- Machine contract: `Config/LowAltitude/semantic_capture_runtime_contract.json`
- Output root: `Saved/AirSim/semantic_70events_rgb_depth_seg_tick100`
- Summary: `Saved/AirSim/semantic_70events_rgb_depth_seg_tick100_summary.csv`
- UAV policy: `one_uav_per_episode`
- UAV control backend: `pose_sync`
- Modalities: `rgb`, `depth`, `seg`
- Host process modality: one of `rgb`, `depth`, or `seg`; never multiple modalities in one host process.

## UAV Policies

- `one_uav_per_episode`: capture the first active runtime UAV per episode and write skip rows for the remaining UAVs.
- `all_uavs_by_separate_runs`: run one host pass per active runtime UAV. Each pass still replaces only one UAV.

## Failure Handling

- `fatal_runtime_unavailable`: stop immediately. Usually UE, RPC, EditorRemote, or memory guard.
- `failed`: the pass failed but the runner may continue to the next pass.
- `skipped_no_runtime_uav`: episode has no active runtime UAV at capture tick.
- `skipped_by_one_uav_policy`: a UAV was intentionally not captured because only one UAV per episode was requested.
- `skipped_by_explicit_capture_entity`: the requested capture entity is not active in the current episode.

## Resume

Use the same output root and summary with `--append-summary` and `--start-index`. Do not create a new timestamped directory.
