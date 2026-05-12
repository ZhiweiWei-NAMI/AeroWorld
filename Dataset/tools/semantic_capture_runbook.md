# Semantic Capture Runbook

This is the durable operating contract for the canonical low-altitude semantic event-chain pipeline.

## Canonical Pipeline

`spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`

## Runtime Rules

- Reuse the existing UE PIE session.
- Reuse AirSim RPC `127.0.0.1:41451`.
- Do not close UE/PIE unless C++ rebuild is required or the user explicitly requests it.
- Capture scripts keep UE open by default.
- Keep output roots deterministic.
- Do not create timestamped or versioned directories.
- AirSim settings are stored under Huawei Share.

## Capture Contract

- One episode is one deterministic event chain.
- Every episode has one key semantic event and continuous interaction.
- Every scene entity is semantic and meaningful.
- Background vehicles and pedestrians are semantic actors, not decoration.
- `U_inspect` is long-lived, full-episode, and must move with orbit/racetrack behavior.
- No fallback, guessing, or compatibility path is allowed.

## Output Roots

- Source scenarios: `Dataset/scenarios/...`
- Deterministic episodes: `Dataset/episodes/<scenario>__seed00`
- Render-ready episodes: `Dataset/render_ready_episodes/<scenario>__seed00`
- Semantic capture root: `Saved/AirSim/semantic_70events_rgb_depth_seg_tick100`

## Validation

- The pipeline must preserve exact contract counts.
- The pipeline must preserve the required event per episode.
- The pipeline must preserve scene and event file boundaries.
- Validators run after render-ready generation and before capture sign-off.

