# Semantic Capture Runbook

This is the durable operating contract for the canonical low-altitude semantic event-chain pipeline. The full handoff is `../../HANDOFF_LOW_ALTITUDE_SEMANTIC_EVENT_CHAIN.md`.

## Canonical Pipeline

`spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`

## Runtime Rules

- Reuse the existing UE PIE session.
- Reuse AirSim RPC `127.0.0.1:41451`.
- Do not close UE/PIE unless C++ rebuild is required or the user explicitly requests it.
- Capture scripts keep UE open by default.
- Use UE editor-hook fixed-world capture for formal images. Do not use AirSim native camera capture.
- Use tick `0..900` with capture interval `5`.
- Keep UE memory guards at 18GB and clear runtime state after each episode or host chunk.
- Keep output roots deterministic.
- Do not create timestamped or versioned directories.
- AirSim settings are stored under Huawei Share.

## Capture Contract

- One episode is one deterministic event chain.
- Every episode has one key semantic event and continuous interaction.
- Every scene entity is semantic and meaningful.
- Background vehicles and pedestrians are semantic actors, not decoration.
- `U_inspect` is long-lived, full-episode, and must move with orbit/racetrack behavior.
- Formal UAV capture runs one UAV and one modality per host run. Complete episodes must include all active UAV views plus high overview.
- No fallback, guessing, or compatibility path is allowed.

## Output Roots

- Source scenarios: `Dataset/scenarios/...`
- Deterministic episodes: `Dataset/episodes/<scenario>__seed00`
- Render-ready episodes: `Dataset/render_ready_episodes/<scenario>__seed00`
- Formal UE capture root: `F:/aw_cap`
- Formal UE capture summary: `F:/aw_cap_summary.csv`

## Validation

- The pipeline must preserve exact contract counts.
- The pipeline must preserve the required event per episode.
- The pipeline must preserve scene and event file boundaries.
- Validators run after render-ready generation and before capture sign-off.
