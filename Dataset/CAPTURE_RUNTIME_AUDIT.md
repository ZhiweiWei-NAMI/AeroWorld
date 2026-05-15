# AeroWorld Capture Runtime Audit

This document records the canonical capture runtime contract for the low-altitude semantic event-chain program. The active handoff is `../HANDOFF_LOW_ALTITUDE_SEMANTIC_EVENT_CHAIN.md`.

## Runtime Contract

- Reuse the already running UE PIE session.
- Reuse AirSim RPC at `127.0.0.1:41451`.
- Do not close UE/PIE unless C++ rebuild is required or the user explicitly requests it.
- Capture runs keep UE open by default.
- Formal images use UE editor-hook fixed-world capture, including UAV RGB/depth/seg and high overview.
- Formal UAV host runs are single UAV and single modality.
- Formal tick range is `0..900` with capture interval `5`.
- UE memory guard is 18GB; cleanup runs after each episode or host chunk.
- Failed runs keep UE open for inspection and deterministic resume.
- Output roots are deterministic only. No timestamped or versioned capture directories.
- AirSim settings are stored under Huawei Share.

## Capture Model

- One episode is one deterministic event chain with one key semantic event and continuous interaction.
- Background vehicles, pedestrians, UAVs, and logical actors are semantic actors, not decoration.
- Every scene entity must physically move when its role requires motion.
- Scene evidence is validated by truth frames, event trace, dynamic labels, and rendered outputs.

## Canonical Pipeline

`spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`

## Modalities

- `rgb`
- `depth`
- `seg`

## Deterministic Output Roots

- Render-ready episodes: `Dataset/render_ready_episodes/<episode>__seed00`
- Formal capture root: `F:/aw_cap`
- Formal capture summary: `F:/aw_cap_summary.csv`

## Invariants

- No fallback.
- No guessing.
- No compatibility path.
- No alternate capture pipeline.
- No hidden timestamp or version directory naming.
