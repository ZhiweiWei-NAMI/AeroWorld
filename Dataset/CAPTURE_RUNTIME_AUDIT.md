# AeroWorld Capture Runtime Audit

This document records the canonical capture runtime contract for the low-altitude semantic event-chain program.

## Runtime Contract

- Reuse the already running UE PIE session.
- Reuse AirSim RPC at `127.0.0.1:41451`.
- Do not close UE/PIE unless C++ rebuild is required or the user explicitly requests it.
- Capture runs keep UE open by default.
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
- Semantic capture root: `Saved/AirSim/semantic_70events_rgb_depth_seg_tick100`
- Semantic capture summary: `Saved/AirSim/semantic_70events_rgb_depth_seg_tick100_summary.csv`

## Invariants

- No fallback.
- No guessing.
- No compatibility path.
- No alternate capture pipeline.
- No hidden timestamp or version directory naming.

