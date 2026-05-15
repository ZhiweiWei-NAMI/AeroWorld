# Low-Altitude Semantic Event-Chain Dataset

This dataset is the canonical low-altitude AeroWorld program for 70 deterministic episode definitions. Read `../HANDOFF_LOW_ALTITUDE_SEMANTIC_EVENT_CHAIN.md` before running generation or capture.

## Canonical Source

- Contract source of truth: `Dataset/tools/semantic_event_contract.py`
- Contract schema: `low_altitude_event_chain_contract_v1`
- Scenario generation contract: `semantic_event_contract_v1`
- Episodes: 70 total
- Base scenarios: 64
- Cross-layer chains: 6

## Layer Counts

| Layer | Count |
|-------|-------|
| L1 Airspace | 7 |
| L2 Infrastructure | 9 |
| L3 Dynamic Constraints | 5 |
| L4 Agents | 24 |
| L5 Environment | 9 |
| L6 Digital Layer | 10 |
| X Cross-Layer | 6 |

## Contract Rules

- Every episode has one key semantic event chain and continuous interaction.
- Background vehicles and pedestrians are semantic actors, not decoration.
- Background V/P semantics are contract-driven and exact per episode.
- `U_inspect` is long-lived: full episode presence, 80 m minimum path, orbit/racetrack motion, no static hover.
- L1 inspect altitude code is `I28`.
- L2 inspect altitude code is `I18`.
- No fallback, guessing, or compatibility paths are allowed.
- Static infrastructure and logical anchors are intentionally grounded and semantically meaningful; they are not filler.

## Canonical Pipeline

`spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`

## Runtime Rules

- Reuse the existing UE PIE session and AirSim RPC `127.0.0.1:41451`.
- Do not close UE/PIE unless C++ rebuild is required or the user explicitly requests it.
- Capture scripts keep UE open by default.
- Formal image capture uses the UE editor-hook fixed-world camera, not AirSim native camera capture.
- Formal episode span is tick `0..900`, captured every `5` ticks.
- Memory guard is 18GB; clear world state, temporary capture actors, and PIE garbage after each episode or host chunk.
- Deterministic output roots only. No timestamp or version directories.
- Failed runs keep UE open for inspection and resume.
- AirSim settings live under Huawei Share.

## Main Output Roots

- Source scenarios: `Dataset/scenarios/...`
- Deterministic episode roots: `Dataset/episodes/<scenario>__seed00`
- Render-ready episode roots: `Dataset/render_ready_episodes/<scenario>__seed00`
- Formal UE capture root: `F:/aw_cap`
- Formal UE capture summary: `F:/aw_cap_summary.csv`

## Canonical Artifacts

- `scene_setup.json`
- `event_script.json`
- `spec.py`
- `episode_manifest.json`
- `scenario_plan.json`
- `global_entity_roster.json`
- `truth_frames.jsonl`
- `event_trace.jsonl`
- `dynamic_labels.jsonl`
- `trajectories.jsonl`
- `weather_meta.jsonl`

Formal UE capture input is `Dataset/render_ready_episodes_capture_filtered/<episode>/render_host_config.json`, not the unfiltered render-ready root.
