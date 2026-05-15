# UAM Dataset Coverage Report

## Canonical Program

- Total scenario definitions: 70
- Base scenarios: 64
- Cross-layer chains: 6
- Canonical contract source: `Dataset/tools/semantic_event_contract.py`
- Canonical generation pipeline: `spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`

## Coverage Summary

- L1 Airspace: 7
- L2 Infrastructure: 9
- L3 Dynamic Constraints: 5
- L4 Agents: 24
- L5 Environment: 9
- L6 Digital Layer: 10
- X Cross-Layer: 6

## Contract Invariants

- Every episode has a single required semantic event chain with continuous interaction.
- Background vehicles and pedestrians are semantic actors, not decoration.
- `U_inspect` is long-lived, full-episode, and motionful.
- L1 inspect altitude code is `I28`.
- L2 inspect altitude code is `I18`.
- No fallback, guessing, or compatibility path is allowed.
- No alternate pipeline is canonical.

## Runtime and Capture

- UE PIE session is reused.
- AirSim RPC `127.0.0.1:41451` is reused.
- Formal image capture uses UE editor-hook fixed-world capture, not AirSim native camera capture.
- Formal episode span is tick `0..900`, captured every `5` ticks.
- Formal output root is `F:/aw_cap`; summary is `F:/aw_cap_summary.csv`.
- Each episode requires high overview plus every active UAV view, captured as single UAV and single modality host runs.
- Output roots are deterministic only.
- AirSim settings live under Huawei Share.
- Capture validation must include truth frames, event trace, dynamic labels, and render-ready outputs.

## File Boundary

- `scene_setup.json` defines the grounded scene.
- `event_script.json` defines the event chain.
- `spec.py` is the source scenario spec.
- `render_ready_episodes/<scenario>__seed00` is the canonical deterministic episode output root.
- Formal UE capture input is `render_ready_episodes_capture_filtered/<scenario>__seedXX`.
