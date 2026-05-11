# Low-Altitude Urban Air Mobility Event Dataset

## Overview

This dataset contains **64 base event scenarios plus 6 cross-layer chains** (70 scenario directories total) for low-altitude urban air mobility event-chain capture and code validation. The scenarios are organized across **33 event types** in a UAM-adapted PEGASUS 6-layer Operational Design Domain (ODD) model.

Each scenario directory contains:

- `spec.py`: concrete `ScenarioSpec` source for the SpecCompiler.
- `scene_setup.json`: grounded entity, asset, placement, camera, weather, and validation metadata.
- `event_script.json`: compiled event triggers, actions, and causal chains.

## Key Features

- **6-layer ODD coverage**: L1 airspace, L2 infrastructure, L3 dynamic constraints, L4 agents, L5 environment, and L6 digital layer.
- **Cross-layer chains**: 6 additional X scenarios linking weather, digital layer failures, airspace constraints, and agent responses.
- **Capture contract**: every episode carries a captureable event chain plus baseline background human, vehicle, and UAV context actors.
- **Grounded placement**: semantic anchors are resolved into ENU coordinates against the Donghu traffic bundle where applicable.
- **Executable validation**: `validate_scene_grounding.py` checks entity references, asset IDs, placement resolution, action coordinates, weather bootstraps, event-chain reachability, background context presence, and per-scenario validation rules.
- **Graph/RDF tooling**: existing tools export frame graphs, PyG data, and RDF triples from generated episodes.

## Capture Contract

- Every episode must expose a captureable event chain in `event_script.json`, `event_trace.jsonl`, `dynamic_labels.jsonl`, and `truth_frames.jsonl`.
- Every layer must include baseline background human, vehicle, and UAV context actors.
- The layer-specific actor remains the semantic target; background context is support data.
- Any code change should move through the Python chain first: `spec.py` -> `spec_compiler.py` -> `regenerate_boundary_scenarios.py` -> `batch_generate.py` -> `convert_to_render_ready.py` -> `batch_render_dataset.py` -> `run_semantic_event_chain_every10.py` -> `episode_render_host.py` -> validators.

## Directory Structure

```text
Dataset/
├── README.md
├── taxonomy.json
├── ontology.ttl
├── coverage_report.md
├── scenarios/
│   ├── L1_airspace/              # 7 base variants
│   ├── L2_infrastructure/        # 9 base variants
│   ├── L3_dynamic_constraints/   # 5 base variants
│   ├── L4_agents/                # 24 base variants
│   ├── L5_environment/           # 9 base variants
│   ├── L6_digital_layer/         # 10 base variants
│   └── X_cross_layer/            # 6 cross-layer chains
└── tools/
    ├── spec_compiler.py
    ├── regenerate_boundary_scenarios.py
    ├── validate_coverage.py
    ├── validate_scene_grounding.py
    ├── batch_generate.py
    ├── batch_render_dataset.py
    ├── generate_graph_labels.py
    ├── export_pyg.py
    ├── export_rdf.py
    └── statistics.py
```

## Quick Start

Regenerate grounded scenario configs:

```bash
python Dataset/tools/regenerate_boundary_scenarios.py
```

Validate taxonomy coverage:

```bash
python Dataset/tools/validate_coverage.py --dataset-root Dataset
```

Validate scene grounding:

```bash
python Dataset/tools/validate_scene_grounding.py --dataset-root Dataset
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Base scenarios | 64 |
| Cross-layer chains | 6 |
| Total scenario definitions | 70 |
| Event types | 33 |
| ODD layers | 6 |
| Mechanism categories | 5 |
| Event steps | 257 |
| Grounding validation | 70/70 pass |
| CAAC emergency coverage | 12/14 (2 out of scope) |

## Event Type Summary

| Layer | Name | Types | Scenario definitions |
|-------|------|-------|----------------------|
| L1 | Airspace Structure | 4 | 7 |
| L2 | Ground Infrastructure | 5 | 9 |
| L3 | Dynamic Constraints | 3 | 5 |
| L4 | Agents | 11 | 24 |
| L5 | Environment | 5 | 9 |
| L6 | Digital Layer | 5 | 10 |
| X | Cross-layer Chains | 6 chains | 6 |

## License

This dataset is intended for capture, validation, and code-driven generation workflows.
