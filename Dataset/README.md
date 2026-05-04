# Low-Altitude Urban Air Mobility (UAM) Multi-Modal Event Dataset

## Overview

A systematically designed multi-modal dataset of low-altitude urban air mobility event scenarios. The dataset contains **64 event scenarios** organized across **33 event types** within a UAM-adapted PEGASUS 6-layer Operational Design Domain (ODD) framework. Each scenario generates event traces, agent trajectories, weather metadata, per-frame heterogeneous graph labels, and RDF knowledge graph triples.

**Version**: 1.0.0
**Date**: 2026-04-29

## Key Features

- **Scientifically grounded taxonomy**: 33 event types derived from NASA Belcastro et al. (2017) 4-domain hazard taxonomy, cross-validated with CAAC MinGui [2025] No.15 (14 emergency event types), and JARUS SORA v2.5 risk framework
- **6-layer UAM ODD model**: Adaptation of PEGASUS (Scholtes et al., 2021) 6-layer model for UAS operations
- **Heterogeneous graph labels**: Per-frame graphs with 7 node types (UAV, ground_vehicle, pedestrian, infrastructure, zone, weather, event) and 14 edge relation types
- **RDF knowledge graph**: OWL-DL compatible with 47 classes, 14 object properties, 12 data properties. ~57K triples across 64 episodes
- **Deterministic generation**: All events generated via declarative JSON event scripts with causal chain definitions
- **Multi-agent coverage**: UAVs, ground vehicles, pedestrians, infrastructure, communication links, weather

## Directory Structure

```
Dataset/
├── README.md
├── taxonomy.json              # 33 event types with scientific references
├── ontology.ttl               # OWL-DL TBox (Turtle format)
├── coverage_report.md         # Generated coverage report
├── scenarios/                 # 64 scenario definitions (spec.py + event_script.json)
│   ├── L1_airspace/           # 4 event types, 7 variants
│   ├── L2_infrastructure/     # 5 event types, 9 variants
│   ├── L3_dynamic_constraints/ # 3 event types, 5 variants
│   ├── L4_agents/             # 11 event types, 24 variants
│   ├── L5_environment/        # 5 event types, 9 variants
│   └── L6_digital_layer/      # 5 event types, 10 variants
├── episodes/                  # Generated episode data (64 episodes)
│   └── {scenario_id}__seed{NN}/
│       ├── episode_manifest.json
│       ├── global_entity_roster.json
│       ├── event_trace.jsonl
│       ├── trajectories.jsonl
│       ├── weather_meta.jsonl
│       └── graphs/
│           ├── frame_graphs.jsonl
│           └── episode_graph.nt
├── knowledge_graph/
│   ├── tbox.ttl               # OWL-DL TBox
│   ├── merged_abox.nt         # Global merged ABox (~57K triples)
│   └── splits/                # Train/val/test episode lists
└── tools/                     # Python tools
    ├── spec_compiler.py       # ScenarioSpec → event_script.json compiler
    ├── action_templates.py    # Shared action templates
    ├── generate_p0_scenarios.py # P0 scenario batch generator
    ├── batch_generate.py      # Episode batch generator
    ├── generate_graph_labels.py # Frame graph label generator
    ├── validate_coverage.py   # Coverage validation
    ├── export_pyg.py          # Export to PyG HeteroData format
    ├── export_rdf.py          # Export to RDF NTriples format
    └── statistics.py          # Dataset statistics
```

## Quick Start

### Validate coverage
```bash
python Dataset/tools/validate_coverage.py --dataset-root Dataset
```

### Generate statistics
```bash
python Dataset/tools/statistics.py --dataset-root Dataset
```

### Export to PyG format
```bash
python Dataset/tools/export_pyg.py --dataset-dir Dataset/episodes --single-file --output Dataset/pyg_export/all_frames.json
```

### Export RDF knowledge graph
```bash
python Dataset/tools/export_rdf.py --dataset-dir Dataset/episodes --output Dataset/knowledge_graph/merged_abox.nt
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total scenarios | 64 |
| Event types | 33 |
| ODD layers | 6 |
| Mechanism categories | 5 |
| Episodes (1 seed) | 64 |
| Total events | 137 |
| RDF triples | 57,450 |
| Frame graphs | ~5,760 (90 frames × 64 episodes) |
| CAAC emergency coverage | 12/14 (2 out of scope) |

## Event Type Summary

| Layer | Name | Types | Description |
|-------|------|-------|-------------|
| L1 | Airspace Structure | 4 | Geofence violations, altitude deviations, intrusions, congestion |
| L2 | Ground Infrastructure | 5 | Station failure, GNSS degradation, charger/pad issues, signal faults |
| L3 | Dynamic Constraints | 3 | Road construction, temporary NFZ, emergency isolation zones |
| L4 | Agents | 11 | UAV-UAV/vehicle/pedestrian/building conflicts, vehicle collisions, pedestrian incidents, crowd events |
| L5 | Environment | 5 | Rain, fog, wind, lighting, temperature |
| L6 | Digital Layer | 5 | C2 loss/degradation, GNSS spoofing, comm jamming, GCS compromise |

## Graph Label Format

Each frame graph is a heterogeneous graph with:
- **Node types**: uav, ground_vehicle, pedestrian, infrastructure, zone, weather, event
- **Edge types**: spatial:near, spatial:approaching, spatial:inside, causal:triggers, temporal:before
- **Compatible with**: PyG HeteroData, DGL HeteroGraph, NetworkX

## References

- Belcastro, C. M., et al. (2017). "Hazards Identification and Analysis for Unmanned Aircraft System Operations." AIAA 2017-3269.
- CAAC MinGui [2025] No.15. "Civil Unmanned Aircraft Event Information Management Measures."
- JARUS SORA v2.5. "Specific Operations Risk Assessment."
- ICAO CICTT. "Aviation Occurrence Categories." ADREP Taxonomy.
- Scholtes, M., et al. (2021). "6-Layer Model for a Structured Description and Categorization of Urban Traffic and Environment." IEEE Access.
- Mlodzian, L., et al. (2023). "nuScenes Knowledge Graph." ICCVW 2023.

## License

This dataset is intended for research purposes.
