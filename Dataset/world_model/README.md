# World model

Recorded episode states, semantic graphs and observations feed the [multimodal state model](model/README.md). The executable path uses frozen Qwen and Sonata, shared typed prediction heads, future observation features and cached rollout. The retained result covers one TRAIN window; it does not establish generalization or calibration.

- `contracts/`: canonical field schema and predicate scope.
- `data/`: source import, causal windows, serialization and sensor geometry.
- `model/`: encoders, shared heads and the `Dataset.world_model.model.run_episode` entry point.
- `configs/`: current serialization, split and window configuration.
- `models/sonata/`: retained external LiDAR encoder assets in local storage.
- `runs/capture/`, `runs/inputs/`, `runs/observations/`, `runs/prediction/`: prepared captures, canonical inputs, actual observation arrays and the current model output.
- `graph/`: semantic graph projection helpers.

[Current results](../../design/belief/multimodal_state_model.md) document measurements and source limits. P09 owns truth production and repair; see [its source boundaries](../../design/p09/README.md). Raw episodes, weights, runtimes and large tensors remain in local storage. Episode IDs and schema protocol identities are preserved when paths move.
