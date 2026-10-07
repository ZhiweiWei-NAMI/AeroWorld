# World model inputs

This directory consumes recorded episode state, semantic graphs and observations. P09 produces and repairs truth; it does not train a new model or create sensor captures.

The planned shared backbone is Qwen. Sonata (Point Transformer V3-S) is the retained external perception encoder. Its weights and isolated runtime remain under `runs/pretrained_encoders_20260822/` in the working deployment and are not stored in Git. Checkpoints, the LiDAR adapter and tensor dimensions are defined during stage B; the retired three-encoder dimensions are not an input contract.

`p01_schema_rollout_v4/` contains the current structured importer. It imports native world-graph initial assertions as well as deltas, preserving their source identity and existing record fields. The P09 repair does not change fixed cohorts, sampling, seeds or losses, and does not replace existing training caches.

`graph/` owns semantic graph projection helpers. The current P09 sources and execution boundaries are documented in [design/p09](../../design/p09/README.md). Raw episodes, sensor captures, weights and large derived products remain in their declared external storage.
