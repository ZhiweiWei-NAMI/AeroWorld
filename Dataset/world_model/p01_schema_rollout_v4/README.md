# P01 schema rollout v4

This package provides the executable P01 canonical-record schema, lossless
serialization, causal supervision windows, typed/model-visible projections,
and the first dynamic entity-field query checkpoint.

`p01v4.model` compiles query membership from cutoff-legal typed input records
and `FIELD_REGISTRY`. Each address is the exact tuple
`(episode_id, entity_id, field_family, target_tick, component)`. Entity ids are
join keys only. All rows use one shared MLP and output projection; future
values and availability attach as labels after the query inventory is fixed.
Real scalar/vector components, circular and bounded metadata, declared enums,
booleans, and the four explicit non-present kinds are supported. Open string
fields without a declared vocabulary are excluded before targets are read.

The CPU checkpoint accepts an existing canonical JSONL file; canonical data is
deliberately not stored in Git. From the repository root:

```bash
CUDA_VISIBLE_DEVICES= \
PYTHONPATH=Dataset/world_model/p01_schema_rollout_v4/src \
python -m p01v4.model.cpu_preliminary \
  --canonical /path/to/pilot.canonical.jsonl \
  --receipt /tmp/p01-b02.json \
  --threads 4 --seed 20261006
```

The measured S02 pilot receipt is stored in the existing
`design/p01_generic/schema_rollout_v4/reports/B02/` results directory. It records the real CPU
forward, backward, optimizer update, dynamic shapes, gradient evidence, and
query-order/inventory checks. It is a preliminary typed-head checkpoint. Graph
context, Qwen/LoRA, multimodal tokens, rollout, and multi-episode evaluation
remain later model stages.
