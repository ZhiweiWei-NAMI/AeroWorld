# P01 implementation decisions

This document is reconstructed from the user's takeover requirements and current executable code. It is not a recovered copy of the truncated historical plan. The original B01 text contains four complete schema/data tasks; the recovered S03 title concerns sensor geometry and episode token-binding visualization, not a training stage. No missing historical milestone is treated as a user decision.

## Fixed user requirements

Qwen is the shared pretrained backbone. Structured variable-size world information and multimodal observations must produce entity-field probability distributions through program-defined dynamic queries and shared heads. Future multimodal embeddings are part of the task. Numerical output is direct, without asking a language model to write names or numbers. A causal single-step rollout and history-cache reuse must be compared with direct future prediction. Real sources, source traceability and world-truth/visibility separation govern the interface.

The user also requires a real episode explanation, sensor geometry, token/entity mapping, visual inspection of LiDAR prediction and actual execution before large training. The current implementation addresses these with explicit limits below. The two-step pilot is an experiment, not a decision to fix the model at two or ten future times.

## Current implementation choices

| Question | Implemented choice | Evidence and consequence |
|---|---|---|
| Graph versus structured text | Full causal initial graph plus delta operations serialized with state/schema definitions | Preserves nodes, predicates, roles and cross-subgraph references; PyG remains a separate consumer, not the backbone input tensor |
| JSON, compact table or field text | Canonical JSONL for traceability; typed/compact views for interfaces; reversible symbol/change text for Qwen | Cutoff-250 legacy encodings exceed 11 million tokens; current short prefix fits 219,777 structured tokens |
| Backbone access | Frozen `Qwen3_5Model` hidden states and native visual embeddings | No LM head/generation; no LoRA in this run |
| Dynamic entities | Query addresses compiled from causal frame entities and declared field components | 28 actors, 232 slots per target; no entity-specific output layer |
| Shared output | Shared contextual trunk, separate outputs by mathematical field type | Normal, Gamma with boundary atom, Beta with endpoint atoms, Von Mises and categorical distributions |
| Direct future versus rollout | Shared horizon-conditioned head used for direct tick 15 and tick-10 feedback then tick 15 | Direct position error is slightly lower in this TRAIN example; neither beats constant velocity |
| State feedback | Conditional point, recording probability and predicted cell kind | Next-step distributions do not integrate earlier continuous uncertainty |
| Future embeddings | Current-region features plus learned residual, frozen future teacher | Explicit RGB/depth/class-seg/LiDAR targets and bindings; common-mask feature error and coarse LiDAR geometry evaluated |
| Spatial support | Native RGB grid, depth/seg patches, indexed LiDAR angular regions | Nominal region labels survive encoding; attention mixes regions; precise actor-instance support needs producer data |

The static schema fixes units and lawful cell kinds before temporal values. Per-entity TRAIN-prefix scales are data records, not per-entity trained heads. An explicit initialization prior for a constant dimension is a model assumption, distinct from replacing a missing truth label by zero. Invalid depth/no-return masks remove unsupported distances from statistics and loss; their storage zeros do not assert world truth.

Sonata receives coordinates and computed PCA normals plus a learned geometry adapter in its checkpoint's three color channels. These are learned geometric features, not measured RGB. The teacher uses a frozen initial adapter. This keeps the pretrained encoder available without claiming that class segmentation, depth and point clouds share Qwen's RGB encoder or that projected point colors have been validated.

The final LiDAR representation retains 512 Sonata channels, seven geometry/occupancy statistics, 64 conditional-range logits and 64 return indicators per region. A coarse decoder predicts 64 × 32 acquired-return cells, with range limited by the actual 200 m sensor configuration. It is not a reversible decoder for the original 262,144 rays. Images shown at future times remain truth references; no future RGB image decoder exists in this package.

## Why the current correction was made

The initial unanchored experiment lowered its total objective but produced 19.624349 m position error and a loss spike. Shared hidden context alone did not provide a suitable physical parameterization. The current head starts from causal motion/history priors with zero residual outputs; bounded and nonnegative means retain physical support; LiDAR uses persistence plus learned residuals and conditional-distance masks. Eight updates then lowered the objective steadily and improved common-hit range error, while motion error remained slightly worse than the prior. Objective values from the two formulations are not a controlled ablation.

Three renderer hardcoded network values were traced to the source converter and removed only from P01's model-visible projection. Actual simulated communication is indexed from its producer output for both input and supervision. Raw archival records and P09 producer code are unchanged. The current full-schema roundtrip claim applies to the retained semantic projection, not to excluded archive/default fields.

## Relevant open implementations

The user's reference name remains **jev**. A plausible match is TypeSafe's Jev, but the historical intended identity was not established. Official [TypeSafe documentation](https://docs.typesafe.ai/introduction) describes typed model interfaces, and [Jev 1.13 model discussion](https://docs.typesafe.ai/model-jaggedness/jev-1.13) identifies limitations in recovering precise continuous quantities. These do not establish the internal implementation needed for P01.

The community [vllm-jev model](https://github.com/mode-io/vllm-jev/blob/main/vllm_jev/model.py) uses pooled Qwen state and a score head; its [endpoint](https://github.com/mode-io/vllm-jev/blob/main/vllm_jev/endpoint.py) calls `encode`. It is a useful example of prediction without text generation, not official evidence for Jev's architecture or a drop-in heterogeneous world-state model. Its [experimental prefix-cache guide](https://github.com/mode-io/vllm-jev/blob/main/docs/guide.md#online-prefix-cache-experimental) records practical counterexamples to assuming cache gains and unchanged outputs. Community [OpenJevv](https://huggingface.co/ldov/openjevv) is another bounded scoring example; it does not solve sensor geometry, calibrated continuous laws or future-observation decoding.

Qwen's [official checkpoint](https://huggingface.co/Qwen/Qwen3.5-2B-Base) and [Transformers interface](https://huggingface.co/docs/transformers/model_doc/qwen3_5) provide the actual local model route. [Sonata](https://github.com/facebookresearch/sonata) provides pretrained point features. The current code uses these real interfaces; claims about precision, cache behavior and geometry are based on the saved execution, not inferred from model availability.

## Next concrete changes

1. Fit the state distribution scale to TRAIN residuals of the implemented physical prior. Preserve split membership and report held-out NLL, interval coverage/width and point errors against the same-address baselines. The current 90% position intervals are too wide to support a calibration claim.
2. Replace prefix-wide alias reassignment with stable incremental graph/state encoding. Measure token cost and cache equality when appending real history, then schedule multiple historical captures within a declared budget. Keep all retained causal semantics; do not silently truncate graph context.
3. Couple predicted observer pose and sensor-local future features before claiming world-coordinate multimodal consistency. Continue reporting direct-horizon versus conditional point-feedback rollout separately; trajectory uncertainty propagation needs an explicit sampling or distributional recurrence decision.
4. Have P09/UE produce capture-time actor/component transforms, bounds, instance IDs, hit IDs and verified depth semantics. Once those inputs are stable, derive the precise recapture list. Current measured rigs and class labels support regional observations, not exact per-entity pixel or point identity.

Larger episode training follows these architecture and input improvements. Supervised typed losses and feature/geometry losses already train the new interfaces. SFT would concern retained language/instruction behavior; RL would need a defined sequential objective and evaluation. Neither is a required substitute for the current supervised numerical path.
