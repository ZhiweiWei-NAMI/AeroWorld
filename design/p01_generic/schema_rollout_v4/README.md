# P01 current model and results

P01 now has a real Qwen multimodal numerical path: causal graph/state text and actual sensor observations → pretrained Qwen hidden states → shared typed distributions and future regional features → predicted feedback → a second cached prediction. The current result is a **single TRAIN-window experiment**, including an unsuccessful parameterization and a measured correction. It is not a completed generalization or calibration study.

The [source README](../../../Dataset/world_model/p01_schema_rollout_v4/README.md) gives the architecture, interfaces and executable command. [Implementation decisions](implementation-plan.md) distinguish user requirements, reconstructed decisions and remaining work. [B01](B01-specification.md) preserves the original four data/schema responsibilities with current references.

## What actually ran

`L4-1_v1__seed00`, observer `u_inspect_l4_1_v1`, uses structured history ticks 0–5 and the actual capture at tick 5. At 10 Hz, target ticks 10 and 15 are 0.5 and 1.0 seconds after cutoff. Training supervises tick 10. Evaluation compares a second predicted step with a direct prediction of tick 15; both are within the same TRAIN episode.

The backbone is frozen Qwen3.5-2B-Base, BF16, hidden width 2,048, 24 layers (18 linear attention, six full attention), with its native vision encoder and a 262,144-token configured limit. Sonata is the sole external perception encoder. The implementation calls `Qwen3_5Model`, not the language head or `generate`. The retained checkpoints are documented in the source command and runtime loader. Official interfaces: [Qwen model](https://huggingface.co/Qwen/Qwen3.5-2B-Base), [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [Sonata source](https://github.com/facebookresearch/sonata).

The current canonical file contains 180,413 records: 78 entities, 41,472 frames, 138,831 edges, 10 events and 22 observation indices. It retains the earlier import's selected observation indices; the full capture inventory is separate. The actual model window contains 28 causal actors, 166 state records, 1,814 graph records (1,307 initial assertions and 507 causal changes) and three real communication rows. There are 281 referenced graph identities and 232 scalar query addresses per target. Query actors come from historical states, not the future roster or target visibility.

Structured text is a reversible projection of retained causal world truth. Archive bookkeeping and the renderer's three hardcoded network fields are excluded. Actual latency, loss ratio, handover and quality come from `aw_data/compute_comm_supplement/L4-1_v1__seed00/communication_state.jsonl` (producer 1.6.0, 914 rows). This is controlled simulation, not radio hardware or ns-3 output. The three historical communication rows are also the current fixed-template text-log input. An independent UE natural-language log stream has not been established. `annotations.visibility_state` describes renderer submission scope and is not an instance visibility label or prediction target. Weather/dust is not supervised in this run.

Source joins use `(episode, tick, raw entity ID)`. For example, canonical line 235 for the observer at tick 5 maps to `truth_frames.jsonl` line 6 and graph node `formal:L4-1_v1__seed00|s:u_inspect_l4_1_v1@5`. The existing PyG path is `Dataset/world_model/graph/model_data.py::convert_window` / `prediction.py::to_pyg`. P01's typed projection is a dictionary interface; this experiment serializes graph semantics into Qwen text rather than training a PyG message-passing adapter.

## Measured result

The fresh run used seed 20261007, eight AdamW updates, learning rate 0.00005, gradient clipping at 1 and 1,024-token chunks. There were 3,199,878 trainable parameters. Each recorded step decreased the training objective, from 3.100339 to 3.055046 after the final update. All projector, geometry-adapter, typed-head, future-feature and decoder branches received finite nonzero gradients; reported norms are before clipping.

| Metric | First step, tick 10 | Two-step rollout, tick 15 | Reference |
|---|---:|---:|---|
| Position component MAE, m | 0.017795 | 0.060643 | Constant velocity: 0.016397 / 0.057221 |
| LiDAR common-hit range MAE, m | 13.223965 | 18.357056 | Persistence: 13.760269 / 20.861345 |
| LiDAR occupied-cell accuracy | 0.987305 | 0.984863 | Same as persistence |
| Communication latency MAE, ms | 0.751706 | 1.538163 | Persistence: 0.763410 / 1.562827 |
| Communication loss-ratio MAE | 0.000199 | 0.000398 | Persistence: 0.000249 / 0.000495 |

Position is measured over 84 scalar components, not 28 Euclidean distances. Speed, velocity and yaw also remain slightly worse than their historical priors. Direct tick-15 position MAE is 0.059939 m, below rollout's 0.060643 m but above constant velocity. Communication has only two UAV samples; handover and quality remain unchanged and both model and persistence score 100%. All evaluated state labels are present. These samples cannot assess absence handling or category transitions.

LiDAR comparisons use the same scan-cell support: 1,028 common occupied cells at tick 10 and 1,025 at tick 15. Actual targets occupy 1,054 and 1,053 of the 2,048 scanned angular cells. The model predicts the same occupied-cell grid as persistence; range regression improves while new-return detection does not. Predicted points are coarse conditional means in the target sensor frame. The [tick-10 figure](derived/current_model/lidar_tick_000010.png) and [tick-15 figure](derived/current_model/lidar_tick_000015.png) show prediction, historical persistence and actual future coarse returns with shared axes.

| Future feature MSE on common support | Tick 10 model / persistence | Tick 15 model / persistence |
|---|---:|---:|
| RGB | 1.531349 / 1.536487 | 1.588019 / 1.595087 |
| Depth | 0.201925 / 0.205442 | 0.403249 / 0.416510 |
| Segmentation | 0.012466 / 0.012500 | 0.034320 / 0.034397 |
| LiDAR | 0.818558 / 0.840245 | 1.990297 / 2.024442 |

These are normalized feature errors with small improvements. They do not establish visually correct future images. The LiDAR loss additionally uses conditional-distance masks and separate geometry/return terms, so its training-loss value is not the common-support metric above.

The nominal central 90% position intervals cover 84/84 components at tick 10 and 82/84 at tick 15; mean half-widths are 1.089269 and 1.089606 m. They are wide relative to point error. The current scale uses raw position-increment RMS even though the mean is anchored to constant velocity; constant dimensions use an explicit model prior. The next scale comparison must fit TRAIN constant-velocity residuals and evaluate calibration on held-out windows. High coverage here is not evidence of calibrated probabilities. Rollout currently feeds point estimates, not samples or integrated continuous distributions, so second-step uncertainty is conditional on point feedback.

An initial unanchored head produced 19.624349 m position component MAE and a training-loss spike to 193.434814 despite a lower final objective than initialization. The retained correction uses type-specific residual outputs, causal physical priors and a conditional-distance/return LiDAR representation. This is an architectural correction across different objectives and learning rates, not an isolated ablation. Earlier LiDAR all-target errors and persistence common-hit errors used different masks and must not be compared as a gain. The current result retains the earlier measurements and those qualifications.

## Tokens, cache and resources

The final text has 219,777 tokens; native RGB brings the prefix to 219,909. Historical sensor suffix length is 374 and the predicted append is 30,627, yielding 250,910 cached tokens. The prior cutoff-250 serialization measurements were 13,576,377 JSONL tokens, 11,062,110 compact-table tokens and 14,491,892 typed-view tokens, all far beyond this context. The implemented frozen-prefix symbol/change format fits the small window; long real histories still need a stable incremental codec and explicit observation scheduling. No truncation or one-token-per-entity assumption resolves that problem.

Before the final run, the capacity calculation reserved 40,000 predicted tokens: 260,377 total, 3,199,512,576 bytes for full-attention KV, 12,798,050,304 bytes for saved CPU-expanded KV and an estimated GPU ceiling of 18,456,125,120 bytes. Assumptions were BF16, six full-attention layers, two KV heads, head width 256, two cache copies and 6 GiB workspace. This estimate is separate from measured peak allocated GPU memory, **17,842,111,488 bytes**, and complete run wall time, **392.13 seconds**.

An empty-cache recomputation using the identical chunk/sensor/append schedule matched cached hidden states and every typed output exactly. Cached append took 24.74 seconds and full recomputation 134.64 seconds in this run. An earlier merged-chunk comparison had maximum hidden difference 3.75, so schedule-independent numerical equivalence is not established. Real-history symbol-table changes, backbone updates or changed projection inputs invalidate the corresponding cache.

Runtime fixes are local to P01: large saved attention tensors are offloaded while padded/broadcast masks preserve CUDA alignment; mutable cache state is copied for backward; Sonata sparse convolutions retain the input Jacobian; a native tensor CSR reduction replaces the incompatible scatter backward. Checkpoint weights and external runtime source were not edited. The matched forward comparison covered the actual four max and four mean reductions.

## Episode and geometry evidence

The [episode viewer](derived/pilot_observations/episode_view.html) contains 180 real RGB frames at ticks 5–900 and all four modalities' archive coverage. Six times (5, 10, 15, 250, 255, 260) have materialized geometry. Actual model region bindings cover ticks 5/10/15: historical input, predicted append and prediction-head output respectively. Tick 15 has no invented sequence token range. Future RGB thumbnails are actual labels, not decoded predictions. The [observation README](derived/pilot_observations/README.md) gives the exact arrays, transforms, scan-index mapping and reproduction commands.

The archive has 180 payloads and 180 sidecars per modality; tick 0 is absent. This episode therefore does not satisfy the formal 181-capture-frame contract, and this inspection does not establish coverage of the entire 210-episode corpus. Every LiDAR scan contains 262,144 ordered rays; no-return coordinates are masked rather than interpreted as real zero-distance points.

Depth/LiDAR alignment has a concrete unresolved source issue. Axial median absolute residuals are about 0.020 m at tick 5, but 10.80 and 10.67 m at ticks 10 and 15; the reported LiDAR pose does not explain the discrepancy. The custom depth producer is absent from the scoped project source. Both axial and radial hypotheses remain visible. AirSim's [documented depth modes](https://microsoft.github.io/AirSim/image_apis/#depthplanar-and-depthperspective) cannot certify this custom producer's semantics.

Segmentation is class-ID, not instance-ID. All six inspected sidecars have null measured world positions for 22 actors, and no per-actor measured bounds or point-hit identities. Observer rigs are measured; UAV command centers and landing-pad spawn centers are labeled candidates. Twelve corridors are logical regions without rasterization. Exact entity-to-pixel/point binding therefore needs P09/UE producer fields, not fabricated geometry. Required producer outputs are capture-time actor/component identity, transform, bounds, ground reference, timestamp, instance-pixel IDs and LiDAR hit IDs. No trajectory, entity, weather or sensor configuration was changed. A recapture list should follow a completed producer fix, not precede it.

The viewer's embedded metadata and token addresses were independently compared with the saved JSON; image files were opened and JavaScript syntax checked. Chromium is absent, so no browser interaction run is claimed.

## Evidence and next work

- [result.json](derived/current_model/result.json): executed architecture, losses, gradients, baselines, cache differences and interval diagnostics.
- [predictions.json](derived/current_model/predictions.json): every typed distribution and fixed address.
- [bindings.json](derived/current_model/bindings.json): schema/entity spans and observation regions.
- [semantic_input.txt](derived/current_model/semantic_input.txt): exact model-visible structured text.
- Local `head.pt` and `future_features.npz`: trained parameters and predicted features for decoder replay; targets are recomputed from immutable observations with the frozen teacher. These local tensors stay outside Git alongside raw arrays and canonical input.

Independent native agents reviewed intent, source causality, implementation and final saved outputs through successive corrections. Saved state metrics were recomputed against original truth/communication rows with zero difference; CPU decoder replay matched the GPU common-hit errors within 0.000002 m. The pinned GLM route failed resolution with `no adapter registered for provider "workbuddy"`; it never started inference. Development used the handover-authorized Sol/high fallback, with separate Sol/xhigh review. GLM completion is not claimed.

The next model work is to fit residual scales on TRAIN, evaluate fixed held-out windows against persistence/constant velocity, and make real-history encoding incremental before increasing context. Geometry work needs the named P09/UE producer fields and coupling of predicted observer pose with future feature geometry. This current small closed loop is ready for review; those research outcomes remain open.
