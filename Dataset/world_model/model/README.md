# Multimodal state model

This package runs a real Qwen multimodal forward pass, shared typed prediction heads, future observation features and a two-step cached rollout. The measured pilot uses one TRAIN window from `L4-1_v1__seed00`: history ticks 0–5, prediction at tick 10, then predicted feedback for tick 15. It establishes an executable architecture; it does not establish generalization or probability calibration.

The implementation uses **Qwen3.5-2B-Base**, its native RGB encoder and `Qwen3_5Model` hidden states. The language generation head is never called. Qwen and Sonata are frozen; 3,199,878 parameters in projection, geometry adaptation, shared distribution heads, future-feature heads and the coarse LiDAR decoder are trained. No LoRA, SFT or RL is used in this run. Keeping pretrained backbone weights preserves the starting representation, but original language capabilities have not been reevaluated with these added sensor tokens.

## Input and output

Static field definitions, units, entity identities, graph predicates and sensor settings precede structured historical records. The causal input contains the full initial truth graph and changes through cutoff, entity states and real simulated communication rows. Three renderer defaults (`annotations.network_latency_ms`, `annotations.network_packet_loss`, `annotations.network_status`) are excluded; actual `communication.*` values come from `aw_data/compute_comm_supplement/`. World truth is separate from observer visibility.

| Input | Actual representation in this run | Backbone insertion |
|---|---|---|
| Structured history and text logs | Reversible symbol/change text; 219,777 tokens including definitions | Text prefix |
| RGB | Native Qwen vision; 98 × 2,048 features, merged 7 × 14 grid | Native image placeholders and M-RoPE in prefix |
| Depth | 128 × 5 patch statistics: mean, std, min, max, valid fraction | Learned projection in sensor suffix |
| Class segmentation | 128 × 256 class fractions, learned class embeddings | Learned projection in sensor suffix |
| LiDAR | 32 × 647 region features: Sonata, geometry, conditional range and return indicators | Learned geometry adapter and feature projection in sensor suffix |

The RGB-bearing prefix is 219,909 tokens. The sensor suffix is 374 tokens, including modality/region structure. One predicted append adds 30,627 tokens, producing a 250,910-token cache within the checkpoint's 262,144-token limit. A visual observation is not one token. Bindings retain episode, tick, observer, sensor, nominal region, source path and actual sequence span. Native attention mixes spatial context, so nominal regions are not exclusive receptive fields.

`QueryAddress(episode_id, entity_id, field_family, target_tick, component)` determines every output location before prediction. At each target, 28 causal actors produce 232 scalar queries: position, velocity, speed and yaw, plus four communication fields for two UAVs. Entity and field spans, global context, history and horizon feed one shared trunk with type-specific outputs; parameters are shared across entities. Future labels do not select query members.

Unbounded reals use Normal laws; nonnegative and bounded fields use boundary-inflated Gamma and endpoint-inflated Beta laws; angles use Von Mises; booleans and enums use categorical probabilities. Recording probability and legal cell-kind probabilities are distinct. Program code assembles addresses, units, enum values and distribution parameters. For example, the saved tick-10 UAV position component 0 has Normal location `6628.351230048681 m` and scale `0.8550742508147626 m`; this is a numerical head result. Type correctness, physical error and calibration have separate measurements.

Four future-feature heads predict residuals relative to the current regional features. A frozen teacher supplies future targets only during supervision. Predicted RGB features return through native image positions; other predicted features return through their projection layers. A learned 64 × 32 angular range/return decoder makes LiDAR geometry inspectable. It is a coarse reconstruction, not a full-scan inverse of Sonata. No RGB image decoder is claimed.

Rollout feeds conditional point estimates and predicted features back into the context. It currently conditions the next distribution on these point estimates; it does not marginalize trajectory uncertainty. Sensor grids remain local to each predicted observer time, without predicted world-pose fusion. Native attention KV, linear-attention recurrent state and convolution state are cloned for independent branches. Cache reuse requires unchanged prefix, weights, positions and projections. Extending real history currently rebuilds the symbol table and therefore requires a fresh prefix.

## Reproduce the current pilot

Run from the repository root. Raw capture preparation and full-episode visualization commands are in the [observation README](../runs/observations/README.md). The local canonical import and retained checkpoints must already exist; raw episodes, runtimes and weights stay outside Git.

```bash
CUDA_VISIBLE_DEVICES=1 \
Dataset/world_model/runtimes/qwen35/bin/python -m Dataset.world_model.model.run_episode \
  --canonical Dataset/world_model/runs/inputs/canonical.jsonl \
  --communication aw_data/compute_comm_supplement/L4-1_v1__seed00/communication_state.jsonl \
  --model /mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35/hf/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --observations Dataset/world_model/runs/observations \
  --cutoff 5 --step 5 --steps 8 --lr 5e-5 --grad-clip 1 --chunk 1024
```

The Qwen runtime used torch 2.5.1 and transformers 5.18.0. The retained Sonata source, weights and dependencies are loaded from `Dataset/world_model/models/sonata/`. `--resume-head` evaluates saved weights and replays a backward pass for gradients; it does not resume optimizer training. The default output is `Dataset/world_model/runs/prediction/`.

The run completed in 392.13 seconds with peak allocated GPU memory of 17,842,111,488 bytes. Its objective decreased from 3.10034 to 3.05505 over eight AdamW updates. Same-schedule cached and full-prefix recomputation gave zero hidden-state and head-output differences. Position component MAE was 0.01780 m versus 0.01640 m for constant velocity; LiDAR common-hit range MAE was 13.224 m versus 13.760 m for persistence. See the [current results and limitations](../../../design/belief/multimodal_state_model.md) before interpreting these numbers.

## Source responsibilities

- `contracts/schema.py`, `contracts/schema.json`: declared types, units, legal values and non-present semantics.
- `data/import_episode.py`, `serialization.py`, `windows.py`, `normalization.py`, `provenance.py`: authentic source import, reversible archive/typed views, cutoff and split scope.
- `data/observations.py`, `geometry.py`: actual sensor arrays, timestamps, calibration, scan regions and episode visualization.
- `model/causal_input.py`, `query_head.py`: causal graph/state index, model-visible text, fixed addresses and TRAIN-prefix scales.
- `model/multimodal.py`, `distributions.py`, `run_episode.py`: encoders, projections, laws, losses, numerical execution, cache rollout and saved results.

`Dataset` remains a namespace package. This package does not alter simulator trajectories, UE capture settings or P09 truth producers.
