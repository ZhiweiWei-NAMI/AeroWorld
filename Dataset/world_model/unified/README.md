# Unified world model with task-aware context compression

This directory implements `Dataset/world_model/RESEARCH_PLAN.md`. Outputs go to `Dataset/world_model/runs/unified/`.

Run every command from the repository root with:

```
PY=Dataset/world_model/runtimes/qwen35/bin/python
PYTHONPATH=. PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Modules

| Module | Plan | Content |
|---|---|---|
| `schema.py` | §2–§4 | Fields, quanta, horizons, answer alphabets, short names |
| `sources.py` | §2 | Reads one parent's producer streams: truth frames, world-truth graph (all predicates with evidence), communication/compute state, objective and producer events, scheduled activations. Masks labels that disagree with the producer rule (§2 label validity) |
| `compaction.py` | §3.2 | Hold-last and velocity-hold motion-knot compaction on integer quanta, with exact reconstruction checks |
| `windows.py` | §3, §5 | Window aliases, compacted record groups (candidates), origin lines, headings, queries and labels |
| `visual.py` | §3.5 | Every actual capture of every observer in the 5 s history as a candidate (four-modal UAV captures and RGB-only fixed-camera captures); capture metadata (modalities, pose, height, optical axis, field of view, entities in view, relations), rendered per query set; geometric relevance; feature loading |
| `frame_cache.py` | §3.5 | Compact per-capture predictor inputs (fp16 RGB tokens, depth/seg codec features, LiDAR statistics + Sonata regions) and per-episode capture metadata with a modality mask and a frozen-feature summary vector for the selector; `--rgb-only` adds the fixed overview camera (camera models from the capture sidecars) |
| `assemble.py` | §3.3–§3.5, §6 | Query sets, text and capture units, the 20K total budget, window background (conventions, queried predicate definitions, answer codes), deterministic serialization, simple selection policies |
| `codes.py` | §4.2–§4.3 | Prefix trees, interval codes (I₀ and log-spaced integer bins up to a bounded support), query encoding, losses, CRPS/RPS |
| `predictor.py` | §4.4 | Qwen3.5-2B with LoRA and a sensor adapter. Marginal mode in query microbatches with per-row hybrid-state isolation; joint mode is time-major teacher forcing. Restricted LM-head scoring |
| `data.py` | §8 | Development holdout, items under the total budget, mixed text/capture training reductions, joint order, group weights |
| `train_predictor.py` | §9 steps 1 and 4 | Multi-GPU (torchrun) predictor training: identical initial weights on every rank (broadcast and checked), DEV convergence checks, code commit recorded |
| `supervise.py` | §9 step 2 | Candidate-set losses over text and captures (policies, random subsets, within-budget block replacement, complementary text–visual groups, swaps, removal effects), predictor frozen |
| `selector.py`, `train_selector.py` | §9 step 3 | Qwen3.5-0.8B chunked reader over record lines and capture metadata (+ summary embedding), bidirectional decision head across chunks; soft Boltzmann inclusion targets + ranking of removal effects |
| `scorer.py` | §10 | Small structured scorer trained on the same targets (selector-capacity comparison) |
| `evaluate.py` | §10 | Loss–budget curves per target group (NLL, Brier, coarse RPS, joint NLL, positions, captures kept) with paired, parent-clustered bootstrap |
| `arms.py` | §8 | ARM paired intervention/counterfactual evaluation on shared parent evidence |
| `convergence.py` | §9 | Training/DEV curves, convergence assessment, checkpoint selection |
| `build.py` | §2, §4.3 | Corpus passes: cached episode sources, `configs/unified_schema.json` (event types, bounded supports), `configs/unified_predicates.json` (predicate expressions and thresholds) |
| `sonata_cache.py`, `camera_index.py` | §3.5 | Frozen Sonata regional LiDAR features and per-capture camera models of every parent capture |
| `scripts/` | §9 | `run_loop.sh` (quick closed loop: warm-up, then rounds of supervision → selector → choices → predictor adaptation → DEV check; `SCOPE=all` for all 210 parents and ARM), `status.sh` |

## Inputs and preparation

- **Base embeddings.** All 210 parents have verified four-modal embeddings (`runs/capture_embeddings/feature_cache`, indexes `parent_complete_210` and `parent_v14_36`): 147,035 captures.
- **Sonata regions** (`sonata_cache.py`): frozen Sonata (zero colour slots, 0.1 m grid) pooled into the 32 LiDAR regions, for all 147,035 captures.
- **Camera index** (`camera_index.py`): `camera_to_world = body_to_world @ camera_to_body` and intrinsics per capture, for all 210 parents.
- **RGB-only captures.** Every parent also has a fixed overview camera (`ground_a_overview_top`), RGB every 0.5 s: 38,010 captures with RGB tokens (embedded in `runs/capture_embeddings/rgb_only`). Their camera models come from the per-tick capture sidecars kept in `capture_metadata/*.meta.tar.zst`: a nadir pinhole, intrinsics from the uncropped field of view and the ROI crop. The orientation was checked by projecting a sidecar's entity records onto a surviving image (`runs/unified/audit/rgb_only_orientation_check.png`).
- **Frame cache** (`frame_cache.py`, then `--rgb-only`): compact predictor inputs (~0.46 MB instead of ~2.2 MB with raw rays, ~0.1 s instead of ~0.5 s per load) and `frame_meta/<episode>.pt`. The metadata gives, per capture, the height above ground, ground elevation, RGB token count, modality mask and 2,631-dimensional summary vector. It covers all 185,045 parent captures: 147,035 four-modal and 38,010 RGB-only.
  - Frame metadata and camera models are **required**: `window_captures()` raises for a capture without them.
  - Only the compact feature files have a fallback, and only for four-modal captures (the full feature cache plus Sonata).
- **Observers and captures.**
  - A parent has 3–25 UAV observers (11 in 48 parents) plus the fixed camera.
  - A 5 s window holds 12–119 four-modal captures, ~300 positions each (228 sensor tokens and a ~45-token header). It also holds ~11 RGB-only captures, ~140 positions each.
  - The largest windows exceed the 20K budget on captures alone.

## Implementation notes

**Total budget.** `assemble.TOTAL_BUDGET = 20000` positions is the whole predictor sequence in the training mode. It covers:
- the background;
- the fixed block (identity lines of entities present at t0, origin lines of the query-addressed entities and tuples);
- the selected text and captures (frame header, sensor tokens, newline);
- the query block, reserved label-free: the longest code of every query's answer domain, in the joint sequence or as the longest marginal continuation;
- the padding of the context and of the joint sequence.

The rest after the background, fixed block, queries and padding is one optional capacity shared by text and captures. There is no separate text, observer or frame limit. When everything fits, everything is kept. In practice the fixed block plus background is ~3.4K positions (p90 4.1K) and the label-free query reserve ~1.9K, which leaves ~14.5K optional positions at 20K.

**Background.** It is concise and window-specific:
- the window, frame, quanta and reconstruction conventions;
- an image note when the window has captures;
- the definitions and thresholds of the queried predicates, from the semantic registry;
- the answer-code conventions of the query groups present.

**Captures.** Every capture of every observer at a tick in [t0 − 5 s, t0] is a candidate. Nothing is fixed by observer count or frame count.
- **Predictor input.** A short header (`image observer=u3 t=-1.2 p=[…] agl=… look=[…] fov=HxV rgb+depth+seg+lidar:`), then the sensor tokens of the modalities the capture has. Four-modal captures give RGB merger tokens, 7×7 pooled depth and segmentation, and 32 LiDAR region tokens (Sonata 512 + 7 statistics + presence). RGB-only captures give RGB tokens with their own modality embedding. The tokens carry M-RoPE positions on the frame's grid with their own temporal index.
- **Selector input.** A metadata line instead: modalities, pose, height, optical axis, field of view, token cost, the entities projected into the image (alias, optical depth, image position) and the relations binding them (directly or through owned objects). The query set's addressed entities and queried relations are listed first, then the nearest others, up to 12 entities and 8 relations.
- **Selector summary embedding.** A learned projection maps the 2,631-value frozen-feature summary to one embedding. Each modality segment is normalized separately, absent modalities are zeroed, and a learned "absent" vector per missing modality is added, so missing never reads as observed zeros.
- **Projection.** Entity records keep z above local ground. UAV camera models are in the render world, so points are lifted by the ground elevation under the camera (a flat-ground approximation). The fixed camera's sidecar pose is in the truth frame.

**Fixed block and labels.**
- **Owner bindings.** A queried relation fixes the origin state of the entities it binds, directly or as the owner of a bound object (`u1.flow1` → `u1`, window `rel_entities` built with `src.owner`). That means their frame line, plus the link or compute line for communication or compute predicates. The same map drives capture relevance, capture metadata, complementary groups and the structured scorer.
- **Booleans.** Origin lines list recorded true booleans in `flags=[...]` and recorded false ones in `off=[...]`; a boolean in neither list is unknown.
- **Label validity is per (tuple, tick).** A horizon's value label is kept whenever that tick's label is valid. Only the first-change label needs a valid origin. Query sampling draws tuples from both kinds of query.

**Selection modes.** In `learned` mode (default), text and captures are selected together, and supervision scores the complete selected text–visual set. In `fixed` mode (a text-only stage), the capture packet is held fixed across text subsets: captures in geometric-relevance order within 40% of the optional capacity. Every stage takes it as `--visual {learned,fixed}`.

**Selector decision.** Units are taken in decreasing p_keep while they fit, with headings on demand. There is no division by cost, because a capture (~300 positions) and a record (~15) differ by 20×, and p_keep estimates inclusion in good feasible sets.

**Supervision.** Every scored set fits the total budget.
- Sets come from the simple policies and random mixed subsets of the pool.
- Block replacement then refills a block of the best set from the rest of the pool.
- Complementary groups remove a kept capture, the kept records of the entities it shows within ±1 s, or both; the interaction term is recorded.
- Swaps and removal effects of captures and per-entity record blocks follow.
- The selector learns from Boltzmann soft inclusion targets (τ = 0.02) plus a pairwise ranking loss on the measured removal effects. Groups are scored by their members' mean logit.

**Marginal-mode memory.**
- **Microbatches.** Queries run in microbatches of `--query-chunk` rows: default 4 for training, 8 without gradients.
- **Isolation.** Each query is its own batch row. Its full-attention layers attend to the context K/V (an expanded view, concatenated per row), and its linear-attention layers start from their own copy of the context's convolution and recurrent states. No attention mask is relied on to separate queries.
- **Backward.** Each microbatch back-propagates into detached context-state leaves and frees its graph. The accumulated state gradients then flow through the context once.
- **No reuse.** Context states are recomputed per item and never reused across optimizer steps.
- **Measured** on one 3090 with a 20K item (132 queries, 25 captures), marginal + joint with backward:
  - chunk 2: 10.3 GiB, 27 s per item;
  - chunk 4: 12.7 GiB, 19 s per item;
  - the former chunk of 16 ran out of memory at 19.8K.

**Bucketed sequence lengths.** Contexts are padded with trailing newlines to a multiple of 64 and continuations to a multiple of 8. Microbatches are padded to a fixed number of rows.

**l2norm kernel tuning.** FLA's l2norm kernels tune once per feature size, not once per row count. Numerics are unchanged.

**Sensor adapter initialization.** Modality embeddings start at word-embedding scale, and projections start small but nonzero (an all-zero position would sit at the RMSNorm singularity).

## Quick closed loop

The driver is `scripts/run_loop.sh`. It trains the 2B predictor and the 0.8B selector in short alternating rounds and checks each round on DEV, so a closed loop is reached within hours rather than after a predictor trained to convergence.

1. **Warm-up `P0`.**
   - The LoRA starts from `dev_p1` step 1250 (short text contexts) with a fresh sensor adapter.
   - Training uses mixed text/capture reductions:
     - text policy: random, recency, uniform time or change point;
     - capture policy: random, recency, uniform time or geometric relevance;
     - visual share: U(0, 0.8);
     - total budget: 10K, 15K or 20K, with 20K drawn half the time (mandatory inputs are ~5.5K).
   - It runs at most `WARM_STEPS` = 400 steps. DEV evaluation runs every 100 steps (48 windows, relevance reduction at 20K), with early stopping, and `convergence.py` selects the checkpoint.
2. **Round r.**
   1. Supervision with the frozen `P(r−1)` on `SUP_WINDOWS` = 150 windows, every parent before any repeat.
   2. Selector r, warm-started from selector r−1, trained on the supervision of rounds 1..r (replay). The structured scorer is trained on the same targets.
   3. Selector choices on `EMIT_WINDOWS` = 300 training windows.
   4. `P(r)` adapted from `P(r−1)` for at most `ADAPT_STEPS` = 150 steps: half selector choices, half random reductions, DEV every 50 steps.
   5. A DEV check: `P(r)` with selector r against recency, relevance and the scorer, one window per DEV parent at 20K.
3. **Scope.** `SCOPE=dev` trains on the 171 fit parents and evaluates on the 39 DEV parents (whole profiles). `SCOPE=all` trains on all 210 parents with fixed step counts, then runs the ARM paired evaluation (relevance baseline and selector). Arms never fit parameters.

Each round both refreshes the selection supervision and adapts the predictor, a full alternation. Every run records its code commit (`args.json`, supervision records), and every stage resumes when the driver is restarted with the same `TAG`.

## Earlier development-round design (superseded)

The previous drivers (`run_dev_round.sh`, `run_final.sh`) trained the predictor to DEV convergence before any selection and are replaced by `run_loop.sh`. They are described here for the record.

**Development round.** It trained on the 171 fit parents and held out 39 DEV parents.

1. **Step 1, predictor `dev_p3`.**
   - The LoRA is warm-started from `dev_p1` step 1250 (short text contexts), with a fresh sensor adapter.
   - Training uses 20K items with mixed text/capture reductions:
     - text policy: random, recency, uniform time or change point;
     - capture policy: random, recency, uniform time or geometric relevance;
     - visual share: U(0, 0.8);
     - total budget: 6K, 12K or 20K, with 20K drawn half the time.
   - Every 250 steps a fixed DEV set is evaluated: 96 windows, deterministic query sets, captures by relevance in half of the optional capacity, text by recency, at 20K. It reports per-group NLL with window-clustered SEs.
   - Training stops early when the group-averaged DEV NLL improves by less than 1% over two evaluations.
   - `convergence.py` plots the curves and selects the checkpoint with the lowest DEV NLL.
2. **Steps 2–3.** Selection supervision (r1), the selector and the structured scorer, built from the selected checkpoint. Loss–budget evaluation on DEV covers every method at 8K and 20K.
3. **Step 4.** Predictor adaptation on the selector's choices (`dev_p2`), mixed with policy reductions, with DEV curves and checkpoint selection.
4. **Step 5.** Refreshed supervision and selector (r2). The fixed (r1) and refreshed (r2) selectors are compared on DEV.

**Fixed by development.** The development round fixes these settings:
- the step counts of the selected checkpoints;
- learning rates, budgets and supervision sizes;
- the selector's epochs and τ;
- whether the refresh helps.

The development DEV numbers are the reported held-out results.

**Final models.** Driver: `scripts/run_final.sh`.
- They are trained on all 210 parents. Step counts are the DEV-selected ones scaled by 210/171, with no checkpoint selection.
- Supervision and selector training use all parents.
- The final models are evaluated only on ARM pairs (`arms.py`, with the relevance baseline and the selector choosing the shared parent evidence). Arms never fit parameters, and no unseen-family claim is made.

## Pipeline commands

```
$PY -m Dataset.world_model.unified.build --sources --schema --predicates      # once
$PY -m Dataset.world_model.unified.sonata_cache --shard 0 --shards 1           # once (GPU)
$PY -m Dataset.world_model.unified.camera_index --workers 24                   # once
$PY -m Dataset.world_model.unified.frame_cache --workers 32                    # once, after the two above
$PY -m Dataset.world_model.unified.frame_cache --rgb-only --workers 24         # once, adds the fixed overview camera
GPUS="1 3 4" TAG=loop bash Dataset/world_model/unified/scripts/run_loop.sh     # quick closed loop on the fit parents
GPUS="1 3 4" TAG=final SCOPE=all bash Dataset/world_model/unified/scripts/run_loop.sh   # all parents, then ARM
```

## Known limitations

- **Selector reading cost.** The selector reads every candidate at inference: the full compacted text history (60K–740K tokens), in chunks of 8,192. That is roughly 10–30 s per window on one 3090. Selector training reads the supervision pool plus as many distractors, up to 48K tokens.
- **Supervision cost.** About 30 scored sets per window at ~5 s each, so ~2.5 min per window per GPU.
- **Training cost.** Predictor training is ~19 s per 20K item per GPU, so an optimizer step with `--accum 2` on three GPUs (6 items) takes ~30–40 s.
- **Fixed-camera pose.** The fixed camera's sidecar pose is taken in the truth frame (ground elevation 0). If the engine placed it in the render world, its height is overstated by the local ground elevation (~17–23 m out of 160–210 m).
- **Geometry.** Capture geometry uses a flat-ground lift around the camera. Occlusion is not modelled; the summary embedding is the only visual-content signal before selection.
- **ARM pairs.** ARM pairs build their own query sets (`arms.py`) and still require a valid parent origin state for a scoped relation.
- **ARM embeddings.** One of the 375 ARM archives (`X1_rain_to_c2loss_to_forced_landing__seed00`, `early_link_recovery`) failed embedding on a LiDAR ray-geometry check and is not embedded.
