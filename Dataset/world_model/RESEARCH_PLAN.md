# AeroWorld: runnable main

## Fixed route

- Qwen3.5-2B predicts future state, predicate and event probabilities from structured text and sensor embeddings.
- Qwen3.5-0.8B reads all actual candidates from the past five seconds and chooses the predictor's context.
- The predictor has a 20,000-position total budget; all modalities and all five-second candidates stay eligible.
- Future labels supply training targets only. Deployment uses the learned selector, never a future-loss oracle.
- TRAIN/DEV/TEST remain frozen. Do not train on DEV or TEST.

## Aim and goal

**Aim.** Predict probability distributions over the future world state (positions, link and compute metrics,
predicates, events) from the last five seconds of multimodal evidence, with a fixed 2B predictor input of at most
20,000 positions. The 0.8B selector decides which evidence enters that input.

**Goal (the only success criterion).** With the predictor held fixed, the selector's own context gives lower
held-out NLL than both simple policies, recency and relevance. The comparison is paired: same windows, queries,
budget and predictor. A gain counts only if the 95% window-cluster bootstrap interval of selector − anchor lies
below zero.

**Correctness rules.**
- Future labels enter only the measured losses the critic is fit to. Deployment is label-free.
- Configuration choices use TRAIN only (cross-validation holding out whole scenario profiles, as DEV does). The
  configuration is frozen before DEV; DEV and TEST are each scored once for it.

## Selector: measured-value ranking (2026-10-14)

1. `collect-values`: for TRAIN windows, the frozen 2B scores recency, relevance and M random complete contexts
   (greedy fills under the exact budget on random priorities over record metadata plus Gumbel noise; a quarter fill
   50-100% of the budget). The 0.8B record states are cached at the same time.
2. `fit-selector`: the 0.8B reads the query prefix and every candidate record once; a small head on each record's
   final state and metadata gives per-head diminishing-returns values, F(S) = sum_h a_h log(1 + sum_{k in S,h} b_k)
   + sum_{k in S} c_k. It is fit by within-window centred least squares to the measured losses.
3. Deployment ranks recency, relevance and K label-free contexts from the same family by the predicted F (mean over
   an ensemble) and keeps the best: the selector leaves an anchor only where its critic predicts a gain.

Optimizing F directly (greedy fills) and the earlier local-edit RL (KEEP/EDIT gate over bundles of ≤4 record swaps;
objectives teacher/pg/calibrated) never beat the anchors and were removed. Details, intermediate evidence and the
frozen configuration: `runs/unified/main/value/NOTES.md`.

**Result (frozen 2B `predictor.pt`; 0.8B critic, K=20, ensemble 5; fit on 1,080 TRAIN windows).**

| Split | Selector | Recency | Relevance | Selector − recency [95% CI] | Selector − relevance [95% CI] |
|---|---|---|---|---|---|
| DEV (163 jobs) | 1.2972 | 1.3207 | 1.3247 | −0.0235 [−0.0515, −0.0039] | −0.0275 [−0.0561, −0.0061] |
| TEST (168 jobs, once) | 1.5819 | 1.5849 | 1.5925 | −0.0030 [−0.0173, +0.0120] | −0.0106 [−0.0242, +0.0021] |

The selector gains on parent windows (latency, global and scheduled events) and loses on ARM windows, where it
sometimes chooses capture-heavy contexts. TEST has been used for this configuration.

## One directory, one entry point

Use the repository root and the pinned runtime. Models and outputs live in `Dataset/world_model/runs/unified/main/`.
Original data, base models and pass-0 checkpoints remain untouched.

```bash
PY=Dataset/world_model/runtimes/qwen35/bin/python
M=Dataset/world_model/runs/unified/main; V=$M/value
export PYTHONPATH=. OMP_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false

# Measured contexts for TRAIN jobs (one process per GPU shard; resumable).
$PY -m Dataset.world_model.unified.value jobs --windows 720 --seed 11 --output $V/jobs_r1.json
CUDA_VISIBLE_DEVICES=1 $PY -m Dataset.world_model.unified.run collect-values --jobs $V/jobs_r1.json --out $V/r1 \
  --predictor $M/predictor.pt --backbone $M/selector_init.pt --samples 12 --shard 0 --shards 1

# Critic fit with profile-held-out cross-validation, then the deployable ensemble.
CUDA_VISIBLE_DEVICES=5 $PY -m Dataset.world_model.unified.run fit-selector --samples "$V/r1/samples.*.jsonl" \
  --features $V/r1/meta --states $V/r1/feat --backbone $M/selector_init.pt --epochs 12 --lr 1e-3 \
  --ensemble 5 --proposals 20 --out $V/final/selector.pt

# Final fit, DEV and TEST (paired against recency and relevance).
bash $V/run_final.sh 20 5

# Direct probability prediction with the selector.
CUDA_VISIBLE_DEVICES=5 $PY -m Dataset.world_model.unified.run predict --selector $V/final/selector.pt \
  --predictor $M/predictor.pt --episode L1-3_v1__seed00 --origin 100 --samples 2 --output $M/prediction.json
```
