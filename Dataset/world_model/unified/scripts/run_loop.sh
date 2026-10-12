#!/bin/bash
# Quick closed loop of the predictor/selector alternation (RESEARCH_PLAN §9) at the 20K total budget.
#
#   warm-up  predictor P0 from the dev_p1 LoRA (fresh sensor adapter), early-stopped on a small fixed DEV set
#   round r  1. supervision with the frozen P(r-1): candidate text/capture sets scored by its grouped losses
#            2. selector r, warm-started from selector r-1, on the supervision of rounds 1..r (replay)
#            3. selector choices on training windows
#            4. predictor P(r), adapted from P(r-1) on the choices mixed with random reductions
#            5. DEV: P(r) with selector r vs recency and relevance (and the structured scorer)
#
# SCOPE=dev trains on the fit parents and evaluates on the DEV parents; SCOPE=all trains on all 210 parents
# (no DEV evaluation, fixed step counts).  VISUAL=fixed runs a text-only stage (capture packet held fixed).
# Every stage resumes from its outputs when the script is restarted with the same TAG.
set -euo pipefail
shopt -s inherit_errexit
cd /mnt/data2/weizhiwei/AERO_WORLD
export PYTHONPATH=. PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PY=Dataset/world_model/runtimes/qwen35/bin/python
TR=Dataset/world_model/runtimes/qwen35/bin/torchrun
U=Dataset/world_model/runs/unified
TAG=${TAG:-loop}
SCOPE=${SCOPE:-dev}
VISUAL=${VISUAL:-learned}
ROUNDS=${ROUNDS:-3}
GPUS=(${GPUS:-1 3 4})
N=${#GPUS[@]}
WARM_STEPS=${WARM_STEPS:-400}
ADAPT_STEPS=${ADAPT_STEPS:-150}
SUP_WINDOWS=${SUP_WINDOWS:-150}
EMIT_WINDOWS=${EMIT_WINDOWS:-300}
SEL_EPOCHS=${SEL_EPOCHS:-2}
EVAL_WINDOWS=${EVAL_WINDOWS:-48}
INIT=${INIT:-$U/predictor/dev_p1/step_001250.pt}
LOG=$U/logs/$TAG.log
mkdir -p $U/logs
say() { echo "{\"time\": \"$(date -Iseconds)\", \"stage\": \"$1\"}" | tee -a $LOG; }
devices() { local IFS=,; echo "${GPUS[*]}"; }
par() {  # a command template on every GPU with --shard i --shards N
  local pids=() i
  for ((i = 0; i < N; i++)); do CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$@" --shard $i --shards $N >> $LOG 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
}
if [ "$SCOPE" = all ]; then SCOPE_ARGS=(--all-parents); EVAL_ARGS=(); else SCOPE_ARGS=(); EVAL_ARGS=(--eval-windows $EVAL_WINDOWS); fi
train() {  # name init steps extra...; sets P to the stage's checkpoint (DEV-selected under SCOPE=dev)
  local name=$1 init=$2 steps=$3; shift 3
  local dir=$U/predictor/$name
  if [ ! -f $dir/done ]; then
    rm -rf $dir                     # an interrupted attempt restarts clean: no mixed logs or overwritten checkpoints
    CUDA_VISIBLE_DEVICES=$(devices) $TR --nproc_per_node $N --master_port $((29500 + RANDOM % 400)) \
        -m Dataset.world_model.unified.train_predictor --name $name --init $init --steps $steps --accum 2 \
        --save-every 50 --eval-every 50 "${EVAL_ARGS[@]}" "${SCOPE_ARGS[@]}" "$@" >> $LOG 2>&1
    touch $dir/done
  fi
  if [ "$SCOPE" = dev ]; then
    $PY -m Dataset.world_model.unified.convergence --runs $name:0 --name $name >> $LOG 2>&1
    P=$($PY -c "import json; print(json.load(open('$dir/selection.json'))['selected_checkpoint'])")
  else
    P=$(ls $dir/step_*.pt | sort | tail -1)
  fi
  test -f "$P"
}

say "warm-up predictor ${TAG}_p0 ($WARM_STEPS steps max, scope $SCOPE)"
train ${TAG}_p0 $INIT $WARM_STEPS --init-text-only --warmup 30 --eval-every 100 --patience 3 --tolerance 0.005
say "P0 = $P"
SEL=""
SUPS=()
for ((r = 1; r <= ROUNDS; r++)); do
  say "round $r: supervision with $P"
  par $PY -m Dataset.world_model.unified.supervise --checkpoint $P --out ${TAG}_sup_r$r --windows $SUP_WINDOWS \
      --visual $VISUAL --seed $r "${SCOPE_ARGS[@]}"
  SUPS+=(${TAG}_sup_r$r)
  say "round $r: selector (init ${SEL:-fresh}, supervision ${SUPS[*]})"
  SPID=""
  if [ ! -f $U/selector/${TAG}_sel_r$r/epoch_$((SEL_EPOCHS - 1)).pt ]; then
    rm -rf $U/selector/${TAG}_sel_r$r
    HOLDOUT=(); [ "$SCOPE" = all ] && HOLDOUT=(--holdout 0)
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY -m Dataset.world_model.unified.train_selector --supervision ${SUPS[@]} \
        --name ${TAG}_sel_r$r --epochs $SEL_EPOCHS ${SEL:+--init $SEL} "${HOLDOUT[@]}" >> $LOG 2>&1 &
    SPID=$!
  fi
  [ -f $U/selector/${TAG}_scorer_r$r/scorer.pt ] || \
    $PY -m Dataset.world_model.unified.scorer --supervision ${SUPS[@]} --name ${TAG}_scorer_r$r >> $LOG 2>&1
  [ -z "$SPID" ] || wait $SPID
  SEL=$U/selector/${TAG}_sel_r$r/epoch_$((SEL_EPOCHS - 1)).pt
  say "round $r: selector choices"
  CHOICES=$U/selector/${TAG}_sel_r$r/choices.jsonl
  [ -f $CHOICES.done ] || { rm -f $CHOICES $CHOICES.[0-9][0-9]; par $PY -m Dataset.world_model.unified.selector --selector $SEL --out $CHOICES \
      --windows $EMIT_WINDOWS --visual $VISUAL --seed $r "${SCOPE_ARGS[@]}"; touch $CHOICES.done; }
  say "round $r: predictor adaptation from $P"
  train ${TAG}_p$r $P $ADAPT_STEPS --warmup 10 --lr 5e-5 --patience 2 --selector-items $CHOICES
  say "round $r: P$r = $P"
  if [ "$SCOPE" = dev ]; then
    par $PY -m Dataset.world_model.unified.evaluate --checkpoint $P --name ${TAG}_eval_r$r --per-episode 1 \
        --budgets 20000 --methods recency relevance --selector $SEL --scorer $U/selector/${TAG}_scorer_r$r/scorer.pt \
        --visual $VISUAL
    $PY -m Dataset.world_model.unified.evaluate --checkpoint none --name ${TAG}_eval_r$r --report >> $LOG 2>&1
  fi
done
echo "{\"predictor\": \"$P\", \"selector\": \"$SEL\", \"rounds\": $ROUNDS, \"scope\": \"$SCOPE\"}" > $U/${TAG}_models.json
if [ "$SCOPE" = all ]; then
  say "ARM paired evaluation"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY -m Dataset.world_model.unified.arms --checkpoint $P --name ${TAG}_relevance \
      --policy relevance >> $LOG 2>&1
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY -m Dataset.world_model.unified.arms --checkpoint $P --name ${TAG}_selector \
      --selector $SEL --visual $VISUAL >> $LOG 2>&1
fi
say "done"
