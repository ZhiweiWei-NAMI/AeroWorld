#!/bin/bash
# One-line status of the unified pipeline runs.
U=/mnt/data2/weizhiwei/AERO_WORLD/Dataset/world_model/runs/unified
for r in $(ls -t $U/predictor 2>/dev/null | head -2); do
  echo "$r: $(grep '"loss"' $U/predictor/$r/log.jsonl | tail -1 | cut -c1-40)"
done
[ -f $U/logs/dev_round.log ] && echo "round: $(grep stage $U/logs/dev_round.log | tail -1)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
