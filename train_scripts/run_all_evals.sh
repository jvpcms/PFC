#!/usr/bin/env bash
# Second pass: score each trained checkpoint in a FRESH process.
#
# Why this exists. Evaluating inside the process that just trained exhausts the
# 4 GB card: the Adam slots and the training graph are still resident, so cuDNN
# has no workspace left to profile a conv algorithm into and every candidate
# fails with RESOURCE_EXHAUSTED. Training itself is fine; only the eval step
# dies, after `best.keras` is already on disk. So the fix is to train first and
# score the checkpoint afterwards, with nothing else on the card.
#
# Waits for the training driver to exit before touching the GPU.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=results/finetune_experiments
LOG=$R/logs

DRIVER_PID="${1:-}"
if [ -n "$DRIVER_PID" ]; then
  echo "=== $(date -Is) waiting for training driver pid $DRIVER_PID"
  while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 30; done
  echo "=== $(date -Is) training driver gone, starting evals"
fi

evaluate () {
  local dir="$1" arm="$2" lr="$3"
  local ckpt="$R/$dir/best.keras"
  if [ ! -f "$ckpt" ]; then
    echo "=== $(date -Is) SKIP $dir (no best.keras)"
    return
  fi
  echo "=== $(date -Is) EVAL $dir"
  timeout 3600 $PY train_scripts/run_finetune_experiments.py \
    --arm "$arm" --lr "$lr" --eval-only "$ckpt" > "$LOG/eval-$dir.log" 2>&1
  local rc=$?
  echo "=== $(date -Is) EVAL $dir rc=$rc"
}

evaluate finetune-lr3e-05  finetune 3e-5
evaluate scratch-lr0.0001  scratch  1e-4
evaluate finetune-lr0.0001 finetune 1e-4
evaluate scratch-lr0.0003  scratch  3e-4
echo "=== $(date -Is) EVALS DONE"
