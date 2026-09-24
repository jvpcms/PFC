#!/usr/bin/env bash
# Third pass: extra learning-rate points, run after the first four arms and
# their evals are done. Nothing new is introduced here -- same script, same
# protocol, same partition, only --lr changes. The point is to check whether
# the small fine-tune-over-scratch margin on the first four runs survives a
# change of learning rate, or whether it is inside the noise of one split.
#
# Waits for the eval driver to exit before touching the GPU: one job at a time
# on a 4 GB card.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=results/finetune_experiments
LOG=$R/logs

EVAL_PID="${1:-}"
if [ -n "$EVAL_PID" ]; then
  echo "=== $(date -Is) waiting for eval driver pid $EVAL_PID"
  while kill -0 "$EVAL_PID" 2>/dev/null; do sleep 30; done
  echo "=== $(date -Is) eval driver gone, starting extra arms"
fi

run () {
  local name="$1"; shift
  echo "=== $(date -Is) START $name"
  timeout 21600 $PY train_scripts/run_finetune_experiments.py "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "=== $(date -Is) END $name rc=$rc"
}

# Training often dies at its own in-process eval step (see run_all_evals.sh),
# so score anything that came out without a metrics.json afterwards.
evaluate () {
  local dir="$1" arm="$2" lr="$3"
  local ckpt="$R/$dir/best.keras"
  if [ ! -f "$ckpt" ]; then
    echo "=== $(date -Is) SKIP $dir (no best.keras)"
    return
  fi
  if [ -f "$R/$dir/metrics.json" ]; then
    echo "=== $(date -Is) SKIP $dir (already scored)"
    return
  fi
  echo "=== $(date -Is) EVAL $dir"
  timeout 3600 $PY train_scripts/run_finetune_experiments.py \
    --arm "$arm" --lr "$lr" --eval-only "$ckpt" > "$LOG/eval-$dir.log" 2>&1
  local rc=$?
  echo "=== $(date -Is) EVAL $dir rc=$rc"
}

run finetune-1e-5 --arm finetune --lr 1e-5
run finetune-3e-4 --arm finetune --lr 3e-4
run scratch-3e-5  --arm scratch  --lr 3e-5

evaluate finetune-lr1e-05  finetune 1e-5
evaluate finetune-lr0.0003 finetune 3e-4
evaluate scratch-lr3e-05   scratch  3e-5
echo "=== $(date -Is) EXTRA DONE"
