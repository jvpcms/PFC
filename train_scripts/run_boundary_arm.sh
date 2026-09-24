#!/usr/bin/env bash
# One extra point: fine-tuning at lr 1e-3.
#
# Why. Across the six trained runs the fine-tune curve is still rising at the
# edge of the sweep (3e-4 is the best of all, 0.8196), so the optimum may lie
# outside the range tried and the reported best would be an artefact of where
# the sweep stopped. This run says whether 3e-4 is an interior peak. Same
# script, same protocol, same partition, only --lr changes.
#
# Takes no argument: the GPU is already free when this is launched.
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

run finetune-1e-3 --arm finetune --lr 1e-3

evaluate finetune-lr0.001 finetune 1e-3
echo "=== $(date -Is) BOUNDARY DONE"
