#!/usr/bin/env bash
# Sequential driver for the overnight run. One GPU, one run at a time.
# Order matters: the two primary arms come first, so a night that runs out
# still leaves the comparison complete.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/finetune_experiments/logs

run () {
  local name="$1"; shift
  echo "=== $(date -Is) START $name"
  timeout 21600 $PY train_scripts/run_finetune_experiments.py "$@" > "$LOG/$name.log" 2>&1
  echo "=== $(date -Is) END $name rc=$?"
}

run finetune-3e-5 --arm finetune --lr 3e-5
run scratch-1e-4  --arm scratch  --lr 1e-4
run finetune-1e-4 --arm finetune --lr 1e-4
run scratch-3e-4  --arm scratch  --lr 3e-4
echo "=== $(date -Is) ALL DONE"
