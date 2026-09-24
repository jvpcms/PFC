#!/usr/bin/env bash
# Half the annotation budget: the same two trainable arms on 40 train tiles.
#
# Question. At 80 tiles the fine-tuned arm beats the from-scratch arm at every
# learning rate, but only by 1.6 points at best. Pre-training should matter more
# when there is less data to learn from, so this halves the TRAIN side and
# measures whether the separation widens.
#
# The 40 are the first by `train_rank`, which is the prefix-stable pick order
# from sample_tiles.py, so they are exactly the tiles a 50-image annotation
# budget would have produced. Note they are NOT a uniform half: the allocator
# front-loads rare strata for coverage, so this subset is more class-balanced
# than a random 40 would be.
#
# The TEST split stays at 20 tiles. It is the measuring instrument, not data,
# and holding it fixed is what makes these runs subtractable from the 80-tile
# ones. Halving it too would change training size and measurement precision at
# once and make any widening of the gap unattributable.
#
# The zero-shot floor is not re-run: it never touches training data, so the
# existing `zeroshot` result applies unchanged to both budgets.
#
# Pairs are ordered so that an interrupted night still leaves a complete
# fine-tune vs scratch comparison at the learning rate that was best for each.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=results/finetune_experiments
LOG=$R/logs
N=40

run () {
  local name="$1"; shift
  echo "=== $(date -Is) START $name"
  timeout 21600 $PY train_scripts/run_finetune_experiments.py --train-subset $N "$@" \
    > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "=== $(date -Is) END $name rc=$rc"
}

# Training often dies at its own in-process eval step on the 4 GB card (see
# run_all_evals.sh), so score anything left without a metrics.json.
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
  timeout 3600 $PY train_scripts/run_finetune_experiments.py --train-subset $N \
    --arm "$arm" --lr "$lr" --eval-only "$ckpt" > "$LOG/eval-$dir.log" 2>&1
  local rc=$?
  echo "=== $(date -Is) EVAL $dir rc=$rc"
}

run half-finetune-1e-4 --arm finetune --lr 1e-4
run half-scratch-1e-4  --arm scratch  --lr 1e-4
run half-finetune-3e-4 --arm finetune --lr 3e-4
run half-scratch-3e-4  --arm scratch  --lr 3e-4
run half-finetune-3e-5 --arm finetune --lr 3e-5
run half-scratch-3e-5  --arm scratch  --lr 3e-5

evaluate finetune-lr0.0001-tr40 finetune 1e-4
evaluate scratch-lr0.0001-tr40  scratch  1e-4
evaluate finetune-lr0.0003-tr40 finetune 3e-4
evaluate scratch-lr0.0003-tr40  scratch  3e-4
evaluate finetune-lr3e-05-tr40  finetune 3e-5
evaluate scratch-lr3e-05-tr40   scratch  3e-5
echo "=== $(date -Is) HALF DONE"
