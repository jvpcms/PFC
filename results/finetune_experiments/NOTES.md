# Overnight fine-tuning run — notes for João

Started 2026-09-22 ~22:23 local. Everything here is for you to read in the morning.
Nothing in this file needs an answer before the runs finish.

## What is running

`train_scripts/run_all_arms.sh` (driver, backgrounded), calling
`train_scripts/run_finetune_experiments.py` once per arm, **sequentially** — never two
at once, the card is 4 GB.

Order, deliberately: the two arms the report needs come first, so a night that runs out
still leaves the three-way comparison complete.

1. `finetune --lr 3e-5`  (arm C, the notebook's default)
2. `scratch  --lr 1e-4`  (arm B)
3. `finetune --lr 1e-4`  (arm C, LR variant)
4. `scratch  --lr 3e-4`  (arm B, LR variant)

Arm A (`zeroshot`) already ran, in the foreground, before the driver started.

Logs: `results/finetune_experiments/logs/<name>.log`, driver timeline in `driver.log`.
Results: `results/finetune_experiments/<name>/metrics.json` plus `history.csv` and
`best.keras`.

## Arm A result — the new floor

The DeepGlobe checkpoint, zero-shot on the 20-tile test split, 5 % Unknown cutoff,
Unknown GT pixels excluded:

| | px acc | mIoU (5 present classes) |
|---|---|---|
| **5 % cutoff** (the protocol) | **0,4213** | **0,2369** |
| 25 % cutoff | 0,4437 | 0,2454 |

Per class at 5 %: Urban 0,3622 · Agriculture_Rangeland 0,3692 · Forest 0,2151 ·
Water 0,1663 · Barren 0,0716.

**This supersedes the 0,373 / 0,239 in `notes/inpe_zero_shot_baseline.md` and in the
notebook header**, which were measured on 10 test tiles. The test split is 20 tiles now.
Use 0,4213 / 0,2369 in the report, not the old pair.

Water is still the worst real class, consistent with the earlier ablation. Barren at
0,0716 is now the floor overall — worth a look at the confusion matrix in
`zeroshot/metrics.json` before writing the results section.

## Decisions taken without you, and why

**The 5 % Unknown cutoff is evaluation-only.** You thought it also filtered training. It
does not, anywhere in the codebase — verified in `Segmentation_DeepGlobe_UNet.ipynb`
cell 4 (tile exporter writes all 4 corners unconditionally), `train_deepglobe_unet.py`
(no filtering at all), and `Finetune_INPE_UNet.ipynb` cell 4 (`load_split` appends every
crop; the threshold appears only in cell 12, on `y_val`). The origin is
`Evaluate_INPE_TestCVAT.ipynb`, which is a zero-fine-tuning **evaluation** notebook.
You accepted the recommendation to leave training unfiltered, which keeps arms B and C
comparable to the arm A checkpoint. Filtering would have dropped 16 of 1280 train
sub-tiles, 1,25 %.

**`MASK_DIR` was repointed** in `train_scripts/eval_inpe_tiles.py`, from
`half_labels/SegmentationClass` to `full_labels/SegmentationClass`. `full_labels/` is the
finished 100-tile CVAT export, unzipped from `~/Downloads/finished_annotations.zip`.
`half_labels/` is untouched. **Anything else that imported `MASK_DIR` now sees 100 tiles
instead of 50** — that includes `eval_inpe_tiles.py`'s own CLI and
`export_finetune_split.py`. Intended, but be aware if you re-run an old notebook.

**The runner is a script, not the notebook.** The notebook needs a live kernel and calls
`wandb.init`, neither of which survives an unattended night. The runner is the notebook's
logic with wandb removed and results written to JSON. Recipe, constants, augmentation,
callbacks and protocol are unchanged. No wandb run will appear for these.

## Timing, measured

**~25 s/epoch** at batch 8 on the 3050 (160 steps of 1280 sub-tiles). Epoch 1 takes
several minutes because of graph tracing; ignore it. A full 80-epoch run is ~35 min, so
all four arms land in roughly 2,5 h, well inside the night. The 6 h `timeout` per run is
slack, not a budget.

If the four finish early I will queue further LR points and append them here.

## Early signal (arm C, lr 3e-5, epoch 9 of up to 80)

Keras `val_miou` 0,6533 against a zero-shot protocol mIoU of 0,2369. Per-class validation
IoU: Water 0,958 · Urban 0,885 · Agriculture_Rangeland 0,864 · Forest 0,844 ·
**Barren 0,369** · Unknown 0,000.

Two readings, neither final:

- Water went from the second-worst zero-shot class to the best. That is consistent with
  the earlier ablation, which attributed most of the zero-shot Water failure to rendering
  rather than to domain shift — fine-tuning on the rendered tiles absorbs exactly that.
- **Barren stays low in both**, 0,072 zero-shot and 0,369 here. It is the one class the
  annotation does not seem to fix, and it is only 3,78 % of pixels (1,76 % of the test
  split). Worth a paragraph in the results section rather than a footnote.

`val_iou_unknown` is 0,000 and will stay there: Unknown is 0,28 % of train pixels. This
is why Keras `val_miou` sits at roughly five-sixths of the mean over present classes.

## A failure found and worked around at 22:45 — read this first

**Training is fine. The in-process evaluation step OOMs on the 4 GB card.**

Arm C (lr 3e-5) trained to completion: early stopping at epoch 40, best epoch 25,
weights restored, `best.keras` written. Then `predict_labels` died:

```
NotFoundError: ... Conv2D ... No algorithm worked!
  Profiling failure on CUDNN engine eng1{}: RESOURCE_EXHAUSTED:
  Out of memory while trying to allocate 80.01MiB.
```

Cause: after `fit`, the Adam slot variables and the training graph are still resident on
the card, so cuDNN has no workspace left to profile a convolution algorithm into and
every candidate engine fails. It is not the batch size and not the data volume — the
inference chunk is 8 sub-tiles.

`metrics.json` is written after the prediction, so arm C has `best.keras` and
`history.csv` but **no `metrics.json`** from that first pass.

**The workaround, already running.** `run_finetune_experiments.py` gained
`--eval-only CHECKPOINT` and `--eval-batch-size` (default 4). A second driver,
`train_scripts/run_all_evals.sh` (pid 855890), is waiting on the training driver and
will then score every `best.keras` in a fresh process with nothing else on the card,
writing the `protocol` block into each `metrics.json` and preserving the training fields
where they already exist.

So the night proceeds as: all four arms train (each will still crash at its own eval
step, harmlessly, after saving `best.keras`), then all four are scored. Logs for the
second pass are `logs/eval-<dir>.log` and `logs/evals.log`.

Nothing was killed and nothing is lost.

**A second, cosmetic bug: the driver's `rc=` is always 0.** In
`train_scripts/run_all_arms.sh`, `echo "=== $(date -Is) END $name rc=$?"` expands
`$(date -Is)` first, which resets `$?` to date's exit status. So `driver.log` reports
`rc=0` for the arm C run that in fact died. **Do not trust `rc=` in `driver.log`** —
check for `metrics.json` instead. I did not fix it in place because bash reads a script
incrementally and editing it mid-run would corrupt the running driver.
`run_all_evals.sh` captures the status correctly.

## Arm C (lr 3e-5) — training outcome

Early stopped at epoch 40, best epoch 25, 40 epochs in ~20 min. Final validation figures
from `history.csv`, Keras metrics (Unknown scored, all 320 sub-tiles), not the protocol
numbers:

`val_miou` ~0,65 · Water ~0,96 · Urban ~0,89 · Agriculture_Rangeland ~0,86 ·
Forest ~0,84 · Barren ~0,37 · Unknown 0,000.

The protocol numbers comparable to the 0,2369 floor come from the second pass.

## Things to check in the morning

- **Did all four finish?** `grep START\|END\|ALL results/finetune_experiments/logs/driver.log`.
  Each run is capped at 6 h by `timeout`; a run that hits the cap exits non-zero and the
  driver moves on. If only runs 1 and 2 finished, that is still a complete experiment.
- **`val_miou` vs `protocol_miou_unk05` are different numbers.** `keras_val_miou` in
  `metrics.json` scores Unknown as a class and keeps all 320 sub-tiles. The reportable
  number is `protocol.unk05.miou_present`. Do not read one against the other, and do not
  compare either to DeepGlobe's 0,813.
- **Every number is a validation score, not held-out.** `test` gates checkpointing and
  early stopping in arms B and C. That is true of all three arms equally so the
  comparison is fair, but the report has to say so.
- **Unknown IoU will read `absent`** in the protocol tables. That is correct: Unknown GT
  pixels are excluded from scoring, so the class has no rows in the confusion matrix.
- If a run crashed, the traceback is at the end of its `.log`. The driver does not stop
  on failure, so a later run may have succeeded where an earlier one died.

## Not done, waiting on you

- The report placeholders in `cap5/section2.tex` are still `--`. I did not fill them,
  because which LR variant is the one to report is a call worth making with the numbers
  in front of you.
- `notes/inpe_zero_shot_baseline.md` still quotes the 50-tile figures and should be
  updated to the 20-tile test numbers above.
- `cap4/section2.tex` §Estado de execução still says the annotation campaign is "em
  curso". It is finished.

---

## Run-by-run training log (appended as each arm finished)

Every training arm behaves the same way: it trains to early stopping, restores
the best weights, writes `best.keras` and `history.csv`, then **dies at the
in-process evaluation step with the cuDNN/BFC OOM described above**. That is
expected and is not a training failure. `metrics.json` is produced afterwards by
`run_all_evals.sh` in a fresh process. Do not read `rc=` in `driver.log`.

The Keras `val_*` figures below are **not** the protocol metric. Keras `MeanIoU`
averages over all six classes including `Unknown`, whose IoU is structurally
0.0000 because no arm ever predicts it, so Keras `val_miou` is roughly 5/6 of a
five-class mean. The protocol metric excludes `Unknown` and drops sub-tiles above
the 5% threshold. Compare arms only through `metrics.json`.

| arm | lr | epochs run | best epoch | val_miou (Keras) | val_acc |
|---|---|---|---|---|---|
| `finetune-lr3e-05` | 3e-5 | 40 | 25 | 0.6584 | 0.9182 |
| `scratch-lr0.0001` | 1e-4 | 34 | 19 | 0.6547 | 0.9135 |

Per-class Keras val IoU at the restored best epoch:

| arm | Water | Urban | Forest | Agri_Rang | Barren | Unknown |
|---|---|---|---|---|---|---|
| `finetune-lr3e-05` | 0.9582 | 0.8948 | 0.8598 | 0.8700 | 0.3674 | 0.0000 |
| `scratch-lr0.0001` | 0.9419 | 0.8941 | 0.8609 | 0.8612 | 0.3701 | 0.0000 |

**Thing worth knowing before reading the final table.** On this Keras metric the
from-scratch arm is level with the fine-tuned arm (0.6547 vs 0.6584, a 0.004
gap on a single 20-tile split). If the protocol metric agrees, that is the
"DeepGlobe pre-training is dispensable" branch that `cap4/section2.tex`
§Desenho experimental already names as a possible outcome, and it would be a
real result rather than a problem. It is **not confirmed** here: these are the
wrong metric, taken on validation rather than under the protocol, and the split
is one partition with no cross-validation. Wait for `metrics.json`.

Also note **Barren is the weak class in every arm** (0.36-0.37), matching the
zero-shot finding that Barren, not Water, is now the floor. Water, which was the
zero-shot failure case at IoU 0.166, is the *best* class after training
(0.94-0.96), so the rendering problem documented in
`notes/inpe_zero_shot_baseline.md` is something training absorbs.

---

## RESULTS -- four arms, protocol metric, 5% Unknown threshold

All five rows are the same test partition (20 tiles, 320 sub-tiles, 286 kept at
5%), the same protocol, `Unknown` excluded from scoring. Source: each arm's
`metrics.json`. Three arms scored in-process, `finetune-lr3e-05` and
`scratch-lr0.0001` scored afterwards by `run_all_evals.sh`.

| run | arm | lr | px acc | mIoU | Urban | Agri_Rang | Forest | Water | Barren |
|---|---|---|---|---|---|---|---|---|---|
| `zeroshot` | transferência direta | -- | 0.4213 | 0.2369 | 0.3622 | 0.3692 | 0.2151 | 0.1663 | 0.0716 |
| `finetune-lr3e-05` | ajuste | 3e-5 | 0.9466 | **0.8117** | 0.8949 | 0.9051 | 0.8643 | 0.9590 | 0.4349 |
| `finetune-lr0.0001` | ajuste | 1e-4 | **0.9469** | 0.8115 | 0.8938 | 0.9073 | 0.8655 | 0.9613 | 0.4296 |
| `scratch-lr0.0001` | do zero | 1e-4 | 0.9414 | 0.8035 | 0.8941 | 0.8950 | 0.8644 | 0.9477 | 0.4162 |
| `scratch-lr0.0003` | do zero | 3e-4 | 0.9340 | 0.7865 | 0.8832 | 0.8861 | 0.8370 | 0.9349 | 0.3912 |

### What this says

1. **Annotation pays for itself, overwhelmingly.** Any trained arm beats direct
   transfer by roughly +0.57 mIoU and +0.52 px acc. The zero-shot floor is not
   close to competitive, so the campaign is justified on its own.
2. **Pre-training pays, but only a little.** Best fine-tune 0.8117 vs best
   scratch 0.8035, a gap of **0.0082 mIoU** and 0.0055 px acc. The direction is
   consistent across both learning rates tried, and fine-tuning also converged
   in fewer epochs, but the margin is small next to the variance of a single
   20-tile split with no cross-validation. Do not state it as a strong result
   without qualification.
3. **Fine-tuning is insensitive to learning rate here** (3e-5 and 1e-4 land
   0.0002 apart), while from-scratch is not (1e-4 beats 3e-4 by 0.017). That
   asymmetry is itself mild evidence for the pre-trained initialisation, which
   is doing the job a careful learning rate has to do otherwise.
4. **Barren is the bottleneck in every arm** (0.39-0.43 against 0.86-0.96 for
   everything else) and it was also the worst zero-shot class (0.0716). Whatever
   limits Barren is not fixed by training on this set. The likely cause is that
   Barren is ~1.9% of scored pixels; that is a hypothesis, not a measurement.
5. **Water is fully recovered.** Worst zero-shot class at 0.1663, best class in
   every trained arm at 0.93-0.96. The rendering problems documented in
   `notes/inpe_zero_shot_baseline.md` (clamped gamma, per-scene endpoints on
   flat ocean) do not prevent a model trained on that rendering from learning
   water. They degrade transfer from a differently-rendered source, not the
   domain itself.

### Which arm to report

Not decided here, deliberately. `finetune-lr3e-05` and `finetune-lr0.0001` are a
statistical tie, so the choice is editorial rather than empirical. The report
placeholders in `PFC_Report/cap5/section2.tex` (`tab:arms_overview`,
`tab:arms_por_classe`) are **left as `--`** for that reason.

### Extra learning-rate points, queued 23:31

`train_scripts/run_extra_arms.sh` (pid 896099) waits for the eval driver, then
runs `finetune --lr 1e-5`, `finetune --lr 3e-4`, `scratch --lr 3e-5`, and scores
anything whose in-process eval died. Purpose is only to check whether the
0.0082 fine-tune margin holds across learning rate. Same script, same protocol,
same partition -- nothing new is introduced. Results land in the same directory
layout and can be appended to the table above. If a run is missing in the
morning, check `results/finetune_experiments/logs/extra.log`.

**Not done, and worth knowing:** there is no seed-repetition run, so the margin
in point 2 has no error bar. Repeating an arm under a different seed would need
a `--seed` flag that the script does not have, and adding one would have been a
new feature.

---

## FULL SWEEP -- seven trained runs plus the floor (protocol, 5% threshold)

`run_extra_arms.sh` finished at 00:44:42 and all three of its evals ran
in-process without the OOM, so the crash is intermittent (allocator
fragmentation) rather than deterministic. Every row below comes from a
`metrics.json`, same partition, same protocol.

| run | arm | lr | px acc | mIoU | Urban | Agri_Rang | Forest | Water | Barren |
|---|---|---|---|---|---|---|---|---|---|
| `zeroshot` | direta | -- | 0.4213 | 0.2369 | 0.3622 | 0.3692 | 0.2151 | 0.1663 | 0.0716 |
| `finetune-lr1e-05` | ajuste | 1e-5 | 0.9465 | 0.8082 | 0.8934 | 0.9055 | 0.8618 | 0.9626 | 0.4175 |
| `finetune-lr3e-05` | ajuste | 3e-5 | 0.9466 | 0.8117 | 0.8949 | 0.9051 | 0.8643 | 0.9590 | 0.4349 |
| `finetune-lr0.0001` | ajuste | 1e-4 | 0.9469 | 0.8115 | 0.8938 | 0.9073 | 0.8655 | 0.9613 | 0.4296 |
| `finetune-lr0.0003` | ajuste | 3e-4 | **0.9480** | **0.8196** | 0.9023 | 0.9070 | 0.8440 | 0.9671 | **0.4777** |
| `scratch-lr3e-05` | do zero | 3e-5 | 0.9338 | 0.7838 | 0.8881 | 0.8853 | 0.8496 | 0.9559 | 0.3399 |
| `scratch-lr0.0001` | do zero | 1e-4 | 0.9414 | 0.8035 | 0.8941 | 0.8950 | 0.8644 | 0.9477 | 0.4162 |
| `scratch-lr0.0003` | do zero | 3e-4 | 0.9340 | 0.7865 | 0.8832 | 0.8861 | 0.8370 | 0.9349 | 0.3912 |

### Revised reading (this supersedes point 2 of the previous section)

The earlier four-run table put the fine-tune advantage at 0.0082 mIoU and I said
it was small enough to be inside one-split noise. With four fine-tune points and
three scratch points the picture is firmer:

- **Every fine-tune run beats every scratch run.** Worst fine-tune 0.8082,
  best scratch 0.8035. Seven runs, no overlap between the two groups. That is a
  much stronger statement than the single 0.0082 comparison, because it no
  longer depends on which learning rate each arm happened to draw.
- **Best against best is now 0.0161 mIoU** (0.8196 vs 0.8035), double the
  earlier figure.
- **The fine-tune arm is flat in learning rate, the scratch arm is peaked.**
  Fine-tune spans 0.8082-0.8196 across a 30x range of learning rate, a spread of
  0.0114. Scratch spans 0.7838-0.8035 across a 10x range, a spread of 0.0197,
  with an interior optimum at 1e-4 and falling off on both sides. Pre-training
  buys robustness to the optimiser setting, not only a better endpoint.

### Caveats that have not gone away

Still one fixed 80/20 partition, still no cross-validation, still no seed
repetition, so none of these numbers carries an error bar and the ordering
within the fine-tune group (0.8082 to 0.8196) should not be read as meaningful.
The group separation is the result, not the ranking inside a group.

### Barren drives the spread

Barren ranges 0.3399-0.4777 across the seven runs while every other class moves
by at most ~0.03. Since mIoU is a five-class mean, Barren alone accounts for
most of the run-to-run variation, including the 3e-4 fine-tune's lead -- it wins
on Barren (0.4777) while posting the *worst* Forest of any fine-tune (0.8440).
Read the headline mIoU with that in mind: it is largely a Barren measurement.

### Boundary check, queued 00:45

The fine-tune curve is still rising at the top of the sweep, so 3e-4 may be a
boundary artefact rather than an optimum. `train_scripts/run_boundary_arm.sh`
(pid 951974) runs `finetune --lr 1e-3` and scores it. If 1e-3 is worse, 3e-4 is
an interior peak and the sweep is closed. If 1e-3 is better, the sweep stopped
too early and the fine-tune numbers above are a lower bound. Log:
`results/finetune_experiments/logs/boundary.log`.

---

## Boundary check result -- sweep closed 01:03:56

| run | arm | lr | px acc | mIoU | Urban | Agri_Rang | Forest | Water | Barren |
|---|---|---|---|---|---|---|---|---|---|
| `finetune-lr0.001` | ajuste | 1e-3 | 0.9419 | 0.8161 | 0.8921 | 0.8975 | 0.8584 | 0.9340 | 0.4985 |

**3e-4 is an interior peak.** mIoU at 1e-3 is 0.8161 against 0.8196 at 3e-4, and
px acc falls further (0.9419 vs 0.9480). The fine-tune curve rises from 1e-5 to
3e-4 and turns over by 1e-3, so the sweep is closed and `finetune-lr0.0003`
stands as the best configuration found. It is not a boundary artefact.

Nothing else changes. All five fine-tune runs (0.8082 to 0.8196) still sit above
all three scratch runs (0.7838 to 0.8035), now across a 100x range of learning
rate, which is the strongest form of the pre-training result this experiment can
give without cross-validation or seed repetition.

Barren continues to behave differently from every other class: it is *highest*
at 1e-3 (0.4985, the best Barren of any run) on a run whose overall mIoU is
lower. Barren and the rest of the taxonomy do not optimise together, which is
another reason not to over-read small mIoU differences inside the fine-tune
group.

## State at the end of the night

Eight trained runs plus the zero-shot floor, all scored, all with `metrics.json`
in `results/finetune_experiments/<run>/`. No GPU job is left running. Nothing
crashed unrecovered. The only failures were the intermittent in-process eval
OOMs, every one of which was re-scored successfully from `best.keras`.

Drivers used, in order: `run_all_arms.sh` -> `run_all_evals.sh` ->
`run_extra_arms.sh` -> `run_boundary_arm.sh`.

**Decisions left for you, deliberately not made:**

1. **Which run the report quotes.** `finetune-lr0.0003` is the defensible choice
   on the numbers. The report tables in `PFC_Report/cap5/section2.tex` are still
   `--`.
2. **Whether to report the sweep at all, or only the three-arm comparison.** The
   report's §Desenho experimental describes three arms. The sweep is eight runs.
   Presenting the sweep strengthens the pre-training claim considerably, but it
   is a different story from the one the methodology chapter currently sets up.
3. **Whether the Barren result needs its own treatment.** It is the weak class
   everywhere, it drives the mIoU spread, and no ablation exists for it.

---

## Half the annotation budget: 40 train tiles, same 20 test tiles

Ran 2026-09-23 by `train_scripts/run_half_arms.sh`. Hypothesis under test: the
fine-tune advantage should grow when there is less data to learn from.

Setup. The 40 are the first by `train_rank`, verified to be an exact prefix
(`train_rank_prefix(80)[:40] == train_rank_prefix(40)`, ranks 0-39, lowest rank
outside the set 40.0), so they are the tiles a 50-image budget would actually
have produced. Test stays at 20 tiles, unchanged, because it is the measuring
instrument and holding it fixed is what makes these runs subtractable from the
full-budget ones. Zero-shot not re-run: it never touches training data.

The prefix is **not** a uniform half. Pixels halve exactly, but the majority
class loses most:

| class | 80 tiles | 40 tiles | px kept |
|---|---|---|---|
| Urban | 3.83% | 3.56% | 46.4% |
| Agriculture_Rangeland | 54.94% | 50.28% | 45.8% |
| Forest | 15.67% | 17.25% | 55.0% |
| Water | 20.99% | 24.12% | 57.5% |
| Barren | 4.29% | 4.62% | 53.8% |

So the 40-tile set is slightly *more* balanced than the 80. Any widening of the
gap therefore could not have been blamed on starved minority classes.

### Result: the hypothesis is not supported

| lr | ft 80 | sc 80 | gap | ft 40 | sc 40 | gap | ft delta | sc delta |
|---|---|---|---|---|---|---|---|---|
| 3e-5 | 81.17 | 78.38 | +2.79 | 77.04 | 72.87 | **+4.17** | -4.12 | -5.50 |
| 1e-4 | 81.15 | 80.35 | +0.80 | 77.56 | 78.53 | **-0.97** | -3.59 | -1.82 |
| 3e-4 | 81.96 | 78.65 | +3.31 | 80.75 | 79.10 | **+1.65** | -1.21 | **+0.45** |

The gap widens at 3e-5, flips sign at 1e-4 and narrows at 3e-4. There is no
consistent direction, so halving the annotation budget does not demonstrably
change how much the DeepGlobe pre-training is worth.

Two further observations sharpen this into a statement about measurement rather
than about pre-training:

1. **`scratch 3e-4` scored *higher* on half the data than on all of it**
   (79.10 vs 78.65). That cannot be a real data-size effect. It puts a concrete
   floor under the noise of this setup: differences of about one point are not
   measurable on a 20-tile test split without seed repetition.
2. **The clean group separation of the full-budget sweep does not survive.** At
   80 tiles every fine-tune run beat every scratch run. At 40 the worst
   fine-tune (77.04) is below the best scratch (79.10), so the two groups
   interleave.

Observation 2 does **not** retroactively weaken the full-budget result, which
stands on its own eight runs over a 100x learning-rate range. It does mean the
pre-training advantage is not robust to shrinking the training set, which is the
opposite of what was expected going in.

### The real pattern is learning rate, not budget

At half budget the fine-tuned arm degrades in strict order of learning rate:
-4.12 at 3e-5, -3.59 at 1e-4, -1.21 at 3e-4. Half the sub-tiles means half the
optimisation steps per epoch, and the low learning rates no longer move the
weights far enough within the early-stopping patience. The flat learning-rate
response reported at full budget (1.14 points across a 100x range) is therefore
a property of having enough data, not of the pre-trained initialisation itself.
At 40 tiles the fine-tune spread across just a 10x range is 3.71 points.

### Per-class, 40 tiles

| run | Urban | Agri_Rang | Forest | Water | Barren | mIoU | px acc |
|---|---|---|---|---|---|---|---|
| finetune 3e-5 | 86.40 | 88.43 | 83.68 | 94.22 | 32.48 | 77.04 | 93.26 |
| finetune 1e-4 | 89.34 | 88.67 | 85.16 | 95.28 | 29.33 | 77.56 | 93.45 |
| finetune 3e-4 | 89.35 | 90.11 | 85.15 | 94.86 | 44.26 | 80.75 | 94.22 |
| scratch 3e-5 | 81.09 | 84.74 | 83.82 | 91.93 | 22.78 | 72.87 | 90.68 |
| scratch 1e-4 | 87.64 | 88.88 | 85.43 | 95.00 | 35.69 | 78.53 | 93.53 |
| scratch 3e-4 | 87.22 | 89.47 | 85.59 | 95.59 | 37.63 | 79.10 | 93.91 |

Barren again accounts for most of the spread, ranging 22.78 to 44.26 while no
other class varies more than 8.26 points. The mIoU ordering of these six runs is
very close to their Barren ordering.

### Not written into the report

These twelve runs are not in `PFC_Report`. Reporting them honestly means
reporting a null result plus evidence that the measurement floor is around one
point, which is a different and more cautious story than the full-budget sweep
currently tells. That is an editorial call.
