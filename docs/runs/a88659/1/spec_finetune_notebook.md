# Spec: fine-tuning notebook for the INPE/CBERS annotated tiles

## Objective

A notebook that fine-tunes the frozen DeepGlobe checkpoint on the 50 hand-annotated
CBERS-4A tiles and logs to wandb. Reuses the existing data-decode and training code
rather than restating it.

## Deliverables

1. `train_scripts/export_finetune_split.py` -- materialises the split on disk.
2. `Finetune_INPE_UNet.ipynb` -- the fine-tuning run.

## User decisions already made -- do not revisit

* **Unknown stays an ordinary class (index 5).** It is trained and scored like any
  other. Do not mask it out of the loss, do not add `sample_weight`.
* **No cross-validation.** Train on the `train` split, validate on the `test` split.
* **Crop is 4x4 non-overlapping 256px per 1024px tile.** 16 sub-tiles, no stride,
  no overlap. DeepGlobe's 3x3 stride-178 overlap grid does not apply here.
* **No encoder freeze by default** (`FREEZE_EPOCHS = 0`), for the reasons in
  `notes/inpe_zero_shot_baseline.md` and the session discussion: the decoder is
  already pretrained on this exact taxonomy so there is no random-head gradient
  shock to protect against, and `trainable=False` would also pin the encoder's
  BatchNorm statistics to DeepGlobe's, forfeiting adaptation to CBERS radiometry.
  The mechanism must still be present and switchable.

## Deliverable 1 -- `train_scripts/export_finetune_split.py`

The CVAT export puts all 50 masks in one flat directory
(`data/inpe_scenes/annotation_tiles/half_labels/SegmentationClass/`) while the RGB
tiles are already split across `annotation_tiles/train/` and `annotation_tiles/test/`.
This script pairs them and writes a split-organised copy:

```
data/inpe_scenes/annotation_tiles/finetune/
    train/images/<tile_id>.png      train/masks/<tile_id>.png
    test/images/<tile_id>.png       test/masks/<tile_id>.png
```

Requirements:

* **Copy, never move.** `half_labels/` is a CVAT export artefact and annotation is
  still in progress -- the remaining 50 tiles will arrive as a new export that has to
  merge cleanly, and `train_scripts/eval_inpe_tiles.py` globs `SegmentationClass/*.png`
  directly and would break if files left it. Copying keeps the script re-runnable when
  batch 2 lands.
* Idempotent: safe to re-run, overwrites its own output, never touches the source.
* The split of each tile comes from which of `annotation_tiles/{train,test}/` holds its
  RGB PNG. Reuse `annotated_tiles()` from `train_scripts/eval_inpe_tiles.py` -- it
  already resolves exactly this and raises on a tile present in both.
* Print a summary: per split, number of tiles and number of distinct scenes.
* Expected today: 40 train tiles / 18 scenes, 10 test tiles / 9 scenes.

## Deliverable 2 -- `Finetune_INPE_UNet.ipynb`

### What to reuse, and from where

Import rather than restate. Both modules are `if __name__ == '__main__'`-guarded, so
importing them is safe; add `sys.path.insert(0, 'train_scripts')` first.

From `train_scripts/eval_inpe_tiles.py`:
`rgb_mask_to_labels`, `annotated_tiles`, `CLASS_NAMES`, `N_CLASSES`, `UNKNOWN_IDX`,
`TILE`, `confusion`, `report`.

From `train_scripts/train_deepglobe_unet.py`:
`bce_dice_loss`, `dice_loss`, `check_gpu`.

`augment_fn` is nested inside `make_datasets()` and cannot be imported -- copy it
verbatim (flips, rot90, brightness 0.2, contrast/saturation 0.8-1.2, hue 0.05). Keep it
identical: it is part of the recipe being carried over, and changing it adds a variable
to a comparison that already has two.

### Notebook structure

**Cell 1 (markdown)** -- what this run is, the zero-shot floor it must beat
(px acc 0.391 / mIoU 0.246 over all 50 tiles; 0.373 / 0.239 on the test split, at the 5%
Unknown threshold), and the DeepGlobe source reference 0.813 with the explicit warning
that it is not a ceiling for this score (different dataset, different Unknown policy,
different tiling).

**Cell 2** -- imports, constants, `check_gpu()`. Constants as notebook-level names so
they are easy to change between runs: `BASE_CHECKPOINT`, `LR`, `BATCH_SIZE`, `EPOCHS`,
`DROPOUT`, `FREEZE_EPOCHS = 0`, `UNFREEZE_LR_DIV`, `ES_PATIENCE`, `LR_FACTOR`,
`LR_PATIENCE`, `SEED`. Defaults: `LR = 3e-5`, `BATCH_SIZE = 8` (4 GB card),
`EPOCHS = 80`, `ES_PATIENCE = 15`, `LR_PATIENCE = 6`, `LR_FACTOR = 0.5`, `DROPOUT = 0.3`.
Seed numpy, python `random`, and `tf.keras.utils.set_random_seed`.

**Cell 3** -- run `export_finetune_split.py`'s function (import it) or call it via
subprocess, then load both splits into memory.

Loading: for each 1024px pair, cut into 16 non-overlapping 256px sub-tiles, decode the
mask with `rgb_mask_to_labels`. Result: `(640, 256, 256, 3)` uint8 + `(640, 256, 256)`
int32 for train, `(160, ...)` for test. That is ~126 MB for train, so hold it in memory
and build `tf.data` with `from_tensor_slices` -- do not write `.npy` tiles to disk.

Assert the shapes and that labels are within `[0, N_CLASSES)`. Print the per-split class
distribution.

**Cell 4** -- `tf.data` pipelines. Train: shuffle with the full buffer and
`reshuffle_each_iteration=True`, map `augment_fn`, batch, prefetch. Val: batch and
prefetch only, no shuffle, no augmentation. Mirror `make_datasets()`.

**Cell 5** -- load the checkpoint and compile.

```python
model = tf.keras.models.load_model(BASE_CHECKPOINT, compile=False)
```

`compile=False` avoids needing `custom_objects` for `bce_dice_loss`.

Compile with Adam at `LR`, loss `bce_dice_loss`, and the same metric set as the base
script: `SparseCategoricalAccuracy(name='accuracy')`,
`MeanIoU(num_classes=N_CLASSES, name='miou', sparse_y_pred=False)`, and one
`tf.keras.metrics.IoU(target_class_ids=[i], name=f'iou_{CLASS_NAMES[i].lower()}',
sparse_y_pred=False)` per class. Write it as a `compile_model(lr)` function, because the
freeze path needs to recompile.

**Freeze mechanism.** `build_unet()` passes `input_tensor=inputs` and taps skips via
`backbone.get_layer(n).output`, so the encoder is flattened into the outer functional
graph -- **after `load_model` there is no nested backbone submodel to toggle**. Implement
the freeze by layer name instead: EfficientNetB0's layers are the ones whose names start
with `stem_`, `block1a`..`block7a`, or `top_`. Set `layer.trainable = False` over that
set, recompile, fit `FREEZE_EPOCHS`, then re-enable, recompile at `LR / UNFREEZE_LR_DIV`,
and continue with `initial_epoch=FREEZE_EPOCHS`. With the default `FREEZE_EPOCHS = 0`
this whole branch is skipped. Print how many layers were matched so a silent no-match is
visible.

**Cell 6** -- wandb. `wandb.init(project='pitcic-segmentation', ...)`, run name encoding
the config (e.g. `finetune-inpe-effb0-bs8-lr3e-05`). The config dict must mirror the base
script's fields so the two runs are comparable in one wandb table, with these values
changed or added:

* `tiling = '4x4 non-overlapping grid (256px)'`
* `class_merge` -- same string as the base script
* `base_checkpoint` -- the checkpoint filename
* `finetune = True`
* `dataset = 'inpe_cbers_annotation_tiles_50'`
* `n_train_tiles`, `n_val_tiles`, `n_train_scenes`, `n_val_scenes`
* `unknown_policy = 'trained and scored as an ordinary class'`

Callbacks, same split-signal logic as the base script:
`WandbMetricsLogger(log_freq='epoch')`; `WandbModelCheckpoint` to
`models/inpe_finetune/best.keras` on `val_miou` / max / `save_best_only=True`;
`ReduceLROnPlateau` on `val_miou` / max, `factor=LR_FACTOR`, `patience=LR_PATIENCE`,
`min_lr=1e-7`; `EarlyStopping` on `val_miou` / max, `patience=ES_PATIENCE`,
`restore_best_weights=True`. (Note: the base script monitors `val_miou` for
ReduceLROnPlateau -- match the code, not the `notes/` handoff, which says `val_loss`.)

`models/inpe_finetune/` must be created if absent.

**Cell 7** -- fit. Handle the freeze branch as described. Save `final.keras` alongside
`best.keras`. `wandb.finish()`.

**Cell 8** -- evaluate the fine-tuned model on the val/test split with the *same*
protocol as the baseline, so the numbers are directly comparable: predict, then
`confusion()` + `report()` imported from `eval_inpe_tiles`. Print the zero-shot numbers
next to the fine-tuned ones.

Predict in chunks (`model(chunk, training=False)`), not `model.predict` over everything
-- `predict` concatenates every `(256,256,6)` softmax map before returning and exhausts
the 4 GB GPU. This is already fixed in `eval_inpe_tiles.py`; do the same here.

**Cell 9 (markdown)** -- a short, prominent caveat: with CV deferred, the test split is
serving as the validation set, so `val_miou`-gated checkpointing and early stopping
select on the same tiles the number is reported on. The result is therefore optimistic
and is not the number for the monografia; it needs scene-grouped CV or a third split once
the remaining 50 annotations land.

## Constraints

* GPU is an RTX 3050 4 GB. Batch 8. Do not raise it.
* Do not modify `train_scripts/train_deepglobe_unet.py` or
  `train_scripts/eval_inpe_tiles.py` -- import from them.
* Do not run the training. Build the notebook and verify it up to the point of `fit`
  (imports resolve, split exports, tensors have the right shapes, model loads and
  compiles). Leave the fit for the user.
