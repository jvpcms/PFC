#!/usr/bin/env python3
"""Run one arm of the CBERS-4A fine-tuning experiment, unattended.

This is `Finetune_INPE_UNet.ipynb` made runnable without a kernel or a network,
so the three arms can go overnight. The recipe, the protocol and the constants
are the notebook's; nothing new is introduced here.

Three arms, identical data, protocol and seed. Only the weights and the training
data differ:

  zeroshot  the DeepGlobe checkpoint, evaluated on `test` without any training.
            Establishes the floor the other two must beat.
  scratch   EfficientNetB0 with ImageNet weights plus a fresh decoder, trained
            only on the 80 Pampa tiles. Measures what the annotation buys on its
            own, with no DeepGlobe contact.
  finetune  the DeepGlobe checkpoint, trained on the same 80 tiles. The proposed
            configuration.

Protocol notes that are easy to get wrong:

* Unknown (index 5) is an ORDINARY TRAINED CLASS. No loss mask, no sample
  weights. It is excluded only when scoring.
* The 5% Unknown cutoff is an EVALUATION rule. Training sees every sub-tile.
* `val_miou` (Keras MeanIoU over 6 classes, every sub-tile, Unknown scored) and
  `protocol_miou_unk05` (Unknown GT excluded, sub-tiles over the cutoff dropped)
  are different quantities. Only the second is comparable across arms against
  the zero-shot floor. Model selection runs on the first.
* `test` doubles as the validation set, so every number is a validation score
  and never a held-out one. That is true of all three arms equally.

Usage:
    .venv/bin/python train_scripts/run_finetune_experiments.py --arm zeroshot
    .venv/bin/python train_scripts/run_finetune_experiments.py --arm finetune --lr 3e-5
    .venv/bin/python train_scripts/run_finetune_experiments.py --arm scratch  --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf

from eval_inpe_tiles import (
    CLASS_NAMES,
    MASK_DIR,
    N_CLASSES,
    REPO_ROOT,
    TILE,
    TILES_DIR,
    UNKNOWN_IDX,
    annotated_tiles,
    confusion,
    rgb_mask_to_labels,
)
from train_deepglobe_unet import bce_dice_loss, build_unet

BASE_CHECKPOINT = REPO_ROOT / "models" / "deepglobe_unet" / (
    "best-20260702-downsample-overlap-color-aug-dropout-30-merged-class"
    "-batch-32-lr-1e-4-miou-67.keras"
)
RESULTS_DIR = REPO_ROOT / "results" / "finetune_experiments"
SEED = 42
THRESHOLDS = (0.05, 0.25)

# Matches the base run's decoder dropout. Only used by the `scratch` arm, which
# builds a model instead of loading one; for the other two the rate is whatever
# the checkpoint carries.
SCRATCH_DROPOUT = 0.3

ENCODER_PREFIXES = ("stem_", "top_") + tuple(f"block{i}" for i in range(1, 8))


# ── Data ──────────────────────────────────────────────────────────────────────
SPLIT_TABLE = REPO_ROOT / "data" / "inpe" / "tile_split.csv"


def train_rank_prefix(n: int) -> list[str]:
    """The first `n` annotated train tiles in global pick order.

    `sample_tiles.py` builds prefix-stable pick orders, so the annotated 80 are
    exactly the first 80 `train_rank` values. Taking the first `n` of those is
    therefore the train set a smaller annotation budget would have produced, and
    not a random subsample: the allocator front-loads rare strata for coverage,
    so a prefix is more class-balanced than a uniform draw of the same size.
    """
    import csv

    annotated = {t for t, s in annotated_tiles("train") if s == "train"}
    ranked = []
    with open(SPLIT_TABLE, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["tile_id"] in annotated and row["train_rank"] not in ("", "None"):
                ranked.append((float(row["train_rank"]), row["tile_id"]))
    ranked.sort()
    if len(ranked) != len(annotated):
        raise ValueError(
            f"{len(annotated)} annotated train tiles but {len(ranked)} ranked in "
            f"{SPLIT_TABLE}"
        )
    if n > len(ranked):
        raise ValueError(f"--train-subset {n} exceeds the {len(ranked)} annotated tiles")
    return [t for _, t in ranked[:n]]


def load_split(split: str, keep: list[str] | None = None):
    """(N,256,256,3) uint8 and (N,256,256) int32 for one split.

    Each 1024px tile becomes 16 non-overlapping 256px sub-tiles. Sub-tiles of one
    tile never straddle the split, because the split is assigned per tile.
    """
    tiles = [t for t, s in annotated_tiles(split) if s == split]
    if keep is not None:
        wanted = set(keep)
        missing = wanted - set(tiles)
        if missing:
            raise ValueError(f"{len(missing)} requested tiles are not in {split}")
        tiles = [t for t in tiles if t in wanted]
    images, labels = [], []
    for tile_id in tiles:
        img = np.array(Image.open(TILES_DIR / split / f"{tile_id}.png").convert("RGB"))
        lbl = rgb_mask_to_labels(
            np.array(Image.open(MASK_DIR / f"{tile_id}.png").convert("RGB"))
        )
        if img.shape[:2] != lbl.shape:
            raise ValueError(f"{tile_id}: image {img.shape[:2]} != mask {lbl.shape}")
        h, w = lbl.shape
        if h % TILE or w % TILE:
            raise ValueError(f"{tile_id}: {h}x{w} is not a multiple of {TILE}")
        for r in range(0, h, TILE):
            for c in range(0, w, TILE):
                images.append(img[r:r + TILE, c:c + TILE])
                labels.append(lbl[r:r + TILE, c:c + TILE])
    return len(tiles), np.stack(images), np.stack(labels).astype(np.int32)


def augment_fn(img, label):
    """Verbatim from make_datasets() in train_deepglobe_unet.py.

    Carried over unchanged on purpose: the arms differ in weights and training
    data, and a third moving part would make the comparison unreadable.
    """
    img_f = tf.cast(img, tf.float32)
    combined = tf.concat([img_f, tf.cast(tf.expand_dims(label, -1), tf.float32)], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)
    img = tf.cast(combined[:, :, :3], tf.uint8)
    label = tf.cast(combined[:, :, 3], tf.int32)
    img_f = tf.cast(img, tf.float32) / 255.0
    img_f = tf.image.random_brightness(img_f, max_delta=0.2)
    img_f = tf.image.random_contrast(img_f, lower=0.8, upper=1.2)
    img_f = tf.image.random_saturation(img_f, lower=0.8, upper=1.2)
    img_f = tf.image.random_hue(img_f, max_delta=0.05)
    img = tf.cast(tf.clip_by_value(img_f * 255.0, 0, 255), tf.uint8)
    return img, label


# ── Model ─────────────────────────────────────────────────────────────────────
def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=bce_dice_loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, name="miou", sparse_y_pred=False),
            *[
                tf.keras.metrics.IoU(
                    num_classes=N_CLASSES,
                    target_class_ids=[i],
                    name=f"iou_{CLASS_NAMES[i].lower()}",
                    sparse_y_pred=False,
                )
                for i in range(N_CLASSES)
            ],
        ],
    )


def build_arm(arm):
    if arm == "scratch":
        model, _ = build_unet(SCRATCH_DROPOUT)
    else:
        model = tf.keras.models.load_model(BASE_CHECKPOINT, compile=False)
    if model.output_shape[-1] != N_CLASSES:
        raise ValueError(f"model outputs {model.output_shape[-1]} classes, expected {N_CLASSES}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def predict_labels(model, images, batch_size):
    """argmax predictions, in chunks.

    `model.predict` concatenates every (256,256,6) softmax map before returning,
    which is the largest allocation in the run and exhausts a 4 GB card.
    """
    preds = np.empty(images.shape[:3], dtype=np.int32)
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size].astype(np.float32)
        preds[start:start + len(chunk)] = np.argmax(
            model(chunk, training=False).numpy(), axis=-1
        )
    return preds


def protocol_scores(y_true, y_pred, thresh):
    """The reportable numbers: drop noisy sub-tiles, exclude Unknown GT pixels."""
    unknown_frac = (y_true == UNKNOWN_IDX).mean(axis=(1, 2))
    keep = unknown_frac < thresh
    if not keep.any():
        return None
    lab, prd = y_true[keep], y_pred[keep]
    valid = lab != UNKNOWN_IDX
    cm = confusion(lab[valid], prd[valid])
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    present = cm.sum(1) > 0
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return {
        "threshold": thresh,
        "sub_tiles_kept": int(keep.sum()),
        "sub_tiles_total": int(len(keep)),
        "pixels_scored": int(valid.sum()),
        "pixels_excluded_unknown": int((~valid).sum()),
        "px_acc": float(np.trace(cm) / cm.sum()),
        "miou_present": float(np.nanmean(np.where(present, iou, np.nan))),
        "iou_per_class": {
            CLASS_NAMES[i]: (float(iou[i]) if present[i] else None) for i in range(N_CLASSES)
        },
        "confusion": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("zeroshot", "scratch", "finetune"))
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8, help="4 GB card: do not raise")
    ap.add_argument("--es-patience", type=int, default=15)
    ap.add_argument("--lr-patience", type=int, default=6)
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    ap.add_argument(
        "--train-subset", type=int, metavar="N",
        help="train on the first N annotated tiles by train_rank instead of all 80. "
             "Used to measure how the arms separate at a smaller annotation budget. "
             "The test split is never reduced, so runs stay comparable to the full ones.",
    )
    ap.add_argument(
        "--eval-batch-size", type=int, default=4,
        help="smaller than --batch-size: cuDNN picks a conv algorithm by profiling, "
             "and on a 4 GB card every candidate can fail on workspace allocation",
    )
    ap.add_argument(
        "--eval-only", metavar="CHECKPOINT",
        help="skip training and score this checkpoint instead. Exists because "
             "evaluating in the same process that just trained exhausts the card: "
             "the optimizer slots and the training graph are still resident, and "
             "cuDNN then has no workspace left to profile a conv algorithm into. "
             "Run training, then score the saved best.keras with this.",
    )
    args = ap.parse_args()

    name = args.arm if args.arm == "zeroshot" else f"{args.arm}-lr{args.lr:g}"
    if args.train_subset:
        name += f"-tr{args.train_subset}"
    if args.tag:
        name += f"-{args.tag}"
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("no GPU visible; this is not meant to run on CPU")

    keep = train_rank_prefix(args.train_subset) if args.train_subset else None
    n_train_tiles, x_train, y_train = load_split("train", keep)
    n_val_tiles, x_val, y_val = load_split("test")
    assert x_train.shape == (n_train_tiles * 16, TILE, TILE, 3), x_train.shape
    assert x_val.shape == (n_val_tiles * 16, TILE, TILE, 3), x_val.shape
    assert y_train.max() < N_CLASSES and y_val.max() < N_CLASSES

    print(f"== {name}")
    print(f"train {n_train_tiles} tiles -> {len(x_train)} sub-tiles | "
          f"test {n_val_tiles} tiles -> {len(x_val)} sub-tiles")

    if args.eval_only:
        model = tf.keras.models.load_model(args.eval_only, compile=False)
        if model.output_shape[-1] != N_CLASSES:
            raise ValueError(
                f"model outputs {model.output_shape[-1]} classes, expected {N_CLASSES}"
            )
        print(f"eval-only: {args.eval_only}")
    else:
        model = build_arm(args.arm)
        compile_model(model, args.lr)

    record = {
        "arm": args.arm,
        "name": name,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs_max": args.epochs,
        "seed": SEED,
        "n_train_tiles": n_train_tiles,
        "train_subset": args.train_subset,
        "train_selection": "first N by train_rank" if args.train_subset else "all annotated",
        "n_val_tiles": n_val_tiles,
        "n_train_subtiles": int(len(x_train)),
        "n_val_subtiles": int(len(x_val)),
        "base_checkpoint": BASE_CHECKPOINT.name if args.arm != "scratch" else None,
        "encoder_init": "imagenet" if args.arm == "scratch" else "deepglobe",
        "unknown_policy": "trained as an ordinary class; excluded only when scoring",
        "tiling": "4x4 non-overlapping 256px grid",
    }

    if args.arm != "zeroshot" and not args.eval_only:
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(len(x_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((x_val, y_val))
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(out / "best.keras"), monitor="val_miou", mode="max",
                save_best_only=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_miou", mode="max", factor=args.lr_factor,
                patience=args.lr_patience, min_lr=1e-7, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_miou", mode="max", patience=args.es_patience,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
        ]
        started = time.time()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                         callbacks=callbacks, verbose=2)
        record["epochs_run"] = len(hist.history["loss"])
        record["train_seconds"] = round(time.time() - started, 1)
        best = int(np.argmax(hist.history["val_miou"]))
        record["best_epoch"] = best + 1
        record["keras_val_miou"] = float(hist.history["val_miou"][best])
        record["keras_val_accuracy"] = float(hist.history["val_accuracy"][best])

    preds = predict_labels(model, x_val, args.eval_batch_size)
    record["protocol"] = {
        f"unk{int(t * 100):02d}": protocol_scores(y_val, preds, t) for t in THRESHOLDS
    }

    metrics_path = out / "metrics.json"
    if args.eval_only and metrics_path.exists():
        # Preserve the training fields from the run that produced the checkpoint;
        # only the protocol block is being recomputed.
        previous = json.loads(metrics_path.read_text())
        previous.update({k: v for k, v in record.items() if k == "protocol"})
        record = previous
    record["eval_checkpoint"] = str(args.eval_only) if args.eval_only else "in-process"
    metrics_path.write_text(json.dumps(record, indent=2))

    for key, sc in record["protocol"].items():
        if sc is None:
            continue
        print(f"\n-- {key}: {sc['sub_tiles_kept']}/{sc['sub_tiles_total']} sub-tiles, "
              f"{sc['pixels_excluded_unknown']:,} Unknown px excluded")
        print(f"   px acc {sc['px_acc']:.4f}   mIoU {sc['miou_present']:.4f}")
        for cls, v in sc["iou_per_class"].items():
            print(f"     {cls:<22} {'absent' if v is None else f'{v:.4f}'}")
    print(f"\nwrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
