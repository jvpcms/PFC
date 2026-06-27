"""
Train U-Net (EfficientNetB0 encoder) on pre-tiled DeepGlobe dataset.
Requires data to be tiled first: python train_scripts/prepare_deepglobe.py

Usage:
    python train_scripts/train_deepglobe_unet.py
    python train_scripts/train_deepglobe_unet.py --epochs 100 --batch-size 16 --lr 5e-5
"""

import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


# ── Constants ─────────────────────────────────────────────────────────────────
TILES_DIR   = Path('data/deepglobe/tiles')
MODELS_DIR  = Path('models/deepglobe_unet')
N_CLASSES   = 7
CLASS_NAMES = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
INPUT_SHAPE = (256, 256, 3)
TILE_SIZE   = 256
N_TILES     = 9
SKIP_LAYERS = [
    'block2a_expand_activation',
    'block3a_expand_activation',
    'block4a_expand_activation',
    'block6a_expand_activation',
]


# ── GPU health check ──────────────────────────────────────────────────────────
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError(
            'No GPU detected. Training on CPU is not supported for this script. '
            'Check CUDA installation and driver compatibility.'
        )
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'GPU(s) available: {[g.name for g in gpus]}')
    print(f'TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}')


# ── Loss ──────────────────────────────────────────────────────────────────────
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), N_CLASSES)
    axes = [1, 2]
    intersection   = tf.reduce_sum(y_true_oh * y_pred, axis=axes)
    union          = tf.reduce_sum(y_true_oh, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)

def bce_dice_loss(y_true, y_pred):
    bce  = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


# ── Data loader ───────────────────────────────────────────────────────────────
def make_datasets(batch_size: int, val_split: float, seed: int):
    tiles_dir = TILES_DIR / 'train'
    if not tiles_dir.exists():
        raise FileNotFoundError(
            f'Tiles not found at {tiles_dir}. '
            'Run: python train_scripts/prepare_deepglobe.py'
        )

    sat_files = sorted(tiles_dir.glob('*_sat.npy'))
    pairs = []
    for s in sat_files:
        m = tiles_dir / s.name.replace('_sat.npy', '_mask.npy')
        if m.exists():
            pairs.append((str(s), str(m)))

    if not pairs:
        raise RuntimeError(f'No sat+mask tile pairs found in {tiles_dir}')

    img_to_pairs = defaultdict(list)
    for s, m in pairs:
        img_id = Path(s).stem.rsplit('_', 2)[0]
        img_to_pairs[img_id].append((s, m))

    img_ids = sorted(img_to_pairs.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(img_ids)

    n_val       = max(1, int(len(img_ids) * val_split))
    val_ids     = set(img_ids[:n_val])
    train_pairs = [p for img_id in img_ids[n_val:] for p in img_to_pairs[img_id]]
    val_pairs   = [p for img_id in val_ids          for p in img_to_pairs[img_id]]

    print(f'Train: {len(img_ids) - n_val} images → {len(train_pairs)} tiles')
    print(f'Val  : {n_val} images → {len(val_pairs)} tiles')

    def _load(sat_path, mask_path):
        img   = np.load(sat_path.numpy().decode()).astype(np.uint8)
        label = np.load(mask_path.numpy().decode()).astype(np.int32)
        return img, label

    def tf_load(sat_path, mask_path):
        img, label = tf.py_function(_load, [sat_path, mask_path], [tf.uint8, tf.int32])
        img.set_shape([256, 256, 3])
        label.set_shape([256, 256])
        return img, label

    def augment_fn(img, label):
        img_f    = tf.cast(img, tf.float32)
        combined = tf.concat([img_f, tf.cast(tf.expand_dims(label, -1), tf.float32)], axis=-1)
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)
        k        = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k)
        img      = tf.cast(combined[:, :, :3], tf.uint8)
        label    = tf.cast(combined[:, :, 3], tf.int32)
        return img, label

    def build_ds(pairs, augment):
        sat_p, mask_p = map(list, zip(*pairs))
        ds = tf.data.Dataset.from_tensor_slices((sat_p, mask_p))
        if augment:
            ds = ds.shuffle(len(sat_p), reshuffle_each_iteration=True)
        ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return build_ds(train_pairs, augment=True), build_ds(val_pairs, augment=False)


# ── Model ─────────────────────────────────────────────────────────────────────
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def decoder_block(x, skip, filters):
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    return conv_block(x, filters)

def build_unet(encoder: str = 'efficientnetb0'):
    inputs   = tf.keras.Input(shape=INPUT_SHAPE)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_tensor=inputs
    )
    skips  = [backbone.get_layer(n).output for n in SKIP_LAYERS]
    bridge = backbone.output
    x = conv_block(bridge, 256)
    x = decoder_block(x, skips[3], 256)
    x = decoder_block(x, skips[2], 128)
    x = decoder_block(x, skips[1], 64)
    x = decoder_block(x, skips[0], 32)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = conv_block(x, 16)
    outputs = tf.keras.layers.Conv2D(N_CLASSES, 1, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


# ── Train ─────────────────────────────────────────────────────────────────────
def main(args):
    check_gpu()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f'unet-effb0-bcedice-bs{args.batch_size}-lr{args.lr}'

    config = dict(
        epochs       = args.epochs,
        lr           = args.lr,
        lr_factor    = args.lr_factor,
        lr_patience  = args.lr_patience,
        es_patience  = args.es_patience,
        batch_size   = args.batch_size,
        val_split    = args.val_split,
        encoder      = 'efficientnetb0',
        input_shape  = INPUT_SHAPE,
        n_classes    = N_CLASSES,
        loss         = 'bce_dice',
        optimizer    = 'adam',
        augmentation = 'hflip+vflip+rot90',
        tiling       = f'{N_TILES}x{N_TILES} ({TILE_SIZE}px, border discarded)',
    )

    train_ds, val_ds = make_datasets(args.batch_size, args.val_split, args.seed)
    print(f'Train batches: {len(train_ds)} | Val batches: {len(val_ds)}')

    model = build_unet()
    per_class_iou = [
        tf.keras.metrics.IoU(
            num_classes=N_CLASSES,
            target_class_ids=[i],
            name=f'iou_{CLASS_NAMES[i].lower()}',
            sparse_y_pred=False,
        )
        for i in range(N_CLASSES)
    ]
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss      = bce_dice_loss,
        metrics   = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, name='miou',
                                     sparse_y_pred=False),
            *per_class_iou,
        ],
    )
    total = sum(tf.size(w).numpy() for w in model.weights)
    print(f'Total params: {total:,}')

    wandb.init(project='pitcic-segmentation', name=run_name, config=config)

    callbacks = [
        WandbMetricsLogger(log_freq='epoch'),
        WandbModelCheckpoint(
            str(MODELS_DIR / 'best.keras'),
            monitor='val_miou',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_miou',
            mode='max',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_miou',
            mode='max',
            patience=args.es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(MODELS_DIR / 'final.keras')
    print(f'Model saved to {MODELS_DIR}')
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepGlobe U-Net segmentation model.')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch-size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--lr-factor',   type=float, default=0.5,
                        help='LR reduction factor on plateau')
    parser.add_argument('--lr-patience', type=int,   default=5,
                        help='Epochs without improvement before LR reduction')
    parser.add_argument('--es-patience', type=int,   default=10,
                        help='Epochs without improvement before early stopping')
    parser.add_argument('--val-split',   type=float, default=0.15)
    parser.add_argument('--seed',        type=int,   default=42)
    main(parser.parse_args())
