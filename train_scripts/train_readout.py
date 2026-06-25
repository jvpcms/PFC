"""Standalone MLP readout trainer — runs in its own process so all GPU memory is
released on exit (lets the hybrid and baseline models train sequentially on a small GPU).

Usage:
    python train_readout.py --x X.npy --y Y.npy --xval Xv.npy --yval Yv.npy --out model.keras
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF C++ stdout/stderr spam

import numpy as np
import tensorflow as tf

for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def build_model(in_dim: int, dropout: float) -> Sequential:
    # Fixed architecture shared by hybrid (in_dim=1368) and baseline (in_dim=18)
    return Sequential([
        Dense(1024, activation="relu", input_shape=(in_dim,)),
        Dropout(dropout),
        Dense(2048, activation="relu"),
        Dropout(dropout),
        Dense(4096, activation="relu"),
        Dropout(dropout),
        Dense(2 * 4096, activation="relu"),
        Dropout(dropout),
        Dense(4096, activation="sigmoid"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--xval", required=True)
    ap.add_argument("--yval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()

    X  = np.load(a.x);   Y  = np.load(a.y)
    Xv = np.load(a.xval); Yv = np.load(a.yval)
    print(f"train X={X.shape} Y={Y.shape}  val X={Xv.shape} Y={Yv.shape}", flush=True)

    model = build_model(X.shape[1], a.dropout)
    model.compile(optimizer=Adam(learning_rate=a.lr), loss="mse")
    model.fit(
        X, Y,
        validation_data=(Xv, Yv),
        epochs=a.epochs,
        batch_size=a.batch,
        callbacks=[EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True)],
        verbose=2,
    )
    model.save(a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
