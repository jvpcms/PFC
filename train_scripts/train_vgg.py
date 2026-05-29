import argparse
import json
import os
import zipfile
from datetime import datetime
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import wandb
from wandb.integration.keras import WandbMetricsLogger

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


MODELS_DIR = Path("models/vgg_classfier")
DATA_DIR = Path("data/EuroSAT_RGB")
ZIP_PATH = Path("data_zip/EuroSAT_RGB.zip")


def ensure_data():
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR.parent)
    print(f"Extracted {ZIP_PATH} -> {DATA_DIR.parent}")


def load_category(category: str) -> np.ndarray:
    folder = DATA_DIR / category
    images = sorted(folder.glob("*.jpg"))
    return np.stack([
        np.array(Image.open(p).convert("RGB").resize((64, 64)))
        for p in images
    ])


def build_vgg_like_model() -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation="softmax"))
    return model


def main(args):
    ensure_data()

    categories = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
    print("Categories:", categories)

    data = {}
    for cat in categories:
        data[cat] = load_category(cat)
        print(f"{cat}: {data[cat].shape}")

    X = np.concatenate([data[cat] for cat in categories], axis=0)
    y = np.concatenate([np.full(data[cat].shape[0], i) for i, cat in enumerate(categories)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )

    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    MODELS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vgg_{args.optimizer}_e{args.epochs}_{timestamp}"
    model_path = MODELS_DIR / f"vgg_like_model_{timestamp}.keras"
    history_path = MODELS_DIR / f"vgg_like_model_history_{timestamp}.json"

    wandb.init(
        project="eurosat-vgg",
        name=run_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "val_split": args.val_split,
        }
    )

    model = build_vgg_like_model()
    model.compile(optimizer=args.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=[WandbMetricsLogger()],
    )

    model.save(model_path)
    with open(history_path, "w") as f:
        json.dump(history.history, f)

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Model saved to {model_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--optimizer", type=str, default="rmsprop")
    main(parser.parse_args())
