#!/usr/bin/env python3
"""Evaluate a segmentation checkpoint on the hand-annotated CBERS-4A tiles.

Establishes the transfer baseline every fine-tune run must beat. The metric
protocol is the one fixed in ``Evaluate_INPE_TestCVAT.ipynb``:

* 1024px annotated tiles are cut into 16 non-overlapping 256px tiles;
* the CVAT export still uses the old 7-class DeepGlobe colors, so it is mapped
  to the merged 6-class taxonomy at load time (Agriculture + Rangeland -> 1);
* Unknown ground-truth pixels are excluded from scoring. Unknown carries two
  meanings here (genuinely unknowable ground, i.e. cloud and shadow, and merely
  unannotated background) and is not a reliable label either way. Predicting
  Unknown on a labeled pixel still counts as an error.
* a sub-tile is dropped entirely once its Unknown fraction reaches the
  threshold, because what is left is noise-dominated. The threshold is reported
  at several values at once (5% and 25% by default): on this dataset a cloudy
  parent tile is all-or-nothing at 5%, so a single threshold would quietly turn
  the figure into a clear-sky-only baseline.

Usage:
    .venv/bin/python train_scripts/eval_inpe_tiles.py
    .venv/bin/python train_scripts/eval_inpe_tiles.py --split train
    .venv/bin/python train_scripts/eval_inpe_tiles.py --unknown-thresh 0.05 0.25 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TILES_DIR = REPO_ROOT / "data" / "inpe_scenes" / "annotation_tiles"
MASK_DIR = TILES_DIR / "full_labels" / "SegmentationClass"
DEFAULT_MODEL = (
    REPO_ROOT
    / "models"
    / "deepglobe_unet"
    / "best-20260702-downsample-overlap-color-aug-dropout-30-merged-class"
    "-batch-32-lr-1e-4-miou-67.keras"
)

TILE = 256
N_CLASSES = 6
CLASS_NAMES = ["Urban", "Agriculture_Rangeland", "Forest", "Water", "Barren", "Unknown"]
UNKNOWN_IDX = 5

# CVAT "Segmentation mask 1.1" export colors -> merged 6-class index.
# background (0,0,0) == Unknown: unannotated pixels are treated as Unknown.
CVAT_COLOR_TO_CLASS = {
    (0, 255, 255): 0,    # Urban
    (255, 255, 0): 1,    # Agriculture
    (255, 0, 255): 1,    # Rangeland (merged)
    (0, 255, 0): 2,      # Forest
    (0, 0, 255): 3,      # Water
    (255, 255, 255): 4,  # Barren
    (0, 0, 0): 5,        # Unknown / background
}


def rgb_mask_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 -> (H,W) int32 class index."""
    labels = np.full(mask_rgb.shape[:2], UNKNOWN_IDX, dtype=np.int32)
    for color, idx in CVAT_COLOR_TO_CLASS.items():
        labels[np.all(mask_rgb == color, axis=-1)] = idx
    return labels


def annotated_tiles(split: str) -> list[tuple[str, str]]:
    """(tile_id, split) for tiles that are both annotated and in the request.

    The split is resolved here and carried with the id. Looking the image up by
    stem alone would silently prefer whichever directory is searched first, so a
    tile re-cut into the other split without deleting the old PNG would be
    scored from the wrong copy while the report still claimed the split asked
    for.
    """
    annotated = {p.stem for p in MASK_DIR.glob("*.png")}
    splits = ("train", "test") if split == "all" else (split,)
    seen: dict[str, str] = {}
    out: list[tuple[str, str]] = []
    for sp in splits:
        for p in sorted((TILES_DIR / sp).glob("*.png")):
            if p.stem not in annotated:
                continue
            if p.stem in seen:
                raise ValueError(
                    f"{p.stem} exists in both {seen[p.stem]}/ and {sp}/ -- "
                    "stale tile, the split is ambiguous"
                )
            seen[p.stem] = sp
            out.append((p.stem, sp))
    return sorted(out)


def load_tiles(tiles: list[tuple[str, str]]):
    """Cut every annotated 1024px tile into 256px sub-tiles."""
    imgs, lbls, parents, unk = [], [], [], []
    for tid, split in tiles:
        img = np.array(Image.open(TILES_DIR / split / f"{tid}.png").convert("RGB"))
        lbl = rgb_mask_to_labels(np.array(Image.open(MASK_DIR / f"{tid}.png").convert("RGB")))
        if img.shape[:2] != lbl.shape:
            raise ValueError(f"{tid}: image {img.shape[:2]} != mask {lbl.shape}")
        h, w = lbl.shape
        if h % TILE or w % TILE:
            raise ValueError(
                f"{tid}: {h}x{w} is not a multiple of {TILE}; the right/bottom "
                "strip would be dropped without notice"
            )
        for r in range(0, h - TILE + 1, TILE):
            for c in range(0, w - TILE + 1, TILE):
                t_lbl = lbl[r:r + TILE, c:c + TILE]
                imgs.append(img[r:r + TILE, c:c + TILE])
                lbls.append(t_lbl)
                parents.append(tid)
                unk.append((t_lbl == UNKNOWN_IDX).mean())
    return imgs, lbls, parents, np.asarray(unk)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def report(cm: np.ndarray) -> None:
    px_acc = np.trace(cm) / cm.sum()
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    present = cm.sum(1) > 0
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)

    print(f"\nPixel accuracy: {px_acc:.4f}")
    print(f'{"class":<22} {"IoU":>8} {"GT px":>12} {"pred px":>12}')
    for i, name in enumerate(CLASS_NAMES):
        iou_s = f"{iou[i]:.4f}" if present[i] else "   n/a"
        print(f"{name:<22} {iou_s:>8} {cm.sum(1)[i]:>12,} {cm.sum(0)[i]:>12,}")

    miou_present = np.nanmean(np.where(present, iou, np.nan))
    print(f"\nmIoU over classes present in GT: {miou_present:.4f}")
    print("(DeepGlobe val reference, 5-class excl. Unknown: 0.813)")
    print(f"pred = Unknown on labeled px: {cm.sum(0)[UNKNOWN_IDX] / cm.sum():.2%}")

    print("\nRow-normalized confusion (recall per GT class)")
    print(" " * 22 + "".join(f"{n[:10]:>11}" for n in CLASS_NAMES))
    row_sum = cm.sum(1, keepdims=True)
    cm_recall = np.where(row_sum > 0, cm / np.maximum(row_sum, 1), np.nan)
    for i, name in enumerate(CLASS_NAMES):
        cells = "".join(
            "        n/a" if np.isnan(v) else f"{v:>11.2f}" for v in cm_recall[i]
        )
        print(f"{name:<22}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--split", choices=("test", "train", "all"), default="test")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--unknown-thresh",
        type=float,
        nargs="+",
        default=[0.05, 0.25],
        help="report the metrics once per threshold. The default reports both "
             "the strict 5%% used on the DeepGlobe pipeline and the 25%% the "
             "notebook recommends when 5%% discards a large share of the tiles.",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    bad = [t for t in args.unknown_thresh if not 0.0 < t <= 1.0]
    if bad:
        ap.error(f"--unknown-thresh takes fractions in (0, 1], got {bad}")

    n_masks = len(list(MASK_DIR.glob("*.png")))
    tiles = annotated_tiles(args.split)
    scenes = {tid.rsplit("_", 2)[0] for tid, _ in tiles}
    print(f"model  {args.model}")
    print(f"masks  {MASK_DIR}  ({n_masks} annotated, {len(tiles)} in split "
          f"{args.split!r}, {len(scenes)} scenes)")
    if not tiles:
        raise SystemExit("no annotated tiles for this split")

    imgs, lbls, parents, unk = load_tiles(tiles)
    images = np.stack(imgs)
    labels = np.stack(lbls)
    parents = np.asarray(parents)

    import tensorflow as tf  # imported late: keeps --help fast

    print(f"tf     {tf.__version__} on "
          f"{[d.device_type for d in tf.config.list_physical_devices()]}")
    model = tf.keras.models.load_model(args.model, compile=False)
    if model.output_shape[-1] != N_CLASSES:
        raise ValueError(
            f"model outputs {model.output_shape[-1]} classes, expected {N_CLASSES}"
        )

    # Raw uint8 in: EfficientNetB0 rescales and normalizes internally.
    # argmax per chunk rather than model.predict over everything: predict
    # concatenates every (256,256,6) softmax map before returning, which is the
    # largest allocation in the run and exhausts a 4 GB GPU at 50 tiles.
    preds = np.empty(images.shape[:3], dtype=np.int32)
    for start in range(0, len(images), args.batch_size):
        chunk = images[start:start + args.batch_size].astype(np.float32)
        probs = model(chunk, training=False).numpy()
        preds[start:start + len(chunk)] = np.argmax(probs, axis=-1)
        print(f"\r  predicting {min(start + len(chunk), len(images))}/{len(images)}",
              end="", flush=True)
    print()

    print("\nPer parent tile. px_acc(all) is over every sub-tile of the parent, "
          "unfiltered:\nthe threshold columns say how many of them each cutoff "
          "would keep, they do not\nfilter this accuracy.")
    head = "".join(f"{t:>9.0%}" for t in args.unknown_thresh)
    print(f'{"parent tile":<45}{"n":>4}{head}{"px_acc(all)":>13}{"top GT":>10}')
    for tid, _ in tiles:
        m = parents == tid
        keeps = "".join(f"{int((unk[m] < t).sum()):>9d}" for t in args.unknown_thresh)
        v = labels[m] != UNKNOWN_IDX
        acc = (preds[m][v] == labels[m][v]).mean() if v.any() else float("nan")
        mix = np.bincount(labels[m][v].ravel(), minlength=N_CLASSES)
        top = CLASS_NAMES[int(mix.argmax())][:9] if v.any() else "-"
        print(f"{tid:<45}{int(m.sum()):>4}{keeps}{acc:>13.3f}{top:>10}")

    for thresh in args.unknown_thresh:
        keep = unk < thresh
        print(f"\n{'=' * 78}\nUnknown threshold {thresh:.0%}: "
              f"{int(keep.sum())} / {len(unk)} sub-tiles kept, "
              f"{int((~keep).sum())} discarded")
        if not keep.any():
            print("  every sub-tile is above the threshold -- nothing to score")
            continue
        if (~keep).any():
            print(f"  mean Unknown fraction in discarded sub-tiles: "
                  f"{unk[~keep].mean():.1%}")
            lost = sorted({p for p, k in zip(parents, keep) if not k})
            for tid in lost:
                m = parents == tid
                print(f"  {tid}: {int((unk[m] < thresh).sum())}/{int(m.sum())} kept")

        lab, prd = labels[keep], preds[keep]
        counts = np.bincount(lab.ravel(), minlength=N_CLASSES)
        print("  GT class distribution over kept sub-tiles")
        for name, ct in zip(CLASS_NAMES, counts):
            print(f"    {name:<22} {ct / counts.sum():6.2%}")

        valid = lab != UNKNOWN_IDX
        print(f"  {valid.mean():.1%} of pixels scored "
              f"({int((~valid).sum()):,} Unknown px excluded)")
        report(confusion(lab[valid], prd[valid]))


if __name__ == "__main__":
    main()
