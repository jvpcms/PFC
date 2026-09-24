#!/usr/bin/env python3
"""Materialise the fine-tuning split of the hand-annotated CBERS-4A tiles.

The CVAT export drops all masks in one flat directory while the RGB tiles are
already split across ``annotation_tiles/{train,test}/``. This pairs them and
writes a split-organised copy::

    data/inpe_scenes/annotation_tiles/finetune/
        train/images/<tile_id>.png   train/masks/<tile_id>.png
        test/images/<tile_id>.png    test/masks/<tile_id>.png

Files are *copied*, never moved: ``half_labels/`` is a CVAT export artefact,
annotation is still in progress, and ``eval_inpe_tiles.py`` globs
``SegmentationClass/*.png`` directly. Copying keeps this re-runnable when the
second annotation batch lands.

Usage:
    .venv/bin/python train_scripts/export_finetune_split.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from eval_inpe_tiles import MASK_DIR, REPO_ROOT, TILES_DIR, annotated_tiles

FINETUNE_DIR = TILES_DIR / "finetune"


def scene_of(tile_id: str) -> str:
    """Scene id of a tile: the tile id minus its trailing row/column indices."""
    return tile_id.rsplit("_", 2)[0]


def export_split() -> dict[str, list[str]]:
    """Copy every annotated tile and its mask into ``finetune/<split>/``.

    Returns the tile ids written, keyed by split.
    """
    written: dict[str, list[str]] = {"train": [], "test": []}
    for split in written:
        for sub in ("images", "masks"):
            d = FINETUNE_DIR / split / sub
            # Clear rather than copy over the top. Copying alone would leave a
            # tile that has since been re-cut into the other split sitting in
            # both, which is train/val leakage that no later assertion catches.
            if d.exists():
                for stale in d.glob("*.png"):
                    stale.unlink()
            d.mkdir(parents=True, exist_ok=True)

    for tile_id, split in annotated_tiles("all"):
        shutil.copyfile(
            TILES_DIR / split / f"{tile_id}.png",
            FINETUNE_DIR / split / "images" / f"{tile_id}.png",
        )
        shutil.copyfile(
            MASK_DIR / f"{tile_id}.png",
            FINETUNE_DIR / split / "masks" / f"{tile_id}.png",
        )
        written[split].append(tile_id)
    return written


def main() -> None:
    written = export_split()
    print(f"source masks   {MASK_DIR.relative_to(REPO_ROOT)}")
    print(f"destination    {FINETUNE_DIR.relative_to(REPO_ROOT)}")
    for split, tile_ids in written.items():
        scenes = {scene_of(t) for t in tile_ids}
        print(f"  {split:<6} {len(tile_ids):>3} tiles  {len(scenes):>3} scenes")


if __name__ == "__main__":
    main()
