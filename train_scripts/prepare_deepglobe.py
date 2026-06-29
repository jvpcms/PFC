"""
Download DeepGlobe dataset from Kaggle and tile into 256x256 NPY patches at 2m resolution.

Pipeline:
  2448px @0.5m  →  BOX downsample  →  612px @2m
  612px image   →  4 corner tiles of 256px (100px cross discarded in center)

Usage:
    python train_scripts/prepare_deepglobe.py
    python train_scripts/prepare_deepglobe.py --tile-size 256 --workers 8
"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image

DATA_DIR   = Path('data/deepglobe')
TILES_DIR  = DATA_DIR / 'tiles_2m'
TARGET_PX  = 612   # 2448px @0.5m → 612px @2m (4× area)
N_CLASSES  = 7

CLASS_COLORS = np.array([
    (0,   255, 255),  # Urban
    (255, 255, 0),    # Agriculture
    (255, 0,   255),  # Rangeland
    (0,   255, 0),    # Forest
    (0,   0,   255),  # Water
    (255, 255, 255),  # Barren
    (0,   0,   0),    # Unknown
], dtype=np.uint8)


def _tile_one_2m(args):
    sat_path, out_dir, tile_size, target_px = args
    img_id    = Path(sat_path).stem.replace('_sat', '')
    mask_path = Path(sat_path).parent / f'{img_id}_mask.png'

    img_2m = np.array(Image.open(sat_path).resize(
        (target_px, target_px), Image.Resampling.BOX))

    label_2m = None
    if Path(mask_path).exists():
        mask_rgb   = np.array(Image.open(mask_path))
        label_full = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        for idx, rgb in enumerate(CLASS_COLORS):
            label_full[np.all(mask_rgb == rgb, axis=-1)] = idx

        h = w = target_px
        blocks  = label_full.reshape(h, 4, w, 4).swapaxes(1, 2).reshape(h * w, 16)
        one_hot = (blocks[:, :, None] == np.arange(N_CLASSES)[None, None, :])
        label_2m = one_hot.sum(axis=1).argmax(axis=1).reshape(h, w).astype(np.uint8)

    out_dir = Path(out_dir)
    offsets = [0, target_px - tile_size]  # corner offsets: [0, 356]
    for r, rs in enumerate(offsets):
        for c, cs in enumerate(offsets):
            stem = f'{img_id}_{r:02d}_{c:02d}'
            np.save(out_dir / f'{stem}_sat.npy',  img_2m[rs:rs + tile_size, cs:cs + tile_size])
            if label_2m is not None:
                np.save(out_dir / f'{stem}_mask.npy', label_2m[rs:rs + tile_size, cs:cs + tile_size])


def tile_split(split: str, tile_size: int, n_workers: int):
    src = DATA_DIR / split
    if not src.exists():
        print(f'{split}: source dir not found, skipping')
        return

    sat_files = sorted(src.glob('*_sat.jpg'))
    out_dir   = TILES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tiles_per_image = 4  # 2×2 corners
    expected = len(sat_files) * n_tiles_per_image
    existing = len(list(out_dir.glob('*_sat.npy')))
    if existing == expected:
        print(f'{split}: already tiled ({existing} tiles) — skipping')
        return

    print(f'{split}: tiling {len(sat_files)} images → {expected} tiles ({n_workers} workers)...')
    args = [(str(p), str(out_dir), tile_size, TARGET_PX) for p in sat_files]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        ex.map(_tile_one_2m, args)

    n_saved = len(list(out_dir.glob('*_sat.npy')))
    print(f'{split}: {n_saved} tiles saved → {out_dir}')


def main(args):
    import kaggle

    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f'Dataset already present at {DATA_DIR} — skipping download.')
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        username  = os.environ.get('KAGGLE_USERNAME')
        api_token = os.environ.get('KAGGLE_API_TOKEN')
        if not username or not api_token:
            raise RuntimeError(
                'KAGGLE_USERNAME and KAGGLE_API_TOKEN env vars must both be set.'
            )

        print('Authenticating with Kaggle...')
        kaggle.api.authenticate()
        print('Downloading DeepGlobe dataset...')
        kaggle.api.dataset_download_files(
            'balraj98/deepglobe-land-cover-classification-dataset',
            path=str(DATA_DIR), unzip=True,
        )
        print(f'Downloaded and extracted to {DATA_DIR}')

    for split in ('train', 'valid', 'test'):
        p = DATA_DIR / split
        if p.exists():
            print(f'{split}: {len(list(p.glob("*_sat.jpg")))} images found')

    n_workers = args.workers or os.cpu_count()
    for split in ('train', 'valid', 'test'):
        tile_split(split, args.tile_size, n_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and tile DeepGlobe at 2m resolution.')
    parser.add_argument('--tile-size', type=int, default=256,
                        help='Tile size in pixels (default: 256)')
    parser.add_argument('--workers',  type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    main(parser.parse_args())
