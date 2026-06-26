"""
Download DeepGlobe dataset from Kaggle and tile into 256x256 NPY patches.

Usage:
    python train_scripts/prepare_deepglobe.py
    python train_scripts/prepare_deepglobe.py --tile-size 512 --n-tiles 4
"""

import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image

DATA_DIR  = Path('data/deepglobe')
TILES_DIR = DATA_DIR / 'tiles'

CLASS_COLORS = np.array([
    (0,   255, 255),  # Urban
    (255, 255, 0),    # Agriculture
    (255, 0,   255),  # Rangeland
    (0,   255, 0),    # Forest
    (0,   0,   255),  # Water
    (255, 255, 255),  # Barren
    (0,   0,   0),    # Unknown
], dtype=np.uint8)


def _tile_one(args):
    sat_path, out_dir, tile_size, n_tiles = args
    img_id    = sat_path.stem.replace('_sat', '')
    mask_path = sat_path.parent / f'{img_id}_mask.png'

    img   = np.array(Image.open(sat_path))
    label = None
    if mask_path.exists():
        mask_rgb = np.array(Image.open(mask_path))
        label = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        for idx, rgb in enumerate(CLASS_COLORS):
            label[np.all(mask_rgb == rgb, axis=-1)] = idx

    out_dir = Path(out_dir)
    for r in range(n_tiles):
        for c in range(n_tiles):
            rs, cs = r * tile_size, c * tile_size
            stem = f'{img_id}_{r:02d}_{c:02d}'
            np.save(out_dir / f'{stem}_sat.npy',  img[rs:rs+tile_size, cs:cs+tile_size])
            if label is not None:
                np.save(out_dir / f'{stem}_mask.npy', label[rs:rs+tile_size, cs:cs+tile_size])


def tile_split(split: str, tile_size: int, n_tiles: int, n_workers: int):
    src = DATA_DIR / split
    if not src.exists():
        print(f'{split}: source dir not found, skipping')
        return

    sat_files = sorted(src.glob('*_sat.jpg'))
    out_dir   = TILES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = len(sat_files) * n_tiles ** 2
    existing = len(list(out_dir.glob('*_sat.npy')))
    if existing == expected:
        print(f'{split}: already tiled ({existing} tiles) — skipping')
        return

    print(f'{split}: tiling {len(sat_files)} images → {expected} tiles ({n_workers} workers)...')
    args = [(p, str(out_dir), tile_size, n_tiles) for p in sat_files]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        ex.map(_tile_one, args)

    n_saved = len(list(out_dir.glob('*_sat.npy')))
    print(f'{split}: {n_saved} tiles saved → {out_dir}')


def main(args):
    import kaggle

    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f'Dataset already present at {DATA_DIR} — skipping download.')
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        username = os.environ.get('KAGGLE_USERNAME')
        api_key  = os.environ.get('KAGGLE_API_TOKEN')
        if not username or not api_key:
            raise RuntimeError(
                'KAGGLE_USERNAME and KAGGLE_API_TOKEN env vars must both be set.'
            )
        os.environ['KAGGLE_KEY'] = api_key

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
        tile_split(split, args.tile_size, args.n_tiles, n_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and tile DeepGlobe dataset.')
    parser.add_argument('--tile-size', type=int, default=256,
                        help='Tile size in pixels (default: 256)')
    parser.add_argument('--n-tiles',  type=int, default=9,
                        help='Number of tiles per side (default: 9, gives 9x9=81 tiles per image)')
    parser.add_argument('--workers',  type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    main(parser.parse_args())
