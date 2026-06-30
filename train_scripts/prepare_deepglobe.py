"""
Download DeepGlobe dataset from Kaggle and tile into 256x256 NPY patches.

Reads from data/deepglobe/train/ (only labeled split).
Applies image-level 85/15 split → tiles/train/ and tiles/val/ (or tiles_2m/ when --downsample).
Split is deterministic: sorted image IDs + fixed seed, stable across runs.

Usage:
    python train_scripts/prepare_deepglobe.py                   # 0.5m, 9×9 grid → tiles/
    python train_scripts/prepare_deepglobe.py --downsample      # 2m, 4 corner tiles → tiles_2m/
    python train_scripts/prepare_deepglobe.py --tile-size 256 --n-tiles 9
"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image

DATA_DIR          = Path('data/deepglobe')
VAL_SPLIT         = 0.15
SEED              = 42
DOWNSAMPLE_FACTOR = 4  # 0.5m → 2m

CLASS_COLORS = np.array([
    (0,   255, 255),  # Urban
    (255, 255, 0),    # Agriculture
    (255, 0,   255),  # Rangeland
    (0,   255, 0),    # Forest
    (0,   0,   255),  # Water
    (255, 255, 255),  # Barren
    (0,   0,   0),    # Unknown
], dtype=np.uint8)

N_CLASSES = len(CLASS_COLORS)


def _decode_mask(mask_rgb):
    label = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for idx, rgb in enumerate(CLASS_COLORS):
        label[np.all(mask_rgb == rgb, axis=-1)] = idx
    return label


def _tile_one(args):
    """0.5m path: sequential n_tiles×n_tiles grid."""
    sat_path, out_dir, tile_size, n_tiles = args
    img_id    = Path(sat_path).stem.replace('_sat', '')
    mask_path = Path(sat_path).parent / f'{img_id}_mask.png'

    img   = np.array(Image.open(sat_path))
    label = _decode_mask(np.array(Image.open(mask_path)))

    out_dir = Path(out_dir)
    for r in range(n_tiles):
        for c in range(n_tiles):
            rs, cs = r * tile_size, c * tile_size
            stem = f'{img_id}_{r:02d}_{c:02d}'
            np.save(out_dir / f'{stem}_sat.npy',  img[rs:rs+tile_size, cs:cs+tile_size])
            np.save(out_dir / f'{stem}_mask.npy', label[rs:rs+tile_size, cs:cs+tile_size])


def _tile_one_2m(args):
    """2m path: area-average image downsample, majority-vote mask, 4 corner tiles."""
    sat_path, out_dir, tile_size = args
    img_id    = Path(sat_path).stem.replace('_sat', '')
    mask_path = Path(sat_path).parent / f'{img_id}_mask.png'

    img   = np.array(Image.open(sat_path))                    # (H, W, 3)
    label = _decode_mask(np.array(Image.open(mask_path)))     # (H, W)

    H, W = img.shape[:2]
    new_h, new_w = H // DOWNSAMPLE_FACTOR, W // DOWNSAMPLE_FACTOR

    # Image: area-average via PIL BOX filter
    img_ds = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BOX))

    # Mask: majority vote — one-hot sum over each DOWNSAMPLE_FACTOR×DOWNSAMPLE_FACTOR block
    label_oh = np.eye(N_CLASSES, dtype=np.int32)[label]       # (H, W, C)
    label_ds = (
        label_oh
        .reshape(new_h, DOWNSAMPLE_FACTOR, new_w, DOWNSAMPLE_FACTOR, N_CLASSES)
        .sum(axis=(1, 3))
        .argmax(axis=-1)
        .astype(np.uint8)
    )

    # 4 corner tiles: offsets = [0, new_h - tile_size]
    offsets = [0, new_h - tile_size]
    out_dir = Path(out_dir)
    for r, rs in enumerate(offsets):
        for c, cs in enumerate(offsets):
            stem = f'{img_id}_{r:02d}_{c:02d}'
            np.save(out_dir / f'{stem}_sat.npy',  img_ds[rs:rs+tile_size, cs:cs+tile_size])
            np.save(out_dir / f'{stem}_mask.npy', label_ds[rs:rs+tile_size, cs:cs+tile_size])


def tile_all(tile_size: int, n_tiles: int, n_workers: int, downsample: bool):
    src = DATA_DIR / 'train'
    if not src.exists():
        raise FileNotFoundError(f'Source not found: {src}')

    sat_files = sorted(src.glob('*_sat.jpg'))
    img_ids   = [p.stem.replace('_sat', '') for p in sat_files]

    rng = np.random.default_rng(SEED)
    order = np.array(img_ids)
    rng.shuffle(order)

    n_val     = max(1, int(len(order) * VAL_SPLIT))
    val_set   = set(order[:n_val])
    train_set = set(order[n_val:])

    print(f'Images — train: {len(train_set)}  val: {len(val_set)}')

    if downsample:
        tiles_dir       = DATA_DIR / 'tiles_2m'
        tiles_per_image = 4           # 2×2 corner tiles
        worker_fn       = _tile_one_2m
    else:
        tiles_dir       = DATA_DIR / 'tiles'
        tiles_per_image = n_tiles ** 2
        worker_fn       = _tile_one

    for split_name, id_set in (('train', train_set), ('val', val_set)):
        out_dir     = tiles_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        split_files = [p for p in sat_files if p.stem.replace('_sat', '') in id_set]
        expected    = len(split_files) * tiles_per_image
        existing    = len(list(out_dir.glob('*_sat.npy')))

        if existing == expected:
            print(f'{split_name}: already tiled ({existing} tiles) — skipping')
            continue

        print(f'{split_name}: {len(split_files)} images → {expected} tiles ({n_workers} workers)...')

        if downsample:
            worker_args = [(str(p), str(out_dir), tile_size) for p in split_files]
        else:
            worker_args = [(str(p), str(out_dir), tile_size, n_tiles) for p in split_files]

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            ex.map(worker_fn, worker_args)

        n_saved = len(list(out_dir.glob('*_sat.npy')))
        print(f'{split_name}: {n_saved} tiles saved → {out_dir}')


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

    src = DATA_DIR / 'train'
    print(f'train: {len(list(src.glob("*_sat.jpg")))} labeled images found')

    n_workers = args.workers or os.cpu_count()
    tile_all(args.tile_size, args.n_tiles, n_workers, args.downsample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and tile DeepGlobe dataset.')
    parser.add_argument('--tile-size',  type=int,  default=256,
                        help='Tile size in pixels (default: 256)')
    parser.add_argument('--n-tiles',    type=int,  default=9,
                        help='Tiles per side for 0.5m grid (default: 9, ignored with --downsample)')
    parser.add_argument('--downsample', action='store_true',
                        help='Downsample 0.5m→2m (4× area avg image + majority-vote mask), output 4 corner tiles → tiles_2m/')
    parser.add_argument('--workers',    type=int,  default=None,
                        help='Parallel workers (default: cpu_count)')
    main(parser.parse_args())
