"""
Export pansharpened CBERS-4A patches as uint8 PNG for CVAT annotation.

Input: scene list JSON written by INPE_Scene_Browser.ipynb (Step 4), pointing
at BDC CB4A-WPM-L4-DN-1 band assets. Bands are streamed via /vsicurl windowed
reads — no scene download. SFIM pansharpening (PAN 2m + MS 8m) done locally:
the BDC PCA-FUSED product has ~8m effective detail and is not used for training.

WPM band layout: BAND0 = PAN 2m, BAND1/2/3/4 = Blue/Green/Red/NIR 8m.
PAN and MS share the same L4 ortho grid → co-registered by construction.

Usage:
    python train_scripts/export_annotation_patches.py
    python train_scripts/export_annotation_patches.py --scenes data/inpe/selected_scenes.json
                                                      --out data/cvat_patches
                                                      --patch-px 1024 --n-patches 30
                                                      --nodata-thresh 0.10

Output (per patch, grouped per scene):
    <out>/<scene_id>/<scene_id>_r<row>_c<col>_sat.png   ← upload to CVAT
    <out>/<scene_id>/<scene_id>_r<row>_c<col>_meta.json ← kept locally for mask tiling later
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

load_dotenv()

GAMMA = 0.45
RATIO = 4        # PAN 2m / MS 8m
EPS   = 1e-6

RGB_BANDS = ['BAND3', 'BAND2', 'BAND1']   # Red, Green, Blue

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    GDAL_HTTP_MULTIRANGE='YES',
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
)


def vsicurl(href: str) -> str:
    token = os.environ.get('BDC_ACCESS_TOKEN')
    return f'/vsicurl/{href}' + (f'?access_token={token}' if token else '')


def compute_scene_stats(rgb_urls: list[str], n_windows: int = 20,
                        win_px: int = 512, seed: int = 42) -> tuple:
    """Scene-level p2/p98 per band from sampled windows (bands have no
    overviews — a decimated full read over HTTP would fetch the whole file)."""
    rng = np.random.default_rng(seed)
    samples = [[] for _ in range(3)]
    for i, url in enumerate(rgb_urls):
        with rasterio.open(url) as src:
            W, H = src.width, src.height
            cols = rng.integers(0, W - win_px, n_windows)
            rows = rng.integers(0, H - win_px, n_windows)
            for c, r in zip(cols, rows):
                a = src.read(1, window=Window(int(c), int(r), win_px, win_px))
                a = a[a > 0]
                if a.size:
                    samples[i].append(a)
    scale = np.empty(3)
    floor = np.empty(3)
    for i in range(3):
        allpx = np.concatenate(samples[i])
        scale[i] = np.percentile(allpx, 98)
        floor[i] = np.percentile(allpx,  2)
    return scale, floor


def sfim_patch(pan_url: str, rgb_urls: list[str],
               col: int, row: int, patch_px: int) -> tuple:
    """Returns ((3, patch_px, patch_px) float32 pansharpened DN, nodata mask)."""
    pan_win = Window(col, row, patch_px, patch_px)
    with rasterio.open(pan_url) as src:
        pan = src.read(1, window=pan_win).astype(np.float32)

    ms_win = Window(col / RATIO, row / RATIO, patch_px / RATIO, patch_px / RATIO)
    ms = np.empty((3, patch_px, patch_px), np.float32)
    for i, url in enumerate(rgb_urls):
        with rasterio.open(url) as src:
            ms[i] = src.read(1, window=ms_win,
                             out_shape=(patch_px, patch_px),
                             resampling=Resampling.bilinear).astype(np.float32)

    nodata = (pan == 0) | np.any(ms == 0, axis=0)
    if nodata.mean() > 0.99:
        return None, nodata

    pan_lp = uniform_filter(pan, size=RATIO * 2)
    sharp  = ms * (pan / (pan_lp + EPS))
    sharp[:, nodata] = 0
    return sharp, nodata


def to_uint8(sharp: np.ndarray, scene_scale: np.ndarray,
             scene_floor: np.ndarray) -> np.ndarray:
    """(3,H,W) float DN → (H,W,3) uint8. Per-scene p2–p98 stretch + gamma.
    Inference preprocessing must apply the identical transform."""
    img = np.moveaxis(sharp, 0, -1).astype(np.float32)
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = np.clip(
            (img[:, :, c] - scene_floor[c]) / (scene_scale[c] - scene_floor[c]),
            0, 1)
    out = np.power(out, GAMMA)
    return (out * 255).astype(np.uint8)


def sample_patch_origins(pan_url: str, patch_px: int, n: int,
                         nodata_thresh: float, seed: int = 42) -> list[tuple]:
    """Grid-sample candidate origins, return up to n with low nodata."""
    with rasterio.open(pan_url) as src:
        W, H = src.width, src.height

        rng = np.random.default_rng(seed)
        # candidate grid: stride ~= sqrt(area / n) to spread evenly
        area = (W - patch_px) * (H - patch_px)
        stride = max(patch_px, int(np.sqrt(area / (n * 4))))
        cols = np.arange(0, W - patch_px, stride)
        rows = np.arange(0, H - patch_px, stride)
        candidates = [(int(c), int(r)) for r in rows for c in cols]
        rng.shuffle(candidates)

        accepted = []
        for col, row in candidates:
            if len(accepted) >= n:
                break
            win  = Window(col, row, patch_px, patch_px)
            band = src.read(1, window=win)
            if (band == 0).mean() <= nodata_thresh:
                accepted.append((col, row))

    return accepted


def export_scene(scene: dict, out_dir: Path, patch_px: int,
                 n_patches: int, nodata_thresh: float) -> int:
    scene_id = scene['scene_id']
    pan_url  = vsicurl(scene['assets']['BAND0'])
    rgb_urls = [vsicurl(scene['assets'][b]) for b in RGB_BANDS]

    out_dir = out_dir / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Scene : {scene_id}')

    print('  computing scene normalization stats (sampled windows)...')
    scene_scale, scene_floor = compute_scene_stats(rgb_urls)
    print(f'  scale (p98 R/G/B): {scene_scale.round(1)}')
    print(f'  floor (p2  R/G/B): {scene_floor.round(1)}')

    origins = sample_patch_origins(pan_url, patch_px, n_patches, nodata_thresh)
    print(f'  {len(origins)} valid origins (nodata ≤ {nodata_thresh:.0%})')

    n_saved = 0
    for idx, (col, row) in enumerate(origins):
        tag = f'{scene_id}_r{row:05d}_c{col:05d}'
        print(f'  [{idx+1:02d}/{len(origins)}] {tag}', end='  ', flush=True)

        sharp, nodata = sfim_patch(pan_url, rgb_urls, col, row, patch_px)
        if sharp is None:
            print('skip (all nodata)')
            continue

        img_u8 = to_uint8(sharp, scene_scale, scene_floor)
        img_u8[nodata] = 0

        img_path = out_dir / f'{tag}_sat.png'
        Image.fromarray(img_u8).save(img_path)

        meta = {
            'scene_id':    scene_id,
            'col':         col,
            'row':         row,
            'patch_px':    patch_px,
            'assets':      scene['assets'],
            'scene_scale': scene_scale.tolist(),
            'scene_floor': scene_floor.tolist(),
            'gamma':       GAMMA,
            'nodata_frac': float(nodata.mean()),
        }
        (out_dir / f'{tag}_meta.json').write_text(json.dumps(meta, indent=2))
        print(f'nodata={nodata.mean()*100:.1f}%  → {img_path.name}')
        n_saved += 1

    return n_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes',        default='data/inpe/selected_scenes.json',
                    help='scene list JSON written by INPE_Scene_Browser.ipynb')
    ap.add_argument('--out',           default='data/cvat_patches')
    ap.add_argument('--patch-px',      type=int, default=1024)
    ap.add_argument('--n-patches',     type=int, default=30,
                    help='patches per scene')
    ap.add_argument('--nodata-thresh', type=float, default=0.10,
                    help='skip patch if nodata fraction > this')
    args = ap.parse_args()

    scenes = json.loads(Path(args.scenes).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'{len(scenes)} scene(s)\n')

    with rasterio.Env(**GDAL_ENV):
        total = sum(export_scene(s, out_dir, args.patch_px,
                                 args.n_patches, args.nodata_thresh)
                    for s in scenes)

    print(f'\nDone. {total} patches in {out_dir}/')
    print('Upload *_sat.png files to CVAT task.')
    print('Keep *_meta.json files — needed for mask tiling after annotation.')


if __name__ == '__main__':
    main()
