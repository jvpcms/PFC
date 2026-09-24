"""
Per-tile stratum composition vector — pixel counts per stratum class for every tile.

Input:
  - strata raster (EPSG:4326, Byte, 1..12, NoData=0) built in QGIS: MapBiomas reclass
    scaffold patched with the manual sub-split overrides (Porto Alegre, Major Lagoons,
    shore/inland sand) via GRASS r.patch.
  - tile grid GeoJSON from INPE_Scene_Browser.ipynb (Step 1d), reprojected to EPSG:4326.
    Tiles are 1024x1024px @2m (2048m) squares built in each scene's native UTM, so in
    EPSG:4326 they are slightly skewed quadrilaterals (~2px of skew at 30m strata
    resolution) — masked by true polygon, not by bounding box.

Counts, not fractions: fractions are trivially derived, but absolute pixel counts are what
the rare-class presence thresholds are actually expressed in ("this tile has 47 Aquaculture
pixels"). A 2048m tile covers ~80x68 strata pixels at 30m.

Fraction convention — always divide by `n_total` (the WHOLE tile), never by `n_valid`:
NoData here is not a data defect, it is land outside the Pampa clip (or ocean); every such
tile is still real imagery an annotator can label, so a tile with enough of a class is an
honest representative of it even when a third of it is NoData. But `n_valid` is a shrinking
denominator, so dividing by it inflates purity for boundary tiles — a tile with 1 classified
pixel scores frac=1.0 and outranks a genuine 5,520-pixel tile. Dividing by `n_total` fixes
that and makes fully-NoData tiles score 0.0 for every class, so they drop out of any
composition-driven selection on their own with no filtering step.

Verified (2026-08-12): NoData tiles are all boundary tiles, no interior holes. The raw
MapBiomas source is itself NoData under them, and the 23 tiles with n_valid==0 overlap the
Pampa polygon by 2-2,762 m2 (<= 3.1 strata pixels) — they survived the QGIS clip because
geometric `intersects` accepts an infinitesimal touch, while the pixel-center masking below
needs a pixel centre inside.

Usage:
    .venv/bin/python train_scripts/tile_strata_composition.py
    .venv/bin/python train_scripts/tile_strata_composition.py --limit 500   # smoke test

Output:
    data/inpe/tile_strata_composition.csv
      tile_id, scene_id, n_total, n_valid, n_nodata, c01..c12
      n_total  = all in-polygon pixels           <- fraction denominator
      n_valid  = in-polygon pixels with a real class (1..12)
      n_nodata = in-polygon pixels outside the strata raster's data area (class 0)
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

N_CLASSES = 12

CLASS_NAMES = {
    1:  'Forest',
    2:  'Agriculture',
    3:  'Rangeland (remaining)',
    4:  'Rocky Outcrop',
    5:  'Porto Alegre',
    6:  'Urban (remaining)',
    7:  'Mining',
    8:  'Major Lagoons',
    9:  'Water/rivers (remaining)',
    10: 'Aquaculture',
    11: 'Shore sand',
    12: 'Inland sand',
}


def tile_windows(gdf, transform, raster_w, raster_h):
    """Pixel window (col_off, row_off, w, h) per tile, rounded outward and clamped."""
    inv = ~transform
    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        # inv maps (x, y) -> (col, row); north-up raster so maxy is the top row
        c0, r0 = inv * (minx, maxy)
        c1, r1 = inv * (maxx, miny)
        col_off = max(0, int(np.floor(c0)))
        row_off = max(0, int(np.floor(r0)))
        col_end = min(raster_w, int(np.ceil(c1)))
        row_end = min(raster_h, int(np.ceil(r1)))
        yield geom, Window(col_off, row_off,
                           max(0, col_end - col_off), max(0, row_end - row_off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strata', default='qgis_projects/pampa_layers/strata_leaf_classes.tif')
    ap.add_argument('--tiles',  default='data/inpe/tiles_valid_swath.geojson')
    ap.add_argument('--out',    default='data/inpe/tile_strata_composition.csv')
    ap.add_argument('--limit',  type=int, default=0, help='process only the first N tiles')
    args = ap.parse_args()

    print(f'reading tiles  : {args.tiles}')
    gdf = gpd.read_file(args.tiles)
    if args.limit:
        gdf = gdf.iloc[:args.limit].copy()
    print(f'  {len(gdf)} tiles, crs={gdf.crs}')

    print(f'reading strata : {args.strata}')
    with rasterio.open(args.strata) as src:
        if gdf.crs != src.crs:
            print(f'  reprojecting tiles {gdf.crs} -> {src.crs}')
            gdf = gdf.to_crs(src.crs)
        base_transform = src.transform
        raster_w, raster_h = src.width, src.height
        # Block=width x 1 (full scanlines): a windowed read pulls whole rows off disk, so
        # ~50k small windowed reads would be ~100GB of I/O. 660MB as uint8 fits in RAM.
        print(f'  loading {raster_w}x{raster_h} ({raster_w * raster_h / 1e6:.0f} MB uint8) into RAM...')
        arr = src.read(1)
    print('  loaded')

    counts = np.zeros((len(gdf), N_CLASSES + 1), dtype=np.int32)   # col 0 = NoData
    empty = 0

    for i, (geom, win) in enumerate(tile_windows(gdf, base_transform, raster_w, raster_h)):
        if win.width == 0 or win.height == 0:
            empty += 1
            continue
        sub = arr[win.row_off:win.row_off + win.height,
                  win.col_off:win.col_off + win.width]
        mask = geometry_mask(
            [geom],
            out_shape=sub.shape,
            transform=window_transform(win, base_transform),
            invert=True,          # True inside the polygon
            all_touched=False,    # pixel-center rule -> tiles don't double-count shared edges
        )
        vals = sub[mask]
        if vals.size:
            counts[i] = np.bincount(vals, minlength=N_CLASSES + 1)[:N_CLASSES + 1]
        else:
            empty += 1

        if (i + 1) % 5000 == 0:
            print(f'  {i + 1}/{len(gdf)} tiles')

    if empty:
        print(f'  {empty} tiles fell fully outside the strata raster extent')

    out = pd.DataFrame({
        'tile_id':  gdf['tile_id'].values,
        'scene_id': gdf['scene_id'].values,
        'n_total':  counts.sum(axis=1),
        'n_valid':  counts[:, 1:].sum(axis=1),
        'n_nodata': counts[:, 0],
    })
    for k in range(1, N_CLASSES + 1):
        out[f'c{k:02d}'] = counts[:, k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'\n-> {out_path}  ({len(out)} rows)')

    # --- summary: what the rarity ordering and presence thresholds will be built from ---
    total_valid = int(out['n_valid'].sum())
    print(f'\n{total_valid:,} in-tile classified pixels '
          f'({out["n_nodata"].sum():,} NoData)')
    print('fractions below are over the whole tile (n_total), see module docstring\n')
    print(f'{"":>4} {"class":<26} {"pixels":>13} {"area%":>7} '
          f'{">=1px":>7} {">=1%":>7} {">=10%":>7} {"major":>7}')
    frac = out[[f'c{k:02d}' for k in range(1, N_CLASSES + 1)]].to_numpy() / \
        np.maximum(out['n_total'].to_numpy()[:, None], 1)
    # a fully-NoData tile is all-zero -> argmax would report class 1; exclude it
    major = np.where(frac.max(axis=1) > 0, frac.argmax(axis=1) + 1, 0)
    for k in range(1, N_CLASSES + 1):
        col = out[f'c{k:02d}'].to_numpy()
        print(f'{k:>4} {CLASS_NAMES[k]:<26} {col.sum():>13,} '
              f'{100 * col.sum() / total_valid:>6.2f}% '
              f'{(col > 0).sum():>7,} {(frac[:, k - 1] >= 0.01).sum():>7,} '
              f'{(frac[:, k - 1] >= 0.10).sum():>7,} {(major == k).sum():>7,}')

    print('\nrarest -> commonest by total pixels: ' +
          ' '.join(str(k) for k in np.argsort(
              [out[f'c{k:02d}'].sum() for k in range(1, N_CLASSES + 1)]) + 1))


if __name__ == '__main__':
    main()
