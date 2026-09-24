"""
Rebuild the tile grid against each scene's TRUE valid swath.

Why: the original grid (INPE_Scene_Browser.ipynb Step 1d) laid tiles over `src.bounds`,
the axis-aligned bounding box. But CBERS-4A L4 products are orthorectified onto a
north-up UTM grid with the imaged swath sitting inside as a ~13deg-rotated rectangle, so
~32% of every bbox is NoData. Tiles in those corners have perfectly good MapBiomas strata
(which covers the whole biome) but no imagery at all, so nothing downstream caught it --
measured, 1.1% of surviving tiles had ZERO imagery and 2.4% were not fully covered.

STAC metadata cannot detect this either: its footprint IS the bbox (13,520 vs 13,521 km2
measured), so the true shape only exists in the pixels. `scene_valid_footprints.gpkg`
carries it, traced from the 8m MS band.

Tiles stay axis-aligned squares. The pixel grid is already north-up (geotransform rotation
terms are exactly 0), so tiles are exact pixel blocks needing no resampling; only the grid
EXTENT was wrong. Rotating tiles to follow the swath would force interpolation of every
tile to recover ~5% more area, which is a bad trade.

Grid origin and indexing are unchanged, so `tile_id` -> pixel window still holds:
    i, j = tile_id.rsplit('_', 2)[-2:];  col = i * 1024;  row = j * 1024

Pipeline reproduced from the original, with the swath check inserted:
  1. per-scene grid from the bbox top-left, full tiles only
  2. NEW: keep tiles whose area is >= MIN_COVERAGE inside the valid swath
  3. cross-scene overlap resolution: drop a tile only if FULLY covered by the union of
     its closer-to-scene-centre neighbours (a pure distance rule discards the loser's
     non-overlapping sliver too)
  4. trim to the Pampa biome by INTERSECTS

Usage:
    .venv/bin/python train_scripts/retile_valid_swath.py
    .venv/bin/python train_scripts/retile_valid_swath.py --min-coverage 0.99

Output:
    data/inpe/tiles_valid_swath.geojson   tile_id, scene_id, path, row, dist_to_center,
                                          coverage, geometry   (EPSG:4326)
"""

import argparse
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

TILE_PX = 1024
RES_M = 2.0
TILE_M = TILE_PX * RES_M      # 2048
COMMON_CRS = 'EPSG:5880'      # SIRGAS 2000 / Brazil Polyconic, metric, for the overlap test
RAW = '/media/jvpcms/9a4d7913-803b-4816-9d70-c54550ae61b8/cbers_work/raw_inpe_data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--footprints', default='data/inpe/scene_valid_footprints.gpkg')
    ap.add_argument('--pampa', default='data/inpe/vectorized_pampa_outline.geojson')
    ap.add_argument('--out', default='data/inpe/tiles_valid_swath.geojson')
    ap.add_argument('--min-coverage', type=float, default=1.0,
                    help='fraction of a tile that must lie inside the valid swath')
    args = ap.parse_args()

    fp = gpd.read_file(args.footprints)
    print(f'{len(fp)} scene footprints | mean valid {fp.frac.mean():.1%}')

    # --- 1+2: per-scene grid, clipped to the valid swath, in each scene's native UTM ---
    rows = []
    for n, (_, f) in enumerate(fp.iterrows(), 1):
        sid = f.scene_id
        pan = glob.glob(f'{RAW}/{sid}/*BAND0*')
        if not pan:
            print(f'  [{n}/{len(fp)}] {sid}: no BAND0, skipped')
            continue
        with rasterio.open(pan[0]) as src:
            b, crs = src.bounds, src.crs
        swath = gpd.GeoSeries([f.geometry], crs=fp.crs).to_crs(crs).iloc[0]
        pswath = prep(swath)
        nx = int((b.right - b.left) // TILE_M)
        ny = int((b.top - b.bottom) // TILE_M)
        cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        kept = 0
        for i in range(nx):
            for j in range(ny):
                x0 = b.left + i * TILE_M
                y1 = b.top - j * TILE_M
                t = box(x0, y1 - TILE_M, x0 + TILE_M, y1)
                # cheap reject first: most tiles are fully in or fully out
                if not pswath.intersects(t):
                    continue
                cov = 1.0 if pswath.contains(t) else t.intersection(swath).area / t.area
                if cov + 1e-9 < args.min_coverage:
                    continue
                rows.append(dict(scene_id=sid, path=int(sid.split('_')[4]),
                                 row=int(sid.split('_')[5]), crs=str(crs),
                                 dist_to_center=float(np.hypot(x0 + TILE_M / 2 - cx,
                                                               y1 - TILE_M / 2 - cy)),
                                 coverage=float(cov), tile_id=f'{sid}_{i}_{j}',
                                 geometry=t))
                kept += 1
        print(f'  [{n}/{len(fp)}] {sid}: {kept:,} tiles kept of {nx*ny:,} in bbox '
              f'({100*kept/(nx*ny):.0f}%)', flush=True)

    df = pd.DataFrame(rows)
    print(f'\n{len(df):,} candidate tiles inside the valid swaths')

    tiles = pd.concat([
        gpd.GeoDataFrame(sub.drop(columns='crs'), geometry='geometry', crs=crs).to_crs(COMMON_CRS)
        for crs, sub in df.groupby('crs')], ignore_index=True)
    tiles = gpd.GeoDataFrame(tiles, geometry='geometry', crs=COMMON_CRS)

    # --- 3: cross-scene overlap resolution ---
    j = gpd.sjoin(tiles, tiles, how='inner', predicate='intersects', lsuffix='a', rsuffix='b')
    j = j[j['scene_id_a'] != j['scene_id_b']]
    geom = dict(zip(tiles['tile_id'], tiles['geometry']))
    closer = {}
    for a, b_, da, db in zip(j['tile_id_a'], j['tile_id_b'],
                             j['dist_to_center_a'], j['dist_to_center_b']):
        if da > db or (da == db and a > b_):
            closer.setdefault(a, []).append(b_)
    drop = set()
    for tid, nb in closer.items():
        if geom[tid].difference(unary_union([geom[x] for x in nb])).area <= 1e-6:
            drop.add(tid)
    tiles = tiles[~tiles['tile_id'].isin(drop)].copy()
    print(f'{len(drop):,} dropped as fully redundant -> {len(tiles):,} survive')

    # --- 4: trim to the Pampa biome (INTERSECTS, matching the original QGIS step) ---
    pampa = gpd.read_file(args.pampa).to_crs(COMMON_CRS).geometry.union_all()
    pp = prep(pampa)
    tiles = tiles[[pp.intersects(g) for g in tiles.geometry]].copy()
    print(f'{len(tiles):,} tiles intersect the Pampa biome')

    out = tiles.to_crs('EPSG:4326')[['tile_id', 'scene_id', 'path', 'row',
                                     'dist_to_center', 'coverage', 'geometry']]
    out.to_file(args.out, driver='GeoJSON')
    print(f'\n-> {args.out}')
    print(f'   coverage: min {out.coverage.min():.4f}  '
          f'fully covered {int((out.coverage > 0.9999).sum()):,}/{len(out):,}')


if __name__ == '__main__':
    main()
