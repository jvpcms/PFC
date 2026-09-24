"""
Export every QGIS-inspectable artifact of the tiling + split into one GeoPackage.

GPKG rather than GeoJSON: a single file, many layers, far more compact — train_eligible
alone is ~38k polygons.

Layers:
  scene_bbox       axis-aligned bounding box per scene — the WRONG basis the original
                   tiler used. Load beside scene_swath to see the bug: ~32% of each bbox
                   is NoData corner. STAC metadata reports exactly this bbox, which is why
                   no metadata source could reveal the true shape.
  scene_swath      TRUE valid-data footprint per scene, traced from the 8m MS band.
                   Mean 67.7% of bbox (min 62.4%, max 69.9%).
  gaps             coverable area (Pampa AND imagery) that NO tile covers: 745 km2,
                   0.39%, in 42 polygons. These are swath seams where adjacent scenes
                   overlap in a band narrower than a couple of tiles, so no single scene
                   fully contains a tile. NOT fixable by downloading (every catalogued
                   path/row is held) and deliberately left open — closing them would need
                   partial tiles or cross-scene mosaicking, both rejected. See logbook
                   2026-08-16.
  tiles_all        the rebuilt grid: inside the swath, overlaps resolved, Pampa-trimmed
  tiles_removed    in the OLD grid but not the new — no/insufficient imagery, or newly
                   redundant  (needs --old-tiles)
  tiles_recovered  in the NEW grid but not the old — wrongly dropped when duplicate
                   resolution compared bboxes and a "covering" neighbour was empty there
  test_eligible    every tile test may ever draw from
  train_eligible   every tile train may ever draw from
  buffer           >=BUFFER_M ring around ALL test_eligible; never annotated
  unpooled         qualifies for no class, never a candidate
  test_n<N>        first 0.2N tiles of the test pick order,  per --n
  train_n<N>       first 0.8N tiles of the train pick order, per --n

The buffer is identical at every N by construction: it surrounds the whole test-eligible
set, not just the tiles drawn, so growing the budget never re-sterilises train tiles.
Likewise test_n100 is a strict subset of test_n500 — the pick orders are prefix-stable.

Tile attributes: tile_id, scene_id, split, test_rank, train_rank, coverage, n_valid,
top_class, top_frac. `coverage` is the fraction of the tile inside its scene's valid swath
(1.0 for every surviving tile). `top_class`/`top_frac` are the dominant stratum and its
share of the whole tile — style by top_class to check that each stratum's picks land where
that stratum actually is.

Usage:
    .venv/bin/python train_scripts/export_split_masks.py
    .venv/bin/python train_scripts/export_split_masks.py --n 100 250 500 1000

Output:
    data/inpe/tile_split_masks.gpkg
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

N_CLASSES = 12
CLASS_NAMES = {
    1: 'Forest', 2: 'Agriculture', 3: 'Rangeland', 4: 'RockyOutcrop',
    5: 'PortoAlegre', 6: 'Urban', 7: 'Mining', 8: 'MajorLagoons',
    9: 'Water', 10: 'Aquaculture', 11: 'ShoreSand', 12: 'InlandSand',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split',       default='data/inpe/tile_split.csv')
    ap.add_argument('--composition', default='data/inpe/tile_strata_composition.csv')
    ap.add_argument('--tiles',       default='data/inpe/tiles_valid_swath.geojson')
    ap.add_argument('--footprints',  default='data/inpe/scene_valid_footprints.gpkg')
    ap.add_argument('--old-tiles',   default='data/inpe/tiles_strictly_covering_pampa.geojson',
                    help='previous grid, for the removed/recovered diff layers')
    ap.add_argument('--pampa',       default='data/inpe/vectorized_pampa_outline.geojson')
    ap.add_argument('--out',         default='data/inpe/tile_split_masks.gpkg')
    ap.add_argument('--n', type=int, nargs='*', default=[100, 250, 500, 1000])
    args = ap.parse_args()

    split = pd.read_csv(args.split)
    comp = pd.read_csv(args.composition)
    C = [f'c{k:02d}' for k in range(1, N_CLASSES + 1)]
    frac = comp[C].to_numpy() / np.maximum(comp['n_total'].to_numpy()[:, None], 1)
    top = frac.argmax(axis=1) + 1
    comp = comp.assign(
        top_class=[CLASS_NAMES[k] if frac[i].max() > 0 else 'none' for i, k in enumerate(top)],
        top_frac=frac.max(axis=1).round(4),
    )

    g = gpd.read_file(args.tiles)[['tile_id', 'scene_id', 'coverage', 'geometry']]
    g = g.merge(split.drop(columns=['scene_id'], errors='ignore'), on='tile_id').merge(
        comp[['tile_id', 'n_valid', 'top_class', 'top_frac']], on='tile_id')
    print(f'{len(g):,} tiles, crs={g.crs}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()          # GPKG appends layers, so start clean

    layers = []

    # --- scene geometry: the wrong basis and the right one, side by side ---
    fp = gpd.read_file(args.footprints)
    layers.append(('scene_swath', fp))
    bbox = fp.copy()
    bbox['geometry'] = fp.geometry.envelope
    layers.append(('scene_bbox', bbox))

    # --- uncovered seams: coverable area no tile reaches ---
    M = 'EPSG:5880'
    pam = gpd.read_file(args.pampa).to_crs(M)
    target = unary_union(pam.geometry.values).intersection(unary_union(fp.to_crs(M).geometry.values))
    gap = target.difference(unary_union(g.to_crs(M).geometry.values))
    parts = sorted((x for x in (gap.geoms if hasattr(gap, 'geoms') else [gap]) if x.area > 1e4),
                   key=lambda x: -x.area)
    if parts:
        gl = gpd.GeoDataFrame({'km2': [x.area / 1e6 for x in parts]}, geometry=parts, crs=M)
        layers.append(('gaps', gl.to_crs('EPSG:4326')))
        print(f'  gaps: {gap.area/1e6:,.0f} km2 = {100*gap.area/target.area:.2f}% of the '
              f'coverable {target.area/1e6:,.0f} km2, in {len(parts)} polygons')

    # --- the grid, and how it changed ---
    layers.append(('tiles_all', g))
    if args.old_tiles and Path(args.old_tiles).exists():
        old = gpd.read_file(args.old_tiles)[['tile_id', 'scene_id', 'geometry']]
        O, N = set(old.tile_id), set(g.tile_id)
        layers.append(('tiles_removed', old[old.tile_id.isin(O - N)]))
        layers.append(('tiles_recovered', g[g.tile_id.isin(N - O)]))

    # --- eligibility ---
    for name in ('test_eligible', 'train_eligible', 'buffer', 'unpooled'):
        layers.append((name, g[g['split'] == name]))

    # --- the actual picks at each N ---
    for N_ in args.n:
        nte, ntr = int(round(0.2 * N_)), int(round(0.8 * N_))
        te = g[g['test_rank'].notna()].nsmallest(nte, 'test_rank')
        tr = g[g['train_rank'].notna()].nsmallest(ntr, 'train_rank')
        if len(te) < nte or len(tr) < ntr:
            print(f'  N={N_}: pool exhausted ({len(te)}/{nte} test, {len(tr)}/{ntr} train)')
        layers.append((f'test_n{N_}', te))
        layers.append((f'train_n{N_}', tr))

    for name, sub in layers:
        if sub is None or sub.empty:
            print(f'  {name:<18} EMPTY, skipped')
            continue
        sub.to_file(out, layer=name, driver='GPKG')
        extra = ''
        if 'top_class' in sub.columns:
            cls = sub['top_class'].value_counts()
            extra = '  top_class: ' + ', '.join(f'{k}={v}' for k, v in cls.head(3).items())
        print(f'  {name:<18} {len(sub):>6,}{extra}')

    print(f'\n-> {out}')
    print('In QGIS: Layer > Add Layer > Add Vector Layer, pick the .gpkg, select all layers.')
    print('To see the swath bug: scene_bbox under scene_swath — the difference is the')
    print('NoData corner the original tiler laid tiles into.')


if __name__ == '__main__':
    main()
