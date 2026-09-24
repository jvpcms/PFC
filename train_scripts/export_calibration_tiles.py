#!/usr/bin/env python3
"""Cut tiles that are NOT in the production selection, for the calibration round.

The annotator calibration round uses tiles deliberately outside the production
N=100, so that inter-annotator agreement can be measured without consuming or
revealing production tiles. This script cuts them with the identical rendering
pipeline, so annotators calibrate on images that look exactly like the ones they
will label.

The one subtlety is the stretch. ``export_annotation_tiles.py`` fits endpoints
per scene over that scene's *selected* tiles, and a calibration tile is by
definition not among them. Preferring the scene's production endpoints puts the
calibration tile on the exact transform its siblings use, and avoids the
per-tile contrast fit that production rejected -- a flat tile stretched over its
own narrow range turns sensor noise into visible static.

But production endpoints can be wildly wrong for a tile drawn from a different
part of the scene. All four production tiles of 206/152 are 100% water (R median
66-135); the barren calibration tile there sits at R median 289 with a 375 DN
span, and rendering it against water endpoints gave a 98% white rectangle.

So endpoints are chosen per tile by span, never by brightness:

* if the tile's own p2..p99.5 span is **wider** than the production span, use its
  own -- it is being stretched *less* than production would, so no noise is
  amplified, and it stops bright tiles clipping to white;
* otherwise use the production endpoints, which is the conservative case and
  preserves the no-amplification guarantee for flat tiles.

Gamma is still solved per tile, exactly as in production. Which basis was used is
recorded per tile in the manifest as ``endpoints_from``.

Output is a flat directory, separate from the production tiles so the two can
never be confused:

    <out>/<tile_id>.png     8-bit RGB, production rendering
    <out>/<tile_id>.tif     4-band Int16 DN with NIR, georeferenced
    <out>/manifest.csv      the rendering parameters actually used

Usage:
    .venv/bin/python train_scripts/export_calibration_tiles.py \
        CBERS_4A_WPM_20210827_211_150_L4_37_52 ... \
        --out-dir /path/to/cbers_work/calibration_tiles
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_worklist import DEFAULT_WORK_DIR, select_tiles
from export_annotation_tiles import (
    DEFAULT_FUSED_DIR,
    TILE_PX,
    TILES_GEOJSON,
    TILE_SPLIT_CSV,
    TileError,
    tile_ij,
    tile_window,
    write_png,
    write_tif,
)
from tile_stretch import describe, fit_endpoints, solve_gamma, to_uint8

DEFAULT_OUT = DEFAULT_WORK_DIR / "calibration_tiles"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tiles", nargs="+", help="tile_ids outside the production selection")
    ap.add_argument("--n", type=int, default=100, help="production N (default 100)")
    ap.add_argument("--fused-dir", type=Path, default=DEFAULT_FUSED_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    split = pd.read_csv(TILE_SPLIT_CSV)
    production = set(select_tiles(split, args.n)["tile_id"])
    known = dict(zip(split["tile_id"], split["coverage"])) if "coverage" in split else {}

    overlap = [t for t in args.tiles if t in production]
    if overlap:
        print(
            f"refusing: {len(overlap)} tile(s) are IN the production {args.n} "
            f"selection, e.g. {overlap[0]} -- calibration tiles must be outside it",
            file=sys.stderr,
        )
        return 1

    missing = [t for t in args.tiles if t not in known]
    if missing:
        print(f"not in tile_split.csv: {', '.join(missing)}", file=sys.stderr)
        return 1

    partial = [t for t in args.tiles if known[t] < 1.0]
    if partial:
        print(f"coverage < 1.0, would contain NoData: {', '.join(partial)}", file=sys.stderr)
        return 1

    geoms = gpd.read_file(TILES_GEOJSON).set_index("tile_id")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_scene: dict[str, list[str]] = {}
    for t in args.tiles:
        by_scene.setdefault(t.rsplit("_L4_", 1)[0] + "_L4", []).append(t)

    rows, failures = [], []
    for scene_id, tiles in sorted(by_scene.items()):
        fused = args.fused_dir / f"{scene_id}_FUSED.tif"
        if not fused.exists():
            failures.append(f"{scene_id}: not fused")
            continue

        prod_ids = list(select_tiles(split, args.n).query("scene_id == @scene_id")["tile_id"])
        if not prod_ids:
            failures.append(f"{scene_id}: no production tiles to fit endpoints on")
            continue

        with rasterio.open(fused) as src:
            # Endpoints from this scene's PRODUCTION tiles, so the calibration
            # tile lands on the same transform its siblings use.
            prod_geoms = geoms.loc[prod_ids].to_crs(src.crs)
            blocks = [
                src.read(window=tile_window(src, g, t))
                for t, g in prod_geoms.geometry.items()
            ]
            lo_prod, hi_prod = fit_endpoints(blocks)
            span_prod = float((hi_prod - lo_prod).mean())

            sub = geoms.loc[tiles].to_crs(src.crs)
            for tile_id, geom in sub.geometry.items():
                try:
                    window = tile_window(src, geom, tile_id)
                    data = src.read(window=window)
                    n_nodata = int((data == 0).all(axis=0).sum())
                    if n_nodata:
                        raise TileError(
                            f"{tile_id}: {n_nodata} NoData px "
                            f"({100 * n_nodata / (TILE_PX**2):.2f}%)"
                        )
                except TileError as exc:
                    failures.append(str(exc))
                    continue

                # Own endpoints only when they stretch the tile LESS than
                # production would; otherwise stay conservative.
                lo_own, hi_own = fit_endpoints([data])
                if float((hi_own - lo_own).mean()) > span_prod:
                    lo, hi, basis = lo_own, hi_own, "own tile (wider span)"
                else:
                    lo, hi, basis = lo_prod, hi_prod, f"production tiles of {scene_id}"

                gamma = solve_gamma(data[:3].astype("f4"), lo, hi)
                write_tif(args.out_dir / f"{tile_id}.tif", data, src, window)
                write_png(
                    args.out_dir / f"{tile_id}.png",
                    to_uint8(data[:3], lo, hi, gamma),
                    src,
                    window,
                )
                i, j = tile_ij(tile_id)
                rows.append(
                    {
                        "tile_id": tile_id,
                        "scene_id": scene_id,
                        "role": "calibration",
                        "i": i,
                        "j": j,
                        "crs": str(src.crs),
                        "col_off": int(window.col_off),
                        "row_off": int(window.row_off),
                        **describe(lo, hi, gamma),
                        "endpoints_from": basis,
                        "src_raster": fused.name,
                        "done_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                )
                print(f"{tile_id}  gamma {1 / gamma:.2f}")

    if rows:
        pd.DataFrame(rows).sort_values("tile_id").to_csv(
            args.out_dir / "manifest.csv", index=False
        )
        print(f"\nwrote {len(rows)} tile(s) to {args.out_dir}")

    if failures:
        print(f"\n{len(failures)} FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
