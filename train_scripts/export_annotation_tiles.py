#!/usr/bin/env python3
"""Cut the selected annotation tiles out of pansharpened CBERS-4A scenes.

For every tile of the frozen selection that belongs to an already-fused scene:

    <split>/<tile_id>.png              8-bit RGB, the scene's stretch baked in
    assets/<split>/<tile_id>.tif       4-band Int16 DN with NIR, georeferenced
    assets/<split>/<tile_id>.wld       world file for the PNG
    assets/<split>/<tile_id>.png.aux.xml
    manifest.csv                       one row per tile

``<split>/`` deliberately contains nothing but PNGs, so it can be handed
straight to CVAT or a data loader. Everything else lives under ``assets/``.
Note this means the PNGs are not georeferenced in place: to inspect a tile's
position in QGIS, load its GeoTIFF, or copy the ``.wld`` back beside the PNG.

The manifest records exactly how each PNG was rendered, so inference
preprocessing can reproduce the identical transform.

The tile grid maps to source-PAN pixels as ``col = i*1024, row = j*1024``, but
the window is taken from the tile polygons in
``data/inpe/tiles_valid_swath.geojson`` rather than from that
formula, because a fused raster could in principle differ from the source PAN in
extent, CRS or grid. The formula is then used as a cross-check.

Nothing is silently skipped. The selection already passed nodata and cloud
checks, and dropping a tile would break the "first 0.2N / 0.8N" contract, so any
tile that cannot be cut correctly is reported by ``tile_id`` and the run fails.

Usage:
    .venv/bin/python train_scripts/export_annotation_tiles.py
    .venv/bin/python train_scripts/export_annotation_tiles.py --scene <scene_id>
    .venv/bin/python train_scripts/export_annotation_tiles.py --force
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_worklist import (
    DEFAULT_TILE_DIR,
    DEFAULT_WORK_DIR,
    REPO_ROOT,
    select_tiles,
)
from tile_stretch import describe, fit_endpoints, solve_gamma, to_uint8

# The valid-swath grid, not tiles_strictly_covering_pampa.geojson. CBERS-4A L4
# products are orthorectified onto a north-up UTM bbox, but the imaged swath
# inside is rotated ~13 deg with NoData in the corners, so the old grid placed
# tiles in empty space. See the 2026-08-16 note in notes/handoff_image_collection.md.
TILES_GEOJSON = REPO_ROOT / "data" / "inpe" / "tiles_valid_swath.geojson"
TILE_SPLIT_CSV = REPO_ROOT / "data" / "inpe" / "tile_split.csv"
DEFAULT_FUSED_DIR = DEFAULT_WORK_DIR / "fused"

TILE_PX = 1024
PIXEL_SIZE = 2.0
MANIFEST_NAME = "manifest.csv"

MANIFEST_COLUMNS = [
    "tile_id", "scene_id", "split_use", "i", "j", "crs", "px_size",
    "col_off", "row_off",
    "stretch_lo_r", "stretch_lo_g", "stretch_lo_b",
    "stretch_hi_r", "stretch_hi_g", "stretch_hi_b",
    "gamma_exponent", "qgis_gamma", "lo_pct", "hi_pct",
    "target_median", "fit_basis",
    "nodata_frac", "src_raster", "done_at",
]


class TileError(Exception):
    """A tile could not be cut correctly. Never swallowed."""


def tile_ij(tile_id: str) -> tuple[int, int]:
    i, j = tile_id.rsplit("_", 2)[-2:]
    return int(i), int(j)


def tile_window(src, geom, tile_id: str):
    """Window for one tile, validated against the raster and the i,j grid."""
    window = from_bounds(*geom.bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()

    if window.width != TILE_PX or window.height != TILE_PX:
        raise TileError(
            f"{tile_id}: window is {window.width}x{window.height}, expected "
            f"{TILE_PX}x{TILE_PX} -- the fused raster is not on the 2 m grid"
        )

    if (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + window.width > src.width
        or window.row_off + window.height > src.height
    ):
        raise TileError(
            f"{tile_id}: window {window} falls outside the raster "
            f"({src.width}x{src.height}) -- wrong scene or a cropped export"
        )

    # The fused grid should still be the source PAN grid; disagreement beyond a
    # pixel means it was resampled.
    i, j = tile_ij(tile_id)
    d_col = abs(window.col_off - i * TILE_PX)
    d_row = abs(window.row_off - j * TILE_PX)
    if d_col > 1 or d_row > 1:
        raise TileError(
            f"{tile_id}: geometry window ({window.col_off},{window.row_off}) "
            f"disagrees with i*{TILE_PX},j*{TILE_PX} "
            f"({i * TILE_PX},{j * TILE_PX}) by ({d_col},{d_row}) px"
        )

    return window


def write_tif(path: Path, data: np.ndarray, src, window) -> None:
    profile = src.profile.copy()
    profile.update(
        width=TILE_PX,
        height=TILE_PX,
        count=data.shape[0],
        dtype="int16",
        nodata=0,
        transform=src.window_transform(window),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        photometric="RGB" if data.shape[0] >= 3 else "MINISBLACK",
    )
    profile.pop("BIGTIFF", None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("int16"))


def write_png(path: Path, rgb8: np.ndarray, src, window) -> None:
    """PNG plus a world file, so it doubles as a QGIS layer."""
    profile = {
        "driver": "PNG",
        "width": TILE_PX,
        "height": TILE_PX,
        "count": 3,
        "dtype": "uint8",
        "crs": src.crs,
        "transform": src.window_transform(window),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(rgb8)

    # GDAL's PNG driver does not emit a world file itself; write it so QGIS (and
    # anything else) can place the image without reading the .aux.xml.
    t = src.window_transform(window)
    path.with_suffix(".wld").write_text(
        f"{t.a}\n{t.b}\n{t.d}\n{t.e}\n{t.c + t.a / 2}\n{t.f + t.e / 2}\n"
    )


def load_manifest(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="dataset size (default 100)")
    ap.add_argument("--fused-dir", type=Path, default=DEFAULT_FUSED_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_TILE_DIR)
    ap.add_argument("--scene", action="append", help="limit to these scene_ids")
    ap.add_argument("--force", action="store_true", help="re-cut tiles already written")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selection = select_tiles(pd.read_csv(TILE_SPLIT_CSV), args.n)

    # Every tile in the current grid is fully inside its scene's imagery swath.
    # If that ever stops holding, the NoData check below would start firing on
    # legitimate tiles, so catch it here where the cause is obvious.
    if "coverage" in selection.columns:
        partial = selection[selection["coverage"] < 1.0]
        if len(partial):
            print(
                f"{len(partial)} selected tile(s) are not fully covered, e.g. "
                f"{partial['tile_id'].iloc[0]} at {partial['coverage'].iloc[0]:.3f}",
                file=sys.stderr,
            )
            return 1

    geoms = gpd.read_file(TILES_GEOJSON).set_index("tile_id")

    missing_geom = set(selection["tile_id"]) - set(geoms.index)
    if missing_geom:
        print(
            f"{len(missing_geom)} selected tiles have no geometry, e.g. "
            f"{sorted(missing_geom)[:3]}",
            file=sys.stderr,
        )
        return 1

    manifest_path = args.out_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    already = set(manifest["tile_id"]) if len(manifest) else set()

    scenes = sorted(selection["scene_id"].unique())
    if args.scene:
        scenes = [s for s in scenes if s in set(args.scene)]

    rows = []
    failures = []
    n_written = 0
    n_skipped_scene = 0

    for scene_id in scenes:
        fused = args.fused_dir / f"{scene_id}_FUSED.tif"
        if not fused.exists():
            n_skipped_scene += 1
            continue

        want = selection[selection["scene_id"] == scene_id]
        todo = want if args.force else want[~want["tile_id"].isin(already)]
        if todo.empty:
            continue

        with rasterio.open(fused) as src:
            sub = geoms.loc[list(todo["tile_id"])].to_crs(src.crs)
            split_of = dict(zip(todo["tile_id"], todo["split_use"]))

            # Pass 1: read and validate every tile of this scene. The stretch is
            # fitted on all of them together, so it cannot be computed until
            # they are all in hand.
            cut = {}
            for tile_id, geom in sub.geometry.items():
                try:
                    window = tile_window(src, geom, tile_id)
                    data = src.read(window=window)
                    # Every tile of the valid-swath grid has full imagery, so
                    # NoData here is a symptom (wrong scene, resampled fusion,
                    # stale grid file), never legitimate data.
                    n_nodata = int((data == 0).all(axis=0).sum())
                    if n_nodata:
                        raise TileError(
                            f"{tile_id}: {n_nodata} NoData px "
                            f"({100 * n_nodata / (TILE_PX**2):.2f}%) -- every tile of "
                            f"the valid-swath grid should be fully imaged"
                        )
                except TileError as exc:
                    failures.append(str(exc))
                    continue
                except Exception as exc:  # noqa: BLE001 - report which tile died
                    failures.append(f"{tile_id}: {exc}")
                    continue
                cut[tile_id] = (window, data)

            if not cut:
                continue

            # Endpoints once per scene, so no tile gets its own contrast; gamma
            # per tile below, so each is exposed correctly.
            lo, hi = fit_endpoints([d for _, d in cut.values()])
            print(
                f"{scene_id}: {len(cut)} tile(s), endpoints "
                f"R {lo[0]:.0f}-{hi[0]:.0f} G {lo[1]:.0f}-{hi[1]:.0f} "
                f"B {lo[2]:.0f}-{hi[2]:.0f}"
            )
            if args.dry_run:
                continue

            # Pass 2: solve each tile's exposure, then write.
            for tile_id, (window, data) in cut.items():
                gamma = solve_gamma(data[:3].astype("f4"), lo, hi)
                stretch = describe(lo, hi, gamma)
                split_use = split_of[tile_id]
                # <split>/ holds nothing but PNGs, so it can be handed to CVAT
                # or a data loader as-is. The GeoTIFF and the PNG's
                # georeferencing sidecars live alongside under assets/.
                dest = args.out_dir / split_use
                assets = args.out_dir / "assets" / split_use
                dest.mkdir(parents=True, exist_ok=True)
                assets.mkdir(parents=True, exist_ok=True)

                write_tif(assets / f"{tile_id}.tif", data, src, window)
                write_png(dest / f"{tile_id}.png", to_uint8(data[:3], lo, hi, gamma), src, window)
                for sidecar in (
                    f"{tile_id}.wld",
                    f"{tile_id}.png.aux.xml",
                    f"{tile_id}.tif.aux.xml",
                ):
                    produced = dest / sidecar
                    if produced.exists():
                        produced.replace(assets / sidecar)

                i, j = tile_ij(tile_id)
                rows.append(
                    {
                        "tile_id": tile_id,
                        "scene_id": scene_id,
                        "split_use": split_use,
                        "i": i,
                        "j": j,
                        "crs": str(src.crs),
                        "px_size": PIXEL_SIZE,
                        "col_off": int(window.col_off),
                        "row_off": int(window.row_off),
                        **stretch,
                        "nodata_frac": round(float((data == 0).all(axis=0).mean()), 4),
                        "src_raster": fused.name,
                        "done_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                )
                n_written += 1

    if rows:
        fresh = pd.DataFrame(rows)
        # An all-NA existing manifest would otherwise change the merged dtypes.
        parts = [df for df in (manifest, fresh) if len(df)]
        merged = pd.concat(parts, ignore_index=True)
        merged = merged.drop_duplicates("tile_id", keep="last").sort_values("tile_id")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(manifest_path, index=False, columns=MANIFEST_COLUMNS)

    total = len(selection)
    done = len(already | {r["tile_id"] for r in rows})
    print()
    print(f"wrote {n_written} tile(s); {done}/{total} of the N={args.n} selection now cut")
    if n_skipped_scene:
        print(f"{n_skipped_scene} scene(s) not fused yet")

    if failures:
        print(f"\n{len(failures)} tile(s) FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
