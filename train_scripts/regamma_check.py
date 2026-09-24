#!/usr/bin/env python3
"""Re-render the tiles whose exposure gamma hit GAMMA_MAX, into a separate tree.

Why this exists. ``tile_stretch.solve_gamma`` fits a per-tile exponent that lands
the tile's rendered median on TARGET_MEDIAN, then clamps it to
[GAMMA_MIN, GAMMA_MAX]. Nine of the 100 annotation tiles need a gamma above the 1.20
ceiling and were therefore rendered too bright -- worst case
206_152_L4_31_11, a 100% water tile that needed 3.78 and rendered at median 205
instead of 128, reading as near-white orange.

Raising the ceiling can only affect a tile whose solved gamma exceeds the old
ceiling, so the blast radius is exactly those nine tiles; every other tile is
independent of the constant's value. This script therefore re-renders only those
and writes them somewhere else, leaving the production PNGs untouched so the two
can be compared before anything is replaced.

Gamma is applied AFTER the endpoints clip to [0, 1], so the clipped fraction is
identical in both renders -- the change is exposure only, never new blown pixels.
Note also that gamma is monotonic, so it cannot reorder the colour channels: a
tile whose per-band endpoints inverted its channel order stays inverted.

Full-precision gamma is used, not the manifest's 4-decimal rounding, which is
itself worth ~1 LSB on about 40% of tiles.

Outputs, under --out:
    new/<tile>.png          re-rendered at the new ceiling (+ .wld for QGIS)
    side_by_side/<tile>.png old on the left, new on the right, 8 px divider
    comparison.csv          gamma and channel medians, before and after

Usage:
    .venv/bin/python train_scripts/regamma_check.py
    .venv/bin/python train_scripts/regamma_check.py --new-max 4.0
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile_stretch import GAMMA_MIN, GAMMA_MAX, TARGET_MEDIAN, to_uint8  # noqa: E402


def solved_gamma(dn: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Unclamped exponent that would land the pooled RGB median on TARGET_MEDIAN."""
    n = np.clip((dn.astype("f4") - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1)
    median = min(max(float(np.median(n)), 1e-3), 1 - 1e-6)
    return float(np.log(TARGET_MEDIAN / 255.0) / np.log(median))


def endpoints(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([row[f"stretch_lo_{c}"] for c in "rgb"], "f4")
    hi = np.array([row[f"stretch_hi_{c}"] for c in "rgb"], "f4")
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/inpe_scenes/annotation_tiles")
    ap.add_argument("--out", default="data/inpe_scenes/annotation_tiles/regamma_check")
    ap.add_argument("--new-max", type=float, default=4.0,
                    help=f"replacement for GAMMA_MAX (currently {GAMMA_MAX})")
    args = ap.parse_args()

    A = Path(args.tiles)
    out = Path(args.out)
    (out / "new").mkdir(parents=True, exist_ok=True)
    (out / "side_by_side").mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(A / "manifest.csv")
    rows = []

    for _, r in man.iterrows():
        tif = glob.glob(str(A / "assets" / "*" / f"{r.tile_id}.tif"))
        if not tif:
            print(f"  no asset for {r.tile_id}, skipped")
            continue
        lo, hi = endpoints(r)
        with rasterio.open(tif[0]) as src:
            dn = src.read()[:3]
            crs, transform = src.crs, src.transform

        g_ideal = solved_gamma(dn, lo, hi)
        if g_ideal <= GAMMA_MAX:
            continue                    # the ceiling never bound this tile

        g_old = float(r.gamma_exponent)
        g_new = float(np.clip(g_ideal, GAMMA_MIN, args.new_max))

        old_rgb = to_uint8(dn, lo, hi, g_old)
        new_rgb = to_uint8(dn, lo, hi, g_new)

        profile = dict(driver="PNG", width=dn.shape[2], height=dn.shape[1],
                       count=3, dtype="uint8", crs=crs, transform=transform)
        dest = out / "new" / f"{r.tile_id}.png"
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(new_rgb)
        t = transform
        dest.with_suffix(".wld").write_text(
            f"{t.a}\n{t.b}\n{t.d}\n{t.e}\n{t.c + t.a / 2}\n{t.f + t.e / 2}\n")

        gap = np.full((3, dn.shape[1], 8), 255, "u1")
        pair = np.concatenate([old_rgb, gap, new_rgb], axis=2)
        sbs = out / "side_by_side" / f"{r.tile_id}.png"
        with rasterio.open(sbs, "w", driver="PNG", width=pair.shape[2],
                           height=pair.shape[1], count=3, dtype="uint8") as dst:
            dst.write(pair)

        med = lambda a: np.median(a.reshape(3, -1), axis=1)
        mo, mn = med(old_rgb), med(new_rgb)
        rows.append(dict(
            tile_id=r.tile_id, split=r.split_use,
            gamma_old=g_old, gamma_ideal=round(g_ideal, 6), gamma_new=g_new,
            clamped_still=bool(g_ideal > args.new_max),
            med_old=float(np.median(old_rgb)), med_new=float(np.median(new_rgb)),
            R_old=mo[0], G_old=mo[1], B_old=mo[2],
            R_new=mn[0], G_new=mn[1], B_new=mn[2],
            warm_old=bool(mo[0] > mo[2]), warm_new=bool(mn[0] > mn[2]),
        ))
        print(f"  {r.tile_id}  gamma {g_old:.2f} -> {g_new:.4f}   "
              f"median {np.median(old_rgb):.0f} -> {np.median(new_rgb):.0f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison.csv", index=False)
    print(f"\n{len(df)} tiles re-rendered (GAMMA_MAX {GAMMA_MAX} -> {args.new_max})")
    print(f"-> {out}/new, {out}/side_by_side, {out}/comparison.csv")
    print("Production PNGs under all/, train/, test/ were NOT modified.")


if __name__ == "__main__":
    main()
