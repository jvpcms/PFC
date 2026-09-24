#!/usr/bin/env python3
"""Turn fused CBERS DN into the 8-bit RGB the annotators and the model see.

The rendering has two parts, fitted separately because they do different jobs:

* **Endpoints (per scene)** -- per-band p2 and p99.5 over the pooled pixels of
  that scene's own selected tiles. These decide how much DN range is stretched
  across 0..255, i.e. how much contrast is manufactured.
* **Gamma (per tile)** -- the exponent that lands this tile's rendered median on
  TARGET_MEDIAN. This decides exposure only.

Why split them. Endpoints must not be per tile: a lagoon tile spans about 6 DN,
and stretching that across the full range turns sensor noise into visible static.
But exposure must not be per scene: a scene's tiles are frequently bimodal --
207/153 holds three tiles of bright sediment-laden lagoon (DN median 325-355)
next to four land tiles (179-273) -- and a single exposure for both rendered the
land tiles at 49-74 out of 255. Splitting the two fixes both ends at once, and
per-image normalisation is in any case the standard convention for satellite
imagery in deep learning.

Why p99.5 rather than the p98 QGIS defaults to: bright sand and surf sit well
above p98 -- on 206/153 the red band runs 119 at p98 but 328 at p99.9 -- so a p98
ceiling renders whole beaches as flat white. And why gamma rather than simply
lowering the ceiling: vegetation is about half a typical scene and sits at only
0.17-0.24 of the p2..p99.5 range, so a linear ramp crushes all canopy texture
into near-black. The tone curve's shape is the problem, not its endpoint.

Measured over the 100 tiles, the rendered median went 43..216 (std 34.7) with a
fixed gamma of 1.8, and 80..205 (std 12.8) with this scheme; the worst tile's
clipped fraction went 22.8% -> 3.3%.

Every fitted number goes into the manifest per tile, so inference preprocessing
can reproduce a tile exactly. The fused GeoTIFF stays raw DN, so all of this can
be refitted without redoing the fusion.
"""

from __future__ import annotations

import numpy as np
import rasterio

# Matches QGIS "Min / Max Value Settings -> User defined" with these percentiles.
LO_PCT = 2.0
HI_PCT = 99.5

# The exposure the whole dataset is normalised to: the rendered median of every
# scene's tiles is driven here. 128 is mid-tone and matches what 206/151 -- the
# scene whose rendering was validated by eye -- already produced.
TARGET_MEDIAN = 128.0

# Gamma is solved rather than fixed, because percentile endpoints do not control
# where the median lands: a tile can sit anywhere inside p2..p99.5 and render
# from near-black to blown out. With a fixed gamma of 1.8 the rendered median
# across the 100 tiles ranged 43..216 (std 34.7); solving it per tile gives
# 80..205 (std 12.8). The clamp keeps a pathological tile -- an all-water one has
# no mid-tones to lift -- from producing an absurd curve.
GAMMA_MIN = 0.30
GAMMA_MAX = 1.20

# Reference only: the fixed gamma this replaced.
LEGACY_QGIS_GAMMA = 1.8


def fit_endpoints(blocks) -> tuple[np.ndarray, np.ndarray]:
    """Per-band (lo, hi) from a scene's own tiles.

    ``blocks`` is an iterable of (bands, H, W) DN arrays -- every tile that will
    be cut from one scene. Fitting on these rather than on the whole fused scene
    matters: the tiles are a small and often unrepresentative sample of their
    scene, so scene-wide endpoints leave a bright tile sitting high in the range
    and clipping. On the worst tile that meant 22.3% of pixels blown to white;
    fitting on the tiles brings it to 0.6%.

    Endpoints are deliberately **per scene**, never per tile. They decide how
    much DN range gets stretched across 0..255, so a per-tile choice would
    amplify a flat tile -- lagoon water spanning 6 DN -- into pure noise. The
    exposure of an individual tile is handled by solve_gamma instead, which
    cannot manufacture contrast because it leaves the endpoints alone.
    """
    x = np.hstack([b[:3].reshape(3, -1) for b in blocks]).astype("f4")
    valid = (x > 0).all(axis=0)
    if valid.sum() < 1000:
        raise ValueError("scene's tiles have essentially no valid pixels")
    x = x[:, valid]

    lo = np.percentile(x, LO_PCT, axis=1)
    hi = np.percentile(x, HI_PCT, axis=1)
    # Guard a degenerate band against a divide-by-~0 in to_uint8.
    hi = np.maximum(hi, lo + 1.0)
    return lo, hi


def solve_gamma(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Exponent that lands this tile's rendered median on TARGET_MEDIAN.

    Solved per tile. A scene's tiles are often bimodal -- 207/153 holds three
    tiles of bright sediment-laden lagoon (DN median 325-355) beside four land
    tiles (179-273) -- so one exposure per scene satisfies neither: fitting the
    pooled median rendered those land tiles at 49-74 out of 255. Per-tile gamma
    puts every one of them at the target.

    This is a monotonic curve over endpoints fixed by the scene, so it shifts
    brightness without inventing contrast, and it matches the per-image
    normalisation that is standard for satellite imagery in deep learning.
    """
    normalised = np.clip((rgb - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1)
    median = float(np.median(normalised))
    median = min(max(median, 1e-3), 1 - 1e-6)
    return float(
        np.clip(np.log(TARGET_MEDIAN / 255.0) / np.log(median), GAMMA_MIN, GAMMA_MAX)
    )


def to_uint8(
    rgb: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    gamma_exponent: float,
) -> np.ndarray:
    """Apply the fitted rendering to an (3, H, W) DN array, returning uint8.

    Nodata (any band zero) is emitted as black, matching how the scene renders
    outside its footprint.
    """
    out = (rgb.astype("f4") - lo[:, None, None]) / (hi - lo)[:, None, None]
    out = np.clip(out, 0.0, 1.0) ** gamma_exponent
    out = (out * 255.0).round().astype("u1")
    out[:, (rgb == 0).any(axis=0)] = 0
    return out


def describe(lo: np.ndarray, hi: np.ndarray, gamma_exponent: float) -> dict:
    """Manifest fields recording exactly how a tile was rendered."""
    return {
        "stretch_lo_r": round(float(lo[0]), 2),
        "stretch_lo_g": round(float(lo[1]), 2),
        "stretch_lo_b": round(float(lo[2]), 2),
        "stretch_hi_r": round(float(hi[0]), 2),
        "stretch_hi_g": round(float(hi[1]), 2),
        "stretch_hi_b": round(float(hi[2]), 2),
        "gamma_exponent": round(gamma_exponent, 4),
        "qgis_gamma": round(1.0 / gamma_exponent, 3),
        "lo_pct": LO_PCT,
        "hi_pct": HI_PCT,
        "target_median": TARGET_MEDIAN,
        "fit_basis": "scene_endpoints+tile_gamma",
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Print the fitted rendering for already-cut tile GeoTIFFs"
    )
    ap.add_argument("tifs", type=Path, nargs="+", help="tiles of ONE scene")
    args = ap.parse_args()

    blocks = []
    for path in args.tifs:
        with rasterio.open(path) as src:
            blocks.append(src.read())
    lo, hi = fit_endpoints(blocks)
    for i, n in enumerate("RGB"):
        print(f"  {n}: {lo[i]:6.0f} .. {hi[i]:6.0f}")
    for path, block in zip(args.tifs, blocks):
        g = solve_gamma(block[:3].astype("f4"), lo, hi)
        print(f"  {path.stem}: gamma exponent {g:.4f} (QGIS {1 / g:.2f})")
