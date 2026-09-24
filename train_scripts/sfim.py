#!/usr/bin/env python3
"""SFIM pansharpening: inject PAN detail without disturbing MS colour.

Smoothing-Filter-based Intensity Modulation computes, per band,

    out = MS_upsampled * PAN / lowpass(PAN)

``PAN / lowpass(PAN)`` is a purely high-frequency ratio centred on 1, so the
low-frequency content of each band -- its colour and brightness -- is carried
over from the MS untouched, and only texture is added.

This is why it is used here instead of GDAL's Brovey (``gdal_pansharpen``).
Brovey computes ``MS_i * PAN / pseudo_pan`` with a single modulation factor
shared by every band, which imposes the PAN's *relative* texture on all of them
equally. Bands with little native contrast are then over-sharpened. Measured on
206/151, comparing each method downsampled back to the 8 m MS grid:

    band   MS (truth)      SFIM            Brovey
    B      132.8 / 12.0    132.5 / 12.0    132.7 / 18.4   <- +53% variance
    G       95.8 / 15.8     95.5 / 15.3     95.6 / 18.0
    R       84.4 / 25.2     84.2 / 24.4     84.2 / 25.5
    NIR    201.4 / 42.1    201.0 / 40.2    200.4 / 41.9

    mean |difference| vs MS:  SFIM 2.7-8.1 DN, Brovey 6.9-16.6 DN

SFIM still injects real detail -- high-frequency energy in blue rises from 1.23
(upsampled MS) to 6.44, against the PAN's own 8.24 -- it simply does not invent
more than the PAN contains. The result is the 8 m composite's colour with 2 m
structure, which is the goal.

The scene is processed in blocks with a halo, since a full scene is ~3.4 Gpx per
band and cannot be held in memory.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

PAN_MS_RATIO = 4

# Halo in MS pixels, wide enough for the cubic upsample (~2 px) and half the
# widest lowpass window (16 PAN px = 4 MS px). 8 leaves ample margin and costs
# nothing measurable.
HALO_MS = 8

# Blocks are sized in MS pixels; 1024 MS px = 4096 PAN px, about 270 MB of
# float32 working set for four bands.
BLOCK_MS = 1024

INT16_MAX = 32767

# Width of the PAN lowpass, in PAN pixels. At the ratio (4) the injected detail
# is exactly what the MS cannot already resolve, and the MS statistics are
# preserved almost perfectly. Widening it smooths the reference further, so the
# PAN/lowpass ratio swings more and more texture is injected -- at the cost of
# reintroducing variance the MS never had. Measured on 206/151 land:
#
#     lowpass  detail(R@2m)  blue variance vs MS  mean|diff| vs MS
#        4         7.50            1.02x               2.35
#        6         8.91            1.10x               3.24
#        8         9.71            1.22x               4.39
#       12        10.66            1.44x               6.32
#       16        11.27            1.60x               7.62
#     (upsampled MS 3.33, PAN itself 10.33; GDAL Brovey sits at 1.68x)
#
# 6 is the default: visibly more texture than the strict choice, still far
# closer to the MS colour than Brovey.
DEFAULT_LOWPASS = 6


def sfim_block(
    pan: np.ndarray,
    ms_up: np.ndarray,
    ratio: int = PAN_MS_RATIO,
    lowpass: int = DEFAULT_LOWPASS,
) -> np.ndarray:
    """SFIM on one block, with the MS already upsampled to the PAN grid."""
    # Crop in case the upsample and the PAN read disagree by a pixel at the
    # scene edge.
    h = min(ms_up.shape[1], pan.shape[0])
    w = min(ms_up.shape[2], pan.shape[1])
    ms_up = ms_up[:, :h, :w]
    pan = pan[:h, :w]

    pan_low = uniform_filter(pan, size=lowpass)

    # A pixel is outside the imaged swath only when EVERY band reads zero.
    # Requiring all bands to be positive is wrong twice over: the PAN carries
    # isolated dead pixels inside good imagery, and NIR legitimately reads 0
    # over dark water (measured on 211/152: 1018 zero-NIR pixels in one tile,
    # 100% of them water, with R/G/B all healthy). Either would punch holes
    # through otherwise perfect tiles.
    valid = (ms_up > 0).any(axis=0) | (pan > 0)
    usable = (pan > 0) & (pan_low > 0)
    ratio_img = np.where(usable, pan / np.maximum(pan_low, 1e-6), 1.0)
    out = ms_up * ratio_img
    return np.where(valid, np.clip(out, 0, INT16_MAX), 0)


def pansharpen_scene(
    pan_path,
    ms_paths,
    out_path,
    band_order=(3, 2, 1, 4),
    block_ms: int = BLOCK_MS,
    lowpass: int = DEFAULT_LOWPASS,
    progress=None,
) -> None:
    """Write an SFIM-pansharpened scene.

    ``ms_paths`` maps source band number to file. ``band_order`` selects and
    orders the output bands; the default emits R, G, B, NIR.
    """
    pan_src = rasterio.open(pan_path)
    ms_srcs = [rasterio.open(ms_paths[b]) for b in band_order]

    profile = pan_src.profile.copy()
    profile.update(
        count=len(band_order),
        dtype="int16",
        nodata=0,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        num_threads="ALL_CPUS",
        BIGTIFF="YES",
        photometric="RGB" if len(band_order) >= 3 else "MINISBLACK",
    )

    ms_w, ms_h = ms_srcs[0].width, ms_srcs[0].height
    r = PAN_MS_RATIO
    n_blocks = ((ms_h + block_ms - 1) // block_ms) * ((ms_w + block_ms - 1) // block_ms)

    try:
        with rasterio.open(out_path, "w", **profile) as dst:
            done = 0
            for my in range(0, ms_h, block_ms):
                for mx in range(0, ms_w, block_ms):
                    bw = min(block_ms, ms_w - mx)
                    bh = min(block_ms, ms_h - my)

                    # Read with a halo so the upsample and lowpass have context,
                    # then discard it before writing.
                    ax = max(0, mx - HALO_MS)
                    ay = max(0, my - HALO_MS)
                    aw = min(ms_w - ax, bw + (mx - ax) + HALO_MS)
                    ah = min(ms_h - ay, bh + (my - ay) + HALO_MS)

                    # Upsample inside GDAL rather than with scipy.zoom: same
                    # cubic kernel, several times faster, and it avoids holding
                    # both resolutions of every band at once.
                    ms_up = np.stack(
                        [
                            s.read(
                                1,
                                window=Window(ax, ay, aw, ah),
                                out_shape=(ah * r, aw * r),
                                resampling=Resampling.cubic,
                            )
                            for s in ms_srcs
                        ]
                    ).astype("f4")
                    pan = pan_src.read(
                        1, window=Window(ax * r, ay * r, aw * r, ah * r)
                    ).astype("f4")

                    out = sfim_block(pan, ms_up, lowpass=lowpass)

                    # Trim the halo and clip to what the destination can hold.
                    ox, oy = (mx - ax) * r, (my - ay) * r
                    ow = min(bw * r, out.shape[2] - ox)
                    oh = min(bh * r, out.shape[1] - oy)
                    if ow <= 0 or oh <= 0:
                        continue
                    ow = min(ow, dst.width - mx * r)
                    oh = min(oh, dst.height - my * r)
                    if ow <= 0 or oh <= 0:
                        continue

                    dst.write(
                        out[:, oy : oy + oh, ox : ox + ow].astype("int16"),
                        window=Window(mx * r, my * r, ow, oh),
                    )
                    done += 1
                    if progress and done % 20 == 0:
                        progress(done, n_blocks)
    finally:
        pan_src.close()
        for s in ms_srcs:
            s.close()
