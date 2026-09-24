#!/usr/bin/env python3
"""Pansharpen downloaded CBERS-4A WPM scenes with GDAL, no QGIS in the loop.

This replaces the manual QGIS recipe from ``tutorial_transcript.txt``, since
both of its steps are thin wrappers over GDAL:

  QGIS "Merge" (BAND1-4 -> comp_colorida) + QGIS "Pansharpening"
      == gdal_pansharpen.py <PAN> <BAND1> <BAND2> <BAND3> <BAND4> <out>

``gdal_pansharpen`` accepts several spectral datasets directly, so the
intermediate ``comp_colorida`` stack is never written.

Two deliberate departures from the manual recipe, both so the output is correct
without any per-layer fiddling downstream:

**Band order is R, G, B, NIR** -- that is, source BAND3, BAND2, BAND1, BAND4.
The raw CBERS order (BAND1..4 = B, G, R, NIR) makes every viewer that defaults
to "band 1 -> red" render the scene with red and blue swapped, which is what
QGIS does with a fresh raster. Emitting R, G, B, NIR instead means the QGIS
default is already natural colour, the colour interpretation tags are set so
QGIS picks them up on load, and the RGB the annotators see matches the RGB the
DeepGlobe-pretrained model was trained on. NIR stays as band 4.

**Band weights are fitted per scene.** Brovey computes
``out_i = MS_i * PAN / pseudo_pan`` with ``pseudo_pan = sum(w_i * MS_i)``.
GDAL's default equal weights (0.25 each) make ``pseudo_pan`` a poor estimate of
the real PAN, because the PAN detector does not respond equally to the four
bands -- so the fused scene drifts off the MS radiometry by a factor that is
not even constant within one scene (measured on 206/153: 1.05x on land, 0.81x
over water). Regressing the downsampled PAN on the four MS bands and passing
those weights brings the fused product back to within ~0.2% of the 8 m MS
levels, so a fused tile and the 8 m composite of the same ground look alike.
Use --weights equal to reproduce the untuned QGIS behaviour instead.

The fit is reported per scene; if it comes out poor the script falls back to
equal weights and says so.

Usage:
    # next pending scene in worklist order
    .venv/bin/python train_scripts/pansharpen_scenes.py --next

    # a specific scene, or several
    .venv/bin/python train_scripts/pansharpen_scenes.py CBERS_4A_WPM_20230704_206_153_L4

    # everything still pending, most-tiles-first
    .venv/bin/python train_scripts/pansharpen_scenes.py --all
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotation_worklist import DEFAULT_TILE_DIR, DEFAULT_WORK_DIR, build_worklist
from sfim import pansharpen_scene as sfim_scene

GDAL_PANSHARPEN = "/usr/bin/gdal_pansharpen.py"
# The venv has no osgeo bindings; the system interpreter does.
SYSTEM_PYTHON = "/usr/bin/python3"

FUSED_GB_ESTIMATE = 20.0  # worst case, uncompressed; DEFLATE lands far below

RESAMPLING = "cubic"  # QGIS Pansharpening default
NODATA = "0"  # declared by the source BAND0..BAND4 rasters
PAN_MS_RATIO = 4

# Output order, as source band numbers: R, G, B, NIR.
OUTPUT_BANDS = (3, 2, 1, 4)

# Weight fitting reads the whole scene decimated to this many pixels a side.
# Sampling a few full-resolution windows instead is not safe: on a coastal scene
# every window can land on open water, where the fit degenerates.
FIT_SIZE = 1500
MIN_FIT_R2 = 0.8


def scene_bands(scene_dir: Path) -> tuple[Path, dict[int, Path]]:
    """(PAN, {band_number: path}) for a scene folder."""
    pan = sorted(scene_dir.glob("*_BAND0.tiff"))
    ms = {}
    missing = [] if pan else ["BAND0"]
    for b in (1, 2, 3, 4):
        hits = sorted(scene_dir.glob(f"*_BAND{b}.tiff"))
        if hits:
            ms[b] = hits[0]
        else:
            missing.append(f"BAND{b}")
    if missing:
        raise SystemExit(f"{scene_dir.name}: missing {', '.join(missing)}")
    return pan[0], ms


def fit_weights(pan_path: Path, ms_paths: dict[int, Path], order: tuple[int, ...]):
    """Non-negative weights so that sum(w_i * MS_i) approximates the PAN band.

    Both sides are read decimated over the full scene, which covers land and
    water in their true proportions and takes a few seconds. The weights are
    solved with NNLS, since an unconstrained fit can hand a negative weight to
    a band that correlates with its neighbours.

    Returns (weights_in_output_order, r2), or (None, r2) if the fit is unusable.
    """
    ms = []
    for b in order:
        with rasterio.open(ms_paths[b]) as src:
            ms.append(src.read(1, out_shape=(FIT_SIZE, FIT_SIZE)).astype("f8"))
    ms = np.stack(ms)
    with rasterio.open(pan_path) as src:
        pan = src.read(1, out_shape=(FIT_SIZE, FIT_SIZE)).astype("f8")

    ok = (ms > 0).all(axis=0) & (pan > 0)
    if ok.sum() < 10_000:
        return None, float("nan")

    A = ms[:, ok].T
    y = pan[ok]
    w, _ = nnls(A, y)
    r2 = 1 - ((y - A @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    if r2 < MIN_FIT_R2 or not w.any():
        return None, r2
    return w, r2


def fused_path(out_dir: Path, scene_id: str) -> Path:
    return out_dir / f"{scene_id}_FUSED.tif"


def pansharpen(
    scene_dir: Path,
    out_path: Path,
    threads: str,
    weights_mode: str,
    dry_run: bool,
    method: str = "sfim",
    lowpass: int | None = None,
) -> bool:
    pan, ms = scene_bands(scene_dir)
    spectral = [ms[b] for b in OUTPUT_BANDS]

    if method == "sfim":
        if dry_run:
            print(f"  SFIM {pan.name} + {len(spectral)} bands -> {out_path.name}")
            return True
        tmp = out_path.with_suffix(".tif.partial")
        tmp.unlink(missing_ok=True)
        t0 = time.time()

        def show(done, total):
            print(f"\r  {100 * done / total:5.1f}%", end="", flush=True)

        try:
            kw = {} if lowpass is None else {"lowpass": lowpass}
            sfim_scene(pan, ms, tmp, band_order=OUTPUT_BANDS, progress=show, **kw)
        except Exception as exc:  # noqa: BLE001 - report and keep the batch going
            tmp.unlink(missing_ok=True)
            print(f"\n  FAILED: {exc}", file=sys.stderr)
            return False
        print("\r  100.0%")
        tmp.rename(out_path)
        out_path.with_suffix(".tif.aux.xml").unlink(missing_ok=True)
        size_gb = out_path.stat().st_size / 1024**3
        print(f"  done in {time.time() - t0:.0f}s -> {out_path.name} ({size_gb:.1f} GB)")
        return True

    weights = None
    if weights_mode == "fit" and not dry_run:
        weights, r2 = fit_weights(pan, ms, OUTPUT_BANDS)
        if weights is None:
            print(f"  weight fit rejected (R2={r2:.3f}), using equal weights")
        else:
            shown = ", ".join(f"{n}={v:.3f}" for n, v in zip("RGBN", weights))
            print(f"  fitted weights {shown}  (R2={r2:.3f})")

    cmd = [
        SYSTEM_PYTHON,
        GDAL_PANSHARPEN,
        # Stated explicitly because the temporary name ends in .partial, which
        # GDAL cannot map to a driver.
        "-of", "GTiff",
        "-r", RESAMPLING,
        "-nodata", NODATA,
        "-threads", threads,
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "BIGTIFF=YES",
        "-co", f"NUM_THREADS={threads}",
        # Tags bands 1-3 as R, G, B at creation time, so viewers open the scene
        # in natural colour. gdal_edit -colorinterp_N cannot do this after the
        # fact on a compressed BigTIFF: it sets band 1 and silently leaves the
        # rest Gray.
        "-co", "PHOTOMETRIC=RGB",
    ]
    if weights is not None:
        for w in weights:
            cmd += ["-w", f"{w:.6f}"]
    cmd += [str(pan), *[str(p) for p in spectral], str(out_path)]

    if dry_run:
        print("  " + " ".join(cmd))
        return True

    tmp = out_path.with_suffix(".tif.partial")
    tmp.unlink(missing_ok=True)
    cmd[-1] = str(tmp)

    t0 = time.time()
    if subprocess.run(cmd).returncode != 0:
        tmp.unlink(missing_ok=True)
        print("  FAILED", file=sys.stderr)
        return False

    # gdal_pansharpen leaves a PAM sidecar that pins bands 2-4 to ColorInterp
    # Gray, which overrides the PHOTOMETRIC=RGB tag written into the TIFF. It
    # carries nothing else worth keeping, so drop it and let the file speak for
    # itself.
    tmp.with_suffix(".tif.partial.aux.xml").unlink(missing_ok=True)

    # Only publish the final name once GDAL succeeded, so an interrupted run
    # never leaves a half-written scene that looks done.
    tmp.rename(out_path)
    out_path.with_suffix(".tif.aux.xml").unlink(missing_ok=True)

    size_gb = out_path.stat().st_size / 1024**3
    print(f"  done in {time.time() - t0:.0f}s -> {out_path.name} ({size_gb:.1f} GB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", nargs="*", help="scene_ids to fuse")
    ap.add_argument("--next", action="store_true", help="fuse the next pending scene")
    ap.add_argument("--all", action="store_true", help="fuse every pending scene")
    ap.add_argument("--n", type=int, default=100, help="dataset size (default 100)")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_WORK_DIR / "raw_inpe_data")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_WORK_DIR / "fused")
    ap.add_argument("--tile-dir", type=Path, default=DEFAULT_TILE_DIR)
    ap.add_argument("--threads", default="ALL_CPUS")
    ap.add_argument(
        "--lowpass",
        type=int,
        default=None,
        help="SFIM lowpass width in PAN px (default 6); higher = more texture, "
             "less faithful colour. See train_scripts/sfim.py for the trade-off.",
    )
    ap.add_argument(
        "--method",
        choices=("sfim", "brovey"),
        default="sfim",
        help="SFIM preserves MS colour (default); brovey is GDAL's, kept for comparison",
    )
    ap.add_argument(
        "--weights",
        choices=("fit", "equal"),
        default="fit",
        help="fit per-scene PAN weights (default) or use GDAL's equal weights",
    )
    ap.add_argument("--force", action="store_true", help="redo scenes already fused")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (args.scenes or args.next or args.all):
        ap.error("give scene_ids, or --next, or --all")

    work, _ = build_worklist(args.n, args.tile_dir)
    downloaded = {
        d.name
        for d in args.raw_dir.iterdir()
        if d.is_dir() and len(list(d.glob("*_BAND*.tiff"))) == 5
    }

    if args.scenes:
        todo = list(args.scenes)
    else:
        # Worklist order, so the most productive scenes fuse first.
        todo = [s for s in work["scene_id"] if s in downloaded]

    tiles_of = dict(zip(work["scene_id"], work["tiles_n"]))
    pending = []
    for scene_id in todo:
        if fused_path(args.out_dir, scene_id).exists() and not args.force:
            if args.scenes or args.all:
                print(f"skip {scene_id} (already fused)")
            continue
        if scene_id not in downloaded:
            print(f"skip {scene_id} (not fully downloaded)", file=sys.stderr)
            continue
        pending.append(scene_id)

    # --next means the next scene still needing work, not the first in the
    # worklist, so it has to be applied after the already-fused ones drop out.
    if args.next:
        pending = pending[:1]

    if not pending:
        print("nothing to do")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.out_dir).free / 1024**3
    print(
        f"{len(pending)} scene(s) to fuse, {free_gb:.0f} GB free, "
        f"method={args.method}, band order R,G,B,NIR"
    )
    if free_gb < FUSED_GB_ESTIMATE and not args.dry_run:
        print("not enough free space for even one scene", file=sys.stderr)
        return 1

    failures = []
    for i, scene_id in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {scene_id}  ({tiles_of.get(scene_id, '?')} tiles)")
        ok = pansharpen(
            args.raw_dir / scene_id,
            fused_path(args.out_dir, scene_id),
            args.threads,
            args.weights,
            args.dry_run,
            args.method,
            args.lowpass,
        )
        if not ok:
            failures.append(scene_id)
        elif not args.dry_run:
            free_gb = shutil.disk_usage(args.out_dir).free / 1024**3
            if free_gb < FUSED_GB_ESTIMATE and i < len(pending):
                print(
                    f"stopping: {free_gb:.0f} GB free, not enough for the next scene.",
                    file=sys.stderr,
                )
                break

    if failures:
        print(f"failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
