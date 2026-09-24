#!/usr/bin/env python3
"""Scene work list for the hand-annotation tile collection.

Answers: which CBERS-4A scenes still need to be downloaded and pansharpened, in
what order, and how many tiles each one yields.

The tile selection in ``data/inpe/tile_split.csv`` is frozen. A dataset of size
N is the first ``0.2N`` tiles by ``test_rank`` plus the first ``0.8N`` by
``train_rank``; both orders are prefix-stable, so raising N is purely additive.

Progress is derived from the cropped tiles on disk (``<tile_id>.tif`` under the
output directory, searched recursively so the ``test/`` and ``train/``
subdirectories used by the cropper are picked up). A scene counts as done when
every one of its selected tiles exists.

Usage:
    .venv/bin/python train_scripts/annotation_worklist.py
    .venv/bin/python train_scripts/annotation_worklist.py --n 200
    .venv/bin/python train_scripts/annotation_worklist.py --next
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TILE_SPLIT_CSV = REPO_ROOT / "data" / "inpe" / "tile_split.csv"
SCENES_GEOJSON = REPO_ROOT / "data" / "inpe" / "selected_scenes_pampa44.geojson"
DEFAULT_WORK_DIR = Path(
    "/media/jvpcms/9a4d7913-803b-4816-9d70-c54550ae61b8/cbers_work"
)
# Tiles live on the external drive beside the raw and fused scenes, not in the
# repo: the internal disk has no room, and they are derived data.
DEFAULT_TILE_DIR = DEFAULT_WORK_DIR / "annotation_tiles"

# Peak disk for one scene: raw MS + raw PAN + comp_colorida + fused output.
SCENE_PEAK_GB = 28.0

TEST_FRACTION = 0.2

SCENE_ID_RE = re.compile(
    r"^CBERS_4A_WPM_(?P<date>\d{8})_(?P<path>\d{3})_(?P<row>\d{3})_L4$"
)

INPE_CATALOG_URL = "http://www2.dgi.inpe.br/catalogo/explore"


def parse_scene_id(scene_id: str) -> dict:
    """Split a BDC scene_id into date, path and row."""
    m = SCENE_ID_RE.match(scene_id)
    if m is None:
        raise ValueError(f"scene_id does not parse: {scene_id!r}")
    return {
        "date": datetime.strptime(m.group("date"), "%Y%m%d").date(),
        "path": int(m.group("path")),
        "row": int(m.group("row")),
    }


def select_tiles(split: pd.DataFrame, n: int) -> pd.DataFrame:
    """First 0.2N tiles by test_rank plus first 0.8N by train_rank."""
    n_test = int(round(n * TEST_FRACTION))
    n_train = n - n_test

    test = split[split["test_rank"].notna()].nsmallest(n_test, "test_rank").copy()
    train = split[split["train_rank"].notna()].nsmallest(n_train, "train_rank").copy()

    if len(test) < n_test or len(train) < n_train:
        raise SystemExit(
            f"not enough ranked tiles for N={n}: "
            f"got {len(test)}/{n_test} test, {len(train)}/{n_train} train"
        )

    test["split_use"] = "test"
    train["split_use"] = "train"
    return pd.concat([test, train], ignore_index=True)


def scene_metadata() -> pd.DataFrame:
    """cloud_cover and in_season per scene, keyed on (path, row, date).

    ``selected_scenes_pampa44.geojson`` stores lgi-stac ids in a different
    format from the BDC ``scene_id``, so a direct id join returns nothing.
    """
    try:
        import geopandas as gpd
    except ImportError:  # metadata is a nicety, not a requirement
        return pd.DataFrame(columns=["path", "row", "date", "cloud_cover", "in_season"])

    scenes = gpd.read_file(SCENES_GEOJSON)
    meta = pd.DataFrame(
        {
            "path": scenes["path"].astype(int),
            "row": scenes["row"].astype(int),
            "date": pd.to_datetime(scenes["datetime"]).dt.date,
            "cloud_cover": scenes["cloud_cover"],
            "in_season": scenes["in_season"],
        }
    )
    return meta


def done_tile_ids(tile_dir: Path) -> set[str]:
    if not tile_dir.exists():
        return set()
    return {p.stem for p in tile_dir.rglob("*.tif")}


def build_worklist(n: int, tile_dir: Path) -> pd.DataFrame:
    split = pd.read_csv(TILE_SPLIT_CSV)
    tiles = select_tiles(split, n)

    parsed = pd.DataFrame([parse_scene_id(s) for s in tiles["scene_id"]])
    tiles = pd.concat([tiles.reset_index(drop=True), parsed], axis=1)

    have = done_tile_ids(tile_dir)
    tiles["done"] = tiles["tile_id"].isin(have)

    grouped = (
        tiles.groupby(["scene_id", "path", "row", "date"], as_index=False)
        .agg(
            tiles_n=("tile_id", "size"),
            test_n=("split_use", lambda s: (s == "test").sum()),
            train_n=("split_use", lambda s: (s == "train").sum()),
            done_n=("done", "sum"),
        )
    )

    meta = scene_metadata()
    if not meta.empty:
        grouped = grouped.merge(meta, on=["path", "row", "date"], how="left")
    else:
        grouped["cloud_cover"] = pd.NA
        grouped["in_season"] = pd.NA

    # Most tiles first, so the early downloads pay for themselves the most.
    grouped = grouped.sort_values(
        ["tiles_n", "cloud_cover", "scene_id"], ascending=[False, True, True]
    ).reset_index(drop=True)
    grouped.insert(0, "order", grouped.index + 1)
    grouped["cum"] = grouped["tiles_n"].cumsum()
    grouped["status"] = [
        "done" if d == t else ("partial" if d else "pending")
        for d, t in zip(grouped["done_n"], grouped["tiles_n"])
    ]
    return grouped, tiles


def check_disk(work_dir: Path) -> tuple[bool, str]:
    if not work_dir.exists():
        return False, f"work dir does not exist (drive not mounted?): {work_dir}"
    free_gb = shutil.disk_usage(work_dir).free / 1024**3
    ok = free_gb >= SCENE_PEAK_GB
    msg = f"{free_gb:.0f} GB free at {work_dir} (need ~{SCENE_PEAK_GB:.0f} GB per scene)"
    return ok, msg


def fmt_cloud(v) -> str:
    return "-" if pd.isna(v) else f"{v:.0f}%"


def print_table(work: pd.DataFrame) -> None:
    header = (
        f"{'#':>3}  {'scene_id':<34} {'path/row':<9} {'date':<10} "
        f"{'cloud':>5} {'tiles':>5} {'test':>4} {'train':>5} {'cum':>4}  status"
    )
    print(header)
    print("-" * len(header))
    for r in work.itertuples():
        print(
            f"{r.order:>3}  {r.scene_id:<34} {r.path}/{r.row:<5} {str(r.date):<10} "
            f"{fmt_cloud(r.cloud_cover):>5} {r.tiles_n:>5} {r.test_n:>4} "
            f"{r.train_n:>5} {r.cum:>4}  {r.status}"
            + (f" ({r.done_n}/{r.tiles_n})" if r.status == "partial" else "")
        )


def print_next(work: pd.DataFrame, tiles: pd.DataFrame, work_dir: Path) -> None:
    pending = work[work["status"] != "done"]
    if pending.empty:
        print("All scenes done. Nothing left to download.")
        return

    r = pending.iloc[0]
    scene_tiles = tiles[tiles["scene_id"] == r.scene_id].sort_values("tile_id")

    print(f"Next scene: {r.scene_id}")
    print(f"  path/row  {r.path}/{r.row}   date {r.date}   cloud {fmt_cloud(r.cloud_cover)}")
    print(f"  yields    {r.tiles_n} tiles ({r.test_n} test, {r.train_n} train)")
    if r.done_n:
        print(f"  partial   {r.done_n}/{r.tiles_n} already cropped")
    print(f"  work dir  {work_dir}")
    print()
    print("  Tiles to cut:")
    for t in scene_tiles.itertuples():
        mark = "x" if t.done else " "
        print(f"    [{mark}] {t.tile_id}  ({t.split_use})")
    print()
    print("  Steps:")
    print(f"    1. download the {r.path}/{r.row} scene of {r.date} (product CB4A-WPM-L4-DN-1)")
    print(f"       from {INPE_CATALOG_URL} into {work_dir}")
    print("    2. QGIS: Merge BAND1-4 (PAN excluded, separate bands, Int16) -> comp_colorida")
    print("    3. QGIS: GDAL Pansharpening, spectral = comp_colorida, panchromatic = BAND0")
    print("    4. run the cropper on the fused raster")
    print("    5. delete the raw + fused scene, then re-run this script")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="dataset size (default 100)")
    ap.add_argument("--tile-dir", type=Path, default=DEFAULT_TILE_DIR)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--next", action="store_true", help="only show the next scene to fetch")
    ap.add_argument("--csv", type=Path, help="also write the work list to this CSV")
    args = ap.parse_args()

    work, tiles = build_worklist(args.n, args.tile_dir)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")

    if args.next:
        print_next(work, tiles, args.work_dir)
        return 0

    n_test = int(round(args.n * TEST_FRACTION))
    print(f"N = {args.n}  ->  {n_test} test + {args.n - n_test} train, over {len(work)} scenes")

    ok, msg = check_disk(args.work_dir)
    print(("disk: " if ok else "DISK: ") + msg)
    print()

    print_table(work)
    print()

    done = work[work["status"] == "done"]
    tiles_done = int(work["done_n"].sum())
    print(
        f"progress: {len(done)}/{len(work)} scenes, {tiles_done}/{len(tiles)} tiles"
    )

    pending = work[work["status"] != "done"]
    if not pending.empty:
        remaining = len(pending)
        # The tail is expensive: a full ~28 GB download/fuse cycle per tile.
        tail = pending[pending["tiles_n"] == 1]
        print(
            f"{remaining} scenes left, ~{SCENE_PEAK_GB:.0f} GB of disk churn each "
            f"(~{remaining * 5.7:.0f} GB still to download)"
        )
        if len(tail):
            print(f"  of which {len(tail)} yield a single tile each")
        print()
        print_next(work, tiles, args.work_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
