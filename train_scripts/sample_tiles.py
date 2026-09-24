"""
Stratified tile sampling: global prefix-stable pick orders for test and train.

Goal: an N-tile dataset is exactly the first 0.8N of the train order plus the first
0.2N of the test order, for any N up to the maximum, with every class represented on
both sides, rare classes not starved, and picks random within each stratum.

Process (settled 2026-08-13, see notes/logbook.md):

  Stage 0  pool membership. Classes processed rarest -> commonest; each claims still-
           unclaimed tiles that qualify for it. Pools PARTITION the tile set, so a tile
           is drawn at most once and no cross-pool dedup is needed. Each pool shuffled
           once (fixed seed) — shuffled, not purity-descending, because purity-first
           ordering yields 2,855 open-water Lagoa dos Patos tiles before any shoreline.
  Split    each pool is cut into a test prefix and a train suffix (see SHARE/CAP below),
           so the split is a prefix/suffix of one shuffle rather than a second sampling.
  Stage 1  floor, coverage-greedy: at each step pick the tile crediting the most still-
           unmet classes, looking only at the first LOOKAHEAD tiles of each candidate
           pool so the choice stays mostly random. Rarity order breaks ties. Beats plain
           rarity-order filling at small budgets, which is where coverage is scarce.
  Stage 2  Sainte-Lague divisor. i* = argmax p_i/(2*c_i+1), draw next from pool_i*.

Credit is INTEGER and MULTI-CLASS: each pick increments c_j for every class the tile
qualifies for, not just the drawing class. Otherwise a class that co-occurs with rare
ones keeps drawing while believing itself underserved. Fractional credit was rejected —
it does not break prefix stability, but the divisors 1,3,5,7... are the odd integers and
Sainte-Lague's apportionment theory is stated over integer seat counts, so reals cost the
citable name. What it would have fixed (three 3%-rock tiles reading as "rock: done") is
fixed by the membership threshold instead.

Prefix stability: pools fixed and shuffled once, deterministic transitions, state
evolution independent of budget (budget is only a stopping condition), append-only. The
first N picks of a larger run are identical to a run of budget N.

The buffer prevents mechanical leakage (same field/river/road in a train and a test tile
via a shared edge), guaranteeing 2,048 m of terrain between splits. It does NOT buy
spatial independence: composition dissimilarity never plateaus (41% of decorrelation at
2 km, still only 88% at 93 km). Describe it as an adjacency buffer, not as independence.

Both sides use `compressed` target shares. Train needs it so rare classes have enough
examples to learn from. Test uses it too because representativeness is restored at
EVALUATION by weighting per-class IoU by true area share, and given that, more
rare-class test tiles is strictly better — tighter per-class estimates, which is the
point of stratifying a test set at all. If the aggregate is ever reported UNWEIGHTED,
`--shares-test proportional` is the honest setting instead (it halves L1 deviation from
the true distribution, 0.78 -> 0.40, at 20 test tiles).

No choice of shares matches the pixel-level distribution: p_i is a pixel share but whole
tiles are drawn and tiles are class mixtures, so the tiles-per-class -> pixels-per-class
map is not diagonal. Measured L1 plateaus near 0.45 even with the floor disabled.
Approximation is all that is wanted.

Usage:
    .venv/bin/python train_scripts/sample_tiles.py
    .venv/bin/python train_scripts/sample_tiles.py --shares-test proportional
    .venv/bin/python train_scripts/sample_tiles.py --compare      # one mode on both sides
    .venv/bin/python train_scripts/sample_tiles.py --exhaustion   # N where each class stops growing

Exits non-zero if any class ends up with zero presence on either side, at any reported
N at or above the minimum viable N (which the run reports).

Output:
    data/inpe/tile_split.csv
      tile_id, scene_id, split, test_rank, train_rank, coverage
      split      = test_eligible | buffer | train_eligible | unpooled
      coverage   = fraction of the tile inside its scene's valid imagery swath (always 1.0)
      test_rank  = 0-based position in the global test pick order (else blank)
      train_rank = 0-based position in the global train pick order (else blank)
"""

import argparse
import heapq
import random
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

N_CLASSES = 12
TILE_M    = 2048.0

# Buffer width, as a true edge-to-edge ground distance: sterilise every tile whose
# geometry lies STRICTLY CLOSER than BUFFER_M to a test-eligible tile.
#
# Must be a GEOMETRIC test, not a centroid or neighbour-count test. Tiles are built
# axis-aligned in each scene's own UTM with independent origins, so grids from adjacent
# scenes do not align. Measured same-scene neighbour distances are cleanly bimodal —
# 3,808 sampled pairs at exactly 0 m (the 8 ring-1 tiles, which share edges/corners),
# NOTHING between 0 and 2,040 m, then ring-2 starting at 2,046.1 m. Cross-scene distances
# are instead spread continuously: 142 sampled pairs overlapping at 0 m, 78 under 500 m,
# 95 in 500-1,000 m, 129 in 1,000-1,500 m.
#
# So any threshold in (0, 2046) isolates ring-1 exactly, and a higher one catches more
# misaligned cross-scene tiles. 2,000 m leaves a 46 m margin below the measured 2,046.1 m
# ring-2 floor, against a ~10 m spread from double reprojection (UTM -> 4326 -> 5880).
#
# Two bugs this replaces, both found by inspecting the exported QGIS layers:
#   1. A centroid radius of 1.5 tile-widths (3,072 m) sized for the 8 same-scene
#      neighbours let misaligned cross-scene train tiles fall outside the circle while
#      being physically adjacent — true minimum test->train gap was 296 m, and 618 of
#      1,341 test tiles sat closer than one tile width.
#   2. Replacing it with buffer(2048) + `intersects` over-buffered, because aligned ring-2
#      tiles sit at exactly the ring boundary and `intersects` counts touching geometry:
#      mean 18.5 buffer neighbours per test tile instead of 8, of which 8.3 were ring-2.
#
# Because ring-1 tiles are at distance 0, this also subsumes a tile-index (x+-1, y+-1)
# rule: it selects the same 8 same-scene neighbours AND the cross-scene ones an index
# rule cannot see, since indices from different scene grids are unrelated.
BUFFER_M  = 2000.0      # just under one tile width (2,048 m); see margin note above

# Centroid radius used only to count a tile's same-class neighbours when measuring how
# interior it is within its own cluster (see PERIPHERY_MAX_POOL). 1.5 tile-widths catches
# the four diagonals at sqrt(2) while excluding the next ring out at 2.0; measured mean
# neighbours per tile on the real grid is 8.1. Approximation is fine for that purpose.
MOORE_D   = TILE_M * 1.5

CLASS_NAMES = {
    1: 'Forest', 2: 'Agriculture', 3: 'Rangeland', 4: 'RockyOutcrop',
    5: 'PortoAlegre', 6: 'Urban', 7: 'Mining', 8: 'MajorLagoons',
    9: 'Water', 10: 'Aquaculture', 11: 'ShoreSand', 12: 'InlandSand',
}

# A tile qualifies for class k if it holds a meaningful SHARE of the class or a
# meaningful ABSOLUTE AREA of it. The fraction rule alone fails at both ends of the
# rarity range: Aquaculture never reaches 10% of any tile (max frac < 0.10, so its pool
# would be empty), and Mining at 10% demands 540 px = 48.6 ha, discarding 30 tiles
# holding genuine quarries (p25 = 155 px = 14 ha). The absolute floors below are set
# from the measured per-tile distributions; 1 strata pixel = 900 m2 = 0.09 ha.
# The two rules are OR'd, so the EFFECTIVE bar per class is the more permissive one.
# n_total is near-constant (~5,415 px; NoData is a value, not a missing pixel), so
# frac >= 0.10 is really an absolute area test: ~542 px = 48.7 ha of the class.
THRESH_FRAC = {k: 0.10 for k in range(1, N_CLASSES + 1)}
THRESH_PX   = {k: None for k in range(1, N_CLASSES + 1)}
# Aquaculture and Mining qualify by absolute area only. 48.7 ha of aquaculture does not
# exist anywhere in the biome, so a fraction rule is meaningless for it: measured, it
# qualifies 0 tiles. Mining's fraction-qualifiers are a verified strict subset of its
# px-qualifiers, so that clause was redundant too. Both set to None so the code states the
# rule actually enforced rather than leaving an inert 0.10 that reads like a purity bar.
THRESH_FRAC[10] = None
THRESH_FRAC[7]  = None
# PortoAlegre wants DENSE METRO, not anything clipping the city edge. At the default 0.10
# its pool held 21 tiles in the 10-20% band, and the N=100 draw returned a 10.0% tile
# (31% Agriculture, 27% Rangeland) for test and an 18% tile that is 80% MajorLagoons for
# train — both useless as representatives, while 59 tiles at 80-100% sat unused. The pool
# is shuffled with no purity preference, so a marginal tile is as likely as a core one;
# the fix has to be at the threshold. Raising it also INCREASES train supply (18 -> 29):
# a smaller test-eligible set scatters fewer tiles through the single dense cluster, so
# the buffer consumes less of it. Tiles dropped here fall through rarity order to the
# class they actually are (that 80%-lagoon tile lands in MajorLagoons).
THRESH_FRAC[5]  = 0.50
THRESH_PX[10] = 10      # Aquaculture: 0.9 ha. Drops 19 likely mixed-pixel tiles (min was 1 px), keeps 58.
THRESH_PX[7]  = 50      # Mining: 4.5 ha. Lifts the pool from 31 to 51 tiles.

FLOOR = {k: 1 for k in range(1, N_CLASSES + 1)}

# Test-eligible size per class = min(SHARE * pool, CAP). Both are required and do
# different jobs.
#
# SHARE keeps the split per-class instead of per-region. A flat region-wide cut (the
# q_test = total/(W+5) = 7.7% formula) let compressed shares over 3,818 picks want ~26 of
# Mining's 31 tiles, so test drained the class outright — measured zeros: Mining 31/0,
# PortoAlegre 145/0, InlandSand 177/1, RockyOutcrop 111/3.
#
# CAP protects rare-class TRAIN supply from buffer collateral damage. A larger
# test-eligible set means a larger buffer, and the buffer sterilises rare-class tiles
# indiscriminately. Measured with SHARE alone (no CAP): test-eligible 9,757, buffer
# 32,904, and rare-class train supply collapses to Aquaculture 6, Mining 4, InlandSand 22,
# RockyOutcrop 22. Mining test is 10 either way (SHARE binds), so the 4-vs-20 gap is
# purely collateral buffering. Note this is NOT about aggregate pool size, which is a
# non-criterion, and CAP is in fact WORSE for maximum dataset size: max N at 8:2 is 6,765
# with CAP=200 versus 8,260 without. It buys rare-class train supply and costs max N.
#
# CAP IS PERMANENT. Raising it later adds test-eligible tiles -> changes the buffer ->
# changes train-eligible -> invalidates the train order and the split of tiles already
# annotated. Note SHARE binds before CAP for every rare class, so CAP only affects the
# five common ones and raising it does not increase rare-class test supply.
SHARE = 0.20
CAP   = 200

# DISABLED (0). Optional mechanism: pick a concentrated class's test tiles from the cluster
# PERIPHERY so their buffers fall outward into other terrain instead of consuming the
# cluster. It raises train-eligible for concentrated classes a lot (Porto Alegre 18 -> 68,
# ShoreSand 150 -> 352, Urban 220 -> 361) while leaving test-eligible counts identical — it
# changes which tiles, never how many.
#
# Turned off because that headroom is not reachable at the planned scale while its cost is
# paid immediately. The earliest class to exhaust its train supply is Porto Alegre at
# N = 1,422 without this mechanism versus N = 2,765 with it (see --exhaustion), and the
# planned dataset is N = 100 to 500. Meanwhile at N = 100 the mechanism changes only 3 of
# the 7 affected classes by 1-2 tiles, yet it still forces the 1-2 Porto Alegre test tiles
# actually annotated to be metro-fringe rather than a random draw from the cluster — for a
# single test tile a representative draw measures the class more honestly.
#
# Like CAP this is a PERMANENT choice: it decides test-eligible membership, hence the
# buffer, hence train-eligible and both pick orders. Enabling it later re-splits everything
# and invalidates the split of any already-annotated tiles. Set to 1000 to enable for pools
# of at most that size; never enable it for large dispersed pools, where almost every tile
# is interior so periphery-first would systematically hand test the class's edge/transition
# tiles — a bias with no compensating benefit.
PERIPHERY_MAX_POOL = 0

LOOKAHEAD = 20     # stage-1 candidates per pool; bounds the coverage-greedy search so picks stay random
SEED      = 42

# Tier 1 (hand-picked in QGIS): Aquaculture and Mining — small enough to eyeball, and
# Aquaculture has no automatable purity signal. Manual picks count toward floors and are
# excluded from the pools so they cannot be drawn twice. Empty is a valid default.
MANUAL_TILE_IDS = set()


def qualifies(frac, counts):
    """(n_tiles, N_CLASSES+1) bool: does tile t qualify for class k."""
    q = np.zeros((len(frac), N_CLASSES + 1), dtype=bool)
    for k in range(1, N_CLASSES + 1):
        m = np.zeros(len(frac), dtype=bool)
        if THRESH_FRAC[k] is not None:
            m |= frac[:, k - 1] >= THRESH_FRAC[k]
        if THRESH_PX[k] is not None:
            m |= counts[:, k - 1] >= THRESH_PX[k]
        q[:, k] = m
    return q


def build_pools(qual, avail, rarity_order, seed):
    """Rarity-first claim. Returns {class: [tile indices]}, pools partition `avail`."""
    claimed = np.zeros(len(qual), dtype=bool)
    rng = random.Random(seed)
    pools = {}
    for k in rarity_order:
        elig = qual[:, k] & avail & ~claimed
        idx = np.flatnonzero(elig).tolist()
        rng.shuffle(idx)
        pools[k] = idx
        claimed |= elig
    return pools


def target_shares(counts, pool_tiles, mode, reference):
    """p_i, the divisor stage's target share per class.

    `reference` differs per side on purpose, each matching that side's job:

      'universe'  (TEST)  — the whole study area. Test must mirror reality, and its own
                            pool cannot serve as that reference because CAP truncates the
                            common classes' test-eligible pools to 1-8% of their size
                            while rare classes keep the full 20% SHARE. That leaves the
                            test-eligible pool structurally rare-heavy (measured L1 0.441
                            against the train-eligible pool), and a per-pool p_i inherits
                            the skew wholesale.
      'pool'      (TRAIN) — its own eligible pool, which IS essentially the study area
                            (76.8% of all tiles, composition close to truth). It is both a
                            valid reference and a more conservative pacer for scarce
                            classes, which preserves the graceful-degradation ceiling.

    Measured at N=100, L1 between pixel distributions, plus the N at which the first class
    exhausts its train supply:

        p_i(test)/p_i(train)     test~truth  train~truth  test~train   binds at N
        pool     / pool               0.781        0.357       0.428         1413
        universe / universe           0.265        0.465       0.249          579
        universe / pool  (CHOSEN)     0.265        0.357       0.155         1413

    The chosen pairing dominates the other two on every metric: test tracks reality ~3x
    better, the two sides agree ~2.8x better, train fidelity is unchanged, and the ceiling
    is unchanged. A universe reference on the TRAIN side would more than halve that ceiling
    (1413 -> 579) by draining Porto Alegre's 18 train tiles far faster.

    proportional : class pixel share. Starves every rare class past its floor —
                   measured at N=500 it left Aquaculture 1, Mining 1, RockyOutcrop 1.
    compressed   : sqrt of that, renormalised. Deliberately over-represents rare classes
                   (RockyOutcrop ~25x its 0.12% area share); recover true-ratio aggregate
                   metrics by reweighting per-class IoU at evaluation, not by sampling.

    p_i affects only the ORDER of picks. Eligibility, the buffer, the train/test split and
    rare-class supply are set by SHARE/CAP/buffer and are untouched by this.
    """
    px = (counts.sum(axis=0) if reference == 'universe'
          else counts[pool_tiles].sum(axis=0)).astype(float)
    p = px / px.sum() if px.sum() else np.full(N_CLASSES, 1.0 / N_CLASSES)
    if mode == 'compressed':
        p = np.sqrt(p)
        p = p / p.sum()
    return p


def order_picks(pools, qual, counts, budget, shares_mode, reference, seeded_credit=None, verbose=True):
    """Stages 1-2 over the given pools. Returns (pick indices, diagnostics)."""
    pool_tiles = np.array(sorted(t for v in pools.values() for t in v), dtype=int)
    if len(pool_tiles) == 0:
        return [], {}
    p = target_shares(counts, pool_tiles, shares_mode, reference)

    c = np.zeros(N_CLASSES + 1, dtype=int) if seeded_credit is None else seeded_credit.copy()
    picked = set()
    cursor = {k: 0 for k in pools}
    picks = []

    def advance(k):
        while cursor[k] < len(pools[k]) and pools[k][cursor[k]] in picked:
            cursor[k] += 1

    def take(k, t):
        picked.add(t)
        picks.append(t)
        c[:] = c + qual[t]

    # --- stage 1: coverage-greedy floor ---
    while len(picks) < budget:
        unmet = [k for k in pools if c[k] < FLOOR[k]]
        if not unmet:
            break
        best = None   # (-score, rarity_rank, pool_pos, k, t)
        for rank, k in enumerate(pools):
            if c[k] >= FLOOR[k]:
                continue
            advance(k)
            for off in range(LOOKAHEAD):
                pos = cursor[k] + off
                if pos >= len(pools[k]):
                    break
                t = pools[k][pos]
                if t in picked:
                    continue
                score = sum(1 for j in unmet if qual[t, j])
                cand = (-score, rank, off, k, t)
                if best is None or cand < best:
                    best = cand
        if best is None:
            break            # every unmet class has an exhausted pool
        take(best[3], best[4])
    n_floor = len(picks)

    # --- stage 2: Sainte-Lague divisor ---
    for k in pools:
        advance(k)
    heap = [(-p[k - 1] / (2 * c[k] + 1), k) for k in pools if cursor[k] < len(pools[k])]
    heapq.heapify(heap)
    while heap and len(picks) < budget:
        _, k = heapq.heappop(heap)
        advance(k)
        if cursor[k] >= len(pools[k]):
            continue
        take(k, pools[k][cursor[k]])
        advance(k)
        if cursor[k] < len(pools[k]):
            heapq.heappush(heap, (-p[k - 1] / (2 * c[k] + 1), k))
        # every class's priority may have shifted via multi-class credit
        heap = [(-p[j - 1] / (2 * c[j] + 1), j) for _, j in heap]
        heapq.heapify(heap)

    if verbose:
        print('  pools: ' + '  '.join(f'{CLASS_NAMES[k]}={len(pools[k]):,}' for k in pools))
        print(f'  floor ({sum(FLOOR.values())} across {N_CLASSES} classes) satisfied by {n_floor} tiles')
    return picks, dict(p=p, n_floor=n_floor)


def min_prefix_covering_all(picks, qual):
    """Shortest prefix of `picks` in which every class appears. None if never."""
    c = np.zeros(N_CLASSES + 1, dtype=int)
    for i, t in enumerate(picks):
        c += qual[t]
        if all(c[k] >= 1 for k in range(1, N_CLASSES + 1)):
            return i + 1
    return None


def coverage_table(picks_test, picks_train, qual, rarity_order, Ns=(20, 30, 50, 100, 500)):
    """Class coverage at 0.8N/0.2N cuts — the direct expression of the 'any N' requirement.

    Below min_N full coverage is arithmetically impossible, not a sampling failure: the
    test cut is only 0.2N tiles, and no fewer than `min_prefix_covering_all` tiles can
    hold all 12 classes (verified — an unbounded greedy search does no better than
    LOOKAHEAD=20). Rows below min_N are reported but do not fail the run.
    """
    need_te = min_prefix_covering_all(picks_test, qual) or 10 ** 9
    need_tr = min_prefix_covering_all(picks_train, qual) or 10 ** 9
    min_N = int(np.ceil(max(need_te / 0.2, need_tr / 0.8)))
    print(f'\nminimum viable N = {min_N} '
          f'(test needs {need_te} tiles = 0.2N, train needs {need_tr} = 0.8N)')

    print('=== class coverage at N (test = first 0.2N, train = first 0.8N) ===')
    hdr = ''.join(f'{CLASS_NAMES[k][:6]:>7}' for k in rarity_order)
    print(f'  {"":<16}{hdr}')
    ok = True
    for N in Ns:
        nte, ntr = int(round(0.2 * N)), int(round(0.8 * N))
        if nte > len(picks_test) or ntr > len(picks_train):
            break
        for lab, pk, n in (('test', picks_test, nte), ('train', picks_train, ntr)):
            sub = np.array(pk[:n], dtype=int)
            row = [int(qual[sub, k].sum()) for k in rarity_order]
            missing = sum(1 for v in row if v == 0)
            tag = f'N={N} {lab}'
            note = ''
            if missing:
                note = (f'   <-- {missing} missing (below min N={min_N}, expected)'
                        if N < min_N else f'   <-- {missing} MISSING')
                if N >= min_N:
                    ok = False
            print(f'  {tag:<16}' + ''.join(f'{v:>7,}' for v in row) + note)
    return ok, min_N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--composition', default='data/inpe/tile_strata_composition.csv')
    ap.add_argument('--tiles',       default='data/inpe/tiles_valid_swath.geojson')
    ap.add_argument('--out',         default='data/inpe/tile_split.csv')
    # Both sides use `compressed`. Train needs it so rare classes have enough examples to
    # learn from (proportional left Aquaculture 1, Mining 1, RockyOutcrop 1 even at N=500).
    # Test also uses it because representativeness is restored at EVALUATION by weighting
    # per-class IoU by true area share, and given that, more rare-class test tiles is
    # strictly better — tighter per-class IoU estimates, which is the point of stratifying
    # a test set at all. Kept as a flag since it is a reporting-dependent choice: if the
    # aggregate is ever reported UNWEIGHTED, --shares-test proportional is the honest
    # setting (it halves L1 from the true distribution, 0.78 -> 0.40, at 20 test tiles).
    #
    # No choice of shares matches the pixel-level distribution: p_i is a pixel share but
    # whole tiles are drawn and tiles are class mixtures, so the tiles-per-class ->
    # pixels-per-class map is not diagonal. Measured L1 plateaus near 0.45 even with the
    # floor disabled. Approximation is all that was wanted here.
    ap.add_argument('--shares-test',  default='compressed', choices=['compressed', 'proportional'])
    ap.add_argument('--shares-train', default='compressed', choices=['compressed', 'proportional'])
    ap.add_argument('--seed',        type=int, default=SEED)
    ap.add_argument('--compare',     action='store_true', help='report both share modes, write nothing')
    ap.add_argument('--exhaustion',  action='store_true',
                    help='also report the N at which each class runs out of train supply')
    args = ap.parse_args()

    d = pd.read_csv(args.composition)
    C = [f'c{k:02d}' for k in range(1, N_CLASSES + 1)]
    counts = d[C].to_numpy()
    frac = counts / np.maximum(d['n_total'].to_numpy()[:, None], 1)
    N = len(d)
    qual = qualifies(frac, counts)

    rarity_order = list(np.argsort(counts.sum(axis=0)) + 1)
    print(f'{N:,} tiles | rarity order: ' + ' '.join(CLASS_NAMES[k] for k in rarity_order))
    print('thresholds: ' + '  '.join(
        f'{CLASS_NAMES[k]}>=' + '|'.join(
            ([f'{THRESH_FRAC[k]:g}'] if THRESH_FRAC[k] is not None else []) +
            ([f'{THRESH_PX[k]}px'] if THRESH_PX[k] is not None else []))
        for k in rarity_order))
    if MANUAL_TILE_IDS:
        print(f'{len(MANUAL_TILE_IDS)} manual (tier-1) tiles')
    print()

    avail = ~d['tile_id'].isin(MANUAL_TILE_IDS).to_numpy()
    pools_all = build_pools(qual, avail, rarity_order, args.seed)

    g = gpd.read_file(args.tiles)[['tile_id', 'coverage', 'geometry']].to_crs('EPSG:5880')
    g = g.set_index('tile_id').loc[d['tile_id']].reset_index()
    cen = np.c_[g.geometry.centroid.x.values, g.geometry.centroid.y.values]
    tree = cKDTree(cen)

    # --- split each pool: test prefix, train suffix. Which tiles go to test is decided by
    # peripherality for concentrated classes (see PERIPHERY_MAX_POOL); the shuffled order is
    # then restored inside each side so the DRAW order stays random. ---
    test_pools, rest_pools = {}, {}
    n_periph = 0
    for k in rarity_order:
        pool = pools_all[k]
        n_k = max(1, min(int(round(SHARE * len(pool))), CAP)) if pool else 0
        if pool and len(pool) <= PERIPHERY_MAX_POOL:
            own = set(pool)
            # same-class neighbour count: low = cluster edge, high = deep interior
            deg = [len(set(tree.query_ball_point(cen[t], MOORE_D)) & own) - 1 for t in pool]
            split_order = [t for _, t in sorted(zip(deg, range(len(pool))))]
            split_order = [pool[i] for i in split_order]
            n_periph += 1
        else:
            split_order = pool
        chosen = set(split_order[:n_k])
        # restore shuffled order within each side
        test_pools[k] = [t for t in pool if t in chosen]
        rest_pools[k] = [t for t in pool if t not in chosen]
    test_idx = np.array(sorted(t for v in test_pools.values() for t in v), dtype=int)
    print(f'test-eligible: {len(test_idx):,} tiles (per class min({SHARE:.0%} of pool, {CAP}); '
          f'periphery-first for {n_periph} pools <= {PERIPHERY_MAX_POOL})')

    # --- buffer the WHOLE test-eligible set, not just what gets annotated: buffering only
    # the drawn tiles would mean growing the test budget later retroactively sterilises
    # train tiles, which is the budget-dependence this design exists to avoid ---
    # buffer(BUFFER_M) + intersects == "distance <= BUFFER_M"; BUFFER_M is set below the
    # ring-2 floor so the touching case cannot pull in the next ring out
    ring = gpd.GeoDataFrame(
        geometry=[g.geometry.iloc[test_idx].union_all().buffer(BUFFER_M)], crs=g.crs)
    hit = gpd.sjoin(g[['tile_id', 'geometry']], ring, how='inner', predicate='intersects')
    in_buf = g['tile_id'].isin(set(hit['tile_id'])).to_numpy()
    in_buf[test_idx] = False          # test tiles are inside their own ring
    buf_idx = np.flatnonzero(in_buf)
    print(f'buffer ({BUFFER_M:,.0f} m edge-to-edge): {len(buf_idx):,} tiles')

    train_pools = {k: [t for t in v if not in_buf[t]] for k, v in rest_pools.items()}
    n_train_elig = sum(len(v) for v in train_pools.values())
    print(f'train-eligible: {n_train_elig:,} tiles ({100 * n_train_elig / N:.1f}%)\n')

    if args.compare:
        for mode in ('proportional', 'compressed'):
            print(f'=== shares = {mode} ===')
            tp, td = order_picks(test_pools, qual, counts, len(test_idx), mode, 'universe')
            rp, _ = order_picks(train_pools, qual, counts, n_train_elig, mode, 'pool', verbose=False)
            coverage_table(tp, rp, qual, rarity_order)
            print()
        return

    print(f'=== test pick order (shares={args.shares_test}) ===')
    test_picks, tdiag = order_picks(test_pools, qual, counts, len(test_idx), args.shares_test, 'universe')
    print(f'=== train pick order (shares={args.shares_train}) ===')
    train_picks, _ = order_picks(train_pools, qual, counts, n_train_elig, args.shares_train, 'pool')

    ok_cov, min_N = coverage_table(test_picks, train_picks, qual, rarity_order)
    if args.exhaustion:
        te_avail = np.zeros(N, bool); te_avail[test_idx] = True
        tr_avail = np.zeros(N, bool); tr_avail[np.array(train_picks, dtype=int)] = True
        report_exhaustion(
            exhaustion_table(test_picks, te_avail, qual, rarity_order, 0.2, 'test') +
            exhaustion_table(train_picks, tr_avail, qual, rarity_order, 0.8, 'train'))

    split = np.full(N, 'unpooled', dtype=object)
    tr = np.zeros(N, bool)
    tr[np.array(train_picks, dtype=int)] = True
    split[tr] = 'train_eligible'
    split[in_buf] = 'buffer'
    split[test_idx] = 'test_eligible'

    out = pd.DataFrame({'tile_id': d['tile_id'], 'scene_id': d['scene_id'], 'split': split})
    out['test_rank'] = pd.Series({t: i for i, t in enumerate(test_picks)}).reindex(range(N)).values
    out['train_rank'] = pd.Series({t: i for i, t in enumerate(train_picks)}).reindex(range(N)).values
    # imagery coverage of the tile by its scene's valid swath; 1.0 for every surviving tile,
    # carried here so the cropper can assert it without opening the geometry file
    out['coverage'] = g.set_index('tile_id')['coverage'].reindex(out['tile_id']).values
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'\n-> {args.out}')
    print(out['split'].value_counts().to_string())

    ok_split = verify(out, qual, rarity_order)
    if not (ok_split and ok_cov):
        print('\nFAILED: see markers above', file=sys.stderr)
        sys.exit(1)
    print(f'\nOK: every class present on both sides at every N >= {min_N}')


def exhaustion_table(picks, avail, qual, rarity_order, split_frac, side):
    """N at which each class runs out of supply on one side.

    A class's supply is every eligible tile qualifying for it — not just the tiles its own
    pool claimed, since a tile claimed by a rarer class can still contain it. Walk that
    side's pick order counting qualifying picks; when the count reaches the total
    available, the class is exhausted, and N = ceil(picks / split_frac) because that side
    is `split_frac` of the dataset.

    Past a class's exhaustion N, growing the dataset adds no further examples of it, so its
    share decays. Degradation is therefore gradual, not a cliff: nothing breaks, the
    scarcest classes simply stop growing one by one.
    """
    rows = []
    for k in rarity_order:
        total = int(qual[avail, k].sum())
        cum, exh = 0, None
        for i, t in enumerate(picks):
            if qual[t, k]:
                cum += 1
                if cum >= total:
                    exh = int(np.ceil((i + 1) / split_frac))
                    break
        rows.append((k, side, total, exh))
    return rows


def report_exhaustion(rows):
    print('\n=== N at which each class exhausts its supply (degradation ceiling) ===')
    print(f'  {"class":<15}{"side":>7}{"avail":>8}{"exhausted at N":>16}')
    for k, side, total, exh in sorted(rows, key=lambda r: (r[3] is None, r[3] or 0)):
        print(f'  {CLASS_NAMES[k]:<15}{side:>7}{total:>8,}' +
              (f'{exh:>16,}' if exh else f'{"never":>16}'))
    finite = [r for r in rows if r[3]]
    if finite:
        k, side, _, e = min(finite, key=lambda r: r[3])
        print(f'  -> first to bind: {CLASS_NAMES[k]} ({side}) at N = {e:,}. '
              f'Below this every class still grows with N.')


def verify(out, qual, rarity_order):
    """Post-hoc gate: every class must have nonzero presence on both sides."""
    print('\n=== per-class presence, both sides ===')
    te = (out['split'] == 'test_eligible').to_numpy()
    tr = (out['split'] == 'train_eligible').to_numpy()
    print(f'  {"class":<14}{"test-elig":>11}{"train-elig":>12}')
    ok = True
    for k in rarity_order:
        m = qual[:, k]
        a, b = int((m & te).sum()), int((m & tr).sum())
        if a == 0 or b == 0:
            ok = False
        print(f'  {CLASS_NAMES[k]:<14}{a:>11,}{b:>12,}' + ('   <-- ZERO' if a == 0 or b == 0 else ''))
    return ok


if __name__ == '__main__':
    main()
