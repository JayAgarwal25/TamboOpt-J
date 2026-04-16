"""Build the FNN training dataset for v6.

Reads the cached AllShowers corpus, samples 5 layouts per shower (grid +
center Gaussian σ=200 + concentric rings at three radii), runs v4's
`GetCounts_planeaware` as a label generator, and writes the resulting
(primary, xy, E, T) tensors plus a z-score normalization dict.

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 01_build_dataset.py

Outputs land in `outputs/v6_run_01/`.
"""
import os
import sys
import time

# Make `modules_v6` importable when running this file from the v6 folder
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import modules_v6   # triggers sys.path injection for v3 + v4
from modules_v6.fnn_surrogate import (
    build_training_pairs, compute_normalization,
)
from modules_v6.constants import (
    SHOWER_CACHE, GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, NUM_SHOWERS,
    BATCH_SIZE_TRAIN, TRAINING_DATASET_FOLDER
)
from modules_v4.tr_geometry    import load_tr_mountain
from modules_v4.tr_surface_map import SurfaceEastMap


# ── Config ───────────────────────────────────────────────────────────────────
MAX_SHOWERS = NUM_SHOWERS
SEED        = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(TRAINING_DATASET_FOLDER, exist_ok=True)

    print("=" * 72)
    print(f"v6/01_build_dataset.py")
    print("=" * 72)
    print(f"shower cache : {SHOWER_CACHE}")
    print(f"geometry     : {GEOMETRY_PATH}")
    print(f"output dir   : {TRAINING_DATASET_FOLDER}")
    print(f"batch size   : {BATCH_SIZE_TRAIN}")
    print(f"max showers  : {MAX_SHOWERS}")
    print(f"device       : {DEVICE}")

    # Mountain + surface map (East = f(N, Up))
    t0 = time.time()
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    surface = SurfaceEastMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    print(f"[load] mountain + surface in {time.time() - t0:.1f}s")
    print(f"  N=[{mountain.n_min:.0f}, {mountain.n_max:.0f}]  "
          f"Up=[{mountain.u_min:.0f}, {mountain.u_max:.0f}]  "
          f"East=[{mountain.east_lo:.0f}, {mountain.east_hi:.0f}]")

    # Build training pairs
    t0 = time.time()
    primary, xy, E, T, strat = build_training_pairs(
        mountain=mountain,
        surface=surface,
        shower_cache_path=os.path.join(SHOWER_CACHE, f"cashed_showers_{NUM_SHOWERS}.pt"),
        batch_size=BATCH_SIZE_TRAIN,
        max_showers=MAX_SHOWERS,
        seed=SEED,
        device=DEVICE,
        verbose=True,
    )
    print(f"[build] training pairs in {time.time() - t0:.1f}s")
    print(f"  primary : {tuple(primary.shape)}  dtype={primary.dtype}")
    print(f"  xy      : {tuple(xy.shape)}       dtype={xy.dtype}")
    print(f"  E       : {tuple(E.shape)}        dtype={E.dtype}")
    print(f"  T       : {tuple(T.shape)}        dtype={T.dtype}")
    print(f"  strat   : {tuple(strat.shape)}    unique={sorted(strat.unique().tolist())}")

    # Log-scale E and T for better FNN training (compresses heavy right tail)
    E = torch.log1p(E)
    T = torch.log1p(T)
    print(f"[log1p] E range [{E.min():.4g}, {E.max():.4g}]  "
          f"T range [{T.min():.4g}, {T.max():.4g}]")

    # Sanity: non-zero E on at least some samples
    n_nonzero = int((E.abs().sum(dim=1) > 0).sum())
    print(f"  samples with any nonzero E : {n_nonzero}/{E.shape[0]}")

    # Z-score stats over the whole training corpus
    stats = compute_normalization(primary, xy, E, T)
    print(f"[norm] in_mean[:5]  = {stats['in_mean'][:5].tolist()}")
    print(f"[norm] in_std[:5]   = {stats['in_std'][:5].tolist()}")
    print(f"[norm] out_mean (E) = {stats['out_mean'][:5].tolist()} ...")
    print(f"[norm] out_std  (E) = {stats['out_std'][:5].tolist()} ...")

    # Persist
    t0 = time.time()
    torch.save(primary, os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"))
    torch.save(xy,      os.path.join(TRAINING_DATASET_FOLDER, "xy.pt"))
    torch.save(E,       os.path.join(TRAINING_DATASET_FOLDER, "E.pt"))
    torch.save(T,       os.path.join(TRAINING_DATASET_FOLDER, "T.pt"))
    torch.save(strat,   os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt"))
    torch.save(stats,   os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt"))
    print(f"[save] tensors in {time.time() - t0:.1f}s  ->  {TRAINING_DATASET_FOLDER}")


if __name__ == "__main__":
    main()
