"""Build the FNN training dataset for v6 — (North, East) detector convention.

North–East sibling of `01_build_dataset.py`, kept step-for-step identical so the
two diff cleanly. The only changes are the (North, East) pieces: detectors are
placed by horizontal map coords (North, East), the mountain extrapolates the
height Up = g(North, East) (`SurfaceUpMap`), and labels come from the NE
`build_training_pairs`. Stored `xy = (North, East)`. Writes to a dedicated
`..._northeast` folder so the original (North, Up) corpus is never overwritten.

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 01_build_dataset_northeast.py

Outputs land in `<RUN_LOCATION>/test_v6_run_01_northeast/`.

Infill mode (adaptive surrogate retraining):

    python 01_build_dataset_northeast.py \\
        --infill_center /path/to/L_star_r0.pt \\
        --infill_sigma 200.0 \\
        --n_infill_layouts 300 \\
        --n_showers_per_infill 1000 \\
        --round 1

L_star_r0.pt is a (N_DETECTORS, 2) tensor with (North, East) per detector, or
a dict with "x" (North) and "y" (East) keys (layout_best.pt format from step 4).
Outputs land in `<RUN_LOCATION>/test_v6_run_01_northeast_infill_r{round}/`.
"""
import argparse
import os
import sys
import time

# Make `modules_v6` importable when running this file from the external folder
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6   # triggers sys.path injection for v3 + v4
from modules_v6.fnn_surrogate_ne import (
    build_training_pairs, compute_normalization, compute_labels_batch,
)
from modules_v6.fnn_surrogate import _load_species_sidecar, encode_primary
from modules_v6.tr_geometry_ne import project_to_mountain_ne
from modules_v6.constants import (
    SHOWER_CACHE, GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, NUM_SHOWERS,
    N_DETECTORS, PRIMARY_DIM,
    BATCH_SIZE_TRAIN, RUN_LOCATION, RECENTER_TO_MOUNTAIN,
    DUAL_SHOWER_CACHE_PATH, DATASET_FRACTION,
)
from modules_v4.tr_geometry    import load_tr_mountain
from modules_v6.tr_surface_map_ne import SurfaceUpMap


# ── Config ───────────────────────────────────────────────────────────────────
# Dedicated output dir (notable name) — never overwrite the (North, Up) corpus.
TRAINING_DATASET_FOLDER = os.path.join(RUN_LOCATION, "test_v6_run_01_northeast")
# Paired dual-species corpus holds 2*NUM_SHOWERS rows (electron block then muon
# block, same primaries); 02 splits them per species via the species_ids.pt
# sidecar (the primary pdg feature now carries the EM/hadronic class).
# DATASET_FRACTION caps how many rows are loaded (split evenly across species) so
# the build fits in RAM — see modules_v6/constants.py.
MAX_SHOWERS = int(DATASET_FRACTION * 2 * NUM_SHOWERS)
SEED        = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RECENTER_TO_MOUNTAIN is imported from modules_v6.constants — edit it there.


def _load_center_layout(path: str):
    """Load a layout tensor from path.

    Accepts either a plain (N_DETECTORS, 2) float32 tensor (col 0 = North,
    col 1 = East) or a dict with "x" (North) and "y" (East) keys, as written
    by 04_optimize_lbfgs_ensemble.py's --save_best_layout flag.
    Returns (N_t, E_t) each of shape (N_DETECTORS,).
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        return raw["x"].float().reshape(-1), raw["y"].float().reshape(-1)
    t = raw.float()
    if t.dim() == 2 and t.shape[1] == 2:
        return t[:, 0], t[:, 1]
    raise ValueError(
        f"Cannot interpret layout from {path}: expected (N, 2) tensor or dict "
        f"with 'x'/'y' keys, got {type(raw)} shape={getattr(raw, 'shape', '?')}"
    )


def build_infill_pairs(mountain, surface, shower_cache_path,
                       layouts_N, layouts_E,
                       n_showers_per_layout, batch_size, seed, device, verbose):
    """Build infill training pairs from pre-sampled layouts near L*.

    layouts_N, layouts_E : (n_layouts, N_DETECTORS) tensors, (North, East).
    For each layout, runs n_showers_per_layout showers through the kernel.
    Output is strategy-major: layout 0's pairs first, then layout 1's, etc.,
    matching the ordering expected by shower_level_split in 02_train_fnn.py.
    Total output pairs = n_layouts * n_showers_per_layout.
    """
    import showerdata

    n_layouts = int(layouts_N.shape[0])
    meta   = showerdata.load_inc_particles(shower_cache_path)
    n_file = meta.pdg.shape[0]
    per_sp = n_file // 2
    k_sp   = min(n_showers_per_layout // 2, per_sp)   # showers per species per layout
    actual_n = 2 * k_sp                                # total showers per layout

    keep_idx = np.concatenate([np.arange(0, k_sp),
                               np.arange(per_sp, per_sp + k_sp)])
    dirs   = torch.as_tensor(meta.directions[keep_idx], dtype=torch.float32)
    energs = torch.as_tensor(meta.energies[keep_idx],   dtype=torch.float32)
    pdg    = torch.as_tensor(meta.pdg[keep_idx],        dtype=torch.long)
    primaries_all = encode_primary(dirs, energs, pdg)      # (actual_n, 5)
    species_all   = _load_species_sidecar(shower_cache_path, keep_idx)  # (actual_n,)

    if RECENTER_TO_MOUNTAIN:
        mtn_cx = 0.5 * (mountain.n_min + mountain.n_max)
        mtn_cy = 0.5 * (mountain.u_min + mountain.u_max)

    n_pairs = n_layouts * actual_n
    out_primary = torch.empty((n_pairs, PRIMARY_DIM), dtype=torch.float32)
    out_xy      = torch.empty((n_pairs, N_DETECTORS, 2), dtype=torch.float32)
    out_E       = torch.empty((n_pairs, N_DETECTORS), dtype=torch.float32)
    out_T       = torch.empty((n_pairs, N_DETECTORS), dtype=torch.float32)
    out_strat   = torch.empty((n_pairs,), dtype=torch.int64)
    out_species = torch.empty((n_pairs,), dtype=torch.int64)

    load_chunk = max(int(batch_size), (4096 // int(batch_size)) * int(batch_size))

    for lay_idx in range(n_layouts):
        x_det = layouts_N[lay_idx].float().to(device)
        y_det = layouts_E[lay_idx].float().to(device)
        ds_offset = lay_idx * actual_n

        for tag, file_start, ds_start in (("e", 0, 0), ("mu", per_sp, k_sp)):
            for c_lo in range(0, k_sp, load_chunk):
                c_hi = min(c_lo + load_chunk, k_sp)
                csz  = c_hi - c_lo

                sub = showerdata.load(shower_cache_path,
                                      start=file_start + c_lo,
                                      stop=file_start + c_hi)
                clouds_chunk = torch.as_tensor(sub.points, dtype=torch.float32)
                del sub

                bad = ~torch.isfinite(clouds_chunk).all(dim=-1)
                if int(bad.sum()):
                    clouds_chunk[bad] = 0.0

                if RECENTER_TO_MOUNTAIN:
                    mask  = (clouds_chunk[:, :, 3] > 0).float()
                    w_sum = mask.sum(dim=1).clamp(min=1.0)
                    cx = (clouds_chunk[:, :, 0] * mask).sum(dim=1) / w_sum
                    cy = (clouds_chunk[:, :, 1] * mask).sum(dim=1) / w_sum
                    dx = (mtn_cx - cx).view(-1, 1)
                    dy = (mtn_cy - cy).view(-1, 1)
                    clouds_chunk[..., 0] = clouds_chunk[..., 0] + dx * mask
                    clouds_chunk[..., 1] = clouds_chunk[..., 1] + dy * mask

                for sb_lo in range(0, csz, batch_size):
                    sb_hi = min(sb_lo + batch_size, csz)
                    B = sb_hi - sb_lo

                    clouds = clouds_chunk[sb_lo:sb_hi].to(device)
                    E_b, T_b = compute_labels_batch(clouds, x_det, y_det, surface)
                    E_b = torch.nan_to_num(E_b, nan=0.0, posinf=0.0, neginf=0.0)
                    T_b = torch.nan_to_num(T_b, nan=0.0, posinf=0.0, neginf=0.0)

                    local_lo = ds_start + c_lo + sb_lo
                    local_hi = ds_start + c_lo + sb_hi
                    dst = slice(ds_offset + local_lo, ds_offset + local_hi)
                    out_primary[dst]     = primaries_all[local_lo:local_hi]
                    out_xy[dst, :, 0]   = x_det.cpu().unsqueeze(0).expand(B, -1)
                    out_xy[dst, :, 1]   = y_det.cpu().unsqueeze(0).expand(B, -1)
                    out_E[dst]           = E_b.cpu()
                    out_T[dst]           = T_b.cpu()
                    out_strat[dst]       = lay_idx
                    out_species[dst]     = species_all[local_lo:local_hi]

                del clouds_chunk

        if verbose:
            print(f"[infill] layout {lay_idx + 1}/{n_layouts} done")

    return out_primary, out_xy, out_E, out_T, out_strat, out_species


def _save_dataset(folder, primary, xy, E, T, strat, species, stats):
    """Persist all dataset tensors to folder."""
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    torch.save(primary, os.path.join(folder, "primary.pt"))
    torch.save(xy,      os.path.join(folder, "xy.pt"))
    torch.save(E,       os.path.join(folder, "E.pt"))
    torch.save(T,       os.path.join(folder, "T.pt"))
    torch.save(strat,   os.path.join(folder, "strategy_ids.pt"))
    torch.save(species, os.path.join(folder, "species_ids.pt"))
    torch.save(stats,   os.path.join(folder, "norm_stats.pt"))
    print(f"[save] tensors in {time.time() - t0:.1f}s  ->  {folder}")


def main():
    ap = argparse.ArgumentParser(
        description="Build FNN training dataset (NE convention). "
                    "Pass --infill_center to run adaptive-retraining infill mode.")
    ap.add_argument("--infill_center", type=str, default=None,
                    help="Path to a layout .pt file (N_DETECTORS x 2 tensor or "
                         "dict with 'x'/'y' keys). Enables infill mode.")
    ap.add_argument("--infill_sigma", type=float, default=200.0,
                    help="Std-dev of Gaussian perturbation per detector (metres). "
                         "Default: 200.")
    ap.add_argument("--n_infill_layouts", type=int, default=300,
                    help="Number of perturbed layouts to generate. Default: 300.")
    ap.add_argument("--n_showers_per_infill", type=int, default=1000,
                    help="Showers per infill layout (both species combined). "
                         "Default: 1000 -> ~300k total pairs for n_infill_layouts=300.")
    ap.add_argument("--round", type=int, default=1,
                    help="Adaptive-loop round number used in the output folder name.")
    args = ap.parse_args()

    print("=" * 72)
    print(f"v6/01_build_dataset_northeast.py")
    print("=" * 72)
    print(f"shower cache : {DUAL_SHOWER_CACHE_PATH}")
    print(f"geometry     : {GEOMETRY_PATH}")
    print(f"device       : {DEVICE}")
    print(f"recenter     : {RECENTER_TO_MOUNTAIN}")

    # Mountain + surface map (Up = g(N, East))
    t0 = time.time()
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    print(f"[load] mountain + surface in {time.time() - t0:.1f}s")
    print(f"  N=[{mountain.n_min:.0f}, {mountain.n_max:.0f}]  "
          f"Up=[{mountain.u_min:.0f}, {mountain.u_max:.0f}]  "
          f"East=[{mountain.east_lo:.0f}, {mountain.east_hi:.0f}]")

    if args.infill_center:
        # ── Infill mode ────────────────────────────────────────────────────
        output_folder = TRAINING_DATASET_FOLDER + f"_infill_r{args.round}"
        print(f"mode         : INFILL  round={args.round}")
        print(f"center       : {args.infill_center}")
        print(f"sigma        : {args.infill_sigma} m")
        print(f"n_layouts    : {args.n_infill_layouts}")
        print(f"showers/layout: {args.n_showers_per_infill}")
        print(f"output dir   : {output_folder}")

        center_N, center_E = _load_center_layout(args.infill_center)
        print(f"[center] N in [{center_N.min():.1f}, {center_N.max():.1f}]  "
              f"E in [{center_E.min():.1f}, {center_E.max():.1f}]")

        # Sample n_infill_layouts Gaussian perturbations of the center layout,
        # then snap each to the mountain surface via project_to_mountain_ne.
        rng = np.random.default_rng(SEED)
        sigma = args.infill_sigma
        n_lay = args.n_infill_layouts
        layouts_N = torch.empty(n_lay, N_DETECTORS, dtype=torch.float32)
        layouts_E = torch.empty(n_lay, N_DETECTORS, dtype=torch.float32)
        for i in range(n_lay):
            dN = torch.tensor(rng.normal(0.0, sigma, N_DETECTORS).astype(np.float32))
            dE = torch.tensor(rng.normal(0.0, sigma, N_DETECTORS).astype(np.float32))
            N_i = (center_N + dN).clamp(mountain.n_min, mountain.n_max)
            E_i = (center_E + dE).clamp(mountain.east_lo, mountain.east_hi)
            # Snap to nearest mountain centroid for any out-of-surface points.
            layouts_N[i], layouts_E[i] = project_to_mountain_ne(mountain, N_i, E_i)
        print(f"[infill] sampled {n_lay} perturbed layouts  "
              f"sigma={sigma:.0f}m")

        t0 = time.time()
        primary, xy, E, T, strat, species = build_infill_pairs(
            mountain=mountain,
            surface=surface,
            shower_cache_path=DUAL_SHOWER_CACHE_PATH,
            layouts_N=layouts_N,
            layouts_E=layouts_E,
            n_showers_per_layout=args.n_showers_per_infill,
            batch_size=BATCH_SIZE_TRAIN,
            seed=SEED,
            device=DEVICE,
            verbose=True,
        )
        print(f"[build] infill pairs in {time.time() - t0:.1f}s")
    else:
        # ── Normal mode ────────────────────────────────────────────────────
        output_folder = TRAINING_DATASET_FOLDER
        print(f"mode         : NORMAL (7 fixed strategies)")
        print(f"output dir   : {output_folder}")
        print(f"batch size   : {BATCH_SIZE_TRAIN}")
        print(f"max showers  : {MAX_SHOWERS}")

        t0 = time.time()
        primary, xy, E, T, strat, species = build_training_pairs(
            mountain=mountain,
            surface=surface,
            shower_cache_path=DUAL_SHOWER_CACHE_PATH,
            batch_size=BATCH_SIZE_TRAIN,
            max_showers=MAX_SHOWERS,
            seed=SEED,
            device=DEVICE,
            verbose=True,
            recenter_to_mountain=RECENTER_TO_MOUNTAIN,
        )
        print(f"[build] training pairs in {time.time() - t0:.1f}s")

    print(f"  primary : {tuple(primary.shape)}  dtype={primary.dtype}")
    print(f"  xy      : {tuple(xy.shape)}       dtype={xy.dtype}   (North, East)")
    print(f"  E       : {tuple(E.shape)}        dtype={E.dtype}")
    print(f"  T       : {tuple(T.shape)}        dtype={T.dtype}")
    print(f"  strat   : {tuple(strat.shape)}    unique={sorted(strat.unique().tolist())}")
    print(f"  species : {tuple(species.shape)}  unique={sorted(species.unique().tolist())}")

    # Log-scale E for better FNN training (compresses heavy right tail)
    E = torch.log1p(E)

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

    _save_dataset(output_folder, primary, xy, E, T, strat, species, stats)


if __name__ == "__main__":
    main()
