#!/usr/bin/env python3
"""
Broader version of detector_grid_scan.py: instead of just 2 hand-picked
detectors (center, edge), sweep 8 detectors spanning the full range of
distances from the mountain bbox center, on the same L-BFGS-best layout.
Tests whether the earlier flat-plateau finding (U range ~1-1.4 pts across the
whole grid for 2 detectors) generalizes across the population, or was
specific to those two picks. Coarser grid + fewer batches than the original
scan to keep runtime reasonable across 8 detectors.
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (layout paths live in one place)
import modules_v6  # noqa: F401

from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
)
from modules_v6.legacy_core.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models
from modules_v6.tr_geometry_ne import project_to_mountain_ne

DEVICE = torch.device("cpu")
BATCH_SEED_BASE = 1000     # avoid seed=42, the training/scoring batch (known outlier)
BATCH_SIZE = 512
N_BATCHES = 5
GRID_N = 15
N_DET_SWEEP = 8

print("=" * 70)
print("Broad detector grid scan: 8 detectors spanning near-center to far-edge")
print("=" * 70)

fnn, recon = load_models(DEVICE, fnn_folder=FNN_FOLDER, recon_dir=RECON_FOLDER + "_deepsets")
mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)

primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                         weights_only=False).float()
n_total = primary_all.shape[0]


def fresh_batch(seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n_total, (BATCH_SIZE,), generator=g)
    return primary_all[idx].to(DEVICE)


BATCHES = [fresh_batch(BATCH_SEED_BASE + b) for b in range(N_BATCHES)]


@torch.no_grad()
def eval_U_mean(x, y):
    Us = [float(utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon)[0].item()) for p in BATCHES]
    return float(np.mean(Us))


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


lbfgs_x, lbfgs_y, lbfgs_U_saved = load_layout(
    _layouts.primary())
base_U = eval_U_mean(lbfgs_x, lbfgs_y)
print(f"L-BFGS best U (saved): {lbfgs_U_saved:.4f}  (re-evaluated, {N_BATCHES} fresh batches): {base_U:.4f}")

cn = 0.5 * (mountain.n_min + mountain.n_max)
ce = 0.5 * (mountain.east_lo + mountain.east_hi)
dist = ((lbfgs_x - cn) ** 2 + (lbfgs_y - ce) ** 2).sqrt()
order = torch.argsort(dist)  # ascending: nearest-to-center first
# Pick N_DET_SWEEP indices evenly spaced through the sorted-by-distance order,
# so we get a spread from near-center to far-edge, not just the two extremes.
pick_positions = np.linspace(0, N_DETECTORS - 1, N_DET_SWEEP).round().astype(int)
sweep_indices = [int(order[p]) for p in pick_positions]
print(f"Sweeping detectors (sorted by distance from center) at percentile positions {pick_positions.tolist()}:")
for idx in sweep_indices:
    print(f"  idx={idx:3d}  dist_to_center={dist[idx]:.1f}m")

n_grid = np.linspace(mountain.n_min, mountain.n_max, GRID_N).astype(np.float32)
e_grid = np.linspace(mountain.east_lo, mountain.east_hi, GRID_N).astype(np.float32)
NN, EE = np.meshgrid(n_grid, e_grid, indexing="ij")
grid_N_flat = torch.as_tensor(NN.reshape(-1), dtype=torch.float32)
grid_E_flat = torch.as_tensor(EE.reshape(-1), dtype=torch.float32)
grid_N_proj, grid_E_proj = project_to_mountain_ne(mountain, grid_N_flat, grid_E_flat)

results = {}
for idx in sweep_indices:
    print(f"\n[idx={idx}] dist_to_center={dist[idx]:.1f}m  sweeping {GRID_N}x{GRID_N} grid ...")
    t0 = time.time()
    n_pts = GRID_N * GRID_N
    U_grid = np.zeros(n_pts, dtype=np.float32)
    orig_N, orig_E = float(lbfgs_x[idx]), float(lbfgs_y[idx])
    for k in range(n_pts):
        x_mod = lbfgs_x.clone()
        y_mod = lbfgs_y.clone()
        x_mod[idx] = grid_N_proj[k]
        y_mod[idx] = grid_E_proj[k]
        U_grid[k] = eval_U_mean(x_mod, y_mod)
    dt = time.time() - t0
    argmax_k = int(U_grid.argmax())
    argmax_N, argmax_E = float(grid_N_proj[argmax_k]), float(grid_E_proj[argmax_k])
    disp = ((argmax_N - orig_N) ** 2 + (argmax_E - orig_E) ** 2) ** 0.5
    rng = float(U_grid.max() - U_grid.min())
    print(f"[idx={idx}] done in {dt:.0f}s.  U range=[{U_grid.min():.3f}, {U_grid.max():.3f}]  "
          f"span={rng:.3f}  base U={base_U:.3f}  argmax gain={U_grid.max()-base_U:+.3f}  "
          f"argmax displacement={disp:.1f}m")
    results[str(idx)] = dict(
        idx=idx, dist_to_center=float(dist[idx]), orig_N=orig_N, orig_E=orig_E,
        base_U=base_U, U_min=float(U_grid.min()), U_max=float(U_grid.max()),
        span=rng, argmax_gain=float(U_grid.max() - base_U), argmax_displacement=disp,
        time_s=dt,
    )

out_json = os.path.join(_layouts.results_dir(), "detector_grid_scan_broad_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_json}")

# Summary plot: span (max-min) vs distance from center.
dists = [results[k]["dist_to_center"] for k in results]
spans = [results[k]["span"] for k in results]
gains = [results[k]["argmax_gain"] for k in results]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(dists, spans)
axes[0].set_xlabel("detector distance from bbox center (m)")
axes[0].set_ylabel("U range across full grid sweep (span)")
axes[0].set_title("How much does moving 1 detector anywhere\nchange U, vs. its distance from center?")
axes[1].scatter(dists, gains)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("detector distance from bbox center (m)")
axes[1].set_ylabel("grid-argmax U - base U (gain)")
axes[1].set_title("Is the optimized position already\nthe best spot for this detector?")
fig.tight_layout()
out_png = os.path.join(_layouts.results_dir(), "detector_grid_scan_broad_summary.png")
fig.savefig(out_png, dpi=150)
print(f"[plot] wrote {out_png}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for idx in sweep_indices:
    r = results[str(idx)]
    print(f"  idx={idx:3d}  dist={r['dist_to_center']:7.1f}m  span={r['span']:.3f}  "
          f"gain={r['argmax_gain']:+.3f}  disp={r['argmax_displacement']:7.1f}m")
print(f"\nMean span across all {N_DET_SWEEP} detectors: {np.mean(spans):.3f}  "
      f"(base U={base_U:.3f}, so relative span = {100*np.mean(spans)/base_U:.2f}%)")
