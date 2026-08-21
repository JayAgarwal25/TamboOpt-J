#!/usr/bin/env python3
"""
Mode connectivity check (Garipov et al. 2018 style, linear-path version):
are two independently-obtained good layouts on the SAME connected plateau,
or are they separate basins that happen to score similarly?

Uses two genuinely independent, trustworthy L-BFGS runs (from different
initial schemes -- "grid" and "center" -- NOT GES, which the Hessian probe
showed sits on a numerical singularity, not a real optimum).

Because the FNN/recon are permutation-invariant, detector index i in run A
and index i in run B are not necessarily the same physical spot -- so the
two layouts are Hungarian-matched (nearest-position assignment, reusing
opt_core.align_to_reference) BEFORE linear interpolation, otherwise a
meaningless index-to-index interpolation could show a fake "barrier" that's
really just a mismatched-pairing artifact.

If U stays high along the straight-line path between the aligned layouts,
that's strong evidence they're on one connected plateau. A real dip in the
middle would mean they're separated by a genuine barrier.
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (layout paths live in one place)
import modules_v6  # noqa: F401

from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models, align_to_reference
from modules_v6.tr_geometry_ne import project_to_mountain_ne

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
BATCH_SEED_BASE = 1000
BATCH_SIZE = 512
N_BATCHES = 6            # cheap experiment (few points), afford more batches for precision
N_STEPS = 41             # t in [0, 1], this many points along the path

print("=" * 70)
print("Mode connectivity: grid-scheme vs. center-scheme L-BFGS optima")
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
def eval_U(x, y):
    Us = [float(utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon)[0].item()) for p in BATCHES]
    return float(np.mean(Us))


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


x_grid, y_grid, U_grid_saved = load_layout(
    _layouts.primary())
x_center, y_center, U_center_saved = load_layout(
    _layouts.secondary())

U_grid_re = eval_U(x_grid, y_grid)
U_center_re = eval_U(x_center, y_center)
print(f"grid-scheme   layout: U saved={U_grid_saved:.4f}    re-evaluated={U_grid_re:.4f}")
print(f"center-scheme layout: U saved={U_center_saved:.4f}  re-evaluated={U_center_re:.4f}")

raw_disp = ((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2).sqrt().mean()
print(f"mean per-detector displacement, index-to-index (before alignment): {raw_disp:.1f}m")

# ── Hungarian-align the center-scheme layout to the grid-scheme layout ─────
xy_stack = np.stack([
    np.stack([x_grid.numpy(), y_grid.numpy()], axis=-1),
    np.stack([x_center.numpy(), y_center.numpy()], axis=-1),
], axis=0)  # (2, n_det, 2)
aligned, perms = align_to_reference(xy_stack, ref_idx=0)
x_center_aligned = torch.as_tensor(aligned[1, :, 0], dtype=torch.float32)
y_center_aligned = torch.as_tensor(aligned[1, :, 1], dtype=torch.float32)

aligned_disp = ((x_grid - x_center_aligned) ** 2 + (y_grid - y_center_aligned) ** 2).sqrt().mean()
print(f"mean per-detector displacement, AFTER Hungarian alignment: {aligned_disp:.1f}m")

U_center_aligned_check = eval_U(x_center_aligned, y_center_aligned)
print(f"center-scheme U after alignment (sanity check, should match re-evaluated): "
      f"{U_center_aligned_check:.4f}")

# ── Linear interpolation path ────────────────────────────────────────────────
print(f"\n[path] sweeping {N_STEPS} points along the straight line from grid-scheme "
      f"to (aligned) center-scheme layout ...")
t0 = time.time()
ts = np.linspace(0.0, 1.0, N_STEPS)
U_path = np.zeros(N_STEPS, dtype=np.float64)
disp_path = np.zeros(N_STEPS, dtype=np.float64)
for k, t in enumerate(ts):
    N_t = (1 - t) * x_grid + t * x_center_aligned
    E_t = (1 - t) * y_grid + t * y_center_aligned
    N_proj, E_proj = project_to_mountain_ne(mountain, N_t, E_t)
    U_path[k] = eval_U(N_proj, E_proj)
    disp_path[k] = float(((N_proj - N_t) ** 2 + (E_proj - E_t) ** 2).sqrt().mean())
    if (k + 1) % 10 == 0:
        print(f"  {k+1}/{N_STEPS}  t={t:.3f}  U={U_path[k]:.3f}  ({time.time()-t0:.0f}s elapsed)")

dt = time.time() - t0
print(f"[path] done in {dt:.0f}s")

min_along_path = float(U_path.min())
min_t = float(ts[np.argmin(U_path)])
endpoints_mean = 0.5 * (U_path[0] + U_path[-1])
dip = endpoints_mean - min_along_path
print(f"\nEndpoints: U(t=0)={U_path[0]:.3f}  U(t=1)={U_path[-1]:.3f}")
print(f"Minimum along path: U={min_along_path:.3f} at t={min_t:.3f}")
print(f"Dip below the endpoints' average: {dip:.3f} ({100*dip/endpoints_mean:.2f}%)")
print(f"Mean mountain-projection correction along path: {disp_path.mean():.1f}m "
      f"(large values here would mean the straight line often leaves the mountain surface)")

results = dict(
    U_grid_saved=U_grid_saved, U_center_saved=U_center_saved,
    U_grid_reeval=U_grid_re, U_center_reeval=U_center_re,
    raw_displacement_m=float(raw_disp), aligned_displacement_m=float(aligned_disp),
    ts=ts.tolist(), U_path=U_path.tolist(), disp_path=disp_path.tolist(),
    min_along_path=min_along_path, min_t=min_t, dip_below_endpoints=dip,
)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mode_connectivity_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(ts, U_path, "-o", markersize=3)
axes[0].axhline(U_path[0], color="C0", linestyle="--", linewidth=0.8, label="grid-scheme U")
axes[0].axhline(U_path[-1], color="C1", linestyle="--", linewidth=0.8, label="center-scheme U")
axes[0].set_xlabel("t (0=grid-scheme, 1=center-scheme)")
axes[0].set_ylabel("U")
axes[0].set_title("U along the straight-line path between two independent optima")
axes[0].legend(fontsize=8)

axes[1].plot(ts, disp_path, "-o", markersize=3, color="gray")
axes[1].set_xlabel("t")
axes[1].set_ylabel("mean mountain-projection correction (m)")
axes[1].set_title("How far off-surface the straight line wanders")
fig.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mode_connectivity.png")
fig.savefig(out_png, dpi=150)
print(f"[plot] wrote {out_png}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if dip < 0.05 * endpoints_mean:
    print(f"  U barely dips ({100*dip/endpoints_mean:.2f}%) along the straight path -> "
          f"consistent with ONE connected plateau, not two separate basins.")
else:
    print(f"  U dips meaningfully ({100*dip/endpoints_mean:.2f}%) in the middle -> "
          f"these two optima are separated by a real barrier, not simply "
          f"two points on the same flat plateau.")
