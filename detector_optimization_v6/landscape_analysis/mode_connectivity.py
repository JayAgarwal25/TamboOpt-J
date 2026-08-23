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

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (layout paths live in one place)
from common import Scorer, TRAINING_DATASET_FOLDER, align_to_reference

from modules_v6.tr_geometry_ne import project_to_mountain_ne

BATCH_SEED_BASE = 1000
BATCH_SIZE = 512
N_BATCHES = 6            # cheap experiment (few points), afford more batches for precision
N_STEPS = 41             # t in [0, 1], this many points along the path

print("=" * 70)
print("Mode connectivity: grid-scheme vs. center-scheme L-BFGS optima")
print("=" * 70)

sc = Scorer(n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed_base=BATCH_SEED_BASE)
fnn, recon = sc.fnn, sc.recon
mountain = sc.mountain

primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                         weights_only=False).float()


fresh_batch = sc.draw

BATCHES = sc.batches


eval_U = sc.U

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
    e_t = (1 - t) * x_grid + t * x_center_aligned
    n_t = (1 - t) * y_grid + t * y_center_aligned
    e_proj, n_proj = project_to_mountain_ne(mountain, e_t, n_t)
    U_path[k] = eval_U(e_proj, n_proj)
    disp_path[k] = float(((e_proj - e_t) ** 2 + (n_proj - n_t) ** 2).sqrt().mean())
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
out_path = os.path.join(_layouts.results_dir(), "mode_connectivity_results.json")
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
out_png = os.path.join(_layouts.results_dir(), "mode_connectivity.png")
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
