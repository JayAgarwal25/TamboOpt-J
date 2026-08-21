#!/usr/bin/env python3
"""
Control for the detector grid scan: is the flat-plateau finding (moving one
detector anywhere on the mountain barely changes U) specific to being near
the L-BFGS-optimized layout, or is it a general property of the surrogate
regardless of layout quality?

Same sweep (grid resolution, batch averaging) as detector_grid_scan_broad.py,
but starting from a RANDOM (uniform, non-optimized) layout instead of
L-BFGS-best. If the random layout shows a MUCH LARGER U range when sweeping
one detector, that shows sensitivity to individual detector placement is
concentrated near the optimum (each detector still "matters" until the
layout is good, then stops mattering). If the random layout is ALSO flat,
that would suggest the surrogate itself is just insensitive to single-
detector position everywhere, a very different (more surrogate-limitation-
flavored) conclusion.
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import modules_v6  # noqa: F401

from modules_v6 import run_world
from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX,
    N_PLANES
)

# Run world resolved from argv/$TAMBO_RUN_WORLD at startup rather than
# inherited from constants.py at import. See modules_v6/run_world.py.
_W = run_world.resolve()
TRAINING_DATASET_FOLDER = _W.dataset_folder
FNN_FOLDER = _W.fnn_folder
RECON_FOLDER = _W.recon_folder

from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models
from modules_v6.tr_geometry_ne import project_to_mountain_ne
from modules_v6.detector_strategies_ne import layout_uniform_random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SEED_BASE = 1000
BATCH_SIZE = 512
N_BATCHES = 5
GRID_N = 15
N_DET_SWEEP = 3          # random layout has no meaningful "center/edge" structure a priori;
                          # still sample a few detectors at different distances from bbox center
RANDOM_LAYOUT_SEED = 7

print("=" * 70)
print("Detector grid scan CONTROL: random (non-optimized) layout")
print("=" * 70)

fnn, recon = load_models(DEVICE, fnn_folder=FNN_FOLDER, recon_dir=RECON_FOLDER + "_deepsets")
mountain = load_tr_mountain(GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
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


rng = np.random.default_rng(RANDOM_LAYOUT_SEED)
rand_x, rand_y = layout_uniform_random(mountain, rng=rng)
base_U = eval_U_mean(rand_x, rand_y)
print(f"Random layout (seed={RANDOM_LAYOUT_SEED}) U (re-evaluated, {N_BATCHES} fresh batches): {base_U:.4f}")

cn = 0.5 * (mountain.n_min + mountain.n_max)
ce = 0.5 * (mountain.east_lo + mountain.east_hi)
dist = ((rand_x - cn) ** 2 + (rand_y - ce) ** 2).sqrt()
order = torch.argsort(dist)
pick_positions = np.linspace(0, N_DETECTORS - 1, N_DET_SWEEP).round().astype(int)
sweep_indices = [int(order[p]) for p in pick_positions]
print(f"Sweeping detectors at percentile positions {pick_positions.tolist()}:")
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
    orig_N, orig_E = float(rand_x[idx]), float(rand_y[idx])
    for k in range(n_pts):
        x_mod = rand_x.clone()
        y_mod = rand_y.clone()
        x_mod[idx] = grid_N_proj[k]
        y_mod[idx] = grid_E_proj[k]
        U_grid[k] = eval_U_mean(x_mod, y_mod)
    dt = time.time() - t0
    argmax_k = int(U_grid.argmax())
    argmax_N, argmax_E = float(grid_N_proj[argmax_k]), float(grid_E_proj[argmax_k])
    disp = ((argmax_N - orig_N) ** 2 + (argmax_E - orig_E) ** 2) ** 0.5
    rng_span = float(U_grid.max() - U_grid.min())
    print(f"[idx={idx}] done in {dt:.0f}s.  U range=[{U_grid.min():.3f}, {U_grid.max():.3f}]  "
          f"span={rng_span:.3f}  base U={base_U:.3f}  argmax gain={U_grid.max()-base_U:+.3f}  "
          f"argmax displacement={disp:.1f}m")
    U_grid_2d = U_grid.reshape(GRID_N, GRID_N)
    results[str(idx)] = dict(
        idx=idx, dist_to_center=float(dist[idx]), orig_N=orig_N, orig_E=orig_E,
        base_U=base_U, U_min=float(U_grid.min()), U_max=float(U_grid.max()),
        span=rng_span, argmax_gain=float(U_grid.max() - base_U), argmax_displacement=disp,
        argmax_N=argmax_N, argmax_E=argmax_E,
        n_grid=n_grid.tolist(), e_grid=e_grid.tolist(), U_grid=U_grid_2d.tolist(),
        time_s=dt,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(e_grid, n_grid, U_grid_2d, shading="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="U (this detector swept, other 99 fixed, RANDOM layout)")
    other_mask = np.ones(N_DETECTORS, dtype=bool)
    other_mask[idx] = False
    ax.scatter(rand_y[other_mask], rand_x[other_mask], s=8, c="white",
               edgecolor="black", linewidth=0.3, label="other 99 detectors (fixed, random)")
    ax.scatter([orig_E], [orig_N], marker="*", s=250, c="red", edgecolor="black", label="original random position")
    ax.scatter([argmax_E], [argmax_N], marker="X", s=150, c="cyan", edgecolor="black", label="grid argmax")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"CONTROL (random layout): U vs. position of detector {idx}")
    ax.legend(loc="upper right", fontsize=8)
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"detector_grid_random_control_{idx}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_png}")

    E_mesh, N_mesh = np.meshgrid(e_grid, n_grid)
    fig3d = plt.figure(figsize=(8, 6.5))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.plot_surface(E_mesh, N_mesh, U_grid_2d, cmap="viridis", edgecolor="none",
                       antialiased=True, alpha=0.95)
    ax3d.scatter([orig_E], [orig_N], [base_U], marker="*", s=200, c="red", depthshade=False,
                 label="original random position")
    ax3d.scatter([argmax_E], [argmax_N], [float(U_grid.max())], marker="X", s=100, c="cyan",
                 depthshade=False, label="grid argmax")
    ax3d.set_xlabel("East (m)")
    ax3d.set_ylabel("North (m)")
    ax3d.set_zlabel("U")
    ax3d.set_title(f"CONTROL (random layout): U vs. position of detector {idx}, 3D")
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.view_init(elev=25, azim=-60)
    fig3d.tight_layout()
    out_png_3d = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"detector_grid_random_control_{idx}_3d.png")
    fig3d.savefig(out_png_3d, dpi=150)
    plt.close(fig3d)
    print(f"[plot] wrote {out_png_3d}")

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_grid_scan_random_control_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_json}")

print("\n" + "=" * 70)
print("SUMMARY (compare against detector_grid_scan_broad_results.json)")
print("=" * 70)
spans = [results[k]["span"] for k in results]
for idx in sweep_indices:
    r = results[str(idx)]
    print(f"  idx={idx:3d}  dist={r['dist_to_center']:7.1f}m  span={r['span']:.3f}  "
          f"gain={r['argmax_gain']:+.3f}  disp={r['argmax_displacement']:7.1f}m")
print(f"\nMean span (random layout): {np.mean(spans):.3f}  base U={base_U:.3f}")
