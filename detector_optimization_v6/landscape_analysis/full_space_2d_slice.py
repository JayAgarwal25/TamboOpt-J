#!/usr/bin/env python3
"""
Random 2D slice through the FULL 200-dim layout space (all 100 detectors
perturbed simultaneously along 2 random directions), around both the
L-BFGS-best layout and a random layout for contrast.

This is the "Visualizing the Loss Landscape of Neural Nets" (Li et al. 2018)
technique, simplified: their "filter normalization" step exists to correct
for a neural-network-weight-specific pathology (ReLU scale invariance across
filters/layers), which doesn't apply here -- our parameters are just
detector (North, East) positions, already on one natural, homogeneous scale
(meters). So plain random Gaussian directions are used directly, no
normalization trick needed.

Direction convention: each direction is (dN, dE), dN/dE ~ N(0,1) per
detector (NOT globally L2-normalized across all 200 dims), so the step size
alpha/beta directly corresponds to "typical per-detector displacement in
meters" along that direction, matching how sigma is used in the basin-
hopping/infill scripts elsewhere in this project.

Complements the single-detector grid scans (which vary only 2 of 200 dims,
tied to ONE detector): this varies all 200 dims at once, testing a more
global notion of flatness.
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

from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models
from modules_v6.tr_geometry_ne import project_to_mountain_ne
from modules_v6.detector_strategies_ne import layout_uniform_random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
BATCH_SEED_BASE = 1000
BATCH_SIZE = 512
N_BATCHES = 4
GRID_N = 21
STEP_RANGE = 400.0   # alpha, beta range: [-STEP_RANGE, +STEP_RANGE] meters
N_DIR_PAIRS = 2
RANDOM_LAYOUT_SEED = 7

print("=" * 70)
print("Random full-space 2D slice (all 100 detectors perturbed at once)")
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
def eval_U(x, y):
    Us = [float(utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon)[0].item()) for p in BATCHES]
    return float(np.mean(Us))


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


lbfgs_x, lbfgs_y, lbfgs_U_saved = load_layout(
    f"{RUN_BASE}/test_v6_run_04_optimize_lbfgs_ensemble_ds_combined/layout_best.pt")
base_U_opt = eval_U(lbfgs_x, lbfgs_y)
print(f"L-BFGS best U (re-evaluated, {N_BATCHES} fresh batches): {base_U_opt:.4f}")

rng_np = np.random.default_rng(RANDOM_LAYOUT_SEED)
rand_x, rand_y = layout_uniform_random(mountain, rng=rng_np)
base_U_rand = eval_U(rand_x, rand_y)
print(f"Random layout U (re-evaluated, {N_BATCHES} fresh batches): {base_U_rand:.4f}")

alphas = np.linspace(-STEP_RANGE, STEP_RANGE, GRID_N).astype(np.float32)
betas = np.linspace(-STEP_RANGE, STEP_RANGE, GRID_N).astype(np.float32)

results = {}
torch.manual_seed(321)
for layout_tag, (base_x, base_y, base_U) in (
    ("optimized", (lbfgs_x, lbfgs_y, base_U_opt)),
    ("random", (rand_x, rand_y, base_U_rand)),
):
    for pair_idx in range(N_DIR_PAIRS):
        dN1 = torch.randn(N_DETECTORS)
        dE1 = torch.randn(N_DETECTORS)
        dN2 = torch.randn(N_DETECTORS)
        dE2 = torch.randn(N_DETECTORS)

        tag = f"{layout_tag}_pair{pair_idx}"
        print(f"\n[{tag}] sweeping {GRID_N}x{GRID_N} grid, step range +/-{STEP_RANGE:.0f}m ...")
        t0 = time.time()
        U_grid = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        # Mean per-detector mountain-projection correction at each grid point --
        # diagnostic for how much of any "flatness" is genuine landscape structure
        # vs. an artifact of project_to_mountain_ne's discrete nearest-centroid snap
        # (see mode_connectivity.py's disp_path for the same check on a 1D path).
        disp_grid = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                N_new = base_x + alpha * dN1 + beta * dN2
                E_new = base_y + alpha * dE1 + beta * dE2
                N_proj, E_proj = project_to_mountain_ne(mountain, N_new, E_new)
                disp_grid[i, j] = float(((N_proj - N_new) ** 2 + (E_proj - E_new) ** 2).sqrt().mean())
                U_grid[i, j] = eval_U(N_proj, E_proj)
            if (i + 1) % 5 == 0:
                print(f"  row {i+1}/{GRID_N}  ({time.time()-t0:.0f}s elapsed)")
        dt = time.time() - t0
        span = float(U_grid.max() - U_grid.min())
        print(f"[{tag}] done in {dt:.0f}s. U range=[{U_grid.min():.3f}, {U_grid.max():.3f}]  "
              f"span={span:.3f}  base U={base_U:.3f}  argmax gain={U_grid.max()-base_U:+.3f}")
        print(f"[{tag}] mountain-projection correction: mean={disp_grid.mean():.1f}m  "
              f"max={disp_grid.max():.1f}m  (large values mean many grid points needed "
              f"off-surface snapping, which can distort the flatness reading)")

        results[tag] = dict(
            layout=layout_tag, base_U=base_U, U_min=float(U_grid.min()),
            U_max=float(U_grid.max()), span=span, argmax_gain=float(U_grid.max() - base_U),
            alphas=alphas.tolist(), betas=betas.tolist(), U_grid=U_grid.tolist(), time_s=dt,
            disp_grid_mean_m=float(disp_grid.mean()), disp_grid_max_m=float(disp_grid.max()),
            disp_grid=disp_grid.tolist(),
        )

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.pcolormesh(betas, alphas, U_grid, shading="auto", cmap="viridis")
        plt.colorbar(im, ax=ax, label="U")
        ax.scatter([0], [0], marker="*", s=250, c="red", edgecolor="black", label="base layout")
        ax.set_xlabel("beta (m, direction 2)")
        ax.set_ylabel("alpha (m, direction 1)")
        ax.set_title(f"Full-space random 2D slice: {tag}")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"full_space_2d_slice_{tag}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_png}")

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_space_2d_slice_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_json}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for tag, r in results.items():
    print(f"  {tag:20s}: base U={r['base_U']:.3f}  span={r['span']:.3f}  "
          f"({100*r['span']/r['base_U']:.2f}% of base U)  gain={r['argmax_gain']:+.3f}  "
          f"snap disp mean/max={r['disp_grid_mean_m']:.1f}/{r['disp_grid_max_m']:.1f}m")
