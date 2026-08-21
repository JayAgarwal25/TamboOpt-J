#!/usr/bin/env python3
"""Random 2D slice through the full layout space.

Perturbs every detector simultaneously along two random directions and maps
utility over the resulting plane, around both an optimized and a random layout
for contrast. This is the standard loss-landscape slice, without the filter
normalization step, which corrects a weight-space pathology that does not apply
to detector positions: they already share one homogeneous physical scale.

Points that fall off the mountain are projected back onto it, so a large mean
projection correction means the reading is distorted and the finer step range
should be trusted instead.
"""
import argparse
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
from modules_v6.detector_strategies_ne import layout_uniform_random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LAYOUT_PATH = _layouts.primary()

ap = argparse.ArgumentParser()
ap.add_argument("--layout_path", type=str, default=DEFAULT_LAYOUT_PATH,
                help="Path to a layout_best.pt to analyze (default: L-BFGS-best).")
ap.add_argument("--layout_tag", type=str, default=None,
                help="Label for this layout. If given, outputs land in "
                     "other_optimizers/<tag>/; if omitted, outputs use the original "
                     "flat filenames (backward-compatible with L-BFGS-only results).")
args = ap.parse_args()
LAYOUT_LABEL = args.layout_tag or "L-BFGS-best"
# Results live beside the other run outputs, not next to the code.
HERE = _layouts.results_dir()
OUT_DIR = os.path.join(HERE, "other_optimizers", args.layout_tag) if args.layout_tag else HERE
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SEED_BASE = 1000
BATCH_SIZE = 512
N_BATCHES = 4
GRID_N = 21
STEP_RANGE = 400.0   # alpha, beta range: [-STEP_RANGE, +STEP_RANGE] meters
N_DIR_PAIRS = 2
RANDOM_LAYOUT_SEED = 7

print("=" * 70)
print(f"Random full-space 2D slice (all 100 detectors perturbed at once) -- layout: {LAYOUT_LABEL}")
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


lbfgs_x, lbfgs_y, lbfgs_U_saved = load_layout(args.layout_path)
base_U_opt = eval_U(lbfgs_x, lbfgs_y)
print(f"[{LAYOUT_LABEL}] U (re-evaluated, {N_BATCHES} fresh batches): {base_U_opt:.4f}")

rng_np = np.random.default_rng(RANDOM_LAYOUT_SEED)
rand_x, rand_y = layout_uniform_random(mountain, rng=rng_np)
base_U_rand = eval_U(rand_x, rand_y)
print(f"Random layout U (re-evaluated, {N_BATCHES} fresh batches): {base_U_rand:.4f}")

alphas = np.linspace(-STEP_RANGE, STEP_RANGE, GRID_N).astype(np.float32)
betas = np.linspace(-STEP_RANGE, STEP_RANGE, GRID_N).astype(np.float32)

# ── Phase 1: compute all panels first (no plotting yet) ────────────────────
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

# ── Phase 2: plot all panels with a SHARED color/z scale across all 4, so
# panels are honestly comparable (a small-range panel shouldn't look just as
# "structured" as a large-range one because each got its own auto-scale). ──
u_global_min = min(r["U_min"] for r in results.values())
u_global_max = max(r["U_max"] for r in results.values())
print(f"\n[shared scale] U range across all {len(results)} panels: "
      f"[{u_global_min:.3f}, {u_global_max:.3f}]")

for tag, r in results.items():
    U_grid = np.array(r["U_grid"])
    base_U = r["base_U"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(betas, alphas, U_grid, shading="auto", cmap="viridis",
                        vmin=u_global_min, vmax=u_global_max)
    plt.colorbar(im, ax=ax, label="U")
    ax.scatter([0], [0], marker="*", s=250, c="red", edgecolor="black", label="base layout")
    ax.set_xlabel("beta (m, direction 2)")
    ax.set_ylabel("alpha (m, direction 1)")
    ax.set_title(f"Full-space random 2D slice ({LAYOUT_LABEL}): {tag}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f"full_space_2d_slice_{tag}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_png}")

    B_mesh, A_mesh = np.meshgrid(betas, alphas)
    fig3d = plt.figure(figsize=(7.5, 6.5))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.plot_surface(B_mesh, A_mesh, U_grid, cmap="viridis", edgecolor="none",
                       antialiased=True, alpha=0.95, vmin=u_global_min, vmax=u_global_max)
    ax3d.set_zlim(u_global_min, u_global_max)
    ax3d.scatter([0], [0], [base_U], marker="*", s=200, c="red", depthshade=False,
                 label="base layout")
    ax3d.set_xlabel("beta (m, direction 2)")
    ax3d.set_ylabel("alpha (m, direction 1)")
    ax3d.set_zlabel("U")
    ax3d.set_title(f"Full-space random 2D slice (3D, {LAYOUT_LABEL}): {tag}")
    ax3d.view_init(elev=25, azim=-60)
    fig3d.tight_layout()
    out_png_3d = os.path.join(OUT_DIR, f"full_space_2d_slice_{tag}_3d.png")
    fig3d.savefig(out_png_3d, dpi=150)
    plt.close(fig3d)
    print(f"[plot] wrote {out_png_3d}")

out_json = os.path.join(OUT_DIR, "full_space_2d_slice_results.json")
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
