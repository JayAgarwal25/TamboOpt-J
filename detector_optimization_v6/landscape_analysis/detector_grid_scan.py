#!/usr/bin/env python3
"""
Single-detector utility grid scan 

Take a near-optimal layout (L-BFGS-best, the one we trust over GES per
standing guidance -- the Hessian probe showed GES-best sits on a numerical
singularity, not a genuine optimum), move ONE detector's position over a grid
spanning the mountain, hold the other 99 fixed, and re-evaluate U at each
grid point with the frozen FNN+recon surrogate. No optimization -- just
repeated forward evaluation. Unlike the Hessian probe (which only sees the
infinitesimal neighborhood of the optimum), this sees the full, possibly
non-quadratic, non-local structure of U as a function of one detector's
position.

Two detectors are swept: the one closest to the mountain bbox center, and the
one farthest from it (closest to the boundary), to compare interior vs.
edge sensitivity.

Reusable across any saved layout via --layout_path/--layout_tag (defaults to
the L-BFGS-best layout, matching the original single-layout investigation).
Pass --layout_tag to run this on a different optimizer's layout (EvoGrad, DE,
GES, v5's (mu+lambda)-ES, etc.) -- outputs then land in
other_optimizers/<tag>/ instead of this directory directly, keeping the
per-optimizer extension study cleanly separated from the original L-BFGS
results.
"""
import argparse
import sys, os, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (layout paths live in one place)
import modules_v6  # injects v3/v4 paths

from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models
from modules_v6.tr_geometry_ne import project_to_mountain_ne

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
DEFAULT_LAYOUT_PATH = _layouts.primary()

ap = argparse.ArgumentParser()
ap.add_argument("--layout_path", type=str, default=DEFAULT_LAYOUT_PATH,
                help="Path to a layout_best.pt to analyze (default: L-BFGS-best).")
ap.add_argument("--layout_tag", type=str, default=None,
                help="Label for this layout (e.g. 'evograd', 'de', 'ges', 'mu_lambda_es'). "
                     "If given, outputs land in other_optimizers/<tag>/ instead of this "
                     "directory directly; if omitted, outputs use the original flat "
                     "filenames (backward-compatible with the L-BFGS-only results).")
args = ap.parse_args()
LAYOUT_LABEL = args.layout_tag or "L-BFGS-best"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "other_optimizers", args.layout_tag) if args.layout_tag else HERE
os.makedirs(OUT_DIR, exist_ok=True)

# NOTE: seed=42 is the exact batch every optimizer script (04_optimize_*.py)
# trains/scores against, so layouts are specifically overfit to it -- it reads
# ~10-15pts high vs genuinely fresh batches (see landscape_analysis.py Q3:
# fresh-batch std across 5 seeds is only ~0.66, but the seed=42 batch is a
# clear outlier relative to that spread). Batch seeds here deliberately avoid
# 42 so this scan measures the same thing Q3 already established is a stable,
# non-overfit estimate.
BATCH_SEED_BASE = 1000
BATCH_SIZE = 512
N_BATCHES = 8     # average over several fresh batches to damp batch noise
GRID_N = 25       # GRID_N x GRID_N points per swept detector

print("=" * 70)
print(f"Single-detector utility grid scan -- layout: {LAYOUT_LABEL}")
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
    Us = []
    for p in BATCHES:
        U, _, _ = utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon)
        Us.append(float(U.item()))
    return float(np.mean(Us))


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


lbfgs_x, lbfgs_y, lbfgs_U_saved = load_layout(args.layout_path)
print(f"[{LAYOUT_LABEL}] U (saved): {lbfgs_U_saved:.4f}")
base_U = eval_U_mean(lbfgs_x, lbfgs_y)
print(f"[{LAYOUT_LABEL}] U (re-evaluated, {N_BATCHES} fresh batches): {base_U:.4f}")

# Pick two detectors: closest to bbox center, and farthest from it (edge).
cn = 0.5 * (mountain.n_min + mountain.n_max)
ce = 0.5 * (mountain.east_lo + mountain.east_hi)
d2 = (lbfgs_x - cn) ** 2 + (lbfgs_y - ce) ** 2
idx_center = int(d2.argmin())
idx_edge = int(d2.argmax())
print(f"idx_center={idx_center}  pos=({lbfgs_x[idx_center]:.1f},{lbfgs_y[idx_center]:.1f})  "
      f"dist_to_bbox_center={d2[idx_center]**0.5:.1f}m")
print(f"idx_edge  ={idx_edge}    pos=({lbfgs_x[idx_edge]:.1f},{lbfgs_y[idx_edge]:.1f})  "
      f"dist_to_bbox_center={d2[idx_edge]**0.5:.1f}m")

# Sweep grid over the full mountain bbox, projected onto the surface once (vectorized).
n_grid = np.linspace(mountain.n_min, mountain.n_max, GRID_N).astype(np.float32)
e_grid = np.linspace(mountain.east_lo, mountain.east_hi, GRID_N).astype(np.float32)
NN, EE = np.meshgrid(n_grid, e_grid, indexing="ij")
grid_N_flat = torch.as_tensor(NN.reshape(-1), dtype=torch.float32)
grid_E_flat = torch.as_tensor(EE.reshape(-1), dtype=torch.float32)
grid_N_proj, grid_E_proj = project_to_mountain_ne(mountain, grid_N_flat, grid_E_flat)

results = {}
for tag, idx in [("center", idx_center), ("edge", idx_edge)]:
    print(f"\n[{tag}] sweeping detector idx={idx} over {GRID_N}x{GRID_N} grid ...")
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
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{n_pts}  ({time.time()-t0:.0f}s elapsed)")
    U_grid_2d = U_grid.reshape(GRID_N, GRID_N)
    argmax_k = int(U_grid.argmax())
    argmax_N, argmax_E = float(grid_N_proj[argmax_k]), float(grid_E_proj[argmax_k])
    dt = time.time() - t0
    print(f"[{tag}] done in {dt:.0f}s. U range [{U_grid.min():.3f}, {U_grid.max():.3f}]  "
          f"orig U={base_U:.3f} at ({orig_N:.1f},{orig_E:.1f})  "
          f"argmax U={U_grid.max():.3f} at ({argmax_N:.1f},{argmax_E:.1f})")

    results[tag] = dict(
        idx=idx, orig_N=orig_N, orig_E=orig_E, base_U=base_U,
        argmax_U=float(U_grid.max()), argmax_N=argmax_N, argmax_E=argmax_E,
        n_grid=n_grid.tolist(), e_grid=e_grid.tolist(),
        U_grid=U_grid_2d.tolist(), U_min=float(U_grid.min()), U_max=float(U_grid.max()),
        time_s=dt,
    )

# Plot both panels with a SHARED color/z scale (see full_space_2d_slice.py for
# rationale) -- otherwise each panel auto-scales to its own tiny range and a
# 0.3-unit wobble can look just as "structured" as a 2-unit one.
u_global_min = min(r["U_min"] for r in results.values())
u_global_max = max(r["U_max"] for r in results.values())
print(f"\n[shared scale] U range across both panels: [{u_global_min:.3f}, {u_global_max:.3f}]")

for tag, r in results.items():
    idx, orig_N, orig_E, base_U = r["idx"], r["orig_N"], r["orig_E"], r["base_U"]
    argmax_N, argmax_E, argmax_U = r["argmax_N"], r["argmax_E"], r["argmax_U"]
    U_grid_2d = np.array(r["U_grid"])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(e_grid, n_grid, U_grid_2d, shading="auto", cmap="viridis",
                        vmin=u_global_min, vmax=u_global_max)
    plt.colorbar(im, ax=ax, label="U (this detector swept, other 99 fixed)")
    other_mask = np.ones(N_DETECTORS, dtype=bool)
    other_mask[idx] = False
    ax.scatter(lbfgs_y[other_mask], lbfgs_x[other_mask], s=8, c="white",
               edgecolor="black", linewidth=0.3, label="other 99 detectors (fixed)")
    ax.scatter([orig_E], [orig_N], marker="*", s=250, c="red",
               edgecolor="black", label=f"optimized position ({LAYOUT_LABEL})")
    ax.scatter([argmax_E], [argmax_N], marker="X", s=150, c="cyan",
               edgecolor="black", label="grid argmax")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"U vs. position of detector {idx} ({tag}), others fixed at {LAYOUT_LABEL}")
    ax.legend(loc="upper right", fontsize=8)
    out_png = os.path.join(OUT_DIR, f"detector_grid_{tag}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    E_mesh, N_mesh = np.meshgrid(e_grid, n_grid)
    fig3d = plt.figure(figsize=(8, 6.5))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.plot_surface(E_mesh, N_mesh, U_grid_2d, cmap="viridis", edgecolor="none",
                       antialiased=True, alpha=0.95, vmin=u_global_min, vmax=u_global_max)
    ax3d.set_zlim(u_global_min, u_global_max)
    ax3d.scatter([orig_E], [orig_N], [base_U], marker="*", s=200, c="red", depthshade=False,
                 label="optimized position")
    ax3d.scatter([argmax_E], [argmax_N], [argmax_U], marker="X", s=100, c="cyan",
                 depthshade=False, label="grid argmax")
    ax3d.set_xlabel("East (m)")
    ax3d.set_ylabel("North (m)")
    ax3d.set_zlabel("U")
    ax3d.set_title(f"U vs. position of detector {idx} ({tag}), 3D")
    ax3d.legend(loc="upper left", fontsize=8)
    ax3d.view_init(elev=25, azim=-60)
    fig3d.tight_layout()
    out_png_3d = os.path.join(OUT_DIR, f"detector_grid_{tag}_3d.png")
    fig3d.savefig(out_png_3d, dpi=150)
    plt.close(fig3d)
    print(f"[plot] wrote {out_png_3d}")
    print(f"[plot] wrote {out_png}")

out_json = os.path.join(OUT_DIR, "detector_grid_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_json}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for tag in ["center", "edge"]:
    r = results[tag]
    disp = ((r["argmax_N"] - r["orig_N"]) ** 2 + (r["argmax_E"] - r["orig_E"]) ** 2) ** 0.5
    gain = r["argmax_U"] - r["base_U"]
    print(f"  {tag:8s} (idx={r['idx']:3d}): base U={r['base_U']:.3f}  "
          f"grid-argmax U={r['argmax_U']:.3f}  gain={gain:+.3f}  "
          f"displacement from optimized position={disp:.1f}m")
