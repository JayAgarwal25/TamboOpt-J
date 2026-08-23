#!/usr/bin/env python3
"""
Random 2D slice through the full 200-dim layout space: all 100 detectors moved
at once along two random directions, around both an optimized layout and a
random one for contrast. Complements the single-detector grid scans, which vary
only the 2 dims of one detector.

Li et al. 2018 ("Visualizing the Loss Landscape of Neural Nets") without the
filter-normalization step, which corrects a ReLU scale invariance that detector
positions do not have: these are metres, already on one homogeneous scale. Each
direction is drawn N(0,1) per detector and NOT globally L2-normalized, so
alpha/beta read directly as typical per-detector displacement in metres.

The default 400m sweep pushes boundary detectors past the mesh snap tolerance
(~160m), so part of what it measures is snapping rather than landscape: the snap
correction ran 39-70m mean, 169m max. Re-run with --step-range 100 --out-prefix
full_space_2d_slice_fine for a cleaner reading, matching the surrogate's own
kernel scale. Everything else is identical, so the two are comparable.

--layout_path and --layout_tag select which saved layout to analyse; a tag sends
output to other_optimizers/<tag>/. The random-layout panel is always drawn.
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
from common import Scorer, N_DETECTORS, TRAINING_DATASET_FOLDER

from modules_v6.tr_geometry_ne import project_to_mountain_ne
from modules_v6.detector_strategies_ne import layout_uniform_random

DEFAULT_LAYOUT_PATH = _layouts.primary()

ap = argparse.ArgumentParser()
ap.add_argument("--layout_path", type=str, default=DEFAULT_LAYOUT_PATH,
                help="Path to a layout_best.pt to analyze (default: L-BFGS-best).")
ap.add_argument("--step-range", dest="step_range", type=float, default=400.0,
                help="alpha/beta sweep half-range in metres. 400 is the original "
                     "run. Use 100 to stay below the mesh snap tolerance (~160m), "
                     "which the 400m run exceeded on boundary-adjacent detectors "
                     "(mean snap correction 39-70m, max 169m). Everything else is "
                     "identical, so the two are directly comparable.")
ap.add_argument("--out-prefix", dest="out_prefix", type=str,
                default="full_space_2d_slice",
                help="Filename stem for the results. Give the 100m run its own "
                     "stem so it does not overwrite the 400m one.")
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
STEP_RANGE = args.step_range   # sweep is [-STEP_RANGE, +STEP_RANGE] metres
N_DIR_PAIRS = 2
RANDOM_LAYOUT_SEED = 7

print("=" * 70)
print(f"Random full-space 2D slice (all 100 detectors perturbed at once) -- layout: {LAYOUT_LABEL}")
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
        de1 = torch.randn(N_DETECTORS)
        dn1 = torch.randn(N_DETECTORS)
        de2 = torch.randn(N_DETECTORS)
        dn2 = torch.randn(N_DETECTORS)

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
                e_new = base_x + alpha * de1 + beta * de2
                n_new = base_y + alpha * dn1 + beta * dn2
                e_proj, n_proj = project_to_mountain_ne(mountain, e_new, n_new)
                disp_grid[i, j] = float(((e_proj - e_new) ** 2 + (n_proj - n_new) ** 2).sqrt().mean())
                U_grid[i, j] = eval_U(e_proj, n_proj)
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
    out_png = os.path.join(OUT_DIR, f"{args.out_prefix}_{tag}.png")
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
    out_png_3d = os.path.join(OUT_DIR, f"{args.out_prefix}_{tag}_3d.png")
    fig3d.savefig(out_png_3d, dpi=150)
    plt.close(fig3d)
    print(f"[plot] wrote {out_png_3d}")

out_json = os.path.join(OUT_DIR, f"{args.out_prefix}_results.json")
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
