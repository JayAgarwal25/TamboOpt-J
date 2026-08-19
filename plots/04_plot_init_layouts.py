"""Show the stage-4 initial layouts, before and after the chain perturbation.

Stage 4 does NOT start Adam from `sample_initial_layout_ne`'s output: each of
the K chains is perturbed by `INIT_OVERDISP_SIGMA` metres per detector per
coordinate (`_build_chain_inits`) and then projected back onto the mountain.
When that sigma is large compared to the mesh (~1.3 km here) it erases the init
scheme's structure, so every scheme degenerates to the same random start. This
script plots both stages side by side so the actual Adam start is visible.

Top row    : base layout from sample_initial_layout_ne (unperturbed).
Bottom row : base + N(0, sigma) then projected — what Adam actually starts from.

`--sigma` overrides INIT_OVERDISP_SIGMA so a candidate value can be previewed
without editing (and re-running) the optimizer.

Run from the v6 folder:

    cd TambOpt
    python plots/04_plot_init_layouts.py
    python plots/04_plot_init_layouts.py --sigma 20        # preview a change
    python plots/04_plot_init_layouts.py -o /path/out.png
"""
import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_DIR = os.path.dirname(_HERE)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v6.tr_geometry_ne import sample_initial_layout_ne, project_to_mountain_ne
from modules_v6.legacy_core.tr_geometry import load_tr_mountain
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, N_DETECTORS, OPT_FOLDER,
)


def _optimizer_config():
    """Read INIT_SCHEMES / INIT_OVERDISP_SIGMA from the optimizer itself, so this
    plot cannot drift from the values the run actually uses. Loaded by path
    because the module name starts with a digit."""
    path = os.path.join(_V6_DIR, "scripts", "04_optimize_lbfgs_ensemble.py")
    spec = importlib.util.spec_from_file_location("_opt04", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tuple(mod.INIT_SCHEMES), float(mod.INIT_OVERDISP_SIGMA)


def main():
    schemes_default, sigma_default = _optimizer_config()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=sigma_default,
                    help=f"per-detector init perturbation [m] (default: "
                         f"INIT_OVERDISP_SIGMA={sigma_default:g} from the optimizer)")
    ap.add_argument("--schemes", nargs="+", default=list(schemes_default),
                    help=f"init schemes to draw (default: {list(schemes_default)})")
    ap.add_argument("--seed", type=int, default=0, help="perturbation seed")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="output png (default: <OPT_FOLDER dir>/init_layouts.png)")
    args = ap.parse_args()

    out = args.output or os.path.join(os.path.dirname(OPT_FOLDER), "init_layouts.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"sigma   : {args.sigma:g} m")
    print(f"schemes : {args.schemes}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    cen = np.asarray(mountain.centroids_ENU)          # (n_tri, 3) East, North, Up
    tri = mtri.Triangulation(cen[:, 0], cen[:, 1])

    n_col = len(args.schemes)
    fig, axes = plt.subplots(2, n_col, figsize=(5.5 * n_col, 9.5), dpi=110,
                             squeeze=False)
    fig.suptitle(f"Stage-4 initial layouts  (INIT_OVERDISP_SIGMA = {args.sigma:g} m)",
                 fontsize=13)

    g = torch.Generator().manual_seed(args.seed)
    for col, scheme in enumerate(args.schemes):
        e_np, n_np = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
        e_t = torch.as_tensor(np.asarray(e_np), dtype=torch.float32)
        n_t = torch.as_tensor(np.asarray(n_np), dtype=torch.float32)

        # Mirrors _build_chain_inits for a single chain: base + N(0, sigma) on
        # every coordinate, then the same mountain projection Adam applies.
        base = torch.cat([e_t, n_t])
        pert = base + torch.randn(base.numel(), generator=g) * args.sigma
        pe, pn = project_to_mountain_ne(mountain, pert[:N_DETECTORS], pert[N_DETECTORS:])

        rows = [(np.asarray(e_np), np.asarray(n_np), "base (sample_initial_layout_ne)"),
                (pe.numpy(), pn.numpy(),
                 f"+N(0,{args.sigma:g}m) then projected  <- Adam starts here")]
        for row, (E, N, tag) in enumerate(rows):
            ax = axes[row][col]
            ax.tricontourf(tri, cen[:, 2], levels=24, cmap="Greys", alpha=0.55)
            ax.scatter(E, N, s=30, c="#2a78d6", edgecolors="white",
                       linewidths=0.4, zorder=3)
            # n_unique exposes projection stacking: a large sigma throws
            # detectors off-mesh, where the projection collapses many onto the
            # same edge triangle.
            n_uniq = len(np.unique(np.stack([E, N], axis=-1).round(1), axis=0))
            ax.set_title(f"{scheme} — {tag}\n"
                         f"E std={E.std():.1f} m,  N std={N.std():.1f} m,  "
                         f"{n_uniq}/{len(E)} distinct", fontsize=9)
            ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
            ax.set_xlim(cen[:, 0].min() - 60, cen[:, 0].max() + 60)
            ax.set_ylim(cen[:, 1].min() - 60, cen[:, 1].max() + 60)
            ax.set_aspect("equal")
            print(f"{scheme:8s} {tag:54s} E std={E.std():7.1f}  N std={N.std():7.1f}  "
                  f"distinct={n_uniq}/{len(E)}")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
