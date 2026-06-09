"""Plot the first N showers from a cached shower checkpoint (no generation).

Sibling of `plot_shower_realizations.py` — same overlay/profile figure, but the
showers come from an existing showerdata cache (e.g. v6_run_00/cashed_showers_*.pt)
instead of being generated. Light: just showerdata + matplotlib, no model, no GPU.

Run:

    cd TambOpt/detector_optimization_v6
    python plots/plot_cached_showers.py --ckpt <path> --n 5
    python plots/plot_cached_showers.py --ckpt <path> --n 5 --mountain

Note: showerdata.load reads the whole file — fine for normal caches, but don't
point it at a 100s-of-GB corpus.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = os.path.dirname(_HERE)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import showerdata

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v6.constants import (
    SHOWER_CACHE,
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
from modules_v4.tr_geometry import load_tr_mountain

_DEFAULT_CKPT = os.path.join(SHOWER_CACHE, f"cashed_showers_dual_2.pt")

# constants.GEOMETRY_PATH may be stale; prefer a local copy, then the new TAMBOSim path.
GEOMETRY_PATH_RESOLVED = next(
    (p for p in (
        os.path.join(_V6, "colca_valley.h5"),
        "/n/home05/zdimitrov/tambo/TAMBOSim/resources/geometry/colca_valley.h5",
        GEOMETRY_PATH,
    ) if os.path.exists(p)),
    GEOMETRY_PATH,
)


def _load_mountain():
    return load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )


def _recenter_to_mountain(reals, mountain):
    """Shift each shower so its energy-weighted (x,y) centroid lands on the
    mountain bbox centre — identical to build_training_pairs(recenter_to_mountain=True)
    / compute_aleatoric_floor.py. Returns new list of shifted (P,5) arrays."""
    cx_t = 0.5 * (mountain.n_min + mountain.n_max)
    cy_t = 0.5 * (mountain.u_min + mountain.u_max)
    out = []
    for pts in reals:
        if not len(pts):
            out.append(pts); continue
        w = pts[:, 3]
        cx = (pts[:, 0] * w).sum() / max(w.sum(), 1e-9)
        cy = (pts[:, 1] * w).sum() / max(w.sum(), 1e-9)
        q = pts.copy()
        q[:, 0] += (cx_t - cx)
        q[:, 1] += (cy_t - cy)
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=_DEFAULT_CKPT, help="cached shower file")
    ap.add_argument("--n", type=int, default=5, help="number of leading showers to plot")
    ap.add_argument("--bins", type=int, default=80, help="heatmap bins per axis")
    ap.add_argument("--mountain", action="store_true",
                    help="apply pipeline mountain normalization (recenter each shower's "
                         "energy-weighted centroid onto the mountain bbox centre) and "
                         "overlay the mountain footprint")
    ap.add_argument("--out", type=str, default=os.path.join(_HERE, "cached_showers.png"))
    args = ap.parse_args()

    print(f"[load] {args.ckpt}")
    data = showerdata.load(args.ckpt)
    points = np.asarray(data.points)                         # (N, P, 5): x,y,layer,e,t
    pdg = np.asarray(data.pdg).reshape(-1)
    n = min(args.n, len(points))
    plabel = f"first {n} showers — {os.path.basename(args.ckpt)}"
    print(f"[load] file has {len(points)} showers; plotting first {n}")

    # Split each shower into its real (non-padded) points.
    reals = []
    for k in range(n):
        pts = points[k]
        m = pts[:, 3] > 0                                    # energy>0 = real point
        reals.append(pts[m])
        print(f"  shower {k}: pdg={int(pdg[k])}  n_points={int(m.sum())}  E_tot={pts[m,3].sum():.3g}")

    mountain = None
    if args.mountain:
        print(f"[mountain] {GEOMETRY_PATH_RESOLVED}  — recentering to bbox centre")
        mountain = _load_mountain()
        reals = _recenter_to_mountain(reals, mountain)

    _plot(reals, pdg[:n], plabel, args.out, mountain=mountain, bins=args.bins)
    print(f"[done] wrote {args.out}")


def _plot(reals, pdg, plabel, out, mountain=None, bins=80):
    """One energy heatmap per shower in a grid: 2D histogram of (x, y) weighted
    by per-point energy (log colour scale). Mountain mode shares a geographic
    extent and underlays the footprint."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    N = len(reals)
    C = int(math.ceil(math.sqrt(N)))
    R = int(math.ceil(N / C))
    fig, axes = plt.subplots(R, C, figsize=(4.2 * C, 3.8 * R), squeeze=False)

    cen = getattr(mountain, "centroids_NUE", None) if mountain is not None else None
    # In mountain mode all cells share one geographic extent (footprint + points).
    shared_extent = None
    if mountain is not None and cen is not None:
        allx = np.concatenate([r[:, 0] for r in reals if len(r)] + [cen[:, 0]])
        ally = np.concatenate([r[:, 1] for r in reals if len(r)] + [cen[:, 1]])
        shared_extent = [allx.min(), allx.max(), ally.min(), ally.max()]

    for k in range(R * C):
        ax = axes[k // C][k % C]
        if k >= N:
            ax.axis("off"); continue
        pts = reals[k]

        if shared_extent is not None:
            ex = shared_extent
        elif len(pts):
            mx = 0.05 * (np.ptp(pts[:, 0]) + 1e-6); my = 0.05 * (np.ptp(pts[:, 1]) + 1e-6)
            ex = [pts[:, 0].min() - mx, pts[:, 0].max() + mx,
                  pts[:, 1].min() - my, pts[:, 1].max() + my]
        else:
            ex = [0, 1, 0, 1]

        rng = [[ex[0], ex[1]], [ex[2], ex[3]]]
        if len(pts):
            H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins,
                                     range=rng, weights=pts[:, 3])
        else:
            H = np.zeros((bins, bins))
        Hm = np.ma.masked_where(H <= 0, H)                   # empty bins → transparent
        cmap = plt.cm.magma.copy(); cmap.set_bad(alpha=0.0)
        norm = (LogNorm(vmin=float(Hm.min()), vmax=float(Hm.max()))
                if Hm.count() and Hm.max() > Hm.min() else None)

        if cen is not None:
            ax.scatter(cen[:, 0], cen[:, 1], s=1, c="0.85", alpha=0.5,
                       edgecolors="none", zorder=0)
        im = ax.imshow(Hm.T, origin="lower", extent=ex,
                       aspect=("equal" if mountain is not None else "auto"),
                       cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"#{k}  pdg={int(pdg[k])}  n={len(pts)}", fontsize=8)
        ax.tick_params(labelsize=6)

    unit = "North/Up [m]" if mountain is not None else "x / y [m]"
    norm_s = " (mountain-normalized)" if mountain is not None else ""
    fig.suptitle(f"Per-shower energy heatmaps{norm_s} — {plabel}   "
                 f"(colour = deposited energy, log; {unit})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
