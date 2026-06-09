"""Grid of showers across input angles (one shower per cell).

Companion to `plot_shower_realizations.py`. Instead of N realizations of ONE
primary, this generates a single shower for each of a 5×5 grid of incident
directions — **azimuth varies across columns, zenith across rows** — at a FIXED
energy and species, so you can see how the shower morphology changes with angle.

Angle/energy ranges match `00_generate_data.py` (the `GenerateShowers` defaults):
zenith ∈ [60°, 100°], azimuth ∈ [0°, 360°), energy ∈ [1e5, 1e8] GeV (here held
fixed at one value). Same generator as the floor/realization scripts: the small
home05 AllShowers + PointCountFM checkpoints (CPU, no holylfs05, no compile stall).

Run:

    cd TambOpt/detector_optimization_v6
    python plots/plot_angle_grid.py                       # 5×5, E=1e7 GeV
    python plots/plot_angle_grid.py --energy 1e6 --mountain
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = os.path.dirname(_HERE)                       # detector_optimization_v6/
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4 (and TAMBO-opt)
from modules.generate_showers import GenerateShowers  # noqa: F401  (path injection)
from allshowers.generate_showers import (
    run_point_count_fm, run_allshowers, build_direction_vector,
    _DEFAULT_POINT_COUNT_MODEL, _DEFAULT_ALLSHOWERS_RUN_DIR,
)
from modules_v6.constants import (
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
from modules_v4.tr_geometry import load_tr_mountain

# 00_generate_data.py uses GenerateShowers defaults for the sampling ranges:
E_MIN, E_MAX = 1e5, 1e8           # GeV
ZENITH_MIN, ZENITH_MAX = 60.0, 100.0
AZIMUTH_MIN, AZIMUTH_MAX = 0.0, 360.0

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


def _recenter_to_mountain(pts, mountain):
    """Translate one shower so its energy-weighted (x,y) centroid lands on the
    mountain bbox centre (pipeline mountain normalization — translation only)."""
    if not len(pts):
        return pts
    cx_t = 0.5 * (mountain.n_min + mountain.n_max)
    cy_t = 0.5 * (mountain.u_min + mountain.u_max)
    w = pts[:, 3]
    cx = (pts[:, 0] * w).sum() / max(w.sum(), 1e-9)
    cy = (pts[:, 1] * w).sum() / max(w.sum(), 1e-9)
    q = pts.copy()
    q[:, 0] += (cx_t - cx)
    q[:, 1] += (cy_t - cy)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5, help="zenith steps (rows)")
    ap.add_argument("--cols", type=int, default=5, help="azimuth steps (cols)")
    ap.add_argument("--energy", type=float, default=1e7, help="fixed primary energy [GeV]")
    ap.add_argument("--pdg", type=int, default=0, help="fixed species label")
    ap.add_argument("--zenith-min", type=float, default=ZENITH_MIN)
    ap.add_argument("--zenith-max", type=float, default=ZENITH_MAX)
    ap.add_argument("--azimuth-min", type=float, default=AZIMUTH_MIN)
    ap.add_argument("--azimuth-max", type=float, default=AZIMUTH_MAX)
    ap.add_argument("--device", type=str,
                    default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="default cuda. This model uses flex_attention: on CPU it falls back "
                         "to a slow O(P^2) math path (materializes the full scores matrix); on "
                         "GPU run_allshowers compiles a fused kernel. Use cpu only if no GPU.")
    ap.add_argument("--mountain", action="store_true",
                    help="mountain-normalize each cell (recenter to bbox centre) + overlay footprint")
    ap.add_argument("--out", type=str, default=os.path.join(_HERE, "shower_angle_grid.png"))
    args = ap.parse_args()

    if not (E_MIN <= args.energy <= E_MAX):
        print(f"[warn] energy {args.energy:.2e} outside training range [{E_MIN:.0e},{E_MAX:.0e}]")

    # Row-major grid: rows = zenith, cols = azimuth. Azimuth wraps, so endpoint=False.
    zeniths  = np.linspace(args.zenith_min, args.zenith_max, args.rows)
    azimuths = np.linspace(args.azimuth_min, args.azimuth_max, args.cols, endpoint=False)
    grid = [(z, a) for z in zeniths for a in azimuths]           # row-major
    ncell = len(grid)
    print(f"[grid] {args.rows}×{args.cols} = {ncell} showers  "
          f"E={args.energy:.2e} GeV  pdg={args.pdg}")
    print(f"  zenith  rows: {np.round(zeniths,1).tolist()}")
    print(f"  azimuth cols: {np.round(azimuths,1).tolist()}")

    energies   = torch.full((ncell, 1), float(args.energy), dtype=torch.float32)
    directions = torch.tensor(
        np.stack([build_direction_vector(float(z), float(a)) for z, a in grid], axis=0),
        dtype=torch.float32)
    labels     = torch.full((ncell,), int(args.pdg), dtype=torch.int64)

    num_points = run_point_count_fm(
        model_path=_DEFAULT_POINT_COUNT_MODEL,
        energies=energies, directions=directions, labels=labels,
    )
    samples = run_allshowers(
        run_dir=_DEFAULT_ALLSHOWERS_RUN_DIR,
        energies=energies, directions=directions, labels=labels,
        num_points=num_points, num_timesteps=16, batch_size=ncell,
        solver="midpoint", device=args.device,
    ).float().cpu().numpy()                                       # (ncell, P, 5)

    cells = []
    for k in range(ncell):
        pts = samples[k]
        pts = pts[pts[:, 3] > 0]                                  # drop padding (energy>0)
        cells.append(pts)

    mountain = None
    if args.mountain:
        print(f"[mountain] {GEOMETRY_PATH_RESOLVED}  — recentering each cell")
        mountain = _load_mountain()
        cells = [_recenter_to_mountain(p, mountain) for p in cells]

    _plot_grid(cells, zeniths, azimuths, args.energy, args.pdg, args.out, mountain)
    print(f"[done] wrote {args.out}")


def _plot_grid(cells, zeniths, azimuths, energy, pdg, out, mountain):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R, C = len(zeniths), len(azimuths)

    # Shared limits across all cells so morphology is comparable cell-to-cell.
    allpts = np.concatenate([p for p in cells if len(p)], axis=0) \
        if any(len(p) for p in cells) else np.zeros((1, 5))
    xs, ys = allpts[:, 0], allpts[:, 1]
    if mountain is not None and getattr(mountain, "centroids_NUE", None) is not None:
        cen = mountain.centroids_NUE
        xs = np.concatenate([xs, cen[:, 0]]); ys = np.concatenate([ys, cen[:, 1]])
    mx = 0.05 * (xs.max() - xs.min() + 1e-6)
    my = 0.05 * (ys.max() - ys.min() + 1e-6)
    xlim = (xs.min() - mx, xs.max() + mx)
    ylim = (ys.min() - my, ys.max() + my)

    fig, axes = plt.subplots(R, C, figsize=(3.0 * C, 3.0 * R),
                             sharex=True, sharey=True, squeeze=False)
    for i in range(R):
        for j in range(C):
            ax = axes[i][j]
            pts = cells[i * C + j]
            if mountain is not None and getattr(mountain, "centroids_NUE", None) is not None:
                ax.scatter(mountain.centroids_NUE[:, 0], mountain.centroids_NUE[:, 1],
                           s=1, c="lightgray", alpha=0.5, edgecolors="none", zorder=0)
            if len(pts):
                e = pts[:, 3]
                s = 2 + 18 * (e / (e.max() + 1e-12))
                ax.scatter(pts[:, 0], pts[:, 1], s=s, c="C0", alpha=0.4,
                           edgecolors="none", zorder=1)
            # Azimuth arrow: incident travel direction projected onto x-y, ∝ (cosφ, sinφ).
            az = np.radians(azimuths[j])
            ux, uy = np.cos(az), np.sin(az)
            if len(pts):
                w = pts[:, 3]
                bx = (pts[:, 0] * w).sum() / max(w.sum(), 1e-9)
                by = (pts[:, 1] * w).sum() / max(w.sum(), 1e-9)
            else:
                bx, by = 0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1])
            L = 0.30 * min(xlim[1] - xlim[0], ylim[1] - ylim[0])
            # Arrow HEAD lands on the shower start (centroid); tail sits one
            # arrow-length back along the incident direction.
            ax.annotate("", xy=(bx, by), xytext=(bx - L * ux, by - L * uy),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.6, alpha=0.9),
                        zorder=3)

            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=6)
            ax.text(0.03, 0.97, f"n={len(pts)}", transform=ax.transAxes,
                    fontsize=6, va="top", ha="left", color="0.4")
            if i == 0:
                ax.set_title(f"φ = {azimuths[j]:.0f}°", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"θ = {zeniths[i]:.0f}°", fontsize=10)

    unit = "North/Up [m]" if mountain is not None else "x / y [m]"
    norm = " (mountain-normalized)" if mountain is not None else ""
    fig.suptitle(
        f"Shower vs incident angle — azimuth φ across columns, zenith θ across rows{norm}\n"
        f"fixed E = {energy:.2e} GeV, species pdg={pdg}   ({unit})   "
        f"[red arrow = azimuth travel direction (cosφ, sinφ)]",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
