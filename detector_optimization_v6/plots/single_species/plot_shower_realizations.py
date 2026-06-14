"""Visualize shower-to-shower (aleatoric) variability at a FIXED primary.

Generates N independent showers from the *same* primary (energy, zenith, azimuth)
and overlays their secondary point clouds so the irreducible realization noise —
the thing the aleatoric floor (`compute_aleatoric_floor.py`, THEORY.md §10.4)
measures — is directly visible.

Uses the same generator as the floor computation: the small home05 AllShowers +
PointCountFM checkpoints (fast, no holylfs05, no torch.compile stall on CPU).

Run:

    cd TambOpt/detector_optimization_v6
    python plot_shower_realizations.py            # 5 showers, seed-picked primary
    python plot_shower_realizations.py --n 5 --energy 1e7 --zenith 80 --azimuth 180
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4 (and TAMBO-opt)
from modules.generate_showers import GenerateShowers  # noqa: F401  (path injection)
from allshowers.generate_showers import (
    sample_primary_particles, run_point_count_fm, run_allshowers,
    build_direction_vector, _DEFAULT_POINT_COUNT_MODEL, _DEFAULT_ALLSHOWERS_RUN_DIR,
)
from modules_v6.constants import (
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
from modules_v4.tr_geometry import load_tr_mountain

# constants.GEOMETRY_PATH may be stale; prefer a local copy, then the new TAMBOSim path.
GEOMETRY_PATH_RESOLVED = next(
    (p for p in (
        os.path.join(_HERE, "colca_valley.h5"),
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
    """Shift each realization so its energy-weighted (x,y) centroid lands on the
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


def _primary(args):
    """Return (energies (N,1), directions (N,3), labels (N,)) — the SAME primary
    repeated N times. Explicit --energy uses (energy, zenith, azimuth); otherwise
    one primary is sampled with --seed."""
    n = args.n
    if args.energy is not None:
        e = torch.tensor([[float(args.energy)]], dtype=torch.float32)
        d = torch.tensor(build_direction_vector(args.zenith, args.azimuth),
                         dtype=torch.float32).reshape(1, 3)
        lab = torch.tensor([0], dtype=torch.int64)
    else:
        prim = sample_primary_particles(n=1, seed=args.seed)
        e, d, lab = prim["energies"], prim["directions"], prim["labels"]
    return (torch.repeat_interleave(e, n, dim=0),
            torch.repeat_interleave(d, n, dim=0),
            torch.repeat_interleave(lab, n, dim=0))


def _primary_label(energies, directions):
    E = float(energies[0, 0])
    dx, dy, dz = (float(directions[0, i]) for i in range(3))
    theta = math.degrees(math.acos(max(-1.0, min(1.0, dz))))
    phi = math.degrees(math.atan2(dy, dx)) % 360.0
    return f"E={E:.3e} GeV,  θ={theta:.1f}°,  φ={phi:.1f}°"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of realizations")
    ap.add_argument("--seed", type=int, default=0, help="picks the primary (if --energy unset)")
    ap.add_argument("--energy", type=float, default=None, help="primary energy [GeV] (explicit)")
    ap.add_argument("--zenith", type=float, default=80.0, help="zenith [deg] (with --energy)")
    ap.add_argument("--azimuth", type=float, default=180.0, help="azimuth [deg] (with --energy)")
    ap.add_argument("--device", type=str,
                    default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="default cuda. This model uses flex_attention: on CPU it falls back "
                         "to a slow O(P^2) math path; on GPU run_allshowers compiles a fused "
                         "kernel. Use cpu only if no GPU.")
    ap.add_argument("--mountain", action="store_true",
                    help="apply pipeline mountain normalization (recenter each shower's "
                         "energy-weighted centroid onto the mountain bbox centre) and "
                         "overlay the mountain footprint")
    ap.add_argument("--out", type=str, default=os.path.join(_HERE, "shower_realizations.png"))
    args = ap.parse_args()

    energies, directions, labels = _primary(args)
    plabel = _primary_label(energies, directions)
    print(f"[primary] {plabel}   (N={args.n} realizations)")

    # Stage 1 — PointCountFM (CPU). Stage 2 — AllShowers.
    num_points = run_point_count_fm(
        model_path=_DEFAULT_POINT_COUNT_MODEL,
        energies=energies, directions=directions, labels=labels,
    )
    samples = run_allshowers(
        run_dir=_DEFAULT_ALLSHOWERS_RUN_DIR,
        energies=energies, directions=directions, labels=labels,
        num_points=num_points, num_timesteps=16, batch_size=args.n,
        solver="midpoint", device=args.device,
    ).float().cpu().numpy()                                  # (N, P, 5): x,y,layer,e,t

    # Split each realization into its real (non-padded) points.
    reals = []
    for k in range(args.n):
        pts = samples[k]
        m = pts[:, 3] > 0                                    # energy>0 = real point
        reals.append(pts[m])
        print(f"  shower {k}: n_points={int(m.sum())}  E_tot={pts[m,3].sum():.3g}")

    mountain = None
    if args.mountain:
        print(f"[mountain] {GEOMETRY_PATH_RESOLVED}  — recentering to bbox centre")
        mountain = _load_mountain()
        reals = _recenter_to_mountain(reals, mountain)

    _plot(reals, plabel, args.out, mountain=mountain)
    print(f"[done] wrote {args.out}")


def _plot(reals, plabel, out, mountain=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(reals)
    colors = plt.cm.tab10(np.linspace(0, 1, max(N, 1)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Common core = energy-weighted centroid over ALL realizations (for the radial panel).
    allpts = np.concatenate(reals, axis=0) if any(len(r) for r in reals) else np.zeros((1, 5))
    w = allpts[:, 3]
    cx = float((allpts[:, 0] * w).sum() / max(w.sum(), 1e-9))
    cy = float((allpts[:, 1] * w).sum() / max(w.sum(), 1e-9))

    # Panel 1 — overlaid (x, y) scatter of secondaries, one colour per realization.
    if mountain is not None:                                  # mountain footprint behind
        cen = getattr(mountain, "centroids_NUE", None)
        if cen is not None:
            axes[0].scatter(cen[:, 0], cen[:, 1], s=3, c="lightgray", alpha=0.5,
                            edgecolors="none", label="mountain", zorder=0)
    for k, pts in enumerate(reals):
        if not len(pts):
            continue
        e = pts[:, 3]
        s = 4 + 30 * (e / (e.max() + 1e-12))                # size ∝ energy
        axes[0].scatter(pts[:, 0], pts[:, 1], s=s, color=colors[k], alpha=0.35,
                        edgecolors="none", label=f"shower {k}  (n={len(pts)})")
    axes[0].scatter([cx], [cy], marker="x", c="k", s=80, label="energy centroid")
    xlab = "North [m]" if mountain is not None else "x [m]"
    ylab = "Up [m]"    if mountain is not None else "y [m]"
    axes[0].set_xlabel(xlab); axes[0].set_ylabel(ylab)
    axes[0].set_aspect("auto" if mountain is not None else "auto")
    _suffix = " (mountain-normalized)" if mountain is not None else ""
    axes[0].set_title(f"Same primary, {N} realizations — secondary (x, y){_suffix}\n{plabel}", fontsize=10)
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8, loc="best")

    # Panel 2 — lateral energy profile (energy vs radius from the common core).
    rmax = 0.0
    for pts in reals:
        if len(pts):
            r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            rmax = max(rmax, float(r.max()))
    bins = np.linspace(0, rmax if rmax > 0 else 1.0, 40)
    for k, pts in enumerate(reals):
        if not len(pts):
            continue
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        h, edges = np.histogram(r, bins=bins, weights=pts[:, 3])
        ctr = 0.5 * (edges[1:] + edges[:-1])
        axes[1].step(ctr, h, where="mid", color=colors[k], alpha=0.85,
                     label=f"shower {k}")
    axes[1].set_xlabel("radius from energy centroid [m]")
    axes[1].set_ylabel("deposited energy per bin")
    axes[1].set_title("Lateral energy profile per realization", fontsize=10)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.suptitle("Shower-to-shower variability at a fixed primary "
                 "(the irreducible aleatoric noise the floor measures)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
