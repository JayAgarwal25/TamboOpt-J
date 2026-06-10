"""Compare ML-generated vs simulated electron showers — Hamza's matched-pair test.

Hamza's two files hold the SAME primaries in the same order:
  simulated : .../h5_files_v3/combined_electrons_test.h5   (ground-truth sim)
  ml        : .../h5_files_v3/ml_electron_test.h5          (ML-generated)
Each `showers[i]` is a (4096, 5) point cloud [x, y, layer, energy, time] (the
clustered representation — cells of 10 m near core / 20 m far, nmax=4096).

The question (Hamza): "if both match, the ML part is fine and the circular-vs-rod
look is a pre-processing/clustering effect, not the model." This script plots
matched sim/ML pairs side by side (2D footprint + optional 3D) for a spread of
showers, and prints per-shower agreement metrics (total energy ratio, footprint
extent) so "match" is quantitative, not just visual.

Run:

    cd TambOpt/detector_optimization_v6
    python plots/plot_hamza_ml_vs_sim.py                 # 6 showers spanning zenith, 2D
    python plots/plot_hamza_ml_vs_sim.py --mode 3d
    python plots/plot_hamza_ml_vs_sim.py --indices 0 10 100 --mode both
"""
import argparse
import os

import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))

_DIR = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3"
SIM_FILE = os.path.join(_DIR, "combined_electrons_test.h5")
ML_FILE  = os.path.join(_DIR, "ml_electron_test.h5")


def _load_shower(f, i):
    """Return (P, 5) float cloud for shower i, padding rows (energy<=0) dropped."""
    shp = f["shape"][:]                      # (N, P, C)
    arr = np.asarray(f["showers"][i]).reshape(int(shp[1]), int(shp[2]))
    return arr[arr[:, 3] > 0]


def _zenith_deg(d):
    """Zenith from a unit direction vector (z = cos of the polar angle)."""
    return np.degrees(np.arccos(np.clip(np.abs(d[2]), -1.0, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="number of showers (spread across zenith)")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="explicit shower indices (overrides --n spread)")
    ap.add_argument("--mode", choices=["2d", "3d", "both"], default="2d")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    with h5py.File(SIM_FILE, "r") as fs, h5py.File(ML_FILE, "r") as fm:
        N = int(fs["shape"][:][0])
        dirs = fs["directions"][:]
        energ = fs["energies"][:, 0]

        if args.indices:
            idxs = [i for i in args.indices if 0 <= i < N]
        else:
            # Spread across zenith so rod-elongation (if any) is visible.
            zen = np.array([_zenith_deg(dirs[i]) for i in range(N)])
            order = np.argsort(zen)
            idxs = [int(order[k]) for k in np.linspace(0, N - 1, args.n).astype(int)]

        print(f"[hamza] comparing {len(idxs)} matched sim/ML showers")
        print(f"  sim: {SIM_FILE}")
        print(f"  ml : {ML_FILE}")

        rows = []
        for i in idxs:
            s = _load_shower(fs, i)
            m = _load_shower(fm, i)
            es, em = s[:, 3].sum(), m[:, 3].sum()
            ratio = em / max(es, 1e-12)
            z = _zenith_deg(dirs[i])
            print(f"  #{i:5d}  E={energ[i]:.2e}  θ={z:5.1f}°  "
                  f"n_sim={len(s):5d} n_ml={len(m):5d}  "
                  f"ΣE_sim={es:.3e} ΣE_ml={em:.3e}  ratio={ratio:.3f}")
            rows.append((i, z, energ[i], s, m, ratio))

    if args.mode in ("2d", "both"):
        out = args.out or os.path.join(_HERE, "hamza_ml_vs_sim_2d.png")
        _plot_2d(rows, out)
        print(f"[done] wrote {out}")
    if args.mode in ("3d", "both"):
        out3 = (args.out or os.path.join(_HERE, "hamza_ml_vs_sim")) \
            .replace(".png", "") + "_3d.png"
        _plot_3d(rows, out3)
        print(f"[done] wrote {out3}")


def _shared_lims(s, m):
    allp = np.concatenate([s, m], axis=0) if len(s) or len(m) else np.zeros((1, 5))
    def lim(a):
        lo, hi = float(a.min()), float(a.max())
        pad = 0.05 * (hi - lo + 1e-6)
        return lo - pad, hi + pad
    return lim(allp[:, 0]), lim(allp[:, 1]), lim(allp[:, 2])


def _plot_2d(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    R = len(rows)
    fig, axes = plt.subplots(R, 2, figsize=(9, 4.2 * R), squeeze=False)
    for r, (i, z, E, s, m, ratio) in enumerate(rows):
        xlim, ylim, _ = _shared_lims(s, m)
        allp = np.concatenate([s, m], axis=0)
        epos = allp[:, 3][allp[:, 3] > 0]
        norm = LogNorm(vmin=max(epos.min(), 1e-3), vmax=epos.max()) if epos.size else None
        for c, (pts, title) in enumerate([(s, "simulated"), (m, "ML")]):
            ax = axes[r][c]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 3], s=4, cmap="inferno",
                           norm=norm, alpha=0.6, edgecolors="none")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=6)
            ax.set_title(f"{title}  #{i}  θ={z:.0f}°  n={len(pts)}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"E={E:.1e} GeV\nΣE_ml/ΣE_sim={ratio:.2f}", fontsize=8)
    fig.suptitle("Hamza test — matched electron showers: simulated (L) vs ML (R)\n"
                 "footprint x–y, color = energy  (clustered cell centroids)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_3d(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    R = len(rows)
    fig = plt.figure(figsize=(10, 4.6 * R))
    for r, (i, z, E, s, m, ratio) in enumerate(rows):
        xlim, ylim, llim = _shared_lims(s, m)
        allp = np.concatenate([s, m], axis=0)
        epos = allp[:, 3][allp[:, 3] > 0]
        norm = LogNorm(vmin=max(epos.min(), 1e-3), vmax=epos.max()) if epos.size else None
        for c, (pts, title) in enumerate([(s, "simulated"), (m, "ML")]):
            ax = fig.add_subplot(R, 2, 2 * r + c + 1, projection="3d")
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 3], s=4,
                           cmap="inferno", norm=norm, alpha=0.5, edgecolors="none")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*llim)
            ax.set_zlabel("layer", fontsize=6); ax.tick_params(labelsize=5)
            ax.view_init(elev=18, azim=-60)
            ax.set_title(f"{title}  #{i}  θ={z:.0f}°  n={len(pts)}", fontsize=9)
    fig.suptitle("Hamza test (3D) — matched electron showers: simulated (L) vs ML (R)\n"
                 "x, y, layer/depth, color = energy", fontsize=12)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
