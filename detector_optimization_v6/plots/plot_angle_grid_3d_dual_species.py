"""Grid of 3D showers across input angles — DUAL-species checkpoints only.

3D version of `plot_angle_grid_dual_species.py`: one shower per cell of a 5×5
grid of incident directions (**azimuth across columns, zenith across rows**) at a
FIXED energy/species, but each cell is a 3D point cloud — (x, y, layer/depth)
colored by per-point (cluster) energy or arrival time. This shows the depth
structure the flat 2D grid hides (e.g. the rod-like vs circularly-symmetric
question along the shower axis).

Generation backend is identical to the 2D grid / 3D single-shower scripts: the
per-species best checkpoints from `00_generate_data_dual_species.py` (stage a
Generator-loadable run-dir → `Generator` with the explicit per-species
`max_points` → `generate`). New-model only — there is no old-model path.

Two figures are written, both spanning every angle cell:
  <out>.png        — color = energy
  <out>_time.png   — color = arrival time

These models use flex_attention — run on GPU. PointCountFM (`compiled.pt`) is
forced to CPU (TorchScript device-baked); AllShowers runs on the chosen device.

Run:

    cd TambOpt/detector_optimization_v6
    python plots/plot_angle_grid_3d_dual_species.py --species electron
    python plots/plot_angle_grid_3d_dual_species.py --species muon --energy 1e6
"""
import argparse
import importlib.util
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
from modules.generate_showers import GenerateShowers  # noqa: F401 — injects TAMBO-opt path
from allshowers.generate_showers import (
    run_point_count_fm, build_direction_vector,
)
from allshowers.generator import generate

# Reuse the dual-species generation backend (config + staging + Generator) so this
# plot uses the exact same per-species models as 00_generate_data_dual_species.py.
# The module name starts with a digit, so load it by path.
_GEN_DUAL_PATH = os.path.join(_V6, "00_generate_data_dual_species.py")
_spec = importlib.util.spec_from_file_location("gen_dual_species", _GEN_DUAL_PATH)
gen_dual = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_dual)
SPECIES           = gen_dual.SPECIES
stage_run_dir     = gen_dual.stage_run_dir
Generator         = gen_dual.Generator
NUM_TIMESTEPS     = gen_dual.NUM_TIMESTEPS
SOLVER            = gen_dual.SOLVER
resample_overclip = gen_dual.resample_overclip   # anti-clip re-roll (shared policy)

# 00_generate_data.py uses GenerateShowers defaults for the sampling ranges:
from modules_v6.constants import (
LOG_E_MIN, LOG_E_MAX, ZENITH_MIN, ZENITH_MAX, AZIMUTH_MIN, AZIMUTH_MAX, 
)
E_MIN, E_MAX = 10**LOG_E_MIN, 10**LOG_E_MAX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", type=str, default="electron", choices=list(SPECIES.keys()),
                    help="which per-species checkpoint to use (from the dual-species config)")
    ap.add_argument("--rows", type=int, default=5, help="zenith steps (rows)")
    ap.add_argument("--cols", type=int, default=5, help="azimuth steps (cols)")
    ap.add_argument("--energy", type=float, default=1e7, help="fixed primary energy [GeV]")
    ap.add_argument("--label", type=int, default=0,
                    help="within-model class label fed to the generator (sampled 0/1 in "
                         "the generation pipeline; fixed here)")
    ap.add_argument("--zenith-min", type=float, default=ZENITH_MIN)
    ap.add_argument("--zenith-max", type=float, default=ZENITH_MAX)
    ap.add_argument("--azimuth-min", type=float, default=AZIMUTH_MIN)
    ap.add_argument("--azimuth-max", type=float, default=AZIMUTH_MAX)
    ap.add_argument("--device", type=str,
                    default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="default cuda. These models use flex_attention: on CPU it falls back "
                         "to a slow O(P^2) math path; on GPU a fused kernel is compiled.")
    ap.add_argument("--batch", type=int, default=12,
                    help="AllShowers generation batch size (lower if CUDA OOM)")
    ap.add_argument("--out", type=str, default=None,
                    help="output PNG (default: shower_angle_grid_3d_<species>.png)")
    args = ap.parse_args()
    out = args.out or os.path.join(_HERE, f"shower_angle_grid_3d_{args.species}.png")

    if not (E_MIN <= args.energy <= E_MAX):
        print(f"[warn] energy {args.energy:.2e} outside training range [{E_MIN:.0e},{E_MAX:.0e}]")

    cfg = SPECIES[args.species]
    pdg = int(cfg["pdg"])

    # Row-major grid: rows = zenith, cols = azimuth. Azimuth wraps, so endpoint=False.
    zeniths  = np.linspace(args.zenith_min, args.zenith_max, args.rows)
    azimuths = np.linspace(args.azimuth_min, args.azimuth_max, args.cols, endpoint=False)
    grid = [(z, a) for z in zeniths for a in azimuths]           # row-major
    ncell = len(grid)
    print(f"[grid-3d] {args.rows}×{args.cols} = {ncell} showers  "
          f"species={args.species}  E={args.energy:.2e} GeV  "
          f"pdg={pdg}  max_points={cfg['max_points']}")
    print(f"  zenith  rows: {np.round(zeniths,1).tolist()}")
    print(f"  azimuth cols: {np.round(azimuths,1).tolist()}")

    energies   = torch.full((ncell, 1), float(args.energy), dtype=torch.float32)
    directions = torch.tensor(
        np.stack([build_direction_vector(float(z), float(a)) for z, a in grid], axis=0),
        dtype=torch.float32)
    labels     = torch.full((ncell,), int(args.label), dtype=torch.int64)

    # Stage the per-species run-dir + build a Generator with the explicit point cap.
    staged_dir, pcfm = stage_run_dir(args.species, cfg)
    gen = Generator(run_dir=staged_dir, num_timesteps=NUM_TIMESTEPS,
                    compile=("cuda" in args.device.lower()), solver=SOLVER)
    gen.max_points = int(cfg["max_points"])

    # Stage 1 — PointCountFM on CPU (TorchScript device-baked → CUDA mismatches).
    num_points = run_point_count_fm(
        model_path=pcfm, energies=energies, directions=directions, labels=labels,
    )
    # Anti-clip re-roll over-cap cells (mainly muons at high E) so the plotted
    # showers use the same truncation policy as the generated corpus.
    num_points = resample_overclip(
        pcfm, energies, directions, labels, num_points, cap=int(cfg["max_points"]),
    )
    # Stage 2 — AllShowers on the chosen device.
    samples = generate(
        generator=gen, energies=energies, num_points=num_points,
        angles=directions, batch_size=int(args.batch), device=args.device, labels=labels,
    ).float().cpu().numpy()                                       # (ncell, P, 5)

    cells = []
    for k in range(ncell):
        pts = samples[k]
        pts = pts[pts[:, 3] > 0]                                  # drop padding (energy>0)
        cells.append(pts)

    base, ext = os.path.splitext(out)
    out_time = f"{base}_time{ext or '.png'}"
    _plot_grid_3d(cells, zeniths, azimuths, args.energy, pdg, args.species, out,
                  color_by="energy")
    print(f"[done] wrote {out}")
    _plot_grid_3d(cells, zeniths, azimuths, args.energy, pdg, args.species, out_time,
                  color_by="time")
    print(f"[done] wrote {out_time}")


def _plot_grid_3d(cells, zeniths, azimuths, energy, pdg, species, out, color_by="energy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    R, C = len(zeniths), len(azimuths)

    # Shared limits + shared color scale across all cells so cells are comparable.
    allpts = np.concatenate([p for p in cells if len(p)], axis=0) \
        if any(len(p) for p in cells) else np.zeros((1, 5))
    xs, ys, ls = allpts[:, 0], allpts[:, 1], allpts[:, 2]

    def _lim(a):
        lo, hi = float(a.min()), float(a.max())
        m = 0.05 * (hi - lo + 1e-6)
        return lo - m, hi + m
    xlim, ylim, llim = _lim(xs), _lim(ys), _lim(ls)

    if color_by == "time":
        tvals = allpts[:, 4]
        lo, hi = np.percentile(tvals, [2, 98]) if tvals.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1e-9
        norm = Normalize(vmin=float(lo), vmax=float(hi))
        cmap = "viridis"; col_idx = 4; col_label = "arrival time (cluster)"
    else:
        e_pos = allpts[:, 3][allpts[:, 3] > 0]
        vmin = float(e_pos.min()) if e_pos.size else 1e-3
        vmax = float(allpts[:, 3].max()) if allpts[:, 3].max() > 0 else 1.0
        norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin * 10))
        cmap = "inferno"; col_idx = 3; col_label = "cluster energy"

    fig = plt.figure(figsize=(3.4 * C, 3.4 * R))
    sc = None
    for i in range(R):
        for j in range(C):
            k = i * C + j
            ax = fig.add_subplot(R, C, k + 1, projection="3d")
            pts = cells[k]
            if len(pts):
                e = pts[:, 3]
                s = 2 + 14 * (e / (e.max() + 1e-12))
                sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, col_idx],
                                s=s, cmap=cmap, norm=norm, alpha=0.55, edgecolors="none")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*llim)
            ax.tick_params(labelsize=5)
            ax.set_zlabel("layer", fontsize=6)
            ax.view_init(elev=18, azim=-60)
            ax.text2D(0.02, 0.95, f"n={len(pts)}", transform=ax.transAxes,
                      fontsize=6, va="top", color="0.4")
            if i == 0:
                ax.set_title(f"φ = {azimuths[j]:.0f}°", fontsize=9)
            if j == 0:
                ax.text2D(-0.18, 0.5, f"θ = {zeniths[i]:.0f}°", transform=ax.transAxes,
                          fontsize=9, va="center", ha="right", rotation=90)

    if sc is not None:
        cb = fig.colorbar(sc, ax=fig.axes, shrink=0.5, pad=0.01)
        cb.set_label(col_label, fontsize=9)

    fig.suptitle(
        f"3D shower vs incident angle — azimuth φ across columns, zenith θ across rows\n"
        f"{species} model (pdg={pdg}), fixed E = {energy:.2e} GeV   "
        f"(x, y, layer/depth; color = {('time' if color_by=='time' else 'energy')}; "
        f"points = clustered cell centroids)",
        fontsize=12)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
