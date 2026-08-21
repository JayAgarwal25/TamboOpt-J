"""How much of an optimized layout's utility depends on detectors being stacked?

`04_optimize_lbfgs_ensemble.py` reports the minimum pairwise separation of the
layout it saves, and the phase3 run came back with 0.541 m (grid, 2 pairs under
1 m) and 0.108 m (center, 12 pairs). Detectors 11 cm apart are not two detectors.
The kernel evaluates each one independently against the same point cloud, so
co-locating them counts the same particles twice and raises U without collecting
anything more.

That makes the reported U ambiguous: some unknown share of it is physics and some
is the double count. This measures the split by enforcing a floor on the pairwise
separation and re-scoring.

    for each d_min: push every pair closer than d_min apart, re-snap to the
    mountain, iterate to a fixed point, then score with the UNMODIFIED
    `eval_true_utility.score` so the numbers are directly comparable to that
    script's, on both the surrogate and the kernel.

HOW TO READ IT. The relaxation confounds two things, and the confound only ever
makes the artifact look BIGGER than it is: separating detectors removes the
double count, but it also moves them off the optimum the sweep found. So

    U survives a 10-20 m floor nearly intact -> stacking was not load-bearing,
                                                the gain is mostly real.
    U collapses toward baseline               -> upper bound on the artifact;
                                                cannot separate the two causes,
                                                but the reported U is not safe.

Baselines (grid, center, unrelaxed) are scored in the same run so the sweep is
read against them rather than in absolute terms.

    python eval/diag_layout_collapse.py --layout <layout_best.pt> \
        --fnn_folder <dir> --recon_dir <dir> --n-events 2048
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# eval_true_utility stays in plots/ (it is shared with upstream), so reach it
# through the repo root rather than through this script's own directory.
_PLOTS = os.path.join(_HERE, "plots")
for _p in (_HERE, _PLOTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import modules_v6  # noqa: F401
from modules_v6.opt_core import load_models
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    FNN_FOLDER, RECON_FOLDER,
)

import eval_true_utility as _etu


def pair_stats(e, n):
    """(min separation, count of pairs under 1 m, count under 10 m)."""
    xy = torch.stack([e, n], dim=-1).double()
    d = torch.cdist(xy, xy)
    d.fill_diagonal_(float("inf"))
    return float(d.min()), int((d < 1.0).sum().item() // 2), int((d < 10.0).sum().item() // 2)


def relax(e, n, mountain, d_min, iters=400, seed=0):
    """Push pairs apart until every separation is >= d_min, re-snapping each step.

    Snapping back onto the mountain footprint can re-close a pair, so this must
    iterate rather than push once. Exactly-coincident points get a tiny random
    kick, otherwise their connecting vector is zero and they never separate.
    """
    if d_min <= 0:
        return e.clone(), n.clone()
    g = torch.Generator().manual_seed(seed)
    e, n = e.clone().double(), n.clone().double()
    for _ in range(iters):
        xy = torch.stack([e, n], dim=-1)
        d = torch.cdist(xy, xy)
        d.fill_diagonal_(float("inf"))
        if float(d.min()) >= d_min:
            break
        i, j = torch.nonzero(d < d_min, as_tuple=True)
        keep = i < j
        i, j = i[keep], j[keep]
        if i.numel() == 0:
            break
        vec = xy[i] - xy[j]
        dist = vec.norm(dim=-1, keepdim=True)
        coincident = (dist < 1e-9).squeeze(-1)
        if coincident.any():
            vec[coincident] = torch.randn(int(coincident.sum()), 2, generator=g).double()
            dist[coincident] = vec[coincident].norm(dim=-1, keepdim=True)
        push = 0.5 * (d_min - dist.squeeze(-1)).clamp_min(0.0)
        step = (vec / dist.clamp_min(1e-9)) * push.unsqueeze(-1)
        upd = torch.zeros_like(xy)
        upd.index_add_(0, i, step)
        upd.index_add_(0, j, -step)
        xy = xy + upd
        e, n = _etu._snap(mountain, xy[:, 0].float(), xy[:, 1].float())
        e, n = e.double(), n.double()
    return e.float(), n.float()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", type=str, required=True,
                    help="layout_best.pt from the stage-4 run to diagnose")
    ap.add_argument("--fnn_folder", type=str, default=None,
                    help="directory with fnn_electron.pt/fnn_muon.pt; without it the "
                         "surrogate silently comes from constants and may not be the "
                         "one this layout was optimized against")
    ap.add_argument("--recon_dir", type=str, default=None)
    ap.add_argument("--corpus", type=str, default=None)
    ap.add_argument("--n-events", type=int, default=2048)
    ap.add_argument("--kernel-chunk", type=int, default=128)
    ap.add_argument("--d-min", type=str, default="0,1,2,5,10,20,50",
                    help="comma-separated separation floors in metres")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--plot_dir", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("layout collapse diagnostic — how much U survives a separation floor?")
    print("=" * 72)
    print(f"device      : {dev}")
    print(f"layout      : {args.layout}")
    print(f"fnn_folder  : {args.fnn_folder or '(constants default)'}")
    print(f"recon_dir   : {args.recon_dir or '(constants default)'}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mountain).to(dev)

    corpus_path, _ = _etu._corpus_paths(args.corpus)
    elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev, corpus_override=args.corpus)
    prim = _etu.build_primaries(corpus_path, B, mountain).to(dev)
    kfn = _etu.KernelDualLabels(elec, muon, surf, dev, chunk=args.kernel_chunk)
    fnn, recon = load_models(dev,
                             fnn_folder=args.fnn_folder or FNN_FOLDER,
                             recon_dir=args.recon_dir or RECON_FOLDER + "_deepsets")
    print(f"events      : {B} of {n_pairs}")

    _etu.LAYOUT_PATH = args.layout
    e_opt, n_opt = _etu.load_layout(mountain)

    rows = []

    def add(tag, e, n, d_min=None):
        U_s, U_t, _, _ = _etu.score(e, n, prim, fnn, kfn, recon, dev)
        mn, p1, p10 = pair_stats(e, n)
        rows.append(dict(tag=tag, d_min=d_min, U_surrogate=U_s, U_true=U_t,
                         min_sep=mn, pairs_lt_1m=p1, pairs_lt_10m=p10))
        print(f"  {tag:22s} d_min={str(d_min):>5}  U_surr={U_s:+8.3f}  "
              f"U_true={U_t:+8.3f}  min_sep={mn:8.3f} m  <1m={p1:3d}  <10m={p10:3d}")

    print("\nbaselines and the layout as saved:")
    e_g, n_g = _etu.grid_layout(mountain);   add("grid baseline", e_g, n_g)
    e_c, n_c = _etu.center_layout(mountain); add("center baseline", e_c, n_c)
    add("optimized (as saved)", e_opt, n_opt, 0)

    print("\nseparation sweep on the optimized layout:")
    for d in [float(x) for x in args.d_min.split(",") if float(x) > 0]:
        e_r, n_r = relax(e_opt, n_opt, mountain, d, seed=args.seed)
        add(f"optimized relaxed", e_r, n_r, d)

    base_t = rows[0]["U_true"]
    opt_t = rows[2]["U_true"]
    print(f"\n  kernel-scored gain of the saved layout over grid: "
          f"{opt_t - base_t:+.3f}  ({100.0 * (opt_t - base_t) / abs(base_t):+.1f}%)")
    print("  Compare that against the relaxed rows. A gain that survives a 10-20 m")
    print("  floor is mostly real; one that decays toward the baseline is bounded")
    print("  above by the stacking artifact (relaxation also moves detectors off")
    print("  the optimum, so it can only overstate the artifact, never understate).")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("tag,d_min,U_surrogate,U_true,min_sep,pairs_lt_1m,pairs_lt_10m\n")
            for r in rows:
                f.write(f"{r['tag']},{r['d_min']},{r['U_surrogate']:.6g},"
                        f"{r['U_true']:.6g},{r['min_sep']:.6g},"
                        f"{r['pairs_lt_1m']},{r['pairs_lt_10m']}\n")
        print(f"[save] {args.csv}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(dict(layout=args.layout, n_events=B, rows=rows), f, indent=2)
        print(f"[save] {args.out_json}")

    if args.plot_dir:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(args.plot_dir, exist_ok=True)
            sw = [r for r in rows if r["tag"].startswith("optimized")]
            d = [r["d_min"] for r in sw]
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
            ax[0].plot(d, [r["U_true"] for r in sw], "o-", label="optimized (kernel U)")
            ax[0].axhline(rows[0]["U_true"], color="k", ls="--", label="grid baseline")
            ax[0].axhline(rows[1]["U_true"], color="gray", ls=":", label="center baseline")
            ax[0].set_xlabel("enforced minimum separation [m]"); ax[0].set_ylabel("kernel-scored U")
            ax[0].set_title("does the gain survive de-stacking?"); ax[0].legend(fontsize=8)
            ax[1].plot(d, [r["U_surrogate"] for r in sw], "s-", color="#F58518",
                       label="optimized (surrogate U)")
            ax[1].axhline(rows[0]["U_surrogate"], color="k", ls="--", label="grid baseline")
            ax[1].set_xlabel("enforced minimum separation [m]"); ax[1].set_ylabel("surrogate U")
            ax[1].set_title("the objective the optimizer actually saw"); ax[1].legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(args.plot_dir, "layout_collapse_sweep.png")
            fig.savefig(p, dpi=140); plt.close(fig)
            print(f"[plot] wrote {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"[plot] SKIPPED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
