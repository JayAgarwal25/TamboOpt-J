"""Regenerate ONLY the DE ensemble plots from already-saved layouts.

Reuses the optimized layouts persisted by 04_optimize_differential_evolution.py
(layouts_all.pt / layout_best.pt / layout_mean.pt) — no re-optimization — and
re-draws layout_ensemble.png + layout_density.png in the (North, Up) cross
section via the SurfaceUpMap projection, so they line up with the L-BFGS
ensemble figures. The DE optimiser works in the North–East plane, so each
detector's East is projected to Up = g(North, East) before plotting.

Usage (from the v6 folder):

    python replot_de_ensemble_up.py                 # default: the grid scheme dir
    python replot_de_ensemble_up.py <dir> [<dir>..] # explicit scheme dir(s)
"""
import importlib.util
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The optimizer module name starts with a digit → load it by path.
_spec = importlib.util.spec_from_file_location(
    "de_opt", os.path.join(_HERE, "04_optimize_differential_evolution.py"))
de = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(de)


def _load_dir(d):
    la = torch.load(os.path.join(d, "layouts_all.pt"), map_location="cpu")
    lb = torch.load(os.path.join(d, "layout_best.pt"),  map_location="cpu")
    lm = torch.load(os.path.join(d, "layout_mean.pt"),  map_location="cpu")
    aligned = np.asarray(la["aligned"])                      # (K, n_det, 2) = (N, E)
    best_x  = np.asarray(lb["x"]);  best_y = np.asarray(lb["y"])
    mean_xy = np.stack([np.asarray(lm["mean_x"]), np.asarray(lm["mean_y"])], axis=-1)
    std_xy  = np.stack([np.asarray(lm["std_x"]),  np.asarray(lm["std_y"])],  axis=-1)
    return aligned, mean_xy, std_xy, best_x, best_y


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="scheme output dir(s) with layouts_all.pt")
    ap.add_argument("--vmax", type=float, default=0.2,
                    help="density colorbar upper limit (plots 0..vmax); pass <=0 to auto-scale")
    args = ap.parse_args()
    dirs = args.dirs or [de.OPT_DIR_TEMPLATE.format(scheme="grid")]
    vmax = args.vmax if args.vmax and args.vmax > 0 else None

    mountain = de.load_tr_mountain(
        de.GEOMETRY_PATH_RESOLVED, de.GEOMETRY_GROUP, de.DET_KEY,
        east_entry=de.EAST_ENTRY, layer_east_dx=de.LAYER_EAST_DX, n_planes=de.N_PLANES,
    )
    # CPU surface — replotting needs no GPU.
    surface = de.SurfaceUpMap.from_mountain(mountain).to("cpu")

    for d in dirs:
        if not os.path.exists(os.path.join(d, "layouts_all.pt")):
            print(f"[skip] {d} (no layouts_all.pt)")
            continue
        aligned, mean_xy, std_xy, best_x, best_y = _load_dir(d)
        print(f"[replot] {d}  aligned={aligned.shape}  density vmax={vmax}")
        de._plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                          mountain, os.path.join(d, "layout_ensemble.png"), surface=surface)
        de._plot_density_heatmap(aligned, best_x, best_y,
                          mountain, os.path.join(d, "layout_density.png"),
                          surface=surface, vmax=vmax)


if __name__ == "__main__":
    main()
