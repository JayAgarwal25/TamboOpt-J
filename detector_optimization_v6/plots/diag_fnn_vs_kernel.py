"""Diagnostic: does the FNN's error vs the kernel GROW on the optimized layout?

If the layout optimizer is exploiting FNN error (a Stage-2 contribution to the
non-transfer), then ‖FNN(E,T) − kernel(E,T)‖ should be larger on the optimized
layout than on the grid baseline. This compares that residual, per label channel,
on both layouts, on the fixed event batch. Read-only; no training.

    python plots/diag_fnn_vs_kernel.py --n-events 512
"""
import argparse, os, sys
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np, torch
import modules_v6  # noqa: F401
from modules_v6.opt_core import load_models
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6 import run_world
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX,
    N_PLANES
)

# Bound in main() from the resolved run world, never at import.
TRAINING_DATASET_FOLDER = None
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_etu", os.path.join(_HERE, "plots", "eval_true_utility.py"))
_etu = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_etu)


@torch.no_grad()
def _residual(e_l, n_l, prim, fnn, kfn, dev):
    """Return (‖ΔE‖, ‖ΔT‖, meanΔE, meanΔT) between FNN and kernel labels."""
    B = prim.shape[0]
    xy = torch.stack([e_l.to(dev), n_l.to(dev)], -1).unsqueeze(0).expand(B, -1, -1)
    f = fnn(prim, xy)          # (B, n_det, 2)  FNN mean
    k = kfn(prim, xy)          # (B, n_det, 2)  kernel truth
    dE = (f[..., 0] - k[..., 0])
    dT = (f[..., 1] - k[..., 1])
    rmseE = dE.pow(2).mean().sqrt().item()
    rmseT = dT.pow(2).mean().sqrt().item()
    return rmseE, rmseT, dE.abs().mean().item(), dT.abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    run_world.add_run_world_args(ap)
    args = ap.parse_args()

    global TRAINING_DATASET_FOLDER
    W = run_world.resolve(args, need_write=False)
    TRAINING_DATASET_FOLDER = W.dataset_folder
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mtn).to(dev)
    elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev)
    kfn = _etu.KernelDualLabels(elec, muon, surf, dev)
    fnn, _ = load_models(dev, fnn_folder=W.fnn_folder, recon_dir=W.recon_dir)   # recon not needed; only the FNN
    prim = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()[:B].to(dev)

    e_g, n_g = _etu.grid_layout(mtn)
    e_o, n_o = _etu.load_layout(mtn)   # LAYOUT_PATH = the optimized full_corpus_grid layout

    print("=" * 72)
    print("FNN-vs-kernel label residual — grid baseline vs optimized layout")
    print(f"layout(opt): {_etu.LAYOUT_PATH}")
    print(f"events     : {B} of {n_pairs} pairs (log-space combined E,T)")
    print("=" * 72)
    for tag, (e_l, n_l) in (("grid     ", (e_g, n_g)), ("optimized", (e_o, n_o))):
        rE, rT, mE, mT = _residual(e_l, n_l, prim, fnn, kfn, dev)
        print(f"  {tag}   RMSE(ΔE)={rE:.4f}  RMSE(ΔT)={rT:.4f}   "
              f"mean|ΔE|={mE:.4f}  mean|ΔT|={mT:.4f}")
    print("\nIf the optimized-layout residual is notably larger than grid, the "
          "optimizer is exploiting FNN error (Stage-2 contribution to non-transfer).")


if __name__ == "__main__":
    main()
