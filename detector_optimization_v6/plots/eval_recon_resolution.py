"""Bounded, physics-interpretable recon resolution — companion to eval_true_utility.py.

`eval_true_utility.py` reports the (reciprocal) composite utility; that number
blows up on the FNN's denoised mean inputs and is hard to read physically. This
script instead reports MEDIAN RESOLUTIONS on a fixed event batch and a fixed
layout, for a chosen recon checkpoint, under BOTH label sources:

    - angular resolution  Δψ   [deg]   (great-circle angle between pred & true dir)
    - energy resolution    |Δlog10 E|  [dex]
    - vertex resolution    ‖Δv‖  [m]    (only if the recon has 7 outputs)

  input = FNN   : recon fed dual-surrogate (mean) (E,T)   → "surrogate" view
  input = KERNEL: recon fed ground-truth kernel (E,T)     → "true" view

Nothing here is optimized or trained; it only scores an existing recon. Point it
at a C0 / T1 / T2 experiment recon with --recon_dir and keep --layout fixed to
compare them apples-to-apples.

    cd TambOpt/detector_optimization_v6
    python plots/eval_recon_resolution.py --recon_dir <dir> --layout grid --n-events 512
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v6.opt_core import load_models
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, LOG_E_MIN, LOG_E_MAX,
    TRAINING_DATASET_FOLDER,
)
# Reuse the already-tested event loader, kernel-label stand-in, and layouts.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_eval_true_utility", os.path.join(_HERE, "plots", "eval_true_utility.py"))
_etu = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_etu)


def _median_resolutions(pred, primary, has_vertex):
    """pred (B, D) raw encode units; primary (B, 8). Returns dict of medians."""
    # Direction: great-circle angle between unit vectors.
    pd = pred[:, 0:3] / pred[:, 0:3].norm(dim=1, keepdim=True).clamp(min=1e-12)
    td = primary[:, 0:3] / primary[:, 0:3].norm(dim=1, keepdim=True).clamp(min=1e-12)
    cos = (pd * td).sum(dim=1).clamp(-1.0, 1.0)
    dpsi_deg = torch.arccos(cos) * (180.0 / math.pi)
    # Energy: |Δ log10 E| in dex (log_e_norm spans [LOG_E_MIN, LOG_E_MAX]).
    dlogE = (pred[:, 3] - primary[:, 3]).abs() * (LOG_E_MAX - LOG_E_MIN)
    out = {"dpsi_deg": float(dpsi_deg.median()),
           "dlogE_dex": float(dlogE.median())}
    if has_vertex:
        dv = (pred[:, 4:7] - primary[:, 5:8]).norm(dim=1)   # metres
        out["dvertex_m"] = float(dv.median())
    return out


@torch.no_grad()
def _recon_pred(e_det, n_det, primary_batch, labeller, recon, device):
    x, y = e_det.to(device), n_det.to(device)
    B = primary_batch.shape[0]
    xy = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1)   # (B, n_det, 2)
    ET = labeller(primary_batch, xy)                                 # (B, n_det, 2)
    feats = torch.stack([xy[..., 0], xy[..., 1], ET[..., 0], ET[..., 1]], dim=-1)
    return recon(feats)                                              # (B, D) raw units


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon_dir", type=str, default=None)
    ap.add_argument("--n-events", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layout", type=str, default="grid",
                    help="'grid', 'center', or a path to a layout_best.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(device)
    elec, muon, B, n_pairs = _etu.load_events(args.n_events, device)
    kernel_fnn = _etu.KernelDualLabels(elec, muon, surface, device)
    fnn, recon = load_models(device, recon_dir=args.recon_dir)
    has_vertex = int(getattr(recon, "output_dim", 4)) == 7

    prim = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()[:B].to(device)

    if args.layout == "grid":
        e_l, n_l = _etu.grid_layout(mountain)
    elif args.layout == "center":
        e_l, n_l = _etu.center_layout(mountain)
    else:
        _etu.LAYOUT_PATH = args.layout
        e_l, n_l = _etu.load_layout(mountain)

    print("=" * 72)
    print("recon resolution — kernel vs surrogate labels, same recon + layout")
    print("=" * 72)
    print(f"device      : {device}")
    print(f"recon_dir   : {args.recon_dir or '(constants default)'}")
    print(f"output_dim  : {getattr(recon, 'output_dim', 4)}  (vertex={'yes' if has_vertex else 'no'})")
    print(f"layout      : {args.layout}")
    print(f"events      : {B} of {n_pairs} pairs")

    for name, labeller in (("surrogate (FNN)", fnn), ("true (KERNEL)", kernel_fnn)):
        pred = _recon_pred(e_l, n_l, prim, labeller, recon, device)
        res = _median_resolutions(pred, prim, has_vertex)
        line = (f"  {name:18s}  Δψ={res['dpsi_deg']:7.3f} deg   "
                f"|Δlog10E|={res['dlogE_dex']:6.4f} dex")
        if has_vertex:
            line += f"   ‖Δvertex‖={res['dvertex_m']:8.1f} m"
        print(line)


if __name__ == "__main__":
    main()
