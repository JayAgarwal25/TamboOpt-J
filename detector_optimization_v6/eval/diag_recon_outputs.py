"""Diagnostic: dump a recon's raw per-column OUTPUT distribution vs the truth,
on the fixed eval batch, for both label sources — to tell a diverged/garbage
output head apart from a metric quirk. Read-only; no training.

    python eval/diag_recon_outputs.py --recon_dir <dir> --layout grid --n-events 512
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
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, TRAINING_DATASET_FOLDER,
    FNN_FOLDER, RECON_FOLDER,
)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_etu", os.path.join(_HERE, "plots", "eval_true_utility.py"))
_etu = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_etu)

_COLS4 = ["dir_x", "dir_y", "dir_z", "log_e_norm"]
_COLS7 = _COLS4 + ["rel_E", "rel_N", "rel_U"]


def _stats(t):
    t = t.detach().float().cpu()
    return f"min={t.min():+.3g} med={t.median():+.3g} max={t.max():+.3g} std={t.std():.3g}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", type=str, default=None)
    ap.add_argument("--n-events", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layout", type=str, default="grid")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mtn).to(dev)
    elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev)
    kfn = _etu.KernelDualLabels(elec, muon, surf, dev)
    fnn, recon = load_models(dev, fnn_folder=FNN_FOLDER,
                             recon_dir=args.recon_dir or RECON_FOLDER + "_deepsets")
    D = int(getattr(recon, "output_dim", 4)); cols = _COLS7 if D == 7 else _COLS4
    prim = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()[:B].to(dev)
    tgt_cols = [0, 1, 2, 3, 5, 6, 7] if D == 7 else [0, 1, 2, 3]

    if args.layout == "grid":
        e_l, n_l = _etu.grid_layout(mtn)
    elif args.layout == "center":
        e_l, n_l = _etu.center_layout(mtn)
    else:
        _etu.LAYOUT_PATH = args.layout; e_l, n_l = _etu.load_layout(mtn)

    print("=" * 72)
    print(f"recon output diagnostic  recon_dir={args.recon_dir}  output_dim={D}  layout={args.layout}")
    print("=" * 72)
    print("TRUE target columns:")
    for j, c in enumerate(cols):
        print(f"  {c:11s} {_stats(prim[:, tgt_cols[j]])}")
    for name, lab in (("surrogate (FNN)", fnn), ("true (KERNEL)", kfn)):
        x, y = e_l.to(dev), n_l.to(dev)
        xy = torch.stack([x, y], -1).unsqueeze(0).expand(B, -1, -1)
        ET = lab(prim, xy)
        feats = torch.stack([xy[..., 0], xy[..., 1], ET[..., 0], ET[..., 1]], -1)
        pred = recon(feats)
        print(f"\nPRED columns  [{name}]:")
        for j, c in enumerate(cols):
            d = (pred[:, j] - prim[:, tgt_cols[j]]).abs()
            print(f"  {c:11s} {_stats(pred[:, j])}   median|Δ|={d.median():.4g}")


if __name__ == "__main__":
    main()
