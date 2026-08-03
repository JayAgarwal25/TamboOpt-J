"""Test #1 — is an optimized layout better under GROUND TRUTH, or only the surrogate?

The optimizers maximize U through the FNN surrogate, so a rising U only proves the
layout looks better *to the surrogate*. If the surrogate is wrong (this project's
stage-1 R^2 ~ 0), maximizing U can just walk detectors into the surrogate's own
artifacts. This script re-scores a layout with the composite objective computed
from the plane-aware KERNEL (`compute_labels_batch`, the ground truth the surrogate
approximates) instead of the FNN, feeding the SAME recon and the SAME weights — so
only the label source differs — and prints:

                    surrogate-U      true-U
      baseline grid      A              C
      optimized          B              D

    B > A but D ~ C   -> the movement is a SURROGATE ARTIFACT.
    B > A and D > C    -> genuine improvement (survives ground truth).

Only the label source is swapped: a kernel-backed stand-in with the FNN's exact
call signature is passed into the UNMODIFIED `opt_core.utility_of_xy`, guaranteeing
identical recon path, transforms and composite weights.

All paths come from constants.py, so this scores whatever run those point at.

    cd TambOpt/detector_optimization_v6
    python plots/eval_true_utility.py --n-events 512
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
import showerdata
from modules_v6.fnn_surrogate_ne import compute_labels_batch, place_clouds_enu
from modules_v6.dual_surrogate import combine_species_outputs
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v6.tr_geometry_ne import sample_initial_layout_ne, project_to_mountain_ne
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models
from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, T_LOG_SCALE,
    DUAL_SHOWER_CACHE_PATH, DUAL_POSITIONS_PATH, TRAINING_DATASET_FOLDER, OPT_FOLDER,
)

# LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_ensemble_full_corpus_grid", "layout_best.pt")
LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_ensemble_full_corpus_center", "layout_best.pt")

class KernelDualLabels:
    """Drop-in for the dual surrogate: same ``(primary_batch, xy_batch) -> (B, n_det, 2)``
    call signature, but the labels come from the plane-aware KERNEL run on the real
    (pre-placed) shower clouds instead of the neural net.

    The two per-species clouds are the ground truth for the B events; every row of
    a call shares one layout (read from ``xy_batch[0]``). The raw per-species counts
    are combined into the surrogate's own output space with
    `dual_surrogate.combine_species_outputs` (log1p(N_tot), log1p(t_tot*T_LOG_SCALE)),
    so the frozen recon sees inputs in the space it was trained on. `primary_batch`
    is ignored: the clouds already fix which events this is."""

    def __init__(self, elec_clouds, muon_clouds, surface, device):
        self.elec = elec_clouds.to(device)
        self.muon = muon_clouds.to(device)
        self.surface = surface

    def __call__(self, primary_batch, xy_batch):
        e_det, n_det = xy_batch[0, :, 0], xy_batch[0, :, 1]      # layout shared across batch
        E_e, T_e = compute_labels_batch(self.elec, e_det, n_det, self.surface)
        E_mu, T_mu = compute_labels_batch(self.muon, e_det, n_det, self.surface)
        pred_e = torch.stack([torch.log1p(E_e), torch.log1p(T_e * T_LOG_SCALE)], dim=-1)
        pred_mu = torch.stack([torch.log1p(E_mu), torch.log1p(T_mu * T_LOG_SCALE)], dim=-1)
        return combine_species_outputs(pred_e, pred_mu)


def load_events(n_events, device):
    """Load and PLACE the first `n_events` events' electron + muon clouds.

    The tau dual corpus is [electron block | muon block] with event i at row i and
    row n_pairs+i (paired, sharing the primary → same decay vertex + direction).
    Placement uses the pipeline's C8 `place_clouds_enu` at the real vertex."""
    positions_all = torch.load(DUAL_POSITIONS_PATH)              # (M, 3) ENU E,N,U
    n_pairs = positions_all.shape[0] // 2
    B = min(n_events, n_pairs)

    e_sub = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=0, stop=B)
    m_sub = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=n_pairs, stop=n_pairs + B)
    elec = torch.as_tensor(e_sub.points, dtype=torch.float32)
    muon = torch.as_tensor(m_sub.points, dtype=torch.float32)
    dirs = torch.as_tensor(e_sub.directions, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    pos = positions_all[:B].float()

    place_clouds_enu(elec, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    place_clouds_enu(muon, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    return elec, muon, B, n_pairs


def _snap(mountain, e, n):
    e, n = project_to_mountain_ne(mountain, e.float().reshape(-1), n.float().reshape(-1))
    return e.float(), n.float()


def load_layout(mountain):
    raw = torch.load(LAYOUT_PATH, map_location="cpu", weights_only=False)
    e, n = (raw["x"], raw["y"]) if isinstance(raw, dict) else (raw[:, 0], raw[:, 1])
    return _snap(mountain, e, n)


def grid_layout(mountain):
    e, n = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme="grid")
    return _snap(mountain, torch.as_tensor(np.asarray(e)), torch.as_tensor(np.asarray(n)))

def center_layout(mountain):
    e, n = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme="center")
    return _snap(mountain, torch.as_tensor(np.asarray(e)), torch.as_tensor(np.asarray(n)))


@torch.no_grad()
def score(e_det, n_det, primary_batch, fnn, kernel_fnn, recon, device):
    """(U_surrogate, U_true, parts_surrogate, parts_true) for one layout."""
    x, y = e_det.to(device), n_det.to(device)
    U_s, _, p_s = utility_of_xy(x, y, primary_batch, fnn, recon)
    U_t, _, p_t = utility_of_xy(x, y, primary_batch, kernel_fnn, recon)
    return float(U_s.item()), float(U_t.item()), p_s, p_t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-events", type=int, default=512,
                    help="fixed primary/cloud batch size for the objective")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-layout", action="store_true",
                    help="use grid layout as baseline")
    ap.add_argument("--recon_dir", type=str, default=None,
                    help="Recon checkpoint directory to score (default: "
                         "constants RECON_FOLDER + '_deepsets'). Point at a "
                         "C0/T1/T2 experiment recon to compare them on one layout.")
    ap.add_argument("--layout", type=str, default=None,
                    help="Path to the OPTIMIZED layout_best.pt to score against the "
                         "baseline (default: the constants full_corpus_grid layout).")
    args = ap.parse_args()
    if args.layout:
        global LAYOUT_PATH
        LAYOUT_PATH = args.layout

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("true-utility evaluator — kernel vs surrogate on the SAME recon + weights")
    print("=" * 72)
    print(f"device      : {device}")
    print(f"corpus      : {DUAL_SHOWER_CACHE_PATH}")
    print(f"layout(opt) : {LAYOUT_PATH}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(device)
    elec, muon, B, n_pairs = load_events(args.n_events, device)
    print(f"events      : {B} of {n_pairs} pairs")
    kernel_fnn = KernelDualLabels(elec, muon, surface, device)

    fnn, recon = load_models(device, recon_dir=args.recon_dir)
    prim = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()[:B].to(device)

    e_o, n_o = load_layout(mountain)
    if args.grid_layout:
        e_g, n_g = grid_layout(mountain)
    else:
        e_g, n_g = center_layout(mountain)
    gs, gt, _, _ = score(e_g, n_g, prim, fnn, kernel_fnn, recon, device)
    os_, ot, ops, opt_ = score(e_o, n_o, prim, fnn, kernel_fnn, recon, device)

    print()
    if args.grid_layout:
        print("GRID LAYOUT (baseline) vs OPTIMIZED LAYOUT")
    else:
        print("CENTER LAYOUT (baseline) vs OPTIMIZED LAYOUT")
    print("                  surrogate-U     true-U")
    print(f"  baseline grid   {gs:11.4f}   {gt:11.4f}")
    print(f"  optimized       {os_:11.4f}   {ot:11.4f}")
    print()
    d_surr, d_true = os_ - gs, ot - gt
    print(f"  ΔU surrogate (opt - grid) : {d_surr:+.4f}")
    print(f"  ΔU true      (opt - grid) : {d_true:+.4f}")
    print(f"  artifact gap (surr - true, optimized) : {os_ - ot:+.4f}")
    print()
    tol = 0.02 * max(abs(gt), abs(gs), 1.0)
    if d_surr <= tol:
        verdict = "optimizer did not raise even surrogate-U here (check the run)."
    elif d_true > tol and d_true >= 0.5 * d_surr:
        verdict = "GENUINE — the gain largely survives ground truth."
    elif d_true > tol:
        verdict = "PARTIAL — some real gain, but the surrogate overstates it."
    else:
        verdict = "ARTIFACT — surrogate-U rose but true-U did not; the movement " \
                  "exploits the surrogate, not the physics."
    print(f"  VERDICT: {verdict}")
    print()
    print("  component breakdown (surrogate | true), optimized layout:")
    for k in ("u_theta", "u_phi", "u_e", "u_pr"):
        print(f"    {k:8s}  {float(ops[k].item()):+9.4f} | {float(opt_[k].item()):+9.4f}")


if __name__ == "__main__":
    main()
