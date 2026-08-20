"""Test #2 — is the ACTIVATION gain real, or only the surrogate's opinion?

Sibling of `eval_true_utility.py`, same trick for a different objective.
`04_optimize_lbfgs_activation.py` maximizes `opt_core.activation_of_xy` through the
FNN surrogate, so a rising U only proves the layout collects more *to the
surrogate*. This re-scores it with the objective computed from the plane-aware
KERNEL (`compute_labels_batch`, the ground truth the surrogate approximates)
instead, and prints:

                    surrogate-U      true-U
      baseline           A              C
      optimized          B              D

    B > A but D ~ C   -> the movement is a SURROGATE ARTIFACT.
    B > A and D > C    -> genuine improvement (survives ground truth).

The swap is free: `KernelDualLabels` already has the FNN's exact
``(primary_batch, xy_batch) -> (B, n_det, 2)`` signature, and `activation_of_xy`
takes the surrogate positionally, so the UNMODIFIED objective is called both ways
and only the label source differs. `expm1(log1p(E_kernel))` recovers the kernel
counts exactly, and `overlap_multiplicity` is layout-only and therefore identical
on both sides — so every difference below is the counts, nothing else.

READ THE RELATIVE COLUMN, NOT THE ABSOLUTE GAP. Unlike the composite U, the two
label sources are not on a common scale here: the surrogate is ~19x LOW on total
particles (measured 1.98e4 vs 3.77e5 for the grid layout), so "artifact gap" is
dominated by that offset and says nothing about whether the layout improved. The
fractional change (opt - base)/base is scale-free and is what the verdict uses.

All paths come from constants.py, so this scores whatever run those point at.

    cd TambOpt
    python plots/eval_true_activation.py                       # grid baseline
    python plots/eval_true_activation.py --center-layout       # center baseline
    python plots/eval_true_activation.py --objective particles --n-events 512
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # plots/
_V6 = os.path.dirname(_HERE)
for _p in (_V6, _HERE):                                     # _HERE: eval_true_utility
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import modules_v6  # noqa: F401 — package import; keeps modules_v6 on the path
from modules_v6.surface_map import SurfaceUpMap
from modules_v6.legacy_core.tr_geometry import load_tr_mountain
from modules_v6.opt_core import (
    activation_of_xy, load_models, PARTICLE_SCALE, DISTINCT_SCALE,
)
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, SIGMA_SPATIAL,
    HELDOUT_SHOWER_CACHE_PATH, OPT_FOLDER,
)
from eval_true_utility import (
    KernelDualLabels, load_events, _snap, grid_layout, center_layout,
)

# The activation optimizer's own output, not the ensemble script's — this exists
# to score THAT run. Override with --layout.
LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_activation_grid", "layout_best.pt")

# Reported for every run whatever --objective selects, so one run answers all three.
PARTS = (("u_distinct", "distinct", DISTINCT_SCALE),
         ("u_particles", "particles", PARTICLE_SCALE),
         ("u_detectors", "detectors", 1.0),
         ("u_pr", "reconstructability", 1.0))


def load_layout(path, mountain):
    """(East, North) from a layout_best.pt, snapped as every other layout is."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    e, n = (raw["x"], raw["y"]) if isinstance(raw, dict) else (raw[:, 0], raw[:, 1])
    return _snap(mountain, e, n)


@torch.no_grad()
def score(e_det, n_det, primary_batch, fnn, kernel_fnn, mode, device):
    """(U_surrogate, U_true, parts_surrogate, parts_true) for one layout.

    Both calls go through the UNMODIFIED `activation_of_xy`; the penalty is left
    off (its default) because this compares label sources, and the off-mesh term is
    layout-only — identical on both sides and so pure noise in the comparison.
    """
    x, y = e_det.to(device), n_det.to(device)
    U_s, _, p_s = activation_of_xy(x, y, primary_batch, fnn, mode=mode)
    U_t, _, p_t = activation_of_xy(x, y, primary_batch, kernel_fnn, mode=mode)
    return float(U_s.item()), float(U_t.item()), p_s, p_t


def _rel(base, opt):
    """Fractional change, the only cross-source-comparable number here."""
    return (opt - base) / abs(base) if abs(base) > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", choices=("distinct", "particles", "detectors"),
                    default="distinct",
                    help="which activation objective is the headline U; all three "
                         "are printed in the breakdown either way (default: distinct)")
    ap.add_argument("--n-events", type=int, default=5120,
                    help="fixed primary/cloud batch size for the objective")
    ap.add_argument("--kernel-chunk", type=int, default=128,
                    help="events per kernel call (0 = whole batch). Memory only: "
                         "the kernel is per-event, so the result is the same for "
                         "any value. Lower this, not --n-events, when the GPU OOMs.")
    ap.add_argument("--layout", default=LAYOUT_PATH,
                    help="optimized layout_best.pt to score")
    ap.add_argument("--center-layout", action="store_true",
                    help="use the center scheme as baseline instead of grid")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("true-activation evaluator — kernel vs surrogate on the SAME objective")
    print("=" * 78)
    print(f"device      : {device}")
    print(f"corpus      : {HELDOUT_SHOWER_CACHE_PATH}  (held-out, unseen by Steps 1-4)")
    print(f"layout(opt) : {args.layout}")
    print(f"objective   : {args.objective}  (sigma={SIGMA_SPATIAL:g} m, "
          f"particle_scale={PARTICLE_SCALE:g}, distinct_scale={DISTINCT_SCALE:g})")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(device)
    elec, muon, B, n_pairs, prim = load_events(args.n_events, device, mountain)
    print(f"events      : {B} of {n_pairs} pairs")
    kernel_fnn = KernelDualLabels(elec, muon, surface, device, chunk=args.kernel_chunk)
    print(f"kernel chunk: {kernel_fnn.chunk} events/call")

    fnn, _ = load_models(device)          # recon unused: activation never reconstructs

    e_o, n_o = load_layout(args.layout, mountain)
    base_name = "CENTER" if args.center_layout else "GRID"
    e_b, n_b = (center_layout if args.center_layout else grid_layout)(mountain)

    gs, gt, gps, gpt = score(e_b, n_b, prim, fnn, kernel_fnn, args.objective, device)
    os_, ot, ops, opt_ = score(e_o, n_o, prim, fnn, kernel_fnn, args.objective, device)

    print()
    print(f"{base_name} LAYOUT (baseline) vs OPTIMIZED LAYOUT   "
          f"[U = {args.objective}]")
    print("                  surrogate-U     true-U")
    print(f"  baseline        {gs:11.4f}   {gt:11.4f}")
    print(f"  optimized       {os_:11.4f}   {ot:11.4f}")
    print()
    d_surr, d_true = os_ - gs, ot - gt
    r_surr, r_true = _rel(gs, os_), _rel(gt, ot)
    print(f"  ΔU surrogate (opt - base) : {d_surr:+.4f}   ({100*r_surr:+.1f}%)")
    print(f"  ΔU true      (opt - base) : {d_true:+.4f}   ({100*r_true:+.1f}%)")
    print(f"  scale offset (true / surrogate, baseline) : {gt / gs:.2f}x")
    print()

    # Verdict on the FRACTIONAL change, not the absolute gap: the two label sources
    # differ by a large constant factor (see the module docstring), so comparing
    # d_surr to d_true directly would call every run an artifact.
    tol = 0.02
    if r_surr <= tol:
        verdict = "optimizer did not raise even surrogate-U here (check the run)."
    elif r_true > tol and r_true >= 0.5 * r_surr:
        verdict = "GENUINE — the gain largely survives ground truth."
    elif r_true > tol:
        verdict = "PARTIAL — some real gain, but the surrogate overstates it."
    else:
        verdict = ("ARTIFACT — surrogate-U rose but true-U did not; the movement "
                   "exploits the surrogate, not the physics.")
    print(f"  VERDICT: {verdict}")

    print()
    print("  all activation measures (surrogate | true):")
    print(f"    {'':22s}{'baseline':>22}{'optimized':>22}")
    for key, label, scale in PARTS:
        if key not in ops:
            continue
        b_s, b_t = float(gps[key].item()), float(gpt[key].item())
        o_s, o_t = float(ops[key].item()), float(opt_[key].item())
        print(f"    {label:20s}{b_s:10.3f} |{b_t:10.3f}"
              f"{o_s:11.3f} |{o_t:10.3f}")
    print(f"\n    (distinct x{DISTINCT_SCALE:g} and particles x{PARTICLE_SCALE:g} "
          f"give raw flux/shower; detectors is out of 100)")


if __name__ == "__main__":
    main()
