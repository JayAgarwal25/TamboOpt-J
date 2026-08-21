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
call signature is passed into the UNMODIFIED `objective.utility_of_xy`, guaranteeing
identical recon path, transforms and composite weights.

All paths come from constants.py, so this scores whatever run those point at.

    cd TambOpt
    python plots/layouts/true_utility.py --n-events 512
"""
import argparse
import os
import sys

# `_HERE` is this file's own directory; `_V6` the repo root, found by walking up
# to the _pathfix.py marker instead of counting parents. Counting is what broke
# the old plots/single_species/ scripts: written for plots/*.py, they resolved
# the "repo root" to plots/ and could not import `modules` at all.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import modules  # noqa: F401 — package import; keeps modules on the path
import showerdata
from modules.data import compute_labels_batch, place_clouds_enu
from modules.surrogates import encode_primary
from modules.surrogates import combine_species_outputs
from modules.geometry import SurfaceUpMap
from modules.geometry import sample_initial_layout_ne, project_to_mountain_ne
from modules.geometry import load_tr_mountain
from modules.optimize import utility_of_xy, load_models
from modules.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, T_LOG_SCALE,
    HELDOUT_SHOWER_CACHE_PATH, HELDOUT_POSITIONS_PATH,
    OPT_FOLDER,
)

# LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_ensemble_full_corpus_grid", "layout_best.pt")
# LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_ensemble_full_corpus_center", "layout_best.pt")
# LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_activation_center", "layout_best.pt")
LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_activation_grid", "layout_best.pt")

class KernelDualLabels:
    """Drop-in for the dual surrogate: same ``(primary_batch, xy_batch) -> (B, n_det, 2)``
    call signature, but the labels come from the plane-aware KERNEL run on the real
    (pre-placed) shower clouds instead of the neural net.

    The two per-species clouds are the ground truth for the B events; every row of
    a call shares one layout (read from ``xy_batch[0]``). The raw per-species counts
    are combined into the surrogate's own output space with
    `dual.combine_species_outputs` (log1p(N_tot), log1p(t_tot*T_LOG_SCALE)),
    so the frozen recon sees inputs in the space it was trained on. `primary_batch`
    is ignored: the clouds already fix which events this is.

    `chunk` splits the call over events. The kernel materializes several
    (B, points, n_det) fp32 tensors at once -- at B=512 with ~25k points/event
    that is 4.79 GiB EACH, and `spatial = exp(-(dx**2 + dy**2)/...)` alone holds
    five of them live, so the whole batch needs ~30 GiB and OOMs a 20 GB card.
    Chunking is exact, not an approximation: compute_labels_batch reduces only
    over a shower's own points and combine_species_outputs is elementwise, so no
    event ever sees another. The utility statistics downstream still average over
    all B events -- unlike lowering --n-events, which shrinks the sample."""

    def __init__(self, elec_clouds, muon_clouds, surface, device, chunk=0):
        self.elec = elec_clouds.to(device)
        self.muon = muon_clouds.to(device)
        self.surface = surface
        # 0 / None -> whole batch at once (previous behaviour).
        self.chunk = int(chunk) or int(self.elec.shape[0])

    def _labels(self, clouds, e_det, n_det):
        E, T = compute_labels_batch(clouds, e_det, n_det, self.surface)
        return torch.stack([torch.log1p(E), torch.log1p(T * T_LOG_SCALE)], dim=-1)

    def __call__(self, primary_batch, xy_batch):
        e_det, n_det = xy_batch[0, :, 0], xy_batch[0, :, 1]      # layout shared across batch
        B = int(self.elec.shape[0])
        out = []
        for lo in range(0, B, self.chunk):
            hi = min(lo + self.chunk, B)
            pred_e  = self._labels(self.elec[lo:hi], e_det, n_det)
            pred_mu = self._labels(self.muon[lo:hi], e_det, n_det)
            out.append(combine_species_outputs(pred_e, pred_mu))
        return out[0] if len(out) == 1 else torch.cat(out, dim=0)


def load_events(n_events, device, mountain):
    """Load and PLACE the first `n_events` events' electron + muon clouds, and
    build their matching `primary_batch` — all from the SAME heldout corpus.

    Reads the HELDOUT corpus (HOLDOUT_FRAC of physical events, split off by
    Step 0 before generation) — NOT the main training corpus. Steps 1-4 (FNN
    train+val, recon train+val, and the stage-4 layout optimizer, which sweeps
    the ENTIRE main corpus with no split of its own) never see these events,
    so "first B rows" here is genuinely unseen by everything upstream, unlike
    reading from DUAL_SHOWER_CACHE_PATH.

    `primary_batch` is built here (via `encode_primary`, identical to Step 1)
    rather than sliced from TRAINING_DATASET_FOLDER/primary.pt — that file only
    ever covers the main corpus, so slicing its first B rows would pair the
    heldout clouds with the WRONG events' ground-truth direction/energy labels,
    not just reintroduce leakage.

    The heldout corpus is [electron block | muon block] with event i at row i
    and row n_pairs+i (paired, sharing the primary → same decay vertex +
    direction). Placement uses the pipeline's C8 `place_clouds_enu` at the
    real vertex."""
    positions_all = torch.load(HELDOUT_POSITIONS_PATH)           # (M, 3) ENU E,N,U
    n_pairs = positions_all.shape[0] // 2
    B = min(n_events, n_pairs)

    e_sub = showerdata.load(HELDOUT_SHOWER_CACHE_PATH, start=0, stop=B)
    m_sub = showerdata.load(HELDOUT_SHOWER_CACHE_PATH, start=n_pairs, stop=n_pairs + B)
    elec = torch.as_tensor(e_sub.points, dtype=torch.float32)
    muon = torch.as_tensor(m_sub.points, dtype=torch.float32)
    dirs = torch.as_tensor(e_sub.directions, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    energies = torch.as_tensor(e_sub.energies, dtype=torch.float32)
    pdg      = torch.as_tensor(e_sub.pdg,      dtype=torch.long)
    pos = positions_all[:B].float()

    place_clouds_enu(elec, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    place_clouds_enu(muon, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)

    array_center = torch.as_tensor(mountain.centroids_ENU, dtype=torch.float32).mean(dim=0)
    primary_batch = encode_primary(dirs, energies, pdg, pos, array_center).to(device)
    return elec, muon, B, n_pairs, primary_batch


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
    ap.add_argument("--n-events", type=int, default=5120,
                    help="fixed primary/cloud batch size for the objective")
    ap.add_argument("--kernel-chunk", type=int, default=128,
                    help="events per kernel call (0 = whole batch). Memory only: "
                         "the result is identical for any value, since the kernel "
                         "is per-event. ~0.6 GiB per (chunk=64) kernel tensor vs "
                         "4.79 GiB at 512 — lower this, not --n-events, when the "
                         "GPU OOMs, so the utility keeps its full sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-layout", action="store_true",
                    help="use grid layout as baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("true-utility evaluator — kernel vs surrogate on the SAME recon + weights")
    print("=" * 72)
    print(f"device      : {device}")
    print(f"corpus      : {HELDOUT_SHOWER_CACHE_PATH}  (held-out, unseen by Steps 1-4)")
    print(f"layout(opt) : {LAYOUT_PATH}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(device)
    elec, muon, B, n_pairs, prim = load_events(args.n_events, device, mountain)
    print(f"events      : {B} of {n_pairs} pairs")
    kernel_fnn = KernelDualLabels(elec, muon, surface, device,
                                  chunk=args.kernel_chunk)
    print(f"kernel chunk: {kernel_fnn.chunk} events/call")

    fnn, recon = load_models(device)

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
