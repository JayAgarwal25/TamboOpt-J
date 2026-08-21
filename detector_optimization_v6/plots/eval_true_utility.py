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

`--objective {composite,detection}` selects which utility does the scoring, for
BOTH the surrogate and the kernel labels, so a layout optimized under
`04_optimize_lbfgs_ensemble.py --objective detection` can be checked against
ground truth the same way -- this matters because the surrogate is known to
over-report detector firing relative to the kernel (see
opt_core.detection_utility_of_xy's caveat), so a detection-optimized layout's
apparent gain has to survive the same surrogate-vs-kernel check the composite
objective already gets here. Default stays `composite`, byte-identical to the
pre-existing behaviour.

All paths come from constants.py, so this scores whatever run those point at.

    cd TambOpt/detector_optimization_v6
    python plots/eval_true_utility.py --n-events 512
    python plots/eval_true_utility.py --objective detection --layout <detection layout_best.pt>
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
from modules_v6.fnn_surrogate_ne import compute_labels_batch, place_clouds_enu, encode_primary
from modules_v6.dual_surrogate import combine_species_outputs
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v6.tr_geometry_ne import sample_initial_layout_ne, project_to_mountain_ne
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import (
    utility_of_xy, detection_utility_of_xy, load_models,
    RECONSTRUCT_THRESHOLD, LAYOUT_THRESHOLD,
)
from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, T_LOG_SCALE,
    HELDOUT_SHOWER_CACHE_PATH, HELDOUT_POSITIONS_PATH,
    OPT_FOLDER,
)

LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_ensemble_full_corpus_grid", "layout_best.pt")
# LAYOUT_PATH = os.path.join(OPT_FOLDER + "_lbfgs_ensemble_full_corpus_center", "layout_best.pt")

class KernelDualLabels:
    """Drop-in for the dual surrogate: same ``(primary_batch, xy_batch) -> (B, n_det, 2)``
    call signature, but the labels come from the plane-aware KERNEL run on the real
    (pre-placed) shower clouds instead of the neural net.

    The two per-species clouds are the ground truth for the B events; every row of
    a call shares one layout (read from ``xy_batch[0]``). The raw per-species counts
    are combined into the surrogate's own output space with
    `dual_surrogate.combine_species_outputs` (log1p(N_tot), log1p(t_tot*T_LOG_SCALE)),
    so the frozen recon sees inputs in the space it was trained on. `primary_batch`
    is ignored: the clouds already fix which events this is.

    Clouds are kept on the HOST and streamed to the device a chunk at a time. The
    kernel materializes (events, points, detectors) intermediates, so a few hundred
    events is already tens of GB; holding every cloud resident on top of that caps
    the event count far below the available validation set. `chunk` bounds the
    resident slice, and since the kernel is per-event independent (sum/mean over the
    point axis) chunking is numerically exact, not an approximation.

    `offset` lets a caller score a sub-batch of the stored events, so an outer loop
    can chunk the FNN and recon passes over the same windows."""

    needs_offset = True     # tells chunking callers this labeller is windowed

    def __init__(self, elec_clouds, muon_clouds, surface, device, chunk=None):
        self.elec = elec_clouds            # host-resident; sliced per call
        self.muon = muon_clouds
        self.surface = surface
        self.device = device
        n = int(elec_clouds.shape[0])
        self.chunk = int(chunk) if chunk else n

    def _labels(self, clouds, e_det, n_det):
        E, T = compute_labels_batch(clouds, e_det, n_det, self.surface)
        return torch.stack([torch.log1p(E), torch.log1p(T * T_LOG_SCALE)], dim=-1)

    def __call__(self, primary_batch, xy_batch, offset=0):
        e_det, n_det = xy_batch[0, :, 0], xy_batch[0, :, 1]      # layout shared across batch
        B = int(primary_batch.shape[0])
        outs = []
        for lo in range(0, B, self.chunk):
            hi = min(lo + self.chunk, B)
            el = self.elec[offset + lo: offset + hi].to(self.device, non_blocking=True)
            mu = self.muon[offset + lo: offset + hi].to(self.device, non_blocking=True)
            pred_e  = self._labels(el, e_det, n_det)
            pred_mu = self._labels(mu, e_det, n_det)
            outs.append(combine_species_outputs(pred_e, pred_mu))
            del el, mu, pred_e, pred_mu
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def _corpus_paths(corpus_override):
    """(corpus, positions) paths — the HELD-OUT pair by default, or an override
    plus its own `<corpus>_positions.pt` sidecar (same naming rule Step 1 uses).

    The default is the held-out corpus rather than the training one because every
    upstream stage reads the training corpus: the FNN and recon both split it
    internally, and the stage-4 sweep has no split at all, so scoring a layout
    there scores it on the events that produced it."""
    if not corpus_override:
        return HELDOUT_SHOWER_CACHE_PATH, HELDOUT_POSITIONS_PATH
    return corpus_override, os.path.splitext(corpus_override)[0] + "_positions.pt"


def build_primaries(corpus_path, B, mountain):
    """Encode the first `B` events' primaries straight from a shower corpus.

    `primary.pt` in TRAINING_DATASET_FOLDER is row-aligned with the corpus Step 1
    consumed, so it cannot be used for a DIFFERENT corpus (e.g. a held-out one) —
    slicing its first B rows would pair held-out clouds with the wrong events'
    ground truth, which is worse than leakage. This rebuilds the same 8-column
    encoding from the corpus metadata plus the decay-position sidecar, mirroring
    `build_training_pairs_ne`, so the held-out path stays byte-comparable with the
    training path (verified: max abs diff 0 on all 8 columns).

    Kept separate from `load_events` so any corpus can be encoded on its own, which
    is what the standalone evaluators in ~/v6_runs rely on."""
    from modules_v6.fnn_surrogate_ne import _load_positions_sidecar

    meta = showerdata.load_inc_particles(corpus_path)
    keep = np.arange(0, B)                       # electron block; the muon row shares the primary
    dirs   = torch.as_tensor(meta.directions[keep], dtype=torch.float32)
    energs = torch.as_tensor(meta.energies[keep],   dtype=torch.float32)
    pdg    = torch.as_tensor(meta.pdg[keep],        dtype=torch.long)
    positions = _load_positions_sidecar(corpus_path, keep)
    array_center = torch.as_tensor(mountain.centroids_ENU,
                                   dtype=torch.float32).mean(dim=0)
    return encode_primary(dirs, energs, pdg, positions, array_center)


def load_events(n_events, device, corpus_override=None):
    """Load and PLACE the first `n_events` events' electron + muon clouds.

    Defaults to the HELD-OUT corpus (see `_corpus_paths`), so "first B rows" is
    genuinely unseen by every upstream stage. The dual corpus is
    [electron block | muon block] with event i at row i and row n_pairs+i (paired,
    sharing the primary → same decay vertex + direction). Placement uses the
    pipeline's C8 `place_clouds_enu` at the real vertex.

    Returns clouds only; pair it with `build_primaries(corpus_path, B, mountain)`
    for the matching ground truth — never with a slice of `primary.pt`, which is
    row-aligned to the TRAINING corpus alone."""
    corpus_path, positions_path = _corpus_paths(corpus_override)
    positions_all = torch.load(positions_path)                   # (M, 3) ENU E,N,U
    n_pairs = positions_all.shape[0] // 2
    B = min(n_events, n_pairs)

    e_sub = showerdata.load(corpus_path, start=0, stop=B)
    m_sub = showerdata.load(corpus_path, start=n_pairs, stop=n_pairs + B)
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
def score(e_det, n_det, primary_batch, fnn, kernel_fnn, recon, device,
         objective="composite", detect_n=None, detect_e=None):
    """(U_surrogate, U_true, parts_surrogate, parts_true) for one layout.

    `objective` picks which utility scores BOTH the surrogate (`fnn`) and the
    kernel labels (`kernel_fnn`), so the two numbers stay directly comparable
    -- composite (default) is `utility_of_xy`, recon-aware, unchanged from
    before. detection is `opt_core.detection_utility_of_xy`, which takes no
    recon argument at all (`recon` is accepted here only so both objectives
    share one call signature; it is simply unused on this path)."""
    x, y = e_det.to(device), n_det.to(device)
    if objective == "detection":
        U_s, _, p_s = detection_utility_of_xy(
            x, y, primary_batch, fnn,
            reconstruct_threshold=detect_n, layout_threshold=detect_e)
        U_t, _, p_t = detection_utility_of_xy(
            x, y, primary_batch, kernel_fnn,
            reconstruct_threshold=detect_n, layout_threshold=detect_e)
    else:
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
    ap.add_argument("--recon_dir", type=str, default=None,
                    help="Recon checkpoint directory to score (default: "
                         "constants RECON_FOLDER + '_deepsets'). Point at a "
                         "C0/T1/T2 experiment recon to compare them on one layout.")
    ap.add_argument("--fnn_folder", type=str, default=None,
                    help="directory holding fnn_electron.pt and fnn_muon.pt. Without "
                         "this the surrogate comes from FNN_FOLDER in constants, which "
                         "need not be the surrogate --recon_dir's recon was trained "
                         "against, and the surrogate-side U would then describe a "
                         "different model than the one under test.")
    ap.add_argument("--layout", type=str, default=None,
                    help="Path to the OPTIMIZED layout_best.pt to score against the "
                         "baseline (default: the constants full_corpus_grid layout).")
    ap.add_argument("--corpus", type=str, default=None,
                    help="Score a DIFFERENT shower corpus than the constants one, "
                         "e.g. the held-out set. Its `<corpus>_positions.pt` sidecar "
                         "is used, and primaries are re-encoded from the corpus "
                         "metadata since primary.pt is row-aligned with the training "
                         "corpus only.")
    ap.add_argument("--objective", type=str, default="composite",
                    choices=("composite", "detection"),
                    help="Which utility scores BOTH the surrogate and the kernel "
                         "labels. 'composite' (default) is the recon-aware utility "
                         "and is byte-identical to previous behaviour. 'detection' "
                         "is the aperture-style objective, needed to check a layout "
                         "optimized with 04 --objective detection against ground "
                         "truth, since the surrogate over-reports detector firing.")
    ap.add_argument("--detect_n", type=float, default=None,
                    help="detection objective: minimum detector count (None uses "
                         "the opt_core default).")
    ap.add_argument("--detect_e", type=float, default=None,
                    help="detection objective: per-detector count threshold (None "
                         "uses the opt_core default).")
    args = ap.parse_args()
    if args.layout:
        global LAYOUT_PATH
        LAYOUT_PATH = args.layout

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus_path, _ = _corpus_paths(args.corpus)

    print("=" * 72)
    print("true-utility evaluator — kernel vs surrogate on the SAME recon + weights")
    print("=" * 72)
    print(f"device      : {device}")
    print(f"corpus      : {corpus_path}"
          f"{'  (override)' if args.corpus else '  (held-out, unseen by Steps 1-4)'}")
    print(f"layout(opt) : {LAYOUT_PATH}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(device)
    elec, muon, B, n_pairs = load_events(args.n_events, device, corpus_override=args.corpus)
    print(f"events      : {B} of {n_pairs} pairs")
    kernel_fnn = KernelDualLabels(elec, muon, surface, device,
                                  chunk=args.kernel_chunk)
    print(f"kernel chunk: {kernel_fnn.chunk} events/call")

    fnn, recon = load_models(device,
                             fnn_folder=args.fnn_folder or FNN_FOLDER,
                             recon_dir=args.recon_dir or RECON_FOLDER + "_deepsets")
    # Always re-encoded from the corpus being scored: primary.pt only ever lines up
    # with the TRAINING corpus, and the default corpus here is the held-out one.
    prim = build_primaries(corpus_path, B, mountain).to(device)

    e_o, n_o = load_layout(mountain)
    if args.grid_layout:
        e_g, n_g = grid_layout(mountain)
    else:
        e_g, n_g = center_layout(mountain)
    _sc = dict(objective=args.objective, detect_n=args.detect_n, detect_e=args.detect_e)
    gs, gt, _, _ = score(e_g, n_g, prim, fnn, kernel_fnn, recon, device, **_sc)
    os_, ot, ops, opt_ = score(e_o, n_o, prim, fnn, kernel_fnn, recon, device, **_sc)

    # The baseline label follows --grid-layout. It used to read "grid"
    # unconditionally while the DEFAULT baseline is center, so a default run
    # reported an optimized-vs-center comparison labelled optimized-vs-grid.
    # The two are far apart -- on the phase3 models grid scores true-U 22.1
    # against center's 7.6 -- so the mislabel inflated the reported gain by an
    # order of magnitude while looking entirely plausible.
    base_name = "grid" if args.grid_layout else "center"
    print()
    print(f"{base_name.upper()} LAYOUT (baseline) vs OPTIMIZED LAYOUT")
    print("                  surrogate-U     true-U")
    print(f"  baseline {base_name:6s} {gs:11.4f}   {gt:11.4f}")
    print(f"  optimized       {os_:11.4f}   {ot:11.4f}")
    print()
    d_surr, d_true = os_ - gs, ot - gt
    print(f"  ΔU surrogate (opt - {base_name}) : {d_surr:+.4f}")
    print(f"  ΔU true      (opt - {base_name}) : {d_true:+.4f}")
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
    # Key set depends on the active objective: composite reports the three
    # reward terms plus the aperture term, detection reports its own
    # diagnostics. Iterate over what the objective actually returned so this
    # never reads a key the active path does not produce.
    for k in ops:
        s_v, t_v = ops[k], opt_[k]
        s_v = float(s_v.item()) if torch.is_tensor(s_v) else float(s_v)
        t_v = float(t_v.item()) if torch.is_tensor(t_v) else float(t_v)
        print(f"    {k:11s}  {s_v:+9.4f} | {t_v:+9.4f}")


if __name__ == "__main__":
    main()
