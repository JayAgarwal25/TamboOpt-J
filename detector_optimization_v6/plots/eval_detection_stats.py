"""Detection diagnostic: how many detectors light up, kernel vs surrogate.

This measures DETECTION, not angular reconstruction quality. It is a narrower
question than eval_recon_resolution.py: on the SAME held-out events and the
SAME layout, how many detectors cross the firing threshold under the
ground-truth plane-aware KERNEL (run directly on the real shower point
clouds) versus under the learned dual-species SURROGATE that stands in for
the kernel everywhere else in the pipeline (stage 3 recon training, stage 4
layout optimization)?

A detector is "lit" if its predicted particle count exceeds --threshold
(default 1.0, the same LAYOUT_THRESHOLD opt_core.py uses to define
reconstructability - see the comment there for why 1.0 and not the old 5e-2).
Both sources are compared in RAW count units, not log-compressed ones:

  kernel     N_tot = E_e + E_mu, from compute_labels_batch run separately on
             the electron and muon clouds. compute_labels_batch returns the
             kernel's raw local_intensity counts directly, no log1p.
  surrogate  fnn(primary, xy) is the frozen dual-species surrogate; it returns
             the combined event response in log1p space, channel 0 being
             log1p(N_tot) (see combine_species_outputs in dual_surrogate.py).
             torch.expm1 on that channel recovers the same raw count units as
             the kernel, so the two sources are threshold-compared on equal
             footing.

Reports, in order:
  a. per-event n_lit distribution (mean, p10/p25/p50/p75/p90, frac n_lit==0),
     for each source separately.
  b. the per-event difference n_lit_surrogate - n_lit_kernel (median, p10,
     p90), so the surrogate's tendency to over- or under-count is explicit.
  c. per-detector confusion between the two sources, aggregated over every
     event and detector: both-lit, kernel-only, surrogate-only, neither
     counts and rates, plus precision/recall/F1 of the surrogate treating the
     kernel as truth. This is the direct answer to "which detectors light up
     and which do not".
  d. detection efficiency vs primary energy: 8 bins spaced uniformly across
     the trained log10(E) range, each reporting n_events, mean n_lit
     (kernel), and the fraction of events with at least N_DET detectors lit,
     for N_DET in (1, 5, 10, 20). The classic aperture-style curve.
  e. the same table vs decay vertex distance from the array centre (8
     quantile bins in the horizontal distance sqrt(rel_E^2 + rel_N^2)).
  f. one compact summary line giving the overall kernel detection fractions
     at each N_DET.

Everything is chunked over events like eval_recon_resolution.py and
eval_true_utility.py, so this scales to tens of thousands of held-out events
without holding (events, points, detectors) kernel intermediates or
(events, detectors) surrogate activations all at once. Chunking here is only
summation and concatenation over the event axis, both of which are exact, not
an approximation of the whole-batch result.

    python plots/eval_detection_stats.py --n-events 8000 --layout grid
    python plots/eval_detection_stats.py --recon_dir <dir> --layout /path/to/layout_best.pt
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6  # noqa: F401 - sys.path injection for v3 + v4
from modules_v6.opt_core import load_models
from modules_v6.fnn_surrogate_ne import compute_labels_batch
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, LOG_E_MIN, LOG_E_MAX,
)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_etu", os.path.join(_HERE, "plots", "eval_true_utility.py"))
_etu = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_etu)

N_DET_THRESHOLDS = (1, 5, 10, 20)
N_ENERGY_BINS = 8
N_VERTEX_BINS = 8


def _percentiles(x):
    """(mean, [p10, p25, p50, p75, p90]) of a 1-D numpy array."""
    p = np.percentile(x, [10, 25, 50, 75, 90])
    return float(x.mean()), p


def _report_distribution(name, n_lit):
    mean, p = _percentiles(n_lit)
    frac_zero = float((n_lit == 0).mean())
    print(f"  {name:12s} mean={mean:6.2f}  "
          f"p10={p[0]:5.1f} p25={p[1]:5.1f} p50={p[2]:5.1f} p75={p[3]:5.1f} p90={p[4]:5.1f}  "
          f"frac(n_lit==0)={frac_zero:.4f}")


def _report_diff(diff):
    med = float(np.median(diff))
    p10, p90 = np.percentile(diff, [10, 90])
    print(f"  n_lit_surrogate - n_lit_kernel : median={med:+.2f}  p10={p10:+.2f}  p90={p90:+.2f}")


def _report_confusion(both, konly, sonly, neither):
    total = both + konly + sonly + neither
    print(f"  both-lit       : {both:12d}  (rate {both / total:.4f})")
    print(f"  kernel-only    : {konly:12d}  (rate {konly / total:.4f})")
    print(f"  surrogate-only : {sonly:12d}  (rate {sonly / total:.4f})")
    print(f"  neither        : {neither:12d}  (rate {neither / total:.4f})")
    precision = both / max(both + sonly, 1)
    recall = both / max(both + konly, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    print(f"  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}  "
          f"(surrogate as prediction, kernel as truth)")


def _report_bins(label, bin_values, edges, n_lit_kernel, n_thresholds):
    """Per-bin n_events, mean kernel n_lit, and fraction detected at each N_DET.

    `edges` has N_BINS+1 entries; digitize on the interior edges alone gives
    exactly N_BINS bins with the last one closed on the right, so the maximum
    value in `bin_values` always lands in the final bin.
    """
    print(f"\n  detection efficiency vs {label}:")
    cols = "  ".join(f"frac(N>={n})" for n in n_thresholds)
    print(f"    {'bin':>24s} {'n_events':>9s} {'mean n_lit':>11s}   {cols}")
    idx = np.digitize(bin_values, edges[1:-1], right=False)
    n_bins = len(edges) - 1
    for b in range(n_bins):
        mask = idx == b
        ne = int(mask.sum())
        rng = f"[{edges[b]:8.3f}, {edges[b + 1]:8.3f})"
        if ne == 0:
            print(f"    {rng:>24s} {ne:9d}  {'--':>11s}")
            continue
        mean_nlit = float(n_lit_kernel[mask].mean())
        fracs = "  ".join(f"{float((n_lit_kernel[mask] >= n).mean()):11.4f}" for n in n_thresholds)
        print(f"    {rng:>24s} {ne:9d}  {mean_nlit:11.2f}   {fracs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-events", type=int, default=4000)
    ap.add_argument("--layout", type=str, default="grid",
                    help="'grid', 'center', or path to a layout_best.pt")
    ap.add_argument("--chunk", type=int, default=64,
                    help="events per forward chunk; bounds peak memory so --n-events "
                         "can be large. Results are identical to a single shot.")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="raw-count threshold above which a detector is 'lit' "
                         "(matches opt_core.LAYOUT_THRESHOLD)")
    ap.add_argument("--corpus", type=str, default=None,
                    help="Score a separate held-out shower corpus (default: the "
                         "constants held-out corpus, unseen by every upstream stage).")
    ap.add_argument("--recon_dir", type=str, default=None,
                    help="passed to opt_core.load_models; only the dual surrogate "
                         "it returns is used here, the recon is not called.")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mtn).to(dev)

    corpus_path, _ = _etu._corpus_paths(args.corpus)
    elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev, corpus_override=args.corpus)
    prim = _etu.build_primaries(corpus_path, B, mtn).to(dev)

    fnn, _recon = load_models(dev, recon_dir=args.recon_dir)

    if args.layout == "grid":
        e_l, n_l = _etu.grid_layout(mtn)
    elif args.layout == "center":
        e_l, n_l = _etu.center_layout(mtn)
    else:
        _etu.LAYOUT_PATH = args.layout
        e_l, n_l = _etu.load_layout(mtn)
    e_l, n_l = e_l.to(dev), n_l.to(dev)

    print("=" * 72)
    print("detection stats - kernel vs surrogate, same events + layout")
    print("=" * 72)
    print(f"device      : {dev}")
    print(f"corpus      : {corpus_path}"
          f"{'  (override)' if args.corpus else '  (held-out, unseen by Steps 1-4)'}")
    print(f"layout      : {args.layout}")
    print(f"events      : {B} of {n_pairs} pairs")
    print(f"threshold   : {args.threshold:g} raw counts")
    print(f"chunk       : {args.chunk} events/call")

    n_lit_kernel = np.empty(B, dtype=np.int64)
    n_lit_surrogate = np.empty(B, dtype=np.int64)
    both = konly = sonly = neither = 0

    with torch.no_grad():
        for lo in range(0, B, args.chunk):
            hi = min(lo + args.chunk, B)
            elec_c = elec[lo:hi].to(dev, non_blocking=True)
            muon_c = muon[lo:hi].to(dev, non_blocking=True)
            E_e, _T_e = compute_labels_batch(elec_c, e_l, n_l, surf)
            E_mu, _T_mu = compute_labels_batch(muon_c, e_l, n_l, surf)
            counts_kernel = E_e + E_mu                             # (hi-lo, n_det) raw

            xy = torch.stack([e_l, n_l], -1).unsqueeze(0).expand(hi - lo, -1, -1)
            pred_ET = fnn(prim[lo:hi], xy)                          # log1p space
            counts_surrogate = torch.expm1(pred_ET[..., 0]).clamp(min=0.0)

            lit_k = counts_kernel > args.threshold
            lit_s = counts_surrogate > args.threshold

            n_lit_kernel[lo:hi] = lit_k.sum(dim=1).cpu().numpy()
            n_lit_surrogate[lo:hi] = lit_s.sum(dim=1).cpu().numpy()

            both += int((lit_k & lit_s).sum().item())
            konly += int((lit_k & ~lit_s).sum().item())
            sonly += int((~lit_k & lit_s).sum().item())
            neither += int((~lit_k & ~lit_s).sum().item())

            del elec_c, muon_c, E_e, E_mu, counts_kernel
            del xy, pred_ET, counts_surrogate, lit_k, lit_s

    print("\n  (a) per-event n_lit distribution:")
    _report_distribution("kernel", n_lit_kernel)
    _report_distribution("surrogate", n_lit_surrogate)

    print("\n  (b) per-event over/under-count (surrogate - kernel):")
    _report_diff(n_lit_surrogate - n_lit_kernel)

    print("\n  (c) per-detector confusion (aggregated over all events x detectors):")
    _report_confusion(both, konly, sonly, neither)

    log_e_norm = prim[:, 3].cpu().numpy()
    log10E = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    e_edges = np.linspace(LOG_E_MIN, LOG_E_MAX, N_ENERGY_BINS + 1)
    _report_bins("log10(E/GeV)  [uniform bins]", log10E, e_edges, n_lit_kernel, N_DET_THRESHOLDS)

    rel_E = prim[:, 5].cpu().numpy()
    rel_N = prim[:, 6].cpu().numpy()
    horiz_dist = np.sqrt(rel_E ** 2 + rel_N ** 2)
    v_edges = np.quantile(horiz_dist, np.linspace(0.0, 1.0, N_VERTEX_BINS + 1))
    _report_bins("vertex horizontal distance [m]  [quantile bins]",
                horiz_dist, v_edges, n_lit_kernel, N_DET_THRESHOLDS)

    print("\n  (f) overall kernel detection fractions:")
    frac_str = "  ".join(f"N>={n}: {float((n_lit_kernel >= n).mean()):.4f}" for n in N_DET_THRESHOLDS)
    print(f"    {frac_str}")


if __name__ == "__main__":
    main()
