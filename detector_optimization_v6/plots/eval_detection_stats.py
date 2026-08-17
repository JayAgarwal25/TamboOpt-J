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
  b2. PARTICLES per event, the total counts summed over all detectors, for each
     source plus their ratio. Detector counts alone are a thresholded view and
     discard magnitude, so this says whether the two sources agree on how much
     signal an event deposits, not merely on how many slots cross the cut.
  b3. where the surrogate's excess lands on slots the kernel leaves DARK, split
     by whether it stays under the firing threshold or crosses it. This
     separates two failure modes that produce identical detector counts: a
     small surplus everywhere that tips near-threshold slots over, versus
     invented bright detectors. Only the ABOVE-threshold share causes false
     detections.
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

With --plot_dir, the same arrays each table above is printed from are also
drawn into PNG figures (n_lit histogram, n_lit scatter density, confusion
matrix, efficiency vs energy, efficiency vs vertex distance) written to that
directory. Plotting never recomputes anything and a plotting failure never
drops the printed tables.

    python plots/eval_detection_stats.py --n-events 8000 --layout grid
    python plots/eval_detection_stats.py --recon_dir <dir> --layout /path/to/layout_best.pt \
        --plot_dir /path/to/figures
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
    return mean, p, frac_zero


def _report_diff(diff):
    med = float(np.median(diff))
    p10, p90 = np.percentile(diff, [10, 90])
    print(f"  n_lit_surrogate - n_lit_kernel : median={med:+.2f}  p10={p10:+.2f}  p90={p90:+.2f}")
    return med, float(p10), float(p90)


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
    return precision, recall, f1


def _report_bins(label, bin_values, edges, n_lit_kernel, n_thresholds):
    """Per-bin n_events, mean kernel n_lit, and fraction detected at each N_DET.

    `edges` has N_BINS+1 entries; digitize on the interior edges alone gives
    exactly N_BINS bins with the last one closed on the right, so the maximum
    value in `bin_values` always lands in the final bin.

    Returns a dict of the per-bin arrays the printed table above is built
    from (`centers`, `edges`, `n_events`, `mean_nlit`, `fracs` shaped
    (n_bins, len(n_thresholds))), so a figure drawn from it can never
    disagree with the table. Empty bins carry NaN in `mean_nlit`/`fracs`,
    matching the printed '--'.
    """
    print(f"\n  detection efficiency vs {label}:")
    cols = "  ".join(f"frac(N>={n})" for n in n_thresholds)
    print(f"    {'bin':>24s} {'n_events':>9s} {'mean n_lit':>11s}   {cols}")
    idx = np.digitize(bin_values, edges[1:-1], right=False)
    n_bins = len(edges) - 1
    n_events = np.zeros(n_bins, dtype=np.int64)
    mean_nlit = np.full(n_bins, np.nan)
    fracs = np.full((n_bins, len(n_thresholds)), np.nan)
    for b in range(n_bins):
        mask = idx == b
        ne = int(mask.sum())
        n_events[b] = ne
        rng = f"[{edges[b]:8.3f}, {edges[b + 1]:8.3f})"
        if ne == 0:
            print(f"    {rng:>24s} {ne:9d}  {'--':>11s}")
            continue
        mean_nlit[b] = float(n_lit_kernel[mask].mean())
        for t, n in enumerate(n_thresholds):
            fracs[b, t] = float((n_lit_kernel[mask] >= n).mean())
        fracs_str = "  ".join(f"{fracs[b, t]:11.4f}" for t in range(len(n_thresholds)))
        print(f"    {rng:>24s} {ne:9d}  {mean_nlit[b]:11.2f}   {fracs_str}")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return dict(centers=centers, edges=edges, n_events=n_events,
                mean_nlit=mean_nlit, fracs=fracs)


# ── Figures ──────────────────────────────────────────────────────────────────
# Every figure below is built ONLY from the arrays/summary values the report
# functions above already returned, never from a recomputation, so a figure
# can never disagree with the table printed above it. Each figure imports
# matplotlib and saves itself independently, guarded by its own try/except:
# a missing matplotlib, a bad --plot_dir, or a single bad figure only prints
# a one-line skip message and never drops the tables already printed.
#
# Kernel and surrogate always share the same two colors across every figure
# (C0 blue for kernel, C1 orange for surrogate), plus a second encoding
# (fill vs dashed step, solid vs dashed line) so the pair stays readable in
# greyscale and to a red-green colorblind reader, not on hue alone.

_THRESH_STYLE = [("C0", "o"), ("C1", "s"), ("C2", "^"), ("C3", "D")]


def _subtitle(layout, n_events, threshold):
    return f"layout={layout}  events={n_events}  threshold={threshold:g} raw counts"


def _plot_nlit_hist(n_lit_kernel, n_lit_surrogate, n_det, med_kernel, med_surrogate,
                     subtitle, path):
    """Per-event lit-detector count, kernel vs surrogate, same integer bins.

    Kernel is drawn as a filled step (so its shape reads at a glance) with
    surrogate as a dashed outline step on top, so the two histograms stay
    separable even where they overlap. Both medians are marked with a
    vertical line in the same color/style as their histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] n_lit histogram skipped ({exc!r})")
        return
    try:
        bins = np.arange(0, n_det + 2) - 0.5   # integer-centered bins, 0..n_det
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(n_lit_kernel, bins=bins, histtype="stepfilled", color="C0",
                alpha=0.35, zorder=2, label=f"kernel (median={med_kernel:.1f})")
        ax.hist(n_lit_kernel, bins=bins, histtype="step", color="C0",
                linewidth=1.4, zorder=3)
        ax.hist(n_lit_surrogate, bins=bins, histtype="step", color="C1",
                linewidth=1.8, linestyle="--", zorder=4,
                label=f"surrogate (median={med_surrogate:.1f})")
        ax.axvline(med_kernel, color="C0", linewidth=1.1, alpha=0.9, zorder=5)
        ax.axvline(med_surrogate, color="C1", linestyle="--", linewidth=1.1,
                   alpha=0.9, zorder=5)
        ax.set_xlim(0, n_det)
        ax.set_xlabel("n_lit (detectors above threshold, per event)")
        ax.set_ylabel("event count")
        ax.set_title(f"per-event lit-detector count, kernel vs surrogate\n{subtitle}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] n_lit histogram skipped ({exc!r})")


def _plot_nlit_scatter(n_lit_kernel, n_lit_surrogate, n_det, subtitle, path):
    """Per-event surrogate n_lit against kernel n_lit, as a log-scaled 2D
    density (hexbin) with the y = x identity line on top.

    This is the per-event companion to the aggregate histogram: two
    distributions can match in aggregate while individual events still
    disagree in both directions, and that only shows up here."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] n_lit scatter skipped ({exc!r})")
        return
    try:
        fig, ax = plt.subplots(figsize=(7, 7))
        hb = ax.hexbin(n_lit_kernel, n_lit_surrogate, gridsize=max(n_det // 4, 10),
                        extent=(0, n_det, 0, n_det), bins="log", cmap="magma",
                        mincnt=1)
        # White-then-black dashed identity line so it stays visible over both
        # the dark and light ends of the density colormap.
        ax.plot([0, n_det], [0, n_det], color="white", linewidth=1.8, zorder=3)
        ax.plot([0, n_det], [0, n_det], color="black", linewidth=0.9,
                linestyle="--", zorder=4, label="y = x (perfect agreement)")
        ax.set_xlim(0, n_det)
        ax.set_ylim(0, n_det)
        ax.set_aspect("equal")
        ax.set_xlabel("n_lit, kernel (per event)")
        ax.set_ylabel("n_lit, surrogate (per event)")
        ax.set_title(f"per-event detector count, surrogate vs kernel\n{subtitle}")
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("event count (log color scale)")
        ax.legend(fontsize=9, loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] n_lit scatter skipped ({exc!r})")


def _plot_confusion(both, konly, sonly, neither, precision, recall, f1, subtitle, path):
    """Per-detector confusion between the two sources as a 2x2 matrix.

    Rows = kernel [dark, lit], columns = surrogate [dark, lit], the same
    both/konly/sonly/neither counts printed in report (c). Cell shading is a
    single sequential hue (rate), and every cell is annotated with both the
    raw count and the rate so the numbers never depend on reading the
    colorbar precisely."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] confusion matrix skipped ({exc!r})")
        return
    try:
        counts = np.array([[neither, sonly],
                            [konly, both]], dtype=float)
        total = counts.sum()
        rates = counts / total

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(rates, cmap="Blues", vmin=0.0, vmax=float(rates.max()))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["dark", "lit"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["dark", "lit"])
        ax.set_xlabel("surrogate")
        ax.set_ylabel("kernel")
        for i in range(2):
            for j in range(2):
                shade = "white" if rates[i, j] > 0.5 * rates.max() else "black"
                ax.text(j, i, f"{int(counts[i, j]):,}\n({rates[i, j]:.4f})",
                        ha="center", va="center", color=shade, fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("rate (fraction of all detector-events)")
        ax.set_title("per-detector confusion, surrogate vs kernel\n"
                     f"{subtitle}\n"
                     f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] confusion matrix skipped ({exc!r})")


def _plot_efficiency(bins, n_thresholds, xlabel, title, subtitle, path,
                      log_x=False, show_binwidth=False):
    """Detection efficiency vs a binned quantity, one line per N_DET
    threshold, plus mean kernel n_lit on a secondary axis.

    `bins` is exactly the dict `_report_bins` returned for the same table,
    so the lines here reproduce the printed fractions rather than
    recomputing them. Empty bins carry NaN and are left as gaps.

    Bins are drawn as STEPS across their true edges. That carries the bin
    width intrinsically, which is what `show_binwidth` used to say with
    horizontal error bars -- but four overlapping sets of them, on quantile
    bins whose widths differ by two orders of magnitude, buried the curves
    they were annotating. The parameter is kept for call compatibility and
    no longer changes the rendering. `log_x` is for a distance/energy-like
    quantity spanning orders of magnitude.

    Mean n_lit and the per-bin event count are sample properties, not
    results, so they go in their own short panel underneath rather than on a
    twin axis crossing the efficiency curves. Sharing the x-axis keeps them
    readable against the same bins."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] {title} skipped ({exc!r})")
        return
    try:
        centers = bins["centers"]
        edges = bins["edges"]
        fracs = bins["fracs"]
        mean_nlit = bins["mean_nlit"]
        n_events = bins["n_events"]

        def _step(axis, vals, **kw):
            """Horizontal segment per bin, across its real edges."""
            v = np.asarray(vals, dtype=float)
            return axis.step(edges, np.concatenate([v, v[-1:]]),
                             where="post", **kw)

        def _band(axis, vals, err, **kw):
            """Shaded +/-err ribbon following the same steps."""
            v, e = np.asarray(vals, float), np.asarray(err, float)
            lo = np.concatenate([v - e, (v - e)[-1:]])
            hi = np.concatenate([v + e, (v + e)[-1:]])
            return axis.fill_between(edges, lo, hi, step="post", **kw)

        fig, (ax, axb) = plt.subplots(
            2, 1, figsize=(9, 6.4), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.10))

        with np.errstate(invalid="ignore", divide="ignore"):
            n_safe = np.where(n_events > 0, n_events, np.nan).astype(float)

        for k, n in enumerate(n_thresholds):
            color, marker = _THRESH_STYLE[k % len(_THRESH_STYLE)]
            p = fracs[:, k]
            # Binomial standard error on the per-bin fraction. Without it there
            # is no way to tell a real feature from a 1000-event wiggle, and
            # these curves are read for their shape.
            with np.errstate(invalid="ignore"):
                err = np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / n_safe)
            _band(ax, p, err, color=color, alpha=0.18, linewidth=0)
            _step(ax, p, color=color, linewidth=1.7, alpha=0.95,
                  label=f"N_DET >= {n}")
            # Markers at bin centres only, so the steps stay the visual
            # carrier and the points just aid reading across bins.
            ax.plot(centers, p, color=color, marker=marker,
                    markersize=4, linestyle="none", alpha=0.95)

        # Anchoring a fraction at 0 is the usual default, but these panels are
        # read for the SHAPE of the curve, and half an empty axis compresses
        # the structure that carries the result. The floor is set below the
        # lowest point instead, with the axis labelled and ticked so nothing
        # is hidden.
        finite = fracs[np.isfinite(fracs)]
        lo = float(np.min(finite)) if finite.size else 0.0
        ax.set_ylim(max(0.0, lo - 0.10), 1.0)
        ax.set_ylabel("fraction of events detected (kernel)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
        ax.set_title(f"{title}\n{subtitle}")

        _step(axb, mean_nlit, color="black", linewidth=1.4)
        axb.plot(centers, mean_nlit, color="black", marker="x", markersize=5,
                 linestyle="none")
        axb.set_ylabel("mean n_lit", fontsize=9)
        axb.set_xlabel(xlabel)
        axb.grid(alpha=0.25)

        # Quantile binning makes events/bin constant by construction, so drawing
        # it would be pure ink. Only show it when the bins actually differ.
        spread = float(np.ptp(n_events)) / max(float(np.mean(n_events)), 1.0)
        if spread > 0.05:
            axc = axb.twinx()
            _step(axc, n_events, color="gray", linewidth=1.0, alpha=0.55)
            axc.set_ylabel("events/bin", color="gray", fontsize=8)
            axc.tick_params(axis="y", labelcolor="gray", labelsize=7)
            axc.set_ylim(bottom=0)
        else:
            axb.text(0.99, 0.06, f"{int(np.median(n_events))} events/bin",
                     transform=axb.transAxes, ha="right", fontsize=7,
                     color="gray")

        if log_x:
            axb.set_xscale("log")
            # Decade-only ticks leave the 1-6 km region, where the structure is,
            # unlabelled. Add 2x/5x subdivisions in plain numbers.
            from matplotlib.ticker import LogLocator, FuncFormatter
            axb.xaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
            axb.xaxis.set_minor_locator(plt.NullLocator())
            axb.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{v:g}" if v < 1000 else f"{v / 1000:g}k"))
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] {title} skipped ({exc!r})")


def _plot_particle_totals(tot_kernel, tot_surrogate, subtitle, path):
    """Per-event total particles, kernel vs surrogate, on log axes.

    The (b2) table quotes deciles and a mean, and the mean is a poor summary
    of a quantity spanning five orders of magnitude -- worse, it hides a SIGN
    FLIP. The surrogate predicts ~25x too MUCH on the dimmest events and ~17x
    too LITTLE on the brightest, so a single "N times less" number is wrong in
    one direction for half the corpus. This draws the whole thing:

      left   joint distribution with y=x and the median surrogate response in
             bins of kernel truth. A slope below 1 IS the compression, and the
             point where the median crosses y=x is where over-prediction turns
             into under-prediction.
      right  the two marginals, where the compressed dynamic range is the
             visible difference in width.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] particle totals skipped ({exc!r})")
        return
    try:
        # Totals can be 0 for a blind event; floor them so log is defined and
        # say so on the axis rather than silently dropping those events.
        floor = 1e-1
        lk = np.log10(np.maximum(tot_kernel, floor))
        ls = np.log10(np.maximum(tot_surrogate, floor))
        lo = float(min(lk.min(), ls.min()))
        hi = float(max(lk.max(), ls.max()))

        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))

        axA.hexbin(lk, ls, gridsize=45, bins="log", cmap="viridis", mincnt=1)
        axA.plot([lo, hi], [lo, hi], "r--", linewidth=1.3, label="y = x")

        edges = np.linspace(lo, hi, 18)
        idx = np.digitize(lk, edges[1:-1])
        xs, ys = [], []
        for i in range(len(edges) - 1):
            m = idx == i
            if int(m.sum()) >= 25:
                xs.append(float(np.median(lk[m])))
                ys.append(float(np.median(ls[m])))
        xs, ys = np.asarray(xs), np.asarray(ys)
        if xs.size:
            axA.plot(xs, ys, "k-o", markersize=4, linewidth=1.6,
                     label="median surrogate | kernel")
            # Where the median response crosses truth: below it the surrogate
            # invents signal, above it the surrogate loses signal.
            d = ys - xs
            sign = np.where(d >= 0)[0]
            if sign.size and sign[-1] + 1 < xs.size:
                j = sign[-1]
                t = d[j] / (d[j] - d[j + 1])
                xc = xs[j] + t * (xs[j + 1] - xs[j])
                axA.axvline(xc, color="0.35", linestyle=":", linewidth=1.2)
                axA.text(0.97, 0.06,
                         f"invents signal below {10 ** xc:,.0f} particles,\n"
                         f"loses it above",
                         transform=axA.transAxes, ha="right", va="bottom",
                         fontsize=8.5, color="0.25",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                   ec="0.75", alpha=0.85))

        # The pile-up at the left edge is real events with zero total, held at
        # the floor so they stay visible. Say so, because that column is the
        # blind-event population and it is the most striking thing on the plot.
        n_floor = int((tot_kernel <= floor).sum())
        if n_floor:
            axA.text(0.02, 0.02,
                     f"left column: {n_floor} events with zero kernel signal",
                     transform=axA.transAxes, fontsize=7.5, color="0.35")
        axA.set_xlabel(f"log10 particles per event, kernel  (floored at {floor:g})")
        axA.set_ylabel("log10 particles per event, surrogate")
        axA.set_title("per event, surrogate vs kernel")
        axA.legend(fontsize=8, loc="upper left")
        axA.grid(alpha=0.25)

        bins = np.linspace(lo, hi, 60)
        axB.hist(lk, bins=bins, alpha=0.45, color="#4C78A8",
                 label=f"kernel (median {np.median(tot_kernel):,.0f})")
        axB.hist(ls, bins=bins, histtype="step", linewidth=1.8, color="#F58518",
                 label=f"surrogate (median {np.median(tot_surrogate):,.0f})")
        axB.set_xlabel("log10 particles per event")
        axB.set_ylabel("events")
        axB.set_title("dynamic range is compressed")
        axB.legend(fontsize=8)
        axB.grid(alpha=0.25)

        fig.suptitle(f"total particles per event\n{subtitle}", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] particle totals skipped ({exc!r})")


def _make_plots(plot_dir, n_lit_kernel, n_lit_surrogate, n_det,
                 med_kernel, med_surrogate, both, konly, sonly, neither,
                 precision, recall, f1, e_bins, v_bins,
                 layout, n_events, threshold,
                 tot_kernel=None, tot_surrogate=None):
    """Write the five detection-diagnostic figures into `plot_dir`.

    Every array passed in here is one already printed above (n_lit_kernel /
    n_lit_surrogate from the counting loop, the confusion counts, the
    per-bin dicts from `_report_bins`), so nothing is recomputed and no
    figure can disagree with the table it illustrates. `plot_dir` is
    created if missing; a failure to create it or to import matplotlib
    skips all five figures with one message and never touches the tables
    already printed."""
    try:
        os.makedirs(plot_dir, exist_ok=True)
    except Exception as exc:
        print(f"[plot] could not create {plot_dir!r}, all figures skipped ({exc!r})")
        return

    sub = _subtitle(layout, n_events, threshold)
    _plot_nlit_hist(n_lit_kernel, n_lit_surrogate, n_det, med_kernel, med_surrogate,
                     sub, os.path.join(plot_dir, "detection_nlit_hist.png"))
    _plot_nlit_scatter(n_lit_kernel, n_lit_surrogate, n_det, sub,
                        os.path.join(plot_dir, "detection_nlit_scatter.png"))
    _plot_confusion(both, konly, sonly, neither, precision, recall, f1, sub,
                     os.path.join(plot_dir, "detection_confusion.png"))
    if tot_kernel is not None and tot_surrogate is not None:
        _plot_particle_totals(tot_kernel, tot_surrogate, sub,
                               os.path.join(plot_dir, "detection_particle_totals.png"))
    _plot_efficiency(e_bins, N_DET_THRESHOLDS, "log10(E / GeV)",
                      "detection efficiency vs primary energy", sub,
                      os.path.join(plot_dir, "detection_efficiency_energy.png"),
                      log_x=False, show_binwidth=False)
    _plot_efficiency(v_bins, N_DET_THRESHOLDS, "decay vertex horizontal distance [m]",
                      "detection efficiency vs decay vertex distance", sub,
                      os.path.join(plot_dir, "detection_efficiency_distance.png"),
                      log_x=True, show_binwidth=True)


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
    ap.add_argument("--fnn_folder", type=str, default=None,
                    help="directory holding fnn_electron.pt and fnn_muon.pt. THIS is "
                         "the model this script measures, so without it every number "
                         "below describes whatever surrogate FNN_FOLDER in constants "
                         "happens to point at, not the one you meant to evaluate.")
    ap.add_argument("--plot_dir", type=str, default=None,
                    help="if given, also write the detection-diagnostic PNG figures "
                         "here (created if missing), in addition to the printed "
                         "tables below; a plotting failure never drops the tables.")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mtn).to(dev)

    corpus_path, _ = _etu._corpus_paths(args.corpus)
    elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev, corpus_override=args.corpus)
    prim = _etu.build_primaries(corpus_path, B, mtn).to(dev)

    fnn, _recon = load_models(dev, fnn_folder=args.fnn_folder, recon_dir=args.recon_dir)

    if args.layout == "grid":
        e_l, n_l = _etu.grid_layout(mtn)
    elif args.layout == "center":
        e_l, n_l = _etu.center_layout(mtn)
    else:
        _etu.LAYOUT_PATH = args.layout
        e_l, n_l = _etu.load_layout(mtn)
    e_l, n_l = e_l.to(dev), n_l.to(dev)
    n_det = int(e_l.shape[0])

    print("=" * 72)
    print("detection stats - kernel vs surrogate, same events + layout")
    print("=" * 72)
    print(f"device      : {dev}")
    print(f"fnn_folder  : {args.fnn_folder or '(constants default)'}")
    print(f"corpus      : {corpus_path}"
          f"{'  (override)' if args.corpus else '  (held-out, unseen by Steps 1-4)'}")
    print(f"layout      : {args.layout}")
    print(f"events      : {B} of {n_pairs} pairs")
    print(f"threshold   : {args.threshold:g} raw counts")
    print(f"chunk       : {args.chunk} events/call")

    n_lit_kernel = np.empty(B, dtype=np.int64)
    n_lit_surrogate = np.empty(B, dtype=np.int64)
    both = konly = sonly = neither = 0

    # Particle-count accounting. n_lit alone cannot distinguish two very
    # different failure modes with identical detector counts: a surrogate that
    # adds a little signal everywhere and tips near-threshold detectors over,
    # versus one that invents bright detectors where the kernel sees nothing.
    # Totals plus the excess split by where it lands relative to the threshold
    # separate them.
    tot_kernel = np.empty(B, dtype=np.float64)      # particles per event, kernel
    tot_surrogate = np.empty(B, dtype=np.float64)   # particles per event, surrogate
    # Excess particles the surrogate adds on slots the KERNEL leaves dark, split
    # by whether that excess is small enough to sit under the firing threshold.
    excess_dark_below = 0.0     # on dark slots, surrogate still under threshold
    excess_dark_above = 0.0     # on dark slots, surrogate pushed over threshold
    n_dark_slots = 0

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

            # Totals over ALL detectors, the event's full signal, not just the
            # slots that happened to cross the threshold.
            tot_kernel[lo:hi] = counts_kernel.sum(dim=1).double().cpu().numpy()
            tot_surrogate[lo:hi] = counts_surrogate.sum(dim=1).double().cpu().numpy()

            # On kernel-dark slots the kernel contributes ~0, so the surrogate's
            # value there IS the excess. Splitting it at the threshold says
            # whether the surplus is harmless background or the thing driving
            # the false detections.
            dark = counts_kernel <= 0.0
            n_dark_slots += int(dark.sum().item())
            s_dark = counts_surrogate[dark]
            over = s_dark > args.threshold
            excess_dark_above += float(s_dark[over].double().sum().item())
            excess_dark_below += float(s_dark[~over].double().sum().item())

            del elec_c, muon_c, E_e, E_mu, counts_kernel
            del xy, pred_ET, counts_surrogate, lit_k, lit_s, dark, s_dark, over

    print("\n  (a) per-event n_lit distribution:")
    _, p_kernel, _ = _report_distribution("kernel", n_lit_kernel)
    _, p_surrogate, _ = _report_distribution("surrogate", n_lit_surrogate)

    print("\n  (b) per-event over/under-count (surrogate - kernel):")
    _report_diff(n_lit_surrogate - n_lit_kernel)

    print("\n  (b2) particles per event (total counts over all detectors):")
    for nm, tot in (("kernel", tot_kernel), ("surrogate", tot_surrogate)):
        q = np.percentile(tot, [10, 50, 90])
        print(f"  {nm:12s} mean={tot.mean():10.2f}  "
              f"p10={q[0]:9.2f} p50={q[1]:9.2f} p90={q[2]:9.2f}")
    _ratio = tot_surrogate / np.maximum(tot_kernel, 1e-9)
    print(f"  surrogate/kernel total ratio : median={np.median(_ratio):.3f}  "
          f"p10={np.percentile(_ratio, 10):.3f} p90={np.percentile(_ratio, 90):.3f}")
    print(f"  mean excess particles per event (surrogate - kernel) : "
          f"{float((tot_surrogate - tot_kernel).mean()):+.2f}")

    print("\n  (b3) where the surrogate's excess lands on kernel-DARK slots:")
    _tot_excess = excess_dark_above + excess_dark_below
    _den = max(_tot_excess, 1e-9)
    print(f"  dark slots                    : {n_dark_slots}")
    print(f"  excess particles on dark slots: {_tot_excess:.1f} total, "
          f"{_tot_excess / max(n_dark_slots, 1):.4f} per dark slot")
    print(f"    of which BELOW threshold    : {excess_dark_below:12.1f}  "
          f"({100.0 * excess_dark_below / _den:5.1f}%)  harmless background")
    print(f"    of which ABOVE threshold    : {excess_dark_above:12.1f}  "
          f"({100.0 * excess_dark_above / _den:5.1f}%)  drives false detections")

    print("\n  (c) per-detector confusion (aggregated over all events x detectors):")
    precision, recall, f1 = _report_confusion(both, konly, sonly, neither)

    log_e_norm = prim[:, 3].cpu().numpy()
    log10E = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    e_edges = np.linspace(LOG_E_MIN, LOG_E_MAX, N_ENERGY_BINS + 1)
    e_bins = _report_bins("log10(E/GeV)  [uniform bins]", log10E, e_edges,
                          n_lit_kernel, N_DET_THRESHOLDS)

    rel_E = prim[:, 5].cpu().numpy()
    rel_N = prim[:, 6].cpu().numpy()
    horiz_dist = np.sqrt(rel_E ** 2 + rel_N ** 2)
    v_edges = np.quantile(horiz_dist, np.linspace(0.0, 1.0, N_VERTEX_BINS + 1))
    v_bins = _report_bins("vertex horizontal distance [m]  [quantile bins]",
                          horiz_dist, v_edges, n_lit_kernel, N_DET_THRESHOLDS)

    print("\n  (f) overall kernel detection fractions:")
    frac_str = "  ".join(f"N>={n}: {float((n_lit_kernel >= n).mean()):.4f}" for n in N_DET_THRESHOLDS)
    print(f"    {frac_str}")

    if args.plot_dir:
        _make_plots(args.plot_dir, n_lit_kernel, n_lit_surrogate, n_det,
                    p_kernel[2], p_surrogate[2], both, konly, sonly, neither,
                    precision, recall, f1, e_bins, v_bins,
                    args.layout, B, args.threshold,
                    tot_kernel=tot_kernel, tot_surrogate=tot_surrogate)


if __name__ == "__main__":
    main()
