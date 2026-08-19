"""Target vs prediction scatter for the trained FNN and recon nets.

Loads the cached corpus (primary / xy / E / T / strategy_ids) and the two
frozen checkpoints (fnn.pt, recon.pt), evaluates each on its respective
shower-level validation split, and saves scatter plots of target vs
prediction with a 1:1 reference line.

FNN plot        : flattened (E, T) over all detectors in the val split.
Recon plot      : raw primary encoding (dir_x, dir_y, dir_z, log_e_norm) over
                  the val split. The recon runs on FNN-predicted (E, T) rather
                  than ground truth, so the scatter reflects the end-to-end
                  FNN -> recon error.

Artifacts:
    outputs/fnn_target_vs_pred.png
    outputs/recon_target_vs_pred.png

Run from the v6 folder:

    cd TambOpt
    python plots/02_plot_nn_target_vs_pred.py
    python plots/02_plot_nn_target_vs_pred.py --dual   # dual-species surrogate

Dual mode (--dual): the FNN scatter is rendered PER SPECIES — each per-species
DeepSets model (fnn_electron.pt / fnn_muon.pt) is compared against its own
species-filtered corpus subset (split on the Step-1 species_ids sidecar), since
the corpus E/T ground truth is per-species.
The recon scatter uses the combined DualSpeciesSurrogate on the full corpus,
exactly as 03_train_recon.py does. Outputs:
    FNN_FOLDER/fnn_electron_target_vs_pred.png
    FNN_FOLDER/fnn_muon_target_vs_pred.png
    FNN_FOLDER/fnn_<species>_conditional.png    P(pred | target), hit-only
    FNN_FOLDER/fnn_<species>_calibration.png    predicted σ vs realised error
    RECON_FOLDER_deepsets/recon_target_vs_pred.png
    RECON_FOLDER_deepsets/recon_conditional.png     P(pred | target), same
                                                     treatment as the FNN's,
                                                     over the recon's 4 channels

`*_conditional.png` is the one to read for surrogate quality. The joint hexbin in
`*_target_vs_pred.png` is dominated by the dark-detector population (target = 0)
and by the fact that the mean head is fitted to a stochastic target, so it
understates the model; the conditional figure drops the dark samples, normalises
per target-column, and puts the compression on the plot as a slope.

Note the recon folder: 03_train_recon_deepsets.py writes to
RECON_FOLDER + "_deepsets", so that is where --dual reads recon.pt and writes
its scatter. Plain RECON_FOLDER belongs to the older flat-MLP 03_train_recon.py.
"""
import os
import sys
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_DIR = os.path.dirname(_HERE)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

# Font sizes, externalized (same convention as plots/opt_plotting.py's FS_*) so a
# caller can override them before rendering — e.g. 05_paper_figures.py scales
# these up before calling in, to compensate for the large figsize LaTeX shrinks
# back down to a fraction of \textwidth.
FS_PANEL_TITLE      = 11   # _conditional_panel's per-channel title
FS_LEGEND           = 8    # _scatter / _conditional_panel legends
FS_LEGEND_DENSE     = 7    # _calibration_panel legend (denser figure)
FS_SUPTITLE         = 12   # _render_fnn_conditional / _render_fnn_calibration suptitles
FS_SUPTITLE_SCATTER = 13   # _render_fnn_scatter / _render_recon_scatter suptitles

# Figsizes, externalized the same way — 05_paper_figures.py overrides these
# module globals (not the figsize= defaults baked into each function) before
# calling in, so the drawn canvas can shrink or grow for the paper without
# editing the render functions themselves.
FIGSIZE_CONDITIONAL = (13, 5.6)    # _render_fnn_conditional
FIGSIZE_CALIBRATION = (7.5, 10)    # _render_fnn_calibration
FIGSIZE_SCATTER      = (10, 4.8)   # _render_fnn_scatter
FIGSIZE_RECON        = (10, 10.6)  # _render_recon_scatter (2x2 grid)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modules_v6  # noqa: F401 — triggers sys.path injection for v3 + v4
from modules_v6.fnn_surrogate import FNNSurrogate
from modules_v6.constants import (
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
    N_DETECTORS, PRIMARY_DIM,
)
from modules_v6.reconstruction import build_recon_from_ckpt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _savefig_multi(fig, output_path, dpi=130, formats=("png",)):
    """Save `fig` under `output_path`'s basename once per extension in
    `formats` (default: just whatever extension `output_path` already has,
    unchanged behaviour for training-script callers). `05_paper_figures.py`
    passes `formats=("png", "pdf")` so every paper figure ships as a
    print-ready vector PDF alongside the PNG used for quick preview; hexbin
    and imshow content still rasterizes fine inside a PDF (embedded as a
    compressed image XObject, not per-point vector paths), so this doesn't
    bloat file size for the density-heavy figures."""
    base, ext = os.path.splitext(output_path)
    if not formats:
        formats = (ext.lstrip(".") or "png",)
    for fmt in formats:
        path = f"{base}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[save] {path}")

# 03_train_recon_deepsets.py writes to RECON_FOLDER + "_deepsets" (its line 50),
# not RECON_FOLDER — that plain folder only exists for the older flat-MLP
# 03_train_recon.py run.
RECON_DEEPSETS_FOLDER = RECON_FOLDER + "_deepsets"

# Seeds match 02_train_fnn.py and 03_train_recon.py
FNN_VAL_SEED   = 0
RECON_VAL_SEED = 1
VAL_FRAC       = 0.10
BATCH          = 1024

# Mirror the log-T transform applied inside 02_train_fnn.py — the FNN was
# trained with log1p(T * 1e8) as its canonical T target, so the ground-truth
# T tensor must be passed through the same transform before the FNN scatter
# is apples-to-apples.
T_LOG_SCALE = 1.0e8


def shower_level_val_idx(strategy_ids: torch.Tensor,
                         val_frac: float,
                         seed: int) -> torch.Tensor:
    """Reproduce the shower-level val indices used during training."""
    n_pairs  = int(strategy_ids.shape[0])
    n_strat  = int(strategy_ids.max().item() + 1)
    n_showers = n_pairs // n_strat

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_showers, generator=g)
    n_val = max(1, int(round(val_frac * n_showers)))

    is_val = torch.zeros(n_showers, dtype=torch.bool)
    is_val[perm[:n_val]] = True

    all_idx = torch.arange(n_pairs, dtype=torch.long)
    shower_of_pair = all_idx - strategy_ids * n_showers
    val_mask = is_val[shower_of_pair]
    return torch.nonzero(val_mask).squeeze(-1)


def _scatter(ax, x, y, title: str, vmin=None, vmax=None, lo=None, hi=None, legend: bool = True):
    """Density-coloured target-vs-prediction panel.

    Hexbin on a LINEAR colour scale clipped at the p95 occupied bin. Plain
    linear is unusable here — counts span 1 to ~1e7, so the densest cell takes
    the whole ramp and every other bin renders as one flat tone. Clipping the top
    keeps equal count steps as equal colour steps (which LogNorm does not) while
    letting the bulk occupy the range. `mincnt=1` leaves empty bins blank so the
    y = x reference line stays readable.

    `vmin` sets the bottom of the scale. `vmax` is accepted for call-site
    compatibility but does NOT set the ceiling — the p95 clip always wins, so
    the pinned values in FNN_DUAL_VLIM no longer fix the top of the scale."""
    import numpy as np
    from matplotlib.colors import Normalize
    lo = 0.0 if lo is None else lo
    hi = float(max(x.max(), y.max())) if hi is None else hi
    hb = ax.hexbin(x, y, gridsize=80, cmap="viridis",
                   norm=Normalize(vmin=vmin), mincnt=1, extent=(lo, hi, lo, hi))
    counts = np.asarray(hb.get_array())
    cb_label = "count"
    if counts.size:
        hb.set_clim(0.0 if vmin is None else float(vmin),
                    float(np.percentile(counts, 95.0)))
        # Say so on the bar, otherwise the saturated core reads as a real plateau.
        cb_label = "count  (linear, clipped at p95)"
    plt.colorbar(hb, ax=ax, label=cb_label, pad=0.02, fraction=0.046)
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=2.0,
            alpha=0.85, label="y = x")
    ax.set_xlabel("target"); ax.set_ylabel("prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title)
    if legend:
        ax.legend(loc="upper left", fontsize=FS_LEGEND, framealpha=1)


def _calibration_panel(ax, sigma, err, channel: str, legend: bool = True):
    """Predicted uncertainty vs realised error, as a 2σ-vs-|error| density, in
    the z-scored space the network is trained in.

    x = 2σ (the nominal 95% half-width the head predicts for that detector),
    y = |prediction − target|. For a calibrated Gaussian head the cloud sits
    mostly BELOW the y = x diagonal: |err| exceeds 2σ only 4.6% of the time.
    Two binned curves cut through the density — the per-σ-bin median and 95th
    percentile of |err| — against their ideal Gaussian values (0.337·x and
    0.98·x), which is what turns "looks about right" into a readable
    over/under-confidence verdict: a p95 curve above y = x means the head is
    over-confident in that σ range, below means it over-inflates σ.

    σ bins are quantile bins (equal counts), so every marker carries the same
    statistical weight regardless of how skewed the σ distribution is."""
    import numpy as np

    two_sig = 2.0 * sigma
    abs_err = np.abs(err)
    hi = float(max(np.percentile(two_sig, 99.5), np.percentile(abs_err, 99.5)))

    # LINEAR colour, clipped at the p99 occupied bin. Plain linear is unusable
    # here: counts span 1 to ~1e7, so the single densest cell takes the entire
    # ramp and every other bin renders as one flat background tone. Clipping the
    # top keeps a linear scale — equal count steps are equal colour steps, which
    # LogNorm does not give — while letting the bulk of the distribution occupy
    # the range. For the sparse tail structure instead, use norm=LogNorm(vmin=1).
    hb = ax.hexbin(two_sig, abs_err, gridsize=80, cmap="viridis",
                   mincnt=1, extent=(0.0, hi, 0.0, hi))
    counts = np.asarray(hb.get_array())
    cb_label = "count"
    if counts.size:
        vmax = float(np.percentile(counts, 95.0))
        hb.set_clim(0.0, vmax)
        # Say so on the bar, otherwise the saturated core reads as a real plateau.
        cb_label = f"count  (linear, clipped at p95)"
    plt.colorbar(hb, ax=ax, label=cb_label, pad=0.02, fraction=0.046)

    # Quantile bins over σ so each point aggregates the same number of samples.
    n_bins = 25
    edges = np.unique(np.percentile(two_sig, np.linspace(0, 100, n_bins + 1)))
    if edges.size >= 3:
        idx = np.clip(np.digitize(two_sig, edges[1:-1]), 0, edges.size - 2)
        centers, med, p95 = [], [], []
        for b in range(edges.size - 1):
            m = idx == b
            if m.sum() < 50:
                continue
            centers.append(np.median(two_sig[m]))
            med.append(np.median(abs_err[m]))
            p95.append(np.percentile(abs_err[m], 95.0))
        ax.plot(centers, p95, color="#d95f02", marker="o", ms=4, lw=2.0,
                label="p95 |err| per σ-bin")
        ax.plot(centers, med, color="#7570b3", marker="s", ms=4, lw=2.0,
                label="median |err| per σ-bin")

    ax.plot([0, hi], [0, hi], color="red", ls="--", lw=2.0, alpha=0.85,
            label="y = x  (ideal p95)")
    ax.plot([0, hi], [0, 0.337 * hi], color="red", ls=":", lw=1.6, alpha=0.7,
            label="0.337·x  (ideal median)")

    cover = float((abs_err <= two_sig).mean())
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("2σ  (predicted, z-scored)")
    ax.set_ylabel("|prediction − target|  (z-scored)")
    # `pad` scaled with the active title fontsize (matplotlib's own default,
    # ~6pt, doesn't scale with fontsize) -- in the stacked 2-panel figure this
    # is what keeps the lower panel's title clear of the upper panel's
    # xlabel; subplot/tight_layout spacing knobs didn't move this at all.
    title_fs = matplotlib.rcParams.get("axes.titlesize", 12)
    if not isinstance(title_fs, (int, float)):
        title_fs = matplotlib.rcParams["font.size"]
    ax.set_title(f"{channel}: 2σ cov. {100 * cover:.1f}% (exp. 95.4%)",
                pad=1.8 * title_fs)
    if legend:
        ax.legend(loc="upper left", fontsize=FS_LEGEND_DENSE, framealpha=1)


# --------------------------------------------------------------------------- #
# Conditional-density figure (fnn_<species>_conditional.png)
#
# Why it exists, in one paragraph. The joint hexbin in *_target_vs_pred.png
# understates the surrogate for two reasons that have nothing to do with model
# quality: a spike of DARK detectors (target exactly 0) owns the colour scale,
# and the mean head is fitted to a stochastic target, so its cloud is *supposed*
# to be narrower than the data. This figure drops the dark samples and normalises
# per target-column, then overlays the mean and ±1σ of the SAME conditional
# (prediction | target) so image and curve agree. Its slope is read against
# Var(pred)/Var(target), not against 1.00 — see `_conditional_panel`.
# --------------------------------------------------------------------------- #

# p16/p84 — the ±1σ interval of a Gaussian, so the drawn band is directly
# comparable to the head's own predicted σ.
SIGMA_QUANTILES = (0.16, 0.84)

# Axis window: p99.9 of the target, not its max. A handful of extreme targets
# would otherwise spend most of the axis on whitespace.
AXIS_PERCENTILE = 99.9

BinnedProfile = namedtuple("BinnedProfile", "bin_center mean p16 p84 count")


def profile_by_bin(key, value, edges, min_count):
    """Group `value` by which `edges` bin its `key` falls into; summarise each bin.

    Returns a `BinnedProfile` whose arrays are already filtered to bins holding
    at least `min_count` samples, so callers can plot them directly.

    Implementation note: `np.bincount` gives the count and both sums in one pass
    each, and accumulates in float64 — which a naive fp32 sum over the ~1e6
    samples a single bin can hold would not. The two quantiles need order, so
    one `lexsort` lays the values out grouped by bin and ascending within each
    group, after which a quantile is a positional offset from the group's start.
    """
    import numpy as np

    n_bins = edges.size - 1
    bin_of = np.clip(np.digitize(key, edges[1:-1]), 0, n_bins - 1)

    count = np.bincount(bin_of, minlength=n_bins)
    divisor = np.maximum(count, 1)                      # empty bins -> 0/1 = 0
    mean = np.bincount(bin_of, weights=value, minlength=n_bins) / divisor

    ordered = value[np.lexsort((value, bin_of))]
    group_start = np.concatenate(([0], np.cumsum(count)[:-1]))

    def quantile(frac):
        offset = (frac * np.maximum(count - 1, 0)).astype(np.intp)
        # Empty bins can point one past the end; they are dropped by `keep`.
        return ordered[np.minimum(group_start + offset, ordered.size - 1)]

    lo_q, hi_q = SIGMA_QUANTILES
    keep = count >= min_count
    bin_center = 0.5 * (edges[:-1] + edges[1:])
    return BinnedProfile(
        bin_center=bin_center[keep],
        mean=mean[keep],
        p16=quantile(lo_q)[keep],
        p84=quantile(hi_q)[keep],
        count=count[keep].astype(np.float64),
    )


def _draw_column_density(ax, target, pred, edges, min_count):
    """Fill the panel with P(prediction | target), each column scaled to its peak.

    Peak-scaling rather than sum-normalising: the conditional is sharp near
    target = 0 and broad at high target, so dividing by the column sum lets one
    column's spike take the whole colour ramp and washes the ridge out
    everywhere else. Scaling to the peak puts every column on the same visual
    footing, which is the point — the SHAPE of the conditional at each target,
    not its height. Columns under `min_count` are left blank rather than
    normalised from a handful of samples.
    """
    import numpy as np

    lo, hi = edges[0], edges[-1]
    # Predictions outside the window are clipped INTO the edge bins so no column
    # loses mass before normalisation. Every statistic elsewhere uses unclipped
    # values, so this affects the picture only.
    joint, _, _ = np.histogram2d(target, np.clip(pred, lo, hi),
                                 bins=(edges, edges))
    populated = joint.sum(axis=1) >= min_count
    density = np.full_like(joint, np.nan)
    density[populated] = joint[populated] / np.maximum(
        joint[populated].max(axis=1, keepdims=True), 1.0)

    # .T because histogram2d indexes [x, y] while imshow wants [row, col].
    im = ax.imshow(density.T, origin="lower", extent=(lo, hi, lo, hi),
                   aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="P(prediction | target)", pad=0.02,
                 fraction=0.046)


def _draw_forward_conditional(ax, profile):
    """Draw mean-prediction-per-target-bin, its ±1σ band, and the y = x line.

    Binned on the TARGET, which is the X axis — so the bin centre is the X
    coordinate and every prediction statistic is a Y coordinate, and the calls
    below read in the natural `(bin_center, statistic)` order. This is the same
    conditioning the column-normalised image behind it uses, so the mean curve
    lands inside the bright ridge instead of cutting across it.
    """
    target_bin = profile.bin_center
    pink = "#ff2d95"

    ax.fill_between(target_bin, profile.p16, profile.p84,
                    color=pink, alpha=0.15, zorder=3)
    ax.plot(target_bin, profile.p16, color=pink, ls=":", lw=1.5, alpha=0.9,
            zorder=4, label="±1σ  (p16 / p84 of prediction)")
    ax.plot(target_bin, profile.p84, color=pink, ls=":", lw=1.5, alpha=0.9,
            zorder=4)
    ax.plot(target_bin, profile.mean, color=pink, marker="s", ms=3.5, lw=2.2,
            zorder=5, label="mean prediction | target")

    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color="red", ls="--", lw=2.0, alpha=0.9,
            zorder=6, label="y = x")


def _conditional_panel(ax, target, pred, channel: str,
                       n_bins=60, min_count=200, legend: bool = True,
                       hit_only: bool = True, lo=None, hi=None):
    """One channel's conditional-density panel. Returns its stats for printing.

    Hit detectors only (`target > 0`) WHEN `hit_only` is set — the default, for
    the FNN's E/T channels, where most detector-samples are exactly dark and
    that population answers a classification question, not a regression one, so
    it is reported as a number instead of a blob. Pass `hit_only=False` for a
    densely-populated target (e.g. the recon's continuous primary channels),
    where every sample is a regression sample and none should be dropped.

    `lo`/`hi` fix the axis window explicitly (recon channels don't share one
    range — dir_z spans [0, 0.5] against dir_x/y/log_e_norm's [0, 1]). Left
    None, the window is `[0, p99.9(target)]`, the FNN's original default.

    Everything on the panel is the FORWARD conditional `prediction | target`:
    the image is normalised per target-column, and the curve, band and slope are
    all binned on the target. One conditioning throughout, so the mean curve sits
    inside the bright ridge rather than cutting across it.

    DO NOT READ THE SLOPE AGAINST 1.00. This direction conditions on a noisy
    realisation of the target while the net is fitted to `E[y | input]`, so
    regression attenuation holds the slope below 1 however good the model is;
    calling the gap to 1.00 "compression" over-accuses the model.

    The reference it IS read against is measured, never claimed. Writing
    `target = f + ε` and `pred = m`, `Cov(m, t) = Cov(m, f)`, so a perfect
    conditional mean `m = f` satisfies `Cov(pred, target) = Var(pred)` exactly —
    at any noise level, with no appeal to the model's own σ. That gives

        expected slope = Var(pred) / Var(target)
        bias ratio     = Cov(pred, target) / Var(pred)     (ideal 1.00)

    both from measured moments alone.

    MEAN, not median: the mean is what the Gaussian NLL fits the head to, and
    the conditional is strongly right-skewed in log1p space near zero.

    Only the slope and its expected value reach the title; the rest goes to
    stdout via the caller.
    """
    import numpy as np

    if hit_only:
        keep = target > 0.0
        target, pred = target[keep], pred[keep]       # rebind: hits only below
        hit_frac = float(keep.mean())
    else:
        hit_frac = 1.0
    if target.size < 10 * n_bins:
        ax.set_title(f"{channel}: too few samples ({target.size})")
        return None

    lo = 0.0 if lo is None else lo
    hi = float(np.percentile(target, AXIS_PERCENTILE)) if hi is None else hi
    edges = np.linspace(lo, hi, n_bins + 1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    _draw_column_density(ax, target, pred, edges, min_count)
    profile = profile_by_bin(key=target, value=pred,
                             edges=edges, min_count=min_count)
    _draw_forward_conditional(ax, profile)

    # # sqrt(count) weights so sparse high-target bins do not drive the fit.
    # slope = float(np.polyfit(profile.bin_center, profile.mean, 1,
    #                          w=np.sqrt(profile.count))[0]) \
    #     if profile.bin_center.size > 1 else float("nan")
    # # Measured moments over every hit sample — no windowing, no binning, and
    # # nothing the model asserts about itself. See the docstring for the identity.
    # t64, p64 = target.astype(np.float64), pred.astype(np.float64)
    # var_t, var_p = float(np.var(t64)), float(np.var(p64))
    # cov = float(np.mean((t64 - t64.mean()) * (p64 - p64.mean())))
    # expected_slope = var_p / max(var_t, 1e-12)
    # bias_ratio = cov / max(var_p, 1e-12)
    # Half the p16-p84 gap: the spread of the model's output at a fixed truth.
    band_sigma = float(np.median(0.5 * (profile.p84 - profile.p16)))

    ax.set_xlabel("target"); ax.set_ylabel("prediction")
    ax.set_title(channel, fontsize=FS_PANEL_TITLE)
    if legend:
        ax.legend(loc="upper left", fontsize=FS_LEGEND, framealpha=0.9)
    return dict(hit_frac=hit_frac, n_hit=int(target.size),
                band_sigma=band_sigma)


Channel = namedtuple("Channel", "tag label target pred")


def _render_fnn_conditional(fnn, primary, xy, E_true, T_true, val_idx,
                            output_path, formats=("png",)):
    """Conditional-density figure, one panel per channel, plus its stdout report.

    Companion to `_render_fnn_scatter`, which shows the same data as a joint
    density dominated by the dark-detector population. Needs no variance head:
    every reference on this figure comes from measured moments (see
    `_conditional_panel`), so it renders identically for a mean-only checkpoint.
    The predicted σ is tested in fnn_*_calibration.png instead."""
    p, x = primary[val_idx], xy[val_idx]
    E_pred, T_pred = (a.flatten().numpy() for a in fnn_predict(fnn, p, x))
    channels = (
        Channel("E", "log1p(E)", E_true[val_idx].flatten().numpy(), E_pred),
        Channel("T", "log1p(T·1e8)", T_true[val_idx].flatten().numpy(), T_pred),
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_CONDITIONAL)
    stats = {ch.tag: _conditional_panel(ax, ch.target, ch.pred, ch.label, legend=False)
             for ax, ch in zip(axes, channels)}

    n_samples = channels[0].target.size
    hit_txt = f"{100 * stats['E']['hit_frac']:.0f}% of " if stats["E"] else ""
    fig.suptitle(f"Conditional density ({hit_txt}{n_samples:,})", fontsize=FS_SUPTITLE)
    handles, labels = next(
        (ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[1]),
        ([], []))
    if labels:
        fig.legend(handles, labels, loc="lower center", fontsize=FS_LEGEND,
                  ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    # rect's bottom=0.1 leaves headroom below the axes' own xlabel ("target")
    # for the legend two lines below it; bbox_to_anchor alone isn't enough --
    # tight_layout has no idea a legend exists outside the axes it's fitting.
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    _savefig_multi(fig, output_path, dpi=130, formats=formats)
    plt.close(fig)

    _print_conditional_stats(stats, channels[0].target, E_pred)


def _print_conditional_stats(stats, E_target, E_pred):
    """Everything computed for the figure that did not earn ink on it."""
    import numpy as np

    # The dark/lit split as the classification it is, at the count >= 1 cut that
    # LAYOUT_THRESHOLD gates the trigger on. E only — a count threshold is
    # meaningless for a time.
    cut = np.log1p(1.0)
    true_lit, pred_lit = E_target >= cut, E_pred >= cut
    print(f"       [E, count>=1] true lit {100 * true_lit.mean():.2f}%  "
          f"pred lit {100 * pred_lit.mean():.2f}%  "
          f"agree {100 * (true_lit == pred_lit).mean():.2f}%  "
          f"recall {100 * (pred_lit & true_lit).sum() / max(true_lit.sum(), 1):.2f}%")

    for tag, st in stats.items():
        if st is None:
            continue
        print(f"       [{tag}] hit {100 * st['hit_frac']:.2f}% "
              f"(n={st['n_hit']:,})  1σ band {st['band_sigma']:.3f}")


@torch.no_grad()
def fnn_predict_sigma(fnn, primary: torch.Tensor, xy: torch.Tensor):
    """Per-detector predicted σ in raw target units, (N, n_det) for E and T.

    Uses the heteroscedastic head's `forward_var` (raw-unit variance), which is
    the un-z-scored counterpart of the logvar the NLL loss trains on."""
    N = primary.shape[0]
    E_sig = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    T_sig = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    for lo in range(0, N, BATCH):
        hi = min(lo + BATCH, N)
        var = fnn.forward_var(primary[lo:hi].to(DEVICE), xy[lo:hi].to(DEVICE))
        sig = var.clamp_min(1e-24).sqrt().cpu()
        E_sig[lo:hi] = sig[..., 0]
        T_sig[lo:hi] = sig[..., 1]
    return E_sig, T_sig


def _render_fnn_calibration(fnn, primary, xy, E_true, T_true, val_idx,
                            output_path, formats=("png",)):
    """Uncertainty-calibration figure for one surrogate: one panel per channel.

    Drawn in the Z-SCORED space the network is actually trained in. The logvar
    head lives entirely in z-scored space and `gaussian_nll_normalized` computes
    its loss there, so plotting raw log1p units showed the head against a scale
    it never optimises. Dividing σ and the error by the channel's `out_std` is
    the exact transform the loss applies, so every ratio (and the coverage
    number) is unchanged — only the axes move into training units.

    Requires a heteroscedastic (mean+variance) head — callers check `forward_var`."""
    import numpy as np

    p   = primary[val_idx]
    x   = xy[val_idx]
    E_t = E_true[val_idx]
    T_t = T_true[val_idx]
    E_p, T_p = fnn_predict(fnn, p, x)
    E_s, T_s = fnn_predict_sigma(fnn, p, x)

    # Per-channel output scale, from the same broadcast-shared buffers forward()
    # reads: E stat at index 0, T stat at index n_det.
    nd = int(getattr(fnn, "n_det", N_DETECTORS))
    E_std = float(fnn.out_std[0])
    T_std = float(fnn.out_std[nd])

    E_err = ((E_p - E_t).flatten().numpy()) / E_std
    T_err = ((T_p - T_t).flatten().numpy()) / T_std
    E_sig = (E_s.flatten().numpy()) / E_std
    T_sig = (T_s.flatten().numpy()) / T_std

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_CALIBRATION)
    _calibration_panel(axes[0], E_sig, E_err, "log1p(E)  [z]", legend=False)
    _calibration_panel(axes[1], T_sig, T_err, "log1p(T·1e8)  [z]", legend=False)
    fig.suptitle(f"Predicted σ vs error (N={E_err.size:,})", fontsize=FS_SUPTITLE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=FS_LEGEND_DENSE,
              ncol=2, bbox_to_anchor=(0.5, -0.12))
    # `rect` top=0.94 reserves headroom for the suptitle; bottom=0.1 reserves
    # headroom below the lower panel's own xlabel for the legend two rows
    # below it (bbox_to_anchor alone doesn't -- tight_layout has no idea a
    # legend exists outside the axes it's fitting).
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    # tight_layout's own h_pad is only advisory and empirically does nothing
    # here -- subplots_adjust after it is what actually forces a bigger gap,
    # which the lower panel's title needs to clear the upper panel's xlabel.
    fig.subplots_adjust(hspace=0.6)
    _savefig_multi(fig, output_path, dpi=130, formats=formats)
    plt.close(fig)
    # z-scored, matching the figure and the NLL loss. mean(err^2/var) is the
    # quantity the NLL balances: 1.0 is calibrated, >1 over-confident.
    E_r = float((E_err ** 2 / np.maximum(E_sig ** 2, 1e-30)).mean())
    T_r = float((T_err ** 2 / np.maximum(T_sig ** 2, 1e-30)).mean())
    print(f"       [z-scored] E: RMSE={np.sqrt((E_err**2).mean()):.4f}  "
          f"mean σ={E_sig.mean():.4f}  mean err²/σ²={E_r:.3f}   "
          f"T: RMSE={np.sqrt((T_err**2).mean()):.4f}  "
          f"mean σ={T_sig.mean():.4f}  mean err²/σ²={T_r:.3f}")


def load_fnn() -> FNNSurrogate:
    # Read width + dropout from the saved config and prefer the FNN's own
    # norm_stats (02_train_fnn.py updates the T slots in-memory for log-T
    # training and ships the modified stats inside fnn.pt; disk norm_stats.pt
    # still holds raw-T values).
    fnn_ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    cfg = fnn_ckpt.get("config", {})
    fnn = FNNSurrogate(
        n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=int(cfg.get("hidden", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(DEVICE)
    fnn.load_state_dict(fnn_ckpt["state_dict"])
    norm_stats = fnn_ckpt.get(
        "norm_stats",
        torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt")),
    )
    fnn.set_normalization(norm_stats)
    fnn.eval()
    print(f"[load] fnn.pt  epoch={fnn_ckpt.get('epoch','?')}  "
          f"val={fnn_ckpt.get('val_total','?')}  "
          f"hidden={int(cfg.get('hidden', 512))} "
          f"lbfgs_iter={fnn_ckpt.get('lbfgs_iter','?')}")
    return fnn


@torch.no_grad()
def fnn_predict(fnn: FNNSurrogate,
                primary: torch.Tensor,
                xy: torch.Tensor):
    N = primary.shape[0]
    E_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    T_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    for lo in range(0, N, BATCH):
        hi = min(lo + BATCH, N)
        pred = fnn(primary[lo:hi].to(DEVICE), xy[lo:hi].to(DEVICE))
        E_pred[lo:hi] = pred[..., 0].cpu()
        T_pred[lo:hi] = pred[..., 1].cpu()
    return E_pred, T_pred


def load_recon(folder: str = RECON_DEEPSETS_FOLDER):
    """Mirror of load_fnn() for the recon checkpoint. Used by the standalone
    CLI path; training scripts pass an already-trained recon in.

    Dispatches on the checkpoint's own config["model_type"] via
    build_recon_from_ckpt, so flat-MLP ("mlp") and DeepSets ("deepsets")
    checkpoints both load. Hardcoding Reconstruction here predated
    03_train_recon_deepsets.py and died on a state_dict shape mismatch
    against its checkpoints."""
    recon_ckpt = torch.load(os.path.join(folder, "recon.pt"),
                            map_location=DEVICE, weights_only=False)
    recon = build_recon_from_ckpt(recon_ckpt, N_DETECTORS, DEVICE)
    print(f"[load] {folder}/recon.pt  "
          f"model={recon_ckpt.get('config', {}).get('model_type', 'mlp')}  "
          f"epoch={recon_ckpt.get('epoch','?')}  "
          f"val={recon_ckpt.get('val_total','?')} "
          f"lbfgs_iter={recon_ckpt.get('lbfgs_iter','?')}")
    return recon


# --------------------------------------------------------------------------- #
# Dual-species loading. Mirrors 02_train_fnn_deepsets.py (per-species FNN) and
# 03_train_recon.py (combined dual surrogate for recon).
# --------------------------------------------------------------------------- #
SPECIES_TAGS = (("electron", 0), ("muon", 1))   # (tag, species id: 0=electron, 1=muon)


def load_species_fnn(species: str):
    """Load one per-species surrogate (fnn_electron.pt / fnn_muon.pt) from
    FNN_FOLDER. Uses build_surrogate_from_ckpt so flat-MLP or DeepSets configs
    both work, with the checkpoint's own per-species norm stats applied."""
    from modules_v6.deepsets_surrogate import build_surrogate_from_ckpt
    path = os.path.join(FNN_FOLDER, f"fnn_{species}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    fnn = build_surrogate_from_ckpt(ckpt, N_DETECTORS, PRIMARY_DIM, DEVICE)
    cfg = ckpt.get("config", {})
    print(f"[load] fnn_{species}.pt  model={cfg.get('model_type', 'fnn')}  "
          f"epoch={ckpt.get('epoch', '?')}  val={ckpt.get('val_total', '?')}")
    return fnn


def _render_fnn_scatter(fnn, primary, xy, E_true, T_true, val_idx, output_path,
                        vmin_E=10, vmax_E=4000, vmin_T=10, vmax_T=2500,
                        formats=("png",)):
    """Pure rendering — no I/O for models or corpus. Caller supplies a loaded
    FNN in eval mode plus the in-memory tensors. T_true must already be
    log1p(T*1e8)-transformed (matching what the FNN was trained against).
    vmin/vmax_{E,T} pin each panel's hexbin colour (count) scale (per species)."""
    p   = primary[val_idx]
    x   = xy[val_idx]
    E_t = E_true[val_idx]
    T_t = T_true[val_idx]
    E_p, T_p = fnn_predict(fnn, p, x)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_SCATTER)
    _scatter(axes[0], E_t.flatten().numpy(), E_p.flatten().numpy(),
             f"log1p(E)",
             vmin=vmin_E, vmax=vmax_E, legend=False)
    _scatter(axes[1], T_t.flatten().numpy(), T_p.flatten().numpy(),
             f"log1p(T·1e8)",
             vmin=vmin_T, vmax=vmax_T, legend=False)
    fig.suptitle(f"Target vs prediction (N={T_t.numel():,})", fontsize=FS_SUPTITLE_SCATTER)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=FS_LEGEND,
              ncol=len(labels), bbox_to_anchor=(0.5, -0.08))
    # bottom=0.08 leaves headroom below the axes' own xlabel for the legend
    # -- tight_layout does not know a legend sits outside the axes it fits.
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _savefig_multi(fig, output_path, dpi=130, formats=formats)
    plt.close(fig)


# Shared by every recon render: (label, axis lo, axis hi) per primary channel.
# dir_x/dir_y are direction cosines and span [-1, 1]; dir_z spans [-0.5, 0.5]
# (half the others' -- see place_clouds_enu / the v6 primary encoding for why).
# log_e_norm's TARGET is bounded to [0, 1] BY CONSTRUCTION (encode_primary's
# (log10(E) - LOG_E_MIN)/(LOG_E_MAX - LOG_E_MIN)), so its window is [0, 1] too --
# unlike dir_x/y/z, [0, 1] is log_e_norm's real physical range, not an
# arbitrary choice. The PREDICTION still overshoots slightly past both ends
# (measured down to about -0.5); that overshoot is real model behaviour and
# is worth its own diagnostic, but is deliberately NOT what sets this axis --
# widening the window to fit an overshooting prediction would treat the
# window as "whatever the model happened to output" rather than the channel's
# actual domain, which is the same mistake the original lo=0.0-everywhere
# version made in the other direction.
#
# An earlier version of this tuple used lo=0.0 for dir_x/y/z too, copied from
# the pre-existing (also wrong) implicit lo=0 in the plain scatter's hexbin
# calls below -- that silently dropped every negative-target/prediction point
# (hexbin ignores anything outside its extent), invisible as whitespace in a
# scatter but a false bright spike in the conditional panel, whose column
# density clips out-of-window values into the edge bin instead of dropping
# them. Measured fractions of negative PREDICTIONS at that old lo=0 window:
# dir_x 63.8%, dir_z 33.7%, log_e_norm 7.4%, dir_y 3.1% of N=572,165 -- real
# recon outputs in a real negative-valued domain (dir_x/y/z), not a defect.
RECON_CHANNELS = (("dir_x", -1.0, 1.0), ("dir_y", -1.0, 1.0),
                  ("dir_z", -0.5, 0.5), ("log_e_norm", 0.0, 1.0))


def _recon_target_pred(fnn, recon, primary, xy, val_idx):
    """(target, pred, N) for the recon channels — the shared data prep behind
    every recon figure. Target = v6 primary encoding [dir_x, dir_y, dir_z,
    log_e_norm] in raw units; recon runs on FNN-PREDICTED (E, T), not ground
    truth, same as 04_optimize, so this reflects the end-to-end FNN -> recon
    error."""
    p = primary[val_idx]
    x = xy[val_idx]
    E_pred, T_pred = fnn_predict(fnn, p, x)
    target = p[:, :4].float()

    N = p.shape[0]
    pred = torch.empty((N, 4), dtype=torch.float32)
    with torch.no_grad():
        for lo in range(0, N, BATCH):
            hi = min(lo + BATCH, N)
            xy_b = x[lo:hi].to(DEVICE)
            E_b  = E_pred[lo:hi].to(DEVICE)
            T_b  = T_pred[lo:hi].to(DEVICE)
            feats = torch.stack([xy_b[..., 0], xy_b[..., 1], E_b, T_b], dim=-1)  # (B, n_det, 4)
            pred[lo:hi] = recon(feats).cpu()                                     # DeepSets recon takes (B, n_det, 4)
    return target, pred, N


def _render_recon_scatter(fnn, recon, primary, xy, val_idx, output_path, formats=("png",)):
    """Pure rendering — caller supplies both nets (eval mode) and the
    in-memory primary/xy tensors. Recon target is `primary[val_idx, :4]`."""
    target, pred, N = _recon_target_pred(fnn, recon, primary, xy, val_idx)

    vmin_s = (1, 1, 1, 1)
    # vmax_s = (100, 100, 100, 200)
    vmax_s = (80, 200, 200, 200)
    # vmax_s = (200, 300, 300, 500)
    # 2x2, not 1x4: four panels side by side left each column too narrow at
    # print size for its own title next to the suptitle above it.
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_RECON)
    for ax, i, (name, lo, hi) in zip(axes.flat, range(4), RECON_CHANNELS):
        _scatter(ax, target[:, i].numpy(), pred[:, i].numpy(), f"Recon  {name}",
                vmin=vmin_s[i], vmax=vmax_s[i], lo=lo, hi=hi, legend=False)
    fig.suptitle(f"Recon: target vs prediction (N={N:,})", fontsize=FS_SUPTITLE_SCATTER)
    handles, labels_leg = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", fontsize=FS_LEGEND,
              ncol=len(labels_leg), bbox_to_anchor=(0.5, -0.04))
    # top=0.94 reserves headroom for the suptitle above the top row's own
    # panel titles; bottom=0.06 leaves headroom below the bottom row's xlabel
    # for the legend -- tight_layout does not know a legend sits outside the
    # axes it fits, nor does it budget for the suptitle unless told to.
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _savefig_multi(fig, output_path, dpi=130, formats=formats)
    plt.close(fig)


def _render_recon_conditional(fnn, recon, primary, xy, val_idx, output_path,
                              formats=("png",)):
    """Conditional-density companion to `_render_recon_scatter`, same
    peak-scaled-column-density + mean/±1σ-band treatment as
    `_render_fnn_conditional` (see `_conditional_panel`), applied to the 4
    recon channels instead of the FNN's E/T.

    `hit_only=False` throughout: unlike the FNN's per-detector E/T, every recon
    target here is a dense, continuous primary-encoding value — there is no
    dark/hit population to split out, so nothing is filtered before binning."""
    target, pred, N = _recon_target_pred(fnn, recon, primary, xy, val_idx)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_RECON)
    for ax, i, (name, lo, hi) in zip(axes.flat, range(4), RECON_CHANNELS):
        _conditional_panel(ax, target[:, i].numpy(), pred[:, i].numpy(),
                           f"Recon  {name}", legend=False,
                           hit_only=False, lo=lo, hi=hi)
    fig.suptitle(f"Recon: conditional density (N={N:,})",
                fontsize=FS_SUPTITLE_SCATTER)
    handles, labels_leg = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", fontsize=FS_LEGEND,
              ncol=len(labels_leg), bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _savefig_multi(fig, output_path, dpi=130, formats=formats)
    plt.close(fig)


def _load_corpus():
    """Load shared tensors + strategy ids. Applies log1p(T*1e8) so T_true
    matches the FNN's training target space (see 02_train_fnn.py).

    Only used by the standalone CLI / when training scripts call into the
    plotters without providing their already-loaded tensors."""
    primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    E_true    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_true    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    T_true = torch.log1p(T_true * T_LOG_SCALE)
    return primary, xy, E_true, T_true, strat_ids


def plot_fnn_only(*, fnn=None,
                  primary=None, xy=None,
                  E_true=None, T_true=None,
                  val_idx=None,
                  species=None,
                  output_path=None):
    """Render fnn_target_vs_pred.png. Every argument is optional: anything
    left as None gets loaded from disk so the standalone CLI still works.

    Training-script callers (02_train_fnn.py) pass everything they already
    have in memory — fnn (with best weights reloaded), primary, xy, E_all,
    T_all (already log1p-transformed in 02), val_idx — and no disk I/O for
    the corpus is performed. T_true MUST be in log-T space if provided.

    `species` ("electron"/"muon") selects the per-species hexbin colour
    scale from FNN_DUAL_VLIM; left None it falls back to the generic
    _render_fnn_scatter defaults.
    """
    if primary is None or xy is None or E_true is None or T_true is None:
        primary, xy, E_true, T_true, strat_ids_disk = _load_corpus()
    else:
        strat_ids_disk = None
    if val_idx is None:
        if strat_ids_disk is None:
            strat_ids_disk = torch.load(
                os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")
            ).long()
        val_idx = shower_level_val_idx(strat_ids_disk, VAL_FRAC, FNN_VAL_SEED)
    if fnn is None:
        fnn = load_fnn()
    if output_path is None:
        os.makedirs(FNN_FOLDER, exist_ok=True)
        output_path = os.path.join(FNN_FOLDER, "fnn_target_vs_pred.png")
    _render_fnn_scatter(fnn, primary, xy, E_true, T_true, val_idx, output_path,
                        **FNN_DUAL_VLIM.get(species, {}))
    base, ext = os.path.splitext(output_path)
    # Needs no variance head — the σ band is optional inside the panel.
    _render_fnn_conditional(
        fnn, primary, xy, E_true, T_true, val_idx,
        base.replace("target_vs_pred", "conditional") + ext)
    if hasattr(fnn, "forward_var"):
        _render_fnn_calibration(
            fnn, primary, xy, E_true, T_true, val_idx,
            base.replace("target_vs_pred", "calibration") + ext)


def plot_recon_only(*, fnn=None, recon=None,
                    primary=None, xy=None,
                    val_idx=None,
                    output_path=None,
                    recon_folder=RECON_DEEPSETS_FOLDER,
                    formats=("png",)):
    """Render recon_target_vs_pred.png. Like `plot_fnn_only`, every argument
    is optional. Training-script callers (03_train_recon.py) pass fnn +
    recon (best weights reloaded) + primary + xy + val_idx; no disk I/O for
    those is then performed.

    `recon_folder` is where a None `recon` is loaded from and where a None
    `output_path` lands — keep the checkpoint and its scatter in the same
    run folder."""
    if primary is None or xy is None:
        primary, xy, _E, _T, strat_ids_disk = _load_corpus()
    else:
        strat_ids_disk = None
    if val_idx is None:
        if strat_ids_disk is None:
            strat_ids_disk = torch.load(
                os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")
            ).long()
        val_idx = shower_level_val_idx(strat_ids_disk, VAL_FRAC, RECON_VAL_SEED)
    if fnn is None:
        fnn = load_fnn()
    if recon is None:
        recon = load_recon(recon_folder)
    if output_path is None:
        os.makedirs(recon_folder, exist_ok=True)
        output_path = os.path.join(recon_folder, "recon_target_vs_pred.png")
    _render_recon_scatter(fnn, recon, primary, xy, val_idx, output_path, formats=formats)
    base, ext = os.path.splitext(output_path)
    _render_recon_conditional(
        fnn, recon, primary, xy, val_idx,
        base.replace("target_vs_pred", "conditional") + ext, formats=formats)


# Per-species hexbin colour (count) limits for the dual FNN scatters:
# (vmin_E, vmax_E, vmin_T, vmax_T). Muon signals are denser than electron.
FNN_DUAL_VLIM = {
    "electron": dict(vmin_E=0, vmax_E=2000,  vmin_T=0, vmax_T=2000),
    "muon":     dict(vmin_E=0, vmax_E=3000, vmin_T=0, vmax_T=1000),
}


def plot_fnn_dual(output_dir=None):
    """Per-species FNN scatter for the dual-species surrogate. Each species'
    DeepSets model is evaluated against its OWN species subset (split on the
    Step-1 species_ids sidecar, the corpus E/T being per-species), reproducing
    02_train_fnn_deepsets.py's split. Writes
    FNN_FOLDER/fnn_<species>_target_vs_pred.png per species. Per-species colour
    scales come from FNN_DUAL_VLIM."""
    primary, xy, E_true, T_true, strat_ids = _load_corpus()
    species_ids = torch.load(
        os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt")).long()
    if output_dir is None:
        output_dir = FNN_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    for tag, species_val in SPECIES_TAGS:
        idx = torch.nonzero(species_ids == species_val).squeeze(-1)
        if idx.numel() == 0:
            print(f"[skip] no {tag} rows (species id {species_val}) in corpus")
            continue
        fnn = load_species_fnn(tag)
        # val_idx is positional within the filtered subset; pass the subset
        # tensors so _render_fnn_scatter indexes them consistently.
        val_idx = shower_level_val_idx(strat_ids[idx], VAL_FRAC, FNN_VAL_SEED)
        out = os.path.join(output_dir, f"fnn_{tag}_target_vs_pred.png")
        _render_fnn_scatter(fnn, primary[idx], xy[idx],
                            E_true[idx], T_true[idx], val_idx, out,
                            **FNN_DUAL_VLIM[tag])
        _render_fnn_conditional(
            fnn, primary[idx], xy[idx], E_true[idx], T_true[idx], val_idx,
            os.path.join(output_dir, f"fnn_{tag}_conditional.png"))
        # Calibration only exists for the heteroscedastic (mean+var) head; a
        # plain FNNSurrogate checkpoint has no forward_var and is skipped.
        if hasattr(fnn, "forward_var"):
            _render_fnn_calibration(
                fnn, primary[idx], xy[idx], E_true[idx], T_true[idx], val_idx,
                os.path.join(output_dir, f"fnn_{tag}_calibration.png"))
        else:
            print(f"[skip] {tag}: no forward_var (not a mean+variance head)")


def plot_recon_dual(output_path=None, recon_folder=RECON_DEEPSETS_FOLDER, formats=("png",)):
    """Recon scatter for the dual-species surrogate. The combined
    DualSpeciesSurrogate (fnn_electron.pt + fnn_muon.pt) feeds the recon on the
    FULL corpus — identical to 03_train_recon.py. recon.pt itself is a single
    (non-per-species) net, so load_recon() is reused unchanged."""
    from modules_v6.dual_surrogate import load_dual_surrogate
    dual = load_dual_surrogate(FNN_FOLDER, DEVICE)
    plot_recon_only(fnn=dual, output_path=output_path,
                    recon_folder=recon_folder, formats=formats)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dual", action="store_true",
                    help="dual-species surrogate: per-species FNN scatters + "
                         "combined-surrogate recon scatter")
    args = ap.parse_args()

    print("=" * 72)
    print("v6/plots/02_plot_nn_target_vs_pred.py" + ("  [dual]" if args.dual else ""))
    print("=" * 72)
    if args.dual:
        plot_fnn_dual()
        plot_recon_dual()
    else:
        plot_fnn_only()
        plot_recon_only()


if __name__ == "__main__":
    main()
