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

    cd TambOpt/detector_optimization_v6
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_DIR = os.path.dirname(_HERE)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

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


def _scatter(ax, x, y, title: str, vmin=None, vmax=None, lo=None, hi=None):
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
    ax.legend(loc="upper left", fontsize=8, framealpha=1)


def _calibration_panel(ax, sigma, err, channel: str):
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
    ax.set_title(f"{channel}: within 2σ — actual {100 * cover:.1f}%, "
                 f"expected 95.4%")
    ax.legend(loc="upper left", fontsize=7, framealpha=1)


def _conditional_panel(ax, target, pred, sigma, channel: str,
                       n_bins=60, min_count=200):
    """P(prediction | target) for HIT detectors, plus the reverse-conditional curve.

    The raw target-vs-prediction hexbin is unreadable for reasons unrelated to
    model quality: a huge spike of DARK detectors (target exactly 0) owns the
    colour scale, and the joint density hides the regression wherever the target
    distribution is thin. So: keep only `target > 0`, and scale each target-column
    to its own PEAK — the figure is then the SHAPE of P(pred | target) at every
    target, equally legible across the range. Columns under `min_count` are left
    blank rather than normalised from a handful of samples.

    The one curve drawn is `target | prediction` — a MEAN plus a p16/p84 (±1σ)
    band — because it is the only summary whose ideal is `y = x`:

    * the other direction (prediction per target-bin) is redundant with the
      peak-scaled density AND misleading — it conditions on a noisy realisation
      of the target while the net is fitted to `E[y | input]`, so regression
      attenuation holds its slope below 1 however good the model is;
    * `E[target | pred] = pred` is exact for a calibrated conditional mean at any
      noise level. MEAN, not median: that identity is about means and is what the
      Gaussian NLL fits the mean head to. The conditional is strongly
      right-skewed in log1p space near zero, so a median curve leaves `y = x`
      even for a perfect model and charges the skew to the model as bias.

    Returns the numbers for the caller to print; only the slope reaches the
    title. `sigma` may be None — the band/σ comparison is then skipped.
    """
    import numpy as np

    hit = target > 0.0
    t, p = target[hit], pred[hit]
    s = None if sigma is None else sigma[hit]
    if t.size < 10 * n_bins:
        ax.set_title(f"{channel}: too few hit samples ({t.size})")
        return None

    # p99.9, not the max: a few extreme targets would spend the axis on whitespace.
    lo, hi = 0.0, float(np.percentile(t, 99.9))
    edges = np.linspace(lo, hi, n_bins + 1)

    # Predictions outside the window are clipped INTO the edge bins so no column
    # loses mass before normalisation; the curve below uses unclipped values.
    H, _, _ = np.histogram2d(t, np.clip(p, lo, hi), bins=(edges, edges))
    ok = H.sum(axis=1) >= min_count
    dens = np.full_like(H, np.nan)
    dens[ok] = H[ok] / np.maximum(H[ok].max(axis=1, keepdims=True), 1.0)
    im = ax.imshow(dens.T, origin="lower", extent=(lo, hi, lo, hi),
                   aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="P(prediction | target)", pad=0.02,
                 fraction=0.046)

    # Binned on the PREDICTION, so the binning variable is the Y axis — hence
    # fill_betweenx and the (value, centre) argument order in every plot call.
    y, avg, p16, p84, rms, cnt = _binned(p, t, s, edges, min_count)
    ax.fill_betweenx(y, p16, p84, color="#ff2d95", alpha=0.15, zorder=3)
    ax.plot(p16, y, color="#ff2d95", ls=":", lw=1.5, alpha=0.9, zorder=4,
            label="±1σ  (p16 / p84 of target)")
    ax.plot(p84, y, color="#ff2d95", ls=":", lw=1.5, alpha=0.9, zorder=4)
    ax.plot(avg, y, color="#ff2d95", marker="s", ms=3.5, lw=2.2, zorder=5,
            label="mean target | prediction")
    ax.plot([lo, hi], [lo, hi], color="red", ls="--", lw=2.0, alpha=0.9,
            zorder=6, label="y = x  (ideal)")

    # sqrt(count) weights so the sparse high-prediction bins do not drive the fit.
    slope = float(np.polyfit(avg, y, 1, w=np.sqrt(cnt))[0]) if y.size > 1 \
        else float("nan")
    # Half the p16-p84 gap is the measured 1σ. Against the head's own predicted
    # σ in the same bins, a ratio > 1 means it is over-confident.
    emp = float(np.median(0.5 * (p84 - p16)))
    ratio = float("nan") if rms is None \
        else emp / max(float(np.median(rms)), 1e-12)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("target"); ax.set_ylabel("prediction")
    ax.set_title(f"{channel}   —   slope {slope:.2f}  (ideal 1.00)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return dict(frac_hit=float(hit.mean()), n_hit=int(t.size), slope=slope,
                sigma=emp, ratio=ratio)


def _binned(bin_var, val, extra, edges, min_count):
    """Per-bin (centre, mean, p16, p84, rms of `extra`, count) of `val`.

    Bincounts for the moments — they accumulate in float64, which a naive fp32
    sum over ~1e6 samples per bin would not. One lexsort for the two quantiles,
    indexed positionally. Bins under `min_count` are dropped, so every returned
    array is already plot-ready. `extra` (predicted σ) may be None."""
    import numpy as np
    nb = edges.size - 1
    idx = np.clip(np.digitize(bin_var, edges[1:-1]), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    n = np.maximum(cnt, 1)
    avg = np.bincount(idx, weights=val, minlength=nb) / n
    rms = None if extra is None else np.sqrt(
        np.bincount(idx, weights=extra.astype(np.float64) ** 2, minlength=nb) / n)

    v = val[np.lexsort((val, idx))]
    start = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    def q(f):  # positional quantile within each bin's contiguous slice
        return v[np.minimum(start + (f * np.maximum(cnt - 1, 0)).astype(np.intp),
                            v.size - 1)]

    k = cnt >= min_count
    ctr = 0.5 * (edges[:-1] + edges[1:])
    return (ctr[k], avg[k], q(0.16)[k], q(0.84)[k],
            None if rms is None else rms[k], cnt[k].astype(np.float64))


def _render_fnn_conditional(fnn, primary, xy, E_true, T_true, val_idx,
                            output_path):
    """Conditional-density figure, one panel per channel.

    Companion to `_render_fnn_scatter`, which shows the same data as a joint
    density dominated by the dark-detector population. Uses the σ head when the
    checkpoint has one."""
    import numpy as np

    p, x = primary[val_idx], xy[val_idx]
    E_p, T_p = (a.flatten().numpy() for a in fnn_predict(fnn, p, x))
    E_s, T_s = (a.flatten().numpy() for a in fnn_predict_sigma(fnn, p, x)) \
        if hasattr(fnn, "forward_var") else (None, None)
    E_t = E_true[val_idx].flatten().numpy()

    panels = (("E", "log1p(E)", E_t, E_p, E_s),
              ("T", "log1p(T·1e8)", T_true[val_idx].flatten().numpy(), T_p, T_s))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    stats = {tag: _conditional_panel(ax, tt, pp, ss, lab)
             for ax, (tag, lab, tt, pp, ss) in zip(axes, panels)}

    hit_txt = f"{100 * stats['E']['frac_hit']:.0f}% of " if stats["E"] else ""
    fig.suptitle(f"Deepsets conditional density — hit detectors "
                 f"({hit_txt}{E_t.size:,})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"[save] {output_path}")

    # Dark/lit split as the classification it is, at the count>=1 cut
    # LAYOUT_THRESHOLD gates the trigger on (E only — meaningless for a time).
    lt, lp = E_t >= np.log1p(1.0), E_p >= np.log1p(1.0)
    print(f"       [E, count>=1] true lit {100 * lt.mean():.2f}%  pred lit "
          f"{100 * lp.mean():.2f}%  agree {100 * (lt == lp).mean():.2f}%  "
          f"recall {100 * (lp & lt).sum() / max(lt.sum(), 1):.2f}%")
    for tag, st in stats.items():
        if st:
            print(f"       [{tag}] hit {100 * st['frac_hit']:.2f}% "
                  f"(n={st['n_hit']:,})  slope {st['slope']:.3f} (ideal 1.000)  "
                  f"1σ band {st['sigma']:.3f}  band/predicted-σ "
                  f"{st['ratio']:.2f}")


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
                            output_path):
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

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 10))
    _calibration_panel(axes[0], E_sig, E_err, "log1p(E)  [z]")
    _calibration_panel(axes[1], T_sig, T_err, "log1p(T·1e8)  [z]")
    # Three lines: the single-column layout is far too narrow for this on one
    # row, and a clipped suptitle loses the sample count. `rect` reserves the
    # headroom tight_layout would otherwise hand to the top panel.
    fig.suptitle("Deepsets predicted σ vs realised error\n"
                 f"N={E_err.size:,} detector-samples", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"[save] {output_path}")
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
                        vmin_E=10, vmax_E=4000, vmin_T=10, vmax_T=2500):
    """Pure rendering — no I/O for models or corpus. Caller supplies a loaded
    FNN in eval mode plus the in-memory tensors. T_true must already be
    log1p(T*1e8)-transformed (matching what the FNN was trained against).
    vmin/vmax_{E,T} pin each panel's hexbin colour (count) scale (per species)."""
    p   = primary[val_idx]
    x   = xy[val_idx]
    E_t = E_true[val_idx]
    T_t = T_true[val_idx]
    E_p, T_p = fnn_predict(fnn, p, x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    _scatter(axes[0], E_t.flatten().numpy(), E_p.flatten().numpy(),
             f"log1p(E)",
             vmin=vmin_E, vmax=vmax_E)
    _scatter(axes[1], T_t.flatten().numpy(), T_p.flatten().numpy(),
             f"log1p(T·1e8)",
             vmin=vmin_T, vmax=vmax_T)
    fig.suptitle(f"Deepsets target vs prediction\n N={T_t.numel():,} detector-samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"[save] {output_path}")


def _render_recon_scatter(fnn, recon, primary, xy, val_idx, output_path):
    """Pure rendering — caller supplies both nets (eval mode) and the
    in-memory primary/xy tensors. Recon target is `primary[val_idx, :4]`."""
    p = primary[val_idx]
    x = xy[val_idx]

    # Recon sees FNN predictions, not ground-truth (E, T) — same as 04_optimize.
    E_pred, T_pred = fnn_predict(fnn, p, x)

    # Target = v6 primary encoding [dir_x, dir_y, dir_z, log_e_norm] in raw units.
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

    labels = ("dir_x", "dir_y", "dir_z", "log_e_norm")
    vmin_s = (1, 1, 1, 1)
    # vmax_s = (100, 100, 100, 200)
    vmax_s = (80, 200, 200, 200)
    # vmax_s = (200, 300, 300, 500)
    hi = (1, 1, 0.5, 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for i, name in enumerate(labels):
        _scatter(axes[i], target[:, i].numpy(), pred[:, i].numpy(), f"Recon  {name}", vmin=vmin_s[i], vmax=vmax_s[i], hi=hi[i]) # TODO
    fig.suptitle(f"Recon target vs prediction — val split  (N={N:,})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"[save] {output_path}")


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
                    recon_folder=RECON_DEEPSETS_FOLDER):
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
    _render_recon_scatter(fnn, recon, primary, xy, val_idx, output_path)


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


def plot_recon_dual(output_path=None, recon_folder=RECON_DEEPSETS_FOLDER):
    """Recon scatter for the dual-species surrogate. The combined
    DualSpeciesSurrogate (fnn_electron.pt + fnn_muon.pt) feeds the recon on the
    FULL corpus — identical to 03_train_recon.py. recon.pt itself is a single
    (non-per-species) net, so load_recon() is reused unchanged."""
    from modules_v6.dual_surrogate import load_dual_surrogate
    dual = load_dual_surrogate(FNN_FOLDER, DEVICE)
    plot_recon_only(fnn=dual, output_path=output_path,
                    recon_folder=recon_folder)


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
