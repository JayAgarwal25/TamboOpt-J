#!/usr/bin/env python3
"""
Consolidation follow-up to detector_removal_analysis.py: is the "critical
five" / "redundant five" ranking from the leave-one-out sweep stable, or an
artifact of the one fixed batch of primaries (BATCH_SEED_BASE=1000,
N_BATCHES=3) that experiment happened to use?

Repeats the leave-one-out sweep K times, each against an independently drawn
set of fresh batches (different seed bases, all avoiding both seed=42 -- the
training/scoring batch, a known outlier -- and 1000, already used by the
original run). Reports, across all K+1 dip vectors (the K new ones plus the
original from detector_removal_results.json):
  - pairwise Pearson correlation (are the per-detector dips consistent in
    relative ranking across independent primary samples?)
  - per-detector mean +/- std of dip (a resampling-based uncertainty band)
  - how often each of the ORIGINAL top-5 critical / bottom-5 redundant
    detectors reappears in each resample's own top-5 / bottom-5
"""
import os, sys, json, time
import numpy as np
import torch

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import layouts as _layouts  # noqa: E402  (layout paths live in one place)
import modules  # noqa: F401 — package import; keeps modules on the path

from modules.constants import N_DETECTORS, TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER
from modules.optimize import utility_of_xy, load_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Results live beside the other run outputs, not next to the code.
HERE = _layouts.results_dir()
BATCH_SIZE = 512
N_BATCHES = 3          # match the original run's convention exactly
N_RESAMPLES = 5
RESAMPLE_SEED_BASES = [2000, 3000, 4000, 5000, 6000]   # avoid 42 and 1000 (already used)
N_HIGHLIGHT = 5

print("=" * 70)
print("Detector removal: batch-resampling robustness check")
print("=" * 70)

fnn, recon = load_models(DEVICE, fnn_folder=FNN_FOLDER, recon_dir=RECON_FOLDER + "_deepsets")

primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                         weights_only=False).float()
n_total = primary_all.shape[0]


def fresh_batches(seed_base, n_batches):
    out = []
    for b in range(n_batches):
        g = torch.Generator().manual_seed(seed_base + b)
        idx = torch.randint(0, n_total, (BATCH_SIZE,), generator=g)
        out.append(primary_all[idx].to(DEVICE))
    return out


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


lbfgs_x, lbfgs_y, _ = load_layout(
    _layouts.primary())

with open(os.path.join(HERE, "detector_removal_results.json")) as f:
    original = json.load(f)
original_dips = np.array(original["leave_one_out"]["dips"], dtype=np.float64)
original_full_U = float(original["full_U"])


@torch.no_grad()
def eval_U(x, y, batches):
    Us = [float(utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon)[0].item()) for p in batches]
    return float(np.mean(Us))


def leave_one_out_dips(batches):
    full_U = eval_U(lbfgs_x, lbfgs_y, batches)
    dips = np.zeros(N_DETECTORS, dtype=np.float64)
    for i in range(N_DETECTORS):
        mask = torch.ones(N_DETECTORS, dtype=torch.bool)
        mask[i] = False
        u_without = eval_U(lbfgs_x[mask], lbfgs_y[mask], batches)
        dips[i] = full_U - u_without
    return full_U, dips


all_dips = [original_dips]   # index 0 = original run
all_full_U = [original_full_U]
labels = ["original(seed_base=1000)"]

for k, seed_base in enumerate(RESAMPLE_SEED_BASES):
    t0 = time.time()
    batches = fresh_batches(seed_base, N_BATCHES)
    full_U, dips = leave_one_out_dips(batches)
    all_dips.append(dips)
    all_full_U.append(full_U)
    labels.append(f"resample{k}(seed_base={seed_base})")
    print(f"[resample {k}] seed_base={seed_base}  full_U={full_U:.4f}  "
          f"dip mean={dips.mean():.4f} std={dips.std():.4f}  ({time.time()-t0:.0f}s)")

all_dips = np.stack(all_dips, axis=0)   # (N_RESAMPLES+1, N_DETECTORS)

print(f"\nfull_U across {len(all_full_U)} independent samples: "
      f"mean={np.mean(all_full_U):.3f} std={np.std(all_full_U):.3f} "
      f"(individual: {[round(u,3) for u in all_full_U]})")

# ── Pairwise correlation between dip vectors (including the original) ─────
n_runs = all_dips.shape[0]
corr = np.corrcoef(all_dips)
print(f"\n[correlation] pairwise Pearson r between all {n_runs} dip vectors "
      f"(1.0 on diagonal):")
for i in range(n_runs):
    row = "  ".join(f"{corr[i,j]:+.3f}" for j in range(n_runs))
    print(f"  {labels[i]:28s}: {row}")
off_diag = corr[~np.eye(n_runs, dtype=bool)]
print(f"  mean off-diagonal r = {off_diag.mean():.3f}  (min={off_diag.min():.3f})")

# ── Per-detector mean/std across the K NEW resamples only (excludes original) ──
resample_dips = all_dips[1:]   # (N_RESAMPLES, N_DETECTORS)
dip_mean = resample_dips.mean(axis=0)
dip_std = resample_dips.std(axis=0)

# ── Rank stability: does each resample's own top/bottom-5 match the original's? ──
original_order = np.argsort(original_dips)
original_critical = set(original_order[-N_HIGHLIGHT:].tolist())
original_redundant = set(original_order[:N_HIGHLIGHT].tolist())

print(f"\n[rank stability] original critical-{N_HIGHLIGHT}: {sorted(original_critical)}")
print(f"[rank stability] original redundant-{N_HIGHLIGHT}: {sorted(original_redundant)}")

critical_hit_counts = {d: 0 for d in original_critical}
redundant_hit_counts = {d: 0 for d in original_redundant}
for k in range(N_RESAMPLES):
    order_k = np.argsort(resample_dips[k])
    critical_k = set(order_k[-N_HIGHLIGHT:].tolist())
    redundant_k = set(order_k[:N_HIGHLIGHT].tolist())
    for d in original_critical:
        if d in critical_k:
            critical_hit_counts[d] += 1
    for d in original_redundant:
        if d in redundant_k:
            redundant_hit_counts[d] += 1
    print(f"  resample{k} own critical-{N_HIGHLIGHT}: {sorted(critical_k)}  "
          f"(overlap with original: {len(critical_k & original_critical)}/{N_HIGHLIGHT})")
    print(f"  resample{k} own redundant-{N_HIGHLIGHT}: {sorted(redundant_k)}  "
          f"(overlap with original: {len(redundant_k & original_redundant)}/{N_HIGHLIGHT})")

print(f"\n[rank stability] how many of {N_RESAMPLES} resamples keep each ORIGINAL "
      f"critical detector in their own top-{N_HIGHLIGHT}:")
for d in sorted(original_critical, key=lambda d: -original_dips[d]):
    print(f"    det {d:3d}  original_dip={original_dips[d]:+.4f}  "
          f"resample_mean={dip_mean[d]:+.4f}+/-{dip_std[d]:.4f}  "
          f"hit {critical_hit_counts[d]}/{N_RESAMPLES}")
print(f"[rank stability] how many of {N_RESAMPLES} resamples keep each ORIGINAL "
      f"redundant detector in their own bottom-{N_HIGHLIGHT}:")
for d in sorted(original_redundant, key=lambda d: original_dips[d]):
    print(f"    det {d:3d}  original_dip={original_dips[d]:+.4f}  "
          f"resample_mean={dip_mean[d]:+.4f}+/-{dip_std[d]:.4f}  "
          f"hit {redundant_hit_counts[d]}/{N_RESAMPLES}")

results = dict(
    labels=labels,
    all_dips=all_dips.tolist(),
    all_full_U=all_full_U,
    correlation_matrix=corr.tolist(),
    mean_offdiag_correlation=float(off_diag.mean()),
    resample_dip_mean=dip_mean.tolist(),
    resample_dip_std=dip_std.tolist(),
    original_critical=sorted(original_critical),
    original_redundant=sorted(original_redundant),
    critical_hit_counts={str(k): v for k, v in critical_hit_counts.items()},
    redundant_hit_counts={str(k): v for k, v in redundant_hit_counts.items()},
)
out_path = os.path.join(HERE, "detector_removal_batch_robustness_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    order = np.argsort(original_dips)
    ax.errorbar(np.arange(N_DETECTORS), dip_mean[order], yerr=dip_std[order],
                fmt="o", markersize=3, capsize=2, alpha=0.6, label="resample mean +/- std")
    ax.plot(np.arange(N_DETECTORS), original_dips[order], "x", color="red",
            markersize=5, label="original run")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("detector rank (by original dip, ascending)")
    ax.set_ylabel("leave-one-out dip")
    ax.set_title("Leave-one-out dip: original vs. resampled batches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(HERE, "detector_removal_batch_robustness.png")
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")
except Exception as exc:
    print(f"[plot] skipped ({exc!r})")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  mean pairwise dip-vector correlation across {n_runs} independent batch "
      f"draws: {off_diag.mean():.3f}")
print(f"  original critical-{N_HIGHLIGHT} average hit rate in resamples' own "
      f"critical-{N_HIGHLIGHT}: {np.mean(list(critical_hit_counts.values()))/N_RESAMPLES*100:.0f}%")
print(f"  original redundant-{N_HIGHLIGHT} average hit rate in resamples' own "
      f"redundant-{N_HIGHLIGHT}: {np.mean(list(redundant_hit_counts.values()))/N_RESAMPLES*100:.0f}%")
