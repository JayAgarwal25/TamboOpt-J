#!/usr/bin/env python3
"""
Redundancy / submodularity-style analysis: how much does each individual
detector actually matter, and how much redundancy is there among the 100?

Part 1: leave-one-out sweep. For each of the 100 detectors, remove it (99
remain) and measure the utility dip vs. the full 100-detector layout. This
works because the FNN/recon (DeepSets) are genuinely permutation- AND
count-invariant -- the pooling (mean/max) doesn't care how many tokens go
in, so evaluating with 99 detectors is a perfectly valid forward pass, not
an approximation.

Part 2: three greedy/random removal SEQUENCES, tracking U as detectors are
progressively stripped away from 100 down to a floor of 20:
  - "highest_dip_first": at each step, remove whichever remaining detector
    currently causes the LARGEST utility dip (the most critical/irreplacable
    one), re-evaluated fresh at every step (not just sorted once up front).
  - "lowest_dip_first": remove whichever remaining detector causes the
    SMALLEST dip (the most redundant one) at each step.
  - "random": remove in one fixed random shuffle order.

If detectors are highly redundant (consistent with the "broad flat plateau"
finding), "lowest_dip_first" should be able to strip away many detectors
before U drops much, while "highest_dip_first" should crater quickly.
Comparing the three curves is a direct, mechanistic test of *why* the
landscape looks flat (redundancy) rather than just re-confirming that it is.

Reusable across any saved layout via --layout_path/--layout_tag (defaults to
the L-BFGS-best layout, matching the original single-layout investigation).
Pass --layout_tag to run this on a different optimizer's layout -- outputs
then land in other_optimizers/<tag>/ instead of this directory directly.
"""
import argparse
import os, sys, json, time
import numpy as np
import torch

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (layout paths live in one place)
import modules_v6  # noqa: F401

from modules_v6.constants import N_DETECTORS, TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER
from modules_v6.opt_core import utility_of_xy, load_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
DEFAULT_LAYOUT_PATH = _layouts.primary()

ap = argparse.ArgumentParser()
ap.add_argument("--layout_path", type=str, default=DEFAULT_LAYOUT_PATH,
                help="Path to a layout_best.pt to analyze (default: L-BFGS-best).")
ap.add_argument("--layout_tag", type=str, default=None,
                help="Label for this layout. If given, outputs land in "
                     "other_optimizers/<tag>/; if omitted, outputs use the original "
                     "flat filenames (backward-compatible with L-BFGS-only results).")
args = ap.parse_args()
LAYOUT_LABEL = args.layout_tag or "L-BFGS-best"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "other_optimizers", args.layout_tag) if args.layout_tag else HERE
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SEED_BASE = 1000     # avoid seed=42 (the training/scoring batch, a known outlier)
BATCH_SIZE = 512
N_BATCHES = 3
REMOVAL_FLOOR = 20          # stop greedy/random removal once this many detectors remain
SEED = 123

print("=" * 70)
print(f"Detector removal / redundancy analysis -- layout: {LAYOUT_LABEL}")
print("=" * 70)

fnn, recon = load_models(DEVICE, fnn_folder=FNN_FOLDER, recon_dir=RECON_FOLDER + "_deepsets")

primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                         weights_only=False).float()
n_total = primary_all.shape[0]


def fresh_batch(seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n_total, (BATCH_SIZE,), generator=g)
    return primary_all[idx].to(DEVICE)


BATCHES = [fresh_batch(BATCH_SEED_BASE + b) for b in range(N_BATCHES)]


@torch.no_grad()
def eval_U(x, y):
    """x, y are 1-D tensors of ANY length (not necessarily 100) -- DeepSets
    pooling is count-invariant, this is a genuine forward pass, not a hack."""
    Us = [float(utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon)[0].item()) for p in BATCHES]
    return float(np.mean(Us))


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


lbfgs_x, lbfgs_y, lbfgs_U_saved = load_layout(args.layout_path)
full_U = eval_U(lbfgs_x, lbfgs_y)
print(f"[{LAYOUT_LABEL}] U (saved): {lbfgs_U_saved:.4f}  (re-evaluated, {N_BATCHES} fresh batches): {full_U:.4f}")
print(f"n_detectors = {N_DETECTORS}")

# ── Part 1: leave-one-out sweep over all 100 detectors ─────────────────────
print(f"\n[part 1] leave-one-out sweep ({N_DETECTORS} evals) ...")
t0 = time.time()
dips = np.zeros(N_DETECTORS, dtype=np.float64)
for i in range(N_DETECTORS):
    mask = torch.ones(N_DETECTORS, dtype=torch.bool)
    mask[i] = False
    u_without = eval_U(lbfgs_x[mask], lbfgs_y[mask])
    dips[i] = full_U - u_without
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{N_DETECTORS}  ({time.time()-t0:.0f}s elapsed)")
print(f"[part 1] done in {time.time()-t0:.0f}s")

order_by_dip = np.argsort(dips)  # ascending: most redundant first, most critical last
print(f"  dip stats: min={dips.min():.4f}  max={dips.max():.4f}  mean={dips.mean():.4f}  "
      f"std={dips.std():.4f}")
print(f"  most REDUNDANT 5 detectors (smallest dip): "
      f"{[(int(i), round(float(dips[i]), 4)) for i in order_by_dip[:5]]}")
print(f"  most CRITICAL 5 detectors (largest dip):   "
      f"{[(int(i), round(float(dips[i]), 4)) for i in order_by_dip[-5:]]}")

loo_results = dict(
    full_U=full_U, dips=dips.tolist(),
    order_by_dip_ascending=order_by_dip.tolist(),
)

# ── Part 2: three removal sequences ─────────────────────────────────────────
def greedy_removal(pick_max: bool, label: str):
    """At each step, evaluate removing each remaining detector, then actually
    remove the one with the largest (pick_max=True) or smallest (False) dip."""
    remaining = list(range(N_DETECTORS))
    curve = [(N_DETECTORS, full_U)]
    t0 = time.time()
    while len(remaining) > REMOVAL_FLOOR:
        best_choice = None
        best_u = None
        for cand in remaining:
            trial = [j for j in remaining if j != cand]
            idx_t = torch.tensor(trial, dtype=torch.long)
            u = eval_U(lbfgs_x[idx_t], lbfgs_y[idx_t])
            if best_u is None or (pick_max and u < best_u) or (not pick_max and u > best_u):
                # removing `cand` gives utility `u`; the LARGEST dip corresponds
                # to the SMALLEST resulting u (pick_max=True picks that), the
                # SMALLEST dip corresponds to the LARGEST resulting u.
                best_u = u
                best_choice = cand
        remaining.remove(best_choice)
        curve.append((len(remaining), best_u))
        if len(remaining) % 10 == 0:
            print(f"  [{label}] {len(remaining)} remaining, U={best_u:.3f}  "
                  f"({time.time()-t0:.0f}s elapsed)")
    return curve


print(f"\n[part 2a] greedy removal: highest-dip-first (most critical removed first) "
      f"down to {REMOVAL_FLOOR} remaining ...")
curve_highest = greedy_removal(pick_max=True, label="highest_dip_first")

print(f"\n[part 2b] greedy removal: lowest-dip-first (most redundant removed first) "
      f"down to {REMOVAL_FLOOR} remaining ...")
curve_lowest = greedy_removal(pick_max=False, label="lowest_dip_first")

print(f"\n[part 2c] random removal order down to {REMOVAL_FLOOR} remaining ...")
rng = np.random.default_rng(SEED)
rand_order = rng.permutation(N_DETECTORS).tolist()
remaining = list(range(N_DETECTORS))
curve_random = [(N_DETECTORS, full_U)]
t0 = time.time()
for det_to_remove in rand_order:
    if len(remaining) <= REMOVAL_FLOOR:
        break
    remaining.remove(det_to_remove)
    idx_t = torch.tensor(remaining, dtype=torch.long)
    u = eval_U(lbfgs_x[idx_t], lbfgs_y[idx_t])
    curve_random.append((len(remaining), u))
    if len(remaining) % 10 == 0:
        print(f"  [random] {len(remaining)} remaining, U={u:.3f}  ({time.time()-t0:.0f}s elapsed)")

results = dict(
    full_U=full_U,
    leave_one_out=loo_results,
    removal_floor=REMOVAL_FLOOR,
    curve_highest_dip_first=curve_highest,
    curve_lowest_dip_first=curve_lowest,
    curve_random=curve_random,
)
out_path = os.path.join(OUT_DIR, "detector_removal_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    for curve, label, style in (
        (curve_highest, "highest-dip-first (critical removed first)", "-o"),
        (curve_lowest, "lowest-dip-first (redundant removed first)", "-o"),
        (curve_random, "random order", "-o"),
    ):
        n_remaining = [c[0] for c in curve]
        u_vals = [c[1] for c in curve]
        ax.plot(n_remaining, u_vals, style, label=label, markersize=3)
    ax.axhline(full_U, color="gray", linestyle="--", linewidth=0.8, label="full 100-detector U")
    ax.set_xlabel("number of detectors remaining")
    ax.set_ylabel("U")
    ax.set_title(f"Utility vs. number of detectors removed, by removal strategy ({LAYOUT_LABEL})")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "detector_removal_curves.png")
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")
except Exception as exc:
    print(f"[plot] skipped ({exc!r})")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Full 100-detector U: {full_U:.3f}")
for curve, label in ((curve_highest, "highest_dip_first"), (curve_lowest, "lowest_dip_first"),
                     (curve_random, "random")):
    u_at_floor = curve[-1][1]
    print(f"  {label:20s}: U at {REMOVAL_FLOOR} remaining = {u_at_floor:.3f}  "
          f"(drop from full = {full_U - u_at_floor:+.3f})")
