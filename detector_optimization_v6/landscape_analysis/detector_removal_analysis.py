#!/usr/bin/env python3
"""
How much does each detector actually matter, and how much redundancy is there
among the 100?

Part 1, leave-one-out: remove each detector in turn and measure the utility dip
against the full layout. Valid rather than approximate, because the DeepSets
pooling is permutation- and count-invariant, so 99 detectors is a genuine
forward pass.

Part 2, three removal sequences from 100 down to a floor of 20. "highest_dip
first" strips the most critical remaining detector at each step, re-evaluated
fresh rather than sorted once; "lowest_dip_first" strips the most redundant;
"random" uses one fixed shuffle. If the plateau really comes from redundancy,
lowest-first should strip many detectors before U moves while highest-first
craters, which tests WHY the landscape is flat rather than restating that it is.

--layout_path and --layout_tag select which saved layout to analyse; a tag sends
output to other_optimizers/<tag>/.
"""
import argparse
import os, sys, json, time
import numpy as np
import torch

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (layout paths live in one place)
from common import Scorer, N_DETECTORS, TRAINING_DATASET_FOLDER


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
# Results live beside the other run outputs, not next to the code.
HERE = _layouts.results_dir()
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

sc = Scorer(n_batches=N_BATCHES, batch_size=BATCH_SIZE, seed_base=BATCH_SEED_BASE)
fnn, recon = sc.fnn, sc.recon

primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                         weights_only=False).float()


fresh_batch = sc.draw

BATCHES = sc.batches


eval_U = sc.U

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
