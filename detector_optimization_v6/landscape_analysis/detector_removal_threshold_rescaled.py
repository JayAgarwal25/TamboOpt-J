#!/usr/bin/env python3
"""
Consolidation follow-up to detector_removal_analysis.py: how much of the
steep U collapse near the 20-detector removal floor is genuine information
loss (redundancy running out) vs. an artifact of the utility function's
reconstructability gate using an ABSOLUTE, not count-scaled, minimum-firing
bar (RECONSTRUCT_THRESHOLD=10, opt_core.py, unchanged from its n_det=100
production value regardless of how many detectors are actually active)?

Repeats the exact same leave-one-out sweep and three removal sequences as
detector_removal_analysis.py, but with the reconstructability threshold
rescaled proportionally to the remaining detector count at each step:
    threshold(n_active) = RECONSTRUCT_THRESHOLD_BASE * (n_active / N_DETECTORS)
so that "enough detectors fired" always means the same FRACTION of whatever
count is currently active, not the same absolute count. This isolates
whether the sharp collapse near the floor is really about specific detectors
being irreplaceable, or just the layout falling below an absolute headcount
the utility function was calibrated for at n=100.

Reuses opt_core.utility_of_xy's `reconstruct_threshold` override parameter
(added specifically to support this ablation without duplicating the
utility computation).
"""
import os, sys, json, time
import numpy as np
import torch

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (layout paths live in one place)
import modules_v6  # noqa: F401

from modules_v6.constants import N_DETECTORS, TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER
from modules_v6.opt_core import utility_of_xy, load_models, RECONSTRUCT_THRESHOLD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
BATCH_SEED_BASE = 1000     # match the original run exactly, for apples-to-apples comparison
BATCH_SIZE = 512
N_BATCHES = 3
REMOVAL_FLOOR = 20
SEED = 123

print("=" * 70)
print("Detector removal / redundancy analysis -- threshold-rescaled ablation")
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
    """Same as the original script's eval_U, except reconstruct_threshold is
    rescaled to n_active/N_DETECTORS instead of staying fixed at the
    production absolute value."""
    n_active = x.shape[0]
    threshold = RECONSTRUCT_THRESHOLD * (n_active / N_DETECTORS)
    Us = [float(utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon,
                               reconstruct_threshold=threshold)[0].item()) for p in BATCHES]
    return float(np.mean(Us))


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


lbfgs_x, lbfgs_y, lbfgs_U_saved = load_layout(
    _layouts.primary())
full_U = eval_U(lbfgs_x, lbfgs_y)
print(f"L-BFGS best U (saved): {lbfgs_U_saved:.4f}  (re-evaluated, {N_BATCHES} fresh batches, "
      f"threshold={RECONSTRUCT_THRESHOLD:.2f}): {full_U:.4f}")

# ── Part 1: leave-one-out sweep (sanity check -- should barely change at n=99) ──
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
order_by_dip = np.argsort(dips)
print(f"  dip stats: min={dips.min():.4f}  max={dips.max():.4f}  mean={dips.mean():.4f}  "
      f"std={dips.std():.4f}")

loo_results = dict(full_U=full_U, dips=dips.tolist(), order_by_dip_ascending=order_by_dip.tolist())

# ── Part 2: three removal sequences, same structure as the original script ──
def greedy_removal(pick_max: bool, label: str):
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
                best_u = u
                best_choice = cand
        remaining.remove(best_choice)
        curve.append((len(remaining), best_u))
        if len(remaining) % 10 == 0:
            print(f"  [{label}] {len(remaining)} remaining, U={best_u:.3f}  "
                  f"({time.time()-t0:.0f}s elapsed)")
    return curve


print(f"\n[part 2a] greedy removal: highest-dip-first, threshold rescaled, "
      f"down to {REMOVAL_FLOOR} remaining ...")
curve_highest = greedy_removal(pick_max=True, label="highest_dip_first")

print(f"\n[part 2b] greedy removal: lowest-dip-first, threshold rescaled, "
      f"down to {REMOVAL_FLOOR} remaining ...")
curve_lowest = greedy_removal(pick_max=False, label="lowest_dip_first")

print(f"\n[part 2c] random removal order, threshold rescaled, down to {REMOVAL_FLOOR} remaining ...")
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
out_path = os.path.join(HERE, "detector_removal_threshold_rescaled_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(HERE, "detector_removal_results.json")) as f:
        original = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))
    for curve, label, style, color in (
        (curve_highest, "highest-dip-first, RESCALED threshold", "-o", "C0"),
        (curve_lowest, "lowest-dip-first, RESCALED threshold", "-o", "C1"),
        (curve_random, "random, RESCALED threshold", "-o", "C2"),
    ):
        n_remaining = [c[0] for c in curve]
        u_vals = [c[1] for c in curve]
        ax.plot(n_remaining, u_vals, style, label=label, markersize=3, color=color)
    for key, label, color in (
        ("curve_highest_dip_first", "highest-dip-first, ORIGINAL (fixed threshold)", "C0"),
        ("curve_lowest_dip_first", "lowest-dip-first, ORIGINAL (fixed threshold)", "C1"),
        ("curve_random", "random, ORIGINAL (fixed threshold)", "C2"),
    ):
        curve_orig = original[key]
        n_remaining = [c[0] for c in curve_orig]
        u_vals = [c[1] for c in curve_orig]
        ax.plot(n_remaining, u_vals, "--", label=label, markersize=3, color=color, alpha=0.5)
    ax.set_xlabel("number of detectors remaining")
    ax.set_ylabel("U")
    ax.set_title("Threshold-rescaled vs. original removal curves")
    ax.invert_xaxis()
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_png = os.path.join(HERE, "detector_removal_threshold_rescaled_curves.png")
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")
except Exception as exc:
    print(f"[plot] skipped ({exc!r})")

print("\n" + "=" * 70)
print("SUMMARY / COMPARISON WITH ORIGINAL (FIXED-THRESHOLD) RUN")
print("=" * 70)
try:
    with open(os.path.join(HERE, "detector_removal_results.json")) as f:
        original = json.load(f)
    print(f"Full 100-detector U: rescaled={full_U:.3f}  original={original['full_U']:.3f}")
    for curve, key, label in (
        (curve_highest, "curve_highest_dip_first", "highest_dip_first"),
        (curve_lowest, "curve_lowest_dip_first", "lowest_dip_first"),
        (curve_random, "curve_random", "random"),
    ):
        u_at_floor_rescaled = curve[-1][1]
        u_at_floor_original = original[key][-1][1]
        print(f"  {label:20s}: U at {REMOVAL_FLOOR} remaining -- "
              f"rescaled={u_at_floor_rescaled:.3f}  original={u_at_floor_original:.3f}  "
              f"difference={u_at_floor_rescaled - u_at_floor_original:+.3f}")
    print("\n  A large positive difference (rescaled >> original) at the floor means the "
          "original run's sharp collapse there was substantially inflated by the ABSOLUTE "
          "reconstructability threshold, not purely genuine information loss. A small "
          "difference means the collapse is mostly real redundancy running out.")
except FileNotFoundError:
    print("  (original detector_removal_results.json not found -- comparison skipped)")
