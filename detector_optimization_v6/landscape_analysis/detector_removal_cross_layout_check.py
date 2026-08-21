#!/usr/bin/env python3
"""
Cross-layout check: is the "small critical core" found by detector_removal_analysis.py
a property of detector redundancy in general, or an artifact of the one specific layout
(center-scheme, saved as "_combined" -- confirmed byte-identical: same U, same x/y) that
was analyzed?

mode_connectivity.py already established that grid-scheme and center-scheme are two
genuinely different physical arrangements (563m apart per detector after Hungarian
alignment) achieving statistically equivalent utility (U=205.9 vs 208.9 saved). Since the
plateau is degenerate, a different arrangement could solve the "which regions need
coverage" problem with an entirely different set of detectors -- there is no reason the
same DETECTOR INDEX, or even the same PHYSICAL LOCATION, should be critical in both.

This script repeats the exact same leave-one-out + 3-removal-sequence experiment as
detector_removal_analysis.py, but on the grid-scheme layout, then:
  1. compares whether grid-scheme ALSO shows a similarly concentrated, non-uniform dip
     distribution (tests whether "a small critical core exists" generalizes as a
     qualitative pattern, independent of which specific layout you look at)
  2. Hungarian-aligns grid-scheme to center-scheme (reusing opt_core.align_to_reference,
     the same approach mode_connectivity.py uses) and checks whether grid-scheme's
     critical detectors sit physically close to where center-scheme's own critical
     detectors are (a "same region needs coverage, different specific detector solves
     it" story) or in entirely different neighborhoods (criticality is tied to the whole
     arrangement, with no fixed physical location that "needs" a detector).
"""
import os, sys, json, time
import numpy as np
import torch

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (layout paths live in one place)
import modules_v6  # noqa: F401

from modules_v6.constants import N_DETECTORS, TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER
from modules_v6.opt_core import utility_of_xy, load_models, align_to_reference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Results live beside the other run outputs, not next to the code.
HERE = _layouts.results_dir()
BATCH_SEED_BASE = 1000     # match the original detector_removal_analysis.py exactly
BATCH_SIZE = 512
N_BATCHES = 3
REMOVAL_FLOOR = 20
SEED = 123
N_HIGHLIGHT = 5

print("=" * 70)
print("Cross-layout check: does grid-scheme have the SAME critical detectors as center-scheme?")
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
    Us = [float(utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon)[0].item()) for p in BATCHES]
    return float(np.mean(Us))


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


grid_x, grid_y, grid_U_saved = load_layout(
    _layouts.primary())
full_U = eval_U(grid_x, grid_y)
print(f"grid-scheme U (saved): {grid_U_saved:.4f}  (re-evaluated, {N_BATCHES} fresh batches): {full_U:.4f}")

# ── Leave-one-out sweep on grid-scheme (Part 1 only -- the fast, cheap part) ──
print(f"\n[part 1] leave-one-out sweep on GRID-SCHEME layout ({N_DETECTORS} evals) ...")
t0 = time.time()
dips = np.zeros(N_DETECTORS, dtype=np.float64)
for i in range(N_DETECTORS):
    mask = torch.ones(N_DETECTORS, dtype=torch.bool)
    mask[i] = False
    u_without = eval_U(grid_x[mask], grid_y[mask])
    dips[i] = full_U - u_without
print(f"[part 1] done in {time.time()-t0:.0f}s")
order_by_dip = np.argsort(dips)
print(f"  dip stats: min={dips.min():.4f}  max={dips.max():.4f}  mean={dips.mean():.4f}  "
      f"std={dips.std():.4f}  (center-scheme was: mean=0.2980 std=0.5334)")

grid_critical = order_by_dip[-N_HIGHLIGHT:][::-1]
grid_redundant = order_by_dip[:N_HIGHLIGHT]
print(f"  grid-scheme most CRITICAL {N_HIGHLIGHT}: "
      f"{[(int(i), round(float(dips[i]), 4)) for i in grid_critical]}")
print(f"  grid-scheme most REDUNDANT {N_HIGHLIGHT}: "
      f"{[(int(i), round(float(dips[i]), 4)) for i in grid_redundant]}")

# ── Part 2: three removal sequences, same structure as the original script ──
def greedy_removal(pick_max: bool, label: str):
    remaining = list(range(N_DETECTORS))
    curve = [(N_DETECTORS, full_U)]
    t0 = time.time()
    while len(remaining) > REMOVAL_FLOOR:
        best_choice, best_u = None, None
        for cand in remaining:
            trial = [j for j in remaining if j != cand]
            idx_t = torch.tensor(trial, dtype=torch.long)
            u = eval_U(grid_x[idx_t], grid_y[idx_t])
            if best_u is None or (pick_max and u < best_u) or (not pick_max and u > best_u):
                best_u, best_choice = u, cand
        remaining.remove(best_choice)
        curve.append((len(remaining), best_u))
        if len(remaining) % 10 == 0:
            print(f"  [{label}] {len(remaining)} remaining, U={best_u:.3f}  ({time.time()-t0:.0f}s elapsed)")
    return curve


print(f"\n[part 2a] greedy removal: highest-dip-first, GRID-SCHEME, down to {REMOVAL_FLOOR} remaining ...")
curve_highest = greedy_removal(pick_max=True, label="highest_dip_first")
print(f"\n[part 2b] greedy removal: lowest-dip-first, GRID-SCHEME, down to {REMOVAL_FLOOR} remaining ...")
curve_lowest = greedy_removal(pick_max=False, label="lowest_dip_first")

print(f"\n[part 2c] random removal order, GRID-SCHEME, down to {REMOVAL_FLOOR} remaining ...")
rng = np.random.default_rng(SEED)
rand_order = rng.permutation(N_DETECTORS).tolist()
remaining = list(range(N_DETECTORS))
curve_random = [(N_DETECTORS, full_U)]
for det_to_remove in rand_order:
    if len(remaining) <= REMOVAL_FLOOR:
        break
    remaining.remove(det_to_remove)
    idx_t = torch.tensor(remaining, dtype=torch.long)
    u = eval_U(grid_x[idx_t], grid_y[idx_t])
    curve_random.append((len(remaining), u))

print(f"\n[comparison] U at {REMOVAL_FLOOR}-detector floor:")
print(f"  GRID-SCHEME:   highest_dip_first={curve_highest[-1][1]:.3f}  "
      f"lowest_dip_first={curve_lowest[-1][1]:.3f}  random={curve_random[-1][1]:.3f}")
print(f"  CENTER-SCHEME (original run): highest_dip_first=13.756  lowest_dip_first=148.829  random=96.243")

# ── Cross-layout spatial comparison: Hungarian-align grid-scheme to center-scheme ──
center_x, center_y, _ = load_layout(
    _layouts.secondary())
with open(os.path.join(HERE, "detector_removal_results.json")) as f:
    center_results = json.load(f)
center_dips = np.array(center_results["leave_one_out"]["dips"], dtype=np.float64)
center_order = np.argsort(center_dips)
center_critical = center_order[-N_HIGHLIGHT:][::-1]
center_redundant = center_order[:N_HIGHLIGHT]

xy_stack = np.stack([
    np.stack([center_x.numpy(), center_y.numpy()], axis=-1),
    np.stack([grid_x.numpy(), grid_y.numpy()], axis=-1),
], axis=0)  # (2, n_det, 2), center-scheme is the reference (ref_idx=0)
aligned, perms = align_to_reference(xy_stack, ref_idx=0)
grid_x_aligned = aligned[1, :, 0]
grid_y_aligned = aligned[1, :, 1]

center_pos = np.stack([center_x.numpy(), center_y.numpy()], axis=-1)   # (100, 2)
grid_pos_aligned = np.stack([grid_x_aligned, grid_y_aligned], axis=-1)  # (100, 2), same ref frame now

def nearest_dist(query_pos, pool_pos):
    d2 = ((query_pos[:, None, :] - pool_pos[None, :, :]) ** 2).sum(-1)
    return np.sqrt(d2.min(axis=1))

# For each layout's own critical/redundant detectors, distance (in the shared, aligned
# reference frame) to the NEAREST detector in the OTHER layout: small = "something in the
# other layout covers roughly the same spot"; large = "this specific spot is only covered
# in this one layout."
center_critical_pos = center_pos[center_critical]
center_redundant_pos = center_pos[center_redundant]
grid_critical_pos = grid_pos_aligned[grid_critical]
grid_redundant_pos = grid_pos_aligned[grid_redundant]

dist_center_critical_to_grid = nearest_dist(center_critical_pos, grid_pos_aligned)
dist_grid_critical_to_center = nearest_dist(grid_critical_pos, center_pos)
dist_center_redundant_to_grid = nearest_dist(center_redundant_pos, grid_pos_aligned)
dist_grid_redundant_to_center = nearest_dist(grid_redundant_pos, center_pos)
# Baseline: typical nearest-neighbor distance for ANY detector in one layout to the other
# (not just the critical/redundant ones), so we know what "close" vs "far" means here.
baseline_all = nearest_dist(center_pos, grid_pos_aligned)

print("\n" + "=" * 70)
print("CROSS-LAYOUT SPATIAL COMPARISON (after Hungarian alignment)")
print("=" * 70)
print(f"  baseline: mean nearest-neighbor distance, ANY center-scheme detector -> "
      f"nearest grid-scheme detector: {baseline_all.mean():.1f}m (std={baseline_all.std():.1f}m)")
print(f"  center-scheme CRITICAL detectors -> nearest grid-scheme detector: "
      f"mean={dist_center_critical_to_grid.mean():.1f}m  {dist_center_critical_to_grid.round(1).tolist()}")
print(f"  center-scheme REDUNDANT detectors -> nearest grid-scheme detector: "
      f"mean={dist_center_redundant_to_grid.mean():.1f}m  {dist_center_redundant_to_grid.round(1).tolist()}")
print(f"  grid-scheme CRITICAL detectors -> nearest center-scheme detector: "
      f"mean={dist_grid_critical_to_center.mean():.1f}m  {dist_grid_critical_to_center.round(1).tolist()}")
print(f"  grid-scheme REDUNDANT detectors -> nearest center-scheme detector: "
      f"mean={dist_grid_redundant_to_center.mean():.1f}m  {dist_grid_redundant_to_center.round(1).tolist()}")
print("\n  If critical detectors' cross-layout distance is close to (or below) baseline, "
      "the OTHER layout still has something covering roughly that spot (criticality "
      "reflects a physical region, solved differently by each layout). If it's well "
      "ABOVE baseline, that region is uniquely covered by only one of the two layouts.")

results = dict(
    grid_full_U=full_U,
    grid_dips=dips.tolist(),
    grid_critical=grid_critical.tolist(),
    grid_redundant=grid_redundant.tolist(),
    grid_curve_highest_dip_first=curve_highest,
    grid_curve_lowest_dip_first=curve_lowest,
    grid_curve_random=curve_random,
    center_critical=center_critical.tolist(),
    center_redundant=center_redundant.tolist(),
    baseline_nearest_neighbor_cross_layout_mean_m=float(baseline_all.mean()),
    baseline_nearest_neighbor_cross_layout_std_m=float(baseline_all.std()),
    dist_center_critical_to_grid_m=dist_center_critical_to_grid.tolist(),
    dist_center_redundant_to_grid_m=dist_center_redundant_to_grid.tolist(),
    dist_grid_critical_to_center_m=dist_grid_critical_to_center.tolist(),
    dist_grid_redundant_to_center_m=dist_grid_redundant_to_center.tolist(),
)
out_path = os.path.join(HERE, "detector_removal_cross_layout_check_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(center_pos[:, 1], center_pos[:, 0], s=30, c="lightgray", label="center-scheme (all)")
    ax.scatter(grid_pos_aligned[:, 1], grid_pos_aligned[:, 0], s=30, c="lightblue",
               marker="^", label="grid-scheme (all, aligned)")
    ax.scatter(center_critical_pos[:, 1], center_critical_pos[:, 0], s=150, c="red",
               edgecolor="black", label="center-scheme critical")
    ax.scatter(grid_critical_pos[:, 1], grid_critical_pos[:, 0], s=150, c="blue",
               marker="^", edgecolor="black", label="grid-scheme critical")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Critical detectors: center-scheme vs. grid-scheme (aligned)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_png = os.path.join(HERE, "detector_removal_cross_layout_check.png")
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")
except Exception as exc:
    print(f"[plot] skipped ({exc!r})")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  grid-scheme dip distribution: mean={dips.mean():.4f} std={dips.std():.4f}  "
      f"(center-scheme: mean=0.2980 std=0.5334) -- {'SIMILARLY concentrated' if dips.std() > 2*abs(dips.mean()) else 'different shape'}")
print(f"  grid-scheme critical detectors (own indices): {grid_critical.tolist()}")
print(f"  center-scheme critical detectors (own indices): {center_critical.tolist()}")
print(f"  mean cross-layout distance, critical detectors: "
      f"center->grid={dist_center_critical_to_grid.mean():.1f}m  grid->center={dist_grid_critical_to_center.mean():.1f}m")
print(f"  baseline (any detector) cross-layout distance: {baseline_all.mean():.1f}m")
