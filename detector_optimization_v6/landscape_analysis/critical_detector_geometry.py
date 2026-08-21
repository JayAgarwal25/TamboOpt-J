#!/usr/bin/env python3
"""
What is physically special about the detectors the leave-one-out sweep
(detector_removal_analysis.py) identified as most critical vs. most
redundant? Pure geometry follow-up -- no FNN/recon forward passes needed,
just the already-saved dip values, the already-saved layout, and the
mountain surface.

For every one of the 100 detectors, computes:
  - Up (elevation), via SurfaceUpMap (East, North) -> Up
  - z_cont (continuous shower-layer depth), via mountain.east_to_z_cont(East)
  - distance from the layout's own centroid (mean North, mean East)
  - distance to its nearest OTHER detector in the same layout (local density)
  - distance to the mountain footprint bounding-box edge (boundary proxy)

Then correlates each feature against the leave-one-out dip across all 100
detectors (Pearson r), and prints the feature values for the specific
detectors flagged as most critical / most redundant, to see whether the
critical few are explained by location (edge vs. interior), elevation,
shower-layer depth, or local sparsity, rather than being unexplainable.
"""
import os, sys, json
import numpy as np
import torch

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import modules_v6  # noqa: F401

from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.tr_surface_map_ne import SurfaceUpMap

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
N_HIGHLIGHT = 5   # how many top critical / redundant detectors to print in detail

print("=" * 70)
print("Critical-detector geometry follow-up")
print("=" * 70)

mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
surface = SurfaceUpMap.from_mountain(mountain)

with open(os.path.join(HERE, "detector_removal_results.json")) as f:
    removal = json.load(f)
dips = np.array(removal["leave_one_out"]["dips"], dtype=np.float64)
assert dips.shape[0] == N_DETECTORS

layout = torch.load(
    f"{RUN_BASE}/test_v6_run_04_optimize_lbfgs_ensemble_ds_combined/layout_best.pt",
    map_location="cpu", weights_only=False)
N_pos = layout["x"].float().reshape(-1)   # North
E_pos = layout["y"].float().reshape(-1)   # East
assert N_pos.shape[0] == N_DETECTORS

with torch.no_grad():
    Up_pos = surface(N_pos, E_pos).numpy()
z_cont = mountain.east_to_z_cont(E_pos.numpy())

N_np, E_np = N_pos.numpy(), E_pos.numpy()
centroid_N, centroid_E = N_np.mean(), E_np.mean()
dist_centroid = np.sqrt((N_np - centroid_N) ** 2 + (E_np - centroid_E) ** 2)

# Nearest OTHER placed detector (local density proxy) -- lower = denser neighborhood.
d2_pairwise = (N_np[:, None] - N_np[None, :]) ** 2 + (E_np[:, None] - E_np[None, :]) ** 2
np.fill_diagonal(d2_pairwise, np.inf)
dist_nearest_other = np.sqrt(d2_pairwise.min(axis=1))

# Distance to the mountain footprint's bounding-box edge (simple boundary proxy;
# smaller = closer to the edge of the surveyed region).
dist_to_edge = np.minimum.reduce([
    N_np - mountain.n_min, mountain.n_max - N_np,
    E_np - mountain.east_lo, mountain.east_hi - E_np,
])

features = dict(
    Up=Up_pos, z_cont=z_cont, dist_from_centroid=dist_centroid,
    dist_nearest_other_detector=dist_nearest_other, dist_to_bbox_edge=dist_to_edge,
)

print(f"\nLeave-one-out dip stats: mean={dips.mean():.4f} std={dips.std():.4f} "
      f"min={dips.min():.4f} max={dips.max():.4f}")

print("\n[correlations] Pearson r between leave-one-out dip and each geometric feature "
      f"(across all {N_DETECTORS} detectors):")
correlations = {}
for name, vals in features.items():
    r = float(np.corrcoef(dips, vals)[0, 1])
    correlations[name] = r
    print(f"  {name:30s}: r = {r:+.3f}")

order_by_dip = np.argsort(dips)  # ascending: most redundant first, most critical last
redundant_idx = order_by_dip[:N_HIGHLIGHT]
critical_idx = order_by_dip[-N_HIGHLIGHT:][::-1]  # most critical first

def _print_group(label, idxs):
    print(f"\n  {label}:")
    for i in idxs:
        i = int(i)
        print(f"    det {i:3d}  dip={dips[i]:+.4f}  N={N_np[i]:+8.1f}  E={E_np[i]:+8.1f}  "
              f"Up={Up_pos[i]:7.1f}  z_cont={z_cont[i]:5.2f}  "
              f"dist_centroid={dist_centroid[i]:7.1f}  nearest_other={dist_nearest_other[i]:6.1f}  "
              f"dist_edge={dist_to_edge[i]:7.1f}")

print(f"\n[highlight] top {N_HIGHLIGHT} MOST CRITICAL detectors (largest dip):")
_print_group("critical", critical_idx)
print(f"\n[highlight] top {N_HIGHLIGHT} MOST REDUNDANT detectors (smallest/most negative dip):")
_print_group("redundant", redundant_idx)

# Group-mean comparison: critical-5 vs redundant-5 vs the remaining 90, per feature.
rest_idx = np.setdiff1d(np.arange(N_DETECTORS), np.concatenate([critical_idx, redundant_idx]))
print(f"\n[group means] critical-{N_HIGHLIGHT} vs redundant-{N_HIGHLIGHT} vs remaining "
      f"{len(rest_idx)}:")
for name, vals in features.items():
    c_mean = vals[critical_idx].mean()
    r_mean = vals[redundant_idx].mean()
    o_mean = vals[rest_idx].mean()
    print(f"  {name:30s}: critical={c_mean:8.2f}  redundant={r_mean:8.2f}  rest={o_mean:8.2f}")

results = dict(
    dips=dips.tolist(),
    features={k: v.tolist() for k, v in features.items()},
    correlations=correlations,
    critical_idx=critical_idx.tolist(),
    redundant_idx=redundant_idx.tolist(),
)
out_path = os.path.join(HERE, "critical_detector_geometry_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(N_np, E_np, c=dips, cmap="RdYlGn_r", s=60, edgecolor="k", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="leave-one-out dip")
    ax.scatter(N_np[critical_idx], E_np[critical_idx], facecolor="none",
               edgecolor="blue", s=200, linewidth=2, label="most critical")
    ax.scatter(N_np[redundant_idx], E_np[redundant_idx], facecolor="none",
               edgecolor="black", s=200, linewidth=2, marker="s", label="most redundant")
    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_title("Leave-one-out dip by detector position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(HERE, "critical_detector_geometry.png")
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")
except Exception as exc:
    print(f"[plot] skipped ({exc!r})")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
strongest = max(correlations, key=lambda k: abs(correlations[k]))
print(f"  Strongest correlate of dip: {strongest} (r={correlations[strongest]:+.3f})")
print("  (|r| < 0.3 for all features would mean location/elevation/depth/density alone "
        "don't explain criticality -- it may depend on the shower-population structure "
        "the utility function weights, not simple geometry.)")
