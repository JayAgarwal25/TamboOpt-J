#!/usr/bin/env python3
"""
Dual-panel summary figure for the detector-removal / criticality analysis.

Left panel: leave-one-out dip heatmap over the TRUE mountain footprint
(mountain.centroids_NUE, the actual irregular triangulated surface -- not
the axis-aligned bounding box critical_detector_geometry.py uses for its
edge-distance feature). Built on the ds_center layout (matching
detector_removal_results.json). Since ds_center was also the MCR
consolidation's reference layout (mcr_consolidation.py), its own detector
index IS the reference-slot index directly (perms[ref_idx] = identity), so
per-slot MCR stats can be indexed by ds_center's own detector index with no
further alignment. Color = mean leave-one-out dip across all 12 independent
L-BFGS optima (from the MCR consolidation); marker size/alpha = cross-layout
critical hit-rate (0-12), i.e. how many of the 12 independent layouts had
this same physical slot among their own top-5 critical detectors. Circled
markers are the statistically-validated recurring hotspots (hit-rate >= 3,
per the MCR permutation-null result, p=0.0049).

Right panel: the existing 3-curve removal plot (detector_removal_curves.png's
underlying data), re-annotated with explicit headline retained-utility
callouts at the 20-detector floor.
"""
import os
import sys
import json

import numpy as np
import torch

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import modules_v6  # noqa: F401
from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
from modules_v4.tr_geometry import load_tr_mountain

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
HOTSPOT_HIT_RATE_THRESHOLD = 3   # per MCR permutation-null result (p=0.0049 at this cut)

print("=" * 70)
print("Critical detector overlay summary (true mountain footprint)")
print("=" * 70)

mountain = load_tr_mountain(GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)

layout = torch.load(
    f"{RUN_BASE}/test_v6_run_04_optimize_lbfgs_ensemble_ds_center/layout_best.pt",
    map_location="cpu", weights_only=False)
N_pos = layout["x"].float().reshape(-1).numpy()   # North
E_pos = layout["y"].float().reshape(-1).numpy()   # East
assert N_pos.shape[0] == N_DETECTORS

with open(os.path.join(HERE, "detector_removal_results.json")) as f:
    removal = json.load(f)
full_U = removal["full_U"]

with open(os.path.join(HERE, "mcr_consolidation", "mcr_consolidation_results.json")) as f:
    mcr = json.load(f)
assert mcr["reference_layout"] == "ds_center", (
    "ds_center must be the MCR reference for direct index reuse; "
    f"got {mcr['reference_layout']!r}")
dip_mean_12 = np.array(mcr["per_slot"]["dip_mean"], dtype=np.float64)
hit_rate_12 = np.array(mcr["per_slot"]["critical_hit_rate"], dtype=np.int64)
assert dip_mean_12.shape[0] == N_DETECTORS

hotspot_idx = np.where(hit_rate_12 >= HOTSPOT_HIT_RATE_THRESHOLD)[0]
print(f"Cross-layout validated hotspots (hit-rate >= {HOTSPOT_HIT_RATE_THRESHOLD}/12): "
      f"{len(hotspot_idx)} detectors -> native indices {hotspot_idx.tolist()}")

# ---- headline removal-curve numbers ----
def u_at_floor(curve, floor=20):
    for n, u in curve:
        if n == floor:
            return u
    raise ValueError(f"floor {floor} not found in curve")

floor = removal["removal_floor"]
u_redundant_first = u_at_floor(removal["curve_lowest_dip_first"], floor)
u_critical_first = u_at_floor(removal["curve_highest_dip_first"], floor)
u_random = u_at_floor(removal["curve_random"], floor)
pct_redundant = 100 * u_redundant_first / full_U
pct_critical = 100 * u_critical_first / full_U
pct_random = 100 * u_random / full_U
print(f"Full U={full_U:.2f}; at {floor}-detector floor: redundant-first={u_redundant_first:.1f} "
      f"({pct_redundant:.0f}%), critical-first={u_critical_first:.1f} ({pct_critical:.0f}%), "
      f"random={u_random:.1f} ({pct_random:.0f}%)")

# ---- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6.5))

# Left panel: true mountain footprint + dip overlay.
ax_left.scatter(mountain.centroids_NUE[:, 0], mountain.centroids_NUE[:, 2],
                s=2, c="lightgray", alpha=0.5, linewidth=0, label="mountain footprint (true surface)")

sizes = 40 + 260 * (hit_rate_12 / 12.0)
alphas_norm = 0.35 + 0.65 * (hit_rate_12 / 12.0)
sc = ax_left.scatter(N_pos, E_pos, c=dip_mean_12, cmap="viridis", s=sizes,
                     edgecolor="k", linewidth=0.5, zorder=3)
# matplotlib scatter doesn't support a per-point alpha array directly pre-3.4-safe
# path, so apply it via facecolor RGBA after the fact.
rgba = sc.get_facecolor()
if rgba.shape[0] == 1:
    rgba = np.repeat(rgba, N_DETECTORS, axis=0)
rgba[:, 3] = alphas_norm
sc.set_facecolor(rgba)

ax_left.scatter(N_pos[hotspot_idx], E_pos[hotspot_idx], facecolor="none",
                edgecolor="red", s=sizes[hotspot_idx] + 120, linewidth=2.2, zorder=4,
                label=f"validated hotspot (hit-rate>={HOTSPOT_HIT_RATE_THRESHOLD}/12)")
for i in hotspot_idx:
    ax_left.annotate(f"{hit_rate_12[i]}/12", (N_pos[i], E_pos[i]),
                      xytext=(6, 6), textcoords="offset points", fontsize=8, color="darkred")

cbar = fig.colorbar(sc, ax=ax_left)
cbar.set_label("mean leave-one-out dip across 12 independent optima")
ax_left.set_xlabel("North (m)")
ax_left.set_ylabel("East (m)")
ax_left.set_title("Criticality over the true mountain footprint\n(marker size/opacity = cross-layout hit-rate)")
ax_left.legend(fontsize=7, loc="best")
ax_left.set_aspect("equal", adjustable="datalim")

# Right panel: removal curves, annotated.
for curve, label, color in [
    (removal["curve_lowest_dip_first"], "redundant-first (lowest dip removed first)", "tab:green"),
    (removal["curve_highest_dip_first"], "critical-first (highest dip removed first)", "tab:red"),
    (removal["curve_random"], "random order", "tab:gray"),
]:
    n_arr = [p[0] for p in curve]
    u_arr = [p[1] for p in curve]
    ax_right.plot(n_arr, u_arr, marker="o", markersize=3, label=label, color=color)

ax_right.axvline(floor, color="k", linestyle=":", linewidth=1, alpha=0.6)
ax_right.annotate(f"~{pct_redundant:.0f}% utility retained\nat {floor} detectors,\nredundant-first",
                   (floor, u_redundant_first), xytext=(floor + 8, u_redundant_first + 5),
                   fontsize=9, color="tab:green",
                   arrowprops=dict(arrowstyle="->", color="tab:green"))
ax_right.annotate(f"~{pct_critical:.0f}% remaining\nat {floor} detectors,\ncritical-first",
                   (floor, u_critical_first), xytext=(floor + 8, u_critical_first + 25),
                   fontsize=9, color="tab:red",
                   arrowprops=dict(arrowstyle="->", color="tab:red"))

ax_right.set_xlabel("detectors remaining")
ax_right.set_ylabel("utility U")
ax_right.set_title(f"Detector removal curves (ds_center, full U={full_U:.1f})")
ax_right.legend(fontsize=8, loc="upper left")
ax_right.invert_xaxis()

fig.suptitle("Detector criticality: physical footprint + removal sensitivity (ds_center layout)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = os.path.join(HERE, "critical_detector_overlay_summary.png")
fig.savefig(out_png, dpi=150)
print(f"\n[plot] wrote {out_png}")
