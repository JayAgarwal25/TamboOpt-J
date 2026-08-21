#!/usr/bin/env python3
"""
Clear categorical (not heatmap) segregation of detectors into "removable"
vs. "keep", one subplot per independent L-BFGS optimum, laid out as a grid.

For each of the 12 layouts, uses that layout's own ACTUAL redundant-first
(lowest-dip-first) removal curve (already saved in its detector_removal_
results.json, no new GPU/model evaluation needed) to find the largest
number of detectors removable while utility stays within a tolerance of the
original -- i.e. "how many can go before it actually starts to hurt." That
count sets the split point; the removable/keep IDENTITY assignment uses the
same layout's static leave-one-out dip ranking (ascending: lowest dip =
first to be removed), which is what the greedy removal curve is seeded from.

This directly answers: which ~60-70 (varies per layout) detectors could be
removed while retaining almost the same utility, shown as two clearly
distinct marker groups rather than a continuous color scale.
"""
import os
import sys
import json

import numpy as np
import torch

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (input/output locations)
import modules_v6  # noqa: F401

# Results live beside the other run outputs, not next to the code.
HERE = _layouts.results_dir()
RUN_BASE = _layouts.RUNS
TOLERANCE = 0.99   # "almost the same utility" = within 1% of full U

LAYOUTS = [
    ("ds_center", "ds_center",
     os.path.join(HERE, "detector_removal_results.json")),
    ("ds_grid", "ds_grid",
     os.path.join(HERE, "detector_removal_cross_layout_check_results.json")),
]
for seed in (101, 202, 303, 404, 505):
    for scheme in ("grid", "center"):
        tag = f"ds_seed{seed}_{scheme}"
        LAYOUTS.append((
            tag, tag,
            os.path.join(HERE, "other_optimizers", f"seed{seed}_{scheme}", "detector_removal_results.json"),
        ))
assert len(LAYOUTS) == 12


def load_layout_xy(run_suffix):
    path = os.path.join(RUN_BASE, f"test_v6_run_04_optimize_lbfgs_ensemble_{run_suffix}", "layout_best.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    x = d["x"].float().reshape(-1).numpy()
    y = d["y"].float().reshape(-1).numpy()
    return np.stack([x, y], axis=-1)


def load_removal_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    # ds_grid uses the cross-layout-check schema (grid_dips / grid_full_U /
    # grid_curve_lowest_dip_first); everything else uses the standard schema.
    if "grid_dips" in data:
        dips = np.asarray(data["grid_dips"], dtype=np.float64)
        full_U = data["grid_full_U"]
        curve = data["grid_curve_lowest_dip_first"]
    else:
        dips = np.asarray(data["leave_one_out"]["dips"], dtype=np.float64)
        full_U = data["full_U"]
        curve = data["curve_lowest_dip_first"]
    return dips, full_U, curve


def find_removable_count(curve, full_U, tolerance):
    """Largest number of detectors removable (100 - n_remaining) while U
    stays >= tolerance * full_U, per the ACTUAL redundant-first curve."""
    threshold = tolerance * full_U
    curve_sorted = sorted(curve, key=lambda p: -p[0])  # descending n_remaining
    last_ok_n = curve_sorted[0][0]
    for n_remaining, u in curve_sorted:
        if u >= threshold:
            last_ok_n = n_remaining
        else:
            break
    return 100 - last_ok_n, last_ok_n


print("=" * 70)
print(f"Detector segregation grid (removable vs. keep, tolerance={TOLERANCE})")
print("=" * 70)

per_layout = []
for tag, run_suffix, json_path in LAYOUTS:
    xy = load_layout_xy(run_suffix)
    dips, full_U, curve = load_removal_data(json_path)
    removable_count, k_remaining = find_removable_count(curve, full_U, TOLERANCE)
    order = np.argsort(dips)  # ascending: lowest dip (most removable) first
    removable_idx = order[:removable_count]
    keep_idx = order[removable_count:]
    per_layout.append(dict(tag=tag, xy=xy, dips=dips, full_U=full_U,
                            removable_count=removable_count, k_remaining=k_remaining,
                            removable_idx=removable_idx, keep_idx=keep_idx))
    print(f"  {tag:<16} full_U={full_U:7.2f}  removable={removable_count:3d}  "
          f"keep={k_remaining:3d}  (within {TOLERANCE*100:.0f}% of full U)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for ax, info in zip(axes.flat, per_layout):
    xy, removable_idx, keep_idx = info["xy"], info["removable_idx"], info["keep_idx"]
    ax.scatter(xy[removable_idx, 0], xy[removable_idx, 1], c="tab:green", s=45,
               edgecolor="k", linewidth=0.4, label="removable", zorder=2)
    ax.scatter(xy[keep_idx, 0], xy[keep_idx, 1], c="tab:red", s=45,
               edgecolor="k", linewidth=0.4, marker="^", label="keep", zorder=3)
    ax.set_title(f"{info['tag']}\n{info['removable_count']} removable / "
                 f"{info['k_remaining']} keep (U={info['full_U']:.1f})", fontsize=10)
    ax.set_xlabel("North (m)", fontsize=8)
    ax.set_ylabel("East (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")

axes.flat[0].legend(fontsize=8, loc="best")
fig.suptitle(f"Detector segregation: removable (green) vs. keep (red triangles), "
             f"per-layout split at {TOLERANCE*100:.0f}% utility retention", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = os.path.join(HERE, "critical_detector_segregation_grid.png")
fig.savefig(out_png, dpi=150)
print(f"\n[plot] wrote {out_png}")

results = dict(
    tolerance=TOLERANCE,
    layouts=[dict(tag=info["tag"], full_U=info["full_U"],
                   removable_count=info["removable_count"], k_remaining=info["k_remaining"],
                   removable_idx=info["removable_idx"].tolist(), keep_idx=info["keep_idx"].tolist())
             for info in per_layout],
)
out_json = os.path.join(HERE, "critical_detector_segregation_grid_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"[data] wrote {out_json}")
