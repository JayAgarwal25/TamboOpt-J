#!/usr/bin/env python3
"""
Grid version of critical_detector_geometry.png's style, extended across all
12 independent L-BFGS optima. Produces TWO separate 3x4 grids (one subplot
per layout each):

  - critical_detector_geometry_grid_critical.png: all 100 detectors colored
    by that layout's own leave-one-out dip (RdYlGn_r, same colormap as the
    original single-layout plot), with only the top-5 MOST CRITICAL
    detectors circled (blue).
  - critical_detector_geometry_grid_redundant.png: same base scatter, with
    only the top-5 MOST REDUNDANT detectors circled (black squares).

No new GPU/model evaluation -- reuses each layout's already-saved
leave-one-out dips and layout_best.pt positions.
"""
import os
import sys
import json

import numpy as np
import torch

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import modules_v6  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
N_HIGHLIGHT = 5            # critical detectors marked
N_REDUNDANT_HIGHLIGHT = 20  # redundant detectors marked (more, to reveal clustering)

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


def load_dips(json_path):
    with open(json_path) as f:
        data = json.load(f)
    if "grid_dips" in data:
        return np.asarray(data["grid_dips"], dtype=np.float64)
    return np.asarray(data["leave_one_out"]["dips"], dtype=np.float64)


print("=" * 70)
print("Critical / redundant detector geometry grid (12 layouts)")
print("=" * 70)

per_layout = []
for tag, run_suffix, json_path in LAYOUTS:
    xy = load_layout_xy(run_suffix)
    dips = load_dips(json_path)
    order = np.argsort(dips)
    redundant_idx = order[:N_REDUNDANT_HIGHLIGHT]
    critical_idx = order[-N_HIGHLIGHT:][::-1]
    per_layout.append(dict(tag=tag, xy=xy, dips=dips,
                           critical_idx=critical_idx, redundant_idx=redundant_idx))
    print(f"  {tag:<16} dip mean/std/max = {dips.mean():.3f}/{dips.std():.3f}/{dips.max():.3f}  "
          f"critical={critical_idx.tolist()}  redundant={redundant_idx.tolist()}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_grid(mode, out_name, title):
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 15), constrained_layout=True)
    for i, (ax, info) in enumerate(zip(axes.flat, per_layout)):
        row, col = divmod(i, n_cols)
        xy, dips = info["xy"], info["dips"]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=dips, cmap="RdYlGn_r", s=55,
                         edgecolor="k", linewidth=0.4, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("leave-one-out dip", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        if mode == "critical":
            idx = info["critical_idx"]
            ax.scatter(xy[idx, 0], xy[idx, 1], facecolor="none", edgecolor="blue",
                       s=220, linewidth=2, zorder=3, label="most critical")
        else:
            idx = info["redundant_idx"]
            ax.scatter(xy[idx, 0], xy[idx, 1], facecolor="none", edgecolor="black",
                       s=140, linewidth=1.6, marker="s", zorder=3, label="most redundant")
        ax.set_title(info["tag"], fontsize=10)
        ax.set_xlabel("North (m)", fontsize=9)
        ax.set_ylabel("East (m)", fontsize=9, labelpad=2)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=14)
    out_path = os.path.join(HERE, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"[plot] wrote {out_path}")


build_grid("critical", "critical_detector_geometry_grid_critical.png",
           f"Top-{N_HIGHLIGHT} MOST CRITICAL detectors (blue circles), all 12 layouts, "
           f"colored by own leave-one-out dip")
build_grid("redundant", "critical_detector_geometry_grid_redundant.png",
           f"Top-{N_REDUNDANT_HIGHLIGHT} MOST REDUNDANT detectors (black squares), all 12 layouts, "
           f"colored by own leave-one-out dip")
