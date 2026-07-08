#!/usr/bin/env python3
"""
Regenerates 3D surface plots (and their heatmap companions) for results that
predate two additions to the source scripts:
  1. 3D `plot_surface` companions (detector_grid_scan.py, full_space_2d_slice.py,
     full_space_2d_slice_fine.py all produce them automatically now).
  2. A SHARED color/z scale across every panel that's meant to be compared
     side by side, instead of each panel auto-scaling to its own range --
     otherwise a panel with a genuinely tiny U range (e.g. one detector moved,
     +/-1 unit) can look just as "structured" as one with a large range (e.g.
     all 100 detectors moved, +/-25 units), which is misleading.

Pure post-processing from already-saved *_results.json -- no model/GPU
computation, no rerun of the underlying experiments.

Grouping used for "comparable":
  - detector_grid_results.json: center + edge share one scale (2 panels).
  - full_space_2d_slice_results.json: its own 4 panels share one scale.
  - full_space_2d_slice_fine_results.json: its own 4 panels share one scale.
  - ALSO: coarse (400m) + fine (100m) full-space slice results (8 panels
    total) share ONE combined scale, specifically so the "does the surface
    get 4x smoother when the step range shrinks 4x, or does real sub-100m
    roughness survive" comparison is visually honest, not an artifact of two
    independently-scaled color bars.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def make_grid_scan_plots():
    with open(os.path.join(HERE, "detector_grid_results.json")) as f:
        data = json.load(f)
    u_min = min(r["U_min"] if "U_min" in r else min(min(row) for row in r["U_grid"]) for r in data.values())
    u_max = max(r["U_max"] if "U_max" in r else max(max(row) for row in r["U_grid"]) for r in data.values())
    print(f"[detector_grid_scan] shared scale: [{u_min:.3f}, {u_max:.3f}]")

    for tag, r in data.items():
        n_grid = np.array(r["n_grid"])
        e_grid = np.array(r["e_grid"])
        U_grid_2d = np.array(r["U_grid"])

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(e_grid, n_grid, U_grid_2d, shading="auto", cmap="viridis",
                            vmin=u_min, vmax=u_max)
        plt.colorbar(im, ax=ax, label="U (this detector swept, other 99 fixed)")
        ax.scatter([r["orig_E"]], [r["orig_N"]], marker="*", s=250, c="red",
                   edgecolor="black", label="optimized position")
        ax.scatter([r["argmax_E"]], [r["argmax_N"]], marker="X", s=150, c="cyan",
                   edgecolor="black", label="grid argmax")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title(f"U vs. position of detector {r['idx']} ({tag})")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(HERE, f"detector_grid_{tag}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {out_png}")

        E_mesh, N_mesh = np.meshgrid(e_grid, n_grid)
        fig3d = plt.figure(figsize=(8, 6.5))
        ax3d = fig3d.add_subplot(projection="3d")
        ax3d.plot_surface(E_mesh, N_mesh, U_grid_2d, cmap="viridis", edgecolor="none",
                           antialiased=True, alpha=0.95, vmin=u_min, vmax=u_max)
        ax3d.set_zlim(u_min, u_max)
        ax3d.scatter([r["orig_E"]], [r["orig_N"]], [r["base_U"]], marker="*", s=200,
                     c="red", depthshade=False, label="optimized position")
        ax3d.scatter([r["argmax_E"]], [r["argmax_N"]], [r["argmax_U"]], marker="X", s=100,
                     c="cyan", depthshade=False, label="grid argmax")
        ax3d.set_xlabel("East (m)")
        ax3d.set_ylabel("North (m)")
        ax3d.set_zlabel("U")
        ax3d.set_title(f"U vs. position of detector {r['idx']} ({tag}), 3D")
        ax3d.legend(loc="upper left", fontsize=8)
        ax3d.view_init(elev=25, azim=-60)
        fig3d.tight_layout()
        out_png = os.path.join(HERE, f"detector_grid_{tag}_3d.png")
        fig3d.savefig(out_png, dpi=150)
        plt.close(fig3d)
        print(f"[plot] wrote {out_png}")


def _slice_panel_plot(tag, r, out_prefix, title_suffix, u_min, u_max):
    alphas = np.array(r["alphas"])
    betas = np.array(r["betas"])
    U_grid = np.array(r["U_grid"])
    base_U = r["base_U"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(betas, alphas, U_grid, shading="auto", cmap="viridis",
                        vmin=u_min, vmax=u_max)
    plt.colorbar(im, ax=ax, label="U")
    ax.scatter([0], [0], marker="*", s=250, c="red", edgecolor="black", label="base layout")
    ax.set_xlabel("beta (m, direction 2)")
    ax.set_ylabel("alpha (m, direction 1)")
    ax.set_title(f"Full-space random 2D slice{title_suffix}: {tag}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(HERE, f"{out_prefix}_{tag}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_png}")

    B_mesh, A_mesh = np.meshgrid(betas, alphas)
    fig3d = plt.figure(figsize=(7.5, 6.5))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.plot_surface(B_mesh, A_mesh, U_grid, cmap="viridis", edgecolor="none",
                       antialiased=True, alpha=0.95, vmin=u_min, vmax=u_max)
    ax3d.set_zlim(u_min, u_max)
    ax3d.scatter([0], [0], [base_U], marker="*", s=200, c="red", depthshade=False,
                 label="base layout")
    ax3d.set_xlabel("beta (m, direction 2)")
    ax3d.set_ylabel("alpha (m, direction 1)")
    ax3d.set_zlabel("U")
    ax3d.set_title(f"Full-space random 2D slice{title_suffix} (3D): {tag}")
    ax3d.view_init(elev=25, azim=-60)
    fig3d.tight_layout()
    out_png = os.path.join(HERE, f"{out_prefix}_{tag}_3d.png")
    fig3d.savefig(out_png, dpi=150)
    plt.close(fig3d)
    print(f"[plot] wrote {out_png}")


def make_full_space_slice_plots(json_path, out_prefix, title_suffix):
    with open(json_path) as f:
        data = json.load(f)
    u_min = min(r["U_min"] for r in data.values())
    u_max = max(r["U_max"] for r in data.values())
    print(f"[{out_prefix}] own-run shared scale: [{u_min:.3f}, {u_max:.3f}]")
    for tag, r in data.items():
        _slice_panel_plot(tag, r, out_prefix, title_suffix, u_min, u_max)
    return data


def make_coarse_fine_combined_plots(coarse_data, fine_data):
    """Coarse (400m) + fine (100m) full-space slice results share ONE scale,
    so the roughness-vs-step-size comparison is visually honest."""
    all_mins = [r["U_min"] for r in coarse_data.values()] + [r["U_min"] for r in fine_data.values()]
    all_maxs = [r["U_max"] for r in coarse_data.values()] + [r["U_max"] for r in fine_data.values()]
    u_min, u_max = min(all_mins), max(all_maxs)
    print(f"[coarse+fine combined] shared scale: [{u_min:.3f}, {u_max:.3f}]")
    for tag, r in coarse_data.items():
        _slice_panel_plot(tag, r, "full_space_2d_slice_cfcombined_coarse", " (400m step)", u_min, u_max)
    for tag, r in fine_data.items():
        _slice_panel_plot(tag, r, "full_space_2d_slice_cfcombined_fine", " (100m step)", u_min, u_max)


print("Regenerating plots with shared, comparable color/z scales ...")
make_grid_scan_plots()
coarse_data = make_full_space_slice_plots(
    os.path.join(HERE, "full_space_2d_slice_results.json"), "full_space_2d_slice", "")
fine_data = make_full_space_slice_plots(
    os.path.join(HERE, "full_space_2d_slice_fine_results.json"), "full_space_2d_slice_fine", ", fine step")
make_coarse_fine_combined_plots(coarse_data, fine_data)
print("Done.")
