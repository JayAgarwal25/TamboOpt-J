#!/usr/bin/env python3
"""
One-time regeneration of 3D surface-plot companions for heatmaps that were
already computed and saved (with their raw U_grid) before 3D plotting was
added to detector_grid_scan.py, full_space_2d_slice.py, and
full_space_2d_slice_fine.py. Pure post-processing: reads the already-saved
*_results.json files and re-renders, no model/GPU computation, no rerun of
the underlying experiments.

Future runs of those three scripts produce their 3D companions directly; this
script only exists to backfill the ones already on disk.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def make_grid_scan_3d(json_path, tag_key):
    with open(json_path) as f:
        data = json.load(f)
    r = data[tag_key]
    n_grid = np.array(r["n_grid"])
    e_grid = np.array(r["e_grid"])
    U_grid_2d = np.array(r["U_grid"])
    E_mesh, N_mesh = np.meshgrid(e_grid, n_grid)
    fig3d = plt.figure(figsize=(8, 6.5))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.plot_surface(E_mesh, N_mesh, U_grid_2d, cmap="viridis", edgecolor="none",
                       antialiased=True, alpha=0.95)
    ax3d.scatter([r["orig_E"]], [r["orig_N"]], [r["base_U"]], marker="*", s=200,
                 c="red", depthshade=False, label="optimized position")
    ax3d.scatter([r["argmax_E"]], [r["argmax_N"]], [r["argmax_U"]], marker="X", s=100,
                 c="cyan", depthshade=False, label="grid argmax")
    ax3d.set_xlabel("East (m)")
    ax3d.set_ylabel("North (m)")
    ax3d.set_zlabel("U")
    ax3d.set_title(f"U vs. position of detector {r['idx']} ({tag_key}), 3D")
    ax3d.view_init(elev=25, azim=-60)
    fig3d.tight_layout()
    out_png = os.path.join(HERE, f"detector_grid_{tag_key}_3d.png")
    fig3d.savefig(out_png, dpi=150)
    plt.close(fig3d)
    print(f"[plot] wrote {out_png}")


def make_full_space_slice_3d(json_path, out_prefix, title_suffix):
    with open(json_path) as f:
        data = json.load(f)
    for tag, r in data.items():
        alphas = np.array(r["alphas"])
        betas = np.array(r["betas"])
        U_grid = np.array(r["U_grid"])
        base_U = r["base_U"]
        B_mesh, A_mesh = np.meshgrid(betas, alphas)
        fig3d = plt.figure(figsize=(7.5, 6.5))
        ax3d = fig3d.add_subplot(projection="3d")
        ax3d.plot_surface(B_mesh, A_mesh, U_grid, cmap="viridis", edgecolor="none",
                           antialiased=True, alpha=0.95)
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


print("Regenerating 3D surface plots from already-saved results ...")

make_grid_scan_3d(os.path.join(HERE, "detector_grid_results.json"), "center")
make_grid_scan_3d(os.path.join(HERE, "detector_grid_results.json"), "edge")

make_full_space_slice_3d(os.path.join(HERE, "full_space_2d_slice_results.json"),
                          "full_space_2d_slice", "")
make_full_space_slice_3d(os.path.join(HERE, "full_space_2d_slice_fine_results.json"),
                          "full_space_2d_slice_fine", ", fine step")

print("Done.")
