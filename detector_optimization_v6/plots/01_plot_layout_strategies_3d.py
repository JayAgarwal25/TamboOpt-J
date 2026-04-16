"""Plot one example detector layout per strategy over the mountain in 3D.

Loads the tensors produced by `01_build_dataset.py` from
`outputs/v6_run_01/` (xy.pt + strategy_ids.pt), picks the first sample
belonging to each strategy id, evaluates East = f(N, Up) via the
SurfaceEastMap, and draws the layout on top of the mountain centroid cloud.
Axes are (North, East, Up) so the mountain's elevation is vertical.

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python plot_layout_strategies_3d.py

Output: `outputs/v6_run_01/layout_strategies_3d.png`.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D proj)

import modules_v6  # noqa: F401 — triggers sys.path injection for v3 + v4
from modules_v6.fnn_surrogate import (
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    _STRATEGIES,
)
from modules_v4.tr_geometry    import load_tr_mountain
from modules_v4.tr_surface_map import SurfaceEastMap


RUN_DIR    = os.path.join(_HERE, "outputs", "v6_run_01")
XY_PATH    = os.path.join(RUN_DIR, "xy.pt")
STRAT_PATH = os.path.join(RUN_DIR, "strategy_ids.pt")
OUTPUT_PNG = os.path.join(RUN_DIR, "layout_strategies_3d.png")
DEVICE     = torch.device("cpu")


def main():
    xy    = torch.load(XY_PATH,    map_location=DEVICE)  # (N_pairs, 100, 2)
    strat = torch.load(STRAT_PATH, map_location=DEVICE)  # (N_pairs,)
    print(f"[load] xy {tuple(xy.shape)}  strat {tuple(strat.shape)}  "
          f"unique={sorted(strat.unique().tolist())}")

    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    surface = SurfaceEastMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)

    cen = mountain.centroids_NUE  # (n_tri, 3) -> [N, Up, East]
    mN, mU, mE = cen[:, 0], cen[:, 1], cen[:, 2]

    n_strat = len(_STRATEGIES)
    fig = plt.figure(figsize=(6.0 * n_strat, 6.0))
    for s_idx, (s_name, fn_name, kwargs) in enumerate(_STRATEGIES):
        mask = (strat == s_idx).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            print(f"[warn] no samples for strategy {s_idx} ({s_name})")
            continue
        sample_idx = int(mask[0])
        layout = xy[sample_idx].float()              # (100, 2) — columns [N, Up]
        x_det  = layout[:, 0]
        y_det  = layout[:, 1]
        east   = surface(x_det, y_det).detach()

        ax = fig.add_subplot(1, n_strat, s_idx + 1, projection="3d")
        ax.scatter(mN, mE, mU, s=1.0, c="lightgray", alpha=0.35,
                   linewidths=0, label="mountain")
        ax.scatter(x_det.numpy(), east.numpy(), y_det.numpy(),
                   s=22, c="crimson", depthshade=True, label="detectors")

        ax.set_title(f"{s_idx}: {s_name}\nsample #{sample_idx}  ({fn_name})",
                     fontsize=10)
        ax.set_xlabel("North [m]")
        ax.set_ylabel("East [m]")
        ax.set_zlabel("Up [m]")
        ax.view_init(elev=22, azim=-60)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("v6 layout strategies — one example per strategy (from 01 outputs)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=140)
    plt.close(fig)
    print(f"[save] {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
