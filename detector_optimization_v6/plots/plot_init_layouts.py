"""Render the initial detector distributions the 04 optimizers start from.

The 04 optimizers (lbfgs / DE / DE-pop) seed from `sample_initial_layout_ne` init
schemes (grid, center) in the (North, East) plane. This draws those starting
layouts on the mountain in the (North, Up) cross section — East projected through
SurfaceUpMap, the same view as the fixed ensemble plots — so the optimization
starting points are visible. PNGs land in <OPT_FOLDER>_init/.

    cd TambOpt/detector_optimization_v6
    python plots/plot_init_layouts.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = os.path.dirname(_HERE)
if _V6 not in sys.path:
    sys.path.insert(0, _V6)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v6.tr_geometry_ne import sample_initial_layout_ne, project_to_mountain_ne
from modules_v6.constants import (
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    N_DETECTORS, OPT_FOLDER,
)

SCHEMES = ("grid", "center")          # the 04 optimizers' INIT_SCHEMES
OUT_DIR = OPT_FOLDER + "_init"

# constants.GEOMETRY_PATH may be stale; prefer a local copy, then the TAMBOSim path.
GEOMETRY_PATH_RESOLVED = next(
    (p for p in (os.path.join(_V6, "colca_valley.h5"),
                 "/n/home05/zdimitrov/tambo/TAMBOSim/resources/geometry/colca_valley.h5",
                 GEOMETRY_PATH) if os.path.exists(p)),
    GEOMETRY_PATH)


@torch.no_grad()
def _to_up(surface, north, east):
    """East → Up via SurfaceUpMap so NE init layouts draw in the (North, Up) plane."""
    dev = surface.grid_up.device
    n = torch.as_tensor(np.asarray(north).reshape(-1), dtype=torch.float32, device=dev)
    e = torch.as_tensor(np.asarray(east ).reshape(-1), dtype=torch.float32, device=dev)
    return surface(n, e).detach().cpu().numpy()


def _init_up(mtn, surface, scheme):
    """Sample one init scheme, project to the mountain (NE), return (North, Up)."""
    N_np, E_np = sample_initial_layout_ne(mtn, n_units=N_DETECTORS, scheme=scheme)
    N, E = project_to_mountain_ne(mtn, torch.as_tensor(N_np).float(),
                                  torch.as_tensor(E_np).float())
    return N.numpy(), _to_up(surface, N.numpy(), E.numpy())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                           n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mtn).to("cpu")          # CPU: no GPU needed
    mtn_N, mtn_U = mtn.centroids_NUE[:, 0], mtn.centroids_NUE[:, 1]

    inits = {s: _init_up(mtn, surface, s) for s in SCHEMES}

    # Combined panel (one subplot per init scheme).
    fig, axes = plt.subplots(1, len(SCHEMES), figsize=(7 * len(SCHEMES), 6),
                             squeeze=False)
    for ax, scheme in zip(axes[0], SCHEMES):
        Nn, Uu = inits[scheme]
        ax.scatter(mtn_N, mtn_U, s=2, c="lightgray", alpha=0.6, label="mountain")
        ax.scatter(Nn, Uu, s=30, c="C3", edgecolors="black", linewidths=0.4,
                   label=f"{scheme} init (n={N_DETECTORS})")
        ax.set_xlabel("North [m]"); ax.set_ylabel("Up [m]"); ax.set_aspect("equal")
        ax.set_title(f"init scheme: {scheme}")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("04 optimization — initial detector distributions (North, Up)",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "init_layouts.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[save] {out}")

    # Individual PNGs per scheme.
    for scheme in SCHEMES:
        Nn, Uu = inits[scheme]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(mtn_N, mtn_U, s=2, c="lightgray", alpha=0.6, label="mountain")
        ax.scatter(Nn, Uu, s=34, c="C3", edgecolors="black", linewidths=0.4,
                   label=f"{scheme} init")
        ax.set_xlabel("North [m]"); ax.set_ylabel("Up [m]"); ax.set_aspect("equal")
        ax.set_title(f"04 init layout: {scheme} (North, Up)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        p = os.path.join(OUT_DIR, f"init_{scheme}.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[save] {p}")
    print(f"[done] init layout plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
