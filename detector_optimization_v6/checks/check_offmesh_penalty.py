"""Does the off-mesh penalty actually stop L-BFGS leaving the mountain?

Runs ONE lbfgs_refine chunk from the same grid init, with the penalty off and
on, and reports how much of the resulting closure path sits outside the 91.3 m
snap radius. Off should reproduce the failure (two thirds of closures outside,
excursions of hundreds of metres); on should keep the path inside, at no cost
to the projected U — the penalty is 0 in-band by construction.

    sbatch check_offmesh_penalty.sh        # needs a GPU; too heavy for a login node
"""
import importlib.util
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_spec = importlib.util.spec_from_file_location(
    "opt4", os.path.join(_HERE, "04_optimize_lbfgs_ensemble.py"))
o = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(o)

# 04 is imported as a library here, so its module-level folder globals are still
# None (they are bound inside its main()). This check resolves its own world.
sys.path.insert(0, os.path.dirname(_HERE))
from modules_v6 import run_world  # noqa: E402

_W = run_world.resolve()


def main():
    mt = o.load_tr_mountain(o.GEOMETRY_PATH_RESOLVED, o.GEOMETRY_GROUP, o.DET_KEY,
                            east_entry=o.EAST_ENTRY, layer_east_dx=o.LAYER_EAST_DX,
                            n_planes=o.N_PLANES)
    cen = np.asarray(mt.centroids_ENU)
    tree = cKDTree(cen[:, :2])
    # Two different radii, and conflating them made the first run of this check
    # unreadable: `snap` is where project_to_mountain_ne fires and therefore
    # where a layout gets destroyed, `onset` is where the penalty wall starts
    # (deliberately inside it). Report against `snap` — that is the failure.
    snap = float(o._ne_max_gap(mt))
    onset = o._penalty_args(mt)[1]
    fnn, recon = o.load_models(o.DEVICE, fnn_folder=_W.fnn_folder,
                               recon_dir=_W.recon_dir)
    prim = torch.load(os.path.join(_W.dataset_folder, "primary.pt"),
                      map_location="cpu", weights_only=False)
    prim = prim[:o.LBFGS_BATCH_PRIMARIES].to(o.DEVICE)
    E, N = o.sample_initial_layout_ne(mt, n_units=o.N_DETECTORS, scheme="grid")
    x0, y0 = torch.tensor(E), torch.tensor(N)

    print(f"snap radius {snap:.1f} m, penalty onset {onset:.1f} m, "
          f"batch = {prim.shape[0]} primaries", flush=True)
    for w in (0.0, o.OFFMESH_PENALTY_W):
        o.OFFMESH_PENALTY = w
        xp, yp, Up, log, _g, ph = o.lbfgs_refine(x0, y0, fnn, recon, prim, mt)
        P = ph.numpy()
        P = np.stack([P[:, :o.N_DETECTORS], P[:, o.N_DETECTORS:]], axis=-1)
        D = tree.query(P.reshape(-1, 2))[0].reshape(P.shape[:2])
        out = D > snap
        # D[-1] is the PROJECTED optimum (always in-band by construction);
        # D[-2] is the last unprojected iterate, i.e. what the snap acts on and
        # what actually decides whether the layout survives.
        pre = D[-2]
        print(f"penalty w={w:<6.0f} closures={len(log):5d}  U_proj={Up:+8.3f}\n"
              f"    path   : {out.any(1).sum():5d}/{len(D)} frames "
              f"({100 * out.any(1).mean():5.1f}%) have a detector past the snap "
              f"radius;  worst {D.max():12.1f} m\n"
              f"    pre-snap layout: {int((pre > snap).sum()):3d}/{o.N_DETECTORS} "
              f"detectors past snap radius, {int((pre > onset).sum()):3d} past onset, "
              f"max {pre.max():8.1f} m, median {np.median(pre):6.1f} m\n"
              f"    snap moved      : {float(np.abs(D[-1] - pre).max()):8.1f} m worst, "
              f"{int((np.abs(D[-1] - pre) > 1.0).sum()):3d} detectors displaced",
              flush=True)


if __name__ == "__main__":
    main()
