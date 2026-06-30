"""Basin-hopping optimizer for detector layout.

Algorithm: random large perturbation → mountain projection → L-BFGS local
polish → Metropolis accept/reject, repeated N_HOPS times. Starting from a
known good layout (--init_from), this explores distinct basins of the
surrogate landscape rather than converging to the same local optimum from many
nearby starts.

Rationale: all first-order optimizers (Adam, L-BFGS, GES) converge to
U~208-210 from diverse starts, confirming the landscape is a broad, flat
plateau. Basin hopping with SIGMA_HOP >> plateau_width (~200m) jumps to
genuinely different regions of layout space, each refined by L-BFGS. If any
basin is substantially better than the current best, it will be found.

Output artifacts in <OPT_FOLDER>_basinhopping{opt_suffix}/:
    layout_best.pt      highest-U accepted layout (mountain-projected)
    hop_log.json        per-hop record: U, delta_U, accepted, best_U
    layout_basinhopping.png   mountain top-down: all accepted layouts + best

Run:
    cd TambOpt/detector_optimization_v6
    python 04_optimize_basinhopping.py --init_from <path/to/layout_best.pt>
    python 04_optimize_basinhopping.py --init_from <path> --n_hops 100 --sigma_hop 1000
"""
import importlib.util
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6
from modules_v6.constants import (
    N_DETECTORS,
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER, OPT_FOLDER,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.tr_geometry_ne import project_to_mountain_ne, sample_initial_layout_ne
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v6.opt_core import utility_of_xy, load_models

_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(_HERE, "plots", "opt_plotting.py"))
_plt = importlib.util.module_from_spec(_plt_spec); _plt_spec.loader.exec_module(_plt)


# ── Config ────────────────────────────────────────────────────────────────────
OPT_DIR_TEMPLATE = OPT_FOLDER + "_basinhopping{suffix}"
RECON_DIR        = RECON_FOLDER + "_deepsets"

N_HOPS      = 50       # number of basin-hop attempts
SIGMA_HOP   = 750.0    # m — perturbation sigma; should be >> plateau breadth (~200m)
TEMPERATURE = 2.0      # Metropolis T in utility units; 0 = greedy (accept only improvements)

# Fixed primary batch for all L-BFGS polishes (reproducible comparisons).
LBFGS_BATCH     = 2048
LBFGS_MAX_ITER  = 1_500
LBFGS_LR        = 1.0
LBFGS_HISTORY   = 20

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── L-BFGS local polish ───────────────────────────────────────────────────────
class _NonFinite(Exception):
    pass


def _lbfgs_refine(init_x: torch.Tensor,
                  init_y: torch.Tensor,
                  fnn, recon,
                  primary_fixed: torch.Tensor,
                  mountain) -> tuple:
    """L-BFGS from (init_x, init_y) on a fixed primary batch.

    Returns (x_proj, y_proj, U_float) after projecting the optimum back to
    the mountain. Falls back to init if the line search diverges.
    """
    xy = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach().clone()
    xy.requires_grad_(True)

    opt = torch.optim.LBFGS(
        [xy], lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY, line_search_fn="strong_wolfe",
        tolerance_grad=1e-11, tolerance_change=1e-13,
    )

    def closure():
        opt.zero_grad()
        U, _, _ = utility_of_xy(xy[:N_DETECTORS], xy[N_DETECTORS:],
                                 primary_fixed, fnn, recon)
        loss = -U
        if not torch.isfinite(loss):
            raise _NonFinite
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except _NonFinite:
        print("  [lbfgs] non-finite loss — aborting")

    with torch.no_grad():
        if not torch.isfinite(xy).all():
            print("  [lbfgs] non-finite iterate — falling back to init")
            xy.data = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0)
        x_proj, y_proj = project_to_mountain_ne(
            mountain, xy[:N_DETECTORS].cpu(), xy[N_DETECTORS:].cpu())
        U_proj, _, _ = utility_of_xy(
            x_proj.to(DEVICE), y_proj.to(DEVICE), primary_fixed, fnn, recon)

    return x_proj.float(), y_proj.float(), float(U_proj.item())


# ── Basin hopping ─────────────────────────────────────────────────────────────
def run_basin_hopping(start_x: torch.Tensor,
                      start_y: torch.Tensor,
                      fnn, recon,
                      primary_fixed: torch.Tensor,
                      mountain,
                      rng: torch.Generator) -> tuple:
    """N_HOPS of basin hopping from (start_x, start_y).

    Returns (best_x, best_y, best_U, hop_log, accepted_layouts).
    accepted_layouts is a list of (x, y) tensors for all accepted hops.
    """
    # L-BFGS-polish the start so we begin at a true local minimum.
    print("[bh] polishing start layout …")
    best_x, best_y, best_U = _lbfgs_refine(
        start_x, start_y, fnn, recon, primary_fixed, mountain)
    print(f"[bh] start U = {best_U:.4f}")

    hop_log = []
    accepted_layouts = [(best_x.clone(), best_y.clone())]

    for hop in range(N_HOPS):
        t0 = time.time()

        # Large random perturbation.
        dx = torch.randn(N_DETECTORS, generator=rng) * SIGMA_HOP
        dy = torch.randn(N_DETECTORS, generator=rng) * SIGMA_HOP
        x_hop, y_hop = project_to_mountain_ne(
            mountain, best_x + dx, best_y + dy)

        # L-BFGS local polish.
        x_opt, y_opt, U_opt = _lbfgs_refine(
            x_hop, y_hop, fnn, recon, primary_fixed, mountain)

        # Metropolis accept/reject.
        delta_U = U_opt - best_U
        if TEMPERATURE > 0:
            accept = (delta_U > 0) or (
                torch.rand(1, generator=rng).item() < math.exp(delta_U / TEMPERATURE))
        else:
            accept = delta_U > 0

        if accept:
            best_x, best_y, best_U = x_opt, y_opt, U_opt
            accepted_layouts.append((best_x.clone(), best_y.clone()))

        elapsed = time.time() - t0
        hop_log.append(dict(
            hop=hop, U_opt=U_opt, delta_U=delta_U,
            accepted=accept, best_U=best_U, elapsed_s=elapsed,
        ))
        print(f"[hop {hop+1:3d}/{N_HOPS}]  U={U_opt:+.4f}  "
              f"delta={delta_U:+.4f}  {'ACCEPT' if accept else 'reject'}  "
              f"best={best_U:+.4f}  ({elapsed:.1f}s)")

    return best_x, best_y, best_U, hop_log, accepted_layouts


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global N_HOPS, SIGMA_HOP, TEMPERATURE, LBFGS_BATCH, LBFGS_MAX_ITER, RECON_DIR

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", type=str, default=None,
                    help="Path to a layout_best.pt (dict with 'x'/'y' keys) or a "
                         "(N_DETECTORS, 2) tensor to warm-start from. If omitted, "
                         "samples a random grid layout.")
    ap.add_argument("--n_hops",      type=int,   default=N_HOPS)
    ap.add_argument("--sigma_hop",   type=float, default=SIGMA_HOP,
                    help="Perturbation sigma in metres (default 750m).")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE,
                    help="Metropolis temperature in utility units. 0 = greedy.")
    ap.add_argument("--lbfgs_batch", type=int,   default=LBFGS_BATCH,
                    help="Fixed primary batch for every L-BFGS polish.")
    ap.add_argument("--lbfgs_iter",  type=int,   default=LBFGS_MAX_ITER)
    ap.add_argument("--fnn_folder",  type=str,   default=None)
    ap.add_argument("--recon_folder",type=str,   default=None)
    ap.add_argument("--opt_suffix",  type=str,   default="",
                    help="Suffix for the output directory name.")
    ap.add_argument("--seed",        type=int,   default=SEED)
    args = ap.parse_args()

    N_HOPS         = args.n_hops
    SIGMA_HOP      = args.sigma_hop
    TEMPERATURE    = args.temperature
    LBFGS_BATCH    = args.lbfgs_batch
    LBFGS_MAX_ITER = args.lbfgs_iter
    if args.recon_folder:
        RECON_DIR = args.recon_folder

    opt_dir = OPT_DIR_TEMPLATE.format(suffix=args.opt_suffix)
    os.makedirs(opt_dir, exist_ok=True)

    print("=" * 72)
    print("v6/04_optimize_basinhopping.py — basin hopping")
    print("=" * 72)
    print(f"device       : {DEVICE}")
    print(f"n_hops       : {N_HOPS}")
    print(f"sigma_hop    : {SIGMA_HOP:.0f} m")
    print(f"temperature  : {TEMPERATURE} utility units")
    print(f"lbfgs_batch  : {LBFGS_BATCH}  lbfgs_iter: {LBFGS_MAX_ITER}")
    print(f"output dir   : {opt_dir}")

    fnn, recon = load_models(DEVICE,
                             fnn_folder=args.fnn_folder or FNN_FOLDER,
                             recon_dir=RECON_DIR)
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    # Fixed primary batch — same for every hop so U values are comparable.
    primary_all = torch.load(
        os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    g_batch = torch.Generator().manual_seed(args.seed)
    idx = torch.randint(0, len(primary_all), (LBFGS_BATCH,), generator=g_batch)
    primary_fixed = primary_all[idx].to(DEVICE)
    print(f"[load] {len(primary_all)} primaries; fixed batch size={LBFGS_BATCH}")

    # Start layout.
    if args.init_from:
        raw = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            start_x = raw["x"].float().reshape(-1)
            start_y = raw["y"].float().reshape(-1)
        else:
            start_x = raw[:, 0].float()
            start_y = raw[:, 1].float()
        print(f"[init] loaded from {args.init_from}")
    else:
        N_np, E_np = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme="grid")
        start_x = torch.as_tensor(N_np, dtype=torch.float32)
        start_y = torch.as_tensor(E_np, dtype=torch.float32)
        print("[init] using grid layout (no --init_from provided)")

    rng = torch.Generator().manual_seed(args.seed + 1)
    best_x, best_y, best_U, hop_log, accepted_layouts = run_basin_hopping(
        start_x, start_y, fnn, recon, primary_fixed, mountain, rng)

    # ── Persist ───────────────────────────────────────────────────────────────
    torch.save({"x": best_x, "y": best_y, "U": best_U},
               os.path.join(opt_dir, "layout_best.pt"))

    n_accepted = sum(1 for h in hop_log if h["accepted"])
    with open(os.path.join(opt_dir, "hop_log.json"), "w") as f:
        json.dump({
            "best_U": best_U,
            "n_accepted": n_accepted,
            "accept_rate": n_accepted / max(N_HOPS, 1),
            "hop_log": hop_log,
            "config": dict(
                n_hops=N_HOPS, sigma_hop=SIGMA_HOP, temperature=TEMPERATURE,
                lbfgs_batch=LBFGS_BATCH, lbfgs_max_iter=LBFGS_MAX_ITER,
                seed=args.seed, init_from=args.init_from,
            ),
        }, f, indent=2)

    # Plot all accepted layouts overlaid on the mountain.
    try:
        surface = SurfaceUpMap.from_mountain(mountain).to("cpu")
        accepted_xy = np.stack(
            [np.stack([x.numpy(), y.numpy()], axis=-1)
             for x, y in accepted_layouts], axis=0)  # (n_accepted+1, n_det, 2)
        mean_xy = accepted_xy.mean(axis=0)
        std_xy  = accepted_xy.std(axis=0)
        _plt.plot_ensemble(
            accepted_xy, mean_xy, std_xy, best_x, best_y,
            mountain, os.path.join(opt_dir, "layout_basinhopping.png"),
            surface=surface, title_kind="basin hopping accepted")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")

    print()
    print("=" * 72)
    print(f"best U = {best_U:.4f}   accepted {n_accepted}/{N_HOPS} hops"
          f"  ({100*n_accepted/max(N_HOPS,1):.0f}%)")
    print(f"output: {opt_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
