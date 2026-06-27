"""Optimize detector positions with a SINGLE differential-evolution run.

Population variant of ``04_optimize_differential_evolution.py``. Instead of
running DE K separate times (once per perturbed start, per scheme) and stacking
the K optima into an ensemble, this seeds ONE DE run with a hand-built initial
**population** and reads the ensemble straight off DE's final population.

The ``init`` population has ``POP_SIZE`` members — ``N_PER_SCHEME`` from each
scheme in ``INIT_SCHEMES`` (15 grid + 15 center = 30). For each scheme member 0
is the deterministic base layout (``sample_initial_layout_ne``); the rest are
Gaussian perturbations of it (``INIT_PERTURB_SIGMA``), all projected to the
mountain. Because ``init`` is an array, scipy overrides ``popsize`` and the
member count IS the population size, so ``popsize`` and ``x0`` are dropped from
the DE call.

Detectors use the **(North, East)** convention: 100 North + 100 East, each
candidate projected to the mountain (``project_to_mountain_ne``) before scoring.
The ensemble = DE's final population (``result.population``, projected): per
detector group the mean/std across members, after Hungarian alignment to the
best member. Artifacts/plots match the L-BFGS ensemble (the East→Up surface
projection draws them in the (North, Up) cross section).

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 04_optimize_differential_evolution_pop.py
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
from scipy.optimize import differential_evolution

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.tr_geometry_ne import (
    _ne_max_gap, project_to_mountain_ne, sample_initial_layout_ne,
)
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER, OPT_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules_v4.tr_geometry      import load_tr_mountain

# Shared optimizer core (objective, alignment, model loading, constants) lives in
# modules_v6/opt_core.py; the figures live in plots/opt_plotting.py (loaded by path).
from modules_v6.opt_core import (
    primary_to_physical_labels, utility_of_xy, align_to_reference, load_models,
    W_THETA, W_PHI, W_E, W_PR, W_DIV,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD, GEOMETRY_PATH_RESOLVED,
)
_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(_HERE, "plots", "opt_plotting.py"))
_plt = importlib.util.module_from_spec(_plt_spec); _plt_spec.loader.exec_module(_plt)


# ── Config ───────────────────────────────────────────────────────────────────
INIT_SCHEMES        = ("grid", "center")   # init population = N_PER_SCHEME from each
N_PER_SCHEME        = 15                    # 15 grid + 15 center → 30-member population
POP_SIZE            = N_PER_SCHEME * len(INIT_SCHEMES)
INIT_PERTURB_SIGMA  = 1000.0   # metres — Gaussian spread of the perturbed members

OPT_DIR             = OPT_FOLDER + "_de_population"

# Differential evolution — one run over the whole population.
# (No popsize: the init array sets the population size. No x0: it would replace
#  one of the chosen members.)
DE_MAXITER          = 1000
DE_TOL              = 1e-4
DE_MUTATION         = (0.5, 1.0)
DE_RECOMBINATION    = 0.7
# DE_BATCH_PRIMARIES: the FIXED batch that makes the objective deterministic, and
# the knob trading objective fidelity vs cost. Peak GPU memory (~0.44 GB / 1000
# showers) AND per-eval time both scale linearly in it. 50k (~22 GB) is a far less
# noisy estimate than 512 and fits a 40 GB A100 with headroom (raise toward ~150k
# on an 80 GB card). Per-eval cost is ~batch/512, so cut DE_MAXITER to keep the
# wall-clock bounded when you grow this.
DE_BATCH_PRIMARIES  = 50_000

# Composite weights (W_*) + reconstructability thresholds + GEOMETRY_PATH_RESOLVED
# are imported from modules_v6/opt_core.py (shared across the 04 optimizers).
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_init_population(mountain, generator: torch.Generator):
    """The DE initial population: N_PER_SCHEME members per init scheme.

    Per scheme, member 0 is the deterministic base layout
    (`sample_initial_layout_ne`); the rest are Gaussian perturbations of it
    (std `INIT_PERTURB_SIGMA`), each projected back to the mountain. Schemes are
    concatenated into one (POP_SIZE, 2*N_DETECTORS) float64 array — scipy's
    `init`. Also returns the per-member scheme label; note it is only nominal,
    since DE mixes the population during the run."""
    members, sources = [], []
    for scheme in INIT_SCHEMES:
        N_np, E_np = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
        bN, bE = project_to_mountain_ne(
            mountain,
            torch.as_tensor(N_np, dtype=torch.float32),
            torch.as_tensor(E_np, dtype=torch.float32),
        )
        base = torch.cat([bN, bE], dim=0)                          # (2*n_det,)
        for j in range(N_PER_SCHEME):
            if j == 0:
                flat = base
            else:
                noise = torch.randn(base.numel(), generator=generator) * INIT_PERTURB_SIGMA
                pN, pE = project_to_mountain_ne(
                    mountain,
                    base[:N_DETECTORS] + noise[:N_DETECTORS],
                    base[N_DETECTORS:] + noise[N_DETECTORS:],
                )
                flat = torch.cat([pN, pE], dim=0)
            members.append(flat.detach().cpu().double().numpy())
            sources.append(scheme)
    pop0 = np.stack(members, axis=0)                               # (POP_SIZE, 2*n_det)
    return pop0, sources


def _run_de(pop0: np.ndarray, bounds, fnn, recon, primary_fixed, mountain):
    """One differential-evolution run over the whole init population.

    `pop0` is scipy's `init` array (so `popsize` is overridden and no `x0` is
    passed — it would displace a chosen member). The objective projects each
    candidate to the mountain, then maximises composite U. Returns the
    OptimizeResult plus a per-generation best-so-far log for the diagnostic
    curves."""
    @torch.no_grad()   # gradient-free DE; opt_core.utility_of_xy is not no_grad-wrapped
    def _score(flat):
        x_det = torch.as_tensor(flat[:N_DETECTORS], dtype=torch.float32, device=DEVICE)
        y_det = torch.as_tensor(flat[N_DETECTORS:], dtype=torch.float32, device=DEVICE)
        x_det, y_det = project_to_mountain_ne(mountain, x_det, y_det)
        U, r, parts = utility_of_xy(x_det, y_det, primary_fixed, fnn, recon)
        return float(U.item()), float(r.mean().item()), parts

    de_log = []
    best = {"U": -float("inf"), "x": pop0[0].copy()}

    def objective(flat):
        U, _, _ = _score(flat)
        if U > best["U"]:
            best["U"] = U
            best["x"] = np.asarray(flat, dtype=np.float64).copy()
        return -U

    def callback(xk, convergence=None):
        # One entry per generation, logged at the running best (monotonic U curve).
        U, r_mean, parts = _score(best["x"])
        de_log.append(dict(
            iter=len(de_log), U=U, r_mean=r_mean,
            u_theta=float(parts["u_theta"].item()),
            u_phi=float(parts["u_phi"].item()),
            u_e=float(parts["u_e"].item()),
            u_pr=float(parts["u_pr"].item()),
        ))

    result = differential_evolution(
        objective, bounds, init=pop0, maxiter=DE_MAXITER,
        tol=DE_TOL, mutation=DE_MUTATION, recombination=DE_RECOMBINATION,
        seed=SEED, polish=False, updating="immediate", workers=1,
        callback=callback,
    )
    return result, de_log


def main():
    print("=" * 72)
    print("v6/04_optimize_differential_evolution_pop.py — single DE run over a population")
    print("=" * 72)
    print(f"device       : {DEVICE}")
    print(f"init schemes : {INIT_SCHEMES}  ({N_PER_SCHEME} each → pop={POP_SIZE})")
    print(f"init σ       : {INIT_PERTURB_SIGMA} m")
    print(f"DE           : maxiter={DE_MAXITER}  batch={DE_BATCH_PRIMARIES}  (popsize set by init array)")

    opt_dir = OPT_DIR
    os.makedirs(opt_dir, exist_ok=True)

    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    n_total_primaries = int(primary_all.shape[0])
    print(f"[load] {n_total_primaries} primaries")

    fnn, recon = load_models(DEVICE)

    mountain = load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    # Differentiable Up = g(North, East): projects DE layouts (native to North–East)
    # into the (North, Up) cross section for the plots, matching the L-BFGS figures.
    surface = SurfaceUpMap.from_mountain(mountain).to(DEVICE)

    # One fixed primary batch → deterministic objective, directly comparable.
    # Sampled WITHOUT replacement (a true unique subsample — matters at large batch).
    g = torch.Generator().manual_seed(SEED)
    n_batch = min(DE_BATCH_PRIMARIES, n_total_primaries)
    idx_fixed = torch.randperm(n_total_primaries, generator=g)[:n_batch]
    primary_fixed = primary_all[idx_fixed].to(DEVICE)

    # DE bounds: 100 North in [n_min, n_max], 100 East in [east_lo, east_hi], each
    # widened by the NE projection tolerance — project_to_mountain_ne keeps points
    # within max_gap of a centroid, and scipy clips the init population / requires
    # candidates inside the bounds. Candidates are mountain-projected before
    # scoring, so the widened box never lets the optimum leave the mountain.
    margin = _ne_max_gap(mountain)
    print(f"[bounds] bbox widened by max_gap={margin:.1f} m")
    bounds = ([(mountain.n_min - margin, mountain.n_max + margin)] * N_DETECTORS +
              [(mountain.east_lo - margin, mountain.east_hi + margin)] * N_DETECTORS)

    # Build the POP_SIZE-member init population (deterministic).
    torch.manual_seed(SEED); np.random.seed(SEED)
    gp = torch.Generator().manual_seed(SEED)
    pop0, sources = _build_init_population(mountain, gp)
    counts = ", ".join(f"{s}={sources.count(s)}" for s in INIT_SCHEMES)
    print(f"[init] population {pop0.shape}  ({counts})")

    # Single differential-evolution run over the whole population.
    print("-" * 72)
    print(f"[de] one run, pop={POP_SIZE}, maxiter={DE_MAXITER}  ->  {opt_dir}")
    t0 = time.time()
    result, de_log = _run_de(pop0, bounds, fnn, recon, primary_fixed, mountain)
    print(f"[de] done in {time.time() - t0:.1f}s  "
          f"nfev={result.nfev}  generations={len(de_log)}  success={result.success}")

    # Ensemble = DE's final population (projected to the mountain).
    final_pop = np.asarray(result.population)                       # (POP_SIZE, 2*n_det)
    energies  = np.asarray(result.population_energies)              # (POP_SIZE,)
    utilities = (-energies).astype(float)                          # U per member
    layouts = []
    for m in final_pop:
        xp, yp = project_to_mountain_ne(
            mountain,
            torch.as_tensor(m[:N_DETECTORS], dtype=torch.float32),
            torch.as_tensor(m[N_DETECTORS:], dtype=torch.float32),
        )
        layouts.append(np.stack([xp.numpy(), yp.numpy()], axis=-1))
    layouts_xy = np.stack(layouts, axis=0)                          # (POP_SIZE, n_det, 2)

    ref_idx = int(np.argmin(energies))                             # best-U member = reference
    aligned, perms = align_to_reference(layouts_xy, ref_idx)
    mean_xy = aligned.mean(axis=0)                                  # (n_det, 2)
    std_xy  = aligned.std(axis=0)                                   # (n_det, 2)

    best_x = torch.as_tensor(aligned[ref_idx, :, 0]).float()
    best_y = torch.as_tensor(aligned[ref_idx, :, 1]).float()
    best_U = float(utilities[ref_idx])
    print(f"[ensemble] pop={POP_SIZE}  best U={best_U:+.3f} (member {ref_idx}, "
          f"src={sources[ref_idx]})  mean σN={std_xy[:,0].mean():.1f}m σE={std_xy[:,1].mean():.1f}m")

    # ── Persist artifacts (same set/keys as the L-BFGS ensemble) ─────────────
    torch.save({"x": best_x, "y": best_y, "U": best_U,
                "run": ref_idx, "source": sources[ref_idx]},
               os.path.join(opt_dir, "layout_best.pt"))
    torch.save({"mean_x": torch.as_tensor(mean_xy[:, 0]),
                "mean_y": torch.as_tensor(mean_xy[:, 1]),
                "std_x":  torch.as_tensor(std_xy[:, 0]),
                "std_y":  torch.as_tensor(std_xy[:, 1])},
               os.path.join(opt_dir, "layout_mean.pt"))
    torch.save({"aligned": torch.as_tensor(aligned),          # (POP_SIZE, n_det, 2)
                "perms": torch.as_tensor(perms),
                "utilities": torch.as_tensor(utilities),
                "source_per_run": sources,
                "ref_idx": ref_idx},
               os.path.join(opt_dir, "layouts_all.pt"))

    with open(os.path.join(opt_dir, "optimize_log.json"), "w") as f:
        json.dump({
            "schemes": list(INIT_SCHEMES),
            "n_per_scheme": N_PER_SCHEME,
            "pop_size": POP_SIZE,
            "source_per_run": sources,
            "ref_idx": ref_idx,
            "ref_source": sources[ref_idx],
            "utilities": utilities.tolist(),
            "best_U": best_U,
            "ensemble_stats": dict(
                mean_std_x=float(std_xy[:, 0].mean()),
                mean_std_y=float(std_xy[:, 1].mean()),
                max_std_x=float(std_xy[:, 0].max()),
                max_std_y=float(std_xy[:, 1].max()),
            ),
            "de_best_U_history": [e["U"] for e in de_log],
            "de_log": de_log,
            "config": dict(
                n_per_scheme=N_PER_SCHEME, pop_size=POP_SIZE,
                init_perturb_sigma=INIT_PERTURB_SIGMA,
                de_maxiter=DE_MAXITER, de_tol=DE_TOL,
                de_mutation=list(DE_MUTATION), de_recombination=DE_RECOMBINATION,
                de_batch_primaries=DE_BATCH_PRIMARIES,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                seed=SEED,
            ),
        }, f, indent=2)

    _plt.plot_curves_de_pop(de_log, os.path.join(opt_dir, "optimize_curves.png"), POP_SIZE)
    _plt.plot_components_de_pop(de_log, os.path.join(opt_dir, "utility_components.png"))
    _plt.plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                       mountain, os.path.join(opt_dir, "layout_ensemble.png"), surface=surface,
                       member_word="member", title_kind="DE population ensemble", count_word="pop")
    _plt.plot_density_heatmap(aligned, best_x, best_y,
                       mountain, os.path.join(opt_dir, "layout_density.png"), surface=surface,
                       member_word="member", count_word="pop", count_suffix="")

    print()
    print("=" * 72)
    print(f"[done] best U={best_U:+.3f}  "
          f"σ̄=({std_xy[:,0].mean():.1f}, {std_xy[:,1].mean():.1f}) m  ({opt_dir})")
    print("=" * 72)


if __name__ == "__main__":
    main()
