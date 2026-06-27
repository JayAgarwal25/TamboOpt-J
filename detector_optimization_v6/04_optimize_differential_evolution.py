"""Optimize detector positions: pre-perturbation, then a differential-evolution ensemble.

Global, gradient-free sibling of ``04_optimize_lbfgs_ensemble.py``, kept
stage-for-stage identical so the two diff cleanly. Instead of refining each
warm-start with **L-BFGS**, stage 2 runs **SciPy differential evolution** to a
global optimum from each of the K perturbed starts, then summarizes the ensemble
of K optimized layouts with a per-position mean and std.

Detectors use the **(North, East)** convention (see THEORY.md §3.5): the layout
is 100 North + 100 East, bounded by the mountain North bbox and the East span
``[east_lo, east_hi]``; each candidate is projected to the mountain
(``project_to_mountain_ne``) before scoring. Requires NE-trained FNN/recon
(``01_build_dataset_northeast.py`` → retrained Steps 2–3).

Per scheme:

1.  Sample the scheme's initial layout (``sample_initial_layout_ne``) and create
    K = ``N_CHAINS`` Gaussian perturbations of it (std ``INIT_OVERDISP_SIGMA``,
    projected back to the mountain).
2.  Run **differential evolution** (``DE_MAXITER``) from each perturbed start on
    a FIXED primary batch (deterministic objective) → K optimized layouts.
3.  **Align** the K layouts by closest position (Hungarian) so each output group
    is the same physical position, not the same detector index.
4.  Per aligned group: **mean and std** of (North, East) across the K runs.

The "combined" run pools the K starts from every scheme.

Artifacts (per scheme + "combined") land in
``<OPT_FOLDER>_de_ensemble_{scheme}/`` (same set as the L-BFGS ensemble).

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 04_optimize_differential_evolution.py
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
from modules_v6.reconstruction import Reconstruction
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
# modules_v6/opt_core.py; the figures live in plots/opt_plotting.py (loaded by
# path — its package dir isn't importable by name). DE keeps the _plot_ensemble /
# _plot_density_heatmap names so replot_de_ensemble_up.py still finds them.
from modules_v6.opt_core import (
    primary_to_physical_labels, utility_of_xy, align_to_reference, load_models,
    W_THETA, W_PHI, W_E, W_PR, W_DIV,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD, GEOMETRY_PATH_RESOLVED,
)
_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(_HERE, "plots", "opt_plotting.py"))
_plt = importlib.util.module_from_spec(_plt_spec); _plt_spec.loader.exec_module(_plt)
_plot_ensemble        = _plt.plot_ensemble
_plot_density_heatmap = _plt.plot_density_heatmap


# ── Config ───────────────────────────────────────────────────────────────────
INIT_SCHEMES         = ("grid", "center")
RUN_COMBINED         = True
COMBINED_SCHEME_NAME = "combined"
OPT_DIR_TEMPLATE     = OPT_FOLDER + "_de_ensemble_{scheme}"
# Recon dir to load (DeepSets recon from 03_train_recon_deepsets.py). Overridable
# with --recon_folder (exact path).
RECON_DIR            = RECON_FOLDER + "_deepsets"

# K perturbed restarts per scheme.
N_CHAINS            = 1
INIT_OVERDISP_SIGMA = 1000.0  # metres — per-restart init spread around scheme init

# Differential evolution (replaces the Adam warm-start + L-BFGS refine)
DE_MAXITER          = 1000
DE_POPSIZE          = 10       # population = popsize × (2·n_det) candidates / generation
DE_TOL              = 1e-4
DE_MUTATION         = (0.5, 1.0)
DE_RECOMBINATION    = 0.7
DE_BATCH_PRIMARIES  = 512     # FIXED batch → deterministic objective for the search

# Composite weights (W_*) + reconstructability thresholds + GEOMETRY_PATH_RESOLVED
# are imported from modules_v6/opt_core.py (shared across the 04 optimizers).
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_chain_inits(init_x: torch.Tensor, init_y: torch.Tensor,
                       K: int, generator: torch.Generator) -> torch.Tensor:
    """K overdispersed starts around (init_x, init_y). Returns (K, 2*n_det) on DEVICE."""
    base = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach()  # (D,)
    perturb = torch.randn(
        K, base.numel(), generator=generator, device="cpu",
    ).to(DEVICE) * INIT_OVERDISP_SIGMA
    return base.unsqueeze(0) + perturb                                        # (K, D)


def _perturbed_de_runs(scheme: str, K: int, generator: torch.Generator,
                       mountain, fnn, recon, primary_all, n_total_primaries):
    """K pre-perturbations of the scheme init → K starting layouts for DE.

    Mirror of `_perturbed_adam_runs`, but differential evolution is the optimizer
    (run later, in `_run_one_scheme`), so here we only build the K perturbed
    starts — there is no Adam pre-optimization stage. Returns
    (starts, start_logs, perturbed_inits, _unused), each length K, so the
    downstream signature matches the L-BFGS ensemble."""
    N_np, E_np = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
    N_t = torch.as_tensor(N_np, dtype=torch.float32)
    E_t = torch.as_tensor(E_np, dtype=torch.float32)
    N_t, E_t = project_to_mountain_ne(mountain, N_t, E_t)
    chains_init = _build_chain_inits(N_t, E_t, K, generator)                  # (K, D)

    starts, start_logs, perturbed_inits, _unused = [], [], [], []
    for k in range(K):
        xk = chains_init[k, :N_DETECTORS].cpu()
        yk = chains_init[k, N_DETECTORS:].cpu()
        xk, yk = project_to_mountain_ne(mountain, xk, yk)
        perturbed_inits.append((xk.float().clone(), yk.float().clone()))
        # No Adam warm-start under DE — the perturbed init IS the DE start.
        starts.append((xk.float().clone(), yk.float().clone()))
        start_logs.append([])
        _unused.append(None)
        print(f"\n[perturb→de] scheme={scheme}  chain {k+1}/{K}  "
              f"N in [{xk.min():.1f}, {xk.max():.1f}]  E in [{yk.min():.1f}, {yk.max():.1f}]")
    return starts, start_logs, perturbed_inits, _unused


def de_refine(init_x: torch.Tensor,
              init_y: torch.Tensor,
              fnn,
              recon: torch.nn.Module,
              primary_fixed: torch.Tensor,
              mountain,
              bounds,
              seed: int):
    """Differential-evolution maximize U from (init_x, init_y) on a fixed batch.

    Mirror of `lbfgs_refine`: optimises the same objective, projects the optimum
    to the mountain, and re-scores on the same fixed batch. Returns
    (x_proj, y_proj, U_proj, iter_log, gen_hist) where `iter_log` is one entry per
    DE generation (best-so-far + utility parts) and `gen_hist` is the best-U per
    generation — the DE analogue of the L-BFGS iter log / gradient history."""
    x0 = torch.cat([init_x, init_y], dim=0).detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()   # gradient-free DE; opt_core.utility_of_xy is not no_grad-wrapped
    def _score(flat):
        x_det = torch.as_tensor(flat[:N_DETECTORS], dtype=torch.float32, device=DEVICE)
        y_det = torch.as_tensor(flat[N_DETECTORS:], dtype=torch.float32, device=DEVICE)
        x_det, y_det = project_to_mountain_ne(mountain, x_det, y_det)
        U, r, parts = utility_of_xy(x_det, y_det, primary_fixed, fnn, recon)
        return float(U.item()), float(r.mean().item()), parts

    iter_log, gen_hist = [], []
    best = {"U": -float("inf"), "x": x0.copy()}

    def objective(flat):
        U, _, _ = _score(flat)
        if U > best["U"]:
            best["U"] = U
            best["x"] = np.asarray(flat, dtype=np.float64).copy()
        return -U

    def callback(xk, convergence=None):
        # One log entry per generation, evaluated at the running best (mirrors the
        # per-iter logging of lbfgs_refine).
        U, r_mean, parts = _score(best["x"])
        iter_log.append(dict(
            iter=len(iter_log), U=U, r_mean=r_mean,
            u_theta=float(parts["u_theta"].item()),
            u_phi=float(parts["u_phi"].item()),
            u_e=float(parts["u_e"].item()),
            u_pr=float(parts["u_pr"].item()),
        ))
        gen_hist.append(U)

    differential_evolution(
        objective, bounds, x0=x0, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
        tol=DE_TOL, mutation=DE_MUTATION, recombination=DE_RECOMBINATION,
        seed=seed, polish=False, init="latinhypercube", updating="immediate",
        workers=1, callback=callback,
    )

    # Project the optimum to the mountain and re-score on the same fixed batch.
    with torch.no_grad():
        x_cpu = torch.as_tensor(best["x"][:N_DETECTORS], dtype=torch.float32)
        y_cpu = torch.as_tensor(best["x"][N_DETECTORS:], dtype=torch.float32)
        x_proj, y_proj = project_to_mountain_ne(mountain, x_cpu, y_cpu)
        U_proj, _, _ = utility_of_xy(
            x_proj.to(DEVICE), y_proj.to(DEVICE), primary_fixed, fnn, recon,
        )
    gen_hist = torch.as_tensor(gen_hist) if gen_hist else torch.zeros(0)
    return x_proj.float(), y_proj.float(), float(U_proj.item()), iter_log, gen_hist


def _run_one_scheme(scheme: str,
                    mountain,
                    fnn,
                    recon: torch.nn.Module,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    per_source, surface=None):
    """Pre-computed starts → DE-refine each → align ensemble → mean/std.

    `per_source` is {source_label: (starts, start_logs, perturbed_inits, _unused)}.
    A single entry = per-scheme run; multiple entries = the combined run."""
    opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
    os.makedirs(opt_dir, exist_ok=True)
    is_combined = len(per_source) > 1
    print("-" * 72)
    print(f"[run] scheme={scheme}"
          f"{'  (sources=' + str(list(per_source)) + ')' if is_combined else ''}  ->  {opt_dir}")

    # Flatten starts across all sources (track which source each came from).
    all_starts, all_start_logs, source_per_run = [], [], []
    for src, (starts, logs, _inits, _unused) in per_source.items():
        for (bx, by), log in zip(starts, logs):
            all_starts.append((bx, by))
            all_start_logs.append(log)
            source_per_run.append(src)

    # One fixed primary batch for the WHOLE scheme so all refinements + scores
    # share the same deterministic objective and are directly comparable.
    g = torch.Generator().manual_seed(SEED)
    idx_fixed = torch.randint(0, n_total_primaries, (DE_BATCH_PRIMARIES,), generator=g)
    primary_fixed = primary_all[idx_fixed].to(DEVICE)

    # DE bounds: 100 North in [n_min, n_max], then 100 East in [east_lo, east_hi],
    # each widened by the NE projection tolerance: project_to_mountain_ne keeps
    # any point within max_gap of a centroid, so valid starts can sit up to
    # ~max_gap OUTSIDE the tight centroid bbox — and scipy requires x0 inside
    # the bounds. Candidates are mountain-projected before scoring, so the
    # widened box never lets the optimum leave the mountain.
    margin = _ne_max_gap(mountain)
    print(f"[bounds] bbox widened by max_gap={margin:.1f} m")
    bounds = ([(mountain.n_min - margin, mountain.n_max + margin)] * N_DETECTORS +
              [(mountain.east_lo - margin, mountain.east_hi + margin)] * N_DETECTORS)

    # Stage 2: differential evolution from every start.
    refined, de_logs, refined_U, all_de_hists = [], [], [], []
    for k, (bx, by) in enumerate(all_starts):
        print(f"[de] refine {k+1}/{len(all_starts)}  (src={source_per_run[k]})")
        xp, yp, Up, lg, hist = de_refine(bx, by, fnn, recon, primary_fixed, mountain,
                                         bounds, seed=SEED + k)
        refined.append((xp, yp))
        refined_U.append(Up)
        de_logs.append(lg)
        all_de_hists.append(hist)
        print(f"  [de] refine {k} U={Up:+.3f}  ({len(lg)} generations)")

    # Per-run best-U-per-generation history (DE analogue of the gradient diagnostic).
    de_hist_per_run = [h.tolist() if hasattr(h, "tolist") else list(h) for h in all_de_hists]

    # Build the (K, n_det, 2) ensemble and align by closest position.
    layouts_xy = np.stack(
        [np.stack([xp.numpy(), yp.numpy()], axis=-1) for xp, yp in refined], axis=0,
    )                                                                # (K, n_det, 2)
    ref_idx = int(np.argmax(refined_U))                              # best-U run = reference
    aligned, perms = align_to_reference(layouts_xy, ref_idx)
    mean_xy = aligned.mean(axis=0)                                   # (n_det, 2)
    std_xy  = aligned.std(axis=0)                                    # (n_det, 2)

    best_x, best_y = refined[ref_idx]
    best_src = source_per_run[ref_idx]
    print(f"[ensemble] K={len(refined)}  best U={refined_U[ref_idx]:+.3f} "
          f"(run {ref_idx}, src={best_src})  "
          f"mean σN={std_xy[:,0].mean():.1f}m σE={std_xy[:,1].mean():.1f}m")

    # ── Persist artifacts (same set as the L-BFGS ensemble) ──────────────────
    torch.save({"x": best_x, "y": best_y, "U": refined_U[ref_idx],
                "run": ref_idx, "source": best_src},
               os.path.join(opt_dir, "layout_best.pt"))
    torch.save({"mean_x": torch.as_tensor(mean_xy[:, 0]),
                "mean_y": torch.as_tensor(mean_xy[:, 1]),
                "std_x":  torch.as_tensor(std_xy[:, 0]),
                "std_y":  torch.as_tensor(std_xy[:, 1])},
               os.path.join(opt_dir, "layout_mean.pt"))
    torch.save({"aligned": torch.as_tensor(aligned),          # (K, n_det, 2)
                "perms": torch.as_tensor(perms),
                "utilities": torch.as_tensor(refined_U),
                "source_per_run": source_per_run,
                "ref_idx": ref_idx},
               os.path.join(opt_dir, "layouts_all.pt"))

    with open(os.path.join(opt_dir, "optimize_log.json"), "w") as f:
        json.dump({
            "scheme": scheme,
            "sources": list(per_source),
            "source_per_run": source_per_run,
            "ref_idx": ref_idx,
            "ref_source": best_src,
            "refined_U": refined_U,
            "best_U": refined_U[ref_idx],
            "ensemble_stats": dict(
                mean_std_x=float(std_xy[:, 0].mean()),
                mean_std_y=float(std_xy[:, 1].mean()),
                max_std_x=float(std_xy[:, 0].max()),
                max_std_y=float(std_xy[:, 1].max()),
            ),
            "de_best_U_history": de_hist_per_run,    # per run: best U per generation
            "de_logs": de_logs,
            "config": dict(
                n_chains=N_CHAINS, init_overdisp_sigma=INIT_OVERDISP_SIGMA,
                de_maxiter=DE_MAXITER, de_popsize=DE_POPSIZE, de_tol=DE_TOL,
                de_mutation=list(DE_MUTATION), de_recombination=DE_RECOMBINATION,
                de_batch_primaries=DE_BATCH_PRIMARIES,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                seed=SEED,
            ),
        }, f, indent=2)

    _plt.plot_curves_de(de_logs, os.path.join(opt_dir, "optimize_curves.png"))
    _plt.plot_components_de(de_logs, os.path.join(opt_dir, "utility_components.png"))
    _plt.plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                       mountain, os.path.join(opt_dir, "layout_ensemble.png"), surface=surface)
    _plt.plot_density_heatmap(aligned, best_x, best_y,
                       mountain, os.path.join(opt_dir, "layout_density.png"), surface=surface)

    print(f"[done] scheme={scheme}  best U={refined_U[ref_idx]:+.3f}  "
          f"σ̄=({std_xy[:,0].mean():.1f}, {std_xy[:,1].mean():.1f}) m  ({opt_dir})")
    return dict(scheme=scheme, best_U=refined_U[ref_idx],
                mean_std_x=float(std_xy[:, 0].mean()),
                mean_std_y=float(std_xy[:, 1].mean()),
                opt_dir=opt_dir)


def main():
    global N_CHAINS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=N_CHAINS,
                    help="Number of perturbed DE starts per init scheme (default from config).")
    ap.add_argument("--recon_folder", type=str, default=None,
                    help="Override RECON_FOLDER from constants.py "
                         "(e.g. to swap between flat MLP and DeepSets recon).")
    ap.add_argument("--opt_suffix", type=str, default="",
                    help="Suffix appended to output directory name "
                         "(e.g. '_mlp' to get de_ensemble_mlp_{scheme}/).")
    args = ap.parse_args()
    N_CHAINS = int(args.chains)

    if args.recon_folder:
        global RECON_DIR
        RECON_DIR = args.recon_folder
        print(f"[recon_folder] overriding recon dir -> {args.recon_folder}")

    if args.opt_suffix:
        global OPT_DIR_TEMPLATE
        OPT_DIR_TEMPLATE = OPT_FOLDER + "_de_ensemble" + args.opt_suffix + "_{scheme}"

    print("=" * 72)
    print("v6/04_optimize_differential_evolution.py — perturbed starts + DE ensemble (North, East)")
    print("=" * 72)
    print(f"device       : {DEVICE}")
    print(f"init schemes : {INIT_SCHEMES}")
    print(f"chains (K)   : {N_CHAINS}  (init σ={INIT_OVERDISP_SIGMA} m)")
    print(f"DE           : maxiter={DE_MAXITER}  popsize={DE_POPSIZE}  batch={DE_BATCH_PRIMARIES}")

    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    n_total_primaries = int(primary_all.shape[0])
    print(f"[load] {n_total_primaries} primaries")

    fnn, recon = load_models(DEVICE, recon_dir=RECON_DIR)

    mountain = load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    # Differentiable Up = g(North, East) used only to project DE layouts (native
    # to the North–East plane) into the (North, Up) cross section for the plots,
    # so they line up with the L-BFGS ensemble figures.
    surface = SurfaceUpMap.from_mountain(mountain).to(DEVICE)

    results = []
    per_scheme = {}     # scheme -> (starts, start_logs, perturbed_inits, _unused)
    for scheme in INIT_SCHEMES:
        print()
        print("=" * 72)
        print(f"init scheme: {scheme}")
        print("=" * 72)
        torch.manual_seed(SEED); np.random.seed(SEED)
        g = torch.Generator().manual_seed(SEED)
        per_scheme[scheme] = _perturbed_de_runs(
            scheme, N_CHAINS, g, mountain, fnn, recon, primary_all, n_total_primaries,
        )
        results.append(_run_one_scheme(
            scheme, mountain, fnn, recon, primary_all, n_total_primaries,
            {scheme: per_scheme[scheme]}, surface=surface,
        ))

    if RUN_COMBINED and len(per_scheme) > 1:
        print()
        print("=" * 72)
        print(f"init scheme: {COMBINED_SCHEME_NAME} (sources={list(per_scheme)})")
        print("=" * 72)
        results.append(_run_one_scheme(
            COMBINED_SCHEME_NAME, mountain, fnn, recon, primary_all, n_total_primaries,
            per_scheme, surface=surface,
        ))

    print()
    print("=" * 72)
    print("summary")
    print("=" * 72)
    for r in results:
        print(f"  {r['scheme']:<10}  best U={r['best_U']:+.3f}  "
              f"σ̄=({r['mean_std_x']:.1f}, {r['mean_std_y']:.1f}) m  ->  {r['opt_dir']}")


if __name__ == "__main__":
    main()
