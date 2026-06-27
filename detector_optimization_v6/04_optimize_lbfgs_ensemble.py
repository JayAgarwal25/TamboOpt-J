"""Optimize detector positions: pre-Adam perturbation, then L-BFGS ensemble.

Frequentist sibling of ``04_optimize_hmc_chains.py``. Instead of sampling a
posterior with NUTS, stage 2 runs **L-BFGS to a local optimum from each of the
K perturbed Adam warm-starts**, then summarizes the ensemble of K optimized
layouts with a per-position mean and std.

Per scheme:

1.  Sample the scheme's initial layout (`mountain.sample_initial_layout`) and
    create K = `N_CHAINS` Gaussian perturbations of it (std
    `INIT_OVERDISP_SIGMA`, projected back to the mountain).
2.  Run Adam (`N_ADAM_EPOCHS`) independently from each perturbed start → K
    Adam-best layouts.
3.  Run L-BFGS (`LBFGS_MAX_ITER`) from each Adam-best on a FIXED primary batch
    (deterministic objective for the line search) → K refined layouts.
4.  **Align** the K refined layouts so each output group corresponds to the
    same *physical position*, not the same detector index. Because the FNN /
    recon are permutation-equivariant, detector index i is not the same unit
    across runs — so we match each run's detectors to a reference layout by
    closest position (Hungarian / `linear_sum_assignment`). This makes the
    grouping network-input invariant.
5.  Per aligned group: **mean and std** of (x, y) across the K runs.

The "combined" run pools the K Adam-bests from every scheme, refines all of
them with L-BFGS, and aligns the full K * len(INIT_SCHEMES) ensemble.

Artifacts (per scheme + "combined") land in
``<OPT_FOLDER>_lbfgs_ensemble_{scheme}/``:

    layout_best.pt          highest-U L-BFGS layout (mountain-projected)
    layout_mean.pt          per-group mean position + std (aligned ensemble)
    layouts_all.pt          aligned (K, n_det, 2) + per-run U + source + perm
    optimize_log.json       Adam + L-BFGS logs + ensemble stats + config
    optimize_curves.png     all Adam chains U + all L-BFGS refinements U
    layout_ensemble.png     mountain top-down: ensemble points + mean + 1σ ellipses

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 04_optimize_lbfgs_ensemble.py
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

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.dual_surrogate import DualSpeciesSurrogate
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER, OPT_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules.layout_optimization import LearnableXY
from modules_v4.tr_geometry      import load_tr_mountain
from modules_v6.tr_geometry_ne   import project_to_mountain_ne, sample_initial_layout_ne
from modules_v6.tr_surface_map_ne import SurfaceUpMap

# Shared optimizer core (objective, alignment, model loading, the gradient-turn
# diagnostic, constants) lives in modules_v6/opt_core.py; the figures live in
# plots/opt_plotting.py (loaded by path). utility_of_xy is NOT no_grad-wrapped,
# so Adam / L-BFGS backprop through it here.
from modules_v6.opt_core import (
    utility_of_xy, align_to_reference, consecutive_cos_distance, load_models,
    W_THETA, W_PHI, W_E, W_PR, W_DIV,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD,
)
_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(_HERE, "plots", "opt_plotting.py"))
_plt = importlib.util.module_from_spec(_plt_spec); _plt_spec.loader.exec_module(_plt)


# ── Config ───────────────────────────────────────────────────────────────────
INIT_SCHEMES         = ("grid", "center")
RUN_COMBINED         = True
COMBINED_SCHEME_NAME = "combined"
OPT_DIR_TEMPLATE     = OPT_FOLDER + "_lbfgs_ensemble_{scheme}"
# Recon dir to load (DeepSets recon from 03_train_recon_deepsets.py). Overridable
# with --recon_folder (exact path). utility_of_xy feeds recon (B, n_det, 4).
RECON_DIR            = RECON_FOLDER + "_deepsets"

# K perturbed restarts per scheme.
N_CHAINS            = 15
INIT_OVERDISP_SIGMA = 1000.0  # metres — per-restart init spread around scheme init

# Adam warm-start
N_ADAM_EPOCHS       = 5_000
PRIMARIES_PER_STEP  = 256
ADAM_LR             = 1.0
GRAD_CLIP           = 100.0
ADAM_LOG_EVERY      = 100

# Gradient-direction diagnostic: window (in steps) for vector-averaging the raw
# gradients before the consecutive-step cosine distance. Averaging the gradient
# VECTORS over W steps cancels zero-mean minibatch noise before the (nonlinear)
# cosine, removing the noise-inflation bias instead of merely smoothing it.
# 1 = no averaging (raw, noisy).
GRAD_COS_WINDOW     = 10

# L-BFGS refinement (stage 2)
LBFGS_MAX_ITER       = 1_500
LBFGS_LR             = 1.0
LBFGS_HISTORY_SIZE   = 20
LBFGS_BATCH_PRIMARIES = 512    # FIXED batch → deterministic objective for line search

# Composite weights (W_*) + reconstructability thresholds are imported from
# modules_v6/opt_core.py (shared across the 04 optimizers).
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Density heatmap colorbar upper limit (plots [0, DENSITY_VMAX]); keeps faint
# structure from being washed out by a few hot cells. <=0 → auto-scale.
DENSITY_VMAX = 0.2


def adam_warm_start(scheme: str,
                    mountain,
                    fnn: DualSpeciesSurrogate,
                    recon: torch.nn.Module,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    init_override):
    """N_ADAM_EPOCHS of Adam with mountain projection. Returns:
       (best_x, best_y, init_x, init_y, log, grad_hist). `init_override=(x, y)`
       is the (already mountain-projected) starting layout; `scheme` is a log
       label. `grad_hist` is a (N_ADAM_EPOCHS, 2*n_det) CPU tensor of the flat
       parameter gradient at each step (for cross-run gradient diagnostics)."""
    N_init, E_init = init_override
    N_init = N_init.float()
    E_init = E_init.float()
    print(f"[adam] init {scheme}  N in [{N_init.min():.1f}, {N_init.max():.1f}]  "
          f"E in [{E_init.min():.1f}, {E_init.max():.1f}]")

    xy_module = LearnableXY(N_init, E_init, device=str(DEVICE)).to(DEVICE)
    optimizer = torch.optim.Adam(xy_module.parameters(), lr=ADAM_LR)

    log = []
    grad_hist = []
    best_u = -float("inf")
    best_x = N_init.clone()
    best_y = E_init.clone()

    for epoch in range(N_ADAM_EPOCHS):
        idx = torch.randint(0, n_total_primaries, (PRIMARIES_PER_STEP,))
        primary_batch = primary_all[idx].to(DEVICE)

        x_det, y_det = xy_module()
        U, r, parts = utility_of_xy(x_det, y_det, primary_batch, fnn, recon)
        loss = -U

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Flat gradient (x then y) before clipping — cosine is scale-invariant,
        # so clipping (uniform rescale) wouldn't change it anyway.
        grad_hist.append(
            torch.cat([xy_module.x.grad.detach().reshape(-1),
                       xy_module.y.grad.detach().reshape(-1)]).cpu()
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        # Project to mountain surface.
        with torch.no_grad():
            N_cpu = xy_module.x.detach().cpu()
            E_cpu = xy_module.y.detach().cpu()
            N_new, E_new = project_to_mountain_ne(mountain, N_cpu, E_cpu)
            xy_module.x.data.copy_(N_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(E_new.to(DEVICE).to(xy_module.y.dtype))

        u_val = float(U.item())
        if u_val > best_u:
            best_u = u_val
            best_x = xy_module.x.detach().cpu().clone()
            best_y = xy_module.y.detach().cpu().clone()

        log.append(dict(
            epoch=epoch + 1, U=u_val, r_mean=float(r.mean().item()),
            u_theta=float(parts["u_theta"].item()),
            u_phi=float(parts["u_phi"].item()),
            u_e=float(parts["u_e"].item()),
            u_pr=float(parts["u_pr"].item()),
        ))
        if epoch == 0 or (epoch + 1) % ADAM_LOG_EVERY == 0 or epoch == N_ADAM_EPOCHS - 1:
            print(f"  [adam {epoch+1:4d}/{N_ADAM_EPOCHS}] U={u_val:+.3f}")

    print(f"[adam] best U={best_u:+.3f}")
    grad_hist = torch.stack(grad_hist, dim=0) if grad_hist else torch.zeros(0)
    return best_x, best_y, N_init, E_init, log, grad_hist


def _build_chain_inits(init_x: torch.Tensor, init_y: torch.Tensor,
                       K: int, generator: torch.Generator) -> torch.Tensor:
    """K overdispersed starts around (init_x, init_y). Returns (K, 2*n_det) on DEVICE."""
    base = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach()  # (D,)
    perturb = torch.randn(
        K, base.numel(), generator=generator, device="cpu",
    ).to(DEVICE) * INIT_OVERDISP_SIGMA
    return base.unsqueeze(0) + perturb                                        # (K, D)


def _perturbed_adam_runs(scheme: str, K: int, generator: torch.Generator,
                         mountain, fnn, recon, primary_all, n_total_primaries,
                         init_center=None):
    """K pre-Adam perturbations of the scheme init → K Adam runs.

    Returns (adam_bests, adam_logs, perturbed_inits, adam_grads), each length K.
    adam_grads[k] is the (N_ADAM_EPOCHS, 2*n_det) per-step gradient history.

    If init_center is a (N_DETECTORS, 2) tensor with (North, East) per detector,
    all K chains are warm-started from that layout with small N(0, 10m) per-chain
    diversity perturbations instead of using the normal scheme initialization.
    """
    if init_center is not None:
        N_t = init_center[:, 0].float()
        E_t = init_center[:, 1].float()
        # Small N(0, 10m) per-chain perturbations for diversity around the warm-start.
        base = torch.cat([N_t.to(DEVICE), E_t.to(DEVICE)], dim=0).detach()
        small_noise = torch.randn(K, base.numel(), generator=generator,
                                  device="cpu").to(DEVICE) * 10.0
        chains_init = base.unsqueeze(0) + small_noise
    else:
        N_np, E_np = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
        N_t = torch.as_tensor(N_np, dtype=torch.float32)
        E_t = torch.as_tensor(E_np, dtype=torch.float32)
        chains_init = _build_chain_inits(N_t, E_t, K, generator)              # (K, D)

    adam_bests, adam_logs, perturbed_inits, adam_grads = [], [], [], []
    for k in range(K):
        xk = chains_init[k, :N_DETECTORS].cpu()
        yk = chains_init[k, N_DETECTORS:].cpu()
        xk, yk = project_to_mountain_ne(mountain, xk, yk)
        perturbed_inits.append((xk.float().clone(), yk.float().clone()))
        print(f"\n[perturb→adam] scheme={scheme}  chain {k+1}/{K}")
        bx, by, _, _, log, ghist = adam_warm_start(
            scheme=scheme, mountain=mountain, fnn=fnn, recon=recon,
            primary_all=primary_all, n_total_primaries=n_total_primaries,
            init_override=(xk, yk),
        )
        adam_bests.append((bx, by))
        adam_logs.append(log)
        adam_grads.append(ghist)
    return adam_bests, adam_logs, perturbed_inits, adam_grads


def lbfgs_refine(init_x: torch.Tensor,
                 init_y: torch.Tensor,
                 fnn: DualSpeciesSurrogate,
                 recon: torch.nn.Module,
                 primary_fixed: torch.Tensor,
                 mountain):
    """L-BFGS-maximize U from (init_x, init_y) on a fixed primary batch.

    Runs unconstrained (the line search needs a smooth objective), then
    projects the optimum back onto the mountain and re-scores it on the same
    fixed batch. Returns (x_proj, y_proj, U_proj, iter_log, grad_hist) where
    grad_hist is a (n_closure_calls, 2*n_det) CPU tensor of the flat gradient
    at each closure evaluation (for cross-run gradient diagnostics)."""
    xy = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach().clone()
    xy.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [xy], lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE, line_search_fn="strong_wolfe",
        tolerance_grad=1e-11,tolerance_change=1e-13,
    )

    iter_log = []
    grad_hist = []

    class _NonFiniteLoss(Exception):
        pass

    def closure():
        optimizer.zero_grad()
        x_det = xy[:N_DETECTORS]
        y_det = xy[N_DETECTORS:]
        U, r, parts = utility_of_xy(x_det, y_det, primary_fixed, fnn, recon)
        loss = -U
        if not torch.isfinite(loss):
            raise _NonFiniteLoss
        loss.backward()
        grad_hist.append(xy.grad.detach().reshape(-1).cpu())   # (2*n_det,)
        iter_log.append(dict(
            iter=len(iter_log), U=float(U.item()), r_mean=float(r.mean().item()),
            u_theta=float(parts["u_theta"].item()),
            u_phi=float(parts["u_phi"].item()),
            u_e=float(parts["u_e"].item()),
            u_pr=float(parts["u_pr"].item()),
        ))
        return loss

    try:
        optimizer.step(closure)
    except _NonFiniteLoss:
        print(f"  [lbfgs] non-finite loss after {len(iter_log)} closure calls — aborting step")

    # A diverged line search can leave NaN in xy; project_to_mountain passes
    # NaN through unsnapped, which would poison the ensemble alignment.
    # Fall back to the (already mountain-projected) Adam-best init.
    with torch.no_grad():
        if not torch.isfinite(xy).all():
            print("  [lbfgs] non-finite iterate — falling back to the Adam-best init")
            xy.data = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0)

    # Project the optimum to the mountain and re-score on the same fixed batch.
    with torch.no_grad():
        x_cpu = xy[:N_DETECTORS].detach().cpu()
        y_cpu = xy[N_DETECTORS:].detach().cpu()
        x_proj, y_proj = project_to_mountain_ne(mountain, x_cpu, y_cpu)
        U_proj, _, _ = utility_of_xy(
            x_proj.to(DEVICE), y_proj.to(DEVICE), primary_fixed, fnn, recon,
        )
    grad_hist = torch.stack(grad_hist, dim=0) if grad_hist else torch.zeros(0)
    return x_proj.float(), y_proj.float(), float(U_proj.item()), iter_log, grad_hist


def _run_one_scheme(scheme: str,
                    mountain,
                    fnn: DualSpeciesSurrogate,
                    recon: torch.nn.Module,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    per_source):
    """Pre-computed Adam-bests → L-BFGS refine each → align ensemble → mean/std.

    `per_source` is {source_label: (adam_bests, adam_logs, perturbed_inits)}.
    A single entry = per-scheme run; multiple entries = the combined run."""
    opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
    os.makedirs(opt_dir, exist_ok=True)
    is_combined = len(per_source) > 1
    print("-" * 72)
    print(f"[run] scheme={scheme}"
          f"{'  (sources=' + str(list(per_source)) + ')' if is_combined else ''}  ->  {opt_dir}")

    # Flatten Adam-bests across all sources (track which source each came from).
    all_bests, all_adam_logs, all_adam_grads, source_per_run = [], [], [], []
    for src, (bests, logs, _inits, agrads) in per_source.items():
        for (bx, by), log, ag in zip(bests, logs, agrads):
            all_bests.append((bx, by))
            all_adam_logs.append(log)
            all_adam_grads.append(ag)
            source_per_run.append(src)

    # One fixed primary batch for the WHOLE scheme so all refinements + scores
    # share the same deterministic objective and are directly comparable.
    g = torch.Generator().manual_seed(SEED)
    idx_fixed = torch.randint(0, n_total_primaries, (LBFGS_BATCH_PRIMARIES,), generator=g)
    primary_fixed = primary_all[idx_fixed].to(DEVICE)

    # Stage 2: L-BFGS refine every Adam-best.
    refined, lbfgs_logs, refined_U, all_lbfgs_grads = [], [], [], []
    for k, (bx, by) in enumerate(all_bests):
        print(f"[lbfgs] refine {k+1}/{len(all_bests)}  (src={source_per_run[k]})")
        xp, yp, Up, lg, ghist = lbfgs_refine(bx, by, fnn, recon, primary_fixed, mountain)
        refined.append((xp, yp))
        refined_U.append(Up)
        lbfgs_logs.append(lg)
        all_lbfgs_grads.append(ghist)
        print(f"  [lbfgs] refine {k} U={Up:+.3f}  ({len(lg)} closure calls)")

    # Per-run consecutive-step gradient cosine distance (Adam + L-BFGS phases),
    # W-step vector-averaged to suppress minibatch-noise inflation.
    adam_cos_per_run  = [consecutive_cos_distance(g, GRAD_COS_WINDOW).tolist()
                         for g in all_adam_grads]
    lbfgs_cos_per_run = [consecutive_cos_distance(g, GRAD_COS_WINDOW).tolist()
                         for g in all_lbfgs_grads]

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
          f"mean σx={std_xy[:,0].mean():.1f}m σy={std_xy[:,1].mean():.1f}m")

    # ── Persist artifacts ───────────────────────────────────────────────────
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
            "grad_cos_consecutive": dict(
                adam=adam_cos_per_run,    # per run: 1 - cos(g_t, g_{t-1}) over Adam steps
                lbfgs=lbfgs_cos_per_run,  # per run: same over L-BFGS closure calls
            ),
            "adam_logs": all_adam_logs,
            "lbfgs_logs": lbfgs_logs,
            "config": dict(
                n_chains=N_CHAINS, init_overdisp_sigma=INIT_OVERDISP_SIGMA,
                n_adam_epochs=N_ADAM_EPOCHS, primaries_per_step=PRIMARIES_PER_STEP,
                adam_lr=ADAM_LR, grad_clip=GRAD_CLIP,
                lbfgs_max_iter=LBFGS_MAX_ITER, lbfgs_lr=LBFGS_LR,
                lbfgs_history_size=LBFGS_HISTORY_SIZE,
                lbfgs_batch_primaries=LBFGS_BATCH_PRIMARIES,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                seed=SEED,
            ),
        }, f, indent=2)

    _plt.plot_curves_lbfgs(all_adam_logs, lbfgs_logs, all_adam_grads, all_lbfgs_grads,
                           os.path.join(opt_dir, "optimize_curves.png"),
                           grad_cos_window=GRAD_COS_WINDOW)
    _plt.plot_components_lbfgs(all_adam_logs, lbfgs_logs,
                              os.path.join(opt_dir, "utility_components.png"))
    # Render the ensemble + density in the (North, Up) cross section. The
    # optimiser works in (North, East); SurfaceUpMap projects East -> Up =
    # g(North, East) so detectors sit ON the mountain profile (CPU is fine —
    # plotting is the last, read-only step). DENSITY_VMAX clamps the heatmap
    # colorbar to [0, DENSITY_VMAX]; <=0 auto-scales.
    try:
        surface = SurfaceUpMap.from_mountain(mountain).to("cpu")
        vmax = DENSITY_VMAX if DENSITY_VMAX and DENSITY_VMAX > 0 else None
        _plt.plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                       mountain, os.path.join(opt_dir, "layout_ensemble.png"),
                       surface=surface, title_kind="L-BFGS ensemble")
        _plt.plot_density_heatmap(aligned, best_x, best_y,
                       mountain, os.path.join(opt_dir, "layout_density.png"),
                       surface=surface, vmax=vmax)
    except Exception as exc:
        print(f"[plot] ensemble/density skipped ({exc!r})")

    print(f"[done] scheme={scheme}  best U={refined_U[ref_idx]:+.3f}  "
          f"σ̄=({std_xy[:,0].mean():.1f}, {std_xy[:,1].mean():.1f}) m  ({opt_dir})")
    return dict(scheme=scheme, best_U=refined_U[ref_idx],
                best_x=best_x, best_y=best_y,
                mean_std_x=float(std_xy[:, 0].mean()),
                mean_std_y=float(std_xy[:, 1].mean()),
                opt_dir=opt_dir)


def main():
    global N_CHAINS, N_ADAM_EPOCHS, LBFGS_MAX_ITER, DENSITY_VMAX
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=N_CHAINS)
    ap.add_argument("--adam-epochs", type=int, default=N_ADAM_EPOCHS)
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_MAX_ITER)
    ap.add_argument("--density-vmax", type=float, default=DENSITY_VMAX,
                    help="density heatmap colorbar upper limit (plots [0, vmax]); "
                         "pass <=0 to auto-scale (default from config)")
    ap.add_argument("--save_best_layout", type=str, default=None,
                    help="Path to save the globally best layout as a "
                         "(N_DETECTORS, 2) tensor with (North, East) per detector.")
    ap.add_argument("--init_from", type=str, default=None,
                    help="Path to a (N_DETECTORS, 2) layout tensor (or dict with "
                         "'x'/'y' keys). Warm-starts all chains from this layout "
                         "with small N(0, 10m) diversity perturbations.")
    ap.add_argument("--fnn_folder", type=str, default=None,
                    help="Override FNN_FOLDER from constants.py (use a fine-tuned "
                         "checkpoint directory from the adaptive retraining loop).")
    ap.add_argument("--recon_folder", type=str, default=None,
                    help="Exact path to the recon dir to load (default: "
                         "RECON_FOLDER + '_deepsets', the 03_train_recon_deepsets.py output).")
    ap.add_argument("--opt_suffix", type=str, default="",
                    help="Suffix appended to the output directory name for each "
                         "scheme (e.g. '_r1' to get lbfgs_ensemble_r1_{scheme}/).")
    args = ap.parse_args()
    N_CHAINS, N_ADAM_EPOCHS, LBFGS_MAX_ITER = \
        int(args.chains), int(args.adam_epochs), int(args.lbfgs_iters)
    DENSITY_VMAX = float(args.density_vmax)

    print("=" * 72)
    print("v6/04_optimize_lbfgs_ensemble.py — Adam warm-start + L-BFGS ensemble")
    print("=" * 72)
    print(f"device       : {DEVICE}")
    print(f"init schemes : {INIT_SCHEMES}")
    print(f"chains (K)   : {N_CHAINS}  (init σ={INIT_OVERDISP_SIGMA} m)")
    print(f"Adam epochs  : {N_ADAM_EPOCHS}  (primaries/step={PRIMARIES_PER_STEP})")
    print(f"L-BFGS       : max_iter={LBFGS_MAX_ITER}  batch={LBFGS_BATCH_PRIMARIES}")

    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    n_total_primaries = int(primary_all.shape[0])
    print(f"[load] {n_total_primaries} primaries")

    if args.opt_suffix:
        global OPT_DIR_TEMPLATE
        OPT_DIR_TEMPLATE = OPT_FOLDER + "_lbfgs_ensemble" + args.opt_suffix + "_{scheme}"

    if args.fnn_folder:
        import modules_v6.constants as _C
        _C.FNN_FOLDER = args.fnn_folder
        global FNN_FOLDER
        FNN_FOLDER = args.fnn_folder
        print(f"[fnn_folder] overriding FNN_FOLDER -> {args.fnn_folder}")

    if args.recon_folder:
        global RECON_DIR
        RECON_DIR = args.recon_folder
        print(f"[recon_folder] overriding recon dir -> {args.recon_folder}")

    fnn, recon = load_models(DEVICE, fnn_folder=FNN_FOLDER, recon_dir=RECON_DIR)

    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    # Optional warm-start center layout (N_DETECTORS, 2) = (North, East).
    init_center = None
    if args.init_from:
        raw = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            N_c = raw["x"].float().reshape(-1)
            E_c = raw["y"].float().reshape(-1)
            init_center = torch.stack([N_c, E_c], dim=-1)
        else:
            init_center = raw.float()
        print(f"[init_from] loaded layout from {args.init_from}  "
              f"shape={tuple(init_center.shape)}")

    results = []
    per_scheme = {}     # scheme -> (adam_bests, adam_logs, perturbed_inits)
    for scheme in INIT_SCHEMES:
        print()
        print("=" * 72)
        print(f"init scheme: {scheme}"
              f"{'  (warm-start from --init_from)' if init_center is not None else ''}")
        print("=" * 72)
        torch.manual_seed(SEED); np.random.seed(SEED)
        g = torch.Generator().manual_seed(SEED)
        per_scheme[scheme] = _perturbed_adam_runs(
            scheme, N_CHAINS, g, mountain, fnn, recon, primary_all, n_total_primaries,
            init_center=init_center,
        )
        results.append(_run_one_scheme(
            scheme, mountain, fnn, recon, primary_all, n_total_primaries,
            {scheme: per_scheme[scheme]},
        ))

    if RUN_COMBINED and len(per_scheme) > 1:
        print()
        print("=" * 72)
        print(f"init scheme: {COMBINED_SCHEME_NAME} (sources={list(per_scheme)})")
        print("=" * 72)
        results.append(_run_one_scheme(
            COMBINED_SCHEME_NAME, mountain, fnn, recon, primary_all, n_total_primaries,
            per_scheme,
        ))

    print()
    print("=" * 72)
    print("summary")
    print("=" * 72)
    for r in results:
        print(f"  {r['scheme']:<10}  best U={r['best_U']:+.3f}  "
              f"σ̄=({r['mean_std_x']:.1f}, {r['mean_std_y']:.1f}) m  ->  {r['opt_dir']}")

    # Save the globally best layout across all schemes if requested.
    if args.save_best_layout:
        best_r = max(results, key=lambda r: r["best_U"])
        best_layout = torch.stack([best_r["best_x"], best_r["best_y"]], dim=-1)
        torch.save(best_layout, args.save_best_layout)
        print(f"\n[save_best_layout] best U={best_r['best_U']:+.3f} "
              f"(scheme={best_r['scheme']})  ->  {args.save_best_layout}")


if __name__ == "__main__":
    main()