"""Guided Evolution Strategy (GES) optimizer for detector layout.

Algorithm: GES (Maheswaranathan et al., ICML 2019) with antithetic sampling.

At each generation, the exact gradient g = dU/dmu is computed via PyTorch
autograd on a fixed primary batch. The ES perturbation distribution is biased
toward g, blending an isotropic Gaussian (for coverage) with a gradient-aligned
component (for direction):

    delta_k = sigma * (alpha * z_k  +  sqrt(1 - alpha^2) * e_k * g_hat)

where z_k ~ N(0, I), e_k ~ N(0, 1), g_hat = g / ||g||.  alpha=1 recovers
vanilla ES; alpha->0 collapses to pure gradient descent.  Antithetic pairs
(mu + delta_k, mu - delta_k) are evaluated without gradients, and an
OpenAI-ES–style fitness-weighted update moves mu.

After N_GEN generations, each chain's best-ever mu is refined with L-BFGS on
the same fixed primary batch (same logic as `04_optimize_lbfgs_ensemble.py`).
N_CHAINS independent GES chains are run per init scheme.

Output artifacts (per scheme + "combined") in
``<OPT_FOLDER>_ges_{scheme}/``:
    layout_best.pt          highest-U GES→L-BFGS layout
    layout_mean.pt          per-group mean + std (aligned ensemble)
    layouts_all.pt          aligned (K, n_det, 2) + per-chain U
    optimize_log.json       per-chain per-gen logs + config

Run:
    cd TambOpt/detector_optimization_v6
    python 04_optimize_ges.py [--chains N] [--gens N] [--opt_suffix STR]
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
from modules_v6.opt_core import (
    utility_of_xy, align_to_reference, load_models,
    W_THETA, W_PHI, W_E, W_PR, W_DIV,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD,
)
_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(_HERE, "plots", "opt_plotting.py"))
_plt = importlib.util.module_from_spec(_plt_spec); _plt_spec.loader.exec_module(_plt)


# ── Config ──────────────────────────────────────────────────────────────────────
INIT_SCHEMES     = ("grid", "center")
RUN_COMBINED     = True
OPT_DIR_TEMPLATE = OPT_FOLDER + "_ges_{scheme}"
RECON_DIR        = RECON_FOLDER + "_deepsets"

# Independent GES chains per init scheme.
N_CHAINS             = 5
INIT_OVERDISP_SIGMA  = 1000.0   # m — per-chain init spread around scheme init

# GES hyperparameters
K_PAIRS     = 50      # antithetic pairs per generation (2*K_PAIRS no-grad evals + 1 grad)
GES_ALPHA   = 0.6     # isotropic fraction: 1 = pure ES, 0 = pure gradient
SIGMA_INIT  = 500.0   # m — initial perturbation scale (geometric annealing)
SIGMA_FINAL = 100.0   # m — final perturbation scale
# GES_LR is a dimensionless step fraction: mu moves GES_LR * g_ES where g_ES has
# magnitude O(sigma). Setting GES_LR~0.05 gives ~0.05*500=25m/gen initially,
# naturally scaling to 0.05*100=5m/gen at convergence.
GES_LR      = 0.05    # dimensionless step fraction (step ≈ GES_LR * sigma per gen)
N_GEN       = 400     # GES generations per chain

# Primary batch (fixed per scheme run, shared across chains for comparable U)
GES_BATCH = 512

# L-BFGS refinement from each chain's best-ever mu (polishing step)
LBFGS_MAX_ITER  = 1_000
LBFGS_LR        = 1.0
LBFGS_HISTORY   = 20

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ──────────────────────────────────────────────────────────────────────
def _anneal_sigma(gen: int, n_gen: int) -> float:
    if n_gen <= 1:
        return SIGMA_INIT
    t = gen / max(n_gen - 1, 1)
    return SIGMA_INIT * ((SIGMA_FINAL / SIGMA_INIT) ** t)


class _NonFiniteLoss(Exception):
    pass


def _lbfgs_refine(mu: torch.Tensor, fnn, recon,
                  primary_fixed: torch.Tensor, mountain):
    """L-BFGS from mu (flat 200D CPU tensor). Returns (mu_projected, U_float)."""
    xy = mu.clone().to(DEVICE).detach()
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
            raise _NonFiniteLoss
        loss.backward()
        return loss
    try:
        opt.step(closure)
    except _NonFiniteLoss:
        print("  [lbfgs] non-finite loss — aborting step")
    except Exception as exc:
        print(f"  [lbfgs] exception: {exc!r}")
    with torch.no_grad():
        if not torch.isfinite(xy).all():
            print("  [lbfgs] non-finite iterate — falling back to init")
            xy.data.copy_(mu.to(DEVICE))
        x_p, y_p = project_to_mountain_ne(
            mountain, xy[:N_DETECTORS].cpu(), xy[N_DETECTORS:].cpu())
        U_p, _, _ = utility_of_xy(
            x_p.to(DEVICE), y_p.to(DEVICE), primary_fixed, fnn, recon)
    return torch.cat([x_p, y_p]).float(), float(U_p.item())


# ── GES chain ────────────────────────────────────────────────────────────────────
def ges_run_one_chain(
    init_x: torch.Tensor,
    init_y: torch.Tensor,
    fnn,
    recon,
    primary_fixed: torch.Tensor,
    mountain,
    chain_idx: int,
) -> tuple:
    """One independent GES chain. Returns (best_mu, best_U, gen_log).

    best_mu is the result of L-BFGS polishing from the best-ever GES mu.
    gen_log is a list of per-generation dicts."""
    D = 2 * N_DETECTORS

    # Initialise mean at mountain-projected init.
    N0, E0 = project_to_mountain_ne(mountain, init_x.cpu(), init_y.cpu())
    mu = torch.cat([N0, E0]).float()   # (D,) CPU

    best_ges_mu = mu.clone()
    best_ges_U  = -float("inf")
    gen_log     = []

    print(f"[ges chain {chain_idx}]  "
          f"N=[{N0.min():.0f},{N0.max():.0f}]  E=[{E0.min():.0f},{E0.max():.0f}]")

    for gen in range(N_GEN):
        sigma = _anneal_sigma(gen, N_GEN)
        t0    = time.time()

        # ── 1. Exact gradient at mu (one backward pass) ──────────────────────
        mu_dev = mu.clone().to(DEVICE).detach().requires_grad_(True)
        U_mu, _, _ = utility_of_xy(
            mu_dev[:N_DETECTORS], mu_dev[N_DETECTORS:], primary_fixed, fnn, recon)
        U_mu.backward()
        g       = mu_dev.grad.detach().cpu()      # (D,)
        g_hat   = g / (g.norm() + 1e-8)           # unit gradient direction
        U_at_mu = float(U_mu.item())

        # ── 2. Sample K antithetic pairs (GES perturbation distribution) ─────
        beta_coeff = math.sqrt(max(0.0, 1.0 - GES_ALPHA ** 2))
        deltas = []
        for _ in range(K_PAIRS):
            z   = torch.randn(D)
            e   = torch.randn(1).item()
            delta = sigma * (GES_ALPHA * z + beta_coeff * e * g_hat)
            deltas.append(delta)

        # ── 3. Evaluate U at projected(mu+delta) and projected(mu-delta) ────────
        # Project each candidate onto the mountain before scoring — the FNN was
        # trained on mountain-projected positions; off-mountain inputs are OOD.
        # Store projected positions for best-point tracking and effective deltas
        # for the gradient estimate (so the update reflects actual surface moves).
        U_plus, U_minus = [], []
        eff_deltas      = []   # effective perturbation after projection
        proj_pts        = []   # [(mu_p, mu_m)] projected flat tensors, for tracking
        with torch.no_grad():
            for delta in deltas:
                xp, yp = project_to_mountain_ne(
                    mountain,
                    mu[:N_DETECTORS] + delta[:N_DETECTORS],
                    mu[N_DETECTORS:] + delta[N_DETECTORS:])
                xm, ym = project_to_mountain_ne(
                    mountain,
                    mu[:N_DETECTORS] - delta[:N_DETECTORS],
                    mu[N_DETECTORS:] - delta[N_DETECTORS:])
                Up, _, _ = utility_of_xy(xp.to(DEVICE), yp.to(DEVICE),
                                         primary_fixed, fnn, recon)
                Um, _, _ = utility_of_xy(xm.to(DEVICE), ym.to(DEVICE),
                                         primary_fixed, fnn, recon)
                U_plus.append(float(Up.item()))
                U_minus.append(float(Um.item()))
                eff_deltas.append(torch.cat([xp - xm, yp - ym]) / 2.0)
                proj_pts.append((torch.cat([xp, yp]), torch.cat([xm, ym])))

        # ── 4. OpenAI-ES fitness-weighted update ─────────────────────────────
        # g_ES has units of meters (same as mu). After return normalization,
        # mean(norm_returns * eff_deltas) ~ O(sigma) meters, so GES_LR is a
        # dimensionless step fraction: step ≈ GES_LR * sigma per generation.
        returns = torch.tensor([p - m for p, m in zip(U_plus, U_minus)])
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()
        delta_stack = torch.stack(eff_deltas)                    # (K, D)
        g_ES = (returns.unsqueeze(-1) * delta_stack).mean(dim=0) # units: meters
        mu = mu + GES_LR * g_ES

        # ── 5. Project back to mountain ───────────────────────────────────────
        N_new, E_new = project_to_mountain_ne(
            mountain, mu[:N_DETECTORS], mu[N_DETECTORS:])
        mu = torch.cat([N_new, E_new]).float()

        # ── Track best (actual sample location, not just current mu) ──────────
        # The best evaluation may be at a sample point mu±delta, not at mu.
        # Store the actual projected location so L-BFGS starts from the right place.
        gen_best   = U_at_mu
        gen_best_p = mu.clone()
        for up, um, (pp, pm) in zip(U_plus, U_minus, proj_pts):
            if up > gen_best:
                gen_best   = up
                gen_best_p = pp.float()
            if um > gen_best:
                gen_best   = um
                gen_best_p = pm.float()
        if gen_best > best_ges_U:
            best_ges_U  = gen_best
            best_ges_mu = gen_best_p.clone()

        elapsed = time.time() - t0
        gen_log.append(dict(
            gen=gen, U_mu=U_at_mu, U_best_gen=gen_best,
            g_norm=float(g.norm().item()), sigma=sigma, elapsed_s=elapsed,
        ))
        if gen == 0 or (gen + 1) % 50 == 0 or gen == N_GEN - 1:
            print(f"  [chain {chain_idx} gen {gen+1:3d}/{N_GEN}]  "
                  f"U_mu={U_at_mu:+.3f}  U_best={gen_best:+.3f}  "
                  f"||g||={g.norm():.1f}  σ={sigma:.0f}m  ({elapsed:.1f}s)")

    # ── 6. L-BFGS polishing from best GES mu ─────────────────────────────────
    print(f"[chain {chain_idx}] L-BFGS polish from best GES mu "
          f"(U_ges={best_ges_U:+.3f}) …")
    best_mu, best_U = _lbfgs_refine(best_ges_mu, fnn, recon, primary_fixed, mountain)
    print(f"[chain {chain_idx}] L-BFGS U={best_U:+.3f}  "
          f"(gain={best_U - best_ges_U:+.3f})")

    return best_mu, best_U, gen_log


# ── Scheme runner ────────────────────────────────────────────────────────────────
def _run_one_scheme(scheme: str,
                    mountain,
                    fnn,
                    recon,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    per_source: dict):
    """Run GES chains for all sources, align ensemble, save artifacts."""
    opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
    os.makedirs(opt_dir, exist_ok=True)
    print("-" * 72)
    print(f"[ges] scheme={scheme}  -> {opt_dir}")

    # One fixed primary batch shared across all chains in this scheme.
    g = torch.Generator().manual_seed(SEED)
    idx_fixed = torch.randint(0, n_total_primaries, (GES_BATCH,), generator=g)
    primary_fixed = primary_all[idx_fixed].to(DEVICE)

    all_results, source_per_run = [], []
    chain_global = 0
    for src, inits in per_source.items():
        for (init_x, init_y) in inits:
            print(f"\n{'='*72}")
            print(f"[ges] chain {chain_global+1}  (src={src})")
            print(f"{'='*72}")
            best_mu, best_U, gen_log = ges_run_one_chain(
                init_x=init_x.float(), init_y=init_y.float(),
                fnn=fnn, recon=recon,
                primary_fixed=primary_fixed,
                mountain=mountain,
                chain_idx=chain_global,
            )
            all_results.append(dict(mu=best_mu, U=best_U, gen_log=gen_log))
            source_per_run.append(src)
            chain_global += 1

    # Ensemble alignment (same logic as 04_optimize_lbfgs_ensemble.py).
    refined_U  = [r["U"] for r in all_results]
    refined_xy = [(r["mu"][:N_DETECTORS], r["mu"][N_DETECTORS:]) for r in all_results]
    layouts_xy = np.stack(
        [np.stack([x.numpy(), y.numpy()], axis=-1) for x, y in refined_xy], axis=0)
    ref_idx         = int(np.argmax(refined_U))
    aligned, perms  = align_to_reference(layouts_xy, ref_idx)
    mean_xy         = aligned.mean(axis=0)
    std_xy          = aligned.std(axis=0)

    best_x, best_y = refined_xy[ref_idx]
    print(f"[ges] ensemble K={len(refined_U)}  best U={refined_U[ref_idx]:+.3f}  "
          f"σ̄=({std_xy[:,0].mean():.1f}, {std_xy[:,1].mean():.1f}) m")

    # ── Persist ──────────────────────────────────────────────────────────────
    torch.save({"x": best_x, "y": best_y, "U": refined_U[ref_idx],
                "chain": ref_idx, "source": source_per_run[ref_idx]},
               os.path.join(opt_dir, "layout_best.pt"))
    torch.save({"mean_x": torch.as_tensor(mean_xy[:, 0]),
                "mean_y": torch.as_tensor(mean_xy[:, 1]),
                "std_x":  torch.as_tensor(std_xy[:, 0]),
                "std_y":  torch.as_tensor(std_xy[:, 1])},
               os.path.join(opt_dir, "layout_mean.pt"))
    torch.save({"aligned":       torch.as_tensor(aligned),
                "perms":         torch.as_tensor(perms),
                "utilities":     torch.as_tensor(refined_U),
                "source_per_run": source_per_run,
                "ref_idx":       ref_idx},
               os.path.join(opt_dir, "layouts_all.pt"))
    with open(os.path.join(opt_dir, "optimize_log.json"), "w") as f:
        json.dump({
            "scheme": scheme,
            "sources": list(per_source),
            "source_per_run": source_per_run,
            "ref_idx": ref_idx,
            "refined_U": refined_U,
            "best_U": refined_U[ref_idx],
            "ensemble_stats": {
                "mean_std_x": float(std_xy[:, 0].mean()),
                "mean_std_y": float(std_xy[:, 1].mean()),
                "max_std_x":  float(std_xy[:, 0].max()),
                "max_std_y":  float(std_xy[:, 1].max()),
            },
            "chain_logs": [r["gen_log"] for r in all_results],
            "config": {
                "n_chains": N_CHAINS, "k_pairs": K_PAIRS,
                "ges_alpha": GES_ALPHA, "sigma_init": SIGMA_INIT,
                "sigma_final": SIGMA_FINAL, "ges_lr": GES_LR,
                "n_gen": N_GEN, "ges_batch": GES_BATCH,
                "lbfgs_max_iter": LBFGS_MAX_ITER,
                "w_theta": W_THETA, "w_phi": W_PHI, "w_e": W_E,
                "w_pr": W_PR, "w_div": W_DIV,
                "layout_threshold": LAYOUT_THRESHOLD,
                "reconstruct_threshold": RECONSTRUCT_THRESHOLD,
                "seed": SEED,
            },
        }, f, indent=2)

    try:
        surface = SurfaceUpMap.from_mountain(mountain).to("cpu")
        _plt.plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                           mountain, os.path.join(opt_dir, "layout_ensemble.png"),
                           surface=surface, title_kind="GES ensemble")
        _plt.plot_density_heatmap(aligned, best_x, best_y,
                                  mountain, os.path.join(opt_dir, "layout_density.png"),
                                  surface=surface)
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")

    print(f"[done] scheme={scheme}  best U={refined_U[ref_idx]:+.3f}  "
          f"σ̄=({std_xy[:,0].mean():.1f},{std_xy[:,1].mean():.1f}) m  -> {opt_dir}")
    return dict(scheme=scheme, best_U=refined_U[ref_idx],
                best_x=best_x, best_y=best_y,
                mean_std_x=float(std_xy[:, 0].mean()),
                mean_std_y=float(std_xy[:, 1].mean()),
                opt_dir=opt_dir)


def _build_chain_inits(init_x, init_y, K, generator):
    """K overdispersed starts around (init_x, init_y). Returns (K, D) CPU tensor."""
    base    = torch.cat([init_x, init_y]).float()
    perturb = torch.randn(K, base.numel(), generator=generator) * INIT_OVERDISP_SIGMA
    return base.unsqueeze(0) + perturb   # (K, D)


def main():
    global N_CHAINS, N_GEN, RECON_DIR, OPT_DIR_TEMPLATE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains",       type=int, default=N_CHAINS)
    ap.add_argument("--gens",         type=int, default=N_GEN)
    ap.add_argument("--recon_folder", type=str, default=None)
    ap.add_argument("--fnn_folder",   type=str, default=None)
    ap.add_argument("--opt_suffix",   type=str, default="")
    ap.add_argument("--init_from",    type=str, default=None,
                    help="Path to a layout_best.pt to warm-start all chains.")
    args = ap.parse_args()
    N_CHAINS = int(args.chains)
    N_GEN    = int(args.gens)
    if args.recon_folder:
        RECON_DIR = args.recon_folder
    if args.opt_suffix:
        OPT_DIR_TEMPLATE = OPT_FOLDER + "_ges" + args.opt_suffix + "_{scheme}"

    print("=" * 72)
    print("v6/04_optimize_ges.py — Guided Evolution Strategy (GES)")
    print("=" * 72)
    print(f"device      : {DEVICE}")
    print(f"init schemes: {INIT_SCHEMES}")
    print(f"chains      : {N_CHAINS}  (init σ={INIT_OVERDISP_SIGMA:.0f}m)")
    print(f"K_PAIRS     : {K_PAIRS}  (2×K={2*K_PAIRS} evals/gen + 1 grad)")
    print(f"GES_ALPHA   : {GES_ALPHA}  (gradient-aligned β={math.sqrt(max(0,1-GES_ALPHA**2)):.2f})")
    print(f"sigma       : {SIGMA_INIT:.0f}m → {SIGMA_FINAL:.0f}m (geometric)")
    print(f"GES_LR      : {GES_LR}  N_GEN={N_GEN}")
    print(f"L-BFGS      : {LBFGS_MAX_ITER} iter (polish after GES)")

    primary_all = torch.load(
        os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    n_total = int(primary_all.shape[0])
    print(f"[load] {n_total} primaries")

    fnn_folder = args.fnn_folder or FNN_FOLDER
    fnn, recon = load_models(DEVICE, fnn_folder=fnn_folder, recon_dir=RECON_DIR)
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    # Optional warm-start center layout.
    init_center = None
    if args.init_from:
        raw = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            init_center = torch.stack(
                [raw["x"].float().reshape(-1), raw["y"].float().reshape(-1)], dim=-1)
        else:
            init_center = raw.float()
        print(f"[init_from] {args.init_from}  shape={tuple(init_center.shape)}")

    results    = []
    per_scheme = {}
    for scheme in INIT_SCHEMES:
        print()
        print("=" * 72)
        print(f"init scheme: {scheme}")
        print("=" * 72)
        torch.manual_seed(SEED); np.random.seed(SEED)
        g = torch.Generator().manual_seed(SEED)

        if init_center is not None:
            N_t = init_center[:, 0].clone()
            E_t = init_center[:, 1].clone()
        else:
            N_np, E_np = sample_initial_layout_ne(
                mountain, n_units=N_DETECTORS, scheme=scheme)
            N_t = torch.as_tensor(N_np, dtype=torch.float32)
            E_t = torch.as_tensor(E_np, dtype=torch.float32)

        chains = _build_chain_inits(N_t, E_t, N_CHAINS, g)   # (K, 200)
        inits  = [(chains[k, :N_DETECTORS].clone(),
                   chains[k, N_DETECTORS:].clone())
                  for k in range(N_CHAINS)]
        per_scheme[scheme] = inits

        results.append(_run_one_scheme(
            scheme, mountain, fnn, recon, primary_all, n_total,
            {scheme: inits},
        ))

    if RUN_COMBINED and len(per_scheme) > 1:
        print()
        print("=" * 72)
        print("init scheme: combined")
        print("=" * 72)
        results.append(_run_one_scheme(
            "combined", mountain, fnn, recon, primary_all, n_total, per_scheme,
        ))

    print()
    print("=" * 72)
    print("summary")
    print("=" * 72)
    for r in results:
        print(f"  {r['scheme']:<10}  best U={r['best_U']:+.3f}  "
              f"σ̄=({r['mean_std_x']:.1f},{r['mean_std_y']:.1f}) m  ->  {r['opt_dir']}")


if __name__ == "__main__":
    main()
