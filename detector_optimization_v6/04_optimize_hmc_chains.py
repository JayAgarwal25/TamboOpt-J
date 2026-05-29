"""Optimize detector positions: pre-Adam perturbation, then multi-chain HMC/NUTS.

Multi-sequence variant of ``04_optimize_nuts.py`` that follows the
Gelman & Rubin (1992) prescription: run K independent NUTS chains from
overdispersed starting points, then aggregate samples + report
convergence diagnostics (R-hat, ESS) computed by Pyro and ArviZ.

Per scheme:

1.  Sample the scheme's initial layout (`mountain.sample_initial_layout`)
    and create K=`NUTS_NUM_CHAINS` Gaussian perturbations of it
    (std `NUTS_INIT_OVERDISP_SIGMA`, projected back to the mountain).
2.  Run Adam (`N_ADAM_EPOCHS`) independently from each perturbed start
    → K Adam-best layouts.
3.  Stack the K Adam-bests as the K NUTS chain inits directly (no extra
    post-Adam perturbation). Pyro's ``MCMC(num_chains=K, ...)`` handles
    chain execution and diagnostic aggregation.

The "combined" run pools K Adam-bests from each scheme into one MCMC
with K_total = K * len(INIT_SCHEMES) chains, so R-hat measures
cross-scheme convergence.

Artifacts (per INIT_SCHEMES entry) land in
``<OPT_FOLDER>_hmc_chains_{scheme}/``:

    layout_best.pt          best NUTS sample (mountain-projected, across all chains)
    layout_adam.pt          Adam-phase best layout (mountain-projected)
    layout_init.pt          initial layout from the chosen scheme
    nuts_samples.pt         pooled NUTS samples + per-sample utilities + chain id
    nuts_diagnostics.csv    arviz.summary() per-coordinate (r_hat, ess_bulk, ess_tail, ...)
    optimize_log.json       Adam log + multi-chain NUTS summary + config + r_hat stats
    optimize_curves.png     Adam U trajectory + per-chain NUTS sample-U histogram
    layout_before_after.png mountain top-down with init / Adam / NUTS layouts
    nuts_diagnostics.png    ArviZ rank + trace plots for representative coords

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 04_optimize_hmc_chains.py
"""
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
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS, MCMC

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.fnn_surrogate import FNNSurrogate
from modules_v6.reconstruction import Reconstruction
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER, OPT_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules.utility_functions   import reconstructability, U_E, U_angle, U_PR
from modules.layout_optimization import LearnableXY
from modules_v4.tr_geometry      import load_tr_mountain


# ── Config ───────────────────────────────────────────────────────────────────
# Per-scheme runs use NUTS_NUM_CHAINS pre-Adam perturbations of the scheme init
# (each gets its own Adam run; the K Adam-bests stack directly as NUTS chain
# inits — no post-Adam perturbation anywhere). The "combined" run uses
# NUTS_NUM_CHAINS pre-Adam perturbations PER scheme stacked together, so
# K_total = NUTS_NUM_CHAINS * len(INIT_SCHEMES) Adam runs feed one NUTS call.
INIT_SCHEMES         = ("grid", "center")
RUN_COMBINED         = True
COMBINED_SCHEME_NAME = "combined"
OPT_DIR_TEMPLATE     = OPT_FOLDER + "_hmc_chains_{scheme}"

# Adam warm-start
N_ADAM_EPOCHS       = 5_000
PRIMARIES_PER_STEP  = 256
ADAM_LR             = 1.0
GRAD_CLIP           = 100.0
ADAM_LOG_EVERY      = 10

# NUTS sampling (multi-chain). Per-chain N can be lower than the single-chain
# variant because pooled effective sample size grows with K.
NUTS_NUM_CHAINS          = 4
NUTS_NUM_SAMPLES         = 1_500      # PER chain (post-warmup)
NUTS_WARMUP              = 1_500      # PER chain
NUTS_TEMPERATURE         = 3.0
NUTS_PRIOR_SIGMA         = 100.0     # metres — the Normal prior anchored at Adam optimum
NUTS_INIT_OVERDISP_SIGMA = 10.0 * NUTS_PRIOR_SIGMA  # per-chain init spread (G&R: > prior)
NUTS_BATCH_PRIMARIES     = 512
NUTS_TARGET_ACCEPT_PROB  = 0.8
NUTS_MAX_TREE_DEPTH      = 7
# Single GPU → keep chains sequential. Multi-process CUDA forking is fragile;
# Pyro still aggregates them into one MCMC object with full diagnostics.
NUTS_MP_CONTEXT          = None      # set to "spawn" only if you have >1 GPU

# Convergence gate (Gelman & Rubin rule of thumb)
R_HAT_WARN_THRESHOLD     = 1.1

# Utility composite weights — match 04_optimize.py
W_THETA = 1e2
W_PHI   = 1e2
W_E     = 1e3
W_PR    = 5e5
W_DIV   = 1e3

# Reconstructability thresholds — match 04_optimize.py
LAYOUT_THRESHOLD      = 5e-2
RECONSTRUCT_THRESHOLD = 10.0

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def primary_to_physical_labels(primary: torch.Tensor):
    """(B, 5) -> (E_GeV, θ_rad, φ_rad). Matches 04_optimize.py."""
    dir_x = primary[:, 0]
    dir_y = primary[:, 1]
    dir_z = primary[:, 2].clamp(-1.0, 1.0)
    log_e_norm = primary[:, 3]
    log_e = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    E_gev = torch.exp(log_e) - 1.0
    theta = torch.arccos(dir_z)
    phi   = torch.atan2(dir_y, dir_x)
    two_pi = 2.0 * math.pi
    phi = torch.where(phi < 0, phi + two_pi, phi)
    return E_gev, theta, phi


def utility_of_xy(x_det: torch.Tensor,
                  y_det: torch.Tensor,
                  primary_batch: torch.Tensor,
                  fnn: FNNSurrogate,
                  recon: Reconstruction):
    """Differentiable composite U for a layout against a primary batch.

    Mirrors the inner loop of `_run_optimization` in 04_optimize.py so the
    two scripts optimize the SAME objective (the U_PR term is computed but
    deliberately omitted from the composite, matching production)."""
    B = primary_batch.shape[0]
    xy_per_det = torch.stack([x_det, y_det], dim=-1)                       # (n_det, 2)
    xy_batch   = xy_per_det.unsqueeze(0).expand(B, -1, -1)                 # (B, n_det, 2)

    pred_ET    = fnn(primary_batch, xy_batch)                              # (B, n_det, 2)
    E_pred_det = pred_ET[..., 0]
    T_pred_det = pred_ET[..., 1]

    recon_feats = torch.stack(
        [xy_batch[..., 0], xy_batch[..., 1], E_pred_det, T_pred_det],
        dim=-1,
    )                                                                      # (B, n_det, 4)
    recon_input = recon_feats.reshape(B, -1)
    pred = recon(recon_input)                                              # (B, 4)
    E_pred_phys, theta_pred, phi_pred = primary_to_physical_labels(pred)
    E_pred_phys = E_pred_phys.clamp(min=1.0)

    E_true, theta_true, phi_true = primary_to_physical_labels(primary_batch)

    r = reconstructability(
        torch.expm1(E_pred_det),
        layout_threshold=LAYOUT_THRESHOLD,
        reconstruct_threshold=RECONSTRUCT_THRESHOLD,
    )
    u_theta = U_angle(theta_pred, theta_true, r)
    u_phi   = U_angle(phi_pred,   phi_true,   r)
    u_e     = U_E    (E_pred_phys, E_true,    r)
    u_pr    = U_PR(r)
    U = (W_THETA * u_theta + W_PHI * u_phi + W_E * u_e) / W_DIV
    return U, r, dict(u_theta=u_theta, u_phi=u_phi, u_e=u_e, u_pr=u_pr)


def adam_warm_start(scheme: str,
                    mountain,
                    fnn: FNNSurrogate,
                    recon: Reconstruction,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    init_override):
    """N_ADAM_EPOCHS of Adam with mountain projection. Returns:
       (best_x, best_y, init_x, init_y, log).

    `init_override=(x, y)` is provided, those tensors are used as the Adam
    starting layout. Caller is responsible for projecting them to the mountain 
    first. `scheme` is then just a label used in log lines.
    """
    N_init, U_init = init_override
    N_init = N_init.float()
    U_init = U_init.float()
    print(f"[adam] init {scheme}  N in [{N_init.min():.1f}, {N_init.max():.1f}]  "
          f"Up in [{U_init.min():.1f}, {U_init.max():.1f}]")

    xy_module = LearnableXY(N_init, U_init, device=str(DEVICE)).to(DEVICE)
    optimizer = torch.optim.Adam(xy_module.parameters(), lr=ADAM_LR)

    log = []
    best_u = -float("inf")
    best_x = N_init.clone()
    best_y = U_init.clone()

    for epoch in range(N_ADAM_EPOCHS):
        idx = torch.randint(0, n_total_primaries, (PRIMARIES_PER_STEP,))
        primary_batch = primary_all[idx].to(DEVICE)

        x_det, y_det = xy_module()
        U, r, parts = utility_of_xy(x_det, y_det, primary_batch, fnn, recon)
        loss = -U

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        # Project to mountain surface
        with torch.no_grad():
            N_cpu  = xy_module.x.detach().cpu()
            Up_cpu = xy_module.y.detach().cpu()
            N_new, Up_new = mountain.project_to_mountain(N_cpu, Up_cpu)
            xy_module.x.data.copy_(N_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(Up_new.to(DEVICE).to(xy_module.y.dtype))

        u_val  = float(U.item())
        r_mean = float(r.mean().item())
        gn     = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

        if u_val > best_u:
            best_u = u_val
            best_x = xy_module.x.detach().cpu().clone()
            best_y = xy_module.y.detach().cpu().clone()

        log.append(dict(
            epoch=epoch + 1, U=u_val, r_mean=r_mean, grad_norm=gn,
            u_theta=float(parts["u_theta"].item()),
            u_phi=float(parts["u_phi"].item()),
            u_e=float(parts["u_e"].item()),
            u_pr=float(parts["u_pr"].item()),
        ))

        if epoch == 0 or (epoch + 1) % ADAM_LOG_EVERY == 0 or epoch == N_ADAM_EPOCHS - 1:
            print(f"  [adam {epoch+1:3d}/{N_ADAM_EPOCHS}] "
                  f"U={u_val:+.3f}  r={r_mean:.3f}  |g|={gn:.2f}")

    print(f"[adam] best U={best_u:+.3f}")
    return best_x, best_y, N_init, U_init, log


def _build_chain_inits(init_x: torch.Tensor, init_y: torch.Tensor,
                       K: int, generator: torch.Generator) -> torch.Tensor:
    """K overdispersed starts around (init_x, init_y). Returns (K, 2*n_det) on DEVICE."""
    base = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach()  # (D,)
    perturb = torch.randn(
        K, base.numel(), generator=generator, device="cpu",
    ).to(DEVICE) * NUTS_INIT_OVERDISP_SIGMA
    return base.unsqueeze(0) + perturb                                        # (K, D)


def _perturbed_adam_runs(scheme: str, K: int, generator: torch.Generator,
                         mountain, fnn, recon, primary_all, n_total_primaries):
    """K pre-Adam perturbations of the scheme init → K Adam runs.

    Returns (adam_bests, adam_logs, perturbed_inits) where each entry is
    length K. The K Adam-bests are what become NUTS chain inits.
    """
    N_np, U_np = mountain.sample_initial_layout(n_units=N_DETECTORS, scheme=scheme)
    N_t = torch.as_tensor(N_np, dtype=torch.float32)
    U_t = torch.as_tensor(U_np, dtype=torch.float32)
    N_t, U_t = mountain.project_to_mountain(N_t, U_t)
    chains_init = _build_chain_inits(N_t, U_t, K, generator)                  # (K, D)

    adam_bests, adam_logs, perturbed_inits = [], [], []
    for k in range(K):
        xk = chains_init[k, :N_DETECTORS].cpu()
        yk = chains_init[k, N_DETECTORS:].cpu()
        xk, yk = mountain.project_to_mountain(xk, yk)
        perturbed_inits.append((xk.float().clone(), yk.float().clone()))
        print(f"\n[perturb→adam] scheme={scheme}  chain {k+1}/{K}")
        bx, by, _, _, log = adam_warm_start(
            scheme=scheme, mountain=mountain, fnn=fnn, recon=recon,
            primary_all=primary_all, n_total_primaries=n_total_primaries,
            init_override=(xk, yk),
        )
        adam_bests.append((bx, by))
        adam_logs.append(log)
    return adam_bests, adam_logs, perturbed_inits


def nuts_sampling_multichain(init_x: torch.Tensor,
                             init_y: torch.Tensor,
                             fnn: FNNSurrogate,
                             recon: Reconstruction,
                             primary_all: torch.Tensor,
                             n_total_primaries: int,
                             init_chains: torch.Tensor):
    """NUTS over pre-built chain inits with a Normal prior anchored at (init_x, init_y).

    `init_chains` is a (K_total, 2*n_det) tensor of starting layouts — one row
    per chain. (init_x, init_y) is used only as the prior anchor
    (`Normal(loc, NUTS_PRIOR_SIGMA)`).

    Returns a dict with pooled samples (CPU), per-sample utility + chain id,
    best-by-utility layout, and Pyro/ArviZ diagnostics (r_hat, ESS, summary df).
    """
    # Fixed primary batch shared across warmup + sampling and across chains so
    # every chain targets the *same* deterministic posterior (otherwise R-hat
    # would inflate due to different objectives per chain).
    g = torch.Generator().manual_seed(SEED)
    idx_fixed = torch.randint(0, n_total_primaries, (NUTS_BATCH_PRIMARIES,), generator=g)
    primary_fixed = primary_all[idx_fixed].to(DEVICE)

    init_xy_chains = init_chains.to(DEVICE)
    K_total = init_xy_chains.shape[0]
    D = init_xy_chains.shape[1]

    prior_loc   = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach()
    prior_scale = torch.full_like(prior_loc, NUTS_PRIOR_SIGMA)

    def potential_fn(params):
        xy_flat = params["xy"]
        x_det = xy_flat[:N_DETECTORS]
        y_det = xy_flat[N_DETECTORS:]
        U_val, _, _ = utility_of_xy(x_det, y_det, primary_fixed, fnn, recon)
        log_prior = dist.Normal(prior_loc, prior_scale).log_prob(xy_flat).sum()
        log_density = U_val / NUTS_TEMPERATURE + log_prior
        return -log_density

    # Run the K chains sequentially in this process. Pyro's multi-chain mode
    # (num_chains > 1) spawns worker processes and tries to pickle the
    # `potential_fn` closure → fails (CUDA + nn.Module + closure all
    # unpicklable). On a single GPU sequential is the same wall time anyway.
    print(f"[nuts] running  chains={K_total} (sequential)  warmup={NUTS_WARMUP}/chain  "
          f"samples={NUTS_NUM_SAMPLES}/chain  target_accept={NUTS_TARGET_ACCEPT_PROB}  "
          f"max_tree_depth={NUTS_MAX_TREE_DEPTH}  σ_prior={NUTS_PRIOR_SIGMA:.0f}m")
    t0 = time.time()
    per_chain_samples = []
    for k in range(K_total):
        pyro.clear_param_store()
        pyro.set_rng_seed(SEED + k)               # different seed per chain
        kernel = NUTS(
            potential_fn=potential_fn,
            adapt_step_size=True,
            target_accept_prob=NUTS_TARGET_ACCEPT_PROB,
            max_tree_depth=NUTS_MAX_TREE_DEPTH,
        )
        mcmc_k = MCMC(
            kernel,
            num_samples=NUTS_NUM_SAMPLES,
            warmup_steps=NUTS_WARMUP,
            initial_params={"xy": init_xy_chains[k]},
            num_chains=1,
            disable_progbar=False,
        )
        print(f"[nuts] chain {k+1}/{K_total}")
        mcmc_k.run()
        per_chain_samples.append(mcmc_k.get_samples()["xy"].detach().cpu())
    dt = time.time() - t0

    samples_by_chain = torch.stack(per_chain_samples, dim=0)                  # (K, N, D)
    K, N, _ = samples_by_chain.shape
    samples_pooled   = samples_by_chain.reshape(K * N, D)                     # (K*N, D)
    chain_ids        = torch.arange(K).unsqueeze(1).expand(K, N).reshape(-1)  # (K*N,)
    print(f"[nuts] sampled {K}x{N}={K*N} layouts in {dt:.1f}s")

    # Re-score every pooled sample on the same fixed batch. Samples live on
    # CPU (we cpu'd them per chain), so move each slice to DEVICE before the
    # FNN forward — primary_fixed and the models are on CUDA.
    utilities = torch.empty(samples_pooled.shape[0], dtype=torch.float32)
    with torch.no_grad():
        for i, s in enumerate(samples_pooled):
            s_dev = s.to(DEVICE)
            x_det = s_dev[:N_DETECTORS]
            y_det = s_dev[N_DETECTORS:]
            U_val, _, _ = utility_of_xy(x_det, y_det, primary_fixed, fnn, recon)
            utilities[i] = float(U_val.item())

    best_idx    = int(utilities.argmax())
    best_sample = samples_pooled[best_idx].cpu()
    best_x = best_sample[:N_DETECTORS]
    best_y = best_sample[N_DETECTORS:]

    # ── Diagnostics via ArviZ on the manually-stacked (K, N, D) array. ──────
    # (Pyro's `mcmc.diagnostics()` / `az.from_pyro(mcmc)` only see one chain
    #  at a time since we ran the chains sequentially.)
    summary_df = None
    idata = None
    try:
        import arviz as az
        idata = az.convert_to_inference_data({"xy": samples_by_chain.numpy()})
        rhat_vals = az.rhat(idata, var_names=["xy"])["xy"].values.flatten()
        ess_vals  = az.ess (idata, var_names=["xy"])["xy"].values.flatten()
        r_hat = torch.as_tensor(rhat_vals, dtype=torch.float32)
        n_eff = torch.as_tensor(ess_vals,  dtype=torch.float32)
        summary_df = az.summary(idata, var_names=["xy"])
    except Exception as exc:
        print(f"[nuts] arviz diagnostics skipped ({exc!r})")
        r_hat = torch.full((D,), float("nan"))
        n_eff = torch.full((D,), float("nan"))

    r_hat_max  = float(r_hat.max())
    r_hat_mean = float(r_hat.mean())
    n_eff_min  = float(n_eff.min())
    n_eff_med  = float(n_eff.median())
    print(f"[nuts] diagnostics: r_hat max={r_hat_max:.3f}  mean={r_hat_mean:.3f}  "
          f"n_eff min={n_eff_min:.0f}  median={n_eff_med:.0f}")
    if r_hat_max > R_HAT_WARN_THRESHOLD:
        print(f"[nuts] ⚠ r_hat max {r_hat_max:.3f} > {R_HAT_WARN_THRESHOLD} "
              f"— chains may not have mixed (Gelman & Rubin rule of thumb).")

    print(f"[nuts] best sample idx={best_idx} (chain={int(chain_ids[best_idx])})  "
          f"U={float(utilities[best_idx]):+.3f}  "
          f"median U={float(utilities.median()):+.3f}  "
          f"min={float(utilities.min()):+.3f}  max={float(utilities.max()):+.3f}")

    return dict(
        samples=samples_pooled.cpu(),
        samples_by_chain=samples_by_chain.cpu(),
        chain_ids=chain_ids,
        num_chains=K,
        num_samples_per_chain=N,
        utilities=utilities,
        best_x=best_x,
        best_y=best_y,
        best_idx=best_idx,
        best_chain=int(chain_ids[best_idx]),
        best_u=float(utilities[best_idx]),
        wall_seconds=dt,
        primary_batch_size=NUTS_BATCH_PRIMARIES,
        r_hat=r_hat,
        n_eff=n_eff,
        r_hat_max=r_hat_max,
        r_hat_mean=r_hat_mean,
        n_eff_min=n_eff_min,
        n_eff_med=n_eff_med,
        summary_df=summary_df,
        idata=idata,
    )


def _plot_curves(adam_log, nuts_result, path: str, adam_logs_all=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = nuts_result["num_chains"]
        N = nuts_result["num_samples_per_chain"]
        per_chain_U = nuts_result["utilities"].reshape(K, N).numpy()

        # Panels (left→right): [Adam best chain]? · [NUTS per-chain hist] ·
        # [all Adam chains]?. The two Adam panels appear only when their data
        # is supplied (adam_log / adam_logs_all).
        have_adam   = adam_log is not None
        have_chains = bool(adam_logs_all)
        n_panels = 1 + int(have_adam) + int(have_chains)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
        axes = list(np.atleast_1d(axes))

        col = 0
        ax_adam = axes[col] if have_adam else None
        if have_adam:
            col += 1
        ax_nuts = axes[col]; col += 1
        ax_chains = axes[col] if have_chains else None

        # Panel 1 — best Adam chain's U trajectory.
        if ax_adam is not None:
            ep = [e["epoch"] for e in adam_log]
            ax_adam.plot(ep, [e["U"] for e in adam_log], color="C0",
                         label="Adam U (best chain)")
            ax_adam.axhline(nuts_result["best_u"], color="C1", linestyle="--",
                            label=f"NUTS best U = {nuts_result['best_u']:.3f}")
            ax_adam.set_xlabel("Adam epoch")
            ax_adam.set_ylabel("U (composite)")
            ax_adam.set_title("Adam warm-start (best chain)")
            ax_adam.grid(alpha=0.3); ax_adam.legend(fontsize=9)

        # Panel 2 — NUTS per-chain sample-U histograms.
        for k in range(K):
            ax_nuts.hist(per_chain_U[k], bins=40, alpha=0.45,
                         label=f"chain {k}  median={np.median(per_chain_U[k]):.2f}",
                         edgecolor="none")
        ax_nuts.axvline(nuts_result["best_u"], color="black", linestyle="--",
                        label=f"pooled best = {nuts_result['best_u']:.3f}")
        ax_nuts.set_xlabel("sample U")
        ax_nuts.set_ylabel("count")
        ax_nuts.set_title(
            f"NUTS samples per chain "
            f"(K={K}, N={N}, r̂max={nuts_result['r_hat_max']:.3f})"
        )
        ax_nuts.grid(alpha=0.3); ax_nuts.legend(fontsize=8)

        # Panel 3 — every Adam chain's U trajectory (one line per chain).
        if ax_chains is not None:
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(adam_logs_all), 1)))
            for k, lg in enumerate(adam_logs_all):
                ep_k = [e["epoch"] for e in lg]
                u_k  = [e["U"]     for e in lg]
                ax_chains.plot(ep_k, u_k, color=colors[k], alpha=0.85, linewidth=1.0,
                               label=f"chain {k}  best={max(u_k):.2f}")
            ax_chains.set_xlabel("Adam epoch")
            ax_chains.set_ylabel("U (composite)")
            ax_chains.set_title(f"all Adam chains (K={len(adam_logs_all)})")
            ax_chains.grid(alpha=0.3); ax_chains.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] curves skipped ({exc!r})")


def _plot_layout(N_init, U_init,
                 x_adam, y_adam,
                 x_nuts, y_nuts,
                 nuts_x_std, nuts_y_std,
                 mountain, path: str):
    """2x2 figure: combined view + one panel per layout (init / Adam / NUTS).

    The NUTS panel adds per-detector marginal posterior std as error bars
    (computed across all NUTS samples — see caller). Colors:
      init  -> C0,  Adam best -> C2,  NUTS best -> C1.
    Colors are kept identical across the combined panel and the per-layout
    sub-panels so a detector reads the same in either view.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        COLOR_INIT = "C0"
        COLOR_ADAM = "C2"
        COLOR_NUTS = "C1"

        # Shared axis bounds: union of all four point sets so every subplot
        # uses the same window and detectors can be compared visually.
        all_x = np.concatenate([N_init, x_adam, x_nuts, mountain.centroids_NUE[:, 0]])
        all_y = np.concatenate([U_init, y_adam, y_nuts, mountain.centroids_NUE[:, 1]])
        pad_x = 0.05 * (all_x.max() - all_x.min() + 1.0)
        pad_y = 0.05 * (all_y.max() - all_y.min() + 1.0)
        xlim = (float(all_x.min() - pad_x), float(all_x.max() + pad_x))
        ylim = (float(all_y.min() - pad_y), float(all_y.max() + pad_y))

        def _mountain(ax):
            ax.scatter(mountain.centroids_NUE[:, 0], mountain.centroids_NUE[:, 1],
                       s=2, c="lightgray", alpha=0.6, label="mountain")

        def _frame(ax, title):
            ax.set_xlabel("North [m]"); ax.set_ylabel("Up [m]")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
            ax.legend(loc="best", fontsize=8)

        fig, axes = plt.subplots(2, 2, figsize=(13, 12))

        # (0, 0) — Combined: all three layouts on the same axes.
        ax = axes[0, 0]
        _mountain(ax)
        ax.scatter(N_init, U_init, s=22, c=COLOR_INIT, label="init",
                   alpha=0.55, edgecolors="none")
        ax.scatter(x_adam, y_adam, s=22, c=COLOR_ADAM, label="Adam best",
                   alpha=0.7, edgecolors="none")
        ax.scatter(x_nuts, y_nuts, s=30, c=COLOR_NUTS, label="NUTS best (pooled)",
                   alpha=0.85, edgecolors="black", linewidths=0.4)
        _frame(ax, "combined: init / Adam best / NUTS best")

        # (0, 1) — Init only.
        ax = axes[0, 1]
        _mountain(ax)
        ax.scatter(N_init, U_init, s=28, c=COLOR_INIT, label="init",
                   alpha=0.85, edgecolors="none")
        _frame(ax, "init layout")

        # (1, 0) — Adam best only.
        ax = axes[1, 0]
        _mountain(ax)
        ax.scatter(x_adam, y_adam, s=28, c=COLOR_ADAM, label="Adam best",
                   alpha=0.9, edgecolors="none")
        _frame(ax, "Adam best layout")

        # (1, 1) — NUTS best with marginal posterior 1σ ellipses.
        # Each ellipse is the axis-aligned 1σ contour from the marginal
        # NUTS posterior (width = 2·σx, height = 2·σy). Lighter fill,
        # darker dot at the centre = the best-sample position.
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection
        ax = axes[1, 1]
        _mountain(ax)
        ellipses = [
            Ellipse(xy=(float(x), float(y)),
                    width=2.0 * float(sx),
                    height=2.0 * float(sy))
            for x, y, sx, sy in zip(x_nuts, y_nuts, nuts_x_std, nuts_y_std)
        ]
        ax.add_collection(PatchCollection(
            ellipses, facecolor=COLOR_NUTS, edgecolor=COLOR_NUTS,
            alpha=0.25, linewidths=0.6,
        ))
        ax.scatter(x_nuts, y_nuts, s=22, c=COLOR_NUTS,
                   edgecolors="black", linewidths=0.4, alpha=0.95,
                   label=f"NUTS best  (1σ ellipse: "
                         f"σ̄x={float(nuts_x_std.mean()):.1f} m,  "
                         f"σ̄y={float(nuts_y_std.mean()):.1f} m)")
        _frame(ax, "NUTS best layout (1σ pooled posterior ellipses)")

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] layout skipped ({exc!r})")


def _plot_diagnostics(nuts_result, path: str):
    """ArviZ rank + trace plots for a handful of representative coordinates.

    Picks the coord with the worst r_hat, the best r_hat, and two arbitrary
    detector x/y entries — keeps the figure readable instead of dumping all
    2*N_DETECTORS dims."""
    idata = nuts_result.get("idata")
    if idata is None:
        print("[plot] diagnostics skipped (no arviz idata)")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import arviz as az

        r_hat = nuts_result["r_hat"].numpy()
        coords_to_show = sorted({
            int(np.argmax(r_hat)),
            int(np.argmin(r_hat)),
            0,
            N_DETECTORS,   # first y-coord
        })

        n = len(coords_to_show)
        fig, axes = plt.subplots(n, 2, figsize=(12, 2.4 * n))
        if n == 1:
            axes = axes.reshape(1, 2)

        for row, c in enumerate(coords_to_show):
            label = f"x[{c}]" if c < N_DETECTORS else f"y[{c - N_DETECTORS}]"
            az.plot_trace(
                idata, var_names=["xy"], coords={"xy_dim_0": [c]},
                axes=axes[row : row + 1, :], show=False,
            )
            axes[row, 0].set_title(f"{label}  (r̂={r_hat[c]:.3f})", fontsize=10)
            axes[row, 1].set_title(f"{label} trace", fontsize=10)

        fig.suptitle(
            f"NUTS multi-chain diagnostics  "
            f"(K={nuts_result['num_chains']}, N={nuts_result['num_samples_per_chain']}/chain, "
            f"r̂max={nuts_result['r_hat_max']:.3f}, ESS_min={nuts_result['n_eff_min']:.0f})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] diagnostics skipped ({exc!r})")


def _load_models():
    """Frozen FNN + recon, matching the conventions in 04_optimize.py."""
    fnn_ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    fnn_cfg  = fnn_ckpt.get("config", {})
    fnn = FNNSurrogate(
        n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=int(fnn_cfg.get("hidden", 512)),
        dropout=float(fnn_cfg.get("dropout", 0.1)),
    ).to(DEVICE)
    fnn.load_state_dict(fnn_ckpt["state_dict"])
    norm_stats = fnn_ckpt.get(
        "norm_stats",
        torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt")),
    )
    fnn.set_normalization(norm_stats)
    fnn.eval()
    for p in fnn.parameters():
        p.requires_grad_(False)
    print(f"[load] fnn.pt    epoch={fnn_ckpt.get('epoch','?')}  "
          f"val={fnn_ckpt.get('val_total', '?')}  hidden={int(fnn_cfg.get('hidden', 512))}")

    recon_ckpt = torch.load(os.path.join(RECON_FOLDER, "recon.pt"), map_location=DEVICE)
    cfg = recon_ckpt.get("config", {})
    recon = Reconstruction(
        n_det=int(recon_ckpt.get("num_detectors", N_DETECTORS)),
        input_features=int(recon_ckpt.get("input_features", 4)),
        output_dim=int(cfg.get("output_dim", 4)),
        hidden=int(cfg.get("hidden", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(DEVICE)
    recon.load_state_dict(recon_ckpt["state_dict"])
    recon.set_normalization(
        in_mean  = recon_ckpt["input_mean" ].to(DEVICE),
        in_std   = recon_ckpt["input_std"  ].to(DEVICE),
        out_mean = recon_ckpt["target_mean"].to(DEVICE),
        out_std  = recon_ckpt["target_std" ].to(DEVICE),
    )
    recon.eval()
    for p in recon.parameters():
        p.requires_grad_(False)
    print(f"[load] recon.pt  epoch={recon_ckpt.get('epoch','?')}  "
          f"val={recon_ckpt.get('val_total', '?')}")

    return fnn, recon


def _run_one_scheme(scheme: str,
                    mountain,
                    fnn: FNNSurrogate,
                    recon: Reconstruction,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    per_source):
    """Pre-computed Adam-bests → one (K * len(per_source))-chain NUTS run.

    `per_source` is an ordered mapping {source_label: (adam_bests, adam_logs,
    perturbed_inits)}. A single entry behaves like the per-scheme run; multiple
    entries are the combined run — chains from every source are stacked and the
    prior is anchored at the midpoint of all source Adam-bests instead of one.
    Same artifact set is written in both cases.
    """
    opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
    os.makedirs(opt_dir, exist_ok=True)
    is_combined = len(per_source) > 1
    print("-" * 72)
    print(f"[run] init_scheme={scheme}{'  (sources=' + str(list(per_source)) + ')' if is_combined else ''}  ->  {opt_dir}")

    # Flatten chains across all sources (preserving source label per chain).
    all_bests, all_logs, all_inits, source_per_chain = [], [], [], []
    for src, (bests, logs, inits) in per_source.items():
        for (bx, by), log, init in zip(bests, logs, inits):
            all_bests.append((bx, by))
            all_logs.append(log)
            all_inits.append(init)
            source_per_chain.append(src)

    # Global Adam-best across all sources → representative warm-start.
    best_k = int(np.argmax([max(e["U"] for e in log) for log in all_logs]))
    x_adam, y_adam = all_bests[best_k]
    adam_log       = all_logs[best_k]
    x_init, y_init = all_inits[best_k]
    best_adam_src  = source_per_chain[best_k]

    init_stacked = torch.cat(
        [torch.cat([bx, by], dim=0).unsqueeze(0) for bx, by in all_bests], dim=0,
    )
    # Prior anchor = the single global Adam-best layout (a real, fully-spread
    # configuration), for both per-scheme and combined runs. NOT the per-index
    # mean across layouts: detectors have no consistent index->position mapping
    # across schemes / permutation-equivariant runs, so averaging would collapse
    # the anchor toward the centroid and bias NUTS toward a too-central layout.
    nuts_result = nuts_sampling_multichain(
        x_adam, y_adam, fnn, recon, primary_all, n_total_primaries,
        init_chains=init_stacked,
    )

    x_nuts_proj, y_nuts_proj = mountain.project_to_mountain(
        nuts_result["best_x"], nuts_result["best_y"],
    )
    best_nuts_src = source_per_chain[nuts_result["best_chain"]]

    # Persist artifacts (identical schema for single + combined; combined adds
    # a `source` field on the best layouts + `source_per_chain` on samples).
    torch.save({"x": x_init, "y": y_init, "scheme": scheme,
                "source": best_adam_src},
               os.path.join(opt_dir, "layout_init.pt"))
    torch.save({"x": x_adam, "y": y_adam,
                "U": adam_log[-1]["U"],
                "best_U": max(e["U"] for e in adam_log),
                "source": best_adam_src},
               os.path.join(opt_dir, "layout_adam.pt"))
    torch.save({"x": x_nuts_proj, "y": y_nuts_proj,
                "x_raw": nuts_result["best_x"], "y_raw": nuts_result["best_y"],
                "U": nuts_result["best_u"],
                "sample_idx": nuts_result["best_idx"],
                "chain": nuts_result["best_chain"],
                "source": best_nuts_src},
               os.path.join(opt_dir, "layout_best.pt"))
    torch.save({"samples": nuts_result["samples"],
                "samples_by_chain": nuts_result["samples_by_chain"],
                "chain_ids": nuts_result["chain_ids"],
                "source_per_chain": source_per_chain,
                "utilities": nuts_result["utilities"],
                "r_hat": nuts_result["r_hat"],
                "n_eff": nuts_result["n_eff"]},
               os.path.join(opt_dir, "nuts_samples.pt"))

    if nuts_result.get("summary_df") is not None:
        csv_path = os.path.join(opt_dir, "nuts_diagnostics.csv")
        nuts_result["summary_df"].to_csv(csv_path)
        print(f"[save] {csv_path}  ({len(nuts_result['summary_df'])} rows)")

    adam_best_U = max(e["U"] for e in adam_log)
    with open(os.path.join(opt_dir, "optimize_log.json"), "w") as f:
        json.dump({
            "adam_log": adam_log,
            "adam_best_U": adam_best_U,
            "adam_best_source": best_adam_src,
            "sources": list(per_source),
            "source_per_chain": source_per_chain,
            "nuts_best_U": nuts_result["best_u"],
            "nuts_best_sample_idx": nuts_result["best_idx"],
            "nuts_best_chain": nuts_result["best_chain"],
            "nuts_best_source": best_nuts_src,
            "nuts_wall_seconds": nuts_result["wall_seconds"],
            "nuts_utility_stats": dict(
                mean=float(nuts_result["utilities"].mean()),
                median=float(nuts_result["utilities"].median()),
                std=float(nuts_result["utilities"].std()),
                min=float(nuts_result["utilities"].min()),
                max=float(nuts_result["utilities"].max()),
            ),
            "nuts_diagnostics": dict(
                r_hat_max=nuts_result["r_hat_max"],
                r_hat_mean=nuts_result["r_hat_mean"],
                n_eff_min=nuts_result["n_eff_min"],
                n_eff_median=nuts_result["n_eff_med"],
                r_hat_warn_threshold=R_HAT_WARN_THRESHOLD,
                converged=nuts_result["r_hat_max"] <= R_HAT_WARN_THRESHOLD,
            ),
            "config": dict(
                init_scheme=scheme,
                n_adam_epochs=N_ADAM_EPOCHS,
                primaries_per_step=PRIMARIES_PER_STEP,
                adam_lr=ADAM_LR, grad_clip=GRAD_CLIP,
                nuts_num_chains=NUTS_NUM_CHAINS,
                nuts_num_samples_per_chain=NUTS_NUM_SAMPLES,
                nuts_warmup_per_chain=NUTS_WARMUP,
                nuts_temperature=NUTS_TEMPERATURE,
                nuts_prior_sigma=NUTS_PRIOR_SIGMA,
                nuts_init_overdisp_sigma=NUTS_INIT_OVERDISP_SIGMA,
                nuts_batch_primaries=NUTS_BATCH_PRIMARIES,
                nuts_target_accept_prob=NUTS_TARGET_ACCEPT_PROB,
                nuts_max_tree_depth=NUTS_MAX_TREE_DEPTH,
                nuts_mp_context=NUTS_MP_CONTEXT,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                seed=SEED,
            ),
        }, f, indent=2)

    _plot_curves(adam_log, nuts_result, os.path.join(opt_dir, "optimize_curves.png"),
                 adam_logs_all=all_logs)

    _samples = nuts_result["samples"]                            # (K*N, 2*n_det)
    _x_std = _samples[:, :N_DETECTORS].std(dim=0)
    _y_std = _samples[:, N_DETECTORS:].std(dim=0)
    _plot_layout(
        x_init.numpy(), y_init.numpy(),
        x_adam.numpy(), y_adam.numpy(),
        x_nuts_proj.numpy(), y_nuts_proj.numpy(),
        _x_std.numpy(), _y_std.numpy(),
        mountain,
        os.path.join(opt_dir, "layout_before_after.png"),
    )
    _plot_diagnostics(nuts_result, os.path.join(opt_dir, "nuts_diagnostics.png"))

    src_str = f"  (best src={best_nuts_src})" if is_combined else ""
    print(f"[done] scheme={scheme}  Adam best U={adam_best_U:+.3f}  "
          f"NUTS best U={nuts_result['best_u']:+.3f}{src_str}  "
          f"r̂max={nuts_result['r_hat_max']:.3f}  ({opt_dir})")
    return dict(scheme=scheme, adam_best_U=adam_best_U,
                nuts_best_U=nuts_result["best_u"],
                r_hat_max=nuts_result["r_hat_max"],
                n_eff_min=nuts_result["n_eff_min"],
                opt_dir=opt_dir)


def main():
    print("=" * 72)
    print("v6/04_optimize_hmc_chains.py — Adam warm-start + multi-chain NUTS")
    print("=" * 72)
    print(f"device           : {DEVICE}")
    print(f"init schemes     : {INIT_SCHEMES}")
    print(f"Adam epochs      : {N_ADAM_EPOCHS}  (primaries/step={PRIMARIES_PER_STEP})")
    print(f"NUTS chains      : {NUTS_NUM_CHAINS}  (mp_context={NUTS_MP_CONTEXT})")
    print(f"NUTS samples     : {NUTS_NUM_SAMPLES}/chain (warmup={NUTS_WARMUP}/chain)")
    print(f"NUTS temp / σ    : T={NUTS_TEMPERATURE}  σ_prior={NUTS_PRIOR_SIGMA} m  "
          f"σ_init={NUTS_INIT_OVERDISP_SIGMA} m")

    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    n_total_primaries = int(primary_all.shape[0])
    print(f"[load] {n_total_primaries} primaries")

    fnn, recon = _load_models()

    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    results = []
    per_scheme = {}                # scheme -> (adam_bests, adam_logs, perturbed_inits)
    for scheme in INIT_SCHEMES:
        print()
        print("=" * 72)
        print(f"init scheme: {scheme}")
        print("=" * 72)
        torch.manual_seed(SEED); np.random.seed(SEED)
        g = torch.Generator().manual_seed(SEED)
        per_scheme[scheme] = _perturbed_adam_runs(
            scheme, NUTS_NUM_CHAINS, g, mountain,
            fnn, recon, primary_all, n_total_primaries,
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
        gain = r["nuts_best_U"] - r["adam_best_U"]
        flag = "" if r["r_hat_max"] <= R_HAT_WARN_THRESHOLD else "  ⚠ NOT CONVERGED"
        print(f"  {r['scheme']:<10}  Adam={r['adam_best_U']:+.3f}  "
              f"NUTS={r['nuts_best_U']:+.3f}  Δ={gain:+.3f}  "
              f"r̂max={r['r_hat_max']:.3f}  ESS_min={r['n_eff_min']:.0f}  "
              f"->  {r['opt_dir']}{flag}")


if __name__ == "__main__":
    main()
