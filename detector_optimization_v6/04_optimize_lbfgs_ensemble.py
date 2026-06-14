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
from scipy.optimize import linear_sum_assignment

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.dual_surrogate import DualSpeciesSurrogate, load_dual_surrogate
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
INIT_SCHEMES         = ("grid", "center")
RUN_COMBINED         = True
COMBINED_SCHEME_NAME = "combined"
OPT_DIR_TEMPLATE     = OPT_FOLDER + "_lbfgs_ensemble_{scheme}"

# K perturbed restarts per scheme.
N_CHAINS            = 15#4
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

# Utility composite weights — match 04_optimize.py
W_THETA = 1e2
W_PHI   = 1e2
W_E     = 2.5e2
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
                  fnn: DualSpeciesSurrogate,
                  recon: Reconstruction):
    """Differentiable composite U for a layout against a primary batch.

    `fnn` is the dual-species wrapper: both per-species surrogates are
    evaluated with the same primary + layout and physically combined, so the
    backprop into (x_det, y_det) flows through BOTH models.

    Mirrors the inner loop of `_run_optimization` in 04_optimize.py so this
    script optimizes the SAME objective (the U_PR term is computed but
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
    return U, r, dict(u_theta=W_THETA * u_theta / W_DIV, u_phi=W_PHI * u_phi / W_DIV, u_e=W_E * u_e / W_DIV, u_pr=W_PR * u_pr / W_DIV)


def adam_warm_start(scheme: str,
                    mountain,
                    fnn: DualSpeciesSurrogate,
                    recon: Reconstruction,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    init_override):
    """N_ADAM_EPOCHS of Adam with mountain projection. Returns:
       (best_x, best_y, init_x, init_y, log, grad_hist). `init_override=(x, y)`
       is the (already mountain-projected) starting layout; `scheme` is a log
       label. `grad_hist` is a (N_ADAM_EPOCHS, 2*n_det) CPU tensor of the flat
       parameter gradient at each step (for cross-run gradient diagnostics)."""
    N_init, U_init = init_override
    N_init = N_init.float()
    U_init = U_init.float()
    print(f"[adam] init {scheme}  N in [{N_init.min():.1f}, {N_init.max():.1f}]  "
          f"Up in [{U_init.min():.1f}, {U_init.max():.1f}]")

    xy_module = LearnableXY(N_init, U_init, device=str(DEVICE)).to(DEVICE)
    optimizer = torch.optim.Adam(xy_module.parameters(), lr=ADAM_LR)

    log = []
    grad_hist = []
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
            N_cpu  = xy_module.x.detach().cpu()
            Up_cpu = xy_module.y.detach().cpu()
            N_new, Up_new = mountain.project_to_mountain(N_cpu, Up_cpu)
            xy_module.x.data.copy_(N_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(Up_new.to(DEVICE).to(xy_module.y.dtype))

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
    return best_x, best_y, N_init, U_init, log, grad_hist


def _build_chain_inits(init_x: torch.Tensor, init_y: torch.Tensor,
                       K: int, generator: torch.Generator) -> torch.Tensor:
    """K overdispersed starts around (init_x, init_y). Returns (K, 2*n_det) on DEVICE."""
    base = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach()  # (D,)
    perturb = torch.randn(
        K, base.numel(), generator=generator, device="cpu",
    ).to(DEVICE) * INIT_OVERDISP_SIGMA
    return base.unsqueeze(0) + perturb                                        # (K, D)


def _consecutive_cos_distance(grad_hist, window: int = 1) -> np.ndarray:
    """Cosine distance between consecutive-step gradients within ONE run.

    grad_hist : (T, D) tensor — flat gradient at each optimizer step of a single
    run. With `window > 1`, the raw gradient VECTORS are first averaged over a
    trailing window of `window` steps (ĝ_t = mean(g over last W)); averaging
    vectors cancels zero-mean minibatch noise *before* the nonlinear cosine, so
    the result reflects the underlying descent direction rather than per-step
    estimator noise. Returns a (T_eff-1,) array of 1 - cos(ĝ_t, ĝ_{t-1}):
    how much the (smoothed) direction turned. ~0 steady, ~1 orthogonal, ~2
    reversal."""
    if grad_hist is None or grad_hist.numel() == 0 or grad_hist.shape[0] < 2:
        return np.zeros(0)
    G = grad_hist.numpy().astype(np.float64)
    W = max(int(window), 1)
    if W > 1 and G.shape[0] >= W:
        # Trailing moving average of the raw vectors via cumulative sums:
        # Ḡ[t] = mean(G[t-W+1 : t+1]). Output length T - W + 1.
        cs = np.cumsum(G, axis=0)
        cs = np.concatenate([np.zeros((1, G.shape[1])), cs], axis=0)
        G = (cs[W:] - cs[:-W]) / W
    if G.shape[0] < 2:
        return np.zeros(0)
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    cos = (G[1:] * G[:-1]).sum(axis=1)        # cos(ĝ_t, ĝ_{t-1})
    return 1.0 - cos


def _perturbed_adam_runs(scheme: str, K: int, generator: torch.Generator,
                         mountain, fnn, recon, primary_all, n_total_primaries):
    """K pre-Adam perturbations of the scheme init → K Adam runs.

    Returns (adam_bests, adam_logs, perturbed_inits, adam_grads), each length K.
    adam_grads[k] is the (N_ADAM_EPOCHS, 2*n_det) per-step gradient history."""
    N_np, U_np = mountain.sample_initial_layout(n_units=N_DETECTORS, scheme=scheme)
    N_t = torch.as_tensor(N_np, dtype=torch.float32)
    U_t = torch.as_tensor(U_np, dtype=torch.float32)
    N_t, U_t = mountain.project_to_mountain(N_t, U_t)
    chains_init = _build_chain_inits(N_t, U_t, K, generator)                  # (K, D)

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
                 recon: Reconstruction,
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
        x_proj, y_proj = mountain.project_to_mountain(x_cpu, y_cpu)
        U_proj, _, _ = utility_of_xy(
            x_proj.to(DEVICE), y_proj.to(DEVICE), primary_fixed, fnn, recon,
        )
    grad_hist = torch.stack(grad_hist, dim=0) if grad_hist else torch.zeros(0)
    return x_proj.float(), y_proj.float(), float(U_proj.item()), iter_log, grad_hist


def _assign(cost: np.ndarray) -> np.ndarray:
    """One-to-one assignment minimizing total cost. Returns col[i] = column
    assigned to row i. Uses scipy's optimal Hungarian if available, else a
    dependency-free greedy global-minimum matcher (good enough for grouping
    n_det≈100 detectors by closest position)."""
    _, col = linear_sum_assignment(cost)
    return col


def align_to_reference(layouts_xy: np.ndarray, ref_idx: int):
    """Permutation-invariant alignment of K layouts to a reference.

    layouts_xy : (K, n_det, 2). For each run, solve the one-to-one assignment
    minimizing total squared distance between its detectors and the reference
    run's detectors, then reorder its detectors so column i of every run is the
    same *physical position group* (not the same network input index).
    Returns (aligned (K, n_det, 2), perms (K, n_det))."""
    K, n_det, _ = layouts_xy.shape
    ref = layouts_xy[ref_idx]
    aligned = np.empty_like(layouts_xy)
    perms = np.empty((K, n_det), dtype=np.int64)
    for k in range(K):
        if k == ref_idx:
            aligned[k] = ref
            perms[k] = np.arange(n_det)
            continue
        L = layouts_xy[k]
        # cost[i, j] = || ref[i] - L[j] ||^2
        diff = ref[:, None, :] - L[None, :, :]      # (n_det, n_det, 2)
        cost = (diff * diff).sum(axis=-1)           # (n_det, n_det)
        col = _assign(cost)
        aligned[k] = L[col]
        perms[k] = col
    return aligned, perms


def _plot_curves(adam_logs, lbfgs_logs, adam_grads, lbfgs_grads, path: str):
    """Two panels: (1) combined Adam→L-BFGS U trajectory, one line per run with
    the SAME color across both phases (Adam solid, L-BFGS dashed) and a vertical
    divider at the switch; (2) consecutive-step gradient cosine distance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(adam_logs), 1)
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
        colors = plt.cm.tab10(np.linspace(0, 1, K))

        # Panel 1 — combined Adam + L-BFGS U, one continuous line per run.
        # Adam epochs 1..A, then L-BFGS closure calls shifted to A+1..A+L.
        adam_switch = max((len(lg) for lg in adam_logs), default=0)
        for k in range(K):
            a_lg = adam_logs[k] if k < len(adam_logs) else []
            l_lg = lbfgs_logs[k] if k < len(lbfgs_logs) else []
            a_u = [e["U"] for e in a_lg]
            l_u = [e["U"] for e in l_lg]
            best = max(a_u + l_u) if (a_u or l_u) else float("nan")
            if a_u:
                axes[0].plot(np.arange(1, len(a_u) + 1), a_u, color=colors[k],
                             alpha=0.85, linewidth=1.0, linestyle="-",
                             label=f"chain {k}  best={best:.2f}")
            if l_u:
                xl = np.arange(adam_switch + 1, adam_switch + 1 + len(l_u))
                axes[0].plot(xl, l_u, color=colors[k], alpha=0.85, linewidth=1.0,
                             linestyle="--",
                             label=None if a_u else f"chain {k}  best={best:.2f}")
        if adam_switch:
            axes[0].axvline(adam_switch + 0.5, color="gray", linestyle=":",
                            alpha=0.7, label="Adam→L-BFGS")
        axes[0].set_xlabel("optimizer step (Adam epochs → L-BFGS calls)")
        axes[0].set_ylabel("U (composite)")
        axes[0].set_title(f"optimization: Adam (solid) + L-BFGS (dashed), K={K}")
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left",)

        # Panel 2 — consecutive-step gradient cosine distance, one line per run.
        # Solid = Adam phase, dashed = L-BFGS phase (same color = same run).
        # Raw (W=1) drawn faint behind the W-step vector-averaged line (bold);
        # both x-shift L-BFGS after the Adam steps. adam_len from the RAW series
        # so Adam and L-BFGS share one x-axis regardless of smoothing window.
        adam_len = max((len(_consecutive_cos_distance(g, 1)) for g in (adam_grads or [])),
                       default=0)
        any_line = False
        for k in range(len(adam_grads or [])):
            for grads, x0, dashed in (
                (adam_grads[k], 0, False),
                (lbfgs_grads[k] if lbfgs_grads else None, adam_len, True),
            ):
                if grads is None:
                    continue
                raw  = _consecutive_cos_distance(grads, 1)
                if len(raw):
                    axes[1].plot(np.arange(x0 + 1, x0 + 1 + len(raw)), raw,
                                 color=colors[k % K], alpha=0.1, linewidth=0.7,
                                 linestyle="--" if dashed else "-")
                    any_line = True
                sm = _consecutive_cos_distance(grads, GRAD_COS_WINDOW)
                if len(sm):
                    # Smoothed series is shorter by (W-1); center it on its window.
                    off = x0 + (len(raw) - len(sm)) // 2 + 1
                    axes[1].plot(np.arange(off, off + len(sm)), sm,
                                 color=colors[k % K], alpha=0.9, linewidth=1.6,
                                 linestyle="--" if dashed else "-",
                                 label=f"run {k}" if not dashed else None)
        if adam_len and lbfgs_grads:
            axes[1].axvline(adam_len + 0.5, color="gray", linestyle=":", alpha=0.6,
                            label="Adam→L-BFGS")
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("cos distance (consecutive grads)")
        axes[1].set_title(f"per-run gradient-direction turn "
                          f"(W={GRAD_COS_WINDOW}-step vector avg; raw faint)")
        axes[1].grid(alpha=0.3)
        if any_line:
            axes[1].legend(fontsize=7)
        else:
            axes[1].text(0.5, 0.5, "no gradient history", ha="center", va="center",
                         transform=axes[1].transAxes, fontsize=10)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] curves skipped ({exc!r})")


def _plot_utility_components(adam_logs, lbfgs_logs, path: str):
    """One subfigure per chain. Each shows the weighted utility sub-parts
    (θ, φ, E contributions = W_x·u_x / W_DIV, which sum to U) over the combined
    Adam→L-BFGS trajectory, plus a bold black line for the overall utility U.
    Adam phase solid, L-BFGS phase dashed; a vertical divider marks the switch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(adam_logs), len(lbfgs_logs), 1)
        ncol = min(K, 5)
        nrow = math.ceil(K / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow),
                                 squeeze=False, sharex=False)
        axes_flat = axes.flatten()

        # (label, log-key, weight). The logged u_* are ALREADY the weighted
        # contributions (W_x * u_x / W_DIV; see utility_of_xy), and they sum to
        # U by construction — so plot them as-is (weight 1.0). Re-applying the
        # weight here double-counts it and shrinks the sub-curves below U.
        parts = [
            ("θ", "u_theta", 1.0, "C0"),
            ("φ", "u_phi",   1.0, "C1"),
            ("E", "u_e",     1.0, "C2"),
        ]
        adam_switch = max((len(lg) for lg in adam_logs), default=0)

        for k in range(K):
            ax = axes_flat[k]
            a_lg = adam_logs[k] if k < len(adam_logs) else []
            l_lg = lbfgs_logs[k] if k < len(lbfgs_logs) else []
            xa = np.arange(1, len(a_lg) + 1)
            xl = np.arange(adam_switch + 1, adam_switch + 1 + len(l_lg))

            for label, key, w, col in parts:
                if a_lg:
                    ax.plot(xa, [e[key] * w for e in a_lg], color=col,
                            linewidth=1.0, linestyle="-", alpha=0.85, label=label)
                if l_lg:
                    ax.plot(xl, [e[key] * w for e in l_lg], color=col,
                            linewidth=1.0, linestyle="--", alpha=0.85,
                            label=None if a_lg else label)
            # Overall utility U (bold black).
            if a_lg:
                ax.plot(xa, [e["U"] for e in a_lg], color="black",
                        linewidth=1.8, linestyle="-", label="U (overall)")
            if l_lg:
                ax.plot(xl, [e["U"] for e in l_lg], color="black",
                        linewidth=1.8, linestyle="--",
                        label=None if a_lg else "U (overall)")
            if adam_switch:
                ax.axvline(adam_switch + 0.5, color="gray", linestyle=":", alpha=0.6)
            ax.set_title(f"chain {k}", fontsize=10)
            ax.set_xlabel("optimizer step"); ax.set_ylabel("utility")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)

        # Hide any unused axes in the grid.
        for j in range(K, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle("per-chain utility decomposition "
                     "(weighted θ/φ/E sub-parts + overall U; Adam solid, L-BFGS dashed)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] utility components skipped ({exc!r})")


def _plot_ensemble(aligned_xy: np.ndarray,
                   mean_xy: np.ndarray,
                   std_xy: np.ndarray,
                   best_x, best_y,
                   mountain, path: str):
    """Mountain top-down: every aligned run's detectors (faint) + per-group
    mean (dark) + 1σ ellipses (width=2σx, height=2σy)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection

        K = aligned_xy.shape[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(mountain.centroids_NUE[:, 0], mountain.centroids_NUE[:, 1],
                   s=2, c="lightgray", alpha=0.6, label="mountain")

        colors = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))
        for k in range(K):
            ax.scatter(aligned_xy[k, :, 0], aligned_xy[k, :, 1], s=8,
                       color=colors[k % 10], alpha=0.35, edgecolors="none",
                       label=f"run {k}" if k < 10 else None)

        ellipses = [
            Ellipse(xy=(float(mx), float(my)),
                    width=2.0 * float(sx), height=2.0 * float(sy))
            for (mx, my), (sx, sy) in zip(mean_xy, std_xy)
        ]
        ax.add_collection(PatchCollection(
            ellipses, facecolor="C1", edgecolor="C1", alpha=0.25, linewidths=0.6,
        ))
        ax.scatter(best_x, best_y, s=26, c="C3",
                   edgecolors="black", linewidths=0.4, alpha=0.95,
                   label=f"best  (σ̄x={std_xy[:,0].mean():.1f} m, "
                         f"σ̄y={std_xy[:,1].mean():.1f} m)")

        ax.set_xlabel("North [m]"); ax.set_ylabel("Up [m]")
        ax.set_aspect("equal")
        ax.set_title(f"L-BFGS ensemble (K={K}) — aligned best + 1σ ellipses")
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] ensemble skipped ({exc!r})")


def _plot_density_heatmap(aligned_xy: np.ndarray,
                          best_x, best_y,
                          mountain, path: str,
                          bins: int = 60):
    """Mountain top-down 2D density of where detectors land across the ensemble.

    Pools every detector position from all K aligned runs (K * n_det points)
    into a 2D histogram over (North, Up); brighter cells = positions favored by
    more runs. The mountain footprint is outlined faintly underneath, and the
    single best-U layout is overlaid as a scatter so the densest regions can be
    read against the actual winning layout."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K, n_det, _ = aligned_xy.shape
        pts = aligned_xy.reshape(-1, 2)                          # (K*n_det, 2)

        # Histogram extent spans EXACTLY the union of the mountain footprint and
        # the detector positions — no outer padding, otherwise the pad rows/cols
        # carry no data and render as a permanent transparent border (the white
        # top/bottom lines). The extreme points then live in the edge bins.
        cen = getattr(mountain, "centroids_NUE", None)
        if cen is not None:
            allx = np.concatenate([cen[:, 0], pts[:, 0]])
            ally = np.concatenate([cen[:, 1], pts[:, 1]])
        else:
            allx, ally = pts[:, 0], pts[:, 1]
        xlo, xhi = float(allx.min()), float(allx.max())
        ylo, yhi = float(ally.min()), float(ally.max())
        extent = [xlo, xhi, ylo, yhi]

        rng = [[extent[0], extent[1]], [extent[2], extent[3]]]
        H, xedges, yedges = np.histogram2d(
            pts[:, 0], pts[:, 1], bins=bins, range=rng,
        )
        H = H / max(K, 1)            # → expected detectors per cell per run

        # Aligned detectors cluster into ~n_det tight spots, so raw bins read as
        # scattered specks. A light Gaussian blur turns those into legible
        # density blobs ("where detectors concentrate"). Falls back to raw H if
        # scipy is unavailable.
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=1.0)
        except Exception:
            pass

        # Color ONLY the mountain footprint: histogram the centroids onto the
        # same grid → cells that contain mountain, dilated a little to bridge the
        # gaps between the (sparse) centroids, then mask everything else so the
        # off-mountain region renders transparent.
        if cen is not None:
            occ, _, _ = np.histogram2d(cen[:, 0], cen[:, 1], bins=bins, range=rng)
            # Detectors are projected onto the mountain, so cells holding a
            # detector are valid mountain too — union them in so no scatter
            # point ever lands outside the colored region (the centroid sample
            # is slightly narrower than the layout footprint at the edges).
            det_occ, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=rng)
            mask = (occ > 0) | (det_occ > 0)
            try:
                from scipy.ndimage import (binary_dilation, binary_fill_holes,
                                           binary_erosion)
                # Dilate to bridge gaps between the sparse centroids, fill the
                # interior, then erode back so the outer boundary stays put —
                # leaves a solid mountain footprint with no speckle holes.
                mask = binary_dilation(mask, iterations=2)
                mask = binary_fill_holes(mask)
                # border_value=1 so the erosion doesn't strip the array-edge
                # rows/cols (otherwise the top & bottom lines go transparent).
                mask = binary_erosion(mask, iterations=1, border_value=1)
            except Exception:
                pass
            H = np.ma.masked_array(H, mask=~mask)

        # Figure aspect ≈ data aspect so the equal-aspect map fills the frame
        # vertically (and the axes-height colorbar ends up as tall as the image).
        data_ar = (extent[3] - extent[2]) / (extent[1] - extent[0])
        fig_w = 14.0
        fig, ax = plt.subplots(figsize=(fig_w, max(fig_w * data_ar + 1.2, 3.0)))

        cmap = plt.cm.magma.copy()
        cmap.set_bad(alpha=0.0)          # masked / off-mountain → transparent
        im = ax.imshow(
            H.T, origin="lower", extent=extent, aspect="equal",
            cmap=cmap, interpolation="bilinear", zorder=0,
        )

        # Colorbar exactly as tall as the map.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("detector density (count per run per cell)")

        ax.scatter(np.asarray(best_x), np.asarray(best_y), s=22, c="cyan",
                   edgecolors="black", linewidths=0.4, alpha=0.95, zorder=3,
                   label="best-U layout")

        ax.set_xlabel("North [m]"); ax.set_ylabel("Up [m]")
        ax.set_title(f"detector placement density (K={K} runs, {bins}×{bins} bins) "
                     f"+ best-U layout")
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] density heatmap skipped ({exc!r})")


def _load_models():
    """Frozen dual-species surrogate + recon.

    The dual wrapper holds fnn_electron.pt + fnn_muon.pt (frozen, eval); its
    combined output feeds recon and reconstructability exactly like the old
    single fnn, and gradients flow through both branches."""
    fnn = load_dual_surrogate(FNN_FOLDER, DEVICE)

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
                    fnn: DualSpeciesSurrogate,
                    recon: Reconstruction,
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
    adam_cos_per_run  = [_consecutive_cos_distance(g, GRAD_COS_WINDOW).tolist()
                         for g in all_adam_grads]
    lbfgs_cos_per_run = [_consecutive_cos_distance(g, GRAD_COS_WINDOW).tolist()
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

    _plot_curves(all_adam_logs, lbfgs_logs, all_adam_grads, all_lbfgs_grads,
                 os.path.join(opt_dir, "optimize_curves.png"))
    _plot_utility_components(all_adam_logs, lbfgs_logs,
                            os.path.join(opt_dir, "utility_components.png"))
    _plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                    mountain, os.path.join(opt_dir, "layout_ensemble.png"))
    _plot_density_heatmap(aligned, best_x, best_y,
                    mountain, os.path.join(opt_dir, "layout_density.png"))

    print(f"[done] scheme={scheme}  best U={refined_U[ref_idx]:+.3f}  "
          f"σ̄=({std_xy[:,0].mean():.1f}, {std_xy[:,1].mean():.1f}) m  ({opt_dir})")
    return dict(scheme=scheme, best_U=refined_U[ref_idx],
                mean_std_x=float(std_xy[:, 0].mean()),
                mean_std_y=float(std_xy[:, 1].mean()),
                opt_dir=opt_dir)


def main():
    global N_CHAINS, N_ADAM_EPOCHS, LBFGS_MAX_ITER
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=N_CHAINS)
    ap.add_argument("--adam-epochs", type=int, default=N_ADAM_EPOCHS)
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_MAX_ITER)
    args = ap.parse_args()
    N_CHAINS, N_ADAM_EPOCHS, LBFGS_MAX_ITER = \
        int(args.chains), int(args.adam_epochs), int(args.lbfgs_iters)

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

    fnn, recon = _load_models()

    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    results = []
    per_scheme = {}     # scheme -> (adam_bests, adam_logs, perturbed_inits)
    for scheme in INIT_SCHEMES:
        print()
        print("=" * 72)
        print(f"init scheme: {scheme}")
        print("=" * 72)
        torch.manual_seed(SEED); np.random.seed(SEED)
        g = torch.Generator().manual_seed(SEED)
        per_scheme[scheme] = _perturbed_adam_runs(
            scheme, N_CHAINS, g, mountain, fnn, recon, primary_all, n_total_primaries,
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


if __name__ == "__main__":
    main()
