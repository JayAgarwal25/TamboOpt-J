"""Optimize detector positions: Adam warm-start, then NUTS sampling.

A two-phase variant of `04_optimize.py`:

1.  Phase 1 — Adam (100 epochs). Standard gradient descent through the
    frozen FNN + recon stack with random primary batches per epoch and
    mountain projection after each step. Mirrors the recipe in
    `04_optimize.py` but truncated to provide only a warm-start.
2.  Phase 2 — NUTS (Pyro). The Adam-best layout becomes the initial
    state for a No-U-Turn Sampler whose target log-density is

        log p(xy) = U(xy) / T  +  log N(xy | xy_adam, σ_prior²)

    where U is the same composite utility used in Phase 1 (evaluated on
    a FIXED primary batch so the target is deterministic) and the
    Normal prior holds samples within ~3σ_prior of the Adam optimum to
    keep them inside the FNN's training distribution. After sampling,
    every draw is re-scored on the same fixed batch and the highest-U
    layout is projected back onto the mountain surface.

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 04_optimize_nuts.py

Artifacts (per INIT_SCHEMES entry) land in
`<OPT_FOLDER>_nuts_{scheme}/`:

    layout_best.pt          best NUTS sample (mountain-projected)
    layout_adam.pt          Adam-phase best layout (mountain-projected)
    layout_init.pt          initial layout from the chosen scheme
    nuts_samples.pt         all NUTS samples + per-sample utilities
    optimize_log.json       Adam per-epoch log + NUTS summary + config
    optimize_curves.png     Adam U trajectory + NUTS sample-U histogram
    layout_before_after.png mountain top-down with init / Adam / NUTS layouts
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
# Each entry produces its own output folder so grid/center variants can be
# compared side-by-side.
INIT_SCHEMES     = ("grid", "center")
OPT_DIR_TEMPLATE = OPT_FOLDER + "_nuts_{scheme}"

# Adam warm-start
N_ADAM_EPOCHS       = 5_000
PRIMARIES_PER_STEP  = 256
ADAM_LR             = 1.0
GRAD_CLIP           = 100.0
ADAM_LOG_EVERY      = 10

# NUTS sampling
NUTS_NUM_SAMPLES         = 5000
NUTS_WARMUP              = 5000
NUTS_TEMPERATURE         = 3.0 # 1   # higher = flatter target, more exploration
NUTS_PRIOR_SIGMA         = 500.0 # 100  # metres around the Adam optimum
NUTS_BATCH_PRIMARIES     = 512     # fixed batch for the deterministic target
NUTS_TARGET_ACCEPT_PROB  = 0.8
NUTS_MAX_TREE_DEPTH      = 7       # cap leapfrog tree to keep wall-time bounded

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
                    n_total_primaries: int):
    """N_ADAM_EPOCHS of Adam with mountain projection. Returns:
       (best_x, best_y, init_x, init_y, log)."""
    N_init_np, U_init_np = mountain.sample_initial_layout(
        n_units=N_DETECTORS, scheme=scheme,
    )
    N_init = torch.as_tensor(N_init_np, dtype=torch.float32)
    U_init = torch.as_tensor(U_init_np, dtype=torch.float32)
    N_init, U_init = mountain.project_to_mountain(N_init, U_init)
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


def nuts_sampling(init_x: torch.Tensor,
                  init_y: torch.Tensor,
                  fnn: FNNSurrogate,
                  recon: Reconstruction,
                  primary_all: torch.Tensor,
                  n_total_primaries: int):
    """Sample layouts from the U-weighted posterior anchored at the Adam optimum.

    Returns a dict with the raw samples (CPU), per-sample utility, and the
    best-by-utility sample (unprojected — caller projects to mountain)."""
    # Deterministic target: fixed primary batch held constant for warmup + sampling.
    g = torch.Generator().manual_seed(SEED)
    idx_fixed = torch.randint(0, n_total_primaries, (NUTS_BATCH_PRIMARIES,), generator=g)
    primary_fixed = primary_all[idx_fixed].to(DEVICE)

    init_xy_flat = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0)  # (2*n_det,)
    init_xy_flat = init_xy_flat.detach()
    prior_loc   = init_xy_flat.clone()
    prior_scale = torch.full_like(prior_loc, NUTS_PRIOR_SIGMA)

    def potential_fn(params):
        xy_flat = params["xy"]
        x_det = xy_flat[:N_DETECTORS]
        y_det = xy_flat[N_DETECTORS:]
        U_val, _, _ = utility_of_xy(x_det, y_det, primary_fixed, fnn, recon)
        log_prior = dist.Normal(prior_loc, prior_scale).log_prob(xy_flat).sum()
        log_density = U_val / NUTS_TEMPERATURE + log_prior
        return -log_density

    pyro.clear_param_store()
    pyro.set_rng_seed(SEED)

    nuts_kernel = NUTS(
        potential_fn=potential_fn,
        adapt_step_size=True,
        target_accept_prob=NUTS_TARGET_ACCEPT_PROB,
        max_tree_depth=NUTS_MAX_TREE_DEPTH,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=NUTS_NUM_SAMPLES,
        warmup_steps=NUTS_WARMUP,
        initial_params={"xy": init_xy_flat},
        disable_progbar=False,
    )

    print(f"[nuts] running  warmup={NUTS_WARMUP}  samples={NUTS_NUM_SAMPLES}  "
          f"target_accept={NUTS_TARGET_ACCEPT_PROB}  max_tree_depth={NUTS_MAX_TREE_DEPTH}")
    t0 = time.time()
    mcmc.run()
    dt = time.time() - t0

    samples = mcmc.get_samples()["xy"].detach()  # (NUTS_NUM_SAMPLES, 2*n_det)
    print(f"[nuts] sampled {samples.shape[0]} layouts in {dt:.1f}s")

    # Re-score each sample on the same fixed batch to pick the best.
    utilities = torch.empty(samples.shape[0], dtype=torch.float32)
    with torch.no_grad():
        for i, s in enumerate(samples):
            x_det = s[:N_DETECTORS]
            y_det = s[N_DETECTORS:]
            U_val, _, _ = utility_of_xy(x_det, y_det, primary_fixed, fnn, recon)
            utilities[i] = float(U_val.item())

    best_idx    = int(utilities.argmax())
    best_sample = samples[best_idx].cpu()
    best_x = best_sample[:N_DETECTORS]
    best_y = best_sample[N_DETECTORS:]

    print(f"[nuts] best sample idx={best_idx}  U={float(utilities[best_idx]):+.3f}  "
          f"median U={float(utilities.median()):+.3f}  "
          f"min={float(utilities.min()):+.3f}  max={float(utilities.max()):+.3f}")

    return dict(
        samples=samples.cpu(),
        utilities=utilities,
        best_x=best_x,
        best_y=best_y,
        best_idx=best_idx,
        best_u=float(utilities[best_idx]),
        wall_seconds=dt,
        primary_batch_size=NUTS_BATCH_PRIMARIES,
    )


def _plot_curves(adam_log, nuts_result, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ep = [e["epoch"] for e in adam_log]
        axes[0].plot(ep, [e["U"] for e in adam_log], color="C0", label="Adam U")
        axes[0].axhline(nuts_result["best_u"], color="C1", linestyle="--",
                        label=f"NUTS best U = {nuts_result['best_u']:.3f}")
        axes[0].set_xlabel("Adam epoch")
        axes[0].set_ylabel("U (composite)")
        axes[0].set_title("Adam warm-start")
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)

        axes[1].hist(nuts_result["utilities"].numpy(), bins=40,
                     color="C1", edgecolor="black", alpha=0.85)
        axes[1].axvline(nuts_result["best_u"], color="C0", linestyle="--",
                        label=f"best = {nuts_result['best_u']:.3f}")
        axes[1].set_xlabel("sample U")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"NUTS samples (N={nuts_result['samples'].shape[0]})")
        axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


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
        ax.scatter(x_nuts, y_nuts, s=30, c=COLOR_NUTS, label="NUTS best",
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
        _frame(ax, "NUTS best layout (1σ posterior ellipses)")

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


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
                    n_total_primaries: int):
    opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
    os.makedirs(opt_dir, exist_ok=True)
    print("-" * 72)
    print(f"[run] init_scheme={scheme}  ->  {opt_dir}")

    # Reseed so each scheme is independent and reproducible.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Phase 1: Adam warm-start
    x_adam, y_adam, x_init, y_init, adam_log = adam_warm_start(
        scheme, mountain, fnn, recon, primary_all, n_total_primaries,
    )

    # Phase 2: NUTS sampling around the Adam optimum
    nuts_result = nuts_sampling(
        x_adam, y_adam, fnn, recon, primary_all, n_total_primaries,
    )

    # Project the NUTS best onto the mountain (NUTS itself runs unconstrained
    # — the prior + Adam warm-start keep samples close to feasible layouts).
    x_nuts_proj, y_nuts_proj = mountain.project_to_mountain(
        nuts_result["best_x"], nuts_result["best_y"],
    )

    # Persist artifacts
    torch.save({"x": x_init, "y": y_init, "scheme": scheme},
               os.path.join(opt_dir, "layout_init.pt"))
    torch.save({"x": x_adam, "y": y_adam,
                "U": adam_log[-1]["U"] if adam_log else None,
                "best_U": max(e["U"] for e in adam_log) if adam_log else None},
               os.path.join(opt_dir, "layout_adam.pt"))
    torch.save({"x": x_nuts_proj, "y": y_nuts_proj,
                "x_raw": nuts_result["best_x"], "y_raw": nuts_result["best_y"],
                "U": nuts_result["best_u"],
                "sample_idx": nuts_result["best_idx"]},
               os.path.join(opt_dir, "layout_best.pt"))
    torch.save({"samples": nuts_result["samples"],
                "utilities": nuts_result["utilities"]},
               os.path.join(opt_dir, "nuts_samples.pt"))

    adam_best_U = max(e["U"] for e in adam_log) if adam_log else None
    with open(os.path.join(opt_dir, "optimize_log.json"), "w") as f:
        json.dump({
            "adam_log": adam_log,
            "adam_best_U": adam_best_U,
            "nuts_best_U": nuts_result["best_u"],
            "nuts_best_sample_idx": nuts_result["best_idx"],
            "nuts_wall_seconds": nuts_result["wall_seconds"],
            "nuts_utility_stats": dict(
                mean=float(nuts_result["utilities"].mean()),
                median=float(nuts_result["utilities"].median()),
                std=float(nuts_result["utilities"].std()),
                min=float(nuts_result["utilities"].min()),
                max=float(nuts_result["utilities"].max()),
            ),
            "config": dict(
                init_scheme=scheme,
                n_adam_epochs=N_ADAM_EPOCHS,
                primaries_per_step=PRIMARIES_PER_STEP,
                adam_lr=ADAM_LR, grad_clip=GRAD_CLIP,
                nuts_num_samples=NUTS_NUM_SAMPLES, nuts_warmup=NUTS_WARMUP,
                nuts_temperature=NUTS_TEMPERATURE,
                nuts_prior_sigma=NUTS_PRIOR_SIGMA,
                nuts_batch_primaries=NUTS_BATCH_PRIMARIES,
                nuts_target_accept_prob=NUTS_TARGET_ACCEPT_PROB,
                nuts_max_tree_depth=NUTS_MAX_TREE_DEPTH,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                seed=SEED,
            ),
        }, f, indent=2)

    _plot_curves(adam_log, nuts_result, os.path.join(opt_dir, "optimize_curves.png"))

    # Per-detector marginal posterior std from the NUTS samples — used as
    # error bars in the NUTS subplot of the layout figure.
    _samples = nuts_result["samples"]                            # (S, 2*n_det)
    _x_std = _samples[:, :N_DETECTORS].std(dim=0)                # (n_det,)
    _y_std = _samples[:, N_DETECTORS:].std(dim=0)
    _plot_layout(
        x_init.numpy(), y_init.numpy(),
        x_adam.numpy(), y_adam.numpy(),
        x_nuts_proj.numpy(), y_nuts_proj.numpy(),
        _x_std.numpy(), _y_std.numpy(),
        mountain,
        os.path.join(opt_dir, "layout_before_after.png"),
    )

    print(f"[done] scheme={scheme}  Adam best U={adam_best_U:+.3f}  "
          f"NUTS best U={nuts_result['best_u']:+.3f}  ({opt_dir})")
    return dict(scheme=scheme, adam_best_U=adam_best_U,
                nuts_best_U=nuts_result["best_u"], opt_dir=opt_dir)


def main():
    print("=" * 72)
    print("v6/04_optimize_nuts.py — Adam warm-start + NUTS sampling")
    print("=" * 72)
    print(f"device           : {DEVICE}")
    print(f"init schemes     : {INIT_SCHEMES}")
    print(f"Adam epochs      : {N_ADAM_EPOCHS}  (primaries/step={PRIMARIES_PER_STEP})")
    print(f"NUTS samples     : {NUTS_NUM_SAMPLES} (warmup={NUTS_WARMUP})")
    print(f"NUTS temp / σ    : T={NUTS_TEMPERATURE}  σ_prior={NUTS_PRIOR_SIGMA} m")

    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    n_total_primaries = int(primary_all.shape[0])
    print(f"[load] {n_total_primaries} primaries")

    fnn, recon = _load_models()

    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    results = []
    for scheme in INIT_SCHEMES:
        print()
        print("=" * 72)
        print(f"init scheme: {scheme}")
        print("=" * 72)
        results.append(_run_one_scheme(
            scheme, mountain, fnn, recon, primary_all, n_total_primaries,
        ))

    print()
    print("=" * 72)
    print("summary")
    print("=" * 72)
    for r in results:
        gain = r["nuts_best_U"] - r["adam_best_U"]
        print(f"  {r['scheme']:<8}  Adam={r['adam_best_U']:+.3f}  "
              f"NUTS={r['nuts_best_U']:+.3f}  Δ={gain:+.3f}  ->  {r['opt_dir']}")


if __name__ == "__main__":
    main()
