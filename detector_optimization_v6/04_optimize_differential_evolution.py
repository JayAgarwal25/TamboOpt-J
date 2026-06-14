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
from scipy.optimize import linear_sum_assignment, differential_evolution

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.dual_surrogate import load_dual_surrogate
from modules_v6.reconstruction import Reconstruction
from modules_v6.tr_geometry_ne import (
    _ne_max_gap, project_to_mountain_ne, sample_initial_layout_ne,
)
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER, OPT_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules.utility_functions   import reconstructability, U_E, U_angle, U_PR
from modules_v4.tr_geometry      import load_tr_mountain


# ── Config ───────────────────────────────────────────────────────────────────
INIT_SCHEMES         = ("grid", "center")
RUN_COMBINED         = True
COMBINED_SCHEME_NAME = "combined"
OPT_DIR_TEMPLATE     = OPT_FOLDER + "_de_ensemble_{scheme}"

# K perturbed restarts per scheme.
N_CHAINS            = 15
INIT_OVERDISP_SIGMA = 1000.0  # metres — per-restart init spread around scheme init

# Differential evolution (replaces the Adam warm-start + L-BFGS refine)
DE_MAXITER          = 100
DE_POPSIZE          = 4       # population = popsize × (2·n_det) candidates / generation
DE_TOL              = 1e-4
DE_MUTATION         = (0.5, 1.0)
DE_RECOMBINATION    = 0.7
DE_BATCH_PRIMARIES  = 512     # FIXED batch → deterministic objective for the search

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

# constants.GEOMETRY_PATH may be stale; prefer a local copy, then the new TAMBOSim path.
GEOMETRY_PATH_RESOLVED = next(
    (p for p in (
        os.path.join(_HERE, "colca_valley.h5"),
        "/n/home05/zdimitrov/tambo/TAMBOSim/resources/geometry/colca_valley.h5",
        GEOMETRY_PATH,
    ) if os.path.exists(p)),
    GEOMETRY_PATH,
)


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


@torch.no_grad()
def utility_of_xy(x_det: torch.Tensor,
                  y_det: torch.Tensor,
                  primary_batch: torch.Tensor,
                  fnn,
                  recon: Reconstruction):
    """Composite U for a (North, East) layout against a primary batch.

    Mirrors `utility_of_xy` in 04_optimize_lbfgs_ensemble.py (same objective, the
    U_PR term computed but omitted from the composite). Gradient-free here, so it
    runs under no_grad; `x_det`/`y_det` are (North, East)."""
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
              recon: Reconstruction,
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


def _assign(cost: np.ndarray) -> np.ndarray:
    """One-to-one assignment minimizing total cost (Hungarian)."""
    _, col = linear_sum_assignment(cost)
    return col


def align_to_reference(layouts_xy: np.ndarray, ref_idx: int):
    """Permutation-invariant alignment of K layouts to a reference.

    layouts_xy : (K, n_det, 2). Matches each run's detectors to the reference by
    minimum total squared distance, then reorders so column i of every run is the
    same physical position group. Returns (aligned (K, n_det, 2), perms (K, n_det))."""
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
        diff = ref[:, None, :] - L[None, :, :]      # (n_det, n_det, 2)
        cost = (diff * diff).sum(axis=-1)           # (n_det, n_det)
        col = _assign(cost)
        aligned[k] = L[col]
        perms[k] = col
    return aligned, perms


def _plot_curves(start_logs, de_logs, path: str):
    """Per-run best-U over DE generations (one line per run). DE analogue of the
    combined Adam→L-BFGS U-trajectory panel; there is a single optimiser, so no
    phase divider and no gradient-cosine panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(de_logs), 1)
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, K))
        for k in range(K):
            lg = de_logs[k] if k < len(de_logs) else []
            u = [e["U"] for e in lg]
            if u:
                best = max(u)
                ax.plot(np.arange(1, len(u) + 1), u, color=colors[k], alpha=0.85,
                        linewidth=1.0, label=f"chain {k}  best={best:.2f}")
        ax.set_xlabel("DE generation")
        ax.set_ylabel("U (composite)")
        ax.set_title(f"Differential evolution: best-U per generation, K={K}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] curves skipped ({exc!r})")


def _plot_utility_components(start_logs, de_logs, path: str):
    """One subfigure per chain: the weighted utility sub-parts (θ, φ, E) over the
    DE generations plus the overall U (bold black). Mirrors the L-BFGS-ensemble
    figure (Adam/L-BFGS phases collapse to a single DE trajectory)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(de_logs), 1)
        ncol = min(K, 5)
        nrow = math.ceil(K / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow),
                                 squeeze=False, sharex=False)
        axes_flat = axes.flatten()
        parts = [("θ", "u_theta", "C0"), ("φ", "u_phi", "C1"), ("E", "u_e", "C2")]

        for k in range(K):
            ax = axes_flat[k]
            lg = de_logs[k] if k < len(de_logs) else []
            x = np.arange(1, len(lg) + 1)
            for label, key, col in parts:
                if lg:
                    ax.plot(x, [e[key] for e in lg], color=col, linewidth=1.0,
                            alpha=0.85, label=label)
            if lg:
                ax.plot(x, [e["U"] for e in lg], color="black", linewidth=1.8,
                        label="U (overall)")
            ax.set_title(f"chain {k}", fontsize=10)
            ax.set_xlabel("DE generation"); ax.set_ylabel("utility")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)

        for j in range(K, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle("per-chain utility decomposition "
                     "(weighted θ/φ/E sub-parts + overall U; DE generations)", fontsize=12)
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
    """Mountain top-down (North, East): every aligned run (faint) + per-group mean
    + 1σ ellipses. Mirrors the L-BFGS-ensemble plot with East on the y-axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection

        K = aligned_xy.shape[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(mountain.centroids_NUE[:, 0], mountain.centroids_NUE[:, 2],
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
                   label=f"best  (σ̄N={std_xy[:,0].mean():.1f} m, "
                         f"σ̄E={std_xy[:,1].mean():.1f} m)")

        ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]")
        ax.set_aspect("equal")
        ax.set_title(f"DE ensemble (K={K}) — aligned best + 1σ ellipses")
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
    """Mountain top-down (North, East) 2D density of detector placements across the
    ensemble. Mirrors the L-BFGS-ensemble heatmap with East on the y-axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K, n_det, _ = aligned_xy.shape
        pts = aligned_xy.reshape(-1, 2)                          # (K*n_det, 2)

        cen = getattr(mountain, "centroids_NUE", None)
        if cen is not None:
            allx = np.concatenate([cen[:, 0], pts[:, 0]])
            ally = np.concatenate([cen[:, 2], pts[:, 1]])
        else:
            allx, ally = pts[:, 0], pts[:, 1]
        extent = [float(allx.min()), float(allx.max()),
                  float(ally.min()), float(ally.max())]

        rng = [[extent[0], extent[1]], [extent[2], extent[3]]]
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=rng)
        H = H / max(K, 1)
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=1.0)
        except Exception:
            pass

        if cen is not None:
            occ, _, _ = np.histogram2d(cen[:, 0], cen[:, 2], bins=bins, range=rng)
            det_occ, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=rng)
            mask = (occ > 0) | (det_occ > 0)
            try:
                from scipy.ndimage import (binary_dilation, binary_fill_holes,
                                           binary_erosion)
                mask = binary_dilation(mask, iterations=2)
                mask = binary_fill_holes(mask)
                mask = binary_erosion(mask, iterations=1, border_value=1)
            except Exception:
                pass
            H = np.ma.masked_array(H, mask=~mask)

        data_ar = (extent[3] - extent[2]) / (extent[1] - extent[0])
        fig_w = 14.0
        fig, ax = plt.subplots(figsize=(fig_w, max(fig_w * data_ar + 1.2, 3.0)))
        cmap = plt.cm.magma.copy()
        cmap.set_bad(alpha=0.0)
        im = ax.imshow(H.T, origin="lower", extent=extent, aspect="equal",
                       cmap=cmap, interpolation="bilinear", zorder=0)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("detector density (count per run per cell)")

        ax.scatter(np.asarray(best_x), np.asarray(best_y), s=22, c="cyan",
                   edgecolors="black", linewidths=0.4, alpha=0.95, zorder=3,
                   label="best-U layout")
        ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]")
        ax.set_title(f"detector placement density (K={K} runs, {bins}×{bins} bins) + best-U layout")
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] density heatmap skipped ({exc!r})")


def _load_models():
    """Frozen dual-species surrogate + recon, matching 04_optimize_lbfgs_ensemble.py.
    The wrapper combines fnn_electron.pt + fnn_muon.pt per event (counts add,
    times average count-weighted)."""
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
    print(f"[load] recon.pt  val={recon_ckpt.get('val_total', '?')}")
    return fnn, recon


def _run_one_scheme(scheme: str,
                    mountain,
                    fnn,
                    recon: Reconstruction,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    per_source):
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

    _plot_curves(all_start_logs, de_logs,
                 os.path.join(opt_dir, "optimize_curves.png"))
    _plot_utility_components(all_start_logs, de_logs,
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

    fnn, recon = _load_models()

    mountain = load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

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
