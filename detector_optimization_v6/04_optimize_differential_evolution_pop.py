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
from modules_v6.tr_surface_map_ne import SurfaceUpMap
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


def _assign(cost: np.ndarray) -> np.ndarray:
    """One-to-one assignment minimizing total cost (Hungarian)."""
    _, col = linear_sum_assignment(cost)
    return col


def align_to_reference(layouts_xy: np.ndarray, ref_idx: int):
    """Permutation-invariant alignment of the population layouts to a reference.

    layouts_xy : (K, n_det, 2). Matches each member's detectors to the reference
    by minimum total squared distance, then reorders so column i of every member
    is the same physical position group. Returns (aligned (K, n_det, 2),
    perms (K, n_det))."""
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


def _plot_curves(de_log, path: str):
    """Best-U over DE generations (single trajectory)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        u = [e["U"] for e in de_log]
        if not u:
            print("[plot] curves skipped (empty log)")
            return
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.plot(np.arange(1, len(u) + 1), u, color="C0", linewidth=1.2,
                label=f"best={max(u):.2f}")
        ax.set_xlabel("DE generation")
        ax.set_ylabel("U (composite)")
        ax.set_title(f"Differential evolution: best-U per generation (pop={POP_SIZE})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] curves skipped ({exc!r})")


def _plot_utility_components(de_log, path: str):
    """Weighted utility sub-parts (θ, φ, E) + overall U over the DE generations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not de_log:
            print("[plot] utility components skipped (empty log)")
            return
        x = np.arange(1, len(de_log) + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, key, col in [("θ", "u_theta", "C0"), ("φ", "u_phi", "C1"), ("E", "u_e", "C2")]:
            ax.plot(x, [e[key] for e in de_log], color=col, linewidth=1.0,
                    alpha=0.85, label=label)
        ax.plot(x, [e["U"] for e in de_log], color="black", linewidth=1.8,
                label="U (overall)")
        ax.set_xlabel("DE generation")
        ax.set_ylabel("utility")
        ax.set_title("utility decomposition (weighted θ/φ/E sub-parts + overall U; DE generations)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] utility components skipped ({exc!r})")


@torch.no_grad()
def _project_ne_to_up(surface, north, east):
    """Map detector (North, East) → Up via the differentiable mountain surface
    Up = g(North, East) (modules_v6.tr_surface_map_ne.SurfaceUpMap).

    DE optimises in the North–East plane, so its layouts carry East, not Up. To
    draw them in the SAME (North, Up) cross section the L-BFGS ensemble uses (whose
    optimiser is native to North–Up), project each detector's East through the
    surface to recover the height it sits at. Returns a numpy array shaped like
    `north`."""
    dev = surface.grid_up.device
    shp = np.asarray(north).shape
    n = torch.as_tensor(np.asarray(north).reshape(-1), dtype=torch.float32, device=dev)
    e = torch.as_tensor(np.asarray(east ).reshape(-1), dtype=torch.float32, device=dev)
    return surface(n, e).detach().cpu().numpy().reshape(shp)


def _plot_ensemble(aligned_xy: np.ndarray,
                   mean_xy: np.ndarray,
                   std_xy: np.ndarray,
                   best_x, best_y,
                   mountain, path: str, surface=None):
    """Mountain top-down ensemble: every population member (faint) + per-group
    mean + 1σ ellipses.

    With `surface` (a SurfaceUpMap) the detector East is projected to Up =
    g(North, East) and the plot is the (North, Up) cross section — the same view
    as the L-BFGS ensemble; mean/std are recomputed in that plane. Without it the
    native (North, East) plane is drawn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection

        if surface is not None:
            up = _project_ne_to_up(surface, aligned_xy[..., 0], aligned_xy[..., 1])
            aligned_xy = np.stack([aligned_xy[..., 0], up], axis=-1)   # (K, n_det, 2)=(N,Up)
            mean_xy = aligned_xy.mean(axis=0)
            std_xy  = aligned_xy.std(axis=0)
            best_y  = _project_ne_to_up(surface, np.asarray(best_x), np.asarray(best_y))
            mtn_y, ylab, ylet = mountain.centroids_NUE[:, 1], "Up [m]", "Up"
        else:
            mtn_y, ylab, ylet = mountain.centroids_NUE[:, 2], "East [m]", "E"

        K = aligned_xy.shape[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(mountain.centroids_NUE[:, 0], mtn_y,
                   s=2, c="lightgray", alpha=0.6, label="mountain")

        colors = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))
        for k in range(K):
            ax.scatter(aligned_xy[k, :, 0], aligned_xy[k, :, 1], s=8,
                       color=colors[k % 10], alpha=0.35, edgecolors="none",
                       label=f"member {k}" if k < 10 else None)

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
                         f"σ̄{ylet}={std_xy[:,1].mean():.1f} m)")

        ax.set_xlabel("North [m]"); ax.set_ylabel(ylab)
        ax.set_aspect("equal")
        ax.set_title(f"DE population ensemble (pop={K}) — aligned best + 1σ ellipses")
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
                          bins: int = 60, surface=None):
    """Mountain top-down 2D density of detector placements across the population.

    With `surface` the detector East is projected to Up = g(North, East) so the
    plot is the (North, Up) cross section — matching the L-BFGS ensemble heatmap;
    without it the native (North, East) plane is drawn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if surface is not None:
            up = _project_ne_to_up(surface, aligned_xy[..., 0], aligned_xy[..., 1])
            aligned_xy = np.stack([aligned_xy[..., 0], up], axis=-1)
            best_y = _project_ne_to_up(surface, np.asarray(best_x), np.asarray(best_y))
            mtn_col, ylab = 1, "Up [m]"
        else:
            mtn_col, ylab = 2, "East [m]"

        K, n_det, _ = aligned_xy.shape
        pts = aligned_xy.reshape(-1, 2)                          # (K*n_det, 2)

        cen = getattr(mountain, "centroids_NUE", None)
        if cen is not None:
            allx = np.concatenate([cen[:, 0], pts[:, 0]])
            ally = np.concatenate([cen[:, mtn_col], pts[:, 1]])
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
            occ, _, _ = np.histogram2d(cen[:, 0], cen[:, mtn_col], bins=bins, range=rng)
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
        cbar.set_label("detector density (count per member per cell)")

        ax.scatter(np.asarray(best_x), np.asarray(best_y), s=22, c="cyan",
                   edgecolors="black", linewidths=0.4, alpha=0.95, zorder=3,
                   label="best-U layout")
        ax.set_xlabel("North [m]"); ax.set_ylabel(ylab)
        ax.set_title(f"detector placement density (pop={K}, {bins}×{bins} bins) + best-U layout")
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

    fnn, recon = _load_models()

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

    _plot_curves(de_log, os.path.join(opt_dir, "optimize_curves.png"))
    _plot_utility_components(de_log, os.path.join(opt_dir, "utility_components.png"))
    _plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                   mountain, os.path.join(opt_dir, "layout_ensemble.png"), surface=surface)
    _plot_density_heatmap(aligned, best_x, best_y,
                   mountain, os.path.join(opt_dir, "layout_density.png"), surface=surface)

    print()
    print("=" * 72)
    print(f"[done] best U={best_U:+.3f}  "
          f"σ̄=({std_xy[:,0].mean():.1f}, {std_xy[:,1].mean():.1f}) m  ({opt_dir})")
    print("=" * 72)


if __name__ == "__main__":
    main()
