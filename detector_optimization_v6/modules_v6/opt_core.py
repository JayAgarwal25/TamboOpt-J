"""Shared non-plotting core for the v6/04 detector-layout optimizers.

Single home for everything the three 04 optimizers
(`04_optimize_lbfgs_ensemble.py`, `04_optimize_differential_evolution.py`,
`04_optimize_differential_evolution_pop.py`) used to each carry their own copy
of: the objective helpers (`primary_to_physical_labels`, `utility_of_xy`), the
ensemble bookkeeping (`assign`, `align_to_reference`), the gradient-turn
diagnostic (`consecutive_cos_distance`), model loading (`load_models`), and the
shared composite weights / thresholds / resolved geometry path.

The matching figure helpers live in `plots/opt_plotting.py` (plotting-only).

Note: `utility_of_xy` is defined WITHOUT `@torch.no_grad()` so the L-BFGS
optimizer can backprop through it; the gradient-free DE optimizers wrap their
score calls in `torch.no_grad()` themselves.
"""
import math
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .constants import (
    N_DETECTORS, FNN_FOLDER, RECON_FOLDER,
    GEOMETRY_PATH, GEOMETRY_PATH_RESOLVED, LOG_E_MIN, LOG_E_MAX,
)
from .dual_surrogate import load_dual_surrogate
from .reconstruction import build_recon_from_ckpt
# modules_v6/__init__ injected the v3 (`modules`) path on package import.
from modules.utility_functions import reconstructability, U_E, U_angle, U_PR

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # detector_optimization_v6/


# ── Shared config (identical across all three 04 optimizers) ──────────────────
# Utility composite weights — match 04_optimize.py
W_THETA = 1e2
W_PHI   = 1e2
W_E     = 2.5e2
W_PR    = 5e5
W_DIV   = 1e3

# Reconstructability thresholds.
#
# LAYOUT_THRESHOLD was 5e-2, which is not a physical hit criterion (a detector
# "seeing" 0.05 particles) and, worse, sits far inside the soft indicator's own
# transition width (~1/tau_layout = 0.2): sigmoid(5*(0 - 0.05)) = 0.44, so a
# COMPLETELY DARK detector contributed 0.44 to the count and 100 dark detectors
# floored the count at ~44/100. Measured counts never fell below 61 and r
# pinned at exactly 1.0000 for every event, making the whole term (and U_PR)
# a layout-independent constant with zero gradient.
#
# 1.0 = "detector saw at least one particle": dark -> 0.0067, floor 0.67/100.
# The detector count then spans p10=13 / p50=48 / p90=82, so the 10-detector
# minimum below is finally a binding constraint rather than a vacuous one.
LAYOUT_THRESHOLD      = 1.0
RECONSTRUCT_THRESHOLD = 10.0   # physical minimum detectors to reconstruct

# tau_reconstruct=5.0 (the upstream default) makes r a step function: it swings
# 0->1 within ~+-1 detector, so almost no event sits in the transition and the
# term hands back no gradient. 0.2 spreads the transition over ~20 detectors
# (n=10 -> r=0.5, n=20 -> 0.88, n=30 -> 0.98), giving r std ~0.20 across the
# population instead of ~0.
TAU_LAYOUT      = 5.0
TAU_RECONSTRUCT = 0.2

# Soft caps on the per-event 1/(err^2 + eps) reward, sized from the measured
# reward distribution on the trained nets (20k events, grid layout) rather than
# from the eps ceiling alone. Rule: keep the cap well above the term's MEDIAN
# reward so the bulk still discriminates, low enough to pull in the tail.
#
#   term   median  hard ceiling  top-5% share of U   -> with cap below
#   theta   218      1000            14.5%              10.3%
#   phi       7.8    1000            49.2%              14.1%
#   E        27.6     100            12.9%               9.3%
#
# phi is the genuinely concentrated one (the recon is poor at phi: median error
# 0.36 rad), so it gets a much tighter cap; its median is far below 50 so the
# bulk is untouched. theta/phi share U_angle but need different caps.
CAP_THETA = 500.0
CAP_PHI   = 50.0
CAP_E     = 80.0

# GEOMETRY_PATH_RESOLVED is centralized in constants (mesh-agnostic: local copy of
# the configured mesh, else the absolute path) and re-exported here for callers
# (04 DE / DE-pop) that import it from opt_core.


# ── Objective helpers ─────────────────────────────────────────────────────────
def primary_to_physical_labels(primary: torch.Tensor):
    """(B, >=4) -> (E_GeV, θ_rad, φ_rad). Inverse of `encode_primary`."""
    dir_x = primary[:, 0]
    dir_y = primary[:, 1]
    dir_z = primary[:, 2].clamp(-1.0, 1.0)
    log_e_norm = primary[:, 3]
    log_e = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    E_gev = torch.pow(10.0, log_e)
    theta = torch.arccos(dir_z)
    phi   = torch.atan2(dir_y, dir_x)
    two_pi = 2.0 * math.pi
    phi = torch.where(phi < 0, phi + two_pi, phi)
    return E_gev, theta, phi


# Off-mesh penalty (see offmesh_penalty). Weight is in units of U: one detector
# sitting a full max_gap BEYOND the snap radius costs w/n_det, so at w=100 with
# 100 detectors that is 1.0 U each — comfortably more than the ~1-2 U a chunk
# gains by overfitting its batch, which is the trade the optimizer was making.
OFFMESH_PENALTY_W = 100.0


# Onset fraction of the snap radius. NOT 1.0, which was the first attempt
# and measurably failed: a penalty that is flat inside the radius has zero
# gradient there, so the only equilibrium is where its OUTWARD-growing
# gradient finally matches the utility's outward pull — necessarily beyond
# the radius (measured ~21 m beyond at w=100). Starting the wall inside
# puts that balance point back in-band. The cost is that U is no longer
# bit-identical to earlier runs for detectors sitting in the outer quarter
# of the band; the alternative is a barrier that does not hold.
PENALTY_ONSET_FRAC = 0.75

# Excess (in units of r0) beyond which the penalty grows LINEARLY instead of
# quadratically. Unbounded quadratic growth was the second failure: a
# strong-Wolfe probe landing 400 km out produced a gradient ~4000x anything
# physical, and although the probe was rejected, the curvature pair it fed into
# L-BFGS's inverse-Hessian approximation made the following directions wilder
# still (worst excursion grew from 797 m to 420,658 m). Beyond this the
# gradient is constant, so a wild probe is expensive but not corrupting.
PENALTY_LINEAR_AT = 1.0


def offmesh_penalty(x_det: torch.Tensor, y_det: torch.Tensor,
                    mesh_en: torch.Tensor, r0: float) -> torch.Tensor:
    """Mean penalised excess distance beyond `r0` of the nearest mesh centroid.

    Huber-shaped in the normalised excess u = (d - r0)/r0: u^2 while u <= a,
    then a linear continuation 2a*u - a^2 (value and slope both continuous at
    u = a). Quadratic near the wall so it is smooth where the optimizer
    actually works, linear far away so the far field cannot dominate.

    Why this exists: nothing in U knew the mountain existed. The only geometry
    was `project_to_mountain_ne`, applied under no_grad AFTER the optimizer
    step, and in L-BFGS only once per sweep chunk. So for a whole chunk the
    optimizer ran free, took the layout a median 29 m / p99 187 m / max 1272 m
    off the mesh — past the 91.3 m radius where the stage-1 training layouts
    were snapped, i.e. where the surrogate has no data — and happily hill-
    climbed the extrapolation, reporting U 36.2 -> 37.7 out there. The snap at
    the chunk end then dragged it back and U collapsed to 16.18 (one run
    returned 100 detectors on 11 distinct points, 28 stacked on one).

    This makes the boundary part of the objective, so the optimizer feels it
    DURING the chunk instead of being caught at the end. Normalised by r0 and
    averaged over detectors, so the weight means the same thing at any mesh
    scale or detector count.

    `r0` is the ONSET radius, which callers set to PENALTY_ONSET_FRAC * max_gap
    — inside the snap radius on purpose (see that constant). At the default
    0.75 the wall starts at 68.5 m and bites at the 91.3 m snap radius with a
    gradient of 9.7e-3 per metre, ~20x the utility's measured outward pull of
    ~5e-4, putting the balance point at ~70 m — in-band, which is the whole
    point. A layout in the inner three quarters of the band still scores
    exactly as it did before.
    """
    d2 = ((x_det[:, None] - mesh_en[None, :, 0]) ** 2
          + (y_det[:, None] - mesh_en[None, :, 1]) ** 2)
    d = d2.min(dim=1).values.clamp_min(1e-12).sqrt()
    u = (d - r0).clamp_min(0.0) / r0
    a = PENALTY_LINEAR_AT
    quad = u.clamp_max(a).pow(2)                  # u^2 up to the knee
    lin = 2.0 * a * (u - a).clamp_min(0.0)        # constant slope past it
    return (quad + lin).mean()


def utility_of_xy(x_det: torch.Tensor,
                  y_det: torch.Tensor,
                  primary_batch: torch.Tensor,
                  fnn,
                  recon,
                  mesh_en: torch.Tensor = None,
                  penalty_w: float = 0.0,
                  penalty_r0: float = None):
    """Composite U for an (East, North) layout against a primary batch.

    Detector coords are (East, North) to match the ENU h5 convention — but this
    function is order-agnostic: it feeds the pair straight to the FNN + recon,
    both trained on the same `xy` column order, so consistency (not the physical
    axis meaning) is all that matters here.

    `fnn` is the dual-species wrapper: both per-species surrogates are evaluated
    with the same primary + layout and physically combined, so the backprop into
    (x_det, y_det) flows through BOTH models. Mirrors the inner loop of
    `_run_optimization` in 04_optimize.py (the U_PR term is computed but
    deliberately omitted from the composite, matching production).

    NOT decorated with `@torch.no_grad()` so the L-BFGS optimizer can
    differentiate it; the gradient-free DE optimizers call it inside their own
    `no_grad` block.

    `mesh_en` (n_centroids, 2) + `penalty_w` > 0 subtract the differentiable
    off-mesh penalty (see offmesh_penalty), putting the mountain boundary into
    the objective itself. OFF by default, so every existing caller — including
    plots/eval_true_utility.py, which relies on this function being unmodified
    to guarantee it scores exactly what the optimizer scored — gets the same
    number as before. `penalty_r0` defaults to the caller's max_gap.

    The returned U is the PENALISED objective, i.e. what the optimizer should
    select on; the raw utility and the penalty are both in `parts` so the logs
    can separate them. With r0 = max_gap a converged in-band layout has penalty
    exactly 0, so reported U stays comparable to earlier runs."""
    B = primary_batch.shape[0]
    xy_per_det = torch.stack([x_det, y_det], dim=-1)                       # (n_det, 2)
    xy_batch   = xy_per_det.unsqueeze(0).expand(B, -1, -1)                 # (B, n_det, 2)

    # Deterministic mean prediction. The layout optimizers (Adam warm-start's
    # argmax-over-epochs best-tracking, L-BFGS's strong_wolfe line search, DE's
    # fitness comparison) all select on this value, so a fresh stochastic
    # sample per call would let them cherry-pick lucky noise draws instead of
    # real improvement — verified: every Adam chain's "best" collapsed by a
    # uniform ~10 points once refined/re-evaluated when this called
    # forward_sample(). Sampling stays confined to stage 3 (recon training).
    pred_ET = fnn(primary_batch, xy_batch)
    E_pred_det = pred_ET[..., 0]
    T_pred_det = pred_ET[..., 1]

    recon_feats = torch.stack(
        [xy_batch[..., 0], xy_batch[..., 1], E_pred_det, T_pred_det],
        dim=-1,
    )                                                                      # (B, n_det, 4)
    pred = recon(recon_feats)                                              # (B, 4); DeepSets recon takes (B, n_det, 4)
    E_pred_phys, theta_pred, phi_pred = primary_to_physical_labels(pred)
    E_pred_phys = E_pred_phys.clamp(min=1.0)

    E_true, theta_true, phi_true = primary_to_physical_labels(primary_batch)

    r = reconstructability(
        torch.expm1(E_pred_det),
        layout_threshold=LAYOUT_THRESHOLD,
        tau_layout=TAU_LAYOUT,
        reconstruct_threshold=RECONSTRUCT_THRESHOLD,
        tau_reconstruct=TAU_RECONSTRUCT,
    )
    u_theta = U_angle(theta_pred, theta_true, r, cap=CAP_THETA)
    u_phi   = U_angle(phi_pred,   phi_true,   r, cap=CAP_PHI)
    u_e     = U_E    (E_pred_phys, E_true,    r, cap=CAP_E)
    u_pr    = U_PR(r)
    U = (W_THETA * u_theta + W_PHI * u_phi + W_E * u_e) / W_DIV
    parts = dict(u_theta=W_THETA * u_theta / W_DIV, u_phi=W_PHI * u_phi / W_DIV,
                 u_e=W_E * u_e / W_DIV, u_pr=W_PR * u_pr / W_DIV)
    if mesh_en is not None and penalty_w > 0.0:
        r0 = penalty_r0 if penalty_r0 else 1.0
        pen = penalty_w * offmesh_penalty(x_det, y_det, mesh_en, r0)
        parts["u_raw"] = U          # utility before the boundary cost
        parts["u_offmesh"] = -pen
        U = U - pen
    return U, r, parts


# ── Ensemble bookkeeping ──────────────────────────────────────────────────────
def assign(cost: np.ndarray) -> np.ndarray:
    """One-to-one assignment minimizing total cost (Hungarian)."""
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
        diff = ref[:, None, :] - L[None, :, :]      # (n_det, n_det, 2)
        cost = (diff * diff).sum(axis=-1)           # (n_det, n_det)
        col = assign(cost)
        aligned[k] = L[col]
        perms[k] = col
    return aligned, perms


def consecutive_cos_distance(grad_hist, window: int = 1) -> np.ndarray:
    """Per-step cosine distance 1 - cos(g_t, g_{t-1}) between consecutive gradient
    vectors, optionally W-step vector-averaged first to cancel zero-mean minibatch
    noise before the (nonlinear) cosine. window=1 → raw, no averaging.

    `grad_hist` is a sequence of flat gradient vectors (one per optimizer step).
    Returns a 1-D array of length max(0, len(series) - 1)."""
    if grad_hist is None or len(grad_hist) < 2:
        return np.zeros(0)
    G = np.asarray([np.asarray(g, dtype=np.float64).reshape(-1) for g in grad_hist])
    if window and window > 1:
        # Vector-average over a sliding window (valid mode) before the cosine.
        kernel = np.ones(window) / window
        G = np.stack([np.convolve(G[:, j], kernel, mode="valid")
                      for j in range(G.shape[1])], axis=1)
        if G.shape[0] < 2:
            return np.zeros(0)
    a = G[1:]
    b = G[:-1]
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den > 0, den, 1.0)
    return 1.0 - num / den


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models(device, fnn_folder=None, recon_dir=None):
    """Frozen dual-species surrogate + DeepSets recon from 03_train_recon_deepsets.py.

    The dual wrapper combines fnn_electron.pt + fnn_muon.pt per event (frozen,
    eval); gradients flow through both branches. `build_recon_from_ckpt` loads
    whichever recon the checkpoint declares (DeepSets here, consuming
    (B, n_det, 4) per-detector features: x, y, and a stochastic sample of
    E/T drawn from the surrogate's predicted distribution), applies its
    normalization, and freezes
    it. Defaults: FNN_FOLDER and RECON_FOLDER + "_deepsets"."""
    fnn_folder = fnn_folder or FNN_FOLDER
    recon_dir  = recon_dir  or (RECON_FOLDER + "_deepsets")
    fnn = load_dual_surrogate(fnn_folder, device)

    recon_ckpt = torch.load(os.path.join(recon_dir, "recon.pt"),
                            map_location=device, weights_only=False)
    recon = build_recon_from_ckpt(recon_ckpt, N_DETECTORS, device)
    print(f"[load] recon.pt  model={recon_ckpt.get('config', {}).get('model_type', 'mlp')}  "
          f"epoch={recon_ckpt.get('epoch', '?')}  val={recon_ckpt.get('val_total', '?')}  <- {recon_dir}")
    return fnn, recon
