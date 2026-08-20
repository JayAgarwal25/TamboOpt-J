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
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .constants import (
    N_DETECTORS, FNN_FOLDER, RECON_FOLDER,
    GEOMETRY_PATH, GEOMETRY_PATH_RESOLVED, LOG_E_MIN, LOG_E_MAX,
    SIGMA_SPATIAL,
)
from .dual_surrogate import load_dual_surrogate
from .reconstruction import build_recon_from_ckpt
# modules_v6/__init__ injected the v3 (`modules`) path on package import.
from modules_v6.utility_functions import reconstructability, U_E, U_angle, U_PR

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root


# ── Shared config (identical across all three 04 optimizers) ──────────────────
# Utility composite weights — match 04_optimize.py
W_THETA = 1e2
W_PHI   = 1e2
W_E     = 2.5e2
W_PR    = 5e5
W_DIV   = 1e3

# Reconstructability thresholds.
# 1.0 = "saw at least one particle". Do NOT lower it: at the old 5e-2 a DARK
# detector scored sigmoid(5*(0-0.05)) = 0.44, flooring the count at ~44/100, so
# r pinned at 1.0000 for every event and the term had zero gradient. At 1.0 a
# dark detector gives 0.0067 and the count spans p10=13 / p50=48 / p90=82.
LAYOUT_THRESHOLD      = 1.0
RECONSTRUCT_THRESHOLD = 10.0   # physical minimum detectors to reconstruct

# tau_reconstruct=5.0 (the upstream default) makes r a step function: it swings
# 0->1 within ~+-1 detector, so almost no event sits in the transition and the
# term hands back no gradient. 0.2 spreads the transition over ~20 detectors
# (n=10 -> r=0.5, n=20 -> 0.88, n=30 -> 0.98), giving r std ~0.20 across the
# population instead of ~0.
TAU_LAYOUT      = 5.0
TAU_RECONSTRUCT = 0.2

# Soft caps on the per-event 1/(err^2 + eps) reward. Sized from the measured
# reward distribution (20k events, grid layout): well above each term's MEDIAN so
# the bulk still discriminates, low enough to pull in the tail.
#   term   median   top-5% share of U   -> with cap
#   theta   218          14.5%             10.3%
#   phi       7.8        49.2%             14.1%   (recon is poor at phi: median
#   E        27.6        12.9%              9.3%    error 0.36 rad -> tighter cap)
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

    Huber-shaped in the normalised excess u = (d - r0)/r0: u^2 up to u = a, then
    the linear continuation 2a*u - a^2 (value and slope continuous at the knee).
    Quadratic where the optimizer works, linear far away so the far field cannot
    dominate. Normalised by r0 and averaged over detectors, so the weight means
    the same thing at any mesh scale or detector count.

    Why it exists: U itself knew nothing about the mountain. `project_to_mountain_ne`
    ran under no_grad AFTER the step (once per L-BFGS chunk), so the optimizer was
    free to hill-climb the surrogate's extrapolation a median 29 m / p99 187 m /
    max 1272 m off-mesh — past the 91.3 m radius where training layouts were
    snapped — reporting U 36.2 -> 37.7 out there. The end-of-chunk snap then
    collapsed U to 16.18 (one run: 100 detectors on 11 distinct points).

    `r0` is the ONSET radius (callers pass PENALTY_ONSET_FRAC * max_gap), inside
    the snap radius on purpose. At 0.75 the wall starts at 68.5 m and at the
    91.3 m snap radius pushes back with 9.7e-3 per metre, ~20x the utility's
    outward pull of ~5e-4 — balance point ~70 m, in-band. Layouts in the inner
    three quarters of the band score exactly as before.
    """
    d2 = ((x_det[:, None] - mesh_en[None, :, 0]) ** 2
          + (y_det[:, None] - mesh_en[None, :, 1]) ** 2)
    d = d2.min(dim=1).values.clamp_min(1e-12).sqrt()
    u = (d - r0).clamp_min(0.0) / r0
    a = PENALTY_LINEAR_AT
    quad = u.clamp_max(a).pow(2)                  # u^2 up to the knee
    lin = 2.0 * a * (u - a).clamp_min(0.0)        # constant slope past it
    return (quad + lin).mean()


def _apply_offmesh_penalty(U: torch.Tensor, parts: dict,
                           x_det: torch.Tensor, y_det: torch.Tensor,
                           mesh_en: Optional[torch.Tensor],
                           penalty_w: float,
                           penalty_r0: Optional[float]) -> torch.Tensor:
    """Subtract the off-mesh penalty from `U`, recording both halves in `parts`.

    A no-op returning `U` unchanged unless a mesh AND a positive weight are both
    supplied, so callers that never pass them keep bit-identical numbers. Adds
    "u_raw" (utility before the boundary cost) and "u_offmesh" to `parts` in
    place and returns the PENALISED objective.
    """
    if mesh_en is None or penalty_w <= 0.0:
        return U
    r0 = penalty_r0 if penalty_r0 else 1.0
    pen = penalty_w * offmesh_penalty(x_det, y_det, mesh_en, r0)
    parts["u_raw"] = U          # utility before the boundary cost
    parts["u_offmesh"] = -pen
    return U - pen


def _surrogate_predict(x_det: torch.Tensor,
                       y_det: torch.Tensor,
                       primary_batch: torch.Tensor,
                       fnn):
    """Broadcast one layout over a primary batch and run the dual surrogate.

    Returns (xy_batch (B, n_det, 2), pred_ET (B, n_det, 2)) — the surrogate's
    log-compressed (E, T) channels. Like its callers this is deliberately NOT
    `@torch.no_grad()`-decorated and detaches nothing, so gradients reach
    (x_det, y_det) through both per-species branches.
    """
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
    return xy_batch, pred_ET


def _predict_primary(x_det: torch.Tensor,
                     y_det: torch.Tensor,
                     primary_batch: torch.Tensor,
                     fnn,
                     recon):
    """Layout -> dual surrogate -> recon -> physical primary labels.

    Returns (E_pred_det, E_pred_phys, theta_pred, phi_pred): the surrogate's
    per-detector E channel (log1p counts, what `reconstructability` expm1s) plus
    the reconstructed primary decoded by `primary_to_physical_labels`.
    """
    xy_batch, pred_ET = _surrogate_predict(x_det, y_det, primary_batch, fnn)
    E_pred_det = pred_ET[..., 0]
    T_pred_det = pred_ET[..., 1]

    recon_feats = torch.stack(
        [xy_batch[..., 0], xy_batch[..., 1], E_pred_det, T_pred_det],
        dim=-1,
    )                                                                      # (B, n_det, 4)
    pred = recon(recon_feats)                                              # (B, 4); DeepSets recon takes (B, n_det, 4)
    E_pred_phys, theta_pred, phi_pred = primary_to_physical_labels(pred)
    E_pred_phys = E_pred_phys.clamp(min=1.0)
    return E_pred_det, E_pred_phys, theta_pred, phi_pred


def utility_of_xy(x_det: torch.Tensor,
                  y_det: torch.Tensor,
                  primary_batch: torch.Tensor,
                  fnn,
                  recon,
                  mesh_en: torch.Tensor = None,
                  penalty_w: float = 0.0,
                  penalty_r0: float = None):
    """Composite U for an (East, North) layout against a primary batch.

    Order-agnostic: the pair goes straight to the FNN + recon, both trained on the
    same `xy` column order, so only consistency matters here, not axis meaning.

    `fnn` is the dual-species wrapper — both per-species surrogates run on the same
    primary + layout and combine physically, so gradients reach (x_det, y_det)
    through BOTH. U_PR is computed but deliberately left out of the composite,
    matching production.

    NOT `@torch.no_grad()`-decorated, so L-BFGS can differentiate it; the DE
    optimizers wrap their own calls instead.

    `mesh_en` + `penalty_w` > 0 subtract the differentiable off-mesh penalty,
    putting the mountain boundary inside the objective. OFF by default so existing
    callers — notably plots/eval_true_utility.py, which depends on scoring exactly
    what the optimizer scored — get identical numbers. The returned U is the
    PENALISED objective; raw utility and penalty are both in `parts`. With
    r0 = max_gap a converged in-band layout has penalty exactly 0.
    """
    E_pred_det, E_pred_phys, theta_pred, phi_pred = _predict_primary(
        x_det, y_det, primary_batch, fnn, recon)

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
    U = _apply_offmesh_penalty(U, parts, x_det, y_det,
                               mesh_en, penalty_w, penalty_r0)
    return U, r, parts


# Divisor putting particles/shower into the same range as the composite U (~35).
# The 04 optimizers assume an objective of order tens (OFFMESH_PENALTY_W is quoted
# "in units of U", GRAD_CLIP=100), so an unscaled objective would swamp the
# boundary penalty.
#
# Calibrated on the SURROGATE, not the kernel — that is what this objective reads.
# Grid layout, 4000 corpus primaries:
#     surrogate   1.98e4 particles/shower, 47.4 soft detectors
#     kernel      3.77e5 particles/shower, 34.8 soft detectors
# The surrogate is ~19x low on particles but ~36% high on detector count, so a
# kernel-derived scale leaves the penalty ~19x overweight. 500 puts grid at ~40.
# Pure rescale — cannot change which layout wins, but does set the
# objective-to-penalty ratio.
PARTICLE_SCALE = 500.0

# PARTICLE_SCALE's counterpart for mode="distinct"; separate because the overlap
# correction divides by m_d, so the two modes do not share a range.
# MEASURED m_d (median over detectors): grid 1.61 (NN spacing 67 m vs sigma 50 m),
# distinct-optimised 1.25 (NN spacing 120 m). At 100 the objective runs ~123-159,
# i.e. ~3-4x above the U ~ 35 that OFFMESH_PENALTY_W is quoted against.
# Left at 100: runs exist at this value and rescaling breaks comparison against
# them. Revisit only alongside OFFMESH_PENALTY_W, never mid-experiment.
DISTINCT_SCALE = 100.0


def overlap_multiplicity(x_det: torch.Tensor, y_det: torch.Tensor,
                         sigma: float = SIGMA_SPATIAL) -> torch.Tensor:
    """Per-detector `m_d = sum_j exp(-r_dj^2 / (2 sigma^2))` (>= 1): how many
    detectors' worth of kernel weight lands on the particles detector d sees.

    The kernel gives each detector its own UNNORMALISED Gaussian with no
    exclusivity, so `sum_d count_d = sum_p e_p * M_p` with M_p the total weight
    particle p receives. Counting each particle once means dividing by M_p, but
    the surrogate only exposes per-detector aggregates — so evaluate it at the
    detector instead, which is >= 1 automatically (the j = d term is 1, no clamp).

    The weight must be the kernel's OWN Gaussian, not an overlap integral:
    `exp(-r^2/(4 sigma^2))` (two normalised Gaussians) looks natural but
    over-corrects by exactly 2x when detectors are dense. With the form above both
    limits are right — spacing >> sigma gives m -> 1, spacing << sigma gives
    m -> 2 pi sigma^2 / s^2, so the corrected sum tends to rho * area.

    Two coincident detectors give m = 2 and half a count each, so stacking gains
    nothing — the collapse degeneracy is removed inside the objective rather than
    fenced off by a spacing penalty.

    Assumes particle density is smooth over `sigma`; a shower core much narrower
    than 50 m is over-corrected. Exact distinct counting needs the per-particle
    (point, detector) tensor, which the surrogate does not expose.
    """
    d2 = ((x_det[:, None] - x_det[None, :]) ** 2
          + (y_det[:, None] - y_det[None, :]) ** 2)
    return torch.exp(-d2 / (2.0 * sigma ** 2)).sum(dim=1)     # diagonal contributes 1


def activation_of_xy(x_det: torch.Tensor,
                     y_det: torch.Tensor,
                     primary_batch: torch.Tensor,
                     fnn,
                     mode: str = "particles",
                     mesh_en: Optional[torch.Tensor] = None,
                     penalty_w: float = 0.0,
                     penalty_r0: Optional[float] = None):
    """How much a layout COLLECTS, differentiably — the activation objective.

    Training-time twin of `plots/eval_activation_counts.py::activation`, but fed by
    the SURROGATE, because the kernel's `compute_labels_batch` is `@torch.no_grad()`
    and cannot be optimized through. The two disagree ~19x on total particles (see
    PARTICLE_SCALE), so these numbers are NOT comparable to the evaluator's — that
    script's kernel scoring on held-out events is the honest check.

        counts    = expm1(fnn(primary, xy)[..., 0])       particles/detector
        p         = sigmoid(TAU_LAYOUT * (counts - LAYOUT_THRESHOLD))
        particles = counts.sum(1) / PARTICLE_SCALE
        detectors = p.sum(1)                              soft trigger count
        distinct  = (counts / m).sum(1) / DISTINCT_SCALE  m = overlap_multiplicity

    `mode` picks which is maximized; all three land in `parts` either way.
        particles  total flux — maximized EXACTLY by stacking on the densest point,
                   since the kernel double-counts. Use only knowing that.
        detectors  coverage; saturates at 1 each, so it pays to spread. The
                   composite U's reconstructability gate is a function of this.
        distinct   overlap-corrected flux: collection area, no stacking degeneracy.

    Contains NO reconstruction term by design — it rewards sitting where the flux
    is, not resolving direction or energy, so it is expected to disagree with
    `utility_of_xy`. Returns (U, r, parts) in the same shape so the ensemble
    machinery is a drop-in; `r` is reported, not optimized.
    """
    if mode not in ("particles", "detectors", "distinct"):
        raise ValueError("mode must be 'particles', 'detectors' or 'distinct', "
                         f"got {mode!r}")
    _, pred_ET = _surrogate_predict(x_det, y_det, primary_batch, fnn)
    # clamp_min(0) matches dual_surrogate.combine_species_outputs: a negative
    # predicted count is not physical, and zeroing its gradient stops the optimizer
    # chasing detectors into the surrogate's undershoot.
    counts = torch.expm1(pred_ET[..., 0]).clamp_min(0.0)
    p = torch.sigmoid(TAU_LAYOUT * (counts - LAYOUT_THRESHOLD))
    n_soft = p.sum(dim=1)
    particles = counts.sum(dim=1) / PARTICLE_SCALE
    # Layout-only, so it is the same (n_det,) vector for every event in the batch.
    m = overlap_multiplicity(x_det, y_det)
    distinct = (counts / m[None, :]).sum(dim=1) / DISTINCT_SCALE
    r = torch.sigmoid(TAU_RECONSTRUCT * (n_soft - RECONSTRUCT_THRESHOLD))

    U = dict(particles=particles, detectors=n_soft, distinct=distinct)[mode].mean()
    parts = dict(u_particles=particles.mean(), u_detectors=n_soft.mean(),
                 u_distinct=distinct.mean(), u_pr=W_PR * U_PR(r) / W_DIV)
    U = _apply_offmesh_penalty(U, parts, x_det, y_det,
                               mesh_en, penalty_w, penalty_r0)
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
