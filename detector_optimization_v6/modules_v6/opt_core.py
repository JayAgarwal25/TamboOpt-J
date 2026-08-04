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
CAP_E     = 50.0

# GEOMETRY_PATH_RESOLVED is centralized in constants (mesh-agnostic: local copy of
# the configured mesh, else the absolute path) and re-exported here for callers
# (04 DE / DE-pop) that import it from opt_core.


# ── Objective helpers ─────────────────────────────────────────────────────────
def primary_to_physical_labels(primary: torch.Tensor):
    """(B, 5) -> (E_GeV, θ_rad, φ_rad). Matches 04_optimize.py."""
    # Normalize before reading off angles. The true primary is already a unit
    # vector, but this is also called on the RECON's raw regression output, whose
    # norm shrinks toward the population mean under MSE training. Taking
    # arccos(dir_z) off an un-normalized vector then biases zenith toward 90 deg:
    # a predicted direction of norm 0.721 reads as 66.4 deg where its actual polar
    # angle is 56.3 deg. Azimuth is scale-invariant so atan2 is unaffected either
    # way. This matches how `plots/eval_recon_resolution.py` measures the angle.
    d = primary[:, 0:3]
    d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-12)
    dir_x = d[:, 0]
    dir_y = d[:, 1]
    dir_z = d[:, 2].clamp(-1.0, 1.0)
    log_e_norm = primary[:, 3]
    log_e = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    # `encode_primary` stores log10(E), so the inverse is 10**log_e. This was
    # torch.exp(log_e) - 1.0, which returned 147..1097 GeV for a 1e5..1e7 GeV band.
    # U_E takes log10 of this, so the mismatch rescaled every energy error by
    # log10(e) = 0.4343 and its square by 5.3 — meaning eps=0.01, sized to floor
    # the reward at 0.1 dex, actually floored it at 0.23 dex, and CAP_E was tuned
    # against the compressed scale.
    E_gev = torch.pow(10.0, log_e)
    theta = torch.arccos(dir_z)
    phi   = torch.atan2(dir_y, dir_x)
    two_pi = 2.0 * math.pi
    phi = torch.where(phi < 0, phi + two_pi, phi)
    return E_gev, theta, phi


def utility_of_xy(x_det: torch.Tensor,
                  y_det: torch.Tensor,
                  primary_batch: torch.Tensor,
                  fnn,
                  recon,
                  reconstruct_threshold: float = None):
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

    `reconstruct_threshold` overrides the module-level RECONSTRUCT_THRESHOLD
    (default None keeps production behavior unchanged) -- lets diagnostics
    rescale the "minimum detectors firing" bar when evaluating layouts with a
    detector count other than the production N_DETECTORS=100.

    NOT decorated with `@torch.no_grad()` so the L-BFGS optimizer can
    differentiate it; the gradient-free DE optimizers call it inside their own
    `no_grad` block."""
    if reconstruct_threshold is None:
        reconstruct_threshold = RECONSTRUCT_THRESHOLD
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
        reconstruct_threshold=reconstruct_threshold,
        tau_reconstruct=TAU_RECONSTRUCT,
    )
    u_theta = U_angle(theta_pred, theta_true, r, cap=CAP_THETA)
    # Azimuth is periodic and primary_to_physical_labels maps it into [0, 2*pi),
    # so the difference has to be wrapped or events straddling the branch cut are
    # scored as nearly a full turn wrong. Zenith lives in [0, pi] and must NOT be
    # wrapped. NOTE: CAP_PHI was calibrated on the unwrapped distribution and is
    # due a recalibration now that the tail it was sized against is gone.
    u_phi   = U_angle(phi_pred,   phi_true,   r, cap=CAP_PHI, period=2.0 * math.pi)
    u_e     = U_E    (E_pred_phys, E_true,    r, cap=CAP_E)
    u_pr    = U_PR(r)
    U = (W_THETA * u_theta + W_PHI * u_phi + W_E * u_e) / W_DIV
    return U, r, dict(u_theta=W_THETA * u_theta / W_DIV, u_phi=W_PHI * u_phi / W_DIV, u_e=W_E * u_e / W_DIV, u_pr=W_PR * u_pr / W_DIV)


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
