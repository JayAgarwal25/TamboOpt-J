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
    GEOMETRY_PATH, LOG_E_MIN, LOG_E_MAX,
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

# Reconstructability thresholds — match 04_optimize.py
LAYOUT_THRESHOLD      = 5e-2
RECONSTRUCT_THRESHOLD = 10.0

# constants.GEOMETRY_PATH may be stale; prefer a local copy, then the TAMBOSim path.
GEOMETRY_PATH_RESOLVED = next(
    (p for p in (
        os.path.join(_REPO, "colca_valley.h5"),
        "/n/home05/zdimitrov/tambo/TAMBOSim/resources/geometry/colca_valley.h5",
        GEOMETRY_PATH,
    ) if os.path.exists(p)),
    GEOMETRY_PATH,
)


# ── Objective helpers ─────────────────────────────────────────────────────────
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
                  fnn,
                  recon):
    """Composite U for a (North, East) layout against a primary batch.

    `fnn` is the dual-species wrapper: both per-species surrogates are evaluated
    with the same primary + layout and physically combined, so the backprop into
    (x_det, y_det) flows through BOTH models. Mirrors the inner loop of
    `_run_optimization` in 04_optimize.py (the U_PR term is computed but
    deliberately omitted from the composite, matching production).

    NOT decorated with `@torch.no_grad()` so the L-BFGS optimizer can
    differentiate it; the gradient-free DE optimizers call it inside their own
    `no_grad` block."""
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
    pred = recon(recon_feats)                                              # (B, 4); DeepSets recon takes (B, n_det, 4)
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
    (B, n_det, 4) per-detector features), applies its normalization, and freezes
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
