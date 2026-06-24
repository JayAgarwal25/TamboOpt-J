"""(mu+lambda)-ES operators for v5 detector layout optimization.

All fitness evaluations are gradient-free — no .backward() is ever called on
detector positions. Layouts are (N_DETECTORS, 2) float32 numpy arrays [North, East].
Fitness is the same composite U as v6's 04_optimize_lbfgs_ensemble.py.
"""

import math
from typing import List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .constants import (
    N_DETECTORS,
    LOG_E_MIN, LOG_E_MAX,
    COS_THETA_MIN, COS_THETA_MAX, PHI_MIN, PHI_MAX,
    W_THETA, W_PHI, W_E, W_DIV,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD,
    ES_SIGMA_INIT, ES_SIGMA_FINAL,
)

# ── CMA-ES helper ─────────────────────────────────────────────────────────────

def project_layout(xy_flat: np.ndarray, mountain) -> np.ndarray:
    """Reshape a flat (N_DETECTORS*2,) CMA-ES solution and project to mountain.

    CMA-ES works in unconstrained R^200. This snaps each detector back onto the
    mountain surface after every ask(). We tell() CMA-ES the ORIGINAL flat vectors
    so its covariance model is not distorted by the projection.

    Returns (N_DETECTORS, 2) float32 [North, East].
    """
    xy = xy_flat.reshape(N_DETECTORS, 2).astype(np.float32)
    x_t = torch.as_tensor(xy[:, 0])
    y_t = torch.as_tensor(xy[:, 1])
    with torch.no_grad():
        x_proj, y_proj = project_to_mountain_ne(mountain, x_t, y_t)
    return np.stack([x_proj.numpy(), y_proj.numpy()], axis=1).astype(np.float32)

# These are injected by modules_v5/__init__.py (v3 on sys.path).
from modules.utility_functions import reconstructability, U_angle, U_E, U_PR
# modules_v6 is on sys.path because 01_run_evolution.py / 02_run_cmaes.py add it.
from modules_v6.tr_geometry_ne import project_to_mountain_ne, sample_initial_layout_ne


# ── Sigma schedule ────────────────────────────────────────────────────────────

def anneal_sigma(gen: int, n_gen: int) -> float:
    """Geometric sigma annealing from SIGMA_INIT (gen 0) to SIGMA_FINAL (gen n_gen-1).

    Returns the mutation standard deviation [m] for generation `gen`.
    """
    if n_gen <= 1:
        return float(ES_SIGMA_INIT)
    t = gen / (n_gen - 1)
    return float(ES_SIGMA_INIT * (ES_SIGMA_FINAL / ES_SIGMA_INIT) ** t)


# ── Layout sampling ───────────────────────────────────────────────────────────

def sample_layout(mountain, rng: np.random.Generator, scheme: str = "random") -> np.ndarray:
    """Sample one (N_DETECTORS, 2) layout [North, East] from the mountain surface."""
    north_np, east_np = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
    return np.stack([north_np, east_np], axis=1).astype(np.float32)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate_and_project(
    xy_np: np.ndarray,
    sigma: float,
    mountain,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add isotropic Gaussian noise (std=sigma [m]) to every detector, then project to mountain."""
    noise = rng.normal(0.0, sigma, size=xy_np.shape).astype(np.float32)
    xy_noisy = xy_np + noise
    x_t = torch.as_tensor(xy_noisy[:, 0])
    y_t = torch.as_tensor(xy_noisy[:, 1])
    with torch.no_grad():
        x_proj, y_proj = project_to_mountain_ne(mountain, x_t, y_t)
    return np.stack([x_proj.numpy(), y_proj.numpy()], axis=1).astype(np.float32)


# ── Crossover ─────────────────────────────────────────────────────────────────

def crossover_layouts(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hungarian-aligned uniform crossover: align xy_b to xy_a, swap ~50% of pairs."""
    n = xy_a.shape[0]
    # Pairwise squared-distance matrix between detectors of a and b.
    diff = xy_a[:, None, :] - xy_b[None, :, :]   # (n, n, 2)
    cost = (diff * diff).sum(axis=2)               # (n, n)
    _, col_ind = linear_sum_assignment(cost)
    xy_b_aligned = xy_b[col_ind]                   # reorder b to match a
    # Uniform crossover: swap each aligned pair independently with p=0.5.
    swap = rng.random(n) < 0.5
    child = xy_a.copy()
    child[swap] = xy_b_aligned[swap]
    return child


# ── Primary sampling ──────────────────────────────────────────────────────────

def sample_primaries(n: int, seed: int = 42) -> torch.Tensor:
    """Generate n synthetic primary encodings matching the v6 training distribution.

    Encoding: [dir_x, dir_y, dir_z, log_e_norm, pdg] shape (n, 5).
    Distribution matches AllShowers corpus: E uniform in log, zenith in [60°,100°], phi uniform.
    """
    rng = np.random.default_rng(seed)
    cos_theta = rng.uniform(COS_THETA_MIN, COS_THETA_MAX, n).astype(np.float32)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, None))
    phi       = rng.uniform(PHI_MIN, PHI_MAX, n).astype(np.float32)
    dir_x     = sin_theta * np.cos(phi)
    dir_y     = sin_theta * np.sin(phi)
    dir_z     = cos_theta
    log_e     = rng.uniform(LOG_E_MIN, LOG_E_MAX, n).astype(np.float32)
    log_e_norm = (log_e - LOG_E_MIN) / (LOG_E_MAX - LOG_E_MIN)
    pdg        = (np.arange(n, dtype=np.float32) % 2)
    return torch.from_numpy(
        np.stack([dir_x, dir_y, dir_z, log_e_norm, pdg], axis=1)
    )


# ── Fitness ───────────────────────────────────────────────────────────────────

def _decode_primary(
    primary: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode (B, 4+) primary encoding to physical (E_GeV, theta_rad, phi_rad).

    Reads only the first four columns [dir_x, dir_y, dir_z, log_e_norm];
    the optional 5th (pdg) is ignored.  Works on both the raw primary batch
    and the recon output (which has the same encoding convention).
    """
    dir_z      = primary[:, 2].clamp(-1.0, 1.0)
    log_e_norm = primary[:, 3]
    log_e      = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    E_gev      = torch.exp(log_e) - 1.0
    theta      = torch.arccos(dir_z)
    phi        = torch.atan2(primary[:, 1], primary[:, 0])
    phi        = torch.where(phi < 0.0, phi + 2.0 * math.pi, phi)
    return E_gev, theta, phi


@torch.no_grad()
def evaluate_single_layout(
    xy_np: np.ndarray,
    fnn,
    recon,
    primary_batch: torch.Tensor,
) -> Tuple[float, float]:
    """Compute (U, u_pr) for one layout against the fixed primary batch. Pure forward pass."""
    device = primary_batch.device
    B = primary_batch.shape[0]

    x_det = torch.as_tensor(xy_np[:, 0], dtype=torch.float32, device=device)
    y_det = torch.as_tensor(xy_np[:, 1], dtype=torch.float32, device=device)

    # (B, n_det, 2) — same layout broadcast over all primaries.
    xy = torch.stack([x_det, y_det], dim=-1).unsqueeze(0).expand(B, -1, -1)

    pred_ET = fnn(primary_batch, xy)               # (B, n_det, 2)
    E_pred  = pred_ET[..., 0]                       # log1p(N_tot)
    T_pred  = pred_ET[..., 1]                       # log1p(t_tot * T_LOG_SCALE)

    recon_in     = torch.stack([xy[..., 0], xy[..., 1], E_pred, T_pred], dim=-1)  # (B, n_det, 4)
    pred_primary = recon(recon_in)                 # (B, 4)

    E_pred_phys, theta_pred, phi_pred = _decode_primary(pred_primary)
    E_pred_phys = E_pred_phys.clamp(min=1.0)
    E_true, theta_true, phi_true = _decode_primary(primary_batch)

    r = reconstructability(
        torch.expm1(E_pred),
        layout_threshold=LAYOUT_THRESHOLD,
        reconstruct_threshold=RECONSTRUCT_THRESHOLD,
    )

    u_theta = U_angle(theta_pred, theta_true, r)
    u_phi   = U_angle(phi_pred,   phi_true,   r)
    u_e     = U_E(E_pred_phys,    E_true,     r)
    u_pr    = U_PR(r)

    U = (W_THETA * u_theta + W_PHI * u_phi + W_E * u_e) / W_DIV
    return float(U.item()), float((W_DIV * u_pr / W_DIV).item())


def evaluate_population(
    layouts: List[np.ndarray],
    fnn,
    recon,
    primary_batch: torch.Tensor,
) -> np.ndarray:
    """Return (len(layouts),) float64 array of U values for each layout."""
    fitnesses = np.empty(len(layouts), dtype=np.float64)
    for i, xy_np in enumerate(layouts):
        u, _ = evaluate_single_layout(xy_np, fnn, recon, primary_batch)
        fitnesses[i] = u
    return fitnesses
