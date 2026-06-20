"""(mu+lambda)-ES operators for v5 detector layout optimization.

All fitness evaluations are GRADIENT-FREE — this module never calls
.backward() and wraps every surrogate call in torch.no_grad().  The ES
must remain gradient-free to be a scientifically fair comparison against
v6's gradient-based L-BFGS/DE on the same objective.

Layout representation
---------------------
A layout is a (N_DETECTORS, 2) float32 numpy array whose columns are
[North, Up] in metres.  Numpy is used for layouts in the loop; torch
tensors only appear inside evaluate_single_layout where the frozen
surrogate runs.

Fitness function
----------------
Identical to v6's utility_of_xy (04_optimize_lbfgs_ensemble.py):

    U = (W_THETA * u_θ + W_PHI * u_φ + W_E * u_E) / W_DIV

U_PR is returned for logging but excluded from the composite, matching
v6 production.  The surrogate is the frozen dual-species FNN + flat-MLP
recon from test_v6_run_02_recentered / test_v6_run_03_recentered.

Public API
----------
    anneal_sigma(gen, n_gen)
    sample_layout(mountain, rng, scheme='random')
    mutate_and_project(xy_np, sigma, mountain, rng)
    crossover_layouts(xy_a, xy_b, rng)
    evaluate_single_layout(xy_np, fnn, recon, primary_batch)
    evaluate_population(layouts, fnn, recon, primary_batch)
    sample_primaries(n, seed)
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

# These are injected by modules_v5/__init__.py (v3 on sys.path).
from modules.utility_functions import reconstructability, U_angle, U_E, U_PR


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
    """Sample one (N_DETECTORS, 2) layout from the mountain surface.

    Args:
        mountain : MountainData from modules_v4.tr_geometry.load_tr_mountain.
        rng      : numpy Generator for reproducible sampling.
        scheme   : 'random' (default) or 'grid' or 'center'.

    Returns:
        (N_DETECTORS, 2) float32 numpy array, columns = [North, Up].
    """
    # sample_initial_layout uses its own internal rng, so we pass a seed
    # derived from the caller's rng to get deterministic but varied layouts.
    seed = int(rng.integers(0, 2**31))
    _orig = np.random.get_state()
    np.random.seed(seed)
    try:
        north_np, up_np = mountain.sample_initial_layout(n_units=N_DETECTORS, scheme=scheme)
    finally:
        np.random.set_state(_orig)
    return np.stack([north_np, up_np], axis=1).astype(np.float32)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate_and_project(
    xy_np: np.ndarray,
    sigma: float,
    mountain,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add isotropic Gaussian noise to every detector, then project to mountain.

    Args:
        xy_np   : (N_DETECTORS, 2) float32 numpy layout [North, Up].
        sigma   : mutation standard deviation [m].
        mountain: MountainData — provides project_to_mountain().
        rng     : numpy Generator.

    Returns:
        New (N_DETECTORS, 2) float32 numpy layout on the mountain surface.
    """
    noise = rng.normal(0.0, sigma, size=xy_np.shape).astype(np.float32)
    xy_noisy = xy_np + noise
    # project_to_mountain snaps any point that drifted off the mountain to its
    # nearest centroid; it operates on torch tensors.
    x_t = torch.as_tensor(xy_noisy[:, 0])
    y_t = torch.as_tensor(xy_noisy[:, 1])
    with torch.no_grad():
        x_proj, y_proj = mountain.project_to_mountain(x_t, y_t)
    return np.stack([x_proj.numpy(), y_proj.numpy()], axis=1).astype(np.float32)


# ── Crossover ─────────────────────────────────────────────────────────────────

def crossover_layouts(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hungarian-aligned uniform crossover between two parent layouts.

    Steps:
      1. Solve the min-cost assignment aligning xy_b's detectors to xy_a's.
      2. Randomly swap ~50% of the aligned detector pairs.

    The alignment ensures that swapped pairs are geographically co-located,
    so the child layout is a physically meaningful blend rather than a
    scrambled concatenation.

    Args:
        xy_a, xy_b : (N_DETECTORS, 2) float32 layouts [North, Up].
        rng        : numpy Generator.

    Returns:
        Child layout (N_DETECTORS, 2) float32.
    """
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

    Distribution (matches AllShowers corpus used to train the surrogate):
      - log10(E/GeV) ~ Uniform[5, 8]
      - cos(theta)   ~ Uniform[cos(100°), cos(60°)]  (downward-going showers)
      - phi          ~ Uniform[0, 2*pi]
      - pdg          alternates 0 (electron) / 1 (muon)

    Encoding: [dir_x, dir_y, dir_z, log_e_norm, pdg]  shape (n, 5)
    where log_e_norm = (log10(E) - LOG_E_MIN) / (LOG_E_MAX - LOG_E_MIN).

    The DualSpeciesSurrogate overrides the pdg column internally, so its
    exact value here does not affect the combined output.

    Args:
        n    : number of primaries.
        seed : numpy rng seed for full reproducibility.

    Returns:
        (n, 5) float32 tensor on CPU.
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
    """Compute utility U for one layout against the fixed primary batch.

    This is a PURE FORWARD PASS — no gradients, no backward.  The
    @torch.no_grad() decorator guarantees that no gradient tape is built
    even if the caller forgot to disable gradients.

    Args:
        xy_np        : (N_DETECTORS, 2) float32 numpy layout [North, Up].
        fnn          : frozen DualSpeciesSurrogate.
        recon        : frozen reconstruction network.
        primary_batch: (B, 5) float32 tensor on the same device as fnn.

    Returns:
        (U, u_pr) where U is the composite utility (the ES fitness) and
        u_pr is the reconstructability term (for logging only).
    """
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
    """Evaluate U for every layout in `layouts`.

    Args:
        layouts      : list of (N_DETECTORS, 2) float32 numpy arrays.
        fnn, recon   : frozen surrogate models.
        primary_batch: (B, 5) tensor on GPU.

    Returns:
        (len(layouts),) float64 numpy array of U values.
    """
    fitnesses = np.empty(len(layouts), dtype=np.float64)
    for i, xy_np in enumerate(layouts):
        u, _ = evaluate_single_layout(xy_np, fnn, recon, primary_batch)
        fitnesses[i] = u
    return fitnesses
