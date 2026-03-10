"""Physics models ported from TamboDirReco.jl to differentiable PyTorch.

Source: tambo_dir_reco/src/utils.jl and tambo_dir_reco/src/reco.jl

All functions operate on PyTorch tensors and support autograd for use in
differentiable optimization pipelines.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Direction / geometry utilities (from utils.jl)
# ---------------------------------------------------------------------------

def get_dir_vec(theta, phi):
    """Convert spherical angles (theta, phi) to a unit direction vector.

    Ported from TamboDirReco.jl `get_dir_vec(theta, phi)`.

    Parameters:
        theta (torch.Tensor): polar angle (radians).
        phi (torch.Tensor): azimuthal angle (radians).

    Returns:
        torch.Tensor: shape (..., 3) unit direction vector [sin(th)cos(ph), sin(th)sin(ph), cos(th)].
    """
    sin_th = torch.sin(theta)
    return torch.stack([
        sin_th * torch.cos(phi),
        sin_th * torch.sin(phi),
        torch.cos(theta)
    ], dim=-1)


def get_angles(v):
    """Extract spherical angles (theta, phi) from a Cartesian vector.

    Ported from TamboDirReco.jl `get_angles(v)`.

    Parameters:
        v (torch.Tensor): shape (..., 3) Cartesian vector.

    Returns:
        tuple: (theta, phi) tensors in radians. phi is in [0, 2*pi).
    """
    phi = torch.atan2(v[..., 1], v[..., 0])
    theta = torch.atan2(torch.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2), v[..., 2])
    phi = phi % (2 * math.pi)
    return theta, phi


def flip_dir(theta, phi):
    """Reverse a direction: theta -> pi - theta, phi -> phi + pi.

    Ported from TamboDirReco.jl `flip_dir(theta, phi)`.

    Parameters:
        theta, phi (torch.Tensor): direction angles in radians.

    Returns:
        tuple: (theta_flipped, phi_flipped).
    """
    return math.pi - theta, phi + math.pi


def get_rotation_matrix(theta, phi):
    """Return the rotation matrix that maps get_dir_vec(theta, phi) to (0,0,1).

    Ported from TamboDirReco.jl `get_R(theta, phi)`: R = RotY(-theta) * RotZ(-phi).

    Parameters:
        theta (float or torch.Tensor): polar angle in radians (scalar).
        phi (float or torch.Tensor): azimuthal angle in radians (scalar).

    Returns:
        torch.Tensor: shape (3, 3) rotation matrix.
    """
    ct = torch.cos(theta) if isinstance(theta, torch.Tensor) else math.cos(theta)
    st = torch.sin(theta) if isinstance(theta, torch.Tensor) else math.sin(theta)
    cp = torch.cos(phi) if isinstance(phi, torch.Tensor) else math.cos(phi)
    sp = torch.sin(phi) if isinstance(phi, torch.Tensor) else math.sin(phi)

    # RotY(-theta) * RotZ(-phi)
    R = torch.tensor([
        [ ct * cp,  ct * sp, st],
        [-sp,       cp,      0.],
        [-st * cp, -st * sp, ct]
    ], dtype=torch.float64 if isinstance(ct, float) else theta.dtype)
    return R


def great_circle_distance(theta1, phi1, theta2, phi2):
    """Compute great-circle angular distance between two directions.

    Ported from TamboDirReco.jl `great_circle_distance(theta1, phi1, theta2, phi2)`.

    This is a proper angular distance metric, unlike simple MSE on angles. It correctly
    handles the spherical geometry and wrapping of azimuthal angles.

    Parameters:
        theta1, phi1: first direction (radians).
        theta2, phi2: second direction (radians).

    Returns:
        torch.Tensor: angular distance in radians.
    """
    cos_dist = (torch.cos(theta1) * torch.cos(theta2)
                + torch.sin(theta1) * torch.sin(theta2) * torch.cos(phi1 - phi2))
    cos_dist = torch.clamp(cos_dist, -1.0, 1.0)
    return torch.acos(cos_dist)


def great_circle_distance_deg(theta1, phi1, theta2, phi2):
    """Same as great_circle_distance but returns result in degrees.

    Parameters:
        theta1, phi1: first direction (radians).
        theta2, phi2: second direction (radians).

    Returns:
        torch.Tensor: angular distance in degrees.
    """
    return great_circle_distance(theta1, phi1, theta2, phi2) * (180.0 / math.pi)


# ---------------------------------------------------------------------------
# Timing model (from reco.jl: timing_llh_per_hit)
# ---------------------------------------------------------------------------

def timing_delay_quadratic(r, a):
    """Quadratic time delay model: dt = a * r^2.

    Ported from TamboDirReco.jl's timing likelihood model where the shower front
    curvature introduces a quadratic time delay as a function of radial distance
    from the shower axis.

    Parameters:
        r (torch.Tensor): radial distance from shower core in shower plane.
        a (torch.Tensor or float): curvature coefficient.

    Returns:
        torch.Tensor: predicted time delay.
    """
    return a * r ** 2


def timing_likelihood(delay_observed, delay_predicted, sigma_t):
    """Gaussian log-likelihood for timing residuals.

    Ported from TamboDirReco.jl `timing_llh_per_hit`. Returns per-hit log-likelihood
    (negative values; higher = better fit).

    Parameters:
        delay_observed (torch.Tensor): observed time delays.
        delay_predicted (torch.Tensor): predicted time delays from model.
        sigma_t (torch.Tensor): time uncertainty per hit.

    Returns:
        torch.Tensor: per-hit log-likelihood values.
    """
    return -0.5 * ((delay_observed - delay_predicted) ** 2) / (sigma_t ** 2)


# ---------------------------------------------------------------------------
# Lateral Distribution Function (from reco.jl: Q_ldf, ldf_llh_per_hit)
# ---------------------------------------------------------------------------

def ldf_model(r, log_A, beta, kappa=0.35, r_ref=100.0):
    """Power-law lateral distribution function (LDF).

    Ported from TamboDirReco.jl `Q_ldf`:
        Q(r) = A * (r/r_ref)^(-beta - kappa * log10(r/r_ref))

    Parameters:
        r (torch.Tensor): radial distance from shower core.
        log_A (torch.Tensor or float): log-amplitude (log_e).
        beta (torch.Tensor or float): power-law slope parameter.
        kappa (float): non-power-law correction coefficient (default 0.35).
        r_ref (float): reference distance (default 100 m).

    Returns:
        torch.Tensor: expected signal at distance r.
    """
    u = r / r_ref
    exponent = -beta - kappa * torch.log10(u)
    return torch.exp(log_A) * u ** exponent


def ldf_likelihood(n_observed, n_predicted):
    """Poisson log-likelihood for LDF hit counts.

    Ported from TamboDirReco.jl `ldf_llh_per_hit`.

    Parameters:
        n_observed (torch.Tensor): observed hit counts.
        n_predicted (torch.Tensor): predicted counts from LDF model.

    Returns:
        torch.Tensor: per-hit Poisson log-likelihood.
    """
    # Clamp to avoid log(0)
    n_predicted_safe = torch.clamp(n_predicted, min=1e-10)
    return n_observed * torch.log(n_predicted_safe) - n_predicted_safe


# ---------------------------------------------------------------------------
# Enhanced utility: U_TH using great-circle distance
# ---------------------------------------------------------------------------

def U_TH_great_circle(th_preds, ph_preds, th_trues, ph_trues, r):
    """Angular utility using proper great-circle distance instead of simple MSE.

    This replaces the original `U_TH` which only compared theta via squared error.
    Great-circle distance correctly accounts for the full 3D angular separation,
    including the azimuthal component.

    Parameters:
        th_preds, ph_preds (torch.Tensor): predicted angles (radians).
        th_trues, ph_trues (torch.Tensor): true angles (radians).
        r (torch.Tensor): reconstructability weights per event.

    Returns:
        torch.Tensor: scalar utility contribution.
    """
    angular_dist = great_circle_distance(th_preds, ph_preds, th_trues, ph_trues)
    u = torch.sum(r / (angular_dist ** 2 + 1e-5))
    return u
