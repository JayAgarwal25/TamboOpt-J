"""Detector response functions: counts, smearing, and timing.

Extracted from SWGOLO7_optimization.ipynb cells 16, 20, 21.
"""

import torch
import torch.nn.functional as F


def GetCounts_differentiable(samples, x_det, y_det, SmearN_fn, fluxB_e,
                             TimeAverage_vectorized_fn, sigma=200.0,
                             electron_scale_factor=10, temperature=0.1):
    """Differentiable version of count extraction from point-cloud showers.

    Replaces bilinear image sampling with a Gaussian kernel over the point
    cloud, which is differentiable w.r.t. detector positions ``x_det``/``y_det``.

    Parameters:
        samples (torch.Tensor): (B, max_points, 5) point-cloud tensor with
            columns ``[x, y, layer_index, energy, time]``.
        x_det (torch.Tensor): detector x positions, shape (num_det,),
            with ``requires_grad=True``.
        y_det (torch.Tensor): detector y positions, shape (num_det,),
            with ``requires_grad=True``.
        SmearN_fn (callable): callable for detector smearing.
        fluxB_e: background flux tensor.
        TimeAverage_vectorized_fn (callable): callable for time averaging.
        sigma (float, optional): Gaussian kernel width [m] controlling each
            detector's spatial collection radius. Defaults to 5.0.
        electron_scale_factor (float, optional): scaling factor for electron
            counts. Defaults to 10.
        temperature (float, optional): sigmoid temperature for soft masking.
            Defaults to 0.1.

    Returns:
        tuple: (local_intensity, et) -- (B, num_det) tensors,
               differentiable w.r.t. x_det and y_det.
    """
    # samples: (B, max_points, 5) — columns: x, y, layer_index, energy, time
    point_x = samples[:, :, 0]  # (B, max_points)
    point_y = samples[:, :, 1]  # (B, max_points)
    point_e = samples[:, :, 3]  # (B, max_points)
    point_t = samples[:, :, 4]  # (B, max_points)

    # Squared distances between every point and every detector
    # dx/dy: (B, max_points, num_det)
    dx = point_x.unsqueeze(2) - x_det.unsqueeze(0).unsqueeze(0)
    dy = point_y.unsqueeze(2) - y_det.unsqueeze(0).unsqueeze(0)
    dist2 = dx ** 2 + dy ** 2  # (B, max_points, num_det)

    # Gaussian kernel weights — gradient flows through dx/dy → x_det, y_det
    kernel = torch.exp(-dist2 / (2 * sigma ** 2))  # (B, max_points, num_det)

    # Energy-weighted intensity at each detector: (B, num_det)
    energy_kernel = point_e.unsqueeze(2) * kernel   # (B, max_points, num_det)
    local_intensity = energy_kernel.sum(dim=1)       # (B, num_det)

    # Energy-weighted arrival time at each detector: (B, num_det)
    et = (point_t.unsqueeze(2) * energy_kernel).sum(dim=1) / local_intensity.clamp(min=1e-8)

    return local_intensity, et


def SmearN(flux, RelResCounts=0.05):
    """Apply detector resolution and threshold gating to expected flux values.

    Parameters:
        flux (torch.Tensor): expected number of particles.
        RelResCounts (float): relative resolution on counts.

    Returns:
        torch.Tensor: noisy, gated counts reflecting detector response.
    """
    gate = torch.sigmoid(2 * (flux - 0.1))
    noise = torch.randn_like(flux)
    noisy = flux + RelResCounts * flux * noise
    return gate * noisy


def TimeAverage_vectorized(T, Nb, Ns, IntegrationWindow=128., sigma_time=10., eps=1e-8):
    """Fully vectorized and differentiable TimeAverage.

    All inputs: (B, num_det) tensors. Gradients flow through T, Nb, Ns.

    Parameters:
        T: arrival time tensor.
        Nb: background count tensor.
        Ns: signal count tensor.
        IntegrationWindow (float): integration window in ns.
        sigma_time (float): time resolution in ns.
        eps (float): numerical stability constant.

    Returns:
        tuple: (mean, std) tensors of shape (B, num_det).
    """
    sqrt12 = torch.tensor([12.0], device=T.device).sqrt()

    STbgr = torch.where(Nb <= 1, IntegrationWindow / sqrt12,
            torch.where(Nb <= 2, torch.full_like(Nb, IntegrationWindow * .2041),
            torch.where(Nb <= 3, torch.full_like(Nb, IntegrationWindow * .16666),
            torch.where(Nb <= 4, torch.full_like(Nb, IntegrationWindow * .1445),
                                 torch.full_like(Nb, IntegrationWindow * .11)))))
    eps_bgr = torch.randn_like(T) - 0.5
    AvTbgr_raw = T + 0.05 * eps_bgr * STbgr
    AvTbgr = T + torch.clamp(AvTbgr_raw - T,
                              -0.5 * IntegrationWindow,
                               0.5 * IntegrationWindow)

    STsig = sigma_time / torch.sqrt(torch.clamp(Ns - 1, min=1e-3))
    eps_sig = torch.randn_like(T)
    AvTsig = T + 0.05 * eps_sig * STsig

    tau = 0.1
    has_bgr = torch.sigmoid((Nb - 0.5) / tau)
    has_sig = torch.sigmoid((Ns - 0.5) / tau)
    neither = (1 - has_bgr) * (1 - has_sig)

    VTbgr = STbgr ** 2
    VTsig = STsig ** 2

    precision = has_bgr / (VTbgr + eps) + has_sig / (VTsig + eps)
    mean_num = has_bgr * AvTbgr / (VTbgr + eps) + has_sig * AvTsig / (VTsig + eps)
    combined_mean = mean_num / (precision + eps)
    combined_std = torch.sqrt(1.0 / (precision + eps))

    fallback_mean = T
    fallback_std = torch.ones_like(T) * (IntegrationWindow / sqrt12)

    mean = neither * fallback_mean + (1 - neither) * combined_mean
    std = neither * fallback_std + (1 - neither) * combined_std

    return mean, std
