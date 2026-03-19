"""Detector response functions: counts, smearing, and timing.

Extracted from SWGOLO7_optimization.ipynb cells 16, 20, 21.
"""

import torch
import torch.nn.functional as F


def GetCounts_differentiable(shower_rgb, x, y, bboxes, SmearN_fn, fluxB_e,
                             TimeAverage_vectorized_fn, electron_scale_factor=10,
                                temperature=0.1):
    """Differentiable version of count extraction. Gradients flow w.r.t. x, y.

    Parameters:
        shower_rgb: (B, 32, 32, 3) tensor.
        x, y: detector positions (num_det,) tensors with requires_grad=True.
        bboxes: (B, 4) bounding box [x_min, x_max, y_min, y_max].
        SmearN_fn: callable for detector smearing.
        fluxB_e: background flux tensor.
        TimeAverage_vectorized_fn: callable for time averaging.
        electron_scale_factor (float): scaling factor for electron counts.
        temperature (float): sigmoid temperature for soft masking.

    Returns:
        tuple: (Ne, Te) -- (B, num_det) tensors, differentiable w.r.t. x, y.
    """
    B = shower_rgb.shape[0]
    num_det = len(x)

    x_min, x_max = bboxes[:, 0], bboxes[:, 1]
    y_min, y_max = bboxes[:, 2], bboxes[:, 3]
    bboxes_width = x_max - x_min
    bboxes_height = y_max - y_min

    particle_density_energy = torch.prod(shower_rgb[:, :, :, :2], dim=3)
    arrival_time_map = shower_rgb[:, :, :, 2]

    # Bilinear sampling (differentiable w.r.t. x, y)
    #   x_norm01[b, d] ∈ [0,1]: fraction along x-axis within bbox b
    #   y_norm01[b, d] ∈ [0,1]: fraction along y-axis within bbox b
    x_norm01 = (x.unsqueeze(0) - x_min.unsqueeze(1)) / bboxes_width.unsqueeze(1)
    y_norm01 = (y.unsqueeze(0) - y_min.unsqueeze(1)) / bboxes_height.unsqueeze(1)

    x_norm11 = 2 * x_norm01 - 1
    y_norm11 = 2 * y_norm01 - 1

    # Build grid: shape (B, num_det, 1, 2)
    # IMPORTANT (from docs): grid[..., 0] = y → H direction (columns) → y_norm11
    #                        grid[..., 1] = x → W direction (rows)    → x_norm11
    # This matches original indexing: particle_density_energy[b, y_idx, x_idx]
    #                                                               ↑H       ↑W
    grid = torch.stack(
        [y_norm11, x_norm11],   # ← (H-coord, W-coord)
        dim=-1
    ).unsqueeze(2)              # (B, num_det, 1, 2)
    
    # Input maps: (B, 1, H=32, W=32)
    inp_intensity = particle_density_energy.unsqueeze(1)
    inp_time = arrival_time_map.unsqueeze(1)

    # grid_sample output: (B, 1, num_det, 1) → squeeze → (B, num_det)
    # padding_mode='zeros': out-of-bbox detectors get 0 (handles edge case)
    # align_corners=True:   extrema ±1 refer to center of corner pixels,
    #                       consistent with normalizing x_min→-1, x_max→1
    local_intensity = F.grid_sample(
        inp_intensity, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(1).squeeze(-1)    # (B, num_det)

    et = F.grid_sample(
        inp_time, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(1).squeeze(-1)    # (B, num_det)

    # Smearing
    e0 = local_intensity * electron_scale_factor
    nes = SmearN_fn(e0)
    neb = SmearN_fn(fluxB_e.unsqueeze(0).expand(B, num_det))
    Ne = nes + neb                                          # (B, num_det)

    # Vectorized arrival time
    TAe_m, TAe_s = TimeAverage_vectorized_fn(et, neb, nes)  # (B, num_det)
    eps = 0.05 * torch.randn_like(TAe_m)
    Te_raw = TAe_m + TAe_s * eps                            # (B, num_det)

    mask_soft = torch.sigmoid(Ne / temperature)             # (B, num_det)
    Te = mask_soft * Te_raw                                 # (B, num_det)

    return Ne, Te


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
    sqrt12 = torch.tensor([12.0]).sqrt()

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
