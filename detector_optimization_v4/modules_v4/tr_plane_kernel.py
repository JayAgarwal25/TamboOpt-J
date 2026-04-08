"""Plane-aware differentiable detector response kernel for v4.

v3's GetCounts_differentiable uses a 2D Gaussian spatial kernel over the point
cloud and ignores the layer_index column (column 2 of the (B, max_points, 5)
samples tensor).  v3 compensates by zeroing out all points whose layer_index
differs from the target plane (filter_plane=20 in ComputeShowerDetection).

v4 needs the kernel to be differentiable in the *plane index* as well, so that
gradients flow from the loss back through z_cont → East → (North, Up).  The
solution is a triangular plane weight:

    plane_w[b, p, i] = relu(1 - |layer_p[b, p] - z_cont[i]|)

This is 1 when layer_p == z_cont_i (exact match), linearly decreases to 0 at
±1 layer away, and is 0 everywhere else.  It is differentiable in z_cont
almost everywhere (non-differentiable only at the kink z_cont = layer_p ± 1,
which has measure zero).

The combined kernel is:
    kernel = spatial_gaussian * plane_w      (B, max_points, n_det)

This is exactly v3's kernel when z_cont ≡ 20 for all detectors: plane_w
becomes 1 for all layer-20 points and 0 for the rest, reproducing filter_plane=20.

Post-processing (SmearN, TimeAverage_vectorized) is identical to v3 — the
callables are imported from v3's modules.detector_response and passed in.
"""

import torch


def GetCounts_planeaware(
    samples: torch.Tensor,
    x_det:   torch.Tensor,
    y_det:   torch.Tensor,
    z_cont:  torch.Tensor,
    SmearN_fn,
    fluxB_e:  torch.Tensor,
    TimeAverage_vectorized_fn,
    sigma:   float = 200.0,
) -> tuple:
    """Plane-aware differentiable count extraction.

    Computes (N_int, T_int) per detector, differentiable w.r.t. x_det, y_det,
    and z_cont (and hence differentiable w.r.t. the learnable North/Up positions
    via the SurfaceEastMap).

    Args:
        samples  : (B, max_points, 5) point-cloud tensor with columns
                   [x, y, layer_index, energy, time].  Padding rows have energy=0.
        x_det    : (n_det,) North coordinates [m], requires_grad may be True.
        y_det    : (n_det,) Up   coordinates [m], requires_grad may be True.
        z_cont   : (n_det,) continuous plane index ∈ [0, n_planes-1],
                   derived as (East - east_min) / plane_dx.  requires_grad may be True.
        SmearN_fn             : accepted for interface compatibility, not called.
        fluxB_e               : accepted for interface compatibility, not called.
        TimeAverage_vectorized_fn : accepted for interface compatibility, not called.
        sigma    : Gaussian spatial kernel width [m] (default 200 m, same as v3).

    Returns:
        (local_intensity, et) : each (B, n_det), differentiable in x_det, y_det, z_cont.
        Matches v3's GetCounts_differentiable return convention (raw values, no post-processing).
    """
    point_x = samples[..., 0]    # (B, P)
    point_y = samples[..., 1]    # (B, P)
    point_l = samples[..., 2]    # (B, P)  layer index (integer, but stored as float)
    point_e = samples[..., 3]    # (B, P)  energy
    point_t = samples[..., 4]    # (B, P)  time

    # ── Spatial Gaussian — identical to v3 ────────────────────────────────────
    # dx, dy : (B, P, n_det)
    dx = point_x.unsqueeze(2) - x_det.unsqueeze(0).unsqueeze(0)
    dy = point_y.unsqueeze(2) - y_det.unsqueeze(0).unsqueeze(0)
    spatial = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))

    # ── Triangular plane weight — differentiable in z_cont ────────────────────
    # delta_l : (B, P, n_det)
    delta_l = point_l.unsqueeze(2) - z_cont.unsqueeze(0).unsqueeze(0)
    plane_w = torch.relu(1.0 - delta_l.abs())

    # ── Combined kernel ───────────────────────────────────────────────────────
    kernel        = spatial * plane_w                              # (B, P, n_det)
    energy_kernel = point_e.unsqueeze(2) * kernel                 # (B, P, n_det)

    local_intensity = energy_kernel.sum(dim=1)                    # (B, n_det)
    et = (
        (point_t.unsqueeze(2) * energy_kernel).sum(dim=1)
        / local_intensity.clamp(min=1e-8)
    )                                                              # (B, n_det)

    # Return raw (local_intensity, et), matching v3's GetCounts_differentiable
    # behaviour.  SmearN_fn / TimeAverage_vectorized_fn are accepted for
    # interface compatibility but not called — v3 also accepts them as kwargs
    # without calling them.
    return local_intensity, et
