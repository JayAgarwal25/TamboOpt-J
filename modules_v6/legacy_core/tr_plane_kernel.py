"""Plane-aware differentiable detector response kernel.

Ground truth for the Step-1 labels: (E, T) per detector from a shower point
cloud, differentiable in the detector positions AND in the plane index, so
gradients reach z_cont -> East -> (North, Up).

The plane weight is triangular,

    plane_w[b, p, i] = relu(1 - |layer_p[b, p] - z_cont[i]|)

i.e. 1 on an exact layer match, falling linearly to 0 one layer away. It is
differentiable in z_cont except at the kinks z_cont = layer_p +- 1 (measure
zero). Combined with the spatial Gaussian:

    kernel = spatial_gaussian * plane_w      (B, max_points, n_det)
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
        SmearN_fn, fluxB_e, TimeAverage_vectorized_fn :
                   accepted for interface compatibility; never called.
        sigma    : Gaussian spatial kernel width [m].

    Returns:
        (local_intensity, arrival_time) : each (B, n_det), raw (no post-processing),
        differentiable in x_det, y_det, z_cont.
    """
    point_x = samples[..., 0]    # (B, P)
    point_y = samples[..., 1]    # (B, P)
    point_l = samples[..., 2]    # (B, P)  layer index (integer, but stored as float)
    point_e = samples[..., 3]    # (B, P)  energy
    point_t = samples[..., 4]    # (B, P)  time

    # dx, dy : (B, P, n_det)
    dx = point_x.unsqueeze(2) - x_det.unsqueeze(0).unsqueeze(0)
    dy = point_y.unsqueeze(2) - y_det.unsqueeze(0).unsqueeze(0)
    spatial = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))

    # delta_l : (B, P, n_det)
    delta_l = point_l.unsqueeze(2) - z_cont.unsqueeze(0).unsqueeze(0)
    plane_w = torch.relu(1.0 - delta_l.abs())

    kernel        = spatial * plane_w                              # (B, P, n_det)
    energy_kernel = point_e.unsqueeze(2) * kernel                 # (B, P, n_det)

    local_intensity = energy_kernel.sum(dim=1)                    # (B, n_det)
    # NOTE: mean over ALL P points, padding rows included. Looks wrong, but it
    # defines every existing label — changing it invalidates the corpus.
    arrival_time = (point_t.unsqueeze(2) * kernel).mean(dim=1)

    return local_intensity, arrival_time
