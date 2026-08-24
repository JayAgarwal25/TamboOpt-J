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

Return convention: local_intensity is the raw kernel-weighted energy sum
(matches v3's GetCounts_differentiable, no post-processing). arrival_time is a
LEADING EDGE: the time of the earliest point whose deposit at that detector
clears `hit_deposit_min`, which is what a photodetector actually reports. It
also makes the multi-species combination a `min`, which is associative and so
independent of how many species the corpus is split into.
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
    hit_deposit_min: float = None,
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
        hit_deposit_min : a point is DETECTED at a detector when its deposit
                   there (energy * kernel acceptance) exceeds this. Defaults to
                   constants.HIT_DEPOSIT_MIN.

    Returns:
        (local_intensity, arrival_time) : each (B, n_det). local_intensity is
        differentiable in x_det, y_det, z_cont; arrival_time is a min over
        detected points and is not. local_intensity is the raw kernel-weighted energy
        sum, matching v3's GetCounts_differentiable convention. Padding rows
        (energy=0) carry zero weight in local_intensity by construction and
        additionally carry zero kernel weight through the plane term whenever
        their layer_index sits outside every detector's ±1 window (the usual
        padding convention), so they cannot move arrival_time under either
        form: dividing a fixed zero contribution by sum(K) or by P is still zero.
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

    # Leading edge: the earliest point whose deposit clears the threshold.
    # Padding rows carry energy 0 (showerdata zero-fills), so they never clear it
    # and drop out on their own, with no separate mask and no dependence on the
    # corpus's padded width. T is a selection, so it carries no gradient; E still
    # does, and layouts get gradients through the surrogate trained on these
    # labels rather than through the kernel.
    if hit_deposit_min is None:
        from modules_v6.constants import HIT_DEPOSIT_MIN
        hit_deposit_min = HIT_DEPOSIT_MIN
    detected     = energy_kernel > hit_deposit_min                 # (B, P, n_det)
    t_masked     = torch.where(detected, point_t.unsqueeze(2),
                               torch.full_like(energy_kernel, float("inf")))
    arrival_time = t_masked.amin(dim=1)                            # (B, n_det)
    # No detected point -> 0, the same "no signal" sentinel the E channel uses.
    arrival_time = torch.where(torch.isinf(arrival_time),
                               torch.zeros_like(arrival_time), arrival_time)

    # Return (local_intensity, arrival_time), matching v3's GetCounts_differentiable
    # behaviour.  SmearN_fn / TimeAverage_vectorized_fn are accepted for
    # interface compatibility but not called — v3 also accepts them as kwargs
    # without calling them.
    return local_intensity, arrival_time
