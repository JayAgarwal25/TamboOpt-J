"""Utility/quality metric functions for detector optimization.

Extracted from SWGOLO7_optimization.ipynb cells 25-28.
"""

import torch


def reconstructability(events, layout_threshold=5e-2, tau_layout=5.,
                       reconstruct_threshold=130., tau_reconstruct=5.):
    """Compute a differentiable reconstructability score per event.

    Parameters:
        events (torch.Tensor): per-event per-detector counts (shape: [Nevents, Nunits]).
        layout_threshold (float): flux threshold for detection.
        tau_layout (float): sigmoid steepness for detection.
        reconstruct_threshold (float): minimum number of detectors for reconstruction.
        tau_reconstruct (float): sigmoid steepness for reconstruction threshold.

    Returns:
        torch.Tensor: reconstructability scores per event in (0, 1).
    """
    soft_detect = torch.sigmoid(tau_layout * (events - layout_threshold))
    n = torch.sum(soft_detect, dim=1)
    r = torch.sigmoid(tau_reconstruct * (n - reconstruct_threshold))
    return r


def U_PR(r):
    """Utility term for reconstructability: grows with sqrt of summed reconstructability.

    Parameters:
        r (torch.Tensor): reconstructability scores per event.

    Returns:
        torch.Tensor: scalar utility contribution.
    """
    u = torch.sqrt(torch.sum(r) + 1e-6)
    return u


def _soft_cap(x, cap):
    """Smooth saturating cap: ~x for x << cap, -> cap as x -> inf.

    The per-event reward 1/(err^2 + eps) is already bounded by 1/eps, but that
    bound can still sit far above the typical event, letting a lucky handful
    dominate the batch mean. `cap` pulls the tail in without the gradient
    cliff a hard clamp would introduce.

    Pick `cap` well ABOVE the median reward, or the bulk of events saturate
    together and the term stops discriminating between layouts at all.
    """
    return cap * torch.tanh(x / cap)


def U_E(E_preds, E_trues, r, cap=None):
    """Utility term for energy performance, weighted by reconstructability.

    Parameters:
        E_preds (torch.Tensor): predicted energies (batch).
        E_trues (torch.Tensor): true energies (batch).
        r (torch.Tensor): reconstructability weights per event.
        cap (float | None): optional soft cap on the per-event reward
            1/(err^2 + .01), whose hard ceiling is 100. None = uncapped
            (original behaviour).

    Returns:
        torch.Tensor: scalar utility contribution.
    """
    inv_err = 1.0 / ((torch.log10(E_preds) - torch.log10(E_trues)) ** 2 + .01)
    if cap is not None:
        inv_err = _soft_cap(inv_err, cap)
    u = torch.mean(r * inv_err)
    return u


def U_angle(angle_preds, angle_trues, r, cap=None):
    """Utility term for angular performance , weighted by reconstructability.

    Parameters:
        angle_preds (torch.Tensor): predicted theta values.
        angle_trues (torch.Tensor): true theta values.
        r (torch.Tensor): reconstructability weights per event.
        cap (float | None): optional soft cap on the per-event reward
            1/(err^2 + .001), whose hard ceiling is 1000. None = uncapped
            (original behaviour). theta and phi need very different values —
            their error distributions are not comparable.

    Returns:
        torch.Tensor: scalar utility contribution.
    """
    inv_err = 1.0 / ((angle_preds - angle_trues) ** 2 + .001)
    if cap is not None:
        inv_err = _soft_cap(inv_err, cap)
    u = torch.mean(r * inv_err)
    return u
