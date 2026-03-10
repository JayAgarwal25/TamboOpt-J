"""Utility/quality metric functions for detector optimization.

Extracted from SWGOLO7_optimization.ipynb cells 25-28.
"""

import torch


def reconstructability(events, layout_threshold=1e-2, tau_layout=5.,
                       reconstruct_threshold=3., tau_reconstruct=5.):
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


def U_E(E_preds, E_trues, r):
    """Utility term for energy performance, weighted by reconstructability.

    Parameters:
        E_preds (torch.Tensor): predicted energies (batch).
        E_trues (torch.Tensor): true energies (batch).
        r (torch.Tensor): reconstructability weights per event.

    Returns:
        torch.Tensor: scalar utility contribution.
    """
    u = torch.sum(r / ((E_preds - E_trues) ** 2 + .01))
    return u


def U_TH(Th_preds, Th_trues, r):
    """Utility term for angular performance (theta), weighted by reconstructability.

    Parameters:
        Th_preds (torch.Tensor): predicted theta values.
        Th_trues (torch.Tensor): true theta values.
        r (torch.Tensor): reconstructability weights per event.

    Returns:
        torch.Tensor: scalar utility contribution.
    """
    u = torch.sum(r / ((Th_preds - Th_trues) ** 2 + .00001))
    return u
