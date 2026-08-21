"""The individual terms the Step-4 objective sums into U."""

import torch


def reconstructability(events, layout_threshold=5e-2, tau_layout=5.,
                       reconstruct_threshold=130., tau_reconstruct=5.):
    """Differentiable reconstructability per event, in (0, 1).

    Two stacked sigmoids: soft-detect each detector above `layout_threshold`,
    then soft-threshold the detector count at `reconstruct_threshold`. The
    `tau_*` are sigmoid steepnesses. `events` is (n_events, n_detectors).
    """
    soft_detect = torch.sigmoid(tau_layout * (events - layout_threshold))
    n = torch.sum(soft_detect, dim=1)
    r = torch.sigmoid(tau_reconstruct * (n - reconstruct_threshold))
    return r


def U_PR(r):
    """Reconstructability term: sqrt of the summed per-event scores."""
    u = torch.sqrt(torch.sum(r) + 1e-6)
    return u


def _soft_cap(x, cap):
    """Smooth saturating cap: ~x for x << cap, -> cap as x -> inf.

    Pulls in the tail of the per-event reward — whose own 1/eps bound can sit far
    above the typical event, letting a lucky handful dominate the batch mean —
    without the gradient cliff a hard clamp would introduce.

    **Pick `cap` well ABOVE the median reward**, or the bulk of events saturate
    together and the term stops discriminating between layouts at all.
    """
    return cap * torch.tanh(x / cap)


def U_E(E_preds, E_trues, r, cap=None):
    """Energy term: inverse squared log-energy error, weighted by `r`.

    `cap` soft-caps the per-event reward 1/(err^2 + .01), whose hard ceiling is
    100; None leaves it uncapped. Pick it well above the median — see `_soft_cap`.
    """
    inv_err = 1.0 / ((torch.log10(E_preds) - torch.log10(E_trues)) ** 2 + .01)
    if cap is not None:
        inv_err = _soft_cap(inv_err, cap)
    u = torch.mean(r * inv_err)
    return u


def U_angle(angle_preds, angle_trues, r, cap=None):
    """Angular term: inverse squared angle error, weighted by `r`.

    `cap` soft-caps the per-event reward 1/(err^2 + .001), hard ceiling 1000;
    None leaves it uncapped. **theta and phi need very different caps** — their
    error distributions are not comparable.
    """
    inv_err = 1.0 / ((angle_preds - angle_trues) ** 2 + .001)
    if cap is not None:
        inv_err = _soft_cap(inv_err, cap)
    u = torch.mean(r * inv_err)
    return u
