"""Dual-species surrogate: the e and mu models combined into one physical event.

The per-species files are the same simulated events split by secondary species,
so a physical event needs both models run on the same primary and layout.

Combination happens in PHYSICAL space, not elementwise, because both channels
are log-compressed (E = log1p(counts), T = log1p(T_phys * T_LOG_SCALE)):

    N_tot = N_e + N_mu                        counts add
    t_tot = (N_e*t_e + N_mu*t_mu) / N_tot     count-weighted, as the kernel defines T

then re-encoded to the same log channels. Keeps the single-surrogate contract
`fnn(primary, xy) -> (B, n_det, 2)`, and both branches stay in the autograd
graph so Step 4 backprops through both. See docs/THEORY.md §3.6 and §5.6.
"""

import os

import torch
import torch.nn as nn

from ..constants import N_DETECTORS, PRIMARY_DIM, T_LOG_SCALE
from .deepsets import build_surrogate_from_ckpt

ELECTRON_CKPT = "fnn_electron.pt"
MUON_CKPT     = "fnn_muon.pt"


def combine_species_outputs(pred_e: torch.Tensor,
                            pred_mu: torch.Tensor) -> torch.Tensor:
    """Physically combine per-species (B, n_det, 2) predictions into one event.

    Differentiable everywhere; negative model outputs are clamped to zero
    counts / zero time before combining (a detector with no predicted signal
    contributes nothing, matching the kernel's behavior on empty clouds).
    """
    n_e  = torch.expm1(pred_e[..., 0]).clamp(min=0.0)        # counts, electron
    n_mu = torch.expm1(pred_mu[..., 0]).clamp(min=0.0)       # counts, muon
    t_e  = torch.expm1(pred_e[..., 1]).clamp(min=0.0) / T_LOG_SCALE
    t_mu = torch.expm1(pred_mu[..., 1]).clamp(min=0.0) / T_LOG_SCALE

    n_tot = n_e + n_mu
    t_tot = (n_e * t_e + n_mu * t_mu) / n_tot.clamp(min=1e-12)

    E_out = torch.log1p(n_tot)
    T_out = torch.log1p(t_tot * T_LOG_SCALE)
    return torch.stack([E_out, T_out], dim=-1)


class DualSpeciesSurrogate(nn.Module):
    """Two frozen per-species surrogates behind the single-surrogate contract.

    forward(primary, xy) evaluates BOTH per-species models on the SAME primary
    (whose pdg feature is the real EM/hadronic class each model was trained on)
    and combines their outputs physically — a primary describes one complete
    event, and both the electron and muon components are always part of it.
    Routing is by model identity (electron vs muon), not by the pdg feature.
    """

    def __init__(self, electron: nn.Module, muon: nn.Module):
        super().__init__()
        self.electron = electron
        self.muon     = muon
        self.n_det    = getattr(electron, "n_det", N_DETECTORS)

    def forward(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            primary : (B, PRIMARY_DIM) — passed unchanged to both models; its
                      pdg feature is the EM/hadronic class both were trained on.
            xy      : (B, n_det, 2) — shared layout, stays in the autograd graph
                      of BOTH branches.
        Returns:
            (B, n_det, 2) combined event response — col 0 = log1p(N_tot),
            col 1 = log1p(t_tot * T_LOG_SCALE).
        """
        pred_e  = self.electron(primary, xy)
        pred_mu = self.muon(primary, xy)
        return combine_species_outputs(pred_e, pred_mu)

    def forward_with_var(self, primary: torch.Tensor, xy: torch.Tensor):
        """(mean, var) — mean is identical to forward(). var is an
        approximate combination: electron + muon raw-unit variances summed
        (independent noise sources); the two components' physical
        combination (count-weighted average, log1p) is nonlinear, so this
        is not a full delta-method propagation, just a reasonable per-
        detector uncertainty signal for recon/optimizer consumption.
        """
        mean   = self.forward(primary, xy)
        var_e  = self.electron.forward_var(primary, xy)
        var_mu = self.muon.forward_var(primary, xy)
        return mean, var_e + var_mu

    def forward_sample(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """One stochastic draw from the predicted (mean, var) distribution,
        same (B, n_det, 2) contract as forward() — a fresh noisy realization
        each call instead of the mean point estimate, so downstream
        training/optimization sees the surrogate's learned aleatoric spread
        directly rather than being handed mean and variance as separate,
        discardable inputs. Reparameterized (mean + eps*std) so gradients
        into (primary, xy) still flow for stage-4's L-BFGS/Adam.
        """
        mean, var = self.forward_with_var(primary, xy)
        eps = torch.randn_like(mean)
        return mean + eps * var.clamp(min=0.0).sqrt()


def load_dual_surrogate(folder: str,
                        device: torch.device,
                        n_det: int = N_DETECTORS,
                        primary_dim: int = PRIMARY_DIM) -> DualSpeciesSurrogate:
    """Load fnn_electron.pt + fnn_muon.pt from `folder` into a frozen wrapper.

    Each checkpoint is built via `build_surrogate_from_ckpt` (flat-MLP or
    DeepSets, chosen by its saved config), gets its own norm stats from the
    checkpoint, and is frozen in eval mode.
    """
    models = {}
    for tag, fname in (("electron", ELECTRON_CKPT), ("muon", MUON_CKPT)):
        path = os.path.join(folder, fname)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        models[tag] = build_surrogate_from_ckpt(ckpt, n_det, primary_dim, device)
        cfg = ckpt.get("config", {})
        print(f"[load] {fname}  model={cfg.get('model_type', 'fnn')}  "
              f"epoch={ckpt.get('epoch', '?')}  val={ckpt.get('val_total', '?')}")
    dual = DualSpeciesSurrogate(models["electron"], models["muon"]).to(device)
    dual.eval()
    return dual
