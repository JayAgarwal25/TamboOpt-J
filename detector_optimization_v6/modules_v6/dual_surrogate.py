"""Dual-species surrogate: two per-species models combined per physical event.

The electron and muon AllShowers models generate two COMPONENTS of the same
physical shower (the per-species training files are the same simulated events
split by secondary species — see 00_generate_data_dual_species.py). Stage 2
therefore trains two parallel DeepSets surrogates, and everything downstream
(recon training in 03, layout optimization in 04) needs the response of the
COMPLETE event: both models evaluated with the same primary and layout, their
outputs combined.

"Combined" is physical, not elementwise, because the surrogate channels are
log-compressed:

  * E channel = log1p(counts)            (01_build_dataset applies log1p)
  * T channel = log1p(T_phys * T_LOG_SCALE)  (02's log-T target transform)

so the correct combination is in physical space:

  N_tot = N_e + N_mu                       (counts add)
  t_tot = (N_e*t_e + N_mu*t_mu) / N_tot    (count-weighted mean arrival time,
                                            matching the kernel's weighted-avg
                                            T definition)

re-encoded back to the same log channels. Downstream code is unchanged: 04's
`reconstructability(torch.expm1(E_pred))` recovers exactly N_tot, and the
recon input units match what 03 trained on.

The wrapper keeps the single-surrogate call contract `fnn(primary, xy) ->
(B, n_det, 2)`, so stages 3-4 swap it in without touching their inner loops.
Both branches stay in the autograd graph — stage 4's backprop through detector
positions flows through BOTH models.
"""

import os

import torch
import torch.nn as nn

from .constants import N_DETECTORS, PRIMARY_DIM, T_LOG_SCALE
from .deepsets_surrogate import build_surrogate_from_ckpt

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
