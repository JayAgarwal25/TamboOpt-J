"""Permutation-equivariant per-detector surrogate. Drop-in for FNNSurrogate.

The response kernel is strictly per-detector-local, `(E_i, T_i) = phi(q, x_i, y_i)`
with no cross-detector term, so a shared per-detector map is the right inductive
bias. A flat MLP had to learn it separately for all 100 slots and approximate
permutation invariance by augmentation; DeepSets is equivariant by construction
(-23% val loss, the only lever that broke the flat-MLP plateau — THEORY.md 10).

    token_i = [q (8), x_i, y_i]              # 10 features per detector
    h_i     = psi(token_i)                   # shared encoder
    c       = mean_i h_i -> context proj     # permutation-INVARIANT pool
    (mu_i, logvar_i) = rho([h_i, c])         # shared decoder, per channel

Heteroscedastic: the decoder emits mu AND logvar per channel. `forward()` returns
the mean only in raw units, preserving FNNSurrogate's contract so Steps 3-4 and
dual_surrogate.py are unchanged; `forward_dist()` adds the logvar for the Gaussian
NLL trainer, letting the model express the aleatoric floor explicitly instead of
collapsing to a conditional mean.

Normalization uses the SAME buffers as FNNSurrogate — in/out mean+std of width
`primary_dim + 2*n_det` (208) and `2*n_det` (200) — so `set_normalization` is
identical and the trainer's in-place log-T stat mutation flows through. Every
xy/E/T slot holds the same stat by construction, so forward reads per-detector
scalars straight out of them. The logvar head stays in z-scored space.
"""

import torch
import torch.nn as nn

from .constants import N_DETECTORS, PRIMARY_DIM


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    """[in→hidden]→(hidden→hidden)×(n_layers-2)→[hidden→out], ReLU + dropout between."""
    assert n_layers >= 2
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepSetsSurrogate(nn.Module):
    """Permutation-equivariant per-detector surrogate. Drop-in for FNNSurrogate.

    Args:
        n_det       : number of detectors (output set size).
        primary_dim : primary encoding width (PRIMARY_DIM = 8: direction, energy,
                      pdg, decay vertex). Never hardcode — checkpoints store it.
        hidden      : per-detector encoder/decoder width.
        context     : pooled-context width (the invariant global summary).
        n_enc       : encoder MLP depth (≥2).
        n_dec       : decoder MLP depth (≥2).
        dropout     : 0.0 recommended — DeepSets is parameter-light and the
                      surrogate underfits, so regularization is counterproductive.
    """

    def __init__(self,
                 n_det:       int = N_DETECTORS,
                 primary_dim: int = PRIMARY_DIM,
                 hidden:      int = 256,
                 context:     int = 64,
                 n_enc:       int = 3,
                 n_dec:       int = 3,
                 dropout:     float = 0.0):
        super().__init__()
        self.n_det       = n_det
        self.primary_dim = primary_dim
        self.hidden      = hidden
        self.context     = context

        token_dim = primary_dim + 2                      # [q, x_i, y_i]
        self.encoder = _mlp(token_dim, hidden, hidden, n_enc, dropout)
        self.context_proj = nn.Linear(hidden, context)
        # 4 outputs per detector: mu_E, mu_T, logvar_E, logvar_T (all z-scored).
        self.decoder = _mlp(hidden + context, hidden, 4, n_dec, dropout)

        # SAME buffer layout as FNNSurrogate so set_normalization is identical
        # and the trainer's log-T stat mutation (out_mean[n_det:]) flows through.
        in_dim  = primary_dim + 2 * n_det
        out_dim = 2 * n_det                              # 200
        self.register_buffer("in_mean",  torch.zeros(in_dim))
        self.register_buffer("in_std",   torch.ones(in_dim))
        self.register_buffer("out_mean", torch.zeros(out_dim))
        self.register_buffer("out_std",  torch.ones(out_dim))

    def set_normalization(self, stats: dict):
        """Identical contract to FNNSurrogate.set_normalization."""
        self.in_mean.copy_(stats["in_mean"])
        self.in_std.copy_(stats["in_std"])
        self.out_mean.copy_(stats["out_mean"])
        self.out_std.copy_(stats["out_std"])

    def _forward_z(self, primary: torch.Tensor, xy: torch.Tensor):
        """Shared encoder/decoder pass, entirely in z-scored space.

        Returns:
            mu_z     : (B, nd, 2) z-scored per-channel mean.
            logvar_z : (B, nd, 2) z-scored per-channel log-variance, clamped
                       to [-10, 10] (var in [4.5e-5, 2.2e4]) for exp() safety.
        """
        B = primary.shape[0]
        nd = self.n_det

        # Per-feature z-score scalars pulled from the broadcast-shared buffers:
        #   in_mean = [primary(5), x0,y0, x1,y1, ...]  → x stat at idx 5, y at 6.
        #   out_mean = [E(100), T(100)]                → E stat at 0, T at n_det.
        p_mean = self.in_mean[:self.primary_dim]                       # (primary_dim,)
        p_std  = self.in_std[:self.primary_dim]
        x_mean, x_std = self.in_mean[self.primary_dim],     self.in_std[self.primary_dim]
        y_mean, y_std = self.in_mean[self.primary_dim + 1], self.in_std[self.primary_dim + 1]

        q_n = (primary - p_mean) / p_std                              # (B, primary_dim)
        q_n = q_n.unsqueeze(1).expand(B, nd, -1)                       # (B, nd, primary_dim)
        x_n = (xy[..., 0] - x_mean) / x_std                           # (B, nd)
        y_n = (xy[..., 1] - y_mean) / y_std
        token = torch.cat([q_n, x_n.unsqueeze(-1), y_n.unsqueeze(-1)], dim=-1)  # (B, nd, 7)

        h = self.encoder(token)                                       # (B, nd, hidden)
        c = self.context_proj(h.mean(dim=1))                          # (B, context)  invariant pool
        c = c.unsqueeze(1).expand(B, nd, -1)                          # (B, nd, context)
        out_n = self.decoder(torch.cat([h, c], dim=-1))              # (B, nd, 4)  z-scored

        mu_z     = out_n[..., :2]                                     # (B, nd, 2): mu_E, mu_T
        logvar_z = out_n[..., 2:].clamp(min=-10.0, max=10.0)          # (B, nd, 2): logvar_E, logvar_T
        return mu_z, logvar_z

    def _unnorm_mean(self, mu_z: torch.Tensor) -> torch.Tensor:
        nd = self.n_det
        E_mean, E_std = self.out_mean[0],  self.out_std[0]
        T_mean, T_std = self.out_mean[nd], self.out_std[nd]
        E_out = mu_z[..., 0] * E_std + E_mean
        T_out = mu_z[..., 1] * T_std + T_mean
        return torch.stack([E_out, T_out], dim=-1)                    # (B, nd, 2)

    def forward(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            primary : (B, primary_dim)
            xy      : (B, n_det, 2)
        Returns:
            (B, n_det, 2) — col0 = E, col1 = T, unnormalized units (mean only;
            drop-in contract for Steps 3-4 and dual_surrogate.py).
        """
        mu_z, _ = self._forward_z(primary, xy)
        return self._unnorm_mean(mu_z)

    def forward_dist(self, primary: torch.Tensor, xy: torch.Tensor):
        """Full predictive distribution, for the Gaussian-NLL trainer.

        Returns:
            mean_raw : (B, n_det, 2) — same as `forward()`, unnormalized units.
            logvar_z : (B, n_det, 2) — z-scored log-variance (dimensionless;
                       compare directly against z-scored residuals).
        """
        mu_z, logvar_z = self._forward_z(primary, xy)
        return self._unnorm_mean(mu_z), logvar_z

    def forward_var(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Raw-unit variance (not z-scored) — for downstream consumers that
        want an explicit per-detector uncertainty channel (recon input,
        dual-species combination), as opposed to forward_dist()'s z-scored
        logvar which is only meant for the training loss.
        """
        nd = self.n_det
        _, logvar_z = self._forward_z(primary, xy)
        E_std, T_std = self.out_std[0], self.out_std[nd]
        var_E = logvar_z[..., 0].exp() * E_std ** 2
        var_T = logvar_z[..., 1].exp() * T_std ** 2
        return torch.stack([var_E, var_T], dim=-1)                    # (B, nd, 2)


def build_surrogate_from_ckpt(ckpt: dict, n_det: int, primary_dim: int, device=None):
    """Construct the right surrogate class from a checkpoint's `config`.

    Lets Steps 3/4 load EITHER a flat-MLP `fnn.pt` (config has no `model_type`,
    or `model_type="fnn"`) OR a DeepSets `fnn.pt` (`model_type="deepsets"`)
    without the caller knowing which. Returns an eval-mode, frozen model with
    normalization already applied from the checkpoint's `norm_stats`.

    Usage (replace the hardcoded `FNNSurrogate(...)` block in 03/04):

        ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
        fnn  = build_surrogate_from_ckpt(ckpt, N_DETECTORS, PRIMARY_DIM, DEVICE)
    """
    from .fnn_surrogate import FNNSurrogate
    cfg = ckpt.get("config", {})
    mtype = cfg.get("model_type", "fnn")
    if mtype == "deepsets":
        model = DeepSetsSurrogate(
            n_det=n_det, primary_dim=primary_dim,
            hidden=int(cfg.get("hidden", 256)),
            context=int(cfg.get("context", 64)),
            n_enc=int(cfg.get("n_enc", 3)),
            n_dec=int(cfg.get("n_dec", 3)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    else:
        model = FNNSurrogate(
            n_det=n_det, primary_dim=primary_dim,
            hidden=int(cfg.get("hidden", 512)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    model.load_state_dict(ckpt["state_dict"])
    if "norm_stats" in ckpt:
        model.set_normalization(ckpt["norm_stats"])
    if device is not None:
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
