"""DeepSets surrogate for detector optimization v6 (drop-in for FNNSurrogate).

Motivation (see THEORY.md §10): the true detector-response kernel is strictly
per-detector-local — `(E_i, T_i) = φ(q, x_i, y_i)` with no cross-detector term
(modules_v4/tr_plane_kernel.py). The production flat MLP must learn this same
local map separately for all 100 input slots and be made permutation-invariant
to 100! orderings via expensive augmentation; its minimum-loss fixed point is
predict-the-mean. The architecture search found a per-detector **DeepSets**
model — the only lever that broke the flat-MLP plateau (−23%).

This module is permutation-EQUIVARIANT by construction, so the trainer can drop
the permutation augmentation entirely. It preserves FNNSurrogate's call contract
exactly:

    model = DeepSetsSurrogate(n_det=100, primary_dim=5, ...)
    model.set_normalization(stats)          # same stats dict as compute_normalization()
    et = model(primary, xy)                 # (B, n_det, 2): col0=E, col1=T, raw units

so it is a literal drop-in for Steps 3 and 4 (which only ever call
`fnn(primary, xy)` and `set_normalization`).

Architecture (classic DeepSets: shared per-detector encoder → pooled context →
shared per-detector decoder):

    token_i = [q (5), x_i, y_i]                       # 7 features, per detector
    h_i     = ψ(token_i)                              # shared encoder MLP
    c       = mean_i h_i  → context projection        # permutation-INVARIANT pool
    (E_i, T_i) = ρ([h_i, c])                          # shared decoder MLP

Z-scoring is baked into forward via the SAME registered buffers FNNSurrogate
uses (in_mean(205), in_std(205), out_mean(200), out_std(200)), so
`set_normalization` is byte-identical and the trainer's in-place log-T stat
mutation (out_mean[n_det:] / out_std[n_det:]) flows through unchanged. In
forward we read the per-detector scalars out of those broadcast-shared buffers
(every xy/E/T slot holds the same stat by construction of compute_normalization).
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
        primary_dim : primary encoding width (5).
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
                 dropout:     float = 0.0,
                 pool:        str = "mean"):
        super().__init__()
        self.n_det       = n_det
        self.primary_dim = primary_dim
        self.hidden      = hidden
        self.context     = context
        self.pool        = pool  # "mean" or "maxmean"

        token_dim = primary_dim + 2                      # [q, x_i, y_i]
        pool_dim  = hidden * 2 if pool == "maxmean" else hidden
        self.encoder = _mlp(token_dim, hidden, hidden, n_enc, dropout)
        self.context_proj = nn.Linear(pool_dim, context)
        self.decoder = _mlp(hidden + context, hidden, 2, n_dec, dropout)

        # SAME buffer layout as FNNSurrogate so set_normalization is identical
        # and the trainer's log-T stat mutation (out_mean[n_det:]) flows through.
        in_dim  = primary_dim + 2 * n_det                # 205
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

    def forward(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            primary : (B, primary_dim)
            xy      : (B, n_det, 2)
        Returns:
            (B, n_det, 2) — col0 = E, col1 = T, unnormalized units.
        """
        B = primary.shape[0]
        nd = self.n_det

        # Per-feature z-score scalars pulled from the broadcast-shared buffers:
        #   in_mean = [primary(5), x0,y0, x1,y1, ...]  → x stat at idx 5, y at 6.
        #   out_mean = [E(100), T(100)]                → E stat at 0, T at n_det.
        p_mean = self.in_mean[:self.primary_dim]                       # (5,)
        p_std  = self.in_std[:self.primary_dim]
        x_mean, x_std = self.in_mean[self.primary_dim],     self.in_std[self.primary_dim]
        y_mean, y_std = self.in_mean[self.primary_dim + 1], self.in_std[self.primary_dim + 1]
        E_mean, E_std = self.out_mean[0],  self.out_std[0]
        T_mean, T_std = self.out_mean[nd], self.out_std[nd]

        q_n = (primary - p_mean) / p_std                              # (B, 5)
        q_n = q_n.unsqueeze(1).expand(B, nd, -1)                       # (B, nd, 5)
        x_n = (xy[..., 0] - x_mean) / x_std                           # (B, nd)
        y_n = (xy[..., 1] - y_mean) / y_std
        token = torch.cat([q_n, x_n.unsqueeze(-1), y_n.unsqueeze(-1)], dim=-1)  # (B, nd, 7)

        h = self.encoder(token)                                       # (B, nd, hidden)
        if self.pool == "maxmean":
            pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)  # (B, 2*hidden)
        else:
            pooled = h.mean(dim=1)                                     # (B, hidden)
        c = self.context_proj(pooled)                                  # (B, context)  invariant pool
        c = c.unsqueeze(1).expand(B, nd, -1)                          # (B, nd, context)
        out_n = self.decoder(torch.cat([h, c], dim=-1))              # (B, nd, 2)  z-scored

        E_out = out_n[..., 0] * E_std + E_mean                        # (B, nd)
        T_out = out_n[..., 1] * T_std + T_mean
        return torch.stack([E_out, T_out], dim=-1)                    # (B, nd, 2)


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
            pool=cfg.get("pool", "mean"),
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
