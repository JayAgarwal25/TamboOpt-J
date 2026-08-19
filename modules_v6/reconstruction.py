"""Reconstruction MLP for v6 — FNN-parity architecture and normalization.

Maps flattened per-detector features `(x, y, E_pred, T_pred)` to the v6 primary
encoding `[dir_x, dir_y, dir_z, log_e_norm]`.

Mirrors `FNNSurrogate` (`modules_v6/fnn_surrogate.py`) one-for-one:
    - hidden 512 → 512 → 512 (ReLU + dropout),
    - per-feature z-score of the input baked into `forward()` via registered buffers,
    - output denormalization (back to raw primary-encoding units) also baked in,
    - same `set_normalization()` contract, same buffer names.

Usage:

    from modules_v6.reconstruction import Reconstruction
    recon = Reconstruction(n_det=100, input_features=4, output_dim=4)
    recon.set_normalization(in_mean, in_std, out_mean, out_std)
    pred = recon(inp_flat)   # raw primary-encoding units
"""

import torch
import torch.nn as nn

from .constants import N_DETECTORS


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int,
         dropout: float = 0.0) -> nn.Sequential:
    """[in→hidden]→(hidden→hidden)×(n_layers-2)→[hidden→out], ReLU between."""
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


class Reconstruction(nn.Module):
    """Flat MLP: (layout + predicted E/T) → primary encoding.

    Input layout (flattened row-major per sample, length `n_det * input_features`):
        [x_0, y_0, E_0, T_0, x_1, y_1, E_1, T_1, ...]
    Output (length `output_dim`, typically 4):
        [dir_x, dir_y, dir_z, log_e_norm]   (v6 primary encoding)

    Normalization buffers are initialized to identity (zeros/ones) and populated
    via `set_normalization(...)`. The FNN pattern is followed exactly: input is
    z-scored on the way in, output is de-z-scored on the way out, so the forward
    pass works in raw units from the caller's perspective.
    """

    def __init__(self,
                 n_det:          int = N_DETECTORS,
                 input_features: int = 4,
                 output_dim:     int = 4,
                 hidden:         int = 512,
                 dropout:        float = 0.1):
        super().__init__()
        self.n_det          = n_det
        self.input_features = input_features
        self.output_dim     = output_dim

        in_dim = n_det * input_features

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

        self.register_buffer("in_mean",  torch.zeros(in_dim))
        self.register_buffer("in_std",   torch.ones(in_dim))
        self.register_buffer("out_mean", torch.zeros(output_dim))
        self.register_buffer("out_std",  torch.ones(output_dim))

    def set_normalization(self,
                          in_mean:  torch.Tensor,
                          in_std:   torch.Tensor,
                          out_mean: torch.Tensor,
                          out_std:  torch.Tensor) -> None:
        """Copy z-score buffers. Shapes must match the registered buffers."""
        self.in_mean.copy_(in_mean)
        self.in_std.copy_(in_std)
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(out_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: `x` of shape `(B, n_det * input_features)` in raw units.
        Returns: `(B, output_dim)` in raw primary-encoding units.
        """
        x_n   = (x - self.in_mean) / self.in_std
        out_n = self.net(x_n)
        return out_n * self.out_std + self.out_mean


class DeepSetsRecon(nn.Module):
    """Permutation-invariant set→4-vector reconstruction.

    token_i = [x_i, y_i, E_i, T_i]         per-detector features
    h_i     = encoder(token_i)              shared encoder
    c       = context_proj(mean_i h_i)      invariant aggregate
    out     = decoder(c)                    single global decode → primary encoding

    Input: (B, n_det, input_features) in raw units.
    """

    def __init__(self,
                 n_det:          int = N_DETECTORS,
                 input_features: int = 4,
                 output_dim:     int = 4,
                 hidden:         int = 256,
                 context:        int = 128,
                 n_enc:          int = 3,
                 n_dec:          int = 2,
                 pool:           str = "mean"):
        super().__init__()
        self.n_det          = n_det
        self.input_features = input_features
        self.output_dim     = output_dim
        self.pool           = pool

        pool_dim = hidden * 2 if pool == "maxmean" else hidden
        self.encoder      = _mlp(input_features, hidden, hidden, n_enc)
        self.context_proj = nn.Linear(pool_dim, context)
        self.decoder      = _mlp(context, hidden, output_dim, n_dec)

        self.register_buffer("in_mean",  torch.zeros(input_features))
        self.register_buffer("in_std",   torch.ones(input_features))
        self.register_buffer("out_mean", torch.zeros(output_dim))
        self.register_buffer("out_std",  torch.ones(output_dim))

    def set_normalization(self,
                          in_mean:  torch.Tensor,
                          in_std:   torch.Tensor,
                          out_mean: torch.Tensor,
                          out_std:  torch.Tensor) -> None:
        self.in_mean.copy_(in_mean)
        self.in_std.copy_(in_std)
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(out_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, n_det, input_features) in raw units.
        Returns: (B, output_dim) in raw primary-encoding units.
        """
        x_n   = (x - self.in_mean) / self.in_std        # broadcast (input_features,)
        h     = self.encoder(x_n)                        # (B, n_det, hidden)
        if self.pool == "maxmean":
            pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)  # (B, 2*hidden)
        else:
            pooled = h.mean(dim=1)                       # (B, hidden)
        c     = self.context_proj(pooled)                # (B, context)  invariant pool
        out_n = self.decoder(c)                          # (B, output_dim)
        return out_n * self.out_std + self.out_mean


def build_recon_from_ckpt(ckpt: dict, n_det: int, device=None):
    """Polymorphic recon loader — dispatches on ckpt["config"]["model_type"].

    Returns a frozen eval-mode model with normalization applied.
    Handles both "mlp" (Reconstruction) and "deepsets" (DeepSetsRecon).
    """
    cfg   = ckpt.get("config", {})
    mtype = cfg.get("model_type", "mlp")
    n_det_ckpt      = int(ckpt.get("num_detectors", n_det))
    input_features  = int(ckpt.get("input_features", 4))

    if mtype == "deepsets":
        model = DeepSetsRecon(
            n_det=n_det_ckpt,
            input_features=input_features,
            hidden=int(cfg.get("hidden", 256)),
            context=int(cfg.get("context", 128)),
            n_enc=int(cfg.get("n_enc", 3)),
            n_dec=int(cfg.get("n_dec", 2)),
            pool=cfg.get("pool", "mean"),
        )
    else:
        model = Reconstruction(
            n_det=n_det_ckpt,
            input_features=input_features,
            hidden=int(cfg.get("hidden", 512)),
            dropout=float(cfg.get("dropout", 0.1)),
        )

    model.load_state_dict(ckpt["state_dict"])
    model.set_normalization(
        in_mean  = ckpt["input_mean"],
        in_std   = ckpt["input_std"],
        out_mean = ckpt["target_mean"],
        out_std  = ckpt["target_std"],
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model