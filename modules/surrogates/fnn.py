"""FNN surrogate for detector optimization v6.

The FNN learns `(primary_features, layout) -> (E, T) per detector`. The dataset
it trains on is built by `modules/dataset_builder.py`; this module holds the
primary encoding, the normalization statistics and the model itself.

`FNNSurrogate` is retained so `plots/02_plot_nn_target_vs_pred.py` can still load
pre-DeepSets checkpoints.

Usage (from a driver script living in the v6 folder):

    import modules  # noqa: F401 — package import; keeps modules on the path
    from modules.surrogates import (
        encode_primary, compute_normalization, FNNSurrogate,
    )
"""

import os

import torch
import torch.nn as nn

from ..constants import (N_DETECTORS, PRIMARY_DIM, LOG_E_MIN, LOG_E_MAX)

# ── Primary encoding ─────────────────────────────────────────────────────────

def encode_primary(directions:   torch.Tensor,
                   energies:     torch.Tensor,
                   pdg:          torch.Tensor,
                   positions:    torch.Tensor,
                   array_center: torch.Tensor) -> torch.Tensor:
    """Raw primary encoding: kinematics + decay vertex.

    Returns (N, PRIMARY_DIM):
    ``[dir_x, dir_y, dir_z, log_e_norm, pdg, rel_E, rel_N, rel_U]``, with
    log_e_norm = (log10(E) - LOG_E_MIN) / (LOG_E_MAX - LOG_E_MIN) ∈ [0, 1] and
    rel_* the decay vertex relative to `array_center` in metres (left unscaled —
    `compute_normalization` z-scores every primary column).

    The vertex is included because tau_wholesky.jl aims every surviving tau at the
    array, so direction alone barely discriminates; without it two taus with equal
    direction and energy give different labels from identical input. Measured
    aleatoric floor: R² >= 0.49 without, >= 0.56 with. Derived summaries (along-axis
    distance, impact parameter) were tested and did not beat the raw triple.

    Cols 0-3 keep their meaning, so Step 3's ``primary[:, :4]`` target is unaffected.

    Args:
        directions   : (N, 3) unit vectors (sin θ cos φ, sin θ sin φ, cos θ).
        energies     : (N,) or (N, 1) primary energies [GeV], range ~[1e5, 1e8].
        pdg          : (N,) EM/hadronic primary class ids (0 or 1) — NOT the e/µ species.
        positions    : (N, 3) ENU decay vertices, from the Step-0 `_positions.pt` sidecar.
        array_center : (3,) ENU centre of the detector region. Passed in, not global,
                       so the encoding cannot go stale when the mesh changes.
    """
    dirs = torch.as_tensor(directions, dtype=torch.float32)
    eng  = torch.as_tensor(energies,   dtype=torch.float32).reshape(-1, 1)
    pdg  = torch.as_tensor(pdg,        dtype=torch.float32).reshape(-1, 1)
    pos  = torch.as_tensor(positions,  dtype=torch.float32)
    ctr  = torch.as_tensor(array_center, dtype=torch.float32).reshape(1, 3)
    log_e = torch.log10(eng)
    log_e_norm = (log_e - LOG_E_MIN) / (LOG_E_MAX - LOG_E_MIN)

    rel = pos - ctr                                       # decay → array-centre frame
    return torch.cat([dirs, log_e_norm, pdg, rel], dim=1)



def _species_sidecar_path(shower_cache_path: str) -> str:
    """Path of the Step-0 e/µ species sidecar paired with a dual corpus .pt
    (`…_dual_<2N>.pt` -> `…_dual_<2N>_species.pt`). Written by
    00_generate_data_dual_species.py; row-aligned with the corpus."""
    base, ext = os.path.splitext(shower_cache_path)
    return base + "_species" + ext


def _load_species_sidecar(shower_cache_path: str, keep_idx) -> torch.Tensor:
    """Load the Step-0 e/µ species sidecar (0=electron, 1=muon) and index it by
    `keep_idx` (the same rows used for the corpus metadata). Raises if missing —
    regenerate the corpus with the updated Step 0 (which writes the sidecar)."""
    path = _species_sidecar_path(shower_cache_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"species sidecar not found: {path}\n"
            f"Regenerate the corpus with the updated 00_generate_data_dual_species.py "
            f"(it writes the e/µ species sidecar; the corpus `pdg` is now the EM/hadronic class).")
    return torch.load(path)[torch.as_tensor(keep_idx)].long()


def compute_normalization(primary:   torch.Tensor,
                          xy:        torch.Tensor,
                          E:         torch.Tensor,
                          T:         torch.Tensor) -> dict:
    """Per-feature z-score statistics for inputs and outputs.

    Input vector layout (for the FNN) is `[primary (PRIMARY_DIM), xy_flat (2*n_det)]`.
    Output vector layout is `[E (100), T (100)]` = 200 (first E then T).

    For the input, primary features keep per-feature stats, but all xy
    slots share one (mean_x, std_x, mean_y, std_y) pair — otherwise
    permutation augmentation creates a train/val z-score mismatch.
    Likewise the output uses one shared (mean, std) for all E slots and
    one for all T slots.
    """
    n_det = E.shape[1]

    # ── Input: primary per-feature, xy shared across detectors ──────────
    p_mean = primary.mean(dim=0)                                    # (PRIMARY_DIM,)
    p_std  = primary.std(dim=0).clamp(min=1e-10)                     # (PRIMARY_DIM,)

    # One scalar mean/std for x coords, one for y coords
    xy_x = xy[..., 0]                                               # (B, n_det)
    xy_y = xy[..., 1]                                               # (B, n_det)
    x_mean = xy_x.mean();  x_std = xy_x.std()
    y_mean = xy_y.mean();  y_std = xy_y.std()

    # Broadcast to the full (205,) layout: [primary(5), x0,y0,x1,y1,...,x99,y99]
    xy_mean = torch.stack([x_mean.expand(n_det),
                           y_mean.expand(n_det)], dim=-1).reshape(-1)   # (200,)
    xy_std  = torch.stack([x_std.expand(n_det),
                           y_std.expand(n_det)], dim=-1).reshape(-1)    # (200,)

    in_mean = torch.cat([p_mean, xy_mean])                          # (205,)
    in_std  = torch.cat([p_std,  xy_std])                           # (205,)

    # ── Output: one shared stat for all E slots, one for all T slots ────
    E_mean = E.mean();  E_std = E.std()
    T_mean = T.mean();  T_std = T.std()

    out_mean = torch.cat([E_mean.expand(n_det),
                          T_mean.expand(n_det)])                    # (200,)
    out_std  = torch.cat([E_std.expand(n_det),
                          T_std.expand(n_det)])                     # (200,)

    return dict(
        in_mean=in_mean,   in_std=in_std,
        out_mean=out_mean, out_std=out_std,
    )


# ── FNN model ────────────────────────────────────────────────────────────────

class FNNSurrogate(nn.Module):
    """Flat MLP that maps (primary, layout) → (E, T) per detector.

    Input:  [primary (5) + xy_flat (2·100)] = 205 features.
    Output: [E (100), T (100)] = 200 features, reshaped to (100, 2).
    Hidden: 512 → 512 → 512 (ReLU + dropout).
    Z-scoring is baked into the forward pass via registered buffers so the
    same model can be used for training and optimization without any extra
    normalization plumbing at call sites.
    """

    def __init__(self,
                 n_det:       int = N_DETECTORS,
                 primary_dim: int = PRIMARY_DIM,
                 hidden:      int = 512,
                 dropout:     float = 0.1):
        super().__init__()
        self.n_det       = n_det
        self.primary_dim = primary_dim

        in_dim  = primary_dim + 2 * n_det
        out_dim = 2 * n_det

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

        self.register_buffer("in_mean",  torch.zeros(in_dim))
        self.register_buffer("in_std",   torch.ones(in_dim))
        self.register_buffer("out_mean", torch.zeros(out_dim))
        self.register_buffer("out_std",  torch.ones(out_dim))

    def set_normalization(self, stats: dict):
        """Copy z-score buffers from `compute_normalization()` output."""
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
            (B, n_det, 2) — column 0 = E, column 1 = T, in unnormalized units.
        """
        B = primary.shape[0]
        flat   = torch.cat([primary, xy.reshape(B, -1)], dim=1)    # (B, 205)
        flat_n = (flat - self.in_mean) / self.in_std
        out_n  = self.net(flat_n)                                   # (B, 200)
        out    = out_n * self.out_std + self.out_mean

        E_out = out[:, :self.n_det]
        T_out = out[:, self.n_det:]
        return torch.stack([E_out, T_out], dim=-1)                 # (B, n_det, 2)
