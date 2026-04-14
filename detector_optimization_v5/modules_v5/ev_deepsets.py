"""DeepSets reconstruction model for detector_optimization_v5.

v3/v4's Reconstruction is a fully-connected network whose input size is
`num_detectors * input_features` — locked to a fixed detector count (90).
v5 needs to evaluate the same NN on layouts ranging from 10,000 detectors
down to 90, so a fixed-size FC network is a non-starter.

DeepSetsReconstruction solves this with a permutation-invariant set
architecture:

    inputs : (B, N, F)
      │
      ▼ phi  (shared per-detector MLP)
    emb    : (B, N, emb)
      │   (optional multiplicative gate via `mask`)
      ▼ sum pooling over dim=1
    pooled : (B, emb)
      │
      ▼ rho  (post-pool decoder)
    output : (B, 3)    [E_norm, theta_norm, phi_norm]   (Tanh-activated)

Key properties:
1. **N is never baked into the weights.**  phi's first Linear is (F, hidden),
   so the same weights work for N=10000 and N=90.
2. **Permutation-invariant.**  Sum pooling throws away detector ordering.
3. **Mask-gradient hook for saliency.**  Passing a per-detector `mask` tensor
   (init 1.0, `requires_grad=True`) lets the fitness module read
   `∂U/∂mask_i` in a single backward pass — an O(1) proxy for leave-one-out
   marginal contribution.  This is the primary mechanism v5 uses to rank
   detectors for pruning.

Normalization convention (matches v3/v4):
- Inputs are per-feature-normalized OUTSIDE the model using the frozen
  training mean/std: `inputs = (raw - mean) / std`.  Do NOT apply
  normalization inside phi — the frozen statistics must be the ones computed
  on the initial training corpus, not per-batch estimates.
- Output is Tanh-activated to land in [-1, 1], consistent with
  `NormalizeLabels` which maps (E, theta, phi) to (0, 1) then v3's
  Reconstruction squashes to [-1, 1] via Tanh.
"""

import torch
import torch.nn as nn


class DeepSetsReconstruction(nn.Module):
    """Permutation-invariant set-based shower reconstruction network.

    Args:
        input_features : number of features per detector (default 7:
                         [x=N, y=Up, z_cont, N_int, T_int, x0, y0]).
        embedding_dim  : per-detector embedding width (default 64).
        hidden_dim     : hidden width shared by phi and rho (default 128).
        output_dim     : number of regression targets (default 3: E, theta, phi).
    """

    def __init__(
        self,
        input_features: int = 7,
        embedding_dim:  int = 64,
        hidden_dim:     int = 128,
        output_dim:     int = 3,
    ):
        super().__init__()
        self.input_features = input_features
        self.embedding_dim  = embedding_dim

        # Per-detector encoder (shared across all detectors).
        # Applied independently to each (B, i) slice; first Linear is
        # (input_features, hidden_dim), NOT (N * input_features, hidden_dim),
        # which is exactly why the network is detector-count agnostic.
        self.phi = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Post-pooling decoder operating on the global (B, embedding_dim) tensor.
        self.rho = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh(),  # matches v3's Reconstruction output convention
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs : (B, N, F) — per-event, per-detector features.
            mask   : (B, N) or (1, N), float, 1.0 = active, 0.0 = pruned/gated.
                     When `mask.requires_grad` is True, `mask.grad` after
                     backward provides per-detector saliency scores used by
                     ev_selection.compute_detector_fitness.
        Returns:
            (B, output_dim) — normalized predictions in [-1, 1].
        """
        if inputs.dim() != 3:
            raise ValueError(
                f"DeepSetsReconstruction expects inputs of shape (B, N, F); "
                f"got {tuple(inputs.shape)}"
            )
        emb = self.phi(inputs)                          # (B, N, emb)
        if mask is not None:
            # Broadcast mask over the embedding dim.  A multiplicative gate
            # rather than a hard selection: gradients flow through mask.
            emb = emb * mask.unsqueeze(-1)
        pooled = emb.sum(dim=1)                         # (B, emb)  — perm-invariant
        return self.rho(pooled)                         # (B, output_dim)
