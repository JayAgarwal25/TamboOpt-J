"""DeepSets binary DETECTION classifier for v6.

The FNN/DeepSets surrogates (deepsets_surrogate.py) regress the PER-DETECTOR
response (E, T) of an event that already fired. Nothing upstream answers the
simpler, prior question the collaborator actually asked for: given a primary
and a candidate layout, will the event be seen AT ALL? This module is a
single differentiable P(detect | primary, layout) surrogate: a classifier,
not a regressor, meant to plug the missing aperture term into the project's
utility (which today only scores reconstruction quality conditioned on
already-fired events).

"Detected" is a design choice made by the trainer (02b_train_detection_
classifier.py), not by this module: at least N_MIN detectors registering
more than E_MIN raw counts. This module only defines the architecture that
predicts P(detected).

Architecture (same DeepSets shape as DeepSetsSurrogate/DeepSetsRecon, see
those files' docstrings for why permutation invariance matters for a
layout with no canonical detector ordering):

    token_i = [q (primary_dim), x_i, y_i]        per-detector, z-scored
    h_i     = encoder(token_i)                    shared per-detector MLP
    pooled  = [mean_i h_i, max_i h_i]              permutation-INVARIANT pool
    c       = context_proj(pooled)                 (B, context)
    logit   = decoder(c)                            (B, 1) -> squeeze -> (B,)

Unlike DeepSetsSurrogate's decoder (which re-attaches the per-detector token
to predict a per-detector output), this decoder only ever sees the pooled
context: the target is a single event-level yes/no, not one value per
detector, so there is nothing per-detector left to decode once pooling has
collapsed the set. This mirrors DeepSetsRecon's context-only decoder more
than DeepSetsSurrogate's.

Normalization buffers follow the FNNSurrogate/DeepSetsSurrogate convention
exactly (in_mean/in_std of length primary_dim + 2*n_det, primary features
kept per-feature, all x slots sharing one stat and all y slots sharing
another so permutation leaves the z-score unchanged) so that
`set_normalization` accepts the same stats dict shape produced by
`modules_v6.fnn_surrogate.compute_normalization` (only the "in_mean"/
"in_std" keys are read here, since a classifier has no z-scored physical
output channel; out_mean/out_std, if present in the dict, are ignored).
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


class DetectionClassifier(nn.Module):
    """Permutation-invariant P(detect | primary, layout) surrogate.

    Args:
        n_det       : number of detectors (input set size).
        primary_dim : primary encoding width (see modules_v6/constants.py
                      PRIMARY_DIM). Never hardcode, checkpoints store it.
        hidden      : per-detector encoder width / decoder width.
        context     : pooled-context width (the invariant global summary).
        n_enc       : encoder MLP depth (>= 2).
        n_dec       : decoder MLP depth (>= 2).
        dropout     : 0.0 recommended, matching the surrogate/recon default;
                      these DeepSets models are parameter-light enough that
                      regularization tends to hurt more than it helps.
        pool        : "mean" or "maxmean" (maxmean default: mean+max pooling
                      carries more information than mean alone, e.g. "was
                      there ANY nearby detector" vs "average distance").
    """

    def __init__(self,
                 n_det:       int = N_DETECTORS,
                 primary_dim: int = PRIMARY_DIM,
                 hidden:      int = 128,
                 context:     int = 64,
                 n_enc:       int = 3,
                 n_dec:       int = 3,
                 dropout:     float = 0.0,
                 pool:        str = "maxmean"):
        super().__init__()
        self.n_det       = n_det
        self.primary_dim = primary_dim
        self.hidden      = hidden
        self.context     = context
        self.n_enc       = n_enc
        self.n_dec       = n_dec
        self.pool        = pool
        self.dropout     = dropout

        token_dim = primary_dim + 2                      # [q, x_i, y_i]
        pool_dim  = hidden * 2 if pool == "maxmean" else hidden
        self.encoder = _mlp(token_dim, hidden, hidden, n_enc, dropout)
        self.context_proj = nn.Linear(pool_dim, context)
        # Single event-level logit, decoded from the pooled context only
        # (see the module docstring for why this differs from
        # DeepSetsSurrogate's per-detector decoder).
        self.decoder = _mlp(context, hidden, 1, n_dec, dropout)

        # Same buffer layout as FNNSurrogate/DeepSetsSurrogate's INPUT half
        # so set_normalization accepts compute_normalization's dict as-is.
        in_dim = primary_dim + 2 * n_det
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std",  torch.ones(in_dim))

    def set_normalization(self, stats: dict):
        """Accepts the same stats dict shape as
        modules_v6.fnn_surrogate.compute_normalization produces. Only
        "in_mean"/"in_std" are consumed; any "out_mean"/"out_std" keys
        present (from a regression-oriented stats dict) are ignored since
        this model has no physical-unit output to un-normalize.
        """
        self.in_mean.copy_(stats["in_mean"])
        self.in_std.copy_(stats["in_std"])

    def _token(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Build the per-detector z-scored token, same construction as
        DeepSetsSurrogate._forward_z (see that file for why the x/y stats
        are pulled from fixed buffer offsets rather than per-slot)."""
        B = primary.shape[0]
        n_active = xy.shape[1]   # actual detector count for this call

        p_mean = self.in_mean[:self.primary_dim]
        p_std  = self.in_std[:self.primary_dim]
        x_mean, x_std = self.in_mean[self.primary_dim],     self.in_std[self.primary_dim]
        y_mean, y_std = self.in_mean[self.primary_dim + 1], self.in_std[self.primary_dim + 1]

        q_n = (primary - p_mean) / p_std                                  # (B, primary_dim)
        q_n = q_n.unsqueeze(1).expand(B, n_active, -1)                    # (B, n_active, primary_dim)
        x_n = (xy[..., 0] - x_mean) / x_std                               # (B, n_active)
        y_n = (xy[..., 1] - y_mean) / y_std
        return torch.cat([q_n, x_n.unsqueeze(-1), y_n.unsqueeze(-1)], dim=-1)  # (B, n_active, token_dim)

    def forward(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            primary : (B, primary_dim)
            xy      : (B, n_det, 2)
        Returns:
            (B,) raw LOGITS, not probabilities.

        Returning a logit (paired with BCEWithLogitsLoss at train time) is
        the numerically stable two-class form of the softmax the original
        request described ("detecting yes vs no ... like a softmax"): for
        two classes, a softmax over logits [z_yes, z_no] is exactly
        sigmoid(z_yes - z_no), i.e. a single logit z = z_yes - z_no. Emitting
        two logits and applying softmax would be an equivalent but strictly
        redundant parameterization (one extra degree of freedom that never
        affects the predicted probability, since softmax is shift-invariant).
        BCEWithLogitsLoss on the single logit is the same model with the
        redundancy removed, and it fuses the sigmoid with the loss for
        better numerical stability than sigmoid() followed by log() would
        give separately.
        """
        token = self._token(primary, xy)                    # (B, n_active, token_dim)
        h = self.encoder(token)                              # (B, n_active, hidden)
        if self.pool == "maxmean":
            pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)  # (B, 2*hidden)
        else:
            pooled = h.mean(dim=1)                            # (B, hidden)
        c = self.context_proj(pooled)                         # (B, context)
        logit = self.decoder(c).squeeze(-1)                   # (B,)
        return logit

    def predict_proba(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """P(detected) in (0, 1), the sigmoid of forward()'s logit. Kept as a
        separate method (rather than folding sigmoid into forward) so the
        trainer can call BCEWithLogitsLoss directly on the raw logit, which
        is more numerically stable than sigmoid + BCELoss."""
        return torch.sigmoid(self.forward(primary, xy))


def build_detection_classifier_from_ckpt(ckpt: dict, n_det: int, primary_dim: int, device=None):
    """Construct a DetectionClassifier from a checkpoint's `config`, mirroring
    modules_v6.deepsets_surrogate.build_surrogate_from_ckpt's contract:
    returns an eval-mode, frozen model with normalization already applied
    from the checkpoint's `norm_stats`.

    Usage:

        ckpt = torch.load(os.path.join(OUTPUT_FOLDER, "detection_classifier.pt"),
                          map_location=DEVICE)
        clf = build_detection_classifier_from_ckpt(ckpt, N_DETECTORS, PRIMARY_DIM, DEVICE)
        p_detect = clf.predict_proba(primary, xy)
    """
    cfg = ckpt.get("config", {})
    model = DetectionClassifier(
        n_det=n_det, primary_dim=primary_dim,
        hidden=int(cfg.get("hidden", 128)),
        context=int(cfg.get("context", 64)),
        n_enc=int(cfg.get("n_enc", 3)),
        n_dec=int(cfg.get("n_dec", 3)),
        dropout=float(cfg.get("dropout", 0.0)),
        pool=cfg.get("pool", "maxmean"),
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
