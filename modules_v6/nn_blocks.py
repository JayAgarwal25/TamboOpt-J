"""Shared network building blocks for the v6 models.

Small `nn.Module` factories used by more than one model file, kept here so the
layer sequence is defined exactly once — checkpoints depend on it.
"""

import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int,
         dropout: float = 0.0) -> nn.Sequential:
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
