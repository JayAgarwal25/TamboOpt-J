"""Layout optimization: learnable positions, separation enforcement, symmetry loss.

Extracted from SWGOLO7_optimization.ipynb cells 23, 74, 75.
"""

import numpy as np
import torch
from torch import device as TorchDevice
from typing import Union

class LearnableXY(torch.nn.Module):
    """Small module holding detector x, y positions as learnable parameters.

    The parameters can be optimized with standard PyTorch optimizers to change the layout.
    """
    def __init__(self, x_init, y_init, device:Union[str, TorchDevice]='cpu'):
        super().__init__()
        self.x = torch.nn.Parameter(x_init.to(device))
        self.y = torch.nn.Parameter(y_init.to(device))

    def forward(self):
        """Return current learnable coordinates as (x, y)."""
        return self.x, self.y


def push_apart(module, min_dist):
    """Enforce a minimum separation between detectors by small pairwise displacements.

    Parameters:
        module (LearnableXY): module exposing `x` and `y` parameters via forward().
        min_dist (float): desired minimum separation between detectors.
    """
    x, y = module()
    coords = torch.stack([x, y], dim=1)

    with torch.no_grad():
        for i in range(coords.shape[0]):
            diffs = coords[i] - coords
            dists = torch.norm(diffs, dim=1)
            mask = (dists < min_dist) & (dists > 0)

            for j in torch.where(mask)[0]:
                direction = diffs[j] / dists[j]
                displacement = 0.5 * (min_dist - dists[j]) * direction
                coords[i] += displacement
                coords[j] -= displacement

        module.x.data.copy_(coords[:, 0])
        module.y.data.copy_(coords[:, 1])


def symmetry_loss(x, y, n_symmetry=3, center=(0.0, 0.0)):
    """Compute a loss that penalizes deviation from n-fold rotational symmetry.

    Parameters:
        x, y (torch.Tensor): coordinates of detectors.
        n_symmetry (int): order of rotational symmetry to enforce.
        center (tuple): center point to rotate about.

    Returns:
        torch.Tensor: scalar symmetry loss (smaller is more symmetric).
    """
    x_centered = x - center[0]
    y_centered = y - center[1]
    coords = torch.stack([x_centered, y_centered], dim=1)

    sym_loss = 0.0
    for i in range(1, n_symmetry):
        theta = 2 * np.pi * i / n_symmetry
        R = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=coords.dtype, device=coords.device)

        rotated = coords @ R.T

        dists = torch.cdist(rotated, coords, p=2)
        min_dists, _ = dists.min(dim=1)

        sym_loss += min_dists.mean()

    return sym_loss / (n_symmetry - 1)
