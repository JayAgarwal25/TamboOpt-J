"""Geometry functions: the ring detector layout generator.

Extracted from SWGOLO7_optimization.ipynb cell 7.
"""

from typing import Union

import torch


def Layouts(n_detectors=100, n_rings=6, radius=500, center=(300,0), device:Union[str, torch.device]='cpu'):
    """Create a detector layout with detectors distributed across concentric rings.

    Parameters:
        n_detectors (int): total number of detectors.
        n_rings (int): number of concentric rings.
        radius (int): radius of the largest ring.
        center (tuple[int, int]): offset of the center in x and y directions.
        device: torch device for output tensors.

    Returns:
        tuple: (x, y) tensors of detector positions in meters.
    """
    R = torch.linspace(5, radius, n_rings, device=device)

    weights = R / R.sum()
    N = torch.round(weights * n_detectors).int()

    diff = n_detectors - N.sum()
    N[-1] += diff

    radii = torch.cat([R[i].repeat(int(n)) for i, n in enumerate(N)])
    angles = torch.cat([
            torch.linspace(0, 2 * torch.pi, int(n) + 1, device=device)[:-1]
            for n in N
        ])
    return center[0] + radii * torch.cos(angles), center[1] + radii * torch.sin(angles)
