"""Layout optimization: detector x, y positions as learnable parameters.

Extracted from SWGOLO7_optimization.ipynb cell 23.
"""

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
