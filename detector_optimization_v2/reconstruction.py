"""Neural network reconstruction model, label normalization, and early stopping.

Extracted from SWGOLO7_optimization.ipynb cells 30, 31, 32.
"""

import numpy as np
import torch
import torch.nn as nn


class Reconstruction(nn.Module):
    """Fully-connected neural network to reconstruct shower properties from flattened detector inputs.

    Inputs (per event) are flattened vectors of length `num_detectors * input_features` where for
    each detector the features are `[x, y, N, T]`.

    Outputs are a vector of length `output_dim` with `[X0, Y0, E_norm, Theta_norm, Phi_norm]`.
    """
    def __init__(self, input_features=6, num_detectors=90, hidden_lay1=256, hidden_lay2=128,
                 hidden_lay3=32, output_dim=3):
        super(Reconstruction, self).__init__()
        self.num_detectors = num_detectors
        self.input_features = input_features

        self.L1 = nn.Linear(num_detectors * input_features, hidden_lay1)
        self.L2 = nn.Linear(hidden_lay1, hidden_lay2)
        self.L3 = nn.Linear(hidden_lay2, hidden_lay3)
        self.L4 = nn.Linear(hidden_lay3, output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass with tanh activation on output layer.

        Parameters:
            x (torch.Tensor): shape (batch_size, num_detectors * input_features).

        Returns:
            torch.Tensor: output with normalized [X0, Y0, E_norm, Theta_norm, Phi_norm].
        """
        out = self.L1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)
        out = self.relu(out)
        out = self.L4(out)
        out = self.tanh(out)
        return out


def NormalizeLabels(E, theta, phi, theta_max=torch.pi * 2):
    """Normalize physical labels to the ranges expected by the network.

    Parameters:
        E, theta, phi (torch.Tensor): true energy, theta and phi.
        theta_max (float): maximum theta value for normalization.

    Returns:
        tuple: normalized (E_norm, theta_norm, phi_norm).
    """
    E_norm = 2 * (E - .1) / (10 - .1) - 1
    theta_norm = 2 * theta / theta_max - 1
    phi_norm = phi / torch.pi
    return E_norm, theta_norm, phi_norm


def DenormalizeLabels(E_norm, theta_norm, phi_norm, theta_max=torch.pi * 2):
    """Inverse of NormalizeLabels: map normalized outputs back to physical units.

    Parameters:
        E_norm, theta_norm, phi_norm (torch.Tensor): normalized network outputs.
        theta_max (float): maximum theta value for denormalization.

    Returns:
        tuple: denormalized (E, theta, phi).
    """
    E = 0.1 + (E_norm + 1) * (10 - 0.1) / 2
    theta = (theta_norm + 1) * theta_max / 2
    phi = phi_norm * torch.pi
    return E, theta, phi


class EarlyStopping:
    """Simple early stopping helper tracking validation loss improvements.

    Parameters:
        patience (int): number of epochs with no improvement before stopping.
        min_delta (float): minimum change to qualify as improvement.
    """
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
