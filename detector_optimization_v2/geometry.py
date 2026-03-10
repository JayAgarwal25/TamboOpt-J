"""Geometry functions: detector layout, effective distance/time, boundary projection.

Extracted from SWGOLO7_optimization.ipynb cells 7, 18, 19, 72, 73.
"""

import numpy as np
import torch


def Layouts(n_detectors=100, n_rings=6):
    """Create a detector layout with detectors distributed across concentric rings.

    Parameters:
        n_detectors (int): total number of detectors.
        n_rings (int): number of concentric rings.

    Returns:
        tuple: (x, y) numpy arrays of detector positions in meters.
    """
    R = np.linspace(5, 500, n_rings)

    weights = R / R.sum()
    N = np.round(weights * n_detectors).astype(int)

    diff = n_detectors - N.sum()
    N[-1] += diff

    radii = np.repeat(R, N)
    angles = np.concatenate([np.linspace(0, 2 * np.pi, n, endpoint=False) for n in N])

    return radii * np.cos(angles), radii * np.sin(angles)


def barycentric_coords(P, A, B, C):
    """Compute barycentric coordinates for each point P w.r.t. triangle ABC.

    Parameters:
        P: Tensor of shape (N, 2).
        A, B, C: Tensors of shape (2,).

    Returns:
        tuple: (u, v) barycentric coordinates.
    """
    v0 = C - A
    v1 = B - A
    v2 = P - A

    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01 + 1e-8
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    return u, v


def project_to_triangle(x, y,
                        A_coords=(-3800.0, 1500.0),
                        B_coords=(1200.0, 1500.0),
                        C_coords=(1200.0, -4100.0)):
    """Project each (x[i], y[i]) inside the given triangle.

    Parameters:
        x, y: tensors of shape (N,).
        A_coords, B_coords, C_coords: triangle vertex coordinates as tuples.

    Returns:
        tuple: projected (x, y) tensors of shape (N,).
    """
    A = torch.tensor(list(A_coords), device=x.device)
    B = torch.tensor(list(B_coords), device=x.device)
    C = torch.tensor(list(C_coords), device=x.device)

    P = torch.stack([x, y], dim=1)

    u, v = barycentric_coords(P, A, B, C)

    inside = (u >= 0) & (v >= 0) & (u + v <= 1)

    u_clipped = torch.clamp(u, 0.0, 1.0)
    v_clipped = torch.clamp(v, 0.0, 1.0)
    uv_sum = u_clipped + v_clipped
    over = uv_sum > 1.0
    u_clipped[over] = u_clipped[over] / uv_sum[over]
    v_clipped[over] = v_clipped[over] / uv_sum[over]

    v0 = C - A
    v1 = B - A
    P_proj = A + u_clipped.unsqueeze(1) * v0 + v_clipped.unsqueeze(1) * v1

    final_P = torch.where(inside.unsqueeze(1), P, P_proj)

    return final_P[:, 0], final_P[:, 1]
