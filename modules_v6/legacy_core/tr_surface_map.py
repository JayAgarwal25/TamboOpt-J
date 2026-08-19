"""Differentiable mountain surface function East = f(North, Up).

SurfaceEastMap builds a regular 256×256 grid of East values over the (North, Up)
bounding box of the detector centroids using scipy's LinearNDInterpolator on the
2161 centroid scatter.  At runtime, forward(x, y) performs an F.grid_sample
bilinear lookup which is differentiable w.r.t. (x, y) — this is the key
ingredient that lets gradients flow from the loss back through z_cont to the
learnable detector positions.

Usage:
    from modules_v4.tr_geometry   import load_tr_mountain
    from modules_v4.tr_surface_map import SurfaceEastMap

    mountain = load_tr_mountain(...)
    surface  = SurfaceEastMap.from_mountain(mountain, grid_h=256, grid_w=256).to(device)

    east_det = surface(x_det, y_det)   # (n_det,)  differentiable in x_det, y_det
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class SurfaceEastMap(nn.Module):
    """Differentiable lookup: East = f(North, Up).

    Stores a (1, 1, H, W) buffer `grid_east` whose rows index Up and columns
    index North over the bbox [n_min, n_max] × [u_min, u_max].
    forward(x, y) normalises (x, y) to [-1, 1] and calls F.grid_sample with
    bilinear interpolation and border padding.

    padding_mode='border' means a detector that wanders outside the mountain
    bbox receives the nearest valid East value instead of NaN — gradients still
    flow, no NaN explosions.
    """

    def __init__(
        self,
        grid_east: torch.Tensor,   # (H, W) float32
        n_min: float,
        n_max: float,
        u_min: float,
        u_max: float,
    ):
        super().__init__()
        # Store as (1, 1, H, W) for F.grid_sample
        self.register_buffer("grid_east", grid_east.float().unsqueeze(0).unsqueeze(0))
        self.n_min = float(n_min)
        self.n_max = float(n_max)
        self.u_min = float(u_min)
        self.u_max = float(u_max)

    @classmethod
    def from_mountain(cls, mountain, grid_h: int = 256, grid_w: int = 256, pad: float = 0.0):
        """Build the surface map from a MountainData object.

        Fits LinearNDInterpolator on the (North, Up) → East centroid scatter,
        evaluates it on a regular (grid_h × grid_w) grid, and fills any NaN
        cells (outside the convex hull) with nearest-neighbour values.

        Args:
            mountain : MountainData from load_tr_mountain().
            grid_h   : number of rows (Up axis).
            grid_w   : number of columns (North axis).
            pad      : extra margin (m) added to each bbox edge (default 0).
        """
        North = mountain.centroids_NUE[:, 0]
        Up    = mountain.centroids_NUE[:, 1]
        East  = mountain.centroids_NUE[:, 2]

        n_min = mountain.n_min - pad
        n_max = mountain.n_max + pad
        u_min = mountain.u_min - pad
        u_max = mountain.u_max + pad

        # Scattered linear interpolant  (North, Up) → East
        points = np.stack([North, Up], axis=1)
        interp_lin  = LinearNDInterpolator(points, East)
        interp_near = NearestNDInterpolator(points, East)

        # Regular grid — rows = Up, columns = North
        Ng = np.linspace(n_min, n_max, grid_w)
        Ug = np.linspace(u_min, u_max, grid_h)
        NN, UU = np.meshgrid(Ng, Ug)           # (H, W) each

        East_grid = interp_lin(NN, UU)         # (H, W), may have NaN outside hull
        nan_mask  = np.isnan(East_grid)
        if nan_mask.any():
            East_grid[nan_mask] = interp_near(NN[nan_mask], UU[nan_mask])

        return cls(
            torch.from_numpy(East_grid.astype(np.float32)),
            n_min, n_max, u_min, u_max,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable lookup: returns East(x, y) for each detector.

        Args:
            x : (n_det,) North coordinates in metres,  requires_grad may be True.
            y : (n_det,) Up   coordinates in metres,  requires_grad may be True.

        Returns:
            east : (n_det,) East coordinates in metres, differentiable in x and y.
        """
        # Normalise to [-1, 1]:  North → grid_sample's x-axis (columns)
        #                        Up    → grid_sample's y-axis (rows)
        nx = 2.0 * (x - self.n_min) / (self.n_max - self.n_min) - 1.0
        uy = 2.0 * (y - self.u_min) / (self.u_max - self.u_min) - 1.0

        # grid_sample expects (N, H_out, W_out, 2) with (x, y) = (col, row) convention
        # We use N=1, H_out=1, W_out=n_det
        grid = torch.stack([nx, uy], dim=-1).to(self.grid_east.dtype)
        grid = grid.view(1, 1, -1, 2)

        out = F.grid_sample(
            self.grid_east,          # (1, 1, H, W)
            grid,                    # (1, 1, n_det, 2)
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return out.view(-1)          # (n_det,)
