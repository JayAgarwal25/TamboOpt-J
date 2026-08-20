"""Differentiable mountain surface function Up = g(North, East).

`SurfaceUpMap` builds a regular 256x256 grid of Up over the (North, East) bbox of
the detector centroids (scipy `LinearNDInterpolator` on the 2161-centroid
scatter). `forward(north, east)` is an `F.grid_sample` bilinear lookup, so it is
differentiable in (north, east).

Usage:
    from modules_v6.legacy_core.tr_geometry      import load_tr_mountain
    from modules_v6.surface_map import SurfaceUpMap

    mountain = load_tr_mountain(...)
    surface  = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(device)

    up_det = surface(north_det, east_det)   # (n_det,)  differentiable in north, east
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class SurfaceUpMap(nn.Module):
    """Differentiable lookup: Up = g(North, East).

    Stores a (1, 1, H, W) buffer `grid_up` whose rows index East and columns
    index North over the bbox [n_min, n_max] × [e_min, e_max].
    forward(north, east) normalises (north, east) to [-1, 1] and calls
    F.grid_sample with bilinear interpolation and border padding.

    padding_mode='border' means a detector that wanders outside the mountain
    bbox receives the nearest valid Up value instead of NaN — gradients still
    flow, no NaN explosions.
    """

    def __init__(
        self,
        grid_up: torch.Tensor,     # (H, W) float32
        n_min: float,
        n_max: float,
        e_min: float,
        e_max: float,
    ):
        super().__init__()
        # Store as (1, 1, H, W) for F.grid_sample
        self.register_buffer("grid_up", grid_up.float().unsqueeze(0).unsqueeze(0))
        self.n_min = float(n_min)
        self.n_max = float(n_max)
        self.e_min = float(e_min)
        self.e_max = float(e_max)

    @classmethod
    def from_mountain(cls, mountain, grid_h: int = 256, grid_w: int = 256, pad: float = 0.0):
        """Build the surface map from a MountainData object.

        Fits LinearNDInterpolator on the (North, East) → Up scatter of the real
        mountain surface — the detector-region triangle *vertices* together with
        the face centroids — evaluates it on a regular (grid_h × grid_w) grid,
        and fills any NaN cells (outside the convex hull) with nearest-neighbour
        values. The grid domain spans all those points (the vertices reach past
        the centroid min/max), so the surface follows the full terrain footprint
        rather than only the rectangle between the detector centroids.

        Args:
            mountain : MountainData from load_tr_mountain().
            grid_h   : number of rows (East axis).
            grid_w   : number of columns (North axis).
            pad      : extra margin (m) added to each bbox edge (default 0).
        """
        East  = mountain.centroids_ENU[:, 0]
        North = mountain.centroids_ENU[:, 1]
        Up    = mountain.centroids_ENU[:, 2]

        # Include the actual mountain triangle vertices (real surface corner
        # points) so the fit reflects the whole terrain, not just the centroid
        # convex hull. Vertices extend slightly past the centroid bbox.
        verts = getattr(mountain, "vertices_ENU", None)
        if verts is not None and len(verts):
            East  = np.concatenate([East,  verts[:, 0]])
            North = np.concatenate([North, verts[:, 1]])
            Up    = np.concatenate([Up,    verts[:, 2]])

        # Grid domain spans all fit points (not just the centroid min/max).
        n_min = float(North.min()) - pad
        n_max = float(North.max()) + pad
        e_min = float(East.min())  - pad
        e_max = float(East.max())  + pad

        # Scattered linear interpolant  (North, East) → Up
        points = np.stack([North, East], axis=1)
        interp_lin  = LinearNDInterpolator(points, Up)
        interp_near = NearestNDInterpolator(points, Up)

        # Regular grid — rows = East, columns = North
        Ng = np.linspace(n_min, n_max, grid_w)
        Eg = np.linspace(e_min, e_max, grid_h)
        NN, EE = np.meshgrid(Ng, Eg)           # (H, W) each

        Up_grid = interp_lin(NN, EE)           # (H, W), may have NaN outside hull
        nan_mask  = np.isnan(Up_grid)
        if nan_mask.any():
            Up_grid[nan_mask] = interp_near(NN[nan_mask], EE[nan_mask])

        return cls(
            torch.from_numpy(Up_grid.astype(np.float32)),
            n_min, n_max, e_min, e_max,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable lookup: returns Up(north, east) for each detector.

        Args:
            x : (n_det,) North coordinates in metres,  requires_grad may be True.
            y : (n_det,) East  coordinates in metres,  requires_grad may be True.

        Returns:
            up : (n_det,) Up coordinates in metres, differentiable in x and y.
        """
        # Normalise to [-1, 1]:  North → grid_sample's x-axis (columns)
        #                        East  → grid_sample's y-axis (rows)
        nx = 2.0 * (x - self.n_min) / (self.n_max - self.n_min) - 1.0
        ey = 2.0 * (y - self.e_min) / (self.e_max - self.e_min) - 1.0

        # grid_sample expects (N, H_out, W_out, 2) with (x, y) = (col, row) convention
        # We use N=1, H_out=1, W_out=n_det
        grid = torch.stack([nx, ey], dim=-1).to(self.grid_up.dtype)
        grid = grid.view(1, 1, -1, 2)

        out = F.grid_sample(
            self.grid_up,            # (1, 1, H, W)
            grid,                    # (1, 1, n_det, 2)
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return out.view(-1)          # (n_det,)
