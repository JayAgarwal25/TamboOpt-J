"""Population container for v5 evolutionary pruning.

A Population holds the **current set of detectors** — their (North, Up)
coordinates on the mountain surface, and the derived continuous plane index
`z_cont`.  Unlike v4's `LearnableXY`, positions are NOT `nn.Parameter`s:
there is no gradient-based optimizer updating them.  The evolutionary
algorithm mutates and prunes this container in place between generations.

Population size shrinks over the course of a run — typically from 10,000
down to 90 — so every operator must handle a variable detector count.

This module also exposes `build_input_batch`, the helper that converts
per-batch shower counts/times + population coordinates into the 7-feature
tensor that DeepSetsReconstruction consumes.  It mirrors v4's cell-40 stack
layout exactly: [x=N, y=Up, z=z_cont, N_int, T_int, x0, y0].
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Population:
    """Variable-size set of detectors on the mountain surface.

    Attributes:
        x        : (N_det,) float32 tensor — North [m].
        y        : (N_det,) float32 tensor — Up [m].
        z_cont   : (N_det,) float32 tensor — continuous AllShowers layer index.
                   Derived from `surface(x, y)` and cached; must be refreshed
                   via `refresh_z_cont()` after any in-place change to (x, y).
        mountain : MountainData from v4's tr_geometry.load_tr_mountain.
        surface  : SurfaceEastMap from v4's tr_surface_map.
    """

    x:        torch.Tensor
    y:        torch.Tensor
    z_cont:   torch.Tensor
    mountain: Any                 # modules_v4.tr_geometry.MountainData
    surface:  Any                 # modules_v4.tr_surface_map.SurfaceEastMap
    device:   torch.device = field(default=None)

    def __post_init__(self):
        if self.device is None:
            self.device = self.x.device

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def initial(
        cls,
        mountain,
        surface,
        n_units: int = 10_000,
        scheme:  str = "grid",
        device:  Any = "cpu",
    ) -> "Population":
        """Sample `n_units` detectors on the mountain surface and derive z_cont.

        Uses v4's `MountainData.sample_initial_layout`, which oversamples a
        grid and filters to points inside the mountain footprint.  For
        `n_units=10000` the default scheme='grid' gives an even covering of
        the full (North, Up) bbox filtered to the mountain surface.
        """
        N_np, U_np = mountain.sample_initial_layout(n_units=n_units, scheme=scheme)
        device = torch.device(device)
        x = torch.as_tensor(np.asarray(N_np), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(U_np), dtype=torch.float32, device=device)
        with torch.no_grad():
            east = surface(x, y)
            z = (mountain.east_entry - east) / mountain.layer_east_dx
        return cls(x=x, y=y, z_cont=z.detach(), mountain=mountain, surface=surface, device=device)

    # ── basic accessors ──────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return int(self.x.shape[0])

    def clone(self) -> "Population":
        """Return a detached deep copy of this population (same mountain/surface refs)."""
        return Population(
            x=self.x.detach().clone(),
            y=self.y.detach().clone(),
            z_cont=self.z_cont.detach().clone(),
            mountain=self.mountain,
            surface=self.surface,
            device=self.device,
        )

    # ── in-place mutators ────────────────────────────────────────────────────

    def refresh_z_cont(self) -> None:
        """Recompute z_cont from current (x, y).  Call after any position change."""
        with torch.no_grad():
            east = self.surface(self.x, self.y)
            self.z_cont = (self.mountain.east_entry - east) / self.mountain.layer_east_dx

    def apply_indices(self, keep_idx: torch.Tensor) -> None:
        """Keep only the detectors at the given indices (in-place)."""
        keep_idx = keep_idx.to(self.x.device).long()
        self.x      = self.x[keep_idx].contiguous()
        self.y      = self.y[keep_idx].contiguous()
        self.z_cont = self.z_cont[keep_idx].contiguous()

    # ── persistence ──────────────────────────────────────────────────────────

    def save_layout(self, path: str) -> None:
        """Write a 3-column (North, Up, z_cont) text file (v4-compatible format)."""
        arr = torch.stack([self.x, self.y, self.z_cont], dim=1).detach().cpu().numpy()
        np.savetxt(path, arr)

    @classmethod
    def load_layout(
        cls,
        path: str,
        mountain,
        surface,
        device: Any = "cpu",
    ) -> "Population":
        """Read a 3-column layout file and wrap it in a Population.

        The stored z_cont is trusted (no surface lookup) so this round-trips
        save→load without drift.
        """
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[None, :]
        device = torch.device(device)
        x = torch.as_tensor(data[:, 0], dtype=torch.float32, device=device)
        y = torch.as_tensor(data[:, 1], dtype=torch.float32, device=device)
        z = torch.as_tensor(data[:, 2], dtype=torch.float32, device=device)
        return cls(x=x, y=y, z_cont=z, mountain=mountain, surface=surface, device=device)


def build_input_batch(
    population: "Population",
    N_list:  torch.Tensor,      # (B, N_det)  plane-interpolated particle counts
    T_list:  torch.Tensor,      # (B, N_det)  plane-interpolated times
    X0:      torch.Tensor,      # (B,)        energy-weighted shower core North [m]
    Y0:      torch.Tensor,      # (B,)        energy-weighted shower core Up    [m]
    core_scale: float = 5000.0,
) -> torch.Tensor:
    """Build the (B, N_det, 7) input tensor consumed by DeepSetsReconstruction.

    Matches v4's feature order exactly so that a frozen input_mean / input_std
    computed on v4-style data is directly applicable:
        feature[0] : x = North   [m]
        feature[1] : y = Up      [m]
        feature[2] : z = z_cont
        feature[3] : N_int
        feature[4] : T_int       [ns]
        feature[5] : x0 / 5000
        feature[6] : y0 / 5000

    NOTE: the caller is responsible for normalization (same frozen mean/std
    pattern as v4 — apply it AFTER this builder returns).
    """
    B, N = N_list.shape
    if N != population.size:
        raise ValueError(
            f"N_list has {N} detectors but population.size={population.size}"
        )
    x_exp  = population.x.unsqueeze(0).expand(B, -1)
    y_exp  = population.y.unsqueeze(0).expand(B, -1)
    z_exp  = population.z_cont.unsqueeze(0).expand(B, -1)
    x0_exp = (X0 / core_scale).unsqueeze(1).expand(-1, N)
    y0_exp = (Y0 / core_scale).unsqueeze(1).expand(-1, N)

    return torch.stack(
        [x_exp, y_exp, z_exp, N_list, T_list, x0_exp, y0_exp],
        dim=2,
    ).float()
