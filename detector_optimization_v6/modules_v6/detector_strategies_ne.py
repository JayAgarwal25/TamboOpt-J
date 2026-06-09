"""(North, East) layout strategies — v6-local mirror of detector_strategies.py.

Diff against modules_v6/detector_strategies.py to see the (North, Up)→(North,
East) change: the bbox anchor uses East (centroid col 2) and the East bbox, and
sampling/projection go through `tr_geometry_ne` instead of the MountainData
methods. All generators return (North, East) tensors on the mountain surface.
"""

import torch
from .constants import N_DETECTORS
import numpy as np
from typing import Tuple

from modules.geometry import Layouts as _v3_Layouts
from .tr_geometry_ne import project_to_mountain_ne, sample_initial_layout_ne

# ── Layout generators (all return (North, East) tensors on the mountain) ─────

def _bbox_anchor(mountain) -> Tuple[float, float]:
    """Centroid nearest the mountain (North, East) bbox center."""
    cn = 0.5 * (mountain.n_min + mountain.n_max)
    ce = 0.5 * (mountain.east_lo + mountain.east_hi)
    d2 = (mountain.centroids_NUE[:, 0] - cn) ** 2 + (mountain.centroids_NUE[:, 2] - ce) ** 2
    k  = int(np.argmin(d2))
    return float(mountain.centroids_NUE[k, 0]), float(mountain.centroids_NUE[k, 2])


def layout_grid(mountain, n_det: int = N_DETECTORS,
                jitter_sigma: float = 20.0, rng=None):
    """NE grid + small Gaussian jitter + mountain projection."""
    if rng is None:
        rng = np.random.default_rng()
    N, E = sample_initial_layout_ne(mountain, n_units=n_det, scheme="grid")
    N = N + rng.normal(0.0, jitter_sigma, N.shape).astype(np.float32)
    E = E + rng.normal(0.0, jitter_sigma, E.shape).astype(np.float32)
    N_t = torch.as_tensor(N, dtype=torch.float32)
    E_t = torch.as_tensor(E, dtype=torch.float32)
    return project_to_mountain_ne(mountain, N_t, E_t)


def layout_center_gaussian(mountain, n_det: int = N_DETECTORS,
                           sigma: float = 200.0, rng=None):
    """Gaussian cluster at the mountain (North, East) bbox-center anchor."""
    if rng is None:
        rng = np.random.default_rng()
    anchor_n, anchor_e = _bbox_anchor(mountain)
    N = anchor_n + rng.normal(0.0, sigma, n_det).astype(np.float32)
    E = anchor_e + rng.normal(0.0, sigma, n_det).astype(np.float32)
    N_t = torch.as_tensor(N, dtype=torch.float32)
    E_t = torch.as_tensor(E, dtype=torch.float32)
    return project_to_mountain_ne(mountain, N_t, E_t)


def layout_rings(mountain, n_det: int = N_DETECTORS,
                 outer_radius: float = 500.0, n_rings: int = 6,
                 jitter_sigma: float = 20.0, rng=None):
    """Concentric rings at the mountain (North, East) center (v3 `Layouts`
    with a random rotation), translated to the anchor and projected."""
    if rng is None:
        rng = np.random.default_rng()
    anchor_n, anchor_e = _bbox_anchor(mountain)

    x, y = _v3_Layouts(
        n_detectors=n_det, n_rings=n_rings, radius=outer_radius,
        center=(0.0, 0.0), device="cpu",
    )
    x = x.to(torch.float32)
    y = y.to(torch.float32)

    rot = float(rng.uniform(0.0, 2.0 * np.pi))
    cos_r, sin_r = float(np.cos(rot)), float(np.sin(rot))
    x_rot = x * cos_r - y * sin_r
    y_rot = x * sin_r + y * cos_r

    N_t = x_rot + anchor_n + rng.normal(0.0, jitter_sigma, n_det).astype(np.float32)
    E_t = y_rot + anchor_e + rng.normal(0.0, jitter_sigma, n_det).astype(np.float32)
    return project_to_mountain_ne(mountain, N_t, E_t)


# ── Dataset builder ──────────────────────────────────────────────────────────

# Seven layout strategies; `args` threaded in below.
_STRATEGIES = [
    ("grid_jit20",        "layout_grid",            dict(jitter_sigma=20.0)),
    ("grid_jit200",       "layout_grid",            dict(jitter_sigma=200.0)),
    ("center_gauss200",   "layout_center_gaussian", dict(sigma=200.0)),
    ("center_gauss400",   "layout_center_gaussian", dict(sigma=400.0)),
    ("rings_R300",        "layout_rings",           dict(outer_radius=300.0,  n_rings=5, jitter_sigma=200.0)),
    ("rings_R800",        "layout_rings",           dict(outer_radius=800.0,  n_rings=6, jitter_sigma=200.0)),
    ("rings_R1800",       "layout_rings",           dict(outer_radius=1800.0, n_rings=8, jitter_sigma=200.0)),
]

_STRATEGY_FNS = {
    "layout_grid":            layout_grid,
    "layout_center_gaussian": layout_center_gaussian,
    "layout_rings":           layout_rings,
}
