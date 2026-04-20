import torch
from .constants import N_DETECTORS
import numpy as np
from typing import Tuple

from modules.geometry           import Layouts as _v3_Layouts

# ── Layout generators (all return (x, y) tensors on the mountain surface) ────

def _bbox_anchor(mountain) -> Tuple[float, float]:
    """Centroid nearest the mountain (N, Up) bbox center."""
    cn = 0.5 * (mountain.n_min + mountain.n_max)
    cu = 0.5 * (mountain.u_min + mountain.u_max)
    d2 = (mountain.centroids_NUE[:, 0] - cn) ** 2 + (mountain.centroids_NUE[:, 1] - cu) ** 2
    k  = int(np.argmin(d2))
    return float(mountain.centroids_NUE[k, 0]), float(mountain.centroids_NUE[k, 1])


def layout_grid(mountain, n_det: int = N_DETECTORS,
                jitter_sigma: float = 20.0, rng=None):
    """v4 grid + small Gaussian jitter + mountain projection."""
    if rng is None:
        rng = np.random.default_rng()
    N, U = mountain.sample_initial_layout(n_units=n_det, scheme="grid")
    N = N + rng.normal(0.0, jitter_sigma, N.shape).astype(np.float32)
    U = U + rng.normal(0.0, jitter_sigma, U.shape).astype(np.float32)
    N_t = torch.as_tensor(N, dtype=torch.float32)
    U_t = torch.as_tensor(U, dtype=torch.float32)
    return mountain.project_to_mountain(N_t, U_t)


def layout_center_gaussian(mountain, n_det: int = N_DETECTORS,
                           sigma: float = 200.0, rng=None):
    """Gaussian cluster at the mountain bbox-center anchor, fixed sigma."""
    if rng is None:
        rng = np.random.default_rng()
    anchor_n, anchor_u = _bbox_anchor(mountain)
    N = anchor_n + rng.normal(0.0, sigma, n_det).astype(np.float32)
    U = anchor_u + rng.normal(0.0, sigma, n_det).astype(np.float32)
    N_t = torch.as_tensor(N, dtype=torch.float32)
    U_t = torch.as_tensor(U, dtype=torch.float32)
    return mountain.project_to_mountain(N_t, U_t)


def layout_rings(mountain, n_det: int = N_DETECTORS,
                 outer_radius: float = 500.0, n_rings: int = 6,
                 jitter_sigma: float = 20.0, rng=None):
    """Concentric rings at mountain center (v3 `Layouts` with a random rotation).

    The layout is built with v3's `Layouts(n_detectors, n_rings, radius, center)`
    helper in a local (dx, dy) frame, then rotated by a random angle, translated
    to the mountain bbox-center anchor, and projected to the mountain.
    """
    if rng is None:
        rng = np.random.default_rng()
    anchor_n, anchor_u = _bbox_anchor(mountain)

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
    U_t = y_rot + anchor_u + rng.normal(0.0, jitter_sigma, n_det).astype(np.float32)
    return mountain.project_to_mountain(N_t, U_t)


# ── Dataset builder ──────────────────────────────────────────────────────────

# Five layout strategies; `args` threaded in below.
_STRATEGIES = [
    ("grid_jit20",        "layout_grid",            dict(jitter_sigma=200.0)),
    ("center_gauss200",   "layout_center_gaussian", dict(sigma=200.0)),
    ("rings_R300",        "layout_rings",           dict(outer_radius=300.0,  n_rings=5, jitter_sigma=200.0)),
    ("rings_R800",        "layout_rings",           dict(outer_radius=800.0,  n_rings=6, jitter_sigma=200.0)),
    ("rings_R1800",       "layout_rings",           dict(outer_radius=1800.0, n_rings=6, jitter_sigma=200.0)),
]

_STRATEGY_FNS = {
    "layout_grid":            layout_grid,
    "layout_center_gaussian": layout_center_gaussian,
    "layout_rings":           layout_rings,
}
