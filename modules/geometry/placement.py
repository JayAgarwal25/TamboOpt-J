"""(North, East) projection + initial-layout helpers.

Free functions rather than `MountainData` methods — pass the loaded `mountain`
in. Centroid columns are East = 0 and **North = 1** of `mountain.centroids_ENU`;
the detector box is the East bbox `[east_lo, east_hi]`.
"""

import math
from typing import Tuple

import numpy as np
import torch

from ..constants import N_DETECTORS

# `_ne_max_gap` is deterministic for a given mountain (seeded `default_rng(0)`)
# yet `project_to_mountain_ne(max_gap=None)` recomputes it on every layout draw —
# ~100k 500×500 pairwise reductions per dataset build. The value is cached on the
# mountain object itself: per-object storage cannot leak across mountains and is
# released with the mountain.
_MAX_GAP_ATTR = "_ne_max_gap_cached"


def _ne_max_gap(mountain) -> float:
    """2× mean nearest-neighbour spacing of the centroids in the (North, East)
    plane — the "inside the mountain" tolerance (mirrors the inline estimate in
    the v4 methods). Memoized on `mountain`."""
    cached = getattr(mountain, _MAX_GAP_ATTR, None)
    if cached is not None:
        return cached
    N_c, E_c = mountain.centroids_ENU[:, 1], mountain.centroids_ENU[:, 0]
    n_sample = min(500, len(N_c))
    idx = np.random.default_rng(0).choice(len(N_c), n_sample, replace=False)
    samp = np.stack([N_c[idx], E_c[idx]], axis=1)
    d2 = ((samp[:, None, :] - samp[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    gap = 2.0 * float(np.sqrt(d2.min(axis=1)).mean())
    setattr(mountain, _MAX_GAP_ATTR, gap)
    return gap


def project_to_mountain_ne(mountain, E: torch.Tensor, N: torch.Tensor,
                           max_gap: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project (East, North) points back to the mountain surface.

    Detector coordinates are (East, North) to match the ENU convention of the h5
    data files. For each point whose distance to the nearest mountain centroid (in
    the East–North plane) exceeds `max_gap`, snap it onto that centroid's
    (East, North).
    """
    device, dtype = E.device, E.dtype
    E_c = torch.as_tensor(mountain.centroids_ENU[:, 0], dtype=dtype, device=device)
    N_c = torch.as_tensor(mountain.centroids_ENU[:, 1], dtype=dtype, device=device)

    if max_gap is None:
        max_gap = _ne_max_gap(mountain)

    d2 = (E[:, None] - E_c[None, :]) ** 2 + (N[:, None] - N_c[None, :]) ** 2
    nearest_d2, nearest_idx = d2.min(dim=1)
    outside = nearest_d2 > (max_gap ** 2)

    E_new, N_new = E.clone(), N.clone()
    E_new[outside] = E_c[nearest_idx[outside]]
    N_new[outside] = N_c[nearest_idx[outside]]
    return E_new, N_new


# ── Initial-layout schemes (all return (E_init, N_init) float32 arrays) ──────
# `on_mountain(pn, pe)` is the shared "within max_gap of a centroid" predicate.

def _layout_grid_candidates(mountain, n_units: int, on_mountain
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Oversampled (North, East) lattice, filtered to the surface, then thinned
    to exactly `n_units` evenly spaced survivors."""
    e_min, e_max = mountain.east_lo, mountain.east_hi
    over = 4
    cols = max(1, int(math.ceil(math.sqrt(over * n_units * (mountain.n_max - mountain.n_min)
                                              / max(e_max - e_min, 1.0)))))
    rows = max(1, int(math.ceil(over * n_units / cols)))
    n_vals = np.linspace(mountain.n_min, mountain.n_max, cols + 2)[1:-1]
    e_vals = np.linspace(e_min, e_max, rows + 2)[1:-1]
    NN, EE = np.meshgrid(n_vals, e_vals)
    cand_n, cand_e = NN.ravel(), EE.ravel()
    keep = np.array([on_mountain(n, e) for n, e in zip(cand_n, cand_e)])
    valid_n, valid_e = cand_n[keep], cand_e[keep]
    if len(valid_n) < n_units:
        raise RuntimeError(f"Only {len(valid_n)} NE grid points on the mountain "
                           f"(need {n_units}); relax max_gap or oversampling.")
    sel = np.linspace(0, len(valid_n) - 1, n_units).round().astype(int)
    return valid_e[sel].astype(np.float32), valid_n[sel].astype(np.float32)


def _layout_random(mountain, n_units: int, on_mountain
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Rejection sampling of uniform (North, East) bbox draws onto the surface."""
    e_min, e_max = mountain.east_lo, mountain.east_hi
    rng = np.random.default_rng()
    out_n, out_e, tries = [], [], 0
    while len(out_n) < n_units and tries < 100 * n_units:
        pn = rng.uniform(mountain.n_min, mountain.n_max)
        pe = rng.uniform(e_min, e_max)
        if on_mountain(pn, pe):
            out_n.append(pn); out_e.append(pe)
        tries += 1
    if len(out_n) < n_units:
        raise RuntimeError(f"Random NE sampling placed only {len(out_n)}/{n_units}")
    return np.array(out_e, dtype=np.float32), np.array(out_n, dtype=np.float32)


def _layout_center(mountain, n_units: int, on_mountain
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Tight Gaussian cluster around the centroid nearest the bbox centre."""
    N_c, E_c = mountain.centroids_ENU[:, 1], mountain.centroids_ENU[:, 0]
    e_min, e_max = mountain.east_lo, mountain.east_hi
    cn = 0.5 * (mountain.n_min + mountain.n_max)
    ce = 0.5 * (e_min + e_max)
    anchor = int(np.argmin((N_c - cn) ** 2 + (E_c - ce) ** 2))
    anchor_n, anchor_e = float(N_c[anchor]), float(E_c[anchor])

    rng = np.random.default_rng(0)
    sigma = 50.0 / 3.0   # ~50 m total spread (≈3σ)
    out_n, out_e, tries = [], [], 0
    while len(out_n) < n_units and tries < 1000 * n_units:
        pn = anchor_n + float(rng.normal(0.0, sigma))
        pe = anchor_e + float(rng.normal(0.0, sigma))
        if on_mountain(pn, pe):
            out_n.append(pn); out_e.append(pe)
        tries += 1
    while len(out_n) < n_units:   # fall back: stack on the anchor
        out_n.append(anchor_n); out_e.append(anchor_e)
    return np.array(out_e, dtype=np.float32), np.array(out_n, dtype=np.float32)


_LAYOUT_SCHEME_FNS = {
    "grid":   _layout_grid_candidates,
    "random": _layout_random,
    "center": _layout_center,
}


def sample_initial_layout_ne(mountain, n_units: int = N_DETECTORS,
                             scheme: str = "grid") -> Tuple[np.ndarray, np.ndarray]:
    """Return (E_init, N_init) on the mountain surface.

    Detector coordinates are (East, North) to match the ENU convention of the h5
    data files: candidates are filtered to those within `max_gap` of a centroid in
    the East–North plane. schemes: 'grid', 'random', 'center'.
    """
    N_c, E_c = mountain.centroids_ENU[:, 1], mountain.centroids_ENU[:, 0]
    max_gap = _ne_max_gap(mountain)

    def _on(pn, pe):
        return ((N_c - pn) ** 2 + (E_c - pe) ** 2).min() <= max_gap ** 2

    if scheme not in _LAYOUT_SCHEME_FNS:
        raise ValueError(f"Unknown scheme '{scheme}'. Use 'grid', 'random', or 'center'.")
    return _LAYOUT_SCHEME_FNS[scheme](mountain, n_units, _on)
