"""Real tau-shower primaries from `tau_wholesky.h5`.

Drop-in for `allshowers.generate_showers.sample_primary_particles`, returning the
same `energies / directions / labels` PLUS a `positions` array — the synthetic
sampler had no position and the old pipeline invented one by re-centering every
cloud onto the mountain; real taus carry a physical ENU decay vertex.

The file is site-local ENU (x=East, y=North, z=Up) anchored at the malata site,
the SAME frame `load_tr_mountain` builds the mountain in:
    position  (3, N) float64 — decay point [m]
    direction (3, N) float64 — unit vector
    energy    (N,)   float64 — primary energy [GeV]
    pdg       (N,)   int64   — 15 (tau) for every row

Two conversions worth knowing:
  - **labels** are sampled randomly (seeded), not read: tau_wholesky has no
    EM/hadronic class, but the AllShowers generator conditions on a 0/1 label.
  - **energy** is filtered to the generator's trained band
    [10**LOG_E_MIN, 10**LOG_E_MAX]; real taus reach well outside it, where the
    flow would be extrapolating.
"""

import h5py
import numpy as np
import torch


def load_tau_primaries(
    h5_path:  str,
    e_min:    float,
    e_max:    float,
    n:        int = 0,
    seed:     int = 0,
    num_classes: int = 2,
):
    """Load real tau primaries from tau_wholesky.h5, filtered to [e_min, e_max] GeV.

    Args:
        h5_path : path to tau_wholesky.h5.
        e_min/e_max : energy band [GeV] to keep (the generator's trained range).
        n       : cap on the number of events kept (0 = all in-band events).
        seed    : RNG seed for the EM/hadronic label draw (deterministic corpus).
        num_classes : number of EM/hadronic classes (0..num_classes-1).

    Returns:
        dict with (row-aligned):
            energies   (M, 1) float32 — primary energies [GeV]
            directions (M, 3) float32 — unit vectors (East, North, Up) = (x, y, z)
            labels     (M,)   int64   — EM/hadronic class (0/1), randomly drawn
            positions  (M, 3) float32 — ENU decay point [m], columns (East, North, Up)
            pdg        (M,)   int64   — source PDG (15), for provenance only
    """
    with h5py.File(h5_path, "r") as f:
        pos_enu = np.asarray(f["position"][...], dtype=np.float64)    # (3, N) east,north,up
        dir_enu = np.asarray(f["direction"][...], dtype=np.float64)   # (3, N) east,north,up
        energy  = np.asarray(f["energy"][...],   dtype=np.float64)    # (N,)
        pdg_src = np.asarray(f["pdg"][...],      dtype=np.int64)      # (N,)

    keep = (energy >= float(e_min)) & (energy <= float(e_max))
    idx  = np.nonzero(keep)[0]
    if n and n > 0:
        idx = idx[:int(n)]

    energies   = energy[idx].astype(np.float32).reshape(-1, 1)
    directions = dir_enu[:, idx].T.astype(np.float32)                 # (M, 3) East,North,Up = x,y,z
    positions  = pos_enu[:, idx].T.astype(np.float32)                 # (M, 3) East,North,Up
    pdg        = pdg_src[idx]

    rng    = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=idx.shape[0]).astype(np.int64)

    return {
        "energies":   torch.from_numpy(energies),
        "directions": torch.from_numpy(directions),
        "labels":     torch.from_numpy(labels),
        "positions":  torch.from_numpy(positions),
        "pdg":        torch.from_numpy(pdg),
    }
