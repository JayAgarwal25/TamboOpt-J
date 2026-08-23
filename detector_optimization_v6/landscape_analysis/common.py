"""Shared setup for the landscape analyses.

Every script here needs the same four things: the frozen surrogate pair, the
mountain, a set of fresh primary batches, and a way to score a layout. Each one
used to carry its own copy of that, around forty lines apiece. This is that copy,
written once.

Importing this module selects the non-interactive matplotlib backend, because
every script here writes PNGs from a batch job and none of them opens a window.
"""

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed 42 is the exact batch every 04_optimize_*.py scores against, so saved
# layouts are overfit to it and it reads ~10-15 points high against genuinely
# fresh batches. Scans here start well above it so they measure the stable,
# non-overfit estimate instead.
BATCH_SEED_BASE = 1000
BATCH_SIZE = 512


def load_mountain():
    """The malata mesh in site-local ENU."""
    return load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                            east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                            n_planes=N_PLANES)


def load_layout(path):
    """(x, y, U) from a saved layout_best.pt. x is East, y is North."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


def bbox_center(mountain):
    """Centre of the mountain bounding box, as (East, North).

    Returned in that order deliberately, to pair with the layout arrays where x
    is East and y is North. The grid-scan scripts used to compare x against the
    north centre and y against the east centre, which picked the wrong detectors
    as "closest to centre" and "closest to the edge".
    """
    return (0.5 * (mountain.east_lo + mountain.east_hi),
            0.5 * (mountain.n_min + mountain.n_max))


def dist_to_center(x, y, mountain):
    """Distance of each detector from the bbox centre, in metres."""
    ce, cn = bbox_center(mountain)
    return ((x - ce) ** 2 + (y - cn) ** 2) ** 0.5


class Scorer:
    """Frozen surrogate plus a fixed set of primary batches, scoring layouts.

    The batches are drawn once and reused for every call, so two layouts scored
    through the same Scorer are compared on identical events and the difference
    between them is not batch noise.
    """

    def __init__(self, n_batches=8, batch_size=BATCH_SIZE,
                 seed_base=BATCH_SEED_BASE, device=DEVICE, verbose=True):
        self.device = device
        self.fnn, self.recon = load_models(device, fnn_folder=FNN_FOLDER,
                                           recon_dir=RECON_FOLDER + "_deepsets")
        self.mountain = load_mountain()
        self.primary_all = torch.load(
            os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
            weights_only=False).float()
        self.n_total = self.primary_all.shape[0]
        self.batches = [self.draw(seed_base + b, batch_size)
                        for b in range(n_batches)]
        if verbose:
            print(f"[scorer] {n_batches} batches of {batch_size} "
                  f"from {self.n_total} primaries on {device}")

    def draw(self, seed, size=BATCH_SIZE):
        """One reproducible primary batch."""
        g = torch.Generator().manual_seed(int(seed))
        idx = torch.randint(0, self.n_total, (size,), generator=g)
        return self.primary_all[idx].to(self.device)

    @torch.no_grad()
    def U_on(self, x, y, primary):
        """U and mean reconstructability of one layout on one batch."""
        U, r, _ = utility_of_xy(x.to(self.device), y.to(self.device),
                                primary, self.fnn, self.recon)
        return float(U.item()), float(r.mean().item())

    def U_per_batch(self, x, y):
        """U of one layout on each held batch."""
        return [self.U_on(x, y, p)[0] for p in self.batches]

    def U(self, x, y):
        """Mean U of one layout over all held batches."""
        return float(np.mean(self.U_per_batch(x, y)))
