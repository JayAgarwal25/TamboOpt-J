"""TR geometry loader for detector_optimization_v4.

Reads TAMBOSim/resources/basic_geometry.h5, projects the 2161 detector-region
triangle centroids from ECEF to local ENU, and returns a MountainData dataclass
used by SurfaceEastMap and the main notebook.

Key facts about the HDF5 file (group colca_valley_30000):
  vertices  (3, 90000)  float64 — ECEF metres
  faces     (3, 179996) int64   — JULIA 1-INDEXED vertex indices
  detector1 (2161,)     int64   — JULIA 1-INDEXED face indices into faces
  location  (2,)        [lon_deg, lat_deg] of the site

AllShowers layer-East mapping (empirically derived from fixture data):
  East at AllShowers layer k:  East_k = EAST_ENTRY + k * (-LAYER_EAST_DX)
                              = -212 - 307 * k   [metres]
  Inverse (z_cont from East):  z_cont = (EAST_ENTRY - East) / LAYER_EAST_DX
                                       = (-212 - East) / 307
  Layer 0 (padding, energy=0): East ≈ -212 m
  Layer 1:                      East ≈  -519 m
  Layer 6:                      East ≈ -2054 m
  Layer 23:                     East ≈ -7267 m

Mountain surface East spans ≈ [-2019, +1182] m.  Only centroids with
East < EAST_ENTRY (= -212 m) have z_cont > 0 and can see shower particles.
The deepest accessible mountain layer is z_cont ≈ 5.9 (East ≈ -2019 m).

Gotchas:
  - faces and detector1 are 1-indexed (Julia) — subtract 1 before using as Python indices.
  - vertices are ECEF, not ENU; rotate to local ENU at the site.
  - z_cont = (EAST_ENTRY - East) / LAYER_EAST_DX   (NOT East/125 as originally planned).
"""

import math
from dataclasses import dataclass

import h5py
import numpy as np
import torch

# ── Default paths / constants ────────────────────────────────────────────────
DEFAULT_GEOMETRY_PATH = "../../TAMBOSim/resources/basic_geometry.h5"
DEFAULT_GROUP         = "colca_valley_30000"
DEFAULT_DET_KEY       = "detector1"
DEFAULT_N_PLANES      = 24
SITE_LON_DEG          = -72.279397
SITE_LAT_DEG          = -15.622267

# AllShowers layer-East calibration (derived from shower fixture point-cloud data)
# East at AllShowers layer k:  East_k ≈ EAST_ENTRY - k * LAYER_EAST_DX
ALLSHOWERS_EAST_ENTRY = -212.0    # m — East at layer 0 (shower entry, padding)
ALLSHOWERS_LAYER_DX   =  307.0    # m — East depth per layer (positive; East decreases per layer)


# ── ECEF → local ENU rotation ────────────────────────────────────────────────

def _ecef_to_enu(centroids_ecef: np.ndarray, lon_deg: float, lat_deg: float) -> np.ndarray:
    """Rotate ECEF (3, D) to local ENU (3, D) around the site.

    Returns rows ordered as [East, North, Up].
    Uses a sphere of mean Earth radius (6 371 000 m) as the origin.
    """
    lon0 = math.radians(lon_deg)
    lat0 = math.radians(lat_deg)
    R_e  = 6_371_000.0

    # Origin in ECEF
    s = R_e * np.array([
        math.cos(lat0) * math.cos(lon0),
        math.cos(lat0) * math.sin(lon0),
        math.sin(lat0),
    ])

    # ENU rotation matrix
    R = np.array([
        [-math.sin(lon0),                 math.cos(lon0),                0.0],
        [-math.sin(lat0) * math.cos(lon0), -math.sin(lat0) * math.sin(lon0), math.cos(lat0)],
        [ math.cos(lat0) * math.cos(lon0),  math.cos(lat0) * math.sin(lon0), math.sin(lat0)],
    ])

    return R @ (centroids_ecef - s[:, None])   # (3, D) rows = [East, North, Up]


# ── MountainData dataclass ────────────────────────────────────────────────────

@dataclass
class MountainData:
    """All geometry info needed by v4.

    centroids_NUE : (n_tri, 3) float64 numpy array, columns = [North, Up, East] in metres.
    n_min / n_max : North bounding box of detector centroids.
    u_min / u_max : Up (elevation) bounding box.
    east_lo / east_hi : actual East span of the centroids (≈ [-2019, +1182]).

    z_cont formula:
        z_cont = (east_entry - East_det) / layer_east_dx
    where:
        east_entry    : East value at AllShowers layer 0 (default -212 m).
        layer_east_dx : East depth per layer (default 307 m, positive;
                        East decreases by this amount per layer going deeper).

    Only centroids with East < east_entry have z_cont > 0 (see shower particles).
    The maximum z_cont reachable on the mountain surface is
        z_cont_max = (east_entry - east_lo) / layer_east_dx  ≈ 5.9
    corresponding to AllShowers layers 0–6.
    """
    centroids_NUE: np.ndarray    # (n_tri, 3) columns [North, Up, East]

    n_min:   float
    n_max:   float
    u_min:   float
    u_max:   float
    east_lo: float               # actual centroid East min (most negative)
    east_hi: float               # actual centroid East max (most positive)

    east_entry:    float         # East at AllShowers layer 0 (default -212 m)
    layer_east_dx: float         # East depth per layer (default 307 m, positive)
    n_planes:      int           # number of AllShowers planes (24)

    @property
    def plane_dx(self) -> float:
        """Alias kept for legacy callers: returns -layer_east_dx (signed East per layer)."""
        return -self.layer_east_dx

    def east_to_z_cont(self, east: float) -> float:
        """Convert an East value to a continuous AllShowers layer index."""
        return (self.east_entry - east) / self.layer_east_dx

    def project_to_mountain(self, N: "torch.Tensor", Up: "torch.Tensor",
                            max_gap: float = None) -> tuple:
        """Project (N, Up) points back to the mountain surface.

        For each point whose distance to the nearest mountain centroid is larger
        than `max_gap`, snap it onto that nearest centroid's (N, Up).  Points
        already on the mountain are returned unchanged.

        Args:
            N, Up   : (n,) tensors of detector (North, Up) coordinates.
            max_gap : distance threshold in metres.  If None, derived from local
                      centroid spacing as 2× the mean nearest-neighbour distance
                      of a 500-sample subset.

        Returns:
            (N_new, Up_new) tensors on the same device / dtype as inputs.
        """
        import torch as _torch

        device = N.device
        dtype  = N.dtype
        N_c = _torch.as_tensor(self.centroids_NUE[:, 0], dtype=dtype, device=device)
        U_c = _torch.as_tensor(self.centroids_NUE[:, 1], dtype=dtype, device=device)

        if max_gap is None:
            n_sample = min(500, len(N_c))
            idx = np.random.default_rng(0).choice(len(N_c), n_sample, replace=False)
            samp = np.stack([self.centroids_NUE[idx, 0],
                             self.centroids_NUE[idx, 1]], axis=1)
            d2 = ((samp[:, None, :] - samp[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            mean_nn = float(np.sqrt(d2.min(axis=1)).mean())
            max_gap = 2.0 * mean_nn

        # (n_det, n_cent) pairwise squared distances — small enough to be fast
        d2 = (N[:, None] - N_c[None, :]) ** 2 + (Up[:, None] - U_c[None, :]) ** 2
        nearest_d2, nearest_idx = d2.min(dim=1)
        outside = nearest_d2 > (max_gap ** 2)

        N_new  = N.clone()
        Up_new = Up.clone()
        N_new[outside]  = N_c[nearest_idx[outside]]
        Up_new[outside] = U_c[nearest_idx[outside]]
        return N_new, Up_new

    def sample_initial_layout(self, n_units: int = 90, scheme: str = "grid") -> tuple:
        """Return (N_init, U_init) numpy arrays of shape (n_units,) on the mountain surface.

        Candidate points are filtered: only those whose distance to the nearest mountain
        centroid is below `max_gap` (auto-derived from local centroid spacing) are kept.
        This rejects bbox "holes" without collapsing many candidates onto the same point.

        scheme='grid'   : dense grid in the bbox, filtered, then thinned to n_units.
        scheme='random' : uniform random in the bbox, accept-reject until n_units valid.
        """
        N_c = self.centroids_NUE[:, 0]
        U_c = self.centroids_NUE[:, 1]

        # Estimate a reasonable "inside the mountain" tolerance from centroid density.
        # Mean nearest-neighbour distance gives roughly the local triangle spacing.
        n_sample = min(500, len(N_c))
        idx = np.random.default_rng(0).choice(len(N_c), n_sample, replace=False)
        sampled = np.stack([N_c[idx], U_c[idx]], axis=1)
        d2 = ((sampled[:, None, :] - sampled[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        mean_nn = float(np.sqrt(d2.min(axis=1)).mean())
        max_gap = 2.0 * mean_nn   # accept points within 2× local spacing of any centroid

        def _is_on_mountain(pn, pu):
            d2 = (N_c - pn) ** 2 + (U_c - pu) ** 2
            return d2.min() <= max_gap ** 2

        if scheme == "grid":
            # Build a dense grid (4× oversampled), filter to mountain-only points,
            # then take an evenly spaced subset of n_units.
            over = 4
            cols = max(1, int(math.ceil(math.sqrt(over * n_units * (self.n_max - self.n_min)
                                                  / max(self.u_max - self.u_min, 1.0)))))
            rows = max(1, int(math.ceil(over * n_units / cols)))
            n_vals = np.linspace(self.n_min, self.n_max, cols + 2)[1:-1]
            u_vals = np.linspace(self.u_min, self.u_max, rows + 2)[1:-1]
            NN, UU = np.meshgrid(n_vals, u_vals)
            cand_n = NN.ravel()
            cand_u = UU.ravel()
            keep = np.array([_is_on_mountain(n, u) for n, u in zip(cand_n, cand_u)])
            valid_n = cand_n[keep]
            valid_u = cand_u[keep]
            if len(valid_n) < n_units:
                raise RuntimeError(f"Only {len(valid_n)} grid points fall on the mountain "
                                   f"(need {n_units}); increase oversampling or relax max_gap.")
            sel = np.linspace(0, len(valid_n) - 1, n_units).round().astype(int)
            return valid_n[sel].astype(np.float32), valid_u[sel].astype(np.float32)

        elif scheme == "random":
            rng = np.random.default_rng()
            out_n, out_u = [], []
            tries = 0
            while len(out_n) < n_units and tries < 100 * n_units:
                pn = rng.uniform(self.n_min, self.n_max)
                pu = rng.uniform(self.u_min, self.u_max)
                if _is_on_mountain(pn, pu):
                    out_n.append(pn); out_u.append(pu)
                tries += 1
            if len(out_n) < n_units:
                raise RuntimeError(f"Random sampling could only place {len(out_n)} of {n_units}")
            return np.array(out_n, dtype=np.float32), np.array(out_u, dtype=np.float32)

        else:
            raise ValueError(f"Unknown scheme '{scheme}'. Use 'grid' or 'random'.")


# ── Top-level loader ──────────────────────────────────────────────────────────

def load_tr_mountain(
    h5_path:        str   = DEFAULT_GEOMETRY_PATH,
    group:          str   = DEFAULT_GROUP,
    det_key:        str   = DEFAULT_DET_KEY,
    east_entry:     float = ALLSHOWERS_EAST_ENTRY,
    layer_east_dx:  float = ALLSHOWERS_LAYER_DX,
    n_planes:       int   = DEFAULT_N_PLANES,
    # Legacy aliases (ignored if the above are set)
    east_min:       float = None,
    east_max:       float = None,
) -> MountainData:
    """Read basic_geometry.h5, compute detector-region centroids in ENU, return MountainData.

    Args:
        h5_path       : path to basic_geometry.h5.
        group         : HDF5 group name (default 'colca_valley_30000').
        det_key       : dataset key for the detector-region triangle indices (default 'detector1').
                        This is a (2161,) array of 1-indexed face indices — subtract 1 in Python.
        east_entry    : East at AllShowers layer 0 (default -212 m, empirically calibrated).
        layer_east_dx : East depth per layer in metres (default 307 m, positive).
        n_planes      : number of AllShowers planes (default 24).
        east_min / east_max : legacy parameters, ignored.  Remove from call sites.
    """
    with h5py.File(h5_path, "r") as f:
        g        = f[group]
        verts    = g["vertices"][...]          # (3, 90000) ECEF float64
        faces    = g["faces"][...] - 1         # (3, 179996) 0-indexed
        det_idx  = g[det_key][...] - 1         # (2161,)     0-indexed

    # Triangle centroids in ECEF
    tri_verts      = verts[:, faces[:, det_idx]]    # (3, 3, 2161)
    centroids_ecef = tri_verts.mean(axis=1)          # (3, 2161)

    # Rotate to local ENU
    enu = _ecef_to_enu(centroids_ecef, SITE_LON_DEG, SITE_LAT_DEG)  # [East, North, Up]
    East, North, Up = enu[0], enu[1], enu[2]

    centroids_NUE = np.stack([North, Up, East], axis=1)   # (2161, 3)

    return MountainData(
        centroids_NUE = centroids_NUE,
        n_min         = float(North.min()),
        n_max         = float(North.max()),
        u_min         = float(Up.min()),
        u_max         = float(Up.max()),
        east_lo       = float(East.min()),
        east_hi       = float(East.max()),
        east_entry    = float(east_entry),
        layer_east_dx = float(layer_east_dx),
        n_planes      = int(n_planes),
    )
