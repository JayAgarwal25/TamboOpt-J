"""Mountain mesh loader: h5 -> ECEF -> site-local ENU.

Reads the detector region out of the mesh, rotates its triangle centroids from
ECEF into ENU anchored at the mesh's own `location`, and returns a
`MountainData`. The current mesh (`data/malata.h5`, group `malata`, key
`detector1`) gives 266 centroids over 162 unique vertices, spanning
North [-956, 716], East [-498, 777], Up [2748, 3712] m.

HDF5 layout:
    vertices  (3, V) float64 — ECEF metres
    faces     (3, F) int64   — JULIA 1-INDEXED vertex indices
    detector1 (D,)   int64   — JULIA 1-INDEXED face indices into `faces`
    location  (2,)           — [lon_deg, lat_deg] of the site

Gotchas:
  - `faces` and `detector1` are Julia 1-indexed — subtract 1 for Python.
  - `vertices` are ECEF, not ENU; they must be rotated at the site.
  - depth is z_cont = (EAST_ENTRY - East) / LAYER_EAST_DX, so with the current
    calibration the mesh spans z_cont 1.45 (east_hi) to 4.00 (east_lo) of the
    24 AllShowers planes.
"""

import math
from dataclasses import dataclass

import h5py
import numpy as np

# ── Default paths / constants ────────────────────────────────────────────────
DEFAULT_GEOMETRY_PATH = "../../TAMBOSim/resources/basic_geometry.h5"
DEFAULT_GROUP         = "colca_valley_30000"
DEFAULT_DET_KEY       = "detector1"
DEFAULT_N_PLANES      = 24
SITE_LON_DEG          = -72.279397
SITE_LAT_DEG          = -15.622267

# Retired colca calibration, kept only as this module's signature defaults.
# THE PIPELINE DOES NOT USE THESE — every caller passes EAST_ENTRY (1500) and
# LAYER_EAST_DX (500) from modules/constants.py, where the live values and the
# reason they must match c8_air_shower.cpp's real plane spacing are documented.
ALLSHOWERS_EAST_ENTRY = -212.0    # m — East at layer 0
ALLSHOWERS_LAYER_DX   =  307.0    # m — East depth per layer (positive)


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
    """The detector-region geometry every stage reads.

    `centroids_ENU` is (n_tri, 3) with columns [East, North, Up] in metres,
    the site-local ENU convention matching the h5 data files. The `n_/u_/east_`
    scalars are its bounding box.

    Depth into the shower is z_cont = (east_entry - East_det) / layer_east_dx,
    so only centroids with East < east_entry can see shower particles. With the
    current calibration the mesh spans z_cont 1.45 to 4.00 of the 24 planes.
    """

    centroids_ENU: np.ndarray    # (n_tri, 3) columns [East, North, Up]

    n_min:   float
    n_max:   float
    u_min:   float
    u_max:   float
    east_lo: float               # actual centroid East min (most negative)
    east_hi: float               # actual centroid East max (most positive)

    east_entry:    float         # East at AllShowers layer 0
    layer_east_dx: float         # East depth per layer [m], positive
    n_planes:      int           # number of AllShowers planes (24)

    # (n_v, 3) [East, North, Up] — the unique triangle vertices of the detector
    # region (real surface corner points). Denser and truer than the face
    # centroids; used by the differentiable surface map. Optional / None for
    # legacy MountainData built without it.
    vertices_ENU:  np.ndarray = None


    @property
    def plane_dx(self) -> float:
        """Alias kept for legacy callers: returns -layer_east_dx (signed East per layer)."""
        return -self.layer_east_dx

    def east_to_z_cont(self, east: float) -> float:
        """Convert an East value to a continuous AllShowers layer index."""
        return (self.east_entry - east) / self.layer_east_dx


# ── Loader stages ─────────────────────────────────────────────────────────────

def _read_detector_region(h5_path: str, group: str, det_key: str):
    """Read the mesh and select the detector-region faces.

    Returns (verts, faces, det_idx, h5_loc): the ECEF vertices (3, n_v), the
    0-indexed faces (3, n_faces), the 0-indexed detector-region face indices,
    and the mesh's own [lon_deg, lat_deg] site location.
    """
    with h5py.File(h5_path, "r") as f:
        g        = f[group]
        verts    = g["vertices"][...]          # (3, 90000) ECEF float64
        faces    = g["faces"][...] - 1         # (3, 179996) 0-indexed
        # `det_key` (e.g. "detector1") is NOT geometry — it is a 1-D array of
        # 1-based (Julia) FACE INDICES into `faces` that select the observation /
        # detector region (the deployable footprint) out of the full mesh. This is
        # what picks the slope the optimiser moves detectors over. For malata it is
        # 266 faces -> 162 unique vertices, a ~1.4 km patch at Up 2748-3712 m (a
        # ~32 deg ramp); the rest of `faces` is the wider terrain + the whole-globe
        # sphere and is deliberately excluded here. Subtract 1 for 0-based indexing.
        det_idx  = g[det_key][...] - 1         # (266,) obs-region face indices, 0-indexed
        h5_loc   = g["location"][...]          # [lon_deg, lat_deg]
    return verts, faces, det_idx, h5_loc


def _resolve_enu_origin(site_lon_deg: float, site_lat_deg: float, h5_loc: np.ndarray):
    """ENU origin: explicit arg > mesh `location` dataset > module default."""
    if site_lon_deg is None:
        site_lon_deg = float(h5_loc[0])
    if site_lat_deg is None:
        site_lat_deg = float(h5_loc[1])
    return site_lon_deg, site_lat_deg


def _triangle_centroids_ecef(verts: np.ndarray, faces: np.ndarray, det_idx: np.ndarray) -> np.ndarray:
    """Triangle centroids of the detector-region faces, in ECEF (3, n_tri)."""
    tri_verts      = verts[:, faces[:, det_idx]]    # (3, 3, 2161)
    centroids_ecef = tri_verts.mean(axis=1)          # (3, 2161)
    return centroids_ecef


def _centroids_to_enu(centroids_ecef: np.ndarray, site_lon_deg: float, site_lat_deg: float):
    """Rotate the centroids to local ENU about the site origin.

    Returns (centroids_ENU, East, North, Up): the (n_tri, 3) [East, North, Up]
    array plus its three rows, which the bounding-box scalars are taken from.
    """
    enu = _ecef_to_enu(centroids_ecef, site_lon_deg, site_lat_deg)  # [East, North, Up]
    East, North, Up = enu[0], enu[1], enu[2]

    centroids_ENU = np.stack([East, North, Up], axis=1)   # (2161, 3) [East, North, Up]
    return centroids_ENU, East, North, Up


def _detector_region_vertices_enu(
    verts:        np.ndarray,
    faces:        np.ndarray,
    det_idx:      np.ndarray,
    site_lon_deg: float,
    site_lat_deg: float,
) -> np.ndarray:
    """Unique triangle vertices of the detector region — the real surface corner
    points (denser + truer than face centroids for the differentiable surface
    map). Rotated to ENU about the same site origin.
    """
    uniq_v        = np.unique(faces[:, det_idx].reshape(-1))
    verts_enu     = _ecef_to_enu(verts[:, uniq_v], site_lon_deg, site_lat_deg)   # (3, n_v)
    return np.stack([verts_enu[0], verts_enu[1], verts_enu[2]], axis=1)  # (n_v, 3) [E,N,U]


# ── Top-level loader ──────────────────────────────────────────────────────────

def load_tr_mountain(
    h5_path:        str   = DEFAULT_GEOMETRY_PATH,
    group:          str   = DEFAULT_GROUP,
    det_key:        str   = DEFAULT_DET_KEY,
    east_entry:     float = ALLSHOWERS_EAST_ENTRY,
    layer_east_dx:  float = ALLSHOWERS_LAYER_DX,
    n_planes:       int   = DEFAULT_N_PLANES,
    site_lon_deg:   float = None,
    site_lat_deg:   float = None,
    # Legacy aliases (ignored if the above are set)
    east_min:       float = None,
    east_max:       float = None,
) -> MountainData:
    """Read the mesh, compute detector-region centroids in ENU, return MountainData.

    Callers pass the live calibration from `modules.constants`; this module's
    own defaults are the retired colca ones and are NOT what the pipeline uses.

    Args:
        h5_path, group, det_key : mesh file, HDF5 group, and the dataset of
                        1-indexed face indices for the detector region.
        east_entry    : East at AllShowers layer 0.
        layer_east_dx : East depth per layer [m], positive.
        n_planes      : number of AllShowers planes.
        site_lon_deg / site_lat_deg : ENU origin. **If None, taken from the mesh's
                        own `location`** so centroids land in the frame anchored
                        at THAT mesh — the malata mesh sits ~33 km east of colca,
                        so a wrong origin offsets everything by that much. Falls
                        back to SITE_LON_DEG/SITE_LAT_DEG only if the mesh has no
                        `location`.
        east_min / east_max : legacy, ignored. Remove from call sites.
    """
    verts, faces, det_idx, h5_loc = _read_detector_region(h5_path, group, det_key)

    site_lon_deg, site_lat_deg = _resolve_enu_origin(site_lon_deg, site_lat_deg, h5_loc)

    centroids_ecef = _triangle_centroids_ecef(verts, faces, det_idx)

    centroids_ENU, East, North, Up = _centroids_to_enu(centroids_ecef, site_lon_deg, site_lat_deg)

    vertices_ENU = _detector_region_vertices_enu(verts, faces, det_idx, site_lon_deg, site_lat_deg)

    return MountainData(
        centroids_ENU = centroids_ENU,
        vertices_ENU  = vertices_ENU,
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
