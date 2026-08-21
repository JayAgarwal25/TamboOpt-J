"""TR geometry loader for detector_optimization_v4.

Reads TAMBOSim/resources/basic_geometry.h5, projects the 2161 detector-region
triangle centroids from ECEF to local ENU, and returns a MountainData dataclass
used by the surface map and the main notebook.

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

    centroids_ENU : (n_tri, 3) float64 numpy array, columns = [East, North, Up] in
                    metres — the site-local ENU convention that matches the h5 data
                    files. `centroids_NUE` is a backward-compat property returning
                    the old [North, Up, East] column order for legacy callers.
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
    centroids_ENU: np.ndarray    # (n_tri, 3) columns [East, North, Up]

    n_min:   float
    n_max:   float
    u_min:   float
    u_max:   float
    east_lo: float               # actual centroid East min (most negative)
    east_hi: float               # actual centroid East max (most positive)

    east_entry:    float         # East at AllShowers layer 0 (default -212 m)
    layer_east_dx: float         # East depth per layer (default 307 m, positive)
    n_planes:      int           # number of AllShowers planes (24)

    # (n_v, 3) [East, North, Up] — the unique triangle vertices of the detector
    # region (real surface corner points). Denser and truer than the face
    # centroids; used by the differentiable surface map. Optional / None for
    # legacy MountainData built without it.
    vertices_ENU:  np.ndarray = None

    # @property
    # def centroids_NUE(self) -> np.ndarray:
    #     """Backward-compat view: the old [North, Up, East] column order, derived
    #     from the canonical ENU field. Legacy callers (v4 scripts, the base
    #     North-Up module family) keep working unchanged; new code should use
    #     `centroids_ENU` ([East, North, Up])."""
    #     return self.centroids_ENU[:, [1, 2, 0]]

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
    """Read basic_geometry.h5, compute detector-region centroids in ENU, return MountainData.

    Args:
        h5_path       : path to basic_geometry.h5.
        group         : HDF5 group name (default 'colca_valley_30000').
        det_key       : dataset key for the detector-region triangle indices (default 'detector1').
                        This is a (2161,) array of 1-indexed face indices — subtract 1 in Python.
        east_entry    : East at AllShowers layer 0 (default -212 m, empirically calibrated).
        layer_east_dx : East depth per layer in metres (default 307 m, positive).
        n_planes      : number of AllShowers planes (default 24).
        site_lon_deg / site_lat_deg : ENU origin (site) longitude/latitude in
                        degrees. If None, taken from the mesh's own `location`
                        dataset ([lon, lat]) so the centroids land in the
                        site-local ENU frame anchored at THAT mesh (e.g. the
                        `malata` mesh sits ~33 km east of the colca site — using
                        the wrong origin offsets it by that much). Falls back to
                        the module SITE_LON_DEG/SITE_LAT_DEG constants only when
                        the mesh has no `location`. For the colca mesh the
                        `location` dataset equals those constants, so existing
                        callers are unaffected.
        east_min / east_max : legacy parameters, ignored.  Remove from call sites.
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
