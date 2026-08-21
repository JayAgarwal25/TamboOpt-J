"""The mountain mesh, its surface, and placement of detectors on it.

`mountain` loads the h5 mesh into site-local ENU; `surface` is the
differentiable Up = g(North, East) map; `placement` draws and projects detector
layouts onto that surface; `primitives` holds the shared layout shapes.
"""

from .mountain import MountainData, load_tr_mountain
from .surface import SurfaceUpMap
from .placement import project_to_mountain_ne, sample_initial_layout_ne
from .primitives import Layouts

__all__ = ["MountainData", "load_tr_mountain", "SurfaceUpMap",
           "project_to_mountain_ne", "sample_initial_layout_ne", "Layouts"]
