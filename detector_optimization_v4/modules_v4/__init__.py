"""v4 modules package.

Adds detector_optimization_v3 to sys.path so the notebook can do:
    from modules.layout_optimization import LearnableXY
    from modules.detector_response   import SmearN, TimeAverage_vectorized
    ...
as if it were running inside the v3 folder.

v4 itself only adds the three TR-specific helpers in this folder:
    tr_geometry.py     — HDF5 loader, ECEF→ENU, MountainData dataclass
    tr_surface_map.py  — SurfaceEastMap: differentiable East = f(N, Up)
    tr_plane_kernel.py — GetCounts_planeaware: v3 Gaussian kernel + triangular plane weight
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.normpath(os.path.join(_HERE, "..", "..", "detector_optimization_v3"))
if _V3 not in sys.path:
    sys.path.insert(0, _V3)
