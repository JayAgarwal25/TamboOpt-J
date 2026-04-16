"""v6 modules package.

Adds detector_optimization_v3 and detector_optimization_v4 to sys.path so
downstream code can do:

    from modules.geometry            import Layouts
    from modules.reconstruction      import Reconstruction
    from modules.utility_functions   import reconstructability, U_PR, U_E, U_angle
    from modules.layout_optimization import LearnableXY
    from modules.generate_showers    import GenerateShowers
    from modules.detector_response   import SmearN, TimeAverage_vectorized

    from modules_v4.tr_geometry      import load_tr_mountain
    from modules_v4.tr_surface_map   import SurfaceEastMap
    from modules_v4.tr_plane_kernel  import GetCounts_planeaware

v6 itself only ships the new helpers in this folder:
    fnn_surrogate.py — FNN model + layout generators + dataset builder
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.normpath(os.path.join(_HERE, "..", "..", "detector_optimization_v3"))
_V4   = os.path.normpath(os.path.join(_HERE, "..", "..", "detector_optimization_v4"))

for _p in (_V3, _V4):
    if _p not in sys.path:
        sys.path.insert(0, _p)
