"""v5 modules package.

Adds detector_optimization_v3 AND detector_optimization_v4 to sys.path so the
notebook can do:
    from modules.generate_showers    import GenerateShowers      # v3
    from modules.shower_computation  import ComputeShowerDetection
    from modules.detector_response   import SmearN, TimeAverage_vectorized
    from modules.utility_functions   import reconstructability, U_PR, U_E, U_angle
    from modules.reconstruction      import NormalizeLabels, DenormalizeLabels, EarlyStopping
    from modules_v4.tr_geometry      import load_tr_mountain
    from modules_v4.tr_surface_map   import SurfaceEastMap
    from modules_v4.tr_plane_kernel  import GetCounts_planeaware

v5 itself only adds the three EA-specific helpers in this folder:
    ev_deepsets.py    — DeepSetsReconstruction (permutation-invariant NN)
    ev_population.py  — Population dataclass + build_input_batch helper
    ev_selection.py   — fitness scoring, pruning, mutation operators
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.normpath(os.path.join(_HERE, "..", "..", "detector_optimization_v3"))
_V4   = os.path.normpath(os.path.join(_HERE, "..", "..", "detector_optimization_v4"))
for _p in (_V3, _V4):
    if _p not in sys.path:
        sys.path.insert(0, _p)
