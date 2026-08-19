"""v6 modules package.

Self-contained: modules_v6.legacy_core vendors the files formerly loaded via
sys.path injection from detector_optimization_v3/modules/ and
detector_optimization_v4/modules_v4/ (those version folders have been
retired to the `legacy-full-repo` git branch). Old call sites used:

    from modules.geometry            import Layouts
    from modules.reconstruction      import Reconstruction
    from modules.utility_functions   import reconstructability, U_PR, U_E, U_angle
    from modules.layout_optimization import LearnableXY
    from modules.generate_showers    import GenerateShowers
    from modules.detector_response   import SmearN, TimeAverage_vectorized

    from modules_v4.tr_geometry      import load_tr_mountain
    from modules_v4.tr_surface_map   import SurfaceEastMap
    from modules_v4.tr_plane_kernel  import GetCounts_planeaware

New call sites:

    from modules_v6.legacy_core.geometry            import Layouts
    from modules_v6.legacy_core.reconstruction      import Reconstruction
    from modules_v6.legacy_core.utility_functions   import reconstructability, U_PR, U_E, U_angle
    from modules_v6.legacy_core.layout_optimization import LearnableXY
    from modules_v6.legacy_core.generate_showers    import GenerateShowers
    from modules_v6.legacy_core.detector_response   import SmearN, TimeAverage_vectorized
    from modules_v6.legacy_core.tr_geometry         import load_tr_mountain
    from modules_v6.legacy_core.tr_surface_map      import SurfaceEastMap
    from modules_v6.legacy_core.tr_plane_kernel     import GetCounts_planeaware

Remaining external dependency: modules_v6.legacy_core.generate_showers
requires the sibling repo /n/home05/zdimitrov/tambo/TAMBO-opt
(util.allshowers_related.generate_showers) to be present on disk. See that
file's docstring for details — this is intentionally not vendored into
TambOpt.

v6 itself also ships the newer helpers directly in this folder:
    fnn_surrogate.py — FNN model + layout generators + dataset builder
"""
