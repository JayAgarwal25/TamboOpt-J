# ── Paths / constants (match v4's active script) ─────────────────────────────

import os


# Mountain mesh. `load_tr_mountain` rotates ECEF vertices into site-local ENU
# anchored at the mesh's own `location`, so detector centroids share the origin
# used by the tau corpus. `malata` = 266-face detector region, ENU bbox
# North [-956, 716], East [-499, 777], Up [2748, 3712] m.
GEOMETRY_PATH = "/n/home05/zdimitrov/tambo/TambOpt/detector_optimization_v6/malata.h5"
GEOMETRY_GROUP = "malata"
DET_KEY        = "detector1"

# Prefer a copy of the mesh next to the repo, else the absolute path. Centralized
# so callers don't recompute it with a stale `colca_valley.h5` fallback.
_GEOM_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           os.path.basename(GEOMETRY_PATH))
GEOMETRY_PATH_RESOLVED = _GEOM_LOCAL if os.path.exists(_GEOM_LOCAL) else GEOMETRY_PATH

# AllShowers longitudinal geometry, from decay_locations/c8_air_shower.cpp:
# 24 ObservationPlanes 500 m apart along the shower axis. LAYER_EAST_DX MUST equal
# that real spacing — the old 150 compressed showers to ~30% of their extent.
# EAST_ENTRY is the East gauge of layer 0; it cancels against the detector z_cont,
# so its absolute value is free.
N_PLANES       = 24
EAST_ENTRY     = 1500.0
LAYER_EAST_DX  = 500.0

# Detector spatial-response Gaussian width [m]. 200 → 50 for malata: 100 detectors
# at ~120 m spacing over ~1.4 km, so 200 m over-smoothed neighbours together.
# Set at dataset-build time; the trained surrogate inherits this resolution.
SIGMA_SPATIAL  = 50.0

# Fixed architecture constants
N_DETECTORS = 100
# [dir_x, dir_y, dir_z, log_e_norm, pdg, rel_E, rel_N, rel_U]
#   pdg       = EM/hadronic primary class 0/1 (NOT the e/µ species)
#   rel_E/N/U = decay vertex relative to the array centre [m]
# The vertex is needed because tau_wholesky.jl aims every tau at the array, so
# direction barely discriminates (aleatoric floor R^2 >= 0.49 without it, >= 0.56
# with). Cols 0-3 keep their meaning, so Step 3's `primary[:, :4]` target is
# unchanged. Bumping this invalidates every checkpoint — rebuild from Step 1.
PRIMARY_DIM = 8

# Primary energy bounds (log10 GeV) for min-max normalization
LOG_E_MIN = 5.0   # log10(1e5 GeV)
LOG_E_MAX = 7.0   # log10(1e7 GeV)

# Direction bounds for sampling primaries
ZENITH_MIN   = 60.0  # degrees
ZENITH_MAX   = 100.0 # degrees
AZIMUTH_MIN  = 0.0   # degrees
AZIMUTH_MAX  = 360.0 # degrees


RUN_LOCATION = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/07_750k_primaires_meanvar"
SHOWER_CACHE   = os.path.join(RUN_LOCATION, "v6_run_00")

TRAINING_DATASET_FOLDER = os.path.join(RUN_LOCATION, "test_v6_run_01_northeast")
FNN_FOLDER              = os.path.join(RUN_LOCATION, "test_v6_run_02_recentered")
RECON_FOLDER            = os.path.join(RUN_LOCATION, "test_v6_run_03_recentered")
# 04_optimize.py appends "_{scheme}" (one folder per init scheme).
OPT_FOLDER              = os.path.join(RUN_LOCATION, "test_v6_run_04_optimize")

# 02: fraction of training-set indices to keep (val set always full).
TRAIN_FRACTION = 1.00
# 01: fraction of the dual corpus to load, applied per species so both stay
# represented. 1.0 is ~501 GB dense and OOMs at --mem=100g.
DATASET_FRACTION = 1.00

# Descriptive only (also the synthetic-primaries fallback count in Step 0).
# With USE_TAU_PRIMARIES the real event count comes from tau_wholesky_n750k.h5
# (~751,931 surviving events) minus the HOLDOUT_FRAC reserve below; Step 1's
# MAX_SHOWERS no longer derives from this (see 01_build_dataset_northeast.py),
# so an approximate value here can't silently truncate the corpus.
NUM_SHOWERS = 750_000
BATCH_SIZE  = 60
BATCH_SIZE_TRAIN  = 20

# ── Held-out final-eval reserve ───────────────────────────────────────────────
# A fixed fraction of PHYSICAL EVENTS, split off in Step 0 before generation and
# written to its own corpus file/sidecars. Steps 1-4 (all FNN/recon train+val
# splits and the stage-4 layout optimizer) only ever read the main corpus below;
# this reserve is exclusively for eval_true_utility.py's final scoring, so a
# layout is never evaluated on the same events used to fit it or train the
# surrogate/recon. Independent seed — unrelated to FNN's (seed 0) or recon's
# (seed 1) own val splits, which stay internal to the main (non-holdout) pool.
HOLDOUT_FRAC = 0.05
HOLDOUT_SEED = 999

# ── Dual-species (paired) pipeline ────────────────────────────────────────────
# Step 0 samples NUM_SHOWERS primaries ONCE and generates both components: rows i
# and N+i share one physical event's (energy, direction, EM/hadronic class).
#
# With USE_TAU_PRIMARIES, primaries come from tau_wholesky.h5 (energy, direction,
# and a real ENU decay position) instead of the synthetic sampler. That file uses
# the same site-local ENU frame as the mesh, so showers and mountain share the
# origin. Energies are filtered to [10**LOG_E_MIN, 10**LOG_E_MAX] in the loader.
USE_TAU_PRIMARIES = True
TAU_WHOLESKY_PATH = "/n/home05/zdimitrov/tambo/TambOpt/detector_optimization_v6/decay_locations/tau_wholesky_n750k.h5"

# TAU_WHOLESKY_PATH = ".../decay_locations/tau_wholesky_n2M_83k.h5"  # built the current corpus
TAU_CORPUS_PATH   = os.path.join(SHOWER_CACHE, "cashed_showers_tau_dual.pt")

# Corpus Step 1 reads. Tau runs use a fixed name (pair count is only known after
# energy filtering); synthetic runs keep the count-based name.
DUAL_SHOWER_CACHE_PATH = (
    TAU_CORPUS_PATH if USE_TAU_PRIMARIES
    else os.path.join(SHOWER_CACHE, f"cashed_showers_dual_{2 * NUM_SHOWERS}.pt"))
# Step-0 sidecars, row-aligned with the corpus. Both derive from the corpus path
# by the `<corpus>_*.pt` rule the Step-1 builders use, so they track it.
#   _species.pt   : e/µ component id (0=electron block, 1=muon block) — the corpus
#                   `pdg` carries the EM/hadronic class, so Step 2 splits on this.
#   _positions.pt : (M, 3) ENU decay vertex (East, North, Up); drives both the
#                   placement and the primary encoding's rel_E/N/U.
DUAL_SPECIES_IDS_PATH = os.path.splitext(DUAL_SHOWER_CACHE_PATH)[0] + "_species.pt"
DUAL_POSITIONS_PATH = os.path.splitext(DUAL_SHOWER_CACHE_PATH)[0] + "_positions.pt"

# Held-out corpus (HOLDOUT_FRAC of events, see above) — same file layout as the
# main corpus/sidecars, written by Step 0, read only by eval_true_utility.py.
HELDOUT_SHOWER_CACHE_PATH = os.path.splitext(DUAL_SHOWER_CACHE_PATH)[0] + "_heldout.pt"
HELDOUT_SPECIES_IDS_PATH  = os.path.splitext(HELDOUT_SHOWER_CACHE_PATH)[0] + "_species.pt"
HELDOUT_POSITIONS_PATH    = os.path.splitext(HELDOUT_SHOWER_CACHE_PATH)[0] + "_positions.pt"

# 02 log-compresses T targets as log1p(T*T_LOG_SCALE); dual_surrogate.py must
# invert the same transform, so the scale lives here. The normalized
# kernel-weighted mean arrival times (GetCounts_planeaware, time_normalized=True)
# are ~1e-6..1e-4 s, so log1p(T*1e6) spans roughly 0.7..4.6. The old 1e8 was
# calibrated to the legacy unnormalized T scale (kernel-weighted sum over the
# padded point count), which ran orders of magnitude smaller.
T_LOG_SCALE = 1.0e6
