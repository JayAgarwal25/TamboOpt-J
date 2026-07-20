# ── Paths / constants (match v4's active script) ─────────────────────────────

import os


# Mountain mesh. `load_tr_mountain` rotates the ECEF vertices into the
# site-local ENU frame anchored at the mesh's own `location` dataset, so the
# detector centroids share the (East, North, Up) origin used by the tau shower
# corpus (tau_wholesky.h5). The `malata` group holds a 266-face detector region
# at location [lon -71.97, lat -15.58]; its ENU bbox is
# North ∈ [-956, 716], East ∈ [-499, 777], Up ∈ [2748, 3712] m.
GEOMETRY_PATH = "/n/home05/zdimitrov/tambo/TambOpt/detector_optimization_v6/malata.h5"
GEOMETRY_GROUP = "malata"
DET_KEY        = "detector1"

# Resolved mesh path used by the optimizers/plots: prefer a copy of the configured
# mesh sitting next to the repo, else the absolute GEOMETRY_PATH. Callers used to
# recompute this with a hardcoded `colca_valley.h5` fallback — which now mismatches
# GEOMETRY_GROUP='malata' and would crash — so it is centralized here and tracks
# whatever mesh GEOMETRY_PATH points at.
_GEOM_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           os.path.basename(GEOMETRY_PATH))
GEOMETRY_PATH_RESOLVED = _GEOM_LOCAL if os.path.exists(_GEOM_LOCAL) else GEOMETRY_PATH
# AllShowers longitudinal geometry — derived from the CORSIKA 8 sampling setup in
# decay_locations/c8_air_shower.cpp (the sim the AllShowers model was trained on):
#   - 24 ObservationPlanes  ->  N_PLANES = 24  (native cloud col2 = integer layer 0..23).
#   - planes sit at  point_k = injection + k * 500 m * propDir  (k = 1..24; showerCore
#     at 12000 m = 24 * 500 m), i.e. 500 m apart ALONG the shower axis  ->  the physical
#     depth per layer index is 500 m, so LAYER_EAST_DX = 500.0.
# Why 500 (not the old, undocumented 150): the kernel maps a native integer layer L to
# depth via z_cont = (EAST_ENTRY - East)/LAYER_EAST_DX. For L to land at its true depth
# the per-layer scale MUST equal the real plane spacing (500 m); 150 compressed the
# shower to ~30% of its longitudinal extent. See c8_air_shower.cpp lines 434-643.
# EAST_ENTRY is the East gauge of layer 0 (the shower start = tau decay, C8 injectionPos);
# it cancels between the cloud shift and the detector z_cont (relative depth only), so its
# absolute value is a free gauge and does not affect labels.
N_PLANES       = 24
EAST_ENTRY     = 1500.0
LAYER_EAST_DX  = 500.0

# Detector spatial-response Gaussian kernel width [m] in the plane-aware kernel
# (compute_labels_batch → GetCounts_planeaware). Reduced 200 → 50 for the malata
# array: its ~1.4 km surface packs 100 detectors at ~120 m spacing, so a 200 m
# kernel over-smoothed neighbouring detectors together. Used at dataset-build time
# (Step 1 labels) and by the aleatoric-floor script; the trained surrogate then
# inherits this resolution.
SIGMA_SPATIAL  = 250.0

# Fixed architecture constants
N_DETECTORS = 100
PRIMARY_DIM = 5   # [dir_x, dir_y, dir_z, log_e_norm, pdg]  (pdg = EM/hadronic primary class, 0/1)

# Primary energy bounds (log10 GeV) for min-max normalization
LOG_E_MIN = 5.0   # log10(1e5 GeV)
LOG_E_MAX = 7.0   # log10(1e8 GeV)

# Direction bounds for sampling priamries
ZENITH_MIN   = 60.0  # degrees
ZENITH_MAX   = 100.0 # degrees
AZIMUTH_MIN  = 0.0   # degrees
AZIMUTH_MAX  = 360.0 # degrees


RUN_LOCATION = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
SHOWER_CACHE   = os.path.join(RUN_LOCATION, "v6_run_00")

TRAINING_DATASET_FOLDER = os.path.join(RUN_LOCATION, "test_v6_run_01_northeast")
FNN_FOLDER              = os.path.join(RUN_LOCATION, "test_v6_run_02_recentered")
RECON_FOLDER            = os.path.join(RUN_LOCATION, "test_v6_run_03_recentered")
# 04_optimize.py appends "_{scheme}" (one folder per init scheme).
OPT_FOLDER              = os.path.join(RUN_LOCATION, "test_v6_run_04_optimize")

# 01_build_dataset.py: per-shower xy translation so every shower's energy-
# weighted centroid lands at the mountain bbox center. Without this only
# ~23% of cache showers overlap the mountain. Set to False to keep raw
# cache positions (the production default before this knob existed).
RECENTER_TO_MOUNTAIN = False # TODO remove this functionality

# 02_train_fnn.py: fraction of training-set indices to keep (val set always
# full). 1.0 = use all 90% train split. Drop to e.g. 0.05 for smoke tests.
TRAIN_FRACTION = 1.00

# 01_build_dataset(_northeast).py: fraction of the dual corpus to LOAD into the
# dataset build, applied per species. 1.0 = all 2*NUM_SHOWERS rows, which dense
# is ~501 GB and OOMs at --mem=100g. 0.10 keeps the first 10% of each species
# block (~50 GB dense), so both electron and muon stay represented.
DATASET_FRACTION = 1.00

# NUM_SHOWERS = 500_000
NUM_SHOWERS = 100_000
# NUM_SHOWERS = 5_000_000
# NUM_SHOWERS = 1_000
# NUM_SHOWERS = 100
BATCH_SIZE  = 60
BATCH_SIZE_TRAIN  = 20

# ── Dual-species (paired) pipeline ────────────────────────────────────────────
# 00_generate_data_dual_species.py samples NUM_SHOWERS primaries ONCE and
# generates BOTH components per primary: electron rows 0..N-1 and muon rows
# N..2N-1 of the corpus share the same (energy, direction, EM/hadronic class) —
# row i and row N+i are two components of ONE physical event. The corpus pdg
# column = the EM/hadronic primary class (0/1), randomly sampled by
# sample_primary_particles and fed to the generator as its conditioning label.
# ── Real tau primaries (tau_wholesky.h5) ─────────────────────────────────────
# When USE_TAU_PRIMARIES, Step 0 draws its primaries (energy, direction, and a
# physical ENU decay POSITION) from tau_wholesky.h5 instead of the synthetic
# `sample_primary_particles`, and Step 1 places each shower at its real position
# (via the `<corpus>_positions.pt` sidecar) instead of re-centering it onto the
# mountain. tau_wholesky.h5 is in the SAME site-local ENU frame the mountain mesh
# defines (origin = the mesh `location`), so mountain and showers share (E,N,U)=0.
# Energies are filtered to the generator's trained band [10**LOG_E_MIN,
# 10**LOG_E_MAX] GeV inside the loader.
USE_TAU_PRIMARIES = True
TAU_WHOLESKY_PATH = "/n/home05/zdimitrov/tambo/TambOpt/detector_optimization_v6/decay_locations/tau_wholesky.h5"
TAU_CORPUS_PATH   = os.path.join(SHOWER_CACHE, "cashed_showers_tau_dual.pt")

# Corpus the Step-1 builder reads. Tau runs use a fixed-name file (the pair count
# is only known after energy filtering); synthetic runs keep the count-based name.
DUAL_SHOWER_CACHE_PATH = (
    TAU_CORPUS_PATH if USE_TAU_PRIMARIES
    else os.path.join(SHOWER_CACHE, f"cashed_showers_dual_{2 * NUM_SHOWERS}.pt"))
# Per-row e/µ species id (0=electron block, 1=muon block) — which secondary
# COMPONENT a row is. Written by Step 0 alongside the corpus (showerdata.Showers
# has no species field; its pdg now carries the EM/hadronic class). Row-aligned
# with the corpus: [0]*NUM_SHOWERS + [1]*NUM_SHOWERS. Default for the canonical
# corpus; derived from the corpus path by the same `<corpus>_species.pt` rule the
# Step-1 builders use, so it tracks DUAL_SHOWER_CACHE_PATH automatically.
DUAL_SPECIES_IDS_PATH = os.path.splitext(DUAL_SHOWER_CACHE_PATH)[0] + "_species.pt"
# Per-row ENU decay position (M, 3) columns (East, North, Up), row-aligned with
# the corpus (electron block then muon block, both sharing the primary → same
# position). Written by Step 0 when USE_TAU_PRIMARIES; Step 1 places each cloud at
# this real position instead of re-centering to the mountain. Same `<corpus>_*`
# sidecar rule so it tracks DUAL_SHOWER_CACHE_PATH automatically.
DUAL_POSITIONS_PATH = os.path.splitext(DUAL_SHOWER_CACHE_PATH)[0] + "_positions.pt"

# 02_train_fnn_deepsets.py log-compresses the T targets as log1p(T*T_LOG_SCALE);
# the dual-surrogate combination (modules_v6/dual_surrogate.py) must invert the
# same transform to average times in physical units, so the scale lives here.
T_LOG_SCALE = 1.0e8
