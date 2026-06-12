# ── Paths / constants (match v4's active script) ─────────────────────────────

import os


GEOMETRY_PATH = "/n/home05/zdimitrov/tambo/TambOpt/detector_optimization_v6/colca_valley.h5"
GEOMETRY_GROUP = "colca_valley_30000"
DET_KEY        = "detector1"
N_PLANES       = 24
EAST_ENTRY     = 1500.0
LAYER_EAST_DX  = 150.0

# Fixed architecture constants
N_DETECTORS = 100
PRIMARY_DIM = 5   # [dir_x, dir_y, dir_z, log_e_norm, pdg]

# Primary energy bounds (log10 GeV) for min-max normalization
LOG_E_MIN = 5.0   # log10(1e5 GeV)
LOG_E_MAX = 8.0   # log10(1e8 GeV)

RUN_LOCATION = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
SHOWER_CACHE   = os.path.join(RUN_LOCATION, "v6_run_00")

# Output folders. Edit these in-place to point a fresh run at a new tree
# (e.g. swap "test_v6_run_01_recentered" -> "v6_run_01" to write to the
# production location instead). 01_build_dataset.py writes to
# TRAINING_DATASET_FOLDER; 02 + 03 read from it; 02 writes fnn.pt to
# FNN_FOLDER; 03 writes recon.pt to RECON_FOLDER; 04 reads both.
TRAINING_DATASET_FOLDER = os.path.join(RUN_LOCATION, "test_v6_run_01_recentered")
FNN_FOLDER              = os.path.join(RUN_LOCATION, "test_v6_run_02_recentered")
RECON_FOLDER            = os.path.join(RUN_LOCATION, "test_v6_run_03_recentered")
# 04_optimize.py appends "_{scheme}" (one folder per init scheme).
OPT_FOLDER              = os.path.join(RUN_LOCATION, "test_v6_run_04_optimize")

# 01_build_dataset.py: per-shower xy translation so every shower's energy-
# weighted centroid lands at the mountain bbox center. Without this only
# ~23% of cache showers overlap the mountain. Set to False to keep raw
# cache positions (the production default before this knob existed).
RECENTER_TO_MOUNTAIN = True

# 02_train_fnn.py: fraction of training-set indices to keep (val set always
# full). 1.0 = use all 90% train split. Drop to e.g. 0.05 for smoke tests.
TRAIN_FRACTION = 1.00

NUM_SHOWERS = 500_000
# NUM_SHOWERS = 100_000
# NUM_SHOWERS = 5_000_000
# NUM_SHOWERS = 50
BATCH_SIZE  = 60
BATCH_SIZE_TRAIN  = 20

# ── Dual-species (paired) pipeline ────────────────────────────────────────────
# 00_generate_data_dual_species.py samples NUM_SHOWERS primaries ONCE and
# generates BOTH components per primary: electron rows 0..N-1 and muon rows
# N..2N-1 of the corpus share the same (energy, direction) — row i and row N+i
# are two components of ONE physical event. pdg column = species id (e=0, µ=1).
DUAL_SHOWER_CACHE_PATH = os.path.join(
    SHOWER_CACHE, f"cashed_showers_dual_{2 * NUM_SHOWERS}.pt")

# 02_train_fnn_deepsets.py log-compresses the T targets as log1p(T*T_LOG_SCALE);
# the dual-surrogate combination (modules_v6/dual_surrogate.py) must invert the
# same transform to average times in physical units, so the scale lives here.
T_LOG_SCALE = 1.0e8
