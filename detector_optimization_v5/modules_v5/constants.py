"""Configuration for the v5 (mu+lambda)-ES optimizer.

Single source of truth for all paths and hyperparameters. Every other v5 ES
module imports from here; edit only this file when tuning.

Design context
--------------
v5 reuses v6's frozen dual-species surrogate (FNN + recon) so the comparison
with v6's L-BFGS/DE isolates the *optimizer* difference only — same objective,
same surrogate, different search strategy.

The surrogate was trained on the North+Up coordinate convention with
N_DETECTORS=100, so the ES also optimizes 100-detector layouts.
"""
import os

# ── Geometry ──────────────────────────────────────────────────────────────────
# colca_valley.h5 is the mountain mesh (30,000 triangles, 2161 detector-region
# faces used to define the mountain boundary).
GEOMETRY_PATH  = (
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/"
    "TambOpt-zlt/detector_optimization_v6/colca_valley.h5"
)
GEOMETRY_GROUP = "colca_valley_30000"
DET_KEY        = "detector1"
N_PLANES       = 24
EAST_ENTRY     = 1500.0   # correct calibration (same as v4 active scripts + v6)
LAYER_EAST_DX  = 150.0

# ── Detector count ────────────────────────────────────────────────────────────
N_DETECTORS = 100   # matches v6 surrogate; ES operates in this 200-D layout space

# ── Frozen v6 surrogate paths ─────────────────────────────────────────────────
# jagarwal's 1.4M-pair dual-species surrogates (200k showers × 7 layouts).
_V6_RUNS = (
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
)
FNN_FOLDER   = os.path.join(_V6_RUNS, "test_v6_run_02_recentered")
RECON_FOLDER = os.path.join(_V6_RUNS, "test_v6_run_03_recentered_deepsets")

# ── Primary encoding bounds (must match v6/modules_v6/constants.py) ──────────
LOG_E_MIN = 5.0   # log10(1e5 GeV)
LOG_E_MAX  = 7.0   # log10(1e7 GeV) — matches jagarwal's v6 surrogates

# ── Utility weights — identical to v6/04_optimize_lbfgs_ensemble.py ──────────
# These values define the composite U = (W_THETA*u_θ + W_PHI*u_φ + W_E*u_E) / W_DIV.
# U_PR is computed for logging but excluded from the composite (matches v6 production).
W_THETA               = 1e2
W_PHI                 = 1e2
W_E                   = 2.5e2
W_DIV                 = 1e3
LAYOUT_THRESHOLD      = 5e-2   # min E_pred to count a detector as "firing"
RECONSTRUCT_THRESHOLD = 10.0   # min firing detectors for a shower to be reconstructed

# ── (mu+lambda)-ES hyperparameters ───────────────────────────────────────────
ES_MU          = 20    # parents kept each generation
ES_LAMBDA      = 5     # offspring per parent (total candidates = MU + MU*LAMBDA = 120)
ES_N_GEN       = 200   # max generations per restart
ES_N_RESTART   = 5     # independent restarts with different random seeds
ES_PLATEAU_TOL = 30    # early stop if best U improves by less than PLATEAU_EPS for this many consecutive gens
ES_PLATEAU_EPS = 1e-4  # minimum improvement to reset plateau counter
ES_SIGMA_INIT  = 200.0 # mutation sigma at generation 0 [m]
ES_SIGMA_FINAL = 20.0  # mutation sigma at generation N_GEN-1 [m] (geometric schedule)
ES_CROSSOVER_P = 0.3   # probability that an offspring is created via crossover before mutation

# Primary batch size for fitness evaluation (fixed across all generations and restarts
# so per-run U values are directly comparable).
N_EVAL_PRIMARIES = 512

# ── Primary sampling distribution ────────────────────────────────────────────
# Matches the AllShowers training domain used to build the surrogate's dataset.
COS_THETA_MIN = -0.17364818  # cos(100°) — most horizontal showers
COS_THETA_MAX =  0.5         # cos(60°)  — most vertical showers
PHI_MIN       =  0.0
PHI_MAX       =  6.2831853   # 2*pi

# ── Output ────────────────────────────────────────────────────────────────────
RUN_OUTPUT_DIR = (
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v5_es_runs"
)
