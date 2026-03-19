# TambOpt — TAMBO Detector Optimization & Simulation Suite

TambOpt is a differentiable end-to-end pipeline for optimizing cosmic ray detector layouts for the TAMBO Observatory. It combines CORSIKA-based air shower simulation, diffusion-model surrogates for fast shower generation, differentiable detector response modeling, neural network reconstruction, and gradient-based layout optimization — all within a PyTorch autograd framework.

---

## Pipeline Overview

```
1. SIMULATION                 2. SURROGATE MODELING           3. DETECTOR RESPONSE
   CORSIKA C++ app               Diffusion model (24-plane       Bilinear interpolation
   → particle shower tracks      RGB images) + FNN bbox          of shower at detector
   → cluster batch jobs           predictor                      positions, smearing,
     (SLURM)                                                     timing (differentiable)

4. RECONSTRUCTION             5. OPTIMIZATION
   4-layer MLP:                  Learnable (x, y) positions
   [N, T] × detectors            Utility = αU_PR + βU_E + γU_TH
   → [X0, Y0, E, θ, φ]          Adam optimizer + constraints
                                  (min separation, boundary)
```

All steps from surrogate generation onward are differentiable, enabling gradient-based optimization without Monte Carlo sampling overhead.

---

## Repository Structure

```
TambOpt/
├── corsika_application/           # C++ CORSIKA v8 particle shower simulator
│   ├── source/                    #   C++ source code
│   └── README.md                  #   Build instructions (FLUKA, Conan, GCC 13.2)
│
├── cluster_scripts/               # SLURM batch job generation
│   └── submit_tambo_jobs.py       #   Log-uniform energy sampling, random angles,
│                                  #   multiple PDG IDs (π±, π0, e±)
│
├── detector_optimization_v2/      # Main optimization pipeline (refactored)
│   ├── geometry.py                #   Concentric ring layouts, triangle boundary projection
│   ├── shower_generation.py       #   Diffusion + FNN shower generation
│   ├── detector_response.py       #   Differentiable counts, smearing, timing
│   ├── reconstruction.py          #   4-layer MLP for shower parameter estimation
│   ├── layout_optimization.py     #   Learnable detector positions, constraints
│   ├── tambo_physics.py           #   Direction, timing, LDF models (from TamboDirReco.jl)
│   ├── utility_functions.py       #   Reconstructability, energy, angular utilities
│   ├── diffusion_model/           #   PlaneDiffusionEvaluator, PlaneFNNGenerator
│   ├── SWGOLO7_optimization.ipynb #   Main optimization notebook
│   ├── SWGOLO7_plots.ipynb        #   Results visualization
│   ├── auto_run_notebook.py       #   Papermill-based automated execution
│   ├── common_gpu_auto_run_notebook_batch.sh  # GPU batch runner
│   ├── outputs/                   #   Data from optimization runs (NN_Files_9 through _15)
│   └── outputs_notebooks/         #   Timestamped output notebooks
│
├── detector_optimization/         # Original optimization pipeline (v1, exploratory)
│   ├── SWGOLO7*.ipynb             #   Earlier optimization notebooks
│   ├── simple_simulator_class/    #   Experimental simulator (simulator.py)
│   ├── diffusion_model/           #   Original diffusion generator
│   └── data_exploration/          #   EDA notebooks
│
├── ml/                            # ML training pipeline
│   ├── nn.py                      #   Feedforward NN (hit-level regression)
│   ├── gnn.py                     #   Graph NN with PyTorch Geometric (PDG classification)
│   ├── pre_processing_per_hit.py  #   Hit-level preprocessing
│   ├── combine.py                 #   Dataset merging
│   ├── normalization_*.py         #   Normalization utilities
│   ├── scaling_NN/                #   Scaling neural network pipeline
│   │   ├── preprocessing/         #     3-step preprocessing (step1–3) + SLURM auto-submit
│   │   ├── FNN/                   #     Feedforward bbox predictor (24 planes × 4 coords)
│   │   └── diffusion_model/       #     Diffusion-based bbox generation (UNet, DDIM) [Unsuccessful]
│   └── README.md                  #   Pipeline execution order
│
├── 2d_diffusion_model/            # Standalone diffusion model evaluation
│   ├── tambo_diffusion_evaluation.py
│   └── tambo_flow_evalution.py
│
├── notebooks/   # Research notebooks
│   ├── 01–04: NN architecture, EDA, FNN training
│   ├── 05–08: Physics-Informed NN (PINA) experiments
│   └── 09: Diffusion model output analysis
│
└── resources/                     # Plot styles and themes
```

---

## Core Modules (detector_optimization_v2)

| Module | Purpose |
|--------|---------|
| `geometry.py` | `Layouts()` generates concentric ring arrays; `project_to_triangle()` enforces site boundary via barycentric coords |
| `shower_generation.py` | `GenerateShowers()` produces synthetic events using diffusion model + FNN bbox predictor, conditioned on energy/angles |
| `detector_response.py` | `GetCounts_differentiable()` computes bilinear-interpolated counts; `SmearN()` adds 5% resolution + threshold; `TimeAverage_vectorized()` handles timing |
| `reconstruction.py` | `Reconstruction` NN: input [x, y, N, T] per detector -> output [X0, Y0, E, theta, phi]; includes normalization helpers and early stopping |
| `layout_optimization.py` | `LearnableXY` wraps positions as `nn.Parameter`; `push_apart()` enforces min separation; `symmetry_loss()` penalizes asymmetry |
| `tambo_physics.py` | Direction/rotation utilities, `great_circle_distance()`, quadratic timing delay, power-law LDF model, Poisson/Gaussian likelihoods (ported from TamboDirReco.jl) |
| `utility_functions.py` | `reconstructability()` (soft detection threshold), `U_PR` (detection utility), `U_E` (energy utility), `U_TH` (angular utility) |

---

## Optimization Objective

The layout optimizer maximizes a weighted utility:

**U = alpha * U_PR + beta * U_E + gamma * U_TH**

- **U_PR**: Fraction of reconstructable events (soft threshold on detector counts)
- **U_E**: Energy reconstruction accuracy (weighted by reconstructability)
- **U_TH**: Angular reconstruction accuracy (great-circle distance on sphere)

Constraints are applied after each gradient step: minimum detector separation (`push_apart`) and triangular site boundary projection.

---

## Technology Stack

- **Simulation**: CORSIKA v8 (C++), FLUKA
- **ML Framework**: PyTorch, PyTorch Lightning, PyTorch Geometric
- **Generative Models**: Custom diffusion models (DDIM sampling), FNN regressors
- **Compute**: SLURM cluster (GPU partitions), Papermill for notebook automation
- **Data**: NumPy, Pandas, SciPy, h5py
- **Visualization**: Matplotlib

---

## Key Usage Notes

- CORSIKA builds require FLUKA and Conan dependencies — see `corsika_application/README.md` for cluster-specific env vars.
- The v2 pipeline (`detector_optimization_v2/`) supersedes v1 with modular, well-separated components.
- Optimization outputs are saved as `Layout_N.txt` files with (x, y) detector coordinates, along with checkpoints and cached tensors.
- `auto_run_notebook.py` uses Papermill to execute notebooks with different parameters for hyperparameter sweeps.
