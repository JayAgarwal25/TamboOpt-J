# detector_optimization_v3

Third iteration of the TAMBO detector-layout optimization pipeline. The big shift from v2 is the **shower surrogate**: v3 swaps the 24-plane diffusion image generator for **AllShowers** — a flow-matching point-cloud generator from the parent `TAMBO-opt` project — and gains a port of the `TamboDirReco.jl` physics utilities (great-circle distance, LDF, timing).

Two consequences of moving to point clouds:

- `detector_response.GetCounts_differentiable` now integrates a **Gaussian kernel** over the point cloud (gradient flows through `dist² → x_det, y_det`) instead of bilinear-sampling an RGB image.
- The pipeline drops bbox normalization and per-plane denormalization — coordinates come out of the generator in physical units already.

End-to-end flow:

```
sample primary particles  ──▶  AllShowers point cloud  ──▶  Gaussian-kernel detector counts (N, T)
        (E, zenith, azimuth)        (samples: x,y,layer,e,t)        (differentiable in x_det, y_det)
                                                                           │
                                                                           ▼
                                                        Reconstruction MLP → [X0, Y0, E, θ, φ]
                                                                           │
                                                                           ▼
                                              Utility U = α·U_PR + β·U_E + γ·U_TH (great-circle)
                                                          backprop → LearnableXY → push_apart
```

---

## Modules (`modules/`)

| File | Public API | Purpose |
|------|-----------|---------|
| `generate_showers.py` | `GenerateShowers` (callable class) | Wraps `allshowers.generate_showers` — samples primary particles in `[e_min, e_max] × [zenith_min, zenith_max] × [azimuth_min, azimuth_max]`, predicts point counts, runs the AllShowers solver. `__call__(num_samples, save)` returns `(samples, energies, directions, labels)` where `samples` is `(N, max_points, 5) [x, y, layer, energy, time]` and the rest are primary event parmters. |
| `shower_computation.py` | `ComputeShowerDetection`, `ReadShowers`, `denormalize_shower` | Pipeline-side wrapper: calls a `GenerateShowers` instance (with optional disk caching by `output_dir/cashed_showers_<N>.pt`), optional plane filtering, computes `(N, T, X0, Y0, energies, directions, labels)` via the differentiable count function |
| `detector_response.py` | `GetCounts_differentiable`, `SmearN`, `TimeAverage_vectorized` | Per-detector Gaussian kernel `exp(-dist²/(2σ²))` over all points, energy-weighted intensity and arrival time. `sigma` controls effective collection radius. Called by `ComputeShowerDetection`. |
| `tambo_physics.py` | `get_dir_vec`, `get_angles`, `flip_dir`, `get_rotation_matrix`, `great_circle_distance(_deg)`, `timing_delay_quadratic`, `timing_likelihood`, `ldf_model`, `ldf_likelihood`, `U_TH_great_circle` | PyTorch port of `TamboDirReco.jl` (`utils.jl`, `reco.jl`). Direction algebra, great-circle distance for angular utility, quadratic timing-delay + Gaussian likelihood, power-law LDF with Poisson likelihood. **Unused at this point.** |
| `reconstruction.py` | `Reconstruction`, `NormalizeLabels`, `DenormalizeLabels`, `EarlyStopping` | 4-layer MLP. **`NormalizeLabels` now maps to [0, 1]** with explicit `(e_min, e_max, theta_min, theta_max, phi_min, phi_max)` (defaults: `1e5–1e8` energy, `60°–100°` zenith, full azimuth) — replaces v2's `[-1, 1]` mapping |
| `geometry.py` | `Layouts`, `project_to_triangle`, `barycentric_coords` | Concentric-ring layout + triangular site-boundary projection (legacy, mostly unused — see v2 README) |
| `layout_optimization.py` | `LearnableXY`, `push_apart`, `symmetry_loss` | Detector positions as `nn.Parameter`; in-place pairwise repulsion; n-fold symmetry penalty |
| `utility_functions.py` | `reconstructability`, `U_PR`, `U_E`, `U_angle` | Soft-threshold reconstructability score; reconstructability-weighted utilities (`U_TH_great_circle` lives in `tambo_physics.py`) |
| `auto_run_notebook.py` | `run_notebook` | Papermill executor with timestamped outputs |

---

## Contents

```
detector_optimization_v3/
├── modules/                                              # All importable Python code
│   ├── generate_showers.py                               #   AllShowers wrapper (point clouds)
│   ├── shower_computation.py                             #   ComputeShowerDetection pipeline glue
│   ├── detector_response.py                              #   Gaussian-kernel counts + timing
│   ├── tambo_physics.py                                  #   TamboDirReco.jl port
│   ├── reconstruction.py                                 #   MLP + [0,1] normalization
│   ├── geometry.py                                       #   Layouts + boundary projection
│   ├── layout_optimization.py                            #   LearnableXY + constraints
│   ├── utility_functions.py                              #   U_PR / U_E / U_angle
│   └── auto_run_notebook.py                              #   Papermill runner
│
├── SWGOLO7_optimization.ipynb                            # Main optimization notebook
├── SWGOLO7_optimization_with_tambo_physics.ipynb         # Variant: U_TH = great-circle distance. AI generated, never used.
├── SWGOLO7_plots.ipynb                                   # Results visualization
├── explore_average_shower_data_diffusion_model_new.ipynb # EDA on AllShowers point clouds
└── output_notebooks/                                     # Timestamped papermill outputs
```

---

## Shower Surrogate: AllShowers

`GenerateShowers` calls into `allshowers.generate_showers` (path hard-coded to `/n/home05/zdimitrov/tambo/TAMBO-opt/allshowers/`):

1. `sample_primary_particles(...)` — uniform sampling in the configured `(E, zenith, azimuth)` ranges with categorical labels.
2. `run_point_count_fm(...)` — flow-matching model predicts the number of points per shower from `(energies, directions, labels)`. Default checkpoint: `num_of_point_clouds_dequantize_compiled.pt`.
3. `run_allshowers(...)` — the AllShowers ODE solver (`solver="midpoint"` by default, `num_timesteps=16`) generates the actual point cloud `(N, max_points, 5)` with columns `[x, y, layer_index, energy, time]` in physical units.
4. Optional `save_output(...)` to a `.pt` cache so subsequent runs can `torch.load` instead of re-simulating.

`ComputeShowerDetection` checks `output_dir/cashed_showers_<N>.pt` first; if present, it reloads and skips the call entirely. Pass `filter_plane=<int>` to keep only points whose `layer_index` matches.

---

## Detector Response: Gaussian-Kernel Counts

`GetCounts_differentiable(samples, x_det, y_det, ...)` is the v3 replacement for v2's bilinear `F.grid_sample`:

```
dx = point_x − x_det                            # (B, max_points, num_det)
dy = point_y − y_det
kernel = exp(-(dx² + dy²) / (2 σ²))             # σ = collection radius
local_intensity = Σ_points (energy · kernel)    # (B, num_det)
et              = Σ (time · energy · kernel) / local_intensity
```

Tune `sigma` to match the physical detector pitch. Gradients flow `local_intensity → kernel → dist² → x_det, y_det`, so the layout optimizer can move detectors to maximize energy collection / reconstructability.

---

## Optimization Objective

```
U = α · U_PR  +  β · U_E  +  γ · U_TH
```

- **`U_PR`** (`utility_functions.U_PR`) — reconstructability score, soft-thresholded over per-detector counts
- **`U_E`** (`utility_functions.U_E`) — reconstructability-weighted energy MSE-style utility
- **`U_TH`** — angular utility. `utility_functions.U_angle` — uses raw `(theta_pred − theta_true)²`

After each gradient step, `layout_optimization.push_apart` enforces minimum detector separation.

---

## Running on the Cluster

`detector_optimization_v3/` does not ship its own SBATCH scripts — re-use the v2 ones, pointing them at the v3 notebook:

```bash
# Adapt ../detector_optimization_v2/common_gpu_auto_run_notebook_batch.sh
# and change the trailing line to:
python modules/auto_run_notebook.py SWGOLO7_optimization.ipynb
# (or run with papermill directly)
```

`auto_run_notebook.py` writes timestamped outputs to `output_notebooks/` (note: singular `output_notebooks/`, unlike v2's `outputs_notebooks/`).

---

## Dependencies

- PyTorch, NumPy, Matplotlib, papermill
- `allshowers.generate_showers` from `/n/home05/zdimitrov/tambo/TAMBO-opt/` — provides `sample_primary_particles`, `run_point_count_fm`, `run_allshowers`, `save_output`. The path is currently hard-coded in `modules/generate_showers.py` (line 5)
- `showerdata` package — used in `shower_computation.ComputeShowerDetection` to load cached `.pt` shower files

---

## Relation to Other Pipelines

- `../detector_optimization/` — v1, exploratory.
- `../detector_optimization_v2/` — diffusion-image surrogate, bilinear detector response.
- `../detector_optimization_v4/` … `_v6/` — further iterations; check each for its current status.
