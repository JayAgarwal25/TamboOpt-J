# detector_optimization_v2

Refactored detector-layout optimization pipeline for the TAMBO Observatory. Replaces the monolithic `SWGOLO7*.ipynb` workflow in `../detector_optimization/` with modular, reusable Python components that are used directly from the optimization notebooks.

End-to-end flow:

```
sample primary particles  ──▶  24-plane diffusion + FNN bbox  ──▶  Bilinear grid_sample → counts (N, T)
 (E, class, zenith, azimuth)     images: (N, 24, 3, 32, 32)            (differentiable in x_det, y_det)
                                 bboxes: (N, 24, 4)
                                         │
                                         ▼
                                 take plane 20, denormalize [-1,1] → [0,1],
                                 map image coords → world coords via bbox
                                                                           │
                                                                           ▼
                                                         Reconstruction MLP → [X0, Y0, E, θ, φ]
                                                                           │
                                                                           ▼
                                                Utility U = α·U_PR + β·U_E + γ·U_angle
                                                          backprop → LearnableXY → push_apart
```

---

## Modules

| File | Public API | Purpose |
|------|-----------|---------|
| `geometry.py` | `Layouts`, `project_to_triangle`, `barycentric_coords` | Concentric-ring detector layout; triangular site-boundary projection via barycentric clipping, not used in this version, kept as legacy |
| `shower_generation.py` | `GenerateShowers`, `ReadShowers`, `denormalize_shower` | Samples random `(E, class_id, zenith, azimuth)`, calls `PlaneDiffusionEvaluator` + `PlaneFNNGenerator`, returns differentiable `(N, T, X0, Y0, E, sin_z, cos_z, sin_a, cos_a)` with respect to detector positions. No differentiation through the model here. |
| `detector_response.py` | `GetCounts_differentiable`, `SmearN`, `TimeAverage_vectorized` | Bilinear-interpolated counts via `F.grid_sample` (gradients flow through detector `x, y`); Gaussian smearing; vectorized timing |
| `reconstruction.py` | `Reconstruction`, `NormalizeLabels`, `DenormalizeLabels`, `EarlyStopping` | 4-layer MLP: flattened `[x, y, N, T]`-per-detector → normalized `[X0, Y0, E, θ, φ]`; label normalization helpers; early-stopping tracker |
| `layout_optimization.py` | `LearnableXY`, `push_apart`, `symmetry_loss` | Detector positions as `nn.Parameter`; in-place minimum-separation repulsion; n-fold rotational symmetry penalty |
| `utility_functions.py` | `reconstructability`, `U_PR`, `U_E`, `U_angle` | Soft-threshold detection score; reconstructability-weighted energy and angular utility terms |

---

## Contents

```
detector_optimization_v2/
├── geometry.py                          # Layout + boundary projection
├── shower_generation.py                 # Diffusion + FNN shower generator
├── detector_response.py                 # Differentiable counts / smearing / timing
├── reconstruction.py                    # MLP reconstructor + normalization + early stopping
├── layout_optimization.py               # Learnable (x,y) + constraints
├── utility_functions.py                 # U_PR / U_E / U_angle
│
├── SWGOLO7_optimization.ipynb                 # Main optimization notebook
├── SWGOLO7_optimization_from_center.ipynb     # Variant: initialize positions from center
├── SWGOLO7_plots.ipynb                        # Results visualization
├── explore_average_shower_raw_data.ipynb              # EDA on raw shower data
├── explore_average_shower_data_difusion_model.ipynb   # EDA on diffusion-generated showers
│
├── diffusion_model/                     # See "Diffusion Model" section below
│   ├── tambo_3D_diffusion_generator.py  #   PlaneDiffusionEvaluator (24-plane DDIM sampler wrapper)
│   ├── tambo_3D_fnn_scaler.py           #   PlaneFNNGenerator (per-plane bbox FNN regressor)
│   └── diffusion_loc_zlt.py             #   Local diffusion components (trainer, samplers, autoregressive driver)
│
├── auto_run_notebook.py                 # Papermill executor (timestamped outputs)
├── common_gpu_auto_run_notebook_batch.sh              # SLURM batch (gpu / gpu_h200)
├── common_gpu_auto_run_notebook_batch_from_center.sh  # SLURM batch (from-center variant)
└── outputs_notebooks/                   # Timestamped papermill runs (filenames encode run notes)
```

---

## Diffusion Model

The shower surrogate. `shower_generation.GenerateShowers` instantiates a `PlaneDiffusionEvaluator` (RGB shower planes) and a `PlaneFNNGenerator` (per-plane bounding boxes), feeds them sampled primary-particle conditions, and stitches the outputs into the `(N, T, X0, Y0, ...)` tuple consumed by the optimizer.

Unlike the v1 generator (which depended on the parent project's `diffusion` module), v2 vendors all diffusion components into `diffusion_loc_zlt.py` and the evaluator imports `DDIMSamplerPlanes` from there directly — no `imports_path` hack required for the sampler itself.

### `diffusion_loc_zlt.py` — local diffusion components

Standardized-data (zero-mean, unit-variance, **unbounded — no clipping**) single-head plane diffusion model. Five public symbols:

| Symbol | Role |
|--------|------|
| `extract(v, t, x_shape)` | Gather + broadcast helper for indexing schedule buffers (`betas`, `alphas_bar`, …) by batched timestep `t` |
| `GaussianDiffusionTrainer` | Training-time wrapper. Samples `t ~ U[0, T)` and noise `ε`, forms `x_t`, predicts `ε̂ = model(x_t, t, …conditions, plane_idx, past_plane)`, returns MSE loss |
| `GaussianDiffusionSampler` | Vanilla DDPM ancestral sampler — full `T`-step reverse process. Mainly used for validation/baseline |
| `DDIMSamplerPlanes` | DDIM sampler — `ddim_steps` evenly spaced timesteps over `[0, T)`, supports `eta` (deterministic at 0) and classifier-free guidance weight `w`. This is what `PlaneDiffusionEvaluator` calls at inference |
| `AutoregressivePlaneGenerator` | Driver that loops `plane_idx = 0..23`, feeding each generated plane back as `past_plane` for the next. Wraps a `DDIMSamplerPlanes`. (`PlaneDiffusionEvaluator.generate_samples` implements the same loop inline.) |

All conditioning vectors carry the same six fields end-to-end: `[p_energy, class_id, sin_zenith, cos_zenith, sin_azimuth, cos_azimuth]`, plus per-step `plane_idx` and `past_plane`.

### `tambo_3D_diffusion_generator.py` — `PlaneDiffusionEvaluator`

Wraps a trained `PlaneDiffusionModule` checkpoint (loaded via PyTorch Lightning) plus a `DDIMSamplerPlanes` built from the constructor's schedule parameters (`beta_1`, `beta_T`, `T`, `eta`, `ddim_steps`, `guidance_w`).

Key methods:
- `load_model()` — `PlaneDiffusionModule.load_from_checkpoint(...)` → eval mode → build sampler
- `generate_samples(num_conditions=None, batch_size=100)` — autoregressive 24-plane loop; returns / caches a single dict `{"conditions": (N, 6) cpu, "images": (N, 24, 3, 32, 32) cpu}` on `evaluator.generated_sets`
- `test_conditions` can be assigned manually (an `(N, 6)` tensor) to skip `extract_test_samples()` and sample at arbitrary points in condition space — `GenerateShowers` uses exactly this path

Still depends on `lightning_training.PlaneDataset` / `PlaneDiffusionModule` from the parent project (added via `imports_path`); the diffusion sampler itself is now local.

### `tambo_3D_fnn_scaler.py` — `PlaneFNNGenerator`

Direct feedforward regressor — no diffusion sampling. Predicts per-plane bounding boxes from the same six-field condition vector, returning `{"conditions": (N, 6), "bboxes": (N, 24, 4)}`. `GenerateShowers` slices `bboxes[:, 20, :]` (plane 20 is the analysis plane) to map normalized shower positions into world coordinates for the differentiable detector response.

Loads bbox standardization stats from `data_dir`; depends on `lightning_training_fnn.PlaneFNNModule` from the parent project.

For the full constructor parameter tables and step-by-step usage, see `../detector_optimization/diffusion_model/README.md` — the class APIs are shared between the two pipelines.

---

## Optimization Objective

The layout optimizer maximizes a weighted utility built from three reconstructability-weighted terms:

```
U = α · U_PR  +  β · U_E  +  γ · U_angle
```

- **`U_PR`** — fraction of reconstructable events (soft sigmoid on per-detector counts)
- **`U_E`** — energy-reconstruction quality, weighted by `r`
- **`U_angle`** — angular-reconstruction quality, weighted by `r`

After each gradient step, the layout is projected back to feasibility via `push_apart` (minimum detector separation).

---

## Running a Notebook on the Cluster

```bash
# Main optimization run (A100 / H200, 15h wallclock):
sbatch common_gpu_auto_run_notebook_batch.sh

# From-center initialization variant:
sbatch common_gpu_auto_run_notebook_batch_from_center.sh
```

`auto_run_notebook.py` uses papermill and writes timestamped outputs to `outputs_notebooks/`. Both batch scripts assume a `multiproc_env` conda environment with PyTorch + papermill.

---

## Dependencies

- PyTorch, PyTorch Lightning (for `PlaneDiffusionModule` / `PlaneFNNModule` checkpoints)
- NumPy, Matplotlib
- papermill (for batch notebook execution)
- Project-local modules used by the diffusion generator: `lightning_training(_fnn)`, `diffusion.DDIMSamplerPlanes` (see `diffusion_model/` and the v1 README under `../detector_optimization/diffusion_model/README.md` for usage details — the generator classes are shared between the two pipelines)

---

## Relation to Other Pipelines

- `../detector_optimization/` — v1, exploratory. Kept for reproducibility of earlier runs.
- `../detector_optimization_v3/` … `_v6/` — further iterations; check each for its current status.
