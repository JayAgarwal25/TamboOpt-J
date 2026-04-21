# detector_optimization_v4

Fourth iteration of the TAMBO detector-layout optimization pipeline. The big shift from v3 is **geometry**: detectors move from a flat 2D plane to the full 3D Colca Valley **mountain surface**, staying on the wall via a differentiable surface map. The shower surrogate (AllShowers point clouds) and most of the pipeline plumbing are reused from v3 via `sys.path` injection.

Three structural additions carry the mountain:

- `SurfaceEastMap` — differentiable `East = f(North, Up)` by bilinear sampling of an interpolated 256×256 grid.
- `z_cont = (EAST_ENTRY − East) / LAYER_EAST_DX` — a **continuous AllShowers layer index** ∈ ℝ, replacing v3's hard `filter_plane=20`.
- `GetCounts_planeaware` — v3's spatial Gaussian kernel × a **triangular plane weight** `relu(1 − |layer_p − z_cont|)`, so points at the layer nearest `z_cont` dominate the sum.

End-to-end flow:

```
sample primary particles  ──▶  AllShowers point cloud  ──▶  spatial Gaussian × triangular plane weight
     (E, zenith, azimuth)           (B, max_points, 5)             (B, max_points, n_det)
                                                                              │
   ┌──────────────────────────┐                                               ▼
   │ LearnableXY (N, Up)      │──▶ SurfaceEastMap ──▶ z_cont ──▶ per-detector kernel → (N_int, T_int)
   │  differentiable positions│                                                │
   └──────────────────────────┘                                                ▼
                                                    Reconstruction MLP (5 feats, active) → [Ê, θ̂, φ̂]
                                                                              │
                                                                              ▼
                                               Utility U = 1e2·U_θ + 1e2·U_φ + 1e8·U_E + 5e5·U_PR
                                                    backprop → (N, Up) via the full chain
```

Gradient path confirmed alive end-to-end (see `gradient_path_analysis.md`), but open objective-shape issues keep detectors nearly static — see **Known Issues** below.

For the broader v1→v6 history, see `../VERSIONS.md`.

---

## Modules (`modules_v4/`)

v4 only ships three new modules. Everything else is imported from v3 via `sys.path` injection in `modules_v4/__init__.py`.

| File | Public API | Purpose |
|------|-----------|---------|
| `tr_geometry.py` | `load_tr_mountain`, `MountainData`, `sample_initial_layout`, `project_to_mountain` | Loads `basic_geometry.h5` (group `colca_valley_30000`), rotates ECEF → ENU, returns `MountainData` with 2161 triangle centroids in local `(N, Up, East)` plus bbox / plane constants. Handles the Julia 1-indexed `faces` / `detector1` arrays |
| `tr_surface_map.py` | `SurfaceEastMap` | `nn.Module` wrapping `LinearNDInterpolator(2161 centroids) → 256×256 regular grid → F.grid_sample(bilinear, padding_mode="border")`. Differentiable `East = f(N, Up)`, border-clamped so wanderers don't NaN |
| `tr_plane_kernel.py` | `GetCounts_planeaware` | Extends v3's spatial Gaussian kernel with a **triangular plane weight** `relu(1 − \|layer_p − z_cont\|)`. Reduces exactly to v3 (`filter_plane=20`) when `z_cont ≡ 20`. Differentiable in `z_cont` |

**Inherited from v3** (imported via `modules_v4.__init__` injecting `../detector_optimization_v3` on `sys.path`):

| v3 Module | Usage in v4 |
|-----------|-------------|
| `modules.generate_showers.GenerateShowers` | Unchanged |
| `modules.shower_computation.ComputeShowerDetection` | Called with `filter_plane=None` (kernel handles plane selection now) |
| `modules.detector_response.{SmearN, TimeAverage_vectorized}` | Interface-compatible callables; `GetCounts_planeaware` does not invoke them (same pattern as v3's `GetCounts_differentiable`) |
| `modules.reconstruction.{Reconstruction, NormalizeLabels, DenormalizeLabels, EarlyStopping}` | `Reconstruction(input_features=NUM_FEATURES, num_detectors=Nunits)` — active scripts use `NUM_FEATURES=5`, `Nunits=90` |
| `modules.layout_optimization.LearnableXY` | Unchanged — now carries `(N, Up)` instead of `(x, y)` |
| `modules.utility_functions.{reconstructability, U_PR, U_E, U_angle}` | Unchanged |
| `modules.geometry.Layouts` | **Not used** — ring layout doesn't apply to a curved mountain |

---

## Contents

```
detector_optimization_v4/
├── CLAUDE.md                                           # Session memory: design, gotchas, coord convention
├── modules_v4/                                         # v4-only modules
│   ├── __init__.py                                     #   Injects v3 into sys.path
│   ├── tr_geometry.py                                  #   HDF5 loader, ECEF→ENU, MountainData
│   ├── tr_surface_map.py                               #   Differentiable East = f(N, Up)
│   └── tr_plane_kernel.py                              #   Spatial Gaussian × triangular plane weight
│
├── SWGOLO7_optimization_tr.ipynb                       # Main optimization notebook
├── SWGOLO7_optimization_tr_same_10.ipynb               # Fixed 10-shower fixture (deterministic gradients)
├── SWGOLO7_optimization_tr_20k_center_init_...ipynb    # 20k showers, center init, angle+energy, Adam lr=1
├── SWGOLO7_optimization_tr_same_10_center_init_*.py    # Python exports of the notebook variants
├── SWGOLO7_optimization_tr_same_300_center_init.py     # 300-shower Python export
│
├── auto_run_notebook.py                                # Papermill runner (timestamped outputs)
├── common_gpu_auto_run_notebook_batch.sh               # SLURM batch script
│
├── stateful-beaming-pie.md                             # 1090-line v3→v4 migration plan
├── gradient_path_analysis.md                           # Authoritative v4-stall diagnosis
│
├── tests/
│   └── test_v4_modules.ipynb                           # 6-cell sanity suite (no GPU)
│
└── outputs_notebooks/                                  # Timestamped papermill outputs
```

---

## Coordinate Convention

| Symbol | Meaning | Learnable? |
|--------|---------|-----------|
| `x` | ENU North [m] | **yes** |
| `y` | ENU Up / elevation [m] | **yes** |
| `z_cont` | `(EAST_ENTRY − East(x, y)) / LAYER_EAST_DX`, continuous AllShowers layer index | **no** (derived) |

AllShowers layer-East calibration: manual selection of `EAST_ENTRY = 1500 m`, `LAYER_EAST_DX = 150 m` via `load_tr_mountain(east_entry=…, layer_east_dx=…)`. Per user (2026-04-14) these are the correct values — runs using the `−212 / 307` defaults sampled all energy from the last plane and the mountain geometry was mismatched.

Only detectors with `East < EAST_ENTRY` have `z_cont > 0` and can see shower particles.

---

## v3 → v4 Differences

| Aspect | v3 | v4 |
|--------|----|----|
| Detector positions | 2D `(x, y)` on a flat plane | 2D `(x = N, y = Up)` on the mountain surface |
| Initial layout | Concentric rings (`Layouts()`) | Grid / random / center on the `(N, Up)` bbox of mountain centroids |
| East coordinate | Fixed (plane 20 hardcoded) | Derived: `East = SurfaceEastMap(N, Up)` — differentiable |
| Plane index | Hard: `filter_plane=20` zeros non-20 energy | Continuous: `z_cont ∈ ℝ` |
| Shower layers used | Only plane 20 | Plane weight picks the layers near each detector's `z_cont`. How many layers are reachable depends on the EAST calibration (see Coordinate Convention): `1500/150` reaches all 24 layers; `−212/307` reaches only layers 0–6. |
| Kernel | Spatial Gaussian only | Spatial Gaussian × triangular plane weight |
| NN features | 6: `[x, y, N_int, T_int, x0, y0]` | 5 (active): `[x, y, z_cont, N_int, T_int]`; 7-feature variant `[…, x0, y0]` is commented out in the active script |
| `reconstructability` N-index | `inputs_batch[:, :, 2]` | `inputs_batch[:, :, 3]` (N is at index 3 under both 5- and 7-feature layouts) |
| Layout save format | 2-col `(x, y)` | 3-col `(N, Up, z_cont)` |
| Visualization | 2D scatter | 2D top-down + 3D mountain scatter + 3D GIF animation |
| `push_apart` / `symmetry_loss` | Used | Dropped (not applicable on curved surface) |
| NN re-training | Not needed (same geometry) | **Required** (new `z_cont` feature, new count distribution) |

---

## NN Feature Vector

Active scripts use **5 features** per detector; the 7-feature variant (adding `x0, y0`) is present but commented out.

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `x = N` | Detector North coordinate [m] |
| 1 | `y = Up` | Detector Up (elevation) coordinate [m] |
| 2 | `z_cont` | Continuous plane index (derived from `SurfaceEastMap(N, Up)`) |
| 3 | `N_int` | Energy-weighted kernel integral from `GetCounts_planeaware` (no `SmearN` applied — kernel accepts `SmearN_fn` but doesn't invoke it) |
| 4 | `T_int` | Plane-weighted arrival time from `GetCounts_planeaware` (`(point_t · kernel).mean(dim=1)` in the current implementation) |
| 5 | `x0` *(commented out)* | Energy-weighted shower core North / 5000 |
| 6 | `y0` *(commented out)* | Energy-weighted shower core Up / 5000 |

`Reconstruction(input_features=NUM_FEATURES, num_detectors=Nunits)` — `NUM_FEATURES = 5` in the active scripts. **`reconstructability` reads `N_int` at feature index 3** (`inputs_batch[:, :, 3]`), whereas v3 had N at index 2.

---

## Geometry Source

File: `TAMBOSim/resources/basic_geometry.h5`, group `colca_valley_30000`.

| Dataset | Shape | Notes |
|---------|-------|-------|
| `vertices` | `(3, 90000)` float64 | ECEF metres |
| `faces` | `(3, 179996)` int64 | **Julia 1-indexed** — subtract 1 |
| `detector1` | `(2161,)` int64 | **Julia 1-indexed** — subtract 1 |
| `location` | `(2,)` | `[lon_deg, lat_deg]` of site |

Site: lon = −72.279397°, lat = −15.622267°. Detector region in local ENU: East [−2019, +1182] m, North [−2497, +2474] m, Up [2442, 3886] m.

---

## Optimization Loop (per-epoch pseudocode)

```python
x_det, y_det  = xy_module()                                                 # LearnableXY: (N, Up)
east_det      = surface(x_det, y_det)                                       # differentiable East
z_cont        = (mountain.east_entry - east_det) / mountain.layer_east_dx   # continuous layer

N_int, T_int, X0, Y0, energy, ... = generate_showers(
    x_det, y_det, z_cont, number_of_showers=Nbatch, use_cache=True
)

# Active layout: 5 features; the 7-feature variant with (x0, y0) is commented out.
inputs_batch = torch.stack(
    [x_exp, y_exp, z_cont_exp, N_int, T_int], dim=2
).float()

r_score = reconstructability(inputs_batch[:, :, 3], reconstruct_threshold=10)   # N at idx 3
U = (1e2 * U_angle(preds_th, th, r_score)
     + 1e2 * U_angle(preds_phi, ph, r_score)
     + 1e8 * U_E(preds_e, energy, r_score)
     + 5e5 * U_PR(r_score))
Loss = -U
Loss.backward()          # gradients: z_cont → East → (N, Up)
optimizer.step()         # active script: SGD(lr=0.5, momentum=0.3)
                         # Apr 13–14 sweep variants: Adam(lr ∈ {0.3, 1, 3})
```

Layout is saved every epoch as `Python_Layout/Layout_{epoch}.txt` (3 cols: `North, Up, z_cont`). Final animation: `layout_evolution_3d.gif`.

---

## Running on the Cluster

```bash
# Overnight 200k run:
sbatch common_gpu_auto_run_notebook_batch.sh
```

`auto_run_notebook.py` uses papermill, writes timestamped copies into `outputs_notebooks/`.

---

## Key Gotchas

1. **Julia 1-indexing** on `faces` / `detector1` — subtract 1 before Python indexing.
2. **ECEF → ENU rotation** required before using HDF5 vertices (rotation is implemented in `_ecef_to_enu` using a mean-Earth-radius sphere of 6 371 000 m).
3. **`z_cont` gradient path**: `relu(1 − |layer − z_cont|)` is the only thing coupling `z_cont` to the loss. Detaching `z_cont` kills the surface-map gradient.
4. **`reconstructability` index**: `N_int` is at feature index 3 in v4 (index 2 in v3).
5. **Never pass `filter_plane=20`** to `ComputeShowerDetection` in v4 — it would zero all non-plane-20 energies before the kernel runs.
6. **Layer accessibility is calibration-dependent.** With the active-script calibration (`EAST_ENTRY = 1500, LAYER_EAST_DX = 150`), `z_cont` spans ~[2.1, 23.5] across the mountain — **all 24 layers are reachable**. With the module-default calibration (`−212 / 307`), `z_cont` spans ~[−4.5, +5.9] and only layers 0–6 are accessible (layers 7–23 fall off the surface).
7. **NN retraining required** after switching from v3 — new `z_cont` feature and redefined `N_int` / `T_int` (plane-weighted via the triangular kernel) mean the v3 checkpoint is not reusable.
8. **Module directory is `modules_v4/`, not `modules/`** — intentional so `from modules.X import Y` still resolves to v3.
9. **Padding rows in point clouds** carry `energy = 0` — they contribute nothing to kernel sums, no special handling needed.
10. **`GetCounts_planeaware` returns raw `(local_intensity, et)`** — `SmearN_fn` and `TimeAverage_vectorized_fn` are accepted as kwargs for v3-interface compatibility but are **not invoked** inside the kernel. The returned `local_intensity` is a Gaussian-plus-triangular-weighted energy sum (not a "smeared particle count"); `et` is `(point_t · kernel).mean(dim=1)` (an unweighted per-kernel mean — note this is *not* energy-weighted; the energy-weighted form is commented out).

---

## Known Issues (from `gradient_path_analysis.md`)

The autograd chain `xy_module.x/.y → SurfaceEastMap → GetCounts_planeaware → NN → utility → loss` is verified intact — `grad_norm` is finite and non-zero every epoch. But runs stall because of *objective shape*, not wiring:

1. **`U_PR` saturates** — `sigmoid(5·(n − 10))` at `n ≈ 90` ≈ 1 for every event, so `U_PR ≈ √90 ≈ 3.16` with weight `5e5` becomes a ~82%-of-total frozen constant. Zero gradient contribution.
2. **`U_E` is numerically zero** — `DenormalizeLabels` returns GeV (1e5–1e8), so `r / ((Ê − E)² + 0.01) ≈ 1e-10` per term, summed to zero in the logs.
3. **Only `U_θ` and `U_φ` drive learning.** `U_θ` gains ~29% in the first 100 epochs then plateaus; `U_φ` oscillates (1k → 30k → 1k) — SGD chatter in a narrow basin.
4. **NN fine-tune is a silent no-op** — `DataLoader(ft_dataset, batch_size=32, drop_last=True)` on `Nfinetune = 10` yields zero batches per epoch.
5. **Tangential-only motion** — over 2000 epochs, mean per-detector displacement is 5.3 m (max 13.2 m); `z_cont` never leaves `[11, 12.4]`.

The Apr 13–14 experiment sweep (see `outputs_notebooks/` filenames) tries `angle_error`, `theta_error`, `phi_error`, `angle_energy`, `mean_u` utility variants at Adam lr ∈ {0.3, 1, 3} on 10 / 300 / 20k / 200k shower batches looking for an objective that actually drives the positions. v6 abandons this approach in favour of two frozen NN surrogates; see `../VERSIONS.md`.

---

## Relation to Other Pipelines

- `../detector_optimization/` — v1, monolithic.
- `../detector_optimization_v2/` — modular refactor, flat 2D, diffusion-image surrogate.
- `../detector_optimization_v3/` — AllShowers point-cloud surrogate, single plane. Imported wholesale by v4.
- `../detector_optimization_v5/` — evolutionary pruning + DeepSets, scaffolding only (Apr 14).
- `../detector_optimization_v6/` — staged pipeline with two frozen NN surrogates (data-gen → FNN → recon → optimize).
- `../VERSIONS.md` — cross-version history and known issues.
