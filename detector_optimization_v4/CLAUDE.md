# detector_optimization_v4 — CLAUDE.md

Project memory for the TAMBO TR (triangulated mountain) detector layout optimizer.
Read this at the start of every session before touching any file.

---

## Goal

Optimize the positions of 90 water-Cherenkov detectors on the 3D Colca Valley mountain wall
so that a reconstruction NN can best infer the primary shower energy, zenith, and azimuth.

**v4 extends v3 by moving detectors from a single fixed plane to the full mountain surface:**
- Learnable parameters: `(x = North, y = Up)` — a 2D point on the mountain wall.
- Derived (not learned): `East = f(N, Up)` via a differentiable surface map; then
  `z_cont = (EAST_ENTRY − East) / LAYER_EAST_DX`, a continuous AllShowers layer index ∈ [0, 23].
  `EAST_ENTRY = −212.0 m`, `LAYER_EAST_DX = 307.0 m` (empirically calibrated from fixture data).
- Gradients flow: `Loss → counts → z_cont → East → (N, Up) → optimizer`.

---

## v3 Baseline (what v4 builds on)

**Source:** `TambOpt/detector_optimization_v3/`  
**Entry notebook:** `SWGOLO7_optimization.ipynb`

### What v3 does

v3 optimizes 90 point detectors placed on a **single fixed plane (plane 20 of 24)** of the
Colca Valley shower model. The learnable parameters are `(x, y)` in a 2D horizontal plane.

- **Initial layout:** concentric rings (`Layouts(n_detectors=90, n_rings=5, radius=300)`).
- **Shower model:** AllShowers point-cloud flow-matching (see "v3 Shower Generation Model"
  section below). Returns `samples (B, max_points, 5)` with columns
  `[x, y, layer_index, energy, time]`.
- **Plane selection:** `ComputeShowerDetection(..., filter_plane=20)` zeroes out the energy
  of all points with `layer_index ≠ 20` before passing `samples` to the kernel.
- **Kernel:** `GetCounts_differentiable(samples, x_det, y_det)` — spatial Gaussian
  `exp(−d²/(2σ²))` with `σ=200 m` over the `(x, y)` columns; ignores `layer_index`.
- **NN input (6 features):** `[x, y, N_int, T_int, x0, y0]` per detector.
  `Reconstruction(input_features=6, num_detectors=90)`.
- **Optimizer:** `SGD(lr=10, momentum=0.3)`, 1200 epochs.
- **Fine-tune:** NN is fine-tuned every 5 optimization epochs on freshly generated data
  at the current detector positions.
- **Utility loss:** `U = (1e2·U_angle_θ + 1e2·U_angle_φ + 1e3·U_E + 5e5·U_PR) / 1e3`.
  `Loss = −U`; higher U = better reconstruction.
- **Layout files:** 2-column `(x, y)` saved every epoch as
  `outputs/.../Python_Layout/Layout_{epoch}.txt`.

### v3 Key Modules (all imported by v4 via sys.path)

| Module | Key exports | v4 usage |
|--------|------------|---------|
| `modules/generate_showers.py` | `GenerateShowers` | imported unchanged |
| `modules/shower_computation.py` | `ComputeShowerDetection` | imported; `filter_plane=None` |
| `modules/detector_response.py` | `SmearN`, `TimeAverage_vectorized` | passed as callables to `GetCounts_planeaware` |
| `modules/reconstruction.py` | `Reconstruction`, `NormalizeLabels`, `DenormalizeLabels`, `EarlyStopping` | imported unchanged; `input_features=7` |
| `modules/layout_optimization.py` | `LearnableXY` | imported unchanged |
| `modules/utility_functions.py` | `reconstructability`, `U_PR`, `U_E`, `U_angle` | imported unchanged |
| `modules/geometry.py` | `Layouts` | **not used** in v4 |

---

## Changes from v3 to v4

| Aspect | v3 | v4 |
|--------|----|----|
| Detector positions | 2D `(x, y)` on a flat plane | 2D `(x=N, y=Up)` on the mountain surface |
| Initial layout | Ring layout from `Layouts()` | Grid on the `(N, Up)` bbox of mountain centroids |
| East coordinate | Fixed (plane 20 is hardcoded) | Derived: `East = surface(N, Up)` — differentiable |
| Plane index | Fixed: `filter_plane=20` zeros non-20 energy | Continuous: `z_cont = (EAST_ENTRY − East) / LAYER_EAST_DX` |
| Shower layers used | Only plane 20 | All 24 layers (plane weight selects the right ones) |
| Kernel | Spatial Gaussian only | Spatial Gaussian × triangular plane weight `relu(1−\|layer−z_cont\|)` |
| NN features | 6: `[x, y, N, T, x0, y0]` | 7: `[x, y, z_cont, N, T, x0, y0]` |
| `reconstructability` index | `inputs_batch[:,:,2]` (N at idx 2) | `inputs_batch[:,:,3]` (N at idx 3) |
| Layout save format | 2-col `(x, y)` | 3-col `(North, Up, z_cont)` |
| Visualization | 2D scatter | 2D top-down + 3D mountain scatter + 3D GIF animation |
| `push_apart` / `symmetry_loss` | Used | Dropped (not applicable to mountain surface) |
| NN re-training | Not needed (same geometry) | **Required** (new feature `z_cont`, new count distribution) |

---

## v4 Coordinate Convention

| Symbol | Meaning | Learnable? |
|--------|---------|-----------|
| `x` | ENU North [m] | **yes** |
| `y` | ENU Up / elevation [m] | **yes** |
| `z_cont` | `(EAST_ENTRY − East(x,y)) / LAYER_EAST_DX`, continuous AllShowers layer index | **no** (derived) |

AllShowers layer-East calibration (empirically derived from fixture data):
- `EAST_ENTRY = −212.0 m` — East at AllShowers layer 0 (shower entry padding).
- `LAYER_EAST_DX = 307.0 m` — East depth per layer (East decreases per layer going deeper).
- Formula: `East_k ≈ EAST_ENTRY − k × LAYER_EAST_DX = −212 − 307k` metres.
- Layer 6 ≈ −2054 m East (deepest accessible on mountain surface; mountain East_lo ≈ −2019 m).
- Layer 0 ≈ −212 m, layer 1 ≈ −519 m, layer 23 ≈ −7273 m (outside mountain).
- **Only detectors with East < EAST_ENTRY (= −212 m) have z_cont > 0 and can see shower particles.**
- Mountain surface East spans ≈ [−2019, +1182] m → max z_cont ≈ 5.9 (AllShowers layers 0–6 only).
- Convention matches the AllShowers `layer_index` column (0–23).

---

## Folder Layout

```
TambOpt/detector_optimization_v4/
├── CLAUDE.md                         ← this file
├── SWGOLO7_optimization_tr.ipynb     ← main optimization notebook
├── modules_v4/
│   ├── __init__.py                   ← adds ../detector_optimization_v3 to sys.path
│   ├── tr_geometry.py                ← HDF5 loader, ECEF→ENU, MountainData dataclass
│   ├── tr_surface_map.py             ← SurfaceEastMap: differentiable East=f(N,Up)
│   └── tr_plane_kernel.py            ← GetCounts_planeaware: Gaussian × triangular plane weight
├── tests/
│   ├── test_v4_modules.ipynb         ← tests 1–6 (no GPU needed)
│   └── fixtures/
│       └── sample_showers_10.pt      ← symlink to v3's cached 10-shower fixture
└── outputs/                          ← populated at runtime
    └── NN_Files_TR_v4/
        ├── inputs.pt / labels.pt / ...
        ├── model_weights.pth
        ├── checkpoint.pth
        └── Python_Layout/
            ├── Layout_0.txt          ← 3 columns: North, Up, z_cont
            ├── Layout_1.txt
            ├── ...
            ├── Utilities.txt
            └── layout_evolution_3d.gif
```

**No verbatim copies.** Everything reusable is imported from v3 via sys.path injection in
`modules_v4/__init__.py`. v4 only ships the three new modules above.

---

## v3 Imports (via sys.path injection)

```python
import modules_v4   # triggers sys.path injection for v3

from modules.generate_showers   import GenerateShowers
from modules.shower_computation import ComputeShowerDetection
from modules.detector_response  import SmearN, TimeAverage_vectorized
from modules.reconstruction     import Reconstruction, NormalizeLabels, DenormalizeLabels, EarlyStopping
from modules.layout_optimization import LearnableXY
from modules.utility_functions  import reconstructability, U_PR, U_E, U_angle

# v4 only
from modules_v4.tr_geometry     import load_tr_mountain
from modules_v4.tr_surface_map  import SurfaceEastMap
from modules_v4.tr_plane_kernel import GetCounts_planeaware
```

---

## Geometry Source

File: `TAMBOSim/resources/basic_geometry.h5`, group `colca_valley_30000`

| Dataset | Shape | Notes |
|---------|-------|-------|
| `vertices` | `(3, 90000)` float64 | ECEF metres |
| `faces` | `(3, 179996)` int64 | **Julia 1-indexed** triangle vertex indices |
| `detector1` | `(2161,)` int64 | **Julia 1-indexed** face indices |
| `location` | `(2,)` | `[lon_deg, lat_deg]` of site |

Site: lon = −72.279397°, lat = −15.622267°.
Detector region in local ENU: East [−2019, +1182] m, North [−2497, +2474] m, Up [2442, 3886] m.

**Critical gotcha:** subtract 1 from `faces` and `detector1` before using as Python indices.

---

## v4 New Modules

### `tr_geometry.py` — `load_tr_mountain()`

Loads the HDF5, computes detector-region triangle centroids in ECEF, rotates to local ENU
(using mean Earth radius sphere), returns a `MountainData` dataclass with:
- `centroids_NUE : (2161, 3)` numpy array, columns `[North, Up, East]` in metres
- `n_min/n_max, u_min/u_max` — North and Up bounding boxes of centroids
- `east_lo/east_hi` — actual East span of centroids (≈ [−2019, +1182])
- `east_min/east_max` — plane-axis bounds (−2000, +1000)
- `n_planes = 24`, `plane_dx = 125.0` m
- `sample_initial_layout(n_units, scheme)` — returns (N_init, U_init) arrays inside bbox

### `tr_surface_map.py` — `SurfaceEastMap`

Differentiable mountain function `East = f(North, Up)`:
1. Fits `scipy.interpolate.LinearNDInterpolator` on the 2161 centroid scatter.
2. Evaluates on a 256×256 regular `(North, Up)` grid; fills NaNs with nearest-neighbour.
3. Wraps in an `nn.Module` using `F.grid_sample(mode="bilinear", padding_mode="border",
   align_corners=True)` — differentiable w.r.t. `(x, y)`.

`padding_mode='border'` clamps detectors that wander outside the mountain bbox to the nearest
valid East value — no NaN explosions, gradients still flow.

### `tr_plane_kernel.py` — `GetCounts_planeaware()`

Extends v3's spatial Gaussian kernel with a **triangular plane weight**:
```
plane_w[b, p, i] = relu(1 − |layer_p[b, p] − z_cont[i]|)
kernel = spatial_gaussian × plane_w       # (B, P, n_det)
```
- When `z_cont ≡ 20` for all detectors: `plane_w = 1` on layer-20 points, 0 elsewhere →
  exactly v3's `filter_plane=20` behaviour (v3 is a strict subset of v4).
- `plane_w` is differentiable in `z_cont` (piecewise linear, non-differentiable only at
  `z_cont = layer_p ± 1` which has measure zero).
- One extra elementwise multiply per (detector, point) pair — same memory class as v3.

---

## v3 Shower Generation Model (AllShowers)

v3 uses the **AllShowers point-cloud flow-matching model** located at
`/n/home05/zdimitrov/tambo/TAMBO-opt/allshowers/`.

### What `GenerateShowers.__call__()` does

1. **Sample primary particles** — calls `sample_primary_particles(n, e_min, e_max,
   zenith_min, zenith_max, azimuth_min, azimuth_max)`.
   Returns a dict with:
   - `energies (N, 1)` — primary energy [GeV]
   - `directions (N, 3)` — unit vector `(sin θ cos φ, sin θ sin φ, cos θ)`
   - `labels (N,)` — particle type (PDG codes)

2. **Predict number of points** — calls `run_point_count_fm(model_path, energies,
   directions, labels)`.
   A small flow-matching model that predicts how many points each shower point cloud should
   have. This determines the ragged `max_points` dimension after padding.

3. **Generate point clouds** — calls `run_allshowers(run_dir, energies, directions, labels,
   num_points, num_timesteps=16, batch_size=30, solver="midpoint", device=...)`.
   - Uses a flow-matching (continuous normalizing flow) model trained on TAMBO shower
     simulations.
   - Each shower is represented as an unordered point cloud.
   - Returns `samples (N, max_points, 5)` float32 with columns:
     `[x, y, layer_index, energy, time]`.
     - `x, y` — transverse position [m] in the shower frame
     - `layer_index` — integer 0–23 (East-aligned detector plane)
     - `energy` — particle energy at detector level [a.u.]
     - `time` — arrival time [ns]
   - Padding rows (to reach `max_points`) have `energy = 0`, so they contribute nothing
     to the Gaussian kernel sum.

4. **Optionally save** to `{output_dir}/cashed_showers_{num_samples}.pt` using the
   `showerdata` format (readable with `showerdata.load(path)`).

### Key defaults
- `num_timesteps = 16` (flow-matching integration steps — fewer = faster, more = better quality)
- `solver = "midpoint"` (Runge-Kutta midpoint rule)
- `batch_size = 30` (showers processed at once by the model)
- Energy: `e_min=1e5, e_max=1e8` GeV
- Zenith: 60°–100° (sub-horizontal / upward-going neutrino geometry)
- Azimuth: 0°–360°

### Checkpoint paths (hardcoded in `GenerateShowers.__init__`)
- Point-count model: `.../allshowers/checkpoints/num_of_point_clouds_dequantize_compiled.pt`
- AllShowers run dir: `.../allshowers/checkpoints/all_showers`

### How v4 uses the shower output

v3 calls `ComputeShowerDetection(..., filter_plane=20)` which zeroes out all points with
`layer_index ≠ 20` before the spatial Gaussian kernel.

v4 passes `filter_plane=None` (no zeroing) and instead calls `GetCounts_planeaware()`
which applies a triangular plane weight inside the kernel. All 24 layers contribute —
the weight function selects the two layers bracketing `z_cont`.

---

## NN Feature Vector

7 features per detector: `[x=N, y=Up, z=z_cont, N_int, T_int, x0, y0]`

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `x = N` | Detector North coordinate [m] |
| 1 | `y = Up` | Detector Up (elevation) coordinate [m] |
| 2 | `z_cont` | Continuous plane index ∈ [0, 23] |
| 3 | `N_int` | Plane-interpolated smeared particle count |
| 4 | `T_int` | Plane-interpolated time average [ns] |
| 5 | `x0` | Energy-weighted shower core North / 5000 |
| 6 | `y0` | Energy-weighted shower core Up / 5000 |

`Reconstruction(input_features=7, num_detectors=90)` — v3's `Reconstruction.__init__`
already accepts `input_features` so no patch is needed.

**Note:** `reconstructability` is called on `inputs_batch[:, :, 3]` (N_int at index 3, not
index 2 as in v3 where N_int was at index 2).

---

## Optimization Loop (per-epoch pseudocode)

```python
x_det_opt, y_det_opt = xy_module()                              # learnable (N, Up)
east_det  = surface(x_det_opt, y_det_opt)                                     # differentiable East lookup
z_cont    = (mountain.east_entry - east_det) / mountain.layer_east_dx        # continuous AllShowers layer index

N_int, T_int, X0, Y0, energy, ... = generate_showers(
    x_det_opt, y_det_opt, z_cont,                               # z_cont captured in closure
    number_of_showers=Nbatch, use_cache=True
)

# 7-feature input tensor
inputs_batch = torch.stack(
    [x_exp, y_exp, z_cont_exp, N_int, T_int, x0_exp, y0_exp], dim=2
).float()

# r_score uses N_int (index 3, not 2)
r_score = reconstructability(inputs_batch[:, :, 3], reconstruct_threshold=10)
U = (1e2 * U_angle(...) + 1e2 * U_angle(...) + 1e3 * U_E(...) + 5e5 * U_PR(...)) / 1e3
Loss = -U
Loss.backward()      # gradients flow through z_cont → East → (N, Up)
optimizer.step()     # SGD lr=10, momentum=0.3
```

Layout saved as 3-column file `(North, Up, z_cont)` every epoch:
```python
np.savetxt(f"{output_dir}/Python_Layout/Layout_{epoch+1}.txt",
           np.column_stack((x.detach().cpu(), y.detach().cpu(), z_cont.detach().cpu())))
```

---

## Key Gotchas

1. **Julia 1-indexing**: `faces` and `detector1` in the HDF5 are 1-indexed — subtract 1.
2. **ECEF vertices**: must rotate to local ENU before use.
3. **z_cont gradient path**: `relu(1 − |layer − z_cont|)` is the ONLY thing coupling
   `z_cont` to the loss. If you detach `z_cont`, the surface-map gradient dies too.
4. **r_score index**: N_int is at feature index 3 in v4 (not 2 as in v3).
5. **`filter_plane=None`**: must not pass `filter_plane=20` to `ComputeShowerDetection`
   — that would zero out all non-plane-20 energies before our kernel can use them.
6. **Only layers 0–6 are accessible**: mountain surface East is ≈ [−2019, +1182] m; only East < −212 m gives z_cont > 0.
   Max z_cont ≈ 5.9. AllShowers layer 6 (East ≈ −2054 m) is the deepest layer the mountain can reach.
   Detectors on the mountain face the interior, so they only see shower particles from layers 0–6.
7. **NN retraining required**: the input distribution changes (z_cont is new, N_int/T_int
   are now plane-interpolated). Must train from scratch in v4.
8. **`modules_v4/` name**: intentionally NOT `modules/` so that
   `from modules.X import Y` still resolves to v3.
9. **Padding rows**: shower rows with `energy=0` contribute nothing to the kernel sums —
   no special handling needed.
10. **`GetCounts_planeaware` must return raw `(local_intensity, et)`** — do NOT call
    `SmearN_fn` or `TimeAverage_vectorized_fn` inside the kernel. Those callables are
    accepted for interface compatibility only (matching v3's signature). Calling them
    breaks the gradient graph (`TimeAverage_vectorized` uses `torch.randn_like` which
    detaches; also its signature requires 3 args `(T, Nb, Ns)` so it crashes). v3's
    own `GetCounts_differentiable` has the same pattern — it never calls those callables.

### Why the Test 6 gradient check needs two sub-tests (not one end-to-end)

Placing detectors at mountain centroid positions and running the AllShowers fixture gives
`Nk = 0` and zero gradients — not a bug, but a geometric mismatch:

- AllShowers shower particles live **inside** the mountain. Layer-6 particles are at
  Up ≈ 2136 m, but the mountain surface floor is Up ≥ 2442 m. The `(North, Up)` ranges
  barely overlap → `spatial = exp(−d²/2σ²) ≈ 0`.
- The nearest mountain centroid to the layer-6 shower core has East ≈ +930 m, giving
  `z_cont = (−212 − 930) / 307 ≈ −3.7` → `plane_w = 0`.
- `kernel = spatial × plane_w ≈ 0`, so `Nk = 0` and all gradients are zero.

The fix is two independent sub-tests linked by chain rule:
- **Sub-test A** — place detectors AT shower-particle positions (spatial = 1), set
  `z_cont = layer − 0.3` (plane_w = 0.7). Verifies `dNk/d(x_det) ≠ 0` and `dNk/d(z_cont) ≠ 0`.
- **Sub-test B** — verify `d(East)/d(N,Up) ≠ 0` through `SurfaceEastMap + LearnableXY`.
- **Chain rule**: `dNk/d(N,Up) = dNk/d(z_cont) × d(z_cont)/d(East) × d(East)/d(N,Up) ≠ 0`.

---

## What Was Dropped from v3

- `Layouts()` / `n_rings` / `radius` — ring layout not applicable to a curved mountain.
- `push_apart` / `symmetry_loss` — not applicable; re-enable if detectors collapse.
- `filter_plane=20` argument — replaced by triangular plane weight in kernel.
- `geometry.py` / `barycentric_coords` / `project_to_triangle` — v3's hardcoded 2D triangle.

---

## Quick References

- v3 source: `TambOpt/detector_optimization_v3/`
- AllShowers framework: `/n/home05/zdimitrov/tambo/TAMBO-opt/allshowers/`
- TR geometry notebooks (Julia): `TAMBOSim/notebooks/create_geometry/`
- Geometry HDF5: `TAMBOSim/resources/basic_geometry.h5`
- v4 tests (no GPU): `tests/test_v4_modules.ipynb` (6 cells, run top-to-bottom)
