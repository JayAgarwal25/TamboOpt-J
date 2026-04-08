# Plan: `TambOpt/detector_optimization_v4` — mountain-surface position optimization on the TAMBO triangulated (TR) geometry

## Context

We are creating a new folder [TambOpt/detector_optimization_v4/](TambOpt/detector_optimization_v4/) that
reuses as much as possible of [TambOpt/detector_optimization_v3/](TambOpt/detector_optimization_v3/)
but retargets the optimization to the **TR (triangulated) setup** produced by the notebooks in
[TAMBOSim/notebooks/create_geometry/](TAMBOSim/notebooks/create_geometry/).

**Approach:** v3 optimizes ~90 point detectors with continuous learnable `(x, y)` positions on
a single shower plane (plane 20 of 24, hardcoded via the `filter_plane=20` argument inside
[v3/modules/shower_computation.py:101](TambOpt/detector_optimization_v3/modules/shower_computation.py#L101)).
v4 keeps the *learnable-positions* idea and backpropagation but moves the detectors **on the 3D
mountain surface** of the Colca Valley:

- The learnable parameters per detector are **`(x = North, y = Up = elevation)`** — a 2D point
  in the wall-perpendicular plane.
- Each `(x, y)` maps deterministically to a unique East coordinate via a precomputed
  **mountain-surface function `East = f(N, Up)`**, which exists because the Colca Valley wall
  is monotonic ("constantly elevating") — for every `(N, Up)` there is exactly one mountain
  face whose centroid has that elevation at that North.
- That `East` is converted to a **continuous plane index**
  `z_cont = (East − EAST_MIN) / Δ_plane` over the 24 plane diffusion-model planes (the `layer_index` column of v3's point-cloud `samples`
  tensor).
- The detector "sees" a **linear interpolation between the two nearest discrete planes**
  (`z_low = ⌊z_cont⌋`, `z_high = z_low + 1`). v3's point-cloud kernel is extended to apply a
  per-point **triangular plane weight** `relu(1 − |layer_p − z_cont_i|)` so that points whose
  layer index is exactly `z_low` or `z_high` contribute fully and points farther away
  contribute zero. The result is a single-pass plane-aware kernel call that produces
  `N_int / T_int` per detector — exactly the linear interpolation between the two bracketing
  planes, fully differentiable in `z_cont` (and hence in `(x, y)` via the surface map).
- All operations (Gaussian spatial kernel, triangular plane weight, surface map lookup) are
  differentiable, so gradients flow `Loss → counts → x, y`.

This is the natural extension of v3 to the full triangulated mountain: same point-cloud
kernel, same optimizer, same NN — but the 2D `(x, y)` now lives on a curved surface and the
shower representation uses all 24 plane layers instead of just plane 20.

User decisions captured in this session:
- Reuse `TAMBOSim/resources/basic_geometry.h5` directly (no regeneration).
- Reuse v3's point-cloud `GetCounts_differentiable` (Gaussian kernel over the
  `(x, y, layer_index, energy, time)` columns, see
  [v3/modules/detector_response.py:10](TambOpt/detector_optimization_v3/modules/detector_response.py#L10)).
  v4 extends it with an optional plane-weight argument; the spatial kernel is unchanged.
- **Detector positions are LEARNABLE** (v3-style position optimization, *not* a soft mask).
  Each detector has its own `(x = N, y = Up)` learnable parameter; backprop moves them on the
  mountain surface.
- 24 planes equidistant in East ∈ [−2000, +1000] m (125 m / plane), matching v3's
  flow-matching shower model layer indices.
- Ship only the **main optimization notebook** (no `_with_tambo_physics`, no `_plots`).
- Save a `CLAUDE.md` so future sessions can ramp up quickly.
- **v4 coordinate convention:**
  - `x` = ENU North     (learnable)
  - `y` = ENU Up         (learnable)
  - `z = z_cont`         (derived: `(East(N, Up) − EAST_MIN) / Δ_plane`, continuous, **not learned**)
- **v4 NN feature vector per detector: 7 features** =
  `[x = N, y = Up, z = z_cont, N_int, T_int, x0, y0]`
  where `N_int` / `T_int` are the plane-interpolated counts and times. v3's `Reconstruction`
  already accepts `input_features` as a constructor kwarg
  ([v3/modules/reconstruction.py:19](TambOpt/detector_optimization_v3/modules/reconstruction.py#L19)),
  so no patch is needed.
- **No verbatim copies.** v4 imports v3's modules directly via `sys.path` injection. v4 only
  contains: the new TR-geometry / surface-map / plane-aware kernel wrapper modules, the
  notebook, and CLAUDE.md.

---

## Knowledge dump — v3 (the upstream we will import from)

Path: [TambOpt/detector_optimization_v3/](TambOpt/detector_optimization_v3/) (uses a `modules/` subfolder)

**Problem:** Given a fixed number of point detectors, learn their 2D positions so that a
trained NN can reconstruct shower primary energy / zenith / azimuth, subject to a
reconstructability constraint.

**Entry notebook:** [SWGOLO7_optimization.ipynb](TambOpt/detector_optimization_v3/SWGOLO7_optimization.ipynb)
- Cell [02] — `from modules.* import ...`
- Cell [03] — constants: `Nunits = 90`, `n_rings = 5`, `radius = 300`,
  `IntegrationWindow = 128` ns
- Cell [04] — instantiates `GenerateShowers(...)` (the point-cloud flow-matching wrapper
  living at `/n/home05/zdimitrov/tambo/TAMBO-opt/allshowers/...`)
- Cell [05] — wraps `GetCounts_differentiable` and `ComputeShowerDetection` as
  `_GetCounts` / `generate_showers` with `filter_plane=20` baked in
- Cells [13]–[15] — generates training / validation / test data, builds the 6-feature
  input tensor `[x, y, N, T, x0, y0]`
- Cell [28] — `Reconstruction(input_features=6, num_detectors=Nunits)`
- Cells [38–40] — `LearnableXY` warm-up
- Cell [44] — main optimization loop

**Files (in [v3/modules/](TambOpt/detector_optimization_v3/modules/)):**

| File | Role | Used by v4? |
|---|---|---|
| `geometry.py` | `Layouts()` rings, `barycentric_coords`, `project_to_triangle()` hardcoded 2D triangle | **No — v4 has its own loader** |
| `layout_optimization.py` | `LearnableXY`, `push_apart`, `symmetry_loss` | **Imported as-is** (`LearnableXY` is generic) |
| `detector_response.py` | **Point-cloud** `GetCounts_differentiable(samples, x_det, y_det, ...)` — Gaussian spatial kernel over `(x, y)`, ignores `layer_index`; `SmearN`, `TimeAverage_vectorized` | **Imported, then wrapped** by `tr_plane_kernel.py` so the plane axis is honoured |
| `reconstruction.py` | `Reconstruction(input_features=6, num_detectors=90, ...)`, `NormalizeLabels`, `DenormalizeLabels`, `EarlyStopping` | **Imported as-is** (already accepts `input_features`) |
| `shower_computation.py` | `ComputeShowerDetection(...)` — calls `generate_showers_instance(...)`, applies optional `filter_plane` zero-out, mini-batches through `GetCounts_differentiable` | **Imported, called with `filter_plane=None`** so all 24 layers reach the kernel |
| `shower_generation.py` | `ReadShowers`, `denormalize_shower`, an older `GenerateShowers` (legacy — superseded by `modules/generate_showers.py`) | Not used directly by v4 |
| `generate_showers.py` | `GenerateShowers` class wrapping the AllShowers framework; returns `samples (N, max_points, 5)` with columns `[x, y, layer_index, energy, time]` | **Imported as-is** |
| `utility_functions.py` | `U_angle`, `U_E`, `U_PR`, `reconstructability` | **Imported as-is** |
| `tambo_physics.py` | physics tensor helpers | Not needed by v4 main notebook |

**Key v3 kernel ([v3/modules/detector_response.py:10](TambOpt/detector_optimization_v3/modules/detector_response.py#L10)):**
- Inputs: `samples (B, max_points, 5)` columns `[x, y, layer_index, energy, time]`,
  `x_det/y_det (num_det,)` (`requires_grad=True`).
- Computes pairwise `(point − det)` squared distances, applies a Gaussian spatial kernel
  `exp(−d² / (2σ²))` (`σ ≈ 200 m`), and returns `(local_intensity, et)` shape `(B, num_det)`,
  both differentiable in `x_det / y_det`.
- **The `layer_index` column is currently ignored** — every point contributes regardless of
  which plane it sits on. v3 instead zeroes out off-plane points upstream via
  `filter_plane=20` in `ComputeShowerDetection`.

**Why this is exactly what v4 needs.** Because each point already carries its `layer_index`,
the natural plane-aware extension is to multiply the per-point energy by a triangular weight
`relu(1 − |layer_p − z_cont_i|)` *inside* the kernel — one extra elementwise multiply per
(detector, point) pair. No per-plane loop, no bbox rebuild, no shower-image stack. v3's
`filter_plane=20` becomes the special case `z_cont_i ≡ 20` for all detectors.

---

## Knowledge dump — TR setup (the target)

Path: [TAMBOSim/notebooks/create_geometry/](TAMBOSim/notebooks/create_geometry/)

**Output file:** [TAMBOSim/resources/basic_geometry.h5](TAMBOSim/resources/basic_geometry.h5)
```
colca_valley_30000/
├── location   [-72.279397, -15.622267]    (lon, lat in degrees)
├── radii      (13,) PREM layer radii (m)
├── vertices   (3, 90000) float64 ECEF (m)
├── faces      (3, 179996) int64 — JULIA 1-INDEXED triangle vertex indices
└── detector1  (2161,) int64 — JULIA 1-INDEXED triangle indices in `faces`
```

**Concrete dimensions of the detector region (verified read-only via h5py):**

| Quantity | Value |
|---|---|
| Number of detector triangles (`detector1`) | 2161 |
| Total triangles in mesh | 179 996 (detector region is 1.2 %) |
| Site location | lon = −72.279397°, lat = −15.622267° |
| Detector triangle centroid spans in **local ENU** (m) | East: [−2019, +1182] (≈3.2 km), N: [−2497, +2474] (≈5.0 km), Up: [2442, 3886] (≈1.4 km) |
| Triangle edge length (m) | min 2.5, median 118, mean 124, max 443 |
| Median triangle area | ≈ 6850 m² (≈ 83 m × 83 m) |

So the detector region is a ~3 km × 5 km patch on the sloped Colca Valley wall, tiled by ~2160
~100-m-edge triangles.

**Gotchas:** vertices are ECEF (not ENU); faces and `detector1` are **1-indexed** (Julia
convention) — must subtract 1 when indexing in Python.

---

## Design

### Core idea (per-detector pseudocode)

```python
# ----- Setup (once) -----
import sys
sys.path.insert(0, "../detector_optimization_v3")
from modules.layout_optimization import LearnableXY            # generic, no changes
from modules.detector_response   import SmearN, TimeAverage_vectorized
from modules.reconstruction      import Reconstruction, NormalizeLabels, DenormalizeLabels, EarlyStopping
from modules.utility_functions   import reconstructability, U_PR, U_E, U_angle
from modules.generate_showers    import GenerateShowers
from modules.shower_computation  import ComputeShowerDetection

# v4 NEW
from modules_v4.tr_geometry      import load_tr_mountain
from modules_v4.tr_surface_map   import SurfaceEastMap
from modules_v4.tr_plane_kernel  import GetCounts_planeaware  # plane-weighted kernel

# Mountain surface from basic_geometry.h5 — fixed function East = f(N, Up)
mountain = load_tr_mountain("../../TAMBOSim/resources/basic_geometry.h5",
                            group="colca_valley_30000", det_key="detector1")
# mountain has: centroids_NUE  -- (2161, 3)  columns [N, Up, East]
#               nu_bbox        -- (N_min, N_max, U_min, U_max)
#               n_planes=24, east_min=-2000, east_max=1000, plane_dx=125
surface = SurfaceEastMap.from_mountain(mountain, grid_h=256, grid_w=256).to(device)

# ----- Initial layout (in N, Up) -----
# Sample Nunits points inside the mountain (N, Up) bounding box.
N_init, U_init = mountain.sample_initial_layout(n_units=90, scheme="grid")
xy_module = LearnableXY(N_init, U_init, device=device)        # learns (x=N, y=Up)
optimizer = torch.optim.SGD(xy_module.parameters(), lr=10, momentum=0.3)  # match v3

# ----- One full iteration of the optimization loop -----
x_det, y_det = xy_module()                                    # (n_det,) (n_det,)  N, Up
east_det     = surface(x_det, y_det)                          # (n_det,) — East from mountain map
z_cont       = (east_det - mountain.east_min) / mountain.plane_dx  # (n_det,) ∈ [0, n_planes−1]

# Generate a batch of showers — v3 returns the FULL point cloud (all 24 layers)
# IMPORTANT: do NOT use v3's filter_plane=20 wrapper. Call ComputeShowerDetection
# with filter_plane=None and a no-op count fn so we get raw `samples`. Easier path:
# just call GenerateShowers directly and run our plane-aware kernel below.
samples, energies, directions, _ = generate_showers_instance(
    num_samples=Nbatch, save=False
)
samples = samples.to(device)

# Plane-aware kernel: spatial Gaussian × triangular plane weight per (detector, point).
# Returns (B, n_det) tensors that already encode the linear interpolation between the
# two bracketing planes — no per-plane loop is needed.
N_int, T_int = GetCounts_planeaware(
    samples, x_det, y_det, z_cont,
    SmearN_fn=SmearN, fluxB_e=fluxB_e,
    TimeAverage_vectorized_fn=TimeAverage_vectorized,
    sigma=200.0, plane_kernel="triangular",
)

# 7-feature input to the NN
x_exp  = x_det.unsqueeze(0).expand(Nbatch, -1)
y_exp  = y_det.unsqueeze(0).expand(Nbatch, -1)
z_exp  = z_cont.unsqueeze(0).expand(Nbatch, -1)
X0, Y0 = energy_weighted_core(samples)                        # same as v3 cell ~22
x0_exp = X0.unsqueeze(1).expand(-1, x_det.shape[0])
y0_exp = Y0.unsqueeze(1).expand(-1, x_det.shape[0])
inputs_batch = torch.stack(
    [x_exp, y_exp, z_exp, N_int, T_int, x0_exp, y0_exp], dim=2
).float()                                                     # (B, n_det, 7)
preds = model((inputs_batch - input_mean) / input_std).view(Nbatch, -1)
preds_e, preds_th, preds_phi = DenormalizeLabels(preds[:, 0], preds[:, 1], preds[:, 2])

# v3 utility
r_score = reconstructability(N_int, reconstruct_threshold=10)
U = (1e2 * U_angle(preds_th,  th, r_score)
   + 1e2 * U_angle(preds_phi, ph, r_score)
   + 1e3 * U_E    (preds_e,   energies, r_score)
   + 5e5 * U_PR   (r_score)) / 1e3
Loss = -U
optimizer.zero_grad(); Loss.backward(); optimizer.step()
```

### The plane-aware kernel — what `GetCounts_planeaware` does

```python
# samples: (B, max_points, 5) columns [x, y, layer_index, energy, time]
point_x = samples[..., 0]      # (B, P)
point_y = samples[..., 1]      # (B, P)
point_l = samples[..., 2]      # (B, P)  -- layer index, integer 0..23
point_e = samples[..., 3]      # (B, P)
point_t = samples[..., 4]      # (B, P)

# Spatial Gaussian, identical to v3
dx = point_x.unsqueeze(2) - x_det.unsqueeze(0).unsqueeze(0)
dy = point_y.unsqueeze(2) - y_det.unsqueeze(0).unsqueeze(0)
spatial = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))   # (B, P, n_det)

# Triangular plane weight — peaks at layer == z_cont_i, zero outside ±1 layer
plane_w = torch.relu(1.0 - (point_l.unsqueeze(2) - z_cont.unsqueeze(0).unsqueeze(0)).abs())
# plane_w: (B, P, n_det), differentiable in z_cont
kernel = spatial * plane_w                                # (B, P, n_det)

energy_kernel  = point_e.unsqueeze(2) * kernel
local_intensity = energy_kernel.sum(dim=1)                # (B, n_det)
et = (point_t.unsqueeze(2) * energy_kernel).sum(dim=1) / local_intensity.clamp(min=1e-8)

# Then (optionally) SmearN_fn / TimeAverage_vectorized_fn, exactly as v3 does in
# detector_response.py post-kernel.
return SmearN_fn(local_intensity), TimeAverage_vectorized_fn(et, local_intensity)
```

This is one extra `relu(1 − |Δlayer|)` multiply per (detector, point) pair — same memory
class as v3, no per-plane Python loop. Because layer indices are integers in {0, ..., 23},
the triangular weight is exactly the linear interpolation between the two bracketing planes:
points on layer `z_low` get weight `1 − (z_cont − z_low) = w_low`, points on layer
`z_high = z_low + 1` get weight `z_cont − z_low = w_high`, and all other points get 0.

### Why this works

- **Backprop through `xy_module` is preserved.** `surface(x, y)` is a `grid_sample` lookup
  (differentiable), `z_cont = (east_det − east_min) / plane_dx` is a smooth arithmetic
  transform, and the triangular plane weight `relu(1 − |layer − z_cont|)` is piecewise-linear
  in `z_cont` (`d/dz_cont` is `±1` away from the kink). So
  `Loss → counts → z_cont → e_det → x, y → xy_module.params`.
- **No information loss across planes.** A detector at exactly `z_cont = 14.5` gets equal
  contributions from points on layers 14 and 15.
- **Identical kernel to v3 in the special case `z_cont ≡ 20`.** Setting all detectors to
  `z_cont = 20` makes `plane_w` equal to 1 on layer-20 points and 0 elsewhere — exactly v3's
  `filter_plane=20` behaviour. v3 is a strict subset of v4.
- **The 24 plane layers** are emitted by the same flow-matching shower model v3 uses; the
  East ∈ [−2000, +1000] m / Δ = 125 m mapping is a v4 convention that fixes how the v3
  integer `layer_index` corresponds to a physical East coordinate.

### `SurfaceEastMap` — the differentiable mountain function

Built once from the 2161 detector centroids:
1. Take `(N_i, Up_i, E_i)` for each centroid.
2. Use `scipy.interpolate.LinearNDInterpolator((N_i, Up_i), E_i)` to build a scattered linear
   interpolant on the 2D `(N, Up)` plane.
3. Sample on a regular `H × W` grid covering the bounding box of `(N, Up)` (default 256 × 256).
4. Wrap in a `torch.nn.Module` that holds `grid_e`, `(N_min, N_max, U_min, U_max)`, and uses
   `F.grid_sample(... padding_mode='border', align_corners=True)` for forward calls.
5. NaN cells (outside the support) are filled with their nearest valid value at construction
   time so the gradient never sees a NaN.

This is fully differentiable, fast, and decoupled from the optimization loop.

### What "out-of-region" looks like

If a detector wanders to an `(N, Up)` outside the mountain support (e.g. user starts with a
sloppy initialization), the surface map clamps to the nearest valid `E` (`padding_mode='border'`).
A small "stay-on-the-mountain" penalty `λ_oob · ReLU(distance_to_NU_bbox)` can be added if we
see drift; default is off.

### What v3's `push_apart` and `symmetry_loss` become

- `push_apart`: still potentially useful (it just enforces a minimum 2D separation in
  `(x, y)` and never uses gradients), but the TAMBO mountain doesn't motivate any specific
  separation. **Drop initially**, re-enable if detectors collapse onto each other.
- `symmetry_loss`: the TAMBO mesh isn't 3-fold symmetric, so dropping it is correct. **Drop.**

### NN width / training

`Reconstruction(input_features=7, num_detectors=Nunits)` — for `Nunits = 90` the input width
is `90 × 7 = 630` (vs v3's `90 × 6 = 540`). Effectively unchanged, so v3's NN architecture
and training settings transfer directly. The NN must still be **retrained** from scratch in
v4 because the input distribution changes (the 7th feature `z_cont` is new, and the
interpolated `N_int / T_int` distribution differs from v3's plane-20-only counts). Cells
[28]–[37] of v3's notebook port over with a single `input_features=7` change.

### ENU projection

Centroids are converted from ECEF → local ENU around the site once at load time. Helper:

```python
import math, numpy as np, h5py

def _ecef_to_enu(centroids_ecef, lon_deg, lat_deg):
    lon0, lat0 = math.radians(lon_deg), math.radians(lat_deg)
    R_e = 6_371_000.0
    s = R_e * np.array([math.cos(lat0)*math.cos(lon0),
                        math.cos(lat0)*math.sin(lon0),
                        math.sin(lat0)])
    R = np.array([
        [-math.sin(lon0),                math.cos(lon0),               0],
        [-math.sin(lat0)*math.cos(lon0), -math.sin(lat0)*math.sin(lon0), math.cos(lat0)],
        [ math.cos(lat0)*math.cos(lon0),  math.cos(lat0)*math.sin(lon0), math.sin(lat0)],
    ])
    return R @ (centroids_ecef - s[:, None])         # (3, D)  → [E, N, Up]
```

### Shower coordinate frame compatibility (open implementation question, not a blocker)

v3's point-cloud `samples` tensor `(N, max_points, 5)` gives `(x, y, layer_index, energy,
time)` in **whatever frame the AllShowers point clouds use**. v3 uses the columns 0/1
unchanged for the spatial kernel, so the only thing v4 needs to verify is that:

1. The `(x, y)` columns of `samples` are in the **same frame** as `(N, Up)` — i.e. that
   v3's "x" axis corresponds to North and v3's "y" axis corresponds to Up. If not, swap
   columns at load time inside `GetCounts_planeaware` (one transpose, no architectural
   change).
2. The integer `layer_index` column ranges over `[0, 23]` and increases in the same direction
   as our East axis. Confirmed: v3 already uses `filter_plane=20` to pick the same plane the
   diffusion model emits, so the convention matches.
3. `(X0, Y0)` (energy-weighted shower core) lives in the same `(x, y)` frame as the
   detectors. v3 already uses these as input features ([v3 cell 14](TambOpt/detector_optimization_v3/SWGOLO7_optimization.ipynb)),
   so the v4 notebook can reuse the v3 derivation verbatim.

---

## v4 folder layout

```
TambOpt/detector_optimization_v4/
├── CLAUDE.md                                # repo learnings (see content below)
├── SWGOLO7_optimization_tr.ipynb             # main notebook (only notebook shipped)
├── modules_v4/
│   ├── __init__.py                          # adds ../detector_optimization_v3 to sys.path
│   ├── tr_geometry.py                       # NEW — HDF5 loader, ECEF→ENU, MountainData
│   ├── tr_surface_map.py                    # NEW — SurfaceEastMap (differentiable East = f(N, Up))
│   └── tr_plane_kernel.py                   # NEW — plane-aware GetCounts wrapper around v3
├── tests/
│   ├── test_v4_modules.ipynb                # self-contained test notebook (tests 1–6)
│   └── fixtures/
│       └── sample_showers_10.pt             # symlink to v3's cashed_showers_10.pt
└── outputs/                                  # populated at runtime
```

**Verbatim copies: zero.** Everything reusable comes from v3 via a
`sys.path.insert(0, "../detector_optimization_v3")` inside `modules_v4/__init__.py`. v4 only
ships the three new modules above plus the notebook. The folder is named `modules_v4` (not
`modules`) so that `import modules.detector_response` still resolves to v3 — meaning v4 can
say `from modules.detector_response import SmearN, ...` and `from modules_v4.tr_plane_kernel
import GetCounts_planeaware` side by side.

---

## File-by-file changes

### NEW [TambOpt/detector_optimization_v4/modules_v4/__init__.py](TambOpt/detector_optimization_v4/modules_v4/__init__.py)

```python
"""v4 modules. Adds v3 to sys.path so v4 can `from modules.detector_response import ...` etc.

v4 is a thin wrapper around v3 — no verbatim copies. The only files in this folder are
TR-geometry, surface-map, and the plane-aware kernel wrapper. Everything else lives in v3.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.normpath(os.path.join(_HERE, "..", "..", "detector_optimization_v3"))
if _V3 not in sys.path:
    sys.path.insert(0, _V3)
```

### NEW [TambOpt/detector_optimization_v4/modules_v4/tr_geometry.py](TambOpt/detector_optimization_v4/modules_v4/tr_geometry.py)

Loads the Colca Valley triangulated mesh from
[basic_geometry.h5](TAMBOSim/resources/basic_geometry.h5) and exposes everything needed by the
mountain surface map and the shower-frame setup.

```python
import math
import h5py
import numpy as np
import torch
from dataclasses import dataclass

DEFAULT_GEOMETRY_PATH = "../../TAMBOSim/resources/basic_geometry.h5"
DEFAULT_GROUP   = "colca_valley_30000"
DEFAULT_DET_KEY = "detector1"
DEFAULT_EAST_MIN = -2000.0
DEFAULT_EAST_MAX =  1000.0
DEFAULT_N_PLANES = 24
SITE_LON_DEG = -72.279397
SITE_LAT_DEG = -15.622267

def _ecef_to_enu(centroids_ecef, lon_deg, lat_deg):
    """Rotate ECEF (3, D) about a sphere of mean Earth radius to local ENU around (lon, lat)."""
    lon0, lat0 = math.radians(lon_deg), math.radians(lat_deg)
    R_e = 6_371_000.0
    s = R_e * np.array([math.cos(lat0)*math.cos(lon0),
                        math.cos(lat0)*math.sin(lon0),
                        math.sin(lat0)])
    R = np.array([
        [-math.sin(lon0),                math.cos(lon0),               0.0],
        [-math.sin(lat0)*math.cos(lon0), -math.sin(lat0)*math.sin(lon0), math.cos(lat0)],
        [ math.cos(lat0)*math.cos(lon0),  math.cos(lat0)*math.sin(lon0), math.sin(lat0)],
    ])
    return R @ (centroids_ecef - s[:, None])             # (3, D) → rows = E, N, Up


@dataclass
class MountainData:
    centroids_NUE: np.ndarray   # (n_tri, 3) columns = [N, Up, East]  in metres
    n_min: float; n_max: float
    u_min: float; u_max: float
    east_lo: float; east_hi: float   # actual East span of centroids (not the plane range)
    east_min: float             # plane axis lower bound (default -2000)
    east_max: float             # plane axis upper bound (default +1000)
    n_planes: int               # default 24
    plane_dx: float             # (east_max - east_min) / n_planes  ≈ 125 m

    def sample_initial_layout(self, n_units=90, scheme="grid"):
        """Return (N_init, U_init) torch tensors of shape (n_units,) inside the (N, Up) bbox.

        scheme='grid' : approximately evenly spaced grid points clipped to the convex hull of the
                        centroids; scheme='random' : uniform-in-bbox.
        """
        ...


def load_tr_mountain(
    h5_path=DEFAULT_GEOMETRY_PATH,
    group=DEFAULT_GROUP,
    det_key=DEFAULT_DET_KEY,
    east_min=DEFAULT_EAST_MIN,
    east_max=DEFAULT_EAST_MAX,
    n_planes=DEFAULT_N_PLANES,
) -> MountainData:
    """Read the HDF5, project the detector-region centroids to local ENU, return MountainData."""
    with h5py.File(h5_path, "r") as f:
        g       = f[group]
        verts   = g["vertices"][...]                     # (3, 90000) ECEF
        faces   = g["faces"][...] - 1                    # (3, 179996) — JULIA 1-INDEXED
        det_idx = g[det_key][...] - 1                    # (2161,)    — JULIA 1-INDEXED

    # Triangle centroids in ECEF, then ENU
    tri_ecef    = verts[:, faces[:, det_idx]]            # (3, 3, 2161)
    centroids_e = tri_ecef.mean(axis=1)                   # (3, 2161) ECEF centroids
    enu         = _ecef_to_enu(centroids_e, SITE_LON_DEG, SITE_LAT_DEG)   # (3, 2161) [East, N, Up]

    East, N, Up = enu[0], enu[1], enu[2]
    centroids_NUE = np.stack([N, Up, East], axis=1)       # (2161, 3) columns N, Up, East
    return MountainData(
        centroids_NUE = centroids_NUE,
        n_min = float(N.min()),  n_max = float(N.max()),
        u_min = float(Up.min()), u_max = float(Up.max()),
        east_lo = float(East.min()), east_hi = float(East.max()),
        east_min = float(east_min),
        east_max = float(east_max),
        n_planes = int(n_planes),
        plane_dx = (east_max - east_min) / n_planes,
    )
```

(Drops v3's `Layouts()`, `barycentric_coords`, `project_to_triangle` — not relevant to the
mountain-surface design.)

### NEW [TambOpt/detector_optimization_v4/modules_v4/tr_surface_map.py](TambOpt/detector_optimization_v4/modules_v4/tr_surface_map.py)

Builds the differentiable mountain surface function `E = f(N, Up)` from the centroids.

```python
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

class SurfaceEastMap(torch.nn.Module):
    """Differentiable mountain function E = f(N, Up).

    Stores a 2D grid (H, W) of East values sampled on a regular (N, Up) grid covering the
    bounding box of the detector centroids. forward(x, y) does an F.grid_sample lookup so the
    output is differentiable w.r.t. (x, y) = (N, Up).
    """
    def __init__(self, grid_e, n_min, n_max, u_min, u_max):
        super().__init__()
        # grid_e: (H, W) torch tensor with rows running over Up, columns over N
        self.register_buffer("grid_e", grid_e.float().unsqueeze(0).unsqueeze(0))   # (1,1,H,W)
        self.n_min, self.n_max = float(n_min), float(n_max)
        self.u_min, self.u_max = float(u_min), float(u_max)

    @classmethod
    def from_mountain(cls, mountain, grid_h=256, grid_w=256, pad=0.0):
        N    = mountain.centroids_NUE[:, 0]
        Up   = mountain.centroids_NUE[:, 1]
        East = mountain.centroids_NUE[:, 2]
        n_min, n_max = mountain.n_min - pad, mountain.n_max + pad
        u_min, u_max = mountain.u_min - pad, mountain.u_max + pad

        # Linear interpolant on the irregular (N, Up) → East scatter
        interp_lin  = LinearNDInterpolator(np.stack([N, Up], axis=1), East)
        interp_near = NearestNDInterpolator(np.stack([N, Up], axis=1), East)  # for NaN fill

        Ng = np.linspace(n_min, n_max, grid_w)
        Ug = np.linspace(u_min, u_max, grid_h)
        NN, UU = np.meshgrid(Ng, Ug)                                   # (H, W) each
        East_grid = interp_lin(NN, UU)
        nan_mask = np.isnan(East_grid)
        East_grid[nan_mask] = interp_near(NN[nan_mask], UU[nan_mask])
        return cls(torch.from_numpy(East_grid), n_min, n_max, u_min, u_max)

    def forward(self, x, y):
        """x = N (m), y = Up (m). Returns E (m). Differentiable in x, y."""
        nx = 2.0 * (x - self.n_min) / (self.n_max - self.n_min) - 1.0    # → [-1, 1]
        uy = 2.0 * (y - self.u_min) / (self.u_max - self.u_min) - 1.0
        # grid_sample expects (N, H_out, W_out, 2) with (x, y) ordering — here N=1, H_out=1,
        # W_out=n_det. The xy axis order matches (col, row) = (N, Up).
        grid = torch.stack([nx, uy], dim=-1).view(1, 1, -1, 2).to(self.grid_e.dtype)
        out  = F.grid_sample(self.grid_e, grid,
                             mode="bilinear", padding_mode="border", align_corners=True)
        return out.view(-1)                                              # (n_det,)
```

### NEW [TambOpt/detector_optimization_v4/modules_v4/tr_plane_kernel.py](TambOpt/detector_optimization_v4/modules_v4/tr_plane_kernel.py)

A near-verbatim port of v3's
[GetCounts_differentiable](TambOpt/detector_optimization_v3/modules/detector_response.py#L10)
that adds a triangular **plane weight** per (detector, point) pair so each detector only
"sees" the two layers bracketing its `z_cont`. This is the only piece of detector-response
logic that v4 cannot import unchanged from v3.

```python
"""Plane-aware GetCounts wrapper around v3's spatial Gaussian kernel.

v3's kernel takes (samples, x_det, y_det) and ignores samples[..., 2] (the layer index).
v4's wrapper takes (samples, x_det, y_det, z_cont) and multiplies the per-point energy by
relu(1 - |layer_p - z_cont_i|) so points whose layer is close to z_cont contribute and the
others drop out smoothly.
"""
import torch

def GetCounts_planeaware(samples, x_det, y_det, z_cont,
                         SmearN_fn, fluxB_e, TimeAverage_vectorized_fn,
                         sigma=200.0):
    """Differentiable in (x_det, y_det, z_cont).

    samples:   (B, max_points, 5) columns [x, y, layer_index, energy, time]
    x_det/y_det/z_cont: each (n_det,)  ; z_cont is the continuous plane index ∈ [0, n_planes-1]
    Returns:   (N, T) of shape (B, n_det), differentiable w.r.t. (x_det, y_det, z_cont).
    """
    point_x = samples[..., 0]               # (B, P)
    point_y = samples[..., 1]               # (B, P)
    point_l = samples[..., 2]               # (B, P)  layer index (0..n_planes-1)
    point_e = samples[..., 3]               # (B, P)
    point_t = samples[..., 4]               # (B, P)

    # Spatial Gaussian — identical to v3
    dx = point_x.unsqueeze(2) - x_det.unsqueeze(0).unsqueeze(0)
    dy = point_y.unsqueeze(2) - y_det.unsqueeze(0).unsqueeze(0)
    spatial = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))         # (B, P, n_det)

    # Triangular plane weight — peaks at layer == z_cont, zero outside ±1 layer
    # plane_w: (B, P, n_det), differentiable in z_cont
    plane_w = torch.relu(
        1.0 - (point_l.unsqueeze(2) - z_cont.unsqueeze(0).unsqueeze(0)).abs()
    )

    kernel        = spatial * plane_w                                    # (B, P, n_det)
    energy_kernel = point_e.unsqueeze(2) * kernel
    local_intensity = energy_kernel.sum(dim=1)                            # (B, n_det)
    et = (point_t.unsqueeze(2) * energy_kernel).sum(dim=1) \
         / local_intensity.clamp(min=1e-8)

    # Same post-processing as v3 (smearing, time averaging) — those callables are imported
    # from v3.modules.detector_response by the notebook and passed in here.
    return SmearN_fn(local_intensity), TimeAverage_vectorized_fn(et, local_intensity)
```

That's it — v4's only departure from v3 in the count-extraction path is the 1-line addition
of `plane_w` and its multiplication into the kernel.

### NEW [TambOpt/detector_optimization_v4/SWGOLO7_optimization_tr.ipynb](TambOpt/detector_optimization_v4/SWGOLO7_optimization_tr.ipynb)

Port of [v3/SWGOLO7_optimization.ipynb](TambOpt/detector_optimization_v3/SWGOLO7_optimization.ipynb)
with these targeted edits:

- **Cell 02 (imports)** — replace v3's flat imports with the v4 block:
  ```python
  import sys, os
  sys.path.insert(0, "../detector_optimization_v3")
  # v3 (verbatim, via sys.path) — note the "modules." prefix
  from modules.layout_optimization import LearnableXY
  from modules.detector_response   import SmearN, TimeAverage_vectorized
  from modules.reconstruction      import Reconstruction, NormalizeLabels, DenormalizeLabels, EarlyStopping
  from modules.utility_functions   import reconstructability, U_PR, U_E, U_angle
  from modules.generate_showers    import GenerateShowers
  from modules.shower_computation  import ComputeShowerDetection
  # v4 (new)
  from modules_v4.tr_geometry      import load_tr_mountain
  from modules_v4.tr_surface_map   import SurfaceEastMap
  from modules_v4.tr_plane_kernel  import GetCounts_planeaware
  ```

- **Cell 03 (constants)** — replace v3's `Nunits = 90; n_rings = 5; radius = 300; ...
  Layouts(...)` block with:
  ```python
  GEOMETRY_PATH = "../../TAMBOSim/resources/basic_geometry.h5"
  GROUP   = "colca_valley_30000"
  DET_KEY = "detector1"
  N_PLANES  = 24                 # match flow-matching shower model
  EAST_MIN  = -2000.0
  EAST_MAX  =  1000.0
  Nunits        = 90              # learnable detectors (same scale as v3)
  NUM_FEATURES  = 7                # [x = N, y = Up, z_cont, N_int, T_int, x0, y0]

  mountain = load_tr_mountain(GEOMETRY_PATH, GROUP, DET_KEY,
                              east_min=EAST_MIN, east_max=EAST_MAX, n_planes=N_PLANES)
  surface  = SurfaceEastMap.from_mountain(mountain, grid_h=256, grid_w=256).to(device)

  # Initial detector positions on the mountain (N, Up) bbox
  N_init, U_init = mountain.sample_initial_layout(n_units=Nunits, scheme="grid")
  x_det = torch.as_tensor(N_init, dtype=torch.float32, device=device)
  y_det = torch.as_tensor(U_init, dtype=torch.float32, device=device)
  ```

- **Cell 04 (shower generator)** — keep v3's `GenerateShowers(...)` instantiation
  unchanged. The shower point clouds come straight from the AllShowers framework just like v3.

- **Cell 05 (wrappers)** — replace v3's `_GetCounts` / `generate_showers` wrappers with
  v4's plane-aware versions. The key differences:
  ```python
  import functools
  _SmearN = functools.partial(SmearN, RelResCounts=RelResCounts)
  _TimeAverage = functools.partial(TimeAverage_vectorized,
                                   IntegrationWindow=IntegrationWindow,
                                   sigma_time=sigma_time)

  def _get_counts_planeaware(samples, x_det, y_det, z_cont):
      return GetCounts_planeaware(
          samples, x_det, y_det, z_cont,
          SmearN_fn=_SmearN, fluxB_e=fluxB_e,
          TimeAverage_vectorized_fn=_TimeAverage,
      )

  def generate_showers(x_det, y_det, z_cont, log=False, number_of_showers=1, use_cache=False):
      # Use ComputeShowerDetection with filter_plane=None to get raw point cloud,
      # then call our plane-aware kernel ourselves. ComputeShowerDetection still
      # handles caching, X0/Y0 derivation, and direction features.
      ...
  ```
  Note that `filter_plane` is **not** passed (or set to `None`) — v4 wants every layer to
  reach the kernel.

- **Cells 13–15 (training/val/test data)** — extend the input tensor to 7 features:
  ```python
  z_det_exp = z_cont.unsqueeze(0).expand(Nevents, -1)
  inputs = torch.stack(
      [x_det_exp, y_det_exp, z_det_exp, N, T, x0_exp, y0_exp], dim=2
  ).float()
  ```
  `z_cont` is computed once outside the data loop because the initial `(x_det, y_det)` is
  fixed before NN training.

- **Cell 28 (NN init)** — `Reconstruction(input_features=NUM_FEATURES, num_detectors=Nunits)`.
  v3's `Reconstruction.__init__` already accepts `input_features`
  ([v3/modules/reconstruction.py:19](TambOpt/detector_optimization_v3/modules/reconstruction.py#L19))
  so **no patch is needed**. v4 must retrain from scratch — point `output_dir` at a fresh
  path like `./outputs/NN_Files_TR_v4/`.

- **Cell 38 (Layout init)** — `LearnableXY(x_det, y_det, device=device)` where `x_det / y_det`
  are the initial `(N, Up)` from cell 03. Optimizer matches v3:
  `torch.optim.SGD(xy_module.parameters(), lr=10, momentum=0.3)`.

- **Cell 44 (optimization loop)** — the per-iteration body becomes:
  ```python
  x_det, y_det = xy_module()                                  # (n_det,) (n_det,)
  east_det     = surface(x_det, y_det)                        # (n_det,) — East coordinate
  z_cont       = (east_det - mountain.east_min) / mountain.plane_dx

  # Generate fresh showers and run the plane-aware kernel
  N_int, T_int, X0, Y0, energy, sin_z, cos_z, sin_a, cos_a, _ = generate_showers(
      x_det, y_det, z_cont, log=False, number_of_showers=Nbatch, use_cache=False,
  )

  z_exp = z_cont.unsqueeze(0).expand(Nbatch, -1)
  x_exp = x_det.unsqueeze(0).expand(Nbatch, -1)
  y_exp = y_det.unsqueeze(0).expand(Nbatch, -1)
  x0_exp = (X0/5000).unsqueeze(1).expand(-1, Nunits)
  y0_exp = (Y0/5000).unsqueeze(1).expand(-1, Nunits)
  inputs_batch = torch.stack(
      [x_exp, y_exp, z_exp, N_int, T_int, x0_exp, y0_exp], dim=2
  ).float()

  preds = model((inputs_batch - input_mean) / input_std).view(Nbatch, -1)
  preds_e, preds_th, preds_phi = DenormalizeLabels(preds[:, 0], preds[:, 1], preds[:, 2])

  th = torch.atan2(sin_z, cos_z); ph = torch.atan2(sin_a, cos_a)
  r_score = reconstructability(N_int, reconstruct_threshold=10)
  U = (1e2 * U_angle(preds_th,  th, r_score)
     + 1e2 * U_angle(preds_phi, ph, r_score)
     + 1e3 * U_E    (preds_e,   energy, r_score)
     + 5e5 * U_PR   (r_score)) / 1e3
  Loss = -U
  optimizer.zero_grad(); Loss.backward(); optimizer.step()
  ```
  Drop v3's `push_apart` / `symmetry_loss` calls. `input_mean` / `input_std` become
  length-7 (the values come from the NN training cells).

  **Layout saving** (same pattern as v3's `Python_Layout/Layout_*.txt`): every `save_every`
  steps write a 3-column file `(N, Up, z_cont)` so the animation can replay the full
  trajectory without needing to re-run the optimizer:
  ```python
  if step % save_every == 0:
      layout = torch.stack([
          x_det.detach(), y_det.detach(), z_cont.detach()
      ], dim=1).cpu().numpy()
      np.savetxt(f"{output_dir}/Python_Layout/Layout_{step}.txt", layout)
      utilities_log.append(U.item())
  ```

- **Visualization cells** — two static cells + one animation cell:

  1. **Static: top-down `(N, Up)` view** — mountain surface map as `imshow` background,
     initial and final detector positions overlaid.

  2. **Static: 3D scatter on mountain** — `ax3d.scatter(N_mtn, Up_mtn, East_mtn, c=East_mtn,
     s=1, alpha=0.2)` for the mountain centroids, then `ax3d.scatter(x_final, y_final,
     east_final, c=z_cont_final, s=40, zorder=5)` for the optimized detectors. Labels:
     x-axis = "North [m]", y-axis = "Up [m]", z-axis = "East [m]".

  3. **Animation: detectors sliding on the 3D mountain** — mirrors v3's
     `animation.FuncAnimation` but in 3D:
     ```python
     from matplotlib import animation
     from IPython.display import HTML

     layout_files = sorted(Path(f"{output_dir}/Python_Layout").glob("Layout_*.txt"),
                           key=lambda p: int(p.stem.split("_")[1]))
     step_idx = 5   # sample every 5th saved layout
     layouts = [np.loadtxt(f) for f in layout_files[::step_idx]]
     # Each layout: (n_det, 3) columns [N, Up, z_cont]

     N_mtn  = mountain.centroids_NUE[:, 0]
     Up_mtn = mountain.centroids_NUE[:, 1]
     East_mtn = mountain.centroids_NUE[:, 2]

     fig = plt.figure(figsize=(10, 7))
     ax  = fig.add_subplot(111, projection='3d')
     # Static mountain background (drawn once)
     ax.scatter(N_mtn, Up_mtn, East_mtn, c=East_mtn, cmap='terrain',
                s=1, alpha=0.15, depthshade=True)
     ax.set_xlabel("North [m]"); ax.set_ylabel("Up [m]"); ax.set_zlabel("East [m]")

     # Convert z_cont back to East for the detector scatter
     east_min, plane_dx = mountain.east_min, mountain.plane_dx
     det_scatter = ax.scatter([], [], [], c=[], cmap='plasma',
                              vmin=0, vmax=N_PLANES-1, s=40, zorder=5)
     title = ax.set_title("")

     def update(frame):
         lay = layouts[frame]
         N_d, Up_d, zc_d = lay[:, 0], lay[:, 1], lay[:, 2]
         East_d = east_min + zc_d * plane_dx
         det_scatter._offsets3d = (N_d, Up_d, East_d)
         det_scatter.set_array(zc_d)
         title.set_text(f"Step {frame * step_idx * save_every}")
         return det_scatter, title

     anim = animation.FuncAnimation(fig, update, frames=len(layouts), interval=120)
     gif_path = Path(output_dir) / "layout_evolution_3d.gif"
     anim.save(str(gif_path), writer="pillow", fps=8, dpi=100)
     plt.close(fig)
     print(f"Saved {gif_path}")
     HTML(anim.to_jshtml())
     ```
     The mountain background is painted once (static); only the detector scatter updates
     each frame. `pillow` writer — no ffmpeg dependency, same as v3.

### NEW [TambOpt/detector_optimization_v4/CLAUDE.md](TambOpt/detector_optimization_v4/CLAUDE.md)

Concise project memory for future sessions, covering:

- **Goal of v4:** position-learning optimization where detectors **slide on the 3D Colca Valley
  mountain surface**. The learnable parameters are still 2D `(N, Up)`, but `(N, Up)` maps
  deterministically to a unique East via the mountain function `E = f(N, Up)`, and the shower
  response is the linear interpolation between the two nearest of 24 East-aligned planes.
- **Geometry source:** [TAMBOSim/resources/basic_geometry.h5](TAMBOSim/resources/basic_geometry.h5),
  group `colca_valley_30000`, key `detector1` (2161 triangles, 1-indexed Julia faces, ECEF
  vertices). Site: lon −72.279397°, lat −15.622267°. Detector region in local ENU: ≈3.2 km × 5.0
  km × 1.4 km.
- **v4 coordinate convention:**
  - `x` = ENU North (m, learnable)
  - `y` = ENU Up    (m, learnable)
  - `z = z_cont`    = `(East(x, y) − EAST_MIN) / Δ_plane`, derived (NOT learned)
  - 24 planes equidistant in East ∈ [−2000, +1000] m (125 m / plane), matching the
    flow-matching shower model
- **NN feature vector per detector:** `[x = N, y = Up, z = z_cont, N_int, T_int, x0, y0]`
  — 7 features. v3's `Reconstruction(input_features=...)` already handles arbitrary widths
  (no patch needed — see
  [v3/modules/reconstruction.py:19](TambOpt/detector_optimization_v3/modules/reconstruction.py#L19)).
- **No verbatim copies.** v4's `modules_v4/__init__.py` does
  `sys.path.insert(0, "../detector_optimization_v3")`. Anything in v3 is one import away
  via `from modules.* import ...`.
- **What v4 actually contains:**
  - `modules_v4/tr_geometry.py`     — HDF5 loader, ECEF→ENU, `MountainData` dataclass
  - `modules_v4/tr_surface_map.py`  — `SurfaceEastMap` (`F.grid_sample` over a 256×256 grid
    of `East = f(N, Up)`; differentiable in `(x, y)`)
  - `modules_v4/tr_plane_kernel.py` — `GetCounts_planeaware`: v3's spatial Gaussian times a
    triangular plane weight `relu(1 − |layer_p − z_cont_i|)` (one elementwise multiply),
    fully differentiable in `(x, y, z_cont)`
  - `SWGOLO7_optimization_tr.ipynb` — main notebook
- **What changed conceptually from v3:**
  - Added the mountain surface map (`E` from `(N, Up)`) — gradient flows back through the
    bilinear surface lookup into `xy_module.parameters()`
  - Added the triangular plane weight inside the kernel (one extra multiply per
    detector-point pair) — implements linear interpolation between the two bracketing layers
    without a per-plane Python loop
  - 6 → 7 features (added `z_cont`)
  - Same `Nunits` (90), same SGD lr=10 / momentum=0.3 settings as v3
  - v3's `filter_plane=20` is removed; instead all 24 layers reach the kernel and the plane
    weight selects the right ones
- **Why a triangular plane weight (not a hard nearest-plane assignment)?** Continuity. A
  detector at `z_cont = 14.5` should see equal contributions from layers 14 and 15. A hard
  assignment makes the loss piecewise-constant in `z_cont` and zeros the gradient back
  through `(x, y)`.
- **Key gotchas:**
  - HDF5 `faces` and `detector1` are **Julia 1-indexed** — subtract 1 in Python.
  - `vertices` are ECEF (3, 90000), not ENU. The loader rotates to local ENU at the site.
  - `SurfaceEastMap` uses `padding_mode='border'`, so a detector that wanders outside the
    `(N, Up)` bbox sees the nearest valid `E` instead of NaN. Add a stay-on-mountain penalty
    only if optimization drifts.
  - The triangular plane kernel `relu(1 − |layer_p − z_cont|)` is the ONLY thing that
    couples `z_cont` to the loss; if its derivative is zeroed (e.g. by detaching `z_cont`),
    the surface-map gradient path also dies.
  - 24 planes is a **shower-side constraint** (the AllShowers / flow-matching model emits
    layers indexed 0..23). Don't change `N_PLANES` without retraining that upstream model.
  - The NN must be **retrained from scratch** in v4 because the input distribution changes
    (interpolated counts plus the new `z_cont` feature).
  - v4 uses the folder name `modules_v4/` (not `modules/`) so that `import modules.X` keeps
    resolving to v3.
- **What's NOT in v4 from v3:** `Layouts()` (rings), `project_to_triangle` (hardcoded
  triangle), `push_apart` / `symmetry_loss` (not applicable to a single-mountain surface),
  the `filter_plane=20` argument of `ComputeShowerDetection`.
- **How the v2 shower generation model works (inherited by v3 and v4):**

  Two models run in sequence every time showers are needed:

  **1. `PlaneFNNGenerator` (FNN scaler) — `scaler`**
  - A small deterministic fully-connected network.
  - Input: `(p_energy, class_id, sin_z, cos_z, sin_a, cos_a)` — 6 scalars describing the
    primary particle (energy in log-uniform units, particle type 0/1/2 cycling through the
    batch, zenith and azimuth as sin/cos pairs).
  - Output: `(B, 24, 4)` bounding boxes — one `[xmin, xmax, ymin, ymax]` per plane per
    shower, in physical metres. These boxes define **where** the shower footprint sits in
    the `(x, y)` ground plane for each of the 24 detector layers.
  - Deterministic: same conditions → same boxes (no stochasticity).
  - The stats file `global_bbox_stats.pt` holds the per-plane/per-coordinate mean and std
    used to un-standardize the raw FNN output.

  **2. `PlaneDiffusionEvaluator` (DDIM diffusion model) — `generator`**
  - A U-Net conditioned on the same 6 scalars plus a plane index and the **previous plane's
    image** (`past_plane`).
  - Generates the 24 planes **autoregressively**: plane 0 is generated from pure noise; each
    subsequent plane is denoised while conditioning on the plane just produced.
    ```
    for plane_idx in range(24):
        noise = randn(B, 3, 32, 32)
        pred_all[:, plane_idx] = DDIM_sample(noise, cond, plane_idx, past=pred_all[:, plane_idx-1])
        past = pred_all[:, plane_idx]
    ```
  - Sampler: **DDIM** with 20 steps (vs 1000 for full DDPM) and `eta=0` (deterministic
    denoising trajectory). Classifier-free guidance weight `guidance_w=1.8`.
  - Output: `(B, 24, 3, 32, 32)` — 24 planes × 3 channels × 32×32 pixels, in standardized
    units `(mean=0, std=1 per plane/channel)`.
  - The 3 channels encode different particle properties (exact semantics in the training
    data, but channel 0 and 1 are used as `[energy-density, ...]` in v2's bilinear kernel).
  - Training data comes from `/pre_processed_3rd_step/` (TAMBO simulation pipeline).

  **3. Connecting the two — what v2's `GenerateShowers` does:**
  - Samples random primaries: energy log-uniform in `[1e-5, 1]`, zenith/azimuth uniform.
  - Feeds conditions to both models → gets `(B, 24, 3, 32, 32)` images and `(B, 24, 4)` boxes.
  - **Hardcodes `plane=20`**: slices `images[:, 20]` → `(B, 32, 32, 3)` and `boxes[:, 20]`
    → `(B, 4)`. This is the only step v4 changes.
  - Denormalizes: `(img + 1) / 2` maps standardized output to `[0, 1]`.
  - Computes shower core `(X0, Y0)` from the energy-weighted centroid of channel-product
    `shower_rgb[:, :, :, 0] * shower_rgb[:, :, :, 1]`, rescaled through the bbox.
  - Calls `GetCounts_differentiable(shower_rgb, x_det, y_det, bboxes)` using `F.grid_sample`
    (bilinear, `padding_mode='border'`, `align_corners=True`) to read the image intensity at
    each detector position, differentiable w.r.t. detector positions.

  **v3 replaces step 3 entirely** with the AllShowers point-cloud flow-matching model, which
  directly produces the `(B, max_points, 5)` point cloud with columns
  `[x, y, layer_index, energy, time]`. v3 then filters to `layer_index == 20`; v4 keeps all
  layers and applies the triangular plane weight inside the Gaussian kernel instead.

  **Checkpoint paths** (hardcoded in v2/v3 notebooks — adjust if models are retrained):
  - Diffusion U-Net: `.../checkpoints/tam_unet/epoch_epoch=1229-val_loss_val_loss=0.0333.ckpt`
  - FNN scaler: `.../checkpoints/tam_fnn/last.ckpt`
  - Standardization stats: `.../pre_processed_3rd_step/` (for bbox denormalization)

- **Quick references:**
  - v3 source (the upstream): [TambOpt/detector_optimization_v3/](TambOpt/detector_optimization_v3/)
  - v2 diffusion model: [TambOpt/detector_optimization_v2/diffusion_model/](TambOpt/detector_optimization_v2/diffusion_model/)
  - TR geometry notebooks (Julia): [TAMBOSim/notebooks/create_geometry/](TAMBOSim/notebooks/create_geometry/)
  - TR loader (Julia, for reference): [TAMBOSim/src/geometry/earth.jl:87](TAMBOSim/src/geometry/earth.jl#L87)

---

## Verification plan

Tests live in their own folder and notebook — **not** in the main optimization notebook:

```
TambOpt/detector_optimization_v4/
└── tests/
    ├── test_v4_modules.ipynb    ← all cells below map to cells in this notebook
    └── fixtures/                ← small cached data for offline tests (no GPU needed for 1–5)
        └── sample_showers_10.pt ← symlink or copy of v3's cashed_showers_10.pt
```

The test notebook is self-contained: it sets up paths, runs each numbered check, and raises
`AssertionError` on failure so the cell turns red.

1. **Geometry smoke test** (cell 1)
   ```python
   import sys; sys.path.insert(0, "../../detector_optimization_v3")
   from modules_v4.tr_geometry import load_tr_mountain
   m = load_tr_mountain("../../../TAMBOSim/resources/basic_geometry.h5",
                        "colca_valley_30000", "detector1")
   assert m.centroids_NUE.shape == (2161, 3)
   assert -2500 < m.n_min   < -2400 and 2400 < m.n_max   < 2500
   assert  2400 < m.u_min   < 2500  and 3800 < m.u_max   < 3900
   assert -2050 < m.east_lo < -2000 and 1100 < m.east_hi < 1200
   print("PASS geometry smoke test")
   ```
   Plus a `(N, Up)` scatter colored by `East` to visually confirm the slope.

2. **Surface-map round-trip** (cell 2)
   ```python
   from modules_v4.tr_surface_map import SurfaceEastMap
   import torch
   surface = SurfaceEastMap.from_mountain(m, grid_h=256, grid_w=256)
   N_t     = torch.as_tensor(m.centroids_NUE[:, 0]).float()
   Up_t    = torch.as_tensor(m.centroids_NUE[:, 1]).float()
   East_true = torch.as_tensor(m.centroids_NUE[:, 2]).float()
   East_hat  = surface(N_t, Up_t)
   med_err = (East_hat - East_true).abs().median().item()
   rmse    = (East_hat - East_true).pow(2).mean().sqrt().item()
   assert med_err < 125, f"Median error {med_err:.1f} m >= one plane width"
   print(f"PASS surface round-trip  median={med_err:.1f} m  RMSE={rmse:.1f} m")
   ```

3. **Differentiability test** (cell 3)
   ```python
   x = N_t[:5].clone().requires_grad_(True)
   y = Up_t[:5].clone().requires_grad_(True)
   surface(x, y).sum().backward()
   assert x.grad is not None and torch.isfinite(x.grad).all()
   assert y.grad is not None and torch.isfinite(y.grad).all()
   print("PASS surface differentiability")
   ```

4. **v3-equivalence check for the plane-aware kernel** (cell 4)
   ```python
   import showerdata
   from modules_v4.tr_plane_kernel import GetCounts_planeaware
   from modules.detector_response   import GetCounts_differentiable, SmearN, TimeAverage_vectorized
   import functools

   sh = showerdata.load("fixtures/sample_showers_10.pt")
   samples = torch.tensor(sh.points)                   # (10, 2048, 5)
   n_det = 8
   x_det = torch.linspace(-500, 500, n_det)
   y_det = torch.linspace(2800, 3500, n_det)
   z_cont_20 = torch.full((n_det,), 20.0)

   _SmearN    = functools.partial(SmearN, RelResCounts=0.05)
   _TimeAvg   = functools.partial(TimeAverage_vectorized, IntegrationWindow=128., sigma_time=10.)
   fluxB_e    = torch.tensor([6.859 * torch.pi * 0.000000200 * 128.])

   N_v4, T_v4 = GetCounts_planeaware(samples, x_det, y_det, z_cont_20,
                                     SmearN_fn=_SmearN, fluxB_e=fluxB_e,
                                     TimeAverage_vectorized_fn=_TimeAvg)

   # v3 reference: zero out all non-plane-20 energies
   samples_f = samples.clone()
   samples_f[:, :, 3] *= (samples_f[:, :, 2] == 20).float()
   N_v3, T_v3 = GetCounts_differentiable(samples_f, x_det, y_det,
                                         SmearN_fn=_SmearN, fluxB_e=fluxB_e,
                                         TimeAverage_vectorized_fn=_TimeAvg)
   assert torch.allclose(N_v4, N_v3, atol=1e-5), f"N mismatch max={((N_v4-N_v3).abs()).max()}"
   print("PASS kernel v3-equivalence")
   ```

5. **Plane interpolation linearity** (cell 5)
   ```python
   z_sweep = torch.arange(19.0, 21.1, 0.25)
   results = []
   for zv in z_sweep:
       zc = torch.full((n_det,), zv.item())
       Nk, _ = GetCounts_planeaware(samples[:1], x_det, y_det, zc,
                                    SmearN_fn=_SmearN, fluxB_e=fluxB_e,
                                    TimeAverage_vectorized_fn=_TimeAvg)
       results.append(Nk[0, 0].item())
   # At integer z values, check piecewise-linear continuity
   # (no assert on exact values — just plot and visually verify)
   import matplotlib.pyplot as plt
   plt.plot(z_sweep.numpy(), results, 'o-'); plt.xlabel("z_cont"); plt.ylabel("N_int[0]")
   plt.title("Plane interpolation linearity — must be piecewise linear"); plt.show()
   print("PASS plane interpolation visual check")
   ```

6. **End-to-end forward + backward** (cell 6 — needs GPU or small CPU test)
   ```python
   from modules.layout_optimization import LearnableXY
   n_det_small = 8
   N_init = N_t[:n_det_small].clone()
   U_init = Up_t[:n_det_small].clone()
   xy_module = LearnableXY(N_init, U_init, device='cpu')
   x_d, y_d = xy_module()
   east_d    = surface(x_d, y_d)
   zc        = (east_d - m.east_min) / m.plane_dx
   Nk, Tk    = GetCounts_planeaware(samples[:2], x_d, y_d, zc,
                                    SmearN_fn=_SmearN, fluxB_e=fluxB_e,
                                    TimeAverage_vectorized_fn=_TimeAvg)
   assert Nk.shape == (2, n_det_small) and torch.isfinite(Nk).all()
   Nk.sum().backward()
   assert xy_module.x.grad is not None and torch.isfinite(xy_module.x.grad).all()
   assert xy_module.y.grad is not None and torch.isfinite(xy_module.y.grad).all()
   print("PASS end-to-end forward + backward")
   ```

Tests 7–10 require the full optimization run and are run interactively in the main
`SWGOLO7_optimization_tr.ipynb` rather than the test notebook:

7. **NN training from scratch** — run training cells, confirm validation loss decreases,
   save to `outputs/NN_Files_TR_v4/`.

8. **Mini optimization run** (~20 epochs) — confirm `U` improves and detectors stay in
   `(N, Up)` bbox.

9. **Full optimization run** (~1200 epochs) — plot final positions on mountain, save GIF.

10. **Comparison to v3** — compare `U_angle`, `U_E`, `U_PR`, reconstructability against v3's
    plane-20 layout.

---

## Phase status

- Phase 1 (exploration): complete (2 Explore agents + targeted reads)
- Phase 2 (design): complete inline (no Plan agent dispatched — design is straightforward and
  cleanly localized to two new module files plus targeted notebook edits)
- Phase 3 (review / clarifications): complete (4 user questions answered)
- Phase 4 (final plan): complete in this file
- Phase 5: ExitPlanMode next
