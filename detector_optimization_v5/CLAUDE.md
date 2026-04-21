# detector_optimization_v5 — CLAUDE.md

Project memory for the TAMBO TR detector layout optimizer — **evolutionary pruning variant**.
Read this at the start of every session before touching any file.

> **Status (2026-04-14): never-used LLM-generated boilerplate.**
> v5 was scaffolded in a single commit (`4c25ef4`, "v5 boilerplate") to keep a parallel
> evolutionary-pruning approach alive alongside v4's gradient-based optimizer. No
> production run has ever exercised this code. Treat it as a design sketch — several
> items below (EAST calibration, v4 baseline LR) are stale relative to the code that
> actually runs in v4 and v6.

---

## Goal

Find the optimal positions of 90 water-Cherenkov detectors on the 3D Colca Valley mountain
wall so that a reconstruction NN can best infer shower energy, zenith, and azimuth — using
an **evolutionary pruning algorithm** instead of gradient-based SGD.

**v5 extends v4 by replacing the optimizer:**
- Start with 10,000 candidate detectors spread across the mountain surface.
- Each generation: score every detector's marginal contribution to reconstruction utility
  via gradient saliency on a per-detector mask, prune the weakest, and mutate survivors.
- Converge to 90 detectors over ~30 generations (geometric decay schedule).
- Use a **DeepSets** (permutation-invariant, set-based) NN that handles variable detector
  counts natively — one model for the whole run, no per-milestone retraining.

---

## v4 Baseline (what v5 builds on)

**Source:** `TambOpt/detector_optimization_v4/`
**Entry notebook:** `SWGOLO7_optimization_tr.ipynb`

v4 optimizes 90 point detectors on the 3D mountain surface via **SGD(lr=0.5, momentum=0.3)**
(the value used by the Apr 13–14 active scripts — `lr=1` was the earlier default referenced
in v4's CLAUDE.md) on learnable (North, Up) parameters, with East derived via a differentiable
surface map. See `detector_optimization_v4/CLAUDE.md` for full details.

---

## Changes from v4 to v5

| Aspect | v4 | v5 |
|--------|----|----|
| Optimizer | SGD on nn.Parameters | Evolutionary pruning + mutation (no gradient on positions) |
| Detector count | Fixed 90 | Variable: 10,000 → 90 over ~30 generations |
| Learnable params | `LearnableXY` (North, Up as nn.Parameter) | None — positions stored in `Population` dataclass |
| NN architecture | v3's `Reconstruction` (flat FC, fixed input size 90×7=630) | `DeepSetsReconstruction` (shared per-det MLP + sum pool, variable N) |
| NN input shape | `(B, 90*7)` flattened | `(B, N, 7)` — N varies per generation |
| Fitness signal | `∂Loss/∂(N,Up)` via chain rule through z_cont/East/surface | `∂U/∂mask` via gradient saliency on per-detector gate |
| Selection | N/A (all 90 survive every epoch) | Keep top-k by saliency score, prune rest |
| Position updates | SGD step on gradient | Gaussian perturbation (mutation) + mountain reprojection |
| Layout file rows | Always 90 | Variable (10,000 → 90) |
| NN fine-tune | Every 5 SGD epochs | Every 5 EA generations |
| push_apart / symmetry_loss | Dropped in v4 | Still dropped |

---

## v5 Coordinate Convention (unchanged from v4)

| Symbol | Meaning | Source |
|--------|---------|--------|
| `x` | ENU North [m] | Stored in Population |
| `y` | ENU Up / elevation [m] | Stored in Population |
| `z_cont` | `(EAST_ENTRY − East(x,y)) / LAYER_EAST_DX`, continuous AllShowers layer index ∈ [0, 23] | Derived (cached) |

AllShowers layer-East calibration — **two coexisting sets of values in the codebase**:

- **Correct values** (v4 active scripts, v6): `EAST_ENTRY = 1500 m`, `LAYER_EAST_DX = 150 m`.
  Per the user (2026-04-14), these are the right numbers — every real v4 run from
  Apr 13 onward uses them. With this calibration, `z_cont` spans ≈ [2.1, 23.5] across
  the mountain so **all 24 AllShowers layers are reachable**.
- **Stale values still present in this file and in `SWGOLO7_optimization_ev.ipynb`**:
  `EAST_ENTRY = −212 m`, `LAYER_EAST_DX = 307 m`. Per user: "all data was sampled from
  the last plane and the mountain was mismatched". With this calibration, mountain East
  ≈ [−2019, +1182] m → max `z_cont ≈ 5.9` (only layers 0–6 reachable).

**Before any real v5 run, update the notebook's `EAST_ENTRY` / `LAYER_EAST_DX` cell to
`1500 / 150`** (and update the "Key Gotchas" layer-accessibility note accordingly).

---

## Folder Layout

```
TambOpt/detector_optimization_v5/
├── CLAUDE.md                         ← this file
├── SWGOLO7_optimization_ev.ipynb     ← main evolutionary optimization notebook
├── modules_v5/
│   ├── __init__.py                   ← adds v3 + v4 to sys.path
│   ├── ev_deepsets.py                ← DeepSetsReconstruction (perm-invariant NN)
│   ├── ev_population.py              ← Population dataclass + build_input_batch
│   └── ev_selection.py               ← compute_detector_fitness, prune_weakest, mutate_positions
├── tests/
│   └── test_v5_modules.ipynb         ← tests (no GPU needed)
└── outputs/
    └── NN_Files_EV_v5/
        ├── inputs.pt / labels.pt / ...
        ├── model_weights.pth
        ├── checkpoint.pth
        └── Python_Layout/
            ├── Layout_0.txt          ← 10000 rows (initial)
            ├── Layout_1.txt          ← fewer rows (gen 1)
            ├── ...
            ├── Layout_N.txt          ← 90 rows (final)
            ├── Utilities.txt
            └── layout_evolution_3d.gif
```

**No verbatim copies.** All reusable code is imported from v3/v4 via sys.path injection
in `modules_v5/__init__.py`. v5 only ships the three new modules above.

---

## v3/v4 Imports (via sys.path injection)

```python
import modules_v5   # triggers sys.path injection for v3 + v4

from modules.generate_showers   import GenerateShowers
from modules.shower_computation  import ComputeShowerDetection
from modules.detector_response   import SmearN, TimeAverage_vectorized
from modules.reconstruction      import NormalizeLabels, DenormalizeLabels, EarlyStopping
from modules.utility_functions   import reconstructability, U_PR, U_E, U_angle

from modules_v4.tr_geometry      import load_tr_mountain
from modules_v4.tr_surface_map   import SurfaceEastMap
from modules_v4.tr_plane_kernel  import GetCounts_planeaware

# v5 only
from modules_v5.ev_deepsets      import DeepSetsReconstruction
from modules_v5.ev_population    import Population, build_input_batch
from modules_v5.ev_selection     import compute_detector_fitness, prune_weakest, mutate_positions
```

**Not used from v3/v4:**
- `Reconstruction` — replaced by `DeepSetsReconstruction`.
- `LearnableXY` — no gradient-based position optimization in v5.
- `Layouts` / `push_apart` / `symmetry_loss` — not applicable.

---

## v5 New Modules

### `ev_deepsets.py` — `DeepSetsReconstruction`

Permutation-invariant set-based NN:
```
phi  : Linear(7, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 64)
pool : sum over detectors → (B, 64)
rho  : Linear(64, 128) → ReLU → Dropout(0.1) → Linear(128, 32) → ReLU → Linear(32, 3) → Tanh
```

- `forward(inputs, mask=None)`: inputs (B, N, 7), mask (B, N) or (1, N), returns (B, 3).
- `mask` is the hook for gradient saliency: passing `requires_grad=True` mask and back-
  propping U lets us read `∂U/∂mask[i]` as detector i's marginal contribution.
- Trained with **random mask-dropout** (10–90% of detectors zeroed per batch) so the model
  learns robust per-detector representations across the full 10k → 90 range.

### `ev_population.py` — `Population`

Dataclass holding `(x, y, z_cont)` tensors + references to `MountainData` and `SurfaceEastMap`.

Key methods:
- `Population.initial(mountain, surface, n_units=10000)` — dense grid sampling.
- `apply_indices(keep_idx)` — in-place pruning.
- `refresh_z_cont()` — recompute z_cont after position changes.
- `save_layout(path)` / `load_layout(path, ...)` — 3-column (N, Up, z_cont) text files.

`build_input_batch(population, N_list, T_list, X0, Y0)` → (B, N_det, 7) feature tensor.

### `ev_selection.py` — Evolutionary operators

- `compute_detector_fitness(population, model, shower_fn, ...)` → `(fitness, U_value)`.
  One forward + backward pass through DeepSets with a per-detector mask gate.
  Returns `∂U/∂mask` as the per-detector fitness score.
- `prune_weakest(population, fitness, target_size)` — keep top-k by score, in-place.
- `mutate_positions(population, sigma, frac)` — Gaussian perturbation + mountain reprojection.

---

## Evolutionary Algorithm (per-generation pseudocode)

```python
# Schedule: geometric decay from 10000 → 90 over ~30 generations
schedule = np.geomspace(10000, 90, n_generations + 1).round().astype(int)

for gen in range(n_generations):
    target = schedule[gen + 1]

    # 1. Score every detector via gradient saliency
    fitness, U = compute_detector_fitness(population, model, shower_fn, ...)

    # 2. Prune weakest detectors
    prune_weakest(population, fitness, target_size=target)

    # 3. Mutate survivors (Gaussian noise + mountain reprojection)
    mutate_positions(population, sigma=sigma_schedule[gen], frac=0.5)

    # 4. Fine-tune NN every 5 generations
    if (gen + 1) % 5 == 0:
        fine_tune_deepsets(model, population, ...)

    # 5. Save layout + checkpoint
    population.save_layout(f"Layout_{gen+1}.txt")
```

---

## NN Feature Vector (identical to v4)

7 features per detector: `[x=N, y=Up, z=z_cont, N_int, T_int, x0, y0]`

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `x = N` | Detector North coordinate [m] |
| 1 | `y = Up` | Detector Up (elevation) coordinate [m] |
| 2 | `z_cont` | Continuous plane index ∈ [0, 23] |
| 3 | `N_int` | Energy-weighted plane-interpolated kernel integral from `GetCounts_planeaware`. **Not actually smeared** — `SmearN_fn` is accepted as a kwarg for v3 interface compatibility but is never called inside v4's kernel |
| 4 | `T_int` | Plane-weighted arrival time from `GetCounts_planeaware` — `(point_t · kernel).mean(dim=1)` in the current implementation (unweighted per-kernel mean; the energy-weighted form is commented out) |
| 5 | `x0` | Energy-weighted shower core North / 5000 |
| 6 | `y0` | Energy-weighted shower core Up / 5000 |

`reconstructability` is called on `inputs[:, :, 3]` (N_int at index 3) — same as v4.

---

## Key Gotchas

1. **Julia 1-indexing**: `faces` and `detector1` in the HDF5 are 1-indexed — subtract 1.
2. **Layout files have variable row count**: 10,000 rows initially → 90 at the end.
   `np.loadtxt` handles this; visualization code must not hard-code row counts.
3. **`mask.grad` detaches if you re-assign `mask`**: always back-prop with the same
   tensor instance that was passed to `model(inputs, mask=...)`.
4. **`mountain.project_to_mountain`** uses a default `max_gap` heuristic (2× mean NN
   distance). For large mutations (`sigma > 200 m`), you may need a custom `max_gap`.
5. **DeepSets pooling is `sum`**: `input_mean/std` normalization must be applied BEFORE
   passing to the model (on the raw (B, N, 7) tensor), not on the pooled embedding.
6. **`reconstructability` index**: N_int is at feature index 3 (same as v4, not 2 as v3).
7. **`filter_plane=None`**: must not pass `filter_plane=20` to `ComputeShowerDetection`.
8. **Y-shift**: AllShowers samples must be vertically recentered to sit inside the mountain
   bbox, using the same lazy `_apply_y_shift` pattern as v4.
9. **NN mask-dropout training**: DeepSets must be trained with random mask dropout so that
   gradient saliency is meaningful across the full 10k→90 range.  Without this, the NN
   over-relies on having all 10k detectors active and saliency degrades after heavy pruning.
10. **Layer accessibility is calibration-dependent.** With the correct `EAST_ENTRY=1500,
    LAYER_EAST_DX=150` calibration (used by v4's active scripts and v6), `z_cont` spans
    ≈ [2.1, 23.5] so **all 24 AllShowers layers are reachable**. With the stale
    `−212 / 307` values still wired into this file and the v5 notebook, mountain East
    ≈ [−2019, +1182] m → max `z_cont ≈ 5.9` (only layers 0–6).

---

## Quick References

- v4 source: `TambOpt/detector_optimization_v4/`
- v3 source: `TambOpt/detector_optimization_v3/`
- AllShowers framework: `/n/home05/zdimitrov/tambo/TAMBO-opt/allshowers/`
- Geometry HDF5: `TAMBOSim/resources/basic_geometry.h5`
- v5 tests (no GPU): `tests/test_v5_modules.ipynb`
