# detector_optimization_v5

Fifth iteration of the TAMBO detector-layout optimizer. The big shift from v4 is the **optimizer**: v5 replaces gradient-based SGD on detector positions with an **evolutionary pruning algorithm** operating on a **DeepSets** (permutation-invariant, set-based) reconstruction NN. Detectors are no longer `nn.Parameter`s — they live in a `Population` dataclass whose size shrinks geometrically from 10,000 → 90 over ~30 generations. 

**None of this code has ever been used, it was generated as LLM boilerplate to enable a paralel approach.**


Three structural pieces carry the EA:

- `Population` — variable-size container `(x, y, z_cont)` + refs to v4's `MountainData` / `SurfaceEastMap`. Supports `apply_indices` (prune), `refresh_z_cont` (after mutation), `save_layout` / `load_layout`.
- `DeepSetsReconstruction` — `phi: Linear(F, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 64)` applied per-detector, then `sum`-pool across detectors, then `rho: Linear(64, 128) → ReLU → Dropout(0.1) → Linear(128, 32) → ReLU → Linear(32, 3) → Tanh`. Accepts an optional **mask** tensor that gates the per-detector embedding multiplicatively — the gradient of `U` w.r.t. that mask is the per-detector fitness signal.
- `compute_detector_fitness` / `prune_weakest` / `mutate_positions` — one-pass gradient-saliency scoring, top-k pruning by raw fitness, and Gaussian perturbation with mountain reprojection.

**Status**: scaffolding committed (`4c25ef4`, "v5 boilerplate") — no production runs yet. See `../VERSIONS.md`.

End-to-end flow (per generation):

```
Population (10k → 90 by schedule[gen])  ──▶  generate showers at current positions
                                                        │
                                                        ▼
                                              build_input_batch → (B, N, 7)
                                                        │
                                                        ▼
            DeepSetsReconstruction(inputs, mask=ones(1,N), requires_grad)
                                                        │
                                                        ▼
                                   U = 1e2·U_θ + 1e2·U_φ + 1e8·U_E + 5e5·U_PR
                                                (divided by w_norm = 1e3)
                                                        │
                                                        ▼
                              U.backward() → mask.grad  ──▶  fitness_scores (N,)
                                                        │
                                                        ▼
                       prune_weakest(…, target_size=schedule[gen+1])   (top-k by raw score)
                                                        │
                                                        ▼
                    mutate_positions(sigma=sigma_schedule[gen], frac=0.5)  (+ reproject)
                                                        │
                                                        ▼
                  every 5 generations: fine-tune DeepSets with random mask dropout
```

For the broader v1→v6 history, see `../VERSIONS.md`.

---

## Modules (`modules_v5/`)

v5 only ships three new modules. Everything else is imported from v3 *and* v4 via `sys.path` injection in `modules_v5/__init__.py`.

| File | Public API | Purpose |
|------|-----------|---------|
| `ev_deepsets.py` | `DeepSetsReconstruction` | Permutation-invariant set NN. `forward(inputs: (B, N, F), mask=None)` returns `(B, 3)`. `mask` is the saliency hook: pass `mask.requires_grad_(True)` and `mask.grad` after backward is the per-detector fitness |
| `ev_population.py` | `Population` (dataclass), `build_input_batch` | `Population.initial(mountain, surface, n_units=10000, scheme="grid")`, `apply_indices`, `refresh_z_cont`, `save_layout` / `load_layout` (3-col text). `build_input_batch` produces the v4-identical 7-feature stack |
| `ev_selection.py` | `compute_detector_fitness`, `prune_weakest`, `mutate_positions` | Fitness: one forward+backward through DeepSets with a `(1, N)` mask gate; returns `(fitness_scores, U_value)`. Pruning: top-k by *raw* saliency (not absolute value). Mutation: Gaussian perturbation + `mountain.project_to_mountain` reprojection |

**Inherited from v3 + v4** (via `modules_v5.__init__` injecting `../detector_optimization_v3` *and* `../detector_optimization_v4` on `sys.path`):

| Upstream module | Usage in v5 |
|-----------------|-------------|
| `modules.generate_showers.GenerateShowers` (v3) | Unchanged |
| `modules.shower_computation.ComputeShowerDetection` (v3) | Called with `filter_plane=None` |
| `modules.detector_response.{SmearN, TimeAverage_vectorized}` (v3) | Interface callables only; not invoked inside `GetCounts_planeaware` |
| `modules.reconstruction.{NormalizeLabels, DenormalizeLabels, EarlyStopping}` (v3) | Unchanged. **`Reconstruction` is NOT used** — replaced by `DeepSetsReconstruction` |
| `modules.utility_functions.{reconstructability, U_PR, U_E, U_angle}` (v3) | Unchanged |
| `modules_v4.tr_geometry.load_tr_mountain`, `MountainData` (v4) | Unchanged |
| `modules_v4.tr_surface_map.SurfaceEastMap` (v4) | Unchanged |
| `modules_v4.tr_plane_kernel.GetCounts_planeaware` (v4) | Unchanged |
| `modules.layout_optimization.LearnableXY` (v3) | **Not used** — no gradient-based position optimization in v5 |
| `modules.geometry.Layouts`, `push_apart`, `symmetry_loss` (v3) | **Not used** |

---

## Contents

```
detector_optimization_v5/
├── CLAUDE.md                                    # Session memory: design, gotchas, coord convention
├── modules_v5/                                  # v5-only modules
│   ├── __init__.py                              #   Injects v3 + v4 into sys.path
│   ├── ev_deepsets.py                           #   DeepSetsReconstruction (perm-invariant NN)
│   ├── ev_population.py                         #   Population dataclass + build_input_batch
│   └── ev_selection.py                          #   Fitness / prune / mutate operators
│
├── SWGOLO7_optimization_ev.ipynb                # Main evolutionary optimization notebook
│
├── tests/
│   └── test_v5_modules.ipynb                    # Module sanity suite (no GPU)
│
└── outputs/                                     # Populated at runtime
    └── NN_Files_EV_v5/
        ├── inputs.pt / labels.pt / ...
        ├── model_weights.pth
        ├── checkpoint.pth
        └── Python_Layout/
            ├── Layout_0.txt                     # 10000 rows (initial)
            ├── Layout_1.txt                     # fewer rows per generation
            ├── ...
            ├── Layout_N.txt                     # 90 rows (final)
            ├── Utilities.txt
            └── layout_evolution_3d.gif
```

---

## Coordinate Convention

| Symbol | Meaning | Source |
|--------|---------|--------|
| `x` | ENU North [m] | Stored in `Population` (float32 tensor, not `nn.Parameter`) |
| `y` | ENU Up / elevation [m] | Stored in `Population` |
| `z_cont` | `(EAST_ENTRY − East(x, y)) / LAYER_EAST_DX`, continuous AllShowers layer index | Derived from `surface(x, y)`; cached on `Population.z_cont`; refreshed via `refresh_z_cont()` after any position change |

**v5 notebook uses the OLD calibration**: `EAST_ENTRY = −212 m`, `LAYER_EAST_DX = 307 m` (both `CLAUDE.md` and `SWGOLO7_optimization_ev.ipynb` set these). That gives a reachable range of only AllShowers layers 0–6 (mountain East ≈ [−2019, +1182] m → max `z_cont ≈ 5.9`).

Per user feedback (2026-04-14), these values are wrong — runs made with them "sampled everything from the last plane and the mountain was mismatched". v4's active scripts and v6 use `1500 / 150` instead. **Update the notebook's `EAST_ENTRY` / `LAYER_EAST_DX` cell before a real v5 run.**

---

## v4 → v5 Differences

| Aspect | v4 | v5 |
|--------|----|----|
| Optimizer | SGD on `nn.Parameter`s (active script: `lr=0.5, mom=0.3`) | Evolutionary pruning + mutation (no gradient on positions) |
| Detector count | Fixed 90 | Variable: **10,000 → 90** over `n_generations = 30` (geometric schedule) |
| Learnable params | `LearnableXY(N, Up)` as `nn.Parameter` | None — positions live in `Population.x`, `Population.y` plain tensors |
| NN architecture | v3's `Reconstruction` (flat FC, fixed `num_detectors × 7 = 630` input) | `DeepSetsReconstruction` (shared per-det MLP + `sum` pool + decoder, variable N) |
| NN input shape | `(B, 90·F)` flattened | `(B, N, F)` — `N` varies per generation |
| Fitness signal | `∂Loss/∂(N, Up)` via chain rule through `z_cont → East → surface` | `∂U/∂mask` via gradient saliency on a per-detector multiplicative gate |
| Selection | N/A (all 90 survive every epoch) | Top-k by `fitness.topk(target_size, largest=True)`; detectors with **negative** saliency are pruned first |
| Position updates | SGD step on gradient | Gaussian perturbation (`mutate_positions(sigma=…, frac=0.5)`) + `mountain.project_to_mountain` reprojection |
| Layout file rows | Always 90 | Variable (10,000 → 90) |
| NN fine-tune cadence | Every 5 SGD epochs | Every 5 EA generations |
| NN input features | 5 (active): `[x, y, z_cont, N_int, T_int]` | **7**: `[x, y, z_cont, N_int, T_int, x0, y0]` (same order as v4's commented-out 7-feature variant) |

---

## NN Feature Vector

7 features per detector, matching v4's commented-out 7-feature layout exactly so a frozen `input_mean` / `input_std` computed on v4-style data is directly applicable.

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `x = N` | Detector North coordinate [m] |
| 1 | `y = Up` | Detector Up (elevation) coordinate [m] |
| 2 | `z_cont` | Continuous plane index |
| 3 | `N_int` | Energy-weighted kernel integral from `GetCounts_planeaware` |
| 4 | `T_int` | Plane-weighted arrival time from `GetCounts_planeaware` |
| 5 | `x0` | Energy-weighted shower core North / `core_scale` (default 5000) |
| 6 | `y0` | Energy-weighted shower core Up / `core_scale` |

`reconstructability` reads `N_int` at feature index 3 (`inputs[:, :, 3]`) — same convention as v4. **Normalize inputs outside the model** (frozen mean/std), not inside `phi` — `sum`-pool would otherwise mix in per-batch statistics.

---

## DeepSets Architecture

```
inputs : (B, N, 7)
  │
  ▼ phi  (shared per-detector MLP, applied to every (B, i, :) slice)
     Linear(7, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 64)
  │
emb    : (B, N, 64)
  │   (optional elementwise mask gate: emb = emb * mask.unsqueeze(-1))
  ▼ sum over dim=1   ← permutation-invariant
pooled : (B, 64)
  │
  ▼ rho  (post-pool decoder)
     Linear(64, 128) → ReLU → Dropout(0.1) → Linear(128, 32) → ReLU → Linear(32, 3) → Tanh
  │
output : (B, 3)   = [Ê_norm, θ̂_norm, φ̂_norm]   ∈ [−1, 1]
```

**Why it's detector-count-agnostic**: `phi`'s first `Linear` is `(input_features, hidden_dim)`, never `(N × input_features, hidden_dim)`. The same weights evaluate `N = 10000` and `N = 90` identically.

**The mask hook** — passing `mask = torch.ones(1, N, requires_grad=True)` and back-propping `U` gives `mask.grad[0, i]` ≈ a first-order approximation of the leave-one-out utility change for detector `i`. That gradient IS the fitness score used by `prune_weakest`.

---

## Evolutionary Loop (per-generation pseudocode)

```python
# Geometric schedule: 10000 → 90 over n_generations steps, strictly decreasing
schedule = np.round(np.geomspace(N_units_init, N_units_final, n_generations + 1)).astype(int)
schedule[0]  = N_units_init
schedule[-1] = N_units_final
# (fix-up pass ensures monotonic decrease by at least 1)

# Mutation sigma decays linearly from coarse to fine:
sigma_mut_schedule = np.linspace(80.0, 10.0, n_generations)

population = Population.initial(mountain, surface, n_units=N_units_init, device=device)

for gen in range(n_generations):
    target    = int(schedule[gen + 1])
    sigma_mut = float(sigma_mut_schedule[gen])

    # 1. Score every detector via gradient saliency
    fitness_scores, U_now = compute_detector_fitness(
        population, model, shower_fn,
        n_samples=Nbatch,
        input_mean=input_mean, input_std=input_std,
        reconstruct_threshold=10.0,
        w_angle=1e2, w_energy=1e8, w_pr=5e5, w_norm=1e3,
    )

    # 2. Prune: keep top-k by raw saliency (negatives go first)
    prune_weakest(population, fitness_scores, target_size=target)

    # 3. Mutate survivors + reproject to the mountain surface
    mutate_positions(population, sigma=sigma_mut, frac=0.5)
    # (population.refresh_z_cont() is called inside mutate_positions)

    # 4. Fine-tune DeepSets every 5 generations (with random mask dropout)
    if (gen + 1) % 5 == 0:
        fine_tune_deepsets(model, population, ...)

    # 5. Persist
    population.save_layout(f"Layout_{gen+1}.txt")
```

The utility inside `compute_detector_fitness` is built from v3/v4's primitives:

```
U = (w_angle · U_angle(θ̂, θ, r) + w_angle · U_angle(φ̂, φ, r) +
     w_energy · U_E(Ê, E, r) + w_pr · U_PR(r)) / w_norm
```

with defaults `w_angle = 1e2`, `w_energy = 1e8`, `w_pr = 5e5`, `w_norm = 1e3`, and `r = reconstructability(inputs[:, :, 3], reconstruct_threshold=10.0)`.

---

## Key Gotchas

1. **Julia 1-indexing** on `faces` / `detector1` — subtract 1 before Python indexing (inherited from v4).
2. **Layout files have variable row count** (10,000 → 90). `np.loadtxt` handles this naturally; any visualization code must not hard-code row counts.
3. **`mask.grad` detaches on reassignment** — always back-prop with the same tensor instance you passed to `model(inputs, mask=...)`.
4. **`mountain.project_to_mountain`** uses `max_gap = 2× mean nearest-neighbour distance` by default. For `sigma_mut ≳ 200 m`, pass an explicit `max_gap` — otherwise survivors that wander far off the surface get snapped onto a distant centroid.
5. **DeepSets pool is `sum`** — `input_mean` / `input_std` normalization must be applied to the raw `(B, N, 7)` tensor BEFORE the model, not on the pooled `(B, 64)` embedding.
6. **`reconstructability` index**: `N_int` is at feature index **3** (matches v4; v3 had it at 2).
7. **`filter_plane=None`** — never pass `filter_plane=20` to `ComputeShowerDetection` in v5 (same rule as v4).
8. **Y-shift of AllShowers samples** — point clouds must be vertically recentered to sit inside the mountain bbox, using the same lazy `_apply_y_shift` pattern as v4.
9. **Train DeepSets with random mask dropout** (e.g. 10–90% detectors zeroed per batch) so that gradient saliency stays meaningful across the entire 10k→90 range. Without this, the model over-relies on having all 10k active and fitness scores degrade as pruning proceeds.
10. **Layer accessibility is calibration-dependent** — with the v5 notebook's `−212 / 307` (the calibration per the user, this is **wrong**), max `z_cont ≈ 5.9` so only layers 0–6 are accessible; with the v6-style `1500 / 150` all 24 layers are reachable. Fix this before a real run.

---

## Stale-Note Warning

`CLAUDE.md` and `SWGOLO7_optimization_ev.ipynb` both set `EAST_ENTRY = −212 m, LAYER_EAST_DX = 307 m`. Per user feedback (2026-04-14) these values are wrong — runs with them sample all energy from the last plane. v4's active scripts and v6 use `EAST_ENTRY = 1500, LAYER_EAST_DX = 150`. Update the notebook cell (and ideally `CLAUDE.md`) before a new session uses either as source of truth.

Separately, `CLAUDE.md` notes v4's baseline optimizer as `SGD(lr=1, momentum=0.3)`; the v4 active script actually runs `SGD(lr=0.5, momentum=0.3)`. Minor doc-fix, not a functional issue.

---

## Relation to Other Pipelines

- `../detector_optimization/` — v1, monolithic.
- `../detector_optimization_v2/` — modular refactor, flat 2D, diffusion-image surrogate.
- `../detector_optimization_v3/` — AllShowers point-cloud surrogate, single plane (`filter_plane=20`). Imported wholesale.
- `../detector_optimization_v4/` — full 3D mountain surface, differentiable surface map, plane-aware kernel. Imported wholesale.
- `../detector_optimization_v6/` — staged pipeline with two frozen NN surrogates (data-gen → FNN → recon → optimize).
- `../VERSIONS.md` — cross-version history and known issues.
