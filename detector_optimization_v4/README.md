# TambOpt detector-optimization — one-month history (2026-03-18 → 2026-04-14)

Chronological record of what was built, what was tested, and what was observed
across `detector_optimization_v2` → `v3` → `v4` → `v5` during this window. Not
a roadmap; a log. All file paths are relative to the `TambOpt/` root unless
stated otherwise.

---

## Summary of the four-week arc

```
v2 (extant)    ──►  v3 (Apr 3)    ──►  v4 (Apr 8)   ──►  v5 (Apr 14)
flat 2D        tune utility +     full 3D Colca    evolutionary
+ TAMBO        swap to            mountain surf.   pruning, no
physics        point-cloud        + differentiable gradient on
(legacy)       diffusion model    plane kernel     positions
               + single-plane     (v3 style of     + DeepSets NN
               filter_plane=20    SGD on (N,Up))
```

Each step reused the previous version's modules via `sys.path` injection —
no verbatim copies.

---

## Phase 1 — v2 tuning and NN-input plumbing (Mar 18 → Apr 2)

All work in `detector_optimization_v2/`. Focus: get a working, stable gradient
pipeline before migrating to the point-cloud shower generator.

**Concepts tested**

| Date   | Commit(s)       | What was tried |
|--------|-----------------|----------------|
| Mar 18 | `853c4c0` `cfa56c4` `996a5df` `fb42485` | Enable label denormalization; border padding on the detector response; runs with shifted normalized data (0–1 range) at 20k and 200k; plots added. Marked "runs successful". |
| Mar 19 | `cc6b155` `982f8f5` `8381530` `de68dfb` `6936875` `f1f69fe` `2dd28a7` `459ca74` `1e1e24f` `e636488` | Raise `reconstruct_threshold`; reduce random-noise amplitude; custom offset for initial detector array; shrink the initial array; updated hyperparams for 20k; review old runs; refactor + README. |
| Mar 20 | `0bb581b` `111da02` | Hyperparameter sweep; remove `torch.no_grad` from the validation pass (so validation forward could produce gradients when wanted). |
| Mar 22 | `90b1204` `5bd4fb4` `b97a599` `343a211` | Conditional logging; `torch.pi`; remove `torch.no_grad` in shower generation; reconstruction parameter tweaks. |
| Mar 25 | `c2f8e2f` `5850052` | **Restructure NN input**: move `x0, y0` (energy-weighted shower core) into the per-detector feature vector, normalized inputs, compute NN on device. Outputs notebooks for "full gradients across network", "xy in inputs", "remove density", "normalized input", "new coeff". |
| Mar 27 | `8d77d00`       | Big refactor: everything on GPU, remove per-detector smearing, generate detector layouts with torch, conditional data generation, utility-function update. |
| Mar 30 | `ac4e996`       | New coefficients + renormalization, extensive 100k output notebooks (new coeff, small steps, smaller LR, no push-apart, from-center init). First experiments contrasting ring vs center init and with/without `push_apart`. |
| Apr 2  | `53cd4f4` `ec6b91a` `0096b88` `25ffc18` `aecc8d3` | **Shower cache**; side-by-side plots of raw showers vs diffusion-model showers; NN training without `push_apart` and with fewer detectors; "optimize only for most particles detected" variant; bash job scripts. |

**Observable outcomes from v2**
- A working gradient-based optimizer on the flat 2D plane geometry existed at
  this stage, iterated through ~20 distinct output notebooks. The last known
  v2 outputs (Apr 1–2) vary detector count, push-apart on/off, and NN
  capacity.
- The utility function was unstable: multiple commits "update hyperparameters",
  "new coefficients", "renormalization". Magnitudes of `U_angle`, `U_E`, `U_PR`
  were repeatedly rescaled.

---

## Phase 2 — v3: swap to point-cloud flow-matching, single plane (Apr 3 → Apr 6)

All work in `detector_optimization_v3/`. Focus: migrate the shower generator
from v2's legacy TAMBO-physics sampler to the AllShowers **point-cloud
flow-matching model**, then tune the NN and kernel to work with point clouds.

**Concepts tested**

| Date  | Commit(s)       | What was tried |
|-------|-----------------|----------------|
| Apr 3 | `a8b7269` `ff2f821` | Create `detector_optimization_v3/` folder. Move scripts to operate on point-cloud `(B, max_points, 5)` tensors `[x, y, layer_index, energy, time]`. First tests of a **Gaussian kernel over the point cloud** (no layer filtering yet). Remove unused files from v2. |
| Apr 5 | `cdae855` `fae3138` | Fix energy normalization; finalize full utility function and tests; first-run outputs `20260404_040000_first_run` and `20260404_0430000_small_radius_init`. |
| Apr 6 | `9604fb7` `cb91b57` `d821b8f` `e930395` | Clean up pylance errors; clear notebook; **add `filter_plane=20`** so all shower points outside the target plane have their energy zeroed before the kernel — effectively reduces the point-cloud kernel to a 2D spatial Gaussian over one layer. "Single plane, start small" output: `20260406_080000_plane_20_small_start.ipynb`. |

**Outcome**
- v3 is the **last known-good** gradient-based position optimizer. Detectors
  move meaningfully, utility climbs. Feature vector: 6 features
  `[x, y, N_int, T_int, x0, y0]`; `SGD(lr=10, momentum=0.3)`; utility weights
  `(1e2·Uθ + 1e2·Uφ + 1e3·U_E + 5e5·U_PR)/1e3`.
- Stable because `filter_plane=20` ensures every detector integrates over the
  same (dense) plane-20 point cloud — the kernel is never starved of points.

---

## Phase 3 — v4: full 3D Colca mountain surface with a plane-aware kernel (Apr 8 → Apr 14)

All work in `detector_optimization_v4/`. Focus: move detectors from a flat 2D
plane onto the curved Colca Valley wall, keep gradients flowing through a
differentiable surface map and a plane-aware kernel.

**Structural additions**

- `modules_v4/tr_geometry.py` — HDF5 loader for `basic_geometry.h5`, ECEF→ENU
  rotation, `MountainData` dataclass, `sample_initial_layout(scheme="grid"|"random"|"center")`,
  `project_to_mountain`. `z_cont = (EAST_ENTRY − East) / LAYER_EAST_DX`.
- `modules_v4/tr_surface_map.py` — `SurfaceEastMap`: `LinearNDInterpolator` on
  2161 centroid scatter → 256×256 regular grid → `F.grid_sample(bilinear,
  padding_mode="border")`. Differentiable `East = f(N, Up)`.
- `modules_v4/tr_plane_kernel.py` — `GetCounts_planeaware`: the v3 spatial
  Gaussian × a new **triangular plane weight** `relu(1 − |layer_p − z_cont|)`.
  Differentiable in `z_cont`, reduces to v3 when `z_cont ≡ 20`.

**Commits and experiments**

| Date    | Commit(s)       | What was tried |
|---------|-----------------|----------------|
| Apr 8   | `2900943`       | Initial v4 import: geometry loader, surface map, plane-aware kernel, CLAUDE.md, test notebook, main optimization notebook. |
| Apr 8   | `6952dd6`       | First v4 run: `SWGOLO7_optimization_tr_output_20260408_160000.ipynb`. |
| Apr 8   | `31ef49d`       | Save clamped-space run: `…_20260408_080000_clamped_detector_space.ipynb` — explored whether clamping positions into a rectangular bbox helps stability. |
| Apr 8   | `1e64fc6`       | Save the v3→v4 migration plan (`stateful-beaming-pie.md`) — 1090-line detailed plan document. |
| Apr 8   | `dc5fd7f`       | Refactor; rename `output_notebooks/` → `outputs_notebooks/`. |
| Apr 8   | `2141e32`       | Add SLURM batch scripts and `auto_run_notebook.py` for overnight 200k runs. |
| Apr 13  | `f3e3838`       | First utility update. |
| Apr 13  | `0af17aa`       | Per-event energy output. |
| Apr 13  | `23b1e5d`       | Add **`scheme="center"`** initial layout (cluster of 90 detectors around the mountain centroid). |
| Apr 13  | `01901f5`       | Decrease LR; save utility more often. |
| Apr 13  | `d0d425a`       | 200k results with performance plots. |
| Apr 13  | `4fcecaf`       | Re-run 200k training data generation. |
| Apr 13  | `cd7581f`       | **Test different utility functions on 10 showers**: `angle_error`, `theta_error`, `phi_error`, `angle_energy`, at Adam lr=0.3 and lr=1. Nine output notebooks in a single commit. Also adds `gradient_path_analysis.md`. |
| Apr 14  | `9798033`       | Save new notebook version + Python export (`SWGOLO7_optimization_tr_same_300_center_init.py`). |
| Apr 14  | `53da42d`       | Update utility (`detector_optimization_v3/modules/utility_functions.py`) and run "mean_u" variants of the 10-shower and 300-shower notebooks at Adam lr=1 and lr=3. |

**Things held constant vs. explicitly varied across v4 experiments**

- *Constant*: 90 detectors; 24 plane layers; Colca-Valley HDF5 geometry;
  `GetCounts_planeaware` (σ=200 m spatial Gaussian × triangular plane weight);
  differentiable surface map; `project_to_mountain` post-step.
- *Varied*: initial layout scheme (`grid` vs `center`); NN input size
  (5 features `[x, y, z_cont, N, T]` vs 7 features with `x0, y0`); utility
  composition (`U_θ + U_φ` vs `… + U_E` vs `… + U_E + U_PR`, with weights
  1e2/1e3/5e5 in various combinations, and a "mean_u" variant); optimizer
  (`SGD(lr=0.5, mom=0.3)` vs `Adam(lr=0.3/1/3)`); shower batch size (10 vs 300
  vs 20k vs 200k); `EAST_ENTRY` / `LAYER_EAST_DX` constants (two calibrations —
  see below).

**Documented finding — `gradient_path_analysis.md` (Apr 13)**

The autograd chain from `xy_module.x/.y` through `SurfaceEastMap` →
`GetCounts_planeaware` → NN → utility → loss is verified intact. `grad_norm`
is finite and non-zero every epoch. But the run stalls because of
*objective-shape* problems, not wiring:

1. `U_PR` is saturated: `reconstructability = sigmoid(5·(n − 10))` with
   `n ≈ 90` evaluates to ~1 for every event, so `U_PR = sqrt(90) ≈ 3.16`.
   Weight 5e5 makes it ~82% of `U_total` as a frozen constant; its derivative
   is ~0. **It adds no gradient.**
2. `U_E = Σ r / ((E_pred − E_true)² + 0.01)` is numerically zero because
   `DenormalizeLabels` returns GeV (1e5–1e8), so each term is ~1e-10 and the
   sum is rounded to 0 in the logs. **No gradient contribution.**
3. Only `U_θ` and `U_φ` drive learning. `U_θ` gains ~29% in the first 100
   epochs, then plateaus. `U_φ` swings wildly (1k → 30k → 1k) — SGD
   oscillation in a narrow, non-convex basin.
4. The NN fine-tune branch is a **silent no-op**: `DataLoader(ft_dataset,
   batch_size=32, …, drop_last=True)` on `Nfinetune=10` → zero batches per
   epoch.
5. Over 2000 epochs, mean per-detector displacement is 5.3 m, max 13.2 m;
   `z_cont` range never leaves `[11, 12.4]`. Detectors move tangent to the
   surface, not across it.

**EAST calibration note.** The CLAUDE.md in this folder says the empirical
AllShowers layer-East calibration is `EAST_ENTRY=−212, LAYER_EAST_DX=307`.
The script in the last commit uses `1500, 150`. Per the user (2026-04-14),
the `1500/150` values are the correct ones — with the old `−212/307` values
"all data was sampled from the last plane and the mountain was mismatched".
CLAUDE.md is out of date on this point.

---

## Phase 4 — v5: evolutionary pruning + DeepSets NN (Apr 14)

Committed as `4c25ef4` "v5 boilerplate". Scaffolding only — no run results yet.

**Concept.** Replace SGD-on-positions with a **mask-based evolutionary
pruning algorithm**:

- Start with 10,000 candidate detectors dense-sampled on the mountain surface.
- Train a **permutation-invariant DeepSets NN** (`phi: per-det MLP → sum
  pool → rho: readout MLP`) that handles variable detector counts natively.
- Each generation: compute per-detector fitness via `∂U/∂mask` (gradient
  saliency on a per-detector gate), prune the weakest, Gaussian-mutate the
  survivors, reproject to mountain surface, optionally fine-tune the NN.
- Geometric schedule: 10000 → 90 over ~30 generations.
- Train DeepSets with **random mask-dropout** so saliency is meaningful
  across the full 10k→90 range.

**Files added** (`detector_optimization_v5/`):
- `CLAUDE.md` (241 lines) — full design doc.
- `SWGOLO7_optimization_ev.ipynb` (869 lines) — main evolutionary notebook.
- `modules_v5/ev_deepsets.py` (113 lines) — `DeepSetsReconstruction`.
- `modules_v5/ev_population.py` (176 lines) — `Population` dataclass +
  `build_input_batch`.
- `modules_v5/ev_selection.py` (241 lines) — `compute_detector_fitness`,
  `prune_weakest`, `mutate_positions`.
- `tests/test_v5_modules.ipynb` (232 lines) — module sanity tests.

`modules_v5/__init__.py` injects `detector_optimization_v3` and `v4` into
`sys.path`, so v5 imports `Population`, `GetCounts_planeaware`,
`SurfaceEastMap`, `load_tr_mountain`, `NormalizeLabels`, `reconstructability`,
`U_PR`, `U_E`, `U_angle` directly from v3/v4 without copying.

**Note.** v5's CLAUDE.md still lists `EAST_ENTRY=−212, LAYER_EAST_DX=307` —
the older (and per the user, wrong) calibration. This should be updated to
`1500/150` before any v5 runs.

---

## Running list of experiments by filename pattern

Output notebooks in `detector_optimization_v4/outputs_notebooks/` decode as:

```
SWGOLO7_optimization_tr_<variant>_<timestamp>_<parameters>.ipynb
```

The April 13–14 variants encode (utility, optimizer, LR, shower count) in
their filenames:

| Filename fragment                | Meaning |
|---------------------------------|---------|
| `output_20260407_160000`        | First v4 run |
| `output_20260408_074341`        | (early iteration) |
| `output_20260408_080000_clamped_detector_space` | positions clamped to bbox |
| `output_20260408_081022_200k_data_generation_and_pretraining` | 200k-shower NN pretrain |
| `output_20260409_030000_200k_data_plots`         | plots of pretrain outputs |
| `output_20260409_030000_200k_retrain`            | 200k retrain; grad_norm printouts visible |
| `same_10_outputs_20260413_070000`                | 10-shower baseline |
| `same_10_center_init_20260413_100000`            | center init, 10 showers, baseline |
| `same_10_center_init_…_120000_angle_error_adam_lr1`   | only `U_angle` objective, Adam lr=1 |
| `same_10_center_init_…_120000_theta_error`       | only `U_θ` objective |
| `same_10_center_init_…_120000_phi_error`         | only `U_φ` objective |
| `same_10_center_init_…_123000_angle_error_adam_lr03` | `U_angle` only, Adam lr=0.3 |
| `same_10_center_init_…_123000_angle_energy_adam_lr03` | `U_angle + U_E`, Adam lr=0.3 |
| `same_300_center_init_…_123000_angle_energy_adam_lr03` | as above, 300 showers |
| `same_10_center_init_…_20260414_023000_angle_energy_adam_lr1_mean_u` | "mean_u" utility reformulation, Adam lr=1 |
| `same_10_center_init_…_030000_angle_energy_adam_lr3_mean_u` | as above, lr=3 |
| `same_300_center_init_…_20260414_023000_angle_energy_adam_lr1_mean_u` | 300-shower version |
| `same_300_center_init_…_030000_angle_energy_adam_lr3_mean_u` | 300-shower, lr=3 |

Interpretation: the Apr 13–14 campaign is a systematic sweep over
**utility subsets × optimizer LR × shower-batch size**, built on the "same
10/300 showers" cached fixture so gradients are deterministic given positions.

---

## Current state (2026-04-14)

- **v2** — frozen, superseded.
- **v3** — last version with provably-moving gradient-based optimization.
  Still imported by v4 and v5 for shower generation, reconstruction,
  utility, normalization, and early-stopping utilities.
- **v4** — gradient path confirmed alive but **objective-shape issues keep
  detectors nearly static** even on the working 10-shower fixture:
  saturated `U_PR`, numerically-dead `U_E`, silent-no-op fine-tune branch,
  NN over-fit to 10 shower instances. The Apr 13–14 sweep is trying
  different utility formulations ("angle_error", "theta_error", "phi_error",
  "angle_energy", "mean_u") and optimizers (Adam lr ∈ {0.3, 1, 3}) to find
  one that actually drives the positions.
- **v5** — boilerplate only. Not yet run. Rethinks the whole optimizer as
  evolutionary pruning with DeepSets, removing gradient-based position
  updates entirely.

---

## Known issues / action items surfaced by the past month's work

1. **`U_PR` saturates at `n ≫ reconstruct_threshold`.** Raise the threshold
   or drop the `5e5` coefficient. Currently contributes ~82% of `U_total`
   with zero gradient.
2. **`U_E` is numerically zero** at GeV scale. Use `(log10(E_pred) −
   log10(E_true))²` or normalize energies to O(1) before the reciprocal.
3. **`DataLoader(ft_dataset, …, drop_last=True)` on `Nfinetune=10`** yields
   zero batches — NN fine-tune never runs. Set `drop_last=False` or
   `batch_size=min(32, Nfinetune)`.
4. **NN over-fits to 10 shower instances**. Grow `Nevents`/`Nbatch` past 10
   before drawing conclusions about optimizer behaviour.
5. **v4 script uses 5 features** `[x, y, z_cont, N, T]`; the 7-feature
   version with `(x0, y0)` is commented out. The April 9 `200k_retrain`
   run used 7.
6. **CLAUDE.md files are stale** in two places: (a) v4's EAST calibration
   `−212/307` is wrong per the user; (b) v5's CLAUDE.md still cites the
   same wrong values.
7. **`gradient_path_analysis.md`** in `detector_optimization_v4/` is the
   authoritative note on the v4 stall — keep it up to date as the utility
   is rewritten.
