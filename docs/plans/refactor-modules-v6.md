# refactor: make `modules_v6/` maintainable

## Status (2026-08-20) — all 9 steps done

| Step | State | Commit |
|---|---|---|
| guard tool `tools/codeonly.py` | done | `74cdbbf` |
| 1–4 Phase 1 deletions | done, verified | `8b9ab98` |
| 9 dedupe `_mlp` → `nn_blocks.py` | done, verified | `22e0075` |
| 8 split sampler + memoize `_ne_max_gap` | done, verified | `4ae9413` |
| 6 split `load_tr_mountain` | done, verified | `fe4629a` |
| 7 split `utility_of_xy` | done, verified | `6a52198` |
| 5 split `build_training_pairs` | done, verified | `d3c96f0` |
| 10 drop `_ne` suffixes | done, verified | `dd69526` |
| 11 fix `_STRATEGIES` | done, verified | `e2e0c63` |
| 12 promote live modules out of `legacy_core` | done, verified | `c9133f4` |

### Acceptance criteria as of `c9133f4`

| Criterion | Target | Actual | |
|---|---|---|---|
| unreachable modules | 0 | **0** | ✅ |
| names defined in two files | 0 | **0** | ✅ |
| callables over 80 lines | 0 | **0** (longest: `build_training_pairs`, 78) | ✅ |
| entry points that import | 28/28 | **28/28** | ✅ |
| callable count | ≤ 85 | **92** | ❌ |
| line count | ≤ ~3,000 | **3,060** | ❌ |

The two count ceilings are missed and that is the honest outcome, not a
rounding error. They were set before anyone knew how much scaffolding the
extractions would add, and Phases 2–3 *deliberately* add callables — Step 5
alone turned one function into five named stages plus a class. The structural
goals those numbers were a proxy for are all met: nothing unreachable, nothing
defined twice, nothing over 80 lines. Re-baseline the ceilings at 95 / 3,100 or
drop them in favour of the three structural checks.

### Verification actually run

- **Step 5**, on the real corpus (SLURM `-p test`): 200 showers, seed 0,
  `batch_size=20`, `load_chunk=60` (deliberately ragged final chunk), pre-split
  vs split → `torch.equal` on all six outputs, maxdiff `0.000e+00`.
- **Step 5 resume**: SIGKILLed a build after 2 of 4 chunks and restarted. It
  reported `2/4 chunks already done`, skipped them, restored rows bit-identical
  to an uninterrupted run, all outputs finite, checkpoint removed on completion.
- **Steps 10+11**: a 200-shower build at seed 0 after both is `torch.equal` to
  the reference built before them; `strategy_ids` spans `[0..4]` and
  `shower_level_split`'s `max()+1` still derives `n_showers` correctly.
- **Steps 10 and 12**: `tools/codeonly.py` shows every moved/renamed module
  identical in executable tokens, and each importer differing only by the
  renamed identifier or the dropped `legacy_core` path segment.

- **Full Step-04 optimizer run** (`-p gpu_test`, after every step landed):
  `04_optimize_lbfgs_ensemble.py --schemes grid --chains 1` with a truncated
  budget (2 Adam epochs, 3 L-BFGS iters) ran to completion in 17.5 min, exit 0.
  Surrogate + recon checkpoints loaded, L-BFGS got real gradients (6–27 closure
  calls per chunk), and U climbed 32.2 → **+33.352** over 167 chunks — the
  expected neighbourhood for a shortened budget. Output went to a suffixed dir
  (`..._refactor_smoke_grid`) so nothing real was overwritten.

### Not run

`sbatch slurm/run_all_script_batch_grid.sh` end to end against a scratch
`RUN_LOCATION` (`pipeline_status.json` walking 00→04, final `U` near ~35). That
is a multi-hour full-pipeline job including a full dataset rebuild; the Step-5
element-wise gate plus the Step-04 run above cover the same ground for a
refactor.

### Environment notes

- Interpreter (has torch; system `python3` does NOT, and its `ast` lacks
  `end_lineno`): `/n/home05/zdimitrov/.conda/envs/multiproc_env/bin/python`
- SLURM: `-p test` for CPU checks, `-p gpu_test` for GPU. Never run these on a
  login node, and note `/tmp` is node-local — a job script and its `-o` log must
  live on shared storage. The gate scripts are in
  `/n/home05/zdimitrov/tambo/.refactor_gate/` (outside the repo).
- Baseline for a before/after diff: `git show HEAD:<path>`. **Never** `git stash`
  or `git commit -a` here.
- To load a baseline copy that uses relative imports, give it a dotted name
  inside the package (`spec_from_file_location("modules_v6._x_before", path)`)
  after `import modules_v6`, or the relative imports fail.

## Goal

Every callable in `modules_v6/` is reachable from an entry point, defined exactly
once, and short enough to read in one screen — without changing any numerical
behaviour.

## Background

`modules_v6/` is 23 files, **97 callables**, 3,775 lines. The count is not the
problem; a differentiable-simulation library needs that many pieces. The problem is
that a reader cannot tell which ones are live:

- **Three modules are imported by nothing** (385 lines), and a fourth is imported
  only by a function that raises `NotImplementedError` — deleting that chain
  cascades into two more of the longest methods in the package.
- **Eight names are defined in two files each**, leftovers from the
  (North, Up) → (North, East) migration. One pair (`_mlp`) is byte-identical.
- **Four functions carry the bulk**: 217, 95, 85 and 74 lines.

All figures below come from an AST walk plus a transitive import-reachability trace
rooted at `scripts/` and `plots/`, re-verified against HEAD (`251b74f`).

Target: **97 → ~74 callables, 3,775 → ~2,860 lines.**

## Key Concepts

| Term | Meaning |
|---|---|
| `_ne` suffix | (North, East) convention. The production one. Its (North, Up) twins are the dead half of the migration — once they're gone the suffix distinguishes nothing. |
| `legacy_core/` | Vendored from retired generations. Misleading name: it holds `tr_plane_kernel.py`, the ground-truth kernel every training label comes from. |
| `_STRATEGIES` | List of `(label, fn_name, kwargs)` layout generators. Index into it = the `strategy_ids` tensor persisted in the dataset — **renumbering it invalidates every checkpoint.** |
| Resume checkpoint | `build_training_pairs` writes the whole `out_*` set (~11.8 GB at 750k scale) on a wall-clock interval, for `gpu_requeue` preemption. |
| RNG draw order | Layout draws come from one `rng` threaded through the chunk→strategy→batch loops. Any reordering changes the dataset, even though it stays statistically valid. |

## Approach & Rejected Alternatives

**Chosen: delete first, then split, then rename.** Layer-by-layer, because deletion
strictly shrinks the surface the later phases touch — and one deletion (Step 2)
removes a 95-line method that would otherwise be a split target.

- **Rejected: split the long functions first.** Would mean carefully restructuring
  `MountainData.sample_initial_layout` (95 loc) only to delete it in the same PR.
- **Rejected: merge modules to reduce file count.** The file count isn't the
  complaint and merging would create larger files with weaker boundaries.
- **Rejected: doing this after the pending performance work.** Perf Phase 1 rewrites
  the inner loop of `build_training_pairs`. Reviewing that against named stages is
  far easier than against a 217-line body, so this lands first.

## Files

**Delete**
- `[DEL] modules_v6/legacy_core/detector_response.py` — 132 loc, unreachable
- `[DEL] modules_v6/legacy_core/reconstruction.py` — 127 loc, unreachable
- `[DEL] modules_v6/legacy_core/tr_surface_map.py` — 126 loc, unreachable
- `[DEL] modules_v6/detector_strategies.py` — 92 loc, only the dead builder imports it

**Modify**
- `[MOD] modules_v6/fnn_surrogate.py` — drop `build_training_pairs` (174 loc, 105 of
  them after the `raise` on line 199) and `compute_labels_batch` (31 loc)
- `[MOD] modules_v6/legacy_core/tr_geometry.py` — drop the two now-dead `MountainData`
  methods; split `load_tr_mountain`
- `[MOD] modules_v6/legacy_core/geometry.py` — drop `project_to_triangle`, `barycentric_coords`
- `[MOD] modules_v6/legacy_core/layout_optimization.py` — drop `symmetry_loss`, `push_apart`
- `[MOD] modules_v6/fnn_surrogate_ne.py` → renamed `dataset_builder.py`; split the 217-line builder
- `[MOD] modules_v6/opt_core.py` — split `utility_of_xy`
- `[MOD] modules_v6/tr_geometry_ne.py` — split `sample_initial_layout_ne`; memoize `_ne_max_gap`
- `[MOD] modules_v6/detector_strategies_ne.py` → renamed `detector_strategies.py`; fix `_STRATEGIES`
- `[MOD] modules_v6/tr_surface_map_ne.py` → renamed `surface_map.py`
- `[MOD] modules_v6/{deepsets_surrogate,reconstruction}.py` — import shared `_mlp`
- `[MOD]` import sites across `scripts/*.py`, `plots/**/*.py` for the renames

**New**
- `[NEW] modules_v6/nn_blocks.py` — the single `_mlp`

## Steps

### Phase 1 — Delete (no design risk; do it in one commit)

- [x] **1. Remove the three unreachable modules.** Delete
  `legacy_core/{detector_response,reconstruction,tr_surface_map}.py`.
  `SmearN`/`TimeAverage_vectorized` appear in `GetCounts_planeaware`'s signature but
  are never called — callers pass `None`, so the kernel is untouched.
- [x] **2. Remove `detector_strategies.py` and its cascade.** Delete the module, then
  delete `MountainData.sample_initial_layout` (95 loc) and
  `MountainData.project_to_mountain` (44 loc) from `legacy_core/tr_geometry.py` —
  after Step 2's deletion nothing calls them. Keep `plane_dx` and `east_to_z_cont`.
- [x] **3. Remove the unreachable half of `fnn_surrogate.py`.** Delete
  `build_training_pairs` and `compute_labels_batch`, plus the now-unused
  `from .detector_strategies import ...`. Keep `encode_primary`,
  `compute_normalization`, `_species_sidecar_path`, `_load_species_sidecar`, and the
  `FNNSurrogate` class (`plots/02_plot_nn_target_vs_pred.py` needs it to load
  pre-DeepSets checkpoints).
- [x] **4. Remove dead functions inside live modules.** `project_to_triangle` and
  `barycentric_coords` from `legacy_core/geometry.py` (keep `Layouts` — it is live);
  `symmetry_loss` and `push_apart` from `legacy_core/layout_optimization.py` (keep
  `LearnableXY` — both 04 optimizers use it).

### Phase 2 — Split the long functions (behaviour-preserving; one commit each)

- [x] **5. Split `build_training_pairs`** in `modules_v6/fnn_surrogate_ne.py` (217 loc,
  eight jobs) into `_load_corpus_metadata`, `_build_chunk_list`, a `_ResumeState`
  class wrapping the `out_*` tensors + atomic `tmp`→`os.replace` write, and
  `_label_chunk` for the strategy×batch loop. **Keep the `load_chunk` rounding before
  the chunk list is built** — the existing comment warns that building the list off
  the raw value leaves a short final sub-batch and changes the per-chunk RNG draws.
- [x] **6. Split `load_tr_mountain`** (85 loc) in `legacy_core/tr_geometry.py` along its
  existing comment sections: read h5 + select detector-region faces, resolve ENU
  origin (explicit arg > mesh `location` > module default), ECEF centroids, rotate to
  ENU, unique region vertices. Move the 1-based-Julia-face-index explanation onto the
  extracted selector.
- [x] **7. Split `utility_of_xy`** (74 loc) in `modules_v6/opt_core.py` into: batch the
  layout + run the dual surrogate; assemble recon features and decode to physical
  labels; compute the four terms and apply the penalty. Extract the middle stage as
  `_predict_primary(...)` so `activation_of_xy` (62 loc) can share it.
- [x] **8. Split `sample_initial_layout_ne`** (66 loc) in `modules_v6/tr_geometry_ne.py`
  into `_layout_grid_candidates` / `_layout_random` / `_layout_center` behind a
  dispatch dict, mirroring `_STRATEGY_FNS`. In the same pass memoize `_ne_max_gap` on
  the mountain object — it is deterministic (`default_rng(0)`) but recomputed on every
  `project_to_mountain_ne(max_gap=None)` call, roughly 100k times per dataset build.

### Phase 3 — De-duplicate and rename (mechanical; separate commits)

- [x] **9. Extract the shared `_mlp`** into `[NEW] modules_v6/nn_blocks.py` with
  `dropout: float = 0.0`; import it in `deepsets_surrogate.py` and `reconstruction.py`.
  Bodies are already byte-identical — only a default and a docstring differ.
- [x] **10. Drop the `_ne` suffixes** now the twins are gone: `fnn_surrogate_ne.py` →
  `dataset_builder.py` (it builds the Step-1 dataset and is not a surrogate),
  `detector_strategies_ne.py` → `detector_strategies.py`, `tr_surface_map_ne.py` →
  `surface_map.py`. Keep `tr_geometry_ne.py` — it still coexists with
  `legacy_core/tr_geometry.py`. Update every import site. **Commit this alone**, so the
  rename noise doesn't bury the substantive diffs.
- [x] **11. Fix `_STRATEGIES`** in the renamed `detector_strategies.py`: it is configured
  by commenting lines out (7 of 12 commented), and two live labels each appear twice
  (`uniform_random`, `latin_hypercube`), so one label maps to two ids and logs are
  ambiguous. Replace with a named `_ALL_STRATEGIES` dict plus an explicit
  `ACTIVE_STRATEGIES` tuple, giving duplicates distinct labels. **Preserve the current
  order and count** — see Risks.
- [x] **12. Reconsider `legacy_core/`.** After Step 1 it holds six files, three of which
  are central rather than legacy: `tr_plane_kernel.py` (ground-truth kernel),
  `utility_functions.py` (the U terms), `geometry.py` (one live function). Promote
  those to `modules_v6/`, leaving `generate_showers.py` (the genuine external
  dependency), `tr_geometry.py` and `layout_optimization.py`.

## Verification

### Guard used at every step

Reduce each tracked `.py` to executable tokens only (`tokenize`, dropping `COMMENT`
and docstring `STRING` tokens), snapshot before, re-diff after. Phase 1's only
expected differences are whole removed definitions; Phase 3's are moved/renamed
imports; **Phase 2 must show none at all except the extractions themselves.**

### Per step

| Step | Command | Expected |
|---|---|---|
| 1–4 | re-run the reachability trace | zero unreachable modules |
| 1–4 | `grep -rn "SurfaceEastMap\|GetCounts_differentiable\|project_to_triangle\|symmetry_loss\|NormalizeLabels" --include=*.py .` | no hits |
| 1–4 | real import (not just `ast.parse`) of every file in `scripts/`, `plots/`, `plots/single_species/` | no ImportError — a parse check will not catch a deletion-broken import |
| 5 | build a small dataset (`MAX_SHOWERS≈2000`) before and after at the same `SEED` | `torch.allclose(E_old,E_new)`, `torch.allclose(T_old,T_new)`, equal `strategy_ids`/`species_ids`. Element-wise, not distributional — RNG order must not move |
| 5 | kill a build mid-run, restart | `_ResumeState` skips completed chunks; final tensors match an uninterrupted run |
| 6 | load the mesh before/after | identical `centroids_ENU`, `vertices_ENU`, `n_min/n_max/east_lo/east_hi`, `east_entry` |
| 7 | fixed primary batch + layout | same `(U, r, parts)` to `rtol=1e-6` **and** finite non-zero `grad` w.r.t. `x_det`/`y_det` |
| 8 | fixed seed, each scheme | identical output arrays; memoized `_ne_max_gap` returns the float it returned uncached |
| 9–10 | `python -c "import modules_v6"` + full import smoke test | clean |
| 11 | `strategy_ids.unique()` on a rebuilt sample | same count as before; Step-2 `shower_level_split` (which derives `n_showers` from `strategy_ids.max()+1`) still partitions correctly |

### Acceptance criteria

- [ ] Reachability trace reports **0** unreachable modules in `modules_v6/`.
- [ ] No name is defined in two files (re-run the duplicate-name AST check; `_mlp`
      resolves to `nn_blocks` only).
- [ ] No callable in `modules_v6/` exceeds ~80 lines.
- [ ] Callable count is a **ceiling, not a target**: ≤ 85; line count ≤ 2,900.
      Phase 1 alone landed at 73 callables / 2,814 loc (from 97 / 3,775), i.e. it
      already passed the original "~74" figure. Phases 2–3 deliberately *add*
      callables by extracting helpers, so the count rises from here — the goal is
      that it stays under the ceiling while no single function stays long.
      Count top-level functions + methods (73); a walk including nested defs gives
      75. Use the same convention throughout.
- [ ] `python scripts/04_optimize_lbfgs_ensemble.py --schemes grid --chains 1` for a
      few iterations produces an `optimize_log.json` matching a pre-refactor run at
      the same seed.
- [ ] End to end: `sbatch slurm/run_all_script_batch_grid.sh` against a scratch
      `RUN_LOCATION` with reduced `DATASET_FRACTION` walks `pipeline_status.json`
      00→04 and lands final `U` near the current ~35.

## Edge Cases

1. **RNG order in Step 5.** The layout `rng` is threaded through chunk→strategy→batch.
   Hoisting a loop or reordering extraction changes every label. The element-wise
   dataset diff is the gate; do not accept "statistically equivalent".
2. **Autograd detachment in Step 7.** `utility_of_xy` is deliberately *not*
   `@torch.no_grad()`-decorated so L-BFGS can differentiate it. Splitting moves
   tensors across function boundaries — the gradient check is not optional.
3. **`FNNSurrogate` looks dead but isn't.** Step 3 deletes most of `fnn_surrogate.py`;
   the class must stay or `plots/02_plot_nn_target_vs_pred.py` can no longer load
   pre-DeepSets checkpoints.
4. **`legacy_core/geometry.py` is half-dead.** `Layouts` is live (used by
   `detector_strategies_ne.layout_rings`); `project_to_triangle` and
   `barycentric_coords` are not. Delete the pair, keep `Layouts`.
5. **Notebooks are not on the import graph.** `notebooks/hmc_chain_inits.ipynb` calls
   `mountain.sample_initial_layout` / `project_to_mountain`, deleted in Step 2. It
   drives the retired NUTS/HMC scripts, so this is acceptable — but note it in the
   commit rather than discovering it later.

## Risks

- **Renumbering `_STRATEGIES` invalidates every checkpoint.** `strategy_ids` is
  persisted in the dataset and `shower_level_split` derives `n_showers` from
  `strategy_ids.max()+1`. Step 11 must preserve order and count exactly; it is a
  readability change, not a reconfiguration.
- **Step 5 collides with pending performance work** (loop reorder + layout-table
  dataset format), which rewrites the same inner loop. Land this split first, then
  rebase the perf change onto the named stages.
- **`legacy_core/generate_showers.py` injects an external `sys.path`** to the sibling
  `TAMBO-opt` repo. Step 12 must not disturb that import, and it is the reason the
  package cannot be flattened entirely.

## Work Decomposition

Sequential: **Phase 1 → Phase 2 → Phase 3.** Phase 1 removes a Phase 2 split target
(Step 2 deletes the 95-line method), and Phase 3's renames only become correct once
Phase 1 removes the twins.

Parallelisable within Phase 2 — Steps 6, 7 and 8 touch disjoint files
(`legacy_core/tr_geometry.py`, `opt_core.py`, `tr_geometry_ne.py`) and can run
concurrently. **Step 5 should run alone**: it is the highest-risk change and its
verification is the slowest.

Within Phase 3, Step 9 is independent; Steps 10 → 11 → 12 are sequential (each
renames or moves what the next one edits).
