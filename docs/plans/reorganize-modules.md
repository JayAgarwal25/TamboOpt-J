# reorganize: `modules_v6/` → `modules/`, grouped by domain

## Context

The previous pass (`refactor-modules-v6.md`) made every callable reachable,
uniquely named and under 80 lines. What it did not fix is the shape of the
package: 20 modules sit flat in one directory, so nothing tells a reader which
of them belong together, and the `_v6` suffix names a generation that no longer
has a sibling.

Two measured problems remain:

1. **No grouping.** 20 flat modules, but the dependency graph is acyclic and
   falls into six obvious clusters — geometry, layouts, showers, surrogates,
   data, optimize.
2. **45% of the package is prose** — 1,017 docstring + 368 comment lines out of
   3,060. Of the docstring lines, 485 are narrative paragraphs and 17
   docstrings are longer than the code they document.

Outcome: `modules/` with six domain subpackages, no stuttering filenames, and
narrative compressed to headline-plus-pointer while every hazard warning stays
at its call site.

## Decisions (confirmed with the user)

| Question | Choice |
|---|---|
| Grouping | Six domain folders |
| Filenames | Renamed to drop stutter (`geometry/tr_geometry.py` → `geometry/mountain.py`) |
| Prose | Level 2 — compress narratives, keep hazards in-file, move forensics to `docs/THEORY.md` |
| Imports | Re-export public names from each folder's `__init__.py` |

## Target layout

```
modules/
  __init__.py
  constants.py                # 30 importers; stays at root
  geometry/
    mountain.py               # <- legacy_core/tr_geometry.py   MountainData, load_tr_mountain
    surface.py                # <- surface_map.py               SurfaceUpMap
    placement.py              # <- tr_geometry_ne.py            project_to_mountain_ne, sample_initial_layout_ne
    primitives.py             # <- geometry.py                  Layouts
  layouts/
    strategies.py             # <- detector_strategies.py       layout_grid, _rings, _latin_hypercube, ...
    learnable.py              # <- legacy_core/layout_optimization.py   LearnableXY
  showers/
    kernel.py                 # <- tr_plane_kernel.py           GetCounts_planeaware
    tau.py                    # <- tau_showers.py               load_tau_primaries
    generate.py               # <- legacy_core/generate_showers.py      GenerateShowers
  surrogates/
    blocks.py                 # <- nn_blocks.py                 _mlp
    fnn.py                    # <- fnn_surrogate.py             encode_primary, FNNSurrogate, sidecars
    deepsets.py               # <- deepsets_surrogate.py        DeepSetsSurrogate
    dual.py                   # <- dual_surrogate.py            DualSpeciesSurrogate
    recon.py                  # <- reconstruction.py            DeepSetsRecon
  data/
    dataset_builder.py        # <- dataset_builder.py           build_training_pairs
  optimize/
    objective.py              # <- opt_core.py                  utility_of_xy, activation_of_xy
    utility.py                # <- utility_functions.py         U_E, U_angle, U_PR
```

`legacy_core/` disappears. `generate.py` keeps its docstring note about the
external `TAMBO-opt` dependency and its `sys.path` injection — that is a
property of the module, not of a directory name.

## Steps

- [x] **1. Move.** `git mv modules_v6 modules`, create the six subpackages,
  `git mv` each file to its new name. Use `git mv` throughout so history follows.
- [x] **2. `__init__.py` re-exports.** Each subpackage re-exports its public
  names (the `public:` sets already inventoried). Private helpers stay private.
- [x] **3. Intra-package imports.** Rewrite relative imports for the new depth.
- [x] **4. External call sites.** 47 files under `scripts/`, `plots/`,
  `notebooks/`, `slurm/`, `docs/`, plus `_pathfix.py` and `README.md`. Rewrite
  by explicit old-path → new-path mapping, not a blind `modules_v6` → `modules`
  substitution, so each import lands on the right subpackage.
- [x] **5. Verify the move** before touching any prose — token guard, import
  smoke over all 28 entry points, and a real Step-04 optimizer run.
- [x] **6. Prose (Level 2), as a separate commit.** Per module: compress
  narrative paragraphs, keep every hazard as a one-line headline plus a
  `docs/THEORY.md` pointer, delete provenance archaeology, and drop `Args:`
  lines that only restate an annotated signature — keeping any that carry
  shapes, units or `MODIFIED IN PLACE`.

## Do not touch

- **`constants.py` comments.** 53% comments and nearly all load-bearing
  calibration hazards (`LAYER_EAST_DX` must be 500, `PRIMARY_DIM` invalidates
  checkpoints). Leave them.
- **`_STRATEGIES` order and count.** Position is the persisted `strategy_ids`
  value; Step-2 `shower_level_split` derives `n_showers` from `max()+1`.
- **RNG draw order** in `build_training_pairs`.
- **`notebooks/utility.ipynb`** — the user has uncommitted work there.

## Outcome

All six steps landed, in three commits: `97e2cc6` (move), `0d243fb` (prose),
`0a58b57` (bytecode). Verified end to end.

| Criterion | Result |
|---|---|
| Unresolved import names (relative included) | **0** |
| Entry points importing | **28/28** |
| Unreachable modules / duplicate names | **0 / 0** |
| Executable code changed by the move | none outside import statements, all 18 modules |
| Executable code changed by the prose pass | one line — a genuine bug fix (see below) |
| Step-04 optimizer, `-p gpu_test`, short budget | exit 0, **U = +33.577** (was +33.352 pre-move) |
| Stray `__pycache__` dirs after a full GPU run | **0** |

**Prose: less than projected.** modules/ went 3,060 -> 2,975 lines and 45% ->
41% prose; 149 docstring lines removed, against the ~400 I estimated. The
estimate was wrong: I counted 485 lines as "narrative" and treated all of it as
removable, but at Level 2 much of that narrative *is* the hazard content Level 2
promises to keep. Cutting to the projected figure would have meant Level 3.
Docstrings longer than the code they document: 17 -> 12.

**One bug found and fixed.** `deepsets.py` had a function-local
`from .fnn_surrogate import FNNSurrogate` that the move missed, because the
import checker skipped relative imports. It raised ModuleNotFoundError at
Step-4 model load — caught by the optimizer run, not by the import smoke test.
The checker now resolves relative imports too.

**Stale docs corrected while compressing** — `geometry/mountain.py` described
the retired colca mesh throughout (group, 2161 centroids, east_entry -212,
dx 307), and `MountainData` advertised a `centroids_NUE` property that is
commented out, which is what broke `plots/plot_init_layouts.py` earlier.

## Verification

| What | How | Bar |
|---|---|---|
| No code changed by the move | `tools/codeonly.py` per file, before vs after | differences confined to import lines |
| No code changed by the prose pass | same | **zero** differences |
| Imports resolve | real import (not `ast.parse`) of all 28 entry points | 28/28 |
| Package is sound | reachability trace + duplicate-name check | 0 unreachable, 0 duplicates |
| Runtime behaviour | `04_optimize_lbfgs_ensemble.py --schemes grid --chains 1` on `-p gpu_test`, short budget | exit 0, `U` in the low 30s |

SLURM: `-p test` for CPU, `-p gpu_test` for GPU. Never a login node. `/tmp` is
node-local — job scripts and their `-o` logs must live on shared storage.
Interpreter: `/n/home05/zdimitrov/.conda/envs/multiproc_env/bin/python`.
