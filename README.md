# TambOpt — TAMBO Detector Optimization & Simulation Suite

TambOpt is a differentiable end-to-end pipeline for optimizing cosmic ray detector layouts for the TAMBO Observatory. This repo is the v6 pipeline (the only maintained generation) — it combines flow-matching shower generation, a differentiable plane-aware detector-response kernel, DeepSets forward-surrogate and reconstruction networks, and gradient/global optimization of detector `(x, y)` positions on the Malata mountain mesh.

Earlier generations (`detector_optimization/` through `_v5`, plus a handful of unrelated exploratory folders) have been retired to the `legacy-full-repo` git branch — see [`VERSIONS.md`](VERSIONS.md) for the full chronological history of what each generation tried and why v6 looks the way it does.

## Layout

```
modules/          # importable library, six domain subpackages:
                  #   geometry/ layouts/ showers/ surrogates/ data/ optimize/
                  #   each re-exports its public names, so:
                  #     from modules.geometry import load_tr_mountain, SurfaceUpMap
scripts/          # 00-04 pipeline stages: generate data → build dataset → train surrogate → train recon → optimize layout
slurm/            # SLURM sbatch wrappers for the scripts/ pipeline and plots/ evaluation scripts
plots/            # plotting + evaluation (surrogate-vs-ground-truth checks, paper figures)
decay_locations/  # CORSIKA observation-plane source + Julia tau-injection pipeline
notebooks/        # EDA / diagnostic notebooks
docs/             # THEORY.md (architecture reference) and dev diary
data/             # mountain mesh geometry (malata.h5)
.pycache/         # all bytecode, gitignored (see below)
```

Bytecode is kept out of the source tree: every SLURM wrapper exports
`PYTHONPYCACHEPREFIX=<repo>/.pycache`, and `_pathfix.py` sets the same as a
fallback for notebooks and interactive sessions, so no `__pycache__/` dirs
appear beside the sources. For an interactive shell that does not go through
either, export it yourself:

```bash
export PYTHONPYCACHEPREFIX=/n/home05/zdimitrov/tambo/TambOpt/.pycache
```

Note the prefix is global to the interpreter, so third-party packages cache
there too — `.pycache/` runs to a few tens of MB. It is gitignored and safe to
delete at any time.

See `docs/THEORY.md` for the full pipeline architecture and the reasoning behind the DeepSets surrogate/reconstruction design.

## Running the pipeline

```
sbatch slurm/run_all_script_batch.sh
```

drives the full `00` → `04` pipeline end-to-end, with `pipeline_status.json` checkpointing so already-completed steps are skipped on re-submission. See `slurm/` for the individual evaluation/plotting job wrappers.
