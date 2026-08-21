# `plots/` — figures and evaluation, grouped by subject

```
lib/       imported helpers, never run directly
showers/   shower morphology
training/  surrogate / recon training diagnostics
layouts/   layout figures and the kernel-truth evaluators
```

Everything outside `lib/` is an entry point run by path
(`python plots/<group>/<name>.py`), never imported by name.

## `lib/` — imported, never run

| file | imported by |
|---|---|
| `opt_plotting.py` | all three `scripts/04_optimize_*`, and `layouts/replot_optimize_curves.py` |
| `paper_style.py` | `layouts/05_paper_figures.py` |
| `geometry_plots.py` | `layouts/05_paper_figures.py`, the malata/trigger notebooks |

## `training/`

| file | driven by |
|---|---|
| `02_nn_target_vs_pred.py` | `slurm/run_plots.sh`; also called at the end of Steps 2 and 3 |
| `aleatoric_floor.py` | `slurm/run_floor_calculation.sh` |

## `layouts/`

| file | driven by |
|---|---|
| `05_paper_figures.py` | `slurm/run_paper_figures.sh`, `run_plots.sh` |
| `04_trajectory_gif.py` | `slurm/run_plots.sh`, `run_traj_activation.sh` |
| `true_utility.py` | `slurm/run_eval_performance.sh` |
| `true_activation.py` | `slurm/run_eval_true_activation.sh` |
| `activation_counts.py` | `slurm/run_eval_activation.sh` |
| `utility_vs_energy.py` | `slurm/run_eval_utility_vs_energy*.sh` |
| `init_stage4.py` | hand-run — Stage-4 initial layouts, before and after chain perturbation |
| `init_cross_section.py` | hand-run — init schemes in the (North, Up) cross-section |
| `replot_optimize_curves.py` | hand-run — re-render a finished run's curves without re-optimizing |

`true_activation.py` and `activation_counts.py` import `true_utility` as a
sibling, so those three have to stay in the same folder.

## `showers/` — all hand-run

Not referenced by any SLURM job. Kept deliberately; each does something the
automated set does not.

| file | what it is for |
|---|---|
| `angle_grid_dual.py` | shower morphology across a grid of arrival angles (2D + 3D) |
| `angle_grid_single.py` | same grid, single-species, generating showers live |
| `realizations.py` | N realizations of **one** primary — shows the aleatoric spread |
| `cached_showers.py` | plot showers from a **cached** corpus checkpoint |
| `hamza_ml_vs_sim.py` | ML-generated vs simulated electron showers, matched pairs |

`scripts/replot_de_ensemble_up.py` is the same kind of tool for DE runs.

## The bootstrap stanza

Every script here opens with the same block, and the duplication is deliberate —
it has to run before any project import can work:

```python
_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
```

`_V6` is found by walking up to a marker rather than by counting parent
directories. Counting is what silently broke the old `plots/single_species/`
scripts: written for files one level under the repo root, they resolved the
"repo root" to `plots/` and could not import `modules` at all. The marker walk
survives a file being moved between groups.

## Output

Figures are **not** written into this directory and are not tracked in git.
`05_paper_figures.py` defaults to `modules.constants.PAPER_FIGURES_DIR`; the
hand-run tools take an explicit `--out`. Both live under
`constants.FIGURES_DIR` on holylfs05.
