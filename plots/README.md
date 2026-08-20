# `plots/` — what runs automatically, and what you run by hand

Two kinds of file live here. The distinction is not visible from the filenames,
so it is written down: anything in the second table is reachable only by typing
its name, and nothing will notice if it breaks.

## Driven by SLURM (part of the pipeline)

| file | driven by |
|---|---|
| `05_paper_figures.py` | `slurm/run_paper_figures.sh`, `run_plots.sh` |
| `02_plot_nn_target_vs_pred.py` | `slurm/run_plots.sh`; also called at the end of Steps 2 and 3 |
| `04_plot_trajectory_gif.py` | `slurm/run_plots.sh`, `run_traj_activation.sh` |
| `eval_true_utility.py` | `slurm/run_eval_performance.sh` |
| `eval_true_activation.py` | `slurm/run_eval_true_activation.sh` |
| `eval_activation_counts.py` | `slurm/run_eval_activation.sh` |
| `eval_utility_vs_energy.py` | `slurm/run_eval_utility_vs_energy*.sh` |
| `compute_aleatoric_floor.py` | `slurm/run_floor_calculation.sh` |

## Imported as libraries

| file | imported by |
|---|---|
| `opt_plotting.py` | all three `scripts/04_optimize_*` optimizers, `modules_v6/opt_core.py` |
| `paper_style.py` | `05_paper_figures.py` |
| `geometry_plots.py` | `05_paper_figures.py`, the malata/trigger notebooks |

## Hand-run only

Not referenced by any SLURM job or importer. Kept deliberately — each does
something the automated set does not.

| file | what it is for |
|---|---|
| `plot_angle_grid_dual_species.py` | shower morphology across a grid of arrival angles (2D + 3D) |
| `plot_cached_showers.py` | plot showers from a **cached** corpus checkpoint |
| `single_species/plot_angle_grid.py` | same grid, single-species, generating showers live |
| `single_species/plot_shower_realizations.py` | N realizations of **one** primary — shows the aleatoric spread |
| `plot_hamza_ml_vs_sim.py` | ML-generated vs simulated electron showers, matched pairs |
| `04_plot_init_layouts.py` | Stage-4 initial layouts, before and after chain perturbation |
| `plot_init_layouts.py` | init schemes drawn in the (North, Up) cross-section |
| `replot_optimize_curves.py` | re-render a finished run's curves without re-optimizing |

`scripts/replot_de_ensemble_up.py` is the same kind of tool for DE runs.

## Output

Figures are **not** written into this directory and are not tracked in git.
`05_paper_figures.py` defaults to `modules_v6.constants.PAPER_FIGURES_DIR`; the
hand-run tools take an explicit `--out`. Both live under
`constants.FIGURES_ARCHIVE` on holylfs05.
