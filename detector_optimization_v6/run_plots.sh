#!/bin/bash
#SBATCH -p test
#SBATCH --mem=60g
#SBATCH --time=01:00:00
#SBATCH -c 8
#SBATCH -J run_plots
#SBATCH -o slurm_logs/slurm-%j-%x.out

# The pipeline stages refuse to guess a run world (modules_v6/run_world.py).
# Export it before submitting:  TAMBO_RUN_WORLD=/path/to/run sbatch <this>
: "${TAMBO_RUN_WORLD:?set TAMBO_RUN_WORLD to the run world root before submitting}"
export TAMBO_RUN_WORLD


module load python

conda deactivate
conda deactivate

conda activate multiproc_env

# python -u plots/02_plot_nn_target_vs_pred.py --dual

# mp4 by default: GIF caps at 50 fps. Needs imageio-ffmpeg.
python -u plots/04_plot_trajectory_gif.py

# python -u plots/04_plot_trajectory_gif.py --monotonic --seconds 20 \
#     -o layout_trajectory_monotonic.gif