#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=60g
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J run_plots
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt


module load python

conda deactivate
conda deactivate

conda activate multiproc_env

# python -u plots/02_plot_nn_target_vs_pred.py --dual

python -u plots/05_paper_figures.py

# mp4 by default: GIF caps at 50 fps. Needs imageio-ffmpeg.
# python -u plots/04_plot_trajectory_gif.py

# python -u plots/04_plot_trajectory_gif.py --monotonic --seconds 20 \
#     -o layout_trajectory_monotonic.gif