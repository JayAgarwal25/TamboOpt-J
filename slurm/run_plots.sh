#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=60g
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J run_plots
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt


source slurm/env.sh

# python -u plots/training/02_nn_target_vs_pred.py --dual

python -u plots/layouts/05_paper_figures.py

# mp4 by default: GIF caps at 50 fps. Needs imageio-ffmpeg.
# python -u plots/layouts/04_trajectory_gif.py

# python -u plots/layouts/04_trajectory_gif.py --monotonic --seconds 20 \
#     -o layout_trajectory_monotonic.gif