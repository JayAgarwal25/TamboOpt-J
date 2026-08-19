#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=60g
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J recon_replot
#SBATCH -o slurm_logs/slurm-%j-%x.out

module load python
conda deactivate
conda deactivate
conda activate multiproc_env

python -u plots/02_plot_nn_target_vs_pred.py --dual
