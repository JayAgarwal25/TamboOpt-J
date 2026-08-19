#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=60g
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J run_paper_figures
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt


module load python

conda deactivate
conda deactivate

conda activate multiproc_env

python -u plots/05_paper_figures.py
