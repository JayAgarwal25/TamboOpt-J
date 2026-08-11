#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J check_offmesh_penalty
#SBATCH -o slurm_logs/slurm-%j-%x.out

module load python
conda deactivate
conda deactivate
conda activate multiproc_env
export PYTHONUNBUFFERED=1

python -u check_offmesh_penalty.py
