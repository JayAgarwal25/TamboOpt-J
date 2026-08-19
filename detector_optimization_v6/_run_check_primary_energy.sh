#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH --mem=20g
#SBATCH --time=00:15:00
#SBATCH -c 4
#SBATCH -J check_e_range
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --open-mode=append

module load python
conda deactivate
conda deactivate
conda activate multiproc_env

python -u _check_primary_energy_range.py
