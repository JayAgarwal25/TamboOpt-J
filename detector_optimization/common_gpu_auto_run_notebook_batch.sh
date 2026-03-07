#!/bin/bash
#SBATCH -p gpu,gpu_h200
#SBATCH --mem=150g
#SBATCH --time=2-00:00:00 			# time in HH:MM:SS
#SBATCH -c 2            			# number of cores
#SBATCH --gres=gpu:1        # requested GPU type

module load python 

conda activate multiproc_env

python auto_run_notebook.py SWGOLO7_optimization.ipynb
