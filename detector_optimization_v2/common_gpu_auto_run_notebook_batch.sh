#!/bin/bash
#SBATCH -p gpu,gpu_h200
#SBATCH --mem=150g
#SBATCH --time=10:00:00
#SBATCH -c 2        
#SBATCH --gres=gpu:1

module load python 

conda activate multiproc_env

python auto_run_notebook.py SWGOLO7_optimization.ipynb
