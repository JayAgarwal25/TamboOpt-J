#!/bin/bash
#SBATCH -p gpu_requeue 	
#SBATCH --mem=20g        			
#SBATCH --time=00:30:00 			
#SBATCH -c 32            			
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -J run_eval_performance
#SBATCH -o slurm_logs/slurm-%j-%x.out

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

python -u plots/eval_true_utility.py