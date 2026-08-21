#!/bin/bash
#SBATCH -p gpu_test 	
#SBATCH --mem=20g        			
#SBATCH --time=00:30:00 			
#SBATCH -c 8            			
#SBATCH --gres=gpu:1
#SBATCH -J run_eval_performance
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

module load python

conda deactivate
conda deactivate

conda activate multiproc_env

# all .pyc under one tree instead of __pycache__/ dirs across the source
export PYTHONPYCACHEPREFIX=/n/home05/zdimitrov/tambo/TambOpt/.pycache

python -u plots/eval_true_utility.py --grid-layout