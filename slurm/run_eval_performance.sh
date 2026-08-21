#!/bin/bash
#SBATCH -p gpu_test 	
#SBATCH --mem=20g        			
#SBATCH --time=00:30:00 			
#SBATCH -c 8            			
#SBATCH --gres=gpu:1
#SBATCH -J run_eval_performance
#SBATCH -o slurm_logs/slurm-%j-%x.out
#SBATCH --chdir=/n/home05/zdimitrov/tambo/TambOpt

source slurm/env.sh

python -u plots/layouts/true_utility.py --grid-layout